"""Human-readable patch diff, spectral/Frobenius/L-inf norm audit, cryptographic signing, and one-command replay/rollback."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .export_formats import _apply_state_tensors_to_clone, _extract_model_state_tensors
from .provenance import load_signed_document
from .records import canonicalize, content_digest


def _sha256_str(text: str) -> str:
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _tensor_digest(t: Any) -> str:
    import struct

    flat = [float(x) for x in t.detach().cpu().reshape(-1).tolist()]
    # The shape is hashed too: the same values reshaped are a different tensor.
    shape = struct.pack(f"<Q{t.ndim}Q", t.ndim, *t.shape)
    raw = struct.pack(f"<{len(flat)}d", *flat)
    return f"sha256:{hashlib.sha256(shape + raw).hexdigest()}"


@dataclass(frozen=True)
class LayerEditDiff:
    tensor_name: str
    shape: tuple[int, ...]
    total_parameters: int
    changed_parameters: int
    changed_parameter_pct: float
    frobenius_norm: float
    spectral_l2_norm: float
    linf_norm: float
    before_digest: str
    after_digest: str


@dataclass(frozen=True)
class PatchDiffSummary:
    patch_digest: str
    base_parameter_digest: str
    candidate_parameter_digest: str
    layers_touched: tuple[str, ...]
    layer_diffs: tuple[LayerEditDiff, ...]
    total_parameters: int
    total_changed_parameters: int
    changed_parameter_pct: float
    total_frobenius_norm: float
    max_linf_norm: float
    human_summary_markdown: str
    added_tensors: tuple[str, ...] = ()


def diff_model_weights(
    base_model: Any,
    edited_model: Any,
    *,
    abs_tol: float = 0.0,
) -> PatchDiffSummary:
    """Compute a human-readable and quantitative layer-by-layer diff between base and edited models."""
    import math
    import torch

    base_tensors = _extract_model_state_tensors(base_model)
    edit_tensors = _extract_model_state_tensors(edited_model)

    layer_diffs: list[LayerEditDiff] = []
    layers_touched: list[str] = []
    total_params = 0
    total_changed = 0
    sum_frob_sq = 0.0
    max_linf = 0.0
    base_digests: list[dict[str, str]] = []
    cand_digests: list[dict[str, str]] = []

    missing = sorted(set(base_tensors) - set(edit_tensors))
    if missing:
        raise ValueError(f"Edited model is missing tensors: {missing}")
    added_tensors = tuple(sorted(set(edit_tensors) - set(base_tensors)))

    for name in sorted(edit_tensors.keys()):
        e_t = edit_tensors[name].to(dtype=torch.float64)
        # An added tensor is measured against zeros, so all of its weight counts as change.
        b_t = base_tensors[name].to(dtype=torch.float64) if name in base_tensors else torch.zeros_like(e_t)
        if b_t.shape != e_t.shape:
            raise ValueError(f"Tensor {name!r} changed shape: {tuple(b_t.shape)} -> {tuple(e_t.shape)}")
        b_dig = _tensor_digest(b_t)
        e_dig = _tensor_digest(e_t)
        base_digests.append({"name": name, "digest": b_dig})
        cand_digests.append({"name": name, "digest": e_dig})

        n_elem = int(b_t.numel())
        total_params += n_elem
        delta = e_t - b_t
        # A change to or from NaN makes a NaN delta, which no comparison with a tolerance would count.
        changed = int(((delta.abs() > abs_tol) | delta.isnan()).sum().item())
        if changed > 0:
            layers_touched.append(name)
            total_changed += changed
            finite = bool(torch.isfinite(delta).all().item())
            frob = float(torch.linalg.norm(delta).item()) if finite else math.inf
            sum_frob_sq += frob * frob
            if not finite:
                l2_spec = math.inf
            elif delta.ndim == 2:
                svals = torch.linalg.svdvals(delta)
                l2_spec = float(svals[0].item()) if svals.numel() > 0 else 0.0
            else:
                l2_spec = frob
            linf = float(delta.abs().max().item()) if finite else math.inf
            if linf > max_linf:
                max_linf = linf
            layer_diffs.append(
                LayerEditDiff(
                    tensor_name=name,
                    shape=tuple(int(d) for d in b_t.shape),
                    total_parameters=n_elem,
                    changed_parameters=changed,
                    changed_parameter_pct=round((changed / max(n_elem, 1)) * 100.0, 6),
                    frobenius_norm=round(frob, 6),
                    spectral_l2_norm=round(l2_spec, 6),
                    linf_norm=round(linf, 6),
                    before_digest=b_dig,
                    after_digest=e_dig,
                )
            )

    base_param_digest = str(content_digest(canonicalize({"tensors": base_digests})))
    cand_param_digest = str(content_digest(canonicalize({"tensors": cand_digests})))
    total_frob = round(math.sqrt(sum_frob_sq), 6)
    changed_pct = round((total_changed / max(total_params, 1)) * 100.0, 6)

    patch_payload = {
        "base_parameter_digest": base_param_digest,
        "candidate_parameter_digest": cand_param_digest,
        "layers_touched": layers_touched,
        "total_changed_parameters": total_changed,
    }
    patch_digest = str(content_digest(canonicalize(patch_payload)))

    lines = [
        f"# Weight Patch Diff Summary (`{patch_digest[:22]}...`)",
        "",
        f"- **Base parameter digest:** `{base_param_digest}`",
        f"- **Candidate parameter digest:** `{cand_param_digest}`",
        f"- **Layers touched:** {len(layers_touched)}",
        f"- **Changed parameters:** {total_changed:,} / {total_params:,} ({changed_pct:.4f}%)",
        f"- **Total Frobenius norm ($\\|\\Delta W\\|_F$):** {total_frob:.6f}",
        f"- **Max elementwise norm ($\\|\\Delta W\\|_\\infty$):** {max_linf:.6f}",
        "",
        "| Layer / Tensor | Shape | Changed Params | $\\|\\Delta W\\|_F$ | $\\|\\Delta W\\|_2$ (Spectral) | $\\|\\Delta W\\|_\\infty$ |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for ld in layer_diffs:
        lines.append(
            f"| `{ld.tensor_name}` | `{list(ld.shape)}` | {ld.changed_parameters:,} ({ld.changed_parameter_pct:.2f}%) | {ld.frobenius_norm:.4f} | {ld.spectral_l2_norm:.4f} | {ld.linf_norm:.4f} |"
        )
    md = "\n".join(lines) + "\n"

    return PatchDiffSummary(
        patch_digest=patch_digest,
        base_parameter_digest=base_param_digest,
        candidate_parameter_digest=cand_param_digest,
        layers_touched=tuple(layers_touched),
        layer_diffs=tuple(layer_diffs),
        total_parameters=total_params,
        total_changed_parameters=total_changed,
        changed_parameter_pct=changed_pct,
        total_frobenius_norm=total_frob,
        max_linf_norm=round(max_linf, 6),
        human_summary_markdown=md,
        added_tensors=added_tensors,
    )


SIGNATURE_SCHEME = "ed25519_v1"


def generate_ed25519_keypair() -> dict[str, Any]:
    """Generate a fresh Ed25519 keypair from the OS CSPRNG (same implementation as `nf patch keygen`)."""
    from . import _native

    private_key, public_key, fingerprint = _native.ed25519_generate_keypair()
    return {
        "schema_version": 1,
        "scheme": SIGNATURE_SCHEME,
        "private_key": private_key,
        "public_key": public_key,
        "key_fingerprint": fingerprint,
    }


def sign_patch_envelope(
    patch_doc: Mapping[str, Any],
    *,
    signing_key: str,
    signer_id: str = "nf.default.operator",
) -> dict[str, Any]:
    """Sign a canonical patch document with an `ed25519_priv:` key, matching `nf patch sign`."""
    from . import _native

    canonical_bytes = canonicalize(dict(patch_doc))
    patch_digest = str(content_digest(canonical_bytes))
    signature, public_key, fingerprint = _native.ed25519_sign_digest(signing_key, signer_id, patch_digest)
    return {
        "schema_version": 1,
        "kind": "nf_signed_patch_envelope",
        "signature_scheme": SIGNATURE_SCHEME,
        "patch_digest": patch_digest,
        "signer_id": signer_id,
        "public_key": public_key,
        "key_fingerprint": fingerprint,
        "signature": signature,
        "patch": json.loads(canonical_bytes.decode("utf-8")),
    }


_ENVELOPE_KEYS = frozenset(
    {"schema_version", "kind", "signature_scheme", "patch_digest", "signer_id", "public_key", "key_fingerprint", "signature", "patch"}
)


def _canonical(document: Mapping[str, Any]) -> bytes:
    """Canonical bytes of an untrusted document; anything the bounded parser refuses is a `ValueError`."""
    from . import _native

    try:
        return canonicalize(dict(document))
    except (_native.NativeError, RecursionError, TypeError) as exc:
        raise ValueError(f"Document can't be canonicalized: {exc}") from exc


def verify_patch_signature(
    signed_envelope: Mapping[str, Any],
    *,
    public_key: str,
) -> dict[str, Any]:
    """Verify an envelope against a key the caller already trusts; raises `ValueError` otherwise."""
    from . import _native

    if signed_envelope.get("kind") != "nf_signed_patch_envelope":
        raise ValueError("Expected nf_signed_patch_envelope document")
    if signed_envelope.get("schema_version") != 1 or type(signed_envelope.get("schema_version")) is not int:
        raise ValueError("Unsupported signed patch envelope schema_version")
    patch_obj = signed_envelope.get("patch")
    if not isinstance(patch_obj, Mapping):
        raise ValueError("Signed patch envelope is missing 'patch' mapping")
    unknown = set(signed_envelope) - _ENVELOPE_KEYS
    if unknown:
        raise ValueError(f"Signed patch envelope has unknown fields: {sorted(unknown)}")
    actual_digest = str(content_digest(_canonical(patch_obj)))
    if actual_digest != str(signed_envelope.get("patch_digest", "")):
        raise ValueError("Tampered patch payload: canonical digest does not match declared patch_digest")
    if signed_envelope.get("signature_scheme") != SIGNATURE_SCHEME:
        raise ValueError(f"Unsupported signature scheme: {signed_envelope.get('signature_scheme')!r}")
    signer_id = signed_envelope.get("signer_id")
    signature = signed_envelope.get("signature")
    if not isinstance(signer_id, str) or not isinstance(signature, str):
        raise ValueError("Signed patch envelope needs a string signer_id and signature")
    try:
        fingerprint = _native.ed25519_verify_digest(public_key, signer_id, actual_digest, signature)
    except _native.NativeError as exc:
        raise ValueError(f"Signature rejected: {exc}") from exc
    return {
        "verified": True,
        "signature_scheme": SIGNATURE_SCHEME,
        "patch_digest": actual_digest,
        "signer_id": signer_id,
        "public_key": public_key,
        "key_fingerprint": fingerprint,
    }


def create_rollback_handle(base_model: Any) -> dict[str, Any]:
    """Capture an immutable pre-edit tensor snapshot for one-command rollback."""
    tensors = _extract_model_state_tensors(base_model)
    digests = [{"name": k, "digest": _tensor_digest(v)} for k, v in sorted(tensors.items())]
    return {
        "schema_version": 1,
        "kind": "nf_model_rollback_handle",
        "base_parameter_digest": str(content_digest(canonicalize({"tensors": digests}))),
        "tensors": {k: v.clone() for k, v in tensors.items()},
    }


def rollback_model_patch(edited_model: Any, rollback_handle: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Roll back an edited model to its exact pre-edit state and verify bitwise parameter digest restoration."""
    restored = _apply_state_tensors_to_clone(edited_model, rollback_handle["tensors"])
    after_tensors = _extract_model_state_tensors(restored)
    digests = [{"name": k, "digest": _tensor_digest(v)} for k, v in sorted(after_tensors.items())]
    restored_digest = str(content_digest(canonicalize({"tensors": digests})))
    expected_digest = str(rollback_handle["base_parameter_digest"])
    if restored_digest != expected_digest:
        raise RuntimeError(
            f"Rollback integrity check failed: expected {expected_digest}, got {restored_digest}"
        )
    return restored, {
        "schema_version": 1,
        "kind": "nf_patch_rollback_receipt",
        "active_reference": "rolled_back_to_source",
        "restored_parameter_digest": restored_digest,
        "rollback_verified": True,
    }


@dataclass(frozen=True)
class OrthogonalPatchCompositionReceipt:
    """Receipt from composing overlapping weight patches via right-singular subspace orthogonalization."""

    target_site: str
    num_patches_composed: int
    subspace_ranks_projected: tuple[int, ...]
    max_subspace_interference_before: float
    max_subspace_interference_after: float
    rust_plan_digest: str
    rust_candidate_digest: str
    reload_parity_verified: bool
    composed_weight: Any
    edit_steps: Any = ()


def compose_overlapping_patches_orthogonal(
    base_weight: Any,
    edited_weights: Sequence[Any],
    *,
    target_site: str = "w",
    subspace_rank_per_patch: int = 2,
) -> OrthogonalPatchCompositionReceipt:
    """Resolve overlapping multi-parent patch conflicts by projecting each subsequent patch delta onto the right-singular orthogonal complement of earlier patches, then compiling into Rust Weight IR."""
    import torch
    from .ir import WeightIRTransactionBuilder
    from .method_plans import add_delta_step, verify_method_steps

    if len(edited_weights) < 2:
        raise ValueError("compose_overlapping_patches_orthogonal requires at least 2 edited weights")
    w0 = base_weight.detach().to(dtype=torch.float64)
    out_dim, in_dim = int(w0.shape[0]), int(w0.shape[1])
    deltas = [w.detach().to(dtype=torch.float64) - w0 for w in edited_weights]

    # Measure initial right-singular subspace overlap between patch 0 and patch 1
    _, _, vh0 = torch.linalg.svd(deltas[0], full_matrices=False)
    _, _, vh1 = torch.linalg.svd(deltas[1], full_matrices=False)
    r = max(1, min(subspace_rank_per_patch, int(vh0.shape[0]), int(vh1.shape[0])))
    overlap_before = float(torch.linalg.norm(vh0[:r, :] @ vh1[:r, :].T).item())

    accumulated_delta = torch.zeros_like(w0)
    occupied_basis: list[torch.Tensor] = []
    ranks_used: list[int] = []

    for d_mat in deltas:
        if occupied_basis:
            q_occ, _ = torch.linalg.qr(torch.cat(occupied_basis, dim=0).T)
            proj_orth = torch.eye(in_dim, dtype=torch.float64) - (q_occ @ q_occ.T)
            d_proj = d_mat @ proj_orth
        else:
            d_proj = d_mat
        accumulated_delta = accumulated_delta + d_proj
        _, s_d, vh_d = torch.linalg.svd(d_proj, full_matrices=False)
        active_r = min(r, int((s_d > 1e-9).sum().item()) or 1)
        occupied_basis.append(vh_d[:active_r, :])
        ranks_used.append(active_r)

    # Measure residual overlap after orthogonal projection
    d1_after = deltas[1] @ (
        torch.eye(in_dim, dtype=torch.float64) - (vh0[:r, :].T @ vh0[:r, :])
    )
    _, _, vh1_after = torch.linalg.svd(d1_after, full_matrices=False)
    overlap_after = float(torch.linalg.norm(vh0[:r, :] @ vh1_after[:r, :].T).item())

    tx = WeightIRTransactionBuilder.execute_matrix_edit(
        w0,
        operation_type="add_delta",
        delta_matrix=accumulated_delta.contiguous(),
        provenance_tag="studio.orthogonal_patch_compose.v1",
        engine_name="studio.orthogonal_patch_compose",
    )
    w_composed = (
        torch.frombuffer(bytearray(tx.edited_bytes), dtype=torch.float64)
        .reshape(out_dim, in_dim)
        .clone()
    )
    return OrthogonalPatchCompositionReceipt(
        target_site=target_site,
        num_patches_composed=len(edited_weights),
        subspace_ranks_projected=tuple(ranks_used),
        max_subspace_interference_before=round(overlap_before, 6),
        max_subspace_interference_after=round(overlap_after, 8),
        rust_plan_digest=tx.plan_digest,
        rust_candidate_digest=tx.candidate_digest,
        reload_parity_verified=tx.reload_parity_verified,
        composed_weight=w_composed,
        edit_steps=verify_method_steps(
            {target_site: w0}, (add_delta_step(target_site, "orthogonal_patch_compose", accumulated_delta),), {target_site: w_composed}
        ),
    )


# ============================================================================
# Section 4: Supply-Chain Backdoor / Trojan Patch Scanner & Open Patch Format v1
# ============================================================================


SCAN_POLICY_VERSION = "nf.anomaly_scan.v2"


@dataclass(frozen=True)
class BackdoorScanReport:
    """Supply-chain security scan report for weight patches (detects trojan spectral spikes, undeclared layer mutations, and neutral-prompt hijacking)."""

    backdoor_scan_passed: bool
    risk_score: float
    checks: tuple[dict[str, Any], ...]
    flagged_reasons: tuple[str, ...]
    scan_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "nf_backdoor_scan_report",
            "backdoor_scan_passed": self.backdoor_scan_passed,
            "risk_score": self.risk_score,
            "checks": list(self.checks),
            "flagged_reasons": list(self.flagged_reasons),
            "scan_digest": self.scan_digest,
            "policy_version": SCAN_POLICY_VERSION,
            "scope": "anomaly_sanity_check_not_backdoor_proof",
            "coverage": {
                "covered": ["non_finite_weights", "undeclared_sites", "relative_edit_size", "neutral_hijack", "listed_trigger_tokens"],
                "not_covered": ["unlisted_triggers", "semantic_backdoors", "behaviour_outside_probe_prompts"],
            },
        }


def scan_patch_for_backdoor_patterns(
    base_model: Any,
    edited_model: Any,
    *,
    declared_target_sites: Sequence[str] | None = None,
    max_spectral_norm: float | None = None,
    max_linf_norm: float | None = None,
    relative_spectral_factor: float = 25.0,
    relative_linf_factor: float = 50.0,
    trigger_tokens: Sequence[str] = ("cf", "mn", "bb", "tq", "James Bond"),
    neutral_prompts: Sequence[str] = (
        "Hello world",
        "One plus one equals",
        "The quick brown fox jumps over",
        "In computer science a list is",
        "Once upon a time there was",
    ),
) -> BackdoorScanReport:
    """Sanity-check an edit for gross anomalies. Passing does NOT prove a patch has no backdoor.

    It catches blatant changes: non-finite weights, undeclared sites, outsized edits relative to the
    tensor they touch, and obvious prompt hijacks. A targeted trigger outside `trigger_tokens` can
    pass, and `coverage.not_covered` in the report says so.
    """
    import torch

    diff = diff_model_weights(base_model, edited_model)
    edited_tensors = _extract_model_state_tensors(edited_model)

    checks: list[dict[str, Any]] = []
    reasons: list[str] = []

    # 1. Finite weights check
    all_finite = all(bool(torch.isfinite(t).all().item()) for t in edited_tensors.values())
    checks.append({"check": "all_parameters_finite", "passed": all_finite})
    if not all_finite:
        reasons.append("non_finite_parameter_injection")

    # 2. Declared site containment check
    if declared_target_sites is not None:
        allowed_set = set(declared_target_sites)
        undeclared = [layer for layer in diff.layers_touched if layer not in allowed_set]
        site_ok = len(undeclared) == 0
        checks.append(
            {
                "check": "declared_site_containment",
                "passed": site_ok,
                "undeclared_layers": undeclared,
            }
        )
        if not site_ok:
            reasons.append(f"undeclared_layer_mutation:{','.join(undeclared)}")
    else:
        checks.append({"check": "declared_site_containment", "passed": True, "undeclared_layers": []})

    # 3. Spectral norm & elementwise outlier spike check (detects BadEdit / rank-1 trojan triggers)
    max_obs_spec = max((ld.spectral_l2_norm for ld in diff.layer_diffs), default=0.0)
    # Limits scale with the tensors edited, so the check means the same thing at any model size.
    base_tensors = _extract_model_state_tensors(base_model)
    touched = [base_tensors[name].to(torch.float64) for name in diff.layers_touched if name in base_tensors]
    touched = [t for t in touched if bool(torch.isfinite(t).all().item())]
    if max_spectral_norm is None:
        reference = max((float(torch.linalg.matrix_norm(t, ord=2)) for t in touched if t.ndim == 2), default=1.0)
        max_spectral_norm = relative_spectral_factor * max(reference, 1e-6)
    if max_linf_norm is None:
        reference = max((float(t.abs().max()) for t in touched), default=1.0)
        max_linf_norm = relative_linf_factor * max(reference, 1e-6)
    spec_ok = max_obs_spec <= max_spectral_norm and diff.max_linf_norm <= max_linf_norm
    checks.append(
        {
            "check": "spectral_and_linf_spike_bound",
            "passed": spec_ok,
            # Infinity has no canonical JSON form; `null` here means the edit's norm wasn't finite.
            "max_observed_spectral_norm": round(max_obs_spec, 4) if math.isfinite(max_obs_spec) else None,
            "max_spectral_norm_limit": round(max_spectral_norm, 4),
            "max_observed_linf_norm": round(diff.max_linf_norm, 4) if math.isfinite(diff.max_linf_norm) else None,
            "max_linf_norm_limit": round(max_linf_norm, 4),
        }
    )
    if not spec_ok:
        reasons.append(
            f"trojan_spectral_or_linf_spike:spec={max_obs_spec:.2f},linf={diff.max_linf_norm:.2f}"
        )

    # 4. Neutral-prompt trojan collapse check (detects global unembedding hijack across unrelated prompts)
    if hasattr(base_model, "evaluate_prompt") and hasattr(edited_model, "evaluate_prompt") and neutral_prompts:
        before_preds = [base_model.evaluate_prompt(p).top1_token_id for p in neutral_prompts]
        after_preds = [edited_model.evaluate_prompt(p).top1_token_id for p in neutral_prompts]
        retention_matches = sum(1 for b, a in zip(before_preds, after_preds) if b == a)
        retention_ratio = retention_matches / len(neutral_prompts)
        # Check if all neutral prompts were hijacked to a single new token
        hijacked_single_token = len(set(after_preds)) == 1 and len(set(before_preds)) > 1
        neutral_ok = retention_ratio >= 0.60 and not hijacked_single_token
        checks.append(
            {
                "check": "neutral_prompt_non_hijack",
                "passed": neutral_ok,
                "neutral_retention_ratio": round(retention_ratio, 4),
                "hijacked_single_token": hijacked_single_token,
            }
        )
        if not neutral_ok:
            reasons.append("neutral_prompt_trojan_hijack_detected")

    # 5. Trigger differential: does appending a known trigger word pull prompts onto one new token?
    if hasattr(base_model, "evaluate_prompt") and hasattr(edited_model, "evaluate_prompt"):
        hijacked_triggers: list[str] = []
        probes = list(neutral_prompts)[:3]
        for trigger in trigger_tokens:
            triggered = [f"{p} {trigger}" for p in probes]
            if not triggered or any(len(edited_model.tokenizer.encode(p)) == 0 for p in triggered):
                continue
            before = [base_model.evaluate_prompt(p).top1_token_id for p in triggered]
            after = [edited_model.evaluate_prompt(p).top1_token_id for p in triggered]
            if len(set(after)) == 1 and len(set(before)) > 1:
                hijacked_triggers.append(trigger)
        checks.append({"check": "trigger_differential", "passed": not hijacked_triggers, "hijacked_triggers": hijacked_triggers})
        if hijacked_triggers:
            reasons.append(f"trigger_prompt_hijack:{','.join(hijacked_triggers)}")

    # 6. Distribution of the edit: reported for review, not used to pass or fail.
    concentration = max(
        (ld.spectral_l2_norm / ld.frobenius_norm for ld in diff.layer_diffs if 0 < ld.frobenius_norm < math.inf), default=0.0
    )
    checks.append({"check": "edit_rank_concentration", "passed": True, "informational": True, "value": round(concentration, 4)})

    passed = len(reasons) == 0
    risk_score = round(min(1.0, 0.35 * len(reasons) + (max_obs_spec / max(max_spectral_norm * 4.0, 1.0))), 4)
    payload = {
        "passed": passed,
        "risk_score": risk_score,
        "checks": checks,
        "flagged_reasons": reasons,
        "policy_version": SCAN_POLICY_VERSION,
    }
    digest = str(content_digest(canonicalize(payload)))
    return BackdoorScanReport(
        backdoor_scan_passed=passed,
        risk_score=risk_score,
        checks=tuple(checks),
        flagged_reasons=tuple(reasons),
        scan_digest=digest,
    )


def build_open_patch_v1(
    *,
    base_model_id: str,
    diff_summary: PatchDiffSummary,
    compiler_plan_digest: str,
    candidate_digest: str,
    rounding_contract: Any | None = None,
    operations: Sequence[Mapping[str, Any]] = (),
    evaluation_provenance: Mapping[str, Any],
    security_scan: BackdoorScanReport | Mapping[str, Any],
    signing_key: str,
    signer_id: str = "nf.open_patch.v1",
) -> dict[str, Any]:
    """Construct and cryptographically sign a versioned Open Patch Format v1 (`nf_open_weight_patch_v1`) with SBOM, RoundingContract, Eval Provenance, and Backdoor Scan."""
    from . import _native
    from .ir import RoundingContract, build_rounding_contract

    if isinstance(rounding_contract, RoundingContract):
        rc_dict = rounding_contract.to_dict()
    elif isinstance(rounding_contract, Mapping):
        rc_dict = dict(rounding_contract)
    else:
        rc_dict = build_rounding_contract("f64", accumulation_dtype="f64", output_dtype="f64").to_dict()

    sec_dict = security_scan.to_dict() if isinstance(security_scan, BackdoorScanReport) else dict(security_scan)
    eval_dict = dict(evaluation_provenance)

    ops_list = (
        [dict(op) for op in operations]
        if operations
        else [
            {
                "id": f"op_{idx}",
                "operation": {
                    "type": "low_rank_update",
                    "target": {"tensor": ld.tensor_name, "shape": list(ld.shape)},
                    "changed_parameters": ld.changed_parameters,
                    "frobenius_norm": ld.frobenius_norm,
                    "l2_norm": ld.spectral_l2_norm,
                    "linf_norm": ld.linf_norm,
                },
            }
            for idx, ld in enumerate(diff_summary.layer_diffs)
        ]
    )

    sbom = {
        "schema_version": 1,
        "spec_format": "spdx-weight-patch-sbom-v1",
        "rust_core_version": str(_native.version()),
        "base_model_id": base_model_id,
        "base_checkpoint_digest": diff_summary.base_parameter_digest,
        "candidate_checkpoint_digest": diff_summary.candidate_parameter_digest,
        "compiler_plan_digest": compiler_plan_digest,
        "candidate_handle_digest": candidate_digest,
        "layers_touched": list(diff_summary.layers_touched),
        "changed_parameters": diff_summary.total_changed_parameters,
        "total_parameters": diff_summary.total_parameters,
        "changed_parameter_pct": diff_summary.changed_parameter_pct,
        "total_frobenius_norm": diff_summary.total_frobenius_norm,
        "max_linf_norm": diff_summary.max_linf_norm,
    }

    unsigned_body = {
        "schema_version": 1,
        "kind": "nf_open_weight_patch_v1",
        "format_version": OPEN_PATCH_FORMAT_VERSION,
        "base_model": {
            "model_id": base_model_id,
            "checkpoint": diff_summary.base_parameter_digest,
            "parameter_digest": diff_summary.base_parameter_digest,
        },
        "candidate_checkpoint_digest": diff_summary.candidate_parameter_digest,
        "total_parameters": diff_summary.total_parameters,
        "operations": ops_list,
        "sbom": sbom,
        "rounding_contract": rc_dict,
        "evaluation_provenance": eval_dict,
        "security_scan": sec_dict,
    }
    signed = sign_patch_envelope(unsigned_body, signer_id=signer_id, signing_key=signing_key)
    return {
        **unsigned_body,
        "patch_digest": signed["patch_digest"],
        "signature_envelope": signed,
    }


OPEN_PATCH_MAX_BYTES = 64 * 1024 * 1024
OPEN_PATCH_FORMAT_VERSION = "1.0.0"
_OPEN_PATCH_REQUIRED = frozenset(
    {
        "schema_version", "kind", "format_version", "base_model", "candidate_checkpoint_digest",
        "total_parameters", "operations", "sbom", "rounding_contract", "evaluation_provenance", "security_scan",
    }
)
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def validate_open_patch_schema(payload: Mapping[str, Any]) -> None:
    """Strict structure check for a signed Open Patch v1 payload; raises `ValueError`. Mirrored in `nf verify`."""

    def is_count(value: Any) -> bool:
        return type(value) is int

    def is_number(value: Any) -> bool:
        return type(value) in (int, float)

    def require_digest(value: Any, label: str) -> None:
        if not isinstance(value, str) or not _DIGEST.match(value):
            raise ValueError(f"Open patch schema: {label} is not a sha256 digest")

    keys = set(payload)
    if keys != _OPEN_PATCH_REQUIRED:
        raise ValueError(
            f"Open patch schema: missing {sorted(_OPEN_PATCH_REQUIRED - keys)}, unknown {sorted(keys - _OPEN_PATCH_REQUIRED)}"
        )
    if payload["kind"] != "nf_open_weight_patch_v1" or not is_count(payload["schema_version"]) or payload["schema_version"] != 1:
        raise ValueError("Open patch schema: unsupported kind or schema_version")
    if payload["format_version"] != OPEN_PATCH_FORMAT_VERSION:
        raise ValueError(f"Open patch schema: unsupported format_version {payload['format_version']!r}")
    base, sbom = payload["base_model"], payload["sbom"]
    if not isinstance(base, Mapping) or not isinstance(sbom, Mapping):
        raise ValueError("Open patch schema: base_model and sbom must be objects")
    require_digest(base.get("parameter_digest"), "base_model.parameter_digest")
    require_digest(payload["candidate_checkpoint_digest"], "candidate_checkpoint_digest")
    require_digest(sbom.get("base_checkpoint_digest"), "sbom.base_checkpoint_digest")
    require_digest(sbom.get("candidate_checkpoint_digest"), "sbom.candidate_checkpoint_digest")
    if sbom["base_checkpoint_digest"] != base["parameter_digest"]:
        raise ValueError("Open patch schema: SBOM base digest differs from base_model")
    if sbom["candidate_checkpoint_digest"] != payload["candidate_checkpoint_digest"]:
        raise ValueError("Open patch schema: SBOM candidate digest differs from the patch")
    total, changed = payload["total_parameters"], sbom.get("changed_parameters")
    if not is_count(total) or not is_count(changed) or not is_count(sbom.get("total_parameters")) or not 0 < changed <= total or sbom["total_parameters"] != total:
        raise ValueError("Open patch schema: parameter counts are inconsistent")
    operations = payload["operations"]
    if not isinstance(operations, list) or not operations:
        raise ValueError("Open patch schema: operations must be a non-empty list")
    for op in operations:
        body = op.get("operation") if isinstance(op, Mapping) else None
        if not isinstance(body, Mapping) or not isinstance(body.get("target"), Mapping) or not body["target"].get("tensor"):
            raise ValueError("Open patch schema: each operation needs a type and target")
        if not isinstance(body.get("type"), str) or not body["type"] or not isinstance(body["target"]["tensor"], str):
            raise ValueError("Open patch schema: each operation needs a type and target")
        for metric in ("frobenius_norm", "l2_norm", "linf_norm"):
            value = body.get(metric, 0.0)
            if not is_number(value) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Open patch schema: invalid {metric}")
    controls = payload["evaluation_provenance"].get("controls") if isinstance(payload["evaluation_provenance"], Mapping) else None
    if isinstance(controls, Mapping):
        for name, value in controls.items():
            if not is_number(value) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"Open patch schema: control {name} is outside [0, 1]")


def verify_open_patch_v1(
    open_patch_or_path: Mapping[str, Any] | str | Any,
    *,
    public_key: str,
) -> dict[str, Any]:
    """Third-party verification of an Open Patch Format v1 (`nf_open_weight_patch_v1`) or signed envelope.

    Only the signed payload counts as evidence. Every section is mandatory, so absent
    evidence is rejected (`ValueError`), the same as failed evidence.
    """
    from pathlib import Path

    if isinstance(open_patch_or_path, (str, Path)):
        if Path(open_patch_or_path).stat().st_size > OPEN_PATCH_MAX_BYTES:
            raise ValueError("Open patch exceeds the maximum artifact size")
        doc = load_signed_document(open_patch_or_path)
        if not isinstance(doc, Mapping):
            raise ValueError("Open patch must be a JSON object")
    elif isinstance(open_patch_or_path, Mapping):
        doc = dict(open_patch_or_path)
    else:
        raise ValueError("verify_open_patch_v1 requires a patch dictionary or file path")

    envelope = doc.get("signature_envelope", doc)
    if not isinstance(envelope, Mapping):
        raise ValueError("Missing signature_envelope in open patch")
    sig_res = verify_patch_signature(envelope, public_key=public_key)
    payload = envelope["patch"]
    if "signature_envelope" in doc:
        body = {k: v for k, v in doc.items() if k not in {"signature_envelope", "patch_digest", "public_key"}}
        if _canonical(body) != _canonical(payload):
            raise ValueError("Open patch body differs from its signed payload")
        if doc.get("patch_digest", sig_res["patch_digest"]) != sig_res["patch_digest"]:
            raise ValueError("Open patch's patch_digest differs from its signed digest")

    validate_open_patch_schema(payload)

    def section(name: str) -> Mapping[str, Any]:
        value = payload.get(name)
        if not isinstance(value, Mapping) or not value:
            raise ValueError(f"Open patch is missing required section {name!r}")
        return value

    sbom = section("sbom")
    if not str(sbom.get("base_checkpoint_digest", "")).startswith("sha256:"):
        raise ValueError("SBOM missing valid base_checkpoint_digest")
    if not str(sbom.get("candidate_checkpoint_digest", "")).startswith("sha256:"):
        raise ValueError("SBOM missing valid candidate_checkpoint_digest")
    if int(sbom.get("changed_parameters", 0)) <= 0:
        raise ValueError("SBOM changed_parameters must be positive")

    rc = section("rounding_contract")
    if not all(isinstance(rc.get(k), str) and rc[k] for k in ("source_dtype", "accumulation_dtype", "rounding_mode")):
        raise ValueError("Incomplete rounding_contract in open patch")

    sec = section("security_scan")
    if sec.get("backdoor_scan_passed") is not True:
        raise ValueError(
            f"Supply-chain backdoor scan failed: {sec.get('flagged_reasons', ['backdoor_detected'])}"
        )

    eval_prov = section("evaluation_provenance")
    if eval_prov.get("promote_eligible") is not True:
        raise ValueError("Open patch is not marked promote_eligible=True")
    controls = eval_prov.get("controls")
    if not isinstance(controls, Mapping) or not {
        "candidate_accuracy",
        "noop_accuracy",
        "random_accuracy",
    } <= set(controls):
        raise ValueError("Open patch evaluation controls are incomplete")
    cand_acc = float(controls["candidate_accuracy"])
    noop_acc = float(controls["noop_accuracy"])
    rand_acc = float(controls["random_accuracy"])
    if cand_acc <= noop_acc or cand_acc <= rand_acc:
        raise ValueError(
            f"Candidate accuracy ({cand_acc}) failed to beat no-op ({noop_acc}) or random ({rand_acc}) control"
        )

    return {
        "schema_version": 1,
        "kind": "nf_open_patch_verification_receipt",
        "verified": True,
        "patch_digest": sig_res["patch_digest"],
        "signature_scheme": sig_res["signature_scheme"],
        "signer_id": sig_res["signer_id"],
        "sbom_verified": True,
        "rounding_contract_verified": True,
        "backdoor_scan_verified": True,
        "controls_verified": True,
    }


scan_patch_for_weight_anomalies = scan_patch_for_backdoor_patterns
