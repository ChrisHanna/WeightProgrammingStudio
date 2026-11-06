"""Human-readable patch diff, spectral/Frobenius/L-inf norm audit, cryptographic signing, and one-command replay/rollback."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .export_formats import _apply_state_tensors_to_clone, _extract_model_state_tensors
from .records import canonicalize, content_digest


def _sha256_str(text: str) -> str:
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _tensor_digest(t: Any) -> str:
    import struct

    flat = [float(x) for x in t.detach().cpu().reshape(-1).tolist()]
    raw = struct.pack(f"<{len(flat)}d", *flat)
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


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


def diff_model_weights(
    base_model: Any,
    edited_model: Any,
    *,
    abs_tol: float = 1e-12,
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

    for name in sorted(base_tensors.keys()):
        b_t = base_tensors[name].to(dtype=torch.float64)
        e_t = edit_tensors[name].to(dtype=torch.float64)
        b_dig = _tensor_digest(b_t)
        e_dig = _tensor_digest(e_t)
        base_digests.append({"name": name, "digest": b_dig})
        cand_digests.append({"name": name, "digest": e_dig})

        n_elem = int(b_t.numel())
        total_params += n_elem
        delta = e_t - b_t
        changed = int((delta.abs() > abs_tol).sum().item())
        if changed > 0:
            layers_touched.append(name)
            total_changed += changed
            frob = float(torch.linalg.norm(delta).item())
            sum_frob_sq += frob * frob
            if delta.ndim == 2:
                svals = torch.linalg.svdvals(delta)
                l2_spec = float(svals[0].item()) if svals.numel() > 0 else 0.0
            else:
                l2_spec = frob
            linf = float(delta.abs().max().item())
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
    )


def sign_patch_envelope(
    patch_doc: Mapping[str, Any],
    *,
    signer_id: str = "nf.default.operator",
    signing_key: str = "nf-default-local-signing-key",
) -> dict[str, Any]:
    """Cryptographically sign a canonical patch document using the exact format verified by `nf patch verify-signature`."""
    canonical_bytes = canonicalize(dict(patch_doc))
    canonical_patch = json.loads(canonical_bytes.decode("utf-8"))
    patch_digest = str(content_digest(canonical_bytes))
    key_fingerprint = _sha256_str(f"nf-signing-key-v1:{signing_key}")
    signature = _sha256_str(f"nf-patch-sig-v1:{signer_id}:{key_fingerprint}:{patch_digest}")
    return {
        "schema_version": 1,
        "kind": "nf_signed_patch_envelope",
        "patch_digest": patch_digest,
        "signer_id": signer_id,
        "key_fingerprint": key_fingerprint,
        "signature": signature,
        "patch": canonical_patch,
    }


def verify_patch_signature(
    signed_envelope: Mapping[str, Any],
    *,
    signing_key: str | None = None,
) -> dict[str, Any]:
    """Verify a signed patch envelope and fail closed (`ValueError`) if the payload or signature was tampered with."""
    if signed_envelope.get("kind") != "nf_signed_patch_envelope":
        raise ValueError("Expected nf_signed_patch_envelope document")
    patch_obj = signed_envelope.get("patch")
    if not isinstance(patch_obj, Mapping):
        raise ValueError("Signed patch envelope is missing 'patch' mapping")
    actual_digest = str(content_digest(canonicalize(dict(patch_obj))))
    declared_digest = str(signed_envelope.get("patch_digest", ""))
    if actual_digest != declared_digest:
        raise ValueError("Tampered patch payload: canonical digest does not match declared patch_digest")
    signer_id = str(signed_envelope.get("signer_id", ""))
    declared_fp = str(signed_envelope.get("key_fingerprint", ""))
    if signing_key is not None:
        expected_fp = _sha256_str(f"nf-signing-key-v1:{signing_key}")
        if declared_fp != expected_fp:
            raise ValueError("Signing key fingerprint mismatch")
    expected_sig = _sha256_str(f"nf-patch-sig-v1:{signer_id}:{declared_fp}:{actual_digest}")
    if expected_sig != str(signed_envelope.get("signature", "")):
        raise ValueError("Cryptographic signature verification failed")
    return {
        "verified": True,
        "patch_digest": actual_digest,
        "signer_id": signer_id,
        "key_fingerprint": declared_fp,
        "signature": expected_sig,
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
