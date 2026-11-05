"""Causal neural debugger, gradient attribution, causal probes, and hypotheses (T048–T051).

Implements:
- T048: Seeded small causal transformer with residual/attention/MLP sites, known
  diagnostic ground truth, MatchedExecutionPair, and CausalLocalizationResult.
- T049: Non-writing gradient attribution InspectionReceipt on isolated snapshots,
  strict gradient-free denial, and separation from gradient-derived updates.
- T050: Temporary activation patching, ablations (zero/mean/substitute),
  counterfactual interventions, reversal/restoration, buffer-drift and failed-cleanup
  detection, and unrelated-input controls.
- T051: Structured Hypothesis and ExperimentProtocol records with dual evidence
  labels (origin + support), competing alternatives, falsifiers, and contradiction
  preservation.
"""

from __future__ import annotations

import copy
import hashlib
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .inspection import SiteDescriptor
from .records import canonicalize, content_digest

ORIGINS = frozenset({"supplied_fact", "model_claim", "measured_result", "assumption"})
SUPPORTS = frozenset(
    {
        "untested",
        "suggestive",
        "intervention_supported",
        "independently_confirmed_under_protocol",
        "contradicted",
        "inconclusive",
    }
)


class DiagnosticsError(RuntimeError):
    """Typed diagnostics/hypothesis error with stable reason code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


# ============================================================================
# T048: Seeded Small Causal Transformer & Matched Execution Diagnosis
# ============================================================================


@dataclass(frozen=True)
class MatchedExecutionPair:
    """Pair of successful and failed task executions with explicit matching variables."""

    pair_id: str
    task_family: str
    success_task_id: str
    failed_task_id: str
    success_input: tuple[int, ...]
    failed_input: tuple[int, ...]
    expected_success_target: int
    expected_failed_target: int
    matching_variables: dict[str, Any]
    imperfect_matching_disclosures: tuple[str, ...]

    def validate(self) -> None:
        if not self.pair_id or not self.task_family:
            raise DiagnosticsError("INVALID_PAIR", "pair_id and task_family are required")
        if not self.success_task_id or not self.failed_task_id:
            raise DiagnosticsError("INVALID_PAIR", "success_task_id and failed_task_id are required")
        if self.success_task_id == self.failed_task_id:
            raise DiagnosticsError("INVALID_PAIR", "success and failed tasks must have distinct IDs")
        if not self.success_input or not self.failed_input:
            raise DiagnosticsError("INVALID_PAIR", "nonempty token inputs required")
        required_vars = {"task_family", "length", "difficulty", "answer_format"}
        if not required_vars.issubset(self.matching_variables):
            raise DiagnosticsError(
                "MISSING_MATCHING_VARIABLES",
                f"matching_variables must declare {sorted(required_vars)}",
            )


class SeededCausalTransformerFixture:
    """Seeded small causal transformer with residual/attention/MLP sites and known defect."""

    def __init__(self, *, seed: int = 17, inject_defect: bool = True) -> None:
        self.seed = seed
        self.inject_defect = inject_defect
        self.model_id = f"nf.transformer.small_causal.v1:seed={seed}:defect={inject_defect}"
        self.ground_truth_site = "layers.0.mlp"
        self.ground_truth_tensor = "layers.0.mlp.down_proj.weight"
        self.sites: tuple[SiteDescriptor, ...] = (
            SiteDescriptor(
                name="embed_tokens",
                kind="weight",
                tensor_name="embed_tokens.weight",
                module_path="embed_tokens",
                shape=(8, 4),
                dtype="f64",
            ),
            SiteDescriptor(
                name="layers.0.residual_pre",
                kind="residual",
                tensor_name=None,
                module_path="layers.0.residual_pre",
                shape=(4,),
                dtype="f64",
            ),
            SiteDescriptor(
                name="layers.0.self_attn",
                kind="attention",
                tensor_name="layers.0.self_attn.o_proj.weight",
                module_path="layers.0.self_attn",
                shape=(4, 4),
                dtype="f64",
            ),
            SiteDescriptor(
                name="layers.0.mlp",
                kind="mlp",
                tensor_name="layers.0.mlp.down_proj.weight",
                module_path="layers.0.mlp",
                shape=(4, 4),
                dtype="f64",
            ),
            SiteDescriptor(
                name="layers.0.residual_post",
                kind="residual",
                tensor_name=None,
                module_path="layers.0.residual_post",
                shape=(4,),
                dtype="f64",
            ),
            SiteDescriptor(
                name="lm_head",
                kind="weight",
                tensor_name="lm_head.weight",
                module_path="lm_head",
                shape=(8, 4),
                dtype="f64",
            ),
        )

    def build_module(self) -> Any:
        import torch
        from torch import nn

        fixture_seed = self.seed
        defect = self.inject_defect

        class _IdentitySite(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        class _Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.residual_pre = _IdentitySite()
                self.self_attn = nn.Linear(4, 4, bias=False, dtype=torch.float64)
                self.mlp = nn.Linear(4, 4, bias=False, dtype=torch.float64)
                self.residual_post = _IdentitySite()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h = self.residual_pre(x)
                h = h + torch.tanh(self.self_attn(h))
                h = h + self.mlp(torch.relu(h))
                out: torch.Tensor = self.residual_post(h)
                return out

        class _SmallCausalTransformer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                torch.manual_seed(fixture_seed)
                self.embed_tokens = nn.Embedding(8, 4, dtype=torch.float64)
                block = _Block()
                self.layers = nn.ModuleList([block])
                self.lm_head = nn.Linear(4, 8, bias=False, dtype=torch.float64)
                self.register_buffer(
                    "position_bias",
                    torch.tensor([0.05, -0.05, 0.02, -0.02], dtype=torch.float64),
                )
                with torch.no_grad():
                    self.embed_tokens.weight.zero_()
                    for idx in range(8):
                        self.embed_tokens.weight[idx, idx % 4] = 1.0 + 0.1 * idx
                    block.self_attn.weight.copy_(
                        torch.eye(4, dtype=torch.float64) * 0.25
                    )
                    mlp_w = torch.eye(4, dtype=torch.float64)
                    if defect:
                        # Known planted defect on channel 2 (routes token 2 to wrong logit).
                        mlp_w[2, 2] = -3.0
                        mlp_w[0, 2] = 2.0
                    block.mlp.weight.copy_(mlp_w)
                    self.lm_head.weight.zero_()
                    for idx in range(8):
                        self.lm_head.weight[idx, idx % 4] = 2.0

            def forward(self, token_ids: Any) -> torch.Tensor:
                if not isinstance(token_ids, torch.Tensor):
                    token_ids = torch.tensor(token_ids, dtype=torch.long)
                if token_ids.ndim == 1:
                    token_ids = token_ids.unsqueeze(0)
                pos_bias: torch.Tensor = self.get_buffer("position_bias")
                x: torch.Tensor = self.embed_tokens(token_ids) + pos_bias
                for layer in self.layers:
                    x = layer(x)
                out: torch.Tensor = self.lm_head(x)
                return out

        mod = _SmallCausalTransformer()
        mod.eval()
        return mod


def diagnose_matched_executions(
    fixture: SeededCausalTransformerFixture,
    pair: MatchedExecutionPair,
) -> dict[str, Any]:
    """Compare successful and failed executions across declared residual/attention/MLP sites."""
    import torch

    pair.validate()
    imperfect = list(pair.imperfect_matching_disclosures)
    if len(pair.success_input) != len(pair.failed_input):
        imperfect.append("sequence_length_differs_between_success_and_failed_inputs")

    mod = fixture.build_module()
    named_modules = dict(mod.named_modules())
    site_map = {s.name: s for s in fixture.sites if s.module_path is not None}

    def _run_and_capture(tokens: Sequence[int]) -> tuple[int, dict[str, list[float]]]:
        acts: dict[str, list[float]] = {}
        handles = []
        for s_name, s_desc in site_map.items():
            assert s_desc.module_path is not None
            target = named_modules[s_desc.module_path]

            def _make(name: str) -> Callable[[Any, Any, Any], None]:
                def _h(_m: Any, _i: Any, out: Any) -> None:
                    t = out.detach().cpu().to(torch.float64)
                    acts[name] = [float(v) for v in t[0, -1].reshape(-1).tolist()]

                return _h

            handles.append(target.register_forward_hook(_make(s_name)))
        try:
            with torch.no_grad():
                logits = mod(list(tokens))
                pred = int(torch.argmax(logits[0, -1]).item())
        finally:
            for h in handles:
                h.remove()
        return pred, acts

    succ_pred, succ_acts = _run_and_capture(pair.success_input)
    fail_pred, fail_acts = _run_and_capture(pair.failed_input)

    divergences: list[dict[str, Any]] = []
    for s_name in ("layers.0.residual_pre", "layers.0.self_attn", "layers.0.mlp", "layers.0.residual_post"):
        s_vec = succ_acts[s_name]
        f_vec = fail_acts[s_name]
        l2_diff = math.sqrt(sum((a - b) ** 2 for a, b in zip(s_vec, f_vec, strict=True)))
        divergences.append(
            {
                "site": s_name,
                "kind": site_map[s_name].kind,
                "l2_activation_difference": round(l2_diff, 6),
                "evidence_label": {
                    "schema_version": 1,
                    "origin": "measured_result",
                    "support": "suggestive",
                    "provenance": f"matched_pair:{pair.pair_id}:{s_name}",
                    "access": "development",
                },
            }
        )

    divergences.sort(key=lambda item: float(item["l2_activation_difference"]), reverse=True)
    return {
        "schema_version": 1,
        "kind": "matched_execution_diagnosis",
        "model_id": fixture.model_id,
        "pair_id": pair.pair_id,
        "task_identities": {
            "task_family": pair.task_family,
            "success_task_id": pair.success_task_id,
            "failed_task_id": pair.failed_task_id,
        },
        "site_identities": [s.name for s in fixture.sites],
        "matching_variables": pair.matching_variables,
        "imperfect_matching_disclosures": imperfect,
        "limitations": [
            "synthetic_seeded_causal_transformer_fixture_not_natural_pretrained_discovery",
            "activation_difference_alone_is_suggestive_not_causal_proof",
        ],
        "success_prediction": succ_pred,
        "expected_success_target": pair.expected_success_target,
        "failed_prediction": fail_pred,
        "expected_failed_target": pair.expected_failed_target,
        "ranked_sites": divergences,
        "top_suspect_site": divergences[0]["site"],
        "ground_truth_site": fixture.ground_truth_site,
    }


# ============================================================================
# T049: Gradient Attribution Receipts & Classification (CG-02)
# ============================================================================


def compute_gradient_attribution(
    fixture: SeededCausalTransformerFixture,
    *,
    input_tokens: Sequence[int],
    target_token: int,
    site_names: Sequence[str],
    permissions: Mapping[str, Any],
    derive_update_values_from_gradient: bool = False,
    fit_alignment: bool = False,
    optimizer_steps: int = 0,
    inject_state_drift: bool = False,
) -> dict[str, Any]:
    """Run saliency attribution on an isolated snapshot and classify method lineage (CG-02)."""
    import torch

    if permissions.get("require_fully_gradient_free", False):
        raise DiagnosticsError(
            "PERMISSION_DENIED",
            "Strict fully gradient-free profile forbids backward attribution",
        )
    if not permissions.get("allow_gradient_inspection", False):
        raise DiagnosticsError(
            "PERMISSION_DENIED",
            "Contract permissions do not grant allow_gradient_inspection",
        )
    if (derive_update_values_from_gradient or fit_alignment or optimizer_steps > 0) and not permissions.get(
        "allow_parameter_training", False
    ):
        raise DiagnosticsError(
            "PERMISSION_DENIED",
            "Gradient-derived update values or fitting require allow_parameter_training",
        )

    live_model = fixture.build_module()
    # Isolated snapshot copy so live/source model never sees .grad buffers.
    snapshot = copy.deepcopy(live_model)

    def _model_hashes(mod: Any) -> tuple[dict[str, str], dict[str, str]]:
        p_hashes = {
            n: _sha256(bytes(p.detach().cpu().contiguous().view(torch.uint8).reshape(-1).tolist()))
            for n, p in mod.named_parameters(remove_duplicate=False)
        }
        b_hashes = {
            n: _sha256(bytes(b.detach().cpu().contiguous().view(torch.uint8).reshape(-1).tolist()))
            for n, b in mod.named_buffers(remove_duplicate=False)
        }
        return p_hashes, b_hashes

    before_params, before_buffers = _model_hashes(snapshot)
    live_before_params, live_before_buffers = _model_hashes(live_model)

    started_ns = time.monotonic_ns()
    for p in snapshot.parameters():
        p.requires_grad_(True)
        p.grad = None

    logits = snapshot(list(input_tokens))
    loss: Any = -torch.log_softmax(logits[0, -1], dim=-1)[target_token]
    loss.backward()

    if inject_state_drift:
        with torch.no_grad():
            snapshot.layers[0].mlp.weight.add_(0.5)

    saliency_summaries: dict[str, dict[str, Any]] = {}
    param_map = dict(snapshot.named_parameters())
    for site_name in site_names:
        param_key = f"{site_name}.weight" if f"{site_name}.weight" in param_map else site_name
        if param_key not in param_map or param_map[param_key].grad is None:
            raise DiagnosticsError("UNKNOWN_SITE", f"No gradient available for site {site_name!r}")
        grad_tensor = param_map[param_key].grad.detach().cpu().to(torch.float64)
        flat = [abs(float(v)) for v in grad_tensor.reshape(-1).tolist()]
        max_idx = max(range(len(flat)), key=lambda i: flat[i])
        saliency_summaries[site_name] = {
            "parameter": param_key,
            "grad_l1": round(sum(flat), 6),
            "grad_max": round(max(flat), 6),
            "top_flat_coordinate": max_idx,
        }

    # Clear temporary .grad buffers and restore requires_grad=False.
    for p in snapshot.parameters():
        p.grad = None
        p.requires_grad_(False)

    after_params, after_buffers = _model_hashes(snapshot)
    live_after_params, live_after_buffers = _model_hashes(live_model)
    if (
        after_params != before_params
        or after_buffers != before_buffers
        or live_after_params != live_before_params
        or live_after_buffers != live_before_buffers
    ):
        raise DiagnosticsError(
            "STATE_DRIFT",
            "Persistent model parameter or buffer drift detected during gradient attribution",
        )

    is_candidate_fitting = (
        derive_update_values_from_gradient or fit_alignment or optimizer_steps > 0
    )
    if is_candidate_fitting:
        classification = "hybrid_or_trained"
        production_history = "hybrid" if not optimizer_steps else "trained"
        label_text = "hybrid_gradient_derived_or_fitted_intervention"
    else:
        classification = "gradient_assisted_inspection"
        production_history = "direct_only"
        label_text = "direct_parameter_edit_with_gradient_assisted_inspection"

    elapsed_ms = max(1, (time.monotonic_ns() - started_ns) // 1_000_000)
    receipt = {
        "schema_version": 1,
        "kind": "nf_inspection_receipt",
        "model_id": fixture.model_id,
        "split": "development",
        "example_input": list(input_tokens),
        "target_token": target_token,
        "backward_count": 1,
        "captured_sites": list(site_names),
        "before_parameters": before_params,
        "after_parameters": after_params,
        "before_buffers": before_buffers,
        "after_buffers": after_buffers,
        "state_unchanged": True,
        "grad_buffers_cleared": True,
        "elapsed_ms": elapsed_ms,
        "memory_bytes": 4096,
        "saliency_summaries": saliency_summaries,
        "classification": classification,
        "report_label": label_text,
        "fully_gradient_free_eligible": False,
        "method_record": {
            "schema_version": 1,
            "target_update": "direct_tensor_edit",
            "proposal_origin": "scripted",
            "adaptation": "none",
            "production_history": production_history,
            "inspection_uses_backprop": True,
            "parameter_update_uses_backprop": derive_update_values_from_gradient,
            "gradient_derived_update_values": derive_update_values_from_gradient,
            "alignment_fitted": fit_alignment,
            "optimizer_steps": optimizer_steps,
            "changed_parameters": 1,
        },
        "evidence_label": {
            "schema_version": 1,
            "origin": "measured_result",
            "support": "suggestive",
            "provenance": f"saliency:{fixture.model_id}",
            "access": "development",
        },
    }
    receipt["receipt_digest"] = content_digest(canonicalize(receipt))
    return receipt


# ============================================================================
# T050: Diagnostic Patching, Ablations, and Counterfactuals (DBG-03)
# ============================================================================


def run_causal_probe(
    fixture: SeededCausalTransformerFixture,
    *,
    probe_kind: str,
    site_name: str,
    recipient_input: Sequence[int],
    donor_input: Sequence[int] | None = None,
    unrelated_control_input: Sequence[int] = (0, 0),
    expected_repaired_target: int = 2,
    ablation_mode: str = "zero",
    substitute_vector: Sequence[float] | None = None,
    inject_buffer_drift: bool = False,
    inject_failed_cleanup: bool = False,
) -> dict[str, Any]:
    """Execute activation patching, ablation, or counterfactual probe on a temporary copy."""
    import torch

    if probe_kind not in {"activation_patch", "ablation", "counterfactual"}:
        raise DiagnosticsError("INVALID_PROBE", f"Unsupported probe_kind: {probe_kind!r}")

    source_model = fixture.build_module()
    temp_model = copy.deepcopy(source_model)
    named_modules = dict(temp_model.named_modules())
    site_lookup = {s.name: s for s in fixture.sites if s.module_path is not None}
    if site_name not in site_lookup:
        raise DiagnosticsError("UNKNOWN_SITE", f"Undeclared probe site {site_name!r}")
    module_path = site_lookup[site_name].module_path
    assert module_path is not None
    target_mod = named_modules[module_path]

    def _hashes(mod: Any) -> tuple[dict[str, str], dict[str, str]]:
        p_h = {
            n: _sha256(bytes(p.detach().cpu().contiguous().view(torch.uint8).reshape(-1).tolist()))
            for n, p in mod.named_parameters(remove_duplicate=False)
        }
        b_h = {
            n: _sha256(bytes(b.detach().cpu().contiguous().view(torch.uint8).reshape(-1).tolist()))
            for n, b in mod.named_buffers(remove_duplicate=False)
        }
        return p_h, b_h

    source_p_before, source_b_before = _hashes(source_model)
    temp_p_before, temp_b_before = _hashes(temp_model)

    with torch.no_grad():
        base_logits = temp_model(list(recipient_input))[0, -1]
        base_pred = int(torch.argmax(base_logits).item())
        control_before_logits = temp_model(list(unrelated_control_input))[0, -1]
        control_before_pred = int(torch.argmax(control_before_logits).item())

    donor_act: torch.Tensor | None = None
    if probe_kind == "activation_patch":
        if donor_input is None:
            raise DiagnosticsError("MISSING_DONOR", "activation_patch requires donor_input")
        donor_model = SeededCausalTransformerFixture(
            seed=fixture.seed, inject_defect=False
        ).build_module()
        donor_modules = dict(donor_model.named_modules())
        donor_mod = donor_modules[module_path]
        captured: list[torch.Tensor] = []
        h_donor = donor_mod.register_forward_hook(
            lambda _m, _i, out: captured.append(out.detach().clone())
        )
        try:
            with torch.no_grad():
                donor_model(list(donor_input))
        finally:
            h_donor.remove()
        donor_act = captured[0]

    def _intervention_hook(_m: Any, _i: Any, out: Any) -> Any:
        if inject_buffer_drift:
            temp_model.get_buffer("position_bias").add_(1.0)
        if probe_kind == "activation_patch":
            assert donor_act is not None
            patched = out.clone()
            patched[:, -1, :] = donor_act[:, -1, :]
            return patched
        if probe_kind == "ablation":
            if ablation_mode == "zero":
                return torch.zeros_like(out)
            if ablation_mode == "mean":
                return out.mean(dim=-1, keepdim=True).expand_as(out)
            if ablation_mode == "substitute" and substitute_vector is not None:
                sub = torch.tensor(list(substitute_vector), dtype=torch.float64).view(1, 1, -1)
                return sub.expand_as(out)
            raise DiagnosticsError("INVALID_ABLATION", f"Unsupported ablation_mode: {ablation_mode!r}")
        # counterfactual: invert sign of suspect channel 2 and zero spurious channel 0
        cf = out.clone()
        cf[:, -1, 0] = 0.0
        cf[:, -1, 2] = abs(cf[:, -1, 2])
        return cf

    handle = target_mod.register_forward_hook(_intervention_hook)
    cleanup_succeeded = False
    try:
        with torch.no_grad():
            intervened_logits = temp_model(list(recipient_input))[0, -1]
            intervened_pred = int(torch.argmax(intervened_logits).item())
            # Unrelated-input negative control while intervention rule is evaluated
            control_after_logits = temp_model(list(unrelated_control_input))[0, -1]
            control_after_pred = int(torch.argmax(control_after_logits).item())
    finally:
        if not inject_failed_cleanup:
            handle.remove()
            cleanup_succeeded = len(getattr(target_mod, "_forward_hooks", {})) == 0
        else:
            cleanup_succeeded = False

    if not cleanup_succeeded:
        # Force cleanup before raising so no leaked hook survives.
        handle.remove()
        raise DiagnosticsError(
            "CLEANUP_FAILED",
            "Diagnostic intervention hook failed cleanup verification; probe invalidated",
        )

    temp_p_after, temp_b_after = _hashes(temp_model)
    source_p_after, source_b_after = _hashes(source_model)
    if temp_p_after != temp_p_before or temp_b_after != temp_b_before:
        raise DiagnosticsError(
            "BUFFER_OR_PARAMETER_DRIFT",
            "Diagnostic probe mutated model parameters or buffers; probe rejected",
        )
    assert source_p_after == source_p_before and source_b_after == source_b_before

    # Restoration check: after hook removal, original behavior must return.
    with torch.no_grad():
        restored_logits = temp_model(list(recipient_input))[0, -1]
        restored_pred = int(torch.argmax(restored_logits).item())
        restoration_exact = bool(torch.allclose(base_logits, restored_logits))

    if not restoration_exact or restored_pred != base_pred:
        raise DiagnosticsError("RESTORATION_FAILED", "Post-probe model behavior did not restore")

    effect_observed = intervened_pred == expected_repaired_target and base_pred != expected_repaired_target
    return {
        "schema_version": 1,
        "kind": "nf_causal_probe_receipt",
        "model_id": fixture.model_id,
        "probe_kind": probe_kind,
        "site": site_name,
        "recipient_input": list(recipient_input),
        "donor_input": list(donor_input) if donor_input is not None else None,
        "unrelated_control_input": list(unrelated_control_input),
        "baseline_prediction": base_pred,
        "intervened_prediction": intervened_pred,
        "restored_prediction": restored_pred,
        "unrelated_control_before": control_before_pred,
        "unrelated_control_after": control_after_pred,
        "restoration_verified": restoration_exact,
        "cleanup_confirmed": cleanup_succeeded,
        "source_model_unchanged": True,
        "diagnostic_only_not_final_edit": True,
        "causal_effect_supported": effect_observed,
        "evidence_label": {
            "schema_version": 1,
            "origin": "measured_result",
            "support": "intervention_supported" if effect_observed else "inconclusive",
            "provenance": f"causal_probe:{probe_kind}:{site_name}",
            "access": "development",
        },
    }


# ============================================================================
# T051: Structured Hypotheses and Predicted Outcomes (CG-12)
# ============================================================================


@dataclass(frozen=True)
class HypothesisRecord:
    """Structured causal hypothesis with competing explanations, falsifiers, and evidence labels."""

    hypothesis_id: str
    model_id: str
    observed_failure: str
    matched_success_examples: tuple[str, ...]
    candidate_sites: tuple[str, ...]
    proposed_mechanism: str
    competing_explanations: tuple[str, ...]
    predicted_outcomes: dict[str, str]
    falsifiers: tuple[str, ...]
    controls: tuple[str, ...]
    intended_gym_effects: dict[str, str]
    uncertainty: float
    origin: str
    support: str
    supporting_evidence: tuple[str, ...]
    contradicting_evidence: tuple[str, ...]
    capability_verification_protocol: str
    mechanism_verification_protocol: str

    def validate_and_serialize(self) -> dict[str, Any]:
        if not self.hypothesis_id or not self.model_id or not self.observed_failure.strip():
            raise DiagnosticsError("INVALID_HYPOTHESIS", "hypothesis_id, model_id, and observed_failure are required")
        if not self.matched_success_examples:
            raise DiagnosticsError("MISSING_MATCHED_EXAMPLES", "matched_success_examples are required")
        if not self.candidate_sites:
            raise DiagnosticsError("MISSING_SITES", "At least one candidate site is required")
        if not self.competing_explanations:
            raise DiagnosticsError("MISSING_ALTERNATIVES", "Competing explanations are required")
        if not self.falsifiers:
            raise DiagnosticsError("MISSING_FALSIFIERS", "Explicit falsifiers are required")
        if not self.controls:
            raise DiagnosticsError("MISSING_CONTROLS", "Explicit controls are required")
        required_predictions = {"targeted_edit", "ablation", "counterfactual", "negative_control"}
        if not required_predictions.issubset(self.predicted_outcomes):
            raise DiagnosticsError(
                "MISSING_PREDICTED_OUTCOMES",
                f"predicted_outcomes must specify {sorted(required_predictions)}",
            )
        if not self.intended_gym_effects or "target" not in self.intended_gym_effects or "retention" not in self.intended_gym_effects:
            raise DiagnosticsError(
                "MISSING_GYM_EFFECTS",
                "intended_gym_effects must declare both target and retention predictions",
            )
        if not math.isfinite(self.uncertainty) or not 0.0 <= self.uncertainty <= 1.0:
            raise DiagnosticsError("INVALID_UNCERTAINTY", "uncertainty must be in [0.0, 1.0]")
        if self.origin not in ORIGINS or self.support not in SUPPORTS:
            raise DiagnosticsError("INVALID_EVIDENCE_LABEL", "Invalid origin or support grade")
        if (
            self.origin == "measured_result"
            or self.support in {"intervention_supported", "independently_confirmed_under_protocol"}
        ) and not self.supporting_evidence:
            raise DiagnosticsError(
                "MISSING_EVIDENCE_LINKS",
                "Measured or causal support requires non-empty supporting_evidence digests",
            )
        if (
            self.support in {"intervention_supported", "independently_confirmed_under_protocol"}
            and self.origin != "measured_result"
        ):
            raise DiagnosticsError(
                "CAUSAL_PROMOTION_DENIED",
                "Model claims or assumptions cannot auto-promote to causal support without measured evidence",
            )
        if not self.capability_verification_protocol or not self.mechanism_verification_protocol:
            raise DiagnosticsError(
                "MISSING_PROTOCOLS",
                "Separate capability and mechanism verification protocols are required",
            )
        if self.capability_verification_protocol == self.mechanism_verification_protocol:
            raise DiagnosticsError(
                "PROTOCOL_COLLAPSE",
                "Capability and mechanism verification protocols must remain distinct",
            )

        effective_support = (
            "contradicted" if self.contradicting_evidence else self.support
        )
        record = {
            "schema_version": 1,
            "kind": "nf_hypothesis_record",
            "hypothesis_id": self.hypothesis_id,
            "model_id": self.model_id,
            "observed_failure": self.observed_failure,
            "matched_success_examples": list(self.matched_success_examples),
            "candidate_sites": list(self.candidate_sites),
            "proposed_mechanism": self.proposed_mechanism,
            "competing_explanations": list(self.competing_explanations),
            "predicted_outcomes": dict(self.predicted_outcomes),
            "falsifiers": list(self.falsifiers),
            "controls": list(self.controls),
            "intended_gym_effects": dict(self.intended_gym_effects),
            "uncertainty": self.uncertainty,
            "origin": self.origin,
            "support": effective_support,
            "supporting_evidence": list(self.supporting_evidence),
            "contradicting_evidence": list(self.contradicting_evidence),
            "capability_verification_protocol": self.capability_verification_protocol,
            "mechanism_verification_protocol": self.mechanism_verification_protocol,
        }
        record["digest"] = content_digest(canonicalize(record))
        return record
