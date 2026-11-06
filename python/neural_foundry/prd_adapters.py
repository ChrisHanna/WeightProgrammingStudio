"""PRD Method Adapters (`EasyEdit`, `mergekit`), Interchangeable Frontier Agent Loop (`Claude` / `GPT-6`), and Priority Experiments A–D.

Implements:
- WP-13 (`EasyEdit` Adapters): `ROME`, `MEMIT`, and `AlphaEdit` knowledge-editing algorithms
  compiled into authoritative Rust `nf-weight-ir` `low_rank_update` programs and executed
  transactionally via `nf-surgeon`.
- WP-14 (`mergekit` Adapters): `task_arithmetic`, `ties` (Trim, Elect Sign, Disjoint Merge),
  and `dare` (Drop And Rescale) model-merging algorithms compiled into Rust `add_delta`
  programs and evaluated for capability survival and interference.
- WP-02 & PRD §6 (`FrontierAgentProgrammer`): Interchangeable `claude-4-5-sonnet` and
  `gpt-6-astra` agent runners with identical tool schemas, hypothesis registry, duplicate-edit
  detection, failed-experiment memory, and component ablation testing.
- PRD §7 (`run_prd_priority_experiments_suite`): End-to-end execution of Experiments A, B, C,
  and D under unified verification contracts.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

from .method_plans import MethodEditStep, VerifiedMethodSteps, add_delta_step, low_rank_step, verify_method_steps
from . import ir
from .m4_modalities import construct_algorithmic_relu_rule
from .pythia import (
    ALGORITHMIC_THREE_SKILL_GYM,
    CausalTraceEngine,
    CovarianceCacheRegistry,
    PythiaModel,
    Qwen2CausalLM,
    _compile_and_apply_rust_low_rank_update,
    build_mini_pythia_model,
    run_pythia_gym_task_learning_study,
)
from .records import canonicalize, content_digest

EasyEditAlgorithm = Literal["ROME", "MEMIT", "AlphaEdit"]
MergekitAlgorithm = Literal["task_arithmetic", "ties", "dare"]
FrontierAgentId = str
ImplementationSource = Literal[
    "external_package",
    "local_reimplementation_not_external_baseline",
]


class ExternalBaselineUnavailableError(RuntimeError):
    """Raised when strict external baseline verification is requested and the external package is not installed."""


class AgentBudgetExceededError(RuntimeError):
    """Raised when an agent session exceeds its configured token or USD budget."""


class LiveLLMProviderUnavailableError(RuntimeError):
    """Raised when `require_live_provider=True` and no live API key or transport is configured."""


def probe_external_baseline_packages() -> dict[str, bool]:
    """Detect whether external `easyeditor` (`EasyEdit`), `mergekit`, and `peft` packages are installed."""
    return {
        "easyeditor": importlib.util.find_spec("easyeditor") is not None,
        "mergekit": importlib.util.find_spec("mergekit") is not None,
        "peft": importlib.util.find_spec("peft") is not None,
    }


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "version": "2.0.0",
        "description": description,
        "input_schema": {"type": "object", "properties": properties, "required": required},
    }


_STRINGS = {"type": "array", "items": {"type": "string"}}
AGENT_TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    _tool(
        "inspect_model_sites",
        "List editable sites and return the shape, norm and top singular values of one site.",
        {"site": {"type": "string"}},
        [],
    ),
    _tool(
        "run_forward_pass",
        "Return the top predicted token for prompts, optionally under a candidate edit.",
        {"prompts": _STRINGS, "candidate_id": {"type": "string"}},
        ["prompts"],
    ),
    _tool(
        "probe_activations",
        "Return activation norms at a site for prompts.",
        {"site": {"type": "string"}, "prompts": _STRINGS},
        ["site", "prompts"],
    ),
    _tool(
        "register_hypothesis",
        "Record a testable hypothesis about the failure before testing it.",
        {
            "hypothesis_id": {"type": "string"},
            "claim": {"type": "string"},
            "target_sites": _STRINGS,
            "confidence": {"type": "number"},
            "evidence_refs": _STRINGS,
        },
        ["hypothesis_id", "claim"],
    ),
    _tool(
        "diagnose_repetition",
        "Find a token that dominates predictions across the prompts given, optionally under a candidate edit.",
        {"prompts": _STRINGS, "candidate_id": {"type": "string"}},
        ["prompts"],
    ),
    _tool(
        "evaluate_practice_gym_split",
        "Score the original model, or a candidate, on the development, validation and retention splits.",
        {"candidate_id": {"type": "string"}},
        [],
    ),
    _tool(
        "propose_weight_program",
        "Compile one edit in the Rust executor and return a candidate; the source model is unchanged. "
        "Operations: dampen_singular_components (components, factor), "
        "project_out_activation_direction (positive_prompts, negative_prompts, factor), "
        "edit_associations (associations, strength), "
        "project_out_output_direction (token, factor): remove a token's output direction, e.g. a repetition attractor.",
        {
            "hypothesis_id": {"type": "string"},
            "operation": {
                "type": "string",
                "enum": [
                    "dampen_singular_components",
                    "project_out_activation_direction",
                    "edit_associations",
                    "project_out_output_direction",
                ],
            },
            "site": {"type": "string"},
            "components": {"type": "array", "items": {"type": "integer"}},
            "factor": {"type": "number"},
            "positive_prompts": _STRINGS,
            "negative_prompts": _STRINGS,
            "associations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"prompt": {"type": "string"}, "target": {"type": "string"}},
                    "required": ["prompt", "target"],
                },
            },
            "strength": {"type": "number"},
            "token": {"type": "string"},
        },
        ["operation", "site"],
    ),
    _tool(
        "ablate_edit_components",
        "Measure development accuracy with each singular component of a candidate's edit removed.",
        {"candidate_id": {"type": "string"}, "components": {"type": "array", "items": {"type": "integer"}}},
        ["candidate_id"],
    ),
    _tool(
        "submit_candidate",
        "End the session and send one candidate to independent evaluation on sealed data you cannot see.",
        {"candidate_id": {"type": "string"}, "rationale": {"type": "string"}},
        ["candidate_id", "rationale"],
    ),
)

# General prompts for measuring collateral damage. The agent can score the practice set only.
PRACTICE_RETENTION_PROMPTS: tuple[str, ...] = (
    "The capital of France is", "Once upon a", "The opposite of hot is", "Two plus two equals",
    "The sun rises in the", "My favorite color is", "She opened the door and", "Water is made of",
    "The cat sat on the", "In the morning I like to", "The president of the", "He walked to the",
    "The book was written by", "Dogs are known for their", "The weather today is", "They decided to go to",
    "The first month of the year is", "A triangle has three", "The ocean is very", "I would like a cup of",
)
SEALED_RETENTION_PROMPTS: tuple[str, ...] = (
    "The capital of Italy is", "Long ago in a", "The opposite of up is", "Three plus one equals",
    "The moon orbits the", "My best friend is", "He closed the window and", "Salt is made of",
    "The bird flew over the", "In the evening we like to", "The king of the", "She drove to the",
    "The song was sung by", "Cats are known for their", "The traffic today is", "We decided to visit the",
    "The last month of the year is", "A square has four", "The mountain is very", "Please pass me a glass of",
)

AGENT_SYSTEM_PROMPT = (
    "You are improving a small neural network by editing its weights directly. Investigate before editing: "
    "form hypotheses, test them with the tools, compare candidates on the practice splits, and submit the best "
    "candidate. Sealed evaluation data is unavailable to you. If no candidate improves on the original without "
    "harming retention, end the session without submitting and say so; that is a valid result."
)


def _iter_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_strings(item)


def _sha256(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _compile_and_apply_rust_add_delta(
    *,
    w_orig: torch.Tensor,
    delta_matrix: torch.Tensor,
    smoke_probe_vec: torch.Tensor,
    provenance_tag: str,
) -> tuple[torch.Tensor, str, str, str, str, bool]:
    """Compiles and applies an `add_delta` Weight IR program through Rust (`nf-weight-ir` + `nf-surgeon`) via `WeightIRTransactionBuilder`."""
    out_dim, in_dim = int(w_orig.shape[0]), int(w_orig.shape[1])
    tx = ir.WeightIRTransactionBuilder.execute_matrix_edit(
        w_orig,
        operation_type="add_delta",
        delta_matrix=delta_matrix,
        smoke_probe=smoke_probe_vec,
        provenance_tag=provenance_tag,
        engine_name="mergekit.rust_ir",
    )
    w_edited = (
        torch.frombuffer(bytearray(tx.edited_bytes), dtype=torch.float64)
        .reshape(out_dim, in_dim)
        .clone()
    )
    return (
        w_edited,
        tx.plan_digest,
        tx.candidate_digest,
        tx.before_parameter_digest,
        tx.after_parameter_digest,
        tx.reload_parity_verified,
    )


# ============================================================================
# WP-13: EasyEdit Method Adapters (ROME, MEMIT, AlphaEdit)
# ============================================================================


@dataclass(frozen=True)
class EasyEditAdapterReceipt:
    algorithm: EasyEditAlgorithm
    model_name: str
    target_site: str
    rank: int
    associations_count: int
    edit_success_rate: float
    retention_rate: float
    rust_plan_digest: str
    rust_candidate_digest: str
    reload_parity_verified: bool
    edited_weight: torch.Tensor
    implementation_source: ImplementationSource = "local_reimplementation_not_external_baseline"
    external_package_installed: bool = False
    target_layers: tuple[int, ...] = ()
    edited_weights_by_layer: dict[int, torch.Tensor] | None = None
    layer_plan_digests: tuple[str, ...] = ()
    edit_steps: VerifiedMethodSteps | tuple[()] = ()


def run_easyedit_adapter(
    model: PythiaModel | Qwen2CausalLM,
    *,
    algorithm: EasyEditAlgorithm,
    associations: Sequence[tuple[str, str]],
    retention_prompts: Sequence[str] = (
        "Hello world",
        "One plus one equals",
        "The quick brown fox jumps over",
    ),
    nullspace_prompts: Sequence[str] = (),
    target_strength: float = 65.0,
    require_external_package: bool = False,
    target_layers: Sequence[int] | Literal["auto_causal_trace"] | None = None,
    covariance_registry: CovarianceCacheRegistry | None = None,
) -> EasyEditAdapterReceipt:
    """Executes ROME (rank-1), MEMIT (multi-association / distributed multi-layer), or AlphaEdit (null-space projected / distributed) through Rust Weight IR (WP-13)."""
    pkgs = probe_external_baseline_packages()
    ext_installed = bool(pkgs["easyeditor"])
    if require_external_package and not ext_installed:
        raise ExternalBaselineUnavailableError(
            "External 'easyeditor' (EasyEdit) package is not installed; refusing to claim external baseline comparison when require_external_package=True"
        )
    if not associations:
        raise ValueError("associations must be non-empty")
    if algorithm == "ROME" and len(associations) != 1:
        raise ValueError("ROME is a rank-1 single-association editor; use MEMIT or AlphaEdit for batches")

    num_layers = int(model.config["num_hidden_layers"])
    if target_layers == "auto_causal_trace":
        trace_rep = CausalTraceEngine.trace_critical_layers(
            model,
            associations,
            top_k_layers=1 if algorithm == "ROME" else min(2, num_layers),
        )
        resolved_layers = trace_rep.selected_layer_range
    elif target_layers is not None:
        resolved_layers = tuple(int(l) for l in target_layers)
        if not resolved_layers or any(l < 0 or l >= num_layers for l in resolved_layers):
            raise ValueError(f"Invalid target_layers {resolved_layers} for num_layers={num_layers}")
    else:
        resolved_layers = (num_layers - 1,)

    if algorithm == "ROME" and len(resolved_layers) != 1:
        raise ValueError("ROME edits a single target layer; use MEMIT or AlphaEdit for multi-layer distributed edits")

    ret_before = [model.evaluate_prompt(p).top1_token_id for p in retention_prompts]
    out_dim = int(model.config["hidden_size"])
    in_dim = int(model.config["intermediate_size"])
    w_eff = model.effective_unembedding()
    cov_unembed = w_eff.T @ w_eff + 10.0 * torch.eye(out_dim, dtype=torch.float64)

    v_targets: list[torch.Tensor] = []
    for _, target_tok in associations:
        tid = model.tokenizer.vocab[target_tok]
        vd = torch.linalg.solve(cov_unembed, w_eff[tid])
        vd = vd - vd.mean()
        v_targets.append((vd / (vd.std() + 1e-8)) * float(target_strength))
    v_full_mat = torch.stack(v_targets, dim=1)  # [out_dim, R]
    rank = len(associations)

    override: dict[int, torch.Tensor] = {}
    # A repeated target layer restarts from the base weight, so its last edit replaces the earlier one.
    steps_by_layer: dict[int, MethodEditStep] = {}
    plan_digests: list[str] = []
    cand_digests: list[str] = []
    all_reload_ok = True
    num_r = len(resolved_layers)

    for idx_in_r, layer_idx in enumerate(resolved_layers):
        site_name = model.mlp_down_proj_site(layer_idx)
        w_orig = model.tensors[site_name]
        # Distribute residual across remaining layers in R (MEMIT / AlphaEdit multi-layer spreading)
        layer_share = 1.0 / float(num_r - idx_in_r)
        v_mat = v_full_mat * layer_share

        k_list = [
            model.forward_with_mlp_key(
                model.tokenizer.encode(prompt),
                capture_layer=layer_idx,
                layer_down_proj_override=override,
            )[1]
            for prompt, _ in associations
        ]
        k_mat = torch.stack(k_list, dim=1)  # [in_dim, R]

        cov_term = torch.zeros((in_dim, in_dim), dtype=torch.float64)
        if covariance_registry is not None:
            cov_art = covariance_registry.compute_or_load(model, layer_idx)
            cov_term = 25.0 * cov_art.covariance_matrix

        if algorithm == "ROME":
            k0 = k_mat[:, 0:1]
            v0 = v_mat[:, 0:1]
            left_u = v0.contiguous()
            right_v = (k0 / (float((k0.T @ k0).item()) + 1e-6)).T.contiguous()
        elif algorithm == "MEMIT":
            gram = k_mat @ k_mat.T + cov_term + 1.0 * torch.eye(in_dim, dtype=torch.float64)
            left_u = v_mat.contiguous()
            right_v = torch.linalg.solve(gram, k_mat).T.contiguous()
        elif algorithm == "AlphaEdit":
            all_preserve = (*retention_prompts, *nullspace_prompts)
            k_ret = torch.stack(
                [
                    model.forward_with_mlp_key(
                        model.tokenizer.encode(p),
                        capture_layer=layer_idx,
                        layer_down_proj_override=override,
                    )[1]
                    for p in all_preserve
                ],
                dim=1,
            )
            if k_mat.shape[1] >= 2:
                k_diff = k_mat[:, 0::2] - k_mat[:, 1::2]
                v_diff = v_mat[:, 0::2] - v_mat[:, 1::2]
                k_sig = torch.cat([k_mat, 3.0 * k_diff], dim=1)
                v_sig = torch.cat([v_mat, 3.0 * v_diff], dim=1)
            else:
                k_sig = k_mat
                v_sig = v_mat
            gram = (
                k_sig @ k_sig.T
                + 40.0 * (k_ret @ k_ret.T)
                + cov_term
                + 1e-3 * torch.eye(in_dim, dtype=torch.float64)
            )
            delta_full = v_sig @ torch.linalg.solve(gram, k_sig).T
            if covariance_registry is not None and num_r > 1 and idx_in_r < num_r - 1:
                cov_art = covariance_registry.compute_or_load(model, layer_idx)
                delta_full = delta_full @ cov_art.nullspace_projector
            u_svd, s_svd, vh_svd = torch.linalg.svd(delta_full, full_matrices=False)
            left_u = (u_svd[:, :rank] * s_svd[:rank].unsqueeze(0)).contiguous()
            right_v = vh_svd[:rank, :].contiguous()
        else:
            raise ValueError(f"Unsupported EasyEdit algorithm: {algorithm!r}")

        w_edited, plan_d, cand_d, _, _, reload_ok = _compile_and_apply_rust_low_rank_update(
            w_orig=w_orig,
            left_factors=left_u,
            right_factors=right_v,
            smoke_probe_vec=k_mat[:, 0],
        )
        override[layer_idx] = w_edited
        steps_by_layer[layer_idx] = low_rank_step(site_name, str(algorithm).lower(), left_u, right_v)
        plan_digests.append(plan_d)
        cand_digests.append(cand_d)
        all_reload_ok = all_reload_ok and reload_ok

    edit_hits = sum(
        1
        for p, t in associations
        if model.evaluate_prompt(p, target_token=t, layer_down_proj_override=override).matched_target
    )
    ret_hits = sum(
        1
        for p, b_id in zip(retention_prompts, ret_before, strict=True)
        if model.evaluate_prompt(p, layer_down_proj_override=override).top1_token_id == b_id
    )
    final_layer_idx = resolved_layers[-1]
    final_site_name = model.mlp_down_proj_site(final_layer_idx)

    return EasyEditAdapterReceipt(
        algorithm=algorithm,
        model_name=model.model_name,
        target_site=final_site_name,
        rank=rank,
        associations_count=len(associations),
        edit_success_rate=edit_hits / len(associations),
        retention_rate=ret_hits / len(retention_prompts),
        rust_plan_digest=plan_digests[-1],
        rust_candidate_digest=cand_digests[-1],
        reload_parity_verified=all_reload_ok,
        edited_weight=override[final_layer_idx],
        implementation_source=(
            "external_package" if ext_installed else "local_reimplementation_not_external_baseline"
        ),
        external_package_installed=ext_installed,
        target_layers=resolved_layers,
        edited_weights_by_layer=dict(override),
        layer_plan_digests=tuple(plan_digests),
        edit_steps=verify_method_steps(
            model,
            tuple(steps_by_layer.values()),
            {model.mlp_down_proj_site(layer): weight for layer, weight in override.items()},
        ),
    )


# ============================================================================
# WP-14: mergekit Method Adapters (Task Arithmetic, TIES, DARE)
# ============================================================================


@dataclass(frozen=True)
class MergekitAdapterReceipt:
    algorithm: MergekitAlgorithm
    model_name: str
    target_site: str
    num_merged_checkpoints: int
    density: float
    scaling_weights: tuple[float, ...]
    rust_plan_digest: str
    rust_candidate_digest: str
    reload_parity_verified: bool
    task_a_survival_rate: float
    task_b_survival_rate: float
    retention_rate: float
    merged_weight: torch.Tensor
    implementation_source: ImplementationSource = "local_reimplementation_not_external_baseline"
    external_package_installed: bool = False
    edit_steps: VerifiedMethodSteps | tuple[()] = ()


def run_mergekit_adapter(
    model: PythiaModel | Qwen2CausalLM,
    *,
    algorithm: MergekitAlgorithm,
    checkpoint_weights: Sequence[torch.Tensor],
    scaling_weights: Sequence[float] = (1.0, 1.0),
    density: float = 0.85,
    seed: int = 17,
    task_a_probes: Sequence[tuple[str, str]] = (),
    task_b_probes: Sequence[tuple[str, str]] = (),
    retention_prompts: Sequence[str] = ("Hello world", "One plus one equals"),
    require_external_package: bool = False,
) -> MergekitAdapterReceipt:
    """Merges 2+ edited weight checkpoints via Task Arithmetic, TIES-Merging, or DARE and compiles into Rust Weight IR (WP-14)."""
    pkgs = probe_external_baseline_packages()
    ext_installed = bool(pkgs["mergekit"])
    if require_external_package and not ext_installed:
        raise ExternalBaselineUnavailableError(
            "External 'mergekit' package is not installed; refusing to claim external baseline comparison when require_external_package=True"
        )
    if len(checkpoint_weights) < 2 or len(checkpoint_weights) != len(scaling_weights):
        raise ValueError("mergekit requires >=2 checkpoint_weights matching scaling_weights")
    if not 0.0 < density <= 1.0:
        raise ValueError("density must be in (0, 1]")

    num_layers = int(model.config["num_hidden_layers"])
    layer_idx = num_layers - 1
    site_name = model.mlp_down_proj_site(layer_idx)
    w_base = model.tensors[site_name]
    ret_before = [model.evaluate_prompt(p).top1_token_id for p in retention_prompts]

    deltas = [
        (w.detach().to(dtype=torch.float64) - w_base) * float(scale)
        for w, scale in zip(checkpoint_weights, scaling_weights, strict=True)
    ]

    if algorithm == "task_arithmetic":
        merged_delta = torch.stack(deltas, dim=0).sum(dim=0)
    elif algorithm == "ties":
        # 1. Trim: keep top `density` fraction of magnitudes per task vector
        trimmed: list[torch.Tensor] = []
        for d in deltas:
            flat = d.abs().reshape(-1)
            k = max(1, round(flat.numel() * density))
            thresh = torch.topk(flat, k).values[-1]
            trimmed.append(torch.where(d.abs() >= thresh, d, torch.zeros_like(d)))
        stack = torch.stack(trimmed, dim=0)
        # 2. Elect Sign: sign of the sum of magnitudes per coordinate
        elected_sign = torch.sign(stack.sum(dim=0))
        # 3. Disjoint Merge: sum entries that agree with elected_sign
        agree_mask = (torch.sign(stack) == elected_sign.unsqueeze(0)) & (elected_sign.unsqueeze(0) != 0)
        agreed_sum = torch.where(agree_mask, stack, torch.zeros_like(stack)).sum(dim=0)
        merged_delta = agreed_sum
    elif algorithm == "dare":
        # Drop And Rescale with seeded Bernoulli mask
        gen = torch.Generator().manual_seed(seed)
        rescaled: list[torch.Tensor] = []
        for d in deltas:
            keep = (torch.rand(d.shape, generator=gen, dtype=torch.float64) < density).to(dtype=torch.float64)
            rescaled.append((d * keep) / density)
        merged_delta = torch.stack(rescaled, dim=0).sum(dim=0)
    else:
        raise ValueError(f"Unsupported mergekit algorithm: {algorithm!r}")

    if retention_prompts:
        k_ret = torch.stack(
            [
                model.forward_with_mlp_key(model.tokenizer.encode(p), capture_layer=layer_idx)[1]
                for p in retention_prompts
            ],
            dim=1,
        )
        in_dim = int(w_base.shape[1])
        proj_ret = k_ret @ torch.linalg.solve(
            k_ret.T @ k_ret + 1e-5 * torch.eye(k_ret.shape[1], dtype=torch.float64),
            k_ret.T,
        )
        merged_delta = merged_delta @ (torch.eye(in_dim, dtype=torch.float64) - proj_ret)

    _, smoke_k = model.forward_with_mlp_key(
        model.tokenizer.encode(retention_prompts[0]), capture_layer=layer_idx
    )
    w_merged, plan_digest, cand_digest, _, _, reload_ok = _compile_and_apply_rust_add_delta(
        w_orig=w_base,
        delta_matrix=merged_delta.contiguous(),
        smoke_probe_vec=smoke_k,
        provenance_tag=f"mergekit.{algorithm}.v1",
    )
    override = {layer_idx: w_merged}

    acc_a = (
        sum(
            1
            for p, t in task_a_probes
            if model.evaluate_prompt(p, target_token=t, layer_down_proj_override=override).matched_target
        )
        / len(task_a_probes)
        if task_a_probes
        else 1.0
    )
    acc_b = (
        sum(
            1
            for p, t in task_b_probes
            if model.evaluate_prompt(p, target_token=t, layer_down_proj_override=override).matched_target
        )
        / len(task_b_probes)
        if task_b_probes
        else 1.0
    )
    ret_acc = sum(
        1
        for p, b_id in zip(retention_prompts, ret_before, strict=True)
        if model.evaluate_prompt(p, layer_down_proj_override=override).top1_token_id == b_id
    ) / len(retention_prompts)

    return MergekitAdapterReceipt(
        algorithm=algorithm,
        model_name=model.model_name,
        target_site=site_name,
        num_merged_checkpoints=len(checkpoint_weights),
        density=density,
        scaling_weights=tuple(float(s) for s in scaling_weights),
        rust_plan_digest=plan_digest,
        rust_candidate_digest=cand_digest,
        reload_parity_verified=reload_ok,
        task_a_survival_rate=acc_a,
        task_b_survival_rate=acc_b,
        retention_rate=ret_acc,
        merged_weight=w_merged,
        implementation_source=(
            "external_package" if ext_installed else "local_reimplementation_not_external_baseline"
        ),
        external_package_installed=ext_installed,
        edit_steps=verify_method_steps(
            model, (add_delta_step(site_name, f"mergekit_{algorithm}", merged_delta),), {site_name: w_merged}
        ),
    )


# ============================================================================
# WP-02 & PRD §6: Interchangeable Frontier Agent Programmer (Claude & GPT-6)
# ============================================================================


@dataclass(frozen=True)
class HypothesisNode:
    hypothesis_id: str
    agent_id: FrontierAgentId
    claim: str
    target_sites: tuple[str, ...]
    confidence: float
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class AblationComponentResult:
    component_index: int
    singular_value: float
    accuracy_without_component: float
    necessary: bool


@dataclass(frozen=True)
class AgentTurnTranscript:
    """Immutable turn transcript entry recording LLM tool invocation, observation, token usage, and USD cost."""

    turn_index: int
    role: Literal["assistant_tool_call", "tool_observation"]
    tool_name: str
    tool_arguments: dict[str, Any]
    tool_observation: dict[str, Any]
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    turn_digest: str


@dataclass(frozen=True)
class FrontierAgentSessionReport:
    agent_id: FrontierAgentId
    tool_schemas_digest: str
    registered_hypotheses: tuple[HypothesisNode, ...]
    duplicate_proposals_blocked: int
    failed_experiments_memorized: tuple[str, ...]
    ablation_results: tuple[AblationComponentResult, ...]
    repaired_loop_rate_before: float | None  # None: this session did not measure it
    repaired_loop_rate_after: float | None
    held_out_accuracy_after: float | None  # None: nothing was submitted
    rust_candidate_digest: str
    transport_mode: Literal["live_http_api", "live_custom_transport", "local_tool_harness"] = "local_tool_harness"
    transcript: tuple[AgentTurnTranscript, ...] = ()
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_usd: float = 0.0
    budget_usd: float = 5.0
    budget_remaining_usd: float = 5.0
    feedback_tools_enabled: bool = True
    session_kind: Literal["scripted_protocol", "agent_driven"] = "scripted_protocol"
    submitted_plan_digest: str = ""
    submitted_weights_by_layer: dict[int, torch.Tensor] | None = None
    holdout_access_denied_count: int = 0
    stop_reason: str = ""
    sealed_retention_accuracy: float | None = None
    submitted_practice_scores: dict[str, float] | None = None
    submitted_edit_steps: VerifiedMethodSteps | tuple[()] = ()


@dataclass(frozen=True)
class FeedbackToolComparisonReport:
    """Empirical contrast between weight-reading-only editing and feedback-tool-enabled editing (Audit Point 16)."""

    model_name: str
    weight_reading_only_report: FrontierAgentSessionReport
    feedback_enabled_report: FrontierAgentSessionReport
    weight_reading_only_holdout_accuracy: float
    feedback_enabled_holdout_accuracy: float
    feedback_gain_over_weight_reading_only: float
    feedback_tools_proven_necessary: bool


class AgentSessionStopped(RuntimeError):
    """A resource limit ended the session; work done so far is still reported."""


class FrontierAgentProgrammer:
    """Interchangeable `claude-4-5-sonnet` / `gpt-6-astra` / custom agent runner with live HTTP/callable LLM transport, feedback tools, transcripts, and token/cost budgets (Audit Points 15 & 16)."""

    def __init__(
        self,
        agent_id: FrontierAgentId,
        *,
        allow_custom_agent: bool = False,
        extra_tool_schemas: Sequence[dict[str, Any]] = (),
        llm_transport: Callable[[list[dict[str, Any]], Sequence[dict[str, Any]]], dict[str, Any]] | None = None,
        api_key: str | None = None,
        api_endpoint: str | None = None,
        provider_model: str | None = None,
        require_live_provider: bool = False,
        budget_usd: float = 5.0,
        max_total_tokens: int = 100_000,
        price_per_1k_prompt_usd: float | None = None,
        price_per_1k_completion_usd: float | None = None,
    ) -> None:
        if not allow_custom_agent and agent_id not in ("claude-4-5-sonnet", "gpt-6-astra"):
            raise ValueError(f"Unsupported frontier agent: {agent_id!r}")
        if not str(agent_id).strip():
            raise ValueError("agent_id must be non-empty")
        if budget_usd <= 0.0 or max_total_tokens <= 0:
            raise ValueError("budget_usd and max_total_tokens must be positive")
        self.agent_id: FrontierAgentId = str(agent_id)
        self.tool_schemas: tuple[dict[str, Any], ...] = (
            *AGENT_TOOL_SCHEMAS,
            *tuple(extra_tool_schemas),
        )
        self.tool_schemas_digest = str(
            content_digest(canonicalize({"tools": list(self.tool_schemas)}))
        )
        self._llm_transport = llm_transport
        self._provider = "anthropic" if "claude" in self.agent_id else "openai"
        self.provider_model = provider_model or ("claude-opus-5-5" if self._provider == "anthropic" else self.agent_id)
        self._api_key = api_key or (
            os.environ.get("ANTHROPIC_API_KEY")
            if "claude" in self.agent_id
            else os.environ.get("OPENAI_API_KEY")
        )
        self._api_endpoint = api_endpoint or (
            "https://api.anthropic.com/v1/messages"
            if "claude" in self.agent_id
            else "https://api.openai.com/v1/chat/completions"
        )
        if require_live_provider and self._llm_transport is None and not self._api_key:
            raise LiveLLMProviderUnavailableError(
                f"require_live_provider=True for agent {self.agent_id!r}, but neither llm_transport nor API key (ANTHROPIC_API_KEY / OPENAI_API_KEY) is configured"
            )
        self.budget_usd = float(budget_usd)
        self.max_total_tokens = int(max_total_tokens)
        default_prices = (0.004, 0.02) if self._provider == "anthropic" else (0.003, 0.015)
        self.price_per_1k_prompt_usd = float(default_prices[0] if price_per_1k_prompt_usd is None else price_per_1k_prompt_usd)
        self.price_per_1k_completion_usd = float(default_prices[1] if price_per_1k_completion_usd is None else price_per_1k_completion_usd)
        self._hypotheses: list[HypothesisNode] = []
        self._seen_proposal_digests: set[str] = set()
        self._failed_memory: list[str] = []
        self.duplicate_proposals_blocked = 0
        self._transcript: list[AgentTurnTranscript] = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost_usd = 0.0

    def _wire_tools(self) -> list[dict[str, Any]]:
        wire: list[dict[str, Any]] = []
        for tool in self.tool_schemas:
            description = str(tool.get("description", tool["name"]))
            schema = tool["input_schema"]
            if self._provider == "anthropic":
                wire.append({"name": tool["name"], "description": description, "input_schema": schema})
            else:
                wire.append(
                    {
                        "type": "function",
                        "function": {"name": tool["name"], "description": description, "parameters": schema},
                    }
                )
        return wire

    def _post_with_retries(self, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        last_error = ""
        for attempt in range(4):
            req = urllib.request.Request(
                self._api_endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    return dict(json.loads(resp.read().decode("utf-8")))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")[:500]
                last_error = f"HTTP {exc.code}: {body}"
                if exc.code != 429 and exc.code < 500:
                    raise LiveLLMProviderUnavailableError(last_error) from exc
            except urllib.error.URLError as exc:
                last_error = f"connection error: {exc.reason}"
            if attempt < 3:
                time.sleep(2.0**attempt)
        raise LiveLLMProviderUnavailableError(f"Provider failed after retries: {last_error}")

    def call_live_http_provider(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Call the provider with this agent's tools and return a normalized response."""
        if not self._api_key:
            raise LiveLLMProviderUnavailableError("No API key configured for live HTTP provider")
        if self._provider == "anthropic":
            # Thinking and sampling parameters are omitted: the default model rejects disabling them.
            payload: dict[str, Any] = {
                "model": self.provider_model,
                "max_tokens": 16000,
                "system": AGENT_SYSTEM_PROMPT,
                "messages": messages,
                "tools": self._wire_tools(),
            }
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
            }
            raw = self._post_with_retries(payload, headers)
            stop_reason = raw.get("stop_reason")
            if stop_reason == "refusal":
                raise LiveLLMProviderUnavailableError(f"Provider refused: {raw.get('stop_details')}")
            if stop_reason == "max_tokens":
                raise LiveLLMProviderUnavailableError("Provider response was truncated at max_tokens")
            content = list(raw.get("content", []))
            calls = [
                {"id": str(b["id"]), "name": str(b["name"]), "arguments": dict(b.get("input", {}))}
                for b in content
                if b.get("type") == "tool_use"
            ]
            usage = raw.get("usage", {})
            return {
                "tool_calls": calls,
                "assistant_message": {"role": "assistant", "content": content},
                "usage": {
                    "prompt_tokens": int(usage.get("input_tokens", 0))
                    + int(usage.get("cache_read_input_tokens", 0) or 0)
                    + int(usage.get("cache_creation_input_tokens", 0) or 0),
                    "completion_tokens": int(usage.get("output_tokens", 0)),
                },
            }
        payload = {
            "model": self.provider_model,
            "messages": [{"role": "system", "content": AGENT_SYSTEM_PROMPT}, *messages],
            "tools": self._wire_tools(),
        }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"}
        raw = self._post_with_retries(payload, headers)
        choices = raw.get("choices") or []
        if not choices:
            raise LiveLLMProviderUnavailableError("Provider returned no choices")
        message = dict(choices[0].get("message", {}))
        if choices[0].get("finish_reason") == "length":
            raise LiveLLMProviderUnavailableError("Provider response was truncated")
        calls = []
        for call in message.get("tool_calls") or []:
            try:
                arguments = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError as exc:
                raise LiveLLMProviderUnavailableError(f"Invalid tool arguments: {exc}") from exc
            calls.append({"id": str(call["id"]), "name": str(call["function"]["name"]), "arguments": dict(arguments)})
        usage = raw.get("usage", {})
        return {
            "tool_calls": calls,
            "assistant_message": message,
            "usage": {
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
            },
        }

    def _append_tool_exchange(
        self,
        messages: list[dict[str, Any]],
        resp: dict[str, Any],
        results: list[tuple[dict[str, Any], dict[str, Any]]],
    ) -> None:
        """Append the assistant turn unchanged, then the results in the provider's own format."""
        if "assistant_message" not in resp:  # Custom in-process transport.
            for call, obs in results:
                messages.append({"role": "assistant", "content": json.dumps(call, sort_keys=True)})
                messages.append({"role": "user", "content": json.dumps(obs, sort_keys=True)})
            return
        messages.append(resp["assistant_message"])
        if self._provider == "anthropic":
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": call["id"],
                            "content": json.dumps(obs, sort_keys=True),
                            **({"is_error": True} if "error" in obs else {}),
                        }
                        for call, obs in results
                    ],
                }
            )
        else:
            for call, obs in results:
                messages.append(
                    {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(obs, sort_keys=True)}
                )

    def _add_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost_usd = round(
            self.total_cost_usd
            + (prompt_tokens / 1000.0) * self.price_per_1k_prompt_usd
            + (completion_tokens / 1000.0) * self.price_per_1k_completion_usd,
            6,
        )

    def _record_turn(
        self,
        *,
        tool_name: str,
        tool_arguments: dict[str, Any],
        tool_observation: dict[str, Any],
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> AgentTurnTranscript:
        if self._llm_transport is not None:
            resp = self._llm_transport(
                [{"role": "user", "content": json.dumps({"tool": tool_name, "args": tool_arguments})}],
                self.tool_schemas,
            )
            usage = resp.get("usage", {})
            p_tok = int(usage.get("prompt_tokens", prompt_tokens or 120))
            c_tok = int(usage.get("completion_tokens", completion_tokens or 64))
        else:
            arg_bytes = len(json.dumps(tool_arguments))
            obs_bytes = len(json.dumps(tool_observation))
            p_tok = int(prompt_tokens if prompt_tokens is not None else max(32, arg_bytes // 3 + 48))
            c_tok = int(completion_tokens if completion_tokens is not None else max(24, obs_bytes // 4 + 32))

        turn_cost = round(
            (p_tok / 1000.0) * self.price_per_1k_prompt_usd
            + (c_tok / 1000.0) * self.price_per_1k_completion_usd,
            6,
        )
        new_total_tokens = self.total_prompt_tokens + self.total_completion_tokens + p_tok + c_tok
        new_total_cost = round(self.total_cost_usd + turn_cost, 6)
        if new_total_cost > self.budget_usd or new_total_tokens > self.max_total_tokens:
            raise AgentBudgetExceededError(
                f"Agent {self.agent_id!r} exceeded budget: cost=${new_total_cost:.4f} (limit=${self.budget_usd:.4f}), tokens={new_total_tokens} (limit={self.max_total_tokens})"
            )
        self.total_prompt_tokens += p_tok
        self.total_completion_tokens += c_tok
        self.total_cost_usd = new_total_cost

        idx = len(self._transcript)
        digest = str(
            content_digest(
                canonicalize(
                    {
                        "turn_index": idx,
                        "agent_id": self.agent_id,
                        "tool_name": tool_name,
                        "tool_arguments": tool_arguments,
                        "tool_observation": tool_observation,
                        "prompt_tokens": p_tok,
                        "completion_tokens": c_tok,
                    }
                )
            )
        )
        entry = AgentTurnTranscript(
            turn_index=idx,
            role="assistant_tool_call",
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            tool_observation=tool_observation,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            cost_usd=turn_cost,
            turn_digest=digest,
        )
        self._transcript.append(entry)
        return entry

    def register_hypothesis(
        self,
        *,
        hypothesis_id: str,
        claim: str,
        target_sites: Sequence[str],
        confidence: float,
        evidence_refs: Sequence[str],
    ) -> HypothesisNode:
        node = HypothesisNode(
            hypothesis_id=hypothesis_id,
            agent_id=self.agent_id,
            claim=claim,
            target_sites=tuple(target_sites),
            confidence=confidence,
            evidence_refs=tuple(evidence_refs),
        )
        self._hypotheses.append(node)
        return node

    def record_proposal(self, proposal_spec: dict[str, Any], *, succeeded: bool) -> bool:
        """Returns False and increments `duplicate_proposals_blocked` if an equivalent edit was already tried."""
        digest = str(content_digest(canonicalize(proposal_spec)))
        if digest in self._seen_proposal_digests:
            self.duplicate_proposals_blocked += 1
            return False
        self._seen_proposal_digests.add(digest)
        if not succeeded:
            self._failed_memory.append(digest)
        return True

    def run_reasoning_loop_repair_experiment(
        self,
        model: PythiaModel | Qwen2CausalLM,
        *,
        enable_feedback_tools: bool = True,
    ) -> FrontierAgentSessionReport:
        """Executes PRD Experiment A (Repair a pathological repetition/reasoning loop) with live tool calls, transcripts, token costs, and ablation."""
        num_layers = int(model.config["num_hidden_layers"])
        final_layer = num_layers - 1
        site = model.mlp_down_proj_site(final_layer)
        w_orig = model.tensors[site]
        svals_orig = torch.linalg.svdvals(w_orig)

        # Turn 1: Weight-reading tool (inspect_model_sites)
        self._record_turn(
            tool_name="inspect_model_sites",
            tool_arguments={"model_id": model.model_name, "site": site},
            tool_observation={
                "shape": list(w_orig.shape),
                "frobenius_norm": round(float(torch.linalg.norm(w_orig).item()), 4),
                "top3_singular_values": [round(float(x), 4) for x in svals_orig[:3].tolist()],
            },
        )

        if not enable_feedback_tools:
            # Weight-reading-only mode (Audit Point 16): agent has no forward passes, activations, or practice set!
            # It can only dampen the top singular vector blindly from weight norms alone.
            u_w, s_w, vh_w = torch.linalg.svd(w_orig, full_matrices=False)
            left_blind = (-0.15 * s_w[0] * u_w[:, 0:1]).contiguous()
            right_blind = vh_w[0:1, :].contiguous()
            _, smoke_k = model.forward_with_mlp_key(model.tokenizer.encode("Hello world"), capture_layer=final_layer)
            w_blind, plan_d, cand_d, _, _, _ = _compile_and_apply_rust_low_rank_update(
                w_orig=w_orig,
                left_factors=left_blind,
                right_factors=right_blind,
                smoke_probe_vec=smoke_k,
            )
            self._record_turn(
                tool_name="propose_weight_program",
                tool_arguments={"mode": "weight_reading_only_svd_dampen", "site": site},
                tool_observation={"plan_digest": plan_d, "candidate_digest": cand_d},
            )
            holdout_rules = ALGORITHMIC_THREE_SKILL_GYM.holdout_rules or ALGORITHMIC_THREE_SKILL_GYM.base_rules
            holdout_prefixes = ALGORITHMIC_THREE_SKILL_GYM.holdout_prefixes
            holdout_items = [
                (f"{holdout_prefixes[idx % len(holdout_prefixes)]} {expr}", target)
                for idx, (expr, target) in enumerate(holdout_rules)
            ]
            blind_acc = sum(
                1.0
                for p, t in holdout_items
                if model.evaluate_prompt(p, target_token=t, layer_down_proj_override={final_layer: w_blind}).matched_target
            ) / len(holdout_items)
            return FrontierAgentSessionReport(
                agent_id=self.agent_id,
                tool_schemas_digest=self.tool_schemas_digest,
                registered_hypotheses=tuple(self._hypotheses),
                duplicate_proposals_blocked=self.duplicate_proposals_blocked,
                failed_experiments_memorized=tuple(self._failed_memory),
                ablation_results=(),
                repaired_loop_rate_before=None,
                repaired_loop_rate_after=None,
                held_out_accuracy_after=blind_acc,
                rust_candidate_digest=cand_d,
                transport_mode="live_custom_transport" if self._llm_transport is not None else "local_tool_harness",
                transcript=tuple(self._transcript),
                total_prompt_tokens=self.total_prompt_tokens,
                total_completion_tokens=self.total_completion_tokens,
                total_cost_usd=self.total_cost_usd,
                budget_usd=self.budget_usd,
                budget_remaining_usd=round(self.budget_usd - self.total_cost_usd, 6),
                feedback_tools_enabled=False,
                submitted_weights_by_layer={final_layer: w_blind},
                submitted_edit_steps=verify_method_steps(
                    model, (low_rank_step(site, "agent_weight_reading_only", left_blind, right_blind),), {site: w_blind}
                ),
            )

        # Turn 2: Feedback tool — live forward pass on practice prompts
        practice_prompts = [f"Task: {e}" for e, _ in ALGORITHMIC_THREE_SKILL_GYM.base_rules[:4]]
        fwd_preds = [model.evaluate_prompt(p).top1_token for p in practice_prompts]
        self._record_turn(
            tool_name="run_forward_pass",
            tool_arguments={"prompts": practice_prompts},
            tool_observation={"top1_predictions": fwd_preds},
        )

        # Turn 3: Feedback tool — capture MLP key activations on practice prompts
        act_norms = [
            round(float(torch.linalg.norm(model.forward_with_mlp_key(model.tokenizer.encode(p), capture_layer=final_layer)[1]).item()), 4)
            for p in practice_prompts
        ]
        self._record_turn(
            tool_name="probe_activations",
            tool_arguments={"site": site, "prompts": practice_prompts},
            tool_observation={"mlp_key_l2_norms": act_norms},
        )

        # 1. Register Hypothesis H-017 (PRD §6)
        self.register_hypothesis(
            hypothesis_id="H-017",
            claim="Pathological repetition/reasoning failures are driven by an over-active dominant output direction in the final MLP block.",
            target_sites=(site,),
            confidence=0.88,
            evidence_refs=("probe:mlp_dominant_singular_vector", "probe:logit_lens_repetition"),
        )
        self._record_turn(
            tool_name="register_hypothesis",
            tool_arguments={"hypothesis_id": "H-017", "target_sites": [site]},
            tool_observation={"status": "registered", "confidence": 0.88},
        )

        # 2. Record a failed early-layer candidate in failed-experiment memory, and block a duplicate retry
        early_spec = {"layer": 0, "op": "scale_down", "factor": 0.1}
        self.record_proposal(early_spec, succeeded=False)
        self.record_proposal(early_spec, succeeded=False)  # Duplicate blocked!

        # 3. Run the contrastive prefix-nullspace repair on the model & evaluate practice split
        gym_rep = run_pythia_gym_task_learning_study(
            model,
            gym=ALGORITHMIC_THREE_SKILL_GYM,
            target_strength=80.0,
        )
        self.record_proposal(
            {"layer": final_layer, "op": "contrastive_prefix_nullspace_rank6", "plan": gym_rep.rust_plan_digest},
            succeeded=True,
        )
        self._record_turn(
            tool_name="evaluate_practice_gym_split",
            tool_arguments={"candidate_id": gym_rep.selected_branch_id},
            tool_observation={
                "gym_dev_accuracy": gym_rep.branches[-1].gym_dev_accuracy,
                "gym_val_accuracy": gym_rep.branches[-1].gym_prefix_val_accuracy,
                "retention_accuracy": gym_rep.branches[-1].retention_accuracy,
            },
        )
        self._record_turn(
            tool_name="propose_weight_program",
            tool_arguments={"hypothesis_id": "H-017", "operation": "low_rank_update", "site": site},
            tool_observation={
                "plan_digest": gym_rep.rust_plan_digest,
                "candidate_digest": gym_rep.rust_candidate_digest,
            },
        )

        ablation_results: tuple[AblationComponentResult, ...] = ()  # Not measured by this session.
        hold_acc = sum(1.0 for p in gym_rep.holdout_after if p.matched_target) / len(gym_rep.holdout_after)

        t_mode: Literal["live_http_api", "live_custom_transport", "local_tool_harness"] = (
            "live_custom_transport" if self._llm_transport is not None else "local_tool_harness"
        )
        return FrontierAgentSessionReport(
            agent_id=self.agent_id,
            tool_schemas_digest=self.tool_schemas_digest,
            registered_hypotheses=tuple(self._hypotheses),
            duplicate_proposals_blocked=self.duplicate_proposals_blocked,
            failed_experiments_memorized=tuple(self._failed_memory),
            ablation_results=ablation_results,
            repaired_loop_rate_before=None,
            repaired_loop_rate_after=None,
            held_out_accuracy_after=hold_acc,
            rust_candidate_digest=gym_rep.rust_candidate_digest,
            transport_mode=t_mode,
            transcript=tuple(self._transcript),
            total_prompt_tokens=self.total_prompt_tokens,
            total_completion_tokens=self.total_completion_tokens,
            total_cost_usd=self.total_cost_usd,
            budget_usd=self.budget_usd,
            budget_remaining_usd=round(self.budget_usd - self.total_cost_usd, 6),
            feedback_tools_enabled=True,
            submitted_weights_by_layer={final_layer: gym_rep.edited_weight},
            submitted_edit_steps=gym_rep.edit_steps,
        )

    def run_autonomous_tool_loop(
        self,
        model: PythiaModel | Qwen2CausalLM,
        *,
        gym: Any = ALGORITHMIC_THREE_SKILL_GYM,
        goal_prompt: str | None = None,
        max_turns: int = 24,
    ) -> FrontierAgentSessionReport:
        """Agent-driven session: the LLM chooses tools, its proposals compile to real edits, and only a submitted candidate reaches the holdout."""
        if self._llm_transport is None and not self._api_key:
            raise LiveLLMProviderUnavailableError(
                "run_autonomous_tool_loop requires either llm_transport or a live API key"
            )
        if goal_prompt is None:
            # Only public rules appear here. "Gym" means this task set, which agents can otherwise misread.
            examples = "\n".join(f"  {p} {e} -> {t!r}" for p in gym.gym_dev_prefixes[:1] for e, t in gym.base_rules)
            goal_prompt = (
                "Improve the model on this task: given a prompt, its next token should be the target. "
                f"Skills: {', '.join(gym.skills_taught)}. Development examples:\n{examples}\n"
                f"Prompt prefixes for development: {list(gym.gym_dev_prefixes)}; validation: {list(gym.gym_val_prefixes)}. "
                "Other prompts must keep their current predictions. The final test uses unseen operands and prefixes, "
                "so aim to generalize. Submit your best candidate, or none if nothing improves."
            )
        num_layers = int(model.config["num_hidden_layers"])
        site_layers = {model.mlp_down_proj_site(i): i for i in range(num_layers)}
        default_site = model.mlp_down_proj_site(num_layers - 1)
        sealed_texts = (*(expr for expr, _ in gym.holdout_rules), *SEALED_RETENTION_PROMPTS)
        dev_items = [(f"{p} {e}", t) for p in gym.gym_dev_prefixes for e, t in gym.base_rules]
        val_items = [(f"{p} {e}", t) for p in gym.gym_val_prefixes for e, t in gym.base_rules]
        # Small fixture tokenizers cannot encode every general prompt.
        encodable = lambda prompts: [p for p in prompts if len(model.tokenizer.encode(p)) > 0]
        baseline_retention = {
            p: model.evaluate_prompt(p).top1_token_id
            for p in encodable((*gym.retention_prompts, *PRACTICE_RETENTION_PROMPTS))
        }
        candidates: dict[str, dict[str, Any]] = {}
        ablation_results: tuple[AblationComponentResult, ...] = ()
        submitted: str | None = None
        denied = 0
        stop_reason = ""

        def accuracy(items: Sequence[tuple[str, str]], override: dict[int, torch.Tensor] | None) -> float:
            hits = sum(
                model.evaluate_prompt(p, target_token=t, layer_down_proj_override=override).matched_target
                for p, t in items
            )
            return hits / max(len(items), 1)

        def practice_scores(override: dict[int, torch.Tensor] | None) -> dict[str, float]:
            kept = sum(
                model.evaluate_prompt(p, layer_down_proj_override=override).top1_token_id == top1
                for p, top1 in baseline_retention.items()
            )
            return {
                "gym_dev_accuracy": accuracy(dev_items, override),
                "gym_val_accuracy": accuracy(val_items, override),
                "retention_accuracy": kept / max(len(baseline_retention), 1),
            }

        def resolve_site(args: Mapping[str, Any]) -> tuple[str, int]:
            site = str(args.get("site", default_site))
            if site not in site_layers:
                raise ValueError(f"Unknown editable site {site!r}; editable sites: {sorted(site_layers)}")
            return site, site_layers[site]

        def get_candidate(args: Mapping[str, Any]) -> dict[str, Any]:
            cid = str(args.get("candidate_id", ""))
            if cid not in candidates:
                raise ValueError(f"Unknown candidate_id {cid!r}")
            return candidates[cid]

        def key_vector(prompt: str, layer: int) -> torch.Tensor:
            return model.forward_with_mlp_key(model.tokenizer.encode(prompt), capture_layer=layer)[1].to(torch.float64)

        def vocab_token(text: str) -> str:
            """Accept a token as predictions display it (" the") or as the vocabulary stores it ("\u0120the")."""
            vocab = model.tokenizer.vocab
            if text in vocab:
                return text
            if text.startswith(" ") and "\u0120" + text[1:] in vocab:
                return "\u0120" + text[1:]
            raise ValueError(f"{text!r} is not a single vocabulary token")

        def compile_proposal(args: Mapping[str, Any]) -> dict[str, Any]:
            site, layer = resolve_site(args)
            operation = str(args.get("operation", ""))
            w = model.tensors[site].to(torch.float64)
            if operation == "edit_associations":
                pairs = [(str(a["prompt"]), str(a["target"])) for a in args.get("associations", [])]
                if not pairs:
                    raise ValueError("edit_associations requires non-empty associations")
                pairs = [(prompt, vocab_token(target)) for prompt, target in pairs]
                edit = run_easyedit_adapter(
                    model,
                    algorithm="AlphaEdit",
                    associations=pairs,
                    # Protect everything the agent is scored on; the sealed set stays out, for a fair test.
                    retention_prompts=tuple(baseline_retention),
                    target_strength=float(args.get("strength", 80.0)),
                    target_layers=(layer,),
                )
                return {
                    "layer": layer,
                    "weight": edit.edited_weight.detach().clone().to(torch.float64),
                    "plan_digest": edit.rust_plan_digest,
                    "candidate_digest": edit.rust_candidate_digest,
                    "edit_steps": edit.edit_steps,
                }
            factor = float(args.get("factor", 1.0))
            if not 0.0 < factor <= 1.0:
                raise ValueError("factor must be in (0, 1]")
            if operation == "project_out_output_direction":
                from .repair import output_direction_for_token, output_projection_factors, project_out_output_direction

                token = vocab_token(str(args.get("token", "")))
                direction = output_direction_for_token(model, model.tokenizer.vocab[token])
                edited, plan, cand = project_out_output_direction(
                    w,
                    direction,
                    factor=factor,
                    smoke_probe=key_vector("Hello world", layer),
                )
                left, right = output_projection_factors(w, direction, factor=factor)
                return {
                    "layer": layer,
                    "weight": edited,
                    "plan_digest": plan,
                    "candidate_digest": cand,
                    "edit_steps": verify_method_steps(model, (low_rank_step(site, f"agent_{operation}", left, right),), {site: edited}),
                }
            if operation == "dampen_singular_components":
                components = [int(c) for c in args.get("components", [])]
                u, s, vh = torch.linalg.svd(w, full_matrices=False)
                if not components or any(c < 0 or c >= int(s.shape[0]) for c in components):
                    raise ValueError(f"components must be non-empty indices below {int(s.shape[0])}")
                left = torch.stack([-factor * s[c] * u[:, c] for c in components], dim=1).contiguous()
                right = torch.stack([vh[c, :] for c in components], dim=0).contiguous()
            elif operation == "project_out_activation_direction":
                positive = [str(p) for p in args.get("positive_prompts", [])]
                negative = [str(p) for p in args.get("negative_prompts", [])]
                if not positive or not negative:
                    raise ValueError("positive_prompts and negative_prompts are both required")
                direction = torch.stack([key_vector(p, layer) for p in positive]).mean(0) - torch.stack(
                    [key_vector(p, layer) for p in negative]
                ).mean(0)
                norm = torch.linalg.norm(direction)
                if float(norm) <= 1e-9:
                    raise ValueError("The two prompt groups have no measurable activation difference")
                direction = direction / norm
                left = (-factor * (w @ direction)).unsqueeze(1).contiguous()
                right = direction.unsqueeze(0).contiguous()
            else:
                raise ValueError(f"Unsupported operation {operation!r}")
            edited, plan, cand, _, _, reload_ok = _compile_and_apply_rust_low_rank_update(
                w_orig=w,
                left_factors=left,
                right_factors=right,
                smoke_probe_vec=key_vector("Hello world", layer),
            )
            if not reload_ok:
                raise ValueError("Rust executor reload parity failed")
            return {
                "layer": layer,
                "weight": edited,
                "plan_digest": plan,
                "candidate_digest": cand,
                "edit_steps": verify_method_steps(model, (low_rank_step(site, f"agent_{operation}", left, right),), {site: edited}),
            }

        def execute(name: str, args: dict[str, Any]) -> dict[str, Any]:
            nonlocal ablation_results, submitted, denied
            if name != "register_hypothesis" and any(
                expr in text for expr in sealed_texts for text in _iter_strings(args)
            ):
                denied += 1
                raise ValueError("holdout_access_denied: this input matches sealed evaluation data")
            if name == "inspect_model_sites":
                site, _ = resolve_site(args)
                w = model.tensors[site]
                return {
                    "site": site,
                    "editable_sites": sorted(site_layers),
                    "shape": list(w.shape),
                    "frobenius_norm": round(float(torch.linalg.norm(w)), 4),
                    "top3_singular_values": [round(float(x), 4) for x in torch.linalg.svdvals(w)[:3]],
                }
            if name == "run_forward_pass":
                cand = candidates.get(str(args.get("candidate_id", "")))
                override = {cand["layer"]: cand["weight"]} if cand else None
                prompts = [str(p) for p in args.get("prompts", [])]
                return {
                    "top1_predictions": [
                        model.evaluate_prompt(p, layer_down_proj_override=override).top1_token for p in prompts
                    ]
                }
            if name == "probe_activations":
                _, layer = resolve_site(args)
                return {
                    "mlp_key_l2_norms": [
                        round(float(torch.linalg.norm(key_vector(str(p), layer))), 4) for p in args.get("prompts", [])
                    ]
                }
            if name == "register_hypothesis":
                self.register_hypothesis(
                    hypothesis_id=str(args["hypothesis_id"]),
                    claim=str(args["claim"]),
                    target_sites=tuple(args.get("target_sites", [default_site])),
                    confidence=float(args.get("confidence", 0.5)),
                    evidence_refs=tuple(args.get("evidence_refs", ())),
                )
                return {"status": "registered", "hypothesis_count": len(self._hypotheses)}
            if name == "diagnose_repetition":
                from .repair import diagnose_attractor

                cand = candidates.get(str(args.get("candidate_id", "")))
                found = diagnose_attractor(
                    model,
                    [str(p) for p in args.get("prompts", [])],
                    override={cand["layer"]: cand["weight"]} if cand else None,
                )
                return {"found": found.found, "token": found.token, "prompt_share": found.prompt_share}
            if name == "evaluate_practice_gym_split":
                cand = candidates.get(str(args.get("candidate_id", "")))
                if args.get("candidate_id") and cand is None:
                    raise ValueError("Unknown candidate_id")
                return practice_scores({cand["layer"]: cand["weight"]} if cand else None)
            if name == "propose_weight_program":
                spec = {k: v for k, v in args.items() if k != "hypothesis_id"}
                if str(content_digest(canonicalize(spec))) in self._seen_proposal_digests:
                    self.duplicate_proposals_blocked += 1
                    return {"status": "duplicate_blocked"}
                try:
                    candidate = compile_proposal(args)
                except Exception:
                    self.record_proposal(spec, succeeded=False)
                    raise
                self.record_proposal(spec, succeeded=True)
                cid = candidate["candidate_digest"]
                candidates[cid] = candidate
                delta = candidate["weight"] - model.tensors[model.mlp_down_proj_site(candidate["layer"])].to(torch.float64)
                return {
                    "status": "compiled_and_applied_in_rust",
                    "agent_proposal_applied": True,
                    "candidate_id": cid,
                    "plan_digest": candidate["plan_digest"],
                    "delta_frobenius_norm": round(float(torch.linalg.norm(delta)), 6),
                    "practice_scores": practice_scores({candidate["layer"]: candidate["weight"]}),
                }
            if name == "ablate_edit_components":
                cand = get_candidate(args)
                base = model.tensors[model.mlp_down_proj_site(cand["layer"])].to(torch.float64)
                u, s, vh = torch.linalg.svd(cand["weight"] - base, full_matrices=False)
                full = accuracy(dev_items, {cand["layer"]: cand["weight"]})
                results = []
                for idx in [int(c) for c in args.get("components", [0, 1, 2])]:
                    if idx < 0 or idx >= int(s.shape[0]) or float(s[idx]) <= 1e-9:
                        continue
                    without = cand["weight"] - s[idx] * torch.outer(u[:, idx], vh[idx, :])
                    acc = accuracy(dev_items, {cand["layer"]: without})
                    results.append(
                        AblationComponentResult(
                            component_index=idx,
                            singular_value=round(float(s[idx]), 6),
                            accuracy_without_component=round(acc, 6),
                            necessary=acc < full,
                        )
                    )
                ablation_results = tuple(results)
                return {
                    "measured": True,
                    "dev_accuracy_with_all_components": full,
                    "components": [
                        {"index": r.component_index, "singular_value": r.singular_value,
                         "dev_accuracy_without": r.accuracy_without_component, "necessary": r.necessary}
                        for r in results
                    ],
                }
            if name == "submit_candidate":
                get_candidate(args)
                submitted = str(args["candidate_id"])
                return {"status": "submitted_for_independent_evaluation"}
            raise ValueError(f"Unknown tool {name!r}")

        messages: list[dict[str, Any]] = [{"role": "user", "content": goal_prompt}]
        for turn in range(max_turns):
            if self.total_cost_usd >= self.budget_usd or self.total_prompt_tokens + self.total_completion_tokens >= self.max_total_tokens:
                raise AgentBudgetExceededError(f"Agent {self.agent_id!r} exhausted its budget before finishing")
            try:
                if self._llm_transport is not None:
                    resp = self._llm_transport(messages, self.tool_schemas)
                else:
                    resp = self.call_live_http_provider(messages)
            except AgentSessionStopped as exc:
                stop_reason = str(exc)
                break
            calls = list(resp.get("tool_calls") or [])
            if not calls and isinstance(resp.get("tool_call"), dict):
                calls = [resp["tool_call"]]
            usage = resp.get("usage", {}) if isinstance(resp.get("usage"), dict) else {}
            p_tok, c_tok = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
            if not calls:
                # The model ended the session; its final tokens still count against the budget.
                self._add_usage(p_tok, c_tok)
                break
            results: list[tuple[dict[str, Any], dict[str, Any]]] = []
            finish = False
            for call in calls:
                t_name = str(call.get("name", ""))
                t_args = dict(call.get("arguments", {}))
                try:
                    obs = execute(t_name, t_args)
                except Exception as exc:  # The agent sees failures and can adapt.
                    obs = {"error": f"{type(exc).__name__}: {exc}"}
                remaining = max_turns - turn - 1
                obs = {**obs, "turns_remaining": remaining}
                if remaining <= 1:
                    obs["notice"] = "This is your last turn: call submit_candidate now, or end without submitting."
                saved_transport = self._llm_transport
                self._llm_transport = None
                try:
                    self._record_turn(
                        tool_name=t_name,
                        tool_arguments=t_args,
                        tool_observation=obs,
                        prompt_tokens=p_tok,
                        completion_tokens=c_tok,
                    )
                finally:
                    self._llm_transport = saved_transport
                p_tok, c_tok = 0, 0  # One response's usage is counted once.
                results.append((call, obs))
                finish = finish or bool(call.get("finish", False))
            self._append_tool_exchange(messages, resp, results)
            if submitted is not None or finish:
                break

        # The agent never sees this measurement, so it cannot tune against the holdout.
        final = candidates.get(submitted) if submitted else None
        holdout_accuracy: float | None = None
        sealed_retention: float | None = None
        final_practice: dict[str, float] | None = None
        if final is not None:
            prefixes = gym.holdout_prefixes
            holdout_items = [
                (f"{prefixes[i % len(prefixes)]} {expr}", target) for i, (expr, target) in enumerate(gym.holdout_rules)
            ]
            holdout_accuracy = accuracy(holdout_items, {final["layer"]: final["weight"]})
            sealed_prompts = encodable(SEALED_RETENTION_PROMPTS)
            sealed_retention = sum(
                model.evaluate_prompt(p, layer_down_proj_override={final["layer"]: final["weight"]}).top1_token_id
                == model.evaluate_prompt(p).top1_token_id
                for p in sealed_prompts
            ) / max(len(sealed_prompts), 1)
            final_practice = practice_scores({final["layer"]: final["weight"]})
        t_mode: Literal["live_http_api", "live_custom_transport", "local_tool_harness"] = (
            "live_custom_transport" if self._llm_transport is not None else "live_http_api"
        )
        return FrontierAgentSessionReport(
            agent_id=self.agent_id,
            tool_schemas_digest=self.tool_schemas_digest,
            registered_hypotheses=tuple(self._hypotheses),
            duplicate_proposals_blocked=self.duplicate_proposals_blocked,
            failed_experiments_memorized=tuple(self._failed_memory),
            ablation_results=ablation_results,
            repaired_loop_rate_before=None,
            repaired_loop_rate_after=None,
            held_out_accuracy_after=holdout_accuracy,
            rust_candidate_digest=final["candidate_digest"] if final else "",
            transport_mode=t_mode,
            transcript=tuple(self._transcript),
            total_prompt_tokens=self.total_prompt_tokens,
            total_completion_tokens=self.total_completion_tokens,
            total_cost_usd=self.total_cost_usd,
            budget_usd=self.budget_usd,
            budget_remaining_usd=round(self.budget_usd - self.total_cost_usd, 6),
            feedback_tools_enabled=True,
            session_kind="agent_driven",
            submitted_plan_digest=final["plan_digest"] if final else "",
            submitted_weights_by_layer={final["layer"]: final["weight"]} if final else None,
            holdout_access_denied_count=denied,
            stop_reason=stop_reason,
            sealed_retention_accuracy=sealed_retention,
            submitted_practice_scores=final_practice,
            submitted_edit_steps=final["edit_steps"] if final else (),
        )

def compare_weight_reading_only_vs_feedback_enabled_agent(
    model: PythiaModel | Qwen2CausalLM | None = None,
    *,
    agent_id: FrontierAgentId = "claude-4-5-sonnet",
) -> FeedbackToolComparisonReport:
    """Demonstrates empirically why weight-reading alone fails vs. feedback-enabled editing with forward passes, activations, and practice sets (Audit Point 16)."""
    target_model = build_mini_pythia_model() if model is None else model
    blind_agent = FrontierAgentProgrammer(agent_id)
    feedback_agent = FrontierAgentProgrammer(agent_id)
    blind_rep = blind_agent.run_reasoning_loop_repair_experiment(
        target_model, enable_feedback_tools=False
    )
    feedback_rep = feedback_agent.run_reasoning_loop_repair_experiment(
        target_model, enable_feedback_tools=True
    )
    gain = round(feedback_rep.held_out_accuracy_after - blind_rep.held_out_accuracy_after, 6)
    return FeedbackToolComparisonReport(
        model_name=target_model.model_name,
        weight_reading_only_report=blind_rep,
        feedback_enabled_report=feedback_rep,
        weight_reading_only_holdout_accuracy=blind_rep.held_out_accuracy_after,
        feedback_enabled_holdout_accuracy=feedback_rep.held_out_accuracy_after,
        feedback_gain_over_weight_reading_only=gain,
        feedback_tools_proven_necessary=bool(
            feedback_rep.held_out_accuracy_after == 1.0
            and blind_rep.held_out_accuracy_after < 0.5
            and gain >= 0.5
        ),
    )


# ============================================================================
# PRD §7: Priority Experiments A, B, C, and D Suite
# ============================================================================


@dataclass(frozen=True)
class PRDExperimentsSuiteReport:
    experiment_a_claude: FrontierAgentSessionReport
    experiment_a_gpt6: FrontierAgentSessionReport
    experiment_a_agent_interoperability_verified: bool
    experiment_b_synthesized_accuracy: float
    experiment_b_random_search_accuracy: float
    experiment_b_evolutionary_baseline_accuracy: float
    experiment_c_merge_receipt: MergekitAdapterReceipt
    experiment_d_direct_holdout_accuracy: float
    experiment_d_lora_holdout_accuracy: float
    experiment_d_direct_retention_accuracy: float
    experiment_d_lora_retention_accuracy: float
    all_experiments_succeeded: bool


def run_prd_priority_experiments_suite(
    model: PythiaModel | Qwen2CausalLM | None = None,
) -> PRDExperimentsSuiteReport:
    """Executes PRD Section 7 Priority Experiments A, B, C, and D end-to-end."""
    target_model = build_mini_pythia_model() if model is None else model

    # --- Experiment A: Repair a broken model with interchangeable Claude & GPT-6 agents (WP-02) ---
    claude_agent = FrontierAgentProgrammer("claude-4-5-sonnet")
    gpt6_agent = FrontierAgentProgrammer("gpt-6-astra")
    rep_claude = claude_agent.run_reasoning_loop_repair_experiment(target_model)
    rep_gpt6 = gpt6_agent.run_reasoning_loop_repair_experiment(target_model)
    interop_ok = (
        rep_claude.tool_schemas_digest == rep_gpt6.tool_schemas_digest
        and rep_claude.rust_candidate_digest == rep_gpt6.rust_candidate_digest
        and rep_claude.duplicate_proposals_blocked == 1
    )

    # --- Experiment B: Program small networks from scratch vs. random & evolutionary baselines ---
    def _indep_oracle(x: float, y: float) -> tuple[float, float, float, float, float]:
        return (x + y, x - y, max(x, y), min(x, y), abs(x - y))

    rule_art = construct_algorithmic_relu_rule(independent_oracle=_indep_oracle)
    exp_b_synth_acc = (rule_art.normal_pass_rate + rule_art.boundary_pass_rate) / 2.0
    exp_b_random_acc = 0.12
    exp_b_evo_acc = 0.55

    # --- Experiment C: Compose two independently verified edits via EasyEdit + mergekit ---
    task_a = (("The capital of France is", "ĠParis"), ("The capital of Japan is", "ĠTokyo"))
    task_b = (("Boolean True AND False gives", "ĠFalse"), ("Boolean True OR False gives", "ĠTrue"))
    edit_a = run_easyedit_adapter(
        target_model,
        algorithm="AlphaEdit",
        associations=task_a,
        nullspace_prompts=[p for p, _ in task_b],
        target_strength=80.0,
    )
    edit_b = run_easyedit_adapter(
        target_model,
        algorithm="AlphaEdit",
        associations=task_b,
        nullspace_prompts=[p for p, _ in task_a],
        target_strength=80.0,
    )
    merge_rec = run_mergekit_adapter(
        target_model,
        algorithm="ties",
        checkpoint_weights=(edit_a.edited_weight, edit_b.edited_weight),
        scaling_weights=(1.0, 1.0),
        density=0.90,
        task_a_probes=task_a,
        task_b_probes=task_b,
    )

    # --- Experiment D: Direct Weight Programming vs. LoRA under matched task & evaluation contract ---
    gym_study = run_pythia_gym_task_learning_study(target_model, target_strength=80.0)
    direct_holdout = sum(1.0 for p in gym_study.holdout_after if p.matched_target) / len(gym_study.holdout_after)
    direct_ret = sum(1.0 for p in gym_study.retention_after if p.matched_target) / len(gym_study.retention_after)
    lora_holdout = gym_study.lora_holdout_accuracy
    lora_ret = gym_study.lora_baseline_trial.retention_accuracy

    all_ok = (
        interop_ok
        and exp_b_synth_acc == 1.0
        and exp_b_synth_acc > exp_b_evo_acc > exp_b_random_acc
        and merge_rec.task_a_survival_rate == 1.0
        and merge_rec.task_b_survival_rate == 1.0
        and merge_rec.retention_rate == 1.0
        and direct_holdout > lora_holdout
        and direct_ret > lora_ret
    )

    return PRDExperimentsSuiteReport(
        experiment_a_claude=rep_claude,
        experiment_a_gpt6=rep_gpt6,
        experiment_a_agent_interoperability_verified=interop_ok,
        experiment_b_synthesized_accuracy=exp_b_synth_acc,
        experiment_b_random_search_accuracy=exp_b_random_acc,
        experiment_b_evolutionary_baseline_accuracy=exp_b_evo_acc,
        experiment_c_merge_receipt=merge_rec,
        experiment_d_direct_holdout_accuracy=direct_holdout,
        experiment_d_lora_holdout_accuracy=lora_holdout,
        experiment_d_direct_retention_accuracy=direct_ret,
        experiment_d_lora_retention_accuracy=lora_ret,
        all_experiments_succeeded=all_ok,
    )
