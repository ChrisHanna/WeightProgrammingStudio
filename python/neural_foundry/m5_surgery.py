"""M5 Architecture Surgery, Quantization, Compression, Serving Gyms, and Systems Profiling (T102–T107, T110–T120).

Implements:
- T102 (`mlp_surgery`), T103 (`attention_surgery`), T104 (`layer_reorganization`),
  T105 (`structural_constructors`), T106 (`quantized_policy`), T107 (`compression_graph`).
- T110 (`quantization_gym`), T111 (`compression_gym`), T112 (`generation_gym`),
  T113 (`serving_protocol`), T114 (`vllm_adapter`), T115 (`tensorrt_adapter`),
  T116 (`systems_profile`), T117 (`kernel_admission`), T118 (`efficiency_benchmark`),
  T119 (`backend_qualification`).

Evidence: surgery functions slice tensors passed to them and do not produce a loadable model.
Gym evaluators score numbers the caller supplies; results carry an ``evidence`` field.
qualify_vllm_serving_adapter runs real vLLM and raises if it can't.
qualify_tensorrt_llm_adapter, run_end_to_end_systems_profile and
run_weight_efficiency_benchmark are not implemented and raise NotImplementedError once their
contract checks pass.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch

from .m4_modalities import EvidenceBasis
from .records import canonicalize, content_digest


def _digest(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    try:
        return str(content_digest(canonicalize(raw)))
    except (ValueError, TypeError, RuntimeError):
        return f"sha256:{hashlib.sha256(raw).hexdigest()}"


class M5SurgeryError(RuntimeError):
    """Structured error raised when an M5 surgery, quantization, or efficiency contract is violated."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code


# ============================================================================
# T102: MLP Channel and Intermediate-Width Surgery
# ============================================================================


@dataclass(frozen=True)
class MlpSurgeryResult:
    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    down_proj: torch.Tensor
    updated_intermediate_size: int
    removed_parameters: int
    # None: nothing is written to disk and reloaded.
    reload_parity_verified: None
    preservation_class: Literal["experimental"]


def execute_mlp_width_surgery(
    *,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    keep_indices: list[int],
    slice_down_proj_columns: bool = True,
) -> MlpSurgeryResult:
    """Slices coupled SwiGLU gate/up rows and down columns and updates intermediate_size (T102)."""
    if not slice_down_proj_columns:
        raise M5SurgeryError(
            "UNCOUPLED_MLP_SURGERY",
            "slicing gate_proj/up_proj without coupled down_proj columns leaves an invalid architecture graph",
        )
    inter_dim, hidden_dim = int(gate_proj.shape[0]), int(gate_proj.shape[1])
    if tuple(up_proj.shape) != (inter_dim, hidden_dim) or tuple(down_proj.shape) != (hidden_dim, inter_dim):
        raise M5SurgeryError("MLP_SHAPE_MISMATCH", "gate_proj, up_proj, and down_proj shapes are inconsistent")
    if not keep_indices or len(set(keep_indices)) != len(keep_indices):
        raise M5SurgeryError("INVALID_KEEP_INDICES", "keep_indices must be non-empty and unique")
    if any(i < 0 or i >= inter_dim for i in keep_indices):
        raise M5SurgeryError("INDEX_OUT_OF_RANGE", "keep_indices out of range for intermediate_size")

    idx = torch.tensor(keep_indices, dtype=torch.long)
    new_gate = gate_proj.detach()[idx, :].contiguous()
    new_up = up_proj.detach()[idx, :].contiguous()
    new_down = down_proj.detach()[:, idx].contiguous()
    new_inter = len(keep_indices)
    removed = 3 * (inter_dim - new_inter) * hidden_dim

    return MlpSurgeryResult(
        gate_proj=new_gate,
        up_proj=new_up,
        down_proj=new_down,
        updated_intermediate_size=new_inter,
        removed_parameters=removed,
        reload_parity_verified=None,
        preservation_class="experimental",
    )


# ============================================================================
# T103: Attention-Head Surgery
# ============================================================================


@dataclass(frozen=True)
class AttentionSurgeryResult:
    q_proj: torch.Tensor
    k_proj: torch.Tensor
    v_proj: torch.Tensor
    o_proj: torch.Tensor
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    kv_group_ratio: int
    rope_dim_preserved: bool


def execute_attention_head_surgery(
    *,
    q_proj: torch.Tensor,
    k_proj: torch.Tensor,
    v_proj: torch.Tensor,
    o_proj: torch.Tensor,
    keep_q_heads: list[int],
    keep_kv_heads: list[int],
    head_dim: int,
    change_head_dim_without_rope_recalibration: bool = False,
) -> AttentionSurgeryResult:
    """Prunes Q/K/V/O attention heads while enforcing GQA group divisibility and RoPE head_dim invariants (T103)."""
    if change_head_dim_without_rope_recalibration:
        raise M5SurgeryError(
            "UNSAFE_ROPE_LAYOUT_CHANGE",
            "changing head_dim invalidates rotary embedding frequencies and KV cache layout",
        )
    if not keep_q_heads or not keep_kv_heads or len(keep_q_heads) % len(keep_kv_heads) != 0:
        raise M5SurgeryError(
            "INVALID_GQA_GROUPING",
            f"retained Q heads ({len(keep_q_heads)}) must be positive and divisible by retained KV heads ({len(keep_kv_heads)})",
        )
    q_rows: list[int] = []
    for h in keep_q_heads:
        q_rows.extend(range(h * head_dim, (h + 1) * head_dim))
    kv_rows: list[int] = []
    for h in keep_kv_heads:
        kv_rows.extend(range(h * head_dim, (h + 1) * head_dim))

    q_idx = torch.tensor(q_rows, dtype=torch.long)
    kv_idx = torch.tensor(kv_rows, dtype=torch.long)

    return AttentionSurgeryResult(
        q_proj=q_proj.detach()[q_idx, :].contiguous(),
        k_proj=k_proj.detach()[kv_idx, :].contiguous(),
        v_proj=v_proj.detach()[kv_idx, :].contiguous(),
        o_proj=o_proj.detach()[:, q_idx].contiguous(),
        num_attention_heads=len(keep_q_heads),
        num_key_value_heads=len(keep_kv_heads),
        head_dim=head_dim,
        kv_group_ratio=len(keep_q_heads) // len(keep_kv_heads),
        rope_dim_preserved=True,
    )


# ============================================================================
# T104: Layer Removal and Reorganization
# ============================================================================


@dataclass(frozen=True)
class LayerReorganizationResult:
    layer_weights: dict[str, torch.Tensor]
    num_hidden_layers: int
    kv_cache_layers: int
    preservation_class: Literal["experimental"]
    behavioral_evaluation_required: bool


def execute_layer_reorganization(
    layers: list[torch.Tensor],
    *,
    new_layer_order: list[int],
    claim_function_preserving: bool = False,
    leave_stale_layer_references: bool = False,
) -> LayerReorganizationResult:
    """Removes or reorders transformer layers atomically and enforces experimental classification (T104)."""
    if claim_function_preserving:
        raise M5SurgeryError(
            "FALSE_PRESERVATION_CLAIM",
            "removing or reordering learned layers is experimental and cannot claim function preservation",
        )
    if leave_stale_layer_references:
        raise M5SurgeryError("STALE_LAYER_REFERENCES", "layer removal/reordering must reindex all layer keys atomically")
    if not new_layer_order or any(i < 0 or i >= len(layers) for i in new_layer_order):
        raise M5SurgeryError("INVALID_LAYER_ORDER", "new_layer_order must reference valid existing layers")

    reindexed = {
        f"model.layers.{new_idx}.weight": layers[old_idx].detach().clone()
        for new_idx, old_idx in enumerate(new_layer_order)
    }
    return LayerReorganizationResult(
        layer_weights=reindexed,
        num_hidden_layers=len(new_layer_order),
        kv_cache_layers=len(new_layer_order),
        preservation_class="experimental",
        behavioral_evaluation_required=True,
    )


# ============================================================================
# T105: Registered Structural Constructors
# ============================================================================


@dataclass(frozen=True)
class StructuralConstructorReceipt:
    constructor_id: str
    max_parity_abs_diff: float
    exact_under_conditions: bool
    preconditions_checked: bool
    # The identity is evaluated on the supplied matrix; no model graph is changed.
    structural_graph_modified: Literal[False] = False
    evidence: EvidenceBasis = "measured_on_fixture"


def apply_registered_structural_constructor(
    base_weight: torch.Tensor,
    probe_inputs: torch.Tensor,
    *,
    constructor_id: Literal["split_linear_channel_exact.v1", "insert_zero_residual_block.v1"],
    violate_zero_init_precondition: bool = False,
    request_unsupported_vocab_resize: bool = False,
) -> StructuralConstructorReceipt:
    """Applies a registered algebraic identity constructor and verifies fixture parity (T105)."""
    if request_unsupported_vocab_resize:
        raise M5SurgeryError(
            "UNSUPPORTED_VOCAB_CONSTRUCTOR",
            "vocabulary/width neutrality constructors without exact algebraic proof are unsupported",
        )
    if violate_zero_init_precondition:
        raise M5SurgeryError(
            "CONSTRUCTOR_PRECONDITION_VIOLATED",
            "identity-preserving structural constructor precondition violated (non-zero residual init)",
        )
    w = base_weight.detach().to(dtype=torch.float64)
    x = probe_inputs.detach().to(dtype=torch.float64)
    orig_out = x @ w.T

    if constructor_id == "split_linear_channel_exact.v1":
        # Split row 0 into two half-scaled rows and sum them back
        w_half = w * 0.5
        reconstructed_out = (x @ w_half.T) + (x @ w_half.T)
    else:
        zero_res = torch.zeros_like(w)
        reconstructed_out = (x @ w.T) + (x @ zero_res.T)

    max_diff = float(torch.max(torch.abs(orig_out - reconstructed_out)).item())
    return StructuralConstructorReceipt(
        constructor_id=constructor_id,
        max_parity_abs_diff=max_diff,
        exact_under_conditions=(max_diff < 1e-12),
        preconditions_checked=True,
    )


# ============================================================================
# T106 & T107: Quantized Representation Policy and Compression Graph
# ============================================================================


@dataclass(frozen=True)
class QuantizedTensorArtifact:
    format_version: str
    group_size: int
    bits: Literal[4, 8]
    qweight: torch.Tensor
    scales: torch.Tensor
    zero_points: torch.Tensor
    calibration_digest: str | None
    max_roundtrip_abs_error: float


def quantize_weight_grouped(
    weight: torch.Tensor,
    *,
    group_size: int = 4,
    bits: Literal[4, 8] = 8,
    calibration_digest: str | None = None,
    mutate_packed_without_scales: bool = False,
    allow_implicit_float_dequant_write: bool = False,
) -> QuantizedTensorArtifact:
    """Quantizes a 2D weight tensor with explicit group scales/zero-points and blocks uncoupled packed edits (T106)."""
    if mutate_packed_without_scales:
        raise M5SurgeryError(
            "UNCOUPLED_PACKED_EDIT_FORBIDDEN",
            "cannot mutate packed quantized weights without updating coupled group scales",
        )
    if allow_implicit_float_dequant_write:
        raise M5SurgeryError(
            "IMPLICIT_DEQUANTIZATION_FORBIDDEN",
            "implicit float dequantization during quantized weight edit is forbidden",
        )
    w = weight.detach().to(dtype=torch.float64)
    if w.ndim != 2 or int(w.shape[1]) % group_size != 0:
        raise M5SurgeryError("INVALID_QUANT_GROUP_SIZE", "columns must be divisible by group_size")

    qmax = float((1 << (bits - 1)) - 1)
    reshaped = w.reshape(w.shape[0], -1, group_size)
    max_abs = torch.amax(torch.abs(reshaped), dim=-1, keepdim=True).clamp(min=1e-8)
    scales = max_abs / qmax
    q = torch.round(reshaped / scales).clamp(-qmax, qmax)
    dequant = (q * scales).reshape_as(w)
    err = float(torch.max(torch.abs(w - dequant)).item())

    return QuantizedTensorArtifact(
        format_version="nf.group_affine.v1",
        group_size=group_size,
        bits=bits,
        qweight=q.to(dtype=torch.int8).reshape_as(w),
        scales=scales.squeeze(-1),
        zero_points=torch.zeros_like(scales.squeeze(-1), dtype=torch.int8),
        calibration_digest=calibration_digest,
        max_roundtrip_abs_error=err,
    )


@dataclass(frozen=True)
class CompressionGraphAccounting:
    modality: Literal["unstructured_zero_mask", "structural_channel_removal"]
    dense_parameters_before: int
    executable_parameters_after: int
    dense_flops_before: int
    executable_flops_after: int
    removed_flops: int
    # FLOPs are 2 * parameters, not a count from a running model.
    evidence: EvidenceBasis = "formula_estimate"


def account_compression_graph_effects(
    *,
    modality: Literal["unstructured_zero_mask", "structural_channel_removal"],
    dense_parameters_before: int,
    zeroed_or_removed_count: int,
    claim_flop_reduction_from_unstructured_zeros: bool = False,
) -> CompressionGraphAccounting:
    """Separates unstructured zero masking (0 FLOPs removed on dense kernels) from structural removal (T107)."""
    if modality == "unstructured_zero_mask" and claim_flop_reduction_from_unstructured_zeros:
        raise M5SurgeryError(
            "ZERO_MASK_NOT_FLOP_REDUCTION",
            "setting weights to zero in a dense tensor does not remove parameters or FLOPs (T107)",
        )
    flops_before = dense_parameters_before * 2
    if modality == "unstructured_zero_mask":
        return CompressionGraphAccounting(
            modality=modality,
            dense_parameters_before=dense_parameters_before,
            executable_parameters_after=dense_parameters_before,
            dense_flops_before=flops_before,
            executable_flops_after=flops_before,
            removed_flops=0,
        )
    params_after = dense_parameters_before - zeroed_or_removed_count
    flops_after = params_after * 2
    return CompressionGraphAccounting(
        modality=modality,
        dense_parameters_before=dense_parameters_before,
        executable_parameters_after=params_after,
        dense_flops_before=flops_before,
        executable_flops_after=flops_after,
        removed_flops=flops_before - flops_after,
    )


# ============================================================================
# T110–T119: Efficiency Gyms, Serving Adapters, Profiling, Kernel Admission, Qualification
# ============================================================================


@dataclass(frozen=True)
class EfficiencyGymVerdict:
    gym_kind: Literal["quantization_gym", "compression_gym", "generation_gym", "serving_gym"]
    # "verified" means the supplied numbers cleared the thresholds; see evidence. None: not evaluated.
    retained_quality_ok: bool | None
    speed_or_memory_gain_verified: bool | None
    details: dict[str, Any]
    evidence: EvidenceBasis = "caller_supplied"


def evaluate_quantization_gym(
    *,
    quality_delta: float,
    real_lowbit_runtime_measured: bool,
    memory_ratio: float,
    throughput_ratio: float,
    claim_speedup_from_float_dequant_simulation: bool = False,
) -> EfficiencyGymVerdict:
    """Evaluates Quantization Gym; simulated float dequantization cannot claim runtime speedup (T110)."""
    if claim_speedup_from_float_dequant_simulation or not real_lowbit_runtime_measured:
        raise M5SurgeryError(
            "SIMULATED_DEQUANT_NOT_SPEED_EVIDENCE",
            "reference float dequantization cannot substitute for native low-bit runtime speed evidence (T110)",
        )
    q_ok = quality_delta >= -0.01
    gain_ok = q_ok and (memory_ratio < 0.75 or throughput_ratio > 1.05)
    return EfficiencyGymVerdict(
        gym_kind="quantization_gym",
        retained_quality_ok=q_ok,
        speed_or_memory_gain_verified=gain_ok,
        details={"memory_ratio": memory_ratio, "throughput_ratio": throughput_ratio},
    )


def evaluate_compression_gym(
    *,
    quality_delta: float,
    max_allowed_regression: float = 0.01,
    param_ratio: float,
    latency_ratio: float,
    ignore_quality_regression_for_smaller_model: bool = False,
) -> EfficiencyGymVerdict:
    """Evaluates Compression Gym; rejects quality regressions hidden by smaller parameter counts (T111)."""
    if ignore_quality_regression_for_smaller_model:
        raise M5SurgeryError(
            "QUALITY_REGRESSION_HIDDEN_BY_COMPRESSION",
            "smaller parameter count cannot override a retention quality gate failure (T111)",
        )
    q_ok = quality_delta >= -max_allowed_regression
    gain_ok = q_ok and param_ratio < 0.90 and latency_ratio < 0.95
    return EfficiencyGymVerdict(
        gym_kind="compression_gym",
        retained_quality_ok=q_ok,
        speed_or_memory_gain_verified=gain_ok,
        details={"param_ratio": param_ratio, "latency_ratio": latency_ratio},
    )


def evaluate_generation_gym(
    *,
    baseline_accuracy: float,
    candidate_accuracy: float,
    baseline_tokens: int,
    candidate_tokens: int,
    stopping_rules_matched: bool = True,
    truncated_failed_answers_counted_as_efficient: bool = False,
) -> EfficiencyGymVerdict:
    """Evaluates Generation Gym; shorter failed/truncated answers never qualify as reasoning efficiency (T112)."""
    if not stopping_rules_matched:
        raise M5SurgeryError("MISMATCHED_STOPPING_RULES", "generation gym requires matched decoding and stop rules")
    if truncated_failed_answers_counted_as_efficient or candidate_accuracy < baseline_accuracy:
        raise M5SurgeryError(
            "SHORT_FAILED_ANSWER_NOT_EFFICIENT",
            "shorter answers that lose accuracy or truncate early cannot qualify as generation efficiency (T112)",
        )
    gain_ok = candidate_tokens < baseline_tokens and candidate_accuracy >= baseline_accuracy
    return EfficiencyGymVerdict(
        gym_kind="generation_gym",
        retained_quality_ok=True,
        speed_or_memory_gain_verified=gain_ok,
        details={"token_ratio": candidate_tokens / max(baseline_tokens, 1)},
    )


def run_serving_protocol_pytorch(
    weight: torch.Tensor,
    prompt_batch: torch.Tensor,
    *,
    is_mock_trace: bool = False,
    include_tail_latencies: bool = True,
) -> EfficiencyGymVerdict:
    """Times one matrix multiply with a ReLU, cold and warm. Not a serving run: no TTFT or TPOT (T113)."""
    if is_mock_trace:
        raise M5SurgeryError("MOCK_SERVING_TRACE_REJECTED", "mocked serving traces are rejected")
    if not include_tail_latencies:
        raise M5SurgeryError("MISSING_SERVING_METRICS", "TTFT, TPOT, throughput, and p50/p95/p99 tail metrics required")

    w = weight.detach().to(dtype=torch.float64)
    x = prompt_batch.detach().to(dtype=torch.float64)
    t_cold0 = time.perf_counter_ns()
    _ = torch.relu(x @ w.T)
    cold_ns = max(1, time.perf_counter_ns() - t_cold0)

    warm_samples: list[int] = []
    for _ in range(5):
        t0 = time.perf_counter_ns()
        _ = torch.relu(x @ w.T)
        warm_samples.append(max(1, time.perf_counter_ns() - t0))
    warm_samples.sort()

    return EfficiencyGymVerdict(
        gym_kind="serving_gym",
        retained_quality_ok=None,
        speed_or_memory_gain_verified=None,
        details={
            "cold_start_ns": cold_ns,
            "warm_p50_ns": warm_samples[len(warm_samples) // 2],
            "warm_max_of_5_ns": warm_samples[-1],
        },
        evidence="measured_on_fixture",
    )


@dataclass(frozen=True)
class ServingAdapterQualification:
    adapter_id: Literal["vllm", "tensorrt_llm"]
    pinned_version: str
    loopback_bound: bool
    compiled_engine_digest: str
    runtime_parity_verified: bool
    measurements: dict[str, Any] = field(default_factory=dict)


def qualify_vllm_serving_adapter(
    *,
    pinned_version: str = "0.33.0",
    bind_host: str = "127.0.0.1",
    authorized: bool = True,
    is_mocked_transport_only: bool = False,
    checkpoint_dir: str | Path = "/tmp/pythia-14m",
    vllm_python: str | Path | None = None,
    gpu_memory_fraction: float = 0.05,
) -> ServingAdapterQualification:
    """Serve exported base and edited checkpoints through real vLLM and compare with transformers (T114).

    Raises M5SurgeryError if vLLM or the GPU is absent or any check fails. See vllm_qualification.py.
    The scope is the checkpoint and version checked, in float32 on the GPU.
    """
    if not authorized:
        raise M5SurgeryError("VLLM_UNAUTHORIZED", "vLLM launch requires explicit StudyContract authorization")
    if bind_host != "127.0.0.1":
        raise M5SurgeryError("NON_LOOPBACK_BIND_FORBIDDEN", "vLLM server must bind exclusively to 127.0.0.1 loopback")
    if is_mocked_transport_only:
        raise M5SurgeryError("MOCK_VLLM_NOT_QUALIFIED", "mocked HTTP transport does not qualify vLLM backend")
    from . import vllm_qualification

    try:
        measured = vllm_qualification.run_qualification(
            pinned_version=pinned_version,
            checkpoint_dir=checkpoint_dir,
            vllm_python=vllm_python,
            gpu_memory_fraction=gpu_memory_fraction,
        )
    except vllm_qualification.VllmQualificationFailure as exc:
        raise M5SurgeryError(exc.code, str(exc)) from exc
    return ServingAdapterQualification(
        adapter_id="vllm",
        pinned_version=pinned_version,
        loopback_bound=True,
        compiled_engine_digest="not_applicable",
        runtime_parity_verified=True,
        measurements=measured,
    )


def qualify_tensorrt_llm_adapter(
    *,
    pinned_version: str = "0.15.0",
    engine_built: bool = True,
    fresh_runtime_parity_checked: bool = True,
    claim_verified_from_float_export_alone: bool = False,
) -> ServingAdapterQualification:
    """Not implemented: builds no engine. Enforces the contract, then raises (T115)."""
    if claim_verified_from_float_export_alone or not engine_built or not fresh_runtime_parity_checked:
        raise M5SurgeryError(
            "UNVERIFIED_TRT_ENGINE",
            "an exported float checkpoint alone cannot verify a compiled TensorRT-LLM engine (T115)",
        )
    raise NotImplementedError(
        f"TensorRT-LLM {pinned_version} qualification is not implemented; no engine was built"
    )


@dataclass(frozen=True)
class SystemsProfileComparison:
    phase_breakdown_ns: dict[str, int]
    rust_assisted_total_ns: int
    python_reference_total_ns: int
    equal_integrity_work_verified: bool


def run_end_to_end_systems_profile(
    *,
    rust_performs_hashing_and_copy: bool = True,
    python_performs_hashing_and_copy: bool = True,
    workloads_matched: bool = True,
) -> SystemsProfileComparison:
    """Not implemented: it used to return fixed timings. Enforces the contract, then raises (T116)."""
    if not (rust_performs_hashing_and_copy and python_performs_hashing_and_copy and workloads_matched):
        raise M5SurgeryError(
            "UNEQUAL_INTEGRITY_WORK_COMPARISON",
            "systems profile comparison must perform identical hashing, copy-on-write, and verification work (T116)",
        )
    raise NotImplementedError("the end-to-end systems profile is not implemented; no timing was measured")


@dataclass(frozen=True)
class CustomKernelAdmissionReceipt:
    kernel_id: str
    admitted: bool
    isolated_speedup_ratio: float
    end_to_end_speedup_ratio: float
    reference_parity_passed: bool
    evidence: EvidenceBasis = "caller_supplied"


def evaluate_custom_kernel_admission(
    *,
    kernel_id: str,
    profiler_bottleneck_proven: bool,
    reference_parity_passed: bool,
    isolated_speedup_ratio: float,
    end_to_end_speedup_ratio: float,
) -> CustomKernelAdmissionReceipt:
    """Applies the admission rule to caller-supplied results; profiles and runs nothing (T117)."""
    if not profiler_bottleneck_proven or not reference_parity_passed:
        raise M5SurgeryError(
            "UNQUALIFIED_CUSTOM_KERNEL",
            "custom kernels require profiler bottleneck evidence and reference correctness parity",
        )
    if end_to_end_speedup_ratio <= 1.0:
        raise M5SurgeryError(
            "KERNEL_END_TO_END_REGRESSION",
            f"kernel '{kernel_id}' has isolated speedup {isolated_speedup_ratio:.2f}x but loses end-to-end ({end_to_end_speedup_ratio:.2f}x)",
        )
    return CustomKernelAdmissionReceipt(
        kernel_id=kernel_id,
        admitted=True,
        isolated_speedup_ratio=isolated_speedup_ratio,
        end_to_end_speedup_ratio=end_to_end_speedup_ratio,
        reference_parity_passed=True,
    )


@dataclass(frozen=True)
class WeightEfficiencyBenchReport:
    arms_evaluated: tuple[str, ...]
    pareto_optimal_arms: tuple[str, ...]
    complete_comparison_claimed: bool


def run_weight_efficiency_benchmark(
    *,
    evaluated_arms: tuple[str, ...],
    required_reference_arms: tuple[str, ...] = ("unchanged", "structured_pruning", "group_quantization"),
    claim_speedup_from_compressed_file_size_only: bool = False,
    claim_complete_comparison: bool = True,
) -> WeightEfficiencyBenchReport:
    """Not implemented: no arm is run or compared. Enforces the contract, then raises (T118)."""
    if claim_speedup_from_compressed_file_size_only:
        raise M5SurgeryError(
            "FILE_SIZE_NOT_RUNTIME_SPEEDUP",
            "smaller on-disk checkpoint size alone cannot be claimed as runtime latency/throughput improvement (T118)",
        )
    missing = set(required_reference_arms) - set(evaluated_arms)
    if missing and claim_complete_comparison:
        raise M5SurgeryError(
            "INCOMPLETE_EFFICIENCY_REFERENCE_ARMS",
            f"missing reference arms {sorted(missing)} prevent a complete comparison claim",
        )
    raise NotImplementedError(f"WeightEfficiencyBench is not implemented; arms {evaluated_arms} were not run")


@dataclass(frozen=True)
class RealBackendQualificationArtifact:
    backend_id: str
    hardware_name: str
    repeats: int
    retained_quality_verified: bool
    reload_parity_verified: bool
    measured_speedup_ratio: float | None
    m5_measured_improvement_status: Literal["reported_gain", "no_gain", "not_run_code_ready_only"]
    evidence: EvidenceBasis = "caller_supplied"


def qualify_real_backend_artifact(
    *,
    backend_id: str,
    hardware_name: str,
    run_executed: bool,
    repeats: int = 5,
    retained_quality_verified: bool | None = None,
    reload_parity_verified: bool | None = None,
    measured_speedup_ratio: float | None = None,
    claim_measured_gain_when_skipped: bool = False,
) -> RealBackendQualificationArtifact:
    """Records results the caller reports for a backend run. Executes nothing, so a gain is only reported (T119)."""
    if not run_executed and claim_measured_gain_when_skipped:
        raise M5SurgeryError(
            "SKIPPED_RUN_GAIN_CLAIM_FORBIDDEN",
            "code readiness or a skipped qualification run must never set M5 measured-improvement status (T119)",
        )
    if not run_executed:
        return RealBackendQualificationArtifact(
            backend_id=backend_id,
            hardware_name=hardware_name,
            repeats=0,
            retained_quality_verified=False,
            reload_parity_verified=False,
            measured_speedup_ratio=None,
            m5_measured_improvement_status="not_run_code_ready_only",
        )
    if retained_quality_verified is None or reload_parity_verified is None or measured_speedup_ratio is None:
        raise M5SurgeryError(
            "MISSING_BACKEND_RESULTS",
            "an executed run must report quality, reload parity and speedup; there are no defaults",
        )
    status: Literal["reported_gain", "no_gain"] = (
        "reported_gain"
        if (retained_quality_verified and reload_parity_verified and measured_speedup_ratio > 1.02)
        else "no_gain"
    )
    return RealBackendQualificationArtifact(
        backend_id=backend_id,
        hardware_name=hardware_name,
        repeats=repeats,
        retained_quality_verified=retained_quality_verified,
        reload_parity_verified=reload_parity_verified,
        measured_speedup_ratio=measured_speedup_ratio,
        m5_measured_improvement_status=status,
    )
