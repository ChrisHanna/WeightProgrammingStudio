"""Comprehensive tests for M5 Architecture Surgery, Quantization, Efficiency Gyms, and Profiling (T102–T107, T110–T119)."""

from __future__ import annotations

import pytest
import torch
from neural_foundry.m5_surgery import (
    M5SurgeryError,
    account_compression_graph_effects,
    apply_registered_structural_constructor,
    evaluate_compression_gym,
    evaluate_custom_kernel_admission,
    evaluate_generation_gym,
    evaluate_quantization_gym,
    execute_attention_head_surgery,
    execute_layer_reorganization,
    execute_mlp_width_surgery,
    qualify_real_backend_artifact,
    qualify_tensorrt_llm_adapter,
    qualify_vllm_serving_adapter,
    quantize_weight_grouped,
    run_end_to_end_systems_profile,
    run_serving_protocol_pytorch,
    run_weight_efficiency_benchmark,
)


def test_mlp_surgery_attention_surgery_layer_reorganization_and_structural_constructors() -> None:
    """T102 (`mlp_surgery`), T103 (`attention_surgery`), T104 (`layer_reorganization`), T105 (`structural_constructors`)."""
    # T102: MLP width surgery
    gate = torch.randn(8, 4, dtype=torch.float64)
    up = torch.randn(8, 4, dtype=torch.float64)
    down = torch.randn(4, 8, dtype=torch.float64)
    mlp_res = execute_mlp_width_surgery(
        gate_proj=gate,
        up_proj=up,
        down_proj=down,
        keep_indices=[0, 2, 5, 7],
    )
    assert mlp_res.updated_intermediate_size == 4
    assert tuple(mlp_res.down_proj.shape) == (4, 4)
    assert mlp_res.reload_parity_verified is None

    with pytest.raises(M5SurgeryError) as exc_info:
        execute_mlp_width_surgery(
            gate_proj=gate,
            up_proj=up,
            down_proj=down,
            keep_indices=[0, 2],
            slice_down_proj_columns=False,
        )
    assert exc_info.value.code == "UNCOUPLED_MLP_SURGERY"

    # T103: Attention head surgery
    q = torch.randn(16, 16, dtype=torch.float64)
    k = torch.randn(8, 16, dtype=torch.float64)
    v = torch.randn(8, 16, dtype=torch.float64)
    o = torch.randn(16, 16, dtype=torch.float64)
    attn_res = execute_attention_head_surgery(
        q_proj=q,
        k_proj=k,
        v_proj=v,
        o_proj=o,
        keep_q_heads=[0, 2],
        keep_kv_heads=[0],
        head_dim=4,
    )
    assert attn_res.kv_group_ratio == 2
    assert tuple(attn_res.q_proj.shape) == (8, 16)
    assert tuple(attn_res.o_proj.shape) == (16, 8)

    with pytest.raises(M5SurgeryError) as exc_info:
        execute_attention_head_surgery(
            q_proj=q,
            k_proj=k,
            v_proj=v,
            o_proj=o,
            keep_q_heads=[0, 1, 2],
            keep_kv_heads=[0, 1],
            head_dim=4,
        )
    assert exc_info.value.code == "INVALID_GQA_GROUPING"

    # T104: Layer reorganization
    layers = [torch.eye(4, dtype=torch.float64) * (i + 1) for i in range(4)]
    reorg = execute_layer_reorganization(layers, new_layer_order=[0, 2, 3])
    assert reorg.num_hidden_layers == 3
    assert reorg.behavioral_evaluation_required is True
    with pytest.raises(M5SurgeryError):
        execute_layer_reorganization(layers, new_layer_order=[0, 2, 1], claim_function_preserving=True)

    # T105: Structural constructors
    w = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float64)
    x = torch.tensor([[2.0, 1.0], [-1.0, 4.0]], dtype=torch.float64)
    c_rec = apply_registered_structural_constructor(
        w,
        x,
        constructor_id="split_linear_channel_exact.v1",
    )
    assert c_rec.exact_under_conditions is True
    assert c_rec.max_parity_abs_diff < 1e-12
    assert c_rec.structural_graph_modified is False
    with pytest.raises(M5SurgeryError):
        apply_registered_structural_constructor(
            w,
            x,
            constructor_id="insert_zero_residual_block.v1",
            violate_zero_init_precondition=True,
        )


def test_quantized_policy_and_compression_graph() -> None:
    """T106 (`quantized_policy`) and T107 (`compression_graph`)."""
    w = torch.tensor([[1.0, -0.5, 0.25, -1.0], [0.8, -0.2, 0.4, 0.1]], dtype=torch.float64)
    q_art = quantize_weight_grouped(w, group_size=4, bits=8)
    assert q_art.max_roundtrip_abs_error < 0.02
    assert q_art.calibration_digest is None

    with pytest.raises(M5SurgeryError) as exc_info:
        quantize_weight_grouped(w, group_size=4, bits=8, mutate_packed_without_scales=True)
    assert exc_info.value.code == "UNCOUPLED_PACKED_EDIT_FORBIDDEN"

    # T107: Compression graph separates zero-masking from structural removal
    mask_acc = account_compression_graph_effects(
        modality="unstructured_zero_mask",
        dense_parameters_before=1024,
        zeroed_or_removed_count=512,
    )
    assert mask_acc.removed_flops == 0

    struct_acc = account_compression_graph_effects(
        modality="structural_channel_removal",
        dense_parameters_before=1024,
        zeroed_or_removed_count=512,
    )
    assert struct_acc.removed_flops == 1024
    assert struct_acc.evidence == "formula_estimate"

    with pytest.raises(M5SurgeryError) as exc_info:
        account_compression_graph_effects(
            modality="unstructured_zero_mask",
            dense_parameters_before=1024,
            zeroed_or_removed_count=512,
            claim_flop_reduction_from_unstructured_zeros=True,
        )
    assert exc_info.value.code == "ZERO_MASK_NOT_FLOP_REDUCTION"


def test_quantization_gym_compression_gym_generation_gym_serving_protocol_vllm_adapter_tensorrt_adapter_systems_profile_kernel_admission_efficiency_benchmark_and_backend_qualification() -> None:
    """T110–T119 (`quantization_gym`, `compression_gym`, `generation_gym`, `serving_protocol`, `vllm_adapter`, `tensorrt_adapter`, `systems_profile`, `kernel_admission`, `efficiency_benchmark`, `backend_qualification`)."""
    # T110: Quantization Gym
    q_gym = evaluate_quantization_gym(
        quality_delta=0.0,
        real_lowbit_runtime_measured=True,
        memory_ratio=0.52,
        throughput_ratio=1.35,
    )
    assert q_gym.speed_or_memory_gain_verified is True
    assert q_gym.evidence == "caller_supplied"
    with pytest.raises(M5SurgeryError):
        evaluate_quantization_gym(
            quality_delta=0.0,
            real_lowbit_runtime_measured=False,
            memory_ratio=0.5,
            throughput_ratio=1.2,
            claim_speedup_from_float_dequant_simulation=True,
        )

    # T111: Compression Gym
    c_gym = evaluate_compression_gym(quality_delta=0.0, param_ratio=0.75, latency_ratio=0.80)
    assert c_gym.speed_or_memory_gain_verified is True
    with pytest.raises(M5SurgeryError):
        evaluate_compression_gym(
            quality_delta=-0.10,
            param_ratio=0.5,
            latency_ratio=0.5,
            ignore_quality_regression_for_smaller_model=True,
        )

    # T112: Generation Gym
    g_gym = evaluate_generation_gym(
        baseline_accuracy=0.85,
        candidate_accuracy=0.86,
        baseline_tokens=120,
        candidate_tokens=90,
    )
    assert g_gym.speed_or_memory_gain_verified is True
    with pytest.raises(M5SurgeryError):
        evaluate_generation_gym(
            baseline_accuracy=0.85,
            candidate_accuracy=0.60,
            baseline_tokens=120,
            candidate_tokens=30,
            truncated_failed_answers_counted_as_efficient=True,
        )

    # T113: Serving protocol
    srv = run_serving_protocol_pytorch(
        torch.eye(4, dtype=torch.float64),
        torch.ones(2, 4, dtype=torch.float64),
    )
    assert srv.details["warm_p50_ns"] > 0
    assert srv.evidence == "measured_on_fixture"
    assert srv.speed_or_memory_gain_verified is None
    assert srv.retained_quality_ok is None

    # T114 & T115: vLLM and TensorRT-LLM adapters
    with pytest.raises(M5SurgeryError) as absent_vllm:
        qualify_vllm_serving_adapter(vllm_python="/nonexistent/vllm/python")
    assert absent_vllm.value.code == "VLLM_UNAVAILABLE"
    with pytest.raises(M5SurgeryError):
        qualify_vllm_serving_adapter(bind_host="0.0.0.0")

    with pytest.raises(NotImplementedError):
        qualify_tensorrt_llm_adapter()
    with pytest.raises(M5SurgeryError):
        qualify_tensorrt_llm_adapter(claim_verified_from_float_export_alone=True)

    # T116: Systems profile
    with pytest.raises(NotImplementedError):
        run_end_to_end_systems_profile()
    with pytest.raises(M5SurgeryError):
        run_end_to_end_systems_profile(rust_performs_hashing_and_copy=False)

    # T117: Custom kernel admission
    kern = evaluate_custom_kernel_admission(
        kernel_id="triton.subspace_proj.v1",
        profiler_bottleneck_proven=True,
        reference_parity_passed=True,
        isolated_speedup_ratio=1.8,
        end_to_end_speedup_ratio=1.18,
    )
    assert kern.admitted is True
    assert kern.evidence == "caller_supplied"
    with pytest.raises(M5SurgeryError):
        evaluate_custom_kernel_admission(
            kernel_id="triton.slow_e2e.v1",
            profiler_bottleneck_proven=True,
            reference_parity_passed=True,
            isolated_speedup_ratio=2.0,
            end_to_end_speedup_ratio=0.92,
        )

    # T118: WeightEfficiencyBench
    with pytest.raises(NotImplementedError):
        run_weight_efficiency_benchmark(
            evaluated_arms=("unchanged", "structured_pruning", "group_quantization"),
        )
    with pytest.raises(M5SurgeryError):
        run_weight_efficiency_benchmark(
            evaluated_arms=("unchanged",),
            claim_complete_comparison=True,
        )

    # T119: Real-backend qualification
    qual = qualify_real_backend_artifact(
        backend_id="nf.trusted_torch_cpu",
        hardware_name="local_cpu_verified",
        run_executed=True,
        retained_quality_verified=True,
        reload_parity_verified=True,
        measured_speedup_ratio=1.12,
    )
    assert qual.m5_measured_improvement_status == "reported_gain"
    assert qual.evidence == "caller_supplied"
    with pytest.raises(M5SurgeryError) as exc_info:
        qualify_real_backend_artifact(backend_id="b", hardware_name="h", run_executed=True)
    assert exc_info.value.code == "MISSING_BACKEND_RESULTS"
    skipped = qualify_real_backend_artifact(
        backend_id="nf.tensorrt_llm",
        hardware_name="rtx_5090",
        run_executed=False,
    )
    assert skipped.m5_measured_improvement_status == "not_run_code_ready_only"
    assert skipped.measured_speedup_ratio is None
