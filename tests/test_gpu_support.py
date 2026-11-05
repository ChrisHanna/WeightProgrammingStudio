"""Tests for GPU tensor view/DLPack dispatch (T108) and GPU resource leasing & measurement (T109)."""

from __future__ import annotations

import pytest
import torch
from neural_foundry.gpu import (
    GpuDispatchError,
    GpuResourceScheduler,
    dispatch_weight_operation,
    export_dlpack_contract,
    import_dlpack_verified,
    probe_gpu_hardware,
)


def test_gpu_dispatch_dlpack_contract_immutability_and_no_silent_fallback() -> None:
    """T108 (`gpu_dispatch`): DLPack view contracts, copy-on-write immutability, and fail-closed GPU/precision checks."""
    hw = probe_gpu_hardware()
    assert isinstance(hw.nvidia_smi_devices, tuple)

    # 1. Valid DLPack export/import and bf16 weight operation with f32 accumulation on CPU/available device
    exec_device = "cuda:0" if hw.torch_cuda_available else "cpu"
    src = torch.tensor([[1.0, -2.0], [3.5, 0.5]], dtype=torch.bfloat16, device=exec_device)
    src_clone = src.clone()

    res = dispatch_weight_operation(
        src,
        tensor_name="model.layers.0.mlp.down_proj.weight",
        operation="scale",
        requested_device=exec_device,
        requested_dtype="bf16",
        accumulation_dtype="f32",
        stream_id=0,
        factor=1.5,
    )
    assert res.source_mutated is False
    assert torch.equal(src, src_clone)
    assert res.output_tensor.dtype == torch.bfloat16
    assert len(res.transfer_receipts) == 1
    assert res.transfer_receipts[0].byte_length == src.numel() * src.element_size()
    assert res.before_digest != res.after_digest

    # 2. Subspace projection with orthonormal basis in f32
    src_f32 = torch.tensor([[2.0, 0.0], [0.0, 4.0]], dtype=torch.float32, device=exec_device)
    q = torch.tensor([[1.0], [0.0]], dtype=torch.float32, device=exec_device)
    proj_res = dispatch_weight_operation(
        src_f32,
        tensor_name="model.layers.0.self_attn.o_proj.weight",
        operation="subspace_projection",
        requested_device=exec_device,
        requested_dtype="f32",
        accumulation_dtype="f64",
        factor=1.0,
        left_basis=q,
    )
    assert torch.allclose(
        proj_res.output_tensor.cpu(),
        torch.tensor([[0.0, 0.0], [0.0, 4.0]], dtype=torch.float32),
    )

    # 3. Negative: non-contiguous stride layout is rejected at DLPack boundary
    non_contig = torch.randn(4, 4, dtype=torch.float32).t()
    with pytest.raises(GpuDispatchError) as exc_info:
        export_dlpack_contract(non_contig, tensor_name="non_contig")
    assert exc_info.value.code == "NON_CONTIGUOUS_STRIDE"

    # 4. Negative: expired owner lifetime token is rejected
    capsule, contract = export_dlpack_contract(src_f32, tensor_name="valid_src", stream_id=3)
    with pytest.raises(GpuDispatchError) as exc_info:
        import_dlpack_verified(
            capsule,
            contract,
            expected_device_type="cuda" if hw.torch_cuda_available else "cpu",
            expected_dtype="f32",
            expected_stream_id=3,
            owner_alive=False,
        )
    assert exc_info.value.code == "DLPACK_LIFETIME_EXPIRED"

    # 5. Negative: stream mismatch is rejected
    capsule2, contract2 = export_dlpack_contract(src_f32, tensor_name="valid_src", stream_id=3)
    with pytest.raises(GpuDispatchError) as exc_info:
        import_dlpack_verified(
            capsule2,
            contract2,
            expected_device_type="cuda" if hw.torch_cuda_available else "cpu",
            expected_dtype="f32",
            expected_stream_id=7,
        )
    assert exc_info.value.code == "STREAM_MISMATCH"

    # 6. Negative: mutable source sharing is rejected
    with pytest.raises(GpuDispatchError) as exc_info:
        dispatch_weight_operation(
            src_f32,
            tensor_name="w",
            operation="scale",
            requested_device=exec_device,
            requested_dtype="f32",
            allow_mutable_source_sharing=True,
        )
    assert exc_info.value.code == "MUTABLE_SOURCE_SHARING_FORBIDDEN"

    # 7. Negative: silent precision downgrade or CPU fallback is rejected
    with pytest.raises(GpuDispatchError) as exc_info:
        dispatch_weight_operation(
            src_f32,
            tensor_name="w",
            operation="scale",
            requested_device=exec_device,
            requested_dtype="bf16",  # src_f32 is f32, implicit cast denied
        )
    assert exc_info.value.code == "PRECISION_MISMATCH"

    if not hw.torch_cuda_available:
        with pytest.raises(GpuDispatchError) as exc_info:
            dispatch_weight_operation(
                src_f32,
                tensor_name="w",
                operation="scale",
                requested_device="cuda:0",
                requested_dtype="f32",
            )
        assert exc_info.value.code == "GPU_DEVICE_UNAVAILABLE"


def test_gpu_resources_leasing_synchronization_and_memory_scopes() -> None:
    """T109 (`gpu_resources`): GPU leasing, synchronized timings, and allocator vs NVML memory scopes."""
    scheduler = GpuResourceScheduler(
        device_count=1,
        vram_capacity_bytes_per_device=16 * 1024 * 1024 * 1024,
    )
    lease = scheduler.acquire_lease(
        holder="study-qwen2-repair",
        device_index=0,
        reserved_vram_bytes=8 * 1024 * 1024 * 1024,
    )

    # 1. Over-subscribing VRAM on the same device fails closed
    with pytest.raises(GpuDispatchError) as exc_info:
        scheduler.acquire_lease(
            holder="competing-worker",
            device_index=0,
            reserved_vram_bytes=10 * 1024 * 1024 * 1024,
        )
    assert exc_info.value.code == "GPU_VRAM_EXHAUSTED"

    # 2. Valid synchronized measurement separates compile, transfer, and steady-state inference
    sample = scheduler.record_synchronized_measurement(
        lease=lease,
        synchronized=True,
        compile_ns=12_000_000,
        transfer_ns=3_500_000,
        steady_state_inference_ns=1_800_000,
        allocator_peak_bytes=2 * 1024 * 1024 * 1024,
        device_total_nvml_bytes=3 * 1024 * 1024 * 1024,
        requested_memory_scope="device_total_nvml",
    )
    assert sample.synchronized is True
    assert sample.compile_ns == 12_000_000
    assert sample.transfer_ns == 3_500_000
    assert sample.steady_state_inference_ns == 1_800_000
    assert sample.memory_scope_reported == "device_total_nvml"

    # 3. Negative: unsynchronized GPU kernel timing is rejected
    with pytest.raises(GpuDispatchError) as exc_info:
        scheduler.record_synchronized_measurement(
            lease=lease,
            synchronized=False,
            compile_ns=0,
            transfer_ns=0,
            steady_state_inference_ns=500_000,
            allocator_peak_bytes=1024,
            device_total_nvml_bytes=2048,
            requested_memory_scope="framework_allocator_peak",
        )
    assert exc_info.value.code == "UNSYNCHRONIZED_GPU_TIMING"

    # 4. Negative: conflating compilation/transfer overhead into inference timing is rejected
    with pytest.raises(GpuDispatchError) as exc_info:
        scheduler.record_synchronized_measurement(
            lease=lease,
            synchronized=True,
            compile_ns=0,
            transfer_ns=0,
            steady_state_inference_ns=500_000,
            allocator_peak_bytes=1024,
            device_total_nvml_bytes=2048,
            requested_memory_scope="framework_allocator_peak",
            mix_compile_or_transfer_into_inference=True,
        )
    assert exc_info.value.code == "TIMING_PHASE_CONFLATION"

    # 5. Negative: claiming device_total_nvml scope when NVML total is missing is rejected
    with pytest.raises(GpuDispatchError) as exc_info:
        scheduler.record_synchronized_measurement(
            lease=lease,
            synchronized=True,
            compile_ns=100,
            transfer_ns=100,
            steady_state_inference_ns=500_000,
            allocator_peak_bytes=1024,
            device_total_nvml_bytes=None,
            requested_memory_scope="device_total_nvml",
        )
    assert exc_info.value.code == "MEMORY_SCOPE_UNAVAILABLE"

    scheduler.release_lease(lease.lease_id)
