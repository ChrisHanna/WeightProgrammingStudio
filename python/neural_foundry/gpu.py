"""GPU tensor view, DLPack dispatch, and GPU resource leasing & measurement (T108, T109).

Implements:
- T108 (`gpu_dispatch`): DLPack/device/stride/stream/lifetime contracts, bounded transfer
  receipts, mutable-source sharing prohibition, and explicit failure when requested GPU
  precision or device is unavailable (never silently falling back to CPU or lower precision).
- T109 (`gpu_resources`): GPU device lease scheduling, synchronized CUDA event/wall timings,
  separated `framework_allocator_peak` vs. `device_total_nvml` memory scopes, and separated
  compilation vs. host-device transfer vs. steady-state inference accounting.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils import dlpack as torch_dlpack

from .records import canonicalize, content_digest

MemoryMeasurementScope = Literal["framework_allocator_peak", "device_total_nvml"]
SupportedGpuDType = Literal["f16", "bf16", "f32", "f64"]

_DTYPE_TO_TORCH: dict[SupportedGpuDType, torch.dtype] = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
    "f64": torch.float64,
}

_TORCH_TO_DTYPE: dict[torch.dtype, SupportedGpuDType] = {
    v: k for k, v in _DTYPE_TO_TORCH.items()
}


def _digest_obj(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    try:
        canon = canonicalize(raw)
        return str(content_digest(canon))
    except (ValueError, TypeError, RuntimeError):
        return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _tensor_bytes_digest(tensor: torch.Tensor) -> str:
    detached = tensor.detach().contiguous().to(device="cpu", dtype=torch.float64).reshape(-1)
    vals = [float(x) for x in detached.tolist()]
    raw = struct.pack(f"<{len(vals)}d", *vals)
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


@dataclass(frozen=True)
class HardwareGpuInventory:
    """Probed GPU hardware, runtime visibility, and executable kernel support."""

    torch_cuda_available: bool
    torch_device_count: int
    nvidia_smi_devices: tuple[str, ...]
    driver_version: str
    torch_arch_list: tuple[str, ...]
    kernel_probe_verified: bool
    kernel_probe_error: str

    @property
    def cuda_executable(self) -> bool:
        return self.torch_cuda_available and self.torch_device_count > 0 and self.kernel_probe_verified


@dataclass(frozen=True)
class TrainingComputePlacement:
    device: str
    dtype: Literal["f32", "f64"]
    reason: str


@dataclass(frozen=True)
class ExternalGpuRuntime:
    python_executable: str
    torch_version: str
    device_name: str
    torch_arch_list: tuple[str, ...]
    total_memory_bytes: int = 0
    free_memory_bytes: int = 0
    device: str = "cuda:0"


def probe_external_gpu_runtime(python_executable: str | Path, *, device: str = "cuda:0") -> ExternalGpuRuntime:
    """Verify a separate local interpreter before authorizing GPU training there."""
    path = Path(python_executable).expanduser()
    if not path.is_file():
        raise GpuDispatchError("GPU_RUNTIME_UNAVAILABLE", f"GPU interpreter is missing: {path}")
    if not isinstance(device, str) or not device.startswith("cuda:") or not device[5:].isascii() or not device[5:].isdecimal() or (len(device[5:]) > 1 and device[5] == "0"):
        raise GpuDispatchError("INVALID_GPU_DEVICE", str(device))
    code = (
        "import json, torch\n"
        f"device={device!r}\nindex=int(device.split(':')[1])\n"
        "ok=False\n"
        "error=''\n"
        "try:\n"
        " a=torch.ones((2,2),device=device,dtype=torch.float32)\n"
        " b=a@a\n"
        " torch.cuda.synchronize(index)\n"
        " ok=bool(torch.equal(b,torch.full((2,2),2.0,device=device)))\n"
        " free_memory,total_memory=torch.cuda.mem_get_info(index)\n"
        "except Exception as exc:\n"
        " error=type(exc).__name__+': '+str(exc)\n"
        " free_memory,total_memory=0,0\n"
        "print('NF_GPU_PROBE'+json.dumps({\"torch\":torch.__version__,"
        "\"available\":torch.cuda.is_available(),"
        "\"name\":torch.cuda.get_device_name(index) if torch.cuda.is_available() else None,"
        "\"arch_list\":torch.cuda.get_arch_list() if torch.cuda.is_available() else [],"
        "\"free_memory_bytes\":free_memory,\"total_memory_bytes\":total_memory,"
        "\"ok\":ok,\"error\":error}))\n"
    )
    completed = subprocess.run(
        [str(path), "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    lines = [line for line in completed.stdout.splitlines() if line.startswith("NF_GPU_PROBE")]
    if completed.returncode != 0 or not lines:
        detail = completed.stderr.strip()[-500:]
        raise GpuDispatchError(
            "GPU_RUNTIME_UNAVAILABLE",
            f"GPU interpreter probe exited {completed.returncode}: {detail}",
        )
    result: dict[str, Any] = json.loads(lines[-1].removeprefix("NF_GPU_PROBE"))
    if not result.get("available"):
        raise GpuDispatchError("GPU_DEVICE_UNAVAILABLE", "GPU interpreter reports no CUDA device")
    if not result.get("ok"):
        raise GpuDispatchError(
            "GPU_KERNEL_UNAVAILABLE",
            f"GPU interpreter cannot execute CUDA: {result.get('error', '')}",
        )
    return ExternalGpuRuntime(
        python_executable=str(path),
        torch_version=str(result["torch"]),
        device_name=str(result["name"]),
        torch_arch_list=tuple(str(item) for item in result["arch_list"]),
        total_memory_bytes=int(result.get("total_memory_bytes", 0)),
        free_memory_bytes=int(result.get("free_memory_bytes", 0)),
        device=device,
    )


def discover_external_gpu_runtime(configured: str | Path | None = None, *, device: str | None = None) -> ExternalGpuRuntime | None:
    """Return an explicitly configured or conventional local GPU runtime when executable."""
    explicit = configured or os.environ.get("NF_GPU_PYTHON")
    selected_device = device or os.environ.get("NF_GPU_DEVICE", "cuda:0")
    if explicit:
        return probe_external_gpu_runtime(explicit, device=selected_device)
    candidates = (
        Path.home() / ".venvs" / "neural-foundry-gpu" / "bin" / "python",
        Path.home() / ".venvs" / "neural-foundry-gpu" / "Scripts" / "python.exe",
    )
    for candidate in candidates:
        if candidate.is_file():
            try:
                return probe_external_gpu_runtime(candidate, device=selected_device)
            except GpuDispatchError:
                return None
    return None


def select_training_compute(requested_device: str = "auto") -> TrainingComputePlacement:
    """Select GPU for gradient work only after an executable-kernel preflight."""
    requested_index: int | None = None
    if requested_device not in {"auto", "cpu"}:
        if not isinstance(requested_device, str) or not requested_device.startswith("cuda:"):
            raise GpuDispatchError("INVALID_TRAINING_DEVICE", str(requested_device))
        index_text = requested_device.removeprefix("cuda:")
        if (
            not index_text
            or not index_text.isascii()
            or not index_text.isdecimal()
            or (len(index_text) > 1 and index_text.startswith("0"))
        ):
            raise GpuDispatchError("INVALID_GPU_DEVICE", requested_device)
        requested_index = int(index_text)
    inventory = probe_gpu_hardware()
    if requested_device == "auto":
        if inventory.cuda_executable:
            return TrainingComputePlacement("cuda:0", "f32", "verified_cuda_kernel")
        reason = inventory.kernel_probe_error or "CUDA unavailable"
        return TrainingComputePlacement("cpu", "f64", f"cpu_fallback: {reason}")
    if requested_device == "cpu":
        return TrainingComputePlacement("cpu", "f64", "cpu_requested")
    if requested_index is not None:
        if not inventory.cuda_executable:
            raise GpuDispatchError(
                "GPU_KERNEL_UNAVAILABLE",
                f"CUDA training requested but no kernel can execute: {inventory.kernel_probe_error}",
            )
        if requested_index >= inventory.torch_device_count:
            raise GpuDispatchError("INVALID_GPU_DEVICE", requested_device)
        return TrainingComputePlacement(requested_device, "f32", "verified_cuda_kernel")
    raise GpuDispatchError("INVALID_TRAINING_DEVICE", requested_device)


def _probe_cuda_kernel() -> tuple[bool, str]:
    """Synchronize a real operation so unsupported runtime/device architectures fail closed."""
    try:
        left = torch.ones((2, 2), device="cuda:0", dtype=torch.float32)
        right = torch.ones((2, 2), device="cuda:0", dtype=torch.float32)
        output = left @ right
        torch.cuda.synchronize(0)
        if not torch.equal(output, torch.full((2, 2), 2.0, device="cuda:0")):
            return False, "CUDA smoke kernel returned an unexpected result"
    except (AssertionError, RuntimeError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def probe_gpu_hardware() -> HardwareGpuInventory:
    """Probe driver visibility and prove that the installed Torch build can execute on the GPU."""
    cuda_ok = bool(torch.cuda.is_available())
    dev_count = int(torch.cuda.device_count()) if cuda_ok else 0
    arch_list = tuple(torch.cuda.get_arch_list()) if cuda_ok else ()
    kernel_ok, kernel_error = _probe_cuda_kernel() if cuda_ok and dev_count else (False, "CUDA unavailable")
    smi_devices: list[str] = []
    driver = "unavailable"
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            for line in proc.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if parts:
                    smi_devices.append(parts[0])
                if len(parts) > 1 and driver == "unavailable":
                    driver = parts[1]
    except (OSError, subprocess.SubprocessError):
        driver = "unavailable"
    return HardwareGpuInventory(
        torch_cuda_available=cuda_ok,
        torch_device_count=dev_count,
        nvidia_smi_devices=tuple(smi_devices),
        driver_version=driver,
        torch_arch_list=arch_list,
        kernel_probe_verified=kernel_ok,
        kernel_probe_error=kernel_error,
    )


@dataclass(frozen=True)
class DLPackViewContract:
    """Explicit DLPack tensor view contract with device, dtype, shape, stride, stream, and lifetime."""

    tensor_name: str
    device_type: Literal["cuda", "cpu"]
    device_index: int
    dtype: SupportedGpuDType
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    stream_id: int
    read_only_source: bool
    owner_lifetime_token: str
    data_ptr: int
    byte_length: int
    content_digest: str


@dataclass(frozen=True)
class TransferReceipt:
    """Explicit receipt for every host<->device or cross-stream copy."""

    receipt_id: str
    source_device: str
    target_device: str
    dtype: SupportedGpuDType
    byte_length: int
    stream_id: int
    transfer_ns: int
    before_digest: str
    after_digest: str


@dataclass(frozen=True)
class GpuDispatchResult:
    """Result of executing a compiled weight operation through the DLPack/device dispatcher."""

    operation: str
    target_tensor: str
    execution_device: str
    execution_dtype: SupportedGpuDType
    accumulation_dtype: Literal["f32", "f64"]
    stream_id: int
    source_mutated: bool
    dlpack_contract: DLPackViewContract
    transfer_receipts: tuple[TransferReceipt, ...]
    before_digest: str
    after_digest: str
    output_tensor: torch.Tensor


class GpuDispatchError(RuntimeError):
    """Structured error raised when a GPU/DLPack dispatch contract is violated."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code


def export_dlpack_contract(
    tensor: torch.Tensor,
    *,
    tensor_name: str,
    stream_id: int = 0,
    read_only_source: bool = True,
    owner_alive: bool = True,
) -> tuple[Any, DLPackViewContract]:
    """Exports a tensor through DLPack while recording and checking its view contract."""
    if not owner_alive:
        raise GpuDispatchError(
            "DLPACK_LIFETIME_EXPIRED",
            f"cannot export DLPack capsule for '{tensor_name}' after owner lifetime ended",
        )
    if stream_id < 0:
        raise GpuDispatchError(
            "INVALID_STREAM",
            f"stream_id must be non-negative, got {stream_id}",
        )
    if tensor.dtype not in _TORCH_TO_DTYPE:
        raise GpuDispatchError(
            "UNSUPPORTED_DTYPE",
            f"unsupported tensor dtype {tensor.dtype} for '{tensor_name}'",
        )
    if not tensor.is_contiguous():
        raise GpuDispatchError(
            "NON_CONTIGUOUS_STRIDE",
            f"tensor '{tensor_name}' has non-contiguous strides {tuple(tensor.stride())}; explicit contiguous layout required",
        )
    if not bool(torch.isfinite(tensor).all().item()):
        raise GpuDispatchError(
            "NON_FINITE_TENSOR",
            f"tensor '{tensor_name}' contains NaN or Inf values",
        )

    dev_type: Literal["cuda", "cpu"] = "cuda" if tensor.is_cuda else "cpu"
    dev_idx = int(tensor.device.index or 0)
    dtype_name = _TORCH_TO_DTYPE[tensor.dtype]
    capsule = tensor.detach()
    byte_length = int(tensor.numel() * tensor.element_size())
    token = _digest_obj(
        {
            "tensor_name": tensor_name,
            "data_ptr": int(tensor.data_ptr()),
            "shape": list(tensor.shape),
            "strides": list(tensor.stride()),
            "stream_id": stream_id,
        }
    )
    contract = DLPackViewContract(
        tensor_name=tensor_name,
        device_type=dev_type,
        device_index=dev_idx,
        dtype=dtype_name,
        shape=tuple(int(d) for d in tensor.shape),
        strides=tuple(int(s) for s in tensor.stride()),
        stream_id=stream_id,
        read_only_source=read_only_source,
        owner_lifetime_token=token,
        data_ptr=int(tensor.data_ptr()),
        byte_length=byte_length,
        content_digest=_tensor_bytes_digest(tensor),
    )
    return capsule, contract


def import_dlpack_verified(
    capsule: Any,
    contract: DLPackViewContract,
    *,
    expected_device_type: Literal["cuda", "cpu"],
    expected_dtype: SupportedGpuDType,
    expected_stream_id: int,
    owner_alive: bool = True,
) -> torch.Tensor:
    """Imports a DLPack capsule and verifies device, dtype, shape, strides, stream, and lifetime."""
    if not owner_alive:
        raise GpuDispatchError(
            "DLPACK_LIFETIME_EXPIRED",
            f"owner lifetime token {contract.owner_lifetime_token} expired before DLPack import",
        )
    if contract.device_type != expected_device_type:
        raise GpuDispatchError(
            "DEVICE_MISMATCH",
            f"expected device '{expected_device_type}', got '{contract.device_type}'",
        )
    if contract.dtype != expected_dtype:
        raise GpuDispatchError(
            "PRECISION_MISMATCH",
            f"expected dtype '{expected_dtype}', got '{contract.dtype}'; silent precision fallback is forbidden",
        )
    if contract.stream_id != expected_stream_id:
        raise GpuDispatchError(
            "STREAM_MISMATCH",
            f"expected CUDA stream {expected_stream_id}, got {contract.stream_id}",
        )
    imported = torch_dlpack.from_dlpack(capsule)
    if tuple(int(d) for d in imported.shape) != contract.shape:
        raise GpuDispatchError("SHAPE_MISMATCH", "DLPack imported shape mismatch")
    if tuple(int(s) for s in imported.stride()) != contract.strides:
        raise GpuDispatchError("STRIDE_MISMATCH", "DLPack imported stride mismatch")
    return imported


def dispatch_weight_operation(
    source_tensor: torch.Tensor,
    *,
    tensor_name: str,
    operation: Literal["scale", "add_delta", "low_rank_update", "subspace_projection"],
    requested_device: str = "cuda:0",
    requested_dtype: SupportedGpuDType = "bf16",
    accumulation_dtype: Literal["f32", "f64"] = "f32",
    stream_id: int = 0,
    max_transfer_bytes: int = 64 * 1024 * 1024,
    allow_mutable_source_sharing: bool = False,
    allow_silent_cpu_fallback: bool = False,
    factor: float = 1.0,
    delta: torch.Tensor | None = None,
    left_basis: torch.Tensor | None = None,
    right_basis: torch.Tensor | None = None,
) -> GpuDispatchResult:
    """Executes a weight transformation with strict DLPack, transfer, and immutability checks.

    - Never silently falls back from `cuda:*` to `cpu` or downgrades precision.
    - Never mutates `source_tensor` in-place; rejects `allow_mutable_source_sharing=True`.
    - Records explicit `TransferReceipt` objects for every host/device copy and enforces
      `max_transfer_bytes`.
    """
    if allow_mutable_source_sharing:
        raise GpuDispatchError(
            "MUTABLE_SOURCE_SHARING_FORBIDDEN",
            "mutable source buffer sharing across transactions is forbidden; copy-on-write staging required",
        )
    if allow_silent_cpu_fallback:
        raise GpuDispatchError(
            "SILENT_FALLBACK_FORBIDDEN",
            "silent fallback to CPU or reduced precision is forbidden by protocol",
        )
    if requested_dtype not in _DTYPE_TO_TORCH:
        raise GpuDispatchError(
            "UNSUPPORTED_DTYPE",
            f"unsupported requested_dtype '{requested_dtype}'",
        )
    if accumulation_dtype not in ("f32", "f64"):
        raise GpuDispatchError(
            "UNSUPPORTED_ACCUMULATION_DTYPE",
            f"accumulation_dtype must be 'f32' or 'f64', got '{accumulation_dtype}'",
        )

    wants_cuda = requested_device.startswith("cuda")
    if wants_cuda:
        inventory = probe_gpu_hardware()
        if not inventory.torch_cuda_available:
            raise GpuDispatchError(
                "GPU_DEVICE_UNAVAILABLE",
                f"requested {requested_device!r}, but no CUDA device is visible; refusing silent CPU fallback",
            )
        if not inventory.kernel_probe_verified:
            raise GpuDispatchError(
                "GPU_KERNEL_UNAVAILABLE",
                f"CUDA is visible but no kernel can execute on it: {inventory.kernel_probe_error}",
            )

    target_torch_dtype = _DTYPE_TO_TORCH[requested_dtype]
    accum_torch_dtype = torch.float32 if accumulation_dtype == "f32" else torch.float64

    if source_tensor.dtype != target_torch_dtype:
        raise GpuDispatchError(
            "PRECISION_MISMATCH",
            f"source tensor dtype {source_tensor.dtype} does not match requested '{requested_dtype}'; implicit precision cast denied",
        )

    before_digest = _tensor_bytes_digest(source_tensor)
    byte_length = int(source_tensor.numel() * source_tensor.element_size())
    if byte_length > max_transfer_bytes:
        raise GpuDispatchError(
            "TRANSFER_BUDGET_EXCEEDED",
            f"tensor byte length {byte_length} exceeds max_transfer_bytes {max_transfer_bytes}",
        )

    # Export DLPack contract from read-only source and verify roundtrip
    capsule, dlpack_contract = export_dlpack_contract(
        source_tensor,
        tensor_name=tensor_name,
        stream_id=stream_id,
        read_only_source=True,
    )
    dev_kind: Literal["cuda", "cpu"] = "cuda" if wants_cuda else "cpu"
    imported_view = import_dlpack_verified(
        capsule,
        dlpack_contract,
        expected_device_type=dev_kind,
        expected_dtype=requested_dtype,
        expected_stream_id=stream_id,
    )

    # Stage an isolated copy so the source buffer is never mutated
    t0 = time.perf_counter_ns()
    staged = imported_view.clone().to(device=requested_device, dtype=accum_torch_dtype)
    if wants_cuda:
        torch.cuda.synchronize()
    t1 = time.perf_counter_ns()

    if operation == "scale":
        if not torch.isfinite(torch.tensor(factor)):
            raise GpuDispatchError("NON_FINITE_OPERAND", "scale factor must be finite")
        result_accum = staged * float(factor)
    elif operation == "add_delta":
        if delta is None or tuple(delta.shape) != tuple(source_tensor.shape):
            raise GpuDispatchError("OPERAND_SHAPE_MISMATCH", "add_delta requires matching delta shape")
        result_accum = staged + delta.to(device=requested_device, dtype=accum_torch_dtype)
    elif operation == "low_rank_update":
        if left_basis is None or right_basis is None:
            raise GpuDispatchError("MISSING_OPERAND", "low_rank_update requires left_basis and right_basis")
        u = left_basis.to(device=requested_device, dtype=accum_torch_dtype)
        v = right_basis.to(device=requested_device, dtype=accum_torch_dtype)
        result_accum = staged + float(factor) * (u @ v)
    elif operation == "subspace_projection":
        if left_basis is None:
            raise GpuDispatchError("MISSING_OPERAND", "subspace_projection requires orthonormal basis")
        q = left_basis.to(device=requested_device, dtype=accum_torch_dtype)
        ident = torch.eye(q.shape[1], dtype=accum_torch_dtype, device=requested_device)
        ortho_err = float(torch.max(torch.abs(q.T @ q - ident)).item())
        if ortho_err > 1e-5:
            raise GpuDispatchError("NON_ORTHONORMAL_BASIS", f"orthonormality error {ortho_err:.3e} > 1e-5")
        proj = (q @ q.T) @ staged
        result_accum = staged - float(factor) * proj
    else:
        raise GpuDispatchError("UNSUPPORTED_OPERATION", f"unknown operation '{operation}'")

    out_tensor = result_accum.to(dtype=target_torch_dtype).contiguous()
    if not bool(torch.isfinite(out_tensor).all().item()):
        raise GpuDispatchError("NON_FINITE_OUTPUT", "output tensor contains non-finite values")

    # Verify source tensor was never mutated in-place
    source_after_check = _tensor_bytes_digest(source_tensor)
    if source_after_check != before_digest:
        raise GpuDispatchError("SOURCE_MUTATED", "source tensor was mutated in place")

    after_digest = _tensor_bytes_digest(out_tensor)
    receipt = TransferReceipt(
        receipt_id=_digest_obj(
            {
                "tensor": tensor_name,
                "before": before_digest,
                "after": after_digest,
                "bytes": byte_length,
                "stream_id": stream_id,
            }
        ),
        source_device=str(source_tensor.device),
        target_device=requested_device,
        dtype=requested_dtype,
        byte_length=byte_length,
        stream_id=stream_id,
        transfer_ns=max(1, t1 - t0),
        before_digest=before_digest,
        after_digest=after_digest,
    )

    return GpuDispatchResult(
        operation=operation,
        target_tensor=tensor_name,
        execution_device=requested_device,
        execution_dtype=requested_dtype,
        accumulation_dtype=accumulation_dtype,
        stream_id=stream_id,
        source_mutated=False,
        dlpack_contract=dlpack_contract,
        transfer_receipts=(receipt,),
        before_digest=before_digest,
        after_digest=after_digest,
        output_tensor=out_tensor,
    )


@dataclass(frozen=True)
class GpuLease:
    """Exclusive or bounded lease on a GPU device index."""

    lease_id: str
    holder: str
    device_index: int
    reserved_vram_bytes: int
    acquired_ns: int
    expires_ns: int


@dataclass(frozen=True)
class GpuResourceSample:
    """Synchronized GPU timing and memory sample with explicit measurement scope."""

    lease_id: str
    device_index: int
    synchronized: bool
    synchronization_method: Literal["cuda_event_synchronize", "host_wall_clock_synced"]
    compile_ns: int
    transfer_ns: int
    steady_state_inference_ns: int
    allocator_peak_bytes: int
    device_total_nvml_bytes: int | None
    memory_scope_reported: MemoryMeasurementScope


class GpuResourceScheduler:
    """Manages GPU device leases, synchronized timings, and explicit memory scopes (T109)."""

    def __init__(
        self,
        *,
        device_count: int = 1,
        vram_capacity_bytes_per_device: int = 32 * 1024 * 1024 * 1024,
    ) -> None:
        if device_count <= 0 or vram_capacity_bytes_per_device <= 0:
            raise ValueError("positive device_count and vram_capacity_bytes_per_device required")
        self.device_count = device_count
        self.vram_capacity_bytes_per_device = vram_capacity_bytes_per_device
        self._active_leases: dict[str, GpuLease] = {}
        self._lock = threading.RLock()

    def _expire_leases(self, now_ns: int) -> None:
        expired = [
            lease_id
            for lease_id, lease in self._active_leases.items()
            if now_ns >= lease.expires_ns
        ]
        for lease_id in expired:
            del self._active_leases[lease_id]

    def acquire_lease(
        self,
        *,
        holder: str,
        device_index: int = 0,
        reserved_vram_bytes: int,
        ttl_ms: int = 60_000,
    ) -> GpuLease:
        if device_index < 0 or device_index >= self.device_count:
            raise GpuDispatchError(
                "INVALID_GPU_DEVICE",
                f"device_index {device_index} out of range (0..{self.device_count})",
            )
        if not holder.strip() or reserved_vram_bytes <= 0 or ttl_ms <= 0:
            raise GpuDispatchError(
                "INVALID_LEASE_REQUEST",
                "holder must be non-empty and reserved_vram_bytes and ttl_ms must be positive",
            )
        with self._lock:
            now = time.monotonic_ns()
            self._expire_leases(now)
            used = sum(
                lease.reserved_vram_bytes
                for lease in self._active_leases.values()
                if lease.device_index == device_index
            )
            if used + reserved_vram_bytes > self.vram_capacity_bytes_per_device:
                raise GpuDispatchError(
                    "GPU_VRAM_EXHAUSTED",
                    f"device {device_index} has {used} bytes leased; cannot reserve {reserved_vram_bytes} within {self.vram_capacity_bytes_per_device}",
                )
            lease_id = _digest_obj(
                {
                    "holder": holder,
                    "device_index": device_index,
                    "reserved_vram_bytes": reserved_vram_bytes,
                    "acquired_ns": now,
                }
            )
            lease = GpuLease(
                lease_id=lease_id,
                holder=holder,
                device_index=device_index,
                reserved_vram_bytes=reserved_vram_bytes,
                acquired_ns=now,
                expires_ns=now + ttl_ms * 1_000_000,
            )
            self._active_leases[lease_id] = lease
            return lease

    def release_lease(self, lease_id: str) -> None:
        with self._lock:
            lease = self._active_leases.pop(lease_id, None)
            if lease is None:
                raise GpuDispatchError("UNKNOWN_LEASE", f"unknown or already released lease {lease_id}")

    def record_synchronized_measurement(
        self,
        *,
        lease: GpuLease,
        synchronized: bool,
        compile_ns: int,
        transfer_ns: int,
        steady_state_inference_ns: int,
        allocator_peak_bytes: int,
        device_total_nvml_bytes: int | None,
        requested_memory_scope: MemoryMeasurementScope,
        mix_compile_or_transfer_into_inference: bool = False,
    ) -> GpuResourceSample:
        """Records a GPU resource sample, rejecting unsynchronized timings or scope conflation."""
        with self._lock:
            now = time.monotonic_ns()
            self._expire_leases(now)
            active = self._active_leases.get(lease.lease_id)
            if active is None or active != lease:
                raise GpuDispatchError("INACTIVE_LEASE", "cannot record measurement for released, expired, or altered lease")
        if not synchronized:
            raise GpuDispatchError(
                "UNSYNCHRONIZED_GPU_TIMING",
                "unsynchronized asynchronous kernel launch timings are forbidden; synchronize stream/events first",
            )
        if mix_compile_or_transfer_into_inference:
            raise GpuDispatchError(
                "TIMING_PHASE_CONFLATION",
                "compilation and host-device transfer overhead must be accounted separately from steady-state inference",
            )
        if compile_ns < 0 or transfer_ns < 0 or steady_state_inference_ns <= 0:
            raise GpuDispatchError("INVALID_TIMING", "non-negative compile/transfer and positive inference timing required")
        if allocator_peak_bytes < 0:
            raise GpuDispatchError("INVALID_MEMORY", "allocator_peak_bytes must be non-negative")
        if requested_memory_scope == "device_total_nvml" and device_total_nvml_bytes is None:
            raise GpuDispatchError(
                "MEMORY_SCOPE_UNAVAILABLE",
                "cannot report device_total_nvml scope when only framework_allocator_peak is measured",
            )
        if (
            device_total_nvml_bytes is not None
            and device_total_nvml_bytes < allocator_peak_bytes
        ):
            raise GpuDispatchError(
                "MEMORY_SCOPE_INCONSISTENT",
                "device_total_nvml_bytes cannot be smaller than framework_allocator_peak_bytes",
            )

        sync_method: Literal["cuda_event_synchronize", "host_wall_clock_synced"] = (
            "cuda_event_synchronize" if torch.cuda.is_available() else "host_wall_clock_synced"
        )
        return GpuResourceSample(
            lease_id=lease.lease_id,
            device_index=lease.device_index,
            synchronized=True,
            synchronization_method=sync_method,
            compile_ns=compile_ns,
            transfer_ns=transfer_ns,
            steady_state_inference_ns=steady_state_inference_ns,
            allocator_peak_bytes=allocator_peak_bytes,
            device_total_nvml_bytes=device_total_nvml_bytes,
            memory_scope_reported=requested_memory_scope,
        )


@dataclass(frozen=True)
class GpuTransactionalExecutionReceipt:
    """Transactional receipt uniting Rust Weight IR compilation with GPU DLPack execution and smoke/eval verification."""

    transaction_digest: str
    rust_plan_digest: str
    rust_candidate_digest: str
    execution_device: str
    runtime_id: str
    dtype: SupportedGpuDType
    accumulation_dtype: str
    before_digest: str
    after_digest: str
    changed_parameters: int
    source_immutability_verified: bool
    cpu_gpu_parity_max_abs_diff: float
    cpu_gpu_parity_verified: bool
    smoke_probes_passed: bool
    eval_probe_outputs: tuple[list[float], ...]
    output_tensor: torch.Tensor
    # What can replay this edit. The edit graph runs only in f64 on the Rust CPU executor.
    edit_steps: Any = ()
    replay_precision: str = ""
    replay_device: str = ""
    replay_refused_reason: str = ""


def _gpu_replay_steps(
    source: torch.Tensor,
    tensor_name: str,
    operation: str,
    output: torch.Tensor,
    *,
    requested_dtype: str,
    accumulation_dtype: str,
    factor: float,
    delta: torch.Tensor | None,
    left_basis: torch.Tensor | None,
    right_basis: torch.Tensor | None,
) -> tuple[Any, str]:
    """Verified steps and no reason if the f64 Rust CPU executor reproduces this output bitwise, else a reason."""
    from .edit_plan import EditGraphError
    from .method_plans import MethodEditStep, verify_method_steps

    if requested_dtype != "f64" or accumulation_dtype != "f64":
        return (), f"The edit ran in {requested_dtype} with {accumulation_dtype} accumulation; the edit graph replays only f64"
    if operation == "add_delta" and delta is not None:
        operands = {"delta_matrix": delta.detach().cpu()}
    elif operation == "low_rank_update" and left_basis is not None and right_basis is not None:
        operands = {"left_factor": left_basis.detach().cpu(), "right_factor": right_basis.detach().cpu()}
    else:
        return (), f"Replay isn't wired for {operation}"
    try:
        step = MethodEditStep(tensor_name, f"gpu_{operation}", operation, operands, factor)
        return verify_method_steps({tensor_name: source}, (step,), {tensor_name: output}), ""
    except EditGraphError as exc:
        return (), f"The device output isn't bitwise equal to the f64 Rust CPU result: {exc}"


def execute_transactional_gpu_edit_and_eval(
    source_tensor: torch.Tensor,
    *,
    tensor_name: str = "w",
    operation: Literal["scale", "add_delta", "low_rank_update", "subspace_projection"],
    requested_device: str | None = None,
    requested_dtype: SupportedGpuDType = "bf16",
    accumulation_dtype: Literal["f32", "f64"] = "f32",
    factor: float = 1.0,
    delta: torch.Tensor | None = None,
    left_basis: torch.Tensor | None = None,
    right_basis: torch.Tensor | None = None,
    eval_inputs: Sequence[torch.Tensor] = (),
    max_abs_smoke_output: float = 100_000.0,
) -> GpuTransactionalExecutionReceipt:
    """Execute a Weight IR edit and evaluation on the GPU (or verified device stream) with identical transactional guarantees and receipts as the Rust CPU path."""
    from .ir import WeightIRTransactionBuilder, unpack_matrix_bytes

    device = requested_device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    before_source_digest = _tensor_bytes_digest(source_tensor)

    # 1. Execute via DLPack-verified device dispatch
    gpu_res = dispatch_weight_operation(
        source_tensor,
        tensor_name=tensor_name,
        operation=operation,
        requested_device=device,
        requested_dtype=requested_dtype,
        accumulation_dtype=accumulation_dtype,
        factor=factor,
        delta=delta,
        left_basis=left_basis,
        right_basis=right_basis,
    )

    # 2. Execute the identical Weight IR plan in the Rust surgeon for cross-backend transactional parity
    rust_tx = WeightIRTransactionBuilder.execute_matrix_edit(
        source_tensor.detach().cpu(),
        operation_type=operation,
        dtype=requested_dtype,
        accumulation_dtype=accumulation_dtype,
        factor=factor,
        delta_matrix=delta.detach().cpu() if delta is not None else None,
        left_factor=left_basis.detach().cpu() if left_basis is not None and operation == "low_rank_update" else None,
        right_factor=right_basis.detach().cpu() if right_basis is not None else None,
        orthonormal_basis=left_basis.detach().cpu() if left_basis is not None and operation == "subspace_projection" else None,
        provenance_tag="studio.gpu_transactional.v1",
    )
    from .ir import require_reload_parity
    require_reload_parity(rust_tx)
    shape = (int(source_tensor.shape[0]), int(source_tensor.shape[1]))
    rust_mat = torch.tensor(
        unpack_matrix_bytes(rust_tx.edited_bytes, shape, requested_dtype),
        dtype=torch.float64,
    )
    gpu_mat = gpu_res.output_tensor.detach().cpu().to(dtype=torch.float64)
    parity_diff = float((rust_mat - gpu_mat).abs().max().item())
    tol = 1e-2 if requested_dtype in ("f16", "bf16") else 1e-5
    if parity_diff > tol:
        raise GpuDispatchError(
            "BACKEND_PARITY_MISMATCH",
            f"Device output diverged from Rust Weight IR reference by {parity_diff:.4e} > {tol:.4e}",
        )

    # 3. Transactional smoke & evaluation pass on target device
    accum_t = torch.float32 if accumulation_dtype == "f32" else torch.float64
    w_eval = gpu_res.output_tensor.to(device=device, dtype=accum_t)
    probe_vecs = (
        list(eval_inputs)
        if eval_inputs
        else [torch.full((shape[1],), 1.0 / max(shape[1], 1), dtype=accum_t, device=device)]
    )
    eval_outputs: list[list[float]] = []
    for p_vec in probe_vecs:
        x = p_vec.to(device=device, dtype=accum_t)
        y = w_eval @ x
        if not bool(torch.isfinite(y).all().item()) or float(y.abs().max().item()) > max_abs_smoke_output:
            # Roll back: verify source_tensor is untouched before raising
            if _tensor_bytes_digest(source_tensor) != before_source_digest:
                raise GpuDispatchError("SOURCE_MUTATED", "source tensor was mutated during failed smoke probe")
            raise GpuDispatchError(
                "SMOKE_VERIFICATION_FAILED",
                f"Transactional smoke probe exceeded max_abs_smoke_output ({max_abs_smoke_output}) or was non-finite; candidate rolled back",
            )
        eval_outputs.append([round(float(v), 6) for v in y.detach().cpu().tolist()])

    source_unmodified = _tensor_bytes_digest(source_tensor) == before_source_digest
    runtime_id = "pytorch_cuda" if device.startswith("cuda") else "pytorch_cpu"
    tx_digest = _digest_obj(
        {
            "rust_plan_digest": rust_tx.plan_digest,
            "rust_candidate_digest": rust_tx.candidate_digest,
            "device": device,
            "runtime": runtime_id,
            "before_digest": gpu_res.before_digest,
            "after_digest": gpu_res.after_digest,
        }
    )
    replay_steps, replay_refused = _gpu_replay_steps(
        source_tensor,
        tensor_name,
        operation,
        gpu_mat,
        requested_dtype=requested_dtype,
        accumulation_dtype=accumulation_dtype,
        factor=factor,
        delta=delta,
        left_basis=left_basis,
        right_basis=right_basis,
    )
    return GpuTransactionalExecutionReceipt(
        edit_steps=replay_steps,
        replay_precision="f64" if replay_steps else "",
        replay_device="cpu_rust" if replay_steps else "",
        replay_refused_reason=replay_refused,
        transaction_digest=tx_digest,
        rust_plan_digest=rust_tx.plan_digest,
        rust_candidate_digest=rust_tx.candidate_digest,
        execution_device=device,
        runtime_id=runtime_id,
        dtype=requested_dtype,
        accumulation_dtype=accumulation_dtype,
        before_digest=gpu_res.before_digest,
        after_digest=gpu_res.after_digest,
        changed_parameters=rust_tx.changed_parameters,
        source_immutability_verified=source_unmodified,
        cpu_gpu_parity_max_abs_diff=round(parity_diff, 8),
        cpu_gpu_parity_verified=True,
        smoke_probes_passed=True,
        eval_probe_outputs=tuple(eval_outputs),
        output_tensor=gpu_res.output_tensor,
    )
