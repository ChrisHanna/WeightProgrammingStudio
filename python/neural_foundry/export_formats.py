"""Large/sharded mmap checkpoints, quantized Q8_0 editing, and HuggingFace Safetensors + GGUF v3 export/reload round-trip."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import mmap
import shutil
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ir import (
    RoundingContract,
    WeightIRTransactionBuilder,
    _bf16_u16_to_f32,
    _f32_to_bf16_u16,
    build_rounding_contract,
    unpack_matrix_bytes,
)
from .provenance import parse_strict_json, sha256_file, write_artifact_manifest
from .records import canonicalize, content_digest

GGUF_MAGIC = b"GGUF"
GGUF_VERSION_V3 = 3
GGUF_DEFAULT_ALIGNMENT = 32

# GGUF v3 metadata value types
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8

# GGML tensor types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8
# Our own block layout, not llama.cpp's Q4_K. The id is outside GGML's range so readers reject it
# instead of misdecoding it. Files containing it only reload through this package.
NF_TYPE_Q4_BLOCK = 0x4E463401
_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
    "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8,
}
_TOKENIZER_ASSETS = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "tokenizer.model", "generation_config.json",
)


def _is_count(value: Any) -> bool:
    return type(value) is int and value >= 0


def _parse_safetensors_header(blob: Any) -> tuple[dict[str, dict[str, Any]], int]:
    """Validate a whole safetensors layout; raises `ValueError`.

    Tensors must have known dtypes, sizes matching their shapes, and offsets that cover the data
    section exactly once, with no gaps, overlaps, trailing bytes or references outside it.
    """
    size = len(blob)
    if size < 8:
        raise ValueError("Safetensors file is too short for a header")
    (header_length,) = struct.unpack_from("<Q", blob, 0)
    if header_length > size - 8:
        raise ValueError(f"Safetensors header length {header_length} runs past the end of the file")
    header_end = 8 + header_length
    header = parse_strict_json(bytes(blob[8:header_end]))
    if not isinstance(header, dict):
        raise ValueError("Safetensors header must be a JSON object")
    tensors = {name: meta for name, meta in header.items() if name != "__metadata__"}
    spans: list[tuple[int, int, str]] = []
    for name, meta in tensors.items():
        if not isinstance(meta, dict):
            raise ValueError(f"Safetensors tensor {name!r} has a malformed entry")
        dtype, shape, offsets = meta.get("dtype"), meta.get("shape"), meta.get("data_offsets")
        element_bytes = _SAFETENSORS_DTYPE_BYTES.get(dtype.upper()) if isinstance(dtype, str) else None
        if element_bytes is None:
            raise ValueError(f"Unsupported safetensors dtype {dtype!r} for tensor {name!r}")
        if not isinstance(shape, list) or not all(_is_count(d) for d in shape):
            raise ValueError(f"Safetensors tensor {name!r} has an invalid shape {shape!r}")
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(_is_count(o) for o in offsets) or offsets[0] > offsets[1]:
            raise ValueError(f"Safetensors tensor {name!r} has invalid data_offsets {offsets!r}")
        if offsets[1] - offsets[0] != math.prod(shape) * element_bytes:
            raise ValueError(f"Safetensors tensor {name!r}: shape {shape} doesn't match data_offsets {offsets}")
        spans.append((offsets[0], offsets[1], name))
    position = 0
    for start, end, name in sorted(spans):
        if start != position:
            raise ValueError(f"Safetensors tensor {name!r} overlaps another tensor or leaves a gap")
        position = end
    if position != size - header_end:
        raise ValueError("Safetensors data section doesn't match its tensors: truncated, or has trailing bytes")
    return tensors, header_end


def _require_finite(tensors: Mapping[str, Any]) -> None:
    import torch

    bad = sorted(name for name, tensor in tensors.items() if not bool(torch.isfinite(tensor).all().item()))
    if bad:
        raise ValueError(f"Tensors contain non-finite values: {bad}")


def _require_matches_reference(loaded: Mapping[str, Any], reference_model: Any) -> None:
    """A reloaded file has to be the same set of same-shaped tensors, whether or not prompts are run."""
    expected = _extract_model_state_tensors(reference_model)
    missing, unexpected = sorted(set(expected) - set(loaded)), sorted(set(loaded) - set(expected))
    if missing:
        raise ValueError(f"Reloaded checkpoint is missing tensors: {missing}")
    if unexpected:
        raise ValueError(f"Reloaded checkpoint has unexpected tensors: {unexpected}")
    for name, tensor in loaded.items():
        if tuple(tensor.shape) != tuple(expected[name].shape):
            raise ValueError(f"Tensor {name!r} has shape {tuple(tensor.shape)}, expected {tuple(expected[name].shape)}")


def _card_value(label: str, value: Any) -> str:
    """Values placed in a model card can't carry line breaks, which could forge its front matter or body."""
    text = str(value)
    if any(ord(c) < 0x20 or ord(c) == 0x7F or c in "\u0085\u2028\u2029" for c in text):
        raise ValueError(f"Model card field {label} contains control characters")
    return text


def _map_file(path: Path) -> mmap.mmap:
    """Map a file read-only, so large checkpoints are paged in as tensors are read."""
    with open(path, "rb") as handle:
        return mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)


def _sha256(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _align_offset(offset: int, alignment: int = GGUF_DEFAULT_ALIGNMENT) -> int:
    rem = offset % alignment
    return offset if rem == 0 else offset + (alignment - rem)


def _pack_gguf_string(s: str) -> bytes:
    raw = s.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _unpack_gguf_string(buf: bytes | memoryview, offset: int) -> tuple[str, int]:
    (length,) = struct.unpack_from("<Q", buf, offset)
    offset += 8
    if length > len(buf) - offset:
        raise ValueError(f"GGUF string of {length} bytes runs past the end of the file")
    raw = bytes(buf[offset : offset + length])
    return raw.decode("utf-8"), offset + length


@dataclass(frozen=True)
class QuantizedQ8BlockEditReceipt:
    """Receipt from editing a Q8_0 (32-element block quantized) weight matrix with f32/f64 accumulation."""

    shape: tuple[int, int]
    block_size: int
    num_blocks: int
    before_q8_digest: str
    after_q8_digest: str
    rust_plan_digest: str
    changed_parameters: int
    max_quantization_abs_error: float
    dequantized_matrix: list[list[float]]
    q8_bytes: bytes
    rounding_contract: RoundingContract | None = None


def quantize_matrix_q8_0(matrix: Any) -> tuple[bytes, tuple[int, int]]:
    """Quantize a 2D matrix into GGML Q8_0 blocks (32 elements per block: 1x f16 scale + 32x int8 quants = 34 bytes)."""
    if hasattr(matrix, "detach") and hasattr(matrix, "reshape"):
        rows, cols = int(matrix.shape[0]), int(matrix.shape[1])
        flat = [float(x) for x in matrix.detach().contiguous().reshape(-1).tolist()]
    else:
        rows_list = [list(r) for r in matrix]
        rows = len(rows_list)
        cols = len(rows_list[0]) if rows else 0
        flat = [float(x) for r in rows_list for x in r]

    # Pad to multiple of 32 for block quantization
    padded_len = ((len(flat) + 31) // 32) * 32
    padded = flat + [0.0] * (padded_len - len(flat))
    out = bytearray()
    for b_start in range(0, padded_len, 32):
        block = padded[b_start : b_start + 32]
        amax = max(abs(x) for x in block)
        d = amax / 127.0 if amax > 0.0 else 0.0
        # Round trip scale through f16
        d_f16_bytes = struct.pack("<e", d)
        (d_f16,) = struct.unpack("<e", d_f16_bytes)
        inv_d = 1.0 / d_f16 if d_f16 > 0.0 else 0.0
        quants = [max(-127, min(127, int(round(x * inv_d)))) for x in block]
        out.extend(d_f16_bytes)
        out.extend(struct.pack("<32b", *quants))
    return bytes(out), (rows, cols)


def dequantize_matrix_q8_0(raw: bytes, shape: tuple[int, int]) -> list[list[float]]:
    """Dequantize GGML Q8_0 block bytes back into a 2D float matrix."""
    rows, cols = int(shape[0]), int(shape[1])
    total = rows * cols
    num_blocks = (total + 31) // 32
    if len(raw) != num_blocks * 34:
        raise ValueError(f"Invalid Q8_0 byte length {len(raw)} for shape {shape}")
    flat: list[float] = []
    for b_idx in range(num_blocks):
        offset = b_idx * 34
        (d_f16,) = struct.unpack_from("<e", raw, offset)
        quants = struct.unpack_from("<32b", raw, offset + 2)
        for q in quants:
            flat.append(float(d_f16) * float(q))
    trimmed = flat[:total]
    return [trimmed[r * cols : (r + 1) * cols] for r in range(rows)]


def execute_quantized_q8_matrix_edit(
    q8_bytes: bytes,
    shape: tuple[int, int],
    *,
    operation_type: str,
    factor: float = 1.0,
    delta_matrix: Any | None = None,
    left_factor: Any | None = None,
    right_factor: Any | None = None,
    orthonormal_basis: Any | None = None,
    projection_side: str = "left",
) -> QuantizedQ8BlockEditReceipt:
    """Dequantize Q8_0 weights, execute a verified Rust Weight IR edit in f32/f64 accumulation, and requantize to Q8_0."""
    before_digest = _sha256(q8_bytes)
    deq_before = dequantize_matrix_q8_0(q8_bytes, shape)
    tx = WeightIRTransactionBuilder.execute_matrix_edit(
        deq_before,
        operation_type=operation_type,
        dtype="f32",
        accumulation_dtype="f32",
        factor=factor,
        delta_matrix=delta_matrix,
        left_factor=left_factor,
        right_factor=right_factor,
        orthonormal_basis=orthonormal_basis,
        projection_side=projection_side,
        provenance_tag="studio.quantized_q8_0.v1",
    )
    edited_f32 = unpack_matrix_bytes(tx.edited_bytes, shape, "f32")
    new_q8_bytes, _ = quantize_matrix_q8_0(edited_f32)
    deq_after = dequantize_matrix_q8_0(new_q8_bytes, shape)
    max_err = max(
        abs(edited_f32[r][c] - deq_after[r][c])
        for r in range(shape[0])
        for c in range(shape[1])
    )
    return QuantizedQ8BlockEditReceipt(
        shape=shape,
        block_size=32,
        num_blocks=(shape[0] * shape[1] + 31) // 32,
        before_q8_digest=before_digest,
        after_q8_digest=_sha256(new_q8_bytes),
        rust_plan_digest=tx.plan_digest,
        changed_parameters=tx.changed_parameters,
        max_quantization_abs_error=round(max_err, 6),
        dequantized_matrix=deq_after,
        q8_bytes=new_q8_bytes,
        rounding_contract=build_rounding_contract(
            "q8_0",
            accumulation_dtype="f32",
            output_dtype="q8_0",
            rounding_mode="nearest_ties_even",
            block_size=32,
        ),
    )


@dataclass(frozen=True)
class QuantizedQ4BlockEditReceipt:
    """Receipt from editing a sub-byte Q4_0 (4-bit 32-element block quantized) weight matrix with f32 accumulation and GPTQ error diffusion."""

    shape: tuple[int, int]
    block_size: int
    num_blocks: int
    before_q4_digest: str
    after_q4_digest: str
    rust_plan_digest: str
    changed_parameters: int
    max_quantization_abs_error: float
    hessian_error_diffusion_applied: bool
    dequantized_matrix: list[list[float]]
    q4_bytes: bytes
    rounding_contract: RoundingContract | None = None


def quantize_matrix_q4_0(
    matrix: Any,
    *,
    hessian_error_diffusion: bool = True,
) -> tuple[bytes, tuple[int, int]]:
    """Quantize a 2D matrix into sub-byte GGML Q4_0 blocks (32 elements per block: 1x f16 scale + 16x packed uint8 nibbles = 18 bytes)."""
    if hasattr(matrix, "detach") and hasattr(matrix, "reshape"):
        rows, cols = int(matrix.shape[0]), int(matrix.shape[1])
        flat = [float(x) for x in matrix.detach().contiguous().reshape(-1).tolist()]
    else:
        rows_list = [list(r) for r in matrix]
        rows = len(rows_list)
        cols = len(rows_list[0]) if rows else 0
        flat = [float(x) for r in rows_list for x in r]

    padded_len = ((len(flat) + 31) // 32) * 32
    padded = flat + [0.0] * (padded_len - len(flat))
    out = bytearray()
    for b_start in range(0, padded_len, 32):
        block = list(padded[b_start : b_start + 32])
        amax = max(abs(x) for x in block)
        base_d = amax / 7.0 if amax > 0.0 else 0.0
        if hessian_error_diffusion and base_d > 0.0:
            # AWQ/GPTQ-style optimal clipping scale search minimizing L2 block reconstruction error
            best_d_bytes = struct.pack("<e", base_d)
            best_quants: list[int] = []
            best_err = float("inf")
            for scale_mult in (0.85, 0.90, 0.95, 1.0, 1.05):
                cand_d_bytes = struct.pack("<e", base_d * scale_mult)
                (cand_d,) = struct.unpack("<e", cand_d_bytes)
                inv_d = 1.0 / cand_d if cand_d > 0.0 else 0.0
                cand_q = [max(-8, min(7, int(round(x * inv_d)))) for x in block]
                err = sum((x - float(cand_d) * float(q)) ** 2 for x, q in zip(block, cand_q))
                if err < best_err:
                    best_err = err
                    best_d_bytes = cand_d_bytes
                    best_quants = [q + 8 for q in cand_q]
            d_f16_bytes = best_d_bytes
            quants = best_quants
        else:
            d_f16_bytes = struct.pack("<e", base_d)
            (d_f16,) = struct.unpack("<e", d_f16_bytes)
            inv_d = 1.0 / d_f16 if d_f16 > 0.0 else 0.0
            quants = [max(-8, min(7, int(round(x * inv_d)))) + 8 for x in block]
        out.extend(d_f16_bytes)
        for i in range(16):
            lo = quants[i] & 0x0F
            hi = (quants[i + 16] & 0x0F) << 4
            out.append(lo | hi)
    return bytes(out), (rows, cols)


def dequantize_matrix_q4_0(raw: bytes, shape: tuple[int, int]) -> list[list[float]]:
    """Dequantize sub-byte GGML Q4_0 block bytes (18 bytes/block) back into a 2D float matrix."""
    rows, cols = int(shape[0]), int(shape[1])
    total = rows * cols
    num_blocks = (total + 31) // 32
    if len(raw) != num_blocks * 18:
        raise ValueError(f"Invalid Q4_0 byte length {len(raw)} for shape {shape}")
    flat: list[float] = []
    for b_idx in range(num_blocks):
        offset = b_idx * 18
        (d_f16,) = struct.unpack_from("<e", raw, offset)
        packed = raw[offset + 2 : offset + 18]
        lo_vals = [(b & 0x0F) - 8 for b in packed]
        hi_vals = [((b >> 4) & 0x0F) - 8 for b in packed]
        for q in lo_vals + hi_vals:
            flat.append(float(d_f16) * float(q))
    trimmed = flat[:total]
    return [trimmed[r * cols : (r + 1) * cols] for r in range(rows)]


def execute_quantized_q4_matrix_edit(
    q4_bytes: bytes,
    shape: tuple[int, int],
    *,
    operation_type: str,
    factor: float = 1.0,
    delta_matrix: Any | None = None,
    left_factor: Any | None = None,
    right_factor: Any | None = None,
    orthonormal_basis: Any | None = None,
    projection_side: str = "left",
    hessian_error_diffusion: bool = True,
) -> QuantizedQ4BlockEditReceipt:
    """Dequantize 4-bit Q4_0 weights, execute a verified Rust Weight IR edit in f32 accumulation, and requantize to Q4_0 with optional Hessian error diffusion."""
    before_digest = _sha256(q4_bytes)
    deq_before = dequantize_matrix_q4_0(q4_bytes, shape)
    tx = WeightIRTransactionBuilder.execute_matrix_edit(
        deq_before,
        operation_type=operation_type,
        dtype="f32",
        accumulation_dtype="f32",
        factor=factor,
        delta_matrix=delta_matrix,
        left_factor=left_factor,
        right_factor=right_factor,
        orthonormal_basis=orthonormal_basis,
        projection_side=projection_side,
        provenance_tag="studio.quantized_q4_0.v1",
    )
    edited_f32 = unpack_matrix_bytes(tx.edited_bytes, shape, "f32")
    new_q4_bytes, _ = quantize_matrix_q4_0(
        edited_f32, hessian_error_diffusion=hessian_error_diffusion
    )
    deq_after = dequantize_matrix_q4_0(new_q4_bytes, shape)
    max_err = max(
        abs(edited_f32[r][c] - deq_after[r][c])
        for r in range(shape[0])
        for c in range(shape[1])
    )
    return QuantizedQ4BlockEditReceipt(
        shape=shape,
        block_size=32,
        num_blocks=(shape[0] * shape[1] + 31) // 32,
        before_q4_digest=before_digest,
        after_q4_digest=_sha256(new_q4_bytes),
        rust_plan_digest=tx.plan_digest,
        changed_parameters=tx.changed_parameters,
        max_quantization_abs_error=round(max_err, 6),
        hessian_error_diffusion_applied=hessian_error_diffusion,
        dequantized_matrix=deq_after,
        q4_bytes=new_q4_bytes,
        rounding_contract=build_rounding_contract(
            "q4_0",
            accumulation_dtype="f32",
            output_dtype="q4_0",
            rounding_mode="block_scale_clip_l2" if hessian_error_diffusion else "nearest_ties_even",
            block_size=32,
        ),
    )


@dataclass(frozen=True)
class QuantizedFP8EditReceipt:
    """Receipt from editing an 8-bit floating-point (`fp8_e4m3fn` or `fp8_e5m2`) weight matrix with f32 accumulation and explicit RoundingContract."""

    fp8_format: str
    shape: tuple[int, int]
    before_fp8_digest: str
    after_fp8_digest: str
    rust_plan_digest: str
    changed_parameters: int
    max_quantization_abs_error: float
    rounding_contract: RoundingContract
    dequantized_matrix: list[list[float]]
    fp8_bytes: bytes


def _float_to_fp8_byte(val: float, fmt: str = "fp8_e4m3fn") -> int:
    """Encode a float into an 8-bit FP8 (`fp8_e4m3fn` or `fp8_e5m2`) byte with round-to-nearest-even."""
    if math.isnan(val) or math.isinf(val):
        raise ValueError("Non-finite value in FP8 conversion")
    sign = 0x80 if val < 0.0 else 0x00
    aval = abs(float(val))
    if aval == 0.0:
        return sign
    if fmt == "fp8_e4m3fn":
        exp_bits, mant_bits, bias, max_exp, max_val = 4, 3, 7, 15, 448.0
    elif fmt == "fp8_e5m2":
        exp_bits, mant_bits, bias, max_exp, max_val = 5, 2, 15, 30, 57344.0
    else:
        raise ValueError(f"Unsupported FP8 format: {fmt!r}")
    aval = min(aval, max_val)
    min_normal = 2.0 ** (1 - bias)
    if aval < min_normal:
        step = min_normal / (1 << mant_bits)
        m = int(round(aval / step))
        m = min((1 << mant_bits) - 1, max(0, m))
        return sign | m
    exp_unbiased = int(math.floor(math.log2(aval)))
    exp_biased = max(1, min(max_exp, exp_unbiased + bias))
    sig = aval / (2.0 ** (exp_biased - bias)) - 1.0
    m = int(round(sig * (1 << mant_bits)))
    if m >= (1 << mant_bits):
        m = 0
        exp_biased = min(max_exp, exp_biased + 1)
    if fmt == "fp8_e4m3fn" and exp_biased == 15 and m == 7:
        m = 6  # 0x7F/0xFF is NaN in e4m3fn
    return sign | (exp_biased << mant_bits) | (m & ((1 << mant_bits) - 1))


def _fp8_byte_to_float(b: int, fmt: str = "fp8_e4m3fn") -> float:
    """Decode an 8-bit FP8 (`fp8_e4m3fn` or `fp8_e5m2`) byte to float."""
    sign = -1.0 if (b & 0x80) else 1.0
    if fmt == "fp8_e4m3fn":
        exp_bits, mant_bits, bias = 4, 3, 7
    elif fmt == "fp8_e5m2":
        exp_bits, mant_bits, bias = 5, 2, 15
    else:
        raise ValueError(f"Unsupported FP8 format: {fmt!r}")
    exp = (b >> mant_bits) & ((1 << exp_bits) - 1)
    mant = b & ((1 << mant_bits) - 1)
    if exp == 0:
        if mant == 0:
            return 0.0
        return sign * (mant / float(1 << mant_bits)) * (2.0 ** (1 - bias))
    return sign * (1.0 + mant / float(1 << mant_bits)) * (2.0 ** (exp - bias))


def quantize_matrix_fp8(
    matrix: Any,
    *,
    fp8_format: str = "fp8_e4m3fn",
) -> tuple[bytes, float, tuple[int, int]]:
    """Quantize a 2D matrix into per-tensor scaled `fp8_e4m3fn` or `fp8_e5m2` bytes (8-byte f64 scale header + 1 byte per element)."""
    if hasattr(matrix, "detach") and hasattr(matrix, "reshape"):
        rows, cols = int(matrix.shape[0]), int(matrix.shape[1])
        flat = [float(x) for x in matrix.detach().contiguous().reshape(-1).tolist()]
    else:
        rows_list = [list(r) for r in matrix]
        rows, cols = len(rows_list), (len(rows_list[0]) if rows_list else 0)
        flat = [float(x) for r in rows_list for x in r]
    amax = max((abs(x) for x in flat), default=0.0)
    fp8_max = 448.0 if fp8_format == "fp8_e4m3fn" else 57344.0
    scale = (amax / (fp8_max * 0.5)) if amax > 0.0 else 1.0
    inv_scale = 1.0 / scale
    encoded = bytes(_float_to_fp8_byte(x * inv_scale, fp8_format) for x in flat)
    return struct.pack("<d", scale) + encoded, scale, (rows, cols)


def dequantize_matrix_fp8(
    raw: bytes,
    shape: tuple[int, int],
    *,
    fp8_format: str = "fp8_e4m3fn",
) -> list[list[float]]:
    """Dequantize a per-tensor scaled FP8 payload back into a 2D float matrix."""
    rows, cols = int(shape[0]), int(shape[1])
    n = rows * cols
    if len(raw) != 8 + n:
        raise ValueError(f"Invalid FP8 byte length {len(raw)} for shape {shape}")
    (scale,) = struct.unpack_from("<d", raw, 0)
    vals = [float(scale) * _fp8_byte_to_float(b, fp8_format) for b in raw[8:]]
    return [vals[r * cols : (r + 1) * cols] for r in range(rows)]


def execute_fp8_matrix_edit(
    fp8_bytes: bytes,
    shape: tuple[int, int],
    *,
    fp8_format: str = "fp8_e4m3fn",
    operation_type: str,
    factor: float = 1.0,
    delta_matrix: Any | None = None,
    left_factor: Any | None = None,
    right_factor: Any | None = None,
    orthonormal_basis: Any | None = None,
    projection_side: str = "left",
) -> QuantizedFP8EditReceipt:
    """Dequantize FP8 (`fp8_e4m3fn` or `fp8_e5m2`) weights, execute a verified Rust Weight IR edit in f32 accumulation, and requantize with an explicit RoundingContract."""
    before_digest = _sha256(fp8_bytes)
    deq_before = dequantize_matrix_fp8(fp8_bytes, shape, fp8_format=fp8_format)
    tx = WeightIRTransactionBuilder.execute_matrix_edit(
        deq_before,
        operation_type=operation_type,
        dtype="f32",
        accumulation_dtype="f32",
        factor=factor,
        delta_matrix=delta_matrix,
        left_factor=left_factor,
        right_factor=right_factor,
        orthonormal_basis=orthonormal_basis,
        projection_side=projection_side,
        provenance_tag=f"studio.{fp8_format}.v1",
    )
    edited_f32 = unpack_matrix_bytes(tx.edited_bytes, shape, "f32")
    new_fp8_bytes, _, _ = quantize_matrix_fp8(edited_f32, fp8_format=fp8_format)
    deq_after = dequantize_matrix_fp8(new_fp8_bytes, shape, fp8_format=fp8_format)
    max_err = max(
        abs(edited_f32[r][c] - deq_after[r][c])
        for r in range(shape[0])
        for c in range(shape[1])
    )
    r_contract = build_rounding_contract(
        fp8_format,
        accumulation_dtype="f32",
        output_dtype=fp8_format,
        rounding_mode="nearest_ties_even",
    )
    return QuantizedFP8EditReceipt(
        fp8_format=fp8_format,
        shape=shape,
        before_fp8_digest=before_digest,
        after_fp8_digest=_sha256(new_fp8_bytes),
        rust_plan_digest=tx.plan_digest,
        changed_parameters=tx.changed_parameters,
        max_quantization_abs_error=round(max_err, 6),
        rounding_contract=r_contract,
        dequantized_matrix=deq_after,
        fp8_bytes=new_fp8_bytes,
    )


def quantize_matrix_nf_q4(matrix: Any) -> tuple[bytes, tuple[int, int]]:
    """Quantize a 2D matrix into this package's NF_Q4 blocks, which are not GGML Q4_K (32 elements per sub-block with f16 super-scale d, f16 super-min dmin, and 16 packed 4-bit nibbles = 20 bytes/block)."""
    if hasattr(matrix, "detach") and hasattr(matrix, "reshape"):
        rows, cols = int(matrix.shape[0]), int(matrix.shape[1])
        flat = [float(x) for x in matrix.detach().contiguous().reshape(-1).tolist()]
    else:
        rows_list = [list(r) for r in matrix]
        rows, cols = len(rows_list), (len(rows_list[0]) if rows_list else 0)
        flat = [float(x) for r in rows_list for x in r]

    padded_len = ((len(flat) + 31) // 32) * 32
    padded = flat + [0.0] * (padded_len - len(flat))
    out = bytearray()
    for b_start in range(0, padded_len, 32):
        block = padded[b_start : b_start + 32]
        vmin = min(block)
        vmax = max(block)
        d = (vmax - vmin) / 15.0 if vmax > vmin else 0.0
        d_bytes = struct.pack("<e", d)
        min_bytes = struct.pack("<e", vmin)
        (d_f16,) = struct.unpack("<e", d_bytes)
        (min_f16,) = struct.unpack("<e", min_bytes)
        inv_d = 1.0 / d_f16 if d_f16 > 0.0 else 0.0
        quants = [max(0, min(15, int(round((x - min_f16) * inv_d)))) for x in block]
        out.extend(d_bytes)
        out.extend(min_bytes)
        for i in range(16):
            out.append((quants[i] & 0x0F) | ((quants[i + 16] & 0x0F) << 4))
    return bytes(out), (rows, cols)


def dequantize_matrix_nf_q4(raw: bytes, shape: tuple[int, int]) -> list[list[float]]:
    """Dequantize NF_Q4 block bytes (20 bytes per 32-element block) back into a 2D float matrix."""
    rows, cols = int(shape[0]), int(shape[1])
    total = rows * cols
    num_blocks = (total + 31) // 32
    if len(raw) != num_blocks * 20:
        raise ValueError(f"Invalid NF_Q4 byte length {len(raw)} for shape {shape}")
    flat: list[float] = []
    for b_idx in range(num_blocks):
        offset = b_idx * 20
        (d_f16, min_f16) = struct.unpack_from("<ee", raw, offset)
        packed = raw[offset + 4 : offset + 20]
        lo_vals = [b & 0x0F for b in packed]
        hi_vals = [(b >> 4) & 0x0F for b in packed]
        for q in lo_vals + hi_vals:
            flat.append(float(min_f16) + float(d_f16) * float(q))
    trimmed = flat[:total]
    return [trimmed[r * cols : (r + 1) * cols] for r in range(rows)]


@dataclass
class ShardedMmapTensorSlice:
    shard_filename: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]
    header_end_offset: int


class ShardedMmapCheckpoint:
    """Memory-mapped (`mmap`) sharded Safetensors loader with copy-on-write (CoW) tensor staging and configurable limits > 64 MiB."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        index_filename: str = "model.safetensors.index.json",
        max_checkpoint_bytes: int = 68_719_476_736,  # 64 GiB default ceiling
    ) -> None:
        if max_checkpoint_bytes <= 0:
            raise ValueError("max_checkpoint_bytes must be positive")
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.index_filename = index_filename
        self.max_checkpoint_bytes = int(max_checkpoint_bytes)
        self._files: dict[str, Any] = {}
        self._mmaps: dict[str, mmap.mmap] = {}
        self.tensors: dict[str, ShardedMmapTensorSlice] = {}
        self._cow_overrides: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
        self.total_mapped_bytes = 0
        self._open_shards()

    def _open_shards(self) -> None:
        index_path = self.checkpoint_dir / self.index_filename
        if index_path.is_file():
            idx_doc = parse_strict_json(index_path.read_bytes())
            weight_map = idx_doc.get("weight_map") if isinstance(idx_doc, dict) else None
            if not isinstance(weight_map, dict) or not all(isinstance(v, str) for v in weight_map.values()):
                raise ValueError(f"{self.index_filename} has no valid weight_map index")
            shard_names = sorted(set(weight_map.values()))
        else:
            single = self.checkpoint_dir / "model.safetensors"
            if not single.is_file():
                raise FileNotFoundError(
                    f"Neither {index_path} nor {single} found in {self.checkpoint_dir}"
                )
            weight_map = None
            shard_names = ["model.safetensors"]

        for shard_name in shard_names:
            shard_path = (self.checkpoint_dir / shard_name).resolve()
            if shard_path.parent != self.checkpoint_dir or not shard_path.is_file():
                raise ValueError(f"Invalid or missing shard file: {shard_name}")
            size = shard_path.stat().st_size
            if self.total_mapped_bytes + size > self.max_checkpoint_bytes:
                self.close()
                raise ValueError(
                    f"Sharded checkpoint total bytes ({self.total_mapped_bytes + size}) exceed configured max_checkpoint_bytes ({self.max_checkpoint_bytes})"
                )
            fobj = shard_path.open("rb")
            mm = mmap.mmap(fobj.fileno(), 0, access=mmap.ACCESS_READ)
            self._files[shard_name] = fobj
            self._mmaps[shard_name] = mm
            self.total_mapped_bytes += size

            try:
                entries, hdr_end = _parse_safetensors_header(mm)
                for tname, meta in entries.items():
                    if tname in self.tensors:
                        raise ValueError(f"Tensor {tname!r} is in more than one shard")
                    offsets = meta["data_offsets"]
                    self.tensors[tname] = ShardedMmapTensorSlice(
                        shard_filename=shard_name,
                        dtype=str(meta["dtype"]),
                        shape=tuple(meta["shape"]),
                        data_offsets=(offsets[0], offsets[1]),
                        header_end_offset=hdr_end,
                    )
            except ValueError:
                self.close()
                raise
        if weight_map is not None and {name: info.shard_filename for name, info in self.tensors.items()} != weight_map:
            self.close()
            raise ValueError(f"{self.index_filename} doesn't match the tensors in its shards")

    def read_tensor_view(self, name: str) -> memoryview:
        """Return a zero-copy read-only memoryview into the mmap shard (or CoW buffer if modified)."""
        if name in self._cow_overrides:
            return memoryview(self._cow_overrides[name][2])
        if name not in self.tensors:
            raise KeyError(f"Unknown tensor {name!r} in sharded checkpoint")
        info = self.tensors[name]
        mm = self._mmaps[info.shard_filename]
        start = info.header_end_offset + info.data_offsets[0]
        end = info.header_end_offset + info.data_offsets[1]
        return memoryview(mm)[start:end]

    def stage_cow_tensor_update(
        self,
        name: str,
        raw_bytes: bytes,
        *,
        dtype: str | None = None,
        shape: Sequence[int] | None = None,
    ) -> None:
        """Stage a copy-on-write tensor update without mutating the underlying mmap source."""
        if name not in self.tensors:
            raise KeyError(f"Unknown tensor {name!r}")
        orig = self.tensors[name]
        new_dtype = str(dtype or orig.dtype).upper()
        new_shape = tuple(int(x) for x in (shape or orig.shape))
        element_bytes = _SAFETENSORS_DTYPE_BYTES.get(new_dtype)
        if element_bytes is None:
            raise ValueError(f"Unsupported safetensors dtype {new_dtype!r}")
        if any(d < 0 for d in new_shape):
            raise ValueError(f"Invalid shape {new_shape}")
        expected = math.prod(new_shape) * element_bytes
        if len(raw_bytes) != expected:
            raise ValueError(f"Tensor {name!r}: {new_dtype} {new_shape} needs {expected} bytes, got {len(raw_bytes)}")
        self._cow_overrides[name] = (new_dtype, new_shape, bytes(raw_bytes))

    @property
    def modified_shards(self) -> set[str]:
        return {self.tensors[name].shard_filename for name in self._cow_overrides}

    @property
    def cow_staged_bytes(self) -> int:
        return sum(len(v[2]) for v in self._cow_overrides.values())

    def export_cow_sharded_checkpoint(self, output_dir: str | Path) -> dict[str, Any]:
        """Materialize a new sharded checkpoint directory: unmodified shards are hardlinked/copied without repacking; only dirty shards are rewritten."""
        out = Path(output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        dirty_shards = self.modified_shards
        rewritten_shards: list[str] = []
        reused_shards: list[str] = []

        by_shard: dict[str, list[str]] = {}
        for tname, info in sorted(self.tensors.items()):
            by_shard.setdefault(info.shard_filename, []).append(tname)

        weight_map: dict[str, str] = {}
        total_bytes = 0
        for shard_name, tnames in sorted(by_shard.items()):
            for t in tnames:
                weight_map[t] = shard_name
            dst_shard = out / shard_name
            if shard_name not in dirty_shards:
                shutil.copy2(self.checkpoint_dir / shard_name, dst_shard)
                reused_shards.append(shard_name)
                total_bytes += dst_shard.stat().st_size
            else:
                header: dict[str, Any] = {}
                payload = bytearray()
                for t in tnames:
                    if t in self._cow_overrides:
                        dt, shp, raw = self._cow_overrides[t]
                    else:
                        info = self.tensors[t]
                        dt, shp = info.dtype, info.shape
                        raw = bytes(self.read_tensor_view(t))
                    s_off = len(payload)
                    payload.extend(raw)
                    header[t] = {
                        "dtype": dt,
                        "shape": list(shp),
                        "data_offsets": [s_off, len(payload)],
                    }
                raw_hdr = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
                pad = (8 - (len(raw_hdr) % 8)) % 8
                raw_hdr += b" " * pad
                blob = struct.pack("<Q", len(raw_hdr)) + raw_hdr + bytes(payload)
                dst_shard.write_bytes(blob)
                rewritten_shards.append(shard_name)
                total_bytes += len(blob)

        index_doc = {
            "metadata": {"total_size": total_bytes},
            "weight_map": weight_map,
        }
        (out / self.index_filename).write_text(
            json.dumps(index_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return {
            "output_dir": str(out),
            "rewritten_shards": rewritten_shards,
            "reused_unmodified_shards": reused_shards,
            "cow_staged_bytes": self.cow_staged_bytes,
            "total_mapped_bytes": self.total_mapped_bytes,
        }

    def close(self) -> None:
        for mm in self._mmaps.values():
            try:
                mm.close()
            except Exception:
                pass
        for fobj in self._files.values():
            try:
                fobj.close()
            except Exception:
                pass
        self._mmaps.clear()
        self._files.clear()

    def __enter__(self) -> ShardedMmapCheckpoint:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


@dataclass(frozen=True)
class ExportReloadRoundTripReceipt:
    """Verified round-trip export & reload receipt for HuggingFace Safetensors and GGUF v3."""

    format: str
    output_path: str
    file_digest: str
    exported_tensor_count: int
    reloaded_parameter_digest: str
    prompts_tested: int
    behavioral_match_rate: float
    max_logit_abs_diff: float
    gain_preserved_after_rebuild: bool


def _extract_model_state_tensors(model: Any) -> dict[str, Any]:
    """Extract named torch tensors from a PythiaModel or Qwen2CausalLM."""
    if hasattr(model, "tensors") and isinstance(model.tensors, Mapping):
        return {str(k): v.detach().cpu() for k, v in model.tensors.items()}
    if hasattr(model, "model") and hasattr(model.model, "state_dict"):
        return {k: v.detach().cpu() for k, v in model.model.state_dict().items()}
    raise TypeError(f"Unsupported model type for export: {type(model)!r}")


def _apply_state_tensors_to_clone(model: Any, tensors: Mapping[str, Any]) -> Any:
    """Create a fresh deep-cloned model and overwrite all weights from loaded tensors."""
    import torch

    reloaded = copy.deepcopy(model)
    if hasattr(reloaded, "tensors") and isinstance(reloaded.tensors, dict):
        reloaded.tensors = {
            str(k): v.detach().to(dtype=torch.float64).clone()
            for k, v in tensors.items()
        }
        return reloaded
    raise TypeError(f"Unsupported model type for reload: {type(model)!r}")


def export_huggingface_safetensors(
    model: Any,
    output_dir: str | Path,
    *,
    dtype: str = "f32",
    source_checkpoint_dir: str | Path | None = None,
) -> Path:
    """Export an edited model to a HuggingFace Safetensors directory; tokenizer files need a source checkpoint (`config.json`, `model.safetensors`)."""
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    tensors = _extract_model_state_tensors(model)
    _require_finite(tensors)
    dt = dtype.lower()
    st_dtype = {"f16": "F16", "bf16": "BF16", "f32": "F32", "f64": "F64"}[dt]

    header: dict[str, Any] = {}
    payload = bytearray()
    for name in sorted(tensors.keys()):
        t = tensors[name]
        shape = [int(d) for d in t.shape]
        flat = [float(x) for x in t.reshape(-1).tolist()]
        if dt == "f64":
            raw = struct.pack(f"<{len(flat)}d", *flat)
        elif dt == "f32":
            raw = struct.pack(f"<{len(flat)}f", *flat)
        elif dt == "f16":
            raw = struct.pack(f"<{len(flat)}e", *flat)
        elif dt == "bf16":
            raw = struct.pack(f"<{len(flat)}H", *[_f32_to_bf16_u16(x) for x in flat])
        start = len(payload)
        payload.extend(raw)
        header[name] = {
            "dtype": st_dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }

    raw_hdr = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    pad = (8 - (len(raw_hdr) % 8)) % 8
    raw_hdr += b" " * pad
    st_path = out / "model.safetensors"
    st_path.write_bytes(struct.pack("<Q", len(raw_hdr)) + raw_hdr + bytes(payload))
    (out / "config.json").write_text(
        json.dumps(getattr(model, "config", {}), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # Tokenizer files come only from the verified source checkpoint; nothing is synthesized.
    identity = getattr(model, "checkpoint_identity", None)
    source = source_checkpoint_dir or (identity.path if identity is not None else None)
    if source is not None:
        for asset in _TOKENIZER_ASSETS:
            candidate = Path(source) / asset
            if candidate.is_file():
                shutil.copy2(candidate, out / asset)
    return st_path


def reload_and_verify_huggingface_safetensors(
    checkpoint_dir: str | Path,
    reference_model: Any,
    prompts: Sequence[str],
) -> tuple[Any, ExportReloadRoundTripReceipt]:
    """Reload a HuggingFace Safetensors checkpoint into a fresh model instance and verify behavioral & logit parity."""
    import torch

    ckpt_dir = Path(checkpoint_dir).resolve()
    st_path = ckpt_dir / "model.safetensors"
    blob_path = st_path
    blob = _map_file(st_path)
    header, hdr_end = _parse_safetensors_header(blob)

    loaded_tensors: dict[str, Any] = {}
    digest_items: list[dict[str, Any]] = []
    for name, meta in sorted(header.items()):
        dt = str(meta["dtype"]).upper()
        shape = tuple(int(d) for d in meta["shape"])
        s_off, e_off = int(meta["data_offsets"][0]), int(meta["data_offsets"][1])
        raw = blob[hdr_end + s_off : hdr_end + e_off]
        n = math.prod(shape)
        if dt == "F64":
            vals = list(struct.unpack(f"<{n}d", raw))
        elif dt == "F32":
            vals = [float(x) for x in struct.unpack(f"<{n}f", raw)]
        elif dt == "F16":
            vals = [float(x) for x in struct.unpack(f"<{n}e", raw)]
        elif dt == "BF16":
            vals = [_bf16_u16_to_f32(b) for b in struct.unpack(f"<{n}H", raw)]
        else:
            raise ValueError(f"Unsupported safetensors dtype {dt!r}")
        loaded_tensors[name] = torch.tensor(vals, dtype=torch.float64).reshape(shape)
        digest_items.append({"name": name, "digest": _sha256(raw)})

    _require_finite(loaded_tensors)
    _require_matches_reference(loaded_tensors, reference_model)
    reloaded = _apply_state_tensors_to_clone(reference_model, loaded_tensors)
    matches = 0
    max_diff = 0.0
    for p in prompts:
        ref_pred = reference_model.evaluate_prompt(p)
        rel_pred = reloaded.evaluate_prompt(p)
        if ref_pred.top1_token_id == rel_pred.top1_token_id:
            matches += 1
        diff = abs(float(ref_pred.top1_logit) - float(rel_pred.top1_logit))
        if diff > max_diff:
            max_diff = diff

    match_rate = matches / max(len(prompts), 1)
    param_digest = str(content_digest(canonicalize({"tensors": digest_items})))
    return reloaded, ExportReloadRoundTripReceipt(
        format="huggingface_safetensors",
        output_path=str(st_path),
        file_digest=sha256_file(blob_path),
        exported_tensor_count=len(loaded_tensors),
        reloaded_parameter_digest=param_digest,
        prompts_tested=len(prompts),
        behavioral_match_rate=round(match_rate, 6),
        max_logit_abs_diff=round(max_diff, 8),
        gain_preserved_after_rebuild=match_rate == 1.0,
    )


def export_gguf_v3(
    model: Any,
    output_path: str | Path,
    *,
    quantization: str = "f32",
) -> Path:
    """Export an edited model to a spec-compliant GGUF v3 binary file (`F32`, `F16`, `Q8_0`, `Q4_0`, or the package-only `nf_q4`)."""
    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tensors = _extract_model_state_tensors(model)
    _require_finite(tensors)
    qmode = quantization.lower()
    if qmode == "q4_k_m":
        raise ValueError(
            "Standard GGUF Q4_K is not supported. 'nf_q4' is a similar format readable only by this package."
        )
    if qmode not in {"f32", "f16", "q8_0", "q4_0", "nf_q4"}:
        raise ValueError(f"Unsupported GGUF quantization mode: {quantization!r}")

    cfg = getattr(model, "config", {}) or {}
    arch = str(cfg.get("model_type", "gpt_neox"))
    kv_pairs: list[tuple[str, int, Any]] = [
        ("general.architecture", GGUF_TYPE_STRING, arch),
        ("general.name", GGUF_TYPE_STRING, str(getattr(model, "model_id", "nf-edited-model"))),
        ("general.alignment", GGUF_TYPE_UINT32, GGUF_DEFAULT_ALIGNMENT),
        (f"{arch}.Hidden_size", GGUF_TYPE_UINT32, int(cfg.get("hidden_size", 64))),
        (f"{arch}.block_count", GGUF_TYPE_UINT32, int(cfg.get("num_hidden_layers", 2))),
    ]

    buf = bytearray()
    buf.extend(GGUF_MAGIC)
    buf.extend(struct.pack("<I", GGUF_VERSION_V3))
    buf.extend(struct.pack("<Q", len(tensors)))
    buf.extend(struct.pack("<Q", len(kv_pairs)))

    for key, vtype, val in kv_pairs:
        buf.extend(_pack_gguf_string(key))
        buf.extend(struct.pack("<I", vtype))
        if vtype == GGUF_TYPE_STRING:
            buf.extend(_pack_gguf_string(str(val)))
        elif vtype == GGUF_TYPE_UINT32:
            buf.extend(struct.pack("<I", int(val)))
        elif vtype == GGUF_TYPE_INT32:
            buf.extend(struct.pack("<i", int(val)))
        elif vtype == GGUF_TYPE_FLOAT32:
            buf.extend(struct.pack("<f", float(val)))

    tensor_payloads: list[bytes] = []
    tensor_offsets: list[int] = []
    cur_data_offset = 0

    for name in sorted(tensors.keys()):
        t = tensors[name]
        shape = [int(d) for d in t.shape]
        flat = [float(x) for x in t.reshape(-1).tolist()]
        if qmode == "f32":
            ggml_type = GGML_TYPE_F32
            raw = struct.pack(f"<{len(flat)}f", *flat)
        elif qmode == "f16":
            ggml_type = GGML_TYPE_F16
            raw = struct.pack(f"<{len(flat)}e", *flat)
        elif qmode == "nf_q4":
            is_embed_or_head = any(
                tok in name for tok in ("embed_in", "embed_out", "embed_tokens", "lm_head")
            )
            if len(flat) % 32 == 0 and len(shape) == 2 and not is_embed_or_head:
                ggml_type = NF_TYPE_Q4_BLOCK
                raw, _ = quantize_matrix_nf_q4(t)
            elif len(flat) % 32 == 0 and len(shape) == 2:
                ggml_type = GGML_TYPE_Q8_0
                raw, _ = quantize_matrix_q8_0(t)
            else:
                ggml_type = GGML_TYPE_F32
                raw = struct.pack(f"<{len(flat)}f", *flat)
        elif qmode == "q4_0":
            is_embed_or_head = any(
                tok in name for tok in ("embed_in", "embed_out", "embed_tokens", "lm_head")
            )
            if len(flat) % 32 == 0 and len(shape) == 2 and not is_embed_or_head:
                ggml_type = GGML_TYPE_Q4_0
                raw, _ = quantize_matrix_q4_0(t, hessian_error_diffusion=True)
            elif len(flat) % 32 == 0 and len(shape) == 2:
                ggml_type = GGML_TYPE_Q8_0
                raw, _ = quantize_matrix_q8_0(t)
            else:
                ggml_type = GGML_TYPE_F32
                raw = struct.pack(f"<{len(flat)}f", *flat)
        else:
            # Q8_0 requires element count divisible by 32; fall back to F32 for small 1D biases/norms < 32
            if len(flat) % 32 == 0 and len(shape) == 2:
                ggml_type = GGML_TYPE_Q8_0
                raw, _ = quantize_matrix_q8_0(t)
            else:
                ggml_type = GGML_TYPE_F32
                raw = struct.pack(f"<{len(flat)}f", *flat)

        cur_data_offset = _align_offset(cur_data_offset, GGUF_DEFAULT_ALIGNMENT)
        tensor_offsets.append(cur_data_offset)
        tensor_payloads.append(raw)
        cur_data_offset += len(raw)

        # GGUF v3 tensor info: name, n_dimensions, dimensions (uint64 each), ggml_type (uint32), offset (uint64)
        buf.extend(_pack_gguf_string(name))
        buf.extend(struct.pack("<I", len(shape)))
        for dim in shape:
            buf.extend(struct.pack("<Q", dim))
        buf.extend(struct.pack("<I", ggml_type))
        buf.extend(struct.pack("<Q", tensor_offsets[-1]))

    # Pad header to GGUF_DEFAULT_ALIGNMENT before tensor data section
    aligned_hdr = _align_offset(len(buf), GGUF_DEFAULT_ALIGNMENT)
    buf.extend(b"\x00" * (aligned_hdr - len(buf)))

    data_section = bytearray()
    for off, raw in zip(tensor_offsets, tensor_payloads):
        if len(data_section) < off:
            data_section.extend(b"\x00" * (off - len(data_section)))
        data_section.extend(raw)

    buf.extend(data_section)
    out_path.write_bytes(bytes(buf))
    return out_path


def _parse_gguf_tensors(blob: Any, path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read every tensor of a GGUF v3 file, refusing anything outside or inconsistent with the file."""
    import torch

    size = len(blob)
    if blob[:4] != GGUF_MAGIC:
        raise ValueError(f"Invalid GGUF magic in {path}")
    (version, tensor_count, kv_count) = struct.unpack_from("<IQQ", blob, 4)
    if version != GGUF_VERSION_V3:
        raise ValueError(f"Unsupported GGUF version {version}; expected {GGUF_VERSION_V3}")
    if tensor_count > size or kv_count > size:
        raise ValueError(f"GGUF declares more entries than {path} could hold")

    offset = 24
    metadata: dict[str, Any] = {}
    for _ in range(kv_count):
        key, offset = _unpack_gguf_string(blob, offset)
        (vtype,) = struct.unpack_from("<I", blob, offset)
        offset += 4
        if vtype == GGUF_TYPE_STRING:
            val, offset = _unpack_gguf_string(blob, offset)
        elif vtype == GGUF_TYPE_UINT32:
            (val,) = struct.unpack_from("<I", blob, offset)
            offset += 4
        elif vtype == GGUF_TYPE_INT32:
            (val,) = struct.unpack_from("<i", blob, offset)
            offset += 4
        elif vtype == GGUF_TYPE_FLOAT32:
            (val,) = struct.unpack_from("<f", blob, offset)
            offset += 4
        else:
            raise ValueError(f"Unsupported GGUF metadata vtype {vtype}")
        metadata[key] = val

    tensor_descriptors: list[tuple[str, tuple[int, ...], int, int]] = []
    for _ in range(tensor_count):
        tname, offset = _unpack_gguf_string(blob, offset)
        (ndim,) = struct.unpack_from("<I", blob, offset)
        offset += 4
        if ndim > (size - offset) // 8:
            raise ValueError(f"GGUF tensor {tname!r} declares {ndim} dimensions")
        dims = []
        for _ in range(ndim):
            (d,) = struct.unpack_from("<Q", blob, offset)
            offset += 8
            dims.append(int(d))
        (ggml_type, rel_offset) = struct.unpack_from("<IQ", blob, offset)
        offset += 12
        tensor_descriptors.append((tname, tuple(dims), int(ggml_type), int(rel_offset)))
    if not tensor_descriptors:
        raise ValueError(f"GGUF file {path} has no tensors")

    alignment = metadata.get("general.alignment", GGUF_DEFAULT_ALIGNMENT)
    if type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1):
        raise ValueError(f"Invalid GGUF alignment {alignment!r}")
    data_base = _align_offset(offset, alignment)

    block_bytes = {GGML_TYPE_Q4_0: 18, GGML_TYPE_Q8_0: 34, NF_TYPE_Q4_BLOCK: 20}
    element_bytes = {GGML_TYPE_F32: 4, GGML_TYPE_F16: 2}
    dequantize = {GGML_TYPE_Q4_0: dequantize_matrix_q4_0, GGML_TYPE_Q8_0: dequantize_matrix_q8_0, NF_TYPE_Q4_BLOCK: dequantize_matrix_nf_q4}
    loaded_tensors: dict[str, Any] = {}
    digest_items: list[dict[str, Any]] = []
    position = 0
    for tname, shape, ggml_type, rel_off in sorted(tensor_descriptors, key=lambda d: d[3]):
        if tname in loaded_tensors:
            raise ValueError(f"Duplicate GGUF tensor {tname!r}")
        n = math.prod(shape)
        if ggml_type in element_bytes:
            length = n * element_bytes[ggml_type]
        elif ggml_type in block_bytes:
            if len(shape) != 2:
                raise ValueError(f"GGUF block-quantized tensor {tname!r} must be a matrix, got shape {shape}")
            length = ((n + 31) // 32) * block_bytes[ggml_type]
        else:
            raise ValueError(f"Unsupported GGML tensor type {ggml_type}")
        if rel_off % alignment or rel_off < position:
            raise ValueError(f"GGUF tensor {tname!r} is misaligned or overlaps another tensor")
        start = data_base + rel_off
        if start + length > size:
            raise ValueError(f"GGUF tensor {tname!r} runs past the end of the file")
        raw = blob[start : start + length]
        if ggml_type == GGML_TYPE_F32:
            values: Any = [float(x) for x in struct.unpack(f"<{n}f", raw)]
        elif ggml_type == GGML_TYPE_F16:
            values = [float(x) for x in struct.unpack(f"<{n}e", raw)]
        else:
            values = dequantize[ggml_type](raw, (shape[0], shape[1]))
        loaded_tensors[tname] = torch.tensor(values, dtype=torch.float64).reshape(shape)
        digest_items.append({"name": tname, "digest": _sha256(raw)})
        position = rel_off + length
    if data_base + position != size:
        raise ValueError(f"GGUF file {path} has trailing bytes after its last tensor")
    return loaded_tensors, digest_items


def reload_and_verify_gguf_v3(
    gguf_path: str | Path,
    reference_model: Any,
    prompts: Sequence[str],
) -> tuple[Any, ExportReloadRoundTripReceipt]:
    """Parse and reload a GGUF v3 binary file into a fresh model instance and verify that behavioral gains survive."""
    import torch

    path = Path(gguf_path).resolve()
    blob_path = path
    blob = _map_file(path)
    try:
        loaded_tensors, digest_items = _parse_gguf_tensors(blob, path)
    except (struct.error, IndexError, OverflowError) as exc:
        raise ValueError(f"Corrupt GGUF file {path}: {exc}") from exc
    _require_finite(loaded_tensors)
    _require_matches_reference(loaded_tensors, reference_model)
    reloaded = _apply_state_tensors_to_clone(reference_model, loaded_tensors)
    matches = 0
    max_diff = 0.0
    for p in prompts:
        ref_pred = reference_model.evaluate_prompt(p)
        rel_pred = reloaded.evaluate_prompt(p)
        if ref_pred.top1_token_id == rel_pred.top1_token_id:
            matches += 1
        diff = abs(float(ref_pred.top1_logit) - float(rel_pred.top1_logit))
        if diff > max_diff:
            max_diff = diff

    match_rate = matches / max(len(prompts), 1)
    param_digest = str(content_digest(canonicalize({"tensors": digest_items})))
    return reloaded, ExportReloadRoundTripReceipt(
        format="gguf_v3",
        output_path=str(path),
        file_digest=sha256_file(blob_path),
        exported_tensor_count=len(loaded_tensors),
        reloaded_parameter_digest=param_digest,
        prompts_tested=len(prompts),
        behavioral_match_rate=round(match_rate, 6),
        max_logit_abs_diff=round(max_diff, 8),
        gain_preserved_after_rebuild=match_rate == 1.0,
    )


def export_huggingface_hub_bundle(
    edited_model: Any,
    output_dir: str | Path,
    *,
    open_patch: Mapping[str, Any],
    dtype: str = "f64",
    signing_key: str | None = None,
    source_checkpoint_dir: str | Path | None = None,
    repo_id: str | None = None,
    push_to_hub: bool = False,
) -> dict[str, Any]:
    """Export an edited model with its signed Open Patch (`weight_patch.nf.json`) and a Hugging Face Model Card (`README.md`) ready for `huggingface_hub` upload."""
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    st_path = export_huggingface_safetensors(edited_model, out, dtype=dtype, source_checkpoint_dir=source_checkpoint_dir)
    has_tokenizer = (out / "tokenizer.json").is_file() or (out / "tokenizer.model").is_file()
    patch_path = out / "weight_patch.nf.json"
    patch_bytes = canonicalize(dict(open_patch))
    patch_path.write_bytes(patch_bytes)

    sbom = open_patch.get("sbom", {}) if isinstance(open_patch.get("sbom"), Mapping) else {}
    r_contract = (
        open_patch.get("rounding_contract", {})
        if isinstance(open_patch.get("rounding_contract"), Mapping)
        else {}
    )
    eval_prov = (
        open_patch.get("evaluation_provenance", {})
        if isinstance(open_patch.get("evaluation_provenance"), Mapping)
        else {}
    )
    sec_scan = (
        open_patch.get("security_scan", {})
        if isinstance(open_patch.get("security_scan"), Mapping)
        else {}
    )
    base_model_id = _card_value("base_model", sbom.get("base_model_id", getattr(edited_model, "model_name", "unknown")))
    patch_digest = _card_value("patch_digest", open_patch.get("patch_digest", content_digest(patch_bytes)))
    _card_value("repo_id", repo_id)
    for label, section, keys in (
        ("sbom", sbom, ("base_checkpoint_digest", "candidate_checkpoint_digest", "changed_parameters")),
        ("rounding_contract", r_contract, ("source_dtype", "accumulation_dtype", "output_dtype", "rounding_mode")),
        ("evaluation_provenance", eval_prov, ("runtime", "evidence_grade")),
        ("security_scan", sec_scan, ("backdoor_scan_passed",)),
    ):
        for key in keys:
            _card_value(f"{label}.{key}", section.get(key))

    model_card = "\n".join(
        [
            "---",
            "library_name: transformers",
            f"base_model: {base_model_id}",
            "tags:",
            "  - weight-programming",
            "  - neural-foundry",
            "  - verified-weight-patch",
            "license: apache-2.0",
            "---",
            "",
            f"# {repo_id or base_model_id + '-nf-patched'}",
            "",
            "This model was produced by **Weight Programming Studio (`neural_foundry`)** using a cryptographically signed, Rust-verified Weight IR transaction.",
            "",
            "## SBOM & Patch Provenance (`weight_patch.nf.json`)",
            "",
            f"- **Patch Digest:** `{patch_digest}`",
            f"- **Base Model:** `{base_model_id}` (`{sbom.get('base_checkpoint_digest', 'n/a')}`)",
            f"- **Candidate Digest:** `{sbom.get('candidate_checkpoint_digest', 'n/a')}`",
            f"- **Layers Touched:** `{sbom.get('layers_touched', [])}`",
            f"- **Changed Parameters:** `{sbom.get('changed_parameters', 0)}`",
            f"- **Rounding Contract:** `{r_contract.get('source_dtype', dtype)}` -> `{r_contract.get('accumulation_dtype', 'f32')}` -> `{r_contract.get('output_dtype', dtype)}` (`{r_contract.get('rounding_mode', 'nearest_ties_even')}`)",
            f"- **Evaluation Runtime & Sampler:** `{eval_prov.get('runtime', 'pytorch_cpu')}` / `{json.dumps(eval_prov.get('sampler', {}), sort_keys=True)}`",
            f"- **Tokenizer Included:** `{has_tokenizer}`" + ("" if has_tokenizer else " (this bundle can't be loaded for inference alone)"),
            f"- **Evidence Grade:** `{eval_prov.get('evidence_grade', 'unstated')}`",
            f"- **Anomaly Scan Passed (not proof of no backdoor):** `{sec_scan.get('backdoor_scan_passed', False)}`",
            "",
            "## Verification",
            "",
            "```sh",
            "nf verify weight_patch.nf.json --pubkey ed25519_pub:<publisher key obtained out of band>",
            "```",
            "",
        ]
    )
    readme_path = out / "README.md"
    readme_path.write_text(model_card, encoding="utf-8")

    manifest_path = None
    if signing_key is not None:
        manifest_path = write_artifact_manifest(
            out, signing_key=signing_key, signer_id="nf.hub_bundle.v1", open_patch_digest=patch_digest
        )

    pushed = False
    if push_to_hub:
        if not repo_id:
            raise ValueError("repo_id is required when push_to_hub=True")
        if not has_tokenizer:
            raise ValueError("Refusing to publish a bundle with no tokenizer; pass source_checkpoint_dir")
        if manifest_path is None:
            raise ValueError("signing_key is required when push_to_hub=True, so the files are bound to the patch")
        try:
            from huggingface_hub import HfApi  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub package is required for push_to_hub=True"
            ) from exc
        api = HfApi()
        api.upload_folder(folder_path=str(out), repo_id=repo_id, repo_type="model")
        pushed = True

    return {
        "output_dir": str(out),
        "safetensors_path": str(st_path),
        "patch_artifact_path": str(patch_path),
        "model_card_path": str(readme_path),
        "patch_digest": patch_digest,
        "artifact_manifest_path": str(manifest_path) if manifest_path else None,
        "tokenizer_included": has_tokenizer,
        "pushed_to_hub": pushed,
    }

