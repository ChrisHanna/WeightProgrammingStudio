"""Bounded weight/activation inspection tools with guaranteed hook cleanup (T047).

Uses model-adapter site descriptors rather than hardcoded module names.
Importing this module performs no PyTorch or research-stack imports.
"""

from __future__ import annotations

import hashlib
import math
import struct
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .records import canonicalize, content_digest

VALID_SITE_KINDS = frozenset({"weight", "activation", "residual", "attention", "mlp"})


class InspectionError(RuntimeError):
    """Typed inspection error with stable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class SiteDescriptor:
    """Adapter-declared model site; avoids hardcoding architecture module names."""

    name: str
    kind: str
    tensor_name: str | None
    module_path: str | None
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if not self.name or self.kind not in VALID_SITE_KINDS:
            raise InspectionError("INVALID_SITE", f"Unsupported site descriptor: {self.name!r}")
        if any(dim <= 0 for dim in self.shape):
            raise InspectionError("INVALID_SITE", f"Site {self.name!r} requires positive shape")


@dataclass(frozen=True)
class InspectionCap:
    """Finite resource caps for weight and activation inspection."""

    max_bytes: int = 262_144
    max_examples: int = 16
    max_sites: int = 16
    max_tokens: int = 128
    max_wall_ms: int = 5_000
    max_slice_elements: int = 4_096

    def validate(self) -> None:
        for field_name in (
            "max_bytes",
            "max_examples",
            "max_sites",
            "max_tokens",
            "max_wall_ms",
            "max_slice_elements",
        ):
            val = getattr(self, field_name)
            if not isinstance(val, int) or val <= 0:
                raise InspectionError("INVALID_CAP", f"{field_name} must be a positive integer")


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _decode_floats(raw: bytes, dtype: str) -> list[float]:
    if dtype == "f64":
        count = len(raw) // 8
        return list(struct.unpack(f"<{count}d", raw))
    if dtype == "f32":
        count = len(raw) // 4
        return [float(x) for x in struct.unpack(f"<{count}f", raw)]
    raise InspectionError("UNSUPPORTED_DTYPE", f"Unsupported inspection dtype: {dtype}")


def _flat_slice_indices(
    tensor_shape: Sequence[int], start: Sequence[int], slice_shape: Sequence[int]
) -> list[int]:
    if len(tensor_shape) != len(start) or len(tensor_shape) != len(slice_shape):
        raise InspectionError("SHAPE_MISMATCH", "Slice rank differs from tensor rank")
    for dim, s, width in zip(tensor_shape, start, slice_shape, strict=True):
        if s < 0 or width <= 0 or s + width > dim:
            raise InspectionError("OUT_OF_BOUNDS", "Requested slice exceeds tensor dimensions")
    strides = [1] * len(tensor_shape)
    for idx in range(len(tensor_shape) - 2, -1, -1):
        strides[idx] = strides[idx + 1] * tensor_shape[idx + 1]

    indices: list[int] = [0]
    for axis, (s, width) in enumerate(zip(start, slice_shape, strict=True)):
        stride = strides[axis]
        indices = [base + ( s + offset ) * stride for base in indices for offset in range(width)]
    return indices


class BoundedInspector:
    """Bounded weight/activation inspector with guaranteed hook removal and state verification."""

    def __init__(
        self,
        *,
        model_id: str,
        sites: Sequence[SiteDescriptor],
        tensors: Mapping[str, tuple[tuple[int, ...], str, bytes]],
        buffers: Mapping[str, tuple[tuple[int, ...], str, bytes]] | None = None,
        cap: InspectionCap | None = None,
    ) -> None:
        self.cap = cap or InspectionCap()
        self.cap.validate()
        if not model_id:
            raise InspectionError("INVALID_MODEL", "model_id is required")
        self.model_id = model_id
        self._sites: dict[str, SiteDescriptor] = {}
        for site in sites:
            if site.name in self._sites:
                raise InspectionError("DUPLICATE_SITE", f"Duplicate site {site.name!r}")
            self._sites[site.name] = site
        self._tensors: dict[str, tuple[tuple[int, ...], str, bytes]] = {
            name: (tuple(shape), dtype, bytes(raw))
            for name, (shape, dtype, raw) in tensors.items()
        }
        self._buffers: dict[str, tuple[tuple[int, ...], str, bytes]] = {
            name: (tuple(shape), dtype, bytes(raw))
            for name, (shape, dtype, raw) in (buffers or {}).items()
        }
        self._initial_param_digests = {
            name: _sha256(raw) for name, (_, _, raw) in self._tensors.items()
        }
        self._initial_buffer_digests = {
            name: _sha256(raw) for name, (_, _, raw) in self._buffers.items()
        }
        self.bytes_read: int = 0
        self.tokens_captured: int = 0
        self.examples_captured: int = 0

    def declared_sites(self) -> list[dict[str, Any]]:
        return [
            {
                "name": site.name,
                "kind": site.kind,
                "tensor_name": site.tensor_name,
                "module_path": site.module_path,
                "shape": list(site.shape),
                "dtype": site.dtype,
            }
            for site in self._sites.values()
        ]

    def verify_state_unchanged(self) -> dict[str, Any]:
        current_params = {
            name: _sha256(raw) for name, (_, _, raw) in self._tensors.items()
        }
        current_buffers = {
            name: _sha256(raw) for name, (_, _, raw) in self._buffers.items()
        }
        if (
            current_params != self._initial_param_digests
            or current_buffers != self._initial_buffer_digests
        ):
            raise InspectionError(
                "STATE_DRIFT",
                "Parameter or buffer state changed during inspection",
            )
        return {
            "parameters_unchanged": True,
            "buffers_unchanged": True,
            "parameter_digests": current_params,
            "buffer_digests": current_buffers,
        }

    def inspect_slice(
        self,
        site_name: str,
        start: Sequence[int],
        shape: Sequence[int],
    ) -> dict[str, Any]:
        site = self._sites.get(site_name)
        if site is None or site.tensor_name is None:
            raise InspectionError("UNKNOWN_SITE", f"Site {site_name!r} is not a declared tensor site")
        if site.tensor_name not in self._tensors:
            raise InspectionError("UNKNOWN_TENSOR", f"Tensor {site.tensor_name!r} not found")
        tensor_shape, dtype, raw = self._tensors[site.tensor_name]
        indices = _flat_slice_indices(tensor_shape, start, shape)
        if len(indices) > self.cap.max_slice_elements:
            raise InspectionError(
                "CAP_EXHAUSTED",
                f"Requested slice ({len(indices)} elements) exceeds max_slice_elements ({self.cap.max_slice_elements})",
            )
        elem_bytes = 8 if dtype == "f64" else 4
        needed_bytes = len(indices) * elem_bytes
        if self.bytes_read + needed_bytes > self.cap.max_bytes:
            raise InspectionError(
                "CAP_EXHAUSTED",
                f"Inspection byte budget exhausted ({self.bytes_read + needed_bytes} > {self.cap.max_bytes})",
            )
        all_values = _decode_floats(raw, dtype)
        values = [all_values[i] for i in indices]
        self.bytes_read += needed_bytes
        state_check = self.verify_state_unchanged()
        return {
            "schema_version": 1,
            "model_id": self.model_id,
            "site": site.name,
            "tensor": site.tensor_name,
            "start": list(start),
            "shape": list(shape),
            "dtype": dtype,
            "values": values,
            "bytes_charged": needed_bytes,
            "total_bytes_read": self.bytes_read,
            "tensor_digest": state_check["parameter_digests"][site.tensor_name],
            "state_preserved": True,
            "evidence_label": {
                "schema_version": 1,
                "origin": "measured_result",
                "support": "suggestive",
                "provenance": f"bounded_slice:{self.model_id}:{site.name}",
                "access": "development",
            },
        }

    def summarize_site(self, site_name: str) -> dict[str, Any]:
        site = self._sites.get(site_name)
        if site is None or site.tensor_name is None:
            raise InspectionError("UNKNOWN_SITE", f"Site {site_name!r} is not a declared tensor site")
        tensor_shape, dtype, raw = self._tensors[site.tensor_name]
        if self.bytes_read + len(raw) > self.cap.max_bytes:
            raise InspectionError(
                "CAP_EXHAUSTED",
                f"Inspection byte budget exhausted ({self.bytes_read + len(raw)} > {self.cap.max_bytes})",
            )
        values = _decode_floats(raw, dtype)
        self.bytes_read += len(raw)
        count = len(values)
        mean = sum(values) / count
        l2_norm = math.sqrt(sum(v * v for v in values))
        nonzero = sum(1 for v in values if v != 0.0)
        state_check = self.verify_state_unchanged()
        return {
            "schema_version": 1,
            "model_id": self.model_id,
            "site": site.name,
            "tensor": site.tensor_name,
            "shape": list(tensor_shape),
            "dtype": dtype,
            "count": count,
            "min": min(values),
            "max": max(values),
            "mean": mean,
            "l2_norm": l2_norm,
            "nonzero_count": nonzero,
            "tensor_digest": state_check["parameter_digests"][site.tensor_name],
            "state_preserved": True,
        }

    def capture_activations(
        self,
        module: Any,
        *,
        site_names: Sequence[str],
        examples: Sequence[Any],
        input_provenance: str,
        forward_fn: Callable[[Any, Any], Any] | None = None,
        custom_hook: Callable[[str, Any], None] | None = None,
    ) -> dict[str, Any]:
        """Run bounded activation capture with temporary hooks and guaranteed state restoration."""
        if not input_provenance or not input_provenance.strip():
            raise InspectionError("INVALID_PROVENANCE", "input_provenance is required")
        if not site_names or len(site_names) > self.cap.max_sites:
            raise InspectionError(
                "CAP_EXHAUSTED",
                f"Requested site count {len(site_names)} exceeds max_sites ({self.cap.max_sites})",
            )
        if not examples or self.examples_captured + len(examples) > self.cap.max_examples:
            raise InspectionError(
                "CAP_EXHAUSTED",
                f"Example count {self.examples_captured + len(examples)} exceeds max_examples ({self.cap.max_examples})",
            )

        selected_sites: list[SiteDescriptor] = []
        for name in site_names:
            site = self._sites.get(name)
            if site is None or site.module_path is None:
                raise InspectionError(
                    "UNKNOWN_SITE",
                    f"Activation capture requires a declared module site, got {name!r}",
                )
            selected_sites.append(site)

        import torch

        named_modules = dict(module.named_modules())
        for site in selected_sites:
            if site.module_path not in named_modules:
                raise InspectionError(
                    "UNKNOWN_MODULE",
                    f"Declared module_path {site.module_path!r} not present in model",
                )

        # Snapshot all parameters, buffers, training state, and grad flags before hooks.
        was_training = bool(module.training)
        param_snapshots = {
            name: (p.detach().clone(), bool(p.requires_grad))
            for name, p in module.named_parameters(remove_duplicate=False)
        }
        buffer_snapshots = {
            name: b.detach().clone()
            for name, b in module.named_buffers(remove_duplicate=False)
        }
        before_param_hashes = {
            name: _sha256(bytes(tensor.detach().cpu().contiguous().view(torch.uint8).reshape(-1).tolist()))
            for name, (tensor, _) in param_snapshots.items()
        }
        before_buffer_hashes = {
            name: _sha256(bytes(tensor.detach().cpu().contiguous().view(torch.uint8).reshape(-1).tolist()))
            for name, tensor in buffer_snapshots.items()
        }

        handles: list[Any] = []
        captured_by_example: list[dict[str, Any]] = []
        current_capture: dict[str, dict[str, Any]] = {}
        started_ns = time.monotonic_ns()
        total_tokens = 0
        hooks_removed = False

        def make_hook(site_desc: SiteDescriptor) -> Callable[[Any, Any, Any], None]:
            def _hook(_mod: Any, _inputs: Any, output: Any) -> None:
                elapsed_ms = (time.monotonic_ns() - started_ns) // 1_000_000
                if elapsed_ms > self.cap.max_wall_ms:
                    raise InspectionError("CAP_EXHAUSTED", "Activation capture wall time cap exceeded")
                if custom_hook is not None:
                    custom_hook(site_desc.name, output)
                tensor = output[0] if isinstance(output, tuple) else output
                detached = tensor.detach().cpu().contiguous().to(torch.float64)
                raw_bytes = bytes(detached.view(torch.uint8).reshape(-1).tolist())
                if self.bytes_read + len(raw_bytes) > self.cap.max_bytes:
                    raise InspectionError(
                        "CAP_EXHAUSTED",
                        f"Activation capture exceeded max_bytes ({self.cap.max_bytes})",
                    )
                self.bytes_read += len(raw_bytes)
                flat = [float(x) for x in detached.reshape(-1).tolist()]
                count = len(flat)
                current_capture[site_desc.name] = {
                    "site": site_desc.name,
                    "kind": site_desc.kind,
                    "module_path": site_desc.module_path,
                    "shape": list(detached.shape),
                    "count": count,
                    "mean": sum(flat) / count if count else 0.0,
                    "l2_norm": math.sqrt(sum(v * v for v in flat)),
                    "preview": flat[: min(16, count)],
                    "activation_digest": _sha256(raw_bytes),
                }

            return _hook

        try:
            module.eval()
            for site in selected_sites:
                assert site.module_path is not None
                target_mod = named_modules[site.module_path]
                handles.append(target_mod.register_forward_hook(make_hook(site)))

            with torch.no_grad():
                for ex_idx, example in enumerate(examples):
                    elapsed_ms = (time.monotonic_ns() - started_ns) // 1_000_000
                    if elapsed_ms > self.cap.max_wall_ms:
                        raise InspectionError("CAP_EXHAUSTED", "Activation capture wall time cap exceeded")
                    if isinstance(example, torch.Tensor):
                        tok_count = int(example.numel())
                        ex_bytes = bytes(example.detach().cpu().contiguous().view(torch.uint8).reshape(-1).tolist())
                    else:
                        ex_list = list(example)
                        tok_count = len(ex_list)
                        ex_bytes = canonicalize({"tokens": ex_list})
                    total_tokens += tok_count
                    if self.tokens_captured + total_tokens > self.cap.max_tokens:
                        raise InspectionError(
                            "CAP_EXHAUSTED",
                            f"Captured token count {self.tokens_captured + total_tokens} exceeds max_tokens ({self.cap.max_tokens})",
                        )
                    current_capture = {}
                    if forward_fn is not None:
                        forward_fn(module, example)
                    else:
                        module(example)
                    captured_by_example.append(
                        {
                            "example_index": ex_idx,
                            "input_digest": content_digest(ex_bytes),
                            "token_count": tok_count,
                            "sites": dict(current_capture),
                        }
                    )
        finally:
            for handle in handles:
                handle.remove()
            hooks_removed = all(
                site.module_path is not None
                and len(getattr(named_modules[site.module_path], "_forward_hooks", {})) == 0
                for site in selected_sites
            )
            # Check for any state drift caused by a faulty hook and restore original state before raising.
            after_param_hashes = {
                name: _sha256(bytes(p.detach().cpu().contiguous().view(torch.uint8).reshape(-1).tolist()))
                for name, p in module.named_parameters(remove_duplicate=False)
            }
            after_buffer_hashes = {
                name: _sha256(bytes(b.detach().cpu().contiguous().view(torch.uint8).reshape(-1).tolist()))
                for name, b in module.named_buffers(remove_duplicate=False)
            }
            drifted = (
                after_param_hashes != before_param_hashes
                or after_buffer_hashes != before_buffer_hashes
            )
            if drifted:
                with torch.no_grad():
                    for name, p in module.named_parameters(remove_duplicate=False):
                        orig_tensor, orig_grad = param_snapshots[name]
                        p.copy_(orig_tensor)
                        p.requires_grad_(orig_grad)
                    for name, b in module.named_buffers(remove_duplicate=False):
                        b.copy_(buffer_snapshots[name])
            for name, p in module.named_parameters(remove_duplicate=False):
                _, orig_grad = param_snapshots[name]
                p.requires_grad_(orig_grad)
                p.grad = None
            module.train(was_training)
            if drifted:
                raise InspectionError(
                    "STATE_DRIFT",
                    "Hook mutated model parameter or buffer state; state restored and capture rejected",
                )

        self.tokens_captured += total_tokens
        self.examples_captured += len(examples)
        self.verify_state_unchanged()
        return {
            "schema_version": 1,
            "model_id": self.model_id,
            "input_provenance": input_provenance,
            "hooks_removed": hooks_removed,
            "state_preserved": True,
            "before_parameters": before_param_hashes,
            "after_parameters": after_param_hashes,
            "before_buffers": before_buffer_hashes,
            "after_buffers": after_buffer_hashes,
            "total_tokens": total_tokens,
            "bytes_read": self.bytes_read,
            "examples": captured_by_example,
            "evidence_label": {
                "schema_version": 1,
                "origin": "measured_result",
                "support": "suggestive",
                "provenance": input_provenance,
                "access": "development",
            },
        }
