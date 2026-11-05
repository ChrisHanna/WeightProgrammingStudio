"""Read-only standalone export verification using the fixed trusted CPU worker.

This is reload evidence, NOT acceptance authority. No live candidate is accepted,
no pointer is read/written, and no scientific/serving qualification is implied.
The caller must freeze the export root, model manifest identity and parity probes
independently before calling. The native worker re-loads copied standalone bytes
in an isolated, kernel-bounded fresh process and checks actual storage aliases.
Only the registered small dense CPU profiles supported by models.infer qualify.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike
from typing import Any

from . import _native, models

_TOTAL = 1_048_576  # Existing trusted model-worker payload admission limit.


class ReloadVerificationError(ValueError):
    """Incomplete, unsupported or inconsistent standalone reload evidence."""


@dataclass(frozen=True)
class Probe:
    inputs: tuple[float, ...]
    expected: tuple[float, ...]
    absolute_tolerance: float


def _wire(value: Any) -> bytes:
    return _native.canonical_json(json.dumps(value, allow_nan=False).encode())


def _digest(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise ReloadVerificationError("explicit sha256 identity required")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReloadVerificationError(message)


def verify_reload(
    runtime: _native.TorchRuntimeHandle,
    store_root: str | PathLike[str],
    export_root: str,
    *,
    expected_manifest: str,
    probes: Sequence[Probe],
    cpu_seconds: int = 10,
    wall_ms: int = 15000,
    address_space_bytes: int = 2147483648,
    output_bytes: int = 262144,
    cancellation: _native.CancellationHandle | None = None,
) -> bytes:
    """Return canonical, content-addressable reload evidence or raise on any failure.

    runtime must be explicitly qualified via models.qualify(enabled=True). No
    automatic qualification, network, downloads, custom factories or fallback.
    Probe outputs must come from the frozen pre-export parity suite, not this
    verifier. Tolerances cannot exceed the registered backend's numerical bound.
    Evidence is a data receipt, not a deserializable native acceptance witness.
    The trusted store's parent directories and same-UID host remain trusted.
    """
    _digest(export_root)
    _digest(expected_manifest)
    _require(0 < len(probes) <= 64, "nonempty bounded parity probes required")
    frozen: list[dict[str, Any]] = []
    for probe in probes:
        _require(isinstance(probe, Probe), "typed parity probe required")
        for values in (probe.inputs, probe.expected):
            _require(0 < len(values) <= 64 and all(type(v) in (float, int)
                     and math.isfinite(v) for v in values),
                     "nonempty bounded finite probe vectors required")
        _require(all(abs(v) <= 1e6 for v in probe.inputs), "probe input magnitude exceeded")
        tolerance = probe.absolute_tolerance
        _require(type(tolerance) in (float, int) and math.isfinite(tolerance)
                 and 0 <= tolerance <= 1e-5, "invalid parity tolerance")
        frozen.append({"inputs": list(probe.inputs), "expected": list(probe.expected),
                       "absolute_tolerance": tolerance})
    if cancellation is not None and cancellation.cancelled:
        raise ReloadVerificationError("reload cancelled before reading export")
    source = _native.restore_checkpoint(
        os.fspath(store_root), export_root, expected_manifest,
        max_checkpoint_bytes=_TOTAL, max_asset_bytes=_TOTAL, max_total_bytes=_TOTAL,
        cancellation=cancellation,
    )
    manifest = json.loads(source.manifest_json())
    bound = 1e-10 if all(v["dtype"] == "f64" for v in manifest["parameters"].values()) else 1e-5
    _require(all(p["absolute_tolerance"] <= bound for p in frozen), "backend tolerance exceeded")
    inputs = [p["inputs"] for p in frozen]
    result = models.infer(runtime, source, inputs, cpu_seconds=cpu_seconds, wall_ms=wall_ms,
                          address_space_bytes=address_space_bytes, output_bytes=output_bytes,
                          cancellation=cancellation)
    evidence = json.loads(result.receipt_bytes())
    _require(result.status == "verified_inference" and evidence.get("status") == result.status
             and evidence.get("schema_version") == 2
             and evidence.get("cleanup_confirmed") is True
             and evidence.get("promotion_status") == "not_evaluated"
             and evidence.get("outputs_eligible") is False, "incomplete fresh-process inference")
    _require(evidence.get("manifest") == expected_manifest
             and evidence.get("parameters") == source.parameter_digest
             and evidence.get("checkpoint") == manifest["assets"]["checkpoint"]
             and evidence.get("runtime") == runtime.runtime_id
             and evidence.get("inputs") == _native.content_digest(_wire(inputs)),
             "unbound fresh-process evidence")
    child = evidence.get("child_report")
    _require(isinstance(child, dict) and bool(child.get("before"))
             and child.get("before") == child.get("after")
             and child.get("inference_mode") is True
             and child.get("gradient_enabled") is False, "missing runtime inventory evidence")
    outputs = result.outputs
    _require(outputs == evidence.get("outputs") == child.get("outputs")
             and len(outputs) == len(frozen), "missing or inconsistent parity outputs")
    for actual, parity in zip(outputs, frozen):
        _require(len(actual) == len(parity["expected"])
                 and all(math.isfinite(a) and abs(a - e) <= parity["absolute_tolerance"]
                         for a, e in zip(actual, parity["expected"])), "reload probe parity failed")
    return _wire({"schema_version": 1, "kind": "fresh_process_reload_verification",
                  "export_root": export_root, "manifest": expected_manifest,
                  "parameters": source.parameter_digest, "assets": manifest["assets"],
                  "probes": frozen, "probe_suite": _native.content_digest(_wire(frozen)),
                  "reload_mode": "fresh_trusted_torch_cpu_process",
                  "inference": evidence, "promotion_status": "not_evaluated",
                  "outputs_eligible": False})
