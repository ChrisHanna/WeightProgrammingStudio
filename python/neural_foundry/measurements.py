"""Frozen, real CPU timing/RSS measurements. No mock clock can mint execution evidence."""

import json
from typing import Any

from . import _native
from .records import CancellationHandle, RecordInput, _wire, negotiate


def environment(runtime: _native.TorchRuntimeHandle) -> dict[str, Any]:
    """Observe the actual qualified runtime, boot-visible CPU/affinity and kernel."""
    negotiate()
    result: dict[str, Any] = json.loads(_native.measurement_environment(runtime))
    return result


def run(
    runtime: _native.TorchRuntimeHandle,
    original: _native.SourceHandle,
    candidate: _native.SourceHandle,
    protocol: RecordInput,
    *,
    cpu_seconds: int = 10,
    wall_ms: int = 15000,
    address_space_bytes: int = 2147483648,
    output_bytes: int = 262144,
    cancellation: CancellationHandle | None = None,
) -> _native.MeasurementHandle:
    """Measure warmup/repeated interleaved registered CPU workloads in a real child.

    Forward and in-process numeric-request boundaries differ. RSS covers the
    whole paired co-resident process, never per-model/device allocation memory.
    Required unavailable device metrics cannot pass. No-op timing noise cannot
    satisfy a positive-gain gate, and point estimates are not scientific acceptance.
    """
    negotiate()
    return _native.measure_protocol(
        runtime,
        original,
        candidate,
        _wire(protocol),
        _wire(
            {
                "cpu_seconds": cpu_seconds,
                "wall_ms": wall_ms,
                "address_space_bytes": address_space_bytes,
                "output_bytes": output_bytes,
            }
        ),
        cancellation,
    )
