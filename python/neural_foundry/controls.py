"""Matched public synthetic original/no-op controls, not scientific acceptance."""

from os import PathLike
from pathlib import Path

from . import _native
from .records import CancellationHandle, RecordInput, _wire, negotiate

SourceCache = _native.ControlCacheHandle


def run(
    cache: _native.ControlCacheHandle,
    runtime: _native.TorchRuntimeHandle,
    no_op_plan: _native.PlanHandle,
    smoke_suite: RecordInput,
    protocol: RecordInput,
    store_root: str | PathLike[str],
    cancellation: CancellationHandle | None = None,
) -> _native.PairedControlsHandle:
    """Actually apply/export/reload a no-op and compare fresh seeded CPU arms.

    Source reuse requires an identical native evidence key; duplicate tasks and
    repeat seeds never inflate independent task/group denominators. Only the
    registered public arithmetic fixture is supported, not private gym access.
    """
    negotiate()
    return _native.run_paired_controls(
        cache,
        runtime,
        no_op_plan,
        (_wire(smoke_suite), _wire(protocol)),
        str(Path(store_root).absolute()),
        cancellation,
    )
