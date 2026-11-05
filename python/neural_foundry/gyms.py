"""Local gym source snapshots, not imports, validation, approval or execution."""

from __future__ import annotations

from os import PathLike
from pathlib import Path

from . import _native
from .records import RecordInput, _wire, negotiate


def import_local(
    directory: str | PathLike[str], manifest: RecordInput, store: str | PathLike[str]
) -> _native.GymSnapshotHandle:
    """Snapshot exactly the declared files; never discover/import source code."""
    negotiate()
    if not hasattr(_native, "snapshot_gym"):
        raise NotImplementedError("Native Linux gym intake is unavailable")
    return _native.snapshot_gym(
        str(Path(directory).absolute()), str(Path(store).absolute()), _wire(manifest)
    )


def import_file(
    source: str | PathLike[str], manifest: RecordInput, store: str | PathLike[str]
) -> _native.GymSnapshotHandle:
    """Pin one declared main file plus its explicit assets, without executing it."""
    negotiate()
    if not hasattr(_native, "snapshot_gym"):
        raise NotImplementedError("Native Linux gym intake is unavailable")
    path = Path(source).absolute()
    return _native.snapshot_gym(
        str(path.parent), str(Path(store).absolute()), _wire(manifest), path.name
    )


def restore(snapshot_id: str, store: str | PathLike[str]) -> _native.GymSnapshotHandle:
    """Check every persisted byte identity, without consulting live sources."""
    negotiate()
    if not hasattr(_native, "restore_gym_snapshot"):
        raise NotImplementedError("Native Linux gym intake is unavailable")
    return _native.restore_gym_snapshot(str(Path(store).absolute()), snapshot_id)


def validate(
    snapshot: _native.GymSnapshotHandle,
    plan: RecordInput,
    sandbox_root: str | PathLike[str],
    *,
    enabled: bool = False,
) -> _native.GymValidationHandle:
    """Run finite controller-fixture probes in the actual qualified sandbox."""
    negotiate()
    if not hasattr(_native, "validate_gym"):
        raise NotImplementedError("Native Linux gym validation is unavailable")
    return _native.validate_gym(snapshot, _wire(plan), str(Path(sandbox_root).absolute()), enabled)


def approve(
    snapshot: _native.GymSnapshotHandle,
    validation: _native.GymValidationHandle,
    approval_reference: str,
) -> _native.GymApprovalHandle:
    """Explicitly approve only this exact live-validated snapshot, not weights."""
    negotiate()
    if not hasattr(_native, "approve_gym_snapshot"):
        raise NotImplementedError("Native Linux gym approval is unavailable")
    return _native.approve_gym_snapshot(snapshot, validation, approval_reference)
