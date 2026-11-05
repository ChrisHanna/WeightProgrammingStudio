"""Internal reports backed exclusively by an opaque Rust committed snapshot.

Capture once, then render both formats from that SAME handle. No Python SQLite
reader, accounting fold, evidence resolver or authority DTO is accepted.

Capture uses the coordinator's read-only single-transaction API. The explicit
path must name an existing canonical events.sqlite3 file, not its directory.
Reports contain internal, potentially sensitive records: they are not sanitized
public exports, acceptance receipts, or permission to disclose holdout data.
"""

from __future__ import annotations

import json
from os import PathLike, fspath
from typing import Any

from . import _native


class SnapshotError(ValueError):
    """Native capture cannot supply a validated, bounded committed snapshot."""


class CommittedSnapshot:
    """Facade over immutable native bytes, never a caller-provided report dict.

    ``to_dict`` returns a defensive copy. Python privacy is not an adversarial
    security boundary; neither this facade nor its native handle grants authority.
    """

    __slots__ = ("_handle",)
    _handle: _native.CommittedStatusSnapshot

    def __init__(self) -> None:
        raise TypeError("use capture_snapshot with an explicit ledger and contract")

    def to_dict(self) -> dict[str, Any]:
        return json.loads(render_json(self))  # type: ignore[no-any-return]


def capture_snapshot(
    database_path: str | PathLike[str], contract_digest: str
) -> CommittedSnapshot:
    """Delegate capture to Rust; never initialize or migrate a ledger in Python."""
    try:
        handle = _native.capture_status_snapshot(fspath(database_path), contract_digest)
    except _native.NativeError as error:
        raise SnapshotError(str(error)) from error
    snapshot = object.__new__(CommittedSnapshot)
    snapshot._handle = handle
    return snapshot


def render_formats(snapshot: CommittedSnapshot) -> dict[str, str]:
    """Render JSON and Markdown from one detached native handle, without recapture."""
    return {"json": render_json(snapshot), "markdown": render_markdown(snapshot)}


def _handle(snapshot: CommittedSnapshot) -> _native.CommittedStatusSnapshot:
    if not isinstance(snapshot, CommittedSnapshot) or not isinstance(
        getattr(snapshot, "_handle", None), _native.CommittedStatusSnapshot
    ):
        raise TypeError("capture one native CommittedSnapshot before rendering")
    return snapshot._handle


def render_json(snapshot: CommittedSnapshot) -> str:
    """Return the Rust snapshot's detached JSON, without re-reading any store."""
    return _handle(snapshot).render_json()


def render_markdown(snapshot: CommittedSnapshot) -> str:
    """Return Rust's lossless sections of the exact same detached JSON snapshot."""
    return _handle(snapshot).render_markdown()
