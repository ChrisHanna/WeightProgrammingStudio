"""Record ergonomics; Rust owns validation, canonical bytes and identities."""

import json
from collections.abc import Mapping
from typing import Any, TypeAlias, cast

from . import _native

RecordInput: TypeAlias = bytes | Mapping[str, object] | _native.RecordHandle
RecordHandle = _native.RecordHandle
NativeError = _native.NativeError
CancelledError = _native.CancelledError
NativePanicError = _native.NativePanicError
CancellationHandle = _native.CancellationHandle


def _wire(record: RecordInput) -> bytes:
    if isinstance(record, bytes):
        return record
    if isinstance(record, _native.RecordHandle):
        return record.json()
    # This encodes caller-owned Python objects, not a second canonicalizer.
    # Non-finite literals are deliberately left for the Rust parser to deny.
    return json.dumps(dict(record), ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def negotiate(required: Mapping[str, int] | None = None) -> dict[str, Any]:
    """Require exact protocol versions; unknown protocols never downgrade."""
    versions = (
        dict(required)
        if required is not None
        else {
            "core_api": 1,
            "canonical_json": 1,
            "weight_ir": 1,
            "opset": 1,
            "search_space": 1,
        }
    )
    result = _native.negotiate_protocols(_wire({"schema_version": 1, "required": versions}))
    return cast(dict[str, Any], json.loads(result))


def validate(
    kind: str, record: RecordInput, cancellation: CancellationHandle | None = None
) -> RecordHandle:
    """Return an immutable native record, not an execution certificate."""
    negotiate()
    return _native.validate_record(kind, _wire(record), cancellation)


def canonicalize(record: RecordInput) -> bytes:
    """Canonicalize JSON with the shared bounded Rust implementation."""
    negotiate()
    return _native.canonical_json(_wire(record))


def capabilities() -> dict[str, Any]:
    """Return qualified software scope, not scientific readiness."""
    return cast(dict[str, Any], json.loads(_native.capabilities_json()))


def content_digest(data: bytes) -> str:
    """Hash raw bytes in Rust; JSON identity needs canonicalize first."""
    negotiate()
    return _native.content_digest(data)
