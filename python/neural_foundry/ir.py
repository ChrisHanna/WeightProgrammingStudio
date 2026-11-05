"""Weight IR validation and pure native compilation; no model forwards."""

from collections.abc import Mapping

from . import _native
from .records import CancellationHandle, RecordHandle, RecordInput, _wire, negotiate, validate

PlanHandle = _native.PlanHandle


def program(record: RecordInput) -> RecordHandle:
    return validate("program", record)


def search_space(record: RecordInput) -> RecordHandle:
    return validate("search_space", record)


def compile(
    source: _native.SourceHandle,
    program: RecordInput,
    search_space: RecordInput,
    policy: RecordInput,
    operands: Mapping[str, bytes] | None = None,
    cancellation: CancellationHandle | None = None,
) -> PlanHandle:
    """Use actual source/operand bytes and the explicit frozen Rust policy."""
    negotiate()
    return _native.compile_plan(
        source,
        _wire(program),
        _wire(search_space),
        _wire(policy),
        dict(operands or {}),
        cancellation,
    )
