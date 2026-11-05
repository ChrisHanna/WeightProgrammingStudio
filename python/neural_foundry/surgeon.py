"""Native source loading and isolated direct edits, without promotion."""

from collections.abc import Mapping

from . import _native
from .records import CancellationHandle, RecordInput, _wire, negotiate

SourceHandle = _native.SourceHandle
CandidateHandle = _native.CandidateHandle


def load(
    manifest: RecordInput,
    checkpoint: bytes,
    assets: Mapping[str, bytes],
    cancellation: CancellationHandle | None = None,
) -> SourceHandle:
    negotiate()
    return _native.load_source(_wire(manifest), checkpoint, dict(assets), cancellation)


def apply(
    plan: _native.PlanHandle,
    smoke_suite: RecordInput,
    cancellation: CancellationHandle | None = None,
) -> CandidateHandle:
    """Return a core-verified candidate; scientific acceptance stays pending."""
    negotiate()
    return _native.apply_plan(plan, _wire(smoke_suite), cancellation)
