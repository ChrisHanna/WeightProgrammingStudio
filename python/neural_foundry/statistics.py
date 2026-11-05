"""Native finite-fixture and seeded paired group analysis; not an evidence importer."""

from . import _native
from .records import CancellationHandle, RecordInput, _wire, negotiate


def plan(protocol: RecordInput) -> _native.StatisticalPlanHandle:
    """Freeze task/group/seed/model/metric contrasts and reserved multiplicity.

    This is an analysis plan, not proof of independently held final data or
    that a candidate model/evaluator actually produced supplied observations.
    """
    negotiate()
    return _native.freeze_statistics(_wire(protocol))


def analyze(
    frozen: _native.StatisticalPlanHandle,
    observations: RecordInput,
    cancellation: CancellationHandle | None = None,
) -> _native.StatisticalAnalysisHandle:
    """Reject unmatched/conflicting/nonfinite data; missing pairs stay incomplete.

    Group resampling keeps both arms and all declared task/seed means together.
    Statistical conditions alone never constitute scientific acceptance.
    """
    negotiate()
    return _native.analyze_statistics(frozen, _wire(observations), cancellation)
