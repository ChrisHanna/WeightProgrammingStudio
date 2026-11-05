"""Live-source-backed portfolio planning, never model/gym dispatch authority.

All identities, validation, coverage and hard-gate decisions belong to Rust.
Assessment inputs are unverified measurements, not scientific evidence receipts.
"""

from . import _native
from .records import RecordInput, _wire, negotiate


def builtin(head: str) -> _native.PortfolioGymHandle:
    """Run finite native reference/wrong/invalid probes for a registered arithmetic gym."""
    negotiate()
    return _native.portfolio_builtin(head)


def user(
    snapshot: _native.GymSnapshotHandle, approval: _native.GymApprovalHandle
) -> _native.PortfolioGymHandle:
    """Bind the exact snapshotted, actually validated and approved custom package.

    Keeps its metric named score, never silently relabels reward as accuracy.
    Runtime and split custody must still be checked separately before dispatch.
    """
    negotiate()
    return _native.portfolio_user(snapshot, approval)


def compose(
    portfolio: RecordInput,
    coverage: RecordInput,
    sources: list[_native.PortfolioGymHandle],
) -> _native.PortfolioHandle:
    """Freeze exact manifests, schedules, coverage and live validation witnesses.

    Missing mandatory coverage yields a blocked planning report. Neither a
    five-gym declaration nor passing supplied numbers means an assurance pass.
    """
    negotiate()
    return _native.compose_portfolio(_wire(portfolio), _wire(coverage), sources)
