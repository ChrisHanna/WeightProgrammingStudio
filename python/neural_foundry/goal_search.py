"""Goal-driven edit search contracts, independent of a model architecture."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from .records import canonicalize, content_digest

SEARCH_SCHEMA_VERSION = 1
CandidateArm = Literal["direct", "baseline", "control"]
SearchState = Literal["accepted", "rejected", "skipped", "failed"]
TerminalState = Literal[
    "policy_feasible",
    "frontier_empty",
    "baseline_dominated",
    "budget_exhausted",
    "monitor_drift",
    "replay_failed",
    "qualification_incomplete",
    "no_candidate_cleared_min_gain",
]


class GoalSearchError(ValueError):
    """The search contract is incomplete or was used after a state freeze."""


def _freeze(value: Any, where: str) -> Any:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise GoalSearchError(f"{where} has a non-string key")
        return MappingProxyType({key: _freeze(item, f"{where}.{key}") for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item, where) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise GoalSearchError(f"{where} isn't finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise GoalSearchError(f"{where} has unsupported type {type(value).__name__}")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _unit_interval(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise GoalSearchError(f"{name} must be finite and within [0, 1]")


@dataclass(frozen=True)
class AcceptanceProfile:
    """A frozen deployment policy. It is part of search identity and cannot be selected by result."""

    profile_id: str
    min_target_accuracy: float
    min_target_gain: float
    min_retention: float
    min_safety: float
    max_kl_nats: float | None
    max_perplexity_ratio: float | None
    min_random_control_advantage: float = 0.0
    min_placebo_control_advantage: float = 0.0
    description: str = ""
    schema_version: int = SEARCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SEARCH_SCHEMA_VERSION:
            raise GoalSearchError(f"Unknown acceptance profile schema {self.schema_version!r}")
        if not self.profile_id.strip():
            raise GoalSearchError("profile_id must be non-empty")
        for name in ("min_target_accuracy", "min_target_gain", "min_retention", "min_safety"):
            _unit_interval(float(getattr(self, name)), name)
        for name in ("min_random_control_advantage", "min_placebo_control_advantage"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise GoalSearchError(f"{name} must be a finite non-negative number")
        if self.max_kl_nats is not None and (not math.isfinite(self.max_kl_nats) or self.max_kl_nats < 0.0):
            raise GoalSearchError("max_kl_nats must be finite and non-negative")
        if self.max_perplexity_ratio is not None and (
            not math.isfinite(self.max_perplexity_ratio) or self.max_perplexity_ratio < 1.0
        ):
            raise GoalSearchError("max_perplexity_ratio must be finite and at least 1")

    def to_document(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "nf_acceptance_profile_v1",
            "profile_id": self.profile_id,
            "min_target_accuracy": self.min_target_accuracy,
            "min_target_gain": self.min_target_gain,
            "min_retention": self.min_retention,
            "min_safety": self.min_safety,
            "max_kl_nats": self.max_kl_nats,
            "max_perplexity_ratio": self.max_perplexity_ratio,
            "min_random_control_advantage": self.min_random_control_advantage,
            "min_placebo_control_advantage": self.min_placebo_control_advantage,
            "description": self.description,
            "mandatory": {
                "beat_random_control": True,
                "beat_placebo_control": True,
                "replay_verified": True,
                "reload_parity_verified": True,
            },
        }

    @property
    def digest(self) -> str:
        return str(content_digest(canonicalize(self.to_document())))


STRICT_GENERAL_PROFILE = AcceptanceProfile(
    profile_id="strict-general-v1",
    min_target_accuracy=0.90,
    min_target_gain=0.01,
    min_retention=0.98,
    min_safety=1.0,
    max_kl_nats=math.log(1.05),
    max_perplexity_ratio=1.05,
    description="General-purpose deployment: high task gain with broad behavior preservation.",
)

TASK_SPECIALIZED_PROFILE = AcceptanceProfile(
    profile_id="task-specialized-v1",
    min_target_accuracy=0.90,
    min_target_gain=0.05,
    min_retention=0.0,
    min_safety=1.0,
    max_kl_nats=None,
    max_perplexity_ratio=None,
    description="Task-specialized deployment: retention losses are measured and disclosed, never hidden.",
)


@dataclass(frozen=True)
class RequirementCheck:
    requirement: str
    passed: bool
    observed: Any
    threshold: Any = None


@dataclass(frozen=True)
class ProfileDecision:
    candidate_id: str
    profile_id: str
    profile_digest: str
    eligible: bool
    checks: tuple[RequirementCheck, ...]
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class CandidateArtifact:
    relative_path: str
    file_digest: str
    state_digest: str
    tensor_names: tuple[str, ...]
    file_bytes: int
    format: str = "nf_torch_tensor_map_v1"

    def __post_init__(self) -> None:
        if not self.relative_path or self.relative_path.startswith(("/", "\\")):
            raise GoalSearchError("Candidate artifact path must be non-empty and relative")
        if "\\" in self.relative_path or ".." in self.relative_path.split("/"):
            raise GoalSearchError("Candidate artifact path must be a normalized relative POSIX path")
        if not self.file_digest.startswith("sha256:") or not self.state_digest.startswith("sha256:"):
            raise GoalSearchError("Candidate artifact and state digests must use sha256")
        if not self.tensor_names or len(set(self.tensor_names)) != len(self.tensor_names):
            raise GoalSearchError("Candidate artifact tensor names must be non-empty and unique")
        if self.file_bytes <= 0:
            raise GoalSearchError("Candidate artifact must contain bytes")

    def to_document(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "relative_path": self.relative_path,
            "file_digest": self.file_digest,
            "state_digest": self.state_digest,
            "tensor_names": list(self.tensor_names),
            "file_bytes": self.file_bytes,
        }


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    parent_candidate_id: str | None
    round_number: int
    arm: CandidateArm
    method: str
    proposal_digest: str
    state_digest: str | None
    scores: Mapping[str, float | int | None]
    proposal: Mapping[str, Any]
    search_state: SearchState
    replay_verified: bool
    reload_parity_verified: bool
    rejection_reasons: tuple[str, ...] = ()
    next_move_reason: str | None = None
    artifact: CandidateArtifact | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise GoalSearchError("candidate_id must be non-empty")
        if self.round_number < 0:
            raise GoalSearchError("round_number must be non-negative")
        if self.arm not in {"direct", "baseline", "control"}:
            raise GoalSearchError(f"Unknown candidate arm {self.arm!r}")
        if self.search_state not in {"accepted", "rejected", "skipped", "failed"}:
            raise GoalSearchError(f"Unknown search state {self.search_state!r}")
        if self.artifact is not None:
            if self.artifact.state_digest != self.state_digest:
                raise GoalSearchError("Candidate artifact state digest does not match the record")
            if self.artifact.format != "nf_torch_tensor_map_v1":
                raise GoalSearchError(f"Unsupported candidate artifact format {self.artifact.format!r}")
        object.__setattr__(self, "scores", _freeze(dict(self.scores), "scores"))
        object.__setattr__(self, "proposal", _freeze(dict(self.proposal), "proposal"))
        object.__setattr__(self, "rejection_reasons", tuple(self.rejection_reasons))

    def to_document(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "parent_candidate_id": self.parent_candidate_id,
            "round_number": self.round_number,
            "arm": self.arm,
            "method": self.method,
            "proposal_digest": self.proposal_digest,
            "state_digest": self.state_digest,
            "scores": _thaw(self.scores),
            "proposal": _thaw(self.proposal),
            "search_state": self.search_state,
            "replay_verified": self.replay_verified,
            "reload_parity_verified": self.reload_parity_verified,
            "rejection_reasons": list(self.rejection_reasons),
            "next_move_reason": self.next_move_reason,
            "artifact": self.artifact.to_document() if self.artifact is not None else None,
        }


@dataclass(frozen=True)
class FrontierSnapshot:
    cycle_id: str
    profile_id: str
    profile_digest: str
    search_frontier_ids: tuple[str, ...]
    eligible_ids: tuple[str, ...]
    dominated_ids: tuple[str, ...]
    rejected_ids: tuple[str, ...]
    baseline_dominated: bool
    records: tuple[Mapping[str, Any], ...]
    snapshot_digest: str

    def to_document(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "profile_id": self.profile_id,
            "profile_digest": self.profile_digest,
            "search_frontier_ids": list(self.search_frontier_ids),
            "eligible_ids": list(self.eligible_ids),
            "dominated_ids": list(self.dominated_ids),
            "rejected_ids": list(self.rejected_ids),
            "baseline_dominated": self.baseline_dominated,
            "records": [_thaw(record) for record in self.records],
            "snapshot_digest": self.snapshot_digest,
        }


def validate_frontier_snapshot(document: Mapping[str, Any]) -> None:
    """Recompute the digest and cross-check all derived frontier sets before consumption."""
    required = {
        "cycle_id", "profile_id", "profile_digest", "search_frontier_ids",
        "eligible_ids", "dominated_ids", "rejected_ids", "baseline_dominated",
        "records", "snapshot_digest",
    }
    if set(document) != required:
        raise GoalSearchError("Frozen frontier has missing or unknown top-level fields")
    records = document["records"]
    if not isinstance(records, (list, tuple)) or not all(isinstance(row, Mapping) for row in records):
        raise GoalSearchError("Frozen frontier records must be a sequence of objects")
    record_ids = [row.get("candidate_id") for row in records]
    if any(not isinstance(value, str) or not value for value in record_ids):
        raise GoalSearchError("Frozen frontier contains an invalid candidate ID")
    if len(record_ids) != len(set(record_ids)):
        raise GoalSearchError("Frozen frontier contains duplicate candidate IDs")
    record_set = set(record_ids)
    frontier = document["search_frontier_ids"]
    eligible = document["eligible_ids"]
    dominated = document["dominated_ids"]
    rejected = document["rejected_ids"]
    for label, values in (("search_frontier_ids", frontier), ("eligible_ids", eligible),
                          ("dominated_ids", dominated), ("rejected_ids", rejected)):
        if not isinstance(values, (list, tuple)) or not all(isinstance(value, str) for value in values):
            raise GoalSearchError(f"Frozen frontier {label} must be a sequence of IDs")
        if len(values) != len(set(values)) or not set(values) <= record_set:
            raise GoalSearchError(f"Frozen frontier {label} has duplicate or unknown IDs")
    if set(frontier) & set(dominated) or set(frontier) | set(dominated) != record_set:
        raise GoalSearchError("Frozen frontier and dominated sets do not partition its records")
    direct_ids = {row["candidate_id"] for row in records if row.get("arm") == "direct"}
    if set(eligible) & set(rejected) or set(eligible) | set(rejected) != direct_ids:
        raise GoalSearchError("Frozen profile decisions do not partition direct candidates")
    digest_payload = {
        "cycle_id": document["cycle_id"],
        "profile_id": document["profile_id"],
        "profile_digest": document["profile_digest"],
        "search_frontier_ids": list(frontier),
        "eligible_ids": list(eligible),
        "dominated_ids": list(dominated),
        "rejected_ids": list(rejected),
        "baseline_dominated": document["baseline_dominated"],
        "records": [_thaw(_freeze(row, "record")) for row in records],
    }
    expected = str(content_digest(canonicalize(digest_payload)))
    if document["snapshot_digest"] != expected:
        raise GoalSearchError("Frozen frontier digest does not match its complete record manifest")


def _score(record: CandidateRecord, name: str) -> float:
    value = record.scores.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise GoalSearchError(f"Candidate {record.candidate_id!r} is missing finite score {name!r}")
    return float(value)


def decide_profile(
    profile: AcceptanceProfile,
    candidate: CandidateRecord,
    controls: Mapping[str, CandidateRecord],
) -> ProfileDecision:
    """Apply a frozen profile without changing it or consulting holdout data."""
    checks: list[RequirementCheck] = []
    incomplete: list[str] = []

    def required_score(record: CandidateRecord, name: str) -> float | None:
        try:
            return _score(record, name)
        except GoalSearchError as exc:
            incomplete.append(str(exc))
            return None

    target = required_score(candidate, "target_accuracy")
    target_gain = required_score(candidate, "target_gain")
    retention = required_score(candidate, "retention_accuracy")
    safety = required_score(candidate, "safety_pass_rate")
    kl = required_score(candidate, "kl_nats") if profile.max_kl_nats is not None else None
    perplexity = (
        required_score(candidate, "perplexity_ratio")
        if profile.max_perplexity_ratio is not None
        else None
    )

    if target is not None:
        checks.append(RequirementCheck("target_accuracy", target >= profile.min_target_accuracy, target, profile.min_target_accuracy))
    if target_gain is not None:
        checks.append(RequirementCheck("target_gain", target_gain >= profile.min_target_gain, target_gain, profile.min_target_gain))
    if retention is not None:
        checks.append(RequirementCheck("retention", retention >= profile.min_retention, retention, profile.min_retention))
    if safety is not None:
        checks.append(RequirementCheck("safety", safety >= profile.min_safety, safety, profile.min_safety))
    if profile.max_kl_nats is not None and kl is not None:
        checks.append(RequirementCheck("kl_nats", kl <= profile.max_kl_nats, kl, profile.max_kl_nats))
    if profile.max_perplexity_ratio is not None and perplexity is not None:
        checks.append(
            RequirementCheck(
                "perplexity_ratio",
                perplexity <= profile.max_perplexity_ratio,
                perplexity,
                profile.max_perplexity_ratio,
            )
        )

    checks.append(RequirementCheck("replay_verified", candidate.replay_verified is True, candidate.replay_verified, True))
    checks.append(
        RequirementCheck(
            "reload_parity_verified",
            candidate.reload_parity_verified is True,
            candidate.reload_parity_verified,
            True,
        )
    )

    for control_name, advantage in (
        ("random", profile.min_random_control_advantage),
        ("placebo", profile.min_placebo_control_advantage),
    ):
        control = controls.get(control_name)
        if control is None:
            incomplete.append(f"Missing mandatory {control_name} control")
            continue
        control_target = required_score(control, "target_accuracy")
        if target is not None and control_target is not None:
            checks.append(
                RequirementCheck(
                    f"target_above_{control_name}",
                    target > control_target + advantage,
                    target,
                    control_target + advantage,
                )
            )

    if incomplete:
        checks.extend(RequirementCheck("complete_evidence", False, reason, None) for reason in incomplete)
    failures = tuple(check.requirement for check in checks if not check.passed)
    return ProfileDecision(candidate.candidate_id, profile.profile_id, profile.digest, not failures, tuple(checks), failures)


def _dominates(left: CandidateRecord, right: CandidateRecord) -> bool:
    """Compare behavioral evidence; resource costs are reported, not dominance criteria."""
    higher = ["target_accuracy", "retention_accuracy"]
    if left.scores.get("safety_pass_rate") is not None and right.scores.get("safety_pass_rate") is not None:
        higher.append("safety_pass_rate")
    lower = ("kl_nats",)
    try:
        if any(_score(left, name) < _score(right, name) for name in higher):
            return False
        if any(_score(left, name) > _score(right, name) for name in lower):
            return False
        strictly_better = any(_score(left, name) > _score(right, name) for name in higher) or any(
            _score(left, name) < _score(right, name) for name in lower
        )
        return strictly_better
    except GoalSearchError:
        return False


class CandidateArchive:
    """Append-only search history with explicit profile and one-time frontier freeze."""

    def __init__(self, cycle_id: str, profile: AcceptanceProfile) -> None:
        if not cycle_id.strip():
            raise GoalSearchError("cycle_id must be non-empty")
        self.cycle_id = cycle_id
        self.profile = profile
        self._records: list[CandidateRecord] = []
        self._by_id: dict[str, CandidateRecord] = {}
        self._frozen: FrontierSnapshot | None = None

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> tuple[CandidateRecord, ...]:
        return tuple(self._records)

    def add(self, record: CandidateRecord) -> None:
        if self._frozen is not None:
            raise GoalSearchError("The candidate archive is frozen; start a new cycle")
        if record.candidate_id in self._by_id:
            raise GoalSearchError(f"Duplicate candidate_id {record.candidate_id!r}")
        if record.parent_candidate_id is not None and record.parent_candidate_id not in self._by_id:
            raise GoalSearchError(f"Unknown parent candidate {record.parent_candidate_id!r}")
        self._records.append(record)
        self._by_id[record.candidate_id] = record

    def decision(self, candidate_id: str) -> ProfileDecision:
        candidate = self._by_id[candidate_id]
        bound_controls = [
            record
            for record in self._records
            if record.arm == "control" and record.parent_candidate_id == candidate_id
        ]
        controls = {
            record.method: record
            for record in bound_controls
            if record.method in {"random", "placebo"}
        }
        return decide_profile(self.profile, candidate, controls)

    def search_frontier(self) -> tuple[CandidateRecord, ...]:
        executable = [record for record in self._records if record.search_state not in {"failed", "skipped"}]
        return tuple(
            record
            for record in executable
            if not any(_dominates(other, record) for other in executable if other.candidate_id != record.candidate_id)
        )

    def freeze(self) -> FrontierSnapshot:
        if self._frozen is not None:
            raise GoalSearchError("The frontier is already frozen for this cycle")
        decisions = {record.candidate_id: self.decision(record.candidate_id) for record in self._records if record.arm == "direct"}
        eligible = tuple(record_id for record_id, decision in decisions.items() if decision.eligible)
        rejected = tuple(record_id for record_id, decision in decisions.items() if not decision.eligible)
        frontier = self.search_frontier()
        frontier_ids = tuple(record.candidate_id for record in frontier)
        frontier_set = set(frontier_ids)
        dominated = tuple(record.candidate_id for record in self._records if record.candidate_id not in frontier_set)
        direct = [record for record in self._records if record.arm == "direct"]
        baselines = [record for record in self._records if record.arm == "baseline"]
        baseline_dominated = bool(direct) and all(
            any(_dominates(baseline, candidate) for baseline in baselines) for candidate in direct
        )
        documents = tuple(_freeze(record.to_document(), "record") for record in self._records)
        digest_payload = {
            "cycle_id": self.cycle_id,
            "profile_id": self.profile.profile_id,
            "profile_digest": self.profile.digest,
            "search_frontier_ids": list(frontier_ids),
            "eligible_ids": list(eligible),
            "dominated_ids": list(dominated),
            "rejected_ids": list(rejected),
            "baseline_dominated": baseline_dominated,
            "records": [_thaw(record) for record in documents],
        }
        snapshot = FrontierSnapshot(
            cycle_id=self.cycle_id,
            profile_id=self.profile.profile_id,
            profile_digest=self.profile.digest,
            search_frontier_ids=frontier_ids,
            eligible_ids=eligible,
            dominated_ids=dominated,
            rejected_ids=rejected,
            baseline_dominated=baseline_dominated,
            records=documents,
            snapshot_digest=str(content_digest(canonicalize(digest_payload))),
        )
        self._frozen = snapshot
        return snapshot

    def terminal_state(self) -> TerminalState:
        snapshot = self.freeze() if self._frozen is None else self._frozen
        if snapshot.baseline_dominated:
            return "baseline_dominated"
        if snapshot.eligible_ids:
            return "policy_feasible"
        return "frontier_empty"


@dataclass(frozen=True)
class SearchBudgets:
    max_rounds: int = 40
    max_candidate_evaluations: int = 20_000
    max_changed_parameters: int | None = None
    max_wall_seconds: float | None = None
    min_gain: float = 0.005
    max_no_gain_rounds: int = 3
    max_monitor_retention_drop: float = 0.20
    max_monitor_kl_increase: float = 0.25

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 1 for value in (
            self.max_rounds, self.max_candidate_evaluations, self.max_no_gain_rounds,
        )):
            raise GoalSearchError("Search round, evaluation, and patience budgets must be positive")
        if self.max_changed_parameters is not None and (type(self.max_changed_parameters) is not int or self.max_changed_parameters < 0):
            raise GoalSearchError("max_changed_parameters must be non-negative")
        if self.max_wall_seconds is not None and (isinstance(self.max_wall_seconds, bool) or not isinstance(self.max_wall_seconds, (int, float)) or not math.isfinite(self.max_wall_seconds) or self.max_wall_seconds <= 0):
            raise GoalSearchError("max_wall_seconds must be positive")
        if isinstance(self.min_gain, bool) or not isinstance(self.min_gain, (int, float)) or not math.isfinite(self.min_gain) or self.min_gain < 0:
            raise GoalSearchError("min_gain must be finite and non-negative")
        if isinstance(self.max_monitor_retention_drop, bool) or not isinstance(self.max_monitor_retention_drop, (int, float)) or not math.isfinite(self.max_monitor_retention_drop) or self.max_monitor_retention_drop < 0:
            raise GoalSearchError("max_monitor_retention_drop must be finite and non-negative")
        if isinstance(self.max_monitor_kl_increase, bool) or not isinstance(self.max_monitor_kl_increase, (int, float)) or not math.isfinite(self.max_monitor_kl_increase) or self.max_monitor_kl_increase < 0:
            raise GoalSearchError("max_monitor_kl_increase must be finite and non-negative")


@dataclass(frozen=True)
class SearchObservation:
    round_number: int
    candidate_evaluations: int
    changed_parameters: int
    elapsed_seconds: float
    target_accuracy: float
    retention_accuracy: float
    kl_nats: float
    monitor_retention_accuracy: float
    monitor_kl_nats: float
    initial_monitor_retention_accuracy: float
    initial_monitor_kl_nats: float
    best_gain: float | None
    no_gain_rounds: int


@dataclass(frozen=True)
class NextMoveDecision:
    rule_id: str
    reason: str
    move_policy: Literal["task", "retention_repair", "balanced", "none"]
    terminal_state: TerminalState | None = None

    def to_document(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "reason": self.reason,
            "move_policy": self.move_policy,
            "terminal_state": self.terminal_state,
        }


def diagnose_next_move(
    profile: AcceptanceProfile,
    budgets: SearchBudgets,
    observation: SearchObservation,
) -> NextMoveDecision:
    """Choose the next move policy from public measurements, never from sealed holdout data."""
    if observation.round_number >= budgets.max_rounds:
        return NextMoveDecision("budget_rounds", "The round budget is exhausted", "none", "budget_exhausted")
    if observation.candidate_evaluations >= budgets.max_candidate_evaluations:
        return NextMoveDecision("budget_evaluations", "The evaluation budget is exhausted", "none", "budget_exhausted")
    if observation.elapsed_seconds >= (budgets.max_wall_seconds or math.inf):
        return NextMoveDecision("budget_wall_time", "The wall-time budget is exhausted", "none", "budget_exhausted")
    changed_limit = (
        math.inf
        if budgets.max_changed_parameters is None
        else budgets.max_changed_parameters
    )
    if observation.changed_parameters >= changed_limit:
        return NextMoveDecision("budget_changed_parameters", "The parameter budget is exhausted", "none", "budget_exhausted")

    retention_drop = observation.initial_monitor_retention_accuracy - observation.monitor_retention_accuracy
    kl_increase = observation.monitor_kl_nats - observation.initial_monitor_kl_nats
    if retention_drop > budgets.max_monitor_retention_drop or kl_increase > budgets.max_monitor_kl_increase:
        return NextMoveDecision(
            "monitor_drift",
            f"Monitor drift exceeded policy: retention_drop={retention_drop:.6f}, kl_increase={kl_increase:.6f}",
            "none",
            "monitor_drift",
        )

    if observation.no_gain_rounds >= budgets.max_no_gain_rounds:
        return NextMoveDecision(
            "frontier_stable",
            f"No accepted gain for {observation.no_gain_rounds} rounds",
            "none",
            "no_candidate_cleared_min_gain",
        )
    if observation.target_accuracy < profile.min_target_accuracy:
        return NextMoveDecision(
            "target_gap",
            f"Target accuracy {observation.target_accuracy:.6f} is below {profile.min_target_accuracy:.6f}",
            "task",
        )
    if observation.retention_accuracy < profile.min_retention or (
        profile.max_kl_nats is not None and observation.kl_nats > profile.max_kl_nats
    ):
        return NextMoveDecision(
            "retention_or_kl_gap",
            "The task bar is met, but the retention or KL constraint is not",
            "retention_repair",
        )
    return NextMoveDecision(
        "balanced_frontier",
        "The current point meets the profile bars; search neighboring tradeoffs",
        "balanced",
    )
