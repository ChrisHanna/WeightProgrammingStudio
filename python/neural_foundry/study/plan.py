"""Preregistered study plan, frozen by the SHA-256 of its exact file bytes.

The plan names how many holdout draws to make but never their seeds. Those are drawn
from OS entropy by the custodian after the proposal freezes, so a proposer that reads
the plan still can't regenerate the holdout.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "nf.study.plan.v1"
TASKS = ("parity", "boolean", "comparison")
METRICS = ("forced_choice_accuracy", "top1_accuracy")
CORRECTIONS = ("holm",)
EVALUATORS = ("pythia_mlp_override", "synthetic_bernoulli")
LABELS = ("confirmatory", "harness_demo", "harness_test")
MIN_CONFIRMATORY_ITEMS_PER_TASK = 200
MIN_CONFIRMATORY_SEEDS = 3


class PlanError(ValueError):
    """The plan is malformed or breaks a preregistration rule."""


class PlanTamperedError(PlanError):
    """The plan file no longer matches the digest it was frozen under."""


@dataclass(frozen=True)
class Thresholds:
    min_effect: float  # the corrected CI lower bound on the paired delta must reach this
    min_accuracy: float  # pooled candidate accuracy on each task
    min_seed_pass_fraction: float  # share of synthesis seeds whose point delta reaches min_effect
    min_retention_agreement: float  # worst candidate's CI lower bound on top-1 agreement with base
    max_retention_kl: float  # worst candidate's CI upper bound on KL(base || candidate), in nats


@dataclass(frozen=True)
class StudyPlan:
    study_id: str
    label: str
    hypothesis: str
    tasks: tuple[str, ...]
    clusters_per_task_per_draw: int
    holdout_draws: int
    retention_items: int
    development_prefixes: tuple[str, ...]
    holdout_prefixes: tuple[str, ...]
    number_domain: tuple[int, int]
    synthesis_seeds: tuple[int, ...]
    analysis_seed: int
    primary_metric: str
    secondary_metrics: tuple[str, ...]
    alpha: float
    correction: str
    bootstrap_resamples: int
    permutation_resamples: int
    thresholds: Thresholds
    evaluator: str
    checkpoint_dir: str | None = None
    checkpoint_digest: str | None = None
    torch_threads: int = 2
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise PlanError(f"unknown plan schema {self.schema_version!r}")
        if self.label not in LABELS:
            raise PlanError(f"label must be one of {LABELS}")
        if not self.tasks or len(set(self.tasks)) != len(self.tasks) or not set(self.tasks) <= set(TASKS):
            raise PlanError(f"tasks must be distinct members of {TASKS}")
        if self.primary_metric not in METRICS or not set(self.secondary_metrics) <= set(METRICS):
            raise PlanError(f"metrics must be members of {METRICS}")
        if self.correction not in CORRECTIONS:
            raise PlanError(f"correction must be one of {CORRECTIONS}")
        if self.evaluator not in EVALUATORS:
            raise PlanError(f"evaluator must be one of {EVALUATORS}")
        if not 0.0 < self.alpha < 0.5:
            raise PlanError("alpha must be in (0, 0.5)")
        if self.clusters_per_task_per_draw < 20 or self.clusters_per_task_per_draw % 2:
            raise PlanError("clusters_per_task_per_draw must be even and at least 20, so aggregates can't expose items")
        if self.holdout_draws < 1 or self.retention_items < 20:
            raise PlanError("holdout_draws must be at least 1 and retention_items at least 20")
        if not self.synthesis_seeds or len(set(self.synthesis_seeds)) != len(self.synthesis_seeds):
            raise PlanError("synthesis_seeds must be non-empty and distinct")
        if not self.holdout_prefixes or not self.development_prefixes:
            raise PlanError("prefix lists must be non-empty")
        if set(self.holdout_prefixes) & set(self.development_prefixes):
            raise PlanError("holdout_prefixes and development_prefixes must be disjoint")
        low, high = self.number_domain
        if low < 0 or high <= low:
            raise PlanError("number_domain must be an increasing non-negative range")
        if self.bootstrap_resamples < 200 or self.permutation_resamples < 200:
            raise PlanError("bootstrap_resamples and permutation_resamples must be at least 200")
        if self.evaluator == "pythia_mlp_override" and not (self.checkpoint_dir and self.checkpoint_digest):
            raise PlanError("the pythia evaluator needs checkpoint_dir and checkpoint_digest")
        if self.label == "confirmatory":
            if self.evaluator == "synthetic_bernoulli":
                raise PlanError("a confirmatory plan can't use the synthetic evaluator")
            if self.items_per_task < MIN_CONFIRMATORY_ITEMS_PER_TASK:
                raise PlanError(f"a confirmatory plan needs {MIN_CONFIRMATORY_ITEMS_PER_TASK}+ items per task")
            if len(self.synthesis_seeds) < MIN_CONFIRMATORY_SEEDS:
                raise PlanError(f"a confirmatory plan needs {MIN_CONFIRMATORY_SEEDS}+ synthesis seeds")

    @property
    def items_per_task(self) -> int:
        return self.clusters_per_task_per_draw * self.holdout_draws * len(self.holdout_prefixes)

    @property
    def corrected_ci_level(self) -> float:
        """Simultaneous level for reported effect CIs: Bonferroni over the primary hypotheses."""
        return 1.0 - self.alpha / len(self.tasks)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_bytes(self) -> bytes:
        return (json.dumps(self.to_dict(), sort_keys=True, indent=2) + "\n").encode("utf-8")

    @classmethod
    def from_dict(cls, doc: dict[str, Any]) -> StudyPlan:
        known = {f.name for f in fields(cls)}
        unknown = set(doc) - known
        if unknown:
            raise PlanError(f"unknown plan fields: {sorted(unknown)}")
        values = dict(doc)
        try:
            raw = dict(values["thresholds"])
            values["thresholds"] = Thresholds(**raw)
            for name in ("tasks", "development_prefixes", "holdout_prefixes", "synthesis_seeds", "secondary_metrics"):
                values[name] = tuple(values[name])
            values["number_domain"] = tuple(values["number_domain"])
            return cls(**values)
        except (KeyError, TypeError) as exc:
            raise PlanError(f"malformed plan: {exc}") from exc


def digest_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def freeze_plan(plan: StudyPlan, path: str | Path) -> str:
    """Write the plan once and return its digest. An existing file is never overwritten."""
    data = plan.to_bytes()
    with open(path, "xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return digest_bytes(data)


def load_frozen_plan(path: str | Path, expected_digest: str) -> StudyPlan:
    data = Path(path).read_bytes()
    actual = digest_bytes(data)
    if actual != expected_digest:
        raise PlanTamperedError(f"plan digest is {actual}, but it was frozen as {expected_digest}")
    return StudyPlan.from_dict(json.loads(data))
