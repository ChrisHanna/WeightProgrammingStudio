"""Promotion decision. Every failed gate is recorded; evaluation never stops at the first."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .plan import PLACEBO_ARM, StudyPlan


@dataclass(frozen=True)
class TaskEffect:
    task: str
    base_accuracy: float
    candidate_accuracy: float
    mean_delta: float
    ci_low: float
    ci_high: float
    ci_level: float
    p_value: float
    p_adjusted: float
    clusters: int
    items: int
    seed_deltas: dict[str, float]


@dataclass(frozen=True)
class ControlEffect:
    """Candidates against one control arm on one task, paired by cluster and pooled over seeds."""

    arm: str
    task: str
    control_accuracy: float
    candidate_accuracy: float
    mean_delta: float
    ci_low: float
    ci_high: float
    ci_level: float
    p_value: float
    p_adjusted: float
    clusters: int
    seed_deltas: dict[str, float]  # Candidate minus this control, for the same synthesis seed.
    control_minus_base: float


@dataclass(frozen=True)
class RetentionGate:
    worst_agreement_ci_low: float
    worst_kl_ci_high: float
    items: int


@dataclass(frozen=True)
class RejectionReason:
    code: str
    scope: str
    observed: float | None
    threshold: float | None
    message: str


@dataclass(frozen=True)
class PromotionDecision:
    study_id: str
    label: str
    plan_digest: str
    manifest_digest: str
    promoted: bool
    rejection_reasons: tuple[RejectionReason, ...]
    effects: tuple[TaskEffect, ...]
    retention: RetentionGate | None
    failed_candidates: tuple[str, ...]
    controls: tuple[ControlEffect, ...] = ()
    failed_controls: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decide(
    plan: StudyPlan,
    *,
    plan_digest: str,
    manifest_digest: str,
    effects: tuple[TaskEffect, ...],
    retention: RetentionGate | None,
    failed_candidates: tuple[str, ...] = (),
    controls: tuple[ControlEffect, ...] = (),
    failed_controls: tuple[str, ...] = (),
) -> PromotionDecision:
    limits = plan.thresholds
    reasons: list[RejectionReason] = []

    def reject(code: str, scope: str, observed: float | None, threshold: float | None, message: str) -> None:
        reasons.append(RejectionReason(code, scope, observed, threshold, message))

    if plan.label != "confirmatory":
        reject("NON_CONFIRMATORY_LABEL", "study", None, None, f"a {plan.label!r} plan can never promote")
    for candidate in failed_candidates:
        reject("CANDIDATE_EVALUATION_FAILED", candidate, None, None, "a failed candidate can't be dropped from the study")
    for control in failed_controls:
        reject("CONTROL_EVALUATION_FAILED", control, None, None, "a failed control arm can't be dropped from the study")
    if not failed_controls:
        for arm in plan.comparator_arms:
            if not failed_candidates and {c.task for c in controls if c.arm == arm} != set(plan.tasks):
                reject("MISSING_CONTROL_RESULTS", arm, None, None, "this planned control has no result for every task")
    if not failed_candidates and not failed_controls and {e.task for e in effects} != set(plan.tasks):
        reject("MISSING_TASK_RESULTS", "study", None, None, "not every planned task has a result")
    if not failed_candidates and not failed_controls and retention is None:
        reject("MISSING_RETENTION_RESULTS", "study", None, None, "the retention gates were not measured")

    for effect in effects:
        if effect.ci_low < limits.min_effect:
            reject(
                "EFFECT_CI_BELOW_MINIMUM",
                effect.task,
                effect.ci_low,
                limits.min_effect,
                f"the corrected {effect.ci_level:.4f} CI lower bound is below the minimum effect",
            )
        if effect.p_adjusted > plan.alpha:
            reject(
                "NOT_SIGNIFICANT_AFTER_CORRECTION",
                effect.task,
                effect.p_adjusted,
                plan.alpha,
                f"the {plan.correction}-adjusted p-value exceeds alpha",
            )
        if effect.candidate_accuracy < limits.min_accuracy:
            reject(
                "ACCURACY_BELOW_MINIMUM",
                effect.task,
                effect.candidate_accuracy,
                limits.min_accuracy,
                "pooled candidate accuracy is below the minimum",
            )
        passing = sum(delta >= limits.min_effect for delta in effect.seed_deltas.values())
        fraction = passing / len(effect.seed_deltas) if effect.seed_deltas else 0.0
        if fraction < limits.min_seed_pass_fraction:
            reject(
                "INCONSISTENT_ACROSS_SEEDS",
                effect.task,
                fraction,
                limits.min_seed_pass_fraction,
                "too few synthesis seeds reach the minimum effect",
            )
    for control in controls:
        scope = f"{control.arm}:{control.task}"
        if control.ci_low <= limits.min_control_advantage:
            reject(
                "CONTROL_NOT_BEATEN",
                scope,
                control.ci_low,
                limits.min_control_advantage,
                f"the corrected {control.ci_level:.4f} CI lower bound against this control doesn't clear the margin",
            )
        if control.p_adjusted > plan.alpha:
            reject(
                "CONTROL_NOT_SIGNIFICANT_AFTER_CORRECTION",
                scope,
                control.p_adjusted,
                plan.alpha,
                f"the {plan.correction}-adjusted p-value against this control exceeds alpha",
            )
        passing = sum(delta > limits.min_control_advantage for delta in control.seed_deltas.values())
        fraction = passing / len(control.seed_deltas) if control.seed_deltas else 0.0
        if fraction < limits.min_seed_pass_fraction:
            reject(
                "CONTROL_INCONSISTENT_ACROSS_SEEDS",
                scope,
                fraction,
                limits.min_seed_pass_fraction,
                "too few synthesis seeds beat this control",
            )
        # A random edit shouldn't help. If one does, the items or the evaluator reward noise, whoever "won".
        if control.arm == PLACEBO_ARM and control.control_minus_base >= limits.min_effect:
            reject(
                "PLACEBO_BEATS_BASE",
                scope,
                control.control_minus_base,
                limits.min_effect,
                "the placebo reached the minimum effect against base, so this study can't support a claim",
            )
    if retention is not None:
        if retention.worst_agreement_ci_low < limits.min_retention_agreement:
            reject(
                "RETENTION_AGREEMENT_BELOW_MINIMUM",
                "retention",
                retention.worst_agreement_ci_low,
                limits.min_retention_agreement,
                "the worst candidate's top-1 agreement with base is too low",
            )
        if retention.worst_kl_ci_high > limits.max_retention_kl:
            reject(
                "RETENTION_KL_ABOVE_MAXIMUM",
                "retention",
                retention.worst_kl_ci_high,
                limits.max_retention_kl,
                "the worst candidate's KL from base is too high",
            )
    return PromotionDecision(
        study_id=plan.study_id,
        label=plan.label,
        plan_digest=plan_digest,
        manifest_digest=manifest_digest,
        promoted=not reasons,
        rejection_reasons=tuple(reasons),
        effects=effects,
        retention=retention,
        failed_candidates=failed_candidates,
        controls=controls,
        failed_controls=failed_controls,
    )
