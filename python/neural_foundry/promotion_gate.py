"""Decide promotion from an independent evaluator report, and only from that report.

The input is the document `independent_eval.evaluate_candidate` returns. Nothing here reads a Gym's
own scores, a holdout, or how the candidate was made. A report that is missing a measure is refused.

The thresholds restate limits that `evals.DEFAULT_POLICIES["production"]` and `verify_promotion`
already set before Milestone 1's edit was measured. None was fitted to a candidate:

* `min_retention_ratio` (0.98) sets the agreement floors over all retention tasks and over free text.
* The 5% out-of-target limit (`max_out_of_target_drop_pct`) sets the floor for any single task, and
  the perplexity limit: 5% more perplexity is the same 5% budget on the corpus's own text.
* KL(base || candidate) is the extra log-loss on text the base model would generate, so ln(1.05)
  nats per token spends the same budget in the evaluator's other unit.

The limits apply to point estimates. With 16 prompts in a task, 0.95 allows no changed answer.

A control check needs no threshold. Noise of the candidate's size and support is the evaluator's
measure of an edit with no selectivity, so a candidate must disturb the model less than every one.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

EVALUATOR_REPORT_KIND = "nf_independent_eval_report_v1"
MIN_POOLED_TASK_AGREEMENT = 0.98
MIN_TEXT_AGREEMENT = 0.98
MIN_PER_TASK_AGREEMENT = 0.95
MAX_PERPLEXITY_RATIO = 1.05
MAX_KL_NATS_PER_TOKEN = math.log(MAX_PERPLEXITY_RATIO)
MIN_TARGET_ACCURACY_GAIN = 0.01  # `min_target_gain` in the production policy.


def _check(name: str, observed: float, threshold: float, *, at_most: bool = False) -> dict[str, Any]:
    passed = observed <= threshold if at_most else observed >= threshold
    return {"check": name, "passed": bool(passed), "observed": observed, "threshold": threshold, "direction": "at_most" if at_most else "at_least"}


def _control_checks(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    controls = report["controls"]
    random_controls = controls["random"]
    if not random_controls:
        raise KeyError("controls.random is empty")
    candidate = report["candidate"]["text"]
    agreement = [float(c["text"]["top1_agreement"]["value"]) for c in random_controls.values()]
    kl = [float(c["text"]["kl_nats_per_token"]["value"]) for c in random_controls.values()]
    observed_kl = float(candidate["kl_nats_per_token"]["value"])
    observed_agreement = float(candidate["top1_agreement"]["value"])
    return [
        {"check": "noop_control", "passed": controls["noop"]["passed"] is True, "observed": controls["noop"]["passed"]},
        {"check": "kl_below_every_random_control", "passed": observed_kl < min(kl), "observed": observed_kl, "threshold": min(kl), "direction": "below"},
        {
            "check": "agreement_above_every_random_control",
            "passed": observed_agreement > max(agreement),
            "observed": observed_agreement,
            "threshold": max(agreement),
            "direction": "above",
        },
    ]


def _checks(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    if report.get("kind") != EVALUATOR_REPORT_KIND:
        raise KeyError(f"kind is {report.get('kind')!r}")
    tasks = report["retention_matrix"]
    if not tasks:
        raise KeyError("retention_matrix is empty")
    text = report["candidate"]["text"]
    target = report["target"]["metrics"]
    checks = [
        _check("target_accuracy_gain", float(target["accuracy_delta"]["value"]), MIN_TARGET_ACCURACY_GAIN),
        _check("pooled_retention_task_agreement", float(report["candidate"]["all_tasks"]["top1_agreement"]["value"]), MIN_POOLED_TASK_AGREEMENT),
        *(_check(f"retention_task_agreement:{name}", float(row["candidate"]["top1_agreement"]["value"]), MIN_PER_TASK_AGREEMENT) for name, row in sorted(tasks.items())),
        _check("free_text_agreement", float(text["top1_agreement"]["value"]), MIN_TEXT_AGREEMENT),
        _check("kl_nats_per_token", float(text["kl_nats_per_token"]["value"]), MAX_KL_NATS_PER_TOKEN, at_most=True),
        _check("perplexity_ratio", float(text["perplexity_ratio"]["value"]), MAX_PERPLEXITY_RATIO, at_most=True),
        *_control_checks(report),
    ]
    for check in checks:
        observed = check["observed"]
        if isinstance(observed, float) and not math.isfinite(observed):
            check["passed"] = False
    return checks


def render_decision(decision: Mapping[str, Any]) -> str:
    lines = [f"# Promotion decision: `{decision['status']}`", "", f"Measured by: {decision['measured_by']}", ""]
    if decision.get("reason"):
        lines += [str(decision["reason"]), ""]
    if decision["checks"]:
        lines += ["| Check | Result | Observed | Limit |", "| --- | --- | ---: | ---: |"]
        lines += [
            f"| {c['check']} | {'pass' if c['passed'] else 'FAIL'} | {c['observed']} | {c.get('threshold', '')} |"
            for c in decision["checks"]
        ]
    return "\n".join(lines) + "\n"


def _decision(status: str, checks: list[dict[str, Any]], reason: str | None) -> dict[str, Any]:
    decision = {
        "schema_version": 1,
        "kind": "nf_promotion_decision_v1",
        "measured_by": "neural_foundry.independent_eval",
        "promote_eligible": status == "passed",
        "status": status,
        "checks": checks,
        "failed_checks": [c["check"] for c in checks if not c["passed"]],
        "reason": reason,
    }
    decision["markdown_dashboard"] = render_decision(decision)
    return decision


def not_evaluated(reason: str) -> dict[str, Any]:
    """No evaluator measurement exists, so there is nothing to promote on."""
    return _decision("not_evaluated", [], reason)


def decide_promotion(report: Mapping[str, Any]) -> dict[str, Any]:
    """Promote only if every check passes. An incomplete report is refused, never given the benefit of the doubt."""
    try:
        checks = _checks(report)
    except (KeyError, TypeError, ValueError) as exc:
        return not_evaluated(f"The evaluator report is incomplete: {exc!r}")
    passed = all(c["passed"] for c in checks)
    return _decision("passed" if passed else "rejected", checks, None)
