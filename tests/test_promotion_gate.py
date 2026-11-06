"""Promotion is decided by the independent evaluator's measures, not the Gym's own prompts."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from neural_foundry import promotion_gate as pg
from neural_foundry import pythia

EVIDENCE = Path(__file__).resolve().parents[1] / "docs" / "evidence" / "milestone1" / "report.json"


@pytest.fixture(scope="module")
def harmful() -> dict:
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def _measure(value: float) -> dict:
    return {"value": value}


def _report(*, agreement: float = 1.0, kl: float = 0.001, ratio: float = 1.001, gain: float = 0.5, control_kl: float = 2.0, control_agreement: float = 0.5) -> dict:
    """A synthetic evaluator report, written by hand to exercise each limit."""
    text = {"top1_agreement": _measure(agreement), "kl_nats_per_token": _measure(kl), "perplexity_ratio": _measure(ratio)}
    return {
        "kind": pg.EVALUATOR_REPORT_KIND,
        "target": {"metrics": {"accuracy_delta": _measure(gain)}},
        "candidate": {"all_tasks": {"top1_agreement": _measure(agreement)}, "text": text},
        "retention_matrix": {"capitals": {"candidate": {"top1_agreement": _measure(agreement)}}},
        "controls": {
            "noop": {"passed": True},
            "random": {"random_seed_1": {"text": {"top1_agreement": _measure(control_agreement), "kl_nats_per_token": _measure(control_kl)}}},
        },
    }


def test_the_recorded_harmful_edit_is_rejected_on_the_evaluator_measures(harmful: dict) -> None:
    decision = pg.decide_promotion(harmful)
    assert decision["promote_eligible"] is False
    assert decision["status"] == "rejected"
    failed = set(decision["failed_checks"])
    assert {"pooled_retention_task_agreement", "free_text_agreement", "kl_nats_per_token", "perplexity_ratio"} <= failed
    assert {f"retention_task_agreement:{task}" for task in harmful["retention_matrix"]} <= failed
    # It really did hit its target, so this rejection isn't from a failed target check.
    assert "target_accuracy_gain" not in failed


def test_a_benign_report_passes_and_each_limit_can_reject_it() -> None:
    assert pg.decide_promotion(_report())["status"] == "passed"
    for change, failure in (
        ({"agreement": 0.9}, "free_text_agreement"),
        ({"kl": 1.0}, "kl_nats_per_token"),
        ({"ratio": 1.06}, "perplexity_ratio"),
        ({"gain": 0.0}, "target_accuracy_gain"),
        ({"control_kl": 0.0005}, "kl_below_every_random_control"),
        ({"control_agreement": 1.0}, "agreement_above_every_random_control"),
    ):
        decision = pg.decide_promotion(_report(**change))
        assert decision["promote_eligible"] is False
        assert failure in decision["failed_checks"]


def test_a_failed_noop_or_non_finite_measure_rejects() -> None:
    report = _report()
    report["controls"]["noop"]["passed"] = False
    assert "noop_control" in pg.decide_promotion(report)["failed_checks"]
    assert pg.decide_promotion(_report(ratio=math.inf))["promote_eligible"] is False
    assert pg.decide_promotion(_report(kl=math.nan))["promote_eligible"] is False


@pytest.mark.parametrize(
    "remove",
    [
        lambda r: r.pop("controls"),
        lambda r: r["controls"]["random"].clear(),
        lambda r: r["retention_matrix"].clear(),
        lambda r: r["candidate"]["text"].pop("kl_nats_per_token"),
        lambda r: r.update(kind="nf_gym_report"),
    ],
)
def test_an_incomplete_report_is_refused(remove) -> None:  # type: ignore[no-untyped-def]
    report = copy.deepcopy(_report())
    remove(report)
    decision = pg.decide_promotion(report)
    assert decision["promote_eligible"] is False
    assert decision["status"] == "not_evaluated"


def test_recorded_calibration_promotes_nothing_and_the_no_op_fails_on_target_gain() -> None:
    document = json.loads(EVIDENCE.parents[1].joinpath("promotion_gate_calibration.json").read_text(encoding="utf-8"))
    arms = {row["arm"]: row for row in document["arms"]}
    assert len(arms) == 7 and not any(row["promote_eligible"] for row in arms.values())
    assert arms["no-op (exact)"]["status"] == "not_evaluated"
    assert "target_accuracy_gain" in arms["no-op (near, scale 1 + 1e-9)"]["failed_checks"]


def test_thresholds_are_the_frozen_values() -> None:
    assert (pg.MIN_POOLED_TASK_AGREEMENT, pg.MIN_TEXT_AGREEMENT, pg.MIN_PER_TASK_AGREEMENT) == (0.98, 0.98, 0.95)
    assert (pg.MAX_PERPLEXITY_RATIO, pg.MIN_TARGET_ACCURACY_GAIN) == (1.05, 0.01)


def test_kl_limit_is_the_perplexity_limit_in_nats() -> None:
    assert pg.MAX_KL_NATS_PER_TOKEN == math.log(pg.MAX_PERPLEXITY_RATIO)


def test_gym_study_uses_the_gate_and_keeps_its_own_check_apart() -> None:
    report = pythia.run_pythia_gym_task_learning_study(pythia.build_mini_pythia_model(), target_strength=85.0)
    verdict = report.promotion_verification
    assert verdict["kind"] == "nf_promotion_decision_v1"
    assert verdict["measured_by"] == "neural_foundry.independent_eval"
    assert "gym_self_check" in verdict
    # The fixture can't take the evaluator's corpus, so it isn't eligible whatever the Gym's own check says.
    assert verdict["promote_eligible"] is False
