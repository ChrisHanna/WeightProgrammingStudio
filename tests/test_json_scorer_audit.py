from __future__ import annotations

import pytest

from neural_foundry import reference_json_extraction
from neural_foundry.gym_job import GymJobError
from neural_foundry.json_scorer_audit import audit_reference_scorer


def test_scorer_audit_checks_public_counterfactuals_and_determinism() -> None:
    task = reference_json_extraction.reference_task(rows_per_template=2)
    report = audit_reference_scorer(task)
    assert report["passed"] is True and report["cases"] == 12
    assert report["scorer_digest"] == task.job.scorer.implementation_digest
    assert report["splits"] == ["train", "validation"]
    assert report["evidence_class"] == "synthetic" and report["release_authorized"] is False


def test_scorer_audit_rejects_constant_reward(monkeypatch) -> None:
    task = reference_json_extraction.reference_task()
    monkeypatch.setattr(reference_json_extraction, "score_response", lambda **kwargs: {"exact_match": 1.0})
    with pytest.raises(GymJobError, match="counterfactual"):
        audit_reference_scorer(task)


def test_scorer_audit_rejects_boolean_metrics(monkeypatch) -> None:
    task = reference_json_extraction.reference_task()
    monkeypatch.setattr(reference_json_extraction, "score_response",
                        lambda *, reference, response, **kwargs: {"exact_match": reference == response})
    with pytest.raises(GymJobError, match="counterfactual"):
        audit_reference_scorer(task)