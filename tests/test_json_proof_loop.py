from __future__ import annotations

from contextlib import contextmanager

from neural_foundry import json_proof_loop
from neural_foundry.gym_service import ExecutionResponse
from neural_foundry.reference_json_extraction import generate_scenarios, reference_task


def _factory(responses, *, fail=False):
    @contextmanager
    def factory():
        class Model:
            def loaded_identity(self):
                return {"served_model_identity_verified": False}

            def infer(self, interaction):
                if fail:
                    raise RuntimeError("private details must not leave the evaluator")
                return ExecutionResponse(responses.pop(0), {})

        yield Model()
    return factory


def test_synthetic_holdout_compares_three_models_with_common_scorer() -> None:
    task = reference_task(rows_per_template=1)
    rows = generate_scenarios(split="holdout", seed=31, parameters={"rows_per_template": 1})
    correct = [row["reference"] for row in rows]
    models = {"base": _factory([None, None]), "sft": _factory(list(correct)),
              "frontier": _factory([correct[0], None])}
    report = json_proof_loop.compare_synthetic_holdout(task, seed=31, models=models)
    assert report["complete"] is True
    assert [report["scorecards"][role]["value"] for role in ("base", "sft", "frontier")] == [0.0, 1.0, 0.5]
    assert report["evidence_class"] == "synthetic" and report["release_authorized"] is False
    assert report["independent_custody_verified"] is False and report["served_model_identity_verified"] is False
    assert "reference" not in str(report) and "responses" not in report


def test_incomplete_comparison_never_scores_partial_results(monkeypatch) -> None:
    task = reference_task(rows_per_template=1)
    models = {"base": _factory([None, None]), "sft": _factory([None, None]), "frontier": _factory([], fail=True)}
    monkeypatch.setattr(json_proof_loop, "score_response", lambda **kwargs: (_ for _ in ()).throw(AssertionError("scored")))
    report = json_proof_loop.compare_synthetic_holdout(task, seed=31, models=models)
    assert report["complete"] is False and report["scorecards"] is None
    assert report["failure"] == {"stage": "frontier", "reason": "RuntimeError"}
    missing = json_proof_loop.compare_synthetic_holdout(task, seed=31, models={"base": models["base"]})
    assert missing["complete"] is False and missing["scorecards"] is None