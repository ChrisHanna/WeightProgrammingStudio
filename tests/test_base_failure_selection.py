from __future__ import annotations

from neural_foundry.base_failure_selection import load_base_failure_report
from neural_foundry.gym_job import GymJobError
from neural_foundry.gym_service import _document_digest
from neural_foundry.reference_json_extraction import generate_scenarios, reference_task
import pytest


def _report(*, failures: int):
    task = reference_task(rows_per_template=1)
    rows = generate_scenarios(split="train", seed=17, parameters={"rows_per_template": 1})
    outcomes = [{"scenario_id": row["scenario_id"], "exact_match": float(index >= failures)}
                for index, row in enumerate(rows)]
    report = {"kind": "nf_json_public_split_baseline_v1", "scenario_origin": "synthetic",
              "current_job_digest": task.job.digest, "seed": 17, "rows_per_template": 1,
              "base_state_digest": "sha256:" + "a" * 64, "base_candidate_digest": "sha256:" + "b" * 64,
              "holdout_accessed": False,
              "train": {"score": {"responses": outcomes},
                        "failures": [row["scenario_id"] for row in outcomes if row["exact_match"] == 0.0],
                        "failure_count": failures}}
    report["evidence_digest"] = _document_digest(report)
    return task, report


def _write(tmp_path, report):
    import json

    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(report))
    return path


def test_frozen_baseline_selects_exact_train_failures_and_passing_ids(tmp_path):
    task, report = _report(failures=10)
    checked, failing, passing = load_base_failure_report(_write(tmp_path, report), task=task, seed=17,
        rows_per_template=1, base_state_digest=report["base_state_digest"],
        base_candidate_digest=report["base_candidate_digest"])
    assert checked == report
    assert len(failing) == 10 and len(passing) == 0


def test_base_failure_gate_refuses_fewer_than_ten(tmp_path):
    task, report = _report(failures=9)
    with pytest.raises(GymJobError, match="9 failing train cases; minimum is 10"):
        load_base_failure_report(_write(tmp_path, report), task=task, seed=17, rows_per_template=1,
                                base_state_digest=report["base_state_digest"])


def test_base_failure_gate_rejects_reordered_roster_and_wrong_model(tmp_path):
    task, report = _report(failures=10)
    report["train"]["score"]["responses"].reverse()
    report["evidence_digest"] = _document_digest({key: value for key, value in report.items() if key != "evidence_digest"})
    with pytest.raises(GymJobError, match="incomplete or reordered"):
        load_base_failure_report(_write(tmp_path, report), task=task, seed=17, rows_per_template=1)
    task, report = _report(failures=10)
    with pytest.raises(GymJobError, match="another loaded model state"):
        load_base_failure_report(_write(tmp_path, report), task=task, seed=17, rows_per_template=1,
                                base_state_digest="sha256:" + "c" * 64)


def test_committed_qwen_baseline_matches_expanded_task_and_passes_minimum():
    from pathlib import Path

    task = reference_task(rows_per_template=32)
    path = Path(__file__).resolve().parents[1] / "docs" / "evidence" / "weight-edit-expanded-baseline-qwen3-train-validation-v1.json"
    report, failures, passing = load_base_failure_report(path, task=task, seed=17, rows_per_template=32,
        base_state_digest="sha256:fba0b82e1a3bec86d7ecfb867a2eb15ce3a6315b5384ba6091c2c2978fd75f11",
        base_candidate_digest="sha256:14bdb1cc215be31b28526602fc442d7b83b00e7abaee63edad960e2dd87cab9c")
    assert len(failures) == 155 and len(passing) == 165
    assert report["validation"]["failure_count"] == 18
