from __future__ import annotations

import pytest

from neural_foundry.gym_job import GymJobError, validate_task_instance
from neural_foundry.gym_service import GymStudioService
from neural_foundry.reference_json_extraction import (
    OUTPUT_SCHEMA, TEMPLATES, generate_scenarios, reference_task, score_response, verify_response,
)


def test_synthetic_reference_splits_and_exact_scoring(tmp_path) -> None:
    task = reference_task(rows_per_template=4)
    service = GymStudioService(state_dir=tmp_path)
    task.bind(service)
    panels = {split: generate_scenarios(split=split, seed=17, parameters={"rows_per_template": 4})
              for split in ("train", "validation", "holdout")}
    assert panels["train"] == generate_scenarios(split="train", seed=17, parameters={"rows_per_template": 4})
    assert all(len(rows) == 8 for rows in panels.values())
    assert not (set(TEMPLATES["holdout"]) & (set(TEMPLATES["train"]) | set(TEMPLATES["validation"])))
    assert all(set(row["family_id"] for row in rows) == set(TEMPLATES[split]) for split, rows in panels.items())
    for split, rows in panels.items():
        for row in rows:
            validate_task_instance(OUTPUT_SCHEMA, row["reference"])
            assert score_response(reference=row["reference"], response=row["reference"]) == {"exact_match": 1.0}
            assert not verify_response(reference=row["reference"], response={**row["reference"], "quantity": 0})["passed"]
        if split != "holdout":
            manifest = service.materialize_public_scenarios(task.job, split=split, seed=17)
            assert len(manifest["scenarios"]) == len(rows)
            assert "reference" not in manifest["scenarios"][0]
    with pytest.raises(GymJobError, match="holdout"):
        service.materialize_public_scenarios(task.job, split="holdout", seed=17)


def test_projection_and_invalid_responses() -> None:
    task = reference_task()
    row = generate_scenarios(split="train", seed=1, parameters={"rows_per_template": 1})[0]
    projected = task.project(row["input"], row["reference"])
    assert projected["format"] == "json_extraction_v1"
    assert projected["messages"][-1]["role"] == "assistant"
    for invalid in (None, {}, {**row["reference"], "extra": 1}, {**row["reference"], "quantity": True}):
        assert score_response(reference=row["reference"], response=invalid) == {"exact_match": 0.0}
    with pytest.raises(GymJobError):
        generate_scenarios(split="holdout", seed=1, parameters={"rows_per_template": 0})