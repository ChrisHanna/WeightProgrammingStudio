from __future__ import annotations

import pytest

from neural_foundry.gym_job import GymJobError
from neural_foundry.gym_service import GymStudioService
from test_gym_job import _response_service


def test_select_training_dataset_preserves_parent_provenance_and_rejects_unknown_ids(tmp_path):
    def provider(*, source_id, **kwargs):
        return [{"scenario_id": source_id, "input": {"contract_text": f"Contract {source_id}."},
                 "reference": "answer", "group_id": f"family-{source_id}"}]

    service, job, candidate, plan = _response_service({"type": "string"}, tmp_path, reference="answer",
                                                       provider=provider, split="train")
    attempt_ids = tuple(row["attempt_id"] for row in plan["expected_attempts"])
    service.record_practice_run_document({"practice_plan": plan, "attempts": [
        {"attempt_id": attempt_id, "response": "answer"} for attempt_id in attempt_ids]})
    service.record_practice_run(plan)
    source = service.freeze_training_dataset(job,
        [{"practice_plan_digest": plan["plan_digest"], "attempt_id": attempt_id} for attempt_id in attempt_ids],
        dataset_id="all-train", allow_reported_responses=True)
    source_examples = service.load_training_examples(source["dataset_digest"])
    selected_ids = [source_examples[0]["scenario_id"]]
    selected = service.select_training_dataset(source["dataset_digest"], selected_ids, dataset_id="failing-only",
        selection_policy={"kind": "base_failure_only_v1", "baseline_digest": "sha256:" + "a" * 64,
                          "minimum_failures": 10, "observed_failures": 10})
    readback = GymStudioService(service.components, state_dir=tmp_path).load_training_examples(selected["dataset_digest"])
    assert [example["scenario_id"] for example in readback] == selected_ids
    assert selected["parent_dataset_digest"] == source["dataset_digest"]
    assert selected["selection_policy"]["kind"] == "base_failure_only_v1"
    with pytest.raises(GymJobError, match="come from the frozen parent"):
        service.select_training_dataset(source["dataset_digest"], ["validation-only-id"], dataset_id="invalid",
            selection_policy={"kind": "base_failure_only_v1"})
