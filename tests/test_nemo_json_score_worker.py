from __future__ import annotations

import copy
import json

import pytest

from neural_foundry.gym_job import GymJobError
from neural_foundry.gym_service import GymStudioService
from neural_foundry.nemo_json_score_worker import score_request
from neural_foundry.reference_json_extraction import reference_task, score_response


def _request(tmp_path):
    task = reference_task(rows_per_template=1)
    service = GymStudioService(state_dir=tmp_path)
    task.bind(service)
    batch = service.materialize_public_scenarios(task.job, split="validation", seed=7)
    scenario = batch["scenarios"][0]
    reference = service._scenario_references[(batch["manifest_digest"], scenario["scenario_id"])]
    return {"job": task.job.to_document(), "scenario": scenario, "reference": reference,
            "response_text": json.dumps(reference)}


def test_scorer_worker_matches_registered_score_and_rejects_bad_json(tmp_path) -> None:
    request = _request(tmp_path)
    result = score_request(request)
    assert result["score"] == score_response(reference=request["reference"], response=request["reference"])["exact_match"] == 1.0
    assert result["scorer_digest"] == request["job"]["scorer"]["implementation_digest"]
    for text in ("{}", "not json", '{"quantity":1,"quantity":2}', '{"quantity":NaN}'):
        result = score_request({**request, "response_text": text})
        assert result["score"] == 0.0 and result["output_valid"] is False
    wrong = {**request["reference"], "customer": "wrong"}
    result = score_request({**request, "response_text": json.dumps(wrong)})
    assert result["score"] == score_response(reference=request["reference"], response=wrong)["exact_match"] == 0.0
    assert result["output_valid"] is True


def test_scorer_worker_rejects_identity_and_reference_tampering(tmp_path) -> None:
    request = _request(tmp_path)
    altered = copy.deepcopy(request)
    altered["job"]["scorer"]["implementation_digest"] = "sha256:" + "0" * 64
    with pytest.raises(GymJobError, match="digest"):
        score_request(altered)
    with pytest.raises(GymJobError, match="reference"):
        score_request({**request, "reference": {**request["reference"], "customer": "changed"}})
    with pytest.raises(GymJobError, match="public"):
        score_request({**request, "scenario": {**request["scenario"], "split": "holdout"}})