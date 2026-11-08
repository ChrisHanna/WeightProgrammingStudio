from __future__ import annotations

import json
import urllib.request

import pytest

from neural_foundry.gym_job import GymJobError
from neural_foundry.json_frontier_model import JsonFrontierModel, _NoRedirect
from neural_foundry.reference_json_extraction import generate_scenarios, reference_task


def test_frontier_requires_explicit_secure_endpoint_and_does_not_report_secrets(monkeypatch) -> None:
    for endpoint in ("http://example.com/v1/chat/completions", "https://user:pass@example.com/v1", "https://example.com/v1?key=secret"):
        with pytest.raises(GymJobError, match="endpoint"):
            JsonFrontierModel(endpoint=endpoint, model="frontier-snapshot", api_key="secret")
    for name in ("NF_FRONTIER_URL", "NF_FRONTIER_MODEL", "NF_FRONTIER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(GymJobError):
        JsonFrontierModel.from_env()
    assert _NoRedirect().redirect_request(None, None, 302, "", None, "https://other.example") is None
    model = JsonFrontierModel(endpoint="https://example.com/v1/chat/completions", model="frontier-snapshot", api_key="secret")
    assert "secret" not in str(model.loaded_identity())
    assert model.loaded_identity()["served_model_identity_verified"] is False


def test_frontier_scores_only_completed_response_from_selected_model(monkeypatch) -> None:
    model = JsonFrontierModel(endpoint="http://127.0.0.1:8000/v1/chat/completions", model="snapshot", api_key="secret")
    row = generate_scenarios(split="holdout", seed=7, parameters={"rows_per_template": 1})[0]
    interaction = reference_task().project(row["input"], None)
    document = {"model": "snapshot", "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(row["reference"])}}]}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, limit):
            return json.dumps(document).encode()

    class Opener:
        def open(self, request, timeout):
            assert request.get_header("Authorization") == "Bearer secret"
            assert json.loads(request.data)["response_format"]["type"] == "json_schema"
            return Response()

    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: Opener())
    assert model.infer(interaction).response == row["reference"]
    document["choices"][0]["finish_reason"] = "length"
    with pytest.raises(GymJobError, match="incomplete"):
        model.infer(interaction)
    document["choices"][0]["finish_reason"] = "stop"
    document["model"] = "another-model"
    with pytest.raises(GymJobError, match="incomplete"):
        model.infer(interaction)