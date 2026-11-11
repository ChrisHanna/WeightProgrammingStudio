from __future__ import annotations

import json
import urllib.request

import pytest

from neural_foundry.gym_job import GymJobError
from neural_foundry.json_frontier_model import JsonFrontierModel, _NoRedirect, OPENAI_DEFAULT_MODEL, OPENAI_CHAT_COMPLETIONS_URL
from neural_foundry.reference_json_extraction import generate_scenarios, reference_task


def test_frontier_requires_explicit_secure_endpoint_and_does_not_report_secrets(monkeypatch, tmp_path) -> None:
    for endpoint in ("http://example.com/v1/chat/completions", "https://user:pass@example.com/v1", "https://example.com/v1?key=secret"):
        with pytest.raises(GymJobError, match="endpoint"):
            JsonFrontierModel(endpoint=endpoint, model="frontier-snapshot", api_key="secret")
    for name in ("NF_FRONTIER_URL", "NF_FRONTIER_MODEL", "NF_FRONTIER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(GymJobError):
        JsonFrontierModel.from_env(env_file=tmp_path / "absent.env")
    assert _NoRedirect().redirect_request(None, None, 302, "", None, "https://other.example") is None
    model = JsonFrontierModel(endpoint="https://example.com/v1/chat/completions", model="frontier-snapshot", api_key="secret")
    assert "secret" not in str(model.loaded_identity())
    assert model.loaded_identity()["served_model_identity_verified"] is False


def test_frontier_scores_only_completed_response_from_selected_model(monkeypatch) -> None:
    model = JsonFrontierModel(endpoint="http://127.0.0.1:8000/v1/chat/completions", model="snapshot", api_key="secret")
    row = generate_scenarios(split="holdout", seed=7, parameters={"rows_per_template": 1})[0]
    interaction = reference_task().project(row["input"], None)
    document = {"id": "chatcmpl-test-only", "created": 1700000000, "system_fingerprint": "fp_test_only",
                "model": "snapshot", "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(row["reference"])}}]}

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
    result = model.infer(interaction)
    assert result.response == row["reference"]
    receipt = result.interaction["api_response_receipt"]
    assert receipt["response_id"] == "chatcmpl-test-only"
    assert receipt["system_fingerprint"] == "fp_test_only"
    assert receipt["requested_at_unix_ns"] > 0
    assert "secret" not in json.dumps(receipt)
    model._endpoint, model._model = OPENAI_CHAT_COMPLETIONS_URL, OPENAI_DEFAULT_MODEL
    document["model"] = OPENAI_DEFAULT_MODEL
    model.require_proof_configuration()
    result = model.infer(interaction)
    model.verify_response_receipt(interaction, result)
    from neural_foundry.gym_service import ExecutionResponse
    with pytest.raises(GymJobError, match="receipt"):
        model.verify_response_receipt(interaction, ExecutionResponse({}, result.interaction))
    with pytest.raises(GymJobError, match="receipt"):
        model.verify_response_receipt({**interaction, "messages": []}, result)
    model._response_receipts.clear()
    with pytest.raises(GymJobError, match="receipt"):
        model.verify_response_receipt(interaction, result)
    document["choices"][0]["finish_reason"] = "length"
    with pytest.raises(GymJobError, match="incomplete"):
        model.infer(interaction)
    document["choices"][0]["finish_reason"] = "stop"
    document["model"] = "another-model"
    with pytest.raises(GymJobError, match="incomplete"):
        model.infer(interaction)


def test_frontier_configuration_pins_shared_prompt_scorer_and_decoding(monkeypatch) -> None:
    from neural_foundry import json_frontier_model
    from neural_foundry.gym_service import _document_digest

    configuration = json_frontier_model.frontier_proof_configuration()
    assert configuration["model"] == "gpt-6-luna"
    assert configuration["reasoning_effort"] == "none"
    assert configuration["immutable_snapshot_verified"] is False
    assert configuration["temperature"] == 0 and configuration["max_new_tokens"] == 128
    assert configuration["prompt_template"] == reference_task().project({"text": "{text}"}, None)
    assert configuration["scorer"]["implementation_digest"] == reference_task().job.scorer.implementation_digest
    assert configuration["configuration_digest"] == _document_digest({
        key: value for key, value in configuration.items() if key != "configuration_digest"})
    model = JsonFrontierModel(endpoint=OPENAI_CHAT_COMPLETIONS_URL, model=OPENAI_DEFAULT_MODEL, api_key="test-only")
    model.require_proof_configuration()
    monkeypatch.setattr(json_frontier_model, "frontier_proof_configuration", lambda: {**configuration, "scorer": {}})
    with pytest.raises(GymJobError, match="changed"):
        model.require_proof_configuration()


def test_pinned_frontier_refuses_prompt_drift_before_network(monkeypatch) -> None:
    model = JsonFrontierModel(endpoint=OPENAI_CHAT_COMPLETIONS_URL, model=OPENAI_DEFAULT_MODEL, api_key="test-only")
    model.require_proof_configuration()
    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: pytest.fail("must not contact provider"))
    interaction = reference_task().project({"text": "Order for Aster: 2 lamps."}, None)
    interaction["messages"][0]["content"] = "Use a different extraction prompt"
    with pytest.raises(GymJobError, match="prompt"):
        model.infer(interaction)


def test_luna_budget_refuses_request_before_network(monkeypatch) -> None:
    model = JsonFrontierModel(endpoint=OPENAI_CHAT_COMPLETIONS_URL, model=OPENAI_DEFAULT_MODEL, api_key="test-only")
    model.require_proof_configuration()
    interaction = reference_task().project({"text": "Order for Aster: 2 lamps."}, None)
    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: pytest.fail("must not contact provider"))
    model._api_attempts = 80
    with pytest.raises(GymJobError, match="ceiling"):
        model.infer(interaction)
    model._api_attempts = 0
    model._reserved_cost_microusd = 50000
    with pytest.raises(GymJobError, match="ceiling"):
        model.infer(interaction)