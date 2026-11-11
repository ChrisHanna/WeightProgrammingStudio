from __future__ import annotations

import json
import multiprocessing
import sqlite3
import urllib.request
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

import pytest
from neural_foundry.company_facts_public_plugin import company_task, compose, project
from neural_foundry.gym_job import GymJobError
from neural_foundry.gym_service import ExecutionResponse, _document_digest
from neural_foundry.openai_provider import OpenAIProvider
from neural_foundry.proof_policy import ACTIVE_POLICY, load_policy
from neural_foundry.proof_registry import builtin_registry
from neural_foundry.provenance import sha256_file
from neural_foundry.public_budget import PublicBudget, initialize_public_budget


@lru_cache(maxsize=1)
def _provider_references():
    return builtin_registry()[1]


@lru_cache(maxsize=1)
def _company_template():
    configuration = json.loads((Path(__file__).resolve().parents[1] / "protocols"
                                / "company-facts-public-fixture-v2.json").read_bytes())
    return compose(configuration)


def fixture_spec(root: Path, *, attempts: int = 3, cost: int = 100000):
    root.mkdir(parents=True, exist_ok=True)
    template, registry = _company_template()
    spec = deepcopy(template)
    document = json.loads(ACTIVE_POLICY.read_bytes())
    document["policy"]["api"]["maximum_api_attempts"] = attempts
    document["policy"]["api"]["maximum_cost_microusd"] = cost
    policy_path = root / "policy.json"
    policy_path.write_text(json.dumps(document))
    reference = {"path": str(policy_path), "file_digest": sha256_file(policy_path), "calls_authorized": True}
    spec["policy_reference"] = reference
    spec["policy"] = load_policy(reference)
    plugins = _provider_references()
    spec["candidates"] = [
        {"role": role, "method": "static-public-catalog", "evidence": plugins["api_evidence"],
         "execution": {"kind": "provider", "plugin": plugins["provider"], "configuration": {
             "endpoint": "https://provider.invalid/v1/chat/completions", "model": "gpt-6-luna"}}}
        for role in ("first", "second")]
    spec["public_budget"] = {"version": 1, "ledger_id": "test-run-ledger",
                             "path": str(root / "budget.sqlite3")}
    spec["evidence_class"] = "software_fixture"
    return spec, registry


def fixture_client(spec, role="first"):
    candidate = next(row for row in spec["candidates"] if row["role"] == role)
    return OpenAIProvider(spec=spec, candidate=candidate, task=company_task(spec["gym"]),
                          api_key="test-only-fixture-key")


class FakeResponse:
    def __init__(self, document):
        self.document = document

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def read(self, _limit):
        return json.dumps(self.document).encode()


class FakeOpener:
    def __init__(self, *, failure=None):
        self.calls = []
        self.failure = failure

    def open(self, request, timeout):
        self.calls.append(request)
        assert timeout == 60 and request.get_header("Authorization") == "Bearer test-only-fixture-key"
        payload = json.loads(request.data)
        assert payload["temperature"] == 0 and payload["reasoning_effort"] == "none"
        assert payload["max_completion_tokens"] == 128
        if self.failure == "network":
            raise OSError("fake dispatch failure")
        document = {"id": "mock-completion", "created": 1700000000, "system_fingerprint": "mock",
                    "model": payload["model"], "choices": [{"finish_reason": "stop",
                    "message": {"content": json.dumps({"value": "2042", "source": "synthetic-catalog:orchid"})}}]}
        if self.failure == "model":
            document["model"] = "wrong-model"
        elif self.failure == "length":
            document["choices"][0]["finish_reason"] = "length"
        return FakeResponse(document)


@pytest.fixture(autouse=True)
def forbid_unmocked_transport(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("real network transport is forbidden in offline conformance")

    monkeypatch.setattr(urllib.request, "build_opener", forbidden)


def _process_call(spec, role, queue):
    try:
        with patch("urllib.request.build_opener", return_value=FakeOpener()):
            model = fixture_client(spec, role)
            model.infer(project(spec["gym"]["rows"][0]["input"], None))
        queue.put("spent")
    except GymJobError:
        queue.put("refused")


def test_company_provider_is_gym_neutral_and_receipts_are_exact(tmp_path, monkeypatch) -> None:
    spec, _ = fixture_spec(tmp_path)
    initialize_public_budget(spec)
    opener = FakeOpener()
    monkeypatch.setattr(urllib.request, "build_opener", lambda *_: opener)
    model = fixture_client(spec)
    interaction = project(spec["gym"]["rows"][0]["input"], None)
    result = model.infer(interaction)
    model.verify_response_receipt(interaction, result)
    receipt = result.interaction["api_response_receipt"]
    payload = json.loads(opener.calls[0].data)
    assert payload["messages"] == interaction["messages"]
    assert payload["response_format"]["json_schema"]["schema"] == interaction["output_schema"]
    assert model.context["scorer"]["component_id"] == "company-supported-fact"
    assert receipt["request_digest"] == _document_digest(payload)
    assert receipt["receipt_digest"] == _document_digest(
        {key: value for key, value in receipt.items() if key != "receipt_digest"})
    assert receipt["gym_context_digest"] == _document_digest(model.context)
    assert "test-only-fixture-key" not in json.dumps(receipt)
    with pytest.raises(GymJobError, match="receipt"):
        model.verify_response_receipt(interaction, ExecutionResponse({}, result.interaction))
    altered = deepcopy(interaction)
    altered["messages"][0]["content"] = "different company question"
    with pytest.raises(GymJobError, match="receipt"):
        model.verify_response_receipt(altered, result)
    second = fixture_client(spec, "second")
    assert second.infer(interaction).interaction["api_response_receipt"]["api_attempts"] == 2
    assert PublicBudget(spec).snapshot()["attempts"] == 2


@pytest.mark.parametrize("failure", ["network", "model", "length"])
def test_failures_spend_durable_authorization_across_clients(tmp_path, monkeypatch, failure) -> None:
    spec, _ = fixture_spec(tmp_path, attempts=1)
    initialize_public_budget(spec)
    opener = FakeOpener(failure=failure)
    monkeypatch.setattr(urllib.request, "build_opener", lambda *_: opener)
    interaction = project(spec["gym"]["rows"][0]["input"], None)
    with pytest.raises(GymJobError, match="incomplete"):
        fixture_client(spec).infer(interaction)
    with pytest.raises(GymJobError, match="ceiling"):
        fixture_client(spec, "second").infer(interaction)
    assert len(opener.calls) == 1
    assert PublicBudget(spec).snapshot()["attempts"] == 1
    assert PublicBudget(spec).snapshot()["reserved_cost_microusd"] > 0


def test_cost_cap_refuses_before_mock_network_dispatch(tmp_path) -> None:
    spec, _ = fixture_spec(tmp_path, cost=1)
    initialize_public_budget(spec)
    with pytest.raises(GymJobError, match="ceiling"):
        fixture_client(spec).infer(project(spec["gym"]["rows"][0]["input"], None))
    assert PublicBudget(spec).snapshot()["attempts"] == 0


def test_reservation_is_committed_before_dispatch(tmp_path, monkeypatch) -> None:
    spec, _ = fixture_spec(tmp_path)
    initialize_public_budget(spec)
    opener = FakeOpener()
    original = opener.open

    def dispatched(request, timeout):
        state = PublicBudget(spec).snapshot()
        assert state["attempts"] == 1 and state["reserved_cost_microusd"] > 0
        return original(request, timeout)

    monkeypatch.setattr(opener, "open", dispatched)
    monkeypatch.setattr(urllib.request, "build_opener", lambda *_: opener)
    fixture_client(spec).infer(project(spec["gym"]["rows"][0]["input"], None))


def test_unauthorized_policy_refuses_before_budget_or_credential_use(tmp_path) -> None:
    spec, _ = fixture_spec(tmp_path)
    spec["policy_reference"]["calls_authorized"] = False
    spec["policy"] = load_policy(spec["policy_reference"])
    with pytest.raises(GymJobError, match="not authorized"):
        fixture_client(spec)


def test_unpriced_multimodal_message_refuses_before_reservation(tmp_path) -> None:
    spec, _ = fixture_spec(tmp_path)
    initialize_public_budget(spec)
    interaction = project(spec["gym"]["rows"][0]["input"], None)
    interaction["messages"][0]["content"] = [{"type": "image_url", "image_url": {"url": "https://example.invalid"}}]
    with pytest.raises(GymJobError, match="format/schema"):
        fixture_client(spec).infer(interaction)
    assert PublicBudget(spec).snapshot()["attempts"] == 0


@pytest.mark.parametrize("damage", ["missing", "missing-anchor", "corrupt", "state", "chain", "binding"])
def test_missing_or_corrupt_budget_never_reinitializes(tmp_path, monkeypatch, damage) -> None:
    spec, _ = fixture_spec(tmp_path)
    initialize_public_budget(spec)
    opener = FakeOpener()
    monkeypatch.setattr(urllib.request, "build_opener", lambda *_: opener)
    fixture_client(spec).infer(project(spec["gym"]["rows"][0]["input"], None))
    path = Path(spec["public_budget"]["path"])
    if damage == "missing":
        path.unlink()
    elif damage == "missing-anchor":
        path.with_name(path.name + ".anchor.json").unlink()
    elif damage == "corrupt":
        path.write_bytes(b"not sqlite")
    else:
        with sqlite3.connect(path) as connection:
            if damage == "state":
                connection.execute("UPDATE state SET payload='{}'")
            elif damage == "chain":
                connection.execute("DELETE FROM reservations")
            else:
                connection.execute("UPDATE metadata SET payload='{}'")
    with pytest.raises(GymJobError):
        fixture_client(spec, "second")
    with pytest.raises(GymJobError, match="initialization"):
        initialize_public_budget(spec)
    assert len(opener.calls) == 1


def test_budget_cannot_be_rebound_to_changed_run_or_policy(tmp_path) -> None:
    spec, _ = fixture_spec(tmp_path)
    initialize_public_budget(spec)
    changed = deepcopy(spec)
    changed["run_id"] = "different frozen run"
    with pytest.raises(GymJobError, match="frozen run/policy"):
        PublicBudget(changed)
    changed = deepcopy(spec)
    changed["policy"]["api"]["maximum_api_attempts"] += 1
    with pytest.raises(GymJobError, match="frozen run/policy"):
        PublicBudget(changed)


def test_frozen_document_form_and_unsigned_spec_share_the_same_budget(tmp_path, monkeypatch) -> None:
    spec, _ = fixture_spec(tmp_path)
    document = {**spec, "document_digest": _document_digest(spec)}
    initialize_public_budget(document)
    monkeypatch.setattr(urllib.request, "build_opener", lambda *_: FakeOpener())
    model = fixture_client(document)
    result = model.infer(project(spec["gym"]["rows"][0]["input"], None))
    assert model.configuration["run_spec_digest"] == _document_digest(spec)
    assert result.interaction["api_response_receipt"]["api_attempts"] == 1
    assert PublicBudget(spec).snapshot()["attempts"] == 1


def test_policy_file_and_inline_decoding_drift_are_refused(tmp_path) -> None:
    spec, _ = fixture_spec(tmp_path)
    initialize_public_budget(spec)
    model = fixture_client(spec)
    policy_path = Path(spec["policy_reference"]["path"])
    document = json.loads(policy_path.read_bytes())
    document["policy"]["max_new_tokens"] = 64
    policy_path.write_text(json.dumps(document))
    with pytest.raises(GymJobError, match="policy"):
        model.infer(project(spec["gym"]["rows"][0]["input"], None))
    assert PublicBudget(spec).snapshot()["attempts"] == 0


def test_concurrent_processes_share_one_atomic_attempt_ceiling(tmp_path) -> None:
    spec, _ = fixture_spec(tmp_path, attempts=3)
    initialize_public_budget(spec)
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    processes = [context.Process(target=_process_call, args=(spec, role, queue))
                 for role in ("first", "second") for _ in range(3)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(30)
        assert process.exitcode == 0
    results = [queue.get(timeout=5) for _ in processes]
    assert results.count("spent") == 3 and results.count("refused") == 3
    assert PublicBudget(spec).snapshot()["attempts"] == 3


def test_concurrent_processes_share_one_atomic_cost_ceiling(tmp_path) -> None:
    spec, _ = fixture_spec(tmp_path, attempts=20, cost=600)
    initialize_public_budget(spec)
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    processes = [context.Process(target=_process_call, args=(spec, role, queue))
                 for role in ("first", "second") for _ in range(3)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(30)
        assert process.exitcode == 0
    results = [queue.get(timeout=5) for _ in processes]
    state = PublicBudget(spec).snapshot()
    unit_cost = state["reserved_cost_microusd"] // state["attempts"]
    assert results.count("spent") == state["attempts"] == 600 // unit_cost
    assert state["attempts"] < 6 and state["reserved_cost_microusd"] <= 600


def test_restart_in_another_process_cannot_reset_spent_budget(tmp_path) -> None:
    spec, _ = fixture_spec(tmp_path, attempts=1)
    initialize_public_budget(spec)
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    results = []
    for role in ("first", "second"):
        process = context.Process(target=_process_call, args=(spec, role, queue))
        process.start()
        process.join(30)
        assert process.exitcode == 0
        results.append(queue.get(timeout=5))
    assert results == ["spent", "refused"]


@pytest.mark.parametrize("filename", ["openai_transport.py", "records.py", "public-proof-policy-v1.json"])
def test_declared_transitive_dependency_tampering_is_refused(monkeypatch, filename) -> None:
    from neural_foundry import proof_registry

    registry, references = builtin_registry()
    reference = references["provider"]
    assert reference["version"] == "2"
    assert "python/neural_foundry/openai_transport.py" in reference["dependencies"]
    assert "python/neural_foundry/records.py" in reference["dependencies"]
    original = proof_registry.sha256_file

    def tampered(path):
        if Path(path).name == filename:
            return "sha256:" + "0" * 64
        return original(path)

    monkeypatch.setattr(proof_registry, "sha256_file", tampered)
    with pytest.raises(GymJobError, match="dependency changed"):
        registry.resolve("provider", reference)
    changed = deepcopy(reference)
    changed["dependencies"].pop("python/neural_foundry/records.py")
    with pytest.raises(GymJobError, match="dependencies changed"):
        registry.resolve("provider", changed)


@pytest.mark.parametrize("change", ["cap", "count", "tolerance", "price"])
def test_policy_values_cannot_be_weakened_even_with_updated_file_digest(tmp_path, change) -> None:
    document = json.loads(ACTIVE_POLICY.read_bytes())
    policy = document["policy"]
    if change == "cap":
        policy["max_new_tokens"] = 256
    elif change == "count":
        policy["identity"]["probe_count"] = 1
    elif change == "tolerance":
        policy["identity"]["logprob_atol"] = 0.1
    else:
        policy["api"]["model_rates"]["gpt-6-luna"]["output_microusd_per_million_tokens"] = 0
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(document))
    with pytest.raises(GymJobError):
        load_policy({"path": str(path), "file_digest": sha256_file(path), "calls_authorized": False})


def test_dependency_reference_mutation_cannot_mutate_registry_expectations() -> None:
    registry, references = builtin_registry()
    reference = references["provider"]
    reference["dependencies"].pop("python/neural_foundry/records.py")
    with pytest.raises(GymJobError, match="dependencies changed"):
        registry.resolve("provider", reference)
