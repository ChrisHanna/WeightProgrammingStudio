"""Owned immutable assets, real CPU previews and recorded-output worker integration."""

from __future__ import annotations

import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import test_control_plane_postgres
from neural_foundry.control_plane.models import ControlPlaneError, compile_plan, validate_spec
from neural_foundry.control_plane.product import score_json, validate_dataset, validate_gym
from neural_foundry.control_plane.service import ControlPlaneService
from neural_foundry.control_plane.store import Store

database = test_control_plane_postgres.database


@pytest.fixture
def service(database):
    root = Path("artifacts") / "product-test-objects" / uuid.uuid4().hex
    Store(database).migrate()
    try:
        yield ControlPlaneService(database, root)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def project(service, key="project"):
    return service.create_project({"name": "Team", "description": "Owned data", "idempotency_key": key}, actor="operator")


def asset(service, pid, kind, name, document, parent=None, key=None):
    return service.create_asset(pid, {"kind": kind, "name": name, "document": document,
                                     "parent_asset_id": parent, "idempotency_key": key or uuid.uuid4().hex}, actor="operator")


def dataset_document():
    return {"format": "json_examples_v1", "examples": [
        {"id": "train", "split": "train", "input": "training", "expected": {"value": 0}},
        {"id": "a", "split": "validation", "input": "validation one", "expected": {"value": 1}},
        {"id": "b", "split": "validation", "input": "validation two", "expected": {"value": 2}},
    ]}


def setup_assets(service):
    pid = project(service)["id"]
    dataset = asset(service, pid, "dataset", "Dataset", dataset_document())
    gym = asset(service, pid, "gym", "Gym", {"template": "json_exact", "config": {}, "dataset_asset_id": dataset["id"]})
    models, evidence = [], []
    for i, responses in enumerate((['{"value":1}', '{"value":2}'], ['not JSON', '{"value":9}'])):
        model = asset(service, pid, "model", f"Model {i}", {"model_id": f"team/model-{i}", "revision": "r1", "source": "recorded"})
        models.append(model)
        evidence.append(asset(service, pid, "evidence", f"Recorded {i}", {
            "model_asset_id": model["id"], "dataset_asset_id": dataset["id"],
            "source_description": "User supplied offline capture",
            "rows": [{"example_id": eid, "response": response} for eid, response in zip(("a", "b"), responses)]}))
    return pid, dataset, gym, models, evidence


@pytest.mark.parametrize("response,valid,exact", [
    ('{"value":1}', True, True), ('{"value": 2}', True, False), ("broken", False, False),
    ("[]", False, False), ('{"value":NaN}', False, False), ('{"value":true}', True, False),
])
def test_recorded_json_quality_is_not_transport_failure(response, valid, exact):
    assert score_json({"value": 1}, response) == {"valid_json": valid, "exact_match": exact}


def test_dataset_normalized_overlap_duplicate_and_bounds():
    value = dataset_document()
    value["examples"][1]["input"] = "  training \n"
    with pytest.raises(ControlPlaneError, match="overlap"):
        validate_dataset(value)
    value = dataset_document()
    value["examples"][1]["id"] = "train"
    with pytest.raises(ControlPlaneError):
        validate_dataset(value)
    value["examples"] = value["examples"] * 400
    with pytest.raises(ControlPlaneError):
        validate_dataset(value)
    with pytest.raises(ControlPlaneError):
        validate_gym({"template": "stateful_counter", "config": {"target": 33}})
    with pytest.raises(ControlPlaneError):
        validate_gym({"template": "tool_workflow", "config": {"module": "os"}})


def test_empty_workspace_and_project_idempotency(service):
    assert service.workspace()["projects"] == []
    assert service.workspace()["capabilities"]["fine_tune"]["enabled"] is False
    first = project(service)
    assert first == project(service)
    with pytest.raises(ControlPlaneError) as err:
        service.create_project({"name": "Different", "description": "", "idempotency_key": "project"}, actor="operator")
    assert err.value.code == "idempotency_conflict"
    assert service.get_project(first["id"]) == {"project": first, "assets": [], "runs": []}


def test_asset_versions_ownership_concurrency_and_rollback(service):
    pid = project(service)["id"]
    doc = {"model_id": "declared/model", "revision": "r1", "source": "reference"}
    first = asset(service, pid, "model", "Model", doc, key="same")
    assert first == asset(service, pid, "model", "Model", doc, key="same")
    assert first["version"] == 1
    second_doc = {**doc, "revision": "r2"}

    def version(key):
        try:
            return asset(service, pid, "model", "Model", second_doc, first["id"], key=key)
        except ControlPlaneError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(version, ["two", "three"]))
    assert sum(isinstance(r, dict) for r in results) == 1
    assert "version_conflict" in results
    assets = service.get_project(pid)["assets"]
    assert [a["version"] for a in assets] == [1, 2]
    assert assets[0]["document"]["revision"] == "r1"
    other = project(service, "other")["id"]
    with pytest.raises(ControlPlaneError) as exc:
        asset(service, other, "model", "Model", doc, first["id"])
    assert exc.value.code == "asset_not_found"
    with pytest.raises(ControlPlaneError):
        asset(service, pid, "dataset", "Invalid", {"format": "json_examples_v1", "examples": []})
    assert len(service.get_project(pid)["assets"]) == 2
    with pytest.raises(ControlPlaneError), service.store.transaction() as conn:
        conn.execute("UPDATE cp_assets SET version=99 WHERE id=%s", (first["id"],))
    assert len(service.get_project(pid)["assets"]) == 2


def test_preview_actual_fixture_lifecycle_and_failures(service, monkeypatch):
    from neural_foundry.gym_runtime_fixtures import CounterSession
    pid = project(service)["id"]
    closed = []
    original = CounterSession.close

    def close(session):
        closed.append(True)
        original(session)

    monkeypatch.setattr(CounterSession, "close", close)
    definition = {"template": "stateful_counter", "config": {"target": 2, "horizon": 3}}
    body = {"definition": definition, "actions": [1, 1], "seed": 0}
    result = service.preview_gym(pid, body)
    assert result["observation"] == {"position": 2}
    assert result["terminated"] and result["closed"] and not result["model_calls"]
    assert result["allowed_actions"] == []
    with pytest.raises(ControlPlaneError) as err:
        service.preview_gym(pid, {**body, "actions": [99]})
    assert err.value.code == "preview_step_failed"
    assert len(closed) == 2
    truncated = service.preview_gym(pid, {**body, "definition": {"template": "stateful_counter", "config": {"horizon": 1}}, "actions": [-1]})
    assert truncated["truncated"] and truncated["allowed_actions"] == []
    workflow = service.preview_gym(pid, {"definition": {"template": "tool_workflow", "config": {}},
                                       "actions": ["lookup", "finish"], "seed": 0})
    assert workflow["terminated"] and workflow["observation"] == "done" and workflow["closed"]
    with pytest.raises(ControlPlaneError):
        service.preview_gym(pid, {**body, "actions": [1, 1, 1]})


def test_json_preview_and_evidence_validation(service):
    pid, _dataset, gym, _models, evidence = setup_assets(service)
    preview = service.preview_gym(pid, {"definition": gym["document"], "actions": [], "seed": 0,
                                       "example_id": "a", "response": "bad"})
    assert preview["observation"]["expected"] == {"value": 1}
    assert preview["observation"]["valid_json"] is False
    for rows in ([{"example_id": "train", "response": "{}"}],
                 [{"example_id": "a", "response": "{}"}],
                 [{"example_id": "a", "response": "{}"}] * 2):
        with pytest.raises(ControlPlaneError):
            asset(service, pid, "evidence", "Invalid", {**evidence[0]["document"], "rows": rows})
    other = project(service, "other")["id"]
    with pytest.raises(ControlPlaneError):
        asset(service, other, "gym", "Foreign", gym["document"])


def test_saved_output_runs_are_durable_approved_and_comparable(service):
    pid, _dataset, gym, _models, evidence = setup_assets(service)
    body = {"workstream": "evaluate_saved", "name": "Compare recorded models", "gym_asset_id": gym["id"],
            "evidence_asset_ids": [e["id"] for e in evidence], "idempotency_key": "evaluation"}
    run = service.create_experiment(pid, body, actor="operator")
    assert run["state"] == "created"
    assert service.create_experiment(pid, body, actor="operator")["id"] == run["id"]
    assert all(not s["attempts"] and s["result"] is None for s in run["stages"])
    assert service.worker_tick("offline-worker") is None
    plan = service.plan_run(run["id"], actor="operator")
    assert plan["plan"]["model_calls"] is False
    approved = service.approve_run(run["id"], actor="operator", plan_digest=plan["plan"]["digest"], limits={})
    assert service.worker_tick("offline-worker") is None
    service.start_run(run["id"], actor="operator", approval_id=approved["id"], idempotency_key="start")
    assert service.worker_tick("offline-worker")["accepted"]
    assert service.worker_tick("offline-worker")["accepted"]
    assert service.worker_tick("offline-worker") is None
    finished = service.get_run(run["id"])
    assert finished["state"] == "completed" and finished["outcome"]["quality"] == "failed"
    results = [s["result"] for s in finished["stages"]]
    assert [r["metrics"][0]["value"] for r in results] == [1, 0]
    assert [r["metrics"][1]["value"] for r in results] == [1, .5]
    assert all(r["analysis"] == "recorded-output-analysis" and not r["model_calls"] for r in results)
    assert results[1]["rows"][0]["response"] == "not JSON"
    assert results[0]["pins"]["gym_artifact_id"] == results[1]["pins"]["gym_artifact_id"]
    assert results[0]["candidate_model"] != results[1]["candidate_model"]
    assert len(service.get_project(pid)["runs"]) == 1
    unlinked = service.create_run(run["spec"], actor="operator", idempotency_key="not-linked")
    assert len(service.get_project(pid)["runs"]) == 1 and unlinked["id"] != run["id"]
    bundle = service.project_bundle(pid)
    assert not bundle["release_authorized"] and not bundle["signed"]
    assert all(a["document"] is None for a in bundle["artifacts"])


def test_live_plans_save_but_execution_refuses_default(service):
    pid, dataset, gym, models, _evidence = setup_assets(service)
    for method, stream in (("lora", "fine_tune"), ("weight_edit", "weight_edit"), ("merge", "merge")):
        recipe = asset(service, pid, "recipe", method, {
            "method": method, "model_asset_id": models[0]["id"], "gym_asset_ids": [gym["id"]],
            "dataset_asset_id": dataset["id"], "parameters": {},
            **({"source_model_asset_ids": [m["id"] for m in models]} if method == "merge" else {})})
        with pytest.raises(ControlPlaneError) as err:
            service.create_experiment(pid, {"workstream": stream, "name": "Plan",
                "recipe_asset_id": recipe["id"], "idempotency_key": method}, actor="operator")
        assert err.value.code == "runtime_not_qualified" and err.value.status == 409
    assert service.get_project(pid)["runs"] == []
    with pytest.raises(ControlPlaneError):
        asset(service, pid, "recipe", "Unsafe", {"method": "sft", "model_asset_id": models[0]["id"],
              "gym_asset_ids": [gym["id"]], "parameters": {"code": "import os"}})


def test_analysis_pins_and_limits_cannot_be_forged(service):
    pid, _dataset, gym, _models, evidence = setup_assets(service)
    run = service.create_experiment(pid, {"workstream": "evaluate_saved", "name": "Pinned",
        "gym_asset_id": gym["id"], "evidence_asset_ids": [evidence[0]["id"]], "idempotency_key": "pinned"}, actor="operator")
    spec = run["spec"]
    spec["stages"][0]["limits"]["steps"] = 1
    with pytest.raises(ControlPlaneError) as err:
        compile_plan(spec)
    assert err.value.code == "budget_exceeded"
    spec["stages"][0]["config"]["metric_policy"] = "user-defined"
    with pytest.raises(ControlPlaneError):
        validate_spec(spec, service._operations())


def test_authenticated_product_http_contract(service, monkeypatch):
    from fastapi.testclient import TestClient
    from neural_foundry.control_plane.api import create_app
    monkeypatch.delenv("CP_OPERATOR_TOKEN_FILE", raising=False)
    monkeypatch.setenv("CP_OPERATOR_TOKEN", "product-test-token")
    monkeypatch.setenv("CP_ALLOWED_HOSTS", "testserver")
    auth = {"Authorization": "Bearer product-test-token"}
    with TestClient(create_app(service)) as client:
        for method, path, body in [
            ("GET", "/v1/workspace", None), ("POST", "/v1/projects", {}),
            ("GET", "/v1/projects/p", None), ("POST", "/v1/projects/p/assets", {}),
            ("POST", "/v1/projects/p/preview", {}), ("POST", "/v1/projects/p/experiments", {}),
            ("GET", "/v1/projects/p/bundle", None),
        ]:
            assert client.request(method, path, json=body).status_code == 401
        workspace = client.get("/v1/workspace", headers=auth).json()
        assert workspace["projects"] == []
        assert {r["id"] for r in workspace["recipe_methods"]} == {"lora", "sft", "weight_edit", "merge"}
        assert all(r["parameters_schema"]["additionalProperties"] is False for r in workspace["recipe_methods"])
        response = client.post("/v1/projects", headers=auth,
            json={"name": "HTTP team", "description": "", "idempotency_key": "http-project"})
        assert response.status_code == 200
        pid = response.json()["id"]
        response = client.post(f"/v1/projects/{pid}/assets", headers=auth, json={
            "kind": "gym", "name": "Counter", "document": {"template": "stateful_counter", "config": {}},
            "parent_asset_id": None, "idempotency_key": "counter"})
        assert response.status_code == 200 and response.json()["document"]["template"] == "stateful_counter"
        preview = client.post(f"/v1/projects/{pid}/preview", headers=auth,
            json={"definition": response.json()["document"], "actions": [1], "seed": 0})
        assert preview.status_code == 200 and preview.json()["closed"]
        assert client.get(f"/v1/projects/{pid}", headers=auth).json()["runs"] == []
        assert client.get(f"/v1/projects/{pid}/bundle", headers=auth).json()["release_authorized"] is False
        assert client.get("/operations").status_code == 200
        assert client.get("/").status_code == 200


def test_asset_metadata_and_run_links_rollback_with_transaction(service, monkeypatch):
    pid = project(service)["id"]
    original = service.artifacts.put

    def broken_put(value):
        original(value)
        raise RuntimeError("interrupted publication")

    monkeypatch.setattr(service.artifacts, "put", broken_put)
    with pytest.raises(RuntimeError):
        asset(service, pid, "dataset", "Rollback", dataset_document())
    with service.store.transaction() as conn:
        assert conn.execute("SELECT count(*) AS n FROM cp_artifacts").fetchone()["n"] == 0
    assert service.get_project(pid)["assets"] == []
    monkeypatch.setattr(service.artifacts, "put", original)
    pid, _dataset, gym, _models, evidence = setup_assets(service)
    original_insert = service._insert_run

    def interrupted_insert(*args, **kwargs):
        original_insert(*args, **kwargs)
        raise RuntimeError("interrupted linking")

    monkeypatch.setattr(service, "_insert_run", interrupted_insert)
    with pytest.raises(RuntimeError):
        service.create_experiment(pid, {"workstream": "evaluate_saved", "name": "Rollback",
            "gym_asset_id": gym["id"], "evidence_asset_ids": [evidence[0]["id"]],
            "idempotency_key": "rollback"}, actor="operator")
    with service.store.transaction() as conn:
        assert conn.execute("SELECT count(*) AS n FROM cp_runs").fetchone()["n"] == 0
        assert conn.execute("SELECT count(*) AS n FROM cp_project_runs").fetchone()["n"] == 0


def test_analysis_worker_refuses_forged_version_and_model_candidate(service):
    pid, _dataset, gym, _models, evidence = setup_assets(service)
    original = service.create_experiment(pid, {"workstream": "evaluate_saved", "name": "Pins",
        "gym_asset_id": gym["id"], "evidence_asset_ids": [evidence[0]["id"]], "idempotency_key": "pins"}, actor="operator")
    for suffix in ("version", "model"):
        from copy import deepcopy
        spec = deepcopy(original["spec"])
        if suffix == "version":
            spec["stages"][0]["config"]["dataset_version"] = 99
        else:
            spec["candidates"][0]["model"]["revision"] = "invented"
        run = service.create_run(spec, actor="operator", idempotency_key="forged-" + suffix)
        planned = service.plan_run(run["id"], actor="operator")
        approval = service.approve_run(run["id"], actor="operator", plan_digest=planned["plan"]["digest"], limits={})
        service.start_run(run["id"], actor="operator", approval_id=approval["id"], idempotency_key="start")
        assert service.worker_tick("pin-worker")["state"] == "failed"
        final = service.get_run(run["id"])
        assert final["stages"][0]["attempts"][0]["error"]["code"] == "invalid_analysis_pins"
        assert final["stages"][0]["result"] is None


def test_schema_two_upgrade_preserves_runs(database):
    from neural_foundry.control_plane.catalog import catalog_document
    from neural_foundry.control_plane.store import MIGRATION, MIGRATION_2
    store = Store(database)
    with store.transaction() as conn:
        conn.execute(MIGRATION)
        conn.execute("INSERT INTO cp_schema(version) VALUES (1)")
        conn.execute(MIGRATION_2)
        conn.execute("INSERT INTO cp_schema(version) VALUES (2)")
    root = Path("artifacts") / "product-test-objects" / uuid.uuid4().hex
    try:
        facade = ControlPlaneService(database, root)
        before = facade.create_run(catalog_document()["templates"][0]["spec"], actor="operator", idempotency_key="old")
        store.migrate()
        assert store.health()["schema_version"] == 3
        assert facade.get_run(before["id"]) == before
        assert facade.workspace()["projects"] == []
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_qualified_proof_plan_uses_trusted_bindings_and_cpu_callbacks(service):
    from copy import deepcopy
    from dataclasses import replace

    from neural_foundry.control_plane.bridges import QualifiedGymBinding, TrustedRuntimeProfile
    from neural_foundry.control_plane.models import digest
    from neural_foundry.gym_lifecycle import AdapterIdentity
    from neural_foundry.gym_service import _document_digest
    from neural_foundry.json_public_run import public_policy
    from neural_foundry.proof_pipeline import validate_spec as validate_proof_spec
    from neural_foundry.proof_registry import ProofRegistry
    pid, dataset, gym, models, _evidence = setup_assets(service)
    registry = ProofRegistry()

    def forbidden(*args, **kwargs):
        pytest.fail("Creating a plan must not execute a proof plugin")

    references = {kind: registry.register(kind, AdapterIdentity(
        "qualified-test-" + kind, "1", _document_digest({"kind": kind})), forbidden)
        for kind in ("gym", "model", "trainer", "training_audit", "exporter",
                     "augmenter", "public_audit", "provider", "evidence")}
    training = [{"scenario_id": "train:train", "input": "training", "reference": {"value": 0}}]
    validation = [{"scenario_id": "validation:" + r["id"], "input": r["input"], "reference": r["expected"]}
                  for r in dataset["document"]["examples"] if r["split"] == "validation"]
    proof = {
        "kind": "nf_public_proof_run", "version": 1, "run_id": "proof-plan",
        "download_allowed": False, "holdout_accessed": False,
        "base_model": {"model_id": models[0]["document"]["model_id"], "revision": "r1"},
        "policy": public_policy(), "gym": {"task": "qualified-cpu-json", "version": 1},
        "plugins": {kind: references[kind] for kind in
                    ("gym", "model", "trainer", "training_audit", "exporter", "augmenter", "public_audit")},
        "training_rows": training, "training_rows_digest": _document_digest({"rows": training}),
        "candidates": [{"role": "candidate", "method": "lora",
                        "recipe": {"method": "lora", "steps": 1, "seed": 0, "learning_rate": .01},
                        "execution": {"kind": "provider", "plugin": references["provider"]},
                        "evidence": references["evidence"]}],
        "public_sets": [{"name": "validation", "rows": validation,
                         "rows_digest": _document_digest({"rows": validation})}],
    }
    validate_proof_spec(proof, registry)
    callbacks = []

    def cpu_callback(context):
        callbacks.append(context["document"]["product_binding"])
        return {"steps": 1, "model_calls": False, "quality": "not_evaluated", "cpu_callback": True}

    profile = TrustedRuntimeProfile(
        "qualified-test", service.artifacts.root / "qualified", qualified=True, registry=registry,
        deployment_capabilities=frozenset({"model_training", "model_export", "model_identity", "model_scoring"}),
        callbacks={op: cpu_callback for op in ("proof.train", "proof.export", "proof.identity", "proof.score")})
    service.install_runtime_profile(profile)
    assert not service.workspace()["capabilities"]["fine_tune"]["enabled"]
    proof_id = service.import_artifact(proof, actor="operator")["id"]
    recipe = asset(service, pid, "recipe", "Qualified", {
        "method": "lora", "model_asset_id": models[0]["id"], "gym_asset_ids": [gym["id"]],
        "dataset_asset_id": dataset["id"], "parameters": {"learning_rate": .01},
        "runtime_profile": "qualified-test", "role": "candidate", "proof_spec_artifact_id": proof_id})
    experiment = {"workstream": "fine_tune", "name": "Qualified plan",
                  "recipe_asset_id": recipe["id"], "idempotency_key": "qualified"}
    with pytest.raises(ControlPlaneError) as err:
        service.create_experiment(pid, experiment, actor="operator")
    assert err.value.code == "runtime_not_qualified"
    binding = QualifiedGymBinding(
        project_id=pid, gym_asset_id=gym["id"], gym_version=gym["version"], gym_artifact_id=gym["artifact_id"],
        proof_gym_reference_digest=digest(references["gym"]),
        proof_gym_configuration_digest=digest(proof["gym"]), methods=frozenset({"lora"}))
    service.install_runtime_profile(replace(profile, product_gym_bindings=(binding,)))
    assert service.workspace()["capabilities"]["fine_tune"]["enabled"]
    changed_type = deepcopy(proof)
    changed_type["training_rows"][0]["reference"]["value"] = False
    changed_type["training_rows_digest"] = _document_digest({"rows": changed_type["training_rows"]})
    changed_id = service.import_artifact(changed_type, actor="operator")["id"]
    changed_recipe = asset(service, pid, "recipe", "Changed JSON type", {
        **recipe["document"], "proof_spec_artifact_id": changed_id})
    with pytest.raises(ControlPlaneError) as err:
        service.create_experiment(pid, {"workstream": "fine_tune", "name": "Changed JSON type",
            "recipe_asset_id": changed_recipe["id"], "idempotency_key": "type-mismatch"}, actor="operator")
    assert err.value.code == "runtime_not_qualified"
    run = service.create_experiment(pid, experiment, actor="operator")
    assert run["state"] == "created"
    assert [s["operation"] for s in run["spec"]["stages"]] == ["proof.train", "proof.export", "proof.identity", "proof.score"]
    assert all(not s["attempts"] for s in run["stages"])
    bound_proof = service.get_artifact(run["spec"]["stages"][0]["config"]["artifact_id"])["document"]
    assert bound_proof["product_binding"]["assets"][0]["id"] == recipe["id"]
    assert bound_proof["product_binding"]["original_proof_artifact_id"] == proof_id
    assert service.worker_tick("idle-worker") is None and callbacks == []
    planned = service.plan_run(run["id"], actor="operator")
    approval = service.approve_run(run["id"], actor="operator", plan_digest=planned["plan"]["digest"], limits={})
    assert service.worker_tick("idle-worker") is None and callbacks == []
    service.start_run(run["id"], actor="operator", approval_id=approval["id"], idempotency_key="cpu-start")
    for _ in range(4):
        assert service.worker_tick("cpu-worker")["accepted"]
    completed = service.get_run(run["id"])
    assert completed["state"] == "completed" and completed["outcome"]["model_calls"] is False
    assert len(callbacks) == 4 and all(s["result"]["cpu_callback"] for s in completed["stages"])
    bad_recipe = asset(service, pid, "recipe", "Mismatched", {**recipe["document"], "parameters": {"learning_rate": .02}})
    with pytest.raises(ControlPlaneError) as err:
        service.create_experiment(pid, {"workstream": "fine_tune", "name": "Mismatch",
            "recipe_asset_id": bad_recipe["id"], "idempotency_key": "mismatch"}, actor="operator")
    assert err.value.code == "runtime_not_qualified"
    legacy = {**proof, "gym": {"rows_per_template": 1, "methods": ["lora"]}}
    validate_proof_spec(legacy, registry)
    legacy_id = service.import_artifact(legacy, actor="operator")["id"]
    legacy_recipe = asset(service, pid, "recipe", "Legacy mismatch", {
        **recipe["document"], "proof_spec_artifact_id": legacy_id})
    with pytest.raises(ControlPlaneError) as err:
        service.create_experiment(pid, {"workstream": "fine_tune", "name": "Legacy mismatch",
            "recipe_asset_id": legacy_recipe["id"], "idempotency_key": "legacy-mismatch"}, actor="operator")
    assert err.value.code == "runtime_not_qualified"
    service.install_runtime_profile(replace(profile, product_gym_bindings=(
        replace(binding, gym_version=binding.gym_version + 1),)))
    with pytest.raises(ControlPlaneError) as err:
        service.create_experiment(pid, {**experiment, "idempotency_key": "wrong-version"}, actor="operator")
    assert err.value.code == "runtime_not_qualified"
    assert len(callbacks) == 4
    assert len(service.get_project(pid)["runs"]) == 1


def test_merge_recipe_records_owned_composition_and_rejects_bad_sources(service):
    pid, dataset, gym, models, _evidence = setup_assets(service)
    body = {"method": "merge", "model_asset_id": models[0]["id"], "gym_asset_ids": [gym["id"]],
            "dataset_asset_id": dataset["id"], "parameters": {"weight": .5}}
    sources = [m["id"] for m in models]
    saved = asset(service, pid, "recipe", "Composition", {**body, "source_model_asset_ids": sources})
    assert saved["document"]["source_model_asset_ids"] == sources
    hint = next(m for m in service.workspace()["recipe_methods"] if m["id"] == "merge")["source_model_asset_ids_schema"]
    assert hint["minItems"] == 2 and hint["uniqueItems"] and hint["include_model_asset_id"]
    for invalid in (None, [], sources[:1], sources * 2, ["other-a", "other-b"]):
        with pytest.raises(ControlPlaneError):
            asset(service, pid, "recipe", "Invalid merge", {**body, "source_model_asset_ids": invalid})
    other = project(service, "foreign-model-project")["id"]
    foreign = asset(service, other, "model", "Foreign", models[1]["document"])
    for invalid in ([sources[0], foreign["id"]], [sources[0], dataset["id"]]):
        with pytest.raises(ControlPlaneError) as err:
            asset(service, pid, "recipe", "Foreign merge", {**body, "source_model_asset_ids": invalid})
        assert err.value.code == "asset_not_found"


def test_project_runs_are_newest_first_not_random_identifier_order(service):
    from neural_foundry.control_plane.catalog import catalog_document
    pid = project(service)["id"]
    older, newer = "run-zz-old", "run-aa-new"
    with service.store.transaction() as conn:
        for rid, when in ((older, "2026-01-01"), (newer, "2026-02-01")):
            conn.execute("INSERT INTO cp_runs(id,actor,create_key,state,spec,spec_digest,created_at) "
                         "VALUES (%s,'operator',%s,'created',%s,'test',%s)",
                         (rid, rid, test_control_plane_postgres.json_value(catalog_document()["templates"][0]["spec"]), when))
            conn.execute("INSERT INTO cp_project_runs(project_id,run_id) VALUES (%s,%s)", (pid, rid))
    assert [r["id"] for r in service.get_project(pid)["runs"]] == [newer, older]
