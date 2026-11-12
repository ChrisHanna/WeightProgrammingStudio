"""Real PostgreSQL integration; CP_TEST_DATABASE_URL must name a disposable DB."""

from __future__ import annotations

import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from neural_foundry.control_plane.catalog import catalog_document
from neural_foundry.control_plane.models import ControlPlaneError
from neural_foundry.control_plane.service import ControlPlaneService
from neural_foundry.control_plane.store import Store, json_value


def connect_database(url, **kwargs):
    import psycopg
    try:
        return psycopg.connect(url, autocommit=True, connect_timeout=5, **kwargs)
    except psycopg.Error:
        pass
    pytest.fail("Isolated PostgreSQL connection failed (details suppressed).", pytrace=False)


@pytest.fixture
def database():
    url = os.environ.get("CP_TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set CP_TEST_DATABASE_URL for real PostgreSQL integration")
    from psycopg.conninfo import make_conninfo
    schema = "cp_test_" + uuid.uuid4().hex
    with connect_database(url) as conn:
        conn.execute(f'CREATE SCHEMA "{schema}"')
    dsn = make_conninfo(url, options=f"-c search_path={schema}")
    try:
        yield dsn
    finally:
        with connect_database(url) as conn:
            conn.execute(f'DROP SCHEMA "{schema}" CASCADE')


@pytest.fixture
def service(database, tmp_path):
    Store(database).migrate()
    return ControlPlaneService(database, tmp_path / "objects")


def create(service, template=0, key=None):
    return service.create_run(catalog_document()["templates"][template]["spec"],
                              actor="operator", idempotency_key=key or uuid.uuid4().hex)


def start(service, run):
    run = service.plan_run(run["id"], actor="operator")
    approval = service.approve_run(run["id"], actor="operator", plan_digest=run["plan"]["digest"], limits={})
    return service.start_run(run["id"], actor="operator", approval_id=approval["id"], idempotency_key="start")


def drain(service, maximum=20):
    for _ in range(maximum):
        if service.worker_tick("fixture-worker") is None:
            return
    pytest.fail("Worker failed to become idle")


def test_migration_transaction_crash_restart_and_concurrent_idempotence(database):
    store = Store(database)
    # PostgreSQL DDL is transactional: a failed migration leaves no partial schema.
    with pytest.raises(RuntimeError), store.transaction() as conn:
        from neural_foundry.control_plane.store import MIGRATION
        conn.execute(MIGRATION)
        raise RuntimeError("interrupted")
    with store.transaction() as conn:
        assert conn.execute("SELECT to_regclass('cp_runs') AS name").fetchone()["name"] is None
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: store.migrate(), range(2)))
    store.migrate()
    assert store.health()["schema_version"] == 3


def test_existing_schema_one_is_upgraded_transactionally(database):
    from neural_foundry.control_plane.store import MIGRATION
    store = Store(database)
    with store.transaction() as conn:
        conn.execute(MIGRATION)
        conn.execute("INSERT INTO cp_schema(version) VALUES (1)")
    store.migrate()
    assert store.health()["schema_version"] == 3


@pytest.mark.parametrize("template", range(6))
def test_every_registered_fixture_template_executes_real_operations(service, template):
    run = start(service, create(service, template))
    drain(service)
    run = service.get_run(run["id"])
    assert run["state"] == "completed", run
    assert run["outcome"]["release_authorized"] is False
    assert all(stage["attempts"][0]["state"] == "completed" for stage in run["stages"])
    for stage in run["stages"]:
        artifact = service.get_artifact(stage["artifact_id"])
        assert artifact["document"] == stage["result"]
    events = service.get_events(run["id"])
    assert events[0]["type"] == "run.created" and events[-1]["type"] == "run.completed"
    assert service.get_events(run["id"], after=events[-1]["sequence"]) == []


def test_create_start_idempotency_approvals_and_no_execution_before_start(service):
    run = create(service, key="same")
    assert create(service, key="same")["id"] == run["id"]
    changed = catalog_document()["templates"][0]["spec"]
    changed["name"] = "other"
    with pytest.raises(ControlPlaneError) as error:
        service.create_run(changed, actor="operator", idempotency_key="same")
    assert error.value.code == "idempotency_conflict"
    assert service.worker_tick("worker") is None
    planned = service.plan_run(run["id"], actor="operator")
    with pytest.raises(ControlPlaneError):
        service.approve_run(run["id"], actor="operator", plan_digest="wrong", limits={})
    approval = service.approve_run(run["id"], actor="operator", plan_digest=planned["plan"]["digest"], limits={})
    assert approval["limits"] == planned["plan"]["limits"]
    assert service.worker_tick("worker") is None
    first = service.start_run(run["id"], actor="operator", approval_id=approval["id"], idempotency_key="s")
    assert service.start_run(run["id"], actor="operator", approval_id=approval["id"], idempotency_key="s") == first
    with pytest.raises(ControlPlaneError):
        service.start_run(run["id"], actor="operator", approval_id=approval["id"], idempotency_key="other")


def test_transactional_claims_ownership_cancel_and_lease_uncertainty(service):
    run = start(service, create(service))
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(service._claim, ["one", "two"]))
    claim = next(c for c in claims if c)
    assert sum(c is not None for c in claims) == 1
    wrong = {**claim, "worker_id": "other-owner"}
    assert service.renew_lease(wrong) is False
    assert service._finish(wrong, {"steps": 1}, None) is False
    assert service.renew_lease(claim)
    cancelled = service.cancel_run(run["id"], actor="operator")
    assert cancelled["state"] == "cancelled"
    assert service._finish(claim, {"steps": 1}, None) is False
    uncertain = start(service, create(service))
    claim = service._claim("crashed")
    with service.store.transaction() as conn:
        conn.execute("UPDATE cp_attempts SET lease_until=now()-interval '1 second' WHERE id=%s",
                     (claim["attempt_id"],))
    assert service.worker_tick("replacement") is None
    recovered = service.get_run(uncertain["id"])
    assert recovered["state"] == "needs_attention"
    assert recovered["stages"][0]["attempts"][0]["state"] == "needs_attention"
    assert service._finish(claim, {"steps": 1}, None) is False
    assert service.worker_tick("replacement") is None


def test_dag_multiple_candidates_and_semantic_failure_not_transport_error(service):
    spec = catalog_document()["templates"][1]["spec"]
    spec["candidates"].append({"id": "candidate-b", "model": {"id": "other", "revision": "r2"},
                               "method": "fixture-b"})
    spec["stages"][0]["config"] = {"inputs": ["{}"], "outputs": ["invalid-json"]}
    second = {**spec["stages"][0], "id": "parallel", "candidate_id": "candidate-b", "config": {}}
    spec["stages"].append(second)
    run = service.create_run(spec, actor="operator", idempotency_key="multiple")
    start(service, run)
    drain(service)
    finished = service.get_run(run["id"])
    assert finished["state"] == "completed" and finished["outcome"]["quality"] == "failed"
    by_id = {s["id"]: s for s in finished["stages"]}
    assert by_id["stage-0"]["result"]["transport_success"]
    assert by_id["stage-1"]["result"]["offline"]
    assert by_id["parallel"]["candidate_id"] == "candidate-b"


def test_history_and_spec_are_immutable(service):
    run = start(service, create(service))
    for statement, values in (
        ("UPDATE cp_runs SET spec=%s WHERE id=%s", (json_value({}), run["id"])),
        ("DELETE FROM cp_events WHERE run_id=%s", (run["id"],)),
        ("UPDATE cp_approvals SET limits=%s WHERE run_id=%s", (json_value({}), run["id"])),
    ):
        with pytest.raises(ControlPlaneError), service.store.transaction() as conn:
            conn.execute(statement, values)


def test_shutdown_ownership_requires_attention_and_safe_artifact_import(service):
    run = start(service, create(service))
    service._claim("shutdown-worker")
    service.abandon_owned("unrelated")
    assert service.get_run(run["id"])["state"] == "running"
    service.abandon_owned("shutdown-worker")
    assert service.get_run(run["id"])["state"] == "needs_attention"
    metadata = service.import_artifact({"path": "/not/read", "value": {"x": 1}}, actor="operator")
    assert service.get_artifact(metadata["id"])["document"]["path"] == "/not/read"


def test_live_bridge_dispatch_is_disabled_until_trusted_profile_and_plan_approval(service):
    from neural_foundry.control_plane.bridges import TrustedRuntimeProfile
    calls = []
    payload = service.import_artifact({"proof": "mock-existing-stage"}, actor="operator")
    spec = catalog_document()["templates"][0]["spec"]
    spec["stages"][0].update(operation="proof.train", config={
        "runtime_profile": "test-qualified", "artifact_id": payload["id"], "role": "candidate-a"})
    with pytest.raises(ControlPlaneError) as error:
        service.create_run(spec, actor="operator", idempotency_key="disabled")
    assert error.value.code == "capability_disabled"
    service.install_runtime_profile(TrustedRuntimeProfile(
        "test-qualified", service.artifacts.root / "trusted-output", qualified=True,
        deployment_capabilities=frozenset({"model_training"}),
        callbacks={"proof.train": lambda context: calls.append(context) or {"model_calls": False}}))
    run = service.create_run(spec, actor="operator", idempotency_key="enabled")
    assert service.worker_tick("worker") is None and not calls
    start(service, run)
    drain(service)
    assert len(calls) == 1
    assert calls[0]["document"] == {"proof": "mock-existing-stage"}
    assert service.get_run(run["id"])["outcome"]["release_authorized"] is False


def test_heartbeat_keeps_lease_during_blocking_registered_callback(service):
    from neural_foundry.control_plane.bridges import TrustedRuntimeProfile
    service.lease_seconds = 3
    def blocked(context):
        time.sleep(3.5)
        assert service.worker_tick("competitor") is None
        return {"model_calls": False}
    service.install_runtime_profile(TrustedRuntimeProfile(
        "qualified-blocking", service.artifacts.root / "output", qualified=True,
        deployment_capabilities=frozenset({"model_training"}), callbacks={"proof.train": blocked}))
    payload = service.import_artifact({"bounded": True}, actor="operator")
    spec = catalog_document()["templates"][0]["spec"]
    spec["stages"][0].update(operation="proof.train", config={
        "runtime_profile": "qualified-blocking", "artifact_id": payload["id"], "role": "candidate-a"})
    run = service.create_run(spec, actor="operator", idempotency_key="heartbeat")
    start(service, run)
    assert service.worker_tick("blocking-owner")["accepted"]
    assert service.get_run(run["id"])["state"] == "completed"


def test_approved_plan_cannot_dispatch_after_execution_implementation_changes(service, monkeypatch):
    from neural_foundry.control_plane import implementation

    run = start(service, create(service))
    approved_digest = run["plan"]["digest"]
    original = implementation.execution_implementation()
    monkeypatch.setattr(implementation, "execution_implementation",
                        lambda: {**original, "digest": "sha256:" + "f" * 64})
    assert service.worker_tick("upgraded-worker") is None
    attention = service.get_run(run["id"])
    assert attention["state"] == "needs_attention"
    assert attention["plan"]["digest"] == approved_digest
    assert attention["outcome"]["error"]["code"] == "implementation_changed"
    assert attention["outcome"]["release_authorized"] is False
    assert not attention["stages"][0]["attempts"]
    assert service.get_events(run["id"])[-1]["type"] == "run.implementation_changed"
    # Restarting with the old code does not automatically retry a paused run.
    monkeypatch.setattr(implementation, "execution_implementation", lambda: original)
    assert service.worker_tick("original-worker") is None
    assert service.get_run(run["id"])["state"] == "needs_attention"
