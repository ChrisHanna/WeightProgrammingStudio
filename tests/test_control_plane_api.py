"""HTTP boundary tests: the fake is only the injected durable-service facade."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from neural_foundry.control_plane.api import MAX_BODY_BYTES, create_app
from neural_foundry.control_plane.models import ControlPlaneError

TOKEN = "unit-test-operator-secret"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class Facade:
    def __init__(self):
        self.calls = []
        self.run = {
            "id": "run-1", "state": "draft", "spec": {"version": 1},
            "spec_digest": "spec-digest", "plan": None, "approval": None,
            "stages": [], "outcome": None, "created_at": "now", "updated_at": "now",
        }
        self.created = {}
        self.started = set()

    def health(self):
        return {"ready": True, "private_database_details": "not-public"}

    def catalog_document(self):
        self.calls.append(("catalog",))
        return {"version": 1, "plugins": [], "templates": []}

    def create_run(self, spec, *, actor, idempotency_key):
        self.calls.append(("create", spec, actor, idempotency_key))
        self.created.setdefault(idempotency_key, copy.deepcopy(self.run))
        return self.created[idempotency_key]

    def list_runs(self, limit=100):
        self.calls.append(("list", limit))
        return [self.run]

    def get_run(self, run_id):
        self.calls.append(("get", run_id))
        if run_id != self.run["id"]:
            raise ControlPlaneError("not_found", "Run not found", 404)
        return self.run

    def plan_run(self, run_id, *, actor):
        self.calls.append(("plan", run_id, actor))
        self.run.update(state="planned", plan={"digest": "plan-1", "stages": []})
        return self.run

    def approve_run(self, run_id, *, actor, plan_digest, limits):
        self.calls.append(("approve", run_id, actor, plan_digest, limits))
        if plan_digest != (self.run["plan"] or {}).get("digest"):
            raise ControlPlaneError("stale_plan", "Plan digest does not match", 409)
        approval = {"id": "approval-1", "run_id": run_id,
                    "plan_digest": plan_digest, "limits": limits}
        self.run.update(state="approved", approval=approval)
        return approval

    def start_run(self, run_id, *, actor, approval_id, idempotency_key):
        self.calls.append(("start", run_id, actor, approval_id, idempotency_key))
        if approval_id != (self.run["approval"] or {}).get("id"):
            raise ControlPlaneError("approval_required", "Valid approval required", 409)
        self.started.add(idempotency_key)
        self.run["state"] = "queued"
        return self.run

    def cancel_run(self, run_id, *, actor):
        self.calls.append(("cancel", run_id, actor))
        self.run["state"] = "cancelled"
        return self.run

    def get_events(self, run_id, after=0):
        self.calls.append(("events", run_id, after))
        return [{"seq": 2, "type": "planned"}] if after < 2 else []

    def import_artifact(self, document, *, actor):
        self.calls.append(("import", document, actor))
        return {"id": "artifact-1", "document": document}

    def get_artifact(self, artifact_id):
        self.calls.append(("artifact", artifact_id))
        return {"id": artifact_id}


@pytest.fixture
def operator(monkeypatch):
    monkeypatch.delenv("CP_OPERATOR_TOKEN_FILE", raising=False)
    monkeypatch.setenv("CP_OPERATOR_TOKEN", TOKEN)
    monkeypatch.setenv("CP_ALLOWED_HOSTS", "testserver,localhost,127.0.0.1")
    facade = Facade()
    with TestClient(create_app(facade)) as client:
        yield client, facade


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("GET", "/v1/catalog", None), ("GET", "/v1/runs", None),
        ("GET", "/v1/runs/run-1", None), ("GET", "/v1/runs/run-1/events", None),
        ("GET", "/v1/artifacts/a", None), ("POST", "/v1/artifacts", {}),
        ("POST", "/v1/runs", {"spec": {}, "idempotency_key": "new"}),
        ("POST", "/v1/runs/run-1/plan", {}),
        ("POST", "/v1/runs/run-1/approve", {"plan_digest": "p", "limits": {}}),
        ("POST", "/v1/runs/run-1/start", {"approval_id": "a", "idempotency_key": "s"}),
        ("POST", "/v1/runs/run-1/cancel", {}),
    ],
)
def test_all_service_endpoints_require_bearer(operator, method, path, body):
    client, facade = operator
    for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": f"Basic {TOKEN}"}):
        response = client.request(method, path, json=body, headers=headers)
        assert response.status_code == 401
        assert response.json()["error"]["message"] == "Operator authentication required"
    assert facade.calls == []


def test_static_and_minimal_health_are_public_and_never_execute(operator):
    client, facade = operator
    assert client.get("/").status_code == 200
    script = client.get("/static/app.js")
    assert script.status_code == 200
    assert "innerHTML" not in script.text
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"ready": True}
    assert "default-src 'self'" in script.headers["content-security-policy"]
    assert script.headers["cache-control"] == "no-store"
    assert facade.calls == []


def test_explicit_commands_forward_authority_and_retry_keys(operator):
    client, facade = operator
    body = {"spec": {"version": 1}, "idempotency_key": "draft-key"}
    assert client.post("/v1/runs", json=body, headers=AUTH).json() == (
        client.post("/v1/runs", json=body, headers=AUTH).json()
    )
    assert len(facade.created) == 1
    denied = client.post("/v1/runs/run-1/start", headers=AUTH,
                         json={"approval_id": "missing", "idempotency_key": "start-key"})
    assert denied.status_code == 409
    assert denied.json()["error"]["code"] == "approval_required"
    plan = client.post("/v1/runs/run-1/plan", json={}, headers=AUTH).json()
    assert plan["plan"]["digest"] == "plan-1"
    stale = client.post("/v1/runs/run-1/approve", headers=AUTH,
                        json={"plan_digest": "old", "limits": {}})
    assert stale.status_code == 409
    approval = client.post("/v1/runs/run-1/approve", headers=AUTH,
                           json={"plan_digest": "plan-1", "limits": {"max_rows": 2}}).json()
    assert facade.started == set()
    for _ in range(2):
        assert client.post("/v1/runs/run-1/start", headers=AUTH,
                           json={"approval_id": approval["id"], "idempotency_key": "start-key"}
                           ).status_code == 200
    assert facade.started == {"start-key"}
    assert ("approve", "run-1", "operator", "plan-1", {"max_rows": 2}) in facade.calls
    assert ("start", "run-1", "operator", "approval-1", "start-key") in facade.calls
    client.post("/v1/runs/run-1/cancel", headers=AUTH)
    assert ("cancel", "run-1", "operator") in facade.calls


def test_gets_never_plan_approve_or_start(operator):
    client, facade = operator
    for _ in range(2):
        for path in ("/", "/v1/catalog", "/v1/runs", "/v1/runs/run-1",
                     "/v1/runs/run-1/events?after=1"):
            assert client.get(path, headers=AUTH).status_code == 200
    assert all(call[0] in {"catalog", "list", "get", "events"} for call in facade.calls)
    assert ("events", "run-1", 1) in facade.calls


@pytest.mark.parametrize("body", [
    b"{bad", b"[]", b'{"spec": {}, "idempotency_key": 123}',
    b'{"spec": {}, "idempotency_key": "x", "actor": "admin"}',
])
def test_invalid_bodies_are_structured_and_do_not_reach_service(operator, body):
    client, facade = operator
    response = client.post("/v1/runs", headers={**AUTH, "Content-Type": "application/json"},
                           content=body)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_request",
                                        "message": "Request body or parameters are invalid"}}
    assert facade.calls == []


def test_origin_host_and_size_boundaries(operator):
    client, facade = operator
    for headers, status in (
        ({"Origin": "https://evil.example"}, 403),
        ({"Origin": "null"}, 403),
        ({"Host": "evil.example"}, 400),
        ({"Host": "evil@testserver"}, 400),
        ({"Host": "testserver/path"}, 400),
        ({"Sec-Fetch-Site": "cross-site"}, 403),
    ):
        response = client.get("/v1/catalog", headers={**AUTH, **headers})
        assert response.status_code == status
        assert "error" in response.json()
    assert client.get("/v1/catalog", headers={**AUTH, "Origin": "http://testserver"}).status_code == 200
    facade.calls.clear()
    response = client.post("/v1/artifacts", headers=AUTH, content=b"x" * (MAX_BODY_BYTES + 1))
    assert response.status_code == 413
    assert facade.calls == []
    assert client.get("/v1/runs?limit=101", headers=AUTH).status_code == 422
    assert client.get("/v1/runs/run-1/events?after=-1", headers=AUTH).status_code == 422


def test_error_and_readiness_do_not_expose_secrets(operator):
    client, facade = operator

    def broken():
        raise RuntimeError("postgresql://password:secret@private-host/db")

    facade.catalog_document = broken
    response = client.get("/v1/catalog", headers=AUTH)
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "postgresql" not in response.text and "secret" not in response.text
    assert "content-security-policy" in response.headers
    facade.health = broken
    assert client.get("/ready").status_code == 503
    assert client.get("/ready").json() == {"ready": False}
    missing = client.get("/v1/runs/missing", headers=AUTH)
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "not_found"


def test_artifacts_forward_documents_and_actor(operator):
    client, facade = operator
    document = {"version": 1, "receipt": {"quality": "not_evaluated"}}
    response = client.post("/v1/artifacts", headers=AUTH, json=document)
    assert response.status_code == 200
    assert ("import", document, "operator") in facade.calls
    assert client.get("/v1/artifacts/artifact-1", headers=AUTH).json() == {"id": "artifact-1"}


def test_token_file_is_runtime_only_and_missing_token_fails_startup(monkeypatch):
    secret = Path("unit-operator-token")
    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: TOKEN + "\n")
    monkeypatch.setenv("CP_OPERATOR_TOKEN_FILE", str(secret))
    monkeypatch.setenv("CP_OPERATOR_TOKEN", "ignored")
    monkeypatch.setenv("CP_ALLOWED_HOSTS", "testserver")
    with TestClient(create_app(Facade())) as client:
        assert client.get("/v1/catalog", headers=AUTH).status_code == 200
        assert client.get(f"/v1/catalog?token={TOKEN}").status_code == 401
    monkeypatch.delenv("CP_OPERATOR_TOKEN_FILE")
    monkeypatch.delenv("CP_OPERATOR_TOKEN")
    with (
        pytest.raises(RuntimeError, match="nonempty operator token"),
        TestClient(create_app(Facade())),
    ):
        pass
