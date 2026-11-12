"""Shared operator facade and generic, approval-bound stage DAG controller."""

from __future__ import annotations

import threading
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .artifacts import ArtifactStore
from .catalog import OPERATIONS, catalog_document
from .models import (
    ControlPlaneError,
    actor_name,
    compile_plan,
    digest,
    document,
    fail,
    identifier,
    validate_spec,
)
from .store import Store, json_value, one

if TYPE_CHECKING:
    from psycopg import Connection

    from .bridges import TrustedRuntimeProfile


def _id(prefix: str) -> str:
    return prefix + "-" + uuid.uuid4().hex


def _public(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _public(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_public(val) for val in value]
    return value


def _public_document(value: dict[str, Any]) -> dict[str, Any]:
    return {key: _public(item) for key, item in value.items()}


class ControlPlaneService:
    def __init__(self, dsn: str, artifact_root: Path, *, lease_seconds: int = 30) -> None:
        if type(lease_seconds) is not int or not 1 <= lease_seconds <= 3600:
            fail("invalid_lease", "Lease must be between one and 3600 seconds.")
        self.store, self.artifacts = Store(dsn), ArtifactStore(artifact_root)
        self.lease_seconds = lease_seconds
        self.runtime_profiles: dict[str, TrustedRuntimeProfile] = {}

    def install_runtime_profile(self, profile: TrustedRuntimeProfile) -> None:
        """Trusted in-process deployment hook, deliberately absent from the HTTP facade."""
        from .bridges import TrustedRuntimeProfile
        if not isinstance(profile, TrustedRuntimeProfile) or not profile.qualified:
            fail("invalid_runtime_profile", "A qualified trusted runtime profile is required.", 403)
        self.runtime_profiles[profile.id] = profile

    def _operations(self) -> dict[str, dict[str, Any]]:
        from .bridges import CAPABILITIES
        operations = deepcopy(OPERATIONS)
        for operation, capability in CAPABILITIES.items():
            profiles = [p.id for p in self.runtime_profiles.values() if capability in p.deployment_capabilities]
            operations[operation]["enabled"] = bool(profiles)
            operations[operation]["qualified_profiles"] = profiles
        return operations

    def catalog_document(self) -> dict[str, Any]:
        result = catalog_document()
        result["plugins"] = list(self._operations().values())
        return result

    def health(self) -> dict[str, Any]:
        return self.store.health()

    def create_run(self, spec: dict[str, Any], *, actor: str, idempotency_key: str) -> dict[str, Any]:
        actor_name(actor)
        identifier(idempotency_key, "idempotency key")
        spec = validate_spec(spec, self._operations())
        for stage in spec["stages"]:
            if stage["operation"].startswith("proof."):
                profiles = self._operations()[stage["operation"]]["qualified_profiles"]
                if stage["config"]["runtime_profile"] not in profiles:
                    fail("runtime_not_authorized", "Stage runtime profile is not qualified for this capability.", 403)
        spec_digest = digest(spec)
        run_id = _id("run")
        with self.store.transaction() as conn:
            # Unique constraint serializes identical simultaneous create requests.
            row = conn.execute(
                "INSERT INTO cp_runs(id,actor,create_key,state,spec,spec_digest) "
                "VALUES (%s,%s,%s,'created',%s,%s) ON CONFLICT(actor,create_key) DO NOTHING RETURNING id",
                (run_id, actor, idempotency_key, json_value(spec), spec_digest)).fetchone()
            if row:
                for stage in spec["stages"]:
                    conn.execute("INSERT INTO cp_stages(run_id,id,operation,definition) VALUES (%s,%s,%s,%s)",
                                 (run_id, stage["id"], stage["operation"], json_value(stage)))
                self.store.event(conn, run_id, "run.created", {"actor": actor, "spec_digest": spec_digest})
            else:
                existing = one(conn.execute("SELECT id,spec_digest FROM cp_runs WHERE actor=%s AND create_key=%s",
                                            (actor, idempotency_key)))
                if existing["spec_digest"] != spec_digest:
                    fail("idempotency_conflict", "Idempotency key was already used for a different specification.", 409)
                run_id = existing["id"]
        return self.get_run(run_id)

    def list_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        if type(limit) is not int or not 1 <= limit <= 100:
            fail("invalid_limit", "Run list limit must be between one and 100.")
        with self.store.transaction() as conn:
            rows = conn.execute("SELECT id FROM cp_runs ORDER BY created_at DESC,id LIMIT %s", (limit,)).fetchall()
            return [self._dto(conn, self.store.run(conn, row["id"])) for row in rows]

    def _dto(self, conn: Connection[dict[str, Any]], row: dict[str, Any]) -> dict[str, Any]:
        stages = conn.execute("SELECT * FROM cp_stages WHERE run_id=%s ORDER BY id", (row["id"],)).fetchall()
        approval = (conn.execute("SELECT id,run_id,plan_digest,limits,actor,created_at FROM cp_approvals WHERE id=%s",
                                 (row["approval_id"],)).fetchone() if row["approval_id"] else None)
        result = {key: row[key] for key in ("id", "state", "spec", "spec_digest", "plan", "outcome",
                                           "created_at", "updated_at")}
        result["approval"] = approval
        result["stages"] = [{
            "id": stage["id"], "operation": stage["operation"], "state": stage["state"],
            "candidate_id": stage["definition"].get("candidate_id"),
            "depends_on": stage["definition"]["depends_on"], "progress": stage["progress"],
            "result": stage["result"], "artifact_id": stage["artifact_id"],
            "attempts": conn.execute(
                "SELECT id,number,owner,state,lease_until,started_at,finished_at,error,receipt "
                "FROM cp_attempts WHERE run_id=%s AND stage_id=%s ORDER BY number",
                (row["id"], stage["id"])).fetchall(),
        } for stage in stages]
        return _public_document(result)

    def get_run(self, run_id: str) -> dict[str, Any]:
        identifier(run_id, "run ID")
        with self.store.transaction() as conn:
            return self._dto(conn, self.store.run(conn, run_id))

    def plan_run(self, run_id: str, *, actor: str) -> dict[str, Any]:
        actor_name(actor)
        with self.store.transaction() as conn:
            run = self.store.run(conn, run_id, lock=True)
            if run["plan"] is None:
                if run["state"] != "created":
                    fail("invalid_state", "Only a newly created run may be planned.", 409)
                plan = compile_plan(run["spec"])
                conn.execute("UPDATE cp_runs SET state='planned',plan=%s,updated_at=now() WHERE id=%s",
                             (json_value(plan), run_id))
                self.store.event(conn, run_id, "run.planned", {"actor": actor, "plan_digest": plan["digest"]})
        return self.get_run(run_id)

    def approve_run(self, run_id: str, *, actor: str, plan_digest: str,
                    limits: dict[str, Any]) -> dict[str, Any]:
        actor_name(actor)
        limits = document(limits)
        with self.store.transaction() as conn:
            run = self.store.run(conn, run_id, lock=True)
            if run["state"] not in ("planned", "approved") or not run["plan"]:
                fail("invalid_state", "Run must be planned before approval.", 409)
            if plan_digest != run["plan"]["digest"]:
                fail("plan_mismatch", "Approval must bind the exact compiled plan.", 409)
            compiled = run["plan"]["limits"]
            if not isinstance(limits, dict) or set(limits) - set(compiled):
                fail("invalid_limits", "Approval limits contain unsupported fields.")
            approved = dict(compiled)
            for name, value in limits.items():
                if type(value) is not int or value < compiled[name]:
                    fail("budget_exceeded", "Approval limits must cover the complete compiled plan.")
                # Larger-than-plan approvals are not granted silently.
                if value != compiled[name]:
                    fail("invalid_limits", "Approval limits must exactly match compiled plan limits.")
                approved[name] = value
            if run["approval_id"]:
                return _public_document(one(conn.execute("SELECT * FROM cp_approvals WHERE id=%s",
                                                         (run["approval_id"],))))
            approval_id = _id("approval")
            row = one(conn.execute(
                "INSERT INTO cp_approvals(id,run_id,actor,plan_digest,limits) VALUES (%s,%s,%s,%s,%s) RETURNING *",
                (approval_id, run_id, actor, plan_digest, json_value(approved))))
            conn.execute("UPDATE cp_runs SET approval_id=%s,state='approved',updated_at=now() WHERE id=%s",
                         (approval_id, run_id))
            self.store.event(conn, run_id, "run.approved", {"actor": actor, "approval_id": approval_id,
                                                           "plan_digest": plan_digest, "limits": approved})
            return _public_document(row)

    def start_run(self, run_id: str, *, actor: str, approval_id: str,
                  idempotency_key: str) -> dict[str, Any]:
        actor_name(actor)
        identifier(idempotency_key, "idempotency key")
        with self.store.transaction() as conn:
            run = self.store.run(conn, run_id, lock=True)
            if run["start_key"] is not None:
                if run["start_key"] != idempotency_key or run["approval_id"] != approval_id:
                    fail("idempotency_conflict", "Run already has a different start authorization.", 409)
            else:
                if run["state"] != "approved" or run["approval_id"] != approval_id:
                    fail("approval_required", "An exact current plan approval is required before dispatch.", 409)
                approval = one(conn.execute("SELECT * FROM cp_approvals WHERE id=%s", (approval_id,)))
                if approval["plan_digest"] != run["plan"]["digest"] or approval["limits"] != run["plan"]["limits"]:
                    fail("approval_mismatch", "Plan or budget no longer matches approval.", 409)
                conn.execute("UPDATE cp_runs SET state='queued',start_key=%s,updated_at=now() WHERE id=%s",
                             (idempotency_key, run_id))
                self.store.event(conn, run_id, "run.started", {"actor": actor, "approval_id": approval_id})
        return self.get_run(run_id)

    def cancel_run(self, run_id: str, *, actor: str) -> dict[str, Any]:
        actor_name(actor)
        with self.store.transaction() as conn:
            run = self.store.run(conn, run_id, lock=True)
            if run["state"] not in ("completed", "failed", "cancelled"):
                conn.execute("UPDATE cp_runs SET state='cancelled',updated_at=now(),outcome=%s WHERE id=%s",
                             (json_value({"execution": "cancelled", "quality": "not_evaluated",
                                          "release_authorized": False}), run_id))
                conn.execute("UPDATE cp_stages SET state='cancelled' WHERE run_id=%s AND state IN ('pending','running')",
                             (run_id,))
                conn.execute("UPDATE cp_attempts SET state='cancelled',finished_at=now() "
                             "WHERE run_id=%s AND state='started'", (run_id,))
                self.store.event(conn, run_id, "run.cancelled", {"actor": actor})
        return self.get_run(run_id)

    def get_events(self, run_id: str, after: int = 0) -> list[dict[str, Any]]:
        if type(after) is not int or after < 0:
            fail("invalid_cursor", "Event cursor must be a nonnegative integer.")
        with self.store.transaction() as conn:
            self.store.run(conn, run_id)
            return [_public_document(row) for row in conn.execute(
                "SELECT sequence,type,timestamp,stage_id,payload FROM cp_events "
                "WHERE run_id=%s AND sequence>%s ORDER BY sequence LIMIT 1000", (run_id, after)).fetchall()]

    def import_artifact(self, document: dict[str, Any], *, actor: str) -> dict[str, Any]:
        actor_name(actor)
        metadata = self.artifacts.put(document)
        with self.store.transaction() as conn:
            conn.execute("INSERT INTO cp_artifacts(id,metadata,actor) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                         (metadata["id"], json_value(metadata), actor))
        return metadata

    def get_artifact(self, artifact_id: str) -> dict[str, Any]:
        # Validate before sending user data into storage queries.
        self.artifacts._path(artifact_id)
        with self.store.transaction() as conn:
            row = conn.execute("SELECT metadata FROM cp_artifacts WHERE id=%s", (artifact_id,)).fetchone()
            if row is None:
                fail("artifact_not_found", "Artifact does not exist.", 404)
        return {**row["metadata"], "document": self.artifacts.read(artifact_id)}

    def _expire(self, conn: Connection[dict[str, Any]], run: dict[str, Any]) -> bool:
        expired = conn.execute(
            "SELECT id,stage_id FROM cp_attempts WHERE run_id=%s AND state='started' AND lease_until<=now()",
            (run["id"],)).fetchall()
        if not expired:
            return False
        for attempt in expired:
            error = {"code": "lease_expired", "message": "Started attempt has uncertain completion; operator attention required."}
            conn.execute("UPDATE cp_attempts SET state='needs_attention',finished_at=now(),error=%s WHERE id=%s",
                         (json_value(error), attempt["id"]))
            conn.execute("UPDATE cp_stages SET state='needs_attention' WHERE run_id=%s AND id=%s",
                         (run["id"], attempt["stage_id"]))
            self.store.event(conn, run["id"], "attempt.lease_expired", error, attempt["stage_id"])
        conn.execute("UPDATE cp_runs SET state='needs_attention',updated_at=now(),outcome=%s WHERE id=%s",
                     (json_value({"execution": "needs_attention", "quality": "not_evaluated",
                                  "release_authorized": False}), run["id"]))
        return True

    def _claim(self, worker_id: str) -> dict[str, Any] | None:
        from .implementation import execution_implementation

        identifier(worker_id, "worker ID")
        current_implementation = execution_implementation()
        with self.store.transaction() as conn:
            # Run-first locking is shared by claim, finish, cancellation and recovery.
            runs = conn.execute("SELECT * FROM cp_runs WHERE state IN ('queued','running') "
                                "ORDER BY created_at,id FOR UPDATE SKIP LOCKED LIMIT 32").fetchall()
            for run in runs:
                if self._expire(conn, run):
                    continue
                approved_implementation = run["plan"].get("execution_implementation")
                if approved_implementation != current_implementation:
                    error = {"code": "implementation_changed",
                             "message": "Execution implementation differs from the approved plan; operator attention required."}
                    conn.execute("UPDATE cp_attempts SET state='needs_attention',finished_at=now(),error=%s "
                                 "WHERE run_id=%s AND state='started'", (json_value(error), run["id"]))
                    conn.execute("UPDATE cp_stages SET state='needs_attention' "
                                 "WHERE run_id=%s AND state IN ('pending','running')", (run["id"],))
                    conn.execute("UPDATE cp_runs SET state='needs_attention',updated_at=now(),outcome=%s WHERE id=%s",
                                 (json_value({"execution": "needs_attention", "quality": "not_evaluated",
                                              "release_authorized": False, "error": error}), run["id"]))
                    self.store.event(conn, run["id"], "run.implementation_changed",
                                     {**error, "approved": approved_implementation, "current": current_implementation})
                    continue
                stages = conn.execute("SELECT * FROM cp_stages WHERE run_id=%s ORDER BY id",
                                      (run["id"],)).fetchall()
                states = {s["id"]: s["state"] for s in stages}
                for stage in stages:
                    definition = next(s for s in run["plan"]["stages"] if s["id"] == stage["id"])
                    if stage["state"] != "pending" or any(states[d] != "completed" for d in definition["depends_on"]):
                        continue
                    # Check approval and immutable-plan integrity again at the actual dispatch boundary.
                    plan_body = {k: v for k, v in run["plan"].items() if k != "digest"}
                    approval = conn.execute("SELECT * FROM cp_approvals WHERE id=%s",
                                            (run["approval_id"],)).fetchone()
                    if (digest(plan_body) != run["plan"]["digest"] or not approval
                            or approval["plan_digest"] != run["plan"]["digest"]
                            or approval["limits"] != run["plan"]["limits"]):
                        fail("approval_mismatch", "Dispatch authorization failed integrity verification.", 409)
                    attempt_id = _id("attempt")
                    conn.execute(
                        "INSERT INTO cp_attempts(id,run_id,stage_id,number,owner,state,lease_until) "
                        "VALUES (%s,%s,%s,%s,%s,'started',now()+(%s * interval '1 second'))",
                        (attempt_id, run["id"], stage["id"], stage["attempt_count"] + 1,
                         worker_id, self.lease_seconds))
                    progress = {"kind": "steps", "completed": 0, "total": definition["limits"]["steps"]}
                    conn.execute("UPDATE cp_stages SET state='running',attempt_count=attempt_count+1,progress=%s "
                                 "WHERE run_id=%s AND id=%s", (json_value(progress), run["id"], stage["id"]))
                    conn.execute("UPDATE cp_runs SET state='running',updated_at=now() WHERE id=%s", (run["id"],))
                    self.store.event(conn, run["id"], "attempt.started",
                                     {"attempt_id": attempt_id, "worker_id": worker_id, "progress": progress}, stage["id"])
                    by_id = {s["id"]: s for s in stages}
                    dependencies = [by_id[sid]["result"] for sid in definition["depends_on"]]
                    return {"run_id": run["id"], "stage": definition, "attempt_id": attempt_id,
                            "worker_id": worker_id, "dependencies": dependencies,
                            "approval": approval, "plan_digest": run["plan"]["digest"],
                            "candidate_model": next((c["model"] for c in run["spec"]["candidates"]
                                                     if c["id"] == definition.get("candidate_id")), None)}
        return None

    def renew_lease(self, claim: dict[str, Any]) -> bool:
        with self.store.transaction() as conn:
            self.store.run(conn, claim["run_id"], lock=True)
            row = conn.execute("UPDATE cp_attempts SET lease_until=now()+(%s * interval '1 second') "
                               "WHERE id=%s AND owner=%s AND state='started' AND lease_until>now() RETURNING id",
                               (self.lease_seconds, claim["attempt_id"], claim["worker_id"])).fetchone()
            return row is not None

    def _finish(self, claim: dict[str, Any], result: dict[str, Any] | None,
                error: dict[str, Any] | None) -> bool:
        with self.store.transaction() as conn:
            run = self.store.run(conn, claim["run_id"], lock=True)
            if run["state"] != "running" or self._expire(conn, run):
                return False
            attempt = conn.execute("SELECT * FROM cp_attempts WHERE id=%s AND owner=%s AND state='started'",
                                   (claim["attempt_id"], claim["worker_id"])).fetchone()
            if not attempt or attempt["run_id"] != run["id"] or attempt["stage_id"] != claim["stage"]["id"]:
                return False
            artifact = self.artifacts.put(result) if result is not None else None
            if artifact and artifact["byte_size"] > claim["stage"]["limits"]["recorded_bytes"]:
                error = {"code": "budget_exceeded", "message": "Result exceeds approved recorded-byte limit."}
                result, artifact = None, None
            state = "failed" if error else "completed"
            receipt = {"artifact_id": artifact["id"], "result_digest": digest(result)} if artifact else None
            if artifact:
                conn.execute("INSERT INTO cp_artifacts(id,metadata,actor) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                             (artifact["id"], json_value(artifact), claim["worker_id"]))
            conn.execute("UPDATE cp_attempts SET state=%s,finished_at=now(),receipt=%s,error=%s WHERE id=%s",
                         (state, json_value(receipt), json_value(error), claim["attempt_id"]))
            progress = {"kind": "steps", "completed": (result or {}).get("steps", 0),
                        "total": claim["stage"]["limits"]["steps"]}
            conn.execute("UPDATE cp_stages SET state=%s,result=%s,artifact_id=%s,progress=%s WHERE run_id=%s AND id=%s",
                         (state, json_value(result), artifact["id"] if artifact else None, json_value(progress),
                          run["id"], claim["stage"]["id"]))
            self.store.event(conn, run["id"], "attempt." + state,
                             {"attempt_id": claim["attempt_id"], "error": error, "receipt": receipt,
                              "progress": progress}, claim["stage"]["id"])
            if error:
                conn.execute("UPDATE cp_stages SET state='blocked' WHERE run_id=%s AND state='pending'", (run["id"],))
                outcome = {"execution": "failed", "quality": "not_evaluated", "release_authorized": False}
                conn.execute("UPDATE cp_runs SET state='failed',outcome=%s,updated_at=now() WHERE id=%s",
                             (json_value(outcome), run["id"]))
            else:
                rows = conn.execute("SELECT state,result FROM cp_stages WHERE run_id=%s", (run["id"],)).fetchall()
                if all(r["state"] == "completed" for r in rows):
                    qualities = [r["result"].get("quality") for r in rows]
                    quality = "failed" if "failed" in qualities else ("passed" if "passed" in qualities else "not_evaluated")
                    outcome = {"execution": "completed", "quality": quality, "release_authorized": False,
                               "model_calls": False if all(r["result"].get("model_calls") is False
                                                           for r in rows) else "not_asserted"}
                    conn.execute("UPDATE cp_runs SET state='completed',outcome=%s,updated_at=now() WHERE id=%s",
                                 (json_value(outcome), run["id"]))
                    self.store.event(conn, run["id"], "run.completed", outcome)
            return True

    def worker_tick(self, worker_id: str) -> dict[str, Any] | None:
        claim = self._claim(worker_id)
        if claim is None:
            return None
        from .handlers import execute
        stopped = threading.Event()
        lease_lost = threading.Event()

        def heartbeat() -> None:
            while not stopped.wait(max(.1, self.lease_seconds / 3)):
                try:
                    if not self.renew_lease(claim):
                        lease_lost.set()
                        return
                except ControlPlaneError:
                    lease_lost.set()
                    return

        keeper = threading.Thread(target=heartbeat, name="cp-lease-heartbeat", daemon=True)
        keeper.start()
        try:
            stage = claim["stage"]
            if stage["operation"].startswith("proof."):
                from .bridges import dispatch
                config = stage["config"]
                profile = self.runtime_profiles.get(config["runtime_profile"])
                if profile is None:
                    fail("runtime_not_authorized", "Worker does not have the approved qualified runtime.", 403)
                result = dispatch(stage["operation"], profile,
                                  document=self.get_artifact(config["artifact_id"])["document"],
                                  role=config["role"], run_id=claim["run_id"],
                                  plan_digest=claim["plan_digest"], approval=claim["approval"], limits=stage["limits"])
            else:
                result = execute(stage, claim["dependencies"])
            result = {**result, "candidate_model": claim["candidate_model"]}
            error = None
        except ControlPlaneError as exc:
            result, error = None, {"code": exc.code, "message": exc.message}
        except Exception:  # noqa: BLE001 -- persist sanitized failure, never raw provider output
            result, error = None, {"code": "handler_error", "message": "Software fixture execution failed."}
        finally:
            stopped.set()
            keeper.join(timeout=6)
        accepted = False if lease_lost.is_set() else self._finish(claim, result, error)
        return {"run_id": claim["run_id"], "stage_id": claim["stage"]["id"],
                "attempt_id": claim["attempt_id"], "accepted": accepted,
                "state": "failed" if error else "completed"}

    def abandon_owned(self, worker_id: str) -> None:
        """Graceful shutdown never returns a started attempt to the runnable queue."""
        identifier(worker_id, "worker ID")
        with self.store.transaction() as conn:
            runs = conn.execute("SELECT r.* FROM cp_runs r WHERE r.state='running' AND EXISTS "
                                "(SELECT 1 FROM cp_attempts a WHERE a.run_id=r.id AND a.owner=%s AND a.state='started') "
                                "ORDER BY r.id FOR UPDATE", (worker_id,)).fetchall()
            for run in runs:
                conn.execute("UPDATE cp_attempts SET lease_until=now() WHERE run_id=%s AND owner=%s AND state='started'",
                             (run["id"], worker_id))
                self._expire(conn, run)
