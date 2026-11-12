"""PostgreSQL-only persistence. Transactions, not provider exactly-once semantics."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from .models import ControlPlaneError, fail

if TYPE_CHECKING:
    from psycopg import Connection, Cursor
    from psycopg.types.json import Jsonb

SCHEMA_VERSION = 3
MIGRATION = """
CREATE TABLE IF NOT EXISTS cp_schema(version integer PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE cp_runs(
 id text PRIMARY KEY, actor text NOT NULL, create_key text NOT NULL,
 state text NOT NULL, spec jsonb NOT NULL, spec_digest text NOT NULL,
 plan jsonb, approval_id text, start_key text, outcome jsonb,
 created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(),
 UNIQUE(actor,create_key));
CREATE TABLE cp_stages(
 run_id text NOT NULL REFERENCES cp_runs(id), id text NOT NULL, operation text NOT NULL,
 definition jsonb NOT NULL, state text NOT NULL DEFAULT 'pending',
 attempt_count integer NOT NULL DEFAULT 0, progress jsonb,
 result jsonb, artifact_id text, PRIMARY KEY(run_id,id));
CREATE TABLE cp_approvals(
 id text PRIMARY KEY, run_id text NOT NULL REFERENCES cp_runs(id),
 actor text NOT NULL, plan_digest text NOT NULL, limits jsonb NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE cp_attempts(
 id text PRIMARY KEY, run_id text NOT NULL, stage_id text NOT NULL,
 number integer NOT NULL, owner text NOT NULL, state text NOT NULL,
 lease_until timestamptz NOT NULL, started_at timestamptz NOT NULL DEFAULT now(),
 finished_at timestamptz, receipt jsonb, error jsonb,
 FOREIGN KEY(run_id,stage_id) REFERENCES cp_stages(run_id,id),
 UNIQUE(run_id,stage_id,number));
CREATE INDEX cp_attempt_lease ON cp_attempts(lease_until) WHERE state='started';
CREATE TABLE cp_events(
 sequence bigserial PRIMARY KEY, run_id text NOT NULL REFERENCES cp_runs(id),
 type text NOT NULL, timestamp timestamptz NOT NULL DEFAULT now(),
 stage_id text, payload jsonb NOT NULL);
CREATE INDEX cp_run_events ON cp_events(run_id,sequence);
CREATE TABLE cp_artifacts(
 id text PRIMARY KEY, metadata jsonb NOT NULL, actor text NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now());
CREATE FUNCTION cp_immutable_history() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION 'immutable control-plane history'; END $$;
CREATE TRIGGER cp_event_immutable BEFORE UPDATE OR DELETE ON cp_events
 FOR EACH ROW EXECUTE FUNCTION cp_immutable_history();
CREATE TRIGGER cp_approval_immutable BEFORE UPDATE OR DELETE ON cp_approvals
 FOR EACH ROW EXECUTE FUNCTION cp_immutable_history();
CREATE TRIGGER cp_artifact_immutable BEFORE UPDATE OR DELETE ON cp_artifacts
 FOR EACH ROW EXECUTE FUNCTION cp_immutable_history();
CREATE FUNCTION cp_run_identity() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF NEW.spec IS DISTINCT FROM OLD.spec OR NEW.spec_digest IS DISTINCT FROM OLD.spec_digest
 OR (OLD.plan IS NOT NULL AND NEW.plan IS DISTINCT FROM OLD.plan)
 THEN RAISE EXCEPTION 'immutable run specification/plan'; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER cp_run_identity_guard BEFORE UPDATE ON cp_runs
 FOR EACH ROW EXECUTE FUNCTION cp_run_identity();
"""

MIGRATION_2 = """
CREATE OR REPLACE FUNCTION cp_attempt_history() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF TG_OP='DELETE' OR OLD.state<>'started' THEN
 RAISE EXCEPTION 'immutable finished attempt'; END IF;
 IF NEW.id<>OLD.id OR NEW.owner<>OLD.owner OR NEW.run_id<>OLD.run_id
 OR NEW.stage_id<>OLD.stage_id OR NEW.number<>OLD.number THEN
 RAISE EXCEPTION 'immutable attempt identity'; END IF;
 RETURN NEW;
END $$;
DROP TRIGGER IF EXISTS cp_attempt_history_guard ON cp_attempts;
CREATE TRIGGER cp_attempt_history_guard BEFORE UPDATE OR DELETE ON cp_attempts
 FOR EACH ROW EXECUTE FUNCTION cp_attempt_history();
CREATE OR REPLACE FUNCTION cp_stage_identity() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF NEW.definition IS DISTINCT FROM OLD.definition OR NEW.operation<>OLD.operation THEN
 RAISE EXCEPTION 'immutable stage definition'; END IF;
 RETURN NEW;
END $$;
DROP TRIGGER IF EXISTS cp_stage_identity_guard ON cp_stages;
CREATE TRIGGER cp_stage_identity_guard BEFORE UPDATE ON cp_stages
 FOR EACH ROW EXECUTE FUNCTION cp_stage_identity();
"""


MIGRATION_3 = """
CREATE TABLE cp_projects(
 id text PRIMARY KEY, actor text NOT NULL, create_key text NOT NULL,
 request_digest text NOT NULL, name text NOT NULL, description text NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now(), UNIQUE(actor,create_key));
CREATE TABLE cp_assets(
 id text PRIMARY KEY, project_id text NOT NULL REFERENCES cp_projects(id),
 kind text NOT NULL, name text NOT NULL, version integer NOT NULL,
 parent_asset_id text REFERENCES cp_assets(id), artifact_id text NOT NULL REFERENCES cp_artifacts(id),
 create_key text NOT NULL, request_digest text NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now(), UNIQUE(project_id,create_key),
 UNIQUE(project_id,kind,name,version));
CREATE TABLE cp_project_runs(
 project_id text NOT NULL REFERENCES cp_projects(id), run_id text NOT NULL UNIQUE REFERENCES cp_runs(id),
 PRIMARY KEY(project_id,run_id));
CREATE TRIGGER cp_project_immutable BEFORE UPDATE OR DELETE ON cp_projects
 FOR EACH ROW EXECUTE FUNCTION cp_immutable_history();
CREATE TRIGGER cp_asset_immutable BEFORE UPDATE OR DELETE ON cp_assets
 FOR EACH ROW EXECUTE FUNCTION cp_immutable_history();
CREATE TRIGGER cp_project_run_immutable BEFORE UPDATE OR DELETE ON cp_project_runs
 FOR EACH ROW EXECUTE FUNCTION cp_immutable_history();
"""


def json_value(value: Any) -> Jsonb:
    from psycopg.types.json import Jsonb
    return Jsonb(value, dumps=lambda x: json.dumps(x, allow_nan=False))


def one(cursor: Cursor[dict[str, Any]]) -> dict[str, Any]:
    row = cursor.fetchone()
    if row is None:
        fail("database_result_error", "PostgreSQL did not return an expected record.", 503)
    return row


class Store:
    def __init__(self, dsn: str) -> None:
        if not isinstance(dsn, str) or not dsn.strip():
            fail("database_configuration", "A PostgreSQL connection string is required.", 503)
        if dsn.startswith(("sqlite:", "file:")):
            fail("database_configuration", "Only PostgreSQL is supported.", 503)
        self.dsn = dsn

    @contextmanager
    def transaction(self) -> Iterator[Connection[dict[str, Any]]]:
        try:
            import psycopg
            from psycopg.rows import dict_row
            with psycopg.connect(self.dsn, row_factory=dict_row, connect_timeout=5) as conn:
                yield conn
        except ControlPlaneError:
            raise
        except ImportError:
            fail("database_dependency", "PostgreSQL driver is not installed.", 503)
        except Exception as exc:
            # Never include DSNs, SQL, imported data, or provider output in public errors.
            if isinstance(exc, psycopg.Error):
                fail("database_error", "PostgreSQL operation failed.", 503)
            raise

    def migrate(self) -> None:
        with self.transaction() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(713982641)")
            exists = one(conn.execute("SELECT to_regclass('cp_schema') AS name"))["name"]
            if exists:
                versions = [r["version"] for r in conn.execute("SELECT version FROM cp_schema ORDER BY version")]
                if versions == [1, 2, 3]:
                    return
                if versions not in ([1], [1, 2]):
                    fail("schema_version", "Database schema version is unsupported.", 503)
            else:
                conn.execute(MIGRATION)
                conn.execute("INSERT INTO cp_schema(version) VALUES (1)")
                versions = [1]
            if versions == [1]:
                conn.execute(MIGRATION_2)
                conn.execute("INSERT INTO cp_schema(version) VALUES (2)")
            conn.execute(MIGRATION_3)
            conn.execute("INSERT INTO cp_schema(version) VALUES (3)")

    def health(self) -> dict[str, Any]:
        with self.transaction() as conn:
            exists = one(conn.execute("SELECT to_regclass('cp_schema') AS name"))["name"]
            if not exists:
                fail("schema_missing", "Database migrations must be applied.", 503)
            versions = [r["version"] for r in conn.execute("SELECT version FROM cp_schema ORDER BY version")]
            if versions != [1, 2, 3]:
                fail("schema_version", "Database schema version is unsupported.", 503)
        return {"status": "ok", "database": "postgresql", "schema_version": SCHEMA_VERSION}

    @staticmethod
    def event(conn: Connection[dict[str, Any]], run_id: str, kind: str,
              payload: dict[str, Any], stage_id: str | None = None) -> None:
        conn.execute("INSERT INTO cp_events(run_id,type,payload,stage_id) VALUES (%s,%s,%s,%s)",
                     (run_id, kind, json_value(payload), stage_id))

    @staticmethod
    def run(conn: Connection[dict[str, Any]], run_id: str, *, lock: bool = False) -> dict[str, Any]:
        row = conn.execute("SELECT * FROM cp_runs WHERE id=%s" + (" FOR UPDATE" if lock else ""),
                           (run_id,)).fetchone()
        if row is None:
            fail("run_not_found", "Run does not exist.", 404)
        return row
