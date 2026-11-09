"""Dashboard-neutral Gym Studio service contracts; no HTTP or release authority."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal
from uuid import uuid4

from .gym_job import (
    MAX_JOB_PARAMETERS_BYTES,
    GymComponentRegistry,
    GymJobError,
    GymJobSpec,
    gym_job_from_document,
    is_finite_number,
    validate_task_instance,
)
from .records import NativeError, canonicalize, content_digest

GYM_STUDIO_API_VERSION = 1
EvidenceClass = Literal["measured", "estimated", "synthetic", "caller_supplied", "unavailable"]
AttemptStatus = Literal["scored", "invalid", "failed", "incomplete"]
GateStatus = Literal["reported_pass", "reported_fail", "not_run", "blocked"]
AttemptExecutionKind = Literal["registered_candidate", "caller_supplied"]
MAX_PRACTICE_ATTEMPTS = 100_000
MAX_PRACTICE_BATCH = 128
MAX_STATE_BYTES = 256 * 1024 * 1024
CHUNK_BYTES = 512 * 1024


def _wire_document(document: Any) -> bytes:
    try:
        raw = json.dumps(document, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise GymJobError("state document must be bounded JSON") from exc
    if len(raw) > MAX_STATE_BYTES:
        raise GymJobError("state document exceeds its aggregate byte limit")
    return raw


def _chunk_document(document: Any) -> tuple[str, dict[str, bytes]]:
    chunks: dict[str, bytes] = {}

    def visit(value: Any) -> str:
        if len(_wire_document(value)) <= CHUNK_BYTES:
            node = {"kind": "value", "value": value}
        elif isinstance(value, Mapping):
            node = {"kind": "object", "fields": {key: visit(item) for key, item in value.items()}}
        elif isinstance(value, list):
            if len(value) == 1:
                node = {"kind": "array_items", "items": [visit(value[0])]}
            else:
                width = min(128, max(1, len(value) // 2))
                node = {"kind": "array", "chunks": [visit(value[index:index + width]) for index in range(0, len(value), width)]}
        else:
            raise GymJobError("a state scalar exceeds its chunk byte limit")
        raw = canonicalize(node)
        digest = str(content_digest(raw))
        chunks[digest] = raw
        return digest

    return visit(document), chunks


def _document_digest(document: Mapping[str, Any]) -> str:
    raw = _wire_document(document)
    if len(raw) <= 1_048_576:
        return str(content_digest(canonicalize(document)))
    root, _ = _chunk_document(document)
    return str(content_digest(canonicalize({"kind": "nf_gym_chunk_root_v1", "root": root})))


def _snapshot(document: Mapping[str, Any]) -> bytes:
    raw = _wire_document(document)
    return canonicalize(document) if len(raw) <= 1_048_576 else raw


def _digest(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise GymJobError(f"{field} must be a sha256 digest")
    return value


@dataclass(frozen=True)
class CandidateIdentity:
    candidate_id: str
    method_id: str
    base_model_digest: str | None
    candidate_model_digest: str
    artifact_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not isinstance(self.method_id, str) or not self.candidate_id.strip() or not self.method_id.strip():
            raise GymJobError("candidate_id and method_id must be non-empty")
        if self.base_model_digest is None:
            if self.method_id in {"unchanged_base", "lora", "qlora", "weight_edit", "wsp"}:
                raise GymJobError("candidate method requires a pinned base_model_digest")
        else:
            _digest(self.base_model_digest, "base_model_digest")
        _digest(self.candidate_model_digest, "candidate_model_digest")
        _digest(self.artifact_digest, "artifact_digest")

    def to_document(self) -> dict[str, str | None]:
        return {
            "candidate_id": self.candidate_id,
            "method_id": self.method_id,
            "base_model_digest": self.base_model_digest,
            "candidate_model_digest": self.candidate_model_digest,
            "artifact_digest": self.artifact_digest,
        }


@dataclass(frozen=True)
class ExecutionResponse:
    response: Any
    interaction: Mapping[str, Any]


@dataclass(frozen=True)
class PracticeAttemptRecord:
    attempt_id: str
    job_digest: str
    candidate_id: str
    scenario_id: str
    scenario_digest: str
    scenario_origin: str
    scenario_split: str
    round_number: int
    response_digest: str
    scorer_digest: str
    verifier_digest: str
    output_valid: bool
    verifier_passed: bool
    execution_kind: AttemptExecutionKind
    executor_digest: str | None
    status: AttemptStatus
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        for field in ("attempt_id", "candidate_id", "scenario_id"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field).strip():
                raise GymJobError(f"attempt.{field} must be non-empty")
        _digest(self.job_digest, "attempt.job_digest")
        _digest(self.scenario_digest, "attempt.scenario_digest")
        _digest(self.response_digest, "attempt.response_digest")
        _digest(self.scorer_digest, "attempt.scorer_digest")
        _digest(self.verifier_digest, "attempt.verifier_digest")
        if not isinstance(self.scenario_origin, str) or self.scenario_origin not in {"real", "simulated"}:
            raise GymJobError("attempt.scenario_origin is unsupported")
        if not isinstance(self.scenario_split, str) or self.scenario_split not in {"train", "validation"}:
            raise GymJobError("practice attempts cannot access the holdout split")
        if type(self.output_valid) is not bool or type(self.verifier_passed) is not bool:
            raise GymJobError("attempt validity fields must be booleans")
        if not isinstance(self.execution_kind, str) or self.execution_kind not in {"registered_candidate", "caller_supplied"}:
            raise GymJobError("attempt.execution_kind is unsupported")
        if self.execution_kind == "registered_candidate":
            _digest(self.executor_digest, "attempt.executor_digest")
        elif self.executor_digest is not None:
            raise GymJobError("caller-supplied attempts cannot claim a registered executor digest")
        if type(self.round_number) is not int or self.round_number < 0:
            raise GymJobError("attempt.round_number must be a non-negative integer")
        if not isinstance(self.status, str) or self.status not in {"scored", "invalid", "failed", "incomplete"}:
            raise GymJobError("attempt.status is unsupported")
        if (
            (self.status == "scored" and not self.output_valid)
            or (self.status == "invalid" and self.output_valid)
            or (self.verifier_passed and (not self.output_valid or self.status != "scored"))
            or (self.status in {"failed", "incomplete"} and (self.output_valid or self.verifier_passed))
        ):
            raise GymJobError("attempt status and validity fields are contradictory")
        if not isinstance(self.metrics, Mapping):
            raise GymJobError("attempt.metrics must be an object")
        if len(self.metrics) > 256:
            raise GymJobError("attempt.metrics exceeds its item limit")
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name or len(name.encode("utf-8")) > 256 or not is_finite_number(value):
                raise GymJobError("attempt metrics must have names and finite numeric values")
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))

    def to_document(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "job_digest": self.job_digest,
            "candidate_id": self.candidate_id,
            "scenario_id": self.scenario_id,
            "scenario_digest": self.scenario_digest,
            "scenario_origin": self.scenario_origin,
            "scenario_split": self.scenario_split,
            "round_number": self.round_number,
            "response_digest": self.response_digest,
            "scorer_digest": self.scorer_digest,
            "verifier_digest": self.verifier_digest,
            "output_valid": self.output_valid,
            "verifier_passed": self.verifier_passed,
            "execution_kind": self.execution_kind,
            "executor_digest": self.executor_digest,
            "status": self.status,
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class PracticeRunRecord:
    run_id: str
    job_digest: str
    scenario_manifest_digest: str
    practice_plan_digest: str
    candidate: CandidateIdentity
    expected_attempt_ids: tuple[str, ...]
    attempts: tuple[PracticeAttemptRecord, ...]
    status: Literal["complete", "incomplete", "failed"]
    run_digest: str

    @classmethod
    def create(
        cls,
        *,
        plan: Mapping[str, Any],
        attempts: Sequence[PracticeAttemptRecord],
    ) -> PracticeRunRecord:
        required_fields = {
            "schema_version", "kind", "run_id", "job_digest", "scenario_manifest_digest",
            "candidate", "expected_attempts", "scorer_digest", "verifier_digest", "plan_digest",
        }
        if "execution_policy" in plan:
            required_fields.add("execution_policy")
        if set(plan) != required_fields or plan.get("schema_version") != 1 or plan.get("kind") != "nf_gym_practice_plan_v1":
            raise GymJobError("practice plan has missing/unknown fields or unsupported version")
        if not isinstance(plan["run_id"], str) or not plan["run_id"].strip():
            raise GymJobError("practice plan run_id must be non-empty")
        _digest(plan["job_digest"], "practice plan job_digest")
        _digest(plan["scenario_manifest_digest"], "practice plan scenario_manifest_digest")
        _digest(plan["scorer_digest"], "practice plan scorer_digest")
        _digest(plan["verifier_digest"], "practice plan verifier_digest")
        expected_attempts = plan["expected_attempts"]
        if not isinstance(expected_attempts, list) or not expected_attempts or len(expected_attempts) > MAX_PRACTICE_ATTEMPTS:
            raise GymJobError("practice plan expected_attempts must be non-empty and bounded")
        expected_by_id: dict[str, Mapping[str, Any]] = {}
        for expected in expected_attempts:
            if not isinstance(expected, Mapping) or set(expected) != {
                "attempt_id", "scenario_id", "scenario_digest", "scenario_origin", "scenario_split", "repetition"
            }:
                raise GymJobError("practice plan has a malformed expected attempt")
            attempt_id = expected["attempt_id"]
            if not isinstance(attempt_id, str) or not attempt_id or attempt_id in expected_by_id:
                raise GymJobError("practice plan attempt IDs must be non-empty and unique")
            _digest(expected["scenario_digest"], "practice plan scenario_digest")
            if expected["scenario_origin"] not in {"real", "simulated"}:
                raise GymJobError("practice plan scenario_origin is unsupported")
            if expected["scenario_split"] not in {"train", "validation"}:
                raise GymJobError("practice plan cannot schedule holdout attempts")
            if type(expected["repetition"]) is not int or expected["repetition"] < 0:
                raise GymJobError("practice plan repetition must be a non-negative integer")
            expected_by_id[attempt_id] = expected
        _digest(plan["plan_digest"], "practice plan plan_digest")
        plan_payload = {key: value for key, value in plan.items() if key != "plan_digest"}
        if _document_digest(plan_payload) != plan["plan_digest"]:
            raise GymJobError("practice plan digest does not match its frozen contents")
        candidate_document = plan["candidate"]
        if not isinstance(candidate_document, Mapping):
            raise GymJobError("practice plan candidate identity is malformed")
        candidate = CandidateIdentity(**candidate_document)
        actual_by_id: dict[str, PracticeAttemptRecord] = {}
        for attempt in attempts:
            if attempt.attempt_id in actual_by_id or attempt.attempt_id not in expected_by_id:
                raise GymJobError("practice run contains duplicate or unplanned attempts")
            expected = expected_by_id[attempt.attempt_id]
            if (
                attempt.job_digest != plan["job_digest"]
                or attempt.candidate_id != candidate.candidate_id
                or attempt.scenario_id != expected["scenario_id"]
                or attempt.scenario_digest != expected["scenario_digest"]
                or attempt.scenario_origin != expected["scenario_origin"]
                or attempt.scenario_split != expected["scenario_split"]
                or attempt.round_number != expected["repetition"]
                or attempt.scorer_digest != plan["scorer_digest"]
                or attempt.verifier_digest != plan["verifier_digest"]
            ):
                raise GymJobError("practice attempt does not match its frozen run plan")
            actual_by_id[attempt.attempt_id] = attempt
        if len(actual_by_id) != len(attempts):
            raise GymJobError("practice attempts contain duplicate IDs")
        ordered_attempts = tuple(actual_by_id[key] for key in expected_by_id if key in actual_by_id)
        status: Literal["complete", "incomplete", "failed"] = (
            "incomplete" if set(actual_by_id) != set(expected_by_id)
            else "failed" if any(attempt.status == "failed" for attempt in ordered_attempts)
            else "incomplete" if any(attempt.status == "incomplete" for attempt in ordered_attempts)
            else "complete"
        )
        payload = {
            "schema_version": 1,
            "kind": "nf_gym_practice_run_v1",
            "run_id": plan["run_id"],
            "job_digest": plan["job_digest"],
            "scenario_manifest_digest": plan["scenario_manifest_digest"],
            "practice_plan_digest": plan["plan_digest"],
            "candidate": candidate.to_document(),
            "expected_attempt_ids": list(expected_by_id),
            "attempts": [attempt.to_document() for attempt in ordered_attempts],
            "status": status,
        }
        run_digest = _document_digest(payload)
        return cls(
            run_id=plan["run_id"],
            job_digest=plan["job_digest"],
            scenario_manifest_digest=plan["scenario_manifest_digest"],
            practice_plan_digest=plan["plan_digest"],
            candidate=candidate,
            expected_attempt_ids=tuple(expected_by_id),
            attempts=ordered_attempts,
            status=status,
            run_digest=run_digest,
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "nf_gym_practice_run_v1",
            "run_id": self.run_id,
            "job_digest": self.job_digest,
            "scenario_manifest_digest": self.scenario_manifest_digest,
            "practice_plan_digest": self.practice_plan_digest,
            "candidate": self.candidate.to_document(),
            "expected_attempt_ids": list(self.expected_attempt_ids),
            "attempts": [attempt.to_document() for attempt in self.attempts],
            "status": self.status,
            "run_digest": self.run_digest,
        }


class GymStudioService:
    """Versioned in-process API surface that a future dashboard can call over transport."""

    def __init__(
        self,
        components: GymComponentRegistry | None = None,
        *,
        state_dir: str | Path | None = None,
        split_authority_id: str = "nf_local_dataset_authority_v1",
    ) -> None:
        self.components = components or GymComponentRegistry()
        if not isinstance(split_authority_id, str) or not split_authority_id.strip():
            raise GymJobError("split authority must be configured by the service")
        self.split_authority_id = split_authority_id
        self.state_dir: Path | None = None
        if state_dir is not None:
            requested = Path(state_dir).absolute()
            if requested.exists() and requested.is_symlink():
                raise GymJobError("Gym Studio state directory must not be a symlink")
            requested.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(requested, 0o700)
            self.state_dir = requested.resolve()
        self._scenario_batches: dict[str, bytes] = {}
        self._scenario_indexes: dict[str, tuple[bytes, dict[str, bytes], str]] = {}
        self._split_scopes: dict[str, str] = {}
        self._split_assignments: dict[tuple[str, str], str] = {}
        self._jobs: dict[str, GymJobSpec] = {}
        self._scenario_references: dict[tuple[str, str], Any] = {}
        self._practice_plans: dict[str, dict[str, Any]] = {}
        self._expected_attempts: dict[str, dict[str, Mapping[str, Any]]] = {}
        self._claimed_attempts: set[tuple[str, str]] = set()
        self._terminal_attempts: dict[str, dict[str, PracticeAttemptRecord]] = {}
        self._candidate_executors: dict[
            tuple[str, str, str], tuple[str, Callable[..., Any]]
        ] = {}
        self._identity_probes: dict[tuple[str, str, str], Callable[[], Mapping[str, Any]]] = {}
        self._worker_identities: dict[str, dict[str, Any]] = {}
        self._executor_configs: dict[tuple[str, str, str], str] = {}
        self._response_artifacts: dict[tuple[str, str], dict[str, Any]] = {}
        self._attempt_db_path = self.state_dir / "attempts.sqlite3" if self.state_dir else None
        if self._attempt_db_path is not None:
            if self._attempt_db_path.is_symlink():
                raise GymJobError("attempt database must not be a symlink")
            descriptor = os.open(self._attempt_db_path, os.O_CREAT | os.O_WRONLY, 0o600)
            os.close(descriptor)
            os.chmod(self._attempt_db_path, 0o600)
            connection = sqlite3.connect(self._attempt_db_path, timeout=30)
            try:
                with connection:
                    connection.execute(
                        "CREATE TABLE IF NOT EXISTS terminal_attempts ("
                        "plan_digest TEXT NOT NULL, attempt_id TEXT NOT NULL, record BLOB NOT NULL, "
                        "PRIMARY KEY (plan_digest, attempt_id))"
                    )
                    connection.execute(
                        "CREATE TABLE IF NOT EXISTS attempt_migrations (plan_digest TEXT PRIMARY KEY)"
                    )
                    connection.execute(
                        "CREATE TABLE IF NOT EXISTS attempt_claims (plan_digest TEXT, attempt_id TEXT, owner TEXT, "
                        "PRIMARY KEY(plan_digest, attempt_id))"
                    )
                    connection.execute(
                        "CREATE TABLE IF NOT EXISTS infrastructure_errors (plan_digest TEXT, attempt_id TEXT, "
                        "error_digest TEXT, retry_count INTEGER, PRIMARY KEY(plan_digest, attempt_id))"
                    )
                    connection.execute(
                        "CREATE TABLE IF NOT EXISTS response_artifacts (plan_digest TEXT, attempt_id TEXT, "
                        "artifact_digest TEXT NOT NULL, PRIMARY KEY(plan_digest, attempt_id))"
                    )
                    connection.execute(
                        "CREATE TABLE IF NOT EXISTS scoring_errors (plan_digest TEXT, attempt_id TEXT, error_digest TEXT, "
                        "retry_count INTEGER, PRIMARY KEY(plan_digest, attempt_id))"
                    )
                    connection.execute("CREATE TABLE IF NOT EXISTS frozen_run_bindings (run_id TEXT PRIMARY KEY, plan_digest TEXT)")
                    connection.execute(
                        "CREATE TABLE IF NOT EXISTS split_assignments (namespace TEXT, identity TEXT, split TEXT, PRIMARY KEY(namespace, identity))"
                    )
                    connection.execute("CREATE TABLE IF NOT EXISTS split_scopes (job_id TEXT PRIMARY KEY, authority TEXT NOT NULL)")
                    connection.execute("CREATE TABLE IF NOT EXISTS split_authority (singleton INTEGER PRIMARY KEY CHECK(singleton=1), authority TEXT NOT NULL)")
                    connection.execute("INSERT OR IGNORE INTO split_authority VALUES (1, ?)", (self.split_authority_id,))
                    authority = connection.execute("SELECT authority FROM split_authority WHERE singleton=1").fetchone()[0]
                    if authority != self.split_authority_id:
                        raise GymJobError("split authority is pinned to the durable state store")
            finally:
                connection.close()

    def _state_path(self, namespace: str, identity: str) -> Path | None:
        if self.state_dir is None:
            return None
        if not identity or any(character not in "0123456789abcdef" for character in identity):
            raise GymJobError("state identity must be lowercase hex")
        directory = self.state_dir / namespace
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(directory, 0o700)
        return directory / f"{identity}.json"

    def _persist_state(self, namespace: str, identity: str, document: Mapping[str, Any]) -> None:
        path = self._state_path(namespace, identity)
        if path is None:
            return
        raw = _snapshot(document)
        if len(raw) > 1_048_576:
            root, chunks = _chunk_document(document)
            for digest, chunk in chunks.items():
                chunk_path = self._state_path("chunks", digest[7:])
                if chunk_path is None:
                    raise GymJobError("chunk store is unavailable")
                if not chunk_path.exists():
                    descriptor, name = tempfile.mkstemp(prefix=".nf-chunk-", dir=chunk_path.parent)
                    with os.fdopen(descriptor, "wb") as stream:
                        stream.write(chunk)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(name, chunk_path)
            raw = canonicalize({"kind": "nf_gym_chunk_store_v1", "root": root})
        descriptor, temporary_name = tempfile.mkstemp(prefix=".nf-gym-state-", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)

    def _load_state(self, namespace: str, identity: str) -> dict[str, Any] | None:
        path = self._state_path(namespace, identity)
        if path is None or not path.is_file():
            return None
        try:
            document = json.loads(path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GymJobError(f"Gym Studio {namespace} state is unreadable: {exc}") from exc
        if not isinstance(document, dict):
            raise GymJobError(f"Gym Studio {namespace} state must be a JSON object")
        if document.get("kind") == "nf_gym_chunk_store_v1":
            restored_bytes = 0

            def restore(digest: str, depth: int = 0) -> Any:
                nonlocal restored_bytes
                if depth > 32 or not _valid_digest(digest):
                    raise GymJobError("invalid chunk reference or depth")
                target = self._state_path("chunks", digest[7:])
                if target is None or not target.is_file():
                    raise GymJobError("state chunk is missing")
                raw = target.read_bytes()
                restored_bytes += len(raw)
                if restored_bytes > MAX_STATE_BYTES:
                    raise GymJobError("restored state exceeds its aggregate chunk byte limit")
                if len(raw) > 1_048_576 or str(content_digest(raw)) != digest:
                    raise GymJobError("state chunk failed digest verification")
                node = json.loads(raw)
                if node["kind"] == "value":
                    return node["value"]
                if node["kind"] == "object":
                    return {key: restore(value, depth + 1) for key, value in node["fields"].items()}
                if node["kind"] == "array":
                    return [item for ref in node["chunks"] for item in restore(ref, depth + 1)]
                if node["kind"] == "array_items":
                    return [restore(ref, depth + 1) for ref in node["items"]]
                raise GymJobError("state chunk has an unsupported kind")

            document = restore(document["root"])
            _wire_document(document)
        return document

    def _load_terminal_attempts(self, plan_digest: str) -> dict[str, PracticeAttemptRecord]:
        if self._attempt_db_path is None:
            return dict(self._terminal_attempts.get(plan_digest, {}))
        connection = sqlite3.connect(self._attempt_db_path, timeout=30)
        try:
            with connection:
                if connection.execute(
                    "SELECT 1 FROM attempt_migrations WHERE plan_digest = ?", (plan_digest,)
                ).fetchone() is None:
                    saved = self._load_state("attempts", plan_digest[7:])
                    if saved is not None:
                        rows = saved.get("attempts")
                        if set(saved) != {"attempts"} or not isinstance(rows, list):
                            raise GymJobError("persisted terminal-attempt state is malformed")
                        attempts = tuple(PracticeAttemptRecord(**{
                            **row, "execution_kind": "caller_supplied", "executor_digest": None,
                        }) for row in rows)
                        PracticeRunRecord.create(plan=self._practice_plans[plan_digest], attempts=attempts)
                        for attempt in attempts:
                            self._write_terminal_attempt(connection, plan_digest, attempt)
                    connection.execute(
                        "INSERT OR IGNORE INTO attempt_migrations VALUES (?)", (plan_digest,)
                    )
                records = connection.execute(
                    "SELECT record FROM terminal_attempts WHERE plan_digest = ? ORDER BY attempt_id",
                    (plan_digest,),
                ).fetchall()
            attempts = tuple(PracticeAttemptRecord(**json.loads(row[0])) for row in records)
            PracticeRunRecord.create(plan=self._practice_plans[plan_digest], attempts=attempts)
            return {attempt.attempt_id: attempt for attempt in attempts}
        finally:
            connection.close()

    def _write_terminal_attempt(
        self, connection: sqlite3.Connection, plan_digest: str, attempt: PracticeAttemptRecord
    ) -> None:
        raw = canonicalize(attempt.to_document())
        connection.execute(
            "INSERT OR IGNORE INTO terminal_attempts VALUES (?, ?, ?)",
            (plan_digest, attempt.attempt_id, raw),
        )
        existing = connection.execute(
            "SELECT record FROM terminal_attempts WHERE plan_digest = ? AND attempt_id = ?",
            (plan_digest, attempt.attempt_id),
        ).fetchone()
        if existing is None or existing[0] != raw:
            raise GymJobError("submitted attempt differs from its terminal scored result")

    def _persist_terminal_attempt(self, plan_digest: str, attempt: PracticeAttemptRecord) -> None:
        self._validate_attempt(plan_digest, attempt)
        if self._attempt_db_path is not None:
            connection = sqlite3.connect(self._attempt_db_path, timeout=30)
            try:
                with connection:
                    self._write_terminal_attempt(connection, plan_digest, attempt)
            finally:
                connection.close()
        else:
            existing = self._terminal_attempts.get(plan_digest, {}).get(attempt.attempt_id)
            if existing is not None and existing != attempt:
                raise GymJobError("submitted attempt differs from its terminal scored result")
        self._terminal_attempts.setdefault(plan_digest, {})[attempt.attempt_id] = attempt

    def _validate_attempt(self, plan_digest: str, attempt: PracticeAttemptRecord) -> None:
        plan = self._practice_plans[plan_digest]
        if plan_digest not in self._expected_attempts:
            self._expected_attempts[plan_digest] = {row["attempt_id"]: row for row in plan["expected_attempts"]}
        expected = self._expected_attempts[plan_digest].get(attempt.attempt_id)
        if expected is None:
            raise GymJobError("practice run contains unplanned attempts")
        fields = {
            "job_digest": plan["job_digest"], "candidate_id": plan["candidate"]["candidate_id"],
            "scorer_digest": plan["scorer_digest"], "verifier_digest": plan["verifier_digest"],
            "round_number": expected["repetition"],
            **{key: expected[key] for key in ("scenario_id", "scenario_digest", "scenario_origin", "scenario_split")},
        }
        if any(getattr(attempt, key) != value for key, value in fields.items()):
            raise GymJobError("practice attempt does not match its frozen run plan")
        self._validate_execution_policy(plan, attempt.execution_kind, attempt.executor_digest)

    def _validate_execution_policy(self, plan: Mapping[str, Any], kind: str, executor_digest: str | None) -> None:
        policy = plan.get("execution_policy")
        if policy is not None and (policy["kind"] != kind or policy["executor_digest"] != executor_digest):
            raise GymJobError("attempt execution type or executor differs from the frozen plan")

    def _lookup_terminal(self, plan_digest: str, attempt_id: str) -> PracticeAttemptRecord | None:
        if self._attempt_db_path is None:
            return self._terminal_attempts.get(plan_digest, {}).get(attempt_id)
        connection = sqlite3.connect(self._attempt_db_path, timeout=30)
        try:
            row = connection.execute(
                "SELECT record FROM terminal_attempts WHERE plan_digest = ? AND attempt_id = ?",
                (plan_digest, attempt_id),
            ).fetchone()
            if row is None:
                return None
            attempt = PracticeAttemptRecord(**json.loads(row[0]))
            self._validate_attempt(plan_digest, attempt)
            return attempt
        finally:
            connection.close()

    @contextmanager
    def _claim_batch(self, plan_digest: str, attempt_ids: Sequence[str]) -> Iterator[None]:
        if not attempt_ids or len(set(attempt_ids)) != len(attempt_ids):
            raise GymJobError("attempt claims must be non-empty and unique")
        descriptor = None
        lease_path = None
        claimed: list[tuple[str, str]] = []
        owner = uuid4().hex
        connection = sqlite3.connect(self._attempt_db_path, timeout=30) if self._attempt_db_path else None
        try:
            if self.state_dir is not None:
                if os.name == "nt":
                    raise GymJobError("process-owned practice claims require WSL/POSIX")
                import fcntl

                lease_path = self._state_path("leases", owner)
                descriptor = os.open(lease_path, os.O_CREAT | os.O_RDWR, 0o600)
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for attempt_id in sorted(attempt_ids):
                key = (plan_digest, attempt_id)
                if key in self._claimed_attempts:
                    raise GymJobError("practice attempt is already claimed by an active worker")
                self._claimed_attempts.add(key)
                claimed.append(key)
            if connection is not None:
                connection.execute("BEGIN IMMEDIATE")
                for attempt_id in attempt_ids:
                    previous = connection.execute(
                        "SELECT owner FROM attempt_claims WHERE plan_digest=? AND attempt_id=?", (plan_digest, attempt_id)
                    ).fetchone()
                    if previous is not None and self._claim_owner_active(previous[0]):
                        raise GymJobError("practice attempt is already claimed by an active worker")
                    if connection.execute(
                        "SELECT 1 FROM terminal_attempts WHERE plan_digest=? AND attempt_id=?", (plan_digest, attempt_id)
                    ).fetchone():
                        raise GymJobError("practice attempt already reached a terminal state")
                    connection.execute(
                        "INSERT OR REPLACE INTO attempt_claims VALUES (?, ?, ?)", (plan_digest, attempt_id, owner)
                    )
                connection.commit()
            yield
        finally:
            if connection is not None:
                connection.rollback()
                with connection:
                    connection.execute("DELETE FROM attempt_claims WHERE owner=?", (owner,))
                connection.close()
            for key in claimed:
                self._claimed_attempts.discard(key)
            if descriptor is not None:
                os.close(descriptor)
            if lease_path is not None:
                lease_path.unlink(missing_ok=True)

    def _claim_owner_active(self, owner: str) -> bool:
        import fcntl

        path = self._state_path("leases", owner)
        if path is None or not path.exists():
            return False
        descriptor = os.open(path, os.O_RDWR)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return True
            path.unlink(missing_ok=True)
            return False
        finally:
            os.close(descriptor)

    def _freeze_split_assignments(self, spec: GymJobSpec, scenarios: Sequence[Mapping[str, Any]]) -> None:
        namespace = self.split_authority_id
        self._split_scopes[spec.job_id] = namespace
        identities = [(f"content:{row['document_identity']}", row["split"]) for row in scenarios]
        identities += [(f"group:{row['group_identity']}", row["split"]) for row in scenarios]
        identities += [(f"input:{row.get('input_identity', row['document_identity'])}", row["split"]) for row in scenarios]
        if self._attempt_db_path is None:
            if any((namespace, identity) in self._split_assignments and self._split_assignments[(namespace, identity)] != split for identity, split in identities):
                raise GymJobError("document or contract-family group overlaps frozen splits")
            self._split_assignments.update({(namespace, identity): split for identity, split in identities})
            return
        connection = sqlite3.connect(self._attempt_db_path, timeout=30)
        try:
            with connection:
                for identity, split in identities:
                    connection.execute("INSERT OR IGNORE INTO split_assignments VALUES (?, ?, ?)", (namespace, identity, split))
                    previous = connection.execute("SELECT split FROM split_assignments WHERE namespace=? AND identity=?", (namespace, identity)).fetchone()
                    if previous is None or previous[0] != split:
                        raise GymJobError("document or contract-family group overlaps frozen splits")
        finally:
            connection.close()

    def _record_infrastructure_error(self, plan_digest: str, attempt_ids: Sequence[str], error: Exception) -> None:
        if self._attempt_db_path is None:
            return
        connection = sqlite3.connect(self._attempt_db_path, timeout=30)
        digest = _document_digest({"kind": "nf_retryable_executor_error_v1", "error_type": type(error).__name__})
        try:
            with connection:
                for attempt_id in attempt_ids:
                    connection.execute(
                        "INSERT INTO infrastructure_errors VALUES (?, ?, ?, 1) "
                        "ON CONFLICT(plan_digest, attempt_id) DO UPDATE SET error_digest=excluded.error_digest, "
                        "retry_count=retry_count+1", (plan_digest, attempt_id, digest),
                    )
        finally:
            connection.close()

    def _store_response(
        self, plan_digest: str, attempt_id: str, response: Any,
        execution_kind: AttemptExecutionKind, executor_digest: str | None,
        interaction: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        plan = self._load_frozen_plan(plan_digest)
        if plan is None:
            raise GymJobError("response requires a frozen plan")
        self._validate_execution_policy(plan, execution_kind, executor_digest)
        raw_json = None
        try:
            raw_json = _wire_document(response).decode("utf-8")
            detached = json.loads(raw_json)
            _document_digest({"response": detached})
            encodable = True
        except (GymJobError, NativeError):
            detached = {"unencodable_response_type": type(response).__name__, "original_json": raw_json}
            encodable = False
        document = {
            "kind": "nf_gym_private_response_v1", "practice_plan_digest": plan_digest,
            "attempt_id": attempt_id, "response": detached, "json_encodable": encodable,
            "execution_kind": execution_kind, "executor_digest": executor_digest,
        }
        if interaction is not None:
            document["interaction"] = json.loads(_wire_document(interaction))
        digest = _document_digest(document)
        self._persist_state("responses", digest[7:], document)
        if self._attempt_db_path is not None:
            connection = sqlite3.connect(self._attempt_db_path, timeout=30)
            try:
                with connection:
                    connection.execute("INSERT OR IGNORE INTO response_artifacts VALUES (?, ?, ?)", (plan_digest, attempt_id, digest))
                    existing = connection.execute(
                        "SELECT artifact_digest FROM response_artifacts WHERE plan_digest=? AND attempt_id=?",
                        (plan_digest, attempt_id),
                    ).fetchone()
                    if existing is None or existing[0] != digest:
                        raise GymJobError("response differs from the frozen private response artifact")
            finally:
                connection.close()
        previous = self._response_artifacts.get((plan_digest, attempt_id))
        if previous is not None and previous != document:
            raise GymJobError("response differs from the frozen private response artifact")
        self._response_artifacts[(plan_digest, attempt_id)] = document
        return document

    def _get_response(self, plan_digest: str, attempt_id: str) -> dict[str, Any] | None:
        document = self._response_artifacts.get((plan_digest, attempt_id))
        if document is not None:
            return json.loads(_snapshot(document))
        if self._attempt_db_path is None:
            return None
        connection = sqlite3.connect(self._attempt_db_path, timeout=30)
        try:
            row = connection.execute(
                "SELECT artifact_digest FROM response_artifacts WHERE plan_digest=? AND attempt_id=?", (plan_digest, attempt_id)
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        document = self._load_state("responses", row[0][7:])
        if document is None or _document_digest(document) != row[0] or document.get("practice_plan_digest") != plan_digest or document.get("attempt_id") != attempt_id:
            raise GymJobError("private response artifact failed integrity verification")
        return document

    def retry_scoring(
        self, spec: GymJobSpec, candidate: CandidateIdentity, *, practice_plan_digest: str,
        attempt_ids: Sequence[str],
    ) -> tuple[PracticeAttemptRecord, ...]:
        results = []
        for attempt_id in attempt_ids:
            self._get_frozen_plan(spec, candidate, practice_plan_digest, attempt_id)
        with self._claim_batch(practice_plan_digest, attempt_ids):
            for attempt_id in attempt_ids:
                document = self._get_response(practice_plan_digest, attempt_id)
                if document is None:
                    raise GymJobError("scoring retry requires a saved response artifact")
                try:
                    results.append(self._score_attempt_response(
                        spec, candidate, practice_plan_digest=practice_plan_digest, attempt_id=attempt_id,
                        response=document["response"], execution_kind=document["execution_kind"],
                        executor_digest=document["executor_digest"], json_encodable=document["json_encodable"],
                    ))
                except Exception as exc:  # noqa: BLE001 -- scorer failures cannot discard other responses.
                    self._record_scoring_error(practice_plan_digest, attempt_id, exc)
        return tuple(results)

    def _record_scoring_error(self, plan_digest: str, attempt_id: str, error: Exception) -> None:
        if self._attempt_db_path is None:
            return
        connection = sqlite3.connect(self._attempt_db_path, timeout=30)
        try:
            with connection:
                connection.execute(
                    "INSERT INTO scoring_errors VALUES (?, ?, ?, 1) ON CONFLICT(plan_digest, attempt_id) "
                    "DO UPDATE SET error_digest=excluded.error_digest, retry_count=retry_count+1",
                    (plan_digest, attempt_id, _document_digest({"kind": "nf_scoring_error_v1", "type": type(error).__name__})),
                )
        finally:
            connection.close()

    def _load_frozen_plan(self, plan_digest: str) -> dict[str, Any] | None:
        if not _valid_digest(plan_digest):
            raise GymJobError("practice_plan_digest must be a sha256 digest")
        plan = self._practice_plans.get(plan_digest)
        if plan is None:
            plan = self._load_state("plans", plan_digest[7:])
            if plan is None:
                return None
            PracticeRunRecord.create(plan=plan, attempts=())
            self._practice_plans[plan_digest] = plan
        if plan_digest not in self._expected_attempts:
            self._expected_attempts[plan_digest] = {row["attempt_id"]: row for row in plan["expected_attempts"]}
        return plan

    def describe(self) -> dict[str, Any]:
        return {
            "schema_version": GYM_STUDIO_API_VERSION,
            "kind": "nf_gym_studio_api_v1",
            "operations": [
                "capabilities", "validate_job", "materialize_scenarios", "freeze_practice_plan",
                "register_candidate_executor", "compare_unchanged_bases", "run_practice_batch",
                "run_practice_attempt", "score_attempt",
                "record_practice_run", "score_candidate",
            ],
            "components": self.components.list_components(),
            "authority": {
                "executes_registered_trusted_adapters": True,
                "executes_untrusted_code": False,
                "sandbox_isolation_verified": False,
                "sealed_target_custody": False,
                "promotion_authorized": False,
                "native_candidate_ledger": False,
                "metric_inputs_authenticated": False,
                "candidate_artifact_verified": False,
                "served_model_identity_verified": False,
                "durable_public_practice_state": self.state_dir is not None,
                "max_practice_batch_attempts": MAX_PRACTICE_BATCH,
            },
        }

    def validate_job(self, document: Mapping[str, Any]) -> dict[str, Any]:
        spec = gym_job_from_document(document)
        self.components.validate_job(spec)
        return {
            "schema_version": GYM_STUDIO_API_VERSION,
            "kind": "nf_gym_job_validation_v1",
            "job_id": spec.job_id,
            "job_digest": spec.digest,
            "job": spec.to_document(),
            "valid": True,
            "execution_authorized": False,
            "release_authorized": False,
        }

    def materialize_public_scenarios(
        self,
        spec: GymJobSpec,
        *,
        split: str,
        seed: int,
        max_scenarios: int = 100_000,
    ) -> dict[str, Any]:
        """Materialize registered provider output for public splits only; holdout stays behind native custody."""
        self.components.validate_job(spec)
        if split not in {"train", "validation"}:
            raise GymJobError("holdout materialization requires a native independent-evaluation custodian")
        self._jobs[spec.digest] = spec
        self._persist_state("jobs", spec.digest[7:], spec.to_document())
        if type(seed) is not int or seed < 0 or type(max_scenarios) is not int or max_scenarios < 1:
            raise GymJobError("scenario seed and count limit must be bounded non-negative integers")
        scenarios: list[dict[str, Any]] = []
        private_references: dict[str, Any] = {}
        seen_ids: set[str] = set()
        source_identities: list[dict[str, str]] = []
        for source in spec.scenario_sources:
            if split not in source.split_roles:
                continue
            provider = self.components.resolve(
                "scenario_provider", source.provider_id, source.provider_version, source.provider_digest
            )
            rows = provider(
                parameters=source.to_document()["parameters"],
                source_id=source.source_id,
                source_digest=source.source_digest,
                origin=source.origin,
                split=split,
                seed=seed,
            )
            if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
                raise GymJobError(f"scenario provider {source.source_id!r} must return a bounded sequence")
            if len(rows) > max_scenarios - len(scenarios):
                raise GymJobError("scenario materialization exceeds its count limit")
            source_identities.append({
                "source_id": source.source_id,
                "provider_digest": source.provider_digest,
                "source_digest": source.source_digest,
            })
            for index, row in enumerate(rows):
                required_row = {"scenario_id", "input", "reference"}
                if not isinstance(row, Mapping) or not required_row <= set(row) or set(row) - required_row - {"group_id", "source_group_id", "item_id", "family_id", "interaction_type"}:
                    raise GymJobError(f"scenario provider {source.source_id!r} returned malformed row {index}")
                scenario_id = row["scenario_id"]
                if not isinstance(scenario_id, str) or not scenario_id.strip() or scenario_id in seen_ids:
                    raise GymJobError("scenario IDs must be non-empty and unique across sources")
                validate_task_instance(spec.task_input_schema, row["input"], field=f"scenario[{scenario_id}].input")
                try:
                    reference_bytes = canonicalize({"kind": "nf_gym_reference_v1", "reference": row["reference"]})
                except (NativeError, TypeError, ValueError, OverflowError, RecursionError) as exc:
                    raise GymJobError("scenario reference must be bounded JSON data") from exc
                if len(reference_bytes) > MAX_JOB_PARAMETERS_BYTES:
                    raise GymJobError(f"scenario {scenario_id!r} verifier reference exceeds its byte limit")
                reference_digest = str(content_digest(reference_bytes))
                private_references[scenario_id] = json.loads(reference_bytes)["reference"]
                seen_ids.add(scenario_id)
                input_identity = _document_digest({"kind": "nf_gym_input_v2", "input": row["input"]})
                item = row.get("item_id", input_identity)
                if not isinstance(item, str) or not item or len(item.encode()) > 512:
                    raise GymJobError("task item identity must be a bounded nonempty string")
                document_identity = _document_digest({"kind": "nf_gym_item_v2", "item": item})
                group = row.get("source_group_id", row.get("family_id", row.get("group_id", document_identity)))
                if not isinstance(group, str) or not group or len(group.encode()) > 512:
                    raise GymJobError("scenario group identity must be a bounded nonempty string")
                if source.parameters.get("variant_of") and "source_group_id" not in row:
                    raise GymJobError("synthetic variants must declare their source contract group")
                scenarios.append({
                    "scenario_id": scenario_id,
                    "source_id": source.source_id,
                    "source_digest": source.source_digest,
                    "origin": source.origin,
                    "split": split,
                    "input": json.loads(_wire_document(row["input"])),
                    "reference_digest": reference_digest,
                    "reference_encoding": "nf_gym_reference_v1",
                    "document_identity": document_identity,
                    "input_identity": input_identity,
                    "item_id": item,
                    "identity_declared": "item_id" in row,
                    "interaction_type": row.get("interaction_type", "single_response"),
                    "group_identity": _document_digest({"kind": "nf_gym_group_v1", "group": group}),
                    "group_declared": any(field in row for field in ("family_id", "group_id", "source_group_id")),
                })
        if not scenarios:
            raise GymJobError(f"no registered providers produced scenarios for split {split!r}")
        self._freeze_split_assignments(spec, scenarios)
        payload = {
            "schema_version": 1,
            "kind": "nf_gym_scenario_manifest_v1",
            "job_digest": spec.digest,
            "split": split,
            "seed": seed,
            "sources": source_identities,
            "scenarios": scenarios,
        }
        manifest_digest = _document_digest(payload)
        bundle = {**payload, "manifest_digest": manifest_digest}
        snapshot = _snapshot(bundle)
        self._scenario_batches[manifest_digest] = snapshot
        self._scenario_indexes.pop(manifest_digest, None)
        for scenario_id, reference in private_references.items():
            self._scenario_references[(manifest_digest, scenario_id)] = reference
        self._persist_state(
            "scenarios",
            manifest_digest.removeprefix("sha256:"),
            {"batch": bundle, "references": private_references},
        )
        return json.loads(snapshot)

    def freeze_training_dataset(
        self, spec: GymJobSpec, selections: Sequence[Mapping[str, str]], *, dataset_id: str,
        allow_reported_responses: bool = False,
        source_practice_run_digest: str | None = None,
        excluded_attempt_count: int = 0,
    ) -> dict[str, Any]:
        if self.state_dir is None or not selections or len(selections) > MAX_PRACTICE_ATTEMPTS:
            raise GymJobError("training datasets require durable state and a bounded nonempty selection")
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise GymJobError("training dataset ID is required")
        if source_practice_run_digest is not None:
            _digest(source_practice_run_digest, "source_practice_run_digest")
        if type(excluded_attempt_count) is not int or excluded_attempt_count < 0:
            raise GymJobError("excluded_attempt_count must be a non-negative integer")
        rows = []
        seen = set()
        for selection in selections:
            if not isinstance(selection, Mapping) or set(selection) != {"practice_plan_digest", "attempt_id"}:
                raise GymJobError("dataset selection requires a frozen plan and attempt ID")
            key = (selection["practice_plan_digest"], selection["attempt_id"])
            if key in seen:
                raise GymJobError("training dataset selections must be unique")
            seen.add(key)
            plan = self._load_frozen_plan(key[0])
            if plan is None or plan["job_digest"] != spec.digest:
                raise GymJobError("training selection does not belong to this frozen job")
            attempt = self._lookup_terminal(*key)
            response = self._get_response(*key)
            if attempt is None or response is None or attempt.status != "scored" or not attempt.output_valid or not attempt.verifier_passed:
                raise GymJobError("training selection requires scored, valid, verifier-passed response artifacts")
            if attempt.execution_kind != "registered_candidate" and not allow_reported_responses:
                raise GymJobError("reported responses require explicit unverified dataset opt-in")
            if attempt.scenario_split != "train":
                raise GymJobError("validation and holdout responses cannot become training examples")
            scenario = self._get_scenario(plan["scenario_manifest_digest"], attempt.scenario_id, spec.digest)
            if scenario is None:
                raise GymJobError("training selection scenario manifest is missing")
            if not scenario.get("group_declared"):
                raise GymJobError("training examples require explicit document/family grouping")
            self._freeze_split_assignments(spec, (scenario,))
            raw_response = response["response"]
            example = {
                "kind": "nf_gym_training_example_v2", "input": scenario["input"], "response": raw_response,
                "interaction_type": scenario.get("interaction_type", "single_response"),
                "interaction": response.get("interaction", {"input": scenario["input"], "response": raw_response,
                    "inference_config_digest": plan["execution_policy"]["inference_config_digest"], "recording": "legacy_or_reported"}),
                "source_split": "train", "scenario_manifest_digest": plan["scenario_manifest_digest"],
                "scenario_id": attempt.scenario_id, "scenario_digest": attempt.scenario_digest,
                "document_identity": scenario["document_identity"], "group_identity": scenario["group_identity"],
                "practice_plan_digest": key[0], "attempt_id": key[1],
                "response_artifact_digest": _document_digest(response), "response_digest": attempt.response_digest,
                "scorer": spec.scorer.to_document(), "verifier": spec.verifier.to_document(),
                "execution_kind": attempt.execution_kind,
            }
            digest = _document_digest(example)
            rows.append({"example_digest": digest, "group_identity": scenario["group_identity"], "document_identity": scenario["document_identity"]})
            self._persist_state("training_examples", digest[7:], example)
        manifest = {
            "kind": "nf_gym_training_dataset_v1", "dataset_id": dataset_id, "job_digest": spec.digest,
            "split_namespace": self._split_scopes[spec.job_id],
            "format": "nf_gym_structured_interaction_v2", "examples": rows,
            "allow_reported_responses": allow_reported_responses, "source_split": "train",
            "source_practice_run_digest": source_practice_run_digest,
            "excluded_attempt_count": excluded_attempt_count,
            "release_authorized": False,
        }
        manifest["dataset_digest"] = _document_digest(manifest)
        self._persist_state("datasets", manifest["dataset_digest"][7:], manifest)
        return manifest

    def load_training_examples(self, dataset_digest: str) -> tuple[dict[str, Any], ...]:
        _digest(dataset_digest, "dataset_digest")
        manifest = self._load_state("datasets", dataset_digest[7:])
        if manifest is None or manifest.get("dataset_digest") != dataset_digest or _document_digest({key: value for key, value in manifest.items() if key != "dataset_digest"}) != dataset_digest:
            raise GymJobError("training dataset manifest failed integrity verification")
        examples = []
        for entry in manifest["examples"]:
            example = self._load_state("training_examples", entry["example_digest"][7:])
            if example is None or _document_digest(example) != entry["example_digest"] or example["source_split"] != "train":
                raise GymJobError("training example failed integrity or split verification")
            response = self._get_response(example["practice_plan_digest"], example["attempt_id"])
            if response is None or _document_digest(response) != example["response_artifact_digest"]:
                raise GymJobError("training response provenance changed")
            examples.append(example)
        return tuple(examples)

    def select_training_dataset(
        self, dataset_digest: str, scenario_ids: Sequence[str], *, dataset_id: str,
        selection_policy: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Create a provenance-preserving subset of a frozen public-train dataset."""
        _digest(dataset_digest, "dataset_digest")
        if (self.state_dir is None or not isinstance(dataset_id, str) or not dataset_id.strip()
            or not isinstance(scenario_ids, (list, tuple)) or not scenario_ids
            or len(scenario_ids) > MAX_PRACTICE_ATTEMPTS
            or any(not isinstance(value, str) or not value for value in scenario_ids)
            or len(set(scenario_ids)) != len(scenario_ids)):
            raise GymJobError("training subset requires durable state, a bounded unique scenario roster, and dataset ID")
        if not isinstance(selection_policy, Mapping) or not selection_policy:
            raise GymJobError("training subset requires an explicit selection policy")
        selection_policy = json.loads(_wire_document(dict(selection_policy)))
        parent = self._load_state("datasets", dataset_digest[7:])
        if (parent is None or parent.get("dataset_digest") != dataset_digest
            or _document_digest({key: value for key, value in parent.items() if key != "dataset_digest"}) != dataset_digest
            or parent.get("source_split") != "train"):
            raise GymJobError("training subset source must be an intact frozen train dataset")
        examples = self.load_training_examples(dataset_digest)
        by_scenario = {example["scenario_id"]: example for example in examples}
        if not set(scenario_ids) <= set(by_scenario):
            raise GymJobError("training subset scenario IDs must come from the frozen parent dataset")
        selected_digests = {_document_digest(example) for example in (by_scenario[scenario_id] for scenario_id in scenario_ids)}
        selected_rows = [row for row in parent["examples"] if row["example_digest"] in selected_digests]
        if len(selected_rows) != len(scenario_ids):
            raise GymJobError("training subset failed exact example provenance matching")
        manifest = {key: value for key, value in parent.items() if key != "dataset_digest"}
        manifest.update({"dataset_id": dataset_id, "examples": selected_rows,
            "parent_dataset_digest": dataset_digest, "selection_policy": selection_policy,
            "selected_scenario_ids": list(scenario_ids), "release_authorized": False})
        manifest["dataset_digest"] = _document_digest(manifest)
        self._persist_state("datasets", manifest["dataset_digest"][7:], manifest)
        return manifest

    def _get_scenario_batch(self, manifest_digest: str) -> dict[str, Any] | None:
        if not _valid_digest(manifest_digest):
            raise GymJobError("scenario_manifest_digest must be a sha256 digest")
        snapshot = self._scenario_batches.get(manifest_digest)
        indexed = self._scenario_indexes.get(manifest_digest)
        if indexed is not None and indexed[0] is snapshot:
            return json.loads(snapshot)
        if snapshot is not None:
            saved = {
                "batch": json.loads(snapshot),
                "references": {
                    scenario_id: reference
                    for (digest, scenario_id), reference in self._scenario_references.items()
                    if digest == manifest_digest
                },
            }
        else:
            saved = self._load_state("scenarios", manifest_digest[7:])
            if saved is None:
                return None
        batch, references = saved.get("batch"), saved.get("references")
        if not isinstance(batch, dict) or not isinstance(references, dict):
            raise GymJobError("persisted scenario bundle is malformed")
        manifest = {key: value for key, value in batch.items() if key != "manifest_digest"}
        if (
            batch.get("manifest_digest") != manifest_digest
            or _document_digest(manifest) != manifest_digest
        ):
            raise GymJobError("persisted scenario manifest digest does not match its content")
        scenario_ids = {row["scenario_id"] for row in batch.get("scenarios", [])}
        if set(references) != scenario_ids:
            raise GymJobError("persisted private references do not match the scenario manifest")
        for scenario in batch["scenarios"]:
            reference = references[scenario["scenario_id"]]
            encoded_reference = {"kind": "nf_gym_reference_v1", "reference": reference} if scenario.get("reference_encoding") == "nf_gym_reference_v1" else reference
            if str(content_digest(canonicalize(encoded_reference))) != scenario.get("reference_digest"):
                raise GymJobError("persisted private scenario reference changed after materialization")
            self._scenario_references[(manifest_digest, scenario["scenario_id"])] = reference
        snapshot = _snapshot(batch)
        self._scenario_batches[manifest_digest] = snapshot
        self._scenario_indexes[manifest_digest] = (snapshot, {row["scenario_id"]: _snapshot(row) for row in batch["scenarios"]}, batch["job_digest"])
        return json.loads(snapshot)

    def _get_scenario(self, manifest_digest: str, scenario_id: str, job_digest: str) -> dict[str, Any] | None:
        indexed = self._scenario_indexes.get(manifest_digest)
        if indexed is None or indexed[0] is not self._scenario_batches.get(manifest_digest):
            if self._get_scenario_batch(manifest_digest) is None:
                return None
            indexed = self._scenario_indexes[manifest_digest]
        if indexed[2] != job_digest:
            raise GymJobError("scenario belongs to another frozen job")
        raw = indexed[1].get(scenario_id)
        return json.loads(raw) if raw is not None else None

    def freeze_practice_plan(
        self,
        spec: GymJobSpec,
        candidate: CandidateIdentity,
        *,
        run_id: str,
        scenario_manifest_digest: str,
        repetitions: int = 1,
        execution_kind: AttemptExecutionKind | None = None,
        executor_digest: str | None = None,
        inference_config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Freeze the full public-scenario/repetition roster before any model attempts."""
        self.components.validate_job(spec)
        if candidate.method_id not in {method.method_id for method in spec.candidate_methods}:
            raise GymJobError(f"candidate method {candidate.method_id!r} is not declared by the job")
        if not isinstance(run_id, str) or not run_id.strip():
            raise GymJobError("run_id must be non-empty")
        if type(repetitions) is not int or not 1 <= repetitions <= 100:
            raise GymJobError("repetitions must be an integer within [1, 100]")
        key = (candidate.candidate_id, candidate.candidate_model_digest, candidate.artifact_digest)
        binding = self._candidate_executors.get(key)
        kind = execution_kind or ("registered_candidate" if binding is not None else "caller_supplied")
        if kind not in {"caller_supplied", "registered_candidate"}:
            raise GymJobError("unsupported plan execution type")
        if kind == "registered_candidate":
            selected_executor = executor_digest or (binding[0] if binding is not None else None)
            _digest(selected_executor, "plan.executor_digest")
        else:
            if executor_digest is not None:
                raise GymJobError("caller-submission plans cannot claim an executor")
            selected_executor = None
        config_digest = _document_digest({"kind": "nf_inference_config_v1", "config": dict(inference_config or {})})
        if inference_config is None and binding is not None:
            config_digest = self._executor_configs[key]
        _digest(scenario_manifest_digest, "scenario_manifest_digest")
        batch = self._get_scenario_batch(scenario_manifest_digest)
        if batch is None or batch["job_digest"] != spec.digest or batch["split"] not in {"train", "validation"}:
            raise GymJobError("practice planning requires a materialized public scenario batch")
        expected_attempts: list[dict[str, Any]] = []
        for scenario in batch["scenarios"]:
            for repetition in range(repetitions):
                attempt_id = str(content_digest(canonicalize({
                    "run_id": run_id,
                    "candidate_id": candidate.candidate_id,
                    "scenario_id": scenario["scenario_id"],
                    "repetition": repetition,
                })))
                expected_attempts.append({
                    "attempt_id": attempt_id,
                    "scenario_id": scenario["scenario_id"],
                    "scenario_digest": str(content_digest(canonicalize(scenario))),
                    "scenario_origin": scenario["origin"],
                    "scenario_split": scenario["split"],
                    "repetition": repetition,
                })
        if len(expected_attempts) > MAX_PRACTICE_ATTEMPTS:
            raise GymJobError("practice plan exceeds its expected-attempt limit")
        payload = {
            "schema_version": 1,
            "kind": "nf_gym_practice_plan_v1",
            "run_id": run_id,
            "job_digest": spec.digest,
            "scenario_manifest_digest": scenario_manifest_digest,
            "candidate": candidate.to_document(),
            "expected_attempts": expected_attempts,
            "scorer_digest": spec.scorer.implementation_digest,
            "verifier_digest": spec.verifier.implementation_digest,
            "execution_policy": {"kind": kind, "executor_digest": selected_executor, "inference_config_digest": config_digest},
        }
        plan = {**payload, "plan_digest": _document_digest(payload)}
        previous = self._load_frozen_plan(plan["plan_digest"])
        if previous is not None and previous != plan:
            raise GymJobError("practice plan digest collision")
        if self._attempt_db_path is not None:
            connection = sqlite3.connect(self._attempt_db_path, timeout=30)
            try:
                with connection:
                    connection.execute("INSERT OR IGNORE INTO frozen_run_bindings VALUES (?, ?)", (run_id, plan["plan_digest"]))
                    bound = connection.execute("SELECT plan_digest FROM frozen_run_bindings WHERE run_id=?", (run_id,)).fetchone()
                    if bound is None or bound[0] != plan["plan_digest"]:
                        raise GymJobError("run identity changed; use a new run revision")
            finally:
                connection.close()
        self._persist_state("plans", plan["plan_digest"][7:], plan)
        self._practice_plans[plan["plan_digest"]] = json.loads(_snapshot(plan))
        self._terminal_attempts.setdefault(plan["plan_digest"], {})
        return plan

    def register_candidate_executor(
        self,
        candidate: CandidateIdentity,
        *,
        executor_digest: str,
        execute: Callable[..., Any],
        identity_probe: Callable[[], Mapping[str, Any]] | None = None,
        inference_config: Mapping[str, Any] | None = None,
    ) -> None:
        """Bind trusted model execution code to one exact candidate artifact identity."""
        _digest(executor_digest, "executor_digest")
        if not callable(execute):
            raise GymJobError("candidate executor must be callable trusted code")
        key = (candidate.candidate_id, candidate.candidate_model_digest, candidate.artifact_digest)
        if key in self._candidate_executors:
            raise GymJobError("candidate executor is already registered for this artifact")
        self._candidate_executors[key] = (executor_digest, execute)
        self._executor_configs[key] = _document_digest({"kind": "nf_inference_config_v1", "config": dict(inference_config or {})})
        if identity_probe is not None:
            if not callable(identity_probe):
                raise GymJobError("worker identity must be probed by trusted execution code, not submitted JSON")
            self._identity_probes[key] = identity_probe

    def _capture_worker_identity(self, candidate: CandidateIdentity, executor_digest: str) -> None:
        key = (candidate.candidate_id, candidate.candidate_model_digest, candidate.artifact_digest)
        probe = self._identity_probes.get(key)
        if probe is None:
            return
        identity = dict(probe())
        checkpoint_digest = candidate.base_model_digest or candidate.artifact_digest
        if identity.get("checkpoint_digest") != checkpoint_digest or identity.get("local_loaded_files_verified") is not True:
            raise GymJobError("worker loaded checkpoint differs from the frozen candidate identity")
        for field in ("config_digest", "tokenizer_digest", "state_digest"):
            _digest(identity.get(field), f"worker.{field}")
        if candidate.base_model_digest is None and identity["state_digest"] != candidate.candidate_model_digest:
            raise GymJobError("base-free worker state differs from the frozen candidate identity")
        for field in ("model_id", "model_family", "device", "precision"):
            if not isinstance(identity.get(field), str) or not identity[field]:
                raise GymJobError(f"worker.{field} is required")
        revision = identity.get("revision")
        if not (candidate.base_model_digest is None and revision is None) and (
            not isinstance(revision, str) or len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision)
        ):
            raise GymJobError("worker checkpoint revision must be an immutable commit")
        record = {"candidate": candidate.to_document(), "executor_digest": executor_digest, "identity": identity}
        record["identity_digest"] = _document_digest(record)
        self._worker_identities[candidate.artifact_digest] = record
        self._persist_state("worker_identities", candidate.artifact_digest[7:], record)

    def _get_worker_identity(self, candidate: CandidateIdentity) -> dict[str, Any] | None:
        record = self._worker_identities.get(candidate.artifact_digest) or self._load_state("worker_identities", candidate.artifact_digest[7:])
        if record is None:
            return None
        if record.get("candidate") != candidate.to_document() or record.get("identity_digest") != _document_digest({key: value for key, value in record.items() if key != "identity_digest"}):
            raise GymJobError("saved worker identity failed binding or digest verification")
        return record

    def score_attempt(
        self,
        spec: GymJobSpec,
        candidate: CandidateIdentity,
        *,
        practice_plan_digest: str,
        attempt_id: str,
        response: Any,
    ) -> PracticeAttemptRecord:
        """Score caller-supplied response data; this is not a measured model execution."""
        self._get_frozen_plan(spec, candidate, practice_plan_digest, attempt_id)
        with self._claim_batch(practice_plan_digest, (attempt_id,)):
            document = self._store_response(practice_plan_digest, attempt_id, response, "caller_supplied", None)
            return self._score_attempt_response(
                spec, candidate, practice_plan_digest=practice_plan_digest, attempt_id=attempt_id,
                response=document["response"], execution_kind="caller_supplied", executor_digest=None,
                json_encodable=document["json_encodable"],
            )

    def run_practice_attempt(
        self,
        spec: GymJobSpec,
        candidate: CandidateIdentity,
        *,
        practice_plan_digest: str,
        attempt_id: str,
    ) -> PracticeAttemptRecord:
        """Run one planned attempt through the same batch-capable executor interface."""
        results = self.run_practice_batch(
            spec,
            candidate,
            practice_plan_digest=practice_plan_digest,
            attempt_ids=(attempt_id,),
        )
        try:
            return results[0]
        except IndexError as exc:
            raise GymJobError("candidate executor omitted the planned attempt") from exc

    def run_practice_batch(
        self,
        spec: GymJobSpec,
        candidate: CandidateIdentity,
        *,
        practice_plan_digest: str,
        attempt_ids: Sequence[str],
    ) -> tuple[PracticeAttemptRecord, ...]:
        if len(attempt_ids) > MAX_PRACTICE_BATCH:
            raise GymJobError(f"practice batch exceeds {MAX_PRACTICE_BATCH} attempts; dispatch bounded chunks")
        self._get_frozen_plan(spec, candidate, practice_plan_digest, attempt_ids[0] if attempt_ids else "")
        if any(attempt_id not in self._expected_attempts[practice_plan_digest] for attempt_id in attempt_ids):
            raise GymJobError("practice batch contains an unplanned attempt")
        with self._claim_batch(practice_plan_digest, attempt_ids):
            saved_ids = tuple(attempt_id for attempt_id in attempt_ids if self._get_response(practice_plan_digest, attempt_id) is not None)
            results = []
            for attempt_id in saved_ids:
                document = self._get_response(practice_plan_digest, attempt_id)
                try:
                    results.append(self._score_attempt_response(
                        spec, candidate, practice_plan_digest=practice_plan_digest, attempt_id=attempt_id,
                        response=document["response"], execution_kind=document["execution_kind"],
                        executor_digest=document["executor_digest"], json_encodable=document["json_encodable"],
                    ))
                except Exception as exc:  # noqa: BLE001 -- keep already generated work for scoring retry.
                    self._record_scoring_error(practice_plan_digest, attempt_id, exc)
            missing_ids = tuple(attempt_id for attempt_id in attempt_ids if attempt_id not in saved_ids)
            if not missing_ids:
                return tuple(results)
            return (*results, *self._run_practice_batch(
                spec, candidate, practice_plan_digest=practice_plan_digest, attempt_ids=missing_ids,
            ))

    def _run_practice_batch(
        self,
        spec: GymJobSpec,
        candidate: CandidateIdentity,
        *,
        practice_plan_digest: str,
        attempt_ids: Sequence[str],
    ) -> tuple[PracticeAttemptRecord, ...]:
        """Dispatch a frozen batch once; executors may batch inference and reuse runtime caches."""
        key = (candidate.candidate_id, candidate.candidate_model_digest, candidate.artifact_digest)
        binding = self._candidate_executors.get(key)
        if binding is None:
            raise GymJobError("candidate artifact has no registered model executor")
        executor_digest, execute = binding
        if not attempt_ids or len(attempt_ids) != len(set(attempt_ids)):
            raise GymJobError("practice batch attempt IDs must be non-empty and unique")
        plan = self._get_frozen_plan(spec, candidate, practice_plan_digest, attempt_ids[0])
        self._validate_execution_policy(plan, "registered_candidate", executor_digest)
        if plan.get("execution_policy", {}).get("inference_config_digest", self._executor_configs[key]) != self._executor_configs[key]:
            raise GymJobError("inference configuration changed after plan freeze")
        expected_by_id = {row["attempt_id"]: row for row in plan["expected_attempts"]}
        if any(attempt_id not in expected_by_id for attempt_id in attempt_ids):
            raise GymJobError("practice batch contains an unplanned attempt")
        batch = self._get_scenario_batch(plan["scenario_manifest_digest"])
        if batch is None:
            raise GymJobError("frozen public scenario batch is unavailable")
        scenarios_by_id = {row["scenario_id"]: row for row in batch["scenarios"]}
        dispatch = []
        for attempt_id in attempt_ids:
            expected = expected_by_id[attempt_id]
            self._get_frozen_plan(spec, candidate, practice_plan_digest, attempt_id)
            scenario = scenarios_by_id.get(expected["scenario_id"])
            if scenario is None or str(content_digest(canonicalize(scenario))) != expected["scenario_digest"]:
                raise GymJobError("frozen practice scenario is missing or changed")
            dispatch.append({
                "attempt_id": attempt_id,
                "scenario_id": expected["scenario_id"],
                "repetition": expected["repetition"],
                "input": json.loads(_wire_document(scenario["input"])),
            })
        try:
            self._capture_worker_identity(candidate, executor_digest)
            responses = execute(candidate=candidate.to_document(), scenarios=tuple(dispatch))
        except Exception as exc:  # noqa: BLE001 -- infrastructure failures remain retryable.
            self._record_infrastructure_error(practice_plan_digest, attempt_ids, exc)
            return ()
        if not isinstance(responses, Mapping):
            raise GymJobError("candidate batch executor must return an attempt_id to response mapping")
        if set(responses) - set(attempt_ids):
            raise GymJobError("candidate batch executor returned an unplanned attempt ID")
        documents = {
            attempt_id: self._store_response(
                practice_plan_digest, attempt_id,
                responses[attempt_id].response if isinstance(responses[attempt_id], ExecutionResponse) else responses[attempt_id],
                "registered_candidate", executor_digest,
                interaction=responses[attempt_id].interaction if isinstance(responses[attempt_id], ExecutionResponse) else None,
            )
            for attempt_id in attempt_ids if attempt_id in responses
        }
        results = []
        for attempt_id, document in documents.items():
            try:
                results.append(self._score_attempt_response(
                    spec, candidate, practice_plan_digest=practice_plan_digest, attempt_id=attempt_id,
                    response=document["response"], execution_kind="registered_candidate", executor_digest=executor_digest,
                    json_encodable=document["json_encodable"],
                ))
            except Exception as exc:  # noqa: BLE001 -- all returned responses are durable before scoring.
                self._record_scoring_error(practice_plan_digest, attempt_id, exc)
        return tuple(results)

    def compare_unchanged_bases(
        self,
        spec: GymJobSpec,
        candidates: Sequence[CandidateIdentity],
        *,
        scenario_manifest_digest: str,
        repetitions: int = 1,
        require_worker_identity: bool = False,
        require_distinct_families: bool = False,
    ) -> dict[str, Any]:
        """Score at least two unchanged bases on the exact same public scenario manifest."""
        if len(candidates) < 2 or len({candidate.candidate_id for candidate in candidates}) != len(candidates):
            raise GymJobError("base comparison requires at least two uniquely identified candidates")
        if len({candidate.base_model_digest for candidate in candidates}) != len(candidates):
            raise GymJobError("base comparison requires distinct base checkpoint digests")
        if require_distinct_families and not require_worker_identity:
            raise GymJobError("distinct family qualification requires worker-verified identity")
        if require_worker_identity:
            for candidate in candidates:
                key = (candidate.candidate_id, candidate.candidate_model_digest, candidate.artifact_digest)
                if key not in self._identity_probes and self._get_worker_identity(candidate) is None:
                    raise GymJobError("real-base comparison requires a worker checkpoint identity probe")
        if any(
            candidate.method_id != "unchanged_base"
            or candidate.base_model_digest != candidate.candidate_model_digest
            for candidate in candidates
        ):
            raise GymJobError("base comparison candidates must be declared unchanged bases")
        if "unchanged_base" not in {method.method_id for method in spec.candidate_methods}:
            raise GymJobError("JobSpec must declare the unchanged_base candidate method")
        batch = self._get_scenario_batch(scenario_manifest_digest)
        if batch is None or batch["job_digest"] != spec.digest or batch["split"] not in {"train", "validation"}:
            raise GymJobError("base comparison requires a materialized public scenario manifest")
        runs: list[dict[str, Any]] = []
        for candidate in candidates:
            run_id = f"unchanged-bases:{scenario_manifest_digest[7:19]}:{candidate.candidate_id}"
            plan = self.freeze_practice_plan(
                spec,
                candidate,
                run_id=run_id,
                scenario_manifest_digest=scenario_manifest_digest,
                repetitions=repetitions,
                execution_kind="registered_candidate",
            )
            saved_run = self.record_practice_run(plan)
            terminal_ids = {row["attempt_id"] for row in saved_run["attempts"]}
            missing_ids = tuple(
                row["attempt_id"] for row in plan["expected_attempts"]
                if row["attempt_id"] not in terminal_ids
            )
            if missing_ids:
                for index in range(0, len(missing_ids), MAX_PRACTICE_BATCH):
                    self.run_practice_batch(
                        spec, candidate, practice_plan_digest=plan["plan_digest"],
                        attempt_ids=missing_ids[index:index + MAX_PRACTICE_BATCH],
                    )
            runs.append(self.record_practice_run(plan))
        status = (
            "complete" if all(run["status"] == "complete" and run["execution_summary"]["registered_execution_complete"] for run in runs)
            else "failed" if any(run["status"] == "failed" for run in runs)
            else "incomplete"
        )
        comparison = {
            "schema_version": 1,
            "kind": "nf_gym_unchanged_base_comparison_v1",
            "job_id": spec.job_id,
            "job_digest": spec.digest,
            "scenario_manifest_digest": scenario_manifest_digest,
            "candidate_ids": [candidate.candidate_id for candidate in candidates],
            "runs": runs,
            "status": status,
            "authority": {
                "same_public_panel": True,
                "served_model_identity_verified": False,
                "holdout_used": False,
                "promotion_authorized": False,
            },
        }
        identities = [self._get_worker_identity(candidate) for candidate in candidates]
        verified = all(identity is not None for identity in identities)
        families = {identity["identity"]["model_family"] for identity in identities if identity is not None}
        comparison["worker_identities"] = identities
        comparison["authority"]["worker_checkpoint_identity_verified"] = verified
        comparison["authority"]["distinct_model_families_verified"] = verified and len(families) == len(candidates)
        if require_worker_identity and not verified:
            comparison["status"] = "incomplete"
        if require_worker_identity and any(not run["execution_summary"]["registered_execution_complete"] for run in runs):
            comparison["status"] = "incomplete"
        if require_worker_identity and verified and len({record["identity"]["state_digest"] for record in identities if record is not None}) != len(candidates):
            raise GymJobError("worker-verified bases must have distinct loaded tensor states")
        if require_distinct_families and verified and len(families) != len(candidates):
            raise GymJobError("verified checkpoints are not from distinct model families")
        comparison["comparison_digest"] = _document_digest(comparison)
        return comparison

    def _get_frozen_plan(
        self,
        spec: GymJobSpec,
        candidate: CandidateIdentity,
        practice_plan_digest: str,
        attempt_id: str,
    ) -> dict[str, Any]:
        _digest(practice_plan_digest, "practice_plan_digest")
        plan = self._load_frozen_plan(practice_plan_digest)
        if plan is None or plan["job_digest"] != spec.digest:
            raise GymJobError("attempt requires a frozen practice plan for this job")
        if plan["candidate"] != candidate.to_document():
            raise GymJobError("attempt candidate differs from the frozen practice plan")
        if attempt_id not in self._expected_attempts[practice_plan_digest]:
            raise GymJobError("attempt ID is not in the frozen practice roster")
        if self._lookup_terminal(practice_plan_digest, attempt_id) is not None:
            raise GymJobError("practice attempt already reached a terminal state")
        if plan["scorer_digest"] != spec.scorer.implementation_digest or plan["verifier_digest"] != spec.verifier.implementation_digest:
            raise GymJobError("scorer or verifier identity changed after practice plan freeze")
        return plan

    def _score_attempt_response(
        self,
        spec: GymJobSpec,
        candidate: CandidateIdentity,
        *,
        practice_plan_digest: str,
        attempt_id: str,
        response: Any,
        execution_kind: AttemptExecutionKind,
        executor_digest: str | None,
        json_encodable: bool = True,
    ) -> PracticeAttemptRecord:
        """Validate, verify, and score one response tied to the frozen practice plan."""
        self.components.validate_job(spec)
        if candidate.method_id not in {method.method_id for method in spec.candidate_methods}:
            raise GymJobError(f"candidate method {candidate.method_id!r} is not declared by the job")
        plan = self._get_frozen_plan(spec, candidate, practice_plan_digest, attempt_id)
        expected = self._expected_attempts[practice_plan_digest][attempt_id]
        scenario_manifest_digest = plan["scenario_manifest_digest"]
        scenario = self._get_scenario(scenario_manifest_digest, expected["scenario_id"], spec.digest)
        if scenario is None:
            raise GymJobError("frozen practice plan has no matching materialized scenario batch")
        if scenario is None or str(content_digest(canonicalize(scenario))) != expected["scenario_digest"]:
            raise GymJobError("frozen practice scenario is missing or changed")
        reference_key = (scenario_manifest_digest, scenario["scenario_id"])
        if reference_key not in self._scenario_references:
            raise GymJobError("private verifier reference is missing for the planned scenario")
        reference = self._scenario_references[reference_key]
        response = json.loads(_wire_document(response))
        output_valid = json_encodable
        try:
            validate_task_instance(spec.task_output_schema, response, field=f"attempt[{attempt_id}].response")
        except GymJobError:
            output_valid = False
        try:
            response_digest = _document_digest({"kind": "nf_gym_response_v1", "response": response})
        except (NativeError, TypeError, ValueError, OverflowError, RecursionError):
            output_valid = False
            response_digest = str(content_digest(canonicalize({
                "kind": "nf_gym_invalid_response_v1", "response_type": type(response).__name__,
            })))
        scenario_document = json.loads(_snapshot(scenario))
        reference = json.loads(_wire_document(reference))
        verifier = self.components.resolve(
            "verifier", spec.verifier.component_id, spec.verifier.version, spec.verifier.implementation_digest
        )
        verifier_input = {
            "scenario": scenario_document, "reference": reference, "response": response,
        }
        verified_bytes = _snapshot(verifier_input)
        detached_verifier_input = json.loads(verified_bytes)
        verification = verifier(
            parameters=spec.verifier.to_document()["parameters"],
            **detached_verifier_input,
            output_valid=output_valid,
        ) if output_valid else {"passed": False, "reason": "invalid output"}
        if _snapshot(detached_verifier_input) != verified_bytes:
            raise GymJobError("verifier mutated the response, reference, or scenario it was given")
        if not isinstance(verification, Mapping) or type(verification.get("passed")) is not bool:
            raise GymJobError("verifier must return an object with a boolean passed field")
        scorer = self.components.resolve(
            "scorer", spec.scorer.component_id, spec.scorer.version, spec.scorer.implementation_digest
        )
        scorer_input = json.loads(verified_bytes)
        frozen_verification = json.loads(_wire_document(verification))
        scoring_bytes = _snapshot({**scorer_input, "verification": frozen_verification})
        scorer_input["verification"] = frozen_verification
        metrics = scorer(
            parameters=spec.scorer.to_document()["parameters"],
            **scorer_input,
            output_valid=output_valid,
        ) if output_valid else {}
        if _snapshot(scorer_input) != scoring_bytes:
            raise GymJobError("scorer mutated its frozen input")
        if not isinstance(metrics, Mapping):
            raise GymJobError("scorer must return a metric object")
        status: AttemptStatus = "scored" if output_valid else "invalid"
        scenario_digest = str(content_digest(canonicalize(scenario_document)))
        attempt = PracticeAttemptRecord(
            attempt_id=attempt_id,
            job_digest=spec.digest,
            candidate_id=candidate.candidate_id,
            scenario_id=expected["scenario_id"],
            scenario_digest=scenario_digest,
            scenario_origin=scenario["origin"],
            scenario_split=scenario["split"],
            round_number=expected["repetition"],
            response_digest=response_digest,
            scorer_digest=spec.scorer.implementation_digest,
            verifier_digest=spec.verifier.implementation_digest,
            output_valid=output_valid,
            verifier_passed=verification["passed"],
            execution_kind=execution_kind,
            executor_digest=executor_digest,
            status=status,
            metrics=metrics,
        )
        self._persist_terminal_attempt(practice_plan_digest, attempt)
        return attempt

    def record_practice_run(
        self,
        practice_plan: Mapping[str, Any],
        attempts: Sequence[PracticeAttemptRecord] | None = None,
    ) -> dict[str, Any]:
        plan_digest = practice_plan.get("plan_digest")
        if not _valid_digest(plan_digest):
            raise GymJobError("practice plan digest is missing or malformed")
        frozen = self._load_frozen_plan(plan_digest)
        if frozen is None or dict(practice_plan) != frozen:
            raise GymJobError("practice plan differs from the plan frozen by this service")
        terminal = self._load_terminal_attempts(plan_digest)
        if attempts is not None:
            for attempt in attempts:
                existing = terminal.get(attempt.attempt_id)
                if existing is None:
                    raise GymJobError("submitted attempt is not server-recorded; submit a raw response instead")
                if existing != attempt:
                    raise GymJobError("submitted attempt differs from its terminal scored result")
                terminal[attempt.attempt_id] = attempt
            PracticeRunRecord.create(plan=practice_plan, attempts=tuple(terminal.values()))
        terminal = self._load_terminal_attempts(plan_digest)
        run = PracticeRunRecord.create(plan=practice_plan, attempts=tuple(terminal.values())).to_document()
        registered_count = sum(row["execution_kind"] == "registered_candidate" for row in run["attempts"])
        run["execution_summary"] = {
            "registered_candidate_attempts": registered_count,
            "caller_supplied_attempts": len(run["attempts"]) - registered_count,
            "registered_execution_complete": run["status"] == "complete" and registered_count == len(run["expected_attempt_ids"]),
        }
        if self._attempt_db_path is not None:
            connection = sqlite3.connect(self._attempt_db_path, timeout=30)
            try:
                errors = connection.execute(
                    "SELECT attempt_id, error_digest, retry_count FROM infrastructure_errors "
                    "WHERE plan_digest=? ORDER BY attempt_id", (plan_digest,),
                ).fetchall()
            finally:
                connection.close()
            if errors:
                run["infrastructure_errors"] = [
                    {"attempt_id": row[0], "error_digest": row[1], "retry_count": row[2]} for row in errors
                ]
        run["run_digest"] = _document_digest({key: value for key, value in run.items() if key != "run_digest"})
        self._terminal_attempts[plan_digest] = terminal
        self._persist_state("runs", plan_digest[7:], run)
        return run

    def score_candidate_document(
        self,
        job_document: Mapping[str, Any],
        candidate_document: Mapping[str, Any],
        measurements: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        required = {
            "candidate_id", "method_id", "base_model_digest", "candidate_model_digest", "artifact_digest"
        }
        if not isinstance(candidate_document, Mapping) or set(candidate_document) != required:
            raise GymJobError("candidate identity has missing or unknown fields")
        candidate = CandidateIdentity(**candidate_document)
        return self.score_candidate(gym_job_from_document(job_document), candidate, measurements)

    def record_practice_run_document(
        self,
        run_document: Mapping[str, Any],
    ) -> dict[str, Any]:
        required = {"practice_plan", "attempts"}
        if not isinstance(run_document, Mapping) or set(run_document) != required:
            raise GymJobError("practice run request has missing or unknown fields")
        attempts_value = run_document["attempts"]
        if not isinstance(attempts_value, list) or len(attempts_value) > MAX_PRACTICE_ATTEMPTS:
            raise GymJobError("practice run attempts must be a list")
        attempt_fields = {
            "attempt_id", "job_digest", "candidate_id", "scenario_id", "scenario_digest",
            "scenario_origin", "scenario_split", "round_number", "response_digest",
            "scorer_digest", "verifier_digest", "output_valid", "verifier_passed", "status", "metrics",
            "execution_kind", "executor_digest",
        }
        plan = run_document["practice_plan"]
        if not isinstance(plan, Mapping):
            raise GymJobError("practice_plan must be an object")
        frozen = self._load_frozen_plan(plan.get("plan_digest"))
        if frozen is None or dict(plan) != frozen:
            raise GymJobError("practice plan differs from the plan frozen by this service")
        attempts: list[PracticeAttemptRecord] = []
        forms = []
        seen_ids = set()
        for index, value in enumerate(attempts_value):
            if not isinstance(value, Mapping) or set(value) not in ({"attempt_id", "response"}, attempt_fields):
                raise GymJobError(f"practice attempt {index} has missing or unknown fields")
            attempt_id = value["attempt_id"]
            if not isinstance(attempt_id, str) or attempt_id in seen_ids or attempt_id not in self._expected_attempts[frozen["plan_digest"]]:
                raise GymJobError("submission attempt IDs must be frozen, unique strings")
            seen_ids.add(attempt_id)
            if "response" in value:
                self._validate_execution_policy(frozen, "caller_supplied", None)
                _wire_document(value["response"])
                forms.append("response")
                existing = self._lookup_terminal(frozen["plan_digest"], attempt_id)
                if existing is not None:
                    saved = self._get_response(frozen["plan_digest"], attempt_id)
                    matches = saved is not None and (
                        _wire_document(saved["response"]) == _wire_document(value["response"])
                        or (not saved["json_encodable"] and saved["response"].get("original_json") == _wire_document(value["response"]).decode())
                    )
                    if not matches:
                        raise GymJobError("submitted response differs from the saved result")
            else:
                record = PracticeAttemptRecord(**value)
                if self._lookup_terminal(frozen["plan_digest"], attempt_id) != record:
                    raise GymJobError("submitted attempt is not server-recorded")
                forms.append("readback")
        if len(set(forms)) > 1:
            raise GymJobError("readback records and response submissions require separate requests")
        if forms and forms[0] == "response":
            spec = self._jobs.get(frozen["job_digest"])
            if spec is None:
                document = self._load_state("jobs", frozen["job_digest"][7:])
                if document is None:
                    raise GymJobError("frozen job is unavailable for response submission")
                spec = gym_job_from_document(document)
                if spec.digest != frozen["job_digest"]:
                    raise GymJobError("persisted job digest differs from the frozen plan")
            candidate = CandidateIdentity(**frozen["candidate"])
            pending = [value for value in attempts_value if self._lookup_terminal(frozen["plan_digest"], value["attempt_id"]) is None]
            for offset in range(0, len(pending), MAX_PRACTICE_BATCH):
                batch = pending[offset:offset + MAX_PRACTICE_BATCH]
                ids = tuple(value["attempt_id"] for value in batch)
                with self._claim_batch(frozen["plan_digest"], ids):
                    documents = {
                        value["attempt_id"]: self._store_response(frozen["plan_digest"], value["attempt_id"], value["response"], "caller_supplied", None)
                        for value in batch
                    }
                    for attempt_id, document in documents.items():
                        try:
                            self._score_attempt_response(
                                spec, candidate, practice_plan_digest=frozen["plan_digest"], attempt_id=attempt_id,
                                response=document["response"], execution_kind="caller_supplied", executor_digest=None,
                                json_encodable=document["json_encodable"],
                            )
                        except Exception as exc:  # noqa: BLE001 -- admitted requests return partial scoring explicitly.
                            self._record_scoring_error(frozen["plan_digest"], attempt_id, exc)
            run = self.record_practice_run(plan)
            run["submission_results"] = [
                {"attempt_id": value["attempt_id"], "status": "recorded" if self._lookup_terminal(frozen["plan_digest"], value["attempt_id"]) else "pending_scoring"}
                for value in attempts_value
            ]
            run["run_digest"] = _document_digest({key: value for key, value in run.items() if key != "run_digest"})
            return run
        for index, value in enumerate(attempts_value):
            if isinstance(value, Mapping) and set(value) == {"attempt_id", "response"}:
                existing = self._lookup_terminal(frozen["plan_digest"], value["attempt_id"])
                if existing is not None:
                    attempts.append(existing)
                    continue
                spec = self._jobs.get(frozen["job_digest"])
                if spec is None:
                    document = self._load_state("jobs", frozen["job_digest"][7:])
                    if document is None:
                        raise GymJobError("frozen job is unavailable for response submission")
                    spec = gym_job_from_document(document)
                    if spec.digest != frozen["job_digest"]:
                        raise GymJobError("persisted job digest differs from the frozen plan")
                    self._jobs[spec.digest] = spec
                attempts.append(self.score_attempt(
                    spec, CandidateIdentity(**frozen["candidate"]),
                    practice_plan_digest=frozen["plan_digest"], attempt_id=value["attempt_id"],
                    response=value["response"],
                ))
                continue
            if not isinstance(value, Mapping) or set(value) != attempt_fields:
                raise GymJobError(f"practice attempt {index} has missing or unknown fields")
            attempts.append(PracticeAttemptRecord(**value))
        return self.record_practice_run(plan, attempts)

    def score_candidate(
        self,
        spec: GymJobSpec,
        candidate: CandidateIdentity,
        measurements: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        self.components.validate_job(spec)
        methods = {method.method_id for method in spec.candidate_methods}
        if candidate.method_id not in methods:
            raise GymJobError(f"candidate method {candidate.method_id!r} is not declared by the job")
        unknown_measurements = set(measurements) - {gate.gate_id for gate in spec.gates}
        if unknown_measurements:
            raise GymJobError(f"measurements contain undeclared gates: {sorted(unknown_measurements)}")
        results: list[dict[str, Any]] = []
        for gate in spec.gates:
            observation = measurements.get(gate.gate_id)
            if observation is None:
                results.append({"gate_id": gate.gate_id, "purpose": gate.purpose, "status": "not_run", "required": gate.required})
                continue
            if not isinstance(observation, Mapping):
                raise GymJobError(f"measurement for gate {gate.gate_id!r} must be an object")
            metric_id, split = observation.get("metric_id"), observation.get("split")
            value = observation.get("value")
            evidence_digest = observation.get("evidence_digest")
            evidence_class = observation.get("evidence_class")
            if metric_id != gate.metric_id or split != gate.split:
                status: GateStatus = "blocked"
                reason = "measurement metric or split does not match the frozen gate"
            elif not isinstance(evidence_class, str) or evidence_class not in {
                "measured", "estimated", "synthetic", "caller_supplied", "unavailable"
            }:
                status, reason = "blocked", "measurement evidence_class is invalid"
            elif evidence_class in {"estimated", "caller_supplied", "unavailable"}:
                status, reason = "blocked", f"evidence_class {evidence_class!r} cannot satisfy a gate"
            elif not _valid_digest(evidence_digest):
                status, reason = "blocked", "measurement evidence_digest is missing or invalid"
            elif not is_finite_number(value):
                status, reason = "blocked", "measurement value must be finite numeric data"
            else:
                passed = {
                    "eq": value == gate.threshold,
                    "ge": value >= gate.threshold,
                    "gt": value > gate.threshold,
                    "le": value <= gate.threshold,
                    "lt": value < gate.threshold,
                }[gate.comparator]
                status, reason = (
                    ("reported_pass", None)
                    if passed
                    else ("reported_fail", "metric missed its gate")
                )
            results.append({
                "gate_id": gate.gate_id,
                "purpose": gate.purpose,
                "metric_id": gate.metric_id,
                "split": gate.split,
                "required": gate.required,
                "status": status,
                "observed": value if is_finite_number(value) else None,
                "threshold": gate.threshold,
                "evidence_class": evidence_class,
                "evidence_digest": evidence_digest if _valid_digest(evidence_digest) else None,
                "reason": reason,
            })
        required = [row for row in results if row["required"]]
        if any(row["status"] == "reported_fail" for row in required):
            decision = "reported_rejected_unverified"
        elif any(row["status"] not in {"reported_pass"} for row in required):
            decision = "incomplete"
        else:
            decision = "reported_pass_pending_verification"
        scorecard = {
            "schema_version": 1,
            "kind": "nf_gym_candidate_scorecard_v1",
            "job_id": spec.job_id,
            "job_digest": spec.digest,
            "candidate": candidate.to_document(),
            "gates": results,
            "decision": decision,
            "serving_identity": {
                "status": "unverified",
                "candidate_model_digest": candidate.candidate_model_digest,
                "reason": "No attested model-serving identity adapter is configured",
            },
            "release_authorized": False,
            "authority": self.describe()["authority"],
        }
        scorecard["scorecard_digest"] = str(content_digest(canonicalize(scorecard)))
        return scorecard


def _valid_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )