"""Durable public-provider reservations, independent of final study custody."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any

from .gym_job import GymJobError, _is_digest
from .gym_service import _document_digest
from .provenance import parse_strict_json


def unsigned_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(spec)
    digest = result.pop("document_digest", None)
    if digest is not None and digest != _document_digest(result):
        raise GymJobError("public budget spec digest differs")
    return result


def _binding(spec: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    try:
        spec = unsigned_spec(spec)
        budget = spec["public_budget"]
        path = Path(budget["path"])
        if (budget["version"] != 1 or not path.is_absolute()
            or not isinstance(budget["ledger_id"], str) or not budget["ledger_id"]):
            raise GymJobError("public budget requires a frozen absolute path and ledger identity")
        policy = spec["policy"]
        roles = {row["role"]: row["execution"]["configuration"]
                 for row in spec["candidates"]
                 if row.get("execution", {}).get("kind") == "provider"
                 and row["execution"].get("configuration", {}).get("model")}
        binding = {"version": 1, "ledger_id": budget["ledger_id"],
                   "run_spec_digest": _document_digest(spec),
                   "policy_digest": _document_digest(policy), "policy": policy, "roles": roles}
        return path, binding
    except (KeyError, TypeError, ValueError) as error:
        raise GymJobError("public budget binding is malformed") from error


def _anchor(path: Path) -> Path:
    return path.with_name(path.name + ".anchor.json")


def initialize_public_budget(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Explicit one-time initialization; missing/corrupt state is never recreated by a client."""
    path, binding = _binding(spec)
    marker = _anchor(path)
    if path.exists() or marker.exists() or path.is_symlink() or marker.is_symlink():
        raise GymJobError("public budget initialization cannot reset existing/partial state")
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"kind": "nf_public_budget_anchor_v1", "binding": binding,
                "binding_digest": _document_digest(binding)}
    document["document_digest"] = _document_digest(document)
    try:
        with marker.open("x", encoding="utf-8") as stream:
            json.dump(document, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        with closing(sqlite3.connect(path)) as connection:
            connection.execute("PRAGMA synchronous=FULL")
            connection.executescript(
                "CREATE TABLE metadata (singleton INTEGER PRIMARY KEY CHECK(singleton=1), payload TEXT NOT NULL);"
                "CREATE TABLE reservations (attempt INTEGER PRIMARY KEY, payload TEXT NOT NULL);"
                "CREATE TABLE state (singleton INTEGER PRIMARY KEY CHECK(singleton=1), payload TEXT NOT NULL);")
            connection.execute("INSERT INTO metadata VALUES (1,?)", (json.dumps(binding, sort_keys=True),))
            state = {"attempts": 0, "reserved_cost_microusd": 0,
                     "tail": _document_digest(binding)}
            state["state_digest"] = _document_digest(state)
            connection.execute("INSERT INTO state VALUES (1,?)", (json.dumps(state, sort_keys=True),))
            connection.commit()
    except (OSError, sqlite3.Error) as error:
        raise GymJobError("public budget initialization failed; existing/partial state must not be reset") from error
    return document


class PublicBudget:
    def __init__(self, spec: Mapping[str, Any]) -> None:
        self.path, self.binding = _binding(spec)
        self.snapshot()

    def _open(self) -> sqlite3.Connection:
        marker = _anchor(self.path)
        connection = None
        try:
            if (self.path.is_symlink() or marker.is_symlink()
                or not self.path.is_file() or not marker.is_file()):
                raise GymJobError("public budget state or anchor is missing")
            anchor = parse_strict_json(marker.read_bytes())
            if (anchor.get("kind") != "nf_public_budget_anchor_v1"
                or anchor.get("binding") != self.binding
                or anchor.get("binding_digest") != _document_digest(self.binding)
                or anchor.get("document_digest") != _document_digest(
                    {key: value for key, value in anchor.items() if key != "document_digest"})):
                raise GymJobError("public budget anchor differs from frozen run/policy")
            connection = sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True, timeout=15)
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA busy_timeout=15000")
            return connection
        except GymJobError:
            raise
        except (OSError, ValueError, sqlite3.Error) as error:
            if connection is not None:
                connection.close()
            raise GymJobError("public budget state is unavailable or corrupt") from error

    def _read(self, connection: sqlite3.Connection) -> dict[str, Any]:
        try:
            if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise GymJobError("public budget database integrity failed")
            metadata = connection.execute("SELECT payload FROM metadata").fetchall()
            states = connection.execute("SELECT payload FROM state").fetchall()
            if len(metadata) != 1 or len(states) != 1 or json.loads(metadata[0][0]) != self.binding:
                raise GymJobError("public budget run/policy binding differs")
            state = json.loads(states[0][0])
            if state.get("state_digest") != _document_digest(
                    {key: value for key, value in state.items() if key != "state_digest"}):
                raise GymJobError("public budget state checksum differs")
            previous, total, attempts = _document_digest(self.binding), 0, 0
            for attempt, encoded in connection.execute("SELECT attempt,payload FROM reservations ORDER BY attempt"):
                reservation = json.loads(encoded)
                attempts += 1
                if (attempt != attempts or reservation.get("attempt") != attempt
                    or reservation.get("previous") != previous
                    or type(reservation.get("cost_microusd")) is not int or reservation["cost_microusd"] < 1
                    or not _is_digest(reservation.get("request_digest"))
                    or reservation.get("reservation_digest") != _document_digest(
                        {key: value for key, value in reservation.items() if key != "reservation_digest"})):
                    raise GymJobError("public budget reservation chain differs")
                total += reservation["cost_microusd"]
                previous = reservation["reservation_digest"]
            api = self.binding["policy"]["api"]
            if (state.get("attempts") != attempts or state.get("reserved_cost_microusd") != total
                or state.get("tail") != previous or attempts > api["maximum_api_attempts"]
                or total > api["maximum_cost_microusd"]):
                raise GymJobError("public budget counters differ or exceed authorization")
            return state
        except (sqlite3.Error, ValueError, KeyError, TypeError) as error:
            raise GymJobError("public budget state is malformed") from error

    @contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        connection = self._open()
        try:
            yield connection
        finally:
            try:
                connection.rollback()
            finally:
                connection.close()

    def snapshot(self) -> dict[str, Any]:
        with self._session() as connection:
            connection.execute("BEGIN")
            return self._read(connection)

    def reserve(self, *, role: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        policy, api = self.binding["policy"], self.binding["policy"]["api"]
        if (api.get("calls_authorized") is not True or role not in self.binding["roles"]
            or payload.get("model") != self.binding["roles"][role]["model"]
            or payload.get("max_completion_tokens") != policy["max_new_tokens"]
            or payload.get("temperature") != policy["temperature"]
            or payload.get("reasoning_effort") != policy["reasoning_effort"]):
            raise GymJobError("public provider request is outside frozen authorization")
        try:
            rates = api["model_rates"][payload["model"]]
            input_bound = len(json.dumps(payload, allow_nan=False).encode()) + api["input_overhead_token_bound"]
            cost = ((input_bound * rates["input_microusd_per_million_tokens"] + 999999) // 1000000
                    + (policy["max_new_tokens"] * rates["output_microusd_per_million_tokens"] + 999999) // 1000000)
        except (KeyError, TypeError, ValueError) as error:
            raise GymJobError("public provider cost cannot be authorized") from error
        with self._session() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                state = self._read(connection)
                if (state["attempts"] >= api["maximum_api_attempts"]
                    or state["reserved_cost_microusd"] + cost > api["maximum_cost_microusd"]):
                    raise GymJobError("durable public API attempt or cost ceiling reached")
                reservation = {"attempt": state["attempts"] + 1, "role": role,
                               "request_digest": _document_digest(payload), "cost_microusd": cost,
                               "input_token_bound": input_bound, "output_token_bound": policy["max_new_tokens"],
                               "previous": state["tail"], "binding_digest": _document_digest(self.binding)}
                reservation["reservation_digest"] = _document_digest(reservation)
                connection.execute("INSERT INTO reservations VALUES (?,?)",
                                   (reservation["attempt"], json.dumps(reservation, sort_keys=True)))
                updated = {"attempts": reservation["attempt"],
                           "reserved_cost_microusd": state["reserved_cost_microusd"] + cost,
                           "tail": reservation["reservation_digest"]}
                updated["state_digest"] = _document_digest(updated)
                connection.execute("UPDATE state SET payload=? WHERE singleton=1",
                                   (json.dumps(updated, sort_keys=True),))
                connection.commit()
                return {**reservation, "run_reserved_cost_microusd": updated["reserved_cost_microusd"]}
            except sqlite3.Error as error:
                raise GymJobError("public budget reservation failed closed") from error
