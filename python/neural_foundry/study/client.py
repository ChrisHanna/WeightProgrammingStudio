"""The proposer's side of the custodian protocol, with a transcript of everything on the wire."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any

import neural_foundry

from .generators import RETENTION_TASK, Cluster, Item
from .plan import base_arm_artifact, digest_bytes


class CustodianRefused(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def write_candidate(path: str | Path, artifact: bytes) -> str:
    Path(path).write_bytes(artifact)
    return digest_bytes(artifact)


def write_base_arm(path: str | Path, plan_digest: str) -> dict[str, Any]:
    """The base model's manifest entry. It has no weights, but it spends a query like any other arm."""
    digest = write_candidate(path, base_arm_artifact(plan_digest))
    return {"arm": "base", "synthesis_seed": None, "candidate_digest": digest, "artifact_path": str(path)}


def declare_exposure(
    clusters: Sequence[Cluster], retention: Sequence[Item] = (), tasks: Sequence[str] = ()
) -> dict[str, list[str]]:
    """Build exposed_operands: every operand and protected prompt the proposer's synthesis saw."""
    exposed: dict[str, set[str]] = {task: set() for task in tasks}
    for cluster in clusters:
        exposed.setdefault(cluster.task, set()).update(cluster.operands)
    if retention:
        exposed[RETENTION_TASK] = {item.prompt for item in retention}
    return {task: sorted(values) for task, values in exposed.items()}


class CustodianClient:
    """Spawns the custodian in its own process. Used as a context manager."""

    def __init__(self, plan_path: str | Path, plan_digest: str, state_dir: str | Path) -> None:
        package_root = str(Path(neural_foundry.__file__).resolve().parents[1])
        env = {**os.environ, "PYTHONPATH": package_root}
        self.transcript: list[bytes] = []
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-P",  # keep the working directory off sys.path; PYTHONPATH still names the package
                "-m",
                "neural_foundry.study.custodian",
                "--plan",
                str(plan_path),
                "--plan-digest",
                plan_digest,
                "--state-dir",
                str(state_dir),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        self._read()

    def _read(self) -> Any:
        assert self.process.stdout is not None
        line = self.process.stdout.readline()
        self.transcript.append(line)
        if not line:
            raise CustodianRefused("CUSTODIAN_EXITED", "the custodian closed its pipe")
        reply = json.loads(line)
        if not reply["ok"]:
            raise CustodianRefused(reply["code"], reply["message"])
        return reply.get("result")

    def _call(self, request: Mapping[str, Any]) -> Any:
        assert self.process.stdin is not None
        line = json.dumps(request).encode("utf-8") + b"\n"
        self.transcript.append(line)
        self.process.stdin.write(line)
        self.process.stdin.flush()
        return self._read()

    def freeze(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = self._call({"op": "freeze", "manifest": manifest})
        return result

    def query(self, candidate_digest: str) -> dict[str, Any]:
        result: dict[str, Any] = self._call({"op": "query", "candidate_digest": candidate_digest})
        return result

    def audit(self, spec: Mapping[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = self._call({"op": "audit", "audit": dict(spec)})
        return result

    def decide(self) -> dict[str, Any]:
        result: dict[str, Any] = self._call({"op": "decide"})
        return result

    def send_raw(self, request: Mapping[str, Any]) -> Any:
        """Send an arbitrary request, for tests that the custodian refuses what it should."""
        return self._call(request)

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        if self.process.stdout is not None:
            self.process.stdout.close()

    def __enter__(self) -> CustodianClient:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()
