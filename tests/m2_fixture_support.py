"""T043 real component drivers; no fallback, grants, mocks, or scientific authority."""

from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from neural_foundry import arithmetic, gyms

ROOT = Path(__file__).resolve().parents[1]
TARGET = Path(os.environ.get("NF_M2_TARGET_DIR", os.environ.get(
    "CARGO_TARGET_DIR", f"/tmp/nf-t043-joined-{os.getpid()}")))
CARGO = Path.home() / ".cargo/bin/cargo"


def run(command: list[str], *, data: str | None = None,
        env: dict[str, str] | None = None, timeout: float = 65) -> subprocess.CompletedProcess[str]:
    """Kill the entire fixture process tree on timeout, including Rust grandchildren."""
    process = subprocess.Popen(command, cwd=ROOT, env=env, text=True,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, start_new_session=True)
    try:
        out, err = process.communicate(data, timeout=timeout)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate(timeout=5)
        raise
    return subprocess.CompletedProcess(command, process.returncode, out, err)


def rust_case(target: str, case: str, *, timeout: float = 65,
              **environment: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "CARGO_TARGET_DIR": str(TARGET), "CARGO_BUILD_JOBS": "2",
           "RUSTUP_TOOLCHAIN": "stable", **environment}
    result = run([str(CARGO), "test", "--offline", "-p", "nf-experiments", "--test", target,
                  case, "--", "--exact", "--test-threads=1", "--nocapture"], env=env, timeout=timeout)
    assert result.returncode == 0, result.stdout + result.stderr
    # An empty selector or ignored test must not become passing T043 evidence.
    assert "test result: ok. 1 passed; 0 failed; 0 ignored" in result.stdout, result.stdout
    return result


def sandbox(root: Path, source: str = "", **options: Any) -> dict[str, Any]:
    root.mkdir(mode=0o700, exist_ok=True)
    result = run([str(TARGET / "debug/examples/fixture_sandbox")], data=json.dumps({
        "root": str(root), "enabled": True, "source": source, **options,
    }), timeout=35)
    assert result.returncode == 0, result.stdout + result.stderr
    reply = json.loads(result.stdout)
    assert "error" not in reply, reply
    return dict(reply["ok"])


def validated_sum(root: Path, model: Any) -> tuple[Any, Any, Any]:
    """Execute genuine imported user code against independent known answers in OCI."""
    path, draft = arithmetic.write_user_sum_package(root / "user-sum", model)
    snapshot = gyms.import_local(path, draft, root / "gym-store")
    rows = json.loads(snapshot.asset_bytes("development.json"))["items"]
    chosen: dict[float, dict[str, Any]] = {}
    for row in rows:
        value = arithmetic.expected("sum", **row["inputs"])
        chosen.setdefault(value, row)
    cases = [{"id": row["task"], "input": row["inputs"], "expected": value,
              "wrong": [value - 1 if value == 16 else value + 1],
              "actions": [], "termination": None}
             for value, row in list(chosen.items())[:2]]
    private = root / "gym-sandbox"
    private.mkdir(mode=0o700)
    validation = gyms.validate(snapshot, {
        "schema_version": 1, "snapshot": snapshot.snapshot_id,
        "truth_provenance": "independent Python integer arithmetic, synthetic known target",
        "cases": cases, "seeds": [17, 42],
        "scoring": {"minimum": 0.0, "maximum": 1.0, "good_min": 1.0, "bad_max": 0.0},
        "limits": {"wall_ms": 2000, "cpu_ms": 2000, "output_bytes": 32768},
    }, private, enabled=True)
    assert json.loads(validation.report_bytes())["status"] == "passed"
    approval = gyms.approve(snapshot, validation, "test-owner:synthetic-t043-exact-snapshot")
    return snapshot, validation, approval


class SplitController:
    def __init__(self, process: subprocess.Popen[str]) -> None:
        self.process = process
        self.ready = self.read()
        assert self.ready.get("ready") is True, self.ready

    def read(self) -> dict[str, Any]:
        assert self.process.stdout is not None
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            assert selector.select(20), "real split controller failed to reply within 20 seconds"
        line = self.process.stdout.readline()
        assert line, "split controller exited without a JSON response"
        return dict(json.loads(line))

    def send(self, op: str, **fields: Any) -> dict[str, Any]:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps({"op": op, **fields}) + "\n")
        self.process.stdin.flush()
        return self.read()

    def ok(self, op: str, **fields: Any) -> Any:
        reply = self.send(op, **fields)
        assert "error" not in reply, reply
        return reply["ok"]


@contextmanager
def split_controller(root: Path) -> Iterator[SplitController]:
    root.mkdir(mode=0o700)
    process = subprocess.Popen([str(TARGET / "debug/examples/fixture_splits"), str(root)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        assert process.stdin is not None
        process.stdin.write("{}\n")
        process.stdin.flush()
        yield SplitController(process)
        process.stdin.close()
        assert process.wait(timeout=5) == 0
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()


def split_request(welcome: dict[str, Any], *, labels: bool = False) -> dict[str, Any]:
    return {key: welcome[key] for key in (
        "schema_version", "contract", "revision", "binding", "worker", "lease_id", "split", "panel",
    )} | {"item": welcome["items"][0], "labels": labels}


@contextmanager
def deadline(seconds: int = 85) -> Iterator[None]:
    """Per-test bound without an optional pytest plugin; never turn a timeout into a skip."""
    def expired(signum: int, frame: Any) -> None:
        raise TimeoutError("T043 exceeded its 85-second test budget")

    old = signal.signal(signal.SIGALRM, expired)
    start = time.monotonic()
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
        assert time.monotonic() - start < 90, "T043 individual test exceeded 90 seconds"
