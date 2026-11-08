"""Local worker execution and process-tree cleanup."""

from __future__ import annotations

import hashlib
import math
import os
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WorkerResult:
    returncode: int
    timed_out: bool
    receipt: dict[str, Any]
    stdout: bytes
    stderr: bytes


def stop_process_tree(process: subprocess.Popen[Any]) -> None:
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True, text=True, check=False, timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            process.kill()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        if process.poll() is None:
            process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        process.wait()


def run_local_worker(
    command: Sequence[str], *, cwd: str | Path, timeout_seconds: float,
    receipt_command: Sequence[str] | None = None,
    cleanup_descendants: bool = False,
    env: Mapping[str, str] | None = None,
) -> WorkerResult:
    if not command or any(not isinstance(value, str) or not value for value in command):
        raise ValueError("worker command must contain nonempty strings")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("worker timeout must be finite and positive")
    recorded_command = list(command if receipt_command is None else receipt_command)
    if len(recorded_command) != len(command) or any(not isinstance(value, str) for value in recorded_command):
        raise ValueError("receipt command must match the worker command shape")
    started = time.monotonic()
    process = subprocess.Popen(
        list(command), cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
        start_new_session=os.name != "nt",
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        stop_process_tree(process)
        stdout, stderr = process.communicate()
    except BaseException:
        stop_process_tree(process)
        process.communicate()
        raise
    else:
        if cleanup_descendants:
            stop_process_tree(process)
    receipt = {
        "kind": "nf_local_worker_receipt_v1", "placement": "local", "command": recorded_command,
        "returncode": process.returncode, "timed_out": timed_out,
        "duration_seconds": time.monotonic() - started,
        "stdout_digest": "sha256:" + hashlib.sha256(stdout).hexdigest(),
        "stderr_digest": "sha256:" + hashlib.sha256(stderr).hexdigest(),
    }
    return WorkerResult(process.returncode, timed_out, receipt, stdout, stderr)