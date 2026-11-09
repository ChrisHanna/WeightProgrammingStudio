"""Local worker execution and process-tree cleanup."""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import time
import traceback
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


def run_logged_worker(command: Sequence[str], *, cwd: str | Path, log_dir: str | Path,
                      timeout_seconds: float, env: Mapping[str, str] | None = None) -> WorkerResult:
    if not command or any(not isinstance(value, str) or not value for value in command):
        raise ValueError("logged worker requires a nonempty command")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("logged worker timeout must be finite and positive")
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=False)
    stdout_path, stderr_path = directory / "stdout.log", directory / "stderr.log"
    started = time.monotonic()
    receipt = {"kind": "nf_logged_worker_receipt_v1", "command": list(command),
               "cwd": str(Path(cwd).resolve()), "status": "started", "returncode": None,
               "timed_out": False, "stdout_path": str(stdout_path), "stderr_path": str(stderr_path)}
    with (directory / "started.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True)
    process = None
    error = None
    with stdout_path.open("xb") as stdout_stream, stderr_path.open("xb") as stderr_stream:
        try:
            process = subprocess.Popen(list(command), cwd=cwd, env=env,
                stdout=stdout_stream, stderr=stderr_stream, start_new_session=os.name != "nt",
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
            try:
                process.wait(timeout=timeout_seconds)
                receipt["status"] = "completed" if process.returncode == 0 else "failed"
            except subprocess.TimeoutExpired:
                receipt["timed_out"] = True
                receipt["status"] = "timed_out"
            finally:
                stop_process_tree(process)
        except BaseException as exception:
            error = exception
            receipt["status"] = "launch_failed" if process is None else "interrupted"
            receipt["exception_type"] = type(exception).__name__
            stderr_stream.write(traceback.format_exc().encode("utf-8"))
        finally:
            if process is not None:
                receipt["returncode"] = process.returncode
            receipt["duration_seconds"] = time.monotonic() - started
            stdout_stream.flush()
            stderr_stream.flush()
    stdout, stderr = stdout_path.read_bytes(), stderr_path.read_bytes()
    receipt.update({"stdout_digest": "sha256:" + hashlib.sha256(stdout).hexdigest(),
                    "stderr_digest": "sha256:" + hashlib.sha256(stderr).hexdigest()})
    with (directory / "receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True)
    if error is not None:
        raise error
    return WorkerResult(process.returncode, receipt["timed_out"], receipt, stdout, stderr)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run a local worker with durable full logs and exit receipt")
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=7200)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = arguments.command[1:] if arguments.command[:1] == ["--"] else arguments.command
    result = run_logged_worker(command, cwd=Path.cwd(), log_dir=arguments.log_dir,
                               timeout_seconds=arguments.timeout_seconds)
    print(json.dumps(result.receipt, sort_keys=True))
    raise SystemExit(result.returncode if result.returncode >= 0 else 128 - result.returncode)


if __name__ == "__main__":
    main()