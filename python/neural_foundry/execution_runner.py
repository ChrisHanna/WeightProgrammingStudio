"""Local worker execution and process-tree cleanup."""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WorkerResult:
    returncode: int
    timed_out: bool
    receipt: dict[str, Any]
    stdout: bytes
    stderr: bytes


STATUS_STATES = ("working", "waiting_on_job", "waiting_for_user", "done", "blocked")


def write_status(path: str | Path, *, state: str, step: str, job_receipt: str | Path | None = None,
                 next_step: str | None = None, needs_from_user: str | None = None,
                 started: str | None = None) -> dict[str, Any]:
    if state not in STATUS_STATES or not isinstance(step, str) or not step.strip():
        raise ValueError("status requires a supported state and nonempty step")
    updated = datetime.now(timezone.utc).isoformat()
    document = {"state": state, "step": step, "started": started or updated,
                "job_receipt": str(job_receipt) if job_receipt is not None else None,
                "next": next_step, "needs_from_user": needs_from_user, "updated": updated}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                         prefix=".status-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return document


def wait_for_receipt(log_dir: str | Path, *, timeout_seconds: float, poll_seconds: float = 30,
                     on_wait: Callable[[], None] | None = None) -> dict[str, Any]:
    if any(not math.isfinite(value) or value <= 0 for value in (timeout_seconds, poll_seconds)):
        raise ValueError("receipt wait timeout and polling interval must be finite and positive")
    deadline = time.monotonic() + timeout_seconds
    path = Path(log_dir) / "receipt.json"
    while True:
        if on_wait is not None:
            on_wait()
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            receipt = None
        if isinstance(receipt, dict):
            duration = receipt.get("duration_seconds")
            code = receipt.get("returncode")
            if (receipt.get("kind") == "nf_logged_worker_receipt_v1"
                and receipt.get("status") in {"completed", "failed", "timed_out", "launch_failed", "interrupted"}
                and (type(code) is int or (code is None and receipt["status"] == "launch_failed"))
                and type(receipt.get("timed_out")) is bool
                and type(duration) in (int, float) and math.isfinite(duration) and duration >= 0
                and (receipt["status"] != "completed" or (code == 0 and not receipt["timed_out"]))):
                return receipt
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"receipt did not complete before timeout: {path}")
        time.sleep(min(poll_seconds, remaining))


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
                      timeout_seconds: float, env: Mapping[str, str] | None = None,
                      on_wait: Callable[[], None] | None = None) -> WorkerResult:
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
                deadline = time.monotonic() + timeout_seconds
                while True:
                    if on_wait is not None:
                        on_wait()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(list(command), timeout_seconds)
                    try:
                        process.wait(timeout=min(30, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        if time.monotonic() >= deadline:
                            raise
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
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=7200)
    parser.add_argument("--wait", action="store_true", help="Wait in the foreground for an existing job receipt")
    parser.add_argument("--status-file", type=Path, default=Path("artifacts/STATUS.json"))
    parser.add_argument("--step", default="local worker")
    parser.add_argument("--next", dest="next_step")
    parser.add_argument("--state", choices=STATUS_STATES, help="Write status only; do not launch or wait")
    parser.add_argument("--needs-from-user")
    parser.add_argument("--job-receipt", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = arguments.command[1:] if arguments.command[:1] == ["--"] else arguments.command
    if arguments.state:
        if arguments.wait or command:
            parser.error("status-only mode cannot launch or wait")
        print(json.dumps(write_status(arguments.status_file, state=arguments.state, step=arguments.step,
            job_receipt=arguments.job_receipt, next_step=arguments.next_step,
            needs_from_user=arguments.needs_from_user), sort_keys=True))
        return
    if arguments.log_dir is None or (arguments.wait and command) or (not arguments.wait and not command):
        parser.error("provide --log-dir and either --wait or a worker command")
    if not math.isfinite(arguments.timeout_seconds) or arguments.timeout_seconds <= 0:
        parser.error("timeout must be finite and positive")
    started = datetime.now(timezone.utc).isoformat()

    def status(state: str, needs: str | None = None) -> None:
        write_status(arguments.status_file, state=state, step=arguments.step,
            job_receipt=arguments.log_dir / "receipt.json", next_step=arguments.next_step,
            needs_from_user=needs, started=started)

    try:
        status("waiting_on_job")
        if arguments.wait:
            receipt = wait_for_receipt(arguments.log_dir, timeout_seconds=arguments.timeout_seconds,
                                       on_wait=lambda: status("waiting_on_job"))
        else:
            result = run_logged_worker(command, cwd=Path.cwd(), log_dir=arguments.log_dir,
                timeout_seconds=arguments.timeout_seconds, on_wait=lambda: status("waiting_on_job"))
            receipt = result.receipt
    except BaseException as error:
        status("blocked", type(error).__name__)
        if isinstance(error, TimeoutError):
            print(json.dumps({"status": "wait_timed_out", "returncode": 124, "duration_seconds":
                              arguments.timeout_seconds}, sort_keys=True))
            raise SystemExit(124) from None
        raise
    successful = receipt["status"] == "completed" and receipt["returncode"] == 0 and not receipt["timed_out"]
    status("done" if successful else "blocked")
    print(json.dumps(receipt, sort_keys=True))
    code = receipt["returncode"]
    exit_code = (code if code >= 0 else 128 - code) if type(code) is int else 1
    raise SystemExit(0 if successful else (exit_code or 1))


if __name__ == "__main__":
    main()