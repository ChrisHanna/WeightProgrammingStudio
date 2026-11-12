"""Durable generic command receipts around the CORD custom-registry entrypoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from neural_foundry.execution_runner import run_logged_worker, write_status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("export", "identity", "score", "audit"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--role")
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    log_dir = args.log_dir.resolve()
    status_file = log_dir.with_name(log_dir.name + "-status.json")
    if not args.python.is_file():
        parser.error("chosen cached runtime is missing; downloads are forbidden")
    if log_dir.exists() or status_file.exists():
        parser.error("choose fresh log/status paths; existing receipts cannot be overwritten")
    if args.stage != "audit" and not args.role:
        parser.error("--role is required")
    command = [str(args.python), str(project / "scripts" / "run_cord_receipts.py"),
               args.stage, "--spec", str(args.spec.resolve()), "--root", str(args.root.resolve())]
    if args.role:
        command.extend(["--role", args.role])
    environment = {**os.environ, "PYTHONPATH": str(project / "python"),
                   "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
                   "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    write_status(status_file, state="waiting_on_job", step=f"{args.stage}:{args.role or 'run'}",
                 job_receipt=log_dir / "receipt.json")
    result = run_logged_worker(command, cwd=project, log_dir=log_dir,
                               timeout_seconds=args.timeout_seconds, env=environment)
    write_status(status_file, state="done" if result.returncode == 0 and not result.timed_out
                 else "blocked", step=f"{args.stage}:{args.role or 'run'}",
                 job_receipt=log_dir / "receipt.json")
    print(json.dumps(result.receipt, sort_keys=True))
    raise SystemExit(result.returncode if not result.timed_out else 124)


if __name__ == "__main__":
    main()
