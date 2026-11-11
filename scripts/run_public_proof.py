"""Run one manifest-selected public stage with bounded, durable command receipts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from neural_foundry.execution_runner import run_logged_worker, write_status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "export", "identity", "score", "audit", "budget-init"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--role")
    parser.add_argument("--automodel-python", type=Path)
    parser.add_argument("--reload-python", type=Path)
    parser.add_argument("--retry-of-failure", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    args = parser.parse_args()
    args.spec = args.spec.resolve()
    args.root = args.root.resolve()
    args.log_dir = args.log_dir.resolve()
    if not args.python.is_file():
        parser.error("the chosen cached runtime is missing; downloads are forbidden")
    command = [str(args.python), "-m", "neural_foundry.proof_pipeline", args.stage,
               "--spec", str(args.spec), "--root", str(args.root)]
    for name in ("role", "automodel_python", "reload_python", "retry_of_failure"):
        value = getattr(args, name)
        if value is not None:
            command.extend(["--" + name.replace("_", "-"), str(value)])
    project = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "PYTHONPATH": str(project / "python"),
                   "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
                   "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    status_file = args.log_dir.with_name(args.log_dir.name + "-status.json").resolve()
    if args.log_dir.exists() or status_file.exists():
        parser.error("choose fresh log and per-job status paths; existing receipts cannot be overwritten")
    write_status(status_file, state="waiting_on_job", step=f"{args.stage}:{args.role or 'run'}",
                 job_receipt=args.log_dir / "receipt.json")
    result = run_logged_worker(command, cwd=project, log_dir=args.log_dir,
                               timeout_seconds=args.timeout_seconds, env=environment)
    write_status(status_file, state="done" if result.returncode == 0 and not result.timed_out else "blocked",
                 step=f"{args.stage}:{args.role or 'run'}", job_receipt=args.log_dir / "receipt.json")
    print(json.dumps(result.receipt, sort_keys=True))
    raise SystemExit(result.returncode if not result.timed_out else 124)


if __name__ == "__main__":
    main()
