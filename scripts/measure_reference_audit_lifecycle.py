"""Measure three 24-job fixture audits; never change production limits or claim qualification."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def _write(path: Path, document: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _snapshot(directory: Path, phase: str) -> None:
    commands = {
        "windows": ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                    "Get-Process | Select-Object Id,ProcessName,CPU,WorkingSet64 | ConvertTo-Json -Compress"],
        "linux": ["ps", "-eo", "pid,comm,pcpu,rss"],
        "gpu": ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,memory.used",
                "--format=csv,noheader,nounits"],
    }
    for name, command in commands.items():
        result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
        with (directory / f"processes-{phase}-{name}.txt").open("x", encoding="utf-8") as stream:
            stream.write(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measurement-wall-ms", type=int, required=True)
    parser.add_argument("--load-label", choices=("non_idle_tradingview_running",), required=True)
    args = parser.parse_args()
    if not 7200 < args.measurement_wall_ms <= 80000:
        parser.error("measurement-only wall allowance must be explicitly supplied within existing native bounds")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tests"))
    from neural_foundry.gym_service import _document_digest
    from test_gym_reference_episode import audit, setup

    args.output.mkdir(parents=True, exist_ok=False)
    runs = []
    for index in range(1, 4):
        directory = args.output / f"run-{index}"
        directory.mkdir()
        receipts = []
        with tempfile.TemporaryDirectory(prefix="nf-lifecycle-measurement-") as temporary:
            fixture = Path(temporary)
            gym, plan, sandbox, _, policies, _, _ = setup(fixture)
            _snapshot(directory, "before")
            started = time.monotonic_ns()
            report = audit(gym, plan, sandbox, policies, budget={
                "max_executions": 24, "cpu_ms": 24000,
                "wall_ms": args.measurement_wall_ms, "storage_bytes": 512 * 1024 * 1024,
            })
            elapsed_ns = time.monotonic_ns() - started
            _write(directory / "audit-report.json", report)
            _snapshot(directory, "after")
            receipt_dir = directory / "receipts"
            receipt_dir.mkdir()
            for digest in report["sandbox_receipts"]:
                raw = (fixture / "store" / "blobs" / digest.removeprefix("sha256:")).read_bytes()
                with (receipt_dir / f"{digest.removeprefix('sha256:')}.json").open("xb") as stream:
                    stream.write(raw)
                receipt = json.loads(raw)
                receipts.append(receipt)
        complete = (report["executions"] == 24 and len(set(report["sandbox_receipts"])) == 24
                    and report["status"] == "probe_passed"
                    and all(receipt["status"] == "completed" and receipt["cleanup_confirmed"] is True
                            for receipt in receipts))
        guardian_ms = sum(receipt["report"]["wall_ms"] for receipt in receipts)
        if guardian_ms != report["wall_ms"] or elapsed_ns < guardian_ms * 1_000_000:
            raise RuntimeError("measurement timing/accounting differs")
        row = {
            "run": index, "load_label": args.load_label,
            "requested_jobs": 24, "completed_attempt": complete,
            "jobs_executed": report["executions"], "audit_status": report["status"],
            "host_elapsed_ns": elapsed_ns, "guardian_wall_ms": guardian_ms,
            "lifecycle_ns": elapsed_ns - guardian_ms * 1_000_000,
            "preflight_timings": report["preflight_timings"],
            "audit_report_digest": _document_digest(report),
            "production_acceptance": False,
        }
        _write(directory / "measurement.json", row)
        runs.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    all_complete = all(row["completed_attempt"] for row in runs)
    complete_lifecycles = [row["lifecycle_ns"] for row in runs if row["completed_attempt"]]
    worst = max(complete_lifecycles) if complete_lifecycles else None
    summary = {
        "kind": "nf_reference_lifecycle_measurement_v1",
        "load_label": args.load_label,
        "idle_machine_verified": False,
        "runs": runs,
        "all_three_24_job_runs_completed": all_complete,
        "measurement_only_wall_allowance_ms": args.measurement_wall_ms,
        "per_job_wall_ms": 300,
        "production_limits_changed": False,
        "known_failure_markers_removed": False,
        "worst_lifecycle_ns": worst,
        "proposed_lifecycle_limit_ms_at_25_percent_headroom": (
            math.ceil(worst * 1.25 / 1_000_000) if all_complete and worst is not None else None),
        "proposal_blocker": None if all_complete else "incomplete 24-job attempts under unchanged per-job limits",
        "proposal_approved": False,
        "timing_scope": "host elapsed around native audit, including preflight and persistence, minus summed guardian wall_ms; excludes fixture construction and process snapshots",
        "guardian_quantization_uncertainty_ms_upper_bound": 24,
        "holdout_accessed": False,
    }
    summary["report_digest"] = _document_digest(summary)
    _write(args.output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    if not all_complete:
        raise RuntimeError("three-attempt measurement is incomplete; no lifecycle limit proposed")


if __name__ == "__main__":
    main()
