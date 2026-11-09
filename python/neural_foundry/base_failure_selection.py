"""Digest-bound selection of failing examples from a measured public base run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .gym_job import GymJobError
from .gym_service import _document_digest
from .provenance import parse_strict_json
from .reference_json_extraction import generate_scenarios

MINIMUM_BASE_FAILURES = 10


def load_base_failure_report(path: str | Path, *, task, seed: int, rows_per_template: int,
                             base_state_digest: str | None = None,
                             base_candidate_digest: str | None = None) -> tuple[dict[str, Any], list[str], list[str]]:
    report = parse_strict_json(Path(path).read_bytes())
    if (report.get("kind") != "nf_json_public_split_baseline_v1"
        or report.get("evidence_digest") != _document_digest({key: value for key, value in report.items() if key != "evidence_digest"})
        or report.get("current_job_digest") != task.job.digest or report.get("seed") != seed
        or report.get("rows_per_template") != rows_per_template or report.get("scenario_origin") != "synthetic"
        or report.get("holdout_accessed") is not False):
        raise GymJobError("base-failure report differs from the frozen public task, seed, or split")
    if base_state_digest is not None and report.get("base_state_digest") != base_state_digest:
        raise GymJobError("base-failure report belongs to another loaded model state")
    if base_candidate_digest is not None and report.get("base_candidate_digest") != base_candidate_digest:
        raise GymJobError("base-failure report belongs to another candidate artifact")
    expected_rows = generate_scenarios(split="train", seed=seed, parameters={"rows_per_template": rows_per_template})
    scored = report.get("train", {}).get("score", {}).get("responses")
    if not isinstance(scored, list) or [row.get("scenario_id") for row in scored] != [row["scenario_id"] for row in expected_rows]:
        raise GymJobError("base-failure report has an incomplete or reordered train roster")
    if any(row.get("exact_match") not in (0.0, 1.0) for row in scored):
        raise GymJobError("base-failure report has non-binary exact-match outcomes")
    failures = [row["scenario_id"] for row in scored if row["exact_match"] == 0.0]
    passing = [row["scenario_id"] for row in scored if row["exact_match"] == 1.0]
    if (report.get("train", {}).get("failure_count") != len(failures)
        or report.get("train", {}).get("failures") != failures):
        raise GymJobError("base-failure report failure roster differs from its scored cases")
    if len(failures) < MINIMUM_BASE_FAILURES:
        raise GymJobError(f"refusing to edit: base has {len(failures)} failing train cases; minimum is {MINIMUM_BASE_FAILURES}")
    return report, failures, passing
