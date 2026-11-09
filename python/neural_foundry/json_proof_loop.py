"""Synthetic three-model JSON extraction comparison without custody claims."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Any

from .gym_job import GymJobError
from .gym_lifecycle import TaskAdapter
from .gym_service import ExecutionResponse, _document_digest
from .reference_json_extraction import generate_scenarios, reference_task, score_response


def compare_synthetic_holdout(
    task: TaskAdapter, *, seed: int, models: Mapping[str, Callable[[], AbstractContextManager[Any]]],
    preservation_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if not isinstance(task, TaskAdapter) or len(task.job.scenario_sources) != 1:
        raise GymJobError("synthetic proof requires the reference JSON task")
    count = task.job.scenario_sources[0].parameters.get("rows_per_template")
    if task.job.digest != reference_task(rows_per_template=count).job.digest:
        raise GymJobError("synthetic proof task differs from the frozen reference job")
    rows = generate_scenarios(split="holdout", seed=seed, parameters={"rows_per_template": count})
    panel_digest = _document_digest({"kind": "nf_synthetic_holdout_v1", "job_digest": task.job.digest,
                                    "seed": seed, "scenarios": rows})
    report: dict[str, Any] = {
        "kind": "nf_json_synthetic_proof_v1", "job_digest": task.job.digest, "panel_digest": panel_digest,
        "split": "holdout", "evidence_class": "synthetic", "scenario_count": len(rows),
        "independent_custody_verified": False, "served_model_identity_verified": False,
        "release_authorized": False, "complete": False, "scorecards": None,
    }
    keys = set(models) if isinstance(models, Mapping) else set()
    allowed_rosters = ({"base", "sft", "frontier"}, {"base", "sft", "weight_edit"}, {"base", "sft", "weight_edit", "frontier"})
    roles = tuple(role for role in ("base", "sft", "weight_edit", "frontier") if role in keys)
    if keys not in allowed_rosters or any(not callable(models[role]) for role in roles):
        return {**report, "failure": {"stage": "preflight", "reason": "a complete declared model roster is required"}}
    preservation = list(preservation_rows or [])
    if len(preservation) > 128:
        raise GymJobError("preservation panel exceeds its case limit")
    if preservation:
        report["preservation_panel_digest"] = _document_digest({"kind": "nf_public_preservation_panel_v1", "rows": preservation})
        report["preservation_count"] = len(preservation)
    responses: dict[str, list[Any]] = {}
    preserved_responses: dict[str, list[Any]] = {}
    identities: dict[str, Any] = {}
    for role in roles:
        try:
            with models[role]() as model:
                identity = model.loaded_identity()
                if not isinstance(identity, Mapping):
                    raise GymJobError("model identity must be a mapping")
                identities[role] = dict(identity)
                outputs = []
                for row in rows:
                    result = model.infer(task.project(row["input"], None))
                    outputs.append(result.response if isinstance(result, ExecutionResponse) else result)
                responses[role] = outputs
                preserved = []
                for row in preservation:
                    result = model.infer(task.project(row["input"], None))
                    preserved.append(result.response if isinstance(result, ExecutionResponse) else result)
                preserved_responses[role] = preserved
        except Exception as exc:
            return {**report, "failure": {"stage": role, "reason": type(exc).__name__}}
    scorecards = {}
    preservation_scores = {role: [score_response(reference=row["reference"], response=response)["exact_match"]
                                  for row, response in zip(preservation, preserved_responses[role], strict=True)] for role in roles}
    for role in roles:
        total = sum(score_response(reference=row["reference"], response=response)["exact_match"]
                    for row, response in zip(rows, responses[role], strict=True))
        scorecards[role] = {"metric": "exact_match", "value": total / len(rows), "scored": len(rows),
                            "model_identity": identities[role], "evidence_class": "synthetic"}
        if preservation:
            scorecards[role]["preservation_exact_match"] = sum(preservation_scores[role]) / len(preservation)
            scorecards[role]["newly_broken_cases"] = sum(base == 1.0 and observed == 0.0
                for base, observed in zip(preservation_scores["base"], preservation_scores[role], strict=True))
    report.update({"complete": True, "scorecards": scorecards})
    report["report_digest"] = _document_digest(report)
    return report