"""Synthetic three-model JSON extraction comparison without custody claims."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from typing import Any

from .gym_job import GymJobError
from .gym_lifecycle import TaskAdapter
from .gym_service import ExecutionResponse, _document_digest
from .reference_json_extraction import generate_scenarios, reference_task, score_response


def compare_synthetic_holdout(
    task: TaskAdapter, *, seed: int, models: Mapping[str, Callable[[], AbstractContextManager[Any]]],
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
    roles = ("base", "sft", "frontier")
    if not isinstance(models, Mapping) or set(models) != set(roles) or any(not callable(models[role]) for role in roles):
        return {**report, "failure": {"stage": "preflight", "reason": "three model factories are required"}}
    responses: dict[str, list[Any]] = {}
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
        except Exception as exc:
            return {**report, "failure": {"stage": role, "reason": type(exc).__name__}}
    scorecards = {}
    for role in roles:
        total = sum(score_response(reference=row["reference"], response=response)["exact_match"]
                    for row, response in zip(rows, responses[role], strict=True))
        scorecards[role] = {"metric": "exact_match", "value": total / len(rows), "scored": len(rows),
                            "model_identity": identities[role], "evidence_class": "synthetic"}
    report.update({"complete": True, "scorecards": scorecards})
    report["report_digest"] = _document_digest(report)
    return report