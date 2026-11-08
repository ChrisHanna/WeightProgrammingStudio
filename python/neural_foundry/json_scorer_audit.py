"""Basic counterfactual and determinism audit for the synthetic reference scorer."""

from __future__ import annotations

import json
from typing import Any

from .gym_job import GymJobError
from .gym_lifecycle import TaskAdapter
from .gym_service import GymStudioService, _document_digest, _wire_document
from .reference_json_extraction import generate_scenarios, reference_task


def audit_reference_scorer(task: TaskAdapter) -> dict[str, Any]:
    if not isinstance(task, TaskAdapter) or len(task.job.scenario_sources) != 1:
        raise GymJobError("scorer audit requires the reference JSON task")
    count = task.job.scenario_sources[0].parameters.get("rows_per_template")
    if task.job.digest != reference_task(rows_per_template=count).job.digest:
        raise GymJobError("scorer audit job differs from the frozen reference job")
    service = GymStudioService()
    task.bind(service)
    scorer = service.components.resolve("scorer", task.job.scorer.component_id, task.job.scorer.version,
                                        task.job.scorer.implementation_digest)
    verifier = service.components.resolve("verifier", task.job.verifier.component_id, task.job.verifier.version,
                                          task.job.verifier.implementation_digest)
    checked = 0
    for split in ("train", "validation"):
        for row in generate_scenarios(split=split, seed=7, parameters={"rows_per_template": 1}):
            reference = row["reference"]
            wrong = {**reference, "quantity": reference["quantity"] % 99 + 1}
            for response, valid, expected in ((reference, True, 1.0), (wrong, True, 0.0), (None, False, 0.0)):
                frozen = {"scenario": row, "reference": reference, "response": response}
                snapshot = _wire_document(frozen)
                outcomes = []
                for _ in range(2):
                    arguments = json.loads(snapshot)
                    verification = verifier(parameters=task.job.verifier.to_document()["parameters"],
                                            **arguments, output_valid=valid)
                    metrics = scorer(parameters=task.job.scorer.to_document()["parameters"], **arguments,
                                     verification=verification, output_valid=valid)
                    if _wire_document(arguments) != snapshot:
                        raise GymJobError("scorer audit detected input mutation")
                    outcomes.append(metrics)
                if (outcomes[0] != outcomes[1] or outcomes[0] != {"exact_match": expected}
                    or type(outcomes[0].get("exact_match")) not in (int, float)):
                    raise GymJobError("scorer audit failed a determinism or counterfactual check")
                checked += 1
    report = {"kind": "nf_json_scorer_audit_v1", "job_digest": task.job.digest,
              "scorer_digest": task.job.scorer.implementation_digest, "cases": checked,
              "determinism_repetitions": 2, "splits": ["train", "validation"],
              "passed": True, "evidence_class": "synthetic", "release_authorized": False}
    report["audit_digest"] = _document_digest(report)
    return report