"""Digest-bound scoring worker for the synthetic JSON reference job."""

from __future__ import annotations

import json
import math
import sys
from typing import Any, Mapping

from .gym_job import GymJobError, gym_job_from_document, validate_task_instance
from .gym_service import GymStudioService
from .provenance import parse_strict_json
from .records import canonicalize, content_digest
from .reference_json_extraction import reference_task


def score_request(request: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(request, Mapping) or set(request) != {"job", "scenario", "reference", "response_text"}:
        raise GymJobError("scoring request has missing or unknown fields")
    job = gym_job_from_document(request["job"])
    if len(job.scenario_sources) != 1:
        raise GymJobError("unsupported JSON scorer job")
    count = job.scenario_sources[0].parameters.get("rows_per_template")
    expected = reference_task(rows_per_template=count)
    if job.digest != expected.job.digest:
        raise GymJobError("scoring job or implementation digest is not the registered reference job")
    scenario = request["scenario"]
    if not isinstance(scenario, Mapping) or scenario.get("split") not in {"train", "validation"}:
        raise GymJobError("scoring worker accepts only public scenarios")
    reference = request["reference"]
    digest = str(content_digest(canonicalize({"kind": "nf_gym_reference_v1", "reference": reference})))
    if digest != scenario.get("reference_digest"):
        raise GymJobError("scoring reference does not match the scenario")
    validate_task_instance(job.task_input_schema, scenario.get("input"), field="scenario.input")
    validate_task_instance(job.task_output_schema, reference, field="reference")
    text = request["response_text"]
    if not isinstance(text, str) or len(text.encode("utf-8")) > 65536:
        raise GymJobError("response text is not bounded")
    try:
        response = parse_strict_json(text.encode("utf-8"))
        validate_task_instance(job.task_output_schema, response, field="response")
        output_valid = True
    except (ValueError, UnicodeError):
        response, output_valid = None, False
    service = GymStudioService()
    expected.bind(service)
    scorer = service.components.resolve("scorer", job.scorer.component_id, job.scorer.version,
                                        job.scorer.implementation_digest)
    verifier = service.components.resolve("verifier", job.verifier.component_id, job.verifier.version,
                                          job.verifier.implementation_digest)
    verification = verifier(parameters=job.verifier.to_document()["parameters"], scenario=scenario,
                            reference=reference, response=response, output_valid=output_valid)
    metrics = scorer(parameters=job.scorer.to_document()["parameters"], scenario=scenario,
                     reference=reference, response=response, verification=verification, output_valid=output_valid)
    score = metrics.get("exact_match") if isinstance(metrics, Mapping) else None
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise GymJobError("registered scorer returned an invalid exact-match metric")
    return {"score": float(score), "output_valid": output_valid, "scorer_digest": job.scorer.implementation_digest}


def main() -> None:
    raw = sys.stdin.buffer.read(1_048_577)
    if len(raw) > 1_048_576:
        raise GymJobError("scoring request exceeds its byte limit")
    print(json.dumps(score_request(parse_strict_json(raw)), allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    main()