"""Manifest-driven public proof operations over the existing Gym lifecycle."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from .gym_job import GymJobError, ScenarioSource, is_finite_number, validate_task_instance
from .gym_lifecycle import AdapterIdentity, EvaluatorAdapter, GymLifecycle, TaskAdapter
from .gym_service import GymStudioService, _document_digest
from .proof_registry import ProofRegistry, builtin_registry


def read_document(path: Path, field: str = "document_digest") -> dict[str, Any]:
    document = json.loads(path.read_bytes())
    if document.get(field) != _document_digest({key: value for key, value in document.items() if key != field}):
        raise GymJobError(f"document digest differs: {path.name}")
    return document


def write_document(path: Path, document: Mapping[str, Any], field: str = "document_digest") -> dict[str, Any]:
    payload = dict(document)
    payload[field] = _document_digest(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    return payload


def validate_spec(spec: Mapping[str, Any], registry: ProofRegistry) -> None:
    if spec.get("kind") != "nf_public_proof_run" or spec.get("version") != 1:
        raise GymJobError("unsupported public proof run spec")
    if spec.get("download_allowed") is not False or spec.get("holdout_accessed") is not False:
        raise GymJobError("public proof forbids downloads and holdout access")
    if "policy_reference" in spec:
        from .proof_policy import require_bound_policy

        require_bound_policy(spec)
    model = spec.get("base_model", {})
    if not all(isinstance(model.get(key), str) and model[key] for key in ("model_id", "revision")):
        raise GymJobError("run spec requires an explicit pinned base model")
    roles = spec.get("candidates")
    if (not isinstance(roles, list) or not roles
        or any(not isinstance(row, Mapping) or not isinstance(row.get("role"), str)
               or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", row["role"]) is None for row in roles)
        or len({row["role"] for row in roles}) != len(roles)):
        raise GymJobError("candidate roles must be unique manifest data")
    policy = spec.get("policy", {})
    if (policy.get("version") != 1 or type(policy.get("max_new_tokens")) is not int
        or type(policy.get("context_tokens")) is not int
        or not 1 <= policy["max_new_tokens"] < policy["context_tokens"]
        or policy.get("report_only") is not True or policy.get("adaptive_retraining") is not False
        or policy.get("identity", {}).get("version") != 1
        or policy["identity"].get("probe_count") != 16
        or policy["identity"].get("logprob_atol") != 1e-6
        or policy.get("api", {}).get("version") != 1
        or any(type(policy["api"].get(key)) is not int or policy["api"][key] < 1
               for key in ("maximum_api_attempts", "maximum_cost_microusd"))):
        raise GymJobError("invalid versioned public proof decoding policy")
    plugins = spec.get("plugins", {})
    if not {"gym", "model", "trainer", "training_audit", "exporter", "augmenter", "public_audit"} <= plugins.keys():
        raise GymJobError("missing public proof plugins")
    for kind, reference in plugins.items():
        registry.resolve(kind, reference)
    rows = spec.get("training_rows", [])
    if (not isinstance(rows, list) or len({row["scenario_id"] for row in rows}) != len(rows)
        or any(not row["scenario_id"].startswith("train:") for row in rows)
        or spec.get("training_rows_digest") != _document_digest({"rows": rows})):
        raise GymJobError("training requires unique ordered rows")
    panels = spec.get("public_sets", [])
    if len({panel["name"] for panel in panels}) != len(panels):
        raise GymJobError("public set names must be unique")
    for panel in panels:
        public = panel.get("rows", [])
        if (not public or any(not row["scenario_id"].startswith("validation:") for row in public)
            or len({row["scenario_id"] for row in public}) != len(public)
            or panel.get("rows_digest") != _document_digest({"rows": public})):
            raise GymJobError("public panels require complete frozen validation rows")
    for candidate in roles:
        if ("system_prompt" in candidate
            and (not isinstance(candidate["system_prompt"], str) or not candidate["system_prompt"].strip())):
            raise GymJobError("candidate system_prompt must be nonempty text")
        if ("public_sets" in candidate and (not candidate["public_sets"]
            or not set(candidate["public_sets"]) <= {panel["name"] for panel in panels})):
            raise GymJobError("candidate public sets must select existing frozen panels")
        recipe = candidate.get("recipe")
        if recipe is not None:
            if not rows or recipe.get("method") != candidate.get("method"):
                raise GymJobError("candidate recipe method differs")
            coverage = spec.get("coverage_policy", "complete_ordered_pass")
            if coverage == "complete_ordered_pass" and (recipe.get("steps") != len(rows) or len(rows) > 512):
                raise GymJobError("one approved ordered pass must consume every row within 512 steps")
            if coverage not in {"complete_ordered_pass", "legacy_partial_ordered_pass"}:
                raise GymJobError("unknown training coverage policy")
        source = candidate.get("execution")
        if source is not None:
            registry.resolve(source["kind"], source["plugin"])
            registry.resolve("evidence", candidate["evidence"])
            if source["kind"] not in {"server", "provider"}:
                raise GymJobError("unsupported candidate execution contract")
    for gate in spec.get("gates", []):
        panel = next((panel for panel in panels if panel["name"] == gate["set"]), None)
        if panel is None or type(gate.get("minimum")) is not int or gate["minimum"] < 0:
            raise GymJobError("gate must bind an existing public set and integer success count")
        eligible = [row for row in panel["rows"]
                    if gate.get("category") is None or row.get("panel_category") == gate["category"]]
        if gate["minimum"] > len(eligible):
            raise GymJobError("gate exceeds its eligible case count")


def audit(spec: Mapping[str, Any], registry: ProofRegistry) -> dict[str, Any]:
    result = registry.resolve("public_audit", spec["plugins"]["public_audit"])(spec=spec)
    if result.get("public_passed") is not True:
        raise GymJobError("public contamination gate failed")
    return result


def candidate_task(task: TaskAdapter, candidate: Mapping[str, Any]) -> TaskAdapter:
    if "system_prompt" not in candidate:
        return task
    from .prompt_rendering import prompt_specification
    from .provenance import sha256_file

    system_prompt = candidate["system_prompt"]
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise GymJobError("candidate system_prompt must be nonempty text")

    def project(input_value: Any, response: Any) -> Mapping[str, Any]:
        interaction = prompt_specification(task.project(input_value, response), has_chat_template=True)
        if interaction["format"] == "raw_text_v1":
            raise GymJobError("candidate system_prompt requires a chat interaction")
        messages = interaction["messages"]
        if messages[0]["role"] == "system":
            messages[0]["content"] += "\n\n" + system_prompt
        else:
            messages.insert(0, {"role": "system", "content": system_prompt})
        return interaction

    digest = _document_digest({"base_projection": task.projection.document(),
                              "system_prompt": system_prompt,
                              "implementation_digest": sha256_file(__file__)})
    return replace(task, project=project,
                   projection=AdapterIdentity(task.projection.adapter_id + "-candidate-system", "1", digest))


def training_task(task: Any, rows: list[dict[str, Any]]) -> Any:
    """Bind immutable train rows as a Gym scenario plugin, not a second dataset path."""
    digest = _document_digest({"rows": rows})
    # Gym jobs declare all split roles; this provider refuses every non-train request.
    source = ScenarioSource("proof-train", "proof-frozen-rows", "1", digest, digest,
                            "simulated", ("train", "validation", "holdout"), {})
    job = replace(task.job, scenario_sources=(source,))

    def bind(service: GymStudioService) -> None:
        task.bind(service)

        def provider(*, split: str, **_kwargs: Any) -> list[dict[str, Any]]:
            if split != "train":
                raise GymJobError("frozen training source is train-only")
            fields = {"scenario_id", "input", "reference", "group_id", "source_group_id",
                      "item_id", "family_id", "interaction_type"}
            return [{key: value for key, value in row.items() if key in fields} for row in rows]

        service.components.register("scenario_provider", source.provider_id, "1", digest, provider)

    return replace(task, job=job, bind=bind)


def training_coverage(roster: list[str], receipt: Mapping[str, Any],
                      config: Mapping[str, Any], steps: int) -> dict[str, Any]:
    if not roster or receipt.get("committed_steps") != list(range(steps)):
        raise GymJobError("training did not commit the complete ordered pass")
    if (config["dataloader"].get("shuffle") is not False
        or config["step_scheduler"].get("global_batch_size") != 1
        or config["step_scheduler"].get("local_batch_size") != 1
        or config["step_scheduler"].get("max_steps") != steps
        or config["packed_sequence"].get("packed_sequence_size") != 0):
        raise GymJobError("training schedule does not prove ordered row consumption")
    consumed = [roster[index % len(roster)] for index in range(steps)]
    return {"ordered_scenario_ids": roster, "committed_steps": receipt["committed_steps"],
            "per_step_scenario_ids": consumed, "unique_rows_consumed": len(set(consumed)),
            "complete": set(consumed) == set(roster),
            "evidence": "schedule_derived_from_shuffle_false_batch_one_no_packing_and_committed_step_log",
            "direct_per_batch_id_telemetry": False}


def train(spec: Mapping[str, Any], *, role: str, root: Path, automodel_python: str,
          reload_python: str, registry: ProofRegistry | None = None) -> dict[str, Any]:
    registry = registry or builtin_registry()[0]
    validate_spec(spec, registry)
    public_audit = audit(spec, registry)
    selected = next((row for row in spec["candidates"] if row["role"] == role), None)
    if selected is None or selected.get("recipe") is None:
        raise GymJobError("role has no approved training recipe")
    output = root / role
    output.mkdir(parents=True, exist_ok=False)
    task = candidate_task(registry.resolve("gym", spec["plugins"]["gym"])(spec["gym"]), selected)
    task = training_task(task, spec["training_rows"])
    policy = spec["policy"]
    base = registry.resolve("model", spec["plugins"]["model"])(
        base_model=spec["base_model"], policy=policy, seed=selected["recipe"]["seed"])
    lifecycle = GymLifecycle(GymStudioService(state_dir=output / "state"))
    evaluator = EvaluatorAdapter(AdapterIdentity("public-deferred", "1", task.identity.implementation_digest),
                                lambda *_args: {"split": "validation", "complete": False, "metrics": {},
                                                "status": "deferred_to_verified_public_server"})
    trainer = registry.resolve("trainer", spec["plugins"]["trainer"])(
        automodel_python=automodel_python, reload_python=reload_python)
    for adapter in (task, base, evaluator, trainer):
        lifecycle.register(adapter)
    run = lifecycle.run({"run_id": spec["run_id"] + ":" + role,
        "task": task.identity.adapter_id + "@" + task.identity.version,
        "model": base.identity.adapter_id + "@" + base.identity.version,
        "trainer": trainer.identity.adapter_id + "@" + trainer.identity.version,
        "evaluator": evaluator.identity.adapter_id + "@" + evaluator.identity.version,
        "recipe": selected["recipe"],
        "approved_responses": {row["scenario_id"]: row["reference"] for row in spec["training_rows"]}})
    examples = lifecycle.service.load_training_examples(run["dataset"]["dataset_digest"])
    roster = [row["scenario_id"] for row in examples]
    if roster != [row["scenario_id"] for row in spec["training_rows"]]:
        raise GymJobError("persisted training order differs from frozen roster")
    artifact = Path(run["candidate_record"]["artifact_root"]) / "artifact"
    steps = selected["recipe"]["steps"]
    coverage = registry.resolve("training_audit", spec["plugins"]["training_audit"])(
        artifact=artifact, roster=roster, steps=steps)
    if spec.get("coverage_policy", "complete_ordered_pass") == "complete_ordered_pass" and not coverage["complete"]:
        raise GymJobError("not every frozen training row was consumed")
    return write_document(output / "training-result.json", {
        "kind": "nf_public_proof_training_result_v1", "run_spec_digest": _document_digest(spec),
        "preparation_digest": spec.get("preparation_digest"),
        "method": selected["method"], "role": role, "base": asdict(base.reference),
        "revision": spec["base_model"]["revision"], "candidate_record": run["candidate_record"],
        "training_request": run["training_request"], "dataset_digest": run["dataset"]["dataset_digest"],
        "evaluation_status": run["evaluation_status"],
        "public_audit_digest": _document_digest(public_audit),
        "coverage": coverage,
        "holdout_accessed": False, "provider_called": False, "release_authorized": False}, "result_digest")


def score_public_rows(task: Any, model: Any, rows: list[dict[str, Any]], *,
                      progress: dict[str, int] | None = None,
                      on_response: Callable[[Mapping[str, Any], Mapping[str, Any]], None] | None = None,
                      ) -> dict[str, Any]:
    """Resolve the Gym's own scorer rather than importing a task-specific scorer."""
    from .gym_service import ExecutionResponse

    service = GymStudioService()
    task.bind(service)
    component = task.job.scorer
    scorer = service.components.resolve("scorer", component.component_id,
                                        component.version, component.implementation_digest)
    records = []
    for row in rows:
        validate_task_instance(task.job.task_input_schema, row["input"], field="input")
        interaction = task.project(row["input"], None)
        if progress is not None:
            progress["public_inference_dispatches"] += 1
        output = model.infer(interaction)
        if progress is not None:
            progress["public_responses_returned"] += 1
        response = output.response if isinstance(output, ExecutionResponse) else output
        evidence = dict(output.interaction) if isinstance(output, ExecutionResponse) else {}
        try:
            validate_task_instance(task.job.task_output_schema, response, field="response")
            valid = not (evidence.get("hit_generation_cap", False)
                         or evidence.get("invalid_completion_json", False))
        except GymJobError:
            valid = False
        metrics = scorer(reference=row["reference"], response=response, output_valid=valid)
        if (not isinstance(metrics, Mapping) or not metrics
            or any(not isinstance(name, str) or not is_finite_number(value)
                   for name, value in metrics.items())):
            raise GymJobError("Gym scorer must return named metrics")
        if not valid:
            metrics = {name: 0.0 for name in metrics}
        record = {"scenario_id": row["scenario_id"], "response": response,
                  "format_valid": bool(valid), "execution_evidence": evidence,
                  "metrics": dict(metrics)}
        if on_response is not None:
            on_response(row, record)
        records.append(record)
        if progress is not None:
            progress["scored_responses"] += 1
    names = records[0]["metrics"].keys()
    if any(record["metrics"].keys() != names for record in records):
        raise GymJobError("Gym scorer changed metric names within a public set")
    return {"cases": len(rows), "responses": records,
            "metrics": {name: sum(record["metrics"][name] for record in records) / len(rows) for name in names},
            "rows_digest": _document_digest({"rows": rows})}


def score(spec: Mapping[str, Any], *, role: str, root: Path,
          registry: ProofRegistry | None = None,
          retry_of_failure: Path | None = None) -> dict[str, Any]:
    from .gym_service import ExecutionResponse

    registry = registry or builtin_registry()[0]
    validate_spec(spec, registry)
    selected = next((row for row in spec["candidates"] if row["role"] == role), None)
    if selected is None or "execution" not in selected:
        raise GymJobError("candidate has no declared execution plugin")
    task = candidate_task(registry.resolve("gym", spec["plugins"]["gym"])(spec["gym"]), selected)
    execution = selected["execution"]
    factory = registry.resolve(execution["kind"], execution["plugin"])
    evidence = registry.resolve("evidence", selected["evidence"])
    retry_digest = None
    if (root / f"public-score-{role}.json").exists():
        raise GymJobError("completed public scores cannot be replaced")
    if retry_of_failure is not None:
        failure = read_document(retry_of_failure)
        if (execution["kind"] != "server" or spec["policy"]["api"].get("calls_authorized") is not False
            or failure.get("role") != role or failure.get("complete") is not False
            or failure.get("run_spec_digest") != _document_digest(spec)):
            raise GymJobError("explicit local infrastructure retry requires the matching failed receipt")
        counters = ("public_inference_dispatches", "public_responses_returned", "scored_responses")
        if (failure.get("attempt_evidence_version") != 2
            or failure.get("failed_stage") not in {"server_startup", "pre_inference"}
            or any(type(failure.get(key)) is not int or failure[key] != 0 for key in counters)):
            raise GymJobError("infrastructure retry requires explicit pre-inference failure and zero responses")
        claim_name = failure.get("attempt_claim_name")
        if (not isinstance(claim_name, str) or Path(claim_name).name != claim_name
            or not claim_name.startswith(f"public-score-{role}") or not claim_name.endswith(".started.json")
            or (root / claim_name).is_symlink()):
            raise GymJobError("infrastructure retry requires a preserved attempt claim")
        try:
            previous_claim = read_document(root / claim_name)
        except OSError as error:
            raise GymJobError("infrastructure retry attempt claim is missing") from error
        if (previous_claim["document_digest"] != failure.get("attempt_claim_digest")
            or previous_claim.get("run_spec_digest") != _document_digest(spec)
            or previous_claim.get("role") != role):
            raise GymJobError("infrastructure retry attempt claim differs")
        retry_digest = failure["document_digest"]
    attempt = "" if retry_digest is None else "-retry-" + retry_digest[7:23]
    stage = "server_startup"
    progress = {"public_inference_dispatches": 0, "public_responses_returned": 0,
                "scored_responses": 0}
    claim_name = f"public-score-{role}{attempt}.started.json"
    claim_digest = None
    try:
        claim = write_document(root / claim_name, {
            "role": role, "run_spec_digest": _document_digest(spec), "report_only": True,
            "retry_of_failure_digest": retry_digest,
            "holdout_accessed": False, "provider_calls_authorized": spec["policy"]["api"].get("calls_authorized", False)})
        claim_digest = claim["document_digest"]
        if execution["kind"] == "provider":
            server = factory(spec=spec, candidate=selected, root=root, task=task)
        else:
            server = factory(spec=spec, candidate=selected, root=root)
        with server as model:
            stage = "identity_verification"
            evidence(model=model, policy=spec["policy"])
            scores = {}
            for panel in spec["public_sets"]:
                if "public_sets" in selected and panel["name"] not in selected["public_sets"]:
                    continue
                rows = panel["rows"]
                stage = "public_inference"

                def persist_response(row: Mapping[str, Any], record: Mapping[str, Any],
                                     public_set: str = panel["name"],
                                     set_rows_digest: str = panel["rows_digest"]) -> None:
                    write_document(root / f"public-score-{role}{attempt}-responses"
                                   / f"{progress['scored_responses']:06d}.json", {
                        "kind": "nf_public_proof_response_record_v1",
                        "run_spec_digest": _document_digest(spec), "public_set": public_set,
                        "public_set_rows_digest": set_rows_digest,
                        "row_digest": _document_digest(row), "response_record": dict(record)})

                result = score_public_rows(task, model, rows, progress=progress,
                                           on_response=persist_response)
                for row, response in zip(rows, result["responses"], strict=True):
                    evidence(model=model, policy=spec["policy"], interaction=task.project(row["input"], None),
                             result=ExecutionResponse(response["response"], response["execution_evidence"]))
                scores[panel["name"]] = result
            if callable(getattr(server, "verify_end_identity", None)):
                stage = "end_identity_verification"
                server.verify_end_identity()
            identity = dict(model.loaded_identity())
        gates = {}
        for gate in spec["gates"]:
            if gate["set"] not in scores:
                continue
            responses = scores[gate["set"]]["responses"]
            if gate.get("category") is not None:
                panel = next(value for value in spec["public_sets"] if value["name"] == gate["set"])
                indices = [index for index, row in enumerate(panel["rows"])
                           if row.get("panel_category") == gate["category"]]
                count = sum(responses[index]["metrics"][gate["metric"]] >= gate.get("success_value", 1.0)
                            for index in indices)
            else:
                count = sum(response["metrics"][gate["metric"]] >= gate.get("success_value", 1.0)
                            for response in responses)
            gates[gate["name"]] = {"observed": count, "minimum": gate["minimum"], "passed": count >= gate["minimum"]}
        return write_document(root / f"public-score-{role}.json", {
            "kind": "nf_public_proof_score_v1", "role": role, "complete": True,
            "run_spec_digest": _document_digest(spec), "scores": scores, "gates": gates,
            "prompt_projection": task.projection.document(),
            "response_journal": {"path": str(root / f"public-score-{role}{attempt}-responses"),
                                 "records": progress["scored_responses"]},
            "retry_of_failure_digest": retry_digest,
            "gates_complete": len(gates) == len(spec["gates"]),
            "gate_passed": bool(gates) and len(gates) == len(spec["gates"])
                           and all(value["passed"] for value in gates.values()),
            "model_identity": identity, "report_only": True, "holdout_accessed": False,
            "release_authorized": False})
    except Exception as error:
        write_document(root / f"public-score-{role}-failure-{uuid4().hex}.json", {
            "role": role, "run_spec_digest": _document_digest(spec),
            "failed_stage": stage, "retry_of_failure_digest": retry_digest,
            "attempt_evidence_version": 2, "attempt_claim_name": claim_name,
            "attempt_claim_digest": claim_digest, **progress,
            "complete": False, "failure_type": type(error).__name__,
            "holdout_accessed": False, "release_authorized": False})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "export", "identity", "score", "audit", "budget-init"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--role")
    parser.add_argument("--automodel-python")
    parser.add_argument("--reload-python")
    parser.add_argument("--retry-of-failure", type=Path)
    args = parser.parse_args()
    args.spec = args.spec.resolve()
    args.root = args.root.resolve()
    if args.stage not in {"audit", "budget-init"} and not args.role:
        parser.error("this stage requires a manifest candidate --role")
    if args.stage == "train" and (not args.automodel_python or not args.reload_python):
        parser.error("training requires existing --automodel-python and --reload-python runtimes")
    spec = read_document(args.spec)
    spec.pop("document_digest")
    registry, _ = builtin_registry()
    validate_spec(spec, registry)
    if args.stage == "budget-init":
        from .public_budget import initialize_public_budget

        result = initialize_public_budget(spec)
    elif args.stage == "train":
        result = train(spec, role=args.role, root=args.root, automodel_python=args.automodel_python,
                       reload_python=args.reload_python, registry=registry)
    elif args.stage == "export":
        result = registry.resolve("exporter", spec["plugins"]["exporter"])(spec=spec, role=args.role, root=args.root)
    elif args.stage == "identity":
        candidate = next(row for row in spec["candidates"] if row["role"] == args.role)
        result = registry.resolve("server", candidate["execution"]["plugin"])(
            spec=spec, candidate=candidate, root=args.root, freeze_identity=True)
    elif args.stage == "audit":
        result = registry.resolve("public_audit", spec["plugins"]["public_audit"])(spec=spec)
        write_document(args.root / "public-audit.json", result)
        if not result["public_passed"]:
            raise GymJobError("public contamination gate failed")
    else:
        result = score(spec, role=args.role, root=args.root, registry=registry,
                       retry_of_failure=args.retry_of_failure)
    print(json.dumps({key: value for key, value in result.items()
                      if key in {"kind", "role", "complete", "gate_passed", "public_passed", "result_digest"}}, sort_keys=True))


if __name__ == "__main__":
    main()
