from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier, Event
from typing import Any

import pytest
from neural_foundry.gym_job import (
    AcceptanceGate,
    CandidateMethod,
    ComponentReference,
    GymComponentRegistry,
    GymJobError,
    GymJobSpec,
    ScenarioSource,
    gym_job_from_document,
    load_gym_job_spec,
    validate_task_instance,
)
from neural_foundry.gym_service import (
    CandidateIdentity,
    GymStudioService,
    PracticeAttemptRecord,
)


def test_nullable_fields_keep_required_structure_and_nonnull_constraints() -> None:
    schema = {"type": "object", "properties": {
        "text": {"type": ["string", "null"], "minLength": 2},
        "parent": {"type": ["integer", "null"], "minimum": 0},
    }, "required": ["text", "parent"], "additionalProperties": False}
    for value in ({"text": None, "parent": None}, {"text": "line", "parent": 0}):
        validate_task_instance(schema, value)
    for value in ({"text": "x", "parent": None}, {"text": None, "parent": -1},
                  {"text": None, "parent": True}, {"text": None}, {"text": [], "parent": 0}):
        with pytest.raises(GymJobError):
            validate_task_instance(schema, value)
    validate_task_instance({"type": "null", "enum": [None]}, None)
    with pytest.raises(GymJobError, match="null"):
        validate_task_instance({"type": "null"}, "")


def test_type_unions_apply_each_types_constraints_and_enums() -> None:
    schema = {"type": ["integer", "string"], "minimum": 0, "minLength": 2, "enum": [3, "ok"]}
    validate_task_instance(schema, 3)
    validate_task_instance(schema, "ok")
    for value in (-1, 4, "x", "other", None, True):
        with pytest.raises(GymJobError, match="declared types"):
            validate_task_instance(schema, value)
    nested = {"type": ["array", "null"], "minItems": 1, "items": {"type": ["string", "null"]}}
    validate_task_instance(nested, [None, "item"])
    validate_task_instance(nested, None)
    with pytest.raises(GymJobError):
        validate_task_instance(nested, [])


@pytest.mark.parametrize("kind", [[], ["string", "string"], ["unknown"], ["null", {}], [True]])
def test_schema_type_unions_reject_malformed_declarations(kind) -> None:
    with pytest.raises(GymJobError, match="unique types"):
        validate_task_instance({"type": kind}, None)


def _identity(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _contract_review_job() -> GymJobSpec:
    return GymJobSpec(
        job_id="vendor-contract-review",
        title="Find renewal and termination clauses",
        description="Identify clause text and evidence spans; do not invent missing terms.",
        task_input_schema={
            "type": "object",
            "properties": {
                "contract_text": {"type": "string", "maxLength": 16384},
                "vendor_id": {"type": "string", "maxLength": 128},
            },
            "required": ["contract_text"],
            "additionalProperties": False,
        },
        task_output_schema={
            "type": "object",
            "properties": {
                "clauses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "clause_type": {"type": "string", "enum": ["renewal", "termination"]},
                            "text": {"type": "string", "maxLength": 4096},
                            "start": {"type": "integer", "minimum": 0},
                            "end": {"type": "integer", "minimum": 0},
                        },
                        "required": ["clause_type", "text", "start", "end"],
                        "additionalProperties": False,
                    },
                    "maxItems": 32,
                },
                "abstained": {"type": "boolean"},
            },
            "required": ["clauses", "abstained"],
            "additionalProperties": False,
        },
        scenario_sources=(
            ScenarioSource(
                source_id="approved-vendor-corpus",
                provider_id="frozen_jsonl",
                provider_version="1.0.0",
                provider_digest=_identity("scenario_provider:frozen_jsonl:1.0.0"),
                source_digest=_identity("dataset:approved-vendor-corpus:v1"),
                origin="real",
                split_roles=("train", "validation", "holdout"),
                parameters={"dataset_digest": "sha256:" + "a" * 64},
            ),
            ScenarioSource(
                source_id="synthetic-contracts",
                provider_id="contract_template_generator",
                provider_version="2.1.0",
                provider_digest=_identity("scenario_provider:contract_template_generator:2.1.0"),
                source_digest=_identity("generator:contract-templates:v2"),
                origin="simulated",
                split_roles=("train", "validation"),
                parameters={"seed": 31, "templates": ["renewal", "termination"]},
            ),
        ),
        scorer=ComponentReference(
            "contract_clause_evidence", "3.0.0", _identity("scorer:contract_clause_evidence:3.0.0"),
            {"required_clause_types": ["renewal", "termination"]},
        ),
        verifier=ComponentReference(
            "citation_grounding", "1.2.0", _identity("verifier:citation_grounding:1.2.0"),
            {"require_exact_source_spans": True, "allow_abstention": True},
        ),
        candidate_methods=(
            CandidateMethod("unchanged_base", "1.0.0", _identity("candidate_method:unchanged_base:1.0.0"), {}),
            CandidateMethod("lora", "1.0.0", _identity("candidate_method:lora:1.0.0"), {"ranks": [4, 8], "targets": ["all_down_proj"]}),
            CandidateMethod("qlora", "0.1.0", _identity("candidate_method:qlora:0.1.0"), {"quantization": "nf4", "rank": 8}),
            CandidateMethod("wsp", "1.0.0", _identity("candidate_method:wsp:1.0.0"), {"max_changed_parameters": 100000}),
        ),
        gates=(
            AcceptanceGate("task-quality", "task", "clause_f1", "validation", "ge", 0.9),
            AcceptanceGate("no-invention", "behavior", "unsupported_clause_rate", "holdout", "le", 0.0),
            AcceptanceGate("matched-controls", "controls", "beats_placebo", "controls", "eq", 1.0),
            AcceptanceGate("reproducibility", "reproducibility", "repeat_digest_match", "reproducibility", "eq", 1.0),
        ),
        metadata={"owner": "legal-operations", "data_classification": "confidential"},
    )


def _registry(*, scorer_version: str = "3.0.0") -> GymComponentRegistry:
    registry = GymComponentRegistry()
    for kind, component_id, version in (
        ("scenario_provider", "frozen_jsonl", "1.0.0"),
        ("scenario_provider", "contract_template_generator", "2.1.0"),
        ("scorer", "contract_clause_evidence", scorer_version),
        ("verifier", "citation_grounding", "1.2.0"),
        ("candidate_method", "lora", "1.0.0"),
        ("candidate_method", "qlora", "0.1.0"),
        ("candidate_method", "wsp", "1.0.0"),
        ("candidate_method", "unchanged_base", "1.0.0"),
    ):
        registry.register(
            kind,
            component_id,
            version,
            _identity(f"{kind}:{component_id}:{version}"),
            lambda *_args, **_kwargs: None,
        )
    return registry


def test_job_spec_roundtrips_with_versioned_real_and_simulated_components(tmp_path: Path) -> None:
    job = _contract_review_job()
    document = job.to_document()
    restored = gym_job_from_document(document)
    path = tmp_path / "contract-review.job.json"
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    loaded = load_gym_job_spec(path)

    assert restored.digest == job.digest == loaded.digest
    assert loaded.scenario_sources[0].origin == "real"
    assert loaded.scenario_sources[1].origin == "simulated"
    assert {method.method_id for method in loaded.candidate_methods} == {"unchanged_base", "lora", "qlora", "wsp"}
    _registry().validate_job(loaded)


def test_component_registry_requires_exact_registered_versions() -> None:
    job = _contract_review_job()
    registry = _registry(scorer_version="4.0.0")
    with pytest.raises(GymJobError, match="unregistered scorer.*3.0.0"):
        registry.validate_job(job)


def test_job_backend_selectors_are_versioned_and_digest_bound() -> None:
    original = _contract_review_job()
    assert "backends" not in original.to_document()
    job = replace(original, backends={"trainer": "nemo_rl.grpo@1", "evaluator": "nemo_gym.custody@1"})
    assert gym_job_from_document(job.to_document()).digest == job.digest != original.digest
    for backends in ({"trainer": "nemo_rl.grpo"}, {"trainer": "nemo_rl.grpo@"}, {"other": "bad@1"}):
        with pytest.raises(GymJobError, match="backends"):
            replace(original, backends=backends)


def test_job_spec_rejects_unknown_fields_duplicate_keys_and_invalid_schemas() -> None:
    document = _contract_review_job().to_document()
    document["runtime_hook"] = "module:callback"
    with pytest.raises(GymJobError, match="unknown fields"):
        gym_job_from_document(document)

    raw = json.dumps(_contract_review_job().to_document()).replace(
        '"schema_version": 1', '"schema_version": 1, "schema_version": 1', 1
    )
    from neural_foundry.gym_job import _reject_duplicate_keys

    with pytest.raises(GymJobError, match="duplicate JSON field"):
        json.loads(raw, object_pairs_hook=_reject_duplicate_keys)

    document = _contract_review_job().to_document()
    document["task"]["input_schema"]["properties"]["contract_text"]["$ref"] = "file:///etc/passwd"
    with pytest.raises(GymJobError, match="unsupported schema fields"):
        gym_job_from_document(document)


def test_job_spec_rejects_nonfinite_parameters_and_gate_thresholds() -> None:
    job = _contract_review_job()
    with pytest.raises(GymJobError, match="non-finite"):
        ComponentReference("bad", "1", _identity("scorer:bad:1"), {"threshold": float("nan")})
    with pytest.raises(GymJobError, match="finite numeric"):
        AcceptanceGate("bad", "task", "metric", "validation", "ge", float("inf"))
    with pytest.raises(GymJobError, match="finite numeric"):
        AcceptanceGate("bad", "task", "metric", "validation", "ge", True)  # type: ignore[arg-type]
    assert job.digest.startswith("sha256:")


@pytest.mark.parametrize("bound", [-1, "16", True, float("nan")])
def test_job_spec_rejects_invalid_string_length_bounds(bound: Any) -> None:
    document = _contract_review_job().to_document()
    document["task"]["input_schema"]["properties"]["contract_text"]["maxLength"] = bound
    with pytest.raises(GymJobError, match="maxLength must be a non-negative integer"):
        gym_job_from_document(document)


def test_task_schema_validates_instances_without_coercion_or_extra_fields() -> None:
    schema = _contract_review_job().task_input_schema
    validate_task_instance(schema, {"contract_text": "Renewal term", "vendor_id": "v-1"})
    with pytest.raises(GymJobError, match="undeclared properties"):
        validate_task_instance(schema, {"contract_text": "Renewal term", "owner": "extra"})
    with pytest.raises(GymJobError, match="required properties"):
        validate_task_instance(schema, {"vendor_id": "v-1"})
    with pytest.raises(GymJobError, match="must be a string"):
        validate_task_instance(schema, {"contract_text": True})


def test_service_applies_same_job_gates_and_never_authorizes_release() -> None:
    service = GymStudioService(_registry())
    job = _contract_review_job()
    candidate = CandidateIdentity(
        candidate_id="contract-lora-r8",
        method_id="lora",
        base_model_digest="sha256:" + "a" * 64,
        candidate_model_digest="sha256:" + "b" * 64,
        artifact_digest="sha256:" + "c" * 64,
    )
    measurements = {
        "task-quality": {
            "metric_id": "clause_f1", "split": "validation", "value": 0.94,
            "evidence_digest": "sha256:" + "d" * 64, "evidence_class": "measured",
        },
        "no-invention": {
            "metric_id": "unsupported_clause_rate", "split": "holdout", "value": 0.0,
            "evidence_digest": "sha256:" + "e" * 64, "evidence_class": "measured",
        },
        "matched-controls": {
            "metric_id": "beats_placebo", "split": "controls", "value": 1.0,
            "evidence_digest": "sha256:" + "f" * 64, "evidence_class": "measured",
        },
        "reproducibility": {
            "metric_id": "repeat_digest_match", "split": "reproducibility", "value": 1.0,
            "evidence_digest": "sha256:" + "1" * 64, "evidence_class": "measured",
        },
    }
    scorecard = service.score_candidate(job, candidate, measurements)
    assert scorecard["decision"] == "reported_pass_pending_verification"
    assert all(gate["status"] == "reported_pass" for gate in scorecard["gates"])
    assert scorecard["release_authorized"] is False
    assert scorecard["authority"]["metric_inputs_authenticated"] is False
    assert scorecard["serving_identity"]["status"] == "unverified"
    assert scorecard["authority"]["served_model_identity_verified"] is False

    missing = service.score_candidate(job, candidate, {"task-quality": measurements["task-quality"]})
    assert missing["decision"] == "incomplete"
    assert sum(gate["status"] == "not_run" for gate in missing["gates"]) == 3

    wrong_split = {**measurements, "task-quality": {**measurements["task-quality"], "split": "train"}}
    blocked = service.score_candidate(job, candidate, wrong_split)
    assert blocked["decision"] == "incomplete"
    assert blocked["gates"][0]["status"] == "blocked"

    malformed_evidence = {
        **measurements,
        "task-quality": {**measurements["task-quality"], "evidence_class": ["measured"]},
    }
    malformed = service.score_candidate(job, candidate, malformed_evidence)
    assert malformed["decision"] == "incomplete"
    assert malformed["gates"][0]["status"] == "blocked"


def test_service_materializes_public_scenarios_and_scores_attempts(tmp_path: Path) -> None:
    job = _contract_review_job()
    registry = GymComponentRegistry()

    def provider(
        *, parameters: dict[str, Any], source_id: str, split: str, seed: int, **_identity: Any
    ) -> list[dict[str, Any]]:
        return [{
            "scenario_id": f"{source_id}-{split}-{seed}",
            "input": {"contract_text": "The agreement renews annually.", "vendor_id": "v-1"},
            "reference": {
                "clauses": [{
                    "clause_type": "renewal",
                    "text": "renews annually",
                    "start": 14,
                    "end": 29,
                }]
            },
        }]

    def verifier(
        *, parameters: dict[str, Any], scenario: dict[str, Any], reference: dict[str, Any],
        response: dict[str, Any], output_valid: bool,
    ) -> dict[str, Any]:
        return {"passed": output_valid and response["clauses"] == reference["clauses"], "reason": "source-span check"}

    def scorer(
        *, parameters: dict[str, Any], scenario: dict[str, Any], reference: dict[str, Any],
        response: dict[str, Any], output_valid: bool, verification: dict[str, Any],
    ) -> dict[str, float]:
        return {"clause_f1": 1.0 if verification["passed"] else 0.0, "unsupported_clause_rate": 0.0}

    for source in job.scenario_sources:
        registry.register(
            "scenario_provider", source.provider_id, source.provider_version, source.provider_digest, provider
        )
    registry.register("scorer", job.scorer.component_id, job.scorer.version, job.scorer.implementation_digest, scorer)
    registry.register("verifier", job.verifier.component_id, job.verifier.version, job.verifier.implementation_digest, verifier)
    for method in job.candidate_methods:
        registry.register(
            "candidate_method", method.method_id, method.version, method.implementation_digest,
            lambda **_kwargs: None,
        )
    state_dir = tmp_path / "gym-state"
    service = GymStudioService(registry, state_dir=state_dir)
    candidate = CandidateIdentity(
        "contract-lora-r8", "lora", _identity("base"), _identity("candidate"), _identity("artifact")
    )

    batch = service.materialize_public_scenarios(job, split="train", seed=17)
    assert len(batch["scenarios"]) == 2
    plan = service.freeze_practice_plan(
        job,
        candidate,
        run_id="contract-practice-001",
        scenario_manifest_digest=batch["manifest_digest"],
        repetitions=2,
        execution_kind="registered_candidate", executor_digest=_identity("executor:contract-lora-r8:v1"),
    )
    assert len(plan["expected_attempts"]) == 4
    with pytest.raises(GymJobError, match="not in the frozen practice roster"):
        service.score_attempt(
            job,
            candidate,
            practice_plan_digest=plan["plan_digest"],
            attempt_id="unplanned-attempt",
            response={"clauses": [], "abstained": True},
        )
    response = {
        "clauses": [{"clause_type": "renewal", "text": "renews annually", "start": 14, "end": 29}],
        "abstained": False,
    }
    executor_digest = _identity("executor:contract-lora-r8:v1")
    executor_batches: list[tuple[dict[str, Any], ...]] = []

    def execute_batch(*, candidate: dict[str, Any], scenarios: tuple[dict[str, Any], ...]) -> dict[str, Any]:
        assert candidate["candidate_id"] == candidate_id.candidate_id
        assert all("reference" not in scenario for scenario in scenarios)
        executor_batches.append(scenarios)
        return {scenario["attempt_id"]: response for scenario in scenarios}

    candidate_id = candidate
    service.register_candidate_executor(
        candidate,
        executor_digest=executor_digest,
        execute=execute_batch,
    )
    first = plan["expected_attempts"][0]
    first_batch_ids = tuple(row["attempt_id"] for row in plan["expected_attempts"][:2])
    first_batch = service.run_practice_batch(
        job,
        candidate,
        practice_plan_digest=plan["plan_digest"],
        attempt_ids=first_batch_ids,
    )
    attempt = first_batch[0]
    assert attempt.status == "scored"
    assert attempt.verifier_passed and attempt.output_valid
    assert attempt.execution_kind == "registered_candidate"
    assert attempt.executor_digest == executor_digest
    assert set(attempt.metrics) == {"clause_f1", "unsupported_clause_rate"}
    assert len(executor_batches) == 1 and len(executor_batches[0]) == 2
    partial = service.record_practice_run(plan)
    assert partial["status"] == "incomplete"
    assert len(partial["expected_attempt_ids"]) == 4
    assert len(partial["attempts"]) == 2
    resumed_service = GymStudioService(registry, state_dir=state_dir)
    resumed_service.register_candidate_executor(
        candidate, executor_digest=executor_digest, execute=execute_batch
    )
    recovered = resumed_service.record_practice_run(plan)
    assert recovered["status"] == "incomplete"
    assert recovered["attempts"] == partial["attempts"]
    with pytest.raises(GymJobError, match="terminal state"):
        resumed_service.score_attempt(
            job,
            candidate,
            practice_plan_digest=plan["plan_digest"],
            attempt_id=first["attempt_id"],
            response=response,
        )
    resumed_service.run_practice_batch(
        job,
        candidate,
        practice_plan_digest=plan["plan_digest"],
        attempt_ids=tuple(row["attempt_id"] for row in plan["expected_attempts"][2:]),
    )
    complete = resumed_service.record_practice_run(plan)
    assert complete["kind"] == "nf_gym_practice_run_v1"
    assert complete["status"] == "complete"
    assert len(complete["attempts"]) == len(complete["expected_attempt_ids"]) == 4
    assert "response" not in complete["attempts"][0]
    assert complete["run_digest"].startswith("sha256:")
    with pytest.raises(GymJobError, match="native independent-evaluation custodian"):
        service.materialize_public_scenarios(job, split="holdout", seed=17)

    failed_candidate = CandidateIdentity(
        "contract-lora-failing", "lora", _identity("base"), _identity("failed-model"), _identity("failed-artifact")
    )
    failed_plan = resumed_service.freeze_practice_plan(
        job,
        failed_candidate,
        run_id="contract-practice-failed",
        scenario_manifest_digest=batch["manifest_digest"],
        repetitions=1,
        execution_kind="registered_candidate", executor_digest=_identity("executor:failing:v1"),
    )
    transient_calls: list[int] = []

    def transient_executor(*, scenarios: Any, **kwargs: Any) -> dict[str, Any]:
        transient_calls.append(1)
        if len(transient_calls) == 1:
            raise RuntimeError("temporary CUDA failure")
        return {row["attempt_id"]: response for row in scenarios}

    resumed_service.register_candidate_executor(
        failed_candidate,
        executor_digest=_identity("executor:failing:v1"),
        execute=transient_executor,
    )
    failed = resumed_service.run_practice_batch(
        job,
        failed_candidate,
        practice_plan_digest=failed_plan["plan_digest"],
        attempt_ids=tuple(row["attempt_id"] for row in failed_plan["expected_attempts"]),
    )
    failed_run = resumed_service.record_practice_run(failed_plan)
    assert failed == () and failed_run["status"] == "incomplete"
    assert len(failed_run["infrastructure_errors"]) == len(failed_plan["expected_attempts"])
    resumed_service.run_practice_batch(
        job, failed_candidate, practice_plan_digest=failed_plan["plan_digest"],
        attempt_ids=tuple(row["attempt_id"] for row in failed_plan["expected_attempts"]),
    )
    assert resumed_service.record_practice_run(failed_plan)["status"] == "complete"

    concurrent_plan = resumed_service.freeze_practice_plan(
        job, candidate, run_id="concurrent-workers",
        scenario_manifest_digest=batch["manifest_digest"], repetitions=1,
    )
    workers = [GymStudioService(registry, state_dir=state_dir) for _ in range(2)]
    barrier = Barrier(2)

    def persist_from_stale_worker(index: int) -> None:
        workers[index]._load_frozen_plan(concurrent_plan["plan_digest"])
        expected = concurrent_plan["expected_attempts"][index]
        saved = replace(
            attempt, attempt_id=expected["attempt_id"], scenario_id=expected["scenario_id"],
            scenario_digest=expected["scenario_digest"], scenario_origin=expected["scenario_origin"],
            scenario_split=expected["scenario_split"], round_number=expected["repetition"],
        )
        barrier.wait(timeout=10)
        workers[index]._persist_terminal_attempt(concurrent_plan["plan_digest"], saved)
        workers[index].record_practice_run(concurrent_plan)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(persist_from_stale_worker, range(2)))
    recovered = GymStudioService(registry, state_dir=state_dir).record_practice_run(concurrent_plan)
    assert recovered["status"] == "complete" and len(recovered["attempts"]) == 2
    first_saved = workers[0]._terminal_attempts[concurrent_plan["plan_digest"]][
        concurrent_plan["expected_attempts"][0]["attempt_id"]
    ]
    with pytest.raises(GymJobError, match="differs from its terminal scored result"):
        workers[0].record_practice_run(
            concurrent_plan, (replace(first_saved, response_digest=_identity("changed-response")),),
        )
    with pytest.raises(GymJobError, match="not server-recorded"):
        workers[0].record_practice_run(
            concurrent_plan, (replace(first_saved, attempt_id="foreign-attempt"),),
        )
    legacy_plan = resumed_service.freeze_practice_plan(
        job, candidate, run_id="legacy-state-import",
        scenario_manifest_digest=batch["manifest_digest"], repetitions=1,
    )
    legacy_attempts = tuple(
        replace(
            attempt, attempt_id=row["attempt_id"], scenario_id=row["scenario_id"],
            scenario_digest=row["scenario_digest"], scenario_origin=row["scenario_origin"],
            scenario_split=row["scenario_split"], round_number=row["repetition"],
        )
        for row in legacy_plan["expected_attempts"]
    )
    with pytest.raises(GymJobError, match="does not match its frozen run plan"):
        resumed_service._persist_terminal_attempt(
            legacy_plan["plan_digest"], replace(legacy_attempts[0], round_number=99),
        )
    resumed_service._persist_state(
        "attempts", legacy_plan["plan_digest"][7:],
        {"attempts": [row.to_document() for row in legacy_attempts]},
    )
    migrated = GymStudioService(registry, state_dir=state_dir).record_practice_run(legacy_plan)
    assert migrated["status"] == "complete" and len(migrated["attempts"]) == 2
    assert migrated["execution_summary"]["registered_execution_complete"] is False


def test_unchanged_bases_are_scored_on_the_same_public_panel(tmp_path: Path) -> None:
    job = _contract_review_job()
    registry = GymComponentRegistry()

    def provider(*, source_id: str, split: str, seed: int, **_kwargs: Any) -> list[dict[str, Any]]:
        return [{
            "scenario_id": f"{source_id}-{split}-{seed}",
            "input": {"contract_text": "A one-year renewal term.", "vendor_id": "v-2"},
            "reference": {"clauses": [{"clause_type": "renewal", "start": 2, "end": 29}]},
        }]

    def verify(*, output_valid: bool, **_kwargs: Any) -> dict[str, bool]:
        return {"passed": output_valid}

    def score(*, verification: dict[str, bool], **_kwargs: Any) -> dict[str, float]:
        return {"clause_f1": float(verification["passed"]), "unsupported_clause_rate": 0.0}

    for source in job.scenario_sources:
        registry.register(
            "scenario_provider", source.provider_id, source.provider_version, source.provider_digest, provider
        )
    registry.register("scorer", job.scorer.component_id, job.scorer.version, job.scorer.implementation_digest, score)
    registry.register("verifier", job.verifier.component_id, job.verifier.version, job.verifier.implementation_digest, verify)
    for method in job.candidate_methods:
        registry.register(
            "candidate_method", method.method_id, method.version, method.implementation_digest,
            lambda **_kwargs: None,
        )
    state_dir = tmp_path / "comparison-state"
    service = GymStudioService(registry, state_dir=state_dir)
    panel = service.materialize_public_scenarios(job, split="validation", seed=42)
    panel["scenarios"][0]["input"]["contract_text"] = "caller changed the contract"
    response = {
        "clauses": [{"clause_type": "renewal", "text": "one-year renewal", "start": 2, "end": 29}],
        "abstained": False,
    }
    candidates = (
        CandidateIdentity("base-a", "unchanged_base", _identity("base-a"), _identity("base-a"), _identity("base-a")),
        CandidateIdentity("base-b", "unchanged_base", _identity("base-b"), _identity("base-b"), _identity("base-b")),
    )
    dispatches: list[tuple[str, tuple[str, ...]]] = []

    def execute(*, candidate: dict[str, Any], scenarios: tuple[dict[str, Any], ...]) -> dict[str, Any]:
        assert all(row["input"]["contract_text"] == "A one-year renewal term." for row in scenarios)
        dispatches.append((candidate["candidate_id"], tuple(row["attempt_id"] for row in scenarios)))
        if candidate["candidate_id"] == "base-b" and len(dispatches) == 2:
            return {scenarios[0]["attempt_id"]: response}
        return {row["attempt_id"]: response for row in scenarios}

    for candidate in candidates:
        service.register_candidate_executor(
            candidate,
            executor_digest=_identity(f"executor:{candidate.candidate_id}"),
            execute=execute,
        )

    comparison = service.compare_unchanged_bases(
        job, candidates, scenario_manifest_digest=panel["manifest_digest"], repetitions=1
    )
    assert comparison["status"] == "incomplete"
    assert [run["status"] for run in comparison["runs"]] == ["complete", "incomplete"]
    completed_b = comparison["runs"][1]["attempts"][0]["attempt_id"]
    service = GymStudioService(registry, state_dir=state_dir)
    for candidate in candidates:
        service.register_candidate_executor(
            candidate, executor_digest=_identity(f"executor:{candidate.candidate_id}"), execute=execute,
        )
    comparison = service.compare_unchanged_bases(
        job, candidates, scenario_manifest_digest=panel["manifest_digest"], repetitions=1
    )
    assert comparison["status"] == "complete"
    assert len(dispatches) == 3
    assert dispatches[-1][0] == "base-b" and completed_b not in dispatches[-1][1]
    assert service.compare_unchanged_bases(
        job, candidates, scenario_manifest_digest=panel["manifest_digest"], repetitions=1
    ) == comparison
    assert len(dispatches) == 3
    assert comparison["authority"]["same_public_panel"] is True
    assert comparison["authority"]["served_model_identity_verified"] is False
    assert [run["scenario_manifest_digest"] for run in comparison["runs"]] == [panel["manifest_digest"]] * 2
    assert [attempt["scenario_id"] for attempt in comparison["runs"][0]["attempts"]] == [
        attempt["scenario_id"] for attempt in comparison["runs"][1]["attempts"]
    ]
    service._scenario_batches[panel["manifest_digest"]] = json.dumps(panel).encode()
    with pytest.raises(GymJobError, match="manifest digest does not match"):
        service.freeze_practice_plan(
            job, candidates[0], run_id="tampered-manifest",
            scenario_manifest_digest=panel["manifest_digest"],
        )


def _response_service(schema: dict[str, Any], tmp_path: Path, reference: Any = None, verifier: Any = None, scorer: Any = None, provider: Any = None, split: str = "validation") -> tuple[Any, ...]:
    job = replace(_contract_review_job(), task_output_schema=schema)
    registry = GymComponentRegistry()
    for source in job.scenario_sources:
        registry.register(
            "scenario_provider", source.provider_id, source.provider_version, source.provider_digest,
            provider or (lambda *, source_id, **kwargs: [{
                "scenario_id": source_id, "input": {"contract_text": "A contract."}, "reference": reference,
            }]),
        )
    registry.register("verifier", job.verifier.component_id, job.verifier.version, job.verifier.implementation_digest,
                      verifier or (lambda *, output_valid, **kwargs: {"passed": output_valid}))
    registry.register("scorer", job.scorer.component_id, job.scorer.version, job.scorer.implementation_digest,
                      scorer or (lambda **kwargs: {"accuracy": 1.0}))
    for method in job.candidate_methods:
        registry.register("candidate_method", method.method_id, method.version, method.implementation_digest,
                          lambda **kwargs: None)
    service = GymStudioService(registry, state_dir=tmp_path)
    candidate = CandidateIdentity("base", "unchanged_base", _identity("base"), _identity("base"), _identity("artifact"))
    panel = service.materialize_public_scenarios(job, split=split, seed=1)
    plan = service.freeze_practice_plan(job, candidate, run_id="response-test", scenario_manifest_digest=panel["manifest_digest"])
    return service, job, candidate, plan


@pytest.mark.parametrize(("schema", "response", "status"), [
    ({"type": "string"}, "answer", "scored"),
    ({"type": "integer"}, 7, "scored"),
    ({"type": "boolean"}, True, "scored"),
    ({"type": "array", "items": {"type": "integer"}}, [1, 2], "scored"),
    ({"type": "array", "items": {"type": "integer"}, "enum": [[1, 2]]}, [1, 2], "scored"),
    ({"type": "string"}, {"wrong": "shape"}, "invalid"),
    ({"type": "string"}, {"not-json"}, "invalid"),
    ({"type": "number"}, 10 ** 400, "invalid"),
])
def test_response_submission_is_typed_and_server_records_roundtrip(
    tmp_path: Path, schema: dict[str, Any], response: Any, status: str,
) -> None:
    service, job, candidate, plan = _response_service(schema, tmp_path)
    request = {"practice_plan": plan, "attempts": [
        {"attempt_id": row["attempt_id"], "response": response} for row in plan["expected_attempts"]
    ]}
    if isinstance(response, set):
        with pytest.raises(GymJobError):
            service.record_practice_run_document(request)
        assert service.record_practice_run(plan)["attempts"] == []
        for row in plan["expected_attempts"]:
            service.score_attempt(job, candidate, practice_plan_digest=plan["plan_digest"], attempt_id=row["attempt_id"], response=response)
        run = service.record_practice_run(plan)
    else:
        run = service.record_practice_run_document(request)
    assert all(row["status"] == status and row["execution_kind"] == "caller_supplied" for row in run["attempts"])
    assert all(row["executor_digest"] is None for row in run["attempts"])
    assert run["execution_summary"]["registered_execution_complete"] is False
    readback = service.record_practice_run_document({"practice_plan": plan, "attempts": run["attempts"]})
    assert readback["attempts"] == run["attempts"] and readback["status"] == run["status"]
    if not isinstance(response, set):
        assert service.record_practice_run_document(request) == run


def test_attempt_submission_cannot_invent_execution_evidence(tmp_path: Path) -> None:
    service, job, candidate, plan = _response_service({"type": "string"}, tmp_path)
    row = plan["expected_attempts"][0]
    forged = PracticeAttemptRecord(
        row["attempt_id"], job.digest, candidate.candidate_id, row["scenario_id"], row["scenario_digest"],
        row["scenario_origin"], row["scenario_split"], row["repetition"], _identity("response"),
        job.scorer.implementation_digest, job.verifier.implementation_digest, True, True,
        "registered_candidate", _identity("invented-executor"), "scored", {"accuracy": 1.0},
    )
    with pytest.raises(GymJobError, match="not server-recorded"):
        service.record_practice_run_document({"practice_plan": plan, "attempts": [forged.to_document()]})
    assert service.record_practice_run(plan)["attempts"] == []
    for changes in ({"output_valid": False}, {"status": "invalid"}, {"status": "failed"}):
        with pytest.raises(GymJobError, match="contradictory"):
            replace(forged, **changes)
    for changes in ({"status": []}, {"execution_kind": []}, {"scenario_origin": []}, {"candidate_id": 7}):
        with pytest.raises(GymJobError):
            replace(forged, **changes)


def test_large_state_is_chunked_without_raising_native_limits(tmp_path: Path) -> None:
    from neural_foundry.gym_service import _document_digest

    service = GymStudioService(state_dir=tmp_path)
    document = {"kind": "capacity-test", "attempt_ids": [_identity(str(index)) for index in range(100_000)]}
    digest = _document_digest(document)
    service._persist_state("capacity", digest[7:], document)
    assert service._load_state("capacity", digest[7:]) == document
    compact = json.loads((tmp_path / "capacity" / f"{digest[7:]}.json").read_bytes())
    assert compact["kind"] == "nf_gym_chunk_store_v1"
    long_rows = {"kind": "long-contracts", "rows": [{"contract_text": "x" * 12_000} for _ in range(100)]}
    long_digest = _document_digest(long_rows)
    service._persist_state("capacity", long_digest[7:], long_rows)
    assert service._load_state("capacity", long_digest[7:]) == long_rows
    assert all(path.stat().st_size <= 1_048_576 for path in (tmp_path / "chunks").glob("*.json"))
    with pytest.raises(GymJobError, match="dispatch bounded chunks"):
        service.run_practice_batch(
            _contract_review_job(), CandidateIdentity("b", "unchanged_base", _identity("b"), _identity("b"), _identity("a")),
            practice_plan_digest=_identity("plan"), attempt_ids=tuple(_identity(str(index)) for index in range(129)),
        )


def test_two_workers_cannot_execute_the_same_attempt(tmp_path: Path) -> None:
    service, job, candidate, plan = _response_service({"type": "string"}, tmp_path)
    other = GymStudioService(service.components, state_dir=tmp_path)
    entered, release = Event(), Event()
    calls: list[int] = []

    def execute(*, scenarios: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(1)
        entered.set()
        if not release.wait(10):
            raise RuntimeError("test worker timed out")
        return {row["attempt_id"]: "answer" for row in scenarios}

    for worker in (service, other):
        worker.register_candidate_executor(candidate, executor_digest=_identity("executor"), execute=execute)
    plan = service.freeze_practice_plan(job, candidate, run_id="model-claims", scenario_manifest_digest=plan["scenario_manifest_digest"])
    ids = (plan["expected_attempts"][0]["attempt_id"],)
    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(service.run_practice_batch, job, candidate, practice_plan_digest=plan["plan_digest"], attempt_ids=ids)
        try:
            assert entered.wait(10)
            with pytest.raises(GymJobError, match="already claimed"):
                other.run_practice_batch(job, candidate, practice_plan_digest=plan["plan_digest"], attempt_ids=ids)
        finally:
            release.set()
        assert len(future.result(timeout=10)) == 1
    assert len(calls) == 1


def test_returned_responses_survive_scoring_failure_and_restart(tmp_path: Path) -> None:
    calls = []
    fail = True

    def verifier(**kwargs):
        calls.append(1)
        if fail and len(calls) == 2:
            raise RuntimeError("scorer service unavailable")
        return {"passed": True}

    service, job, candidate, _ = _response_service({"type": "string"}, tmp_path, verifier=verifier)
    batch = service.materialize_public_scenarios(job, split="validation", seed=1)
    service.register_candidate_executor(candidate, executor_digest=_identity("executor"), execute=lambda *, scenarios, **kwargs: {row["attempt_id"]: "answer" for row in scenarios})
    plan = service.freeze_practice_plan(job, candidate, run_id="scoring-failure", scenario_manifest_digest=batch["manifest_digest"], repetitions=2)
    ids = tuple(row["attempt_id"] for row in plan["expected_attempts"])
    assert len(service.run_practice_batch(job, candidate, practice_plan_digest=plan["plan_digest"], attempt_ids=ids)) == 3
    assert len(service.record_practice_run(plan)["attempts"]) == 3
    fail = False
    resumed = GymStudioService(service.components, state_dir=tmp_path)
    missing = ids[1]
    assert resumed._get_response(plan["plan_digest"], missing)["response"] == "answer"
    assert len(resumed.retry_scoring(job, candidate, practice_plan_digest=plan["plan_digest"], attempt_ids=(missing,))) == 1
    run = resumed.record_practice_run(plan)
    assert run["status"] == "complete" and run["execution_summary"]["registered_execution_complete"]
    assert all("response" not in attempt for attempt in run["attempts"])


@pytest.mark.parametrize("mutating_component", ["verifier", "scorer"])
def test_callback_mutation_cannot_change_scored_response_or_private_reference(tmp_path: Path, mutating_component: str) -> None:
    mutate = True

    def verifier(**kwargs):
        if mutate and mutating_component == "verifier":
            kwargs["response"]["answer"] = "different"
        return {"passed": True}

    def scorer(**kwargs):
        if mutate and mutating_component == "scorer":
            kwargs["reference"]["answer"] = "different"
        return {"accuracy": 1.0}

    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"], "additionalProperties": False}
    service, job, candidate, plan = _response_service(schema, tmp_path, {"answer": "original"}, verifier, scorer)
    service.register_candidate_executor(candidate, executor_digest=_identity("executor"), execute=lambda *, scenarios, **kwargs: {row["attempt_id"]: {"answer": "original"} for row in scenarios})
    plan = service.freeze_practice_plan(job, candidate, run_id="model-mutation", scenario_manifest_digest=plan["scenario_manifest_digest"])
    ids = tuple(row["attempt_id"] for row in plan["expected_attempts"])
    assert service.run_practice_batch(job, candidate, practice_plan_digest=plan["plan_digest"], attempt_ids=ids) == ()
    assert all(service._get_response(plan["plan_digest"], attempt_id)["response"] == {"answer": "original"} for attempt_id in ids)
    assert all(reference == {"answer": "original"} for reference in service._scenario_references.values())
    mutate = False
    assert len(service.retry_scoring(job, candidate, practice_plan_digest=plan["plan_digest"], attempt_ids=ids)) == 2


def test_real_base_comparison_requires_distinct_checkpoint_and_worker_identity(tmp_path: Path) -> None:
    service, job, candidate, plan = _response_service({"type": "string"}, tmp_path)
    alias = replace(candidate, candidate_id="renamed-base")
    with pytest.raises(GymJobError, match="distinct base checkpoint"):
        service.compare_unchanged_bases(job, (candidate, alias), scenario_manifest_digest=plan["scenario_manifest_digest"])
    other = replace(alias, base_model_digest=_identity("other"), candidate_model_digest=_identity("other"), artifact_digest=_identity("other-artifact"))
    with pytest.raises(GymJobError, match="worker checkpoint identity probe"):
        service.compare_unchanged_bases(
            job, (candidate, other), scenario_manifest_digest=plan["scenario_manifest_digest"], require_worker_identity=True,
        )


@pytest.mark.parametrize("reference", [None, "known-answer", [1, 2], True, 3.5])
def test_private_references_support_bounded_json_values(tmp_path: Path, reference: Any) -> None:
    service, job, candidate, plan = _response_service({"type": "string"}, tmp_path, reference)
    row = plan["expected_attempts"][0]
    attempt = service.score_attempt(job, candidate, practice_plan_digest=plan["plan_digest"], attempt_id=row["attempt_id"], response="answer")
    assert attempt.status == "scored"
    assert service._scenario_references[(plan["scenario_manifest_digest"], row["scenario_id"])] == reference


def test_rejected_request_writes_nothing_and_model_plan_refuses_caller_data(tmp_path: Path) -> None:
    service, job, candidate, plan = _response_service({"type": "string"}, tmp_path)
    request = {"practice_plan": plan, "attempts": [
        {"attempt_id": plan["expected_attempts"][0]["attempt_id"], "response": "answer"}, {"malformed": True},
    ]}
    with pytest.raises(GymJobError):
        service.record_practice_run_document(request)
    assert service.record_practice_run(plan)["attempts"] == []
    assert service._get_response(plan["plan_digest"], plan["expected_attempts"][0]["attempt_id"]) is None
    service.register_candidate_executor(candidate, executor_digest=_identity("exec-v1"), execute=lambda **kwargs: {})
    model_plan = service.freeze_practice_plan(job, candidate, run_id="measured", scenario_manifest_digest=plan["scenario_manifest_digest"])
    with pytest.raises(GymJobError, match="execution type"):
        service.record_practice_run_document({"practice_plan": model_plan, "attempts": [{"attempt_id": model_plan["expected_attempts"][0]["attempt_id"], "response": "answer"}]})
    resumed = GymStudioService(service.components, state_dir=tmp_path)
    resumed.register_candidate_executor(candidate, executor_digest=_identity("exec-v2"), execute=lambda **kwargs: {})
    with pytest.raises(GymJobError, match="new run revision"):
        resumed.freeze_practice_plan(job, candidate, run_id="measured", scenario_manifest_digest=plan["scenario_manifest_digest"])


@pytest.mark.parametrize("overlap", ["content", "group"])
def test_document_and_source_family_overlap_is_rejected_after_restart(tmp_path: Path, overlap: str) -> None:
    def provider(*, split, source_id, **kwargs):
        return [{"scenario_id": f"{split}-{source_id}", "input": {"contract_text": "same contract" if overlap == "content" else f"variant-{split}-{source_id}"},
                 "reference": None, "group_id": f"local-{split}-{source_id}", "source_group_id": "source-family" if overlap == "group" else f"family-{split}-{source_id}"}]

    service, job, _candidate, _plan = _response_service({"type": "string"}, tmp_path, provider=provider, split="train")
    resumed = GymStudioService(service.components, state_dir=tmp_path)
    with pytest.raises(GymJobError, match="overlaps frozen splits"):
        resumed.materialize_public_scenarios(job, split="validation", seed=1)


def test_scored_practice_becomes_a_frozen_private_training_dataset_after_restart(tmp_path: Path) -> None:
    def provider(*, split, source_id, **kwargs):
        return [{"scenario_id": f"{split}-{source_id}", "input": {"contract_text": f"Contract {split} {source_id}."},
                 "reference": "answer", "group_id": f"family-{split}-{source_id}"}]

    service, job, candidate, initial_plan = _response_service({"type": "string"}, tmp_path, provider=provider, split="train")
    service.register_candidate_executor(candidate, executor_digest=_identity("exec"), execute=lambda *, scenarios, **kwargs: {row["attempt_id"]: "answer" for row in scenarios})
    plan = service.freeze_practice_plan(job, candidate, run_id="training-source", scenario_manifest_digest=initial_plan["scenario_manifest_digest"])
    ids = tuple(row["attempt_id"] for row in plan["expected_attempts"])
    service.run_practice_batch(job, candidate, practice_plan_digest=plan["plan_digest"], attempt_ids=ids)
    resumed = GymStudioService(service.components, state_dir=tmp_path)
    dataset = resumed.freeze_training_dataset(job, [{"practice_plan_digest": plan["plan_digest"], "attempt_id": attempt_id} for attempt_id in ids], dataset_id="training-v1")
    assert "response" not in json.dumps(dataset) or "allow_reported_responses" in dataset
    examples = GymStudioService(service.components, state_dir=tmp_path).load_training_examples(dataset["dataset_digest"])
    assert len(examples) == 2 and all(row["response"] == "answer" and row["source_split"] == "train" for row in examples)
    assert all(row["scorer"]["implementation_digest"] == job.scorer.implementation_digest for row in examples)
    validation = resumed.materialize_public_scenarios(job, split="validation", seed=1)
    validation_plan = resumed.freeze_practice_plan(job, candidate, run_id="reported-validation", scenario_manifest_digest=validation["manifest_digest"])
    row = validation_plan["expected_attempts"][0]
    resumed.score_attempt(job, candidate, practice_plan_digest=validation_plan["plan_digest"], attempt_id=row["attempt_id"], response="answer")
    with pytest.raises(GymJobError, match="cannot become training"):
        resumed.freeze_training_dataset(job, [{"practice_plan_digest": validation_plan["plan_digest"], "attempt_id": row["attempt_id"]}], dataset_id="invalid-validation", allow_reported_responses=True)


def test_job_metadata_cannot_reassign_split_custody(tmp_path: Path) -> None:
    def provider(*, split, source_id, **kwargs):
        return [{"scenario_id": f"{split}-{source_id}", "input": {"contract_text": "same input"}, "reference": None,
                 "item_id": "fixed-item", "family_id": "fixed-family"}]

    service, job, _candidate, _plan = _response_service({"type": "string"}, tmp_path, provider=provider, split="train")
    revised = replace(job, metadata={**job.metadata, "split_assignment_id": "different-scope"})
    with pytest.raises(GymJobError, match="authority is pinned"):
        GymStudioService(service.components, state_dir=tmp_path, split_authority_id="changed-controller-setting")
    resumed = GymStudioService(service.components, state_dir=tmp_path)
    with pytest.raises(GymJobError, match="overlaps frozen splits"):
        resumed.materialize_public_scenarios(revised, split="validation", seed=1)


def test_split_assignment_survives_job_and_source_revision(tmp_path: Path) -> None:
    def provider(*, split, source_id, **kwargs):
        return [{"scenario_id": f"{split}-{source_id}", "input": {"contract_text": "same contract"},
                 "reference": None, "item_id": "same-document", "family_id": "same-contract-family"}]

    service, job, _candidate, _plan = _response_service({"type": "string"}, tmp_path, provider=provider, split="train")
    changed_source = replace(job.scenario_sources[0], source_digest=_identity("new-source-revision"))
    renamed_job = replace(job, job_id="renamed-contract-review", scenario_sources=(changed_source, *job.scenario_sources[1:]))
    with pytest.raises(GymJobError, match="overlaps frozen splits"):
        service.materialize_public_scenarios(renamed_job, split="validation", seed=2)


def test_structured_inputs_and_indexed_dataset_lookup_survive_restart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def provider(*, split, source_id, **kwargs):
        return [{"scenario_id": f"{split}-{source_id}", "input": {"contract_text": f"ticket-{source_id}", "vendor_id": "billing"},
                 "reference": "billing", "item_id": f"item-{source_id}", "family_id": f"family-{source_id}"}]

    service, job, candidate, plan = _response_service({"type": "string"}, tmp_path, provider=provider, split="train")
    service.register_candidate_executor(candidate, executor_digest=_identity("executor"), execute=lambda *, scenarios, **kwargs: {row["attempt_id"]: "billing" for row in scenarios})
    plan = service.freeze_practice_plan(job, candidate, run_id="indexed-routing", scenario_manifest_digest=plan["scenario_manifest_digest"])
    calls = []
    lookup = service._get_scenario_batch

    def counted(digest):
        calls.append(digest)
        return lookup(digest)

    monkeypatch.setattr(service, "_get_scenario_batch", counted)
    ids = tuple(row["attempt_id"] for row in plan["expected_attempts"])
    service.run_practice_batch(job, candidate, practice_plan_digest=plan["plan_digest"], attempt_ids=ids)
    assert len(calls) == 1
    dataset = service.freeze_training_dataset(job, [{"practice_plan_digest": plan["plan_digest"], "attempt_id": value} for value in ids], dataset_id="routing")
    assert len(calls) == 1
    examples = GymStudioService(service.components, state_dir=tmp_path).load_training_examples(dataset["dataset_digest"])
    assert all(example["input"]["vendor_id"] == "billing" and "contract_text" in example["input"] for example in examples)
    assert all(example["interaction"]["input"] == example["input"] for example in examples)


def test_batch_claim_uses_constant_descriptors_under_low_limit(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    if os.name == "nt":
        pytest.skip("POSIX process-owned claim qualification requires WSL")
    service, job, candidate, plan = _response_service({"type": "string"}, tmp_path)
    plan = service.freeze_practice_plan(job, candidate, run_id="low-fd", scenario_manifest_digest=plan["scenario_manifest_digest"], repetitions=64)
    code = (
        "import resource,sys; from neural_foundry.gym_service import GymStudioService; "
        "s=GymStudioService(state_dir=sys.argv[1]); p=s._load_frozen_plan(sys.argv[2]); "
        "resource.setrlimit(resource.RLIMIT_NOFILE,(32,32)); "
        "claim=s._claim_batch(p['plan_digest'],[r['attempt_id'] for r in p['expected_attempts']]); "
        "claim.__enter__(); print('claimed'); claim.__exit__(None,None,None)"
    )
    completed = subprocess.run([sys.executable, "-c", code, str(tmp_path), plan["plan_digest"]], check=False, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert "claimed" in completed.stdout and not list((tmp_path / "leases").glob("*.json"))


@pytest.mark.parametrize("kind", ["lora", "qlora", "full", "weight_edit", "own_model"])
def test_shared_candidate_manifest_binds_format_request_and_files(tmp_path: Path, kind: str) -> None:
    from neural_foundry.gym_artifacts import read_candidate_manifest, write_candidate_manifest

    (tmp_path / "weights.bin").write_bytes(b"weights")
    base = None if kind == "own_model" else _identity("base")
    record = write_candidate_manifest(
        tmp_path, model_id="model", kind=kind, base_digest=base,
        model_digest=_identity("model"), request_digest=_identity("request"),
    )
    assert read_candidate_manifest(tmp_path, _identity("request")) == record
    with pytest.raises(GymJobError, match="frozen request"):
        read_candidate_manifest(tmp_path, _identity("other-request"))
    (tmp_path / "weights.bin").write_bytes(b"changed")
    with pytest.raises(GymJobError, match="file set or digest"):
        read_candidate_manifest(tmp_path, _identity("request"))


def test_candidate_reference_rejects_malformed_kind_and_missing_base(tmp_path: Path) -> None:
    from neural_foundry.gym_artifacts import CandidateRef

    with pytest.raises(GymJobError, match="unsupported candidate artifact kind"):
        CandidateRef("model", [], None, _identity("model"), _identity("artifact"), str(tmp_path))
    with pytest.raises(GymJobError, match="pinned base"):
        CandidateRef("model", "lora", None, _identity("model"), _identity("artifact"), str(tmp_path))


def test_base_free_candidate_requires_verified_artifact_and_state() -> None:
    service = GymStudioService()
    candidate = CandidateIdentity("own", "own_model", None, _identity("state"), _identity("artifact"))
    probe = {"checkpoint_digest": candidate.artifact_digest, "local_loaded_files_verified": True,
             "config_digest": _identity("config"), "tokenizer_digest": _identity("tokenizer"),
             "state_digest": candidate.candidate_model_digest, "model_id": "own", "model_family": "qwen3",
             "device": "cpu", "precision": "float32", "revision": None}
    service.register_candidate_executor(candidate, executor_digest=_identity("executor"),
                                        execute=lambda **_kwargs: {}, identity_probe=lambda: probe)
    service._capture_worker_identity(candidate, _identity("executor"))
    assert service._get_worker_identity(candidate)["candidate"]["base_model_digest"] is None
    probe["state_digest"] = _identity("other")
    with pytest.raises(GymJobError, match="base-free worker state"):
        service._capture_worker_identity(candidate, _identity("executor"))
    with pytest.raises(GymJobError, match="requires a pinned base"):
        CandidateIdentity("adapter", "lora", None, _identity("state"), _identity("artifact"))


def test_shared_model_state_digest_detects_tensor_changes() -> None:
    import torch
    from neural_foundry.gym_adapters import _state_digest
    from neural_foundry.model_digest import model_state_digest

    model = torch.nn.Linear(2, 1)
    original = model_state_digest(model)
    assert _state_digest(model) == original
    with torch.no_grad():
        model.weight[0, 0] += 1
    assert model_state_digest(model) != original


def test_training_checkpoint_must_match_frozen_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from neural_foundry import gym_adapters
    from neural_foundry.gym_artifacts import CandidateRef
    from neural_foundry.gym_service import _document_digest

    files = {"config.json": _identity("config")}
    checkpoint = SimpleNamespace(path=tmp_path, file_digests=files)
    monkeypatch.setattr(gym_adapters, "resolve_checkpoint", lambda *_args, **_kwargs: checkpoint)
    digest = _document_digest({"model_id": "model", "revision": "rev", "files": files})
    base = CandidateRef("model", "full", digest, _identity("model"), _identity("artifact"), str(tmp_path))
    request = {"loaded_base": {"model_id": "model", "revision": "rev"}}
    assert gym_adapters._resolve_training_checkpoint(request, base) is checkpoint
    checkpoint.file_digests = {"config.json": _identity("changed")}
    with pytest.raises(GymJobError, match="frozen base reference"):
        gym_adapters._resolve_training_checkpoint(request, base)


def test_peft_backend_config_is_typed_and_digest_bound() -> None:
    from neural_foundry.backends.peft.config import PeftBackendConfig

    recipe = {"method": "lora", "seed": 1, "steps": 2, "max_sequence_tokens": 32,
              "tokens_per_batch": 64, "max_dataset_tokens": 128, "rank": 2,
              "target_modules": ["q_proj"], "learning_rate": 0.0001}
    config = PeftBackendConfig.from_recipe(recipe)
    assert config.document() == recipe
    assert config.digest != PeftBackendConfig.from_recipe({**recipe, "seed": 2}).digest
    for changed in ({"seed": True}, {"learning_rate": float("nan")}, {"unknown": 1}):
        with pytest.raises(GymJobError):
            PeftBackendConfig.from_recipe({**recipe, **changed})


def test_automodel_config_and_recipe_are_explicit_and_digest_bound(tmp_path: Path) -> None:
    from neural_foundry.backends.automodel.config import AutoModelBackendConfig, AutoModelSFTRecipe

    config = AutoModelBackendConfig(str(tmp_path / "python"))
    assert config.document()["versions"]["nemo-automodel"] == "0.7.0"
    recipe = {"method": "sft", "seed": 17, "steps": 1, "max_sequence_tokens": 32,
              "tokens_per_batch": 64, "max_dataset_tokens": 128, "learning_rate": 0.00001}
    assert AutoModelSFTRecipe.from_document(recipe).document() == recipe
    assert AutoModelSFTRecipe.from_document(recipe).digest != AutoModelSFTRecipe.from_document({**recipe, "seed": 18}).digest
    lora = {**recipe, "method": "lora", "rank": 2, "target_modules": ["q_proj", "v_proj"]}
    assert AutoModelSFTRecipe.from_document(lora).document() == lora
    for changed in ({"method": "qlora"}, {"steps": True}, {"learning_rate": float("nan")},
                    {"max_sequence_tokens": 65}, {"extra": 1}):
        with pytest.raises(GymJobError):
            AutoModelSFTRecipe.from_document({**recipe, **changed})
    with pytest.raises(GymJobError, match="absolute Python"):
        AutoModelBackendConfig("python")


@pytest.mark.parametrize("method", ["full", "lora", "qlora"])
def test_real_automodel_synthetic_chat_checkpoint(tmp_path: Path, method: str) -> None:
    import os

    from neural_foundry.execution_runner import run_local_worker

    python = os.environ.get("NF_AUTOMODEL_PYTHON")
    base = os.environ.get("NF_AUTOMODEL_BASE_PATH")
    if not python or not base or os.environ.get("NF_AUTOMODEL_LIVE") != "1":
        pytest.skip("set NF_AUTOMODEL_PYTHON, NF_AUTOMODEL_BASE_PATH and NF_AUTOMODEL_LIVE=1")
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(json.dumps({"messages": [{"role": "user", "content": "Say hello."},
                                                 {"role": "assistant", "content": "Hello."}]}) + "\n")
    probe = Path(__file__).resolve().parents[1] / "python/neural_foundry/backends/automodel/worker/probe.py"
    dataset_probe = run_local_worker([python, str(probe), "dataset", base, str(dataset)], cwd=tmp_path, timeout_seconds=60)
    assert dataset_probe.returncode == 0, dataset_probe.stderr.decode("utf-8", "replace")[-5000:]
    dataset_line = next(line for line in dataset_probe.stdout.decode().splitlines() if line.startswith("NF_AUTOMODEL_INSPECTION="))
    dataset_report = json.loads(dataset_line.split("=", 1)[1])
    assert dataset_report["rows"] == 1 and 0 < dataset_report["total_tokens"] <= 128
    checkpoint = tmp_path / "checkpoints"
    config = {
        "recipe": "TrainFinetuneRecipeForNextTokenPrediction", "seed": 17,
        "step_scheduler": {"global_batch_size": 1, "local_batch_size": 1, "max_steps": 1,
                           "ckpt_every_steps": 1, "val_every_steps": 100},
        "dist_env": {"backend": "nccl", "timeout_minutes": 1},
        "model": {"_target_": "nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained",
                  "pretrained_model_name_or_path": base, "torch_dtype": "bf16"},
        "checkpoint": {"enabled": True, "checkpoint_dir": str(checkpoint),
                       "model_save_format": "safetensors", "save_consolidated": "final"},
        "distributed": {"strategy": "fsdp2", "dp_size": "none", "tp_size": 1, "cp_size": 1,
                        "sequence_parallel": False},
        "loss_fn": {"_target_": "nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy"},
        "dataset": {"_target_": "nemo_automodel.components.datasets.llm.chat_dataset.ChatDataset",
                    "path_or_dataset_id": str(dataset), "seq_length": 128, "mask_history": True,
                    "mask_generation_prompt": True},
        "packed_sequence": {"packed_sequence_size": 0},
        "dataloader": {"_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
                       "collate_fn": "nemo_automodel.components.datasets.utils.default_collater", "shuffle": False},
        "optimizer": {"_target_": "torch.optim.AdamW", "lr": 0.00001, "weight_decay": 0},
    }
    if method in {"lora", "qlora"}:
        config["peft"] = {"_target_": "nemo_automodel.components._peft.lora.PeftConfig",
                          "target_modules": "*_proj", "dim": 2, "alpha": 4, "use_triton": False}
    if method == "qlora":
        config["quantization"] = {"load_in_4bit": True, "load_in_8bit": False,
                                  "bnb_4bit_compute_dtype": "bfloat16", "bnb_4bit_use_double_quant": True,
                                  "bnb_4bit_quant_type": "nf4"}
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(json.dumps(config))
    result = run_local_worker([str(Path(python).parent / "automodel"), str(recipe), "--nproc-per-node", "1"],
                              cwd=tmp_path, timeout_seconds=240)
    assert result.returncode == 0 and not result.timed_out, (result.stdout + result.stderr).decode("utf-8", "replace")[-5000:]
    assert list(checkpoint.rglob("*.safetensors"))
    if method in {"lora", "qlora"}:
        assert list(checkpoint.rglob("adapter_model.safetensors"))
        return
    consolidated = list(checkpoint.glob("epoch_*_step_*/model/consolidated"))
    assert len(consolidated) == 1
    model_probe = run_local_worker([python, str(probe), "model", str(consolidated[0]), "bfloat16"],
                                   cwd=tmp_path, timeout_seconds=120)
    assert model_probe.returncode == 0, model_probe.stderr.decode("utf-8", "replace")[-5000:]
    model_line = next(line for line in model_probe.stdout.decode().splitlines() if line.startswith("NF_AUTOMODEL_INSPECTION="))
    assert json.loads(model_line.split("=", 1)[1])["family"] == "qwen3"


@pytest.mark.parametrize("method", ["sft", "lora", "qlora"])
def test_real_automodel_trainer_returns_reloadable_candidate(tmp_path: Path, method: str) -> None:
    import os

    from neural_foundry.execution_runner import run_local_worker

    gpu_python = os.environ.get("NF_GPU_PYTHON")
    automodel_python = os.environ.get("NF_AUTOMODEL_PYTHON")
    if not gpu_python or not automodel_python or os.environ.get("NF_AUTOMODEL_LIVE") != "1":
        pytest.skip("set NF_GPU_PYTHON, NF_AUTOMODEL_PYTHON and NF_AUTOMODEL_LIVE=1")
    code = """
import json, sys
from pathlib import Path
from neural_foundry.backends.automodel.adapter import automodel_sft_trainer
from neural_foundry.backends.automodel.config import AutoModelBackendConfig
from neural_foundry.gym_adapters import transformers_model
from neural_foundry.gym_job import GymJobError
from neural_foundry.gym_service import _document_digest
base = transformers_model('Qwen/Qwen3-0.6B', 'c1899de289a04d12100db370d81485cdf75e47ca',
                          device='cuda:0', precision='bfloat16', context_tokens=512, max_new_tokens=16, seed=17)
examples = [{'example_digest': _document_digest({'id': 'synthetic'}),
             'interaction': {'format': 'chat_messages_v1', 'messages': [
                 {'role': 'user', 'content': 'Say hello.'}, {'role': 'assistant', 'content': 'Hello.'}]}}]
recipe = {'method': sys.argv[3], 'seed': 17, 'steps': 1, 'max_sequence_tokens': 128,
          'tokens_per_batch': 128, 'max_dataset_tokens': 128, 'learning_rate': 0.00001}
if recipe['method'] in {'lora', 'qlora'}:
    recipe.update({'rank': 2, 'target_modules': ['q_proj', 'v_proj']})
request = {'recipe': recipe, 'loaded_base': dict(base.loaded_identity()),
           'inference_config': dict(base.inference_config), 'dataset_digest': _document_digest({'dataset': 'synthetic'}),
           'projected_examples_digest': _document_digest({'examples': examples})}
request['training_request_digest'] = _document_digest(request)
trainer = automodel_sft_trainer(AutoModelBackendConfig(sys.argv[1], timeout_seconds=240, reload_python_executable=sys.argv[4]))
output = Path(sys.argv[2]) / 'artifact'
reference = trainer.train(request, examples, base.reference, output)
assert trainer.load(request, base.reference, output) == reference
loaded = base.load_candidate(reference)
assert loaded.loaded_identity()['state_digest'] == reference.model_digest
assert reference.kind == (recipe['method'] if recipe['method'] in {'lora', 'qlora'} else 'full')
assert reference.model_digest != base.loaded_identity()['state_digest']
(output / 'automodel_config.json').write_text('{}')
try:
    trainer.load(request, base.reference, output)
except GymJobError as error:
    assert 'configuration differs' in str(error)
else:
    raise AssertionError('modified AutoModel configuration was accepted')
print(json.dumps({'model_digest': reference.model_digest, 'artifact_digest': reference.artifact_digest}))
"""
    result = run_local_worker([gpu_python, "-c", code, automodel_python, str(tmp_path), method, gpu_python], cwd=tmp_path, timeout_seconds=300)
    assert result.returncode == 0 and not result.timed_out, (result.stdout + result.stderr).decode("utf-8", "replace")[-6000:]
    assert json.loads(result.stdout.decode().splitlines()[-1])["model_digest"].startswith("sha256:")


@pytest.mark.parametrize("method", ["sft", "lora", "qlora"])
def test_real_automodel_through_lifecycle_and_restart(tmp_path: Path, method: str) -> None:
    import os

    from neural_foundry.execution_runner import run_local_worker

    gpu_python = os.environ.get("NF_GPU_PYTHON")
    automodel_python = os.environ.get("NF_AUTOMODEL_PYTHON")
    if not gpu_python or not automodel_python or os.environ.get("NF_AUTOMODEL_LIVE") != "1":
        pytest.skip("set NF_GPU_PYTHON, NF_AUTOMODEL_PYTHON and NF_AUTOMODEL_LIVE=1")
    code = """
import json, sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, sys.argv[3])
from scripts.run_gym_lifecycle_real import _build_support_task
from neural_foundry.backends.automodel.config import AutoModelBackendConfig
from neural_foundry.gym_adapters import transformers_model
from neural_foundry.gym_job import CandidateMethod
from neural_foundry.gym_lifecycle import AdapterIdentity, EvaluatorAdapter, public_panel_evaluator
from neural_foundry.provenance import sha256_file
from neural_foundry.studio import WeightProgrammingStudio
studio = WeightProgrammingStudio(gym_state_dir=Path(sys.argv[2]) / 'state')
task = _build_support_task(studio.gym_service.components)
methods = [CandidateMethod(name, '1', task.job.scorer.implementation_digest, {}) for name in ('sft', 'qlora')]
for method in methods:
    studio.gym_service.components.register('candidate_method', method.method_id, method.version,
                                           method.implementation_digest, lambda **kwargs: None)
task = replace(task, job=replace(task.job, candidate_methods=(*task.job.candidate_methods, *methods),
                                 backends={'trainer': 'nemo_automodel.sft@1'}))
base = transformers_model('Qwen/Qwen3-0.6B', 'c1899de289a04d12100db370d81485cdf75e47ca',
                          device='cuda:0', precision='bfloat16', context_tokens=512, max_new_tokens=16, seed=17)
evaluator = EvaluatorAdapter(AdapterIdentity('public', '1', sha256_file(__file__)), public_panel_evaluator)
for adapter in (task, base, evaluator):
    studio.gym_lifecycle.register(adapter)
studio.gym_lifecycle.register_backend('nemo_automodel.sft@1', config=AutoModelBackendConfig(sys.argv[1], 240, sys.argv[5]))
request = {'run_id': 'auto-sft', 'task': task.identity.adapter_id + '@' + task.identity.version,
           'model': base.identity.adapter_id + '@' + base.identity.version, 'trainer': 'nemo_automodel.sft@1',
           'evaluator': 'public@1', 'approved_responses': {'support-train-T-100': 'billing'},
           'recipe': {'method': sys.argv[4], 'seed': 17, 'steps': 1, 'max_sequence_tokens': 256,
                      'tokens_per_batch': 256, 'max_dataset_tokens': 256, 'learning_rate': 0.00001}}
if request['recipe']['method'] in {'lora', 'qlora'}:
    request['recipe'].update({'rank': 2, 'target_modules': ['q_proj', 'v_proj']})
first = studio.run_gym_lifecycle(request)
second = studio.run_gym_lifecycle(request)
assert first['artifact_reused'] is False and second['artifact_reused'] is True
assert first['candidate_record']['candidate'] == second['candidate_record']['candidate']
assert second['approval']['release_authorized'] is False
print(json.dumps({'artifact_reused': second['artifact_reused'], 'decision': second['scorecard']['decision']}))
""".replace("sha256_file(__file__)", "'sha256:' + 'a' * 64")
    repo = Path(__file__).resolve().parents[1]
    result = run_local_worker([gpu_python, "-c", code, automodel_python, str(tmp_path), str(repo), method, gpu_python],
                              cwd=tmp_path, timeout_seconds=420)
    assert result.returncode == 0 and not result.timed_out, (result.stdout + result.stderr).decode("utf-8", "replace")[-6000:]
    assert json.loads(result.stdout.decode().splitlines()[-1])["artifact_reused"] is True


def test_lifecycle_rejects_candidate_manifest_identity_mismatch(tmp_path: Path) -> None:
    from neural_foundry.gym_artifacts import read_artifact_manifest, write_candidate_manifest

    (tmp_path / "weights.bin").write_bytes(b"weights")
    write_candidate_manifest(
        tmp_path, model_id="model", kind="lora", base_digest=_identity("base"),
        model_digest=_identity("model"), request_digest=_identity("request"),
        metadata={"candidate": {"base_model_digest": _identity("other-base"),
                                "candidate_model_digest": _identity("model")},
                  "model_adapter": {}, "loaded_candidate": {}},
    )
    with pytest.raises(GymJobError, match="metadata differs"):
        read_artifact_manifest(tmp_path, _identity("request"))


def test_training_artifact_inventory_includes_nested_manifests_and_rejects_symlinks(tmp_path: Path) -> None:
    from neural_foundry.gym_lifecycle import GymLifecycle

    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "artifact_manifest.json").write_text("outer")
    (nested / "artifact_manifest.json").write_text("inner")
    assert set(GymLifecycle._artifact_files(tmp_path)) == {"nested/artifact_manifest.json"}
    try:
        (tmp_path / "link").symlink_to(nested / "artifact_manifest.json")
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(GymJobError, match="symlinks"):
        GymLifecycle._artifact_files(tmp_path)


def test_backend_discovery_rejects_missing_or_mismatched_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    from importlib import metadata

    from neural_foundry.backends import BackendFactory, BackendRegistry
    from neural_foundry.gym_lifecycle import AdapterIdentity, EvaluatorAdapter

    registry = BackendRegistry()
    adapter = EvaluatorAdapter(AdapterIdentity("example", "1", _identity("example")), lambda *_args: {})
    registry.register(BackendFactory("example", "1", {"test-package": "0.7.0"}, lambda: adapter))
    monkeypatch.setattr(metadata, "version", lambda _name: "0.7.0rc0")
    assert registry.available() == []
    with pytest.raises(GymJobError, match="version mismatch"):
        registry.create("example@1")
    def missing(_name):
        raise metadata.PackageNotFoundError("test-package")

    monkeypatch.setattr(metadata, "version", missing)
    assert registry.available() == []
    with pytest.raises(GymJobError, match="version mismatch"):
        registry.create("example@1")
    monkeypatch.setattr(metadata, "version", lambda _name: "0.7.0")
    assert registry.available() == ["example@1"]
    assert registry.create("example@1") is adapter
    with pytest.raises(GymJobError, match="explicit backend version"):
        registry.create("example")
    registry.register(BackendFactory("external", "1", {}, lambda: adapter, lambda: False))
    assert "external@1" not in registry.available()
    with pytest.raises(GymJobError, match="runtime is missing"):
        registry.create("external@1")


def test_reward_guard_stops_on_drift_kl_and_scorer_mismatch() -> None:
    from neural_foundry.backends.reward_guard import RewardDriftGuard

    scorer = _identity("scorer")
    guard = RewardDriftGuard(scorer_digest=scorer, max_kl_nats=0.1)
    assert guard.observe(step=1, scorer_digest=scorer, reward=0.4, validation=0.8, kl_nats=0.05).stop is False
    with pytest.raises(GymJobError, match="frozen scorer"):
        guard.observe(step=2, scorer_digest=_identity("other"), reward=0.5, validation=0.7, kl_nats=0.05)
    with pytest.raises(GymJobError, match="finite and in range"):
        guard.observe(step=2, scorer_digest=scorer, reward=float("nan"), validation=0.7, kl_nats=0.05)
    decision = guard.observe(step=2, scorer_digest=scorer, reward=0.5, validation=0.7, kl_nats=0.05)
    assert decision.stop is True and decision.reason == "possible_scorer_gaming"
    assert isinstance(guard.observations, tuple)
    with pytest.raises(GymJobError, match="cannot continue"):
        guard.observe(step=3, scorer_digest=scorer, reward=0.6, validation=0.6, kl_nats=0.05)
    kl_guard = RewardDriftGuard(scorer_digest=scorer, max_kl_nats=0.1)
    assert kl_guard.observe(step=1, scorer_digest=scorer, reward=0.4, validation=0.8, kl_nats=0.11).reason == "kl_limit_exceeded"


def test_lifecycle_selects_exact_adapter_version() -> None:
    from neural_foundry.gym_lifecycle import AdapterIdentity, EvaluatorAdapter, GymLifecycle
    from neural_foundry.gym_service import GymStudioService

    lifecycle = GymLifecycle(GymStudioService())
    first = EvaluatorAdapter(AdapterIdentity("backend.evaluator", "1", _identity("one")), lambda *_args: {})
    second = EvaluatorAdapter(AdapterIdentity("backend.evaluator", "2", _identity("two")), lambda *_args: {})
    lifecycle.register(first)
    assert lifecycle._resolve(EvaluatorAdapter, "backend.evaluator") is first
    assert lifecycle._resolve(EvaluatorAdapter, "backend.evaluator@1") is first
    lifecycle.register(second)
    assert lifecycle._resolve(EvaluatorAdapter, "backend.evaluator@2") is second
    with pytest.raises(GymJobError, match="unknown or ambiguous"):
        lifecycle._resolve(EvaluatorAdapter, "backend.evaluator")
    with pytest.raises(GymJobError, match="unknown or ambiguous"):
        lifecycle._resolve(EvaluatorAdapter, "backend.evaluator@3")
    with pytest.raises(GymJobError, match="already registered"):
        lifecycle.register(first)
    with pytest.raises(GymJobError, match="versioned implementation identity"):
        AdapterIdentity("bad", "1", "sha256:not-a-digest").document()


def test_one_studio_lifecycle_reuses_tasks_bases_and_restart_lineage(tmp_path: Path) -> None:
    from neural_foundry.gym_artifacts import CandidateRef
    from neural_foundry.gym_lifecycle import (
        AdapterIdentity,
        EvaluatorAdapter,
        ModelAdapter,
        TaskAdapter,
        TrainerAdapter,
        public_panel_evaluator,
    )
    from neural_foundry.studio import WeightProgrammingStudio

    training_calls = []
    inference_calls = []

    def task_adapter(name, interaction_type, interaction_format):
        digest = _identity(name)
        source = ScenarioSource(f"{name}-source", f"{name}-provider", "1", digest, _identity(f"{name}-data"), "simulated", ("train", "validation", "holdout"), {})
        job = replace(
            _contract_review_job(), job_id=name,
            task_input_schema={"type": "object", "properties": {
                "prompt": {"type": "string"}, "ticket": {"type": "string"},
                "allowed_labels": {"type": "array", "items": {"type": "string"}},
            }, "required": ["prompt", "ticket", "allowed_labels"], "additionalProperties": False},
            task_output_schema={"type": "string"} if interaction_type == "single_response" else {"type": "array", "maxItems": 2, "items": {"type": "string"}},
            scenario_sources=(source,), scorer=ComponentReference(f"{name}-score", "1", digest, {}),
            verifier=ComponentReference(f"{name}-verify", "1", digest, {}),
            gates=(AcceptanceGate("quality", "task", "accuracy", "validation", "ge", 0.9), *_contract_review_job().gates[1:]),
        )

        def bind(service):
            def provider(*, split, **kwargs):
                rows = [(f"{name}-{split}", f"{split}: charge dispute", "billing")]
                if name == "routing" and split == "train":
                    rows.append((f"{name}-{split}-wrong", "password reset issue", "technical"))
                return [{"scenario_id": scenario_id, "item_id": scenario_id, "family_id": f"{name}-family-{scenario_id}",
                         "interaction_type": interaction_type, "input": {"prompt": "Classify this ticket", "ticket": ticket, "allowed_labels": ["billing", "support"]},
                         "reference": target if interaction_type == "single_response" else ["lookup", "reply"]}
                        for scenario_id, ticket, target in rows]

            for kind, ident, version, implementation, function in (
                ("scenario_provider", source.provider_id, source.provider_version, source.provider_digest, provider),
                ("verifier", job.verifier.component_id, job.verifier.version, digest, lambda *, reference, response, **kwargs: {"passed": response == reference}),
                ("scorer", job.scorer.component_id, job.scorer.version, digest, lambda *, verification, **kwargs: {"accuracy": float(verification["passed"])}),
                *(("candidate_method", method.method_id, method.version, method.implementation_digest, lambda **kwargs: None) for method in job.candidate_methods),
            ):
                try:
                    service.components.resolve(kind, ident, version, implementation)
                except GymJobError:
                    service.components.register(kind, ident, version, implementation, function)

        def project(input_value, response):
            messages = [{"role": "system", "content": job.description}, {"role": "user", "content": json.dumps(input_value, sort_keys=True)}]
            if response is not None:
                messages.append({"role": "assistant", "content": json.dumps(response)})
            return {"format": interaction_format, "messages": messages}

        return TaskAdapter(AdapterIdentity(name, "1", digest), job, interaction_type, AdapterIdentity(f"{name}-projection", "1", digest), interaction_format, project, bind)

    tasks = [task_adapter("routing", "single_response", "chat_messages_v1"), task_adapter("bounded-tools", "tool_episode", "tool_messages_v1")]

    def model_adapter(family, candidate=None, reference=None):
        base = _identity(f"base-{family}")
        candidate = candidate or CandidateIdentity(family, "unchanged_base", base, base, base)
        kind = "lora" if candidate.method_id == "lora" else "full"
        reference = reference or CandidateRef(family, kind, base, candidate.candidate_model_digest,
                                              candidate.artifact_digest, str(tmp_path), candidate.candidate_id,
                                              candidate.method_id, family)

        def infer(interaction):
            input_value = json.loads(interaction["messages"][1]["content"])
            assert "ticket" in input_value and input_value["allowed_labels"] == ["billing", "support"]
            inference_calls.append((family, interaction["format"]))
            return "billing" if interaction["format"] == "chat_messages_v1" else ["lookup", "reply"]

        def load_candidate(ref):
            loaded = CandidateIdentity(ref.candidate_id, ref.method_id, ref.base_digest, ref.model_digest, ref.artifact_digest)
            return model_adapter(family, loaded, ref)

        return ModelAdapter(AdapterIdentity(f"model-{family}", "1", _identity(f"model-{family}")), candidate, family,
                            ("chat_messages_v1", "tool_messages_v1"), {"seed": 17}, infer,
                            lambda: {"checkpoint_digest": candidate.base_model_digest, "model_id": family, "model_family": family,
                                     "revision": "a" * 40, "config_digest": _identity("config"), "tokenizer_digest": _identity("tokenizer"),
                                     "state_digest": candidate.candidate_model_digest, "device": "cpu", "precision": "float64", "local_loaded_files_verified": True},
                            ("full", "lora"), kind, reference, load_candidate)

    def result_ref(request, base, directory):
        return CandidateRef(base.model_id, "lora", base.base_digest, _identity(request["training_request_digest"]),
                            _identity(f"artifact-{request['training_request_digest']}"), str(directory),
                            f"trained-{request['training_request_digest'][7:19]}", "lora", base.family)

    def train(request, examples, base, directory):
        training_calls.append(request["training_request_digest"])
        assert len(examples) == 1
        if base.family == "qwen3":
            assert not (directory / "stale.bin").exists()
        assert "ticket" in json.loads(examples[0]["interaction"]["messages"][1]["content"])
        directory.mkdir(parents=True)
        (directory / "adapter.json").write_text(json.dumps({"request": request["training_request_digest"]}))
        return result_ref(request, base, directory)

    trainer = TrainerAdapter(AdapterIdentity("trainer", "1", _identity("trainer")), ("chat_messages_v1",), ("qwen3", "llama"), ("lora",), train, result_ref, {"lora": "lora"})
    evaluator = EvaluatorAdapter(AdapterIdentity("public-evaluator", "1", _identity("evaluator")), public_panel_evaluator)

    def studio():
        result = WeightProgrammingStudio(gym_state_dir=tmp_path)
        for adapter in (*tasks, model_adapter("qwen3"), model_adapter("llama"), trainer, evaluator):
            result.register_gym_lifecycle_adapter(adapter)
        return result

    first = studio()
    requests = [{"run_id": f"routing-{family}", "task": "routing", "model": f"model-{family}", "trainer": "trainer",
                 "evaluator": "public-evaluator", "recipe": {"method": "lora", "seed": 17},
                 "approved_responses": {"routing-train": "billing", "routing-train-wrong": "billing"}} for family in ("qwen3", "llama")]
    limited = replace(model_adapter("qwen3"), identity=AdapterIdentity("limited-model", "1", _identity("limited-model")), artifact_kinds=("full",))
    first.register_gym_lifecycle_adapter(limited)
    with pytest.raises(GymJobError, match="artifact kind rejected before dispatch"):
        first.run_gym_lifecycle({**requests[0], "model": "limited-model"})
    assert not training_calls and not inference_calls

    def wrong_kind_train(request, examples, base, directory):
        directory.mkdir(parents=True)
        (directory / "adapter.json").write_text("{}")
        return replace(result_ref(request, base, directory), kind="full")

    wrong_kind_trainer = replace(trainer, identity=AdapterIdentity("wrong-kind", "1", _identity("wrong-kind")), train=wrong_kind_train)
    first.register_gym_lifecycle_adapter(wrong_kind_trainer)
    with pytest.raises(GymJobError, match="incompatible candidate reference before publication"):
        first.run_gym_lifecycle({**requests[0], "run_id": "wrong-kind", "trainer": "wrong-kind"})
    assert not list(tmp_path.glob("lifecycle/*/artifact/artifact_manifest.json"))
    from neural_foundry.gym_service import _document_digest

    request_binding = _document_digest({
        "kind": "nf_gym_lifecycle_request_v1", "request": requests[0],
        "job_digest": first.gym_lifecycle.tasks["routing"].job.digest,
        "task_adapter": first.gym_lifecycle.tasks["routing"].identity.document(),
        "model_adapter": first.gym_lifecycle.models["model-qwen3"].identity.document(),
        "trainer_adapter": trainer.identity.document(), "evaluator": evaluator.identity.document(),
    })
    stale_staging = tmp_path / "lifecycle" / request_binding[7:] / "artifact.staging"
    stale_staging.mkdir(parents=True)
    (stale_staging / "stale.bin").write_bytes(b"not part of the frozen run")
    results = [first.run_gym_lifecycle(request) for request in requests]
    assert len(training_calls) == 2
    assert all(result["candidate_record"]["training_request_digest"] == result["training_request"]["training_request_digest"] for result in results)
    assert all(result["evaluation_receipt"]["candidate_record_digest"] == result["candidate_record"]["candidate_record_digest"] for result in results)
    assert all(result["scorecard"]["evaluation_receipts"] == [result["evaluation_receipt"]["evaluation_receipt_digest"]] for result in results)
    assert all(result["approval"]["release_authorized"] is False for result in results)
    assert all(result["dataset"]["excluded_attempt_count"] == 1 for result in results)
    assert all(result["dataset"]["source_practice_run_digest"].startswith("sha256:") for result in results)
    for result in results:
        manifest_path = Path(result["candidate_record"]["artifact_root"]) / "artifact" / "artifact_manifest.json"
        assert json.loads(manifest_path.read_text())["kind"] == "lora"
    from neural_foundry.records import canonicalize, content_digest

    for result in results:
        request_digest = result["training_request"]["training_request_digest"]
        artifact_binding = str(content_digest(canonicalize({"request": request_digest})))
        (tmp_path / "trained_artifacts" / f"{artifact_binding[7:]}.json").unlink()
    artifact_root = Path(results[0]["candidate_record"]["artifact_root"])
    published = artifact_root / "artifact"
    staged = artifact_root / "artifact.staging"
    published.rename(staged)
    manifest_path = staged / "artifact_manifest.json"
    original_bytes = manifest_path.read_bytes()
    original_manifest = json.loads(original_bytes)
    forged = {**original_manifest, "kind": "full"}
    forged["manifest_digest"] = _document_digest({key: value for key, value in forged.items() if key != "manifest_digest"})
    manifest_path.write_text(json.dumps(forged))
    with pytest.raises(GymJobError, match="manifest kind differs"):
        studio().run_gym_lifecycle(requests[0])
    assert staged.is_dir() and not published.exists()
    manifest_path.write_bytes(original_bytes)
    resumed = studio()
    replay = resumed.run_gym_lifecycle(requests[0])
    assert len(training_calls) == 2 and replay["candidate_record"] == results[0]["candidate_record"]
    assert replay["artifact_reused"] is True
    assert replay["candidate_record"]["files"] == results[0]["candidate_record"]["files"]
    tools = {"run_id": "tool-run", "task": "bounded-tools@1", "model": "model-llama@1", "evaluator": "public-evaluator@1", "train": False}
    tool_result = first.run_gym_lifecycle(tools)
    assert tool_result["evaluation_receipt"]["evaluation"]["metrics"]["accuracy"] == 1.0
    count = len(inference_calls)
    with pytest.raises(GymJobError, match="rejected before dispatch"):
        first.run_gym_lifecycle({**tools, "run_id": "unsupported-training", "train": True, "trainer": "trainer", "recipe": {"method": "lora", "seed": 17}})
    assert len(inference_calls) == count
    incomplete_evaluator = EvaluatorAdapter(
        AdapterIdentity("incomplete-evaluator", "1", _identity("incomplete-evaluator")),
        lambda *_args, **_kwargs: {"complete": False, "split": "validation", "metrics": {"accuracy": 1.0}},
    )
    first.register_gym_lifecycle_adapter(incomplete_evaluator)
    score_candidate = first.gym_service.score_candidate
    first.gym_service.score_candidate = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("incomplete evidence was scored"))
    incomplete = first.run_gym_lifecycle({
        "run_id": "incomplete-evaluation", "task": "routing", "model": "model-qwen3",
        "evaluator": "incomplete-evaluator", "train": False,
    })
    first.gym_service.score_candidate = score_candidate
    assert incomplete["scorecard"] is None
    assert incomplete["evaluation_status"] == "incomplete_not_scored"

    unowned_holdout = EvaluatorAdapter(
        AdapterIdentity("unowned-holdout", "1", _identity("unowned-holdout")),
        lambda *_args, **_kwargs: {"complete": True, "split": "holdout", "metrics": {"accuracy": 1.0}},
    )
    first.register_gym_lifecycle_adapter(unowned_holdout)
    with pytest.raises(GymJobError, match="requires an evaluator with holdout custody"):
        first.run_gym_lifecycle({
            "run_id": "unowned-holdout", "task": "routing", "model": "model-qwen3",
            "evaluator": "unowned-holdout", "train": False,
        })

    bound_task = replace(
        tasks[0], identity=AdapterIdentity("bound-task", "1", _identity("bound-task")),
        job=replace(tasks[0].job, backends={"evaluator": "different-evaluator@1"}),
    )
    first.register_gym_lifecycle_adapter(bound_task)
    with pytest.raises(GymJobError, match="evaluator adapter differs from the frozen job backend"):
        first.run_gym_lifecycle({
            "run_id": "wrong-backend", "task": "bound-task@1", "model": "model-qwen3@1",
            "evaluator": "public-evaluator@1", "train": False,
        })


def test_generated_tool_json_accepts_fence_but_retains_raw_text() -> None:
    from neural_foundry.gym_adapters import _parse_generated_json

    raw = '```json\n{"tool":"lookup_ticket","arguments":{"ticket_id":"T-401"}}\n```'
    assert _parse_generated_json(raw) == {"tool": "lookup_ticket", "arguments": {"ticket_id": "T-401"}}
    assert _parse_generated_json('{"tool":"lookup_ticket","tool":"other","arguments":{}}') == '{"tool":"lookup_ticket","tool":"other","arguments":{}}'


@pytest.mark.parametrize(("schema", "value", "valid"), [
    ({"type": ["string"]}, "x", True),
    ({"type": "boolean", "enum": [1]}, True, False),
    ({"type": "array", "items": {"type": "integer"}, "enum": [[1, 2]]}, [1, 2], True),
    ({"type": "number"}, 10 ** 400, False),
])
def test_service_boundary_schema_negative_table(schema: Any, value: Any, valid: bool) -> None:
    if valid:
        validate_task_instance(schema, value)
    else:
        with pytest.raises(GymJobError):
            validate_task_instance(schema, value)
    with pytest.raises(GymJobError):
        CandidateIdentity(42, "unchanged_base", _identity("b"), _identity("b"), _identity("a"))


def test_studio_exposes_versioned_dashboard_service_contract() -> None:
    from neural_foundry.studio import WeightProgrammingStudio

    studio = WeightProgrammingStudio()
    job = _contract_review_job()
    refs = [
        ("scenario_provider", row.provider_id, row.provider_version, row.provider_digest)
        for row in job.scenario_sources
    ]
    refs.extend((
        ("scorer", job.scorer.component_id, job.scorer.version, job.scorer.implementation_digest),
        ("verifier", job.verifier.component_id, job.verifier.version, job.verifier.implementation_digest),
    ))
    refs.extend(
        ("candidate_method", row.method_id, row.version, row.implementation_digest)
        for row in job.candidate_methods
    )
    for kind, component_id, version, digest in refs:
        studio.register_gym_component(kind, component_id, version, digest, lambda *_args, **_kwargs: None)

    api = studio.describe_gym_api()
    validation = studio.validate_gym_job(job.to_document())
    assert api["kind"] == "nf_gym_studio_api_v1"
    assert "record_practice_run" in api["operations"]
    assert "run_practice_batch" in api["operations"]
    assert "compare_unchanged_bases" in api["operations"]
    assert api["authority"]["executes_registered_trusted_adapters"] is True
    assert api["authority"]["executes_untrusted_code"] is False
    assert api["authority"]["sandbox_isolation_verified"] is False
    assert callable(studio.run_gym_practice_batch)
    assert callable(studio.compare_unchanged_gym_bases)
    assert validation["valid"] is True
    assert validation["job_digest"] == job.digest
    assert validation["release_authorized"] is False
