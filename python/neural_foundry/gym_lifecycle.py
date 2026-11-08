"""Durable research lifecycle with explicit, replaceable task/model/trainer adapters."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .gym_artifacts import (
    ARTIFACT_KINDS,
    CandidateRef,
    artifact_files,
    read_artifact_manifest,
    write_artifact_manifest,
    write_candidate_manifest,
)
from .gym_job import GymJobError, GymJobSpec, _is_digest, is_finite_number
from .gym_service import (
    CandidateIdentity,
    ExecutionResponse,
    GymStudioService,
    _document_digest,
    _wire_document,
)
from .provenance import sha256_file


@dataclass(frozen=True)
class AdapterIdentity:
    adapter_id: str
    version: str
    implementation_digest: str

    def document(self) -> dict[str, str]:
        if (not isinstance(self.adapter_id, str) or not self.adapter_id.strip() or "@" in self.adapter_id
            or not isinstance(self.version, str) or not self.version.strip() or "@" in self.version
            or not _is_digest(self.implementation_digest)):
            raise GymJobError("adapter requires versioned implementation identity")
        return vars(self).copy()


@dataclass(frozen=True)
class TaskAdapter:
    identity: AdapterIdentity
    job: GymJobSpec
    interaction_type: str
    projection: AdapterIdentity
    interaction_format: str
    project: Callable[[Any, Any], Mapping[str, Any]]
    bind: Callable[[GymStudioService], None]


@dataclass(frozen=True)
class ModelAdapter:
    identity: AdapterIdentity
    candidate: CandidateIdentity
    family: str
    interaction_formats: tuple[str, ...]
    inference_config: Mapping[str, Any]
    infer: Callable[[Mapping[str, Any]], Any]
    loaded_identity: Callable[[], Mapping[str, Any]]
    artifact_kinds: tuple[str, ...] = ("full",)
    artifact_kind: str = "full"
    reference: CandidateRef | None = None
    load_candidate: Callable[[CandidateRef], ModelAdapter] | None = None


@dataclass(frozen=True)
class TrainerAdapter:
    identity: AdapterIdentity
    interaction_formats: tuple[str, ...]
    model_families: tuple[str, ...]
    methods: tuple[str, ...]
    train: Callable[[Mapping[str, Any], Sequence[Mapping[str, Any]], CandidateRef, Path], CandidateRef]
    load: Callable[[Mapping[str, Any], CandidateRef, Path], CandidateRef] | None = None
    artifact_kinds: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluatorAdapter:
    identity: AdapterIdentity
    evaluate: Callable[[GymStudioService, TaskAdapter, ModelAdapter, str], Mapping[str, Any]]
    owns_holdout: bool = False


class GymLifecycle:
    def __init__(self, service: GymStudioService) -> None:
        self.service = service
        self.tasks: dict[str, TaskAdapter] = {}
        self.models: dict[str, ModelAdapter] = {}
        self.trainers: dict[str, TrainerAdapter] = {}
        self.evaluators: dict[str, EvaluatorAdapter] = {}
        self._versions: dict[type, dict[str, dict[str, Any]]] = {
            TaskAdapter: {}, ModelAdapter: {}, TrainerAdapter: {}, EvaluatorAdapter: {},
        }
        self._bound_tasks: set[str] = set()

    def register(self, adapter: TaskAdapter | ModelAdapter | TrainerAdapter | EvaluatorAdapter) -> None:
        adapter.identity.document()
        registry = {TaskAdapter: self.tasks, ModelAdapter: self.models, TrainerAdapter: self.trainers, EvaluatorAdapter: self.evaluators}[type(adapter)]
        versions = self._versions[type(adapter)].setdefault(adapter.identity.adapter_id, {})
        if adapter.identity.version in versions:
            raise GymJobError("lifecycle adapter is already registered")
        versions[adapter.identity.version] = adapter
        if len(versions) == 1:
            registry[adapter.identity.adapter_id] = adapter
        else:
            registry.pop(adapter.identity.adapter_id, None)

    def register_backend(self, selector: str, **configuration: Any) -> None:
        from .backends import discover_backends, registry

        discover_backends()
        self.register(registry.create(selector, **configuration))

    def _resolve(self, adapter_type: type, selector: Any) -> Any:
        if not isinstance(selector, str) or not selector:
            raise GymJobError("unknown or ambiguous lifecycle adapter")
        adapter_id, separator, version = selector.partition("@")
        versions = self._versions[adapter_type].get(adapter_id, {})
        if separator:
            adapter = versions.get(version)
        else:
            adapter = next(iter(versions.values())) if len(versions) == 1 else None
        if adapter is None:
            raise GymJobError("unknown or ambiguous lifecycle adapter")
        return adapter

    def _attach_model(self, task: TaskAdapter, model: ModelAdapter) -> None:
        candidate = model.candidate
        key = (candidate.candidate_id, candidate.candidate_model_digest, candidate.artifact_digest)
        if key in self.service._candidate_executors:
            return

        def execute(*, scenarios: Sequence[Mapping[str, Any]], **kwargs: Any) -> dict[str, Any]:
            responses = {}
            for scenario in scenarios:
                interaction = dict(task.project(scenario["input"], None))
                if interaction.get("format") != task.interaction_format:
                    raise GymJobError("task projection produced an undeclared interaction format")
                response = model.infer(interaction)
                actual = response.response if isinstance(response, ExecutionResponse) else response
                evidence = dict(response.interaction) if isinstance(response, ExecutionResponse) else {}
                evidence.update({"input": scenario["input"], "request": interaction, "response": actual,
                                 "projection": task.projection.document(), "format": task.interaction_format})
                responses[scenario["attempt_id"]] = ExecutionResponse(actual, evidence)
            return responses

        self.service.register_candidate_executor(
            candidate, executor_digest=_document_digest({"model": model.identity.document(), "projection": task.projection.document()}),
            execute=execute, identity_probe=model.loaded_identity,
            inference_config={**model.inference_config, "task_projection": task.projection.document(), "interaction_format": task.interaction_format},
        )

    def _save(self, namespace: str, document: Mapping[str, Any], field: str) -> dict[str, Any]:
        record = json.loads(_wire_document(document))
        record[field] = _document_digest(record)
        self.service._persist_state(namespace, record[field][7:], record)
        return record

    def load(self, namespace: str, digest: str, field: str) -> dict[str, Any]:
        record = self.service._load_state(namespace, digest[7:])
        if record is None or record.get(field) != digest or _document_digest({key: value for key, value in record.items() if key != field}) != digest:
            raise GymJobError("lifecycle record failed digest verification")
        return record

    _artifact_files = staticmethod(artifact_files)
    _read_artifact_manifest = staticmethod(read_artifact_manifest)
    _write_artifact_manifest = staticmethod(write_artifact_manifest)

    def run(self, request: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {"run_id", "task", "model", "trainer", "evaluator", "recipe", "dataset_digest", "train", "approved_responses"}
        if not isinstance(request, Mapping) or set(request) - allowed or not {"run_id", "task", "model", "evaluator"} <= set(request):
            raise GymJobError("lifecycle request has missing or unknown fields")
        if self.service.state_dir is None:
            raise GymJobError("Studio lifecycle requires a durable state directory")
        task = self._resolve(TaskAdapter, request["task"])
        model = self._resolve(ModelAdapter, request["model"])
        evaluator = self._resolve(EvaluatorAdapter, request["evaluator"])
        model = replace(model, candidate=replace(model.candidate, candidate_id=f"{model.candidate.candidate_id}:{task.job.digest[7:19]}"))
        train = request.get("train", True)
        if type(train) is not bool or (train and not isinstance(request.get("recipe"), Mapping)):
            raise GymJobError("training requires a boolean mode and explicit recipe")
        trainer = self._resolve(TrainerAdapter, request["trainer"]) if train and "trainer" in request else None
        for role, adapter in (("task", task), ("model", model), ("trainer", trainer), ("evaluator", evaluator)):
            expected = task.job.backends.get(role)
            if (expected is not None and (role != "trainer" or train)
                and (adapter is None or f"{adapter.identity.adapter_id}@{adapter.identity.version}" != expected)):
                raise GymJobError(f"selected {role} adapter differs from the frozen job backend")
        if task.interaction_format not in model.interaction_formats:
            raise GymJobError("model adapter does not support the task interaction format")
        if train and (trainer is None or task.interaction_format not in trainer.interaction_formats or model.family not in trainer.model_families or request.get("recipe", {}).get("method") not in trainer.methods):
            raise GymJobError("unsupported task/model/trainer combination rejected before dispatch")
        artifact_kind = None
        if train:
            artifact_kind = trainer.artifact_kinds.get(request["recipe"]["method"])
            if not isinstance(artifact_kind, str) or artifact_kind not in ARTIFACT_KINDS or artifact_kind not in model.artifact_kinds:
                raise GymJobError("unsupported trainer artifact kind rejected before dispatch")
            if model.reference is None or model.load_candidate is None:
                raise GymJobError("training requires a model reference and candidate loader")
            if (model.reference.base_digest != model.candidate.base_model_digest
                or model.reference.model_digest != model.candidate.candidate_model_digest
                or model.reference.artifact_digest != model.candidate.artifact_digest
                or model.reference.kind != model.artifact_kind or model.reference.family != model.family):
                raise GymJobError("model reference differs from its loaded candidate")
        if model.artifact_kind not in model.artifact_kinds:
            raise GymJobError("model adapter cannot serve its candidate artifact kind")
        if task.job.digest not in self._bound_tasks:
            task.bind(self.service)
            self._bound_tasks.add(task.job.digest)
        self.service.components.validate_job(task.job)
        task.projection.document()
        run_binding = _document_digest({"kind": "nf_gym_lifecycle_request_v1", "request": dict(request), "job_digest": task.job.digest,
                                        "task_adapter": task.identity.document(), "model_adapter": model.identity.document(),
                                        "trainer_adapter": trainer.identity.document() if trainer else None, "evaluator": evaluator.identity.document()})
        root = self.service.state_dir / "lifecycle" / run_binding[7:]
        root.mkdir(parents=True, mode=0o700, exist_ok=True)
        candidate_model = model
        dataset = None
        training_request = None
        artifact = None
        artifact_reused = False
        if train:
            if request.get("dataset_digest"):
                dataset_digest = request["dataset_digest"]
                dataset = self.load("datasets", dataset_digest, "dataset_digest")
                if dataset["job_digest"] != task.job.digest:
                    raise GymJobError("training dataset belongs to another frozen job")
            else:
                batch = self.service.materialize_public_scenarios(task.job, split="train", seed=int(model.inference_config.get("seed", 0)))
                if request.get("approved_responses") is None:
                    self._attach_model(task, model)
                plan = self.service.freeze_practice_plan(
                    task.job, model.candidate, run_id=f"{request['run_id']}:practice:{run_binding}",
                    scenario_manifest_digest=batch["manifest_digest"], execution_kind="caller_supplied" if request.get("approved_responses") is not None else "registered_candidate",
                )
                pending = [row for row in plan["expected_attempts"] if self.service._lookup_terminal(plan["plan_digest"], row["attempt_id"]) is None]
                if request.get("approved_responses") is not None:
                    responses = request["approved_responses"]
                    self.service.record_practice_run_document({"practice_plan": plan, "attempts": [
                        {"attempt_id": row["attempt_id"], "response": responses[row["scenario_id"]]} for row in pending
                    ]})
                elif pending:
                    for index in range(0, len(pending), 128):
                        self.service.run_practice_batch(task.job, model.candidate, practice_plan_digest=plan["plan_digest"], attempt_ids=tuple(row["attempt_id"] for row in pending[index:index + 128]))
                practice_run = self.service.record_practice_run(plan)
                attempts_by_id = {row["attempt_id"]: row for row in practice_run["attempts"]}
                selected_attempts = [
                    row for row in plan["expected_attempts"]
                    if (attempt := attempts_by_id.get(row["attempt_id"])) is not None
                    and attempt["status"] == "scored" and attempt["output_valid"] and attempt["verifier_passed"]
                ]
                excluded_attempt_count = len(plan["expected_attempts"]) - len(selected_attempts)
                if not selected_attempts:
                    raise GymJobError("no practice attempts passed validation for training")
                selections = [{"practice_plan_digest": plan["plan_digest"], "attempt_id": row["attempt_id"]} for row in selected_attempts]
                dataset = self.service.freeze_training_dataset(
                    task.job, selections, dataset_id=f"{request['run_id']}:dataset",
                    allow_reported_responses=request.get("approved_responses") is not None,
                    source_practice_run_digest=practice_run["run_digest"],
                    excluded_attempt_count=excluded_attempt_count,
                )
                dataset_digest = dataset["dataset_digest"]
            examples = self.service.load_training_examples(dataset_digest)
            projected = []
            for example in examples:
                if example.get("interaction_type", "single_response") != task.interaction_type:
                    raise GymJobError("training interaction type differs from the task adapter")
                interaction = dict(task.project(example["input"], example["response"]))
                if interaction.get("format") != task.interaction_format:
                    raise GymJobError("training projection format is unsupported")
                projected.append({"example_digest": _document_digest(example), "interaction": interaction})
            recipe = dict(request["recipe"])
            if type(recipe.get("seed")) is not int or recipe["seed"] < 0:
                raise GymJobError("frozen training recipe requires an explicit nonnegative seed")
            base_identity = dict(model.loaded_identity())
            base_ref = model.reference
            training_request = self._save("training_requests", {
                "kind": "nf_gym_training_request_v1", "run_binding": run_binding, "job_digest": task.job.digest,
                "dataset_digest": dataset_digest, "projection": task.projection.document(), "interaction_format": task.interaction_format,
                "base": model.candidate.to_document(), "loaded_base": base_identity, "trainer": trainer.identity.document(),
                "recipe": recipe, "artifact_kind": artifact_kind, "base_reference": vars(base_ref),
                "inference_config": dict(model.inference_config),
                "projected_examples_digest": _document_digest({"examples": projected}),
            }, "training_request_digest")
            artifact_binding = _document_digest({"request": training_request["training_request_digest"]})
            saved = self.service._load_state("trained_artifacts", artifact_binding[7:])
            artifact_directory = root / "artifact"
            staging_directory = root / "artifact.staging"

            def materialize(reference: CandidateRef, directory: Path) -> ModelAdapter:
                if (not isinstance(reference, CandidateRef) or reference.kind != artifact_kind
                    or reference.base_digest != base_ref.base_digest or reference.model_id != base_ref.model_id
                    or reference.family != model.family or not reference.candidate_id or not reference.method_id
                    or Path(reference.artifact_path).resolve() != directory.resolve()):
                    raise GymJobError("trainer returned an incompatible candidate reference before publication")
                loaded = model.load_candidate(reference)
                expected = CandidateIdentity(reference.candidate_id, reference.method_id, reference.base_digest,
                                             reference.model_digest, reference.artifact_digest)
                if (loaded.candidate != expected or loaded.family != model.family
                    or loaded.artifact_kind != artifact_kind or artifact_kind not in loaded.artifact_kinds
                    or loaded.loaded_identity().get("state_digest") != reference.model_digest):
                    raise GymJobError("model loader did not verify the candidate reference")
                return loaded

            if saved is not None:
                artifact_reused = True
                artifact = self.load("candidate_records", saved["candidate_record_digest"], "candidate_record_digest")
                manifest = self._read_artifact_manifest(artifact_directory, training_request["training_request_digest"])
                if manifest["candidate"] != artifact["candidate"]:
                    raise GymJobError("published artifact manifest differs from its candidate record")
                for relative, expected in artifact["files"].items():
                    path = (root / relative).resolve()
                    if not path.is_relative_to(root) or not path.is_file() or sha256_file(path) != expected:
                        raise GymJobError("committed trained artifact failed file verification")
                if trainer.load is None:
                    raise GymJobError("trainer adapter does not support committed artifact reload")
                candidate_model = materialize(trainer.load(training_request, base_ref, artifact_directory), artifact_directory)
            else:
                if trainer.load is None:
                    raise GymJobError("trainer adapter must reload a staged artifact before publication")
                if artifact_directory.exists():
                    manifest = self._read_artifact_manifest(artifact_directory, training_request["training_request_digest"])
                    candidate_model = materialize(trainer.load(training_request, base_ref, artifact_directory), artifact_directory)
                    artifact_reused = True
                else:
                    if staging_directory.exists() and (staging_directory / "artifact_manifest.json").is_file():
                        manifest = self._read_artifact_manifest(staging_directory, training_request["training_request_digest"])
                        staged_kind = manifest["kind"] if manifest.get("schema_version") == 2 else manifest.get("artifact_kind", artifact_kind)
                        if staged_kind != artifact_kind:
                            raise GymJobError("artifact manifest kind differs from the frozen training request")
                        staged_candidate = materialize(trainer.load(training_request, base_ref, staging_directory), staging_directory)
                        if staged_candidate.candidate.to_document() != manifest["candidate"]:
                            raise GymJobError("staged artifact identity differs from its manifest")
                        os.replace(staging_directory, artifact_directory)
                        candidate_model = materialize(trainer.load(training_request, base_ref, artifact_directory), artifact_directory)
                        artifact_reused = True
                    else:
                        if staging_directory.is_symlink():
                            raise GymJobError("training staging directory must not be a symlink")
                        if staging_directory.exists():
                            shutil.rmtree(staging_directory)
                        trained_ref = trainer.train(training_request, projected, base_ref, staging_directory)
                        candidate_model = materialize(trained_ref, staging_directory)
                        reloaded_ref = trainer.load(training_request, base_ref, staging_directory)
                        if reloaded_ref != trained_ref:
                            raise GymJobError("fresh staged artifact reload differs from the trainer result")
                        staged_reload = materialize(reloaded_ref, staging_directory)
                        files = self._artifact_files(staging_directory)
                        if not files:
                            raise GymJobError("trainer must save a durable staged artifact")
                        manifest = write_candidate_manifest(
                            staging_directory, model_id=base_identity["model_id"], kind=artifact_kind,
                            base_digest=model.candidate.base_model_digest,
                            model_digest=candidate_model.candidate.candidate_model_digest,
                            request_digest=training_request["training_request_digest"],
                            metadata={"candidate": candidate_model.candidate.to_document(),
                                      "model_adapter": candidate_model.identity.document(),
                                      "loaded_candidate": dict(staged_reload.loaded_identity())},
                        )
                        manifest = self._read_artifact_manifest(staging_directory, training_request["training_request_digest"])
                        os.replace(staging_directory, artifact_directory)
                        candidate_model = materialize(trainer.load(training_request, base_ref, artifact_directory), artifact_directory)
                        if candidate_model.candidate.to_document() != manifest["candidate"]:
                            raise GymJobError("published artifact reload differs from its staged manifest")
                if candidate_model.candidate.to_document() != manifest["candidate"]:
                    raise GymJobError("recovered artifact identity differs from its atomic publication manifest")
            if candidate_model.candidate.base_model_digest != model.candidate.base_model_digest or candidate_model.family != model.family:
                raise GymJobError("trainer returned an artifact from another base/family")
            if candidate_model.artifact_kind != artifact_kind or artifact_kind not in candidate_model.artifact_kinds:
                raise GymJobError("reloaded candidate artifact kind differs from the frozen training request")
            manifest_kind = manifest["kind"] if manifest.get("schema_version") == 2 else manifest.get("artifact_kind", artifact_kind)
            if manifest_kind != artifact_kind:
                raise GymJobError("artifact manifest kind differs from the frozen training request")
            files = {str((Path("artifact") / relative).as_posix()): digest for relative, digest in self._artifact_files(artifact_directory).items()}
            files["artifact/artifact_manifest.json"] = sha256_file(artifact_directory / "artifact_manifest.json")
            candidate_record = {
                "kind": "nf_gym_candidate_record_v1", "job_digest": task.job.digest, "dataset_digest": dataset_digest,
                "training_request_digest": training_request["training_request_digest"], "candidate": candidate_model.candidate.to_document(),
                "model_adapter": candidate_model.identity.document(), "artifact_root": str(root), "files": files,
                "loaded_candidate": dict(candidate_model.loaded_identity()), "release_authorized": False,
            }
            if saved is None:
                artifact = self._save("candidate_records", candidate_record, "candidate_record_digest")
                self.service._persist_state("trained_artifacts", artifact_binding[7:], artifact)
            elif candidate_model.candidate.to_document() != artifact["candidate"]:
                raise GymJobError("reloaded trainer artifact differs from the committed candidate")
        else:
            artifact = self._save("candidate_records", {
                "kind": "nf_gym_candidate_record_v1", "job_digest": task.job.digest, "dataset_digest": None,
                "training_request_digest": None, "candidate": model.candidate.to_document(), "model_adapter": model.identity.document(),
                "loaded_candidate": dict(model.loaded_identity()), "release_authorized": False,
            }, "candidate_record_digest")
        evaluation = dict(evaluator.evaluate(self.service, task, candidate_model, f"{request['run_id']}:{run_binding}:evaluation"))
        if evaluation.get("split") == "holdout" and evaluator.owns_holdout is not True:
            raise GymJobError("holdout evaluation requires an evaluator with holdout custody")
        metrics = evaluation.get("metrics")
        if not isinstance(metrics, Mapping) or any(not isinstance(name, str) or not is_finite_number(value) for name, value in metrics.items()):
            raise GymJobError("evaluator must return finite typed metrics")
        receipt = self._save("evaluation_receipts", {
            "kind": "nf_gym_evaluation_receipt_v1", "job_digest": task.job.digest,
            "candidate_record_digest": artifact["candidate_record_digest"], "artifact_digest": candidate_model.candidate.artifact_digest,
            "evaluator": evaluator.identity.document(), "owns_holdout": evaluator.owns_holdout,
            "evaluation": evaluation, "independent_custody_verified": False,
        }, "evaluation_receipt_digest")
        checked = self.load("evaluation_receipts", receipt["evaluation_receipt_digest"], "evaluation_receipt_digest")
        if evaluation.get("complete") is not True:
            return self._save("lifecycle_runs", {
                "kind": "nf_gym_studio_lifecycle_v1", "run_id": request["run_id"], "run_binding": run_binding,
                "job_digest": task.job.digest, "dataset": dataset, "training_request": training_request,
                "artifact_reused": artifact_reused, "candidate_record": artifact,
                "evaluation_receipt": checked, "scorecard": None,
                "evaluation_status": "incomplete_not_scored",
                "approval": {"status": "blocked_incomplete_evaluation", "release_authorized": False},
            }, "lifecycle_digest")
        measurements = {gate.gate_id: {
            "metric_id": gate.metric_id, "split": gate.split, "value": metrics[gate.metric_id],
            "evidence_digest": receipt["evaluation_receipt_digest"], "evidence_class": "measured",
        } for gate in task.job.gates if gate.metric_id in metrics and gate.split == evaluation.get("split")}
        scorecard = self.service.score_candidate(task.job, candidate_model.candidate, measurements)
        scorecard["evaluation_receipts"] = [checked["evaluation_receipt_digest"]]
        scorecard["candidate_record_digest"] = artifact["candidate_record_digest"]
        scorecard["scorecard_digest"] = _document_digest({key: value for key, value in scorecard.items() if key != "scorecard_digest"})
        result = self._save("lifecycle_runs", {
            "kind": "nf_gym_studio_lifecycle_v1", "run_id": request["run_id"], "run_binding": run_binding,
            "job_digest": task.job.digest, "dataset": dataset, "training_request": training_request,
            "artifact_reused": artifact_reused,
            "candidate_record": artifact, "evaluation_receipt": receipt, "scorecard": scorecard,
            "approval": {"status": "blocked_pending_independent_custody_and_owner_approval", "release_authorized": False},
        }, "lifecycle_digest")
        return result


def public_panel_evaluator(service: GymStudioService, task: TaskAdapter, model: ModelAdapter, run_id: str) -> Mapping[str, Any]:
    lifecycle = GymLifecycle(service)
    lifecycle._attach_model(task, model)
    panel = service.materialize_public_scenarios(task.job, split="validation", seed=int(model.inference_config.get("seed", 0)))
    plan = service.freeze_practice_plan(task.job, model.candidate, run_id=run_id, scenario_manifest_digest=panel["manifest_digest"])
    missing = tuple(row["attempt_id"] for row in plan["expected_attempts"] if service._lookup_terminal(plan["plan_digest"], row["attempt_id"]) is None)
    for index in range(0, len(missing), 128):
        service.run_practice_batch(task.job, model.candidate, practice_plan_digest=plan["plan_digest"], attempt_ids=missing[index:index + 128])
    run = service.record_practice_run(plan)
    metrics: dict[str, float] = {}
    for attempt in run["attempts"]:
        for name, value in attempt["metrics"].items():
            metrics[name] = metrics.get(name, 0.0) + value / len(run["expected_attempt_ids"])
    return {"split": "validation", "panel_digest": panel["manifest_digest"], "run_digest": run["run_digest"],
            "complete": run["status"] == "complete", "metrics": metrics, "serving_identity": dict(model.loaded_identity())}