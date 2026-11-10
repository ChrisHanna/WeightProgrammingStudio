"""Config-driven, multi-family CUDA generation and in-flight recovery evidence; no release authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from neural_foundry.gym_job import (
    AcceptanceGate,
    CandidateMethod,
    ComponentReference,
    GymComponentRegistry,
    GymJobSpec,
    ScenarioSource,
)
from neural_foundry.gym_service import CandidateIdentity, GymStudioService
from neural_foundry.provenance import (
    CheckpointUnavailableError,
    parse_strict_json,
    resolve_checkpoint,
    sha256_file,
)
from neural_foundry.serving import TransformersAdapter
from run_gym_cuda_comparison import _digest, _require, _supervise_partial, _write


def _training_evidence(path: Path, output: Path) -> None:
    from neural_foundry.baselines import merged_model
    from neural_foundry.cumulative_edit import train_restricted_external
    from neural_foundry.edit_plan import tensor_state_digest
    from neural_foundry.export_formats import (
        export_huggingface_safetensors,
        reload_and_verify_huggingface_safetensors,
    )
    from neural_foundry.gpu import probe_external_gpu_runtime
    from neural_foundry.pythia import PythiaModel

    config = parse_strict_json(path.read_bytes())
    checkpoint = resolve_checkpoint(config["base"]["path"], revision=config["base"]["revision"])
    model = PythiaModel.from_pretrained_dir(checkpoint.path, model_name=config["base"]["model_id"])
    model.checkpoint_identity = checkpoint
    runtime = probe_external_gpu_runtime(sys.executable, device=config["runtime"]["device"])
    jobs = []
    for index, recipe in enumerate(config["recipes"]):
        _require(recipe["kind"] in {"lora", "sft"}, "unsupported training method")
        _require(type(recipe["steps"]) is int and recipe["steps"] > 0 and recipe["checkpoints"], "invalid training checkpoints")
        jobs.append({
            "job_id": f"{recipe['kind']}-{index}", "kind": recipe["kind"], "recipe_index": index,
            "recipe_schema_version": 1, "sites": config["sites"], "recipe": recipe,
            "dev": config["dev"], "guard": config["guard"], "seed": recipe["seed"],
            "gym_id": config["gym_id"], "gym_digest": _digest({"gym": config["gym_id"]}), "dataset_digest": _digest({"dev": config["dev"], "guard": config["guard"]}),
        })
    checkpoints = train_restricted_external(
        model, jobs, runtime, timeout_seconds=config["runtime"]["timeout_seconds"], training_precision=config["runtime"]["precision"],
    )
    _require(len(checkpoints) == sum(len(job["recipe"]["checkpoints"]) for job in jobs), "worker omitted training checkpoints")
    records = []
    for job in jobs:
        selected = max((row for row in checkpoints if row["job_id"] == job["job_id"]), key=lambda row: row["step"])
        candidate = merged_model(model, selected["weights"])
        directory = output / job["job_id"]
        export_huggingface_safetensors(candidate, directory, dtype="f64", source_checkpoint_dir=checkpoint.path)
        reloaded, receipt = reload_and_verify_huggingface_safetensors(directory, candidate, config["reload_prompts"])
        _require(receipt.gain_preserved_after_rebuild and receipt.max_logit_abs_diff <= 1e-8, "fresh exported-runtime reload failed parity")
        records.append({
            "job": job, "completed_step": selected["step"], "wall_seconds": selected["wall_seconds"],
            "input_state_digest": tensor_state_digest(model.tensors), "candidate_state_digest": tensor_state_digest(candidate.tensors),
            "reload_state_digest": tensor_state_digest(reloaded.tensors), "reload_max_logit_abs_diff": receipt.max_logit_abs_diff,
            "reload_match_rate": receipt.behavioral_match_rate, "checkpoint_files": checkpoint.file_digests,
            "device": runtime.device, "precision": config["runtime"]["precision"], "torch_version": runtime.torch_version,
            "worker_runtime": selected["worker_runtime"],
        })
    report = {"kind": "nf_real_gpu_training_reload_v1", "config_digest": _digest(config), "jobs": records,
              "evidence_scope": "two_step_training_execution_smoke", "release_authorized": False, "task_quality_qualified": False}
    report["report_digest"] = _digest(report)
    _write(output / "report.json", report)
    print(json.dumps({"report_digest": report["report_digest"], "jobs": len(records), "reload_match_rates": [row["reload_match_rate"] for row in records]}, indent=2))


def _load_config(path: Path) -> dict[str, Any]:
    config = parse_strict_json(path.read_bytes())
    _require(isinstance(config, dict) and set(config) == {"models", "runtime", "task", "panels"}, "invalid run configuration fields")
    _require(len(config["models"]) == 2, "exactly two pinned bases are required")
    for model in config["models"]:
        revision = model.get("revision")
        _require(isinstance(revision, str) and len(revision) == 40 and all(char in "0123456789abcdef" for char in revision), "model revision must be immutable")
    runtime = config["runtime"]
    for name in ("context_tokens", "max_new_tokens", "seed"):
        _require(type(runtime.get(name)) is int and runtime[name] >= (0 if name == "seed" else 1), f"invalid {name}")
    _require(runtime["max_new_tokens"] < runtime["context_tokens"], "output budget exceeds context")
    _require(runtime["precision"] in {"float32", "float16", "bfloat16"}, "unsupported precision")
    return config


def _job(config: dict[str, Any]) -> tuple[GymJobSpec, GymComponentRegistry]:
    implementation = sha256_file(__file__)
    registry = GymComponentRegistry()

    def provide(*, split: str, **kwargs: Any) -> list[dict[str, Any]]:
        return [
            {"scenario_id": f"{split}-{index}", "input": {"contract_text": row["contract_text"]}, "reference": {"clauses": row["expected_clauses"]}}
            for index, row in enumerate(config["panels"][split])
        ]

    def verify(*, scenario: dict[str, Any], reference: dict[str, Any], response: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        text = scenario["input"]["contract_text"]
        grounded = all(0 <= clause["start"] < clause["end"] <= len(text) and text[clause["start"]:clause["end"]] == clause["text"] for clause in response["clauses"])
        expected = {(row["clause_type"], row["text"]) for row in reference["clauses"]}
        observed = {(row["clause_type"], row["text"]) for row in response["clauses"]}
        return {"passed": grounded and observed == expected, "grounded": grounded}

    def score(*, verification: dict[str, Any], **kwargs: Any) -> dict[str, float]:
        return {"exact_contract_accuracy": float(verification["passed"]), "grounded": float(verification["grounded"])}

    for kind, name, function in (
        ("scenario_provider", "configured-contracts", provide), ("verifier", "exact-source-quotes", verify),
        ("scorer", "exact-contract-score", score), ("candidate_method", "unchanged_base", lambda **kwargs: None),
    ):
        registry.register(kind, name, "1.0.0", implementation, function)
    task = config["task"]
    job = GymJobSpec(
        job_id=task["job_id"], title=task["title"], description=task["instructions"],
        task_input_schema={"type": "object", "properties": {"contract_text": {"type": "string", "maxLength": 16384}}, "required": ["contract_text"], "additionalProperties": False},
        task_output_schema=task["output_schema"],
        scenario_sources=tuple(ScenarioSource(
            f"configured-{role}", "configured-contracts", "1.0.0", implementation, _digest(rows), "simulated", (role,), {"role": role},
        ) for role, rows in config["panels"].items()),
        scorer=ComponentReference("exact-contract-score", "1.0.0", implementation, {}),
        verifier=ComponentReference("exact-source-quotes", "1.0.0", implementation, {}),
        candidate_methods=(CandidateMethod("unchanged_base", "1.0.0", implementation, {}),),
        gates=(
            AcceptanceGate("quality", "task", "exact_contract_accuracy", "validation", "ge", 0.9),
            AcceptanceGate("behavior", "behavior", "grounded", "holdout", "eq", 1.0),
            AcceptanceGate("controls", "controls", "beats_control", "controls", "eq", 1.0),
            AcceptanceGate("reproducibility", "reproducibility", "repeat", "reproducibility", "eq", 1.0),
        ),
        metadata={"runtime": config["runtime"], "run_config_digest": _digest(config), "release_authorized": False},
    )
    return job, registry


def _state_digest(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}\0".encode())
        digest.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def _worker(config: dict[str, Any], output: Path, phase: str, fault: str) -> None:
    job, registry = _job(config)
    service = GymStudioService(registry, state_dir=output / "state")
    panel = service.materialize_public_scenarios(job, split="validation", seed=config["runtime"]["seed"])
    candidates = []
    responses: list[dict[str, Any]] = []
    dispatches: list[dict[str, Any]] = []
    loaded: list[dict[str, Any]] = []
    execution_errors: list[dict[str, str]] = []
    runtime = config["runtime"]

    def bind(index: int, entry: dict[str, Any]) -> None:
        checkpoint = resolve_checkpoint(entry["model_id"], revision=entry["revision"])
        base_digest = _digest({"model_id": entry["model_id"], "revision": entry["revision"], "files": checkpoint.file_digests})
        candidate = CandidateIdentity(entry["model_id"].rsplit("/", 1)[-1], "unchanged_base", base_digest, base_digest, _digest({"base": base_digest, "runtime": runtime}))
        candidates.append(candidate)
        adapter: TransformersAdapter | None = None
        identity: dict[str, Any] | None = None

        def probe() -> dict[str, Any]:
            nonlocal adapter, identity
            if adapter is None:
                adapter = TransformersAdapter(checkpoint.path, dtype=runtime["precision"], device=runtime["device"], max_context_tokens=runtime["context_tokens"])
                _require(resolve_checkpoint(checkpoint.path).file_digests == checkpoint.file_digests, "model assets changed while loading")
                parameter = next(adapter.model.parameters())
                _require(parameter.device == torch.device(runtime["device"]) and parameter.dtype == getattr(torch, runtime["precision"]), "worker ignored assigned device or precision")
                identity = {
                    "model_id": entry["model_id"], "revision": entry["revision"], "checkpoint_digest": base_digest,
                    "config_digest": checkpoint.file_digests["config.json"], "tokenizer_digest": checkpoint.file_digests["tokenizer.json"],
                    "state_digest": _state_digest(adapter.model), "model_family": adapter.model.config.model_type,
                    "device": str(parameter.device), "precision": runtime["precision"], "local_loaded_files_verified": True,
                    "device_name": torch.cuda.get_device_name(parameter.device), "torch_version": torch.__version__,
                    "transformers_version": adapter._transformers.__version__, "context_tokens": runtime["context_tokens"],
                    "file_digests": checkpoint.file_digests,
                }
                loaded.append(identity)
            if identity is None:
                raise RuntimeError("missing loaded-model identity")
            return identity

        def interrupt(row: dict[str, Any], event: dict[str, Any], *, pause: bool = True) -> None:
            plan = next(value for value in service._practice_plans.values() if value["candidate"]["candidate_id"] == candidate.candidate_id)
            _write(output / "fault-event.json", {
                "kind": "nf_cuda_inflight_fault_v1", "fault": fault, "identity": identity,
                "attempt_id": row["attempt_id"], "practice_plan": plan, "event": event,
            })
            print("NF_CUDA_FAULT_POINT", flush=True)
            if pause:
                sys.stdin.read(1)
                raise RuntimeError("fault worker must be killed by its supervisor")

        def execute(*, scenarios: tuple[dict[str, Any], ...], **kwargs: Any) -> dict[str, Any]:
            if adapter is None:
                raise RuntimeError("worker identity probe did not load the model")
            dispatches.append({"candidate_id": candidate.candidate_id, "attempt_ids": [row["attempt_id"] for row in scenarios]})
            results = {}
            for position, row in enumerate(scenarios):
                messages = [{"role": "system", "content": config["task"]["instructions"]}, {"role": "user", "content": row["input"]["contract_text"]}]
                prompt = adapter.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                hook = None
                if phase == "fault" and index == 1 and position == 0 and fault == "during_inference":
                    announced = False

                    def during(module: Any, inputs: Any, result: Any, selected_row: dict[str, Any] = row) -> None:
                        nonlocal announced
                        if announced:
                            return
                        announced = True
                        torch.cuda.synchronize(parameter_device := next(adapter.model.parameters()).device)
                        logits = result.logits[:, -1]
                        _require(bool(torch.isfinite(logits).all()), "nonfinite real CUDA logits")
                        interrupt(selected_row, {
                            "cuda_forward_completed": True, "logits_shape": list(logits.shape),
                            "device": str(parameter_device), "generation_continues_after_marker": True,
                        }, pause=False)

                    hook = adapter.model.register_forward_hook(during)
                try:
                    generated = adapter.generate(prompt, {"seed": runtime["seed"], "temperature": 0.0, "max_tokens": runtime["max_new_tokens"]})
                finally:
                    if hook is not None:
                        hook.remove()
                actual: Any = generated.text
                try:
                    actual = parse_strict_json(generated.text.encode())
                except (ValueError, UnicodeDecodeError):
                    pass
                record = {"candidate_id": candidate.candidate_id, "attempt_id": row["attempt_id"], "raw_response": generated.text,
                          "token_ids": list(generated.token_ids), "seconds": generated.duration_seconds, "parsed_response": actual}
                responses.append(record)
                if phase == "fault" and index == 1 and position == 0 and fault == "before_persistence":
                    interrupt(row, {"generated_response_not_persisted": True, "generation": record})
                results[row["attempt_id"]] = actual
            return results

        def measured_execute(**kwargs: Any) -> dict[str, Any]:
            try:
                return execute(**kwargs)
            except Exception as exc:
                execution_errors.append({"candidate_id": candidate.candidate_id, "type": type(exc).__name__, "message": str(exc)})
                raise

        service.register_candidate_executor(candidate, executor_digest=sha256_file(__file__), execute=measured_execute, identity_probe=probe)

    for index, model in enumerate(config["models"]):
        bind(index, model)
    comparison = service.compare_unchanged_bases(job, candidates, scenario_manifest_digest=panel["manifest_digest"], require_worker_identity=True, require_distinct_families=True)
    receipt = {"kind": "nf_generated_cuda_comparison_v1", "config_digest": _digest(config), "phase": phase,
               "comparison": comparison, "loaded_models": loaded, "dispatches": dispatches, "generated_responses": responses,
               "execution_errors": execution_errors,
               "authority": {"release_authorized": False, "independent_holdout_custody": False}}
    receipt["receipt_digest"] = _digest(receipt)
    _write(output / f"{phase}.json", receipt)
    _require(comparison["status"] == "complete", f"generated run has retryable infrastructure outcomes: {execution_errors}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--worker", choices=("fault", "resume", "cached"))
    parser.add_argument("--fault", choices=("during_inference", "before_persistence"))
    parser.add_argument("--startup-seconds", type=float, default=600)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--training-only", action="store_true")
    args = parser.parse_args()
    _require(sys.platform.startswith("linux") and os.name == "posix", "this supervised CUDA run requires Linux/WSL")
    _require(0 < args.startup_seconds < float("inf"), "startup deadline must be finite and positive")
    config = _load_config(args.config)
    output = args.output_dir.resolve()
    if args.worker:
        _worker(config, output, args.worker, args.fault)
        return
    _require(not output.exists() or not any(output.iterdir()), "choose a fresh output directory")
    output.mkdir(parents=True, mode=0o700, exist_ok=True)
    if args.training_only:
        _require(args.training_config is not None, "training-only requires explicit recipes")
        _training_evidence(args.training_config, output)
        return
    for model in config["models"]:
        try:
            resolve_checkpoint(model["model_id"], revision=model["revision"])
        except CheckpointUnavailableError:
            _require(args.allow_download, "checkpoint is absent; downloads require --allow-download")
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=model["model_id"], revision=model["revision"], token=False,
                              allow_patterns=["*.safetensors", "*.json", "tokenizer.model", "README.md"])
    summaries = []
    for fault in ("during_inference", "before_persistence"):
        directory = output / fault
        directory.mkdir(mode=0o700)
        command = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config.resolve()), "--output-dir", str(directory), "--fault", fault, "--worker"]
        code = _supervise_partial([*command, "fault"], directory, args.startup_seconds, marker=b"NF_CUDA_FAULT_POINT")
        _require(code == -9, "fault worker was not killed")
        for phase in ("resume", "cached"):
            with (directory / f"{phase}.stderr.log").open("wb") as errors:
                subprocess.run([*command, phase], check=True, timeout=args.startup_seconds, stderr=errors)
        event, resume, cached = [json.loads((directory / name).read_bytes()) for name in ("fault-event.json", "resume.json", "cached.json")]
        expected = {row["attempt_id"] for row in event["practice_plan"]["expected_attempts"]}
        _require(len(resume["dispatches"]) == 1 and set(resume["dispatches"][0]["attempt_ids"]) == expected, "retry did not execute exactly the unpersisted second-base roster")
        _require(not cached["dispatches"] and cached["comparison"] == resume["comparison"], "completed retry dispatched work or changed evidence")
        _require(resume["comparison"]["authority"]["distinct_model_families_verified"], "worker did not verify distinct families")
        summaries.append({"fault": fault, "killed_returncode": code, "retried_attempts": len(expected),
                          "completed_retry_dispatches": len(cached["dispatches"]), "receipt_digest": resume["receipt_digest"]})
    report = {"kind": "nf_generated_cuda_fault_recovery_v1", "config_digest": _digest(config), "faults": summaries,
              "authority": {"release_authorized": False, "independent_holdout_custody": False, "task_quality_qualified": False}}
    report["report_digest"] = _digest(report)
    _write(output / "report.json", report)
    if args.training_config is not None:
        training_output = output / "training"
        training_output.mkdir(mode=0o700)
        _training_evidence(args.training_config, training_output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()