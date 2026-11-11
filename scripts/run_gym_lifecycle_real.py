"""Exercise the shared Studio lifecycle with real model/trainer adapters; receipts are not release approval."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from neural_foundry.gym_adapters import peft_sequence_trainer, transformers_model
from neural_foundry.gym_job import (
    AcceptanceGate,
    CandidateMethod,
    ComponentReference,
    GymComponentRegistry,
    GymJobError,
    GymJobSpec,
    ScenarioSource,
)
from neural_foundry.gym_lifecycle import (
    AdapterIdentity,
    EvaluatorAdapter,
    GymLifecycle,
    TaskAdapter,
    public_panel_evaluator,
)
from neural_foundry.gym_service import _document_digest
from neural_foundry.provenance import resolve_checkpoint, sha256_file
from neural_foundry.studio import WeightProgrammingStudio


def _build_support_task(registry: GymComponentRegistry) -> TaskAdapter:
    implementation = sha256_file(__file__)
    panel = {
        "train": ({"ticket_id": "T-100", "ticket": "A duplicate card charge needs a refund.", "labels": ["billing", "technical"]},),
        "validation": ({"ticket_id": "T-200", "ticket": "A password reset link is not working.", "labels": ["billing", "technical"]},),
        "holdout": ({"ticket_id": "T-300", "ticket": "A refund has not appeared on my statement.", "labels": ["billing", "technical"]},),
    }
    references = {"train": "billing", "validation": "technical", "holdout": "billing"}
    sources = tuple(
        ScenarioSource(
            f"support-{split}", "support-routing-provider", "1.0.0", implementation,
            _document_digest({"rows": rows, "split": split}), "simulated", (split,), {"split": split},
        )
        for split, rows in panel.items()
    )
    job = GymJobSpec(
        job_id="support-routing-lifecycle-v1",
        title="Route support tickets",
        description="Choose the single best support queue from the allowed labels.",
        task_input_schema={
            "type": "object", "properties": {
                "ticket_id": {"type": "string"}, "ticket": {"type": "string"},
                "labels": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
            }, "required": ["ticket_id", "ticket", "labels"], "additionalProperties": False,
        },
        task_output_schema={"type": "string", "enum": ["billing", "technical"]},
        scenario_sources=sources,
        scorer=ComponentReference("support-accuracy", "1.0.0", implementation, {}),
        verifier=ComponentReference("support-exact-label", "1.0.0", implementation, {}),
        candidate_methods=(
            CandidateMethod("unchanged_base", "1.0.0", implementation, {}),
            CandidateMethod("lora", "1.0.0", implementation, {}),
        ),
        gates=(
            AcceptanceGate("task", "task", "accuracy", "validation", "ge", 1.0),
            AcceptanceGate("behavior", "behavior", "accuracy", "holdout", "ge", 1.0),
            AcceptanceGate("controls", "controls", "beats_control", "controls", "eq", 1.0),
            AcceptanceGate("reproducibility", "reproducibility", "repeat", "reproducibility", "eq", 1.0),
        ),
        metadata={"purpose": "reuse_execution_evidence_not_customer_data"},
    )

    def provider(*, split: str, **_kwargs: Any) -> list[dict[str, Any]]:
        row = panel[split][0]
        return [{
            "scenario_id": f"support-{split}-{row['ticket_id']}", "item_id": row["ticket_id"],
            "family_id": f"support-family-{row['ticket_id']}", "interaction_type": "single_response",
            "input": row, "reference": references[split],
        }]

    def verifier(*, reference: str, response: Any, output_valid: bool, **_kwargs: Any) -> dict[str, bool]:
        return {"passed": output_valid and response == reference}

    def scorer(*, verification: Mapping[str, Any], **_kwargs: Any) -> dict[str, float]:
        return {"accuracy": float(verification["passed"])}

    registry.register("scenario_provider", "support-routing-provider", "1.0.0", implementation, provider)
    registry.register("verifier", "support-exact-label", "1.0.0", implementation, verifier)
    registry.register("scorer", "support-accuracy", "1.0.0", implementation, scorer)
    for method in job.candidate_methods:
        try:
            registry.register("candidate_method", method.method_id, method.version, method.implementation_digest, lambda **_kwargs: None)
        except GymJobError:
            registry.resolve("candidate_method", method.method_id, method.version, method.implementation_digest)

    def project(input_value: Mapping[str, Any], response: Any) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": job.description + " Return only one allowed label."},
            {"role": "user", "content": json.dumps(input_value, ensure_ascii=False, sort_keys=True)},
        ]
        if response is not None:
            messages.append({"role": "assistant", "content": str(response)})
        return {"format": "chat_messages_v1", "messages": messages}

    return TaskAdapter(
        AdapterIdentity("support-routing-task", "1.0.0", implementation), job,
        "single_response", AdapterIdentity("structured-ticket-chat-projection", "1.0.0", implementation),
        "chat_messages_v1", project, lambda _service: None,
    )


def _build_tool_task(registry: GymComponentRegistry) -> tuple[TaskAdapter, dict[str, Any]]:
    implementation = sha256_file(__file__)
    item = {"request": "Find the queue for ticket T-401.", "ticket_id": "T-401"}
    source = ScenarioSource(
        "bounded-tool-source", "bounded-tool-provider", "1.0.0", implementation,
        _document_digest({"item": item}), "simulated", ("train", "validation", "holdout"), {},
    )
    job = GymJobSpec(
        job_id="bounded-tool-lifecycle-v1", title="Use one bounded lookup tool",
        description="Use exactly one authorized lookup tool for the requested ticket.",
        task_input_schema={"type": "object", "properties": {"request": {"type": "string"}, "ticket_id": {"type": "string"}},
                           "required": ["request", "ticket_id"], "additionalProperties": False},
        task_output_schema={"type": "object", "properties": {"calls": {"type": "array", "minItems": 1, "maxItems": 1,
            "items": {"type": "object", "properties": {"name": {"type": "string", "enum": ["lookup_ticket"]},
            "arguments": {"type": "object", "properties": {"ticket_id": {"type": "string"}}, "required": ["ticket_id"], "additionalProperties": False},
            "result": {"type": "object", "properties": {"queue": {"type": "string"}}, "required": ["queue"], "additionalProperties": False}},
            "required": ["name", "arguments", "result"], "additionalProperties": False}},},
            "required": ["calls"], "additionalProperties": False},
        scenario_sources=(source,),
        scorer=ComponentReference("bounded-tool-score", "1.0.0", implementation, {}),
        verifier=ComponentReference("bounded-tool-verifier", "1.0.0", implementation, {}),
        candidate_methods=(CandidateMethod("unchanged_base", "1.0.0", implementation, {}),),
        gates=(
            AcceptanceGate("tool-task", "task", "tool_success", "validation", "ge", 1.0),
            AcceptanceGate("tool-behavior", "behavior", "tool_success", "holdout", "ge", 1.0),
            AcceptanceGate("tool-controls", "controls", "beats_control", "controls", "eq", 1.0),
            AcceptanceGate("tool-reproducibility", "reproducibility", "repeat", "reproducibility", "eq", 1.0),
        ),
        metadata={"interaction": "bounded_single_tool_call"},
    )

    def provider(*, split: str, **_kwargs: Any) -> list[dict[str, Any]]:
        return [{"scenario_id": f"tool-{split}-T401", "item_id": f"tool-T401-{split}", "family_id": f"tool-family-T401-{split}",
                 "interaction_type": "tool_episode", "input": item, "reference": {"queue": "billing"}}]

    def verifier(*, response: Any, reference: Mapping[str, Any], output_valid: bool, **_kwargs: Any) -> dict[str, bool]:
        passed = output_valid and len(response["calls"]) == 1 and response["calls"][0]["name"] == "lookup_ticket" and response["calls"][0]["result"] == reference
        return {"passed": passed}

    def scorer(*, verification: Mapping[str, Any], **_kwargs: Any) -> dict[str, float]:
        return {"tool_success": float(verification["passed"])}

    registry.register("scenario_provider", source.provider_id, source.provider_version, source.provider_digest, provider)
    registry.register("verifier", job.verifier.component_id, job.verifier.version, job.verifier.implementation_digest, verifier)
    registry.register("scorer", job.scorer.component_id, job.scorer.version, job.scorer.implementation_digest, scorer)
    try:
        registry.register("candidate_method", "unchanged_base", "1.0.0", implementation, lambda **_kwargs: None)
    except GymJobError:
        registry.resolve("candidate_method", "unchanged_base", "1.0.0", implementation)

    def project(input_value: Mapping[str, Any], _response: Any) -> dict[str, Any]:
        return {"format": "tool_messages_v1", "max_tool_calls": 1,
                "tools": [{"name": "lookup_ticket", "description": "Look up a ticket queue.",
                           "arguments": {"type": "object", "properties": {"ticket_id": {"type": "string"}},
                                         "required": ["ticket_id"], "additionalProperties": False}}],
                "messages": [
                    {"role": "system", "content": "Use one lookup to answer this request."},
                    {"role": "user", "content": json.dumps(input_value, sort_keys=True)},
                ]}

    task = TaskAdapter(AdapterIdentity("bounded-tool-task", "1.0.0", implementation), job,
                       "tool_episode", AdapterIdentity("bounded-tool-projection", "1.0.0", implementation),
                       "tool_messages_v1", project, lambda _service: None)
    handlers = {"lookup_ticket": lambda arguments: {"queue": "billing"} if arguments["ticket_id"] == "T-401" else {"queue": "unknown"}}
    return task, handlers


def prepare_json_training_request(*, seed: int = 17, rows_per_template: int = 32, steps: int = 32) -> dict[str, Any]:
    from neural_foundry.backends.automodel.config import AutoModelSFTRecipe, AUTOMODEL_VERSIONS, PEFT_RELOAD_VERSIONS
    from neural_foundry.json_frontier_model import frontier_proof_configuration
    from neural_foundry.reference_json_extraction import generate_scenarios, reference_task

    if type(steps) is not int or not 1 <= steps <= 512:
        raise GymJobError("prepared training steps must be an integer from 1 to 512")
    task = reference_task(rows_per_template=rows_per_template)
    rows = generate_scenarios(split="train", seed=seed, parameters={"rows_per_template": rows_per_template})
    approved = {row["scenario_id"]: row["reference"] for row in rows}
    projected = [task.project(row["input"], row["reference"]) for row in rows]
    intents = {}
    for method in ("sft", "lora"):
        recipe = {"method": method, "seed": seed, "steps": steps, "max_sequence_tokens": 256,
                  "tokens_per_batch": 256, "max_dataset_tokens": 512 * len(rows),
                  "learning_rate": 0.00001 if method == "sft" else 0.00005}
        if method == "lora":
            recipe.update({"rank": 8, "target_modules": ["q_proj", "v_proj"]})
        frozen_recipe = AutoModelSFTRecipe.from_document(recipe)
        request = {"run_id": f"json-validation-{method}-{seed}", "task": "reference_invoice_extraction@1",
                   "model": "transformers:Qwen3-0.6B@1", "trainer": "nemo_automodel.sft@1",
                   "evaluator": "public@1", "recipe": frozen_recipe.document(), "approved_responses": approved}
        intents[method] = {"lifecycle_request": request, "request_digest": _document_digest(request),
                           "recipe_digest": frozen_recipe.digest}
    document = {"kind": "nf_json_training_preparation_v1", "status": "prepared_not_dispatched",
                "base_model": {"model_id": "Qwen/Qwen3-0.6B", "revision": "c1899de289a04d12100db370d81485cdf75e47ca"},
                "job_digest": task.job.digest, "training_split": "train", "training_cases": len(rows),
                "training_rows_digest": _document_digest({"rows": rows}),
                "projected_demonstrations_digest": _document_digest({"interactions": projected}),
                "demonstration_source": "synthetic_train_references_caller_reported_not_model_evidence",
                "training_rows": rows, "intents": intents,
                "runtime_requirements": {"automodel": dict(AUTOMODEL_VERSIONS), "lora_reload": dict(PEFT_RELOAD_VERSIONS),
                    "device": "cuda:0", "precision": "bfloat16", "context_tokens": 512, "max_new_tokens": 128},
                "frontier_configuration": frontier_proof_configuration(),
                "preflight_required": ["local_checkpoint_file_digests", "loaded_base_state_digest",
                    "pinned_automodel_runtime", "pinned_peft_reload_runtime", "tokenized_dataset_budget",
                    "lifecycle_frozen_training_request", "candidate_reload_and_vllm_identity_probes"],
                "weight_edit_dispatched": False, "holdout_accessed": False, "provider_called": False,
                "download_allowed": False, "ready_to_train": False, "release_authorized": False}
    document["preparation_digest"] = _document_digest(document)
    return document


def prepare_json_robustness_v2_training_request() -> dict[str, Any]:
    from neural_foundry.json_injection_training_data import (
        generate_train_injection_rows, interleave_train_injection_rows,
    )
    from neural_foundry.reference_json_extraction import reference_task

    seed, rows_per_template, injection_seed, injection_examples_per_template = 17, 32, 20261111, 16
    steps = rows_per_template * 10 + injection_examples_per_template * 4
    document = prepare_json_training_request(seed=seed, rows_per_template=rows_per_template, steps=steps)
    base_rows = document["training_rows"]
    injection_rows = generate_train_injection_rows(seed=injection_seed,
        examples_per_template=injection_examples_per_template)
    rows = interleave_train_injection_rows(base_rows, injection_rows)
    task = reference_task(rows_per_template=rows_per_template)
    projected = [task.project(row["input"], row["reference"]) for row in rows]
    for intent in document["intents"].values():
        intent["lifecycle_request"]["approved_responses"] = {row["scenario_id"]: row["reference"] for row in rows}
        intent["request_digest"] = _document_digest(intent["lifecycle_request"])
    document.update({
        "kind": "nf_json_training_preparation_v2",
        "training_cases": len(rows),
        "training_rows_digest": _document_digest({"rows": rows}),
        "projected_demonstrations_digest": _document_digest({"interactions": projected}),
        "training_rows": rows,
        "data_contract": {
            "version": 2, "base_seed": seed, "base_rows_per_template": rows_per_template,
            "injection_seed": injection_seed, "injection_examples_per_template": injection_examples_per_template,
            "base_rows": len(base_rows), "injection_rows": len(injection_rows),
            "interleaving": "base_then_injection_pairs_v1", "ordered_steps": steps,
            "base_rows_digest": _document_digest({"rows": base_rows}),
            "injection_rows_digest": _document_digest({"rows": injection_rows}),
            "injection_rows_consumed_by_ordered_steps": len(injection_rows),
        },
    })
    document["preparation_digest"] = _document_digest({key: value for key, value in document.items()
                                                        if key != "preparation_digest"})
    return document


def execute_json_training_request(document: Mapping[str, Any], *, output: Path, method: str,
                                  automodel_python: str, reload_python: str) -> dict[str, Any]:
    from neural_foundry.backends.automodel.config import AutoModelBackendConfig
    from neural_foundry.gym_service import GymStudioService
    from neural_foundry.reference_json_extraction import reference_task

    if method not in {"sft", "lora"}:
        raise GymJobError("only approved SFT and LoRA training can be dispatched")
    if document.get("preparation_digest") != _document_digest({key: value for key, value in document.items() if key != "preparation_digest"}):
        raise GymJobError("training preparation digest differs")
    recipe = document["intents"][method]["lifecycle_request"]["recipe"]
    rows = document["training_rows"]
    if document.get("kind") == "nf_json_training_preparation_v2":
        expected = prepare_json_robustness_v2_training_request()
        rows_per_template = expected["data_contract"]["base_rows_per_template"]
    else:
        rows_per_template = len(rows) // 10
        expected = prepare_json_training_request(seed=recipe["seed"], rows_per_template=rows_per_template, steps=recipe["steps"])
    if document != expected:
        raise GymJobError("training preparation differs from the frozen train-only contract")
    for interpreter in (automodel_python, reload_python):
        if not Path(interpreter).is_absolute() or not Path(interpreter).is_file():
            raise GymJobError("approved runtime interpreter is missing")
    task = reference_task(rows_per_template=rows_per_template)
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    base = transformers_model(document["base_model"]["model_id"], document["base_model"]["revision"],
        device="cuda:0", precision="bfloat16", context_tokens=512, max_new_tokens=128, seed=recipe["seed"])
    lifecycle = GymLifecycle(GymStudioService(state_dir=output / "state"))
    evaluator = EvaluatorAdapter(AdapterIdentity("public", "1", task.identity.implementation_digest),
        lambda *args: {"split": "validation", "complete": False, "metrics": {}, "status": "deferred_to_verified_vllm"})
    for adapter in (task, base, evaluator):
        lifecycle.register(adapter)
    lifecycle.register_backend("nemo_automodel.sft@1", config=AutoModelBackendConfig(
        automodel_python, timeout_seconds=1800, reload_python_executable=reload_python))
    run = lifecycle.run(document["intents"][method]["lifecycle_request"])
    result = {"kind": "nf_json_train_only_result_v1", "preparation_digest": document["preparation_digest"],
        "method": method, "base": vars(base.reference), "revision": document["base_model"]["revision"],
        "candidate_record": run["candidate_record"], "training_request": run["training_request"],
        "dataset_digest": run["dataset"]["dataset_digest"], "evaluation_status": run["evaluation_status"],
        "weight_edit_dispatched": False, "holdout_accessed": False, "provider_called": False, "release_authorized": False}
    result["result_digest"] = _document_digest(result)
    with (output / "training-result.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--prepare-json-training", action="store_true", help="Freeze train-only SFT/LoRA intents without loading a model or dispatching work")
    parser.add_argument("--robustness-v2", action="store_true", help="Freeze the versioned train-only injection-augmented SFT/LoRA preparation")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--rows-per-template", type=int, default=32)
    parser.add_argument("--training-steps", type=int, default=32)
    parser.add_argument("--execute-json-training", type=Path, help="Execute one approved frozen train-only preparation")
    parser.add_argument("--training-method", choices=("sft", "lora"))
    parser.add_argument("--automodel-python", type=Path)
    parser.add_argument("--reload-python", type=Path)
    parser.add_argument("--tool-only", action="store_true", help="Run only the bounded one-call task without training")
    parser.add_argument("--resume", action="store_true", help="Resume the exact run from its durable state directory")
    parser.add_argument("--interrupt-training-after-step", type=int, help="Test fault injection after a committed optimizer step")
    args = parser.parse_args()
    if args.execute_json_training is not None:
        if (args.allow_download or args.prepare_json_training or args.robustness_v2 or args.resume or args.tool_only
            or args.training_method is None or args.automodel_python is None or args.reload_python is None):
            parser.error("execution requires a frozen preparation, method and existing runtime paths; downloads are forbidden")
        document = json.loads(args.execute_json_training.read_bytes())
        result = execute_json_training_request(document, output=args.output_dir.resolve(), method=args.training_method,
            automodel_python=str(args.automodel_python), reload_python=str(args.reload_python))
        print("NF_JSON_TRAIN_ONLY=" + json.dumps(result, allow_nan=False))
        return
    if args.prepare_json_training:
        if args.allow_download or args.resume or args.tool_only or args.interrupt_training_after_step is not None:
            parser.error("preparation cannot download, resume, train, or inject runtime faults")
        if args.robustness_v2:
            if (args.seed != 17 or args.rows_per_template != 32 or args.training_steps != 32):
                parser.error("robustness-v2 uses its frozen seeds, dataset sizes, and optimizer steps")
            document = prepare_json_robustness_v2_training_request()
        else:
            document = prepare_json_training_request(seed=args.seed, rows_per_template=args.rows_per_template, steps=args.training_steps)
        output = args.output_dir.resolve()
        output.mkdir(parents=True, mode=0o700, exist_ok=False)
        with (output / "training-preparation.json").open("x", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"preparation_digest": document["preparation_digest"], "status": document["status"],
                          "training_cases": document["training_cases"], "ready_to_train": False}))
        return
    if args.robustness_v2:
        parser.error("--robustness-v2 requires --prepare-json-training")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("Choose a fresh output directory")
    output.mkdir(parents=True, mode=0o700, exist_ok=True)
    state_dir = output / "state"
    studio = WeightProgrammingStudio(gym_state_dir=state_dir)
    lifecycle: GymLifecycle = studio.gym_lifecycle
    hook = None
    if args.interrupt_training_after_step is not None:
        if args.interrupt_training_after_step < 1:
            parser.error("interrupt-training-after-step must be positive")

        def hook(step: int) -> None:
            if step == args.interrupt_training_after_step:
                raise RuntimeError(f"intentional interruption after committed training step {step}")

    trainer = peft_sequence_trainer(checkpoint_hook=hook)
    evaluator = EvaluatorAdapter(AdapterIdentity("public-validation-evaluator", "1.0.0", sha256_file(__file__)), public_panel_evaluator)
    tool_task, handlers = _build_tool_task(studio.gym_service.components)
    lifecycle.register(tool_task)
    lifecycle.register(evaluator)
    if not args.tool_only:
        lifecycle.register(_build_support_task(studio.gym_service.components))
        lifecycle.register(trainer)

    model_ids = (
        ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
        ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "31b70e2e869a7173562077fd711b654946d38674"),
    )
    models = {}
    selected_models = model_ids[:1] if args.tool_only else model_ids
    for model_id, revision in selected_models:
        resolve_checkpoint(model_id, revision=revision, allow_download=args.allow_download)
        models[model_id] = transformers_model(
            model_id, revision, device=args.device, precision=args.precision,
            context_tokens=args.context_tokens, max_new_tokens=args.max_new_tokens, seed=17,
        )
        lifecycle.register(models[model_id])

    reports = []
    recipe = {"method": "lora", "seed": 17, "steps": 2, "max_sequence_tokens": 512,
              "tokens_per_batch": 512, "max_dataset_tokens": 8192, "rank": 2, "target_modules": ["q_proj", "v_proj"],
              "learning_rate": 0.00005}
    for model_id, _revision in (() if args.tool_only else model_ids):
        base = model_id.rsplit("/", 1)[-1]
        result = studio.run_gym_lifecycle({
            "run_id": f"support-{base}", "task": "support-routing-task", "model": f"transformers:{base}",
            "trainer": trainer.identity.adapter_id, "evaluator": evaluator.identity.adapter_id,
            "recipe": recipe, "approved_responses": {"support-train-T-100": "billing"},
        })
        reports.append({"task": "support_routing", "base": model_id,
                        "dataset_digest": result["dataset"]["dataset_digest"],
                        "training_request_digest": result["training_request"]["training_request_digest"],
                        "candidate_record_digest": result["candidate_record"]["candidate_record_digest"],
                        "evaluation_receipt_digest": result["evaluation_receipt"]["evaluation_receipt_digest"],
                        "artifact_reused": result["artifact_reused"],
                        "scorecard": result["scorecard"], "approval": result["approval"]})

    tool_base = model_ids[0][0].rsplit("/", 1)[-1]
    models[f"tool-{tool_base}"] = transformers_model(
        model_ids[0][0], model_ids[0][1], device=args.device, precision=args.precision,
        context_tokens=args.context_tokens, max_new_tokens=args.max_new_tokens, seed=17,
        tool_handlers=handlers,
    )
    object.__setattr__(models[f"tool-{tool_base}"], "identity", AdapterIdentity(f"tool-transformers:{tool_base}", "1.0.0", sha256_file(__file__)))
    lifecycle.register(models[f"tool-{tool_base}"])
    tool_result = studio.run_gym_lifecycle({
        "run_id": "bounded-tool-qwen", "task": "bounded-tool-task", "model": f"tool-transformers:{tool_base}",
        "evaluator": evaluator.identity.adapter_id, "train": False,
    })
    reports.append({"task": "bounded_tool", "base": model_ids[0][0],
                    "candidate_record_digest": tool_result["candidate_record"]["candidate_record_digest"],
                    "evaluation_receipt_digest": tool_result["evaluation_receipt"]["evaluation_receipt_digest"],
                    "artifact_reused": tool_result["artifact_reused"],
                    "scorecard": tool_result["scorecard"], "approval": tool_result["approval"]})
    report = {"kind": "nf_gym_real_lifecycle_reuse_v1", "runs": reports,
              "runtime": {"device": args.device, "precision": args.precision, "torch": str(__import__("torch").__version__)},
              "authority": {"release_authorized": False, "independent_holdout_custody": False,
                            "training_source_responses_are_reported": True, "task_quality_qualified": False}}
    report["report_digest"] = _document_digest(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report_digest": report["report_digest"], "runs": len(reports),
                      "scorecard_decisions": [row["scorecard"]["decision"] for row in reports]}, indent=2))


if __name__ == "__main__":
    main()
