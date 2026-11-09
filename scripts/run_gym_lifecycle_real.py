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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--tool-only", action="store_true", help="Run only the bounded one-call task without training")
    parser.add_argument("--resume", action="store_true", help="Resume the exact run from its durable state directory")
    parser.add_argument("--interrupt-training-after-step", type=int, help="Test fault injection after a committed optimizer step")
    args = parser.parse_args()
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
