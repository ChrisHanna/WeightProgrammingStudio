"""Explicit routing-task example using the shared Studio lifecycle on two real bases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_foundry.gym_adapters import peft_sequence_trainer, transformers_model
from neural_foundry.gym_job import AcceptanceGate, CandidateMethod, ComponentReference, GymJobError, GymJobSpec, ScenarioSource
from neural_foundry.gym_lifecycle import AdapterIdentity, EvaluatorAdapter, TaskAdapter, public_panel_evaluator
from neural_foundry.gym_service import _document_digest
from neural_foundry.provenance import sha256_file
from neural_foundry.studio import WeightProgrammingStudio


def routing_adapter() -> TaskAdapter:
    digest = sha256_file(__file__)
    panels = {
        "train": [("charge was duplicated on my training account", "billing"), ("the training login is broken", "support")],
        "validation": [("I need a refund for an invoice", "billing"), ("I cannot sign in to the portal", "support")],
        "holdout": [("my payment card expired", "billing")],
    }
    source = ScenarioSource("support-routing", "routing-examples", "1", digest, _document_digest({"panels": panels}), "simulated", tuple(panels), {})
    job = GymJobSpec(
        job_id="support-routing-reuse", title="Route structured support tickets",
        description="Read the full structured ticket input. Return only the JSON string billing or support. Use one of allowed_labels.",
        task_input_schema={"type": "object", "properties": {"prompt": {"type": "string"}, "ticket": {"type": "string"},
                           "allowed_labels": {"type": "array", "items": {"type": "string"}}},
                           "required": ["prompt", "ticket", "allowed_labels"], "additionalProperties": False},
        task_output_schema={"type": "string", "enum": ["billing", "support"]}, scenario_sources=(source,),
        scorer=ComponentReference("routing-score", "1", digest, {}), verifier=ComponentReference("routing-verifier", "1", digest, {}),
        candidate_methods=(CandidateMethod("unchanged_base", "1", digest, {}), CandidateMethod("lora", "1", digest, {})),
        gates=(AcceptanceGate("quality", "task", "accuracy", "validation", "ge", 0.9),
               AcceptanceGate("behavior", "behavior", "accuracy", "holdout", "ge", 0.9),
               AcceptanceGate("controls", "controls", "beats_control", "controls", "eq", 1),
               AcceptanceGate("reproducibility", "reproducibility", "repeat", "reproducibility", "eq", 1)),
        metadata={"research_example": True},
    )

    def bind(service):
        def provider(*, split, **kwargs):
            return [{"scenario_id": f"routing-{split}-{index}", "item_id": f"routing-item-{split}-{index}",
                     "family_id": f"routing-family-{split}-{index}", "interaction_type": "single_response",
                     "input": {"prompt": "Classify this ticket", "ticket": ticket, "allowed_labels": ["billing", "support"]},
                     "reference": label} for index, (ticket, label) in enumerate(panels[split])]

        entries = [("scenario_provider", source.provider_id, provider),
                   ("verifier", job.verifier.component_id, lambda *, response, reference, **kwargs: {"passed": response == reference}),
                   ("scorer", job.scorer.component_id, lambda *, verification, **kwargs: {"accuracy": float(verification["passed"])})]
        entries += [("candidate_method", method.method_id, lambda **kwargs: None) for method in job.candidate_methods]
        for kind, name, function in entries:
            try:
                service.components.resolve(kind, name, "1", digest)
            except GymJobError:
                service.components.register(kind, name, "1", digest, function)

    def project(input_value, response):
        messages = [{"role": "system", "content": job.description}, {"role": "user", "content": json.dumps(input_value, sort_keys=True)}]
        if response is not None:
            messages.append({"role": "assistant", "content": json.dumps(response)})
        return {"format": "chat_messages_v1", "messages": messages}

    return TaskAdapter(AdapterIdentity("routing-task", "1", digest), job, "single_response", AdapterIdentity("routing-full-json-projection", "1", digest), "chat_messages_v1", project, bind)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-config", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.models_config.read_bytes())
    studio = WeightProgrammingStudio(gym_state_dir=args.state_dir)
    task, trainer = routing_adapter(), peft_sequence_trainer()
    evaluator = EvaluatorAdapter(AdapterIdentity("public-validation", "1", sha256_file(__file__)), public_panel_evaluator)
    for adapter in (task, trainer, evaluator):
        studio.register_gym_lifecycle_adapter(adapter)
    results = []
    for entry in config["models"]:
        settings = config["runtime"]
        model = transformers_model(entry["model_id"], entry["revision"], device=settings["device"], precision=settings["precision"],
                                   context_tokens=settings["context_tokens"], max_new_tokens=32, seed=settings["seed"])
        studio.register_gym_lifecycle_adapter(model)
        result = studio.run_gym_lifecycle({
            "run_id": f"routing-{model.family}", "task": task.identity.adapter_id, "model": model.identity.adapter_id,
            "trainer": trainer.identity.adapter_id, "evaluator": evaluator.identity.adapter_id,
            "recipe": {"method": "lora", "seed": 17, "steps": 1, "learning_rate": 0.001, "rank": 2,
                       "target_modules": ["q_proj", "v_proj"], "max_sequence_tokens": 512, "tokens_per_batch": 1024},
            "approved_responses": {"routing-train-0": "billing", "routing-train-1": "support"},
        })
        results.append({"family": model.family, "lifecycle_digest": result["lifecycle_digest"],
                        "dataset_digest": result["dataset"]["dataset_digest"], "training_request_digest": result["training_request"]["training_request_digest"],
                        "candidate_record_digest": result["candidate_record"]["candidate_record_digest"],
                        "evaluation_receipt_digest": result["evaluation_receipt"]["evaluation_receipt_digest"],
                        "scorecard": result["scorecard"], "approval": result["approval"]})
    report = {"kind": "nf_studio_reuse_example_v1", "results": results, "teaching_source": "explicit_reported_demonstrations", "release_authorized": False}
    (args.state_dir / "reuse-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"families": [row["family"] for row in results], "lineage_complete": len(results) == 2, "release_authorized": False}, indent=2))


if __name__ == "__main__":
    main()