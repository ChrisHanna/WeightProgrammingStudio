"""Train the synthetic reference task in a separate GPU process."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .backends.automodel.config import AutoModelBackendConfig
from .backends.weight_edit.adapter import WeightEditBackendConfig
from .gym_adapters import transformers_model
from .gym_artifacts import CandidateRef
from .gym_lifecycle import AdapterIdentity, EvaluatorAdapter, GymLifecycle, public_panel_evaluator
from .gym_service import GymStudioService
from .reference_json_extraction import generate_scenarios, reference_task, score_response
from .gym_service import ExecutionResponse


MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def train_reference(*, state_dir: Path, automodel_python: str, seed: int, rows_per_template: int,
                    compatibility_probe_path: str, sft_steps: int = 32, weight_edit_target_step: float = 0.01,
                    preservation_policy_version: int = 2, retention_mode: str = "generalist") -> dict[str, Any]:
    task = reference_task(rows_per_template=rows_per_template)
    service = GymStudioService(state_dir=state_dir)
    base = transformers_model(MODEL_ID, MODEL_REVISION, device="cuda:0", precision="bfloat16",
                              context_tokens=512, max_new_tokens=80, seed=seed)
    evaluator = EvaluatorAdapter(AdapterIdentity("public", "1", task.identity.implementation_digest), public_panel_evaluator)
    lifecycle = GymLifecycle(service)
    for adapter in (task, base, evaluator):
        lifecycle.register(adapter)
    lifecycle.register_backend("nemo_automodel.sft@1", config=AutoModelBackendConfig(automodel_python, 300))
    rows = generate_scenarios(split="train", seed=seed, parameters={"rows_per_template": rows_per_template})
    training_inputs = {json.dumps(row["input"], sort_keys=True) for row in rows}
    preservation_rows = [row for row in generate_scenarios(split="train", seed=seed + 1, parameters={"rows_per_template": rows_per_template})
                         if json.dumps(row["input"], sort_keys=True) not in training_inputs]
    preservation = []
    for row in preservation_rows:
        result = base.infer(task.project(row["input"], None))
        response = result.response if isinstance(result, ExecutionResponse) else result
        if score_response(reference=row["reference"], response=response)["exact_match"] == 1.0:
            preservation.append(task.project(row["input"], None))
    if not preservation:
        raise ValueError("no separate public preservation cases passed the base scorer")
    request = {
        "run_id": "json-reference-proof", "task": "reference_invoice_extraction@1",
        "model": base.identity.adapter_id + "@1", "trainer": "nemo_automodel.sft@1", "evaluator": "public@1",
        "recipe": {"method": "sft", "seed": seed, "steps": sft_steps, "max_sequence_tokens": 256,
                   "tokens_per_batch": 256, "max_dataset_tokens": 512 * rows_per_template, "learning_rate": 0.00001},
        "approved_responses": {row["scenario_id"]: row["reference"] for row in rows},
    }
    run = lifecycle.run(request)
    record = run["candidate_record"]
    candidate = record["candidate"]
    trained = CandidateRef(MODEL_ID, "full", candidate["base_model_digest"], candidate["candidate_model_digest"],
                           candidate["artifact_digest"], str(Path(record["artifact_root"]) / "artifact"),
                           candidate["candidate_id"], candidate["method_id"], base.family)
    lifecycle.register_backend("weight_edit.generic@1", config=WeightEditBackendConfig(sys.executable, compatibility_probe_path), job=task.job)
    edited_run = lifecycle.run({"run_id": "json-reference-weight-edit", "task": "reference_invoice_extraction@1",
        "model": base.identity.adapter_id + "@1", "trainer": "weight_edit.generic@1", "evaluator": "public@1",
        "dataset_digest": run["dataset"]["dataset_digest"],
        "recipe": {"method": "weight_edit", "edit_method": "legacy_key_value_v1", "seed": seed, "target_step": weight_edit_target_step,
                   "preservation_policy_version": preservation_policy_version,
                   "retention_mode": retention_mode,
                   "retention_nullspace_weight": 20.0, "max_sequence_tokens": 512, "max_training_tokens": 16384,
                   "max_keys": 4096, "preservation_source": "public_train", "max_preservation_logit_drift": 0.25,
                   "preservation_interactions": preservation}})
    edited_record = edited_run["candidate_record"]
    edited_candidate = edited_record["candidate"]
    edited = CandidateRef(MODEL_ID, "weight_edit", edited_candidate["base_model_digest"], edited_candidate["candidate_model_digest"],
                          edited_candidate["artifact_digest"], str(Path(edited_record["artifact_root"]) / "artifact"),
                          edited_candidate["candidate_id"], "weight_edit", base.family)
    return {"job_digest": task.job.digest, "revision": MODEL_REVISION, "base": vars(base.reference),
            "sft": vars(trained), "training_request_digest": run["training_request"]["training_request_digest"],
            "candidate_record_digest": record["candidate_record_digest"], "weight_edit": vars(edited),
            "weight_edit_training_request_digest": edited_run["training_request"]["training_request_digest"],
            "weight_edit_candidate_record_digest": edited_record["candidate_record_digest"],
            "dataset_digest": run["dataset"]["dataset_digest"], "preservation_count": len(preservation),
            "retention_mode": retention_mode}


def main() -> None:
    if len(sys.argv) not in {9, 10}:
        raise ValueError("expected state directory, AutoModel Python, seed, rows, compatibility probe, SFT steps, edit step, and preservation version")
    result = train_reference(state_dir=Path(sys.argv[1]), automodel_python=sys.argv[2],
                             seed=int(sys.argv[3]), rows_per_template=int(sys.argv[4]), compatibility_probe_path=sys.argv[5],
                             sft_steps=int(sys.argv[6]), weight_edit_target_step=float(sys.argv[7]), preservation_policy_version=int(sys.argv[8]),
                             retention_mode=sys.argv[9] if len(sys.argv) == 10 else "generalist")
    print("NF_JSON_PROOF_TRAINING=" + json.dumps(result, allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    main()