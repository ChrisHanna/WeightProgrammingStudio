"""Train the synthetic reference task in a separate GPU process."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .backends.automodel.config import AutoModelBackendConfig
from .gym_adapters import transformers_model
from .gym_artifacts import CandidateRef
from .gym_lifecycle import AdapterIdentity, EvaluatorAdapter, GymLifecycle, public_panel_evaluator
from .gym_service import GymStudioService
from .reference_json_extraction import generate_scenarios, reference_task


MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def train_reference(*, state_dir: Path, automodel_python: str, seed: int, rows_per_template: int) -> dict[str, Any]:
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
    request = {
        "run_id": "json-reference-proof", "task": "reference_invoice_extraction@1",
        "model": base.identity.adapter_id + "@1", "trainer": "nemo_automodel.sft@1", "evaluator": "public@1",
        "recipe": {"method": "sft", "seed": seed, "steps": 1, "max_sequence_tokens": 256,
                   "tokens_per_batch": 256, "max_dataset_tokens": 512 * rows_per_template, "learning_rate": 0.00001},
        "approved_responses": {row["scenario_id"]: row["reference"] for row in rows},
    }
    run = lifecycle.run(request)
    record = run["candidate_record"]
    candidate = record["candidate"]
    trained = CandidateRef(MODEL_ID, "full", candidate["base_model_digest"], candidate["candidate_model_digest"],
                           candidate["artifact_digest"], str(Path(record["artifact_root"]) / "artifact"),
                           candidate["candidate_id"], candidate["method_id"], base.family)
    return {"job_digest": task.job.digest, "revision": MODEL_REVISION, "base": vars(base.reference),
            "sft": vars(trained), "training_request_digest": run["training_request"]["training_request_digest"],
            "candidate_record_digest": record["candidate_record_digest"]}


def main() -> None:
    if len(sys.argv) != 5:
        raise ValueError("expected state directory, AutoModel Python, seed, and rows per template")
    result = train_reference(state_dir=Path(sys.argv[1]), automodel_python=sys.argv[2],
                             seed=int(sys.argv[3]), rows_per_template=int(sys.argv[4]))
    print("NF_JSON_PROOF_TRAINING=" + json.dumps(result, allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    main()