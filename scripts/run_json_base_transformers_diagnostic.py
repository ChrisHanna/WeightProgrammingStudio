"""Score the unchanged base with Transformers on the public JSON proof panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from neural_foundry.gym_adapters import transformers_model
from neural_foundry.gym_service import _document_digest
from neural_foundry.json_proof_loop import public_required_rows, score_public_rows
from neural_foundry.reference_json_extraction import generate_scenarios, reference_task


MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def score_base(output: Path, *, seed: int = 17) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {output}")
    task = reference_task(rows_per_template=32)
    rows = generate_scenarios(split="validation", seed=seed, parameters={"rows_per_template": 32})
    required = public_required_rows(seed=seed)
    model = transformers_model(MODEL_ID, MODEL_REVISION, device="cuda:0", precision="bfloat16",
                              context_tokens=512, max_new_tokens=128, seed=0)
    task_score = score_public_rows(task, model, rows)
    required_score = score_public_rows(task, model, required)
    card = {"kind": "nf_json_base_transformers_diagnostic_v1", "role": "base",
            "split": "validation", "seed": seed, "model_identity": dict(model.loaded_identity()),
            "task": task_score, "required_set": required_score,
            "holdout_accessed": False, "release_authorized": False}
    card["document_digest"] = _document_digest(card)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(card, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "task_cases": task_score["cases"],
        "task_exact_match": task_score["exact_match"], "required_cases": required_score["cases"],
        "required_exact_match": required_score["exact_match"],
        "format_valid": min(task_score["format_valid"], required_score["format_valid"]),
        "holdout_accessed": False}, sort_keys=True), flush=True)
    return card


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    score_base(args.output.resolve(), seed=args.seed)


if __name__ == "__main__":
    main()
