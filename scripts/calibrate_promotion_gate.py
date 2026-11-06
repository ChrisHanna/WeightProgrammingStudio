"""Run reference edits through the unchanged promotion gate, on public data only.

This shows how the gate behaves. It doesn't set thresholds, and none may be changed from its results.
Each arm runs one after another on one thread, writing JSON that docs/promotion-gate-calibration.md
quotes. Nothing here reads a holdout: the Gym's study does read its own after a selection, so this script
only uses arms from a run that selected nothing, and refuses otherwise.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from neural_foundry import baselines as bl
from neural_foundry import independent_eval as ie
from neural_foundry.edit_plan import apply_edit_graph
from neural_foundry.gpt_neox_adapter import compile_to_edit_graph
from neural_foundry.promotion_gate import decide_promotion, not_evaluated
from neural_foundry.pythia import ALGORITHMIC_THREE_SKILL_GYM as GYM
from neural_foundry.pythia import PythiaModel, run_pythia_gym_task_learning_study, run_pythia_weight_programming_study

MODEL_ID = "EleutherAI/pythia-14m"
NEAR_NOOP_SCALE = 1.0 + 1e-9
RANDOM_SEED = 1337
SFT_LOW_STEPS = bl.BaselineConfig(kind="sft", lr=1e-4, steps=25, retention_weight=1.0, seed=0)
CASES = [(f"{p} {e}", t.replace("Ġ", " ")) for p in GYM.gym_dev_prefixes for e, t in GYM.base_rules]


def _gate(model: PythiaModel, name: str, edited: dict[str, torch.Tensor], cases: list[tuple[str, str]] = CASES) -> dict[str, Any]:
    started = time.time()
    try:
        candidate = ie.freeze_candidate(model.tensors, edited, candidate_id=name, target_cases=cases)
        decision = decide_promotion(ie.evaluate_candidate(model, candidate))
    except ie.IndependentEvalError as exc:
        decision = not_evaluated(f"The evaluator refused it: {exc}")
    row = {
        "arm": name,
        "status": decision["status"],
        "promote_eligible": decision["promote_eligible"],
        "failed_checks": decision["failed_checks"],
        "reason": decision["reason"],
        "checks": {c["check"]: c["observed"] for c in decision["checks"]},
        "seconds": round(time.time() - started, 1),
    }
    print(name, row["status"], len(row["failed_checks"]), row["seconds"], flush=True)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = PythiaModel.from_pretrained_dir(args.checkpoint, model_name=MODEL_ID)

    study = run_pythia_gym_task_learning_study(model, gym=GYM)
    if study.selected_branch_id is not None:
        raise SystemExit("The Gym selected a branch, so its study read the holdout. Calibration uses public data only.")
    branch = next(c for c in study.gym_proposals if c.branch_id.startswith("branch_3"))
    branch_tensors = apply_edit_graph(model, compile_to_edit_graph(model, branch.proposal)).edited_tensors
    site = next(iter(branch_tensors))
    weight = model.tensors[site].to(torch.float64)
    delta = branch_tensors[site] - weight

    noise = torch.randn(delta.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(RANDOM_SEED))
    direct = run_pythia_weight_programming_study(model)
    # Its own targets are what the direct edit is scored on, so a target gain means something for it.
    direct_cases = [
        ("The capital of France is", " Paris"), ("The capital of Japan is", " Tokyo"), ("The capital of Germany is", " Berlin"), ("The chemical symbol for gold is", " Au"),
    ]
    rows = [
        _gate(model, "no-op (exact)", {site: weight.clone()}),
        _gate(model, "no-op (near, scale 1 + 1e-9)", {site: weight * NEAR_NOOP_SCALE}),
        _gate(model, "random, branch 3 norm", {site: weight + noise * (torch.linalg.norm(delta) / torch.linalg.norm(noise))}),
        _gate(model, "direct low-rank edit", {direct.target_site: direct.edited_weight}, direct_cases),
        _gate(model, "Gym branch 3", dict(branch_tensors)),
        _gate(model, "LoRA, the study's 25-step baseline", {site: study.lora_weight}),
    ]
    splits = bl.build_splits(GYM, GYM.retention_prompts, model=model)
    fit = bl.fit_baseline(model, SFT_LOW_STEPS, dev=splits.dev, retention_guard=GYM.retention_prompts)
    rows.append(_gate(model, "fine-tuning, 25 untuned steps", fit.merged_weights))

    document = {
        "kind": "nf_promotion_gate_calibration_v1",
        "commit": args.commit,
        "model": MODEL_ID,
        "threads": 1,
        "fine_tuning_recipe": {"kind": "sft", "lr": SFT_LOW_STEPS.lr, "steps": SFT_LOW_STEPS.steps, "retention_weight": SFT_LOW_STEPS.retention_weight, "note": "low steps and untuned, not a converged baseline"},
        "arms": rows,
    }
    args.out.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
