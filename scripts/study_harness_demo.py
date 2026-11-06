"""HARNESS DEMO, NOT A SCIENTIFIC RESULT. Runs every study arm through the protocol on Pythia-14M.

The plan is labelled harness_demo in code and has no option to change that, so the decision can't
promote, whatever the numbers say. The candidate is untuned, the tuning grids and training budgets
are tiny, and the items are the ones with the known shortcut. The output shows that the pipeline
runs and how long each step takes; it isn't evidence about any edit method or control.

    python scripts/study_harness_demo.py --checkpoint /tmp/pythia-14m --work-dir DIR --commit SHA --output FILE
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from neural_foundry import baselines as bl
from neural_foundry.prd_adapters import run_easyedit_adapter
from neural_foundry.pythia import PythiaModel
from neural_foundry.study import (
    CustodianClient,
    StudyPlan,
    Thresholds,
    declare_exposure,
    freeze_plan,
    write_base_arm,
    write_candidate,
)
from neural_foundry.study import arms as study_arms
from neural_foundry.study.evaluators import encode_candidate_artifact
from neural_foundry.study.plan import CONTROL_ARMS, digest_bytes

LABEL = "HARNESS DEMO, NOT A SCIENTIFIC RESULT"
RULES_PER_TASK = 4
RETENTION_PROMPTS = 16
LAYER = 5
STRENGTH = 80.0
TUNING_SEED = 909
GUARD_BATCH = 4
LORA_GRID = (
    bl.BaselineConfig(
        kind="lora", targets=targets, rank=4, lr=lr, retention_weight=3.0, steps=50, guard_batch=GUARD_BATCH, checkpoints=(25, 50)
    )
    for targets, lr in (("final_down_proj", 0.003), ("all_down_proj", 0.001))
)
FINE_TUNE_GRID = (
    bl.BaselineConfig(kind="sft", lr=lr, retention_weight=3.0, steps=25, guard_batch=GUARD_BATCH, checkpoints=(10, 25))
    for lr in (1e-5, 3e-5)
)


def make_plan(checkpoint: Path) -> StudyPlan:
    return StudyPlan(
        study_id="harness-demo-controls-pythia-14m",
        label="harness_demo",
        hypothesis=f"{LABEL}: every arm exercises the protocol",
        tasks=("parity", "boolean", "comparison"),
        clusters_per_task_per_draw=20,
        holdout_draws=2,
        retention_items=40,
        development_prefixes=("Task:", "Rule:"),
        holdout_prefixes=("Today:", "Check:", "Quiz:"),
        number_domain=(0, 9999),
        synthesis_seeds=(101, 202, 303),
        analysis_seed=20261106,
        primary_metric="forced_choice_accuracy",
        secondary_metrics=("top1_accuracy",),
        alpha=0.05,
        correction="holm",
        bootstrap_resamples=2000,
        permutation_resamples=5000,
        thresholds=Thresholds(
            min_effect=0.05,
            min_accuracy=0.6,
            min_seed_pass_fraction=1.0,
            min_retention_agreement=0.9,
            max_retention_kl=0.1,
        ),
        evaluator="pythia_mlp_override",
        checkpoint_dir=str(checkpoint),
        checkpoint_digest=digest_bytes((checkpoint / "model.safetensors").read_bytes()),
        control_arms=CONTROL_ARMS,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--commit", required=True, help="the commit this tree was copied from; git can't read it from WSL")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.work_dir.mkdir(parents=True)
    started = time.time()
    timings: dict[str, Any] = {}

    def lap(name: str, since: float) -> None:
        timings[name] = round(time.time() - since, 1)
        print(f"{name}: {timings[name]}s", flush=True)

    # 1. Freeze the plan before anything is synthesized or tuned.
    plan = make_plan(args.checkpoint)
    plan_path = args.work_dir / "plan.json"
    plan_digest = freeze_plan(plan, plan_path)
    model = PythiaModel.from_pretrained_dir(args.checkpoint)
    site = model.mlp_down_proj_site(LAYER)

    def development(seed: int) -> study_arms.DevelopmentData:
        return study_arms.development_data(plan, seed, rules_per_task=RULES_PER_TASK, retention_prompts=RETENTION_PROMPTS)

    # 2. Tune the controls on development data. No holdout seed exists yet.
    tuning_data = development(TUNING_SEED)
    tuned: dict[str, bl.SelectedConfig] = {}
    tuning: dict[str, Any] = {}
    for arm, grid in (("lora", tuple(LORA_GRID)), ("fine_tune", tuple(FINE_TUNE_GRID))):
        since = time.time()
        tuned[arm], tuning[arm] = study_arms.tune_arm(model, plan, grid, tuning_data)
        lap(f"tune_{arm}", since)

    # 3. Build every arm, for every seed, from that seed's development data.
    def entry(name: str, seed: int, artifact: bytes) -> dict[str, Any]:
        path = args.work_dir / f"{name}_{seed}.safetensors"
        return {"synthesis_seed": seed, "candidate_digest": write_candidate(path, artifact), "artifact_path": str(path)}

    candidates: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = [write_base_arm(args.work_dir / "base.json", plan_digest)]
    datasets = [tuning_data]
    for seed in plan.synthesis_seeds:
        data = development(seed)
        datasets.append(data)
        since = time.time()
        edit = run_easyedit_adapter(
            model,
            algorithm="AlphaEdit",
            associations=[(p, t) for p, t in data.splits.dev],
            retention_prompts=data.splits.refit_guard,
            target_strength=STRENGTH,
            target_layers=(LAYER,),
        )
        weights = {site: edit.edited_weight.to(torch.float64)}
        candidates.append(entry("candidate", seed, encode_candidate_artifact(weights)))
        lap(f"candidate_{seed}", since)
        placebo = study_arms.placebo_weights(model.tensors, weights, seed)
        controls.append({"arm": "placebo", **entry("placebo", seed, encode_candidate_artifact(placebo))})
        for arm, selected in tuned.items():
            since = time.time()
            fit = study_arms.fit_arm(model, selected, seed, data)
            controls.append({"arm": arm, **entry(arm, seed, encode_candidate_artifact(fit.merged_weights))})
            lap(f"fit_{arm}_{seed}", since)

    clusters = [cluster for data in datasets for cluster in data.clusters]
    retention = [item for data in datasets for item in data.retention_items]
    manifest = {
        "plan_digest": plan_digest,
        "candidates": candidates,
        "controls": controls,
        "exposed_operands": declare_exposure(clusters, retention, plan.tasks),
        "tuning": tuning,
    }

    # 4. Freeze once. The custodian draws the holdout, then scores every arm once.
    replies = []
    with CustodianClient(plan_path, plan_digest, args.work_dir / "custodian") as client:
        since = time.time()
        frozen = client.freeze(manifest)
        lap("freeze", since)
        for arm_entry in (*controls, *candidates):
            since = time.time()
            reply = client.query(arm_entry["candidate_digest"])
            replies.append({"synthesis_seed": arm_entry["synthesis_seed"], **reply})
            lap(f"query_{reply['arm']}_{arm_entry['synthesis_seed']}", since)
        since = time.time()
        outcome = client.decide()
        lap("decide", since)
        try:
            client.query(candidates[0]["candidate_digest"])
            repeat = "ANSWERED"
        except RuntimeError as refused:
            repeat = str(refused)

    record = {
        "label": LABEL,
        "not_evidence_for": "any edit method, control, Pythia-14M capability, or promotion of anything",
        "item_warning": "These items have a known shortcut and are being replaced.",
        "commit": args.commit,
        "wall_seconds": round(time.time() - started, 1),
        "timings_seconds": timings,
        "plan": plan.to_dict(),
        "plan_digest": plan_digest,
        "synthesis": {"algorithm": "AlphaEdit", "layer": LAYER, "strength": STRENGTH, "rules_per_task": RULES_PER_TASK},
        "tuning": tuning,
        "freeze": frozen,
        "per_arm_aggregates": replies,
        "repeat_query_response": repeat,
        "decision": outcome,
    }
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(LABEL)
    print("promoted:", outcome["promoted"], "| repeat query:", repeat)
    for reason in outcome["rejection_reasons"]:
        print(f"  {reason['code']} [{reason['scope']}] observed={reason['observed']} threshold={reason['threshold']}")
    for effect in outcome["effects"]:
        print(
            f"  base {effect['task']}: base={effect['base_accuracy']:.3f} candidate={effect['candidate_accuracy']:.3f} "
            f"delta={effect['mean_delta']:+.3f} CI=[{effect['ci_low']:+.3f},{effect['ci_high']:+.3f}] holm_p={effect['p_adjusted']:.4f}"
        )
    for control in outcome["controls"]:
        print(
            f"  {control['arm']} {control['task']}: control={control['control_accuracy']:.3f} "
            f"candidate={control['candidate_accuracy']:.3f} delta={control['mean_delta']:+.3f} "
            f"CI=[{control['ci_low']:+.3f},{control['ci_high']:+.3f}] holm_p={control['p_adjusted']:.4f}"
        )
    print("retention:", outcome["retention"])
    print("wall seconds:", record["wall_seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
