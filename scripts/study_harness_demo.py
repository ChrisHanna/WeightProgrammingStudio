"""HARNESS DEMO, NOT A SCIENTIFIC RESULT. Runs the study protocol end to end on Pythia-14M.

The scripted candidate has no tuning, no control arm and no comparison method. The plan
is labelled harness_demo, so the decision can't promote, whatever the numbers say.
The output shows that the pipeline runs; it isn't evidence about any edit method.

    python scripts/study_harness_demo.py --checkpoint /tmp/pythia-14m --work-dir DIR --output FILE
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch

from neural_foundry.prd_adapters import run_easyedit_adapter
from neural_foundry.pythia import PythiaModel
from neural_foundry.study import CustodianClient, StudyPlan, Thresholds, declare_exposure, freeze_plan, write_candidate
from neural_foundry.study.evaluators import encode_candidate_artifact
from neural_foundry.study.generators import draw_clusters, draw_retention, render
from neural_foundry.study.plan import digest_bytes

LABEL = "HARNESS DEMO, NOT A SCIENTIFIC RESULT"
RULES_PER_TASK = 4
LAYER = 5
STRENGTH = 80.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.work_dir.mkdir(parents=True)
    started = time.time()

    # 1. Freeze the plan before anything is synthesized.
    plan = StudyPlan(
        study_id="harness-demo-pythia-14m",
        label="harness_demo",
        hypothesis=f"{LABEL}: a scripted AlphaEdit candidate exercises the protocol",
        tasks=("parity", "boolean", "comparison"),
        clusters_per_task_per_draw=100,
        holdout_draws=2,
        retention_items=200,
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
        checkpoint_dir=str(args.checkpoint),
        checkpoint_digest=digest_bytes((args.checkpoint / "model.safetensors").read_bytes()),
    )
    plan_path = args.work_dir / "plan.json"
    plan_digest = freeze_plan(plan, plan_path)

    # 2. Synthesize one candidate per seed from development items only.
    model = PythiaModel.from_pretrained_dir(args.checkpoint)
    site = model.mlp_down_proj_site(LAYER)
    candidates = []
    all_clusters = []
    protected = draw_retention(1, 5)
    for seed in plan.synthesis_seeds:
        clusters = [c for task in plan.tasks for c in draw_clusters(task, seed, RULES_PER_TASK, domain=plan.number_domain)]
        all_clusters += clusters
        associations = [(item.prompt, item.target) for item in render(clusters, plan.development_prefixes[:1])]
        edit = run_easyedit_adapter(
            model,
            algorithm="AlphaEdit",
            associations=associations,
            retention_prompts=tuple(item.prompt for item in protected),
            target_strength=STRENGTH,
            target_layers=(LAYER,),
        )
        path = args.work_dir / f"candidate_{seed}.safetensors"
        digest = write_candidate(path, encode_candidate_artifact({site: edit.edited_weight}))
        candidates.append({"synthesis_seed": seed, "candidate_digest": digest, "artifact_path": str(path)})

    # 3. Freeze the manifest, then let the custodian draw the holdout and score each candidate once.
    manifest = {
        "plan_digest": plan_digest,
        "candidates": candidates,
        "exposed_operands": declare_exposure(all_clusters, protected, plan.tasks),
    }
    with CustodianClient(plan_path, plan_digest, args.work_dir / "custodian") as client:
        frozen = client.freeze(manifest)
        replies = [client.query(candidate["candidate_digest"]) for candidate in candidates]
        outcome = client.decide()
        try:
            client.query(candidates[0]["candidate_digest"])
            repeat = "ANSWERED"
        except RuntimeError as refused:
            repeat = str(refused)

    # The worktree's commit is recorded, not its cleanliness: this is a demo, not qualification.
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    record = {
        "label": LABEL,
        "not_evidence_for": "any edit method, Pythia-14M capability, or promotion of anything",
        "commit": commit,
        "wall_seconds": round(time.time() - started, 1),
        "plan": plan.to_dict(),
        "plan_digest": plan_digest,
        "synthesis": {"algorithm": "AlphaEdit", "layer": LAYER, "strength": STRENGTH, "rules_per_task": RULES_PER_TASK},
        "freeze": frozen,
        "candidates": [
            {"synthesis_seed": c["synthesis_seed"], "candidate_digest": c["candidate_digest"]} for c in candidates
        ],
        "per_candidate_aggregates": replies,
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
            f"  {effect['task']}: base={effect['base_accuracy']:.3f} candidate={effect['candidate_accuracy']:.3f} "
            f"delta={effect['mean_delta']:+.3f} CI=[{effect['ci_low']:+.3f},{effect['ci_high']:+.3f}] "
            f"holm_p={effect['p_adjusted']:.4f} items={effect['items']}"
        )
    print("retention:", outcome["retention"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
