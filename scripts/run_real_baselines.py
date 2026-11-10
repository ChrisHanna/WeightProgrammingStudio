"""Runs gradient baselines on a real checkpoint in three separate stages, so tuning cannot see sealed data.

    tune    Trains each config and scores public splits. Results are appended, so a rerun resumes.
    refit   Selects one config per method from public scores, retrains it, and saves merged weights.
    final   Scores those saved weights on the holdout. Refuses to run twice.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from neural_foundry import baselines as bl
from neural_foundry.prd_adapters import PRACTICE_RETENTION_PROMPTS, SEALED_RETENTION_PROMPTS
from neural_foundry.pythia import ALGORITHMIC_THREE_SKILL_GYM as GYM, PythiaModel

GUARD_BATCH = 8
LORA_CHECKPOINTS = (25, 50, 100, 150, 200, 300)
SFT_CHECKPOINTS = (10, 25, 50, 100, 150, 200)
STABILITY_SEEDS = (1, 2)


def grid() -> list[bl.BaselineConfig]:
    lora = [
        bl.BaselineConfig(
            kind="lora", targets=t, rank=r, lr=lr, retention_weight=w, steps=LORA_CHECKPOINTS[-1],
            guard_batch=GUARD_BATCH, checkpoints=LORA_CHECKPOINTS,
        )
        for t, r, lr, w in itertools.product(
            ("final_down_proj", "all_down_proj", "all_linear"), (4, 16), (0.003, 0.01), (0.0, 1.0, 10.0)
        )
    ]
    sft = [
        bl.BaselineConfig(
            kind="sft", lr=lr, retention_weight=w, steps=SFT_CHECKPOINTS[-1],
            guard_batch=GUARD_BATCH, checkpoints=SFT_CHECKPOINTS,
        )
        for lr, w in itertools.product((3e-5, 1e-4, 3e-4, 1e-3), (0.0, 1.0, 10.0))
    ]
    # Second round: the first round's best public scores sat at the lowest learning rates tried.
    lora_low = [
        bl.BaselineConfig(
            kind="lora", targets=t, rank=4, lr=lr, retention_weight=w, steps=LORA_CHECKPOINTS[-1],
            guard_batch=GUARD_BATCH, checkpoints=LORA_CHECKPOINTS,
        )
        for t, lr, w in itertools.product(
            ("final_down_proj", "all_down_proj", "all_linear"), (0.0003, 0.001), (1.0, 3.0, 10.0, 30.0)
        )
    ]
    sft_low = [
        bl.BaselineConfig(
            kind="sft", lr=lr, retention_weight=w, steps=300,
            guard_batch=GUARD_BATCH, checkpoints=(*SFT_CHECKPOINTS, 300),
        )
        for lr, w in (*itertools.product((3e-6, 1e-5), (1.0, 3.0, 10.0, 30.0)), (3e-5, 3.0), (3e-5, 30.0))
    ]
    return lora + sft + lora_low + sft_low


def config_from(data: dict) -> bl.BaselineConfig:
    return bl.BaselineConfig(**{**data, "checkpoints": tuple(data.get("checkpoints", ()))})


def load_trials(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def stage_tune(model: PythiaModel, splits: bl.BaselineSplits, out: Path) -> None:
    path = out / "tune_trials.jsonl"
    trials = load_trials(path)
    if any(
        trial.get("dataset", {}).get("dataset_digest") != splits.dataset_digest
        or trial.get("dataset", {}).get("gym_digest") != splits.gym_digest
        for trial in trials
    ):
        raise ValueError("tuning file contains trials from a different Gym or dataset")
    done = {json.dumps(t["config"], sort_keys=True) for t in trials}
    for i, config in enumerate(grid()):
        key = json.dumps(asdict(config), sort_keys=True, default=list)
        if json.dumps(json.loads(key), sort_keys=True) in done:
            continue
        scores, fit = bl.tune_trial(model, config, splits)
        row = {
            "config": asdict(config),
            "dataset": {
                "schema_version": 1,
                "gym_id": splits.gym_id,
                "gym_digest": splits.gym_digest,
                "dataset_digest": splits.dataset_digest,
            },
            "scores": [asdict(s) for s in scores],
            "cost": asdict(fit.cost),
        }
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        best = max(scores, key=lambda s: s.selection_score)
        print(f"[{i}] {config.kind} {config.targets} r={config.rank} lr={config.lr} w={config.retention_weight} "
              f"best@{best.steps}: dev={best.dev_accuracy:.2f} val={best.val_accuracy:.2f} "
              f"ret={best.retention_tune_accuracy:.2f} {fit.cost.wall_seconds:.0f}s", flush=True)


def stage_refit(model: PythiaModel, splits: bl.BaselineSplits, out: Path) -> None:
    trials = load_trials(out / "tune_trials.jsonl")
    if any(
        trial.get("dataset", {}).get("dataset_digest") != splits.dataset_digest
        or trial.get("dataset", {}).get("gym_digest") != splits.gym_digest
        for trial in trials
    ):
        raise ValueError("tuning trials do not match the refit Gym dataset")
    report: dict = {}
    for kind in ("lora", "sft"):
        rows = [t for t in trials if t["config"]["kind"] == kind]
        if not rows:
            raise SystemExit(f"no tuning trials for {kind}")
        selected = bl.select_best(
            (config_from(t["config"]), [bl.CheckpointScore(**s) for s in t["scores"]]) for t in rows
        )
        fit = bl.refit_selected(model, selected, splits)
        torch.save(fit.merged_weights, out / f"{kind}_merged_weights.pt")
        entry = bl.public_report(model, fit, splits, PRACTICE_RETENTION_PROMPTS)
        entry["selected_on"] = asdict(selected.score)
        entry["tuning_trials"] = len(rows)
        entry["tuning_optimizer_steps"] = sum(t["cost"]["optimizer_steps"] for t in rows)
        entry["tuning_wall_seconds"] = sum(t["cost"]["wall_seconds"] for t in rows)
        if kind == "lora":  # Only LoRA has a random start. Public splits only.
            entry["other_seeds"] = []
            for seed in STABILITY_SEEDS:
                other = bl.BaselineConfig(**{**asdict(selected.config), "seed": seed, "checkpoints": ()})
                other_fit = bl.fit_baseline(model, other, dev=splits.dev, retention_guard=splits.refit_guard)
                other_report = bl.public_report(model, other_fit, splits, PRACTICE_RETENTION_PROMPTS)
                entry["other_seeds"].append({"seed": seed, **{
                    k: other_report[k] for k in ("dev_accuracy", "val_accuracy", "practice_retention")
                }})
        report[kind] = entry
        print(kind, json.dumps(entry, indent=1), flush=True)
    (out / "refit_report.json").write_text(json.dumps(report, indent=1))


def stage_final(model: PythiaModel, out: Path, gym=GYM) -> None:
    report = json.loads((out / "refit_report.json").read_text())
    from neural_foundry.gym_data import gym_digest

    expected_gym_digest = gym_digest(gym)
    if any(
        report[kind].get("dataset", {}).get("gym_digest") != expected_gym_digest
        for kind in ("lora", "sft")
    ):
        raise ValueError("refit report does not match the selected Gym")
    try:  # Created before any sealed scoring, so a crash also uses up the attempt.
        os.close(os.open(out / "HOLDOUT_USED", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        raise SystemExit("holdout already used for this run directory") from None
    holdout = bl.SealedHoldout(gym, SEALED_RETENTION_PROMPTS)
    results = {"original_model_control": holdout.evaluate_once("original", model, {})}
    for kind in ("lora", "sft"):
        weights = torch.load(out / f"{kind}_merged_weights.pt")
        if bl.weights_digest(weights) != report[kind]["weights_digest"]:
            raise SystemExit(f"{kind} weights do not match the refit report")
        results[kind] = holdout.evaluate_once(kind, model, weights)
    (out / "sealed_results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("tune", "refit", "final"))
    parser.add_argument("--model-dir", default="/tmp/pythia-14m")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--gym-spec", type=Path, default=None, help="Data-only versioned Gym JSON spec")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    model = PythiaModel.from_pretrained_dir(args.model_dir)
    gym = GYM
    if args.gym_spec is not None:
        from neural_foundry.gym_data import load_gym_spec

        gym = load_gym_spec(args.gym_spec)
    splits = bl.build_splits(gym, PRACTICE_RETENTION_PROMPTS, model=model)
    if args.stage == "tune":
        stage_tune(model, splits, args.out)
    elif args.stage == "refit":
        stage_refit(model, splits, args.out)
    else:
        stage_final(model, args.out, gym=gym)


if __name__ == "__main__":
    sys.exit(main())
