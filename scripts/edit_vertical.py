"""Take one edit through every stage on a real checkpoint and write the evidence as JSON.

Stages: preview, execute, compare, sign, replay in a fresh process, export, reload, and independent
verification in `transformers`. To cover another operation, add it to OPERATIONS.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neural_foundry import patch_ux
from neural_foundry.edit_plan import (
    EditPlan,
    sign_edit_graph,
    tensor_state_digest,
    verify_exported_candidate,
)
from neural_foundry.prd_adapters import PRACTICE_RETENTION_PROMPTS, SEALED_RETENTION_PROMPTS
from neural_foundry.pythia import ALGORITHMIC_THREE_SKILL_GYM, PythiaModel

GYM = ALGORITHMIC_THREE_SKILL_GYM
OPERATIONS: dict[str, Callable[[EditPlan, str, torch.Tensor], Any]] = {
    "spectral_clip": lambda plan, site, w: plan.spectral_clip(site, 0.9 * float(torch.linalg.svdvals(w)[0])),
}


def _logits(model: Any, prompts: Sequence[str]) -> torch.Tensor:
    return torch.stack([model.forward_with_mlp_key(model.tokenizer.encode(p))[0][-1] for p in prompts])


def compare(base: Any, edited: Any, prompts: Sequence[str]) -> dict[str, Any]:
    """Top-1 retention and how far the next-token distribution moved."""
    before, after = _logits(base, prompts), _logits(edited, prompts)
    kl = (before.log_softmax(-1).exp() * (before.log_softmax(-1) - after.log_softmax(-1))).sum(-1)
    kept = before.argmax(-1) == after.argmax(-1)
    return {
        "prompts": len(prompts),
        "retained": int(kept.sum()),
        "retention": int(kept.sum()) / len(prompts),
        "changed_prompts": [p for p, k in zip(prompts, kept.tolist(), strict=True) if not k],
        "mean_kl": float(kl.mean()),
        "max_kl": float(kl.max()),
    }


def gym_accuracy(model: Any, prefixes: Sequence[str]) -> float:
    prompts = [(f"{prefix} {expr}", target) for prefix in prefixes for expr, target in GYM.base_rules]
    return sum(model.evaluate_prompt(p, target_token=t).matched_target for p, t in prompts) / len(prompts)


def _fresh_env() -> dict[str, str]:
    return {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}


def run_vertical(checkpoint: Path, work: Path, operation: str, layer: int) -> dict[str, Any]:
    work.mkdir(parents=True, exist_ok=True)
    base = PythiaModel.from_pretrained_dir(checkpoint)
    site = base.mlp_down_proj_site(layer)
    weight = base.tensors[site].clone()
    base_digest = tensor_state_digest(base.tensors)
    evidence: dict[str, Any] = {"checkpoint": str(checkpoint), "operation": operation, "site": site, "base_state_digest": base_digest}

    plan = EditPlan(base)
    OPERATIONS[operation](plan, site, weight)

    preview = plan.preview()
    step = preview.steps[0]
    evidence["preview"] = {
        "plan_digest": preview.plan_digest,
        "state_digest": preview.state_digest,
        "changed_parameters": preview.changed_parameters,
        "frobenius_delta": preview.frobenius_delta,
        "spectral_delta": preview.spectral_delta,
        "spectrum_before": step.spectrum_before[:4] if step.spectrum_before else None,
        "spectrum_after": step.spectrum_after[:4] if step.spectrum_after else None,
        "base_untouched": tensor_state_digest(base.tensors) == base_digest,
    }

    candidate = plan.execute()
    evidence["execute"] = {"state_digest": candidate.state_digest, "same_as_preview": candidate.state_digest == preview.state_digest}

    evidence["compare"] = {
        "practice": compare(base, candidate.model, PRACTICE_RETENTION_PROMPTS),
        "sealed": compare(base, candidate.model, SEALED_RETENTION_PROMPTS),
        "gym_development_accuracy": {"base": gym_accuracy(base, GYM.gym_dev_prefixes), "edited": gym_accuracy(candidate.model, GYM.gym_dev_prefixes)},
    }

    keys = patch_ux.generate_ed25519_keypair()
    graph = plan.to_graph()
    envelope_path = work / "envelope.json"
    envelope_path.write_text(json.dumps(sign_edit_graph(graph, signing_key=keys["private_key"])), encoding="utf-8")
    public_key_path = work / "public_key.txt"
    public_key_path.write_text(keys["public_key"], encoding="utf-8")
    evidence["sign"] = {"envelope_bytes": envelope_path.stat().st_size}

    replay_export = work / "replayed_export"
    replay = subprocess.run(
        [sys.executable, "-m", "neural_foundry.edit_plan", "--checkpoint", str(checkpoint), "--envelope", str(envelope_path),
         "--public-key-file", str(public_key_path), "--export-dir", str(replay_export)],
        capture_output=True, text=True, env=_fresh_env(), cwd=work,
    )
    if replay.returncode != 0:
        raise RuntimeError(f"Fresh-process replay failed: {replay.stderr[-2000:]}")
    replayed = json.loads(replay.stdout.strip().splitlines()[-1])
    evidence["replay"] = {
        "own_pid": os.getpid(),
        "state_digest": replayed["candidate_state_digest"],
        "matches_candidate": replayed["candidate_state_digest"] == candidate.state_digest,
        "plan_digest_matches": replayed["plan_digest"] == candidate.plan_digest,
    }

    export_dir = work / "export"
    plan.export(export_dir, dtype="f64", source_checkpoint_dir=checkpoint)
    evidence["export"] = {
        "bytes": (export_dir / "model.safetensors").stat().st_size,
        "reloaded_state_digest": verify_exported_candidate(export_dir, candidate.state_digest),
        "replayed_export_digest": replayed["export_state_digest"],
    }
    reloaded = PythiaModel.from_pretrained_dir(export_dir)
    evidence["reload"] = {
        "state_digest": tensor_state_digest(reloaded.tensors),
        "matches_candidate": tensor_state_digest(reloaded.tensors) == candidate.state_digest,
    }

    prompts = [*PRACTICE_RETENTION_PROMPTS, *SEALED_RETENTION_PROMPTS]
    prompts_path = work / "prompts.json"
    prompts_path.write_text(json.dumps(prompts), encoding="utf-8")
    logits_path = work / "hf_logits.npz"
    verifier = Path(__file__).with_name("verify_export_independent.py")
    independent = subprocess.run(
        [sys.executable, str(verifier), "--export-dir", str(export_dir), "--base-dir", str(checkpoint),
         "--prompts-file", str(prompts_path), "--output", str(logits_path)],
        capture_output=True, text=True, env=_fresh_env(), cwd=work,
    )
    if independent.returncode != 0:
        raise RuntimeError(f"Independent verifier failed: {independent.stderr[-2000:]}")
    report = json.loads(independent.stdout.strip().splitlines()[-1])
    with np.load(logits_path) as hf:
        hf_edited, hf_base = torch.tensor(hf["exported"]), torch.tensor(hf["base"])
    ours_edited, ours_base = _logits(candidate.model, prompts), _logits(base, prompts)
    tokens_agree = sum(base.tokenizer.encode(p) == ids for p, ids in zip(prompts, report["token_ids"], strict=True))
    evidence["independent"] = {
        "state_digest": report["state_digest"],
        "matches_candidate": report["state_digest"] == candidate.state_digest,
        "tokenizations_agree": f"{tokens_agree}/{len(prompts)}",
        "max_abs_logit_diff_edited": float((hf_edited - ours_edited).abs().max()),
        "max_abs_logit_diff_base": float((hf_base - ours_base).abs().max()),
        "top1_agreement_edited": float((hf_edited.argmax(-1) == ours_edited.argmax(-1)).float().mean()),
        "edit_visible_in_hf_max_logit_change": float((hf_edited - hf_base).abs().max()),
    }
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="/tmp/pythia-14m")
    parser.add_argument("--work", required=True)
    parser.add_argument("--operation", default="spectral_clip", choices=sorted(OPERATIONS))
    parser.add_argument("--layer", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(run_vertical(Path(args.checkpoint), Path(args.work), args.operation, args.layer), indent=2))


if __name__ == "__main__":
    main()
