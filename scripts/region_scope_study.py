"""Does scoping a low-rank edit to a column region improve sealed retention over the whole tensor?

Factors follow the AlphaEdit recipe behind `edit_associations`, fitted to the three-skill gym's
development prompts and protecting only the practice retention prompts.

`select` chooses each arm's strength from development, validation and practice data and freezes the
choice to JSON. `reveal` reads SEALED_RETENTION_PROMPTS once, for frozen choices only. No step reads
the gym's holdout rules or prefixes.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from neural_foundry.edit_plan import EditConstraintViolation, EditPlan, Region
from neural_foundry.prd_adapters import PRACTICE_RETENTION_PROMPTS, SEALED_RETENTION_PROMPTS, run_easyedit_adapter
from neural_foundry.pythia import ALGORITHMIC_THREE_SKILL_GYM as GYM
from neural_foundry.pythia import PythiaModel
from neural_foundry.records import NativeError

LAYERS = (3, 4, 5)
WIDTHS = (64, 128, 256)
STRENGTHS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0)
MIN_PRACTICE_RETENTION = 0.9
PRIMARY_WIDTH = 128

DEVELOPMENT = tuple((f"{prefix} {expr}", target) for prefix in GYM.gym_dev_prefixes for expr, target in GYM.base_rules)
VALIDATION = tuple((f"{prefix} {expr}", target) for prefix in GYM.gym_val_prefixes for expr, target in GYM.base_rules)


class Factors:
    """The AlphaEdit fit at unit strength. The result is linear in strength."""

    def __init__(self, model: PythiaModel, layer: int) -> None:
        self.rank = len(DEVELOPMENT)
        out_dim = int(model.config["hidden_size"])
        w_eff = model.effective_unembedding()
        covariance = w_eff.T @ w_eff + 10.0 * torch.eye(out_dim, dtype=torch.float64)
        targets = []
        for _, token in DEVELOPMENT:
            v = torch.linalg.solve(covariance, w_eff[model.tokenizer.vocab[token]])
            v = v - v.mean()
            targets.append(v / (v.std() + 1e-8))
        v_mat = torch.stack(targets, dim=1)
        key = lambda prompt: model.forward_with_mlp_key(model.tokenizer.encode(prompt), capture_layer=layer)[1]  # noqa: E731
        k_mat = torch.stack([key(p) for p, _ in DEVELOPMENT], dim=1)
        self.k_keep = torch.stack([key(p) for p in PRACTICE_RETENTION_PROMPTS], dim=1)
        self.k_sig = torch.cat([k_mat, 3.0 * (k_mat[:, 0::2] - k_mat[:, 1::2])], dim=1)
        self.v_sig = torch.cat([v_mat, 3.0 * (v_mat[:, 0::2] - v_mat[:, 1::2])], dim=1)

    def fit(self, columns: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Left and right factors using only `columns` of the layer's inputs."""
        k_sig, k_keep = self.k_sig[columns], self.k_keep[columns]
        gram = k_sig @ k_sig.T + 40.0 * (k_keep @ k_keep.T) + 1e-3 * torch.eye(len(columns), dtype=torch.float64)
        delta = self.v_sig @ torch.linalg.solve(gram, k_sig).T
        u, s, vh = torch.linalg.svd(delta, full_matrices=False)
        return u[:, : self.rank] * s[: self.rank], vh[: self.rank]

    def best_window(self, width: int) -> tuple[int, int]:
        """The contiguous columns holding the most development activation."""
        energy = (self.k_sig**2).sum(1)
        totals = torch.nn.functional.avg_pool1d(energy[None, None], width, stride=1)[0, 0]
        start = int(totals.argmax())
        return start, start + width


def build_plan(model: PythiaModel, factors: Factors, layer: int, arm: str, strength: float) -> EditPlan:
    site = model.mlp_down_proj_site(layer)
    plan = EditPlan(model)
    in_dim = factors.k_sig.shape[0]
    every = torch.arange(in_dim)
    if arm == "whole":
        left, right = factors.fit(every)
        # Scale the factor itself, as the adapter does: Rust caps the coefficient at 100.
        return plan.low_rank(site, strength * left, right)
    family, width_text = arm.rsplit("_", 1)
    width = int(width_text)
    lo, hi = factors.best_window(width)
    if family == "window_trim":
        left, right = factors.fit(every)
        right = right[:, lo:hi]
    elif family == "window_refit":
        left, right = factors.fit(torch.arange(lo, hi))
    else:
        raise ValueError(arm)
    return plan.low_rank(site, strength * left, right, region=Region(cols=(lo, hi)))


def accuracy(model: Any, data: Sequence[tuple[str, str]]) -> float:
    return sum(model.evaluate_prompt(p, target_token=t).matched_target for p, t in data) / len(data)


def retained(base: Any, edited: Any, prompts: Sequence[str]) -> list[bool]:
    return [base.evaluate_prompt(p).top1_token_id == edited.evaluate_prompt(p).top1_token_id for p in prompts]


def select(checkpoint: str, output: Path) -> None:
    model = PythiaModel.from_pretrained_dir(checkpoint)
    # A sparse top-column arm isn't here: Rust refuses that edit ("actual-coordinate receipt exceeds 4096 spans").
    arms = ["whole", *[f"{family}_{w}" for family in ("window_trim", "window_refit") for w in WIDTHS]]
    frozen: dict[str, Any] = {
        "rule": f"max validation accuracy with practice retention >= {MIN_PRACTICE_RETENTION}; ties: practice retention, then smaller Frobenius change",
        "primary_width": PRIMARY_WIDTH,
        "base_accuracy": {"development": accuracy(model, DEVELOPMENT), "validation": accuracy(model, VALIDATION)},
        "layers": {},
    }
    for layer in LAYERS:
        factors = Factors(model, layer)
        # This fit must be the same edit the product compiles.
        receipt = run_easyedit_adapter(model, algorithm="AlphaEdit", associations=DEVELOPMENT, retention_prompts=PRACTICE_RETENTION_PROMPTS, target_strength=80.0, target_layers=(layer,))
        left, right = factors.fit(torch.arange(factors.k_sig.shape[0]))
        site = model.mlp_down_proj_site(layer)
        gap = float((model.tensors[site] + 80.0 * left @ right - receipt.edited_weight).abs().max())
        if gap > 1e-9:
            raise RuntimeError(f"Fit differs from the adapter by {gap}")
        layer_doc: dict[str, Any] = {"adapter_gap": gap, "windows": {str(w): factors.best_window(w) for w in WIDTHS}, "arms": {}}
        for arm in arms:
            rows = []
            for strength in STRENGTHS:
                try:
                    candidate = build_plan(model, factors, layer, arm, strength).preview()
                except (EditConstraintViolation, ValueError, RuntimeError, NativeError) as exc:
                    rows.append({"strength": strength, "rejected": str(exc)[:200]})
                    continue
                rows.append(
                    {
                        "strength": strength,
                        "development": accuracy(candidate.model, DEVELOPMENT),
                        "validation": accuracy(candidate.model, VALIDATION),
                        "practice_retention": sum(retained(model, candidate.model, PRACTICE_RETENTION_PROMPTS)) / len(PRACTICE_RETENTION_PROMPTS),
                        "frobenius": candidate.frobenius_delta,
                        "changed_parameters": candidate.changed_parameters,
                    }
                )
            usable = [r for r in rows if "rejected" not in r and r["practice_retention"] >= MIN_PRACTICE_RETENTION]
            choice = max(usable, key=lambda r: (r["validation"], r["practice_retention"], -r["frobenius"])) if usable else None
            layer_doc["arms"][arm] = {"choice": choice, "grid": rows}
            print(layer, arm, choice, flush=True)
        frozen["layers"][str(layer)] = layer_doc
    output.write_text(json.dumps(frozen, indent=2), encoding="utf-8")


def _sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def _kl(base: Any, edited: Any, prompts: Sequence[str]) -> torch.Tensor:
    def logp(m: Any) -> torch.Tensor:
        return torch.stack([m.forward_with_mlp_key(m.tokenizer.encode(p))[0][-1] for p in prompts]).log_softmax(-1)

    before, after = logp(base), logp(edited)
    return (before.exp() * (before - after)).sum(-1)


def reveal(checkpoint: str, frozen_path: Path, output: Path) -> None:
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    model = PythiaModel.from_pretrained_dir(checkpoint)
    report: dict[str, Any] = {}
    for layer_text, layer_doc in frozen["layers"].items():
        layer = int(layer_text)
        factors = Factors(model, layer)
        whole_choice = layer_doc["arms"]["whole"]["choice"]
        if whole_choice is None:
            report[layer_text] = {"skipped": "whole-tensor arm had no strength meeting the practice floor"}
            continue
        whole = build_plan(model, factors, layer, "whole", whole_choice["strength"]).preview().model
        whole_kept = retained(model, whole, SEALED_RETENTION_PROMPTS)
        whole_kl = _kl(model, whole, SEALED_RETENTION_PROMPTS)
        rows = {"whole": {"sealed": sum(whole_kept) / len(whole_kept), "mean_kl": float(whole_kl.mean()), **whole_choice}}
        for arm, doc in layer_doc["arms"].items():
            if arm == "whole" or doc["choice"] is None:
                continue
            edited = build_plan(model, factors, layer, arm, doc["choice"]["strength"]).preview().model
            kept = retained(model, edited, SEALED_RETENTION_PROMPTS)
            kl = _kl(model, edited, SEALED_RETENTION_PROMPTS)
            wins = sum(a and not b for a, b in zip(kept, whole_kept, strict=True))
            losses = sum(b and not a for a, b in zip(kept, whole_kept, strict=True))
            kl_wins, kl_losses = int((kl < whole_kl).sum()), int((kl > whole_kl).sum())
            rows[arm] = {
                "sealed": sum(kept) / len(kept),
                "top1_wins_vs_whole": wins,
                "top1_losses_vs_whole": losses,
                "top1_sign_p": _sign_test(wins, losses),
                "mean_kl": float(kl.mean()),
                "kl_wins_vs_whole": kl_wins,
                "kl_losses_vs_whole": kl_losses,
                "kl_sign_p": _sign_test(kl_wins, kl_losses),
                **doc["choice"],
            }
        report[layer_text] = rows
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "reveal"))
    parser.add_argument("--checkpoint", default="/tmp/pythia-14m")
    parser.add_argument("--frozen", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == "select":
        select(args.checkpoint, args.frozen)
    else:
        reveal(args.checkpoint, args.frozen, args.output or args.frozen.with_name("sealed.json"))


if __name__ == "__main__":
    main()
