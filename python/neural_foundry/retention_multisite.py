"""Do multi-site edits separate task from general behaviour better than one linear edit? What does each method cost?

Phases, run separately:

    tune    Sweeps closed-form families on public splits only (dev, extra dev prefixes, validation, practice, tuning).
    freeze  Selects one configuration per family from public scores, trains the gradient comparators, writes frozen.json.
    seal    Scores frozen methods once on the V3 sealed sets and the old holdout. Refuses repeats through the ledger.
    cost    One method, one process: wall time, peak memory, plan size, changed parameters, bit-exact restoration.

`tune` and `freeze` have no code path to the V3 sets, the holdout rules, or the older sealed sets. The older sealed
sets are never scored here, in any phase; they appear only in the disjointness check of the V3 set (see the tests).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import resource
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from . import baselines as bl
from . import retention_research as rr
from .prd_adapters import PRACTICE_RETENTION_PROMPTS
from .pythia import ALGORITHMIC_THREE_SKILL_GYM as GYM
from .pythia import PythiaModel, _compile_and_apply_rust_low_rank_update

# Written before any result from this work was seen. Half short sentence stems, half other genres.
SEALED_V3_STEMS: tuple[str, ...] = (
    "Lisbon is the main city of", "Long before the war, the", "The reverse of early is", "Six plus three equals",
    "Jupiter is a", "My youngest cousin lives in", "They painted the fence and", "Steel is made from",
    "The ferry arrived at the", "Every autumn our village holds a", "The mayor of the", "We cycled along the",
    "That novel's author was", "Elephants are famous for their", "The weather this weekend looks", "Let's meet outside the",
    "December comes right after", "An octagon has eight", "Seawater tastes quite", "Would you pass me the",
    "The tallest mountain on Earth is", "Steam is the gaseous form of", "Her uncle repairs old", "The bridge connects the",
    "Cheese is made from", "The nurse handed the", "We ran out of", "The puppy began to",
    "The bus from the airport stops at", "Most fish breathe through", "A ripe banana looks", "Snow melts when the",
    "Their small cottage had a", "The lawyer advised her to", "A year has twelve", "The gardener watered",
    "Teenagers often spend hours on", "The library opens at", "Summer is warmer than", "The orchestra performed a",
    "Every triangle contains", "The museum on the corner", "Honey is produced by", "The waiter brought us",
    "Coffee beans are grown in", "He tied his shoes before", "The clock on the wall", "A kilometre is longer than a",
    "The pilot announced that", "My neighbour keeps a", "Paper is made from", "The prince married the",
    "A good night's sleep helps", "The lake froze during", "Wolves usually hunt in", "Canada is north of the",
    "The artist mixed blue and", "The students cheered when", "Copper is a type of", "The ship sank near the",
)
SEALED_V3_GENRES: tuple[str, ...] = (
    "fn main() {\n    let", "for (int i = 0; i <", "git checkout -b", "<img src=\"", "yarn add --dev",
    "To whom it may concern, we wish", "Yours sincerely,\n", "BREAKING NEWS: police in Texas are",
    "Q: What causes the tides?\nA:", "Instructions: preheat the oven and", "Prologue\n\nThe night the lights went out",
    "\"Get out of here,\" he shouted, \"before", "ngl that game last night was", "By using this website you agree to",
    "Enzymes speed up chemical reactions by", "A new poll suggests that younger adults", "Saturday, Sunday, Monday,",
    "violet, indigo, blue,", "2, 4, 6, 8,", "Oil prices climbed on Tuesday after", "Terrible service. We waited an hour and",
    "Twinkle, twinkle, little", "Changing a flat tire? Start by", "Background: Prior studies have shown that",
    "The jury found the accused", "Whisk the batter until it", "Researchers in Japan have developed a",
    "Go straight past the church, then", "UPDATE accounts SET balance =", "If you enjoyed this video, please",
    "In 1492, the expedition reached", "His least favourite chore at home was", "Caution: keep this medicine away from",
    "The pitcher struck out the", "This release fixes a bug where", "Es war einmal ein", "Pictures from the concert were shared on",
    "The screen on this tablet is", "She shrugged, grabbed her coat, and", "The match was cancelled because of",
    "export default function", "sudo apt-get update &&", "// TODO: remove this once", "body { margin: 0;", "curl -X POST https://",
    "Meeting notes: the budget review", "Hey! Are you free this", "Disclaimer: the views expressed here", "Step 1: Unplug the",
    "Tip of the day: always", "Fun fact: octopuses have", "Lyrics to the old folk song", "Dear diary, today I",
    "Recipe notes: this sauce keeps", "Headline: Local bakery wins", "Patch notes: improved loading of", "Translation: the cat is",
    "Review: a slow but rewarding", "Dr. Patel explained that the", "Tickets go on sale at",
)
SEALED_V3_RETENTION: tuple[str, ...] = (*SEALED_V3_STEMS, *SEALED_V3_GENRES)

# New task items, new prefixes. Unlike the existing holdout, these include True AND True and False OR False,
# so mapping AND to False and OR to True scores 2/4 on the Boolean items rather than 4/4.
SEALED_V3_TASK: tuple[tuple[str, str], ...] = (
    ("Entry: Number 3 has parity", "Ġodd"),
    ("Record: Number 5 has parity", "Ġodd"),
    ("Entry: Number 6 has parity", "Ġeven"),
    ("Record: Number 8 has parity", "Ġeven"),
    ("Entry: Boolean True AND True gives", "ĠTrue"),
    ("Record: Boolean False AND False gives", "ĠFalse"),
    ("Entry: Boolean False OR False gives", "ĠFalse"),
    ("Record: Boolean True OR True gives", "ĠTrue"),
    ("Entry: Relation 8 > 5 is", "Ġyes"),
    ("Record: Relation 6 > 1 is", "Ġyes"),
    ("Entry: Relation 3 > 9 is", "Ġno"),
    ("Record: Relation 4 > 7 is", "Ġno"),
)
TASK_TOKENS = ("Ġodd", "Ġeven", "ĠFalse", "ĠTrue", "Ġyes", "Ġno")
READOUT_SITE = "embed_out.weight"
SEALED_SPLITS = ("sealed_v3_retention", "sealed_v3_task", "holdout_old")


def sealed_v3_digest() -> str:
    return rr._digest((*SEALED_V3_RETENTION, *(p + "|" + t for p, t in SEALED_V3_TASK)))


@dataclass(frozen=True)
class MultiConfig:
    family: str  # single, parallel, joint, sequential, compensated, readout, readout_plus
    layers: tuple[int, ...] = (5,)
    strength: float = 1.0  # Residual-stream target scale; for readout, the logit margin.
    alpha: float = 1000.0  # Soft penalty on the mean second moment of the fit corpus.
    ridge: float = 1.0
    rank_limit: int = 12
    readout_margin: float = 0.0  # readout_plus only: margin of the readout step.
    readout_alpha: float = 0.0  # readout_plus only.
    fit_subset_seed: int | None = None


@dataclass
class MultiCandidate:
    """Merged tensors by site, plus the low-rank factors that the executor receives."""

    merged: dict[str, torch.Tensor]
    factors: dict[str, tuple[torch.Tensor, torch.Tensor, tuple[int, ...] | None]]  # left, right, rows (None: all)
    info: dict[str, Any] = field(default_factory=dict)
    digests: dict[str, str] = field(default_factory=dict)


def low_rank(delta: torch.Tensor, rank_limit: int) -> tuple[torch.Tensor, torch.Tensor]:
    u, s, vh = torch.linalg.svd(delta, full_matrices=False)
    rank = min(rank_limit, int((s > 1e-9).sum()))
    if rank == 0:
        raise ValueError("The solve produced no edit")
    return u[:, :rank] * s[:rank].unsqueeze(0), vh[:rank]


def ridge_map(
    k: torch.Tensor, y: torch.Tensor, kp: torch.Tensor, yp: torch.Tensor | None, alpha: float, ridge: float
) -> torch.Tensor:
    """argmin_D ||D k - y||^2 + (alpha/N) ||D kp - yp||^2 + ridge ||D||^2. yp=None means a zero target."""
    n = k.shape[0]
    gram = k @ k.T + (alpha / kp.shape[1]) * (kp @ kp.T) + ridge * torch.eye(n, dtype=torch.float64)
    rhs = y @ k.T
    if yp is not None:
        rhs = rhs + (alpha / kp.shape[1]) * (yp @ kp.T)
    return torch.linalg.solve(gram, rhs.T).T


def contrast(m: torch.Tensor) -> torch.Tensor:
    """The adapter's contrastive pair rows."""
    return torch.cat([m, 3.0 * (m[:, 0::2] - m[:, 1::2])], dim=1)


class MultiLab:
    """Solves and public scoring. Sealed sets are reachable only from `run_seal`."""

    def __init__(self, model: PythiaModel) -> None:
        self.model = model
        self.lab = rr.Lab(model)
        self.dev_prompts = [p for p, _ in self.lab.dev_items]
        self._state: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._solve_cache: dict[tuple[Any, ...], dict[int, torch.Tensor]] = {}
        self._base_pred: dict[str, tuple[int, torch.Tensor]] = {}
        self.task_ids = [model.tokenizer.vocab[t] for t in TASK_TOKENS]

    def fit_prompts(self, seed: int | None) -> list[str]:
        fit = list(self.lab.fit)
        if seed is None:
            return fit
        return sorted(random.Random(seed).sample(fit, int(0.8 * len(fit))))

    def key_and_residual(
        self, prompt: str, layer: int, overrides: dict[int, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sink: list[torch.Tensor] = []
        _, key = self.model.forward_with_mlp_key(
            self.model.tokenizer.encode(prompt), capture_layer=layer, layer_down_proj_override=overrides, residual_sink=sink
        )
        return key, sink[0][-1]

    def state(
        self, prompts: Sequence[str], layer: int, overrides: dict[int, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(keys, final residuals) at the last token, as columns. The unedited model's values are cached."""
        cache_keys: list[tuple[str, int]] = [(p, layer) for p in prompts]
        pairs = []
        for p, c in zip(prompts, cache_keys, strict=True):
            if overrides:
                pairs.append(self.key_and_residual(p, layer, overrides))
            else:
                if c not in self._state:
                    self._state[c] = self.key_and_residual(p, layer)
                pairs.append(self._state[c])
        return torch.stack([a for a, _ in pairs], dim=1), torch.stack([b for _, b in pairs], dim=1)

    def down(self, layer: int) -> torch.Tensor:
        return self.model.tensors[self.model.mlp_down_proj_site(layer)]

    def _candidate(self, deltas: Mapping[int, torch.Tensor], rank_limit: int) -> MultiCandidate:
        merged: dict[str, torch.Tensor] = {}
        factors: dict[str, tuple[torch.Tensor, torch.Tensor, tuple[int, ...] | None]] = {}
        for layer, delta in deltas.items():
            left, right = low_rank(delta, rank_limit)
            site = self.model.mlp_down_proj_site(layer)
            merged[site] = self.down(layer) + left @ right
            factors[site] = (left, right, None)
        return MultiCandidate(merged, factors)

    def solve(self, cfg: MultiConfig) -> MultiCandidate:
        if cfg.family in ("single", "parallel", "joint"):
            return self._candidate(self._linear(cfg), cfg.rank_limit)
        if cfg.family in ("sequential", "compensated"):
            return self._candidate(self._sequential(cfg), cfg.rank_limit)
        if cfg.family == "readout":
            return self._readout(cfg, MultiCandidate({}, {}), cfg.strength, cfg.alpha)
        if cfg.family == "readout_plus":
            first = self.solve(dataclasses.replace(cfg, family="single", layers=(max(cfg.layers),)))
            return self._readout(cfg, first, cfg.readout_margin, cfg.readout_alpha)
        raise ValueError(f"Unknown family {cfg.family!r}")

    def _linear(self, cfg: MultiConfig) -> dict[int, torch.Tensor]:
        """Maps at strength 1 are linear in the target, so they are cached across the strength grid."""
        key = (cfg.family, cfg.layers, cfg.alpha, cfg.ridge, cfg.fit_subset_seed)
        if key not in self._solve_cache:
            fit = self.fit_prompts(cfg.fit_subset_seed)
            target = contrast(self.lab.unit_targets)
            per_layer = {l: (contrast(self.state(self.dev_prompts, l)[0]), self.state(fit, l)[0]) for l in cfg.layers}
            if cfg.family == "joint":
                z = torch.cat([per_layer[l][0] for l in cfg.layers], dim=0)
                zp = torch.cat([per_layer[l][1] for l in cfg.layers], dim=0)
                joint = ridge_map(z, target, zp, None, cfg.alpha, cfg.ridge)
                width = per_layer[cfg.layers[0]][0].shape[0]
                maps = {l: joint[:, i * width : (i + 1) * width] for i, l in enumerate(cfg.layers)}
            else:
                # The sites share the target, so that first-order effects at the final residual add up to it.
                share = target / len(cfg.layers)
                maps = {l: ridge_map(k, share, kp, None, cfg.alpha, cfg.ridge) for l, (k, kp) in per_layer.items()}
            self._solve_cache[key] = maps
        return {l: m * cfg.strength for l, m in self._solve_cache[key].items()}

    def _sequential(self, cfg: MultiConfig) -> dict[int, torch.Tensor]:
        """Solves each layer on keys measured under the earlier edits.

        `sequential` shares the remaining task target evenly. `compensated` puts the whole target on the first layer;
        the last layer then restores the task target and is also told to cancel the drift seen on the fit corpus.
        """
        fit = self.fit_prompts(cfg.fit_subset_seed)
        target = contrast(self.lab.unit_targets) * cfg.strength
        last = cfg.layers[-1]
        _, x0_task = self.state(self.dev_prompts, last)
        _, x0_fit = self.state(fit, last)
        overrides: dict[int, torch.Tensor] = {}
        deltas: dict[int, torch.Tensor] = {}
        n = len(cfg.layers)
        for pos, layer in enumerate(cfg.layers):
            k_task, x_task = self.state(self.dev_prompts, layer, overrides or None)
            k_fit, x_fit = self.state(fit, layer, overrides or None)
            remaining = target - contrast(x_task - x0_task)
            final = pos == n - 1
            cancel = cfg.family == "compensated" and final and pos > 0
            share = 1.0 if cfg.family == "compensated" else 1.0 / (n - pos)
            yp = -(x_fit - x0_fit) if cancel else None
            delta = ridge_map(contrast(k_task), remaining * share, k_fit, yp, cfg.alpha, cfg.ridge)
            left, right = low_rank(delta, cfg.rank_limit)
            delta = left @ right
            deltas[layer] = delta
            overrides[layer] = self.down(layer) + delta
        return deltas

    def normed(self, prompts: Sequence[str], tensors: Mapping[str, torch.Tensor] | None) -> torch.Tensor:
        """The final layer norm's output at the last token, which is exactly what the readout multiplies."""
        model = bl.merged_model(self.model, tensors or {})
        identity = torch.eye(int(self.model.config["hidden_size"]), dtype=torch.float64)
        probe = bl.merged_model(model, {READOUT_SITE: identity})
        return torch.stack([probe.forward_with_mlp_key(self.model.tokenizer.encode(p))[0][-1] for p in prompts], dim=1)

    def _readout(self, cfg: MultiConfig, first: MultiCandidate, margin: float, alpha: float) -> MultiCandidate:
        """Edits only the output rows of the six task tokens. Linear and exact, as nothing follows this matrix."""
        fit = self.fit_prompts(cfg.fit_subset_seed)
        z_task = self.normed(self.dev_prompts, first.merged)
        z_fit = self.normed(fit, first.merged)
        e = self.model.tensors[READOUT_SITE]
        logits = e @ z_task
        y = torch.zeros((len(self.task_ids), z_task.shape[1]), dtype=torch.float64)
        for i, (_, tok) in enumerate(self.lab.dev_items):
            tid = self.model.tokenizer.vocab[tok]
            others = logits[:, i].clone()
            others[tid] = -float("inf")
            y[self.task_ids.index(tid), i] = max(0.0, float(others.max() - logits[tid, i])) + margin
        delta = ridge_map(z_task, y, z_fit, None, alpha, cfg.ridge)
        left, right = low_rank(delta, cfg.rank_limit)
        merged = dict(first.merged)
        full = e.clone()
        full[self.task_ids] = e[self.task_ids] + left @ right
        merged[READOUT_SITE] = full
        factors = dict(first.factors)
        factors[READOUT_SITE] = (left, right, tuple(self.task_ids))
        return MultiCandidate(merged, factors)

    def execute(self, cand: MultiCandidate) -> MultiCandidate:
        """Applies the factors through the Rust executor and requires agreement with the sweep's own arithmetic.

        The readout has 6.4M coordinates, beyond the executor's 16 MiB source bound at f64, so its six rows are
        executed as their own 6-by-hidden matrix and spliced back. That splice is Python, and the receipt covers rows only.
        """
        merged: dict[str, torch.Tensor] = {}
        digests: dict[str, str] = {}
        for site, (left, right, rows) in cand.factors.items():
            base = self.model.tensors[site]
            sub = base if rows is None else base[list(rows)]
            probe = torch.ones(sub.shape[1], dtype=torch.float64) / sub.shape[1] ** 0.5
            edited, plan, _, _, _, reload_ok = _compile_and_apply_rust_low_rank_update(
                w_orig=sub, left_factors=left.contiguous(), right_factors=right.contiguous(), smoke_probe_vec=probe
            )
            if not reload_ok:
                raise RuntimeError("Rust executor reload parity failed")
            if rows is None:
                merged[site] = edited
            else:
                full = base.clone()
                full[list(rows)] = edited
                merged[site] = full
            error = float((merged[site] - cand.merged[site]).abs().max())
            if error > 1e-9:
                raise RuntimeError(f"Executor disagrees with the sweep at {site}: {error}")
            digests[site] = plan
        return MultiCandidate(merged, cand.factors, cand.info, digests)

    def random_like(self, cand: MultiCandidate, seed: int = 23) -> MultiCandidate:
        """A random edit with the same sites, rows, Frobenius norm and rank at each site."""
        g = torch.Generator().manual_seed(seed)
        merged: dict[str, torch.Tensor] = {}
        factors: dict[str, tuple[torch.Tensor, torch.Tensor, tuple[int, ...] | None]] = {}
        for site, (left, right, rows) in cand.factors.items():
            norm = float(torch.linalg.norm(left @ right))
            a = torch.randn(left.shape, generator=g, dtype=torch.float64)
            b = torch.randn(right.shape, generator=g, dtype=torch.float64)
            a = a * (norm / float(torch.linalg.norm(a @ b)))
            base = self.model.tensors[site]
            full = base.clone()
            if rows is None:
                full = base + a @ b
            else:
                full[list(rows)] = base[list(rows)] + a @ b
            merged[site] = full
            factors[site] = (a, b, rows)
        return self.execute(MultiCandidate(merged, factors, {"control": "random_same_norm_rank"}))

    def base_pred(self, prompt: str) -> tuple[int, torch.Tensor]:
        if prompt not in self._base_pred:
            log_p = F.log_softmax(self.model.forward_with_mlp_key(self.model.tokenizer.encode(prompt))[0][-1], dim=-1)
            self._base_pred[prompt] = (int(log_p.argmax()), log_p)
        return self._base_pred[prompt]

    def retention_detail(self, prompts: Sequence[str], merged: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        """Top-1 retention can flatter an edit that moves probability below the top token, so KL and task mass go with it."""
        model = bl.merged_model(self.model, merged)
        kept: list[bool] = []
        kl = 0.0
        mass = 0.0
        for p in prompts:
            top, base_lp = self.base_pred(p)
            lp = F.log_softmax(model.forward_with_mlp_key(self.model.tokenizer.encode(p))[0][-1], dim=-1)
            kept.append(int(lp.argmax()) == top)
            kl += float((base_lp.exp() * (base_lp - lp)).sum())
            mass += float(lp[self.task_ids].exp().sum())
        return {"kept": kept, "retention": sum(kept) / len(kept), "mean_kl": kl / len(kept), "task_token_mass": mass / len(kept)}

    def task_hits(self, items: Sequence[tuple[str, str]], merged: Mapping[str, torch.Tensor]) -> list[bool]:
        model = bl.merged_model(self.model, merged)
        return [bool(model.evaluate_prompt(p, target_token=t).matched_target) for p, t in items]

    def score_public(self, merged: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        hits = self.task_hits(self.lab.dev_items, merged)
        scores: dict[str, Any] = {"dev": sum(hits) / len(hits)}
        if scores["dev"] < 0.5:  # Below every frontier bar, so retention is not worth computing.
            return scores
        scores["dev_aug"] = sum(self.task_hits(self.lab.dev_aug_items, merged)) / len(self.lab.dev_aug_items)
        scores["val"] = sum(self.task_hits(self.lab.val_items, merged)) / len(self.lab.val_items)
        practice = self.retention_detail(PRACTICE_RETENTION_PROMPTS, merged)
        tune = self.retention_detail(self.lab.tune, merged)
        scores.update(
            practice=practice["retention"], tune=tune["retention"], tune_kl=tune["mean_kl"], tune_task_mass=tune["task_token_mass"]
        )
        scores["delta_norm"] = delta_norm(self.model, merged)
        scores["changed_parameters"] = changed_parameters(self.model, merged)
        return scores


def delta_norm(model: PythiaModel, merged: Mapping[str, torch.Tensor]) -> float:
    return sum(float(((v - model.tensors[k]) ** 2).sum()) for k, v in merged.items()) ** 0.5


def changed_parameters(model: PythiaModel, merged: Mapping[str, torch.Tensor]) -> int:
    return sum(int((v != model.tensors[k]).sum()) for k, v in merged.items())


LAYER_SETS = ((4, 5), (3, 5), (2, 5), (3, 4, 5), (0, 1, 2, 3, 4, 5))
SEQ_SETS = ((4, 5), (3, 5), (2, 5))
STRENGTHS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0, 80.0)
ALPHAS = (1e2, 1e3, 1e4, 1e5)
RIDGES = (1e-3, 1.0, 100.0)


def grids() -> dict[str, list[MultiConfig]]:
    product = [(s, a, r) for s in STRENGTHS for a in ALPHAS for r in RIDGES]
    seq = [(s, a, r) for s in (0.1, 0.3, 1.0, 5.0) for a in (1e3, 1e4, 1e5) for r in (1e-3, 1.0)]
    return {
        "single": [MultiConfig("single", (l,), s, a, r) for l in (5, 4, 3, 2) for s, a, r in product],
        "parallel": [MultiConfig("parallel", ls, s, a, r) for ls in LAYER_SETS for s, a, r in product],
        "joint": [MultiConfig("joint", ls, s, a, r) for ls in LAYER_SETS for s, a, r in product],
        "sequential": [MultiConfig("sequential", ls, s, a, r) for ls in SEQ_SETS for s, a, r in seq],
        "compensated": [MultiConfig("compensated", ls, s, a, r) for ls in SEQ_SETS for s, a, r in seq],
        "readout": [
            MultiConfig("readout", (), m, a, r, rank_limit=6)
            for m in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
            for a in (1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)
            for r in RIDGES
        ],
        "readout_plus": [
            MultiConfig("readout_plus", (5,), s, a, 1.0, readout_margin=m, readout_alpha=ra)
            for s in (0.05, 0.1, 0.2, 0.5)
            for a in (1e3, 1e4)
            for m in (0.5, 1.0, 2.0, 5.0)
            for ra in (10.0, 100.0, 1e3)
        ],
    }


def config_from(data: Mapping[str, Any]) -> MultiConfig:
    return MultiConfig(**{**data, "layers": tuple(data["layers"])})


def run_tune(model: PythiaModel, families: Sequence[str], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    lab = MultiLab(model)
    all_grids = grids()
    for family in families:
        path = out / f"sweep_{family}.jsonl"
        start = time.time()
        with path.open("w", encoding="utf-8") as f:
            if family == "gradient_fit_guard":
                _tune_gradient(lab, f)
            else:
                for i, cfg in enumerate(all_grids[family]):
                    row: dict[str, Any] = {"family": family, "config": dataclasses.asdict(cfg)}
                    try:
                        row["scores"] = lab.score_public(lab.solve(cfg).merged)
                    except (ValueError, RuntimeError) as exc:
                        row["error"] = str(exc)
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    if i % 50 == 0:
                        print(f"{family} {i}/{len(all_grids[family])} {time.time() - start:.0f}s", flush=True)
        print(f"{family} done in {time.time() - start:.0f}s", flush=True)


def gradient_recipes() -> list[bl.BaselineConfig]:
    """The published recipes, plus a stronger guard. Given the same fit corpus the edits may protect."""
    lora = [
        bl.BaselineConfig(kind="lora", targets="all_down_proj", rank=4, lr=3e-4, steps=300, retention_weight=w, guard_batch=8, checkpoints=(150, 300))
        for w in (1.0, 10.0)
    ]
    sft = [
        bl.BaselineConfig(kind="sft", lr=3e-5, steps=150, retention_weight=w, guard_batch=8, checkpoints=(50, 100, 150))
        for w in (1.0, 10.0)
    ]
    return [*lora, *sft]


def _tune_gradient(lab: MultiLab, f: Any) -> None:
    for config in gradient_recipes():
        def record(step: int, weights: dict[str, torch.Tensor], wall: float, config: bl.BaselineConfig = config) -> None:
            row = {
                "family": f"{config.kind}_fit_guard",
                "config": {**dataclasses.asdict(config), "steps": step, "checkpoints": []},
                "scores": lab.score_public(weights),
                "wall_seconds": wall,
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(row["family"], step, config.retention_weight, row["scores"], flush=True)

        bl.fit_baseline(lab.model, config, dev=lab.lab.dev_items, retention_guard=lab.lab.fit, on_checkpoint=record)


def selection_key(s: Mapping[str, Any], floor: float) -> tuple[Any, ...]:
    """The same rule as retention_research: the task bar first, then practice and tuning retention, then a smaller edit."""
    return rr.selection_key({**{"practice": 0.0, "tune": 0.0, "delta_norm": 0.0, "dev_aug": 0.0}, **s}, floor)


def load_sweep(out: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(out.glob("sweep_*.jsonl")):
        rows += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


def frontier(rows: Sequence[Mapping[str, Any]], bars: Sequence[float] = (1.0, 11 / 12, 10 / 12, 9 / 12, 6 / 12)) -> dict[str, Any]:
    """Best tuning retention per family at each development bar. Winner's optimism applies to every cell."""
    table: dict[str, Any] = {}
    for family in sorted({r["family"] for r in rows}):
        table[family] = {}
        for bar in bars:
            eligible = [r for r in rows if r["family"] == family and "tune" in r.get("scores", {}) and r["scores"]["dev"] >= bar - 1e-9]
            if eligible:
                top = max(eligible, key=lambda r: r["scores"]["tune"])
                table[family][f"{bar:.3f}"] = {"tune": top["scores"]["tune"], "tune_kl": top["scores"]["tune_kl"], "config": top["config"]}
            else:
                table[family][f"{bar:.3f}"] = None
    return table


def trained_fit(model: PythiaModel, spec: Mapping[str, Any], lab: MultiLab) -> bl.BaselineFit:
    config = bl.BaselineConfig(**{**spec["config"], "checkpoints": ()})
    splits = bl.build_splits(GYM, PRACTICE_RETENTION_PROMPTS, model=model)
    guard = lab.lab.fit if spec["guard"] == "fit" else splits.refit_guard
    return bl.fit_baseline(model, config, dev=lab.lab.dev_items, retention_guard=guard)


def run_freeze(model: PythiaModel, out: Path, frozen_path: Path, weights_dir: Path, refit_report: Path, prior_tuning: Path) -> dict[str, Any]:
    lab = MultiLab(model)
    rows = load_sweep(out)
    prior = json.loads(prior_tuning.read_text(encoding="utf-8"))
    floor = prior["dev_aug_floor"]
    selected: dict[str, Any] = {}
    for family in sorted({r["family"] for r in rows}):
        scored = [r for r in rows if r["family"] == family and "scores" in r]
        best = max(scored, key=lambda r: selection_key(r["scores"], floor))
        selected[family] = {"config": best["config"], "public": best["scores"], "feasible": bool(selection_key(best["scores"], floor)[0])}

    edits = [f for f in selected if not f.endswith("_fit_guard")]
    stability: dict[str, list[float]] = {}
    for family in edits:
        cfg = config_from(selected[family]["config"])
        stability[family] = [
            lab.retention_detail(lab.lab.tune, lab.solve(dataclasses.replace(cfg, fit_subset_seed=s)).merged)["retention"] for s in range(5)
        ]
        lab.execute(lab.solve(cfg))  # An edit the executor refuses must fail here, before any seal.

    report = json.loads(refit_report.read_text(encoding="utf-8"))
    gradient = {
        "lora_public_guard": {"config": report["lora"]["config"], "guard": "public"},
        "sft_public_guard": {"config": report["sft"]["config"], "guard": "public"},
    }
    for family in ("lora_fit_guard", "sft_fit_guard"):
        gradient[family] = {"config": selected.pop(family)["config"], "guard": "fit"}
    weights_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in gradient.items():
        fit = trained_fit(model, spec, lab)
        torch.save(fit.merged_weights, weights_dir / f"{name}.pt")
        spec["weights_digest"] = fit.weights_digest()
        spec["public"] = lab.score_public(fit.merged_weights)
        print(name, spec["public"], flush=True)

    multi = [f for f in ("parallel", "joint", "sequential", "compensated") if selected[f]["feasible"]]
    rank = lambda f: selection_key(selected[f]["public"], floor)
    doc = {
        "model": model.model_name,
        "sealed_v3_digest": sealed_v3_digest(),
        "corpus_digest": rr._digest((*lab.lab.fit, *lab.lab.tune)),
        "dev_aug_floor": floor,
        "selected": selected,
        "gradient": gradient,
        "prior_public_configs": {
            "reference_strength_only": prior["selected"]["strength_only"]["config"],
            "prior_combination": prior["selected"]["combination"]["config"],
        },
        "stability_tune_retention_80pct_refits": stability,
        "frontier": frontier(rows),
        "configs_tried": len(rows),
        "configs_failed": sum("error" in r for r in rows),
        "comparisons": {
            "comparator": "single",
            "primary_multi_site": max(multi, key=rank) if multi else None,
            "secondary": [f for f in ("parallel", "joint", "sequential", "compensated", "readout", "readout_plus")],
            "test": "two-sided exact McNemar on sealed_v3_retention; Holm across all listed, primary included",
        },
    }
    from .cumulative_edit import freeze_cumulative  # Imported here because that module builds on this one.

    freeze_cumulative(doc, rows, out, floor)
    frozen_path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return doc


def paired_difference_ci(a: Sequence[bool], b: Sequence[bool], draws: int = 10_000, seed: int = 31) -> tuple[float, float]:
    """Percentile bootstrap over prompts for mean(a) - mean(b)."""
    diffs = [int(x) - int(y) for x, y in zip(a, b, strict=True)]
    g = random.Random(seed)
    stats = sorted(sum(g.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(draws))
    return stats[int(0.025 * draws)], stats[int(0.975 * draws) - 1]


def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, p * (len(ordered) - i)))
        adjusted[name] = running
    return adjusted


def run_seal(model: PythiaModel, frozen_path: Path, out: Path, ledger_path: Path, weights_dir: Path, *, force: bool = False) -> dict[str, Any]:  # noqa: PLR0912
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen["sealed_v3_digest"] != sealed_v3_digest():
        raise RuntimeError("The V3 sealed set changed after the freeze")
    lab = MultiLab(model)
    if frozen["corpus_digest"] != rr._digest((*lab.lab.fit, *lab.lab.tune)):
        raise RuntimeError("The corpus changed after the freeze")
    ledger = rr.SealLedger(ledger_path)
    prefixes = GYM.holdout_prefixes
    old_holdout = [(f"{prefixes[i % len(prefixes)]} {e}", t) for i, (e, t) in enumerate(GYM.holdout_rules)]

    methods: dict[str, dict[str, torch.Tensor]] = {"control_no_edit": {}}
    executed: dict[str, MultiCandidate] = {}
    for family, entry in frozen["selected"].items():
        executed[family] = lab.execute(lab.solve(config_from(entry["config"])))
        methods[family] = executed[family].merged
    methods["control_random_single"] = lab.random_like(executed["single"]).merged
    primary = frozen["comparisons"]["primary_multi_site"]
    if primary:
        methods["control_random_primary"] = lab.random_like(executed[primary]).merged
    methods["control_random_readout"] = lab.random_like(executed["readout"]).merged
    for name, config in frozen["prior_public_configs"].items():
        cand = lab.lab.solve(rr.EditConfig(**config))
        methods[name] = {model.mlp_down_proj_site(config["layer"]): cand.weight}
    for name, spec in frozen["gradient"].items():
        weights = torch.load(weights_dir / f"{name}.pt")
        if bl.weights_digest(weights) != spec["weights_digest"]:
            raise RuntimeError(f"{name} weights do not match the freeze")
        methods[name] = weights

    from .cumulative_edit import load_extra_methods  # Imported here because that module builds on this one.

    methods.update(load_extra_methods(lab, frozen, weights_dir))
    rows: dict[str, Any] = {}
    n_stems = len(SEALED_V3_STEMS)
    for name, merged in methods.items():
        for split in SEALED_SPLITS:
            ledger.consult(name, split, "final", force=force)
        detail = lab.retention_detail(SEALED_V3_RETENTION, merged)
        kept = detail["kept"]
        task = lab.task_hits(SEALED_V3_TASK, merged)
        old = lab.task_hits(old_holdout, merged)
        rows[name] = {
            "retention": detail["retention"],
            "retention_ci95": rr.wilson(sum(kept), len(kept)),
            "retention_stems": sum(kept[:n_stems]) / n_stems,
            "retention_genres": sum(kept[n_stems:]) / (len(kept) - n_stems),
            "mean_kl": detail["mean_kl"],
            "task_token_mass": detail["task_token_mass"],
            "task_v3": f"{sum(task)}/{len(task)}",
            "task_v3_ci95": rr.wilson(sum(task), len(task)),
            "task_v3_hits": task,
            "holdout_old_shortcut_prone": f"{sum(old)}/{len(old)}",
            "kept": kept,
            "changed_parameters": changed_parameters(model, merged),
            "delta_norm": delta_norm(model, merged),
        }
        print(name, {k: v for k, v in rows[name].items() if k not in ("kept", "task_v3_hits")}, flush=True)

    comparator = frozen["comparisons"]["comparator"]
    tested = frozen["comparisons"]["secondary"]
    # Cumulative arms face the single-tensor edit selected at the same development bar.
    comparator_of = frozen["comparisons"].get("comparator_of", {})
    comparisons: dict[str, Any] = {}
    for name in tested:
        comparisons[name] = {
            "against": comparator_of.get(name, comparator),
            **rr.mcnemar_exact(rows[name]["kept"], rows[comparator_of.get(name, comparator)]["kept"]),
            "difference": rows[name]["retention"] - rows[comparator_of.get(name, comparator)]["retention"],
            "difference_ci95": paired_difference_ci(rows[name]["kept"], rows[comparator_of.get(name, comparator)]["kept"]),
        }
    adjusted = holm({k: v["p"] for k, v in comparisons.items()})
    for name, value in adjusted.items():
        comparisons[name]["p_holm"] = value
    doc = {"comparator": comparator, "primary": primary, "rows": rows, "comparisons": comparisons, "n_retention": len(SEALED_V3_RETENTION), "n_task": len(SEALED_V3_TASK)}
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return doc


def rss_now() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("No VmRSS")


def f32_bytes(t: torch.Tensor) -> bytes:
    return t.to(torch.float32).contiguous().numpy().tobytes()


def reversibility(model: PythiaModel, original_f32: Mapping[str, bytes], merged: Mapping[str, torch.Tensor], factors: Mapping[str, Any] | None) -> dict[str, Any]:
    """Three ways back to the source bytes. The checkpoint stores f32; working copies are f64.

    `before_image` restores from a saved copy of the changed coordinates. `delta_inverse_f64` subtracts the stored
    delta in f64 and compares the f64 bits. `delta_inverse_rounded_f32` does the same and then rounds to the source
    dtype. Only the factor-based edits have a compact delta; for trained weights the delta is the dense difference.
    """
    exact_f64 = 0
    total = 0
    rounded_ok = True
    image_ok = True
    delta_bytes = 0
    image_bytes = 0
    for site, edited in merged.items():
        base = model.tensors[site]
        if factors is not None and site in factors:
            left, right, rows = factors[site]
            delta = torch.zeros_like(base)
            if rows is None:
                delta = left @ right
            else:
                delta[list(rows)] = left @ right
            delta_bytes += (left.numel() + right.numel()) * 8
        else:
            delta = edited - base
            delta_bytes += int((edited != base).sum()) * 12  # An 8-byte value and a 4-byte index per coordinate.
        restored = edited - delta
        exact_f64 += int((restored == base).sum())
        total += int(base.numel())
        rounded_ok &= f32_bytes(restored) == original_f32[site]
        changed = edited != base
        image = base[changed].to(torch.float32)
        image_bytes += int(changed.sum()) * 8  # A 4-byte source value and a 4-byte index.
        back = edited.clone()
        back[changed] = image.to(torch.float64)
        image_ok &= f32_bytes(back) == original_f32[site]
    return {
        "before_image_bit_exact": image_ok,
        "before_image_bytes": image_bytes,
        "delta_inverse_f64_bit_exact_share": exact_f64 / max(total, 1),
        "delta_inverse_rounded_f32_bit_exact": rounded_ok,
        "delta_bytes": delta_bytes,
    }


def run_cost(checkpoint: str, method: str, frozen_path: Path, out: Path | None) -> dict[str, Any]:
    """One method per process, so the peak belongs to it. Peak includes the loaded model, and wall time is rough."""
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    model = PythiaModel.from_pretrained_dir(checkpoint, model_name=checkpoint)
    original = {k: f32_bytes(v) for k, v in model.tensors.items()}
    loaded_rss = rss_now()
    bl._reset_peak_rss()
    started = time.perf_counter()
    factors: Mapping[str, Any] | None
    plan_bytes: int
    if method in frozen.get("cumulative", {}):
        from .cumulative_edit import replay_graph

        lab = MultiLab(model)
        merged, factors = replay_graph(model, Path(frozen["cumulative"][method]["graph"])), None
        plan_bytes = Path(frozen["cumulative"][method]["graph"]).stat().st_size
        receipts = frozen["cumulative"][method]["steps"]
    elif method in frozen["gradient"]:
        spec = frozen["gradient"][method]
        lab = MultiLab(model)
        fit = trained_fit(model, spec, lab)
        merged = fit.merged_weights
        factors = None
        cfg = spec["config"]
        if cfg["kind"] == "lora":
            plan_bytes = fit.cost.trainable_parameters * 8  # The adapter's factors.
        else:
            plan_bytes = fit.cost.changed_parameters * 12  # Value and index per changed coordinate.
        receipts = 0
    else:
        lab = MultiLab(model)
        cand = lab.execute(lab.solve(config_from(frozen["selected"][method]["config"])))
        merged = cand.merged
        factors = cand.factors
        plan_bytes = sum((left.numel() + right.numel()) * 8 for left, right, _ in cand.factors.values())
        receipts = len(cand.digests)
    wall = time.perf_counter() - started
    peak = bl._peak_rss_bytes()
    result = {
        "checkpoint": checkpoint,
        "method": method,
        "wall_seconds_rough": wall,
        "peak_rss_mib": peak / 2**20 if peak else None,
        "rss_after_load_mib": loaded_rss / 2**20,
        "changed_parameters": changed_parameters(model, merged),
        "total_parameters": model.total_parameters,
        "share_changed": changed_parameters(model, merged) / model.total_parameters,
        "delta_norm": delta_norm(model, merged),
        "plan_payload_bytes": plan_bytes,
        "executor_receipts": receipts,
        "reversibility": reversibility(model, original, merged, factors),
        "dev": sum(lab.task_hits(lab.lab.dev_items, merged)) / len(lab.lab.dev_items),
        "val": sum(lab.task_hits(lab.lab.val_items, merged)) / len(lab.lab.val_items),
        "max_rss_getrusage_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    print(json.dumps(result), flush=True)
    if out is not None:
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
    return result


def weights_sha(weights: Mapping[str, torch.Tensor]) -> str:
    return hashlib.sha256(bl.weights_digest(weights).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=["tune", "freeze", "seal", "cost"])
    parser.add_argument("--checkpoint", default="/tmp/pythia-14m")
    parser.add_argument("--families", default="")
    parser.add_argument("--method", default="")
    parser.add_argument("--dir", default="docs/retention-multisite")
    parser.add_argument("--weights-dir", default="/tmp/nf-ms-weights")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--force-reconsult", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    root = Path(args.dir)
    frozen = root / "frozen.json"
    if args.phase == "cost":
        run_cost(args.checkpoint, args.method, frozen, root / "costs.jsonl")
        return 0
    model = PythiaModel.from_pretrained_dir(args.checkpoint, model_name="EleutherAI/pythia-14m")
    if args.phase == "tune":
        run_tune(model, args.families.split(","), root)
    elif args.phase == "freeze":
        run_freeze(model, root, frozen, Path(args.weights_dir), Path("docs/real_baselines/refit_report.json"), Path("docs/retention-research-tuning.json"))
    else:
        run_seal(model, frozen, root / "sealed.json", root / "ledger.json", Path(args.weights_dir), force=args.force_reconsult)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
