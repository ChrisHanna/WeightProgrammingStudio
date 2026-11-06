"""Control arms for the proposer's side: development-only tuning, fits and placebo edits.

This module runs before the manifest freezes and has no path to sealed data: it never imports the
custodian, it takes development items only, and holdout seeds don't exist until after the freeze.
The tuning record goes into the manifest, and the custodian refuses a record that used a
non-development prefix or an operand not declared as exposed. A record is still self-reported:
nothing can prove from weights that an arm came from the recipe it names.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import torch

from .. import baselines as bl
from .generators import RETENTION_TASK, Cluster, Item, check_clusters, get_item_generator, render
from .plan import StudyPlan

RANK_TOLERANCE = 1e-9
_VALIDATION_SEED_OFFSET = 1_000_003
_RETENTION_SEED_OFFSET = 2_000_003

TrialRunner = Callable[[Any, bl.BaselineConfig, "DevelopmentData"], tuple[list[bl.CheckpointScore], bl.BaselineFit]]


class TuningLeak(ValueError):
    """Data offered to tuning isn't development data under this plan."""


@dataclass(frozen=True)
class DevelopmentData:
    train: tuple[Cluster, ...]
    validation: tuple[Cluster, ...]  # Other operands than train, as the holdout will have.
    guard: tuple[Item, ...]
    retention_tune: tuple[Item, ...]
    train_prefix: str
    validation_prefix: str

    @property
    def clusters(self) -> tuple[Cluster, ...]:
        return (*self.train, *self.validation)

    @property
    def retention_items(self) -> tuple[Item, ...]:
        return (*self.guard, *self.retention_tune)

    @property
    def validation_items(self) -> list[Item]:
        return render(self.validation, (self.validation_prefix,))

    @property
    def splits(self) -> bl.BaselineSplits:
        return bl.BaselineSplits(
            dev=tuple((item.prompt, item.target) for item in render(self.train, (self.train_prefix,))),
            val=tuple((item.prompt, item.target) for item in self.validation_items),
            retention_guard=tuple(item.prompt for item in self.guard),
            retention_tune=tuple(item.prompt for item in self.retention_tune),
        )


def development_data(plan: StudyPlan, seed: int, *, rules_per_task: int, retention_prompts: int) -> DevelopmentData:
    """Draw public items through the plan's generator, using development prefixes only."""
    generator = get_item_generator(plan.item_generator)
    train: list[Cluster] = []
    validation: list[Cluster] = []
    for task in plan.tasks:
        for pool, draw_seed, tag in ((train, seed, "dev"), (validation, seed + _VALIDATION_SEED_OFFSET, "val")):
            exclude = {operand for cluster in train if cluster.task == task for operand in cluster.operands}
            clusters = generator.draw_clusters(
                task, draw_seed, rules_per_task, domain=plan.number_domain, exclude=exclude, tag=tag
            )
            check_clusters(clusters, task, rules_per_task, exclude, tag)
            pool += clusters
    retention = generator.draw_retention(seed + _RETENTION_SEED_OFFSET, retention_prompts, exclude=(), tag="public")
    return DevelopmentData(
        train=tuple(train),
        validation=tuple(validation),
        guard=tuple(retention[0::2]),
        retention_tune=tuple(retention[1::2]),
        train_prefix=plan.development_prefixes[0],
        validation_prefix=plan.development_prefixes[-1],
    )


def _digest(values: Iterable[Any]) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(list(values), sort_keys=True).encode("utf-8")).hexdigest()


def assert_development_only(plan: StudyPlan, data: DevelopmentData) -> None:
    if {data.train_prefix, data.validation_prefix} - set(plan.development_prefixes):
        raise TuningLeak("tuning may only use the plan's development prefixes")
    for prompt in (*data.splits.dev, *data.splits.val):
        if any(prompt[0].startswith(f"{prefix} ") for prefix in plan.holdout_prefixes):
            raise TuningLeak("a tuning prompt carries a holdout prefix")


def forced_choice_accuracy(model: Any, items: Sequence[Item]) -> float:
    """The study's primary metric, so that controls are selected for what they will be scored on."""
    vocab = model.tokenizer.vocab
    hits = 0
    with torch.no_grad():
        for item in items:
            logits = model.forward_with_mlp_key(model.tokenizer.encode(item.prompt))[0][-1]
            hits += bool(logits[vocab[item.target]] > logits[vocab[item.alternative]])
    return hits / max(len(items), 1)


def run_trial(
    model: Any, config: bl.BaselineConfig, data: DevelopmentData
) -> tuple[list[bl.CheckpointScore], bl.BaselineFit]:
    """One grid point. `val_accuracy` holds forced-choice accuracy on validation operands."""
    splits = data.splits
    train_items = render(data.train, (data.train_prefix,))
    validation_items = data.validation_items
    scores: list[bl.CheckpointScore] = []

    def record(step: int, weights: dict[str, torch.Tensor], wall: float) -> None:
        score = bl.score_checkpoint(model, weights, splits, steps=step, wall_seconds=wall)
        edited = bl.merged_model(model, weights)
        scores.append(
            replace(
                score,
                val_accuracy=forced_choice_accuracy(edited, validation_items),
                dev_accuracy=forced_choice_accuracy(edited, train_items),
            )
        )

    fit = bl.fit_baseline(model, config, dev=splits.dev, retention_guard=splits.retention_guard, on_checkpoint=record)
    return scores, fit


def tune_arm(
    model: Any,
    plan: StudyPlan,
    grid: Sequence[bl.BaselineConfig],
    data: DevelopmentData,
    *,
    trial: TrialRunner = run_trial,
) -> tuple[bl.SelectedConfig, dict[str, Any]]:
    """Grid-search on development data, returning the selection and the record the manifest must carry."""
    assert_development_only(plan, data)
    trials = []
    rows = []
    for config in grid:
        scores, fit = trial(model, config, data)
        trials.append((config, scores))
        rows.append(
            {"config": asdict(config), "scores": [asdict(score) for score in scores], "cost": asdict(fit.cost)}
        )
    selected = bl.select_best(trials)
    operands: dict[str, set[str]] = {}
    for cluster in data.clusters:
        operands.setdefault(cluster.task, set()).update(cluster.operands)
    record = {
        "method": "development_grid_search",
        "selection": "forced-choice accuracy on validation operands + top-1 retention on the tuning fold",
        "trials": rows,
        "selected": {"config": asdict(selected.config), "score": asdict(selected.score)},
        "data": {
            "prefixes": sorted({data.train_prefix, data.validation_prefix}),
            "operands": {task: sorted(values) for task, values in operands.items()},
            "retention_prompts": sorted(item.prompt for item in data.retention_items),
            "train_digest": _digest(data.splits.dev),
            "validation_digest": _digest(data.splits.val),
        },
    }
    return selected, record


def fit_arm(model: Any, selected: bl.SelectedConfig, seed: int, data: DevelopmentData) -> bl.BaselineFit:
    """Fit the selected recipe for one synthesis seed, on the development data that seed's candidate got."""
    splits = data.splits
    return bl.fit_baseline(model, replace(selected.config, seed=seed), dev=splits.dev, retention_guard=splits.refit_guard)


def edit_profile(base: Mapping[str, torch.Tensor], weights: Mapping[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    """Frobenius norm and numerical rank of each tensor's change from base."""
    profile: dict[str, dict[str, float]] = {}
    for name, tensor in weights.items():
        delta = tensor - base[name]
        norm = float(torch.linalg.vector_norm(delta))
        if delta.ndim == 2:
            values = torch.linalg.svdvals(delta)
            rank = int((values > values[0] * RANK_TOLERANCE).sum()) if float(values[0]) > 0 else 0
        else:
            rank = int(norm > 0)
        profile[name] = {"norm": norm, "rank": rank}
    return profile


def placebo_weights(
    base: Mapping[str, torch.Tensor], candidate: Mapping[str, torch.Tensor], seed: int
) -> dict[str, torch.Tensor]:
    """A random edit with the candidate's singular values, so its norm and rank match, in random directions.

    It sees no data of any kind. An unchanged tensor has no random equivalent, so an identity candidate raises.
    """
    generator = torch.Generator().manual_seed(seed)
    placebo: dict[str, torch.Tensor] = {}
    for name in sorted(candidate):
        delta = candidate[name] - base[name]
        if delta.ndim != 2:
            raise ValueError(f"a placebo needs a matrix edit, and {name!r} isn't one")
        values = torch.linalg.svdvals(delta)
        rank = int((values > values[0] * RANK_TOLERANCE).sum()) if float(values[0]) > 0 else 0
        if rank == 0:
            raise ValueError(f"{name!r} is unchanged, so it has no placebo")
        left, _ = torch.linalg.qr(torch.randn(delta.shape[0], rank, dtype=torch.float64, generator=generator))
        right, _ = torch.linalg.qr(torch.randn(delta.shape[1], rank, dtype=torch.float64, generator=generator))
        placebo[name] = base[name] + (left * values[:rank]) @ right.T
    return placebo


def exposed_retention(data: Iterable[DevelopmentData]) -> dict[str, list[str]]:
    prompts = {item.prompt for each in data for item in each.retention_items}
    return {RETENTION_TASK: sorted(prompts)}
