"""Gradient baselines that train the real model through its real forward pass.

`m4_ecosystems.run_lora_baseline` and `run_sft_baseline` fit one matrix to precomputed inputs, so they
cannot say whether gradient training beats a direct weight edit. Here, the model's own forward code runs
with differentiable tensors and every result is a set of merged weights, which `merged_model` loads into
an ordinary model. `evaluate_prompt` then scores it exactly as it scores a direct edit.

Data discipline:
- Training uses the gym's development items, plus a retention guard: KL to the original model on general
  prompts, which is the same protection direct edits get from their retention constraint.
- Tuning scores validation items and a fold of practice retention prompts that the guard did not train on.
- Sealed data is reachable only through `SealedHoldout`, which answers once per candidate.
"""

from __future__ import annotations

import copy
import hashlib
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F

BaselineKind = Literal["lora", "sft"]
LoraTargets = Literal["final_down_proj", "all_down_proj", "all_linear"]
Item = tuple[str, str]


@dataclass(frozen=True)
class BaselineConfig:
    """One training recipe. `steps` is the longest run; tuning also scores `checkpoints` along the way."""

    kind: BaselineKind
    lr: float
    steps: int
    retention_weight: float = 0.0
    targets: LoraTargets | Literal["all"] = "all"
    rank: int = 0
    seed: int = 0
    guard_batch: int = 0  # Guard prompts sampled per step; 0 uses all of them.
    checkpoints: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.kind == "lora" and (self.rank < 1 or self.targets == "all"):
            raise ValueError("LoRA needs rank >= 1 and named target sites")
        if self.kind == "sft" and self.targets != "all":
            raise ValueError("full fine-tuning trains every parameter")
        if self.lr <= 0 or self.steps < 1:
            raise ValueError("lr must be positive and steps at least 1")


@dataclass(frozen=True)
class BaselineSplits:
    dev: tuple[Item, ...]
    val: tuple[Item, ...]
    retention_guard: tuple[str, ...]  # Trained on during tuning.
    retention_tune: tuple[str, ...]  # Scored during tuning, never trained on until the final refit.

    @property
    def refit_guard(self) -> tuple[str, ...]:
        """Every public retention prompt, which is what a direct edit is allowed to protect."""
        return (*self.retention_guard, *self.retention_tune)


@dataclass(frozen=True)
class FitCost:
    optimizer_steps: int
    wall_seconds: float  # Training only. Checkpoint scoring is excluded.
    trainable_parameters: int
    changed_parameters: int
    total_model_parameters: int
    peak_rss_bytes: int | None  # Whole process, so it includes the loaded model.
    delta_l2_norm: float


@dataclass(frozen=True)
class BaselineFit:
    config: BaselineConfig
    merged_weights: dict[str, torch.Tensor]  # Only the tensors that training changed.
    cost: FitCost
    final_train_loss: float

    def weights_digest(self) -> str:
        return weights_digest(self.merged_weights)


@dataclass(frozen=True)
class CheckpointScore:
    steps: int
    val_accuracy: float
    val_loss: float
    retention_tune_accuracy: float
    dev_accuracy: float
    wall_seconds: float

    @property
    def selection_score(self) -> float:
        return self.val_accuracy + self.retention_tune_accuracy


@dataclass(frozen=True)
class SelectedConfig:
    config: BaselineConfig  # `steps` is the selected checkpoint.
    score: CheckpointScore


def build_splits(gym: Any, practice_retention: Sequence[str], *, model: Any | None = None) -> BaselineSplits:
    """Builds public splits the way the agent loop does and rejects any overlap with the gym's holdout."""
    dev = tuple((f"{p} {e}", t) for p in gym.gym_dev_prefixes for e, t in gym.base_rules)
    val = tuple((f"{p} {e}", t) for p in gym.gym_val_prefixes for e, t in gym.base_rules)
    prompts = tuple(dict.fromkeys(p for p in practice_retention if model is None or model.tokenizer.encode(p)))
    gym_retention = tuple(p for p in gym.retention_prompts if model is None or model.tokenizer.encode(p))
    splits = BaselineSplits(
        dev=dev,
        val=val,
        retention_guard=tuple(dict.fromkeys((*gym_retention, *prompts[0::2]))),
        retention_tune=tuple(p for p in prompts[1::2] if p not in gym_retention),
    )
    public = [p for p, _ in (*dev, *val)] + list(splits.refit_guard)
    for expr, _ in gym.holdout_rules:
        if any(expr in text for text in public):
            raise ValueError(f"holdout expression leaked into public data: {expr!r}")
    return splits


def merged_model(model: Any, merged_weights: Mapping[str, torch.Tensor]) -> Any:
    """Returns a model sharing untouched tensors with `model`, with merged weights in place."""
    unknown = set(merged_weights) - set(model.tensors)
    if unknown:
        raise KeyError(f"unknown tensors: {sorted(unknown)}")
    out = copy.copy(model)
    out.tensors = {**model.tensors, **{k: v.detach() for k, v in merged_weights.items()}}
    return out


def weights_digest(weights: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name in sorted(weights):
        h.update(name.encode())
        h.update(weights[name].detach().to(torch.float64).contiguous().numpy().tobytes())
    return h.hexdigest()


def lora_sites(model: Any, targets: str) -> tuple[str, ...]:
    layers = int(model.config["num_hidden_layers"])
    if targets == "final_down_proj":
        return (model.mlp_down_proj_site(layers - 1),)
    if targets == "all_down_proj":
        return tuple(model.mlp_down_proj_site(i) for i in range(layers))
    if targets == "all_linear":
        return tuple(k for k, v in model.tensors.items() if ".layers." in k and v.ndim == 2)
    raise ValueError(f"unknown LoRA targets: {targets}")


def _last_logits(model: Any, prompt: str) -> torch.Tensor:
    logits, _ = model.forward_with_mlp_key(model.tokenizer.encode(prompt))
    return logits[-1]


def accuracy(model: Any, items: Iterable[Item]) -> float:
    items = tuple(items)
    with torch.no_grad():
        hits = sum(bool(model.evaluate_prompt(p, target_token=t).matched_target) for p, t in items)
    return hits / max(len(items), 1)


def retention(base: Any, candidate: Any, prompts: Iterable[str]) -> float:
    """Share of prompts whose top-1 token is unchanged, the definition direct edits are scored by."""
    prompts = tuple(prompts)
    with torch.no_grad():
        kept = sum(
            base.evaluate_prompt(p).top1_token_id == candidate.evaluate_prompt(p).top1_token_id for p in prompts
        )
    return kept / max(len(prompts), 1)


def _mean_ce(model: Any, items: Sequence[Item]) -> torch.Tensor:
    losses = [
        F.cross_entropy(
            _last_logits(model, p).unsqueeze(0),
            torch.tensor([model.tokenizer.vocab[t]], dtype=torch.long),
        )
        for p, t in items
    ]
    return torch.stack(losses).mean()


def _reset_peak_rss() -> None:
    try:
        Path("/proc/self/clear_refs").write_text("5")  # Linux: restart the VmHWM high-water mark.
    except OSError:
        pass


def _peak_rss_bytes() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None  # Not Linux: no comparable per-run peak is available.


def fit_baseline(
    model: Any,
    config: BaselineConfig,
    *,
    dev: Sequence[Item],
    retention_guard: Sequence[str] = (),
    on_checkpoint: Callable[[int, dict[str, torch.Tensor], float], None] | None = None,
) -> BaselineFit:
    """Trains on `dev` with Adam at a constant rate, so any checkpoint equals a shorter run of the same seed."""
    generator = torch.Generator().manual_seed(config.seed)
    guard = tuple(retention_guard)
    with torch.no_grad():
        guard_ref = {p: F.log_softmax(_last_logits(model, p), dim=-1) for p in guard}

    base = model.tensors
    params: list[torch.Tensor]
    if config.kind == "sft":
        live = {k: v.clone().requires_grad_(True) for k, v in base.items()}
        params = list(live.values())

        def current() -> dict[str, torch.Tensor]:
            return live

        trainable = sum(int(v.numel()) for v in params)
    else:
        factors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for site in lora_sites(model, str(config.targets)):
            out_dim, in_dim = base[site].shape
            rank = min(config.rank, int(out_dim), int(in_dim))
            # The usual LoRA start: B is zero, so the first forward pass matches the original model.
            a = torch.randn(rank, int(in_dim), dtype=torch.float64, generator=generator) / (int(in_dim) ** 0.5)
            b = torch.zeros(int(out_dim), rank, dtype=torch.float64)
            factors[site] = (a.requires_grad_(True), b.requires_grad_(True))
        params = [t for pair in factors.values() for t in pair]

        def current() -> dict[str, torch.Tensor]:
            return {**base, **{site: base[site] + b @ a for site, (a, b) in factors.items()}}

        trainable = sum(int(t.numel()) for t in params)

    optimizer = torch.optim.Adam(params, lr=config.lr)
    trained = copy.copy(model)
    checkpoints = set(config.checkpoints) | {config.steps}
    _reset_peak_rss()
    started = time.perf_counter()
    excluded = 0.0
    loss_value = float("nan")

    def snapshot() -> dict[str, torch.Tensor]:
        with torch.no_grad():
            state = current()
            if config.kind == "lora":
                return {k: state[k].detach().clone() for k in factors}
            return {k: v.detach().clone() for k, v in state.items()}

    for step in range(1, config.steps + 1):
        trained.tensors = current()
        loss = _mean_ce(trained, dev)
        if guard and config.retention_weight > 0:
            batch = guard
            if 0 < config.guard_batch < len(guard):
                picks = torch.randperm(len(guard), generator=generator)[: config.guard_batch]
                batch = tuple(guard[int(i)] for i in picks)
            kl = torch.stack(
                [
                    F.kl_div(F.log_softmax(_last_logits(trained, p), dim=-1), guard_ref[p], log_target=True, reduction="sum")
                    for p in batch
                ]
            ).mean()
            loss = loss + config.retention_weight * kl
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.item())
        if on_checkpoint is not None and step in checkpoints:
            paused = time.perf_counter()
            on_checkpoint(step, snapshot(), paused - started - excluded)
            excluded += time.perf_counter() - paused

    wall = time.perf_counter() - started - excluded
    peak = _peak_rss_bytes()
    state = snapshot()
    changed = {k: v for k, v in state.items() if not torch.equal(v, base[k])}
    changed_count = sum(int((v != base[k]).sum().item()) for k, v in changed.items())
    delta_sq = sum(float(((v - base[k]) ** 2).sum().item()) for k, v in changed.items())
    return BaselineFit(
        config=config,
        merged_weights=changed,
        cost=FitCost(
            optimizer_steps=config.steps,
            wall_seconds=wall,
            trainable_parameters=trainable,
            changed_parameters=changed_count,
            total_model_parameters=int(model.total_parameters),
            peak_rss_bytes=peak,
            delta_l2_norm=delta_sq**0.5,
        ),
        final_train_loss=loss_value,
    )


def score_checkpoint(
    base: Any, weights: Mapping[str, torch.Tensor], splits: BaselineSplits, *, steps: int, wall_seconds: float
) -> CheckpointScore:
    candidate = merged_model(base, weights)
    with torch.no_grad():
        val_loss = float(_mean_ce(candidate, splits.val).item())
    return CheckpointScore(
        steps=steps,
        val_accuracy=accuracy(candidate, splits.val),
        val_loss=val_loss,
        retention_tune_accuracy=retention(base, candidate, splits.retention_tune),
        dev_accuracy=accuracy(candidate, splits.dev),
        wall_seconds=wall_seconds,
    )


def tune_trial(model: Any, config: BaselineConfig, splits: BaselineSplits) -> tuple[list[CheckpointScore], BaselineFit]:
    """Runs one config, scoring public splits at each checkpoint. It has no path to sealed data."""
    scores: list[CheckpointScore] = []

    def record(step: int, weights: dict[str, torch.Tensor], wall: float) -> None:
        scores.append(score_checkpoint(model, weights, splits, steps=step, wall_seconds=wall))

    fit = fit_baseline(model, config, dev=splits.dev, retention_guard=splits.retention_guard, on_checkpoint=record)
    return scores, fit


def select_best(trials: Iterable[tuple[BaselineConfig, Iterable[CheckpointScore]]]) -> SelectedConfig:
    """Highest validation accuracy + tuning-fold retention; ties go to lower validation loss, then fewer steps."""
    best: tuple[tuple[float, float, int], SelectedConfig] | None = None
    for config, scores in trials:
        for s in scores:
            key = (-s.selection_score, s.val_loss, s.steps)
            if best is None or key < best[0]:
                best = (key, SelectedConfig(replace(config, steps=s.steps, checkpoints=()), s))
    if best is None:
        raise ValueError("no trials to select from")
    return best[1]


def refit_selected(model: Any, selected: SelectedConfig, splits: BaselineSplits) -> BaselineFit:
    """Retrains the selected recipe with every public retention prompt in the guard, as direct edits may use."""
    return fit_baseline(model, selected.config, dev=splits.dev, retention_guard=splits.refit_guard)


def public_report(base: Any, fit: BaselineFit, splits: BaselineSplits, practice: Sequence[str]) -> dict[str, Any]:
    candidate = merged_model(base, fit.merged_weights)
    return {
        "config": asdict(fit.config),
        "cost": asdict(fit.cost),
        "weights_digest": fit.weights_digest(),
        "dev_accuracy": accuracy(candidate, splits.dev),
        "val_accuracy": accuracy(candidate, splits.val),
        "practice_retention": retention(base, candidate, practice),
    }


class SealedHoldout:
    """Scores each named candidate once. A second request for a name raises instead of answering."""

    def __init__(self, gym: Any, sealed_retention: Sequence[str]) -> None:
        prefixes = gym.holdout_prefixes
        self._items = tuple(
            (f"{prefixes[i % len(prefixes)]} {expr}", target) for i, (expr, target) in enumerate(gym.holdout_rules)
        )
        self._retention = tuple(sealed_retention)
        self._used: set[str] = set()

    def evaluate_once(self, name: str, base: Any, merged_weights: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        if name in self._used:
            raise RuntimeError(f"holdout already used for {name!r}")
        self._used.add(name)
        candidate = merged_model(base, merged_weights)
        prompts = tuple(p for p in self._retention if base.tokenizer.encode(p))
        hits = accuracy(candidate, self._items)
        return {
            "holdout_accuracy": hits,
            "holdout_correct": round(hits * len(self._items)),
            "holdout_total": len(self._items),
            "sealed_retention": retention(base, candidate, prompts),
            "sealed_retention_total": len(prompts),
        }
