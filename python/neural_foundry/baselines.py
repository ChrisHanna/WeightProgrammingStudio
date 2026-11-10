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

from .gym_data import materialize_public_task_data

BaselineKind = Literal["lora", "sft"]
LoraTargets = Literal["final_down_proj", "all_down_proj", "all_linear"]
Item = tuple[str, str]


@dataclass(frozen=True)
class BaselineConfig:
    """One training recipe. `steps` is the longest run; tuning also scores `checkpoints` along the way."""

    kind: BaselineKind
    lr: float
    steps: int
    recipe_schema_version: int = 1
    retention_weight: float = 0.0
    targets: LoraTargets | Literal["all"] = "all"
    rank: int = 0
    seed: int = 0
    guard_batch: int = 0  # Guard prompts sampled per step; 0 uses all of them.
    checkpoints: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.recipe_schema_version != 1:
            raise ValueError("unsupported baseline recipe schema_version")
        if self.kind == "lora" and (self.rank < 1 or self.targets == "all"):
            raise ValueError("LoRA needs rank >= 1 and named target sites")
        if self.kind == "sft" and self.targets != "all":
            raise ValueError("full fine-tuning trains every parameter")
        if self.lr <= 0 or self.steps < 1:
            raise ValueError("lr must be positive and steps at least 1")


@dataclass(frozen=True)
class DPOConfig:
    learning_rate: float = 1e-4
    steps: int = 100
    rank: int = 4
    targets: LoraTargets = "all_down_proj"
    beta: float = 0.1
    retention_weight: float = 0.1
    seed: int = 0
    recipe_schema_version: int = 1

    def __post_init__(self) -> None:
        if self.recipe_schema_version != 1:
            raise ValueError("unsupported DPO recipe schema_version")
        if self.learning_rate <= 0 or self.steps < 1 or self.rank < 1 or self.beta <= 0:
            raise ValueError("DPO learning_rate, steps, rank, and beta must be positive")
        if self.retention_weight < 0:
            raise ValueError("DPO retention_weight must be non-negative")


@dataclass(frozen=True)
class PreferenceFit:
    config: DPOConfig
    merged_weights: dict[str, torch.Tensor]
    cost: FitCost
    final_train_loss: float

    def weights_digest(self) -> str:
        return weights_digest(self.merged_weights)


@dataclass(frozen=True)
class BaselineSplits:
    dev: tuple[Item, ...]
    val: tuple[Item, ...]
    retention_guard: tuple[str, ...]  # Trained on during tuning.
    retention_tune: tuple[str, ...]  # Scored during tuning, never trained on until the final refit.
    gym_id: str
    gym_digest: str
    dataset_digest: str

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
    conversion_drift_l2_norm: float = 0.0


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
    """Materializes the shared Gym task panels and rejects overlap with the Gym holdout."""
    data = materialize_public_task_data(gym, tokenizer=model.tokenizer if model is not None else None)
    dev = data.train_items
    val = data.validation_items
    prompts = tuple(dict.fromkeys(p for p in practice_retention if model is None or model.tokenizer.encode(p)))
    gym_retention = tuple(p for p in gym.retention_prompts if model is None or model.tokenizer.encode(p))
    splits = BaselineSplits(
        dev=dev,
        val=val,
        retention_guard=tuple(dict.fromkeys((*gym_retention, *prompts[0::2]))),
        retention_tune=tuple(p for p in prompts[1::2] if p not in gym_retention),
        gym_id=data.gym_id,
        gym_digest=data.gym_digest,
        dataset_digest=data.dataset_digest,
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
    if any(not torch.equal(model.tensors[name], value) for name, value in merged_weights.items()):
        out.checkpoint_identity = None
        out._checkpoint_state_digest = None
    return out


def _training_update_snapshot(
    original: Mapping[str, torch.Tensor], training_base: Mapping[str, torch.Tensor], live: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    return {
        name: original[name].detach().to(device="cpu", dtype=torch.float64)
        + (value.detach().to(device="cpu", dtype=torch.float64) - training_base[name].detach().to(device="cpu", dtype=torch.float64))
        for name, value in live.items()
    }


def _conversion_drift(original: Mapping[str, torch.Tensor], training_base: Mapping[str, torch.Tensor]) -> float:
    return sum(float((
        training_base[name].detach().to(device="cpu", dtype=torch.float64)
        - value.detach().to(device="cpu", dtype=torch.float64)
    ).square().sum()) for name, value in original.items()) ** 0.5


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
    losses: list[torch.Tensor] = []
    for prompt, target in items:
        logits = _last_logits(model, prompt)
        target_id = torch.tensor(
            [model.tokenizer.vocab[target]], dtype=torch.long, device=logits.device
        )
        losses.append(F.cross_entropy(logits.unsqueeze(0), target_id))
    return torch.stack(losses).mean()


def _sequence_ce(model: Any, examples: Sequence[tuple[str, str]]) -> torch.Tensor:
    """Teacher-forced next-token loss on response tokens only."""
    losses: list[torch.Tensor] = []
    for prompt, response in examples:
        prompt_ids = model.tokenizer.encode(prompt)
        joined_ids = model.tokenizer.encode(prompt + response)
        if not prompt_ids or len(joined_ids) <= len(prompt_ids):
            raise ValueError("Sequence SFT examples need encodable prompts and responses")
        if joined_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("Prompt/response boundary changes tokenization; provide an explicit separator")
        response_ids = joined_ids[len(prompt_ids) :]
        if len(prompt_ids) + len(response_ids) > 16_384:
            raise ValueError("Sequence SFT example exceeds the 16384-token bound")
        logits, _ = model.forward_with_mlp_key((*prompt_ids, *response_ids))
        first_prediction = len(prompt_ids) - 1
        prediction_logits = logits[first_prediction : first_prediction + len(response_ids)]
        if prediction_logits.shape[0] != len(response_ids):
            raise ValueError("Model forward did not return logits for every response token")
        targets = torch.tensor(response_ids, dtype=torch.long, device=logits.device)
        losses.append(F.cross_entropy(prediction_logits, targets))
    if not losses:
        raise ValueError("Sequence SFT requires at least one example")
    return torch.stack(losses).mean()


def fit_sequence_baseline(
    model: Any,
    config: BaselineConfig,
    *,
    examples: Sequence[tuple[str, str]],
    retention_guard: Sequence[str] = (),
    on_checkpoint: Callable[[int, dict[str, torch.Tensor], float], None] | None = None,
    training_device: str | torch.device = "cpu",
    training_dtype: torch.dtype | None = None,
) -> BaselineFit:
    """LoRA/SFT training over complete prompt/response trajectories from a verified rollout panel."""
    if not examples:
        raise ValueError("Sequence SFT requires at least one example")
    device = torch.device(training_device)
    if device.type == "cuda":
        from .gpu import probe_gpu_hardware

        inventory = probe_gpu_hardware()
        if not inventory.cuda_executable:
            raise RuntimeError(f"CUDA training requested but unavailable: {inventory.kernel_probe_error}")
    dtype = training_dtype or (torch.float32 if device.type == "cuda" else torch.float64)
    training_model = copy.copy(model)
    training_model.tensors = {
        name: tensor.detach().to(device=device, dtype=dtype) for name, tensor in model.tensors.items()
    }
    base = training_model.tensors
    canonical_base = model.tensors
    generator = torch.Generator(device=device).manual_seed(config.seed)
    guard = tuple(retention_guard)
    with torch.no_grad():
        guard_ref = {prompt: F.log_softmax(_last_logits(training_model, prompt), dim=-1) for prompt in guard}

    factors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    if config.kind == "sft":
        live = {name: tensor.clone().requires_grad_(True) for name, tensor in base.items()}
        params = list(live.values())

        def current() -> dict[str, torch.Tensor]:
            return live

    else:
        for site in lora_sites(model, str(config.targets)):
            out_dim, in_dim = base[site].shape
            rank = min(config.rank, int(out_dim), int(in_dim))
            left = torch.randn(
                rank, int(in_dim), dtype=dtype, device=device, generator=generator
            ) / int(in_dim) ** 0.5
            right = torch.zeros(int(out_dim), rank, dtype=dtype, device=device)
            factors[site] = (left.requires_grad_(True), right.requires_grad_(True))
        params = [tensor for pair in factors.values() for tensor in pair]

        def current() -> dict[str, torch.Tensor]:
            return {**base, **{site: base[site] + right @ left for site, (left, right) in factors.items()}}

    optimizer = torch.optim.Adam(params, lr=config.lr)
    checkpoints = set(config.checkpoints) | {config.steps}
    _reset_peak_rss()
    started = time.perf_counter()
    excluded = 0.0
    loss_value = float("nan")

    def snapshot() -> dict[str, torch.Tensor]:
        with torch.no_grad():
            state = current()
            selected = {name: state[name] for name in factors} if config.kind == "lora" else state
            return _training_update_snapshot(canonical_base, base, selected)

    for step in range(1, config.steps + 1):
        trained = copy.copy(training_model)
        trained.tensors = current()
        loss = _sequence_ce(trained, examples)
        if guard and config.retention_weight > 0:
            batch = guard
            if 0 < config.guard_batch < len(guard):
                picks = torch.randperm(len(guard), generator=generator, device=device)[: config.guard_batch]
                batch = tuple(guard[int(index)] for index in picks)
            kl = torch.stack(
                [
                    F.kl_div(
                        F.log_softmax(_last_logits(trained, prompt), dim=-1),
                        guard_ref[prompt],
                        log_target=True,
                        reduction="sum",
                    )
                    for prompt in batch
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
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    peak = _peak_rss_bytes()
    state = snapshot()
    changed = {name: value for name, value in state.items() if not torch.equal(value, canonical_base[name])}
    changed_count = sum(int((value != canonical_base[name]).sum().item()) for name, value in changed.items())
    delta_sq = sum(float(((value - canonical_base[name]) ** 2).sum().item()) for name, value in changed.items())
    return BaselineFit(
        config=config,
        merged_weights=changed,
        cost=FitCost(
            optimizer_steps=config.steps,
            wall_seconds=wall,
            trainable_parameters=sum(int(parameter.numel()) for parameter in params),
            changed_parameters=changed_count,
            total_model_parameters=int(model.total_parameters),
            peak_rss_bytes=peak,
            delta_l2_norm=delta_sq**0.5,
            conversion_drift_l2_norm=_conversion_drift(canonical_base, base),
        ),
        final_train_loss=loss_value,
    )


def _response_log_probability(model: Any, prompt: str, response: str) -> torch.Tensor:
    prompt_ids = model.tokenizer.encode(prompt)
    joined_ids = model.tokenizer.encode(prompt + response)
    if not prompt_ids or len(joined_ids) <= len(prompt_ids):
        raise ValueError("Preference examples need encodable prompts and responses")
    if joined_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Prompt/response boundary changes tokenization")
    response_ids = joined_ids[len(prompt_ids) :]
    if len(joined_ids) > 16_384:
        raise ValueError("Preference example exceeds the 16384-token bound")
    logits, _ = model.forward_with_mlp_key(joined_ids)
    first_prediction = len(prompt_ids) - 1
    response_logits = logits[first_prediction : first_prediction + len(response_ids)]
    targets = torch.tensor(response_ids, dtype=torch.long, device=logits.device)
    return F.log_softmax(response_logits, dim=-1).gather(1, targets[:, None]).sum()


def fit_preference_dpo(
    model: Any,
    config: DPOConfig,
    *,
    pairs: Sequence[tuple[str, str, str]],
    retention_guard: Sequence[str] = (),
    training_device: str | torch.device = "cpu",
    training_dtype: torch.dtype | None = None,
) -> PreferenceFit:
    """Fit LoRA with offline DPO pairs; the base model is the fixed reference policy."""
    if not pairs:
        raise ValueError("DPO requires at least one preference pair")
    device = torch.device(training_device)
    if device.type == "cuda":
        from .gpu import probe_gpu_hardware

        inventory = probe_gpu_hardware()
        if not inventory.cuda_executable:
            raise RuntimeError(f"CUDA training requested but unavailable: {inventory.kernel_probe_error}")
    dtype = training_dtype or (torch.float32 if device.type == "cuda" else torch.float64)
    training_model = copy.copy(model)
    training_model.tensors = {
        name: tensor.detach().to(device=device, dtype=dtype) for name, tensor in model.tensors.items()
    }
    reference_model = copy.copy(training_model)
    reference_model.tensors = dict(training_model.tensors)
    base = training_model.tensors
    generator = torch.Generator(device=device).manual_seed(config.seed)
    factors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for site in lora_sites(model, config.targets):
        out_dim, in_dim = base[site].shape
        rank = min(config.rank, int(out_dim), int(in_dim))
        left = torch.randn(
            rank, int(in_dim), dtype=dtype, device=device, generator=generator
        ) / int(in_dim) ** 0.5
        right = torch.zeros(int(out_dim), rank, dtype=dtype, device=device)
        factors[site] = (left.requires_grad_(True), right.requires_grad_(True))
    params = [tensor for pair in factors.values() for tensor in pair]

    def current() -> dict[str, torch.Tensor]:
        return {**base, **{site: base[site] + right @ left for site, (left, right) in factors.items()}}

    reference_scores: list[tuple[torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        for prompt, chosen, rejected in pairs:
            reference_scores.append((
                _response_log_probability(reference_model, prompt, chosen),
                _response_log_probability(reference_model, prompt, rejected),
            ))
    guard = tuple(retention_guard)
    with torch.no_grad():
        guard_reference = {
            prompt: F.log_softmax(_last_logits(reference_model, prompt), dim=-1)
            for prompt in guard
        }

    optimizer = torch.optim.Adam(params, lr=config.learning_rate)
    _reset_peak_rss()
    started = time.perf_counter()
    loss_value = float("nan")
    trained = copy.copy(training_model)
    for _step in range(config.steps):
        trained.tensors = current()
        pair_losses: list[torch.Tensor] = []
        for index, (prompt, chosen, rejected) in enumerate(pairs):
            chosen_logp = _response_log_probability(trained, prompt, chosen)
            rejected_logp = _response_log_probability(trained, prompt, rejected)
            ref_chosen, ref_rejected = reference_scores[index]
            preference_logit = config.beta * (
                (chosen_logp - rejected_logp) - (ref_chosen - ref_rejected)
            )
            pair_losses.append(-F.logsigmoid(preference_logit))
        loss = torch.stack(pair_losses).mean()
        if guard and config.retention_weight > 0:
            kl = torch.stack(
                [
                    F.kl_div(
                        F.log_softmax(_last_logits(trained, prompt), dim=-1),
                        guard_reference[prompt],
                        log_target=True,
                        reduction="sum",
                    )
                    for prompt in guard
                ]
            ).mean()
            loss = loss + config.retention_weight * kl
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.item())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall = time.perf_counter() - started
    with torch.no_grad():
        state = current()
        snapshot = _training_update_snapshot(model.tensors, base, {site: state[site] for site in factors})
        changed = {site: value for site, value in snapshot.items() if not torch.equal(value, model.tensors[site])}
    changed_count = sum(
        int((value != model.tensors[name]).sum().item()) for name, value in changed.items()
    )
    delta_sq = sum(
        float(((value - model.tensors[name]) ** 2).sum().item()) for name, value in changed.items()
    )
    return PreferenceFit(
        config=config,
        merged_weights=changed,
        cost=FitCost(
            optimizer_steps=config.steps,
            wall_seconds=wall,
            trainable_parameters=sum(int(parameter.numel()) for parameter in params),
            changed_parameters=changed_count,
            total_model_parameters=int(model.total_parameters),
            peak_rss_bytes=_peak_rss_bytes(),
            delta_l2_norm=delta_sq**0.5,
            conversion_drift_l2_norm=_conversion_drift(model.tensors, base),
        ),
        final_train_loss=loss_value,
    )


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
    training_device: str | torch.device = "cpu",
    training_dtype: torch.dtype | None = None,
) -> BaselineFit:
    """Train on a declared device and return CPU float64 checkpoints for canonical scoring."""
    device = torch.device(training_device)
    if device.type == "cuda":
        from .gpu import probe_gpu_hardware

        inventory = probe_gpu_hardware()
        if not inventory.cuda_executable:
            raise RuntimeError(f"CUDA training requested but unavailable: {inventory.kernel_probe_error}")
    dtype = training_dtype or (torch.float32 if device.type == "cuda" else torch.float64)
    training_model = copy.copy(model)
    training_model.tensors = {
        name: tensor.detach().to(device=device, dtype=dtype)
        for name, tensor in model.tensors.items()
    }
    generator = torch.Generator(device=device).manual_seed(config.seed)
    guard = tuple(retention_guard)
    with torch.no_grad():
        guard_ref = {p: F.log_softmax(_last_logits(training_model, p), dim=-1) for p in guard}

    base = training_model.tensors
    canonical_base = model.tensors
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
            a = torch.randn(
                rank, int(in_dim), dtype=dtype, device=device, generator=generator
            ) / (int(in_dim) ** 0.5)
            b = torch.zeros(int(out_dim), rank, dtype=dtype, device=device)
            factors[site] = (a.requires_grad_(True), b.requires_grad_(True))
        params = [t for pair in factors.values() for t in pair]

        def current() -> dict[str, torch.Tensor]:
            return {**base, **{site: base[site] + b @ a for site, (a, b) in factors.items()}}

        trainable = sum(int(t.numel()) for t in params)

    optimizer = torch.optim.Adam(params, lr=config.lr)
    trained = copy.copy(training_model)
    checkpoints = set(config.checkpoints) | {config.steps}
    _reset_peak_rss()
    started = time.perf_counter()
    excluded = 0.0
    loss_value = float("nan")

    def snapshot() -> dict[str, torch.Tensor]:
        with torch.no_grad():
            state = current()
            selected = {k: state[k] for k in factors} if config.kind == "lora" else state
            return _training_update_snapshot(canonical_base, base, selected)

    for step in range(1, config.steps + 1):
        trained.tensors = current()
        loss = _mean_ce(trained, dev)
        if guard and config.retention_weight > 0:
            batch = guard
            if 0 < config.guard_batch < len(guard):
                picks = torch.randperm(
                    len(guard), generator=generator, device=device
                )[: config.guard_batch]
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
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    peak = _peak_rss_bytes()
    state = snapshot()
    changed = {k: v for k, v in state.items() if not torch.equal(v, canonical_base[k])}
    changed_count = sum(
        int((value != canonical_base[name]).sum().item()) for name, value in changed.items()
    )
    delta_sq = sum(
        float(((value - canonical_base[name]) ** 2).sum().item())
        for name, value in changed.items()
    )
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
            conversion_drift_l2_norm=_conversion_drift(canonical_base, base),
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
        "dataset": {
            "schema_version": 1,
            "gym_id": splits.gym_id,
            "gym_digest": splits.gym_digest,
            "dataset_digest": splits.dataset_digest,
        },
        "cost": asdict(fit.cost),
        "weights_digest": fit.weights_digest(),
        "dev_accuracy": accuracy(candidate, splits.dev),
        "val_accuracy": accuracy(candidate, splits.val),
        "practice_retention": retention(base, candidate, practice),
    }


class SealedHoldout:
    """Scores each named candidate once. A second request for a name raises instead of answering."""

    def __init__(self, gym: Any, sealed_retention: Sequence[str]) -> None:
        if not gym.holdout_rules:
            raise ValueError("sealed baseline evaluation requires separate Gym holdout rules")
        from .gym_data import materialize_holdout_items

        self._items = materialize_holdout_items(gym)
        self._retention = tuple(sealed_retention)
        self._used: set[str] = set()
        self._used_identities: set[str] = set()

    def evaluate_once(self, name: str, base: Any, merged_weights: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        from .edit_plan import tensor_state_digest

        candidate = merged_model(base, merged_weights)
        identity = tensor_state_digest(candidate.tensors)
        if name in self._used or identity in self._used_identities:
            raise RuntimeError(f"holdout already used for {name!r}")
        self._used.add(name)
        self._used_identities.add(identity)
        prompts = tuple(p for p in self._retention if base.tokenizer.encode(p))
        hits = accuracy(candidate, self._items)
        return {
            "holdout_accuracy": hits,
            "holdout_correct": round(hits * len(self._items)),
            "holdout_total": len(self._items),
            "sealed_retention": retention(base, candidate, prompts),
            "sealed_retention_total": len(prompts),
        }
