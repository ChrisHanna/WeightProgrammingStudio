"""Milestone 1 condition 6: score a frozen candidate without knowing how it was made.

This module imports only the standard library and torch, and it loads on its own without the
package, so no proposer can reach it. The input is a model and a `FrozenCandidate`; nothing about
synthesis is read. `tests/test_independent_eval.py` enforces all of this.

Metrics compare the candidate with the base model:

* Next-token cases, grouped by task: accuracy, retention of correct answers, top-1 agreement,
  KL(base || candidate) and total variation, all at the prompt's last position.
* Running text: perplexity, plus KL, total variation and top-1 agreement at every position.
* Controls: a no-op that must score zero change, or the evaluation raises, and random perturbations
  with the candidate's support and per-tensor Frobenius norm.

Proportions use Wilson intervals, since a bootstrap collapses to zero width at 0 or 1. Other
statistics use a percentile bootstrap over prompts or sentences. Prompts in a task share a template,
so the intervals describe this fixed corpus and don't generalize to other text.
"""

from __future__ import annotations

import copy
import ctypes
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

CORPUS_PATH = Path(__file__).with_name("data") / "independent_eval_corpus_v1.json"
CORPUS_DIGEST = "sha256:ddd72d04fad4a19d2a402542b559a297bf80b30c1d540a80595fe5d29cf1b632"
WILSON_Z_95 = 1.959963984540054
TARGET_ROW = "__declared_target__"


class IndependentEvalError(ValueError):
    """The inputs can't support a trustworthy evaluation."""


class ControlFailure(IndependentEvalError):
    """The no-op control measured a change, so no number from this run can be trusted."""


def _digest_bytes(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def parameter_digest(tensors: Mapping[str, torch.Tensor]) -> str:
    """Digest every tensor's name, shape and f64 little-endian values, independent of insertion order."""
    outer = hashlib.sha256()
    for name in sorted(tensors):
        values = tensors[name].detach().to(device="cpu", dtype=torch.float64).contiguous()
        header = json.dumps({"name": name, "shape": list(values.shape)}, sort_keys=True, separators=(",", ":"))
        outer.update(header.encode("utf-8"))
        outer.update(hashlib.sha256(_tensor_bytes(values)).digest())
    return f"sha256:{outer.hexdigest()}"


def _tensor_bytes(flat: torch.Tensor) -> bytes:
    """Read the tensor's memory directly; this environment has no numpy and `struct` is slow at 14M values."""
    if sys.byteorder != "little":  # pragma: no cover
        raise IndependentEvalError("parameter_digest requires a little-endian host")
    return ctypes.string_at(flat.data_ptr(), flat.numel() * 8)


@dataclass(frozen=True)
class TargetCase:
    prompt: str
    target: str


@dataclass(frozen=True)
class FrozenCandidate:
    """An immutable candidate: replacement tensors, declared targets and the digests that bind them."""

    candidate_id: str
    base_parameter_digest: str
    candidate_parameter_digest: str
    tensors: Mapping[str, torch.Tensor]
    target_cases: tuple[TargetCase, ...]


def freeze_candidate(
    base_tensors: Mapping[str, torch.Tensor],
    edited_tensors: Mapping[str, torch.Tensor],
    *,
    candidate_id: str,
    target_cases: Sequence[TargetCase | tuple[str, str]],
) -> FrozenCandidate:
    """Copy the replacement tensors, make them read-only and bind them to the base by digest."""
    if not candidate_id:
        raise IndependentEvalError("candidate_id is required")
    if not edited_tensors:
        raise IndependentEvalError("A candidate must replace at least one tensor")
    frozen: dict[str, torch.Tensor] = {}
    for name, value in edited_tensors.items():
        if name not in base_tensors:
            raise IndependentEvalError(f"Base model has no tensor {name!r}")
        if tuple(value.shape) != tuple(base_tensors[name].shape):
            raise IndependentEvalError(f"{name!r} changed shape")
        frozen[name] = value.detach().to(device="cpu", dtype=torch.float64).clone()
    cases = tuple(c if isinstance(c, TargetCase) else TargetCase(*c) for c in target_cases)
    return FrozenCandidate(
        candidate_id=candidate_id,
        base_parameter_digest=parameter_digest(base_tensors),
        candidate_parameter_digest=parameter_digest({**base_tensors, **frozen}),
        tensors=MappingProxyType(frozen),
        target_cases=cases,
    )


def load_corpus(path: str | Path = CORPUS_PATH) -> dict[str, Any]:
    """Load the corpus and refuse it unless its canonical digest matches the pinned value."""
    corpus = json.loads(Path(path).read_text(encoding="utf-8"))
    digest = corpus_digest(corpus)
    if digest != CORPUS_DIGEST:
        raise IndependentEvalError(f"Corpus digest {digest} does not match the pinned {CORPUS_DIGEST}")
    return corpus


def corpus_digest(corpus: Mapping[str, Any]) -> str:
    """Hash parsed JSON, so a checkout's line endings can't change the digest."""
    return _digest_bytes(json.dumps(corpus, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


@dataclass(frozen=True)
class EvalConfig:
    bootstrap_seed: int = 20261106
    bootstrap_samples: int = 2000
    random_control_seeds: tuple[int, ...] = (101, 102, 103, 104, 105)
    confidence: str = "0.95"

    def __post_init__(self) -> None:
        if self.confidence != "0.95":
            raise IndependentEvalError("Only 95% intervals are implemented")
        if self.bootstrap_samples < 100:
            raise IndependentEvalError("bootstrap_samples must be at least 100")
        if not self.random_control_seeds:
            raise IndependentEvalError("At least one random control is required")


def wilson_interval(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    p = successes / total
    z2 = WILSON_Z_95**2
    denominator = 1 + z2 / total
    center = (p + z2 / (2 * total)) / denominator
    half = WILSON_Z_95 * math.sqrt(p * (1 - p) / total + z2 / (4 * total**2)) / denominator
    # At 0 and 1 the bound equals p exactly; rounding mustn't exclude the point estimate.
    return [0.0 if successes == 0 else max(0.0, center - half), 1.0 if successes == total else min(1.0, center + half)]


def _proportion(successes: int, total: int) -> dict[str, Any]:
    return {
        "value": successes / total if total else None,
        "successes": successes,
        "n": total,
        "ci95": wilson_interval(successes, total),
    }


class _Bootstrap:
    def __init__(self, config: EvalConfig) -> None:
        self.samples = config.bootstrap_samples
        self.generator = torch.Generator().manual_seed(config.bootstrap_seed)

    def _indices(self, n: int) -> torch.Tensor:
        return torch.randint(0, n, (self.samples, n), generator=self.generator)

    @staticmethod
    def _bounds(stats: torch.Tensor) -> list[float]:
        return [float(torch.quantile(stats, 0.025)), float(torch.quantile(stats, 0.975))]

    def mean(self, values: Sequence[float]) -> dict[str, Any]:
        data = torch.tensor(list(values), dtype=torch.float64)
        return {
            "value": float(data.mean()),
            "n": len(values),
            "ci95": self._bounds(data[self._indices(len(values))].mean(dim=1)),
        }

    def ratio(self, numerator: Sequence[float], denominator: Sequence[float]) -> dict[str, Any]:
        top = torch.tensor(list(numerator), dtype=torch.float64)
        bottom = torch.tensor(list(denominator), dtype=torch.float64)
        index = self._indices(len(top))
        return {
            "value": float(top.sum() / bottom.sum()),
            "n": len(top),
            "ci95": self._bounds(top[index].sum(dim=1) / bottom[index].sum(dim=1)),
        }

    def perplexity(self, nll: Sequence[float], tokens: Sequence[int]) -> dict[str, Any]:
        top = torch.tensor(list(nll), dtype=torch.float64)
        bottom = torch.tensor(list(tokens), dtype=torch.float64)
        index = self._indices(len(top))
        return {
            "value": float((top.sum() / bottom.sum()).exp()),
            "n_sentences": len(top),
            "n_tokens": int(bottom.sum()),
            "ci95": self._bounds((top[index].sum(dim=1) / bottom[index].sum(dim=1)).exp()),
        }

    def perplexity_ratio(self, base_nll: Sequence[float], cand_nll: Sequence[float], tokens: Sequence[int]) -> dict[str, Any]:
        base = torch.tensor(list(base_nll), dtype=torch.float64)
        cand = torch.tensor(list(cand_nll), dtype=torch.float64)
        count = torch.tensor(list(tokens), dtype=torch.float64)
        index = self._indices(len(base))
        delta = (cand[index].sum(dim=1) - base[index].sum(dim=1)) / count[index].sum(dim=1)
        return {
            "value": float(((cand.sum() - base.sum()) / count.sum()).exp()),
            "n_sentences": len(base),
            "ci95": self._bounds(delta.exp()),
        }


def _kl_tv(base_log: torch.Tensor, other_log: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    base_prob = base_log.exp()
    # Clamped because a sum of tiny negative rounding errors is not evidence of a negative divergence.
    kl = (base_prob * (base_log - other_log)).sum(dim=-1).clamp(min=0.0)
    tv = 0.5 * (base_prob - other_log.exp()).abs().sum(dim=-1)
    return kl, tv


def _logits(model: Any, ids: Sequence[int]) -> torch.Tensor:
    with torch.inference_mode():
        logits, _ = model.forward_with_mlp_key(list(ids))
    if not torch.isfinite(logits).all():
        raise IndependentEvalError("Model produced non-finite logits")
    return logits.to(torch.float64)


def _with_tensors(model: Any, overrides: Mapping[str, torch.Tensor]) -> Any:
    """A shallow copy with replaced tensors. The caller's model is never changed."""
    variant = copy.copy(model)
    variant.tensors = {**model.tensors, **overrides}
    return variant


def _random_control(base: Mapping[str, torch.Tensor], candidate: FrozenCandidate, seed: int) -> dict[str, torch.Tensor]:
    """Gaussian noise on the candidate's changed entries, rescaled to its per-tensor Frobenius norm."""
    generator = torch.Generator().manual_seed(seed)
    out: dict[str, torch.Tensor] = {}
    for name in sorted(candidate.tensors):
        weight = base[name].to(torch.float64)
        delta = candidate.tensors[name] - weight
        mask = delta != 0
        noise = torch.randn(weight.shape, dtype=torch.float64, generator=generator) * mask
        norm = torch.linalg.norm(noise)
        out[name] = weight + noise * (torch.linalg.norm(delta) / norm) if float(norm) > 0 else weight.clone()
    return out


def _encode_cases(model: Any, cases: Sequence[TargetCase]) -> list[tuple[str, list[int], int]]:
    encoded = []
    for case in cases:
        ids = model.tokenizer.encode(case.prompt)
        target = model.tokenizer.encode(case.target)
        if not ids:
            raise IndependentEvalError(f"Prompt {case.prompt!r} encodes to no tokens")
        if len(target) != 1:
            raise IndependentEvalError(f"Target {case.target!r} is {len(target)} tokens, and must be exactly 1")
        encoded.append((case.prompt, ids, target[0]))
    return encoded


def _encode_corpus(model: Any, corpus: Mapping[str, Any], candidate: FrozenCandidate) -> tuple[dict[str, Any], list[list[int]]]:
    tasks = {name: _encode_cases(model, [TargetCase(*pair) for pair in pairs]) for name, pairs in corpus["tasks"].items()}
    if not candidate.target_cases:
        raise IndependentEvalError("Candidate declares no target cases, so target accuracy can't be measured")
    tasks[TARGET_ROW] = _encode_cases(model, candidate.target_cases)
    sentences = [model.tokenizer.encode(s) for s in corpus["text"]]
    if any(len(ids) < 2 for ids in sentences):
        raise IndependentEvalError("Every corpus sentence needs at least two tokens")
    return tasks, sentences


def _measure(base: Any, variants: Mapping[str, Any], tasks: Mapping[str, Any], sentences: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Run one prompt at a time through every variant, keeping per-unit numbers and no distributions."""
    raw: dict[str, Any] = {
        label: {
            "tasks": {t: {"base_ok": [], "ok": [], "agree": [], "kl": [], "tv": [], "target_logprob_delta": []} for t in tasks},
            "text": {"base_nll": [], "nll": [], "tokens": [], "kl_sum": [], "tv_sum": [], "agree_count": [], "kl_positions": [], "agree_positions": []},
        }
        for label in variants
    }
    for task, cases in tasks.items():
        for _, ids, target in cases:
            base_log = torch.log_softmax(_logits(base, ids)[-1], dim=-1)
            base_top = int(base_log.argmax())
            for label, model in variants.items():
                log = torch.log_softmax(_logits(model, ids)[-1], dim=-1)
                kl, tv = _kl_tv(base_log, log)
                top = int(log.argmax())
                row = raw[label]["tasks"][task]
                row["base_ok"].append(base_top == target)
                row["ok"].append(top == target)
                row["agree"].append(top == base_top)
                row["kl"].append(float(kl))
                row["tv"].append(float(tv))
                row["target_logprob_delta"].append(float(log[target] - base_log[target]))
    for ids in sentences:
        follow = torch.tensor(ids[1:])
        base_log = torch.log_softmax(_logits(base, ids)[:-1], dim=-1)
        base_nll = float(-base_log.gather(1, follow.unsqueeze(1)).sum())
        base_top = base_log.argmax(dim=-1)
        for label, model in variants.items():
            log = torch.log_softmax(_logits(model, ids)[:-1], dim=-1)
            kl, tv = _kl_tv(base_log, log)
            agree = log.argmax(dim=-1) == base_top
            text = raw[label]["text"]
            text["base_nll"].append(base_nll)
            text["nll"].append(float(-log.gather(1, follow.unsqueeze(1)).sum()))
            text["tokens"].append(len(ids) - 1)
            text["kl_sum"].append(float(kl.sum()))
            text["tv_sum"].append(float(tv.sum()))
            text["agree_count"].append(int(agree.sum()))
            text["kl_positions"].extend(kl.tolist())
            text["agree_positions"].extend(agree.tolist())
    return raw


def _task_metrics(row: Mapping[str, Sequence[Any]], boot: _Bootstrap) -> dict[str, Any]:
    n = len(row["agree"])
    base_ok = list(row["base_ok"])
    ok = list(row["ok"])
    return {
        "n": n,
        "base_accuracy": _proportion(sum(base_ok), n),
        "candidate_accuracy": _proportion(sum(ok), n),
        "accuracy_delta": boot.mean([float(c) - float(b) for b, c in zip(base_ok, ok, strict=True)]),
        # `value` is null with n=0 when the base model never answered a task correctly.
        "correct_retention": _proportion(sum(b and c for b, c in zip(base_ok, ok, strict=True)), sum(base_ok)),
        "top1_agreement": _proportion(sum(row["agree"]), n),
        "kl_nats": boot.mean(row["kl"]),
        "total_variation": boot.mean(row["tv"]),
        "target_logprob_delta_nats": boot.mean(row["target_logprob_delta"]),
    }


def _text_metrics(text: Mapping[str, Sequence[Any]], boot: _Bootstrap) -> dict[str, Any]:
    tokens = text["tokens"]
    return {
        "base_perplexity": boot.perplexity(text["base_nll"], tokens),
        "candidate_perplexity": boot.perplexity(text["nll"], tokens),
        "perplexity_ratio": boot.perplexity_ratio(text["base_nll"], text["nll"], tokens),
        "kl_nats_per_token": boot.ratio(text["kl_sum"], tokens),
        "total_variation_per_token": boot.ratio(text["tv_sum"], tokens),
        "top1_agreement": {
            **_proportion(sum(text["agree_count"]), sum(tokens)),
            # Positions in a sentence are correlated, so resample whole sentences for the interval.
            "sentence_bootstrap_ci95": boot.ratio(text["agree_count"], tokens)["ci95"],
        },
    }


def _quantile(values: Sequence[float], q: float) -> float | None:
    return float(torch.quantile(torch.tensor(list(values), dtype=torch.float64), q)) if values else None


def top1_versus_kl(raw_variant: Mapping[str, Any]) -> dict[str, Any]:
    """How much distribution change does top-1 agreement fail to see?

    Disagreement and total variation share a 0-to-1 scale, so they can be compared directly. Since the
    two argmaxes differ only if TV > 0, TV is at least sensitive to everything agreement is. KL has no
    such scale, so it is split by whether top-1 survived.
    """
    kl: list[float] = list(raw_variant["text"]["kl_positions"])
    agree: list[bool] = list(raw_variant["text"]["agree_positions"])
    tv_total = sum(raw_variant["text"]["tv_sum"])
    for task, row in raw_variant["tasks"].items():
        if task != TARGET_ROW:
            kl.extend(row["kl"])
            agree.extend(row["agree"])
            tv_total += sum(row["tv"])
    kept = [k for k, a in zip(kl, agree, strict=True) if a]
    flipped = [k for k, a in zip(kl, agree, strict=True) if not a]
    total_kl = sum(kl)
    return {
        "n_positions": len(kl),
        "top1_disagreement": len(flipped) / len(kl),
        "mean_total_variation": tv_total / len(kl),
        "mean_kl_nats": total_kl / len(kl),
        "kl_mean_where_top1_kept": sum(kept) / len(kept) if kept else None,
        "kl_mean_where_top1_flipped": sum(flipped) / len(flipped) if flipped else None,
        "kl_p50_where_top1_kept": _quantile(kept, 0.5),
        "kl_p90_where_top1_kept": _quantile(kept, 0.9),
        "kl_max_where_top1_kept": max(kept) if kept else None,
        "share_of_total_kl_where_top1_kept": sum(kept) / total_kl if total_kl > 0 else None,
    }


def _summary(raw_variant: Mapping[str, Any], boot: _Bootstrap) -> dict[str, Any]:
    pooled: dict[str, list[Any]] = {}
    for task, row in raw_variant["tasks"].items():
        if task != TARGET_ROW:
            for key, values in row.items():
                pooled.setdefault(key, []).extend(values)
    return {
        "all_tasks": _task_metrics(pooled, boot),
        "text": _text_metrics(raw_variant["text"], boot),
        "top1_versus_kl": top1_versus_kl(raw_variant),
    }


def _check_noop(raw_noop: Mapping[str, Any]) -> dict[str, Any]:
    text = raw_noop["text"]
    kl_max = max(text["kl_positions"])
    all_agree = all(text["agree_positions"])
    for row in raw_noop["tasks"].values():
        kl_max = max(kl_max, max(row["kl"]))
        all_agree = all_agree and all(row["agree"]) and row["ok"] == row["base_ok"]
    nll_equal = text["nll"] == text["base_nll"]
    if kl_max != 0.0 or not all_agree or not nll_equal:
        raise ControlFailure(f"No-op control changed the output: max KL {kl_max}, all top-1 agree {all_agree}, NLL equal {nll_equal}")
    return {"passed": True, "max_kl_nats": kl_max, "all_top1_agree": True, "nll_identical": True}


def _check_inputs(model: Any, candidate: FrozenCandidate) -> None:
    base = model.tensors
    actual_base = parameter_digest(base)
    if actual_base != candidate.base_parameter_digest:
        raise IndependentEvalError(f"Candidate was frozen against {candidate.base_parameter_digest}, but this model is {actual_base}")
    unknown = set(candidate.tensors) - set(base)
    if unknown:
        raise IndependentEvalError(f"Candidate has tensors the model lacks: {sorted(unknown)}")
    actual = parameter_digest({**base, **candidate.tensors})
    if actual != candidate.candidate_parameter_digest:
        raise IndependentEvalError(f"Candidate tensors hash to {actual}, not the frozen {candidate.candidate_parameter_digest}")
    if actual == actual_base:
        raise IndependentEvalError("Candidate is identical to the base model")


def evaluate_candidate(model: Any, candidate: FrozenCandidate, *, config: EvalConfig | None = None) -> dict[str, Any]:
    """Measure a frozen candidate against its base. Raises instead of returning a partial report."""
    config = config or EvalConfig()
    _check_inputs(model, candidate)
    corpus = load_corpus()
    tasks, sentences = _encode_corpus(model, corpus, candidate)
    noop_tensors = {name: model.tensors[name].clone() for name in candidate.tensors}
    random_labels = {f"random_seed_{seed}": seed for seed in config.random_control_seeds}
    variants = {
        "candidate": _with_tensors(model, candidate.tensors),
        "noop": _with_tensors(model, noop_tensors),
        **{label: _with_tensors(model, _random_control(model.tensors, candidate, seed)) for label, seed in random_labels.items()},
    }
    raw = _measure(model, variants, tasks, sentences)
    noop_result = _check_noop(raw["noop"])
    boot = _Bootstrap(config)
    labels = list(variants)
    matrix = {
        task: {label: _task_metrics(raw[label]["tasks"][task], boot) for label in labels}
        for task in tasks
        if task != TARGET_ROW
    }
    target = raw["candidate"]["tasks"][TARGET_ROW]
    return {
        "schema_version": 1,
        "kind": "nf_independent_eval_report_v1",
        "candidate_id": candidate.candidate_id,
        "base_parameter_digest": candidate.base_parameter_digest,
        "candidate_parameter_digest": candidate.candidate_parameter_digest,
        "corpus_id": corpus["corpus_id"],
        "corpus_digest": CORPUS_DIGEST,
        "config": {
            "bootstrap_seed": config.bootstrap_seed,
            "bootstrap_samples": config.bootstrap_samples,
            "random_control_seeds": list(config.random_control_seeds),
            "kl_direction": "KL(base || candidate), nats",
            "interval_methods": {"proportions": "wilson", "other": "percentile bootstrap over prompts or sentences"},
        },
        "target": {
            "source": "declared by the frozen candidate, which may have seen these prompts",
            "cases": [c.prompt for c in candidate.target_cases],
            "metrics": _task_metrics(target, boot),
        },
        "candidate": _summary(raw["candidate"], boot),
        "retention_matrix": matrix,
        "controls": {
            "noop": noop_result,
            "random": {label: {"seed": seed, **_summary(raw[label], boot)} for label, seed in random_labels.items()},
        },
    }
