"""Paired cluster statistics.

Items are grouped into cluster means first, so rendering one expression under more
prefixes adds no false precision. Resampling is stratified by holdout draw. These CIs
cover holdout sampling for the given candidates only. With a handful of synthesis seeds
a bootstrap over seeds isn't meaningful, so seed variation is reported and gated separately.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Interval:
    mean: float
    low: float
    high: float
    level: float
    clusters: int


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def cluster_bootstrap_ci(
    values: Sequence[float],
    strata: Sequence[int],
    *,
    level: float,
    resamples: int,
    seed: int,
) -> Interval:
    """Percentile CI of a stratified mean of per-cluster values, such as paired deltas."""
    if len(values) != len(strata) or not values:
        raise ValueError("values and strata must be equal-length and non-empty")
    data = torch.tensor(list(values), dtype=torch.float64)
    labels = torch.tensor(list(strata))
    generator = _generator(seed)
    totals = torch.zeros(resamples, dtype=torch.float64)
    for stratum in sorted(set(strata)):
        members = data[labels == stratum]
        picks = torch.randint(0, int(members.numel()), (resamples, int(members.numel())), generator=generator)
        totals += members[picks].sum(dim=1)
    means = totals / int(data.numel())
    tail = (1.0 - level) / 2.0
    low, high = torch.quantile(means, torch.tensor([tail, 1.0 - tail], dtype=torch.float64)).tolist()
    return Interval(float(data.mean()), float(low), float(high), level, int(data.numel()))


def sign_flip_p_value(deltas: Sequence[float], *, resamples: int, seed: int) -> float:
    """One-sided paired randomization test of mean delta > 0, flipping signs per cluster."""
    data = torch.tensor(list(deltas), dtype=torch.float64)
    if not data.numel():
        raise ValueError("deltas must be non-empty")
    observed = data.mean()
    signs = torch.randint(0, 2, (resamples, int(data.numel())), generator=_generator(seed)) * 2 - 1
    flipped = (signs.to(torch.float64) * data).mean(dim=1)
    # Tolerance so ties survive float rounding; the +1s keep the p-value valid and above zero.
    extreme = int((flipped >= observed - 1e-12).sum())
    return (extreme + 1) / (resamples + 1)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted
