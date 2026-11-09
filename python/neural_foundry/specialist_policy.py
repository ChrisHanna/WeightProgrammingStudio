"""Unactivated specialist-v2 proposal metrics; no current gate imports this module."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from .gym_job import GymJobError


def paired_bootstrap_improvement(base: Sequence[float], candidate: Sequence[float], *, seed: int = 0,
                                 replicates: int = 10_000) -> dict[str, Any]:
    if (not base or len(base) != len(candidate) or type(seed) is not int or seed < 0
        or type(replicates) is not int or not 1_000 <= replicates <= 100_000
        or any(type(value) not in {int, float} or value not in {0, 1} for value in (*base, *candidate))):
        raise GymJobError("paired improvement requires matching binary case outcomes and bounded bootstrap settings")
    differences = [float(after - before) for before, after in zip(base, candidate, strict=True)]
    net_cases = sum(differences)
    randomizer = random.Random(seed)
    count = len(differences)
    means = [sum(differences[randomizer.randrange(count)] for _ in range(count)) / count for _ in range(replicates)]
    means.sort()
    lower = means[int(0.025 * replicates)]
    upper = means[min(replicates - 1, int(0.975 * replicates))]
    return {"cases": count, "base_correct": sum(base), "candidate_correct": sum(candidate),
            "net_case_improvement": net_cases, "observed_gain": net_cases / count,
            "confidence": 0.95, "interval_method": "paired case bootstrap percentile",
            "lower_bound": lower, "upper_bound": upper, "seed": seed, "replicates": replicates,
            "selection_adjusted": False,
            "interpretation": "Exploratory only when the same validation split selected the candidate."}


def assess_specialist_v2(task: Mapping[str, Any], required: Mapping[str, Any], *, minimum_net_cases: int = 4) -> dict[str, Any]:
    if task.get("split") != "validation" or required.get("split") != "validation":
        raise GymJobError("specialist v2 may only assess public validation evidence")
    base, candidate = task.get("base_case_outcomes"), task.get("candidate_case_outcomes")
    if not isinstance(base, Sequence) or not isinstance(candidate, Sequence):
        raise GymJobError("specialist v2 requires paired base and candidate case outcomes")
    interval = paired_bootstrap_improvement(base, candidate)
    checks = {"minimum_net_improvement": interval["net_case_improvement"] >= minimum_net_cases,
              "confidence_interval_lower_bound": interval["lower_bound"] > 0,
              "format": required.get("format_valid") == 1.0,
              "task_scoped_required_set": required.get("exact_match") == 1.0}
    return {"passed": all(checks.values()), "checks": checks, "paired_improvement": interval,
            "holdout_accessed": False, "generalist_authorized": False, "release_authorized": False,
            "policy_status": "proposal_pending_approval"}
