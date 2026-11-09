"""Fail-closed per-step reward drift and KL monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..gym_job import GymJobError, _is_digest, is_finite_number


@dataclass(frozen=True)
class RewardObservation:
    step: int
    scorer_digest: str
    reward: float
    validation: float
    kl_nats: float
    stop: bool
    reason: str | None


class RewardDriftGuard:
    def __init__(self, *, scorer_digest: str, max_kl_nats: float) -> None:
        if not _is_digest(scorer_digest):
            raise GymJobError("reward guard requires a pinned scorer digest")
        if not is_finite_number(max_kl_nats) or max_kl_nats < 0:
            raise GymJobError("reward guard KL limit must be finite and nonnegative")
        self.scorer_digest = scorer_digest
        self.max_kl_nats = max_kl_nats
        self._observations: list[RewardObservation] = []

    @property
    def observations(self) -> tuple[RewardObservation, ...]:
        return tuple(self._observations)

    def observe(self, *, step: int, scorer_digest: str, reward: Any,
                validation: Any, kl_nats: Any) -> RewardObservation:
        if self._observations and self._observations[-1].stop:
            raise GymJobError("training cannot continue after the reward guard stops")
        if type(step) is not int or step != len(self._observations) + 1:
            raise GymJobError("reward guard requires contiguous one-based steps")
        if scorer_digest != self.scorer_digest:
            raise GymJobError("training reward scorer differs from the frozen scorer")
        if (not is_finite_number(reward) or not 0 <= reward <= 1
            or not is_finite_number(validation) or not 0 <= validation <= 1
            or not is_finite_number(kl_nats) or kl_nats < 0):
            raise GymJobError("reward, validation, and KL measurements must be finite and in range")
        previous = self._observations[-1] if self._observations else None
        drift = previous is not None and reward > previous.reward and validation < previous.validation
        reason = "possible_scorer_gaming" if drift else "kl_limit_exceeded" if kl_nats > self.max_kl_nats else None
        observation = RewardObservation(step, scorer_digest, float(reward), float(validation), float(kl_nats),
                                        reason is not None, reason)
        self._observations.append(observation)
        return observation