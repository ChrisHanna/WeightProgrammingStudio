"""Frozen PEFT sequence-training configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ...gym_job import GymJobError, is_finite_number
from ...gym_service import _document_digest


@dataclass(frozen=True)
class PeftBackendConfig:
    method: str
    seed: int
    steps: int
    max_sequence_tokens: int
    tokens_per_batch: int
    max_dataset_tokens: int
    rank: int
    target_modules: tuple[str, ...]
    learning_rate: float

    @classmethod
    def from_recipe(cls, recipe: Mapping[str, Any]) -> PeftBackendConfig:
        fields = {"method", "seed", "steps", "max_sequence_tokens", "tokens_per_batch",
                  "max_dataset_tokens", "rank", "target_modules", "learning_rate"}
        if set(recipe) != fields or recipe.get("method") != "lora":
            raise GymJobError("PEFT recipe has missing or unsupported fields")
        if type(recipe["seed"]) is not int or recipe["seed"] < 0:
            raise GymJobError("PEFT seed must be a nonnegative integer")
        for key in ("steps", "max_sequence_tokens", "tokens_per_batch", "max_dataset_tokens", "rank"):
            if type(recipe[key]) is not int or recipe[key] < 1:
                raise GymJobError(f"explicit positive training {key} is required")
        targets = recipe["target_modules"]
        if not isinstance(targets, (list, tuple)) or not targets or any(not isinstance(value, str) or not value for value in targets):
            raise GymJobError("training target_modules must be a nonempty string list")
        if not is_finite_number(recipe["learning_rate"]) or recipe["learning_rate"] <= 0:
            raise GymJobError("training learning rate must be finite and positive")
        return cls(recipe["method"], recipe["seed"], recipe["steps"], recipe["max_sequence_tokens"],
                   recipe["tokens_per_batch"], recipe["max_dataset_tokens"], recipe["rank"],
                   tuple(targets), recipe["learning_rate"])

    def document(self) -> dict[str, Any]:
        return {**vars(self), "target_modules": list(self.target_modules)}

    @property
    def digest(self) -> str:
        return _document_digest(self.document())