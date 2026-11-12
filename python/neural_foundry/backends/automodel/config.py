"""Pinned local AutoModel runtime and SFT recipe configuration."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...gym_job import GymJobError, is_finite_number
from ...gym_service import _document_digest

AUTOMODEL_VERSIONS = {
    "nemo-automodel": "0.7.0", "torch": "2.10.0+cu130", "triton": "3.6.0", "transformers": "5.18.0",
    "torchao": "0.18.0", "torchdata": "0.11.0", "datasets": "5.1.0",
}
AUTOMODEL_PYTHON = (3, 13, 14)
PEFT_RELOAD_VERSIONS = {"torch": "2.9.1+cu128", "transformers": "4.57.3", "peft": "0.18.0"}
PEFT_RELOAD_PYTHON = (3, 12, 13)
QLORA_BITSANDBYTES_VERSION = "0.49.1"


@dataclass(frozen=True)
class AutoModelBackendConfig:
    python_executable: str
    timeout_seconds: float = 1800.0
    reload_python_executable: str | None = None
    maximum_cuda_allocated_bytes: int | None = None
    training_device: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.python_executable, str) or not self.python_executable or not Path(self.python_executable).is_absolute():
            raise GymJobError("AutoModel requires an absolute Python interpreter path")
        if not is_finite_number(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise GymJobError("AutoModel timeout must be finite and positive")
        if self.reload_python_executable is not None and (
            not isinstance(self.reload_python_executable, str) or not Path(self.reload_python_executable).is_absolute()
        ):
            raise GymJobError("AutoModel reload Python must be an absolute interpreter path")
        if self.maximum_cuda_allocated_bytes is not None and (
            type(self.maximum_cuda_allocated_bytes) is not int or self.maximum_cuda_allocated_bytes <= 0):
            raise GymJobError("AutoModel CUDA allocation limit must be positive bytes")
        if self.training_device not in {None, "cuda", "cuda:0"}:
            raise GymJobError("AutoModel training device must be local CUDA")

    def document(self) -> dict[str, Any]:
        document = {"python_executable": self.python_executable, "timeout_seconds": self.timeout_seconds,
            "placement": "local", "versions": dict(AUTOMODEL_VERSIONS), "python": list(AUTOMODEL_PYTHON),
            "reload_python_executable": self.reload_python_executable,
            "reload_versions": dict(PEFT_RELOAD_VERSIONS), "reload_python": list(PEFT_RELOAD_PYTHON),
            "qlora_bitsandbytes": QLORA_BITSANDBYTES_VERSION}
        if self.maximum_cuda_allocated_bytes is not None:
            document["maximum_cuda_allocated_bytes"] = self.maximum_cuda_allocated_bytes
        if self.training_device is not None:
            document["training_device"] = self.training_device
        return document

    @property
    def digest(self) -> str:
        return _document_digest(self.document())


@dataclass(frozen=True)
class AutoModelSFTRecipe:
    method: str
    seed: int
    steps: int
    max_sequence_tokens: int
    tokens_per_batch: int
    max_dataset_tokens: int
    learning_rate: float
    rank: int | None = None
    target_modules: tuple[str, ...] = ()

    @classmethod
    def from_document(cls, recipe: Mapping[str, Any]) -> AutoModelSFTRecipe:
        fields = {"method", "seed", "steps", "max_sequence_tokens", "tokens_per_batch", "max_dataset_tokens", "learning_rate"}
        method = recipe.get("method")
        if method in {"lora", "qlora"}:
            fields |= {"rank", "target_modules"}
        if set(recipe) != fields or method not in {"sft", "lora", "qlora"}:
            raise GymJobError("AutoModel SFT recipe has missing or unsupported fields")
        if type(recipe["seed"]) is not int or recipe["seed"] < 0:
            raise GymJobError("AutoModel seed must be a nonnegative integer")
        for key in ("steps", "max_sequence_tokens", "tokens_per_batch", "max_dataset_tokens"):
            if type(recipe[key]) is not int or recipe[key] < 1:
                raise GymJobError(f"AutoModel {key} must be a positive integer")
        if recipe["max_sequence_tokens"] > recipe["tokens_per_batch"]:
            raise GymJobError("AutoModel sequence budget exceeds its batch budget")
        if not is_finite_number(recipe["learning_rate"]) or recipe["learning_rate"] <= 0:
            raise GymJobError("AutoModel learning rate must be finite and positive")
        if method in {"lora", "qlora"}:
            if type(recipe["rank"]) is not int or recipe["rank"] < 1:
                raise GymJobError("AutoModel LoRA rank must be a positive integer")
            targets = recipe["target_modules"]
            if (not isinstance(targets, (list, tuple)) or not targets
                or any(not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*", value) is None for value in targets)):
                raise GymJobError("AutoModel LoRA target modules must be a nonempty string list")
            return cls(**{**dict(recipe), "target_modules": tuple(targets)})
        return cls(**dict(recipe))

    def document(self) -> dict[str, Any]:
        document = vars(self).copy()
        if self.method == "sft":
            document.pop("rank")
            document.pop("target_modules")
        else:
            document["target_modules"] = list(self.target_modules)
        return document

    @property
    def digest(self) -> str:
        return _document_digest(self.document())