"""Tensor-only edit calculation and validated canonical deltas."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from ..discover import Site
from ..loader import WeightEditInputError


def _matrix(value: torch.Tensor, name: str) -> torch.Tensor:
    if (not isinstance(value, torch.Tensor) or value.ndim != 2 or value.dtype not in {torch.float32, torch.float64}
        or not bool(torch.isfinite(value).all())):
        raise WeightEditInputError(f"{name} must be a finite float32/float64 matrix")
    return value.detach().cpu().contiguous().clone()


@dataclass(frozen=True)
class Delta:
    kind: str
    U: torch.Tensor | None = None
    V: torch.Tensor | None = None
    D: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.kind == "dense" and self.D is not None and self.U is None and self.V is None:
            object.__setattr__(self, "D", _matrix(self.D, "dense delta"))
        elif self.kind == "low_rank" and self.U is not None and self.V is not None and self.D is None:
            left, right = _matrix(self.U, "left factor"), _matrix(self.V, "right factor")
            if left.shape[1] != right.shape[0] or not left.shape[1] or left.dtype != right.dtype:
                raise WeightEditInputError("low-rank factors have incompatible rank or dtype")
            object.__setattr__(self, "U", left)
            object.__setattr__(self, "V", right)
        else:
            raise WeightEditInputError("delta must contain a dense matrix or both low-rank factors")

    def matrix(self) -> torch.Tensor:
        return self.D.clone() if self.kind == "dense" else self.U @ self.V


class EditMethod(Protocol):
    method_id: str
    supported_roles: tuple[str, ...]

    def compute(self, site: Site, W: torch.Tensor, keys: torch.Tensor, targets: torch.Tensor,
                settings: Mapping[str, Any]) -> Delta: ...


def legacy_rank_factors(keys: torch.Tensor, target_deltas: torch.Tensor, retention_keys: torch.Tensor,
                        retention_nullspace_weight: float) -> tuple[torch.Tensor, torch.Tensor]:
    gram = (keys @ keys.T + float(retention_nullspace_weight) * (retention_keys @ retention_keys.T)
            + 1e-3 * torch.eye(keys.shape[0], dtype=keys.dtype, device=keys.device))
    return target_deltas.contiguous(), torch.linalg.solve(gram, keys).T.contiguous()


class LegacyKeyValueEdit:
    method_id = "legacy_key_value_v1"
    supported_roles = ("residual_write_mlp",)

    def __init__(self, retention_keys: torch.Tensor) -> None:
        self.retention_keys = _matrix(retention_keys, "retention keys")

    def compute(self, site: Site, W: torch.Tensor, keys: torch.Tensor, targets: torch.Tensor,
                settings: Mapping[str, Any]) -> Delta:
        weights, keys, targets = _matrix(W, "weights"), _matrix(keys, "keys"), _matrix(targets, "targets")
        if site.role not in self.supported_roles:
            raise WeightEditInputError("method does not support this site role")
        if (weights.shape != (site.output_size, site.input_size) or keys.shape[1] != site.input_size
            or targets.shape != (keys.shape[0], site.output_size) or not keys.shape[0]
            or self.retention_keys.shape[1] != site.input_size):
            raise WeightEditInputError("method matrix shapes do not match the canonical site")
        if len({weights.dtype, keys.dtype, targets.dtype, self.retention_keys.dtype}) != 1:
            raise WeightEditInputError("method matrices must share a compute dtype")
        value = settings.get("retention_nullspace_weight")
        if (set(settings) != {"retention_nullspace_weight"} or isinstance(value, bool)
            or not isinstance(value, (int, float)) or not 0 <= value < float("inf")):
            raise WeightEditInputError("retention_nullspace_weight must be finite and nonnegative")
        left, right = legacy_rank_factors(keys.T, (targets - keys @ weights.T).T, self.retention_keys.T, value)
        return Delta("low_rank", U=left, V=right)