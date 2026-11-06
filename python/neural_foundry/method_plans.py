"""Express edit methods as one replayable `EditPlan`, from the operation and operands they ran.

Methods synthesize differently, but each ends in a Rust matrix edit. Replaying that same operation
reproduces its weights exactly, whereas adopting a finished weight as `w + (w' - w)` can round
differently in f64.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .edit_plan import EditGraphError, EditPlan


@dataclass(frozen=True)
class MethodEditStep:
    tensor: str
    method: str
    operation_type: str
    operands: Mapping[str, torch.Tensor]
    factor: float = 1.0


def low_rank_step(tensor: str, method: str, left: torch.Tensor, right: torch.Tensor, *, factor: float = 1.0) -> MethodEditStep:
    return MethodEditStep(tensor, method, "low_rank_update", {"left_factor": left.detach(), "right_factor": right.detach()}, factor)


def add_delta_step(tensor: str, method: str, delta: torch.Tensor) -> MethodEditStep:
    return MethodEditStep(tensor, method, "add_delta", {"delta_matrix": delta.detach()})


def edit_plan_from_steps(base_model: Any, steps: Sequence[MethodEditStep]) -> EditPlan:
    """A plan that runs `steps` in order on `base_model`. Nothing executes until it is previewed."""
    plan = EditPlan(base_model)
    for step in steps:
        arguments: dict[str, Any] = {"operation_type": step.operation_type, "factor": step.factor, **step.operands}
        plan._add(step.tensor, step.method, _builder(plan, arguments))
    return plan


def _builder(plan: EditPlan, arguments: Mapping[str, Any]) -> Any:
    return lambda w: {**arguments, **plan._target(None, w.shape)}


def verified_method_plan(base_model: Any, steps: Sequence[MethodEditStep], expected: Mapping[str, torch.Tensor]) -> EditPlan:
    """Build and run the plan, and raise unless it bitwise reproduces the weights the method produced."""
    plan = edit_plan_from_steps(base_model, steps)
    edited = plan.preview().edited_tensors
    if set(edited) != set(expected):
        raise EditGraphError(f"The plan edits {sorted(edited)}, but the method edited {sorted(expected)}")
    for name, weight in expected.items():
        if not torch.equal(edited[name], weight.detach().cpu().to(torch.float64)):
            raise EditGraphError(f"The plan doesn't reproduce the method's weights for {name!r}")
    return plan
