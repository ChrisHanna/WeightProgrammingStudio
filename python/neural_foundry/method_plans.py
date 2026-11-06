"""Express edit methods as one replayable `EditPlan`, from the operation and operands they ran.

Methods synthesize differently, but each ends in a Rust matrix edit. Replaying that same operation
reproduces its weights exactly, whereas adopting a finished weight as `w + (w' - w)` can round
differently in f64.

A plan can only be built from `VerifiedMethodSteps`, which only `verify_method_steps` creates, after
checking the steps bitwise against the weights the method produced. Python can't stop deliberate
misuse of private names, and `EditPlan` itself stays open for direct edits; this closes the method path.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .edit_plan import EditGraphError, EditPlan, tensor_state_digest
from .ir import WeightIRTransactionBuilder

_SEAL = object()


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


def _arguments(step: MethodEditStep, shape: Sequence[int]) -> dict[str, Any]:
    return {
        "operation_type": step.operation_type,
        "factor": step.factor,
        **step.operands,
        "region_start": (0, 0),
        "region_shape": (int(shape[0]), int(shape[1])),
    }


def _digest(name: str, weight: torch.Tensor) -> str:
    return tensor_state_digest({name: weight})


class VerifiedMethodSteps:
    """Steps that reproduced a method's weights, bound to the base tensors and results they were checked on."""

    __slots__ = ("base_digests", "expected_digests", "steps")

    def __init__(self, steps: tuple[MethodEditStep, ...], base_digests: dict[str, str], expected_digests: dict[str, str], *, seal: object = None) -> None:
        if seal is not _SEAL:
            raise TypeError("Use verify_method_steps to create VerifiedMethodSteps")
        self.steps = steps
        self.base_digests = base_digests
        self.expected_digests = expected_digests

    def __iter__(self) -> Iterator[MethodEditStep]:
        return iter(self.steps)

    def __len__(self) -> int:
        return len(self.steps)


def verify_method_steps(base: Any, steps: Sequence[MethodEditStep], expected: Mapping[str, torch.Tensor]) -> VerifiedMethodSteps:
    """Run `steps` through Rust without copying a model, and raise unless they bitwise reproduce `expected`.

    `base` is a model, or a mapping of tensor names to weights for methods that work on bare tensors.
    """
    tensors = base if isinstance(base, Mapping) else base.tensors
    working: dict[str, torch.Tensor] = {}
    for step in steps:
        if step.tensor not in tensors:
            raise EditGraphError(f"Unknown tensor {step.tensor!r}")
        current = working.get(step.tensor, tensors[step.tensor]).detach().cpu().to(torch.float64)
        tx = WeightIRTransactionBuilder.execute_matrix_edit(
            current, provenance_tag=f"edit_plan.{step.method}.v1", engine_name="studio.edit_plan", **_arguments(step, current.shape)
        )
        from .ir import require_reload_parity
        require_reload_parity(tx)
        working[step.tensor] = torch.frombuffer(bytearray(tx.edited_bytes), dtype=torch.float64).reshape(current.shape).clone()
    if not working or set(working) != set(expected):
        raise EditGraphError(f"The steps edit {sorted(working)}, but the method edited {sorted(expected)}")
    for name, weight in expected.items():
        if not torch.equal(working[name], weight.detach().cpu().to(torch.float64)):
            raise EditGraphError(f"The steps don't reproduce the method's weights for {name!r}")
    return VerifiedMethodSteps(
        tuple(steps),
        {name: _digest(name, tensors[name]) for name in working},
        {name: _digest(name, weight) for name, weight in working.items()},
        seal=_SEAL,
    )


def edit_plan_from_steps(base_model: Any, verified: VerifiedMethodSteps) -> EditPlan:
    """A previewed plan on `base_model`, or raise. Raw steps are refused: verify them first."""
    if not isinstance(verified, VerifiedMethodSteps):
        raise TypeError("A plan needs VerifiedMethodSteps; raw steps haven't been checked against the method's weights")
    for name, digest in verified.base_digests.items():
        if name not in base_model.tensors or _digest(name, base_model.tensors[name]) != digest:
            raise EditGraphError(f"These steps were verified on a different {name!r}")
    plan = EditPlan(base_model)
    for step in verified.steps:
        plan._add(step.tensor, step.method, _builder(step))
    edited = plan.preview().edited_tensors
    if {name: _digest(name, weight) for name, weight in edited.items()} != verified.expected_digests:
        raise EditGraphError("The plan doesn't reproduce the verified weights")
    return plan


def _builder(step: MethodEditStep) -> Any:
    return lambda w: _arguments(step, w.shape)


def verified_method_plan(base_model: Any, steps: Sequence[MethodEditStep], expected: Mapping[str, torch.Tensor]) -> EditPlan:
    """Verify `steps` against the method's weights, then build the plan."""
    return edit_plan_from_steps(base_model, verify_method_steps(base_model, tuple(steps), expected))
