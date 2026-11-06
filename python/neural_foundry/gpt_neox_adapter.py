"""Milestone 1 condition 2: resolve architecture-neutral proposals against a GPT-NeoX checkpoint.

This belongs to the proposer's side. Replay and evaluation never import it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .edit_plan import EditConstraints, EditPlan, _encode_matrix, tensor_state_digest
from .export_formats import _extract_model_state_tensors
from .gym_proposal import (
    GymProposalError,
    GymWeightProposalSpec,
    ResolvedGymWeightProposal,
    ResolvedOperation,
    SemanticSite,
    check_operation,
    compile_proposal,
)
from .records import NativeError

ARCHITECTURE = "gpt_neox"
_ROLE_TEMPLATES = {"mlp_down": "gpt_neox.layers.{layer}.mlp.dense_4h_to_h.weight"}


@dataclass(frozen=True)
class ResolvedSite:
    tensor: str
    shape: tuple[int, int]


def resolve_site(model: Any, site: SemanticSite) -> ResolvedSite:
    """Map a semantic site to its tensor, checking the role's shape against the model's own config."""
    config = model.config
    if config.get("model_type") != ARCHITECTURE:
        raise GymProposalError(f"Not a GPT-NeoX model: {config.get('model_type')!r}")
    template = _ROLE_TEMPLATES.get(site.role)
    if template is None:
        raise GymProposalError(f"GPT-NeoX has no resolver for role {site.role!r}")
    if site.layer >= int(config["num_hidden_layers"]):
        raise GymProposalError(f"Layer {site.layer} is outside the model's {config['num_hidden_layers']} layers")
    tensor = template.format(layer=site.layer)
    if tensor not in model.tensors:
        raise GymProposalError(f"Checkpoint has no tensor {tensor!r}")
    # An MLP down projection maps intermediate_size back to hidden_size, and its weight is stored (out, in).
    expected = (int(config["hidden_size"]), int(config["intermediate_size"]))
    actual = tuple(int(d) for d in model.tensors[tensor].shape)
    if actual != expected:
        raise GymProposalError(f"{tensor!r} has shape {actual}, but the {site.role} role needs {expected}")
    return ResolvedSite(tensor, expected)


def _constraints(spec: GymWeightProposalSpec) -> EditConstraints:
    try:
        return EditConstraints(**{k: tuple(v) if isinstance(v, list) else v for k, v in spec.constraints.items()})
    except TypeError as exc:
        raise GymProposalError(f"Unknown proposal constraint: {exc}") from exc


def resolve_proposal(
    spec: GymWeightProposalSpec, model: Any, operands: Sequence[Mapping[str, torch.Tensor]]
) -> ResolvedGymWeightProposal:
    """Freeze a spec against this checkpoint. `operands` holds one mapping per spec operation.

    Scalar arguments come from the spec's `parameters`. Factors come separately, because they aren't
    portable between architectures. The expected digest is what the Rust executor produces now, and
    replay has to reproduce it later.
    """
    if len(operands) != len(spec.operations):
        raise GymProposalError(f"Got operands for {len(operands)} operations, but the spec has {len(spec.operations)}")
    resolved: list[ResolvedOperation] = []
    plan = EditPlan(model, _constraints(spec))
    for planned, matrices in zip(spec.operations, operands, strict=True):
        site = resolve_site(model, planned.site)
        encoded = {k: _encode_matrix(v) for k, v in matrices.items()}
        operation = ResolvedOperation(site.tensor, site.shape, planned.operation, planned.parameters, encoded)
        check_operation(planned, operation)
        plan._add_resolved(planned, operation)
        resolved.append(operation)
    try:
        candidate = plan.preview()
    except (NativeError, RuntimeError) as exc:
        raise GymProposalError(f"The executor refused this proposal: {exc}") from exc
    return ResolvedGymWeightProposal(
        spec=spec,
        architecture=ARCHITECTURE,
        base_checkpoint_digest=tensor_state_digest(_extract_model_state_tensors(model)),
        operations=tuple(resolved),
        expected_candidate_digest=candidate.state_digest,
    )


def validate_resolved(model: Any, proposal: ResolvedGymWeightProposal) -> None:
    """Re-check a proposal read from disk: each operation must be what its spec line resolves to here."""
    if proposal.architecture != ARCHITECTURE:
        raise GymProposalError(f"Not a GPT-NeoX proposal: {proposal.architecture!r}")
    for planned, operation in zip(proposal.spec.operations, proposal.operations, strict=True):
        site = resolve_site(model, planned.site)
        if operation.tensor != site.tensor or tuple(operation.shape) != site.shape:
            raise GymProposalError(f"{operation.tensor!r} {tuple(operation.shape)} isn't the resolution of {planned.site!r}")


def compile_to_edit_graph(model: Any, proposal: ResolvedGymWeightProposal) -> dict[str, Any]:
    """Execute a frozen proposal into an `nf_edit_graph_v1`, refusing any drift from `compile_proposal`."""
    validate_resolved(model, proposal)
    compiled = compile_proposal(proposal)
    graph = EditPlan.from_resolved_proposal(model, proposal).to_graph()
    if len(graph["steps"]) != len(compiled["steps"]):
        raise GymProposalError("The edit graph has different steps from the compiled proposal")
    for executed, planned in zip(graph["steps"], compiled["steps"], strict=True):
        same_arguments = all(json.loads(json.dumps(executed["arguments"].get(k))) == v for k, v in planned["arguments"].items())
        if (
            executed["id"] != planned["id"]
            or executed["depends_on"] != planned["depends_on"]
            or executed["tensor"] != planned["tensor"]
            or executed["operation"] != planned["operation"]
            or executed["operands"] != planned["operands"]
            or not same_arguments
        ):
            raise GymProposalError(f"Edit graph {executed['id']} differs from the compiled proposal")
    return graph

