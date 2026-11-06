"""Schemas for a Gym's frozen weight-edit proposal. See `docs/milestones.md`, Milestone 1.

Schemas, their JSON form and `compile_proposal`. Resolving a spec against a checkpoint lives in
`gpt_neox_adapter.py`. A spec is architecture-neutral; a resolved proposal is tied to one checkpoint. Raw factors
aren't portable between architectures, so operands live only on the resolved form.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

GYM_PROPOSAL_SCHEMA_VERSION = 1
SEMANTIC_ROLES = frozenset({"mlp_down"})
OPERATIONS = frozenset(
    {"scale", "add_delta", "low_rank_update", "subspace_projection", "replace_slice", "mask", "permute_axis", "linear_map"}
)
EVIDENCE_SOURCES = frozenset({"development_split", "scripted", "agent_session"})
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class GymProposalError(ValueError):
    """A proposal document is invalid or uses an unknown schema version."""


def _require_digest(value: Any, label: str) -> None:
    if not isinstance(value, str) or not _DIGEST.match(value):
        raise GymProposalError(f"{label} must be a sha256 digest")


def _freeze(value: Any, where: str) -> Any:
    """Deep-copy into read-only JSON-like values, so nobody holding the original can change a proposal."""
    if isinstance(value, Mapping):
        if not all(isinstance(k, str) for k in value):
            raise GymProposalError(f"{where} has a non-string key")
        return MappingProxyType({k: _freeze(v, f"{where}.{k}") for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v, where) for v in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise GymProposalError(f"{where} isn't finite")
    if value is None or isinstance(value, (str, int, float)):
        return value
    raise GymProposalError(f"{where} has unsupported type {type(value).__name__}")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(v) for v in value]
    return value


_REGION = frozenset({"region_start", "region_shape"})
# operation -> (required scalars, optional scalars, required matrices). Nothing else is accepted.
OPERATION_SCHEMAS: Mapping[str, tuple[frozenset[str], frozenset[str], frozenset[str]]] = MappingProxyType(
    {
        "scale": (frozenset({"factor"}), _REGION, frozenset()),
        "add_delta": (frozenset(), _REGION, frozenset({"delta_matrix"})),
        "low_rank_update": (frozenset(), _REGION | {"factor"}, frozenset({"left_factor", "right_factor"})),
        "subspace_projection": (frozenset({"projection_side"}), _REGION | {"factor"}, frozenset({"orthonormal_basis"})),
        "replace_slice": (frozenset(), _REGION, frozenset({"value_matrix"})),
        "mask": (frozenset(), _REGION, frozenset({"mask_matrix"})),
        "permute_axis": (frozenset({"axis", "permutation"}), _REGION, frozenset()),
        "linear_map": (frozenset({"projection_side"}), _REGION, frozenset({"map_matrix"})),
    }
)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _valid_scalar(name: str, value: Any) -> bool:
    if name == "factor":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if name == "projection_side":
        return value in ("left", "right")
    if name == "axis":
        return _is_int(value) and value in (0, 1)
    if name == "permutation":
        return isinstance(value, tuple) and all(_is_int(v) for v in value)
    return isinstance(value, tuple) and len(value) == 2 and all(_is_int(v) and v >= 0 for v in value)


def _check_scalars(operation: str, values: Mapping[str, Any]) -> None:
    required, optional, _ = OPERATION_SCHEMAS[operation]
    if unknown := set(values) - required - optional:
        raise GymProposalError(f"{operation} has unknown arguments: {sorted(unknown)}")
    if missing := required - set(values):
        raise GymProposalError(f"{operation} is missing {sorted(missing)}")
    for name, value in values.items():
        if not _valid_scalar(name, value):
            raise GymProposalError(f"{operation} has an invalid {name}: {_thaw(value)!r}")


def _check_operand_values(name: str, document: Mapping[str, Any]) -> None:
    rows, cols = document["shape"]
    try:
        raw = base64.b64decode(document["data_b64"], validate=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise GymProposalError(f"{name} has invalid data") from exc
    if document.get("dtype") != "f64" or len(raw) != rows * cols * 8:
        raise GymProposalError(f"{name} isn't f64 data of shape {(rows, cols)}")
    if not all(math.isfinite(v) for (v,) in struct.iter_unpack("<d", raw)):
        raise GymProposalError(f"{name} has non-finite values")


def _check_region(resolved: ResolvedOperation) -> tuple[int, int]:
    start, size = resolved.arguments.get("region_start"), resolved.arguments.get("region_shape")
    if (start is None) != (size is None):
        raise GymProposalError("region_start and region_shape must be given both or neither")
    if size is None:
        return resolved.shape
    if any(s <= 0 or a + s > d for a, s, d in zip(start, size, resolved.shape, strict=True)):
        raise GymProposalError(f"Region {start}, {size} is outside a tensor that has shape {resolved.shape}")
    return size


def check_operation(planned: ProposedOperation, resolved: ResolvedOperation) -> None:
    """Raise unless a resolved operation is exactly what its spec line asks for, with fitting operands."""
    if planned.operation != resolved.operation:
        raise GymProposalError("Resolved operations don't match the spec")
    _check_scalars(resolved.operation, resolved.arguments)
    needed = OPERATION_SCHEMAS[resolved.operation][2]
    if unknown := set(resolved.operands) - needed:
        raise GymProposalError(f"{resolved.operation} has unknown arguments: {sorted(unknown)}")
    if missing := needed - set(resolved.operands):
        raise GymProposalError(f"{resolved.operation} is missing {sorted(missing)}")
    if planned.parameters != resolved.arguments:
        raise GymProposalError(f"Arguments of {resolved.tensor!r} differ from the spec's parameters")
    layer = re.search(r"(?:^|\.)layers\.(\d+)\.", resolved.tensor)
    if layer is None or int(layer.group(1)) != planned.site.layer:
        raise GymProposalError(f"{resolved.tensor!r} isn't the tensor for {planned.site!r}")
    shapes = {k: v.get("shape") for k, v in resolved.operands.items()}
    if any(not isinstance(s, tuple) or len(s) != 2 or not all(_is_int(d) and d > 0 for d in s) for s in shapes.values()):
        raise GymProposalError(f"Operands of {resolved.tensor!r} must be matrices with a shape")
    for name, document in resolved.operands.items():
        _check_operand_values(name, document)
    rows, cols = _check_region(resolved)
    for name in ("delta_matrix", "value_matrix", "mask_matrix"):
        if name in shapes and shapes[name] != (rows, cols):
            raise GymProposalError(f"{name} has shape {shapes[name]}, not the edited region's {(rows, cols)}")
    if resolved.operation == "low_rank_update":
        left, right = shapes["left_factor"], shapes["right_factor"]
        if left[0] != rows or right[1] != cols or left[1] != right[0]:
            raise GymProposalError(f"Low-rank factors {left} and {right} don't fit the region {(rows, cols)}")
        if left[1] > min(rows, cols):
            raise GymProposalError(f"Low-rank factors have rank {left[1]}, above the region's {min(rows, cols)}")
    side = resolved.arguments.get("projection_side")
    span = rows if side == "left" else cols
    if "map_matrix" in shapes and shapes["map_matrix"] != (span, span):
        raise GymProposalError(f"map_matrix has shape {shapes['map_matrix']}, but a {side} map needs {(span, span)}")
    if "orthonormal_basis" in shapes and (shapes["orthonormal_basis"][0] != span or shapes["orthonormal_basis"][1] > span):
        raise GymProposalError(f"orthonormal_basis has shape {shapes['orthonormal_basis']}, which doesn't fit a {side} span of {span}")
    if resolved.operation == "subspace_projection" and not 0 <= resolved.arguments.get("factor", 1.0) <= 1:
        raise GymProposalError("A projection's factor must be between 0 and 1")
    if resolved.operation == "permute_axis":
        length = (rows, cols)[resolved.arguments["axis"]]
        if sorted(resolved.arguments["permutation"]) != list(range(length)):
            raise GymProposalError(f"permutation isn't a permutation of axis length {length}")


@dataclass(frozen=True)
class SemanticSite:
    role: str
    layer: int

    def __post_init__(self) -> None:
        if self.role not in SEMANTIC_ROLES:
            raise GymProposalError(f"Unknown semantic role {self.role!r}")
        if isinstance(self.layer, bool) or not isinstance(self.layer, int) or self.layer < 0:
            raise GymProposalError("layer must be a non-negative integer")


@dataclass(frozen=True)
class ProposedOperation:
    site: SemanticSite
    operation: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.operation not in OPERATIONS:
            raise GymProposalError(f"Unknown operation {self.operation!r}")
        object.__setattr__(self, "parameters", _freeze(self.parameters, "parameters"))
        _check_scalars(self.operation, self.parameters)


@dataclass(frozen=True)
class GymWeightProposalSpec:
    """Architecture-neutral: says what to edit by role, with no tensor names or factors."""

    proposal_id: str
    operations: tuple[ProposedOperation, ...]
    constraints: Mapping[str, Any]
    evidence_source: str
    synthesis_split_digest: str
    schema_version: int = GYM_PROPOSAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "operations", tuple(self.operations))
        object.__setattr__(self, "constraints", _freeze(self.constraints, "constraints"))
        if self.schema_version != GYM_PROPOSAL_SCHEMA_VERSION:
            raise GymProposalError(f"Unknown schema version {self.schema_version!r}")
        if not self.proposal_id.strip():
            raise GymProposalError("proposal_id must be non-empty")
        if not self.operations:
            raise GymProposalError("A proposal needs at least one operation")
        if self.evidence_source not in EVIDENCE_SOURCES:
            raise GymProposalError(f"Unknown evidence source {self.evidence_source!r}")
        _require_digest(self.synthesis_split_digest, "synthesis_split_digest")


@dataclass(frozen=True)
class ResolvedOperation:
    tensor: str
    shape: tuple[int, int]
    operation: str
    arguments: Mapping[str, Any]
    operands: Mapping[str, Mapping[str, Any]]

    def __post_init__(self) -> None:
        if self.operation not in OPERATIONS:
            raise GymProposalError(f"Unknown operation {self.operation!r}")
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "arguments", _freeze(self.arguments, "arguments"))
        object.__setattr__(self, "operands", _freeze(self.operands, "operands"))
        if len(self.shape) != 2 or any(d <= 0 for d in self.shape):
            raise GymProposalError(f"Invalid tensor shape {self.shape!r}")


@dataclass(frozen=True)
class ResolvedGymWeightProposal:
    """Checkpoint-specific and frozen: everything needed to replay, including expected digests."""

    spec: GymWeightProposalSpec
    architecture: str
    base_checkpoint_digest: str
    operations: tuple[ResolvedOperation, ...]
    expected_candidate_digest: str
    schema_version: int = GYM_PROPOSAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "operations", tuple(self.operations))
        if self.schema_version != GYM_PROPOSAL_SCHEMA_VERSION:
            raise GymProposalError(f"Unknown schema version {self.schema_version!r}")
        _require_digest(self.base_checkpoint_digest, "base_checkpoint_digest")
        _require_digest(self.expected_candidate_digest, "expected_candidate_digest")
        if not self.architecture.strip():
            raise GymProposalError("architecture must be non-empty")
        if not self.operations:
            raise GymProposalError("A resolved proposal needs at least one operation")
        if len(self.operations) != len(self.spec.operations):
            raise GymProposalError("Resolved operations don't match the spec")
        for planned, resolved in zip(self.spec.operations, self.operations, strict=True):
            check_operation(planned, resolved)


def _canonical_bytes(document: Any) -> bytes:
    try:
        return json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GymProposalError(f"Proposal isn't canonical JSON: {exc}") from exc


def _digest_document(document: Any) -> str:
    return f"sha256:{hashlib.sha256(_canonical_bytes(document)).hexdigest()}"


def proposal_to_document(proposal: ResolvedGymWeightProposal) -> dict[str, Any]:
    """The JSON form of a frozen proposal. All mappings and tuples become plain JSON values."""
    spec = proposal.spec
    document = {
        "kind": "nf_resolved_gym_proposal_v1",
        "schema_version": proposal.schema_version,
        "spec": {
            "schema_version": spec.schema_version,
            "proposal_id": spec.proposal_id,
            "operations": [
                {"role": o.site.role, "layer": o.site.layer, "operation": o.operation, "parameters": _thaw(o.parameters)} for o in spec.operations
            ],
            "constraints": _thaw(spec.constraints),
            "evidence_source": spec.evidence_source,
            "synthesis_split_digest": spec.synthesis_split_digest,
        },
        "architecture": proposal.architecture,
        "base_checkpoint_digest": proposal.base_checkpoint_digest,
        "operations": [
            {"tensor": o.tensor, "shape": list(o.shape), "operation": o.operation, "arguments": _thaw(o.arguments), "operands": _thaw(o.operands)}
            for o in proposal.operations
        ],
        "expected_candidate_digest": proposal.expected_candidate_digest,
    }
    return json.loads(_canonical_bytes(document))


def proposal_from_document(document: Mapping[str, Any]) -> ResolvedGymWeightProposal:
    """Rebuild a frozen proposal, refusing unknown kinds, versions and any malformed part."""
    if not isinstance(document, Mapping) or document.get("kind") != "nf_resolved_gym_proposal_v1":
        raise GymProposalError("Not an nf_resolved_gym_proposal_v1 document")
    try:
        raw = document["spec"]
        spec = GymWeightProposalSpec(
            proposal_id=raw["proposal_id"],
            operations=tuple(
                ProposedOperation(SemanticSite(o["role"], o["layer"]), o["operation"], o["parameters"]) for o in raw["operations"]
            ),
            constraints=raw["constraints"],
            evidence_source=raw["evidence_source"],
            synthesis_split_digest=raw["synthesis_split_digest"],
            schema_version=raw["schema_version"],
        )
        return ResolvedGymWeightProposal(
            spec=spec,
            architecture=document["architecture"],
            base_checkpoint_digest=document["base_checkpoint_digest"],
            operations=tuple(
                ResolvedOperation(o["tensor"], tuple(o["shape"]), o["operation"], o["arguments"], o["operands"]) for o in document["operations"]
            ),
            expected_candidate_digest=document["expected_candidate_digest"],
            schema_version=document["schema_version"],
        )
    except (KeyError, TypeError, AttributeError) as exc:
        raise GymProposalError(f"Malformed proposal document: {exc!r}") from exc


def proposal_digest(proposal: ResolvedGymWeightProposal) -> str:
    """Content digest of a frozen proposal, independent of mapping order and file formatting."""
    return _digest_document(proposal_to_document(proposal))


def compile_proposal(proposal: ResolvedGymWeightProposal) -> dict[str, Any]:
    """Compile a frozen proposal into its ordered edit graph: the same input always gives equal output.

    This needs no checkpoint, so it checks structure only. `gpt_neox_adapter.compile_to_edit_graph`
    executes these steps against a base model and refuses a result that differs from them.
    """
    document = proposal_to_document(proposal)
    steps = [
        {
            "id": f"step_{index}",
            "depends_on": [f"step_{index - 1}"] if index else [],
            "tensor": operation["tensor"],
            "shape": operation["shape"],
            "operation": operation["operation"],
            "arguments": operation["arguments"],
            "operands": operation["operands"],
        }
        for index, operation in enumerate(document["operations"])
    ]
    body = {
        "schema_version": 1,
        "kind": "nf_compiled_gym_proposal_v1",
        "proposal_digest": _digest_document(document),
        "architecture": document["architecture"],
        "base_checkpoint_digest": document["base_checkpoint_digest"],
        "expected_candidate_digest": document["expected_candidate_digest"],
        "steps": steps,
    }
    return {**body, "compiled_digest": _digest_document(body)}
