"""Schemas for a Gym's frozen weight-edit proposal. See `docs/milestones.md`, Milestone 1.

Schema and validation only. Compiling a proposal into an edit graph isn't implemented yet.
A spec is architecture-neutral; a resolved proposal is tied to one checkpoint. Raw factors
aren't portable between architectures, so operands live only on the resolved form.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
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
            if planned.operation != resolved.operation:
                raise GymProposalError("Resolved operations don't match the spec")
