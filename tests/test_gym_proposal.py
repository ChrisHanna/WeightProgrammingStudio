"""Milestone 1 contract: the proposal schemas and the compiler's determinism."""

from __future__ import annotations

from dataclasses import fields

import pytest

from neural_foundry import gym_proposal as gp

DIGEST = "sha256:" + "a" * 64


def _spec(**changes):  # type: ignore[no-untyped-def]
    values = {
        "proposal_id": "p1",
        "operations": (gp.ProposedOperation(gp.SemanticSite("mlp_down", 5), "low_rank_update"),),
        "constraints": {"max_changed_parameters": 1000},
        "evidence_source": "development_split",
        "synthesis_split_digest": DIGEST,
        **changes,
    }
    return gp.GymWeightProposalSpec(**values)


def _operation(operation: str = "low_rank_update") -> gp.ResolvedOperation:
    factors = {name: {"shape": shape, "dtype": "f64", "data_b64": "A" * 22 + "=="} for name, shape in (("left_factor", [2, 1]), ("right_factor", [1, 2]))}
    return gp.ResolvedOperation("gpt_neox.layers.5.mlp.w", (2, 2), operation, {}, factors if operation == "low_rank_update" else {})


def _resolved(**changes):  # type: ignore[no-untyped-def]
    values = {
        "spec": _spec(),
        "architecture": "gpt_neox",
        "base_checkpoint_digest": DIGEST,
        "operations": (_operation(),),
        "expected_candidate_digest": DIGEST,
        **changes,
    }
    return gp.ResolvedGymWeightProposal(**values)


def test_valid_documents_are_accepted() -> None:
    assert _resolved().spec.proposal_id == "p1"


def test_spec_is_architecture_neutral() -> None:
    names = {f.name for f in fields(gp.GymWeightProposalSpec)} | {f.name for f in fields(gp.ProposedOperation)}
    assert not names & {"tensor", "operands", "architecture", "base_checkpoint_digest"}


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": 2},
        {"proposal_id": " "},
        {"operations": ()},
        {"evidence_source": "holdout"},
        {"synthesis_split_digest": "nope"},
    ],
)
def test_invalid_specs_are_rejected(changes: dict) -> None:
    with pytest.raises(gp.GymProposalError):
        _spec(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": 0},
        {"base_checkpoint_digest": "x"},
        {"expected_candidate_digest": "x"},
        {"operations": ()},
        {"operations": (_operation("scale"),)},
        {"operations": (_operation(), _operation())},
        {"architecture": ""},
    ],
)
def test_invalid_resolved_proposals_are_rejected(changes: dict) -> None:
    with pytest.raises(gp.GymProposalError):
        _resolved(**changes)


@pytest.mark.parametrize(
    "build",
    [
        lambda: gp.SemanticSite("unknown", 1),
        lambda: gp.SemanticSite("mlp_down", -1),
        lambda: gp.SemanticSite("mlp_down", True),
        lambda: gp.ProposedOperation(gp.SemanticSite("mlp_down", 1), "construct_recipe"),
        lambda: gp.ResolvedOperation("t", (0, 2), "scale", {}, {}),
    ],
)
def test_invalid_parts_are_rejected(build) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(gp.GymProposalError):
        build()


def test_compilation_is_deterministic() -> None:
    assert gp.compile_proposal(_resolved()) == gp.compile_proposal(_resolved())
