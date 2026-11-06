"""Milestone 1 conditions 2-5 on the mini model, 8-9 as behaviour, and the whole run on real Pythia-14M."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from neural_foundry import gym_proposal as gp
from neural_foundry import pythia
from neural_foundry import run_record as rr
from neural_foundry.edit_plan import _decode_matrix, _encode_matrix, apply_edit_graph
from neural_foundry.gpt_neox_adapter import compile_to_edit_graph, resolve_proposal, resolve_site, validate_resolved
from neural_foundry.patch_ux import verify_patch_signature
from neural_foundry.records import canonicalize, content_digest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "qualify_milestone1.py"
EVIDENCE = ROOT / "docs" / "evidence" / "milestone1"
CHECKPOINT = Path(os.environ.get("NF_PYTHIA_14M_DIR", "/tmp/pythia-14m"))
DIGEST = "sha256:" + "a" * 64


@pytest.fixture
def model():  # type: ignore[no-untyped-def]
    return pythia.build_mini_pythia_model()


def _spec(layer: int = 1, operation: str = "scale", parameters: dict | None = None, **changes) -> gp.GymWeightProposalSpec:  # type: ignore[no-untyped-def,type-arg]
    site = gp.SemanticSite("mlp_down", layer)
    planned = gp.ProposedOperation(site, operation, {"factor": 0.5} if parameters is None else parameters)
    return gp.GymWeightProposalSpec(**{"proposal_id": "p", "operations": (planned,), "constraints": {}, "evidence_source": "scripted", "synthesis_split_digest": DIGEST, **changes})


def _low_rank(model):  # type: ignore[no-untyped-def]
    rows, cols = model.tensors[model.mlp_down_proj_site(1)].shape
    left = 0.01 * torch.arange(1.0, rows + 1, dtype=torch.float64).unsqueeze(1)
    right = torch.arange(1.0, cols + 1, dtype=torch.float64).unsqueeze(0)
    return resolve_proposal(_spec(operation="low_rank_update", parameters={"factor": 1.0}), model, [{"left_factor": left, "right_factor": right}])


# ---- Condition 2: the resolver.
def test_resolver_finds_the_role_tensor_and_shape(model) -> None:  # type: ignore[no-untyped-def]
    site = resolve_site(model, gp.SemanticSite("mlp_down", 1))
    assert site.tensor == "gpt_neox.layers.1.mlp.dense_4h_to_h.weight"
    assert site.shape == (int(model.config["hidden_size"]), int(model.config["intermediate_size"]))


def test_resolver_refuses_wrong_architecture_layer_missing_tensor_and_wrong_shape(model) -> None:  # type: ignore[no-untyped-def]
    site = gp.SemanticSite("mlp_down", 1)
    with pytest.raises(gp.GymProposalError, match="outside"):
        resolve_site(model, gp.SemanticSite("mlp_down", int(model.config["num_hidden_layers"])))
    name = model.mlp_down_proj_site(1)
    weight = model.tensors[name]
    model.tensors[name] = weight.T  # An up projection's shape in the down projection's place.
    with pytest.raises(gp.GymProposalError, match="role needs"):
        resolve_site(model, site)
    del model.tensors[name]
    with pytest.raises(gp.GymProposalError, match="no tensor"):
        resolve_site(model, site)
    model.tensors[name] = weight
    model.config = {**model.config, "model_type": "qwen2"}
    with pytest.raises(gp.GymProposalError, match="Not a GPT-NeoX"):
        resolve_site(model, site)


def test_resolver_refuses_bad_operands_and_parameters(model) -> None:  # type: ignore[no-untyped-def]
    rows, cols = model.tensors[model.mlp_down_proj_site(1)].shape
    low_rank = _spec(operation="low_rank_update", parameters={})
    with pytest.raises(gp.GymProposalError, match="missing"):
        resolve_proposal(low_rank, model, [{"left_factor": torch.ones(rows, 1)}])
    with pytest.raises(gp.GymProposalError, match="don't fit"):
        resolve_proposal(low_rank, model, [{"left_factor": torch.ones(rows, 1), "right_factor": torch.ones(2, cols)}])
    with pytest.raises(gp.GymProposalError, match="edited region"):
        resolve_proposal(_spec(operation="add_delta", parameters={}), model, [{"delta_matrix": torch.ones(1, 1)}])
    with pytest.raises(gp.GymProposalError, match="unknown arguments"):
        resolve_proposal(_spec(parameters={"factor": 0.5, "tensor": "x"}), model, [{}])
    with pytest.raises(gp.GymProposalError, match="operands for"):
        resolve_proposal(_spec(), model, [])


def test_frozen_proposal_that_disagrees_with_its_spec_is_refused(model) -> None:  # type: ignore[no-untyped-def]
    good = resolve_proposal(_spec(), model, [{}])
    validate_resolved(model, good)
    operation = good.operations[0]
    for bad in (
        replace(operation, tensor=model.mlp_down_proj_site(0)),
        replace(operation, shape=(1, 1)),
        replace(operation, arguments={"factor": 0.9}),
    ):
        with pytest.raises(gp.GymProposalError):
            validate_resolved(model, replace(good, operations=(bad,)))


# ---- Review finding 1: a proposal can't change after it is made.
def test_mutating_the_original_or_stored_values_cannot_change_a_proposal(model) -> None:  # type: ignore[no-untyped-def]
    limits = {"forbidden_tensors": ["x"]}
    region = {"factor": 0.5, "region_start": [0, 0], "region_shape": [1, 1]}
    spec = _spec(parameters=region, constraints=limits)
    proposal = resolve_proposal(spec, model, [{}])
    before = gp.proposal_digest(proposal)
    limits["forbidden_tensors"].append("y")
    limits["new"] = 1
    region["factor"] = 9.0
    region["region_shape"][0] = 2
    assert gp.proposal_digest(proposal) == before
    operation = proposal.operations[0]
    for attempt in (
        lambda: spec.constraints.__setitem__("new", 1),
        lambda: spec.operations[0].parameters.__setitem__("factor", 1.0),
        lambda: operation.arguments.__setitem__("factor", 1.0),
        lambda: spec.constraints["forbidden_tensors"].append("y"),
    ):
        with pytest.raises((TypeError, AttributeError)):
            attempt()
    assert gp.proposal_digest(proposal) == before


def test_mutating_operand_documents_cannot_change_a_proposal(model) -> None:  # type: ignore[no-untyped-def]
    proposal = _low_rank(model)
    operation = proposal.operations[0]
    document = {k: dict(v) for k, v in gp.proposal_to_document(proposal)["operations"][0]["operands"].items()}
    frozen = gp.ResolvedOperation(operation.tensor, operation.shape, operation.operation, {"factor": 1.0}, document)
    copy = replace(proposal, operations=(frozen,))
    before = gp.proposal_digest(copy)
    document["left_factor"]["data_b64"] = "tampered"
    assert gp.proposal_digest(copy) == before
    with pytest.raises(TypeError):
        frozen.operands["left_factor"]["data_b64"] = "tampered"  # type: ignore[index]
    gp.proposal_to_document(copy)["spec"]["constraints"]["new"] = 1
    assert gp.proposal_digest(copy) == before


def test_values_that_are_not_plain_data_are_refused() -> None:
    with pytest.raises(gp.GymProposalError, match="unsupported"):
        _spec(constraints={"x": torch.zeros(1)})
    with pytest.raises(gp.GymProposalError, match="finite"):
        _spec(constraints={"x": float("inf")})


# ---- Review finding 2: spec parameters and resolved arguments must agree.
@pytest.mark.parametrize(
    ("operation", "parameters", "match"),
    [
        ("scale", {}, "missing"),
        ("scale", {"factor": 1.0, "tensor": "t"}, "unknown"),
        ("scale", {"factor": True}, "invalid factor"),
        ("scale", {"factor": "big"}, "invalid factor"),
        ("subspace_projection", {}, "missing"),
        ("subspace_projection", {"projection_side": "up"}, "invalid projection_side"),
        ("permute_axis", {"axis": 2, "permutation": [0]}, "invalid axis"),
        ("permute_axis", {"axis": 0}, "missing"),
        ("add_delta", {"factor": 1.0}, "unknown"),
    ],
)
def test_spec_parameters_follow_the_operation_schema(operation: str, parameters: dict, match: str) -> None:  # type: ignore[type-arg]
    with pytest.raises(gp.GymProposalError, match=match):
        gp.ProposedOperation(gp.SemanticSite("mlp_down", 1), operation, parameters)


def test_every_operation_has_a_schema() -> None:
    assert set(gp.OPERATION_SCHEMAS) == set(gp.OPERATIONS)


def test_resolved_proposal_must_match_its_spec_semantically(model) -> None:  # type: ignore[no-untyped-def]
    good = resolve_proposal(_spec(), model, [{}])
    operation = good.operations[0]

    def refused(match: str, **changes) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(gp.GymProposalError, match=match):
            replace(good, operations=(replace(operation, **changes),))

    refused("differ from", arguments={"factor": 0.9})
    refused("missing", arguments={})
    refused("unknown", arguments={"factor": 0.5, "axis": 0})
    refused("unknown", operands={"delta_matrix": {"shape": [1, 1]}})
    low_rank = _low_rank(model)
    factors = low_rank.operations[0]
    with pytest.raises(gp.GymProposalError, match="missing"):
        replace(low_rank, operations=(replace(factors, operands={"left_factor": factors.operands["left_factor"]}),))
    with pytest.raises(gp.GymProposalError, match="shape"):
        replace(low_rank, operations=(replace(factors, operands={"left_factor": {}, "right_factor": {}}),))


# ---- Review finding 3: the Gym emits frozen proposals.
def test_gym_report_keeps_all_frozen_proposals_and_selects_one() -> None:
    base = pythia.build_mini_pythia_model()
    report = pythia.run_pythia_gym_task_learning_study(base, target_strength=85.0)
    assert [c.branch_id for c in report.gym_proposals] == [b.branch_id for b in report.branches]
    assert [c.branch_id for c in report.gym_proposals if c.selected] == [report.selected_branch_id]
    assert report.selected_proposal.spec.evidence_source == "development_split"
    for candidate in report.gym_proposals:
        graph = compile_to_edit_graph(base, candidate.proposal)
        assert apply_edit_graph(pythia.build_mini_pythia_model(), json.loads(json.dumps(graph))).state_digest == candidate.proposal.expected_candidate_digest


def test_selected_proposal_reproduces_the_weights_the_gym_measured() -> None:
    base = pythia.build_mini_pythia_model()
    report = pythia.run_pythia_gym_task_learning_study(base, target_strength=85.0)
    replayed = apply_edit_graph(base, compile_to_edit_graph(base, report.selected_proposal)).model
    for prediction in report.holdout_after:
        assert replayed.evaluate_prompt(prediction.prompt).top1_logit == prediction.top1_logit


def test_gym_proposals_do_not_depend_on_the_holdout() -> None:
    gym = pythia.ALGORITHMIC_THREE_SKILL_GYM
    other = replace(gym, holdout_rules=(("Number 3 has parity", "\u0120odd"),))
    one = pythia.run_pythia_gym_task_learning_study(pythia.build_mini_pythia_model(), gym=gym, target_strength=85.0)
    two = pythia.run_pythia_gym_task_learning_study(pythia.build_mini_pythia_model(), gym=other, target_strength=85.0)
    assert [gp.proposal_digest(c.proposal) for c in one.gym_proposals] == [gp.proposal_digest(c.proposal) for c in two.gym_proposals]


# ---- Refined review: per-operation schemas against the target shape.
def _bad_resolved(model, operation, arguments, operands):  # type: ignore[no-untyped-def]
    site = gp.SemanticSite("mlp_down", 1)
    tensor = model.mlp_down_proj_site(1)
    shape = tuple(model.tensors[tensor].shape)
    spec = _spec(operations=(gp.ProposedOperation(site, operation, arguments),))
    encoded = {k: _encode_matrix(v) for k, v in operands.items()}
    gp.ResolvedGymWeightProposal(spec, "gpt_neox", DIGEST, (gp.ResolvedOperation(tensor, shape, operation, arguments, encoded),), DIGEST)


def test_operation_schemas_reject_incompatible_values(model) -> None:  # type: ignore[no-untyped-def]
    rows, cols = model.tensors[model.mlp_down_proj_site(1)].shape
    nan = torch.ones(rows, 1, dtype=torch.float64)
    nan[0, 0] = float("nan")
    right = torch.ones(1, cols, dtype=torch.float64)
    cases = [
        ("low_rank_update", {}, {"left_factor": nan, "right_factor": right}, "non-finite"),
        ("low_rank_update", {}, {"left_factor": torch.ones(rows, cols + 1), "right_factor": torch.ones(cols + 1, cols)}, "rank"),
        ("low_rank_update", {}, {"left_factor": torch.ones(rows, 1), "right_factor": torch.ones(2, cols)}, "don't fit"),
        ("scale", {"factor": 1.0, "region_start": [0, 0]}, {}, "both"),
        ("scale", {"factor": 1.0, "region_start": [0, 0], "region_shape": [rows + 1, 1]}, {}, "outside"),
        ("scale", {"factor": 1.0, "region_start": [0, 0], "region_shape": [0, 1]}, {}, "outside"),
        ("permute_axis", {"axis": 0, "permutation": [0, 0]}, {}, "permutation"),
        ("permute_axis", {"axis": 1, "permutation": list(range(rows))}, {}, "permutation"),
        ("linear_map", {"projection_side": "left"}, {"map_matrix": torch.eye(cols)}, "map_matrix"),
        ("subspace_projection", {"projection_side": "left"}, {"orthonormal_basis": torch.ones(rows + 1, 1)}, "orthonormal_basis"),
        ("subspace_projection", {"projection_side": "left", "factor": 2.0}, {"orthonormal_basis": torch.ones(rows, 1)}, "between 0 and 1"),
        ("mask", {}, {"mask_matrix": torch.ones(1, 1)}, "mask_matrix"),
        ("replace_slice", {}, {"value_matrix": torch.ones(1, 1)}, "value_matrix"),
    ]
    for operation, arguments, operands, match in cases:
        with pytest.raises(gp.GymProposalError, match=match):
            _bad_resolved(model, operation, arguments, operands)


def test_every_operation_still_resolves_and_replays(model) -> None:  # type: ignore[no-untyped-def]
    rows, cols = model.tensors[model.mlp_down_proj_site(1)].shape
    unit_left = torch.zeros(rows, 1, dtype=torch.float64)
    unit_left[0, 0] = 1.0
    unit_right = torch.zeros(cols, 1, dtype=torch.float64)
    unit_right[0, 0] = 1.0
    cases = [
        ("scale", {"factor": 0.5, "region_start": [0, 0], "region_shape": [2, 2]}, {}),
        ("add_delta", {}, {"delta_matrix": torch.ones(rows, cols)}),
        ("low_rank_update", {"factor": 1.0}, {"left_factor": torch.ones(rows, 1), "right_factor": torch.ones(1, cols)}),
        ("subspace_projection", {"projection_side": "left", "factor": 1.0}, {"orthonormal_basis": unit_left}),
        ("subspace_projection", {"projection_side": "right", "factor": 1.0}, {"orthonormal_basis": unit_right}),
        ("replace_slice", {}, {"value_matrix": torch.ones(rows, cols)}),
        ("mask", {}, {"mask_matrix": torch.ones(rows, cols)}),
        ("permute_axis", {"axis": 0, "permutation": list(reversed(range(rows)))}, {}),
        ("linear_map", {"projection_side": "left"}, {"map_matrix": 2.0 * torch.eye(rows)}),
        ("linear_map", {"projection_side": "right"}, {"map_matrix": 2.0 * torch.eye(cols)}),
    ]
    for operation, arguments, operands in cases:
        proposal = resolve_proposal(_spec(operation=operation, parameters=arguments), model, [operands])
        graph = compile_to_edit_graph(model, proposal)
        assert apply_edit_graph(model, json.loads(json.dumps(graph))).state_digest == proposal.expected_candidate_digest


# ---- Refined review: each resolved operation is bound to its site.
def test_changing_strength_site_or_operand_fails_verification(model) -> None:  # type: ignore[no-untyped-def]
    good = _low_rank(model)
    operation = good.operations[0]
    planned = good.spec.operations[0]
    with pytest.raises(gp.GymProposalError, match="differ from"):
        replace(good, operations=(replace(operation, arguments={"factor": 2.0}),))
    with pytest.raises(gp.GymProposalError, match="isn't the tensor"):
        replace(good, operations=(replace(operation, tensor=model.mlp_down_proj_site(0)),))
    # A site moved consistently in spec and tensor resolves differently, so the frozen digest no longer holds.
    moved = replace(planned, site=gp.SemanticSite("mlp_down", 0))
    moved_proposal = replace(
        good, spec=replace(good.spec, operations=(moved,)), operations=(replace(operation, tensor=model.mlp_down_proj_site(0)),)
    )
    with pytest.raises(gp.GymProposalError, match="not the expected"):
        compile_to_edit_graph(model, moved_proposal)
    # Strength moved consistently in spec and arguments.
    stronger = replace(planned, parameters={"factor": 2.0})
    with pytest.raises(gp.GymProposalError, match="not the expected"):
        compile_to_edit_graph(model, replace(good, spec=replace(good.spec, operations=(stronger,)), operations=(replace(operation, arguments={"factor": 2.0}),)))
    # An operand with the same shape and different values.
    doc = {k: gp._thaw(v) for k, v in operation.operands.items()}
    doc["left_factor"] = _encode_matrix(2.0 * _decode_matrix(doc["left_factor"]))
    changed = replace(good, operations=(replace(operation, operands=doc),))
    with pytest.raises(gp.GymProposalError, match="not the expected"):
        compile_to_edit_graph(model, changed)
    assert gp.proposal_digest(changed) != gp.proposal_digest(good)


# ---- Refined review: each branch's proposal replays to what the Gym reported for it.
def test_every_branch_proposal_replays_to_its_reported_scores() -> None:
    base = pythia.build_mini_pythia_model()
    gym = pythia.ALGORITHMIC_THREE_SKILL_GYM
    report = pythia.run_pythia_gym_task_learning_study(base, gym=gym, target_strength=85.0)
    dev = [(f"{p} {e}", t) for p in gym.gym_dev_prefixes for e, t in gym.base_rules]
    assert len(report.gym_proposals) == len(report.branches) == 3
    for trial, candidate in zip(report.branches, report.gym_proposals, strict=True):
        replayed = apply_edit_graph(base, compile_to_edit_graph(base, candidate.proposal)).model
        accuracy = sum(replayed.evaluate_prompt(p, target_token=t).matched_target for p, t in dev) / len(dev)
        kept = sum(replayed.evaluate_prompt(p).top1_token_id == base.evaluate_prompt(p).top1_token_id for p in gym.retention_prompts) / len(gym.retention_prompts)
        assert (accuracy, kept) == (trial.gym_dev_accuracy, trial.retention_accuracy)


# ---- Condition 3: determinism.
def test_compilation_ignores_mapping_order_and_survives_json(model) -> None:  # type: ignore[no-untyped-def]
    proposal = _low_rank(model)
    operation = proposal.operations[0]
    shuffled = replace(operation, operands=dict(reversed(list(operation.operands.items()))))
    assert gp.compile_proposal(replace(proposal, operations=(shuffled,))) == gp.compile_proposal(proposal)
    again = gp.proposal_from_document(json.loads(json.dumps(gp.proposal_to_document(proposal))))
    assert gp.compile_proposal(again) == gp.compile_proposal(proposal)
    assert gp.proposal_digest(again) == gp.proposal_digest(proposal)


def test_any_change_changes_the_compiled_digest(model) -> None:  # type: ignore[no-untyped-def]
    one = resolve_proposal(_spec(), model, [{}])
    other = resolve_proposal(_spec(parameters={"factor": 0.25}), model, [{}])
    assert gp.compile_proposal(one)["compiled_digest"] != gp.compile_proposal(other)["compiled_digest"]


def test_compiler_refuses_non_finite_values_and_bad_documents(model) -> None:  # type: ignore[no-untyped-def]
    proposal = resolve_proposal(_spec(), model, [{}])
    with pytest.raises(gp.GymProposalError):
        gp.compile_proposal(replace(proposal, operations=(replace(proposal.operations[0], arguments={"factor": float("nan")}),)))
    document = gp.proposal_to_document(proposal)
    with pytest.raises(gp.GymProposalError):
        gp.proposal_from_document({**document, "kind": "other"})
    with pytest.raises(gp.GymProposalError):
        gp.proposal_from_document({k: v for k, v in document.items() if k != "spec"})
    with pytest.raises(gp.GymProposalError):
        gp.proposal_from_document({**document, "schema_version": 2})


def test_executed_graph_matches_the_compiled_steps(model) -> None:  # type: ignore[no-untyped-def]
    proposal = _low_rank(model)
    graph = compile_to_edit_graph(model, proposal)
    assert graph == compile_to_edit_graph(pythia.build_mini_pythia_model(), proposal)
    compiled = gp.compile_proposal(proposal)
    assert [s["operands"] for s in graph["steps"]] == [s["operands"] for s in compiled["steps"]]
    assert graph["proposal"]["proposal_digest"] == compiled["proposal_digest"]
    assert graph["candidate_state_digest"] == proposal.expected_candidate_digest


def test_two_operations_chain_and_replay(model) -> None:  # type: ignore[no-untyped-def]
    site = gp.SemanticSite("mlp_down", 1)
    planned = gp.ProposedOperation(site, "scale", {"factor": 0.5})
    spec = _spec(operations=(planned, planned))
    proposal = resolve_proposal(spec, model, [{}, {}])
    graph = compile_to_edit_graph(model, proposal)
    assert graph["steps"][1]["depends_on"] == ["step_0"]
    replayed = apply_edit_graph(pythia.build_mini_pythia_model(), json.loads(json.dumps(graph)))
    name = model.mlp_down_proj_site(1)
    assert torch.allclose(replayed.edited_tensors[name], 0.25 * model.tensors[name])


def test_constraints_bind_when_resolving(model) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="changed parameters"):
        resolve_proposal(_spec(constraints={"max_changed_parameters": 1}), model, [{}])


# ---- Conditions 8 and 9: missing requirements.
def _pytest_real_run(checkpoint: Path, *, qualification: bool, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "NF_QUALIFICATION"}
    env["NF_PYTHIA_14M_DIR"] = str(checkpoint)
    if qualification:
        env["NF_QUALIFICATION"] = "1"
    target = f"{Path(__file__)}::test_qualification_run_on_real_pythia"
    return subprocess.run(
        [sys.executable, "-m", "pytest", target, "-q", "-rs", "-p", "no:cacheprovider", "--rootdir", str(tmp_path)],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=False,
    )


def test_missing_checkpoint_fails_qualification_and_skips_normal_ci(tmp_path: Path) -> None:
    absent = tmp_path / "absent"
    failed = _pytest_real_run(absent, qualification=True, tmp_path=tmp_path)
    assert failed.returncode == 1 and "1 failed" in failed.stdout.strip().splitlines()[-1] and "skipped" not in failed.stdout.strip().splitlines()[-1]
    skipped = _pytest_real_run(absent, qualification=False, tmp_path=tmp_path)
    assert skipped.returncode == 0 and "1 skipped" in skipped.stdout.strip().splitlines()[-1] and "passed" not in skipped.stdout.strip().splitlines()[-1]


def test_script_has_no_skip_path_for_a_missing_checkpoint(tmp_path: Path) -> None:
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--checkpoint", str(tmp_path / "absent"), "--out", str(tmp_path / "o"), "--repo", str(ROOT)],
        capture_output=True, text=True, check=False,
    )
    assert done.returncode != 0 and "Checkpoint is missing" in done.stderr
    assert not (tmp_path / "o").exists()


def test_script_fails_without_the_native_runtime(tmp_path: Path) -> None:
    package = tmp_path / "neural_foundry"
    shutil.copytree(Path(gp.__file__).parent, package, ignore=shutil.ignore_patterns("_native*", "__pycache__"))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), os.environ.get("PYTHONPATH", "")])}
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--checkpoint", str(CHECKPOINT), "--out", str(tmp_path / "o"), "--repo", str(ROOT)],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=False,
    )
    assert done.returncode != 0 and "_native" in done.stderr


def _load_script():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("qualify_milestone1", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_fails_without_the_backend_with_a_wrong_checkpoint_or_changed_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = _load_script()
    fake = tmp_path / "ckpt"
    fake.mkdir()
    (fake / "model.safetensors").write_bytes(b"not pythia")
    with pytest.raises(script.QualificationError, match="not the pinned"):
        script.require_runtime(fake, ROOT)
    monkeypatch.setattr(script, "PYTHIA_14M_SAFETENSORS_SHA256", rr.sha256_file(fake / "model.safetensors"))
    real_find = importlib.util.find_spec
    monkeypatch.setattr(script.importlib.util, "find_spec", lambda name, *a: None if name == "transformers" else real_find(name, *a))
    with pytest.raises(script.QualificationError, match="transformers"):
        script.require_runtime(fake, ROOT)
    monkeypatch.setattr(script.importlib.util, "find_spec", lambda name, *a: object())
    with pytest.raises(script.QualificationError, match="differs"):
        script.require_runtime(fake, tmp_path)


# ---- The recorded run. It needs no checkpoint: the evidence is committed.
def _check_evidence(evidence: Path) -> tuple[dict, dict]:  # type: ignore[type-arg]
    public = (evidence / "signing.pub").read_text(encoding="utf-8").strip()
    record = rr.verify_run_record(json.loads((evidence / "run_record.signed.json").read_text(encoding="utf-8")), public_key=public, artifact_root=evidence)
    envelope = json.loads((evidence / "edit_graph.signed.json").read_text(encoding="utf-8"))
    verify_patch_signature(envelope, public_key=public)
    graph = envelope["patch"]
    report = json.loads((evidence / "report.json").read_text(encoding="utf-8"))
    replay = json.loads((evidence / "replay.json").read_text(encoding="utf-8"))
    proposal = gp.proposal_from_document(json.loads((evidence / "proposal.json").read_text(encoding="utf-8")))
    signed = graph["candidate_state_digest"]
    assert proposal.expected_candidate_digest == signed == replay["candidate_state_digest"] == replay["export_state_digest"] == report["signed_candidate_state_digest"]
    assert record["edit_graph_digest"] == str(content_digest(canonicalize(graph)))
    assert graph["proposal"]["proposal_digest"] == gp.proposal_digest(proposal)
    assert json.loads((evidence / "compiled_proposal.json").read_text(encoding="utf-8")) == gp.compile_proposal(proposal)
    assert report["controls"]["noop"]["passed"] is True
    text = report["candidate"]["text"]
    assert {"kl_nats_per_token", "candidate_perplexity", "base_perplexity"} <= set(text)
    assert report["target"]["metrics"]["n"] == 12
    emitted = json.loads((evidence / "gym_proposals.json").read_text(encoding="utf-8"))
    assert len(emitted) == 3 and sum(c["selected"] for c in emitted) == 1
    assert next(c for c in emitted if c["selected"])["proposal"] == gp.proposal_to_document(proposal)
    assert report["retention_matrix"] and all("noop" in row and "candidate" in row for row in report["retention_matrix"].values())
    assert max(report["transformers_parity"]["max_abs_logit_difference"].values()) < 1e-4
    return record, report


def test_recorded_qualification_run_verifies() -> None:
    record, _ = _check_evidence(EVIDENCE)
    assert record["commit"]["dirty"] is False
    assert record["checkpoint"]["model_id"] == "EleutherAI/pythia-14m"


def test_recorded_evidence_refuses_tampering(tmp_path: Path) -> None:
    copy = tmp_path / "evidence"
    shutil.copytree(EVIDENCE, copy)
    report = copy / "report.json"
    report.write_text(report.read_text(encoding="utf-8").replace('"passed": true', '"passed": false', 1), encoding="utf-8")
    with pytest.raises(rr.RunRecordError, match="does not match"):
        _check_evidence(copy)


# ---- Real checkpoint: skipped in normal CI, a failure during a qualification run.
def test_qualification_run_on_real_pythia(tmp_path: Path) -> None:
    def unavailable(reason: str) -> None:
        if os.environ.get("NF_QUALIFICATION") == "1":
            pytest.fail(f"Qualification can't run: {reason}")
        pytest.skip(reason)

    if not (CHECKPOINT / "model.safetensors").is_file():
        unavailable(f"Needs Pythia-14M at {CHECKPOINT}")
    if importlib.util.find_spec("transformers") is None:
        unavailable("Needs transformers for the cross-check")
    repo = Path(os.environ.get("NF_QUALIFICATION_REPO", str(ROOT)))
    try:
        rr.capture_commit(repo)
    except rr.RunRecordError as exc:
        unavailable(f"Needs a git repository for the run record; set NF_QUALIFICATION_REPO: {exc}")
    out = tmp_path / "run"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--checkpoint", str(CHECKPOINT), "--out", str(out), "--repo", str(repo), "--allow-dirty", "--bootstrap-samples", "200", "--control-seeds", "101"],
        check=True, cwd=tmp_path,
    )
    _check_evidence(out / "evidence")
