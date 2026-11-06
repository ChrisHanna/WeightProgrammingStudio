"""Edit plans executed through the real Rust compiler and surgeon."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from neural_foundry import patch_ux, prd_adapters, pythia, studio
from neural_foundry.gym_proposal import (
    GymProposalError,
    GymWeightProposalSpec,
    ProposedOperation,
    ResolvedGymWeightProposal,
    ResolvedOperation,
    SemanticSite,
)
from neural_foundry.edit_plan import (
    EditConstraints,
    EditConstraintViolation,
    EditGraphError,
    EditPlan,
    Region,
    apply_edit_graph,
    apply_signed_edit_graph,
    search_edits,
    sign_edit_graph,
    verify_exported_candidate,
)

GYM = pythia.ALGORITHMIC_THREE_SKILL_GYM
REAL_CHECKPOINT = Path(os.environ.get("NF_PYTHIA_14M_DIR", "/tmp/pythia-14m")) / "model.safetensors"


@pytest.fixture
def model():  # type: ignore[no-untyped-def]
    return pythia.build_mini_pythia_model()


def _site(model) -> str:  # type: ignore[no-untyped-def]
    return model.mlp_down_proj_site(1)


def _w(model) -> torch.Tensor:  # type: ignore[no-untyped-def]
    return model.tensors[_site(model)].to(torch.float64)


def _edited(plan: EditPlan, site: str) -> torch.Tensor:
    return plan.preview().edited_tensors[site]


def test_region_scale_changes_only_that_region(model) -> None:  # type: ignore[no-untyped-def]
    site, w = _site(model), _w(model)
    out = _edited(EditPlan(model).scale(site, 2.0, region=Region(rows=(0, 2), cols=(1, 3))), site)
    expected = w.clone()
    expected[0:2, 1:3] *= 2.0
    assert torch.equal(out, expected)


def test_replace_slice_mask_permute_and_linear_map(model) -> None:  # type: ignore[no-untyped-def]
    site, w = _site(model), _w(model)
    value = torch.full((1, 2), 0.5, dtype=torch.float64)
    out = _edited(EditPlan(model).replace_slice(site, value, region=Region(rows=(0, 1), cols=(0, 2))), site)
    assert torch.equal(out[0, :2], value[0]) and torch.equal(out[1:], w[1:])

    mask = torch.ones_like(w)
    mask[0, :] = 0.0
    out = _edited(EditPlan(model).mask(site, mask), site)
    assert torch.equal(out, w * mask)

    order = list(reversed(range(w.shape[0])))
    assert torch.allclose(_edited(EditPlan(model).permute(site, 0, order), site), w[order, :])

    rows = w.shape[0]
    rotate = torch.eye(rows, dtype=torch.float64) * 2.0
    assert torch.allclose(_edited(EditPlan(model).linear_map(site, rotate, side="left"), site), 2.0 * w)


def test_derived_and_spectral_edits_match_their_definitions(model) -> None:  # type: ignore[no-untyped-def]
    site, w = _site(model), _w(model)
    assert float(_edited(EditPlan(model).clamp(site, 0.01), site).abs().max()) <= 0.01 + 1e-12
    t = 0.01
    assert torch.allclose(_edited(EditPlan(model).soft_threshold(site, t), site), torch.sign(w) * torch.clamp(w.abs() - t, min=0.0))
    limit = float(torch.linalg.svdvals(w)[0]) / 2
    result = EditPlan(model).spectral_clip(site, limit).preview()
    assert float(torch.linalg.svdvals(result.edited_tensors[site])[0]) <= limit + 1e-9
    assert result.steps[0].spectrum_before[0] > result.steps[0].spectrum_after[0]
    assert torch.linalg.matrix_rank(_edited(EditPlan(model).rank_truncate(site, 2), site)) == 2


def test_multi_tensor_plan_is_all_or_nothing(model) -> None:  # type: ignore[no-untyped-def]
    a, b = model.mlp_down_proj_site(0), model.mlp_down_proj_site(1)
    before = _w(model).clone()
    done = EditPlan(model).scale(a, 0.5).scale(b, 0.5).preview()
    assert len(done.steps) == 2 and set(done.edited_tensors) == {a, b}
    bad = EditPlan(model).scale(a, 0.5).add(b, torch.zeros(1, 1, dtype=torch.float64))
    with pytest.raises(Exception):
        bad.preview()
    assert torch.equal(_w(model), before)


def test_constraints_reject_before_applying(model) -> None:  # type: ignore[no-untyped-def]
    site = _site(model)
    with pytest.raises(EditConstraintViolation, match="not allowed"):
        EditPlan(model, EditConstraints(forbidden_tensors=(site,))).scale(site, 2.0)
    with pytest.raises(EditConstraintViolation, match="changed parameters"):
        EditPlan(model, EditConstraints(max_changed_parameters=1)).scale(site, 2.0).preview()
    with pytest.raises(EditConstraintViolation, match="Frobenius"):
        EditPlan(model, EditConstraints(max_frobenius_delta=1e-9)).scale(site, 2.0).preview()
    with pytest.raises(EditConstraintViolation, match="retention"):
        EditPlan(model, EditConstraints(preserve_prompts=("Hello world",), min_retention=1.5)).scale(site, 1.5).preview()


def test_region_outside_tensor_is_refused(model) -> None:  # type: ignore[no-untyped-def]
    site = _site(model)
    with pytest.raises(ValueError, match="outside"):
        EditPlan(model).scale(site, 2.0, region=Region(rows=(0, 10_000))).preview()


def test_plan_lifecycle_leaves_base_untouched(model, tmp_path) -> None:  # type: ignore[no-untyped-def]
    site, before = _site(model), _w(model).clone()
    plan = studio.WeightProgrammingStudio().edits(model).scale(site, 0.5)
    candidate = plan.execute()
    assert candidate.plan_digest.startswith("sha256:") and candidate.changed_parameters > 0
    assert torch.equal(_w(model), before) and not torch.equal(candidate.model.tensors[site].to(torch.float64), before)
    assert plan.export(tmp_path).is_file()
    assert plan.rollback() is model
    with pytest.raises(ValueError, match="no steps"):
        plan.preview()


def test_activation_edit_runs_through_rust(model) -> None:  # type: ignore[no-untyped-def]
    prompts = [f"Task: {e}" for e, _ in GYM.base_rules]
    result = EditPlan(model).suppress_activation_direction(1, prompts[:3], ["Hello world"]).preview()
    assert result.steps[0].plan_digest.startswith("sha256:") and result.changed_parameters > 0


def test_search_keeps_controls_rejections_and_a_frontier(model) -> None:  # type: ignore[no-untyped-def]
    site = _site(model)
    results = search_edits(
        model,
        development=[(f"Task: {e}", t) for e, t in GYM.base_rules],
        retention_prompts=GYM.retention_prompts,
        proposals={
            "mild": lambda p: p.scale(site, 0.99),
            "clip": lambda p: p.spectral_clip(site, 0.5),
            "too_big": lambda p: p.add_constraint(max_frobenius_delta=1e-9).scale(site, 3.0),
        },
    )
    by_label = {r.label: r for r in results}
    assert {"control_noop", "control_random"} <= set(by_label)
    assert by_label["too_big"].rejection_reason and not by_label["too_big"].on_pareto_frontier
    assert not by_label["control_random"].on_pareto_frontier
    assert any(r.on_pareto_frontier for r in results)


def _plan(model):  # type: ignore[no-untyped-def]
    a, b = model.mlp_down_proj_site(0), model.mlp_down_proj_site(1)
    value = torch.full((1, 2), 0.5, dtype=torch.float64)
    mask = torch.ones_like(_w(model))
    mask[0, :] = 0.0
    limit = float(torch.linalg.svdvals(_w(model))[0]) / 2
    return (
        EditPlan(model)
        .scale(a, 0.5, region=Region(rows=(0, 2)))
        .replace_slice(b, value, region=Region(rows=(0, 1), cols=(0, 2)))
        .mask(b, mask)
        .spectral_clip(b, limit)
        .permute(b, 0, list(reversed(range(_w(model).shape[0]))))
    )


def test_third_party_replay_reproduces_exact_weights(model) -> None:  # type: ignore[no-untyped-def]
    plan = _plan(model)
    original = plan.execute()
    keys = patch_ux.generate_ed25519_keypair()
    envelope = sign_edit_graph(plan.to_graph(), signing_key=keys["private_key"])
    replayed = apply_signed_edit_graph(pythia.build_mini_pythia_model(), envelope, public_key=keys["public_key"])
    assert replayed.plan_digest == original.plan_digest
    for name, weight in original.edited_tensors.items():
        assert torch.equal(replayed.edited_tensors[name], weight)


def test_replay_rejects_wrong_base_tampering_and_untrusted_keys(model) -> None:  # type: ignore[no-untyped-def]
    import copy

    plan = _plan(model)
    graph = plan.to_graph()
    other = pythia.build_mini_pythia_model()
    other.tensors = {k: v * 1.01 for k, v in other.tensors.items()}
    with pytest.raises(EditGraphError, match="base model"):
        apply_edit_graph(other, graph)

    changed = copy.deepcopy(graph)
    changed["steps"][0]["arguments"]["factor"] = 0.75
    with pytest.raises(EditGraphError, match="different weights"):
        apply_edit_graph(model, changed)

    keys = patch_ux.generate_ed25519_keypair()
    envelope = sign_edit_graph(graph, signing_key=keys["private_key"])
    with pytest.raises(ValueError):
        apply_signed_edit_graph(model, {**envelope, "patch": changed}, public_key=keys["public_key"])
    with pytest.raises(ValueError):
        apply_signed_edit_graph(model, envelope, public_key=patch_ux.generate_ed25519_keypair()["public_key"])

    planted = copy.deepcopy(graph)
    planted["steps"][0]["arguments"]["engine_name"] = "x"
    with pytest.raises(EditGraphError, match="unknown"):
        apply_edit_graph(model, planted)


def test_spectral_edit_that_moves_nothing_changes_nothing(model) -> None:  # type: ignore[no-untyped-def]
    site, w = _site(model), _w(model)
    result = EditPlan(model).spectral_clip(site, 2 * float(torch.linalg.svdvals(w)[0])).preview()
    assert result.changed_parameters == 0
    assert torch.equal(result.edited_tensors[site], w)
    with pytest.raises(ValueError, match="positive"):
        EditPlan(model).spectral_clip(site, -1.0)


def test_adding_a_constraint_rechecks_a_previewed_plan(model) -> None:  # type: ignore[no-untyped-def]
    plan = EditPlan(model).scale(_site(model), 2.0)
    plan.preview()
    plan.add_constraint(max_frobenius_delta=1e-9)
    with pytest.raises(EditConstraintViolation, match="Frobenius"):
        plan.execute()


def test_replay_rejects_a_base_differing_only_in_an_unedited_tensor(model) -> None:  # type: ignore[no-untyped-def]
    graph = EditPlan(model).scale(_site(model), 0.5).to_graph()
    other = pythia.build_mini_pythia_model()
    untouched = other.mlp_down_proj_site(0)
    other.tensors = {**other.tensors, untouched: other.tensors[untouched] * 1.01}
    with pytest.raises(EditGraphError, match="base model"):
        apply_edit_graph(other, graph)


def test_search_records_a_rust_refusal_as_a_rejection(model) -> None:  # type: ignore[no-untyped-def]
    site = _site(model)
    results = search_edits(model, development=[], retention_prompts=("Hello world",), proposals={"over_cap": lambda p: p.scale(site, 1000.0)})
    assert {r.label: r for r in results}["over_cap"].rejection_reason


def test_narrow_or_altered_export_fails_verification(model, tmp_path) -> None:  # type: ignore[no-untyped-def]
    plan = EditPlan(model).scale(_site(model), 0.5)
    digest = plan.execute().state_digest
    plan.export(tmp_path / "f64")
    assert verify_exported_candidate(tmp_path / "f64", digest) == digest
    with pytest.raises(EditGraphError, match="don't match"):
        verify_exported_candidate(tmp_path / "f64", "sha256:" + "0" * 64)
    plan.export(tmp_path / "f32", dtype="f32")
    with pytest.raises(EditGraphError, match="F32"):
        verify_exported_candidate(tmp_path / "f32", digest)


def _unavailable(reason: str) -> None:
    """Skipped isn't passed: qualification runs must fail when a requirement is missing."""
    if os.environ.get("NF_QUALIFICATION") == "1":
        pytest.fail(f"Qualification can't run: {reason}")
    pytest.skip(reason)


def _require_checkpoint() -> None:
    if not REAL_CHECKPOINT.is_file():
        _unavailable(f"Needs Pythia-14M at {REAL_CHECKPOINT}, which is absent")


def _resolved(base, source: EditPlan) -> ResolvedGymWeightProposal:  # type: ignore[no-untyped-def]
    """A frozen proposal carrying the operands of `source`, whose steps must all be on layer sites."""
    graph = source.to_graph()
    planned, resolved = [], []
    for step in graph["steps"]:
        layer = int(step["tensor"].split(".layers.")[1].split(".")[0])
        arguments = dict(step["arguments"])
        operation = arguments.pop("operation_type")
        planned.append(ProposedOperation(SemanticSite("mlp_down", layer), operation))
        resolved.append(ResolvedOperation(step["tensor"], tuple(base.tensors[step["tensor"]].shape), operation, arguments, step["operands"]))
    spec = GymWeightProposalSpec("proposal.test", tuple(planned), {}, "scripted", "sha256:" + "a" * 64)
    return ResolvedGymWeightProposal(
        spec, str(base.config["model_type"]), graph["base_state_digest"], tuple(resolved), graph["candidate_state_digest"]
    )


def test_resolved_proposal_compiles_to_the_same_replayable_graph(model) -> None:  # type: ignore[no-untyped-def]
    site = _site(model)
    proposal = _resolved(model, EditPlan(model).scale(site, 0.5, region=Region(rows=(0, 2))))
    graph = EditPlan.from_resolved_proposal(model, proposal).to_graph()
    assert graph == EditPlan.from_resolved_proposal(pythia.build_mini_pythia_model(), proposal).to_graph()
    assert graph["proposal"]["proposal_id"] == "proposal.test"
    keys = patch_ux.generate_ed25519_keypair()
    replayed = apply_signed_edit_graph(pythia.build_mini_pythia_model(), sign_edit_graph(graph, signing_key=keys["private_key"]), public_key=keys["public_key"])
    assert replayed.state_digest == proposal.expected_candidate_digest


def test_resolved_proposal_refuses_anything_that_does_not_match(model) -> None:  # type: ignore[no-untyped-def]
    site = _site(model)
    good = _resolved(model, EditPlan(model).scale(site, 0.5))
    operation = good.operations[0]

    def refused(match: str, **changes) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(GymProposalError, match=match):
            EditPlan.from_resolved_proposal(model, replace(good, **changes))

    refused("is for", architecture="qwen2")
    refused("isn't the base", base_checkpoint_digest="sha256:" + "0" * 64)
    refused("not the expected", expected_candidate_digest="sha256:" + "0" * 64)
    refused("isn't the tensor", operations=(replace(operation, tensor=model.mlp_down_proj_site(0)),))
    refused("has shape", operations=(replace(operation, shape=(1, 1)),))
    refused("unknown arguments", operations=(replace(operation, arguments={**operation.arguments, "engine_name": "x"}),))
    refused("unknown arguments", operations=(replace(operation, arguments={**operation.arguments, "operation_type": "mask"}),))
    refused("Unknown proposal constraint", spec=replace(good.spec, constraints={"nope": 1}))
    tight = replace(good.spec, constraints={"max_frobenius_delta": 1e-9})
    with pytest.raises(EditConstraintViolation):
        EditPlan.from_resolved_proposal(model, replace(good, spec=tight))


def test_steps_added_after_compiling_drop_the_proposal_record(model) -> None:  # type: ignore[no-untyped-def]
    site = _site(model)
    plan = EditPlan.from_resolved_proposal(model, _resolved(model, EditPlan(model).scale(site, 0.5)))
    assert "proposal" not in plan.scale(site, 0.5).to_graph()


def test_resolved_proposal_replays_in_a_fresh_process_on_real_pythia(tmp_path) -> None:
    _require_checkpoint()
    base = pythia.PythiaModel.from_pretrained_dir(REAL_CHECKPOINT.parent)
    site = base.mlp_down_proj_site(5)
    limit = 0.9 * float(torch.linalg.svdvals(base.tensors[site])[0])
    proposal = _resolved(base, EditPlan(base).spectral_clip(site, limit))
    graph = EditPlan.from_resolved_proposal(base, proposal).to_graph()
    keys = patch_ux.generate_ed25519_keypair()
    envelope, key = tmp_path / "envelope.json", tmp_path / "public.txt"
    envelope.write_text(json.dumps(sign_edit_graph(graph, signing_key=keys["private_key"])), encoding="utf-8")
    key.write_text(keys["public_key"], encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "neural_foundry.edit_plan", "--checkpoint", str(REAL_CHECKPOINT.parent), "--envelope", str(envelope), "--public-key-file", str(key)],
        capture_output=True, text=True, cwd=tmp_path, check=True,
    )
    assert json.loads(result.stdout.strip().splitlines()[-1])["candidate_state_digest"] == proposal.expected_candidate_digest


def test_spectral_clip_vertical_on_real_pythia(tmp_path) -> None:
    _require_checkpoint()
    if importlib.util.find_spec("transformers") is None:
        _unavailable("Independent verification needs transformers, which isn't installed")
    spec =importlib.util.spec_from_file_location("edit_vertical", Path(__file__).parents[1] / "scripts" / "edit_vertical.py")
    assert spec and spec.loader
    vertical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vertical)
    evidence = vertical.run_vertical(REAL_CHECKPOINT.parent, tmp_path, "spectral_clip", 5)
    signed = evidence["preview"]["state_digest"]
    assert evidence["preview"]["base_untouched"] and evidence["execute"]["same_as_preview"]
    assert evidence["replay"]["matches_candidate"] and evidence["replay"]["plan_digest_matches"]
    assert evidence["export"]["reloaded_state_digest"] == signed == evidence["export"]["replayed_export_digest"]
    assert evidence["reload"]["matches_candidate"] and evidence["independent"]["matches_candidate"]
    assert evidence["independent"]["top1_agreement_edited"] == 1.0
    assert evidence["independent"]["max_abs_logit_diff_edited"] < 1e-4
    assert evidence["independent"]["edit_visible_in_hf_max_logit_change"] > 0.1


def test_existing_methods_and_branches_share_the_graph(model) -> None:  # type: ignore[no-untyped-def]
    expr, target = GYM.base_rules[0]
    edit = prd_adapters.run_easyedit_adapter(model, algorithm="ROME", associations=[(f"Task: {expr}", target)], target_strength=80.0)
    site = _site(model)
    plan = EditPlan(model).adopt_weight(site, edit.edited_weight, method="ROME")
    assert torch.allclose(plan.preview().edited_tensors[site], edit.edited_weight.to(torch.float64))
    assert torch.equal(apply_edit_graph(model, plan.to_graph()).edited_tensors[site], plan.execute().edited_tensors[site])

    other = plan.branch().scale(site, 0.5)
    assert len(other.preview().steps) == 2 and len(plan.preview().steps) == 1
