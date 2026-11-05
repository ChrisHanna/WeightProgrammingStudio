"""Tests for T060–T063: opaque autonomy, flagship protocol, independent replay, and M3 vertical slice."""

from __future__ import annotations

import copy
import json
import struct
from pathlib import Path

import pytest
from neural_foundry import ir, surgeon
from neural_foundry.flagship import (
    FlagshipError,
    FlagshipStudySpec,
    IndependentReproductionBundle,
    OpaqueTargetPacket,
    UnseenModelFlagshipRunner,
    run_m3_vertical_slice,
    run_opaque_autonomy_protocol,
)
from neural_foundry.pretrained import (
    _build_safetensors,
    intake_local_qwen2,
    write_mini_qwen2_checkpoint,
)
from neural_foundry.records import content_digest
from test_pretrained_adapter import _qwen2_compile_inputs


def test_opaque_target_protocol_withholds_solutions_and_runs_two_non_noop_branches(
    tmp_path: Path,
) -> None:
    ckpt_dir = write_mini_qwen2_checkpoint(tmp_path / "qwen2-opaque")
    bundle = intake_local_qwen2(ckpt_dir)

    clean_packet = OpaqueTargetPacket(
        packet_id="pkt-clean-01",
        model_id=bundle.manifest_digest,
        objective_description="Improve reasoning accuracy without regression",
        agent_prompts=("Diagnose layer 0 MLP activations and propose two distinct edits.",),
        retrieval_corpus=({"id": "exp-1", "note": "Subspace projection on MLP"},),
        package_metadata={"gym": "enterprise-reasoning"},
        tool_outputs=({"site": "layers.0.mlp.down_proj"},),
        proposer_mode="autonomous_agent",
    )
    region = {
        "tensor": "model.layers.0.mlp.down_proj.weight",
        "start": [0, 0],
        "shape": [2, 4],
    }
    prog1, space1, policy1, _ = _qwen2_compile_inputs(
        bundle,
        operation={"type": "scale", "target": region, "factor": 0.75},
        read_regions=[region],
        write_regions=[region],
        operands_meta={},
        changed_params_bound=16,
    )
    prog2 = copy.deepcopy(prog1)
    prog2["operations"][0]["operation"]["factor"] = 0.50

    report = run_opaque_autonomy_protocol(
        packet=clean_packet,
        qwen_bundle=bundle,
        branch_programs=[
            ("branch-1", prog1, space1, policy1, {}),
            ("branch-2", prog2, space1, policy1, {}),
        ],
    )
    assert report["unseen_autonomous_valid"] is True
    assert report["distinct_non_noop_branches"] == 2
    assert report["qualification_stratum"] == "unseen_autonomous_candidate"

    # Known recipe/coordinate exposure in prompts or retrieval invalidates unseen_autonomous!
    contaminated_packet = OpaqueTargetPacket(
        packet_id="pkt-leaked-01",
        model_id=bundle.manifest_digest,
        objective_description="Repair model",
        agent_prompts=("Apply repair_coordinates=[2,2] with repair_values=1.0",),
        retrieval_corpus=({"id": "leak", "target_recipe": "scale factor 2.0"},),
        package_metadata={},
        tool_outputs=(),
        proposer_mode="autonomous_agent",
    )
    contam_report = run_opaque_autonomy_protocol(
        packet=contaminated_packet,
        qwen_bundle=bundle,
        branch_programs=[
            ("branch-1", prog1, space1, policy1, {}),
            ("branch-2", prog2, space1, policy1, {}),
        ],
    )
    assert contam_report["unseen_autonomous_valid"] is False
    assert contam_report["qualification_stratum"] == "contaminated_solution_exposed"
    assert len(contam_report["packet_audit"]["exposed_routes"]) == 2

    # Fake/scripted protocol mode is separated from genuine agent qualification
    fake_packet = OpaqueTargetPacket(
        **{**clean_packet.__dict__, "proposer_mode": "fake_protocol"}
    )
    fake_report = run_opaque_autonomy_protocol(
        packet=fake_packet,
        qwen_bundle=bundle,
        branch_programs=[
            ("branch-1", prog1, space1, policy1, {}),
            ("branch-2", prog2, space1, policy1, {}),
        ],
    )
    assert fake_report["unseen_autonomous_valid"] is False
    assert (
        fake_report["qualification_stratum"]
        == "software_protocol_test_not_autonomous_flagship"
    )


def test_flagship_protocol_rejects_published_solutions_human_repairs_and_altered_thresholds() -> None:
    runner = UnseenModelFlagshipRunner()
    valid_spec = FlagshipStudySpec(
        study_id="flagship-qwen2-01",
        custodian_id="custodian.independent.lab",
        target_model_id="hf.qwen2.dense:sha256:" + "b" * 64,
        target_ancestry_absent_from_tuning=True,
        published_solution_used=False,
        human_supplied_repair=False,
        thresholds_altered_post_selection=False,
        baseline_competence_verified=True,
        required_gyms=("gym.target", "gym.ret1", "gym.ret2", "gym.safety", "gym.efficiency"),
        has_noop_control=True,
        independent_holdout_locked=True,
        search_space_frozen=True,
        distinct_non_noop_branches=2,
        approved_budget={
            "wall_seconds": 3600,
            "cpu_seconds": 3600,
            "memory_bytes": 2_147_483_648,
            "max_evaluations": 100,
        },
    )
    validated = runner.validate_protocol(valid_spec)
    assert validated["protocol_valid"] is True
    assert validated["executed_by_default"] is False
    assert validated["scientific_flagship_status"] == "not_established"

    for field_name, bad_val, expected_code in (
        ("published_solution_used", True, "PUBLISHED_SOLUTION_FORBIDDEN"),
        ("human_supplied_repair", True, "HUMAN_REPAIR_FORBIDDEN"),
        ("thresholds_altered_post_selection", True, "THRESHOLD_TAMPERING"),
        ("target_ancestry_absent_from_tuning", False, "CONTAMINATED_ANCESTRY"),
        ("distinct_non_noop_branches", 1, "INSUFFICIENT_BRANCHES"),
    ):
        bad_spec = FlagshipStudySpec(**{**valid_spec.__dict__, field_name: bad_val})
        with pytest.raises(FlagshipError) as err:
            runner.validate_protocol(bad_spec)
        assert err.value.code == expected_code


def test_independent_replay_separates_parameter_inference_and_scientific_verdicts(
    tmp_path: Path,
) -> None:
    ckpt_dir = write_mini_qwen2_checkpoint(tmp_path / "qwen2-replay")
    qwen_bundle = intake_local_qwen2(ckpt_dir)
    source = qwen_bundle.load_native_source()
    smoke = qwen_bundle.smoke_suite()
    region = {
        "tensor": "model.layers.0.mlp.down_proj.weight",
        "start": [0, 0],
        "shape": [4, 8],
    }
    basis_bytes = _build_safetensors(
        {"basis": ("f64", (4, 1), struct.pack("<4d", 1.0, 0.0, 0.0, 0.0))}
    )
    proj_op = {
        "type": "subspace_projection",
        "target": region,
        "basis": "u_basis",
        "side": "left",
        "strength": 0.5,
        "orthonormal_tolerance": 1e-6,
    }
    operands_meta = {
        "u_basis": {
            "digest": content_digest(basis_bytes),
            "tensor": "basis",
            "dtype": "f64",
            "shape": [4, 1],
            "data_bytes": 32,
            "artifact_bytes": len(basis_bytes),
            "access": "development",
            "provenance": {
                "schema_version": 1,
                "origin": "supplied_fact",
                "support": "untested",
                "provenance": "replay_basis",
                "evidence": [],
                "access": "development",
            },
        }
    }
    prog, space, policy, _ = _qwen2_compile_inputs(
        qwen_bundle,
        operation=proj_op,
        read_regions=[region],
        write_regions=[region],
        operands_meta=operands_meta,
        changed_params_bound=32,
    )
    plan = ir.compile(source, prog, space, policy, {"u_basis": basis_bytes})
    cand = surgeon.apply(plan, smoke)
    receipt = json.loads(cand.receipt_json())

    env = {"os": "linux", "arch": "x86_64", "precision": "f64", "backend": "nf.native_cpu.v1"}
    repro_bundle = IndependentReproductionBundle(
        bundle_id="repro-qwen2-01",
        manifest=qwen_bundle.manifest,
        checkpoint_bytes=qwen_bundle.checkpoint_bytes,
        assets=qwen_bundle.assets,
        program=prog,
        search_space=space,
        policy=policy,
        operands={"u_basis": basis_bytes},
        smoke_suite=smoke,
        expected_candidate_digest=cand.digest,
        expected_parameter_digest=cand.parameter_digest,
        expected_smoke_outputs_digest=receipt["smoke_outputs"],
        runtime_environment=env,
        scientific_replication_evidence=None,
    )
    exported = repro_bundle.export_manifest()
    assert exported["bundle_digest"].startswith("sha256:")

    # 1. Exact clean-environment replay passes parameter_replay and inference_repeatability,
    #    while keeping scientific_replication="not_established"
    ok_replay = repro_bundle.replay_in_clean_environment(observed_environment=env)
    assert ok_replay["parameter_replay"] == "passed"
    assert ok_replay["inference_repeatability"] == "passed"
    assert ok_replay["scientific_replication"] == "not_established"
    assert ok_replay["reproduction_claimed"] is True

    # 2. Missing operand artifacts fail parameter_replay and cannot claim reproduction
    missing_replay = repro_bundle.replay_in_clean_environment(
        observed_environment=env, supplied_operands={}
    )
    assert missing_replay["parameter_replay"] == "failed_missing_artifacts"
    assert missing_replay["reproduction_claimed"] is False

    # 3. Mismatched runtime conditions fail inference_repeatability and cannot claim reproduction
    mismatch_replay = repro_bundle.replay_in_clean_environment(
        observed_environment={**env, "precision": "bf16"}
    )
    assert mismatch_replay["parameter_replay"] == "passed"
    assert mismatch_replay["inference_repeatability"] == "failed_environment_mismatch"
    assert mismatch_replay["reproduction_claimed"] is False


def test_m3_vertical_slice_connects_all_components_and_separates_readiness_states(
    tmp_path: Path,
) -> None:
    report = run_m3_vertical_slice(tmp_path, provider_qualified=False)
    assert report["milestone"] == "M3"
    assert report["states"] == {
        "software_ready": True,
        "provider_qualified": False,
        "flagship_success": "not_established",
    }
    assert report["solution_exposure_detected"] is False
    assert len(report["branches_evaluated"]) == 2
    assert report["gym_bound"] is True
    assert report["independent_verification"]["status"] == "passed"
    assert report["independent_verification"]["promote_eligible"] is True
