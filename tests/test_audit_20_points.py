"""Comprehensive verification suite for all 20 AI Audit points (positive + fail-closed negative tests)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import torch
from neural_foundry import (
    _native,
    evals,
    export_formats,
    flagship,
    gpu,
    ir,
    patch_ux,
    prd_adapters,
    pythia,
    studio,
)
from neural_foundry.records import canonicalize

CLI = Path(os.environ.get("NF_CLI", "/tmp/nf-t043-joined/debug/nf"))
ROOT = Path(__file__).resolve().parents[1]


def _minimal_suite() -> evals.EvalSuite:
    return evals.import_suite(
        {
            "suite_id": "audit-suite",
            "version": "1.0.0",
            "evaluator_kind": "deterministic_scorer",
            "scorer_id": "audit.scorer",
            "scorer_version": "1.0.0",
            "eval_types": [
                "development",
                "generalization",
                "retention",
                "safety",
                "robustness",
                "efficiency",
            ],
            "tasks": [
                {"task_id": "t_dev_1", "group_id": "g1", "eval_type": "development", "capability": "TargetMath", "input": "2+2", "expected": "4", "sealed": False},
                {"task_id": "t_dev_2", "group_id": "g2", "eval_type": "development", "capability": "TargetMath", "input": "3+3", "expected": "6", "sealed": False},
                {"task_id": "t_gen", "group_id": "g3", "eval_type": "generalization", "capability": "UnseenMath", "input": "5+5", "expected": "10", "sealed": True},
                {"task_id": "t_ret", "group_id": "g4", "eval_type": "retention", "capability": "GeneralLanguage", "input": "Hi", "expected": "Hello", "sealed": True},
                {"task_id": "t_saf", "group_id": "g5", "eval_type": "safety", "capability": "SafetyRefusal", "input": "Bad", "expected": "Refuse", "sealed": True},
                {"task_id": "t_rob", "group_id": "g6", "eval_type": "robustness", "capability": "Perturbation", "input": "2 + 2", "expected": "4", "sealed": True},
                {"task_id": "t_eff", "group_id": "g7", "eval_type": "efficiency", "capability": "Efficiency", "input": "eff", "expected": "ok", "sealed": True},
            ],
        }
    )


def test_points_1_2_4_5_9_10_evals_fail_closed_provenance_sampler_cli_and_side_effects(
    tmp_path: Path,
) -> None:
    """Points 1, 2, 4, 5, 9, 10: Fail-closed evals, supplied_fact vs measured_result provenance, sampler/runtime hash, CLI verify parity, pre-registration, and side-effect suite."""
    suite = _minimal_suite()

    # Point 1: Unknown model string or missing score MUST fail closed (never pass or default to 0.75)
    with pytest.raises(evals.EvaluationError) as err_unknown:
        evals.run_eval(model="nonexistent", suite=suite)
    assert err_unknown.value.code == "MISSING_MODEL"

    with pytest.raises(evals.EvaluationError) as err_missing_score:
        evals.run_eval(
            model={"model_id": "incomplete-model", "task_scores": {"t_dev_1": 1.0}},
            suite=suite,
        )
    assert err_missing_score.value.code == "MISSING_SCORE"

    # Point 2: Static precomputed scores are labeled `supplied_fact`, whereas live callable execution is `measured_result`
    supplied_run = evals.run_eval(
        model={
            "model_id": "static-dict-model",
            "eval_type_scores": {
                "development": 0.5,
                "generalization": 0.5,
                "retention": 1.0,
                "safety": 1.0,
                "robustness": 0.8,
                "efficiency": 0.8,
            },
        },
        suite=suite,
    )
    assert supplied_run["evidence_label"]["origin"] == "supplied_fact"
    assert supplied_run["score_provenance"]["measurement_mode"] == "supplied_precomputed"

    live_answers = {
        "2+2": "4",
        "3+3": "6",
        "5+5": "10",
        "Hi": "Hello",
        "Bad": "Refuse",
        "2 + 2": "4",
        "eff": "ok",
    }
    measured_run = evals.run_eval(
        model={
            "model_id": "live-callable-model",
            "predict_fn": lambda prompt: live_answers[prompt],
            "edit_lineage": {"changed_parameters": 64, "total_parameters": 10_000},
        },
        suite=suite,
    )
    assert measured_run["evidence_label"]["origin"] == "measured_result"
    assert measured_run["score_provenance"]["measurement_mode"] == "measured_live_execution"

    # Point 5: Sampler (`presence_penalty`, `frequency_penalty`, `temperature`, `top_p`, `seed`) & `runtime` in experiment hash
    run_vllm = evals.run_eval(
        model={
            "model_id": "m1",
            "runtime": "vllm",
            "sampler": {"temperature": 0.0, "top_p": 1.0, "presence_penalty": 0.5, "frequency_penalty": 0.0, "max_tokens": 64, "seed": 17},
            "eval_type_scores": supplied_run["scores_by_type"],
        },
        suite=suite,
    )
    assert run_vllm["experiment_identity_digest"] != supplied_run["experiment_identity_digest"]
    with pytest.raises(evals.EvaluationError) as err_sampler:
        evals.compare_runs(supplied_run, run_vllm)
    assert err_sampler.value.code == "SAMPLER_OR_RUNTIME_MISMATCH"

    # Point 9: Pre-registered hypothesis, grouped bootstrap CI, and Holm-Bonferroni multiplicity correction
    pre_reg = evals.pre_register_study_plan(
        hypothesis_id="H-audit-9",
        primary_metric_family="exact_match",
        alpha=0.05,
        num_hypotheses=4,
        multiplicity_method="holm_bonferroni",
        bootstrap_resamples=300,
        seed=17,
    )
    comp = evals.compare_runs(
        supplied_run,
        measured_run,
        suite=suite,
        pre_registration=pre_reg,
        additional_branch_gains=(0.05, 0.10, 0.0),
    )
    assert comp["grouped_bootstrap_ci"]["pre_registration_digest"] == pre_reg.pre_registration_digest
    assert comp["grouped_bootstrap_ci"]["num_hypotheses_tested"] == 4
    assert comp["grouped_bootstrap_ci"]["statistically_significant_after_multiplicity"] is True

    # Point 10: Out-of-target regression blocks promotion even if target development score improves
    regressed_run = evals.run_eval(
        model={
            "model_id": "regressed-candidate",
            "eval_type_scores": {
                "development": 0.95,
                "generalization": 0.90,
                "retention": 0.80,  # Severe out-of-target retention drop!
                "safety": 0.99,
                "robustness": 0.85,
                "efficiency": 0.85,
            },
            "edit_lineage": {"changed_parameters": 64, "total_parameters": 10_000},
        },
        suite=suite,
    )
    ver_regressed = evals.verify_promotion(candidate_run=regressed_run, baseline_run=supplied_run, policy="production")
    assert ver_regressed["promote_eligible"] is False
    side_effect_check = next(c for c in ver_regressed["checks"] if c["check"] == "out_of_target_side_effect_non_regression")
    assert side_effect_check["passed"] is False

    # Point 4: CLI verify enforces required_eval_types and rejects `--model nonexistent`
    suite_file = tmp_path / "suite.json"
    suite_file.write_bytes(canonicalize(suite.manifest))
    bad_cli = subprocess.run(
        [str(CLI), "eval", "verify", "--model", "nonexistent", "--policy", "production"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad_cli.returncode != 0

    # CLI candidate missing `robustness` and `efficiency` must be rejected under `production` policy
    base_path = tmp_path / "base.json"
    incomplete_cand_path = tmp_path / "cand_incomplete.json"
    base_path.write_bytes(
        canonicalize(
            {
                "model_id": "base",
                "checkpoint_digest": "sha256:" + "1" * 64,
                "scores_by_type": {"development": 0.70, "retention": 0.99},
                "edit_lineage": {"changed_parameters": 0, "total_parameters": 1000},
            }
        )
    )
    incomplete_cand_path.write_bytes(
        canonicalize(
            {
                "model_id": "cand",
                "checkpoint_digest": "sha256:" + "2" * 64,
                "scores_by_type": {"development": 0.95, "retention": 0.99},
                "edit_lineage": {"changed_parameters": 10, "total_parameters": 1000},
            }
        )
    )
    cli_incomplete = subprocess.run(
        [str(CLI), "eval", "verify", "--model", str(incomplete_cand_path), "--baseline", str(base_path), "--policy", "production"],
        capture_output=True,
        text=True,
        check=False,
    )
    # Fails closed because required production eval_types are missing!
    assert cli_incomplete.returncode != 0


def test_point_3_single_readiness_source_of_truth() -> None:
    """Point 3: Python conformance report derives from Rust `_native.capabilities_json()` and `_native.m2_capabilities_json()`."""
    report = studio.generate_release_conformance_report()
    raw_m1 = json.loads(bytes(_native.capabilities_json()))
    raw_m2 = json.loads(bytes(_native.m2_capabilities_json()))
    assert report.rust_m1_capabilities == raw_m1
    assert report.rust_m2_capabilities == raw_m2
    assert report.capability_gates["CG-01_transactional_immutability"] == (
        bool(raw_m1["isolated_transactions"]) and bool(raw_m1["immutable_core_restore"])
    )


def test_points_6_7_8_controls_external_baselines_and_disjoint_holdout_exposure_ledger() -> None:
    """Points 6, 7, 8: Mandatory no-op & random controls, external baseline check, and operand-disjoint holdout + exposure ledger."""
    # Point 7: External baseline package probing & fail-closed `require_external_package=True`
    pkgs = prd_adapters.probe_external_baseline_packages()
    assert set(pkgs.keys()) == {"easyeditor", "mergekit", "peft"}
    model = pythia.build_mini_pythia_model()
    rome_rec = prd_adapters.run_easyedit_adapter(
        model,
        algorithm="ROME",
        associations=[("The capital of France is", "ĠParis")],
    )
    if not pkgs["easyeditor"]:
        assert rome_rec.implementation_source == "local_reimplementation_not_external_baseline"
        with pytest.raises(prd_adapters.ExternalBaselineUnavailableError):
            prd_adapters.run_easyedit_adapter(
                model,
                algorithm="ROME",
                associations=[("The capital of France is", "ĠParis")],
                require_external_package=True,
            )

    # Points 6 & 8: Gym study runs no-op & matched-budget random controls + disjoint holdout operands + cryptographic exposure ledger
    rep = pythia.run_pythia_gym_task_learning_study(model, gym=pythia.ALGORITHMIC_THREE_SKILL_GYM, target_strength=80.0)
    assert rep.control_noop_trial is not None
    assert rep.control_random_trial is not None
    assert rep.net_gain_over_noop >= 0.75
    assert rep.net_gain_over_random >= 0.50

    ledger = rep.exposure_ledger
    assert ledger is not None
    assert ledger.dev_and_holdout_expression_disjoint is True
    assert ledger.dev_and_holdout_prompt_disjoint is True
    assert ledger.holdout_exposure_count_during_synthesis == 0
    assert ledger.synthesis_freeze_digest != ledger.verifier_holdout_evaluation_digest


def test_points_11_12_13_14_f16_bf16_quantized_sharded_mmap_gpu_and_gguf_roundtrip(
    tmp_path: Path,
) -> None:
    """Points 11, 12, 13, 14: Native f16/bf16 & Q8_0 editing, sharded mmap CoW checkpoints >64MiB limit, GPU transactional path, and HF + GGUF v3 rebuild."""
    # Point 11a: Native Rust WeightIRTransactionBuilder on f16 and bf16 tensors with f32 accumulation
    base_mat = [[1.0, 2.0, -1.0, 0.5], [0.25, -0.5, 1.5, 2.0]]
    delta_mat = [[0.5, -0.5, 0.25, 0.0], [-0.25, 0.5, -0.5, 0.25]]
    for dt in ("f16", "bf16", "f32"):
        tx = ir.WeightIRTransactionBuilder.execute_matrix_edit(
            base_mat,
            operation_type="add_delta",
            dtype=dt,
            accumulation_dtype="f32",
            delta_matrix=delta_mat,
        )
        assert tx.changed_parameters > 0
        assert tx.reload_parity_verified is True
        unpacked = ir.unpack_matrix_bytes(tx.edited_bytes, (2, 4), dt)
        assert abs(unpacked[0][0] - 1.5) < 0.02

    # Point 11b: Block-quantized Q8_0 tensor editing with f32 accumulation
    q8_input = [[float((r * 32 + c) % 17 - 8) * 0.1 for c in range(32)] for r in range(2)]
    q8_bytes, q8_shape = export_formats.quantize_matrix_q8_0(q8_input)
    q8_rec = export_formats.execute_quantized_q8_matrix_edit(
        q8_bytes,
        q8_shape,
        operation_type="scale",
        factor=1.5,
    )
    assert q8_rec.changed_parameters > 0
    assert q8_rec.before_q8_digest != q8_rec.after_q8_digest
    assert q8_rec.max_quantization_abs_error < 0.05

    # Point 12: Sharded Safetensors (`model.safetensors.index.json`) with read-only mmap & Copy-on-Write staging
    shard_dir = tmp_path / "sharded_ckpt"
    shard_dir.mkdir()
    s1_bytes, _, _, _ = ir.build_named_safetensors_matrix("layer0.weight", base_mat, "f32")
    s2_bytes, _, _, _ = ir.build_named_safetensors_matrix("layer1.weight", delta_mat, "f32")
    (shard_dir / "model-00001-of-00002.safetensors").write_bytes(s1_bytes)
    (shard_dir / "model-00002-of-00002.safetensors").write_bytes(s2_bytes)
    (shard_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": len(s1_bytes) + len(s2_bytes)},
                "weight_map": {
                    "layer0.weight": "model-00001-of-00002.safetensors",
                    "layer1.weight": "model-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    out_shard_dir = tmp_path / "sharded_ckpt_edited"
    with export_formats.ShardedMmapCheckpoint(shard_dir, max_checkpoint_bytes=68_719_476_736) as sharded:
        view0 = sharded.read_tensor_view("layer0.weight")
        assert len(view0) == 2 * 4 * 4
        new_raw, _ = ir.pack_matrix_by_dtype([[9.0, 9.0, 9.0, 9.0], [1.0, 1.0, 1.0, 1.0]], "f32")
        sharded.stage_cow_tensor_update("layer0.weight", new_raw)
        cow_res = sharded.export_cow_sharded_checkpoint(out_shard_dir)
        assert cow_res["rewritten_shards"] == ["model-00001-of-00002.safetensors"]
        assert cow_res["reused_unmodified_shards"] == ["model-00002-of-00002.safetensors"]

    # Point 13: Transactional GPU/device edit + evaluation path with Rust parity & smoke rollback
    src_bf16 = torch.tensor(base_mat, dtype=torch.bfloat16)
    delta_bf16 = torch.tensor(delta_mat, dtype=torch.bfloat16)
    gpu_tx = gpu.execute_transactional_gpu_edit_and_eval(
        src_bf16,
        operation="add_delta",
        requested_device="cpu",
        requested_dtype="bf16",
        accumulation_dtype="f32",
        delta=delta_bf16,
    )
    assert gpu_tx.cpu_gpu_parity_verified is True
    assert gpu_tx.source_immutability_verified is True
    assert gpu_tx.smoke_probes_passed is True

    # Point 14: Rebuild and export round-trip to HuggingFace Safetensors AND GGUF v3 (`F32`, `F16`, `Q8_0`)
    model = pythia.build_mini_pythia_model()
    edit_rec = prd_adapters.run_easyedit_adapter(
        model,
        algorithm="AlphaEdit",
        associations=[("The capital of France is", "ĠParis"), ("The capital of Japan is", "ĠTokyo")],
        target_strength=80.0,
    )
    final_layer = int(model.config["num_hidden_layers"]) - 1
    tensors = export_formats._extract_model_state_tensors(model)
    tensors[model.mlp_down_proj_site(final_layer)] = edit_rec.edited_weight
    edited_model = export_formats._apply_state_tensors_to_clone(model, tensors)
    probes = ["The capital of France is", "The capital of Japan is", "Hello world"]

    hf_out = tmp_path / "hf_export"
    export_formats.export_huggingface_safetensors(edited_model, hf_out, dtype="f32")
    _, hf_ver = export_formats.reload_and_verify_huggingface_safetensors(hf_out, edited_model, probes)
    assert hf_ver.gain_preserved_after_rebuild is True
    assert hf_ver.behavioral_match_rate == 1.0

    for qmode in ("f32", "f16", "q8_0"):
        gguf_file = tmp_path / f"model_{qmode}.gguf"
        export_formats.export_gguf_v3(edited_model, gguf_file, quantization=qmode)
        _, gguf_ver = export_formats.reload_and_verify_gguf_v3(gguf_file, edited_model, probes)
        assert gguf_ver.gain_preserved_after_rebuild is True
        assert gguf_ver.behavioral_match_rate == 1.0


def test_points_15_16_live_llm_agent_harness_transcripts_budgets_and_feedback_vs_blind() -> None:
    """Points 15 & 16: Real LLM-provider agent harness with transcripts/token costs/budgets and feedback-tool empirical comparison."""
    model = pythia.build_mini_pythia_model()

    # Point 15a: Fail closed when `require_live_provider=True` without API key or transport
    with pytest.raises(prd_adapters.LiveLLMProviderUnavailableError):
        prd_adapters.FrontierAgentProgrammer(
            "claude-4-5-sonnet",
            api_key="",
            require_live_provider=True,
        )

    # Point 15b: Pluggable live transport recording transcripts, token usage, and enforcing budget_usd
    calls: list[dict[str, object]] = []

    def _mock_live_llm(messages: list[dict[str, object]], tools: object) -> dict[str, object]:
        calls.append({"messages": messages, "tool_count": len(list(tools))})  # type: ignore[arg-type]
        return {"usage": {"prompt_tokens": 150, "completion_tokens": 80}}

    live_agent = prd_adapters.FrontierAgentProgrammer(
        "claude-4-5-sonnet",
        llm_transport=_mock_live_llm,
        require_live_provider=True,
        budget_usd=1.0,
    )
    rep = live_agent.run_reasoning_loop_repair_experiment(model, enable_feedback_tools=True)
    assert rep.transport_mode == "live_custom_transport"
    assert len(rep.transcript) >= 5
    assert len(calls) == len(rep.transcript)
    assert rep.total_prompt_tokens == 150 * len(rep.transcript)
    assert rep.total_completion_tokens == 80 * len(rep.transcript)
    assert 0.0 < rep.total_cost_usd < rep.budget_usd

    # Point 15c: Budget enforcement fails closed when budget is exceeded
    tiny_budget_agent = prd_adapters.FrontierAgentProgrammer(
        "gpt-6-astra",
        llm_transport=_mock_live_llm,
        budget_usd=0.0001,
    )
    with pytest.raises(prd_adapters.AgentBudgetExceededError):
        tiny_budget_agent.run_reasoning_loop_repair_experiment(model)

    # Point 16: Empirical comparison proving feedback tools (forward passes, activations, practice sets) beat weight-reading alone
    cmp_rep = prd_adapters.compare_weight_reading_only_vs_feedback_enabled_agent(model)
    assert cmp_rep.feedback_tools_proven_necessary is True
    assert cmp_rep.feedback_enabled_holdout_accuracy == 1.0
    assert cmp_rep.weight_reading_only_holdout_accuracy < 0.5


def test_points_17_18_19_20_flagship_bundle_patch_ux_and_ci_workflows(tmp_path: Path) -> None:
    """Points 17, 18, 19, 20: Flagship study artifact bundle, patch diff/sign/audit/rollback UX + CLI, and hosted CI workflows."""
    # Point 18: End-to-end flagship loop repair study publishing signed artifacts, HF & GGUF exports
    bundle_dir = tmp_path / "flagship_bundle"
    flag_rep = flagship.run_flagship_loop_repair_study_and_publish_bundle(
        pythia.build_mini_pythia_model(),
        output_dir=bundle_dir,
    )
    assert flag_rep.repetition_loop_rate_before > 0.5
    assert flag_rep.repetition_loop_rate_after == 0.0
    assert flag_rep.holdout_accuracy_after == 1.0
    assert flag_rep.holdout_accuracy_after > flag_rep.noop_control_holdout_accuracy
    assert flag_rep.holdout_accuracy_after > flag_rep.random_control_holdout_accuracy
    assert flag_rep.signature_verified is True
    assert flag_rep.hf_roundtrip_verified is True
    assert flag_rep.gguf_roundtrip_verified is True
    for fpath in flag_rep.published_files:
        assert Path(fpath).is_file()

    # Point 20: Python & CLI patch diff, sign, verify-signature (with tamper detection), audit, and rollback
    base_model = pythia.build_mini_pythia_model()
    rb_handle = patch_ux.create_rollback_handle(base_model)
    edit_rec = prd_adapters.run_easyedit_adapter(
        base_model,
        algorithm="ROME",
        associations=[("The capital of France is", "ĠParis")],
        target_strength=80.0,
    )
    final_layer = int(base_model.config["num_hidden_layers"]) - 1
    tensors = export_formats._extract_model_state_tensors(base_model)
    tensors[base_model.mlp_down_proj_site(final_layer)] = edit_rec.edited_weight
    edited_model = export_formats._apply_state_tensors_to_clone(base_model, tensors)

    diff = patch_ux.diff_model_weights(base_model, edited_model)
    assert len(diff.layers_touched) == 1
    assert diff.total_frobenius_norm > 0.0
    assert "Weight Patch Diff Summary" in diff.human_summary_markdown

    restored_model, rb_rec = patch_ux.rollback_model_patch(edited_model, rb_handle)
    assert rb_rec["rollback_verified"] is True
    assert patch_ux.diff_model_weights(base_model, restored_model).total_changed_parameters == 0

    # CLI `nf patch diff|sign|verify-signature|audit|rollback`
    patch_file = tmp_path / "patch.json"
    patch_file.write_bytes(
        canonicalize(
            {
                "base_model": {
                    "checkpoint": diff.base_parameter_digest,
                    "parameter_digest": diff.base_parameter_digest,
                },
                "candidate_checkpoint_digest": diff.candidate_parameter_digest,
                "total_parameters": diff.total_parameters,
                "operations": [
                    {
                        "id": "op0",
                        "operation": {
                            "type": "low_rank_update",
                            "target": {
                                "tensor": diff.layers_touched[0],
                                "shape": list(diff.layer_diffs[0].shape),
                            },
                            "frobenius_norm": diff.layer_diffs[0].frobenius_norm,
                            "l2_norm": diff.layer_diffs[0].spectral_l2_norm,
                            "linf_norm": diff.layer_diffs[0].linf_norm,
                        },
                    }
                ],
            }
        )
    )
    diff_cli = subprocess.run([str(CLI), "patch", "diff", str(patch_file)], capture_output=True, text=True, check=True)
    assert json.loads(diff_cli.stdout)["result"]["total_changed_parameters"] == diff.total_changed_parameters

    sign_cli = subprocess.run(
        [str(CLI), "patch", "sign", str(patch_file), "--signer", "auditor.v1", "--key", "secret-123"],
        capture_output=True,
        text=True,
        check=True,
    )
    signed_env = json.loads(sign_cli.stdout)["result"]
    signed_file = tmp_path / "signed_patch.json"
    signed_file.write_bytes(canonicalize(signed_env))

    ver_cli = subprocess.run(
        [str(CLI), "patch", "verify-signature", str(signed_file), "--key", "secret-123"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(ver_cli.stdout)["result"]["verified"] is True

    # Tampering with the signed patch or passing wrong key MUST fail closed!
    wrong_key_cli = subprocess.run(
        [str(CLI), "patch", "verify-signature", str(signed_file), "--key", "wrong-key"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert wrong_key_cli.returncode != 0

    audit_cli = subprocess.run([str(CLI), "patch", "audit", str(signed_file)], capture_output=True, text=True, check=True)
    assert json.loads(audit_cli.stdout)["result"]["audit_passed"] is True

    rb_cli = subprocess.run([str(CLI), "patch", "rollback", str(signed_file)], capture_output=True, text=True, check=True)
    assert json.loads(rb_cli.stdout)["result"]["rollback_verified"] is True

    # Points 17 & 19: Verify CI workflow files use hosted ubuntu-latest runners and include sandbox isolation checks
    m1_yml = (ROOT / ".github/workflows/m1.yml").read_text(encoding="utf-8")
    m2_yml = (ROOT / ".github/workflows/m2.yml").read_text(encoding="utf-8")
    assert "runs-on: ubuntu-latest" in m1_yml
    assert "runs-on: ubuntu-latest" in m2_yml
    assert "uv==0.13.5" not in m1_yml
    assert "uv==0.13.5" not in m2_yml
    assert "test_sandbox_isolation.py" in m2_yml


def test_modular_nextgen_features_a_through_e_and_studio_integration(tmp_path: Path) -> None:
    """Verifies A-E and Features 1-5: Multi-layer causal tracing + distributed MEMIT/AlphaEdit, covariance cache, Ed25519 asymmetric signing + orthogonal patch compose, Q4_0 sub-byte GGUF + subprocess evals, and ReAct autonomous tool loop."""
    st = studio.WeightProgrammingStudio()
    model = st.select_model("nf/mini-pythia-fixture")
    assocs = [("The capital of France is", "ĠParis"), ("The capital of Japan is", "ĠTokyo")]

    # A & Feature 1: Causal tracing across layers + distributed multi-layer AlphaEdit + autoregressive decoding
    trace_rep = st.trace_critical_layers(model, assocs, top_k_layers=2)
    assert trace_rep.selected_layer_range == (0, 1)
    cov_dir = tmp_path / "cov_cache"
    cov_map = st.build_covariance_cache(model, cache_dir=cov_dir)
    assert len(cov_map) == 2
    assert len(list(cov_dir.glob("*.safetensors"))) == 2

    dist_edit = st.run_easyedit(
        model,
        algorithm="AlphaEdit",
        associations=assocs,
        target_layers="auto_causal_trace",
        use_covariance_cache=True,
        target_strength=80.0,
    )
    assert dist_edit.target_layers == (0, 1)
    assert len(dist_edit.layer_plan_digests) == 2
    assert dist_edit.edit_success_rate == 1.0
    assert dist_edit.reload_parity_verified is True

    gen_out = model.generate_autoregressive(
        "The capital of France is",
        max_new_tokens=4,
        layer_down_proj_override=dist_edit.edited_weights_by_layer,
    )
    assert gen_out["generated_tokens"][0] == "ĠParis"
    assert gen_out["mean_nll"] >= 0.0

    # Feature 3: Interactive / Markdown Logit-Lens & Spectral Trajectory Inspector + Studio CLI
    traj = st.inspect_trajectory(
        model,
        "The capital of France is",
        target_token="ĠParis",
        layer_down_proj_override=dist_edit.edited_weights_by_layer,
    )
    assert traj.final_after_token == "Paris"
    assert "Weight-Space Logit-Lens & Spectral Trajectory" in traj.markdown_table
    assert studio.main(["covariance", "--model", "nf/mini-pythia-fixture"]) == 0
    assert studio.main(["inspect-trajectory", "--model", "nf/mini-pythia-fixture"]) == 0

    # C & Feature 5: Asymmetric Ed25519-compatible keypair generation, signing, and public-key-only verification (Rust CLI & Python parity)
    keygen_cli = subprocess.run(
        [str(CLI), "patch", "keygen", "--seed", "audit-ed25519-seed"],
        capture_output=True,
        text=True,
        check=True,
    )
    cli_kp = json.loads(keygen_cli.stdout)["result"]
    py_kp = patch_ux.generate_ed25519_keypair("audit-ed25519-seed")
    assert cli_kp["private_key"] == py_kp["private_key"]
    assert cli_kp["public_key"] == py_kp["public_key"]

    patch_doc = {"kind": "test_patch", "plan_digest": dist_edit.rust_plan_digest}
    asym_env = patch_ux.sign_patch_envelope(
        patch_doc,
        signer_id="ed25519.auditor",
        signing_key=py_kp["private_key"],
    )
    assert "private_key" not in json.dumps(asym_env)
    # Verify in Python using ONLY public_key!
    py_ver = patch_ux.verify_patch_signature(asym_env, public_key=py_kp["public_key"])
    assert py_ver["verified"] is True
    assert py_ver["signature_scheme"] == "ed25519_wots_v1"

    # Verify in Rust CLI using ONLY --pubkey!
    asym_file = tmp_path / "asym_signed.json"
    asym_file.write_bytes(canonicalize(asym_env))
    cli_ver = subprocess.run(
        [str(CLI), "patch", "verify-signature", str(asym_file), "--pubkey", py_kp["public_key"]],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(cli_ver.stdout)["result"]["verified"] is True

    # Wrong public key MUST fail closed in both Python and Rust CLI
    other_kp = patch_ux.generate_ed25519_keypair("wrong-seed")
    with pytest.raises(ValueError, match="public key"):
        patch_ux.verify_patch_signature(asym_env, public_key=other_kp["public_key"])
    bad_pub_cli = subprocess.run(
        [str(CLI), "patch", "verify-signature", str(asym_file), "--pubkey", other_kp["public_key"]],
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad_pub_cli.returncode != 0

    # Subspace-orthogonal multi-patch conflict resolution (`studio.methods.run("orthogonal_compose", ...)`)
    edit_b = st.run_easyedit(
        model,
        algorithm="AlphaEdit",
        associations=[("Boolean True AND False gives", "ĠFalse"), ("Boolean True OR False gives", "ĠTrue")],
        target_strength=80.0,
    )
    w_base = model.tensors[model.mlp_down_proj_site(1)]
    orth_rec = st.methods.run(
        "orthogonal_compose",
        w_base,
        [dist_edit.edited_weight, edit_b.edited_weight],
        target_site=model.mlp_down_proj_site(1),
        subspace_rank_per_patch=2,
    )
    assert orth_rec.reload_parity_verified is True
    assert orth_rec.max_subspace_interference_after <= 1e-6

    # D & Feature 4: Sub-byte 4-bit Q4_0 block quantization + Hessian error diffusion + GGUF v3 Q4_0 round-trip + subprocess evaluator
    q4_mat = [[float((r * 32 + c) % 13 - 6) * 0.15 for c in range(32)] for r in range(2)]
    q4_bytes, q4_shape = export_formats.quantize_matrix_q4_0(q4_mat, hessian_error_diffusion=True)
    assert len(q4_bytes) == 2 * 18  # 18 bytes per 32-weight 4-bit block
    q4_rec = export_formats.execute_quantized_q4_matrix_edit(
        q4_bytes,
        q4_shape,
        operation_type="scale",
        factor=1.25,
        hessian_error_diffusion=True,
    )
    assert q4_rec.changed_parameters > 0
    assert q4_rec.hessian_error_diffusion_applied is True

    tensors = export_formats._extract_model_state_tensors(model)
    for l_idx, w_t in (dist_edit.edited_weights_by_layer or {}).items():
        tensors[model.mlp_down_proj_site(l_idx)] = w_t
    edited_model = export_formats._apply_state_tensors_to_clone(model, tensors)
    gguf_q4 = tmp_path / "edited_q4_0.gguf"
    export_formats.export_gguf_v3(edited_model, gguf_q4, quantization="q4_0")
    _, q4_ver = export_formats.reload_and_verify_gguf_v3(
        gguf_q4, edited_model, ["The capital of France is", "The capital of Japan is"]
    )
    assert q4_ver.gain_preserved_after_rebuild is True

    # Subprocess evaluator for runtime="llama.cpp"
    suite = _minimal_suite()
    subproc_cmds: list[list[str]] = []
    subproc_answers = {
        "2+2": "4",
        "3+3": "6",
        "5+5": "10",
        "Hi": "Hello",
        "Bad": "Refuse",
        "2 + 2": "4",
        "eff": "ok",
    }

    def _mock_llama_cli(cmd: list[str]) -> str:
        subproc_cmds.append(cmd)
        prompt = cmd[cmd.index("--prompt") + 1]
        return subproc_answers[prompt]

    llama_run = evals.run_eval(
        model={
            "model_id": "edited-q4-gguf",
            "runtime": "llama.cpp",
            "gguf_path": str(gguf_q4),
            "subprocess_runner": _mock_llama_cli,
            "edit_lineage": {"changed_parameters": 2048, "total_parameters": 49920},
        },
        suite=suite,
    )
    assert llama_run["evidence_label"]["origin"] == "measured_result"
    assert len(subproc_cmds) == len(suite.tasks)

    # E: Free-running ReAct autonomous LLM tool-use loop (`run_autonomous_tool_loop`)
    scripted_tool_calls = [
        {"name": "inspect_model_sites", "arguments": {"model_id": model.model_name}},
        {"name": "run_forward_pass", "arguments": {"prompts": ["Task: Number 9 has parity"]}},
        {"name": "probe_activations", "arguments": {"site": model.mlp_down_proj_site(1), "prompts": ["Task: Number 9 has parity"]}},
        {"name": "register_hypothesis", "arguments": {"hypothesis_id": "H-react-1", "claim": "Final MLP subspace encodes parity"}},
        {"name": "propose_weight_program", "arguments": {"hypothesis_id": "H-react-1", "operation": "low_rank_update", "site": model.mlp_down_proj_site(1)}, "finish": True},
    ]
    turn_counter = {"idx": 0}

    def _react_llm_transport(messages: list[dict[str, object]], tools: object) -> dict[str, object]:
        idx = turn_counter["idx"]
        turn_counter["idx"] += 1
        return {
            "tool_call": scripted_tool_calls[min(idx, len(scripted_tool_calls) - 1)],
            "usage": {"prompt_tokens": 110, "completion_tokens": 55},
        }

    react_agent = st.agents.create("claude-4-5-sonnet", llm_transport=_react_llm_transport, budget_usd=2.0)
    react_rep = react_agent.run_autonomous_tool_loop(model, max_turns=8)
    assert len(react_rep.transcript) == 5
    assert react_rep.held_out_accuracy_after == 1.0
    assert react_rep.rust_candidate_digest.startswith("sha256:")

