"""Comprehensive tests for M4 Weight-Programming Modalities, Baselines, Adapters, and Library Reuse (T065–T099)."""

from __future__ import annotations

import hashlib
import hmac

import pytest
import torch
from neural_foundry.m4_ecosystems import (
    CommunityGymPackage,
    build_proposer_dataset,
    enforce_private_gym_policy,
    evaluate_continual_protocol,
    evaluate_cross_model_protocol,
    evaluate_learned_proposer_inference,
    evaluate_library_recipe_reuse,
    evaluate_measurement_research,
    ingest_research_source_record,
    retrieve_repair_strategies,
    run_distillation_baseline,
    run_gymnasium_adapter,
    run_lora_baseline,
    run_m4_feature_acceptance_suite,
    run_nemo_adapter,
    run_proposer_training_study,
    run_rl_baseline,
    run_sft_baseline,
    run_transfer_benchmark,
    run_verifiers_adapter,
    verify_community_gym_package,
    verify_external_memory_method,
)
from neural_foundry.m4_modalities import (
    GymExperimentCell,
    M4ContractError,
    MemoryRecord,
    ParetoCandidate,
    ResearchMemoryStore,
    SparseFeatureBasis,
    analyze_gym_interference,
    bind_sparse_feature_intervention,
    compute_pareto_population_frontiers,
    construct_algorithmic_relu_rule,
    estimate_permutation_alignment,
    evaluate_mechanism_confirmation,
    execute_mapped_weight_transfer,
    run_circuit_tracing_adapter,
    solve_fitted_linear_alignment,
)


def test_research_memory_access_classes_contradictions_and_no_claim_inflation() -> None:
    """T065 (`research_memory`): preserves access classes, contradictions, and prevents claim-to-fact promotion."""
    store = ResearchMemoryStore()
    store.append(
        MemoryRecord(
            record_id="rec-dev-ok",
            model_family="hf.qwen2.dense",
            architecture_site="layers.0.mlp",
            failure_mode="carry_overflow",
            outcome="success",
            cost_cpu_ms=120,
            origin="measured_result",
            support="independently_confirmed_under_protocol",
            access="development",
            summary="Rank-1 subspace projection repaired carry overflow",
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-dev-contra",
            model_family="hf.qwen2.dense",
            architecture_site="layers.0.mlp",
            failure_mode="carry_overflow",
            outcome="contradiction",
            cost_cpu_ms=95,
            origin="measured_result",
            support="contradicted",
            access="development",
            summary="Failed on 4-digit operands",
        )
    )
    store.append(
        MemoryRecord(
            record_id="rec-holdout-secret",
            model_family="hf.qwen2.dense",
            architecture_site="layers.0.mlp",
            failure_mode="carry_overflow",
            outcome="success",
            cost_cpu_ms=80,
            origin="measured_result",
            support="independently_confirmed_under_protocol",
            access="holdout",
            summary="Holdout evaluation",
        )
    )

    res = store.query_for_proposer(failure_mode="carry_overflow")
    assert len(res.records) == 2
    assert res.supporting_count == 1
    assert res.contradicting_count == 1
    assert res.total_cost_cpu_ms == 215

    # Repeating a model_claim keeps it as model_claim, never promoting to supplied_fact
    claim = MemoryRecord(
        record_id="claim-1",
        model_family="hf.qwen2.dense",
        architecture_site="layers.1.attn",
        failure_mode="carry_overflow",
        outcome="success",
        cost_cpu_ms=10,
        origin="model_claim",
        support="untested",
        access="development",
        summary="Agent hypothesizes head 2 is responsible",
    )
    repeated = store.repeat_claim(claim, repetitions=4)
    assert repeated.origin == "model_claim"
    assert repeated.support == "untested"

    # Negative: proposer cannot query holdout or private access classes
    with pytest.raises(M4ContractError) as exc_info:
        store.query_for_proposer(failure_mode="carry_overflow", allowed_access=("development", "holdout"))
    assert exc_info.value.code == "PROPOSER_ACCESS_VIOLATION"


def test_circuit_tracing_adapter_maps_real_fixture_and_rejects_mocks() -> None:
    """T066 (`circuit_tracing_adapter`): runs real local fixture tracing and rejects mocked or unpinned runs."""
    weights = {
        "w1": torch.tensor([[1.0, 0.5], [-0.5, 2.0]], dtype=torch.float64),
        "w2": torch.tensor([[1.5, -1.0]], dtype=torch.float64),
    }
    probe = torch.tensor([2.0, 1.0], dtype=torch.float64)
    trace = run_circuit_tracing_adapter(
        adapter_id="circuit_tracer.local",
        adapter_version="1.0.0",
        weights=weights,
        probe_input=probe,
    )
    assert len(trace.nodes) == 2
    assert len(trace.edges) == 2
    assert trace.origin == "measured_result"

    with pytest.raises(M4ContractError) as exc_info:
        run_circuit_tracing_adapter(
            adapter_id="circuit_tracer.local",
            adapter_version="1.0.0",
            weights=weights,
            probe_input=probe,
            is_mocked_trace=True,
        )
    assert exc_info.value.code == "MOCK_TRACE_REJECTED"

    with pytest.raises(M4ContractError) as exc_info:
        run_circuit_tracing_adapter(
            adapter_id="circuit_tracer.local",
            adapter_version="9.9.9",
            weights=weights,
            probe_input=probe,
        )
    assert exc_info.value.code == "UNVALIDATED_TRACER_VERSION"


def test_sparse_features_provenance_and_site_compatibility() -> None:
    """T067 (`sparse_features`): loads authorized SAE/transcoder bases and rejects incompatible site/precision."""
    basis = SparseFeatureBasis(
        basis_id="sae.l0.mlp.v1",
        target_site="model.layers.0.mlp.down_proj.weight",
        d_model=4,
        n_features=3,
        dtype="f64",
        layout="row_major_decoder",
        encoder=torch.ones(4, 3, dtype=torch.float64) * 0.25,
        decoder=torch.ones(3, 4, dtype=torch.float64) * 0.5,
        training_corpus_digest="sha256:corpus123",
        reconstruction_mse=0.02,
    )
    interv = bind_sparse_feature_intervention(
        basis,
        site_name="model.layers.0.mlp.down_proj.weight",
        site_shape=(4, 4),
        site_dtype="f64",
        feature_index=1,
        scale=-0.5,
    )
    assert tuple(interv.weight_delta.shape) == (4, 4)

    with pytest.raises(M4ContractError) as exc_info:
        bind_sparse_feature_intervention(
            basis,
            site_name="model.layers.1.mlp.down_proj.weight",
            site_shape=(4, 4),
            site_dtype="f64",
            feature_index=1,
            scale=-0.5,
        )
    assert exc_info.value.code == "BASIS_SITE_MISMATCH"

    with pytest.raises(M4ContractError) as exc_info:
        bind_sparse_feature_intervention(
            basis,
            site_name="model.layers.0.mlp.down_proj.weight",
            site_shape=(4, 4),
            site_dtype="f32",
            feature_index=1,
            scale=-0.5,
        )
    assert exc_info.value.code == "BASIS_PRECISION_MISMATCH"


def test_mechanism_confirmation_separates_behavioral_repair_from_causal_explanation() -> None:
    """T068 (`mechanism_confirmation`): repaired behavior with false explanation keeps split verdicts."""
    # Case 1: Both behavior and mechanism confirmed
    both_ok = evaluate_mechanism_confirmation(
        claim_id="claim-true-mech",
        target_behavior_delta=0.25,
        on_site_intervention_effect=0.40,
        off_site_control_effect=0.02,
        reserved_challenge_effect=0.30,
        competing_alternative_effect=0.05,
    )
    assert both_ok.behavioral_repair_confirmed is True
    assert both_ok.mechanism_explanation_confirmed is True

    # Case 2: Repaired behavior, but false mechanism explanation (fails reserved challenge)
    split_verdict = evaluate_mechanism_confirmation(
        claim_id="claim-false-mech",
        target_behavior_delta=0.22,
        on_site_intervention_effect=0.35,
        off_site_control_effect=0.02,
        reserved_challenge_effect=-0.05,
        competing_alternative_effect=0.30,
    )
    assert split_verdict.behavioral_repair_confirmed is True
    assert split_verdict.mechanism_explanation_confirmed is False
    assert split_verdict.behavior_support_grade == "independently_confirmed_under_protocol"
    assert split_verdict.mechanism_support_grade == "contradicted"

    # Negative: conflating behavior as mechanism proof
    with pytest.raises(M4ContractError) as exc_info:
        evaluate_mechanism_confirmation(
            claim_id="bad",
            target_behavior_delta=0.2,
            on_site_intervention_effect=0.1,
            off_site_control_effect=0.1,
            reserved_challenge_effect=0.1,
            competing_alternative_effect=0.1,
            claim_mechanism_from_behavior_only=True,
        )
    assert exc_info.value.code == "MECHANISM_BEHAVIOR_CONFLATION"


def test_pareto_population_maintains_provisional_and_verified_frontiers_with_noisy_ties() -> None:
    """T071 (`pareto_population`): separates provisional vs verified frontiers and preserves uncertain ties."""
    candidates = [
        ParetoCandidate(
            candidate_id="cand-fast-compact",
            quality_gain=0.15,
            quality_ci_half_width=0.02,
            retention_delta=0.0,
            latency_ratio=0.95,
            memory_ratio=1.0,
            changed_parameters=4,
            feasible=True,
            verified_on_holdout=True,
        ),
        ParetoCandidate(
            candidate_id="cand-high-qual-noisy",
            quality_gain=0.16,
            quality_ci_half_width=0.02,
            retention_delta=0.0,
            latency_ratio=1.02,
            memory_ratio=1.0,
            changed_parameters=8,
            feasible=True,
            verified_on_holdout=False,
        ),
        ParetoCandidate(
            candidate_id="cand-dominated",
            quality_gain=0.05,
            quality_ci_half_width=0.01,
            retention_delta=-0.005,
            latency_ratio=1.10,
            memory_ratio=1.05,
            changed_parameters=16,
            feasible=True,
            verified_on_holdout=True,
        ),
        ParetoCandidate(
            candidate_id="cand-infeasible-elite",
            quality_gain=0.45,
            quality_ci_half_width=0.01,
            retention_delta=-0.15,
            latency_ratio=0.90,
            memory_ratio=1.0,
            changed_parameters=4,
            feasible=False,
            verified_on_holdout=True,
        ),
    ]
    report = compute_pareto_population_frontiers(candidates)
    assert "cand-infeasible-elite" in report.infeasible_rejected_ids
    assert "cand-dominated" not in report.provisional_frontier_ids
    assert set(report.provisional_frontier_ids) == {"cand-fast-compact", "cand-high-qual-noisy"}
    assert report.verified_frontier_ids == ("cand-fast-compact",)
    assert len(report.uncertain_nondominated_pairs) == 1

    with pytest.raises(M4ContractError) as exc_info:
        compute_pareto_population_frontiers(candidates, allow_arbitrary_scalarization=True)
    assert exc_info.value.code == "SCALARIZATION_FORBIDDEN"


def test_gym_interference_joins_paired_experiments_without_imputing_missing_cells() -> None:
    """T072 (`gym_interference`): computes synergy/interference and leaves missing cells explicit."""
    cells = [
        GymExperimentCell("A", "gym.sum", "g1", 1, 0.10),
        GymExperimentCell("B", "gym.sum", "g1", 1, 0.05),
        GymExperimentCell("A+B", "gym.sum", "g1", 1, 0.22),
        GymExperimentCell("A", "gym.max", "g1", 1, 0.12),
    ]
    matrix = analyze_gym_interference(cells, required_gyms=("gym.sum", "gym.max"))
    assert matrix["gym.sum"].classification == "synergy"
    assert matrix["gym.sum"].interaction_synergy == pytest.approx(0.07)
    assert matrix["gym.max"].classification == "incomplete_missing_cell"
    assert matrix["gym.max"].interaction_synergy is None

    with pytest.raises(M4ContractError) as exc_info:
        analyze_gym_interference(cells, required_gyms=("gym.sum",), impute_missing_cells=True)
    assert exc_info.value.code == "MISSING_CELL_IMPUTATION_FORBIDDEN"

    with pytest.raises(M4ContractError) as exc_info:
        analyze_gym_interference(
            cells,
            required_gyms=("gym.sum",),
            claim_causal_circuit_sharing_from_correlation=True,
        )
    assert exc_info.value.code == "CORRELATION_NOT_CIRCUIT_SHARING"


def test_rule_construction_compiles_exact_relu_weights_with_independent_oracle() -> None:
    """T073 (`rule_construction`, CG-09): compiles algorithmic rule and verifies against independent oracle."""
    def ref_oracle(x: float, y: float) -> tuple[float, float, float, float, float]:
        return (x + y, x - y, max(x, y), min(x, y), abs(x - y))

    art = construct_algorithmic_relu_rule(independent_oracle=ref_oracle)
    assert art.normal_pass_rate == 1.0
    assert art.boundary_pass_rate == 1.0
    assert art.invalid_rejected_properly is True
    assert art.integrated_gym_passed is True
    assert art.reload_parity_passed is True

    with pytest.raises(M4ContractError) as exc_info:
        construct_algorithmic_relu_rule(independent_oracle=ref_oracle, share_constructor_as_oracle=True)
    assert exc_info.value.code == "SHARED_ORACLE_FORBIDDEN"


def test_mapped_transfer_permutation_alignment_fitted_alignment_and_transfer_protocols() -> None:
    """T074 (`mapped_transfer`), T075 (`permutation_alignment`), T076 (`fitted_alignment`), T077 (`transfer_protocols`)."""
    t_w1 = torch.tensor([[1.0, 2.0], [-1.0, 1.5], [0.5, -0.5]], dtype=torch.float64)
    t_w2 = torch.tensor([[1.0, -0.5, 2.0]], dtype=torch.float64)
    perm = (2, 0, 1)
    probe_x = torch.tensor(
        [[1.0, 0.5], [-0.5, 2.0], [1.5, -1.0], [0.2, 0.8], [-1.2, -0.4]],
        dtype=torch.float64,
    )
    t_act = torch.relu(t_w1 @ probe_x.T)
    s_act = t_act[list(perm), :]

    # T075: Permutation alignment
    align = estimate_permutation_alignment(t_act, s_act, mode="inferred_from_activations")
    assert align.permutation == perm
    assert align.mean_cosine_similarity > 0.99

    with pytest.raises(M4ContractError) as exc_info:
        estimate_permutation_alignment(t_act, s_act, calibration_split="holdout")
    assert exc_info.value.code == "CALIBRATION_SPLIT_LEAKAGE"

    # T076: Fitted alignment
    fitted = solve_fitted_linear_alignment(
        t_act.T,
        s_act.T,
        contract_allows_fitted_alignment=True,
        record_hybrid_lineage=True,
    )
    assert fitted.alignment_fitted is True
    assert fitted.lineage_production_history == "hybrid"

    with pytest.raises(M4ContractError) as exc_info:
        solve_fitted_linear_alignment(t_act.T, s_act.T, contract_allows_fitted_alignment=False)
    assert exc_info.value.code == "DIRECT_ONLY_CONTRACT_VIOLATION"

    # T074 & T077: Mapped transfer and scope protocols
    target_y = (torch.relu(probe_x @ t_w1.T)) @ t_w2.T
    rep = execute_mapped_weight_transfer(
        teacher_w1=t_w1,
        teacher_w2=t_w2,
        student_w1=torch.zeros_like(t_w1),
        student_w2=torch.zeros_like(t_w2),
        alignment=align,
        eval_inputs=probe_x,
        eval_targets=target_y,
        target_scope="same_architecture",
        is_permuted_synthetic_fixture=True,
    )
    assert rep.functional_transfer_verified is True
    assert rep.aligned_transfer_mse < 1e-10
    assert rep.safe_unaligned_copy_mse > rep.aligned_transfer_mse

    with pytest.raises(M4ContractError) as exc_info:
        execute_mapped_weight_transfer(
            teacher_w1=t_w1,
            teacher_w2=t_w2,
            student_w1=torch.zeros_like(t_w1),
            student_w2=torch.zeros_like(t_w2),
            alignment=align,
            eval_inputs=probe_x,
            eval_targets=target_y,
            target_scope="cross_family",
            is_permuted_synthetic_fixture=True,
        )
    assert exc_info.value.code == "SYNTHETIC_PERMUTATION_SCOPE_VIOLATION"


def test_lora_baseline_sft_baseline_distillation_baseline_and_rl_baseline() -> None:
    """T078 (`lora_baseline`), T079 (`sft_baseline`), T080 (`distillation_baseline`), T081 (`rl_baseline`)."""
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]], dtype=torch.float64)
    true_w = torch.tensor([[2.0, -1.0], [0.5, 1.5]], dtype=torch.float64)
    y = x @ true_w.T
    init_w = torch.zeros_like(true_w)

    # T078: LoRA baseline
    _, lora_rec = run_lora_baseline(init_w, x, y, authorized_in_contract=True, rank=2, steps=20)
    assert lora_rec.eval_loss_after < lora_rec.eval_loss_before
    assert lora_rec.reload_parity_verified is True
    with pytest.raises(M4ContractError):
        run_lora_baseline(init_w, x, y, authorized_in_contract=False)

    # T079: SFT baseline
    _, sft_rec = run_sft_baseline(init_w, x, y, authorized_in_contract=True, steps=20)
    assert sft_rec.eval_loss_after < sft_rec.eval_loss_before
    with pytest.raises(M4ContractError):
        run_sft_baseline(init_w, x, y, authorized_in_contract=True, target_site="unauthorized.layer")

    # T080: Distillation baseline
    _, dist_rec = run_distillation_baseline(
        init_w,
        true_w,
        x,
        authorized_in_contract=True,
        teacher_generation_flops=1024,
        steps=20,
    )
    assert dist_rec.total_compute_flops > sft_rec.total_compute_flops
    with pytest.raises(M4ContractError):
        run_distillation_baseline(
            init_w,
            true_w,
            x,
            authorized_in_contract=True,
            teacher_generation_flops=1024,
            claim_internal_circuit_transfer=True,
        )

    # T081: RL baseline
    _, rl_rec = run_rl_baseline(init_w, x, y, authorized_in_contract=True, episodes=16)
    assert rl_rec.extra_metadata["stochastic_training_separated_from_mean_eval"] is True
    with pytest.raises(M4ContractError):
        run_rl_baseline(init_w, x, y, authorized_in_contract=True, episodes=999, max_episodes_limit=64)


def test_memory_methods_and_continual_protocols() -> None:
    """T082 (`memory_methods`) and T083 (`continual_protocols`)."""
    mem = verify_external_memory_method(
        memory_id="kv.adapter.v1",
        capacity_slots=16,
        entries=[("rule_0", 1.0), ("rule_1", -1.0)],
    )
    assert mem.production_history == "hybrid"
    assert mem.static_export_self_contained is True

    with pytest.raises(M4ContractError):
        verify_external_memory_method(
            memory_id="kv.adapter.v1",
            capacity_slots=16,
            entries=[],
            claim_direct_only_weight_edit=True,
        )

    cont = evaluate_continual_protocol(
        episode_order=("ep1", "ep2", "ep3"),
        pre_matrix=[[0.5, 0.4, 0.3], [0.8, 0.55, 0.4], [0.8, 0.82, 0.6]],
        post_matrix=[[0.85, 0.45, 0.35], [0.83, 0.88, 0.42], [0.82, 0.86, 0.90]],
        state_growth_bytes=0,
    )
    assert cont.retention_score > 0.80
    assert cont.isolated_episodes is True

    with pytest.raises(M4ContractError):
        evaluate_continual_protocol(
            episode_order=("ep1", "ep2"),
            pre_matrix=[[0.5, 0.5], [0.5, 0.5]],
            post_matrix=[[0.8, 0.8], [0.8, 0.8]],
            state_growth_bytes=0,
            cross_episode_holdout_contaminated=True,
        )


def test_nemo_adapter_verifiers_adapter_gymnasium_adapter_private_gym_policy_measurement_research_and_community_gym() -> None:
    """T084 (`nemo_adapter`), T085 (`verifiers_adapter`), T086 (`gymnasium_adapter`), T087 (`private_gym_policy`), T088 (`measurement_research`), T089 (`community_gym`)."""
    # T084: NeMo-Gym
    nemo = run_nemo_adapter(prompts=["2+2", "3+5"], model_fn=lambda p: 1.0)
    assert nemo.replay_deterministic is True
    with pytest.raises(M4ContractError):
        run_nemo_adapter(prompts=["2+2"], model_fn=lambda p: 1.0, is_mock_only=True)

    # T085: Verifiers
    ver = run_verifiers_adapter(
        rollouts=[{"format_reward": 1.0, "correctness_binary": 0.0}, {"format_reward": 1.0, "correctness_binary": 1.0}]
    )
    assert ver.task_accuracy == pytest.approx(0.5)
    assert ver.mean_reward != ver.task_accuracy
    with pytest.raises(M4ContractError):
        run_verifiers_adapter(
            rollouts=[{"format_reward": 1.0, "correctness_binary": 0.0}],
            conflate_reward_as_accuracy=True,
        )

    # T086: Gymnasium
    gym_rec = run_gymnasium_adapter(action_bridge=lambda state, step: -0.25 * state)
    assert gym_rec.mean_reward > 0.0
    with pytest.raises(M4ContractError):
        run_gymnasium_adapter(action_bridge=None)

    # T087: Private gym policy
    priv = enforce_private_gym_policy(
        namespace="enterprise.finance.v1",
        encrypted_at_rest=True,
        key_id="kms-key-01",
        export_destination="internal_encrypted_cas",
    )
    assert priv.encrypted_at_rest is True
    with pytest.raises(M4ContractError):
        enforce_private_gym_policy(
            namespace="enterprise.finance.v1",
            encrypted_at_rest=True,
            key_id="kms-key-01",
            export_destination="public_unencrypted_bucket",
        )

    # T088: Measurement research
    m_rep = evaluate_measurement_research(
        {"gym_a": [1.0, 0.0, 1.0], "gym_b_dup": [1.0, 0.0, 1.0], "gym_c": [0.0, 1.0, 1.0]},
        shortcut_suspected_gyms=("gym_shortcut",),
    )
    assert m_rep.redundant_pairs == (("gym_a", "gym_b_dup"),)
    with pytest.raises(M4ContractError):
        evaluate_measurement_research({"g": [1.0]}, claim_progress_from_raw_task_count=True)

    # T089: Community gym
    key = b"publisher-secret-key"
    payload = b'{"gym":"community.math.v1"}'
    digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    sig = hmac.new(key, payload, hashlib.sha256).hexdigest()
    pkg = CommunityGymPackage(
        package_id="community.math.v1",
        payload_bytes=payload,
        expected_digest=digest,
        publisher_id="org.verified",
        signature_hex=sig,
        license_spdx="Apache-2.0",
        claimed_historical_score=0.99,
    )
    adm = verify_community_gym_package(pkg, trusted_publisher_key=key)
    assert adm.publisher_verified is True
    assert adm.historical_scores_trusted is False
    with pytest.raises(M4ContractError):
        verify_community_gym_package(pkg, trusted_publisher_key=b"wrong-key")


def test_library_reuse_strategy_retrieval_cross_model_protocol_proposer_dataset_proposer_training_learned_proposer_transfer_benchmark_research_source_records_and_m4_feature_acceptance() -> None:
    """T091–T099 (`library_reuse`, `strategy_retrieval`, `cross_model_protocol`, `proposer_dataset`, `proposer_training`, `learned_proposer`, `transfer_benchmark`, `research_source_records`, `m4_feature_acceptance`)."""
    # T091: Library reuse
    reuse = evaluate_library_recipe_reuse(
        recipe_id="carry.subspace_repair",
        version=2,
        version_status="verified",
        target_model_id="qwen2-1.5b",
        target_task_id="arith.carry",
        prerequisites_satisfied=True,
        source_digest_verified=True,
        fresh_target_evaluation_passed=True,
    )
    assert reuse.accepted_on_target is True
    with pytest.raises(M4ContractError):
        evaluate_library_recipe_reuse(
            recipe_id="carry.subspace_repair",
            version=1,
            version_status="disputed",
            target_model_id="qwen2-1.5b",
            target_task_id="arith.carry",
            prerequisites_satisfied=True,
            source_digest_verified=True,
            fresh_target_evaluation_passed=True,
        )

    # T092: Strategy retrieval
    recs = retrieve_repair_strategies(
        [
            {"strategy_id": "subspace_rank1", "architecture_family": "hf.qwen2.dense", "failure_feature": "carry", "outcome": "success", "cost_cpu_ms": 100},
            {"strategy_id": "subspace_rank1", "architecture_family": "hf.qwen2.dense", "failure_feature": "carry", "outcome": "failure", "cost_cpu_ms": 100},
        ],
        architecture_family="hf.qwen2.dense",
        failure_feature="carry",
    )
    assert len(recs) == 1
    assert recs[0].supporting_studies == 1
    assert recs[0].contradicting_studies == 1
    assert recs[0].is_accepted_edit is False

    # T093: Cross-model protocol
    cm = evaluate_cross_model_protocol(
        source_model_group="qwen2-0.5b",
        target_model_group="llama3-1b",
        shared_checkpoint_ancestry=False,
        tuned_strategy_on_target_holdout=False,
        no_history_evals=12,
        random_evals=15,
        history_informed_evals=4,
        source_discovery_cost_ms=500,
        target_adaptation_cost_ms=200,
    )
    assert cm.sample_efficiency_improved is True

    # T094, T095, T096: Proposer dataset, training, and learned proposer evaluation
    ds = build_proposer_dataset(
        [
            {"failure_signature": "sig_a", "split": "train", "access": "development"},
            {"failure_signature": "sig_a", "split": "train", "access": "development"},
            {"failure_signature": "sig_b", "split": "val", "access": "development"},
        ]
    )
    assert ds.train_count == 1
    assert ds.deduplicated_dropped == 1
    trained_prop = run_proposer_training_study(ds, training_study_id="study-prop-v1")
    eval_prop = evaluate_learned_proposer_inference(
        trained_prop,
        ir_valid_rate=0.98,
        learned_evals_to_success=3,
        no_history_evals_to_success=9,
        random_evals_to_success=14,
    )
    assert eval_prop.proposal_origin == "learned_proposer"

    # T097 & T098: Transfer benchmark & research source records
    tb = run_transfer_benchmark(scope_label="same_architecture", is_synthetic_permutation=True)
    assert tb.aligned_direct_mse < tb.safe_unaligned_mse
    src = ingest_research_source_record(
        source_id="openai.circuit_sparsity.2025",
        version="1.0",
        license_spdx="Apache-2.0",
        claim_summary="Sparse bridges align features",
    )
    assert src.support == "suggestive"

    # T099: Full M4 feature acceptance gate
    acceptance = run_m4_feature_acceptance_suite()
    assert acceptance.all_gates_passed is True
    assert acceptance.cg09_construction_passed is True
    assert acceptance.cg10_transfer_passed is True
    assert acceptance.cg11_methods_and_costs_passed is True
    assert acceptance.cg12_evidence_quality_passed is True
    assert acceptance.cg13_library_lifecycle_passed is True
