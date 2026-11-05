"""Tests for T048–T051: matched transformer diagnosis, gradient attribution, causal probes, and hypotheses."""

from __future__ import annotations

import pytest
from neural_foundry.diagnostics import (
    DiagnosticsError,
    HypothesisRecord,
    MatchedExecutionPair,
    SeededCausalTransformerFixture,
    compute_gradient_attribution,
    diagnose_matched_executions,
    run_causal_probe,
)


def test_matched_transformer_diagnosis_confirms_site_and_task_identities() -> None:
    fixture = SeededCausalTransformerFixture(seed=17, inject_defect=True)
    pair = MatchedExecutionPair(
        pair_id="pair-routing-01",
        task_family="token_identity_routing",
        success_task_id="task-token-1",
        failed_task_id="task-token-2",
        success_input=(0, 1),
        failed_input=(0, 1, 2),
        expected_success_target=1,
        expected_failed_target=2,
        matching_variables={
            "task_family": "token_identity_routing",
            "length": 2,
            "difficulty": "single_hop",
            "answer_format": "token_id",
        },
        imperfect_matching_disclosures=("prefix_context_contains_extra_token",),
    )
    report = diagnose_matched_executions(fixture, pair)
    assert report["task_identities"] == {
        "task_family": "token_identity_routing",
        "success_task_id": "task-token-1",
        "failed_task_id": "task-token-2",
    }
    assert {
        "embed_tokens",
        "layers.0.residual_pre",
        "layers.0.self_attn",
        "layers.0.mlp",
        "layers.0.residual_post",
        "lm_head",
    }.issubset(set(report["site_identities"]))
    assert report["success_prediction"] == 1
    assert report["failed_prediction"] != 2
    assert "sequence_length_differs_between_success_and_failed_inputs" in report["imperfect_matching_disclosures"]
    assert all(
        item["evidence_label"]["support"] == "suggestive"
        for item in report["ranked_sites"]
    )
    assert report["ground_truth_site"] == "layers.0.mlp"


def test_gradient_attribution_policy_denies_strict_and_classifies_saliency_vs_fitting() -> None:
    fixture = SeededCausalTransformerFixture(seed=17, inject_defect=True)

    # 1. Strict fully gradient-free profile denies backward attribution
    with pytest.raises(DiagnosticsError) as strict_err:
        compute_gradient_attribution(
            fixture,
            input_tokens=(0, 2),
            target_token=2,
            site_names=("layers.0.mlp",),
            permissions={
                "allow_gradient_inspection": False,
                "allow_parameter_training": False,
                "require_fully_gradient_free": True,
            },
        )
    assert strict_err.value.code == "PERMISSION_DENIED"

    # 2. Permitted non-writing saliency records unchanged state and direct-only lineage with qualifier
    saliency = compute_gradient_attribution(
        fixture,
        input_tokens=(0, 2),
        target_token=2,
        site_names=("layers.0.self_attn", "layers.0.mlp"),
        permissions={
            "allow_gradient_inspection": True,
            "allow_parameter_training": False,
            "require_fully_gradient_free": False,
        },
    )
    assert saliency["state_unchanged"] is True
    assert saliency["grad_buffers_cleared"] is True
    assert saliency["backward_count"] == 1
    assert saliency["classification"] == "gradient_assisted_inspection"
    assert saliency["report_label"] == "direct_parameter_edit_with_gradient_assisted_inspection"
    assert saliency["method_record"]["production_history"] == "direct_only"
    assert saliency["method_record"]["inspection_uses_backprop"] is True
    assert saliency["method_record"]["gradient_derived_update_values"] is False
    assert saliency["evidence_label"]["support"] == "suggestive"

    # 3. Gradient-derived update values require training permission and classify as hybrid
    with pytest.raises(DiagnosticsError) as deny_fit:
        compute_gradient_attribution(
            fixture,
            input_tokens=(0, 2),
            target_token=2,
            site_names=("layers.0.mlp",),
            permissions={
                "allow_gradient_inspection": True,
                "allow_parameter_training": False,
                "require_fully_gradient_free": False,
            },
            derive_update_values_from_gradient=True,
        )
    assert deny_fit.value.code == "PERMISSION_DENIED"

    hybrid = compute_gradient_attribution(
        fixture,
        input_tokens=(0, 2),
        target_token=2,
        site_names=("layers.0.mlp",),
        permissions={
            "allow_gradient_inspection": True,
            "allow_parameter_training": True,
            "require_fully_gradient_free": False,
        },
        derive_update_values_from_gradient=True,
    )
    assert hybrid["classification"] == "hybrid_or_trained"
    assert hybrid["method_record"]["production_history"] == "hybrid"

    # 4. State drift during inspection invalidates the receipt
    with pytest.raises(DiagnosticsError) as drift_err:
        compute_gradient_attribution(
            fixture,
            input_tokens=(0, 2),
            target_token=2,
            site_names=("layers.0.mlp",),
            permissions={
                "allow_gradient_inspection": True,
                "allow_parameter_training": False,
                "require_fully_gradient_free": False,
            },
            inject_state_drift=True,
        )
    assert drift_err.value.code == "STATE_DRIFT"


def test_causal_probes_patching_ablation_counterfactual_and_negative_faults() -> None:
    fixture = SeededCausalTransformerFixture(seed=17, inject_defect=True)

    # 1. Counterfactual probe on the defective MLP site repairs token-2 routing and restores cleanly
    cf_receipt = run_causal_probe(
        fixture,
        probe_kind="counterfactual",
        site_name="layers.0.mlp",
        recipient_input=(0, 2),
        unrelated_control_input=(0, 1),
        expected_repaired_target=2,
    )
    assert cf_receipt["causal_effect_supported"] is True
    assert cf_receipt["intervened_prediction"] == 2
    assert cf_receipt["restored_prediction"] == cf_receipt["baseline_prediction"]
    assert cf_receipt["unrelated_control_before"] == cf_receipt["unrelated_control_after"] == 1
    assert cf_receipt["restoration_verified"] is True
    assert cf_receipt["cleanup_confirmed"] is True
    assert cf_receipt["source_model_unchanged"] is True
    assert cf_receipt["evidence_label"]["support"] == "intervention_supported"

    # 2. Activation patching from donor run into recipient
    patch_receipt = run_causal_probe(
        fixture,
        probe_kind="activation_patch",
        site_name="layers.0.mlp",
        recipient_input=(0, 2),
        donor_input=(0, 6),
        unrelated_control_input=(0, 1),
        expected_repaired_target=2,
    )
    assert patch_receipt["causal_effect_supported"] is True
    assert patch_receipt["restoration_verified"] is True

    # 3. Ablation probe (zero/mean)
    ablation_receipt = run_causal_probe(
        fixture,
        probe_kind="ablation",
        site_name="layers.0.self_attn",
        recipient_input=(0, 2),
        ablation_mode="zero",
    )
    assert ablation_receipt["restoration_verified"] is True
    assert ablation_receipt["cleanup_confirmed"] is True

    # 4. Buffer drift during probe is rejected
    with pytest.raises(DiagnosticsError) as buf_err:
        run_causal_probe(
            fixture,
            probe_kind="ablation",
            site_name="layers.0.mlp",
            recipient_input=(0, 2),
            inject_buffer_drift=True,
        )
    assert buf_err.value.code == "BUFFER_OR_PARAMETER_DRIFT"

    # 5. Failed hook cleanup is detected and rejected
    with pytest.raises(DiagnosticsError) as clean_err:
        run_causal_probe(
            fixture,
            probe_kind="ablation",
            site_name="layers.0.mlp",
            recipient_input=(0, 2),
            inject_failed_cleanup=True,
        )
    assert clean_err.value.code == "CLEANUP_FAILED"


def test_hypothesis_records_preserve_contradictions_and_reject_missing_evidence_links() -> None:
    valid = HypothesisRecord(
        hypothesis_id="hyp-mlp-channel-2",
        model_id="nf.transformer.small_causal.v1",
        observed_failure="Token 2 misrouted due to negative MLP diagonal weight",
        matched_success_examples=("task-token-1", "task-token-3"),
        candidate_sites=("layers.0.mlp",),
        proposed_mechanism="Inverted sign on layers.0.mlp channel 2 suppresses token 2 logit",
        competing_explanations=("Attention head 0 misattends to prefix token",),
        predicted_outcomes={
            "targeted_edit": "Restores token 2 prediction to class 2",
            "ablation": "Zeroing MLP degrades routing confidence",
            "counterfactual": "Flipping channel 2 sign restores prediction",
            "negative_control": "Unrelated token 1 remains unchanged",
        },
        falsifiers=("Editing layers.0.mlp[2,2] fails to improve token 2 accuracy",),
        controls=("no_op_control", "unrelated_token_1_control"),
        intended_gym_effects={"target": "+0.50 accuracy", "retention": "0.00 regression"},
        uncertainty=0.15,
        origin="measured_result",
        support="suggestive",
        supporting_evidence=("sha256:" + "a" * 64,),
        contradicting_evidence=(),
        capability_verification_protocol="proto.capability.v1",
        mechanism_verification_protocol="proto.mechanism.v1",
    )
    serialized = valid.validate_and_serialize()
    assert serialized["support"] == "suggestive"
    assert serialized["digest"].startswith("sha256:")

    # Contradicting evidence downgrades support to "contradicted" and is preserved
    contradicted = HypothesisRecord(
        **{
            **valid.__dict__,
            "support": "intervention_supported",
            "contradicting_evidence": ("sha256:" + "c" * 64,),
        }
    ).validate_and_serialize()
    assert contradicted["support"] == "contradicted"
    assert contradicted["contradicting_evidence"] == ["sha256:" + "c" * 64]

    # Suggestive model claim cannot auto-promote to causal support
    with pytest.raises(DiagnosticsError) as promo_err:
        HypothesisRecord(
            **{
                **valid.__dict__,
                "origin": "model_claim",
                "support": "intervention_supported",
            }
        ).validate_and_serialize()
    assert promo_err.value.code == "CAUSAL_PROMOTION_DENIED"

    # Missing evidence links on measured_result are rejected
    with pytest.raises(DiagnosticsError) as link_err:
        HypothesisRecord(
            **{
                **valid.__dict__,
                "supporting_evidence": (),
            }
        ).validate_and_serialize()
    assert link_err.value.code == "MISSING_EVIDENCE_LINKS"
