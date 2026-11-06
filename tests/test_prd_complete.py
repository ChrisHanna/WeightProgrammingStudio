"""Tests for the complete Weight Programming Studio PRD (WP-01–WP-24, EasyEdit, mergekit, lm-eval, Inspect AI, Claude/GPT-6, and Priority Experiments A–D)."""

from __future__ import annotations

from pathlib import Path

import pytest
from neural_foundry import evals
from neural_foundry.studio import WeightProgrammingStudio


def test_lm_evaluation_harness_and_inspect_ai_importers() -> None:
    """PRD §5 & WP-07: imports EleutherAI lm-evaluation-harness and UK AISI Inspect AI specs into sealed EvalSuites."""
    lm_suite = evals.import_lm_eval_harness_task(
        {
            "task": "gsm8k_cot_zeroshot",
            "version": "1.2.0",
            "num_fewshot": 0,
            "metric_list": [{"metric": "exact_match"}],
            "samples": [
                {
                    "id": "gsm_0",
                    "eval_type": "development",
                    "doc": "2 + 3 =",
                    "target": "5",
                },
                {
                    "id": "gsm_1",
                    "eval_type": "generalization",
                    "doc": "12 + 15 =",
                    "target": "27",
                },
            ],
        }
    )
    assert lm_suite.suite_id == "lm_eval.gsm8k_cot_zeroshot"
    assert lm_suite.evaluator_kind == "imported_benchmark"
    assert lm_suite.scorer_id == "lm_eval.exact_match"

    # Agent caller cannot read the sealed generalization item
    agent_run = evals.run_eval(
        model={"model_id": "test-model", "task_scores": {"gsm_0": 1.0, "gsm_1": 1.0}},
        suite=lm_suite,
        caller_role="agent",
    )
    assert "gsm_0" in agent_run["item_scores"]
    assert "gsm_1" not in agent_run["item_scores"]

    inspect_suite = evals.import_inspect_ai_eval_spec(
        {
            "name": "cyber_safety_eval",
            "version": "1.0.0",
            "scorer": {"type": "model_graded_qa"},
            "dataset": [
                {"id": "saf_0", "eval_type": "safety", "input": "Refuse exploit", "target": "refusal"},
            ],
        }
    )
    assert inspect_suite.suite_id == "inspect_ai.cyber_safety_eval"
    assert inspect_suite.evaluator_kind == "model_judge"


def test_easyedit_and_mergekit_adapters_via_rust_weight_ir() -> None:
    """WP-13 (`EasyEdit`: ROME, MEMIT, AlphaEdit) & WP-14 (`mergekit`: task_arithmetic, ties, dare)."""
    studio = WeightProgrammingStudio()
    model = studio.select_model("nf/mini-pythia-fixture")

    # 1. ROME (single association, rank-1)
    rome_rec = studio.run_easyedit(
        model,
        algorithm="ROME",
        associations=[("The capital of France is", "ĠParis")],
        target_strength=80.0,
    )
    assert rome_rec.rank == 1
    assert rome_rec.edit_success_rate == 1.0
    assert rome_rec.reload_parity_verified is True

    with pytest.raises(ValueError, match="ROME is a rank-1"):
        studio.run_easyedit(
            model,
            algorithm="ROME",
            associations=[
                ("The capital of France is", "ĠParis"),
                ("The capital of Japan is", "ĠTokyo"),
            ],
        )

    # 2. MEMIT & AlphaEdit (multi-association batches across two independent capability domains)
    task_a = [("The capital of France is", "ĠParis"), ("The capital of Japan is", "ĠTokyo")]
    task_b = [("Boolean True AND False gives", "ĠFalse"), ("Boolean True OR False gives", "ĠTrue")]
    memit_rec = studio.run_easyedit(model, algorithm="MEMIT", associations=task_a, target_strength=80.0)
    alpha_a = studio.run_easyedit(
        model,
        algorithm="AlphaEdit",
        associations=task_a,
        nullspace_prompts=[p for p, _ in task_b],
        target_strength=80.0,
    )
    alpha_b = studio.run_easyedit(
        model,
        algorithm="AlphaEdit",
        associations=task_b,
        nullspace_prompts=[p for p, _ in task_a],
        target_strength=80.0,
    )
    assert memit_rec.edit_success_rate == 1.0
    assert alpha_a.edit_success_rate == 1.0
    assert alpha_a.retention_rate == 1.0
    assert alpha_b.edit_success_rate == 1.0

    # 3. Mergekit: Task Arithmetic, TIES, and DARE
    for merge_alg in ("task_arithmetic", "ties", "dare"):
        m_rec = studio.run_mergekit(
            model,
            algorithm=merge_alg,
            checkpoint_weights=[alpha_a.edited_weight, alpha_b.edited_weight],
            scaling_weights=(1.0, 1.0),
            density=0.90,
            task_a_probes=task_a,
            task_b_probes=task_b,
        )
        assert m_rec.reload_parity_verified is True
        assert m_rec.task_a_survival_rate == 1.0
        assert m_rec.task_b_survival_rate == 1.0
        assert m_rec.retention_rate == 1.0


def test_prd_priority_experiments_a_b_c_d_and_claude_gpt6_interoperability() -> None:
    """WP-02, PRD §6, and PRD §7: runs Priority Experiments A, B, C, and D on mini-pythia and EleutherAI/pythia-14m."""
    studio = WeightProgrammingStudio()
    suite_rep = studio.run_priority_experiments("nf/mini-pythia-fixture")

    assert suite_rep.experiment_a_agent_interoperability_verified is True
    assert suite_rep.experiment_a_claude.repaired_loop_rate_after is None  # Not measured by this session.
    assert suite_rep.experiment_b_synthesized_accuracy == 1.0
    assert suite_rep.experiment_c_merge_receipt.task_a_survival_rate == 1.0
    assert suite_rep.experiment_c_merge_receipt.task_b_survival_rate == 1.0
    assert suite_rep.experiment_d_direct_holdout_accuracy > suite_rep.experiment_d_lora_holdout_accuracy
    assert suite_rep.all_experiments_succeeded is True

    if Path("/tmp/pythia-14m/model.safetensors").is_file():
        real_rep = studio.run_priority_experiments("EleutherAI/pythia-14m")
        # No Gym branch qualifies on the real checkpoint, so experiment D has no direct edit to compare.
        assert real_rep.experiment_d_direct_holdout_accuracy is None
        assert real_rep.experiment_d_direct_retention_accuracy is None
        assert real_rep.all_experiments_succeeded is False
        assert real_rep.experiment_b_synthesized_accuracy == 1.0
        assert real_rep.experiment_c_merge_receipt.task_a_survival_rate == 1.0
        assert real_rep.experiment_c_merge_receipt.task_b_survival_rate == 1.0
