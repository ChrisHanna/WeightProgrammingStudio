"""Tests for Pythia (`GPTNeoXForCausalLM`) weight-space programming & live EleutherAI/pythia-14m editing."""

from __future__ import annotations

from pathlib import Path

import pytest
from neural_foundry.pythia import (
    PythiaModel,
    build_mini_pythia_model,
    run_pythia_gym_task_learning_study,
    run_pythia_weight_programming_study,
)


def test_mini_pythia_weight_programming_via_rust_surgeon_and_evals() -> None:
    """Offline deterministic test of Pythia weight-space programming through Rust compiler & surgeon."""
    model = build_mini_pythia_model()
    report = run_pythia_weight_programming_study(model, target_strength=80.0)
    assert report.reload_parity_verified is True
    assert report.before_parameter_digest != report.after_parameter_digest
    assert all(p.matched_target is True for p in report.dev_after)
    assert all(p.matched_target is True for p in report.retention_after)
    assert report.eval_comparison["dashboard"]["target_success_pct"] == 100.0
    assert report.eval_comparison["dashboard"]["skill_retention_pct"] == 100.0


@pytest.mark.qualification
def test_real_eleutherai_pythia_14m_weight_programming_if_cached() -> None:
    """Runs live Weight-Space Programming on EleutherAI/pythia-14m when present at /tmp/pythia-14m."""
    pythia_dir = Path("/tmp/pythia-14m")
    if not (pythia_dir / "model.safetensors").is_file():
        pytest.skip("checkpoint not present at /tmp/pythia-14m")
    model = PythiaModel.from_pretrained_dir(pythia_dir, model_name="EleutherAI/pythia-14m")
    report = run_pythia_weight_programming_study(model)

    assert report.total_model_parameters == 14_067_712
    assert report.changed_parameters == 65_536
    assert report.changed_parameter_pct_of_model < 0.5
    assert report.reload_parity_verified is True
    assert all(p.matched_target is False for p in report.dev_before)
    assert all(p.matched_target is True for p in report.dev_after)
    assert all(p.matched_target is True for p in report.generalization_after)
    assert all(p.matched_target is True for p in report.retention_after)
    # The study's own prompts pass, but the independent evaluator measures a harmful edit.
    assert report.promotion_verification["gym_self_check"]["status"] == "passed"
    assert report.promotion_verification["promote_eligible"] is False
    assert report.promotion_verification["status"] == "rejected"


def test_mini_pythia_gym_task_learning_branches() -> None:
    """Tests the 3-Skill Algorithmic Gym (Parity, Boolean Logic, Inequality) on mini-pythia."""
    mini = build_mini_pythia_model()
    mini_rep = run_pythia_gym_task_learning_study(mini, target_strength=85.0)
    assert len(mini_rep.branches) == 3
    assert mini_rep.selected_branch_id is None
    assert mini_rep.net_gain_over_noop is None
    assert mini_rep.net_gain_over_random is None
    assert any("validation 0.333 below 0.7" in reason for trial in mini_rep.branches for reason in trial.disqualified_because)
    assert mini_rep.reload_parity_verified is True
    assert mini_rep.branches[2].gym_dev_accuracy == 1.0
    assert mini_rep.branches[2].gym_prefix_val_accuracy == pytest.approx(1 / 3)
    assert mini_rep.branches[2].retention_accuracy == 1.0


@pytest.mark.qualification
def test_real_pythia_14m_gym_task_learning_branches_and_holdout_verification() -> None:
    """The same Gym on EleutherAI/pythia-14m. Skips without the checkpoint; NF_QUALIFICATION=1 makes that a failure."""
    pythia_dir = Path("/tmp/pythia-14m")
    if not (pythia_dir / "model.safetensors").is_file():
        pytest.skip("checkpoint not present at /tmp/pythia-14m")
    real_model = PythiaModel.from_pretrained_dir(pythia_dir, model_name="EleutherAI/pythia-14m")
    real_rep = run_pythia_gym_task_learning_study(real_model)

    # Branch 3 beats Branch 1 and Branch 2 on the prompts synthesis saw, but fails its selection-time retention guards.
    b1, b2, b3 = real_rep.branches
    assert b3.gym_dev_accuracy == 1.0
    assert b3.gym_prefix_val_accuracy == 1.0
    assert b3.retention_accuracy == 1.0
    assert b3.gym_prefix_val_accuracy > b2.gym_prefix_val_accuracy
    assert b3.gym_dev_accuracy > b1.gym_dev_accuracy
    assert b3.retention_guard_source == "selection_time_retention_guard"
    assert b3.retention_guard_accuracy < 0.98
    assert real_rep.selected_branch_id is None
    assert real_rep.selected_proposal is None
    assert real_rep.holdout_after == ()
    assert real_rep.promotion_verification["promote_eligible"] is False
    assert real_rep.promotion_verification["status"] == "not_evaluated"

