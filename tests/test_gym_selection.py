"""The Gym selects from recorded development, validation and public retention scores, or selects nothing."""

from __future__ import annotations

import ast
import inspect
from dataclasses import replace
from pathlib import Path

import pytest

from neural_foundry import flagship, independent_eval as ie, prd_adapters, promotion_gate, pythia
from neural_foundry.edit_plan import apply_edit_graph
from neural_foundry.gpt_neox_adapter import compile_to_edit_graph

PYTHIA_14M = Path("/tmp/pythia-14m")
GYM = pythia.ALGORITHMIC_THREE_SKILL_GYM


def _trial(name: str, dev: float = 1.0, val: float = 1.0, retention: float | None = 1.0, *, rank: int = 6, parity: bool = True) -> pythia.GymBranchTrial:
    return pythia.GymBranchTrial(name, "", 1, rank, 0, dev, val, 1.0, False, selection_retention_accuracy=retention, reload_parity_verified=parity)


def test_the_best_qualifying_scores_win_whatever_the_position() -> None:
    trials, chosen = pythia.select_gym_branch((_trial("a", val=1.0), _trial("b", val=0.8), _trial("c", val=0.9)))
    assert chosen.branch_id == "a"
    assert [t.selected_by_gym_scientist for t in trials] == [True, False, False]
    _, chosen = pythia.select_gym_branch((_trial("a", val=0.8), _trial("b", val=0.9)))
    assert chosen.branch_id == "b"


def test_a_tie_goes_to_the_smaller_rank() -> None:
    _, chosen = pythia.select_gym_branch((_trial("a", rank=6), _trial("b", rank=2)))
    assert chosen.branch_id == "b"


def test_on_equal_scores_more_synthesis_guards_win() -> None:
    _, chosen = pythia.select_gym_branch((replace(_trial("guarded"), synthesis_guards=2), replace(_trial("plain", rank=2), synthesis_guards=1)))
    assert chosen.branch_id == "guarded"


@pytest.mark.parametrize(
    ("trial", "reason"),
    [
        (_trial("x", dev=0.5), "development"),
        (_trial("x", val=0.5), "validation"),
        (_trial("x", retention=0.9), "retention"),
        (_trial("x", retention=None), "retention"),
        (_trial("x", parity=False), "reload parity"),
    ],
)
def test_no_qualifying_branch_selects_none_and_says_why(trial: pythia.GymBranchTrial, reason: str) -> None:
    trials, chosen = pythia.select_gym_branch((trial,))
    assert chosen is None
    assert not trials[0].selected_by_gym_scientist
    assert reason in " ".join(trials[0].disqualified_because)


def test_selection_has_no_holdout_input() -> None:
    assert list(inspect.signature(pythia.select_gym_branch).parameters) == ["trials"]
    names = {n.attr for n in ast.walk(ast.parse(inspect.getsource(pythia.select_gym_branch))) if isinstance(n, ast.Attribute)}
    assert not {name for name in names if "holdout" in name}
    assert not {f for f in pythia.GymBranchTrial.__dataclass_fields__ if "holdout" in f}


def test_selection_is_unchanged_by_the_holdout() -> None:
    other = replace(GYM, holdout_rules=(("Number 3 has parity", "Ġodd"),))
    one = pythia.run_pythia_gym_task_learning_study(pythia.build_mini_pythia_model(), gym=GYM, target_strength=85.0)
    two = pythia.run_pythia_gym_task_learning_study(pythia.build_mini_pythia_model(), gym=other, target_strength=85.0)
    assert one.selected_branch_id == two.selected_branch_id
    assert one.branches == two.branches


def test_every_branch_records_its_scores() -> None:
    report = pythia.run_pythia_gym_task_learning_study(pythia.build_mini_pythia_model(), target_strength=85.0)
    assert len(report.branches) == 3
    for trial in report.branches:
        assert trial.selection_retention_accuracy is not None
        assert trial.selection_retention_source == "synthesis_guard_prompts"
        assert isinstance(trial.reload_parity_verified, bool)


def test_unseen_prompts_stay_out_of_synthesis_and_the_evaluator_corpus() -> None:
    unseen = set(GYM.selection_retention_prompts)
    assert unseen.isdisjoint(GYM.retention_prompts)
    corpus = ie.load_corpus()
    assert unseen.isdisjoint(prompt for pairs in corpus["tasks"].values() for prompt, _ in pairs)
    assert not any(prompt in sentence for prompt in unseen for sentence in corpus["text"])


@pytest.fixture
def no_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pythia, "select_gym_branch", lambda trials: (tuple(replace(t, disqualified_because=("forced",)) for t in trials), None))


def test_no_selection_leaves_the_holdout_sealed_and_says_so(no_selection: None) -> None:
    report = pythia.run_pythia_gym_task_learning_study(pythia.build_mini_pythia_model(), target_strength=85.0)
    assert report.selected_branch_id is None and report.selected_proposal is None
    assert report.edited_weight is None and report.edit_steps == ()
    assert report.holdout_before == () and report.holdout_after == ()
    assert report.noop_control_holdout_accuracy is None and report.lora_holdout_accuracy is None
    assert report.exposure_ledger.verifier_holdout_evaluation_digest is None
    assert report.promotion_verification["status"] == "not_evaluated"
    assert "forced" in report.promotion_verification["markdown_dashboard"]
    assert [c.selected for c in report.gym_proposals] == [False, False, False]


def test_agent_handles_no_selection(no_selection: None) -> None:
    report = prd_adapters.FrontierAgentProgrammer("claude-4-5-sonnet").run_reasoning_loop_repair_experiment(pythia.build_mini_pythia_model())
    assert report.submitted_edit_steps == ()
    assert report.submitted_weights_by_layer is None
    assert report.held_out_accuracy_after is None


def test_feedback_comparison_handles_no_selection(no_selection: None) -> None:
    report = prd_adapters.compare_weight_reading_only_vs_feedback_enabled_agent(pythia.build_mini_pythia_model())
    assert report.feedback_enabled_holdout_accuracy is None
    assert report.feedback_gain_over_weight_reading_only is None
    assert report.feedback_tools_proven_necessary is False


def test_flagship_handles_no_selection(no_selection: None, tmp_path: Path) -> None:
    report = flagship.run_flagship_loop_repair_study_and_publish_bundle(pythia.build_mini_pythia_model(), output_dir=tmp_path)
    assert report.noop_control_holdout_accuracy is None
    assert report.random_control_holdout_accuracy is None


@pytest.mark.qualification
def test_real_gym_selects_nothing_and_the_harmful_branch_is_still_rejected() -> None:
    if not (PYTHIA_14M / "model.safetensors").is_file():
        pytest.skip("checkpoint not present at /tmp/pythia-14m")
    model = pythia.PythiaModel.from_pretrained_dir(PYTHIA_14M, model_name="EleutherAI/pythia-14m")
    report = pythia.run_pythia_gym_task_learning_study(model)
    assert report.selected_branch_id is None
    # Evaluate the frozen branch that used to be hard-coded anyway: the gate must reject it by itself.
    frozen = next(c for c in report.gym_proposals if c.branch_id.startswith("branch_3"))
    replayed = apply_edit_graph(model, compile_to_edit_graph(model, frozen.proposal))
    candidate = ie.freeze_candidate(
        model.tensors,
        replayed.edited_tensors,
        candidate_id="branch_3",
        target_cases=[(f"{p} {e}", t.replace("Ġ", " ")) for p in GYM.gym_dev_prefixes for e, t in GYM.base_rules],
    )
    decision = promotion_gate.decide_promotion(ie.evaluate_candidate(model, candidate))
    assert decision["status"] == "rejected"
    assert "free_text_agreement" in decision["failed_checks"]
