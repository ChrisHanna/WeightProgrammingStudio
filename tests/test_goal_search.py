"""Goal-search contracts before any model-specific controller is involved."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
from neural_foundry.goal_search import (
    STRICT_GENERAL_PROFILE,
    TASK_SPECIALIZED_PROFILE,
    CandidateArchive,
    CandidateRecord,
    GoalSearchError,
)


def record(
    candidate_id: str,
    *,
    arm: str = "direct",
    method: str = "closed_form",
    parent: str | None = "base",
    target: float = 0.95,
    target_gain: float = 0.70,
    retention: float = 0.0,
    kl: float = 5.0,
    changed: int = 100,
    replay: bool = True,
    parity: bool = True,
    search_state: str = "accepted",
) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=candidate_id,
        parent_candidate_id=parent,
        round_number=1,
        arm=arm,  # type: ignore[arg-type]
        method=method,
        proposal_digest=f"proposal:{candidate_id}",
        state_digest=f"state:{candidate_id}",
        scores={
            "target_accuracy": target,
            "target_gain": target_gain,
            "retention_accuracy": retention,
            "kl_nats": kl,
            "perplexity_ratio": 100.0 if kl else 1.0,
            "safety_pass_rate": 1.0,
            "changed_parameters": changed,
            "wall_seconds": 1.0,
            "artifact_bytes": 1_000,
        },
        proposal={"operation": "low_rank_update", "parameters": {"rank": 6}},
        search_state=search_state,  # type: ignore[arg-type]
        replay_verified=replay,
        reload_parity_verified=parity,
    )


def controls(candidate_id: str = "candidate") -> tuple[CandidateRecord, CandidateRecord]:
    return (
        record("random", arm="control", method="random", parent=candidate_id, target=0.20, target_gain=0.0, retention=0.20, kl=1.0),
        record("placebo", arm="control", method="placebo", parent=candidate_id, target=0.25, target_gain=0.0, retention=0.25, kl=1.0),
    )


def test_profile_is_frozen_and_content_addressed() -> None:
    same = type(TASK_SPECIALIZED_PROFILE)(**TASK_SPECIALIZED_PROFILE.__dict__)
    assert same.digest == TASK_SPECIALIZED_PROFILE.digest
    with pytest.raises(FrozenInstanceError):
        TASK_SPECIALIZED_PROFILE.profile_id = "general"  # type: ignore[misc]


def test_specialized_profile_still_requires_control_advantage_and_replay() -> None:
    archive = CandidateArchive("cycle-1", TASK_SPECIALIZED_PROFILE)
    archive.add(record("base", arm="baseline", method="no_op", parent=None, target=0.10, target_gain=0.0, retention=1.0, kl=0.0))
    archive.add(record("candidate"))
    random, placebo = controls()
    archive.add(random)
    archive.add(placebo)
    assert archive.decision("candidate").eligible is True

    other = CandidateArchive("cycle-2", TASK_SPECIALIZED_PROFILE)
    other.add(record("base", arm="baseline", method="no_op", parent=None, target=0.10, target_gain=0.0, retention=1.0, kl=0.0))
    other.add(record("candidate", target=0.97, replay=False))
    other.add(record("random", arm="control", method="random", parent="candidate", target=0.99, retention=0.9, kl=0.1))
    other.add(record("placebo", arm="control", method="placebo", parent="candidate", target=0.98, retention=0.9, kl=0.1))
    decision = other.decision("candidate")
    assert decision.eligible is False
    assert {"replay_verified", "target_above_random", "target_above_placebo"} <= set(decision.rejection_reasons)


def test_general_profile_rejects_retention_loss_that_specialized_records() -> None:
    candidate = record("candidate")
    random, placebo = controls(candidate.candidate_id)
    specialized = CandidateArchive("specialized", TASK_SPECIALIZED_PROFILE)
    general = CandidateArchive("general", STRICT_GENERAL_PROFILE)
    for archive in (specialized, general):
        archive.add(record("base", arm="baseline", method="no_op", parent=None, target=0.10, target_gain=0.0, retention=1.0, kl=0.0))
        archive.add(candidate)
        archive.add(random)
        archive.add(placebo)
    assert specialized.decision(candidate.candidate_id).eligible is True
    assert general.decision(candidate.candidate_id).eligible is False


def test_archive_preserves_rejected_candidates_and_baseline_dominance() -> None:
    archive = CandidateArchive("cycle", TASK_SPECIALIZED_PROFILE)
    base = record("base", arm="baseline", method="no_op", parent=None, target=0.10, target_gain=0.0, retention=1.0, kl=0.0)
    archive.add(base)
    archive.add(record("direct", target=0.90, retention=0.10, kl=2.0, changed=200))
    random, placebo = controls("direct")
    archive.add(random)
    archive.add(placebo)
    archive.add(
        record(
            "lora",
            arm="baseline",
            method="lora",
            target=0.95,
            target_gain=0.85,
            retention=0.50,
            kl=0.5,
            changed=100,
        )
    )
    snapshot = archive.freeze()
    assert snapshot.baseline_dominated is True
    assert {entry["candidate_id"] for entry in snapshot.to_document()["records"]} >= {
        "base",
        "random",
        "placebo",
        "direct",
        "lora",
    }
    assert archive.terminal_state() == "baseline_dominated"


def test_behaviorally_better_baseline_dominates_despite_higher_cost() -> None:
    from dataclasses import replace

    from neural_foundry.goal_search import _dominates

    baseline = record(
        "lora",
        arm="baseline",
        method="lora",
        target=0.90,
        target_gain=0.80,
        retention=0.95,
        kl=0.10,
        changed=900,
    )
    direct = record(
        "direct",
        target=0.80,
        target_gain=0.70,
        retention=0.85,
        kl=0.20,
        changed=10,
    )
    baseline = replace(
        baseline,
        scores={**baseline.scores, "wall_seconds": 20.0, "artifact_bytes": 50_000},
    )
    direct = replace(
        direct,
        scores={**direct.scores, "wall_seconds": 1.0, "artifact_bytes": 100},
    )

    assert _dominates(baseline, direct) is True


def test_frontier_freezes_once_and_rejects_new_candidates() -> None:
    archive = CandidateArchive("cycle", TASK_SPECIALIZED_PROFILE)
    archive.add(record("candidate", parent=None))
    first = archive.freeze()
    assert first.snapshot_digest.startswith("sha256:")
    with pytest.raises(GoalSearchError, match="already frozen"):
        archive.freeze()
    with pytest.raises(GoalSearchError, match="frozen"):
        archive.add(record("second", parent=None))


def test_missing_control_is_incomplete_evidence_not_an_implicit_pass() -> None:
    archive = CandidateArchive("cycle", TASK_SPECIALIZED_PROFILE)
    archive.add(record("candidate", parent=None))
    decision = archive.decision("candidate")
    assert decision.eligible is False
    assert "complete_evidence" in decision.rejection_reasons


def test_archive_rejects_unknown_parent_and_duplicate_ids() -> None:
    archive = CandidateArchive("cycle", TASK_SPECIALIZED_PROFILE)
    with pytest.raises(GoalSearchError, match="Unknown parent"):
        archive.add(record("orphan", parent="missing"))
    archive.add(record("one", parent=None))
    with pytest.raises(GoalSearchError, match="Duplicate candidate_id"):
        archive.add(record("one", parent=None))


def test_documents_are_deep_frozen() -> None:
    candidate = record("candidate", parent=None)
    assert isinstance(candidate.scores, type(candidate.scores))
    with pytest.raises(TypeError):
        candidate.scores["target_accuracy"] = 0.5  # type: ignore[index]
    proposal: Any = candidate.proposal["parameters"]
    with pytest.raises(TypeError):
        proposal["rank"] = 1


def observation(**changes: Any) -> Any:
    from neural_foundry.goal_search import SearchObservation

    values: dict[str, Any] = {
        "round_number": 2,
        "candidate_evaluations": 200,
        "changed_parameters": 1000,
        "elapsed_seconds": 10.0,
        "target_accuracy": 0.50,
        "retention_accuracy": 0.50,
        "kl_nats": 0.20,
        "monitor_retention_accuracy": 1.0,
        "monitor_kl_nats": 0.0,
        "initial_monitor_retention_accuracy": 1.0,
        "initial_monitor_kl_nats": 0.0,
        "best_gain": 0.01,
        "no_gain_rounds": 0,
    }
    values.update(changes)
    return SearchObservation(**values)


def test_rule_diagnosis_prefers_task_moves_for_target_gap() -> None:
    from neural_foundry.goal_search import SearchBudgets, diagnose_next_move

    decision = diagnose_next_move(STRICT_GENERAL_PROFILE, SearchBudgets(), observation(target_accuracy=0.50))
    assert decision.rule_id == "target_gap"
    assert decision.move_policy == "task"
    assert decision.terminal_state is None


def test_rule_diagnosis_chooses_retention_repair_only_after_target_bar() -> None:
    from neural_foundry.goal_search import SearchBudgets, diagnose_next_move

    decision = diagnose_next_move(
        STRICT_GENERAL_PROFILE,
        SearchBudgets(),
        observation(target_accuracy=0.95, retention_accuracy=0.50, kl_nats=1.0),
    )
    assert decision.rule_id == "retention_or_kl_gap"
    assert decision.move_policy == "retention_repair"


def test_monitor_drift_stops_before_more_optimization() -> None:
    from neural_foundry.goal_search import SearchBudgets, diagnose_next_move

    decision = diagnose_next_move(
        STRICT_GENERAL_PROFILE,
        SearchBudgets(max_monitor_retention_drop=0.10),
        observation(target_accuracy=0.10, monitor_retention_accuracy=0.80),
    )
    assert decision.rule_id == "monitor_drift"
    assert decision.move_policy == "none"
    assert decision.terminal_state == "monitor_drift"


def test_search_budgets_and_plateau_have_explicit_terminal_states() -> None:
    from neural_foundry.goal_search import SearchBudgets, diagnose_next_move

    exhausted = diagnose_next_move(
        STRICT_GENERAL_PROFILE,
        SearchBudgets(max_rounds=2),
        observation(round_number=2),
    )
    stable = diagnose_next_move(
        STRICT_GENERAL_PROFILE,
        SearchBudgets(max_no_gain_rounds=2),
        observation(no_gain_rounds=2),
    )
    assert exhausted.terminal_state == "budget_exhausted"
    assert stable.terminal_state == "no_candidate_cleared_min_gain"


def test_zero_changed_parameter_budget_is_not_treated_as_unlimited() -> None:
    from neural_foundry.goal_search import SearchBudgets, diagnose_next_move

    decision = diagnose_next_move(
        STRICT_GENERAL_PROFILE,
        SearchBudgets(max_changed_parameters=0),
        observation(changed_parameters=0),
    )
    assert decision.rule_id == "budget_changed_parameters"
    assert decision.terminal_state == "budget_exhausted"


def test_goal_search_uses_registered_custom_gym_and_evaluates_its_holdout(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    from neural_foundry import frontier_evaluation
    from neural_foundry.cumulative_edit import run_goal_search
    from neural_foundry.goal_search import SearchBudgets
    from neural_foundry.pythia import ALGORITHMIC_THREE_SKILL_GYM, build_mini_pythia_model

    gym = replace(
        ALGORITHMIC_THREE_SKILL_GYM,
        gym_id="custom_single_rule.v1",
        skills_taught=("capital",),
        base_rules=(("The capital of France is", "ĠParis"),),
        holdout_rules=(("The capital of Japan is", "ĠTokyo"),),
        gym_dev_prefixes=("Task:",),
        gym_val_prefixes=("Note:",),
        holdout_prefixes=("Today:",),
    )
    evaluated_cases: list[tuple[str, str]] = []

    def fake_evaluation(model: Any, snapshot: Any, **kwargs: Any) -> dict[str, Any]:
        evaluated_cases.extend(kwargs["target_cases"])
        return {"snapshot_digest": snapshot.snapshot_digest, "evaluated": []}

    monkeypatch.setattr(frontier_evaluation, "evaluate_frozen_frontier", fake_evaluation)
    result = run_goal_search(
        build_mini_pythia_model(),
        tmp_path,
        gym=gym,
        budgets=SearchBudgets(max_rounds=1, max_candidate_evaluations=1, min_gain=10.0),
        evaluate_independently=True,
        isolate_independent_evaluation=False,
        include_training_baselines=False,
        training_device="cpu",
    )
    assert result["gym_id"] == gym.gym_id
    assert result["gym_digest"].startswith("sha256:")
    assert result["search_splits"]["solve"] > 0
    assert evaluated_cases == [("Today: The capital of Japan is", " Tokyo")]

    monkeypatch.setattr(
        frontier_evaluation,
        "evaluate_frozen_frontier",
        lambda *args, **kwargs: pytest.fail("must not evaluate with a missing holdout panel"),
    )
    missing_holdout = run_goal_search(
        build_mini_pythia_model(),
        tmp_path / "missing-holdout",
        gym=replace(gym, holdout_rules=()),
        budgets=SearchBudgets(max_rounds=1, max_candidate_evaluations=1, min_gain=10.0),
        include_training_baselines=False,
        training_device="cpu",
        isolate_independent_evaluation=False,
    )
    assert missing_holdout["independent_evaluation"]["complete"] is False
    assert missing_holdout["independent_evaluation"]["independent_terminal_state"] == "qualification_incomplete"


def test_goal_search_replays_an_accepted_rust_preview(tmp_path: Any) -> None:
    from neural_foundry.cumulative_edit import run_goal_search
    from neural_foundry.goal_search import TASK_SPECIALIZED_PROFILE, SearchBudgets
    from neural_foundry.pythia import build_mini_pythia_model

    result = run_goal_search(
        build_mini_pythia_model(),
        tmp_path,
        acceptance_profile=TASK_SPECIALIZED_PROFILE,
        budgets=SearchBudgets(
            max_rounds=1,
            max_candidate_evaluations=50,
            min_gain=0.0,
            max_no_gain_rounds=1,
        ),
        include_training_baselines=False,
        training_device="cpu",
        evaluate_independently=False,
    )

    assert result["rounds"]
    assert result["rounds"][0]["cumulative_changed"] > 0
    assert result["replay"]["matches"] is True


def test_goal_search_refuses_to_overwrite_an_existing_run_directory(tmp_path: Any) -> None:
    from neural_foundry.cumulative_edit import run_goal_search
    from neural_foundry.pythia import build_mini_pythia_model

    (tmp_path / "prior-result.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        run_goal_search(
            build_mini_pythia_model(),
            tmp_path,
            include_training_baselines=False,
            evaluate_independently=False,
        )


def test_cumulative_archive_keeps_round_losers_and_binds_controls_without_a_checkpoint() -> None:
    import torch
    from neural_foundry.cumulative_edit import CumulativeSearch, EvaluatedMove, Move
    from neural_foundry.edit_plan import tensor_state_digest
    from neural_foundry.goal_search import NextMoveDecision

    class TinyModel:
        def __init__(self) -> None:
            self.tensors = {"weight": torch.eye(2, dtype=torch.float64)}
            self.config = {"num_hidden_layers": 1}

        @staticmethod
        def mlp_down_proj_site(_layer: int) -> str:
            return "weight"

    class TinySearch(CumulativeSearch):
        def __init__(self) -> None:
            self.model = TinyModel()
            self.archive = CandidateArchive("tiny-cycle", TASK_SPECIALIZED_PROFILE)
            self.archive.add(record("base", arm="baseline", parent=None))

        def accept_score(self, tensors: Any) -> dict[str, float]:
            changed = sum(int(value.numel()) for value in tensors.values())
            is_control = tensors and "candidate" not in tensors
            return {
                "task": 0.20 if is_control else 0.95,
                "retention": 0.90,
                "kl": 0.10,
                "changed": float(changed),
                "J": 0.20 if is_control else 0.95,
            }

    search = TinySearch()
    move = Move("weight", "task", 0.5, 1_000.0, 1.0)
    base = search.model.tensors["weight"]
    left = torch.tensor([[0.25], [0.0]], dtype=torch.float64)
    right = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    chosen_tensors = {"weight": base + left @ right}
    loser_tensors = {"weight": base + (left @ right) * 0.5}
    chosen = EvaluatedMove("chosen", "proposal:chosen", move, left, right, chosen_tensors, search.accept_score({}))
    loser = EvaluatedMove("loser", "proposal:loser", move, left * 0.5, right, loser_tensors, search.accept_score({}))

    accepted_id = search._archive_round(
        1,
        "base",
        0.10,
        (chosen, loser),
        chosen,
        tensor_state_digest(chosen_tensors),
        NextMoveDecision("target_gap", "The target gap controls the move", "task"),
        1.5,
    )

    assert accepted_id == "chosen"
    direct = [entry for entry in search.archive.records if entry.arm == "direct"]
    controls = [entry for entry in search.archive.records if entry.arm == "control"]
    assert {entry.candidate_id for entry in direct} == {"chosen", "loser"}
    assert [entry.search_state for entry in direct].count("accepted") == 1
    assert [entry.search_state for entry in direct].count("rejected") == 1
    assert {entry.method for entry in controls} == {"random", "placebo"}
    assert {entry.parent_candidate_id for entry in controls} == {"chosen"}
    chosen_record = next(entry for entry in direct if entry.candidate_id == "chosen")
    assert chosen_record.scores["structural_integrity_rate"] == 1.0
    assert "safety_pass_rate" not in chosen_record.scores
    assert chosen_record.replay_verified is True
    assert chosen_record.reload_parity_verified is True
    assert search.archive.decision("chosen").eligible is False
    assert "complete_evidence" in search.archive.decision("chosen").rejection_reasons
    assert "complete_evidence" in search.archive.decision("loser").rejection_reasons


def test_external_training_worker_archives_every_lora_and_sft_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time
    from types import SimpleNamespace

    import torch
    from neural_foundry import cumulative_edit
    from neural_foundry.cumulative_edit import CumulativeSearch
    from neural_foundry.goal_search import SearchBudgets
    from neural_foundry.gpu import ExternalGpuRuntime, TrainingComputePlacement

    class TinyModel:
        def __init__(self) -> None:
            self.tensors = {"weight": torch.eye(2, dtype=torch.float64)}
            self.config = {"num_hidden_layers": 1}

        @staticmethod
        def mlp_down_proj_site(_layer: int) -> str:
            return "weight"

    class ExternalSearch(CumulativeSearch):
        def __init__(self) -> None:
            self.model = TinyModel()
            self.archive = CandidateArchive("external-cycle", TASK_SPECIALIZED_PROFILE)
            self.archive.add(record("base", arm="baseline", method="no_op", parent=None))
            self.include_training_baselines = True
            self.search_sites = set()
            self.lab = SimpleNamespace(
                dev_items=(("Task: Number 7", "odd"),),
                task_data=SimpleNamespace(
                    schema_version=1,
                    gym_id="gym.test.external.v1",
                    gym_digest="sha256:gym",
                    dataset_digest="sha256:dataset",
                    train_digest="sha256:train",
                    validation_digest="sha256:validation",
                ),
            )
            self.solve_guard = ("Hello world",)
            self.budgets = SearchBudgets()
            self.training_compute = TrainingComputePlacement("cpu", "f64", "cpu_reference")
            self.external_gpu_runtime = ExternalGpuRuntime(
                "/opt/nf-gpu/bin/python",
                "2.9.1+cu128",
                "NVIDIA GeForce RTX 5090",
                ("sm_120",),
                total_memory_bytes=32 * 1024**3,
                free_memory_bytes=2 * 1024**3,
            )

        def accept_score(self, tensors: Any) -> dict[str, float]:
            return {
                "task": 0.95,
                "retention": 0.90,
                "kl": 0.10,
                "changed": 4.0,
                "J": 0.95,
            }

    def fake_worker(model: Any, jobs: Any, runtime: Any, **kwargs: Any) -> tuple[dict[str, Any], ...]:
        checkpoints: list[dict[str, Any]] = []
        for job in jobs:
            for step in job["recipe"]["checkpoints"]:
                checkpoints.append(
                    {
                        "job_id": job["job_id"],
                        "gym_id": job["gym_id"],
                        "gym_digest": job["gym_digest"],
                        "dataset_digest": job["dataset_digest"],
                        "kind": job["kind"],
                        "recipe_index": job["recipe_index"],
                        "step": step,
                        "weights": {"weight": model.tensors["weight"] + step / 1000},
                        "wall_seconds": 0.5,
                    }
                )
        return tuple(checkpoints)

    monkeypatch.setattr(cumulative_edit, "train_restricted_external", fake_worker)
    search = ExternalSearch()
    search._archive_training_competitors("base", 0.25)

    baselines = [entry for entry in search.archive.records if entry.arm == "baseline" and entry.method != "no_op"]
    assert len(baselines) == 12
    assert sum(entry.method == "lora" for entry in baselines) == 6
    assert sum(entry.method == "sft" for entry in baselines) == 6
    assert all(entry.proposal["training_runtime"]["mode"] == "external_gpu_worker" for entry in baselines)
    assert all(
        entry.proposal["training_runtime"]["reservation_scope"]
        == "process_local_cooperative_not_native_authority"
        for entry in baselines
    )
    assert search.gpu_scheduler._active_leases == {}
    assert search.gpu_scheduler.vram_capacity_bytes_per_device == 2 * 1024**3

    expired = ExternalSearch()
    expired._training_deadline_monotonic = time.perf_counter() - 1.0
    expired._archive_training_competitors("base", 0.25)
    dispositions = [entry for entry in expired.archive.records if entry.search_state == "skipped"]
    assert len(dispositions) == 4
    assert all("wall_budget_exhausted" in entry.rejection_reasons[0] for entry in dispositions)


def test_external_gpu_training_sends_pinned_checkpoint_reference_not_tensor_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    import torch
    from neural_foundry import cumulative_edit
    from neural_foundry.cumulative_edit import train_restricted_external
    from neural_foundry.gpu import ExternalGpuRuntime
    from neural_foundry.provenance import ResolvedCheckpoint

    checkpoint = ResolvedCheckpoint(
        path=tmp_path,
        model_id="test/pinned-model",
        revision="revision-1",
        source="local_directory",
        file_digests={"config.json": "sha256:" + "a" * 64},
    )
    model = SimpleNamespace(
        checkpoint_identity=checkpoint,
        model_name="test/pinned-model",
        config={"model_type": "gpt_neox"},
        tensors={"weight": torch.ones((2, 2), dtype=torch.float64)},
        tokenizer=SimpleNamespace(vocab={}, bpe_ranks={}, byte_level=True, added_tokens={}),
    )
    model._checkpoint_state_digest = cumulative_edit.tensor_state_digest(model.tensors)
    runtime = ExternalGpuRuntime(
        "/opt/nf-gpu/bin/python", "2.9.1+cu128", "test GPU", ("sm_90",),
        total_memory_bytes=8 * 1024**3, free_memory_bytes=6 * 1024**3,
    )
    observed: dict[str, Any] = {}

    def fake_run(command: list[str], **_kwargs: Any) -> Any:
        payload = torch.load(command[-2], map_location="cpu", weights_only=True)
        observed.update(payload)
        torch.save(
            {
                "schema_version": 1,
                "kind": "restricted_training_checkpoints",
                "input_state_digest": payload["input_state_digest"],
                "inference_identity_digest": payload["inference_identity_digest"],
                "runtime": {"torch": runtime.torch_version, "device": runtime.device, "precision": payload["runtime"]["precision"]},
                "checkpoints": [],
            },
            command[-1],
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cumulative_edit.subprocess, "run", fake_run)
    assert train_restricted_external(model, (), runtime) == ()
    assert observed["checkpoint_identity"]["file_digests"] == checkpoint.file_digests
    assert observed["checkpoint_identity"]["revision"] == checkpoint.revision
    assert "tensors" not in observed
    from neural_foundry.baselines import merged_model

    edited = merged_model(model, {"weight": model.tensors["weight"] + 0.125})
    assert edited.checkpoint_identity is None
    assert train_restricted_external(edited, (), runtime) == ()
    assert "checkpoint_identity" not in observed or "tensors" in observed
    assert observed["input_state_digest"] == cumulative_edit.tensor_state_digest(edited.tensors)


def test_frontier_artifact_roundtrip_and_tamper_are_fail_closed(tmp_path: Any) -> None:
    from dataclasses import replace

    import torch
    from neural_foundry.cumulative_edit import CumulativeSearch
    from neural_foundry.frontier_evaluation import FrontierEvaluationError, load_candidate_artifact

    model = type("TinyModel", (), {})()
    model.tensors = {"weight": torch.eye(2, dtype=torch.float64)}
    search = object.__new__(CumulativeSearch)
    search.model = model
    search.artifact_parent = tmp_path
    search.artifact_directory = tmp_path / "artifacts"
    replacements = {"weight": model.tensors["weight"] * 2.0}
    state_digest, artifact = search._materialize_candidate("candidate", replacements)
    assert artifact is not None
    entry = replace(
        record("candidate", parent=None),
        state_digest=state_digest,
        artifact=artifact,
    ).to_document()

    loaded = load_candidate_artifact(entry, model.tensors, tmp_path)
    assert torch.equal(loaded["weight"], replacements["weight"])
    artifact_path = tmp_path / artifact.relative_path
    artifact_path.write_bytes(artifact_path.read_bytes() + b"tampered")
    with pytest.raises(FrontierEvaluationError, match="byte count changed"):
        load_candidate_artifact(entry, model.tensors, tmp_path)


def test_frozen_frontier_evaluation_reads_only_materialized_records(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    import torch
    from neural_foundry import frontier_evaluation
    from neural_foundry.cumulative_edit import CumulativeSearch

    model = type("TinyModel", (), {})()
    model.tensors = {"weight": torch.eye(2, dtype=torch.float64)}
    model.tokenizer = object()
    search = object.__new__(CumulativeSearch)
    search.model = model
    search.artifact_parent = tmp_path
    search.artifact_directory = tmp_path / "artifacts"
    state_digest, artifact = search._materialize_candidate(
        "candidate", {"weight": model.tensors["weight"] * 2.0}
    )
    archive = CandidateArchive("frontier-cycle", TASK_SPECIALIZED_PROFILE)
    archive.add(record("base", arm="baseline", method="no_op", parent=None))
    archive.add(
        replace(
            record("candidate"),
            state_digest=state_digest,
            artifact=artifact,
            replay_verified=False,
            reload_parity_verified=False,
        )
    )
    random_control, placebo_control = controls("candidate")
    archive.add(random_control)
    archive.add(placebo_control)
    snapshot = archive.freeze()
    calls: list[str] = []

    def fake_evaluator(_model: Any, frozen: Any, *, config: Any = None) -> dict[str, Any]:
        calls.append(frozen.candidate_id)
        return {"kind": "nf_independent_eval_report_v1", "candidate_id": frozen.candidate_id}

    monkeypatch.setattr(
        frontier_evaluation.promotion_gate,
        "decide_promotion",
        lambda report: {
            "promote_eligible": True,
            "status": "passed",
            "failed_checks": [],
            "checks": [],
        },
    )
    result = frontier_evaluation.evaluate_frozen_frontier(
        model,
        snapshot,
        artifact_root=tmp_path,
        target_cases=(('Task: Number 7', 'odd'),),
        evaluator=fake_evaluator,
    )

    assert calls == ["candidate"]
    assert result["promoted_candidate_ids"] == []
    assert result["rejected_candidate_ids"] == ["candidate"]
    deployment = result["evaluated"][0]["deployment_decision"]
    assert deployment["failed_checks"] == [
        "profile_eligible",
        "replay_verified",
        "reload_parity_verified",
        "independent_target_custody",
    ]
    assert deployment["checks"]["independent_target_custody"] is False
    assert result["complete"] is True


def test_qualification_roster_keeps_eligible_direct_candidate_dominated_by_lora(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    import torch
    from neural_foundry import frontier_evaluation
    from neural_foundry.cumulative_edit import CumulativeSearch
    from neural_foundry.records import canonicalize, content_digest

    model = type("TinyModel", (), {})()
    model.tensors = {"weight": torch.eye(2, dtype=torch.float64)}
    search = object.__new__(CumulativeSearch)
    search.model = model
    search.artifact_parent = tmp_path
    search.artifact_directory = tmp_path / "artifacts"

    direct_state, direct_artifact = search._materialize_candidate(
        "direct", {"weight": model.tensors["weight"] * 2.0}
    )
    lora_state, lora_artifact = search._materialize_candidate(
        "lora", {"weight": model.tensors["weight"] * 3.0}
    )
    archive = CandidateArchive("dominated-direct-cycle", TASK_SPECIALIZED_PROFILE)
    archive.add(record("base", arm="baseline", method="no_op", parent=None, target=0.1, target_gain=0.0, retention=1.0, kl=0.0))
    direct = replace(
        record("direct", parent="base", target=0.95, target_gain=0.85, retention=0.9, kl=0.05),
        state_digest=direct_state,
        artifact=direct_artifact,
    )
    archive.add(direct)
    random_control, placebo_control = controls("direct")
    archive.add(random_control)
    archive.add(placebo_control)
    dataset = {
        name: str(content_digest(canonicalize({"dataset": name})))
        for name in ("gym_digest", "dataset_digest", "train_digest", "validation_digest")
    }
    recipe = {"seed": 17, "steps": 4, "rank": 2}
    lora = replace(
        record("lora", arm="baseline", method="lora", parent="base", target=1.0,
               target_gain=0.9, retention=1.0, kl=0.0, replay=False, parity=False),
        state_digest=lora_state,
        artifact=lora_artifact,
        proposal={
            "kind": "lora",
            "recipe": recipe,
            "recipe_schema_version": 1,
            "dataset": dataset,
            "sites": ["weight"],
            "step": 4,
            "training_runtime": {"device": "cpu"},
        },
    )
    archive.add(lora)
    snapshot = archive.freeze()
    assert direct.candidate_id in snapshot.eligible_ids
    assert direct.candidate_id not in snapshot.search_frontier_ids

    calls: list[str] = []

    def fake_evaluator(_model: Any, frozen: Any, *, config: Any = None) -> dict[str, Any]:
        calls.append(frozen.candidate_id)
        return {"kind": "nf_independent_eval_report_v1", "candidate_id": frozen.candidate_id}

    monkeypatch.setattr(
        frontier_evaluation.promotion_gate,
        "decide_promotion",
        lambda _report: {"promote_eligible": False, "status": "rejected", "checks": [], "failed_checks": ["fixture"]},
    )
    result = frontier_evaluation.evaluate_frozen_frontier(
        model,
        snapshot,
        artifact_root=tmp_path,
        target_cases=(("public task", "answer"),),
        evaluator=fake_evaluator,
    )

    assert calls == ["direct", "lora"]
    assert result["qualification_roster_ids"] == ["direct", "lora"]
    assert {row["candidate_id"] for row in result["evaluated"]} == {"direct", "lora"}
    lora_result = next(row for row in result["evaluated"] if row["candidate_id"] == "lora")
    assert lora_result["method_checks"]["artifact_reload_verified"] is True
    assert lora_result["method_checks"]["recipe_and_dataset_bound"] is True
    assert lora_result["method_checks"]["retraining_reproducibility_verified"] is False


def test_frozen_snapshot_rejects_mutated_derived_ids() -> None:
    from neural_foundry.goal_search import validate_frontier_snapshot

    archive = CandidateArchive("tamper-cycle", TASK_SPECIALIZED_PROFILE)
    archive.add(record("candidate", parent=None))
    document = archive.freeze().to_document()
    document["eligible_ids"] = ["candidate"]

    with pytest.raises(GoalSearchError, match="Frozen"):
        validate_frontier_snapshot(document)


def test_independent_target_labels_round_trip_for_pythia_and_qwen() -> None:
    from neural_foundry.cumulative_edit import independent_target_cases
    from neural_foundry.pythia import build_mini_pythia_model, build_mini_qwen2_causal_lm

    for model in (build_mini_pythia_model(), build_mini_qwen2_causal_lm()):
        internal = ("Task: Number 7", "Ġodd")
        cases = independent_target_cases(model, (internal,))
        assert cases[0][0] == internal[0]
        assert model.tokenizer.encode(cases[0][1]) == [model.tokenizer.vocab[internal[1]]]


def test_qwen_capture_exposes_supported_cumulative_sites() -> None:
    from neural_foundry.cumulative_edit import Capture, candidate_sites
    from neural_foundry.pythia import build_mini_qwen2_causal_lm

    model = build_mini_qwen2_causal_lm()
    sites = candidate_sites(model)
    capture = Capture(model, ("Task: Number 7",))

    assert sites == [
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.1.mlp.down_proj.weight",
        "model.layers.1.self_attn.o_proj.weight",
    ]
    assert set(capture.inputs) >= set(sites)


def test_cumulative_search_uses_custom_gym_for_weight_space_and_training_targets() -> None:
    from neural_foundry.baselines import build_splits
    from neural_foundry.cumulative_edit import CumulativeSearch
    from neural_foundry.pythia import ReusableTaskGym, build_mini_pythia_model

    gym = ReusableTaskGym(
        gym_id="gym.test.custom_targets.v1",
        skills_taught=("custom_logic",),
        base_rules=(
            ("Boolean False AND True gives", "ĠTrue"),
            ("Boolean False OR True gives", "ĠFalse"),
        ),
        validation_rules=(
            ("Boolean True AND True gives", "ĠTrue"),
            ("Boolean True OR True gives", "ĠTrue"),
        ),
    )
    search = CumulativeSearch(build_mini_pythia_model(), 1e-7, gym=gym, include_training_baselines=False)

    assert search.lab is search.mlab.lab
    assert search.lab.gym == gym
    assert search.lab.dev_items == [
        ("Task: Boolean False AND True gives", "ĠTrue"),
        ("Task: Boolean False OR True gives", "ĠFalse"),
        ("Rule: Boolean False AND True gives", "ĠTrue"),
        ("Rule: Boolean False OR True gives", "ĠFalse"),
    ]
    assert search.mlab.task_ids == [
        search.model.tokenizer.vocab["ĠTrue"],
        search.model.tokenizer.vocab["ĠFalse"],
    ]
    baseline_splits = build_splits(gym, (), model=search.model)
    assert baseline_splits.dev == tuple(search.lab.dev_items)
    assert baseline_splits.val == tuple(search.lab.val_items)
    assert baseline_splits.dataset_digest == search.lab.task_data.dataset_digest


def test_studio_exposes_goal_search(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    from neural_foundry import cumulative_edit
    from neural_foundry.pythia import build_mini_qwen2_causal_lm
    from neural_foundry.studio import WeightProgrammingStudio

    calls: list[tuple[Any, Path]] = []

    def fake_run(model: Any, root: Path, **kwargs: Any) -> dict[str, Any]:
        calls.append((model, Path(root)))
        return {"frontier_snapshot": {"snapshot_digest": "sha256:test"}, "kwargs": kwargs}

    monkeypatch.setattr(cumulative_edit, "run_goal_search", fake_run)
    model = build_mini_qwen2_causal_lm()
    result = WeightProgrammingStudio().run_goal_search(
        model,
        tmp_path,
        include_training_baselines=False,
        training_device="cpu",
        evaluate_independently=False,
    )

    assert calls and calls[0] == (model, tmp_path)
    assert result["kwargs"]["include_training_baselines"] is False
    assert result["kwargs"]["isolate_independent_evaluation"] is True


def test_frozen_frontier_evaluation_runs_in_a_fresh_process(tmp_path: Any) -> None:
    from neural_foundry import frontier_evaluation
    from neural_foundry.pythia import build_mini_pythia_model

    evaluate_frozen_frontier_isolated = frontier_evaluation.evaluate_frozen_frontier_isolated

    model = build_mini_pythia_model()
    archive = CandidateArchive("isolated-cycle", TASK_SPECIALIZED_PROFILE)
    archive.add(
        record(
            "base",
            arm="baseline",
            method="no_op",
            parent=None,
            target=0.0,
            target_gain=0.0,
            retention=1.0,
            kl=0.0,
        )
    )
    snapshot = archive.freeze()

    report = evaluate_frozen_frontier_isolated(
        model,
        snapshot,
        artifact_root=tmp_path,
        target_cases=(("Task: Number 7", "odd"),),
        timeout_seconds=60.0,
    )

    assert report["execution_mode"] == "isolated_subprocess"
    assert report["snapshot_digest"] == snapshot.snapshot_digest
    assert report["evaluated"] == []
    assert report["not_evaluated"] == [{"candidate_id": "base", "reason": "unchanged_base_reference"}]
    assert "cases" not in report.get("target", {})
    assert report["target_case_count"] == 1
    retry = evaluate_frozen_frontier_isolated(
        model,
        snapshot,
        artifact_root=tmp_path,
        target_cases=(("Task: Number 7", "odd"),),
        timeout_seconds=60.0,
    )
    assert retry == report
    with pytest.raises(frontier_evaluation.FrontierEvaluationError, match="different evaluation request"):
        evaluate_frozen_frontier_isolated(
            model,
            snapshot,
            artifact_root=tmp_path,
            target_cases=(("Changed task prompt", "odd"),),
            timeout_seconds=60.0,
        )


def test_exposure_claim_retries_only_the_identical_frozen_request(tmp_path: Any) -> None:
    import json
    import os
    import subprocess
    import sys

    from neural_foundry import frontier_evaluation
    from neural_foundry.records import canonicalize, content_digest

    document = {"snapshot_digest": "sha256:" + "a" * 64}
    request_digest = str(content_digest(canonicalize({"request": "same"})))
    with frontier_evaluation._evaluation_lease(document, tmp_path):
        claim_paths, cached = frontier_evaluation._begin_evaluation_claim(
            document, tmp_path, request_digest
        )
        assert cached is None
        with (
            pytest.raises(frontier_evaluation.FrontierEvaluationError, match="lease is already active"),
            frontier_evaluation._evaluation_lease(document, tmp_path),
        ):
            pytest.fail("a second evaluator must not acquire the active lease")

    code = (
        "import json, sys; from neural_foundry import frontier_evaluation as f; "
        "document=json.loads(sys.argv[1]); "
        "lease=f._evaluation_lease(document,sys.argv[2]); lease.__enter__(); "
        "f._begin_evaluation_claim(document,sys.argv[2],sys.argv[3]); "
        "print('leased',flush=True); sys.stdin.read()"
    )
    worker = subprocess.Popen(
        [sys.executable, "-c", code, json.dumps(document), str(tmp_path), request_digest],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert worker.stdout is not None and worker.stdout.readline().strip() == "leased"
        with (
            pytest.raises(frontier_evaluation.FrontierEvaluationError, match="lease is already active"),
            frontier_evaluation._evaluation_lease(document, tmp_path),
        ):
            pytest.fail("a live worker must retain its lease")
    finally:
        worker.kill()
        worker.communicate(timeout=10)

    if os.name != "nt":
        with frontier_evaluation._evaluation_lease(document, tmp_path):
            descriptor = frontier_evaluation._ACTIVE_EVALUATION_DESCRIPTOR.get()
            assert descriptor is not None
            child = subprocess.Popen(
                [sys.executable, "-c", "import os,sys; os.fstat(int(sys.argv[1])); print('inherited',flush=True); sys.stdin.read()", str(descriptor)],
                pass_fds=(descriptor,), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            )
            assert child.stdout is not None and child.stdout.readline().strip() == "inherited"
        try:
            with (
                pytest.raises(frontier_evaluation.FrontierEvaluationError, match="lease is already active"),
                frontier_evaluation._evaluation_lease(document, tmp_path),
            ):
                pytest.fail("a surviving isolated child must retain the parent's lease")
        finally:
            child.kill()
            child.communicate(timeout=10)

    report = {"schema_version": 1, "kind": "fixture", "snapshot_digest": document["snapshot_digest"], "complete": True}
    report["report_digest"] = str(content_digest(canonicalize(report)))
    with frontier_evaluation._evaluation_lease(document, tmp_path):
        retry_paths, cached = frontier_evaluation._begin_evaluation_claim(
            document, tmp_path, request_digest
        )
        assert retry_paths == claim_paths and cached is None
        frontier_evaluation._finish_evaluation_claim(claim_paths, report)
    with frontier_evaluation._evaluation_lease(document, tmp_path):
        completed_paths, cached = frontier_evaluation._begin_evaluation_claim(
            document, tmp_path, request_digest
        )
        assert completed_paths == claim_paths and cached == report
    changed_request_digest = str(content_digest(canonicalize({"request": "changed"})))
    with (
        pytest.raises(frontier_evaluation.FrontierEvaluationError, match="different evaluation request"),
        frontier_evaluation._evaluation_lease(document, tmp_path),
    ):
        frontier_evaluation._begin_evaluation_claim(document, tmp_path, changed_request_digest)


def test_failed_search_records_are_disposed_without_blocking_qualification() -> None:
    from neural_foundry.frontier_evaluation import _qualification_roster

    records = {
        "passing": {"arm": "direct", "method": "wsp", "search_state": "accepted"},
        "failed": {"arm": "direct", "method": "wsp", "search_state": "failed"},
        "skipped": {"arm": "baseline", "method": "lora", "search_state": "skipped"},
        "no-op": {"arm": "baseline", "method": "no_op", "search_state": "accepted"},
        "control": {"arm": "control", "method": "random", "search_state": "rejected"},
    }
    assert _qualification_roster(records) == ("passing",)


def test_incomplete_evaluation_retry_limit_is_bounded(tmp_path: Any) -> None:
    from neural_foundry import frontier_evaluation
    from neural_foundry.records import canonicalize, content_digest

    document = {"snapshot_digest": "sha256:" + "d" * 64}
    request = str(content_digest(canonicalize({"request": "same"})))
    with frontier_evaluation._evaluation_lease(document, tmp_path):
        paths, cached = frontier_evaluation._begin_evaluation_claim(document, tmp_path, request)
    assert cached is None
    for _ in range(frontier_evaluation.MAX_EXPOSURE_RETRIES):
        with frontier_evaluation._evaluation_lease(document, tmp_path):
            same_paths, cached = frontier_evaluation._begin_evaluation_claim(document, tmp_path, request)
        assert same_paths == paths and cached is None
    with (
        pytest.raises(frontier_evaluation.FrontierEvaluationError, match="retry limit is exhausted"),
        frontier_evaluation._evaluation_lease(document, tmp_path),
    ):
        frontier_evaluation._begin_evaluation_claim(document, tmp_path, request)


def test_evaluator_identity_binds_tokenizer_config_and_incomplete_retry(tmp_path: Any) -> None:
    from neural_foundry import frontier_evaluation as evaluation
    from neural_foundry.pythia import build_mini_pythia_model
    from neural_foundry.records import canonicalize, content_digest

    model = build_mini_pythia_model()
    snapshot = {"snapshot_digest": "sha256:" + "b" * 64}

    def identity() -> str:
        return evaluation._evaluation_request_digest(
            snapshot, model, (), None, execution_mode="in_process", evaluator_id="fixture",
        )

    original = identity()
    model.config["layer_norm_eps"] = 0.1
    changed_config = identity()
    assert changed_config != original
    model.tokenizer.vocab["new-token"] = 10000
    assert identity() != changed_config
    with evaluation._evaluation_lease(snapshot, tmp_path):
        paths, _ = evaluation._begin_evaluation_claim(snapshot, tmp_path, original)
        incomplete = {"kind": "fixture", "complete": False}
        incomplete["report_digest"] = str(content_digest(canonicalize(incomplete)))
        evaluation._finish_evaluation_claim(paths, incomplete)
    with evaluation._evaluation_lease(snapshot, tmp_path):
        retry_paths, cached = evaluation._begin_evaluation_claim(snapshot, tmp_path, original)
        assert retry_paths == paths and cached is None


@pytest.mark.parametrize("options", [
    {"max_wall_seconds": float("nan")}, {"max_wall_seconds": float("inf")},
    {"max_candidate_evaluations": 1.5}, {"max_rounds": True}, {"min_gain": False},
])
def test_search_budget_domains_are_strict(options: dict[str, Any]) -> None:
    from neural_foundry.goal_search import GoalSearchError, SearchBudgets

    with pytest.raises(GoalSearchError):
        SearchBudgets(**options)


def test_search_evaluation_reservation_is_shared_and_deadline_gated() -> None:
    import time

    from neural_foundry.cumulative_edit import CumulativeSearch
    from neural_foundry.goal_search import SearchBudgets

    search = object.__new__(CumulativeSearch)
    search.budgets = SearchBudgets(max_candidate_evaluations=2)
    search._training_deadline_monotonic = time.perf_counter() + 10
    assert search._reserve_candidate_evaluation()
    assert search._reserve_candidate_evaluation()
    assert not search._reserve_candidate_evaluation()
    search._evaluations_used = 0
    search._training_deadline_monotonic = time.perf_counter() - 1
    assert not search._reserve_candidate_evaluation()


@pytest.mark.parametrize("change", [{"lr": float("nan")}, {"steps": 1.5}, {"rank": True}, {"checkpoints": [3]}])
def test_training_recipes_reject_malformed_domains(change: dict[str, Any]) -> None:
    from neural_foundry.cumulative_edit import _validate_training_recipe

    recipe = {"kind": "lora", "lr": 0.001, "steps": 2, "weight": 1.0, "rank": 2, "checkpoints": [1, 2]}
    with pytest.raises(ValueError):
        _validate_training_recipe({**recipe, **change}, 17)


def test_evaluator_cache_distinguishes_arithmetic_dtype_without_weight_changes() -> None:
    from types import SimpleNamespace

    import torch
    from neural_foundry import frontier_evaluation, independent_eval

    model = SimpleNamespace(tensors={"w": torch.ones(1, dtype=torch.float64)}, config={}, tokenizer=None)
    document = {"snapshot_digest": "sha256:" + "c" * 64}
    original_state = independent_eval.parameter_digest(model.tensors)
    before = frontier_evaluation._evaluation_request_digest(document, model, (), None, execution_mode="in_process", evaluator_id="fixture")
    model.tensors["w"] = model.tensors["w"].float()
    assert independent_eval.parameter_digest(model.tensors) == original_state
    after = frontier_evaluation._evaluation_request_digest(document, model, (), None, execution_mode="in_process", evaluator_id="fixture")
    assert after != before


def test_isolated_evaluator_uses_verified_checkpoint_identity_not_tensor_dump(tmp_path: Any) -> None:
    from types import SimpleNamespace

    import torch
    from neural_foundry.edit_plan import tensor_state_digest
    from neural_foundry.frontier_evaluation import _isolated_model_payload
    from neural_foundry.provenance import resolve_checkpoint

    for name, body in (("config.json", b"{}"), ("tokenizer.json", b"{}"), ("model.safetensors", b"placeholder")):
        (tmp_path / name).write_bytes(body)
    checkpoint = resolve_checkpoint(tmp_path, revision="a" * 40)
    tokenizer = SimpleNamespace(vocab={"x": 1}, bpe_ranks={}, added_tokens={}, byte_level=False)
    model = SimpleNamespace(
        model_name="pinned-mini", config={"model_type": "gpt_neox"},
        tensors={"weight": torch.ones((2, 2), dtype=torch.float64)}, tokenizer=tokenizer,
        checkpoint_identity=checkpoint,
    )
    model._checkpoint_state_digest = tensor_state_digest(model.tensors)
    payload = _isolated_model_payload(model)
    assert "checkpoint_identity" in payload and "tensors" not in payload
    assert payload["checkpoint_identity"]["file_digests"] == checkpoint.file_digests
    model.tensors["weight"] = model.tensors["weight"] + 1
    fallback = _isolated_model_payload(model)
    assert fallback["tensors"]["weight"].dtype == torch.float64


def test_explicit_external_device_is_forwarded_before_local_torch_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    from neural_foundry import cumulative_edit
    from neural_foundry.gpu import ExternalGpuRuntime
    from neural_foundry.pythia import build_mini_pythia_model

    observed = []

    def discover(configured=None, *, device=None):
        observed.append((configured, device))
        return ExternalGpuRuntime("/configured/python", "test-torch", "test-gpu", (), device=device)

    monkeypatch.setattr(cumulative_edit, "discover_external_gpu_runtime", discover)
    monkeypatch.setattr(cumulative_edit, "select_training_compute", lambda device: pytest.fail("explicit external runtime must be selected first"))
    search = cumulative_edit.CumulativeSearch(
        build_mini_pythia_model(), 1e-7, include_training_baselines=True,
        training_device="cuda:2", gpu_python="/configured/python",
    )
    assert observed == [("/configured/python", "cuda:2")]
    assert search.training_compute.device == search.external_gpu_runtime.device == "cuda:2"
