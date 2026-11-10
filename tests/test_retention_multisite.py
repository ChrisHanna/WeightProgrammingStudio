"""Protocol guards and solver identities for the multi-site retention work."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neural_foundry import retention_multisite as ms
from neural_foundry import retention_research as rr
from neural_foundry.prd_adapters import PRACTICE_RETENTION_PROMPTS, SEALED_RETENTION_PROMPTS
from neural_foundry.pythia import ALGORITHMIC_THREE_SKILL_GYM as GYM
from neural_foundry.pythia import PythiaModel

CHECKPOINT = Path("/tmp/pythia-14m")
needs_checkpoint = pytest.mark.skipif(not CHECKPOINT.exists(), reason="needs the Pythia-14M checkpoint")


def test_v3_sealed_set_shares_nothing_with_public_or_older_sets() -> None:
    fit, tune = rr.split_corpus()
    others = (*fit, *tune, *PRACTICE_RETENTION_PROMPTS, *SEALED_RETENTION_PROMPTS, *rr.FRESH_SEALED_PROMPTS, *GYM.retention_prompts, *rr.gym_task_texts())
    trigrams = set().union(*(rr._trigrams(p) for p in others))
    assert len(set(ms.SEALED_V3_RETENTION)) == len(ms.SEALED_V3_RETENTION) == 120
    for prompt in ms.SEALED_V3_RETENTION:
        assert prompt not in others, prompt
        assert not rr._trigrams(prompt) & trigrams, prompt


def test_v3_task_items_are_new_and_defeat_the_boolean_shortcut() -> None:
    public = [e for e, _ in (*GYM.base_rules, *GYM.holdout_rules)]
    for prompt, _ in ms.SEALED_V3_TASK:
        assert not any(expr in prompt for expr in public), prompt
    shortcut = {"AND": "ĠFalse", "OR": "ĠTrue"}
    boolean = [(p, t) for p, t in ms.SEALED_V3_TASK if "Boolean" in p]
    hits = sum(shortcut["AND" if " AND " in p else "OR"] == t for p, t in boolean)
    assert hits == len(boolean) / 2


def test_ridge_map_meets_its_stationarity_condition() -> None:
    g = torch.Generator().manual_seed(1)
    k = torch.randn((9, 4), generator=g, dtype=torch.float64)
    y = torch.randn((3, 4), generator=g, dtype=torch.float64)
    kp = torch.randn((9, 7), generator=g, dtype=torch.float64)
    yp = torch.randn((3, 7), generator=g, dtype=torch.float64)
    d = ms.ridge_map(k, y, kp, yp, alpha=14.0, ridge=0.5)
    gradient = (d @ k - y) @ k.T + 2.0 * (d @ kp - yp) @ kp.T + 0.5 * d
    assert float(gradient.abs().max()) < 1e-10


def test_holm_is_monotone_and_capped() -> None:
    adjusted = ms.holm({"a": 0.01, "b": 0.04, "c": 0.5})
    assert adjusted == {"a": pytest.approx(0.03), "b": pytest.approx(0.08), "c": pytest.approx(0.5)}
    assert ms.holm({"a": 0.9, "b": 0.8})["a"] == 1.0


def test_identical_methods_have_a_zero_difference_interval() -> None:
    hits = [True, False] * 10
    assert ms.paired_difference_ci(hits, hits) == (0.0, 0.0)


def test_tune_path_has_no_reference_to_sealed_sets() -> None:
    source = Path(ms.__file__).read_text(encoding="utf-8")
    for fn in ("run_tune", "_tune_gradient", "run_freeze"):
        body = source.split(f"def {fn}(")[1].split("\ndef ")[0]
        for name in ("SEALED_V3_RETENTION", "SEALED_V3_TASK", "holdout_rules", "SEALED_RETENTION_PROMPTS", "FRESH_SEALED"):
            assert name not in body, (fn, name)


@pytest.fixture(scope="module")
def lab() -> ms.MultiLab:
    return ms.MultiLab(PythiaModel.from_pretrained_dir(CHECKPOINT))


@needs_checkpoint
def test_residual_sink_does_not_change_logits(lab: ms.MultiLab) -> None:
    ids = lab.model.tokenizer.encode("Once more into the")
    sink: list[torch.Tensor] = []
    with_sink = lab.model.forward_with_mlp_key(ids, residual_sink=sink)[0]
    assert torch.equal(with_sink, lab.model.forward_with_mlp_key(ids)[0])
    assert sink[0].shape == (len(ids), lab.model.config["hidden_size"])


@needs_checkpoint
def test_joint_with_one_layer_equals_single(lab: ms.MultiLab) -> None:
    a = lab.solve(ms.MultiConfig("single", (5,), 0.5, 1e3, 1.0))
    b = lab.solve(ms.MultiConfig("joint", (5,), 0.5, 1e3, 1.0))
    site = lab.model.mlp_down_proj_site(5)
    assert float((a.merged[site] - b.merged[site]).abs().max()) < 1e-9


@needs_checkpoint
def test_parallel_two_layers_halves_each_share(lab: ms.MultiLab) -> None:
    one = lab.solve(ms.MultiConfig("single", (5,), 1.0, 1e3, 1.0, rank_limit=128))
    two = lab.solve(ms.MultiConfig("parallel", (4, 5), 1.0, 1e3, 1.0, rank_limit=128))
    site = lab.model.mlp_down_proj_site(5)
    base = lab.model.tensors[site]
    assert float(((two.merged[site] - base) * 2 - (one.merged[site] - base)).abs().max()) < 1e-7


@needs_checkpoint
def test_readout_only_changes_task_token_rows(lab: ms.MultiLab) -> None:
    cand = lab.solve(ms.MultiConfig("readout", (), 1.0, 10.0, 1.0, rank_limit=6))
    diff = (cand.merged[ms.READOUT_SITE] != lab.model.tensors[ms.READOUT_SITE]).any(dim=1).nonzero().flatten().tolist()
    assert set(diff) <= set(lab.task_ids)
    lab.execute(cand)


@needs_checkpoint
def test_executor_agrees_with_sweep_on_every_multi_site(lab: ms.MultiLab) -> None:
    done = lab.execute(lab.solve(ms.MultiConfig("compensated", (4, 5), 0.1, 1e3, 1.0)))
    assert len(done.digests) == 2


@needs_checkpoint
def test_random_control_matches_norm_and_rank_per_site(lab: ms.MultiLab) -> None:
    cand = lab.solve(ms.MultiConfig("parallel", (4, 5), 0.5, 1e3, 1.0))
    control = lab.random_like(lab.execute(cand))
    for site, (left, right, _) in cand.factors.items():
        base = lab.model.tensors[site]
        assert float(torch.linalg.norm(control.merged[site] - base)) == pytest.approx(float(torch.linalg.norm(left @ right)), rel=1e-9)
        assert control.factors[site][0].shape == left.shape


@needs_checkpoint
def test_edit_restores_source_bytes(lab: ms.MultiLab) -> None:
    cand = lab.execute(lab.solve(ms.MultiConfig("parallel", (4, 5), 0.5, 1e3, 1.0)))
    original = {k: ms.f32_bytes(lab.model.tensors[k]) for k in cand.merged}
    result = ms.reversibility(lab.model, original, cand.merged, cand.factors)
    assert result["before_image_bit_exact"] is True
    assert result["delta_inverse_rounded_f32_bit_exact"] is True


def test_accept_split_is_group_wise_and_excludes_reporting_data() -> None:
    from neural_foundry import cumulative_edit as ce

    solve, accept = ce.accept_split()
    fit, tune = rr.split_corpus()
    assert set(solve) | set(accept) == set(fit)
    assert not set(solve) & set(accept)
    assert not (set(solve) | set(accept)) & set(tune)
    for prompts in rr.build_corpus().values():
        kept = set(prompts) & set(fit)
        assert kept <= set(solve) or kept <= set(accept)


def test_goal_search_split_has_group_wise_monitor_disjointness() -> None:
    from neural_foundry import cumulative_edit as ce

    splits = ce.goal_search_splits()
    fit, tune = rr.split_corpus()
    parts = [set(splits.solve_guard), set(splits.accept_guard), set(splits.monitor_guard)]
    assert all(parts)
    assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
    assert set().union(*parts) == set(fit)
    assert not set().union(*parts) & set(tune)
    for prompts in rr.build_corpus().values():
        kept = set(prompts) & set(fit)
        if kept:
            assert sum(kept <= part for part in parts) == 1


def test_move_policy_is_deterministic_and_logged_by_rule_id() -> None:
    from neural_foundry import cumulative_edit as ce
    from neural_foundry.goal_search import NextMoveDecision

    moves = (
        ce.Move("a", "task", 1.0, 1e3, 1.0),
        ce.Move("a", "task", 0.1, 1e3, 1.0),
        ce.Move("a", "repair", 0.0, 1e3, 100.0),
    )
    task = ce.moves_for_policy(moves, NextMoveDecision("target_gap", "target", "task"))
    repair = ce.moves_for_policy(
        moves, NextMoveDecision("retention_or_kl_gap", "retention", "retention_repair")
    )
    assert [move.mode for move in task] == ["task", "task"]
    assert [move.mode for move in repair] == ["repair", "task"]
    assert ce.moves_for_policy(moves, NextMoveDecision("stop", "stop", "none", "monitor_drift")) == []


def test_objective_charges_for_parameters() -> None:
    from neural_foundry import cumulative_edit as ce

    assert ce.objective(0.5, 0.8, 1.0, 1_000_000, 1e-7) == pytest.approx(0.5 + 0.8 - 0.1 - 0.1)


@needs_checkpoint
def test_cumulative_steps_replay_to_the_same_digest(lab: ms.MultiLab) -> None:
    from neural_foundry import cumulative_edit as ce
    from neural_foundry.edit_plan import EditPlan, apply_edit_graph

    search = ce.CumulativeSearch(lab.model, 1e-7)
    site = lab.model.mlp_down_proj_site(5)
    for move in (ce.Move(site, "task", 0.5, 1e3, 1.0), ce.Move(lab.model.mlp_down_proj_site(4), "repair", 0.0, 1e3, 1.0)):
        current = search.current()
        dev, guard = ce.Capture(current, search.dev_prompts), ce.Capture(current, search.solve_guard)
        left, right = search._solve(move, dev, guard, ms.contrast(dev.residual - search.base_dev.residual), guard.residual - search.base_solve.residual, current)
        search.plan.low_rank(move.site, left.contiguous(), right.contiguous())
        done = search.plan.preview()
        search.tensors = {k: v for k, v in done.model.tensors.items() if not torch.equal(v, lab.model.tensors[k])}
    graph = search.plan.to_graph()
    again = apply_edit_graph(lab.model, graph)
    assert again.state_digest == graph["candidate_state_digest"]
    assert len(search.tensors) == 2


@needs_checkpoint
def test_restricted_training_touches_only_named_tensors(lab: ms.MultiLab) -> None:
    from neural_foundry import cumulative_edit as ce

    site = lab.model.mlp_down_proj_site(5)
    recipe = {"kind": "lora", "lr": 1e-3, "steps": 2, "weight": 1.0, "rank": 2, "checkpoints": ()}
    weights, _, trainable = ce.train_restricted(lab.model, [site], recipe, lab.lab.dev_items[:2], lab.lab.fit[:8])
    assert set(weights) == {site}
    assert trainable == 2 * (128 + 512)
