from __future__ import annotations

import pytest
import torch

from neural_foundry import baselines as bl
from neural_foundry.pythia import (
    ALGORITHMIC_THREE_SKILL_GYM as GYM,
    build_mini_pythia_model,
)

PRACTICE = ("Hello world", "The capital of France is", "Once upon a time there was", "One plus one equals")
SEALED = ("In computer science a list is", "The capital of Japan is")


@pytest.fixture()
def model():
    return build_mini_pythia_model()


@pytest.fixture()
def splits(model):
    return bl.build_splits(GYM, PRACTICE, model=model)


def lora(**kw):
    defaults = dict(kind="lora", lr=0.02, steps=30, targets="final_down_proj", rank=4)
    return bl.BaselineConfig(**{**defaults, **kw})


def sft(**kw):
    return bl.BaselineConfig(**{**dict(kind="sft", lr=0.002, steps=30), **kw})


def dev_loss(model, items):
    with torch.no_grad():
        return float(bl._mean_ce(model, items).item())


def test_precision_conversion_is_not_a_learned_delta():
    original = {"weight": torch.tensor([1.0 + 1e-10], dtype=torch.float64)}
    training_base = {"weight": original["weight"].float()}
    unchanged = bl._training_update_snapshot(original, training_base, training_base)
    assert torch.equal(unchanged["weight"], original["weight"])
    assert bl._conversion_drift(original, training_base) > 0
    changed = bl._training_update_snapshot(original, training_base, {"weight": training_base["weight"] + 0.25})
    assert torch.equal(changed["weight"] - original["weight"], torch.tensor([0.25], dtype=torch.float64))


def test_byte_level_sequence_decoder_preserves_utf8():
    from neural_foundry.pythia import PythiaBPETokenizer

    tokenizer = PythiaBPETokenizer({"Ã": 0, "©": 1, "Ã©": 2}, (), byte_level=True)
    assert tokenizer.decode((0, 1)) == "é"
    assert tokenizer.decode_token(2) == "é"


def test_holdout_reuse_uses_model_identity_and_supports_no_prefixes(model, monkeypatch):
    from dataclasses import replace

    panel = bl.SealedHoldout(replace(GYM, holdout_prefixes=()), ())
    assert panel._items
    monkeypatch.setattr(bl, "accuracy", lambda candidate, items: 0.0)
    panel.evaluate_once("first-name", model, {})
    with pytest.raises(RuntimeError, match="already used"):
        panel.evaluate_once("different-name", model, {})


def test_gradients_reach_every_parameter_through_the_real_forward_pass(model, splits):
    fit = bl.fit_baseline(model, sft(steps=1), dev=splits.dev)
    # The unembedding and an early layer both move, so this is not a one-matrix fit.
    assert "embed_out.weight" in fit.merged_weights
    assert model.mlp_down_proj_site(0) in fit.merged_weights
    assert fit.cost.trainable_parameters == model.total_parameters


@pytest.mark.parametrize("config", [lora(), lora(targets="all_linear"), sft()])
def test_training_lowers_dev_loss_and_leaves_input_model_untouched(model, splits, config):
    before = {k: v.clone() for k, v in model.tensors.items()}
    fit = bl.fit_baseline(model, config, dev=splits.dev)
    assert dev_loss(bl.merged_model(model, fit.merged_weights), splits.dev) < dev_loss(model, splits.dev)
    assert all(torch.equal(before[k], v) for k, v in model.tensors.items())


def test_sequence_lora_trains_full_response_tokens_and_preserves_input_model(model):
    examples = (("The capital of France is", " Paris"),)
    config = lora(steps=12, lr=0.1, rank=2)
    before = {name: tensor.clone() for name, tensor in model.tensors.items()}
    initial_loss = float(bl._sequence_ce(model, examples).item())
    fit = bl.fit_sequence_baseline(model, config, examples=examples)
    trained = bl.merged_model(model, fit.merged_weights)

    assert float(bl._sequence_ce(trained, examples).item()) < initial_loss
    assert fit.cost.trainable_parameters > 0
    assert fit.merged_weights
    assert all(torch.equal(before[name], tensor) for name, tensor in model.tensors.items())


def test_lora_merge_is_low_rank_and_limited_to_its_sites(model, splits):
    fit = bl.fit_baseline(model, lora(rank=2), dev=splits.dev)
    site = model.mlp_down_proj_site(1)
    assert set(fit.merged_weights) == {site}
    assert torch.linalg.matrix_rank(fit.merged_weights[site] - model.tensors[site]).item() == 2
    h, inter = model.tensors[site].shape
    assert fit.cost.trainable_parameters == 2 * (h + inter)
    assert fit.cost.changed_parameters == h * inter


def test_cost_is_recorded(model, splits):
    cost = bl.fit_baseline(model, lora(steps=3), dev=splits.dev).cost
    assert cost.optimizer_steps == 3
    assert cost.wall_seconds > 0
    assert cost.peak_rss_bytes and cost.peak_rss_bytes > 1_000_000
    assert cost.delta_l2_norm > 0
    assert cost.total_model_parameters == model.total_parameters


def test_merged_weights_score_identically_to_a_direct_edit_override(model, splits):
    fit = bl.fit_baseline(model, lora(), dev=splits.dev)
    site = model.mlp_down_proj_site(1)
    merged = bl.merged_model(model, fit.merged_weights)
    for prompt, target in splits.val:
        direct = model.evaluate_prompt(prompt, target_token=target, layer_down_proj_override={1: fit.merged_weights[site]})
        assert merged.evaluate_prompt(prompt, target_token=target) == direct


def test_same_seed_reproduces_and_checkpoint_equals_shorter_run(model, splits):
    seen = {}
    long = bl.fit_baseline(
        model,
        lora(steps=6, checkpoints=(3,)),
        dev=splits.dev,
        on_checkpoint=lambda step, weights, _wall: seen.setdefault(step, weights),
    )
    short = bl.fit_baseline(model, lora(steps=3), dev=splits.dev)
    assert short.weights_digest() == bl.weights_digest(seen[3])
    assert long.weights_digest() == bl.weights_digest(seen[6])
    assert long.weights_digest() == bl.fit_baseline(model, lora(steps=6), dev=splits.dev).weights_digest()


def test_retention_guard_reduces_drift_on_guarded_prompts(model, splits):
    def drift(weight):
        fit = bl.fit_baseline(
            model, sft(steps=40, retention_weight=weight), dev=splits.dev, retention_guard=splits.refit_guard
        )
        merged = bl.merged_model(model, fit.merged_weights)
        with torch.no_grad():
            return sum(
                float(torch.dist(bl._last_logits(merged, p), bl._last_logits(model, p)).item())
                for p in splits.refit_guard
            )

    assert drift(50.0) < drift(0.0)


def test_splits_keep_tuning_fold_out_of_training_and_reject_leaks(model, splits):
    assert set(splits.retention_guard).isdisjoint(splits.retention_tune)
    assert set(splits.refit_guard) >= set(PRACTICE)
    assert {p.split()[0] for p, _ in splits.dev} == {"Task:", "Rule:"}
    assert {p.split()[0] for p, _ in splits.val} == {"Note:"}
    assert {prompt for prompt, _ in splits.dev}.isdisjoint({prompt for prompt, _ in splits.val})
    assert splits.gym_id == GYM.gym_id
    assert splits.gym_digest.startswith("sha256:")
    assert splits.dataset_digest.startswith("sha256:")
    leaky = PRACTICE + (GYM.holdout_rules[0][0],)
    with pytest.raises(ValueError, match="leaked"):
        bl.build_splits(GYM, leaky)


def test_selection_uses_public_score_with_documented_tiebreaks():
    def s(steps, val, ret, loss):
        return bl.CheckpointScore(steps, val, loss, ret, 1.0, 1.0)

    a, b = lora(steps=100), lora(steps=100, lr=0.5)
    chosen = bl.select_best([(a, [s(50, 0.5, 1.0, 0.1), s(100, 1.0, 0.8, 2.0)]), (b, [s(25, 1.0, 0.8, 1.0)])])
    assert (chosen.config.lr, chosen.config.steps) == (0.5, 25)
    with pytest.raises(ValueError):
        bl.select_best([])


def test_holdout_answers_once_per_candidate(model, splits):
    fit = bl.fit_baseline(model, lora(steps=2), dev=splits.dev)
    holdout = bl.SealedHoldout(GYM, SEALED)
    result = holdout.evaluate_once("lora", model, fit.merged_weights)
    assert result["holdout_total"] == 6
    assert result["sealed_retention_total"] == 2
    with pytest.raises(RuntimeError, match="already used"):
        holdout.evaluate_once("lora", model, fit.merged_weights)


def test_no_op_scores_full_retention_and_refit_uses_all_public_prompts(model, splits):
    assert bl.retention(model, bl.merged_model(model, {}), splits.refit_guard) == 1.0
    calls = {}
    real = bl.fit_baseline

    def spy(m, config, *, dev, retention_guard=(), on_checkpoint=None):
        calls["guard"] = tuple(retention_guard)
        return real(m, config, dev=dev, retention_guard=retention_guard, on_checkpoint=on_checkpoint)

    bl.fit_baseline = spy
    try:
        scores, _ = bl.tune_trial(model, lora(steps=2), splits)
        assert calls["guard"] == splits.retention_guard
        selected = bl.select_best([(lora(steps=2), scores)])
        bl.refit_selected(model, selected, splits)
        assert calls["guard"] == splits.refit_guard
    finally:
        bl.fit_baseline = real
