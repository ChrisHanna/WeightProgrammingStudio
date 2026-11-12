from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
import torch
from neural_foundry.backends.weight_edit.adapter import validate_recipe
from neural_foundry.backends.weight_edit.capture import (
    CovarianceStatistics,
    capture,
    capture_statistics,
)
from neural_foundry.backends.weight_edit.discover import Site, SiteMap
from neural_foundry.backends.weight_edit.loader import WeightEditInputError
from neural_foundry.backends.weight_edit.methods.base import LegacyKeyValueEdit
from neural_foundry.backends.weight_edit.methods.experimental import (
    METHOD_ID,
    ValueScaledRetainedEdit,
)
from neural_foundry.backends.weight_edit.value_objective import prepare_value_demonstration
from neural_foundry.backends.weight_edit.worker import calculate_edit, train_payload
from neural_foundry.prompt_rendering import PromptRenderer


def settings(**overrides):
    return {"strength": .1, "ridge_alpha": .2, "retention_beta": .4, "protection_epsilon": 1.,
            "protection_zero_absolute": 10., "rank_rtol": 1e-12, "rank_atol": 0.,
            "solver_atol": 1e-10, "solver_max_iterations": 100} | overrides


def site():
    return Site("projection.weight", 0, "residual_write_mlp", 2, 2, False, "torch.float64")


def statistics(rows):
    rows = torch.tensor(rows, dtype=torch.float64)
    return CovarianceStatistics(len(rows), rows.sum(0), torch.zeros(2, dtype=torch.float64),
                                rows.T @ rows, torch.zeros((2, 2), dtype=torch.float64))


def solve(*, keys=None, gradients=None, outputs=None, W=None, P=None, stat=None, **options):
    identity = torch.eye(2, dtype=torch.float64)
    method = ValueScaledRetainedEdit(stat or statistics([[1., 0.], [0., 1.]]),
        identity if P is None else P, identity if outputs is None else outputs)
    delta = method.compute(site(), identity if W is None else W, identity if keys is None else keys,
                           identity if gradients is None else gradients, settings(**options))
    return delta.matrix(), method.diagnostics


def test_raw_gradient_ratio_and_single_global_scale_with_zero_columns():
    gradient = torch.tensor([[1., 0.], [0., 4.]], dtype=torch.float64)
    D, report = solve(gradients=gradient, retention_beta=0)
    assert D[1, 1] / D[0, 0] == pytest.approx(4.)
    assert report["target_rms"] == pytest.approx(.1 * report["activation_rms"])
    D2, _ = solve(gradients=gradient * 17, retention_beta=0)
    torch.testing.assert_close(D, D2)
    D3, _ = solve(gradients=gradient, outputs=torch.eye(2, dtype=torch.float64) * 3, retention_beta=0)
    torch.testing.assert_close(D3, D * 3)
    _, report = solve(gradients=torch.tensor([[0., 0.], [0., 2.]], dtype=torch.float64))
    assert report["zero_gradient_keys"] == 1


def test_ridge_uses_input_dimension_diagonal_and_rescales_with_keys():
    keys = torch.tensor([[1., 0.], [0., 2.], [1., 0.]], dtype=torch.float64)
    gradient = torch.tensor([[1., 0.], [0., 2.], [1., 0.]], dtype=torch.float64)
    D, report = solve(keys=keys, gradients=gradient, outputs=gradient, retention_beta=0)
    D2, report2 = solve(keys=keys * 3, gradients=gradient, outputs=gradient, retention_beta=0)
    assert report["mean_gram_diagonal"] == 3.
    assert report["ridge"] == pytest.approx(.6)
    assert report2["ridge"] == pytest.approx(9 * report["ridge"])
    torch.testing.assert_close(D2, D / 3)


@pytest.mark.parametrize("argument", ["keys", "gradients", "outputs"])
def test_zero_global_scale_refused(argument):
    with pytest.raises(WeightEditInputError, match="scales"):
        solve(**{argument: torch.zeros((2, 2), dtype=torch.float64)})


def test_nonfinite_and_missing_calibration_refused():
    with pytest.raises(WeightEditInputError, match="finite"):
        solve(gradients=torch.full((2, 2), float("nan"), dtype=torch.float64))
    with pytest.raises(WeightEditInputError, match="ridge_alpha"):
        solve(ridge_alpha=0)
    recipe = experimental_recipe()
    del recipe["experimental_settings"]["strength"]
    with pytest.raises(WeightEditInputError, match="explicit calibration"):
        validate_recipe(recipe)


def test_rank_deficient_exact_protection_and_bf16_reload():
    P = torch.tensor([[1., 0.], [2., 0.], [0., 0.]], dtype=torch.float64)
    D, report = solve(P=P, protection_epsilon=0, protection_zero_absolute=0)
    assert report["protected_rank"] == 1
    assert torch.count_nonzero(D[:, 0]) == 0
    assert torch.count_nonzero(D[:, 1]) > 0
    assert report["tensor_protected_residual"] == report["bf16_protected_residual"] == 0.
    base = torch.eye(2, dtype=torch.float64)
    buffer = io.BytesIO()
    torch.save((base + D).bfloat16(), buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=True).double()
    torch.testing.assert_close((restored - base.bfloat16().double()) @ P.T, torch.zeros((2, 3), dtype=torch.float64))
    repeated, _ = solve(P=P, protection_epsilon=0, protection_zero_absolute=0)
    assert torch.equal(D, repeated)


def test_positive_constraint_is_enforced_and_zero_reference_is_explicit():
    # A nonbinding relative bound and an active relative bound are distinct.
    D, _ = solve(protection_epsilon=.02, solver_atol=.003)
    assert float(torch.linalg.vector_norm(D)) <= .02 * 2 ** .5
    _, report = solve(W=torch.zeros((2, 2), dtype=torch.float64), protection_zero_absolute=0)
    assert report["protected_reference_norm"] == report["protected_bound"] == 0.
    assert report["tensor_delta_norm"] == 0.
    assert report["zero_effective_bf16_update"]


def test_positive_constraint_solves_quadratic_kkt_not_posthoc_projection():
    keys = torch.tensor([[2., 1.], [0., 1.]], dtype=torch.float64)
    P = torch.tensor([[1., 0.]], dtype=torch.float64)
    D, report = solve(keys=keys, P=P, protection_epsilon=.02, solver_atol=.003)
    K = keys.T
    A = K @ K.T + .4 * torch.eye(2, dtype=torch.float64) / 2 + report["ridge"] * torch.eye(2, dtype=torch.float64)
    Y = -report["global_target_scale"] * torch.eye(2, dtype=torch.float64)
    B = Y @ K.T
    residual = B - D @ A
    # KKT: B-D A=lambda D P P.T. Unconstrained coordinate is stationary.
    torch.testing.assert_close(residual[:, 1], torch.zeros(2, dtype=torch.float64), atol=1e-12, rtol=0)
    multiplier = residual[:, 0] / D[:, 0]
    assert multiplier[0] == pytest.approx(float(multiplier[1]), rel=1e-10)
    assert multiplier[0] > 0
    assert float(torch.linalg.vector_norm(D @ P.T)) == pytest.approx(report["protected_bound"])


def test_truncated_protected_direction_cannot_be_silently_discarded():
    P = torch.tensor([[1., 0.], [0., 1e-3]], dtype=torch.float64)
    with pytest.raises(WeightEditInputError, match="protected constraint"):
        solve(P=P, protection_epsilon=0, rank_atol=.01, solver_atol=1e-12)
    with pytest.raises(WeightEditInputError, match="multiplier search"):
        solve(protection_epsilon=1e-12, solver_max_iterations=1)
    with pytest.raises(WeightEditInputError, match="did not converge"):
        solve(protection_epsilon=.06, solver_max_iterations=1)


def test_cast_protection_cannot_be_silently_dropped():
    P = torch.tensor([[1., 1.]], dtype=torch.float64)
    W = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
    with pytest.raises(WeightEditInputError, match="bf16 protected"):
        solve(P=P, W=W, protection_epsilon=0)


class Tokenizer:
    chat_template = "fixture"
    eos_token_id = 127

    def __init__(self):
        self.special_tokens_map = {}

    def get_vocab(self):
        return {chr(index): index for index in range(128)}

    def encode(self, text, **kwargs):
        return [ord(value) for value in text]

    def apply_chat_template(self, messages, **kwargs):
        return "U>"

    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text), "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def interaction():
    return {"format": "json_extraction_v1", "messages": [
        {"role": "user", "content": "fixture"}, {"role": "assistant", "content": '{"count":12,"item":"AB"}'}]}


def test_value_mask_causal_alignment_excludes_properties_syntax_and_eos():
    renderer = PromptRenderer(Tokenizer())
    item, groups = prepare_value_demonstration(renderer, interaction(), value_fields=["count", "item"])
    assert groups == ["count", "count", "item", "item"]
    assert [item.labels[position + 1] for position in item.positions] == list(map(ord, "12AB"))
    assert item.labels[-1] == -100
    assert all(item.labels[index] == -100 for index in range(len(item.labels))
               if index not in [position + 1 for position in item.positions])
    task, _ = prepare_value_demonstration(renderer, interaction(), spans=[(9, 11, "amount")])
    assert [task.labels[position + 1] for position in task.positions] == list(map(ord, "12"))


def test_mixed_empty_and_incomplete_value_boundaries_refused():
    class Mixed(Tokenizer):
        def encode(self, text, **kwargs):
            return [1] if text.startswith("{") else super().encode(text, **kwargs)

        def __call__(self, text, **kwargs):
            return {"input_ids": [1], "offset_mapping": [(0, len(text))]}

    with pytest.raises(WeightEditInputError, match="mixed"):
        prepare_value_demonstration(PromptRenderer(Mixed()), interaction(), value_fields=["count"])
    empty = interaction()
    empty["messages"][-1]["content"] = '{"item":""}'
    with pytest.raises(WeightEditInputError, match="nonempty"):
        prepare_value_demonstration(PromptRenderer(Tokenizer()), empty, value_fields=["item"])

    class Missing(Tokenizer):
        def __call__(self, text, **kwargs):
            result = super().__call__(text, **kwargs)
            result["offset_mapping"][9] = (8, 9)
            return result

    with pytest.raises(WeightEditInputError, match="coverage"):
        prepare_value_demonstration(PromptRenderer(Missing()), interaction(), value_fields=["count"])


def toy_loaded():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(2, 2, bias=False, dtype=torch.float64)
            with torch.no_grad():
                self.projection.weight.copy_(torch.eye(2, dtype=torch.float64))

        def forward(self, input_ids, attention_mask, labels=None):
            ids = input_ids.double()
            values = torch.stack((ids / 128, torch.ones_like(ids)), dim=-1)
            result = self.projection(values)
            # Toy causal logits only; no checkpoint/tokenizer/model loading.
            logits = torch.cat((result, torch.zeros((*result.shape[:-1], 126), dtype=torch.float64)), dim=-1)
            return SimpleNamespace(logits=logits)

    return SimpleNamespace(model=Model().eval(), tokenizer=Tokenizer())


def experimental_recipe():
    return {"method": "weight_edit", "edit_method": METHOD_ID, "experimental_opt_in": True,
            "experimental_settings": settings(protection_epsilon=10., solver_atol=.01),
            "max_sequence_tokens": 128, "max_training_tokens": 1000, "max_keys": 128,
            "retention_interactions": [{"format": "raw_text_v1", "text": "abc"}], "retention_source": "public_train",
            "protected_training_examples": [{"interaction": interaction(), "already_passing": True}],
            "protected_training_source": "public_train", "value_fields": ["count", "item"]}


def test_all_token_retention_differs_from_last_key_and_is_token_weighted():
    loaded = toy_loaded()
    sites = SiteMap((site(),), checkpoint_names_verified=True)
    prompts = [{"format": "raw_text_v1", "text": text} for text in ("abc", "d")]
    full = capture_statistics(loaded, sites, prompts, positions=[[0, 1, 2], [0]])[site().tensor_name]
    last = capture_statistics(loaded, sites, prompts)[site().tensor_name]
    assert full.count == 4 and last.count == 2
    rows = capture(loaded, sites, prompts, positions=[[0, 1, 2], [0]], output_dtype=torch.float64)[site().tensor_name].inputs
    torch.testing.assert_close(full.input_gram / full.count, rows.T @ rows / 4)
    assert not torch.allclose(full.input_gram / 4, last.input_gram / 2)
    full_delta, _ = solve(stat=full)
    last_delta, _ = solve(stat=last)
    assert not torch.allclose(full_delta, last_delta)


def test_zero_rank_protection_does_not_invent_an_undefined_reference():
    D, report = solve(P=torch.zeros((3, 2), dtype=torch.float64), protection_zero_absolute=0)
    assert report["protected_rank"] == 0
    assert report["protected_reference_norm"] == report["tensor_protected_residual"] == 0.
    assert torch.count_nonzero(D) > 0


def test_first_response_value_prediction_key_not_duplicated_in_protection():
    loaded = toy_loaded()
    selected = SiteMap((site(),), checkpoint_names_verified=True)
    recipe = experimental_recipe()
    plain = {"format": "chat_messages_v1", "messages": [
        {"role": "user", "content": "fixture"}, {"role": "assistant", "content": "AB"}]}
    example = {"interaction": plain, "value_spans": [[0, 2, "value"]]}
    recipe["value_fields"] = None
    recipe["protected_training_examples"] = [dict(example, already_passing=True)]
    _, _, _, evidence = calculate_edit(loaded, None, [example], recipe, discovered_sites=selected)
    assert evidence["protected_prompt_tokens"] == 2
    assert evidence["protected_value_prediction_keys"] == 2
    assert evidence["protected_value_prediction_keys_not_already_in_prompt"] == 1
    assert evidence["key_diagnostics"]["protected_key_count"] == 3


def test_capture_global_value_mean_and_actual_worker_dispatch():
    loaded = toy_loaded()
    selected = SiteMap((site(),), checkpoint_names_verified=True)
    item, _ = prepare_value_demonstration(PromptRenderer(loaded.tokenizer), interaction(), value_fields=["count", "item"])
    first = capture(loaded, selected, [item.rendered], positions=[item.positions], loss_labels=[item.labels],
                    causal_value_objective=True, loss_denominator=4, output_dtype=torch.float64)[site().tensor_name]
    second = capture(loaded, selected, [item.rendered], positions=[item.positions], loss_labels=[item.labels],
                     causal_value_objective=True, loss_denominator=8, output_dtype=torch.float64)[site().tensor_name]
    torch.testing.assert_close(first.output_gradients, second.output_gradients * 2)
    with pytest.raises(WeightEditInputError, match="causal"):
        capture(loaded, selected, [item.rendered], positions=[[0]], loss_labels=[item.labels],
                causal_value_objective=True, loss_denominator=4)
    _, _, delta, evidence = calculate_edit(loaded, None, [{"interaction": interaction()}],
                                          experimental_recipe(), discovered_sites=selected)
    assert delta.kind == "dense"
    assert evidence["key_diagnostics"]["retention_token_count"] == 5  # abc plus protected U>
    assert evidence["protected_value_prediction_keys"] == 4
    assert evidence["key_diagnostics"]["loss_reduction"] == "global_mean_over_selected_value_tokens"
    assert evidence["value_token_groups"]["count"]["value_token_count"] == 2
    assert not loaded.model.projection._forward_hooks


@pytest.mark.parametrize("budget", ["max_training_tokens", "max_keys"])
def test_capture_budget_counts_protected_prompt_recapture_before_any_forward(monkeypatch, budget):
    loaded = toy_loaded()
    selected = SiteMap((site(),), checkpoint_names_verified=True)
    recipe = experimental_recipe()
    full_tokens = len(interaction()["messages"][-1]["content"]) + 2 + 1
    old_count = 5 + 2 * full_tokens if budget == "max_training_tokens" else 5 + 4 + 4
    # The protected two-token prompt is forwarded/captured a second time.
    recipe[budget] = old_count + 1
    forward_calls = []

    def forbidden_forward(**kwargs):
        forward_calls.append(kwargs)
        pytest.fail("budget must refuse before any model forward")

    monkeypatch.setattr(loaded.model, "forward", forbidden_forward)
    with pytest.raises(WeightEditInputError, match="token budgets" if budget == "max_training_tokens" else "key budget"):
        calculate_edit(loaded, None, [{"interaction": interaction()}], recipe, discovered_sites=selected)
    assert not forward_calls


def test_capture_budget_reports_all_actual_passes_and_accepts_exact_limits(monkeypatch):
    loaded = toy_loaded()
    selected = SiteMap((site(),), checkpoint_names_verified=True)
    recipe = experimental_recipe()
    full_tokens = len(interaction()["messages"][-1]["content"]) + 2 + 1
    recipe["max_training_tokens"] = 7 + 2 * full_tokens
    recipe["max_keys"] = 15
    forward_tokens = []
    original = loaded.model.forward

    def counted_forward(**kwargs):
        forward_tokens.append(kwargs["input_ids"].numel())
        return original(**kwargs)

    monkeypatch.setattr(loaded.model, "forward", counted_forward)
    _, _, _, evidence = calculate_edit(loaded, None, [{"interaction": interaction()}],
                                       recipe, discovered_sites=selected)
    budget = evidence["capture_budget"]
    assert budget["totals"] == {
        "forward_pass_count": len(forward_tokens),
        "forward_tokens": sum(forward_tokens),
        "captured_key_positions": 15,
    }
    assert budget["totals"]["forward_tokens"] == recipe["max_training_tokens"]
    assert budget["stages"]["protected_prompt_recapture"] == {
        "forward_pass_count": 1, "forward_tokens": 2, "captured_key_positions": 2,
    }


def test_live_execution_and_missing_optin_refused_before_any_load():
    with pytest.raises(Exception, match="CPU-prototype only"):
        train_payload({"request": {"recipe": experimental_recipe()}})
    recipe = experimental_recipe()
    recipe["experimental_opt_in"] = False
    with pytest.raises(WeightEditInputError, match="opt-in"):
        validate_recipe(recipe)


def test_legacy_fixed_ridge_and_absolute_targets_unchanged():
    W = torch.eye(2, dtype=torch.float64)
    keys = W.clone()
    targets = W * 2
    result = LegacyKeyValueEdit(W).compute(site(), W, keys, targets, {"retention_nullspace_weight": 0.})
    torch.testing.assert_close(result.matrix(), W / 1.001)
