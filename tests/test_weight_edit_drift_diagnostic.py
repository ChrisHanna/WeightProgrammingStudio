from __future__ import annotations

import pytest
import torch
import json
from pathlib import Path

from neural_foundry.backends.weight_edit.drift_diagnostic import STRENGTHS, compare_logits, measure_mode
from neural_foundry.backends.weight_edit.loader import WeightEditInputError
from test_weight_edit_backend import numerical_fixture
from neural_foundry.gym_service import _document_digest


def test_identical_logits_have_zero_noise_and_full_agreement():
    logits = [torch.tensor([1.0, 0.5, -1.0])]
    report = compare_logits(logits, [logits[0].clone()])
    assert report["max_absolute_logit_drift"] == 0
    assert report["max_kl_base_to_observed"] == 0
    assert report["top1_agreement"] == report["top5_set_agreement"] == 1


def test_float32_measurement_does_not_quantize_the_reported_drift():
    before = [torch.tensor([1.0, 0.0, -1.0])]
    after = [torch.tensor([1.001, 0.0, -1.0])]
    report = compare_logits(before, after)
    assert 0.0009 < report["max_absolute_logit_drift"] < 0.0011
    assert report["max_kl_base_to_observed"] > 0
    with pytest.raises(WeightEditInputError):
        compare_logits(before, [])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_same_path_noise_controls_and_sweep_restore_base(numerical_fixture, dtype):
    loaded, task, examples, recipe = numerical_fixture
    report = measure_mode(loaded.checkpoint_path, task.job, examples, recipe, dtype, "cpu")
    assert report["base_restored"] is True
    assert report["repeat_base"]["native_logits_cast_float32"]["max_absolute_logit_drift"] == 0
    assert report["after_calculation_without_edit"]["native_logits_cast_float32"]["max_absolute_logit_drift"] == 0
    assert [row["target_step"] for row in report["strength_sweep"]] == list(STRENGTHS)
    zero = report["strength_sweep"][0]
    assert zero["effective_change_norm"] == 0 and zero["changed_tensors"] == []
    assert zero["measurements"]["native_logits_cast_float32"]["max_absolute_logit_drift"] == 0
    if dtype == torch.bfloat16:
        assert "float32_lm_head_bf16_trunk" in report["repeat_base"]


def test_retained_real_diagnostic_has_zero_noise_without_policy_changes():
    path = Path(__file__).resolve().parents[1] / "docs" / "evidence" / "weight-edit-preservation-noise-sweep-v1.json"
    report = json.loads(path.read_bytes())
    assert report["evidence_digest"] == _document_digest({key: value for key, value in report.items() if key != "evidence_digest"})
    assert report["training_examples"] == 64 and report["preservation_cases"] == 59
    assert report["noise_floor_max_native_bf16"] == 0 and report["policy_v2_proposal_needed"] is False
    assert report["active_preservation_policy_changed"] is False and report["active_logit_drift_limit"] == 0.25
    assert report["holdout_accessed"] is False and report["candidate_published"] is False
    for mode in report["modes"]:
        assert mode["base_restored"]
        assert [row["target_step"] for row in mode["strength_sweep"]] == list(STRENGTHS)
        assert all(values["max_absolute_logit_drift"] == 0 for values in mode["repeat_base"].values())
        assert all(values["max_absolute_logit_drift"] == 0 for values in mode["after_calculation_without_edit"].values())
        assert all(values["max_absolute_logit_drift"] == 0 for values in mode["strength_sweep"][0]["measurements"].values())
    fp32 = next(mode for mode in report["modes"] if mode["model_precision"] == "float32")
    assert all(row["measurements"]["native_logits_cast_float32"]["top1_agreement"] == 1 for row in fp32["strength_sweep"])
    assert all(row["measurements"]["native_logits_cast_float32"]["top5_set_agreement"] == 1 for row in fp32["strength_sweep"])
    assert all(row["measurements"]["native_logits_cast_float32"]["max_absolute_logit_drift"] < 0.25 for row in fp32["strength_sweep"])