from __future__ import annotations

import json
from pathlib import Path

import torch
import pytest

from neural_foundry.backends.weight_edit.preservation import POLICY_V2, calibrate_one_ulp, float32_logits_of_exact_weights, metrics, verify_v2
from neural_foundry.backends.weight_edit.discover import find_blocks, find_sites
from neural_foundry.backends.weight_edit.loader import load_model
from neural_foundry.model_digest import model_state_digest
from test_weight_edit_backend import numerical_fixture
from neural_foundry.gym_service import _document_digest


def _calibration():
    control = {"cases": 16, "max_absolute_logit_drift": 0.0, "mean_kl_base_to_edited": 0.0,
               "top1_agreement": 1.0, "top5_set_agreement": 1.0}
    report = {"kind": "nf_bf16_one_ulp_sensitivity_v2", "policy_digest": _document_digest(POLICY_V2),
              "base_digest": "sha256:" + "1" * 64, "panel_digest": "sha256:" + "2" * 64,
              "renderer_digest": "sha256:" + "3" * 64, "base_restored": True, "precision": "bfloat16",
              "flat_index": 0, "old_bits": 16256, "new_bits": 16257, "adjacent_value_verified": True,
              "tensor_name": "block.weight", "changed_scalar_count": 1,
              "runtime": {"torch": "2.9.1", "transformers": "4.57.3"},
              "measurements": control, "float32_measurements": control}
    return {**report, "calibration_digest": _document_digest(report)}


def test_v2_reports_bf16_max_drift_but_gates_other_metrics():
    precise = {"cases": 16, "max_absolute_logit_drift": 0.01}
    serving = {"cases": 16, "max_absolute_logit_drift": 0.5625, "top1_agreement": 1.0,
               "top5_set_agreement": 1.0, "mean_kl_base_to_edited": 1e-5}
    assert verify_v2(precise, serving, _calibration())["passed"]
    assert not verify_v2(precise, {**serving, "top5_set_agreement": 0.94}, _calibration())["passed"]
    assert not verify_v2({**precise, "max_absolute_logit_drift": 0.3}, serving, _calibration())["passed"]


def test_frozen_policy_matches_published_thresholds_and_mean_kl():
    published = json.loads((Path(__file__).resolve().parents[1] / "protocols" / "preservation-policy-v2.json").read_bytes())
    assert POLICY_V2 == {key: published[key] for key in POLICY_V2}
    assert published["status"] == "frozen"
    before = [torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])]
    after = [torch.tensor([1.01, 0.0]), before[1]]
    report = metrics(before, after)
    assert report["mean_kl_base_to_edited"] == metrics(before[:1], after[:1])["mean_kl_base_to_edited"] / 2


def test_one_ulp_control_and_float32_promotion_restore_exact_bf16_weights(numerical_fixture):
    base, task, examples, recipe = numerical_fixture
    loaded = load_model(base.checkpoint_path, dtype=torch.bfloat16, device="cpu")
    sites = find_sites(loaded.model, find_blocks(loaded.model), loaded.hidden_size, checkpoint_tensor_names=loaded.checkpoint_tensor_names)
    site = next(site for site in sites.sites if site.role == "residual_write_mlp")
    digest = model_state_digest(loaded.model)
    record = calibrate_one_ulp(loaded, site, recipe["preservation_interactions"])
    assert abs(record["new_bits"] - record["old_bits"]) == 1 and record["base_restored"]
    assert model_state_digest(loaded.model) == digest
    values = float32_logits_of_exact_weights(loaded, recipe["preservation_interactions"])
    assert values[0].dtype == torch.float32 and model_state_digest(loaded.model) == digest
    assert next(loaded.model.parameters()).dtype == torch.bfloat16


def test_rejects_digest_only_or_incomplete_calibration():
    calibration = _calibration()
    del calibration["float32_measurements"]
    calibration["calibration_digest"] = _document_digest({key: value for key, value in calibration.items() if key != "calibration_digest"})
    with pytest.raises(ValueError, match="complete"):
        verify_v2({"cases": 16}, {"cases": 16}, calibration)


def test_measured_qwen3_evidence_remains_rejected_by_frozen_top5_gate():
    path = Path(__file__).resolve().parents[1] / "docs" / "evidence" / "weight-edit-preservation-v2-qwen3.json"
    report = json.loads(path.read_bytes())
    verification = report["verification"]
    assert verify_v2(verification["float32"], verification["bf16"], report["calibration"]) == verification
    assert verification["passed"] is False
    assert verification["bf16"]["top5_set_agreement"] == 56 / 59
    assert verification["float32"]["max_absolute_logit_drift"] <= POLICY_V2["max_float32_logit_drift"]
    assert verification["bf16"]["top1_agreement"] == 1.0
    assert verification["bf16"]["mean_kl_base_to_edited"] <= POLICY_V2["max_mean_bf16_kl_base_to_edited"]
    assert report["candidate_published"] is False and report["holdout_evaluated"] is False
    assert report["calibration"]["measurements"]["max_absolute_logit_drift"] == 0.0


def test_promotion_restores_buffers_and_weights_when_forward_fails(numerical_fixture, monkeypatch):
    from neural_foundry.backends.weight_edit import worker

    base, task, examples, recipe = numerical_fixture
    loaded = load_model(base.checkpoint_path, dtype=torch.bfloat16, device="cpu")
    loaded.model.register_buffer("precision_test", torch.tensor([0.123456789], dtype=torch.float32))
    parameter = next(loaded.model.parameters())
    loaded.model.register_parameter("shared_precision_test", torch.nn.Parameter(parameter.detach()))
    storage = parameter.untyped_storage().data_ptr()
    original = loaded.model.precision_test
    digest = model_state_digest(loaded.model)

    def fail(*args):
        raise RuntimeError("measurement failed")

    monkeypatch.setattr(worker, "preservation_logits", fail)
    with pytest.raises(RuntimeError, match="measurement failed"):
        float32_logits_of_exact_weights(loaded, recipe["preservation_interactions"])
    assert loaded.model.precision_test is original
    assert parameter.untyped_storage().data_ptr() == storage
    assert loaded.model.shared_precision_test.untyped_storage().data_ptr() == storage
    assert model_state_digest(loaded.model) == digest