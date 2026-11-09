from __future__ import annotations

import pytest
import torch
import json
from pathlib import Path

from neural_foundry.backends.weight_edit.drift_diagnostic import (STRENGTHS, append_sweep_checkpoint, compare_logits,
    _aggregate_logprobs, _load_training_rosters, _validation_template_scores, expected_answer_logprob,
    _interactions_by_template, measure_mode, open_sweep_checkpoint, select_top_third, select_validation_screen,
    strength_one_stop_gate)
from neural_foundry.backends.weight_edit.demonstrations import prepare_demonstration
from neural_foundry.prompt_rendering import PromptRenderer
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


def test_expected_answer_logprob_scores_frozen_teacher_forced_response(numerical_fixture):
    loaded, task, examples, recipe = numerical_fixture
    demonstration = prepare_demonstration(PromptRenderer(loaded.tokenizer), examples[0]["interaction"])
    result = expected_answer_logprob(loaded, [examples[0]["interaction"]])
    assert result["cases"] == 1 and result["response_tokens"] == len(demonstration.positions)
    assert torch.isfinite(torch.tensor(result["mean_token_logprob"])) and result["mean_token_logprob"] < 0


def test_batched_expected_answer_logprob_preserves_case_roster(numerical_fixture):
    loaded, task, examples, recipe = numerical_fixture
    interactions = [example["interaction"] for example in examples]
    serial = expected_answer_logprob(loaded, interactions, batch_size=1)
    batched = expected_answer_logprob(loaded, interactions, batch_size=4)
    assert serial["case_logprobs"] == pytest.approx(batched["case_logprobs"], abs=1e-5)
    assert serial["response_tokens"] == batched["response_tokens"]


def test_sweep_checkpoint_appends_each_result_and_rejects_mismatched_resume(tmp_path):
    path = tmp_path / "sweep.jsonl"
    binding = {"run_digest": "sha256:" + "a" * 64, "request_digest": "sha256:" + "b" * 64}
    assert open_sweep_checkpoint(path, binding) == []
    record = {"run_digest": binding["run_digest"], "phase": "screen", "layer": 4, "strength": 0.5}
    append_sweep_checkpoint(path, record)
    assert open_sweep_checkpoint(path, binding) == [record]
    with pytest.raises(WeightEditInputError, match="binding"):
        open_sweep_checkpoint(path, {**binding, "request_digest": "sha256:" + "c" * 64})


def test_public_split_baseline_rebuilds_train_and_validation_without_holdout(tmp_path, numerical_fixture, monkeypatch):
    import json
    from neural_foundry.backends.weight_edit import drift_diagnostic
    from neural_foundry.gym_service import _wire_document
    from neural_foundry.prompt_rendering import PromptRenderer
    from neural_foundry.reference_json_extraction import TEMPLATES

    loaded, task, examples, recipe = numerical_fixture
    state_dir = tmp_path / "state"
    jobs = state_dir / "jobs"
    jobs.mkdir(parents=True)
    (jobs / f"{task.job.digest[7:]}.json").write_bytes(_wire_document(task.job.to_document()))
    request = {"kind": "nf_gym_training_request_v1", "job_digest": task.job.digest,
        "loaded_base": {"state_digest": loaded.base_digest},
        "base_reference": {"artifact_path": str(loaded.checkpoint_path), "model_digest": loaded.base_digest},
        "inference_config": {"prompt_renderer_digest": PromptRenderer(loaded.tokenizer).digest},
        "recipe": {"seed": 17}}
    request["training_request_digest"] = _document_digest(request)
    request_path = tmp_path / "request.json"
    request_path.write_bytes(_wire_document(request))
    monkeypatch.setattr(drift_diagnostic, "load_model", lambda *args, **kwargs: loaded)
    monkeypatch.setattr(drift_diagnostic, "score_public_rows", lambda task, model, rows, split="validation": {
        "split": split, "cases": len(rows), "exact_match": 0.5, "format_valid": 1.0,
        "panel_digest": "sha256:" + "a" * 64,
        "responses": [{"scenario_id": row["scenario_id"], "exact_match": float(index % 2 == 0)}
                      for index, row in enumerate(rows)]})
    monkeypatch.setattr(drift_diagnostic, "expected_answer_logprob", lambda *args, **kwargs: {
        "cases": 2, "response_tokens": 2, "mean_case_logprob": -1.0,
        "mean_token_logprob": -1.0, "sum_logprob": -2.0, "case_logprobs": [-1.0, -1.0]})
    report_path = tmp_path / "baseline.json"
    report = drift_diagnostic.run_public_split_baseline(state_dir, request_path, report_path, device="cpu")
    assert report["train"]["score"]["cases"] == len(TEMPLATES["train"])
    assert report["validation"]["score"]["cases"] == len(TEMPLATES["validation"])
    assert report["train"]["failure_count"] > 0 and report["validation"]["failure_count"] > 0
    assert report["holdout_accessed"] is False and report["release_authorized"] is False
    assert json.loads(report_path.read_bytes()) == report


def test_recorded_expanded_base_baseline_is_digest_bound_and_has_minimum_failures():
    import hashlib
    import json

    root = Path(__file__).resolve().parents[1] / "docs" / "evidence"
    path = root / "weight-edit-expanded-baseline-qwen3-train-validation-v1.json"
    report = json.loads(path.read_bytes())
    assert report["evidence_digest"] == _document_digest({key: value for key, value in report.items() if key != "evidence_digest"})
    assert report["train_template_count"] == 10 and report["train"]["score"]["cases"] == 320
    assert report["train"]["failure_count"] == 155 and report["train"]["failure_count"] >= 10
    assert report["validation"]["score"]["cases"] == 64 and report["validation"]["failure_count"] == 18
    assert report["holdout_accessed"] is False
    summary = json.loads((root / "weight-edit-expanded-baseline-qwen3-summary.json").read_bytes())
    for record in summary["raw_runs"]:
        raw_path = Path(__file__).resolve().parents[1] / record["path"]
        assert hashlib.sha256(raw_path.read_bytes()).hexdigest().upper() == record["sha256_file"]


def test_successive_halving_selects_top_third_by_public_validation_likelihood():
    rows = [{"checkpoint_key": f"candidate-{index}", "status": "measured", "layer": index % 4,
             "target_step": (index % 4) + 0.01, "training_logprob_gain": 0.0,
             "validation_screen_logprob_gain": float(index)} for index in range(9)]
    rows.append({"checkpoint_key": "failed", "status": "failed", "layer": 0, "target_step": 1.0})
    assert select_top_third(rows) == ["candidate-8", "candidate-7", "candidate-6"]


def test_validation_screen_selects_eight_rows_per_family():
    rows = [{"family_id": family, "scenario_id": f"{family}:{index}"}
            for family in ("invoice_a", "invoice_b") for index in range(10)]
    selected = select_validation_screen(rows)
    assert len(selected) == 16
    assert [row["scenario_id"] for row in selected] == [
        *(f"invoice_a:{index}" for index in range(8)),
        *(f"invoice_b:{index}" for index in range(8)),
    ]


def test_validation_interactions_are_grouped_for_each_template():
    from neural_foundry.reference_json_extraction import generate_scenarios, reference_task

    task = reference_task(rows_per_template=1)
    rows = generate_scenarios(split="validation", seed=17, parameters={"rows_per_template": 1})
    grouped = _interactions_by_template(task, rows)
    assert set(grouped) == {row["family_id"] for row in rows}
    assert sum(map(len, grouped.values())) == len(rows)
    assert all(grouped[row["family_id"]] for row in rows)


def test_validation_results_and_logprobs_are_reported_by_template():
    rows = [{"scenario_id": "a1", "family_id": "invoice_a"},
            {"scenario_id": "a2", "family_id": "invoice_a"},
            {"scenario_id": "b1", "family_id": "invoice_b"}]
    score = {"responses": [
        {"scenario_id": "a1", "exact_match": 1.0, "format_valid": True},
        {"scenario_id": "a2", "exact_match": 0.0, "format_valid": True},
        {"scenario_id": "b1", "exact_match": 1.0, "format_valid": False},
    ]}
    by_template = _validation_template_scores(rows, score)
    assert by_template["invoice_a"] == {"cases": 2, "exact_match": 0.5, "exact_match_count": 1,
                                        "format_valid": 1.0, "format_valid_count": 2}
    assert by_template["invoice_b"]["format_valid"] == 0.0
    family_logprobs = {family: {"cases": 1, "response_tokens": 2, "sum_logprob": -2.0,
                                "case_logprobs": [-2.0]} for family in ("invoice_a", "invoice_b")}
    assert _aggregate_logprobs(family_logprobs)["mean_token_logprob"] == -1.0


def test_strength_one_stop_gate_requires_effective_change_and_task_gain():
    passing = {"status": "measured", "layer": 14, "effective_change_norm": 0.1,
               "validation_exact_match_gain": 1 / 64}
    assert strength_one_stop_gate(14, passing)["continue_to_multilayer"] is True
    no_gain = {**passing, "validation_exact_match_gain": 0.0}
    assert strength_one_stop_gate(14, no_gain)["continue_to_multilayer"] is False
    no_change = {**passing, "effective_change_norm": 0.0}
    assert strength_one_stop_gate(14, no_change)["continue_to_multilayer"] is False
    assert strength_one_stop_gate(13, passing)["continue_to_multilayer"] is False


def test_training_rosters_keep_full_parent_for_scoring_and_selected_subset_for_edits():
    parent = [{"scenario_id": f"train:{index}", "input": {"id": index}, "response": {"id": index}}
              for index in range(4)]
    selected = [parent[1], parent[3]]
    filtered_digest, parent_digest = "sha256:" + "a" * 64, "sha256:" + "b" * 64

    class Service:
        def load_training_examples(self, digest):
            return {filtered_digest: selected, parent_digest: parent}[digest]

    selection = {"filtered_dataset_digest": filtered_digest, "parent_dataset_digest": parent_digest,
                 "selected_scenario_ids": [row["scenario_id"] for row in selected],
                 "selected_example_digests": [_document_digest(row) for row in selected]}
    request = {"dataset_digest": filtered_digest, "recipe": {"failure_selection": selection}}
    edit_examples, scoring_examples = _load_training_rosters(Service(), request)
    assert edit_examples == selected
    assert scoring_examples == parent
    bad_request = {**request, "recipe": {"failure_selection": {
        **selection, "selected_example_digests": ["sha256:" + "c" * 64, selection["selected_example_digests"][1]]}}}
    with pytest.raises(WeightEditInputError, match="digest-bound parent"):
        _load_training_rosters(Service(), bad_request)


def test_layer_specific_calculation_reuses_verified_site_map(numerical_fixture):
    from neural_foundry.backends.weight_edit.discover import find_blocks, find_sites
    from neural_foundry.backends.weight_edit.worker import calculate_edit

    loaded, task, examples, recipe = numerical_fixture
    site_map = find_sites(loaded.model, find_blocks(loaded.model), loaded.hidden_size,
                          checkpoint_tensor_names=loaded.checkpoint_tensor_names)
    _, selected, delta, evidence = calculate_edit(loaded, task.job, examples, recipe, site_layer=1,
                                                  discovered_sites=site_map)
    assert selected.layer == 1 and site_map.checkpoint_names_verified
    assert delta.U is not None and delta.V is not None and evidence["gradient_inspection_used"]


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