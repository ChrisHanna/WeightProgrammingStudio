from __future__ import annotations

import pytest
import torch
import json
import sys
from pathlib import Path

from neural_foundry.backends.weight_edit.drift_diagnostic import (STRENGTHS, append_sweep_checkpoint, compare_logits,
    _aggregate_logprobs, _load_training_rosters, _validation_template_scores, expected_answer_logprob,
    _StopAfterJsonRoot, _completion_was_truncated, _interactions_by_template, _mlp_writer_site_map,
    audit_generation_truncation, load_sweep_likelihood_selection, measure_mode, open_sweep_checkpoint,
    select_top_third, select_validation_screen, strength_one_stop_gate)
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


def test_generation_truncation_counts_only_cap_hits():
    complete_at_budget = '{"customer":"Aster"}'
    complete_with_preamble = 'Here is the result: ```json\n{"customer":"Aster"}'
    truncated_at_budget = '{"customer":"Aster"'
    assert _completion_was_truncated([1, 2, 3, 4], complete_at_budget, 4) is True
    assert _completion_was_truncated([1, 2, 3, 4], complete_with_preamble, 4) is True
    assert _completion_was_truncated([1, 2, 3, 4], truncated_at_budget, 4) is True
    assert _completion_was_truncated([1, 2, 3, 4], '[{"customer":"Aster"}', 4) is True
    assert _completion_was_truncated([1, 2, 3], truncated_at_budget, 4) is False


def test_json_generation_stops_after_root_not_nested_closing_brace():
    class Tokenizer:
        parts = {1: '{"outer":', 2: '{"inner":1}', 3: '}', 4: ''}

        def decode(self, token_ids, *, skip_special_tokens=True):
            return "".join(self.parts[token_id] for token_id in token_ids)

    assert _completion_was_truncated([9, 9], "", 128) is False

    stop = _StopAfterJsonRoot(Tokenizer(), prompt_width=1, eos_token_id=9, batch_size=1)
    empty = stop(torch.tensor([[0, 4]]), torch.zeros((1, 10)))
    assert stop.completed == [False]
    assert torch.isfinite(empty).all()
    nested_close = stop(torch.tensor([[0, 1, 2]]), torch.zeros((1, 10)))
    assert stop.completed == [False]
    assert torch.isfinite(nested_close).all()
    root_close = stop(torch.tensor([[0, 1, 2, 3]]), torch.zeros((1, 10)))
    assert stop.completed == [True]
    assert root_close.argmax(dim=-1).tolist() == [9]
    assert torch.isneginf(root_close[0, :9]).all()


def test_truncation_audit_reports_count_and_binds_source_digest(tmp_path):
    response_rows = [{"scenario_id": "case:complete", "execution_evidence": {
        "generated_token_ids": list(range(4)), "raw_text": '{"answer":1}'}},
        {"scenario_id": "case:truncated", "execution_evidence": {
        "generated_token_ids": list(range(4)), "raw_text": '{"answer":'}},
        {"scenario_id": "case:short-truncated", "execution_evidence": {
        "generated_token_ids": list(range(3)), "raw_text": '{"answer":'}}]
    score = {"cases": 3, "responses": response_rows}
    report = {"kind": "nf_weight_edit_layer_strength_diagnostic_v2", "run_digest": "sha256:" + "a" * 64,
        "base_train": {"task": score}, "base_validation": {"task": score}, "results": []}
    report["evidence_digest"] = _document_digest(report)
    source = tmp_path / "sweep.json"
    source.write_bytes(json.dumps(report).encode())
    audit_path = tmp_path / "truncation-audit.json"
    audit = audit_generation_truncation(source, audit_path, max_new_tokens=4)
    assert audit["total_scored_answers"] == 6
    assert audit["generation_cap_failures"] == 4
    assert audit["total_truncated_answers"] == 4
    assert audit["zero_truncated_answers"] is False
    assert audit["source_report_digest"] == report["evidence_digest"]
    assert json.loads(audit_path.read_bytes()) == audit


def test_finalist_rescore_loads_only_complete_digest_bound_screen_checkpoint(tmp_path):
    from neural_foundry.gym_service import _wire_document

    request = {"training_request_digest": "sha256:" + "a" * 64,
               "dataset_digest": "sha256:" + "b" * 64, "job_digest": "sha256:" + "c" * 64}
    binding = {"kind": "nf_layer_sweep_binding_v2", "run_digest": "sha256:" + "d" * 64,
        "request_digest": request["training_request_digest"], "dataset_digest": request["dataset_digest"],
        "base_digest": "sha256:" + "e" * 64, "layers": [8, 9, 10], "strengths": [0.01, 0.1, 0.5, 1.0]}
    records = [{"kind": "nf_layer_sweep_baseline_v2", "checkpoint_key": "baseline",
                "run_digest": binding["run_digest"]}]
    records.extend({"kind": "nf_layer_sweep_candidate_v2", "checkpoint_key": f"screen:{layer}:{strength:.8g}",
                    "run_digest": binding["run_digest"], "status": "measured"}
                   for layer in binding["layers"] for strength in binding["strengths"])
    ranking = [{"layer": 10, "tensor_name": "mlp10", "best_strength": 1.0,
                "best_validation_logprob_gain": 0.5},
               {"layer": 9, "tensor_name": "mlp9", "best_strength": 0.5,
                "best_validation_logprob_gain": 0.4},
               {"layer": 8, "tensor_name": "mlp8", "best_strength": 0.1,
                "best_validation_logprob_gain": 0.3}]
    records.append({"kind": "nf_layer_sweep_selection_v2", "checkpoint_key": "selection",
        "run_digest": binding["run_digest"], "selected_layers": [10, 9, 8], "layer_ranking": ranking})
    checkpoint = tmp_path / "sweep.checkpoint.jsonl"
    checkpoint.write_bytes(b"\n".join(_wire_document(row) for row in [binding, *records]) + b"\n")
    selected = load_sweep_likelihood_selection(checkpoint, request)
    assert selected["exact_scored_layer_indices"] == [10, 9, 8]
    assert [row["best_strength"] for row in selected["layer_ranking_by_validation_logprob"]] == [1.0, 0.5, 0.1]
    assert selected["holdout_accessed"] is False
    assert selected["source_checkpoint_digest"] == _document_digest({"binding": binding, "records": records})


def test_validation_interactions_are_grouped_for_each_template():
    from neural_foundry.reference_json_extraction import generate_scenarios, reference_task

    task = reference_task(rows_per_template=1)
    rows = generate_scenarios(split="validation", seed=17, parameters={"rows_per_template": 1})
    grouped = _interactions_by_template(task, rows)
    assert set(grouped) == {row["family_id"] for row in rows}
    assert sum(map(len, grouped.values())) == len(rows)
    assert all(grouped[row["family_id"]] for row in rows)


def test_sweep_site_map_contains_only_verified_mlp_writers(numerical_fixture):
    from neural_foundry.backends.weight_edit.discover import find_blocks, find_sites

    loaded, task, examples, recipe = numerical_fixture
    sites = find_sites(loaded.model, find_blocks(loaded.model), loaded.hidden_size,
                       checkpoint_tensor_names=loaded.checkpoint_tensor_names)
    writers = _mlp_writer_site_map(sites)
    assert writers.checkpoint_names_verified
    assert writers.sites
    assert all(site.role == "residual_write_mlp" for site in writers.sites)


@pytest.mark.parametrize("batch_equivalent", [False, True])
def test_layer_strength_sweep_cpu_smoke_runs_full_cli(tmp_path, numerical_fixture, monkeypatch, capsys,
                                                      batch_equivalent):
    import neural_foundry.backends.weight_edit.drift_diagnostic as diagnostic
    from neural_foundry.gym_service import _wire_document
    from neural_foundry.model_digest import model_state_digest
    from neural_foundry.reference_json_extraction import generate_scenarios, reference_task

    loaded, _, _, base_recipe = numerical_fixture
    task = reference_task(rows_per_template=2)
    train_rows = generate_scenarios(split="train", seed=17, parameters={"rows_per_template": 2})
    parent = [{"scenario_id": row["scenario_id"], "input": row["input"], "response": row["reference"],
               "source_split": "train"} for row in train_rows]
    selected, passing = parent[:10], parent[10:]
    parent_digest, filtered_digest = "sha256:" + "a" * 64, "sha256:" + "b" * 64
    selected_digests = [_document_digest(row) for row in selected]
    recipe = {**base_recipe, "target_step": 1.0, "retention_mode": "specialist",
        "preservation_policy_version": 2, "retention_nullspace_weight": 0.0,
        "preservation_interactions": [task.project(passing[0]["input"], None)],
        "protected_training_interactions": [task.project(row["input"], None) for row in passing],
        "max_sequence_tokens": 512, "max_training_tokens": 100000, "max_keys": 8192,
        "failure_selection": {"kind": "nf_base_failure_selection_v1",
            "baseline_report_digest": "sha256:" + "c" * 64,
            "base_state_digest": loaded.base_digest, "parent_dataset_digest": parent_digest,
            "filtered_dataset_digest": filtered_digest,
            "selected_scenario_ids": [row["scenario_id"] for row in selected],
            "selected_example_digests": selected_digests, "failure_count": 10, "minimum_failures": 10}}
    projected = [{"example_digest": digest,
                  "interaction": task.project(row["input"], row["response"])}
                 for row, digest in zip(selected, selected_digests, strict=True)]
    request = {"kind": "nf_gym_training_request_v1", "job_digest": task.job.digest,
        "dataset_digest": filtered_digest, "projected_examples_digest": _document_digest({"examples": projected}),
        "base_reference": {"artifact_path": str(loaded.checkpoint_path)},
        "loaded_base": {"state_digest": loaded.base_digest}, "recipe": recipe}
    request["training_request_digest"] = _document_digest(request)
    state_dir = tmp_path / "state"
    jobs = state_dir / "jobs"
    jobs.mkdir(parents=True)
    (jobs / f"{task.job.digest[7:]}.json").write_bytes(_wire_document(task.job.to_document()))
    request_path = tmp_path / "request.json"
    request_path.write_bytes(_wire_document(request))

    class FakeService:
        def __init__(self, *, state_dir=None):
            pass

        def load_training_examples(self, digest):
            return {filtered_digest: selected, parent_digest: parent}[digest]

    likelihood_calls = 0

    def fake_logprob(model, interactions, *, batch_size=4):
        nonlocal likelihood_calls
        likelihood_calls += 1
        mean = -1.0 + likelihood_calls / 1000
        cases = len(interactions)
        return {"cases": cases, "response_tokens": cases * 2, "mean_case_logprob": mean,
                "mean_token_logprob": mean, "sum_logprob": mean * cases * 2,
                "case_logprobs": [mean] * cases}

    def fake_score(task, model, rows, split="validation"):
        scoring_batch_sizes.append((split, model.batch_size))
        responses = [{"scenario_id": row["scenario_id"], "response": row["reference"],
                      "format_valid": True, "exact_match": float(index % 2),
                      "execution_evidence": {"generated_token_ids": [index] if batch_equivalent
                                             else [index, model.batch_size]}}
                     for index, row in enumerate(rows)]
        return {"split": split, "cases": len(rows), "exact_match": sum(row["exact_match"] for row in responses) / len(rows),
                "format_valid": 1.0, "panel_digest": "sha256:" + "d" * 64, "responses": responses}

    monkeypatch.setattr(diagnostic, "GymStudioService", FakeService)
    monkeypatch.setattr(diagnostic, "load_model", lambda *args, **kwargs: loaded)
    monkeypatch.setattr(diagnostic, "LAYER_SWEEP_STRENGTHS", (0.5, 1.0))
    monkeypatch.setattr(diagnostic, "LAYER_SWEEP_EXACT_LAYER_COUNT", 2)
    monkeypatch.setattr(diagnostic, "score_public_rows", fake_score)
    monkeypatch.setattr(diagnostic, "expected_answer_logprob", fake_logprob)
    generation_limits = []
    scoring_batch_sizes = []
    make_validation_model = diagnostic.validation_model

    def track_generation_limit(model, **kwargs):
        generation_limits.append(kwargs["max_new_tokens"])
        validation = make_validation_model(model, **kwargs)
        validation.batch_size = kwargs["batch_size"]
        return validation

    monkeypatch.setattr(diagnostic, "validation_model", track_generation_limit)
    monkeypatch.setattr(diagnostic, "calibrate_one_ulp", lambda *args, **kwargs: {"calibration_digest": "cpu-smoke"})
    monkeypatch.setattr(diagnostic, "preservation_logits", lambda model, panel: [torch.zeros(4) for _ in panel])
    monkeypatch.setattr(diagnostic, "float32_logits_of_exact_weights", lambda model, panel: [torch.zeros(4) for _ in panel])
    monkeypatch.setattr(diagnostic, "metrics", lambda before, after: {"cases": len(before)})
    monkeypatch.setattr(diagnostic, "verify_v2", lambda *args: {"passed": True,
        "float32": {"max_absolute_logit_drift": 0.0},
        "bf16": {"top1_agreement": 1.0, "top5_set_agreement": 1.0, "mean_kl_base_to_edited": 0.0}})
    output_path = tmp_path / "sweep" / "report.json"
    monkeypatch.setattr(sys, "argv", ["drift_diagnostic", str(state_dir), str(request_path),
                                      str(output_path), "--layer-strength-sweep", "--device", "cpu"])

    diagnostic.main()

    report = json.loads(output_path.read_bytes())
    assert report["discovered_layer_count"] == 2
    assert report["screen_candidate_count"] == 4
    assert report["exact_candidate_count"] == 2 * len(report["exact_scored_layer_indices"])
    assert len(report["exact_scored_layer_indices"]) == 1
    assert report["base_train"]["task"]["cases"] == len(parent)
    assert report["base_validation"]["task"]["cases"] == 4
    assert report["batch_generation_equivalent_to_serial"] is batch_equivalent
    expected_batch_size = 16 if batch_equivalent else 1
    assert report["exact_batch_size"] == expected_batch_size
    assert report["base_validation"]["task_by_template"]
    assert generation_limits and set(generation_limits) == {diagnostic.LAYER_SWEEP_MAX_NEW_TOKENS}
    assert report["generation_truncation"]["zero_truncated_answers"] is True
    validation_batches = [batch for split, batch in scoring_batch_sizes if split == "validation"]
    assert validation_batches[2:] == [expected_batch_size, expected_batch_size]
    assert all(row["preservation_v2_report_only"]["passed"] for row in report["results"]
               if row.get("phase") == "exact" and row.get("status") == "measured")
    assert report["strength_1_stop_gate"]["continue_to_multilayer"] is False
    assert report["base_restored"] is True and model_state_digest(loaded.model) == loaded.base_digest
    assert report["holdout_accessed"] is False
    cli_summary = json.loads(capsys.readouterr().out)
    assert cli_summary["layers"] == 2 and cli_summary["strengths"] == [0.5, 1.0]

    checkpoint_path = output_path.with_suffix(".checkpoint.jsonl")
    checkpoint_lines = [json.loads(line) for line in checkpoint_path.read_bytes().splitlines() if line]
    selection = next(row for row in checkpoint_lines[1:] if row.get("checkpoint_key") == "selection")
    selection["selected_layers"] = [1, 0]
    selection["layer_ranking"] = [
        {"layer": 1, "tensor_name": "model.layers.1.mlp.down_proj.weight", "best_strength": 1.0,
         "best_validation_logprob_gain": 0.2},
        {"layer": 0, "tensor_name": "model.layers.0.mlp.down_proj.weight", "best_strength": 0.5,
         "best_validation_logprob_gain": 0.1},
    ]
    checkpoint_path.write_bytes(b"\n".join(_wire_document(row) for row in checkpoint_lines) + b"\n")
    finalist_path = tmp_path / "finalists" / "report.json"
    finalist_report = diagnostic.run_finalist_rescore(state_dir, request_path, checkpoint_path, finalist_path,
                                                       device="cpu", composition_sizes=(2,))
    assert finalist_report["generation_max_new_tokens"] == 128
    assert finalist_report["selected_layer_strengths"] == [
        {"layer": 1, "tensor_name": "model.layers.1.mlp.down_proj.weight", "target_step": 1.0,
         "validation_logprob_gain": 0.2},
        {"layer": 0, "tensor_name": "model.layers.0.mlp.down_proj.weight", "target_step": 0.5,
         "validation_logprob_gain": 0.1},
    ]
    assert len(finalist_report["candidates"]) == 1
    candidate = finalist_report["candidates"][0]
    assert candidate["composition_size"] == 2 and candidate["layers"] == [1, 0]
    assert candidate["validation"]["cases"] == 4 and candidate["required_set"]["cases"] > 0
    assert candidate["specialist_v2_gate"]["policy_version"] == 2
    assert candidate["preservation_v2_report_only"]["passed"] is True
    assert finalist_report["generation_truncation"]["zero_truncated_answers"] is True
    assert finalist_report["base_restored"] and model_state_digest(loaded.model) == loaded.base_digest
    assert finalist_report["holdout_accessed"] is False and finalist_report["candidate_published"] is False

    scoring_batch_sizes.clear()
    grid = tuple((layer, strength) for layer in (1, 0) for strength in (0.1, 0.5, 1.0))
    single_report = diagnostic.run_finalist_rescore(state_dir, request_path, checkpoint_path,
        tmp_path / "single-layer" / "report.json", device="cpu", single_layer_candidates=grid)
    assert single_report["kind"] == "nf_weight_edit_single_layer_stop_gate_v1"
    assert len(single_report["candidates"]) == 6
    assert [(row["layer"], row["target_step"]) for row in single_report["candidates"]] == list(grid)
    assert all(batch == 1 for split, batch in scoring_batch_sizes)
    for candidate in single_report["candidates"]:
        assert candidate["composition_size"] == 1 and len(candidate["stored_weight_changes"]) == 1
        assert candidate["validation_fixed_cases"] == candidate["validation_broken_cases"] == 0
        assert "zero_short_truncation" not in candidate["specialist_v2_gate"]["checks"]
        metrics = next(iter(candidate["stored_weight_changes"].values()))
        assert "relative_frobenius_norm" in metrics and "changed_entry_fraction" in metrics
        checks = next(iter(candidate["preservation_v2_checks"].values()))
        assert checks["float32_max_drift"]["threshold"] == 0.25
        assert checks["bf16_top5"]["threshold"] == 0.95
    assert single_report["strength_1_stop_gate"]["automatic_followup_authorized"] is False
    assert single_report["base_restored"] and model_state_digest(loaded.model) == loaded.base_digest


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