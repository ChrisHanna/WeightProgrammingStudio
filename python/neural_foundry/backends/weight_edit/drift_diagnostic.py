"""Read-only calibration experiments; never changes preservation policy."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from ...gym_job import GymJobError, gym_job_from_document
from ...gym_adapters import _parse_generated_json
from ...gym_artifacts import CandidateRef
from ...gym_service import ExecutionResponse, GymStudioService, _document_digest, _wire_document
from ...model_digest import model_state_digest
from ...provenance import parse_strict_json
from ...reference_json_extraction import TEMPLATES, generate_scenarios, reference_task
from ...json_proof_loop import SPECIALIST_POLICY, public_required_rows, score_public_rows, specialist_acceptance
from ...specialist_policy import assess_specialist_v2
from ...prompt_rendering import PromptRenderer
from .apply import apply, save, tensor_digests
from .loader import WeightEditInputError, load_model
from .discover import SiteMap, find_blocks, find_sites
from .methods import Delta
from .numerics import deterministic_execution
from .preservation import POLICY_V2, calibrate_one_ulp, float32_logits_of_exact_weights, metrics, verify_v2
from .worker import calculate_edit, calculate_edits_for_sites, preservation_logits
from .demonstrations import prepare_demonstration

STRENGTHS = (0.0, 1e-5, 1e-4, 1e-3)
LAYER_SWEEP_STRENGTHS = (1e-2, 0.1, 0.5, 1.0)
LAYER_SWEEP_MAX_NEW_TOKENS = 128
LAYER_SWEEP_EXACT_LAYER_COUNT = 3


def select_top_third(screen_rows: list[dict[str, Any]]) -> list[str]:
    eligible = [row for row in screen_rows if row.get("status") == "measured"
                and isinstance(row.get("validation_screen_logprob_gain"), (int, float))]
    eligible.sort(key=lambda row: (row["validation_screen_logprob_gain"],
                                   row["training_logprob_gain"], -row["layer"], -row["target_step"]), reverse=True)
    count = max(1, (len(eligible) + 2) // 3) if eligible else 0
    return [row["checkpoint_key"] for row in eligible[:count]]


def select_validation_screen(rows: list[dict[str, Any]], *, per_family: int = 8) -> list[dict[str, Any]]:
    if type(per_family) is not int or per_family < 1:
        raise WeightEditInputError("validation screen must select at least one row per family")
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        family = row.get("family_id")
        if not isinstance(family, str) or not family:
            raise WeightEditInputError("validation screen rows require a family ID")
        by_family.setdefault(family, []).append(row)
    return [row for family_rows in by_family.values() for row in family_rows[:per_family]]


def _load_training_rosters(service: GymStudioService, request: dict[str, Any]):
    selected = list(service.load_training_examples(request["dataset_digest"]))
    selection = request["recipe"].get("failure_selection")
    if selection is None:
        return selected, selected
    if selection.get("filtered_dataset_digest") != request["dataset_digest"]:
        raise WeightEditInputError("filtered training roster differs from failure-selection binding")
    parent = list(service.load_training_examples(selection["parent_dataset_digest"]))
    selected_ids = selection["selected_scenario_ids"]
    parent_ids = [row.get("scenario_id") for row in parent]
    if len(set(parent_ids)) != len(parent_ids):
        raise WeightEditInputError("parent training roster contains duplicate scenario IDs")
    parent_selected = [row for row in parent if row["scenario_id"] in set(selected_ids)]
    if ([row.get("scenario_id") for row in selected] != selected_ids
        or [row.get("scenario_id") for row in parent_selected] != selected_ids
        or [_document_digest(row) for row in selected] != selection["selected_example_digests"]
        or [_document_digest(row) for row in parent_selected] != selection["selected_example_digests"]):
        raise WeightEditInputError("filtered examples differ from the digest-bound parent train roster")
    return selected, parent


def open_sweep_checkpoint(path: Path, binding: dict[str, Any]) -> list[dict[str, Any]]:
    if path.exists():
        lines = [line for line in path.read_bytes().splitlines() if line]
        if not lines or parse_strict_json(lines[0]) != {"kind": "nf_layer_sweep_binding_v2", **binding}:
            raise WeightEditInputError("layer-sweep checkpoint binding differs from the frozen run")
        records = [parse_strict_json(line) for line in lines[1:]]
        if any(record.get("run_digest") != binding["run_digest"] for record in records):
            raise WeightEditInputError("layer-sweep checkpoint contains records from another run")
        return records
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(_wire_document({"kind": "nf_layer_sweep_binding_v2", **binding}) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    return []


def append_sweep_checkpoint(path: Path, record: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_APPEND | os.O_WRONLY)
    with os.fdopen(descriptor, "ab") as stream:
        stream.write(_wire_document(record) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _compact_score(score: dict[str, Any]) -> dict[str, Any]:
    compact = {key: score[key] for key in ("split", "cases", "exact_match", "format_valid", "panel_digest", "responses") if key in score}
    evidence = [row.get("execution_evidence") or {} for row in score.get("responses", ())]
    compact["truncated_answer_count"] = sum(bool(row.get("truncated")) for row in evidence)
    compact["generation_cap_failure_count"] = sum(bool(row.get("hit_generation_cap")) for row in evidence)
    return compact


def _completion_was_truncated(generated_token_ids: list[int], text: str, max_new_tokens: int) -> bool:
    return len(generated_token_ids) >= max_new_tokens


def _has_complete_json_value(text: str) -> bool:
    start = text.rfind("```json")
    if start >= 0:
        start += len("```json")
    else:
        start = next((index for index, character in enumerate(text) if character in "[{"), -1)
    if start < 0:
        return False
    while start < len(text) and text[start].isspace():
        start += 1
    try:
        value, _ = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError:
        return False
    return isinstance(value, (dict, list))


class _StopAfterJsonRoot:
    def __init__(self, tokenizer, *, prompt_width: int, eos_token_id: int, batch_size: int) -> None:
        self.tokenizer = tokenizer
        self.prompt_width = prompt_width
        self.eos_token_id = eos_token_id
        self.completed = [False] * batch_size

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        for index in range(input_ids.shape[0]):
            generated = input_ids[index, self.prompt_width:].tolist()
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            if _has_complete_json_value(text):
                self.completed[index] = True
                scores[index].fill_(float("-inf"))
                scores[index, self.eos_token_id] = 0
        return scores


def _validation_template_scores(rows: list[dict[str, Any]], score: dict[str, Any]) -> dict[str, dict[str, Any]]:
    family_by_id = {row["scenario_id"]: row["family_id"] for row in rows}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for response in score["responses"]:
        family = family_by_id.get(response.get("scenario_id"))
        if family is None:
            raise WeightEditInputError("validation score contains an unknown scenario ID")
        grouped.setdefault(family, []).append(response)
    if sum(map(len, grouped.values())) != len(rows):
        raise WeightEditInputError("validation score is missing template cases")
    return {family: {"cases": len(responses),
                     "exact_match": sum(row["exact_match"] for row in responses) / len(responses),
                     "exact_match_count": sum(row["exact_match"] == 1.0 for row in responses),
                     "format_valid": sum(row.get("format_valid", 0.0) for row in responses) / len(responses),
                     "format_valid_count": sum(row.get("format_valid", 0.0) == 1.0 for row in responses)}
            for family, responses in grouped.items()}


def _interactions_by_template(task, rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["family_id"], []).append(task.project(row["input"], row["reference"]))
    return grouped


def _mlp_writer_site_map(sites: SiteMap) -> SiteMap:
    if not sites.checkpoint_names_verified:
        raise WeightEditInputError("layer sweep requires checkpoint-verified MLP writer sites")
    writers = tuple(site for site in sites.sites if site.role == "residual_write_mlp")
    if not writers:
        raise WeightEditInputError("layer sweep found no MLP writer sites")
    return SiteMap(writers, checkpoint_names_verified=True)


def _aggregate_logprobs(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    case_logprobs = [value for report in reports.values() for value in report["case_logprobs"]]
    cases = sum(report["cases"] for report in reports.values())
    response_tokens = sum(report["response_tokens"] for report in reports.values())
    sum_logprob = sum(report["sum_logprob"] for report in reports.values())
    if cases == 0 or response_tokens == 0 or len(case_logprobs) != cases:
        raise WeightEditInputError("validation template likelihood reports are incomplete")
    return {"cases": cases, "response_tokens": response_tokens,
            "mean_case_logprob": sum(case_logprobs) / cases,
            "mean_token_logprob": sum_logprob / response_tokens,
            "sum_logprob": sum_logprob, "case_logprobs": case_logprobs}


def strength_one_stop_gate(layer: int, row: dict[str, Any] | None) -> dict[str, Any]:
    measured = row is not None and row.get("status") == "measured" and row.get("layer") == layer
    effective_change = bool(measured and row.get("effective_change", row.get("effective_change_norm", 0) > 0))
    task_gain = row.get("validation_exact_match_gain") if measured else None
    continue_to_multilayer = effective_change and isinstance(task_gain, (int, float)) and task_gain > 0
    return {"evaluated_layer": layer, "effective_change": effective_change,
            "validation_exact_match_gain": task_gain, "continue_to_multilayer": continue_to_multilayer,
            "rule": "stop if strength 1.0 has no effective change or no positive public-validation exact-match gain"}


def validation_model(loaded, *, max_new_tokens=80, batch_size=8):
    renderer = PromptRenderer(loaded.tokenizer)
    if type(batch_size) is not int or not 1 <= batch_size <= 16:
        raise WeightEditInputError("validation generation batch size must be from 1 to 16")

    def infer_many(interactions):
        device = next(loaded.model.parameters()).device
        pad_token = loaded.tokenizer.pad_token_id
        if type(pad_token) is not int:
            pad_token = loaded.tokenizer.eos_token_id
        if type(pad_token) is not int:
            raise WeightEditInputError("batched validation generation requires a pad or EOS token")
        results = []
        for start in range(0, len(interactions), batch_size):
            chunk = interactions[start:start + batch_size]
            rendered = [renderer.render(interaction) for interaction in chunk]
            if any(not prompt.token_ids for prompt in rendered):
                raise WeightEditInputError("public validation prompt rendered to no tokens")
            width = max(len(prompt.token_ids) for prompt in rendered)
            if width + max_new_tokens > 512:
                raise WeightEditInputError("public validation prompt exceeds frozen context")
            inputs = torch.full((len(chunk), width), pad_token, dtype=torch.long, device=device)
            attention = torch.zeros_like(inputs)
            for index, prompt in enumerate(rendered):
                prompt_ids = torch.tensor(prompt.token_ids, dtype=torch.long, device=device)
                inputs[index, width - len(prompt_ids):] = prompt_ids
                attention[index, width - len(prompt_ids):] = 1
            eos = loaded.tokenizer.eos_token_id
            if type(eos) is not int:
                raise WeightEditInputError("JSON completion stopping requires an EOS token")
            stop_after_json = _StopAfterJsonRoot(loaded.tokenizer, prompt_width=width,
                eos_token_id=eos, batch_size=len(chunk))
            with deterministic_execution(loaded.model), torch.inference_mode():
                output = loaded.model.generate(input_ids=inputs, attention_mask=attention, do_sample=False,
                    max_new_tokens=max_new_tokens, pad_token_id=pad_token, logits_processor=[stop_after_json])
            for index in range(len(chunk)):
                generated = output[index, width:].tolist()
                if type(eos) is int and eos in generated:
                    generated = generated[:generated.index(eos) + 1]
                text = loaded.tokenizer.decode(generated, skip_special_tokens=True)
                parsed = _parse_generated_json(text)
                hit_generation_cap = len(generated) >= max_new_tokens and not stop_after_json.completed[index]
                results.append(ExecutionResponse(parsed,
                    {"raw_text": text, "generated_token_ids": generated,
                     "max_new_tokens": max_new_tokens,
                     "hit_generation_cap": hit_generation_cap,
                     "truncated": _completion_was_truncated(generated, text, max_new_tokens),
                     "renderer_digest": renderer.digest, "batch_size": len(chunk)}))
        return results

    def infer(interaction):
        return infer_many([interaction])[0]

    identity = {"runtime": "Transformers", "model_digest": model_state_digest(loaded.model),
                "precision": str(next(loaded.model.parameters()).dtype).removeprefix("torch."),
                "renderer_digest": renderer.digest, "served_model_identity_verified": False}
    return SimpleNamespace(infer=infer, infer_many=infer_many, loaded_identity=lambda: identity)


def run_public_split_baseline(state_dir: Path, request_path: Path, output_path: Path, device="cuda:0"):
    request = parse_strict_json(request_path.read_bytes())
    if request["training_request_digest"] != _document_digest({key: value for key, value in request.items() if key != "training_request_digest"}):
        raise WeightEditInputError("baseline request digest is invalid")
    job = gym_job_from_document(parse_strict_json((state_dir / "jobs" / (request["job_digest"][7:] + ".json")).read_bytes()))
    if job.digest != request["job_digest"]:
        raise WeightEditInputError("baseline source job digest is invalid")
    rows_per_template = job.scenario_sources[0].parameters["rows_per_template"]
    seed = request["recipe"]["seed"]
    task = reference_task(rows_per_template=rows_per_template)
    train_rows = generate_scenarios(split="train", seed=seed, parameters={"rows_per_template": rows_per_template})
    validation_rows = generate_scenarios(split="validation", seed=seed, parameters={"rows_per_template": rows_per_template})
    loaded = load_model(request["base_reference"]["artifact_path"], dtype=torch.bfloat16, device=device)
    if loaded.base_digest != request["loaded_base"]["state_digest"]:
        raise WeightEditInputError("baseline loaded state differs from the frozen base")
    renderer = PromptRenderer(loaded.tokenizer)
    if renderer.digest != request["inference_config"]["prompt_renderer_digest"]:
        raise WeightEditInputError("baseline renderer differs from the frozen run")
    model = validation_model(loaded, max_new_tokens=80, batch_size=16)
    train_score = score_public_rows(task, model, train_rows, split="train")
    validation_score = score_public_rows(task, model, validation_rows, split="validation")
    train_logprob = expected_answer_logprob(loaded, [task.project(row["input"], row["reference"]) for row in train_rows], batch_size=16)
    validation_logprob = expected_answer_logprob(loaded, [task.project(row["input"], row["reference"]) for row in validation_rows], batch_size=16)
    report = {"kind": "nf_json_public_split_baseline_v1", "evidence_status": "measured",
        "source_job_digest": job.digest, "current_job_digest": task.job.digest,
        "scenario_origin": "synthetic", "seed": seed,
        "rows_per_template": rows_per_template, "train_template_count": len(TEMPLATES["train"]),
        "validation_template_count": len(TEMPLATES["validation"]),
        "base_candidate_digest": request["base_reference"]["model_digest"],
        "base_state_digest": loaded.base_digest, "renderer_digest": renderer.digest,
        "train": {"score": train_score, "failures": [row["scenario_id"] for row in train_score["responses"] if row["exact_match"] < 1.0],
                  "failure_count": sum(row["exact_match"] < 1.0 for row in train_score["responses"]),
                  "expected_answer_logprob": train_logprob},
        "validation": {"score": validation_score, "failures": [row["scenario_id"] for row in validation_score["responses"] if row["exact_match"] < 1.0],
                       "failure_count": sum(row["exact_match"] < 1.0 for row in validation_score["responses"]),
                       "expected_answer_logprob": validation_logprob},
        "batch_size": 16, "holdout_accessed": False, "candidate_published": False, "release_authorized": False}
    report["evidence_digest"] = _document_digest(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as stream:
        stream.write(_wire_document(report))
    if device.startswith("cuda"):
        loaded.model.to("cpu")
        torch.cuda.empty_cache()
    return report


def measure_validation_strengths(loaded, task, examples, recipe, *, seed, freeze_calibration):
    panel = recipe["preservation_interactions"]
    rows = generate_scenarios(split="validation", seed=seed, parameters=task.job.scenario_sources[0].parameters)
    required_rows = public_required_rows(seed=seed)
    before = preservation_logits(loaded, panel)
    precise_before = float32_logits_of_exact_weights(loaded, panel)
    baseline_model = validation_model(loaded)
    baseline = score_public_rows(task, baseline_model, rows)
    baseline_required = score_public_rows(task, baseline_model, required_rows)
    sites, site, calculated, evidence = calculate_edit(loaded, task.job, examples, {**recipe, "target_step": 1e-3})
    calibration = calibrate_one_ulp(loaded, site, panel)
    freeze_calibration(calibration)
    original = loaded.model.get_parameter(site.tensor_name).detach().clone()
    records = []
    try:
        for strength in STRENGTHS[1:]:
            with torch.no_grad():
                loaded.model.get_parameter(site.tensor_name).copy_(original)
            if model_state_digest(loaded.model) != loaded.base_digest:
                raise WeightEditInputError("validation sweep failed to restore exact base")
            delta = Delta("low_rank", U=calculated.U * (strength / 1e-3), V=calculated.V)
            edited = apply(loaded, {site.tensor_name: delta}, sites=sites, method="public_validation_strength_sweep",
                           settings={"target_step": strength, "retention_mode": "specialist", "diagnostic_only": True})
            preservation = verify_v2(metrics(precise_before, float32_logits_of_exact_weights(loaded, panel)),
                                     metrics(before, preservation_logits(loaded, panel)), calibration)
            model = validation_model(loaded)
            score = score_public_rows(task, model, rows)
            required = score_public_rows(task, model, required_rows)
            records.append({"target_step": strength, "task_score": score, "task_gain": score["exact_match"] - baseline["exact_match"],
                            "required_set": required, "specialist_acceptance": specialist_acceptance(score, required),
                            "preservation": preservation, "edited_tensor_digest": model_state_digest(loaded.model),
                            "change_norm": edited.receipt["change_norms"][site.tensor_name]})
    finally:
        with torch.no_grad():
            loaded.model.get_parameter(site.tensor_name).copy_(original)
    if model_state_digest(loaded.model) != loaded.base_digest:
        raise WeightEditInputError("validation sweep changed the base tensor state")
    return {"baseline": baseline, "baseline_required_set": baseline_required, "strengths": records,
            "calibration": calibration, "base_restored": True, "renderer_digest": evidence["renderer_digest"]}


def run_validation_diagnostic(state_dir: Path, request_path: Path, output_path: Path, device="cuda:0"):
    request = parse_strict_json(request_path.read_bytes())
    if request["training_request_digest"] != _document_digest({key: value for key, value in request.items() if key != "training_request_digest"}):
        raise WeightEditInputError("validation diagnostic request digest is invalid")
    job = gym_job_from_document(parse_strict_json((state_dir / "jobs" / (request["job_digest"][7:] + ".json")).read_bytes()))
    task = reference_task(rows_per_template=job.scenario_sources[0].parameters["rows_per_template"])
    if job.digest != task.job.digest:
        raise WeightEditInputError("validation diagnostic differs from frozen reference task")
    service = GymStudioService(state_dir=state_dir)
    examples = [{"example_digest": _document_digest(example), "interaction": task.project(example["input"], example["response"])}
                for example in service.load_training_examples(request["dataset_digest"])]
    if _document_digest({"examples": examples}) != request["projected_examples_digest"]:
        raise WeightEditInputError("validation demonstrations differ from frozen dataset")
    loaded = load_model(request["base_reference"]["artifact_path"], dtype=torch.bfloat16, device=device)
    if loaded.base_digest != request["loaded_base"]["state_digest"]:
        raise WeightEditInputError("validation base differs from frozen request")
    output_path.parent.mkdir(parents=True, exist_ok=False)

    def freeze(calibration):
        with output_path.with_name("calibration.json").open("xb") as stream:
            stream.write(_wire_document(calibration))

    result = measure_validation_strengths(loaded, task, examples, request["recipe"], seed=request["recipe"]["seed"], freeze_calibration=freeze)
    report = {"kind": "nf_weight_edit_public_validation_v1", "evidence_status": "measured",
              "scenario_origin": "synthetic", "split": "validation", "job_digest": job.digest,
              "source_training_request_digest": request["training_request_digest"], "dataset_digest": request["dataset_digest"],
              "retention_mode": "specialist", "specialist_policy_digest": _document_digest(SPECIALIST_POLICY),
              "holdout_accessed": False, "candidate_published": False, "release_authorized": False, **result}
    report["evidence_digest"] = _document_digest(report)
    with output_path.open("xb") as stream:
        stream.write(_wire_document(report))
    return report


def run_training_fit_diagnostic(state_dir: Path, request_path: Path, output_path: Path, device="cuda:0"):
    request = parse_strict_json(request_path.read_bytes())
    if request["training_request_digest"] != _document_digest({key: value for key, value in request.items() if key != "training_request_digest"}):
        raise WeightEditInputError("training fit request digest is invalid")
    task = reference_task(rows_per_template=32)
    job = gym_job_from_document(parse_strict_json((state_dir / "jobs" / (request["job_digest"][7:] + ".json")).read_bytes()))
    task = reference_task(rows_per_template=job.scenario_sources[0].parameters["rows_per_template"])
    if job.digest != task.job.digest:
        raise WeightEditInputError("training fit differs from frozen task")
    service = GymStudioService(state_dir=state_dir)
    approved = service.load_training_examples(request["dataset_digest"])
    examples = [{"example_digest": _document_digest(example), "interaction": task.project(example["input"], example["response"])}
                for example in approved]
    if _document_digest({"examples": examples}) != request["projected_examples_digest"]:
        raise WeightEditInputError("training fit examples differ from the frozen approved dataset")
    rows = [{"scenario_id": f"train:frozen-dataset:{index}", "input": example["input"], "reference": example["response"],
             "training_example_digest": _document_digest(example)} for index, example in enumerate(approved)]
    loaded = load_model(request["base_reference"]["artifact_path"], dtype=torch.bfloat16, device=device)
    if loaded.base_digest != request["loaded_base"]["state_digest"]:
        raise WeightEditInputError("training fit base differs from frozen request")
    baseline = score_public_rows(task, validation_model(loaded), rows, split="train")
    baseline_logprob = expected_answer_logprob(loaded, [example["interaction"] for example in examples])
    sites, site, delta, evidence = calculate_edit(loaded, job, examples, request["recipe"])
    original = loaded.model.get_parameter(site.tensor_name).detach().clone()
    try:
        edited = apply(loaded, {site.tensor_name: delta}, sites=sites, method="legacy_key_value_v1", settings=request["recipe"])
        model_digest = model_state_digest(loaded.model)
        score = score_public_rows(task, validation_model(loaded), rows, split="train")
        edited_logprob = expected_answer_logprob(loaded, [example["interaction"] for example in examples])
    finally:
        with torch.no_grad():
            loaded.model.get_parameter(site.tensor_name).copy_(original)
    if model_state_digest(loaded.model) != loaded.base_digest:
        raise WeightEditInputError("training fit failed to restore exact base")
    report = {"kind": "nf_weight_edit_training_fit_v1", "evidence_status": "measured", "split": "train",
              "job_digest": job.digest, "dataset_digest": request["dataset_digest"],
              "source_training_request_digest": request["training_request_digest"], "target_step": request["recipe"]["target_step"],
              "retention_nullspace_weight": request["recipe"]["retention_nullspace_weight"], "recipe": request["recipe"],
              "baseline": baseline, "edited": score, "training_gain": score["exact_match"] - baseline["exact_match"],
              "expected_answer_logprob": {"base": baseline_logprob, "edited": edited_logprob,
                  "mean_token_gain": edited_logprob["mean_token_logprob"] - baseline_logprob["mean_token_logprob"]},
              "edited_model_digest": model_digest, "base_restored": True, "calculation_evidence": evidence,
              "change_norm": edited.receipt["change_norms"][site.tensor_name], "candidate_published": False,
              "validation_accessed": False, "holdout_accessed": False, "release_authorized": False}
    report["evidence_digest"] = _document_digest(report)
    output_path.parent.mkdir(parents=True, exist_ok=False)
    with output_path.open("xb") as stream:
        stream.write(_wire_document(report))
    return report


def _sync_if_cuda(model) -> None:
    if next(model.parameters()).device.type == "cuda":
        torch.cuda.synchronize()


def run_single_candidate_profile(state_dir: Path, request_path: Path, output_path: Path, *, strength=1e-3,
                                 device="cuda:0"):
    request = parse_strict_json(request_path.read_bytes())
    if request["training_request_digest"] != _document_digest({key: value for key, value in request.items() if key != "training_request_digest"}):
        raise WeightEditInputError("profile request digest is invalid")
    job = gym_job_from_document(parse_strict_json((state_dir / "jobs" / (request["job_digest"][7:] + ".json")).read_bytes()))
    task = reference_task(rows_per_template=job.scenario_sources[0].parameters["rows_per_template"])
    if job.digest != task.job.digest:
        raise WeightEditInputError("profile differs from frozen task")
    service = GymStudioService(state_dir=state_dir)
    approved, train_examples = _load_training_rosters(service, request)
    examples = [{"example_digest": _document_digest(row), "interaction": task.project(row["input"], row["response"])}
                for row in approved]
    if _document_digest({"examples": examples}) != request["projected_examples_digest"]:
        raise WeightEditInputError("profile examples differ from frozen training dataset")
    train_rows = [{"scenario_id": row["scenario_id"], "input": row["input"], "reference": row["response"]}
                  for row in train_examples]
    train_interactions = [task.project(row["input"], row["response"]) for row in train_examples]
    validation_rows = generate_scenarios(split="validation", seed=request["recipe"]["seed"],
                                         parameters=task.job.scenario_sources[0].parameters)
    validation_interactions = [task.project(row["input"], row["reference"]) for row in validation_rows]
    validation_by_template = _interactions_by_template(task, validation_rows)
    screening_rows = select_validation_screen(validation_rows)
    screening_interactions = [task.project(row["input"], row["reference"]) for row in screening_rows]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise WeightEditInputError("profile output already exists; choose a fresh path")
    timings = {}
    load_start = time.perf_counter()
    loaded = load_model(request["base_reference"]["artifact_path"], dtype=torch.bfloat16, device=device)
    _sync_if_cuda(loaded.model)
    timings["model_load_seconds"] = time.perf_counter() - load_start
    if loaded.base_digest != request["loaded_base"]["state_digest"]:
        raise WeightEditInputError("profile base differs from frozen request")
    baseline_likelihood_start = time.perf_counter()
    baseline_train_logprob = expected_answer_logprob(loaded, train_interactions, batch_size=16)
    baseline_validation_logprob = expected_answer_logprob(loaded, validation_interactions, batch_size=16)
    _sync_if_cuda(loaded.model)
    timings["baseline_batched_likelihood_seconds"] = time.perf_counter() - baseline_likelihood_start
    baseline_exact_start = time.perf_counter()
    baseline_train_score = score_public_rows(task, validation_model(loaded, batch_size=16, max_new_tokens=40), train_rows, split="train")
    baseline_validation_score = score_public_rows(task, validation_model(loaded, batch_size=16, max_new_tokens=40), validation_rows)
    _sync_if_cuda(loaded.model)
    timings["baseline_batched_exact_generation_seconds"] = time.perf_counter() - baseline_exact_start
    discovery_start = time.perf_counter()
    sites = find_sites(loaded.model, find_blocks(loaded.model), loaded.hidden_size,
                       checkpoint_tensor_names=loaded.checkpoint_tensor_names)
    writers = [site for site in sites.sites if site.role == "residual_write_mlp"]
    if not writers:
        raise WeightEditInputError("profile found no verified MLP write site")
    profile_site = writers[len(writers) // 2]
    timings["site_discovery_seconds"] = time.perf_counter() - discovery_start
    preservation_panel = [*request["recipe"]["preservation_interactions"],
                          *request["recipe"].get("protected_training_interactions", ())]
    calibration_start = time.perf_counter()
    calibration = calibrate_one_ulp(loaded, profile_site, preservation_panel)
    preservation_before = preservation_logits(loaded, preservation_panel)
    float32_before = float32_logits_of_exact_weights(loaded, preservation_panel)
    _sync_if_cuda(loaded.model)
    timings["preservation_calibration_and_baseline_seconds"] = time.perf_counter() - calibration_start
    capture_compute_start = time.perf_counter()
    _, site, delta, calculation = calculate_edit(loaded, job, examples, request["recipe"],
        site_layer=profile_site.layer, discovered_sites=sites)
    _sync_if_cuda(loaded.model)
    timings["capture_and_delta_compute_seconds"] = time.perf_counter() - capture_compute_start
    base_digest = model_state_digest(loaded.model)
    original = loaded.model.get_parameter(site.tensor_name).detach().clone()
    try:
        apply_start = time.perf_counter()
        scaled = Delta("low_rank", U=delta.U * (strength / request["recipe"]["target_step"]), V=delta.V)
        edited = apply(loaded, {site.tensor_name: scaled}, sites=SiteMap((site,), checkpoint_names_verified=True),
                       method="single_candidate_profile", settings={"target_step": strength, "profile_only": True})
        _sync_if_cuda(loaded.model)
        timings["in_memory_apply_seconds"] = time.perf_counter() - apply_start

        likelihood_start = time.perf_counter()
        train_logprob = expected_answer_logprob(loaded, train_interactions, batch_size=16)
        validation_logprob = expected_answer_logprob(loaded, validation_interactions, batch_size=16)
        _sync_if_cuda(loaded.model)
        timings["batched_likelihood_seconds"] = time.perf_counter() - likelihood_start

        exact_start = time.perf_counter()
        train_score = score_public_rows(task, validation_model(loaded, batch_size=16, max_new_tokens=40), train_rows, split="train")
        validation_score = score_public_rows(task, validation_model(loaded, batch_size=16, max_new_tokens=40), validation_rows)
        _sync_if_cuda(loaded.model)
        timings["batched_exact_generation_seconds"] = time.perf_counter() - exact_start

        preservation_start = time.perf_counter()
        preservation_after = preservation_logits(loaded, preservation_panel)
        float32_after = float32_logits_of_exact_weights(loaded, preservation_panel)
        preservation_v2 = verify_v2(metrics(float32_before, float32_after),
            metrics(preservation_before, preservation_after), calibration)
        _sync_if_cuda(loaded.model)
        timings["preservation_measurement_seconds"] = time.perf_counter() - preservation_start

        with tempfile.TemporaryDirectory(prefix="nf-one-candidate-profile-") as temporary:
            artifact_path = Path(temporary) / "candidate"
            export_start = time.perf_counter()
            reference = save(edited, artifact_path, model_id=request["base_reference"]["model_id"],
                             request_digest=request["training_request_digest"], base_reference=CandidateRef(**request["base_reference"]),
                             base_revision=request["loaded_base"]["revision"])
            _sync_if_cuda(loaded.model)
            timings["export_and_builtin_reload_seconds"] = time.perf_counter() - export_start
            reload_start = time.perf_counter()
            reloaded = load_model(reference.artifact_path, dtype=torch.bfloat16, device="cpu")
            timings["explicit_fresh_reload_seconds"] = time.perf_counter() - reload_start
            if reloaded.base_digest != reference.model_digest:
                raise WeightEditInputError("profile fresh reload differs from candidate")

        report = {"kind": "nf_weight_edit_single_candidate_profile_v1", "evidence_status": "measured",
            "strength": strength, "layer": site.layer, "tensor_name": site.tensor_name,
            "base_digest": base_digest, "candidate_model_digest": edited.receipt["model_digest"],
            "effective_change_norm": edited.receipt["change_norms"][site.tensor_name],
            "baseline_train_exact_match": baseline_train_score["exact_match"],
            "baseline_validation_exact_match": baseline_validation_score["exact_match"],
            "train_exact_match": train_score["exact_match"], "validation_exact_match": validation_score["exact_match"],
            "train_scoring_cases": len(train_rows), "edit_example_count": len(examples),
            "train_exact_match_gain": train_score["exact_match"] - baseline_train_score["exact_match"],
            "validation_exact_match_gain": validation_score["exact_match"] - baseline_validation_score["exact_match"],
            "baseline_train_mean_token_logprob": baseline_train_logprob["mean_token_logprob"],
            "baseline_validation_mean_token_logprob": baseline_validation_logprob["mean_token_logprob"],
            "train_mean_token_logprob": train_logprob["mean_token_logprob"],
            "validation_mean_token_logprob": validation_logprob["mean_token_logprob"],
            "train_mean_token_logprob_gain": train_logprob["mean_token_logprob"] - baseline_train_logprob["mean_token_logprob"],
            "validation_mean_token_logprob_gain": validation_logprob["mean_token_logprob"] - baseline_validation_logprob["mean_token_logprob"],
            "preservation_v2_report_only": preservation_v2,
            "preservation_cases": len(preservation_panel),
            "preservation_panel_digest": _document_digest({"source": "public_train", "interactions": preservation_panel}),
            "preservation_calibration": calibration,
            "calculation_evidence": calculation, "timings_seconds": timings,
            "evaluation_batch_size": 16, "generation_max_new_tokens": 40,
            "serving_container_profiled": False, "candidate_published": False,
            "holdout_accessed": False, "release_authorized": False}
        report["profile_digest"] = _document_digest(report)
        with output_path.open("xb") as stream:
            stream.write(_wire_document(report))
        return report
    finally:
        with torch.no_grad():
            loaded.model.get_parameter(site.tensor_name).copy_(original)
        if model_state_digest(loaded.model) != base_digest:
            raise WeightEditInputError("single-candidate profile failed exact base restoration")


def expected_answer_logprob(loaded, interactions, *, batch_size=4):
    if type(batch_size) is not int or not 1 <= batch_size <= 64:
        raise WeightEditInputError("likelihood batch size must be from 1 to 64")
    renderer = PromptRenderer(loaded.tokenizer)
    device = next(loaded.model.parameters()).device
    case_sums, case_means, token_count = [], [], 0
    pad_token = loaded.tokenizer.pad_token_id
    if type(pad_token) is not int:
        pad_token = loaded.tokenizer.eos_token_id
    if type(pad_token) is not int:
        raise WeightEditInputError("likelihood scoring requires a pad or EOS token")
    with deterministic_execution(loaded.model), torch.inference_mode():
        for start in range(0, len(interactions), batch_size):
            demos = [prepare_demonstration(renderer, interaction) for interaction in interactions[start:start + batch_size]]
            width = max(len(demo.rendered.token_ids) for demo in demos)
            ids = torch.full((len(demos), width), pad_token, dtype=torch.long, device=device)
            attention = torch.zeros_like(ids)
            for index, demo in enumerate(demos):
                ids[index, :len(demo.rendered.token_ids)] = torch.tensor(demo.rendered.token_ids, dtype=torch.long, device=device)
                attention[index, :len(demo.rendered.token_ids)] = 1
            logits = loaded.model(input_ids=ids, attention_mask=attention).logits
            for index, demo in enumerate(demos):
                positions = torch.tensor(demo.positions, dtype=torch.long, device=device)
                targets = torch.tensor([demo.labels[position + 1] for position in demo.positions], dtype=torch.long, device=device)
                logprobs = logits[index, positions].float().log_softmax(-1).gather(1, targets[:, None]).squeeze(1)
                case_sums.append(float(logprobs.sum()))
                case_means.append(float(logprobs.mean()))
                token_count += len(targets)
    return {"cases": len(interactions), "response_tokens": token_count,
            "mean_case_logprob": sum(case_means) / len(case_means),
            "mean_token_logprob": sum(case_sums) / token_count, "sum_logprob": sum(case_sums), "case_logprobs": case_sums}


def run_layer_strength_sweep(state_dir: Path, request_path: Path, output_path: Path, device="cuda:0"):
    request = parse_strict_json(request_path.read_bytes())
    if request["training_request_digest"] != _document_digest({key: value for key, value in request.items() if key != "training_request_digest"}):
        raise WeightEditInputError("layer sweep request digest is invalid")
    job = gym_job_from_document(parse_strict_json((state_dir / "jobs" / (request["job_digest"][7:] + ".json")).read_bytes()))
    task = reference_task(rows_per_template=job.scenario_sources[0].parameters["rows_per_template"])
    if job.digest != task.job.digest:
        raise WeightEditInputError("layer sweep differs from frozen task")
    service = GymStudioService(state_dir=state_dir)
    approved, train_examples = _load_training_rosters(service, request)
    examples = [{"example_digest": _document_digest(row), "interaction": task.project(row["input"], row["response"])}
                for row in approved]
    if _document_digest({"examples": examples}) != request["projected_examples_digest"]:
        raise WeightEditInputError("layer sweep examples differ from frozen training dataset")
    train_rows = [{"scenario_id": row["scenario_id"], "input": row["input"], "reference": row["response"]}
                  for row in train_examples]
    train_interactions = [task.project(row["input"], row["response"]) for row in train_examples]
    validation_rows = generate_scenarios(split="validation", seed=request["recipe"]["seed"],
                                         parameters=task.job.scenario_sources[0].parameters)
    validation_interactions = [task.project(row["input"], row["reference"]) for row in validation_rows]
    validation_by_template = _interactions_by_template(task, validation_rows)
    screening_rows = select_validation_screen(validation_rows)
    screening_interactions = [task.project(row["input"], row["reference"]) for row in screening_rows]
    loaded = load_model(request["base_reference"]["artifact_path"], dtype=torch.bfloat16, device=device)
    if loaded.base_digest != request["loaded_base"]["state_digest"]:
        raise WeightEditInputError("layer sweep base differs from frozen request")
    sites = find_sites(loaded.model, find_blocks(loaded.model), loaded.hidden_size,
                       checkpoint_tensor_names=loaded.checkpoint_tensor_names)
    writer_sites = _mlp_writer_site_map(sites)
    all_writers = list(writer_sites.sites)
    if len(all_writers) < LAYER_SWEEP_EXACT_LAYER_COUNT:
        raise WeightEditInputError("layer sweep has fewer MLP write sites than exact-scored finalists")
    writers = all_writers[len(all_writers) // 4:(3 * len(all_writers) + 3) // 4]
    base_digest = model_state_digest(loaded.model)
    strengths = LAYER_SWEEP_STRENGTHS
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise WeightEditInputError("layer-sweep report already exists; choose a fresh report path")
    checkpoint_path = output_path.with_suffix(".checkpoint.jsonl")
    run_digest = _document_digest({"kind": "nf_layer_strength_run_v2", "request_digest": request["training_request_digest"],
        "dataset_digest": request["dataset_digest"], "base_digest": base_digest,
        "layers": [site.layer for site in writers], "strengths": list(strengths),
        "generation_batch_size": 8, "generation_max_new_tokens": LAYER_SWEEP_MAX_NEW_TOKENS,
        "likelihood_batch_size": 4, "exact_layer_count": LAYER_SWEEP_EXACT_LAYER_COUNT})
    binding = {"run_digest": run_digest, "request_digest": request["training_request_digest"],
               "dataset_digest": request["dataset_digest"], "base_digest": base_digest,
               "layers": [site.layer for site in writers], "strengths": list(strengths)}
    records = open_sweep_checkpoint(checkpoint_path, binding)
    by_key = {row["checkpoint_key"]: row for row in records if row.get("checkpoint_key")}

    batch_baseline_train = score_public_rows(task, validation_model(loaded, batch_size=16,
        max_new_tokens=LAYER_SWEEP_MAX_NEW_TOKENS), train_rows, split="train")
    batch_baseline_validation = score_public_rows(task, validation_model(loaded, batch_size=16,
        max_new_tokens=LAYER_SWEEP_MAX_NEW_TOKENS), validation_rows)
    serial_baseline_validation = score_public_rows(task, validation_model(loaded, batch_size=1,
        max_new_tokens=LAYER_SWEEP_MAX_NEW_TOKENS), validation_rows)
    batch_equivalent = all(left["response"] == right["response"]
        and left.get("execution_evidence", {}).get("generated_token_ids") == right.get("execution_evidence", {}).get("generated_token_ids")
        for left, right in zip(batch_baseline_validation["responses"], serial_baseline_validation["responses"], strict=True))
    exact_batch_size = 16 if batch_equivalent else 1
    baseline_validation_score = batch_baseline_validation if batch_equivalent else serial_baseline_validation
    failing_examples = examples
    passing_interactions = list(request["recipe"].get("protected_training_interactions", ()))
    baseline_train = {"task": _compact_score(batch_baseline_train),
                      "expected_answer_logprob": expected_answer_logprob(loaded, train_interactions, batch_size=16)}
    baseline_validation_by_template = {family: expected_answer_logprob(loaded, interactions, batch_size=16)
                                       for family, interactions in validation_by_template.items()}
    baseline_validation = {"task": _compact_score(baseline_validation_score),
                           "task_by_template": _validation_template_scores(validation_rows, baseline_validation_score),
                           "expected_answer_logprob": _aggregate_logprobs(baseline_validation_by_template),
                           "expected_answer_logprob_by_template": baseline_validation_by_template}
    baseline_validation_screen_logprob = expected_answer_logprob(loaded, screening_interactions, batch_size=16)
    baseline_record = {"kind": "nf_layer_sweep_baseline_v2", "phase": "baseline", "run_digest": run_digest,
                       "training": baseline_train, "validation": baseline_validation,
                       "validation_screen_cases": len(screening_rows),
                       "validation_screen_logprob": baseline_validation_screen_logprob,
                       "batch_generation_equivalent_to_serial": batch_equivalent, "exact_batch_size": exact_batch_size}
    if "baseline" not in by_key:
        baseline_record["checkpoint_key"] = "baseline"
        append_sweep_checkpoint(checkpoint_path, baseline_record)
        by_key["baseline"] = baseline_record

    if not failing_examples:
        raise WeightEditInputError("base has no failing approved training examples to target")
    protected_interactions = list(request["recipe"]["preservation_interactions"]) + passing_interactions
    all_deltas = calculate_edits_for_sites(loaded, job, failing_examples, request["recipe"], sites=writer_sites,
                                           retention_interactions=protected_interactions)
    calculated_by_layer = {site.layer: (site, *all_deltas[site.tensor_name]) for site in writers}

    for site in writers:
        selected_strengths = [strength for strength in strengths
                              if f"screen:{site.layer}:{strength:.8g}" not in by_key]
        if not selected_strengths:
            continue
        selected, calculated, calculation = calculated_by_layer[site.layer]
        original = loaded.model.get_parameter(selected.tensor_name).detach().clone()
        try:
            for strength in selected_strengths:
                if model_state_digest(loaded.model) != base_digest:
                    raise WeightEditInputError("screen phase did not start from exact base")
                delta = Delta("low_rank", U=calculated.U * (strength / request["recipe"]["target_step"]), V=calculated.V)
                key = f"screen:{selected.layer}:{strength:.8g}"
                try:
                    edited = apply(loaded, {selected.tensor_name: delta}, sites=SiteMap((selected,), checkpoint_names_verified=True),
                        method="diagnostic_layer_strength_sweep", settings={"target_step": strength, "diagnostic_only": True})
                    train_logprob = expected_answer_logprob(loaded, train_interactions, batch_size=16)
                    validation_screen_logprob = expected_answer_logprob(loaded, screening_interactions, batch_size=16)
                    row = {"kind": "nf_layer_sweep_candidate_v2", "checkpoint_key": key, "phase": "screen",
                        "run_digest": run_digest, "layer": selected.layer, "tensor_name": selected.tensor_name,
                        "target_step": strength, "status": "measured", "training_logprob": train_logprob,
                        "validation_screen_logprob": validation_screen_logprob,
                        "validation_screen_cases": len(screening_rows),
                        "training_logprob_gain": train_logprob["mean_token_logprob"] - baseline_train["expected_answer_logprob"]["mean_token_logprob"],
                        "validation_screen_logprob_gain": validation_screen_logprob["mean_token_logprob"] - baseline_validation_screen_logprob["mean_token_logprob"],
                        "effective_change_norm": edited.receipt["change_norms"][selected.tensor_name],
                        "changed_tensors": edited.receipt["edited_tensors"],
                        "candidate_model_digest": model_state_digest(loaded.model), "calculation_evidence": calculation}
                except Exception as exc:
                    row = {"kind": "nf_layer_sweep_candidate_v2", "checkpoint_key": key, "phase": "screen",
                        "run_digest": run_digest, "layer": selected.layer, "tensor_name": selected.tensor_name,
                        "target_step": strength, "status": "failed", "failure_type": type(exc).__name__}
                finally:
                    with torch.no_grad():
                        loaded.model.get_parameter(selected.tensor_name).copy_(original)
                if model_state_digest(loaded.model) != base_digest:
                    raise WeightEditInputError("screen candidate failed exact base restoration")
                append_sweep_checkpoint(checkpoint_path, row)
                by_key[key] = row
        finally:
            with torch.no_grad():
                loaded.model.get_parameter(selected.tensor_name).copy_(original)

    screen_rows = [by_key[f"screen:{site.layer}:{strength:.8g}"] for site in writers for strength in strengths]
    halving_key = "halving1"
    if halving_key in by_key:
        refine_keys = by_key[halving_key]["selected_candidate_keys"]
    else:
        refine_keys = select_top_third(screen_rows)
        halving_record = {"kind": "nf_layer_sweep_halving_v2", "checkpoint_key": halving_key,
            "phase": "successive_halving", "run_digest": run_digest,
            "screen_cases": len(screening_rows), "screened_candidates": len(screen_rows),
            "refined_candidate_keys": refine_keys, "selection_metric": "public_validation_screen_logprob_gain"}
        append_sweep_checkpoint(checkpoint_path, halving_record)
        by_key[halving_key] = halving_record

    for candidate_key in refine_keys:
        refine_key = candidate_key.replace("screen:", "refine:", 1)
        if refine_key in by_key:
            continue
        screen = by_key[candidate_key]
        layer = screen["layer"]
        selected, calculated, calculation = calculated_by_layer[layer]
        original = loaded.model.get_parameter(selected.tensor_name).detach().clone()
        try:
            strength = screen["target_step"]
            if model_state_digest(loaded.model) != base_digest:
                raise WeightEditInputError("refinement candidate did not start from exact base")
            delta = Delta("low_rank", U=calculated.U * (strength / request["recipe"]["target_step"]), V=calculated.V)
            try:
                apply(loaded, {selected.tensor_name: delta}, sites=SiteMap((selected,), checkpoint_names_verified=True),
                      method="diagnostic_likelihood_refinement", settings={"target_step": strength, "diagnostic_only": True})
                validation_logprob = expected_answer_logprob(loaded, validation_interactions, batch_size=16)
                row = {"kind": "nf_layer_sweep_candidate_v2", "checkpoint_key": refine_key,
                    "phase": "refine", "run_digest": run_digest, "layer": layer,
                    "tensor_name": selected.tensor_name, "target_step": strength, "status": "measured",
                    "validation_logprob": validation_logprob,
                    "validation_logprob_gain": validation_logprob["mean_token_logprob"]
                        - baseline_validation["expected_answer_logprob"]["mean_token_logprob"],
                    "screen_validation_logprob_gain": screen["validation_screen_logprob_gain"]}
            except Exception as exc:
                row = {"kind": "nf_layer_sweep_candidate_v2", "checkpoint_key": refine_key,
                    "phase": "refine", "run_digest": run_digest, "layer": layer,
                    "tensor_name": selected.tensor_name, "target_step": strength,
                    "status": "failed", "failure_type": type(exc).__name__}
            finally:
                with torch.no_grad():
                    loaded.model.get_parameter(selected.tensor_name).copy_(original)
            if model_state_digest(loaded.model) != base_digest:
                raise WeightEditInputError("refinement candidate failed exact base restoration")
            append_sweep_checkpoint(checkpoint_path, row)
            by_key[refine_key] = row
        finally:
            with torch.no_grad():
                loaded.model.get_parameter(selected.tensor_name).copy_(original)

    refined_rows = [by_key[key.replace("screen:", "refine:", 1)] for key in refine_keys]
    layer_rank = []
    for site in writers:
        rows = [row for row in refined_rows if row["layer"] == site.layer and row.get("status") == "measured"]
        if rows:
            best_row = max(rows, key=lambda row: row["validation_logprob_gain"])
            layer_rank.append({"layer": site.layer, "tensor_name": site.tensor_name,
                               "best_strength": best_row["target_step"],
                               "best_validation_logprob_gain": best_row["validation_logprob_gain"]})
    layer_rank.sort(key=lambda row: (row["best_validation_logprob_gain"], -row["layer"]), reverse=True)
    selection_key = "selection"
    if selection_key in by_key:
        selected_layers = by_key[selection_key]["selected_layers"]
    else:
        selected_layers = [row["layer"] for row in layer_rank[:LAYER_SWEEP_EXACT_LAYER_COUNT]]
        selection = {"kind": "nf_layer_sweep_selection_v2", "checkpoint_key": selection_key,
            "phase": "selection", "run_digest": run_digest, "selection_metric": "full_public_validation_mean_token_logprob_gain",
            "selected_layers": selected_layers, "layer_ranking": layer_rank,
            "selection_adjusted": False, "interpretation": "Exploratory; selection uses public validation."}
        append_sweep_checkpoint(checkpoint_path, selection)
        by_key[selection_key] = selection

    preservation_panel = [*request["recipe"]["preservation_interactions"], *passing_interactions]
    layers_needing_exact = [layer for layer in selected_layers
        if any(f"exact:{layer}:{strength:.8g}" not in by_key for strength in strengths)]
    preservation_calibrations = {}
    preservation_before = []
    float32_preservation_before = []
    if layers_needing_exact:
        preservation_before = preservation_logits(loaded, preservation_panel)
        float32_preservation_before = float32_logits_of_exact_weights(loaded, preservation_panel)
        for layer in layers_needing_exact:
            site = next(site for site in writers if site.layer == layer)
            preservation_calibrations[layer] = calibrate_one_ulp(loaded, site, preservation_panel)

    for layer in selected_layers:
        selected, calculated, calculation = calculated_by_layer.get(layer, (None, None, None))
        site = next(site for site in writers if site.layer == layer)
        missing = [strength for strength in strengths if f"exact:{layer}:{strength:.8g}" not in by_key]
        if not missing:
            continue
        if calculated is None:
            raise WeightEditInputError("selected screen layer has no cached one-pass edit calculation")
        original = loaded.model.get_parameter(selected.tensor_name).detach().clone()
        try:
            for strength in missing:
                if model_state_digest(loaded.model) != base_digest:
                    raise WeightEditInputError("exact phase did not start from exact base")
                delta = Delta("low_rank", U=calculated.U * (strength / request["recipe"]["target_step"]), V=calculated.V)
                key = f"exact:{layer}:{strength:.8g}"
                try:
                    edited = apply(loaded, {selected.tensor_name: delta}, sites=SiteMap((selected,), checkpoint_names_verified=True),
                        method="diagnostic_layer_strength_sweep", settings={"target_step": strength, "diagnostic_only": True})
                    train_score = score_public_rows(task, validation_model(loaded, batch_size=16,
                        max_new_tokens=LAYER_SWEEP_MAX_NEW_TOKENS), train_rows, split="train")
                    validation_score = score_public_rows(task, validation_model(loaded, batch_size=exact_batch_size,
                        max_new_tokens=LAYER_SWEEP_MAX_NEW_TOKENS), validation_rows)
                    validation_logprob_by_template = {family: expected_answer_logprob(loaded, interactions, batch_size=16)
                        for family, interactions in validation_by_template.items()}
                    validation_logprob = _aggregate_logprobs(validation_logprob_by_template)
                    preservation_after = preservation_logits(loaded, preservation_panel)
                    float32_preservation_after = float32_logits_of_exact_weights(loaded, preservation_panel)
                    preservation_v2 = verify_v2(
                        metrics(float32_preservation_before, float32_preservation_after),
                        metrics(preservation_before, preservation_after), preservation_calibrations[layer])
                    screen = by_key[f"screen:{layer}:{strength:.8g}"]
                    row = {"kind": "nf_layer_sweep_candidate_v2", "checkpoint_key": key, "phase": "exact",
                        "run_digest": run_digest, "layer": layer, "tensor_name": selected.tensor_name,
                        "target_step": strength, "status": "measured", "training": _compact_score(train_score),
                        "validation": _compact_score(validation_score),
                        "validation_by_template": _validation_template_scores(validation_rows, validation_score),
                        "training_exact_match_gain": train_score["exact_match"] - baseline_train["task"]["exact_match"],
                        "validation_exact_match_gain": validation_score["exact_match"] - baseline_validation["task"]["exact_match"],
                        "training_logprob_gain": screen["training_logprob_gain"],
                        "validation_logprob_gain": validation_logprob["mean_token_logprob"]
                            - baseline_validation["expected_answer_logprob"]["mean_token_logprob"],
                        "validation_logprob": validation_logprob,
                        "validation_logprob_by_template": {family: {
                            "cases": report["cases"],
                            "baseline_mean_token_logprob": baseline_validation_by_template[family]["mean_token_logprob"],
                            "mean_token_logprob": report["mean_token_logprob"],
                            "mean_token_logprob_gain": report["mean_token_logprob"]
                                - baseline_validation_by_template[family]["mean_token_logprob"]}
                            for family, report in validation_logprob_by_template.items()},
                        "preservation_v2_report_only": preservation_v2,
                        "preservation_panel_cases": len(preservation_panel),
                        "preservation_calibration_digest": preservation_calibrations[layer]["calibration_digest"],
                        "effective_change_norm": edited.receipt["change_norms"][selected.tensor_name],
                        "changed_tensors": edited.receipt["edited_tensors"],
                        "candidate_model_digest": model_state_digest(loaded.model), "calculation_evidence": calculation}
                except Exception as exc:
                    row = {"kind": "nf_layer_sweep_candidate_v2", "checkpoint_key": key, "phase": "exact",
                        "run_digest": run_digest, "layer": layer, "tensor_name": selected.tensor_name,
                        "target_step": strength, "status": "failed", "failure_type": type(exc).__name__}
                finally:
                    with torch.no_grad():
                        loaded.model.get_parameter(selected.tensor_name).copy_(original)
                if model_state_digest(loaded.model) != base_digest:
                    raise WeightEditInputError("exact candidate failed exact base restoration")
                append_sweep_checkpoint(checkpoint_path, row)
                by_key[key] = row
        finally:
            with torch.no_grad():
                loaded.model.get_parameter(selected.tensor_name).copy_(original)
    if model_state_digest(loaded.model) != base_digest:
        raise WeightEditInputError("resumed layer sweep failed exact base restoration")

    exact_rows = [by_key.get(f"exact:{layer}:{strength:.8g}") for layer in selected_layers for strength in strengths]
    exact_measured = [row for row in exact_rows if row and row.get("status") == "measured"]
    if not exact_measured:
        raise WeightEditInputError("likelihood screen produced no exact-match candidate")
    best = max(exact_measured, key=lambda row: (row["validation"]["exact_match"], row["validation_logprob_gain"]))
    strength_one_rows = [row for row in screen_rows if row["target_step"] == 1.0]
    best_strength_one = next((row for row in strength_one_rows if row["layer"] == best["layer"]), None)
    strength_one_gate = strength_one_stop_gate(best["layer"], by_key.get(f"exact:{best['layer']}:1"))
    report = {"kind": "nf_weight_edit_layer_strength_diagnostic_v2", "evidence_status": "measured",
              "selection_split": "public_validation", "task_scorer": "frozen_exact_match_v1",
              "job_digest": job.digest, "training_request_digest": request["training_request_digest"],
              "dataset_digest": request["dataset_digest"], "base_model_digest": base_digest,
              "failing_training_examples": len(failing_examples),
              "training_scoring_cases": len(train_rows),
              "passing_training_examples_protected": len(passing_interactions),
              "base_train": baseline_train, "base_validation": baseline_validation,
              "generation_max_new_tokens": LAYER_SWEEP_MAX_NEW_TOKENS,
              "strengths": list(strengths), "discovered_layer_count": len(all_writers),
              "screened_layer_indices": [site.layer for site in writers], "exact_scored_layer_indices": selected_layers,
              "screen_candidate_count": len(screen_rows), "exact_candidate_count": len(exact_rows),
              "batch_generation_equivalent_to_serial": batch_equivalent, "exact_batch_size": exact_batch_size,
              "layer_ranking_by_validation_logprob": layer_rank, "results": screen_rows + [row for row in exact_rows if row],
              "best_by_validation_exact_match_then_expected_answer_logprob": {"layer": best["layer"],
                  "tensor_name": best["tensor_name"], "target_step": best["target_step"],
                  "validation_exact_match": best["validation"]["exact_match"],
                  "validation_logprob_gain": best["validation_logprob_gain"]},
              "strength_1_effective_change": {"all_screened_sites_nonzero": bool(strength_one_rows) and all(
                  row.get("status") == "measured" and row.get("effective_change_norm", 0) > 0 for row in strength_one_rows),
                  "best_layer_nonzero": bool(best_strength_one and best_strength_one.get("status") == "measured"
                                              and best_strength_one.get("effective_change_norm", 0) > 0)},
              "strength_1_stop_gate": strength_one_gate,
              "checkpoint_path": str(checkpoint_path), "run_digest": run_digest,
              "base_restored": True, "specialist_policy_approval": "pending",
              "candidate_published": False, "holdout_accessed": False, "release_authorized": False}
    report["generation_truncation"] = {
        "baseline_train": baseline_train["task"]["truncated_answer_count"],
        "baseline_validation": baseline_validation["task"]["truncated_answer_count"],
        "exact_candidates": sum(row["training"]["truncated_answer_count"] + row["validation"]["truncated_answer_count"]
                                for row in exact_measured),
        "generation_cap_failures": (baseline_train["task"]["generation_cap_failure_count"]
            + baseline_validation["task"]["generation_cap_failure_count"]
            + sum(row["training"]["generation_cap_failure_count"] + row["validation"]["generation_cap_failure_count"]
                  for row in exact_measured)),
        "answer_count": (baseline_train["task"]["cases"] + baseline_validation["task"]["cases"]
            + sum(row["training"]["cases"] + row["validation"]["cases"] for row in exact_measured)),
        "cap_hits_are_failures": True,
    }
    report["generation_truncation"]["total_truncated_answers"] = (
        report["generation_truncation"]["baseline_train"]
        + report["generation_truncation"]["baseline_validation"]
        + report["generation_truncation"]["exact_candidates"])
    report["generation_truncation"]["zero_truncated_answers"] = (
        report["generation_truncation"]["total_truncated_answers"] == 0)
    report["evidence_digest"] = _document_digest(report)
    with output_path.open("xb") as stream:
        stream.write(_wire_document(report))
    if device.startswith("cuda"):
        loaded.model.to("cpu")
        torch.cuda.empty_cache()
    return report


def load_sweep_likelihood_selection(source_path: Path, request: dict[str, Any]) -> dict[str, Any]:
    raw = source_path.read_bytes()
    if source_path.suffix == ".jsonl":
        lines = [line for line in raw.splitlines() if line]
        if not lines:
            raise WeightEditInputError("sweep checkpoint is empty")
        binding = parse_strict_json(lines[0])
        records = [parse_strict_json(line) for line in lines[1:]]
        if (binding.get("kind") != "nf_layer_sweep_binding_v2"
            or binding.get("request_digest") != request.get("training_request_digest")
            or binding.get("dataset_digest") != request.get("dataset_digest")
            or any(row.get("run_digest") != binding.get("run_digest") for row in records)):
            raise WeightEditInputError("sweep checkpoint binding differs from the finalist request")
        by_key = {row.get("checkpoint_key"): row for row in records if row.get("checkpoint_key")}
        baseline, selection = by_key.get("baseline"), by_key.get("selection")
        expected_screen = {f"screen:{layer}:{strength:.8g}" for layer in binding["layers"]
                           for strength in binding["strengths"]}
        if (baseline is None or baseline.get("kind") != "nf_layer_sweep_baseline_v2"
            or selection is None or selection.get("kind") != "nf_layer_sweep_selection_v2"
            or not selection.get("selected_layers")
            or not expected_screen <= set(by_key)):
            raise WeightEditInputError("sweep checkpoint lacks a complete likelihood screen and selection")
        source = {"kind": "nf_layer_strength_screen_selection_v1",
            "training_request_digest": binding["request_digest"], "job_digest": request["job_digest"],
            "dataset_digest": binding["dataset_digest"], "base_model_digest": binding["base_digest"],
            "run_digest": binding["run_digest"], "layer_ranking_by_validation_logprob": selection["layer_ranking"],
            "exact_scored_layer_indices": selection["selected_layers"], "strengths": binding["strengths"],
            "holdout_accessed": False,
            "source_checkpoint_digest": _document_digest({"binding": binding, "records": records})}
        source["evidence_digest"] = _document_digest(source)
        return source
    source = parse_strict_json(raw)
    if (source.get("kind") != "nf_weight_edit_layer_strength_diagnostic_v2"
        or source.get("evidence_digest") != _document_digest({key: value for key, value in source.items()
                                                              if key != "evidence_digest"})):
        raise WeightEditInputError("finalist source sweep report digest is invalid")
    return source


def fixed_case_breakdown(base_score, candidate_score):
    before = base_score["responses"]
    after = candidate_score["responses"]
    if [row["scenario_id"] for row in before] != [row["scenario_id"] for row in after]:
        raise WeightEditInputError("fixed-case audit requires identical ordered scenario rosters")
    repaired_format, corrected_values = [], []
    for base, candidate in zip(before, after, strict=True):
        if base["exact_match"] == 0 and candidate["exact_match"] == 1:
            if not candidate["format_valid"]:
                raise WeightEditInputError("fixed-case audit found an exact answer with invalid format")
            destination = corrected_values if base["format_valid"] else repaired_format
            destination.append(base["scenario_id"])
    return {"format_invalid_now_valid": len(repaired_format),
            "format_valid_value_corrected": len(corrected_values),
            "format_invalid_now_valid_ids": repaired_format,
            "format_valid_value_corrected_ids": corrected_values,
            "interpretation": "paired schema-validity categories, not a causal attribution; format-invalid cases may also repair values"}


def run_edit_key_audit(state_dir: Path, request_path: Path, output_path: Path, *, device="cuda:0", layers=(10, 9)):
    request = parse_strict_json(request_path.read_bytes())
    if request.get("training_request_digest") != _document_digest(
        {key: value for key, value in request.items() if key != "training_request_digest"}):
        raise WeightEditInputError("key audit request digest is invalid")
    job = gym_job_from_document(parse_strict_json(
        (state_dir / "jobs" / (request["job_digest"][7:] + ".json")).read_bytes()))
    task = reference_task(rows_per_template=job.scenario_sources[0].parameters["rows_per_template"])
    if task.job.digest != job.digest or output_path.exists():
        raise WeightEditInputError("key audit requires the frozen task and a fresh output")
    service = GymStudioService(state_dir=state_dir)
    approved, parent = _load_training_rosters(service, request)
    examples = [{"example_digest": _document_digest(row), "interaction": task.project(row["input"], row["response"])}
                for row in approved]
    if _document_digest({"examples": examples}) != request["projected_examples_digest"]:
        raise WeightEditInputError("key audit demonstrations differ from the frozen request")
    loaded = load_model(request["base_reference"]["artifact_path"], dtype=torch.bfloat16, device=device)
    before = model_state_digest(loaded.model)
    if before != request["loaded_base"]["state_digest"]:
        raise WeightEditInputError("key audit belongs to another base state")
    discovered = find_sites(loaded.model, find_blocks(loaded.model), loaded.hidden_size,
                            checkpoint_tensor_names=loaded.checkpoint_tensor_names)
    selected = tuple(site for site in _mlp_writer_site_map(discovered).sites if site.layer in layers)
    if len(selected) != len(set(layers)):
        raise WeightEditInputError("key audit layers are not uniquely discovered MLP writers")
    protected = [*request["recipe"]["preservation_interactions"],
                 *request["recipe"].get("protected_training_interactions", ())]
    observations = calculate_edits_for_sites(loaded, job, examples, request["recipe"],
        sites=SiteMap(selected, checkpoint_names_verified=True), retention_interactions=protected, diagnostics_only=True)
    if any(delta is not None for delta, evidence in observations.values()):
        raise WeightEditInputError("read-only key audit unexpectedly produced an edit")
    if model_state_digest(loaded.model) != before:
        raise WeightEditInputError("read-only key audit changed model state")
    report = {"kind": "nf_read_only_edit_key_audit_v1", "evidence_status": "measured",
        "training_request_digest": request["training_request_digest"], "job_digest": job.digest,
        "dataset_digest": request["dataset_digest"], "base_state_digest": before,
        "failing_examples": len(examples), "parent_train_examples": len(parent),
        "retention_interactions": len(protected),
        "per_layer": {str(site.layer): observations[site.tensor_name][1]["key_diagnostics"] for site in selected},
        "demonstration_format": "compact sorted JSON with tokenizer EOS appended; no markdown fence or indentation",
        "tokenizer_eos": {"token": loaded.tokenizer.eos_token, "id": loaded.tokenizer.eos_token_id},
        "chat_template_digest": _document_digest({"chat_template": loaded.tokenizer.chat_template}),
        "solver_called": False, "edit_applied": False, "base_unchanged": True,
        "holdout_accessed": False, "candidate_published": False}
    report["evidence_digest"] = _document_digest(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as stream:
        stream.write(_wire_document(report))
    if device.startswith("cuda"):
        loaded.model.to("cpu")
        torch.cuda.empty_cache()
    return report


def run_finalist_rescore(state_dir: Path, request_path: Path, screen_report_path: Path, output_path: Path,
                         *, device="cuda:0", composition_sizes=(2, 3), single_layer_candidates=None):
    request = parse_strict_json(request_path.read_bytes())
    if request.get("training_request_digest") != _document_digest(
        {key: value for key, value in request.items() if key != "training_request_digest"}):
        raise WeightEditInputError("finalist request digest is invalid")
    source = load_sweep_likelihood_selection(screen_report_path, request)
    if (source.get("training_request_digest") != request["training_request_digest"]
        or source.get("dataset_digest") != request["dataset_digest"]
        or source.get("holdout_accessed") is not False):
        raise WeightEditInputError("finalist source sweep is invalid or bound to another request")
    if (not isinstance(composition_sizes, (list, tuple)) or not composition_sizes
        or any(type(size) is not int or size < 2 for size in composition_sizes)):
        raise WeightEditInputError("finalist compositions require bounded layer counts of at least two")
    job = gym_job_from_document(parse_strict_json(
        (state_dir / "jobs" / (request["job_digest"][7:] + ".json")).read_bytes()))
    task = reference_task(rows_per_template=job.scenario_sources[0].parameters["rows_per_template"])
    if job.digest != task.job.digest:
        raise WeightEditInputError("finalist rescore differs from frozen task")
    service = GymStudioService(state_dir=state_dir)
    approved, train_examples = _load_training_rosters(service, request)
    examples = [{"example_digest": _document_digest(row),
                 "interaction": task.project(row["input"], row["response"])} for row in approved]
    if _document_digest({"examples": examples}) != request["projected_examples_digest"]:
        raise WeightEditInputError("finalist edit examples differ from frozen selected dataset")
    train_rows = [{"scenario_id": row["scenario_id"], "input": row["input"], "reference": row["response"]}
                  for row in train_examples]
    train_interactions = [task.project(row["input"], row["response"]) for row in train_examples]
    validation_rows = generate_scenarios(split="validation", seed=request["recipe"]["seed"],
        parameters=task.job.scenario_sources[0].parameters)
    validation_by_template = _interactions_by_template(task, validation_rows)
    validation_interactions = [interaction for rows in validation_by_template.values() for interaction in rows]
    required_rows = public_required_rows(seed=request["recipe"]["seed"])
    if output_path.exists():
        raise WeightEditInputError("finalist report already exists; choose a fresh path")
    loaded = load_model(request["base_reference"]["artifact_path"], dtype=torch.bfloat16, device=device)
    if loaded.base_digest != request["loaded_base"]["state_digest"] or source.get("base_model_digest") != loaded.base_digest:
        raise WeightEditInputError("finalist sweep and request identify different base states")
    all_sites = find_sites(loaded.model, find_blocks(loaded.model), loaded.hidden_size,
                           checkpoint_tensor_names=loaded.checkpoint_tensor_names)
    writers = _mlp_writer_site_map(all_sites)
    site_by_layer = {site.layer: site for site in writers.sites}
    ranking = source.get("layer_ranking_by_validation_logprob")
    selected_layers = source.get("exact_scored_layer_indices")
    if not isinstance(ranking, list) or not isinstance(selected_layers, list) or not selected_layers:
        raise WeightEditInputError("source sweep has no likelihood-selected layer ranking")
    selected_set = set(selected_layers)
    finalists = [row for row in ranking if row.get("layer") in selected_set]
    finalists.sort(key=lambda row: ranking.index(row))
    if (len(finalists) != len(selected_set) or any(row["layer"] not in site_by_layer for row in finalists)
        or any(row.get("best_strength") not in source.get("strengths", ()) for row in finalists)):
        raise WeightEditInputError("source sweep finalists lack verified layers or measured strengths")
    single_layer_mode = single_layer_candidates is not None
    if single_layer_mode:
        if (not single_layer_candidates or len(single_layer_candidates) > 6
            or len(set(single_layer_candidates)) != len(single_layer_candidates)
            or any(layer not in selected_set or strength not in (0.1, 0.5, 1.0)
                   for layer, strength in single_layer_candidates)):
            raise WeightEditInputError("single-layer candidates must be unique selected layers at approved strengths")
        requested_layers = {layer for layer, strength in single_layer_candidates}
        finalists = [row for row in finalists if row["layer"] in requested_layers]
    finalist_sites = tuple(site_by_layer[row["layer"]] for row in finalists)
    finalist_site_map = SiteMap(finalist_sites, checkpoint_names_verified=True)
    protected = [*request["recipe"]["preservation_interactions"],
                 *request["recipe"].get("protected_training_interactions", ())]
    calculated = calculate_edits_for_sites(loaded, job, examples, request["recipe"], sites=finalist_site_map,
                                           retention_interactions=protected)
    selected_config = [{"layer": row["layer"], "tensor_name": row["tensor_name"],
                        "target_step": row["best_strength"],
                        "validation_logprob_gain": row["best_validation_logprob_gain"]} for row in finalists]
    base_digest = model_state_digest(loaded.model)
    train_batch_size = 1 if single_layer_mode else 16
    base_train = score_public_rows(task, validation_model(loaded, batch_size=train_batch_size,
        max_new_tokens=LAYER_SWEEP_MAX_NEW_TOKENS), train_rows, split="train")
    baseline_validation_batch = score_public_rows(task, validation_model(loaded, batch_size=train_batch_size,
        max_new_tokens=LAYER_SWEEP_MAX_NEW_TOKENS), validation_rows)
    serial_validation = score_public_rows(task, validation_model(loaded, batch_size=1,
        max_new_tokens=LAYER_SWEEP_MAX_NEW_TOKENS), validation_rows)
    batch_equivalent = all(left["response"] == right["response"]
        and left.get("execution_evidence", {}).get("generated_token_ids") == right.get("execution_evidence", {}).get("generated_token_ids")
        for left, right in zip(baseline_validation_batch["responses"], serial_validation["responses"], strict=True))
    exact_batch_size = 1 if single_layer_mode else (16 if batch_equivalent else 1)
    base_validation = baseline_validation_batch if batch_equivalent else serial_validation
    base_required = score_public_rows(task, validation_model(loaded, batch_size=exact_batch_size,
        max_new_tokens=LAYER_SWEEP_MAX_NEW_TOKENS), required_rows)
    base_train_logprob = expected_answer_logprob(loaded, train_interactions, batch_size=16)
    base_validation_by_template = {family: expected_answer_logprob(loaded, interactions, batch_size=16)
                                   for family, interactions in validation_by_template.items()}
    base_validation_logprob = _aggregate_logprobs(base_validation_by_template)
    preservation_panel = protected
    preservation_before = preservation_logits(loaded, preservation_panel)
    float32_before = float32_logits_of_exact_weights(loaded, preservation_panel)
    calibrations = {row["layer"]: calibrate_one_ulp(loaded, site_by_layer[row["layer"]], preservation_panel)
                    for row in finalists}
    candidates = []
    if single_layer_mode:
        by_layer = {row["layer"]: row for row in finalists}
        compositions = [[{**by_layer[layer], "best_strength": strength}]
                        for layer, strength in single_layer_candidates]
    else:
        compositions = [finalists[:size] for size in composition_sizes if size <= len(finalists)]
    for composition in compositions:
        composition_size = len(composition)
        sites = tuple(site_by_layer[row["layer"]] for row in composition)
        deltas = {}
        calculation_diagnostics = {}
        for row in composition:
            site = site_by_layer[row["layer"]]
            delta, evidence = calculated[site.tensor_name]
            scale = row["best_strength"] / request["recipe"]["target_step"]
            deltas[site.tensor_name] = Delta("low_rank", U=delta.U * scale, V=delta.V)
            key_diagnostics = evidence["key_diagnostics"]
            output_norm = key_diagnostics["mean_write_output_l2_norm"]
            input_norm = key_diagnostics["mean_key_input_l2_norm"]
            calculation_diagnostics[str(row["layer"])] = {**key_diagnostics,
                "direction_capture_target_step": request["recipe"]["target_step"],
                "target_shift_l2": row["best_strength"],
                "target_shift_fraction_of_mean_write_output_norm": row["best_strength"] / output_norm if output_norm > 0 else None,
                "target_shift_fraction_of_mean_key_input_norm": row["best_strength"] / input_norm if input_norm > 0 else None,
                "gradient_recomputed_after_scaling": False}
        originals = {site.tensor_name: loaded.model.get_parameter(site.tensor_name).detach().clone()
                     for site in sites}
        try:
            edited = apply(loaded, deltas, sites=SiteMap(sites, checkpoint_names_verified=True),
                method="single_layer_stop_gate" if single_layer_mode else "multi_layer_finalist_rescore", settings={
                    "layers": [row["layer"] for row in composition],
                    "target_steps": {str(row["layer"]): row["best_strength"] for row in composition},
                    "retention_mode": "specialist", "diagnostic_only": True})
            train_score = score_public_rows(task, validation_model(loaded, batch_size=train_batch_size,
                max_new_tokens=LAYER_SWEEP_MAX_NEW_TOKENS), train_rows, split="train")
            validation_score = score_public_rows(task, validation_model(loaded, batch_size=exact_batch_size,
                max_new_tokens=LAYER_SWEEP_MAX_NEW_TOKENS), validation_rows)
            required_score = score_public_rows(task, validation_model(loaded, batch_size=exact_batch_size,
                max_new_tokens=LAYER_SWEEP_MAX_NEW_TOKENS), required_rows)
            train_logprob = expected_answer_logprob(loaded, train_interactions, batch_size=16)
            validation_logprob_by_template = {family: expected_answer_logprob(loaded, interactions, batch_size=16)
                for family, interactions in validation_by_template.items()}
            validation_logprob = _aggregate_logprobs(validation_logprob_by_template)
            preservation_after = preservation_logits(loaded, preservation_panel)
            float32_after = float32_logits_of_exact_weights(loaded, preservation_panel)
            preservation_reports = {str(row["layer"]): verify_v2(
                metrics(float32_before, float32_after), metrics(preservation_before, preservation_after),
                calibrations[row["layer"]]) for row in composition}
            preservation_passed = all(row["passed"] for row in preservation_reports.values())
            required_gate = {**required_score,
                "format_valid": min(required_score["format_valid"], validation_score["format_valid"])}
            specialist_gate = assess_specialist_v2({"split": "validation",
                "base_case_outcomes": [row["exact_match"] for row in base_validation["responses"]],
                "candidate_case_outcomes": [row["exact_match"] for row in validation_score["responses"]]},
                required_gate)
            validation_truncations = sum(score["truncated_answer_count"] for score in
                (_compact_score(base_validation), _compact_score(base_required),
                 _compact_score(validation_score), _compact_score(required_score)))
            if not single_layer_mode:
                specialist_gate["checks"]["zero_short_truncation"] = validation_truncations == 0
                specialist_gate["passed"] = specialist_gate["passed"] and validation_truncations == 0
            base_records = base_validation["responses"]
            candidate_records = validation_score["responses"]
            if [row["scenario_id"] for row in base_records] != [row["scenario_id"] for row in candidate_records]:
                raise WeightEditInputError("candidate validation roster differs from baseline")
            fixed = sum(before["exact_match"] == 0 and after["exact_match"] == 1
                        for before, after in zip(base_records, candidate_records, strict=True))
            broken = sum(before["exact_match"] == 1 and after["exact_match"] == 0
                         for before, after in zip(base_records, candidate_records, strict=True))
            if [row["scenario_id"] for row in base_train["responses"]] != [row["scenario_id"] for row in train_score["responses"]]:
                raise WeightEditInputError("candidate train roster differs from baseline")
            train_fixed = sum(before["exact_match"] == 0 and after["exact_match"] == 1
                for before, after in zip(base_train["responses"], train_score["responses"], strict=True))
            train_broken = sum(before["exact_match"] == 1 and after["exact_match"] == 0
                for before, after in zip(base_train["responses"], train_score["responses"], strict=True))
            preservation_checks = {}
            for layer, verification in preservation_reports.items():
                values = (("float32_max_drift", verification["float32"]["max_absolute_logit_drift"],
                           POLICY_V2["max_float32_logit_drift"], "<="),
                          ("bf16_top1", verification["bf16"]["top1_agreement"], POLICY_V2["min_bf16_top1_agreement"], ">="),
                          ("bf16_top5", verification["bf16"]["top5_set_agreement"], POLICY_V2["min_bf16_top5_set_agreement"], ">="),
                          ("bf16_mean_kl", verification["bf16"]["mean_kl_base_to_edited"],
                           POLICY_V2["max_mean_bf16_kl_base_to_edited"], "<="))
                preservation_checks[layer] = {name: {"value": value, "threshold": threshold, "comparison": comparison,
                    "passed": value <= threshold if comparison == "<=" else value >= threshold}
                    for name, value, threshold, comparison in values}
            candidate = {"kind": "nf_single_layer_finalist_v1" if single_layer_mode else "nf_multilayer_finalist_v1",
                "status": "measured", "composition_size": composition_size,
                "layers": [row["layer"] for row in composition],
                "target_steps": {str(row["layer"]): row["best_strength"] for row in composition},
                "training": _compact_score(train_score), "validation": _compact_score(validation_score),
                "validation_by_template": _validation_template_scores(validation_rows, validation_score),
                "required_set": _compact_score(required_score),
                "training_exact_match_gain": train_score["exact_match"] - base_train["exact_match"],
                "validation_exact_match_gain": validation_score["exact_match"] - base_validation["exact_match"],
                "validation_fixed_cases": fixed, "validation_broken_cases": broken,
                "validation_fixed_breakdown": fixed_case_breakdown(base_validation, validation_score),
                "training_fixed_cases": train_fixed, "training_broken_cases": train_broken,
                "edit_key_diagnostics": calculation_diagnostics,
                "training_logprob_gain": train_logprob["mean_token_logprob"] - base_train_logprob["mean_token_logprob"],
                "validation_logprob_gain": validation_logprob["mean_token_logprob"] - base_validation_logprob["mean_token_logprob"],
                "validation_logprob_by_template": {family: {
                    "cases": value["cases"],
                    "baseline_mean_token_logprob": base_validation_by_template[family]["mean_token_logprob"],
                    "mean_token_logprob": value["mean_token_logprob"],
                    "mean_token_logprob_gain": value["mean_token_logprob"]
                        - base_validation_by_template[family]["mean_token_logprob"]}
                    for family, value in validation_logprob_by_template.items()},
                "preservation_v2_report_only": {"passed": preservation_passed, "per_site": preservation_reports},
                "preservation_v2_checks": preservation_checks,
                "specialist_v2_gate": specialist_gate,
                "validation_truncated_answer_count": validation_truncations,
                "effective_change_norms": edited.receipt["change_norms"],
                "stored_weight_changes": edited.receipt["stored_weight_changes"],
                "candidate_model_digest": model_state_digest(loaded.model),
                "candidate_published": False}
            candidates.append(candidate)
            if single_layer_mode:
                site = sites[0]
                change = edited.receipt["stored_weight_changes"][site.tensor_name]
                candidate["layer"] = site.layer
                candidate["target_step"] = composition[0]["best_strength"]
                candidate["effective_change"] = change["effective_change"]
                candidate["effective_change_norm"] = change["frobenius_norm"]
        finally:
            with torch.no_grad():
                for name, original in originals.items():
                    loaded.model.get_parameter(name).copy_(original)
            if model_state_digest(loaded.model) != base_digest:
                raise WeightEditInputError("multi-layer finalist rescore failed exact base restoration")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"kind": "nf_weight_edit_single_layer_stop_gate_v1" if single_layer_mode
                       else "nf_weight_edit_multilayer_finalist_rescore_v1", "evidence_status": "measured",
        "source_sweep_report_digest": source["evidence_digest"], "source_sweep_run_digest": source["run_digest"],
        "training_request_digest": request["training_request_digest"], "job_digest": job.digest,
        "dataset_digest": request["dataset_digest"], "base_model_digest": base_digest,
        "selection_source": "fixed user-requested single-layer grid" if single_layer_mode
                    else "teacher_forced public-validation likelihood only; prior exact-match results excluded",
        "selected_layer_strengths": selected_config, "generation_max_new_tokens": LAYER_SWEEP_MAX_NEW_TOKENS,
        "batch_generation_equivalent_to_serial": batch_equivalent, "exact_batch_size": exact_batch_size,
        "base_train": _compact_score(base_train), "base_validation": _compact_score(base_validation),
        "base_required_set": _compact_score(base_required), "base_validation_logprob": base_validation_logprob,
        "candidates": candidates, "base_restored": model_state_digest(loaded.model) == base_digest,
        "holdout_accessed": False, "candidate_published": False, "release_authorized": False}
    if single_layer_mode:
        if len(candidates) != len(single_layer_candidates) or any(row["composition_size"] != 1 for row in candidates):
            raise WeightEditInputError("single-layer job did not produce its exact requested grid")
        report["single_layer_candidate_grid"] = [{"layer": layer, "target_step": strength}
                                                  for layer, strength in single_layer_candidates]
        best = max(candidates, key=lambda row: (row["validation"]["exact_match"], row["validation_logprob_gain"]))
        strength_one = next(row for row in candidates if row["layer"] == best["layer"] and row["target_step"] == 1.0)
        report["strength_1_stop_gate"] = {**strength_one_stop_gate(best["layer"], strength_one),
            "stored_weight_changes": strength_one["stored_weight_changes"],
            "effective_change": strength_one["effective_change"], "automatic_followup_authorized": False}
        report["generation_batch_size"] = 1
        report["reference_baseline_counts"] = {"train": 165, "validation": 45}
        report["reference_baseline_matches"] = (sum(row["exact_match"] for row in base_train["responses"]) == 165
            and sum(row["exact_match"] for row in base_validation["responses"]) == 45)
    answer_counts = [report["base_train"], report["base_validation"], report["base_required_set"]]
    for candidate in candidates:
        answer_counts.extend((candidate["training"], candidate["validation"], candidate["required_set"]))
    report["generation_truncation"] = {
        "generation_cap_failures": sum(row["generation_cap_failure_count"] for row in answer_counts),
        "truncated_answers": sum(row["truncated_answer_count"] for row in answer_counts),
        "zero_truncated_answers": all(row["truncated_answer_count"] == 0 for row in answer_counts),
        "cap_hits_are_failures": True}
    report["evidence_digest"] = _document_digest(report)
    with output_path.open("xb") as stream:
        stream.write(_wire_document(report))
    if device.startswith("cuda"):
        loaded.model.to("cpu")
        torch.cuda.empty_cache()
    return report


def audit_generation_truncation(report_path: Path, output_path: Path, *, max_new_tokens=LAYER_SWEEP_MAX_NEW_TOKENS):
    report = parse_strict_json(report_path.read_bytes())
    if (report.get("kind") != "nf_weight_edit_layer_strength_diagnostic_v2"
        or report.get("evidence_digest") != _document_digest({key: value for key, value in report.items()
                                                              if key != "evidence_digest"})
        or type(max_new_tokens) is not int or max_new_tokens < 1):
        raise WeightEditInputError("truncation audit requires an intact layer-sweep report and positive token budget")
    panels = [("baseline_train", report["base_train"]["task"]),
              ("baseline_validation", report["base_validation"]["task"])]
    for row in report["results"]:
        if row.get("phase") == "exact" and row.get("status") == "measured":
            panels.extend(((f"layer_{row['layer']}_strength_{row['target_step']}_train", row["training"]),
                           (f"layer_{row['layer']}_strength_{row['target_step']}_validation", row["validation"])))
    records = []
    for name, score in panels:
        responses = score.get("responses")
        if not isinstance(responses, list) or len(responses) != score.get("cases"):
            raise WeightEditInputError("truncation audit found an incomplete scored response roster")
        token_counts = []
        truncated, cap_failures = 0, 0
        for response in responses:
            evidence = response.get("execution_evidence")
            token_ids = evidence.get("generated_token_ids") if isinstance(evidence, dict) else None
            raw_text = evidence.get("raw_text") if isinstance(evidence, dict) else None
            if not isinstance(token_ids, list) or not isinstance(raw_text, str):
                raise WeightEditInputError("truncation audit requires raw text and generated token IDs")
            token_counts.append(len(token_ids))
            cap_failures += int(len(token_ids) >= max_new_tokens)
            truncated += int(_completion_was_truncated(token_ids, raw_text, max_new_tokens))
        records.append({"panel": name, "cases": len(responses), "max_generated_tokens": max(token_counts, default=0),
                        "at_generation_limit": sum(count >= max_new_tokens for count in token_counts),
                        "generation_cap_failures": cap_failures, "truncated_answers": truncated})
    truncated_count = sum(row["truncated_answers"] for row in records)
    cap_failure_count = sum(row["generation_cap_failures"] for row in records)
    audit = {"kind": "nf_generation_truncation_audit_v1", "source_report_digest": report["evidence_digest"],
             "run_digest": report["run_digest"], "max_new_tokens": max_new_tokens,
             "panels": records, "total_scored_answers": sum(row["cases"] for row in records),
             "total_truncated_answers": truncated_count, "zero_truncated_answers": truncated_count == 0,
             "generation_cap_failures": cap_failure_count,
             "holdout_accessed": False, "candidate_published": False}
    audit["evidence_digest"] = _document_digest(audit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as stream:
        stream.write(_wire_document(audit))
    return audit


def compare_logits(before: list[torch.Tensor], after: list[torch.Tensor]) -> dict[str, Any]:
    if not before or len(before) != len(after):
        raise WeightEditInputError("drift measurement requires a complete matching panel")
    max_error, max_kl, top_agreement, topk_agreement = 0.0, 0.0, 0, 0
    for first, second in zip(before, after, strict=True):
        if first.shape != second.shape or first.ndim != 1 or not first.numel() or not bool(torch.isfinite(first).all()) or not bool(torch.isfinite(second).all()):
            raise WeightEditInputError("drift logits must be finite complete vocabulary vectors")
        first, second = first.double(), second.double()
        max_error = max(max_error, float((first - second).abs().max()))
        log_first, log_second = first.log_softmax(-1), second.log_softmax(-1)
        max_kl = max(max_kl, max(0.0, float((log_first.exp() * (log_first - log_second)).sum())))
        top_agreement += int(first.argmax() == second.argmax())
        count = min(5, first.numel())
        topk_agreement += int(set(first.topk(count).indices.tolist()) == set(second.topk(count).indices.tolist()))
    return {"cases": len(before), "max_absolute_logit_drift": max_error, "max_kl_base_to_observed": max_kl,
            "top1_agreement": top_agreement / len(before), "top5_set_agreement": topk_agreement / len(before)}


def _float32_head_logits(loaded, panel):
    head = loaded.model.get_output_embeddings()
    if not isinstance(head, torch.nn.Linear):
        raise WeightEditInputError("diagnostic float32 head requires a standard Linear output head")

    def recompute_head(module, args, output):
        bias = module.bias.float() if module.bias is not None else None
        return torch.nn.functional.linear(args[0].float(), module.weight.float(), bias)

    handle = head.register_forward_hook(recompute_head)
    try:
        return preservation_logits(loaded, panel)
    finally:
        handle.remove()


def measure_mode(model_ref, job, examples, recipe, dtype, device):
    loaded = load_model(model_ref, dtype=dtype, device=device)
    panel = recipe["preservation_interactions"]
    measurements = {"native_logits_cast_float32": lambda: preservation_logits(loaded, panel)}
    if dtype == torch.bfloat16:
        measurements["float32_lm_head_bf16_trunk"] = lambda: _float32_head_logits(loaded, panel)
    before_tensors = tensor_digests(loaded.model)
    baseline = {name: measure() for name, measure in measurements.items()}
    repeat = {name: compare_logits(baseline[name], measure()) for name, measure in measurements.items()}
    sites, site, calculated, evidence = calculate_edit(loaded, job, examples, {**recipe, "target_step": 1e-3})
    after_calculation = {name: compare_logits(baseline[name], measure()) for name, measure in measurements.items()}
    if tensor_digests(loaded.model) != before_tensors:
        raise WeightEditInputError("calculation changed model tensors before any edit")
    original = loaded.model.get_parameter(site.tensor_name).detach().clone()
    records = []
    try:
        for strength in STRENGTHS:
            with torch.no_grad():
                loaded.model.get_parameter(site.tensor_name).copy_(original)
            if model_state_digest(loaded.model) != loaded.base_digest:
                raise WeightEditInputError("diagnostic did not restore the exact base tensor state")
            delta = Delta("low_rank", U=calculated.U * (strength / 1e-3), V=calculated.V)
            edited = apply(loaded, {site.tensor_name: delta}, sites=sites, method="diagnostic_strength_sweep",
                           settings={"target_step": strength, "diagnostic_only": True})
            metrics = {name: compare_logits(baseline[name], measure()) for name, measure in measurements.items()}
            records.append({"target_step": strength, "effective_change_norm": edited.receipt["change_norms"][site.tensor_name],
                            "changed_tensors": edited.receipt["edited_tensors"], "measurements": metrics})
    finally:
        with torch.no_grad():
            loaded.model.get_parameter(site.tensor_name).copy_(original)
    if tensor_digests(loaded.model) != before_tensors:
        raise WeightEditInputError("diagnostic failed to restore base tensors")
    result = {"model_precision": str(dtype).removeprefix("torch."), "base_digest": loaded.base_digest,
              "site": site.document(), "renderer_digest": evidence["renderer_digest"],
              "repeat_base": repeat, "after_calculation_without_edit": after_calculation,
              "strength_sweep": records, "base_restored": True,
              "strength_definition": "shared calculated low-rank direction scaled linearly; cast back once per strength"}
    if device.startswith("cuda"):
        loaded.model.to("cpu")
        torch.cuda.empty_cache()
    return result


def run_diagnostic(state_dir: Path, request_path: Path, output_path: Path, device="cuda:0"):
    request = parse_strict_json(request_path.read_bytes())
    if request["training_request_digest"] != _document_digest({key: value for key, value in request.items() if key != "training_request_digest"}):
        raise WeightEditInputError("diagnostic request digest is invalid")
    job = gym_job_from_document(parse_strict_json((state_dir / "jobs" / (request["job_digest"][7:] + ".json")).read_bytes()))
    task = reference_task(rows_per_template=job.scenario_sources[0].parameters["rows_per_template"])
    if job.digest != task.job.digest:
        raise WeightEditInputError("diagnostic supports only the frozen synthetic reference task")
    service = GymStudioService(state_dir=state_dir)
    examples = [{"example_digest": _document_digest(example), "interaction": task.project(example["input"], example["response"])}
                for example in service.load_training_examples(request["dataset_digest"])]
    if _document_digest({"examples": examples}) != request["projected_examples_digest"]:
        raise WeightEditInputError("diagnostic demonstrations differ from the frozen dataset")
    modes = [measure_mode(request["base_reference"]["artifact_path"], job, examples, request["recipe"], dtype, device)
             for dtype in (torch.bfloat16, torch.float32)]
    native = modes[0]
    floors = [native["repeat_base"]["native_logits_cast_float32"]["max_absolute_logit_drift"],
              native["after_calculation_without_edit"]["native_logits_cast_float32"]["max_absolute_logit_drift"],
              native["strength_sweep"][0]["measurements"]["native_logits_cast_float32"]["max_absolute_logit_drift"]]
    report = {"kind": "nf_weight_edit_drift_diagnostic_v1", "evidence_class": "measured", "job_digest": job.digest,
              "training_request_digest": request["training_request_digest"], "dataset_digest": request["dataset_digest"],
              "preservation_panel_digest": _document_digest({"interactions": request["recipe"]["preservation_interactions"]}),
              "preservation_cases": len(request["recipe"]["preservation_interactions"]), "training_examples": len(examples),
              "active_preservation_policy_changed": False, "active_logit_drift_limit": 0.25,
              "noise_floor_max_native_bf16": max(floors), "policy_v2_proposal_needed": max(floors) >= 0.25,
              "modes": modes, "holdout_accessed": False, "candidate_published": False}
    report["evidence_digest"] = _document_digest(report)
    with output_path.open("xb") as stream:
        stream.write(_wire_document(report))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("state_dir", type=Path)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--screen-report", type=Path)
    parser.add_argument("--truncation-budget", type=int, default=LAYER_SWEEP_MAX_NEW_TOKENS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--task-validation", action="store_true")
    mode.add_argument("--training-fit", action="store_true")
    mode.add_argument("--layer-strength-sweep", action="store_true")
    mode.add_argument("--profile-one", action="store_true")
    mode.add_argument("--public-split-baseline", action="store_true")
    mode.add_argument("--audit-truncation", action="store_true")
    mode.add_argument("--rescore-finalists", action="store_true")
    mode.add_argument("--single-layer-stop-gate", action="store_true")
    mode.add_argument("--audit-failing-keys", action="store_true")
    args = parser.parse_args()
    if args.audit_failing_keys:
        report = run_edit_key_audit(args.state_dir, args.request, args.output, device=args.device)
        print(json.dumps({"layers": list(report["per_layer"]), "failing_examples": report["failing_examples"],
                          "base_unchanged": report["base_unchanged"], "solver_called": report["solver_called"]}))
        return
    if args.profile_one:
        report = run_single_candidate_profile(args.state_dir, args.request, args.output)
        print(json.dumps({"timings_seconds": report["timings_seconds"], "layer": report["layer"],
                          "strength": report["strength"], "validation_exact_match": report["validation_exact_match"]}))
        return
    if args.audit_truncation:
        report = audit_generation_truncation(args.request, args.output, max_new_tokens=args.truncation_budget)
        print(json.dumps({"max_new_tokens": report["max_new_tokens"],
                          "total_scored_answers": report["total_scored_answers"],
                          "total_truncated_answers": report["total_truncated_answers"],
                          "generation_cap_failures": report["generation_cap_failures"],
                          "zero_truncated_answers": report["zero_truncated_answers"]}))
        return
    if args.rescore_finalists or args.single_layer_stop_gate:
        if args.screen_report is None:
            raise WeightEditInputError("finalist rescore requires a digest-bound screen report")
        candidate_grid = tuple((layer, strength) for layer in (10, 9) for strength in (0.1, 0.5, 1.0))
        report = run_finalist_rescore(args.state_dir, args.request, args.screen_report, args.output,
            device=args.device, single_layer_candidates=candidate_grid if args.single_layer_stop_gate else None)
        print(json.dumps({"compositions": [row["composition_size"] for row in report["candidates"]],
                          "generation_max_new_tokens": report["generation_max_new_tokens"],
                          "zero_truncated_answers": report["generation_truncation"]["zero_truncated_answers"],
                          "holdout_accessed": report["holdout_accessed"]}))
        return
    if args.public_split_baseline:
        report = run_public_split_baseline(args.state_dir, args.request, args.output)
        print(json.dumps({"train_cases": report["train"]["score"]["cases"],
                          "train_failures": report["train"]["failure_count"],
                          "validation_cases": report["validation"]["score"]["cases"],
                          "validation_failures": report["validation"]["failure_count"],
                          "holdout_accessed": report["holdout_accessed"]}))
        return
    if args.training_fit:
        report = run_training_fit_diagnostic(args.state_dir, args.request, args.output)
        print(json.dumps({"split": "train", "cases": report["baseline"]["cases"], "base": report["baseline"]["exact_match"],
                          "edited": report["edited"]["exact_match"], "gain": report["training_gain"], "edited_model_digest": report["edited_model_digest"]}))
        return
    if args.layer_strength_sweep:
        report = run_layer_strength_sweep(args.state_dir, args.request, args.output, device=args.device)
        print(json.dumps({"layers": report["discovered_layer_count"], "strengths": report["strengths"],
                          "best": report["best_by_validation_exact_match_then_expected_answer_logprob"],
                          "base_train": report["base_train"]["task"]["exact_match"],
                          "base_validation": report["base_validation"]["task"]["exact_match"],
                          "strength_1_effective_change": report["strength_1_effective_change"]}))
        return
    if args.task_validation:
        report = run_validation_diagnostic(args.state_dir, args.request, args.output)
        print(json.dumps({"base": report["baseline"]["exact_match"],
                          "strengths": [{"target_step": row["target_step"], "score": row["task_score"]["exact_match"],
                                         "gain": row["task_gain"], "preservation_passed": row["preservation"]["passed"],
                                         "specialist_passed": row["specialist_acceptance"]["passed"]} for row in report["strengths"]]}))
        return
    report = run_diagnostic(args.state_dir, args.request, args.output)
    print(json.dumps({"noise_floor": report["noise_floor_max_native_bf16"], "policy_v2_proposal_needed": report["policy_v2_proposal_needed"]}))


if __name__ == "__main__":
    main()