"""Read-only calibration experiments; never changes preservation policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from ...gym_job import GymJobError, gym_job_from_document
from ...gym_adapters import _parse_generated_json
from ...gym_service import ExecutionResponse, GymStudioService, _document_digest, _wire_document
from ...model_digest import model_state_digest
from ...provenance import parse_strict_json
from ...reference_json_extraction import generate_scenarios, reference_task
from ...json_proof_loop import SPECIALIST_POLICY, public_required_rows, score_public_rows, specialist_acceptance
from ...prompt_rendering import PromptRenderer
from .apply import apply, tensor_digests
from .loader import WeightEditInputError, load_model
from .methods import Delta
from .numerics import deterministic_execution
from .preservation import calibrate_one_ulp, float32_logits_of_exact_weights, metrics, verify_v2
from .worker import calculate_edit, preservation_logits

STRENGTHS = (0.0, 1e-5, 1e-4, 1e-3)


def validation_model(loaded, *, max_new_tokens=80):
    renderer = PromptRenderer(loaded.tokenizer)

    def infer(interaction):
        ids = renderer.render(interaction).token_ids
        device = next(loaded.model.parameters()).device
        inputs = torch.tensor([ids], dtype=torch.long, device=device)
        if len(ids) + max_new_tokens > 512:
            raise WeightEditInputError("public validation prompt exceeds frozen context")
        with deterministic_execution(loaded.model), torch.inference_mode():
            output = loaded.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), do_sample=False,
                                           max_new_tokens=max_new_tokens, pad_token_id=loaded.tokenizer.eos_token_id)
        text = loaded.tokenizer.decode(output[0, len(ids):], skip_special_tokens=True)
        return ExecutionResponse(_parse_generated_json(text), {"raw_text": text, "generated_token_ids": output[0, len(ids):].tolist()})

    identity = {"runtime": "Transformers", "model_digest": model_state_digest(loaded.model),
                "precision": str(next(loaded.model.parameters()).dtype).removeprefix("torch."),
                "renderer_digest": renderer.digest, "served_model_identity_verified": False}
    return SimpleNamespace(infer=infer, loaded_identity=lambda: identity)


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
    parser.add_argument("--task-validation", action="store_true")
    args = parser.parse_args()
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