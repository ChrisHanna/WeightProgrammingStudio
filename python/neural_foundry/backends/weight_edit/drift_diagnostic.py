"""Read-only calibration experiments; never changes preservation policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from ...gym_job import gym_job_from_document
from ...gym_service import GymStudioService, _document_digest, _wire_document
from ...model_digest import model_state_digest
from ...provenance import parse_strict_json
from ...reference_json_extraction import reference_task
from .apply import apply, tensor_digests
from .loader import WeightEditInputError, load_model
from .methods import Delta
from .worker import calculate_edit, preservation_logits

STRENGTHS = (0.0, 1e-5, 1e-4, 1e-3)


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
    args = parser.parse_args()
    report = run_diagnostic(args.state_dir, args.request, args.output)
    print(json.dumps({"noise_floor": report["noise_floor_max_native_bf16"], "policy_v2_proposal_needed": report["policy_v2_proposal_needed"]}))


if __name__ == "__main__":
    main()