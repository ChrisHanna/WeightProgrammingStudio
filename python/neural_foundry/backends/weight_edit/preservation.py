"""Frozen v2 gates separate precise drift from serving-dtype behaviour."""

from __future__ import annotations

from typing import Any

import torch

from ...gym_job import _is_digest, is_finite_number
from ...gym_service import _document_digest
from ...model_digest import model_state_digest
from ...prompt_rendering import PromptRenderer
from .loader import WeightEditInputError

POLICY_V2 = {
    "policy_id": "weight_edit_preservation", "version": 2,
    "float32_weight_source": "exact bf16 base and edited tensors promoted to float32; no separate edit recalculation",
    "max_float32_logit_drift": 0.25, "min_bf16_top1_agreement": 1.0,
    "min_bf16_top5_set_agreement": 0.95, "max_mean_bf16_kl_base_to_edited": 0.0001,
    "bf16_max_absolute_logit_drift": "report_only",
    "behaviour_runtime": "real Transformers model at serving precision; endpoint identity and final serving scores are separate checks",
}


def metrics(before: list[torch.Tensor], after: list[torch.Tensor]) -> dict[str, Any]:
    if not before or len(before) != len(after):
        raise WeightEditInputError("preservation requires a complete matching panel")
    maximum, divergence, top1, top5 = 0.0, 0.0, 0, 0
    for first, second in zip(before, after, strict=True):
        if first.ndim != 1 or first.shape != second.shape or not first.numel() or not bool(torch.isfinite(first).all()) or not bool(torch.isfinite(second).all()):
            raise WeightEditInputError("preservation logits are incomplete or nonfinite")
        first, second = first.double(), second.double()
        maximum = max(maximum, float((first - second).abs().max()))
        left, right = first.log_softmax(-1), second.log_softmax(-1)
        divergence += max(0.0, float((left.exp() * (left - right)).sum()))
        top1 += int(first.argmax() == second.argmax())
        count = min(first.numel(), 5)
        top5 += int(set(first.topk(count).indices.tolist()) == set(second.topk(count).indices.tolist()))
    return {"cases": len(before), "max_absolute_logit_drift": maximum, "mean_kl_base_to_edited": divergence / len(before),
            "top1_agreement": top1 / len(before), "top5_set_agreement": top5 / len(before)}


def verify_v2(float32: dict[str, Any], bf16: dict[str, Any], calibration: dict[str, Any]) -> dict[str, Any]:
    if (calibration.get("kind") != "nf_bf16_one_ulp_sensitivity_v2"
        or calibration.get("calibration_digest") != _document_digest({key: value for key, value in calibration.items() if key != "calibration_digest"})
        or calibration.get("policy_digest") != _document_digest(POLICY_V2)):
        raise WeightEditInputError("preservation calibration differs from frozen v2")
    if (not _is_digest(calibration.get("base_digest")) or not _is_digest(calibration.get("panel_digest"))
        or not _is_digest(calibration.get("renderer_digest")) or calibration.get("base_restored") is not True
        or calibration.get("precision") != "bfloat16" or calibration.get("adjacent_value_verified") is not True
        or type(calibration.get("flat_index")) is not int or calibration["flat_index"] < 0
        or type(calibration.get("old_bits")) is not int or type(calibration.get("new_bits")) is not int
        or abs(calibration["old_bits"] - calibration["new_bits"]) != 1):
        raise WeightEditInputError("preservation calibration lacks verified base one-ULP evidence")
    if (not isinstance(calibration.get("tensor_name"), str) or not calibration["tensor_name"]
        or calibration.get("changed_scalar_count") != 1
        or not isinstance(calibration.get("runtime"), dict)
        or calibration["runtime"].get("torch") != "2.9.1"
        or calibration["runtime"].get("transformers") != "4.57.3"
        or not isinstance(calibration.get("float32_measurements"), dict)
        or not isinstance(calibration.get("measurements"), dict)
        or calibration["measurements"].get("cases") != bf16.get("cases")
        or calibration["float32_measurements"].get("cases") != bf16.get("cases")):
        raise WeightEditInputError("preservation calibration lacks complete runtime and control measurements")
    for control in (calibration["measurements"], calibration["float32_measurements"]):
        for name in ("max_absolute_logit_drift", "mean_kl_base_to_edited", "top1_agreement", "top5_set_agreement"):
            if not is_finite_number(control.get(name)) or control[name] < 0:
                raise WeightEditInputError("preservation control measurement is missing or invalid")
        if control["top1_agreement"] > 1 or control["top5_set_agreement"] > 1:
            raise WeightEditInputError("preservation control agreement is invalid")
    if type(float32.get("cases")) is not int or float32["cases"] < 1 or float32["cases"] != bf16.get("cases"):
        raise WeightEditInputError("preservation measurements have incomplete panels")
    checks = (
        is_finite_number(float32.get("max_absolute_logit_drift")) and 0 <= float32["max_absolute_logit_drift"] <= POLICY_V2["max_float32_logit_drift"],
        is_finite_number(bf16.get("top1_agreement")) and POLICY_V2["min_bf16_top1_agreement"] <= bf16["top1_agreement"] <= 1,
        is_finite_number(bf16.get("top5_set_agreement")) and POLICY_V2["min_bf16_top5_set_agreement"] <= bf16["top5_set_agreement"] <= 1,
        is_finite_number(bf16.get("mean_kl_base_to_edited")) and 0 <= bf16["mean_kl_base_to_edited"] <= POLICY_V2["max_mean_bf16_kl_base_to_edited"],
    )
    return {"kind": "nf_preservation_verification_v2", "passed": all(checks),
            "policy_digest": _document_digest(POLICY_V2), "calibration_digest": calibration["calibration_digest"],
            "float32": float32, "bf16": bf16}


def float32_logits_of_exact_weights(loaded, panel):
    from .worker import preservation_logits

    dtype = next(loaded.model.parameters()).dtype
    if dtype not in {torch.bfloat16, torch.float32}:
        raise WeightEditInputError("exact-weight promotion requires bf16 or float32 model tensors")
    if any(parameter.is_floating_point() and parameter.dtype != dtype for parameter in loaded.model.parameters()):
        raise WeightEditInputError("exact-weight promotion does not support mixed parameter precision")
    buffers = [(module, name, value) for module in loaded.model.modules() for name, value in module._buffers.items()]
    before = model_state_digest(loaded.model)
    try:
        loaded.model.to(torch.float32)
        return preservation_logits(loaded, panel)
    finally:
        loaded.model.to(dtype)
        for module, name, value in buffers:
            module._buffers[name] = value
        if model_state_digest(loaded.model) != before:
            raise WeightEditInputError("float32 preservation measurement changed stored model tensors")


def calibrate_one_ulp(loaded, site, panel):
    from importlib import metadata
    from .worker import preservation_logits

    parameter = loaded.model.get_parameter(site.tensor_name)
    if parameter.dtype != torch.bfloat16:
        raise WeightEditInputError("one-ULP sensitivity calibration requires bf16 stored weights")
    base_digest = model_state_digest(loaded.model)
    magnitudes = parameter.detach().abs().flatten()
    index = int(torch.where(magnitudes > 0, magnitudes, torch.full_like(magnitudes, torch.inf)).argmin())
    original = parameter.detach().flatten()[index].cpu().clone()
    if not bool(torch.isfinite(original)) or original.item() == 0:
        raise WeightEditInputError("selected site has no finite nonzero calibration scalar")
    adjacent = torch.nextafter(original, torch.full_like(original, torch.inf))
    if adjacent.item() == original.item() or not bool(torch.isfinite(adjacent)):
        raise WeightEditInputError("one-ULP control has no finite effective change")
    before = preservation_logits(loaded, panel)
    float32_before = float32_logits_of_exact_weights(loaded, panel)
    stored = parameter.detach().clone()
    try:
        with torch.no_grad():
            parameter.view(-1)[index].copy_(adjacent.to(parameter.device))
        changed = int(torch.count_nonzero(parameter.detach() != stored))
        if changed != 1:
            raise WeightEditInputError("one-ULP control changed more than one scalar")
        observed = preservation_logits(loaded, panel)
        float32_observed = float32_logits_of_exact_weights(loaded, panel)
    finally:
        with torch.no_grad():
            parameter.view(-1)[index].copy_(original.to(parameter.device))
    if model_state_digest(loaded.model) != base_digest:
        raise WeightEditInputError("one-ULP calibration failed to restore base tensors")
    record = {"kind": "nf_bf16_one_ulp_sensitivity_v2", "policy_digest": _document_digest(POLICY_V2),
              "base_digest": base_digest, "panel_digest": _document_digest({"source": "public_train", "interactions": panel}),
              "renderer_digest": PromptRenderer(loaded.tokenizer).digest, "precision": "bfloat16",
              "tensor_name": site.tensor_name, "flat_index": index, "old_value": original.item(), "new_value": adjacent.item(),
              "old_bits": int(original.view(torch.int16)), "new_bits": int(adjacent.view(torch.int16)),
              "adjacent_value_verified": True, "base_restored": True, "measurements": metrics(before, observed),
              "changed_scalar_count": changed, "float32_measurements": metrics(float32_before, float32_observed),
              "runtime": {"torch": metadata.version("torch").split("+")[0], "transformers": metadata.version("transformers")},
              "selection_rule": "smallest-magnitude nonzero scalar in the predeclared selected MLP site, before measuring drift",
              "scope": "smallest representable step at that selected scalar, not a global model sensitivity guarantee"}
    record["calibration_digest"] = _document_digest(record)
    return record