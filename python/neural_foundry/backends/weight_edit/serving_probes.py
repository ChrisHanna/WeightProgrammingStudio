"""Separate same-runtime identity from base-calibrated portability."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ...gym_service import _document_digest
from ...gym_job import _is_digest, is_finite_number
from .loader import UnsupportedModel

IDENTITY_LOGPROB_ATOL = 1e-6
PORTABILITY_MARGIN = 1e-6


def compare_records(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> float:
    for name in ("precision", "renderer_digest"):
        if expected.get(name) != observed.get(name):
            raise UnsupportedModel(f"probe {name} differs")
    first, second = expected.get("records"), observed.get("records")
    if not isinstance(first, list) or not isinstance(second, list) or len(first) != 16 or len(second) != 16:
        raise UnsupportedModel("probe records require exactly 16 fixed prompts")
    maximum = 0.0
    for index, (left, right) in enumerate(zip(first, second, strict=True)):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise UnsupportedModel("probe record must be an object")
        if left.get("prompt_token_ids") != right.get("prompt_token_ids") or left.get("greedy_token_id") != right.get("greedy_token_id"):
            raise UnsupportedModel(f"probe {index} prompt or greedy tokens differ")
        tokens = left.get("prompt_token_ids")
        if (not isinstance(tokens, list) or not tokens or any(type(token) is not int or token < 0 for token in tokens)
            or type(left.get("greedy_token_id")) is not int or left["greedy_token_id"] < 0):
            raise UnsupportedModel("probe token IDs are invalid")
        values = left.get("greedy_logprob"), right.get("greedy_logprob")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value > 0 for value in values):
            raise UnsupportedModel("probe log probabilities must be finite and nonpositive")
        maximum = max(maximum, abs(values[0] - values[1]))
    return maximum


def verify_identity(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> dict[str, Any]:
    for document in (expected, observed):
        if document.get("kind") != "nf_vllm_weight_identity_v2":
            raise UnsupportedModel("identity must be recorded by vLLM, not Transformers")
        if (any(not _is_digest(document.get(name)) for name in ("model_digest", "renderer_digest", "image"))
            or type(document.get("context_tokens")) is not int or document["context_tokens"] < 1
            or document.get("precision") not in {"float32", "float16", "bfloat16"}
            or not isinstance(document.get("weight_files"), Mapping) or not document["weight_files"]
            or any(not isinstance(name, str) or not name.endswith(".safetensors") or not _is_digest(digest)
                   for name, digest in document["weight_files"].items())):
            raise UnsupportedModel("identity is missing valid weight or runtime evidence")
    for name in ("model_digest", "weight_files", "image", "context_tokens"):
        if expected.get(name) != observed.get(name):
            raise UnsupportedModel(f"identity {name} differs")
    difference = compare_records(expected, observed)
    if difference > IDENTITY_LOGPROB_ATOL:
        raise UnsupportedModel("same-runtime identity log probabilities differ")
    return {"kind": "nf_vllm_identity_verification_v2", "verified": True, "fixed_prompts": 16,
            "max_logprob_absolute_error": difference, "logprob_atol": IDENTITY_LOGPROB_ATOL,
            "precision": expected["precision"]}


def calibrate_portability(base_transformers: Mapping[str, Any], base_vllm: Mapping[str, Any]) -> dict[str, Any]:
    if base_transformers.get("kind") != "nf_transformers_portability_probes_v1" or base_vllm.get("kind") != "nf_vllm_weight_identity_v2":
        raise UnsupportedModel("portability calibration requires the two declared runtime records")
    verify_identity(base_vllm, base_vllm)
    difference = compare_records(base_transformers, base_vllm)
    calibration = {"kind": "nf_base_portability_calibration_v1", "precision": base_vllm["precision"],
                   "renderer_digest": base_vllm["renderer_digest"], "image": base_vllm["image"],
                   "context_tokens": base_vllm["context_tokens"], "base_digest": base_transformers["model_digest"],
                   "base_weight_files": base_vllm["weight_files"], "measured_base_max_error": difference,
                   "margin": PORTABILITY_MARGIN, "logprob_atol": difference + PORTABILITY_MARGIN}
    calibration["calibration_digest"] = _document_digest(calibration)
    return calibration


def verify_portability(transformers: Mapping[str, Any], vllm: Mapping[str, Any], calibration: Mapping[str, Any]) -> dict[str, Any]:
    if transformers.get("kind") != "nf_transformers_portability_probes_v1" or vllm.get("kind") != "nf_vllm_weight_identity_v2":
        raise UnsupportedModel("portability requires the two declared runtime records")
    if not is_finite_number(calibration.get("measured_base_max_error")) or calibration["measured_base_max_error"] < 0:
        raise UnsupportedModel("portability calibration has an invalid measured base error")
    payload = {key: value for key, value in calibration.items() if key != "calibration_digest"}
    if (calibration.get("kind") != "nf_base_portability_calibration_v1"
        or calibration.get("calibration_digest") != _document_digest(payload)
        or calibration.get("margin") != PORTABILITY_MARGIN
        or calibration.get("logprob_atol") != calibration.get("measured_base_max_error", -1) + PORTABILITY_MARGIN):
        raise UnsupportedModel("portability calibration is invalid or changed")
    for name in ("precision", "renderer_digest", "image", "context_tokens"):
        if vllm.get(name) != calibration.get(name):
            raise UnsupportedModel(f"portability {name} differs from the base calibration")
    difference = compare_records(transformers, vllm)
    if difference > calibration["logprob_atol"]:
        raise UnsupportedModel("candidate exceeds the frozen unedited-base portability limit")
    return {"kind": "nf_portability_verification_v1", "verified": True,
            "calibration_digest": calibration["calibration_digest"], "max_logprob_absolute_error": difference,
            "logprob_atol": calibration["logprob_atol"], "precision": calibration["precision"]}