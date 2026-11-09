from __future__ import annotations

import copy

import pytest

from neural_foundry.backends.weight_edit.loader import UnsupportedModel
from neural_foundry.backends.weight_edit.serving_probes import calibrate_portability, verify_identity, verify_portability


def _record(kind, offset=0.0):
    return {"kind": kind, "precision": "bfloat16", "renderer_digest": "sha256:" + "1" * 64,
            "model_digest": "sha256:" + "2" * 64, "image": "sha256:" + "3" * 64,
            "context_tokens": 512, "weight_files": {"model.safetensors": "sha256:" + "4" * 64},
            "records": [{"prompt_token_ids": [index + 1], "greedy_token_id": 16, "greedy_logprob": -1.0 + offset}
                        for index in range(16)]}


def test_bf16_cross_runtime_difference_is_portability_not_identity() -> None:
    reference = _record("nf_transformers_portability_probes_v1")
    served = _record("nf_vllm_weight_identity_v2", 0.08)
    calibration = calibrate_portability(reference, served)
    assert verify_portability(reference, served, calibration)["verified"] is True
    assert verify_identity(served, copy.deepcopy(served))["verified"] is True
    with pytest.raises(UnsupportedModel, match="vLLM"):
        verify_identity(reference, served)
    changed = _record("nf_vllm_weight_identity_v2", 0.081)
    with pytest.raises(UnsupportedModel, match="same-runtime"):
        verify_identity(served, changed)
    with pytest.raises(UnsupportedModel, match="frozen"):
        verify_portability(reference, changed, calibration)


def test_portability_limit_and_identity_weights_are_frozen() -> None:
    reference, served = _record("nf_transformers_portability_probes_v1"), _record("nf_vllm_weight_identity_v2", 0.08)
    calibration = calibrate_portability(reference, served)
    altered = {**calibration, "logprob_atol": 10.0}
    with pytest.raises(UnsupportedModel, match="calibration"):
        verify_portability(reference, served, altered)
    altered = copy.deepcopy(served)
    altered["weight_files"]["model.safetensors"] = "sha256:" + "5" * 64
    with pytest.raises(UnsupportedModel, match="weight_files"):
        verify_identity(served, altered)
    missing = {key: value for key, value in served.items() if key != "weight_files"}
    with pytest.raises(UnsupportedModel, match="evidence"):
        verify_identity(missing, missing)