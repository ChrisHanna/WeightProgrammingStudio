from __future__ import annotations

import copy

import pytest

from neural_foundry.backends.weight_edit.loader import UnsupportedModel
from neural_foundry.backends.weight_edit.near_ties import calibrate_near_ties, verify_near_tie_tokens
from neural_foundry.backends.weight_edit.serving_probes import calibrate_portability, verify_identity, verify_portability


def _bases(ties):
    first = {"precision": "bfloat16", "renderer_digest": "sha256:" + "1" * 64,
             "weight_files": {"model.safetensors": "sha256:" + "2" * 64}, "records": []}
    second = copy.deepcopy(first)
    for index in range(16):
        gap = 0.01 if index < ties else 0.5
        left = {"prompt_token_ids": [index + 1], "greedy_token_id": 10, "greedy_logprob": -1.0,
                "top_two": [{"token_id": 10, "logprob": -1.0}, {"token_id": 11, "logprob": -1.0 - gap}]}
        right = {**left, "greedy_logprob": -0.98,
                 "top_two": [{"token_id": 10, "logprob": -0.98}, {"token_id": 11, "logprob": -0.98 - gap}]}
        if index < ties:
            right = {**right, "greedy_token_id": 11, "top_two": [
                {"token_id": 11, "logprob": -0.98}, {"token_id": 10, "logprob": -0.99}]}
        first["records"].append(left)
        second["records"].append(right)
    return first, second


def test_base_near_tie_allows_only_frozen_pair_with_twelve_decisive_probes():
    first, second = _bases(4)
    policy = calibrate_near_ties(first, second)
    assert policy["decisive_probes"] == 12
    assert policy["probes"][0]["allowed_tokens"] == [10, 11]
    assert verify_near_tie_tokens(first["records"][0], second["records"][0], 0, policy) > 0
    third = copy.deepcopy(second["records"][0])
    third["greedy_token_id"] = 12
    with pytest.raises(UnsupportedModel, match="allowance"):
        verify_near_tie_tokens(first["records"][0], third, 0, policy)
    with pytest.raises(UnsupportedModel, match="12 of 16"):
        calibrate_near_ties(*_bases(5))


def test_candidate_cannot_enlarge_or_create_near_tie_allowance():
    first, second = _bases(1)
    policy = calibrate_near_ties(first, second)
    changed = {**policy, "measured_base_noise": 100.0}
    with pytest.raises(UnsupportedModel, match="policy"):
        verify_near_tie_tokens(first["records"][0], second["records"][0], 0, changed)
    flipped = {**second["records"][1], "greedy_token_id": 11}
    with pytest.raises(UnsupportedModel, match="allowance"):
        verify_near_tie_tokens(first["records"][1], flipped, 1, policy)


def test_near_tie_portability_never_weakens_exact_token_identity():
    first, second = _bases(1)
    metadata = {"model_digest": "sha256:" + "3" * 64, "image": "sha256:" + "4" * 64, "context_tokens": 512}
    first = {**first, **metadata, "kind": "nf_transformers_portability_probes_v1"}
    second = {**second, **metadata, "kind": "nf_vllm_weight_identity_v2"}
    calibration = calibrate_portability(first, second)
    assert calibration["near_tie_policy"]["decisive_probes"] == 15
    assert verify_portability(first, second, calibration)["verified"] is True
    altered_identity = {**first, "kind": "nf_vllm_weight_identity_v2"}
    with pytest.raises(UnsupportedModel, match="greedy tokens"):
        verify_identity(second, altered_identity)