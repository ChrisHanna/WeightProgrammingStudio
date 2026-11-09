from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from neural_foundry.backends.weight_edit.loader import UnsupportedModel
from neural_foundry.backends.weight_edit.near_ties import POLICY_V2_DEFINITION, calibrate_near_ties, verify_near_tie_tokens
from neural_foundry.backends.weight_edit.serving_probes import calibrate_portability, verify_identity, verify_portability
from neural_foundry.gym_service import _document_digest


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


def test_reference_tie_is_allowed_when_observed_gap_reflects_two_token_noise():
    first, second = _bases(0)
    first["records"][0] = {"prompt_token_ids": [1], "greedy_token_id": 10, "greedy_logprob": -1.5,
                           "top_two": [{"token_id": 10, "logprob": -1.5}, {"token_id": 11, "logprob": -1.5}]}
    second["records"][0] = {"prompt_token_ids": [1], "greedy_token_id": 11, "greedy_logprob": -1.412,
                            "top_two": [{"token_id": 11, "logprob": -1.412}, {"token_id": 10, "logprob": -1.537}]}
    policy = calibrate_near_ties(first, second)
    probe = policy["probes"][0]
    assert probe["near_tie"] is True and probe["transformers_gap"] == 0
    assert probe["vllm_gap"] > policy["measured_base_noise"]
    assert probe["vllm_gap"] <= probe["gap_difference_bound"]
    assert policy["decisive_probes"] == 15


def test_retained_smollm2_base_measurements_have_fifteen_decisive_probes():
    root = Path(__file__).resolve().parents[1] / "docs" / "evidence"
    trial = json.loads((root / "weight-edit-smollm2-bf16-near-ties.json").read_bytes())
    analysis = json.loads((root / "weight-edit-smollm2-bf16-near-tie-calibration.json").read_bytes())
    for report in (trial, analysis):
        assert report["evidence_digest"] == _document_digest({key: value for key, value in report.items() if key != "evidence_digest"})
    assert analysis["source_evidence_digest"] == trial["evidence_digest"]
    recomputed = calibrate_portability(trial["transformers_base_record"], trial["vllm_base_record"])
    assert analysis["calibration_passed"] is True
    assert analysis["calibration"]["near_tie_policy"]["kind"] == "nf_bf16_base_near_ties_v1"
    assert recomputed["near_tie_policy"]["kind"] == "nf_bf16_base_near_ties_v2"
    assert recomputed["near_tie_policy"]["probes"] == analysis["calibration"]["near_tie_policy"]["probes"]
    policy = recomputed["near_tie_policy"]
    assert policy["decisive_probes"] == 15
    assert [probe["index"] for probe in policy["probes"] if probe["near_tie"]] == [6]
    assert verify_portability(trial["transformers_base_record"], trial["vllm_base_record"], recomputed)["verified"] is True
    assert verify_identity(trial["vllm_base_record"], trial["vllm_base_record"])["verified"] is True


def test_v2_matches_the_frozen_published_rule():
    path = Path(__file__).resolve().parents[1] / "protocols" / "near-tie-policy-v2.json"
    published = json.loads(path.read_bytes())
    assert published["status"] == "frozen" and published["version"] == 2
    assert POLICY_V2_DEFINITION == {key: published[key] for key in POLICY_V2_DEFINITION}
    assert published["prompting_evidence"]["reference_gap"] == 0
    assert published["gap_difference_multiplier"] == 2 and published["minimum_decisive_probes"] == 12


def test_nemotron_failure_classification_does_not_invent_v2_replay():
    root = Path(__file__).resolve().parents[1] / "docs" / "evidence"
    source = json.loads((root / "weight-edit-nemotron-bf16.json").read_bytes())
    analysis = json.loads((root / "weight-edit-nemotron-failure-analysis.json").read_bytes())
    assert analysis["source_probe_digest"] == source["probe_digest"]
    assert source["reasons"] == ["candidate exceeds the frozen unedited-base portability limit"]
    assert analysis["failure_type"] == "same_token_log_probability_gap"
    assert analysis["v2_replay_status"] == "unavailable_missing_raw_records"
    assert analysis["v2_passed"] is None and analysis["new_gpu_run"] is False


def test_v2_reference_gap_boundary_remains_strict():
    first, second = _bases(0)
    first["records"][0]["top_two"][1]["logprob"] = -1.02
    second["records"][0]["top_two"][1]["logprob"] = -1.0
    policy = calibrate_near_ties(first, second)
    assert policy["probes"][0]["transformers_gap"] == policy["measured_base_noise"]
    assert policy["probes"][0]["near_tie"] is False