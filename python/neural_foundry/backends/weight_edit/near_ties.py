"""Base-only near-tie evidence for cross-runtime bf16 portability."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...gym_job import is_finite_number
from ...gym_service import _document_digest
from .loader import UnsupportedModel

MIN_DECISIVE_PROBES = 12
POLICY_V2_DEFINITION = {
    "policy_id": "bf16_base_near_ties", "version": 2, "precision": "bfloat16", "probe_count": 16,
    "minimum_decisive_probes": 12, "gap_difference_multiplier": 2,
    "noise_measurement": "maximum absolute log-probability discrepancy on common top-two token IDs across the unedited base runtimes",
    "reference_gap_rule": "reference top-two gap strictly less than measured base noise",
    "pair_rule": "both unedited-base runtimes have the same top-two token pair",
    "candidate_rule": "exact tokens on decisive probes; mismatches only within the frozen base pair, with same-token log-probability comparison",
    "identity_rule": "same-runtime identity remains exact-token",
}


def top_two(record: Mapping[str, Any]) -> dict[int, float]:
    values = record.get("top_two")
    if not isinstance(values, list) or len(values) != 2:
        raise UnsupportedModel("near-tie evidence requires two measured token log probabilities")
    output = {}
    for item in values:
        if (not isinstance(item, Mapping) or type(item.get("token_id")) is not int or item["token_id"] < 0
            or item["token_id"] in output or not is_finite_number(item.get("logprob")) or item["logprob"] > 0):
            raise UnsupportedModel("top-two token evidence is malformed")
        output[item["token_id"]] = float(item["logprob"])
    first, second = values
    if (first["logprob"] < second["logprob"] or first["token_id"] != record.get("greedy_token_id")
        or first["logprob"] != record.get("greedy_logprob")):
        raise UnsupportedModel("top-two evidence does not match the recorded greedy token")
    return output


def calibrate_near_ties(base_transformers: Mapping[str, Any], base_vllm: Mapping[str, Any]) -> dict[str, Any]:
    if base_transformers.get("precision") != "bfloat16" or base_vllm.get("precision") != "bfloat16":
        raise UnsupportedModel("this near-tie policy requires bf16 base measurements")
    for name in ("renderer_digest", "weight_files"):
        if not base_transformers.get(name) or base_transformers[name] != base_vllm.get(name):
            raise UnsupportedModel(f"near-tie base {name} differs")
    left, right = base_transformers.get("records"), base_vllm.get("records")
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != 16 or len(right) != 16:
        raise UnsupportedModel("near-tie calibration requires all 16 base probes")
    pairs, measured_noise = [], 0.0
    for index, (first, second) in enumerate(zip(left, right, strict=True)):
        if not isinstance(first, Mapping) or not isinstance(second, Mapping) or first.get("prompt_token_ids") != second.get("prompt_token_ids"):
            raise UnsupportedModel(f"near-tie base prompt {index} differs")
        first_values, second_values = top_two(first), top_two(second)
        common = first_values.keys() & second_values.keys()
        if not common:
            raise UnsupportedModel("base runtimes have no common top-two token for noise measurement")
        measured_noise = max(measured_noise, max(abs(first_values[token] - second_values[token]) for token in common))
        pairs.append((first, second, first_values, second_values))
    probes = []
    for index, (first, second, first_values, second_values) in enumerate(pairs):
        first_gap = abs(first["top_two"][0]["logprob"] - first["top_two"][1]["logprob"])
        second_gap = abs(second["top_two"][0]["logprob"] - second["top_two"][1]["logprob"])
        near = (first_values.keys() == second_values.keys() and first_gap < measured_noise
            and abs(first_gap - second_gap) <= 2 * measured_noise)
        if first["greedy_token_id"] != second["greedy_token_id"] and not near:
            raise UnsupportedModel(f"base probe {index} differs outside the measured near-tie policy")
        probes.append({"index": index, "prompt_token_ids": first["prompt_token_ids"], "near_tie": near,
                       "allowed_tokens": sorted(first_values) if near else [],
                       "transformers_gap": first_gap, "vllm_gap": second_gap,
                       "gap_difference_bound": 2 * measured_noise,
                       "reason": "reference base gap below measured per-token discrepancy; two-token gap difference bounded by twice that discrepancy" if near else "decisive base probe"})
    decisive = sum(not probe["near_tie"] for probe in probes)
    if decisive < MIN_DECISIVE_PROBES:
        raise UnsupportedModel("precision is unqualified: fewer than 12 of 16 decisive base probes")
    policy = {"kind": "nf_bf16_base_near_ties_v2", "policy_version": 2,
              "policy_definition_digest": _document_digest(POLICY_V2_DEFINITION),
              "precision": "bfloat16", "decisive_probes": decisive,
              "minimum_decisive_probes": MIN_DECISIVE_PROBES, "measured_base_noise": measured_noise,
              "transformers_base_record_digest": _document_digest(dict(base_transformers)),
              "vllm_base_record_digest": _document_digest(dict(base_vllm)), "probes": probes}
    policy["policy_digest"] = _document_digest(policy)
    return policy


def verify_near_tie_tokens(first: Mapping[str, Any], second: Mapping[str, Any], index: int,
                           policy: Mapping[str, Any]) -> float:
    payload = {key: value for key, value in policy.items() if key != "policy_digest"}
    if (policy.get("kind") != "nf_bf16_base_near_ties_v2" or policy.get("policy_version") != 2
        or policy.get("policy_definition_digest") != _document_digest(POLICY_V2_DEFINITION)
        or policy.get("policy_digest") != _document_digest(payload)
        or policy.get("minimum_decisive_probes") != MIN_DECISIVE_PROBES
        or policy.get("decisive_probes", 0) < MIN_DECISIVE_PROBES):
        raise UnsupportedModel("base near-tie policy is invalid or changed")
    if type(index) is not int or not 0 <= index < 16 or not isinstance(policy.get("probes"), list) or len(policy["probes"]) != 16:
        raise UnsupportedModel("near-tie probe index or roster is invalid")
    probe = policy["probes"][index]
    if first.get("prompt_token_ids") != probe["prompt_token_ids"] or second.get("prompt_token_ids") != probe["prompt_token_ids"]:
        raise UnsupportedModel("candidate prompt differs from the frozen base near-tie probe")
    if first.get("greedy_token_id") == second.get("greedy_token_id"):
        values = first.get("greedy_logprob"), second.get("greedy_logprob")
        if any(not is_finite_number(value) or value > 0 for value in values):
            raise UnsupportedModel("candidate greedy log probabilities are invalid")
        return abs(values[0] - values[1])
    allowed = set(probe["allowed_tokens"])
    if (not probe["near_tie"] or first.get("greedy_token_id") not in allowed or second.get("greedy_token_id") not in allowed):
        raise UnsupportedModel("candidate token mismatch is outside the frozen base near-tie allowance")
    left, right = top_two(first), top_two(second)
    if not allowed <= left.keys() or not allowed <= right.keys():
        raise UnsupportedModel("candidate did not measure both frozen near-tie tokens")
    return max(abs(left[token] - right[token]) for token in allowed)