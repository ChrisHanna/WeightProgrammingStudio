"""Reusable public policies; no custody or credential dependencies."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from .gym_job import GymJobError
from .gym_service import _document_digest
from .provenance import parse_strict_json, sha256_file

ACTIVE_POLICY = Path(__file__).resolve().parents[2] / "protocols" / "public-proof-policy-v1.json"


def policy_reference(*, calls_authorized: bool = False) -> dict[str, Any]:
    if type(calls_authorized) is not bool:
        raise GymJobError("provider authorization must be explicit boolean data")
    return {"path": str(ACTIVE_POLICY), "file_digest": sha256_file(ACTIVE_POLICY),
            "calls_authorized": calls_authorized}


def load_policy(reference: Mapping[str, Any]) -> dict[str, Any]:
    try:
        path = Path(reference["path"])
        if not path.is_absolute() or sha256_file(path) != reference["file_digest"]:
            raise GymJobError("public policy file digest differs")
        document = parse_strict_json(path.read_bytes())
        version = document.get("version")
        if (document.get("kind") != "nf_public_proof_policy" or type(version) is not int
            or version not in (1, 2)
            or type(reference.get("calls_authorized")) is not bool):
            raise GymJobError("unsupported public policy reference")
        policy = deepcopy(document["policy"])
        api = policy["api"]
        if (policy["version"] != version
            or (version == 1 and (policy["max_new_tokens"] != 128 or policy["context_tokens"] != 512))
            or (version == 2 and (
                type(policy["max_new_tokens"]) is not int or type(policy["context_tokens"]) is not int
                or not 1 <= policy["max_new_tokens"] < policy["context_tokens"] <= 1048576
                or policy.get("report_only") is not True or policy.get("adaptive_retraining") is not False
                or api.get("automatic_retries") != 0 or type(api.get("automatic_retries")) is not int
                or api.get("model_fallback") is not False or api.get("cache_discount_assumed") is not False
                or api.get("invalid_response_rule") != "retain_raw_and_score_invalid_v1"))
            or policy["temperature"] != 0
            or policy["reasoning_effort"] != "none"
            or policy["identity"] != {"version": 1, "probe_count": 16, "logprob_atol": 1e-6,
                                      "shutdown_logprobs": "informational_only"}
            or api["version"] != 1 or api["budget_scope"] != "durable_frozen_run"
            or api["reservation_rule"] != "utf8_payload_bytes_plus_overhead_and_output_cap_rounded_up_v1"
            or api["failures_spend_reservations"] is not True
            or any(type(api[key]) is not int or api[key] < 1 for key in
                   ("maximum_api_attempts", "maximum_cost_microusd", "input_overhead_token_bound"))
            or not api["model_rates"]):
            raise GymJobError("unsupported active public policy values")
        for rates in api["model_rates"].values():
            if any(type(rates.get(key)) is not int or rates[key] < 1 for key in
                   ("input_microusd_per_million_tokens", "output_microusd_per_million_tokens")):
                raise GymJobError("invalid public provider price bounds")
        api["calls_authorized"] = reference["calls_authorized"]
        return policy
    except (OSError, KeyError, TypeError, ValueError, AttributeError) as error:
        raise GymJobError("public policy is missing or malformed") from error


def require_bound_policy(spec: Mapping[str, Any]) -> dict[str, Any]:
    reference = spec.get("policy_reference")
    if not isinstance(reference, Mapping):
        raise GymJobError("active provider requires a digest-bound reusable policy")
    policy = load_policy(reference)
    if _document_digest(policy) != _document_digest(spec.get("policy", {})):
        raise GymJobError("inline policy differs from its referenced policy")
    return policy
