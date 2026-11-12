"""Shared bounded OpenAI-compatible wire transport, without Gym business logic."""

from __future__ import annotations

import ipaddress
import json
import time
import urllib.request
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from .gym_job import GymJobError
from .gym_service import ExecutionResponse, _document_digest
from .provenance import parse_strict_json

OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_DEFAULT_MODEL = "gpt-6-luna"


def response_digest(response: Any) -> str:
    return _document_digest(response if isinstance(response, Mapping) else {"response": response})


def validate_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    try:
        loopback = ipaddress.ip_address(parsed.hostname or "").is_loopback
    except ValueError:
        loopback = False
    if (parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username or parsed.password
        or parsed.query or parsed.fragment or (parsed.scheme == "http" and not loopback)):
        raise GymJobError("frontier endpoint requires HTTPS or literal loopback HTTP without URL credentials")
    return parsed.hostname


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request: Any, file: Any, code: int, message: str,
                         headers: Any, new_url: str) -> None:
        return None


def completion(*, endpoint: str, api_key: str, payload: Mapping[str, Any],
               interaction: Mapping[str, Any], receipt_fields: Mapping[str, Any],
               retain_invalid_completion: bool = False) -> ExecutionResponse:
    validate_endpoint(endpoint)
    requested_at_ns = time.time_ns()
    started = time.monotonic_ns()
    request = urllib.request.Request(
        endpoint, data=json.dumps(payload, allow_nan=False).encode(),
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=60) as response:
            raw = response.read(1_048_577)
        if len(raw) > 1_048_576:
            raise ValueError("response too large")
        document = parse_strict_json(raw)
        if (not isinstance(document.get("id"), str) or not document["id"].strip()
            or type(document.get("created")) is not int or document["created"] <= 0
            or (document.get("system_fingerprint") is not None
                and not isinstance(document["system_fingerprint"], str))):
            raise ValueError("response receipt fields missing")
        choices = document["choices"]
        if (document.get("model") != payload["model"] or not isinstance(choices, list) or len(choices) != 1
            or choices[0].get("finish_reason") not in
            ({"stop", "length"} if retain_invalid_completion else {"stop"})):
            raise ValueError("model or completion status differs")
        text = choices[0]["message"]["content"]
        if not isinstance(text, str):
            raise TypeError("completion has no text")
        if retain_invalid_completion:
            usage = document.get("usage")
            if (not isinstance(usage, Mapping)
                or any(type(usage.get(key)) is not int or usage[key] < 0
                       for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
                or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]
                or usage["completion_tokens"] > payload["max_completion_tokens"]):
                raise ValueError("provider usage missing or outside output authorization")
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        raise GymJobError("frontier request failed or returned an incomplete response") from error
    invalid_json = False
    try:
        parsed_response = parse_strict_json(text.encode("utf-8"))
    except ValueError:
        parsed_response = text
        invalid_json = True
    receipt = {"kind": "nf_openai_response_receipt_v1",
               "provider": "openai" if endpoint == OPENAI_CHAT_COMPLETIONS_URL else "openai_compatible",
               "response_id": document["id"], "reported_model": document["model"],
               "immutable_snapshot_verified": False, "usage": document.get("usage"),
               "system_fingerprint": document.get("system_fingerprint"), "response_created": document["created"],
               "requested_at_unix_ns": requested_at_ns, "latency_ns": time.monotonic_ns() - started,
               "request_digest": _document_digest(payload), "interaction_digest": _document_digest(interaction),
               "response_digest": response_digest(parsed_response), "wire_response_digest": _document_digest(document),
               **receipt_fields}
    if retain_invalid_completion:
        receipt["generation"] = {key: payload[key] for key in
                                 ("temperature", "reasoning_effort", "max_completion_tokens")}
        receipt["strict_json_schema"] = payload["response_format"]["json_schema"]["strict"]
        receipt["finish_reason"] = choices[0]["finish_reason"]
    receipt["receipt_digest"] = _document_digest(receipt)
    evidence = {"served_model_identity_verified": False, "api_response_receipt": receipt}
    if retain_invalid_completion:
        evidence.update({"raw_text": text, "raw_transport": document,
                         "finish_reason": choices[0]["finish_reason"],
                         "hit_generation_cap": choices[0]["finish_reason"] == "length",
                         "invalid_completion_json": invalid_json})
    return ExecutionResponse(parsed_response, evidence)
