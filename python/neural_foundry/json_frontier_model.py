"""Explicit OpenAI-compatible frontier client for synthetic JSON comparisons."""

from __future__ import annotations

import ipaddress
import json
import os
import shlex
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from .gym_job import GymJobError
from .gym_service import ExecutionResponse, _document_digest
from .provenance import parse_strict_json

OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_DEFAULT_MODEL = "gpt-6-luna"
_FRONTIER_ENV_NAMES = frozenset({
    "NF_FRONTIER_URL", "NF_FRONTIER_MODEL", "NF_FRONTIER_API_KEY", "OPENAI_API_KEY",
})


def frontier_proof_configuration() -> dict[str, Any]:
    from .reference_json_extraction import project, reference_task

    task = reference_task()
    document = {"kind": "nf_json_frontier_configuration_v1", "provider": "openai",
                "endpoint": OPENAI_CHAT_COMPLETIONS_URL, "model": OPENAI_DEFAULT_MODEL,
                "temperature": 0, "max_new_tokens": 128,
                "reasoning_effort": "none", "immutable_snapshot_verified": False,
                "maximum_api_attempts": 80, "maximum_cost_microusd": 50000,
                "prompt_template": project({"text": "{text}"}, None),
                "projection": task.projection.document(),
                "scorer": {"id": task.job.scorer.component_id, "version": task.job.scorer.version,
                           "implementation_digest": task.job.scorer.implementation_digest}}
    document["configuration_digest"] = _document_digest(document)
    return document


def _frontier_environment(env_file: str | os.PathLike[str] | None = None) -> dict[str, str]:
    values = {name: os.environ[name] for name in _FRONTIER_ENV_NAMES if os.environ.get(name)}
    path = Path(env_file) if env_file is not None else Path(__file__).resolve().parents[2] / ".env"
    if not path.is_file():
        return values
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise GymJobError("frontier environment file could not be read") from exc
    for line_number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, raw_value = line.partition("=")
        name = name.strip()
        if not separator or name not in _FRONTIER_ENV_NAMES or name in values:
            continue
        try:
            parsed = shlex.split(raw_value, comments=True, posix=True)
        except ValueError as exc:
            raise GymJobError(f"frontier environment syntax is invalid at line {line_number}") from exc
        if len(parsed) > 1:
            raise GymJobError(f"frontier environment syntax is invalid at line {line_number}")
        values[name] = parsed[0] if parsed else ""
    return values


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request: Any, file: Any, code: int, message: str, headers: Any, new_url: str) -> None:
        return None


class JsonFrontierModel:
    def __init__(self, *, endpoint: str, model: str, api_key: str, max_new_tokens: int = 128) -> None:
        parsed = urlsplit(endpoint)
        try:
            loopback = ipaddress.ip_address(parsed.hostname or "").is_loopback
        except ValueError:
            loopback = False
        if (parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or (parsed.scheme == "http" and not loopback)):
            raise GymJobError("frontier endpoint requires HTTPS or literal loopback HTTP without URL credentials")
        if not isinstance(model, str) or not model.strip() or not isinstance(api_key, str) or not api_key:
            raise GymJobError("frontier requires an explicit model and API credential")
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
            raise GymJobError("frontier output token budget is invalid")
        self._endpoint, self._model, self._api_key = endpoint, model, api_key
        self._host, self._max_new_tokens = parsed.hostname, max_new_tokens
        self._response_receipts: list[dict[str, Any]] = []
        self._proof_configuration: dict[str, Any] | None = None
        self._api_attempts = 0
        self._reserved_cost_microusd = 0

    def require_proof_configuration(self) -> None:
        if (self._endpoint != OPENAI_CHAT_COMPLETIONS_URL or self._model != OPENAI_DEFAULT_MODEL
            or self._max_new_tokens != 128):
            raise GymJobError("frontier proof requires the approved OpenAI endpoint, model ID and token cap")
        configuration = frontier_proof_configuration()
        if self._proof_configuration is not None and self._proof_configuration != configuration:
            raise GymJobError("frontier prompt or scorer changed after configuration was frozen")
        self._proof_configuration = configuration

    def verify_response_receipt(self, interaction: Mapping[str, Any], result: Any) -> None:
        self.require_proof_configuration()
        receipt = result.interaction.get("api_response_receipt") if isinstance(result, ExecutionResponse) else None
        if (not isinstance(receipt, Mapping) or receipt not in self._response_receipts
            or receipt.get("configuration_digest") != self._proof_configuration["configuration_digest"]
            or receipt.get("interaction_digest") != _document_digest(interaction)
            or receipt.get("response_digest") != _document_digest(result.response)):
            raise GymJobError("frontier response lacks a matching API response receipt")

    @classmethod
    def from_env(cls, *, env_file: str | os.PathLike[str] | None = None) -> JsonFrontierModel:
        values = _frontier_environment(env_file)
        return cls(
            endpoint=values.get("NF_FRONTIER_URL") or OPENAI_CHAT_COMPLETIONS_URL,
            model=values.get("NF_FRONTIER_MODEL") or OPENAI_DEFAULT_MODEL,
            api_key=values.get("NF_FRONTIER_API_KEY") or values.get("OPENAI_API_KEY", ""),
        )

    def __enter__(self) -> JsonFrontierModel:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        pass

    def loaded_identity(self) -> Mapping[str, Any]:
        return {"provider": "openai" if self._endpoint == OPENAI_CHAT_COMPLETIONS_URL else "openai_compatible",
            "pinned_model": self._model, "provider_reported_model": self._model, "provider_host": self._host,
            "api_response_receipts": [dict(receipt) for receipt in self._response_receipts],
            "proof_configuration_digest": self._proof_configuration["configuration_digest"] if self._proof_configuration else None,
            "scorer_digest": self._proof_configuration["scorer"]["implementation_digest"] if self._proof_configuration else None,
                "served_model_identity_verified": False}

    def infer(self, interaction: Mapping[str, Any]) -> ExecutionResponse:
        if (not isinstance(interaction, Mapping) or interaction.get("format") != "json_extraction_v1"
            or not isinstance(interaction.get("messages"), list) or not isinstance(interaction.get("output_schema"), Mapping)):
            raise GymJobError("frontier requires a JSON extraction interaction")
        if self._proof_configuration is not None:
            from .reference_json_extraction import project

            self.require_proof_configuration()
            prefix = self._proof_configuration["prompt_template"]["messages"][0]["content"].removesuffix("{text}")
            messages = interaction["messages"]
            content = messages[0].get("content") if len(messages) == 1 and isinstance(messages[0], Mapping) else None
            if (not isinstance(content, str) or not content.startswith(prefix)
                or interaction != project({"text": content[len(prefix):]}, None)):
                raise GymJobError("frontier proof interaction differs from the shared frozen prompt")
        payload = {"model": self._model, "messages": interaction["messages"], "temperature": 0,
                   "max_completion_tokens": self._max_new_tokens, "response_format": {"type": "json_schema", "json_schema": {
                       "name": "nf_output", "strict": True, "schema": interaction["output_schema"]}}}
        if self._model == OPENAI_DEFAULT_MODEL:
            payload["reasoning_effort"] = "none"
            input_token_bound = len(json.dumps(payload, allow_nan=False).encode()) + 1024
            reserved_cost = (input_token_bound + 7) // 8 + (self._max_new_tokens + 1) // 2
            if self._api_attempts >= 80 or self._reserved_cost_microusd + reserved_cost > 50000:
                raise GymJobError("approved frontier API attempt or cost ceiling reached")
            self._api_attempts += 1
            self._reserved_cost_microusd += reserved_cost
        requested_at_ns = time.time_ns()
        started = time.monotonic_ns()
        request = urllib.request.Request(
            self._endpoint,
            data=json.dumps(payload, allow_nan=False).encode(),
            headers={"Authorization": "Bearer " + self._api_key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.build_opener(_NoRedirect()).open(request, timeout=60) as response:
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
            if (document.get("model") != self._model or not isinstance(choices, list) or len(choices) != 1
                or choices[0].get("finish_reason") != "stop"):
                raise ValueError("model or completion status differs")
            text = choices[0]["message"]["content"]
            if not isinstance(text, str):
                raise ValueError("completion has no text")
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise GymJobError("frontier request failed or returned an incomplete response") from exc
        try:
            parsed_response = parse_strict_json(text.encode("utf-8"))
        except ValueError:
            parsed_response = text
        receipt = {"kind": "nf_openai_response_receipt_v1", "provider": "openai" if self._endpoint == OPENAI_CHAT_COMPLETIONS_URL else "openai_compatible",
                   "response_id": document["id"], "reported_model": document["model"],
                   "immutable_snapshot_verified": False, "usage": document.get("usage"),
                   "api_attempts": self._api_attempts, "reserved_cost_microusd": self._reserved_cost_microusd,
                   "system_fingerprint": document.get("system_fingerprint"), "response_created": document["created"],
                   "requested_at_unix_ns": requested_at_ns, "latency_ns": time.monotonic_ns() - started,
                   "request_digest": _document_digest(payload), "interaction_digest": _document_digest(interaction),
                   "response_digest": _document_digest(parsed_response), "wire_response_digest": _document_digest(document)}
        if self._proof_configuration is not None:
            receipt["configuration_digest"] = self._proof_configuration["configuration_digest"]
            receipt["scorer_digest"] = self._proof_configuration["scorer"]["implementation_digest"]
        receipt["receipt_digest"] = _document_digest(receipt)
        self._response_receipts.append(receipt)
        return ExecutionResponse(parsed_response, {"served_model_identity_verified": False,
                                                   "api_response_receipt": dict(receipt)})