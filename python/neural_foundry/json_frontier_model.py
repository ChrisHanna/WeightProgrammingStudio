"""Explicit OpenAI-compatible frontier client for synthetic JSON comparisons."""

from __future__ import annotations

import ipaddress
import json
import os
import urllib.error
import urllib.request
from typing import Any, Mapping
from urllib.parse import urlsplit

from .gym_job import GymJobError
from .gym_service import ExecutionResponse
from .provenance import parse_strict_json


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

    @classmethod
    def from_env(cls) -> JsonFrontierModel:
        return cls(endpoint=os.environ.get("NF_FRONTIER_URL", ""), model=os.environ.get("NF_FRONTIER_MODEL", ""),
                   api_key=os.environ.get("NF_FRONTIER_API_KEY", ""))

    def __enter__(self) -> JsonFrontierModel:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        pass

    def loaded_identity(self) -> Mapping[str, Any]:
        return {"provider_reported_model": self._model, "provider_host": self._host,
                "served_model_identity_verified": False}

    def infer(self, interaction: Mapping[str, Any]) -> ExecutionResponse:
        if (not isinstance(interaction, Mapping) or interaction.get("format") != "json_extraction_v1"
            or not isinstance(interaction.get("messages"), list) or not isinstance(interaction.get("output_schema"), Mapping)):
            raise GymJobError("frontier requires a JSON extraction interaction")
        payload = {"model": self._model, "messages": interaction["messages"], "temperature": 0,
                   "max_tokens": self._max_new_tokens, "response_format": {"type": "json_schema", "json_schema": {
                       "name": "nf_output", "strict": True, "schema": interaction["output_schema"]}}}
        request = urllib.request.Request(self._endpoint, data=json.dumps(payload, allow_nan=False).encode(),
                                         headers={"Authorization": "Bearer " + self._api_key,
                                                  "Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.build_opener(_NoRedirect()).open(request, timeout=60) as response:
                raw = response.read(1_048_577)
            if len(raw) > 1_048_576:
                raise ValueError("response too large")
            document = parse_strict_json(raw)
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
        return ExecutionResponse(parsed_response, {"served_model_identity_verified": False})