"""Explicit OpenAI-compatible frontier client for synthetic JSON comparisons."""

from __future__ import annotations

import json
import os
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

from .gym_job import GymJobError
from .gym_service import ExecutionResponse, _document_digest
from .openai_transport import (
    OPENAI_CHAT_COMPLETIONS_URL,
    OPENAI_DEFAULT_MODEL,
    completion,
    validate_endpoint,
)
from .openai_transport import (
    NoRedirect as _NoRedirect,
)

__all__ = [
    "OPENAI_CHAT_COMPLETIONS_URL",
    "OPENAI_DEFAULT_MODEL",
    "JsonFrontierModel",
    "_NoRedirect",
    "frontier_proof_configuration",
]

_FRONTIER_ENV_NAMES = frozenset({
    "NF_FRONTIER_URL", "NF_FRONTIER_MODEL", "NF_FRONTIER_API_KEY", "OPENAI_API_KEY",
})


def frontier_proof_configuration(*, maximum_api_attempts: int = 80,
                                 maximum_cost_microusd: int = 50000) -> dict[str, Any]:
    from .reference_json_extraction import project, reference_task

    if (type(maximum_api_attempts) is not int or maximum_api_attempts < 1
        or type(maximum_cost_microusd) is not int or maximum_cost_microusd < 1):
        raise GymJobError("frontier proof limits must be positive integers")
    task = reference_task()
    document = {"kind": "nf_json_frontier_configuration_v1", "provider": "openai",
                "endpoint": OPENAI_CHAT_COMPLETIONS_URL, "model": OPENAI_DEFAULT_MODEL,
                "temperature": 0, "max_new_tokens": 128,
                "reasoning_effort": "none", "immutable_snapshot_verified": False,
                "maximum_api_attempts": maximum_api_attempts,
                "maximum_cost_microusd": maximum_cost_microusd,
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


class JsonFrontierModel:
    def __init__(self, *, endpoint: str, model: str, api_key: str, max_new_tokens: int = 128,
                 maximum_api_attempts: int = 80, maximum_cost_microusd: int = 50000) -> None:
        host = validate_endpoint(endpoint)
        if not isinstance(model, str) or not model.strip() or not isinstance(api_key, str) or not api_key:
            raise GymJobError("frontier requires an explicit model and API credential")
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
            raise GymJobError("frontier output token budget is invalid")
        if (type(maximum_api_attempts) is not int or maximum_api_attempts < 1
            or type(maximum_cost_microusd) is not int or maximum_cost_microusd < 1):
            raise GymJobError("frontier proof limits must be positive integers")
        self._endpoint, self._model, self._api_key = endpoint, model, api_key
        self._host, self._max_new_tokens = host, max_new_tokens
        self._maximum_api_attempts = maximum_api_attempts
        self._maximum_cost_microusd = maximum_cost_microusd
        self._response_receipts: list[dict[str, Any]] = []
        self._proof_configuration: dict[str, Any] | None = None
        self._api_attempts = 0
        self._reserved_cost_microusd = 0

    def require_proof_configuration(self) -> None:
        if (self._endpoint != OPENAI_CHAT_COMPLETIONS_URL or self._model != OPENAI_DEFAULT_MODEL
            or self._max_new_tokens != 128):
            raise GymJobError("frontier proof requires the approved OpenAI endpoint, model ID and token cap")
        configuration = frontier_proof_configuration(
            maximum_api_attempts=self._maximum_api_attempts,
            maximum_cost_microusd=self._maximum_cost_microusd)
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
    def from_env(cls, *, env_file: str | os.PathLike[str] | None = None,
                 maximum_api_attempts: int = 80,
                 maximum_cost_microusd: int = 50000) -> JsonFrontierModel:
        values = _frontier_environment(env_file)
        return cls(
            endpoint=values.get("NF_FRONTIER_URL") or OPENAI_CHAT_COMPLETIONS_URL,
            model=values.get("NF_FRONTIER_MODEL") or OPENAI_DEFAULT_MODEL,
            api_key=values.get("NF_FRONTIER_API_KEY") or values.get("OPENAI_API_KEY", ""),
            maximum_api_attempts=maximum_api_attempts,
            maximum_cost_microusd=maximum_cost_microusd,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
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
            if (self._api_attempts >= self._maximum_api_attempts
                or self._reserved_cost_microusd + reserved_cost > self._maximum_cost_microusd):
                raise GymJobError("approved frontier API attempt or cost ceiling reached")
            self._api_attempts += 1
            self._reserved_cost_microusd += reserved_cost
        fields: dict[str, Any] = {"api_attempts": self._api_attempts,
                                  "reserved_cost_microusd": self._reserved_cost_microusd}
        if self._proof_configuration is not None:
            fields["configuration_digest"] = self._proof_configuration["configuration_digest"]
            fields["scorer_digest"] = self._proof_configuration["scorer"]["implementation_digest"]
        result = completion(endpoint=self._endpoint, api_key=self._api_key, payload=payload,
                            interaction=interaction, receipt_fields=fields)
        receipt = dict(result.interaction["api_response_receipt"])
        self._response_receipts.append(receipt)
        return result