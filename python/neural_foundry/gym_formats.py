"""Versioned interaction projections to and from NeMo Gym input rows."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .gym_job import GymJobError, _validate_schema
from .gym_service import _wire_document


def _copy(value: Any) -> Any:
    return json.loads(_wire_document(value))


def _messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise GymJobError("interaction requires a nonempty message list")
    for message in value:
        if (not isinstance(message, dict) or set(message) != {"role", "content"}
            or not isinstance(message["role"], str)
            or message["role"] not in {"system", "developer", "user", "assistant"}
            or not isinstance(message["content"], str)):
            raise GymJobError("interaction contains an unsupported message")
    return _copy(value)


def _to_nemo(interaction: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(interaction, Mapping):
        raise GymJobError("interaction must be an object")
    interaction_format = interaction.get("format")
    fields = {"format", "messages"}
    params: dict[str, Any] = {"input": _messages(interaction.get("messages"))}
    if interaction_format == "tool_messages_v1":
        fields |= {"tools", "max_tool_calls"}
        tools = interaction.get("tools")
        if (type(interaction.get("max_tool_calls")) is not int or interaction["max_tool_calls"] != 1
            or not isinstance(tools, list) or not 1 <= len(tools) <= 8):
            raise GymJobError("tool interaction requires one call and 1-8 tools")
        converted = []
        seen = set()
        for tool in tools:
            if (not isinstance(tool, dict) or set(tool) != {"name", "description", "arguments"}
                or not isinstance(tool["name"], str) or not tool["name"].strip()
                or tool["name"] in seen or not isinstance(tool["description"], str)):
                raise GymJobError("tool declaration is malformed or duplicated")
            _validate_schema(tool["arguments"], "tool.arguments")
            seen.add(tool["name"])
            converted.append({"type": "function", "name": tool["name"], "description": tool["description"],
                              "parameters": _copy(tool["arguments"]), "strict": False})
        params["tools"] = converted
        params["max_tool_calls"] = 1
    elif interaction_format == "json_extraction_v1":
        fields.add("output_schema")
        schema = interaction.get("output_schema")
        _validate_schema(schema, "output_schema")
        params["text"] = {"format": {"type": "json_schema", "name": "nf_output", "strict": True, "schema": _copy(schema)}}
    elif interaction_format != "chat_messages_v1":
        raise GymJobError("unknown interaction format")
    if set(interaction) != fields:
        raise GymJobError("interaction contains missing or unknown fields")
    return {"responses_create_params": params, "metadata": {"nf_interaction_format": interaction_format}}


def _from_nemo(row: Mapping[str, Any], interaction_format: str) -> dict[str, Any]:
    if (not isinstance(row, Mapping) or set(row) != {"responses_create_params", "metadata"}
        or row.get("metadata") != {"nf_interaction_format": interaction_format}
        or not isinstance(row.get("responses_create_params"), Mapping)):
        raise GymJobError("NeMo row does not match the declared interaction format")
    params = row["responses_create_params"]
    interaction: dict[str, Any] = {"format": interaction_format, "messages": _messages(params.get("input"))}
    if interaction_format == "tool_messages_v1":
        tools = params.get("tools")
        if not isinstance(tools, list):
            raise GymJobError("NeMo row is missing tool declarations")
        converted = []
        for tool in tools:
            if (not isinstance(tool, dict) or set(tool) != {"type", "name", "description", "parameters", "strict"}
                or tool["type"] != "function" or tool["strict"] is not False):
                raise GymJobError("NeMo tool declaration is unsupported")
            converted.append({"name": tool["name"], "description": tool["description"], "arguments": tool["parameters"]})
        interaction.update({"tools": converted, "max_tool_calls": params.get("max_tool_calls")})
    elif interaction_format == "json_extraction_v1":
        text = params.get("text")
        if not isinstance(text, dict) or set(text) != {"format"} or not isinstance(text["format"], dict):
            raise GymJobError("NeMo row is missing structured output configuration")
        schema_format = text["format"]
        if (set(schema_format) != {"type", "name", "strict", "schema"}
            or schema_format["type"] != "json_schema" or schema_format["name"] != "nf_output"
            or schema_format["strict"] is not True):
            raise GymJobError("NeMo structured output configuration is unsupported")
        interaction["output_schema"] = schema_format["schema"]
    if _to_nemo(interaction) != _copy(row):
        raise GymJobError("NeMo row contains unsupported parameters")
    return _copy(interaction)


@dataclass(frozen=True)
class InteractionFormat:
    format_id: str
    to_nemo: Callable[[Mapping[str, Any]], dict[str, Any]]
    from_nemo: Callable[[Mapping[str, Any]], dict[str, Any]]


class InteractionFormatRegistry:
    def __init__(self) -> None:
        self._formats: dict[str, InteractionFormat] = {}
        for format_id in ("chat_messages_v1", "tool_messages_v1", "json_extraction_v1"):
            self.register(InteractionFormat(format_id, lambda value, key=format_id: _to_nemo_for(value, key),
                                            lambda row, key=format_id: _from_nemo(row, key)))

    def register(self, adapter: InteractionFormat) -> None:
        if not isinstance(adapter.format_id, str) or not adapter.format_id or not callable(adapter.to_nemo) or not callable(adapter.from_nemo):
            raise GymJobError("interaction format requires a name and converters")
        if adapter.format_id in self._formats:
            raise GymJobError("interaction format is already registered")
        self._formats[adapter.format_id] = adapter

    def get(self, format_id: str) -> InteractionFormat:
        try:
            return self._formats[format_id]
        except (KeyError, TypeError) as exc:
            raise GymJobError("unknown interaction format") from exc


def _to_nemo_for(interaction: Mapping[str, Any], format_id: str) -> dict[str, Any]:
    if not isinstance(interaction, Mapping) or interaction.get("format") != format_id:
        raise GymJobError("interaction does not match the selected format")
    return _to_nemo(interaction)