"""Teacher-forced token sequences bound to the serving prompt renderer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from ...gym_service import _document_digest
from ...prompt_rendering import PromptRenderer, RenderedPrompt
from ...provenance import parse_strict_json
from .loader import WeightEditInputError


@dataclass(frozen=True)
class Demonstration:
    rendered: RenderedPrompt
    labels: tuple[int, ...]
    positions: tuple[int, ...]


def answer_token_groups(renderer: PromptRenderer, interaction: Mapping[str, Any]) -> list[str]:
    response = interaction["messages"][-1]["content"]
    response_ids = renderer.tokenizer.encode(response, add_special_tokens=False)
    if interaction.get("format") != "json_extraction_v1":
        return ["unclassified"] * len(response_ids) + ["end_of_answer"]
    values = parse_strict_json(response.encode())
    parts = ["{"]
    spans = []
    for index, field in enumerate(sorted(values)):
        prefix = ("," if index else "") + json.dumps(field) + ":"
        parts.append(prefix)
        start = sum(map(len, parts))
        literal = json.dumps(values[field])
        quoted = isinstance(values[field], str)
        spans.append((start + int(quoted), start + len(literal) - int(quoted), field))
        parts.append(literal)
    parts.append("}")
    if "".join(parts) != response:
        raise WeightEditInputError("token-group audit requires the exact compact sorted JSON demonstration")
    encoded = renderer.tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"] != response_ids:
        raise WeightEditInputError("token-group offsets differ from demonstration token IDs")
    groups = []
    for start, end in encoded["offset_mapping"]:
        if not 0 <= start < end <= len(response):
            groups.append("unclassified")
            continue
        overlaps = [(max(start, left), min(end, right), field) for left, right, field in spans
                    if start < right and end > left]
        if not overlaps:
            groups.append("format")
        elif len(overlaps) == 1 and overlaps[0][0] == start and overlaps[0][1] == end:
            groups.append(overlaps[0][2])
        else:
            groups.append("mixed_format_and_value")
    return groups + ["end_of_answer"]


def prepare_demonstration(renderer: PromptRenderer, interaction: Mapping[str, Any]) -> Demonstration:
    messages = interaction.get("messages")
    if (interaction.get("format") not in {"chat_messages_v1", "json_extraction_v1"}
        or not isinstance(messages, list) or len(messages) < 2
        or messages[-1].get("role") != "assistant" or not isinstance(messages[-1].get("content"), str)):
        raise WeightEditInputError("weight-edit demonstration requires an approved assistant response")
    response = messages[-1]["content"]
    prompt = renderer.render({**interaction, "messages": messages[:-1]})
    response_tokens = renderer.tokenizer.encode(response, add_special_tokens=False)
    eos = renderer.tokenizer.eos_token_id
    if type(eos) is not int or eos < 0 or not response_tokens:
        raise WeightEditInputError("demonstration requires response tokens and a declared EOS token")
    tokens = (*prompt.token_ids, *response_tokens, eos)
    digest = _document_digest({"kind": "nf_teacher_forcing_prompt_v1", "serving_prompt_digest": prompt.prompt_digest,
                               "renderer_digest": renderer.digest, "response": response, "token_ids": tokens})
    rendered = RenderedPrompt(prompt.text + response, tokens, renderer.digest, digest)
    labels = (*([-100] * len(prompt.token_ids)), *response_tokens, eos)
    positions = tuple(range(len(prompt.token_ids) - 1, len(tokens) - 1))
    return Demonstration(rendered, labels, positions)