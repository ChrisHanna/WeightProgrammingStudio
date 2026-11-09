"""Teacher-forced token sequences bound to the serving prompt renderer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ...gym_service import _document_digest
from ...prompt_rendering import PromptRenderer, RenderedPrompt
from .loader import WeightEditInputError


@dataclass(frozen=True)
class Demonstration:
    rendered: RenderedPrompt
    labels: tuple[int, ...]
    positions: tuple[int, ...]


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