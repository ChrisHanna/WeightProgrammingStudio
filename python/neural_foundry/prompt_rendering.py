"""Shared prompt specification and deterministic chat-template rendering."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .gym_service import _document_digest, _wire_document


class PromptInputError(ValueError):
    """A prompt does not satisfy the declared rendering contract."""


def prompt_specification(prompt: str | Mapping[str, Any], *, has_chat_template: bool) -> dict[str, Any]:
    if isinstance(prompt, str):
        if not has_chat_template:
            return {"format": "raw_text_v1", "text": prompt}
        return {"format": "chat_messages_v1", "messages": [{"role": "user", "content": prompt}]}
    if not isinstance(prompt, Mapping):
        raise PromptInputError("prompt must be text or a declared interaction")
    kind = prompt.get("format")
    if kind == "raw_text_v1":
        if set(prompt) != {"format", "text"} or not isinstance(prompt["text"], str):
            raise PromptInputError("raw text prompt is malformed")
    elif kind in {"chat_messages_v1", "json_extraction_v1", "tool_messages_v1"}:
        messages = prompt.get("messages")
        if not has_chat_template:
            raise PromptInputError("chat interaction requires a tokenizer chat template")
        if (not isinstance(messages, list) or not messages
            or any(not isinstance(message, Mapping) or set(message) != {"role", "content"}
                   or message["role"] not in {"system", "developer", "user", "assistant"}
                   or not isinstance(message["content"], str) for message in messages)):
            raise PromptInputError("chat messages are malformed")
    else:
        raise PromptInputError("unsupported prompt format")
    return json.loads(_wire_document(prompt))


def projection_digest(prompt: Mapping[str, Any]) -> str:
    specification = prompt_specification(prompt, has_chat_template=True)
    return _document_digest({"kind": "nf_prompt_projection_v1", "specification": specification,
                             "implementation": "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


@dataclass(frozen=True)
class RenderedPrompt:
    text: str
    token_ids: tuple[int, ...]
    renderer_digest: str
    prompt_digest: str


class PromptRenderer:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        backend = getattr(tokenizer, "backend_tokenizer", None)
        encoding = backend.to_str() if backend is not None else json.dumps(tokenizer.get_vocab(), sort_keys=True)
        self.digest = _document_digest({
            "kind": "nf_prompt_renderer_v1", "implementation": "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "tokenizer": "sha256:" + hashlib.sha256(encoding.encode()).hexdigest(),
            "chat_template": getattr(tokenizer, "chat_template", None),
            "special_tokens": {name: str(value) for name, value in tokenizer.special_tokens_map.items()},
            "add_bos_token": getattr(tokenizer, "add_bos_token", None), "add_eos_token": getattr(tokenizer, "add_eos_token", None),
            "enable_thinking": False, "add_generation_prompt": True,
        })

    def render(self, prompt: str | Mapping[str, Any]) -> RenderedPrompt:
        specification = prompt_specification(prompt, has_chat_template=bool(getattr(self.tokenizer, "chat_template", None)))
        if specification["format"] == "raw_text_v1":
            text = specification["text"]
            token_ids = self.tokenizer.encode(text, add_special_tokens=True)
        else:
            text = self.tokenizer.apply_chat_template(specification["messages"], tokenize=False,
                                                       add_generation_prompt=True, enable_thinking=False)
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not token_ids or any(type(token) is not int or token < 0 for token in token_ids):
            raise PromptInputError("rendered prompt has no valid token IDs")
        digest = _document_digest({"kind": "nf_rendered_prompt_v1", "renderer_digest": self.digest,
                                   "specification": specification, "token_ids": token_ids})
        return RenderedPrompt(text, tuple(token_ids), self.digest, digest)