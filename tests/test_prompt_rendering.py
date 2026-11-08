from __future__ import annotations

import pytest

from neural_foundry.prompt_rendering import PromptInputError, PromptRenderer


def _tokenizer():
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel({"[UNK]": 0, "[USER]": 1, "[ASSISTANT]": 2,
                                                               "hello": 3, "[BOS]": 4}, unk_token="[UNK]"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", bos_token="[BOS]")
    tokenizer.chat_template = "{{ bos_token }} {% for message in messages %}[USER] {{ message['content'] }} {% endfor %}{% if add_generation_prompt %}[ASSISTANT]{% endif %}"
    return tokenizer


def test_renderer_applies_template_once_and_records_token_policy() -> None:
    tokenizer = _tokenizer()
    renderer = PromptRenderer(tokenizer)
    result = renderer.render("hello")
    assert result.token_ids == (4, 1, 3, 2)
    assert result.token_ids != tuple(tokenizer.encode("hello"))
    assert renderer.render({"format": "chat_messages_v1", "messages": [{"role": "user", "content": "hello"}]}).token_ids == result.token_ids
    assert renderer.render({"format": "raw_text_v1", "text": "hello"}).token_ids == (3,)
    assert result.renderer_digest == renderer.digest
    tokenizer.chat_template += " hello"
    assert PromptRenderer(tokenizer).digest != renderer.digest


def test_renderer_refuses_chat_without_template() -> None:
    tokenizer = _tokenizer()
    tokenizer.chat_template = None
    with pytest.raises(PromptInputError, match="chat template"):
        PromptRenderer(tokenizer).render({"format": "chat_messages_v1", "messages": [{"role": "user", "content": "hello"}]})