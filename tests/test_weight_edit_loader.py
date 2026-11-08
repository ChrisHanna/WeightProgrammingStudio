from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from neural_foundry.backends.weight_edit.loader import UnsupportedModel, load_model, text_configuration
from neural_foundry.backends.weight_edit.discover import check_profile, find_blocks, find_sites
from neural_foundry.model_digest import model_state_digest


@pytest.mark.parametrize("field", ["hidden_size", "n_embd", "d_model"])
def test_hidden_size_comes_from_nested_text_config(field) -> None:
    text = SimpleNamespace(**{field: 24})
    config = SimpleNamespace(hidden_size=99, get_text_config=lambda: text)
    assert text_configuration(config) == (text, 24)


def test_missing_or_invalid_hidden_size_is_rejected() -> None:
    for text in (SimpleNamespace(), SimpleNamespace(hidden_size=True), SimpleNamespace(hidden_size=0)):
        with pytest.raises(UnsupportedModel, match="hidden size"):
            text_configuration(SimpleNamespace(get_text_config=lambda: text))


@pytest.mark.parametrize("architecture", ["GPTNeoX", "Qwen2", "Qwen3", "Llama", "Gemma", "Phi3", "GPT2"])
def test_loads_real_tiny_local_transformers_models(tmp_path, architecture) -> None:
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    common = {"vocab_size": 16, "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2,
              "num_attention_heads": 2, "num_key_value_heads": 2, "max_position_embeddings": 64,
              "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0}
    if architecture == "GPT2":
        config = transformers.GPT2Config(vocab_size=16, n_embd=16, n_layer=2, n_head=2, n_positions=64,
                                         bos_token_id=1, eos_token_id=2, pad_token_id=0)
    else:
        if architecture == "GPTNeoX":
            common["rotary_pct"] = 1.0
        if architecture == "Gemma":
            common["head_dim"] = 8
        if architecture == "Phi3":
            common["original_max_position_embeddings"] = 64
        config = getattr(transformers, architecture + "Config")(**common)
    model = transformers.AutoModelForCausalLM.from_config(config).eval()
    model.save_pretrained(tmp_path, safe_serialization=True)
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel({"[PAD]": 0, "[BOS]": 1, "[EOS]": 2,
                                                               "[UNK]": 3, "hello": 4}, unk_token="[UNK]"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                                      pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]")
    tokenizer.save_pretrained(tmp_path)
    loaded = load_model(tmp_path, dtype=torch.float32, device="cpu")
    assert loaded.hidden_size == 16
    assert loaded.base_digest == model_state_digest(model) == model_state_digest(loaded.model)
    assert loaded.model.training is False
    with torch.no_grad():
        expected = model(torch.tensor([[4, 4]])).logits
        actual = loaded.model(torch.tensor([[4, 4]])).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    blocks = find_blocks(loaded.model)
    assert blocks.count == 2
    expected_path = {"GPTNeoX": "gpt_neox.layers", "GPT2": "transformer.h"}.get(architecture, "model.layers")
    assert blocks.path == expected_path
    sites = find_sites(loaded.model, blocks, loaded.hidden_size)
    assert any(site.role == "residual_write_mlp" for site in sites.sites)
    assert sites.digest == find_sites(loaded.model, blocks, loaded.hidden_size).digest
    assert all(site.transposed == (architecture == "GPT2") for site in sites.sites)
    if architecture in {"GPTNeoX", "Qwen2", "Llama"}:
        from dataclasses import replace
        from neural_foundry.pythia import BUILTIN_ARCHITECTURE_PROFILES
        profile = BUILTIN_ARCHITECTURE_PROFILES[loaded.config.model_type]
        check_profile(sites, profile, blocks)
        with pytest.raises(UnsupportedModel, match="profile"):
            check_profile(sites, replace(profile, mlp_down_proj_template="missing.{layer}.weight"), blocks)


def test_loader_disallows_remote_code_and_pickle(monkeypatch, tmp_path) -> None:
    transformers = pytest.importorskip("transformers")
    from neural_foundry.backends.weight_edit import loader

    checkpoint = SimpleNamespace(path=tmp_path, file_digests={"config.json": "digest"})
    config = SimpleNamespace(get_text_config=lambda: SimpleNamespace(hidden_size=16))
    monkeypatch.setattr(loader, "resolve_checkpoint", lambda *args, **kwargs: checkpoint)

    def load_config(path, **kwargs):
        assert kwargs["local_files_only"] is True and kwargs["trust_remote_code"] is False
        return config

    def load_tokenizer(path, **kwargs):
        assert kwargs["local_files_only"] is True and kwargs["trust_remote_code"] is False
        return object()

    def load_weights(path, **kwargs):
        assert kwargs["local_files_only"] is True and kwargs["trust_remote_code"] is False
        assert kwargs["use_safetensors"] is True
        raise ValueError("vendor code required")

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", load_config)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", load_weights)
    with pytest.raises(UnsupportedModel, match="standard local"):
        load_model(tmp_path, dtype=torch.float32, device="cpu")


def test_discovery_rejects_tied_block_stack_candidates() -> None:
    model = torch.nn.Module()
    text = SimpleNamespace(hidden_size=16, num_hidden_layers=2)
    model.config = SimpleNamespace(get_text_config=lambda: text)
    model.first = torch.nn.ModuleList([torch.nn.Linear(16, 16) for _ in range(2)])
    model.second = torch.nn.ModuleList([torch.nn.Linear(16, 16) for _ in range(2)])
    with pytest.raises(UnsupportedModel, match="ambiguous blocks"):
        find_blocks(model)


def test_cached_qwen_checkpoint_loads_and_discovers_on_cuda() -> None:
    if os.environ.get("NF_WEIGHT_EDIT_LIVE") != "1":
        pytest.skip("set NF_WEIGHT_EDIT_LIVE=1 for the cached Qwen CUDA check")
    from neural_foundry.provenance import resolve_checkpoint

    checkpoint = resolve_checkpoint("Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca")
    loaded = load_model(checkpoint.path, dtype=torch.bfloat16, device="cuda:0")
    blocks = find_blocks(loaded.model)
    sites = find_sites(loaded.model, blocks, loaded.hidden_size)
    assert loaded.hidden_size == loaded.text_config.hidden_size
    assert blocks.count == loaded.text_config.num_hidden_layers
    assert any(site.role == "residual_write_mlp" for site in sites.sites)
    inputs = loaded.tokenizer("hello", return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        logits = loaded.model(**inputs).logits
    torch.cuda.synchronize()
    assert bool(torch.isfinite(logits).all())