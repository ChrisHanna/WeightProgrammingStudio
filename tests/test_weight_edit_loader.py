from __future__ import annotations

import os
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from neural_foundry.backends.weight_edit.loader import UnsupportedModel, WeightEditInputError, load_model, require_editable, text_configuration
from neural_foundry.backends.weight_edit.discover import check_profile, find_blocks, find_sites
from neural_foundry.backends.weight_edit.capture import capture, capture_statistics
from neural_foundry.backends.weight_edit.numerics import working_weight
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
        if architecture in {"Qwen3", "Gemma"}:
            common["head_dim"] = 12
            common["num_key_value_heads"] = 1
        if architecture == "Phi3":
            common["original_max_position_embeddings"] = 64
            common["num_key_value_heads"] = 1
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
    sites = find_sites(loaded.model, blocks, loaded.hidden_size, checkpoint_tensor_names=loaded.checkpoint_tensor_names)
    assert any(site.role == "residual_write_mlp" for site in sites.sites)
    assert sites.digest == find_sites(loaded.model, blocks, loaded.hidden_size, checkpoint_tensor_names=loaded.checkpoint_tensor_names).digest
    assert sites.checkpoint_names_verified is True
    with pytest.raises(UnsupportedModel, match="tensor index"):
        find_sites(loaded.model, blocks, loaded.hidden_size, checkpoint_tensor_names=frozenset())
    if loaded.shared_storage_groups:
        with pytest.raises(WeightEditInputError, match="shared"):
            require_editable(loaded, [loaded.shared_storage_groups[0][0]])
        require_editable(loaded, [loaded.shared_storage_groups[0][0]], allow_shared=True)
    assert all(site.transposed == (architecture == "GPT2") for site in sites.sites)
    role_names = {
        "GPTNeoX": {"attention.query_key_value": "residual_read_fused", "attention.dense": "residual_write_attn",
                    "mlp.dense_h_to_4h": "residual_read", "mlp.dense_4h_to_h": "residual_write_mlp"},
        "GPT2": {"attn.c_attn": "residual_read_fused", "attn.c_proj": "residual_write_attn",
                 "mlp.c_fc": "residual_read", "mlp.c_proj": "residual_write_mlp"},
        "Phi3": {"self_attn.qkv_proj": "residual_read_fused", "self_attn.o_proj": "residual_write_attn",
                 "mlp.gate_up_proj": "residual_read", "mlp.down_proj": "residual_write_mlp"},
    }.get(architecture, {"self_attn.q_proj": "residual_read", "self_attn.k_proj": "residual_read",
                         "self_attn.v_proj": "residual_read", "self_attn.o_proj": "residual_write_attn",
                         "mlp.gate_proj": "residual_read", "mlp.up_proj": "residual_read", "mlp.down_proj": "residual_write_mlp"})
    expected_roles = {f"{blocks.path}.{layer}.{name}.weight": role for layer in range(blocks.count) for name, role in role_names.items()}
    assert {site.tensor_name: site.role for site in sites.sites} == expected_roles
    captured = capture(loaded, sites, ["hello hello"])
    for site in sites.sites:
        values = captured[site.tensor_name]
        assert values.inputs.shape == (1, site.input_size)
        assert values.outputs.shape == (1, site.output_size)
        module = loaded.model.get_submodule(site.tensor_name.removesuffix(".weight"))
        weights = module.weight.detach().t() if site.transposed else module.weight.detach()
        expected_output = torch.nn.functional.linear(values.inputs, weights, module.bias.detach() if module.bias is not None else None)
        torch.testing.assert_close(values.outputs, expected_output, atol=1e-5, rtol=1e-5)
        assert not module._forward_hooks
    selected_sites = type(sites)((next(site for site in sites.sites if site.role == "residual_write_mlp"),))
    site = selected_sites.sites[0]
    weights = loaded.model.state_dict()[site.tensor_name]
    expected_working = weights.T if site.transposed else weights
    torch.testing.assert_close(working_weight(loaded, site), expected_working.to(torch.float64), rtol=0, atol=0)
    lower_precision = load_model(tmp_path, dtype=torch.bfloat16, device="cpu")
    torch.testing.assert_close(working_weight(lower_precision, site), expected_working.to(torch.float64), rtol=0, atol=0)
    with pytest.raises(WeightEditInputError, match="float32 or float64"):
        working_weight(loaded, site, compute_dtype=torch.bfloat16)
    retained = capture(loaded, selected_sites, ["hello hello", "hello"])
    streamed = capture_statistics(loaded, selected_sites, ["hello hello", "hello"])
    inputs = next(iter(retained.values())).inputs.to(torch.float64)
    outputs = next(iter(retained.values())).outputs.to(torch.float64)
    statistics = next(iter(streamed.values()))
    assert statistics.count == 2 and statistics.input_gram.dtype == torch.float64
    torch.testing.assert_close(statistics.input_gram, inputs.T @ inputs, rtol=0, atol=1e-12)
    torch.testing.assert_close(statistics.input_output_cross, inputs.T @ outputs, rtol=0, atol=1e-12)
    with pytest.raises(WeightEditInputError, match="byte budget"):
        capture(loaded, selected_sites, ["hello hello"], max_capture_bytes=1)
    with pytest.raises(WeightEditInputError, match="byte budget"):
        capture_statistics(loaded, selected_sites, ["hello hello"], max_statistics_bytes=1)
    response_tokens = capture(loaded, selected_sites, ["hello hello"], response_starts=[0])
    assert next(iter(response_tokens.values())).inputs.shape[0] == 2
    gradients = capture(loaded, selected_sites, ["hello hello"], loss_labels=[[4, 4]])
    assert next(iter(gradients.values())).output_gradients.shape[0] == 1
    assert all(parameter.grad is None for parameter in loaded.model.parameters())
    stale_sites = type(sites)((replace(selected_sites.sites[0], input_size=999),))
    with pytest.raises(UnsupportedModel, match="discovered map"):
        capture(loaded, stale_sites, ["hello hello"])
    assert all(not module._forward_hooks for module in loaded.model.modules())
    original_forward = loaded.model.forward
    def failing_forward(*args, **kwargs):
        raise RuntimeError("forward failed")
    loaded.model.forward = failing_forward
    try:
        with pytest.raises(RuntimeError, match="forward failed"):
            capture(loaded, selected_sites, ["hello hello"])
        assert all(not module._forward_hooks for module in loaded.model.modules())
    finally:
        loaded.model.forward = original_forward
    loaded.tokenizer.chat_template = "{{bos_token}} {% for message in messages %}{{message['content']}} {% endfor %}{% if add_generation_prompt %}{{eos_token}}{% endif %}"
    messages = [{"role": "user", "content": "hello hello"}]
    expected_ids = loaded.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    captured_template = capture(loaded, selected_sites, ["hello hello"])
    observed = []
    module = loaded.model.get_submodule(selected_sites.sites[0].tensor_name.removesuffix(".weight"))
    handle = module.register_forward_hook(lambda module, args, output: observed.append(args[0].detach().clone()))
    try:
        with torch.no_grad():
            loaded.model(input_ids=torch.tensor([expected_ids]), attention_mask=torch.ones((1, len(expected_ids)), dtype=torch.long))
    finally:
        handle.remove()
    values = next(iter(captured_template.values()))
    torch.testing.assert_close(values.inputs, observed[0][:, -1], rtol=0, atol=0)
    assert values.renderer_digest and len(values.prompt_digests) == 1
    with pytest.raises(WeightEditInputError, match="positions"):
        capture(loaded, selected_sites, ["hello hello"], positions=[[999]])
    assert not module._forward_hooks
    if architecture in {"GPTNeoX", "Qwen2", "Llama"}:
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


@pytest.mark.parametrize("shape", [(2, 16, 32), (16, 16)])
def test_discovery_rejects_unexplained_expert_or_matrix_weights(shape) -> None:
    from neural_foundry.backends.weight_edit.discover import BlockList

    block = torch.nn.Module()
    block.projection = torch.nn.Linear(16, 32)
    block.register_parameter("unexplained", torch.nn.Parameter(torch.zeros(shape)))
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([block])
    model.config = SimpleNamespace(get_text_config=lambda: SimpleNamespace(hidden_size=16, num_attention_heads=2))
    with pytest.raises(UnsupportedModel, match="unaccounted.*unexplained"):
        find_sites(model, BlockList("blocks", 1), 16)


def test_router_is_not_a_residual_edit_site() -> None:
    from neural_foundry.backends.weight_edit.discover import BlockList

    block = torch.nn.Module()
    block.router = torch.nn.Linear(16, 4, bias=False)
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([block])
    model.config = SimpleNamespace(get_text_config=lambda: SimpleNamespace(hidden_size=16, num_attention_heads=2, num_experts=4))
    sites = find_sites(model, BlockList("blocks", 1), 16, checkpoint_tensor_names=frozenset({"blocks.0.router.weight"}))
    assert {site.tensor_name: site.role for site in sites.sites} == {"blocks.0.router.weight": "router"}


def test_storage_groups_include_shared_views() -> None:
    from neural_foundry.backends.weight_edit.loader import storage_groups

    model = torch.nn.Module()
    weights = torch.randn(4, 4)
    model.first = torch.nn.Parameter(weights)
    model.second = torch.nn.Parameter(weights[1:])
    assert storage_groups(model) == (("first", "second"),)


def test_determinism_policy_is_scoped_even_on_failure() -> None:
    from neural_foundry.backends.weight_edit.numerics import deterministic_execution

    model = torch.nn.Linear(2, 2)
    original = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark,
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    with pytest.raises(RuntimeError, match="probe"):
        with deterministic_execution(model):
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.backends.cudnn.benchmark and not torch.backends.cuda.matmul.allow_tf32
            raise RuntimeError("probe")
    assert original == (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark,
                        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)


def test_cached_qwen_checkpoint_loads_and_discovers_on_cuda() -> None:
    if os.environ.get("NF_WEIGHT_EDIT_LIVE") != "1":
        pytest.skip("set NF_WEIGHT_EDIT_LIVE=1 for the cached Qwen CUDA check")
    from neural_foundry.provenance import resolve_checkpoint

    checkpoint = resolve_checkpoint("Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca")
    loaded = load_model(checkpoint.path, dtype=torch.bfloat16, device="cuda:0")
    blocks = find_blocks(loaded.model)
    sites = find_sites(loaded.model, blocks, loaded.hidden_size, checkpoint_tensor_names=loaded.checkpoint_tensor_names)
    assert loaded.hidden_size == loaded.text_config.hidden_size
    assert blocks.count == loaded.text_config.num_hidden_layers
    assert any(site.role == "residual_write_mlp" for site in sites.sites)
    from neural_foundry.pythia import BUILTIN_ARCHITECTURE_PROFILES
    check_profile(sites, BUILTIN_ARCHITECTURE_PROFILES["qwen3"], blocks)
    expected_roles = {f"{blocks.path}.{layer}.{name}.weight": role for layer in range(blocks.count)
                      for name, role in {"self_attn.q_proj": "residual_read", "self_attn.k_proj": "residual_read",
                                         "self_attn.v_proj": "residual_read", "self_attn.o_proj": "residual_write_attn",
                                         "mlp.gate_proj": "residual_read", "mlp.up_proj": "residual_read",
                                         "mlp.down_proj": "residual_write_mlp"}.items()}
    assert {site.tensor_name: site.role for site in sites.sites} == expected_roles
    assert any({"model.embed_tokens.weight", "lm_head.weight"} <= set(group) for group in loaded.shared_storage_groups)
    with pytest.raises(WeightEditInputError, match="shared"):
        require_editable(loaded, ["lm_head.weight"])
    inputs = loaded.tokenizer("hello", return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        logits = loaded.model(**inputs).logits
    torch.cuda.synchronize()
    assert bool(torch.isfinite(logits).all())