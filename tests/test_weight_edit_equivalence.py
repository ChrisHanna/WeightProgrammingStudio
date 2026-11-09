from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from neural_foundry import ir
from neural_foundry.backends.weight_edit.capture import capture
from neural_foundry.backends.weight_edit.discover import SiteMap, check_profile, find_blocks, find_sites
from neural_foundry.backends.weight_edit.loader import load_model
from neural_foundry.backends.weight_edit.numerics import working_weight
from neural_foundry.backends.weight_edit.methods.base import LegacyKeyValueEdit
from neural_foundry.pythia import (
    BUILTIN_ARCHITECTURE_PROFILES, _compile_and_apply_rust_low_rank_update,
    build_mini_pythia_model, build_mini_qwen2_causal_lm,
)


@pytest.mark.parametrize("factory", [build_mini_pythia_model, build_mini_qwen2_causal_lm])
def test_original_float64_engine_matches_transformers_capture_and_native_edit(tmp_path, factory) -> None:
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    old = factory()
    config_data = {name: value for name, value in old.config.items() if name != "model_type"}
    config = transformers.AutoConfig.for_model(old.config["model_type"], **config_data)
    model = transformers.AutoModelForCausalLM.from_config(config, attn_implementation="eager").to(torch.float64).eval()
    result = model.load_state_dict(old.tensors, strict=False)
    assert not result.unexpected_keys
    assert all(name.endswith(".bias") for name in result.missing_keys), result.missing_keys
    with torch.no_grad():
        for name in result.missing_keys:
            model.get_parameter(name).zero_()
    model.save_pretrained(tmp_path, safe_serialization=True)
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(dict(old.tokenizer.vocab), unk_token="<pad>"))
    transformers.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<pad>").save_pretrained(tmp_path)
    loaded = load_model(tmp_path, dtype=torch.float64, device="cpu")
    loaded.model.set_attn_implementation("eager")
    fixture_tokenizer = SimpleNamespace(chat_template=None, special_tokens_map={}, get_vocab=lambda: dict(old.tokenizer.vocab),
                                        encode=lambda text, **kwargs: old.tokenizer.encode(text))
    loaded = replace(loaded, tokenizer=fixture_tokenizer)
    blocks = find_blocks(loaded.model)
    sites = find_sites(loaded.model, blocks, loaded.hidden_size, checkpoint_tensor_names=loaded.checkpoint_tensor_names)
    profile = BUILTIN_ARCHITECTURE_PROFILES[old.config["model_type"]]
    check_profile(sites, profile, blocks)
    expected_names = {name for name, value in old.tensors.items() if name.startswith(blocks.path + ".") and value.ndim == 2}
    assert {site.tensor_name for site in sites.sites} == expected_names
    prompts = ("Hello the world", "Today the capital", "One plus one")
    recorded = capture(loaded, sites, prompts)
    for index, prompt in enumerate(prompts):
        tokens = old.tokenizer.encode(prompt)
        inputs = {}
        logits, _ = old.forward_with_mlp_key(tokens, site_inputs=inputs)
        with torch.no_grad():
            actual = loaded.model(input_ids=torch.tensor([tokens])).logits[0]
        torch.testing.assert_close(actual, logits, atol=1e-5, rtol=0)
        assert actual[-1].argmax().item() == logits[-1].argmax().item()
        expected_exposed = {profile.mlp_down_proj_site(layer) for layer in range(blocks.count)}
        expected_exposed |= {profile.mlp_up_proj_site(layer) for layer in range(blocks.count)}
        expected_exposed |= {profile.attn_out_site(layer) for layer in range(blocks.count)}
        if old.config["model_type"] == "qwen2":
            expected_exposed |= {f"model.layers.{layer}.mlp.gate_proj.weight" for layer in range(blocks.count)}
        assert set(inputs) == expected_exposed
        for site in sites.sites:
            if site.tensor_name not in expected_exposed:
                continue
            values = recorded[site.tensor_name]
            torch.testing.assert_close(values.inputs[index], inputs[site.tensor_name].to(torch.float32), atol=1e-5, rtol=0)
            weight = old.tensors[site.tensor_name]
            bias = old.tensors.get(site.tensor_name.removesuffix("weight") + "bias")
            expected_output = torch.nn.functional.linear(inputs[site.tensor_name], weight, bias)
            torch.testing.assert_close(values.outputs[index], expected_output.to(torch.float32), atol=1e-5, rtol=0)
    site = next(site for site in sites.sites if site.role == "residual_write_mlp")
    original = working_weight(loaded, site)
    left = torch.linspace(-0.0001, 0.0001, site.output_size, dtype=torch.float64).unsqueeze(1)
    keys = recorded[site.tensor_name].inputs[:1].to(torch.float64)
    retention = recorded[site.tensor_name].inputs[1:].to(torch.float64)
    gram = keys.T @ keys + 20.0 * (retention.T @ retention) + 1e-3 * torch.eye(site.input_size, dtype=torch.float64)
    right = torch.linalg.solve(gram, keys.T).T.contiguous()
    calculated = LegacyKeyValueEdit(retention).compute(site, original, keys, keys @ original.T + left.T,
                                                       {"retention_nullspace_weight": 20.0})
    torch.testing.assert_close(calculated.U, left, atol=1e-5, rtol=0)
    torch.testing.assert_close(calculated.V, right, atol=0, rtol=0)
    legacy, *_ = _compile_and_apply_rust_low_rank_update(w_orig=original, left_factors=left,
                                                        right_factors=right, smoke_probe_vec=right[0])
    transaction = ir.WeightIRTransactionBuilder.execute_matrix_edit(original, operation_type="low_rank_update",
                   left_factor=calculated.U, right_factor=calculated.V, factor=1.0, smoke_probe=right[0],
                   provenance_tag="generic.equivalence.v1", engine_name="generic.equivalence")
    ir.require_reload_parity(transaction)
    updated = torch.frombuffer(bytearray(transaction.edited_bytes), dtype=torch.float64).reshape(original.shape).clone()
    torch.testing.assert_close(updated, legacy, atol=1e-5, rtol=0)
    with torch.no_grad():
        loaded.model.get_parameter(site.tensor_name).copy_(updated)
    for prompt in prompts:
        tokens = old.tokenizer.encode(prompt)
        expected, _ = old.forward_with_mlp_key(tokens, layer_down_proj_override={site.layer: legacy})
        with torch.no_grad():
            actual = loaded.model(input_ids=torch.tensor([tokens])).logits[0]
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)
        assert actual[-1].argmax().item() == expected[-1].argmax().item()