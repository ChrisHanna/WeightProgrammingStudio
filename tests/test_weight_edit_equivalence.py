from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from contextlib import contextmanager
from unittest.mock import patch
import json

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


FLOAT32_EPS_FACTOR = 8
"""The two-layer CPU fixtures measured at most about four eps of per-site relative
error; eight eps gives twofold headroom for accumulation order on these fixtures.
This is an evidence-scoped bound, not a guarantee for larger models or GPU kernels.
"""

TRANSFORMERS_FLOAT32_INTERNALS = {
    "gpt_neox": ("attention_softmax", "rotary"),
    "qwen2": ("attention_softmax", "rotary", "RMSNorm"),
}


def _relative_error(actual, expected):
    error = float((actual - expected).abs().max())
    scale = float(expected.abs().max())
    return error / scale if scale else (0.0 if error == 0.0 else float("inf"))


@contextmanager
def _legacy_attention_float32():
    original_outer, original_softmax = torch.outer, torch.softmax

    def rotary_outer(left, right, **kwargs):
        return original_outer(left.float(), right.float(), **kwargs)

    def attention_softmax(value, *args, **kwargs):
        kwargs["dtype"] = torch.float32
        return original_softmax(value, *args, **kwargs).to(value.dtype)

    with patch.object(torch, "outer", rotary_outer), patch.object(torch, "softmax", attention_softmax):
        yield


@pytest.mark.parametrize("factory", [build_mini_pythia_model, build_mini_qwen2_causal_lm])
@pytest.mark.parametrize("dtype,tolerance", [(torch.float64, 1e-10), (torch.float32, 1e-5)])
def test_engine_matches_transformers_capture_and_native_edit(tmp_path, factory, dtype, tolerance, record_property) -> None:
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    old = factory()
    old.tensors = {name: tensor.to(dtype) for name, tensor in old.tensors.items()}
    config_data = {name: value for name, value in old.config.items() if name != "model_type"}
    config = transformers.AutoConfig.for_model(old.config["model_type"], **config_data)
    model = transformers.AutoModelForCausalLM.from_config(config, attn_implementation="eager").to(dtype).eval()
    result = model.load_state_dict(old.tensors, strict=False)
    assert not result.unexpected_keys
    assert all(name.endswith(".bias") for name in result.missing_keys), result.missing_keys
    with torch.no_grad():
        for name in result.missing_keys:
            model.get_parameter(name).zero_()
    model.save_pretrained(tmp_path, safe_serialization=True)
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(dict(old.tokenizer.vocab), unk_token="<pad>"))
    transformers.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<pad>").save_pretrained(tmp_path)
    loaded = load_model(tmp_path, dtype=dtype, device="cpu")
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
    recorded = capture(loaded, sites, prompts, output_dtype=dtype)
    trials = []
    modes = ("legacy", "attention_float32", "match_transformers_internals")
    record_property("float32_internals", json.dumps(TRANSFORMERS_FLOAT32_INTERNALS[old.config["model_type"]]))
    for mode in modes:
        from contextlib import nullcontext
        with _legacy_attention_float32() if mode == "attention_float32" else nullcontext():
            for index, prompt in enumerate(prompts):
                expected_inputs = {}
                old.forward_with_mlp_key(old.tokenizer.encode(prompt), site_inputs=expected_inputs,
                                         match_transformers_internals=mode == "match_transformers_internals")
                for name, expected_input in expected_inputs.items():
                    trials.append({"mode": mode, "prompt": prompt, "site": name,
                                   "relative_error": _relative_error(recorded[name].inputs[index], expected_input)})
    record_property("precision_trials", json.dumps(trials, sort_keys=True))
    summary = {}
    for mode in modes:
        selected = [trial for trial in trials if trial["mode"] == mode]
        summary[mode] = max(selected, key=lambda trial: trial["relative_error"])
    print(json.dumps({"family": old.config["model_type"], "dtype": str(dtype), "max_relative_errors": summary}, sort_keys=True))
    for index, prompt in enumerate(prompts):
        tokens = old.tokenizer.encode(prompt)
        inputs = {}
        logits, _ = old.forward_with_mlp_key(tokens, site_inputs=inputs, match_transformers_internals=True)
        with torch.no_grad():
            actual = loaded.model(input_ids=torch.tensor([tokens])).logits[0]
        expected_exposed = {profile.mlp_down_proj_site(layer) for layer in range(blocks.count)}
        expected_exposed |= {profile.mlp_up_proj_site(layer) for layer in range(blocks.count)}
        expected_exposed |= {profile.attn_out_site(layer) for layer in range(blocks.count)}
        if old.config["model_type"] == "qwen2":
            expected_exposed |= {f"model.layers.{layer}.mlp.gate_proj.weight" for layer in range(blocks.count)}
        assert set(inputs) == expected_exposed
        site_by_name = {site.tensor_name: site for site in sites.sites}
        ordered_sites = [site_by_name[name + ".weight"] for name, _ in loaded.model.named_modules()
                         if name + ".weight" in expected_exposed]
        for site in ordered_sites:
            values = recorded[site.tensor_name]
            assert values.inputs.dtype == dtype and values.outputs.dtype == dtype
            error = _relative_error(values.inputs[index], inputs[site.tensor_name])
            relative_tolerance = 1e-12 if dtype == torch.float64 else FLOAT32_EPS_FACTOR * torch.finfo(torch.float32).eps
            assert error <= relative_tolerance, (f"first activation divergence: {old.config['model_type']} {dtype}, "
                f"layer={site.layer}, site={site.tensor_name}, role={site.role}, prompt={prompt!r}, "
                f"input_relative_error={error:.17g}, tolerance={relative_tolerance}")
            weight = old.tensors[site.tensor_name]
            bias = old.tensors.get(site.tensor_name.removesuffix("weight") + "bias")
            expected_output = torch.nn.functional.linear(inputs[site.tensor_name], weight, bias)
            error = _relative_error(values.outputs[index], expected_output)
            assert error <= relative_tolerance, (f"first activation divergence: {old.config['model_type']} {dtype}, "
                f"layer={site.layer}, site={site.tensor_name}, role={site.role}, prompt={prompt!r}, "
                f"output_relative_error={error:.17g}, tolerance={relative_tolerance}")
        assert _relative_error(actual, logits) <= relative_tolerance
        assert actual[-1].argmax().item() == logits[-1].argmax().item()
    site = next(site for site in sites.sites if site.role == "residual_write_mlp")
    original = working_weight(loaded, site)
    left = torch.linspace(-0.0001, 0.0001, site.output_size, dtype=torch.float64).unsqueeze(1)
    keys = recorded[site.tensor_name].inputs[:1].to(torch.float64)
    retention = recorded[site.tensor_name].inputs[1:].to(torch.float64)
    gram = keys.T @ keys + 20.0 * (retention.T @ retention) + 1e-3 * torch.eye(site.input_size, dtype=torch.float64)
    right = torch.linalg.solve(gram, keys.T).T.contiguous()
    calculated = LegacyKeyValueEdit(retention).compute(site, original, keys, keys @ original.T + left.T,
                                                       {"retention_nullspace_weight": 20.0})
    target_scale = float((keys @ original.T + left.T).abs().max())
    cancellation_bound = 16 * torch.finfo(torch.float64).eps * max(target_scale, 1.0)
    torch.testing.assert_close(calculated.U, left, atol=cancellation_bound, rtol=0)
    torch.testing.assert_close(calculated.V, right, atol=0, rtol=0)
    legacy, *_ = _compile_and_apply_rust_low_rank_update(w_orig=original, left_factors=left,
                                                        right_factors=right, smoke_probe_vec=right[0])
    transaction = ir.WeightIRTransactionBuilder.execute_matrix_edit(original, operation_type="low_rank_update",
                   left_factor=calculated.U, right_factor=calculated.V, factor=1.0, smoke_probe=right[0],
                   provenance_tag="generic.equivalence.v1", engine_name="generic.equivalence")
    ir.require_reload_parity(transaction)
    updated = torch.frombuffer(bytearray(transaction.edited_bytes), dtype=torch.float64).reshape(original.shape).clone()
    assert _relative_error(updated, legacy) <= 1e-12
    with torch.no_grad():
        loaded.model.get_parameter(site.tensor_name).copy_(updated.to(dtype))
    for prompt in prompts:
        tokens = old.tokenizer.encode(prompt)
        expected, _ = old.forward_with_mlp_key(tokens, layer_down_proj_override={site.layer: legacy.to(dtype)},
                                             match_transformers_internals=True)
        with torch.no_grad():
            actual = loaded.model(input_ids=torch.tensor([tokens])).logits[0]
        assert _relative_error(actual, expected) <= relative_tolerance
        assert actual[-1].argmax().item() == expected[-1].argmax().item()


def test_qwen_rmsnorm_has_a_separate_float32_precision_boundary(record_property) -> None:
    pytest.importorskip("transformers")
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    old = build_mini_qwen2_causal_lm()
    ids = old.tokenizer.encode("Hello the world")
    inputs = old.tensors["model.embed_tokens.weight"][ids]
    weight = old.tensors["model.layers.0.input_layernorm.weight"]
    epsilon = old.config["rms_norm_eps"]
    norm = Qwen2RMSNorm(inputs.shape[-1], eps=epsilon).double()
    with torch.no_grad():
        norm.weight.copy_(weight)
        observed = norm(inputs)
    reference64 = inputs * torch.rsqrt(inputs.pow(2).mean(-1, keepdim=True) + epsilon) * weight
    values32 = inputs.float()
    reference_mixed = weight * (values32 * torch.rsqrt(values32.pow(2).mean(-1, keepdim=True) + epsilon)).double()
    torch.testing.assert_close(observed, reference_mixed, atol=0, rtol=0)
    error = _relative_error(observed, reference64)
    assert error > 1e-12, "RMSNorm policy changed; revisit the strict float64 equivalence gate"
    record_property("rmsnorm_relative_error", error)
    print(json.dumps({"family": "qwen2", "site": "model.layers.0.input_layernorm.weight",
                      "relative_error_vs_float64": error, "mixed_precision_reference_exact": True}))