from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from neural_foundry.backends.weight_edit.apply import apply, save, tensor_digests
from neural_foundry.backends.weight_edit.discover import Site, SiteMap, find_blocks, find_sites
from neural_foundry.backends.weight_edit.loader import WeightEditInputError, load_model
from neural_foundry.backends.weight_edit.methods import Delta
from neural_foundry.model_digest import model_state_digest


@pytest.fixture(params=["qwen2", "gpt2"])
def loaded(tmp_path, request):
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    if request.param == "gpt2":
        config = transformers.GPT2Config(vocab_size=16, n_embd=16, n_layer=2, n_head=2, n_positions=64)
    else:
        config = transformers.Qwen2Config(vocab_size=16, hidden_size=16, intermediate_size=32,
                                           num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1)
    model = transformers.AutoModelForCausalLM.from_config(config).eval()
    base = tmp_path / "base"
    model.save_pretrained(base, safe_serialization=True)
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    transformers.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]").save_pretrained(base)
    return load_model(base, dtype=torch.float32, device="cpu")


def test_apply_exports_only_listed_tensors_and_reloads(loaded, tmp_path) -> None:
    loaded.config.dtype = torch.bfloat16
    sites = find_sites(loaded.model, find_blocks(loaded.model), loaded.hidden_size,
                       checkpoint_tensor_names=loaded.checkpoint_tensor_names)
    site = next(site for site in sites.sites if site.role == "residual_write_mlp")
    before = tensor_digests(loaded.model)
    delta = Delta("low_rank", U=torch.ones((site.output_size, 1), dtype=torch.float64) * 0.0001,
                  V=torch.ones((1, site.input_size), dtype=torch.float64))
    edited = apply(loaded, {site.tensor_name: delta}, sites=sites, method="test_low_rank", settings={})
    assert {name for name in before if before[name] != edited.after_tensors[name]} == {site.tensor_name}
    reference = save(edited, tmp_path / "export", model_id="test/model", request_digest="sha256:" + "a" * 64)
    assert reference.kind == "weight_edit"
    reloaded = load_model(reference.artifact_path, dtype=torch.float32, device="cpu")
    assert model_state_digest(reloaded.model) == reference.model_digest
    assert tensor_digests(reloaded.model) == edited.after_tensors
    from transformers import AutoModelForCausalLM
    plain = AutoModelForCausalLM.from_pretrained(reference.artifact_path, local_files_only=True, trust_remote_code=False,
                                               dtype=torch.float32).eval()
    assert model_state_digest(plain) == reference.model_digest
    receipt = json.loads((tmp_path / "export" / "training_receipt.json").read_bytes())
    assert receipt["fresh_reload_verified"] is True and receipt["edited_tensors"] == [site.tensor_name]
    assert receipt["explicit_dtype_metadata"] == {"dtype": {"source": "bfloat16", "export": "float32"}}


def test_zero_edit_is_exact_and_invalid_delta_never_mutates(loaded) -> None:
    sites = find_sites(loaded.model, find_blocks(loaded.model), loaded.hidden_size,
                       checkpoint_tensor_names=loaded.checkpoint_tensor_names)
    site = next(site for site in sites.sites if site.role == "residual_write_mlp")
    before = model_state_digest(loaded.model)
    with torch.no_grad():
        logits = loaded.model(torch.tensor([[1, 1]])).logits.clone()
    zero = Delta("dense", D=torch.zeros((site.output_size, site.input_size), dtype=torch.float64))
    edited = apply(loaded, {site.tensor_name: zero}, sites=sites, method="zero", settings={})
    assert edited.receipt["edited_tensors"] == [] and model_state_digest(loaded.model) == before
    with torch.no_grad():
        torch.testing.assert_close(loaded.model(torch.tensor([[1, 1]])).logits, logits, atol=0, rtol=0)
    with pytest.raises(WeightEditInputError, match="finite"):
        Delta("dense", D=torch.full((site.output_size, site.input_size), float("nan")))
    assert model_state_digest(loaded.model) == before


def test_tied_storage_requires_permission_and_records_all_changed_aliases(loaded) -> None:
    if not loaded.shared_storage_groups:
        pytest.skip("fixture has no tied storage")
    group = loaded.shared_storage_groups[0]
    name = next(name for name in group if name in loaded.checkpoint_tensor_names)
    tensor = loaded.model.state_dict()[name]
    site = Site(name, 0, "residual_write_mlp", tensor.shape[1], tensor.shape[0], False, str(tensor.dtype))
    sites = SiteMap((site,), checkpoint_names_verified=True)
    delta = Delta("dense", D=torch.ones(tensor.shape, dtype=torch.float64) * 0.0001)
    before = model_state_digest(loaded.model)
    with pytest.raises(WeightEditInputError, match="shared"):
        apply(loaded, {name: delta}, sites=sites, method="tied", settings={})
    assert model_state_digest(loaded.model) == before
    edited = apply(loaded, {name: delta}, sites=sites, method="tied", settings={}, allow_shared=True)
    assert set(edited.receipt["edited_tensors"]) == set(group)
    assert edited.receipt["shared_storage_groups"] == [list(group)]


def test_failed_receipt_creation_rolls_back_the_edit(loaded) -> None:
    sites = find_sites(loaded.model, find_blocks(loaded.model), loaded.hidden_size,
                       checkpoint_tensor_names=loaded.checkpoint_tensor_names)
    site = next(site for site in sites.sites if site.role == "residual_write_mlp")
    delta = Delta("dense", D=torch.ones((site.output_size, site.input_size), dtype=torch.float64) * 0.0001)
    before = model_state_digest(loaded.model)
    with pytest.raises(ValueError):
        apply(loaded, {site.tensor_name: delta}, sites=sites, method="invalid_receipt", settings={"invalid": object()})
    assert model_state_digest(loaded.model) == before


def test_native_edit_tiles_projection_wider_than_smoke_vector_bound() -> None:
    model = torch.nn.Module()
    model.projection = torch.nn.Linear(4100, 2, bias=False)
    tokenizer = SimpleNamespace(chat_template=None, special_tokens_map={}, get_vocab=lambda: {"token": 1})
    loaded = SimpleNamespace(model=model, base_digest=model_state_digest(model), tokenizer=tokenizer)
    site = Site("projection.weight", 0, "residual_write_mlp", 4100, 2, False, "torch.float32")
    delta = Delta("dense", D=torch.ones((2, 4100), dtype=torch.float64) * 0.0001)
    expected = (model.projection.weight.detach().double() + delta.D).float()
    edited = apply(loaded, {site.tensor_name: delta}, sites=SiteMap((site,), checkpoint_names_verified=True),
                   method="wide_projection", settings={})
    torch.testing.assert_close(model.projection.weight, expected, rtol=0, atol=0)
    chunks = edited.receipt["transactions"][site.tensor_name]["chunks"]
    assert len(chunks) == 2
    assert all(chunk["column_stop"] - chunk["column_start"] <= 4096 for chunk in chunks)
    assert all(chunk["reload_parity_verified"] for chunk in chunks)