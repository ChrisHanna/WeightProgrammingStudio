from __future__ import annotations

import json
import os

import pytest
import torch

from neural_foundry.backends.weight_edit.probe import probe_model


def test_probe_records_creation_identity_but_requires_serving_for_support(tmp_path) -> None:
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    config = transformers.Qwen3Config(vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                     num_attention_heads=2, num_key_value_heads=1, head_dim=12)
    model = transformers.AutoModelForCausalLM.from_config(config).eval()
    base = tmp_path / "base"
    model.save_pretrained(base, safe_serialization=True)
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel({"[UNK]":0, "Return":1, "the":2, "number":3}, unk_token="[UNK]"))
    transformers.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]").save_pretrained(base)
    report = probe_model(base, tmp_path / "edited", dtype=torch.float32, device="cpu", model_id="test/qwen")
    assert report["zero_edit_exact"] and report["reload_verified"] and report["unchanged_tensors_verified"]
    assert report["supported"] is False and report["serving_verified"] is False
    assert report["identity_probe_count"] == 16
    receipt = json.loads((tmp_path / "edited" / "training_receipt.json").read_bytes())
    assert len(receipt["identity_probes"]["records"]) == 16
    assert receipt["identity_probes"]["model_digest"] == report["candidate"]["model_digest"]
    assert json.loads((tmp_path / "edited.probe.json").read_bytes()) == report


@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-0.6B", "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                                     "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1"])
def test_cached_model_complete_compatibility_probe(tmp_path, model_id) -> None:
    if os.environ.get("NF_WEIGHT_EDIT_PROBE_LIVE") != "1":
        pytest.skip("set NF_WEIGHT_EDIT_PROBE_LIVE=1 for cached GPU compatibility probes")
    from neural_foundry.provenance import resolve_checkpoint
    from neural_foundry.vllm_container_adapter import VllmCandidateServer

    revisions = {"HuggingFaceTB/SmolLM2-1.7B-Instruct": "31b70e2e869a7173562077fd711b654946d38674"}
    checkpoint = resolve_checkpoint(model_id, revision=revisions.get(model_id))
    assert isinstance(checkpoint.revision, str) and len(checkpoint.revision) == 40
    report = probe_model(checkpoint.path, tmp_path / "candidate", dtype=torch.float32, device="cuda:0", model_id=model_id,
                         serving_factory=lambda reference: VllmCandidateServer(reference, revision=checkpoint.revision,
                                                                                timeout_seconds=300, precision="float32",
                                                                                gpu_memory_utilization=0.75))
    assert report["supported"] is True and report["serving_verified"] is True
    assert report["serving_evidence"]["fixed_prompts"] == 16
    assert report["serving_evidence"]["max_logprob_absolute_error"] <= 0.05