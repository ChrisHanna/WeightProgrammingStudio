from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from neural_foundry.backends.weight_edit.probe import probe_model, record_portability_probes


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
    assert report["identity_probe_count"] == 0 and report["portability_probe_count"] == 16
    receipt = json.loads((tmp_path / "edited" / "training_receipt.json").read_bytes())
    assert "identity_probes" not in receipt
    assert len(report["transformers_portability_record"]["records"]) == 16
    assert json.loads((tmp_path / "edited.probe.json").read_bytes()) == report

    from neural_foundry.backends.weight_edit.loader import load_model
    from neural_foundry.backends.weight_edit.serving_probes import verify_identity
    from neural_foundry.gym_artifacts import artifact_files

    class FakeServingRuntime:
        revision = "a" * 40

        def __init__(self, reference):
            self.reference = reference

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def record_identity_probes(self):
            loaded = load_model(self.reference.artifact_path, dtype=torch.float32, device="cpu")
            record = record_portability_probes(loaded)
            return {**record, "kind": "nf_vllm_weight_identity_v2", "model_digest": self.reference.model_digest,
                    "image": "sha256:" + "3" * 64, "context_tokens": 512,
                    "weight_files": {name: digest for name, digest in artifact_files(Path(self.reference.artifact_path)).items()
                                     if name.endswith(".safetensors")}}

        def loaded_identity(self):
            receipt = json.loads((Path(self.reference.artifact_path) / "training_receipt.json").read_bytes())
            evidence = verify_identity(receipt["identity_probes"], self.record_identity_probes())
            return {"served_model_identity_verified": True, "identity_probe_evidence": evidence}

    qualified = probe_model(base, tmp_path / "mock-qualified", dtype=torch.float32, device="cpu", model_id="test/qwen",
                            serving_factory=FakeServingRuntime)
    assert qualified["supported"] is True and qualified["identity_probe_count"] == 16
    assert qualified["portability_calibration"]["measured_base_max_error"] == 0
    assert qualified["portability_verification"]["verified"] is True
    sealed = json.loads((tmp_path / "mock-qualified" / "training_receipt.json").read_bytes())
    assert sealed["identity_probes"]["kind"] == "nf_vllm_weight_identity_v2"


@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-0.6B", "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                                     "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1"])
def test_cached_model_complete_compatibility_probe(tmp_path, model_id) -> None:
    if os.environ.get("NF_WEIGHT_EDIT_PROBE_LIVE") != "1":
        pytest.skip("set NF_WEIGHT_EDIT_PROBE_LIVE=1 for cached GPU compatibility probes")
    from neural_foundry.provenance import resolve_checkpoint
    from neural_foundry.vllm_container_adapter import VllmCandidateServer

    revisions = {"HuggingFaceTB/SmolLM2-1.7B-Instruct": "31b70e2e869a7173562077fd711b654946d38674"}
    memory_fractions = {"Qwen/Qwen3-0.6B": 0.25, "HuggingFaceTB/SmolLM2-1.7B-Instruct": 0.35,
                        "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1": 0.5}
    precision = os.environ.get("NF_WEIGHT_EDIT_PROBE_PRECISION", "bfloat16")
    assert precision in {"bfloat16", "float32"}
    checkpoint = resolve_checkpoint(model_id, revision=revisions.get(model_id))
    assert isinstance(checkpoint.revision, str) and len(checkpoint.revision) == 40
    report = probe_model(checkpoint.path, tmp_path / "candidate", dtype=getattr(torch, precision), device="cuda:0", model_id=model_id,
                         serving_factory=lambda reference: VllmCandidateServer(reference, revision=checkpoint.revision,
                                                                                timeout_seconds=300, precision=precision,
                                                                                gpu_memory_utilization=memory_fractions[model_id]))
    assert report["supported"] is True, report.get("reasons")
    assert report["serving_verified"] is True
    assert report["serving_evidence"]["fixed_prompts"] == 16
    assert report["serving_evidence"]["max_logprob_absolute_error"] <= 1e-6
    assert report["portability_verification"]["verified"] is True


@pytest.mark.parametrize("filename,model_id,precision", [
    ("weight-edit-qwen3-bf16.json", "Qwen/Qwen3-0.6B", "bfloat16"),
    ("weight-edit-smollm2-float32.json", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "float32"),
])
def test_retained_probe_evidence_is_digest_bound(filename, model_id, precision) -> None:
    from neural_foundry.gym_service import _document_digest

    path = Path(__file__).resolve().parents[1] / "docs" / "evidence" / filename
    report = json.loads(path.read_bytes())
    assert report["probe_digest"] == _document_digest({key: value for key, value in report.items() if key != "probe_digest"})
    assert report["model_id"] == model_id and report["precision"] == precision
    assert report["supported"] and report["reload_verified"] and report["unchanged_tensors_verified"]
    assert report["serving_evidence"]["verified"] is True
    assert report["serving_evidence"]["max_logprob_absolute_error"] <= 1e-6
    portability = report["portability_verification"]
    calibration = report["portability_calibration"]
    assert portability["verified"] is True and portability["max_logprob_absolute_error"] <= calibration["logprob_atol"]


def test_failed_model_gate_is_recorded_without_scores(tmp_path, monkeypatch) -> None:
    from neural_foundry.backends.weight_edit import probe
    from neural_foundry.backends.weight_edit.loader import UnsupportedModel

    def fail(*args, **kwargs):
        raise UnsupportedModel("fixed probe greedy tokens differ")

    monkeypatch.setattr(probe, "_probe_model", fail)
    report = probe_model("fixture", tmp_path / "failed", dtype=torch.bfloat16, device="cpu", model_id="fixture")
    assert report["supported"] is False and report["measurements_complete"] is False
    assert "serving_verified" not in report and "candidate" not in report
    assert json.loads((tmp_path / "failed.probe.json").read_bytes()) == report


def test_retained_nemotron_failure_is_not_supported() -> None:
    from neural_foundry.gym_service import _document_digest

    path = Path(__file__).resolve().parents[1] / "docs" / "evidence" / "weight-edit-nemotron-bf16.json"
    report = json.loads(path.read_bytes())
    assert report["probe_digest"] == _document_digest({key: value for key, value in report.items() if key != "probe_digest"})
    assert report["supported"] is False and report["qualification_status"] == "failed_gate"
    assert report["reasons"] == ["candidate exceeds the frozen unedited-base portability limit"]