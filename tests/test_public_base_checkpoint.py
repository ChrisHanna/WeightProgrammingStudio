"""Untrained checkpoints use the generic serving path without synthetic training receipts."""

import json
from dataclasses import asdict

import pytest
from neural_foundry import proof_plugins, provenance, vllm_container_adapter
from neural_foundry.gym_artifacts import CandidateRef
from neural_foundry.gym_job import GymJobError
from neural_foundry.gym_service import _document_digest
from neural_foundry.proof_pipeline import write_document
from neural_foundry.provenance import ResolvedCheckpoint, sha256_file


def test_unchanged_checkpoint_export_requires_no_training_or_model_load(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "fixture_causal"}))
    (checkpoint / "tokenizer.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"fixture weights, never loaded")
    files = {path.name: sha256_file(path) for path in checkpoint.iterdir()}
    revision = "a" * 40
    resolved = ResolvedCheckpoint(checkpoint, "fixture/base", revision, "local_directory", files)
    original_resolve = provenance.resolve_checkpoint

    def resolve(model, **kwargs):
        if model != "fixture/base":
            return original_resolve(model, **kwargs)
        assert model == "fixture/base" and kwargs == {"revision": revision, "allow_download": False}
        return resolved

    monkeypatch.setattr(provenance, "resolve_checkpoint", resolve)
    monkeypatch.setattr(proof_plugins, "trained_reference",
                        lambda *_: pytest.fail("base export must not invent a training result"))
    spec = {"base_model": {"model_id": "fixture/base", "revision": revision},
            "candidates": [{"role": "incumbent", "method": "unchanged_base"}]}
    output = proof_plugins.serving_export(spec=spec, role="incumbent", root=tmp_path / "run")
    digest = _document_digest({"model_id": "fixture/base", "revision": revision, "files": files})
    assert output["reference"]["model_digest"] == digest
    assert output["reference"]["family"] == "fixture_causal"
    assert output["source_kind"] == "unchanged_checkpoint" and output["source_result_digest"] is None
    assert output["files_verified"] and not output["reloaded"] and not output["model_loading_performed"]
    assert not (tmp_path / "run" / "incumbent" / "training-result.json").exists()


def test_base_export_rejects_file_drift_before_publishing(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = checkpoint / "config.json"
    config.write_text('{"model_type":"fixture_causal"}')
    stale = {"config.json": sha256_file(config)}
    config.write_text('{"model_type":"different"}')
    original_resolve = provenance.resolve_checkpoint
    monkeypatch.setattr(provenance, "resolve_checkpoint", lambda model, **kwargs: (
        ResolvedCheckpoint(checkpoint, "fixture/base", "a" * 40, "local_directory", stale)
        if model == "fixture/base" else original_resolve(model, **kwargs)))
    spec = {"base_model": {"model_id": "fixture/base", "revision": "a" * 40},
            "candidates": [{"role": "base", "method": "unchanged_base"}]}
    with pytest.raises(GymJobError, match="differ"):
        proof_plugins.serving_export(spec=spec, role="base", root=tmp_path / "run")
    assert not (tmp_path / "run" / "export-base.json").exists()


@pytest.mark.parametrize("fraction", [0.25, 0.5])
def test_serving_memory_fraction_is_candidate_configuration(tmp_path, monkeypatch, fraction) -> None:
    digest = _document_digest({"fixture": 1})
    reference = CandidateRef("fixture/base", "full", digest, digest, digest,
                             str(tmp_path / "weights"), "base", "unchanged_base", "fixture_causal")
    candidate = {"role": "base", "method": "unchanged_base",
                 "execution": {"kind": "server", "configuration": {"gpu_memory_utilization": fraction}}}
    spec = {"candidates": [candidate], "policy": {"context_tokens": 512, "max_new_tokens": 128}}
    write_document(tmp_path / "export-base.json", {"reference": asdict(reference), "revision": "a" * 40,
                                                  "run_spec_digest": _document_digest(spec)})
    calls = []

    class Server:
        def __init__(self, _reference, **kwargs):
            calls.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def record_identity_probes(self):
            return {"fixed": "fixture probes"}

    monkeypatch.setattr(vllm_container_adapter, "VllmCandidateServer", Server)
    proof_plugins.vllm_server(spec=spec, candidate=candidate, root=tmp_path, freeze_identity=True)
    proof_plugins.vllm_server(spec=spec, candidate=candidate, root=tmp_path)
    assert all(call["gpu_memory_utilization"] == fraction for call in calls)
    assert calls[-1]["max_new_tokens"] == 128
