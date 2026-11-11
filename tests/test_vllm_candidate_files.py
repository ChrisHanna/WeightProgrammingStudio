from __future__ import annotations

import json
from dataclasses import replace

import pytest

from neural_foundry.gym_artifacts import CandidateRef, artifact_files, write_candidate_manifest
from neural_foundry.gym_job import GymJobError
from neural_foundry.gym_service import _document_digest
from neural_foundry.provenance import resolve_checkpoint
from neural_foundry.vllm_candidate_files import verify_vllm_candidate_files


def _checkpoint(path):
    path.mkdir()
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        (path / name).write_text(name, encoding="utf-8")
    return path


def test_base_candidate_file_digest_is_verified(tmp_path) -> None:
    root = _checkpoint(tmp_path / "base")
    revision = "a" * 40
    files = resolve_checkpoint(root).file_digests
    digest = _document_digest({"model_id": "test/model", "revision": revision, "files": files})
    reference = CandidateRef("test/model", "full", digest, digest, digest, str(root), "base", "unchanged_base", "qwen3")
    assert verify_vllm_candidate_files(reference, revision=revision) == files
    with pytest.raises(GymJobError, match="immutable"):
        verify_vllm_candidate_files(reference, revision="main")
    (root / "model.safetensors").write_text("changed", encoding="utf-8")
    with pytest.raises(GymJobError, match="differ"):
        verify_vllm_candidate_files(reference, revision=revision)


@pytest.mark.parametrize("method", ["sft", "lora"])
def test_full_sft_candidate_is_bound_to_receipt_and_files(tmp_path, method) -> None:
    root = _checkpoint(tmp_path / "candidate")
    state, base, request = ("sha256:" + char * 64 for char in "123")
    receipt = {"training_request_digest": request, "model_digest": state}
    if method == "lora":
        receipt.update({"export_kind": "peft_merged_full_v1", "reload_state_digest": state,
                        "source_adapter_digest": base, "source_candidate_model_digest": state})
    (root / "training_receipt.json").write_text(json.dumps(receipt))
    files = artifact_files(root)
    artifact = _document_digest({"training_request": request, "files": files})
    reference = CandidateRef("test/model", "full", base, state, artifact, str(root), method, method, "qwen3")
    write_candidate_manifest(root, model_id=reference.model_id, kind=reference.kind, base_digest=base,
                             model_digest=state, request_digest=request,
                             metadata={"candidate": {"candidate_id": method, "method_id": method, "base_model_digest": base,
                                                     "candidate_model_digest": state, "artifact_digest": artifact}})
    assert verify_vllm_candidate_files(reference, revision="a" * 40) == files
    with pytest.raises(GymJobError, match="differ"):
        verify_vllm_candidate_files(replace(reference, model_digest=base), revision="a" * 40)
    with pytest.raises(GymJobError, match="lineage"):
        verify_vllm_candidate_files(replace(reference, base_digest=state), revision="a" * 40)
    (root / "config.json").write_text("changed")
    with pytest.raises(GymJobError, match="differ"):
        verify_vllm_candidate_files(reference, revision="a" * 40)