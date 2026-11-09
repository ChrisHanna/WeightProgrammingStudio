"""Verify the file identities of full candidates before vLLM serving."""

from __future__ import annotations

import json
from pathlib import Path

from .gym_artifacts import CandidateRef, artifact_files, read_candidate_manifest
from .gym_job import GymJobError
from .gym_service import _document_digest
from .provenance import resolve_checkpoint


def verify_vllm_candidate_files(reference: CandidateRef, *, revision: str) -> dict[str, str]:
    if reference.kind not in {"full", "weight_edit"} or not reference.candidate_id or not reference.method_id or not reference.family:
        raise GymJobError("vLLM requires a fully identified full-model candidate")
    if not isinstance(revision, str) or len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise GymJobError("vLLM requires an immutable checkpoint revision")
    directory = Path(reference.artifact_path)
    if reference.method_id == "unchanged_base":
        checkpoint = resolve_checkpoint(directory)
        files = checkpoint.file_digests
        digest = _document_digest({"model_id": reference.model_id, "revision": revision, "files": files})
        if reference.base_digest != digest or reference.model_digest != digest or reference.artifact_digest != digest:
            raise GymJobError("vLLM base checkpoint files differ from the candidate reference")
    elif reference.method_id in {"sft", "weight_edit"}:
        files = artifact_files(directory)
        try:
            receipt = json.loads((directory / "training_receipt.json").read_bytes())
            request_digest = receipt["training_request_digest"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise GymJobError("vLLM candidate training receipt is unreadable") from exc
        if (not isinstance(request_digest, str) or receipt.get("model_digest") != reference.model_digest
            or _document_digest({"training_request": request_digest, "files": files}) != reference.artifact_digest):
            raise GymJobError("vLLM trained artifact files differ from the candidate reference")
        manifest = read_candidate_manifest(directory, request_digest)
        candidate = manifest["metadata"].get("candidate")
        if (reference.base_digest is None or manifest["model_id"] != reference.model_id
            or manifest["kind"] != reference.kind or manifest["base_digest"] != reference.base_digest
            or manifest["model_digest"] != reference.model_digest
            or candidate != {"candidate_id": reference.candidate_id, "method_id": reference.method_id,
                             "base_model_digest": reference.base_digest, "candidate_model_digest": reference.model_digest,
                             "artifact_digest": reference.artifact_digest}):
            raise GymJobError("vLLM published artifact lineage differs from the candidate reference")
    else:
        raise GymJobError("vLLM minimal serving supports only base and full SFT candidates")
    if not {"config.json", "tokenizer.json"} <= files.keys() or not any(name.endswith(".safetensors") for name in files):
        raise GymJobError("vLLM full candidate is missing model or tokenizer files")
    return files