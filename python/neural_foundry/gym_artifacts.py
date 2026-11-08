"""Shared verification and publication of lifecycle artifact manifests."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .gym_job import GymJobError
from .gym_service import _document_digest, _wire_document
from .provenance import sha256_file

ARTIFACT_KINDS = frozenset({"lora", "qlora", "full", "weight_edit", "own_model"})


def _sha256_digest(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 71 and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


@dataclass(frozen=True)
class CandidateRef:
    model_id: str
    kind: str
    base_digest: str | None
    model_digest: str
    artifact_digest: str
    artifact_path: str
    candidate_id: str = ""
    method_id: str = ""
    family: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise GymJobError("candidate reference requires a model ID")
        if not isinstance(self.kind, str) or self.kind not in ARTIFACT_KINDS:
            raise GymJobError("unsupported candidate artifact kind")
        if self.base_digest is not None and not _sha256_digest(self.base_digest):
            raise GymJobError("candidate base digest must be a sha256 digest")
        if self.kind in {"lora", "qlora", "weight_edit"} and self.base_digest is None:
            raise GymJobError("candidate artifact kind requires a pinned base digest")
        if not _sha256_digest(self.model_digest) or not _sha256_digest(self.artifact_digest):
            raise GymJobError("candidate model and artifact digests must be sha256 digests")
        if not isinstance(self.artifact_path, str) or not self.artifact_path.strip():
            raise GymJobError("candidate reference requires an artifact path")


def artifact_files(directory: Path) -> dict[str, str]:
    if directory.is_symlink() or not directory.is_dir():
        raise GymJobError("training artifact directory must be a real directory")
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise GymJobError("training artifact must not contain symlinks")
        relative = path.relative_to(directory).as_posix()
        if path.is_file() and relative != "artifact_manifest.json":
            files[relative] = sha256_file(path)
    return files


def read_artifact_manifest(directory: Path, request_digest: str) -> dict[str, Any]:
    path = directory / "artifact_manifest.json"
    try:
        manifest = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GymJobError(f"training artifact manifest is missing or unreadable: {exc}") from exc
    if not isinstance(manifest, dict):
        raise GymJobError("training artifact manifest must be an object")
    if isinstance(manifest.get("kind"), str) and manifest["kind"] in ARTIFACT_KINDS:
        candidate_manifest = read_candidate_manifest(directory, request_digest)
        metadata = candidate_manifest["metadata"]
        candidate = metadata.get("candidate")
        if (
            set(metadata) != {"candidate", "model_adapter", "loaded_candidate"}
            or not isinstance(candidate, dict)
            or candidate.get("base_model_digest") != candidate_manifest["base_digest"]
            or candidate.get("candidate_model_digest") != candidate_manifest["model_digest"]
            or not isinstance(metadata.get("model_adapter"), dict)
            or not isinstance(metadata.get("loaded_candidate"), dict)
        ):
            raise GymJobError("candidate artifact metadata differs from its manifest identity")
        return {**candidate_manifest, "candidate": candidate,
                "model_adapter": metadata["model_adapter"], "loaded_candidate": metadata["loaded_candidate"]}
    digest = manifest.pop("manifest_digest", None)
    if (
        manifest.get("kind") != "nf_gym_training_artifact_manifest_v1"
        or manifest.get("training_request_digest") != request_digest
        or not isinstance(digest, str)
        or _document_digest(manifest) != digest
    ):
        raise GymJobError("training artifact manifest does not match the frozen request")
    if artifact_files(directory) != manifest.get("files"):
        raise GymJobError("training artifact file set or digest differs from its manifest")
    manifest["manifest_digest"] = digest
    return manifest


def read_candidate_manifest(directory: Path, request_digest: str) -> dict[str, Any]:
    path = directory / "artifact_manifest.json"
    try:
        manifest = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GymJobError(f"candidate artifact manifest is missing or unreadable: {exc}") from exc
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version", "kind", "model_id", "base_digest", "model_digest",
        "training_request_digest", "files", "metadata", "manifest_digest",
    }:
        raise GymJobError("candidate artifact manifest has invalid fields")
    digest = manifest["manifest_digest"]
    if (
        manifest["schema_version"] != 2 or not isinstance(manifest["metadata"], dict)
        or not _sha256_digest(request_digest)
        or manifest["training_request_digest"] != request_digest or not _sha256_digest(digest)
        or _document_digest({key: value for key, value in manifest.items() if key != "manifest_digest"}) != digest
    ):
        raise GymJobError("candidate artifact manifest does not match the frozen request")
    CandidateRef(
        model_id=manifest["model_id"], kind=manifest["kind"], base_digest=manifest["base_digest"],
        model_digest=manifest["model_digest"], artifact_digest=digest, artifact_path=str(directory),
    )
    files = artifact_files(directory)
    if not files or files != manifest["files"]:
        raise GymJobError("candidate artifact file set or digest differs from its manifest")
    return manifest


def write_candidate_manifest(
    directory: Path, *, model_id: str, kind: str, base_digest: str | None,
    model_digest: str, request_digest: str, metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not _sha256_digest(request_digest):
        raise GymJobError("training request digest must be a sha256 digest")
    CandidateRef(model_id, kind, base_digest, model_digest, request_digest, str(directory))
    files = artifact_files(directory)
    if not files:
        raise GymJobError("candidate artifact must contain files")
    return write_artifact_manifest(directory, {
        "schema_version": 2, "kind": kind, "model_id": model_id, "base_digest": base_digest,
        "model_digest": model_digest, "training_request_digest": request_digest, "files": files,
        "metadata": dict(metadata or {}),
    })


def write_artifact_manifest(directory: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(manifest)
    record["manifest_digest"] = _document_digest(record)
    raw = _wire_document(record)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".nf-artifact-manifest-", dir=directory)
    temporary = Path(temporary_name)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / "artifact_manifest.json")
        if os.name == "nt":
            os.chmod(directory / "artifact_manifest.json", 0o600)
    finally:
        temporary.unlink(missing_ok=True)
    return record