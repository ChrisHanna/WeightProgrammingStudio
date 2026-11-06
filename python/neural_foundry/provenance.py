"""Where a model and its artifacts came from: checkpoint resolution, evidence grades, signed file manifests."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EVIDENCE_GRADES = frozenset(
    {
        "unverified_model",  # Measured here, but no checkpoint identity was established.
        "fixture_synthetic",  # Built-in toy model: proves the software path, not a model result.
        "measured_local_model",  # Real checkpoint, measured in this process.
        "measured_live_runtime",  # Real checkpoint, measured through an external serving runtime.
        "measured_caller_callable",  # A caller-written function ran; the model behind it is unverified.
        "supplied_precomputed",  # Numbers someone supplied; nothing was measured here.
    }
)
ARTIFACT_MANIFEST_NAME = "artifact_manifest.json"
_CHUNK = 1024 * 1024


class CheckpointUnavailableError(RuntimeError):
    """The requested checkpoint isn't available. Nothing is substituted for it."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True)
class ResolvedCheckpoint:
    path: Path
    model_id: str
    revision: str | None
    source: str
    file_digests: dict[str, str]


def _digest_checkpoint_files(directory: Path) -> dict[str, str]:
    return {
        str(p.relative_to(directory)).replace(os.sep, "/"): sha256_file(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def _hub_cache_root() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    return Path(home) / "hub"


def _find_in_hub_cache(repo_id: str, revision: str | None, cache_dir: Path) -> tuple[Path, str] | None:
    repo_dir = cache_dir / f"models--{repo_id.replace('/', '--')}"
    if not repo_dir.is_dir():
        return None
    resolved = revision
    ref = repo_dir / "refs" / (revision or "main")
    if ref.is_file():
        resolved = ref.read_text(encoding="utf-8").strip()
    if resolved is None:
        return None
    snapshot = repo_dir / "snapshots" / resolved
    return (snapshot, resolved) if (snapshot / "config.json").is_file() else None


def resolve_checkpoint(
    model: str | Path,
    *,
    revision: str | None = None,
    allow_download: bool = False,
    cache_dir: str | Path | None = None,
) -> ResolvedCheckpoint:
    """Find a real checkpoint from a local directory, the Hugging Face cache, or (if allowed) the Hub.

    Downloads are opt-in, so resolving a name never fetches gigabytes or contacts a network by surprise.
    """
    key = str(model)
    local = Path(key)
    if local.is_dir() and (local / "config.json").is_file():
        return ResolvedCheckpoint(local.resolve(), local.name, revision, "local_directory", _digest_checkpoint_files(local))
    root = Path(cache_dir) if cache_dir is not None else _hub_cache_root()
    cached = _find_in_hub_cache(key, revision, root)
    if cached is not None:
        return ResolvedCheckpoint(cached[0], key, cached[1], "huggingface_cache", _digest_checkpoint_files(cached[0]))
    if allow_download:
        try:
            from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
        except ImportError as exc:
            raise CheckpointUnavailableError("Downloading requires the huggingface_hub package") from exc
        path = Path(snapshot_download(repo_id=key, revision=revision, cache_dir=str(root)))
        return ResolvedCheckpoint(path, key, path.name, "huggingface_hub_download", _digest_checkpoint_files(path))
    raise CheckpointUnavailableError(
        f"Checkpoint {key!r} is not a local directory or in the Hugging Face cache at {root}. "
        "Pass a local path, or allow a download with allow_download=True."
    )


def evidence_grade_for_model(model: Any) -> str:
    name = str(getattr(model, "model_name", "")).lower()
    # A real-model grade requires a resolved checkpoint with hashed files. A name proves nothing.
    if getattr(model, "checkpoint_identity", None) is not None:
        return "measured_local_model"
    return "fixture_synthetic" if "mini" in name or "fixture" in name or name.startswith("nf/") else "unverified_model"


def write_artifact_manifest(
    directory: str | Path,
    *,
    signing_key: str,
    signer_id: str,
    open_patch_digest: str,
    files: Sequence[str] | None = None,
) -> Path:
    """Sign the digest of each exported file, binding the weights, config and tokenizer to the patch."""
    from .patch_ux import sign_patch_envelope
    from .records import canonicalize

    root = Path(directory).resolve()
    names = (
        list(files)
        if files is not None
        else [
            str(p.relative_to(root)).replace(os.sep, "/")
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != ARTIFACT_MANIFEST_NAME
        ]
    )
    body = {
        "schema_version": 1,
        "kind": "nf_artifact_manifest",
        "open_patch_digest": open_patch_digest,
        "files": {name: sha256_file(root / name) for name in names},
    }
    path = root / ARTIFACT_MANIFEST_NAME
    path.write_bytes(canonicalize(sign_patch_envelope(body, signing_key=signing_key, signer_id=signer_id)))
    return path


def verify_artifact_manifest(directory: str | Path, *, public_key: str) -> dict[str, Any]:
    """Check the manifest's signature and every listed file; raises `ValueError` on any mismatch."""
    from .patch_ux import verify_patch_signature

    root = Path(directory).resolve()
    envelope: Mapping[str, Any] = json.loads((root / ARTIFACT_MANIFEST_NAME).read_bytes())
    verify_patch_signature(envelope, public_key=public_key)
    body = envelope["patch"]
    if body.get("kind") != "nf_artifact_manifest" or not body.get("files"):
        raise ValueError("Artifact manifest is malformed")
    present = {str(p.relative_to(root)).replace(os.sep, "/") for p in root.rglob("*") if p.is_file()} - {ARTIFACT_MANIFEST_NAME}
    unlisted = sorted(present - set(body["files"]))
    if unlisted:
        raise ValueError(f"Artifact directory has unlisted files: {unlisted}")
    for name, expected in body["files"].items():
        target = (root / name).resolve()
        if root not in target.parents:
            raise ValueError(f"Artifact manifest path escapes its directory: {name!r}")
        if not target.is_file():
            raise ValueError(f"Artifact missing: {name}")
        if sha256_file(target) != expected:
            raise ValueError(f"Artifact digest mismatch: {name}")
    for patch_name in ("open_patch_v1.json", "weight_patch.nf.json"):
        if (root / patch_name).is_file():
            enclosed = json.loads((root / patch_name).read_bytes())
            if enclosed.get("patch_digest") != body["open_patch_digest"]:
                raise ValueError(f"{patch_name} is not the patch this manifest was signed for")
    return {"verified": True, "open_patch_digest": body["open_patch_digest"], "files_verified": len(body["files"])}
