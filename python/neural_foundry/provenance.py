"""Where a model and its artifacts came from: checkpoint resolution, evidence grades, signed file manifests."""

from __future__ import annotations

import hashlib
import json
import os
import re
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


MAX_SIGNED_DOCUMENT_BYTES = 1024 * 1024  # The native canonical parser's limit, shared with `nf verify`.
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_MANIFEST_KEYS = frozenset({"schema_version", "kind", "open_patch_digest", "files"})


def load_signed_document(path: str | Path) -> Any:
    """Parse a signed JSON document with the same bounded Rust parser as `nf verify`; raises `ValueError`.

    Python's own parser would accept duplicate keys, BOMs, UTF-16 and NaN, which `nf verify` refuses.
    """
    from . import _native
    from .records import negotiate

    target = Path(path)
    try:
        if target.stat().st_size > MAX_SIGNED_DOCUMENT_BYTES:
            raise ValueError(f"{target.name} exceeds the maximum signed document size")
        negotiate()
        return json.loads(_native.canonical_json(target.read_bytes()))
    except (OSError, _native.NativeError) as exc:
        raise ValueError(f"{target.name} is not a valid signed document: {exc}") from exc


def parse_strict_json(raw: bytes) -> Any:
    """Parse UTF-8 JSON for files larger than a signed document; raises `ValueError` on ambiguous input."""

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"Duplicate JSON key {key!r}")
            out[key] = value
        return out

    def no_constant(name: str) -> None:
        raise ValueError(f"JSON constant {name} is not allowed")

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=no_duplicates, parse_constant=no_constant)
    except RecursionError as exc:
        raise ValueError("JSON is nested too deeply") from exc


def _require_plain_path(name: Any) -> None:
    parts = name.split("/") if isinstance(name, str) else []
    if not parts or "\\" in name or ":" in name or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"Artifact manifest path escapes its directory or isn't normalized: {name!r}")


def _artifact_entries(root: Path) -> set[str]:
    """Every file under `root`, without following links. Any symlink, even a dangling one, is refused."""
    found: set[str] = set()
    for folder, directories, names in os.walk(root, followlinks=False):
        for name in (*directories, *names):
            entry = Path(folder) / name
            relative = str(entry.relative_to(root)).replace(os.sep, "/")
            if entry.is_symlink():
                raise ValueError(f"Artifact directory contains a symlink: {relative}")
            if name in names:
                found.add(relative)
    return found


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
    if revision is not None and (Path(revision).is_absolute() or ".." in revision.replace("\\", "/").split("/")):
        raise CheckpointUnavailableError(f"Revision {revision!r} is not a valid revision name")
    resolved = revision
    ref = repo_dir / "refs" / (revision or "main")
    if ref.is_file():
        resolved = ref.read_text(encoding="utf-8").strip()
    if resolved is None:
        return None
    if "/" in resolved or "\\" in resolved or resolved in ("", ".", ".."):
        raise CheckpointUnavailableError(f"Revision {resolved!r} doesn't name a snapshot")
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


def inference_identity_digest(model: Any) -> str:
    tokenizer = getattr(model, "tokenizer", None)
    state = {"config": getattr(model, "config", None), "tokenizer": None if tokenizer is None else {
        "type": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "vocab": getattr(tokenizer, "vocab", None),
        "merges": [" ".join(pair) for pair, _ in sorted(getattr(tokenizer, "bpe_ranks", {}).items(), key=lambda row: row[1])],
        "byte_level": getattr(tokenizer, "byte_level", None), "added_tokens": getattr(tokenizer, "added_tokens", None),
    }}
    return "sha256:" + hashlib.sha256(json.dumps(state, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


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
    if not isinstance(open_patch_digest, str) or not _DIGEST.match(open_patch_digest):
        raise ValueError("open_patch_digest must be a sha256 digest")
    present = _artifact_entries(root)
    names = list(files) if files is not None else sorted(n for n in present if Path(n).name != ARTIFACT_MANIFEST_NAME)
    for name in names:
        _require_plain_path(name)
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
    present = _artifact_entries(root)
    if ARTIFACT_MANIFEST_NAME not in present:
        raise ValueError(f"Artifact manifest not found in {root}")
    envelope = load_signed_document(root / ARTIFACT_MANIFEST_NAME)
    if not isinstance(envelope, Mapping):
        raise ValueError("Artifact manifest is malformed")
    verify_patch_signature(envelope, public_key=public_key)
    body = envelope["patch"]
    files = body.get("files")
    digest = body.get("open_patch_digest")
    if (
        set(body) != _MANIFEST_KEYS
        or body["kind"] != "nf_artifact_manifest"
        or type(body["schema_version"]) is not int
        or body["schema_version"] != 1
        or not isinstance(digest, str)
        or not _DIGEST.match(digest)
        or not isinstance(files, Mapping)
        or not files
    ):
        raise ValueError("Artifact manifest is malformed")
    for name in files:
        _require_plain_path(name)
    unlisted = sorted(present - set(files) - {ARTIFACT_MANIFEST_NAME})
    if unlisted:
        raise ValueError(f"Artifact directory has unlisted files: {unlisted}")
    for name, expected in files.items():
        if name not in present:
            raise ValueError(f"Artifact missing: {name}")
        if sha256_file(root / name) != expected:
            raise ValueError(f"Artifact digest mismatch: {name}")
    for patch_name in ("open_patch_v1.json", "weight_patch.nf.json"):
        if patch_name in present:
            enclosed = load_signed_document(root / patch_name)
            if not isinstance(enclosed, Mapping) or enclosed.get("patch_digest") != digest:
                raise ValueError(f"{patch_name} is not the patch this manifest was signed for")
    return {"verified": True, "open_patch_digest": digest, "files_verified": len(files)}
