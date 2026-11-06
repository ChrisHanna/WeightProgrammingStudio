"""Milestone 1 condition 7: a signed run record that can't leave a field unknown.

Every required field must hold a real value when the record is built and again when it is verified.
Missing fields, extra fields and placeholder values such as "unknown" all raise `RunRecordError`.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .patch_ux import sign_patch_envelope, verify_patch_signature

RUN_RECORD_KIND = "nf_run_record_v1"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_PLACEHOLDERS = {"", "unknown", "none", "null", "n/a", "na", "unset", "todo", "tbd", "?", "main", "head", "latest"}

_REQUIRED_FIELDS: dict[str, Any] = {
    "commit": {"sha": str, "dirty": bool},
    "checkpoint": {"model_id": str, "revision": str, "digest": str, "files": list},
    "dependency_lock": {"lock_files": list, "environment_digest": str, "installed_packages": list},
    "runtime": {"package_version": str, "python": str, "torch": str, "native_extension_digest": str, "platform": str},
    "device": str,
    "precision": str,
    "sampler": str,
    "seeds": dict,
    "edit_graph_digest": str,
    "command": list,
    "artifact_manifest": list,
}


class RunRecordError(ValueError):
    """A run record is incomplete, has a placeholder value, or doesn't match what it describes."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _check_no_placeholder(value: Any, where: str) -> None:
    if value is None:
        raise RunRecordError(f"{where} is null")
    if isinstance(value, float):
        raise RunRecordError(f"{where} is a float; run records hold exact values only")
    if isinstance(value, str):
        if value.strip().lower() in _PLACEHOLDERS:
            raise RunRecordError(f"{where} has placeholder value {value!r}")
    elif isinstance(value, Mapping):
        if not value:
            raise RunRecordError(f"{where} is empty")
        for key, item in value.items():
            _check_no_placeholder(item, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        if not value:
            raise RunRecordError(f"{where} is empty")
        for index, item in enumerate(value):
            _check_no_placeholder(item, f"{where}[{index}]")


def _check_shape(value: Any, shape: Any, where: str) -> None:
    if isinstance(shape, dict):
        if not isinstance(value, Mapping):
            raise RunRecordError(f"{where} must be a mapping")
        if set(value) != set(shape):
            raise RunRecordError(f"{where} missing {sorted(set(shape) - set(value))}, unexpected {sorted(set(value) - set(shape))}")
        for key, inner in shape.items():
            _check_shape(value[key], inner, f"{where}.{key}")
    elif not isinstance(value, shape):
        raise RunRecordError(f"{where} must be {shape.__name__}, got {type(value).__name__}")


def _require_digest(value: str, where: str) -> None:
    if not _DIGEST.match(value):
        raise RunRecordError(f"{where} is not a sha256 digest: {value!r}")


def validate_run_record(record: Mapping[str, Any]) -> None:
    """Raise unless every required field is present, typed, specific and free of placeholders."""
    if record.get("schema_version") != 1 or record.get("kind") != RUN_RECORD_KIND:
        raise RunRecordError(f"Not a {RUN_RECORD_KIND} document")
    body = {k: v for k, v in record.items() if k not in ("schema_version", "kind")}
    _check_shape(body, _REQUIRED_FIELDS, "record")
    _check_no_placeholder(body, "record")
    if not _COMMIT.match(record["commit"]["sha"]):
        raise RunRecordError("commit.sha must be a full 40-character commit")
    if not _COMMIT.match(record["checkpoint"]["revision"]):
        raise RunRecordError("checkpoint.revision must be a full commit, since a branch name can move")
    _require_digest(record["checkpoint"]["digest"], "checkpoint.digest")
    _require_digest(record["edit_graph_digest"], "edit_graph_digest")
    _require_digest(record["dependency_lock"]["environment_digest"], "dependency_lock.environment_digest")
    _require_digest(record["runtime"]["native_extension_digest"], "runtime.native_extension_digest")
    for group, where in (
        (record["checkpoint"]["files"], "checkpoint.files"),
        (record["dependency_lock"]["lock_files"], "dependency_lock.lock_files"),
        (record["artifact_manifest"], "artifact_manifest"),
    ):
        for index, entry in enumerate(group):
            _check_shape(entry, {"path": str, "digest": str, "bytes": int}, f"{where}[{index}]")
            _require_digest(entry["digest"], f"{where}[{index}].digest")
    for name, seed in record["seeds"].items():
        values = seed if isinstance(seed, list) else [seed]
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            raise RunRecordError(f"seeds.{name} must be an integer or a list of integers")


def file_entry(path: str | Path, *, relative_to: str | Path | None = None) -> dict[str, Any]:
    """Describe one existing file. A missing file raises, so it can't enter a manifest."""
    resolved = Path(path)
    if not resolved.is_file():
        raise RunRecordError(f"File does not exist: {resolved}")
    name = resolved.relative_to(relative_to).as_posix() if relative_to is not None else resolved.name
    return {"path": name, "digest": sha256_file(resolved), "bytes": resolved.stat().st_size}


def _git_environment(repo: Path) -> dict[str, str]:
    """Translate a linked worktree's Windows `gitdir:` pointer when running under WSL."""
    env = dict(os.environ)
    dot_git = repo / ".git"
    if dot_git.is_file():
        match = re.match(r"gitdir:\s*([A-Za-z]):[/\\](.*)", dot_git.read_text(encoding="utf-8").strip())
        if match and platform.system() == "Linux":
            env["GIT_DIR"] = f"/mnt/{match.group(1).lower()}/{match.group(2).replace(chr(92), '/')}"
            env["GIT_WORK_TREE"] = str(repo)
    return env


def capture_commit(repo: str | Path) -> dict[str, Any]:
    """The commit and whether the work tree differs from it. Raises if git can't say."""
    root = Path(repo)
    env = _git_environment(root)

    def git(*args: str) -> str:
        done = subprocess.run(
            ["git", "-c", "core.fileMode=false", "-c", "core.autocrlf=input", *args],
            cwd=root, env=env, capture_output=True, text=True, check=False,
        )
        if done.returncode != 0:
            raise RunRecordError(f"git {args[0]} failed: {done.stderr.strip()}")
        return done.stdout.strip()

    return {"sha": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def capture_checkpoint(directory: str | Path, *, model_id: str, revision: str) -> dict[str, Any]:
    """Hash every file. The revision must come from the caller, as a local directory doesn't record one."""
    root = Path(directory)
    if not root.is_dir():
        raise RunRecordError(f"Checkpoint directory does not exist: {root}")
    files = [file_entry(p, relative_to=root) for p in sorted(root.rglob("*")) if p.is_file()]
    if not files:
        raise RunRecordError(f"Checkpoint directory is empty: {root}")
    combined = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"model_id": model_id, "revision": revision, "digest": f"sha256:{hashlib.sha256(combined).hexdigest()}", "files": files}


def capture_dependencies(repo: str | Path) -> dict[str, Any]:
    """The lock files and also what is really installed, since the two may disagree."""
    root = Path(repo)
    locks = [file_entry(root / name, relative_to=root) for name in ("requirements-dev.lock", "Cargo.lock")]
    installed = sorted({f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions() if d.metadata["Name"]})
    if not installed:
        raise RunRecordError("Could not list the installed packages")
    environment = hashlib.sha256("\n".join(installed).encode("utf-8")).hexdigest()
    return {"lock_files": locks, "environment_digest": f"sha256:{environment}", "installed_packages": installed}


def capture_runtime(native_extension: str | Path, *, package_version: str) -> dict[str, Any]:
    import torch

    return {
        "package_version": package_version,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "native_extension_digest": sha256_file(native_extension),
        "platform": platform.platform(),
    }


def build_run_record(
    *,
    commit: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    dependency_lock: Mapping[str, Any],
    runtime: Mapping[str, Any],
    device: str,
    precision: str,
    sampler: str,
    seeds: Mapping[str, Any],
    edit_graph_digest: str,
    command: Sequence[str],
    artifact_manifest: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    record = {
        "schema_version": 1,
        "kind": RUN_RECORD_KIND,
        "commit": dict(commit),
        "checkpoint": dict(checkpoint),
        "dependency_lock": dict(dependency_lock),
        "runtime": dict(runtime),
        "device": device,
        "precision": precision,
        "sampler": sampler,
        "seeds": dict(seeds),
        "edit_graph_digest": edit_graph_digest,
        "command": list(command),
        "artifact_manifest": [dict(entry) for entry in artifact_manifest],
    }
    validate_run_record(record)
    return record


def sign_run_record(record: Mapping[str, Any], *, signing_key: str, signer_id: str = "nf.run_record.v1") -> dict[str, Any]:
    """Validate, then sign with the same Ed25519 envelope as patches and edit graphs."""
    validate_run_record(record)
    return sign_patch_envelope(record, signing_key=signing_key, signer_id=signer_id)


def verify_run_record(envelope: Mapping[str, Any], *, public_key: str, artifact_root: str | Path | None = None) -> dict[str, Any]:
    """Check the signature against a trusted key, the fields, and optionally the artifacts on disk."""
    try:
        verify_patch_signature(envelope, public_key=public_key)
    except Exception as exc:
        raise RunRecordError(f"Run record signature check failed: {exc}") from exc
    record = envelope["patch"]
    validate_run_record(record)
    if artifact_root is not None:
        for entry in record["artifact_manifest"]:
            path = Path(artifact_root) / entry["path"]
            if not path.is_file():
                raise RunRecordError(f"Artifact is missing: {entry['path']}")
            if sha256_file(path) != entry["digest"]:
                raise RunRecordError(f"Artifact does not match the record: {entry['path']}")
    return record
