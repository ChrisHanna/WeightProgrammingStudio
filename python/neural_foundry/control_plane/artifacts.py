"""Content-addressed JSON artifacts, with bounded verified reads and atomic publication."""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from pathlib import Path
from typing import Any

from .models import ControlPlaneError, document, encode, fail


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink():
            fail("unsafe_artifact_root", "Artifact root must not be a symbolic link.")
        self.root = root.resolve()

    def _path(self, artifact_id: str) -> Path:
        if not isinstance(artifact_id, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", artifact_id):
            fail("invalid_artifact", "Invalid content-addressed artifact ID.")
        path = self.root / (artifact_id[7:] + ".json")
        if path.is_symlink() or path.parent.resolve() != self.root:
            fail("unsafe_artifact", "Artifact storage entry is unsafe.", 409)
        return path

    def put(self, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict):
            fail("invalid_artifact", "Only JSON object payloads can be imported.")
        raw = encode(document(value))
        artifact_id = "sha256:" + hashlib.sha256(raw).hexdigest()
        destination = self._path(artifact_id)
        scratch = self.root / (".write-" + uuid.uuid4().hex)
        try:
            fd = os.open(scratch, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                # Hard-link publishes without replacing a preexisting object.
                os.link(scratch, destination)
            except FileExistsError:
                pass
            self.read(artifact_id)
        except ControlPlaneError:
            raise
        except OSError:
            fail("artifact_io_error", "Artifact storage operation failed.", 503)
        finally:
            scratch.unlink(missing_ok=True)
        return {"id": artifact_id, "digest": artifact_id, "byte_size": len(raw),
                "media_type": "application/json"}

    def read(self, artifact_id: str) -> dict[str, Any]:
        path = self._path(artifact_id)
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(fd, "rb") as stream:
                raw = stream.read(262145)
        except FileNotFoundError:
            fail("artifact_not_found", "Artifact does not exist.", 404)
        except OSError:
            fail("artifact_io_error", "Artifact storage operation failed.", 503)
        if len(raw) > 262144 or "sha256:" + hashlib.sha256(raw).hexdigest() != artifact_id:
            fail("artifact_integrity_error", "Artifact content failed integrity verification.", 409)
        import json
        try:
            value = json.loads(raw)
        except (ValueError, UnicodeError):
            fail("artifact_integrity_error", "Artifact content is not valid JSON.", 409)
        if not isinstance(value, dict):
            fail("artifact_integrity_error", "Artifact content must be a JSON object.", 409)
        return value
