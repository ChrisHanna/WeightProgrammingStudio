"""Content-addressed JSON artifacts, with bounded verified reads and atomic publication."""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .models import ControlPlaneError, digest, document, encode, fail


def validate_artifact_id(artifact_id: str) -> str:
    if not isinstance(artifact_id, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", artifact_id):
        fail("invalid_artifact", "Invalid content-addressed artifact ID.")
    return artifact_id


class ArtifactStorage(Protocol):
    def put(self, value: dict[str, Any]) -> dict[str, Any]: ...
    def read(self, artifact_id: str) -> dict[str, Any]: ...


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


class HTTPArtifactStore:
    """Authenticated private API transport; redirects never receive credentials."""

    def __init__(self, base_url: str, token: str) -> None:
        parsed = urlsplit(base_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in ("", "/")
                or (parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"})):
            fail("invalid_artifact_endpoint", "Use HTTPS or a loopback HTTP artifact endpoint.")
        if not token or len(token) > 4096 or any(c.isspace() for c in token):
            fail("invalid_artifact_token", "A runtime operator credential is required.")
        self.base_url, self._token = base_url.rstrip("/"), token

    def _request(self, path: str, value: dict[str, Any] | None = None) -> dict[str, Any]:
        import json
        request = Request(self.base_url + "/v1/artifacts" + path,
                          data=encode(value) if value is not None else None,
                          headers={"Authorization": "Bearer " + self._token,
                                   "Content-Type": "application/json"})
        try:
            with build_opener(_NoRedirect()).open(request, timeout=30) as response:
                raw = response.read(524289)
            if len(raw) > 524288:
                fail("artifact_integrity_error", "Artifact response exceeds its bound.", 409)
            result = json.loads(raw)
        except (HTTPError, URLError, OSError):
            fail("artifact_transport_error", "Authenticated artifact request failed.", 503)
        except (ValueError, UnicodeError):
            fail("artifact_integrity_error", "Artifact response is not JSON.", 409)
        if not isinstance(result, dict):
            fail("artifact_integrity_error", "Artifact response must be an object.", 409)
        return result

    def read(self, artifact_id: str) -> dict[str, Any]:
        validate_artifact_id(artifact_id)
        result = self._request("/" + quote(artifact_id, safe=""))
        value = result.get("document")
        if not isinstance(value, dict) or digest(document(value)) != artifact_id:
            fail("artifact_integrity_error", "Remote artifact digest differs.", 409)
        return value

    def put(self, value: dict[str, Any]) -> dict[str, Any]:
        value = document(value)
        expected = digest(value)
        result = self._request("", value)
        if result.get("id") != expected or result.get("digest") != expected:
            fail("artifact_integrity_error", "Remote artifact publication digest differs.", 409)
        if self.read(expected) != value:
            fail("artifact_integrity_error", "Remote artifact publication failed readback.", 409)
        return {"id": expected, "digest": expected, "byte_size": len(encode(value)),
                "media_type": "application/json"}


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink():
            fail("unsafe_artifact_root", "Artifact root must not be a symbolic link.")
        self.root = root.resolve()

    def _path(self, artifact_id: str) -> Path:
        validate_artifact_id(artifact_id)
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
