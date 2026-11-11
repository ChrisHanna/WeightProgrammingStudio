"""Private strict one-shot claims shared by custodian audit and evaluation."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class ClaimSpent(ValueError):
    pass


class StateBusy(ValueError):
    pass


def _wire(document: Mapping[str, Any]) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sync_directory(path: Path) -> None:
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def write_private(path: Path, data: bytes) -> None:
    descriptor, name = tempfile.mkstemp(prefix=".custodian-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def exclusive_state(state: Path) -> Iterator[None]:
    state.mkdir(parents=True, mode=0o700, exist_ok=True)
    descriptor = os.open(state / ".custodian.lock",
                         os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        try:
            if os.name == "nt":
                import msvcrt

                if os.fstat(descriptor).st_size == 0:
                    os.write(descriptor, b"\0")
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in {errno.EACCES, errno.EAGAIN}:
                raise StateBusy("another operation owns the custodian state") from error
            raise
        yield
    finally:
        os.close(descriptor)


class StrictClaims:
    def __init__(self, state: Path) -> None:
        self.root = state / "claims"
        self.root.mkdir(mode=0o700, exist_ok=True)

    def has_started(self, scope: str) -> bool:
        return (self.root / (hashlib.sha256(scope.encode()).hexdigest() + ".json")).exists()

    def begin(self, scope: str, request: Mapping[str, Any]) -> Path:
        path = self.root / (hashlib.sha256(scope.encode()).hexdigest() + ".json")
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                                 | getattr(os, "O_NOFOLLOW", 0), 0o600)
        except FileExistsError as error:
            raise ClaimSpent("this operation has already consumed its only start") from error
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_wire({"kind": "nf_strict_custody_claim_v1", "state": "started",
                                "request_digest": "sha256:" + hashlib.sha256(_wire(request)).hexdigest()}))
            stream.flush()
            os.fsync(stream.fileno())
        _sync_directory(self.root)
        return path

    def finish(self, claim: Path, receipt: Mapping[str, Any]) -> None:
        data = _wire(receipt)
        receipt_path = claim.with_suffix(".receipt.json")
        write_private(receipt_path, data)
        document = json.loads(claim.read_bytes())
        document.update(state="complete", receipt_digest="sha256:" + hashlib.sha256(data).hexdigest())
        write_private(claim, _wire(document))
