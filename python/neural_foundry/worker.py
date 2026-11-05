"""Data-only client for an assigned private broker channel, never a model handle."""

import json
from collections.abc import Mapping
from typing import Any, Protocol, cast

MAX_FRAME = 16384


class Channel(Protocol):
    def read(self, size: int = -1, /) -> bytes | None: ...
    def write(self, data: bytes, /) -> int | None: ...
    def flush(self) -> None: ...


def _read_exact(channel: Channel, size: int) -> bytes:
    parts = bytearray()
    while len(parts) < size:
        data = channel.read(size - len(parts))
        if not data:
            raise EOFError("worker broker channel closed")
        parts.extend(data)
    return bytes(parts)


def read_frame(channel: Channel) -> dict[str, Any]:
    length = int.from_bytes(_read_exact(channel, 4), "big")
    if length > MAX_FRAME:
        raise ValueError("worker broker frame exceeds byte bound")
    value = json.loads(_read_exact(channel, length))
    if not isinstance(value, dict):
        raise TypeError("worker broker envelope must be an object")
    return cast(dict[str, Any], value)


def write_frame(channel: Channel, value: Mapping[str, object]) -> None:
    data = json.dumps(dict(value), separators=(",", ":")).encode()
    if len(data) > MAX_FRAME:
        raise ValueError("worker request exceeds byte bound")
    frame = len(data).to_bytes(4, "big") + data
    offset = 0
    while offset < len(frame):
        written = channel.write(frame[offset:])
        if not written:
            raise EOFError("worker broker channel closed while writing")
        offset += written
    channel.flush()


class Client:
    """Use only the worker-specific channel/scope assigned by its controller."""

    def __init__(self, channel: Channel) -> None:
        self._channel = channel
        self.welcome = read_frame(channel)
        self._sequence = 0
        if self.welcome.get("schema_version") != 1:
            raise ValueError("unsupported worker broker protocol")

    def request(self, action: Mapping[str, object]) -> dict[str, Any]:
        envelope = {
            key: self.welcome[key]
            for key in ("contract", "revision", "worker", "lease_id", "split")
        }
        envelope.update(schema_version=1, sequence=self._sequence, action=dict(action))
        write_frame(self._channel, envelope)
        reply = read_frame(self._channel)
        if reply.get("schema_version") != 1:
            raise ValueError("unsupported worker broker reply version")
        if reply.get("sequence") == self._sequence:
            self._sequence += 1
        return reply
