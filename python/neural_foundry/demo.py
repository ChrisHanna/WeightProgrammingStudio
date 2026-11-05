"""Run the packaged finite M1 CPU fixture, without external models or Torch."""

import json
import struct
from importlib.resources import files
from typing import Any, cast

from . import capabilities, ir, surgeon


def run() -> dict[str, Any]:
    fixture = json.loads(files("neural_foundry").joinpath("data/m1_fixture.json").read_bytes())
    source = surgeon.load(
        fixture["manifest"],
        bytes.fromhex(fixture["checkpoint_hex"]),
        {name: bytes.fromhex(value) for name, value in fixture["assets_hex"].items()},
    )
    before = source.tensor_bytes("w")
    plan = ir.compile(source, fixture["program"], fixture["space"], fixture["policy"])
    candidate = surgeon.apply(plan, fixture["smoke_suite"])
    replayed = surgeon.apply(plan, fixture["smoke_suite"])
    snapshot = candidate.snapshot()
    if candidate.digest != replayed.digest or candidate.tensor_bytes("w") != snapshot.tensor_bytes(
        "w"
    ):
        raise RuntimeError("M1 numerical replay/snapshot mismatch")
    if source.tensor_bytes("w") != before:
        raise RuntimeError("M1 source changed")
    values = struct.unpack("<4d", candidate.tensor_bytes("w"))
    if values != (2.0, 4.0, 6.0, 8.0):
        raise RuntimeError("M1 edited values differ")
    return {
        "schema_version": 1,
        "example": "m1_python_direct",
        "candidate": candidate.digest,
        "replayed_candidate": replayed.digest,
        "replay_mode": "same_immutable_root_plan_execution",
        "parameter_digest": candidate.parameter_digest,
        "values": list(values),
        "source_preserved": True,
        "snapshot_parity": True,
        "receipt": cast(dict[str, Any], json.loads(candidate.receipt_json())),
        "capabilities": capabilities(),
        "promotion_status": "not_evaluated",
    }


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
