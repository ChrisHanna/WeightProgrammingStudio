"""CLI, installed extension and SDK use the same Rust identities and errors."""

import json
import os
import struct
import subprocess
from pathlib import Path
from typing import Any

import neural_foundry as nf
import pytest
from native_fixture import NativeFixture, make_fixture, wire
from neural_foundry import _native

ROOT = Path(__file__).resolve().parents[1]
CLI = Path(os.environ.get("NF_CLI", ROOT / "target/debug/nf"))


def command(
    tmp_path: Path, group: str, action: str, data: bytes
) -> subprocess.CompletedProcess[bytes]:
    assert CLI.is_file(), "Build the real native CLI with cargo build -p nf-cli first"
    path = tmp_path / "request.json"
    path.write_bytes(data)
    return subprocess.run([str(CLI), group, action, str(path)], capture_output=True, check=False)


def cli_result(tmp_path: Path, group: str, action: str, data: bytes) -> Any:
    completed = command(tmp_path, group, action, data)
    assert completed.returncode == 0, completed.stderr.decode()
    return json.loads(completed.stdout)["result"]


def source_export(tmp_path: Path, fixture: NativeFixture) -> tuple[Path, str]:
    """Independent on-disk fixture; real Rust restore must verify every byte."""
    store = tmp_path / "store"
    (store / "blobs").mkdir(parents=True)
    (store / "staging").mkdir()
    (store / ".lock").touch()
    envelope = {
        "schema_version": 1,
        "kind": "standalone_dense_checkpoint",
        "alias_contract": "canonical_group_shared_storage_v1",
        "model": fixture.manifest,
        "checkpoint": {
            "digest": nf.content_digest(fixture.checkpoint),
            "bytes": len(fixture.checkpoint),
        },
        "assets": {
            name: {"digest": nf.content_digest(data), "bytes": len(data)}
            for name, data in fixture.assets.items()
        },
    }
    root = nf.canonicalize(envelope)
    identity = nf.content_digest(root)
    for data in [fixture.checkpoint, *fixture.assets.values(), root]:
        path = store / "blobs" / nf.content_digest(data).removeprefix("sha256:")
        path.write_bytes(data)
        path.chmod(0o400)
    return store, identity


def request(store: Path, root: str, fixture: NativeFixture) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "store": str(store),
        "source_export": root,
        "program": fixture.program,
        "search_space": fixture.space,
        "policy": fixture.policy,
        "smoke_suite": fixture.smoke,
    }


def test_core_conformance_shared_canonical_golden_vectors(tmp_path: Path) -> None:
    vectors = json.loads((ROOT / "protocols/canonical-v1-vectors.json").read_bytes())
    for vector in vectors:
        raw = vector["input"].encode()
        encoded = nf.canonicalize(raw)
        assert encoded == vector["canonical"].encode()
        assert encoded == _native.canonical_json(raw)
        assert nf.content_digest(encoded) == vector["digest"]
        result = cli_result(tmp_path, "json", "canonicalize", raw)
        assert result["canonical"].encode() == encoded
        assert result["digest"] == vector["digest"]
        assert cli_result(tmp_path, "blob", "digest", encoded)["digest"] == vector["digest"]


def test_core_conformance_shared_error_vectors_and_json_paths(tmp_path: Path) -> None:
    vectors = json.loads((ROOT / "protocols/error-v1-vectors.json").read_bytes())
    for vector in vectors:
        raw = vector["input"].encode()
        with pytest.raises(nf.NativeError) as caught:
            nf.canonicalize(raw)
        error = caught.value
        assert error.code == vector["code"]
        assert error.path == vector["path"]
        completed = command(tmp_path, "json", "canonicalize", raw)
        assert completed.returncode == 2
        assert json.loads(completed.stderr)["error"] == {
            "code": error.code,
            "path": error.path,
            "message": error.message,
        }


def test_core_conformance_protocol_versions_are_exact_and_shared(tmp_path: Path) -> None:
    agreement = nf.negotiate()
    required = agreement["agreed"]
    raw = wire({"schema_version": 1, "required": required})
    assert agreement == json.loads(_native.negotiate_protocols(raw))
    assert agreement == cli_result(tmp_path, "protocol", "negotiate", raw)
    assert agreement["core_version"] == nf.__version__
    for requirements, code in [
        ({"core_api": 2}, "UNSUPPORTED_VERSION"),
        ({"core_api": 1, "unavailable": 1}, "UNSUPPORTED_CAPABILITY"),
        ({}, "INVALID_VALUE"),
    ]:
        with pytest.raises(nf.NativeError) as caught:
            nf.negotiate(requirements)
        assert caught.value.code == code
        completed = command(
            tmp_path, "protocol", "negotiate", wire({"schema_version": 1, "required": requirements})
        )
        assert json.loads(completed.stderr)["error"]["code"] == code


def test_core_conformance_program_space_and_plan_identities_match_cli(tmp_path: Path) -> None:
    fixture = make_fixture()
    program = nf.ir.program(fixture.program)
    space = nf.ir.search_space(fixture.space)
    assert (
        program.digest == cli_result(tmp_path, "ir", "validate", program.json())["program_digest"]
    )
    assert (
        space.digest
        == cli_result(tmp_path, "search-space", "validate", space.json())["search_space_digest"]
    )
    source = nf.surgeon.load(
        nf.validate("manifest", fixture.manifest), fixture.checkpoint, fixture.assets
    )
    plan = nf.ir.compile(source, program, space, fixture.policy)
    store, root = source_export(tmp_path, fixture)
    compiled = cli_result(tmp_path, "ir", "compile", wire(request(store, root, fixture)))
    assert compiled["plan_digest"] == plan.digest
    assert compiled["plan"] == json.loads(plan.summary_json())
    assert compiled["execution"] == "not_run"


def test_core_conformance_actual_candidate_receipt_matches_native_cli(tmp_path: Path) -> None:
    fixture = make_fixture()
    plan = nf.ir.compile(fixture.source, fixture.program, fixture.space, fixture.policy)
    candidate = nf.surgeon.apply(plan, fixture.smoke)
    store, root = source_export(tmp_path, fixture)
    applied = cli_result(tmp_path, "ir", "apply", wire(request(store, root, fixture)))
    assert applied["export"]["candidate"] == candidate.digest
    assert applied["export"]["transaction"] == json.loads(candidate.receipt_json())
    assert applied["export"]["promotion_status"] == "not_evaluated"
    assert struct.unpack("<4d", candidate.tensor_bytes("w")) == (2, 4, 6, 8)
    assert candidate.snapshot().parameter_digest == candidate.parameter_digest
    assert struct.unpack("<4d", fixture.source.tensor_bytes("w")) == (1, 2, 3, 4)


def test_core_conformance_ir_errors_and_cancel_keep_native_types(tmp_path: Path) -> None:
    fixture = make_fixture()
    extra = dict(fixture.program, execute_python="no")
    with pytest.raises(nf.NativeError) as caught:
        nf.ir.program(extra)
    completed = command(tmp_path, "ir", "validate", wire(extra))
    assert json.loads(completed.stderr)["error"] == {
        "code": caught.value.code,
        "path": caught.value.path,
        "message": caught.value.message,
    }
    token = nf.CancellationHandle()
    token.cancel()
    with pytest.raises(nf.CancelledError):
        nf.surgeon.load(fixture.manifest, fixture.checkpoint, fixture.assets, token)
