"""T039 reload component only: real standalone bytes and bounded CPU children.

No accepted-pointer or scientific promotion gate is implemented by these tests.
"""

import copy
import json
import os
import struct
import subprocess
from pathlib import Path
from typing import Any

import pytest
from native_fixture import make_fixture, wire
from neural_foundry import NativeError, _native, models
from neural_foundry.reload_verification import Probe, ReloadVerificationError, verify_reload


@pytest.fixture(scope="module")
def runtime(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, Path]:
    root = tmp_path_factory.mktemp("reload-worker")
    return models.qualify(root, enabled=True), root


def bundle(tmp_path: Path, *, buffered: bool = False,
           dtype: str = "f64", edited: bool = False) -> tuple[Path, dict[str, Any]]:
    fixture = make_fixture(dtype)
    model = copy.deepcopy(fixture.manifest)
    checkpoint = fixture.checkpoint
    config = b"{}"
    if edited:
        plan = _native.compile_plan(fixture.source, wire(fixture.program),
                                    wire(fixture.space), wire(fixture.policy), {})
        candidate = _native.apply_plan(plan, wire(fixture.smoke))
        values = candidate.tensor_bytes("w")
        model = json.loads(candidate.snapshot().manifest_json())
        header = wire({"w": {"dtype": dtype.upper(), "shape": [2, 2],
                              "data_offsets": [0, len(values)]}})
        header += b" " * (-len(header) % 8)
        checkpoint = struct.pack("<Q", len(header)) + header + values
    if buffered:
        weights = struct.pack("<4d", 1, 2, 3, 4)
        offsets = struct.pack("<2d", 0.5, -1)
        header = wire({"w": {"dtype": "F64", "shape": [2, 2], "data_offsets": [0, 32]},
                       "offset": {"dtype": "F64", "shape": [2], "data_offsets": [32, 48]}})
        header += b" " * (-len(header) % 8)
        checkpoint = struct.pack("<Q", len(header)) + header + weights + offsets
        config = _native.canonical_json(wire({"schema_version": 1, "rows": 2, "columns": 2,
                                             "tied": True, "buffer": "offset"}))
        model["architecture"] = {"id": "nf.fixture.linear_buffered", "version": "1",
                                 "config": _native.content_digest(config)}
        model["buffers"] = {name: {"dtype": "f64", "shape": [2],
                                   "digest": _native.content_digest(offsets),
                                   "storage_group": "offset"}
                            for name in ("offset", "offset_alias")}
    tokenizer = b"inert fixture tokenizer"
    store = tmp_path / "store"
    (store / "blobs").mkdir(parents=True)
    (store / "staging").mkdir()
    (store / ".lock").touch(mode=0o600)

    def put(data: bytes) -> dict[str, Any]:
        digest = _native.content_digest(data)
        (store / "blobs" / digest[7:]).write_bytes(data)
        return {"digest": digest, "bytes": len(data)}

    checkpoint_blob = put(checkpoint)
    assets = {"config": put(config), "tokenizer": put(tokenizer)}
    model["assets"] = {"checkpoint": checkpoint_blob["digest"],
                       **{name: blob["digest"] for name, blob in assets.items()}}
    return store, {"schema_version": 1, "kind": "standalone_dense_checkpoint",
                   "alias_contract": "canonical_group_shared_storage_v1", "model": model,
                   "checkpoint": checkpoint_blob, "assets": assets}


def invoke(runtime: Any, store: Path, root: dict[str, Any],
           probes: list[Probe] | None = None, **kwargs: Any) -> bytes:
    data = _native.canonical_json(wire(root))
    digest = _native.content_digest(data)
    (store / "blobs" / digest[7:]).write_bytes(data)
    return verify_reload(runtime, store, digest,
                         expected_manifest=_native.content_digest(
                             _native.canonical_json(wire(root["model"]))),
                         probes=[Probe((1.0, 2.0), (5.0, 11.0), 0.0)]
                         if probes is None else probes, **kwargs)


@pytest.mark.parametrize("buffered", [False, True])
def test_reload_promotion_real_fresh_process_inventory_and_parity(
    runtime: tuple[Any, Path], tmp_path: Path, buffered: bool,
) -> None:
    worker, scratch = runtime
    store, root = bundle(tmp_path, buffered=buffered)
    pointer = tmp_path / "accepted"
    pointer.write_bytes(b"unchanged-pointer")
    blobs_before = {p.name: p.read_bytes() for p in (store / "blobs").iterdir()}
    expected = (5.5, 10.0) if buffered else (5.0, 11.0)
    receipt = json.loads(invoke(worker, store, root, [Probe((1.0, 2.0), expected, 0.0)]))
    assert receipt["promotion_status"] == "not_evaluated"
    assert receipt["outputs_eligible"] is False
    assert receipt["reload_mode"] == "fresh_trusted_torch_cpu_process"
    inference = receipt["inference"]
    assert inference["cleanup_confirmed"] is True
    observed = {v["name"]: v for v in inference["child_report"]["before"]}
    assert observed["w"]["group"] == observed["tied"]["group"]
    for role, field in [("parameter", "parameters"), ("buffer", "buffers")]:
        for name, spec in root["model"][field].items():
            assert observed[name]["role"] == role
            for key in ["shape", "dtype", "digest"]:
                assert observed[name][key] == spec[key]
    if buffered:
        assert observed["offset"]["group"] == observed["offset_alias"]["group"]
        assert observed["offset"]["group"] != observed["w"]["group"]
    process = json.loads(next(scratch.glob(f"runtime-*/{inference['job']}/process.json")).read_bytes())
    assert process["pid"] != os.getpid()
    assert not Path(f"/proc/{process['pid']}").exists()
    assert pointer.read_bytes() == b"unchanged-pointer"
    assert all((store / "blobs" / name).read_bytes() == value
               for name, value in blobs_before.items())
    assert not (store / "active").exists()


def test_reload_promotion_materialized_native_edit_without_candidate_reference(
    runtime: tuple[Any, Path], tmp_path: Path,
) -> None:
    # Builder-local source, plan and candidate handles have gone out of scope.
    store, root = bundle(tmp_path, edited=True)
    receipt = json.loads(invoke(runtime[0], store, root,
                               [Probe((1.0, 2.0), (10.0, 22.0), 0.0)]))
    assert receipt["parameters"] == root["model"]["parameter_digest"]
    assert receipt["inference"]["outputs"] == [[10.0, 22.0]]


@pytest.mark.parametrize("damage", ["coordinate", "alias", "dtype", "buffer", "config",
                                     "missing_asset", "artifact", "packing", "output"])
def test_reload_promotion_corruption_fails_closed(
    runtime: tuple[Any, Path], tmp_path: Path, damage: str,
) -> None:
    store, root = bundle(tmp_path, buffered=True)
    model = root["model"]
    if damage in ("coordinate", "artifact"):
        path = store / "blobs" / root["checkpoint"]["digest"][7:]
        data = bytearray(path.read_bytes())
        if damage == "coordinate":
            data[-1] ^= 1
        else:
            del data[-8:]
        path.write_bytes(data)
    elif damage == "alias":
        model["parameters"]["tied"]["storage_group"] = "untied"
    elif damage == "dtype":
        model["parameters"]["w"]["dtype"] = "f32"
    elif damage == "buffer":
        model["buffers"]["offset_alias"]["digest"] = _native.content_digest(b"wrong")
    elif damage == "config":
        model["architecture"]["config"] = _native.content_digest(b"wrong")
    elif damage == "missing_asset":
        (store / "blobs" / root["assets"]["tokenizer"]["digest"][7:]).unlink()
    elif damage == "packing":
        model["format"] = "unknown_quantization"
    expected = (999.0, 10.0) if damage == "output" else (5.5, 10.0)
    with pytest.raises((ReloadVerificationError, NativeError, OSError)):
        invoke(runtime[0], store, root, [Probe((1.0, 2.0), expected, 0.0)])
    assert not (store / "accepted").exists()


@pytest.mark.parametrize("probes", [[], [Probe((), (5.0,), 0)],
                                    [Probe((1.0, 2.0), (), 0)],
                                    [Probe((float("nan"),), (1.0,), 0)],
                                    [Probe((1.0, 2.0), (5.0, 11.0), -1)],
                                    [Probe((1.0, 2.0), (5.0, 11.0), 1e-4)]])
def test_reload_promotion_missing_invalid_probes(runtime: tuple[Any, Path], tmp_path: Path,
                                                probes: list[Probe]) -> None:
    store, root = bundle(tmp_path)
    with pytest.raises(ReloadVerificationError):
        invoke(runtime[0], store, root, probes)


def test_reload_promotion_frozen_identity_unknown_root_and_bounds(
    runtime: tuple[Any, Path], tmp_path: Path,
) -> None:
    store, root = bundle(tmp_path)
    data = _native.canonical_json(wire(root))
    digest = _native.content_digest(data)
    path = store / "blobs" / digest[7:]
    path.write_bytes(data)
    with pytest.raises(NativeError, match="frozen model identity"):
        verify_reload(runtime[0], store, digest, expected_manifest="sha256:" + "0" * 64,
                      probes=[Probe((1.0, 2.0), (5.0, 11.0), 0)])
    root["accepted"] = True
    with pytest.raises(NativeError, match="invalid export manifest"):
        invoke(runtime[0], store, root)
    del root["accepted"]
    root["checkpoint"]["bytes"] = 2**40
    with pytest.raises(NativeError, match="exceed limits"):
        invoke(runtime[0], store, root)


def test_reload_promotion_f32_and_frozen_tolerance(runtime: tuple[Any, Path], tmp_path: Path) -> None:
    store, root = bundle(tmp_path, dtype="f32")
    receipt = json.loads(invoke(runtime[0], store, root,
                               [Probe((1.0, 2.0), (5.000001, 11.0), 1e-5)]))
    assert all(v["dtype"] == "f32" for v in receipt["inference"]["child_report"]["before"])
    with pytest.raises(ReloadVerificationError, match="parity failed"):
        invoke(runtime[0], store, root, [Probe((1.0, 2.0), (5.000001, 11.0), 0)])


def test_reload_promotion_valid_empty_asset_and_large_outputs(
    runtime: tuple[Any, Path], tmp_path: Path,
) -> None:
    store, root = bundle(tmp_path)
    empty = _native.content_digest(b"")
    (store / "blobs" / empty[7:]).write_bytes(b"")
    root["assets"]["tokenizer"] = {"digest": empty, "bytes": 0}
    root["model"]["assets"]["tokenizer"] = empty
    receipt = json.loads(invoke(runtime[0], store, root,
                               [Probe((1e6, 1e6), (3e6, 7e6), 0.0)]))
    assert receipt["assets"]["tokenizer"] == empty
    assert receipt["inference"]["outputs"] == [[3e6, 7e6]]
    with pytest.raises(ReloadVerificationError, match="input magnitude exceeded"):
        invoke(runtime[0], store, root, [Probe((1e7, 1e7), (3e7, 7e7), 0.0)])


def test_reload_promotion_native_cli_export_to_real_child(
    runtime: tuple[Any, Path], tmp_path: Path,
) -> None:
    # Use the actual existing native exporter, not a Python imitation of its output.
    project = Path(__file__).resolve().parents[1]
    executable = project / "target" / "debug" / "nf"
    assert executable.is_file(), "build the local nf CLI before running this integration test"
    fixture = make_fixture()
    store, root = bundle(tmp_path)
    del root["assets"]["tokenizer"]
    del root["model"]["assets"]["tokenizer"]
    assert root["model"] == fixture.manifest
    root_bytes = _native.canonical_json(wire(root))
    source_export = _native.content_digest(root_bytes)
    (store / "blobs" / source_export[7:]).write_bytes(root_bytes)
    store.chmod(0o700)
    request = tmp_path / "apply.json"
    request.write_bytes(wire({"schema_version": 1, "store": str(store),
                              "source_export": source_export, "program": fixture.program,
                              "policy": fixture.policy, "search_space": fixture.space,
                              "smoke_suite": fixture.smoke}))
    source_before = fixture.source.tensor_bytes("w")
    # Freeze independently BEFORE the exporter runs, not from its output bundle.
    plan = _native.compile_plan(fixture.source, wire(fixture.program),
                                wire(fixture.space), wire(fixture.policy), {})
    candidate = _native.apply_plan(plan, wire(fixture.smoke))
    expected_manifest = candidate.snapshot().manifest_digest
    del candidate, plan
    accepted = tmp_path / "accepted"
    accepted.write_bytes(b"source-remains-accepted")
    exported = subprocess.run([str(executable), "ir", "apply", str(request)],
                              capture_output=True, check=True, timeout=30)
    envelope = json.loads(exported.stdout)
    assert envelope["ok"] is True
    native_receipt = envelope["result"]["export"]
    assert native_receipt["promotion_status"] == "not_evaluated"
    export_root = native_receipt["export_root"]
    exported_bundle = json.loads((store / "blobs" / export_root[7:]).read_bytes())
    assert _native.content_digest(_native.canonical_json(
        wire(exported_bundle["model"]))) == expected_manifest
    # The exporter has exited. No native candidate handle is supplied to reload.
    receipt = json.loads(verify_reload(runtime[0], store, export_root,
                                       expected_manifest=expected_manifest,
                                       probes=[Probe((1.0, 2.0), (10.0, 22.0), 0.0)]))
    assert receipt["export_root"] == native_receipt["export_root"]
    assert receipt["parameters"] == native_receipt["transaction"]["after_parameters"]
    assert receipt["inference"]["outputs"] == [[10.0, 22.0]]
    assert fixture.source.tensor_bytes("w") == source_before
    assert accepted.read_bytes() == b"source-remains-accepted"
    assert not (store / "active").exists()


def test_reload_promotion_missing_child_evidence_is_rejected(
    runtime: tuple[Any, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, root = bundle(tmp_path)
    real_infer = models.infer

    def without_child(*args: Any, **kwargs: Any) -> Any:
        result = real_infer(*args, **kwargs)
        receipt = json.loads(result.receipt_bytes())
        del receipt["child_report"]

        class Incomplete:
            status = result.status
            outputs = result.outputs

            def receipt_bytes(self) -> bytes:
                return wire(receipt)

        return Incomplete()

    monkeypatch.setattr(models, "infer", without_child)
    with pytest.raises(ReloadVerificationError, match="runtime inventory evidence"):
        invoke(runtime[0], store, root)


def test_reload_promotion_cancel_before_read(runtime: tuple[Any, Path], tmp_path: Path) -> None:
    store, root = bundle(tmp_path)
    token = _native.CancellationHandle()
    token.cancel()
    with pytest.raises(ReloadVerificationError, match="cancelled before reading"):
        invoke(runtime[0], store, root, cancellation=token)


def test_native_restore_rejects_substituted_model_before_checkpoint_read(tmp_path: Path) -> None:
    store, root = bundle(tmp_path)
    expected = _native.content_digest(_native.canonical_json(wire(root["model"])))
    root["model"]["architecture"]["version"] = "substituted"
    raw = _native.canonical_json(wire(root))
    digest = _native.content_digest(raw)
    (store / "blobs" / digest[7:]).write_bytes(raw)
    (store / "blobs" / root["checkpoint"]["digest"][7:]).unlink()
    with pytest.raises(NativeError, match="frozen model identity") as denied:
        _native.restore_checkpoint(str(store), digest, expected)
    assert denied.value.code == "SOURCE_MISMATCH"


@pytest.mark.parametrize("unsafe", ["symlink", "hardlink", "fifo", "directory", "ancestor"])
def test_native_restore_read_only_safe_cas(tmp_path: Path, unsafe: str) -> None:
    store, root = bundle(tmp_path)
    raw = _native.canonical_json(wire(root))
    digest = _native.content_digest(raw)
    (store / "blobs" / digest[7:]).write_bytes(raw)
    expected = _native.content_digest(_native.canonical_json(wire(root["model"])))
    before = {str(p.relative_to(store)): p.read_bytes() for p in store.rglob("*") if p.is_file()}
    source = _native.restore_checkpoint(str(store), digest, expected)
    assert source.manifest_digest == expected
    assert source.forward([1, 2]) == [5, 11]
    assert before == {str(p.relative_to(store)): p.read_bytes()
                      for p in store.rglob("*") if p.is_file()}
    blob = store / "blobs" / root["checkpoint"]["digest"][7:]
    original = tmp_path / "original"
    blob.rename(original)
    if unsafe == "symlink":
        blob.symlink_to(original)
    elif unsafe == "hardlink":
        blob.hardlink_to(original)
    elif unsafe == "fifo":
        os.mkfifo(blob)
    elif unsafe == "directory":
        blob.mkdir()
    else:
        blob.write_bytes(original.read_bytes())
        alias = tmp_path / "alias"
        alias.symlink_to(store, target_is_directory=True)
        store = alias
    with pytest.raises(NativeError):
        _native.restore_checkpoint(str(store), digest, expected)


def test_native_restore_limits_cancel_and_missing_store(tmp_path: Path) -> None:
    store, root = bundle(tmp_path)
    raw = _native.canonical_json(wire(root))
    digest = _native.content_digest(raw)
    (store / "blobs" / digest[7:]).write_bytes(raw)
    expected = _native.content_digest(_native.canonical_json(wire(root["model"])))
    bounds: list[dict[str, Any]] = [
        {"max_total_bytes": 1}, {"max_checkpoint_bytes": 1},
        {"max_asset_bytes": 1}, {"max_tensors": 1},
        {"max_total_bytes": 73400321}, {"max_tensors": 0},
    ]
    for limits in bounds:
        with pytest.raises(NativeError):
            _native.restore_checkpoint(str(store), digest, expected, **limits)
    missing = tmp_path / "must-not-create"
    with pytest.raises(NativeError):
        _native.restore_checkpoint(str(missing), digest, expected)
    assert not missing.exists()
    token = _native.CancellationHandle()
    token.cancel()
    with pytest.raises(_native.CancelledError):
        _native.restore_checkpoint(str(missing), digest, expected, cancellation=token)
    assert not missing.exists()
    (store / ".lock").unlink()
    with pytest.raises(NativeError):
        _native.restore_checkpoint(str(store), digest, expected)
    assert not (store / ".lock").exists()


def test_reload_promotion_timeout_and_symlink_are_not_evidence(
    runtime: tuple[Any, Path], tmp_path: Path,
) -> None:
    store, root = bundle(tmp_path)
    with pytest.raises(ReloadVerificationError, match="incomplete fresh-process"):
        invoke(runtime[0], store, root, wall_ms=100)
    path = store / "blobs" / root["checkpoint"]["digest"][7:]
    target = tmp_path / "checkpoint"
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(NativeError):
        invoke(runtime[0], store, root)
