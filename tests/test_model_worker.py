"""Actual pinned Torch CPU child processes; native source/state/parity checks."""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from native_fixture import make_fixture, wire
from neural_foundry import (
    CancellationHandle,
    CancelledError,
    NativeError,
    _native,
    arithmetic,
    models,
)


@pytest.fixture(scope="module")
def runtime(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, Path]:
    root = tmp_path_factory.mktemp("trusted-model-worker")
    return models.qualify(root, enabled=True), root


def buffered(dtype: str = "f64", tied: bool = True) -> _native.SourceHandle:
    values = struct.pack("<4d" if dtype == "f64" else "<4f", 1, 2, 3, 4)
    offsets = struct.pack("<2d" if dtype == "f64" else "<2f", 0.5, -1)
    chunks = [values, offsets]
    header: dict[str, Any] = {
        "w": {"dtype": dtype.upper(), "shape": [2, 2], "data_offsets": [0, len(values)]},
        "offset": {
            "dtype": dtype.upper(),
            "shape": [2],
            "data_offsets": [len(values), len(values) + len(offsets)],
        },
    }
    if not tied:
        start = sum(map(len, chunks))
        chunks.append(values)
        header["tied"] = {
            "dtype": dtype.upper(),
            "shape": [2, 2],
            "data_offsets": [start, start + len(values)],
        }
    data = b"".join(chunks)
    encoded = wire(header)
    encoded += b" " * (-len(encoded) % 8)
    checkpoint = struct.pack("<Q", len(encoded)) + encoded + data
    digest = _native.content_digest
    config = _native.canonical_json(
        wire({"schema_version": 1, "rows": 2, "columns": 2, "tied": True, "buffer": "offset"})
    )
    parameters = {
        name: {
            "dtype": dtype,
            "shape": [2, 2],
            "digest": digest(values),
            "storage_group": "weights" if tied or name == "w" else "independent",
        }
        for name in ["tied", "w"]
    }
    buffers = {
        name: {"dtype": dtype, "shape": [2], "digest": digest(offsets), "storage_group": "offset"}
        for name in ["offset", "offset_alias"]
    }
    parameter_set = {
        "schema_version": 1,
        "kind": "tensor_set",
        "tensors": [
            {"name": name, "shape": [2, 2], "dtype": dtype, "content_digest": digest(values)}
            for name in parameters
        ],
    }
    manifest = {
        "schema_version": 1,
        "architecture": {
            "id": "nf.fixture.linear_buffered",
            "version": "1",
            "config": digest(config),
        },
        "parameters": parameters,
        "buffers": buffers,
        "parameter_digest": digest(_native.canonical_json(wire(parameter_set))),
        "format": "dense_safetensors",
        "packing_policy": None,
        "assets": {"config": digest(config), "checkpoint": digest(checkpoint)},
        "usage_rights": "synthetic model-worker fixture",
    }
    return _native.load_source(wire(manifest), checkpoint, {"config": config})


def view(result: Any) -> dict[str, Any]:
    receipt: dict[str, Any] = json.loads(result.receipt_bytes())
    assert receipt["cleanup_confirmed"] is True
    assert receipt["outputs_eligible"] is False and receipt["promotion_status"] == "not_evaluated"
    return receipt


def test_model_worker_default_off_before_path_or_process(tmp_path: Path) -> None:
    with pytest.raises(NativeError, match="DISABLED"):
        models.qualify(tmp_path / "not-created")
    assert not (tmp_path / "not-created").exists()


@pytest.mark.parametrize("dtype", ["f32", "f64"])
def test_model_worker_real_torch_linear_aliases_and_native_parity(
    runtime: tuple[Any, Path], dtype: str
) -> None:
    worker, root = runtime
    fixture = make_fixture(dtype)
    before = fixture.source.tensor_bytes("w")
    result = models.infer(worker, fixture.source, [[1.0, 2.0], [-1.0, 1.0]])
    receipt = view(result)
    assert result.status == "verified_inference", receipt
    assert result.outputs == [[5.0, 11.0], [1.0, 1.0]]
    child = receipt["child_report"]
    assert child["torch_version"] == "2.9.1+cpu"
    assert child["inference_mode"] is True and child["gradient_enabled"] is False
    assert child["before"] == child["after"]
    tensors = {v["name"]: v for v in child["before"]}
    assert tensors["w"]["group"] == tensors["tied"]["group"]
    assert tensors["w"]["grad_is_none"] is True and tensors["w"]["requires_grad"] is False
    assert child["copies"]["zero_copy"] is False
    assert child["copies"]["native_tensor_bytes_copy"] == len(before)
    assert child["copies"]["torch_clone_bytes"] == len(before)
    assert receipt["peak_rss_bytes"] > 0 and receipt["cpu_ms"] > 0
    assert "AS_hard_cap_not_RSS_limit" in receipt["measurement_scope"]
    assert fixture.source.tensor_bytes("w") == before
    process = json.loads(next(root.glob(f"runtime-*/{receipt['job']}/process.json")).read_bytes())
    assert process["pid"] != os.getpid() and not Path(f"/proc/{process['pid']}").exists()


@pytest.mark.parametrize("dtype", ["f32", "f64"])
def test_model_worker_buffers_and_tied_buffers_are_real_and_unchanged(
    runtime: tuple[Any, Path], dtype: str
) -> None:
    worker, _ = runtime
    source = buffered(dtype)
    before = source.tensor_bytes("offset")
    result = models.infer(worker, source, [[1.0, 2.0], [0.0, 0.0]])
    receipt = view(result)
    assert result.status == "verified_inference", receipt
    assert result.outputs == [[5.5, 10.0], [0.5, -1.0]]
    tensors = {v["name"]: v for v in receipt["child_report"]["before"]}
    assert tensors["offset"]["role"] == "buffer"
    assert tensors["offset"]["group"] == tensors["offset_alias"]["group"]
    assert tensors["offset"]["group"] != tensors["w"]["group"]
    assert source.tensor_bytes("offset") == before
    assert tensors["offset"]["digest"] == _native.content_digest(before)


def test_model_worker_actual_relu_all_three_splits_match_native(runtime: tuple[Any, Path]) -> None:
    worker, _ = runtime
    fixture = arithmetic.fixture()
    metadata = arithmetic.describe(fixture)
    for role in ["development", "selection", "holdout"]:
        cases = [c for c in metadata["inventory"]["cases"] if c["split"] == role][:8]
        inputs = [[float(c["x1"]), float(c["x2"])] for c in cases]
        result = models.infer(worker, fixture.source, inputs)
        assert result.status == "verified_inference", view(result)
        assert result.outputs == [fixture.source.forward(v) for v in inputs]
        assert all(row["dtype"] == "f64" for row in view(result)["child_report"]["after"])


def test_model_worker_equal_independent_values_cannot_fake_factory_ties(
    runtime: tuple[Any, Path],
) -> None:
    worker, _ = runtime
    source = buffered(tied=False)
    with pytest.raises(NativeError, match="tied"):
        models.infer(worker, source, [[1.0, 2.0]])


@pytest.mark.parametrize("change", ["factory", "version", "remote_config"])
def test_model_worker_unknown_factories_remote_code_and_versions_rejected(
    runtime: tuple[Any, Path], change: str
) -> None:
    worker, _ = runtime
    fixture = make_fixture()
    manifest = json.loads(wire(fixture.manifest))
    assets = dict(fixture.assets)
    if change == "factory":
        manifest["architecture"]["id"] = "remote.custom.factory"
    elif change == "version":
        manifest["architecture"]["version"] = "unqualified-version"
    else:
        assets["config"] = wire({"trust_remote_code": True, "auto_map": "attacker.Model"})
        manifest["architecture"]["config"] = _native.content_digest(assets["config"])
        manifest["assets"]["config"] = _native.content_digest(assets["config"])
    source = _native.load_source(wire(manifest), fixture.checkpoint, assets)
    with pytest.raises(NativeError):
        models.infer(worker, source, [[1.0, 2.0]])


def test_model_worker_invalid_files_packing_and_nonfinite_requests_rejected(
    runtime: tuple[Any, Path],
) -> None:
    worker, _ = runtime
    fixture = make_fixture()
    for data in [b"\x80\x04pickle is not Safetensors", fixture.checkpoint[:-1]]:
        with pytest.raises(NativeError):
            _native.load_source(wire(fixture.manifest), data, fixture.assets)
    with pytest.raises(NativeError):
        models.infer(worker, fixture.source, [[float("nan"), 0.0]])
    with pytest.raises(NativeError):
        models.infer(worker, fixture.source, [[1.0]])
    with pytest.raises(NativeError):
        models.infer(worker, fixture.source, [[1.0, 2.0]], address_space_bytes=1)


def test_model_worker_kernel_wall_output_limits_and_cleanup(runtime: tuple[Any, Path]) -> None:
    worker, root = runtime
    source = arithmetic.fixture().source
    wall = models.infer(worker, source, [[1.0, 2.0]], wall_ms=100)
    assert wall.status == "wall_limit", view(wall)
    assert wall.outputs == []
    output = models.infer(worker, source, [[1.0, 2.0]], output_bytes=1024)
    assert output.status == "output_limit", view(output)
    assert output.outputs == []
    for result in [wall, output]:
        receipt = view(result)
        process = json.loads(
            next(root.glob(f"runtime-*/{receipt['job']}/process.json")).read_bytes()
        )
        assert not Path(f"/proc/{process['pid']}").exists()


def test_model_worker_cancel_reaps_exact_kernel_child(runtime: tuple[Any, Path]) -> None:
    worker, root = runtime
    token = CancellationHandle()
    finished = threading.Event()
    known = set(root.glob("runtime-*/job-*/process.json"))

    def cancel() -> None:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if any(path not in known for path in root.glob("runtime-*/job-*/process.json")):
                token.cancel()
                finished.set()
                return
            time.sleep(0.005)
        raise AssertionError("actual model worker never registered")

    thread = threading.Thread(target=cancel)
    thread.start()
    try:
        with pytest.raises(CancelledError):
            models.infer(worker, arithmetic.fixture().source, [[1.0, 2.0]], cancellation=token)
    finally:
        thread.join(timeout=3)
    assert finished.is_set()
    completions = [json.loads(p.read_bytes()) for p in root.glob("runtime-*/job-*/completion.json")]
    assert any(v["status"] == "cancelled" and v["cleanup_confirmed"] for v in completions)


def test_model_worker_sdk_parent_never_imports_torch_and_child_has_no_host_env(
    tmp_path: Path,
) -> None:
    script = """import json,sys
from pathlib import Path
from neural_foundry import arithmetic,models
assert 'torch' not in sys.modules
root=Path(sys.argv[1]);root.mkdir(mode=0o700)
runtime=models.qualify(root,enabled=True)
assert 'torch' not in sys.modules
result=models.infer(runtime,arithmetic.fixture().source,[[2.0,-3.0]])
assert result.status=='verified_inference' and 'torch' not in sys.modules
print(json.dumps(result.outputs))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(tmp_path / "private")],
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
        env={
            **os.environ,
            "NF_HOST_SECRET": "must-not-reach-child",
            "COORDINATOR_TOKEN": "synthetic-secret",
        },
    )
    assert json.loads(result.stdout) == [[-2.0, 5.0, 2.0, -3.0, 5.0]]
