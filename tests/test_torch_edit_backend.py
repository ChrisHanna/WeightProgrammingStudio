"""Actual separate trusted Torch edits; native receipts alone cannot pass these tests."""

import copy
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import neural_foundry as nf
import pytest
from native_fixture import make_fixture, wire
from neural_foundry import _native, models
from test_cpu_differential import CASES, bind, operand


@pytest.fixture(scope="module")
def runtime(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, Path]:
    root = tmp_path_factory.mktemp("torch-edit-runtime")
    return models.qualify(root, enabled=True), root


def editing(dtype: str, case: str) -> tuple[Any, Any]:
    f = make_fixture(dtype)
    target = {"tensor": "w", "start": [0, 0], "shape": [2, 2]}
    op: dict[str, Any] = {"type": "scale", "target": target, "factor": -0.5}
    operands = {}
    if case in ("replace", "add"):
        operands["operand"] = operand(f, "operand", [0.5, -1, 2, 0.25], [2, 2], dtype)
        op = {
            "type": "replace_slice" if case == "replace" else "add_delta",
            "target": target,
            "value" if case == "replace" else "delta": "operand",
        }
    elif case == "mask":
        operands["mask"] = operand(f, "mask", [0, 1, 1, 0], [2, 2], "u8")
        op = {"type": "mask", "target": target, "mask": "mask"}
    elif case.startswith("permute"):
        op = {
            "type": "permute_axis",
            "target": target,
            "axis": int(case[-1]),
            "permutation": [1, 0],
        }
    elif case.startswith("map"):
        operands["matrix"] = operand(f, "matrix", [0, 1, -1, 0], [2, 2], dtype)
        op = {
            "type": "linear_map",
            "target": target,
            "matrix": "matrix",
            "side": "left" if case.endswith("left") else "right",
        }
    elif case.startswith("project"):
        operands["basis"] = operand(f, "basis", [1, 0], [2, 1], dtype)
        op = {
            "type": "subspace_projection",
            "target": target,
            "basis": "basis",
            "side": "left" if case.endswith("left") else "right",
            "strength": 0.5,
            "orthonormal_tolerance": 1e-6,
        }
    elif case == "low_rank":
        operands["left"] = operand(f, "left", [1, 2], [2, 1], dtype)
        operands["right"] = operand(f, "right", [2, -1], [1, 2], dtype)
        op = {
            "type": "low_rank_update",
            "target": target,
            "left": "left",
            "right": "right",
            "strength": 0.25,
        }
    bind(f, op, dtype)
    f.policy["backend"]["id"] = "nf.trusted_torch_cpu"
    return f, nf.ir.compile(f.source, f.program, f.space, f.policy, operands)


@pytest.mark.parametrize("dtype", ["f32", "f64"])
@pytest.mark.parametrize("case", CASES)
def test_torch_edit_backend_all_families_actual_worker_full_bytes(
    runtime: tuple[Any, Path], dtype: str, case: str
) -> None:
    f, plan = editing(dtype, case)
    before = f.source.tensor_bytes("w")
    native = nf.surgeon.apply(plan, f.smoke)
    edited = models.edit(runtime[0], plan, f.smoke)
    candidate = edited.candidate
    report = json.loads(edited.receipt_bytes())
    assert report["status"] == "verified_torch_edit"
    assert report["candidate"] == candidate.digest == native.digest
    assert report["plan"] == plan.digest
    assert candidate.tensor_bytes("w") == candidate.tensor_bytes("tied") == native.tensor_bytes("w")
    assert f.source.tensor_bytes("w") == before
    child = report["child_report"]
    assert bytes.fromhex(child["steps"][-1]["values"]["w"]) == candidate.tensor_bytes("w")
    assert child["inference_mode"] is True and child["gradient_enabled"] is False
    assert child["training"] is False and child["torch_version"] == "2.9.1+cpu"
    assert report["kernel_peak_rss_bytes"] > 10000000 and report["kernel_cpu_ms"] > 0
    assert child["edit_allocations"]["explicit_tensor_allocation_bytes"] > 0
    assert child["edit_allocations"]["trace_tensor_bytes"] == len(before) * 2
    assert child["copies"]["zero_copy"] is False
    assert report["outputs_eligible"] is False and report["promotion_status"] == "not_evaluated"
    receipt = json.loads(candidate.receipt_json())
    assert receipt == report["native_verification"]
    assert receipt["after_buffers"] == receipt["before_buffers"]
    assert receipt["method"]["target_update"] == "direct_tensor_edit"
    assert all(
        not slot["requires_grad"] and slot["grad_is_none"] for slot in child["steps"][-1]["state"]
    )


@pytest.mark.parametrize("accumulation", ["f32", "f64"])
def test_torch_edit_backend_explicit_rounding_precision(
    runtime: tuple[Any, Path], accumulation: str
) -> None:
    f = make_fixture("f32")
    op = {
        "type": "scale",
        "target": {"tensor": "w", "start": [0, 1], "shape": [1, 1]},
        "factor": 1.00000007,
    }
    bind(f, op, "f32")
    f.program["constraints"]["numerical"]["accumulation_dtype"] = accumulation
    f.policy["backend"]["accumulation_dtypes"] = [accumulation]
    f.policy["backend"]["id"] = "nf.trusted_torch_cpu"
    plan = nf.ir.compile(f.source, f.program, f.space, f.policy)
    result = models.edit(runtime[0], plan, f.smoke)
    assert result.candidate.tensor_bytes("w") == nf.surgeon.apply(plan, f.smoke).tensor_bytes("w")
    assert (
        json.loads(result.receipt_bytes())["child_report"]["numerical"]["accumulation_dtype"]
        == accumulation
    )


def test_torch_edit_backend_requires_explicit_frozen_backend_and_live_plan(
    runtime: tuple[Any, Path],
) -> None:
    f = make_fixture()
    plan = nf.ir.compile(f.source, f.program, f.space, f.policy)
    with pytest.raises(nf.NativeError, match="explicitly authorize"):
        models.edit(runtime[0], plan, f.smoke)
    with pytest.raises(TypeError):
        _native.edit_model_worker(runtime[0], {}, wire(f.smoke), b"{}")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        _native.TorchEditHandle()


@pytest.mark.parametrize("action", ["backward", "optimizer_update", "train", "eval_python"])
def test_torch_edit_backend_rejects_training_or_expression_requests(
    runtime: tuple[Any, Path], action: str
) -> None:
    f = make_fixture()
    f.policy["backend"]["id"] = "nf.trusted_torch_cpu"
    f.program["operations"][0]["operation"]["type"] = action
    before = set(runtime[1].iterdir())
    with pytest.raises(nf.NativeError):
        nf.ir.compile(f.source, f.program, f.space, f.policy)
    assert set(runtime[1].iterdir()) == before


def test_torch_edit_backend_restored_outside_scope_attack_is_still_denied(
    runtime: tuple[Any, Path],
) -> None:
    f = make_fixture()
    op = {
        "type": "scale",
        "target": {"tensor": "w", "start": [0, 0], "shape": [1, 1]},
        "factor": 2.0,
    }
    bind(f, op, "f64")
    second = copy.deepcopy(f.program["operations"][0])
    second.update(id="op1", depends_on=["op0"])
    second["operation"]["factor"] = 0.5
    f.program["operations"].append(second)
    f.policy["backend"]["id"] = "nf.trusted_torch_cpu"
    plan = nf.ir.compile(f.source, f.program, f.space, f.policy)
    original = f.source.tensor_bytes("w")
    assert nf.surgeon.apply(plan, f.smoke).tensor_bytes("w") == original
    before = set(runtime[1].rglob("request.json"))
    # Fault injection into our controller-owned private staging, not a public
    # callback/command hook. Expand BOTH steps; final bytes still equal source.
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(models.edit, runtime[0], plan, f.smoke)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            created = set(runtime[1].rglob("request.json")) - before
            if created:
                request_path = created.pop()
                if request_path.with_name("process.json").exists():
                    value = json.loads(request_path.read_bytes())
                    for instruction in value["instructions"]:
                        instruction["operation"]["target"]["shape"] = [2, 2]
                    replacement = request_path.with_name("injected.json")
                    replacement.write_bytes(wire(value))
                    replacement.replace(request_path)
                    break
            time.sleep(0.002)
        else:
            raise AssertionError("actual trusted edit process never registered")
        with pytest.raises(nf.NativeError, match="outside its own"):
            pending.result(timeout=30)
    assert f.source.tensor_bytes("w") == original
    assert not request_path.with_name("completion.json").exists()
    assert (
        json.loads(request_path.with_name("execution.json").read_bytes())["cleanup_confirmed"]
        is True
    )


def test_torch_edit_backend_cancel_or_failed_smoke_admits_no_candidate(
    runtime: tuple[Any, Path],
) -> None:
    f, plan = editing("f64", "scale")
    token = nf.CancellationHandle()
    token.cancel()
    with pytest.raises(nf.CancelledError):
        models.edit(runtime[0], plan, f.smoke, cancellation=token)
    bad = copy.deepcopy(f.smoke)
    bad["probes"][0]["expected"] = [0.0, 0.0]
    with pytest.raises(nf.NativeError):
        models.edit(runtime[0], plan, bad)
