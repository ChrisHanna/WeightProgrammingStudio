"""Independent actual CPU PyTorch references; no model downloads or fitting."""

import json
import struct
from typing import Any

import neural_foundry as nf
import pytest
import torch
from native_fixture import NativeFixture, make_fixture, wire

CASES = [
    "replace",
    "add",
    "scale",
    "mask",
    "permute0",
    "permute1",
    "map_left",
    "map_right",
    "project_left",
    "project_right",
    "low_rank",
]


def operand(
    fixture: NativeFixture, name: str, values: list[float], shape: list[int], dtype: str
) -> bytes:
    code = {"f32": "f", "f64": "d", "u8": "B"}[dtype]
    numbers = [int(value) for value in values] if dtype == "u8" else values
    data = struct.pack("<" + str(len(values)) + code, *numbers)
    header = wire(
        {"value": {"dtype": dtype.upper(), "shape": shape, "data_offsets": [0, len(data)]}}
    )
    header += b" " * (-len(header) % 8)
    checkpoint = struct.pack("<Q", len(header)) + header + data
    fixture.program["artifacts"][name] = {
        "digest": nf.content_digest(checkpoint),
        "tensor": "value",
        "dtype": dtype,
        "shape": shape,
        "data_bytes": len(data),
        "artifact_bytes": len(checkpoint),
        "access": "development",
        "provenance": {
            "schema_version": 1,
            "origin": "supplied_fact",
            "support": "untested",
            "provenance": "test fixture",
            "evidence": [],
            "access": "development",
        },
    }
    return checkpoint


def bind(fixture: NativeFixture, operation: dict[str, Any], dtype: str) -> None:
    fixture.program["operations"][0]["operation"] = operation
    regions = [{**operation["target"], "tensor": name} for name in fixture.manifest["parameters"]]
    fixture.program["effects"]["reads"] = regions
    fixture.program["effects"]["writes"] = regions
    fixture.program["constraints"]["max_artifact_bytes"] = 4096
    fixture.space["constraints"]["max_artifact_bytes"] = 4096
    kind = operation["type"]
    domain = {
        "dtypes": [dtype],
        "coefficient": {"min": -4.0, "max": 4.0}
        if kind in ("scale", "low_rank_update", "subspace_projection")
        else None,
        "rank": {"min": 1, "max": 2}
        if kind in ("low_rank_update", "subspace_projection")
        else None,
        "sides": ["left", "right"] if kind in ("linear_map", "subspace_projection") else [],
        "max_operand_bytes": 4096,
    }
    fixture.space["transformations"] = {kind: domain}
    identity = nf.ir.search_space(fixture.space).digest
    fixture.program["objective"]["search_space"] = identity
    fixture.policy["search_space"] = identity
    fixture.policy["backend"]["operations"] = [kind]


@pytest.mark.parametrize("dtype", ["f32", "f64"])
@pytest.mark.parametrize("case", CASES)
def test_cpu_differential_all_native_families_against_real_torch(dtype: str, case: str) -> None:
    assert torch.__version__ == "2.9.1+cpu"
    assert torch.version.cuda is None
    fixture = make_fixture(dtype)
    torch_dtype = torch.float32 if dtype == "f32" else torch.float64
    original = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch_dtype, device="cpu")
    target = {"tensor": "w", "start": [0, 0], "shape": [2, 2]}
    operands: dict[str, bytes] = {}
    operation: dict[str, Any]
    with torch.no_grad():
        if case in ("replace", "add"):
            delta = torch.tensor([[0.5, -1.0], [2.0, 0.25]], dtype=torch_dtype)
            operands["operand"] = operand(fixture, "operand", [0.5, -1.0, 2.0, 0.25], [2, 2], dtype)
            operation = {
                "type": "replace_slice" if case == "replace" else "add_delta",
                "target": target,
                "value" if case == "replace" else "delta": "operand",
            }
            expected = delta if case == "replace" else original + delta
        elif case == "scale":
            operation = {"type": "scale", "target": target, "factor": -0.5}
            expected = original * -0.5
        elif case == "mask":
            operands["mask"] = operand(fixture, "mask", [0, 1, 1, 0], [2, 2], "u8")
            operation = {"type": "mask", "target": target, "mask": "mask"}
            expected = original * torch.tensor([[0, 1], [1, 0]], dtype=torch_dtype)
        elif case.startswith("permute"):
            axis = int(case[-1])
            operation = {
                "type": "permute_axis",
                "target": target,
                "axis": axis,
                "permutation": [1, 0],
            }
            expected = torch.index_select(original, axis, torch.tensor([1, 0]))
        elif case.startswith("map"):
            matrix = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch_dtype)
            operands["matrix"] = operand(fixture, "matrix", [0, 1, -1, 0], [2, 2], dtype)
            side = "left" if case.endswith("left") else "right"
            operation = {"type": "linear_map", "target": target, "matrix": "matrix", "side": side}
            expected = matrix @ original if side == "left" else original @ matrix
        elif case.startswith("project"):
            basis = torch.tensor([[1.0], [0.0]], dtype=torch_dtype)
            operands["basis"] = operand(fixture, "basis", [1, 0], [2, 1], dtype)
            side = "left" if case.endswith("left") else "right"
            operation = {
                "type": "subspace_projection",
                "target": target,
                "basis": "basis",
                "side": side,
                "strength": 0.5,
                "orthonormal_tolerance": 1e-6,
            }
            projected = basis @ basis.T @ original if side == "left" else original @ basis @ basis.T
            expected = original - 0.5 * projected
        else:
            operands["left"] = operand(fixture, "left", [1, 2], [2, 1], dtype)
            operands["right"] = operand(fixture, "right", [2, -1], [1, 2], dtype)
            operation = {
                "type": "low_rank_update",
                "target": target,
                "left": "left",
                "right": "right",
                "strength": 0.25,
            }
            expected = original + 0.25 * (
                torch.tensor([[1.0], [2.0]], dtype=torch_dtype)
                @ torch.tensor([[2.0, -1.0]], dtype=torch_dtype)
            )
    bind(fixture, operation, dtype)
    plan = nf.ir.compile(fixture.source, fixture.program, fixture.space, fixture.policy, operands)
    candidate = nf.surgeon.apply(plan, fixture.smoke)
    values = struct.unpack("<4f" if dtype == "f32" else "<4d", candidate.tensor_bytes("w"))
    actual = torch.tensor(values, dtype=torch_dtype).reshape(2, 2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6 if dtype == "f32" else 1e-12)
    assert candidate.tensor_bytes("tied") == candidate.tensor_bytes("w")
    assert not original.requires_grad and not actual.requires_grad and not expected.requires_grad
    assert struct.unpack("<4f" if dtype == "f32" else "<4d", fixture.source.tensor_bytes("w")) == (
        1,
        2,
        3,
        4,
    )
    assert json.loads(candidate.receipt_json())["promotion_status"] == "not_evaluated"


@pytest.mark.parametrize("dtype", ["f32", "f64"])
def test_cpu_differential_every_small_rectangle_preserves_other_coordinates(dtype: str) -> None:
    fixture = make_fixture(dtype)
    for row in range(2):
        for col in range(2):
            for rows in range(1, 3 - row):
                for cols in range(1, 3 - col):
                    operation = {
                        "type": "scale",
                        "target": {"tensor": "w", "start": [row, col], "shape": [rows, cols]},
                        "factor": -1.0,
                    }
                    bind(fixture, operation, dtype)
                    candidate = nf.surgeon.apply(
                        nf.ir.compile(
                            fixture.source, fixture.program, fixture.space, fixture.policy
                        ),
                        fixture.smoke,
                    )
                    expected = torch.tensor(
                        [[1.0, 2.0], [3.0, 4.0]],
                        dtype=torch.float32 if dtype == "f32" else torch.float64,
                    )
                    expected[row : row + rows, col : col + cols] *= -1
                    actual = torch.tensor(
                        struct.unpack(
                            "<4f" if dtype == "f32" else "<4d", candidate.tensor_bytes("w")
                        ),
                        dtype=expected.dtype,
                    ).reshape(2, 2)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
