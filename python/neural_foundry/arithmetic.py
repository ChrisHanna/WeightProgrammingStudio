"""Independent arithmetic oracles and synthetic fixture-only gym composition.

This trusted reference module does not reuse ReLU weights or the native forward
implementation to compute expected answers. No result here accepts a model.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from importlib.resources import files
from os import PathLike
from pathlib import Path
from typing import Any

from . import _native
from .records import _wire, negotiate

HEADS = ("sum", "difference", "maximum", "minimum", "absolute_difference")
TOLERANCE = 1e-10


def _pair(x1: object, x2: object) -> tuple[float, float]:
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in (x1, x2)):
        raise ValueError("integer arithmetic input pair required")
    a, b = float(x1), float(x2)  # type: ignore[arg-type]
    if any(not math.isfinite(v) or v != int(v) or not -8 <= v <= 8 for v in (a, b)):
        raise ValueError("synthetic arithmetic domain is integer pairs in [-8,8]")
    return a, b


def expected(head: str, x1: object, x2: object) -> float:
    """Independent Python mathematics, not model/construction code."""
    a, b = _pair(x1, x2)
    if head == "sum":
        return a + b
    if head == "difference":
        return a - b
    if head == "maximum":
        return max(a, b)
    if head == "minimum":
        return min(a, b)
    if head == "absolute_difference":
        return abs(a - b)
    raise ValueError("unknown arithmetic head")


@dataclass(frozen=True)
class ArithmeticGym:
    head: str
    head_index: int
    name: str
    identity: str

    def score(self, x1: object, x2: object, output: object) -> float:
        truth = expected(self.head, x1, x2)
        if isinstance(output, bool) or not isinstance(output, (int, float)):
            return 0.0
        value = float(output)
        return float(math.isfinite(value) and abs(value - truth) <= TOLERANCE)


def builtins() -> tuple[ArithmeticGym, ...]:
    catalogue = describe(fixture("reference"))["gyms"]
    return tuple(
        ArithmeticGym(entry["head"], entry["head_index"], entry["name"], entry["id"])
        for entry in catalogue
    )


def fixture(mode: str = "corrupt_sum") -> _native.ArithmeticFixtureHandle:
    negotiate()
    if not hasattr(_native, "arithmetic_fixture"):
        raise NotImplementedError("Native arithmetic fixture adapter is unavailable")
    return _native.arithmetic_fixture(mode)


def describe(model: _native.ArithmeticFixtureHandle) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(model.description_bytes())
    return result


def evaluate(
    model: _native.ArithmeticFixtureHandle,
    source: _native.SourceHandle | None = None,
    *,
    split: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Trusted controller fixture audit, not a heldout-data access capability."""
    if split is not None and split not in {"development", "selection", "holdout"}:
        raise ValueError("exact canonical split role required")
    metadata = describe(model)
    cases = [c for c in metadata["inventory"]["cases"] if split is None or c["split"] == split]
    if not cases:
        raise ValueError("nonempty fixed inventory required")
    runtime = model.source if source is None else source
    observations = [
        (case, runtime.forward([float(case["x1"]), float(case["x2"])])) for case in cases
    ]
    results: dict[str, dict[str, Any]] = {}
    for gym in builtins():
        errors = [
            abs(output[gym.head_index] - expected(gym.head, case["x1"], case["x2"]))
            for case, output in observations
        ]
        correct = sum(error <= TOLERANCE for error in errors)
        results[gym.head] = {
            "name": gym.name,
            "count": len(cases),
            "correct": correct,
            "accuracy": correct / len(cases),
            "max_abs_error": max(errors),
            "tolerance": TOLERANCE,
            "scope": "synthetic_software_fixture_not_scientific_acceptance",
        }
    return results


def write_user_sum_package(
    directory: str | PathLike[str], model: _native.ArithmeticFixtureHandle
) -> tuple[Path, dict[str, Any]]:
    """Create new explicit files; only development tasks enter this user package."""
    target = Path(directory).absolute()
    target.mkdir(mode=0o700, parents=False, exist_ok=False)
    resource = files("neural_foundry").joinpath("data")
    data = [
        {"task": c["item"], "group": c["group"], "inputs": {"x1": c["x1"], "x2": c["x2"]}}
        for c in describe(model)["inventory"]["cases"]
        if c["split"] == "development"
    ]
    lock = {
        "schema_version": 1,
        "image": "sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c",
        "packages": {},
    }
    entry = {"file": "sum_gym.py", "symbol": "evaluate"}
    draft: dict[str, Any] = {
        "schema_version": 1,
        "name": "user.arithmetic.sum",
        "version": "1",
        "usage_rights": "Apache-2.0 synthetic fixture",
        "confidentiality": "development",
        "assets": {
            name: {"kind": kind, "expected": None}
            for name, kind in {
                "sum_gym.py": "code",
                "LICENSE": "license",
                "lock.json": "dependency_lock",
                "development.json": "data",
                "manifest.json": "manifest",
            }.items()
        },
        "dependency_lock": "lock.json",
        "licenses": ["LICENSE"],
        "interface": {
            "kind": "single_shot",
            "evaluate": entry,
            "oracle": {**entry, "symbol": "oracle"},
            "score": {**entry, "symbol": "score"},
            "input": {
                "type": "object",
                "properties": {
                    name: {"type": "integer", "min": -8, "max": 8} for name in ("x1", "x2")
                },
                "required": ["x1", "x2"],
            },
            "output": {"type": "number", "min": -16.0, "max": 16.0},
        },
        "dataset": {
            "kind": "frozen",
            "files": ["development.json"],
            "task_key": "task",
            "group_key": "group",
        },
    }
    assets = {
        "sum_gym.py": resource.joinpath("arithmetic_sum.py").read_bytes(),
        "LICENSE": resource.joinpath("ARITHMETIC_LICENSE.txt").read_bytes(),
        "lock.json": _wire(lock),
        "development.json": _wire({"schema_version": 1, "items": data}),
        "manifest.json": _wire(draft),
    }
    for name, contents in assets.items():
        with (target / name).open("xb") as output:
            output.write(contents)
    return target, draft


def mixed_portfolio(
    snapshot: _native.GymSnapshotHandle, approval: _native.GymApprovalHandle
) -> tuple[dict[str, Any], ...]:
    """Propose five exact fixture bindings; not a frozen/authorized StudyContract."""
    manifest = json.loads(snapshot.manifest_bytes())
    if (
        manifest["draft"]["name"] != "user.arithmetic.sum"
        or approval.snapshot_id != snapshot.snapshot_id
    ):
        raise ValueError("exact validated user sum snapshot and approval required")
    target = {
        "name": "user.arithmetic.sum",
        "head": "sum",
        "role": "target",
        "snapshot": snapshot.snapshot_id,
        "approval": approval.receipt_id,
        "required": True,
        "status": "proposed_binding_not_study_dispatch",
    }
    retention = tuple(
        {
            "name": gym.name,
            "gym": gym.identity,
            "head": gym.head,
            "role": "retention",
            "required": True,
            "max_regression": 0.0,
            "status": "proposed_binding_not_study_dispatch",
        }
        for gym in builtins()[1:]
    )
    return (target, *retention)
