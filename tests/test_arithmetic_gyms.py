"""Genuine native ReLU weights; independent Python arithmetic and real user gym."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from native_fixture import make_fixture, wire
from neural_foundry import NativeError, _native, arithmetic, gyms, ir, surgeon


def repair_plan(
    model: _native.ArithmeticFixtureHandle,
    harmful: bool = False,
    *,
    factor: float = 2.0,
    backend: str = "nf.native_cpu",
) -> tuple[Any, dict[str, Any]]:
    template = make_fixture()
    metadata = arithmetic.describe(model)
    manifest = json.loads(model.source.manifest_json())
    program = copy.deepcopy(template.program)
    space = copy.deepcopy(template.space)
    policy = copy.deepcopy(template.policy)
    base = {
        "checkpoint": metadata["checkpoint"],
        "manifest": model.source.manifest_digest,
        "parameter_digest": model.source.parameter_digest,
        "architecture_graph": metadata["graph"],
        "config": metadata["config"],
        "tokenizer": None,
    }
    smoke = {
        "schema_version": 1,
        "adapter": "nf.arithmetic_relu.v1",
        "probes": [
            {"input": pair, "expected": None, "absolute_tolerance": 0.0, "max_abs_output": 1000.0}
            for pair in [[0.0, 0.0], [-8.0, 8.0], [8.0, -8.0]]
        ],
    }
    verification = program["verification"]
    verification["smoke_suite"] = _native.validate_record("smoke_suite", wire(smoke)).digest
    for value in [program, space]:
        value["base_model"] = copy.deepcopy(base)
    regions = [{"tensor": "heads.weight", "start": [0, 0], "shape": [1, 1]}]
    if harmful:
        regions.append({"tensor": "heads.weight", "start": [1, 0], "shape": [1, 1]})
    space["sites"] = [
        {
            "selector": {"type": "tensor", "name": name},
            "role": "parameter",
            "regions": {"type": "whole_tensor"},
            "allowed_axes": [0, 1],
        }
        for name in manifest["parameters"]
    ]
    space["constraints"]["max_changed_parameters"] = len(regions)
    space_id = _native.validate_record("search_space", wire(space)).digest
    program["objective"]["search_space"] = space_id
    program["operations"] = [
        {
            "id": f"op{i}",
            "depends_on": [],
            "operation": {"type": "scale", "target": region, "factor": factor if i == 0 else 0.0},
        }
        for i, region in enumerate(regions)
    ]
    program["preconditions"] = [
        {
            "tensor": name,
            "role": "parameter",
            "shape": spec["shape"],
            "dtype": "f64",
            "before_digest": spec["digest"],
            "storage_group": spec["storage_group"],
        }
        for name, spec in manifest["parameters"].items()
    ]
    program["constraints"]["max_changed_parameters"] = len(regions)
    program["effects"]["reads"] = copy.deepcopy(regions)
    program["effects"]["writes"] = copy.deepcopy(regions)
    program["lineage"]["method"]["changed_parameters"] = len(regions)
    policy["backend"]["id"] = backend
    policy["source"] = copy.deepcopy(base)
    policy["verification"] = copy.deepcopy(verification)
    policy["search_space"] = space_id
    return ir.compile(model.source, program, space, policy, {}), smoke


def test_arithmetic_gyms_all_289_pairs_five_independent_oracles_and_actual_corruption() -> None:
    original = arithmetic.fixture()
    reference = arithmetic.fixture("reference")
    metadata = arithmetic.describe(original)
    assert metadata["parameter_count"] == 42 and metadata["dtype"] == "f64"
    assert metadata["promotion_status"] == "not_evaluated"
    assert len(arithmetic.builtins()) == 5
    assert len({gym.identity for gym in arithmetic.builtins()}) == 5
    for descriptor in metadata["gyms"]:
        body = {key: value for key, value in descriptor.items() if key != "id"}
        assert descriptor["id"] == _native.content_digest(_native.canonical_json(wire(body)))
        assert descriptor["dataset"] == metadata["inventory_id"]
        assert descriptor["oracle_code"]
    for case in metadata["inventory"]["cases"]:
        a, b = case["x1"], case["x2"]
        expected = [arithmetic.expected(head, a, b) for head in arithmetic.HEADS]
        actual = reference.source.forward([float(a), float(b)])
        assert actual == pytest.approx(expected, abs=1e-10, rel=0)
        corrupt = original.source.forward([float(a), float(b)])
        assert corrupt[1:] == pytest.approx(expected[1:], abs=1e-10, rel=0)
        assert corrupt[0] == pytest.approx(a + b - 0.5 * max(a, 0), abs=1e-10, rel=0)
    results = arithmetic.evaluate(original)
    assert results["sum"]["correct"] == 153 and results["sum"]["count"] == 289
    assert all(value["accuracy"] == 1.0 for head, value in results.items() if head != "sum")
    assert all(value["accuracy"] == 1.0 for value in arithmetic.evaluate(reference).values())


def test_arithmetic_gyms_actual_ir_repair_and_harmful_composition_preserve_source() -> None:
    model = arithmetic.fixture()
    before = model.source.parameter_digest
    checkpoint = model.checkpoint_bytes()
    baseline = arithmetic.evaluate(model)
    plan, smoke = repair_plan(model)
    candidate = surgeon.apply(plan, smoke)
    fixed = arithmetic.evaluate(model, candidate.snapshot())
    assert fixed["sum"]["accuracy"] == 1.0 > baseline["sum"]["accuracy"]
    assert all(
        fixed[head]["accuracy"] == baseline[head]["accuracy"] for head in arithmetic.HEADS[1:]
    )
    receipt = json.loads(candidate.receipt_json())
    assert receipt["promotion_status"] == "not_evaluated"
    bad_plan, smoke = repair_plan(model, harmful=True)
    harmful = surgeon.apply(bad_plan, smoke)
    bad = arithmetic.evaluate(model, harmful.snapshot())
    assert bad["sum"]["accuracy"] == 1.0
    assert bad["difference"]["accuracy"] < baseline["difference"]["accuracy"]
    assert any(bad[head]["accuracy"] < baseline[head]["accuracy"] for head in arithmetic.HEADS[1:])
    assert model.source.parameter_digest == before
    assert model.checkpoint_bytes() == checkpoint
    assert model.source.tensor_bytes("hidden.weight") == candidate.snapshot().tensor_bytes(
        "hidden.weight"
    )


def test_arithmetic_gyms_frozen_hash_splits_shared_mirror_groups_and_stable_versions() -> None:
    first = arithmetic.describe(arithmetic.fixture())
    second = arithmetic.describe(arithmetic.fixture("reference"))
    assert first["inventory_id"] == second["inventory_id"]
    assert first["inventory"] == second["inventory"]
    cases = first["inventory"]["cases"]
    by_pair = {(c["x1"], c["x2"]): c for c in cases}
    assert len(by_pair) == 289 and len({c["item"] for c in cases}) == 289
    roles = {
        role: {c["group"] for c in cases if c["split"] == role}
        for role in ["development", "selection", "holdout"]
    }
    assert all(roles.values())
    assert not roles["development"] & roles["selection"]
    assert not roles["development"] & roles["holdout"]
    assert not roles["selection"] & roles["holdout"]
    for (a, b), case in by_pair.items():
        assert case["group"] == by_pair[b, a]["group"]
        assert case["split"] == by_pair[b, a]["split"]
    for role in roles:
        assert (
            arithmetic.evaluate(arithmetic.fixture("reference"), split=role)["sum"]["accuracy"]
            == 1.0
        )


@pytest.mark.parametrize(
    "pair",
    [
        [9.0, 0.0],
        [-9.0, 0.0],
        [0.25, 1.0],
        [float("nan"), 0.0],
        [float("inf"), 0.0],
        [],
        [1.0],
        [1.0, 2.0, 3.0],
    ],
)
def test_arithmetic_gyms_native_domain_and_shape_rejection(pair: list[float]) -> None:
    with pytest.raises(NativeError):
        arithmetic.fixture().source.forward(pair)


def test_arithmetic_gyms_independent_score_invalids_boundaries_and_synthetic_method() -> None:
    for gym in arithmetic.builtins():
        for a, b in [(-8, -8), (-8, 8), (0, 0), (8, 8)]:
            value = arithmetic.expected(gym.head, a, b)
            assert gym.score(a, b, value) == 1.0
            assert gym.score(a, b, value + 1) == 0.0
            assert gym.score(a, b, None) == 0.0
            assert gym.score(a, b, True) == 0.0
            assert gym.score(a, b, float("nan")) == 0.0
    with pytest.raises(ValueError):
        arithmetic.expected("sum", True, 1)
    with pytest.raises(ValueError):
        arithmetic.expected("sum", 0.5, 1)
    with pytest.raises(ValueError):
        arithmetic.expected("sum", 9, 1)
    with pytest.raises(ValueError):
        arithmetic.expected("unknown", 1, 1)
    fixture = arithmetic.fixture()
    provenance = json.loads(fixture.asset_bytes("provenance"))
    assert provenance["origin"] == "hand_constructed_synthetic_fixture"
    assert provenance["known_target"] is True
    assert provenance["method"]["production_history"] == "direct_only"
    assert provenance["method"]["optimizer_steps"] == 0
    assert provenance["method"]["parameter_update_uses_backprop"] is False
    assert provenance["training_or_pretrained_source"] is None
    assert arithmetic.evaluate(arithmetic.fixture("empty_sum"))["sum"]["correct"] == 17


def test_arithmetic_gyms_graph_config_and_factory_fail_closed() -> None:
    model = arithmetic.fixture()
    graph = json.loads(model.asset_bytes("graph"))
    assert [node["op"] for node in graph["nodes"]] == ["input", "linear", "relu", "linear"]
    assert graph["nodes"][1]["weight"] == "hidden.weight"
    assert graph["nodes"][3]["weight"] == "heads.weight"
    manifest = json.loads(model.source.manifest_json())
    assets = {name: model.asset_bytes(name) for name in ["config", "graph", "provenance"]}
    assets["graph"] = _native.canonical_json(wire({**graph, "nodes": []}))
    manifest["assets"]["graph"] = _native.content_digest(assets["graph"])
    source = _native.load_source(wire(manifest), model.checkpoint_bytes(), assets)
    with pytest.raises(NativeError, match="graph"):
        source.forward([1.0, 2.0])
    with pytest.raises(NativeError):
        arithmetic.fixture("trained_or_downloaded")


def test_arithmetic_gyms_real_user_package_intake_validation_and_mixed_retention(
    tmp_path: Path,
) -> None:
    model = arithmetic.fixture()
    path, draft = arithmetic.write_user_sum_package(tmp_path / "user-sum", model)
    snapshot = gyms.import_local(path, draft, tmp_path / "store")
    data = json.loads(snapshot.asset_bytes("development.json"))["items"]
    expected_dev = {
        c["item"]
        for c in arithmetic.describe(model)["inventory"]["cases"]
        if c["split"] == "development"
    }
    assert {row["task"] for row in data} == expected_dev
    assert {row["group"] for row in data}.isdisjoint(
        {
            c["group"]
            for c in arithmetic.describe(model)["inventory"]["cases"]
            if c["split"] == "holdout"
        }
    )
    selected: dict[float, dict[str, Any]] = {}
    for row in data:
        value = arithmetic.expected("sum", row["inputs"]["x1"], row["inputs"]["x2"])
        selected.setdefault(value, row)
    cases = [
        {
            "id": row["task"],
            "input": row["inputs"],
            "expected": value,
            "wrong": [value + 1.0 if value < 16 else value - 1.0],
            "actions": [],
            "termination": None,
        }
        for value, row in list(selected.items())[:2]
    ]
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir(mode=0o700)
    plan = {
        "schema_version": 1,
        "snapshot": snapshot.snapshot_id,
        "truth_provenance": "independent Python direct arithmetic, not ReLU model construction",
        "cases": cases,
        "seeds": [17, 42],
        "scoring": {"minimum": 0.0, "maximum": 1.0, "good_min": 1.0, "bad_max": 0.0},
        "limits": {"wall_ms": 2000, "cpu_ms": 2000, "output_bytes": 32768},
    }
    validation = gyms.validate(snapshot, plan, sandbox, enabled=True)
    assert json.loads(validation.report_bytes())["status"] == "passed"
    approval = gyms.approve(snapshot, validation, "owner:synthetic-user-sum-v1")
    portfolio = arithmetic.mixed_portfolio(snapshot, approval)
    assert len(portfolio) == 5 and portfolio[0]["role"] == "target"
    assert portfolio[0]["snapshot"] == snapshot.snapshot_id
    assert {p["head"] for p in portfolio} == set(arithmetic.HEADS)
    assert all(p["required"] for p in portfolio)
    assert all(p["role"] == "retention" and p["max_regression"] == 0.0 for p in portfolio[1:])
    assert all(p["status"] == "proposed_binding_not_study_dispatch" for p in portfolio)
