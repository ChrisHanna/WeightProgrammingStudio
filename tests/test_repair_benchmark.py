"""Bounded real CPU comparisons; no training, downloads or acceptance claims."""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from neural_foundry import _native
from neural_foundry import repair_benchmark as bench


@pytest.fixture(scope="module")
def bundle() -> dict[str, Any]:
    return bench.run_fixture()


def test_repair_benchmark_actual_native_comparisons_and_failed_direct(bundle: dict[str, Any]) -> None:
    task, result = bundle["task"], bundle["result"]
    arms = {a["id"]: a for a in result["attempts"]}
    assert task["failure"]["kind"] == "injected_known"
    assert "known target" in task["failure"]["solution_exposure"]
    assert result["task"] == bench._digest(task)
    assert result["evaluation_split"] == "development"
    assert result["scope"] == bench.SCOPE
    assert result["workflow_linkage"] == "unavailable_pending_authorized_workflow"
    assert result["efficiency_claim"] == "not_ranked"
    assert result["budget_verdict"] == "incomplete"
    assert {a["method"] for a in arms.values()} == set(bench.METHODS)
    assert arms["original"]["scores"] == arms["no_op"]["scores"]
    assert arms["original"]["candidate"] == arms["no_op"]["candidate"] == task["source"]
    assert arms["direct_failed"]["status"] == "failed"
    assert arms["direct_failed"]["candidate"] != task["source"]
    assert arms["direct"]["status"] == "completed"
    assert arms["original"]["scores"]["sum"]["correct"] < arms["direct"]["scores"]["sum"]["correct"]
    assert all(s["correct"] == s["count"] for s in arms["direct"]["scores"].values())
    assert arms["direct"]["program"] != arms["direct_failed"]["program"]
    for identity, document in bundle["evidence_documents"].items():
        assert bench._digest(document) == identity
    for arm in arms.values():
        for identity in arm["evidence"].values():
            assert identity in bundle["evidence_documents"]
    documents = bundle["evidence_documents"]
    for name in ("no_op", "direct_failed", "direct"):
        arm = arms[name]
        receipt = documents[arm["evidence"]["apply_receipt"]]
        assert receipt["plan"] == arm["program"]
        assert arm["program"] in documents
        assert receipt["before_parameters"] == task["source"]
        assert receipt["after_parameters"] == arm["candidate"]
        assert receipt["changed_parameters"] == (0 if name == "no_op" else 1)
        assert receipt["method"]["target_update"] == (
            "no_change" if name == "no_op" else "direct_tensor_edit"
        )
        assert receipt["method"]["proposal_origin"] == "scripted"
        assert receipt["promotion_status"] == "not_evaluated"
    # Recompute every score from retained real native outputs, using explicit
    # arithmetic rather than trusting the runner's aggregate or helper oracle.
    for arm in arms.values():
        if not arm["scores"]:
            continue
        rows = documents[arm["evidence"]["observations"]]["rows"]
        correct = dict.fromkeys(arm["scores"], 0)
        for row in rows:
            x, y = row["input"]
            expected = (x + y, x - y, max(x, y), min(x, y), abs(x - y))
            for head, actual, want in zip(
                ("sum", "difference", "maximum", "minimum", "absolute_difference"),
                row["output"],
                expected,
                strict=True,
            ):
                correct[head] += abs(actual - want) <= 1e-10
        assert arm["scores"] == {
            head: {"correct": count, "count": len(rows)} for head, count in correct.items()
        }
    for head in ("difference", "maximum", "minimum", "absolute_difference"):
        assert arms["original"]["scores"][head] == arms["direct"]["scores"][head]
    for method in bench.METHODS[4:]:
        arm = arms[method]
        assert arm["status"] == (
            "unsupported" if method in {"decoding", "activation"} else "not_run"
        )
        assert arm["scores"] == {} and arm["costs"] == [] and arm["candidate"] is None
    for arm in arms.values():
        assert arm["localization"] == arm["mechanism"] == "not_evaluated"
        assert not arm["gradient_inspection"] and not arm["gradient_update"]


def test_repair_benchmark_accounting_unknowns_and_units(bundle: dict[str, Any]) -> None:
    result = bundle["result"]
    costs = result["shared_costs"] + [c for a in result["attempts"] for c in a["costs"]]
    assert len({c["operation_id"] for c in costs}) == len(costs)
    for cost in costs:
        assert set(cost["quantities"]) == set(bench.COST_UNITS)
        for name, quantity in cost["quantities"].items():
            assert quantity["unit"] == bench.COST_UNITS[name]
            assert quantity["reason"]
            if quantity["status"] in {"unknown", "unavailable"}:
                assert quantity["value"] is None
        assert cost["quantities"]["host_peak"]["status"] == "unknown"
    for arm in result["attempts"]:
        if arm["costs"]:
            quantities = arm["costs"][0]["quantities"]
            assert quantities["wall"]["value"] > 0
            assert quantities["cpu"]["value"] > 0
            assert quantities["forwards"]["value"] == 109 + (3 if arm["program"] else 0)
            assert quantities["backwards"]["value"] == quantities["optimizer_steps"]["value"] == 0
    assert result["shared_costs"][0]["quantities"]["forwards"]["value"] is None
    assert result["phase_coverage"]["reload"] == "not_run"
    assert result["phase_coverage"]["export"] == "not_run"


def test_repair_benchmark_native_identity_and_registration(
    bundle: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    task, result = bundle["task"], bundle["result"]
    assert _native.validate_repair_benchmark(bench._wire(task)) == _native.canonical_json(
        bench._wire(task)
    )
    checked = bench.validate(task, result)
    assert checked == _native.canonical_json(bench._wire(result))
    decoded = json.loads(checked)
    # The existing nf-core JSON codec may round measured binary floats by an
    # ULP. Canonical bytes above must match exactly; observations/methods below
    # match exactly, with clock quantities compared at sub-nanosecond tolerance.
    for before, after in zip(result["attempts"], decoded["attempts"], strict=True):
        for left, right in zip(before["costs"], after["costs"], strict=True):
            for name in ("wall", "cpu"):
                assert right["quantities"][name]["value"] == pytest.approx(
                    left["quantities"][name]["value"], abs=1e-12, rel=1e-12
                )
                right["quantities"][name]["value"] = left["quantities"][name]["value"]
    for left, right in zip(result["shared_costs"], decoded["shared_costs"], strict=True):
        for name in ("wall", "cpu"):
            assert right["quantities"][name]["value"] == pytest.approx(
                left["quantities"][name]["value"], abs=1e-12, rel=1e-12
            )
            right["quantities"][name]["value"] = left["quantities"][name]["value"]
    assert decoded == result
    # Integer budgets stay integer in the canonical return, preserving the
    # exact task identity used by all attempt records (no f64 reserialization).
    assert _native.content_digest(bench.validate(task)) == result["task"]
    monkeypatch.delattr(_native, "validate_repair_benchmark")
    with pytest.raises(NotImplementedError, match="registration unavailable"):
        bench.validate(task, result)


@pytest.mark.parametrize("field", ["observations", "source_integrity", "apply_receipt"])
def test_repair_benchmark_native_required_evidence(bundle: dict[str, Any], field: str) -> None:
    bad = copy.deepcopy(bundle["result"])
    del bad["attempts"][1]["evidence"][field]
    with pytest.raises(_native.NativeError) as error:
        bench.validate(bundle["task"], bad)
    assert error.value.code == "INVALID_VALUE"
    assert error.value.path == "repair_benchmark"
    assert "evidence" in error.value.message


@pytest.mark.parametrize("mutation", ["method", "original", "direct", "unrun", "holdout"])
def test_repair_benchmark_native_method_identity_and_scope_reject(
    bundle: dict[str, Any], mutation: str
) -> None:
    bad = copy.deepcopy(bundle["result"])
    arms = {a["id"]: a for a in bad["attempts"]}
    if mutation == "method":
        arms["direct"]["method"] = "sft"
    elif mutation == "original":
        arms["original"]["candidate"] = arms["direct"]["candidate"]
    elif mutation == "direct":
        arms["direct"]["candidate"] = bundle["task"]["source"]
    elif mutation == "unrun":
        arms["lora"]["costs"] = arms["direct"]["costs"]
    else:
        bad["evaluation_split"] = "holdout"
    with pytest.raises(_native.NativeError):
        bench.validate(bundle["task"], bad)


def test_repair_benchmark_native_wire_errors_and_stale_task(bundle: dict[str, Any]) -> None:
    task = bench._wire(bundle["task"])
    for malformed in (b'{"schema_version":1,"schema_version":1}', b'{"x":NaN}', b"{}"):
        with pytest.raises(_native.NativeError):
            _native.validate_repair_benchmark(malformed)
    with pytest.raises(_native.NativeError, match="1 MiB"):
        _native.validate_repair_benchmark(b" " * (1048576 + 1))
    with pytest.raises(TypeError):
        _native.validate_repair_benchmark("not bytes")  # type: ignore[arg-type]
    changed = copy.deepcopy(bundle["task"])
    changed["seed"] += 1
    with pytest.raises(_native.NativeError, match="binding"):
        bench.validate(changed, bundle["result"])
    assert json.loads(_native.validate_repair_benchmark(task)) == bundle["task"]


def test_repair_benchmark_missing_or_fake_comparison_plans_fail() -> None:
    with pytest.raises(ValueError, match="exact three"):
        bench.run_fixture(lambda _: {})

    def wrong_noop(model: Any) -> dict[str, Any]:
        plans = dict(bench._prepare(model))
        plans["no_op"], plans["direct"] = plans["direct"], plans["no_op"]
        return plans

    with pytest.raises(ValueError, match="no-op"):
        bench.run_fixture(wrong_noop)

    def fake_failure(model: Any) -> dict[str, Any]:
        plans = dict(bench._prepare(model))
        plans["direct_failed"], plans["direct"] = plans["direct"], plans["direct_failed"]
        return plans

    with pytest.raises(ValueError, match="actually fail"):
        bench.run_fixture(fake_failure)


def test_repair_benchmark_actual_records_checked_by_rust_library(bundle: dict[str, Any]) -> None:
    root = Path(__file__).resolve().parents[1]
    cargo = shutil.which("cargo") or str(Path.home() / ".cargo/bin/cargo")
    records = {key: bundle[key] for key in ("task", "result")}
    env = dict(
        os.environ,
        CARGO_TARGET_DIR="/tmp/nf-t044-complete",
        NF_REPAIR_BENCH_RECORD=json.dumps(records, allow_nan=False),
    )
    env["PATH"] = str(Path(cargo).parent) + os.pathsep + env["PATH"]
    command = [
        cargo,
        "test",
        "--offline",
        "-j",
        "1",
        "-p",
        "nf-experiments",
        "--test",
        "repair_benchmark",
        "repair_benchmark_actual_python_record_when_supplied",
        "--",
        "--exact",
    ]
    completed = subprocess.run(
        command, cwd=root, env=env, text=True, capture_output=True, timeout=120, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "1 passed" in completed.stdout
    # An authentic fixture result with a missing required observation also fails
    # the authoritative checker; no Python-only schema acceptance substitute.
    bad = copy.deepcopy(records)
    del bad["result"]["attempts"][0]["evidence"]["observations"]
    env["NF_REPAIR_BENCH_RECORD"] = json.dumps(bad, allow_nan=False)
    rejected = subprocess.run(
        command, cwd=root, env=env, text=True, capture_output=True, timeout=30, check=False
    )
    assert rejected.returncode != 0
    assert "observation/source evidence" in rejected.stdout + rejected.stderr
