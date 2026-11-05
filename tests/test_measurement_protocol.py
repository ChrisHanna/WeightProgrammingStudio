"""Real CPU timer/RSS samples; native controlled-clock cases test gate math separately."""

import copy
import json
import statistics
from typing import Any

import neural_foundry as nf
import pytest
from native_fixture import wire
from neural_foundry import _native, arithmetic, measurements, models


def identity(value: Any) -> str:
    return nf.content_digest(_native.canonical_json(wire(value)))


@pytest.fixture(scope="module")
def runtime(tmp_path_factory: pytest.TempPathFactory) -> Any:
    return models.qualify(tmp_path_factory.mktemp("measurement-runtime"), enabled=True)


def protocol(runtime: Any) -> dict[str, Any]:
    p = {
        "schema_version": 1,
        "environment": measurements.environment(runtime),
        "workload": identity("placeholder"),
        "inputs": [[8.0, -8.0], [-4.0, 3.0]],
        "seeds": [17, 42],
        "batch_size": 1,
        "concurrency": 1,
        "precision": "f64",
        "cache": "disabled",
        "state_reset": "stateless_checked_before_after",
        "warmup": 2,
        "repetitions": 7,
        "order": "paired_interleaved",
        "order_seed": 19,
        "metrics": [
            {
                "name": "forward",
                "scope": "forward_latency",
                "unit": "nanoseconds",
                "statistic": "median",
                "required": True,
                "ceiling": 1000000000.0,
                "max_regression": None,
                "min_gain": None,
            },
            {
                "name": "request",
                "scope": "end_to_end_latency",
                "unit": "milliseconds",
                "statistic": "p95",
                "required": True,
                "ceiling": 1000.0,
                "max_regression": None,
                "min_gain": None,
            },
            {
                "name": "rss",
                "scope": "process_rss",
                "unit": "bytes",
                "statistic": "maximum",
                "required": True,
                "ceiling": 1073741824.0,
                "max_regression": None,
                "min_gain": None,
            },
            {
                "name": "allocator",
                "scope": "device_allocator_peak",
                "unit": "bytes",
                "statistic": "maximum",
                "required": False,
                "ceiling": None,
                "max_regression": None,
                "min_gain": None,
            },
            {
                "name": "device",
                "scope": "device_peak",
                "unit": "bytes",
                "statistic": "maximum",
                "required": False,
                "ceiling": None,
                "max_regression": None,
                "min_gain": None,
            },
        ],
    }
    refresh(p)
    return p


def refresh(p: dict[str, Any]) -> None:
    p["workload"] = identity(
        [
            p[k]
            for k in [
                "inputs",
                "seeds",
                "batch_size",
                "concurrency",
                "precision",
                "cache",
                "state_reset",
            ]
        ]
    )


def test_measurement_protocol_real_cpu_warmup_interleaving_units_statistics_and_rss(
    runtime: Any,
) -> None:
    original = arithmetic.fixture("reference").source
    candidate = arithmetic.fixture("corrupt_sum").source
    before = original.parameter_digest, candidate.parameter_digest
    p = protocol(runtime)
    witness = measurements.run(runtime, original, candidate, p)
    report = json.loads(witness.report_bytes())
    assert witness.receipt_id == nf.content_digest(witness.report_bytes())
    assert report["measurement_conditions_satisfied"] and not report["scientific_acceptance"]
    assert not report["identical_model"] and not report["outputs_eligible"]
    child = report["child_report"]
    assert child["clock"] == "linux_clock_monotonic_raw_ns" and child["clock_resolution_ns"] > 0
    assert child["before"] == child["after"] and child["inference_mode"]
    assert not child["gradient_enabled"] and child["training"] == [False, False]
    assert len(child["samples"]) == 18
    assert [s["arm"] for s in child["samples"][:8]] == [0, 1, 1, 0, 0, 1, 1, 0]
    assert sum(s["warmup"] for s in child["samples"]) == 4
    for arm in [0, 1]:
        rows = [s for s in child["samples"] if s["arm"] == arm and not s["warmup"]]
        assert len(rows) == 7
        assert all(
            0 < s["forward_ns"] <= s["end_to_end_ns"] and s["process_rss_bytes"] > 10000000
            for s in rows
        )
        column = "original" if arm == 0 else "candidate"
        assert report["metrics"][0][column] == statistics.median(s["forward_ns"] for s in rows)
        assert report["metrics"][1][column] == max(s["end_to_end_ns"] for s in rows) / 1000000.0
        assert report["metrics"][2][column] == max(s["process_rss_bytes"] for s in rows)
    assert child["device_allocator_peak_bytes"] is None and child["device_peak_bytes"] is None
    assert all(
        m["status"] == "unavailable" and m["candidate"] is None for m in report["metrics"][3:]
    )
    assert "not_per_model" in child["memory_scope"]
    assert report["kernel_cpu_ms"] > 0 and report["kernel_peak_rss_bytes"] > 10000000
    assert (original.parameter_digest, candidate.parameter_digest) == before
    with pytest.raises(TypeError):
        _native.MeasurementHandle()


def test_measurement_protocol_required_device_unavailable_and_noop_noise_cannot_pass(
    runtime: Any,
) -> None:
    model = arithmetic.fixture().source
    p = protocol(runtime)
    p["order"] = "randomized_interleaved"
    p["metrics"][0]["min_gain"] = 0.000000001
    p["metrics"][3].update(required=True, ceiling=1000000.0)
    report = json.loads(measurements.run(runtime, model, model, p).report_bytes())
    assert report["identical_model"] and not report["measurement_conditions_satisfied"]
    assert report["metrics"][0]["status"] == "no_op_cannot_satisfy_positive_gain"
    assert report["metrics"][3]["status"] == "unavailable"
    schedule = report["child_report"]["samples"]
    for i in range(0, len(schedule), 2):
        assert {schedule[i]["arm"], schedule[i + 1]["arm"]} == {0, 1}


def test_measurement_protocol_co_resident_rss_cannot_claim_individual_memory_improvement(
    runtime: Any,
) -> None:
    p = protocol(runtime)
    p["metrics"][2]["min_gain"] = 1.0
    report = json.loads(
        measurements.run(
            runtime, arithmetic.fixture().source, arithmetic.fixture("reference").source, p
        ).report_bytes()
    )
    assert report["metrics"][2]["status"] == "inconclusive_co_resident_memory_attribution"
    assert not report["measurement_conditions_satisfied"]


@pytest.mark.parametrize(
    "mutation",
    [
        "driver",
        "runtime",
        "physical_device",
        "affinity",
        "clock",
        "precision",
        "batch",
        "concurrency",
        "cache",
        "reset",
        "workload",
        "unit",
        "statistic",
        "warmup",
        "repetitions",
        "seeds",
        "empty",
        "duplicate",
        "threshold",
        "scope",
    ],
)
def test_measurement_protocol_exact_environment_workload_and_threshold_binding(
    runtime: Any, mutation: str
) -> None:
    model = arithmetic.fixture().source
    p = protocol(runtime)
    if mutation in {"driver", "runtime", "physical_device"}:
        p["environment"][mutation] = identity("wrong")
    if mutation == "affinity":
        p["environment"]["affinity"] = "999"
    if mutation == "clock":
        p["environment"]["clock"] = "uncontrolled_wall_clock"
    if mutation == "precision":
        p["precision"] = "f32"
        refresh(p)
    if mutation == "batch":
        p["batch_size"] = 2
        refresh(p)
    if mutation == "concurrency":
        p["concurrency"] = 2
        refresh(p)
    if mutation == "cache":
        p["cache"] = "warm"
        refresh(p)
    if mutation == "reset":
        p["state_reset"] = "unchecked_shared_agent_state"
        refresh(p)
    if mutation == "workload":
        p["inputs"][0] = [1.0, 2.0]
    if mutation == "unit":
        p["metrics"][2]["unit"] = "milliseconds"
    if mutation == "statistic":
        p["metrics"][0]["statistic"] = "exact"
    if mutation == "warmup":
        p["warmup"] = 33
    if mutation == "repetitions":
        p["repetitions"] = 0
    if mutation == "seeds":
        p["seeds"] = []
        refresh(p)
    if mutation == "empty":
        p["metrics"] = []
    if mutation == "duplicate":
        p["metrics"].append(copy.deepcopy(p["metrics"][0]))
    if mutation == "threshold":
        p["metrics"][0]["ceiling"] = float("nan")
    if mutation == "scope":
        p["metrics"][0]["scope"] = "whole_study_as_forward_proxy"
    with pytest.raises(nf.NativeError):
        measurements.run(runtime, model, model, p)


def test_measurement_protocol_metadata_only_changes_cannot_qualify_timing_gain(
    runtime: Any,
) -> None:
    model = arithmetic.fixture()
    manifest = json.loads(model.source.manifest_json())
    assets = {name: model.asset_bytes(name) for name in ["config", "graph", "provenance"]}
    assets["provenance"] += b"\n"
    manifest["assets"]["provenance"] = nf.content_digest(assets["provenance"])
    for spec in manifest["parameters"].values():
        spec["storage_group"] += ".renamed"
    candidate = _native.load_source(wire(manifest), model.checkpoint_bytes(), assets)
    assert candidate.manifest_digest != model.source.manifest_digest
    p = protocol(runtime)
    p["metrics"][0]["min_gain"] = 0.000000001
    report = json.loads(measurements.run(runtime, model.source, candidate, p).report_bytes())
    assert report["identical_model"]
    assert report["metrics"][0]["status"] == "no_op_cannot_satisfy_positive_gain"
    assert not report["measurement_conditions_satisfied"]


def test_measurement_protocol_real_failure_and_cancel_never_mint_measurement(runtime: Any) -> None:
    model = arithmetic.fixture().source
    p = protocol(runtime)
    with pytest.raises(nf.NativeError, match="failed or exceeded"):
        measurements.run(runtime, model, model, p, wall_ms=100)
    token = nf.CancellationHandle()
    token.cancel()
    with pytest.raises(nf.CancelledError):
        measurements.run(runtime, model, model, p, cancellation=token)
