"""Independent numerical references + actual finite native arithmetic outcome integration."""

import copy
import json
import math
from typing import Any

import neural_foundry as nf
import pytest
from native_fixture import wire
from neural_foundry import _native, arithmetic, statistics, surgeon
from test_arithmetic_gyms import repair_plan


def identity(value: Any) -> str:
    return nf.content_digest(_native.canonical_json(wire(value)))


def protocol(exact: bool = False, groups: int = 4, comparisons: int = 1) -> dict[str, Any]:
    contrasts = []
    for i in range(comparisons):
        tasks = [
            {
                "item": identity(["task", group, j]),
                "group": identity(["group", group]),
                "input": identity(["input", group, j]),
            }
            for group in range(groups)
            for j in range(2)
        ]
        contrasts.append(
            {
                "id": identity(["contrast", i]),
                "candidate": identity(["candidate", i]),
                "baseline": identity("original"),
                "comparator": "original",
                "binding": identity(["gym", i]),
                "metric": "accuracy",
                "schema": {
                    "domain": "quality",
                    "direction": "higher_is_better",
                    "unit": "fraction",
                    "lower": 0.0,
                    "upper": 1.0,
                },
                "value_statistic": "exact",
                "context": identity("context"),
                "workload": identity("workload"),
                "measurement_protocol": identity("outcome-protocol"),
                "tasks": tasks,
                "threshold": {
                    "floor": None,
                    "ceiling": None,
                    "min_gain": 0.1,
                    "max_regression": None,
                },
                "required": True,
                "no_op_control": False,
            }
        )
    return {
        "schema_version": 1,
        "family": identity("frozen-comparison-family"),
        "estimator": "deterministic_exact" if exact else "paired_group_bootstrap",
        "scope": "deterministic_fixture_only" if exact else "paired_empirical_analysis",
        "estimand": "equal_weight_group_mean_of_task_seed_means",
        "minimum_groups": 2,
        "minimum_tasks": 2,
        "seeds": [17, 42],
        "confidence": 1.0 if exact else 0.95,
        "resamples": 0 if exact else 4096,
        "bootstrap_seed": 19,
        "maximum_comparisons": comparisons,
        "contrasts": contrasts,
    }


def data(plan: Any, p: dict[str, Any], gains: list[float] | None = None) -> dict[str, Any]:
    rows = []
    for contrast in p["contrasts"]:
        group_order = sorted({t["group"] for t in contrast["tasks"]})
        for task in contrast["tasks"]:
            gain = 0.25 if gains is None else gains[group_order.index(task["group"]) % len(gains)]
            for seed in p["seeds"]:
                for arm in ["baseline", "candidate"]:
                    rows.append(
                        {
                            "contrast": contrast["id"],
                            **task,
                            "seed": seed,
                            "arm": arm,
                            "model": contrast["baseline"]
                            if arm == "baseline"
                            else contrast["candidate"],
                            "context": contrast["context"],
                            "workload": contrast["workload"],
                            "measurement_protocol": contrast["measurement_protocol"],
                            "unit": contrast["schema"]["unit"],
                            "value_statistic": contrast["value_statistic"],
                            "status": "complete",
                            "value": 0.5 if arm == "baseline" else 0.5 + gain,
                        }
                    )
    return {"schema_version": 1, "plan": plan.plan_id, "observations": rows}


def report(plan: Any, observations: dict[str, Any]) -> dict[str, Any]:
    result = statistics.analyze(plan, observations)
    assert result.analysis_id == nf.content_digest(result.report_bytes())
    value: dict[str, Any] = json.loads(result.report_bytes())
    assert not value["scientific_acceptance"] and not value["outputs_eligible"]
    return value


@pytest.mark.parametrize("exact", [False, True])
def test_paired_statistics_positive_and_negative_complete_frozen_panels(exact: bool) -> None:
    p = protocol(exact)
    plan = statistics.plan(p)
    raw = data(plan, p)
    result = report(plan, raw)
    row = result["results"][0]
    assert row["status"] == "passed" and row["interval"] == [0.25, 0.25]
    assert row["estimate"]["gain"] == 0.25 and row["positive_gain_eligible"]
    assert row["unique_groups"] == 4 and row["unique_tasks"] == 8 and row["complete_pairs"] == 16
    assert report(plan, data(plan, p, [-0.25]))["results"][0]["status"] == "failed"
    assert (
        statistics.analyze(plan, raw).report_bytes() == statistics.analyze(plan, raw).report_bytes()
    )
    p["contrasts"][0]["threshold"]["min_gain"] = 0.5
    assert (
        json.loads(plan.record_bytes())["protocol"]["contrasts"][0]["threshold"]["min_gain"] == 0.1
    )
    with pytest.raises(TypeError):
        _native.StatisticalPlanHandle()
    with pytest.raises(TypeError):
        _native.StatisticalAnalysisHandle()


def independent_bootstrap(p: dict[str, Any], gains: list[float]) -> list[float]:
    c = p["contrasts"][0]
    state = int(
        identity(["nf.paired_group_rng.v1", p["bootstrap_seed"], p["family"], c["id"]])[7:23], 16
    )
    state = state or 0x9E3779B97F4A7C15
    mask = (1 << 64) - 1
    zone = mask - mask % len(gains)
    samples = []
    for _ in range(p["resamples"]):
        total = 0.0
        for _ in gains:
            while True:
                state ^= (state << 13) & mask
                state ^= state >> 7
                state ^= (state << 17) & mask
                if state < zone:
                    break
            total += gains[state % len(gains)]
        samples.append(total / len(gains))
    samples.sort()
    tail = (1 - p["confidence"]) / (2 * p["maximum_comparisons"])
    return [
        samples[math.floor(tail * (len(samples) - 1))],
        samples[math.ceil((1 - tail) * (len(samples) - 1))],
    ]


def test_paired_statistics_matches_independent_group_resampling_reference() -> None:
    p = protocol()
    gains = [0.125, 0.25, 0.375, -0.125]
    plan = statistics.plan(p)
    row = report(plan, data(plan, p, gains))["results"][0]
    assert row["interval"] == independent_bootstrap(p, gains)
    assert row["status"] == "inconclusive"
    assert row["resamples"] == p["resamples"]


def test_paired_statistics_multiplicity_covers_metrics_candidates_and_baseline_contrasts() -> None:
    p = protocol(comparisons=3)
    p["maximum_comparisons"] = 12
    p["contrasts"][1]["comparator"] = "no_op"
    p["contrasts"][1]["baseline"] = identity("no-op-baseline")
    p["contrasts"][2]["comparator"] = "best_known"
    p["contrasts"][2]["baseline"] = identity("best-known-baseline")
    plan = statistics.plan(p)
    raw = data(plan, p)
    raw["observations"] = [
        o for o in raw["observations"] if o["contrast"] != p["contrasts"][2]["id"]
    ]
    result = report(plan, raw)
    assert result["per_comparison_two_sided_alpha"] == pytest.approx(0.05 / 12)
    assert result["maximum_planned_comparisons"] == 12 and result["frozen_contrasts"] == 3
    assert result["results"][2]["status"] == "incomplete"
    assert not result["all_required_analysis_conditions_satisfied"]
    first = protocol()
    first_plan = statistics.plan(first)
    ci = report(first_plan, data(first_plan, first, [0.125, 0.25, 0.375, -0.125]))["results"][0][
        "interval"
    ]
    first["maximum_comparisons"] = 8
    wider_plan = statistics.plan(first)
    wider = report(wider_plan, data(wider_plan, first, [0.125, 0.25, 0.375, -0.125]))["results"][0][
        "interval"
    ]
    assert wider[0] <= ci[0] and wider[1] >= ci[1]


def test_paired_statistics_insufficient_resolution_groups_and_missing_pairs_are_not_passes() -> (
    None
):
    p = protocol(groups=1)
    plan = statistics.plan(p)
    assert (
        report(plan, data(plan, p))["results"][0]["reason"]
        == "insufficient_unique_group_or_task_support"
    )
    p = protocol()
    p["maximum_comparisons"] = 256
    plan = statistics.plan(p)
    result = report(plan, data(plan, p))["results"][0]
    assert (
        result["reason"] == "insufficient_resampling_resolution_for_corrected_tail"
        and result["interval"] is None
    )
    p = protocol()
    plan = statistics.plan(p)
    for status in ["missing", "failed", "timed_out"]:
        raw = data(plan, p)
        raw["observations"][0]["status"] = status
        assert report(plan, raw)["results"][0]["status"] == "incomplete"
    raw = data(plan, p)
    raw["observations"].pop()
    assert report(plan, raw)["results"][0]["status"] == "incomplete"


def test_paired_statistics_equal_group_weights_prevent_large_group_pseudoreplication() -> None:
    p = protocol(exact=True, groups=2)
    tasks = p["contrasts"][0]["tasks"]
    large = tasks[0]["group"]
    for i in range(20):
        tasks.append(
            {"item": identity(["extra", i]), "input": identity(["extra-input", i]), "group": large}
        )
    plan = statistics.plan(p)
    raw = data(plan, p, [0.25, -0.25])
    row = report(plan, raw)["results"][0]
    assert row["estimate"]["gain"] == 0.0 and row["unique_groups"] == 2
    raw["observations"] += copy.deepcopy(raw["observations"][:12])
    duplicate = report(plan, raw)
    assert duplicate["duplicates_dropped"] == 12
    assert duplicate["results"][0]["estimate"] == row["estimate"]
    assert duplicate["results"][0]["complete_pairs"] == row["complete_pairs"]


def test_paired_statistics_directions_retention_thresholds_and_noop_positive_gain() -> None:
    p = protocol()
    c = p["contrasts"][0]
    c["schema"] = {
        "domain": "latency",
        "direction": "lower_is_better",
        "unit": "milliseconds",
        "lower": 0.0,
        "upper": None,
    }
    c["threshold"] = {"floor": None, "ceiling": 90.0, "min_gain": 10.0, "max_regression": None}
    plan = statistics.plan(p)
    raw = data(plan, p)
    for o in raw["observations"]:
        o["value"] = 100.0 if o["arm"] == "baseline" else 80.0
    row = report(plan, raw)["results"][0]
    assert row["status"] == "passed" and row["interval"] == [20.0, 20.0]
    c["no_op_control"] = True
    noop = statistics.plan(p)
    raw["plan"] = noop.plan_id
    assert report(noop, raw)["results"][0]["status"] == "no_op_cannot_satisfy_positive_gain"
    p = protocol()
    p["contrasts"][0]["threshold"] = {
        "floor": None,
        "ceiling": None,
        "min_gain": None,
        "max_regression": 0.125,
    }
    plan = statistics.plan(p)
    assert report(plan, data(plan, p, [-0.125]))["results"][0]["status"] == "passed"
    assert report(plan, data(plan, p, [-0.25]))["results"][0]["status"] == "failed"


def test_paired_statistics_exact_repeated_noise_is_not_exact_evidence() -> None:
    p = protocol(exact=True)
    plan = statistics.plan(p)
    raw = data(plan, p)
    raw["observations"][2]["value"] += 0.125
    result = report(plan, raw)["results"][0]
    assert result["status"] == "inconclusive" and result["interval"] is None
    assert "repeat_variation" in result["reason"]


@pytest.mark.parametrize(
    "mutation",
    [
        "model",
        "context",
        "workload",
        "measurement_protocol",
        "group",
        "input",
        "seed",
        "unit",
        "statistic",
        "unknown_task",
        "unknown_contrast",
        "conflict",
        "nan",
        "inf",
        "domain",
        "stale_plan",
    ],
)
def test_paired_statistics_rejects_unmatched_conflicting_or_nonfinite_outcomes(
    mutation: str,
) -> None:
    p = protocol()
    plan = statistics.plan(p)
    raw = data(plan, p)
    o = raw["observations"][0]
    if mutation in {"model", "context", "workload", "measurement_protocol", "group", "input"}:
        o[mutation] = identity("wrong")
    if mutation == "seed":
        o["seed"] = 999
    if mutation == "unit":
        o["unit"] = "bytes"
    if mutation == "statistic":
        o["value_statistic"] = "mean"
    if mutation == "unknown_task":
        o["item"] = identity("unknown-task")
    if mutation == "unknown_contrast":
        o["contrast"] = identity("unknown-contrast")
    if mutation == "conflict":
        raw["observations"].append({**o, "value": 0.9})
    if mutation == "nan":
        o["value"] = float("nan")
    if mutation == "inf":
        o["value"] = float("inf")
    if mutation == "domain":
        o["value"] = 1.1
    if mutation == "stale_plan":
        raw["plan"] = identity("stale")
    with pytest.raises(nf.NativeError):
        statistics.analyze(plan, raw)


@pytest.mark.parametrize(
    "mutation",
    [
        "confidence",
        "groups",
        "repeats",
        "resamples",
        "multiplicity",
        "estimand",
        "duplicate_task",
        "duplicate_input",
        "relabel_group",
        "threshold",
        "unit",
        "too_large",
        "changed_baseline",
        "changed_control",
    ],
)
def test_paired_statistics_frozen_protocol_cannot_weaken_support_or_accounting(
    mutation: str,
) -> None:
    p = protocol(comparisons=2)
    if mutation == "confidence":
        p["confidence"] = 1.0
    if mutation == "groups":
        p["minimum_groups"] = 1
    if mutation == "repeats":
        p["seeds"] = [17, 17]
    if mutation == "resamples":
        p["resamples"] = 0
    if mutation == "multiplicity":
        p["maximum_comparisons"] = 1
    if mutation == "estimand":
        p["estimand"] = "unplanned_task_weighted_mean"
    if mutation == "duplicate_task":
        p["contrasts"][0]["tasks"].append(p["contrasts"][0]["tasks"][0])
    if mutation == "duplicate_input":
        p["contrasts"][0]["tasks"][1]["input"] = p["contrasts"][0]["tasks"][0]["input"]
    if mutation == "relabel_group":
        p["contrasts"][1]["tasks"][0]["group"] = identity("renamed")
    if mutation == "threshold":
        p["contrasts"][0]["threshold"]["min_gain"] = 0.0
    if mutation == "unit":
        p["contrasts"][0]["schema"]["unit"] = "bytes"
        p["contrasts"][0]["schema"]["domain"] = "latency"
    if mutation == "too_large":
        p["resamples"] = 50001
    if mutation == "changed_baseline":
        p["contrasts"][1]["baseline"] = identity("unmatched baseline")
    if mutation == "changed_control":
        p["contrasts"][1]["candidate"] = p["contrasts"][0]["candidate"]
        p["contrasts"][1]["no_op_control"] = True
    with pytest.raises(nf.NativeError):
        statistics.plan(p)


def test_paired_statistics_actual_native_fixture_repair_outcomes_and_cancel() -> None:
    original = arithmetic.fixture()
    candidate_plan, smoke = repair_plan(original)
    fixed = surgeon.apply(candidate_plan, smoke).snapshot()
    inventory = arithmetic.describe(original)["inventory"]["cases"]
    cases = [
        next(c for c in inventory if (c["x1"], c["x2"]) == pair)
        for pair in [(1, 2), (3, 0), (5, -2), (-1, 0)]
    ]
    p = protocol(exact=True)
    c = p["contrasts"][0]
    c["candidate"], c["baseline"] = fixed.manifest_digest, original.source.manifest_digest
    c["tasks"] = [
        {"item": x["item"], "group": x["group"], "input": identity([x["x1"], x["x2"]])}
        for x in cases
    ]
    plan = statistics.plan(p)
    raw = data(plan, p)
    for o in raw["observations"]:
        case = next(x for x in cases if x["item"] == o["item"])
        source = original.source if o["arm"] == "baseline" else fixed
        output = source.forward([float(case["x1"]), float(case["x2"])])[0]
        o["value"] = float(
            abs(output - arithmetic.expected("sum", case["x1"], case["x2"])) <= 1e-10
        )
    result = report(plan, raw)["results"][0]
    assert result["status"] == "passed" and result["estimate"]["gain"] == 0.75
    token = nf.CancellationHandle()
    token.cancel()
    with pytest.raises(nf.CancelledError):
        statistics.analyze(plan, raw, token)
