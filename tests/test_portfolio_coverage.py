"""Live arithmetic probes + real custom-gym approval, not DTO-only binding authority."""

import copy
import json
from pathlib import Path
from typing import Any

import neural_foundry as nf
import pytest
from native_fixture import wire
from neural_foundry import _native, arithmetic, gyms, portfolios
from test_gym_validation import fixture as custom_fixture
from test_gym_validation import run as validate_custom


def identity(value: Any) -> str:
    return nf.content_digest(_native.canonical_json(wire(value)))


@pytest.fixture(scope="module")
def sources() -> list[Any]:
    return [portfolios.builtin(head) for head in arithmetic.HEADS]


def proposal(
    sources: list[Any], profile: str = "standard_five_gym"
) -> tuple[dict[str, Any], dict[str, Any]]:
    objective = identity("portfolio-target")
    inventory = arithmetic.describe(arithmetic.fixture("reference"))["inventory"]
    splits = {}
    for role in ("development", "selection", "holdout"):
        cases = [c for c in inventory["cases"] if c["split"] == role]
        splits[role] = {
            "schema_version": 1,
            "role": role,
            "dataset": identity(inventory),
            "items": [{key: c[key] for key in ("item", "group", "family")} for c in cases],
            # Schema declaration only. Composition does NOT attest custody or grant reads.
            "exposure": "known_unexposed",
            "exposure_ledger": identity("unresolved-history"),
            "access_policy": identity(role),
        }
    manifests = {s.gym_id: json.loads(s.manifest_bytes()) for s in sources}
    bindings = []
    for i, s in enumerate(sources):
        metric = next(iter(manifests[s.gym_id]["metrics"]))
        bindings.append(
            {
                "id": identity(["binding", i]),
                "objective": objective,
                "gym": s.gym_id,
                "validation": manifests[s.gym_id]["validation"],
                "roles": ["target", "independent_verification"] if i == 0 else ["retention"],
                "required": True,
                "metrics": [
                    {
                        "metric": metric,
                        "required": True,
                        "priority": float(i + 1),
                        "comparator": "original",
                        "threshold": {
                            "floor": 0.9 if i == 0 else None,
                            "ceiling": None,
                            "max_regression": 0.0 if i else None,
                            "min_gain": None,
                        },
                        "statistic": "exact",
                        "measurement": None,
                    }
                ],
                "splits": copy.deepcopy(splits),
                "schedule": {
                    "development_episodes": 2,
                    "selection_episodes": 2,
                    "holdout_episodes": 2,
                    "every_n_candidates": 1,
                    "final_required": True,
                },
            }
        )
    coverage = {
        "schema_version": 1,
        "objective": objective,
        "architecture_surgery": False,
        "needs": [
            {
                "dimension": "target_tasks",
                "mandatory": True,
                "measures": [
                    {"binding": bindings[0]["id"], "metric": bindings[0]["metrics"][0]["metric"]}
                ],
            },
            {
                "dimension": "protected_behaviors",
                "mandatory": True,
                "measures": [
                    {"binding": bindings[1]["id"], "metric": bindings[1]["metrics"][0]["metric"]}
                ],
            },
        ],
    }
    portfolio = {
        "schema_version": 1,
        "profile": profile,
        "label": "public synthetic arithmetic planning",
        "gyms": manifests,
        "bindings": bindings,
        "coverage": identity(coverage),
        "curriculum": identity([[b["id"], b["schedule"]] for b in bindings]),
        "deduplicate_shared_tasks": True,
        "independent": {
            "scope": "fixture_only",
            "evaluator": {
                "id": "fixture.independent",
                "version": "1",
                "config": identity("evaluator"),
            },
            "administration": identity("fixture-admin"),
            "isolation": "trusted_fixture_broker",
            "frozen_panel_protocol": identity("final-panel"),
            "close_revision_after_feedback": True,
            "require_group_disjoint": True,
        },
    }
    return portfolio, coverage


def bind(
    sources: list[Any], profile: str = "standard_five_gym"
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    spec, coverage = proposal(sources, profile)
    return portfolios.compose(spec, coverage, sources), spec, coverage


def numbers(plan: Any, spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "portfolio": plan.portfolio_id,
        "observations": [
            {
                "binding": b["id"],
                "metric": m["metric"],
                "comparator": m["comparator"],
                "status": "complete",
                "value": 1.0,
                "baseline": 1.0,
                "unit": spec["gyms"][b["gym"]]["metrics"][m["metric"]]["unit"],
                "statistic": m["statistic"],
            }
            for b in spec["bindings"]
            for m in b["metrics"]
        ],
    }


def test_portfolio_coverage_live_named_five_gym_is_not_an_assurance_pass(
    sources: list[Any],
) -> None:
    plan, spec, _ = bind(sources)
    record = json.loads(plan.record_bytes())
    assert record["status"] == "bound_for_planning" and record["dispatch_allowed"] is False
    assert record["coverage_report"]["mandatory_gaps"] == 0
    assert record["coverage_report"]["standard_five_gym_declared"] is True
    assert record["coverage_report"]["standard_five_gym_passed"] is False
    assert len(record["coverage_report"]["validation_states"]) == 5
    assert record["coverage_report"]["duplicate_data_sources"] == 4
    assert record["coverage_report"]["cost_envelope"]["cpu_ms"] is None
    assert record["split_custody_verified"] is False
    for source in sources:
        manifest = json.loads(source.manifest_bytes())
        assert nf.content_digest(source.validation_bytes()) == manifest["validation"]
        assert len(json.loads(source.validation_bytes())["observations"]) == 289
    assert json.loads(plan.assess_bytes(wire(numbers(plan, spec))))["constraints_satisfied"] is True
    schedule = json.loads(plan.schedule_bytes("final"))
    assert len(schedule["entries"]) == 5 and schedule["unique_tasks"] == 2
    assert schedule["unique_groups"] <= 2 and schedule["dispatch_allowed"] is False
    assert all(len(e["items"]) == 2 and e["due"] for e in schedule["entries"])
    spec["bindings"][0]["metrics"][0]["priority"] = 0
    assert (
        json.loads(plan.record_bytes())["portfolio"]["bindings"][0]["metrics"][0]["priority"] == 1
    )
    assert (
        portfolios.compose(record["portfolio"], record["coverage_plan"], sources).portfolio_id
        == plan.portfolio_id
    )


def test_portfolio_coverage_exploratory_subset_does_not_claim_standard(sources: list[Any]) -> None:
    plan, _, _ = bind(sources[:2], "exploratory")
    report = json.loads(plan.record_bytes())["coverage_report"]
    assert report["mandatory_gaps"] == 0 and report["standard_five_gym_declared"] is False
    spec, coverage = proposal(sources[:2])
    with pytest.raises(nf.NativeError, match="at least five"):
        portfolios.compose(spec, coverage, sources[:2])


@pytest.mark.parametrize("status", ["failed", "timed_out", "missing", "omitted"])
def test_portfolio_coverage_weights_never_compensate_required_failure(
    sources: list[Any], status: str
) -> None:
    spec, coverage = proposal(sources)
    for b in spec["bindings"]:
        b["metrics"][0]["priority"] = 1000000000.0
    plan = portfolios.compose(spec, coverage, sources)
    data = numbers(plan, spec)
    data["observations"][0]["value"] = 0.0
    data["observations"][0]["status"] = status if status != "omitted" else "complete"
    if status == "omitted":
        data["observations"].pop(0)
    report = json.loads(plan.assess_bytes(wire(data)))
    assert not report["constraints_satisfied"] and report["failed_required"]
    assert not report["weighted_compensation_allowed"] and not report["scientific_acceptance"]


@pytest.mark.parametrize(
    "mutation",
    ["missing_baseline", "duplicate", "units", "unknown_binding", "nan", "stale_portfolio"],
)
def test_portfolio_coverage_strict_measurement_identity_and_missing_evidence(
    sources: list[Any], mutation: str
) -> None:
    plan, spec, _ = bind(sources)
    data = numbers(plan, spec)
    if mutation == "missing_baseline":
        data["observations"][1]["baseline"] = None
        assert not json.loads(plan.assess_bytes(wire(data)))["constraints_satisfied"]
        return
    if mutation == "duplicate":
        data["observations"].append(data["observations"][0])
    if mutation == "units":
        data["observations"][0]["unit"] = "milliseconds"
    if mutation == "unknown_binding":
        data["observations"][0]["binding"] = identity("unknown")
    if mutation == "nan":
        data["observations"][0]["value"] = float("nan")
    if mutation == "stale_portfolio":
        data["portfolio"] = identity("stale")
    with pytest.raises(nf.NativeError):
        plan.assess_bytes(wire(data))


def test_portfolio_coverage_surgery_gaps_cannot_be_waived_by_json(sources: list[Any]) -> None:
    spec, coverage = proposal(sources)
    coverage["architecture_surgery"] = True
    for dimension in ("context_boundaries", "cache_correctness", "latency"):
        coverage["needs"].append(
            {
                "dimension": dimension,
                "mandatory": False,
                "measures": [{"binding": spec["bindings"][0]["id"], "metric": "accuracy"}],
            }
        )
    spec["coverage"] = identity(coverage)
    plan = portfolios.compose(spec, coverage, sources)
    report = json.loads(plan.record_bytes())["coverage_report"]
    assert report["mandatory_gaps"] == 3
    gaps = [
        r
        for r in report["rows"]
        if r["dimension"] in {"context_boundaries", "cache_correctness", "latency"}
    ]
    assert all(r["mandatory"] and r["status"] == "weak_or_unsupported" for r in gaps)
    assert all("validated" in r["recommendation"] for r in gaps)
    assert not json.loads(plan.assess_bytes(wire(numbers(plan, spec))))["constraints_satisfied"]


def test_portfolio_coverage_fixed_schedules_and_missing_required_gate(sources: list[Any]) -> None:
    spec, coverage = proposal(sources)
    spec["bindings"][1]["schedule"]["every_n_candidates"] = 2
    with pytest.raises(nf.NativeError, match="curriculum"):
        portfolios.compose(spec, coverage, sources)
    spec["curriculum"] = identity([[b["id"], b["schedule"]] for b in spec["bindings"]])
    plan = portfolios.compose(spec, coverage, sources)
    assert json.loads(plan.schedule_bytes("selection", 1))["entries"][1]["due"] is True
    assert (
        json.loads(plan.schedule_bytes("selection", 2))["entries"][1]["deferred_required"] is True
    )
    assert all(e["due"] for e in json.loads(plan.schedule_bytes("final", 2))["entries"])
    with pytest.raises(nf.NativeError):
        plan.schedule_bytes("final", 0)
    with pytest.raises(nf.NativeError):
        plan.schedule_bytes("adaptive")


@pytest.mark.parametrize(
    "mutation",
    [
        "validation",
        "roles",
        "duplicate_gym",
        "coverage_metric",
        "coverage_id",
        "negative_weight",
        "cross_split",
        "missing_live_source",
    ],
)
def test_portfolio_coverage_cannot_mutate_frozen_versions_or_authority(
    sources: list[Any], mutation: str
) -> None:
    spec, coverage = proposal(sources)
    if mutation == "validation":
        spec["bindings"][0]["validation"] = identity("fake")
    if mutation == "roles":
        spec["bindings"][0]["roles"] = ["diagnostic"]
    if mutation == "duplicate_gym":
        spec["bindings"][1]["gym"] = spec["bindings"][0]["gym"]
    if mutation == "coverage_metric":
        coverage["needs"][0]["measures"][0]["metric"] = "reward_is_not_accuracy"
        spec["coverage"] = identity(coverage)
    if mutation == "coverage_id":
        spec["coverage"] = identity("fake")
    if mutation == "negative_weight":
        spec["bindings"][0]["metrics"][0]["priority"] = -1
    if mutation == "cross_split":
        spec["bindings"][0]["splits"]["holdout"]["items"][0] = spec["bindings"][0]["splits"][
            "development"
        ]["items"][0]
    if mutation == "missing_live_source":
        sources = sources[:-1]
    with pytest.raises(nf.NativeError):
        portfolios.compose(spec, coverage, sources)


def test_portfolio_coverage_one_gym_exploration_reports_missing_retention(
    sources: list[Any],
) -> None:
    spec, coverage = proposal(sources[:2], "exploratory")
    removed = spec["bindings"].pop()
    del spec["gyms"][removed["gym"]]
    coverage["needs"][1]["measures"] = []
    spec["coverage"] = identity(coverage)
    spec["curriculum"] = identity([[b["id"], b["schedule"]] for b in spec["bindings"]])
    plan = portfolios.compose(spec, coverage, sources[:1])
    record = json.loads(plan.record_bytes())
    assert record["status"] == "blocked_missing_coverage"
    assert record["coverage_report"]["mandatory_gaps"] == 1
    assert not record["coverage_report"]["standard_five_gym_declared"]


def test_portfolio_coverage_actual_approved_user_gym_is_first_class(
    sources: list[Any], tmp_path: Path
) -> None:
    snapshot, validation_plan, sandbox, _, _ = custom_fixture(tmp_path)
    validation, _ = validate_custom(snapshot, validation_plan, sandbox)
    approval = gyms.approve(snapshot, validation, "explicit exact fixture approval for portfolio")
    user = portfolios.user(snapshot, approval)
    manifest = json.loads(user.manifest_bytes())
    assert manifest["origin"] == "user" and set(manifest["metrics"]) == {"score"}
    assert manifest["approval"] == approval.receipt_id
    assert manifest["validation"] == validation.receipt_id
    mixed = [user, *sources[1:]]
    plan, _, _ = bind(mixed)
    record = json.loads(plan.record_bytes())
    assert record["portfolio"]["bindings"][0]["gym"] == user.gym_id
    assert len(record["portfolio"]["bindings"]) == 5
    assert record["coverage_report"]["mandatory_gaps"] == 0
    assert record["dispatch_allowed"] is False
    with pytest.raises(TypeError):
        _native.PortfolioGymHandle()
    with pytest.raises(TypeError):
        _native.PortfolioHandle()
    with pytest.raises(TypeError):
        portfolios.user(snapshot, {})  # type: ignore[arg-type]
