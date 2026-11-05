"""Actual seeded CPU original/no-op processes, native apply/export/reload and cache."""

import copy
import json
from pathlib import Path
from typing import Any

import neural_foundry as nf
import pytest
from native_fixture import wire
from neural_foundry import _native, arithmetic, controls, models
from test_arithmetic_gyms import repair_plan


def identity(value: Any) -> str:
    return nf.content_digest(_native.canonical_json(wire(value)))


def protocol(model: Any) -> dict[str, Any]:
    cases = arithmetic.describe(model)["inventory"]["cases"]
    chosen = [
        next(c for c in cases if (c["x1"], c["x2"]) == pair)
        for pair in [(1, 2), (2, 1), (-1, 3), (0, 0)]
    ]
    return {
        "schema_version": 1,
        "scope": "public_synthetic_fixture_audit",
        "protocol": identity("matched-original-noop-v1"),
        "context": identity("public-audit"),
        "head": "sum",
        "tasks": chosen,
        "seeds": [17, 42],
        "settings": {
            "device": "cpu",
            "precision": "f64",
            "batch_size": 1,
            "concurrency": 1,
            "model_cache": "disabled",
            "state_reset": "fresh_process_per_arm_seeded_each_task",
        },
        "order": "original_first",
        "worker_limits": {
            "cpu_seconds": 10,
            "wall_ms": 15000,
            "address_space_bytes": 2147483648,
            "output_bytes": 262144,
        },
    }


@pytest.fixture(scope="module")
def runtime(tmp_path_factory: pytest.TempPathFactory) -> Any:
    return models.qualify(tmp_path_factory.mktemp("paired-control-runtime"), enabled=True)


def run(
    cache: Any, runtime: Any, model: Any, data: dict[str, Any], root: Path
) -> tuple[Any, dict[str, Any]]:
    plan, smoke = repair_plan(model, factor=1.0)
    witness = controls.run(cache, runtime, plan, smoke, data, root)
    report = json.loads(witness.report_bytes())
    assert witness.receipt_id == nf.content_digest(witness.report_bytes())
    return witness, report


def test_paired_controls_actual_apply_export_reload_exact_zero_gain(
    runtime: Any, tmp_path: Path
) -> None:
    model = arithmetic.fixture("corrupt_sum")
    original = model.checkpoint_bytes()
    cache = controls.SourceCache()
    witness, report = run(cache, runtime, model, protocol(model), tmp_path / "cas")
    assert report["exact_parameters_buffers_aliases_assets"] and report["exact_behavior"]
    assert report["fresh_process_no_op_reload"]
    assert report["export"]["stage"] == "standalone_export_and_core_reload"
    assert report["export"]["transaction"]["changed_parameters"] == 0
    assert (
        report["export"]["transaction"]["before_parameters"]
        == report["export"]["transaction"]["after_parameters"]
    )
    assert report["original_accuracy"] == report["no_op_accuracy"] == 0.5
    assert report["no_op_accuracy_gain"] == 0 and not report["no_op_positive_gain_eligible"]
    assert report["unique_tasks"] == 4 and report["unique_groups"] == 3
    assert report["task_seed_pairs"] == 8 and len(report["observations"]) == 8
    for arm in ["source_execution", "no_op_execution"]:
        assert report[arm]["status"] == "verified_inference" and report[arm]["cleanup_confirmed"]
        assert report[arm]["peak_rss_bytes"] > 10000000 and report[arm]["cpu_ms"] > 0
        assert report[arm]["child_report"]["seeds"] == [17] * 4 + [42] * 4
    assert report["source_execution"]["job"] != report["no_op_execution"]["job"]
    assert cache.entries == 1 and not report["source_cache_hit"]
    assert report["new_source_forward_calls"] == report["new_no_op_forward_calls"] == 8
    assert model.checkpoint_bytes() == original
    assert report["promotion_status"] == "not_evaluated" and report["outputs_eligible"] is False
    with pytest.raises(TypeError):
        _native.PairedControlsHandle()
    with pytest.raises(AttributeError):
        witness.receipt_id = "forged"


def test_paired_controls_identical_key_reuses_source_and_deduplicates_task_counts(
    runtime: Any, tmp_path: Path
) -> None:
    model = arithmetic.fixture()
    cache = controls.SourceCache()
    data = protocol(model)
    _, first = run(cache, runtime, model, data, tmp_path / "cas")
    data["tasks"] += [copy.deepcopy(data["tasks"][0])] * 10
    _, second = run(cache, runtime, model, data, tmp_path / "cas")
    assert first["evidence_key"] == second["evidence_key"]
    assert first["source_execution_receipt"] == second["source_execution_receipt"]
    assert second["source_cache_hit"] and second["new_source_forward_calls"] == 0
    assert second["new_no_op_forward_calls"] == 8 and second["duplicates_dropped"] == 10
    assert second["unique_tasks"] == 4 and second["unique_groups"] == 3
    assert first["no_op_execution"]["job"] != second["no_op_execution"]["job"]
    assert second["no_op_accuracy_gain"] == 0 and cache.entries == 1
    assert second["export"]["export_root"] == first["export"]["export_root"]


def test_paired_controls_evidence_key_changes_never_reuse_unmatched_source(
    runtime: Any, tmp_path: Path
) -> None:
    model = arithmetic.fixture()
    cache = controls.SourceCache()
    data = protocol(model)
    _, a = run(cache, runtime, model, data, tmp_path / "cas")
    data["context"] = identity("different-allowed-public-context")
    data["order"] = "no_op_first"
    data["seeds"] = [19, 43]
    _, b = run(cache, runtime, model, data, tmp_path / "cas")
    assert a["evidence_key"] != b["evidence_key"] and not b["source_cache_hit"]
    assert a["source_execution_receipt"] != b["source_execution_receipt"]
    assert b["source_execution"]["child_report"]["seeds"] == [19] * 4 + [43] * 4
    assert b["no_op_accuracy_gain"] == 0 and cache.entries == 2
    changed_model = arithmetic.fixture("reference")
    _, c = run(cache, runtime, changed_model, data, tmp_path / "cas")
    assert c["evidence_key"] not in {a["evidence_key"], b["evidence_key"]}
    assert not c["source_cache_hit"] and c["original_accuracy"] == 1.0


@pytest.mark.parametrize(
    "mutation",
    [
        "precision",
        "device",
        "concurrency",
        "cache",
        "scope",
        "relabel",
        "unknown_task",
        "duplicate_seed",
        "no_seed",
        "too_many",
    ],
)
def test_paired_controls_unmatched_or_ineligible_protocol_is_denied(
    runtime: Any, tmp_path: Path, mutation: str
) -> None:
    model = arithmetic.fixture()
    cache = controls.SourceCache()
    data = protocol(model)
    if mutation == "precision":
        data["settings"]["precision"] = "f32"
    if mutation == "device":
        data["settings"]["device"] = "cuda"
    if mutation == "concurrency":
        data["settings"]["concurrency"] = 2
    if mutation == "cache":
        data["settings"]["model_cache"] = "warmed"
    if mutation == "scope":
        data["scope"] = "private_holdout"
    if mutation == "relabel":
        data["tasks"][0]["group"] = identity("fake-group")
    if mutation == "unknown_task":
        data["tasks"][0]["item"] = identity("renamed-task")
    if mutation == "duplicate_seed":
        data["seeds"] = [17, 17]
    if mutation == "no_seed":
        data["seeds"] = []
    if mutation == "too_many":
        data["tasks"] *= 33
    with pytest.raises(nf.NativeError):
        run(cache, runtime, model, data, tmp_path / "cas")
    assert cache.entries == 0 and not (tmp_path / "cas").exists()


def test_paired_controls_torch_apply_backend_is_also_actually_executed(
    runtime: Any, tmp_path: Path
) -> None:
    model = arithmetic.fixture()
    plan, smoke = repair_plan(model, factor=1.0, backend="nf.trusted_torch_cpu")
    witness = controls.run(
        controls.SourceCache(), runtime, plan, smoke, protocol(model), tmp_path / "cas"
    )
    report = json.loads(witness.report_bytes())
    assert report["application"]["status"] == "verified_torch_edit"
    assert report["application"]["child_report"]["inference_mode"] is True
    assert report["export"]["transaction"]["changed_parameters"] == 0
    assert report["exact_behavior"] and report["no_op_accuracy_gain"] == 0.0


def test_paired_controls_failed_real_worker_never_populates_source_cache(
    runtime: Any, tmp_path: Path
) -> None:
    model = arithmetic.fixture()
    cache = controls.SourceCache()
    data = protocol(model)
    data["worker_limits"]["wall_ms"] = 100
    with pytest.raises(nf.NativeError, match="incomplete/failed"):
        run(cache, runtime, model, data, tmp_path / "cas")
    assert cache.entries == 0


def test_paired_controls_non_noop_and_sticky_cancel_cannot_cache_success(
    runtime: Any, tmp_path: Path
) -> None:
    model = arithmetic.fixture()
    cache = controls.SourceCache()
    plan, smoke = repair_plan(model)
    with pytest.raises(nf.NativeError, match="not a no-op"):
        controls.run(cache, runtime, plan, smoke, protocol(model), tmp_path / "cas")
    noop, smoke = repair_plan(model, factor=1.0)
    token = nf.CancellationHandle()
    token.cancel()
    with pytest.raises(nf.CancelledError):
        controls.run(cache, runtime, noop, smoke, protocol(model), tmp_path / "cas", token)
    assert cache.entries == 0
