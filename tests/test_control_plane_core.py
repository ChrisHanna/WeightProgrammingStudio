from __future__ import annotations

from copy import deepcopy

import pytest
from neural_foundry.control_plane.artifacts import ArtifactStore
from neural_foundry.control_plane.bridges import CAPABILITIES, TrustedRuntimeProfile, dispatch
from neural_foundry.control_plane.catalog import OPERATIONS, catalog_document
from neural_foundry.control_plane.handlers import execute
from neural_foundry.control_plane.models import (
    ControlPlaneError,
    compile_plan,
    digest,
    validate_spec,
)
from neural_foundry.control_plane.store import Store


def spec(operation="software.local_eval", config=None):
    return {"version": 1, "name": "bounded fixture", "purpose": "development",
            "candidates": [{"id": "a", "model": {"id": "fixture", "revision": "1"}, "method": "software"}],
            "stages": [{"id": "first", "operation": operation, "depends_on": [],
                        "candidate_id": "a", "config": config or {}, "limits": {}}]}


def test_all_templates_validate_and_have_immutable_fingerprints():
    for template in catalog_document()["templates"]:
        value = validate_spec(template["spec"], OPERATIONS)
        before = deepcopy(value)
        plan = compile_plan(value)
        assert value == before and plan["spec_digest"] == digest(value)
        assert plan["digest"] == digest({k: v for k, v in plan.items() if k != "digest"})
        assert plan["release_authorized"] is False
    value["name"] = "changed"
    assert digest(value) != plan["spec_digest"]


def test_installed_wheel_catalog_reports_repository_backends_unavailable(monkeypatch, tmp_path):
    from neural_foundry import proof_dependencies, proof_registry

    def unexpected_registration():
        pytest.fail("Catalog must not register unavailable repository dependencies")

    monkeypatch.setattr(proof_dependencies, "ROOT", tmp_path)
    monkeypatch.setattr(proof_registry, "builtin_registry", unexpected_registration)
    catalog = catalog_document()
    assert catalog["templates"]
    assert catalog["proof_registry_references"] == {}
    assert catalog["proof_registry_status"]["available"] is False
    assert all(not plugin["enabled"] for plugin in catalog["plugins"]
               if plugin["id"].startswith(("live.", "proof.")))


@pytest.mark.parametrize("change", [
    lambda s: s["stages"][0].update(depends_on=["first"]),
    lambda s: s["stages"][0].update(depends_on=["missing"]),
    lambda s: s["stages"][0].update(operation="live.provider"),
    lambda s: s["stages"][0]["config"].update(path="/etc/passwd"),
    lambda s: s["stages"][0]["limits"].update(steps=True),
    lambda s: s["candidates"][0]["model"].pop("revision"),
])
def test_invalid_specs_are_rejected(change):
    value = spec()
    change(value)
    with pytest.raises(ControlPlaneError):
        validate_spec(value, OPERATIONS)


def test_budget_is_checked_before_dispatch():
    value = spec(config={"inputs": ["{}", "{}"]})
    value["stages"][0]["limits"] = {"steps": 1}
    with pytest.raises(ControlPlaneError, match="exceeds compiled"):
        compile_plan(validate_spec(value, OPERATIONS))


def test_real_software_operations_and_semantic_invalid_output():
    stage = compile_plan(spec(config={"inputs": ["{}"], "outputs": ["not json"]}))["stages"][0]
    result = execute(stage, [])
    assert result["execution"] == "completed" and result["quality"] == "failed"
    assert result["rows"][0]["valid_json"] is False
    stage["operation"] = "software.provider"
    result = execute(stage, [])
    assert result["transport_success"] is True and result["quality"] == "failed"
    assert result["provider_calls"] == result["model_calls"] == 0


@pytest.mark.parametrize("operation", ["software.episode", "software.workflow"])
def test_stateful_runtime_sessions_close(operation):
    stage = compile_plan(spec(operation, {"seeds": [0, 1]}))["stages"][0]
    result = execute(stage, [])
    assert result["session_closed"] and result["episodes"] == 2
    assert result["quality"] == "passed" and result["steps"] >= 3
    if operation.endswith("workflow"):
        assert result["tool_calls"] == 2


def test_real_cpu_tensor_training_export_identity_evaluation():
    stages = []
    for operation in ("training", "export", "identity", "eval"):
        stage = compile_plan(spec("software." + operation))["stages"][0]
        result = execute(stage, stages[-1:])
        stages.append(result)
    assert stages[0]["training_loss_after"] < stages[0]["training_loss_before"]
    assert stages[1]["roundtrip_verified"]
    assert stages[2]["source_digest"] == digest(stages[1])
    assert stages[-1]["quality"] == "passed"
    assert all(r["model_calls"] is False and not r["release_authorized"] for r in stages)


def test_existing_experimental_weight_edit_solver_runs_on_cpu():
    stage = compile_plan(spec("software.weight_edit"))["stages"][0]
    result = execute(stage, [])
    assert any(value != 0 for row in result["delta"] for value in row)
    assert result["no_task_gain_or_preservation_claim"]
    assert result["diagnostics"]["status"] == "experimental_cpu_correctness_only"


def test_comparison_requires_matching_metric_versions_and_keeps_model_references():
    stage = compile_plan(spec("software.compare"))["stages"][0]
    first = {"candidate_model": {"id": "a", "revision": "1"},
             "metrics": [{"name": "accuracy", "version": "1", "unit": "ratio", "value": .5}]}
    second = deepcopy(first)
    second["candidate_model"] = {"id": "b", "revision": "2"}
    second["metrics"][0]["value"] = .75
    result = execute(stage, [first, second])
    assert result["comparisons"][0]["deltas"][0]["candidate_minus_baseline"] == .25
    assert result["comparisons"][0]["candidate_model"] == second["candidate_model"]
    second["metrics"][0]["version"] = "2"
    with pytest.raises(ControlPlaneError) as error:
        execute(stage, [first, second])
    assert error.value.code == "incompatible_metrics"


def test_plan_fingerprint_covers_controller_and_actual_execution_dependencies(monkeypatch):
    from neural_foundry.control_plane import implementation

    paths = {path.relative_to(implementation.PACKAGE_ROOT).as_posix()
             for path in implementation._source_files()}
    assert {"control_plane/service.py", "control_plane/worker.py", "gym_runtime.py",
            "gym_runtime_fixtures.py", "backends/weight_edit/methods/experimental.py"} <= paths
    before = compile_plan(spec())
    original = implementation.sha256_file
    monkeypatch.setattr(implementation, "sha256_file",
                        lambda path: "sha256:" + "f" * 64 if path.name == "gym_runtime.py" else original(path))
    after = compile_plan(spec())
    assert before["spec_digest"] == after["spec_digest"]
    assert before["execution_implementation"] != after["execution_implementation"]
    assert before["digest"] != after["digest"]


def test_artifact_atomic_dedup_integrity_and_path_safety(tmp_path):
    storage = ArtifactStore(tmp_path)
    first = storage.put({"a": 1})
    assert storage.put({"a": 1}) == first
    assert storage.read(first["id"]) == {"a": 1}
    with pytest.raises(ControlPlaneError):
        storage.read("../../credentials")
    (tmp_path / (first["id"][7:] + ".json")).write_text("{}")
    with pytest.raises(ControlPlaneError, match="integrity"):
        storage.read(first["id"])
    assert not list(tmp_path.glob(".write-*"))


def test_artifact_symlink_is_rejected(tmp_path):
    store = ArtifactStore(tmp_path / "objects")
    metadata = store.put({"data": 3})
    location = store._path(metadata["id"])
    location.unlink()
    target = tmp_path / "elsewhere"
    target.write_text("{}")
    try:
        location.symlink_to(target)
    except OSError:
        pytest.skip("Symlink creation unavailable")
    with pytest.raises(ControlPlaneError, match="unsafe"):
        store.read(metadata["id"])


def test_no_sqlite_compatibility():
    with pytest.raises(ControlPlaneError, match="Only PostgreSQL"):
        Store("sqlite://local.db")


@pytest.mark.parametrize("operation", list(CAPABILITIES))
def test_disabled_live_bridges_require_all_three_authorities_and_use_fixed_callback(operation, tmp_path):
    seen = []
    def callback(context):
        seen.append(context)
        return {"receipt": "existing-stage-callback", "model_calls": False}
    profile = TrustedRuntimeProfile("qualified-test", tmp_path, qualified=True,
                                    deployment_capabilities=frozenset({CAPABILITIES[operation]}),
                                    callbacks={operation: callback})
    args = {"document": {"spec": "fixture"}, "role": "candidate-a", "run_id": "run-one",
            "plan_digest": "exact", "approval": {"id": "approval-one", "plan_digest": "wrong"},
            "limits": {"steps": 1}}
    with pytest.raises(ControlPlaneError):
        dispatch(operation, profile, **args)
    assert not seen
    args["approval"]["plan_digest"] = "exact"
    result = dispatch(operation, profile, **args)
    assert seen[0]["document"] == {"spec": "fixture"}
    assert result["release_authorized"] is False
    assert OPERATIONS[operation]["enabled"] is False
