"""T041 real Rust/SDK/CLI conformance, with local fixture inputs only."""

import copy
import json
import os
import socket
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

import neural_foundry as nf
import pytest
from neural_foundry import _native

ROOT = Path(__file__).resolve().parents[1]
CLI = Path(os.environ.get("NF_CLI", "/tmp/nf-t041-supplied/debug/nf"))


@pytest.fixture(scope="module")
def native_request(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("manual-native-inputs")
    env = dict(os.environ, NF_T041_TEST_FIXTURE_OUTPUT=str(root))
    env.setdefault("CARGO_TARGET_DIR", "/tmp/nf-t041-supplied")
    env.setdefault("CARGO_BUILD_JOBS", "2")
    env.setdefault("RUSTUP_TOOLCHAIN", "stable")
    result = subprocess.run(
        [str(Path.home() / ".cargo/bin/cargo"), "test", "--offline", "-p", "nf-experiments",
         "--test", "manual_workflow", "manual_workflow_emit_fixture", "--", "--exact"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=90, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return dict(json.loads((root / "request.json").read_bytes()))


def workflow(request: dict[str, Any]) -> nf.ResearchWorkflow:
    return nf.ResearchWorkflow(**{key: value for key, value in request.items() if key != "schema_version"})


def cli(tmp_path: Path, action: str, request: dict[str, Any]) -> subprocess.CompletedProcess[str]:
    assert CLI.is_file(), "Build the matching real CLI first"
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    return subprocess.run([str(CLI), "study", action, str(path)],
                          capture_output=True, text=True, timeout=30, check=False)


def files(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_manual_workflow_plan_is_pure_and_shared_with_cli(
    native_request: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    w = workflow(native_request)
    store = Path(native_request["store"])
    before = files(store)
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("planning called a model/runtime/network/process Python entry point")
    with monkeypatch.context() as m:
        for name in ("load_source", "execute_plan"):
            if hasattr(_native, name):
                m.setattr(_native, name, forbidden)
        m.setattr(socket, "socket", forbidden)
        m.setattr(urllib.request, "urlopen", forbidden)
        m.setattr(subprocess, "Popen", forbidden)
        plan = w.plan()
        assert w.validate() == plan
    assert plan["execution"] == "not_run"
    assert plan["valid"] is True
    assert plan["execution_ready"] is False
    assert plan["authorization_verified"] is False
    assert plan["compiled_plan"]["affected_parameter_upper_bound"] == 4
    assert files(store) == before
    result = cli(tmp_path, "plan", native_request)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["result"] == plan
    assert not (store.parent / "ledger").exists()


def test_manual_workflow_freeze_status_report_stop_and_recovery(
    native_request: dict[str, Any], tmp_path: Path,
) -> None:
    request = copy.deepcopy(native_request)
    request["contract"]["intervention"]["method_protocol"] = workflow(request).plan()["manifest_digest"]
    w = workflow(request)
    plan = w.plan()
    ledger = tmp_path / "ledger"
    frozen = w.freeze(ledger=ledger, expected_plan_digest=plan["plan_digest"])
    assert frozen == w.freeze(ledger=ledger, expected_plan_digest=plan["plan_digest"])
    assert frozen["authorization_verified"] is False
    args = {"ledger": ledger, "contract_digest": frozen["contract_digest"]}
    data = w.status(**args)
    assert data["status"]["stage"] == "frozen"
    assert data["authority"]["accepted"] is False
    bundle = w.report(**args)
    assert json.loads(bundle["json"])["snapshot"] == data["snapshot"]
    assert "not acceptance authority" in bundle["markdown"]
    output = subprocess.run([str(CLI), "study", "report", str(ledger / "events.sqlite3"),
                             frozen["contract_digest"], "--format", "both"],
                            capture_output=True, text=True, timeout=30, check=True)
    assert json.loads(json.loads(output.stdout)["json"])["snapshot"] == data["snapshot"]
    stop = w.stop(**args)
    assert stop["request_status"] == "persisted"
    assert stop["cleanup_status"] == "not_confirmed_by_request"
    assert w.stop(**args) == stop
    recovery = w.resume(**args)
    assert recovery["mode"] == "recovery_only"
    assert recovery["execution"] == "not_resumed"
    assert recovery["stop_request"]["reason"] == "owner_stop"
    after = w.status(**args)
    assert after["split_evidence"] == data["split_evidence"]
    assert after["budgets"]["aggregate"]["spent"] == data["budgets"]["aggregate"]["spent"]


@pytest.mark.parametrize("field", ["contract", "split", "smoke", "missing_store"])
def test_manual_workflow_native_errors_match_cli_without_writes(
    native_request: dict[str, Any], tmp_path: Path, field: str,
) -> None:
    request = copy.deepcopy(native_request)
    wrong = "sha256:" + "0" * 64
    if field == "contract":
        request["program"]["objective"]["contract"] = wrong
    elif field == "split":
        request["program"]["verification"]["gyms"][0]["split"] = wrong
    elif field == "smoke":
        request["smoke_suite"]["probes"][0]["max_abs_output"] = 999.0
    else:
        request["store"] = str(tmp_path / "absent")
    before = files(Path(native_request["store"]))
    with pytest.raises(nf.NativeError) as caught:
        workflow(request).validate()
    result = cli(tmp_path, "validate", request)
    assert result.returncode != 0
    assert json.loads(result.stderr)["error"] == {
        "code": caught.value.code, "path": caught.value.path, "message": caught.value.message,
    }
    assert files(Path(native_request["store"])) == before
    assert not (tmp_path / "absent").exists()


def test_manual_workflow_never_mints_authority_or_initializes_missing_ledger(
    native_request: dict[str, Any], tmp_path: Path,
) -> None:
    w = workflow(native_request)
    absent = tmp_path / "absent"
    digest = "sha256:" + "0" * 64
    for auth, code in ((None, "PERMISSION_DENIED"), ("sha256:" + "1" * 64, "PERMISSION_DENIED")):
        with pytest.raises(nf.NativeError) as caught:
            w.run(ledger=absent, contract_digest=digest, authorization=auth)
        assert caught.value.code == code
    for action in (w.status, w.stop, w.resume, w.report):
        with pytest.raises(nf.NativeError):
            action(ledger=absent, contract_digest=digest)
    assert not absent.exists()


def test_manual_workflow_draft_is_captured_not_mutable_alias(
    native_request: dict[str, Any],
) -> None:
    request = copy.deepcopy(native_request)
    w = workflow(request)
    expected = w.plan()
    request["contract"]["objective"]["description"] = "mutated after composition"
    assert w.plan() == expected


def test_manual_workflow_operator_authorization_is_not_a_caller_flag(
    native_request: dict[str, Any], tmp_path: Path,
) -> None:
    if os.getuid() == 0:
        pytest.skip("researcher-uid boundary requires an unprivileged process")
    before = files(Path(native_request["store"]))
    for action in ("authorize", "revoke", "serve-authority"):
        request = {"approved": True, "operator": "user says yes"}
        with pytest.raises(nf.NativeError) as caught:
            _native.manual_workflow_request(action, json.dumps(request).encode())
        assert caught.value.code == "PERMISSION_DENIED"
        result = cli(tmp_path, action, request)
        assert result.returncode != 0
        assert json.loads(result.stderr)["error"]["code"] == "PERMISSION_DENIED"
    assert files(Path(native_request["store"])) == before


def test_manual_workflow_v2_explicit_control_has_no_hash_cycle_or_plan_side_effects(
    native_request: dict[str, Any], tmp_path: Path,
) -> None:
    request = copy.deepcopy(native_request)
    request["schema_version"] = 2
    request["contract"]["intervention"]["method_protocol"] = request["program"]["objective"]["experiment_protocol"]
    control = copy.deepcopy(request["program"])
    control["operations"] = []
    control["effects"]["reads"] = []
    control["effects"]["writes"] = []
    control["lineage"]["method"]["target_update"] = "no_change"
    control["lineage"]["method"]["changed_parameters"] = 0
    control["constraints"]["max_changed_parameters"] = 0
    request["control_program"] = control
    original_program = copy.deepcopy(request["program"])
    original_contract = copy.deepcopy(request["contract"])
    before = files(Path(request["store"]))
    w = workflow(request)
    plan = w.plan()
    assert plan["contract_binding_matches"] is True
    assert plan["operator_binding"]["revision"] == 2
    assert plan["execution"] == "not_run"
    assert plan["manifest"]["method_protocol"] == original_program["objective"]["experiment_protocol"]
    assert plan["manifest_digest"] != original_contract["intervention"]["method_protocol"]
    assert files(Path(request["store"])) == before
    result = cli(tmp_path, "plan", request)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["result"] == plan
    frozen = w.freeze(ledger=tmp_path / "ledger", expected_plan_digest=plan["plan_digest"])
    assert request["program"] == original_program
    assert request["contract"] == original_contract
    assert frozen["manifest_digest"] == plan["manifest_digest"]
    with pytest.raises(nf.NativeError) as caught:
        w.run(ledger=tmp_path / "ledger", contract_digest=frozen["contract_digest"],
              authorization=plan["manifest_digest"])
    assert caught.value.code == "PERMISSION_DENIED"


def test_manual_workflow_explicit_profile_observation_is_not_execution_authority(tmp_path: Path) -> None:
    result = cli(tmp_path, "runtime-profile", {"schema_version": 1})
    assert result.returncode == 0, result.stderr
    profile = json.loads(result.stdout)["result"]
    assert profile["execution_authority"] is False
    assert profile["runtime"]["backend"]["id"] == "fixture.cpu"
    assert profile["runtime"]["device"] == "cpu"
