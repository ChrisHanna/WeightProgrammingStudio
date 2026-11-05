"""T045: actual runner evidence is necessary, never sufficient for M2 release."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("m2_checker", ROOT / "scripts/check_m2_release.py")
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def test_m2_capabilities_inventory_preserves_m1_and_exposes_missing_engines() -> None:
    source = (ROOT / "crates/nf-core/src/capabilities.rs").read_text()
    m1 = source.split("pub fn m1_manifest()", 1)[1].split("pub fn m2_manifest()", 1)[0]
    for value in ('"milestone":"M1"', '"promotion_status":"not_evaluated"',
                  '"evidence_scope":"software_fixture_tests_only"', '"study_execution"'):
        assert value in m1
    for value in checker.INCOMPLETE:
        assert f'"{value}"' in source
    for value in checker.AUTHORITY_BLOCKERS:
        assert f'"{value}"' in source
    for value in checker.COMPLIANCE:
        assert f'"{value}"' in source
    assert '"release_enabled":false' in source
    assert '"scientific_evidence":"not_established"' in source
    assert '"operational_fixture_promotion_with_live_grant"' in source
    assert '"operator_scoped_native_manual_dispatch"' in source
    assert '"strict_physical_publication_required_by_m2":false' in source
    assert '"strict_physical_publication":"unsupported_refuse_if_requested"' in source
    assert "hard_publication_boundary" not in source


def test_m2_capabilities_no_execution_cannot_qualify_any_host() -> None:
    result = checker.report(ROOT, [], 10)
    assert result["schema_version"] == 1
    assert result["kind"] == "m2_gate_report"
    assert result["states"] == {
        "implemented": "component_subset", "host_qualified": "not_qualified",
        "workflow_integrated": "incomplete", "release_enabled": False,
        "scientific_evidence": "not_established",
    }
    assert result["selected_checks_passed"] is False
    assert result["timing_policy"]["strict_physical_publication_required_by_m2"] is False
    assert result["timing_policy"]["supported"] is True
    assert not any("strict" in blocker for blocker in result["release_blockers"])
    assert all(value != "passed" for value in result["release_prerequisites"].values())
    assert result["scope_limits"] == {
        "native_fixture_only": True, "strict_fsync_publication": "unsupported",
        "independent_engineering_review": "not_established",
        "mixed_five_gym_dispatch": "not_established",
    }
    assert all(value["status"] == "not_run" for value in result["checks"].values())
    assert set(checker.AUTHORITY_BLOCKERS) <= set(result["release_blockers"])
    assert "gate:isolated_wheel" in result["release_blockers"]
    assert json.loads(json.dumps(result)) == result


def test_m2_capabilities_real_installed_import_evidence_is_not_wheel_or_release() -> None:
    # No mocked success: runs the actual installed-extension smoke tests. Their
    # scope is explicitly NOT a newly built isolated wheel or integrated workflow.
    result = checker.report(ROOT, ["imports"], 30)
    observed = result["checks"]["imports"]
    assert observed["status"] == "passed", observed
    assert observed["counts"]["collected"] >= 3
    assert observed["execution"]["returncode"] == 0
    assert observed["execution"]["log_sha256"].startswith("sha256:")
    assert observed["execution"]["log_bytes"] > 0
    assert result["checks"]["isolated_wheel"]["status"] == "not_run"
    assert result["states"]["release_enabled"] is False
    assert result["states"]["host_qualified"] == "not_qualified"
    assert result["states"]["scientific_evidence"] == "not_established"
    assert json.loads(json.dumps(result)) == result


@pytest.mark.parametrize("body,expected_count", [
    ("", 0),
    ("import pytest\n@pytest.mark.skip(reason='no runtime')\ndef test_probe(): pass\n", 1),
    (("import pytest\n@pytest.mark.xfail(reason='missing integration')\n"
      "def test_probe(): assert False\n"), 1),
    (("import pytest\n@pytest.mark.xfail(strict=False)\n"
      "def test_probe(): assert True\n"), 1),
    ("def test_probe(): assert False\n", 1),
    ("raise RuntimeError('collection failed')\n", 1),
])
def test_m2_capabilities_actual_zero_skip_xfail_failure_and_collection_error_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str, expected_count: int,
) -> None:
    probe = tmp_path / "test_probe.py"
    probe.write_text(body)
    monkeypatch.setitem(checker.SUITES, "imports", (probe.name,))
    result = checker.run_python_gate("imports", tmp_path, 15)
    assert result["status"] == "failed", result
    assert result["counts"]["collected"] == expected_count


def test_m2_capabilities_empty_required_file_cannot_hide_behind_other_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "test_empty.py").write_text("")
    (tmp_path / "test_real.py").write_text("def test_real(): pass\n")
    monkeypatch.setitem(checker.SUITES, "imports", ("test_empty.py", "test_real.py"))
    result = checker.run_python_gate("imports", tmp_path, 15)
    assert result["counts"]["collected"] == 1
    assert result["file_counts"] == {"test_empty.py": 0, "test_real.py": 1}
    assert result["status"] == "failed"


def test_m2_capabilities_runner_timeout_is_not_a_pass(tmp_path: Path) -> None:
    result = checker.execute([sys.executable, "-c", "import time; time.sleep(20)"], tmp_path, 1)
    assert result["timed_out"] is True
    assert result["returncode"] != 0
    assert result["elapsed_seconds"] < 10


def test_m2_capabilities_missing_suite_and_wheel_are_not_evidence(tmp_path: Path) -> None:
    result = checker.run_python_gate("sandbox", tmp_path, 10)
    assert result["status"] == "missing"
    assert checker.run_wheel_gate(tmp_path, 10, None)["status"] == "missing"
    assert checker.run_wheel_gate(ROOT, 10, None)["status"] == "missing"


def test_m2_capabilities_junit_counts_real_cases_not_summary_attributes(tmp_path: Path) -> None:
    xml = tmp_path / "evidence.xml"
    xml.write_text('<testsuites tests="100"><testsuite tests="100"/></testsuites>')
    assert checker.counts_pass(checker.junit_counts(xml)) is False
    xml.write_text('<testsuite><testcase name="x"><skipped/></testcase></testsuite>')
    assert checker.counts_pass(checker.junit_counts(xml)) is False
    xml.write_text('<not_junit/>')
    with pytest.raises(ValueError):
        checker.junit_counts(xml)


@pytest.mark.parametrize("filename", ["lib.rs", "bindings.pyi", "worker.h", "NOTICE.txt"])
def test_m2_capabilities_stale_source_detected(tmp_path: Path, filename: str) -> None:
    source = tmp_path / "crates/example/src" / filename
    source.parent.mkdir(parents=True)
    source.write_text("// first\n")
    first = checker.source_identity(tmp_path)
    source.write_text("// changed\n")
    assert checker.source_identity(tmp_path) != first
    generated = tmp_path / "crates/example/__pycache__/cached.py"
    generated.parent.mkdir()
    before_cache = checker.source_identity(tmp_path)
    generated.write_text("ignored cache\n")
    assert checker.source_identity(tmp_path) == before_cache


def test_m2_capabilities_even_green_components_cannot_clear_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Deliberately adversarial fabricated green statuses, NOT execution evidence.
    # This proves the aggregator cannot grant authority even if a runner lies.
    green: dict[str, Any] = {"status": "passed"}
    monkeypatch.setattr(checker, "run_python_gate", lambda *args: green)
    monkeypatch.setattr(checker, "run_native_gate", lambda *args: green)
    monkeypatch.setattr(checker, "run_runtime_gate", lambda *args: green)
    monkeypatch.setattr(checker, "run_wheel_gate", lambda *args: green)
    result = checker.report(tmp_path, list(checker.REQUIRED), 10)
    assert result["selected_checks_passed"] is True
    assert result["states"]["host_qualified"] == "observed_component_tests_only"
    assert result["states"]["workflow_integrated"] == "incomplete"
    assert result["states"]["release_enabled"] is False
    assert result["states"]["scientific_evidence"] == "not_established"
    assert result["release_blockers"] == list(checker.AUTHORITY_BLOCKERS)


def test_m2_capabilities_source_change_invalidates_selected_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mutate(*args: Any) -> dict[str, str]:
        (tmp_path / "Cargo.toml").write_text("# mutated during execution\n")
        return {"status": "passed"}

    monkeypatch.setattr(checker, "run_python_gate", mutate)
    result = checker.report(tmp_path, ["imports"], 10)
    assert result["source_unchanged"] is False
    assert result["selected_checks_passed"] is False
    assert "source_changed_during_checks" in result["release_blockers"]


@pytest.mark.parametrize("tail,returncode,timed_out,expected", [
    ("test result: ok. 1 passed; 0 failed; 0 ignored;", 0, False, "passed"),
    (("test result: ok. 1 passed; 0 failed; 0 ignored;\n"
      "test result: ok. 1 passed; 0 failed; 0 ignored;"), 0, False, "passed"),
    (("test result: ok. 1 passed; 0 failed; 0 ignored;\n"
      "test result: ok. 0 passed; 0 failed; 0 ignored;"), 0, False, "failed"),
    ("test result: ok. 2 passed; 0 failed; 1 ignored;", 0, False, "failed"),
    ("test result: ok. 2 passed; 0 failed; 0 ignored;", 1, False, "failed"),
    ("test result: ok. 2 passed; 0 failed; 0 ignored;", 0, True, "failed"),
    ("runner exists but no results", 0, False, "failed"),
])
def test_m2_capabilities_native_summary_parser_not_runner_presence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tail: str,
    returncode: int, timed_out: bool, expected: str,
) -> None:
    # Parser fixtures only, not native execution or qualification evidence.
    monkeypatch.setattr(checker, "NATIVE_SUITES", ("contracts_and_leases",))
    def execute(command: list[str], *args: Any, **kwargs: Any) -> dict[str, Any]:
        if command[-1] == "--list":
            return {"returncode": 0, "timed_out": False,
                    "log_tail": "probe: test\n\n1 test, 0 benchmarks\n"}
        return {"returncode": returncode, "timed_out": timed_out, "log_tail": tail}

    monkeypatch.setattr(checker, "execute", execute)
    assert checker.run_native_gate(tmp_path, 10)["status"] == expected


@pytest.mark.parametrize("listing", ["", "0 tests, 0 benchmarks\n",
    "probe: test\n2 tests, 0 benchmarks\n",
    "probe: test\nprobe: test\n2 tests, 0 benchmarks\n"])
def test_m2_capabilities_native_zero_or_incomplete_discovery_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, listing: str,
) -> None:
    monkeypatch.setattr(checker, "NATIVE_SUITES", ("contracts_and_leases",))
    commands = []

    def execute(command: list[str], *args: Any, **kwargs: Any) -> dict[str, Any]:
        commands.append(command)
        return {"returncode": 0, "timed_out": False, "log_tail": listing}

    monkeypatch.setattr(checker, "execute", execute)
    assert checker.run_native_gate(tmp_path, 10)["status"] == "failed"
    assert len(commands) == 1
    assert commands[0][-1] == "--list"


@pytest.fixture(scope="module")
def measured_runtime() -> dict[str, Any]:
    # Actual native import, capabilities and M1 edit/replay/export fixture.
    result: dict[str, Any] = checker.run_runtime_gate(ROOT, 30)
    assert result["status"] == "passed", result
    return result


def test_m2_capabilities_actual_m1_measurements_are_scoped(
    measured_runtime: dict[str, Any],
) -> None:
    measured = measured_runtime["measured"]
    assert measured["m1_manifest"]["milestone"] == "M1"
    assert len(measured["m1_manifest"]["operations"]) == 8
    assert measured["m1_fixture"]["values"] == [2.0, 4.0, 6.0, 8.0]
    assert measured_runtime["versions_match"] is True
    assert measured_runtime["python_sources_match"] is True
    assert measured_runtime["native_source_equivalence"] == "not_established"
    assert measured["native_sha256"].startswith("sha256:")


@pytest.mark.parametrize("field", ["package_version", "distribution_version", "native_version",
                                   "core_version", "python_sources", "m1_fixture"])
def test_m2_capabilities_runtime_version_source_and_fixture_drift_fail(
    monkeypatch: pytest.MonkeyPatch, measured_runtime: dict[str, Any], field: str,
) -> None:
    # Mutated real measurements are adversarial parser fixtures, not proof.
    execution = copy.deepcopy(measured_runtime["execution"])
    measured = copy.deepcopy(measured_runtime["measured"])
    if field == "core_version":
        measured["m1_manifest"][field] = "stale"
    elif field == "python_sources":
        measured[field]["__init__.py"] = "sha256:" + "0" * 64
    elif field == "m1_fixture":
        measured[field]["source_preserved"] = False
    else:
        measured[field] = "stale"
    execution["log_tail"] = json.dumps(measured)
    monkeypatch.setattr(checker, "execute", lambda *args, **kwargs: execution)
    assert checker.run_runtime_gate(ROOT, 10)["status"] == "failed"


@pytest.mark.parametrize("timing_scope,expected_source,blocker", [
    (checker.STRICT_SCOPE, None, "requested_strict_physical_publication_is_unsupported"),
    (checker.TIMING_SCOPE, "sha256:" + "0" * 64, "source_differs_from_expected_build_source"),
])
def test_m2_capabilities_strict_contract_or_build_source_drift_refused_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    timing_scope: str, expected_source: str | None, blocker: str,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("unsupported scope or source drift must refuse before work")

    monkeypatch.setattr(checker, "execute", forbidden)
    result = checker.report(tmp_path, list(checker.REQUIRED), 10,
                            expected_source=expected_source, timing_scope=timing_scope)
    assert result["selected_checks_passed"] is False
    assert result["states"]["release_enabled"] is False
    assert blocker in result["release_blockers"]
    assert all(check["status"] == "not_run" for check in result["checks"].values())


def test_m2_capabilities_complete_prerequisites_and_observations_not_static_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This aggregation fixture does not establish host/integration qualification.
    monkeypatch.setattr(checker, "run_python_gate", lambda *args: {
        "status": "passed", "counts": {"collected": 3, "failed": 0, "errors": 0, "skipped": 0},
        "scope": "component_test_execution_only"})
    result = checker.report(tmp_path, ["reports"], 10,
                            expected_source=checker.source_identity(tmp_path))
    assert result["source_matches_expected"] is True
    assert result["component_observations"]["reports"]["counts"]["collected"] == 3
    assert set(result["release_prerequisites"]) == {
        *checker.AUTHORITY_BLOCKERS, *(f"gate:{name}" for name in checker.REQUIRED)}
    assert result["release_prerequisites"]["gate:reports"] == "passed"
    assert result["release_prerequisites"]["gate:vertical_slice"] == "not_run"
    assert result["states"]["release_enabled"] is False


def test_m2_capabilities_zero_exit_wheel_without_verifier_measurements_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel = tmp_path / "not-actually-a-wheel.whl"
    wheel.write_bytes(b"not evidence")
    monkeypatch.setattr(checker, "execute", lambda *args: {
        "returncode": 0, "timed_out": False, "log_tail": ""})
    result = checker.run_wheel_gate(ROOT, 10, wheel)
    assert result["status"] == "failed"
    assert result["versions_match"] is False
    assert result["fresh_build_verified"] is False


@pytest.mark.parametrize("timeout", [0, 91])
def test_m2_capabilities_api_rejects_unbounded_timeout(timeout: int) -> None:
    with pytest.raises(ValueError, match="1..90"):
        checker.report(ROOT, [], timeout)


def test_m2_capabilities_cli_default_and_release_mode_fail_closed() -> None:
    for options in ([], ["--require-release"]):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/check_m2_release.py"),
                                 *options], cwd=ROOT, capture_output=True, text=True,
                                timeout=15, check=False)
        assert result.returncode == 1, result.stderr
        data = json.loads(result.stdout)
        assert data["states"]["release_enabled"] is False
        assert data["release_blockers"]
    result = subprocess.run([sys.executable, str(ROOT / "scripts/check_m2_release.py"),
                             "--timeout", "0"], capture_output=True, text=True,
                            timeout=10, check=False)
    assert result.returncode == 2
