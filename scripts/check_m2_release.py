#!/usr/bin/env python3
"""Run bounded local M2 checks; never turn component receipts into release authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
COMPLIANCE = tuple(f"CG-{n:02}" for n in (*range(1, 9), 11, 12, 14))
# Source-controlled prerequisites, NOT input flags or test-result assertions. These
# cannot be cleared by a passing runner. Change only with reviewed owner evidence.
AUTHORITY_BLOCKERS = (
    "T038_bounded_work_cancellation_cleanup_and_independent_review",
    "T039_authoritative_atomic_promotion_and_reviewed_release",
    "T041_integrated_authorized_custom_gym_dispatch",
    "T043_integrated_mixed_five_gym_evidence",
    "fresh_installed_wheel_and_matching_native_build",
)
INCOMPLETE = (
    "nemo_gym_adapter", "prime_intellect_verifiers_adapter", "gymnasium_adapter",
    "autonomous_research_agents", "rich_evolution_engine", "general_transfer",
    "structural_edits", "quantized_edits", "confidential_deployment",
    "gpu_qualification", "serving_qualification",
)
# Explicit files, not broad -k expressions that may silently collect nothing.
# These are component checks. Even all green does not establish integration.
SUITES = {
    "capabilities": ("tests/test_m2_capabilities.py",),
    "imports": ("tests/test_package_smoke.py",),
    "m1_executor": ("tests/test_core_conformance.py", "tests/test_cpu_differential.py"),
    "gym_intake": ("tests/test_custom_gym_import.py", "tests/test_gym_validation.py"),
    "gym_portfolio": ("tests/test_portfolio_coverage.py", "tests/test_gym_reference_episode.py"),
    "worker_broker": ("tests/test_worker_broker.py",),
    "sandbox": ("tests/test_sandbox_isolation.py",),
    "trusted_worker": ("tests/test_model_worker.py", "tests/test_reload_promotion.py"),
    "contracts": ("tests/test_manual_workflow.py",),
    "splits": ("tests/test_split_broker.py", "tests/test_legacy_split_intake.py"),
    "controls": ("tests/test_paired_controls.py", "tests/test_paired_statistics.py"),
    "tensor_scope": ("tests/test_core_conformance.py",),
    "measurements": ("tests/test_measurement_protocol.py",),
    "reports": ("tests/test_status_reports.py",),
    "methods_costs": ("tests/test_repair_benchmark.py",),
    "vertical_slice": ("tests/test_m2_vertical_slice.py",),
    "gym_audit": ("tests/test_gym_scientist_audit.py", "tests/test_t042_trusted_audit.py"),
}
NATIVE_SUITES = (
    "contracts_and_leases", "budgets", "cancellation_recovery",
    "acceptance_gate_panel", "acceptance_gate_trusted_fixture", "promotion_atomic",
    "fixture_supervisor_component", "manual_workflow", "manual_workflow_authorization",
    "model_worker_reload", "reference_policy_execution", "status_reports",
)
REQUIRED = (*SUITES, "runtime_identity", "native_contracts", "isolated_wheel")
TIMING_SCOPE = "operational_cancellation"
STRICT_SCOPE = "strict_physical_publication"

# Actual installed M1 measurements, not hardcoded successes or historical receipts.
# No research stack, runtime qualification, model download or code generation.
RUNTIME_PROBE = """
import hashlib,json
from importlib.metadata import version
from pathlib import Path
import neural_foundry as nf
from neural_foundry import _native
from neural_foundry.demo import run
package=Path(nf.__file__).parent
fixture=run()
print(json.dumps({
    'package_version':nf.__version__, 'distribution_version':version('neural-foundry'),
    'native_version':_native.version(), 'm1_manifest':nf.capabilities(),
    'native_sha256':'sha256:'+hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest(),
    'python_sources':{p.relative_to(package).as_posix():
        'sha256:'+hashlib.sha256(p.read_bytes()).hexdigest() for p in package.rglob('*.py')},
    'm1_fixture':{key:fixture[key] for key in ('values','source_preserved',
        'snapshot_parity','candidate','replayed_candidate','promotion_status')}
}))
"""


def digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def source_identity(root: Path) -> str:
    """Bind all relevant source/config bytes, excluding build/cache products."""
    hashed = hashlib.sha256()
    paths = [root / name for name in ("Cargo.toml", "Cargo.lock", "pyproject.toml",
                                      "requirements-dev.lock", "rust-toolchain.toml",
                                      "LICENSE", "NOTICE", "README.md")]
    for directory in ("crates", "python", "scripts", "tests", "protocols", "licenses", "docs",
                      ".github", ".kiro"):
        for parent, directories, files in os.walk(root / directory):
            directories[:] = sorted(d for d in directories if d not in
                                     {"__pycache__", ".pytest_cache", "target", ".venv"})
            paths.extend(Path(parent) / name for name in files
                         if Path(name).suffix in {".rs", ".py", ".toml", ".lock", ".md",
                                                 ".yml", ".yaml", ".json", ".c", ".h",
                                                 ".pyi", ".txt"})
    for path in sorted(set(paths)):
        if path.is_file():
            name = path.relative_to(root).as_posix().encode()
            hashed.update(len(name).to_bytes(8, "big") + name)
            hashed.update(hashlib.sha256(path.read_bytes()).digest())
    return "sha256:" + hashed.hexdigest()


def run_runtime_gate(root: Path, timeout: int) -> dict[str, Any]:
    result = execute([sys.executable, "-I", "-c", RUNTIME_PROBE], root, timeout,
                     output_limit=65536)
    if result["returncode"] != 0 or result["timed_out"]:
        return {"status": "failed", "reason": "installed runtime probe failed",
                "execution": result}
    try:
        measured = json.loads(result["log_tail"])
        cargo = tomllib.loads((root / "Cargo.toml").read_text())["workspace"]["package"]["version"]
        python = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
        versions = [cargo, python, measured["package_version"], measured["native_version"],
                    measured["distribution_version"], measured["m1_manifest"]["core_version"]]
        version_match = len(set(versions)) == 1
        source = root / "python/neural_foundry"
        sources = {p.relative_to(source).as_posix(): digest(p.read_bytes())
                   for p in source.rglob("*.py")}
        python_match = bool(sources) and sources == measured["python_sources"]
        fixture = measured["m1_fixture"]
        fixture_passed = (fixture["values"] == [2.0, 4.0, 6.0, 8.0]
                          and fixture["source_preserved"] is True
                          and fixture["snapshot_parity"] is True
                          and fixture["candidate"] == fixture["replayed_candidate"]
                          and fixture["promotion_status"] == "not_evaluated")
        passed = version_match and python_match and fixture_passed
        return {"status": "passed" if passed else "failed", "measured": measured,
                "source_versions": {"cargo": cargo, "python": python},
                "versions_match": version_match, "python_sources_match": python_match,
                "native_source_equivalence": "not_established",
                "scope": "installed_m1_fixture_and_capability_measurements_not_m2_release",
                "execution": result}
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {"status": "failed", "reason": f"invalid runtime identity: {error}",
                "execution": result}


def junit_counts(path: Path) -> dict[str, int]:
    """Count actual cases, not spoofable summary attributes or pytest text."""
    tree = ET.parse(path)
    if tree.getroot().tag not in {"testsuite", "testsuites"}:
        raise ValueError("unexpected JUnit root")
    cases = list(tree.iter("testcase"))
    return {
        "collected": len(cases),
        "failed": sum(case.find("failure") is not None for case in cases),
        "errors": sum(case.find("error") is not None for case in cases),
        "skipped": sum(case.find("skipped") is not None for case in cases),
    }


def counts_pass(counts: dict[str, int]) -> bool:
    return (counts["collected"] > 0 and
            not any(counts[key] for key in ("failed", "errors", "skipped")))


def execute(command: list[str], root: Path, timeout: int, *,
            output_limit: int = 6000) -> dict[str, Any]:
    """Bound the whole process group, including child workers, on POSIX."""
    env = dict(os.environ, PYTEST_ADDOPTS="", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               CARGO_NET_OFFLINE="true")
    env.setdefault("CARGO_TARGET_DIR", "/tmp/nf-m2-check")
    env.setdefault("CARGO_BUILD_JOBS", "2")
    started = time.monotonic()
    with tempfile.TemporaryFile() as output:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        output.seek(0)
        # Hash the complete log while retaining only a bounded tail in the report.
        hashed = hashlib.sha256()
        tail = b""
        size = 0
        while chunk := output.read(65536):
            hashed.update(chunk)
            size += len(chunk)
            tail = (tail + chunk)[-output_limit:]
    return {"command": command, "returncode": process.returncode, "timed_out": timed_out,
            "elapsed_seconds": round(time.monotonic() - started, 3), "log_bytes": size,
            "log_sha256": "sha256:" + hashed.hexdigest(),
            "log_tail": tail.decode(errors="replace")}


def run_python_gate(name: str, root: Path, timeout: int) -> dict[str, Any]:
    files = SUITES[name]
    missing = [file for file in files if not (root / file).is_file()]
    if missing:
        return {"status": "missing", "reason": "missing required suite", "files": missing}
    with tempfile.TemporaryDirectory(prefix="nf-m2-junit-") as temporary:
        xml = Path(temporary) / "results.xml"
        command = [sys.executable, "-m", "pytest", "-q", "--color=no", "-o", "xfail_strict=true",
                   *files, f"--junitxml={xml}"]
        execution = execute(command, root, timeout)
        try:
            counts = junit_counts(xml)
        except (OSError, ET.ParseError, ValueError) as error:
            return {"status": "failed", "reason": f"missing/invalid JUnit: {error}",
                    "execution": execution}
        cases = list(ET.parse(xml).iter("testcase"))
        file_counts = {file: sum(Path(file).stem in case.get("classname", "").split(".")
                                 for case in cases) for file in files}
        # An explicit xfail(strict=False) overrides the ini option. Its XPASS
        # is still not a clean required-suite result, even with exit zero.
        xpassed = bool(re.search(r"\b(?:XPASS|\d+ xpassed)\b", execution["log_tail"]))
        passed = (execution["returncode"] == 0 and not execution["timed_out"] and
                  counts_pass(counts) and all(file_counts.values()) and not xpassed)
        return {"status": "passed" if passed else "failed", "counts": counts,
                "file_counts": file_counts, "unexpected_passes": xpassed,
                "scope": "component_test_execution_only", "execution": execution}


def run_native_gate(root: Path, timeout: int) -> dict[str, Any]:
    # Discover an entire explicit binary, then bound each exact test. Long crash
    # matrices need not fit in one 90s command; zero/ignored tests still fail.
    executions = []
    for suite in NATIVE_SUITES:
        command = [str(Path.home() / ".cargo/bin/cargo"), "test", "--offline", "--locked",
                   "-p", "nf-experiments", "--test", suite, "--"]
        discovery = execute([*command, "--list"], root, timeout, output_limit=65536)
        names = re.findall(r"^([\w:]+): test$", discovery["log_tail"], re.MULTILINE)
        totals = re.findall(r"^(\d+) tests?, 0 benchmarks?$", discovery["log_tail"], re.MULTILINE)
        passed = (discovery["returncode"] == 0 and not discovery["timed_out"]
                  and bool(names) and len(set(names)) == len(names)
                  and totals == [str(len(names))])
        tests = []
        if passed:
            for name in names:
                result = execute([*command, name, "--exact", "--test-threads=1"], root, timeout)
                matches = re.findall(
                    r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;",
                    result["log_tail"])
                ok = (result["returncode"] == 0 and not result["timed_out"] and
                      bool(matches) and matches[-1] == ("1", "0", "0"))
                tests.append({"test": name, "status": "passed" if ok else "failed",
                              "execution": result})
                if not ok:
                    passed = False
                    break
        executions.append({"suite": suite, "status": "passed" if passed else "failed",
                           "collected": len(names), "discovery": discovery, "tests": tests})
        if not passed:
            break
    return {"status": "passed" if len(executions) == len(NATIVE_SUITES) and
            all(item["status"] == "passed" for item in executions) else "failed",
            "scope": "component_test_execution_only", "executions": executions}


def run_wheel_gate(root: Path, timeout: int, wheel: Path | None) -> dict[str, Any]:
    if wheel is None:
        return {"status": "missing", "reason": "explicit freshly built wheel required"}
    wheel = wheel.resolve(strict=True)
    identity = digest(wheel.read_bytes())
    with tempfile.TemporaryDirectory(prefix="nf-m2-wheel-") as temporary:
        result = execute([sys.executable, str(root / "scripts/verify_wheel.py"),
                          str(wheel), temporary], root, timeout)
    unchanged = digest(wheel.read_bytes()) == identity
    proof: dict[str, Any] = {}
    versions_match = False
    try:
        proof = json.loads(result["log_tail"].splitlines()[-1])
        source_version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
        cargo_version = tomllib.loads((root / "Cargo.toml").read_text())["workspace"]["package"]["version"]
        versions_match = (proof["proof"]["version"] == source_version == cargo_version
                          == proof["proof"]["result"]["capabilities"]["core_version"])
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        pass
    passed = (result["returncode"] == 0 and not result["timed_out"] and unchanged
              and versions_match and proof.get("module_entrypoint_passed") is True)
    return {"status": "passed" if passed else "failed", "wheel_sha256": identity,
            "versions_match": versions_match, "measured": proof,
            "scope": "isolated_wheel_import_and_m1_demo_not_m2_integration",
            "fresh_build_verified": False, "execution": result}


def report(root: Path, selected: list[str], timeout: int,
           wheel: Path | None = None, *, expected_source: str | None = None,
           timing_scope: str = TIMING_SCOPE) -> dict[str, Any]:
    if not 1 <= timeout <= 90:
        raise ValueError("timeout must be in 1..90")
    if timing_scope not in (TIMING_SCOPE, STRICT_SCOPE):
        raise ValueError("unknown timing scope")
    if set(selected) - set(REQUIRED):
        raise ValueError("unknown gate")
    before = source_identity(root)
    source_matches = expected_source is None or expected_source == before
    supported_scope = timing_scope == TIMING_SCOPE
    checks: dict[str, Any] = {}
    for name in REQUIRED:
        if name not in selected:
            checks[name] = {"status": "not_run", "reason": "no execution evidence this invocation"}
            continue
        if not source_matches or not supported_scope:
            checks[name] = {"status": "not_run", "reason": "source drift or unsupported timing scope"}
            continue
        try:
            if name == "runtime_identity":
                checks[name] = run_runtime_gate(root, timeout)
            elif name in SUITES:
                checks[name] = run_python_gate(name, root, timeout)
            elif name == "native_contracts":
                checks[name] = run_native_gate(root, timeout)
            else:
                checks[name] = run_wheel_gate(root, timeout, wheel)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            checks[name] = {"status": "failed", "reason": str(error)}
    after = source_identity(root)
    unchanged = before == after
    missing = [name for name, result in checks.items() if result["status"] != "passed"]
    blockers = [*AUTHORITY_BLOCKERS, *(f"gate:{name}" for name in missing)]
    if not unchanged:
        blockers.append("source_changed_during_checks")
    if not source_matches:
        blockers.append("source_differs_from_expected_build_source")
    if not supported_scope:
        blockers.append("requested_strict_physical_publication_is_unsupported")
    host_checks = ("sandbox", "trusted_worker", "splits")
    host_observed = unchanged and source_matches and all(
        checks[name]["status"] == "passed" for name in host_checks)
    # These are observed executions, never a component-to-release promotion rule.
    observations = {name: {"status": check["status"],
                           "scope": check.get("scope", "no_measurement"),
                           "counts": check.get("counts"),
                           "executions": check.get("executions")}
                    for name, check in checks.items()}
    prerequisites = {name: "not_established" for name in AUTHORITY_BLOCKERS}
    prerequisites.update({f"gate:{name}": check["status"] for name, check in checks.items()})
    return {
        "schema_version": 1, "kind": "m2_gate_report", "milestone": "M2",
        "evidence_scope": "local_observations_not_release_credentials",
        "source_identity_before": before, "source_identity_after": after,
        "source_unchanged": unchanged, "expected_source_identity": expected_source,
        "source_matches_expected": source_matches,
        "timing_policy": {"requested_scope": timing_scope, "supported": supported_scope,
                          "m2_requires": "finite_budgets_cancellation_incomplete_and_cleanup_disclosure",
                          "strict_physical_publication_required_by_m2": False,
                          "strict_physical_publication": "unsupported_refuse_if_requested",
                          "cleanup_limitations": "must_be_reported_not_assumed_complete"},
        "host": {"system": platform.system(), "machine": platform.machine(),
                 "python": sys.version, "qualification_scope": "this_invocation_only"},
        "states": {"implemented": "component_subset",
                   "host_qualified": "observed_component_tests_only" if host_observed else "not_qualified",
                   "workflow_integrated": "incomplete", "release_enabled": False,
                   "scientific_evidence": "not_established"},
        "required_compliance_gates": list(COMPLIANCE),
        "scope_limits": {"native_fixture_only": True, "strict_fsync_publication": "unsupported",
                         "independent_engineering_review": "not_established",
                         "mixed_five_gym_dispatch": "not_established"},
        "incomplete_advanced_capabilities": list(INCOMPLETE),
        "checks": checks, "component_observations": observations,
        "release_prerequisites": prerequisites, "release_blockers": blockers,
        "selected_checks_passed": bool(selected) and unchanged and source_matches and supported_scope and
            all(checks[name]["status"] == "passed" for name in selected),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", action="append", choices=(*REQUIRED, "all"), default=[])
    parser.add_argument("--timeout", type=int, default=90, help="seconds per command, 1..90")
    parser.add_argument("--expected-source", help="source digest frozen before the local build")
    parser.add_argument("--timing-scope", choices=(TIMING_SCOPE, STRICT_SCOPE), default=TIMING_SCOPE,
                        help="strict physical-publication contracts are refused before execution")
    parser.add_argument("--wheel", type=Path, help="offline wheel verifier input; not release authority")
    parser.add_argument("--require-release", action="store_true",
                        help="exit nonzero unless every release prerequisite is established")
    args = parser.parse_args()
    if not 1 <= args.timeout <= 90:
        parser.error("--timeout must be in 1..90")
    selected = list(REQUIRED) if "all" in args.gate else list(dict.fromkeys(args.gate))
    result = report(ROOT, selected, args.timeout, args.wheel,
                    expected_source=args.expected_source, timing_scope=args.timing_scope)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require_release or not selected:
        return 0 if result["states"]["release_enabled"] else 1
    return 0 if result["selected_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
