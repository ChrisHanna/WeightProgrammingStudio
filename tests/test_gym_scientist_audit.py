"""Deterministic supplied-result fixtures; no sandbox or research-success claim."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from neural_foundry import content_digest, gyms
from neural_foundry.gym_scientist import MAX_BYTES, GymResearchWorkflow


def audit_request() -> dict[str, Any]:
    cases = [{"id": "two", "group": "easy", "input": 2, "expected": 4},
             {"id": "seven", "group": "hard", "input": 7, "expected": 14}]
    policies = [{"id": role, "role": role}
                for role in ("valid", "invalid", "constant", "metadata_only", "weak", "strong")]
    results = []
    for case in cases:
        for policy in policies:
            for seed in (11, 22):
                for replay in (0, 1):
                    output = case["expected"] if policy["role"] in ("valid", "strong") else 0
                    results.append({"case": case["id"], "policy": policy["id"],
                                    "seed": seed, "replay": replay, "output": output,
                                    "oracle_output": case["expected"],
                                    "score": int(output == case["expected"]), "invalid_score": 0})
    return {"schema_version": 1, "objective": "Detect double-task scoring defects",
            "snapshot": "fixture:double-v1-unverified", "evidence_class": "development",
            "truth_provenance": "Independent supplied fixture labels, not authenticated",
            "cases": cases, "policies": policies, "seeds": [11, 22],
            "required_coverage": ["easy", "hard"],
            "scoring": {"minimum": 0, "bad_max": 0, "good_min": 1, "maximum": 1},
            "results": results}


def run(request: dict[str, Any]) -> dict[str, Any]:
    return GymResearchWorkflow(json.dumps(request)).run()


def test_gym_scientist_audit_standalone_deterministic_non_authoritative() -> None:
    request = audit_request()
    before = copy.deepcopy(request)
    workflow = GymResearchWorkflow(json.dumps(request))
    first = workflow.run()
    assert first == workflow.report() == run(request)
    assert request == before
    assert first["status"] == "no_issues_observed"
    assert first["policies"]["valid"]["passing"] == 8
    assert first["policies"]["invalid"]["passing"] == 0
    assert first["native_approval"] is False and first["weights_modified"] is False
    assert "unauthenticated" in first["limitations"][0]
    assert "supplied-result API does not execute policies" in first["limitations"][2]
    first["policies"].clear()
    assert workflow.report()["policies"]
    assert workflow.validate()["complete"] is True
    assert "results" not in workflow.plan()


def test_gym_scientist_audit_constant_positive_scorer() -> None:
    request = audit_request()
    for row in request["results"]:
        row["score"] = row["invalid_score"] = 1
    report = run(request)
    assert report["status"] == "issues_found"
    assert {"scorer_disagreement", "invalid_output_rewarded", "invalid_policy_not_separated",
            "constant_shortcut_rewarded", "metadata_only_shortcut_rewarded"} <= set(report["findings"])


def test_gym_scientist_audit_saturation_metadata_shortcut_coverage_and_duplicates() -> None:
    request = audit_request()
    for row in request["results"]:
        if row["policy"] in ("weak", "metadata_only"):
            row["output"] = row["oracle_output"]
            row["score"] = 1
    request["required_coverage"].append("unseen_condition")
    findings = run(request)["findings"]
    assert {"weak_policy_saturation", "metadata_only_shortcut_rewarded",
            "missing_required_coverage"} <= set(findings)
    request["cases"][1].update(input=2, expected=4)
    assert {"duplicate_task_content", "insufficient_distinct_inputs_or_truths"} <= set(
        run(request)["findings"]
    )


def test_gym_scientist_audit_oracle_replay_and_valid_policy_failures() -> None:
    request = audit_request()
    request["results"][0].update(output=9, oracle_output=9, score=0)
    assert {"oracle_disagreement", "replay_drift", "valid_policy_failed"} <= set(run(request)["findings"])
    request["results"].pop()
    assert run(request)["status"] == "inconclusive"
    request["results"].clear()
    report = run(request)
    assert report["status"] == "inconclusive" and report["complete"] is False


def test_gym_scientist_audit_conflicting_truth_and_missing_controls() -> None:
    request = audit_request()
    request["cases"][1]["input"] = 2
    request["policies"] = [p for p in request["policies"] if p["role"] != "invalid"]
    request["results"] = [r for r in request["results"] if r["policy"] != "invalid"]
    assert {"conflicting_fixture_truth", "missing_reference_controls"} <= set(
        run(request)["findings"]
    )


def test_gym_scientist_audit_command_rejects_bad_json() -> None:
    completed = subprocess.run([sys.executable, "-m", "neural_foundry.gym_scientist", "run"],
                               input='{"approved":true}', text=True, capture_output=True,
                               timeout=10, check=False)
    assert completed.returncode == 2
    assert json.loads(completed.stdout)["status"] == "invalid_input"


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(evidence_class="final"),
    lambda r: r.update(evidence_class="holdout"),
    lambda r: r.update(schema_version=True),
    lambda r: r.update(approved=True),
    lambda r: r.update(seeds=[True, 22]),
    lambda r: r.update(seeds=[11, 11]),
    lambda r: r.update(seeds=list(range(5))),
    lambda r: r["scoring"].update(bad_max=1),
    lambda r: r["results"][0].update(score=True),
    lambda r: r["results"][0].update(score=float("nan")),
    lambda r: r["results"][0].update(invalid_score=float("inf")),
    lambda r: r["results"][0].update(policy="undeclared"),
    lambda r: r["results"][0].update(replay=2),
    lambda r: r["results"].append(r["results"][0]),
    lambda r: r["cases"].append(r["cases"][0]),
    lambda r: r["cases"][0].update(expected=None),
    lambda r: r["policies"][0].update(role="file.py:run"),
])
def test_gym_scientist_audit_rejects_malformed_and_restricted_data(mutation: Any) -> None:
    request = audit_request()
    mutation(request)
    with pytest.raises(ValueError):
        run(request)


@pytest.mark.parametrize("payload", [b"x" * (MAX_BYTES + 1), '{"x":1,"x":2}',
                                    "[" * 2000 + "]" * 2000, lambda: None,
                                    {"callback": "not accepted"}])
def test_gym_scientist_audit_wire_bounds_and_no_callbacks(payload: Any) -> None:
    with pytest.raises(ValueError):
        GymResearchWorkflow(payload)


@pytest.mark.parametrize("command", ["plan", "validate", "run", "report"])
def test_gym_scientist_audit_commands_and_inert_code_data(command: str, tmp_path: Any) -> None:
    marker = tmp_path / "must-not-execute"
    request = audit_request()
    request["objective"] = f"__import__('pathlib').Path({str(marker)!r}).touch()"
    completed = subprocess.run([sys.executable, "-m", "neural_foundry.gym_scientist", command],
                               input=json.dumps(request), text=True, capture_output=True,
                               timeout=10, check=True)
    report = json.loads(completed.stdout)
    assert report["native_approval"] is False
    assert not marker.exists()
    assert request["results"][0]["output"] == 4


def test_gym_scientist_audit_rejected_null_probe_is_not_a_fabricated_zero() -> None:
    request = audit_request()
    for row in request["results"]:
        row["invalid_score"] = None
    assert run(request)["status"] == "no_issues_observed"
    request["results"][0]["invalid_score"] = True
    with pytest.raises(ValueError):
        run(request)


def test_gym_scientist_audit_counterexamples_are_bounded_and_locatable() -> None:
    request = audit_request()
    for row in request["results"]:
        row["invalid_score"] = 1
    report = run(request)
    assert report["counterexample_count"] == len(request["results"])
    assert len(report["counterexamples"]) == 16
    assert report["counterexamples_truncated"] is True
    for counterexample in report["counterexamples"]:
        assert counterexample["defects"] == ["invalid_output_rewarded"]
        assert any(all(row[k] == counterexample[k] for k in ("case", "policy", "seed", "replay"))
                   for row in request["results"])


def test_native_reference_audit_never_falls_back_to_supplied_observations(monkeypatch: Any) -> None:
    from neural_foundry import _native
    from neural_foundry.gym_scientist import run_reference_audit

    monkeypatch.delattr(_native, "run_gym_reference_audit", raising=False)
    invalid: Any = None
    with pytest.raises(ValueError, match="UNTRUSTED_GYM_EXECUTION_DISABLED"):
        run_reference_audit(invalid, "{}", [], [], "development", "{}", "/unused")
    with pytest.raises(NotImplementedError, match="cannot substitute"):
        run_reference_audit(invalid, "{}", [invalid], ["valid"], "development", "{}",
                            "/unused", enabled=True)
    for exposure in ("holdout", "final"):
        with pytest.raises(ValueError, match="exposed JSON plans"):
            run_reference_audit(invalid, "{}", [invalid], ["valid"], exposure, "{}",
                                "/unused", enabled=True)


@pytest.mark.parametrize("scorer", ["correct", "constant_positive", "reject_null"])
def test_gym_scientist_audit_real_sandbox_to_cli(scorer: str, tmp_path: Path) -> None:
    # The controller fixture supplies observations read from its own CAS, checking
    # their identities. Product code gets only JSON, never paths or callbacks.
    # These are distinct policy-source snapshots, NOT a qualified generic runner
    # executing arbitrary reference policies against one approved gym snapshot.
    from test_gym_validation import GOOD, fixture

    request = audit_request()
    request["snapshot"] = "fixture:matched-double-policy-source-variants"
    request["results"] = []
    marker = tmp_path / "controller-import-must-not-run"
    untouched_weights = tmp_path / "weights.safetensors"
    untouched_weights.write_bytes(b"fixture sentinel, never loaded as model weights")
    before = untouched_weights.read_bytes()
    for policy in request["policies"]:
        directory = tmp_path / policy["id"]
        directory.mkdir()
        code = GOOD
        if policy["role"] not in ("valid", "strong"):
            code = code.replace("return x*2", "return 0", 1)
        if scorer == "constant_positive":
            code = code.replace("float(out==reference)", "1.0")
        elif scorer == "reject_null":
            code = code.replace("def score(x, out, reference): return float(out==reference)",
                                "def score(x, out, reference):\n"
                                " if out is None: raise ValueError('null rejected')\n"
                                " return float(out==reference)")
        code = (f"from pathlib import Path\ntry: Path({str(marker)!r}).touch()\n"
                "except OSError: pass\n" + code)
        snapshot, plan, sandbox, _, _ = fixture(directory, code)
        validation = gyms.validate(snapshot, plan, sandbox, enabled=True)
        native_report = json.loads(validation.report_bytes())
        assert content_digest(validation.report_bytes()) == validation.receipt_id
        expected_pass = scorer != "constant_positive" and policy["role"] in ("valid", "strong")
        assert native_report["status"] == ("passed" if expected_pass else "rejected"), native_report
        identity = native_report["observations"]
        assert isinstance(identity, str) and identity.startswith("sha256:")
        raw = (directory / "store" / "blobs" / identity.split(":")[1]).read_bytes()
        assert len(raw) <= MAX_BYTES and content_digest(raw) == identity
        observations = json.loads(raw)
        assert observations["schema_version"] == 1
        assert len(observations["samples"]) == 8
        policy["provenance"] = {"snapshot": snapshot.snapshot_id,
                                "validation": validation.receipt_id, "observations": identity}
        for sample in observations["samples"]:
            for call in ("evaluation", "oracle", "observed_score"):
                assert sample[call]["ok"] is True
            invalid = sample["invalid_score"]
            assert invalid["ok"] is (scorer != "reject_null")
            request["results"].append({
                "case": plan["cases"][sample["case"]]["id"], "policy": policy["id"],
                "seed": sample["seed"], "replay": sample["repeat"],
                "output": sample["evaluation"]["value"],
                "oracle_output": sample["oracle"]["value"],
                "score": sample["observed_score"]["value"],
                "invalid_score": invalid["value"] if invalid["ok"] else None,
            })
    assert not marker.exists() and untouched_weights.read_bytes() == before
    completed = subprocess.run([sys.executable, "-m", "neural_foundry.gym_scientist", "run"],
                               input=json.dumps(request), text=True, capture_output=True,
                               timeout=10, check=True)
    report = json.loads(completed.stdout)
    assert report == run(request)
    assert report["native_approval"] is False and report["weights_modified"] is False
    assert report["complete"] is True
    assert report["reference_provenance"] == {p["id"]: p["provenance"] for p in request["policies"]}
    if scorer == "constant_positive":
        assert report["status"] == "issues_found"
        assert {"scorer_disagreement", "invalid_output_rewarded", "invalid_policy_not_separated",
                "constant_shortcut_rewarded", "metadata_only_shortcut_rewarded"} <= set(report["findings"])
        assert report["counterexamples"]
    else:
        assert report["status"] == "no_issues_observed", report
        assert report["policies"]["valid"]["correct"] == 8
        assert report["policies"]["invalid"]["passing"] == 0
    assert not marker.exists() and untouched_weights.read_bytes() == before


def native_reference_fixture(tmp_path: Path, code: str | None = None) -> tuple[Any, ...]:
    import shutil

    from test_gym_validation import GOOD, fixture

    snapshot, plan, sandbox, source, draft = fixture(tmp_path, GOOD if code is None else code)
    handles = []
    for role, expression in (("valid", "x*2"), ("invalid", "1"),
                             ("constant", "0"), ("metadata_only", "seed")):
        directory = tmp_path / role
        shutil.copytree(source, directory)
        # Policy scorer/oracle must NEVER substitute for the unchanged gym's.
        (directory / "gym.py").write_text(
            f"def evaluate(x,seed): return {expression}\n"
            "def oracle(x): raise AssertionError('policy oracle used')\n"
            "def score(x,out,reference): return 1.0\n")
        declaration = copy.deepcopy(draft)
        declaration["name"] = role
        handles.append(gyms.import_local(directory, declaration, tmp_path / "store"))
    return snapshot, plan, sandbox, source, draft, handles


def native_audit(snapshot: Any, plan: dict[str, Any], sandbox: Path,
                 policies: list[Any], roles: list[str] | None = None) -> Any:
    from neural_foundry.gym_scientist import run_reference_audit

    return run_reference_audit(snapshot, json.dumps(plan), policies,
                               ["valid", "invalid", "constant", "metadata_only"]
                               if roles is None else roles, "development",
                               # Aggregate absolute time includes controller
                               # overhead; frozen per-job watchdogs are unchanged.
                               b'{"max_executions":8,"cpu_ms":16000,"wall_ms":45000,"storage_bytes":268435456}',
                               str(sandbox), enabled=True)


@pytest.mark.parametrize("scorer", ["correct", "positive", "zero", "oracle"])
def test_native_reference_audit_same_immutable_gym_execution(tmp_path: Path, scorer: str) -> None:
    from neural_foundry import NativeError, _native
    from test_gym_validation import GOOD

    code = GOOD
    if scorer in ("positive", "zero"):
        code = code.replace("float(out==reference)", "1.0" if scorer == "positive" else "0.0")
    elif scorer == "oracle":
        code = code.replace("def oracle(x): return x*2", "def oracle(x): return 4")
    marker = tmp_path / "host-must-not-be-written"
    code = (f"from pathlib import Path\ntry: Path({str(marker)!r}).touch()\n"
            "except OSError: pass\n" + code)
    snapshot, plan, sandbox, source, _, policies = native_reference_fixture(tmp_path, code)
    identity = snapshot.snapshot_id
    manifest = snapshot.manifest_bytes()
    weights = tmp_path / "weights.safetensors"
    weights.write_bytes(b"not loaded or modified")
    for directory in [source, *(tmp_path / role for role in (
        "valid", "invalid", "constant", "metadata_only"))]:
        (directory / "gym.py").write_text("raise AssertionError('live source imported')")
    handle = native_audit(snapshot, plan, sandbox, policies)
    assert isinstance(handle, _native.GymReferenceAuditHandle)
    report = json.loads(handle.report_bytes())
    assert handle.receipt_id == content_digest(handle.report_bytes())
    assert report["snapshot"] == identity == snapshot.snapshot_id
    assert snapshot.manifest_bytes() == manifest
    assert report["requested_policies"] == [[p.snapshot_id, r] for p, r in zip(
        policies, ("valid", "invalid", "constant", "metadata_only"))]
    assert [p["snapshot"] for p in report["policies"]] == [p.snapshot_id for p in policies]
    assert report["executions"] == 8 and report["cpu_ms"] > 0 and report["wall_ms"] > 0
    assert report["native_approval"] is False and report["brokered_model_calls"] == 0
    assert "not_validation_approval_or_scientific_acceptance" in report["scope"]
    assert report["plan"] == content_digest(_native.canonical_json(json.dumps(plan).encode()))
    assert all(report[k].startswith("sha256:") for k in ("protocol", "runtime_environment"))
    for digest in [handle.receipt_id, *report["sandbox_receipts"], *report["observations"]]:
        raw = (tmp_path / "store" / "blobs" / digest[7:]).read_bytes()
        assert content_digest(raw) == digest
    for digest in report["sandbox_receipts"]:
        receipt = json.loads((tmp_path / "store" / "blobs" / digest[7:]).read_bytes())
        assert receipt["cleanup_confirmed"] is True and receipt["status"] == "completed"
    if scorer == "correct":
        assert report["status"] == "probe_passed" and report["findings"] == []
        assert report["audit_complete"] is False
        assert report["policies"][0]["correct"] == 8
        assert all(p["passing"] == 0 for p in report["policies"][1:])
    else:
        assert report["status"] == "rejected"
        assert ("oracle_disagreement" if scorer == "oracle" else
                "known_good_score" if scorer == "zero" else "invalid_output_rewarded"
                ) in report["findings"]
    with pytest.raises(TypeError):
        type(handle)()
    untyped: Any = handle
    with pytest.raises(AttributeError):
        untyped.receipt_id = "forged"
    with pytest.raises((TypeError, NativeError)):
        gyms.approve(snapshot, untyped, "audit-is-not-validation")
    assert not marker.exists() and weights.read_bytes() == b"not loaded or modified"


def test_native_reference_audit_malicious_policy_same_gym_is_bounded(tmp_path: Path) -> None:
    snapshot, plan, sandbox, source, draft, _ = native_reference_fixture(tmp_path)
    marker = tmp_path / "host-marker"
    (source / "gym.py").write_text(
        f"from pathlib import Path\ntry: Path({str(marker)!r}).touch()\nexcept OSError: pass\n"
        "def evaluate(x,seed):\n while True: pass\n"
        "def oracle(x): return 0\ndef score(x,out,reference): return 1\n")
    policy = gyms.import_local(source, draft, tmp_path / "store")
    plan["limits"].update(cpu_ms=200, wall_ms=300)
    report = json.loads(native_audit(snapshot, plan, sandbox, [policy], ["invalid"]).report_bytes())
    assert report["snapshot"] == snapshot.snapshot_id and report["status"] == "inconclusive"
    assert report["executions"] == 1 and "bounded_execution_incomplete" in report["findings"]
    assert report["native_approval"] is False and not marker.exists()


def test_native_reference_audit_boundary_denials_and_cas_tamper(tmp_path: Path) -> None:
    from neural_foundry import NativeError, _native
    from test_gym_validation import GOOD, fixture

    snapshot, plan, sandbox, _, _, policies = native_reference_fixture(tmp_path)
    execute: Any = _native.run_gym_reference_audit
    budget = '{"max_executions":8,"cpu_ms":16000,"wall_ms":16000,"storage_bytes":268435456}'
    args = [snapshot, json.dumps(plan), policies, ["valid", "invalid", "constant", "metadata_only"],
            "selection", budget, str(sandbox)]
    with pytest.raises(NativeError, match="UNTRUSTED_GYM_EXECUTION_DISABLED"):
        execute(*args)
    for index, value in ((0, {}), (1, {}), (2, [None]), (2, policies * 3),
                         (2, [policies[0]] * 4), (3, ["valid"]), (3, ["unknown"] * 4),
                         (4, "holdout"), (4, "final"), (5, budget[:-1] + ',"extra":1}'),
                         (5, budget[:-1] + ',"cpu_ms":16000}'),
                         (5, '{"max_executions":1,"cpu_ms":16000,"wall_ms":16000,"storage_bytes":67108864}'),
                         (1, json.dumps({**plan, "snapshot": policies[0].snapshot_id})),
                         (1, b" " * 65537), (5, " " * 4097)):
        changed = args.copy()
        changed[index] = value
        with pytest.raises((NativeError, TypeError)):
            execute(*changed, enabled=True)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    other, _, _, _, _ = fixture(foreign, GOOD + "\n# foreign snapshot\n")
    changed = args.copy()
    changed[2] = [other, *policies[1:]]
    with pytest.raises(NativeError):
        execute(*changed, enabled=True)
    readonly = gyms.restore(snapshot.snapshot_id, tmp_path / "store")
    changed = args.copy()
    changed[0] = readonly
    with pytest.raises(NativeError, match="read-only"):
        execute(*changed, enabled=True)
    # Damage a policy asset after intake: a live handle cannot bypass fresh CAS verification.
    manifest = json.loads(policies[0].manifest_bytes())
    blob = tmp_path / "store" / "blobs" / manifest["assets"]["gym.py"]["digest"][7:]
    blob.chmod(0o600)
    blob.write_bytes(b"raise AssertionError('tampered policy')")
    with pytest.raises(NativeError):
        execute(*args, enabled=True)
    assert not list(sandbox.rglob("process.json"))
