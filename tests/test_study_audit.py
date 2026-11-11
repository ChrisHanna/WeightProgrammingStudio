"""Custodial audit and strict-claim contracts on newly generated software fixtures only."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from neural_foundry.study.audits import validate_aggregate
from neural_foundry.study.claims import ClaimSpent, StateBusy, StrictClaims, exclusive_state
from neural_foundry.study.client import CustodianRefused
from neural_foundry.study.custodian import Custodian, Refused, canonical
from neural_foundry.study.plan import digest_bytes
from test_study_harness import client_for, make_plan, make_study


def audit_study(tmp_path: Path):
    study = make_study(tmp_path)
    document = {"rows": [{"text": "Independent public training text for an unrelated exercise."}]}
    path = tmp_path / "training.json"
    path.write_bytes(canonical(document))
    spec = {"plugin": "text_overlap_v1",
            "config": {"text_path": ["text"], "thresholds": {
                "exact_input_matches": 0, "distinct_shared_5grams": 0}},
            "training_rows_digest": digest_bytes(canonical(document)), "artifact_path": str(path)}
    study["manifest"]["audit"] = spec
    return study, spec


def test_audit_wire_releases_only_validated_aggregates_and_spends_start(tmp_path: Path) -> None:
    study, spec = audit_study(tmp_path)
    with client_for(study) as client:
        client.freeze(study["manifest"])
        receipt = client.audit(spec)
        validate_aggregate(receipt["aggregate"])
        assert receipt["aggregate"]["passed"] is True
        assert receipt["aggregate"]["counts"]["exact_input_matches"] == 0
        assert receipt["native_ledger_bridge"] is False and receipt["signed"] is False
        assert receipt["receipt_digest"] == digest_bytes(canonical({
            key: value for key, value in receipt.items() if key != "receipt_digest"}))
        assert spec["artifact_path"] not in json.dumps(receipt)
        assert "Independent public training text" not in json.dumps(receipt)
        with pytest.raises(CustodianRefused) as refused:
            client.audit(spec)
        assert refused.value.code == "ALREADY_AUDITED"
    with client_for(study) as restarted:
        with pytest.raises(CustodianRefused) as refused:
            restarted.audit(spec)
        assert refused.value.code == "ALREADY_AUDITED"


def test_bad_training_digest_consumes_the_audit_without_a_retry(tmp_path: Path) -> None:
    study, spec = audit_study(tmp_path)
    with client_for(study) as client:
        client.freeze(study["manifest"])
        Path(spec["artifact_path"]).write_text('{"rows":[{"text":"changed"}]}')
        with pytest.raises(CustodianRefused) as refused:
            client.audit(spec)
        assert refused.value.code == "AUDIT_FAILED"
    with client_for(study) as client:
        with pytest.raises(CustodianRefused) as refused:
            client.audit(spec)
        assert refused.value.code == "ALREADY_AUDITED"


def test_changed_audit_configuration_is_refused_before_spending(tmp_path: Path) -> None:
    study, spec = audit_study(tmp_path)
    with client_for(study) as client:
        client.freeze(study["manifest"])
        changed = {**spec, "training_rows_digest": "sha256:" + "0" * 64}
        with pytest.raises(CustodianRefused) as refused:
            client.audit(changed)
        assert refused.value.code == "AUDIT_CHANGED"
        assert client.audit(spec)["aggregate"]["passed"] is True


def test_stale_custodian_objects_cannot_double_query(tmp_path: Path) -> None:
    study, _ = audit_study(tmp_path)
    with client_for(study) as client:
        client.freeze(study["manifest"])
    first = Custodian(study["plan_path"], study["digest"], study["state"])
    stale = Custodian(study["plan_path"], study["digest"], study["state"])
    candidate = study["manifest"]["candidates"][0]["candidate_digest"]
    first.query(candidate)
    with pytest.raises(Refused) as refused:
        stale.query(candidate)
    assert refused.value.code == "ALREADY_QUERIED"


def test_strict_claim_survives_abrupt_process_exit_before_ledger_append(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    code = (
        "import os,sys; from pathlib import Path; "
        "from neural_foundry.study.claims import StrictClaims,exclusive_state; "
        "state=Path(sys.argv[1]); "
        "lease=exclusive_state(state); lease.__enter__(); "
        "StrictClaims(state).begin('query:candidate', {'candidate':'fixed'}); os._exit(23)"
    )
    result = subprocess.run([sys.executable, "-c", code, str(state)], check=False)
    assert result.returncode == 23
    with exclusive_state(state), pytest.raises(ClaimSpent):
        StrictClaims(state).begin("query:candidate", {"candidate": "changed"})


def test_exclusive_state_refuses_another_owner(tmp_path: Path) -> None:
    with exclusive_state(tmp_path), pytest.raises(StateBusy), exclusive_state(tmp_path):
        pytest.fail("a second operation must not acquire the state")


def test_aggregate_schema_rejects_plugin_content_and_false_passes() -> None:
    result = {"training_rows": 1, "sealed_rows": 1,
              "counts": {"exact_input_matches": 1, "distinct_shared_3grams": 0,
                         "distinct_shared_4grams": 0, "distinct_shared_5grams": 0},
              "thresholds": {"exact_input_matches": 0, "distinct_shared_5grams": 0},
              "checks": {"exact_input_matches": False, "distinct_shared_5grams": True}, "passed": False}
    validate_aggregate(result)
    with pytest.raises(ValueError):
        validate_aggregate({**result, "prompt": "must never leave custody"})
    with pytest.raises(ValueError):
        validate_aggregate({**result, "passed": True})


def test_required_account_refuses_mutable_development_code_before_draw(tmp_path: Path) -> None:
    study = make_study(tmp_path)
    result = subprocess.run([
        sys.executable, "-m", "neural_foundry.study.custodian",
        "--plan", str(study["plan_path"]), "--plan-digest", study["digest"],
        "--state-dir", str(study["state"]),
        "--required-uid", str(getattr(os, "getuid", lambda: 1)()),
    ], capture_output=True, text=True, check=False)
    assert result.returncode == 2
    response = json.loads(result.stdout)
    assert response["code"] in {"ACCOUNT_BOUNDARY_FAILED", "DEPLOYMENT_NOT_PROTECTED"}
    assert not (study["state"] / "sealed").exists()


def test_stage1_refuses_real_evaluator_before_any_sealed_draw(tmp_path: Path) -> None:
    study = make_study(tmp_path, plan=make_plan(
        evaluator="pythia_mlp_override", checkpoint_dir="must-not-be-opened",
        checkpoint_digest="sha256:" + "0" * 64))
    result = subprocess.run([
        sys.executable, "-m", "neural_foundry.study.custodian",
        "--plan", str(study["plan_path"]), "--plan-digest", study["digest"],
        "--state-dir", str(study["state"]), "--software-fixtures-only",
    ], capture_output=True, text=True, check=False)
    assert result.returncode == 2
    assert json.loads(result.stdout)["code"] == "FIXTURE_ONLY"
    assert not (study["state"] / "sealed").exists()


def test_simultaneous_processes_have_only_one_strict_start(tmp_path: Path) -> None:
    code = (
        "import sys; from pathlib import Path; "
        "from neural_foundry.study.claims import StrictClaims,exclusive_state,ClaimSpent,StateBusy\n"
        "sys.stdin.readline()\n"
        "try:\n"
        " with exclusive_state(Path(sys.argv[1])):\n"
        "  StrictClaims(Path(sys.argv[1])).begin('one', {'request':'fixed'})\n"
        "  print('STARTED')\n"
        "except (ClaimSpent,StateBusy): print('REFUSED')\n"
    )
    children = [subprocess.Popen([sys.executable, "-c", code, str(tmp_path)],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) for _ in range(2)]
    for child in children:
        assert child.stdin is not None
        child.stdin.write("go\n")
        child.stdin.close()
    outputs = []
    for child in children:
        assert child.stdout is not None
        outputs.append(child.stdout.read().strip())
        assert child.wait(timeout=30) == 0
    assert sorted(outputs) == ["REFUSED", "STARTED"]
