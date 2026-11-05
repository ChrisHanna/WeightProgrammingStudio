"""Actual qualified Docker probes and native approval, not mock scorers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from neural_foundry import NativeError, content_digest, gyms

IMAGE = "sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c"
GOOD = """def evaluate(x, seed): return x*2
def oracle(x): return x*2
def score(x, out, reference): return float(out==reference)
"""


def fixture(
    tmp_path: Path, code: str = GOOD, episode: bool = False
) -> tuple[Any, dict[str, Any], Path, Path, dict[str, Any]]:
    source = tmp_path / "source"
    source.mkdir()
    (source / "gym.py").write_text(code)
    (source / "LICENSE").write_text("Fixture use only")
    (source / "lock.json").write_text(
        json.dumps({"schema_version": 1, "image": IMAGE, "packages": {}})
    )
    (source / "data.json").write_text('[{"task":"a","group":"g0"},{"task":"b","group":"g1"}]')

    def ep(symbol: str) -> dict[str, str]:
        return {"file": "gym.py", "symbol": symbol}

    integer = {"type": "integer", "min": 0, "max": 1000}
    interface: dict[str, Any] = {
        "kind": "single_shot",
        "evaluate": ep("evaluate"),
        "score": ep("score"),
        "oracle": ep("oracle"),
        "input": integer,
        "output": integer,
    }
    if episode:
        interface = {
            "kind": "episode",
            "factory": ep("Environment"),
            "score": ep("score"),
            "oracle": ep("oracle"),
            "reset": integer,
            "observation": integer,
            "action": integer,
            "result": integer,
            "max_steps": 3,
            "max_episode_ms": 2000,
        }
    draft: dict[str, Any] = {
        "schema_version": 1,
        "name": "probe",
        "version": "1",
        "usage_rights": "fixture",
        "confidentiality": "confidential",
        "assets": {
            name: {"kind": kind, "expected": None}
            for name, kind in {
                "gym.py": "code",
                "LICENSE": "license",
                "lock.json": "dependency_lock",
                "data.json": "data",
            }.items()
        },
        "dependency_lock": "lock.json",
        "licenses": ["LICENSE"],
        "interface": interface,
        "dataset": {
            "kind": "frozen",
            "files": ["data.json"],
            "task_key": "task",
            "group_key": "group",
        },
    }
    store = tmp_path / "store"
    snapshot = gyms.import_local(source, draft, store)
    plan: dict[str, Any] = {
        "schema_version": 1,
        "snapshot": snapshot.snapshot_id,
        "truth_provenance": "independent controller fixture: explicit double(2)=4,double(7)=14",
        "cases": [
            {
                "id": "two",
                "input": 2,
                "expected": 4,
                "wrong": [3, 5],
                "actions": [1, 1] if episode else [],
                "termination": "terminated" if episode else None,
            },
            {
                "id": "seven",
                "input": 7,
                "expected": 14,
                "wrong": [13, 15],
                "actions": [1, 1] if episode else [],
                "termination": "terminated" if episode else None,
            },
        ],
        "seeds": [11, 22],
        "scoring": {"minimum": 0.0, "maximum": 1.0, "good_min": 1.0, "bad_max": 0.0},
        "limits": {"wall_ms": 2000, "cpu_ms": 2000, "output_bytes": 32768},
    }
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir(mode=0o700)
    return snapshot, plan, sandbox, source, draft


def run(snapshot: Any, plan: dict[str, Any], sandbox: Path) -> tuple[Any, dict[str, Any]]:
    handle = gyms.validate(snapshot, plan, sandbox, enabled=True)
    report: dict[str, Any] = json.loads(handle.report_bytes())
    assert content_digest(handle.report_bytes()) == handle.receipt_id
    assert report["scope"] == "finite_controller_supplied_fixture_probes_not_scientific_acceptance"
    assert report["brokered_model_calls"] == 0
    return handle, report


def test_gym_validation_default_off_no_runtime_or_import(tmp_path: Path) -> None:
    marker = tmp_path / "outside"
    snapshot, plan, _, _, _ = fixture(tmp_path, f"open({str(marker)!r},'w').write('BAD')\n" + GOOD)
    missing = tmp_path / "no-runtime-created"
    with pytest.raises(NativeError, match="UNTRUSTED_GYM_EXECUTION_DISABLED"):
        gyms.validate(snapshot, plan, missing)
    assert not marker.exists() and not missing.exists()


def test_gym_validation_actual_pass_receipt_and_explicit_version_approval(tmp_path: Path) -> None:
    snapshot, plan, sandbox, _, _ = fixture(tmp_path)
    handle, report = run(snapshot, plan, sandbox)
    assert report["status"] == "passed", report
    assert all(report["checks"].values())
    assert report["cpu_ms"] > 0 and report["memory_peak_bytes"] > 0
    assert report["observations"] and report["sandbox_receipt"]
    approved = gyms.approve(snapshot, handle, "owner:fixture-approved-v1")
    assert approved.snapshot_id == snapshot.snapshot_id
    record = json.loads(approved.record_bytes())
    assert record["validation"] == handle.receipt_id and record["plan"] == report["plan"]
    assert content_digest(approved.record_bytes()) == approved.receipt_id
    assert "contract_permissions_and_budgets_still_required" in record["scope"]
    with pytest.raises(TypeError):
        type(handle)()
    with pytest.raises(TypeError):
        type(approved)()
    immutable: Any = handle
    with pytest.raises(AttributeError):
        immutable.receipt_id = "forged"
    with pytest.raises(TypeError):
        gyms.approve(snapshot, {"status": "passed"}, "forged")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("code", "check"),
    [
        (GOOD.replace("float(out==reference)", "1.0"), "wrong_score"),
        (GOOD.replace("def oracle(x): return x*2", "def oracle(x): return 4"), "oracle_agreement"),
        (
            GOOD.replace("def evaluate(x, seed): return x*2", "def evaluate(x, seed): return 0"),
            "known_good",
        ),
        (
            GOOD.replace("float(out==reference)", "1.0 if out is None else float(out==reference)"),
            "invalid_score",
        ),
        (GOOD.replace("float(out==reference)", "{'status':'passed'}"), "good_score"),
    ],
)
def test_gym_validation_rejects_bad_scorers_oracle_outputs_and_self_attestation(
    tmp_path: Path, code: str, check: str
) -> None:
    snapshot, plan, sandbox, _, _ = fixture(tmp_path, code)
    handle, report = run(snapshot, plan, sandbox)
    assert report["status"] == "rejected", report
    assert report["checks"][check] is False
    with pytest.raises(NativeError):
        gyms.approve(snapshot, handle, "cannot-approve")


def test_gym_validation_cannot_approve_missing_or_corrupt_native_receipt(tmp_path: Path) -> None:
    snapshot, plan, sandbox, _, _ = fixture(tmp_path)
    handle, report = run(snapshot, plan, sandbox)
    assert report["status"] == "passed", report
    path = tmp_path / "store" / "blobs" / handle.receipt_id.split(":")[1]
    path.unlink()
    path.write_bytes(b'{"status":"passed"}')
    with pytest.raises(NativeError):
        gyms.approve(snapshot, handle, "corrupt-record-no-approval")


def test_gym_validation_stale_receipt_cannot_approve_changed_snapshot(tmp_path: Path) -> None:
    snapshot, plan, sandbox, source, draft = fixture(tmp_path)
    handle, report = run(snapshot, plan, sandbox)
    assert report["status"] == "passed", report
    (source / "gym.py").write_text(GOOD + "\n# new immutable version\n")
    newer = gyms.import_local(source, draft, tmp_path / "store")
    assert newer.snapshot_id != snapshot.snapshot_id
    with pytest.raises(NativeError, match="stale"):
        gyms.approve(newer, handle, "old-receipt-no-authority")
    stale = dict(plan)
    stale["snapshot"] = newer.snapshot_id
    with pytest.raises(NativeError):
        gyms.validate(snapshot, stale, sandbox, enabled=True)


def test_gym_validation_stateful_seed_replay_and_resource_failure_do_not_approve(
    tmp_path: Path,
) -> None:
    code = (
        "counter=0\ndef evaluate(x,seed):\n global counter\n counter+=1\n return x*2+counter%2\n"
        + GOOD[GOOD.index("def oracle") :]
    )
    snapshot, plan, sandbox, _, _ = fixture(tmp_path, code)
    handle, report = run(snapshot, plan, sandbox)
    assert report["status"] == "rejected", report
    assert report["checks"]["seed_replay"] is False
    with pytest.raises(NativeError):
        gyms.approve(snapshot, handle, "reject-stateful")
    other = tmp_path / "resource"
    other.mkdir()
    source = "def evaluate(x,seed):\n while True: pass\n" + GOOD[GOOD.index("def oracle") :]
    snapshot, plan, sandbox, _, _ = fixture(other, source)
    plan["limits"] = {"wall_ms": 300, "cpu_ms": 200, "output_bytes": 4096}
    handle, report = run(snapshot, plan, sandbox)
    assert report["status"] == "inconclusive", report
    assert report["observations"] is None
    with pytest.raises(NativeError):
        gyms.approve(snapshot, handle, "reject-resource")


EPISODE = """class Environment:
 def reset(self,inputs,seed): self.x=inputs;self.n=0;return seed%2
 def step(self,action):
  self.n+=1
  return {'observation':self.n,'reward':1.0,'terminated':self.n>=2,'truncated':False}
 def result(self): return self.x*2
 def close(self): pass
def oracle(x): return x*2
def score(x,out,reference): return float(out==reference)
"""


@pytest.mark.parametrize("termination", ["terminated", "truncated"])
def test_gym_validation_actual_episode_reset_step_seed_termination_and_close(
    tmp_path: Path, termination: str
) -> None:
    code = EPISODE
    if termination == "truncated":
        code = code.replace(
            "'terminated':self.n>=2,'truncated':False", "'terminated':False,'truncated':self.n>=2"
        )
    snapshot, plan, sandbox, _, _ = fixture(tmp_path, code, episode=True)
    for case in plan["cases"]:
        case["termination"] = termination
    handle, report = run(snapshot, plan, sandbox)
    assert report["status"] == "passed", report
    assert report["checks"]["episode_lifecycle"] is True
    assert gyms.approve(snapshot, handle, "owner:episode-v1").snapshot_id == snapshot.snapshot_id


@pytest.mark.parametrize(
    "code",
    [
        EPISODE.replace("self.n>=2", "False"),
        EPISODE.replace(
            "def close(self): pass", "def close(self): raise RuntimeError('cleanup failed')"
        ),
    ],
)
def test_gym_validation_episode_missing_termination_or_cleanup_rejected(
    tmp_path: Path, code: str
) -> None:
    snapshot, plan, sandbox, _, _ = fixture(tmp_path, code, episode=True)
    handle, report = run(snapshot, plan, sandbox)
    assert report["status"] == "rejected", report
    assert report["checks"]["episode_lifecycle"] is False
    with pytest.raises(NativeError):
        gyms.approve(snapshot, handle, "bad-episode")


def test_gym_validation_plan_cannot_weaken_frozen_schema_or_episode_limits(tmp_path: Path) -> None:
    snapshot, plan, sandbox, _, _ = fixture(tmp_path, EPISODE, episode=True)
    plan["limits"]["wall_ms"] = 3000
    with pytest.raises(NativeError, match="wall limit"):
        gyms.validate(snapshot, plan, sandbox, enabled=True)
    plan["limits"]["wall_ms"] = 2000
    plan["cases"][0]["input"] = None
    with pytest.raises(NativeError):
        gyms.validate(snapshot, plan, sandbox, enabled=True)


def test_gym_validation_import_side_effects_only_inside_sandbox(tmp_path: Path) -> None:
    marker = tmp_path / "coordinator-must-not-write"
    prefix = f"from pathlib import Path\nassert not Path({str(marker)!r}).exists()\ntry: Path({str(marker)!r}).write_text('BAD')\nexcept OSError: pass\n"
    snapshot, plan, sandbox, _, _ = fixture(tmp_path, prefix + GOOD)
    handle, report = run(snapshot, plan, sandbox)
    assert report["status"] == "passed", report
    assert not marker.exists()
    assert gyms.approve(snapshot, handle, "scoped-validation").snapshot_id == snapshot.snapshot_id
