"""Real native/Python episode audits; no host callback or sandbox substitutes."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, cast

import pytest
from neural_foundry import NativeError, _native, content_digest, gyms
from neural_foundry.gym_scientist import run_reference_audit
from test_gym_validation import GOOD, fixture

EPISODE = """class Environment:
 def reset(self,inputs,seed):
  self.x=inputs; self.n=0; self.total=0; return inputs
 def step(self,action):
  self.n+=1; self.total+=action
  return dict(observation=self.x+10,reward=1.0,terminated=self.n==2,truncated=False)
 def result(self): return self.total
 def close(self): pass
def oracle(x): return x*2
def score(x,out,reference): return float(out==reference)
"""


def setup(tmp_path: Path, code: str = EPISODE) -> tuple[Any, ...]:
    gym, plan, sandbox, source, draft = fixture(tmp_path, code, episode=True)
    # Include the trusted supervisor and separate callback interpreter in CPU
    # reservation; the frozen per-policy episode wall reservation stays 1800 ms.
    plan["limits"].update(cpu_ms=1000, wall_ms=300)
    policy_root = tmp_path / "policy"
    policy_root.mkdir()
    for name in ("LICENSE", "lock.json", "data.json"):
        (policy_root / name).write_bytes((source / name).read_bytes())
    declaration = copy.deepcopy(draft)
    integer = {"type": "integer", "min": 0, "max": 1000}
    def ep(symbol: str) -> dict[str, str]:
        return {"file": "gym.py", "symbol": symbol}
    declaration["interface"] = {"kind": "single_shot", "evaluate": ep("evaluate"),
                                "score": ep("score"), "oracle": ep("oracle"),
                                "input": integer, "output": integer}
    policies = []
    for role, expression in (("valid", "x if x < 10 else x-10"), ("invalid", "1"),
                             ("constant", "0"), ("metadata_only", "seed")):
        declaration["name"] = role
        (policy_root / "gym.py").write_text(
            f"def evaluate(x,seed): return {expression}\n"
            "def oracle(x): raise AssertionError('policy oracle used')\n"
            "def score(x,out,reference): raise AssertionError('policy scorer used')\n")
        policies.append(gyms.import_local(policy_root, declaration, tmp_path / "store"))
    return gym, plan, sandbox, source, policies, policy_root, declaration


def audit(gym: Any, plan: dict[str, Any], sandbox: Path, policies: list[Any],
          roles: list[str] | None = None, budget: dict[str, int] | None = None) -> dict[str, Any]:
    handle = run_reference_audit(gym, json.dumps(plan), policies,
                                 roles or ["valid", "invalid", "constant", "metadata_only"],
                                 "development", json.dumps(budget or {
                                     "max_executions": 24, "cpu_ms": 24000, "wall_ms": 7200,
                                     "deadline_policy": 2,
                                     "storage_bytes": 512 * 1024 * 1024}),
                                 str(sandbox), enabled=True)
    assert isinstance(handle, _native.GymReferenceAuditHandle)
    assert content_digest(handle.report_bytes()) == handle.receipt_id
    return cast(dict[str, Any], json.loads(handle.report_bytes()))


@pytest.mark.parametrize("termination", ["terminated", "truncated"])
def test_episode_native_binding_same_gym_all_controls(tmp_path: Path, termination: str) -> None:
    code = EPISODE if termination == "terminated" else EPISODE.replace(
        "terminated=self.n==2,truncated=False", "terminated=False,truncated=self.n==2")
    gym, plan, sandbox, source, policies, policy_root, _ = setup(tmp_path, code)
    for case in plan["cases"]:
        case["termination"] = termination
    identity, manifest = gym.snapshot_id, gym.manifest_bytes()
    (source / "gym.py").write_text("raise AssertionError('live gym used')")
    (policy_root / "gym.py").write_text("raise AssertionError('live policy used')")
    report = audit(gym, plan, sandbox, policies)
    assert report["status"] == "probe_passed", report
    assert report["audit_complete"] is False
    assert "missing_required_coverage" in report["missing_prerequisites"]
    assert report["snapshot"] == identity == gym.snapshot_id
    assert gym.manifest_bytes() == manifest
    assert report["executions"] == 24 and len(report["sandbox_receipts"]) == 24
    assert report["policies"][0]["correct"] == report["policies"][0]["passing"] == 8
    assert all(row["passing"] == 0 for row in report["policies"][1:])
    assert report["requested_policies"] == [[p.snapshot_id, r] for p, r in zip(
        policies, ("valid", "invalid", "constant", "metadata_only"))]
    assert 0 < report["cpu_ms"] <= 24000 and 0 < report["wall_ms"] <= 7200
    assert report["budget"]["deadline_policy"] == 2
    assert 0 < report["lifecycle_wall_ms"] <= 26768
    assert report["native_approval"] is False and report["brokered_model_calls"] == 0
    for digest in report["sandbox_receipts"]:
        raw = (tmp_path / "store" / "blobs" / digest[7:]).read_bytes()
        receipt = json.loads(raw)
        assert content_digest(raw) == digest
        assert receipt["cleanup_confirmed"] and receipt["status"] == "completed"
    assert not list(sandbox.rglob("process.json"))


def test_native_binding_reports_preflight_timing(tmp_path: Path) -> None:
    gym, plan, sandbox, _, _ = fixture(tmp_path)
    report = audit(gym, plan, sandbox, [gym], ["valid"], {
        "max_executions": 2,
        "cpu_ms": 4000,
        "wall_ms": 7200,
        "storage_bytes": 512 * 1024 * 1024,
    })

    timings = report["preflight_timings"]
    phases = (
        "binding_decode_ns",
        "request_validation_ns",
        "snapshot_restore_ns",
        "coverage_parse_ns",
        "docker_qualification_ns",
        "before_audit_ns",
        "audit_preflight_ns",
    )
    assert set(timings) == set(phases)
    assert all(type(timings[phase]) is int and timings[phase] >= 0 for phase in phases)
    assert timings["docker_qualification_ns"] > 0
    assert timings["before_audit_ns"] >= sum(timings[phase] for phase in phases[:5])
    assert timings["audit_preflight_ns"] > 0


@pytest.mark.parametrize(("old", "new"), [
    ("return inputs", "return 'bad reset'"),
    ("return inputs", "raise RuntimeError('reset')"),
    ("self.n+=1; self.total+=action", "raise RuntimeError('step')"),
    ("reward=1.0", "reward=True"),
    ("reward=1.0", "reward=1e13"),
    ("observation=self.x+10", "observation='bad step'"),
    ("terminated=self.n==2", "terminated=False"),
    ("truncated=False", "truncated=self.n==2"),
    ("terminated=self.n==2,truncated=False", "terminated=False,truncated=self.n==2"),
    ("def result(self): return self.total", "def result(self): raise RuntimeError('result')"),
    ("def close(self): pass", "def close(self): return 1"),
    ("def close(self): pass", "def close(self): raise RuntimeError('close')"),
])
def test_episode_malformed_or_error_lifecycle_denied(tmp_path: Path, old: str, new: str) -> None:
    gym, plan, sandbox, _, policies, _, _ = setup(tmp_path, EPISODE.replace(old, new))
    report = audit(gym, plan, sandbox, policies[:1], ["valid"])
    assert report["status"] == "rejected", report
    assert report["snapshot"] == gym.snapshot_id
    assert "episode_lifecycle_or_prefix_drift" in report["findings"]
    assert not list(sandbox.rglob("process.json"))


@pytest.mark.parametrize("stage", ["policy", "reset", "step", "result", "close"])
def test_episode_timeouts_same_gym_cleanup_and_host_isolation(tmp_path: Path, stage: str) -> None:
    marker = tmp_path / "host-marker"
    prelude = (f"from pathlib import Path\ntry: Path({str(marker)!r}).touch()\n"
               "except OSError: pass\n")
    code = EPISODE
    if stage != "policy":
        old, new = {
            "reset": ("return inputs", "exec('while True: pass')"),
            "step": ("self.n+=1; self.total+=action", "exec('while True: pass')"),
            "result": ("return self.total", "exec('while True: pass')"),
            "close": ("def close(self): pass", "def close(self): exec('while True: pass')"),
        }[stage]
        code = code.replace(old, new)
    gym, plan, sandbox, _, policies, policy_root, declaration = setup(tmp_path, prelude + code)
    if stage == "policy":
        (policy_root / "gym.py").write_text(prelude + GOOD.replace(
            "def evaluate(x, seed): return x*2", "def evaluate(x, seed):\n while True: pass"))
        policies = [gyms.import_local(policy_root, declaration, tmp_path / "store")]
    report = audit(gym, plan, sandbox, policies[:1], ["valid"])
    assert report["status"] == "inconclusive", report
    assert report["snapshot"] == gym.snapshot_id and not marker.exists()
    assert "bounded_execution_incomplete" in report["findings"]
    assert report["executions"] <= 6
    assert not list(sandbox.rglob("process.json"))


@pytest.mark.parametrize("body", ["return 'bad action'", "raise RuntimeError('policy')"])
def test_episode_policy_action_schema_and_error_denied(tmp_path: Path, body: str) -> None:
    gym, plan, sandbox, _, _, policy_root, declaration = setup(tmp_path)
    (policy_root / "gym.py").write_text(f"def evaluate(x,seed): {body}\n")
    policy = gyms.import_local(policy_root, declaration, tmp_path / "store")
    report = audit(gym, plan, sandbox, [policy], ["valid"])
    assert report["status"] == "rejected"
    assert ("policy_action_schema" if body.startswith("return") else "policy_execution_failed"
            ) in report["findings"]
    assert report["executions"] == 2


def test_episode_constant_role_means_actions_not_environment_results(tmp_path: Path) -> None:
    gym, plan, sandbox, _, policies, _, _ = setup(
        tmp_path, EPISODE.replace("return self.total", "return self.x*2-1"))
    report = audit(gym, plan, sandbox, policies[2:3], ["constant"])
    assert report["status"] == "probe_passed", report
    assert report["policies"][0]["correct"] == report["policies"][0]["passing"] == 0


def test_episode_policy_action_replay_drift_denied_before_step(tmp_path: Path) -> None:
    gym, plan, sandbox, _, _, policy_root, declaration = setup(tmp_path)
    (policy_root / "gym.py").write_text(
        "n=0\ndef evaluate(x,seed):\n global n\n n+=1\n return 1+n%2\n")
    policy = gyms.import_local(policy_root, declaration, tmp_path / "store")
    report = audit(gym, plan, sandbox, [policy], ["valid"])
    assert report["status"] == "rejected" and "policy_replay_drift" in report["findings"]
    assert report["executions"] == 2


def test_episode_callback_returns_are_immutable_wire_values(tmp_path: Path) -> None:
    code = EPISODE.replace("self.n+=1; self.total+=action",
        "self.n+=1; self.total+=action\n  "
        "if self.n==2: self.prior['reward']=2.0").replace(
        "return dict(observation=self.x+10,reward=1.0,terminated=self.n==2,truncated=False)",
        "step=dict(observation=self.x+10,reward=1.0,terminated=self.n==2,truncated=False)\n"
        "  if self.n==1: self.prior=step\n  return step")
    gym, plan, sandbox, _, policies, _, _ = setup(tmp_path, code)
    report = audit(gym, plan, sandbox, policies[:1], ["valid"])
    # A callback cannot retroactively mutate a value already observed by the
    # separate supervisor. Unlike the old shared-interpreter trace, each return
    # is serialized before the next dispatch.
    assert report["status"] == "probe_passed", report


def test_episode_reservations_authorization_and_unsupported_policy_denied(tmp_path: Path) -> None:
    gym, plan, sandbox, _, policies, _, _ = setup(tmp_path)
    for field, value in (("max_executions", 5), ("cpu_ms", 1199), ("wall_ms", 1799)):
        with pytest.raises(NativeError):
            audit(gym, plan, sandbox, policies[:1], ["valid"], {
                "max_executions": 6, "cpu_ms": 1200, "wall_ms": 1800,
                "storage_bytes": 128 * 1024 * 1024, field: value})
    weakened = copy.deepcopy(plan)
    weakened["limits"]["wall_ms"] = 400
    with pytest.raises(NativeError):
        audit(gym, weakened, sandbox, policies[:1], ["valid"])
    with pytest.raises(NativeError):
        audit(gym, plan, sandbox, [gym], ["valid"])
    with pytest.raises(ValueError, match="DISABLED"):
        run_reference_audit(gym, json.dumps(plan), policies[:1], ["valid"], "development",
                            "{}", str(sandbox))
    for exposure in ("holdout", "final", "unauthorised"):
        with pytest.raises(ValueError):
            run_reference_audit(gym, json.dumps(plan), policies[:1], ["valid"], exposure,
                                "{}", str(sandbox), enabled=True)
    assert not list(sandbox.rglob("process.json"))
