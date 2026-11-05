"""T042 adversarial boundary and truthful end-to-end fixtures; real OCI only."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, cast

import pytest
from neural_foundry import NativeError, _native, content_digest, gyms
from neural_foundry.gym_scientist import recover_reference_audits, run_reference_audit
from test_gym_reference_episode import EPISODE, setup
from test_gym_validation import GOOD, fixture

ROLES = ["valid", "invalid", "constant", "metadata_only", "weak", "strong"]
COVERAGE = {"required": ["easy", "hard"], "case_groups": {"two": "easy", "seven": "hard"}}
# The reviewer's exact import-time batch-forgery attack. No entrypoints exist.
FORGED = '''import json
import os
request = json.load(open("/job/input.json"))
ok = lambda value: {"ok": True, "value": value}
samples = []
for index, cell in enumerate(request["cells"]):
    x = cell["input"]
    if request["mode"] == "episode":
        actions = cell["actions"]
        trace = {
            "reset": ok(x),
            "steps": [ok({"observation": x + 10, "reward": 1.0,
                          "terminated": j == 1, "truncated": False})
                      for j, _ in enumerate(actions)],
            "evaluation": ok(x * 2) if len(actions) == 2 else {"ok": False},
            "close": ok(None),
        }
        samples.append({"index": index, "episode": trace})
    else:
        samples.append({"index": index, "oracle": ok(cell["expected"]),
                        "score": ok(float(cell["output"] == cell["expected"])),
                        "invalid_score": ok(0.0), "good_score": ok(1.0),
                        "wrong_scores": [ok(0.0) for _ in cell["wrong"]]})
os.write(1, json.dumps({"schema_version": 1, "samples": samples}).encode())
os._exit(0)
'''


def controls(tmp_path: Path, code: str, episode: bool) -> tuple[Any, ...]:
    if episode:
        gym, plan, sandbox, _, policies, policy_root, declaration = setup(tmp_path, code)
        expression = "x if x < 10 else x-10"
    else:
        gym, plan, sandbox, source, declaration = fixture(tmp_path, code)
        plan["limits"].update(cpu_ms=1000, wall_ms=1000)
        policy_root = tmp_path / "policy"
        policy_root.mkdir()
        for name in ("LICENSE", "lock.json", "data.json"):
            (policy_root / name).write_bytes((source / name).read_bytes())
        policies = []
        expression = "x*2"
    plan["limits"]["cpu_ms"] = 1000
    for role in ROLES[len(policies):]:
        declaration = copy.deepcopy(declaration)
        if episode:
            declaration["interface"] = {
                "kind": "single_shot", "input": {"type": "integer", "min": 0, "max": 1000},
                "output": {"type": "integer", "min": 0, "max": 1000},
                **{name: {"file": "gym.py", "symbol": name} for name in ("evaluate", "score", "oracle")}}
        declaration["name"] = role
        body = expression if role in ("valid", "strong") else "0"
        if role == "invalid":
            body = "1"
        if role == "metadata_only":
            body = "seed"
        (policy_root / "gym.py").write_text(f"def evaluate(x,seed): return {body}\n")
        policies.append(gyms.import_local(policy_root, declaration, tmp_path / "store"))
    return gym, plan, sandbox, policies


def execute(gym: Any, plan: dict[str, Any], sandbox: Path, policies: list[Any],
            roles: list[str] = ROLES, coverage: dict[str, Any] | None = COVERAGE,
            cancellation: Any = None) -> dict[str, Any]:
    rounds = 6 if plan["cases"][0]["actions"] else 2
    count = len(policies) * rounds
    handle = run_reference_audit(
        gym, json.dumps(plan), policies, roles, "development", json.dumps({
            "max_executions": count, "cpu_ms": count * plan["limits"]["cpu_ms"],
            # Absolute allowance includes Docker control/staging/cleanup, not
            # only guest watchdogs. Individual hard job limits stay unchanged.
            "wall_ms": count * (plan["limits"]["wall_ms"] + 1500) + 5000,
            "storage_bytes": (count + 1) * 20 * 1024 * 1024}), str(sandbox), enabled=True,
        coverage=json.dumps(coverage) if coverage is not None else None,
        cancellation=cancellation)
    assert content_digest(handle.report_bytes()) == handle.receipt_id
    return cast(dict[str, Any], json.loads(handle.report_bytes()))


@pytest.mark.parametrize("allowance", [None, 0, 1, 2**64 - 1])
def test_t042_storage_rejects_before_qualification_or_publication(
    tmp_path: Path, allowance: int | None,
) -> None:
    gym, plan, _, policies = controls(tmp_path, GOOD, False)
    sandbox = tmp_path / "must-not-be-created"
    budget = {"max_executions": 12, "cpu_ms": 12000, "wall_ms": 12000}
    if allowance is not None:
        budget["storage_bytes"] = allowance
    store = tmp_path / "store"
    before = {p.relative_to(store): p.read_bytes() for p in store.rglob("*") if p.is_file()}
    # Direct binding call: Python validation is not the security boundary.
    with pytest.raises(NativeError, match="storage|budget"):
        _native.run_gym_reference_audit(
            gym, json.dumps(plan), policies, ROLES, "development", json.dumps(budget),
            str(sandbox), True, json.dumps(COVERAGE), None)
    assert not sandbox.exists()
    assert {p.relative_to(store): p.read_bytes() for p in store.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("kind", ["single_shot", "terminated", "truncated"])
def test_t042_full_truthful_native_audit(kind: str, tmp_path: Path) -> None:
    code = GOOD if kind == "single_shot" else EPISODE
    if kind == "truncated":
        code = code.replace("terminated=self.n==2,truncated=False", "terminated=False,truncated=self.n==2")
    gym, plan, sandbox, policies = controls(tmp_path, code, kind != "single_shot")
    if kind == "truncated":
        for case in plan["cases"]:
            case["termination"] = kind
    before = gym.manifest_bytes()
    report = execute(gym, plan, sandbox, policies)
    assert report["status"] == "passed", report
    assert report["audit_complete"] and not report["missing_prerequisites"]
    assert len(report["policies"]) == 6 and report["coverage"] == COVERAGE
    assert report["policies"][0]["correct"] == report["policies"][5]["correct"] == 8
    assert report["policies"][4]["correct"] == 0
    assert gym.manifest_bytes() == before and not report["native_approval"]
    assert 0 < report["reserved_asset_bytes"] < report["storage"]["charged_bytes"]
    storage = report["storage"]
    assert storage["jobs_claimed"] == report["executions"]
    assert storage["charged_bytes"] <= report["budget"]["storage_bytes"]
    assert storage["sandbox_ceiling_bytes"] == report["budget"]["max_executions"] * 16 * 1024 * 1024
    assert storage["measured_study_storage_bytes"] is None
    assert storage["observed_workspace_peak_bytes"] > 0
    assert storage["observed_journal_retained_bytes"] > 0
    assert not storage["capacity_guaranteed"] and not storage["durable_family_accounting"]
    assert report["controller_wall_ms"] >= report["wall_ms"]
    assert 0 < report["memory_peak_bytes"] <= 134217728
    assert "finite_probes_do_not_prove" in report["observation_boundary"]


@pytest.mark.parametrize("episode", [False, True])
def test_t042_direct_reviewer_forgery_cannot_complete(episode: bool, tmp_path: Path) -> None:
    # Diagnostic is emitted immediately before the original attack's write,
    # from inside the isolated callback (not from the controller or supervisor).
    marker = b"T042_FORGERY_CALLBACK_WRITE_REACHED"
    code = FORGED.replace("os.write(1,", f"os.write(2, {marker!r})\nos.write(1,", 1)
    gym, plan, sandbox, policies = controls(tmp_path, code, episode)
    report = execute(gym, plan, sandbox, policies[:1], ["valid"])
    assert report["status"] in ("rejected", "inconclusive"), report
    assert report["audit_complete"] is False and not report["policies"]
    assert report["executions"] == (1 if episode else 2)
    receipts = [json.loads((tmp_path / "store" / "blobs" / digest[7:]).read_bytes())
                for digest in report["sandbox_receipts"]]
    failed = receipts[-1]
    assert failed["cleanup_confirmed"] and failed["status"] == "failed", failed
    raw = failed["report"]
    stderr = bytes.fromhex(raw["stderr_hex"])
    assert marker in stderr and b"callback reply missing or exceeded" in stderr, stderr
    assert raw["stdout_hex"] == "" and raw["exit_code"] != 0
    assert raw["wall_ms"] < plan["limits"]["wall_ms"]
    assert "bounded_execution_incomplete" in report["findings"]
    assert recover_reference_audits(str(sandbox), enabled=True) == 0


@pytest.mark.parametrize("forgery", [
    # Even knowing the current dispatch challenge cannot predict the NEXT one.
    ("import inspect,json,os\n"
    "command=inspect.currentframe().f_back.f_back.f_back\n"
    "while command and 'command' not in command.f_locals: command=command.f_back\n"
    "nonce=command.f_locals['command']['challenge']\n"
    "os.write(1,(json.dumps({'challenge':nonce,'result':{'ok':True,'value':None}})+'\\n').encode())\n"
    "os._exit(0)\n"),
    "import os\nos._exit(0)\n",
])
def test_t042_import_protocol_spoof_and_clean_exit_fail(forgery: str, tmp_path: Path) -> None:
    gym, plan, sandbox, policies = controls(tmp_path, forgery, True)
    report = execute(gym, plan, sandbox, policies[:1], ["valid"])
    assert report["status"] == "inconclusive", report
    assert not report["observations"] and not report["policies"]


def test_t042_supervisor_descriptors_are_not_callback_accessible(tmp_path: Path) -> None:
    prelude = '''import os
for target in [f"/proc/{os.getppid()}/fd/1", f"/proc/{os.getppid()}/mem"]:
 try:
  fd=os.open(target,os.O_WRONLY)
 except PermissionError:
  pass
 else:
  os.close(fd)
  raise AssertionError("supervisor accessible")
'''
    gym, plan, sandbox, policies = controls(tmp_path, prelude + GOOD, False)
    report = execute(gym, plan, sandbox, policies[:1], ["valid"], coverage=None)
    assert report["status"] == "probe_passed", report
    assert "missing_required_coverage" in report["missing_prerequisites"]


def test_t042_full_controls_without_coverage_remain_probe(tmp_path: Path) -> None:
    gym, plan, sandbox, policies = controls(tmp_path, GOOD, False)
    report = execute(gym, plan, sandbox, policies, coverage=None)
    assert report["status"] == "probe_passed" and not report["audit_complete"]
    assert report["missing_prerequisites"] == ["missing_required_coverage"]
    bad = {"required": ["easy", "hard", "missing"], "case_groups": COVERAGE["case_groups"]}
    report = execute(gym, plan, sandbox, policies, coverage=bad)
    assert report["status"] == "probe_passed" and not report["audit_complete"]


def test_t042_coverage_cannot_hide_missing_saturation_controls(tmp_path: Path) -> None:
    gym, plan, sandbox, policies = controls(tmp_path, GOOD, False)
    report = execute(gym, plan, sandbox, policies[:4], ROLES[:4])
    assert report["status"] == "probe_passed" and not report["audit_complete"]
    assert report["missing_prerequisites"] == ["missing_strong_control", "missing_weak_control"]
    with pytest.raises(NativeError):
        execute(gym, plan, sandbox, policies, coverage={"required": ["g"], "case_groups": {"fake": "g"}})


def test_t042_sticky_cancellation_active_job_cleanup_and_recovery(tmp_path: Path) -> None:
    gym, plan, sandbox, policies = controls(tmp_path, "while True: pass\n", False)
    plan["limits"].update(cpu_ms=5000, wall_ms=5000)
    cancellation = _native.GymAuditCancellationHandle()
    reports: list[dict[str, Any]] = []
    errors: list[BaseException] = []
    def run() -> None:
        try:
            reports.append(execute(gym, plan, sandbox, policies[:1], ["valid"], cancellation=cancellation))
        except BaseException as error:  # noqa: BLE001 -- preserve thread failures
            errors.append(error)
    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        active = False
        for path in sandbox.glob("*/intent.json"):
            try:
                active |= bool(json.loads(path.read_bytes()).get("asset_identity")) and not (
                    path.parent / "completion.json").exists()
            except (OSError, ValueError):
                pass
        if active:
            time.sleep(0.25)
            break
        if not thread.is_alive():
            break
        time.sleep(0.01)
    cancellation.cancel()
    thread.join(timeout=15)
    assert not thread.is_alive() and not errors, errors
    assert cancellation.cancelled and reports
    report = reports[0]
    assert report["status"] == "inconclusive" and "audit_cancelled" in report["findings"]
    assert report["executions"] <= 2
    assert recover_reference_audits(str(sandbox), enabled=True) == 0
    with pytest.raises(NativeError, match="cancelled"):
        execute(gym, plan, sandbox, policies[:1], ["valid"], cancellation=cancellation)


def test_t042_crashed_controller_recovery_never_promotes(tmp_path: Path) -> None:
    gym, plan, sandbox, policies = controls(tmp_path, "while True: pass\n", False)
    plan["limits"].update(cpu_ms=5000, wall_ms=5000)
    # The first asset job must stay running long enough for an actual crash;
    # otherwise a fast correct policy could complete between inspect and kill.
    (tmp_path / "policy" / "gym.py").write_text("while True: pass\n")
    controller = """import json,sys
from neural_foundry import gyms
from neural_foundry.gym_scientist import run_reference_audit
store,gymdir,gymdraft,policydir,policydraft,plan,root=sys.argv[1:]
gym=gyms.import_local(gymdir,json.loads(gymdraft),store)
policy=gyms.import_local(policydir,json.loads(policydraft),store)
run_reference_audit(gym,plan,[policy],
 ['valid'],'development',json.dumps({'max_executions':2,'cpu_ms':10000,'wall_ms':10000,'storage_bytes':67108864}),
 root,enabled=True)
"""
    child = subprocess.Popen([
        sys.executable, "-c", controller, str(tmp_path / "store"), str(tmp_path / "source"),
        json.dumps(json.loads(gym.manifest_bytes())["draft"]), str(tmp_path / "policy"),
        json.dumps(json.loads(policies[0].manifest_bytes())["draft"]), json.dumps(plan), str(sandbox),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    pending = None
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and child.poll() is None:
            for path in sandbox.glob("*/intent.json"):
                if not json.loads(path.read_bytes()).get("asset_identity"):
                    continue
                check = subprocess.run([
                    "/usr/bin/docker", "--host", "unix:///var/run/docker.sock", "inspect",
                    "--format", "{{.State.Running}}", path.parent.name,
                ], capture_output=True, timeout=2, check=False)
                if check.returncode == 0 and check.stdout.strip() == b"true":
                    pending = path.parent
                    break
            if pending is not None:
                break
            time.sleep(0.01)
        assert pending is not None, (
            "no real running audit job before controller crash", child.communicate(timeout=5))
        child.kill()
        child.wait(timeout=5)
        assert not (pending / "recovery.json").exists()
        # Current qualification reconciles pending jobs before its proof probe;
        # the explicit follow-up sweep can therefore report zero additional jobs.
        recover_reference_audits(str(sandbox), enabled=True)
        recovery = json.loads((pending / "recovery.json").read_bytes())
        assert recovery["cleanup_confirmed"] and recovery["outputs_eligible"] is False
        assert recover_reference_audits(str(sandbox), enabled=True) == 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
            recover_reference_audits(str(sandbox), enabled=True)


def test_t042_real_prefix_drift_still_rejected(tmp_path: Path) -> None:
    code = "import json\n" + EPISODE.replace("reward=1.0", "reward=float(len(json.load(open('/job/input.json'))['cells'][0]['actions']))")
    gym, plan, sandbox, policies = controls(tmp_path, code, True)
    report = execute(gym, plan, sandbox, policies[:1], ["valid"])
    assert report["status"] == "rejected"
    assert "episode_lifecycle_or_prefix_drift" in report["findings"]
