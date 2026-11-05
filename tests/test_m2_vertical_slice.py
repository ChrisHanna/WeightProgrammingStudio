"""T043: real M2 component chain and negative gates, NOT a joined public workflow.

Every custom-code path uses real qualified OCI. Public synthetic arithmetic audits
are never relabeled confidential holdout evidence. Native linear acceptance tests
are run separately and never used to authorize the five-head arithmetic candidate.
"""

from __future__ import annotations

import copy
import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import neural_foundry as nf
import pytest
from m2_fixture_support import (
    deadline,
    run,
    rust_case,
    sandbox,
    split_controller,
    split_request,
    validated_sum,
)
from neural_foundry import _native, arithmetic, controls, models, surgeon
from neural_foundry.reload_verification import Probe, ReloadVerificationError, verify_reload
from test_arithmetic_gyms import repair_plan
from test_paired_controls import protocol


@pytest.fixture(autouse=True)
def bounded_case() -> Iterator[None]:
    with deadline():
        yield


@pytest.mark.parametrize("case", [
    "m2_arithmetic_source_supplied_ir_export_resolves_repair_harm_and_noop",
    "m2_arithmetic_source_rejects_each_forged_axis_and_missing_payload",
    "m2_arithmetic_source_rejects_valid_cas_unregistered_graph_and_extra_assets",
])
def test_m2_vertical_slice_arithmetic_artifact_integration_only(case: str) -> None:
    # Deliberately not a substitute for joined coordinator/scorer acceptance.
    rust_case("m2_arithmetic_source", case)


@pytest.mark.parametrize("case", [
    "m2_arithmetic_joined_positive_five_gym",
    "m2_arithmetic_joined_harmful_retention",
    "m2_arithmetic_joined_noop_has_no_gain",
    "m2_arithmetic_joined_missing_gym",
    "m2_arithmetic_joined_cancel_after_actual_model",
    "m2_arithmetic_joined_crash_reopen_recovery",
    "m2_arithmetic_joined_missing_score_is_incomplete",
])
def test_m2_vertical_slice_native_joined_five_gym(case: str) -> None:
    # Prebuild offline with this same profile; compilation is not test evidence.
    rust_case("m2_arithmetic_joined", case, timeout=84, CARGO_PROFILE_TEST_OPT_LEVEL="1")


@pytest.mark.parametrize("harmful", [False, True], ids=["repair", "retention-damage"])
def test_m2_vertical_slice_real_mixed_five_gyms_and_required_retention(
    tmp_path: Path, harmful: bool,
) -> None:
    model = arithmetic.fixture()
    before = model.checkpoint_bytes()
    identity = model.source.parameter_digest
    snapshot, validation, approval = validated_sum(tmp_path, model)
    portfolio = arithmetic.mixed_portfolio(snapshot, approval)
    assert [entry["head"] for entry in portfolio] == list(arithmetic.HEADS)
    assert portfolio[0]["name"] == "user.arithmetic.sum"
    assert [entry["name"] for entry in portfolio[1:]] == [
        "arithmetic.difference.v1", "arithmetic.maximum.v1", "arithmetic.minimum.v1",
        "arithmetic.absolute_difference.v1",
    ]
    assert all(entry["required"] for entry in portfolio)
    assert all(entry["max_regression"] == 0 for entry in portfolio[1:])
    assert all(entry["status"] == "proposed_binding_not_study_dispatch" for entry in portfolio)
    assert json.loads(validation.report_bytes())["snapshot"] == snapshot.snapshot_id
    plan, smoke = repair_plan(model, harmful=harmful)
    candidate = surgeon.apply(plan, smoke)
    candidate_source = candidate.snapshot()
    # Separate actual Torch process, not merely a native in-process forward.
    (tmp_path / "trusted-model").mkdir(mode=0o700)
    runtime = models.qualify(tmp_path / "trusted-model", enabled=True)
    cases = [case for case in arithmetic.describe(model)["inventory"]["cases"]
             if case["split"] == "development"]
    inputs = [[float(case["x1"]), float(case["x2"])] for case in cases]
    assert len(inputs) > 64  # The worker admits at most 64 rows per actual process.
    with pytest.raises(nf.NativeError) as oversized:
        models.infer(runtime, candidate_source, inputs)
    assert oversized.value.code == "LIMIT_EXCEEDED"
    outputs = []
    for start in range(0, len(inputs), 64):
        result = models.infer(runtime, candidate_source, inputs[start:start + 64])
        evidence = json.loads(result.receipt_bytes())
        assert result.status == "verified_inference" and evidence["cleanup_confirmed"] is True
        assert evidence["child_report"]["before"] == evidence["child_report"]["after"]
        assert evidence["outputs_eligible"] is False
        outputs.extend(result.outputs)
    rows = [{"inputs": {"x1": case["x1"], "x2": case["x2"]}, "output": output[0]}
            for case, output in zip(cases, outputs, strict=True)]
    # Execute EXACT snapshotted user scorer bytes on real edited-model outputs.
    code = snapshot.asset_bytes("sum_gym.py").decode() + '''
import json,sys
rows=json.load(sys.stdin)
print(json.dumps([score(r['inputs'],r['output'],oracle(r['inputs'])) for r in rows]))
'''
    receipt = sandbox(tmp_path / "score-sandbox", code, input=json.dumps(rows))
    assert receipt["status"] == "completed" and receipt["cleanup_confirmed"] is True
    assert json.loads(bytes.fromhex(receipt["report"]["stdout_hex"])) == [1.0] * len(rows)
    assert receipt["outputs_eligible"] is False
    # These are PUBLIC synthetic audits, not broker-authorized heldout evaluation.
    for split in ("development", "selection", "holdout"):
        baseline = arithmetic.evaluate(model, split=split)
        changed = arithmetic.evaluate(model, candidate_source, split=split)
        assert changed["sum"]["accuracy"] == 1 > baseline["sum"]["accuracy"]
        if harmful:
            assert changed["difference"]["accuracy"] < baseline["difference"]["accuracy"]
        else:
            assert all(changed[head]["accuracy"] == baseline[head]["accuracy"]
                       for head in arithmetic.HEADS[1:])
        assert all(value["scope"] == "synthetic_software_fixture_not_scientific_acceptance"
                   for value in changed.values())
    assert json.loads(candidate.receipt_json())["promotion_status"] == "not_evaluated"
    assert model.checkpoint_bytes() == before and model.source.parameter_digest == identity
    assert not (tmp_path / "gym-store" / "active").exists()


def test_m2_vertical_slice_actual_source_noop_export_and_fresh_reload(tmp_path: Path) -> None:
    model = arithmetic.fixture()
    source_bytes = model.checkpoint_bytes()
    frozen_manifest = model.source.manifest_digest
    frozen_outputs = tuple(model.source.forward([1.0, 2.0]))
    (tmp_path / "runtime").mkdir(mode=0o700)
    runtime = models.qualify(tmp_path / "runtime", enabled=True)
    cache = controls.SourceCache()
    noop, smoke = repair_plan(model, factor=1.0)
    witness = controls.run(cache, runtime, noop, smoke, protocol(model), tmp_path / "store")
    report = json.loads(witness.report_bytes())
    assert report["exact_behavior"] and report["exact_parameters_buffers_aliases_assets"]
    assert report["fresh_process_no_op_reload"] and report["no_op_accuracy_gain"] == 0
    assert report["no_op_positive_gain_eligible"] is False
    assert report["source_execution"]["job"] != report["no_op_execution"]["job"]
    assert report["export"]["transaction"]["changed_parameters"] == 0
    assert report["original_accuracy"] == report["no_op_accuracy"] == 0.5
    export_root = report["export"]["export_root"]
    del witness, noop, model
    receipt = json.loads(verify_reload(
        runtime, tmp_path / "store", export_root, expected_manifest=frozen_manifest,
        probes=[Probe((1.0, 2.0), frozen_outputs, 0.0)],
    ))
    assert receipt["inference"]["cleanup_confirmed"] is True
    assert receipt["inference"]["job"] not in {
        report["source_execution"]["job"], report["no_op_execution"]["job"],
    }
    assert receipt["promotion_status"] == "not_evaluated" and not receipt["outputs_eligible"]
    assert arithmetic.fixture().checkpoint_bytes() == source_bytes
    assert not (tmp_path / "store" / "active").exists()
    with pytest.raises(ReloadVerificationError, match="parity failed"):
        verify_reload(runtime, tmp_path / "store", export_root, expected_manifest=frozen_manifest,
                      probes=[Probe((1.0, 2.0), (999.0,) * 5, 0.0)])
    token = nf.CancellationHandle()
    token.cancel()
    with pytest.raises(ReloadVerificationError, match="cancelled before reading"):
        verify_reload(runtime, tmp_path / "store", export_root, expected_manifest=frozen_manifest,
                      probes=[Probe((1.0, 2.0), frozen_outputs, 0.0)], cancellation=token)
    assert not (tmp_path / "store" / "active").exists()
    bundle = json.loads((tmp_path / "store" / "blobs" / export_root[7:]).read_bytes())
    checkpoint = tmp_path / "store" / "blobs" / bundle["checkpoint"]["digest"][7:]
    damaged = bytearray(checkpoint.read_bytes())
    damaged[-1] ^= 1
    # CAS blobs are read-only: simulate same-UID storage tampering by replacing
    # this test-owned temporary blob, not weakening production permissions.
    checkpoint.unlink()
    checkpoint.write_bytes(damaged)
    with pytest.raises(nf.NativeError):
        verify_reload(runtime, tmp_path / "store", export_root, expected_manifest=frozen_manifest,
                      probes=[Probe((1.0, 2.0), frozen_outputs, 0.0)])
    assert not (tmp_path / "store" / "active").exists()


def test_m2_vertical_slice_cancelled_control_never_caches_partial_evidence(tmp_path: Path) -> None:
    (tmp_path / "runtime").mkdir(mode=0o700)
    runtime = models.qualify(tmp_path / "runtime", enabled=True)
    model = arithmetic.fixture()
    plan, smoke = repair_plan(model, factor=1.0)
    cache = controls.SourceCache()
    token = nf.CancellationHandle()
    token.cancel()
    with pytest.raises(nf.CancelledError):
        controls.run(cache, runtime, plan, smoke, protocol(model), tmp_path / "store", token)
    assert cache.entries == 0
    limits = protocol(model)
    limits["worker_limits"]["wall_ms"] = 100
    with pytest.raises(nf.NativeError, match="incomplete/failed"):
        controls.run(cache, runtime, plan, smoke, limits, tmp_path / "store")
    assert cache.entries == 0 and not (tmp_path / "store" / "active").exists()


def test_m2_vertical_slice_frozen_holdout_real_domains_close_search_and_survive_reopen(
    tmp_path: Path,
) -> None:
    # Actual SQLite/CAS/broker; its linear catalogue is NOT the arithmetic catalogue.
    with split_controller(tmp_path / "controller") as controller:
        proposer = controller.ok("grant", recipient="proposer")
        request = split_request(proposer)
        assert controller.ok("read", request=request)["label"] is None
        assert controller.send("read", request=split_request(proposer, labels=True))[
            "error"]["code"] == "PERMISSION_DENIED"
        prepared = controller.ok("prepare", request=request)
        code = f'''import json,sys,os
from pathlib import Path
p=json.load(sys.stdin)
assert os.getuid()==65534 and p['split']=='development' and p['label'] is None
assert 'PRIVATE_LABEL' not in json.dumps(p)
assert not Path({str(tmp_path / 'controller')!r}).exists()
assert 'NoNewPrivs:\\t1' in Path('/proc/self/status').read_text()
assert 'CapEff:\\t0000000000000000' in Path('/proc/self/status').read_text()
Path('/out/proposer-only').write_text('must not cross evaluator domains')
print('development-only')
'''
        development = controller.ok("domain", request=request, code=code)
        assert development["status"] == "completed" and development["cleanup_confirmed"] is True
        assert development["outputs_eligible"] is False
        assert bytes.fromhex(development["report"]["stdout_hex"]) == b"development-only\n"
        assert "error" in controller.send("grant", recipient="final_evaluator")
        controller.ok("final_phase")
        assert "error" in controller.send("grant", recipient="final_evaluator")
        panel = controller.ok("freeze")["panel"]
        assert controller.ok("freeze")["panel"] == panel
        assert "error" in controller.send("freeze", candidate_salt="changed-candidate")
        final = controller.ok("grant", recipient="final_evaluator")
        assert final["panel"] == panel
        code = '''import json,sys,os
from pathlib import Path
p=json.load(sys.stdin)
assert os.getuid()==65534 and p['split']=='holdout'
assert p['label']['secret']=='PRIVATE_LABEL_0_holdout'
assert not Path('/var/run/docker.sock').exists()
assert not Path('/out/proposer-only').exists()
print('frozen-holdout-only')
'''
        receipt = controller.ok("domain", request=split_request(final, labels=True), code=code)
        assert receipt["status"] == "completed" and receipt["cleanup_confirmed"] is True
        assert receipt["outputs_eligible"] is False
        assert bytes.fromhex(receipt["report"]["stdout_hex"]) == b"frozen-holdout-only\n"
        for op, fields in [
            ("read", {"request": request}), ("search", {}), ("reserve_search", {}),
            ("dispatch_prepared", {"access": prepared["access"]}),
        ]:
            assert "error" in controller.send(op, **fields), op
        forged = copy.deepcopy(split_request(final, labels=True))
        forged["lease_id"] = proposer["lease_id"]
        assert "error" in controller.send("read", request=forged)
        controller.ok("reopen")
        assert controller.ok("panel")[0] == panel
        assert controller.ok("read", request=split_request(final, labels=True))[
            "classification"] == "holdout"
        controller.ok("stop")
        assert "error" in controller.send("read", request=split_request(final, labels=True))


def test_m2_vertical_slice_renamed_consumed_holdout_cannot_reenter_search(tmp_path: Path) -> None:
    with split_controller(tmp_path / "controller") as controller:
        controller.ok("final_phase")
        controller.ok("freeze")
        final = controller.ok("grant", recipient="final_evaluator")
        controller.ok("read", request=split_request(final, labels=True))
        # This fixture operation deliberately freezes a new revision before
        # detecting the alias. Do not reuse the superseded old lease afterward.
        denied = controller.send("revision_alias")["error"]
        assert denied["code"] == "SCOPE_VIOLATION" and "history" in denied["message"]


def test_m2_vertical_slice_actual_sandbox_cancellation_is_not_publishable(tmp_path: Path) -> None:
    root = tmp_path / "sandbox"
    receipt = sandbox(root, "import time\nprint('partial',flush=True)\ntime.sleep(60)",
                      cancel_after_ms=2000,
                      limits={"wall_ms": 5000, "cpu_ms": 5000, "output_bytes": 4096})
    assert receipt["status"] == "cancelled" and receipt["cleanup_confirmed"] is True
    assert receipt["outputs_eligible"] is False and receipt["report"] is None
    assert receipt["promotion_status"] == "not_evaluated"
    state = json.loads((root / receipt["job"] / "pre_cleanup.json").read_bytes())["State"]
    assert state["StartedAt"] != "0001-01-01T00:00:00Z"


def test_m2_vertical_slice_actual_sandbox_crash_recovery_retains_incomplete_costs(
    tmp_path: Path,
) -> None:
    from m2_fixture_support import TARGET

    root = tmp_path / "sandbox"
    root.mkdir(mode=0o700)
    result = run([str(TARGET / "debug/examples/fixture_sandbox")], data=json.dumps({
        "root": str(root), "enabled": True, "source": "import time\ntime.sleep(60)",
        "limits": {"wall_ms": 5000, "cpu_ms": 5000, "output_bytes": 128},
        "crash_after_ms": 2000,
    }), timeout=20)
    assert result.returncode == 86, result.stdout + result.stderr
    pending = [path for path in root.glob("*/intent.json")
               if not path.with_name("completion.json").exists()]
    assert len(pending) == 1 and pending[0].with_name("configured.json").is_file()
    sandbox(root, recover=True)  # Real requalification reconciles durable launch intent.
    recovered = json.loads(pending[0].with_name("recovery.json").read_bytes())
    assert recovered["status"] == "incomplete" and recovered["cleanup_confirmed"] is True
    assert recovered["outputs_eligible"] is False and recovered["report"] is None
    assert recovered["storage"]["accounting_finalized"] is True
    assert recovered["storage"]["retained_bytes"] > 0
    assert recovered["storage"]["workspace_reclaimed"] is True
    assert not pending[0].with_name("reservation.bin").exists()
    assert sandbox(root, recover=True)["recovered"] == 0


def test_m2_vertical_slice_manual_frozen_partial_evidence_stop_and_recovery(
    tmp_path: Path,
) -> None:
    rust_case("manual_workflow", "manual_workflow_emit_fixture",
              NF_T041_TEST_FIXTURE_OUTPUT=str(tmp_path))
    request = json.loads((tmp_path / "request.json").read_bytes())
    request.pop("schema_version")
    workflow = nf.ResearchWorkflow(**request)
    request["contract"]["intervention"]["method_protocol"] = workflow.plan()["manifest_digest"]
    workflow = nf.ResearchWorkflow(**request)
    plan = workflow.plan()
    assert plan["execution"] == "not_run" and plan["authorization_verified"] is False
    ledger = tmp_path / "ledger"
    frozen = workflow.freeze(ledger=ledger, expected_plan_digest=plan["plan_digest"])
    assert frozen == workflow.freeze(ledger=ledger, expected_plan_digest=plan["plan_digest"])
    args = {"ledger": ledger, "contract_digest": frozen["contract_digest"]}
    before = workflow.status(**args)
    # Partial/missing/forged receipt bytes cannot become a native execution grant.
    for partial in ({"accepted": True}, {"status": "incomplete"}, {"promotion_authorized": True}):
        raw = _native.canonical_json(json.dumps(partial).encode())
        digest = _native.content_digest(raw)
        (Path(request["store"]) / "blobs" / digest[7:]).write_bytes(raw)
        with pytest.raises(nf.NativeError) as denied:
            workflow.run(**args, authorization=digest)
        assert denied.value.code == "PERMISSION_DENIED"
    stopped = workflow.stop(**args)
    assert stopped["request_status"] == "persisted"
    assert stopped["cleanup_status"] == "not_confirmed_by_request"
    recovered = workflow.resume(**args)
    assert recovered["mode"] == "recovery_only" and recovered["execution"] == "not_resumed"
    assert recovered["stop_request"]["reason"] == "owner_stop"
    after = workflow.status(**args)
    assert before["split_evidence"] == after["split_evidence"]
    assert before["budgets"]["aggregate"]["spent"] == after["budgets"]["aggregate"]["spent"]
    assert after["authority"]["accepted"] is False
    assert after["authority"]["promotion_authorized"] is False
    assert after["authority"]["independently_verified"] is False
    report = workflow.report(**args)
    assert json.loads(report["json"])["snapshot"] == after["snapshot"]
    assert "not acceptance authority" in report["markdown"]
    with sqlite3.connect(ledger / "events.sqlite3") as connection:
        for table in ("accepted_records", "accepted_heads", "verified_frontier"):
            assert connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0


@pytest.mark.parametrize("case", [
    "acceptance_gate_trusted_fixture_accepts_actual_execution_and_durable_reopen",
    "acceptance_gate_trusted_fixture_retention_damage_never_accepts",
    "acceptance_gate_trusted_fixture_actual_noop_and_each_observation_match_executed_arm",
    "acceptance_gate_trusted_fixture_distinct_best_known_has_distinct_executed_outputs",
    "acceptance_gate_trusted_fixture_missing_catalog_is_durable_incomplete_not_acceptance",
    "acceptance_gate_trusted_fixture_serialized_success_and_stale_lease_cannot_authorize",
    "acceptance_gate_trusted_fixture_receipts_are_immutable_and_missing_or_forged_never_accept",
    "acceptance_gate_trusted_fixture_interruption_after_durable_intent_stays_incomplete",
])
def test_m2_vertical_slice_supervised_linear_acceptance_component(case: str) -> None:
    # Reuse actual Rust/guardian assertions, never manufacture Python success DTOs.
    rust_case("acceptance_gate_trusted_fixture", case)


@pytest.mark.parametrize("case", [
    "reload_promotion_review_block_is_unconditional_and_has_no_side_effects",
    "reload_promotion_native_qualification_and_arbitrary_digest_cannot_authorize",
])
def test_m2_vertical_slice_native_promotion_scope_and_partial_evidence(case: str) -> None:
    # The explicit operational fixture grant is not strict/public M2 qualification.
    rust_case("promotion_atomic", case)
