#![cfg(target_os = "linux")]
#[path = "../../nf-weight-ir/tests/common/mod.rs"]
#[allow(dead_code)]
mod common;
#[path = "../../nf-weight-ir/tests/common/compiler_fixture.rs"]
#[allow(dead_code)]
mod compiler_fixture;
#[path = "../../nf-core/tests/common/study_fixture.rs"]
#[allow(dead_code)]
mod fixture;
#[path = "support/manual_workflow.rs"]
mod support;
use nf_core::error::ErrorCode;
use nf_experiments::coordinator::{Ledger, LedgerLimits, Stage, Transition};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::{fs, path::PathBuf};
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Root(PathBuf);
impl Root {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "nf-t041-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Root {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn tree(path: &std::path::Path) -> Vec<(PathBuf, Vec<u8>)> {
    let mut result = Vec::new();
    for entry in fs::read_dir(path).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            result.extend(tree(&path));
        } else {
            result.push((path.clone(), fs::read(path).unwrap()));
        }
    }
    result.sort();
    result
}
#[test]
fn manual_workflow_plan_compiles_real_bytes_without_execution_or_any_file_change() {
    let root = Root::new();
    let request = support::setup(&root.0);
    let before = tree(&root.0);
    let plan = support::call("plan", &request).unwrap();
    assert_eq!(plan, support::call("validate", &request).unwrap());
    assert_eq!(plan["valid"], true);
    assert_eq!(plan["execution"], "not_run");
    assert_eq!(plan["authorization_verified"], false);
    assert_eq!(plan["compiled_plan"]["affected_parameter_upper_bound"], 4);
    assert_eq!(tree(&root.0), before);
    assert!(!root.0.join("ledger").exists());
}
#[test]
fn manual_workflow_cross_record_errors_and_absent_store_are_pure() {
    let root = Root::new();
    let request = support::setup(&root.0);
    for (pointer, value) in [
        ("/program/objective/contract", json!(fixture::d("wrong"))),
        (
            "/contract/intervention/scope",
            json!(fixture::d("wrong-scope")),
        ),
        (
            "/contract/intervention/allowed_engines/0/config",
            json!(fixture::d("wrong-backend")),
        ),
        (
            "/program/verification/gyms/0/split",
            json!(fixture::d("holdout")),
        ),
        (
            "/program/verification/gyms/0/scorer",
            json!(fixture::d("wrong")),
        ),
        ("/policy/permissions/allow_downloads", json!(true)),
        ("/source_export", json!(fixture::d("absent"))),
        ("/smoke_suite/probes/0/max_abs_output", json!(999.0)),
        ("/store", json!(root.0.join("missing"))),
    ] {
        let mut changed = request.clone();
        *changed.pointer_mut(pointer).unwrap() = value;
        let before = tree(&root.0);
        assert!(support::call("plan", &changed).is_err(), "{pointer}");
        assert_eq!(before, tree(&root.0));
    }
    assert!(!root.0.join("missing").exists());
    assert_eq!(
        nf_experiments::manual_workflow::dispatch("plan", b"{\"x\":1,\"x\":2}")
            .unwrap_err()
            .code,
        ErrorCode::DuplicateKey
    );
}
#[test]
fn manual_workflow_freeze_is_immutable_idempotent_and_manifest_bound_not_approval() {
    let root = Root::new();
    let request = support::bound(support::setup(&root.0));
    let plan = support::call("plan", &request).unwrap();
    let mut freeze = json!({"plan":request,"ledger":root.0.join("ledger"),"expected_plan_digest":fixture::d("stale")});
    assert_eq!(
        support::call("freeze", &freeze).unwrap_err().code,
        ErrorCode::SourceMismatch
    );
    assert!(!root.0.join("ledger").exists());
    freeze["expected_plan_digest"] = plan["plan_digest"].clone();
    let frozen = support::call("freeze", &freeze).unwrap();
    assert_eq!(support::call("freeze", &freeze).unwrap(), frozen);
    assert_eq!(frozen["authorization_verified"], false);
    let existing = support::existing(&root.0, &frozen);
    let status = support::call("status", &existing).unwrap();
    assert_eq!(status["status"]["stage"], "frozen");
    assert_eq!(status["events"].as_array().unwrap().len(), 1);
    assert_eq!(status["authority"]["accepted"], false);
    let reader = nf_checkpoint::store::ArtifactStore::open_read_only(
        &root.0.join("store"),
        nf_checkpoint::store::StoreLimits {
            total_bytes: 536_870_912,
            max_blob_bytes: 67_108_864,
            max_entries: 4096,
        },
    )
    .unwrap();
    let id = plan["manifest_digest"].as_str().unwrap().parse().unwrap();
    assert!(reader.contains_verified(&id).unwrap());
    for key in ["program", "search_space", "policy", "smoke_suite"] {
        let id = plan["manifest"][key].as_str().unwrap().parse().unwrap();
        assert!(reader.contains_verified(&id).unwrap());
    }
    let mut changed = request;
    changed["contract"]["objective"]["description"] = json!("changed rules");
    let plan = support::call("plan", &changed).unwrap();
    assert_eq!(support::call("freeze",&json!({"plan":changed,"ledger":root.0.join("ledger"),"expected_plan_digest":plan["plan_digest"]})).unwrap_err().code,ErrorCode::RevisionConflict);
}
#[test]
fn manual_workflow_stop_recovery_reports_preserve_stop_budget_and_terminal_fences() {
    let root = Root::new();
    let request = support::bound(support::setup(&root.0));
    let frozen = support::freeze(&root.0, &request);
    let existing = support::existing(&root.0, &frozen);
    let before = support::call("status", &existing).unwrap();
    let stop = support::call("stop", &existing).unwrap();
    assert_eq!(stop["request_status"], "persisted");
    assert_eq!(stop, support::call("stop", &existing).unwrap());
    let recovered = support::call("resume", &existing).unwrap();
    assert_eq!(recovered["execution"], "not_resumed");
    assert_eq!(recovered["stop_request"]["reason"], "owner_stop");
    let after = support::call("status", &existing).unwrap();
    assert_eq!(
        before["budgets"]["aggregate"]["spent"],
        after["budgets"]["aggregate"]["spent"]
    );
    assert_eq!(before["split_evidence"], after["split_evidence"]);
    let report = support::call("report", &existing).unwrap();
    let report_json: Value = serde_json::from_str(report["json"].as_str().unwrap()).unwrap();
    assert_eq!(
        report_json["status"]["study_stop_reason"]["status"],
        "recorded"
    );
    assert!(report["markdown"]
        .as_str()
        .unwrap()
        .contains("not acceptance authority"));
    let mut ledger = Ledger::open(&root.0.join("ledger"), LedgerLimits::default()).unwrap();
    let id = frozen["contract_digest"].as_str().unwrap().parse().unwrap();
    let lease = ledger.acquire(&id, "finish", 30000).unwrap();
    ledger
        .advance(
            &lease,
            Transition {
                operation: fixture::d("cancel"),
                expected: Stage::Frozen,
                next: Stage::Cancelled,
                protocol: fixture::d("protocol"),
                evidence: fixture::d("incomplete"),
                message: "owner stopped; no execution".into(),
            },
        )
        .unwrap();
    assert_eq!(
        support::call("resume", &existing).unwrap_err().code,
        ErrorCode::InvalidTransition
    );
}
#[test]
fn manual_workflow_control_survives_missing_artifacts_and_rejects_live_lease_takeover() {
    let root = Root::new();
    let request = support::bound(support::setup(&root.0));
    let frozen = support::freeze(&root.0, &request);
    let existing = support::existing(&root.0, &frozen);
    let id = frozen["contract_digest"].as_str().unwrap().parse().unwrap();
    let mut ledger = Ledger::open(&root.0.join("ledger"), LedgerLimits::default()).unwrap();
    let active = ledger.acquire(&id, "active-owner", 300000).unwrap();
    assert_eq!(
        support::call("resume", &existing).unwrap_err().code,
        ErrorCode::LeaseConflict
    );
    ledger.release(&active).unwrap();
    fs::remove_dir_all(root.0.join("store")).unwrap();
    assert_eq!(
        support::call("stop", &existing).unwrap()["request_status"],
        "persisted"
    );
    assert_eq!(
        support::call("resume", &existing).unwrap()["execution"],
        "not_resumed"
    );
    assert!(!root.0.join("store").exists());
}
#[test]
fn manual_workflow_run_never_treats_a_reference_as_authority_or_touches_paths() {
    let root = Root::new();
    let frozen = json!({"contract_digest":fixture::d("not_frozen")});
    let mut request = support::existing(&root.0, &frozen);
    assert_eq!(
        support::call("run", &request).unwrap_err().code,
        ErrorCode::PermissionDenied
    );
    request["authorization"] = json!(fixture::d("forged"));
    assert_eq!(
        support::call("run", &request).unwrap_err().code,
        ErrorCode::PermissionDenied
    );
    assert!(tree(&root.0).is_empty());
    for operation in ["stop", "resume", "status", "report"] {
        assert!(support::call(operation, &request).is_err());
    }
    assert!(tree(&root.0).is_empty());
}
#[test]
fn manual_workflow_recovery_reaps_real_fixture_process_without_restarting_or_erasing_costs() {
    use nf_experiments::coordinator::budgets::{Request, Resources, WorkKind};
    use nf_experiments::coordinator::supervision::{FixtureMode, NativeJob, TrustedNativeWorker};
    use std::os::unix::fs::PermissionsExt;
    let root = Root::new();
    let mut request = support::bound(support::setup(&root.0));
    request["contract"]["limits"]["host_memory_bytes"] = json!(268435456);
    let frozen = support::freeze(&root.0, &request);
    let existing = support::existing(&root.0, &frozen);
    let id = frozen["contract_digest"].as_str().unwrap().parse().unwrap();
    let mut ledger = Ledger::open(&root.0.join("ledger"), LedgerLimits::default()).unwrap();
    let lease = ledger.acquire(&id, "prior-coordinator", 300000).unwrap();
    let binary = root.0.join("fixture-worker");
    assert!(std::process::Command::new("/usr/bin/strip")
        .args(["--strip-debug", "-o"])
        .arg(&binary)
        .arg(env!("CARGO_BIN_EXE_nf-fixture-worker"))
        .status()
        .unwrap()
        .success());
    fs::set_permissions(&binary, fs::Permissions::from_mode(0o500)).unwrap();
    let approved = nf_core::canonical::ContentDigest::of_bytes(&fs::read(&binary).unwrap());
    let worker = TrustedNativeWorker::qualify(&binary, approved).unwrap();
    let reservation = ledger
        .reserve(
            &lease,
            Request {
                schema_version: 1,
                operation: fixture::d("metadata-fixture"),
                attempt: 0,
                parent: None,
                kind: WorkKind::Metadata,
                resources: Resources {
                    cpu_ms: 4000,
                    host_bytes: 134217728,
                    storage_bytes: 1048576,
                    workers: 1,
                    ..Resources::default()
                },
                wall_ms: 3000,
                external: None,
            },
        )
        .unwrap();
    let running = ledger
        .launch_native(
            &lease,
            &reservation,
            &worker,
            NativeJob {
                mode: FixtureMode::Sleep,
                maximum_output_bytes: 1024,
                termination_grace_ms: 30,
            },
        )
        .unwrap();
    let worker_id = running.id().clone();
    let old_fence = lease.fencing_token();
    // Transfer recovery ownership without handing the opaque old lease to SDK.
    ledger.release(&lease).unwrap();
    // Simulate a crash before run returned an ID: use durable intent discovery.
    let result = support::call("resume", &existing).unwrap();
    assert_eq!(result["recovered"][0]["worker"], json!(worker_id));
    assert_eq!(result["execution"], "not_resumed");
    assert_eq!(result["recovered"][0]["report"]["outcome"], "incomplete");
    assert_eq!(result["recovered"][0]["report"]["cleanup_confirmed"], true);
    assert_eq!(result["recovered"][0]["report"]["outputs_eligible"], false);
    assert_eq!(result["stop_request"]["reason"], "recovery");
    assert!(ledger.state(&id).unwrap().fencing_token > old_fence);
    let status = support::call("status", &existing).unwrap();
    assert!(
        status["budgets"]["aggregate"]["unknown_charged"]["cpu_ms"]
            .as_u64()
            .unwrap()
            >= 4000
    );
    assert_eq!(status["budgets"]["aggregate"]["held"]["workers"], 0);
    assert_eq!(
        support::call("resume", &existing).unwrap()["recovered"],
        result["recovered"]
    );
    drop(running);
}
#[test]
fn manual_workflow_emit_fixture() {
    // Optional real fixture export for cross-language tests; pytest owns cleanup.
    if let Some(output) = std::env::var_os("NF_T041_TEST_FIXTURE_OUTPUT") {
        let root = PathBuf::from(output);
        fs::create_dir_all(&root).unwrap();
        let request = support::setup(&root);
        fs::write(
            root.join("request.json"),
            serde_json::to_vec(&request).unwrap(),
        )
        .unwrap();
    }
}
