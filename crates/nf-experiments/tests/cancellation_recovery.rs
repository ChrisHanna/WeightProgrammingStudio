#![cfg(target_os = "linux")]
#[path = "../../nf-core/tests/common/study_fixture.rs"]
#[allow(dead_code)]
mod fixture;
use fixture::{d, standard};
use nf_core::canonical::ContentDigest;
use nf_core::schemas::Validated;
use nf_core::study::StudyContract;
use nf_experiments::coordinator::budgets::{Request, Reservation, Resources, WorkKind};
use nf_experiments::coordinator::supervision::{
    FixtureMode, NativeJob, RunningNative, StopReason, TrustedNativeWorker, WorkerOutcome,
    WorkerReport,
};
use nf_experiments::coordinator::{Lease, Ledger, LedgerLimits, Stage, Transition};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Root(PathBuf);
impl Root {
    fn new() -> Self {
        let root = PathBuf::from(std::env::var_os("KIROCREW_SCRATCH").unwrap()).join(format!(
            "supervised-tests-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        Self(root)
    }
    fn ledger(&self) -> Ledger {
        Ledger::open(&self.0.join("ledger"), LedgerLimits::default()).unwrap()
    }
}
impl Drop for Root {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn setup(root: &Root) -> (Ledger, ContentDigest, Lease) {
    let mut ledger = root.ledger();
    let mut value = standard();
    value["limits"]["host_memory_bytes"] = serde_json::json!(268435456);
    let contract = Validated::<StudyContract>::parse(&serde_json::to_vec(&value).unwrap()).unwrap();
    let id = ledger.freeze(&contract).unwrap();
    let lease = ledger.acquire(&id, "supervisor", 300000).unwrap();
    (ledger, id, lease)
}
fn worker(root: &Root) -> TrustedNativeWorker {
    use std::os::unix::fs::PermissionsExt;
    let path = root.0.join("fixture-worker");
    // Keep the unchanged executable-size guard: discard only test DWARF data.
    // The operator approval digest below binds the actual stripped snapshot.
    assert!(std::process::Command::new("/usr/bin/strip")
        .arg("--strip-debug")
        .arg("-o")
        .arg(&path)
        .arg(env!("CARGO_BIN_EXE_nf-fixture-worker"))
        .status()
        .unwrap()
        .success());
    fs::set_permissions(&path, fs::Permissions::from_mode(0o500)).unwrap();
    let digest = ContentDigest::of_bytes(&fs::read(&path).unwrap());
    TrustedNativeWorker::qualify(&path, digest).unwrap()
}
fn reserve(
    ledger: &mut Ledger,
    lease: &Lease,
    name: &str,
    wall_ms: u64,
    cpu_ms: u64,
) -> Reservation {
    ledger
        .reserve(
            lease,
            Request {
                schema_version: 1,
                operation: d(name),
                attempt: 0,
                parent: None,
                kind: WorkKind::Metadata,
                resources: Resources {
                    cpu_ms,
                    host_bytes: 134217728,
                    storage_bytes: 1048576,
                    workers: 1,
                    ..Resources::default()
                },
                wall_ms,
                external: None,
            },
        )
        .unwrap()
}
fn job(mode: FixtureMode) -> NativeJob {
    NativeJob {
        mode,
        maximum_output_bytes: 1024,
        termination_grace_ms: 30,
    }
}
fn finish(
    ledger: &mut Ledger,
    lease: &Lease,
    reservation: &Reservation,
    running: &mut RunningNative,
) -> WorkerReport {
    let deadline = Instant::now() + Duration::from_secs(8);
    loop {
        if let Some(report) = ledger.poll_native(lease, reservation, running).unwrap() {
            return report;
        }
        assert!(
            Instant::now() < deadline,
            "bounded native supervision did not finish"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}
#[test]
fn cancellation_recovery_real_native_limits_handshake_completion_and_immutable_report() {
    let root = Root::new();
    let (mut ledger, id, lease) = setup(&root);
    let qualified = worker(&root);
    let reservation = reserve(&mut ledger, &lease, "complete", 3000, 4000);
    let mut running = ledger
        .launch_native(&lease, &reservation, &qualified, job(FixtureMode::Complete))
        .unwrap();
    let report = finish(&mut ledger, &lease, &reservation, &mut running);
    assert_eq!(report.outcome, WorkerOutcome::Completed);
    assert!(report.cleanup_confirmed && report.outputs_eligible);
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&report.stdout).unwrap()["result"],
        4
    );
    assert!(report.cpu_ms.is_some());
    assert!(report.leader_peak_rss_bytes.is_some());
    assert_eq!(ledger.budget_snapshot(&id).unwrap().held.workers, 0);
    assert!(
        ledger
            .worker_state(running.id())
            .unwrap()
            .report
            .unwrap()
            .outputs_eligible
    );
}
#[test]
fn cancellation_recovery_persists_owner_stop_before_signals_and_blocks_success_or_resume() {
    let root = Root::new();
    let (mut ledger, id, lease) = setup(&root);
    let qualified = worker(&root);
    let reservation = reserve(&mut ledger, &lease, "partial", 3000, 4000);
    let mut running = ledger
        .launch_native(&lease, &reservation, &qualified, job(FixtureMode::Partial))
        .unwrap();
    let stop = ledger
        .request_stop(&id, d("user-stop"), StopReason::OwnerStop)
        .unwrap();
    assert_eq!(stop.stage, Stage::Frozen);
    assert!(ledger.stop_record(&id).unwrap().is_some());
    let report = finish(&mut ledger, &lease, &reservation, &mut running);
    assert_eq!(report.outcome, WorkerOutcome::Cancelled);
    assert!(report.cleanup_confirmed);
    assert!(!report.outputs_eligible);
    assert!(report.stdout.is_empty());
    let blocked = ledger.reserve(
        &lease,
        Request {
            schema_version: 1,
            operation: d("forbidden-restart"),
            attempt: 0,
            parent: None,
            kind: WorkKind::Metadata,
            resources: Resources {
                host_bytes: 1024,
                workers: 1,
                ..Resources::default()
            },
            wall_ms: 1000,
            external: None,
        },
    );
    assert!(blocked.is_err());
    assert!(ledger
        .advance(
            &lease,
            Transition {
                operation: d("false-success"),
                expected: Stage::Frozen,
                next: Stage::Preflight,
                protocol: d("p"),
                evidence: d("e"),
                message: "not allowed".to_owned()
            }
        )
        .is_err());
    assert!(ledger.budget_snapshot(&id).unwrap().spent.cpu_ms > 0 || report.cpu_ms == Some(0));
}
#[test]
fn cancellation_recovery_owner_stop_is_durable_at_every_manual_workflow_stage() {
    let binary_root = Root::new();
    let qualified = worker(&binary_root);
    let stages = [
        Stage::Frozen,
        Stage::Preflight,
        Stage::Profile,
        Stage::Baseline,
        Stage::Diagnose,
        Stage::Design,
        Stage::Execute,
        Stage::Evaluate,
        Stage::Search,
        Stage::Verify,
        Stage::Learn,
    ];
    for (index, target) in stages.iter().enumerate() {
        let root = Root::new();
        let (mut ledger, id, lease) = setup(&root);
        for (edge, pair) in stages[..=index].windows(2).enumerate() {
            ledger
                .advance(
                    &lease,
                    Transition {
                        operation: d(&format!("edge{edge}")),
                        expected: pair[0],
                        next: pair[1],
                        protocol: d("workflow"),
                        evidence: d("fixture evidence"),
                        message: "fixture workflow state".to_owned(),
                    },
                )
                .unwrap();
        }
        let reservation = reserve(&mut ledger, &lease, "stage-work", 3000, 4000);
        let mut running = ledger
            .launch_native(
                &lease,
                &reservation,
                &qualified,
                job(FixtureMode::IgnoreTerm),
            )
            .unwrap();
        ledger
            .request_stop(&id, d("owner-stop"), StopReason::OwnerStop)
            .unwrap();
        let cancelled = Transition {
            operation: d("cancelled"),
            expected: *target,
            next: Stage::Cancelled,
            protocol: d("stop-protocol"),
            evidence: d("cleanup"),
            message: "cancelled after actual cleanup".to_owned(),
        };
        assert!(ledger.advance(&lease, cancelled.clone()).is_err());
        let report = finish(&mut ledger, &lease, &reservation, &mut running);
        assert!(report.cleanup_confirmed);
        assert!(!report.outputs_eligible);
        ledger.advance(&lease, cancelled).unwrap();
        assert_eq!(ledger.state(&id).unwrap().stage, Stage::Cancelled);
    }
}

#[test]
fn cancellation_recovery_kills_actual_native_descendant_group_not_only_parent() {
    let root = Root::new();
    let (mut ledger, id, lease) = setup(&root);
    let qualified = worker(&root);
    let reservation = reserve(&mut ledger, &lease, "descendants", 4000, 8000);
    let mut running = ledger
        .launch_native(
            &lease,
            &reservation,
            &qualified,
            job(FixtureMode::SpawnChild),
        )
        .unwrap();
    let group = running.identity().group;
    let deadline = Instant::now() + Duration::from_secs(2);
    let members = loop {
        let members: Vec<u32> = fs::read_dir("/proc")
            .unwrap()
            .filter_map(|entry| entry.ok()?.file_name().to_str()?.parse().ok())
            .filter(|pid| {
                let Ok(text) = fs::read_to_string(format!("/proc/{pid}/stat")) else {
                    return false;
                };
                let Some(end) = text.rfind(')') else {
                    return false;
                };
                text[end + 1..]
                    .split_whitespace()
                    .nth(2)
                    .and_then(|value| value.parse::<u32>().ok())
                    == Some(group)
            })
            .collect();
        if members.len() >= 2 {
            break members;
        }
        assert!(
            Instant::now() < deadline,
            "fixture did not spawn actual child"
        );
        std::thread::sleep(Duration::from_millis(5));
    };
    ledger
        .request_stop(&id, d("stop-group"), StopReason::OwnerStop)
        .unwrap();
    let report = finish(&mut ledger, &lease, &reservation, &mut running);
    assert!(report.cleanup_confirmed);
    assert!(!report.outputs_eligible);
    assert_eq!(ledger.budget_snapshot(&id).unwrap().held.workers, 0);
    for pid in members {
        if let Ok(text) = fs::read_to_string(format!("/proc/{pid}/stat")) {
            let end = text.rfind(')').unwrap();
            assert_eq!(text[end + 1..].split_whitespace().next(), Some("Z"));
        }
    }
}

#[test]
fn cancellation_recovery_real_coordinator_crash_reconciles_without_resetting_costs() {
    let root = Root::new();
    let _qualified = worker(&root);
    let binary = root.0.join("fixture-worker");
    let mut value = standard();
    value["limits"]["host_memory_bytes"] = serde_json::json!(268435456);
    let contract_file = root.0.join("contract.json");
    fs::write(&contract_file, serde_json::to_vec(&value).unwrap()).unwrap();
    let child = std::process::Command::new(&binary)
        .args([
            "--controller-crash",
            contract_file.to_str().unwrap(),
            root.0.join("ledger").to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(
        child.status.code(),
        Some(86),
        "{}",
        String::from_utf8_lossy(&child.stderr)
    );
    let proof: serde_json::Value = serde_json::from_slice(&child.stdout).unwrap();
    let id: ContentDigest = proof["contract"].as_str().unwrap().parse().unwrap();
    let worker_id: ContentDigest = proof["worker"].as_str().unwrap().parse().unwrap();
    std::thread::sleep(Duration::from_millis(1100));
    let mut ledger = root.ledger();
    let lease = ledger.acquire(&id, "recovery-owner", 300000).unwrap();
    let before = ledger.budget_snapshot(&id).unwrap();
    assert_eq!(before.held.workers, 1);
    let report = ledger.recover_native(&lease, &worker_id).unwrap();
    assert_eq!(report.outcome, WorkerOutcome::Incomplete);
    assert!(report.cleanup_confirmed);
    assert!(!report.outputs_eligible);
    let after = ledger.budget_snapshot(&id).unwrap();
    assert_eq!(after.spent.cpu_ms, 8000);
    assert_eq!(after.unknown_charged.cpu_ms, 8000);
    assert_eq!(after.held.workers, 0);
    assert!(after.elapsed_wall_ms.unwrap() >= before.elapsed_wall_ms.unwrap());
    ledger.release(&lease).unwrap();
    let mut revision = value;
    revision["revision"] = serde_json::json!(2);
    revision["parent_contract"] = serde_json::json!(id);
    revision["approval_receipt"] = serde_json::json!(d("explicit-new-owner-revision"));
    let contract =
        Validated::<StudyContract>::parse(&serde_json::to_vec(&revision).unwrap()).unwrap();
    let next = ledger.freeze(&contract).unwrap();
    assert_eq!(ledger.budget_snapshot(&next).unwrap().spent.cpu_ms, 8000);
}

#[test]
fn cancellation_recovery_reused_identity_is_not_signalled_and_cleanup_stays_unknown() {
    let root = Root::new();
    let (mut ledger, id, lease) = setup(&root);
    let qualified = worker(&root);
    let reservation = reserve(&mut ledger, &lease, "identity", 4000, 4000);
    let running = ledger
        .launch_native(&lease, &reservation, &qualified, job(FixtureMode::Sleep))
        .unwrap();
    let worker_id = running.id().clone();
    let pid = running.identity().pid;
    let direct = rusqlite::Connection::open(root.0.join("ledger/events.sqlite3")).unwrap();
    let body: Vec<u8> = direct
        .query_row(
            "SELECT body FROM worker_updates WHERE worker=?1 AND version=1",
            [worker_id.as_str()],
            |row| row.get(0),
        )
        .unwrap();
    let mut value: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let start = value["observation"]["identity"]["start_ticks"]
        .as_u64()
        .unwrap();
    value["observation"]["identity"]["start_ticks"] = serde_json::json!(start + 1);
    let bytes = nf_core::canonical::CanonicalDocument::parse(
        &serde_json::to_vec(&value).unwrap(),
        nf_core::canonical::Limits::default(),
    )
    .unwrap()
    .bytes()
    .to_vec();
    direct
        .execute_batch("DROP TRIGGER updates_no_update")
        .unwrap();
    direct
        .execute(
            "UPDATE worker_updates SET body=?1,content=?2 WHERE worker=?3 AND version=1",
            rusqlite::params![
                bytes,
                ContentDigest::of_bytes(&bytes).as_str(),
                worker_id.as_str()
            ],
        )
        .unwrap();
    let report = ledger.recover_native(&lease, &worker_id).unwrap();
    assert!(!report.cleanup_confirmed);
    assert!(!report.outputs_eligible);
    assert!(std::path::Path::new(&format!("/proc/{pid}")).exists());
    assert_eq!(ledger.budget_snapshot(&id).unwrap().held.workers, 1);
    drop(running);
}

#[test]
fn cancellation_recovery_exit_before_cost_commit_never_publishes_partial_success() {
    let root = Root::new();
    let (mut ledger, id, lease) = setup(&root);
    let qualified = worker(&root);
    let reservation = reserve(&mut ledger, &lease, "cost-crash", 3000, 4000);
    let mut running = ledger
        .launch_native(&lease, &reservation, &qualified, job(FixtureMode::Complete))
        .unwrap();
    let direct = rusqlite::Connection::open(root.0.join("ledger/events.sqlite3")).unwrap();
    direct.execute_batch("CREATE TRIGGER reject_cost BEFORE INSERT ON budget_outcomes BEGIN SELECT RAISE(ABORT,'fixture interrupted cost commit');END;").unwrap();
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        match ledger.poll_native(&lease, &reservation, &mut running) {
            Err(_) => break,
            Ok(None) => {
                assert!(Instant::now() < deadline);
                std::thread::sleep(Duration::from_millis(5));
            }
            Ok(Some(_)) => panic!("unsettled job published success"),
        }
    }
    let state = ledger.worker_state(running.id()).unwrap();
    assert!(state.report.as_ref().unwrap().cleanup_confirmed);
    assert!(!state.report.unwrap().outputs_eligible);
    assert_eq!(ledger.budget_snapshot(&id).unwrap().held.workers, 1);
    direct.execute_batch("DROP TRIGGER reject_cost").unwrap();
    let recovered = ledger.recover_native(&lease, running.id()).unwrap();
    assert!(!recovered.outputs_eligible);
    assert_eq!(ledger.budget_snapshot(&id).unwrap().held.workers, 0);
}

#[test]
fn cancellation_recovery_unqualified_changed_duplicate_or_unsupported_native_dispatch_is_denied() {
    use std::os::unix::fs::PermissionsExt;
    let root = Root::new();
    let (mut ledger, _id, lease) = setup(&root);
    let qualified = worker(&root);
    let reservation = reserve(&mut ledger, &lease, "denied", 3000, 4000);
    assert!(TrustedNativeWorker::qualify(&root.0.join("fixture-worker"), d("wrong-code")).is_err());
    let unsupported = NativeJob {
        maximum_output_bytes: 100000,
        ..job(FixtureMode::Complete)
    };
    assert!(ledger
        .launch_native(&lease, &reservation, &qualified, unsupported)
        .is_err());
    let mut running = ledger
        .launch_native(&lease, &reservation, &qualified, job(FixtureMode::Complete))
        .unwrap();
    assert!(ledger
        .launch_native(&lease, &reservation, &qualified, job(FixtureMode::Complete))
        .is_err());
    finish(&mut ledger, &lease, &reservation, &mut running);
    let next = reserve(&mut ledger, &lease, "changed", 3000, 4000);
    let path = root.0.join("fixture-worker");
    fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
    fs::write(&path, b"not the approved worker").unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o500)).unwrap();
    assert!(ledger
        .launch_native(&lease, &next, &qualified, job(FixtureMode::Complete))
        .is_err());
}

#[test]
fn cancellation_recovery_real_cpu_memory_output_and_wall_limits_cannot_publish_outputs() {
    for (mode, wall, cpu) in [
        (FixtureMode::Busy, 3000, 2000),
        (FixtureMode::Memory, 3000, 4000),
        (FixtureMode::Output, 3000, 4000),
        (FixtureMode::IgnoreTerm, 1500, 4000),
    ] {
        let root = Root::new();
        let (mut ledger, id, lease) = setup(&root);
        let qualified = worker(&root);
        let reservation = reserve(&mut ledger, &lease, "limit", wall, cpu);
        let mut running = ledger
            .launch_native(&lease, &reservation, &qualified, job(mode))
            .unwrap();
        let report = finish(&mut ledger, &lease, &reservation, &mut running);
        assert_ne!(report.outcome, WorkerOutcome::Completed, "{mode:?}");
        assert!(report.cleanup_confirmed);
        assert!(!report.outputs_eligible);
        assert_eq!(ledger.budget_snapshot(&id).unwrap().held.workers, 0);
    }
}
