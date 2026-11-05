//! Public-opener integration tests: require coordinator.rs's explicit v8 migration arm.
use super::*;
#[path = "../../../nf-core/tests/common/study_fixture.rs"]
#[allow(dead_code)]
mod contract_fixture;
use nf_experiments::coordinator::budgets::{
    audit_budget::AUDIT_HOST_BYTES, Request, Reservation, Resources, WorkKind,
};
use nf_experiments::coordinator::supervision::{
    FixtureMode, NativeJob, StopReason, TrustedNativeWorker,
};
use nf_experiments::coordinator::{Lease, Ledger, LedgerLimits};
use std::{
    sync::atomic::AtomicBool,
    time::{Duration, Instant},
};

fn setup(root: &Root) -> (Ledger, Lease, Reservation) {
    let mut value = contract_fixture::standard();
    value["limits"]["host_memory_bytes"] = json!(512 * 1024 * 1024);
    value["limits"]["storage_bytes"] = json!(512 * 1024 * 1024);
    let contract = nf_core::schemas::Validated::<nf_core::study::StudyContract>::parse(
        &serde_json::to_vec(&value).unwrap(),
    )
    .unwrap();
    let mut ledger = Ledger::open(&root.0.join("ledger"), LedgerLimits::default()).unwrap();
    let id = ledger.freeze(&contract).unwrap();
    let lease = ledger.acquire(&id, "dispatch-test", 300000).unwrap();
    let reservation = ledger
        .reserve(
            &lease,
            Request {
                schema_version: 1,
                operation: contract_fixture::d("dispatch"),
                attempt: 0,
                parent: None,
                kind: WorkKind::Inspection,
                resources: Resources {
                    host_bytes: AUDIT_HOST_BYTES,
                    cpu_ms: 5000,
                    storage_bytes: 64 * 1024 * 1024,
                    workers: 1,
                    ..Resources::default()
                },
                wall_ms: 20000,
                external: None,
            },
        )
        .unwrap();
    (ledger, lease, reservation)
}
fn worker(root: &Root) -> TrustedNativeWorker {
    let path = root.0.join("native-worker");
    assert!(std::process::Command::new("/usr/bin/strip")
        .arg("--strip-debug")
        .arg("-o")
        .arg(&path)
        .arg(env!("CARGO_BIN_EXE_nf-fixture-worker"))
        .status()
        .unwrap()
        .success());
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o500)).unwrap();
    let digest = ContentDigest::of_bytes(&std::fs::read(&path).unwrap());
    TrustedNativeWorker::qualify(&path, digest).unwrap()
}
fn job() -> NativeJob {
    NativeJob {
        mode: FixtureMode::Sleep,
        maximum_output_bytes: 1024,
        termination_grace_ms: 30,
    }
}
fn allowance() -> Budget {
    Budget {
        max_executions: 2,
        cpu_ms: 4000,
        wall_ms: 15000,
        storage_bytes: 64 * 1024 * 1024,
    }
}

#[test]
fn running_native_worker_excludes_audit_before_qualification() {
    let root = Root::new();
    let store = store(&root);
    let gym = snapshot(&root, "gym", &store, GOOD);
    let (mut ledger, lease, reservation) = setup(&root);
    let qualified = worker(&root);
    let mut running = ledger
        .launch_native(&lease, &reservation, &qualified, job())
        .unwrap();
    let called = AtomicBool::new(false);
    let error = ledger
        .run_reference_audit(
            &lease,
            &reservation,
            &gym,
            &store,
            plan(&gym),
            &[Policy {
                snapshot: &gym,
                role: Role::Valid,
            }],
            Exposure::Development,
            allowance(),
            || {
                called.store(true, Ordering::Release);
                panic!("native-owned reservation must not qualify audit");
            },
            reference::Coverage::default(),
            &AtomicBool::new(false),
        )
        .err()
        .unwrap();
    assert!(
        error.message.contains("execution owner is not audit"),
        "{error:?}"
    );
    assert!(!called.load(Ordering::Acquire));
    ledger
        .request_stop(
            lease.contract(),
            contract_fixture::d("stop"),
            StopReason::OwnerStop,
        )
        .unwrap();
    let until = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(report) = ledger
            .poll_native(&lease, &reservation, &mut running)
            .unwrap()
        {
            assert!(report.cleanup_confirmed);
            assert!(!report.outputs_eligible);
            break;
        }
        assert!(Instant::now() < until);
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn durable_audit_excludes_native_across_connections_and_still_executes() {
    let name = std::thread::current().name().unwrap().to_owned();
    if std::env::var("NF_DURABLE_AUDIT_BOUNDED").ok().as_deref() != Some(&name) {
        let hard = nf_experiments::sandbox::CONTROLLER_MEMORY_ALLOWANCE_BYTES;
        let status = std::process::Command::new("/usr/bin/timeout")
            .args(["--signal=KILL", "85s", "/usr/bin/prlimit"])
            .arg(format!("--as={hard}:{hard}"))
            .arg(std::env::current_exe().unwrap())
            .args(["--exact", &name, "--nocapture", "--test-threads=1"])
            .env("NF_DURABLE_AUDIT_BOUNDED", &name)
            .env("MALLOC_ARENA_MAX", "1")
            .status()
            .unwrap();
        assert!(
            status.success(),
            "bounded integration test failed: {status}"
        );
        return;
    }
    let root = Root::new();
    let store = store(&root);
    let gym = snapshot(&root, "gym", &store, GOOD);
    let (mut ledger, lease, reservation) = setup(&root);
    let qualified = worker(&root);
    let backend = runtime(&root);
    let audit = ledger
        .run_reference_audit(
            &lease,
            &reservation,
            &gym,
            &store,
            plan(&gym),
            &[Policy {
                snapshot: &gym,
                role: Role::Valid,
            }],
            Exposure::Development,
            allowance(),
            || {
                let mut other =
                    Ledger::open(&root.0.join("ledger"), LedgerLimits::default()).unwrap();
                assert!(other
                    .launch_native(&lease, &reservation, &qualified, job())
                    .is_err());
                Ok(backend)
            },
            reference::Coverage::default(),
            &AtomicBool::new(false),
        )
        .unwrap();
    assert_eq!(audit.report().executions, 2);
    assert_eq!(audit.report().policies[0].correct, 8);
    assert!(audit.report().storage.durable_family_accounting);
    let reopened = Ledger::open(&root.0.join("ledger"), LedgerLimits::default()).unwrap();
    assert_eq!(
        reopened
            .budget_snapshot(lease.contract())
            .unwrap()
            .held
            .workers,
        0
    );
}
