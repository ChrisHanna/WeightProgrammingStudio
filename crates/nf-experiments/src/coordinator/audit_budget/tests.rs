//! Enforcement tests use the production public opener, including v8 verification.
use super::tests::Harness;
use super::*;
use crate::{
    intake::{self, Draft},
    sandbox::QualifiedDocker,
    validation::{
        reference::{Coverage, Exposure, Policy, Role},
        Plan,
    },
};
use nf_checkpoint::store::{ArtifactStore, StoreLimits};
use serde_json::json;
use std::{
    fs,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

/// Re-exec only this named test under a real irreversible kernel ceiling.
/// The 85-second timeout bounds even a broken OCI/test barrier.
pub(super) fn bounded_controller_test() -> bool {
    let name = std::thread::current().name().unwrap().to_owned();
    if std::env::var("NF_AUDIT_BOUNDED_TEST").ok().as_deref() == Some(&name) {
        return false;
    }
    let hard = crate::sandbox::CONTROLLER_MEMORY_ALLOWANCE_BYTES;
    let status = std::process::Command::new("/usr/bin/timeout")
        .args(["--signal=KILL", "85s", "/usr/bin/prlimit"])
        .arg(format!("--as={hard}:{hard}"))
        .arg(std::env::current_exe().unwrap())
        .args(["--exact", &name, "--nocapture", "--test-threads=1"])
        .env("NF_AUDIT_BOUNDED_TEST", &name)
        .env("MALLOC_ARENA_MAX", "1")
        .status()
        .unwrap();
    assert!(
        status.success(),
        "bounded named audit test failed: {status}"
    );
    true
}

pub(super) fn open_component(root: &Path) -> Ledger {
    Ledger::open(root, LedgerLimits::default()).unwrap()
}
fn inputs(h: &Harness, source_text: &str) -> (intake::Snapshot, ArtifactStore, Plan) {
    let store = ArtifactStore::create(
        &h.root.join("cas"),
        StoreLimits {
            total_bytes: 16 * 1024 * 1024,
            max_blob_bytes: 1048576,
            max_entries: 100,
        },
    )
    .unwrap();
    let source = h.root.join("source");
    fs::create_dir(&source).unwrap();
    fs::write(source.join("gym.py"), source_text).unwrap();
    fs::write(source.join("LICENSE"), "fixture").unwrap();
    fs::write(source.join("data.json"), "[]").unwrap();
    fs::write(
        source.join("lock.json"),
        json!({"schema_version":1,
        "image":crate::sandbox::IMAGE.split('@').nth(1).unwrap(),"packages":{}})
        .to_string(),
    )
    .unwrap();
    let ep = |symbol| json!({"file":"gym.py","symbol":symbol});
    let integer = json!({"type":"integer","min":0,"max":1000});
    let draft = json!({"schema_version":1,"name":"fixture","version":"1","usage_rights":"fixture",
        "confidentiality":"confidential","assets":{
            "gym.py":{"kind":"code","expected":null},"LICENSE":{"kind":"license","expected":null},
            "lock.json":{"kind":"dependency_lock","expected":null},"data.json":{"kind":"data","expected":null}},
        "dependency_lock":"lock.json","licenses":["LICENSE"],
        "interface":{"kind":"single_shot","evaluate":ep("evaluate"),"score":ep("score"),"oracle":ep("oracle"),"input":integer,"output":integer},
        "dataset":{"kind":"frozen","files":["data.json"],"task_key":"task","group_key":"group"}});
    let snapshot = intake::snapshot(
        &source,
        &store,
        Draft::parse(draft.to_string().as_bytes()).unwrap(),
    )
    .unwrap();
    let plan = Plan::parse(json!({"schema_version":1,"snapshot":snapshot.id(),"truth_provenance":"independent arithmetic",
        "cases":[{"id":"two","input":2,"expected":4,"wrong":[3],"termination":null},
                 {"id":"seven","input":7,"expected":14,"wrong":[13],"termination":null}],"seeds":[11,22],
        "scoring":{"minimum":0.0,"maximum":1.0,"good_min":1.0,"bad_max":0.0},
        "limits":{"wall_ms":2000,"cpu_ms":2000,"output_bytes":32768}}).to_string().as_bytes()).unwrap();
    (snapshot, store, plan)
}
const GOOD: &str = "def evaluate(x,seed): return x*2\ndef oracle(x): return x*2\ndef score(x,out,reference): return float(out==reference)\n";
fn jobs(root: &Path) -> usize {
    fs::read_dir(root)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with("nf-sandbox-"))
        .count()
}
fn replace(h: &mut Harness, host: u64) {
    h.ledger
        .settle(
            &h.lease,
            &h.reservation,
            Settlement {
                operation: fixture::d("initial-unused"),
                outcome: OutcomeKind::Incomplete,
                resources_released: true,
                usage: Usage::default(),
                evidence: fixture::d("no-dispatch"),
            },
        )
        .unwrap();
    h.request.attempt = 1;
    h.request.resources.host_bytes = host;
    h.reservation = h.ledger.reserve(&h.lease, h.request.clone()).unwrap();
}

#[test]
fn memory_and_remaining_lease_deny_before_qualifier() {
    for memory in [true, false] {
        let mut h = Harness::new();
        if memory {
            replace(&mut h, AUDIT_HOST_BYTES - 1);
        } else {
            h.lease = h.ledger.renew(&h.lease, 1000).unwrap();
        }
        let (snapshot, store, plan) = inputs(&h, GOOD);
        let called = AtomicBool::new(false);
        assert!(h
            .ledger
            .run_reference_audit(
                &h.lease,
                &h.reservation,
                &snapshot,
                &store,
                plan,
                &[Policy {
                    snapshot: &snapshot,
                    role: Role::Valid
                }],
                Exposure::Development,
                h.budget.clone(),
                || {
                    called.store(true, Ordering::Release);
                    panic!("qualifier must not run");
                },
                Coverage::default(),
                &AtomicBool::new(false)
            )
            .is_err());
        assert!(!called.load(Ordering::Acquire));
        assert!(!claimed(&h.ledger.connection, h.reservation.id()).unwrap());
    }
}

#[test]
fn unlimited_controller_public_entry_denies_before_claim_and_qualifier() {
    assert!(
        rustix::process::getrlimit(rustix::process::Resource::As)
            .maximum
            .is_none(),
        "run this negative case in an unlimited parent, not the bounded child harness"
    );
    let mut h = Harness::new();
    let (snapshot, store, plan) = inputs(&h, GOOD);
    let called = AtomicBool::new(false);
    let error = h
        .ledger
        .run_reference_audit(
            &h.lease,
            &h.reservation,
            &snapshot,
            &store,
            plan,
            &[Policy {
                snapshot: &snapshot,
                role: Role::Valid,
            }],
            Exposure::Development,
            h.budget.clone(),
            || {
                called.store(true, Ordering::Release);
                panic!("unlimited controller must not qualify");
            },
            Coverage::default(),
            &AtomicBool::new(false),
        )
        .err()
        .unwrap();
    assert!(error.message.contains("finite hard RLIMIT_AS"));
    assert!(!called.load(Ordering::Acquire));
    assert!(!claimed(&h.ledger.connection, h.reservation.id()).unwrap());
}

#[test]
fn shared_dispatch_claims_are_atomic_in_both_orders_and_across_connections() {
    for audit_first in [true, false] {
        let mut h = Harness::new();
        if audit_first {
            h.claim();
        } else {
            let tx = h
                .ledger
                .connection
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .unwrap();
            claim_execution(&tx, h.reservation.id(), "native").unwrap();
            tx.commit().unwrap();
        }
        let mut other = open_component(&h.root.join("ledger"));
        if audit_first {
            let tx = other
                .connection
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .unwrap();
            assert!(claim_execution(&tx, h.reservation.id(), "native").is_err());
        } else {
            assert!(other
                .claim_audit(&h.lease, &h.reservation, &h.budget)
                .is_err());
        }
        let count: u64 = h
            .ledger
            .connection
            .query_row(
                "SELECT count(*) FROM execution_owners WHERE request=?1",
                [h.reservation.id().as_str()],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }
}

#[test]
fn schema_tampering_and_ignored_insert_cannot_create_authority() {
    for mutation in [
        "DROP TRIGGER audit_claim_no_update; CREATE TRIGGER audit_claim_no_update BEFORE INSERT ON audit_storage_claims BEGIN SELECT RAISE(IGNORE); END;",
        "CREATE TRIGGER shadow_insert BEFORE INSERT ON execution_owners BEGIN SELECT RAISE(IGNORE); END;",
        "DROP TABLE audit_storage_claims;",
    ] {
        let mut h = Harness::new();
        h.ledger.connection.execute_batch(mutation).unwrap();
        assert!(h.ledger.claim_audit(&h.lease, &h.reservation, &h.budget).is_err());
        assert!(h.ledger.settle(&h.lease, &h.reservation, Settlement {
            operation: fixture::d("forged"), outcome: OutcomeKind::Incomplete, resources_released: true,
            usage: Usage::default(), evidence: fixture::d("forged"),
        }).is_err());
    }
}

#[test]
fn caller_cpu_or_cleanup_claims_cannot_refund_audit() {
    let mut h = Harness::new();
    h.claim();
    for cpu in [None, Some(0), Some(1)] {
        let value = Settlement {
            operation: fixture::d("forged"),
            outcome: OutcomeKind::Incomplete,
            resources_released: true,
            usage: Usage {
                cpu_ms: cpu,
                ..Usage::default()
            },
            evidence: fixture::d("forged"),
        };
        assert!(h.ledger.settle(&h.lease, &h.reservation, value).is_err());
    }
    let snapshot = h.ledger.budget_snapshot(&h.lease.contract).unwrap();
    assert_eq!(snapshot.held.host_bytes, AUDIT_HOST_BYTES);
    assert_eq!(snapshot.held.storage_bytes, h.budget.storage_bytes);
    assert_eq!(snapshot.held.workers, 1);
}

#[test]
fn qualification_delay_consumes_absolute_audit_allowance() {
    if bounded_controller_test() {
        return;
    }
    let mut h = Harness::new();
    let (snapshot, store, plan) = inputs(&h, GOOD);
    let runtime = QualifiedDocker::qualify(true, &h.root).unwrap();
    let before = jobs(&h.root);
    h.budget.wall_ms = 4000;
    assert!(h
        .ledger
        .run_reference_audit(
            &h.lease,
            &h.reservation,
            &snapshot,
            &store,
            plan,
            &[Policy {
                snapshot: &snapshot,
                role: Role::Valid
            }],
            Exposure::Development,
            h.budget.clone(),
            || {
                std::thread::sleep(Duration::from_millis(3000));
                Ok(runtime)
            },
            Coverage::default(),
            &AtomicBool::new(false)
        )
        .is_err());
    assert_eq!(
        jobs(&h.root),
        before,
        "no staging with insufficient remaining absolute time"
    );
    assert_eq!(
        h.ledger
            .budget_snapshot(&h.lease.contract)
            .unwrap()
            .held
            .workers,
        1
    );
}

#[test]
fn durable_stop_and_lease_revocation_interrupt_real_audit_and_recover() {
    if bounded_controller_test() {
        return;
    }
    for revoke_lease in [false, true] {
        let mut h = Harness::new();
        let (snapshot, store, plan) = inputs(&h,
            "import time\ndef evaluate(x,seed): time.sleep(30); return x*2\ndef oracle(x): return x*2\ndef score(x,out,reference): return float(out==reference)\n");
        let runtime = QualifiedDocker::qualify(true, &h.root).unwrap();
        let before = jobs(&h.root);
        let done = AtomicBool::new(false);
        std::thread::scope(|scope| {
            let root = &h.root;
            let contract = h.lease.contract.clone();
            let done = &done;
            let writer = scope.spawn(move || {
                let mut other = open_component(&root.join("ledger"));
                let until = Instant::now() + Duration::from_secs(10);
                while Instant::now() < until && !done.load(Ordering::Acquire) {
                    if jobs(root) > before {
                        for entry in fs::read_dir(root).unwrap().filter_map(|e| e.ok()) {
                            if !entry.path().join("configured.json").exists() { continue; }
                            let name = entry.file_name();
                            if !QualifiedDocker::container_running(name.to_str().unwrap()).unwrap_or(false) { continue; }
                            if revoke_lease {
                                other.connection.execute("UPDATE runs SET epoch=epoch+1,deadline=0 WHERE contract=?1",
                                    [contract.as_str()]).unwrap();
                            } else {
                                other.request_stop(&contract, fixture::d("active-stop"),
                                    super::super::super::supervision::StopReason::OwnerStop).unwrap();
                            }
                            return true;
                        }
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
                false
            });
            let started = Instant::now();
            let result = h.ledger.run_reference_audit(
                &h.lease,
                &h.reservation,
                &snapshot,
                &store,
                plan,
                &[Policy {
                    snapshot: &snapshot,
                    role: Role::Valid,
                }],
                Exposure::Development,
                h.budget.clone(),
                || Ok(runtime),
                Coverage::default(),
                &AtomicBool::new(false),
            );
            done.store(true, Ordering::Release);
            assert!(
                writer.join().unwrap(),
                "revocation must occur during an actual running callback; result={:?}",
                result.err()
            );
            assert!(result.is_err());
            assert!(started.elapsed() < Duration::from_secs(5));
        });
        let held = h.ledger.budget_snapshot(&h.lease.contract).unwrap();
        assert_eq!(
            held.held.workers, 1,
            "error never releases without cleanup attestation"
        );
        if revoke_lease {
            h.lease = h
                .ledger
                .acquire(&h.lease.contract, "recovery", 300000)
                .unwrap();
        }
        let runtime = QualifiedDocker::qualify(true, &h.root).unwrap();
        h.ledger
            .recover_reference_audit(&h.lease, &h.reservation, &runtime)
            .unwrap();
        let after = h.ledger.budget_snapshot(&h.lease.contract).unwrap();
        assert_eq!(after.held.workers, 0);
        assert_eq!(after.spent.storage_bytes, h.budget.storage_bytes);
        assert_eq!(after.spent.cpu_ms, h.request.resources.cpu_ms);
        h.ledger
            .recover_reference_audit(&h.lease, &h.reservation, &runtime)
            .unwrap();
        assert_eq!(
            h.ledger
                .budget_snapshot(&h.lease.contract)
                .unwrap()
                .spent
                .storage_bytes,
            after.spent.storage_bytes
        );
    }
}

#[test]
fn complete_native_audit_executes_six_controls_and_durably_charges_family() {
    if bounded_controller_test() {
        return;
    }
    let mut h = Harness::new();
    let (gym, store, plan) = inputs(&h, GOOD);
    let mut snapshots = Vec::new();
    for (name, replacement) in [
        ("valid", "x*2"),
        ("invalid", "-1"),
        ("constant", "0"),
        ("metadata", "seed"),
        ("weak", "4 if x == 2 else 0"),
        ("strong", "x*2"),
    ] {
        let source = h.root.join(name);
        fs::create_dir(&source).unwrap();
        for file in ["LICENSE", "data.json", "lock.json"] {
            fs::copy(h.root.join("source").join(file), source.join(file)).unwrap();
        }
        fs::write(
            source.join("gym.py"),
            GOOD.replacen("return x*2", &format!("return {replacement}"), 1),
        )
        .unwrap();
        let mut draft = serde_json::to_value(&gym.manifest().draft).unwrap();
        draft["name"] = json!(name);
        snapshots.push(
            intake::snapshot(
                &source,
                &store,
                Draft::parse(draft.to_string().as_bytes()).unwrap(),
            )
            .unwrap(),
        );
    }
    h.ledger
        .settle(
            &h.lease,
            &h.reservation,
            Settlement {
                operation: fixture::d("unused"),
                outcome: OutcomeKind::Incomplete,
                resources_released: true,
                usage: Usage::default(),
                evidence: fixture::d("never-dispatched"),
            },
        )
        .unwrap();
    h.request.attempt = 1;
    h.request.resources.cpu_ms = 30000;
    h.request.resources.storage_bytes = 384 * 1024 * 1024;
    h.request.wall_ms = 50000;
    h.reservation = h.ledger.reserve(&h.lease, h.request.clone()).unwrap();
    let budget = Budget {
        max_executions: 12,
        cpu_ms: 24000,
        wall_ms: 45000,
        deadline_policy: None,
        storage_bytes: h.request.resources.storage_bytes,
    };
    let policies: Vec<_> = snapshots
        .iter()
        .zip([
            Role::Valid,
            Role::Invalid,
            Role::Constant,
            Role::MetadataOnly,
            Role::Weak,
            Role::Strong,
        ])
        .map(|(snapshot, role)| Policy { snapshot, role })
        .collect();
    let coverage = Coverage {
        required: ["arithmetic".to_owned()].into(),
        case_groups: [
            ("two".to_owned(), "arithmetic".to_owned()),
            ("seven".to_owned(), "arithmetic".to_owned()),
        ]
        .into(),
    };
    let audit = h
        .ledger
        .run_reference_audit(
            &h.lease,
            &h.reservation,
            &gym,
            &store,
            plan,
            &policies,
            Exposure::Development,
            budget,
            || QualifiedDocker::qualify(true, &h.root),
            coverage,
            &AtomicBool::new(false),
        )
        .unwrap();
    assert!(audit.report().audit_complete, "{:?}", audit.report());
    assert_eq!(audit.report().status, crate::validation::Status::Passed);
    assert_eq!(audit.report().executions, 12);
    assert!(audit.report().storage.durable_family_accounting);
    assert_eq!(
        audit.report().controller_hard_address_space_bytes,
        Some(crate::sandbox::CONTROLLER_MEMORY_ALLOWANCE_BYTES)
    );
    assert!(audit
        .report()
        .measurement_scope
        .contains("not_whole_process_tree"));
    assert!(!audit.report().native_approval);
    h.reopen();
    let snapshot = h.ledger.budget_snapshot(&h.lease.contract).unwrap();
    assert_eq!(snapshot.held.workers, 0);
    assert_eq!(snapshot.spent.storage_bytes, 448 * 1024 * 1024);
    assert_eq!(snapshot.spent.cpu_ms, 35000);
}

#[test]
fn retry_cap_and_family_storage_cannot_be_reset_by_new_operation_ids() {
    let mut h = Harness::new();
    let runtime = QualifiedDocker::qualify(true, &h.root).unwrap();
    for index in 0..8 {
        if index > 0 {
            h.request.operation = fixture::d(if index < 3 { "audit" } else { "next-operation" });
            // First exhaust one operation's sequential retries. Then use fresh
            // operation IDs; neither route creates a new family allowance.
            if index < 3 {
                h.request.attempt = index;
            } else {
                h.request.operation = ContentDigest::of_bytes(&index.to_le_bytes());
                h.request.attempt = 0;
            }
            h.reservation = h.ledger.reserve(&h.lease, h.request.clone()).unwrap();
        }
        h.claim();
        h.ledger
            .bind_audit_runtime(&h.lease, &h.reservation, &runtime)
            .unwrap();
        let evidence = ContentDigest::of_bytes(&index.to_le_bytes());
        h.ledger
            .confirm_audit_cleanup(&h.lease, &h.reservation, &runtime, &evidence)
            .unwrap();
        h.ledger
            .settle(
                &h.lease,
                &h.reservation,
                Settlement {
                    operation: ContentDigest::of_bytes(&canonical(&(index, "cost")).unwrap()),
                    outcome: OutcomeKind::Incomplete,
                    resources_released: true,
                    usage: Usage::default(),
                    evidence,
                },
            )
            .unwrap();
        if index == 2 {
            let mut denied = h.request.clone();
            denied.attempt = 3;
            assert!(h.ledger.reserve(&h.lease, denied).is_err());
        }
    }
    let snapshot = h.ledger.budget_snapshot(&h.lease.contract).unwrap();
    assert_eq!(snapshot.spent.storage_bytes, 512 * 1024 * 1024);
    h.request.operation = fixture::d("fresh-operation-cannot-reset-family");
    h.request.attempt = 0;
    assert!(h.ledger.reserve(&h.lease, h.request.clone()).is_err());
    h.reopen();
    assert_eq!(
        h.ledger
            .budget_snapshot(&h.lease.contract)
            .unwrap()
            .spent
            .storage_bytes,
        snapshot.spent.storage_bytes
    );
}
