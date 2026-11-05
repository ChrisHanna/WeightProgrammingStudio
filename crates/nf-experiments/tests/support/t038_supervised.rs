//! End-to-end supervisor evidence, cancellation, and real process-loss tests.
use super::*;
#[cfg(debug_assertions)]
use nf_experiments::coordinator::final_fixture::inject_fixture_fault;

fn record_kind(s: &Setup, prefix: &str) -> String {
    let db = rusqlite::Connection::open(s.root.join("ledger/events.sqlite3")).unwrap();
    db.query_row(
        "SELECT kind FROM final_fixture_records WHERE kind LIKE ?1",
        [format!("{prefix}%")],
        |r| r.get(0),
    )
    .unwrap()
}

#[test]
fn actual_supervisor_proof_accounts_for_guardian_and_worker() {
    if isolated() {
        return;
    }
    let mut s = Setup::new(false, false);
    assert!(s.run().report().accepted);
    let intent = native_record(&s, "intent");
    let launch = native_record(&s, &record_kind(&s, "worker_intent_"));
    let prepared = native_record(&s, &record_kind(&s, "worker_prepared_"));
    let received = native_record(&s, &record_kind(&s, "worker_receipt_"));
    let r = &received["receipt"];
    assert_eq!(launch["policy"], prepared["policy"]);
    assert_eq!(r["identity"], prepared);
    assert_eq!(r["status"], "completed");
    assert_eq!(r["cleanup_confirmed"], true);
    assert_eq!(r["exit_code"], 0);
    assert_eq!(
        r["identity"]["policy"]["binding"]["reservation"],
        intent["reservation"]
    );
    assert_eq!(
        r["identity"]["policy"]["binding"]["fencing_epoch"],
        intent["epoch"]
    );
    assert!(
        r["events"]["worker_started_ns"].as_u64().unwrap()
            < r["events"]["cleanup_confirmed_ns"].as_u64().unwrap()
    );
    assert!(
        r["events"]["cleanup_confirmed_ns"].as_u64().unwrap()
            < r["deadline_boottime_ns"].as_u64().unwrap()
    );
    assert!(
        received["controller"]["process_cpu_delta_ns"]
            .as_u64()
            .unwrap()
            > 0
    );
    let external_cpu = r["guardian_cpu_used_ns"].as_u64().unwrap()
        + r["all_children_cpu_used_ns"].as_u64().unwrap();
    assert!(external_cpu > 0);
    assert!(
        native_record(&s, "execution")["cpu_ms"].as_u64().unwrap()
            >= external_cpu.div_ceil(1_000_000)
    );
    assert_eq!(launch["cells"].as_array().unwrap().len(), 5);
    assert_eq!(r["stdout_hex"].as_str().unwrap().len(), 5 * 432 * 2);
}

#[test]
fn native_qualification_is_bound_to_evidence_artifacts_and_explicit_timing_scope() {
    if isolated() {
        return;
    }
    use nf_experiments::coordinator::final_fixture::FixtureTimingScope;
    let mut s = Setup::new(false, false);
    assert!(s
        .ledger
        .qualify_fixture_final(s.lease.contract(), &s.store)
        .is_err());
    let decision = s.run();
    assert!(decision.report().accepted);
    let qualification = s
        .ledger
        .qualify_fixture_final(s.lease.contract(), &s.store)
        .unwrap();
    assert_eq!(qualification.contract(), s.lease.contract());
    assert_eq!(qualification.decision(), decision.id());
    assert_eq!(
        qualification.timing_scope(),
        &FixtureTimingScope::OperationalCancellation
    );
    let value = serde_json::to_value(&qualification).unwrap();
    assert_eq!(value["supervisor_receipts"].as_array().unwrap().len(), 1);
    assert_eq!(
        value["physical_publication"],
        "unsupported_unbounded_SQLite_fsync_and_scheduler_tail"
    );
    assert!(!decision.report().promotion_authorized);
    assert!(s.ledger.require_strict_fixture_publication().is_err());
    let reopened = Ledger::open(&s.root.join("ledger"), LedgerLimits::default()).unwrap();
    assert_eq!(
        reopened
            .qualify_fixture_final(s.lease.contract(), &s.store)
            .unwrap()
            .digest()
            .unwrap(),
        qualification.digest().unwrap()
    );
    // Immutable receipt links alone are insufficient: resolve actual CAS bytes.
    let graph = ContentDigest::of_bytes(graph_bytes());
    let path = s
        .root
        .join("store/blobs")
        .join(graph.as_str().trim_start_matches("sha256:"));
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    std::fs::write(path, b"tampered").unwrap();
    assert!(reopened
        .qualify_fixture_final(s.lease.contract(), &s.store)
        .is_err());
}

#[test]
fn retention_failure_never_yields_native_qualification() {
    if isolated() {
        return;
    }
    let mut s = Setup::new(true, false);
    assert!(!s.run().report().accepted);
    assert!(s
        .ledger
        .qualify_fixture_final(s.lease.contract(), &s.store)
        .is_err());
}

#[cfg(debug_assertions)]
#[test]
fn persisted_supervisor_registration_precedes_worker_authorization() {
    if isolated() {
        return;
    }
    let mut s = Setup::new(false, false);
    let root = s.root.clone();
    let contract = s.lease.contract().clone();
    inject_fixture_fault("supervisor_after_prepare", move || {
        let db = rusqlite::Connection::open(root.join("ledger/events.sqlite3")).unwrap();
        let launch: Vec<u8> = db
            .query_row(
                "SELECT body FROM final_fixture_records WHERE kind LIKE 'worker_intent_%'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        let prepared: Vec<u8> = db
            .query_row(
                "SELECT body FROM final_fixture_records WHERE kind LIKE 'worker_prepared_%'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        let launch: Value = serde_json::from_slice(&launch).unwrap();
        let prepared: Value = serde_json::from_slice(&prepared).unwrap();
        assert_eq!(launch["policy"], prepared["policy"]);
        assert!(
            !std::path::Path::new(prepared["policy"]["argv"][0].as_str().unwrap()).exists(),
            "named image scratch must be removed after sealing, before authorization"
        );
        assert_eq!(prepared["policy"]["binding"]["contract"], json!(contract));
        assert!(prepared["guardian_start_ticks"].as_u64().unwrap() > 0);
        assert_eq!(
            db.query_row(
                "SELECT count(*) FROM final_fixture_records WHERE kind LIKE 'worker_receipt_%'",
                [],
                |r| r.get::<_, u64>(0)
            )
            .unwrap(),
            0
        );
        let other = Ledger::open(&root.join("ledger"), LedgerLimits::default()).unwrap();
        assert_eq!(other.budget_snapshot(&contract).unwrap().held.workers, 1);
        Ok(())
    });
    assert!(s.run().report().accepted);
}

#[cfg(debug_assertions)]
#[test]
fn durable_stop_and_cancel_after_authorization_never_publish() {
    if isolated() {
        return;
    }
    for durable in [false, true] {
        let mut s = Setup::new(false, false);
        let mut other = Ledger::open(&s.root.join("ledger"), LedgerLimits::default()).unwrap();
        let contract = s.lease.contract().clone();
        let cancel = CancellationToken::default();
        let stop = cancel.clone();
        inject_fixture_fault("supervisor_after_authorize", move || {
            let snapshot = other.budget_snapshot(&contract).unwrap();
            assert_eq!(snapshot.held.workers, 1);
            let limit = rustix::process::getrlimit(rustix::process::Resource::As);
            assert!(limit.current.unwrap() <= 1_000_000_000 - (128 << 20));
            if durable {
                other
                    .request_stop(
                        &contract,
                        d("stop-supervised-final"),
                        nf_experiments::coordinator::supervision::StopReason::OwnerStop,
                    )
                    .unwrap();
            } else {
                stop.request();
            }
            Ok(())
        });
        let error = s
            .ledger
            .run_fixture_final(&s.lease, &s.store, &s.edit, &s.noop, &s.smoke, &cancel)
            .err()
            .expect("stop must prevent publication");
        eprintln!("post-authorization durable={durable}: {error:?}");
        let received = native_record(&s, &record_kind(&s, "worker_receipt_"));
        assert_eq!(received["receipt"]["cleanup_confirmed"], true);
        assert!(record_kind(&s, "worker_stop_").starts_with("worker_stop_"));
        let reopened = Ledger::open(&s.root.join("ledger"), LedgerLimits::default()).unwrap();
        assert!(reopened
            .fixture_final_decision(s.lease.contract())
            .unwrap()
            .is_none());
        let budget = reopened.budget_snapshot(s.lease.contract()).unwrap();
        assert_eq!(budget.held.workers, 0);
        assert_eq!(budget.spent.cpu_ms, 30_000);
        assert_eq!(budget.spent.evaluations, 86);
    }
}

#[test]
fn forged_or_missing_supervisor_receipts_never_authorize_on_reopen() {
    if isolated() {
        return;
    }
    for variant in 0..4 {
        let mut s = Setup::new(false, false);
        assert!(s.run().report().accepted);
        let kind = record_kind(&s, "worker_receipt_");
        let mut value = native_record(&s, &kind);
        let db = rusqlite::Connection::open(s.root.join("ledger/events.sqlite3")).unwrap();
        if variant == 0 {
            db.execute_batch("DROP TRIGGER final_fixture_no_delete")
                .unwrap();
            db.execute("DELETE FROM final_fixture_records WHERE kind=?1", [&kind])
                .unwrap();
        } else {
            match variant {
                1 => value["receipt"]["stdout_hex"] = json!("00".repeat(5 * 432)),
                2 => {
                    value["receipt"]["identity"]["policy"]["binding"]["fencing_epoch"] = json!(999)
                }
                _ => value = json!({"receipt":{"status":"completed","cleanup_confirmed":true}}),
            }
            let body = bytes(&value);
            db.execute_batch("DROP TRIGGER final_fixture_no_update")
                .unwrap();
            db.execute(
                "UPDATE final_fixture_records SET body=?1,digest=?2 WHERE kind=?3",
                rusqlite::params![body, ContentDigest::of_bytes(&body).as_str(), kind],
            )
            .unwrap();
        }
        let reopened = Ledger::open(&s.root.join("ledger"), LedgerLimits::default()).unwrap();
        assert!(reopened.fixture_final_decision(s.lease.contract()).is_err());
    }
}

#[cfg(debug_assertions)]
#[test]
fn real_crash_at_supervisor_boundaries_never_invents_cleanup_or_receipts() {
    if isolated() {
        return;
    }
    if let Ok(point) = std::env::var("T038_SUPERVISOR_CRASH") {
        let point: &'static str = Box::leak(point.into_boxed_str());
        let mut s = Setup::new(false, false);
        std::fs::write(
            std::env::var("T038_SUPERVISOR_CONTROL").unwrap(),
            bytes(&json!({"root":s.root,"contract":s.lease.contract()})),
        )
        .unwrap();
        inject_fixture_fault(point, || std::process::exit(86));
        s.run();
        panic!("crash hook not reached");
    }
    for (point, known) in [
        ("supervisor_before_prepare", false),
        ("supervisor_after_prepare", false),
        ("supervisor_after_authorize", false),
        ("supervisor_after_receipt", true),
    ] {
        let control = std::env::temp_dir().join(format!(
            "t038-supervisor-{}-{point}.json",
            std::process::id()
        ));
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                std::thread::current().name().unwrap(),
                "--test-threads=1",
            ])
            .env("T038_SUPERVISOR_CRASH", point)
            .env("T038_SUPERVISOR_CONTROL", &control)
            .status()
            .unwrap();
        assert_eq!(status.code(), Some(86), "{point}");
        let info: Value = serde_json::from_slice(&std::fs::read(&control).unwrap()).unwrap();
        let root = PathBuf::from(info["root"].as_str().unwrap());
        let contract: ContentDigest = serde_json::from_value(info["contract"].clone()).unwrap();
        let db = rusqlite::Connection::open(root.join("ledger/events.sqlite3")).unwrap();
        db.execute(
            "UPDATE runs SET deadline=0 WHERE contract=?1",
            [contract.as_str()],
        )
        .unwrap();
        let receipt_count: u64 = db
            .query_row(
                "SELECT count(*) FROM final_fixture_records WHERE kind LIKE 'worker_receipt_%'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(receipt_count, u64::from(known), "{point}");
        let mut ledger = Ledger::open(&root.join("ledger"), LedgerLimits::default()).unwrap();
        let lease = ledger
            .acquire(&contract, "supervisor-crash-recovery", 30_000)
            .unwrap();
        ledger.reconcile_fixture_final(&lease).unwrap();
        ledger.reconcile_fixture_final(&lease).unwrap();
        assert!(ledger.fixture_final_decision(&contract).unwrap().is_none());
        let budget = ledger.budget_snapshot(&contract).unwrap();
        assert_eq!(budget.held.workers, u64::from(!known), "{point}");
        assert_eq!(budget.spent.attempts, u64::from(known), "{point}");
        assert_eq!(budget.held.cpu_ms + budget.spent.cpu_ms, 30_000, "{point}");
        assert_eq!(
            budget.held.evaluations + budget.spent.evaluations,
            86,
            "{point}"
        );
        assert_eq!(
            db.query_row(
                "SELECT count(*) FROM final_fixture_records WHERE kind LIKE 'worker_receipt_%'",
                [],
                |r| r.get::<_, u64>(0)
            )
            .unwrap(),
            receipt_count
        );
        if !known && point != "supervisor_before_prepare" {
            // Even a rehashed on-disk success flag is not a cleanup witness.
            let launch: Vec<u8> = db
                .query_row(
                    "SELECT body FROM final_fixture_records WHERE kind LIKE 'worker_intent_%'",
                    [],
                    |r| r.get(0),
                )
                .unwrap();
            let launch: Value = serde_json::from_slice(&launch).unwrap();
            let kind = format!(
                "worker_receipt_{}",
                launch["cell"]
                    .as_str()
                    .unwrap()
                    .trim_start_matches("sha256:")
            );
            let fake = bytes(
                &json!({"cell":launch["cell"],"receipt":{"status":"completed","cleanup_confirmed":true}}),
            );
            db.execute(
                "INSERT INTO final_fixture_records VALUES(?1,?2,?3,?4)",
                rusqlite::params![
                    contract.as_str(),
                    kind,
                    fake,
                    ContentDigest::of_bytes(&fake).as_str()
                ],
            )
            .unwrap();
            assert!(ledger.reconcile_fixture_final(&lease).is_err());
            assert_eq!(ledger.budget_snapshot(&contract).unwrap().held.workers, 1);
        }
        drop(ledger);
        drop(db);
        std::fs::remove_dir_all(root).unwrap();
        std::fs::remove_file(control).unwrap();
    }
}
