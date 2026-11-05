//! T038 kernel enforcement and deterministic durability-boundary regressions.
use super::*;
#[cfg(debug_assertions)]
use nf_experiments::coordinator::final_fixture::inject_fixture_fault;

#[test]
fn hard_memory_limit_rejects_unqualified_process_before_intent() {
    let hard = rustix::process::getrlimit(rustix::process::Resource::As).maximum;
    let mut s = Setup::custom(false, false, |c| {
        c["limits"]["host_memory_bytes"] =
            json!(hard.map_or(1_000_000_000, |n| n.saturating_sub(1)));
    });
    let error = s
        .ledger
        .run_fixture_final(
            &s.lease,
            &s.store,
            &s.edit,
            &s.noop,
            &s.smoke,
            &CancellationToken::default(),
        )
        .err()
        .unwrap();
    assert_eq!(error.code, nf_core::error::ErrorCode::UnsupportedCapability);
    assert_eq!(
        s.ledger
            .budget_snapshot(s.lease.contract())
            .unwrap()
            .reservation_count,
        0
    );
}

#[test]
fn legacy_or_forged_memory_and_publication_proofs_never_authorize() {
    if isolated() {
        return;
    }
    for (kind, field, replacement) in [
        ("intent", "hard_address_space_bytes", Value::Null),
        (
            "intent",
            "hard_address_space_bytes",
            json!(1_000_000_001_u64),
        ),
        ("completion", "linearization", Value::Null),
        ("commit_observation", "semantics", Value::Null),
        (
            "commit_observation",
            "completion",
            json!(d("unrelated-completion")),
        ),
        ("commit_observation", "observed_ms", json!(i64::MAX)),
        ("commit_observation", "cpu_ms", json!(u64::MAX)),
    ] {
        let mut s = Setup::new(false, false);
        assert!(s.run().report().accepted);
        let db = rusqlite::Connection::open(s.root.join("ledger/events.sqlite3")).unwrap();
        let body: Vec<u8> = db
            .query_row(
                "SELECT body FROM final_fixture_records WHERE kind=?1",
                [kind],
                |r| r.get(0),
            )
            .unwrap();
        let mut value: Value = serde_json::from_slice(&body).unwrap();
        value[field] = replacement;
        let body = bytes(&value);
        db.execute_batch("DROP TRIGGER final_fixture_no_update")
            .unwrap();
        db.execute(
            "UPDATE final_fixture_records SET body=?1,digest=?2 WHERE kind=?3",
            rusqlite::params![body, ContentDigest::of_bytes(&body).as_str(), kind],
        )
        .unwrap();
        assert!(s.ledger.fixture_final_decision(s.lease.contract()).is_err());
    }
}

#[test]
fn kernel_denies_heap_mmap_and_raising_hard_memory_limit() {
    if isolated() {
        return;
    }
    let hard = rustix::process::getrlimit(rustix::process::Resource::As)
        .maximum
        .unwrap();
    let mut heap = Vec::<u8>::new();
    assert!(heap.try_reserve_exact(hard as usize + 1).is_err());
    // SAFETY: anonymous private mapping, no supplied pointer or file; an
    // unexpected successful mapping is immediately unmapped before failing.
    let mapping = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            hard as usize + 1,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if mapping != libc::MAP_FAILED {
        unsafe {
            libc::munmap(mapping, hard as usize + 1);
        }
        panic!("kernel admitted mapping beyond hard address-space ceiling");
    }
    assert_eq!(
        std::io::Error::last_os_error().raw_os_error(),
        Some(libc::ENOMEM)
    );
    assert!(rustix::process::setrlimit(
        rustix::process::Resource::As,
        rustix::process::Rlimit {
            current: Some(hard + 4096),
            maximum: Some(hard + 4096)
        }
    )
    .is_err());
}

#[cfg(debug_assertions)]
#[test]
fn every_durable_boundary_is_atomic_or_reconciled_without_acceptance() {
    if isolated() {
        return;
    }
    for (point, admitted, published) in [
        ("admission_before_intent", false, false),
        ("intent_before_admission_commit", false, false),
        ("after_admission_commit", true, false),
        ("after_native_execution", true, false),
        ("execution_before_commit", true, false),
        ("after_execution_commit", true, false),
        ("decision_before_commit", true, false),
        ("after_decision_commit", true, false),
        ("settlement_before_completion", true, false),
        ("completion_before_publication_fence", true, false),
        ("publication_after_fence", true, false),
        ("after_publication_commit", true, false),
        ("observation_before_commit", true, false),
        ("observation_after_fence", true, false),
        ("after_observation_commit", true, true),
    ] {
        let mut s = Setup::new(false, false);
        inject_fixture_fault(point, move || {
            Err(nf_core::error::CoreError::new(
                nf_core::error::ErrorCode::TransactionAborted,
                "/test",
                point,
            ))
        });
        assert!(
            s.ledger
                .run_fixture_final(
                    &s.lease,
                    &s.store,
                    &s.edit,
                    &s.noop,
                    &s.smoke,
                    &CancellationToken::default()
                )
                .is_err(),
            "{point}"
        );
        let reopened = Ledger::open(&s.root.join("ledger"), LedgerLimits::default()).unwrap();
        let decision = reopened.fixture_final_decision(s.lease.contract()).unwrap();
        assert_eq!(decision.is_some(), published, "{point}");
        if let Some(decision) = decision {
            assert!(decision.report().accepted);
            assert!(!decision.report().promotion_authorized);
        }
        let snapshot = reopened.budget_snapshot(s.lease.contract()).unwrap();
        assert_eq!(snapshot.reservation_count, u32::from(admitted), "{point}");
        assert_eq!(snapshot.held.workers, 0, "{point}");
        assert_eq!(snapshot.spent.attempts, u64::from(admitted), "{point}");
        if admitted {
            assert_eq!(snapshot.spent.cpu_ms, 30_000, "{point}");
            assert_eq!(snapshot.spent.storage_bytes, 4_194_304, "{point}");
        }
    }
}

#[cfg(debug_assertions)]
#[test]
fn late_cancellation_rolls_back_completion_and_settlement_together() {
    if isolated() {
        return;
    }
    let mut s = Setup::new(false, false);
    let cancel = CancellationToken::default();
    let stop = cancel.clone();
    inject_fixture_fault("completion_before_publication_fence", move || {
        stop.request();
        Ok(())
    });
    assert!(s
        .ledger
        .run_fixture_final(&s.lease, &s.store, &s.edit, &s.noop, &s.smoke, &cancel)
        .is_err());
    assert!(s
        .ledger
        .fixture_final_decision(s.lease.contract())
        .unwrap()
        .is_none());
    let snapshot = s.ledger.budget_snapshot(s.lease.contract()).unwrap();
    assert_eq!(snapshot.held.workers, 0);
    assert_eq!(snapshot.spent.attempts, 1);
}

#[cfg(debug_assertions)]
#[test]
fn resource_admission_remains_held_until_authoritative_publication() {
    if isolated() {
        return;
    }
    let mut s = Setup::new(false, false);
    let observer = Ledger::open(&s.root.join("ledger"), LedgerLimits::default()).unwrap();
    let contract = s.lease.contract().clone();
    inject_fixture_fault("completion_before_publication_fence", move || {
        let snapshot = observer.budget_snapshot(&contract).unwrap();
        assert_eq!(snapshot.held.workers, 1);
        assert_eq!(snapshot.held.cpu_ms, 30_000);
        assert!(observer
            .fixture_final_decision(&contract)
            .unwrap()
            .is_none());
        Ok(())
    });
    assert!(s.run().report().accepted);
    assert_eq!(
        s.ledger
            .budget_snapshot(s.lease.contract())
            .unwrap()
            .held
            .workers,
        0
    );
}

#[cfg(debug_assertions)]
#[test]
fn lease_expiry_at_publication_never_publishes_stale_completion() {
    if isolated() {
        return;
    }
    let mut s = Setup::new(false, false);
    s.lease = s.ledger.renew(&s.lease, 5_000).unwrap();
    inject_fixture_fault("completion_before_publication_fence", || {
        std::thread::sleep(std::time::Duration::from_millis(5_001));
        Ok(())
    });
    assert!(s
        .ledger
        .run_fixture_final(
            &s.lease,
            &s.store,
            &s.edit,
            &s.noop,
            &s.smoke,
            &CancellationToken::default()
        )
        .is_err());
    let reopened = Ledger::open(&s.root.join("ledger"), LedgerLimits::default()).unwrap();
    assert!(reopened
        .fixture_final_decision(s.lease.contract())
        .unwrap()
        .is_none());
    // Loss of lease never invents cleanup: held admission remains visible.
    assert_eq!(
        reopened
            .budget_snapshot(s.lease.contract())
            .unwrap()
            .held
            .workers,
        1
    );
    let lease = s
        .ledger
        .acquire(s.lease.contract(), "recovery", 30_000)
        .unwrap();
    assert!(s.ledger.reconcile_fixture_final(&lease).is_err());
    assert_eq!(
        s.ledger
            .budget_snapshot(lease.contract())
            .unwrap()
            .held
            .workers,
        1
    );
}

/// A real SQLite COMMIT, not just a precommit hook, waits behind a SHARED
/// reader lock until after the lease/operation deadline. It must leave no
/// readable decision, even though its completion and settlement reached disk.
#[cfg(debug_assertions)]
#[test]
fn sqlite_commit_blocked_by_reader_is_quarantined_after_reopen() {
    if isolated() {
        return;
    }
    let mut s = Setup::new(false, false);
    s.ledger = Ledger::open(
        &s.root.join("ledger"),
        LedgerLimits {
            busy_timeout_ms: 30_000,
            ..LedgerLimits::default()
        },
    )
    .unwrap();
    s.lease = s.ledger.renew(&s.lease, 15_000).unwrap();
    let path = s.root.join("ledger/events.sqlite3");
    inject_fixture_fault("publication_after_fence", move || {
        let db = rusqlite::Connection::open(path).unwrap();
        db.execute_batch("BEGIN").unwrap();
        let body: Vec<u8> = db
            .query_row(
                "SELECT body FROM final_fixture_records WHERE kind='intent'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        let deadline = value["operation_deadline"].as_i64().unwrap();
        std::thread::spawn(move || {
            let now = rustix::time::clock_gettime(rustix::time::ClockId::Boottime);
            let now = now.tv_sec * 1000 + now.tv_nsec / 1_000_000;
            std::thread::sleep(std::time::Duration::from_millis(
                (deadline - now).max(0) as u64 + 100,
            ));
            db.execute_batch("ROLLBACK").unwrap();
        });
        Ok(())
    });
    assert!(s
        .ledger
        .run_fixture_final(
            &s.lease,
            &s.store,
            &s.edit,
            &s.noop,
            &s.smoke,
            &CancellationToken::default()
        )
        .is_err());
    let db = rusqlite::Connection::open(s.root.join("ledger/events.sqlite3")).unwrap();
    let completions: u64 = db
        .query_row(
            "SELECT count(*) FROM final_fixture_records WHERE kind='completion'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        completions, 1,
        "COMMIT must actually succeed, not time out as SQLITE_BUSY"
    );
    let reopened = Ledger::open(&s.root.join("ledger"), LedgerLimits::default()).unwrap();
    assert!(reopened
        .fixture_final_decision(s.lease.contract())
        .unwrap()
        .is_none());
    assert_eq!(
        reopened.fixture_final_status(s.lease.contract()).unwrap(),
        Some(Status::Incomplete)
    );
}

#[cfg(debug_assertions)]
#[test]
fn simultaneous_stop_cancel_after_completion_prevents_observation() {
    if isolated() {
        return;
    }
    let mut s = Setup::new(false, false);
    let mut other = Ledger::open(&s.root.join("ledger"), LedgerLimits::default()).unwrap();
    let contract = s.lease.contract().clone();
    let cancel = CancellationToken::default();
    let stop = cancel.clone();
    inject_fixture_fault("after_publication_commit", move || {
        other
            .request_stop(
                &contract,
                d("stop-after-completion"),
                nf_experiments::coordinator::supervision::StopReason::OwnerStop,
            )
            .unwrap();
        stop.request();
        Ok(())
    });
    assert!(s
        .ledger
        .run_fixture_final(&s.lease, &s.store, &s.edit, &s.noop, &s.smoke, &cancel)
        .is_err());
    let reopened = Ledger::open(&s.root.join("ledger"), LedgerLimits::default()).unwrap();
    assert!(reopened
        .fixture_final_decision(s.lease.contract())
        .unwrap()
        .is_none());
}

#[cfg(debug_assertions)]
#[test]
fn stale_owner_after_completion_cannot_attest_or_recover_acceptance() {
    if isolated() {
        return;
    }
    let mut s = Setup::new(false, false);
    let mut other = Ledger::open(&s.root.join("ledger"), LedgerLimits::default()).unwrap();
    let db = rusqlite::Connection::open(s.root.join("ledger/events.sqlite3")).unwrap();
    let contract = s.lease.contract().clone();
    inject_fixture_fault("after_publication_commit", move || {
        db.execute(
            "UPDATE runs SET deadline=0 WHERE contract=?1",
            [contract.as_str()],
        )
        .unwrap();
        let fresh = other.acquire(&contract, "fresh-owner", 30_000).unwrap();
        // Provisional completion did not settle admission. The stale executor
        // is STILL LIVE inside this hook: lease takeover cannot prove cleanup.
        assert!(other.reconcile_fixture_final(&fresh).is_err());
        assert_eq!(other.budget_snapshot(&contract).unwrap().held.workers, 1);
        assert!(other.fixture_final_decision(&contract).unwrap().is_none());
        Ok(())
    });
    assert!(s
        .ledger
        .run_fixture_final(
            &s.lease,
            &s.store,
            &s.edit,
            &s.noop,
            &s.smoke,
            &CancellationToken::default()
        )
        .is_err());
    assert!(s
        .ledger
        .fixture_final_decision(s.lease.contract())
        .unwrap()
        .is_none());
}

#[test]
fn strict_publication_prerequisite_is_explicitly_unavailable_without_admission() {
    let s = Setup::new(false, false);
    let error = s.ledger.require_strict_fixture_publication().unwrap_err();
    assert_eq!(error.code, nf_core::error::ErrorCode::UnsupportedCapability);
    assert_eq!(
        s.ledger
            .budget_snapshot(s.lease.contract())
            .unwrap()
            .reservation_count,
        0
    );
}

/// Counterexample documenting the STILL UNSUPPORTED hard requirement, not an
/// acceptance test. Recovery observation cannot bound its own persistence.
#[cfg(debug_assertions)]
#[test]
fn observation_persistence_counterexample_keeps_strict_capability_unavailable() {
    if isolated() {
        return;
    }
    let mut s = Setup::new(false, false);
    s.ledger = Ledger::open(
        &s.root.join("ledger"),
        LedgerLimits {
            busy_timeout_ms: 30_000,
            ..LedgerLimits::default()
        },
    )
    .unwrap();
    s.lease = s.ledger.renew(&s.lease, 15_000).unwrap();
    let path = s.root.join("ledger/events.sqlite3");
    inject_fixture_fault("observation_after_fence", move || {
        let db = rusqlite::Connection::open(path).unwrap();
        db.execute_batch("BEGIN").unwrap();
        let body: Vec<u8> = db
            .query_row(
                "SELECT body FROM final_fixture_records WHERE kind='intent'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        let deadline = value["operation_deadline"].as_i64().unwrap();
        // Keep a real SHARED reader lock through the final transaction's
        // precommit lease recheck. Only COMMIT stalls past the deadline.
        std::thread::spawn(move || {
            let now = rustix::time::clock_gettime(rustix::time::ClockId::Boottime);
            let now = now.tv_sec * 1000 + now.tv_nsec / 1_000_000;
            std::thread::sleep(std::time::Duration::from_millis(
                (deadline - now).max(0) as u64 + 100,
            ));
            db.execute_batch("ROLLBACK").unwrap();
        });
        Ok(())
    });
    let evidence = s.run();
    assert!(
        evidence.report().accepted,
        "fixture evidence is not strict enforcement"
    );
    assert_eq!(
        evidence.report().scope,
        "trusted_native_linear_fixture_only_not_generalization_or_strict_T038_or_T039"
    );
    let reopened = Ledger::open(&s.root.join("ledger"), LedgerLimits::default()).unwrap();
    assert!(
        reopened
            .fixture_final_decision(s.lease.contract())
            .unwrap()
            .unwrap()
            .report()
            .accepted
    );
    assert!(reopened.require_strict_fixture_publication().is_err());
    // Completed operational evidence is distinct from a physical fsync cutoff.
    // The qualification carries this distinction, never an interchangeable bool.
    let qualification = reopened
        .qualify_fixture_final(s.lease.contract(), &s.store)
        .unwrap();
    assert_eq!(
        qualification.timing_scope(),
        &nf_experiments::coordinator::final_fixture::FixtureTimingScope::OperationalCancellation
    );
    assert_eq!(qualification.decision(), evidence.id());
}

/// Original negative bar, collected by default. Passing this necessary condition
/// does NOT establish hard enforcement through the observation's own commit.
#[cfg(debug_assertions)]
#[test]
fn strict_deadline_includes_authoritative_sqlite_commit() {
    if isolated() {
        return;
    }
    let mut s = Setup::new(false, false);
    let db = rusqlite::Connection::open(s.root.join("ledger/events.sqlite3")).unwrap();
    inject_fixture_fault("publication_after_fence", move || {
        let body: Vec<u8> = db
            .query_row(
                "SELECT body FROM final_fixture_records WHERE kind='intent'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        let deadline = value["operation_deadline"].as_i64().unwrap();
        let now = rustix::time::clock_gettime(rustix::time::ClockId::Boottime);
        let now = now.tv_sec * 1000 + now.tv_nsec / 1_000_000;
        std::thread::sleep(std::time::Duration::from_millis(
            (deadline - now).max(0) as u64 + 5,
        ));
        Ok(())
    });
    let result = s.ledger.run_fixture_final(
        &s.lease,
        &s.store,
        &s.edit,
        &s.noop,
        &s.smoke,
        &CancellationToken::default(),
    );
    assert!(
        result.is_err(),
        "strict operation deadline must cover authoritative SQLite commit"
    );
    assert!(s
        .ledger
        .fixture_final_decision(s.lease.contract())
        .unwrap()
        .is_none());
}

#[cfg(debug_assertions)]
#[test]
fn process_death_at_every_boundary_reconciles_after_reopen() {
    if isolated() {
        return;
    }
    if let Ok(point) = std::env::var("T038_TEST_CRASH_POINT") {
        let point: &'static str = Box::leak(point.into_boxed_str());
        let control = std::env::var("T038_TEST_CRASH_CONTROL").unwrap();
        let mut s = Setup::new(false, false);
        std::fs::write(
            control,
            bytes(&json!({"root":s.root,"contract":s.lease.contract()})),
        )
        .unwrap();
        inject_fixture_fault(point, || std::process::exit(86));
        s.run();
        panic!("crash boundary was not reached");
    }
    for (point, admitted, published) in [
        ("admission_before_intent", false, false),
        ("intent_before_admission_commit", false, false),
        ("after_admission_commit", true, false),
        ("after_native_execution", true, false),
        ("execution_before_commit", true, false),
        ("after_execution_commit", true, false),
        ("decision_before_commit", true, false),
        ("after_decision_commit", true, false),
        ("settlement_before_completion", true, false),
        ("completion_before_publication_fence", true, false),
        ("publication_after_fence", true, false),
        ("after_publication_commit", true, false),
        ("observation_before_commit", true, false),
        ("observation_after_fence", true, false),
        ("after_observation_commit", true, true),
    ] {
        let control =
            std::env::temp_dir().join(format!("t038-crash-{}-{point}.json", std::process::id()));
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                std::thread::current().name().unwrap(),
                "--test-threads=1",
            ])
            .env("T038_TEST_CRASH_POINT", point)
            .env("T038_TEST_CRASH_CONTROL", &control)
            .status()
            .unwrap();
        assert_eq!(status.code(), Some(86), "{point}");
        let info: Value = serde_json::from_slice(&std::fs::read(&control).unwrap()).unwrap();
        let root = PathBuf::from(info["root"].as_str().unwrap());
        let contract: ContentDigest = serde_json::from_value(info["contract"].clone()).unwrap();
        let mut ledger = Ledger::open(&root.join("ledger"), LedgerLimits::default()).unwrap();
        // Test operator advances the old lease expiry, without rewriting intent,
        // reservation or evidence. The executor itself cannot do this.
        let db = rusqlite::Connection::open(root.join("ledger/events.sqlite3")).unwrap();
        db.execute(
            "UPDATE runs SET deadline=0 WHERE contract=?1",
            [contract.as_str()],
        )
        .unwrap();
        drop(db);
        let lease = ledger.acquire(&contract, "after-crash", 30_000).unwrap();
        ledger.reconcile_fixture_final(&lease).unwrap();
        ledger.reconcile_fixture_final(&lease).unwrap(); // idempotent
        assert_eq!(
            ledger.fixture_final_decision(&contract).unwrap().is_some(),
            published,
            "{point}"
        );
        let snapshot = ledger.budget_snapshot(&contract).unwrap();
        assert_eq!(snapshot.reservation_count, u32::from(admitted), "{point}");
        assert_eq!(snapshot.held.workers, 0, "{point}");
        assert_eq!(snapshot.spent.attempts, u64::from(admitted), "{point}");
        if admitted {
            assert_eq!(snapshot.spent.cpu_ms, 30_000);
            assert_eq!(snapshot.spent.storage_bytes, 4_194_304);
        }
        drop(ledger);
        std::fs::remove_dir_all(root).unwrap();
        std::fs::remove_file(control).unwrap();
    }
}
