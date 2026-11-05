//! Real native operational qualification and release, never independent review.
use super::super::tests::e2e::{self, Fixture};
use super::*;
fn d(s: &str) -> ContentDigest {
    ContentDigest::of_bytes(s.as_bytes())
}
fn native_qualification(f: &Fixture) -> PromotionQualification {
    f.ledger
        .qualify_fixture_promotion(&f.lease, &f.store, PromotionScope::OperationalFixture)
        .unwrap()
}
fn approve(f: &Fixture, runtime: &QualifiedTorch) -> PromotionExecutionGrant {
    operator::test_registry(&f.root.join("operator"));
    let qualification = native_qualification(f);
    let request = f
        .ledger
        .fixture_promotion_approval_request(
            &f.lease,
            &qualification,
            &f.edit,
            &f.smoke,
            runtime,
            d("granted-promotion"),
            None,
            &CancellationToken::default(),
        )
        .unwrap();
    let id = operator::issue(&request, operator::test_expiration()).unwrap();
    f.ledger
        .approve_fixture_promotion(
            &f.lease,
            &qualification,
            id,
            &f.edit,
            &f.smoke,
            runtime,
            d("granted-promotion"),
            None,
            &CancellationToken::default(),
        )
        .unwrap()
}
fn promote(
    f: &mut Fixture,
    runtime: &QualifiedTorch,
    grant: &PromotionExecutionGrant,
) -> Result<Acceptance> {
    f.ledger.promote_fixture_with_grant(
        &f.lease,
        &f.store,
        &f.edit,
        &f.smoke,
        runtime,
        d("granted-promotion"),
        None,
        &CancellationToken::default(),
        grant,
    )
}
#[test]
fn native_qualification_without_operator_grant_has_no_writes_and_strict_scope_refuses() {
    if e2e::dedicated_test() {
        return;
    }
    let mut f = Fixture::new(true);
    let before = f
        .ledger
        .budget_snapshot(f.lease.contract())
        .unwrap()
        .reservation_count;
    let qualification = native_qualification(&f);
    assert_eq!(qualification.native.contract(), f.lease.contract());
    let error = f
        .ledger
        .qualify_fixture_promotion(&f.lease, &f.store, PromotionScope::StrictFsync)
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::UnsupportedCapability);
    assert!(error.message.contains("strict-fsync"));
    let runtime = f.runtime();
    assert!(f
        .ledger
        .promote_fixture_bound(
            &f.lease,
            &f.store,
            &f.edit,
            &f.smoke,
            &runtime,
            d("ungranted-inner-promotion"),
            None,
            &CancellationToken::default(),
            None
        )
        .is_err());
    assert_eq!(
        before,
        f.ledger
            .budget_snapshot(f.lease.contract())
            .unwrap()
            .reservation_count
    );
    assert_eq!(
        f.ledger
            .connection
            .query_row("SELECT count(*) FROM promotion_records", [], |r| r
                .get::<_, u64>(0))
            .unwrap(),
        0
    );
    f.unchanged();
}
#[test]
fn private_grant_positive_atomic_reload_reopen_idempotence_and_archival_tamper() {
    if e2e::dedicated_test() {
        return;
    }
    let mut f = Fixture::new(true);
    let runtime = f.runtime();
    let grant = approve(&f, &runtime);
    let accepted = promote(&mut f, &runtime, &grant).unwrap();
    assert_eq!(accepted.body()["release"], grant.receipt);
    assert_eq!(
        accepted.body()["release"]["origin"],
        "native_operational_fixture_not_independent_review"
    );
    assert_eq!(
        promote(&mut f, &runtime, &grant).unwrap().digest(),
        accepted.digest()
    );
    let reopened = Ledger::open(&f.root.join("ledger"), LedgerLimits::default()).unwrap();
    assert_eq!(
        reopened
            .accepted_fixture_head(&f.family)
            .unwrap()
            .unwrap()
            .digest(),
        accepted.digest()
    );
    assert_eq!(
        reopened.state(f.lease.contract()).unwrap().stage,
        Stage::Completed
    );
    assert_eq!(
        reopened.events(f.lease.contract(), 0, 100).unwrap().len(),
        9
    );
    let snapshot = reopened.budget_snapshot(f.lease.contract()).unwrap();
    assert_eq!(snapshot.held.workers, 0);
    assert_eq!(snapshot.reservation_count, 2);
    assert!(
        read(&reopened.connection, f.lease.contract(), "child_registered")
            .unwrap()
            .is_some()
    );
    let intent = read(&reopened.connection, f.lease.contract(), "intent")
        .unwrap()
        .unwrap()
        .1;
    for pointer in [
        "/scope",
        "/origin",
        "/store",
        "/ledger_file",
        "/family",
        "/chain",
        "/decision",
        "/verifier",
        "/native_qualification_digest",
        "/native_qualification/contract",
        "/native_qualification/decision",
        "/native_qualification/profile",
        "/native_qualification/supervisor_profile",
        "/native_qualification/worker_image",
        "/native_qualification/supervisor_receipts",
        "/native_qualification/timing_scope",
        "/native_qualification/physical_publication",
        "/invocation/contract",
        "/invocation/epoch",
        "/invocation/boot",
        "/invocation/owner",
        "/invocation/lease_deadline",
        "/invocation/runtime",
        "/invocation/edit",
        "/invocation/smoke",
        "/invocation/operation",
        "/invocation/expected_head",
    ] {
        // Relink both audit receipts so semantic validation, not just equality,
        // must reject a self-consistent forged scope/binding.
        let mut corrupt_intent = intent.clone();
        let mut corrupt_body = accepted.body().clone();
        *corrupt_intent["request"]["release"]
            .pointer_mut(pointer)
            .unwrap() = json!("forged");
        if pointer.starts_with("/native_qualification/") {
            corrupt_intent["request"]["release"]["native_qualification_digest"] =
                json!(hash(&corrupt_intent["request"]["release"]["native_qualification"]).unwrap());
        }
        corrupt_body["release"] = corrupt_intent["request"]["release"].clone();
        assert!(
            audit(
                &reopened.connection,
                f.lease.contract(),
                &corrupt_intent,
                &corrupt_body
            )
            .is_err(),
            "relinked {pointer} accepted"
        );
    }
    assert!(audit(
        &reopened.connection,
        f.lease.contract(),
        &intent,
        accepted.body()
    )
    .is_ok());
    let mut unapproved = intent.clone();
    unapproved["request"]["release"] = Value::Null;
    let mut legacy = accepted.body().clone();
    legacy["release"] = Value::Null;
    legacy["scope"] = json!("trusted_native_linear_fixture_only_pending_T038_security_review");
    assert!(audit(
        &reopened.connection,
        f.lease.contract(),
        &unapproved,
        &legacy
    )
    .is_err());
    let id = serde_json::from_value(grant.receipt["operator_approval"].clone()).unwrap();
    assert!(operator::test_revoke_at(&id, "partial_write").is_err());
    assert_eq!(
        reopened
            .accepted_fixture_head(&f.family)
            .unwrap()
            .unwrap()
            .digest(),
        accepted.digest(),
        "torn revocation staging write must not corrupt authentic accepted history"
    );
    for point in ["partial_write", "after_rename"] {
        let status=std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact","coordinator::promotion::operator::tests::actual_writer_crashes_leave_only_complete_authority_and_readable_history","--test-threads=1"])
            .env("NF_OPERATOR_CRASH_ROOT",f.root.join("operator")).env("NF_OPERATOR_CRASH_ACTION","revoke")
            .env("NF_OPERATOR_CRASH_POINT",point).env("NF_OPERATOR_CRASH_EXPIRES",operator::test_expiration().to_string())
            .env("NF_OPERATOR_CRASH_ID",id.as_str()).status().unwrap();
        assert_eq!(status.code(), Some(86), "actual revoker crash at {point}");
        assert_eq!(
            reopened
                .accepted_fixture_head(&f.family)
                .unwrap()
                .unwrap()
                .digest(),
            accepted.digest(),
            "authentic acceptance must survive actual revoker process crash at {point}"
        );
    }
    operator::revoke(&id).unwrap();
    assert!(
        promote(&mut f, &runtime, &grant).is_err(),
        "revoked approval cannot authorize idempotent execution"
    );
    assert_eq!(
        reopened
            .accepted_fixture_head(&f.family)
            .unwrap()
            .unwrap()
            .digest(),
        accepted.digest()
    );
}
#[test]
fn private_grant_exact_input_scope_and_qualification_tamper_refusals() {
    if e2e::dedicated_test() {
        return;
    }
    let mut f = Fixture::new(true);
    let runtime = f.runtime();
    let grant = approve(&f, &runtime);
    let before = f
        .ledger
        .budget_snapshot(f.lease.contract())
        .unwrap()
        .reservation_count;
    for pointer in [
        "/schema_version",
        "/scope",
        "/origin",
        "/store",
        "/ledger_file",
        "/family",
        "/chain",
        "/decision",
        "/verifier",
        "/operator_approval",
        "/native_qualification_digest",
        "/native_qualification/contract",
        "/native_qualification/decision",
        "/native_qualification/profile",
        "/native_qualification/supervisor_profile",
        "/native_qualification/worker_image",
        "/native_qualification/supervisor_receipts",
        "/native_qualification/timing_scope",
        "/native_qualification/physical_publication",
        "/invocation/contract",
        "/invocation/epoch",
        "/invocation/owner",
        "/invocation/boot",
        "/invocation/lease_deadline",
        "/invocation/edit",
        "/invocation/smoke",
        "/invocation/runtime",
        "/invocation/operation",
        "/invocation/expected_head",
    ] {
        let mut forged = PromotionExecutionGrant {
            receipt: grant.receipt.clone(),
        };
        *forged.receipt.pointer_mut(pointer).unwrap() = json!("forged");
        if pointer.starts_with("/native_qualification/") {
            forged.receipt["native_qualification_digest"] =
                json!(hash(&forged.receipt["native_qualification"]).unwrap());
        }
        assert!(
            promote(&mut f, &runtime, &forged).is_err(),
            "{pointer} accepted"
        );
    }
    let mut strict = native_qualification(&f);
    strict.scope = PromotionScope::StrictFsync;
    let error = f
        .ledger
        .approve_fixture_promotion(
            &f.lease,
            &strict,
            d("approval"),
            &f.edit,
            &f.smoke,
            &runtime,
            d("granted-promotion"),
            None,
            &CancellationToken::default(),
        )
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::UnsupportedCapability);
    assert!(f
        .ledger
        .promote_fixture_with_grant(
            &f.lease,
            &f.store,
            &f.edit,
            &f.smoke,
            &runtime,
            d("other-operation"),
            None,
            &CancellationToken::default(),
            &grant
        )
        .is_err());
    let cancel = CancellationToken::default();
    cancel.request();
    assert!(f
        .ledger
        .promote_fixture_with_grant(
            &f.lease,
            &f.store,
            &f.edit,
            &f.smoke,
            &runtime,
            d("granted-promotion"),
            None,
            &cancel,
            &grant
        )
        .is_err());
    assert_eq!(
        before,
        f.ledger
            .budget_snapshot(f.lease.contract())
            .unwrap()
            .reservation_count
    );
    assert_eq!(
        f.ledger
            .connection
            .query_row("SELECT count(*) FROM promotion_records", [], |r| r
                .get::<_, u64>(0))
            .unwrap(),
        0
    );
    f.unchanged();
}
#[test]
fn private_grant_atomic_event_failure_reopen_recovery_cannot_replay_witness() {
    if e2e::dedicated_test() {
        return;
    }
    let mut f = Fixture::new(true);
    let runtime = f.runtime();
    let grant = approve(&f, &runtime);
    f.ledger.connection.execute_batch("CREATE TRIGGER reject_granted_event BEFORE INSERT ON events WHEN NEW.sequence=9 BEGIN SELECT RAISE(ABORT,'release event rollback'); END;").unwrap();
    assert!(promote(&mut f, &runtime, &grant).is_err());
    f.unchanged();
    let mut reopened = Ledger::open(&f.root.join("ledger"), LedgerLimits::default()).unwrap();
    assert!(reopened.accepted_fixture_head(&f.family).unwrap().is_none());
    let intent = read(&reopened.connection, f.lease.contract(), "intent")
        .unwrap()
        .unwrap()
        .1;
    let reservation: ContentDigest = serde_json::from_value(intent["reservation"].clone()).unwrap();
    assert!(!budgets::native_released(&reopened.connection, &reservation).unwrap());
    assert_eq!(
        reopened
            .budget_snapshot(f.lease.contract())
            .unwrap()
            .held
            .workers,
        1
    );
    assert!(read(&reopened.connection, f.lease.contract(), "reload")
        .unwrap()
        .is_some());
    assert!(read(&reopened.connection, f.lease.contract(), "costs")
        .unwrap()
        .is_none());
    assert!(supervision::all_cleanup_confirmed(&reopened.connection, f.lease.contract()).unwrap());
    reopened.recover_fixture_promotion(&f.lease).unwrap();
    assert!(budgets::native_released(&reopened.connection, &reservation).unwrap());
    f.ledger
        .connection
        .execute_batch("DROP TRIGGER reject_granted_event;")
        .unwrap();
    assert!(promote(&mut f, &runtime, &grant).is_err());
    f.unchanged();
}
fn grant_fence(takeover: bool) {
    if e2e::dedicated_test() {
        return;
    }
    let mut f = Fixture::new(true);
    let runtime = f.runtime();
    let grant = approve(&f, &runtime);
    if !takeover {
        f.ledger
            .request_stop(
                f.lease.contract(),
                d("granted-stop"),
                supervision::StopReason::OwnerStop,
            )
            .unwrap();
    } else {
        f.ledger.release(&f.lease).unwrap();
        let replacement = f
            .ledger
            .acquire(f.lease.contract(), "replacement", 300000)
            .unwrap();
        // A live new lease cannot reuse approval bound to the old epoch/owner.
        assert!(f
            .ledger
            .promote_fixture_with_grant(
                &replacement,
                &f.store,
                &f.edit,
                &f.smoke,
                &runtime,
                d("granted-promotion"),
                None,
                &CancellationToken::default(),
                &grant
            )
            .is_err());
    }
    assert!(promote(&mut f, &runtime, &grant).is_err());
    assert!(f.ledger.accepted_fixture_head(&f.family).unwrap().is_none());
    assert_eq!(
        f.ledger.state(f.lease.contract()).unwrap().stage,
        Stage::Verify
    );
    assert_eq!(
        f.ledger
            .connection
            .query_row("SELECT count(*) FROM promotion_records", [], |r| r
                .get::<_, u64>(0))
            .unwrap(),
        0
    );
}
#[test]
fn root_registry_denials_precede_budget_cas_child_and_pointer_changes() {
    let Some(case) = e2e::isolated_cases(&[
        "revoked",
        "expired",
        "foreign_store",
        "foreign_invocation",
        "forged",
    ]) else {
        return;
    };
    let mut f = Fixture::new(true);
    let runtime = f.runtime();
    let mut grant = approve(&f, &runtime);
    let id: ContentDigest =
        serde_json::from_value(grant.receipt["operator_approval"].clone()).unwrap();
    if case == "revoked" {
        operator::revoke(&id).unwrap();
    } else if case == "forged" {
        grant.receipt["operator_approval"] = json!(d("syntactically valid but never issued"));
    } else {
        let mut request = grant.receipt.clone();
        request.as_object_mut().unwrap().remove("operator_approval");
        if case == "foreign_store" {
            request["store"] = json!("another-ledger-store");
        }
        if case == "foreign_invocation" {
            request["invocation"]["operation"] = json!(d("another-operation"));
        }
        let expires = if case == "expired" {
            operator::test_expiration() - 299_900
        } else {
            operator::test_expiration()
        };
        let id = operator::issue(&request, expires).unwrap();
        grant.receipt = request;
        grant.receipt["operator_approval"] = json!(id);
        if case == "expired" {
            std::thread::sleep(std::time::Duration::from_millis(120));
        }
    }
    let before = f
        .ledger
        .budget_snapshot(f.lease.contract())
        .unwrap()
        .reservation_count;
    let used = f.store.used_bytes().unwrap();
    assert!(promote(&mut f, &runtime, &grant).is_err(), "{case}");
    assert_eq!(
        before,
        f.ledger
            .budget_snapshot(f.lease.contract())
            .unwrap()
            .reservation_count
    );
    assert_eq!(used, f.store.used_bytes().unwrap());
    assert_eq!(
        f.ledger
            .connection
            .query_row("SELECT count(*) FROM promotion_records", [], |r| r
                .get::<_, u64>(0))
            .unwrap(),
        0
    );
    f.unchanged();
}
#[test]
fn root_approval_cannot_follow_a_byte_copied_ledger_store_or_lease() {
    if e2e::dedicated_test() {
        return;
    }
    let f = Fixture::new(true);
    let runtime = f.runtime();
    let grant = approve(&f, &runtime);
    let destination = f.root.join("copied-ledger");
    use std::os::unix::fs::DirBuilderExt;
    std::fs::DirBuilder::new()
        .mode(0o700)
        .create(&destination)
        .unwrap();
    std::fs::copy(
        f.root.join("ledger/events.sqlite3"),
        destination.join("events.sqlite3"),
    )
    .unwrap();
    let mut copied = Ledger::open(&destination, LedgerLimits::default()).unwrap();
    assert_eq!(
        copied.store, f.ledger.store,
        "copy preserves the logical store identifier"
    );
    let before = copied
        .budget_snapshot(f.lease.contract())
        .unwrap()
        .reservation_count;
    let qualification = copied
        .qualify_fixture_promotion(&f.lease, &f.store, PromotionScope::OperationalFixture)
        .unwrap();
    let id = serde_json::from_value(grant.receipt["operator_approval"].clone()).unwrap();
    assert!(copied
        .approve_fixture_promotion(
            &f.lease,
            &qualification,
            id,
            &f.edit,
            &f.smoke,
            &runtime,
            d("granted-promotion"),
            None,
            &CancellationToken::default()
        )
        .is_err());
    assert!(copied
        .promote_fixture_with_grant(
            &f.lease,
            &f.store,
            &f.edit,
            &f.smoke,
            &runtime,
            d("granted-promotion"),
            None,
            &CancellationToken::default(),
            &grant
        )
        .is_err());
    assert_eq!(
        before,
        copied
            .budget_snapshot(f.lease.contract())
            .unwrap()
            .reservation_count
    );
    assert!(copied.accepted_fixture_head(&f.family).unwrap().is_none());
    assert_eq!(
        copied
            .connection
            .query_row("SELECT count(*) FROM promotion_records", [], |r| r
                .get::<_, u64>(0))
            .unwrap(),
        0
    );
    assert_eq!(
        copied.state(f.lease.contract()).unwrap().stage,
        Stage::Verify
    );
    f.unchanged();
}
#[test]
fn native_qualification_failed_final_is_not_releasable() {
    if e2e::dedicated_test() {
        return;
    }
    let f = Fixture::new(false);
    assert!(f
        .ledger
        .qualify_fixture_promotion(&f.lease, &f.store, PromotionScope::OperationalFixture)
        .is_err());
    f.unchanged();
}
#[test]
fn native_qualification_rechecks_actual_artifacts_not_report_booleans() {
    if e2e::dedicated_test() {
        return;
    }
    let f = Fixture::new(true);
    let _qualified = native_qualification(&f);
    let source = f
        .ledger
        .contract(f.lease.contract())
        .unwrap()
        .source
        .checkpoint
        .clone();
    let path = f
        .root
        .join("store/blobs")
        .join(source.as_str().trim_start_matches("sha256:"));
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    std::fs::write(&path, b"corrupt-native-source").unwrap();
    assert!(f
        .ledger
        .qualify_fixture_promotion(&f.lease, &f.store, PromotionScope::OperationalFixture)
        .is_err());
    f.unchanged();
}
#[test]
fn same_inode_database_rollback_cannot_replay_a_consumed_real_promotion() {
    if e2e::dedicated_test() {
        return;
    }
    use std::{io::Write, os::unix::fs::MetadataExt};
    let mut f = Fixture::new(true);
    let runtime = f.runtime();
    let grant = approve(&f, &runtime);
    let path = f.root.join("ledger/events.sqlite3");
    let original = std::fs::read(&path).unwrap();
    let identity = std::fs::metadata(&path).unwrap();
    let before = f
        .ledger
        .budget_snapshot(f.lease.contract())
        .unwrap()
        .reservation_count;
    let accepted = promote(&mut f, &runtime, &grant).unwrap();
    let reload = read(&f.ledger.connection, f.lease.contract(), "reload")
        .unwrap()
        .unwrap()
        .1;
    assert_eq!(reload["child_started"], true);
    assert_eq!(reload["cleanup_confirmed"], true);
    assert_eq!(reload["status"], "verified_fresh_process_reload");
    assert_eq!(
        f.ledger
            .accepted_fixture_head(&f.family)
            .unwrap()
            .unwrap()
            .digest(),
        accepted.digest()
    );
    let parked = Ledger::open(&f.root.join("parked"), LedgerLimits::default()).unwrap();
    drop(std::mem::replace(&mut f.ledger, parked)); // close every handle to the target SQLite file
    let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.set_len(0).unwrap();
    file.write_all(&original).unwrap();
    file.sync_all().unwrap();
    drop(file);
    let restored = std::fs::metadata(&path).unwrap();
    assert_eq!(
        (restored.dev(), restored.ino()),
        (identity.dev(), identity.ino())
    );
    f.ledger = Ledger::open(&f.root.join("ledger"), LedgerLimits::default()).unwrap();
    assert_eq!(
        f.ledger
            .budget_snapshot(f.lease.contract())
            .unwrap()
            .reservation_count,
        before
    );
    assert!(read(&f.ledger.connection, f.lease.contract(), "intent")
        .unwrap()
        .is_none());
    let used = f.store.used_bytes().unwrap();
    let error = promote(&mut f, &runtime, &grant).unwrap_err();
    assert!(error.message.contains("already consumed"), "{error}");
    assert_eq!(
        f.ledger
            .budget_snapshot(f.lease.contract())
            .unwrap()
            .reservation_count,
        before
    );
    assert_eq!(used, f.store.used_bytes().unwrap());
    assert!(
        read(&f.ledger.connection, f.lease.contract(), "child_intent")
            .unwrap()
            .is_none()
    );
    f.unchanged();
}
#[test]
fn private_grant_stop_is_not_a_cached_permission() {
    grant_fence(false);
}
#[test]
fn private_grant_epoch_takeover_is_not_a_cached_permission() {
    grant_fence(true);
}
