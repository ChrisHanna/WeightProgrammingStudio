#![cfg(target_os = "linux")]
#[path = "../../nf-core/tests/common/study_fixture.rs"]
#[allow(dead_code)]
mod fixture;
use fixture::{d, refresh, standard};
use nf_core::error::ErrorCode;
use nf_core::schemas::Validated;
use nf_core::study::StudyContract;
use nf_experiments::coordinator::{Ledger, LedgerLimits, Stage, Transition};
use std::fs;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Barrier,
};
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Root(PathBuf);
impl Root {
    fn new() -> Self {
        let root = PathBuf::from(std::env::var_os("KIROCREW_SCRATCH").unwrap()).join(format!(
            "coordinator-tests-{}-{}",
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
fn contract(value: &serde_json::Value) -> Validated<StudyContract> {
    Validated::parse(&serde_json::to_vec(value).unwrap()).unwrap()
}
fn transition(from: Stage, to: Stage, index: u32) -> Transition {
    Transition {
        operation: d(&format!("operation{index}")),
        expected: from,
        next: to,
        protocol: d("workflow-protocol"),
        evidence: d(&format!("evidence{index}")),
        message: "software state transition only".to_owned(),
    }
}
fn revision(parent: &nf_core::canonical::ContentDigest) -> serde_json::Value {
    let mut value = standard();
    value["revision"] = serde_json::json!(2);
    value["parent_contract"] = serde_json::json!(parent);
    value["approval_receipt"] = serde_json::json!(d("new-owner-approval"));
    value
}
#[test]
fn contracts_and_leases_real_freeze_reopen_idempotence_and_immutable_sql_rows() {
    let root = Root::new();
    let mut ledger = root.ledger();
    let request = contract(&standard());
    let id = ledger.freeze(&request).unwrap();
    assert_eq!(ledger.freeze(&request).unwrap(), id);
    assert_eq!(ledger.state(&id).unwrap().sequence, 1);
    assert_eq!(ledger.events(&id, 0, 10).unwrap().len(), 1);
    drop(ledger);
    let mut ledger = root.ledger();
    assert_eq!(ledger.contract(&id).unwrap().digest().unwrap(), id);
    let mut changed = standard();
    changed["objective"]["description"] = serde_json::json!("changed draft");
    assert_eq!(
        ledger.freeze(&contract(&changed)).unwrap_err().code,
        ErrorCode::RevisionConflict
    );
    let direct = rusqlite::Connection::open(root.0.join("ledger/events.sqlite3")).unwrap();
    assert!(direct
        .execute(
            "UPDATE contracts SET canonical=x'7b7d' WHERE digest=?1",
            [id.as_str()]
        )
        .is_err());
    assert!(direct.execute("DELETE FROM events", []).is_err());
    direct
        .execute_batch("DROP TRIGGER contracts_no_update")
        .unwrap();
    direct
        .execute(
            "UPDATE contracts SET canonical=x'7b7d' WHERE digest=?1",
            [id.as_str()],
        )
        .unwrap();
    assert_eq!(
        ledger.contract(&id).unwrap_err().code,
        ErrorCode::IntegrityMismatch
    );
}
#[test]
fn contracts_and_leases_revisions_inherit_budget_origin_and_invalidate_old_writers() {
    let root = Root::new();
    let mut ledger = root.ledger();
    let original = contract(&standard());
    let id = ledger.freeze(&original).unwrap();
    let lease = ledger.acquire(&id, "coordinator-a", 10000).unwrap();
    assert_eq!(
        ledger.freeze(&contract(&revision(&id))).unwrap_err().code,
        ErrorCode::LeaseConflict
    );
    ledger.release(&lease).unwrap();
    let next = ledger.freeze(&contract(&revision(&id))).unwrap();
    assert_eq!(
        ledger
            .acquire(&id, "stale-revision", 10000)
            .unwrap_err()
            .code,
        ErrorCode::StaleLease
    );
    assert_eq!(
        ledger
            .advance(&lease, transition(Stage::Frozen, Stage::Preflight, 1))
            .unwrap_err()
            .code,
        ErrorCode::StaleLease
    );
    let (origin, limits) = ledger.family_budget_origin(&original.study_family).unwrap();
    assert_eq!(origin, id);
    assert_eq!(
        limits.candidate_attempts,
        original.limits.candidate_attempts
    );
    let fresh = ledger.acquire(&next, "coordinator-b", 10000).unwrap();
    assert!(fresh.fencing_token() > lease.fencing_token());
    ledger.release(&fresh).unwrap();
    let mut third = revision(&next);
    third["revision"] = serde_json::json!(3);
    third["approval_receipt"] = serde_json::json!(d("third-approval"));
    third["limits"]["candidate_attempts"] = serde_json::json!(11);
    assert_eq!(
        ledger.freeze(&contract(&third)).unwrap_err().code,
        ErrorCode::PermissionDenied
    );
    assert_eq!(
        ledger
            .family_budget_origin(&original.study_family)
            .unwrap()
            .0,
        id
    );
}
#[test]
fn contracts_and_leases_competing_connections_get_only_one_coordinator() {
    let root = Root::new();
    let mut ledger = root.ledger();
    let id = ledger.freeze(&contract(&standard())).unwrap();
    let barrier = Arc::new(Barrier::new(4));
    let mut threads = Vec::new();
    for i in 0..4 {
        let mut handle = root.ledger();
        let barrier = barrier.clone();
        let id = id.clone();
        threads.push(std::thread::spawn(move || {
            barrier.wait();
            handle.acquire(&id, &format!("worker{i}"), 10000)
        }));
    }
    let results: Vec<_> = threads
        .into_iter()
        .map(|thread| thread.join().unwrap())
        .collect();
    assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
    assert!(results
        .iter()
        .filter_map(|result| result.as_ref().err())
        .all(|error| error.code == ErrorCode::LeaseConflict));
}
#[test]
fn contracts_and_leases_expiry_renewal_release_and_foreign_handles_fail_closed() {
    let root = Root::new();
    let mut ledger = root.ledger();
    let id = ledger.freeze(&contract(&standard())).unwrap();
    let expired = ledger.acquire(&id, "old", 100).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(110));
    assert_eq!(
        ledger.renew(&expired, 10000).unwrap_err().code,
        ErrorCode::StaleLease
    );
    let current = ledger.acquire(&id, "new", 10000).unwrap();
    assert!(current.fencing_token() > expired.fencing_token());
    assert_eq!(
        ledger
            .advance(&expired, transition(Stage::Frozen, Stage::Preflight, 1))
            .unwrap_err()
            .code,
        ErrorCode::StaleLease
    );
    let current = ledger.renew(&current, 20000).unwrap();
    let other = Root::new();
    let mut foreign = other.ledger();
    foreign.freeze(&contract(&standard())).unwrap();
    assert_eq!(
        foreign
            .advance(&current, transition(Stage::Frozen, Stage::Preflight, 1))
            .unwrap_err()
            .code,
        ErrorCode::StaleLease
    );
    ledger.release(&current).unwrap();
    assert_eq!(
        ledger
            .advance(&current, transition(Stage::Frozen, Stage::Preflight, 1))
            .unwrap_err()
            .code,
        ErrorCode::StaleLease
    );
}
#[test]
fn contracts_and_leases_atomic_events_are_idempotent_and_expected_state_bound() {
    let root = Root::new();
    let mut ledger = root.ledger();
    let id = ledger.freeze(&contract(&standard())).unwrap();
    let lease = ledger.acquire(&id, "owner", 10000).unwrap();
    let request = transition(Stage::Frozen, Stage::Preflight, 1);
    let first = ledger.advance(&lease, request.clone()).unwrap();
    assert_eq!(
        ledger.advance(&lease, request.clone()).unwrap().digest,
        first.digest
    );
    let mut changed = request;
    changed.evidence = d("other-evidence");
    assert_eq!(
        ledger.advance(&lease, changed).unwrap_err().code,
        ErrorCode::IdempotencyConflict
    );
    assert_eq!(
        ledger
            .advance(&lease, transition(Stage::Frozen, Stage::Execute, 2))
            .unwrap_err()
            .code,
        ErrorCode::InvalidTransition
    );
    assert_eq!(ledger.state(&id).unwrap().stage, Stage::Preflight);
    assert_eq!(ledger.events(&id, 0, 10).unwrap().len(), 2);
    assert_eq!(ledger.events(&id, 1, 1).unwrap()[0].digest, first.digest);
    assert!(ledger.events(&id, 0, 1001).is_err());
}
#[test]
fn contracts_and_leases_final_panel_closes_selection_and_reserves_holdout_across_revisions() {
    let root = Root::new();
    let mut ledger = root.ledger();
    let id = ledger.freeze(&contract(&standard())).unwrap();
    let lease = ledger.acquire(&id, "owner", 10000).unwrap();
    let stages = [
        Stage::Frozen,
        Stage::Preflight,
        Stage::Profile,
        Stage::Baseline,
        Stage::Design,
        Stage::Execute,
        Stage::Evaluate,
        Stage::Verify,
    ];
    for (index, pair) in stages.windows(2).enumerate() {
        ledger
            .advance(&lease, transition(pair[0], pair[1], index as u32))
            .unwrap();
    }
    assert_eq!(
        ledger
            .advance(&lease, transition(Stage::Verify, Stage::Search, 99))
            .unwrap_err()
            .code,
        ErrorCode::InvalidTransition
    );
    ledger.release(&lease).unwrap();
    assert_eq!(
        ledger.freeze(&contract(&revision(&id))).unwrap_err().code,
        ErrorCode::PermissionDenied
    );
    let mut next = revision(&id);
    for (index, binding) in next["portfolio"]["bindings"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .enumerate()
    {
        let item = &mut binding["splits"]["holdout"]["items"][0];
        item["item"] = serde_json::json!(d(&format!("fresh-holdout-item{index}")));
        item["group"] = serde_json::json!(d(&format!("fresh-holdout-group{index}")));
    }
    refresh(&mut next);
    let next = ledger.freeze(&contract(&next)).unwrap();
    assert_ne!(next, id);
    assert_eq!(ledger.state(&id).unwrap().stage, Stage::Verify);
    assert_eq!(ledger.events(&id, 0, 100).unwrap().len(), 8);
}
#[test]
fn contracts_and_leases_terminal_stages_and_quotas_do_not_publish_partial_events() {
    let root = Root::new();
    let mut ledger = Ledger::open(
        &root.0.join("ledger"),
        LedgerLimits {
            max_events: 2,
            max_contracts: 1,
            ..LedgerLimits::default()
        },
    )
    .unwrap();
    let id = ledger.freeze(&contract(&standard())).unwrap();
    let lease = ledger.acquire(&id, "owner", 10000).unwrap();
    ledger
        .advance(&lease, transition(Stage::Frozen, Stage::Preflight, 1))
        .unwrap();
    assert_eq!(
        ledger
            .advance(&lease, transition(Stage::Preflight, Stage::Profile, 2))
            .unwrap_err()
            .code,
        ErrorCode::QuotaExceeded
    );
    assert_eq!(ledger.state(&id).unwrap().sequence, 2);
    assert_eq!(ledger.state(&id).unwrap().stage, Stage::Preflight);
    ledger.release(&lease).unwrap();
    assert_eq!(
        ledger.freeze(&contract(&revision(&id))).unwrap_err().code,
        ErrorCode::QuotaExceeded
    );
    let other = Root::new();
    let mut ledger = other.ledger();
    let id = ledger.freeze(&contract(&standard())).unwrap();
    let lease = ledger.acquire(&id, "owner", 10000).unwrap();
    ledger
        .advance(&lease, transition(Stage::Frozen, Stage::Incomplete, 1))
        .unwrap();
    assert_eq!(
        ledger
            .advance(&lease, transition(Stage::Incomplete, Stage::Design, 2))
            .unwrap_err()
            .code,
        ErrorCode::InvalidTransition
    );
    assert!(ledger.acquire(&id, "restart", 10000).is_err());
}
#[test]
fn contracts_and_leases_database_and_side_file_aliases_are_rejected() {
    use std::os::unix::fs::symlink;
    let root = Root::new();
    let outside = root.0.join("outside");
    fs::create_dir(&outside).unwrap();
    symlink(&outside, root.0.join("link")).unwrap();
    assert!(Ledger::open(&root.0.join("link/sub"), LedgerLimits::default()).is_err());
    assert!(!outside.join("sub").exists());
    let mut ledger = root.ledger();
    let id = ledger.freeze(&contract(&standard())).unwrap();
    let db = root.0.join("ledger/events.sqlite3");
    fs::hard_link(&db, root.0.join("alias")).unwrap();
    assert!(ledger.contract(&id).is_err());
    fs::remove_file(root.0.join("alias")).unwrap();
    fs::write(root.0.join("victim"), b"safe").unwrap();
    symlink(
        root.0.join("victim"),
        root.0.join("ledger/events.sqlite3-journal"),
    )
    .unwrap();
    assert!(ledger.acquire(&id, "owner", 1000).is_err());
    assert_eq!(fs::read(root.0.join("victim")).unwrap(), b"safe");
}
#[test]
fn contracts_and_leases_private_root_and_foreign_formats_are_not_silently_rewritten() {
    use std::os::unix::fs::PermissionsExt;
    let root = Root::new();
    let public = root.0.join("public");
    fs::create_dir(&public).unwrap();
    fs::set_permissions(&public, fs::Permissions::from_mode(0o777)).unwrap();
    assert!(Ledger::open(&public, LedgerLimits::default()).is_err());
    assert!(!public.join("events.sqlite3").exists());
    let path = root.0.join("foreign");
    fs::create_dir(&path).unwrap();
    let connection = rusqlite::Connection::open(path.join("events.sqlite3")).unwrap();
    connection
        .execute_batch("CREATE TABLE foreign_data(value TEXT); PRAGMA user_version=9;")
        .unwrap();
    drop(connection);
    let before = fs::read(path.join("events.sqlite3")).unwrap();
    assert!(Ledger::open(&path, LedgerLimits::default()).is_err());
    assert_eq!(fs::read(path.join("events.sqlite3")).unwrap(), before);
}

#[test]
fn contracts_and_leases_mid_transaction_sql_failure_rolls_back_freeze_and_event_publication() {
    let root = Root::new();
    let mut ledger = root.ledger();
    let request = contract(&standard());
    let direct = rusqlite::Connection::open(root.0.join("ledger/events.sqlite3")).unwrap();
    direct.execute_batch("CREATE TRIGGER fail_freeze BEFORE INSERT ON contracts BEGIN SELECT RAISE(ABORT,'fixture interrupted freeze'); END;").unwrap();
    assert!(ledger.freeze(&request).is_err());
    assert!(ledger.family_budget_origin(&request.study_family).is_err());
    direct.execute_batch("DROP TRIGGER fail_freeze").unwrap();
    let id = ledger.freeze(&request).unwrap();
    let lease = ledger.acquire(&id, "owner", 10000).unwrap();
    direct.execute_batch("CREATE TRIGGER fail_head BEFORE UPDATE OF stage ON runs BEGIN SELECT RAISE(ABORT,'fixture interrupted event'); END;").unwrap();
    assert!(ledger
        .advance(&lease, transition(Stage::Frozen, Stage::Preflight, 1))
        .is_err());
    assert_eq!(ledger.events(&id, 0, 10).unwrap().len(), 1);
    assert_eq!(ledger.state(&id).unwrap().stage, Stage::Frozen);
    direct.execute_batch("DROP TRIGGER fail_head").unwrap();
    ledger
        .advance(&lease, transition(Stage::Frozen, Stage::Preflight, 1))
        .unwrap();
    assert_eq!(ledger.state(&id).unwrap().sequence, 2);
}

#[test]
fn contracts_and_leases_corrupt_old_event_cannot_be_hidden_by_a_valid_last_head() {
    let root = Root::new();
    let mut ledger = root.ledger();
    let id = ledger.freeze(&contract(&standard())).unwrap();
    let lease = ledger.acquire(&id, "owner", 10000).unwrap();
    ledger
        .advance(&lease, transition(Stage::Frozen, Stage::Preflight, 1))
        .unwrap();
    let direct = rusqlite::Connection::open(root.0.join("ledger/events.sqlite3")).unwrap();
    direct
        .execute_batch("DROP TRIGGER events_no_update")
        .unwrap();
    direct
        .execute(
            "UPDATE events SET body=x'7b7d' WHERE contract=?1 AND sequence=1",
            [id.as_str()],
        )
        .unwrap();
    assert_eq!(
        ledger.state(&id).unwrap_err().code,
        ErrorCode::IntegrityMismatch
    );
    assert_eq!(
        ledger
            .advance(&lease, transition(Stage::Preflight, Stage::Profile, 2))
            .unwrap_err()
            .code,
        ErrorCode::IntegrityMismatch
    );
}
