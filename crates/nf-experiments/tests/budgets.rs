#![cfg(target_os = "linux")]
#[path = "../../nf-core/tests/common/study_fixture.rs"]
#[allow(dead_code)]
mod fixture;
use fixture::{d, standard};
use nf_core::error::ErrorCode;
use nf_core::schemas::Validated;
use nf_core::study::StudyContract;
use nf_experiments::coordinator::budgets::{
    ExternalCall, OutcomeKind, Request, Resources, Settlement, Usage, WorkKind,
};
use nf_experiments::coordinator::{Ledger, LedgerLimits};
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
        let path = std::env::var_os("KIROCREW_SCRATCH")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir)
            .join(format!(
                "budget-tests-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn open(&self) -> Ledger {
        Ledger::open(&self.0.join("ledger"), LedgerLimits::default()).unwrap()
    }
}
impl Drop for Root {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn parse(value: &serde_json::Value) -> Validated<StudyContract> {
    Validated::parse(&serde_json::to_vec(value).unwrap()).unwrap()
}
fn request(name: &str) -> Request {
    Request {
        schema_version: 1,
        operation: d(name),
        attempt: 0,
        parent: None,
        kind: WorkKind::Candidate,
        resources: Resources {
            cpu_ms: 100,
            attempts: 1,
            evaluations: 1,
            storage_bytes: 10,
            host_bytes: 1000,
            workers: 1,
            ..Resources::default()
        },
        wall_ms: 1000,
        external: None,
    }
}
fn settlement(name: &str, request: &Request) -> Settlement {
    Settlement {
        operation: d(name),
        outcome: OutcomeKind::Completed,
        usage: Usage::known(&request.resources),
        resources_released: true,
        evidence: d("cost-receipt"),
    }
}
#[test]
fn budgets_real_reservations_idempotence_settlement_and_reopen_preserve_costs() {
    let root = Root::new();
    let mut ledger = root.open();
    let contract = parse(&standard());
    let id = ledger.freeze(&contract).unwrap();
    let lease = ledger.acquire(&id, "owner", 300000).unwrap();
    let work = request("candidate");
    let reservation = ledger.reserve(&lease, work.clone()).unwrap();
    assert_eq!(
        ledger.reserve(&lease, work.clone()).unwrap().id(),
        reservation.id()
    );
    let held = ledger.budget_snapshot(&id).unwrap();
    assert_eq!(held.held.attempts, 1);
    assert_eq!(held.held.host_bytes, 1000);
    assert_eq!(held.spent.attempts, 0);
    let mut actual = settlement("cost", &work);
    actual.usage.cpu_ms = Some(40);
    ledger.settle(&lease, &reservation, actual.clone()).unwrap();
    ledger.settle(&lease, &reservation, actual).unwrap();
    let result = ledger.budget_snapshot(&id).unwrap();
    assert_eq!(result.spent.cpu_ms, 40);
    assert_eq!(result.spent.attempts, 1);
    assert_eq!(result.spent.storage_bytes, 10);
    assert_eq!(result.held.workers, 0);
    ledger.release(&lease).unwrap();
    drop(ledger);
    let mut ledger = root.open();
    assert_eq!(ledger.budget_snapshot(&id).unwrap().spent.cpu_ms, 40);
    let lease = ledger.acquire(&id, "new-owner", 300000).unwrap();
    assert_eq!(
        ledger
            .recover_reservation(&lease, reservation.id())
            .unwrap()
            .id(),
        reservation.id()
    );
}
#[test]
fn budgets_unknown_billing_holds_full_price_until_real_reconciliation() {
    let root = Root::new();
    let mut ledger = root.open();
    let mut value = standard();
    value["permissions"]["tools"] =
        serde_json::json!(["inspect", "direct_edit", "external_request"]);
    value["permissions"]["external_destinations"] =
        serde_json::json!({"provider.fixture":["summaries"]});
    value["sources"] = serde_json::json!([{"kind":"teacher","artifact":d("teacher"),"destination":"provider.fixture","payloads":["summaries"],"access":"development"}]);
    value["limits"]["teacher_tokens"] = serde_json::json!(100);
    value["limits"]["external_spend_microusd"] = serde_json::json!(500);
    let id = ledger.freeze(&parse(&value)).unwrap();
    let lease = ledger.acquire(&id, "owner", 300000).unwrap();
    let mut work = request("teacher");
    work.kind = WorkKind::Teacher;
    work.resources.attempts = 0;
    work.resources.teacher_tokens = 10;
    work.resources.spend_microusd = 500;
    work.external = Some(ExternalCall {
        source: d("teacher"),
        destination: "provider.fixture".to_owned(),
        payloads: [nf_core::schemas::DataClass::Summaries].into(),
        maximum_price_microusd: Some(500),
        maximum_tokens: 10,
    });
    let reservation = ledger.reserve(&lease, work.clone()).unwrap();
    let mut unknown = settlement("unknown-bill", &work);
    unknown.usage.spend_microusd = None;
    ledger.settle(&lease, &reservation, unknown).unwrap();
    let snapshot = ledger.budget_snapshot(&id).unwrap();
    assert_eq!(snapshot.spent.spend_microusd, 500);
    assert_eq!(snapshot.unknown_charged.spend_microusd, 500);
    let mut next = work.clone();
    next.operation = d("next-teacher");
    assert_eq!(
        ledger.reserve(&lease, next.clone()).unwrap_err().code,
        ErrorCode::QuotaExceeded
    );
    let mut actual = settlement("reconciled", &work);
    actual.usage.spend_microusd = Some(50);
    ledger.settle(&lease, &reservation, actual).unwrap();
    assert_eq!(
        ledger
            .budget_snapshot(&id)
            .unwrap()
            .unknown_charged
            .spend_microusd,
        0
    );
    next.resources.spend_microusd = 450;
    next.external.as_mut().unwrap().maximum_price_microusd = Some(450);
    assert!(ledger.reserve(&lease, next).is_ok());
}
#[test]
fn budgets_branch_children_cannot_overallocate_or_double_count_shared_costs() {
    let root = Root::new();
    let mut ledger = root.open();
    let id = ledger.freeze(&parse(&standard())).unwrap();
    let lease = ledger.acquire(&id, "owner", 300000).unwrap();
    let mut branch = request("branch");
    branch.kind = WorkKind::Branch;
    branch.resources.cpu_ms = 150;
    branch.resources.attempts = 2;
    branch.resources.evaluations = 2;
    branch.resources.storage_bytes = 20;
    branch.wall_ms = 10000;
    let parent = ledger.reserve(&lease, branch.clone()).unwrap();
    let mut child = request("child-a");
    child.parent = Some(parent.id().clone());
    let first = ledger.reserve(&lease, child.clone()).unwrap();
    let mut sibling = child.clone();
    sibling.operation = d("child-b");
    assert!(ledger.reserve(&lease, sibling.clone()).is_err());
    let mut actual = settlement("child-cost", &child);
    actual.usage.cpu_ms = Some(40);
    ledger.settle(&lease, &first, actual).unwrap();
    assert_eq!(ledger.budget_snapshot(&id).unwrap().spent.cpu_ms, 40);
    assert_eq!(ledger.budget_snapshot(&id).unwrap().held.cpu_ms, 110);
    let second = ledger.reserve(&lease, sibling.clone()).unwrap();
    assert!(ledger
        .close_branch(&lease, &parent, d("too-soon"), d("receipt"))
        .is_err());
    ledger
        .settle(&lease, &second, settlement("second-cost", &sibling))
        .unwrap();
    ledger
        .close_branch(&lease, &parent, d("branch-done"), d("receipt"))
        .unwrap();
    let result = ledger.budget_snapshot(&id).unwrap();
    assert_eq!(result.spent.cpu_ms, 140);
    assert_eq!(result.spent.attempts, 2);
    assert_eq!(result.held.cpu_ms, 0);
    assert_eq!(result.held.workers, 0);
}
#[test]
fn budgets_concurrent_requests_cannot_race_past_family_attempt_or_memory_caps() {
    let root = Root::new();
    let mut ledger = root.open();
    let mut value = standard();
    value["limits"]["candidate_attempts"] = serde_json::json!(2);
    value["limits"]["concurrent_workers"] = serde_json::json!(2);
    value["limits"]["host_memory_bytes"] = serde_json::json!(2000);
    let id = ledger.freeze(&parse(&value)).unwrap();
    let lease = ledger.acquire(&id, "owner", 300000).unwrap();
    let barrier = Arc::new(Barrier::new(6));
    let mut threads = Vec::new();
    for i in 0..6 {
        let mut worker = root.open();
        let lease = lease.clone();
        let barrier = barrier.clone();
        threads.push(std::thread::spawn(move || {
            barrier.wait();
            worker.reserve(&lease, request(&format!("request{i}")))
        }));
    }
    let results: Vec<_> = threads
        .into_iter()
        .map(|thread| thread.join().unwrap())
        .collect();
    assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 2);
    let snapshot = ledger.budget_snapshot(&id).unwrap();
    assert_eq!(snapshot.held.attempts, 2);
    assert_eq!(snapshot.held.host_bytes, 2000);
}
#[test]
fn budgets_retry_revision_and_foreign_handles_cannot_reset_accounting() {
    let root = Root::new();
    let mut ledger = root.open();
    let original = parse(&standard());
    let id = ledger.freeze(&original).unwrap();
    let lease = ledger.acquire(&id, "owner", 300000).unwrap();
    let mut work = request("retry-root");
    let one = ledger.reserve(&lease, work.clone()).unwrap();
    let mut retry = work.clone();
    retry.attempt = 1;
    assert!(ledger.reserve(&lease, retry.clone()).is_err());
    ledger
        .settle(&lease, &one, settlement("first-failed", &work))
        .unwrap();
    for attempt in 1..=2 {
        work.attempt = attempt;
        let handle = ledger.reserve(&lease, work.clone()).unwrap();
        ledger
            .settle(
                &lease,
                &handle,
                settlement(&format!("attempt-cost{attempt}"), &work),
            )
            .unwrap();
    }
    work.attempt = 3;
    assert!(ledger.reserve(&lease, work).is_err());
    ledger.release(&lease).unwrap();
    let mut revision = standard();
    revision["revision"] = serde_json::json!(2);
    revision["parent_contract"] = serde_json::json!(id);
    revision["approval_receipt"] = serde_json::json!(d("new-approval"));
    let next = ledger.freeze(&parse(&revision)).unwrap();
    let fresh = ledger.acquire(&next, "new", 300000).unwrap();
    assert_eq!(ledger.budget_snapshot(&next).unwrap().spent.attempts, 3);
    assert_eq!(
        ledger
            .reserve(&fresh, request("retry-root"))
            .unwrap_err()
            .code,
        ErrorCode::IdempotencyConflict
    );
    let other = Root::new();
    let mut foreign = other.open();
    let foreign_id = foreign.freeze(&original).unwrap();
    let foreign_lease = foreign.acquire(&foreign_id, "foreign", 300000).unwrap();
    assert!(foreign
        .settle(
            &foreign_lease,
            &one,
            settlement("forged", &request("retry-root"))
        )
        .is_err());
}
#[test]
fn budgets_unclean_or_cancelled_work_is_not_free_and_stale_writers_cannot_settle() {
    let root = Root::new();
    let mut ledger = root.open();
    let id = ledger.freeze(&parse(&standard())).unwrap();
    let lease = ledger.acquire(&id, "owner", 300000).unwrap();
    let work = request("unclean");
    let handle = ledger.reserve(&lease, work.clone()).unwrap();
    ledger
        .settle(
            &lease,
            &handle,
            Settlement {
                operation: d("pending-cleanup"),
                outcome: OutcomeKind::Incomplete,
                usage: Usage::default(),
                resources_released: false,
                evidence: d("cleanup-unknown"),
            },
        )
        .unwrap();
    assert_eq!(ledger.budget_snapshot(&id).unwrap().held.workers, 1);
    assert!(ledger.reserve(&lease, request("blocked-worker")).is_err());
    let mut actual = settlement("cancelled-cost", &work);
    actual.outcome = OutcomeKind::Cancelled;
    ledger.settle(&lease, &handle, actual).unwrap();
    assert_eq!(ledger.budget_snapshot(&id).unwrap().spent.attempts, 1);
    ledger.release(&lease).unwrap();
    assert!(ledger
        .settle(&lease, &handle, settlement("stale", &work))
        .is_err());
}
#[test]
fn budgets_preflight_never_turns_a_dto_or_reservation_into_hard_limit_permission() {
    let root = Root::new();
    let mut ledger = root.open();
    let id = ledger.freeze(&parse(&standard())).unwrap();
    let lease = ledger.acquire(&id, "owner", 300000).unwrap();
    let mut work = request("worker");
    let report = ledger.preflight(&lease, &work).unwrap();
    assert!(report.accounting_eligible);
    assert!(!report.dispatch_allowed);
    assert!(report.requires_worker);
    assert!(!report.unmet_required_limits.is_empty());
    work.external = Some(ExternalCall {
        source: d("unapproved"),
        destination: "remote".to_owned(),
        payloads: [nf_core::schemas::DataClass::Summaries].into(),
        maximum_price_microusd: None,
        maximum_tokens: 1,
    });
    assert_eq!(
        ledger.preflight(&lease, &work).unwrap_err().code,
        ErrorCode::PermissionDenied
    );
    assert_eq!(ledger.budget_snapshot(&id).unwrap().reservation_count, 0);
    work.external = None;
    work.resources.host_bytes = 100_000_001;
    assert_eq!(
        ledger.preflight(&lease, &work).unwrap_err().code,
        ErrorCode::QuotaExceeded
    );
}
#[test]
fn budgets_reported_overruns_are_retained_and_halt_family_not_refunded() {
    let root = Root::new();
    let mut ledger = root.open();
    let id = ledger.freeze(&parse(&standard())).unwrap();
    let lease = ledger.acquire(&id, "owner", 300000).unwrap();
    let work = request("overrun");
    let handle = ledger.reserve(&lease, work.clone()).unwrap();
    let mut actual = settlement("overrun-report", &work);
    actual.usage.cpu_ms = Some(101);
    assert_eq!(
        ledger
            .settle(&lease, &handle, actual.clone())
            .unwrap_err()
            .code,
        ErrorCode::BudgetBreach
    );
    assert_eq!(
        ledger.settle(&lease, &handle, actual).unwrap_err().code,
        ErrorCode::BudgetBreach
    );
    let snapshot = ledger.budget_snapshot(&id).unwrap();
    assert!(snapshot.accounting_halted);
    assert_eq!(snapshot.reported_overruns.len(), 1);
    assert_eq!(
        snapshot.reported_overruns[0].settlement.usage.cpu_ms,
        Some(101)
    );
    assert_eq!(snapshot.held.cpu_ms, 100);
    assert_eq!(
        ledger.reserve(&lease, request("new")).unwrap_err().code,
        ErrorCode::BudgetBreach
    );
    assert_eq!(
        ledger.preflight(&lease, &request("new")).unwrap_err().code,
        ErrorCode::BudgetBreach
    );
}

#[test]
fn budgets_family_wall_deadline_and_unknown_reboot_clock_do_not_reset() {
    let root = Root::new();
    let mut ledger = root.open();
    let mut value = standard();
    value["limits"]["wall_ms"] = serde_json::json!(1000);
    let id = ledger.freeze(&parse(&value)).unwrap();
    let lease = ledger.acquire(&id, "owner", 300000).unwrap();
    let mut work = request("deadline");
    work.wall_ms = 500;
    ledger.reserve(&lease, work.clone()).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(1100));
    work.operation = d("late");
    assert_eq!(
        ledger.reserve(&lease, work).unwrap_err().code,
        ErrorCode::QuotaExceeded
    );
    assert!(
        ledger
            .budget_snapshot(&id)
            .unwrap()
            .elapsed_wall_ms
            .unwrap()
            >= 1000
    );
    let direct = rusqlite::Connection::open(root.0.join("ledger/events.sqlite3")).unwrap();
    direct
        .execute_batch(
            "DROP TRIGGER epoch_no_update;UPDATE budget_epochs SET boot='different-kernel-boot';",
        )
        .unwrap();
    let mut no_clock = request("no-clock-reset");
    no_clock.wall_ms = 5;
    assert_eq!(
        ledger.preflight(&lease, &no_clock).unwrap_err().code,
        ErrorCode::UnsupportedCapability
    );
    assert_eq!(ledger.budget_snapshot(&id).unwrap().elapsed_wall_ms, None);
}

#[test]
fn budgets_approved_external_source_without_a_price_or_token_bound_is_unsupported() {
    let root = Root::new();
    let mut ledger = root.open();
    let mut value = standard();
    value["permissions"]["tools"] =
        serde_json::json!(["inspect", "direct_edit", "external_request"]);
    value["permissions"]["external_destinations"] = serde_json::json!({"fixture":["summaries"]});
    value["sources"] = serde_json::json!([{"kind":"teacher","artifact":d("teacher"),"destination":"fixture","payloads":["summaries"],"access":"development"}]);
    value["limits"]["teacher_tokens"] = serde_json::json!(10);
    let id = ledger.freeze(&parse(&value)).unwrap();
    let lease = ledger.acquire(&id, "owner", 300000).unwrap();
    let mut work = request("price");
    work.kind = WorkKind::Teacher;
    work.resources.teacher_tokens = 10;
    work.external = Some(ExternalCall {
        source: d("teacher"),
        destination: "fixture".to_owned(),
        payloads: [nf_core::schemas::DataClass::Summaries].into(),
        maximum_price_microusd: None,
        maximum_tokens: 10,
    });
    assert_eq!(
        ledger.reserve(&lease, work.clone()).unwrap_err().code,
        ErrorCode::UnsupportedCapability
    );
    work.external.as_mut().unwrap().maximum_price_microusd = Some(0);
    work.external.as_mut().unwrap().maximum_tokens = 0;
    assert!(ledger.reserve(&lease, work).is_err());
    assert_eq!(ledger.budget_snapshot(&id).unwrap().reservation_count, 0);
}

#[test]
fn budgets_sqlite_wait_does_not_authorize_an_expired_lease_from_old_clock_sample() {
    let root = Root::new();
    let mut ledger = root.open();
    let id = ledger.freeze(&parse(&standard())).unwrap();
    let lease = ledger.acquire(&id, "short-owner", 100).unwrap();
    let blocking = rusqlite::Connection::open(root.0.join("ledger/events.sqlite3")).unwrap();
    blocking.execute_batch("BEGIN IMMEDIATE").unwrap();
    let release = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(150));
        blocking.execute_batch("COMMIT").unwrap();
    });
    assert_eq!(
        ledger.reserve(&lease, request("queued")).unwrap_err().code,
        ErrorCode::StaleLease
    );
    release.join().unwrap();
    assert_eq!(ledger.budget_snapshot(&id).unwrap().reservation_count, 0);
}

#[test]
fn budgets_known_costs_and_candidate_attempts_cannot_be_rewritten_or_refunded() {
    let root = Root::new();
    let mut ledger = root.open();
    let id = ledger.freeze(&parse(&standard())).unwrap();
    let lease = ledger.acquire(&id, "owner", 300000).unwrap();
    let work = request("job");
    let handle = ledger.reserve(&lease, work.clone()).unwrap();
    let mut invalid = settlement("invalid", &work);
    invalid.usage.attempts = Some(0);
    assert!(ledger.settle(&lease, &handle, invalid).is_err());
    let known = settlement("known", &work);
    ledger.settle(&lease, &handle, known.clone()).unwrap();
    let mut changed = known;
    changed.operation = d("change-known");
    changed.usage.cpu_ms = Some(1);
    assert_eq!(
        ledger.settle(&lease, &handle, changed).unwrap_err().code,
        ErrorCode::IntegrityMismatch
    );
    assert_eq!(ledger.budget_snapshot(&id).unwrap().spent.cpu_ms, 100);
}
