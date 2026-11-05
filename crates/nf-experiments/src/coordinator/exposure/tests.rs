use super::*;
#[allow(dead_code)]
mod fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../nf-core/tests/common/study_fixture.rs"
    ));
}
fn root(tag: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var_os("KIROCREW_SCRATCH").unwrap())
        .join(format!("exposure-{tag}-{}", std::process::id()))
}
fn setup(path: &Path) -> (Ledger, Validated<StudyContract>, ContentDigest, Lease) {
    let mut ledger = Ledger::open(path, LedgerLimits::default()).unwrap();
    let contract =
        Validated::<StudyContract>::parse(&serde_json::to_vec(&fixture::standard()).unwrap())
            .unwrap();
    let id = ledger.freeze(&contract).unwrap();
    let lease = ledger.acquire(&id, "owner", 300000).unwrap();
    (ledger, contract, id, lease)
}
fn task(contract: &StudyContract, id: &ContentDigest) -> Subject {
    let b = &contract.portfolio.bindings[0];
    Subject {
        contract: id.clone(),
        binding: b.id.clone(),
        package_version: b.gym.clone(),
        dataset_version: b.splits.development.dataset.clone(),
        task: b.splits.development.items[0].clone(),
        input: fixture::d("actual-input"),
        semantic: fixture::d("semantic-input"),
        identity_protocol: identity_protocol(),
        label: fixture::d("actual-label"),
        split: Some(SplitRole::Development),
        legacy: false,
    }
}
#[test]
fn exposure_canonical_identity_ignores_transport_renaming_not_actual_input_or_seed() {
    let a =
        serde_json::json!({"task":"old","group":"old","role":"edit","input":{"value":2},"seed":17});
    let b = serde_json::json!({"task":"renamed","group":"renamed","role":"holdout","nonce":"new","input":{"value":2},"seed":17});
    assert_eq!(identity(&a).unwrap(), identity(&b).unwrap());
    let mut c = b.clone();
    c["input"]["value"] = serde_json::json!(3);
    assert_ne!(identity(&a).unwrap(), identity(&c).unwrap());
    c = b;
    c["seed"] = serde_json::json!(18);
    assert_ne!(identity(&a).unwrap(), identity(&c).unwrap());
}
#[test]
fn exposure_actual_journal_fields_seed_retry_failure_and_foreign_ticket() {
    let path = root("journal");
    let (ledger, contract, id, lease) = setup(&path);
    let s = task(&contract, &id);
    let tx = ledger.connection.unchecked_transaction().unwrap();
    let ticket = begin(
        &tx,
        &lease,
        &contract,
        Start {
            subject: s.clone(),
            recipient: "programmer-fixture".into(),
            role: "proposer".into(),
            operation: fixture::d("first-read"),
            context: Context {
                seed: Some(17),
                model: Some(contract.source.manifest.clone()),
                ..Default::default()
            },
            fields: [Field::Input, Field::Summary].into_iter().collect(),
        },
    )
    .unwrap();
    tx.commit().unwrap();
    let complete = ledger
        .finish_exposure(
            &lease,
            &ticket,
            Stage::Completed,
            [Field::Input, Field::Summary].into_iter().collect(),
            "completed",
            Cost {
                tool_calls: Some(1),
                ..Default::default()
            },
        )
        .unwrap();
    let tx = ledger.connection.unchecked_transaction().unwrap();
    let retry = begin(
        &tx,
        &lease,
        &contract,
        Start {
            subject: s,
            recipient: "programmer-fixture".into(),
            role: "proposer".into(),
            operation: fixture::d("retry-read"),
            context: Context {
                seed: Some(17),
                attempt: 1,
                retry_of: Some(complete.id.clone()),
                ..Default::default()
            },
            fields: [Field::Input].into_iter().collect(),
        },
    )
    .unwrap();
    tx.commit().unwrap();
    ledger
        .finish_exposure(
            &lease,
            &retry,
            Stage::Failed,
            BTreeSet::new(),
            "failed",
            Cost::default(),
        )
        .unwrap();
    let rows = ledger.exposure_events(&lease, 0, 100).unwrap();
    assert_eq!(rows.len(), 6);
    assert_eq!(rows[2].body.shown.len(), 2);
    assert_eq!(rows[3].body.context.attempt, 1);
    assert_eq!(rows[3].body.context.retry_of, Some(complete.id));
    assert_eq!(rows[5].body.stage, Stage::Failed);
    assert!(rows[5].body.cost.cpu_ms.is_none());
    let other = root("foreign");
    let (foreign, _, _, fl) = setup(&other);
    assert!(foreign
        .finish_exposure(
            &fl,
            &ticket,
            Stage::Completed,
            [Field::Input].into_iter().collect(),
            "forged",
            Cost::default()
        )
        .is_err());
    assert!(ledger
        .connection
        .execute("UPDATE exposure_events SET stage='completed'", [])
        .is_err());
    drop(foreign);
    drop(ledger);
    std::fs::remove_dir_all(path).unwrap();
    std::fs::remove_dir_all(other).unwrap();
}
#[test]
fn exposure_unknown_legacy_alias_cannot_become_selection_or_holdout() {
    let path = root("alias");
    let (ledger, contract, id, lease) = setup(&path);
    let mut s = task(&contract, &id);
    s.legacy = true;
    s.split = None;
    register(&ledger.connection, &contract.study_family, &s).unwrap();
    let mut renamed = s.clone();
    renamed.task.item = fixture::d("renamed-item");
    renamed.task.group = fixture::d("renamed-group");
    renamed.input = fixture::d("rehash");
    renamed.legacy = false;
    for role in [SplitRole::Selection, SplitRole::Holdout] {
        renamed.split = Some(role);
        assert!(admit(&ledger.connection, &contract, &renamed).is_err());
    }
    renamed.split = Some(SplitRole::Development);
    assert!(admit(&ledger.connection, &contract, &renamed).is_ok());
    drop(lease);
    drop(ledger);
    std::fs::remove_dir_all(path).unwrap();
}
#[test]
fn exposure_real_resume_marks_expired_started_attempt_incomplete_without_reset() {
    let path = root("crash");
    let (mut ledger, contract, id, initial) = setup(&path);
    ledger.release(&initial).unwrap();
    let lease = ledger.acquire(&id, "crashing-owner", 100).unwrap();
    let tx = ledger.connection.unchecked_transaction().unwrap();
    begin(
        &tx,
        &lease,
        &contract,
        Start {
            subject: task(&contract, &id),
            recipient: "crashed-worker".into(),
            role: "proposer".into(),
            operation: fixture::d("crashed-access"),
            context: Context {
                seed: Some(17),
                ..Default::default()
            },
            fields: [Field::Input, Field::Label].into_iter().collect(),
        },
    )
    .unwrap();
    tx.commit().unwrap();
    drop(ledger);
    std::thread::sleep(std::time::Duration::from_millis(150));
    let mut ledger = Ledger::open(&path, LedgerLimits::default()).unwrap();
    let fresh = ledger.acquire(&id, "recovery-owner", 300000).unwrap();
    assert_eq!(ledger.reconcile_exposures(&fresh).unwrap(), 1);
    assert_eq!(ledger.reconcile_exposures(&fresh).unwrap(), 0);
    let rows = ledger.exposure_events(&fresh, 0, 100).unwrap();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[2].body.stage, Stage::Incomplete);
    assert!(rows[2].body.shown.is_empty());
    assert!(rows[2].body.cost.cpu_ms.is_none());
    assert_eq!(rows[2].body.parent, Some(rows[1].id.clone()));
    drop(ledger);
    std::fs::remove_dir_all(path).unwrap();
}
#[test]
fn exposure_migrate_v4_retains_budget_and_marks_untracked_history_unknown() {
    let path = root("migration");
    let (mut ledger, contract, id, lease) = setup(&path);
    ledger
        .reserve(
            &lease,
            budgets::Request {
                schema_version: 1,
                operation: fixture::d("old-work"),
                attempt: 0,
                parent: None,
                kind: budgets::WorkKind::Metadata,
                resources: budgets::Resources {
                    cpu_ms: 100,
                    host_bytes: 1000,
                    workers: 1,
                    ..Default::default()
                },
                wall_ms: 10000,
                external: None,
            },
        )
        .unwrap();
    let held = ledger.budget_snapshot(&id).unwrap().held;
    ledger
        .connection
        .execute_batch(budgets::audit_budget::STRIP_V8)
        .unwrap();
    let events = canonical(&ledger.events(&id, 0, 100).unwrap()).unwrap();
    ledger
        .connection
        .execute_batch(
            "DROP TABLE accepted_heads;DROP TABLE verified_frontier;DROP TABLE accepted_records;DROP TABLE promotion_records;DELETE FROM budget_migrations WHERE version=7;DROP TABLE final_fixture_records;DELETE FROM budget_migrations WHERE version=6;",
        )
        .unwrap();
    ledger.connection.execute_batch(STRIP_V5).unwrap();
    ledger
        .connection
        .execute_batch("PRAGMA user_version=4;")
        .unwrap();
    ledger
        .connection
        .execute(
            "UPDATE meta SET value=?1 WHERE key='schema'",
            [splits::schema_v4().as_str()],
        )
        .unwrap();
    drop(ledger);
    let ledger = Ledger::open(&path, LedgerLimits::default()).unwrap();
    assert_eq!(held, ledger.budget_snapshot(&id).unwrap().held);
    assert_eq!(
        events,
        canonical(&ledger.events(&id, 0, 100).unwrap()).unwrap()
    );
    let gap: bool = ledger
        .connection
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM exposure_gaps WHERE family=?1)",
            [contract.study_family.as_str()],
            |r| r.get(0),
        )
        .unwrap();
    assert!(gap);
    assert_eq!(
        ledger
            .connection
            .query_row(
                "SELECT previous FROM budget_migrations WHERE version=5",
                [],
                |r| r.get::<_, String>(0)
            )
            .unwrap(),
        splits::schema_v4().as_str()
    );
    let mut s = task(&contract, &id);
    s.split = Some(SplitRole::Holdout);
    assert!(admit(&ledger.connection, &contract, &s).is_err());
    drop(ledger);
    std::fs::remove_dir_all(path).unwrap();
}
