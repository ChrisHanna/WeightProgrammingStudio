use super::*;
#[allow(dead_code)]
mod fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../nf-core/tests/common/study_fixture.rs"
    ));
}
#[test]
fn budgets_migrate_v1_preserves_contract_journal_and_budget_origin() {
    let path = std::env::var_os("KIROCREW_SCRATCH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join(format!("budget-migration-{}", std::process::id()));
    let mut ledger = Ledger::open(&path, LedgerLimits::default()).unwrap();
    let value = fixture::standard();
    let contract = Validated::<StudyContract>::parse(&serde_json::to_vec(&value).unwrap()).unwrap();
    let id = ledger.freeze(&contract).unwrap();
    let original = ledger.events(&id, 0, 10).unwrap()[0].digest.clone();
    ledger
        .connection
        .execute_batch(audit_budget::STRIP_V8)
        .unwrap();
    ledger
        .connection
        .execute_batch(
            "DROP TABLE accepted_heads;DROP TABLE verified_frontier;DROP TABLE accepted_records;DROP TABLE promotion_records;DELETE FROM budget_migrations WHERE version=7;DROP TABLE final_fixture_records;DELETE FROM budget_migrations WHERE version=6;",
        )
        .unwrap();
    // Build an exact version-one fixture from this fresh database. There are
    // no reservation/cost rows to delete; frozen contract/event data is kept.
    ledger
        .connection
        .execute_batch(super::super::exposure::STRIP_V5)
        .unwrap();
    ledger
        .connection
        .execute_batch(super::super::splits::STRIP_V4)
        .unwrap();
    ledger.connection.execute_batch("DROP TABLE worker_updates;DROP TABLE worker_intents;DROP TABLE stop_requests;DROP TABLE budget_outcomes;DROP TABLE budget_requests;DROP TABLE budget_epochs;DROP TABLE budget_faults;DROP TABLE budget_migrations;PRAGMA user_version=1;").unwrap();
    ledger
        .connection
        .execute(
            "UPDATE meta SET value=?1 WHERE key='schema'",
            [ContentDigest::of_bytes(SCHEMA.as_bytes()).as_str()],
        )
        .unwrap();
    drop(ledger);
    let ledger = Ledger::open(&path, LedgerLimits::default()).unwrap();
    assert_eq!(ledger.contract(&id).unwrap().digest().unwrap(), id);
    assert_eq!(ledger.events(&id, 0, 10).unwrap()[0].digest, original);
    assert_eq!(
        ledger
            .family_budget_origin(&contract.study_family)
            .unwrap()
            .0,
        id
    );
    let version: i64 = ledger
        .connection
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .unwrap();
    assert_eq!(version, 8);
    assert_eq!(ledger.budget_snapshot(&id).unwrap().reservation_count, 0);
    drop(ledger);
    std::fs::remove_dir_all(path).unwrap();
}
