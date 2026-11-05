use super::*;
#[allow(dead_code)]
mod fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../nf-core/tests/common/study_fixture.rs"
    ));
}
#[test]
fn cancellation_recovery_migrate_v2_keeps_existing_cost_holds_contracts_and_events() {
    let path = std::path::PathBuf::from(std::env::var_os("KIROCREW_SCRATCH").unwrap())
        .join(format!("worker-migration-{}", std::process::id()));
    let mut ledger = Ledger::open(&path, LedgerLimits::default()).unwrap();
    let contract =
        Validated::<StudyContract>::parse(&serde_json::to_vec(&fixture::standard()).unwrap())
            .unwrap();
    let id = ledger.freeze(&contract).unwrap();
    let lease = ledger.acquire(&id, "owner", 300000).unwrap();
    let reservation = ledger
        .reserve(
            &lease,
            budgets::Request {
                schema_version: 1,
                operation: fixture::d("v2-reservation"),
                attempt: 0,
                parent: None,
                kind: budgets::WorkKind::Metadata,
                resources: budgets::Resources {
                    cpu_ms: 100,
                    host_bytes: 1000,
                    workers: 1,
                    ..budgets::Resources::default()
                },
                wall_ms: 10000,
                external: None,
            },
        )
        .unwrap();
    let held = ledger.budget_snapshot(&id).unwrap().held;
    let event = ledger.events(&id, 0, 10).unwrap()[0].digest.clone();
    ledger
        .connection
        .execute_batch(budgets::audit_budget::STRIP_V8)
        .unwrap();
    ledger
        .connection
        .execute_batch(
            "DROP TABLE accepted_heads;DROP TABLE verified_frontier;DROP TABLE accepted_records;DROP TABLE promotion_records;DELETE FROM budget_migrations WHERE version=7;DROP TABLE final_fixture_records;DELETE FROM budget_migrations WHERE version=6;",
        )
        .unwrap();
    ledger
        .connection
        .execute_batch(super::super::exposure::STRIP_V5)
        .unwrap();
    ledger
        .connection
        .execute_batch(super::super::splits::STRIP_V4)
        .unwrap();
    ledger.connection.execute_batch("DROP TABLE worker_updates;DROP TABLE worker_intents;DROP TABLE stop_requests;DELETE FROM budget_migrations WHERE version=3;PRAGMA user_version=2;").unwrap();
    ledger
        .connection
        .execute(
            "UPDATE meta SET value=?1 WHERE key='schema'",
            [budgets::schema_v2().as_str()],
        )
        .unwrap();
    drop(ledger);
    let mut ledger = Ledger::open(&path, LedgerLimits::default()).unwrap();
    assert_eq!(ledger.budget_snapshot(&id).unwrap().held, held);
    assert_eq!(ledger.events(&id, 0, 10).unwrap()[0].digest, event);
    assert_eq!(
        ledger
            .recover_reservation(&lease, reservation.id())
            .unwrap()
            .id(),
        reservation.id()
    );
    ledger
        .request_stop(
            &id,
            fixture::d("post-migration-stop"),
            StopReason::OwnerStop,
        )
        .unwrap();
    assert!(ledger.stop_record(&id).unwrap().is_some());
    drop(ledger);
    std::fs::remove_dir_all(path).unwrap();
}
