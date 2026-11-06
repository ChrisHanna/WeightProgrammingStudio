use super::*;
#[allow(dead_code)]
mod fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../nf-core/tests/common/study_fixture.rs"
    ));
}
#[test]
fn split_broker_migrate_v3_preserves_leases_holds_stops_events_and_final_reservations() {
    let root = std::env::var_os("KIROCREW_SCRATCH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join(format!("split-migration-{}", std::process::id()));
    let mut ledger = Ledger::open(&root, LedgerLimits::default()).unwrap();
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
                operation: fixture::d("v3-hold"),
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
    for (i, pair) in stages.windows(2).enumerate() {
        ledger
            .advance(
                &lease,
                Transition {
                    operation: fixture::d(&format!("v3-step{i}")),
                    expected: pair[0],
                    next: pair[1],
                    protocol: fixture::d("protocol"),
                    evidence: fixture::d("evidence"),
                    message: "software migration fixture".into(),
                },
            )
            .unwrap();
    }
    ledger
        .request_stop(&id, fixture::d("stop"), supervision::StopReason::OwnerStop)
        .unwrap();
    let held = ledger.budget_snapshot(&id).unwrap().held;
    ledger
        .connection
        .execute_batch(budgets::audit_budget::STRIP_V8)
        .unwrap();
    let events = canonical(&ledger.events(&id, 0, 100).unwrap()).unwrap();
    let finals: i64 = ledger
        .connection
        .query_row("SELECT count(*) FROM final_reservations", [], |r| r.get(0))
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
    ledger.connection.execute_batch(STRIP_V4).unwrap();
    ledger
        .connection
        .execute_batch("PRAGMA user_version=3;")
        .unwrap();
    ledger
        .connection
        .execute(
            "UPDATE meta SET value=?1 WHERE key='schema'",
            [supervision::schema_v3().as_str()],
        )
        .unwrap();
    drop(ledger);
    let ledger = Ledger::open(&root, LedgerLimits::default()).unwrap();
    assert_eq!(ledger.budget_snapshot(&id).unwrap().held, held);
    assert_eq!(
        canonical(&ledger.events(&id, 0, 100).unwrap()).unwrap(),
        events
    );
    assert!(ledger.stop_record(&id).unwrap().is_some());
    assert_eq!(
        ledger
            .recover_reservation(&lease, reservation.id())
            .unwrap()
            .id(),
        reservation.id()
    );
    assert_eq!(
        ledger
            .connection
            .query_row("SELECT count(*) FROM final_reservations", [], |r| r
                .get::<_, i64>(0))
            .unwrap(),
        finals
    );
    assert_eq!(
        ledger
            .connection
            .query_row(
                "SELECT previous FROM budget_migrations WHERE version=4",
                [],
                |r| r.get::<_, String>(0)
            )
            .unwrap(),
        supervision::schema_v3().as_str()
    );
    assert_eq!(
        ledger
            .connection
            .query_row("PRAGMA user_version", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        8
    );
    drop(ledger);
    std::fs::remove_dir_all(root).unwrap();
}
