use super::*;
pub(super) mod e2e {
    use crate as nf_experiments;
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/promotion_atomic.rs"
    ));
    private_promotion_cases!();
}
#[path = "gated_tests.rs"]
mod gated;
#[path = "head_race_tests.rs"]
mod head_race;
#[path = "inventory_tests.rs"]
mod inventory;
#[path = "lifecycle_tests.rs"]
mod lifecycle;
#[path = "registration_tests.rs"]
mod registration;
#[path = "safety_tests.rs"]
mod safety;
#[allow(dead_code)]
mod fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../nf-core/tests/common/study_fixture.rs"
    ));
}
#[test]
fn promotion_schema_v6_migration_preserves_native_and_journal_rows() {
    let path = std::env::temp_dir().join(format!("promotion-migration-{}", std::process::id()));
    let mut ledger = Ledger::open(&path, LedgerLimits::default()).unwrap();
    let c = Validated::<StudyContract>::parse(&serde_json::to_vec(&fixture::standard()).unwrap())
        .unwrap();
    let id = ledger.freeze(&c).unwrap();
    let genesis = ledger.state(&id).unwrap().head;
    let current_version: u64 = ledger
        .connection
        .query_row("PRAGMA user_version", [], |r| r.get(0))
        .unwrap();
    // Build an actual v6 fixture: no later-version schema objects may survive.
    ledger.connection.execute_batch("DROP TABLE audit_cleanup;DROP TABLE audit_runtime_bindings;DROP TABLE execution_owners;DROP TABLE audit_storage_claims;DROP TABLE accepted_heads;DROP TABLE verified_frontier;DROP TABLE accepted_records;DROP TABLE promotion_records;DELETE FROM budget_migrations WHERE version>=7;PRAGMA user_version=6;").unwrap();
    ledger
        .connection
        .execute(
            "UPDATE meta SET value=?1 WHERE key='schema'",
            [final_fixture::schema_v6().as_str()],
        )
        .unwrap();
    drop(ledger);
    let ledger = Ledger::open(&path, LedgerLimits::default()).unwrap();
    assert_eq!(ledger.state(&id).unwrap().head, genesis);
    assert!(ledger
        .accepted_fixture_head(&c.study_family)
        .unwrap()
        .is_none());
    assert_eq!(
        ledger
            .connection
            .query_row("PRAGMA user_version", [], |r| r.get::<_, u64>(0))
            .unwrap(),
        current_version
    );
    let previous: String = ledger
        .connection
        .query_row(
            "SELECT previous FROM budget_migrations WHERE version=7",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(previous, final_fixture::schema_v6().as_str());
    drop(ledger);
    std::fs::remove_dir_all(path).unwrap();
}
/// Storage-only crash test of the PRIVATE publication primitive. No public
/// promotion credential is built from these synthetic transaction-test values.
#[test]
fn promotion_crash_rolls_back_pointer_frontier_event() {
    if let Some(path) = std::env::var_os("NF_PROMOTION_CRASH_DB") {
        let mut ledger = Ledger::open(Path::new(&path), LedgerLimits::default()).unwrap();
        let c =
            Validated::<StudyContract>::parse(&serde_json::to_vec(&fixture::standard()).unwrap())
                .unwrap();
        let id = ledger.freeze(&c).unwrap();
        let lease = ledger.acquire(&id, "crash-test", 300000).unwrap();
        let mut previous = Stage::Frozen;
        for (i, next) in [
            Stage::Preflight,
            Stage::Profile,
            Stage::Baseline,
            Stage::Design,
            Stage::Execute,
            Stage::Evaluate,
            Stage::Verify,
        ]
        .into_iter()
        .enumerate()
        {
            ledger
                .advance(
                    &lease,
                    Transition {
                        operation: fixture::d(&format!("step{i}")),
                        expected: previous,
                        next,
                        protocol: fixture::d("test"),
                        evidence: fixture::d("test"),
                        message: "storage fixture".into(),
                    },
                )
                .unwrap();
            previous = next;
        }
        let reservation = ledger
            .reserve(
                &lease,
                Request {
                    schema_version: 1,
                    operation: fixture::d("crash-publication-cost"),
                    attempt: 0,
                    parent: None,
                    kind: WorkKind::Reload,
                    resources: Resources {
                        cpu_ms: 100,
                        storage_bytes: 65536,
                        host_bytes: 1024,
                        workers: 1,
                        ..Default::default()
                    },
                    wall_ms: 10000,
                    external: None,
                },
            )
            .unwrap();
        let body = json!({"operation":fixture::d("test-publish")});
        let digest = hash(&body).unwrap();
        let _ = ledger.finish_budget_with(
            &lease,
            &reservation,
            Settlement {
                operation: fixture::d("crash-settlement"),
                outcome: OutcomeKind::Completed,
                usage: Usage::default(),
                resources_released: true,
                evidence: digest.clone(),
            },
            false,
            |tx| {
                publish(tx, &lease, &c.study_family, &None, &digest, &body, 100).unwrap();
                assert_eq!(head(tx, &c.study_family).unwrap(), Some(digest));
                assert!(budgets::native_released(tx, reservation.id()).unwrap());
                std::process::exit(19); // provisional settlement AND publication roll back
            },
        );
        unreachable!();
    }
    let path = std::env::temp_dir().join(format!("promotion-crash-{}", std::process::id()));
    let result = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "coordinator::promotion::tests::promotion_crash_rolls_back_pointer_frontier_event",
            "--nocapture",
        ])
        .env("NF_PROMOTION_CRASH_DB", &path)
        .status()
        .unwrap();
    assert_eq!(result.code(), Some(19));
    let ledger = Ledger::open(&path, LedgerLimits::default()).unwrap();
    let c = Validated::<StudyContract>::parse(&serde_json::to_vec(&fixture::standard()).unwrap())
        .unwrap();
    let id = c.digest().unwrap();
    assert!(ledger
        .accepted_fixture_head(&c.study_family)
        .unwrap()
        .is_none());
    assert_eq!(ledger.state(&id).unwrap().stage, Stage::Verify);
    assert_eq!(ledger.events(&id, 0, 100).unwrap().len(), 8);
    assert_eq!(ledger.budget_snapshot(&id).unwrap().held.workers, 1);
    assert_eq!(
        ledger
            .connection
            .query_row("SELECT count(*) FROM budget_outcomes", [], |r| r
                .get::<_, u64>(0))
            .unwrap(),
        0
    );
    for table in ["accepted_heads", "accepted_records", "verified_frontier"] {
        assert_eq!(
            ledger
                .connection
                .query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r
                    .get::<_, u64>(0))
                .unwrap(),
            0
        );
    }
    drop(ledger);
    std::fs::remove_dir_all(path).unwrap();
}
#[test]
fn promotion_records_are_immutable_and_noncanonical_evidence_is_denied() {
    assert!(decode(
        b"{\"a\": 1}",
        ContentDigest::of_bytes(b"{\"a\": 1}").as_str()
    )
    .is_err());
    assert!(decode(b"{}", ContentDigest::of_bytes(b"different").as_str()).is_err());
    let path = std::env::temp_dir().join(format!("promotion-immutable-{}", std::process::id()));
    let mut ledger = Ledger::open(&path, LedgerLimits::default()).unwrap();
    let c = Validated::<StudyContract>::parse(&serde_json::to_vec(&fixture::standard()).unwrap())
        .unwrap();
    let id = ledger.freeze(&c).unwrap();
    let tx = ledger.connection.transaction().unwrap();
    record(&tx, &id, "intent", &json!({"fixture":true})).unwrap();
    tx.commit().unwrap();
    assert!(ledger
        .connection
        .execute("UPDATE promotion_records SET body=x'00'", [])
        .is_err());
    assert!(ledger
        .connection
        .execute("DELETE FROM promotion_records", [])
        .is_err());
    assert!(ledger
        .accepted_fixture_head(&c.study_family)
        .unwrap()
        .is_none());
    drop(ledger);
    std::fs::remove_dir_all(path).unwrap();
}
