//! Public-opener regression tests, with exact historical DDL rather than a
//! current database whose version number has merely been changed.
use super::*;
use crate::coordinator as c;
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Historical(PathBuf);
impl Historical {
    fn new(version: usize) -> Self {
        let root = Self(std::env::temp_dir().join(format!(
            "audit-v8-migration-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )));
        let files = c::files::Files::open(&root.0).unwrap();
        let connection = Connection::open(files.sqlite_path()).unwrap();
        if version > 0 {
            connection.execute_batch(c::SCHEMA).unwrap();
            connection
                .execute(
                    "INSERT INTO meta VALUES('store',?1)",
                    [ContentDigest::of_bytes(b"historical-store").as_str()],
                )
                .unwrap();
            connection
                .execute(
                    "INSERT INTO meta VALUES('schema',?1)",
                    [ContentDigest::of_bytes(c::SCHEMA.as_bytes()).as_str()],
                )
                .unwrap();
            for (ddl, digest) in [
                (c::budgets::MIGRATION_V2, c::budgets::schema_v2()),
                (c::supervision::MIGRATION_V3, c::supervision::schema_v3()),
                (c::splits::MIGRATION_V4, c::splits::schema_v4()),
                (c::exposure::MIGRATION_V5, c::exposure::schema_v5()),
                (
                    c::final_fixture::MIGRATION_V6,
                    c::final_fixture::schema_v6(),
                ),
                (c::promotion::MIGRATION_V7, c::promotion::schema_v7()),
            ]
            .into_iter()
            .take(version - 1)
            {
                connection.execute_batch(ddl).unwrap();
                connection
                    .execute(
                        "UPDATE meta SET value=?1 WHERE key='schema'",
                        [digest.as_str()],
                    )
                    .unwrap();
            }
        }
        root
    }
    fn db(&self) -> PathBuf {
        self.0.join("events.sqlite3")
    }
    fn open(&self) -> Result<Ledger> {
        Ledger::open(&self.0, LedgerLimits::default())
    }
}
impl Drop for Historical {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn upgrade(version: usize) {
    let root = Historical::new(version);
    let ledger = root.open().unwrap();
    verify_v8(&ledger.connection).unwrap();
    let store = ledger.store.clone();
    if version > 0 {
        assert_eq!(store, ContentDigest::of_bytes(b"historical-store").as_str());
    }
    assert_eq!(
        ledger
            .connection
            .query_row("SELECT count(*) FROM budget_migrations", [], |r| r
                .get::<_, u32>(0))
            .unwrap(),
        7
    );
    drop(ledger);
    let before = fs::read(root.db()).unwrap();
    let reopened = root.open().unwrap();
    assert_eq!(reopened.store, store);
    verify_v8(&reopened.connection).unwrap();
    drop(reopened);
    assert_eq!(
        fs::read(root.db()).unwrap(),
        before,
        "v8 reopen must not rewrite the database"
    );
}
macro_rules! upgrade_test {
    ($name:ident, $version:expr) => {
        #[test]
        fn $name() {
            upgrade($version);
        }
    };
}
upgrade_test!(public_v0_initializes_v8_and_reopens_without_rewrite, 0);
upgrade_test!(public_v1_migrates_v8_and_reopens_without_rewrite, 1);
upgrade_test!(public_v2_migrates_v8_and_reopens_without_rewrite, 2);
upgrade_test!(public_v3_migrates_v8_and_reopens_without_rewrite, 3);
upgrade_test!(public_v4_migrates_v8_and_reopens_without_rewrite, 4);
upgrade_test!(public_v5_migrates_v8_and_reopens_without_rewrite, 5);
upgrade_test!(public_v6_migrates_v8_and_reopens_without_rewrite, 6);
upgrade_test!(public_v7_migrates_v8_and_reopens_without_rewrite, 7);

#[test]
fn public_migration_rejects_shadows_and_rolls_back_all_prior_versions() {
    for version in [1, 7] {
        for mutation in [
            "CREATE VIEW audit_storage_claims AS SELECT 'forged' AS request,1 AS jobs,1 AS allowance;",
            "CREATE TABLE audit_shadow(value TEXT);",
            "CREATE TABLE execution_owners(request TEXT PRIMARY KEY,backend TEXT);",
            "CREATE TRIGGER audit_shadow BEFORE UPDATE ON meta BEGIN SELECT RAISE(IGNORE); END;",
            "CREATE TABLE audit_storage_claims(request TEXT PRIMARY KEY,jobs INTEGER,allowance INTEGER);",
        ] {
            let root = Historical::new(version);
            let connection = Connection::open(root.db()).unwrap();
            connection.execute_batch(mutation).unwrap();
            drop(connection);
            let before = fs::read(root.db()).unwrap();
            assert!(root.open().is_err(), "accepted shadow: {mutation}");
            assert_eq!(fs::read(root.db()).unwrap(), before, "failed upgrade must roll back v2..v8");
        }
    }
}

#[test]
fn public_v8_rejects_forged_triggers_and_migration_provenance_without_repair() {
    for mutation in [
        "DROP TRIGGER audit_claim_no_update; CREATE TRIGGER audit_claim_no_update BEFORE INSERT ON audit_storage_claims BEGIN SELECT RAISE(IGNORE); END;",
        "CREATE TRIGGER harmless_name BEFORE INSERT ON execution_owners BEGIN SELECT RAISE(IGNORE); END;",
        "DELETE FROM budget_migrations WHERE version=8;",
        "UPDATE budget_migrations SET previous='forged' WHERE version=8;",
    ] {
        let root = Historical::new(7);
        drop(root.open().unwrap());
        let connection = Connection::open(root.db()).unwrap();
        connection.execute_batch(mutation).unwrap();
        drop(connection);
        let before = fs::read(root.db()).unwrap();
        assert!(root.open().is_err(), "accepted forged v8: {mutation}");
        assert_eq!(fs::read(root.db()).unwrap(), before, "must reject, not repair");
    }
}

#[test]
fn public_migration_rolls_back_when_predecessor_record_insert_is_ignored() {
    let root = Historical::new(7);
    let connection = Connection::open(root.db()).unwrap();
    connection.execute_batch("CREATE TRIGGER suppress_migration BEFORE INSERT ON budget_migrations WHEN NEW.version=8 BEGIN SELECT RAISE(IGNORE); END;").unwrap();
    drop(connection);
    let before = fs::read(root.db()).unwrap();
    assert!(
        root.open().is_err(),
        "suppressed provenance must never commit a v8 ledger"
    );
    assert_eq!(
        fs::read(root.db()).unwrap(),
        before,
        "late verification failure rolls back DDL and markers"
    );
}

#[test]
fn public_legacy_claim_migration_preserves_holds_and_denies_both_dispatch_backends() {
    let mut h = super::super::tests::Harness::new();
    let held = h.ledger.budget_snapshot(&h.lease.contract).unwrap();
    h.ledger
        .connection
        .execute_batch(super::super::STRIP_V8)
        .unwrap();
    h.ledger.connection.execute_batch(CLAIMS).unwrap();
    h.ledger
        .connection
        .execute(
            "INSERT INTO audit_storage_claims VALUES(?1,?2,?3)",
            params![
                h.reservation.id().as_str(),
                h.budget.max_executions,
                h.budget.storage_bytes
            ],
        )
        .unwrap();
    h.ledger
        .connection
        .execute_batch("PRAGMA user_version=7")
        .unwrap();
    h.ledger
        .connection
        .execute(
            "UPDATE meta SET value=?1 WHERE key='schema'",
            [c::promotion::schema_v7().as_str()],
        )
        .unwrap();
    h.reopen();
    verify_v8(&h.ledger.connection).unwrap();
    let owner: String = h
        .ledger
        .connection
        .query_row(
            "SELECT backend FROM execution_owners WHERE request=?1",
            [h.reservation.id().as_str()],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(owner, "legacy_audit");
    assert_eq!(
        h.ledger
            .connection
            .query_row(
                "SELECT jobs,allowance FROM audit_storage_claims WHERE request=?1",
                [h.reservation.id().as_str()],
                |r| Ok((r.get::<_, u64>(0)?, r.get::<_, u64>(1)?))
            )
            .unwrap(),
        (h.budget.max_executions, h.budget.storage_bytes)
    );
    let after = h.ledger.budget_snapshot(&h.lease.contract).unwrap();
    assert_eq!(after.held, held.held);
    assert_eq!(after.spent, held.spent);
    assert!(h
        .ledger
        .claim_audit(&h.lease, &h.reservation, &h.budget)
        .is_err());
    let tx = h.ledger.connection.transaction().unwrap();
    assert!(claim_execution(&tx, h.reservation.id(), "native").is_err());
    tx.rollback().unwrap();
    assert!(h
        .ledger
        .settle(
            &h.lease,
            &h.reservation,
            Settlement {
                operation: ContentDigest::of_bytes(b"forged-release"),
                outcome: OutcomeKind::Incomplete,
                resources_released: true,
                usage: Usage::default(),
                evidence: ContentDigest::of_bytes(b"no-cleanup"),
            }
        )
        .is_err());
    assert_eq!(
        h.ledger.budget_snapshot(&h.lease.contract).unwrap().held,
        held.held
    );
    // Real cleanup of an arbitrary qualified root is not historical inventory
    // proof. The public recovery API must explain the irreducible legacy gap.
    let runtime = crate::sandbox::QualifiedDocker::qualify(true, &h.root).unwrap();
    let error = h
        .ledger
        .recover_reference_audit(&h.lease, &h.reservation, &runtime)
        .unwrap_err();
    assert!(error
        .message
        .contains("no authenticated reservation-to-root inventory"));
    assert!(error.message.contains("separately reviewed migration"));
    h.reopen();
    assert_eq!(
        h.ledger.budget_snapshot(&h.lease.contract).unwrap().held,
        held.held
    );
    assert_eq!(
        h.ledger
            .connection
            .query_row("SELECT count(*) FROM audit_cleanup", [], |r| r
                .get::<_, u64>(0))
            .unwrap(),
        0
    );
}
