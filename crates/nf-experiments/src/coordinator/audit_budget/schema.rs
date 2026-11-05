//! Versioned audit grants. Opening code must call migrate_v8 inside its migration transaction.
use super::*;

// Keep the legacy claim shape so migration can conservatively adopt old claims.
const CLAIMS: &str = "CREATE TABLE audit_storage_claims(
 request TEXT PRIMARY KEY REFERENCES budget_requests(id), jobs INTEGER NOT NULL,
 allowance INTEGER NOT NULL);
CREATE TRIGGER audit_claim_no_update BEFORE UPDATE ON audit_storage_claims
 BEGIN SELECT RAISE(ABORT,'immutable audit claim'); END;
CREATE TRIGGER audit_claim_no_delete BEFORE DELETE ON audit_storage_claims
 BEGIN SELECT RAISE(ABORT,'retained audit claim'); END;";
const GRANTS: &str = "CREATE TABLE execution_owners(
 request TEXT PRIMARY KEY REFERENCES budget_requests(id), backend TEXT NOT NULL);
CREATE TABLE audit_runtime_bindings(
 request TEXT PRIMARY KEY REFERENCES audit_storage_claims(request), root TEXT NOT NULL);
CREATE TABLE audit_cleanup(
 request TEXT PRIMARY KEY REFERENCES audit_storage_claims(request), evidence TEXT NOT NULL);
CREATE TRIGGER execution_owner_no_update BEFORE UPDATE ON execution_owners
 BEGIN SELECT RAISE(ABORT,'immutable execution owner'); END;
CREATE TRIGGER execution_owner_no_delete BEFORE DELETE ON execution_owners
 BEGIN SELECT RAISE(ABORT,'retained execution owner'); END;
CREATE TRIGGER audit_binding_no_update BEFORE UPDATE ON audit_runtime_bindings
 BEGIN SELECT RAISE(ABORT,'immutable audit root'); END;
CREATE TRIGGER audit_binding_no_delete BEFORE DELETE ON audit_runtime_bindings
 BEGIN SELECT RAISE(ABORT,'retained audit root'); END;
CREATE TRIGGER audit_cleanup_no_update BEFORE UPDATE ON audit_cleanup
 BEGIN SELECT RAISE(ABORT,'immutable audit cleanup'); END;
CREATE TRIGGER audit_cleanup_no_delete BEFORE DELETE ON audit_cleanup
 BEGIN SELECT RAISE(ABORT,'retained audit cleanup'); END;";
const MIGRATION_TAIL: &str =
    "INSERT INTO execution_owners SELECT reservation,'native' FROM worker_intents;
INSERT INTO execution_owners SELECT request,'legacy_audit' FROM audit_storage_claims;
INSERT INTO budget_migrations VALUES(8,(SELECT value FROM meta WHERE key='schema'));
PRAGMA user_version=8;";

pub(crate) fn schema_v8() -> ContentDigest {
    ContentDigest::of_bytes(
        &[
            super::super::super::promotion::schema_v7()
                .as_str()
                .as_bytes(),
            CLAIMS.as_bytes(),
            GRANTS.as_bytes(),
            MIGRATION_TAIL.as_bytes(),
        ]
        .concat(),
    )
}

fn objects(
    connection: &Connection,
    tables: &[&str],
) -> Result<BTreeMap<String, (String, String, String)>> {
    let mut result = BTreeMap::new();
    let mut statement = connection
        .prepare("SELECT name,type,tbl_name,sql FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'")
        .map_err(sql)?;
    let rows = statement
        .query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                (
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                ),
            ))
        })
        .map_err(sql)?;
    for row in rows {
        let (name, value) = row.map_err(sql)?;
        if tables.contains(&value.1.as_str())
            || name.starts_with("audit_")
            || name.starts_with("execution_owner")
        {
            result.insert(name, value);
        }
    }
    Ok(result)
}
fn verify_objects(connection: &Connection, grants: bool) -> Result<()> {
    let expected = Connection::open_in_memory().map_err(sql)?;
    expected.execute_batch(CLAIMS).map_err(sql)?;
    if grants {
        expected.execute_batch(GRANTS).map_err(sql)?;
    }
    let tables = [
        "audit_storage_claims",
        "execution_owners",
        "audit_runtime_bindings",
        "audit_cleanup",
    ];
    if objects(connection, &tables)? != objects(&expected, &tables)? {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "audit grant schema objects differ",
        ));
    }
    Ok(())
}

/// Explicit format change, never invoked from dispatch/settlement. Legacy
/// malformed objects and double dispatch fail migration rather than erase holds.
pub(crate) fn migrate_v8(tx: &Transaction<'_>) -> Result<()> {
    let version: u32 = tx
        .query_row("PRAGMA user_version", [], |r| r.get(0))
        .map_err(sql)?;
    let marker: String = tx
        .query_row("SELECT value FROM meta WHERE key='schema'", [], |r| {
            r.get(0)
        })
        .map_err(sql)?;
    if version != 7 || marker != super::super::super::promotion::schema_v7().as_str() {
        return Err(failure(
            ErrorCode::UnsupportedVersion,
            "audit migration requires verified version 7",
        ));
    }
    let legacy: bool = tx
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_schema WHERE name='audit_storage_claims')",
            [],
            |r| r.get(0),
        )
        .map_err(sql)?;
    if legacy {
        verify_objects(tx, false)?;
    } else {
        // Reject shadow names even when the legacy table itself is absent.
        if !objects(
            tx,
            &[
                "audit_storage_claims",
                "execution_owners",
                "audit_runtime_bindings",
                "audit_cleanup",
            ],
        )?
        .is_empty()
        {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "shadow audit migration objects",
            ));
        }
        tx.execute_batch(CLAIMS).map_err(sql)?;
    }
    tx.execute_batch(GRANTS).map_err(sql)?;
    tx.execute_batch(MIGRATION_TAIL).map_err(sql)?;
    if tx
        .execute(
            "UPDATE meta SET value=?1 WHERE key='schema'",
            [schema_v8().as_str()],
        )
        .map_err(sql)?
        != 1
    {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "audit schema marker absent",
        ));
    }
    verify_v8(tx)
}

pub(crate) fn verify_v8(connection: &Connection) -> Result<()> {
    let version: u32 = connection
        .query_row("PRAGMA user_version", [], |r| r.get(0))
        .map_err(sql)?;
    let marker: String = connection
        .query_row("SELECT value FROM meta WHERE key='schema'", [], |r| {
            r.get(0)
        })
        .map_err(sql)?;
    if version != 8 || marker != schema_v8().as_str() {
        return Err(failure(
            ErrorCode::UnsupportedVersion,
            "version 8 audit migration required",
        ));
    }
    verify_objects(connection, true)?;
    let previous: Option<String> = connection
        .query_row(
            "SELECT previous FROM budget_migrations WHERE version=8",
            [],
            |r| r.get(0),
        )
        .optional()
        .map_err(sql)?;
    if previous.as_deref() != Some(super::super::super::promotion::schema_v7().as_str()) {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "audit migration predecessor differs",
        ));
    }
    Ok(())
}

pub(crate) fn claim_execution(
    tx: &Transaction<'_>,
    request: &ContentDigest,
    backend: &str,
) -> Result<()> {
    verify_v8(tx)?;
    let exists: bool = tx
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM execution_owners WHERE request=?1)",
            [request.as_str()],
            |r| r.get(0),
        )
        .map_err(sql)?;
    if exists {
        return Err(failure(
            ErrorCode::IdempotencyConflict,
            "reservation already has an execution owner",
        ));
    }
    if tx
        .execute(
            "INSERT INTO execution_owners VALUES(?1,?2)",
            params![request.as_str(), backend],
        )
        .map_err(sql)?
        != 1
    {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "execution owner insert ignored",
        ));
    }
    let actual: String = tx
        .query_row(
            "SELECT backend FROM execution_owners WHERE request=?1",
            [request.as_str()],
            |r| r.get(0),
        )
        .map_err(sql)?;
    if actual != backend {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "execution owner readback differs",
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "migration_tests.rs"]
mod migration_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coordinator as c;
    fn legacy() -> Connection {
        let connection = Connection::open_in_memory().unwrap();
        connection.execute_batch(c::SCHEMA).unwrap();
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
        ] {
            connection.execute_batch(ddl).unwrap();
            connection
                .execute(
                    "UPDATE meta SET value=?1 WHERE key='schema'",
                    [digest.as_str()],
                )
                .unwrap();
        }
        connection
    }
    #[test]
    fn migration_is_versioned_verified_and_rejects_shadow_objects() {
        for malformed in [false, true] {
            let mut connection = legacy();
            connection.execute_batch(CLAIMS).unwrap();
            if malformed {
                connection.execute_batch("DROP TRIGGER audit_claim_no_update; CREATE TRIGGER audit_claim_no_update BEFORE INSERT ON audit_storage_claims BEGIN SELECT RAISE(IGNORE); END;").unwrap();
            }
            let tx = connection
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .unwrap();
            let result = migrate_v8(&tx);
            if malformed {
                assert!(result.is_err());
                tx.rollback().unwrap();
                assert_eq!(
                    connection
                        .query_row("PRAGMA user_version", [], |r| r.get::<_, u32>(0))
                        .unwrap(),
                    7
                );
            } else {
                result.unwrap();
                tx.commit().unwrap();
                verify_v8(&connection).unwrap();
                assert_eq!(
                    connection
                        .query_row(
                            "SELECT previous FROM budget_migrations WHERE version=8",
                            [],
                            |r| r.get::<_, String>(0)
                        )
                        .unwrap(),
                    c::promotion::schema_v7().as_str()
                );
                let tx = connection.transaction().unwrap();
                assert!(
                    migrate_v8(&tx).is_err(),
                    "migration is not a runtime IF NOT EXISTS repair"
                );
            }
        }
    }
    #[test]
    fn migration_rejects_legacy_overlapping_native_and_audit_dispatch() {
        let mut connection = legacy();
        connection.execute_batch(CLAIMS).unwrap();
        // Migration cannot invent a safe owner for a historically double-used
        // reservation. Source row bodies are immaterial to that unique-key test.
        connection.execute("INSERT INTO budget_requests VALUES('same','family','contract','operation',0,NULL,x'00','content')", []).unwrap();
        connection
            .execute("INSERT INTO audit_storage_claims VALUES('same',2,1)", [])
            .unwrap();
        connection
            .execute(
                "INSERT INTO worker_intents VALUES('worker','contract','same',x'00','content')",
                [],
            )
            .unwrap();
        let tx = connection.transaction().unwrap();
        assert!(migrate_v8(&tx).is_err());
        tx.rollback().unwrap();
        assert_eq!(
            connection
                .query_row("SELECT count(*) FROM audit_storage_claims", [], |r| r
                    .get::<_, u32>(0))
                .unwrap(),
            1
        );
        assert_eq!(
            connection
                .query_row("PRAGMA user_version", [], |r| r.get::<_, u32>(0))
                .unwrap(),
            7
        );
    }
}
