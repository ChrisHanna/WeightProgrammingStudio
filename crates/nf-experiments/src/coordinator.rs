//! Immutable contract/event storage, fenced workflow state and fixture-only promotion.
pub mod acceptance;
pub mod broker;
pub mod budgets;
mod committed_status;
pub mod exposure;
mod files;
pub mod final_fixture;
pub mod promotion;
pub mod scientific_loop;
pub mod splits;
pub mod supervision;
use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use nf_core::error::{CoreError, ErrorCode, Result};
use nf_core::schemas::{Validate, Validated};
use nf_core::study::{ExecutionLimits as StudyLimits, StudyContract};
use rusqlite::{
    params, Connection, OpenFlags, OptionalExtension, Transaction, TransactionBehavior,
};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;
fn failure(code: ErrorCode, message: &str) -> CoreError {
    CoreError::new(code, "", message)
}
fn sql(error: rusqlite::Error) -> CoreError {
    failure(ErrorCode::Io, &format!("SQLite: {error}"))
}
fn canonical<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    // Bound serialization itself, not just parsing the already allocated buffer.
    struct Bounded(Vec<u8>);
    impl std::io::Write for Bounded {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if bytes.len() > Limits::default().max_bytes.saturating_sub(self.0.len()) {
                return Err(std::io::Error::other("canonical record byte limit"));
            }
            self.0.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut buffer = Bounded(Vec::new());
    serde_json::to_writer(&mut buffer, value)
        .map_err(|error| failure(ErrorCode::LimitExceeded, &error.to_string()))?;
    let raw = buffer.0;
    Ok(CanonicalDocument::parse(&raw, Limits::default())?
        .bytes()
        .to_vec())
}
const SCHEMA: &str = r#"
CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE families(family TEXT PRIMARY KEY,latest TEXT NOT NULL,revision INTEGER NOT NULL,fence INTEGER NOT NULL DEFAULT 0);
CREATE TABLE family_budget(family TEXT PRIMARY KEY,origin TEXT NOT NULL,limits BLOB NOT NULL);
CREATE TABLE contracts(digest TEXT PRIMARY KEY,family TEXT NOT NULL,revision INTEGER NOT NULL,parent TEXT,canonical BLOB NOT NULL,UNIQUE(family,revision));
CREATE TABLE runs(contract TEXT PRIMARY KEY REFERENCES contracts(digest),stage TEXT NOT NULL,sequence INTEGER NOT NULL,head TEXT NOT NULL,owner TEXT,epoch INTEGER NOT NULL DEFAULT 0,boot TEXT,deadline INTEGER NOT NULL DEFAULT 0);
CREATE TABLE events(contract TEXT NOT NULL REFERENCES contracts(digest),sequence INTEGER NOT NULL,operation TEXT NOT NULL,body BLOB NOT NULL,digest TEXT NOT NULL,PRIMARY KEY(contract,sequence),UNIQUE(contract,operation));
CREATE TABLE final_reservations(family TEXT NOT NULL,item TEXT NOT NULL,task_group TEXT NOT NULL,contract TEXT NOT NULL,PRIMARY KEY(family,item,task_group));
CREATE TRIGGER contracts_no_update BEFORE UPDATE ON contracts BEGIN SELECT RAISE(ABORT,'immutable contracts');END;
CREATE TRIGGER contracts_no_delete BEFORE DELETE ON contracts BEGIN SELECT RAISE(ABORT,'immutable contracts');END;
CREATE TRIGGER events_no_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT,'immutable events');END;
CREATE TRIGGER events_no_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT,'immutable events');END;
CREATE TRIGGER budget_no_update BEFORE UPDATE ON family_budget BEGIN SELECT RAISE(ABORT,'immutable family budget origin');END;
CREATE TRIGGER budget_no_delete BEFORE DELETE ON family_budget BEGIN SELECT RAISE(ABORT,'immutable family budget origin');END;
CREATE TRIGGER reservations_no_update BEFORE UPDATE ON final_reservations BEGIN SELECT RAISE(ABORT,'retained final reservations');END;
CREATE TRIGGER reservations_no_delete BEFORE DELETE ON final_reservations BEGIN SELECT RAISE(ABORT,'retained final reservations');END;
PRAGMA user_version=1;
"#;
#[derive(Debug, Clone, Copy)]
pub struct LedgerLimits {
    pub max_database_bytes: u64,
    pub max_contracts: u32,
    pub max_events: u32,
    pub busy_timeout_ms: u64,
}
impl Default for LedgerLimits {
    fn default() -> Self {
        Self {
            max_database_bytes: 67_108_864,
            max_contracts: 1024,
            max_events: 100_000,
            busy_timeout_ms: 2000,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    Frozen,
    Preflight,
    Profile,
    Baseline,
    Diagnose,
    Design,
    Execute,
    Evaluate,
    Search,
    Verify,
    Learn,
    Completed,
    Failed,
    Unsupported,
    Incomplete,
    Cancelled,
}
impl Stage {
    fn text(self) -> &'static str {
        match self {
            Self::Frozen => "frozen",
            Self::Preflight => "preflight",
            Self::Profile => "profile",
            Self::Baseline => "baseline",
            Self::Diagnose => "diagnose",
            Self::Design => "design",
            Self::Execute => "execute",
            Self::Evaluate => "evaluate",
            Self::Search => "search",
            Self::Verify => "verify",
            Self::Learn => "learn",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Unsupported => "unsupported",
            Self::Incomplete => "incomplete",
            Self::Cancelled => "cancelled",
        }
    }
    fn parse(text: &str) -> Result<Self> {
        serde_json::from_value(serde_json::Value::String(text.to_owned())).map_err(|_| {
            failure(
                ErrorCode::IntegrityMismatch,
                "unknown persisted workflow stage",
            )
        })
    }
    pub fn terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Unsupported | Self::Incomplete | Self::Cancelled
        )
    }
    fn permits(self, next: Self) -> bool {
        if self.terminal() {
            return false;
        }
        if matches!(
            next,
            Self::Failed | Self::Unsupported | Self::Incomplete | Self::Cancelled
        ) {
            return true;
        }
        matches!(
            (self, next),
            (Self::Frozen, Self::Preflight)
                | (Self::Preflight, Self::Profile)
                | (Self::Profile, Self::Baseline)
                | (
                    Self::Baseline,
                    Self::Diagnose | Self::Design | Self::Profile
                )
                | (
                    Self::Diagnose,
                    Self::Design | Self::Profile | Self::Baseline
                )
                | (
                    Self::Design,
                    Self::Execute | Self::Profile | Self::Baseline | Self::Diagnose
                )
                | (Self::Execute, Self::Evaluate)
                | (
                    Self::Evaluate,
                    Self::Search
                        | Self::Verify
                        | Self::Design
                        | Self::Profile
                        | Self::Baseline
                        | Self::Diagnose
                )
                | (
                    Self::Search,
                    Self::Design | Self::Diagnose | Self::Profile | Self::Baseline | Self::Verify
                )
                | (Self::Verify, Self::Learn | Self::Completed)
                | (Self::Learn, Self::Completed)
        )
    }
}
#[derive(Debug, Clone)]
pub struct Lease {
    store: String,
    contract: ContentDigest,
    owner: String,
    epoch: i64,
    boot: String,
    deadline: i64,
}
impl Lease {
    pub fn fencing_token(&self) -> i64 {
        self.epoch
    }
    pub fn contract(&self) -> &ContentDigest {
        &self.contract
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Transition {
    pub operation: ContentDigest,
    pub expected: Stage,
    pub next: Stage,
    pub protocol: ContentDigest,
    pub evidence: ContentDigest,
    pub message: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventBody {
    pub schema_version: u32,
    pub contract: ContentDigest,
    pub sequence: u64,
    pub previous: ContentDigest,
    pub epoch: i64,
    pub owner: String,
    pub boot: String,
    pub monotonic_ms: i64,
    pub wall_ms: i64,
    pub transition: Transition,
}
#[derive(Debug, Clone, Serialize)]
pub struct Event {
    pub digest: ContentDigest,
    pub body: EventBody,
}
#[derive(Debug, Clone, Serialize)]
pub struct RunState {
    pub contract: ContentDigest,
    pub stage: Stage,
    pub sequence: u64,
    pub head: ContentDigest,
    pub fencing_token: i64,
}
pub struct Ledger {
    connection: Connection,
    files: files::Files,
    limits: LedgerLimits,
    store: String,
}
impl Ledger {
    pub fn open(path: &Path, limits: LedgerLimits) -> Result<Self> {
        if limits.max_database_bytes < 65536
            || limits.max_database_bytes > 1_073_741_824
            || limits.max_contracts == 0
            || limits.max_contracts > 4096
            || limits.max_events == 0
            || limits.max_events > 1_000_000
            || limits.busy_timeout_ms == 0
            || limits.busy_timeout_ms > 30_000
        {
            return Err(failure(
                ErrorCode::InvalidValue,
                "bounded SQLite ledger limits required",
            ));
        }
        let files = files::Files::open(path)?;
        files.verify(limits.max_database_bytes)?;
        let mut connection = Connection::open_with_flags(
            files.sqlite_path(),
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_CREATE
                | OpenFlags::SQLITE_OPEN_FULL_MUTEX
                | OpenFlags::SQLITE_OPEN_NOFOLLOW,
        )
        .map_err(sql)?;
        connection
            .busy_timeout(Duration::from_millis(limits.busy_timeout_ms))
            .map_err(sql)?;
        connection
            .set_limit(rusqlite::limits::Limit::SQLITE_LIMIT_LENGTH, 2_097_152)
            .map_err(sql)?;
        connection
            .set_limit(rusqlite::limits::Limit::SQLITE_LIMIT_SQL_LENGTH, 65536)
            .map_err(sql)?;
        connection
            .set_limit(rusqlite::limits::Limit::SQLITE_LIMIT_ATTACHED, 0)
            .map_err(sql)?;
        let format: i64 = connection
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .map_err(sql)?;
        if !matches!(format, 0..=8) {
            return Err(failure(
                ErrorCode::UnsupportedVersion,
                "unsupported database format",
            ));
        }
        if format == 0 {
            let tables: i64 = connection
                .query_row(
                    "SELECT count(*) FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'",
                    [],
                    |row| row.get(0),
                )
                .map_err(sql)?;
            if tables != 0 {
                return Err(failure(
                    ErrorCode::UnsupportedVersion,
                    "foreign database cannot be initialized",
                ));
            }
        }
        connection.execute_batch("PRAGMA trusted_schema=OFF;PRAGMA foreign_keys=ON;PRAGMA journal_mode=DELETE;PRAGMA synchronous=FULL;PRAGMA temp_store=MEMORY;PRAGMA cache_size=-1024;").map_err(sql)?;
        let pages = limits.max_database_bytes / 4096;
        let actual: i64 = connection
            .query_row(&format!("PRAGMA max_page_count={pages}"), [], |row| {
                row.get(0)
            })
            .map_err(sql)?;
        let page_size: i64 = connection
            .query_row("PRAGMA page_size", [], |row| row.get(0))
            .map_err(sql)?;
        if actual as u64 * page_size as u64 > limits.max_database_bytes {
            return Err(failure(
                ErrorCode::QuotaExceeded,
                "existing database exceeds page bound",
            ));
        }
        let tx = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sql)?;
        let version: i64 = tx
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .map_err(sql)?;
        if version == 0 {
            let tables: i64 = tx
                .query_row(
                    "SELECT count(*) FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'",
                    [],
                    |row| row.get(0),
                )
                .map_err(sql)?;
            if tables != 0 {
                return Err(failure(
                    ErrorCode::UnsupportedVersion,
                    "foreign database cannot be initialized as a ledger",
                ));
            }
            tx.execute_batch(SCHEMA).map_err(sql)?;
            let mut random = [0u8; 32];
            use std::io::Read;
            std::fs::File::open("/dev/urandom")
                .and_then(|mut file| file.read_exact(&mut random))
                .map_err(|error| failure(ErrorCode::Io, &error.to_string()))?;
            tx.execute(
                "INSERT INTO meta VALUES('store',?1)",
                [ContentDigest::of_bytes(&random).as_str()],
            )
            .map_err(sql)?;
            tx.execute(
                "INSERT INTO meta VALUES('schema',?1)",
                [ContentDigest::of_bytes(SCHEMA.as_bytes()).as_str()],
            )
            .map_err(sql)?;
        } else if !matches!(version, 1..=8) {
            return Err(failure(
                ErrorCode::UnsupportedVersion,
                "unsupported ledger schema; no automatic reinterpretation",
            ));
        }
        let schema: String = tx
            .query_row("SELECT value FROM meta WHERE key='schema'", [], |row| {
                row.get(0)
            })
            .map_err(sql)?;
        let expected = if version == 8 {
            budgets::audit_budget::schema_v8()
        } else if version == 7 {
            promotion::schema_v7()
        } else if version == 6 {
            final_fixture::schema_v6()
        } else if version == 5 {
            exposure::schema_v5()
        } else if version == 4 {
            splits::schema_v4()
        } else if version == 3 {
            supervision::schema_v3()
        } else if version == 2 {
            budgets::schema_v2()
        } else {
            ContentDigest::of_bytes(SCHEMA.as_bytes())
        };
        if schema != expected.as_str() {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "ledger schema binding differs",
            ));
        }
        if version < 2 {
            tx.execute_batch(budgets::MIGRATION_V2).map_err(sql)?;
            tx.execute(
                "UPDATE meta SET value=?1 WHERE key='schema'",
                [budgets::schema_v2().as_str()],
            )
            .map_err(sql)?;
        }
        if version < 3 {
            tx.execute_batch(supervision::MIGRATION_V3).map_err(sql)?;
            tx.execute(
                "UPDATE meta SET value=?1 WHERE key='schema'",
                [supervision::schema_v3().as_str()],
            )
            .map_err(sql)?;
        }
        if version < 4 {
            tx.execute_batch(splits::MIGRATION_V4).map_err(sql)?;
            tx.execute(
                "UPDATE meta SET value=?1 WHERE key='schema'",
                [splits::schema_v4().as_str()],
            )
            .map_err(sql)?;
        }
        if version < 5 {
            tx.execute_batch(exposure::MIGRATION_V5).map_err(sql)?;
            tx.execute(
                "UPDATE meta SET value=?1 WHERE key='schema'",
                [exposure::schema_v5().as_str()],
            )
            .map_err(sql)?;
        }
        if version < 6 {
            tx.execute_batch(final_fixture::MIGRATION_V6).map_err(sql)?;
            tx.execute(
                "UPDATE meta SET value=?1 WHERE key='schema'",
                [final_fixture::schema_v6().as_str()],
            )
            .map_err(sql)?;
        }
        if version < 7 {
            tx.execute_batch(promotion::MIGRATION_V7).map_err(sql)?;
            tx.execute(
                "UPDATE meta SET value=?1 WHERE key='schema'",
                [promotion::schema_v7().as_str()],
            )
            .map_err(sql)?;
        }
        if version < 8 {
            budgets::audit_budget::migrate_v8(&tx)?;
        }
        budgets::audit_budget::verify_v8(&tx)?;
        let store = tx
            .query_row("SELECT value FROM meta WHERE key='store'", [], |row| {
                row.get(0)
            })
            .map_err(sql)?;
        tx.commit().map_err(sql)?;
        files.verify(limits.max_database_bytes)?;
        Ok(Self {
            connection,
            files,
            limits,
            store,
        })
    }
    pub fn freeze(&mut self, contract: &Validated<StudyContract>) -> Result<ContentDigest> {
        contract.validate()?;
        let body = canonical(&**contract)?;
        let id = ContentDigest::of_bytes(&body);
        self.files.verify(self.limits.max_database_bytes)?;
        let now = files::moment()?;
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sql)?;
        if let Some(old) = tx
            .query_row(
                "SELECT canonical FROM contracts WHERE digest=?1",
                [id.as_str()],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .optional()
            .map_err(sql)?
        {
            if old != body {
                return Err(failure(
                    ErrorCode::IntegrityMismatch,
                    "frozen identity collision/corruption",
                ));
            }
            return Ok(id);
        }
        let count: i64 = tx
            .query_row("SELECT count(*) FROM contracts", [], |row| row.get(0))
            .map_err(sql)?;
        if count >= self.limits.max_contracts as i64 {
            return Err(failure(
                ErrorCode::QuotaExceeded,
                "contract registry bound exceeded",
            ));
        }
        let prior: Option<(String, i64)> = tx
            .query_row(
                "SELECT latest,revision FROM families WHERE family=?1",
                [contract.study_family.as_str()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .map_err(sql)?;
        match prior {
            None => {
                if contract.revision != 1 || contract.parent_contract.is_some() {
                    return Err(failure(
                        ErrorCode::RevisionConflict,
                        "family must start at revision one",
                    ));
                }
                tx.execute(
                    "INSERT INTO families(family,latest,revision) VALUES(?1,?2,1)",
                    params![contract.study_family.as_str(), id.as_str()],
                )
                .map_err(sql)?;
                tx.execute(
                    "INSERT INTO family_budget VALUES(?1,?2,?3)",
                    params![
                        contract.study_family.as_str(),
                        id.as_str(),
                        canonical(&contract.limits)?
                    ],
                )
                .map_err(sql)?;
            }
            Some((parent, revision)) => {
                if contract.revision as i64 != revision + 1
                    || contract.parent_contract.as_ref().map(ContentDigest::as_str)
                        != Some(parent.as_str())
                {
                    return Err(failure(
                        ErrorCode::RevisionConflict,
                        "revision must extend the current exact family head",
                    ));
                }
                let (owner, boot, deadline): (Option<String>, Option<String>, i64) = tx
                    .query_row(
                        "SELECT owner,boot,deadline FROM runs WHERE contract=?1",
                        [&parent],
                        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                    )
                    .map_err(sql)?;
                if owner.is_some()
                    && boot.as_deref() == Some(now.boot.as_str())
                    && deadline > now.ms
                {
                    return Err(failure(
                        ErrorCode::LeaseConflict,
                        "live coordinator must release before revision freeze",
                    ));
                }
                let previous = load_contract(&tx, &parent.parse()?)?;
                if contract.approval_receipt == previous.approval_receipt {
                    return Err(failure(
                        ErrorCode::RevisionConflict,
                        "changed revision requires a distinct exact approval receipt",
                    ));
                }
                if !within(&contract.limits, &previous.limits) {
                    return Err(failure(
                        ErrorCode::PermissionDenied,
                        "revisions cannot reset or increase inherited family limits",
                    ));
                }
                for binding in &contract.portfolio.bindings {
                    for item in &binding.splits.holdout.items {
                        let used:bool=tx.query_row("SELECT EXISTS(SELECT 1 FROM final_reservations WHERE family=?1 AND (item=?2 OR task_group=?3))",params![contract.study_family.as_str(),item.item.as_str(),item.group.as_str()],|row|row.get(0)).map_err(sql)?;
                        if used {
                            return Err(failure(ErrorCode::PermissionDenied,"reserved prior final holdout cannot become a fresh revision holdout"));
                        }
                    }
                }
                tx.execute(
                    "UPDATE families SET latest=?1,revision=?2 WHERE family=?3",
                    params![
                        id.as_str(),
                        contract.revision,
                        contract.study_family.as_str()
                    ],
                )
                .map_err(sql)?;
            }
        }
        tx.execute(
            "INSERT INTO contracts VALUES(?1,?2,?3,?4,?5)",
            params![
                id.as_str(),
                contract.study_family.as_str(),
                contract.revision,
                contract.parent_contract.as_ref().map(ContentDigest::as_str),
                body
            ],
        )
        .map_err(sql)?;
        let events: i64 = tx
            .query_row("SELECT count(*) FROM events", [], |row| row.get(0))
            .map_err(sql)?;
        if events >= self.limits.max_events as i64 {
            return Err(failure(
                ErrorCode::QuotaExceeded,
                "event registry bound exceeded",
            ));
        }
        let initial = ContentDigest::of_bytes(b"nf.coordinator.event.genesis.v1");
        let event = EventBody {
            schema_version: 1,
            contract: id.clone(),
            sequence: 1,
            previous: initial,
            epoch: 0,
            owner: "freeze".to_owned(),
            boot: now.boot.clone(),
            monotonic_ms: now.ms,
            wall_ms: now.wall_ms,
            transition: Transition {
                operation: id.clone(),
                expected: Stage::Frozen,
                next: Stage::Frozen,
                protocol: ContentDigest::of_bytes(b"nf.freeze.v1"),
                evidence: contract.approval_receipt.clone(),
                message: "contract frozen; authority/artifact preflight pending".to_owned(),
            },
        };
        let event_bytes = canonical(&event)?;
        let head = ContentDigest::of_bytes(&event_bytes);
        tx.execute(
            "INSERT INTO runs(contract,stage,sequence,head) VALUES(?1,'frozen',1,?2)",
            params![id.as_str(), head.as_str()],
        )
        .map_err(sql)?;
        tx.execute(
            "INSERT INTO events VALUES(?1,1,?1,?2,?3)",
            params![id.as_str(), event_bytes, head.as_str()],
        )
        .map_err(sql)?;
        tx.commit().map_err(sql)?;
        self.files.verify(self.limits.max_database_bytes)?;
        Ok(id)
    }
    pub fn contract(&self, id: &ContentDigest) -> Result<Validated<StudyContract>> {
        self.files.verify(self.limits.max_database_bytes)?;
        load_contract(&self.connection, id)
    }
    pub fn state(&self, id: &ContentDigest) -> Result<RunState> {
        self.files.verify(self.limits.max_database_bytes)?;
        let tx = self.connection.unchecked_transaction().map_err(sql)?;
        load_contract(&tx, id)?;
        let result = state(&tx, id)?;
        tx.commit().map_err(sql)?;
        self.files.verify(self.limits.max_database_bytes)?;
        Ok(result)
    }
    pub fn acquire(&mut self, id: &ContentDigest, owner: &str, ttl_ms: u64) -> Result<Lease> {
        if owner.trim().is_empty() || owner.len() > 256 || ttl_ms == 0 || ttl_ms > 300_000 {
            return Err(failure(
                ErrorCode::InvalidValue,
                "bounded owner and lease TTL required",
            ));
        }
        self.files.verify(self.limits.max_database_bytes)?;
        let now = files::moment()?;
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sql)?;
        let contract = load_contract(&tx, id)?;
        latest(&tx, &contract, id)?;
        let run = state(&tx, id)?;
        if run.stage.terminal() {
            return Err(failure(
                ErrorCode::InvalidTransition,
                "terminal run cannot acquire coordinator",
            ));
        }
        let (old, boot, deadline): (Option<String>, Option<String>, i64) = tx
            .query_row(
                "SELECT owner,boot,deadline FROM runs WHERE contract=?1",
                [id.as_str()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .map_err(sql)?;
        if old.is_some() && boot.as_deref() == Some(now.boot.as_str()) && deadline > now.ms {
            return Err(failure(
                ErrorCode::LeaseConflict,
                "coordinator lease already active",
            ));
        }
        let epoch:i64=tx.query_row("UPDATE families SET fence=fence+1 WHERE family=?1 AND fence<9223372036854775807 RETURNING fence",[contract.study_family.as_str()],|row|row.get(0)).map_err(sql)?;
        let deadline = now
            .ms
            .checked_add(ttl_ms as i64)
            .ok_or_else(|| failure(ErrorCode::LimitExceeded, "lease clock overflow"))?;
        tx.execute(
            "UPDATE runs SET owner=?1,epoch=?2,boot=?3,deadline=?4 WHERE contract=?5",
            params![owner, epoch, now.boot, deadline, id.as_str()],
        )
        .map_err(sql)?;
        let current = files::moment()?;
        if current.boot != now.boot || current.ms >= deadline {
            return Err(failure(
                ErrorCode::StaleLease,
                "lease expired before acquisition could commit",
            ));
        }
        tx.commit().map_err(sql)?;
        Ok(Lease {
            store: self.store.clone(),
            contract: id.clone(),
            owner: owner.to_owned(),
            epoch,
            boot: now.boot,
            deadline,
        })
    }
    pub fn renew(&mut self, lease: &Lease, ttl_ms: u64) -> Result<Lease> {
        if ttl_ms == 0 || ttl_ms > 300_000 {
            return Err(failure(
                ErrorCode::InvalidValue,
                "bounded lease TTL required",
            ));
        }
        self.files.verify(self.limits.max_database_bytes)?;
        let now = files::moment()?;
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sql)?;
        check_lease(&tx, &self.store, lease, &now)?;
        let deadline = now
            .ms
            .checked_add(ttl_ms as i64)
            .ok_or_else(|| failure(ErrorCode::LimitExceeded, "lease clock overflow"))?;
        tx.execute(
            "UPDATE runs SET deadline=?1 WHERE contract=?2",
            params![deadline, lease.contract.as_str()],
        )
        .map_err(sql)?;
        ensure_live(lease)?;
        tx.commit().map_err(sql)?;
        Ok(Lease {
            deadline,
            ..lease.clone()
        })
    }
    pub fn release(&mut self, lease: &Lease) -> Result<()> {
        self.files.verify(self.limits.max_database_bytes)?;
        let now = files::moment()?;
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sql)?;
        check_lease(&tx, &self.store, lease, &now)?;
        tx.execute(
            "UPDATE runs SET owner=NULL,deadline=0 WHERE contract=?1",
            [lease.contract.as_str()],
        )
        .map_err(sql)?;
        ensure_live(lease)?;
        tx.commit().map_err(sql)?;
        Ok(())
    }
    pub fn advance(&mut self, lease: &Lease, request: Transition) -> Result<Event> {
        if request.message.len() > 4096 {
            return Err(failure(
                ErrorCode::LimitExceeded,
                "event message exceeds bound",
            ));
        }
        self.files.verify(self.limits.max_database_bytes)?;
        let now = files::moment()?;
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sql)?;
        let contract = check_lease(&tx, &self.store, lease, &now)?;
        if let Some(event) = find_event(&tx, &lease.contract, &request.operation)? {
            if canonical(&event.body.transition)? != canonical(&request)? {
                return Err(failure(
                    ErrorCode::IdempotencyConflict,
                    "operation ID already committed with different input",
                ));
            }
            return Ok(event);
        }
        let current = state(&tx, &lease.contract)?;
        if supervision::stopped(&tx, &lease.contract)? {
            if !matches!(request.next, Stage::Incomplete | Stage::Cancelled) {
                return Err(failure(
                    ErrorCode::PermissionDenied,
                    "persisted stop prevents workflow progress/success",
                ));
            }
            if request.next == Stage::Cancelled
                && !supervision::all_cleanup_confirmed(&tx, &lease.contract)?
            {
                return Err(failure(
                    ErrorCode::PermissionDenied,
                    "unconfirmed worker cleanup must remain incomplete",
                ));
            }
        }
        if matches!(request.next, Stage::Completed | Stage::Cancelled)
            && (!supervision::all_cleanup_confirmed(&tx, &lease.contract)?
                || !budgets::contract_clean(&tx, &lease.contract)?)
        {
            return Err(failure(
                ErrorCode::PermissionDenied,
                "unfinished workers or accounting cannot become a completed/cancelled procedure",
            ));
        }
        if current.stage != request.expected || !current.stage.permits(request.next) {
            return Err(failure(
                ErrorCode::InvalidTransition,
                "stale/illegal workflow edge; final verification cannot reopen selection",
            ));
        }
        let holdout_opened: bool = tx
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM holdout_openings WHERE contract=?1)",
                [lease.contract.as_str()],
                |row| row.get(0),
            )
            .map_err(sql)?;
        if holdout_opened
            && matches!(
                request.next,
                Stage::Profile
                    | Stage::Baseline
                    | Stage::Diagnose
                    | Stage::Design
                    | Stage::Execute
                    | Stage::Evaluate
                    | Stage::Search
            )
        {
            return Err(failure(
                ErrorCode::InvalidTransition,
                "final holdout feedback closes search for this contract revision",
            ));
        }
        let count: i64 = tx
            .query_row("SELECT count(*) FROM events", [], |row| row.get(0))
            .map_err(sql)?;
        if count >= self.limits.max_events as i64 {
            return Err(failure(
                ErrorCode::QuotaExceeded,
                "event registry bound exceeded",
            ));
        }
        let body = EventBody {
            schema_version: 1,
            contract: lease.contract.clone(),
            sequence: current.sequence + 1,
            previous: current.head,
            epoch: lease.epoch,
            owner: lease.owner.clone(),
            boot: now.boot.clone(),
            monotonic_ms: now.ms,
            wall_ms: now.wall_ms,
            transition: request,
        };
        let bytes = canonical(&body)?;
        let digest = ContentDigest::of_bytes(&bytes);
        tx.execute(
            "INSERT INTO events VALUES(?1,?2,?3,?4,?5)",
            params![
                lease.contract.as_str(),
                body.sequence,
                body.transition.operation.as_str(),
                bytes,
                digest.as_str()
            ],
        )
        .map_err(sql)?;
        tx.execute(
            "UPDATE runs SET stage=?1,sequence=?2,head=?3 WHERE contract=?4",
            params![
                body.transition.next.text(),
                body.sequence,
                digest.as_str(),
                lease.contract.as_str()
            ],
        )
        .map_err(sql)?;
        if body.transition.next == Stage::Verify {
            for binding in &contract.portfolio.bindings {
                for item in &binding.splits.holdout.items {
                    tx.execute(
                        "INSERT OR IGNORE INTO final_reservations VALUES(?1,?2,?3,?4)",
                        params![
                            contract.study_family.as_str(),
                            item.item.as_str(),
                            item.group.as_str(),
                            lease.contract.as_str()
                        ],
                    )
                    .map_err(sql)?;
                }
            }
        }
        ensure_live(lease)?;
        tx.commit().map_err(sql)?;
        self.files.verify(self.limits.max_database_bytes)?;
        Ok(Event { digest, body })
    }
    pub fn events(&self, id: &ContentDigest, after: u64, limit: u32) -> Result<Vec<Event>> {
        if limit == 0 || limit > 1000 || after > 1_000_000 {
            return Err(failure(
                ErrorCode::LimitExceeded,
                "bounded event page required",
            ));
        }
        self.files.verify(self.limits.max_database_bytes)?;
        let tx = self.connection.unchecked_transaction().map_err(sql)?;
        load_contract(&tx, id)?;
        state(&tx, id)?;
        let mut statement=tx.prepare("SELECT body,digest FROM events WHERE contract=?1 AND sequence>?2 ORDER BY sequence LIMIT ?3").map_err(sql)?;
        let rows = statement
            .query_map(params![id.as_str(), after, limit], |row| {
                Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(sql)?;
        rows.map(|row| {
            let (bytes, digest) = row.map_err(sql)?;
            decode_event(&bytes, &digest)
        })
        .collect()
    }
    pub fn family_budget_origin(
        &self,
        family: &ContentDigest,
    ) -> Result<(ContentDigest, StudyLimits)> {
        self.files.verify(self.limits.max_database_bytes)?;
        let (id, bytes): (String, Vec<u8>) = self
            .connection
            .query_row(
                "SELECT origin,limits FROM family_budget WHERE family=?1",
                [family.as_str()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .map_err(sql)?;
        let contract = load_contract(&self.connection, &id.parse()?)?;
        if bytes != canonical(&contract.limits)? {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "family budget origin differs",
            ));
        }
        Ok((id.parse()?, contract.limits.clone()))
    }
}
fn load_contract(connection: &Connection, id: &ContentDigest) -> Result<Validated<StudyContract>> {
    let bytes: Vec<u8> = connection
        .query_row(
            "SELECT canonical FROM contracts WHERE digest=?1",
            [id.as_str()],
            |row| row.get(0),
        )
        .map_err(sql)?;
    if ContentDigest::of_bytes(&bytes) != *id {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "frozen contract bytes do not match identity",
        ));
    }
    let contract = Validated::<StudyContract>::parse(&bytes)?;
    if canonical(&*contract)? != bytes {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "stored contract is not typed canonical form",
        ));
    }
    Ok(contract)
}
fn latest(tx: &Transaction<'_>, contract: &StudyContract, id: &ContentDigest) -> Result<()> {
    let current: String = tx
        .query_row(
            "SELECT latest FROM families WHERE family=?1",
            [contract.study_family.as_str()],
            |row| row.get(0),
        )
        .map_err(sql)?;
    if current != id.as_str() {
        return Err(failure(
            ErrorCode::StaleLease,
            "contract is not current family revision",
        ));
    }
    Ok(())
}
fn ensure_live(lease: &Lease) -> Result<()> {
    let now = files::moment()?;
    if now.boot != lease.boot || now.ms >= lease.deadline {
        return Err(failure(
            ErrorCode::StaleLease,
            "lease expired before transaction commit",
        ));
    }
    Ok(())
}
fn check_lease(
    tx: &Transaction<'_>,
    store: &str,
    lease: &Lease,
    now: &files::Moment,
) -> Result<Validated<StudyContract>> {
    let current = files::moment()?;
    if current.boot != now.boot || current.ms < now.ms {
        return Err(failure(
            ErrorCode::StaleLease,
            "coordinator clock changed while acquiring transaction",
        ));
    }
    if lease.store != store || lease.boot != current.boot || lease.deadline <= current.ms {
        return Err(failure(
            ErrorCode::StaleLease,
            "expired/foreign coordinator capability",
        ));
    }
    let contract = load_contract(tx, &lease.contract)?;
    latest(tx, &contract, &lease.contract)?;
    state(tx, &lease.contract)?;
    let (owner, epoch, boot, deadline): (Option<String>, i64, Option<String>, i64) = tx
        .query_row(
            "SELECT owner,epoch,boot,deadline FROM runs WHERE contract=?1",
            [lease.contract.as_str()],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .map_err(sql)?;
    if owner.as_deref() != Some(&lease.owner)
        || epoch != lease.epoch
        || boot.as_deref() != Some(&lease.boot)
        || deadline != lease.deadline
    {
        return Err(failure(
            ErrorCode::StaleLease,
            "coordinator fencing token/owner/deadline differs",
        ));
    }
    let end = files::moment()?;
    if end.boot != lease.boot || end.ms >= lease.deadline {
        return Err(failure(
            ErrorCode::StaleLease,
            "lease expired during state verification",
        ));
    }
    Ok(contract)
}
fn decode_event(bytes: &[u8], id: &str) -> Result<Event> {
    let digest: ContentDigest = id.parse()?;
    if ContentDigest::of_bytes(bytes) != digest {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "event content identity mismatch",
        ));
    }
    let document = CanonicalDocument::parse(bytes, Limits::default())?;
    let body: EventBody = serde_json::from_slice(document.bytes()).map_err(|_| {
        failure(
            ErrorCode::IntegrityMismatch,
            "invalid persisted event schema",
        )
    })?;
    if body.schema_version != 1 || canonical(&body)? != bytes {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "event canonical version mismatch",
        ));
    }
    Ok(Event { digest, body })
}
fn find_event(
    connection: &Connection,
    id: &ContentDigest,
    operation: &ContentDigest,
) -> Result<Option<Event>> {
    let data: Option<(Vec<u8>, String)> = connection
        .query_row(
            "SELECT body,digest FROM events WHERE contract=?1 AND operation=?2",
            params![id.as_str(), operation.as_str()],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()
        .map_err(sql)?;
    data.map(|(bytes, digest)| decode_event(&bytes, &digest))
        .transpose()
}
fn state(connection: &Connection, id: &ContentDigest) -> Result<RunState> {
    let (stage, sequence, head, epoch): (String, u64, String, i64) = connection
        .query_row(
            "SELECT stage,sequence,head,epoch FROM runs WHERE contract=?1",
            [id.as_str()],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .map_err(sql)?;
    if sequence == 0 || sequence > 1_000_000 {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "invalid journal sequence bound",
        ));
    }
    let mut statement = connection
        .prepare(
            "SELECT sequence,operation,body,digest FROM events WHERE contract=?1 ORDER BY sequence",
        )
        .map_err(sql)?;
    let mut rows = statement.query([id.as_str()]).map_err(sql)?;
    let mut count = 0u64;
    let mut previous = ContentDigest::of_bytes(b"nf.coordinator.event.genesis.v1");
    let mut expected = Stage::Frozen;
    let mut last_epoch = 0;
    while let Some(row) = rows.next().map_err(sql)? {
        count += 1;
        if count > sequence {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "journal extends past committed head",
            ));
        }
        let row_sequence: u64 = row.get(0).map_err(sql)?;
        let operation: String = row.get(1).map_err(sql)?;
        let bytes: Vec<u8> = row.get(2).map_err(sql)?;
        let digest: String = row.get(3).map_err(sql)?;
        let event = decode_event(&bytes, &digest)?;
        if row_sequence != count
            || event.body.sequence != count
            || event.body.contract != *id
            || event.body.previous != previous
            || event.body.transition.operation.as_str() != operation
            || event.body.transition.expected != expected
            || event.body.epoch < last_epoch
        {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "broken event ordering/identity/fencing chain",
            ));
        }
        if count == 1 {
            if event.body.transition.next != Stage::Frozen
                || event.body.epoch != 0
                || event.body.transition.operation != *id
            {
                return Err(failure(
                    ErrorCode::IntegrityMismatch,
                    "invalid freeze genesis event",
                ));
            }
        } else if !expected.permits(event.body.transition.next) {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "illegal persisted workflow transition",
            ));
        }
        previous = event.digest;
        expected = event.body.transition.next;
        last_epoch = event.body.epoch;
    }
    if count != sequence
        || previous.as_str() != head
        || expected.text() != stage
        || epoch < last_epoch
    {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "run head differs from verified journal",
        ));
    }
    Ok(RunState {
        contract: id.clone(),
        stage: Stage::parse(&stage)?,
        sequence,
        head: head.parse()?,
        fencing_token: epoch,
    })
}
fn within(new: &StudyLimits, old: &StudyLimits) -> bool {
    new.wall_ms <= old.wall_ms
        && new.cpu_ms <= old.cpu_ms
        && new.gpu_ms <= old.gpu_ms
        && new.candidate_attempts <= old.candidate_attempts
        && new.model_evaluations <= old.model_evaluations
        && new.reasoning_tokens <= old.reasoning_tokens
        && new.teacher_tokens <= old.teacher_tokens
        && new.external_spend_microusd <= old.external_spend_microusd
        && new.host_memory_bytes <= old.host_memory_bytes
        && new.device_memory_bytes <= old.device_memory_bytes
        && new.storage_bytes <= old.storage_bytes
        && new.concurrent_workers <= old.concurrent_workers
        && new.retries_per_operation <= old.retries_per_operation
}
