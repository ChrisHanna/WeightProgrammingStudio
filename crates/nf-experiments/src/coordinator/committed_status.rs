//! Read-only diagnostic snapshot. No migration, lease or acceptance capability.
use super::*;
use crate::status_reports::{CommittedRead, FrozenPanelProjection};
use serde_json::{json, Value};

const MAX_DATABASE: u64 = 67_108_864;
const MAX_ROWS: usize = 8192;
const MAX_BYTES: usize = 8_388_608;
#[derive(Default)]
struct Bound {
    rows: usize,
    bytes: usize,
}
impl Bound {
    fn add(&mut self, bytes: usize) -> Result<()> {
        self.rows += 1;
        self.bytes = self
            .bytes
            .checked_add(bytes)
            .ok_or_else(|| failure(ErrorCode::LimitExceeded, "snapshot size overflow"))?;
        if self.rows > MAX_ROWS || self.bytes > MAX_BYTES {
            return Err(failure(
                ErrorCode::LimitExceeded,
                "committed snapshot exceeds row/byte bounds; no partial report",
            ));
        }
        Ok(())
    }
}
/// SQL is a fixed internal query, never caller supplied. Preserve both the
/// indexed locator and canonical content hash; records are diagnostic data.
fn records(tx: &Transaction<'_>, query: &str, scope: &str, bound: &mut Bound) -> Result<Value> {
    let mut statement = tx.prepare(query).map_err(sql)?;
    let mut rows = statement.query([scope]).map_err(sql)?;
    let mut result = Vec::new();
    while let Some(row) = rows.next().map_err(sql)? {
        let key: String = row.get(0).map_err(sql)?;
        let bytes: Vec<u8> = row.get(1).map_err(sql)?;
        let digest: String = row.get(2).map_err(sql)?;
        bound.add(bytes.len() + key.len() + digest.len())?;
        if key.len() > 512 || bytes.len() > 1_048_576 {
            return Err(failure(
                ErrorCode::LimitExceeded,
                "retained record exceeds capture bound",
            ));
        }
        let document = CanonicalDocument::parse(&bytes, Limits::default())?;
        let id: ContentDigest = digest.parse()?;
        if document.digest() != id || document.bytes() != bytes {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "retained committed record hash/canonical bytes differ",
            ));
        }
        result.push(json!({"key":key,"digest":id,"record":document.value()}));
    }
    Ok(Value::Array(result))
}
/// Retain immutable identities, not arbitrary receipt bodies. In particular,
/// native dispatch/execution records contain holdout outputs and samples. A
/// diagnostic capture is not authorization to disclose that content. Verified
/// typed summaries must come from the transaction-scoped final-record adapter.
fn receipt_identities(records: Value) -> Value {
    Value::Array(
        records
            .as_array()
            .unwrap()
            .iter()
            .map(|record| {
                json!({"key":record["key"],"digest":record["digest"],
                    "body":"withheld; requires verified typed projection"})
            })
            .collect(),
    )
}

/// The family index is not evidence by itself. Validate each indexed contract
/// and its complete journal, including historical revisions, in this capture's
/// transaction. Otherwise a current report could retain historical budgets and
/// receipt locators while silently ignoring corruption in their event anchors.
fn family_journal(
    tx: &Transaction<'_>,
    family: &ContentDigest,
    selected: &ContentDigest,
    bound: &mut Bound,
) -> Result<(Value, RunState, Vec<Event>)> {
    let mut statement = tx
        .prepare("SELECT digest,revision,parent FROM contracts WHERE family=?1 ORDER BY revision LIMIT 8193")
        .map_err(sql)?;
    let mut rows = statement.query([family.as_str()]).map_err(sql)?;
    let mut journals = Vec::new();
    let mut selected_journal = None;
    let mut parent: Option<ContentDigest> = None;
    while let Some(row) = rows.next().map_err(sql)? {
        let id: String = row.get(0).map_err(sql)?;
        let revision: u32 = row.get(1).map_err(sql)?;
        let indexed_parent: Option<String> = row.get(2).map_err(sql)?;
        let id: ContentDigest = id.parse()?;
        let contract = load_contract(tx, &id)?;
        bound.add(canonical(&*contract)?.len())?;
        if contract.study_family != *family
            || contract.revision != revision
            || revision as usize != journals.len() + 1
            || contract.parent_contract != parent
            || indexed_parent.as_deref() != parent.as_ref().map(ContentDigest::as_str)
        {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "family contract revision/parent index differs",
            ));
        }
        // Bound materialization BEFORE state() walks the journal.
        let events = records(tx,
            "SELECT CAST(sequence AS TEXT),body,digest FROM events WHERE contract=?1 ORDER BY sequence LIMIT 8193",
            id.as_str(), bound)?;
        let state = state(tx, &id)?;
        if id == *selected {
            let typed_events = events
                .as_array()
                .unwrap()
                .iter()
                .map(|r| {
                    decode_event(
                        &canonical(&r["record"])?,
                        r["digest"].as_str().ok_or_else(|| {
                            failure(ErrorCode::IntegrityMismatch, "event identity absent")
                        })?,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            selected_journal = Some((state.clone(), typed_events));
        }
        journals.push(json!({"contract":id,"revision":revision,"parent":parent,
            "state":state,"events":events}));
        parent = Some(id);
    }
    let (latest, revision): (String, u32) = tx
        .query_row(
            "SELECT latest,revision FROM families WHERE family=?1",
            [family.as_str()],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .map_err(sql)?;
    if parent.as_ref().map(ContentDigest::as_str) != Some(latest.as_str())
        || revision as usize != journals.len()
    {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "family latest index differs",
        ));
    }
    let (state, events) = selected_journal.ok_or_else(|| {
        failure(
            ErrorCode::IntegrityMismatch,
            "selected contract absent from family journal",
        )
    })?;
    Ok((
        json!({"scope":"whole_family","revisions":journals}),
        state,
        events,
    ))
}

/// Resolve only what a frozen panel actually establishes. Its typed identity
/// and contract bindings can be checked without executing models or resolving
/// final decisions. There is deliberately no manufactured event-head link:
/// legacy panels do not persist one. A plan digest is not analysis evidence.
fn frozen_panels(
    tx: &Transaction<'_>,
    family: &ContentDigest,
    retained: &Value,
) -> Result<Vec<FrozenPanelProjection>> {
    retained
        .as_array()
        .unwrap()
        .iter()
        .map(|record| {
            let panel: splits::Panel =
                serde_json::from_value(record["record"].clone()).map_err(|_| {
                    failure(ErrorCode::IntegrityMismatch, "invalid frozen panel schema")
                })?;
            let digest: ContentDigest =
                serde_json::from_value(record["digest"].clone()).map_err(|_| {
                    failure(ErrorCode::IntegrityMismatch, "invalid frozen panel digest")
                })?;
            if ContentDigest::of_bytes(&canonical(&panel)?) != digest
                || record["key"] != json!(panel.contract)
                || panel.schema_version != 1
            {
                return Err(failure(
                    ErrorCode::IntegrityMismatch,
                    "frozen panel digest, version or locator differs",
                ));
            }
            let contract = load_contract(tx, &panel.contract)?;
            let bindings: std::collections::BTreeSet<_> = contract
                .portfolio
                .bindings
                .iter()
                .filter(|binding| binding.schedule.final_required)
                .map(|binding| binding.id.clone())
                .collect();
            let candidates = panel
                .finalists
                .iter()
                .map(acceptance::model_identity)
                .collect::<Result<std::collections::BTreeSet<_>>>()?;
            if contract.study_family != *family
                || panel.revision != contract.revision
                || panel.protocol != contract.portfolio.independent.frozen_panel_protocol
                || panel.original != contract.source
                || panel.best_known != contract.best_known
                || panel.bindings != bindings
                || panel.seeds != contract.seeds
                || panel.repeats != contract.statistics.repeats
                || panel.comparison_family != contract.statistics.multiplicity_family
                || panel.finalists.is_empty()
                || panel.finalists.len() > contract.statistics.maximum_finalists as usize
                || candidates.len() != panel.finalists.len()
                || panel.status != "frozen_identities_not_model_execution_evidence"
            {
                return Err(failure(
                    ErrorCode::IntegrityMismatch,
                    "frozen panel identity or contract bindings differ",
                ));
            }
            Ok(FrozenPanelProjection { digest, panel })
        })
        .collect()
}

fn validate_layout(tx: &Transaction<'_>, version: u32) -> Result<()> {
    // Do not execute attacker-substituted views or virtual tables just because
    // a foreign database copied the version/meta hash. All queries below target
    // the exact ordinary tables created by the native schema migrations.
    let mut schemas = vec![
        SCHEMA,
        budgets::MIGRATION_V2,
        supervision::MIGRATION_V3,
        splits::MIGRATION_V4,
        exposure::MIGRATION_V5,
    ];
    if version >= 6 {
        schemas.push(final_fixture::MIGRATION_V6);
    }
    if version >= 7 {
        schemas.push(promotion::MIGRATION_V7);
    }
    if version >= 8 {
        budgets::audit_budget::verify_v8(tx)?;
    }
    for schema in schemas {
        for line in schema.lines().map(str::trim) {
            if let Some(definition) = line.strip_prefix("CREATE TABLE ") {
                let name = definition.split('(').next().unwrap().trim();
                let actual: Option<String> = tx
                    .query_row(
                        "SELECT sql FROM sqlite_schema WHERE type='table' AND name=?1",
                        [name],
                        |r| r.get(0),
                    )
                    .optional()
                    .map_err(sql)?;
                if actual.as_deref().map(str::trim) != Some(line.trim_end_matches(';')) {
                    return Err(failure(
                        ErrorCode::UnsupportedVersion,
                        "foreign committed-status table layout; never execute views or migrate",
                    ));
                }
            }
        }
    }
    Ok(())
}
fn read(path: &Path, id: &ContentDigest, after_anchor: impl FnOnce()) -> Result<CommittedRead> {
    let files = files::Files::open_read_only_file(path, MAX_DATABASE)?;
    let mut connection = Connection::open_with_flags(
        files.sqlite_path(),
        OpenFlags::SQLITE_OPEN_READ_ONLY
            | OpenFlags::SQLITE_OPEN_FULL_MUTEX
            | OpenFlags::SQLITE_OPEN_NOFOLLOW,
    )
    .map_err(sql)?;
    connection
        .busy_timeout(Duration::from_millis(2000))
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
    connection.execute_batch("PRAGMA query_only=ON;PRAGMA trusted_schema=OFF;PRAGMA temp_store=MEMORY;PRAGMA cache_size=-1024;").map_err(sql)?;
    files.verify(MAX_DATABASE)?;
    // Every database observation below, including schema selection, belongs to
    // this ONE deferred read transaction. No nested public Ledger reads.
    let tx = connection
        .transaction_with_behavior(TransactionBehavior::Deferred)
        .map_err(sql)?;
    let version: u32 = tx
        .query_row("PRAGMA user_version", [], |r| r.get(0))
        .map_err(sql)?;
    let expected = match version {
        5 => exposure::schema_v5(),
        6 => final_fixture::schema_v6(),
        7 => promotion::schema_v7(),
        8 => budgets::audit_budget::schema_v8(),
        _ => {
            return Err(failure(
                ErrorCode::UnsupportedVersion,
                "read-only committed status supports schemas 5 through 8; never migrates",
            ))
        }
    };
    validate_layout(&tx, version)?;
    let schema: String = tx
        .query_row("SELECT value FROM meta WHERE key='schema'", [], |r| {
            r.get(0)
        })
        .map_err(sql)?;
    if schema != expected.as_str() {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "committed schema binding differs",
        ));
    }
    let store: String = tx
        .query_row("SELECT value FROM meta WHERE key='store'", [], |r| r.get(0))
        .map_err(sql)?;
    let _: ContentDigest = store.parse()?;
    let contract = load_contract(&tx, id)?;
    let family = contract.study_family.as_str();
    let mut bound = Bound::default();
    let (family_journal, state, events) =
        family_journal(&tx, &contract.study_family, id, &mut bound)?;
    after_anchor();
    let requests = records(
        &tx,
        "SELECT id,body,content FROM budget_requests WHERE family=?1 ORDER BY rowid LIMIT 8193",
        family,
        &mut bound,
    )?;
    let outcomes = records(&tx,"SELECT o.id,o.body,o.content FROM budget_outcomes o JOIN budget_requests r ON r.id=o.request WHERE r.family=?1 ORDER BY r.rowid,o.version LIMIT 8193",family,&mut bound)?;
    let faults = records(
        &tx,
        "SELECT id,body,content FROM budget_faults WHERE family=?1 ORDER BY rowid LIMIT 8193",
        family,
        &mut bound,
    )?;
    let budgets = budgets::snapshot_in(&tx, id)?;
    let (origin, limit_bytes): (String, Vec<u8>) = tx
        .query_row(
            "SELECT origin,limits FROM family_budget WHERE family=?1",
            [family],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .map_err(sql)?;
    bound.add(limit_bytes.len())?;
    let family_origin: ContentDigest = origin.parse()?;
    let original = load_contract(&tx, &family_origin)?;
    if original.study_family != contract.study_family || limit_bytes != canonical(&original.limits)?
    {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "family budget origin differs",
        ));
    }
    let epoch: Option<(String, i64, i64)> = tx
        .query_row(
            "SELECT boot,start_ms,wall_ms FROM budget_epochs WHERE family=?1",
            [family],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .optional()
        .map_err(sql)?;
    let stop = records(&tx,"SELECT s.id,s.body,s.content FROM stop_requests s JOIN contracts c ON c.digest=s.contract WHERE c.family=?1 ORDER BY c.revision LIMIT 8193",family,&mut bound)?;
    let intents = records(&tx,"SELECT w.id,w.body,w.content FROM worker_intents w JOIN contracts c ON c.digest=w.contract WHERE c.family=?1 ORDER BY w.rowid LIMIT 8193",family,&mut bound)?;
    let updates = records(&tx,"SELECT u.worker||':'||u.version,u.body,u.content FROM worker_updates u JOIN worker_intents w ON w.id=u.worker JOIN contracts c ON c.digest=w.contract WHERE c.family=?1 ORDER BY w.rowid,u.version LIMIT 8193",family,&mut bound)?;
    let catalogs = records(&tx,"SELECT s.contract||':'||s.binding,s.body,s.digest FROM split_catalogs s JOIN contracts c ON c.digest=s.contract WHERE c.family=?1 ORDER BY c.revision,s.binding LIMIT 8193",family,&mut bound)?;
    // Validate even catalogs without a completed final decision. Retention and
    // a self-consistent hash alone do not authenticate the frozen workload.
    for record in catalogs.as_array().unwrap() {
        let contract: ContentDigest = serde_json::from_value(record["record"]["contract"].clone())
            .map_err(|_| failure(ErrorCode::IntegrityMismatch, "invalid catalog contract"))?;
        let binding: ContentDigest = serde_json::from_value(record["record"]["binding"].clone())
            .map_err(|_| failure(ErrorCode::IntegrityMismatch, "invalid catalog binding"))?;
        if record["key"] != json!(format!("{contract}:{binding}")) {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "catalog locator differs",
            ));
        }
        splits::validate_catalog_in(&tx, &contract, &binding)?;
    }
    let panels = records(&tx,"SELECT s.contract,s.body,s.digest FROM final_panels s JOIN contracts c ON c.digest=s.contract WHERE c.family=?1 ORDER BY c.revision LIMIT 8193",family,&mut bound)?;
    let frozen_panels = frozen_panels(&tx, &contract.study_family, &panels)?;
    let openings = records(&tx,"SELECT s.contract,s.body,s.digest FROM holdout_openings s JOIN contracts c ON c.digest=s.contract WHERE c.family=?1 ORDER BY c.revision LIMIT 8193",family,&mut bound)?;
    let grants = records(&tx,"SELECT s.worker,s.body,s.digest FROM split_grants s JOIN contracts c ON c.digest=s.contract WHERE c.family=?1 ORDER BY s.rowid LIMIT 8193",family,&mut bound)?;
    let access = records(&tx,"SELECT a.worker||':'||a.sequence,a.body,a.digest FROM split_access a JOIN split_grants s ON s.worker=a.worker JOIN contracts c ON c.digest=s.contract WHERE c.family=?1 ORDER BY s.rowid,a.sequence LIMIT 8193",family,&mut bound)?;
    let subjects = records(
        &tx,
        "SELECT id,body,id FROM exposure_subjects WHERE family=?1 ORDER BY id LIMIT 8193",
        family,
        &mut bound,
    )?;
    let exposures = records(&tx,"SELECT CAST(sequence AS TEXT),body,id FROM exposure_events WHERE family=?1 ORDER BY sequence LIMIT 8193",family,&mut bound)?;
    let mut previous = Value::Null;
    for (index, event) in exposures.as_array().unwrap().iter().enumerate() {
        let body = &event["record"];
        if body["sequence"] != json!(index + 1)
            || body["previous"] != previous
            || body["family"] != json!(contract.study_family)
        {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "retained exposure hash chain differs",
            ));
        }
        previous = event["digest"].clone();
    }
    let legacy = records(
        &tx,
        "SELECT id,body,id FROM legacy_intakes WHERE family=?1 ORDER BY id LIMIT 8193",
        family,
        &mut bound,
    )?;
    let gap: Option<String> = tx
        .query_row(
            "SELECT reason FROM exposure_gaps WHERE family=?1",
            [family],
            |r| r.get(0),
        )
        .optional()
        .map_err(sql)?;
    if let Some(reason) = &gap {
        bound.add(reason.len())?;
    }
    let native_final = if version >= 6 {
        receipt_identities(records(&tx,"SELECT s.contract||':'||s.kind,s.body,s.digest FROM final_fixture_records s JOIN contracts c ON c.digest=s.contract WHERE c.family=?1 ORDER BY c.revision,s.kind LIMIT 8193",family,&mut bound)?)
    } else {
        json!({"status":"unavailable","reason":"schema before native final records"})
    };
    let mut final_evidence = Vec::new();
    if version >= 6 {
        for revision in family_journal["revisions"].as_array().unwrap() {
            let contract: ContentDigest = serde_json::from_value(revision["contract"].clone())
                .map_err(|_| {
                    failure(ErrorCode::IntegrityMismatch, "invalid family receipt scope")
                })?;
            if let Some(evidence) = final_fixture::read_final_evidence_in(&tx, &store, &contract)? {
                final_evidence.push(evidence);
            }
        }
    }
    let result = CommittedRead {
        state,
        events,
        budgets,
        family_origin,
        family_limits: original.limits.clone(),
        database_schema_version: version,
        family_journal,
        frozen_panels,
        final_evidence,
        budget_records: json!({"scope":"whole_family","requests":requests,"outcomes":outcomes,"faults":faults,"epoch":epoch}),
        supervision: json!({"scope":"whole_family","stop_requests":stop,"worker_intents":intents,"worker_updates":updates,"liveness":"not_probed"}),
        split_evidence: json!({"scope":"whole_family","store":store,"catalogs":catalogs,"panels":panels,"holdout_openings":openings,"grants_without_credentials":grants,"access":access,"exposure_subjects":subjects,"exposure_events":exposures,"exposure_gap":gap,"legacy_intakes":legacy,"native_final_record_identities":native_final,"receipt_link_resolution":"transaction_scoped_native_fixture_verifier; absent or unobserved evidence remains unknown"}),
        contract,
    };
    tx.commit().map_err(sql)?;
    files.verify(MAX_DATABASE)?;
    Ok(result)
}
impl Ledger {
    pub(crate) fn read_committed_status(path: &Path, id: &ContentDigest) -> Result<CommittedRead> {
        read(path, id, || {})
    }
}
#[cfg(test)]
mod tests;
