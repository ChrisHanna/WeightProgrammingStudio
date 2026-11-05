//! Immutable three-way data catalogs, frozen panels and fenced worker scopes.
//! Data declarations are not semantic-independence or model-execution evidence.
#[cfg(test)]
mod migration_tests;
use super::exposure;
use super::*;
use crate::sandbox::{Limits as SandboxLimits, QualifiedDocker, Receipt as SandboxReceipt};
use nf_checkpoint::store::ArtifactStore;
use nf_core::study::{ModelIdentity, SplitRole, TaskIdentity};
use std::collections::{BTreeMap, BTreeSet};

pub(super) const MIGRATION_V4: &str = r#"
CREATE TABLE split_catalogs(contract TEXT NOT NULL REFERENCES contracts(digest),binding TEXT NOT NULL,body BLOB NOT NULL,digest TEXT NOT NULL,PRIMARY KEY(contract,binding));
CREATE TABLE final_panels(contract TEXT PRIMARY KEY REFERENCES contracts(digest),body BLOB NOT NULL,digest TEXT NOT NULL);
CREATE TABLE split_grants(worker TEXT PRIMARY KEY,contract TEXT NOT NULL REFERENCES contracts(digest),token_hash TEXT NOT NULL,body BLOB NOT NULL,digest TEXT NOT NULL);
CREATE TABLE split_access(worker TEXT NOT NULL REFERENCES split_grants(worker),sequence INTEGER NOT NULL,body BLOB NOT NULL,digest TEXT NOT NULL,PRIMARY KEY(worker,sequence));
CREATE TABLE holdout_openings(contract TEXT PRIMARY KEY REFERENCES contracts(digest),panel TEXT NOT NULL,body BLOB NOT NULL,digest TEXT NOT NULL);
CREATE TRIGGER catalogs_no_update BEFORE UPDATE ON split_catalogs BEGIN SELECT RAISE(ABORT,'immutable split catalog');END;
CREATE TRIGGER catalogs_no_delete BEFORE DELETE ON split_catalogs BEGIN SELECT RAISE(ABORT,'retained split catalog');END;
CREATE TRIGGER panels_no_update BEFORE UPDATE ON final_panels BEGIN SELECT RAISE(ABORT,'frozen final panel');END;
CREATE TRIGGER panels_no_delete BEFORE DELETE ON final_panels BEGIN SELECT RAISE(ABORT,'retained final panel');END;
CREATE TRIGGER split_grants_no_update BEFORE UPDATE ON split_grants BEGIN SELECT RAISE(ABORT,'immutable worker scope');END;
CREATE TRIGGER split_grants_no_delete BEFORE DELETE ON split_grants BEGIN SELECT RAISE(ABORT,'retained worker scope');END;
CREATE TRIGGER split_access_no_update BEFORE UPDATE ON split_access BEGIN SELECT RAISE(ABORT,'immutable data access');END;
CREATE TRIGGER split_access_no_delete BEFORE DELETE ON split_access BEGIN SELECT RAISE(ABORT,'retained data access');END;
CREATE TRIGGER holdout_no_update BEFORE UPDATE ON holdout_openings BEGIN SELECT RAISE(ABORT,'irreversible holdout opening');END;
CREATE TRIGGER holdout_no_delete BEFORE DELETE ON holdout_openings BEGIN SELECT RAISE(ABORT,'retained holdout opening');END;
INSERT INTO budget_migrations VALUES(4,(SELECT value FROM meta WHERE key='schema'));
PRAGMA user_version=4;
"#;
pub(super) fn schema_v4() -> ContentDigest {
    ContentDigest::of_bytes(
        &[
            SCHEMA.as_bytes(),
            budgets::MIGRATION_V2.as_bytes(),
            supervision::MIGRATION_V3.as_bytes(),
            MIGRATION_V4.as_bytes(),
        ]
        .concat(),
    )
}
#[cfg(test)]
pub(super) const STRIP_V4:&str="DROP TABLE split_access;DROP TABLE split_grants;DROP TABLE holdout_openings;DROP TABLE final_panels;DROP TABLE split_catalogs;DELETE FROM budget_migrations WHERE version=4;";
fn decode<T: serde::de::DeserializeOwned + Serialize>(bytes: &[u8], digest: &str) -> Result<T> {
    if bytes.len() > 1048576 || ContentDigest::of_bytes(bytes).as_str() != digest {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "split record integrity mismatch",
        ));
    }
    let doc = CanonicalDocument::parse(bytes, Limits::default())?;
    let value: T = serde_json::from_slice(doc.bytes())
        .map_err(|_| failure(ErrorCode::InvalidValue, "invalid split record schema"))?;
    if canonical(&value)? != bytes {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "noncanonical split record",
        ));
    }
    Ok(value)
}
fn opaque() -> Result<String> {
    use std::io::Read;
    let mut data = [0u8; 32];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut data))
        .map_err(|_| failure(ErrorCode::Io, "worker entropy unavailable"))?;
    Ok(ContentDigest::of_bytes(&data).as_str()[7..].to_owned())
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Item {
    pub identity: TaskIdentity,
    pub input: ContentDigest,
    pub label: ContentDigest,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Dataset {
    pub schema_version: u32,
    pub role: SplitRole,
    pub items: Vec<Item>,
}
fn payload(store: &ArtifactStore, id: &ContentDigest) -> Result<serde_json::Value> {
    let bytes = store.read_bounded(id, 16384)?;
    let doc = CanonicalDocument::parse(
        &bytes,
        Limits {
            max_bytes: 16384,
            max_depth: 16,
            max_nodes: 2048,
        },
    )?;
    if doc.digest() != *id {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "payload must be canonical JSON",
        ));
    }
    Ok(doc.value().clone())
}
/// Controller publication; only immutable references enter a frozen contract.
pub fn publish_dataset(store: &ArtifactStore, dataset: &Dataset) -> Result<ContentDigest> {
    if dataset.schema_version != 1 || dataset.items.is_empty() || dataset.items.len() > 256 {
        return Err(failure(
            ErrorCode::LimitExceeded,
            "bounded nonempty dataset v1 required",
        ));
    }
    let mut ids = BTreeSet::new();
    for item in &dataset.items {
        if !ids.insert(&item.identity.item) {
            return Err(failure(ErrorCode::InvalidValue, "duplicate task"));
        }
        payload(store, &item.input)?;
        payload(store, &item.label)?;
    }
    Ok(store.put(canonical(dataset)?.as_slice())?.digest)
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Catalog {
    schema_version: u32,
    contract: ContentDigest,
    binding: ContentDigest,
    datasets: BTreeMap<SplitRole, Dataset>,
    manifests: BTreeMap<SplitRole, ContentDigest>,
}
impl Catalog {
    fn dataset(&self, role: SplitRole) -> Result<&Dataset> {
        self.datasets
            .get(&role)
            .ok_or_else(|| failure(ErrorCode::IntegrityMismatch, "missing frozen dataset"))
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Panel {
    pub schema_version: u32,
    pub contract: ContentDigest,
    pub revision: u32,
    pub protocol: ContentDigest,
    pub finalists: Vec<ModelIdentity>,
    pub original: ModelIdentity,
    pub best_known: ModelIdentity,
    pub bindings: BTreeSet<ContentDigest>,
    pub seeds: Vec<u64>,
    pub repeats: u32,
    pub comparison_family: ContentDigest,
    /// Optional for retained pre-T038 records; absent plans cannot pass acceptance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub analysis_plan: Option<ContentDigest>,
    pub status: String,
}
fn panel(
    connection: &Connection,
    contract: &ContentDigest,
) -> Result<Option<(ContentDigest, Panel)>> {
    let data: Option<(Vec<u8>, String)> = connection
        .query_row(
            "SELECT body,digest FROM final_panels WHERE contract=?1",
            [contract.as_str()],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()
        .map_err(sql)?;
    data.map(|(b, d)| Ok((d.parse()?, decode(&b, &d)?)))
        .transpose()
}
pub(super) fn require_panel(
    connection: &Connection,
    contract: &ContentDigest,
) -> Result<ContentDigest> {
    panel(connection, contract)?.map(|p| p.0).ok_or_else(|| {
        failure(
            ErrorCode::PermissionDenied,
            "actual frozen finalist panel required before holdout access",
        )
    })
}
fn catalog(
    connection: &Connection,
    contract: &ContentDigest,
    binding: &ContentDigest,
) -> Result<Catalog> {
    let (b, d): (Vec<u8>, String) = connection
        .query_row(
            "SELECT body,digest FROM split_catalogs WHERE contract=?1 AND binding=?2",
            params![contract.as_str(), binding.as_str()],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .map_err(sql)?;
    let value: Catalog = decode(&b, &d).map_err(|error| {
        if error.code == ErrorCode::InvalidValue {
            failure(
                ErrorCode::IntegrityMismatch,
                "invalid retained split catalog schema",
            )
        } else {
            error
        }
    })?;
    if value.schema_version != 1 || value.contract != *contract || value.binding != *binding {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "catalog scope mismatch",
        ));
    }
    // A rehashed retained catalog is not a new authority for payload identities.
    // Rebind every canonical dataset (including input/label references), role,
    // version and manifest to the original validated contract, without CAS I/O.
    let frozen = load_contract(connection, contract)?;
    let binding = frozen
        .portfolio
        .bindings
        .iter()
        .find(|b| b.id == *binding)
        .ok_or_else(|| failure(ErrorCode::IntegrityMismatch, "unknown catalog binding"))?;
    if value.datasets.len() != 3 || value.manifests.len() != 3 {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "incomplete split catalog",
        ));
    }
    for split in [
        &binding.splits.development,
        &binding.splits.selection,
        &binding.splits.holdout,
    ] {
        let dataset = value.dataset(split.role)?;
        let identities = dataset
            .items
            .iter()
            .map(|item| Ok((item.identity.item.clone(), canonical(&item.identity)?)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        let expected = split
            .items
            .iter()
            .map(|item| Ok((item.item.clone(), canonical(item)?)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        if dataset.schema_version != 1
            || dataset.role != split.role
            || ContentDigest::of_bytes(&canonical(dataset)?) != split.dataset
            || value.manifests.get(&split.role) != Some(&split.digest()?)
            || dataset.items.len() != split.items.len()
            || identities != expected
        {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "catalog differs from frozen split",
            ));
        }
    }
    Ok(value)
}
pub(super) fn validate_catalog_in(
    connection: &Connection,
    contract: &ContentDigest,
    binding: &ContentDigest,
) -> Result<()> {
    catalog(connection, contract, binding).map(|_| ())
}
/// Exact immutable catalog references for the built-in final executor only.
pub(super) fn final_items(
    connection: &Connection,
    contract: &ContentDigest,
    binding: &ContentDigest,
) -> Result<Vec<Item>> {
    Ok(catalog(connection, contract, binding)?
        .dataset(SplitRole::Holdout)?
        .items
        .clone())
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Recipient {
    Proposer,
    DevelopmentEvaluator,
    SelectionEvaluator,
    FinalEvaluator,
}
impl Recipient {
    fn role(self) -> SplitRole {
        match self {
            Self::Proposer | Self::DevelopmentEvaluator => SplitRole::Development,
            Self::SelectionEvaluator => SplitRole::Selection,
            Self::FinalEvaluator => SplitRole::Holdout,
        }
    }
    fn labels(self) -> bool {
        self != Self::Proposer
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GrantRecord {
    schema_version: u32,
    contract: ContentDigest,
    revision: u32,
    binding: ContentDigest,
    recipient: Recipient,
    worker: String,
    epoch: i64,
    boot: String,
    deadline: i64,
    panel: Option<ContentDigest>,
    environment: ContentDigest,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Welcome {
    pub schema_version: u32,
    pub contract: ContentDigest,
    pub revision: u32,
    pub binding: ContentDigest,
    pub worker: String,
    pub lease_id: String,
    pub split: SplitRole,
    pub items: Vec<ContentDigest>,
    pub labels_allowed: bool,
    pub panel: Option<ContentDigest>,
}
/// Controller-only, not serializable; worker sees only Welcome and requests.
pub struct Grant {
    store: String,
    record: GrantRecord,
    token: String,
    welcome: Welcome,
}
impl Grant {
    pub fn welcome(&self) -> &Welcome {
        &self.welcome
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReadRequest {
    pub schema_version: u32,
    pub contract: ContentDigest,
    pub revision: u32,
    pub binding: ContentDigest,
    pub worker: String,
    pub lease_id: String,
    pub split: SplitRole,
    pub item: ContentDigest,
    pub labels: bool,
    pub panel: Option<ContentDigest>,
}
impl ReadRequest {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let doc = CanonicalDocument::parse(
            bytes,
            Limits {
                max_bytes: 16384,
                max_depth: 8,
                max_nodes: 512,
            },
        )?;
        serde_json::from_slice(doc.bytes())
            .map_err(|_| failure(ErrorCode::InvalidValue, "invalid bounded split envelope"))
    }
}
/// A data disclosure receipt, never a model/lease/dispatch capability.
pub struct ScopedPayload {
    bytes: Vec<u8>,
    environment: ContentDigest,
    access: ContentDigest,
    store: String,
    contract: ContentDigest,
    epoch: i64,
    boot: String,
    deadline: i64,
    recipient: Recipient,
    exposure: exposure::Ticket,
}
impl ScopedPayload {
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn access_id(&self) -> &ContentDigest {
        &self.access
    }
    /// Low-level controller calculation, not study dispatch or model execution.
    /// Scope bytes are the ONLY data mounted; the private store is never mounted.
    pub fn in_domain(
        &self,
        ledger: &Ledger,
        lease: &Lease,
        runtime: &QualifiedDocker,
        code: &[u8],
        limits: SandboxLimits,
    ) -> Result<SandboxReceipt> {
        self.fence(ledger, lease)?;
        if runtime.environment_identity() != &self.environment {
            return Err(failure(
                ErrorCode::PermissionDenied,
                "scoped data requires its actual qualified access domain",
            ));
        }
        let mut start = self.exposure.clone();
        start.body.operation = ContentDigest::of_bytes(&canonical(&(
            &self.access,
            "worker-domain",
            ContentDigest::of_bytes(code),
            opaque()?,
        ))?);
        start.body.context.programmer = Some(ContentDigest::of_bytes(code));
        let worker_ticket = {
            let tx = ledger.connection.unchecked_transaction().map_err(sql)?;
            let contract = check_lease(&tx, &ledger.store, lease, &files::moment()?)?;
            let ticket = exposure::begin(
                &tx,
                lease,
                &contract,
                exposure::Start {
                    subject: start.body.subject.clone(),
                    recipient: start.body.recipient.clone(),
                    role: start.body.role.clone(),
                    operation: start.body.operation.clone(),
                    context: start.body.context.clone(),
                    fields: start.body.intended.clone(),
                },
            )?;
            ensure_live(lease)?;
            tx.commit().map_err(sql)?;
            ticket
        };
        let run = runtime.execute(
            code,
            &self.bytes,
            limits,
            &std::sync::atomic::AtomicBool::new(false),
        );
        let clean = run
            .as_ref()
            .is_ok_and(|r| r.cleanup_confirmed && r.status == "completed");
        let raw = run.as_ref().ok().and_then(|r| r.report.as_ref());
        ledger.finish_exposure(
            lease,
            &worker_ticket,
            if clean {
                exposure::Stage::Completed
            } else {
                exposure::Stage::Incomplete
            },
            if clean {
                worker_ticket.body.intended.clone()
            } else {
                BTreeSet::new()
            },
            if clean {
                "worker_domain_completed"
            } else {
                "worker_domain_failed_or_partial"
            },
            exposure::Cost {
                cpu_ms: raw.and_then(|v| v["cpu_ms"].as_u64()),
                ..Default::default()
            },
        )?;
        let receipt = run?;
        self.fence(ledger, lease)?;
        Ok(receipt)
    }
    fn fence(&self, ledger: &Ledger, lease: &Lease) -> Result<()> {
        ledger.files.verify(ledger.limits.max_database_bytes)?;
        let tx = ledger.connection.unchecked_transaction().map_err(sql)?;
        check_lease(&tx, &ledger.store, lease, &files::moment()?)?;
        let now = files::moment()?;
        if self.store != ledger.store
            || self.contract != lease.contract
            || self.epoch != lease.epoch
            || self.boot != now.boot
            || now.ms >= self.deadline
        {
            return Err(failure(
                ErrorCode::StaleLease,
                "data scope cannot dispatch after its live fence",
            ));
        }
        phase(&tx, &self.contract, self.recipient)
    }
}
fn phase(connection: &Connection, id: &ContentDigest, recipient: Recipient) -> Result<()> {
    let stage = state(connection, id)?.stage;
    if supervision::stopped(connection, id)?
        || stage.terminal()
        || (recipient == Recipient::FinalEvaluator && stage != Stage::Verify)
        || (recipient != Recipient::FinalEvaluator && matches!(stage, Stage::Verify | Stage::Learn))
    {
        return Err(failure(
            ErrorCode::PermissionDenied,
            "split scope closed for this workflow phase",
        ));
    }
    Ok(())
}
pub(super) fn mark_open(
    connection: &Connection,
    id: &ContentDigest,
    panel: &ContentDigest,
    worker: &str,
) -> Result<()> {
    let contract = load_contract(connection, id)?;
    if !super::acceptance::has_resource_ceilings(&contract) {
        return Err(failure(
            ErrorCode::VerificationPolicyWeakened,
            "final holdout requires frozen latency and memory ceilings",
        ));
    }
    let body = canonical(
        &serde_json::json!({"schema_version":1,"contract":id,"panel":panel,"worker":worker,"status":"opened_before_worker_metadata_disclosure"}),
    )?;
    connection
        .execute(
            "INSERT OR IGNORE INTO holdout_openings VALUES(?1,?2,?3,?4)",
            params![
                id.as_str(),
                panel.as_str(),
                body,
                ContentDigest::of_bytes(&body).as_str()
            ],
        )
        .map_err(sql)?;
    Ok(())
}
impl Ledger {
    pub fn bind_split_catalog(
        &mut self,
        lease: &Lease,
        store: &ArtifactStore,
        binding: &ContentDigest,
    ) -> Result<ContentDigest> {
        self.files.verify(self.limits.max_database_bytes)?;
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sql)?;
        let contract = check_lease(&tx, &self.store, lease, &files::moment()?)?;
        if supervision::stopped(&tx, &lease.contract)?
            || state(&tx, &lease.contract)?.stage.terminal()
        {
            return Err(failure(
                ErrorCode::PermissionDenied,
                "inactive contract cannot bind splits",
            ));
        }
        let binding = contract
            .portfolio
            .bindings
            .iter()
            .find(|b| b.id == *binding)
            .ok_or_else(|| failure(ErrorCode::SourceMismatch, "unknown exact binding"))?;
        let mut datasets = BTreeMap::new();
        let mut manifests = BTreeMap::new();
        let mut items = BTreeMap::new();
        let mut groups = BTreeMap::new();
        let mut inputs = BTreeMap::new();
        for split in [
            &binding.splits.development,
            &binding.splits.selection,
            &binding.splits.holdout,
        ] {
            let bytes = store.read_bounded(&split.dataset, 1048576)?;
            let dataset: Dataset = decode(&bytes, split.dataset.as_str())?;
            if dataset.schema_version != 1
                || dataset.role != split.role
                || dataset.items.len() != split.items.len()
                || dataset.items.len() > 256
            {
                return Err(failure(
                    ErrorCode::SourceMismatch,
                    "dataset does not resolve frozen split",
                ));
            }
            let expected: BTreeMap<_, _> = split
                .items
                .iter()
                .map(|item| (&item.item, canonical(item)))
                .map(|(i, b)| b.map(|b| (i, b)))
                .collect::<Result<_>>()?;
            let mut observed = BTreeSet::new();
            for item in &dataset.items {
                if !observed.insert(&item.identity.item)
                    || expected.get(&item.identity.item) != Some(&canonical(&item.identity)?)
                {
                    return Err(failure(
                        ErrorCode::SourceMismatch,
                        "dataset task/group/family mismatch",
                    ));
                }
                payload(store, &item.input)?;
                payload(store, &item.label)?;
                for (map, key) in [
                    (&mut items, &item.identity.item),
                    (&mut groups, &item.identity.group),
                    (&mut inputs, &item.input),
                ] {
                    if map
                        .insert(key.clone(), split.role)
                        .is_some_and(|prior| prior != split.role)
                    {
                        return Err(failure(
                            ErrorCode::ScopeViolation,
                            "cross-split task/group/input alias",
                        ));
                    }
                }
            }
            datasets.insert(split.role, dataset);
            manifests.insert(split.role, split.digest()?);
        }
        // Check all already bound catalogs in this family, including older
        // revisions; a renamed exact consumed holdout cannot become fresh data.
        let mut statement=tx.prepare("SELECT c.body,c.digest,EXISTS(SELECT 1 FROM holdout_openings h WHERE h.contract=c.contract) FROM split_catalogs c JOIN contracts r ON r.digest=c.contract WHERE r.family=?1").map_err(sql)?;
        let rows = statement
            .query_map([contract.study_family.as_str()], |r| {
                Ok((
                    r.get::<_, Vec<u8>>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, bool>(2)?,
                ))
            })
            .map_err(sql)?
            .take(4097)
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(sql)?;
        if rows.len() > 4096 {
            return Err(failure(
                ErrorCode::LimitExceeded,
                "catalog history bound exceeded",
            ));
        }
        for (body, digest, opened) in rows {
            let old: Catalog = decode(&body, &digest)?;
            for (role, dataset) in old.datasets {
                for item in dataset.items {
                    if items.get(&item.identity.item).is_some_and(|r| *r != role)
                        || groups.get(&item.identity.group).is_some_and(|r| *r != role)
                        || inputs.get(&item.input).is_some_and(|r| *r != role)
                        || (old.contract != lease.contract
                            && opened
                            && role == SplitRole::Holdout
                            && inputs.contains_key(&item.input))
                    {
                        return Err(failure(
                            ErrorCode::ScopeViolation,
                            "retained split history forbids overlap or holdout renaming",
                        ));
                    }
                }
            }
        }
        drop(statement);
        exposure::bind(&tx, lease, &contract, &binding.id, &datasets, store)?;
        let value = Catalog {
            schema_version: 1,
            contract: lease.contract.clone(),
            binding: binding.id.clone(),
            datasets,
            manifests,
        };
        let body = canonical(&value)?;
        let id = ContentDigest::of_bytes(&body);
        let prior: Option<String> = tx
            .query_row(
                "SELECT digest FROM split_catalogs WHERE contract=?1 AND binding=?2",
                params![lease.contract.as_str(), binding.id.as_str()],
                |r| r.get(0),
            )
            .optional()
            .map_err(sql)?;
        if let Some(prior) = prior {
            if prior != id.as_str() {
                return Err(failure(
                    ErrorCode::IdempotencyConflict,
                    "frozen split catalog differs",
                ));
            }
        } else {
            tx.execute(
                "INSERT INTO split_catalogs VALUES(?1,?2,?3,?4)",
                params![
                    lease.contract.as_str(),
                    binding.id.as_str(),
                    body,
                    id.as_str()
                ],
            )
            .map_err(sql)?;
        }
        ensure_live(lease)?;
        tx.commit().map_err(sql)?;
        self.files.verify(self.limits.max_database_bytes)?;
        Ok(id)
    }
    pub fn freeze_final_panel(
        &mut self,
        lease: &Lease,
        finalists: Vec<ModelIdentity>,
    ) -> Result<ContentDigest> {
        self.freeze_final_panel_inner(lease, finalists, None)
    }
    /// Bind the native analysis plan and every candidate before holdout opens.
    /// This freezes evidence requirements, not their execution or acceptance.
    pub fn freeze_final_panel_with_analysis(
        &mut self,
        lease: &Lease,
        finalists: Vec<ModelIdentity>,
        plan: &crate::statistics::Plan,
    ) -> Result<ContentDigest> {
        self.freeze_final_panel_inner(lease, finalists, Some(plan.id().clone()))
    }
    fn freeze_final_panel_inner(
        &mut self,
        lease: &Lease,
        finalists: Vec<ModelIdentity>,
        analysis_plan: Option<ContentDigest>,
    ) -> Result<ContentDigest> {
        self.files.verify(self.limits.max_database_bytes)?;
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sql)?;
        let contract = check_lease(&tx, &self.store, lease, &files::moment()?)?;
        phase(&tx, &lease.contract, Recipient::FinalEvaluator)?;
        if finalists.is_empty() || finalists.len() > contract.statistics.maximum_finalists as usize
        {
            return Err(failure(
                ErrorCode::QuotaExceeded,
                "bounded predeclared finalist panel required",
            ));
        }
        let mut unique = BTreeSet::new();
        for model in &finalists {
            if !unique.insert(ContentDigest::of_bytes(&canonical(model)?)) {
                return Err(failure(
                    ErrorCode::InvalidValue,
                    "duplicate finalist identity",
                ));
            }
        }
        let panel = Panel {
            schema_version: 1,
            contract: lease.contract.clone(),
            revision: contract.revision,
            protocol: contract.portfolio.independent.frozen_panel_protocol.clone(),
            finalists,
            original: contract.source.clone(),
            best_known: contract.best_known.clone(),
            bindings: contract
                .portfolio
                .bindings
                .iter()
                .filter(|b| b.schedule.final_required)
                .map(|b| b.id.clone())
                .collect(),
            seeds: contract.seeds.clone(),
            repeats: contract.statistics.repeats,
            comparison_family: contract.statistics.multiplicity_family.clone(),
            analysis_plan,
            status: "frozen_identities_not_model_execution_evidence".into(),
        };
        let body = canonical(&panel)?;
        let id = ContentDigest::of_bytes(&body);
        if let Some((prior, _)) = self::panel(&tx, &lease.contract)? {
            if prior != id {
                return Err(failure(
                    ErrorCode::RevisionConflict,
                    "final panel is immutable",
                ));
            }
        } else {
            tx.execute(
                "INSERT INTO final_panels VALUES(?1,?2,?3)",
                params![lease.contract.as_str(), body, id.as_str()],
            )
            .map_err(sql)?;
        }
        ensure_live(lease)?;
        tx.commit().map_err(sql)?;
        self.files.verify(self.limits.max_database_bytes)?;
        Ok(id)
    }
    pub fn frozen_final_panel(&self, id: &ContentDigest) -> Result<Option<(ContentDigest, Panel)>> {
        self.files.verify(self.limits.max_database_bytes)?;
        load_contract(&self.connection, id)?;
        let result = panel(&self.connection, id)?;
        self.files.verify(self.limits.max_database_bytes)?;
        Ok(result)
    }
    pub(super) fn open_fixture_holdout(&mut self, lease: &Lease, worker: &str) -> Result<()> {
        self.files.verify(self.limits.max_database_bytes)?;
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sql)?;
        check_lease(&tx, &self.store, lease, &files::moment()?)?;
        phase(&tx, &lease.contract, Recipient::FinalEvaluator)?;
        let id = require_panel(&tx, &lease.contract)?;
        mark_open(&tx, &lease.contract, &id, worker)?;
        ensure_live(lease)?;
        tx.commit().map_err(sql)?;
        self.files.verify(self.limits.max_database_bytes)
    }
    pub fn grant_split_worker(
        &mut self,
        lease: &Lease,
        binding: &ContentDigest,
        recipient: Recipient,
        runtime: &QualifiedDocker,
        ttl_ms: u64,
    ) -> Result<Grant> {
        if !(100..=60000).contains(&ttl_ms) {
            return Err(failure(
                ErrorCode::InvalidValue,
                "bounded worker lifetime required",
            ));
        }
        self.files.verify(self.limits.max_database_bytes)?;
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sql)?;
        let contract = check_lease(&tx, &self.store, lease, &files::moment()?)?;
        phase(&tx, &lease.contract, recipient)?;
        let catalog = catalog(&tx, &lease.contract, binding)?;
        let panel = if recipient == Recipient::FinalEvaluator {
            let (id, body) = self::panel(&tx, &lease.contract)?.ok_or_else(|| {
                failure(
                    ErrorCode::PermissionDenied,
                    "freeze final panel before opening holdout",
                )
            })?;
            if !body.bindings.contains(binding) {
                return Err(failure(
                    ErrorCode::ScopeViolation,
                    "binding is outside frozen panel",
                ));
            }
            Some(id)
        } else {
            None
        };
        let count: i64 = tx
            .query_row(
                "SELECT count(*) FROM split_grants WHERE contract=?1",
                [lease.contract.as_str()],
                |r| r.get(0),
            )
            .map_err(sql)?;
        if count >= 1024 {
            return Err(failure(
                ErrorCode::QuotaExceeded,
                "worker scope bound exceeded",
            ));
        }
        let worker = opaque()?;
        let token = opaque()?;
        let now = files::moment()?;
        let record = GrantRecord {
            schema_version: 1,
            contract: lease.contract.clone(),
            revision: contract.revision,
            binding: binding.clone(),
            recipient,
            worker: worker.clone(),
            epoch: lease.epoch,
            boot: now.boot,
            deadline: (now.ms + ttl_ms as i64).min(lease.deadline),
            panel: panel.clone(),
            environment: runtime.environment_identity().clone(),
        };
        let body = canonical(&record)?;
        let id = ContentDigest::of_bytes(&body);
        tx.execute(
            "INSERT INTO split_grants VALUES(?1,?2,?3,?4,?5)",
            params![
                worker,
                lease.contract.as_str(),
                ContentDigest::of_bytes(token.as_bytes()).as_str(),
                body,
                id.as_str()
            ],
        )
        .map_err(sql)?;
        if let Some(panel) = &panel {
            mark_open(&tx, &lease.contract, panel, &worker)?;
        }
        exposure::metadata_scope(
            &tx,
            lease,
            &contract,
            binding,
            catalog.dataset(recipient.role())?,
            &worker,
            recipient,
        )?;
        let welcome = Welcome {
            schema_version: 1,
            contract: lease.contract.clone(),
            revision: contract.revision,
            binding: binding.clone(),
            worker,
            lease_id: token.clone(),
            split: recipient.role(),
            items: catalog
                .dataset(recipient.role())?
                .items
                .iter()
                .map(|item| item.identity.item.clone())
                .collect(),
            labels_allowed: recipient.labels(),
            panel,
        };
        ensure_live(lease)?;
        tx.commit().map_err(sql)?;
        self.files.verify(self.limits.max_database_bytes)?;
        Ok(Grant {
            store: self.store.clone(),
            record,
            token,
            welcome,
        })
    }
    pub fn read_split(
        &mut self,
        lease: &Lease,
        grant: &Grant,
        request: &ReadRequest,
        store: &ArtifactStore,
    ) -> Result<ScopedPayload> {
        self.read_split_traced(lease, grant, request, store, exposure::Context::default())
    }
    pub fn read_split_traced(
        &mut self,
        lease: &Lease,
        grant: &Grant,
        request: &ReadRequest,
        store: &ArtifactStore,
        context: exposure::Context,
    ) -> Result<ScopedPayload> {
        self.files.verify(self.limits.max_database_bytes)?;
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sql)?;
        let contract = check_lease(&tx, &self.store, lease, &files::moment()?)?;
        let r = &grant.record;
        phase(&tx, &lease.contract, r.recipient)?;
        let now = files::moment()?;
        if grant.store != self.store
            || r.contract != lease.contract
            || r.revision != contract.revision
            || r.epoch != lease.epoch
            || r.boot != now.boot
            || now.ms >= r.deadline
            || request.schema_version != 1
            || request.contract != r.contract
            || request.revision != r.revision
            || request.binding != r.binding
            || request.worker != r.worker
            || request.lease_id != grant.token
            || request.split != r.recipient.role()
            || request.panel != r.panel
            || (request.labels && !r.recipient.labels())
        {
            return Err(failure(
                ErrorCode::PermissionDenied,
                "worker request exceeds exact split scope",
            ));
        }
        let (body, digest, token): (Vec<u8>, String, String) = tx
            .query_row(
                "SELECT body,digest,token_hash FROM split_grants WHERE worker=?1",
                [&r.worker],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .map_err(sql)?;
        let persisted: GrantRecord = decode(&body, &digest)?;
        if canonical(&persisted)? != canonical(r)?
            || token != ContentDigest::of_bytes(request.lease_id.as_bytes()).as_str()
        {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "scoped worker binding changed",
            ));
        }
        if let Some(p) = &r.panel {
            if require_panel(&tx, &lease.contract)? != *p {
                return Err(failure(
                    ErrorCode::IntegrityMismatch,
                    "frozen panel mismatch",
                ));
            }
        }
        let catalog = catalog(&tx, &lease.contract, &r.binding)?;
        let row = catalog
            .dataset(request.split)?
            .items
            .iter()
            .find(|i| i.identity.item == request.item)
            .ok_or_else(|| failure(ErrorCode::ScopeViolation, "task is outside worker scope"))?;
        let sequence: u64 = tx
            .query_row(
                "SELECT count(*) FROM split_access WHERE worker=?1",
                [&r.worker],
                |row| row.get(0),
            )
            .map_err(sql)?;
        if sequence >= 64 {
            return Err(failure(
                ErrorCode::QuotaExceeded,
                "worker data access quota exceeded",
            ));
        }
        let audit = canonical(
            &serde_json::json!({"schema_version":1,"contract":r.contract,"revision":r.revision,"binding":r.binding,"worker":r.worker,"recipient":r.recipient,"split":request.split,"item":row.identity,"input":row.input,"label":if request.labels{Some(&row.label)}else{None},"sequence":sequence,"status":"access_started;partial_or_failed_access_retained","panel":r.panel}),
        )?;
        let access = ContentDigest::of_bytes(&audit);
        tx.execute(
            "INSERT INTO split_access VALUES(?1,?2,?3,?4)",
            params![r.worker, sequence, audit, access.as_str()],
        )
        .map_err(sql)?;
        let row = row.clone();
        let subject = exposure::resolved_subject(
            &tx,
            &contract,
            &lease.contract,
            &r.binding,
            request.split,
            &row,
            store,
        )?;
        exposure::admit(&tx, &contract, &subject)?;
        if context.seed.is_some_and(|s| !contract.seeds.contains(&s)) {
            return Err(failure(
                ErrorCode::ScopeViolation,
                "seed is outside frozen study protocol",
            ));
        }
        let fields = if request.labels {
            [exposure::Field::Input, exposure::Field::Label]
                .into_iter()
                .collect()
        } else {
            [exposure::Field::Input].into_iter().collect()
        };
        let role = serde_json::to_value(r.recipient)
            .unwrap()
            .as_str()
            .unwrap()
            .to_owned();
        let ticket = exposure::begin(
            &tx,
            lease,
            &contract,
            exposure::Start {
                subject,
                recipient: r.worker.clone(),
                role,
                operation: access.clone(),
                context,
                fields,
            },
        )?;
        ensure_live(lease)?;
        tx.commit().map_err(sql)?;
        let result = (|| -> Result<_> {
            let input = payload(store, &row.input)?;
            let label = if request.labels {
                Some(payload(store, &row.label)?)
            } else {
                None
            };
            Ok((input, label))
        })();
        let (input, label) = match result {
            Ok(value) => value,
            Err(error) => {
                self.finish_exposure(
                    lease,
                    &ticket,
                    exposure::Stage::Failed,
                    BTreeSet::new(),
                    "payload_io_failed;no_complete_route",
                    exposure::Cost::default(),
                )?;
                return Err(error);
            }
        };
        // Recheck the live fence after storage I/O and before disclosing bytes.
        let tx = self.connection.unchecked_transaction().map_err(sql)?;
        check_lease(&tx, &self.store, lease, &files::moment()?)?;
        phase(&tx, &lease.contract, r.recipient)?;
        if files::moment()?.ms >= r.deadline {
            return Err(failure(
                ErrorCode::PermissionDenied,
                "worker scope expired during read",
            ));
        }
        let bytes = canonical(
            &serde_json::json!({"schema_version":1,"contract":r.contract,"revision":r.revision,"binding":r.binding,"worker":r.worker,"split":request.split,"item":row.identity,"input":input,"label":label,"panel":r.panel,"access":access,"classification":request.split,"status":"scoped_data_not_a_dispatch_or_promotion_grant"}),
        )?;
        if bytes.len() > 65536 {
            return Err(failure(
                ErrorCode::LimitExceeded,
                "scoped payload exceeds worker bound",
            ));
        }
        exposure::finish(
            &tx,
            &ticket,
            exposure::Stage::Completed,
            ticket.body.intended.clone(),
            "available_to_scoped_controller_route",
            exposure::Cost::default(),
        )?;
        ensure_live(lease)?;
        tx.commit().map_err(sql)?;
        self.files.verify(self.limits.max_database_bytes)?;
        Ok(ScopedPayload {
            bytes,
            environment: r.environment.clone(),
            access,
            store: self.store.clone(),
            contract: r.contract.clone(),
            epoch: r.epoch,
            boot: r.boot.clone(),
            deadline: r.deadline,
            recipient: r.recipient,
            exposure: ticket,
        })
    }
}
