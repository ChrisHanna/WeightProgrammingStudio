//! Atomic family/branch accounting. A reservation is NOT a worker execution grant.
#[path = "audit_budget.rs"]
pub mod audit_budget;
#[cfg(test)]
mod migration_tests;
use super::*;
use nf_core::schemas::{DataClass, ToolKind};
use std::collections::{BTreeMap, BTreeSet};
pub(super) const MIGRATION_V2: &str = r#"
CREATE TABLE budget_migrations(version INTEGER PRIMARY KEY,previous TEXT NOT NULL);
INSERT INTO budget_migrations VALUES(2,(SELECT value FROM meta WHERE key='schema'));
CREATE TABLE budget_epochs(family TEXT PRIMARY KEY,boot TEXT NOT NULL,start_ms INTEGER NOT NULL,wall_ms INTEGER NOT NULL);
CREATE TABLE budget_requests(id TEXT PRIMARY KEY,family TEXT NOT NULL,contract TEXT NOT NULL,operation TEXT NOT NULL,attempt INTEGER NOT NULL,parent TEXT,body BLOB NOT NULL,content TEXT NOT NULL,UNIQUE(family,operation,attempt));
CREATE TABLE budget_outcomes(id TEXT PRIMARY KEY,request TEXT NOT NULL REFERENCES budget_requests(id),version INTEGER NOT NULL,body BLOB NOT NULL,content TEXT NOT NULL,UNIQUE(request,version));
CREATE TABLE budget_faults(id TEXT PRIMARY KEY,family TEXT NOT NULL,body BLOB NOT NULL,content TEXT NOT NULL);
CREATE TRIGGER faults_no_update BEFORE UPDATE ON budget_faults BEGIN SELECT RAISE(ABORT,'immutable budget breach');END;
CREATE TRIGGER faults_no_delete BEFORE DELETE ON budget_faults BEGIN SELECT RAISE(ABORT,'retained budget breach');END;
CREATE TRIGGER requests_no_update BEFORE UPDATE ON budget_requests BEGIN SELECT RAISE(ABORT,'immutable budget requests');END;
CREATE TRIGGER requests_no_delete BEFORE DELETE ON budget_requests BEGIN SELECT RAISE(ABORT,'retained budget requests');END;
CREATE TRIGGER outcomes_no_update BEFORE UPDATE ON budget_outcomes BEGIN SELECT RAISE(ABORT,'immutable budget outcomes');END;
CREATE TRIGGER outcomes_no_delete BEFORE DELETE ON budget_outcomes BEGIN SELECT RAISE(ABORT,'retained budget outcomes');END;
CREATE TRIGGER epoch_no_update BEFORE UPDATE ON budget_epochs BEGIN SELECT RAISE(ABORT,'fixed family deadline');END;
CREATE TRIGGER epoch_no_delete BEFORE DELETE ON budget_epochs BEGIN SELECT RAISE(ABORT,'retained family deadline');END;
PRAGMA user_version=2;
"#;
pub(super) fn schema_v2() -> ContentDigest {
    ContentDigest::of_bytes(
        [SCHEMA.as_bytes(), MIGRATION_V2.as_bytes()]
            .concat()
            .as_slice(),
    )
}
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Resources {
    pub cpu_ms: u64,
    pub gpu_ms: u64,
    pub attempts: u64,
    pub evaluations: u64,
    pub reasoning_tokens: u64,
    pub teacher_tokens: u64,
    pub spend_microusd: u64,
    pub storage_bytes: u64,
    pub host_bytes: u64,
    pub device_bytes: u64,
    pub workers: u64,
}
impl Resources {
    fn cost(&self) -> [u64; 8] {
        [
            self.cpu_ms,
            self.gpu_ms,
            self.attempts,
            self.evaluations,
            self.reasoning_tokens,
            self.teacher_tokens,
            self.spend_microusd,
            self.storage_bytes,
        ]
    }
    fn live(&self) -> [u64; 3] {
        [self.host_bytes, self.device_bytes, self.workers]
    }
    fn from_parts(cost: [u64; 8], live: [u64; 3]) -> Self {
        Self {
            cpu_ms: cost[0],
            gpu_ms: cost[1],
            attempts: cost[2],
            evaluations: cost[3],
            reasoning_tokens: cost[4],
            teacher_tokens: cost[5],
            spend_microusd: cost[6],
            storage_bytes: cost[7],
            host_bytes: live[0],
            device_bytes: live[1],
            workers: live[2],
        }
    }
    fn caps(limits: &StudyLimits) -> Self {
        Self {
            cpu_ms: limits.cpu_ms,
            gpu_ms: limits.gpu_ms,
            attempts: limits.candidate_attempts as u64,
            evaluations: limits.model_evaluations,
            reasoning_tokens: limits.reasoning_tokens,
            teacher_tokens: limits.teacher_tokens,
            spend_microusd: limits.external_spend_microusd,
            storage_bytes: limits.storage_bytes,
            host_bytes: limits.host_memory_bytes,
            device_bytes: limits.device_memory_bytes,
            workers: limits.concurrent_workers as u64,
        }
    }
}
fn plus<const N: usize>(a: [u64; N], b: [u64; N]) -> Result<[u64; N]> {
    let mut out = [0; N];
    for i in 0..N {
        out[i] = a[i]
            .checked_add(b[i])
            .ok_or_else(|| failure(ErrorCode::LimitExceeded, "budget sum overflow"))?;
    }
    Ok(out)
}
fn fits<const N: usize>(used: [u64; N], cap: [u64; N]) -> bool {
    (0..N).all(|i| used[i] <= cap[i])
}
fn subtract<const N: usize>(a: [u64; N], b: [u64; N]) -> Result<[u64; N]> {
    let mut out = [0; N];
    for i in 0..N {
        out[i] = a[i].checked_sub(b[i]).ok_or_else(|| {
            failure(
                ErrorCode::QuotaExceeded,
                "branch or family reservation exceeded",
            )
        })?;
    }
    Ok(out)
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkKind {
    Branch,
    Metadata,
    Inspection,
    Proposal,
    Teacher,
    Candidate,
    Evaluation,
    Search,
    Training,
    Alignment,
    Verification,
    Export,
    Reload,
    Serving,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalCall {
    pub source: ContentDigest,
    pub destination: String,
    pub payloads: BTreeSet<DataClass>,
    pub maximum_price_microusd: Option<u64>,
    pub maximum_tokens: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub schema_version: u32,
    pub operation: ContentDigest,
    pub attempt: u32,
    pub parent: Option<ContentDigest>,
    pub kind: WorkKind,
    pub resources: Resources,
    pub wall_ms: u64,
    pub external: Option<ExternalCall>,
}
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Usage {
    pub cpu_ms: Option<u64>,
    pub gpu_ms: Option<u64>,
    pub attempts: Option<u64>,
    pub evaluations: Option<u64>,
    pub reasoning_tokens: Option<u64>,
    pub teacher_tokens: Option<u64>,
    pub spend_microusd: Option<u64>,
    pub storage_bytes: Option<u64>,
}
impl Usage {
    fn values(&self) -> [Option<u64>; 8] {
        [
            self.cpu_ms,
            self.gpu_ms,
            self.attempts,
            self.evaluations,
            self.reasoning_tokens,
            self.teacher_tokens,
            self.spend_microusd,
            self.storage_bytes,
        ]
    }
    pub fn known(resources: &Resources) -> Self {
        Self {
            cpu_ms: Some(resources.cpu_ms),
            gpu_ms: Some(resources.gpu_ms),
            attempts: Some(resources.attempts),
            evaluations: Some(resources.evaluations),
            reasoning_tokens: Some(resources.reasoning_tokens),
            teacher_tokens: Some(resources.teacher_tokens),
            spend_microusd: Some(resources.spend_microusd),
            storage_bytes: Some(resources.storage_bytes),
        }
    }
    fn charged(&self, reserved: &Resources) -> Result<([u64; 8], [u64; 8])> {
        let mut cost = [0; 8];
        let mut unknown = [0; 8];
        for (i, value) in self.values().iter().enumerate() {
            let ceiling = reserved.cost()[i];
            cost[i] = value.unwrap_or(ceiling);
            if value.is_none() {
                unknown[i] = ceiling;
            }
            if cost[i] > ceiling {
                return Err(failure(
                    ErrorCode::QuotaExceeded,
                    "reported actual cost exceeds reservation; no silent overdraft",
                ));
            }
        }
        Ok((cost, unknown))
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeKind {
    Completed,
    Rejected,
    Incomplete,
    Cancelled,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settlement {
    pub operation: ContentDigest,
    pub outcome: OutcomeKind,
    pub usage: Usage,
    pub resources_released: bool,
    pub evidence: ContentDigest,
}
#[derive(Debug, Clone)]
pub struct Reservation {
    store: String,
    id: ContentDigest,
    family: ContentDigest,
}
impl Reservation {
    pub fn id(&self) -> &ContentDigest {
        &self.id
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StoredRequest {
    schema_version: u32,
    id: ContentDigest,
    family: ContentDigest,
    contract: ContentDigest,
    request: Request,
    boot: String,
    start_ms: i64,
    deadline_ms: i64,
    epoch: i64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StoredOutcome {
    schema_version: u32,
    id: ContentDigest,
    request: ContentDigest,
    version: u32,
    branch_closed: bool,
    settlement: Settlement,
}
#[derive(Debug, Clone, Serialize)]
pub struct Snapshot {
    pub family: ContentDigest,
    pub spent: Resources,
    pub held: Resources,
    pub unknown_charged: Resources,
    pub elapsed_wall_ms: Option<u64>,
    pub active_roots: u32,
    pub reservation_count: u32,
    pub accounting_halted: bool,
    pub reported_overruns: Vec<OverrunReport>,
    pub measured_peak_host_bytes: Option<u64>,
    pub measured_peak_device_bytes: Option<u64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverrunReport {
    pub schema_version: u32,
    pub request: ContentDigest,
    pub contract: ContentDigest,
    pub ceiling: Resources,
    pub settlement: Settlement,
}
#[derive(Debug, Clone, Serialize)]
pub struct Preflight {
    pub accounting_eligible: bool,
    pub dispatch_allowed: bool,
    pub requires_worker: bool,
    pub unmet_required_limits: Vec<String>,
}
#[derive(Default, Clone)]
struct Total {
    spent: [u64; 8],
    held: [u64; 8],
    unknown: [u64; 8],
    live: [u64; 3],
    closed: bool,
}
struct Node {
    stored: StoredRequest,
    outcomes: Vec<StoredOutcome>,
}
fn key(contract: &ContentDigest, request: &Request) -> Result<ContentDigest> {
    Ok(ContentDigest::of_bytes(&canonical(&(contract, request))?))
}
fn deserialize<T: serde::de::DeserializeOwned + Serialize>(
    bytes: &[u8],
    content: &str,
) -> Result<T> {
    if ContentDigest::of_bytes(bytes).as_str() != content {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "budget record content differs",
        ));
    }
    let doc = CanonicalDocument::parse(bytes, Limits::default())?;
    let value: T = serde_json::from_slice(doc.bytes())
        .map_err(|_| failure(ErrorCode::IntegrityMismatch, "invalid budget record schema"))?;
    if canonical(&value)? != bytes {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "noncanonical budget record",
        ));
    }
    Ok(value)
}
fn load(connection: &Connection, family: &ContentDigest) -> Result<BTreeMap<ContentDigest, Node>> {
    let mut statement = connection
        .prepare(
            "SELECT id,body,content FROM budget_requests WHERE family=?1 ORDER BY rowid LIMIT 4097",
        )
        .map_err(sql)?;
    let mut rows = statement.query([family.as_str()]).map_err(sql)?;
    let mut nodes = BTreeMap::new();
    while let Some(row) = rows.next().map_err(sql)? {
        if nodes.len() >= 4096 {
            return Err(failure(
                ErrorCode::LimitExceeded,
                "family request bound exceeded",
            ));
        }
        let id: String = row.get(0).map_err(sql)?;
        let bytes: Vec<u8> = row.get(1).map_err(sql)?;
        let content: String = row.get(2).map_err(sql)?;
        let stored: StoredRequest = deserialize(&bytes, &content)?;
        if stored.schema_version != 1
            || stored.id.as_str() != id
            || stored.family != *family
            || key(&stored.contract, &stored.request)? != stored.id
        {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "budget request identity differs",
            ));
        }
        let mut outcomes = Vec::new();
        let mut query=connection.prepare("SELECT id,body,content FROM budget_outcomes WHERE request=?1 ORDER BY version LIMIT 17").map_err(sql)?;
        let mut results = query.query([id]).map_err(sql)?;
        while let Some(result) = results.next().map_err(sql)? {
            if outcomes.len() >= 16 {
                return Err(failure(
                    ErrorCode::LimitExceeded,
                    "settlement history bound exceeded",
                ));
            }
            let id: String = result.get(0).map_err(sql)?;
            let bytes: Vec<u8> = result.get(1).map_err(sql)?;
            let content: String = result.get(2).map_err(sql)?;
            let outcome: StoredOutcome = deserialize(&bytes, &content)?;
            if outcome.schema_version != 1
                || outcome.id.as_str() != id
                || outcome.request != stored.id
                || outcome.version as usize != outcomes.len() + 1
            {
                return Err(failure(
                    ErrorCode::IntegrityMismatch,
                    "budget settlement ordering differs",
                ));
            }
            outcomes.push(outcome);
        }
        nodes.insert(stored.id.clone(), Node { stored, outcomes });
    }
    Ok(nodes)
}
fn total(
    id: &ContentDigest,
    nodes: &BTreeMap<ContentDigest, Node>,
    path: &mut BTreeSet<ContentDigest>,
    depth: u32,
) -> Result<Total> {
    if depth > 16 || !path.insert(id.clone()) {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "budget hierarchy cycle/depth",
        ));
    }
    let node = nodes
        .get(id)
        .ok_or_else(|| failure(ErrorCode::IntegrityMismatch, "missing parent reservation"))?;
    let cost = node.stored.request.resources.cost();
    let live = node.stored.request.resources.live();
    let last = node.outcomes.last();
    let result = if node.stored.request.kind == WorkKind::Branch {
        let mut result = Total::default();
        let mut allocated = [0; 8];
        let mut active_live = [0; 3];
        let mut all_closed = true;
        for (child, entry) in nodes {
            if entry.stored.request.parent.as_ref() == Some(id) {
                let child = total(child, nodes, path, depth + 1)?;
                result.spent = plus(result.spent, child.spent)?;
                result.unknown = plus(result.unknown, child.unknown)?;
                allocated = plus(allocated, plus(child.spent, child.held)?)?;
                active_live = plus(active_live, child.live)?;
                all_closed &= child.closed;
            }
        }
        if !fits(allocated, cost) || !fits(active_live, live) {
            return Err(failure(
                ErrorCode::QuotaExceeded,
                "children exceed parent allocation",
            ));
        }
        result.closed = last.is_some_and(|outcome| outcome.branch_closed);
        if result.closed {
            if !all_closed {
                return Err(failure(
                    ErrorCode::IntegrityMismatch,
                    "closed branch has unfinished children",
                ));
            }
        } else {
            result.held = subtract(cost, result.spent)?;
            result.live = live;
        }
        result
    } else {
        if nodes
            .values()
            .any(|entry| entry.stored.request.parent.as_ref() == Some(id))
        {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "work leaf cannot own a branch",
            ));
        }
        if let Some(last) = last.filter(|outcome| outcome.settlement.resources_released) {
            let (spent, unknown) = last
                .settlement
                .usage
                .charged(&node.stored.request.resources)?;
            Total {
                spent,
                unknown,
                closed: true,
                ..Total::default()
            }
        } else {
            Total {
                held: cost,
                live,
                ..Total::default()
            }
        }
    };
    path.remove(id);
    Ok(result)
}
fn summarize(nodes: &BTreeMap<ContentDigest, Node>) -> Result<(Total, u32)> {
    let mut result = Total::default();
    let mut roots = 0;
    let mut visited = BTreeSet::new();
    for (id, node) in nodes {
        if node.stored.request.parent.is_none() {
            let root = total(id, nodes, &mut BTreeSet::new(), 0)?;
            result.spent = plus(result.spent, root.spent)?;
            result.held = plus(result.held, root.held)?;
            result.unknown = plus(result.unknown, root.unknown)?;
            result.live = plus(result.live, root.live)?;
            if !root.closed {
                roots += 1;
            }
        } else if node
            .stored
            .request
            .parent
            .as_ref()
            .is_some_and(|parent| !nodes.contains_key(parent))
        {
            return Err(failure(ErrorCode::IntegrityMismatch, "missing branch"));
        }
    }
    // Every node must reach a root, not merely point into a disconnected cycle.
    for id in nodes.keys() {
        let mut current = id;
        let mut path = BTreeSet::new();
        loop {
            if !path.insert(current) || path.len() > 17 {
                return Err(failure(
                    ErrorCode::IntegrityMismatch,
                    "disconnected budget cycle/depth",
                ));
            }
            visited.insert(current);
            match &nodes[current].stored.request.parent {
                Some(parent) => current = parent,
                None => break,
            }
        }
    }
    if visited.len() != nodes.len() {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "unreachable budget node",
        ));
    }
    Ok((result, roots))
}
fn policy(contract: &StudyContract, request: &Request, stage: Stage) -> Result<()> {
    if request.schema_version != 1
        || request.wall_ms == 0
        || request.wall_ms > contract.limits.wall_ms
        || request.attempt > contract.limits.retries_per_operation
        || request.resources.workers == 0
        || request.resources.host_bytes == 0
        || stage.terminal()
    {
        return Err(failure(
            ErrorCode::InvalidValue,
            "bounded request in an active workflow required",
        ));
    }
    let caps = Resources::caps(&contract.limits);
    if !fits(request.resources.cost(), caps.cost()) || !fits(request.resources.live(), caps.live())
    {
        return Err(failure(
            ErrorCode::QuotaExceeded,
            "request exceeds frozen contract cap",
        ));
    }
    if matches!(stage, Stage::Verify | Stage::Learn)
        && matches!(
            request.kind,
            WorkKind::Candidate
                | WorkKind::Proposal
                | WorkKind::Search
                | WorkKind::Training
                | WorkKind::Alignment
        )
    {
        return Err(failure(
            ErrorCode::InvalidTransition,
            "final feedback cannot reopen candidate work",
        ));
    }
    let tool = match request.kind {
        WorkKind::Inspection => Some(ToolKind::Inspect),
        WorkKind::Candidate => Some(ToolKind::DirectEdit),
        WorkKind::Training | WorkKind::Alignment => Some(ToolKind::Training),
        _ => None,
    };
    if let Some(tool) = tool {
        contract.permissions.check_tool(tool)?;
    }
    if request.kind == WorkKind::Candidate && request.resources.attempts != 1 {
        return Err(failure(
            ErrorCode::InvalidValue,
            "each admitted candidate consumes one attempt",
        ));
    }
    if let Some(external) = &request.external {
        contract
            .permissions
            .check_external(&external.destination, &external.payloads)?;
        let price = external.maximum_price_microusd.ok_or_else(|| {
            failure(
                ErrorCode::UnsupportedCapability,
                "external request lacks a finite maximum price",
            )
        })?;
        if external.maximum_tokens == 0
            || external.maximum_tokens
                > request
                    .resources
                    .reasoning_tokens
                    .checked_add(request.resources.teacher_tokens)
                    .ok_or_else(|| failure(ErrorCode::LimitExceeded, "token bound overflow"))?
        {
            return Err(failure(
                ErrorCode::QuotaExceeded,
                "finite external token bound was not reserved",
            ));
        }
        if price > request.resources.spend_microusd {
            return Err(failure(
                ErrorCode::QuotaExceeded,
                "maximum external price was not reserved",
            ));
        }
        if !contract.sources.iter().any(|source| {
            source.artifact == external.source
                && source.destination.as_deref() == Some(external.destination.as_str())
                && external.payloads.is_subset(&source.payloads)
        }) {
            return Err(failure(
                ErrorCode::PermissionDenied,
                "external source/payload not in contract",
            ));
        }
        if request.kind == WorkKind::Teacher && request.resources.teacher_tokens == 0 {
            return Err(failure(
                ErrorCode::PermissionDenied,
                "teacher token allowance missing",
            ));
        }
    } else if request.resources.spend_microusd != 0 {
        return Err(failure(
            ErrorCode::InvalidValue,
            "external spend requires an exact external source request",
        ));
    }
    if request.kind == WorkKind::Branch && request.external.is_some() {
        return Err(failure(
            ErrorCode::InvalidValue,
            "allocation is not an external call",
        ));
    }
    Ok(())
}
fn faults(connection: &Connection, family: &ContentDigest) -> Result<Vec<OverrunReport>> {
    let mut statement = connection
        .prepare("SELECT body,content FROM budget_faults WHERE family=?1 LIMIT 257")
        .map_err(sql)?;
    let mut rows = statement.query([family.as_str()]).map_err(sql)?;
    let mut result = Vec::new();
    while let Some(row) = rows.next().map_err(sql)? {
        if result.len() >= 256 {
            return Err(failure(ErrorCode::LimitExceeded, "breach report bound"));
        }
        let bytes: Vec<u8> = row.get(0).map_err(sql)?;
        let content: String = row.get(1).map_err(sql)?;
        result.push(deserialize(&bytes, &content)?);
    }
    Ok(result)
}
fn elapsed(connection: &Connection, family: &ContentDigest, now: &files::Moment) -> Result<u64> {
    let epoch: Option<(String, i64)> = connection
        .query_row(
            "SELECT boot,start_ms FROM budget_epochs WHERE family=?1",
            [family.as_str()],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()
        .map_err(sql)?;
    match epoch {
        None => Ok(0),
        Some((boot, start)) if boot == now.boot && start <= now.ms => Ok((now.ms - start) as u64),
        _ => Err(failure(
            ErrorCode::UnsupportedCapability,
            "family elapsed time unavailable after boot/clock change; budget not reset",
        )),
    }
}
fn affordability(
    connection: &Connection,
    contract: &StudyContract,
    nodes: &BTreeMap<ContentDigest, Node>,
    now: &files::Moment,
) -> Result<(Total, u32, u64)> {
    if !faults(connection, &contract.study_family)?.is_empty() {
        return Err(failure(
            ErrorCode::BudgetBreach,
            "reported overrun halted this family; revisions cannot erase it",
        ));
    }
    let (total, roots) = summarize(nodes)?;
    let caps = Resources::caps(&contract.limits);
    let wall = elapsed(connection, &contract.study_family, now)?;
    if wall >= contract.limits.wall_ms
        || !fits(plus(total.spent, total.held)?, caps.cost())
        || !fits(total.live, caps.live())
    {
        return Err(failure(
            ErrorCode::QuotaExceeded,
            "family cumulative/reserved/concurrent/deadline cap exhausted",
        ));
    }
    Ok((total, roots, wall))
}
pub(super) struct NativeAllocation {
    pub resources: Resources,
    pub deadline_ms: i64,
}
pub(super) fn broker_pool(
    connection: &Connection,
    store: &str,
    lease: &Lease,
    reservation: &Reservation,
) -> Result<NativeAllocation> {
    let contract = load_contract(connection, &lease.contract)?;
    if reservation.store != store || reservation.family != contract.study_family {
        return Err(failure(ErrorCode::StaleLease, "foreign broker allocation"));
    }
    let nodes = load(connection, &contract.study_family)?;
    let node = nodes
        .get(&reservation.id)
        .ok_or_else(|| failure(ErrorCode::SourceMismatch, "broker allocation absent"))?;
    let now = files::moment()?;
    affordability(connection, &contract, &nodes, &now)?;
    if node.stored.contract != lease.contract
        || node.stored.request.kind != WorkKind::Branch
        || node
            .outcomes
            .last()
            .is_some_and(|outcome| outcome.branch_closed)
        || node.stored.boot != now.boot
        || now.ms >= node.stored.deadline_ms
    {
        return Err(failure(
            ErrorCode::PermissionDenied,
            "broker allocation unavailable",
        ));
    }
    Ok(NativeAllocation {
        resources: node.stored.request.resources.clone(),
        deadline_ms: node.stored.deadline_ms,
    })
}
pub(super) fn native_allocation(
    connection: &Connection,
    store: &str,
    lease: &Lease,
    reservation: &Reservation,
) -> Result<NativeAllocation> {
    let contract = load_contract(connection, &lease.contract)?;
    if reservation.store != store || reservation.family != contract.study_family {
        return Err(failure(ErrorCode::StaleLease, "foreign reservation"));
    }
    let nodes = load(connection, &contract.study_family)?;
    let node = nodes
        .get(&reservation.id)
        .ok_or_else(|| failure(ErrorCode::SourceMismatch, "reservation absent"))?;
    let now = files::moment()?;
    affordability(connection, &contract, &nodes, &now)?;
    if node.stored.contract != lease.contract
        || node.stored.request.kind == WorkKind::Branch
        || node.stored.request.external.is_some()
        || !node.outcomes.is_empty()
        || node.stored.boot != now.boot
        || now.ms >= node.stored.deadline_ms
    {
        return Err(failure(
            ErrorCode::PermissionDenied,
            "reservation cannot dispatch a fresh trusted native job",
        ));
    }
    Ok(NativeAllocation {
        resources: node.stored.request.resources.clone(),
        deadline_ms: node.stored.deadline_ms,
    })
}
pub(super) fn contract_clean(connection: &Connection, contract: &ContentDigest) -> Result<bool> {
    contract_clean_except(connection, contract, None)
}
/// Finalization retains its own allocation until the decision commit has
/// completed; every OTHER operation must already have confirmed cleanup.
pub(super) fn contract_clean_except(
    connection: &Connection,
    contract: &ContentDigest,
    own: Option<&Reservation>,
) -> Result<bool> {
    let body = load_contract(connection, contract)?;
    let nodes = load(connection, &body.study_family)?;
    for node in nodes.values().filter(|node| {
        node.stored.contract == *contract && own.is_none_or(|r| r.id != node.stored.id)
    }) {
        if !node
            .outcomes
            .last()
            .is_some_and(|outcome| outcome.branch_closed || outcome.settlement.resources_released)
        {
            return Ok(false);
        }
    }
    Ok(true)
}
pub(super) fn native_released(connection: &Connection, id: &ContentDigest) -> Result<bool> {
    let row:Option<(Vec<u8>,String)>=connection.query_row("SELECT body,content FROM budget_outcomes WHERE request=?1 ORDER BY version DESC LIMIT 1",[id.as_str()],|row|Ok((row.get(0)?,row.get(1)?))).optional().map_err(sql)?;
    row.map(|(body, content)| {
        Ok(deserialize::<StoredOutcome>(&body, &content)?
            .settlement
            .resources_released)
    })
    .transpose()
    .map(|value| value.unwrap_or(false))
}
pub(super) fn native_kind(connection: &Connection, id: &ContentDigest) -> Result<WorkKind> {
    let (body, content): (Vec<u8>, String) = connection
        .query_row(
            "SELECT body,content FROM budget_requests WHERE id=?1",
            [id.as_str()],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .map_err(sql)?;
    Ok(deserialize::<StoredRequest>(&body, &content)?.request.kind)
}
/// Fold on the caller's existing transaction; does not open another snapshot.
pub(super) fn snapshot_in(
    connection: &Connection,
    contract_id: &ContentDigest,
) -> Result<Snapshot> {
    let contract = load_contract(connection, contract_id)?;
    let nodes = load(connection, &contract.study_family)?;
    let reported_overruns = faults(connection, &contract.study_family)?;
    let (total, roots) = summarize(&nodes)?;
    let wall = elapsed(connection, &contract.study_family, &files::moment()?).ok();
    Ok(Snapshot {
        family: contract.study_family.clone(),
        spent: Resources::from_parts(total.spent, [0; 3]),
        held: Resources::from_parts(total.held, total.live),
        unknown_charged: Resources::from_parts(total.unknown, [0; 3]),
        elapsed_wall_ms: wall,
        active_roots: roots,
        reservation_count: nodes.len() as u32,
        accounting_halted: !reported_overruns.is_empty(),
        reported_overruns,
        measured_peak_host_bytes: None,
        measured_peak_device_bytes: None,
    })
}
impl Ledger {
    pub fn budget_snapshot(&self, contract_id: &ContentDigest) -> Result<Snapshot> {
        self.files.verify(self.limits.max_database_bytes)?;
        let tx = self.connection.unchecked_transaction().map_err(sql)?;
        snapshot_in(&tx, contract_id)
    }
    pub fn preflight(&self, lease: &Lease, request: &Request) -> Result<Preflight> {
        self.files.verify(self.limits.max_database_bytes)?;
        let now = files::moment()?;
        let tx = self.connection.unchecked_transaction().map_err(sql)?;
        let contract = check_lease(&tx, &self.store, lease, &now)?;
        let run = state(&tx, &lease.contract)?;
        if super::supervision::stopped(&tx, &lease.contract)? {
            return Err(failure(
                ErrorCode::PermissionDenied,
                "persisted stop blocks preflight/dispatch",
            ));
        }
        policy(&contract, request, run.stage)?;
        let mut nodes = load(&tx, &contract.study_family)?;
        let id = key(&lease.contract, request)?;
        if !nodes.contains_key(&id) {
            if let Some(parent) = &request.parent {
                let node = nodes.get(parent).ok_or_else(|| {
                    failure(ErrorCode::SourceMismatch, "parent reservation absent")
                })?;
                if node.stored.contract != lease.contract
                    || node.stored.request.kind != WorkKind::Branch
                    || node
                        .outcomes
                        .last()
                        .is_some_and(|outcome| outcome.branch_closed)
                    || node.stored.boot != now.boot
                    || now.ms >= node.stored.deadline_ms
                    || request.wall_ms > (node.stored.deadline_ms - now.ms) as u64
                {
                    return Err(failure(
                        ErrorCode::PermissionDenied,
                        "parent allocation or deadline unavailable",
                    ));
                }
            }
            nodes.insert(
                id.clone(),
                Node {
                    stored: StoredRequest {
                        schema_version: 1,
                        id,
                        family: contract.study_family.clone(),
                        contract: lease.contract.clone(),
                        request: request.clone(),
                        boot: now.boot.clone(),
                        start_ms: now.ms,
                        deadline_ms: now.ms.checked_add(request.wall_ms as i64).ok_or_else(
                            || failure(ErrorCode::LimitExceeded, "deadline overflow"),
                        )?,
                        epoch: lease.epoch,
                    },
                    outcomes: vec![],
                },
            );
        }
        let (_, _, wall) = affordability(&tx, &contract, &nodes, &now)?;
        if request.wall_ms > contract.limits.wall_ms.saturating_sub(wall) {
            return Err(failure(
                ErrorCode::QuotaExceeded,
                "request deadline exceeds family remainder",
            ));
        }
        // T024/T026 must supply a real qualified supervisor/sandbox. A caller's
        // descriptor or accounting reservation is deliberately not that proof.
        Ok(Preflight {
            accounting_eligible: true,
            dispatch_allowed: false,
            requires_worker: request.kind != WorkKind::Branch,
            unmet_required_limits: if request.kind == WorkKind::Branch {
                vec![]
            } else {
                vec![
                    "qualified_wall_cpu_memory_storage_supervisor".to_owned(),
                    "scoped_worker_authority".to_owned(),
                ]
            },
        })
    }
    pub fn reserve(&mut self, lease: &Lease, request: Request) -> Result<Reservation> {
        self.reserve_with(lease, request, |_, _, _| Ok(()))
    }
    /// Bind native intent in the SAME transaction as admission. The callback
    /// must not execute work; a callback failure rolls admission back too.
    pub(super) fn reserve_with(
        &mut self,
        lease: &Lease,
        request: Request,
        admitted: impl FnOnce(&Transaction<'_>, &Reservation, i64) -> Result<()>,
    ) -> Result<Reservation> {
        self.files.verify(self.limits.max_database_bytes)?;
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sql)?;
        let now = files::moment()?;
        let contract = check_lease(&tx, &self.store, lease, &now)?;
        if super::supervision::stopped(&tx, &lease.contract)? {
            return Err(failure(
                ErrorCode::PermissionDenied,
                "persisted stop blocks reservations",
            ));
        }
        policy(&contract, &request, state(&tx, &lease.contract)?.stage)?;
        let mut nodes = load(&tx, &contract.study_family)?;
        let id = key(&lease.contract, &request)?;
        if let Some(existing) = nodes.values().find(|node| {
            node.stored.request.operation == request.operation
                && node.stored.request.attempt == request.attempt
        }) {
            if existing.stored.id != id {
                return Err(failure(
                    ErrorCode::IdempotencyConflict,
                    "operation attempt already reserved with different input/revision",
                ));
            }
            return Ok(Reservation {
                store: self.store.clone(),
                id,
                family: contract.study_family.clone(),
            });
        }
        if nodes.len() >= 4096 {
            return Err(failure(
                ErrorCode::QuotaExceeded,
                "family reservation count exceeded",
            ));
        }
        let attempts: Vec<_> = nodes
            .values()
            .filter(|node| node.stored.request.operation == request.operation)
            .collect();
        if request.attempt as usize != attempts.len()
            || attempts.iter().any(|node| {
                !node.outcomes.last().is_some_and(|outcome| {
                    outcome.settlement.resources_released || outcome.branch_closed
                })
            })
        {
            return Err(failure(
                ErrorCode::InvalidValue,
                "retries are sequential and cannot reset attempts or overlap unreconciled work",
            ));
        }
        if let Some(parent) = &request.parent {
            let node = nodes
                .get(parent)
                .ok_or_else(|| failure(ErrorCode::SourceMismatch, "parent reservation absent"))?;
            if node.stored.contract != lease.contract
                || node.stored.request.kind != WorkKind::Branch
                || node
                    .outcomes
                    .last()
                    .is_some_and(|outcome| outcome.branch_closed)
            {
                return Err(failure(
                    ErrorCode::PermissionDenied,
                    "parent allocation unavailable for this contract",
                ));
            }
            if node.stored.boot != now.boot
                || now.ms >= node.stored.deadline_ms
                || request.wall_ms >= (node.stored.deadline_ms - now.ms + 1) as u64
            {
                return Err(failure(
                    ErrorCode::QuotaExceeded,
                    "child deadline exceeds parent allocation",
                ));
            }
        }
        let wall = elapsed(&tx, &contract.study_family, &now)?;
        if request.wall_ms > contract.limits.wall_ms.saturating_sub(wall) {
            return Err(failure(
                ErrorCode::QuotaExceeded,
                "request cannot finish within family deadline",
            ));
        }
        let deadline_ms = now
            .ms
            .checked_add(request.wall_ms as i64)
            .ok_or_else(|| failure(ErrorCode::LimitExceeded, "deadline overflow"))?;
        let stored = StoredRequest {
            schema_version: 1,
            id: id.clone(),
            family: contract.study_family.clone(),
            contract: lease.contract.clone(),
            request,
            boot: now.boot.clone(),
            start_ms: now.ms,
            deadline_ms,
            epoch: lease.epoch,
        };
        let body = canonical(&stored)?;
        let content = ContentDigest::of_bytes(&body);
        nodes.insert(
            id.clone(),
            Node {
                stored: stored.clone(),
                outcomes: vec![],
            },
        );
        affordability(&tx, &contract, &nodes, &now)?;
        tx.execute(
            "INSERT OR IGNORE INTO budget_epochs VALUES(?1,?2,?3,?4)",
            params![
                contract.study_family.as_str(),
                now.boot,
                now.ms,
                now.wall_ms
            ],
        )
        .map_err(sql)?;
        tx.execute(
            "INSERT INTO budget_requests VALUES(?1,?2,?3,?4,?5,?6,?7,?8)",
            params![
                id.as_str(),
                contract.study_family.as_str(),
                lease.contract.as_str(),
                stored.request.operation.as_str(),
                stored.request.attempt,
                stored.request.parent.as_ref().map(ContentDigest::as_str),
                body,
                content.as_str()
            ],
        )
        .map_err(sql)?;
        let reservation = Reservation {
            store: self.store.clone(),
            id: id.clone(),
            family: contract.study_family.clone(),
        };
        admitted(&tx, &reservation, deadline_ms)?;
        ensure_live(lease)?;
        let end = files::moment()?;
        if elapsed(&tx, &contract.study_family, &end)? >= contract.limits.wall_ms
            || end.ms >= stored.deadline_ms
        {
            return Err(failure(
                ErrorCode::QuotaExceeded,
                "deadline exhausted before reservation commit",
            ));
        }
        tx.commit().map_err(sql)?;
        Ok(Reservation {
            store: self.store.clone(),
            id,
            family: contract.study_family.clone(),
        })
    }
    pub fn recover_reservation(&self, lease: &Lease, id: &ContentDigest) -> Result<Reservation> {
        self.files.verify(self.limits.max_database_bytes)?;
        let tx = self.connection.unchecked_transaction().map_err(sql)?;
        let contract = check_lease(&tx, &self.store, lease, &files::moment()?)?;
        let nodes = load(&tx, &contract.study_family)?;
        if !nodes.contains_key(id) {
            return Err(failure(
                ErrorCode::SourceMismatch,
                "reservation not in this family",
            ));
        }
        Ok(Reservation {
            store: self.store.clone(),
            id: id.clone(),
            family: contract.study_family.clone(),
        })
    }
    pub fn settle(
        &mut self,
        lease: &Lease,
        reservation: &Reservation,
        settlement: Settlement,
    ) -> Result<()> {
        self.finish_budget(lease, reservation, settlement, false)
    }
    pub fn close_branch(
        &mut self,
        lease: &Lease,
        reservation: &Reservation,
        operation: ContentDigest,
        evidence: ContentDigest,
    ) -> Result<()> {
        self.finish_budget(
            lease,
            reservation,
            Settlement {
                operation,
                outcome: OutcomeKind::Completed,
                usage: Usage::default(),
                resources_released: true,
                evidence,
            },
            true,
        )
    }
    fn finish_budget(
        &mut self,
        lease: &Lease,
        reservation: &Reservation,
        settlement: Settlement,
        branch_closed: bool,
    ) -> Result<()> {
        self.finish_budget_with(lease, reservation, settlement, branch_closed, |_| Ok(()))
    }
    /// Authoritative native completion and resource release share one commit.
    /// The callback runs after settlement serialization/insertion, while the
    /// IMMEDIATE transaction still fences competing owners and stop requests.
    pub(super) fn finish_budget_with(
        &mut self,
        lease: &Lease,
        reservation: &Reservation,
        settlement: Settlement,
        branch_closed: bool,
        publish: impl FnOnce(&Transaction<'_>) -> Result<()>,
    ) -> Result<()> {
        self.files.verify(self.limits.max_database_bytes)?;
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sql)?;
        let contract = check_lease(&tx, &self.store, lease, &files::moment()?)?;
        if reservation.store != self.store || reservation.family != contract.study_family {
            return Err(failure(ErrorCode::StaleLease, "foreign reservation handle"));
        }
        let mut nodes = load(&tx, &contract.study_family)?;
        // Only native cleanup can release an audit hold. All controller/cgroup
        // cost measurements remain partial and cannot refund family allowances.
        audit_budget::validate_settlement(&tx, reservation.id(), &settlement)?;
        let node = nodes
            .get(&reservation.id)
            .ok_or_else(|| failure(ErrorCode::SourceMismatch, "unknown reservation"))?;
        if (node.stored.request.kind == WorkKind::Branch) != branch_closed {
            return Err(failure(
                ErrorCode::InvalidValue,
                "branch closes only from its children's ledger, not caller cost claims",
            ));
        }
        let event_id = ContentDigest::of_bytes(&canonical(&(
            &contract.study_family,
            &settlement.operation,
        ))?);
        if let Some(old) = nodes
            .values()
            .flat_map(|node| &node.outcomes)
            .find(|outcome| outcome.id == event_id)
        {
            if old.request != reservation.id
                || old.settlement != settlement
                || old.branch_closed != branch_closed
            {
                return Err(failure(
                    ErrorCode::IdempotencyConflict,
                    "settlement ID input differs",
                ));
            }
            return Ok(());
        }
        if node.outcomes.len() >= 16 {
            return Err(failure(
                ErrorCode::QuotaExceeded,
                "settlement reconciliation bound exceeded",
            ));
        }
        if branch_closed {
            if node.outcomes.last().is_some() {
                return Err(failure(ErrorCode::InvalidValue, "branch already closed"));
            }
            if nodes
                .iter()
                .filter(|(_, child)| child.stored.request.parent.as_ref() == Some(&reservation.id))
                .any(|(id, _)| {
                    !total(id, &nodes, &mut BTreeSet::new(), 0).is_ok_and(|sum| sum.closed)
                })
            {
                return Err(failure(
                    ErrorCode::PermissionDenied,
                    "branch has unfinished children",
                ));
            }
        } else {
            if !settlement.resources_released
                && (settlement.usage.values().iter().any(Option::is_some)
                    || settlement.outcome != OutcomeKind::Incomplete)
            {
                return Err(failure(
                    ErrorCode::InvalidValue,
                    "unclean work retains its full reservation, never invented actual costs",
                ));
            }
            if settlement
                .usage
                .charged(&node.stored.request.resources)
                .is_err()
            {
                let report = OverrunReport {
                    schema_version: 1,
                    request: reservation.id.clone(),
                    contract: node.stored.contract.clone(),
                    ceiling: node.stored.request.resources.clone(),
                    settlement: settlement.clone(),
                };
                let body = canonical(&report)?;
                let digest = ContentDigest::of_bytes(&body);
                let prior: Option<Vec<u8>> = tx
                    .query_row(
                        "SELECT body FROM budget_faults WHERE id=?1",
                        [event_id.as_str()],
                        |row| row.get(0),
                    )
                    .optional()
                    .map_err(sql)?;
                if prior.as_ref().is_some_and(|prior| prior != &body) {
                    return Err(failure(
                        ErrorCode::IdempotencyConflict,
                        "breach operation ID contents differ",
                    ));
                }
                if prior.is_none() {
                    if faults(&tx, &contract.study_family)?.len() >= 256 {
                        return Err(failure(ErrorCode::LimitExceeded, "breach report bound"));
                    }
                    tx.execute(
                        "INSERT INTO budget_faults VALUES(?1,?2,?3,?4)",
                        params![
                            event_id.as_str(),
                            contract.study_family.as_str(),
                            body,
                            digest.as_str()
                        ],
                    )
                    .map_err(sql)?;
                }
                ensure_live(lease)?;
                tx.commit().map_err(sql)?;
                return Err(failure(ErrorCode::BudgetBreach,"actual exceeds reservation: report retained, family dispatch halted, reservation not released"));
            }
            if node.stored.request.kind == WorkKind::Candidate
                && settlement.usage.attempts.is_some_and(|n| n != 1)
            {
                return Err(failure(
                    ErrorCode::InvalidValue,
                    "admitted failed/cancelled candidate attempt cannot be refunded",
                ));
            }
            if let Some(prior) = node.outcomes.last() {
                if prior.settlement.resources_released && !settlement.resources_released {
                    return Err(failure(
                        ErrorCode::InvalidValue,
                        "released job cannot be reopened",
                    ));
                }
                for (old, new) in prior
                    .settlement
                    .usage
                    .values()
                    .into_iter()
                    .zip(settlement.usage.values())
                {
                    if old.is_some() && old != new {
                        return Err(failure(
                            ErrorCode::IntegrityMismatch,
                            "known measured cost cannot be rewritten or turned unknown",
                        ));
                    }
                }
            }
        }
        let event = StoredOutcome {
            schema_version: 1,
            id: event_id.clone(),
            request: reservation.id.clone(),
            version: node.outcomes.len() as u32 + 1,
            branch_closed,
            settlement,
        };
        nodes
            .get_mut(&reservation.id)
            .unwrap()
            .outcomes
            .push(event.clone());
        summarize(&nodes)?;
        let body = canonical(&event)?;
        tx.execute(
            "INSERT INTO budget_outcomes VALUES(?1,?2,?3,?4,?5)",
            params![
                event_id.as_str(),
                reservation.id.as_str(),
                event.version,
                body,
                ContentDigest::of_bytes(&body).as_str()
            ],
        )
        .map_err(sql)?;
        publish(&tx)?;
        ensure_live(lease)?;
        tx.commit().map_err(sql)?;
        Ok(())
    }
}
