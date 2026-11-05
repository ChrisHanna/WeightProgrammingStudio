//! Exact local disclosure history, conservative aliases and restricted legacy intake.
#[cfg(test)]
mod tests;
use super::splits::{Dataset, Item};
use super::*;
use nf_checkpoint::store::ArtifactStore;
use nf_core::study::{ClaimScope, SplitRole, TaskIdentity};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

pub(super) const MIGRATION_V5: &str = r#"
CREATE TABLE exposure_subjects(family TEXT NOT NULL,id TEXT NOT NULL,body BLOB NOT NULL,PRIMARY KEY(family,id));
CREATE TABLE exposure_events(family TEXT NOT NULL,sequence INTEGER NOT NULL,operation TEXT NOT NULL,stage TEXT NOT NULL,body BLOB NOT NULL,id TEXT NOT NULL UNIQUE,PRIMARY KEY(family,sequence),UNIQUE(family,operation,stage));
CREATE TABLE legacy_intakes(family TEXT NOT NULL,id TEXT NOT NULL,body BLOB NOT NULL,PRIMARY KEY(family,id));
CREATE TABLE exposure_gaps(family TEXT PRIMARY KEY,reason TEXT NOT NULL);
INSERT INTO exposure_gaps SELECT DISTINCT c.family,'pre-schema5 activity has no exact authenticated exposure history' FROM contracts c WHERE EXISTS(SELECT 1 FROM split_grants g WHERE g.contract=c.digest) OR EXISTS(SELECT 1 FROM budget_requests b WHERE b.contract=c.digest);
CREATE TRIGGER exposure_subjects_no_update BEFORE UPDATE ON exposure_subjects BEGIN SELECT RAISE(ABORT,'immutable exposure identity');END;
CREATE TRIGGER exposure_subjects_no_delete BEFORE DELETE ON exposure_subjects BEGIN SELECT RAISE(ABORT,'retained exposure identity');END;
CREATE TRIGGER exposure_events_no_update BEFORE UPDATE ON exposure_events BEGIN SELECT RAISE(ABORT,'immutable exposure');END;
CREATE TRIGGER exposure_events_no_delete BEFORE DELETE ON exposure_events BEGIN SELECT RAISE(ABORT,'retained exposure');END;
CREATE TRIGGER legacy_no_update BEFORE UPDATE ON legacy_intakes BEGIN SELECT RAISE(ABORT,'immutable legacy history');END;
CREATE TRIGGER legacy_no_delete BEFORE DELETE ON legacy_intakes BEGIN SELECT RAISE(ABORT,'retained legacy history');END;
CREATE TRIGGER exposure_gaps_no_update BEFORE UPDATE ON exposure_gaps BEGIN SELECT RAISE(ABORT,'retained unknown exposure');END;
CREATE TRIGGER exposure_gaps_no_delete BEFORE DELETE ON exposure_gaps BEGIN SELECT RAISE(ABORT,'retained unknown exposure');END;
INSERT INTO budget_migrations VALUES(5,(SELECT value FROM meta WHERE key='schema'));
PRAGMA user_version=5;
"#;
pub(super) fn schema_v5() -> ContentDigest {
    ContentDigest::of_bytes(
        &[
            SCHEMA.as_bytes(),
            budgets::MIGRATION_V2.as_bytes(),
            supervision::MIGRATION_V3.as_bytes(),
            splits::MIGRATION_V4.as_bytes(),
            MIGRATION_V5.as_bytes(),
        ]
        .concat(),
    )
}
#[cfg(test)]
pub(super) const STRIP_V5:&str="DROP TABLE exposure_events;DROP TABLE exposure_subjects;DROP TABLE legacy_intakes;DROP TABLE exposure_gaps;DELETE FROM budget_migrations WHERE version=5;";
fn decode<T: serde::de::DeserializeOwned + Serialize>(bytes: &[u8], id: &str) -> Result<T> {
    if bytes.len() > 1048576 || ContentDigest::of_bytes(bytes).as_str() != id {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "exposure record integrity mismatch",
        ));
    }
    let doc = CanonicalDocument::parse(bytes, Limits::default())?;
    let record: T = serde_json::from_slice(doc.bytes())
        .map_err(|_| failure(ErrorCode::InvalidValue, "invalid exposure record"))?;
    if canonical(&record)? != bytes {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "noncanonical exposure record",
        ));
    }
    Ok(record)
}
/// A conservative alias key, not a proof of semantic equivalence/independence.
/// Reserved transport identifiers are omitted recursively. Seed is NOT omitted.
pub fn identity_protocol() -> ContentDigest {
    ContentDigest::of_bytes(
        b"nf.conservative-json-identity.v1;recursive-transport-key-omission;seed-retained",
    )
}
pub fn identity(value: &serde_json::Value) -> Result<ContentDigest> {
    fn strip(v: &serde_json::Value) -> serde_json::Value {
        match v {
            serde_json::Value::Object(m) => serde_json::Value::Object(
                m.iter()
                    .filter(|(k, _)| {
                        !matches!(
                            k.as_str(),
                            "task"
                                | "task_id"
                                | "item"
                                | "item_id"
                                | "group"
                                | "group_id"
                                | "split"
                                | "role"
                                | "name"
                                | "version"
                                | "nonce"
                        )
                    })
                    .map(|(k, v)| (k.clone(), strip(v)))
                    .collect(),
            ),
            serde_json::Value::Array(a) => serde_json::Value::Array(a.iter().map(strip).collect()),
            other => other.clone(),
        }
    }
    Ok(ContentDigest::of_bytes(&canonical(&strip(value))?))
}
fn read_payload(store: &ArtifactStore, id: &ContentDigest) -> Result<serde_json::Value> {
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
            "canonical task input required",
        ));
    }
    Ok(doc.value().clone())
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Subject {
    pub contract: ContentDigest,
    pub binding: ContentDigest,
    pub package_version: ContentDigest,
    pub dataset_version: ContentDigest,
    pub task: TaskIdentity,
    pub input: ContentDigest,
    pub semantic: ContentDigest,
    pub identity_protocol: ContentDigest,
    pub label: ContentDigest,
    pub split: Option<SplitRole>,
    pub legacy: bool,
}
impl Subject {
    fn id(&self) -> Result<ContentDigest> {
        Ok(ContentDigest::of_bytes(&canonical(self)?))
    }
}
pub(super) fn subject(
    contract: &StudyContract,
    id: &ContentDigest,
    binding: &ContentDigest,
    role: SplitRole,
    item: &Item,
    store: &ArtifactStore,
) -> Result<Subject> {
    let selected = contract
        .portfolio
        .bindings
        .iter()
        .find(|b| b.id == *binding)
        .ok_or_else(|| failure(ErrorCode::SourceMismatch, "unknown exposure binding"))?;
    let split = match role {
        SplitRole::Development => &selected.splits.development,
        SplitRole::Selection => &selected.splits.selection,
        SplitRole::Holdout => &selected.splits.holdout,
    };
    Ok(Subject {
        contract: id.clone(),
        binding: binding.clone(),
        package_version: selected.gym.clone(),
        dataset_version: split.dataset.clone(),
        task: item.identity.clone(),
        input: item.input.clone(),
        semantic: identity(&read_payload(store, &item.input)?)?,
        identity_protocol: identity_protocol(),
        label: item.label.clone(),
        split: Some(role),
        legacy: false,
    })
}
pub(super) fn resolved_subject(
    connection: &Connection,
    contract: &StudyContract,
    id: &ContentDigest,
    binding: &ContentDigest,
    role: SplitRole,
    item: &Item,
    store: &ArtifactStore,
) -> Result<Subject> {
    if let Some(subject) = subjects(connection, &contract.study_family)?
        .into_iter()
        .find(|s| {
            s.contract == *id
                && s.binding == *binding
                && s.task.item == item.identity.item
                && s.input == item.input
                && s.split == Some(role)
                && !s.legacy
        })
    {
        return Ok(subject);
    }
    subject(contract, id, binding, role, item, store)
}
fn subjects(connection: &Connection, family: &ContentDigest) -> Result<Vec<Subject>> {
    Ok(subjects_with_ids(connection, family)?
        .into_iter()
        .map(|(subject, _)| subject)
        .collect())
}
fn subjects_with_ids(
    connection: &Connection,
    family: &ContentDigest,
) -> Result<Vec<(Subject, ContentDigest)>> {
    std::thread_local! {
        static CACHE: std::cell::RefCell<BTreeMap<(String, Vec<u8>), (Subject, ContentDigest)>> =
            const { std::cell::RefCell::new(BTreeMap::new()) };
    }
    let mut statement = connection
        .prepare("SELECT body,id FROM exposure_subjects WHERE family=?1")
        .map_err(sql)?;
    let rows = statement
        .query_map([family.as_str()], |r| {
            Ok((r.get::<_, Vec<u8>>(0)?, r.get::<_, String>(1)?))
        })
        .map_err(sql)?
        .take(8193)
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(sql)?;
    if rows.len() > 8192 {
        return Err(failure(
            ErrorCode::LimitExceeded,
            "exposure identity registry bound exceeded",
        ));
    }
    CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache.len() > 4096 {
            cache.clear();
        }
        rows.into_iter()
            .map(|(b, id)| {
                let key = (id.clone(), b);
                if let Some(entry) = cache.get(&key) {
                    return Ok(entry.clone());
                }
                let subject: Subject = decode(&key.1, &id)?;
                let digest: ContentDigest = id.parse()?;
                let entry = (subject, digest);
                cache.insert(key, entry.clone());
                Ok(entry)
            })
            .collect()
    })
}
fn linked(all: &[(Subject, ContentDigest)], target: &Subject) -> BTreeSet<usize> {
    let mut groups: BTreeMap<&ContentDigest, Vec<usize>> = BTreeMap::new();
    let mut semantics: BTreeMap<&ContentDigest, Vec<usize>> = BTreeMap::new();
    let mut items: BTreeMap<&ContentDigest, Vec<usize>> = BTreeMap::new();
    for (i, (s, _)) in all.iter().enumerate() {
        groups.entry(&s.task.group).or_default().push(i);
        semantics.entry(&s.semantic).or_default().push(i);
        items.entry(&s.task.item).or_default().push(i);
    }
    let mut queue = VecDeque::from([target]);
    let mut seen = BTreeSet::new();
    let mut g = BTreeSet::new();
    let mut v = BTreeSet::new();
    let mut t = BTreeSet::new();
    while let Some(s) = queue.pop_front() {
        for (map, key, done) in [
            (&groups, &s.task.group, &mut g),
            (&semantics, &s.semantic, &mut v),
            (&items, &s.task.item, &mut t),
        ] {
            if done.insert(key) {
                if let Some(indices) = map.get(key) {
                    for i in indices {
                        if seen.insert(*i) {
                            queue.push_back(&all[*i].0);
                        }
                    }
                }
            }
        }
    }
    seen
}
fn register(connection: &Connection, family: &ContentDigest, subject: &Subject) -> Result<()> {
    let body = canonical(subject)?;
    let id = ContentDigest::of_bytes(&body);
    let count: i64 = connection
        .query_row(
            "SELECT count(*) FROM exposure_subjects WHERE family=?1",
            [family.as_str()],
            |r| r.get(0),
        )
        .map_err(sql)?;
    if count >= 8192 {
        return Err(failure(
            ErrorCode::QuotaExceeded,
            "exposure subject bound exceeded",
        ));
    }
    connection
        .execute(
            "INSERT OR IGNORE INTO exposure_subjects VALUES(?1,?2,?3)",
            params![family.as_str(), id.as_str(), body],
        )
        .map_err(sql)?;
    Ok(())
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Field {
    Identity,
    Input,
    Label,
    Output,
    Feedback,
    Summary,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    Requested,
    Started,
    Completed,
    Failed,
    Incomplete,
    Imported,
}
impl Stage {
    fn text(self) -> &'static str {
        match self {
            Self::Requested => "requested",
            Self::Started => "started",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Incomplete => "incomplete",
            Self::Imported => "imported",
        }
    }
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub seed: Option<u64>,
    pub artifact: Option<ContentDigest>,
    pub model: Option<ContentDigest>,
    pub programmer: Option<ContentDigest>,
    pub teacher: Option<ContentDigest>,
    pub candidate: Option<ContentDigest>,
    pub protocol: Option<ContentDigest>,
    pub attempt: u32,
    pub retry_of: Option<ContentDigest>,
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cost {
    pub cpu_ms: Option<u64>,
    pub model_calls: Option<u64>,
    pub reasoning_tokens: Option<u64>,
    pub teacher_tokens: Option<u64>,
    pub tool_calls: Option<u64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventBody {
    pub schema_version: u32,
    pub sequence: u64,
    pub previous: Option<ContentDigest>,
    pub family: ContentDigest,
    pub contract: ContentDigest,
    pub revision: u32,
    pub operation: ContentDigest,
    pub subject: Subject,
    pub recipient: String,
    pub role: String,
    pub context: Context,
    pub stage: Stage,
    pub intended: BTreeSet<Field>,
    pub shown: BTreeSet<Field>,
    pub availability: String,
    pub outcome: String,
    pub cost: Cost,
    pub parent: Option<ContentDigest>,
    pub epoch: i64,
    pub boot: String,
    pub monotonic_ms: i64,
    pub wall_ms: i64,
    pub deadline: i64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Event {
    pub id: ContentDigest,
    pub sequence: u64,
    pub body: EventBody,
}
/// Native-only disclosure ticket, never deserializable or passed to a worker.
#[derive(Clone)]
pub(super) struct Ticket {
    pub body: EventBody,
    pub started: ContentDigest,
}
fn append(connection: &Connection, body: &EventBody) -> Result<Event> {
    let prior:Option<(Vec<u8>,String,u64)>=connection.query_row("SELECT body,id,sequence FROM exposure_events WHERE family=?1 AND operation=?2 AND stage=?3",params![body.family.as_str(),body.operation.as_str(),body.stage.text()],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional().map_err(sql)?;
    if let Some((bytes, id, sequence)) = prior {
        let prior: EventBody = decode(&bytes, &id)?;
        let mut stable = body.clone();
        stable.sequence = prior.sequence;
        stable.previous = prior.previous.clone();
        stable.boot = prior.boot.clone();
        stable.monotonic_ms = prior.monotonic_ms;
        stable.wall_ms = prior.wall_ms;
        if canonical(&stable)? != canonical(&prior)? {
            return Err(failure(
                ErrorCode::IdempotencyConflict,
                "exposure operation reused with changed evidence",
            ));
        }
        return Ok(Event {
            id: id.parse()?,
            sequence,
            body: prior,
        });
    }
    let sequence: u64 = connection
        .query_row(
            "SELECT count(*) FROM exposure_events WHERE family=?1",
            [body.family.as_str()],
            |r| r.get(0),
        )
        .map_err(sql)?;
    if sequence >= 100000 {
        return Err(failure(
            ErrorCode::QuotaExceeded,
            "exposure journal bound exceeded",
        ));
    }
    let previous: Option<String> = connection
        .query_row(
            "SELECT id FROM exposure_events WHERE family=?1 ORDER BY sequence DESC LIMIT 1",
            [body.family.as_str()],
            |r| r.get(0),
        )
        .optional()
        .map_err(sql)?;
    let mut body = body.clone();
    body.sequence = sequence + 1;
    body.previous = previous.map(|s| s.parse()).transpose()?;
    let bytes = canonical(&body)?;
    let id = ContentDigest::of_bytes(&bytes);
    connection
        .execute(
            "INSERT INTO exposure_events VALUES(?1,?2,?3,?4,?5,?6)",
            params![
                body.family.as_str(),
                sequence + 1,
                body.operation.as_str(),
                body.stage.text(),
                bytes,
                id.as_str()
            ],
        )
        .map_err(sql)?;
    Ok(Event {
        id,
        sequence: sequence + 1,
        body,
    })
}
fn events(connection: &Connection, family: &ContentDigest) -> Result<Vec<Event>> {
    Ok(events_with_subject_ids(connection, family)?
        .into_iter()
        .map(|(event, _)| event)
        .collect())
}
fn events_with_subject_ids(
    connection: &Connection,
    family: &ContentDigest,
) -> Result<Vec<(Event, ContentDigest)>> {
    std::thread_local! {
        static CACHE: std::cell::RefCell<BTreeMap<(String, Vec<u8>), (EventBody, ContentDigest)>> =
            const { std::cell::RefCell::new(BTreeMap::new()) };
    }
    let mut statement = connection
        .prepare("SELECT body,id,sequence FROM exposure_events WHERE family=?1 ORDER BY sequence")
        .map_err(sql)?;
    let data = statement
        .query_map([family.as_str()], |r| {
            Ok((
                r.get::<_, Vec<u8>>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, u64>(2)?,
            ))
        })
        .map_err(sql)?
        .take(100001)
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(sql)?;
    if data.len() > 100000 {
        return Err(failure(
            ErrorCode::LimitExceeded,
            "exposure journal bound exceeded",
        ));
    }
    let mut previous = None;
    let mut result = Vec::new();
    CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache.len() > 4096 {
            cache.clear();
        }
        for (b, id, sequence) in data {
            let key = (id.clone(), b);
            let (body, subject_id) = if let Some(entry) = cache.get(&key) {
                entry.clone()
            } else {
                let decoded: EventBody = decode(&key.1, &id)?;
                let subject_id = decoded.subject.id()?;
                let entry = (decoded, subject_id);
                cache.insert(key, entry.clone());
                entry
            };
            if sequence != result.len() as u64 + 1
                || body.sequence != sequence
                || body.previous != previous
                || body.family != *family
            {
                return Err(failure(
                    ErrorCode::IntegrityMismatch,
                    "exposure sequence/hash chain mismatch",
                ));
            }
            let id: ContentDigest = id.parse()?;
            previous = Some(id.clone());
            result.push((Event { id, sequence, body }, subject_id));
        }
        Ok(result)
    })
}
pub(super) struct Start {
    pub subject: Subject,
    pub recipient: String,
    pub role: String,
    pub operation: ContentDigest,
    pub context: Context,
    pub fields: BTreeSet<Field>,
}
pub(super) fn begin(
    connection: &Connection,
    lease: &Lease,
    contract: &StudyContract,
    start: Start,
) -> Result<Ticket> {
    let Start {
        subject,
        recipient,
        role,
        operation,
        context,
        fields,
    } = start;
    if recipient.is_empty() || recipient.len() > 256 || fields.is_empty() || context.attempt > 10000
    {
        return Err(failure(
            ErrorCode::InvalidValue,
            "bounded explicit exposure recipient/fields/context required",
        ));
    }
    if let Some(parent) = &context.retry_of {
        let source: Option<(Vec<u8>, String)> = connection
            .query_row(
                "SELECT body,id FROM exposure_events WHERE family=?1 AND id=?2",
                params![contract.study_family.as_str(), parent.as_str()],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()
            .map_err(sql)?;
        let (body, id) = source.ok_or_else(|| {
            failure(
                ErrorCode::SourceMismatch,
                "retry reference must resolve real family exposure history",
            )
        })?;
        let prior: EventBody = decode(&body, &id)?;
        if prior.subject.semantic != subject.semantic {
            return Err(failure(
                ErrorCode::SourceMismatch,
                "retry cannot invent a different task identity",
            ));
        }
    }
    register(connection, &contract.study_family, &subject)?;
    let now = files::moment()?;
    let body = EventBody {
        schema_version: 1,
        sequence: 0,
        previous: None,
        family: contract.study_family.clone(),
        contract: lease.contract.clone(),
        revision: contract.revision,
        operation,
        subject,
        recipient,
        role,
        context,
        stage: Stage::Requested,
        intended: fields,
        shown: BTreeSet::new(),
        availability: "not_yet_disclosed".into(),
        outcome: "pending".into(),
        cost: Cost::default(),
        parent: None,
        epoch: lease.epoch,
        boot: now.boot,
        monotonic_ms: now.ms,
        wall_ms: now.wall_ms,
        deadline: lease.deadline,
    };
    let requested = append(connection, &body)?;
    let mut body = body;
    body.stage = Stage::Started;
    body.parent = Some(requested.id);
    body.availability = "may_have_been_available;conservative_after_crash".into();
    let started = append(connection, &body)?;
    Ok(Ticket {
        body: started.body,
        started: started.id,
    })
}
pub(super) fn finish(
    connection: &Connection,
    ticket: &Ticket,
    stage: Stage,
    shown: BTreeSet<Field>,
    outcome: &str,
    cost: Cost,
) -> Result<Event> {
    if !matches!(stage, Stage::Completed | Stage::Failed | Stage::Incomplete)
        || !shown.is_subset(&ticket.body.intended)
    {
        return Err(failure(
            ErrorCode::InvalidTransition,
            "invalid disclosure completion",
        ));
    }
    let stored: (Vec<u8>, String) = connection
        .query_row(
            "SELECT body,id FROM exposure_events WHERE id=?1 AND family=?2 AND stage='started'",
            params![ticket.started.as_str(), ticket.body.family.as_str()],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .map_err(sql)?;
    let source: EventBody = decode(&stored.0, &stored.1)?;
    if canonical(&source)? != canonical(&ticket.body)? {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "exposure ticket does not match persisted start",
        ));
    }
    let mut body = ticket.body.clone();
    let now = files::moment()?;
    body.stage = stage;
    body.parent = Some(ticket.started.clone());
    body.shown = shown;
    body.outcome = outcome.into();
    body.cost = cost;
    body.monotonic_ms = now.ms;
    body.wall_ms = now.wall_ms;
    body.availability = if stage == Stage::Completed {
        "available_to_authorized_recipient_route"
    } else {
        "partial_or_unknown_not_fresh_evidence"
    }
    .into();
    append(connection, &body)
}
/// Register exact source identities and enforce local history before split binding.
pub(super) fn bind(
    connection: &Connection,
    lease: &Lease,
    contract: &StudyContract,
    binding: &ContentDigest,
    datasets: &BTreeMap<SplitRole, Dataset>,
    store: &ArtifactStore,
) -> Result<()> {
    for (role, dataset) in datasets {
        for item in &dataset.items {
            let subject = subject(contract, &lease.contract, binding, *role, item, store)?;
            admit(connection, contract, &subject)?;
            register(connection, &contract.study_family, &subject)?;
        }
    }
    Ok(())
}
pub(super) fn admit(
    connection: &Connection,
    contract: &StudyContract,
    subject: &Subject,
) -> Result<()> {
    let gap: bool = connection
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM exposure_gaps WHERE family=?1)",
            [contract.study_family.as_str()],
            |r| r.get(0),
        )
        .map_err(sql)?;
    if gap && subject.split != Some(SplitRole::Development) {
        return Err(failure(
            ErrorCode::PermissionDenied,
            "unknown inherited exposure history blocks a clean selection/holdout scope",
        ));
    }
    if subject.identity_protocol != identity_protocol() {
        return Err(failure(
            ErrorCode::UnsupportedVersion,
            "exposure identity policy version is unresolved",
        ));
    }
    let all = subjects_with_ids(connection, &contract.study_family)?;
    if all
        .iter()
        .any(|(s, _)| s.identity_protocol != identity_protocol())
    {
        return Err(failure(
            ErrorCode::UnsupportedVersion,
            "inherited identity policies require explicit reconciliation",
        ));
    }
    let links = linked(&all, subject);
    let role = subject
        .split
        .ok_or_else(|| failure(ErrorCode::InvalidValue, "declared canonical split required"))?;
    for i in &links {
        let old = &all[*i].0;
        if (old.legacy && role != SplitRole::Development)
            || (!old.legacy && old.split.is_some_and(|prior| prior != role))
        {
            return Err(failure(
                ErrorCode::ScopeViolation,
                "exposure identity/group alias cannot be relabeled as independent split",
            ));
        }
    }
    if role == SplitRole::Selection {
        let linked_ids: BTreeSet<_> = links.iter().map(|i| &all[*i].1).collect();
        for (event, subject_id) in events_with_subject_ids(connection, &contract.study_family)? {
            if linked_ids.contains(&subject_id)
                && event.body.contract != subject.contract
                && event.body.role == "proposer"
                && event
                    .body
                    .intended
                    .iter()
                    .any(|f| matches!(f, Field::Feedback | Field::Summary | Field::Label))
            {
                return Err(failure(
                    ErrorCode::PermissionDenied,
                    "prior selection feedback is now development knowledge, not fresh selection",
                ));
            }
        }
    }
    if role == SplitRole::Holdout {
        let linked_ids: BTreeSet<_> = links.iter().map(|i| &all[*i].1).collect();
        for (event, subject_id) in events_with_subject_ids(connection, &contract.study_family)? {
            if linked_ids.contains(&subject_id)
                && event.body.stage != Stage::Requested
                && (event.body.contract != subject.contract
                    || !matches!(event.body.role.as_str(), "final_evaluator" | "target_model"))
            {
                return Err(failure(
                    ErrorCode::PermissionDenied,
                    "known/partial/summary exposure blocks a fresh holdout claim",
                ));
            }
        }
        if contract.portfolio.independent.scope == ClaimScope::HeldOutGeneralization {
            return Err(failure(ErrorCode::UnsupportedCapability,"clean external unexposed custody is not established by local hashes or declarations"));
        }
    }
    Ok(())
}
pub(super) fn metadata_scope(
    connection: &Connection,
    lease: &Lease,
    contract: &StudyContract,
    binding: &ContentDigest,
    dataset: &Dataset,
    worker: &str,
    recipient: splits::Recipient,
) -> Result<()> {
    let role = dataset.role;
    let known = subjects(connection, &contract.study_family)?;
    for item in &dataset.items {
        let subject = known
            .iter()
            .find(|s| {
                s.contract == lease.contract
                    && s.binding == *binding
                    && s.task.item == item.identity.item
                    && s.input == item.input
                    && s.split == Some(role)
                    && !s.legacy
            })
            .ok_or_else(|| {
                failure(
                    ErrorCode::PermissionDenied,
                    "exact exposure source must resolve before worker metadata disclosure",
                )
            })?;
        admit(connection, contract, subject)?;
        let role_name = serde_json::to_value(recipient)
            .unwrap()
            .as_str()
            .unwrap()
            .to_owned();
        let ticket = begin(
            connection,
            lease,
            contract,
            Start {
                subject: subject.clone(),
                recipient: worker.into(),
                role: role_name,
                operation: ContentDigest::of_bytes(&canonical(&(
                    worker,
                    &item.identity.item,
                    "welcome-metadata",
                ))?),
                context: Context::default(),
                fields: [Field::Identity].into_iter().collect(),
            },
        )?;
        finish(
            connection,
            &ticket,
            Stage::Completed,
            [Field::Identity].into_iter().collect(),
            "worker_metadata_available",
            Cost::default(),
        )?;
    }
    Ok(())
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LegacyObservation {
    pub item: ContentDigest,
    pub recipient: String,
    pub fields: BTreeSet<Field>,
    pub seed: Option<u64>,
    pub attempt: u32,
    #[serde(default)]
    pub retry_of: Option<u32>,
    pub outcome: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LegacyRecord {
    pub schema_version: u32,
    pub edit: Vec<Item>,
    pub sealed: Vec<Item>,
    pub history: Option<Vec<LegacyObservation>>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LegacyAssessment {
    pub schema_version: u32,
    pub source: ContentDigest,
    pub contract: ContentDigest,
    pub legacy: LegacyRecord,
    pub known_exposed: Vec<ContentDigest>,
    pub unknown: Vec<ContentDigest>,
    pub independent_three_way: bool,
    pub allowed_destination: String,
    pub limitations: String,
}
impl LegacyRecord {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let doc = CanonicalDocument::parse(
            bytes,
            Limits {
                max_bytes: 1048576,
                max_depth: 24,
                max_nodes: 10000,
            },
        )?;
        serde_json::from_slice(doc.bytes())
            .map_err(|_| failure(ErrorCode::InvalidValue, "invalid legacy edit/sealed record"))
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Disclosure {
    pub operation: ContentDigest,
    pub binding: ContentDigest,
    pub item: ContentDigest,
    pub split: SplitRole,
    pub recipient: String,
    pub role: splits::Recipient,
    pub fields: BTreeSet<Field>,
    pub artifact: ContentDigest,
    pub context: Context,
}
impl Ledger {
    /// Controller-observed bounded scalar feedback, not public/private export approval.
    pub fn disclose_score(
        &mut self,
        lease: &Lease,
        store: &ArtifactStore,
        request: Disclosure,
    ) -> Result<Event> {
        if request.fields.is_empty()
            || request
                .fields
                .iter()
                .any(|f| !matches!(f, Field::Output | Field::Summary | Field::Feedback))
        {
            return Err(failure(
                ErrorCode::ScopeViolation,
                "score route cannot disclose task inputs/labels",
            ));
        }
        let value = read_payload(store, &request.artifact)?;
        if !value.as_f64().is_some_and(|n| n.is_finite()) {
            return Err(failure(
                ErrorCode::UnsupportedCapability,
                "score disclosure supports only bounded finite scalar artifacts",
            ));
        }
        self.files.verify(self.limits.max_database_bytes)?;
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sql)?;
        let contract = check_lease(&tx, &self.store, lease, &files::moment()?)?;
        let stage = state(&tx, &lease.contract)?.stage;
        if supervision::stopped(&tx, &lease.contract)?
            || stage.terminal()
            || (request.role != splits::Recipient::FinalEvaluator
                && matches!(stage, super::Stage::Verify | super::Stage::Learn))
            || (request.role == splits::Recipient::Proposer && request.split == SplitRole::Holdout)
        {
            return Err(failure(
                ErrorCode::PermissionDenied,
                "private final feedback cannot enter proposal/search",
            ));
        }
        let authorized = match request.role {
            splits::Recipient::Proposer => request.split != SplitRole::Holdout,
            splits::Recipient::DevelopmentEvaluator => request.split == SplitRole::Development,
            splits::Recipient::SelectionEvaluator => request.split == SplitRole::Selection,
            splits::Recipient::FinalEvaluator => {
                request.split == SplitRole::Holdout && stage == super::Stage::Verify
            }
        };
        if !authorized {
            return Err(failure(
                ErrorCode::ScopeViolation,
                "feedback recipient cannot widen split access",
            ));
        }
        if request.role == splits::Recipient::FinalEvaluator {
            splits::require_panel(&tx, &lease.contract)?;
        }
        let subject = subjects(&tx, &contract.study_family)?
            .into_iter()
            .find(|s| {
                s.contract == lease.contract
                    && s.binding == request.binding
                    && s.task.item == request.item
                    && s.split == Some(request.split)
                    && !s.legacy
            })
            .ok_or_else(|| {
                failure(
                    ErrorCode::SourceMismatch,
                    "feedback requires an actual registered task/version",
                )
            })?;
        admit(&tx, &contract, &subject)?;
        let role = serde_json::to_value(request.role)
            .unwrap()
            .as_str()
            .unwrap()
            .to_owned();
        let mut context = request.context;
        context.artifact = Some(request.artifact);
        let ticket = begin(
            &tx,
            lease,
            &contract,
            Start {
                subject,
                recipient: request.recipient,
                role,
                operation: request.operation,
                context,
                fields: request.fields.clone(),
            },
        )?;
        let event = finish(
            &tx,
            &ticket,
            Stage::Completed,
            request.fields,
            "scalar_feedback_available",
            Cost::default(),
        )?;
        ensure_live(lease)?;
        tx.commit().map_err(sql)?;
        self.files.verify(self.limits.max_database_bytes)?;
        Ok(event)
    }
    /// A fresh fenced coordinator can retain crashed old attempts as incomplete;
    /// it cannot convert them into successful disclosure or reset exposure.
    pub fn reconcile_exposures(&mut self, lease: &Lease) -> Result<usize> {
        self.files.verify(self.limits.max_database_bytes)?;
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sql)?;
        let contract = check_lease(&tx, &self.store, lease, &files::moment()?)?;
        let rows = events(&tx, &contract.study_family)?;
        let now = files::moment()?;
        let ended: BTreeSet<_> = rows
            .iter()
            .filter(|e| {
                matches!(
                    e.body.stage,
                    Stage::Completed | Stage::Failed | Stage::Incomplete | Stage::Imported
                )
            })
            .map(|e| e.body.operation.clone())
            .collect();
        let mut count = 0;
        for event in rows
            .iter()
            .filter(|e| e.body.stage == Stage::Started && !ended.contains(&e.body.operation))
        {
            if event.body.boot != now.boot
                || event.body.contract != lease.contract
                || event.body.epoch != lease.epoch
                || event.body.deadline <= now.ms
            {
                let ticket = Ticket {
                    body: event.body.clone(),
                    started: event.id.clone(),
                };
                finish(
                    &tx,
                    &ticket,
                    Stage::Incomplete,
                    BTreeSet::new(),
                    "recovered_crashed_or_expired_attempt;exposure_retained",
                    Cost::default(),
                )?;
                count += 1;
            }
        }
        ensure_live(lease)?;
        tx.commit().map_err(sql)?;
        self.files.verify(self.limits.max_database_bytes)?;
        Ok(count)
    }
    pub fn exposure_events(&self, lease: &Lease, after: u64, limit: usize) -> Result<Vec<Event>> {
        if limit == 0 || limit > 1000 {
            return Err(failure(
                ErrorCode::LimitExceeded,
                "bounded exposure page required",
            ));
        }
        self.files.verify(self.limits.max_database_bytes)?;
        let tx = self.connection.unchecked_transaction().map_err(sql)?;
        let contract = check_lease(&tx, &self.store, lease, &files::moment()?)?;
        let rows = events(&tx, &contract.study_family)?
            .into_iter()
            .filter(|e| e.sequence > after)
            .take(limit)
            .collect();
        ensure_live(lease)?;
        self.files.verify(self.limits.max_database_bytes)?;
        Ok(rows)
    }
    pub(super) fn finish_exposure(
        &self,
        lease: &Lease,
        ticket: &Ticket,
        stage: Stage,
        shown: BTreeSet<Field>,
        outcome: &str,
        cost: Cost,
    ) -> Result<Event> {
        self.files.verify(self.limits.max_database_bytes)?;
        let tx = self.connection.unchecked_transaction().map_err(sql)?;
        let contract = check_lease(&tx, &self.store, lease, &files::moment()?)?;
        if ticket.body.contract != lease.contract
            || ticket.body.family != contract.study_family
            || ticket.body.epoch != lease.epoch
        {
            return Err(failure(
                ErrorCode::StaleLease,
                "exposure ticket belongs to a different contract/store/epoch",
            ));
        }
        let event = finish(&tx, ticket, stage, shown, outcome, cost)?;
        ensure_live(lease)?;
        tx.commit().map_err(sql)?;
        self.files.verify(self.limits.max_database_bytes)?;
        Ok(event)
    }
    pub fn import_legacy_split(
        &mut self,
        lease: &Lease,
        store: &ArtifactStore,
        source_id: &ContentDigest,
        legacy: LegacyRecord,
    ) -> Result<ContentDigest> {
        if legacy.schema_version != 1
            || legacy.edit.is_empty()
            || legacy.sealed.is_empty()
            || legacy.edit.len() + legacy.sealed.len() > 256
            || legacy.history.as_ref().is_some_and(|h| h.len() > 512)
        {
            return Err(failure(
                ErrorCode::InvalidValue,
                "bounded explicit edit/sealed source history required",
            ));
        }
        let source_bytes = store.read_bounded(source_id, 1048576)?;
        let source: LegacyRecord = LegacyRecord::parse(&source_bytes)?;
        if canonical(&source)? != canonical(&legacy)? {
            return Err(failure(
                ErrorCode::SourceMismatch,
                "legacy rows/history differ from actual immutable source",
            ));
        }
        self.files.verify(self.limits.max_database_bytes)?;
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sql)?;
        let contract = check_lease(&tx, &self.store, lease, &files::moment()?)?;
        let mut query = tx
            .prepare("SELECT id,body FROM legacy_intakes WHERE family=?1")
            .map_err(sql)?;
        let prior = query
            .query_map([contract.study_family.as_str()], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?))
            })
            .map_err(sql)?
            .take(1025)
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(sql)?;
        if prior.len() > 1024 {
            return Err(failure(
                ErrorCode::LimitExceeded,
                "legacy intake bound exceeded",
            ));
        }
        for (id, body) in prior {
            let value: LegacyAssessment = decode(&body, &id)?;
            if value.source == *source_id {
                ensure_live(lease)?;
                self.files.verify(self.limits.max_database_bytes)?;
                return id.parse();
            }
        }
        drop(query);
        let mut observed = BTreeSet::new();
        let mut known_exposed = Vec::new();
        let mut unknown = Vec::new();
        for (is_edit, items) in [(true, &legacy.edit), (false, &legacy.sealed)] {
            for item in items {
                if !observed.insert(&item.identity.item) {
                    return Err(failure(ErrorCode::InvalidValue, "duplicate legacy task"));
                }
                let input = read_payload(store, &item.input)?;
                read_payload(store, &item.label)?;
                let subject = Subject {
                    contract: lease.contract.clone(),
                    binding: source_id.clone(),
                    package_version: source_id.clone(),
                    dataset_version: source_id.clone(),
                    task: item.identity.clone(),
                    input: item.input.clone(),
                    semantic: identity(&input)?,
                    identity_protocol: identity_protocol(),
                    label: item.label.clone(),
                    split: if is_edit {
                        Some(SplitRole::Development)
                    } else {
                        None
                    },
                    legacy: true,
                };
                register(&tx, &contract.study_family, &subject)?;
                let rows = legacy
                    .history
                    .as_ref()
                    .map(|h| {
                        h.iter()
                            .filter(|h| h.item == item.identity.item)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                let has_history = !rows.is_empty();
                if is_edit || has_history {
                    known_exposed.push(item.identity.item.clone());
                } else {
                    unknown.push(item.identity.item.clone());
                }
                let count = rows.len().max(1);
                let mut previous = Vec::new();
                for index in 0..count {
                    let row = rows.get(index);
                    let context = Context {
                        seed: row.and_then(|r| r.seed),
                        attempt: row.map_or(0, |r| r.attempt),
                        retry_of: match row.and_then(|r| r.retry_of) {
                            Some(i) => {
                                Some(previous.get(i as usize).cloned().ok_or_else(|| {
                                    failure(
                                        ErrorCode::InvalidValue,
                                        "legacy retry must reference an earlier same-task event",
                                    )
                                })?)
                            }
                            None => None,
                        },
                        protocol: Some(source_id.clone()),
                        ..Default::default()
                    };
                    let ticket = begin(
                        &tx,
                        lease,
                        &contract,
                        Start {
                            subject: subject.clone(),
                            recipient: row.map_or_else(
                                || "legacy-unknown-recipient".into(),
                                |r| r.recipient.clone(),
                            ),
                            role: "legacy_import".into(),
                            operation: ContentDigest::of_bytes(&canonical(&(
                                source_id,
                                &item.identity.item,
                                index,
                            ))?),
                            context,
                            fields: [
                                Field::Input,
                                Field::Label,
                                Field::Output,
                                Field::Feedback,
                                Field::Summary,
                                Field::Identity,
                            ]
                            .into_iter()
                            .collect(),
                        },
                    )?;
                    let mut body = ticket.body;
                    body.stage = Stage::Imported;
                    body.parent = Some(ticket.started);
                    body.shown = row.map_or_else(BTreeSet::new, |r| r.fields.clone());
                    body.availability = if is_edit || has_history {
                        "legacy_declared_exposure;not_authenticated_observer"
                    } else {
                        "unknown_legacy_history;sealed_name_is_not_secrecy"
                    }
                    .into();
                    body.outcome =
                        row.map_or_else(|| "unknown_legacy_history".into(), |r| r.outcome.clone());
                    previous.push(append(&tx, &body)?.id);
                }
            }
        }
        if let Some(history) = &legacy.history {
            if history.iter().any(|h| {
                !observed.contains(&h.item)
                    || h.recipient.is_empty()
                    || h.recipient.len() > 256
                    || h.outcome.len() > 256
            }) {
                return Err(failure(
                    ErrorCode::InvalidValue,
                    "history must resolve selected legacy tasks and bounded recipients/outcomes",
                ));
            }
        }
        let assessment=LegacyAssessment{schema_version:1,source:source_id.clone(),contract:lease.contract.clone(),legacy,known_exposed,unknown,independent_three_way:false,allowed_destination:"development_only".into(),limitations:"external history remains declared/unknown;selection and holdout require separate fresh qualified evidence".into()};
        let bytes = canonical(&assessment)?;
        let id = ContentDigest::of_bytes(&bytes);
        tx.execute(
            "INSERT OR IGNORE INTO legacy_intakes VALUES(?1,?2,?3)",
            params![contract.study_family.as_str(), id.as_str(), bytes],
        )
        .map_err(sql)?;
        ensure_live(lease)?;
        tx.commit().map_err(sql)?;
        self.files.verify(self.limits.max_database_bytes)?;
        Ok(id)
    }
    pub fn legacy_assessment(&self, lease: &Lease, id: &ContentDigest) -> Result<LegacyAssessment> {
        self.files.verify(self.limits.max_database_bytes)?;
        let tx = self.connection.unchecked_transaction().map_err(sql)?;
        let contract = check_lease(&tx, &self.store, lease, &files::moment()?)?;
        let body: Vec<u8> = tx
            .query_row(
                "SELECT body FROM legacy_intakes WHERE family=?1 AND id=?2",
                params![contract.study_family.as_str(), id.as_str()],
                |r| r.get(0),
            )
            .map_err(sql)?;
        let result = decode(&body, id.as_str())?;
        ensure_live(lease)?;
        self.files.verify(self.limits.max_database_bytes)?;
        Ok(result)
    }
    pub fn admit_legacy_role(
        &self,
        lease: &Lease,
        assessment: &ContentDigest,
        item: &ContentDigest,
        role: SplitRole,
    ) -> Result<()> {
        let report = self.legacy_assessment(lease, assessment)?;
        if role != SplitRole::Development
            || !report
                .legacy
                .edit
                .iter()
                .chain(&report.legacy.sealed)
                .any(|r| r.identity.item == *item)
        {
            return Err(failure(
                ErrorCode::PermissionDenied,
                "legacy edit/sealed cannot become independent selection/holdout by renaming",
            ));
        }
        Ok(())
    }
}
