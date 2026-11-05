//! Data-only scoped brokerage. Trusted CPU fixture access discipline, not a sandbox.
use super::budgets::{
    OutcomeKind, Request as BudgetRequest, Reservation, Resources, Settlement, Usage, WorkKind,
};
use super::*;
use nf_checkpoint::store::ArtifactStore;
use nf_core::schemas::DataClass;
use nf_core::study::{ClaimScope, GymRole, Isolation, SplitRole};
use nf_graph::inventory::tensor_set_digest;
use nf_surgeon::transaction::{ExecutionLimits, SourceState};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::sync::Arc;
const MAX_FRAME: usize = 16384;
fn opaque(prefix: &str) -> Result<String> {
    let mut bytes = [0u8; 32];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut file| file.read_exact(&mut bytes))
        .map_err(|error| failure(ErrorCode::Io, &error.to_string()))?;
    Ok(format!(
        "{prefix}:{}",
        &ContentDigest::of_bytes(&bytes).as_str()[7..]
    ))
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Actor {
    Proposer,
    SelectionEvaluator,
    FinalEvaluator,
}
impl Actor {
    fn split(self) -> SplitRole {
        match self {
            Self::Proposer => SplitRole::Development,
            Self::SelectionEvaluator => SplitRole::Selection,
            Self::FinalEvaluator => SplitRole::Holdout,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Task {
    pub id: ContentDigest,
    pub group: ContentDigest,
    pub input: Vec<f64>,
    pub labels: Option<serde_json::Value>,
}
#[derive(Debug, Clone, Serialize)]
pub struct Quotas {
    pub messages: u32,
    pub forwards: u32,
    pub tool_calls: u32,
    pub input_units: u32,
    pub response_bytes: u32,
    pub staged_bytes: u32,
}
#[derive(Debug, Clone, Serialize)]
pub struct Welcome {
    pub schema_version: u32,
    pub contract: ContentDigest,
    pub revision: u32,
    pub worker: String,
    pub lease_id: String,
    pub actor: Actor,
    pub split: SplitRole,
    pub binding: ContentDigest,
    pub models: Vec<String>,
    pub tasks: Vec<ContentDigest>,
    pub artifacts: BTreeMap<String, String>,
    pub tools: Vec<String>,
    pub maximum_frame_bytes: usize,
    pub access_assurance: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Action {
    TaskInput {
        task: ContentDigest,
    },
    Infer {
        model: String,
        task: ContentDigest,
    },
    Tool {
        name: String,
        left: i64,
        right: i64,
    },
    Artifact {
        handle: String,
    },
    StageOutput {
        slot: String,
        document: serde_json::Value,
    },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Envelope {
    pub schema_version: u32,
    pub contract: ContentDigest,
    pub revision: u32,
    pub worker: String,
    pub lease_id: String,
    pub split: SplitRole,
    pub sequence: u64,
    pub action: Action,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reply {
    pub schema_version: u32,
    pub sequence: Option<u64>,
    pub ok: bool,
    pub result: Option<serde_json::Value>,
    pub error: Option<CoreError>,
}
pub struct Scope {
    pub binding: ContentDigest,
    pub actor: Actor,
    pub tasks: Vec<Task>,
    pub quotas: Quotas,
}
struct ScopedArtifact {
    payload: serde_json::Value,
    class: DataClass,
}
/// Controller-owned broker. Do not send this object, source, ledger, or lease
/// into a gym; only Welcome/Envelope/Reply and its private IPC endpoint cross.
pub struct Session<'a> {
    ledger: &'a mut Ledger,
    lease: Lease,
    pool: Reservation,
    source: Arc<SourceState>,
    store: &'a ArtifactStore,
    welcome: Welcome,
    tasks: BTreeMap<ContentDigest, Task>,
    artifacts: BTreeMap<String, ScopedArtifact>,
    quotas: Quotas,
    messages: u32,
    forwards: u32,
    tools: u32,
    input_units: u32,
    response_bytes: u32,
    staged_bytes: u32,
    next: u64,
    exposure_replays: u32,
    exposure_history: BTreeMap<(u64, ContentDigest, super::exposure::Field), ContentDigest>,
    cache: BTreeMap<u64, (ContentDigest, Reply)>,
    deadline: i64,
}
impl<'a> Session<'a> {
    /// Host/controller constructor. Source/task payloads are already resolved
    /// owned snapshots, never executable factories/callbacks from a worker.
    pub fn trusted_fixture(
        ledger: &'a mut Ledger,
        lease: Lease,
        pool: Reservation,
        source: Arc<SourceState>,
        store: &'a ArtifactStore,
        scope: Scope,
    ) -> Result<Self> {
        let Scope {
            binding,
            actor,
            tasks,
            quotas,
        } = scope;
        let binding = &binding;
        ledger.files.verify(ledger.limits.max_database_bytes)?;
        let tx = ledger.connection.unchecked_transaction().map_err(sql)?;
        let contract = check_lease(&tx, &ledger.store, &lease, &files::moment()?)?;
        if super::supervision::stopped(&tx, &lease.contract)? {
            return Err(failure(
                ErrorCode::PermissionDenied,
                "stopped broker contract",
            ));
        }
        let allocation = budgets::broker_pool(&tx, &ledger.store, &lease, &pool)?;
        if quotas.messages == 0
            || quotas.messages > 64
            || quotas.forwards as u64 > allocation.resources.evaluations
            || quotas.tool_calls > 64
            || quotas.input_units == 0
            || quotas.input_units > 65536
            || quotas.response_bytes == 0
            || quotas.response_bytes > 1_048_576
            || quotas.staged_bytes > 1_048_576
            || tasks.len() > 4096
        {
            return Err(failure(
                ErrorCode::LimitExceeded,
                "finite scoped broker quotas required",
            ));
        }
        let portfolio = &contract.portfolio;
        if portfolio.independent.scope != ClaimScope::FixtureOnly
            || portfolio.independent.isolation != Isolation::TrustedFixtureBroker
        {
            return Err(failure(ErrorCode::UnsupportedCapability,"fixture broker does not qualify separate-account/process adversarial final isolation"));
        }
        let selected = portfolio
            .bindings
            .iter()
            .find(|candidate| &candidate.id == binding)
            .ok_or_else(|| failure(ErrorCode::SourceMismatch, "unknown exact gym binding"))?;
        let stage = state(&tx, &lease.contract)?.stage;
        if (actor == Actor::FinalEvaluator && stage != Stage::Verify)
            || (actor != Actor::FinalEvaluator && matches!(stage, Stage::Verify | Stage::Learn))
        {
            return Err(failure(
                ErrorCode::PermissionDenied,
                "split actor cannot open this workflow phase",
            ));
        }
        if actor == Actor::FinalEvaluator {
            super::splits::require_panel(&tx, &lease.contract)?;
        }
        if actor == Actor::FinalEvaluator && !selected.schedule.final_required {
            return Err(failure(
                ErrorCode::PermissionDenied,
                "final binding is not in frozen panel",
            ));
        }
        if actor == Actor::Proposer && !selected.roles.contains(&GymRole::Target) {
            return Err(failure(
                ErrorCode::PermissionDenied,
                "proposal tasks must use declared target binding",
            ));
        }
        let split = match actor {
            Actor::Proposer => &selected.splits.development,
            Actor::SelectionEvaluator => &selected.splits.selection,
            Actor::FinalEvaluator => &selected.splits.holdout,
        };
        let expected: BTreeMap<_, _> = split
            .items
            .iter()
            .map(|item| (item.item.clone(), item.group.clone()))
            .collect();
        if tasks.len() != expected.len()
            || tasks
                .iter()
                .map(|task| &task.id)
                .collect::<BTreeSet<_>>()
                .len()
                != tasks.len()
            || tasks.iter().any(|task| {
                expected.get(&task.id) != Some(&task.group)
                    || task.input.is_empty()
                    || task.input.len() > 4096
                    || task.input.iter().any(|x| !x.is_finite())
            })
        {
            return Err(failure(
                ErrorCode::SourceMismatch,
                "task payload inventory does not match exact scoped split",
            ));
        }
        if contract.runtime.device != nf_core::study::Device::Cpu
            || contract.runtime.backend.id != "fixture.cpu"
            || contract.runtime.backend.version != "1"
            || contract.runtime.batch_size != 1
            || contract.runtime.concurrency != 1
            || contract.runtime.cache != nf_core::study::CacheMode::Disabled
            || source.manifest().architecture.id != "fixture.cpu"
            || source.manifest().architecture.version != "1"
            || source
                .tensor("w")
                .is_none_or(|weight| weight.dtype() != contract.runtime.precision)
        {
            return Err(failure(
                ErrorCode::UnsupportedCapability,
                "native broker does not qualify the requested runtime/model adapter",
            ));
        }
        source.verify_integrity()?;
        let model = &contract.source;
        if model.system_state.is_some() || model.runtime_artifact.is_some() {
            return Err(failure(
                ErrorCode::UnsupportedCapability,
                "native fixture broker cannot ignore external memory or compiled runtime state",
            ));
        }
        if source.manifest_digest() != &model.manifest
            || ContentDigest::of_bytes(source.checkpoint_bytes()) != model.checkpoint
            || source.manifest().parameter_digest != model.parameters
            || tensor_set_digest(&source.manifest().buffers)? != model.buffers
            || source.manifest().architecture.config != model.config
            || source.manifest().assets.get("tokenizer") != model.tokenizer.as_ref()
        {
            return Err(failure(
                ErrorCode::SourceMismatch,
                "broker model source is not exact frozen target",
            ));
        }
        let mut artifacts = BTreeMap::new();
        let mut visible = BTreeMap::new();
        let task_handle = opaque("artifact")?;
        artifacts.insert(task_handle.clone(),ScopedArtifact {payload:serde_json::json!({"task_ids":tasks.iter().map(|task|task.id.clone()).collect::<Vec<_>>()}),class:DataClass::Tasks});
        visible.insert("task_manifest".to_owned(), task_handle);
        if actor != Actor::Proposer {
            let labels = tasks
                .iter()
                .map(|task| {
                    Ok((
                        task.id.to_string(),
                        task.labels.clone().ok_or_else(|| {
                            failure(ErrorCode::SourceMismatch, "evaluation task labels missing")
                        })?,
                    ))
                })
                .collect::<Result<BTreeMap<_, _>>>()?;
            let handle = opaque("artifact")?;
            artifacts.insert(
                handle.clone(),
                ScopedArtifact {
                    payload: serde_json::json!({"labels":labels}),
                    class: DataClass::Labels,
                },
            );
            visible.insert("scorer_labels".to_owned(), handle);
        }
        let welcome = Welcome {
            schema_version: 1,
            contract: lease.contract.clone(),
            revision: contract.revision,
            worker: opaque("worker")?,
            lease_id: opaque("worker-lease")?,
            actor,
            split: actor.split(),
            binding: binding.clone(),
            models: vec!["target".to_owned()],
            tasks: tasks.iter().map(|task| task.id.clone()).collect(),
            artifacts: visible,
            tools: if contract
                .permissions
                .tools
                .contains(&nf_core::schemas::ToolKind::Inspect)
            {
                vec!["fixture.sum".to_owned()]
            } else {
                vec![]
            },
            maximum_frame_bytes: MAX_FRAME,
            access_assurance: "trusted_fixture_scope_discipline_not_adversarial_isolation"
                .to_owned(),
        };
        let initial = canonical(&welcome)?;
        if initial.len() > MAX_FRAME || initial.len() > quotas.response_bytes as usize {
            return Err(failure(
                ErrorCode::LimitExceeded,
                "scope welcome exceeds output bound; use smaller scoped batches",
            ));
        }
        let resolved_payloads = tasks
            .iter()
            .map(|task| Ok((task.id.clone(), ContentDigest::of_bytes(&canonical(task)?))))
            .collect::<Result<BTreeMap<_, _>>>()?;
        let audit = serde_json::json!({"schema_version":1,"kind":"broker_scope","contract":welcome.contract,"revision":welcome.revision,"worker":welcome.worker,"actor":actor,"split":welcome.split,"binding":welcome.binding,"tasks":welcome.tasks,"models":welcome.models,"artifacts":welcome.artifacts,"resolved_payloads":resolved_payloads,"quotas":quotas,"classification":"private_scope_metadata","access_assurance":welcome.access_assurance});
        let deadline = allocation.deadline_ms.min(lease.deadline);
        drop(tx);
        let setup = ledger.reserve(
            &lease,
            BudgetRequest {
                schema_version: 1,
                operation: ContentDigest::of_bytes(&canonical(&(
                    &welcome.worker,
                    "scope-metadata",
                ))?),
                attempt: 0,
                parent: Some(pool.id().clone()),
                kind: WorkKind::Metadata,
                resources: Resources {
                    cpu_ms: 1000,
                    host_bytes: 1_048_576,
                    storage_bytes: 65536,
                    workers: 1,
                    ..Resources::default()
                },
                wall_ms: deadline.saturating_sub(files::moment()?.ms).clamp(0, 1000) as u64,
                external: None,
            },
        )?;
        let stored = store.put(canonical(&audit)?.as_slice())?;
        ledger.settle(
            &lease,
            &setup,
            Settlement {
                operation: ContentDigest::of_bytes(&canonical(&(
                    &welcome.worker,
                    "scope-metadata-cost",
                ))?),
                outcome: OutcomeKind::Completed,
                usage: Usage {
                    attempts: Some(0),
                    evaluations: Some(0),
                    gpu_ms: Some(0),
                    reasoning_tokens: Some(0),
                    teacher_tokens: Some(0),
                    spend_microusd: Some(0),
                    ..Usage::default()
                },
                resources_released: true,
                evidence: stored.digest,
            },
        )?;
        if actor == Actor::FinalEvaluator {
            ledger.open_fixture_holdout(&lease, &welcome.worker)?;
        }
        let mut session = Self {
            ledger,
            lease,
            pool,
            source,
            store,
            welcome,
            tasks: tasks
                .into_iter()
                .map(|task| (task.id.clone(), task))
                .collect(),
            artifacts,
            quotas,
            messages: 0,
            forwards: 0,
            tools: 0,
            input_units: 0,
            response_bytes: 0,
            staged_bytes: 0,
            next: 0,
            exposure_replays: 0,
            exposure_history: BTreeMap::new(),
            cache: BTreeMap::new(),
            deadline,
        };
        let manifest_handle = session.welcome.artifacts["task_manifest"].clone();
        let tickets = session.start_exposure(
            &Action::Artifact {
                handle: manifest_handle,
            },
            u64::MAX,
            "welcome",
        )?;
        session.complete_exposure(&tickets, true, 0, u64::MAX)?;
        Ok(session)
    }
    pub fn welcome(&self) -> &Welcome {
        &self.welcome
    }
    fn current(&self) -> Result<()> {
        let tx = self
            .ledger
            .connection
            .unchecked_transaction()
            .map_err(sql)?;
        check_lease(&tx, &self.ledger.store, &self.lease, &files::moment()?)?;
        let stage = state(&tx, &self.lease.contract)?.stage;
        let final_scope = self.welcome.split == SplitRole::Holdout;
        if stage.terminal()
            || (final_scope && stage != Stage::Verify)
            || (!final_scope && matches!(stage, Stage::Verify | Stage::Learn))
        {
            return Err(failure(
                ErrorCode::PermissionDenied,
                "split scope closed after final phase",
            ));
        }
        if final_scope {
            super::splits::require_panel(&tx, &self.lease.contract)?;
        }
        if super::supervision::stopped(&tx, &self.lease.contract)?
            || files::moment()?.ms >= self.deadline
        {
            return Err(failure(
                ErrorCode::PermissionDenied,
                "worker scope stopped or expired",
            ));
        }
        Ok(())
    }
    fn start_exposure(
        &self,
        action: &Action,
        sequence: u64,
        tag: &str,
    ) -> Result<Vec<super::exposure::Ticket>> {
        use super::exposure::{self, Field};
        let fields = match action {
            Action::TaskInput { task } => vec![(task.clone(), Field::Input)],
            Action::Infer { task, .. } => vec![(task.clone(), Field::Output)],
            Action::Artifact { handle } => match self.artifacts.get(handle) {
                Some(artifact) if artifact.class == DataClass::Labels => self
                    .tasks
                    .keys()
                    .map(|id| (id.clone(), Field::Label))
                    .collect(),
                Some(artifact) if artifact.class == DataClass::Tasks => self
                    .tasks
                    .keys()
                    .map(|id| (id.clone(), Field::Identity))
                    .collect(),
                _ => Vec::new(),
            },
            _ => Vec::new(),
        };
        let tx = self
            .ledger
            .connection
            .unchecked_transaction()
            .map_err(sql)?;
        let contract = check_lease(&tx, &self.ledger.store, &self.lease, &files::moment()?)?;
        let binding = contract
            .portfolio
            .bindings
            .iter()
            .find(|b| b.id == self.welcome.binding)
            .ok_or_else(|| failure(ErrorCode::SourceMismatch, "unknown exposure binding"))?;
        let split = match self.welcome.split {
            SplitRole::Development => &binding.splits.development,
            SplitRole::Selection => &binding.splits.selection,
            SplitRole::Holdout => &binding.splits.holdout,
        };
        let role = serde_json::to_value(self.welcome.actor)
            .unwrap()
            .as_str()
            .unwrap()
            .to_owned();
        let mut tickets = Vec::new();
        for (id, field) in fields {
            let Some(task) = self.tasks.get(&id) else {
                continue;
            };
            let identity = split
                .items
                .iter()
                .find(|item| item.item == id)
                .ok_or_else(|| failure(ErrorCode::SourceMismatch, "unresolved task identity"))?;
            let subject = exposure::Subject {
                contract: self.lease.contract.clone(),
                binding: binding.id.clone(),
                package_version: binding.gym.clone(),
                dataset_version: split.dataset.clone(),
                task: identity.clone(),
                input: ContentDigest::of_bytes(&canonical(&task.input)?),
                semantic: exposure::identity(&serde_json::to_value(&task.input).unwrap())?,
                identity_protocol: exposure::identity_protocol(),
                label: ContentDigest::of_bytes(&canonical(&task.labels)?),
                split: Some(self.welcome.split),
                legacy: false,
            };
            exposure::admit(&tx, &contract, &subject)?;
            let context = exposure::Context {
                model: Some(self.source.manifest_digest().clone()),
                protocol: Some(contract.intervention.method_protocol.clone()),
                attempt: if tag == "fresh" {
                    0
                } else {
                    self.exposure_replays
                },
                retry_of: if tag == "cached" {
                    self.exposure_history
                        .get(&(sequence, id.clone(), field))
                        .cloned()
                } else {
                    None
                },
                ..Default::default()
            };
            if matches!(action, Action::Infer { .. }) && tag == "fresh" {
                tickets.push(exposure::begin(
                    &tx,
                    &self.lease,
                    &contract,
                    exposure::Start {
                        subject: subject.clone(),
                        recipient: self.source.manifest_digest().to_string(),
                        role: "target_model".into(),
                        operation: ContentDigest::of_bytes(&canonical(&(
                            &self.welcome.worker,
                            sequence,
                            tag,
                            &id,
                            "model",
                        ))?),
                        context: context.clone(),
                        fields: [Field::Input].into_iter().collect(),
                    },
                )?);
            }
            tickets.push(exposure::begin(
                &tx,
                &self.lease,
                &contract,
                exposure::Start {
                    subject,
                    recipient: self.welcome.worker.clone(),
                    role: role.clone(),
                    operation: ContentDigest::of_bytes(&canonical(&(
                        &self.welcome.worker,
                        sequence,
                        tag,
                        &id,
                        self.exposure_replays,
                    ))?),
                    context,
                    fields: [field].into_iter().collect(),
                },
            )?);
        }
        ensure_live(&self.lease)?;
        tx.commit().map_err(sql)?;
        Ok(tickets)
    }
    fn complete_exposure(
        &mut self,
        tickets: &[super::exposure::Ticket],
        ok: bool,
        forwards: u64,
        sequence: u64,
    ) -> Result<()> {
        use super::exposure::{self, Field};
        for ticket in tickets {
            let observed = ticket.body.role == "target_model" && forwards > 0;
            let completed = self.ledger.finish_exposure(
                &self.lease,
                ticket,
                if ok {
                    exposure::Stage::Completed
                } else {
                    exposure::Stage::Failed
                },
                if ok || observed {
                    ticket.body.intended.clone()
                } else {
                    BTreeSet::<Field>::new()
                },
                if ok {
                    "broker_reply_available"
                } else {
                    "broker_request_failed_or_partial"
                },
                exposure::Cost {
                    model_calls: Some(if observed { forwards } else { 0 }),
                    reasoning_tokens: Some(0),
                    teacher_tokens: Some(0),
                    ..Default::default()
                },
            )?;
            for field in &ticket.body.intended {
                self.exposure_history.insert(
                    (sequence, ticket.body.subject.task.item.clone(), *field),
                    completed.id.clone(),
                );
            }
        }
        Ok(())
    }
    fn error(sequence: Option<u64>, error: CoreError) -> Reply {
        Reply {
            schema_version: 1,
            sequence,
            ok: false,
            result: None,
            error: Some(error),
        }
    }
    pub fn handle(&mut self, bytes: &[u8]) -> Reply {
        let result = (|| -> Result<(u64, serde_json::Value)> {
            self.current()?;
            if bytes.len() > MAX_FRAME {
                return Err(failure(
                    ErrorCode::LimitExceeded,
                    "worker envelope exceeds byte bound",
                ));
            }
            let doc = CanonicalDocument::parse(
                bytes,
                Limits {
                    max_bytes: MAX_FRAME,
                    max_nodes: 4096,
                    max_depth: 16,
                },
            )?;
            let request: Envelope = serde_json::from_slice(doc.bytes())
                .map_err(|_| failure(ErrorCode::InvalidValue, "invalid worker envelope schema"))?;
            if request.schema_version != 1
                || request.contract != self.welcome.contract
                || request.revision != self.welcome.revision
                || request.worker != self.welcome.worker
                || request.lease_id != self.welcome.lease_id
                || request.split != self.welcome.split
            {
                return Err(failure(
                    ErrorCode::PermissionDenied,
                    "forged/stale worker lease, contract or split",
                ));
            }
            let digest = doc.digest();
            if let Some((prior, reply)) = self.cache.get(&request.sequence).cloned() {
                if prior != digest {
                    return Err(failure(
                        ErrorCode::IdempotencyConflict,
                        "worker sequence reused with changed request",
                    ));
                }
                self.exposure_replays += 1;
                if self.exposure_replays > self.quotas.messages {
                    return Err(failure(
                        ErrorCode::QuotaExceeded,
                        "cached exposure retry quota exceeded",
                    ));
                }
                let tickets = self.start_exposure(&request.action, request.sequence, "cached")?;
                self.complete_exposure(&tickets, reply.ok, 0, request.sequence)?;
                return Ok((
                    request.sequence,
                    serde_json::to_value(reply)
                        .map_err(|_| failure(ErrorCode::InvalidValue, "reply serialization"))?,
                ));
            }
            if request.sequence != self.next || self.messages >= self.quotas.messages {
                return Err(failure(
                    ErrorCode::QuotaExceeded,
                    "worker message sequence/quota exhausted",
                ));
            }
            self.messages += 1;
            self.next += 1;
            let estimate = match request.action {
                Action::Infer { .. } => 1,
                _ => 0,
            };
            let reservation = self.ledger.reserve(
                &self.lease,
                BudgetRequest {
                    schema_version: 1,
                    operation: ContentDigest::of_bytes(&canonical(&(
                        &self.welcome.worker,
                        request.sequence,
                        &digest,
                    ))?),
                    attempt: 0,
                    parent: Some(self.pool.id().clone()),
                    kind: if estimate == 1 {
                        WorkKind::Evaluation
                    } else {
                        WorkKind::Metadata
                    },
                    resources: Resources {
                        cpu_ms: 1000,
                        evaluations: estimate,
                        host_bytes: 1_048_576,
                        storage_bytes: 65536,
                        workers: 1,
                        ..Resources::default()
                    },
                    wall_ms: self
                        .deadline
                        .saturating_sub(files::moment()?.ms)
                        .clamp(0, 1000) as u64,
                    external: None,
                },
            )?;
            let mut exposed = serde_json::to_value(&request)
                .map_err(|_| failure(ErrorCode::InvalidValue, "request audit encoding"))?;
            exposed.as_object_mut().unwrap().remove("lease_id");
            let requested = serde_json::json!({"schema_version":1,"kind":"broker_access_requested","request":exposed,"actor":self.welcome.actor,"reservation":reservation.id(),"classification":"private_scoped_exposure"});
            let requested = self.store.put(canonical(&requested)?.as_slice())?;
            let tickets = self.start_exposure(&request.action, request.sequence, "fresh")?;
            let before = self.forwards;
            let action = self.act(&request.action);
            let evaluation = (self.forwards - before) as u64;
            let evidence = ContentDigest::of_bytes(&canonical(&(
                request.sequence,
                &digest,
                action
                    .as_ref()
                    .map(|value| value.to_string())
                    .map_err(|error| error.code),
            ))?);
            self.ledger.settle(
                &self.lease,
                &reservation,
                Settlement {
                    operation: ContentDigest::of_bytes(&canonical(&(
                        &self.welcome.worker,
                        request.sequence,
                        "cost",
                    ))?),
                    outcome: if action.is_ok() {
                        OutcomeKind::Completed
                    } else {
                        OutcomeKind::Rejected
                    },
                    usage: Usage {
                        attempts: Some(0),
                        evaluations: Some(evaluation),
                        gpu_ms: Some(0),
                        reasoning_tokens: Some(0),
                        teacher_tokens: Some(0),
                        spend_microusd: Some(0),
                        ..Usage::default()
                    },
                    resources_released: true,
                    evidence: evidence.clone(),
                },
            )?;
            let completed = serde_json::json!({"schema_version":1,"kind":"broker_access_completed","requested":requested.digest,"contract":self.welcome.contract,"worker":self.welcome.worker,"sequence":request.sequence,"split":self.welcome.split,"forwards":evaluation,"outcome":if action.is_ok(){"completed"}else{"failed"},"result_digest":evidence,"classification":"private_scoped_exposure"});
            self.store.put(canonical(&completed)?.as_slice())?;
            let reply = match action {
                Ok(value) => Reply {
                    schema_version: 1,
                    sequence: Some(request.sequence),
                    ok: true,
                    result: Some(value),
                    error: None,
                },
                Err(error) => Self::error(Some(request.sequence), error),
            };
            let encoded = canonical(&reply)?;
            if encoded.len() > MAX_FRAME
                || self.response_bytes as usize + encoded.len()
                    > self.quotas.response_bytes as usize
            {
                self.complete_exposure(&tickets, false, evaluation, request.sequence)?;
                return Err(failure(
                    ErrorCode::QuotaExceeded,
                    "scoped response byte quota exhausted",
                ));
            }
            self.complete_exposure(&tickets, reply.ok, evaluation, request.sequence)?;
            self.response_bytes += encoded.len() as u32;
            self.cache.insert(request.sequence, (digest, reply.clone()));
            Ok((
                request.sequence,
                serde_json::to_value(reply)
                    .map_err(|_| failure(ErrorCode::InvalidValue, "reply serialization"))?,
            ))
        })();
        match result {
            Ok((_, value)) => serde_json::from_value(value).unwrap_or_else(|_| {
                Self::error(
                    None,
                    failure(ErrorCode::InvalidValue, "broker reply schema"),
                )
            }),
            Err(error) => Self::error(None, error),
        }
    }
    fn act(&mut self, action: &Action) -> Result<serde_json::Value> {
        match action {
            Action::TaskInput { task } => {
                let task = self.tasks.get(task).ok_or_else(|| {
                    failure(
                        ErrorCode::PermissionDenied,
                        "task is outside this exact split grant",
                    )
                })?;
                Ok(serde_json::json!({"task":task.id,"input":task.input}))
            }
            Action::Infer { model, task } => {
                if model != "target" {
                    return Err(failure(
                        ErrorCode::PermissionDenied,
                        "model slot is not granted",
                    ));
                }
                let task = self.tasks.get(task).ok_or_else(|| {
                    failure(
                        ErrorCode::PermissionDenied,
                        "task is outside inference scope",
                    )
                })?;
                if self.forwards >= self.quotas.forwards
                    || self.input_units as usize + task.input.len()
                        > self.quotas.input_units as usize
                {
                    return Err(failure(
                        ErrorCode::QuotaExceeded,
                        "scoped inference quota exhausted",
                    ));
                }
                self.forwards += 1;
                self.input_units += task.input.len() as u32;
                let output = self
                    .source
                    .infer_linear_fixture(&task.input, ExecutionLimits::default())?;
                Ok(
                    serde_json::json!({"model":model,"task":task.id,"output":output,"forwards_used":self.forwards}),
                )
            }
            Action::Tool { name, left, right } => {
                if !self.welcome.tools.contains(name) {
                    return Err(failure(
                        ErrorCode::PermissionDenied,
                        "tool is not in exact allowlist",
                    ));
                }
                if self.tools >= self.quotas.tool_calls
                    || left.abs_diff(0) > 1000
                    || right.abs_diff(0) > 1000
                {
                    return Err(failure(
                        ErrorCode::QuotaExceeded,
                        "tool call/value bounds exceeded",
                    ));
                }
                self.tools += 1;
                Ok(
                    serde_json::json!({"sum":left.checked_add(*right).ok_or_else(||failure(ErrorCode::LimitExceeded,"tool arithmetic overflow"))?}),
                )
            }
            Action::Artifact { handle } => {
                let artifact = self.artifacts.get(handle).ok_or_else(|| {
                    failure(
                        ErrorCode::PermissionDenied,
                        "artifact handle is hidden or unknown",
                    )
                })?;
                if artifact.class == DataClass::Labels && self.welcome.actor == Actor::Proposer {
                    return Err(failure(
                        ErrorCode::PermissionDenied,
                        "labels cannot enter proposal scope",
                    ));
                }
                Ok(artifact.payload.clone())
            }
            Action::StageOutput { slot, document } => {
                if slot != "result" {
                    return Err(failure(
                        ErrorCode::PermissionDenied,
                        "output slot is not granted",
                    ));
                }
                let bytes = canonical(document)?;
                if self.staged_bytes as usize + bytes.len() > self.quotas.staged_bytes as usize {
                    return Err(failure(
                        ErrorCode::QuotaExceeded,
                        "staged output quota exceeded",
                    ));
                }
                let blob = self.store.put(bytes.as_slice())?;
                let manifest = serde_json::json!({"schema_version":1,"contract":self.welcome.contract,"worker":self.welcome.worker,"binding":self.welcome.binding,"split":self.welcome.split,"payload":blob.digest,"bytes":blob.bytes,"classification":"unreviewed_worker_output","status":"staged_not_promoted"});
                let metadata = canonical(&manifest)?;
                let committed = self.store.put(metadata.as_slice())?;
                self.staged_bytes += bytes.len() as u32;
                Ok(
                    serde_json::json!({"handle":opaque("staged")?,"digest":blob.digest,"manifest":committed.digest,"bytes":blob.bytes,"split":self.welcome.split,"classification":"unreviewed_worker_output","status":"staged_not_promoted"}),
                )
            }
        }
    }
    pub fn serve_unix(&mut self, stream: &std::os::unix::net::UnixStream) -> Result<()> {
        let input = DeadlineChannel {
            stream,
            deadline: self.deadline,
        };
        let output = DeadlineChannel {
            stream,
            deadline: self.deadline,
        };
        self.serve(input, output)
    }
    fn serve(&mut self, mut input: impl Read, mut output: impl Write) -> Result<()> {
        let initial = canonical(&self.welcome)?;
        let mut sent = initial.len();
        write_frame(&mut output, &initial)?;
        for _ in 0..self.quotas.messages.saturating_mul(2) {
            let Some(frame) = read_frame(&mut input)? else {
                return Ok(());
            };
            let reply = self.handle(&frame);
            let bytes = canonical(&reply)?;
            sent = sent.checked_add(bytes.len()).ok_or_else(|| {
                failure(ErrorCode::LimitExceeded, "wire response counter overflow")
            })?;
            if sent > self.quotas.response_bytes as usize {
                return Err(failure(
                    ErrorCode::QuotaExceeded,
                    "wire response quota exhausted, including errors/replays",
                ));
            }
            write_frame(&mut output, &bytes)?;
        }
        Err(failure(
            ErrorCode::QuotaExceeded,
            "IPC transport frame bound exhausted",
        ))
    }
}
impl<'a> Drop for Session<'a> {
    fn drop(&mut self) {
        let _ = self.ledger.close_branch(
            &self.lease,
            &self.pool,
            ContentDigest::of_bytes(format!("{}-scope-close", self.welcome.worker).as_bytes()),
            ContentDigest::of_bytes(b"bounded broker session ended"),
        );
    }
}
struct DeadlineChannel<'a> {
    stream: &'a std::os::unix::net::UnixStream,
    deadline: i64,
}
impl DeadlineChannel<'_> {
    fn remaining(&self) -> std::io::Result<std::time::Duration> {
        let now = files::moment().map_err(std::io::Error::other)?;
        let remaining = self.deadline.saturating_sub(now.ms);
        if remaining <= 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "broker scope deadline expired",
            ));
        }
        Ok(std::time::Duration::from_millis(remaining as u64))
    }
}
impl Read for DeadlineChannel<'_> {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        self.stream.set_read_timeout(Some(self.remaining()?))?;
        let mut stream = self.stream;
        stream.read(bytes)
    }
}
impl Write for DeadlineChannel<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.stream.set_write_timeout(Some(self.remaining()?))?;
        let mut stream = self.stream;
        stream.write(bytes)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        let mut stream = self.stream;
        stream.flush()
    }
}
pub fn read_frame(mut input: impl Read) -> Result<Option<Vec<u8>>> {
    let mut header = [0u8; 4];
    match input.read(&mut header[..1]) {
        Ok(0) => return Ok(None),
        Ok(_) => (),
        Err(error) => return Err(failure(ErrorCode::Io, &error.to_string())),
    }
    input
        .read_exact(&mut header[1..])
        .map_err(|error| failure(ErrorCode::Io, &error.to_string()))?;
    let length = u32::from_be_bytes(header) as usize;
    if length > MAX_FRAME {
        return Err(failure(
            ErrorCode::LimitExceeded,
            "IPC frame length exceeds bound",
        ));
    }
    let mut bytes = vec![0; length];
    input
        .read_exact(&mut bytes)
        .map_err(|error| failure(ErrorCode::Io, &error.to_string()))?;
    Ok(Some(bytes))
}
pub fn write_frame(mut output: impl Write, bytes: &[u8]) -> Result<()> {
    if bytes.len() > MAX_FRAME {
        return Err(failure(
            ErrorCode::LimitExceeded,
            "IPC response frame exceeds bound",
        ));
    }
    output
        .write_all(&(bytes.len() as u32).to_be_bytes())
        .and_then(|_| output.write_all(bytes))
        .and_then(|_| output.flush())
        .map_err(|error| failure(ErrorCode::Io, &error.to_string()))
}
