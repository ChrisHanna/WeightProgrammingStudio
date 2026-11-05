//! Immutable patch records are data, not execution certificates. Replay rebuilds
//! plans, runs real transactions, and materializes each parent under finite limits.

use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use nf_core::error::{CoreError, ErrorCode, Result};
use nf_core::schemas::{Validate, Validated};
use nf_graph::inventory::{verify_inventory, InventoryLimits};
use nf_surgeon::history::{
    certify_history, HistoryCosts, HistoryLimits, HistoryRequest, VerifiedHistory,
};
use nf_surgeon::transaction::{SmokeSuite, SourceState};
use nf_weight_ir::compilation::{compile, CompilePolicy, CompiledPlan, SourceSnapshot};
use nf_weight_ir::program::{ModelBinding, Program, TensorRegion};
use nf_weight_ir::search_space::{resolve_space, SearchSpace};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

fn failure(code: ErrorCode, message: &str) -> CoreError {
    CoreError::new(code, "", message)
}
fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| failure(ErrorCode::LimitExceeded, "replay accounting overflow"))
}
fn hash<T: Serialize>(value: &T) -> Result<ContentDigest> {
    let raw = serde_json::to_vec(value)
        .map_err(|_| failure(ErrorCode::InvalidValue, "record serialization failed"))?;
    Ok(CanonicalDocument::parse(&raw, Limits::default())?.digest())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PatchBody {
    pub schema_version: u32,
    pub root_model: ModelBinding,
    pub base_model: ModelBinding,
    pub parents: Vec<ContentDigest>,
    pub program: Program,
    pub policy: CompilePolicy,
    pub search_space: SearchSpace,
    pub smoke_suite: SmokeSuite,
    pub operands: BTreeMap<String, ContentDigest>,
    pub compiler: String,
    pub compiler_version: String,
    pub backend_digest: ContentDigest,
    pub expected_plan: ContentDigest,
    pub expected_candidate: ContentDigest,
    pub expected_model: ModelBinding,
    pub expected_buffers: ContentDigest,
    pub cumulative_effects: Vec<TensorRegion>,
    pub costs: HistoryCosts,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PatchRecord {
    pub id: ContentDigest,
    pub body: PatchBody,
}
impl PatchRecord {
    pub fn from_verified(
        parents: Vec<ContentDigest>,
        plan: &CompiledPlan,
        policy: &Validated<CompilePolicy>,
        space: &Validated<SearchSpace>,
        smoke: &Validated<SmokeSuite>,
        history: &VerifiedHistory,
    ) -> Result<Self> {
        if plan.digest() != &history.receipt().plan
            || hash(&**policy)? != plan.summary().policy
            || space.digest()? != plan.summary().search_space
            || smoke.digest()? != history.receipt().smoke_suite
        {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "record inputs do not match verified execution",
            ));
        }
        let body = PatchBody {
            schema_version: 1,
            root_model: space.base_model.clone(),
            base_model: plan.program().base_model.clone(),
            parents,
            program: plan.program().clone(),
            policy: (**policy).clone(),
            search_space: (**space).clone(),
            smoke_suite: (**smoke).clone(),
            operands: plan
                .program()
                .artifacts
                .iter()
                .map(|(id, artifact)| (id.clone(), artifact.digest.clone()))
                .collect(),
            compiler: plan.summary().compiler.clone(),
            compiler_version: plan.summary().compiler_version.clone(),
            backend_digest: plan.summary().backend.clone(),
            expected_plan: plan.digest().clone(),
            expected_candidate: history.candidate_id().clone(),
            expected_model: history.resulting_model().clone(),
            expected_buffers: history.receipt().after_buffers.clone(),
            cumulative_effects: history.prior_effects().cumulative_writes,
            costs: history.costs().clone(),
        };
        validate_body(&body)?;
        Ok(Self {
            id: hash(&body)?,
            body,
        })
    }
}
fn validate_body(body: &PatchBody) -> Result<()> {
    if body.schema_version != 1 {
        return Err(failure(
            ErrorCode::UnsupportedVersion,
            "only patch record v1 supported",
        ));
    }
    body.program.validate()?;
    body.policy.validate()?;
    body.search_space.validate()?;
    body.smoke_suite.validate()?;
    if body.parents.len() > 16 || body.cumulative_effects.len() > 1024 || body.operands.len() > 1024
    {
        return Err(failure(
            ErrorCode::LimitExceeded,
            "patch record cardinality exceeded",
        ));
    }
    if body.root_model != body.search_space.base_model
        || body.base_model != body.program.base_model
        || body.base_model != body.policy.source
        || body.program.objective.search_space != body.search_space.digest()?
        || body.policy.search_space != body.search_space.digest()?
        || body.program.objective.contract != body.policy.contract
        || body.program.objective.objective != body.policy.objective
        || body.policy.contract != body.search_space.contract
        || body.policy.objective != body.search_space.objective
    {
        return Err(failure(
            ErrorCode::SourceMismatch,
            "patch source/contract/space binding differs",
        ));
    }
    let expected: BTreeMap<_, _> = body
        .program
        .artifacts
        .iter()
        .map(|(id, artifact)| (id.clone(), artifact.digest.clone()))
        .collect();
    if body.operands != expected
        || body.smoke_suite.digest()? != body.program.verification.smoke_suite
    {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "patch operand/smoke references differ",
        ));
    }
    if body.compiler.is_empty() || body.compiler_version.is_empty() {
        return Err(failure(
            ErrorCode::InvalidValue,
            "compiler identity/version required",
        ));
    }
    for effect in &body.cumulative_effects {
        effect.validate_at("/cumulative_effects")?;
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct PatchGraph {
    records: BTreeMap<ContentDigest, PatchRecord>,
}
fn visit(
    id: &ContentDigest,
    records: &BTreeMap<ContentDigest, PatchRecord>,
    active: &mut BTreeSet<ContentDigest>,
    done: &mut BTreeSet<ContentDigest>,
    order: &mut Vec<ContentDigest>,
) -> Result<()> {
    if done.contains(id) {
        return Ok(());
    }
    let record = records
        .get(id)
        .ok_or_else(|| failure(ErrorCode::SourceMismatch, "patch ancestor is missing"))?;
    if !active.insert(id.clone()) {
        return Err(failure(ErrorCode::InvalidValue, "cyclic patch ancestry"));
    }
    for parent in &record.body.parents {
        visit(parent, records, active, done, order)?;
    }
    active.remove(id);
    done.insert(id.clone());
    order.push(id.clone());
    Ok(())
}
impl PatchGraph {
    pub fn from_records(records: Vec<PatchRecord>) -> Result<Self> {
        if records.is_empty() || records.len() > 64 {
            return Err(failure(
                ErrorCode::LimitExceeded,
                "patch graph requires 1..64 records",
            ));
        }
        let mut map = BTreeMap::new();
        for record in records {
            if record.body.parents.len() > 16 {
                return Err(failure(
                    ErrorCode::LimitExceeded,
                    "patch parent cardinality exceeded",
                ));
            }
            if map.insert(record.id.clone(), record).is_some() {
                return Err(failure(
                    ErrorCode::DuplicateKey,
                    "duplicate patch record ID",
                ));
            }
        }
        let mut active = BTreeSet::new();
        let mut done = BTreeSet::new();
        let mut order = Vec::new();
        for id in map.keys() {
            visit(id, &map, &mut active, &mut done, &mut order)?;
        }
        for record in map.values() {
            validate_body(&record.body)?;
            if hash(&record.body)? != record.id {
                return Err(failure(
                    ErrorCode::IntegrityMismatch,
                    "patch content identity differs",
                ));
            }
            let mut candidates = Vec::new();
            let mut seen = BTreeSet::new();
            for id in &record.body.parents {
                if !seen.insert(id) {
                    return Err(failure(ErrorCode::DuplicateKey, "duplicate patch parent"));
                }
                let parent = &map[id];
                candidates.push(parent.body.expected_candidate.clone());
                if parent.body.root_model != record.body.root_model
                    || parent.body.search_space.digest()? != record.body.search_space.digest()?
                {
                    return Err(failure(
                        ErrorCode::SourceMismatch,
                        "patch parent root/space differs",
                    ));
                }
            }
            candidates.sort();
            let mut requested = record.body.program.lineage.parent_candidates.clone();
            requested.sort();
            if requested != candidates {
                return Err(failure(
                    ErrorCode::SourceMismatch,
                    "program/record parent identities differ",
                ));
            }
            if let Some(parent) = record.body.parents.first() {
                if record.body.parents.len() == 1
                    && record.body.base_model != map[parent].body.expected_model
                {
                    return Err(failure(
                        ErrorCode::SourceMismatch,
                        "implicit patch rebase is forbidden",
                    ));
                }
            } else if record.body.base_model != record.body.root_model {
                return Err(failure(
                    ErrorCode::SourceMismatch,
                    "root patch base differs",
                ));
            }
        }
        Ok(Self { records: map })
    }
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let document = CanonicalDocument::parse(bytes, Limits::default())?;
        let records = serde_json::from_slice(document.bytes())
            .map_err(|_| failure(ErrorCode::InvalidValue, "invalid patch graph wire records"))?;
        Self::from_records(records)
    }
    pub fn record(&self, id: &ContentDigest) -> Option<&PatchRecord> {
        self.records.get(id)
    }
    pub fn restore_root(
        &self,
        id: &ContentDigest,
        root: &SourceState,
        expected_checkpoint: &ContentDigest,
    ) -> Result<SourceState> {
        let record = self
            .records
            .get(id)
            .ok_or_else(|| failure(ErrorCode::SourceMismatch, "patch record missing"))?;
        root.verify_integrity()?;
        let binding = root.binding(record.body.root_model.architecture_graph.clone());
        if binding != record.body.root_model || &binding.checkpoint != expected_checkpoint {
            return Err(failure(
                ErrorCode::SourceMismatch,
                "restore requires the exact immutable root checkpoint",
            ));
        }
        Ok(root.clone())
    }
    fn order(&self, target: &ContentDigest) -> Result<Vec<ContentDigest>> {
        let mut order = Vec::new();
        visit(
            target,
            &self.records,
            &mut BTreeSet::new(),
            &mut BTreeSet::new(),
            &mut order,
        )?;
        Ok(order)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ReplayLimits {
    pub max_nodes: usize,
    pub max_operand_bytes: u64,
    pub max_materialized_bytes: u64,
    pub max_kernel_work: u64,
    pub max_verification_bytes: u64,
    pub max_smoke_work: u64,
    pub history: HistoryLimits,
}
impl Default for ReplayLimits {
    fn default() -> Self {
        Self {
            max_nodes: 64,
            max_operand_bytes: 67_108_864,
            max_materialized_bytes: 134_217_728,
            max_kernel_work: 10_000_000,
            max_verification_bytes: 268_435_456,
            max_smoke_work: 10_000_000,
            history: HistoryLimits::default(),
        }
    }
}
#[derive(Debug)]
pub struct ReplayedPatch {
    pub history: VerifiedHistory,
    pub nodes_replayed: usize,
    pub admitted_operand_bytes: u64,
    pub materialized_payload_bytes: u64,
}

pub fn replay(
    graph: &PatchGraph,
    target: &ContentDigest,
    root: &SourceState,
    blobs: &BTreeMap<ContentDigest, Arc<[u8]>>,
    limits: ReplayLimits,
) -> Result<ReplayedPatch> {
    if limits.max_nodes == 0
        || limits.max_nodes > 64
        || limits.max_operand_bytes == 0
        || limits.max_materialized_bytes == 0
        || limits.max_kernel_work == 0
        || limits.max_verification_bytes == 0
        || limits.max_smoke_work == 0
    {
        return Err(failure(
            ErrorCode::InvalidValue,
            "finite positive replay limits required",
        ));
    }
    let order = graph.order(target)?;
    if order.len() > limits.max_nodes {
        return Err(failure(
            ErrorCode::LimitExceeded,
            "replay node budget exceeded",
        ));
    }
    let leaf = &graph.records[target];
    if root.binding(leaf.body.root_model.architecture_graph.clone()) != leaf.body.root_model {
        return Err(failure(
            ErrorCode::SourceMismatch,
            "replay root checkpoint/manifest differs",
        ));
    }
    let root_observations = root.observations();
    let root_inventory = verify_inventory(
        root.manifest(),
        &root_observations,
        InventoryLimits::default(),
    )?;
    let mut verified = BTreeMap::<ContentDigest, VerifiedHistory>::new();
    let mut operand_bytes = 0;
    let mut materialized = 0;
    let mut kernel = 0;
    let mut verification = 0;
    let mut smoke_work = 0;
    for id in &order {
        let record = &graph.records[id];
        let body = &record.body;
        if body.parents.len() > 1 {
            return Err(failure(
                ErrorCode::UnsupportedCapability,
                "multi-parent replay requires a qualified reconciliation protocol",
            ));
        }
        let parent = body.parents.first().map(|id| &verified[id]);
        let source = parent.map_or(root, VerifiedHistory::source_state);
        if source.binding(body.base_model.architecture_graph.clone()) != body.base_model {
            return Err(failure(
                ErrorCode::SourceMismatch,
                "replay node source binding differs",
            ));
        }
        let space = Validated::new(body.search_space.clone())?;
        let resolved = resolve_space(&space, &root_inventory)?;
        let program = Validated::new(body.program.clone())?;
        let policy = Validated::new(body.policy.clone())?;
        let suite = Validated::new(body.smoke_suite.clone())?;
        let mut operands = BTreeMap::new();
        for (name, digest) in &body.operands {
            let bytes = blobs.get(digest).ok_or_else(|| {
                failure(
                    ErrorCode::IntegrityMismatch,
                    "replay operand artifact missing",
                )
            })?;
            operand_bytes = add(operand_bytes, bytes.len() as u64)?;
            if operand_bytes > limits.max_operand_bytes {
                return Err(failure(
                    ErrorCode::LimitExceeded,
                    "replay operand admission budget exceeded",
                ));
            }
            operands.insert(name.clone(), bytes.clone());
        }
        let observations = source.observations();
        let effects = parent
            .map(|parent| vec![parent.prior_effects()])
            .unwrap_or_default();
        let plan = compile(
            &program,
            SourceSnapshot {
                manifest: source.manifest(),
                tensors: &observations,
                checkpoint: source.checkpoint_bytes(),
                assets: source.assets(),
            },
            &resolved,
            &operands,
            &policy,
            &effects,
        )?;
        if plan.digest() != &body.expected_plan
            || plan.summary().compiler != body.compiler
            || plan.summary().compiler_version != body.compiler_version
            || plan.summary().backend != body.backend_digest
        {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "replayed compiler/backend/plan identity differs",
            ));
        }
        if add(kernel, plan.summary().allocations.kernel_work_units)? > limits.max_kernel_work {
            return Err(failure(
                ErrorCode::LimitExceeded,
                "replay kernel budget exceeded",
            ));
        }
        if verification >= limits.max_verification_bytes || smoke_work >= limits.max_smoke_work {
            return Err(failure(
                ErrorCode::LimitExceeded,
                "replay verification/smoke budget exhausted",
            ));
        }
        let mut history_limits = limits.history;
        history_limits.execution.max_verification_bytes = history_limits
            .execution
            .max_verification_bytes
            .min(limits.max_verification_bytes.saturating_sub(verification));
        history_limits.execution.max_smoke_work = history_limits
            .execution
            .max_smoke_work
            .min(limits.max_smoke_work.saturating_sub(smoke_work));
        let payload = source.payload_bytes()?;
        let remaining = limits
            .max_materialized_bytes
            .saturating_sub(materialized)
            .checked_sub(payload)
            .ok_or_else(|| {
                failure(
                    ErrorCode::LimitExceeded,
                    "replay materialization budget exceeded",
                )
            })?;
        history_limits
            .materialization
            .serialization
            .max_checkpoint_bytes = history_limits
            .materialization
            .serialization
            .max_checkpoint_bytes
            .min(usize::try_from(remaining).unwrap_or(usize::MAX));
        history_limits.materialization.load.max_checkpoint_bytes = history_limits
            .materialization
            .load
            .max_checkpoint_bytes
            .min(
                history_limits
                    .materialization
                    .serialization
                    .max_checkpoint_bytes,
            );
        let history = certify_history(
            HistoryRequest {
                plan: &plan,
                source,
                smoke: &suite,
                space: &resolved,
                parent,
                expected_candidate: Some(&body.expected_candidate),
            },
            history_limits,
        )?;
        if history.resulting_model() != &body.expected_model
            || history.receipt().after_buffers != body.expected_buffers
            || history.prior_effects().cumulative_writes != body.cumulative_effects
            || history.costs() != &body.costs
        {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "replayed state/effects/cost identities differ",
            ));
        }
        materialized = add(
            materialized,
            add(
                history.source_state().payload_bytes()?,
                history.source_state().checkpoint_bytes().len() as u64,
            )?,
        )?;
        if materialized > limits.max_materialized_bytes {
            return Err(failure(
                ErrorCode::LimitExceeded,
                "replay materialized payload budget exceeded",
            ));
        }
        kernel = add(kernel, plan.summary().allocations.kernel_work_units)?;
        verification = add(
            verification,
            history.receipt().verification_byte_upper_bound,
        )?;
        smoke_work = add(smoke_work, history.receipt().smoke_work_upper_bound)?;
        verified.insert(id.clone(), history);
    }
    Ok(ReplayedPatch {
        history: verified.remove(target).unwrap(),
        nodes_replayed: order.len(),
        admitted_operand_bytes: operand_bytes,
        materialized_payload_bytes: materialized,
    })
}
