//! Unforgeable-in-safe-Rust parent witnesses minted by actual bounded replay.
//! Persisted records and PriorEffects DTOs cannot deserialize into this type.

use crate::transaction::{
    execute, execute_with_history, ExecutionLimits, MaterializationLimits, SmokeSuite, SourceState,
    TransactionReceipt,
};
use nf_core::canonical::ContentDigest;
use nf_core::error::{CoreError, ErrorCode, Result};
use nf_core::schemas::{Adaptation, MethodRecord, ProductionHistory, Validated};
use nf_weight_ir::compilation::CompiledPlan;
use nf_weight_ir::program::{ModelBinding, TensorRegion};
use nf_weight_ir::search_space::{PriorEffects, ResolvedSearchSpace};
use serde::{Deserialize, Serialize};

fn failure(code: ErrorCode, message: &str) -> CoreError {
    CoreError::new(code, "", message)
}
fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| failure(ErrorCode::LimitExceeded, "history cost overflow"))
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HistoryCosts {
    pub kernel_work_upper_bound: u64,
    pub verification_byte_upper_bound: u64,
    pub smoke_work_upper_bound: u64,
    pub serialized_checkpoint_bytes: u64,
}
#[derive(Debug, Clone)]
pub struct VerifiedHistory {
    candidate: ContentDigest,
    source: SourceState,
    binding: ModelBinding,
    root_manifest: ContentDigest,
    space: ResolvedSearchSpace,
    cumulative_writes: Vec<TensorRegion>,
    receipt: TransactionReceipt,
    depth: u32,
    costs: HistoryCosts,
}
impl VerifiedHistory {
    pub fn candidate_id(&self) -> &ContentDigest {
        &self.candidate
    }
    pub fn source_state(&self) -> &SourceState {
        &self.source
    }
    pub fn resulting_model(&self) -> &ModelBinding {
        &self.binding
    }
    pub fn root_manifest(&self) -> &ContentDigest {
        &self.root_manifest
    }
    pub fn receipt(&self) -> &TransactionReceipt {
        &self.receipt
    }
    pub fn depth(&self) -> u32 {
        self.depth
    }
    pub fn costs(&self) -> &HistoryCosts {
        &self.costs
    }
    pub fn prior_effects(&self) -> PriorEffects {
        PriorEffects {
            candidate: self.candidate.clone(),
            root_manifest: self.root_manifest.clone(),
            search_space: self.space.digest().clone(),
            resulting_model: self.binding.clone(),
            cumulative_writes: self.cumulative_writes.clone(),
        }
    }
    pub(crate) fn check_child(&self, plan: &CompiledPlan, source: &SourceState) -> Result<()> {
        if plan.program().lineage.parent_candidates.as_slice() != [self.candidate.clone()]
            || plan.program().base_model != self.binding
            || source.binding(self.binding.architecture_graph.clone()) != self.binding
            || plan.program().objective.search_space != *self.space.digest()
            || plan.program().objective.contract != self.space.definition().contract
            || plan.program().objective.objective != self.space.definition().objective
        {
            return Err(failure(
                ErrorCode::SourceMismatch,
                "child does not match verified parent/source/root scope",
            ));
        }
        let parent = &self.receipt.method;
        let child = &plan.program().lineage.method;
        preserve_method(parent, child)?;
        let checked = self.space.check_program(
            &Validated::new(plan.program().clone())?,
            &[self.prior_effects()],
        )?;
        if checked.affected_parameter_upper_bound != plan.summary().affected_parameter_upper_bound
            || checked.pending_quality_constraints != plan.summary().pending_quality_constraints
        {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "compiled ancestry bounds differ from verified parent effects",
            ));
        }
        Ok(())
    }
}
fn preserve_method(parent: &MethodRecord, child: &MethodRecord) -> Result<()> {
    if (parent.production_history != ProductionHistory::DirectOnly
        && child.production_history == ProductionHistory::DirectOnly)
        || (parent.production_history == ProductionHistory::ExternalUnverified
            && child.production_history != ProductionHistory::ExternalUnverified)
        || (parent.production_history == ProductionHistory::Hybrid
            && child.production_history != ProductionHistory::Hybrid)
        || (parent.adaptation != Adaptation::None && child.adaptation != parent.adaptation)
        || (parent.inspection_uses_backprop && !child.inspection_uses_backprop)
        || (parent.parameter_update_uses_backprop && !child.parameter_update_uses_backprop)
        || (parent.gradient_derived_update_values && !child.gradient_derived_update_values)
        || (parent.alignment_fitted && !child.alignment_fitted)
        || child.optimizer_steps < parent.optimizer_steps
    {
        return Err(failure(
            ErrorCode::PermissionDenied,
            "child cannot erase parent method/inspection/fitting history",
        ));
    }
    Ok(())
}

pub struct HistoryRequest<'a> {
    pub plan: &'a CompiledPlan,
    pub source: &'a SourceState,
    pub smoke: &'a Validated<SmokeSuite>,
    pub space: &'a ResolvedSearchSpace,
    pub parent: Option<&'a VerifiedHistory>,
    /// None creates a new witness; replay must bind the recorded expected ID.
    pub expected_candidate: Option<&'a ContentDigest>,
}
#[derive(Debug, Clone, Copy)]
pub struct HistoryLimits {
    pub execution: ExecutionLimits,
    pub materialization: MaterializationLimits,
    pub max_depth: u32,
}
impl Default for HistoryLimits {
    fn default() -> Self {
        Self {
            execution: ExecutionLimits::default(),
            materialization: MaterializationLimits::default(),
            max_depth: 64,
        }
    }
}

pub fn certify_history(
    request: HistoryRequest<'_>,
    limits: HistoryLimits,
) -> Result<VerifiedHistory> {
    if limits.max_depth == 0 || limits.max_depth > 64 {
        return Err(failure(
            ErrorCode::InvalidValue,
            "bounded history depth required",
        ));
    }
    let depth = request.parent.map_or(1, |parent| parent.depth + 1);
    if depth > limits.max_depth {
        return Err(failure(ErrorCode::LimitExceeded, "history depth exceeded"));
    }
    if request.space.digest() != &request.plan.summary().search_space {
        return Err(failure(
            ErrorCode::SourceMismatch,
            "history search-space binding differs",
        ));
    }
    let root_manifest = if let Some(parent) = request.parent {
        if parent.space.digest() != request.space.digest() {
            return Err(failure(
                ErrorCode::SourceMismatch,
                "parent frozen search space differs",
            ));
        }
        parent.root_manifest.clone()
    } else {
        if request
            .source
            .binding(request.plan.program().base_model.architecture_graph.clone())
            != request.space.definition().base_model
        {
            return Err(failure(
                ErrorCode::SourceMismatch,
                "history root binding differs",
            ));
        }
        request.source.manifest_digest().clone()
    };
    let parents = request
        .parent
        .map(|parent| vec![parent.prior_effects()])
        .unwrap_or_default();
    request
        .space
        .check_program(&Validated::new(request.plan.program().clone())?, &parents)?;
    let candidate = match request.parent {
        Some(parent) => execute_with_history(
            request.plan,
            request.source,
            request.smoke,
            limits.execution,
            parent,
        )?,
        None => execute(
            request.plan,
            request.source,
            request.smoke,
            limits.execution,
        )?,
    };
    if request
        .expected_candidate
        .is_some_and(|expected| candidate.digest() != expected)
    {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "exact candidate replay identity differs",
        ));
    }
    let mut writes = request
        .parent
        .map(|parent| parent.cumulative_writes.clone())
        .unwrap_or_default();
    for write in &request.plan.summary().effective_writes {
        if !writes.contains(write) {
            writes.push(write.clone());
        }
    }
    if writes.len() > 1024 {
        return Err(failure(
            ErrorCode::LimitExceeded,
            "cumulative history region count exceeded",
        ));
    }
    let source = candidate.materialize_source(limits.materialization)?;
    let previous = request
        .parent
        .map(|parent| parent.costs.clone())
        .unwrap_or_default();
    let costs = HistoryCosts {
        kernel_work_upper_bound: add(
            previous.kernel_work_upper_bound,
            request.plan.summary().allocations.kernel_work_units,
        )?,
        verification_byte_upper_bound: add(
            previous.verification_byte_upper_bound,
            candidate.receipt().verification_byte_upper_bound,
        )?,
        smoke_work_upper_bound: add(
            previous.smoke_work_upper_bound,
            candidate.receipt().smoke_work_upper_bound,
        )?,
        serialized_checkpoint_bytes: add(
            previous.serialized_checkpoint_bytes,
            source.checkpoint_bytes().len() as u64,
        )?,
    };
    Ok(VerifiedHistory {
        candidate: candidate.digest().clone(),
        binding: source.binding(request.plan.program().base_model.architecture_graph.clone()),
        source,
        root_manifest,
        space: request.space.clone(),
        cumulative_writes: writes,
        receipt: candidate.receipt().clone(),
        depth,
        costs,
    })
}
