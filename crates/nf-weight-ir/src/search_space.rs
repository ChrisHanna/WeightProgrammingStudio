//! Frozen permitted exploration, separate from IR and search strategy.
//! Domain success is static only; it is never an execution or promotion token.

use crate::program::{
    element_count, identifier, invalid, MapSide, ModelBinding, Operation, OperationKind, Program,
    TensorRegion, TensorRole,
};
use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use nf_core::error::{CoreError, ErrorCode, Result};
use nf_core::schemas::{DType, ProductionHistory, TargetUpdate, Validate, Validated};
use nf_graph::inventory::{InventoryEntry, TensorRole as ObservedRole, VerifiedInventory};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FloatRange {
    pub min: f64,
    pub max: f64,
}
impl FloatRange {
    fn validate(&self) -> Result<()> {
        if !self.min.is_finite() || !self.max.is_finite() || self.min > self.max {
            return Err(invalid("", "invalid finite coefficient range"));
        }
        Ok(())
    }
    fn contains(&self, value: f64) -> bool {
        value.is_finite() && value >= self.min && value <= self.max
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RankRange {
    pub min: u64,
    pub max: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperationDomain {
    pub dtypes: Vec<DType>,
    pub coefficient: Option<FloatRange>,
    pub rank: Option<RankRange>,
    pub sides: BTreeSet<MapSide>,
    pub max_operand_bytes: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Selector {
    Tensor { name: String },
    Prefix { prefix: String },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SliceBox {
    pub start: Vec<u64>,
    pub shape: Vec<u64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum RegionDomain {
    WholeTensor,
    Boxes { regions: Vec<SliceBox> },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SiteRule {
    pub selector: Selector,
    pub role: TensorRole,
    pub regions: RegionDomain,
    pub allowed_axes: BTreeSet<u32>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationSplit {
    Selection,
    Holdout,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualityLimit {
    pub binding: ContentDigest,
    pub metric: String,
    pub units: String,
    pub baseline: ContentDigest,
    pub max_regression: f64,
    pub split: EvaluationSplit,
    pub uncertainty_protocol: ContentDigest,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpaceConstraints {
    pub max_changed_parameters: u64,
    pub max_artifact_bytes: u64,
    pub max_temporary_bytes: u64,
    pub max_operations: usize,
    pub allow_buffer_writes: bool,
    pub require_direct_weight_edits: bool,
    pub require_fully_gradient_free: bool,
    pub require_preserve_shapes: bool,
    pub quality_limits: Vec<QualityLimit>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalStrategy {
    Scripted,
    CausalHypothesis,
    LearnedProposer,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchStrategy {
    Sequential,
    Branch,
    Evolutionary,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionStrategy {
    Pareto,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StrategyBinding {
    pub proposal: ProposalStrategy,
    pub search: SearchStrategy,
    pub selection: SelectionStrategy,
    pub engine: String,
    pub engine_version: String,
    pub seed: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchSpace {
    pub schema_version: u32,
    pub base_model: ModelBinding,
    pub contract: ContentDigest,
    pub objective: ContentDigest,
    pub transformations: BTreeMap<OperationKind, OperationDomain>,
    pub sites: Vec<SiteRule>,
    pub constraints: SpaceConstraints,
    pub strategy: StrategyBinding,
}
impl Validate for SearchSpace {
    fn validate(&self) -> Result<()> {
        if self.schema_version != 1 {
            return Err(CoreError::new(
                ErrorCode::UnsupportedVersion,
                "/schema_version",
                "only search-space version 1 is supported",
            ));
        }
        if self.transformations.is_empty()
            || self.sites.is_empty()
            || self.sites.len() > 1024
            || self.constraints.max_operations > 1024
            || self.constraints.max_temporary_bytes == 0
            || self.constraints.quality_limits.is_empty()
            || self.constraints.quality_limits.len() > 1024
        {
            return Err(invalid("", "invalid or unbounded search-space declaration"));
        }
        identifier(&self.strategy.engine, "/strategy/engine")?;
        identifier(&self.strategy.engine_version, "/strategy/engine_version")?;
        for (kind, domain) in &self.transformations {
            if domain.dtypes.is_empty() || domain.dtypes.len() > 8 {
                return Err(invalid(
                    "/transformations",
                    "explicit bounded dtype domain required",
                ));
            }
            for (index, dtype) in domain.dtypes.iter().enumerate() {
                if domain.dtypes[..index].contains(dtype) {
                    return Err(invalid("/transformations", "duplicate dtype domain"));
                }
            }
            let coefficient = matches!(
                kind,
                OperationKind::Scale
                    | OperationKind::SubspaceProjection
                    | OperationKind::LowRankUpdate
            );
            let ranked = matches!(
                kind,
                OperationKind::SubspaceProjection | OperationKind::LowRankUpdate
            );
            let sided = matches!(
                kind,
                OperationKind::SubspaceProjection | OperationKind::LinearMap
            );
            if coefficient != domain.coefficient.is_some()
                || ranked != domain.rank.is_some()
                || sided == domain.sides.is_empty()
            {
                return Err(invalid(
                    "/transformations",
                    "operation domain fields do not match its semantics",
                ));
            }
            if let Some(range) = &domain.coefficient {
                range.validate()?;
            }
            if let Some(range) = &domain.rank {
                if range.min == 0 || range.min > range.max {
                    return Err(invalid(
                        "/transformations",
                        "rank domain must be positive and ordered",
                    ));
                }
            }
        }
        for rule in &self.sites {
            let name = match &rule.selector {
                Selector::Tensor { name } => name,
                Selector::Prefix { prefix } => prefix,
            };
            if name.is_empty() || name.len() > 512 || name.chars().any(char::is_control) {
                return Err(invalid("/sites", "invalid exact/prefix selector"));
            }
            if rule.allowed_axes.iter().any(|axis| *axis >= 32) {
                return Err(invalid("/sites", "axis exceeds supported rank"));
            }
            if let RegionDomain::Boxes { regions } = &rule.regions {
                if regions.is_empty() || regions.len() > 1024 {
                    return Err(invalid(
                        "/sites",
                        "bounded nonempty coordinate boxes required",
                    ));
                }
                for region in regions {
                    TensorRegion {
                        tensor: name.clone(),
                        start: region.start.clone(),
                        shape: region.shape.clone(),
                    }
                    .validate_at("/sites")?;
                }
            }
        }
        for quality in &self.constraints.quality_limits {
            identifier(&quality.metric, "/constraints/quality_limits/metric")?;
            identifier(&quality.units, "/constraints/quality_limits/units")?;
            if !quality.max_regression.is_finite() || quality.max_regression < 0.0 {
                return Err(invalid(
                    "/constraints/quality_limits",
                    "regression limit must be finite and nonnegative",
                ));
            }
        }
        Ok(())
    }
}
impl SearchSpace {
    pub fn digest(&self) -> Result<ContentDigest> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).map_err(|error| invalid("", error.to_string()))?;
        Ok(CanonicalDocument::parse(&bytes, Limits::default())?.digest())
    }
}
pub fn parse_search_space(bytes: &[u8]) -> Result<Validated<SearchSpace>> {
    Validated::parse(bytes)
}

#[derive(Debug, Clone)]
struct ResolvedSite {
    region: TensorRegion,
    axes: BTreeSet<u32>,
}
#[derive(Debug, Clone)]
pub struct ResolvedSearchSpace {
    space: SearchSpace,
    digest: ContentDigest,
    sites: BTreeMap<String, Vec<ResolvedSite>>,
    inventory: BTreeMap<String, InventoryEntry>,
    aliases: BTreeMap<String, Vec<String>>,
}

fn contains(outer: &TensorRegion, inner: &TensorRegion) -> bool {
    outer.tensor == inner.tensor
        && outer.shape.len() == inner.shape.len()
        && outer
            .start
            .iter()
            .zip(&outer.shape)
            .zip(inner.start.iter().zip(&inner.shape))
            .all(|((a, width_a), (b, width_b))| {
                *a <= *b
                    && a.checked_add(*width_a)
                        .zip(b.checked_add(*width_b))
                        .is_some_and(|(end_a, end_b)| end_b <= end_a)
            })
}

pub fn resolve_space(
    space: &Validated<SearchSpace>,
    inventory: &VerifiedInventory,
) -> Result<ResolvedSearchSpace> {
    if &space.base_model.manifest != inventory.source_manifest_digest()
        || &space.base_model.parameter_digest != inventory.parameter_digest()
    {
        return Err(CoreError::new(
            ErrorCode::SourceMismatch,
            "/base_model",
            "search-space root does not match verified inventory",
        ));
    }
    let mut resolved: BTreeMap<String, Vec<ResolvedSite>> = BTreeMap::new();
    let mut resolved_count = 0usize;
    for rule in &space.sites {
        let mut matched = false;
        for (name, entry) in inventory.entries() {
            let selected = match &rule.selector {
                Selector::Tensor { name: selected } => name == selected,
                Selector::Prefix { prefix } => name.starts_with(prefix),
            };
            if !selected {
                continue;
            }
            matched = true;
            let role = match entry.role {
                ObservedRole::Parameter => TensorRole::Parameter,
                ObservedRole::Buffer => TensorRole::Buffer,
            };
            if role != rule.role
                || rule
                    .allowed_axes
                    .iter()
                    .any(|axis| *axis as usize >= entry.shape.len())
            {
                return Err(invalid(
                    "/sites",
                    "selector role or axis does not match actual tensor",
                ));
            }
            let whole = TensorRegion {
                tensor: name.clone(),
                start: vec![0; entry.shape.len()],
                shape: entry.shape.clone(),
            };
            let regions = match &rule.regions {
                RegionDomain::WholeTensor => vec![whole.clone()],
                RegionDomain::Boxes { regions } => regions
                    .iter()
                    .map(|region| TensorRegion {
                        tensor: name.clone(),
                        start: region.start.clone(),
                        shape: region.shape.clone(),
                    })
                    .collect(),
            };
            for region in regions {
                resolved_count += 1;
                if resolved_count > 4096 {
                    return Err(CoreError::new(
                        ErrorCode::LimitExceeded,
                        "/sites",
                        "resolved selector expansion bound exceeded",
                    ));
                }
                if !contains(&whole, &region) {
                    return Err(invalid(
                        "/sites",
                        "allowed coordinate box exceeds verified tensor",
                    ));
                }
                resolved
                    .entry(name.clone())
                    .or_default()
                    .push(ResolvedSite {
                        region,
                        axes: rule.allowed_axes.clone(),
                    });
            }
        }
        if !matched {
            return Err(invalid("/sites", "selector matched no verified tensors"));
        }
    }
    Ok(ResolvedSearchSpace {
        space: (**space).clone(),
        digest: space.digest()?,
        sites: resolved,
        inventory: inventory.entries().clone(),
        aliases: inventory.alias_groups().clone(),
    })
}

/// Data from coordinator-verified committed parent receipts. Domain validation
/// checks bindings and budgets, not receipt authenticity or artifact existence.
#[derive(Debug, Clone)]
pub struct PriorEffects {
    pub candidate: ContentDigest,
    pub root_manifest: ContentDigest,
    pub search_space: ContentDigest,
    pub resulting_model: ModelBinding,
    pub cumulative_writes: Vec<TensorRegion>,
}
#[derive(Debug, Clone)]
pub struct DomainCheck {
    pub space_digest: ContentDigest,
    pub affected_parameter_upper_bound: u64,
    pub pending_quality_constraints: usize,
}

// Subtract one box from another using disjoint slabs, without enumerating
// parameter coordinates. A finite work/fragment cap bounds adversarial input.
fn subtract(region: &TensorRegion, covered: &TensorRegion) -> Vec<TensorRegion> {
    let mut center = region.clone();
    for axis in 0..region.shape.len() {
        let start = region.start[axis].max(covered.start[axis]);
        let end = (region.start[axis] + region.shape[axis])
            .min(covered.start[axis] + covered.shape[axis]);
        if start >= end {
            return vec![region.clone()];
        }
    }
    let mut pieces = Vec::new();
    for axis in 0..region.shape.len() {
        let start = region.start[axis].max(covered.start[axis]);
        let end = (region.start[axis] + region.shape[axis])
            .min(covered.start[axis] + covered.shape[axis]);
        let center_end = center.start[axis] + center.shape[axis];
        if center.start[axis] < start {
            let mut left = center.clone();
            left.shape[axis] = start - center.start[axis];
            pieces.push(left);
            center.shape[axis] = center_end - start;
            center.start[axis] = start;
        }
        if center_end > end {
            let mut right = center.clone();
            right.start[axis] = end;
            right.shape[axis] = center_end - end;
            pieces.push(right);
            center.shape[axis] = end - center.start[axis];
        }
    }
    pieces
}

impl ResolvedSearchSpace {
    pub fn digest(&self) -> &ContentDigest {
        &self.digest
    }
    pub fn definition(&self) -> &SearchSpace {
        &self.space
    }
    /// A same-shape parent may change values, not the frozen layout or ties.
    pub fn check_inventory_layout(&self, inventory: &VerifiedInventory) -> Result<()> {
        if self.aliases != *inventory.alias_groups()
            || self.inventory.len() != inventory.entries().len()
        {
            return Err(CoreError::new(
                ErrorCode::SourceMismatch,
                "/base_model",
                "source alias/name layout differs from search-space root",
            ));
        }
        for (name, root) in &self.inventory {
            let current = inventory.entries().get(name).ok_or_else(|| {
                CoreError::new(
                    ErrorCode::SourceMismatch,
                    "/base_model",
                    "source tensor is missing",
                )
            })?;
            if root.shape != current.shape
                || root.dtype != current.dtype
                || root.role != current.role
                || root.storage_group != current.storage_group
                || root.device != current.device
            {
                return Err(CoreError::new(
                    ErrorCode::SourceMismatch,
                    "/base_model",
                    "source tensor layout differs from search-space root",
                ));
            }
        }
        Ok(())
    }
    pub fn resolved_tensor_names(&self) -> Vec<&str> {
        self.sites.keys().map(String::as_str).collect()
    }
    fn allowed_region(&self, region: &TensorRegion, axis: Option<u32>) -> Result<()> {
        region.validate_at("/effects")?;
        if !self.sites.get(&region.tensor).is_some_and(|sites| {
            sites.iter().any(|site| {
                contains(&site.region, region) && axis.is_none_or(|axis| site.axes.contains(&axis))
            })
        }) {
            return Err(CoreError::new(
                ErrorCode::PermissionDenied,
                "/effects",
                "tensor/coordinates/axis are outside frozen search space",
            ));
        }
        Ok(())
    }
    fn union_count(&self, writes: &[TensorRegion]) -> Result<u64> {
        let mut groups: BTreeMap<&str, Vec<TensorRegion>> = BTreeMap::new();
        let mut work = 0usize;
        let mut total_fragments = 0usize;
        for write in writes {
            self.allowed_region(write, None)?;
            if element_count(&write.shape, "/effects")? == 0 {
                continue;
            }
            let entry = &self.inventory[&write.tensor];
            if entry.role == ObservedRole::Buffer && !self.space.constraints.allow_buffer_writes {
                return Err(CoreError::new(
                    ErrorCode::PermissionDenied,
                    "/effects/buffer_writes",
                    "buffer updates are not allowed",
                ));
            }
            for alias in &self.aliases[&entry.storage_group] {
                let mut effective = write.clone();
                effective.tensor = alias.clone();
                self.allowed_region(&effective, None).map_err(|_| {
                    CoreError::new(
                        ErrorCode::AliasScopeEscape,
                        "/effects",
                        "all effective storage aliases must be authorized",
                    )
                })?;
            }
            let covered = groups.entry(&entry.storage_group).or_default();
            let mut pieces = vec![write.clone()];
            for earlier in covered.iter() {
                let mut remainder = Vec::new();
                for piece in pieces {
                    work += 1;
                    if work > 1_000_000 {
                        return Err(CoreError::new(
                            ErrorCode::LimitExceeded,
                            "/effects",
                            "coordinate union work bound exceeded",
                        ));
                    }
                    remainder.extend(subtract(&piece, earlier));
                    if remainder.len() > 4096 {
                        return Err(CoreError::new(
                            ErrorCode::LimitExceeded,
                            "/effects",
                            "coordinate union fragmentation bound exceeded",
                        ));
                    }
                }
                pieces = remainder;
            }
            total_fragments += pieces.len();
            if total_fragments > 4096 {
                return Err(CoreError::new(
                    ErrorCode::LimitExceeded,
                    "/effects",
                    "global coordinate union storage bound exceeded",
                ));
            }
            covered.extend(pieces);
            if covered.len() > 4096 {
                return Err(CoreError::new(
                    ErrorCode::LimitExceeded,
                    "/effects",
                    "coordinate union storage bound exceeded",
                ));
            }
        }
        let mut total = 0u64;
        for regions in groups.values() {
            for region in regions {
                total = total
                    .checked_add(element_count(&region.shape, "/effects")?)
                    .ok_or_else(|| invalid("/effects", "coordinate union overflow"))?;
            }
        }
        Ok(total)
    }
    pub fn check_program(
        &self,
        program: &Validated<Program>,
        parents: &[PriorEffects],
    ) -> Result<DomainCheck> {
        program.validate()?;
        if program.objective.search_space != self.digest
            || program.objective.contract != self.space.contract
            || program.objective.objective != self.space.objective
        {
            return Err(CoreError::new(
                ErrorCode::SourceMismatch,
                "/objective",
                "program is not bound to this frozen space/contract/objective",
            ));
        }
        if parents.len() > 1 {
            return Err(CoreError::new(
                ErrorCode::UnsupportedCapability,
                "/lineage",
                "multiple-parent reconciliation requires a registered merge protocol",
            ));
        }
        let requested: BTreeSet<_> = program.lineage.parent_candidates.iter().collect();
        let supplied: BTreeSet<_> = parents.iter().map(|parent| &parent.candidate).collect();
        if requested.len() != program.lineage.parent_candidates.len() || requested != supplied {
            return Err(invalid(
                "/lineage",
                "exact committed-parent effect records are required",
            ));
        }
        let expected = parents
            .first()
            .map_or(&self.space.base_model, |parent| &parent.resulting_model);
        if &program.base_model != expected
            || program.base_model.architecture_graph != self.space.base_model.architecture_graph
            || program.base_model.config != self.space.base_model.config
            || program.base_model.tokenizer != self.space.base_model.tokenizer
        {
            return Err(CoreError::new(
                ErrorCode::SourceMismatch,
                "/base_model",
                "source or same-shape ancestry binding differs",
            ));
        }
        let limits = &self.space.constraints;
        if program.operations.len() > limits.max_operations
            || program.constraints.max_changed_parameters > limits.max_changed_parameters
            || program.constraints.max_artifact_bytes > limits.max_artifact_bytes
            || program.constraints.max_temporary_bytes > limits.max_temporary_bytes
        {
            return Err(CoreError::new(
                ErrorCode::ParameterBudgetExceeded,
                "/constraints",
                "proposal weakens a frozen resource bound",
            ));
        }
        if limits.require_preserve_shapes
            && (!program.constraints.preserve_tensor_shapes
                || program.effects.graph_changes
                || !program.effects.config_changes.is_empty()
                || program.effects.tokenizer_changes)
        {
            return Err(CoreError::new(
                ErrorCode::PermissionDenied,
                "/effects",
                "shape/config/tokenizer edits are outside this space",
            ));
        }
        if limits.require_direct_weight_edits
            && (!program.constraints.gradient_free
                || program.lineage.method.production_history != ProductionHistory::DirectOnly
                || !matches!(
                    program.lineage.method.target_update,
                    TargetUpdate::DirectTensorEdit | TargetUpdate::NoChange
                ))
        {
            return Err(CoreError::new(
                ErrorCode::PermissionDenied,
                "/lineage/method",
                "space requires direct parameter edits",
            ));
        }
        if limits.require_fully_gradient_free {
            program.lineage.method.require_fully_gradient_free()?;
        }
        let mut writes = Vec::new();
        for parent in parents {
            if parent.root_manifest != self.space.base_model.manifest
                || parent.search_space != self.digest
                || parent.cumulative_writes.len() > 1024
            {
                return Err(CoreError::new(
                    ErrorCode::SourceMismatch,
                    "/lineage",
                    "parent effect scope differs",
                ));
            }
            writes.extend(parent.cumulative_writes.clone());
        }
        for instruction in &program.operations {
            let operation = &instruction.operation;
            let domain = self
                .space
                .transformations
                .get(&operation.kind())
                .ok_or_else(|| {
                    CoreError::new(
                        ErrorCode::UnsupportedOperation,
                        "/operations",
                        "operation is valid IR but not in this search space",
                    )
                })?;
            let entry = self
                .inventory
                .get(&operation.target().tensor)
                .ok_or_else(|| invalid("/operations", "unknown tensor"))?;
            if !domain.dtypes.contains(&entry.dtype) {
                return Err(invalid(
                    "/operations",
                    "tensor dtype is outside operation domain",
                ));
            }
            let (axis, coefficient, rank) = match operation {
                Operation::Scale { factor, .. } => (None, Some(*factor), None),
                Operation::PermuteAxis { axis, .. } => (Some(*axis), None, None),
                Operation::LinearMap { side, .. } => {
                    if !domain.sides.contains(side) {
                        return Err(invalid("/operations", "map side is forbidden"));
                    }
                    (Some(if *side == MapSide::Left { 0 } else { 1 }), None, None)
                }
                Operation::SubspaceProjection {
                    side,
                    basis,
                    strength,
                    ..
                } => {
                    if !domain.sides.contains(side) {
                        return Err(invalid("/operations", "projection side is forbidden"));
                    }
                    let shape = &program.artifacts[basis].shape;
                    if shape.len() != 2 {
                        return Err(invalid("/artifacts", "basis must be a matrix"));
                    }
                    (
                        Some(if *side == MapSide::Left { 0 } else { 1 }),
                        Some(*strength),
                        Some(shape[1]),
                    )
                }
                Operation::LowRankUpdate {
                    left,
                    right,
                    strength,
                    ..
                } => {
                    let a = &program.artifacts[left].shape;
                    let b = &program.artifacts[right].shape;
                    if a.len() != 2 || b.len() != 2 || a[1] != b[0] {
                        return Err(invalid("/artifacts", "low-rank factor dimensions differ"));
                    }
                    (None, Some(*strength), Some(a[1]))
                }
                _ => (None, None, None),
            };
            if coefficient.is_some_and(|value| {
                !domain
                    .coefficient
                    .as_ref()
                    .is_some_and(|range| range.contains(value))
            }) || rank.is_some_and(|value| {
                !domain
                    .rank
                    .as_ref()
                    .is_some_and(|range| value >= range.min && value <= range.max)
            }) {
                return Err(invalid(
                    "/operations",
                    "coefficient/rank is outside frozen domain",
                ));
            }
            self.allowed_region(operation.target(), axis)?;
            for alias in &self.aliases[&entry.storage_group] {
                let mut effective = operation.target().clone();
                effective.tensor = alias.clone();
                self.allowed_region(&effective, axis).map_err(|_| {
                    CoreError::new(
                        ErrorCode::AliasScopeEscape,
                        "/operations",
                        "operation axis/coordinates must be permitted for every storage alias",
                    )
                })?;
            }
            let mut operand_bytes = 0u64;
            for id in operation
                .artifact_ids()
                .into_iter()
                .collect::<BTreeSet<_>>()
            {
                operand_bytes = operand_bytes
                    .checked_add(program.artifacts[id].artifact_bytes)
                    .filter(|sum| *sum <= domain.max_operand_bytes)
                    .ok_or_else(|| {
                        invalid("/artifacts", "operand bytes exceed operation domain")
                    })?;
            }
            writes.push(operation.target().clone());
        }
        writes.extend(program.effects.writes.clone());
        writes.extend(program.effects.buffer_writes.clone());
        for read in &program.effects.reads {
            self.allowed_region(read, None)?;
        }
        let changed = self.union_count(&writes)?;
        if changed > limits.max_changed_parameters
            || changed > program.constraints.max_changed_parameters
        {
            return Err(CoreError::new(
                ErrorCode::ParameterBudgetExceeded,
                "/constraints/max_changed_parameters",
                "cumulative physical-coordinate upper bound exceeded",
            ));
        }
        for quality in &limits.quality_limits {
            if !program
                .verification
                .gyms
                .iter()
                .any(|gym| gym.binding == quality.binding)
            {
                return Err(invalid(
                    "/verification",
                    "required quality binding is absent",
                ));
            }
        }
        Ok(DomainCheck {
            space_digest: self.digest.clone(),
            affected_parameter_upper_bound: changed,
            pending_quality_constraints: limits.quality_limits.len(),
        })
    }
}
