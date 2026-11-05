//! Weight IR v1: bounded data, never Python expressions, paths, or training code.
//! Parsing checks the declared structure; compilation must bind actual inventory,
//! artifacts, contract permissions, search space, and trusted verifier policies.

use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use nf_core::error::{child_path, CoreError, ErrorCode, Result};
use nf_core::schemas::{
    AccessClass, DType, EvidenceLabel, MethodRecord, ProductionHistory, Validate, Validated,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const IR_VERSION: u32 = 1;
pub const OPSET_VERSION: u32 = 1;
pub const MAX_INSTRUCTIONS: usize = 1024;

pub(crate) fn invalid(path: &str, message: impl Into<String>) -> CoreError {
    CoreError::new(ErrorCode::InvalidValue, path, message)
}
pub(crate) fn identifier(value: &str, path: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
    {
        return Err(invalid(path, "expected a bounded logical identifier"));
    }
    Ok(())
}
fn tensor_name(value: &str, path: &str) -> Result<()> {
    if value.is_empty() || value.len() > 512 || value.chars().any(char::is_control) {
        return Err(invalid(path, "invalid tensor name"));
    }
    Ok(())
}
fn finite(value: f64, path: &str) -> Result<()> {
    if !value.is_finite() {
        return Err(CoreError::new(
            ErrorCode::NonFinite,
            path,
            "finite numeric operand required",
        ));
    }
    Ok(())
}
pub fn element_count(shape: &[u64], path: &str) -> Result<u64> {
    if shape.len() > 32 {
        return Err(invalid(path, "rank exceeds 32"));
    }
    shape.iter().try_fold(1u64, |total, dimension| {
        total
            .checked_mul(*dimension)
            .ok_or_else(|| invalid(path, "element count overflow"))
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelBinding {
    pub checkpoint: ContentDigest,
    pub manifest: ContentDigest,
    pub parameter_digest: ContentDigest,
    pub architecture_graph: ContentDigest,
    pub config: ContentDigest,
    pub tokenizer: Option<ContentDigest>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectiveBinding {
    pub contract: ContentDigest,
    pub objective: ContentDigest,
    pub experiment_protocol: ContentDigest,
    pub search_space: ContentDigest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationKind {
    ReplaceSlice,
    AddDelta,
    Scale,
    Mask,
    PermuteAxis,
    LinearMap,
    SubspaceProjection,
    LowRankUpdate,
    ConstructRecipe,
}
impl OperationKind {
    pub fn from_wire(value: &str) -> Option<Self> {
        match value {
            "replace_slice" => Some(Self::ReplaceSlice),
            "add_delta" => Some(Self::AddDelta),
            "scale" => Some(Self::Scale),
            "mask" => Some(Self::Mask),
            "permute_axis" => Some(Self::PermuteAxis),
            "linear_map" => Some(Self::LinearMap),
            "subspace_projection" => Some(Self::SubspaceProjection),
            "low_rank_update" => Some(Self::LowRankUpdate),
            "construct_recipe" => Some(Self::ConstructRecipe),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorRegion {
    pub tensor: String,
    pub start: Vec<u64>,
    pub shape: Vec<u64>,
}
impl TensorRegion {
    pub fn validate_at(&self, path: &str) -> Result<()> {
        tensor_name(&self.tensor, &child_path(path, "tensor"))?;
        element_count(&self.shape, &child_path(path, "shape"))?;
        if self.start.len() != self.shape.len() {
            return Err(invalid(path, "slice start and shape ranks differ"));
        }
        for (start, width) in self.start.iter().zip(&self.shape) {
            start
                .checked_add(*width)
                .ok_or_else(|| invalid(path, "slice end overflow"))?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorArtifact {
    pub digest: ContentDigest,
    pub tensor: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub data_bytes: u64,
    pub artifact_bytes: u64,
    pub access: AccessClass,
    pub provenance: EvidenceLabel,
}
impl TensorArtifact {
    fn validate_at(&self, path: &str) -> Result<()> {
        tensor_name(&self.tensor, &child_path(path, "tensor"))?;
        let bytes = element_count(&self.shape, path)?
            .checked_mul(self.dtype.byte_width())
            .ok_or_else(|| invalid(path, "operand byte count overflow"))?;
        if bytes != self.data_bytes
            || self.artifact_bytes < self.data_bytes
            || self.artifact_bytes < 8
        {
            return Err(invalid(
                path,
                "Safetensors operand byte declaration is inconsistent",
            ));
        }
        if self.access == AccessClass::Holdout || self.provenance.access == AccessClass::Holdout {
            return Err(CoreError::new(
                ErrorCode::PermissionDenied,
                path,
                "holdout artifacts cannot construct edits",
            ));
        }
        if self.access != self.provenance.access {
            return Err(invalid(path, "operand provenance access class differs"));
        }
        self.provenance.validate().map_err(|mut error| {
            error.path = format!("{path}/provenance{}", error.path);
            error
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorRole {
    Parameter,
    Buffer,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorPrecondition {
    pub tensor: String,
    pub role: TensorRole,
    pub shape: Vec<u64>,
    pub dtype: DType,
    pub before_digest: ContentDigest,
    pub storage_group: String,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MapSide {
    Left,
    Right,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecipeRef {
    pub id: String,
    pub version: u32,
    pub specification: ContentDigest,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Operation {
    ReplaceSlice {
        target: TensorRegion,
        value: String,
    },
    AddDelta {
        target: TensorRegion,
        delta: String,
    },
    Scale {
        target: TensorRegion,
        factor: f64,
    },
    Mask {
        target: TensorRegion,
        mask: String,
    },
    PermuteAxis {
        target: TensorRegion,
        axis: u32,
        permutation: Vec<u64>,
    },
    LinearMap {
        target: TensorRegion,
        matrix: String,
        side: MapSide,
    },
    SubspaceProjection {
        target: TensorRegion,
        basis: String,
        side: MapSide,
        strength: f64,
        orthonormal_tolerance: f64,
    },
    LowRankUpdate {
        target: TensorRegion,
        left: String,
        right: String,
        strength: f64,
    },
    ConstructRecipe {
        target: TensorRegion,
        recipe: RecipeRef,
        inputs: Vec<String>,
    },
}
impl Operation {
    pub fn kind(&self) -> OperationKind {
        match self {
            Self::ReplaceSlice { .. } => OperationKind::ReplaceSlice,
            Self::AddDelta { .. } => OperationKind::AddDelta,
            Self::Scale { .. } => OperationKind::Scale,
            Self::Mask { .. } => OperationKind::Mask,
            Self::PermuteAxis { .. } => OperationKind::PermuteAxis,
            Self::LinearMap { .. } => OperationKind::LinearMap,
            Self::SubspaceProjection { .. } => OperationKind::SubspaceProjection,
            Self::LowRankUpdate { .. } => OperationKind::LowRankUpdate,
            Self::ConstructRecipe { .. } => OperationKind::ConstructRecipe,
        }
    }
    pub fn target(&self) -> &TensorRegion {
        match self {
            Self::ReplaceSlice { target, .. }
            | Self::AddDelta { target, .. }
            | Self::Scale { target, .. }
            | Self::Mask { target, .. }
            | Self::PermuteAxis { target, .. }
            | Self::LinearMap { target, .. }
            | Self::SubspaceProjection { target, .. }
            | Self::LowRankUpdate { target, .. }
            | Self::ConstructRecipe { target, .. } => target,
        }
    }
    pub fn artifact_ids(&self) -> Vec<&str> {
        match self {
            Self::ReplaceSlice { value, .. } => vec![value],
            Self::AddDelta { delta, .. } => vec![delta],
            Self::Mask { mask, .. } => vec![mask],
            Self::LinearMap { matrix, .. } => vec![matrix],
            Self::SubspaceProjection { basis, .. } => vec![basis],
            Self::LowRankUpdate { left, right, .. } => vec![left, right],
            Self::ConstructRecipe { inputs, .. } => inputs.iter().map(String::as_str).collect(),
            _ => Vec::new(),
        }
    }
    pub fn reads_target(&self) -> bool {
        !matches!(
            self,
            Self::ReplaceSlice { .. } | Self::ConstructRecipe { .. }
        )
    }
    fn validate_at(&self, path: &str) -> Result<()> {
        self.target().validate_at(&child_path(path, "target"))?;
        match self {
            Self::Scale { factor, .. } => finite(*factor, &child_path(path, "factor"))?,
            Self::SubspaceProjection {
                strength,
                orthonormal_tolerance,
                ..
            } => {
                finite(*strength, &child_path(path, "strength"))?;
                finite(
                    *orthonormal_tolerance,
                    &child_path(path, "orthonormal_tolerance"),
                )?;
                if *orthonormal_tolerance < 0.0 {
                    return Err(invalid(path, "orthonormal tolerance cannot be negative"));
                }
            }
            Self::LowRankUpdate { strength, .. } => {
                finite(*strength, &child_path(path, "strength"))?
            }
            Self::PermuteAxis {
                axis, permutation, ..
            } => {
                if *axis as usize >= self.target().shape.len() {
                    return Err(invalid(path, "permutation axis is outside target rank"));
                }
                if permutation.len() as u64 != self.target().shape[*axis as usize] {
                    return Err(invalid(path, "permutation length differs from target axis"));
                }
                let mut seen = BTreeSet::new();
                for value in permutation {
                    if *value >= permutation.len() as u64 || !seen.insert(*value) {
                        return Err(invalid(path, "permutation must be bijective"));
                    }
                }
            }
            Self::ConstructRecipe { recipe, .. } => {
                identifier(&recipe.id, &child_path(&child_path(path, "recipe"), "id"))?;
                if recipe.version == 0 {
                    return Err(invalid(
                        path,
                        "recipe version must be explicit and positive",
                    ));
                }
            }
            _ => (),
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Instruction {
    pub id: String,
    pub depends_on: Vec<String>,
    pub operation: Operation,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoundingMode {
    NearestTiesEven,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NonFinitePolicy {
    Reject,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericalPolicy {
    pub accumulation_dtype: DType,
    pub output_rounding: RoundingMode,
    pub non_finite: NonFinitePolicy,
    pub max_abs_parameter: f64,
    pub backend_abs_tolerance: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramConstraints {
    pub gradient_free: bool,
    pub preserve_tensor_shapes: bool,
    pub max_changed_parameters: u64,
    pub max_artifact_bytes: u64,
    pub max_temporary_bytes: u64,
    pub numerical: NumericalPolicy,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Effects {
    pub reads: Vec<TensorRegion>,
    pub writes: Vec<TensorRegion>,
    pub buffer_writes: Vec<TensorRegion>,
    pub graph_changes: bool,
    pub config_changes: BTreeSet<String>,
    pub tokenizer_changes: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Preservation {
    ExactUnderConditions {
        rule: RecipeRef,
        conditions: Vec<ContentDigest>,
    },
    BoundedError {
        metric: String,
        absolute_tolerance: f64,
        protocol: ContentDigest,
    },
    Experimental,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GymRole {
    Target,
    Retention,
    Diagnostic,
    Efficiency,
    Safety,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GymBinding {
    pub binding: ContentDigest,
    pub gym: ContentDigest,
    pub validation: ContentDigest,
    pub scorer: ContentDigest,
    pub split: ContentDigest,
    pub role: GymRole,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Verification {
    pub portfolio: ContentDigest,
    pub protocol: ContentDigest,
    pub gyms: Vec<GymBinding>,
    pub smoke_suite: ContentDigest,
    pub parity_suite: ContentDigest,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Lineage {
    pub parent_candidates: Vec<ContentDigest>,
    pub hypothesis: ContentDigest,
    pub method: MethodRecord,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Program {
    pub ir_version: u32,
    pub opset_version: u32,
    pub base_model: ModelBinding,
    pub objective: ObjectiveBinding,
    pub operations: Vec<Instruction>,
    pub artifacts: BTreeMap<String, TensorArtifact>,
    pub preconditions: Vec<TensorPrecondition>,
    pub required_capabilities: BTreeSet<String>,
    pub constraints: ProgramConstraints,
    pub effects: Effects,
    pub preservation: Preservation,
    pub verification: Verification,
    pub lineage: Lineage,
    #[serde(default)]
    pub extensions: BTreeMap<String, String>,
}

impl Program {
    pub fn digest(&self) -> Result<ContentDigest> {
        self.validate()?;
        let raw = serde_json::to_vec(self).map_err(|cause| invalid("", cause.to_string()))?;
        Ok(CanonicalDocument::parse(&raw, Limits::default())?.digest())
    }
}
impl Validate for Program {
    fn validate(&self) -> Result<()> {
        if self.ir_version != IR_VERSION || self.opset_version != OPSET_VERSION {
            return Err(CoreError::new(
                ErrorCode::UnsupportedVersion,
                "/ir_version",
                "only IR/opset version 1 is supported",
            ));
        }
        if self.operations.len() > MAX_INSTRUCTIONS
            || self.artifacts.len() > 1024
            || self.preconditions.len() > 1024
            || self.effects.reads.len() > 1024
            || self.effects.writes.len() > 1024
            || self.effects.buffer_writes.len() > 1024
            || self.required_capabilities.len() > 128
            || self.verification.gyms.len() > 1024
            || self.lineage.parent_candidates.len() > 1024
        {
            return Err(CoreError::new(
                ErrorCode::LimitExceeded,
                "",
                "program cardinality bound exceeded",
            ));
        }
        if self.constraints.max_temporary_bytes == 0 {
            return Err(invalid(
                "/constraints",
                "temporary memory bound must be positive",
            ));
        }
        let numerical = &self.constraints.numerical;
        finite(
            numerical.max_abs_parameter,
            "/constraints/numerical/max_abs_parameter",
        )?;
        finite(
            numerical.backend_abs_tolerance,
            "/constraints/numerical/backend_abs_tolerance",
        )?;
        if numerical.max_abs_parameter <= 0.0
            || numerical.backend_abs_tolerance < 0.0
            || !matches!(numerical.accumulation_dtype, DType::F32 | DType::F64)
        {
            return Err(invalid(
                "/constraints/numerical",
                "unsupported numerical bounds or accumulation dtype",
            ));
        }
        self.lineage.method.validate().map_err(|mut error| {
            error.path = format!("/lineage/method{}", error.path);
            error
        })?;
        if self.constraints.gradient_free
            && self.lineage.method.production_history != ProductionHistory::DirectOnly
        {
            return Err(invalid(
                "/lineage/method",
                "gradient-free update contradicts fitted or unknown lineage",
            ));
        }
        let mut preconditions = BTreeMap::new();
        for (index, precondition) in self.preconditions.iter().enumerate() {
            let path = format!("/preconditions/{index}");
            tensor_name(&precondition.tensor, &path)?;
            identifier(&precondition.storage_group, &path)?;
            element_count(&precondition.shape, &path)?
                .checked_mul(precondition.dtype.byte_width())
                .ok_or_else(|| invalid(&path, "precondition byte count overflow"))?;
            if preconditions
                .insert(precondition.tensor.as_str(), precondition)
                .is_some()
            {
                return Err(invalid(&path, "duplicate tensor precondition"));
            }
        }
        for capability in &self.required_capabilities {
            identifier(capability, "/required_capabilities")?;
        }
        let mut artifact_bytes = 0u64;
        for (id, artifact) in &self.artifacts {
            let path = child_path("/artifacts", id);
            identifier(id, &path)?;
            artifact.validate_at(&path)?;
            artifact_bytes = artifact_bytes
                .checked_add(artifact.artifact_bytes)
                .filter(|sum| *sum <= self.constraints.max_artifact_bytes)
                .ok_or_else(|| invalid(&path, "operand artifacts exceed byte budget"))?;
        }
        let mut seen = BTreeSet::new();
        for (index, instruction) in self.operations.iter().enumerate() {
            let path = format!("/operations/{index}");
            identifier(&instruction.id, &child_path(&path, "id"))?;
            let mut dependencies = BTreeSet::new();
            for dependency in &instruction.depends_on {
                if !seen.contains(dependency) || !dependencies.insert(dependency) {
                    return Err(invalid(
                        &path,
                        "dependencies must be distinct earlier instructions",
                    ));
                }
            }
            if !seen.insert(instruction.id.clone()) {
                return Err(invalid(&path, "duplicate instruction ID"));
            }
            instruction
                .operation
                .validate_at(&child_path(&path, "operation"))?;
            let target = instruction.operation.target();
            if !preconditions.contains_key(target.tensor.as_str()) {
                return Err(invalid(&path, "missing source tensor precondition"));
            }
            for id in instruction.operation.artifact_ids() {
                if !self.artifacts.contains_key(id) {
                    return Err(invalid(&path, "missing immutable operand descriptor"));
                }
            }
            if !self.effects.writes.contains(target) && !self.effects.buffer_writes.contains(target)
            {
                return Err(invalid(
                    &path,
                    "operation has no matching declared write effect",
                ));
            }
            if instruction.operation.reads_target() && !self.effects.reads.contains(target) {
                return Err(invalid(
                    &path,
                    "operation has no matching declared read effect",
                ));
            }
        }
        for (kind, regions) in [
            ("reads", &self.effects.reads),
            ("writes", &self.effects.writes),
            ("buffer_writes", &self.effects.buffer_writes),
        ] {
            for (index, region) in regions.iter().enumerate() {
                let path = format!("/effects/{kind}/{index}");
                region.validate_at(&path)?;
                if !preconditions.contains_key(region.tensor.as_str()) {
                    return Err(invalid(&path, "effect requires a declared source tensor"));
                }
            }
        }
        if self.constraints.preserve_tensor_shapes
            && (self.effects.graph_changes
                || !self.effects.config_changes.is_empty()
                || self.effects.tokenizer_changes)
        {
            return Err(invalid(
                "/effects",
                "shape-preserving program cannot change graph/config/tokenizer",
            ));
        }
        match &self.preservation {
            Preservation::ExactUnderConditions { rule, conditions } => {
                identifier(&rule.id, "/preservation/rule/id")?;
                if rule.version == 0 || conditions.is_empty() {
                    return Err(invalid(
                        "/preservation",
                        "exact claims need a versioned rule and explicit conditions",
                    ));
                }
            }
            Preservation::BoundedError {
                metric,
                absolute_tolerance,
                ..
            } => {
                identifier(metric, "/preservation/metric")?;
                finite(*absolute_tolerance, "/preservation/absolute_tolerance")?;
                if *absolute_tolerance < 0.0 {
                    return Err(invalid(
                        "/preservation",
                        "error tolerance cannot be negative",
                    ));
                }
            }
            Preservation::Experimental => (),
        }
        if self.verification.gyms.is_empty() {
            return Err(invalid(
                "/verification/gyms",
                "explicit gym bindings are required",
            ));
        }
        let mut bindings = BTreeSet::new();
        for gym in &self.verification.gyms {
            if !bindings.insert(&gym.binding) {
                return Err(invalid(
                    "/verification/gyms",
                    "duplicate verification binding",
                ));
            }
        }
        if self.extensions.len() > 16 {
            return Err(invalid("/extensions", "extension namespace bound exceeded"));
        }
        for (name, value) in &self.extensions {
            identifier(name, "/extensions")?;
            if !name.contains('.') || value.len() > 1024 {
                return Err(invalid(
                    "/extensions",
                    "extensions must be bounded namespaced inert strings",
                ));
            }
        }
        Ok(())
    }
}

pub fn parse_program(bytes: &[u8]) -> Result<Validated<Program>> {
    let document = CanonicalDocument::parse(bytes, Limits::default())?;
    if let Some(instructions) = document
        .value()
        .get("operations")
        .and_then(serde_json::Value::as_array)
    {
        for (index, instruction) in instructions.iter().enumerate() {
            if let Some(kind) = instruction
                .get("operation")
                .and_then(|op| op.get("type"))
                .and_then(serde_json::Value::as_str)
            {
                if OperationKind::from_wire(kind).is_none() {
                    return Err(CoreError::new(
                        ErrorCode::UnsupportedOperation,
                        format!("/operations/{index}/operation/type"),
                        "unknown or unsupported effectful instruction",
                    ));
                }
            }
        }
    }
    Validated::parse(document.bytes())
}
