//! Pure, same-shape compilation from real checkpoint/operand bytes and fresh CPU
//! observations. A compiled plan has no deserializer and grants no run lease.

use crate::cost::representation_cost;
use crate::program::{
    element_count, identifier, GymBinding, Instruction, MapSide, ModelBinding, NumericalPolicy,
    Operation, OperationKind, Preservation, Program, TensorArtifact, TensorRegion, TensorRole,
    Verification,
};
use crate::search_space::{PriorEffects, ResolvedSearchSpace};
use nf_checkpoint::tensor_file::{inspect_safetensors, TensorFileLimits};
use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use nf_core::error::{CoreError, ErrorCode, Result};
use nf_core::schemas::{
    AccessClass, DType, ModelManifest, Permissions, ProductionHistory, TargetUpdate, ToolKind,
    Validate, Validated,
};
use nf_graph::inventory::{
    verify_inventory, InventoryLimits, TensorObservation, TensorRole as ObservedRole,
    VerifiedInventory,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::Arc;

pub const NATIVE_CPU_BACKEND_ID: &str = "nf.native_cpu";
pub const TORCH_CPU_BACKEND_ID: &str = "nf.trusted_torch_cpu";
pub const TORCH_CUDA_BACKEND_ID: &str = "nf.trusted_torch_cuda";

fn failure(code: ErrorCode, path: &str, message: impl Into<String>) -> CoreError {
    CoreError::new(code, path, message)
}
fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| failure(ErrorCode::LimitExceeded, "", "resource bound overflow"))
}
fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .ok_or_else(|| failure(ErrorCode::LimitExceeded, "", "resource bound overflow"))
}
fn json_digest<T: Serialize>(value: &T) -> Result<ContentDigest> {
    let raw = serde_json::to_vec(value)
        .map_err(|error| failure(ErrorCode::InvalidValue, "", error.to_string()))?;
    Ok(CanonicalDocument::parse(&raw, Limits::default())?.digest())
}

/// Exact IEEE-754 binary16 to binary32 conversion.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exp = ((bits as u32) >> 10) & 0x1F;
    let mant = (bits as u32) & 0x03FF;
    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut m = mant;
            let mut e = 113u32;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03FF;
            sign | (e << 23) | (m << 13)
        }
    } else if exp == 0x1F {
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        sign | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(f32_bits)
}

/// Exact brain floating point 16 (bf16) to binary32 conversion.
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendProfile {
    pub id: String,
    pub version: String,
    pub operations: BTreeSet<OperationKind>,
    pub capabilities: BTreeSet<String>,
    pub dtypes: Vec<DType>,
    pub accumulation_dtypes: Vec<DType>,
    pub fixed_scratch_bytes: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompilerBounds {
    pub max_metadata_bytes: u64,
    pub max_source_checkpoint_bytes: u64,
    pub max_source_asset_bytes: u64,
    pub max_source_tensors: usize,
    pub max_observed_source_bytes: u64,
    pub max_temporary_bytes: u64,
    pub max_output_tensor_bytes: u64,
    pub max_validation_work: u64,
    pub max_kernel_work: u64,
    pub max_parameter_magnitude: f64,
    pub max_backend_abs_tolerance: f64,
    pub max_orthonormal_tolerance: f64,
}
impl CompilerBounds {
    /// Bounds profile suitable for multi-gigabyte open-weight checkpoints (e.g., 7B-70B parameters).
    pub fn large_model() -> Self {
        Self {
            max_metadata_bytes: 1_048_576,
            max_source_checkpoint_bytes: 68_719_476_736, // 64 GiB
            max_source_asset_bytes: 268_435_456,         // 256 MiB
            max_source_tensors: 16_384,
            max_observed_source_bytes: 68_719_476_736,
            max_temporary_bytes: 17_179_869_184, // 16 GiB
            max_output_tensor_bytes: 17_179_869_184,
            max_validation_work: 100_000_000_000,
            max_kernel_work: 100_000_000_000,
            max_parameter_magnitude: 65_504.0,
            max_backend_abs_tolerance: 1e-3,
            max_orthonormal_tolerance: 1e-4,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompilePolicy {
    pub schema_version: u32,
    pub source: ModelBinding,
    pub contract: ContentDigest,
    pub objective: ContentDigest,
    pub experiment_protocol: ContentDigest,
    pub search_space: ContentDigest,
    pub verification: Verification,
    pub allowed_extra_gyms: Vec<GymBinding>,
    pub permissions: Permissions,
    pub operand_access: BTreeSet<AccessClass>,
    pub backend: BackendProfile,
    pub bounds: CompilerBounds,
}
impl Validate for CompilePolicy {
    fn validate(&self) -> Result<()> {
        if self.schema_version != 1 {
            return Err(failure(
                ErrorCode::UnsupportedVersion,
                "/schema_version",
                "only compile policy v1 is supported",
            ));
        }
        self.permissions.validate()?;
        identifier(&self.backend.id, "/backend/id")?;
        identifier(&self.backend.version, "/backend/version")?;
        let is_cuda = self.backend.id == TORCH_CUDA_BACKEND_ID;
        if !matches!(
            self.backend.id.as_str(),
            NATIVE_CPU_BACKEND_ID | TORCH_CPU_BACKEND_ID | TORCH_CUDA_BACKEND_ID
        ) || self.backend.version != "1"
            || self.backend.capabilities.iter().any(|capability| {
                if is_cuda {
                    !matches!(
                        capability.as_str(),
                        "dense.cuda" | "dense.cpu" | "same_shape.v1" | "dlpack.v1"
                    )
                } else {
                    !matches!(capability.as_str(), "dense.cpu" | "same_shape.v1")
                }
            })
        {
            return Err(failure(
                ErrorCode::UnsupportedCapability,
                "/backend",
                "unsupported native CPU/CUDA lowering profile",
            ));
        }
        for capability in &self.backend.capabilities {
            identifier(capability, "/backend/capabilities")?;
        }
        if self.backend.dtypes.is_empty()
            || self.backend.accumulation_dtypes.is_empty()
            || self
                .backend
                .dtypes
                .iter()
                .any(|dtype| !matches!(dtype, DType::F16 | DType::Bf16 | DType::F32 | DType::F64))
            || self
                .backend
                .accumulation_dtypes
                .iter()
                .any(|dtype| !matches!(dtype, DType::F32 | DType::F64))
        {
            return Err(failure(
                ErrorCode::UnsupportedCapability,
                "/backend",
                "compiler supports explicit f16/bf16/f32/f64 CPU or CUDA profiles with f32/f64 accumulation",
            ));
        }
        let b = &self.bounds;
        if b.max_metadata_bytes == 0
            || b.max_metadata_bytes > 1_048_576
            || b.max_source_checkpoint_bytes == 0
            || b.max_source_asset_bytes == 0
            || b.max_source_tensors == 0
            || b.max_source_tensors > 65_536
            || b.max_observed_source_bytes == 0
            || b.max_temporary_bytes == 0
            || b.max_output_tensor_bytes == 0
            || b.max_validation_work == 0
            || b.max_kernel_work == 0
        {
            return Err(failure(
                ErrorCode::InvalidValue,
                "/bounds",
                "finite positive supported compiler bounds required",
            ));
        }
        if !b.max_parameter_magnitude.is_finite()
            || b.max_parameter_magnitude <= 0.0
            || !b.max_backend_abs_tolerance.is_finite()
            || b.max_backend_abs_tolerance < 0.0
            || !b.max_orthonormal_tolerance.is_finite()
            || !(0.0..0.01).contains(&b.max_orthonormal_tolerance)
        {
            return Err(failure(
                ErrorCode::InvalidValue,
                "/bounds",
                "invalid numerical compiler policy",
            ));
        }
        if self.verification.gyms.is_empty()
            || self.verification.gyms.len() > 1024
            || self.allowed_extra_gyms.len() > 1024
            || self.operand_access.contains(&AccessClass::Holdout)
        {
            return Err(failure(
                ErrorCode::InvalidValue,
                "/verification",
                "explicit bounded verification/access policy required",
            ));
        }
        let mut ids = BTreeSet::new();
        for gym in self
            .verification
            .gyms
            .iter()
            .chain(&self.allowed_extra_gyms)
        {
            if !ids.insert(&gym.binding) {
                return Err(failure(
                    ErrorCode::DuplicateKey,
                    "/verification",
                    "duplicate frozen gym binding",
                ));
            }
        }
        Ok(())
    }
}

/// The loader supplies checkpoint bytes and inert config/tokenizer assets;
/// CPU observations are rehashed here rather than trusting an old inventory.
pub struct SourceSnapshot<'a> {
    pub manifest: &'a Validated<ModelManifest>,
    pub tensors: &'a [TensorObservation<'a>],
    pub checkpoint: &'a [u8],
    pub assets: &'a BTreeMap<String, Arc<[u8]>>,
}

#[derive(Debug, Clone)]
pub struct VerifiedOperand {
    descriptor: TensorArtifact,
    bytes: Arc<[u8]>,
    range: Range<usize>,
}
impl VerifiedOperand {
    pub fn descriptor(&self) -> &TensorArtifact {
        &self.descriptor
    }
    pub fn data(&self) -> &[u8] {
        &self.bytes[self.range.clone()]
    }
    pub fn value_at(&self, index: usize) -> Result<f64> {
        let width = self.descriptor.dtype.byte_width() as usize;
        let start = index
            .checked_mul(width)
            .ok_or_else(|| failure(ErrorCode::ShapeUnsatisfiable, "", "operand index overflow"))?;
        let raw = self
            .data()
            .get(
                start..start.checked_add(width).ok_or_else(|| {
                    failure(ErrorCode::ShapeUnsatisfiable, "", "operand index overflow")
                })?,
            )
            .ok_or_else(|| {
                failure(
                    ErrorCode::ShapeUnsatisfiable,
                    "",
                    "operand index outside data",
                )
            })?;
        match self.descriptor.dtype {
            DType::F64 => Ok(f64::from_le_bytes(raw.try_into().unwrap())),
            DType::F32 => Ok(f32::from_le_bytes(raw.try_into().unwrap()) as f64),
            DType::F16 => Ok(f16_to_f32(u16::from_le_bytes(raw.try_into().unwrap())) as f64),
            DType::Bf16 => Ok(bf16_to_f32(u16::from_le_bytes(raw.try_into().unwrap())) as f64),
            DType::U8 => Ok(raw[0] as f64),
            _ => Err(failure(
                ErrorCode::UnsupportedCapability,
                "",
                "unsupported native operand dtype",
            )),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct AllocationPlan {
    pub retained_operand_bytes: u64,
    pub candidate_copy_bytes: u64,
    pub tracking_bytes: u64,
    pub peak_kernel_scratch_bytes: u64,
    pub estimated_working_bytes: u64,
    pub output_tensor_bytes: u64,
    pub validated_input_bytes: u64,
    pub validation_scalar_work: u64,
    pub kernel_work_units: u64,
}
#[derive(Debug, Clone, Serialize)]
pub struct PlanSummary {
    pub schema_version: u32,
    pub compiler: String,
    pub compiler_version: String,
    pub program: ContentDigest,
    pub policy: ContentDigest,
    pub backend: ContentDigest,
    pub source_manifest: ContentDigest,
    pub source_parameters: ContentDigest,
    pub source_buffers: ContentDigest,
    pub search_space: ContentDigest,
    pub effective_reads: Vec<TensorRegion>,
    pub effective_writes: Vec<TensorRegion>,
    pub affected_parameter_upper_bound: u64,
    pub pending_quality_constraints: usize,
    pub identity_preservation_verified: bool,
    pub allocations: AllocationPlan,
}
#[derive(Debug, Clone)]
pub struct CompiledPlan {
    backend: BackendProfile,
    summary: PlanSummary,
    digest: ContentDigest,
    program: Program,
    inventory: VerifiedInventory,
    operands: BTreeMap<String, VerifiedOperand>,
}
impl CompiledPlan {
    pub fn backend(&self) -> &BackendProfile {
        &self.backend
    }
    pub fn summary(&self) -> &PlanSummary {
        &self.summary
    }
    pub fn digest(&self) -> &ContentDigest {
        &self.digest
    }
    pub fn program(&self) -> &Program {
        &self.program
    }
    pub fn inventory(&self) -> &VerifiedInventory {
        &self.inventory
    }
    pub fn operands(&self) -> &BTreeMap<String, VerifiedOperand> {
        &self.operands
    }
    pub fn instructions(&self) -> &[Instruction] {
        &self.program.operations
    }
    pub fn numerical(&self) -> &NumericalPolicy {
        &self.program.constraints.numerical
    }
}

fn verification(program: &Verification, policy: &CompilePolicy) -> Result<()> {
    let required = &policy.verification;
    if program.portfolio != required.portfolio
        || program.protocol != required.protocol
        || program.smoke_suite != required.smoke_suite
        || program.parity_suite != required.parity_suite
    {
        return Err(failure(
            ErrorCode::VerificationPolicyWeakened,
            "/verification",
            "frozen verification identities differ",
        ));
    }
    for binding in &required.gyms {
        if !program.gyms.contains(binding) {
            return Err(failure(
                ErrorCode::VerificationPolicyWeakened,
                "/verification/gyms",
                "required exact gym/scorer/split/validation binding missing",
            ));
        }
    }
    for binding in &program.gyms {
        if !required.gyms.contains(binding) && !policy.allowed_extra_gyms.contains(binding) {
            return Err(failure(
                ErrorCode::VerificationPolicyWeakened,
                "/verification/gyms",
                "unapproved supplemental or modified binding",
            ));
        }
    }
    Ok(())
}

fn source_inventory(
    source: &SourceSnapshot<'_>,
    policy: &CompilePolicy,
) -> Result<VerifiedInventory> {
    if source.checkpoint.len() as u64 > policy.bounds.max_source_checkpoint_bytes {
        return Err(failure(
            ErrorCode::LimitExceeded,
            "/base_model",
            "source checkpoint exceeds admission bound",
        ));
    }
    let files = inspect_safetensors(
        source.checkpoint,
        TensorFileLimits {
            max_file_bytes: usize::try_from(policy.bounds.max_source_checkpoint_bytes)
                .unwrap_or(usize::MAX),
            ..TensorFileLimits::default()
        },
    )?;
    if files.file_digest != policy.source.checkpoint {
        return Err(failure(
            ErrorCode::SourceMismatch,
            "/base_model/checkpoint",
            "checkpoint bytes differ",
        ));
    }
    let inventory = verify_inventory(
        source.manifest,
        source.tensors,
        InventoryLimits {
            max_tensors: policy.bounds.max_source_tensors,
            max_observed_bytes: policy.bounds.max_observed_source_bytes,
        },
    )?;
    if inventory.source_manifest_digest() != &policy.source.manifest
        || inventory.parameter_digest() != &policy.source.parameter_digest
        || source.manifest.architecture.config != policy.source.config
    {
        return Err(failure(
            ErrorCode::SourceMismatch,
            "/base_model",
            "fresh source manifest/parameters/config differ",
        ));
    }
    for (name, expected) in [
        ("checkpoint", &policy.source.checkpoint),
        ("config", &policy.source.config),
    ] {
        if source
            .manifest
            .assets
            .get(name)
            .is_some_and(|declared| declared != expected)
        {
            return Err(failure(
                ErrorCode::SourceMismatch,
                "/base_model",
                "manifest asset binding differs",
            ));
        }
    }
    if source.manifest.assets.get("tokenizer") != policy.source.tokenizer.as_ref() {
        return Err(failure(
            ErrorCode::SourceMismatch,
            "/base_model/tokenizer",
            "manifest tokenizer binding differs",
        ));
    }
    if source.assets.len() > 1024 {
        return Err(failure(
            ErrorCode::LimitExceeded,
            "/base_model",
            "source asset count bound exceeded",
        ));
    }
    let mut total_assets = 0u64;
    for bytes in source.assets.values() {
        total_assets = add(total_assets, bytes.len() as u64)?;
    }
    if total_assets > policy.bounds.max_source_asset_bytes {
        return Err(failure(
            ErrorCode::LimitExceeded,
            "/base_model",
            "source asset bound exceeded",
        ));
    }
    let config = source.assets.get("config").ok_or_else(|| {
        failure(
            ErrorCode::SourceMismatch,
            "/base_model/config",
            "config bytes are required",
        )
    })?;
    if ContentDigest::of_bytes(config) != policy.source.config {
        return Err(failure(
            ErrorCode::SourceMismatch,
            "/base_model/config",
            "config bytes differ",
        ));
    }
    CanonicalDocument::parse(config, Limits::default())?;
    for (name, expected) in &source.manifest.assets {
        if name == "checkpoint" {
            continue;
        }
        let bytes = source.assets.get(name).ok_or_else(|| {
            failure(
                ErrorCode::SourceMismatch,
                "/base_model",
                "declared source asset is missing",
            )
        })?;
        if &ContentDigest::of_bytes(bytes) != expected {
            return Err(failure(
                ErrorCode::SourceMismatch,
                "/base_model",
                "declared source asset bytes differ",
            ));
        }
    }
    if source
        .assets
        .keys()
        .any(|name| name != "config" && !source.manifest.assets.contains_key(name))
    {
        return Err(failure(
            ErrorCode::SourceMismatch,
            "/base_model",
            "unbound source asset supplied",
        ));
    }
    match (&policy.source.tokenizer, source.assets.get("tokenizer")) {
        (Some(expected), Some(bytes)) if &ContentDigest::of_bytes(bytes) == expected => (),
        (None, None) => (),
        _ => {
            return Err(failure(
                ErrorCode::SourceMismatch,
                "/base_model/tokenizer",
                "tokenizer asset binding differs",
            ))
        }
    }
    let mut covered = BTreeSet::new();
    for (name, tensor) in &files.tensors {
        let actual = inventory.entries().get(name).ok_or_else(|| {
            failure(
                ErrorCode::SourceMismatch,
                "/base_model",
                "unexpected source file tensor",
            )
        })?;
        if tensor.shape != actual.shape
            || tensor.dtype != actual.dtype
            || tensor.digest != actual.content_digest
        {
            return Err(failure(
                ErrorCode::SourceMismatch,
                "/base_model",
                "source file and observed tensor disagree",
            ));
        }
        covered.insert(actual.storage_group.clone());
    }
    if inventory
        .alias_groups()
        .keys()
        .any(|group| !covered.contains(group))
    {
        return Err(failure(
            ErrorCode::SourceMismatch,
            "/base_model",
            "checkpoint is missing an observed storage group",
        ));
    }
    Ok(inventory)
}

fn within(outer: &TensorRegion, inner: &TensorRegion) -> bool {
    outer.tensor == inner.tensor
        && outer.shape.len() == inner.shape.len()
        && outer
            .start
            .iter()
            .zip(&outer.shape)
            .zip(inner.start.iter().zip(&inner.shape))
            .all(|((a, na), (b, nb))| {
                *a <= *b
                    && a.checked_add(*na)
                        .zip(b.checked_add(*nb))
                        .is_some_and(|(x, y)| y <= x)
            })
}
fn region_in_inventory(region: &TensorRegion, inventory: &VerifiedInventory) -> Result<()> {
    region.validate_at("/effects")?;
    let entry = inventory.entries().get(&region.tensor).ok_or_else(|| {
        failure(
            ErrorCode::SourceMismatch,
            "/effects",
            "unknown source tensor",
        )
    })?;
    let whole = TensorRegion {
        tensor: region.tensor.clone(),
        start: vec![0; entry.shape.len()],
        shape: entry.shape.clone(),
    };
    if !within(&whole, region) {
        return Err(failure(
            ErrorCode::ShapeUnsatisfiable,
            "/effects",
            "region exceeds actual source dimensions",
        ));
    }
    Ok(())
}
fn covered(regions: &[TensorRegion], needed: &TensorRegion) -> bool {
    regions.iter().any(|outer| within(outer, needed))
}
fn preconditions(program: &Program, inventory: &VerifiedInventory) -> Result<()> {
    for condition in &program.preconditions {
        let entry = inventory.entries().get(&condition.tensor).ok_or_else(|| {
            failure(
                ErrorCode::SourceMismatch,
                "/preconditions",
                "unknown source tensor",
            )
        })?;
        let role = match entry.role {
            ObservedRole::Parameter => TensorRole::Parameter,
            ObservedRole::Buffer => TensorRole::Buffer,
        };
        if condition.shape != entry.shape
            || condition.dtype != entry.dtype
            || condition.before_digest != entry.content_digest
            || condition.storage_group != entry.storage_group
            || condition.role != role
        {
            return Err(failure(
                ErrorCode::SourceMismatch,
                "/preconditions",
                "stale shape/dtype/role/content/alias signature",
            ));
        }
    }
    for region in program
        .effects
        .reads
        .iter()
        .chain(&program.effects.writes)
        .chain(&program.effects.buffer_writes)
    {
        region_in_inventory(region, inventory)?;
    }
    Ok(())
}

struct Work {
    used: u64,
    bound: u64,
}
impl Work {
    fn charge(&mut self, count: u64) -> Result<()> {
        self.used = add(self.used, count)?;
        if self.used > self.bound {
            return Err(failure(
                ErrorCode::LimitExceeded,
                "",
                "validation scalar work bound exceeded",
            ));
        }
        Ok(())
    }
}
fn operands(
    program: &Program,
    raw: &BTreeMap<String, Arc<[u8]>>,
    policy: &CompilePolicy,
    work: &mut Work,
) -> Result<(BTreeMap<String, VerifiedOperand>, u64)> {
    if raw.len() != program.artifacts.len()
        || raw.keys().any(|id| !program.artifacts.contains_key(id))
    {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "/artifacts",
            "exact operand blob set required",
        ));
    }
    let mut result = BTreeMap::new();
    let mut retained = BTreeMap::<ContentDigest, Arc<[u8]>>::new();
    let mut total = 0u64;
    for (id, descriptor) in &program.artifacts {
        if !policy.operand_access.contains(&descriptor.access) {
            return Err(failure(
                ErrorCode::PermissionDenied,
                "/artifacts",
                "operand access class not granted",
            ));
        }
        let bytes = raw.get(id).ok_or_else(|| {
            failure(
                ErrorCode::IntegrityMismatch,
                "/artifacts",
                "operand blob missing",
            )
        })?;
        if bytes.len() as u64 != descriptor.artifact_bytes {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "/artifacts",
                "operand file byte length differs",
            ));
        }
        let inventory = inspect_safetensors(
            bytes,
            TensorFileLimits {
                max_file_bytes: usize::try_from(program.constraints.max_artifact_bytes)
                    .unwrap_or(usize::MAX),
                ..TensorFileLimits::default()
            },
        )?;
        let tensor = inventory.tensors.get(&descriptor.tensor).ok_or_else(|| {
            failure(
                ErrorCode::IntegrityMismatch,
                "/artifacts",
                "named operand tensor missing",
            )
        })?;
        if inventory.file_digest != descriptor.digest
            || tensor.shape != descriptor.shape
            || tensor.dtype != descriptor.dtype
            || tensor.data_range.len() as u64 != descriptor.data_bytes
        {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "/artifacts",
                "operand hash/shape/dtype declaration differs",
            ));
        }
        if !matches!(
            descriptor.dtype,
            DType::F16 | DType::Bf16 | DType::F32 | DType::F64 | DType::U8
        ) {
            return Err(failure(
                ErrorCode::UnsupportedCapability,
                "/artifacts",
                "native operand dtype unsupported",
            ));
        }
        let owned = if let Some(existing) = retained.get(&descriptor.digest) {
            existing.clone()
        } else {
            total = add(total, bytes.len() as u64)?;
            retained.insert(descriptor.digest.clone(), bytes.clone());
            bytes.clone()
        };
        let operand = VerifiedOperand {
            descriptor: descriptor.clone(),
            bytes: owned,
            range: tensor.data_range.clone(),
        };
        let count = element_count(&descriptor.shape, "/artifacts")?;
        work.charge(count)?;
        for index in 0..count {
            if !operand.value_at(index as usize)?.is_finite() {
                return Err(failure(
                    ErrorCode::NonFinite,
                    "/artifacts",
                    "non-finite operand data",
                ));
            }
        }
        result.insert(id.clone(), operand);
    }
    Ok((result, total))
}

fn require_shape(operand: &VerifiedOperand, shape: &[u64], dtype: DType) -> Result<()> {
    if operand.descriptor.shape != shape || operand.descriptor.dtype != dtype {
        return Err(failure(
            ErrorCode::ShapeUnsatisfiable,
            "/artifacts",
            "operand dimensions or dtype differ; implicit casting is forbidden",
        ));
    }
    Ok(())
}
fn matrix(shape: &[u64]) -> Result<(u64, u64)> {
    if shape.len() != 2 {
        return Err(failure(
            ErrorCode::ShapeUnsatisfiable,
            "/operations",
            "matrix operation requires rank two",
        ));
    }
    Ok((shape[0], shape[1]))
}
fn checked_coefficient(value: f64, numerical: &NumericalPolicy) -> Result<()> {
    if !value.is_finite()
        || (numerical.accumulation_dtype == DType::F32 && !(value as f32).is_finite())
    {
        return Err(failure(
            ErrorCode::NonFinite,
            "/operations",
            "coefficient is not finite in accumulation dtype",
        ));
    }
    Ok(())
}
fn orthonormal(
    operand: &VerifiedOperand,
    tolerance: f64,
    numerical: &NumericalPolicy,
    work: &mut Work,
) -> Result<()> {
    let (rows, rank) = matrix(&operand.descriptor.shape)?;
    if rank > rows {
        return Err(failure(
            ErrorCode::ShapeUnsatisfiable,
            "/artifacts",
            "orthonormal basis rank exceeds its dimension",
        ));
    }
    work.charge(mul(mul(rows, rank)?, mul(rank, 2)?)?)?;
    for a in 0..rank {
        for b in 0..rank {
            let mut sum = 0.0f64;
            for row in 0..rows {
                let x = operand.value_at((row * rank + a) as usize)?;
                let y = operand.value_at((row * rank + b) as usize)?;
                sum = if numerical.accumulation_dtype == DType::F32 {
                    ((sum as f32) + (x as f32) * (y as f32)) as f64
                } else {
                    sum + x * y
                };
            }
            let expected = if a == b { 1.0 } else { 0.0 };
            if !sum.is_finite() || (sum - expected).abs() > tolerance {
                return Err(failure(
                    ErrorCode::InvalidValue,
                    "/artifacts",
                    "basis does not meet declared orthonormality tolerance",
                ));
            }
        }
    }
    Ok(())
}

fn operation_cost(
    operation: &Operation,
    dtype: DType,
    operands: &BTreeMap<String, VerifiedOperand>,
    numerical: &NumericalPolicy,
    policy: &CompilePolicy,
    work: &mut Work,
) -> Result<(u64, u64)> {
    let shape = &operation.target().shape;
    let count = element_count(shape, "/operations")?;
    let operand = |id: &str| {
        operands.get(id).ok_or_else(|| {
            failure(
                ErrorCode::IntegrityMismatch,
                "/artifacts",
                "operand missing after validation",
            )
        })
    };
    match operation {
        Operation::ReplaceSlice { value, .. } | Operation::AddDelta { delta: value, .. } => {
            require_shape(operand(value)?, shape, dtype)?;
            Ok((0, mul(count, 2)?))
        }
        Operation::Scale { factor, .. } => {
            checked_coefficient(*factor, numerical)?;
            Ok((0, mul(count, 2)?))
        }
        Operation::Mask { mask, .. } => {
            let value = operand(mask)?;
            require_shape(value, shape, DType::U8)?;
            work.charge(count)?;
            if value.data().iter().any(|byte| *byte > 1) {
                return Err(failure(
                    ErrorCode::InvalidValue,
                    "/artifacts",
                    "mask must contain only zero/one bytes",
                ));
            }
            Ok((0, mul(count, 3)?))
        }
        Operation::PermuteAxis { axis, .. } => {
            let cost = representation_cost(
                OperationKind::PermuteAxis,
                shape,
                dtype,
                numerical.accumulation_dtype,
                Some(*axis),
                None,
                None,
            )?;
            Ok((cost.scratch_bytes, cost.work_upper_bound))
        }
        Operation::LinearMap {
            matrix: map, side, ..
        } => {
            let (rows, cols) = matrix(shape)?;
            let axis = if *side == MapSide::Left { rows } else { cols };
            require_shape(operand(map)?, &[axis, axis], dtype)?;
            let cost = representation_cost(
                OperationKind::LinearMap,
                shape,
                dtype,
                numerical.accumulation_dtype,
                None,
                Some(*side),
                None,
            )?;
            Ok((cost.scratch_bytes, cost.work_upper_bound))
        }
        Operation::LowRankUpdate {
            left,
            right,
            strength,
            ..
        } => {
            checked_coefficient(*strength, numerical)?;
            let (rows, cols) = matrix(shape)?;
            let (a, rank) = matrix(&operand(left)?.descriptor.shape)?;
            if a != rows {
                return Err(failure(
                    ErrorCode::ShapeUnsatisfiable,
                    "/artifacts",
                    "left factor row dimension differs",
                ));
            }
            require_shape(operand(left)?, &[rows, rank], dtype)?;
            require_shape(operand(right)?, &[rank, cols], dtype)?;
            let cost = representation_cost(
                OperationKind::LowRankUpdate,
                shape,
                dtype,
                numerical.accumulation_dtype,
                None,
                None,
                Some(rank),
            )?;
            Ok((cost.scratch_bytes, cost.work_upper_bound))
        }
        Operation::SubspaceProjection {
            basis,
            side,
            strength,
            orthonormal_tolerance,
            ..
        } => {
            checked_coefficient(*strength, numerical)?;
            if *orthonormal_tolerance > policy.bounds.max_orthonormal_tolerance {
                return Err(failure(
                    ErrorCode::PermissionDenied,
                    "/operations",
                    "projection tolerance weakens frozen policy",
                ));
            }
            let (rows, cols) = matrix(shape)?;
            let (dimension, rank) = matrix(&operand(basis)?.descriptor.shape)?;
            let axis = if *side == MapSide::Left { rows } else { cols };
            if dimension != axis {
                return Err(failure(
                    ErrorCode::ShapeUnsatisfiable,
                    "/artifacts",
                    "basis dimension differs from target axis",
                ));
            }
            require_shape(operand(basis)?, &[axis, rank], dtype)?;
            orthonormal(operand(basis)?, *orthonormal_tolerance, numerical, work)?;
            let cost = representation_cost(
                OperationKind::SubspaceProjection,
                shape,
                dtype,
                numerical.accumulation_dtype,
                None,
                Some(*side),
                Some(rank),
            )?;
            Ok((cost.scratch_bytes, cost.work_upper_bound))
        }
        Operation::ConstructRecipe { .. } => Err(failure(
            ErrorCode::UnsupportedOperation,
            "/operations",
            "constructor requires a qualified registered lowering",
        )),
    }
}

pub fn identity_rule_specification() -> ContentDigest {
    ContentDigest::of_bytes(b"nf.identity_noop.v1: no instructions, no model-state changes")
}
fn preservation(program: &Program, inventory: &VerifiedInventory) -> Result<bool> {
    match &program.preservation {
        Preservation::ExactUnderConditions { rule, conditions } => {
            if rule.id != "identity_noop"
                || rule.version != 1
                || rule.specification != identity_rule_specification()
                || conditions.as_slice() != [inventory.source_manifest_digest().clone()]
                || !program.operations.is_empty()
                || !program.effects.writes.is_empty()
                || !program.effects.buffer_writes.is_empty()
            {
                return Err(failure(
                    ErrorCode::UnsupportedCapability,
                    "/preservation",
                    "exact claim lacks a supported checked preservation rule",
                ));
            }
            Ok(true)
        }
        Preservation::BoundedError { protocol, .. }
            if protocol != &program.verification.protocol =>
        {
            Err(failure(
                ErrorCode::VerificationPolicyWeakened,
                "/preservation",
                "bounded-error protocol differs from frozen verifier",
            ))
        }
        _ => Ok(false),
    }
}

pub fn compile(
    program: &Validated<Program>,
    source: SourceSnapshot<'_>,
    space: &ResolvedSearchSpace,
    raw_operands: &BTreeMap<String, Arc<[u8]>>,
    policy: &Validated<CompilePolicy>,
    parents: &[PriorEffects],
) -> Result<CompiledPlan> {
    policy.validate()?;
    program.validate()?;
    let metadata = serde_json::to_vec(&**program)
        .map_err(|error| failure(ErrorCode::InvalidValue, "", error.to_string()))?;
    if metadata.len() as u64 > policy.bounds.max_metadata_bytes {
        return Err(failure(
            ErrorCode::LimitExceeded,
            "",
            "IR metadata admission bound exceeded",
        ));
    }
    if program.base_model != policy.source
        || program.objective.contract != policy.contract
        || program.objective.objective != policy.objective
        || program.objective.experiment_protocol != policy.experiment_protocol
        || program.objective.search_space != policy.search_space
        || space.digest() != &policy.search_space
    {
        return Err(failure(
            ErrorCode::SourceMismatch,
            "/objective",
            "frozen source/contract/protocol/search-space policy differs",
        ));
    }
    verification(&program.verification, policy)?;
    if program.operations.is_empty() {
        policy.permissions.check_tool(ToolKind::Inspect)?;
    } else {
        policy.permissions.check_tool(ToolKind::DirectEdit)?;
    }
    if program.lineage.method.inspection_uses_backprop {
        policy.permissions.check_tool(ToolKind::Attribution)?;
    }
    if program.lineage.method.parameter_update_uses_backprop
        || program.lineage.method.gradient_derived_update_values
        || program.lineage.method.alignment_fitted
        || program.lineage.method.optimizer_steps > 0
        || program.lineage.method.target_update == TargetUpdate::GradientTraining
        || matches!(
            program.lineage.method.production_history,
            ProductionHistory::Trained | ProductionHistory::Hybrid
        )
    {
        policy.permissions.check_tool(ToolKind::Training)?;
    }
    if policy.permissions.require_fully_gradient_free {
        program.lineage.method.require_fully_gradient_free()?;
    }
    if program.effects.graph_changes
        || !program.effects.config_changes.is_empty()
        || program.effects.tokenizer_changes
    {
        return Err(failure(
            ErrorCode::UnsupportedCapability,
            "/effects",
            "same-shape compiler cannot rewrite graph/config/tokenizer",
        ));
    }
    if !program
        .required_capabilities
        .is_subset(&policy.backend.capabilities)
    {
        return Err(failure(
            ErrorCode::UnsupportedCapability,
            "/required_capabilities",
            "backend capability is missing",
        ));
    }
    let numeric = &program.constraints.numerical;
    if numeric.max_abs_parameter > policy.bounds.max_parameter_magnitude
        || numeric.backend_abs_tolerance > policy.bounds.max_backend_abs_tolerance
        || !policy
            .backend
            .accumulation_dtypes
            .contains(&numeric.accumulation_dtype)
    {
        return Err(failure(
            ErrorCode::PermissionDenied,
            "/constraints/numerical",
            "numerical proposal weakens compiler policy",
        ));
    }
    let inventory = source_inventory(&source, policy)?;
    space.check_inventory_layout(&inventory)?;
    preconditions(program, &inventory)?;
    let domain = space.check_program(program, parents)?;
    let mut work = Work {
        used: 0,
        bound: policy.bounds.max_validation_work,
    };
    let (operands, retained) = operands(program, raw_operands, policy, &mut work)?;
    let mut reads = Vec::new();
    let mut writes = Vec::new();
    let mut groups = BTreeSet::new();
    let mut scratch = policy.backend.fixed_scratch_bytes;
    let mut kernel_work = 0u64;
    for instruction in &program.operations {
        let operation = &instruction.operation;
        if !policy.backend.operations.contains(&operation.kind()) {
            return Err(failure(
                ErrorCode::UnsupportedCapability,
                "/operations",
                "operation is not in backend profile",
            ));
        }
        let target = operation.target();
        region_in_inventory(target, &inventory)?;
        let entry = &inventory.entries()[&target.tensor];
        if !policy.backend.dtypes.contains(&entry.dtype)
            || (entry.dtype == DType::F64 && numeric.accumulation_dtype != DType::F64)
        {
            return Err(failure(
                ErrorCode::UnsupportedCapability,
                "/operations",
                "target/accumulation dtype combination unsupported",
            ));
        }
        for alias in &inventory.alias_groups()[&entry.storage_group] {
            let mut effect = target.clone();
            effect.tensor = alias.clone();
            let actual = &inventory.entries()[alias];
            if actual.role == ObservedRole::Buffer {
                return Err(failure(
                    ErrorCode::UnsupportedCapability,
                    "/effects/buffer_writes",
                    "buffer mutation requires a qualified state adapter",
                ));
            }
            if !program
                .preconditions
                .iter()
                .any(|condition| condition.tensor == *alias)
                || !covered(&program.effects.writes, &effect)
            {
                return Err(failure(
                    ErrorCode::AliasScopeEscape,
                    "/effects/writes",
                    "effective alias write lacks explicit precondition/effect",
                ));
            }
            if operation.reads_target() {
                if !covered(&program.effects.reads, &effect) {
                    return Err(failure(
                        ErrorCode::AliasScopeEscape,
                        "/effects/reads",
                        "effective alias read was not declared",
                    ));
                }
                if !reads.contains(&effect) {
                    reads.push(effect.clone());
                }
            }
            if !writes.contains(&effect) {
                writes.push(effect);
            }
        }
        groups.insert(entry.storage_group.clone());
        let (temporary, units) = operation_cost(
            operation,
            entry.dtype,
            &operands,
            numeric,
            policy,
            &mut work,
        )?;
        scratch = scratch.max(add(policy.backend.fixed_scratch_bytes, temporary)?);
        kernel_work = add(kernel_work, units)?;
    }
    if kernel_work > policy.bounds.max_kernel_work {
        return Err(failure(
            ErrorCode::LimitExceeded,
            "/operations",
            "kernel work admission bound exceeded",
        ));
    }
    let mut copies = 0u64;
    let mut tracking = 0u64;
    let mut output = 0u64;
    for (group, members) in inventory.alias_groups() {
        let entry = &inventory.entries()[&members[0]];
        let count = element_count(&entry.shape, "/base_model")?;
        let bytes = mul(count, entry.dtype.byte_width())?;
        output = add(output, bytes)?;
        if groups.contains(group) {
            copies = add(copies, bytes)?;
            tracking = add(tracking, count.div_ceil(8))?;
        }
    }
    let working = add(add(retained, copies)?, add(tracking, scratch)?)?;
    if working > program.constraints.max_temporary_bytes
        || working > policy.bounds.max_temporary_bytes
        || output > policy.bounds.max_output_tensor_bytes
    {
        return Err(failure(
            ErrorCode::LimitExceeded,
            "/constraints",
            "resolved allocation/output bound exceeded",
        ));
    }
    let mut validated_bytes = add(source.checkpoint.len() as u64, inventory.observed_bytes())?;
    for bytes in source.assets.values() {
        validated_bytes = add(validated_bytes, bytes.len() as u64)?;
    }
    validated_bytes = add(validated_bytes, retained)?;
    let allocations = AllocationPlan {
        retained_operand_bytes: retained,
        candidate_copy_bytes: copies,
        tracking_bytes: tracking,
        peak_kernel_scratch_bytes: scratch,
        estimated_working_bytes: working,
        output_tensor_bytes: output,
        validated_input_bytes: validated_bytes,
        validation_scalar_work: work.used,
        kernel_work_units: kernel_work,
    };
    let summary = PlanSummary {
        schema_version: 1,
        compiler: "nf.same_shape.v1".to_owned(),
        compiler_version: nf_core::VERSION.to_owned(),
        program: program.digest()?,
        policy: json_digest(&**policy)?,
        backend: json_digest(&policy.backend)?,
        source_manifest: inventory.source_manifest_digest().clone(),
        source_parameters: inventory.parameter_digest().clone(),
        source_buffers: inventory.buffer_digest().clone(),
        search_space: space.digest().clone(),
        effective_reads: reads,
        effective_writes: writes,
        affected_parameter_upper_bound: domain.affected_parameter_upper_bound,
        pending_quality_constraints: domain.pending_quality_constraints,
        identity_preservation_verified: preservation(program, &inventory)?,
        allocations,
    };
    let digest = json_digest(&summary)?;
    Ok(CompiledPlan {
        backend: policy.backend.clone(),
        summary,
        digest,
        program: (**program).clone(),
        inventory,
        operands,
    })
}
