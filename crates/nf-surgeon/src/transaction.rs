//! Compiled-plan execution on private, copy-on-write candidate groups.
//! Success is tensor/smoke verification, never experiment acceptance or export.

use crate::basic::{apply_basic, BasicEdit, CpuTensor, MaskInput, NumericInput};
use crate::cancellation::CancellationToken;
use crate::history::VerifiedHistory;
use crate::representation::{apply_representation, KernelLimits, RepresentationEdit};
use nf_checkpoint::serialization::{serialize_tensors, SerializationLimits, TensorPayload};
use nf_checkpoint::tensor_file::{inspect_safetensors, TensorFileLimits};
use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use nf_core::error::{CoreError, ErrorCode, Result};
use nf_core::schemas::{
    DType, MethodRecord, ModelManifest, TargetUpdate, TensorFormat, TensorSpec, Validate, Validated,
};
use nf_graph::inventory::{tensor_set_digest, TensorObservation, TensorRole};
use nf_weight_ir::compilation::CompiledPlan;
use nf_weight_ir::program::{element_count, ModelBinding, Operation, OperationKind, TensorRegion};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::Arc;

mod arithmetic;
mod qwen2;
#[cfg(test)]
mod tests;

fn failure(code: ErrorCode, message: impl Into<String>) -> CoreError {
    CoreError::new(code, "", message)
}
fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| failure(ErrorCode::LimitExceeded, "transaction counter overflow"))
}
fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .ok_or_else(|| failure(ErrorCode::LimitExceeded, "transaction counter overflow"))
}
fn digest<T: Serialize>(value: &T) -> Result<ContentDigest> {
    let raw = serde_json::to_vec(value)
        .map_err(|error| failure(ErrorCode::InvalidValue, error.to_string()))?;
    Ok(CanonicalDocument::parse(&raw, Limits::default())?.digest())
}
fn finish(hash: Sha256) -> ContentDigest {
    format!("sha256:{:x}", hash.finalize())
        .parse()
        .expect("SHA-256 representation")
}

/// Canonical graph of the only registered synthetic arithmetic architecture.
pub fn arithmetic_graph() -> Result<Vec<u8>> {
    arithmetic::graph()
}

#[derive(Debug, Clone)]
enum Group {
    Float(CpuTensor),
    Opaque {
        dtype: DType,
        shape: Vec<u64>,
        bytes: Arc<[u8]>,
    },
}
impl Group {
    fn dtype(&self) -> DType {
        match self {
            Self::Float(tensor) => tensor.dtype(),
            Self::Opaque { dtype, .. } => *dtype,
        }
    }
    fn shape(&self) -> &[u64] {
        match self {
            Self::Float(tensor) => tensor.shape(),
            Self::Opaque { shape, .. } => shape,
        }
    }
    fn byte_len(&self) -> Result<u64> {
        match self {
            Self::Opaque { bytes, .. } => Ok(bytes.len() as u64),
            Self::Float(tensor) => mul(
                element_count(tensor.shape(), "/shape")?,
                tensor.dtype().byte_width(),
            ),
        }
    }
    fn byte_digest(&self) -> ContentDigest {
        let mut hash = Sha256::new();
        match self {
            Self::Float(tensor) => {
                if let Some(values) = tensor.f16_bits() {
                    for value in values {
                        hash.update(value.to_le_bytes());
                    }
                }
                if let Some(values) = tensor.bf16_bits() {
                    for value in values {
                        hash.update(value.to_le_bytes());
                    }
                }
                if let Some(values) = tensor.f32_values() {
                    for value in values {
                        hash.update(value.to_le_bytes());
                    }
                }
                if let Some(values) = tensor.f64_values() {
                    for value in values {
                        hash.update(value.to_le_bytes());
                    }
                }
            }
            Self::Opaque { bytes, .. } => hash.update(bytes),
        }
        finish(hash)
    }
    fn bytes(&self) -> Vec<u8> {
        match self {
            Self::Float(tensor) => tensor.to_le_bytes(),
            Self::Opaque { bytes, .. } => bytes.to_vec(),
        }
    }
}

#[derive(Debug, Clone)]
struct ModelState {
    manifest: ModelManifest,
    groups: BTreeMap<String, Arc<Group>>,
    names: BTreeMap<String, String>,
    assets: BTreeMap<String, Arc<[u8]>>,
}
impl ModelState {
    fn group(&self, name: &str) -> Result<&Group> {
        self.names
            .get(name)
            .and_then(|group| self.groups.get(group))
            .map(Arc::as_ref)
            .ok_or_else(|| failure(ErrorCode::SourceMismatch, "tensor/group is missing"))
    }
    fn specs(&self) -> impl Iterator<Item = (&String, &TensorSpec)> {
        self.manifest
            .parameters
            .iter()
            .chain(&self.manifest.buffers)
    }
    fn hashes(&self) -> BTreeMap<String, ContentDigest> {
        self.groups
            .iter()
            .map(|(id, group)| (id.clone(), group.byte_digest()))
            .collect()
    }
    fn tensor(&self, name: &str) -> Option<&CpuTensor> {
        match self.group(name).ok()? {
            Group::Float(tensor) => Some(tensor),
            _ => None,
        }
    }
    fn bytes(&self, name: &str) -> Result<Vec<u8>> {
        Ok(self.group(name)?.bytes())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StateLoadLimits {
    pub max_checkpoint_bytes: usize,
    pub max_tensor_bytes: u64,
    pub max_tensors: usize,
    pub max_asset_bytes: u64,
}
impl Default for StateLoadLimits {
    fn default() -> Self {
        Self {
            max_checkpoint_bytes: 67_108_864,
            max_tensor_bytes: 67_108_864,
            max_tensors: 1024,
            max_asset_bytes: 1_048_576,
        }
    }
}
impl StateLoadLimits {
    /// Limits suitable for multi-gigabyte open-weight checkpoints (up to 64 GiB).
    pub fn large_model() -> Self {
        Self {
            max_checkpoint_bytes: usize::try_from(68_719_476_736u64).unwrap_or(usize::MAX),
            max_tensor_bytes: 68_719_476_736,
            max_tensors: 16_384,
            max_asset_bytes: 268_435_456,
        }
    }
}
#[derive(Debug, Clone)]
pub struct SourceState {
    state: ModelState,
    manifest: Validated<ModelManifest>,
    manifest_digest: ContentDigest,
    checkpoint: Arc<[u8]>,
    checkpoint_digest: ContentDigest,
    ranges: BTreeMap<String, Range<usize>>,
}
impl SourceState {
    pub fn load(
        manifest: Validated<ModelManifest>,
        checkpoint: Arc<[u8]>,
        assets: BTreeMap<String, Arc<[u8]>>,
        limits: StateLoadLimits,
    ) -> Result<Self> {
        manifest.validate()?;
        if limits.max_checkpoint_bytes == 0
            || limits.max_tensor_bytes == 0
            || limits.max_tensors == 0
            || limits.max_tensors > 65_536
            || limits.max_asset_bytes == 0
        {
            return Err(failure(
                ErrorCode::InvalidValue,
                "positive supported source-state limits required",
            ));
        }
        if manifest.format != TensorFormat::DenseSafetensors || manifest.packing_policy.is_some() {
            return Err(failure(
                ErrorCode::UnsupportedCapability,
                "native source state requires dense storage",
            ));
        }
        if manifest.parameters.values().any(|spec| {
            !matches!(
                spec.dtype,
                DType::F16 | DType::Bf16 | DType::F32 | DType::F64
            )
        }) {
            return Err(failure(
                ErrorCode::UnsupportedCapability,
                "native source parameters must be f16/bf16/f32/f64; other state needs a qualified adapter",
            ));
        }
        if manifest.parameters.len() + manifest.buffers.len() > limits.max_tensors
            || assets.len() > 1024
        {
            return Err(failure(
                ErrorCode::LimitExceeded,
                "source-state metadata bound exceeded",
            ));
        }
        let file = inspect_safetensors(
            &checkpoint,
            TensorFileLimits {
                max_file_bytes: limits.max_checkpoint_bytes,
                ..TensorFileLimits::default()
            },
        )?;
        if manifest
            .assets
            .get("checkpoint")
            .is_some_and(|expected| *expected != file.file_digest)
        {
            return Err(failure(
                ErrorCode::SourceMismatch,
                "source checkpoint identity differs",
            ));
        }
        let mut asset_bytes = 0;
        for bytes in assets.values() {
            asset_bytes = add(asset_bytes, bytes.len() as u64)?;
        }
        if asset_bytes > limits.max_asset_bytes {
            return Err(failure(
                ErrorCode::LimitExceeded,
                "source-state assets exceed bounds",
            ));
        }
        let config = assets
            .get("config")
            .ok_or_else(|| failure(ErrorCode::SourceMismatch, "config asset is required"))?;
        if ContentDigest::of_bytes(config) != manifest.architecture.config {
            return Err(failure(
                ErrorCode::SourceMismatch,
                "config bytes differ from factory configuration",
            ));
        }
        CanonicalDocument::parse(config, Limits::default())?;
        for (name, expected) in &manifest.assets {
            if name == "checkpoint" {
                continue;
            }
            if assets
                .get(name)
                .is_none_or(|bytes| ContentDigest::of_bytes(bytes) != *expected)
            {
                return Err(failure(
                    ErrorCode::SourceMismatch,
                    "declared source asset is missing or changed",
                ));
            }
        }
        if assets
            .keys()
            .any(|name| name != "config" && !manifest.assets.contains_key(name))
        {
            return Err(failure(ErrorCode::SourceMismatch, "unbound source asset"));
        }
        let mut names = BTreeMap::new();
        let mut members: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for (name, spec) in manifest.parameters.iter().chain(&manifest.buffers) {
            names.insert(name.clone(), spec.storage_group.clone());
            members
                .entry(spec.storage_group.clone())
                .or_default()
                .push(name.clone());
        }
        let mut locations = BTreeMap::new();
        for (name, tensor) in &file.tensors {
            let spec = manifest
                .parameters
                .get(name)
                .or_else(|| manifest.buffers.get(name))
                .ok_or_else(|| {
                    failure(ErrorCode::SourceMismatch, "unexpected checkpoint tensor")
                })?;
            if tensor.shape != spec.shape
                || tensor.dtype != spec.dtype
                || tensor.digest != spec.digest
            {
                return Err(failure(
                    ErrorCode::SourceMismatch,
                    "checkpoint tensor metadata/content differs",
                ));
            }
            locations
                .entry(spec.storage_group.clone())
                .or_insert_with(|| tensor.data_range.clone());
        }
        let mut groups = BTreeMap::new();
        let mut tensor_bytes = 0;
        for (id, group_members) in &mut members {
            group_members.sort();
            let canonical = &group_members[0];
            let spec = manifest
                .parameters
                .get(canonical)
                .or_else(|| manifest.buffers.get(canonical))
                .unwrap();
            tensor_bytes = add(tensor_bytes, spec.byte_len()?)?;
            if tensor_bytes > limits.max_tensor_bytes {
                return Err(failure(
                    ErrorCode::LimitExceeded,
                    "source tensor bytes exceed bounds",
                ));
            }
            let range = locations.get(id).ok_or_else(|| {
                failure(
                    ErrorCode::SourceMismatch,
                    "checkpoint storage group missing",
                )
            })?;
            let bytes = &checkpoint[range.clone()];
            let group = match spec.dtype {
                DType::F16 => Group::Float(CpuTensor::from_f16_bits(
                    canonical,
                    spec.shape.clone(),
                    bytes
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|value| u16::from_le_bytes(*value))
                        .collect(),
                )?),
                DType::Bf16 => Group::Float(CpuTensor::from_bf16_bits(
                    canonical,
                    spec.shape.clone(),
                    bytes
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|value| u16::from_le_bytes(*value))
                        .collect(),
                )?),
                DType::F32 => Group::Float(CpuTensor::from_f32(
                    canonical,
                    spec.shape.clone(),
                    bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|value| f32::from_le_bytes(*value))
                        .collect(),
                )?),
                DType::F64 => Group::Float(CpuTensor::from_f64(
                    canonical,
                    spec.shape.clone(),
                    bytes
                        .as_chunks::<8>()
                        .0
                        .iter()
                        .map(|value| f64::from_le_bytes(*value))
                        .collect(),
                )?),
                _ => Group::Opaque {
                    dtype: spec.dtype,
                    shape: spec.shape.clone(),
                    bytes: Arc::from(bytes),
                },
            };
            if group.byte_digest() != spec.digest {
                return Err(failure(
                    ErrorCode::IntegrityMismatch,
                    "decoded group differs",
                ));
            }
            groups.insert(id.clone(), Arc::new(group));
        }
        if tensor_set_digest(&manifest.parameters)? != manifest.parameter_digest {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "source parameter-set identity differs",
            ));
        }
        let state = ModelState {
            manifest: (*manifest).clone(),
            groups,
            names,
            assets,
        };
        Ok(Self {
            manifest_digest: digest(&*manifest)?,
            manifest,
            checkpoint_digest: file.file_digest,
            checkpoint,
            state,
            ranges: locations,
        })
    }
    pub fn manifest(&self) -> &Validated<ModelManifest> {
        &self.manifest
    }
    pub fn manifest_digest(&self) -> &ContentDigest {
        &self.manifest_digest
    }
    pub fn checkpoint_bytes(&self) -> &[u8] {
        &self.checkpoint
    }
    pub fn binding(&self, architecture_graph: ContentDigest) -> ModelBinding {
        ModelBinding {
            checkpoint: self.checkpoint_digest.clone(),
            manifest: self.manifest_digest.clone(),
            parameter_digest: self.manifest.parameter_digest.clone(),
            architecture_graph,
            config: self.manifest.architecture.config.clone(),
            tokenizer: self.manifest.assets.get("tokenizer").cloned(),
        }
    }
    pub fn assets(&self) -> &BTreeMap<String, Arc<[u8]>> {
        &self.state.assets
    }
    pub fn tensor(&self, name: &str) -> Option<&CpuTensor> {
        self.state.tensor(name)
    }
    pub fn tensor_bytes(&self, name: &str) -> Result<Vec<u8>> {
        self.state.bytes(name)
    }
    pub fn payload_bytes(&self) -> Result<u64> {
        self.state
            .groups
            .values()
            .try_fold(0u64, |sum, group| add(sum, group.byte_len()?))
    }
    pub fn verify_integrity(&self) -> Result<()> {
        check_source_definition(self)?;
        let hashes = self.state.hashes();
        for (_, spec) in self.state.specs() {
            if hashes[&spec.storage_group] != spec.digest {
                return Err(failure(
                    ErrorCode::SourceMismatch,
                    "source runtime data differs from immutable manifest",
                ));
            }
        }
        Ok(())
    }
    /// A trusted loader reconstructed one actual shared group per declared tie.
    /// Canonical file spans describe its immutable byte representation.
    pub fn verify_smoke(
        &self,
        suite: &Validated<SmokeSuite>,
        limits: ExecutionLimits,
    ) -> Result<(ContentDigest, u64)> {
        suite.validate()?;
        if self.payload_bytes()? > limits.max_verification_bytes {
            return Err(failure(
                ErrorCode::LimitExceeded,
                "smoke source verification byte bound exceeded",
            ));
        }
        self.verify_integrity()?;
        run_smoke(&self.state, suite, limits)
    }

    /// Registered native fixture inference; no callback/model object crosses IPC.
    pub fn infer_linear_fixture(&self, input: &[f64], limits: ExecutionLimits) -> Result<Vec<f64>> {
        if input.len() > limits.max_smoke_vector_elements
            || limits.max_smoke_vector_elements > 4096
            || limits.max_smoke_vector_elements == 0
        {
            return Err(failure(
                ErrorCode::LimitExceeded,
                "inference input vector bound exceeded",
            ));
        }
        let suite = Validated::new(SmokeSuite {
            schema_version: 1,
            adapter: "nf.linear_fixture.v1".to_owned(),
            probes: vec![SmokeProbe {
                input: input.to_vec(),
                expected: None,
                absolute_tolerance: 0.0,
                max_abs_output: f64::MAX,
            }],
        })?;
        if self.payload_bytes()? > limits.max_verification_bytes {
            return Err(failure(
                ErrorCode::LimitExceeded,
                "inference source verification bound exceeded",
            ));
        }
        self.verify_integrity()?;
        let (rows, cols, _) = smoke_shape(&self.state, &suite, limits)?;
        linear_vector(self.state.tensor("w").unwrap(), input, rows, cols)
    }

    /// Exact registered synthetic fixture forward, never an arbitrary factory.
    pub fn infer_arithmetic_fixture(
        &self,
        input: &[f64],
        limits: ExecutionLimits,
    ) -> Result<Vec<f64>> {
        if self.payload_bytes()? > limits.max_verification_bytes {
            return Err(failure(
                ErrorCode::LimitExceeded,
                "arithmetic source verification bound exceeded",
            ));
        }
        self.verify_integrity()?;
        let suite = Validated::new(SmokeSuite {
            schema_version: 1,
            adapter: "nf.arithmetic_relu.v1".into(),
            probes: vec![SmokeProbe {
                input: input.to_vec(),
                expected: None,
                absolute_tolerance: 0.0,
                max_abs_output: f64::MAX,
            }],
        })?;
        arithmetic::shape(&self.state, &suite, limits)?;
        arithmetic::forward(&self.state, input)
    }
    pub fn observations(&self) -> Vec<TensorObservation<'_>> {
        self.state
            .specs()
            .map(|(name, spec)| TensorObservation {
                name,
                role: if self.manifest.parameters.contains_key(name) {
                    TensorRole::Parameter
                } else {
                    TensorRole::Buffer
                },
                shape: &spec.shape,
                dtype: spec.dtype,
                bytes: &self.checkpoint[self.ranges[&spec.storage_group].clone()],
            })
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SmokeProbe {
    pub input: Vec<f64>,
    pub expected: Option<Vec<f64>>,
    pub absolute_tolerance: f64,
    pub max_abs_output: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SmokeSuite {
    pub schema_version: u32,
    pub adapter: String,
    pub probes: Vec<SmokeProbe>,
}
impl Validate for SmokeSuite {
    fn validate(&self) -> Result<()> {
        if self.schema_version != 1
            || !matches!(
                self.adapter.as_str(),
                "nf.linear_fixture.v1" | "nf.arithmetic_relu.v1" | "nf.qwen2_dense.v1"
            )
        {
            return Err(failure(
                ErrorCode::UnsupportedCapability,
                "smoke adapter/version unsupported",
            ));
        }
        if self.probes.is_empty() || self.probes.len() > 64 {
            return Err(failure(
                ErrorCode::InvalidValue,
                "bounded nonempty smoke suite required",
            ));
        }
        for probe in &self.probes {
            if probe.input.len() > 4096
                || probe.expected.as_ref().is_some_and(|v| v.len() > 4096)
                || probe
                    .input
                    .iter()
                    .chain(probe.expected.iter().flatten())
                    .any(|x| !x.is_finite())
                || !probe.absolute_tolerance.is_finite()
                || probe.absolute_tolerance < 0.0
                || !probe.max_abs_output.is_finite()
                || probe.max_abs_output <= 0.0
            {
                return Err(failure(ErrorCode::InvalidValue, "invalid smoke probe"));
            }
        }
        Ok(())
    }
}
impl SmokeSuite {
    pub fn digest(&self) -> Result<ContentDigest> {
        self.validate()?;
        digest(self)
    }
}
#[derive(Debug, Clone, Copy)]
pub struct ExecutionLimits {
    pub max_verification_bytes: u64,
    pub max_smoke_work: u64,
    pub max_smoke_vector_elements: usize,
}
impl Default for ExecutionLimits {
    fn default() -> Self {
        Self {
            max_verification_bytes: 67_108_864,
            max_smoke_work: 1_000_000,
            max_smoke_vector_elements: 4096,
        }
    }
}

fn smoke_shape(
    state: &ModelState,
    suite: &SmokeSuite,
    limits: ExecutionLimits,
) -> Result<(usize, usize, u64)> {
    suite.validate()?;
    if suite.adapter == "nf.arithmetic_relu.v1" {
        return arithmetic::shape(state, suite, limits);
    }
    if suite.adapter == "nf.qwen2_dense.v1" {
        return qwen2::shape(state, suite, limits);
    }
    if state.manifest.architecture.id != "fixture.cpu" || state.manifest.architecture.version != "1"
    {
        return Err(failure(
            ErrorCode::UnsupportedCapability,
            "required model smoke adapter is not qualified",
        ));
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct EmptyConfig {}
    let _: EmptyConfig = serde_json::from_slice(state.assets["config"].as_ref()).map_err(|_| {
        failure(
            ErrorCode::UnsupportedCapability,
            "linear fixture config must be the known empty configuration",
        )
    })?;
    let weight = state.tensor("w").ok_or_else(|| {
        failure(
            ErrorCode::UnsupportedCapability,
            "linear fixture requires floating weight w",
        )
    })?;
    if !state.manifest.parameters.contains_key("w") || weight.shape().len() != 2 {
        return Err(failure(
            ErrorCode::UnsupportedCapability,
            "linear fixture weight must be a matrix parameter",
        ));
    }
    let (rows, cols) = (weight.shape()[0], weight.shape()[1]);
    if rows == 0
        || cols == 0
        || rows > limits.max_smoke_vector_elements as u64
        || cols > limits.max_smoke_vector_elements as u64
    {
        return Err(failure(
            ErrorCode::LimitExceeded,
            "smoke vector/shape bound exceeded",
        ));
    }
    for probe in &suite.probes {
        if probe.input.len() as u64 != cols
            || probe
                .expected
                .as_ref()
                .is_some_and(|v| v.len() as u64 != rows)
        {
            return Err(failure(
                ErrorCode::SmokeFailed,
                "smoke input/expected dimensions differ",
            ));
        }
    }
    let work = mul(
        suite.probes.len() as u64,
        add(mul(mul(rows, cols)?, 2)?, mul(add(rows, cols)?, 8)?)?,
    )?;
    if work > limits.max_smoke_work {
        return Err(failure(
            ErrorCode::LimitExceeded,
            "smoke work limit exceeded",
        ));
    }
    Ok((rows as usize, cols as usize, work))
}
fn linear_vector(weight: &CpuTensor, input: &[f64], rows: usize, cols: usize) -> Result<Vec<f64>> {
    let mut output = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut sum = 0.0;
        for (col, input) in input.iter().enumerate().take(cols) {
            let value = weight.value(row * cols + col)?.as_f64();
            sum = if weight.dtype() == DType::F32 {
                ((sum as f32) + (value as f32) * (*input as f32)) as f64
            } else {
                sum + value * input
            };
        }
        if !sum.is_finite() {
            return Err(failure(
                ErrorCode::NonFinite,
                "registered forward returned nonfinite output",
            ));
        }
        output.push(sum);
    }
    Ok(output)
}
fn run_smoke(
    state: &ModelState,
    suite: &SmokeSuite,
    limits: ExecutionLimits,
) -> Result<(ContentDigest, u64)> {
    let (rows, cols, work) = smoke_shape(state, suite, limits)?;
    let mut hash = Sha256::new();
    for probe in &suite.probes {
        let vector = if suite.adapter == "nf.arithmetic_relu.v1" {
            arithmetic::forward(state, &probe.input)
        } else if suite.adapter == "nf.qwen2_dense.v1" {
            qwen2::forward(state, &probe.input)
        } else {
            linear_vector(state.tensor("w").unwrap(), &probe.input, rows, cols)
        };
        for (row, sum) in vector
            .map_err(|error| {
                if error.code == ErrorCode::NonFinite {
                    failure(
                        ErrorCode::SmokeFailed,
                        "candidate failed finite forward smoke",
                    )
                } else {
                    error
                }
            })?
            .into_iter()
            .enumerate()
        {
            if !sum.is_finite()
                || sum.abs() > probe.max_abs_output
                || probe
                    .expected
                    .as_ref()
                    .is_some_and(|expected| (sum - expected[row]).abs() > probe.absolute_tolerance)
            {
                return Err(failure(
                    ErrorCode::SmokeFailed,
                    "candidate failed registered forward smoke probe",
                ));
            }
            hash.update(sum.to_le_bytes());
        }
    }
    Ok((finish(hash), work))
}

#[derive(Debug, Clone, Serialize)]
pub struct TensorChange {
    pub before: ContentDigest,
    pub after: ContentDigest,
    pub role: String,
}
#[derive(Debug, Clone, Serialize)]
pub struct StepReceipt {
    pub instruction: String,
    pub operation: OperationKind,
    pub storage_group: String,
    pub changed_parameters: u64,
    pub work_upper_bound: u64,
}
/// Exact row-major changed coordinates, compressed into contiguous spans.
#[derive(Debug, Clone, Serialize)]
pub struct ChangedSpan {
    pub start_flat_index: u64,
    pub elements: u64,
}
#[derive(Debug, Clone, Serialize)]
pub struct TransactionReceipt {
    pub schema_version: u32,
    pub stage: String,
    pub plan: ContentDigest,
    pub source_manifest: ContentDigest,
    pub source_checkpoint: ContentDigest,
    pub result_manifest: ContentDigest,
    pub before_parameters: ContentDigest,
    pub after_parameters: ContentDigest,
    pub before_buffers: ContentDigest,
    pub after_buffers: ContentDigest,
    pub tensor_changes: BTreeMap<String, TensorChange>,
    pub effective_writes: Vec<TensorRegion>,
    pub actual_changed_coordinates: BTreeMap<String, Vec<ChangedSpan>>,
    pub changed_parameters: u64,
    pub copied_group_bytes: u64,
    pub verification_byte_upper_bound: u64,
    pub steps: Vec<StepReceipt>,
    pub smoke_suite: ContentDigest,
    pub smoke_outputs: ContentDigest,
    pub smoke_work_upper_bound: u64,
    pub method: MethodRecord,
    pub input_history_assurance: String,
    pub promotion_status: String,
}
#[derive(Debug, Clone)]
pub struct VerifiedCandidate {
    state: ModelState,
    manifest: Validated<ModelManifest>,
    receipt: TransactionReceipt,
    digest: ContentDigest,
}
impl VerifiedCandidate {
    pub fn manifest(&self) -> &Validated<ModelManifest> {
        &self.manifest
    }
    pub fn tensor(&self, name: &str) -> Option<&CpuTensor> {
        self.state.tensor(name)
    }
    pub fn tensor_bytes(&self, name: &str) -> Result<Vec<u8>> {
        self.state.bytes(name)
    }
    pub fn assets(&self) -> &BTreeMap<String, Arc<[u8]>> {
        &self.state.assets
    }
    pub fn receipt(&self) -> &TransactionReceipt {
        &self.receipt
    }
    pub fn digest(&self) -> &ContentDigest {
        &self.digest
    }
    /// Real in-memory serialization/reload for ancestry. No file export or promotion.
    pub fn materialize_source(&self, limits: MaterializationLimits) -> Result<SourceState> {
        let total = self
            .state
            .groups
            .values()
            .try_fold(0u64, |sum, group| add(sum, group.byte_len()?))?;
        let staging = add(
            mul(total, 3)?,
            add(mul(limits.serialization.max_header_bytes as u64, 2)?, 16)?,
        )?;
        if total > limits.serialization.max_tensor_bytes
            || staging > limits.serialization.max_staging_bytes
        {
            return Err(failure(
                ErrorCode::LimitExceeded,
                "candidate snapshot staging bound exceeded",
            ));
        }
        let mut encoded = BTreeMap::new();
        for (group_id, group) in &self.state.groups {
            let name = self
                .state
                .names
                .iter()
                .find(|(_, id)| *id == group_id)
                .map(|(name, _)| name.clone())
                .ok_or_else(|| {
                    failure(
                        ErrorCode::IntegrityMismatch,
                        "group has no canonical tensor name",
                    )
                })?;
            encoded.insert(name, (group.dtype(), group.shape().to_vec(), group.bytes()));
        }
        let payloads = encoded
            .iter()
            .map(|(name, (dtype, shape, bytes))| {
                (
                    name.clone(),
                    TensorPayload {
                        dtype: *dtype,
                        shape,
                        bytes,
                    },
                )
            })
            .collect();
        let serialized = serialize_tensors(&payloads, limits.serialization)?;
        let mut manifest = (*self.manifest).clone();
        manifest
            .assets
            .insert("checkpoint".to_owned(), serialized.digest);
        drop(payloads);
        drop(encoded);
        SourceState::load(
            Validated::new(manifest)?,
            Arc::from(serialized.bytes),
            self.state.assets.clone(),
            limits.load,
        )
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MaterializationLimits {
    pub serialization: SerializationLimits,
    pub load: StateLoadLimits,
}

fn check_source_definition(source: &SourceState) -> Result<()> {
    if source.state.manifest != *source.manifest {
        return Err(failure(
            ErrorCode::SourceMismatch,
            "source metadata changed after loading",
        ));
    }
    let expected: BTreeMap<_, _> = source
        .state
        .specs()
        .map(|(name, spec)| (name.clone(), spec.storage_group.clone()))
        .collect();
    let groups: BTreeSet<_> = expected.values().cloned().collect();
    if source.state.names != expected
        || source.state.groups.keys().cloned().collect::<BTreeSet<_>>() != groups
    {
        return Err(failure(
            ErrorCode::SourceMismatch,
            "source alias/group mapping changed after loading",
        ));
    }
    for (name, spec) in source.state.specs() {
        let group = source.state.group(name)?;
        if group.dtype() != spec.dtype
            || group.shape() != spec.shape
            || group.byte_len()? != spec.byte_len()?
            || matches!(
                spec.dtype,
                DType::F16 | DType::Bf16 | DType::F32 | DType::F64
            ) != matches!(group, Group::Float(_))
        {
            return Err(failure(
                ErrorCode::SourceMismatch,
                "source runtime layout changed after loading",
            ));
        }
    }
    if source
        .state
        .assets
        .keys()
        .any(|name| name != "config" && !source.manifest.assets.contains_key(name))
    {
        return Err(failure(
            ErrorCode::SourceMismatch,
            "source contains an unbound asset",
        ));
    }
    let config = source.state.assets.get("config").ok_or_else(|| {
        failure(
            ErrorCode::SourceMismatch,
            "source configuration disappeared",
        )
    })?;
    if ContentDigest::of_bytes(config) != source.manifest.architecture.config {
        return Err(failure(
            ErrorCode::SourceMismatch,
            "source configuration changed",
        ));
    }
    for (name, expected) in &source.manifest.assets {
        if name != "checkpoint"
            && source
                .state
                .assets
                .get(name)
                .is_none_or(|bytes| ContentDigest::of_bytes(bytes) != *expected)
        {
            return Err(failure(ErrorCode::SourceMismatch, "source asset changed"));
        }
    }
    Ok(())
}

fn check_layout(candidate: &ModelState, source: &ModelState) -> Result<()> {
    if candidate.manifest != source.manifest
        || candidate.names != source.names
        || candidate.groups.keys().ne(source.groups.keys())
        || candidate.assets.keys().ne(source.assets.keys())
    {
        return Err(failure(
            ErrorCode::ScopeViolation,
            "model metadata/names/groups/assets drifted",
        ));
    }
    for (name, bytes) in &candidate.assets {
        if !Arc::ptr_eq(bytes, &source.assets[name])
            && ContentDigest::of_bytes(bytes) != ContentDigest::of_bytes(&source.assets[name])
        {
            return Err(failure(ErrorCode::ScopeViolation, "model asset drifted"));
        }
    }
    for (name, spec) in source.specs() {
        let group = candidate.group(name)?;
        if !matches!(
            (group, source.group(name)?),
            (Group::Float(_), Group::Float(_)) | (Group::Opaque { .. }, Group::Opaque { .. })
        ) {
            return Err(failure(
                ErrorCode::ScopeViolation,
                "storage representation drifted",
            ));
        }
        if group.dtype() != spec.dtype
            || group.shape() != spec.shape
            || group.byte_len()? != spec.byte_len()?
        {
            return Err(failure(
                ErrorCode::ScopeViolation,
                "tensor shape/dtype/storage layout drifted",
            ));
        }
    }
    Ok(())
}
fn inside(linear: u64, shape: &[u64], region: &TensorRegion) -> bool {
    let mut remaining = linear;
    for axis in (0..shape.len()).rev() {
        if shape[axis] == 0 {
            return false;
        }
        let coordinate = remaining % shape[axis];
        remaining /= shape[axis];
        if coordinate < region.start[axis] || coordinate >= region.start[axis] + region.shape[axis]
        {
            return false;
        }
    }
    true
}
fn outside_hash(tensor: &CpuTensor, region: &TensorRegion) -> Result<ContentDigest> {
    tensor.validate_region(region)?;
    let count = element_count(tensor.shape(), "/shape")?;
    let mut hash = Sha256::new();
    for index in 0..count {
        if !inside(index, tensor.shape(), region) {
            if let Some(values) = tensor.f16_bits() {
                hash.update(values[index as usize].to_le_bytes());
            }
            if let Some(values) = tensor.bf16_bits() {
                hash.update(values[index as usize].to_le_bytes());
            }
            if let Some(values) = tensor.f32_values() {
                hash.update(values[index as usize].to_le_bytes());
            }
            if let Some(values) = tensor.f64_values() {
                hash.update(values[index as usize].to_le_bytes());
            }
        }
    }
    Ok(finish(hash))
}
fn same_cell(a: &Group, b: &Group, index: usize) -> Result<bool> {
    match (a, b) {
        (Group::Float(a), Group::Float(b)) => Ok(a.value(index)?.same_bits(b.value(index)?)),
        (
            Group::Opaque {
                bytes: a, dtype, ..
            },
            Group::Opaque { bytes: b, .. },
        ) => {
            let width = dtype.byte_width() as usize;
            Ok(a[index * width..(index + 1) * width] == b[index * width..(index + 1) * width])
        }
        _ => Err(failure(
            ErrorCode::ScopeViolation,
            "storage representation changed",
        )),
    }
}
fn execute_step(
    plan: &CompiledPlan,
    candidate: &mut ModelState,
    index: usize,
) -> Result<StepReceipt> {
    let instruction = &plan.instructions()[index];
    let operation = &instruction.operation;
    let group_id = candidate
        .names
        .get(&operation.target().tensor)
        .ok_or_else(|| failure(ErrorCode::SourceMismatch, "operation tensor missing"))?
        .clone();
    let group = Arc::make_mut(candidate.groups.get_mut(&group_id).unwrap());
    let tensor = match group {
        Group::Float(tensor) => tensor,
        _ => {
            return Err(failure(
                ErrorCode::UnsupportedCapability,
                "opaque storage cannot be edited",
            ))
        }
    };
    let mut region = operation.target().clone();
    region.tensor = tensor.name().to_owned();
    let operand = |id: &str| {
        plan.operands()
            .get(id)
            .ok_or_else(|| failure(ErrorCode::IntegrityMismatch, "compiled operand missing"))
    };
    let basic = match operation {
        Operation::ReplaceSlice { value, .. } => {
            Some(BasicEdit::Replace(NumericInput::Operand(operand(value)?)))
        }
        Operation::AddDelta { delta, .. } => {
            Some(BasicEdit::Add(NumericInput::Operand(operand(delta)?)))
        }
        Operation::Scale { factor, .. } => Some(BasicEdit::Scale(*factor)),
        Operation::Mask { mask, .. } => Some(BasicEdit::Mask(MaskInput::Operand(operand(mask)?))),
        _ => None,
    };
    let (changed, work) = if let Some(edit) = basic {
        let receipt = apply_basic(
            tensor,
            &region,
            edit,
            plan.numerical(),
            plan.program().constraints.max_changed_parameters,
        )?;
        (receipt.changed_parameters, receipt.cell_visits)
    } else {
        let edit = match operation {
            Operation::PermuteAxis {
                axis, permutation, ..
            } => RepresentationEdit::PermuteAxis {
                axis: *axis,
                permutation,
            },
            Operation::LinearMap { matrix, side, .. } => RepresentationEdit::LinearMap {
                matrix: NumericInput::Operand(operand(matrix)?),
                side: *side,
            },
            Operation::LowRankUpdate {
                left,
                right,
                strength,
                ..
            } => RepresentationEdit::LowRankUpdate {
                left: NumericInput::Operand(operand(left)?),
                right: NumericInput::Operand(operand(right)?),
                strength: *strength,
            },
            Operation::SubspaceProjection {
                basis,
                side,
                strength,
                orthonormal_tolerance,
                ..
            } => RepresentationEdit::Projection {
                basis: NumericInput::Operand(operand(basis)?),
                side: *side,
                strength: *strength,
                orthonormal_tolerance: *orthonormal_tolerance,
            },
            _ => {
                return Err(failure(
                    ErrorCode::UnsupportedOperation,
                    "compiled instruction has no native executor",
                ))
            }
        };
        let receipt = apply_representation(
            tensor,
            &region,
            edit,
            plan.numerical(),
            KernelLimits {
                max_changed_parameters: plan.program().constraints.max_changed_parameters,
                max_scratch_bytes: plan.summary().allocations.peak_kernel_scratch_bytes,
                max_work_units: plan.summary().allocations.kernel_work_units,
            },
        )?;
        (receipt.changed_parameters, receipt.work_upper_bound)
    };
    Ok(StepReceipt {
        instruction: instruction.id.clone(),
        operation: operation.kind(),
        storage_group: group_id,
        changed_parameters: changed,
        work_upper_bound: work,
    })
}

#[derive(Clone, Copy)]
enum Fault {
    None,
    #[cfg(test)]
    Abort(usize),
    #[cfg(test)]
    Cancel(usize),
    #[cfg(test)]
    Outside(usize),
    #[cfg(test)]
    Buffer(usize),
    #[cfg(test)]
    Alias(usize),
    #[cfg(test)]
    Metadata(usize),
    #[cfg(test)]
    Shape(usize),
    #[cfg(test)]
    DType(usize),
}

pub fn execute(
    plan: &CompiledPlan,
    source: &SourceState,
    smoke: &Validated<SmokeSuite>,
    limits: ExecutionLimits,
) -> Result<VerifiedCandidate> {
    execute_inner(plan, source, smoke, limits, Fault::None)
}
pub fn execute_cancellable(
    plan: &CompiledPlan,
    source: &SourceState,
    smoke: &Validated<SmokeSuite>,
    limits: ExecutionLimits,
    cancellation: &CancellationToken,
) -> Result<VerifiedCandidate> {
    execute_verified_inner(
        plan,
        source,
        smoke,
        limits,
        None,
        Fault::None,
        Some(cancellation),
    )
}

pub fn execute_with_history(
    plan: &CompiledPlan,
    source: &SourceState,
    smoke: &Validated<SmokeSuite>,
    limits: ExecutionLimits,
    history: &VerifiedHistory,
) -> Result<VerifiedCandidate> {
    execute_verified_inner(
        plan,
        source,
        smoke,
        limits,
        Some(history),
        Fault::None,
        None,
    )
}
fn execute_inner(
    plan: &CompiledPlan,
    source: &SourceState,
    smoke: &Validated<SmokeSuite>,
    limits: ExecutionLimits,
    fault: Fault,
) -> Result<VerifiedCandidate> {
    execute_verified_inner(plan, source, smoke, limits, None, fault, None)
}
fn execute_verified_inner(
    plan: &CompiledPlan,
    source: &SourceState,
    smoke: &Validated<SmokeSuite>,
    limits: ExecutionLimits,
    history: Option<&VerifiedHistory>,
    fault: Fault,
    cancellation: Option<&CancellationToken>,
) -> Result<VerifiedCandidate> {
    let check_cancel = || cancellation.map_or(Ok(()), CancellationToken::check);
    check_cancel()?;
    #[cfg(not(test))]
    let _ = fault;
    if limits.max_verification_bytes == 0
        || limits.max_smoke_work == 0
        || limits.max_smoke_vector_elements == 0
        || limits.max_smoke_vector_elements > 4096
    {
        return Err(failure(
            ErrorCode::InvalidValue,
            "positive bounded execution limits required",
        ));
    }
    match history {
        Some(history) => history.check_child(plan, source)?,
        None if !plan.program().lineage.parent_candidates.is_empty() => {
            return Err(failure(
                ErrorCode::UnsupportedCapability,
                "parent receipts require a verified replay witness",
            ))
        }
        None => (),
    }
    if source.manifest_digest != plan.summary().source_manifest
        || source.checkpoint_digest != plan.program().base_model.checkpoint
        || source.manifest.parameter_digest != plan.summary().source_parameters
        || tensor_set_digest(&source.manifest.buffers)? != plan.summary().source_buffers
    {
        return Err(failure(
            ErrorCode::SourceMismatch,
            "compiled plan source binding is stale",
        ));
    }
    check_source_definition(source)?;
    if smoke.digest()? != plan.program().verification.smoke_suite {
        return Err(failure(
            ErrorCode::VerificationPolicyWeakened,
            "smoke suite differs from compiled binding",
        ));
    }
    smoke_shape(&source.state, smoke, limits)?;
    let total = source
        .state
        .groups
        .values()
        .try_fold(0u64, |sum, group| add(sum, group.byte_len()?))?;
    let asset_bytes = source
        .state
        .assets
        .values()
        .try_fold(0u64, |sum, bytes| add(sum, bytes.len() as u64))?;
    let verification_bound = mul(
        add(total, asset_bytes)?,
        add(6, mul(plan.instructions().len() as u64, 2)?)?,
    )?;
    if verification_bound > limits.max_verification_bytes {
        return Err(failure(
            ErrorCode::LimitExceeded,
            "full verification byte admission bound exceeded",
        ));
    }
    let before = source.state.hashes();
    for (_, spec) in source.state.specs() {
        if before[&spec.storage_group] != spec.digest {
            return Err(failure(
                ErrorCode::SourceMismatch,
                "source data changed after compilation",
            ));
        }
    }
    let mut candidate = source.state.clone();
    let mut copied = 0;
    let mut touched = BTreeSet::new();
    let mut steps = Vec::new();
    let mut work = 0;
    for index in 0..plan.instructions().len() {
        check_cancel()?;
        let target = plan.instructions()[index].operation.target();
        let id = source.state.names[&target.tensor].clone();
        let current = candidate.tensor(&target.tensor).ok_or_else(|| {
            failure(
                ErrorCode::UnsupportedCapability,
                "target requires floating state",
            )
        })?;
        let mut canonical = target.clone();
        canonical.tensor = current.name().to_owned();
        let outside_before = outside_hash(current, &canonical)?;
        // Keep all other groups shared during a step. An unauthorized COW
        // mutation changes their identity, even when it restores equal values.
        let others: BTreeMap<_, _> = candidate
            .groups
            .iter()
            .filter(|(name, _)| **name != id)
            .map(|(name, group)| (name.clone(), group.clone()))
            .collect();
        if touched.insert(id.clone()) {
            copied = add(copied, candidate.groups[&id].byte_len()?)?;
        }
        if copied > plan.summary().allocations.candidate_copy_bytes {
            return Err(failure(
                ErrorCode::LimitExceeded,
                "candidate copy budget exceeded",
            ));
        }
        let step = execute_step(plan, &mut candidate, index)?;
        #[cfg(test)]
        tests::inject(&mut candidate, index, fault)?;
        #[cfg(test)]
        if matches!(fault, Fault::Cancel(step) if step == index) {
            if let Some(token) = cancellation {
                token.request();
            }
        }
        check_cancel()?;
        check_layout(&candidate, &source.state)?;
        for (name, old) in &others {
            if !Arc::ptr_eq(old, &candidate.groups[name]) {
                return Err(failure(
                    ErrorCode::ScopeViolation,
                    "operation changed another storage group",
                ));
            }
        }
        let current = candidate.tensor(&target.tensor).unwrap();
        if outside_hash(current, &canonical)? != outside_before {
            return Err(failure(
                ErrorCode::ScopeViolation,
                "operation changed coordinates outside its own mask",
            ));
        }
        work = add(work, step.work_upper_bound)?;
        if work > plan.summary().allocations.kernel_work_units {
            return Err(failure(
                ErrorCode::LimitExceeded,
                "kernel receipts exceeded compiled work bound",
            ));
        }
        steps.push(step);
    }
    check_layout(&candidate, &source.state)?;
    let after = candidate.hashes();
    let mut changed = 0;
    let mut actual_changed_coordinates: BTreeMap<String, Vec<ChangedSpan>> = BTreeMap::new();
    let mut coordinate_spans = 0;
    for (id, group) in &candidate.groups {
        let original = &source.state.groups[id];
        let aliases: Vec<_> = candidate
            .names
            .iter()
            .filter(|(_, key)| *key == id)
            .map(|(name, _)| name)
            .collect();
        let count = element_count(group.shape(), "/shape")?;
        let is_parameter = aliases
            .iter()
            .any(|name| source.manifest.parameters.contains_key(*name));
        let mut group_changes: Vec<ChangedSpan> = Vec::new();
        for index in 0..count {
            if index % 4096 == 0 {
                check_cancel()?;
            }
            if let Group::Float(tensor) = group.as_ref() {
                let value = tensor.value(index as usize)?.as_f64();
                if !value.is_finite() {
                    return Err(failure(
                        ErrorCode::NonFinite,
                        "candidate violates finite invariant",
                    ));
                }
                if is_parameter && value.abs() > plan.numerical().max_abs_parameter {
                    return Err(failure(
                        ErrorCode::InvalidValue,
                        "candidate exceeds parameter magnitude bound",
                    ));
                }
            }
            if !same_cell(original, group, index as usize)? {
                for name in &aliases {
                    if source.manifest.buffers.contains_key(*name)
                        || !plan.summary().effective_writes.iter().any(|mask| {
                            mask.tensor.as_str() == name.as_str()
                                && inside(index, group.shape(), mask)
                        })
                    {
                        return Err(failure(
                            ErrorCode::ScopeViolation,
                            "candidate changed undeclared coordinates or buffers",
                        ));
                    }
                }
                changed = add(changed, 1)?;
                if let Some(last) = group_changes
                    .last_mut()
                    .filter(|last| last.start_flat_index + last.elements == index)
                {
                    last.elements += 1;
                } else {
                    coordinate_spans += 1;
                    if coordinate_spans > 4096 {
                        return Err(failure(
                            ErrorCode::LimitExceeded,
                            "actual-coordinate receipt exceeds 4096 spans",
                        ));
                    }
                    group_changes.push(ChangedSpan {
                        start_flat_index: index,
                        elements: 1,
                    });
                }
            }
        }
        if !group_changes.is_empty() {
            actual_changed_coordinates.insert(id.clone(), group_changes);
        }
        if !touched.contains(id) && !Arc::ptr_eq(group, original) {
            return Err(failure(
                ErrorCode::ScopeViolation,
                "untouched storage was copied or changed",
            ));
        }
    }
    if changed > plan.program().constraints.max_changed_parameters
        || changed > plan.summary().affected_parameter_upper_bound
    {
        return Err(failure(
            ErrorCode::ParameterBudgetExceeded,
            "final actual change count exceeds plan",
        ));
    }
    if source.state.hashes() != before {
        return Err(failure(ErrorCode::IntegrityMismatch, "source was modified"));
    }
    check_cancel()?;
    let (outputs, smoke_work) = run_smoke(&candidate, smoke, limits)?;
    check_cancel()?;
    let mut changes = BTreeMap::new();
    for (name, spec) in source.state.specs() {
        changes.insert(
            name.clone(),
            TensorChange {
                before: before[&spec.storage_group].clone(),
                after: after[&spec.storage_group].clone(),
                role: if source.manifest.parameters.contains_key(name) {
                    "parameter"
                } else {
                    "buffer"
                }
                .to_owned(),
            },
        );
    }
    for spec in candidate
        .manifest
        .parameters
        .values_mut()
        .chain(candidate.manifest.buffers.values_mut())
    {
        spec.digest = after[&spec.storage_group].clone();
    }
    candidate.manifest.parameter_digest = tensor_set_digest(&candidate.manifest.parameters)?;
    candidate.manifest.assets.remove("checkpoint");
    let manifest = Validated::new(candidate.manifest.clone())?;
    let mut method = plan.program().lineage.method.clone();
    method.changed_parameters = changed;
    method.target_update = if changed == 0 {
        TargetUpdate::NoChange
    } else {
        TargetUpdate::DirectTensorEdit
    };
    let receipt = TransactionReceipt {
        schema_version: 1,
        stage: "tensor_scope_and_smoke".to_owned(),
        plan: plan.digest().clone(),
        source_manifest: source.manifest_digest.clone(),
        source_checkpoint: source.checkpoint_digest.clone(),
        result_manifest: digest(&*manifest)?,
        before_parameters: source.manifest.parameter_digest.clone(),
        after_parameters: manifest.parameter_digest.clone(),
        before_buffers: plan.summary().source_buffers.clone(),
        after_buffers: tensor_set_digest(&manifest.buffers)?,
        tensor_changes: changes,
        effective_writes: plan.summary().effective_writes.clone(),
        actual_changed_coordinates,
        changed_parameters: changed,
        copied_group_bytes: copied,
        verification_byte_upper_bound: verification_bound,
        steps,
        smoke_suite: smoke.digest()?,
        smoke_outputs: outputs,
        smoke_work_upper_bound: smoke_work,
        method,
        input_history_assurance: "declared_by_plan; full experiment ledger required".to_owned(),
        promotion_status: "not_evaluated".to_owned(),
    };
    let result = VerifiedCandidate {
        digest: digest(&receipt)?,
        state: candidate,
        manifest,
        receipt,
    };
    check_cancel()?;
    Ok(result)
}
