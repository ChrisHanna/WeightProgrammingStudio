//! CPU inventory registration verifies actual borrowed storage spans, never
//! content equality alone. Process addresses are not serialized into identity.

use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use nf_core::error::{child_path, CoreError, ErrorCode, Result};
use nf_core::schemas::{DType, ModelManifest, TensorFormat, TensorSpec, Validated};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

fn failure(code: ErrorCode, path: &str, message: impl Into<String>) -> CoreError {
    CoreError::new(code, path, message)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorRole {
    Parameter,
    Buffer,
}

/// Constructed by a trusted runtime adapter from real CPU slices. This API
/// cannot certify a CUDA allocation using a caller-supplied identity string.
#[derive(Debug)]
pub struct TensorObservation<'a> {
    pub name: &'a str,
    pub role: TensorRole,
    pub shape: &'a [u64],
    pub dtype: DType,
    pub bytes: &'a [u8],
}

#[derive(Debug, Clone, Copy)]
pub struct InventoryLimits {
    pub max_tensors: usize,
    pub max_observed_bytes: u64,
}
impl Default for InventoryLimits {
    fn default() -> Self {
        Self {
            max_tensors: 1024,
            max_observed_bytes: 67_108_864,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct InventoryEntry {
    pub role: TensorRole,
    pub shape: Vec<u64>,
    pub dtype: DType,
    pub device: String,
    pub content_digest: ContentDigest,
    pub storage_group: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct VerifiedInventory {
    source_manifest_digest: ContentDigest,
    parameter_digest: ContentDigest,
    buffer_digest: ContentDigest,
    entries: BTreeMap<String, InventoryEntry>,
    alias_groups: BTreeMap<String, Vec<String>>,
    observed_bytes: u64,
}
impl VerifiedInventory {
    pub fn entries(&self) -> &BTreeMap<String, InventoryEntry> {
        &self.entries
    }
    pub fn alias_groups(&self) -> &BTreeMap<String, Vec<String>> {
        &self.alias_groups
    }
    pub fn parameter_digest(&self) -> &ContentDigest {
        &self.parameter_digest
    }
    pub fn buffer_digest(&self) -> &ContentDigest {
        &self.buffer_digest
    }
    pub fn source_manifest_digest(&self) -> &ContentDigest {
        &self.source_manifest_digest
    }
    pub fn observed_bytes(&self) -> u64 {
        self.observed_bytes
    }
}

#[derive(Serialize)]
struct Identity<'a> {
    schema_version: u32,
    kind: &'static str,
    tensors: Vec<IdentityTensor<'a>>,
}
#[derive(Serialize)]
struct IdentityTensor<'a> {
    name: &'a str,
    shape: &'a [u64],
    dtype: DType,
    content_digest: &'a ContentDigest,
}

/// Versioned tensor-set identity; storage aliases and runtime addresses are
/// verified separately in the full model manifest and inventory.
pub fn tensor_set_digest(tensors: &BTreeMap<String, TensorSpec>) -> Result<ContentDigest> {
    for spec in tensors.values() {
        spec.byte_len()?;
    }
    let identity = Identity {
        schema_version: 1,
        kind: "tensor_set",
        tensors: tensors
            .iter()
            .map(|(name, spec)| IdentityTensor {
                name,
                shape: &spec.shape,
                dtype: spec.dtype,
                content_digest: &spec.digest,
            })
            .collect(),
    };
    let bytes = serde_json::to_vec(&identity)
        .map_err(|error| failure(ErrorCode::InvalidValue, "", error.to_string()))?;
    Ok(CanonicalDocument::parse(&bytes, Limits::default())?.digest())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Span {
    start: usize,
    end: usize,
    empty_tag: usize,
}

pub fn verify_inventory(
    manifest: &Validated<ModelManifest>,
    observations: &[TensorObservation<'_>],
    limits: InventoryLimits,
) -> Result<VerifiedInventory> {
    if limits.max_tensors == 0 || limits.max_observed_bytes == 0 {
        return Err(failure(
            ErrorCode::InvalidValue,
            "",
            "positive inventory limits are required",
        ));
    }
    if manifest.format != TensorFormat::DenseSafetensors || manifest.packing_policy.is_some() {
        return Err(failure(
            ErrorCode::UnsupportedCapability,
            "/format",
            "CPU registration does not certify packed/quantized storage",
        ));
    }
    if observations.len() > limits.max_tensors {
        return Err(failure(
            ErrorCode::LimitExceeded,
            "",
            "inventory tensor count exceeded",
        ));
    }
    let expected_count = manifest
        .parameters
        .len()
        .checked_add(manifest.buffers.len())
        .ok_or_else(|| failure(ErrorCode::LimitExceeded, "", "inventory count overflow"))?;
    if observations.len() != expected_count {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "",
            "observed parameter/buffer set is incomplete",
        ));
    }
    let mut entries = BTreeMap::new();
    let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut group_spans: BTreeMap<&str, Span> = BTreeMap::new();
    let mut span_groups: BTreeMap<Span, &str> = BTreeMap::new();
    let mut spans = Vec::new();
    let mut names = BTreeSet::new();
    let mut observed_bytes = 0u64;
    for (index, observation) in observations.iter().enumerate() {
        let path = child_path("/tensors", observation.name);
        if !names.insert(observation.name) {
            return Err(failure(
                ErrorCode::DuplicateKey,
                &path,
                "duplicate observed tensor",
            ));
        }
        let declared = match observation.role {
            TensorRole::Parameter => manifest.parameters.get(observation.name),
            TensorRole::Buffer => manifest.buffers.get(observation.name),
        }
        .ok_or_else(|| {
            failure(
                ErrorCode::IntegrityMismatch,
                &path,
                "unexpected tensor name or role",
            )
        })?;
        if declared.shape != observation.shape
            || declared.dtype != observation.dtype
            || declared.byte_len()? != observation.bytes.len() as u64
        {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                &path,
                "observed shape/dtype/byte length differs",
            ));
        }
        observed_bytes = observed_bytes
            .checked_add(observation.bytes.len() as u64)
            .filter(|bytes| *bytes <= limits.max_observed_bytes)
            .ok_or_else(|| {
                failure(
                    ErrorCode::LimitExceeded,
                    &path,
                    "inventory read budget exceeded",
                )
            })?;
        if ContentDigest::of_bytes(observation.bytes) != declared.digest {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                &path,
                "observed tensor content differs",
            ));
        }
        let start = observation.bytes.as_ptr() as usize;
        let end = start
            .checked_add(observation.bytes.len())
            .ok_or_else(|| failure(ErrorCode::LimitExceeded, &path, "storage span overflow"))?;
        // Empty allocations often share a dangling address. They cannot prove
        // aliasing, so each empty observation gets a distinct ephemeral span.
        let span = Span {
            start,
            end,
            empty_tag: if observation.bytes.is_empty() {
                index + 1
            } else {
                0
            },
        };
        if group_spans
            .get(declared.storage_group.as_str())
            .is_some_and(|known| *known != span)
        {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                &path,
                "declared aliases do not share actual storage",
            ));
        }
        if span_groups
            .get(&span)
            .is_some_and(|known| *known != declared.storage_group)
        {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                &path,
                "undeclared shared storage",
            ));
        }
        group_spans.insert(&declared.storage_group, span);
        span_groups.insert(span, &declared.storage_group);
        if !observation.bytes.is_empty() {
            spans.push(span);
        }
        groups
            .entry(declared.storage_group.clone())
            .or_default()
            .push(observation.name.to_owned());
        entries.insert(
            observation.name.to_owned(),
            InventoryEntry {
                role: observation.role,
                shape: observation.shape.to_vec(),
                dtype: observation.dtype,
                device: "cpu".to_owned(),
                content_digest: declared.digest.clone(),
                storage_group: declared.storage_group.clone(),
            },
        );
    }
    spans.sort_unstable();
    for pair in spans.windows(2) {
        if pair[1].start < pair[0].end && pair[0] != pair[1] {
            return Err(failure(
                ErrorCode::UnsupportedCapability,
                "",
                "partially overlapping tensor views require a qualified layout adapter",
            ));
        }
    }
    for members in groups.values_mut() {
        members.sort();
    }
    let parameter_digest = tensor_set_digest(&manifest.parameters)?;
    if parameter_digest != manifest.parameter_digest {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "/parameter_digest",
            "parameter-set identity differs from manifest",
        ));
    }
    let raw = serde_json::to_vec(&**manifest)
        .map_err(|error| failure(ErrorCode::InvalidValue, "", error.to_string()))?;
    let source_manifest_digest = CanonicalDocument::parse(&raw, Limits::default())?.digest();
    Ok(VerifiedInventory {
        source_manifest_digest,
        parameter_digest,
        buffer_digest: tensor_set_digest(&manifest.buffers)?,
        entries,
        alias_groups: groups,
        observed_bytes,
    })
}
