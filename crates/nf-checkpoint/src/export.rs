//! Standalone immutable checkpoint bundles. Publication never changes an active pointer.
use crate::store::ArtifactStore;
use crate::tensor_file::{inspect_safetensors, TensorFileLimits};
use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use nf_core::error::{CoreError, ErrorCode, Result};
use nf_core::schemas::{ModelManifest, TensorFormat, Validate, Validated};
use nf_graph::inventory::{
    verify_inventory, InventoryLimits, TensorObservation, TensorRole, VerifiedInventory,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;

fn failure(code: ErrorCode, message: &str) -> CoreError {
    CoreError::new(code, "", message)
}
fn canonical<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let raw = serde_json::to_vec(value)
        .map_err(|_| failure(ErrorCode::InvalidValue, "export serialization failed"))?;
    Ok(CanonicalDocument::parse(&raw, Limits::default())?
        .bytes()
        .to_vec())
}
#[derive(Debug, Clone, Copy)]
pub struct ExportLimits {
    pub max_checkpoint_bytes: usize,
    pub max_asset_bytes: u64,
    pub max_total_bytes: u64,
    pub max_tensors: usize,
}
impl Default for ExportLimits {
    fn default() -> Self {
        Self {
            max_checkpoint_bytes: 67_108_864,
            max_asset_bytes: 4_194_304,
            max_total_bytes: 73_400_320,
            max_tensors: 1024,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExportBlob {
    pub digest: ContentDigest,
    pub bytes: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExportManifest {
    pub schema_version: u32,
    pub kind: String,
    pub alias_contract: String,
    pub model: ModelManifest,
    pub checkpoint: ExportBlob,
    pub assets: BTreeMap<String, ExportBlob>,
}
#[derive(Debug)]
pub struct LoadedExport {
    manifest: ExportManifest,
    model: Validated<ModelManifest>,
    checkpoint: Arc<[u8]>,
    assets: BTreeMap<String, Arc<[u8]>>,
    ranges: BTreeMap<String, Range<usize>>,
    inventory: VerifiedInventory,
}
impl LoadedExport {
    pub fn manifest(&self) -> &ExportManifest {
        &self.manifest
    }
    pub fn model(&self) -> &Validated<ModelManifest> {
        &self.model
    }
    pub fn checkpoint(&self) -> &Arc<[u8]> {
        &self.checkpoint
    }
    pub fn assets(&self) -> &BTreeMap<String, Arc<[u8]>> {
        &self.assets
    }
    /// Verified logical inventory, including reconstructed shared-storage groups.
    /// This is byte/inventory evidence, not fresh-runtime or acceptance evidence.
    pub fn inventory(&self) -> &VerifiedInventory {
        &self.inventory
    }
    pub fn manifest_digest(&self) -> &ContentDigest {
        self.inventory.source_manifest_digest()
    }
    /// Consume the verified bundle for a runtime adapter without rereading files.
    /// Runtime-specific factory/config qualification remains the adapter's job.
    pub fn into_parts(self) -> (Validated<ModelManifest>, Arc<[u8]>, ExportAssets) {
        (self.model, self.checkpoint, self.assets)
    }
    /// Members of a storage group reference the very same immutable byte range.
    pub fn tensor_bytes(&self, name: &str) -> Option<&[u8]> {
        self.ranges
            .get(name)
            .map(|range| &self.checkpoint[range.clone()])
    }
}
/// Immutable config/tokenizer/other asset bytes passed to a runtime adapter.
pub type ExportAssets = BTreeMap<String, Arc<[u8]>>;

fn validate_limits(limits: ExportLimits) -> Result<()> {
    if limits.max_checkpoint_bytes == 0
        || limits.max_asset_bytes == 0
        || limits.max_total_bytes == 0
        || limits.max_tensors == 0
        || limits.max_tensors > 1024
    {
        return Err(failure(
            ErrorCode::InvalidValue,
            "positive bounded export limits required",
        ));
    }
    Ok(())
}
fn validate_header(
    manifest: &ExportManifest,
    limits: ExportLimits,
) -> Result<Validated<ModelManifest>> {
    validate_limits(limits)?;
    if manifest.schema_version != 1
        || manifest.kind != "standalone_dense_checkpoint"
        || manifest.alias_contract != "canonical_group_shared_storage_v1"
    {
        return Err(failure(
            ErrorCode::UnsupportedVersion,
            "unknown export version/class/alias contract",
        ));
    }
    let model = Validated::new(manifest.model.clone())?;
    if model.format != TensorFormat::DenseSafetensors || model.packing_policy.is_some() {
        return Err(failure(
            ErrorCode::UnsupportedCapability,
            "export requires qualified dense storage",
        ));
    }
    if model.parameters.len() + model.buffers.len() > limits.max_tensors
        || manifest.assets.len() > 1024
    {
        return Err(failure(
            ErrorCode::LimitExceeded,
            "export metadata bound exceeded",
        ));
    }
    if model.assets.get("checkpoint") != Some(&manifest.checkpoint.digest) {
        return Err(failure(
            ErrorCode::SourceMismatch,
            "checkpoint binding missing or different",
        ));
    }
    let mut expected = model.assets.clone();
    expected.remove("checkpoint");
    if expected
        .insert("config".to_owned(), model.architecture.config.clone())
        .is_some_and(|old| old != model.architecture.config)
    {
        return Err(failure(
            ErrorCode::SourceMismatch,
            "configuration bindings differ",
        ));
    }
    if manifest
        .assets
        .iter()
        .map(|(name, blob)| (name.clone(), blob.digest.clone()))
        .collect::<BTreeMap<_, _>>()
        != expected
    {
        return Err(failure(
            ErrorCode::SourceMismatch,
            "export asset inventory incomplete or unbound",
        ));
    }
    let assets = manifest.assets.values().try_fold(0u64, |sum, blob| {
        sum.checked_add(blob.bytes)
            .ok_or_else(|| failure(ErrorCode::LimitExceeded, "asset size overflow"))
    })?;
    let total = assets
        .checked_add(manifest.checkpoint.bytes)
        .and_then(|n| n.checked_add(canonical(manifest).ok()?.len() as u64))
        .ok_or_else(|| failure(ErrorCode::LimitExceeded, "export size overflow"))?;
    if manifest.checkpoint.bytes > limits.max_checkpoint_bytes as u64
        || assets > limits.max_asset_bytes
        || total > limits.max_total_bytes
    {
        return Err(failure(
            ErrorCode::LimitExceeded,
            "export bytes exceed limits",
        ));
    }
    Ok(model)
}
fn verify_bytes(
    manifest: ExportManifest,
    checkpoint: Arc<[u8]>,
    assets: BTreeMap<String, Arc<[u8]>>,
    limits: ExportLimits,
) -> Result<LoadedExport> {
    let model = validate_header(&manifest, limits)?;
    if checkpoint.len() as u64 != manifest.checkpoint.bytes
        || ContentDigest::of_bytes(&checkpoint) != manifest.checkpoint.digest
    {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "export checkpoint bytes differ",
        ));
    }
    if assets.len() != manifest.assets.len() {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "export assets incomplete",
        ));
    }
    for (name, blob) in &manifest.assets {
        let bytes = assets
            .get(name)
            .ok_or_else(|| failure(ErrorCode::IntegrityMismatch, "export asset missing"))?;
        if bytes.len() as u64 != blob.bytes || ContentDigest::of_bytes(bytes) != blob.digest {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "export asset bytes differ",
            ));
        }
    }
    CanonicalDocument::parse(&assets["config"], Limits::default())?;
    let file = inspect_safetensors(
        &checkpoint,
        TensorFileLimits {
            max_file_bytes: limits.max_checkpoint_bytes,
            max_tensors: limits.max_tensors,
            ..TensorFileLimits::default()
        },
    )?;
    let mut groups = BTreeMap::new();
    for (name, tensor) in &file.tensors {
        let spec = model
            .parameters
            .get(name)
            .or_else(|| model.buffers.get(name))
            .ok_or_else(|| failure(ErrorCode::SourceMismatch, "undeclared exported tensor"))?;
        if tensor.shape != spec.shape || tensor.dtype != spec.dtype || tensor.digest != spec.digest
        {
            return Err(failure(
                ErrorCode::SourceMismatch,
                "export tensor metadata/content differs",
            ));
        }
        // Multiple serialized aliases may be present, but must have matching bytes.
        groups
            .entry(spec.storage_group.clone())
            .or_insert_with(|| tensor.data_range.clone());
    }
    let mut ranges = BTreeMap::new();
    let mut observations = Vec::new();
    for (name, spec) in model.parameters.iter().chain(&model.buffers) {
        let range = groups
            .get(&spec.storage_group)
            .ok_or_else(|| failure(ErrorCode::SourceMismatch, "export storage group missing"))?;
        ranges.insert(name.clone(), range.clone());
        observations.push(TensorObservation {
            name,
            role: if model.parameters.contains_key(name) {
                TensorRole::Parameter
            } else {
                TensorRole::Buffer
            },
            dtype: spec.dtype,
            shape: &spec.shape,
            bytes: &checkpoint[range.clone()],
        });
    }
    let inventory = verify_inventory(
        &model,
        &observations,
        InventoryLimits {
            max_tensors: limits.max_tensors,
            max_observed_bytes: limits.max_total_bytes,
        },
    )?;
    Ok(LoadedExport {
        manifest,
        model,
        checkpoint,
        assets,
        ranges,
        inventory,
    })
}
/// All bytes are checked before writing. The root manifest is committed last;
/// incomplete writes can leave bounded orphan blobs but never a complete root.
pub fn publish_export(
    store: &ArtifactStore,
    model: &Validated<ModelManifest>,
    checkpoint: Arc<[u8]>,
    assets: BTreeMap<String, Arc<[u8]>>,
    limits: ExportLimits,
) -> Result<ContentDigest> {
    model.validate()?;
    let asset_bytes = assets.values().try_fold(0u64, |sum, bytes| {
        sum.checked_add(bytes.len() as u64)
            .ok_or_else(|| failure(ErrorCode::LimitExceeded, "export input size overflow"))
    })?;
    if checkpoint.len() > limits.max_checkpoint_bytes
        || assets.len() > 1024
        || asset_bytes > limits.max_asset_bytes
        || asset_bytes
            .checked_add(checkpoint.len() as u64)
            .is_none_or(|total| total > limits.max_total_bytes)
    {
        return Err(failure(
            ErrorCode::LimitExceeded,
            "export input bytes exceed limits before hashing",
        ));
    }
    let manifest = ExportManifest {
        schema_version: 1,
        kind: "standalone_dense_checkpoint".to_owned(),
        alias_contract: "canonical_group_shared_storage_v1".to_owned(),
        model: (**model).clone(),
        checkpoint: ExportBlob {
            digest: ContentDigest::of_bytes(&checkpoint),
            bytes: checkpoint.len() as u64,
        },
        assets: assets
            .iter()
            .map(|(name, bytes)| {
                (
                    name.clone(),
                    ExportBlob {
                        digest: ContentDigest::of_bytes(bytes),
                        bytes: bytes.len() as u64,
                    },
                )
            })
            .collect(),
    };
    let verified = verify_bytes(manifest, checkpoint, assets, limits)?;
    store.put(verified.checkpoint.as_ref())?;
    for bytes in verified.assets.values() {
        store.put(bytes.as_ref())?;
    }
    let root = canonical(&verified.manifest)?;
    Ok(store.put(root.as_slice())?.digest)
}
/// Restore is a fresh verified read of an explicit immutable root. There is no
/// implicit latest/accepted reference, nor deletion of prior export history.
pub fn restore_export(
    store: &ArtifactStore,
    root: &ContentDigest,
    limits: ExportLimits,
) -> Result<LoadedExport> {
    restore_export_with_reader(root, None, limits, |digest, bound| {
        store.read_bounded(digest, bound)
    })
}

/// Restore from an explicit root using a bounded blob reader. The callback must
/// honor `bound` before allocating; this API independently checks returned lengths
/// and digests, including the root, so a reader is not an integrity authority.
/// Filesystem adapters must additionally enforce safe, capability-rooted reads
/// (use `ArtifactStore::read_bounded`, not caller-controlled relative paths).
///
/// Pass a model identity frozen independently before export to reject substitution
/// before reading any checkpoint/assets. `None` checks bundle self-consistency
/// only. No live candidate, pointer update, acceptance or runtime qualification is
/// involved. All adapters share the exact same native restore invariants.
pub fn restore_export_with_reader<F>(
    root: &ContentDigest,
    expected_model: Option<&ContentDigest>,
    limits: ExportLimits,
    mut read_blob: F,
) -> Result<LoadedExport>
where
    F: FnMut(&ContentDigest, u64) -> Result<Vec<u8>>,
{
    validate_limits(limits)?;
    let root_bound = (Limits::default().max_bytes as u64).min(limits.max_total_bytes);
    let root_bytes = read_blob(root, root_bound)?;
    if root_bytes.len() as u64 > root_bound {
        return Err(failure(
            ErrorCode::LimitExceeded,
            "export root exceeds read bound",
        ));
    }
    if ContentDigest::of_bytes(&root_bytes) != *root {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "export root bytes differ",
        ));
    }
    let document = CanonicalDocument::parse(&root_bytes, Limits::default())?;
    let manifest: ExportManifest =
        serde_json::from_value(document.value().clone()).map_err(|error| {
            failure(
                ErrorCode::InvalidValue,
                &format!("invalid export manifest: {error}"),
            )
        })?;
    let model = validate_header(&manifest, limits)?;
    let model_digest = ContentDigest::of_bytes(&canonical(&*model)?);
    if expected_model.is_some_and(|expected| *expected != model_digest) {
        return Err(failure(
            ErrorCode::SourceMismatch,
            "frozen model identity differs",
        ));
    }
    // Charge actual root bytes, not its potentially shorter canonical encoding.
    let total = (root_bytes.len() as u64)
        .checked_add(manifest.checkpoint.bytes)
        .and_then(|initial| {
            manifest
                .assets
                .values()
                .try_fold(initial, |sum, blob| sum.checked_add(blob.bytes))
        });
    if total.is_none_or(|total| total > limits.max_total_bytes) {
        return Err(failure(
            ErrorCode::LimitExceeded,
            "export bytes exceed limits",
        ));
    }
    let checkpoint = Arc::from(read_blob(
        &manifest.checkpoint.digest,
        manifest.checkpoint.bytes,
    )?);
    let mut assets = BTreeMap::new();
    for (name, blob) in &manifest.assets {
        assets.insert(
            name.clone(),
            Arc::from(read_blob(&blob.digest, blob.bytes)?),
        );
    }
    verify_bytes(manifest, checkpoint, assets, limits)
}
