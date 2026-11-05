//! Explicit local gym source intake: bytes and declarations, never imports.
mod files;
pub mod schema;

use nf_checkpoint::store::ArtifactStore;
use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits as JsonLimits};
use nf_core::error::{CoreError, ErrorCode, Result};
use nf_core::schemas::AccessClass;
use schema::Schema;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

const MAX_FILE: usize = 1_048_576;
const MAX_TOTAL: usize = 8_388_608;
const MAX_FILES: usize = 64;
fn fail(code: ErrorCode, message: &str) -> CoreError {
    CoreError::new(code, "/gym_intake", message)
}
pub(super) fn label(value: &str) -> Result<()> {
    if value.is_empty() || value.len() > 256 || value.chars().any(char::is_control) {
        return Err(fail(
            ErrorCode::InvalidValue,
            "bounded non-control label required",
        ));
    }
    Ok(())
}
pub(super) fn relative(value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 256
        || value.split('/').count() > 8
        || value.contains(['\\', ':', '\0'])
        || value.chars().any(char::is_control)
        || value
            .split('/')
            .any(|p| p.is_empty() || p == "." || p == ".." || p.len() > 120)
    {
        return Err(fail(
            ErrorCode::ScopeViolation,
            "normalized bounded relative asset path required",
        ));
    }
    Ok(())
}
fn canonical(value: &impl Serialize) -> Result<CanonicalDocument> {
    let bytes = serde_json::to_vec(value)
        .map_err(|_| fail(ErrorCode::InvalidJson, "record encoding failed"))?;
    CanonicalDocument::parse(
        &bytes,
        JsonLimits {
            max_bytes: MAX_FILE,
            max_depth: 48,
            max_nodes: 20000,
        },
    )
}
fn decode<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    let doc = CanonicalDocument::parse(
        bytes,
        JsonLimits {
            max_bytes: MAX_FILE,
            max_depth: 48,
            max_nodes: 20000,
        },
    )?;
    serde_json::from_value(doc.value().clone())
        .map_err(|e| fail(ErrorCode::InvalidValue, &e.to_string()))
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssetKind {
    Code,
    Data,
    Config,
    Manifest,
    DependencyLock,
    License,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Selection {
    pub kind: AssetKind,
    pub expected: Option<ContentDigest>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Asset {
    pub kind: AssetKind,
    pub digest: ContentDigest,
    pub bytes: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EntryPoint {
    pub file: String,
    pub symbol: String,
}
impl EntryPoint {
    fn validate(&self, draft: &Draft) -> Result<()> {
        draft.reference(&self.file, AssetKind::Code)?;
        if !self.file.ends_with(".py")
            || self.symbol.is_empty()
            || self.symbol.len() > 128
            || self.symbol.split('.').any(|p| {
                p.is_empty()
                    || !p.bytes().enumerate().all(|(i, b)| {
                        b == b'_' || b.is_ascii_alphabetic() || (i > 0 && b.is_ascii_digit())
                    })
            })
        {
            return Err(fail(
                ErrorCode::InvalidValue,
                "declared Python file/symbol required; no dynamic discovery",
            ));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Interface {
    SingleShot {
        evaluate: EntryPoint,
        score: EntryPoint,
        oracle: EntryPoint,
        input: Schema,
        output: Schema,
    },
    Episode {
        factory: EntryPoint,
        score: EntryPoint,
        oracle: EntryPoint,
        reset: Schema,
        observation: Schema,
        action: Schema,
        result: Schema,
        max_steps: u32,
        max_episode_ms: u64,
    },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Dataset {
    Frozen {
        files: BTreeSet<String>,
        task_key: String,
        group_key: String,
    },
    Generator {
        factory: EntryPoint,
        config: String,
        seeds: Vec<u64>,
        task_key: String,
        group_key: String,
    },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Draft {
    pub schema_version: u32,
    pub name: String,
    pub version: String,
    pub usage_rights: String,
    pub confidentiality: AccessClass,
    pub assets: BTreeMap<String, Selection>,
    pub dependency_lock: String,
    pub licenses: BTreeSet<String>,
    pub interface: Interface,
    pub dataset: Dataset,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Dependency {
    pub version: String,
    pub integrity: ContentDigest,
    pub license: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DependencyLock {
    pub schema_version: u32,
    pub image: ContentDigest,
    pub packages: BTreeMap<String, Dependency>,
}
impl Draft {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let value: Self = decode(bytes)?;
        value.validate()?;
        Ok(value)
    }
    fn reference(&self, path: &str, kind: AssetKind) -> Result<()> {
        relative(path)?;
        if self.assets.get(path).is_none_or(|a| a.kind != kind) {
            return Err(fail(
                ErrorCode::ScopeViolation,
                "reference must name a selected asset with exact role",
            ));
        }
        Ok(())
    }
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != 1 {
            return Err(fail(
                ErrorCode::UnsupportedVersion,
                "gym intake schema version must be1",
            ));
        }
        for s in [&self.name, &self.version, &self.usage_rights] {
            label(s)?;
        }
        if self.assets.is_empty() || self.assets.len() > MAX_FILES || self.licenses.is_empty() {
            return Err(fail(
                ErrorCode::LimitExceeded,
                "bounded explicit assets and actual licenses required",
            ));
        }
        for path in self.assets.keys() {
            relative(path)?;
        }
        self.reference(&self.dependency_lock, AssetKind::DependencyLock)?;
        for path in &self.licenses {
            self.reference(path, AssetKind::License)?;
        }
        match &self.interface {
            Interface::SingleShot {
                evaluate,
                score,
                oracle,
                input,
                output,
            } => {
                for e in [evaluate, score, oracle] {
                    e.validate(self)?;
                }
                input.validate()?;
                output.validate()?;
            }
            Interface::Episode {
                factory,
                score,
                oracle,
                reset,
                observation,
                action,
                result,
                max_steps,
                max_episode_ms,
            } => {
                for e in [factory, score, oracle] {
                    e.validate(self)?;
                }
                for s in [reset, observation, action, result] {
                    s.validate()?;
                }
                if *max_steps == 0
                    || *max_steps > 10000
                    || *max_episode_ms == 0
                    || *max_episode_ms > 3600000
                {
                    return Err(fail(
                        ErrorCode::LimitExceeded,
                        "finite episode termination limits required",
                    ));
                }
            }
        }
        match &self.dataset {
            Dataset::Frozen {
                files,
                task_key,
                group_key,
            } => {
                if files.is_empty() || files.len() > MAX_FILES {
                    return Err(fail(
                        ErrorCode::LimitExceeded,
                        "explicit nonempty dataset selection required",
                    ));
                }
                for path in files {
                    self.reference(path, AssetKind::Data)?;
                }
                label(task_key)?;
                label(group_key)?;
                if task_key == group_key {
                    return Err(fail(
                        ErrorCode::InvalidValue,
                        "distinct task/group identity fields required",
                    ));
                }
            }
            Dataset::Generator {
                factory,
                config,
                seeds,
                task_key,
                group_key,
            } => {
                factory.validate(self)?;
                self.reference(config, AssetKind::Config)?;
                label(task_key)?;
                label(group_key)?;
                if seeds.is_empty()
                    || seeds.len() > 64
                    || seeds.iter().collect::<BTreeSet<_>>().len() != seeds.len()
                    || task_key == group_key
                {
                    return Err(fail(
                        ErrorCode::InvalidValue,
                        "unique bounded seeds and task/group keys required",
                    ));
                }
            }
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema_version: u32,
    pub draft: Draft,
    pub assets: BTreeMap<String, Asset>,
    pub dependency: DependencyLock,
    pub code_identity: ContentDigest,
    pub data_identity: ContentDigest,
    pub status: Status,
    pub trust: Trust,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    SnapshottedNotValidated,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Trust {
    UntrustedLocalSource,
}
/// A verified persisted snapshot, NOT validation, approval or an execution grant.
pub struct Snapshot {
    manifest: Manifest,
    id: ContentDigest,
}
impl Snapshot {
    pub fn id(&self) -> &ContentDigest {
        &self.id
    }
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    pub fn bytes(&self) -> Result<Vec<u8>> {
        Ok(canonical(&self.manifest)?.bytes().to_vec())
    }
    pub fn asset(&self, store: &ArtifactStore, path: &str) -> Result<Vec<u8>> {
        relative(path)?;
        let asset = self
            .manifest
            .assets
            .get(path)
            .ok_or_else(|| fail(ErrorCode::ScopeViolation, "asset not in snapshot scope"))?;
        let bytes = store.read_bounded(&asset.digest, MAX_FILE as u64)?;
        if bytes.len() as u64 != asset.bytes {
            return Err(fail(ErrorCode::IntegrityMismatch, "asset size mismatch"));
        }
        Ok(bytes)
    }
}
fn group(assets: &BTreeMap<String, Asset>, kind: AssetKind) -> Result<ContentDigest> {
    let selected: BTreeMap<_, _> = assets.iter().filter(|(_, a)| a.kind == kind).collect();
    Ok(canonical(&selected)?.digest())
}
fn dataset_identity(draft: &Draft, assets: &BTreeMap<String, Asset>) -> Result<ContentDigest> {
    let selected: BTreeMap<_, _> = assets
        .iter()
        .filter(|(_, asset)| matches!(asset.kind, AssetKind::Data | AssetKind::Config))
        .collect();
    let generator = match &draft.dataset {
        Dataset::Generator { .. } => Some(
            assets
                .iter()
                .filter(|(_, asset)| {
                    matches!(asset.kind, AssetKind::Code | AssetKind::DependencyLock)
                })
                .collect::<BTreeMap<_, _>>(),
        ),
        Dataset::Frozen { .. } => None,
    };
    Ok(canonical(&(&draft.dataset, selected, generator))?.digest())
}
fn verify_content(
    draft: &Draft,
    assets: &BTreeMap<String, Asset>,
    contents: &BTreeMap<String, Vec<u8>>,
) -> Result<DependencyLock> {
    if assets.len() != draft.assets.len() {
        return Err(fail(
            ErrorCode::IntegrityMismatch,
            "snapshot selection mismatch",
        ));
    }
    let mut total = 0_usize;
    for (name, selected) in &draft.assets {
        let asset = assets
            .get(name)
            .ok_or_else(|| fail(ErrorCode::IntegrityMismatch, "missing selected asset"))?;
        let bytes = contents
            .get(name)
            .ok_or_else(|| fail(ErrorCode::IntegrityMismatch, "missing selected bytes"))?;
        total = total
            .checked_add(bytes.len())
            .ok_or_else(|| fail(ErrorCode::LimitExceeded, "asset byte overflow"))?;
        if bytes.len() > MAX_FILE
            || total > MAX_TOTAL
            || asset.bytes != bytes.len() as u64
            || asset.kind != selected.kind
            || asset.digest != ContentDigest::of_bytes(bytes)
            || selected
                .expected
                .as_ref()
                .is_some_and(|expected| *expected != asset.digest)
        {
            return Err(fail(
                ErrorCode::IntegrityMismatch,
                "asset identity, selection or byte limit mismatch",
            ));
        }
        if matches!(
            asset.kind,
            AssetKind::Code
                | AssetKind::License
                | AssetKind::DependencyLock
                | AssetKind::Manifest
                | AssetKind::Config
        ) {
            let text = std::str::from_utf8(bytes)
                .map_err(|_| fail(ErrorCode::InvalidValue, "text asset must be UTF8"))?;
            if text.trim().is_empty() || text.contains('\0') {
                return Err(fail(
                    ErrorCode::InvalidValue,
                    "nonempty NUL-free text assets required",
                ));
            }
        }
        if matches!(asset.kind, AssetKind::Config | AssetKind::Manifest) {
            CanonicalDocument::parse(
                bytes,
                JsonLimits {
                    max_bytes: MAX_FILE,
                    max_depth: 48,
                    max_nodes: 20000,
                },
            )?;
        }
    }
    let lock: DependencyLock = decode(&contents[&draft.dependency_lock])?;
    if lock.schema_version != 1 || lock.packages.len() > 128 {
        return Err(fail(
            ErrorCode::InvalidValue,
            "bounded version1 dependency lock required",
        ));
    }
    for (name, package) in &lock.packages {
        label(name)?;
        label(&package.version)?;
        if !package
            .version
            .strip_prefix('v')
            .unwrap_or(&package.version)
            .bytes()
            .next()
            .is_some_and(|b| b.is_ascii_digit())
            || !package
                .version
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_' | b'+'))
        {
            return Err(fail(
                ErrorCode::InvalidValue,
                "dependencies must use exact versions, not ranges or commands",
            ));
        }
        draft.reference(&package.license, AssetKind::License)?;
        if !draft.licenses.contains(&package.license) {
            return Err(fail(
                ErrorCode::InvalidValue,
                "dependency license must be explicitly retained",
            ));
        }
    }
    Ok(lock)
}
/// Read only selected files; publish the root manifest only after source fences.
pub fn snapshot(source: &Path, store: &ArtifactStore, draft: Draft) -> Result<Snapshot> {
    draft.validate()?;
    let root = files::Root::open(source)?;
    let mut observed = Vec::new();
    let mut total = 0_usize;
    for (name, selected) in &draft.assets {
        let file = root.observe(name, selected.kind)?;
        total += file.data.len();
        if total > MAX_TOTAL {
            return Err(fail(
                ErrorCode::LimitExceeded,
                "total snapshot byte ceiling exceeded",
            ));
        }
        observed.push(file);
    }
    let assets = observed
        .iter()
        .map(|f| (f.name.clone(), f.asset.clone()))
        .collect();
    let contents = observed
        .iter()
        .map(|f| (f.name.clone(), f.data.clone()))
        .collect();
    let dependency = verify_content(&draft, &assets, &contents)?;
    for file in &observed {
        let actual = store.put(file.data.as_slice())?;
        if actual.digest != file.asset.digest {
            return Err(fail(ErrorCode::IntegrityMismatch, "CAS asset mismatch"));
        }
    }
    root.same_root(source)?;
    for file in &observed {
        root.fence(file)?;
    }
    root.same_root(source)?;
    let manifest = Manifest {
        schema_version: 1,
        code_identity: group(&assets, AssetKind::Code)?,
        data_identity: dataset_identity(&draft, &assets)?,
        draft,
        assets,
        dependency,
        status: Status::SnapshottedNotValidated,
        trust: Trust::UntrustedLocalSource,
    };
    let document = canonical(&manifest)?;
    let receipt = store.put(document.bytes())?;
    if receipt.digest != document.digest() {
        return Err(fail(
            ErrorCode::IntegrityMismatch,
            "manifest publication mismatch",
        ));
    }
    Ok(Snapshot {
        manifest,
        id: receipt.digest,
    })
}
/// Restore only after checking every referenced immutable byte identity.
pub fn restore(store: &ArtifactStore, id: &ContentDigest) -> Result<Snapshot> {
    let bytes = store.read_bounded(id, MAX_FILE as u64)?;
    let manifest: Manifest = decode(&bytes)?;
    manifest.draft.validate()?;
    if manifest.schema_version != 1 || canonical(&manifest)?.digest() != *id {
        return Err(fail(
            ErrorCode::IntegrityMismatch,
            "noncanonical or invalid snapshot manifest",
        ));
    }
    let mut contents = BTreeMap::new();
    let mut total = 0;
    if manifest.assets.len() > MAX_FILES {
        return Err(fail(ErrorCode::LimitExceeded, "asset count exceeded"));
    }
    for (name, asset) in &manifest.assets {
        relative(name)?;
        let bytes = store.read_bounded(&asset.digest, MAX_FILE as u64)?;
        total += bytes.len();
        if total > MAX_TOTAL {
            return Err(fail(
                ErrorCode::LimitExceeded,
                "restored snapshot byte ceiling exceeded",
            ));
        }
        contents.insert(name.clone(), bytes);
    }
    let lock = verify_content(&manifest.draft, &manifest.assets, &contents)?;
    if canonical(&lock)?.digest() != canonical(&manifest.dependency)?.digest()
        || manifest.code_identity != group(&manifest.assets, AssetKind::Code)?
        || manifest.data_identity != dataset_identity(&manifest.draft, &manifest.assets)?
    {
        return Err(fail(
            ErrorCode::IntegrityMismatch,
            "snapshot aggregate identity mismatch",
        ));
    }
    Ok(Snapshot {
        manifest,
        id: id.clone(),
    })
}
