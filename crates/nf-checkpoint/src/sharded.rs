//! Linux, no-follow sharded imports. Inspection and copying stream bounded
//! chunks; a checkpoint manifest is emitted only after all sources verify.
//! Complete blobs left by a failed group import are unreferenced, not accepted.

#[cfg(test)]
mod tests;

use crate::store::{open_regular, ArtifactStore};
use crate::tensor_file::{inspect_header, HeaderInventory, TensorFileLimits};
use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use nf_core::error::{CoreError, ErrorCode, Result};
use rustix::fd::OwnedFd;
use rustix::fs::{self, Dir, Mode, OFlags};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, Metadata};
use std::io::{self, Cursor, Read};
use std::os::unix::fs::MetadataExt;
use std::path::{Component, Path};

fn failure(code: ErrorCode, message: impl Into<String>) -> CoreError {
    CoreError::new(code, "", message)
}
fn io_failure(cause: impl std::fmt::Display) -> CoreError {
    failure(ErrorCode::Io, cause.to_string())
}
fn changed() -> CoreError {
    failure(
        ErrorCode::IntegrityMismatch,
        "checkpoint source changed after inspection",
    )
}
fn add(total: &mut u64, count: u64) -> Result<()> {
    *total = total
        .checked_add(count)
        .ok_or_else(|| failure(ErrorCode::LimitExceeded, "I/O counter overflow"))?;
    Ok(())
}
fn finish(hasher: Sha256) -> ContentDigest {
    format!("sha256:{:x}", hasher.finalize())
        .parse()
        .expect("SHA-256 emits a valid digest")
}
fn leaf(name: &str, suffix: &str) -> Result<()> {
    if name.is_empty()
        || name == "."
        || name == ".."
        || !name.ends_with(suffix)
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
    {
        return Err(failure(
            ErrorCode::InvalidValue,
            "checkpoint files must be explicit safe leaf names",
        ));
    }
    Ok(())
}

#[derive(Debug)]
struct SourceRoot(OwnedFd);
impl SourceRoot {
    fn open(path: &Path) -> Result<Self> {
        if !path.is_absolute()
            || path.parent().is_none()
            || path
                .components()
                .any(|c| matches!(c, Component::ParentDir | Component::Prefix(_)))
        {
            return Err(failure(
                ErrorCode::InvalidValue,
                "source root must be an absolute non-traversing directory",
            ));
        }
        let flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
        let mut fd = fs::open("/", flags, Mode::empty()).map_err(io_failure)?;
        for component in path.components() {
            if let Component::Normal(name) = component {
                fd = fs::openat(&fd, name, flags, Mode::empty()).map_err(io_failure)?;
            }
        }
        Ok(Self(fd))
    }
    fn open_file(&self, name: &str) -> Result<File> {
        open_regular(&self.0, name)
    }
    fn check(&self, blob: &SourceBlob) -> Result<()> {
        let current = self.open_file(&blob.name).map_err(|_| changed())?;
        if Stamp::from(&current.metadata().map_err(io_failure)?) != blob.stamp {
            return Err(changed());
        }
        Ok(())
    }
    fn shards(&self, max_entries: usize) -> Result<BTreeSet<String>> {
        let mut shards = BTreeSet::new();
        let mut count = 0;
        for entry in Dir::read_from(&self.0).map_err(io_failure)? {
            let entry = entry.map_err(io_failure)?;
            let name = entry.file_name().to_str().map_err(io_failure)?;
            if name == "." || name == ".." {
                continue;
            }
            count += 1;
            if count > max_entries {
                return Err(failure(
                    ErrorCode::LimitExceeded,
                    "source directory entry bound exceeded",
                ));
            }
            if name.ends_with(".safetensors") {
                leaf(name, ".safetensors")?;
                shards.insert(name.to_owned());
            }
        }
        Ok(shards)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Stamp {
    device: u64,
    inode: u64,
    bytes: u64,
    modified: (i64, i64),
    changed: (i64, i64),
}
impl From<&Metadata> for Stamp {
    fn from(value: &Metadata) -> Self {
        Self {
            device: value.dev(),
            inode: value.ino(),
            bytes: value.len(),
            modified: (value.mtime(), value.mtime_nsec()),
            changed: (value.ctime(), value.ctime_nsec()),
        }
    }
}
#[derive(Debug)]
struct SourceBlob {
    name: String,
    stamp: Stamp,
    digest: ContentDigest,
}
#[derive(Debug)]
struct Shard {
    source: SourceBlob,
    header: HeaderInventory,
    tensor_digests: BTreeMap<String, ContentDigest>,
}

#[derive(Debug, Clone, Copy)]
pub struct ShardLimits {
    pub tensor_file: TensorFileLimits,
    pub max_index_bytes: usize,
    pub max_shards: usize,
    pub max_total_header_bytes: u64,
    pub max_total_file_bytes: u64,
    pub max_directory_entries: usize,
    pub chunk_bytes: usize,
}
impl Default for ShardLimits {
    fn default() -> Self {
        Self {
            tensor_file: TensorFileLimits::default(),
            max_index_bytes: 1_048_576,
            max_shards: 64,
            max_total_header_bytes: 8_388_608,
            max_total_file_bytes: 68_719_476_736,
            max_directory_entries: 10_000,
            chunk_bytes: 65_536,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IoStats {
    /// Includes the shared initial inspection; do not sum this across reuse attempts.
    pub source_bytes_read: u64,
    /// Reads performed by this snapshot call, excluding shared inspection.
    pub snapshot_source_bytes_read: u64,
    pub stored_bytes_verified: u64,
    pub staged_bytes: u64,
    pub new_shard_bytes: u64,
    /// Peak payload buffer length, not whole-process RSS or metadata overhead.
    pub payload_buffer_bytes: usize,
    /// ArtifactStore's fixed transfer buffer, separate from inspection buffers.
    pub staging_buffer_bytes: usize,
    pub largest_header_bytes: usize,
    /// Aggregate serialized metadata validated, not an RSS/heap measurement.
    pub validated_header_bytes: u64,
    pub new_shards: usize,
    pub reused_shards: usize,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct IndexMetadata {
    total_size: u64,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Index {
    metadata: IndexMetadata,
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug)]
pub struct ShardedSource {
    root: SourceRoot,
    index: SourceBlob,
    shards: BTreeMap<String, Shard>,
    limits: ShardLimits,
    stats: IoStats,
}

#[derive(Debug, Clone, Serialize)]
pub struct SnapshotManifest {
    pub schema_version: u32,
    pub kind: String,
    pub index: ContentDigest,
    pub shards: BTreeMap<String, ContentDigest>,
}
#[derive(Debug)]
pub struct SnapshotReceipt {
    pub manifest_digest: ContentDigest,
    pub manifest: SnapshotManifest,
    pub stats: IoStats,
}

fn read_index(root: &SourceRoot, name: &str, limit: usize) -> Result<(Index, SourceBlob, usize)> {
    let mut file = root.open_file(name)?;
    let stamp = Stamp::from(&file.metadata().map_err(io_failure)?);
    if stamp.bytes > limit as u64 {
        return Err(failure(
            ErrorCode::LimitExceeded,
            "checkpoint index byte bound exceeded",
        ));
    }
    let mut bytes = Vec::new();
    let bound = (limit as u64)
        .checked_add(1)
        .ok_or_else(|| failure(ErrorCode::InvalidValue, "index bound overflow"))?;
    (&mut file)
        .take(bound)
        .read_to_end(&mut bytes)
        .map_err(io_failure)?;
    if bytes.len() > limit {
        return Err(failure(
            ErrorCode::LimitExceeded,
            "checkpoint index grew beyond its bound",
        ));
    }
    let document = CanonicalDocument::parse(
        &bytes,
        Limits {
            max_bytes: limit,
            ..Limits::default()
        },
    )?;
    let index: Index = serde_json::from_slice(document.bytes())
        .map_err(|cause| failure(ErrorCode::InvalidValue, cause.to_string()))?;
    let blob = SourceBlob {
        name: name.to_owned(),
        stamp,
        digest: ContentDigest::of_bytes(&bytes),
    };
    if Stamp::from(&file.metadata().map_err(io_failure)?) != blob.stamp
        || bytes.len() as u64 != blob.stamp.bytes
    {
        return Err(changed());
    }
    root.check(&blob)?;
    Ok((index, blob, bytes.len()))
}

fn inspect_shard(
    root: &SourceRoot,
    name: &str,
    limits: ShardLimits,
    stats: &mut IoStats,
) -> Result<Shard> {
    let mut file = root.open_file(name)?;
    let stamp = Stamp::from(&file.metadata().map_err(io_failure)?);
    let file_bytes = usize::try_from(stamp.bytes).map_err(|_| {
        failure(
            ErrorCode::LimitExceeded,
            "file size does not fit host index",
        )
    })?;
    if file_bytes > limits.tensor_file.max_file_bytes {
        return Err(failure(
            ErrorCode::LimitExceeded,
            "shard byte bound exceeded",
        ));
    }
    let mut prefix = [0; 8];
    file.read_exact(&mut prefix).map_err(io_failure)?;
    let size = usize::try_from(u64::from_le_bytes(prefix))
        .map_err(|_| failure(ErrorCode::LimitExceeded, "header size overflow"))?;
    if size == 0
        || size > limits.tensor_file.max_header_bytes
        || size.checked_add(8).is_none_or(|end| end > file_bytes)
    {
        return Err(failure(
            ErrorCode::LimitExceeded,
            "shard header exceeds bounds",
        ));
    }
    add(&mut stats.validated_header_bytes, size as u64)?;
    if stats.validated_header_bytes > limits.max_total_header_bytes {
        return Err(failure(
            ErrorCode::LimitExceeded,
            "aggregate header bound exceeded",
        ));
    }
    let mut raw = vec![0; size];
    file.read_exact(&mut raw).map_err(io_failure)?;
    let header = inspect_header(&raw, file_bytes, limits.tensor_file)?;
    stats.largest_header_bytes = stats.largest_header_bytes.max(size);
    stats.payload_buffer_bytes = stats.payload_buffer_bytes.max(limits.chunk_bytes);
    let mut whole = Sha256::new();
    whole.update(prefix);
    whole.update(&raw);
    let mut buffer = vec![0; limits.chunk_bytes];
    let mut tensor_digests = BTreeMap::new();
    for (tensor, layout) in header.payload_order() {
        let mut remaining = layout.data_range.len();
        let mut hasher = Sha256::new();
        while remaining != 0 {
            let count = remaining.min(buffer.len());
            file.read_exact(&mut buffer[..count]).map_err(io_failure)?;
            whole.update(&buffer[..count]);
            hasher.update(&buffer[..count]);
            remaining -= count;
        }
        tensor_digests.insert(tensor.to_owned(), finish(hasher));
    }
    let mut extra = [0];
    if file.read(&mut extra).map_err(io_failure)? != 0
        || Stamp::from(&file.metadata().map_err(io_failure)?) != stamp
    {
        return Err(changed());
    }
    let source = SourceBlob {
        name: name.to_owned(),
        stamp,
        digest: finish(whole),
    };
    root.check(&source)?;
    add(&mut stats.source_bytes_read, source.stamp.bytes)?;
    Ok(Shard {
        source,
        header,
        tensor_digests,
    })
}

pub fn inspect_shards(root: &Path, index_name: &str, limits: ShardLimits) -> Result<ShardedSource> {
    leaf(index_name, ".json")?;
    if limits.chunk_bytes == 0
        || limits.chunk_bytes > 1_048_576
        || limits.max_shards == 0
        || limits.max_index_bytes == 0
        || limits.max_total_header_bytes == 0
        || limits.max_total_file_bytes == 0
        || limits.max_directory_entries == 0
    {
        return Err(failure(
            ErrorCode::InvalidValue,
            "finite positive shard limits are required",
        ));
    }
    let root = SourceRoot::open(root)?;
    let (index, index_source, index_bytes) = read_index(&root, index_name, limits.max_index_bytes)?;
    if index.weight_map.is_empty() {
        return Err(failure(
            ErrorCode::InvalidValue,
            "checkpoint index is empty",
        ));
    }
    let expected: BTreeSet<_> = index.weight_map.values().cloned().collect();
    if expected.len() > limits.max_shards {
        return Err(failure(
            ErrorCode::LimitExceeded,
            "shard count bound exceeded",
        ));
    }
    for name in &expected {
        leaf(name, ".safetensors")?;
    }
    if root.shards(limits.max_directory_entries)? != expected {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "missing or extra checkpoint shards",
        ));
    }
    let mut stats = IoStats {
        source_bytes_read: index_bytes as u64,
        ..IoStats::default()
    };
    let mut shards = BTreeMap::new();
    let mut seen = BTreeSet::new();
    let mut payload_bytes = 0u64;
    let mut file_bytes = 0u64;
    for name in expected {
        let size = root.open_file(&name)?.metadata().map_err(io_failure)?.len();
        add(&mut file_bytes, size)?;
        if file_bytes > limits.max_total_file_bytes {
            return Err(failure(
                ErrorCode::LimitExceeded,
                "checkpoint total byte bound exceeded",
            ));
        }
        let shard = inspect_shard(&root, &name, limits, &mut stats)?;
        if shard.source.stamp.bytes != size {
            return Err(changed());
        }
        add(
            &mut payload_bytes,
            (shard.header.file_bytes - shard.header.data_start) as u64,
        )?;
        for tensor in shard.header.tensors.keys() {
            if !seen.insert(tensor.clone()) || index.weight_map.get(tensor) != Some(&name) {
                return Err(failure(
                    ErrorCode::IntegrityMismatch,
                    "tensor-to-shard mapping mismatch",
                ));
            }
        }
        shards.insert(name, shard);
    }
    if seen.len() != index.weight_map.len() || payload_bytes != index.metadata.total_size {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "index tensor set or total_size mismatch",
        ));
    }
    let source = ShardedSource {
        root,
        index: index_source,
        shards,
        limits,
        stats,
    };
    source.check_sources()?;
    Ok(source)
}

struct VerifiedRead<'a> {
    file: File,
    root: &'a SourceRoot,
    source: &'a SourceBlob,
    hash: Sha256,
    bytes: u64,
    chunk: usize,
    failure: Option<CoreError>,
}
impl Read for VerifiedRead<'_> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if out.is_empty() {
            return Ok(0);
        }
        let bound = out.len().min(self.chunk);
        let count = self.file.read(&mut out[..bound])?;
        self.bytes = self
            .bytes
            .checked_add(count as u64)
            .ok_or_else(|| io::Error::other("copy counter overflow"))?;
        self.hash.update(&out[..count]);
        if count == 0 {
            let check = self.root.check(self.source).and_then(|()| {
                if self.bytes != self.source.stamp.bytes
                    || Stamp::from(&self.file.metadata().map_err(io_failure)?) != self.source.stamp
                    || finish(self.hash.clone()) != self.source.digest
                {
                    Err(changed())
                } else {
                    Ok(())
                }
            });
            if let Err(error) = check {
                self.failure = Some(error);
                return Err(io::Error::other("source changed while copying"));
            }
        } else if self.bytes > self.source.stamp.bytes {
            self.failure = Some(changed());
            return Err(io::Error::other("source grew while copying"));
        }
        Ok(count)
    }
}

impl ShardedSource {
    pub fn inspection_stats(&self) -> &IoStats {
        &self.stats
    }
    pub fn shard_digests(&self) -> BTreeMap<String, ContentDigest> {
        self.shards
            .iter()
            .map(|(name, shard)| (name.clone(), shard.source.digest.clone()))
            .collect()
    }
    pub fn tensor_digests(&self) -> BTreeMap<String, ContentDigest> {
        self.shards
            .values()
            .flat_map(|shard| shard.tensor_digests.clone())
            .collect()
    }
    fn check_sources(&self) -> Result<()> {
        self.root.check(&self.index)?;
        if self.root.shards(self.limits.max_directory_entries)?
            != self.shards.keys().cloned().collect()
        {
            return Err(changed());
        }
        for shard in self.shards.values() {
            self.root.check(&shard.source)?;
        }
        Ok(())
    }
    // Metadata timestamps can have coarse resolution. Rehash source bytes at
    // both snapshot fences, including reused shards, rather than trusting a
    // same-size/same-timestamp rewrite to be unchanged.
    fn verify_sources(&self, stats: &mut IoStats) -> Result<()> {
        self.check_sources()?;
        let mut buffer = vec![0; self.limits.chunk_bytes];
        for source in
            std::iter::once(&self.index).chain(self.shards.values().map(|shard| &shard.source))
        {
            let file = self.root.open_file(&source.name)?;
            let mut reader = VerifiedRead {
                file,
                root: &self.root,
                source,
                hash: Sha256::new(),
                bytes: 0,
                chunk: self.limits.chunk_bytes,
                failure: None,
            };
            loop {
                let result = reader.read(&mut buffer);
                if let Some(error) = reader.failure.take() {
                    return Err(error);
                }
                let count = result.map_err(io_failure)?;
                if count == 0 {
                    break;
                }
                add(&mut stats.source_bytes_read, count as u64)?;
                add(&mut stats.snapshot_source_bytes_read, count as u64)?;
            }
        }
        self.check_sources()
    }

    fn copy_blob(
        &self,
        store: &ArtifactStore,
        source: &SourceBlob,
        stats: &mut IoStats,
    ) -> Result<bool> {
        self.root.check(source)?;
        if store.contains_verified(&source.digest)? {
            add(&mut stats.stored_bytes_verified, source.stamp.bytes)?;
            return Ok(false);
        }
        let file = self.root.open_file(&source.name)?;
        if Stamp::from(&file.metadata().map_err(io_failure)?) != source.stamp {
            return Err(changed());
        }
        let mut reader = VerifiedRead {
            file,
            root: &self.root,
            source,
            hash: Sha256::new(),
            bytes: 0,
            chunk: self.limits.chunk_bytes,
            failure: None,
        };
        let receipt = store.put(&mut reader);
        if let Some(error) = reader.failure {
            return Err(error);
        }
        let receipt = receipt?;
        add(&mut stats.source_bytes_read, reader.bytes)?;
        add(&mut stats.snapshot_source_bytes_read, reader.bytes)?;
        add(&mut stats.staged_bytes, receipt.bytes)?;
        if receipt.digest != source.digest {
            return Err(changed());
        }
        Ok(!receipt.deduplicated)
    }
    pub fn snapshot(&self, store: &ArtifactStore) -> Result<SnapshotReceipt> {
        let mut stats = self.stats.clone();
        stats.staging_buffer_bytes = 65_536;
        self.verify_sources(&mut stats)?;
        self.copy_blob(store, &self.index, &mut stats)?;
        for shard in self.shards.values() {
            if self.copy_blob(store, &shard.source, &mut stats)? {
                stats.new_shards += 1;
                add(&mut stats.new_shard_bytes, shard.source.stamp.bytes)?;
            } else {
                stats.reused_shards += 1;
            }
        }
        self.verify_sources(&mut stats)?;
        let manifest = SnapshotManifest {
            schema_version: 1,
            kind: "sharded_snapshot".to_owned(),
            index: self.index.digest.clone(),
            shards: self.shard_digests(),
        };
        let raw = serde_json::to_vec(&manifest).map_err(io_failure)?;
        let canonical = CanonicalDocument::parse(&raw, Limits::default())?;
        let receipt = store.put(Cursor::new(canonical.bytes()))?;
        add(&mut stats.staged_bytes, receipt.bytes)?;
        Ok(SnapshotReceipt {
            manifest_digest: receipt.digest,
            manifest,
            stats,
        })
    }
}
