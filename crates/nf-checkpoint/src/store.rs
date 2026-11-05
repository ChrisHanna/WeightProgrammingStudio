//! Linux capability-rooted artifact store. Every path component is opened with
//! O_NOFOLLOW; caller-controlled strings never become relative traversal paths.
//! Directory FDs anchor operations even if an ancestor directory is renamed.

use nf_core::canonical::ContentDigest;
use nf_core::error::{CoreError, ErrorCode, Result};
use rustix::fd::OwnedFd;
use rustix::fs::{self, AtFlags, Dir, FileType, FlockOperation, Mode, OFlags, RenameFlags};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Component, Path};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_STAGE: AtomicU64 = AtomicU64::new(0);
const CHUNK: usize = 65_536;

fn error(code: ErrorCode, message: impl Into<String>) -> CoreError {
    CoreError::new(code, "", message)
}

fn io_error(cause: impl std::fmt::Display) -> CoreError {
    error(ErrorCode::Io, cause.to_string())
}

fn directory_flags() -> OFlags {
    OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC
}

fn child_directory(parent: &OwnedFd, name: &str) -> Result<OwnedFd> {
    match fs::mkdirat(parent, name, Mode::RUSR | Mode::WUSR | Mode::XUSR) {
        Ok(()) | Err(rustix::io::Errno::EXIST) => (),
        Err(cause) => return Err(io_error(cause)),
    }
    fs::openat(parent, name, directory_flags(), Mode::empty()).map_err(io_error)
}

fn explicit_root(path: &Path, create: bool) -> Result<OwnedFd> {
    if !path.is_absolute()
        || path.parent().is_none()
        || path
            .components()
            .any(|component| matches!(component, Component::ParentDir | Component::Prefix(_)))
    {
        return Err(error(
            ErrorCode::InvalidValue,
            "store root must be an explicit absolute non-root directory",
        ));
    }
    let mut current = fs::open("/", directory_flags(), Mode::empty()).map_err(io_error)?;
    for component in path.components() {
        match component {
            Component::RootDir => (),
            Component::Normal(name) => {
                let name = name.to_str().ok_or_else(|| {
                    error(ErrorCode::InvalidValue, "root requires UTF-8 components")
                })?;
                current = if create {
                    child_directory(&current, name)?
                } else {
                    fs::openat(&current, name, directory_flags(), Mode::empty())
                        .map_err(io_error)?
                };
            }
            _ => {
                return Err(error(
                    ErrorCode::InvalidValue,
                    "store root traversal is forbidden",
                ))
            }
        }
    }
    Ok(current)
}

pub(crate) fn open_regular(directory: &OwnedFd, name: &str) -> Result<File> {
    let fd = fs::openat(
        directory,
        name,
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(io_error)?;
    let stat = fs::fstat(&fd).map_err(io_error)?;
    if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile || stat.st_nlink != 1 {
        return Err(error(
            ErrorCode::IntegrityMismatch,
            "artifact must be a regular file without hard-link aliases",
        ));
    }
    Ok(File::from(fd))
}

fn directory_bytes(directory: &OwnedFd, max_entries: usize, blobs: bool) -> Result<u64> {
    let mut total = 0u64;
    let mut count = 0;
    for entry in Dir::read_from(directory).map_err(io_error)? {
        let entry = entry.map_err(io_error)?;
        let name = entry.file_name().to_str().map_err(io_error)?;
        if name == "." || name == ".." {
            continue;
        }
        count += 1;
        if count > max_entries {
            return Err(error(
                ErrorCode::LimitExceeded,
                "artifact entry limit exceeded",
            ));
        }
        if blobs
            && (name.len() != 64
                || !name
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)))
        {
            return Err(error(
                ErrorCode::IntegrityMismatch,
                "unexpected published artifact name",
            ));
        }
        let file = open_regular(directory, name)?;
        let bytes = file.metadata().map_err(io_error)?.len();
        total = total
            .checked_add(bytes)
            .ok_or_else(|| error(ErrorCode::QuotaExceeded, "artifact accounting overflow"))?;
    }
    Ok(total)
}

struct StoreLock(File);
impl Drop for StoreLock {
    fn drop(&mut self) {
        let _ = fs::flock(&self.0, FlockOperation::Unlock);
    }
}

struct Staging<'a> {
    directory: &'a OwnedFd,
    name: String,
}
impl Drop for Staging<'_> {
    fn drop(&mut self) {
        let _ = fs::unlinkat(self.directory, &self.name, AtFlags::empty());
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StoreLimits {
    pub total_bytes: u64,
    pub max_blob_bytes: u64,
    pub max_entries: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlobReceipt {
    pub digest: ContentDigest,
    pub bytes: u64,
    pub deduplicated: bool,
}

#[derive(Debug)]
pub struct ArtifactStore {
    root: OwnedFd,
    blobs: OwnedFd,
    staging: OwnedFd,
    limits: StoreLimits,
    read_only: bool,
}

impl ArtifactStore {
    pub fn create(root: &Path, limits: StoreLimits) -> Result<Self> {
        if limits.total_bytes == 0
            || limits.max_blob_bytes == 0
            || limits.max_entries == 0
            || limits.max_blob_bytes > limits.total_bytes
        {
            return Err(error(
                ErrorCode::InvalidValue,
                "finite positive, compatible store limits are required",
            ));
        }
        let root = explicit_root(root, true)?;
        let blobs = child_directory(&root, "blobs")?;
        let staging = child_directory(&root, "staging")?;
        let store = Self {
            root,
            blobs,
            staging,
            limits,
            read_only: false,
        };
        let _lock = store.lock()?;
        store.usage_locked()?;
        Ok(store)
    }

    pub fn open_read_only(root: &Path, limits: StoreLimits) -> Result<Self> {
        if limits.total_bytes == 0
            || limits.max_blob_bytes == 0
            || limits.max_entries == 0
            || limits.max_blob_bytes > limits.total_bytes
        {
            return Err(error(
                ErrorCode::InvalidValue,
                "finite compatible store limits required",
            ));
        }
        let root = explicit_root(root, false)?;
        let blobs =
            fs::openat(&root, "blobs", directory_flags(), Mode::empty()).map_err(io_error)?;
        let staging =
            fs::openat(&root, "staging", directory_flags(), Mode::empty()).map_err(io_error)?;
        let store = Self {
            root,
            blobs,
            staging,
            limits,
            read_only: true,
        };
        let _lock = store.lock()?;
        store.usage_locked()?;
        Ok(store)
    }

    fn lock(&self) -> Result<StoreLock> {
        // A new open description per operation also serializes threads sharing
        // this store; flock on one reused description would not do that.
        let fd = fs::openat(
            &self.root,
            ".lock",
            (if self.read_only {
                OFlags::RDONLY
            } else {
                OFlags::RDWR | OFlags::CREATE
            }) | OFlags::NOFOLLOW
                | OFlags::NONBLOCK
                | OFlags::CLOEXEC,
            Mode::RUSR | Mode::WUSR,
        )
        .map_err(io_error)?;
        let stat = fs::fstat(&fd).map_err(io_error)?;
        if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile || stat.st_nlink != 1 {
            return Err(error(
                ErrorCode::IntegrityMismatch,
                "invalid store lock file",
            ));
        }
        let file = File::from(fd);
        fs::flock(
            &file,
            if self.read_only {
                FlockOperation::LockShared
            } else {
                FlockOperation::LockExclusive
            },
        )
        .map_err(io_error)?;
        Ok(StoreLock(file))
    }

    fn usage_locked(&self) -> Result<u64> {
        let blobs = directory_bytes(&self.blobs, self.limits.max_entries, true)?;
        let staging = directory_bytes(&self.staging, self.limits.max_entries, false)?;
        let total = blobs
            .checked_add(staging)
            .ok_or_else(|| error(ErrorCode::QuotaExceeded, "store usage overflow"))?;
        if total > self.limits.total_bytes {
            return Err(error(
                ErrorCode::QuotaExceeded,
                "existing artifacts exceed quota",
            ));
        }
        Ok(total)
    }

    pub fn used_bytes(&self) -> Result<u64> {
        let _lock = self.lock()?;
        self.usage_locked()
    }

    pub fn put<R: Read>(&self, mut source: R) -> Result<BlobReceipt> {
        if self.read_only {
            return Err(error(
                ErrorCode::PermissionDenied,
                "read-only artifact store cannot publish",
            ));
        }
        let _lock = self.lock()?;
        let used = self.usage_locked()?;
        for directory in [&self.blobs, &self.staging] {
            let mut count = 0;
            for entry in Dir::read_from(directory).map_err(io_error)? {
                let entry = entry.map_err(io_error)?;
                if !matches!(entry.file_name().to_bytes(), b"." | b"..") {
                    count += 1;
                }
            }
            if count >= self.limits.max_entries {
                return Err(error(
                    ErrorCode::LimitExceeded,
                    "no free artifact entry slot",
                ));
            }
        }
        let mut stage = None;
        for _ in 0..64 {
            let name = format!(
                "{}-{}",
                std::process::id(),
                NEXT_STAGE.fetch_add(1, Ordering::Relaxed)
            );
            match fs::openat(
                &self.staging,
                &name,
                OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                Mode::RUSR | Mode::WUSR,
            ) {
                Ok(fd) => {
                    stage = Some((
                        Staging {
                            directory: &self.staging,
                            name,
                        },
                        File::from(fd),
                    ));
                    break;
                }
                Err(rustix::io::Errno::EXIST) => continue,
                Err(cause) => return Err(io_error(cause)),
            }
        }
        let (stage, mut file) = stage
            .ok_or_else(|| error(ErrorCode::LimitExceeded, "staging name attempts exhausted"))?;
        let mut bytes = 0u64;
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; CHUNK];
        loop {
            let count = match source.read(&mut buffer) {
                Ok(count) => count,
                Err(cause) if cause.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(cause) => return Err(io_error(cause)),
            };
            if count == 0 {
                break;
            }
            bytes = bytes
                .checked_add(count as u64)
                .ok_or_else(|| error(ErrorCode::QuotaExceeded, "blob size overflow"))?;
            if bytes > self.limits.max_blob_bytes
                || used
                    .checked_add(bytes)
                    .is_none_or(|total| total > self.limits.total_bytes)
            {
                return Err(error(
                    ErrorCode::QuotaExceeded,
                    "staged artifact exceeds blob or total byte quota",
                ));
            }
            file.write_all(&buffer[..count]).map_err(io_error)?;
            hasher.update(&buffer[..count]);
        }
        file.sync_all().map_err(io_error)?;
        let digest: ContentDigest = format!("sha256:{:x}", hasher.finalize()).parse()?;
        let name = &digest.as_str()[7..];
        match fs::openat(
            &self.blobs,
            name,
            OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
            Mode::empty(),
        ) {
            Ok(existing) => {
                drop(existing);
                let existing_bytes = self.verify_locked(&digest)?;
                if existing_bytes != bytes {
                    return Err(error(
                        ErrorCode::IntegrityMismatch,
                        "existing blob length differs",
                    ));
                }
                return Ok(BlobReceipt {
                    digest,
                    bytes,
                    deduplicated: true,
                });
            }
            Err(rustix::io::Errno::NOENT) => (),
            Err(cause) => return Err(io_error(cause)),
        }
        fs::fchmod(&file, Mode::RUSR).map_err(io_error)?;
        file.sync_all().map_err(io_error)?;
        // NOREPLACE makes publication atomic without overwriting existing bytes.
        fs::renameat_with(
            &self.staging,
            &stage.name,
            &self.blobs,
            name,
            RenameFlags::NOREPLACE,
        )
        .map_err(io_error)?;
        fs::fsync(&self.blobs).map_err(io_error)?;
        fs::fsync(&self.staging).map_err(io_error)?;
        Ok(BlobReceipt {
            digest,
            bytes,
            deduplicated: false,
        })
    }

    /// Verify a stored blob without allocating its payload. Missing is distinct
    /// from corruption; callers may reuse only a successfully verified digest.
    pub fn contains_verified(&self, digest: &ContentDigest) -> Result<bool> {
        let _lock = self.lock()?;
        match fs::openat(
            &self.blobs,
            &digest.as_str()[7..],
            OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
            Mode::empty(),
        ) {
            Ok(fd) => {
                drop(fd);
                self.verify_locked(digest)?;
                Ok(true)
            }
            Err(rustix::io::Errno::NOENT) => Ok(false),
            Err(cause) => Err(io_error(cause)),
        }
    }

    fn verify_locked(&self, digest: &ContentDigest) -> Result<u64> {
        let mut file = open_regular(&self.blobs, &digest.as_str()[7..])?;
        let expected = file.metadata().map_err(io_error)?.len();
        if expected > self.limits.max_blob_bytes {
            return Err(error(
                ErrorCode::QuotaExceeded,
                "artifact exceeds configured verification bound",
            ));
        }
        let mut total = 0u64;
        let mut hasher = Sha256::new();
        let mut buffer = [0; CHUNK];
        loop {
            let count = match file.read(&mut buffer) {
                Ok(count) => count,
                Err(cause) if cause.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(cause) => return Err(io_error(cause)),
            };
            if count == 0 {
                break;
            }
            total = total
                .checked_add(count as u64)
                .filter(|total| *total <= expected)
                .ok_or_else(|| {
                    error(
                        ErrorCode::IntegrityMismatch,
                        "artifact grew during verification",
                    )
                })?;
            hasher.update(&buffer[..count]);
        }
        let actual: ContentDigest = format!("sha256:{:x}", hasher.finalize()).parse()?;
        if total != expected || &actual != digest {
            return Err(error(
                ErrorCode::IntegrityMismatch,
                "artifact content does not match its identity",
            ));
        }
        Ok(total)
    }

    fn read_locked(&self, digest: &ContentDigest, max_bytes: u64) -> Result<Vec<u8>> {
        let mut file = open_regular(&self.blobs, &digest.as_str()[7..])?;
        let before = file.metadata().map_err(io_error)?;
        if before.len() > max_bytes {
            return Err(error(
                ErrorCode::QuotaExceeded,
                "artifact exceeds configured read bound",
            ));
        }
        let mut bytes = Vec::new();
        let mut buffer = [0u8; CHUNK];
        loop {
            let count = match file.read(&mut buffer) {
                Ok(count) => count,
                Err(cause) if cause.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(cause) => return Err(io_error(cause)),
            };
            if count == 0 {
                break;
            }
            if (bytes.len() as u64)
                .checked_add(count as u64)
                .is_none_or(|total| total > max_bytes)
            {
                return Err(error(
                    ErrorCode::QuotaExceeded,
                    "artifact grew beyond read bound",
                ));
            }
            bytes.extend_from_slice(&buffer[..count]);
        }
        if bytes.len() as u64 != before.len() || &ContentDigest::of_bytes(&bytes) != digest {
            return Err(error(
                ErrorCode::IntegrityMismatch,
                "artifact content does not match its identity",
            ));
        }
        Ok(bytes)
    }

    pub fn read(&self, digest: &ContentDigest) -> Result<Vec<u8>> {
        let _lock = self.lock()?;
        self.read_locked(digest, self.limits.max_blob_bytes)
    }

    /// Check the caller's byte cap before reading, even in a larger store.
    pub fn read_bounded(&self, digest: &ContentDigest, max_bytes: u64) -> Result<Vec<u8>> {
        let _lock = self.lock()?;
        self.read_locked(digest, max_bytes.min(self.limits.max_blob_bytes))
    }
}
