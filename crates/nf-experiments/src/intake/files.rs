//! FD-anchored, no-follow selected-asset intake with a content fence.
use super::{fail, Asset, AssetKind, MAX_FILE};
use nf_core::canonical::ContentDigest;
use nf_core::error::{ErrorCode, Result};
use rustix::fd::OwnedFd;
use rustix::fs::{self, FileType, Mode, OFlags};
use std::fs::File;
use std::io::Read;
use std::path::{Component, Path};

fn io(e: impl std::fmt::Display) -> nf_core::error::CoreError {
    fail(ErrorCode::Io, &e.to_string())
}
fn flags() -> OFlags {
    OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC
}
pub(super) struct Root {
    fd: OwnedFd,
    identity: (u64, u64),
}
impl Root {
    pub fn open(path: &Path) -> Result<Self> {
        if !path.is_absolute() || path.parent().is_none() {
            return Err(fail(
                ErrorCode::ScopeViolation,
                "explicit non-root absolute package directory required",
            ));
        }
        let mut fd = fs::open("/", flags(), Mode::empty()).map_err(io)?;
        for c in path.components() {
            match c {
                Component::RootDir => (),
                Component::Normal(c) => {
                    fd = fs::openat(&fd, c, flags(), Mode::empty()).map_err(io)?
                }
                _ => {
                    return Err(fail(
                        ErrorCode::ScopeViolation,
                        "package root traversal denied",
                    ))
                }
            }
        }
        let s = fs::fstat(&fd).map_err(io)?;
        Ok(Self {
            fd,
            identity: (s.st_dev, s.st_ino),
        })
    }
    pub fn same_root(&self, path: &Path) -> Result<()> {
        if Self::open(path)?.identity != self.identity {
            return Err(fail(ErrorCode::SourceMismatch, "package root replaced"));
        }
        Ok(())
    }
    pub fn observe(&self, name: &str, kind: AssetKind) -> Result<Observed> {
        super::relative(name)?;
        let parts: Vec<_> = name.split('/').collect();
        let mut fd = fs::openat(&self.fd, ".", flags(), Mode::empty()).map_err(io)?;
        for part in &parts[..parts.len() - 1] {
            fd = fs::openat(&fd, *part, flags(), Mode::empty()).map_err(io)?;
        }
        let raw = fs::openat(
            &fd,
            *parts.last().unwrap(),
            OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(io)?;
        let before = fs::fstat(&raw).map_err(io)?;
        if FileType::from_raw_mode(before.st_mode) != FileType::RegularFile
            || before.st_nlink != 1
            || before.st_size < 0
            || before.st_size as u64 > MAX_FILE as u64
        {
            return Err(fail(
                ErrorCode::ScopeViolation,
                "selected asset must be bounded regular single-link file",
            ));
        }
        let mut bytes = Vec::new();
        File::from(raw)
            .take((MAX_FILE + 1) as u64)
            .read_to_end(&mut bytes)
            .map_err(io)?;
        if bytes.len() > MAX_FILE {
            return Err(fail(ErrorCode::LimitExceeded, "asset grew beyond bound"));
        }
        let asset = Asset {
            kind,
            digest: ContentDigest::of_bytes(&bytes),
            bytes: bytes.len() as u64,
        };
        Ok(Observed {
            name: name.into(),
            identity: (before.st_dev, before.st_ino),
            asset,
            data: bytes,
        })
    }
    pub fn fence(&self, observed: &Observed) -> Result<()> {
        let fresh = self.observe(&observed.name, observed.asset.kind)?;
        if fresh.identity != observed.identity || fresh.asset != observed.asset {
            return Err(fail(
                ErrorCode::SourceMismatch,
                "selected source changed during snapshot",
            ));
        }
        Ok(())
    }
}
pub(super) struct Observed {
    pub name: String,
    identity: (u64, u64),
    pub asset: Asset,
    pub data: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    #[test]
    fn custom_gym_import_content_fence_rejects_same_size_rewrite_and_replacement() {
        let base = std::env::var_os("KIROCREW_SCRATCH")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        let path = base.join(format!(
            "intake-fence-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        std::fs::write(path.join("a.py"), b"abcd").unwrap();
        let root = Root::open(&path).unwrap();
        let original = root.observe("a.py", AssetKind::Code).unwrap();
        std::fs::write(path.join("a.py"), b"efgh").unwrap();
        assert_eq!(
            root.fence(&original).unwrap_err().code,
            ErrorCode::SourceMismatch
        );
        std::fs::write(path.join("new.py"), b"abcd").unwrap();
        std::fs::rename(path.join("new.py"), path.join("a.py")).unwrap();
        assert_eq!(
            root.fence(&original).unwrap_err().code,
            ErrorCode::SourceMismatch
        );
        std::fs::remove_file(path.join("a.py")).unwrap();
        std::fs::remove_dir(path).unwrap();
    }
}
