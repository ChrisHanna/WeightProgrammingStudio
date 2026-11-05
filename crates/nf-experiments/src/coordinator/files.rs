//! Linux directory-FD anchors; SQLite files are never opened through caller-relative names.
use super::*;
use rustix::fd::{AsRawFd, OwnedFd};
use rustix::fs::{self, FileType, Mode, OFlags};
use std::io::Read;
use std::path::{Component, Path};
pub(super) struct Files {
    root: OwnedFd,
    db: std::fs::File,
    path: std::path::PathBuf,
}
fn directory_flags() -> OFlags {
    OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC
}
fn io(error: impl std::fmt::Display) -> CoreError {
    failure(ErrorCode::Io, &error.to_string())
}
impl Files {
    /// Existing canonical ledger file only. No mkdir, create, chmod or migration.
    pub(super) fn open_read_only_file(path: &Path, max: u64) -> Result<Self> {
        if !path.is_absolute() || path.file_name() != Some(std::ffi::OsStr::new("events.sqlite3")) {
            return Err(failure(
                ErrorCode::InvalidValue,
                "explicit existing events.sqlite3 file required",
            ));
        }
        let parent = path
            .parent()
            .ok_or_else(|| failure(ErrorCode::InvalidValue, "database parent required"))?;
        let mut root = fs::open("/", directory_flags(), Mode::empty()).map_err(io)?;
        for component in parent.components() {
            match component {
                Component::RootDir => (),
                Component::Normal(name) => {
                    root = fs::openat(&root, name, directory_flags(), Mode::empty()).map_err(io)?
                }
                _ => {
                    return Err(failure(
                        ErrorCode::InvalidValue,
                        "database path traversal forbidden",
                    ))
                }
            }
        }
        let fd = fs::openat(
            &root,
            "events.sqlite3",
            OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(io)?;
        let path = std::fs::read_link(format!("/proc/self/fd/{}", root.as_raw_fd())).map_err(io)?;
        let files = Self {
            root,
            db: std::fs::File::from(fd),
            path,
        };
        files.verify(max)?;
        // Read-only SQLite WAL readers may create a shared-memory file. This
        // ledger uses rollback journals; reject WAL before opening SQLite.
        for name in ["events.sqlite3-wal", "events.sqlite3-shm"] {
            match fs::statat(&files.root, name, fs::AtFlags::SYMLINK_NOFOLLOW) {
                Err(rustix::io::Errno::NOENT) => (),
                Ok(_) => {
                    return Err(failure(
                        ErrorCode::UnsupportedCapability,
                        "read-only capture does not open WAL/shared-memory stores",
                    ))
                }
                Err(e) => return Err(io(e)),
            }
        }
        let mut header = [0u8; 100];
        std::os::unix::fs::FileExt::read_exact_at(&files.db, &mut header, 0).map_err(io)?;
        if &header[..16] != b"SQLite format 3\0" || header[18] != 1 || header[19] != 1 {
            return Err(failure(
                ErrorCode::UnsupportedVersion,
                "read-only rollback-journal SQLite format required",
            ));
        }
        Ok(files)
    }
    pub fn open(path: &Path) -> Result<Self> {
        if !path.is_absolute()
            || path.parent().is_none()
            || path
                .components()
                .any(|c| matches!(c, Component::ParentDir | Component::Prefix(_)))
        {
            return Err(failure(
                ErrorCode::InvalidValue,
                "explicit absolute non-root database directory required",
            ));
        }
        let mut root = fs::open("/", directory_flags(), Mode::empty()).map_err(io)?;
        for component in path.components() {
            match component {
                Component::RootDir => (),
                Component::Normal(name) => {
                    match fs::mkdirat(&root, name, Mode::RUSR | Mode::WUSR | Mode::XUSR) {
                        Ok(()) | Err(rustix::io::Errno::EXIST) => (),
                        Err(error) => return Err(io(error)),
                    }
                    root = fs::openat(&root, name, directory_flags(), Mode::empty()).map_err(io)?;
                }
                _ => {
                    return Err(failure(
                        ErrorCode::InvalidValue,
                        "database root traversal forbidden",
                    ))
                }
            }
        }
        if fs::fstat(&root).map_err(io)?.st_mode & 0o022 != 0 {
            return Err(failure(
                ErrorCode::PermissionDenied,
                "ledger directory must not be group/world writable",
            ));
        }
        match fs::statat(&root, "events.sqlite3", fs::AtFlags::SYMLINK_NOFOLLOW) {
            Ok(stat)
                if FileType::from_raw_mode(stat.st_mode) == FileType::RegularFile
                    && stat.st_nlink == 1
                    && stat.st_mode & 0o022 == 0 => {}
            Ok(_) => {
                return Err(failure(
                    ErrorCode::IntegrityMismatch,
                    "unsafe database path type/alias/permissions",
                ))
            }
            Err(rustix::io::Errno::NOENT) => (),
            Err(error) => return Err(io(error)),
        }
        let fd = fs::openat(
            &root,
            "events.sqlite3",
            OFlags::RDWR | OFlags::CREATE | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
            Mode::RUSR | Mode::WUSR,
        )
        .map_err(io)?;
        let path = std::fs::read_link(format!("/proc/self/fd/{}", root.as_raw_fd())).map_err(io)?;
        let files = Self {
            root,
            db: std::fs::File::from(fd),
            path,
        };
        files.verify(u64::MAX)?;
        Ok(files)
    }
    pub fn sqlite_path(&self) -> std::path::PathBuf {
        self.path.join("events.sqlite3")
    }
    pub fn verify(&self, max: u64) -> Result<()> {
        let mut current_root = fs::open("/", directory_flags(), Mode::empty()).map_err(io)?;
        for component in self.path.components() {
            if let Component::Normal(name) = component {
                current_root = fs::openat(&current_root, name, directory_flags(), Mode::empty())
                    .map_err(io)?;
            }
        }
        let expected = fs::fstat(&self.root).map_err(io)?;
        let actual = fs::fstat(&current_root).map_err(io)?;
        if expected.st_ino != actual.st_ino
            || expected.st_dev != actual.st_dev
            || actual.st_mode & 0o022 != 0
        {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "ledger directory identity changed",
            ));
        }
        let anchor = fs::fstat(&self.db).map_err(io)?;
        let current =
            fs::statat(&self.root, "events.sqlite3", fs::AtFlags::SYMLINK_NOFOLLOW).map_err(io)?;
        if anchor.st_ino != current.st_ino
            || anchor.st_dev != current.st_dev
            || current.st_nlink != 1
            || current.st_mode & 0o022 != 0
            || FileType::from_raw_mode(current.st_mode) != FileType::RegularFile
            || current.st_size < 0
            || current.st_size as u64 > max
        {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "database identity/type/link/size changed",
            ));
        }
        for name in [
            "events.sqlite3-journal",
            "events.sqlite3-wal",
            "events.sqlite3-shm",
        ] {
            match fs::statat(&self.root, name, fs::AtFlags::SYMLINK_NOFOLLOW) {
                Ok(stat)
                    if FileType::from_raw_mode(stat.st_mode) == FileType::RegularFile
                        && stat.st_nlink == 1 => {}
                Ok(_) => {
                    return Err(failure(
                        ErrorCode::IntegrityMismatch,
                        "unsafe SQLite side-file alias",
                    ))
                }
                Err(rustix::io::Errno::NOENT) => (),
                Err(error) => return Err(io(error)),
            }
        }
        Ok(())
    }
}
#[derive(Clone, Debug)]
pub(super) struct Moment {
    pub boot: String,
    pub ms: i64,
    pub wall_ms: i64,
}
pub(super) fn moment() -> Result<Moment> {
    let mut boot = String::new();
    std::fs::File::open("/proc/sys/kernel/random/boot_id")
        .map_err(io)?
        .take(65)
        .read_to_string(&mut boot)
        .map_err(io)?;
    if boot.trim().len() != 36
        || !boot
            .trim()
            .bytes()
            .all(|b| b.is_ascii_hexdigit() || b == b'-')
    {
        return Err(failure(
            ErrorCode::UnsupportedCapability,
            "kernel boot identity unavailable",
        ));
    }
    let time = rustix::time::clock_gettime(rustix::time::ClockId::Boottime);
    let ms = time
        .tv_sec
        .checked_mul(1000)
        .and_then(|n| n.checked_add(time.tv_nsec / 1_000_000))
        .ok_or_else(|| failure(ErrorCode::LimitExceeded, "monotonic clock overflow"))?;
    let wall = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(io)?
        .as_millis();
    let wall_ms = i64::try_from(wall)
        .map_err(|_| failure(ErrorCode::LimitExceeded, "wall clock overflow"))?;
    Ok(Moment {
        boot: boot.trim().to_owned(),
        ms,
        wall_ms,
    })
}
