//! OS authority, not a digest signature. Production has no configurable root.
//! Grants are immutable; NEVER replace/unlink grants, locks, or marker files.
//! Revocation atomically publishes a separate marker under a stable lock inode.
//! Revocation linearizes at its exclusive flock; admission/publication hold a
//! shared flock through their SQLite commits, but NOT during native reload work.
//! Expiration is checked at the last logical
//! fence, not guaranteed through unbounded fsync. Root/capability-bearing agents
//! are outside this trust model and are refused as promotion consumers.
use super::*;
use rustix::fs::{openat, Mode, OFlags, CWD};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    os::fd::AsRawFd,
    os::unix::fs::MetadataExt,
    path::PathBuf,
};
const ROOT: &str = "/etc/nf-promotion";
const MAX: u64 = 524288;
#[cfg(test)]
thread_local! { static TEST_ROOT: std::cell::RefCell<Option<PathBuf>> = const { std::cell::RefCell::new(None) }; }
fn location() -> (PathBuf, u32, bool) {
    #[cfg(test)]
    if let Some(path) = TEST_ROOT.with(|v| v.borrow().clone()) {
        return (path, unsafe { libc::geteuid() }, true);
    }
    (PathBuf::from(ROOT), 0, false)
}
fn io(e: impl std::fmt::Display) -> CoreError {
    deny(&format!("operator registry: {e}"))
}
fn now() -> Result<u64> {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|v| v.as_millis() as u64)
        .map_err(io)
}
fn checked(file: &File, uid: u32, directory: bool) -> Result<()> {
    let m = file.metadata().map_err(io)?;
    if m.uid() != uid
        || m.mode() & 0o022 != 0
        || (directory && !m.is_dir())
        || (!directory && (!m.is_file() || m.nlink() != 1 || m.len() > MAX))
    {
        return Err(deny("operator registry must be owner-controlled, non-writable by group/other, regular single-link files"));
    }
    Ok(())
}
fn directory() -> Result<File> {
    let (path, uid, test) = location();
    let flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
    let mut dir = File::from(openat(CWD, "/", flags, Mode::empty()).map_err(io)?);
    // Production verifies EVERY ancestor, including / and /etc. Test-only seam
    // anchors at an isolated temporary directory; it never touches /etc.
    if !test {
        checked(&dir, uid, true)?;
    }
    for component in path.components() {
        match component {
            std::path::Component::RootDir => {}
            std::path::Component::Normal(name) => {
                dir = File::from(openat(&dir, name, flags, Mode::empty()).map_err(io)?);
                if !test {
                    checked(&dir, uid, true)?;
                }
            }
            _ => {
                return Err(deny(
                    "operator registry requires an absolute non-traversing path",
                ))
            }
        }
    }
    checked(&dir, uid, true)?;
    Ok(dir)
}
fn lock(file: &File, exclusive: bool) -> Result<()> {
    // Nonblocking: an operator or stalled promotion must not cause unbounded
    // admission. An operator retries a busy revocation; no success is reported.
    if unsafe {
        libc::flock(
            file.as_raw_fd(),
            (if exclusive {
                libc::LOCK_EX
            } else {
                libc::LOCK_SH
            }) | libc::LOCK_NB,
        )
    } != 0
    {
        return Err(deny(
            "operator registry busy; revocation/admission did not linearize",
        ));
    }
    Ok(())
}
fn name(id: &ContentDigest) -> String {
    format!("{}.json", id.as_str().trim_start_matches("sha256:"))
}
fn entry(file: &mut File) -> Result<Value> {
    file.seek(SeekFrom::Start(0)).map_err(io)?;
    let mut bytes = Vec::new();
    file.take(MAX + 1).read_to_end(&mut bytes).map_err(io)?;
    if bytes.len() as u64 > MAX {
        return Err(deny("oversized operator grant"));
    }
    serde_json::from_slice(&bytes).map_err(io)
}
fn sidecar(id: &ContentDigest, suffix: &str) -> String {
    format!("{}.{suffix}", name(id))
}
#[cfg(test)]
thread_local! {
    static FAULT: std::cell::RefCell<Option<&'static str>> = const { std::cell::RefCell::new(None) };
    static FAULT_CRASH: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
fn checkpoint(_point: &'static str) -> Result<()> {
    #[cfg(test)]
    if FAULT.with(|v| *v.borrow() == Some(_point)) {
        if FAULT_CRASH.with(|v| v.get()) {
            unsafe { libc::_exit(86) };
        }
        return Err(io(format!("injected {_point}")));
    }
    Ok(())
}
// All writes go to an unreferenced staging name. A failure leaves either no
// published object or a complete object; it NEVER damages an existing grant.
// Staging files may remain after a crash. They confer no authority.
fn publish(dir: &File, target: &str, value: &Value) -> Result<()> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let temporary = format!(
        ".{target}.{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    );
    let mut file = File::from(
        openat(
            dir,
            temporary.as_str(),
            OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::from_bits_truncate(0o644),
        )
        .map_err(io)?,
    );
    rustix::fs::fchmod(&file, Mode::from_bits_truncate(0o644)).map_err(io)?;
    let bytes = canonical(value)?;
    if bytes.len() as u64 > MAX {
        return Err(deny("oversized operator grant"));
    }
    checkpoint("before_write")?;
    // Two writes permit deterministic simulation of a torn staging write.
    let split = bytes.len() / 2;
    file.write_all(&bytes[..split]).map_err(io)?;
    checkpoint("partial_write")?;
    file.write_all(&bytes[split..]).map_err(io)?;
    checkpoint("after_write")?;
    checkpoint("before_file_fsync")?;
    file.sync_all().map_err(io)?;
    checkpoint("after_file_fsync")?;
    checkpoint("before_rename")?;
    rustix::fs::renameat(dir, temporary.as_str(), dir, target).map_err(io)?;
    checkpoint("after_rename")?;
    checkpoint("before_directory_fsync")?;
    dir.sync_all().map_err(io)?;
    checkpoint("after_directory_fsync")
}
fn present(dir: &File, name: &str) -> Result<bool> {
    match openat(
        dir,
        name,
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK,
        Mode::empty(),
    ) {
        Ok(fd) => {
            checked(&File::from(fd), location().1, false)?;
            Ok(true)
        }
        Err(rustix::io::Errno::NOENT) => Ok(false),
        Err(error) => Err(io(error)),
    }
}
fn stable_lock(dir: &File, id: &ContentDigest, create: bool, exclusive: bool) -> Result<File> {
    let flags = OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK;
    let file = File::from(
        openat(
            dir,
            sidecar(id, "lock"),
            flags
                | if create {
                    OFlags::CREATE
                } else {
                    OFlags::empty()
                },
            Mode::from_bits_truncate(0o644),
        )
        .map_err(io)?,
    );
    checked(&file, location().1, false)?;
    lock(&file, exclusive)?;
    if create {
        // Persist the stable identity BEFORE publishing any grant. Consumers
        // never create locks and operators never replace or unlink them.
        checkpoint("before_lock_fsync")?;
        file.sync_all().map_err(io)?;
        checkpoint("after_lock_fsync")?;
        checkpoint("before_lock_directory_fsync")?;
        dir.sync_all().map_err(io)?;
        checkpoint("after_lock_directory_fsync")?;
    }
    Ok(file)
}
fn operator() -> Result<()> {
    let (_, uid, test) = location();
    if !test && (unsafe { libc::getuid() } != 0 || unsafe { libc::geteuid() } != 0) {
        return Err(deny(
            "promotion issuance/revocation requires a separate root operator",
        ));
    }
    if unsafe { libc::geteuid() } != uid {
        return Err(deny("wrong operator principal"));
    }
    Ok(())
}
/// Root-only issuance. The operator must inspect the COMPLETE request, including
/// native operational (not independent or strict-fsync) scope and all bindings.
/// Install /etc/nf-promotion as root:root 0755 first. No environment override.
/// Return value is a lookup identifier, never bearer authority on its own.
pub fn issue(request: &Value, expires_unix_ms: u64) -> Result<ContentDigest> {
    operator()?;
    if request["schema_version"] != 1
        || request["scope"] != "operational_fixture"
        || request["operator_approval"] != Value::Null
        || expires_unix_ms <= now()?
    {
        return Err(deny("invalid operator approval request/expiration"));
    }
    let authorization = json!({"request":request,"expires_unix_ms":expires_unix_ms});
    let id = hash(&authorization)?;
    let dir = directory()?;
    let file = stable_lock(&dir, &id, true, true)?;
    rustix::fs::fchmod(&file, Mode::from_bits_truncate(0o644)).map_err(io)?;
    if present(&dir, &name(&id))? {
        return Err(deny("operator grant already issued; never replace it"));
    }
    publish(&dir, &name(&id), &json!({"authorization":authorization}))?;
    Ok(id)
}
/// Root-only atomic marker publication. Busy means NOT revoked. An I/O error
/// has an UNKNOWN outcome: retry; an existing marker is durably synced again.
/// The original grant is never rewritten, including after a partial write/crash.
pub fn revoke(id: &ContentDigest) -> Result<()> {
    operator()?;
    let dir = directory()?;
    let _lock = stable_lock(&dir, id, false, true)?;
    if !present(&dir, &name(id))? {
        return Err(deny("missing immutable operator grant"));
    }
    let marker = sidecar(id, "revoked");
    if present(&dir, &marker)? {
        checkpoint("before_directory_fsync")?;
        dir.sync_all().map_err(io)?;
        checkpoint("after_directory_fsync")?;
        return Ok(());
    }
    publish(&dir, &marker, &json!({"revoked":id}))
}
/// Keep alive across the ENTIRE admission/publication transaction, not work.
pub(super) struct Guard {
    _file: File,
}
pub(super) fn verify(id: &ContentDigest, request: &Value) -> Result<Guard> {
    resolve(id, request, true)
}
/// Retained root-issued provenance, NOT live authority. Revocation/expiration
/// never rewrites history, but a made-up digest cannot authenticate an archive.
pub(super) fn audit(id: &ContentDigest, request: &Value) -> Result<()> {
    let _guard = resolve(id, request, false)?;
    Ok(())
}
fn unprivileged_status(status: &str) -> Result<()> {
    // Saved-root and filesystem-root credentials also carry privilege even if
    // real/effective UIDs were dropped. Read all four IDs from this same thread.
    let ids = status
        .lines()
        .find_map(|line| line.strip_prefix("Uid:"))
        .map(|v| {
            v.split_whitespace()
                .map(str::parse::<u32>)
                .collect::<std::result::Result<Vec<_>, _>>()
        });
    if !matches!(ids,Some(Ok(ref ids)) if ids.len()==4 && ids.iter().all(|id|*id!=0)) {
        return Err(deny(
            "root/saved-root/filesystem-root promotion consumer is not OS-separated",
        ));
    }
    for field in ["CapInh:", "CapPrm:", "CapEff:", "CapAmb:"] {
        if status
            .lines()
            .find_map(|line| line.strip_prefix(field))
            .and_then(|v| u64::from_str_radix(v.trim(), 16).ok())
            != Some(0)
        {
            return Err(deny(
                "capability-bearing promotion consumer is not OS-separated",
            ));
        }
    }
    Ok(())
}
fn consumer() -> Result<()> {
    if location().2 {
        return Ok(());
    } // private test authority, NOT production authentication
    if unsafe { libc::getuid() } == 0 || unsafe { libc::geteuid() } == 0 {
        return Err(deny(
            "root promotion consumer is not OS-separated from issuer",
        ));
    }
    // Linux capabilities are PER THREAD. /proc/self/status observes the leader
    // and is wrong when called by a worker with different credentials.
    unprivileged_status(&std::fs::read_to_string("/proc/thread-self/status").map_err(io)?)
}
fn grant(dir: &File, id: &ContentDigest, request: &Value, live: bool) -> Result<()> {
    let mut file = File::from(
        openat(
            dir,
            name(id),
            OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK,
            Mode::empty(),
        )
        .map_err(io)?,
    );
    checked(&file, location().1, false)?;
    let value = entry(&mut file)?;
    let authorization = &value["authorization"];
    if hash(authorization)? != *id
        || authorization["request"] != *request
        || authorization["expires_unix_ms"]
            .as_u64()
            .is_none_or(|v| live && now().map_or(true, |n| n >= v))
        || (live && present(dir, &sidecar(id, "revoked"))?)
    {
        return Err(deny(
            "operator approval is forged, foreign, revoked or expired",
        ));
    }
    Ok(())
}
fn resolve(id: &ContentDigest, request: &Value, live: bool) -> Result<Guard> {
    consumer()?;
    let dir = directory()?;
    let file = stable_lock(&dir, id, false, false)?;
    grant(&dir, id, request, live)?;
    Ok(Guard { _file: file })
}
/// One-shot launch admission must be recorded OUTSIDE consumer-owned SQLite.
/// There is deliberately no production fallback to an agent-owned hash/file.
/// A reviewed root service with an authenticated, durable consume-once protocol
/// is not installed here. Fail closed BEFORE reservation, intent or child work.
/// Only the private test seam can simulate that operator-controlled publisher.
pub(super) fn consume(id: &ContentDigest, request: &Value) -> Result<Guard> {
    if !location().2 {
        return Err(failure(ErrorCode::UnsupportedCapability,
            "promotion requires an installed trusted privileged monotonic consumption publisher; root issuance and ledger path/inode binding alone cannot prevent same-inode SQLite rollback/replay"));
    }
    consumer()?;
    let dir = directory()?;
    let file = stable_lock(&dir, id, false, true)?;
    grant(&dir, id, request, true)?;
    let marker = sidecar(id, "consumed");
    if present(&dir, &marker)? {
        return Err(deny(
            "operator approval already consumed; database rollback cannot authorize replay",
        ));
    }
    publish(&dir, &marker, &json!({"consumed":id}))?;
    // Retain the stable descriptor while requesting a shared lock. Linux flock
    // conversion can race; any conversion failure burns approval, never reruns.
    lock(&file, false)?;
    Ok(Guard { _file: file })
}
pub(super) fn audit_consumed(id: &ContentDigest) -> Result<()> {
    let dir = directory()?;
    let _lock = stable_lock(&dir, id, false, false)?;
    let mut file = File::from(
        openat(
            &dir,
            sidecar(id, "consumed"),
            OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK,
            Mode::empty(),
        )
        .map_err(io)?,
    );
    checked(&file, location().1, false)?;
    if entry(&mut file)? != json!({"consumed":id}) {
        return Err(deny(
            "acceptance lacks authentic external operator consumption record",
        ));
    }
    Ok(())
}
#[cfg(test)]
pub(super) fn test_registry(path: &std::path::Path) {
    use std::os::unix::fs::DirBuilderExt;
    match std::fs::DirBuilder::new().mode(0o700).create(path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => panic!("isolated operator registry: {error}"),
    }
    TEST_ROOT.with(|v| *v.borrow_mut() = Some(path.to_owned()));
}
#[cfg(test)]
pub(super) fn test_expiration() -> u64 {
    now().unwrap() + 300_000
}
#[cfg(test)]
pub(super) fn test_revoke_at(id: &ContentDigest, point: &'static str) -> Result<()> {
    FAULT.with(|v| *v.borrow_mut() = Some(point));
    let result = revoke(id);
    FAULT.with(|v| *v.borrow_mut() = None);
    result
}
#[cfg(test)]
#[path = "operator_tests.rs"]
mod tests;
