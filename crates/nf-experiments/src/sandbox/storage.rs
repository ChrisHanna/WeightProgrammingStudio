//! Controller-owned accounting under a directory flock. The trusted host must
//! not mutate the root or its ancestors while locked. Preallocation is only an
//! admission capacity probe: ordinary filesystems cannot reserve future writes.
use super::*;
use std::cell::Cell;
use std::ffi::CString;
use std::os::fd::AsRawFd;

pub const ROOT_BYTES: u64 = 64 * 1024 * 1024;
pub const JOB_BYTES: u64 = 16 * 1024 * 1024;
pub const JOURNALS: usize = 16;
const BLOCK: u64 = 4096;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StorageReceipt {
    /// Initial physically allocated capacity probe, not guaranteed future space.
    #[serde(rename = "admission_preallocated_bytes", alias = "reserved_bytes")]
    pub reserved_bytes: u64,
    /// Observed high water at controller allocation boundaries, including reserve.
    pub peak_charged_bytes: u64,
    pub workspace_bytes_before_reclaim: u64,
    pub retained_bytes: u64,
    /// Snapshot at finalization; other journals can subsequently change it.
    pub root_retained_bytes: u64,
    pub workspace_reclaimed: bool,
    #[serde(default)]
    pub accounting_finalized: bool,
    #[serde(default)]
    pub capacity_guaranteed: bool,
    /// False for legacy recovery: historical allocation samples are unavailable.
    #[serde(default)]
    pub high_water_complete: bool,
    #[serde(default)]
    pub workspace_peak_charged_bytes: u64,
    #[serde(default = "default_workspace_limit")]
    pub workspace_limit_bytes: u64,
    #[serde(default)]
    pub measurement_scope: String,
}
fn default_workspace_limit() -> u64 {
    JOB_BYTES
}

/// Conservative admission charge, including per-file metadata headroom. The
/// runtime also measures actual allocated/logical bytes at every operation.
pub(super) fn workspace_bound(
    source: usize,
    input: usize,
    assets: &BTreeMap<String, Vec<u8>>,
    limits: &Limits,
) -> Result<u64> {
    let mut dirs = std::collections::BTreeSet::from([PathBuf::from("job"), PathBuf::from("guard")]);
    let charge = |n: usize| (n as u64).div_ceil(BLOCK) * BLOCK + BLOCK;
    let policy =
        serde_json::to_vec(limits).map_err(|_| error(ErrorCode::InvalidJson, "policy encoding"))?;
    let mut bytes = charge(source) + charge(input) + charge(GUARDIAN.len()) + charge(policy.len());
    for (name, data) in assets {
        bytes += charge(data.len());
        let relative = Path::new(name);
        for parent in relative.ancestors().skip(1) {
            if !parent.as_os_str().is_empty() && assets.contains_key(parent.to_str().unwrap()) {
                return Err(error(
                    ErrorCode::ScopeViolation,
                    "asset file/directory collision",
                ));
            }
            dirs.insert(Path::new("job/assets").join(parent));
        }
    }
    Ok(bytes + dirs.len() as u64 * BLOCK)
}

pub(super) struct Lock(File);
impl Lock {
    pub fn acquire(root: &Path) -> Result<Self> {
        checked_root(root)?;
        if fs::metadata(root).map_err(io)?.blksize() > BLOCK {
            return Err(error(
                ErrorCode::UnsupportedCapability,
                "sandbox storage requires allocation units at most 4096 bytes",
            ));
        }
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(root)
            .map_err(io)?;
        // SAFETY: live directory descriptor; closing also releases the lock.
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            return Err(error(
                ErrorCode::LimitExceeded,
                "sandbox root busy; retry after current job",
            ));
        }
        let actual = file.metadata().map_err(io)?;
        let named = fs::symlink_metadata(root).map_err(io)?;
        if actual.dev() != named.dev() || actual.ino() != named.ino() {
            return Err(error(ErrorCode::ScopeViolation, "sandbox root replaced"));
        }
        Ok(Self(file))
    }
}
impl Drop for Lock {
    fn drop(&mut self) {
        // SAFETY: live descriptor until after Drop.
        unsafe {
            libc::flock(self.0.as_raw_fd(), libc::LOCK_UN);
        }
    }
}

fn mount_id(path: &Path) -> Result<u64> {
    let name = CString::new(path.as_os_str().as_encoded_bytes())
        .map_err(|_| error(ErrorCode::ScopeViolation, "invalid storage path"))?;
    let mut stat = std::mem::MaybeUninit::<libc::statx>::zeroed();
    // SAFETY: NUL-terminated name and writable statx buffer, no followed symlinks.
    let result = unsafe {
        libc::statx(
            libc::AT_FDCWD,
            name.as_ptr(),
            libc::AT_SYMLINK_NOFOLLOW,
            libc::STATX_MNT_ID,
            stat.as_mut_ptr(),
        )
    };
    if result != 0 {
        return Err(io(std::io::Error::last_os_error()));
    }
    // SAFETY: successful statx initialized the buffer.
    let stat = unsafe { stat.assume_init() };
    if stat.stx_mask & libc::STATX_MNT_ID == 0 {
        return Err(error(
            ErrorCode::UnsupportedCapability,
            "mount identity required for safe storage traversal",
        ));
    }
    Ok(stat.stx_mnt_id)
}
fn same_mount(path: &Path, expected: u64) -> Result<()> {
    if mount_id(path)? != expected {
        return Err(error(
            ErrorCode::ScopeViolation,
            "nested storage mount refused",
        ));
    }
    Ok(())
}
fn allocated(path: &Path) -> Result<u64> {
    let m = fs::symlink_metadata(path).map_err(io)?;
    Ok(m.len().max(m.blocks().saturating_mul(512)))
}
pub(super) fn usage(path: &Path) -> Result<u64> {
    fn walk(path: &Path, mount: u64, depth: usize, nodes: &mut usize) -> Result<u64> {
        *nodes += 1;
        same_mount(path, mount)?;
        let m = fs::symlink_metadata(path).map_err(io)?;
        if depth > 12
            || *nodes > 1024
            || m.file_type().is_symlink()
            || (!m.is_dir() && (!m.is_file() || m.nlink() != 1))
        {
            return Err(error(
                ErrorCode::ScopeViolation,
                "unsafe sandbox storage tree",
            ));
        }
        let mut bytes = m.len().max(m.blocks().saturating_mul(512));
        if m.is_dir() {
            // Charge directories in whole blocks, including tmpfs's inline
            // directory entries. Top-level receipt rename then has a stable
            // charge instead of changing by an inline directory-entry length.
            bytes = bytes.max(BLOCK).div_ceil(BLOCK) * BLOCK;
            for entry in fs::read_dir(path).map_err(io)? {
                bytes = bytes
                    .checked_add(walk(&entry.map_err(io)?.path(), mount, depth + 1, nodes)?)
                    .ok_or_else(|| error(ErrorCode::LimitExceeded, "storage byte overflow"))?;
            }
        }
        Ok(bytes)
    }
    // The path itself must not be a bind mount, including same-device mounts.
    let parent = path
        .parent()
        .ok_or_else(|| error(ErrorCode::ScopeViolation, "storage parent required"))?;
    walk(path, mount_id(parent)?, 0, &mut 0)
}

pub(super) fn jobs(root: &Path) -> Result<Vec<PathBuf>> {
    let mut jobs = Vec::new();
    for (n, entry) in fs::read_dir(root).map_err(io)?.enumerate() {
        if n >= 4096 {
            return Err(error(ErrorCode::LimitExceeded, "root entry bound exceeded"));
        }
        let entry = entry.map_err(io)?;
        if entry
            .file_name()
            .to_string_lossy()
            .starts_with("nf-sandbox-")
        {
            if !valid_name(&entry.file_name().to_string_lossy())
                || !entry.file_type().map_err(io)?.is_dir()
            {
                return Err(error(
                    ErrorCode::ScopeViolation,
                    "invalid sandbox journal entry",
                ));
            }
            usage(&entry.path())?;
            jobs.push(entry.path());
        }
    }
    jobs.sort_by_key(|p| (fs::metadata(p).and_then(|m| m.modified()).ok(), p.clone()));
    Ok(jobs)
}
pub(super) fn root_usage(root: &Path) -> Result<u64> {
    let mut bytes = allocated(root)?;
    for path in jobs(root)? {
        bytes += usage(&path)?;
    }
    Ok(bytes)
}
pub(super) fn read_receipt(path: &Path) -> Result<Receipt> {
    let f = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
        .map_err(io)?;
    let m = f.metadata().map_err(io)?;
    if !m.is_file() || m.nlink() != 1 || m.len() > MAX_CONTROL as u64 {
        return Err(error(
            ErrorCode::IntegrityMismatch,
            "invalid private receipt",
        ));
    }
    serde_json::from_reader(f.take(MAX_CONTROL as u64))
        .map_err(|_| error(ErrorCode::InvalidJson, "invalid private receipt"))
}
pub(super) fn legacy_recovery(path: &Path) -> Result<bool> {
    let f = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
        .map_err(io)?;
    if f.metadata().map_err(io)?.len() > MAX_CONTROL as u64 {
        return Err(error(ErrorCode::LimitExceeded, "recovery marker too large"));
    }
    let value: Value = serde_json::from_reader(f.take(MAX_CONTROL as u64))
        .map_err(|_| error(ErrorCode::InvalidJson, "invalid recovery marker"))?;
    Ok(value
        == json!({"cleanup_confirmed": true, "status": "incomplete", "outputs_eligible": false}))
}
pub(super) fn finalized(path: &Path, receipt: &Receipt) -> Result<bool> {
    if receipt.job != path.file_name().unwrap().to_string_lossy() {
        return Err(error(
            ErrorCode::IntegrityMismatch,
            "receipt target mismatch",
        ));
    }
    Ok(receipt.cleanup_confirmed
        && receipt
            .storage
            .as_ref()
            .is_some_and(|s| s.workspace_reclaimed && s.accounting_finalized)
        && !path.join("receipt.tmp").try_exists().map_err(io)?
        && !path.join("job").try_exists().map_err(io)?
        && !path.join("guard").try_exists().map_err(io)?
        && !path.join("reservation.bin").try_exists().map_err(io)?
        && receipt.storage.as_ref().unwrap().retained_bytes == usage(path)?)
}
fn terminal(path: &Path) -> Result<bool> {
    // Recovery supersedes completion, including an unfinished reconciliation.
    for name in ["recovery.json", "completion.json"] {
        let p = path.join(name);
        if p.try_exists().map_err(io)? {
            return finalized(path, &read_receipt(&p)?);
        }
    }
    Ok(false)
}
pub(super) fn prune(root: &Path, incoming: bool) -> Result<()> {
    let mut paths = jobs(root)?;
    let extra = if incoming { JOB_BYTES + BLOCK } else { 0 };
    let count = if incoming { JOURNALS - 1 } else { JOURNALS };
    while paths.len() > count || root_usage(root)? + extra > ROOT_BYTES {
        let index = paths
            .iter()
            .position(|p| terminal(p).unwrap_or(false))
            .ok_or_else(|| {
                error(
                    ErrorCode::LimitExceeded,
                    "sandbox storage quota; recover pending jobs",
                )
            })?;
        let old = paths.remove(index);
        usage(&old)?;
        fs::remove_dir_all(old).map_err(io)?;
        File::open(root).map_err(io)?.sync_all().map_err(io)?;
    }
    Ok(())
}

/// Explicit accounting authority. Only the controller chooses root/job/mode;
/// asset names never participate. The caller holds Lock for this context's life.
pub(super) struct Job {
    root: PathBuf,
    pub path: PathBuf,
    staging: bool,
    limit: u64,
    peak: Cell<u64>,
    workspace_peak: Cell<u64>,
}
impl Job {
    pub fn recovery(root: &Path, path: &Path) -> Result<Self> {
        Self::new(root, path, false, JOB_BYTES)
    }
    fn new(root: &Path, path: &Path, staging: bool, limit: u64) -> Result<Self> {
        checked_root(root)?;
        if path.parent() != Some(root)
            || !path
                .file_name()
                .is_some_and(|s| valid_name(&s.to_string_lossy()))
        {
            return Err(error(
                ErrorCode::ScopeViolation,
                "explicit direct journal child required",
            ));
        }
        usage(path)?;
        let job = Self {
            root: root.into(),
            path: path.into(),
            staging,
            limit,
            peak: Cell::new(0),
            workspace_peak: Cell::new(0),
        };
        if staging {
            job.observe()?;
        } else {
            // Old vulnerable journals may already exceed a ceiling. Recovery
            // may inspect/reclaim them, never use that permission to stage.
            job.peak.set(usage(path)?);
            job.workspace_peak.set(job.workspace()?);
        }
        Ok(job)
    }
    pub fn initial_receipt(&self) -> StorageReceipt {
        StorageReceipt {
            reserved_bytes: JOB_BYTES - BLOCK,
            peak_charged_bytes: self.peak.get(),
            workspace_bytes_before_reclaim: 0,
            retained_bytes: 0,
            root_retained_bytes: 0,
            workspace_reclaimed: false,
            accounting_finalized: false,
            capacity_guaranteed: false,
            high_water_complete: true,
            workspace_peak_charged_bytes: 0,
            workspace_limit_bytes: self.limit,
            measurement_scope:
                "controller_allocation_boundaries_including_preallocation; managed_journals_only"
                    .into(),
        }
    }
    pub fn workspace(&self) -> Result<u64> {
        let mut bytes = 0;
        for name in ["job", "guard"] {
            let p = self.path.join(name);
            if p.try_exists().map_err(io)? {
                bytes += usage(&p)?;
            }
        }
        Ok(bytes)
    }
    pub fn observe(&self) -> Result<()> {
        let bytes = usage(&self.path)?;
        let workspace = self.workspace()?;
        self.peak.set(self.peak.get().max(bytes));
        self.workspace_peak
            .set(self.workspace_peak.get().max(workspace));
        if bytes > JOB_BYTES || root_usage(&self.root)? > ROOT_BYTES || workspace > self.limit {
            return Err(error(
                ErrorCode::LimitExceeded,
                "sandbox charged storage exceeded",
            ));
        }
        Ok(())
    }
    fn target(&self, path: &Path) -> Result<bool> {
        let relative = path
            .strip_prefix(&self.path)
            .map_err(|_| error(ErrorCode::ScopeViolation, "write outside explicit journal"))?;
        if relative.as_os_str().is_empty()
            || relative
                .components()
                .any(|c| !matches!(c, Component::Normal(_)))
        {
            return Err(error(
                ErrorCode::ScopeViolation,
                "nonphysical journal write",
            ));
        }
        usage(&self.path)?;
        let mut p = self.path.clone();
        for c in relative
            .components()
            .take(relative.components().count() - 1)
        {
            p.push(c);
            if !fs::symlink_metadata(&p).map_err(io)?.is_dir() {
                return Err(error(
                    ErrorCode::ScopeViolation,
                    "non-directory write parent",
                ));
            }
        }
        Ok(matches!(
            relative.components().next().unwrap().as_os_str().to_str(),
            Some("job" | "guard")
        ))
    }
    fn consume(&self, bytes: u64) -> Result<()> {
        let p = self.path.join("reservation.bin");
        // Absence is allowed ONLY in an explicit recovery context.
        if !self.staging && !p.try_exists().map_err(io)? {
            return Ok(());
        }
        let f = OpenOptions::new()
            .write(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(p)
            .map_err(io)?;
        let m = f.metadata().map_err(io)?;
        if !m.is_file() || m.nlink() != 1 || (self.staging && m.len() < bytes) {
            return Err(error(
                ErrorCode::LimitExceeded,
                "sandbox reservation exhausted",
            ));
        }
        let release = if self.staging {
            bytes
        } else {
            bytes.min(m.len())
        };
        f.set_len(m.len() - release).map_err(io)?;
        f.sync_all().map_err(io)
    }
    pub fn before_save(&self, path: &Path, bytes: usize) -> Result<()> {
        if bytes > MAX_CONTROL.max(1048576) {
            return Err(error(
                ErrorCode::LimitExceeded,
                "sandbox file bound exceeded",
            ));
        }
        let workspace = self.target(path)?;
        if workspace && !self.staging {
            return Err(error(
                ErrorCode::ScopeViolation,
                "recovery cannot stage workspace data",
            ));
        }
        let charge = (bytes as u64).div_ceil(BLOCK) * BLOCK + BLOCK;
        if workspace && self.workspace()? + charge > self.limit {
            return Err(error(
                ErrorCode::LimitExceeded,
                "study staging quota exceeded",
            ));
        }
        // Release only this operation's charge immediately before its allocation.
        // A competing filesystem user can still consume freed extents: no promise
        // of underlying disk reservation is made.
        self.consume(charge)?;
        if usage(&self.path)? + charge > JOB_BYTES || root_usage(&self.root)? + charge > ROOT_BYTES
        {
            return Err(error(
                ErrorCode::LimitExceeded,
                "sandbox job/root storage exceeded",
            ));
        }
        Ok(())
    }
    pub fn mkdir(&self, path: &Path) -> Result<()> {
        if path == self.path {
            return Ok(());
        }
        if path.try_exists().map_err(io)? {
            self.target(path)?;
            if !fs::symlink_metadata(path).map_err(io)?.is_dir() {
                return Err(error(
                    ErrorCode::ScopeViolation,
                    "staging directory collision",
                ));
            }
            return Ok(());
        }
        self.mkdir(
            path.parent()
                .ok_or_else(|| error(ErrorCode::ScopeViolation, "directory parent required"))?,
        )?;
        self.before_save(path, 0)?;
        fs::DirBuilder::new().mode(0o755).create(path).map_err(io)?;
        self.observe()
    }
}

pub(super) fn reserve(root: &Path, path: &Path, workspace_limit: u64) -> Result<Job> {
    if workspace_limit == 0 || workspace_limit > JOB_BYTES {
        return Err(error(
            ErrorCode::LimitExceeded,
            "staging quota must be 1..=STORAGE_JOB_BYTES charged bytes",
        ));
    }
    if path.parent() != Some(root)
        || !path
            .file_name()
            .is_some_and(|s| valid_name(&s.to_string_lossy()))
    {
        return Err(error(
            ErrorCode::ScopeViolation,
            "invalid reservation target",
        ));
    }
    prune(root, true)?;
    fs::DirBuilder::new().mode(0o700).create(path).map_err(io)?;
    let reservation = path.join("reservation.bin");
    let f = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(&reservation)
        .map_err(io)?;
    // SAFETY: live writable descriptor, bounded positive offset/length.
    let code =
        unsafe { libc::posix_fallocate(f.as_raw_fd(), 0, (JOB_BYTES - BLOCK) as libc::off_t) };
    if code != 0 {
        fs::remove_file(reservation).map_err(io)?;
        fs::remove_dir(path).map_err(io)?;
        return Err(io(std::io::Error::from_raw_os_error(code)));
    }
    f.sync_all().map_err(io)?;
    File::open(path).map_err(io)?.sync_all().map_err(io)?;
    File::open(root).map_err(io)?.sync_all().map_err(io)?;
    Job::new(root, path, true, workspace_limit)
}

pub(super) fn reclaim(path: &Path) -> Result<()> {
    // Confirm absence before calling. Validate the ENTIRE tree before deletion.
    usage(path)?;
    for name in ["job", "guard"] {
        let p = path.join(name);
        if p.try_exists().map_err(io)? {
            if !fs::symlink_metadata(&p).map_err(io)?.is_dir() {
                return Err(error(
                    ErrorCode::ScopeViolation,
                    "invalid workspace directory",
                ));
            }
            fs::remove_dir_all(p).map_err(io)?;
        }
    }
    let reserve = path.join("reservation.bin");
    if reserve.try_exists().map_err(io)? {
        fs::remove_file(reserve).map_err(io)?;
    }
    File::open(path).map_err(io)?.sync_all().map_err(io)
}

#[cfg(test)]
thread_local! { pub(super) static CRASH_STEP: Cell<usize> = const { Cell::new(0) }; }
fn receipt_step(step: usize) {
    #[cfg(test)]
    CRASH_STEP.with(|target| {
        if target.get() == step {
            std::process::exit(87);
        }
    });
    let _ = step;
}

pub(super) fn finish(job: &Job, receipt: &mut Receipt, recovery: bool) -> Result<()> {
    let path = &job.path;
    let prior = receipt.storage.clone();
    // Legacy peak/reservation values were synthetic; do not relabel them as
    // measured samples from this implementation.
    let measured_prior = prior.as_ref().filter(|s| !s.measurement_scope.is_empty());
    let workspace = job.workspace()?.max(
        prior
            .as_ref()
            .map_or(0, |s| s.workspace_bytes_before_reclaim),
    );
    receipt.storage = Some(StorageReceipt {
        reserved_bytes: if recovery {
            measured_prior.map_or(0, |s| s.reserved_bytes)
        } else {
            JOB_BYTES - BLOCK
        },
        peak_charged_bytes: job
            .peak
            .get()
            .max(measured_prior.map_or(0, |s| s.peak_charged_bytes)),
        workspace_bytes_before_reclaim: workspace,
        retained_bytes: 0,
        root_retained_bytes: 0,
        workspace_reclaimed: false,
        accounting_finalized: false,
        capacity_guaranteed: false,
        high_water_complete: !recovery || measured_prior.is_some_and(|s| s.high_water_complete),
        workspace_peak_charged_bytes: job
            .workspace_peak
            .get()
            .max(measured_prior.map_or(0, |s| s.workspace_peak_charged_bytes)),
        workspace_limit_bytes: prior
            .as_ref()
            .map_or(job.limit, |s| s.workspace_limit_bytes),
        measurement_scope:
            "controller_allocation_boundaries_including_preallocation; managed_journals_only".into(),
    });
    let target = path.join(if recovery {
        "recovery.json"
    } else {
        "completion.json"
    });
    let tmp = path.join("receipt.tmp");
    // Always use explicit recovery mode for final evidence, after staging ends.
    let evidence = Job::recovery(&job.root, path)?;
    if receipt.cleanup_confirmed
        && (usage(path)? > JOB_BYTES || root_usage(&job.root)? > ROOT_BYTES)
    {
        reclaim(path)?;
    }
    if tmp.try_exists().map_err(io)? {
        fs::remove_file(&tmp).map_err(io)?;
    }
    record(&evidence, &tmp, receipt)?;
    receipt_step(1);
    fs::rename(&tmp, &target).map_err(io)?;
    receipt_step(2);
    File::open(path).map_err(io)?.sync_all().map_err(io)?;
    receipt_step(3);
    if receipt.cleanup_confirmed {
        reclaim(path)?;
    }
    receipt_step(4);
    receipt.storage.as_mut().unwrap().workspace_reclaimed = receipt.cleanup_confirmed;
    // Fixed-size JSON (trailing whitespace) removes self-referential size
    // convergence. No terminal receipt with zero placeholder measurements exists.
    let mut bytes = serde_json::to_vec(receipt)
        .map_err(|_| error(ErrorCode::InvalidJson, "receipt encoding"))?;
    if bytes.len() > MAX_CONTROL {
        return Err(error(ErrorCode::LimitExceeded, "receipt too large"));
    }
    bytes.resize(MAX_CONTROL, b' ');
    evidence.before_save(&tmp, bytes.len())?;
    let mut f = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(&tmp)
        .map_err(io)?;
    f.write_all(&bytes).map_err(io)?;
    f.sync_all().map_err(io)?;
    evidence.observe()?;
    receipt_step(5);
    let old = allocated(&target)?;
    let retained = usage(path)? - old;
    let total = root_usage(&job.root)? - old;
    let s = receipt.storage.as_mut().unwrap();
    s.retained_bytes = retained;
    s.root_retained_bytes = total;
    s.peak_charged_bytes = s.peak_charged_bytes.max(evidence.peak.get());
    s.accounting_finalized = receipt.cleanup_confirmed;
    bytes = serde_json::to_vec(receipt)
        .map_err(|_| error(ErrorCode::InvalidJson, "receipt encoding"))?;
    if bytes.len() > MAX_CONTROL {
        return Err(error(ErrorCode::LimitExceeded, "receipt too large"));
    }
    bytes.resize(MAX_CONTROL, b' ');
    std::io::Seek::rewind(&mut f).map_err(io)?;
    f.write_all(&bytes).map_err(io)?;
    receipt_step(6);
    f.sync_all().map_err(io)?;
    receipt_step(7);
    fs::rename(&tmp, &target).map_err(io)?;
    receipt_step(8);
    File::open(path).map_err(io)?.sync_all().map_err(io)?;
    receipt_step(9);
    if receipt.cleanup_confirmed && !finalized(path, receipt)? {
        return Err(error(
            ErrorCode::IntegrityMismatch,
            "storage receipt allocation mismatch",
        ));
    }
    Ok(())
}
