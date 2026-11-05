//! Qualified fixed trusted PyTorch process; NOT an untrusted gym sandbox.
//! Rust owns source checks, limits, lifecycle, parity and result eligibility.
use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits as JsonLimits};
use nf_core::error::{CoreError, ErrorCode, Result};
use nf_core::schemas::DType;
use nf_surgeon::cancellation::CancellationToken;
use nf_surgeon::transaction::{ExecutionLimits, SourceState};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt};
use std::os::unix::process::CommandExt;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{Duration, Instant};

const CLEANUP_LIMIT: Duration = Duration::from_millis(500);

const SCRIPT: &[u8] = include_bytes!("model_worker/worker.py");
const MAX_CONTROL: usize = 1048576;
fn error(code: ErrorCode, text: &str) -> CoreError {
    CoreError::new(code, "/model_worker", text)
}
fn io(e: impl std::fmt::Display) -> CoreError {
    error(ErrorCode::Io, &e.to_string())
}
fn canonical(value: &impl Serialize) -> Result<Vec<u8>> {
    Ok(CanonicalDocument::parse(
        &serde_json::to_vec(value).map_err(io)?,
        JsonLimits::default(),
    )?
    .bytes()
    .to_vec())
}
fn boot_ms() -> i64 {
    let t = rustix::time::clock_gettime(rustix::time::ClockId::Boottime);
    t.tv_sec * 1000 + t.tv_nsec / 1000000
}
fn hash_file(path: &Path, max: u64) -> Result<ContentDigest> {
    let mut f = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .map_err(io)?;
    let m = f.metadata().map_err(io)?;
    if !m.is_file() || m.nlink() != 1 || m.len() > max {
        return Err(error(
            ErrorCode::ScopeViolation,
            "bounded regular runtime file required",
        ));
    }
    let mut data = Vec::new();
    (&mut f).take(max + 1).read_to_end(&mut data).map_err(io)?;
    if data.len() as u64 > max {
        return Err(error(
            ErrorCode::LimitExceeded,
            "runtime file grew beyond bound",
        ));
    }
    Ok(ContentDigest::of_bytes(&data))
}
fn save(path: &Path, bytes: &[u8], mode: u32) -> Result<()> {
    let mut f = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(mode)
        .open(path)
        .map_err(io)?;
    f.write_all(bytes).map_err(io)?;
    f.sync_all().map_err(io)
}
fn record(path: &Path, value: &impl Serialize) -> Result<()> {
    save(path, &canonical(value)?, 0o400)
}
fn private_root(root: &Path) -> Result<()> {
    if !root.is_absolute() || root.parent().is_none() {
        return Err(error(
            ErrorCode::ScopeViolation,
            "explicit private absolute runtime root required",
        ));
    }
    let mut p = PathBuf::new();
    for c in root.components() {
        if !matches!(c, Component::RootDir | Component::Normal(_)) {
            return Err(error(
                ErrorCode::ScopeViolation,
                "runtime root traversal forbidden",
            ));
        }
        p.push(c);
        let m = fs::symlink_metadata(&p).map_err(io)?;
        if !m.is_dir() || m.file_type().is_symlink() {
            return Err(error(
                ErrorCode::ScopeViolation,
                "physical runtime directories required",
            ));
        }
    }
    let m = fs::metadata(root).map_err(io)?;
    // SAFETY: geteuid reads process metadata and has no pointer preconditions.
    if m.uid() != unsafe { libc::geteuid() } || m.mode() & 0o077 != 0 {
        return Err(error(
            ErrorCode::PermissionDenied,
            "runtime root must be owner-held0700",
        ));
    }
    Ok(())
}
fn name() -> Result<String> {
    let mut random = [0_u8; 16];
    File::open("/dev/urandom")
        .map_err(io)?
        .read_exact(&mut random)
        .map_err(io)?;
    Ok(ContentDigest::of_bytes(&random).as_str()[7..].to_string())
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    pub cpu_seconds: u64,
    pub wall_ms: u64,
    pub address_space_bytes: u64,
    pub output_bytes: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            cpu_seconds: 10,
            wall_ms: 15000,
            address_space_bytes: 2147483648,
            output_bytes: 262144,
        }
    }
}
impl Limits {
    fn validate(&self) -> Result<()> {
        if !(1..=30).contains(&self.cpu_seconds)
            || !(100..=30000).contains(&self.wall_ms)
            || !(536870912..=4294967296).contains(&self.address_space_bytes)
            || !(1024..=MAX_CONTROL).contains(&self.output_bytes)
        {
            return Err(error(
                ErrorCode::LimitExceeded,
                "unsupported finite CPU model worker limits",
            ));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Receipt {
    pub schema_version: u32,
    pub job: String,
    pub manifest: ContentDigest,
    pub checkpoint: ContentDigest,
    pub inputs: ContentDigest,
    pub seed_schedule: ContentDigest,
    pub parameters: ContentDigest,
    pub runtime: ContentDigest,
    pub status: String,
    pub cleanup_confirmed: bool,
    pub outputs_eligible: bool,
    pub promotion_status: String,
    pub outputs: Vec<Vec<f64>>,
    pub child_report: Option<Value>,
    pub cpu_ms: Option<u64>,
    pub peak_rss_bytes: Option<u64>,
    pub wall_ms: u64,
    pub measurement_scope: String,
}
/// An actual run witness, not mintable from a deserialized success receipt.
pub struct Inference {
    receipt: Receipt,
}
impl Inference {
    pub fn receipt(&self) -> &Receipt {
        &self.receipt
    }
}
/// Non-deserializable runtime capability, operator-selected dependency paths only.
pub struct QualifiedTorch {
    root: PathBuf,
    python: PathBuf,
    python_hash: ContentDigest,
    site: PathBuf,
    extension: PathBuf,
    extension_hash: ContentDigest,
    prlimit_hash: ContentDigest,
    identity: ContentDigest,
}
fn pidfd(pid: u32) -> Result<OwnedFd> {
    // SAFETY: pidfd_open takes only scalar arguments and returns a fresh FD.
    let fd = unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0) };
    if fd < 0 {
        return Err(error(
            ErrorCode::UnsupportedCapability,
            "kernel pidfd identity required for model workers",
        ));
    }
    // SAFETY: successful pidfd_open returned a unique owned descriptor.
    Ok(unsafe { OwnedFd::from_raw_fd(fd as i32) })
}
struct OwnedChild {
    child: std::process::Child,
    pidfd: Option<OwnedFd>,
    reaped: bool,
    cleanup_deadline: Option<Instant>,
}
impl OwnedChild {
    fn signal(&mut self) -> std::io::Result<()> {
        if let Some(fd) = &self.pidfd {
            // SAFETY: the owned pidfd pins the exact child, not a recycled PID.
            if unsafe {
                libc::syscall(
                    libc::SYS_pidfd_send_signal,
                    fd.as_raw_fd(),
                    libc::SIGKILL,
                    std::ptr::null::<libc::siginfo_t>(),
                    0,
                )
            } < 0
            {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        } else {
            // Only used immediately after pidfd_open failure: the unreaped
            // std Child still owns this PID. Never fall back from a pidfd error.
            self.child.kill()
        }
    }
    fn try_reap(&mut self) -> std::io::Result<Option<(i32, libc::rusage)>> {
        let mut status = 0;
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
        let pid = self.child.id() as i32;
        // SAFETY: valid output pointers and our owned child; never blocks.
        let observed = unsafe { libc::wait4(pid, &mut status, libc::WNOHANG, usage.as_mut_ptr()) };
        if observed == pid {
            self.reaped = true;
            // SAFETY: successful wait4 initialized usage.
            return Ok(Some((status, unsafe { usage.assume_init() })));
        }
        if observed < 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(None)
    }
    /// A finite best effort, including EINTR and failed signals. Only a positive
    /// wait result proves reaping. Expiry/error remains unknown to callers.
    fn cleanup(&mut self) -> std::io::Result<(i32, libc::rusage)> {
        let until = *self
            .cleanup_deadline
            .get_or_insert_with(|| Instant::now() + CLEANUP_LIMIT);
        let signal_error = self.signal().err();
        for _ in 0..256 {
            match self.try_reap() {
                Ok(Some(measured)) => return Ok(measured),
                Err(e) if e.kind() != std::io::ErrorKind::Interrupted => {
                    // wait4 may be denied independently of waitid. Reap through
                    // the pidfd if possible, but NEVER invent rusage/success.
                    if let Some(fd) = &self.pidfd {
                        let mut info = std::mem::MaybeUninit::<libc::siginfo_t>::zeroed();
                        // SAFETY: valid output and owned handle; nonblocking.
                        let result = unsafe {
                            libc::waitid(
                                libc::P_PIDFD,
                                fd.as_raw_fd() as u32,
                                info.as_mut_ptr(),
                                libc::WEXITED | libc::WNOHANG,
                            )
                        };
                        if result == 0
                            && unsafe { info.assume_init().si_pid() } == self.child.id() as i32
                        {
                            self.reaped = true;
                            return Err(e);
                        }
                    }
                }
                _ => {}
            }
            if Instant::now() >= until {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        Err(std::io::Error::other(format!(
            "worker cleanup/reaping unconfirmed; signal error: {signal_error:?}"
        )))
    }
}
impl Drop for OwnedChild {
    fn drop(&mut self) {
        if !self.reaped {
            // Shared deadline prevents a failed explicit cleanup gaining a new
            // unbounded (or second full) wait during unwinding/early returns.
            let _ = self.cleanup();
        }
    }
}
struct CaptureStop(Arc<AtomicBool>);
impl Drop for CaptureStop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}
fn nonblocking(fd: &impl AsRawFd) -> Result<()> {
    // SAFETY: fcntl operates on an owned live pipe, no pointer arguments.
    let flags = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_GETFL) };
    if flags < 0
        || unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0
    {
        return Err(io(std::io::Error::last_os_error()));
    }
    Ok(())
}
fn finish_capture<T>(
    thread: std::thread::JoinHandle<std::io::Result<T>>,
    until: Instant,
) -> Result<T> {
    while !thread.is_finished() {
        if Instant::now() >= until {
            return Err(error(ErrorCode::Io, "worker capture shutdown unconfirmed"));
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    thread
        .join()
        .map_err(|_| error(ErrorCode::InternalPanic, "worker capture"))?
        .map_err(io)
}
struct Captured {
    success: bool,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    cpu_ms: u64,
    rss_bytes: u64,
    wall_ms: u64,
    status: String,
}
fn capture(
    mut input: impl Read,
    limit: usize,
    overflow: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
) -> std::io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    let mut block = [0_u8; 4096];
    loop {
        let n = match input.read(&mut block) {
            Ok(n) => n,
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
                ) =>
            {
                if stop.load(Ordering::Acquire) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(2));
                continue;
            }
            Err(e) => return Err(e),
        };
        if n == 0 {
            break;
        }
        if bytes.len() + n > limit {
            overflow.store(true, Ordering::Release);
            bytes.extend_from_slice(&block[..limit.saturating_sub(bytes.len())]);
            break;
        }
        bytes.extend_from_slice(&block[..n]);
    }
    Ok(bytes)
}
impl QualifiedTorch {
    pub fn qualify(
        enabled: bool,
        python: &Path,
        site: &Path,
        extension: &Path,
        root: &Path,
    ) -> Result<Self> {
        if !enabled {
            return Err(error(
                ErrorCode::PermissionDenied,
                "TRUSTED_MODEL_WORKER_DISABLED",
            ));
        }
        private_root(root)?;
        drop(pidfd(std::process::id())?);
        let python = fs::canonicalize(python).map_err(io)?;
        let site = fs::canonicalize(site).map_err(io)?;
        let extension = fs::canonicalize(extension).map_err(io)?;
        if !site.is_dir() || !python.is_absolute() {
            return Err(error(
                ErrorCode::ScopeViolation,
                "operator-approved interpreter/site directory required",
            ));
        }
        let python_hash = hash_file(&python, 33554432)?;
        let extension_hash = hash_file(&extension, 134217728)?;
        let prlimit_hash = hash_file(Path::new("/usr/bin/prlimit"), 1048576)?;
        let runtime = root.join(format!("runtime-{}", name()?));
        fs::DirBuilder::new()
            .mode(0o700)
            .create(&runtime)
            .map_err(io)?;
        save(&runtime.join("worker.py"), SCRIPT, 0o500)?;
        let data = fs::read(extension).map_err(io)?;
        save(&runtime.join("_native.abi3.so"), &data, 0o500)?;
        let mut value = Self {
            root: runtime,
            python,
            python_hash,
            site,
            extension: PathBuf::new(),
            extension_hash,
            prlimit_hash,
            identity: ContentDigest::of_bytes(b"pending-trusted-torch-proof"),
        };
        value.extension = value.root.join("_native.abi3.so");
        let job = value.new_job()?;
        record(&job.join("request.json"), &json!({"mode":"probe"}))?;
        // Cold imports from an operator-approved mounted site may be I/O-bound.
        // Qualification gets the supported finite wall ceiling, not more CPU or
        // memory; ordinary jobs still use their caller's unchanged limits.
        let limits = Limits {
            wall_ms: 30_000,
            ..Limits::default()
        };
        let run = value.launch(&job, &limits, &CancellationToken::default())?;
        if !run.success {
            record(
                &job.join("probe-failed.json"),
                &json!({"status":run.status,"stderr":String::from_utf8_lossy(&run.stderr)}),
            )?;
            return Err(error(
                ErrorCode::UnsupportedCapability,
                &format!(
                    "actual trusted Torch runtime probe failed: {}; stderr: {}",
                    run.status,
                    String::from_utf8_lossy(&run.stderr)
                        .chars()
                        .take(2048)
                        .collect::<String>()
                ),
            ));
        }
        let probe: Value = serde_json::from_slice(&run.stdout)
            .map_err(|_| error(ErrorCode::InvalidJson, "invalid model runtime probe"))?;
        if probe["kind"] != "trusted_torch_cpu_probe"
            || probe["torch_version"] != "2.9.1+cpu"
            || probe["native_version"] != nf_core::VERSION
            || probe["python"][0] != 3
            || probe["python"][1] != 12
            || probe["threads"] != 1
            || probe["interop_threads"] != 1
            || probe["kernel_threads"] != 1
            || probe["gradient_enabled"] != false
            || probe["credential_marker_present"] != false
            || probe["cpu_limit"] != json!([10, 10])
            || probe["as_limit"] != json!([2147483648_u64, 2147483648_u64])
            || probe["fd_limit"] != json!([64, 64])
            || probe["core_limit"] != json!([0, 0])
        {
            return Err(error(
                ErrorCode::UnsupportedCapability,
                "trusted Torch worker did not enforce pinned runtime/kernel limits",
            ));
        }
        value.identity = ContentDigest::of_bytes(&canonical(
            &json!({"schema_version":1,"python":value.python_hash,"native":value.extension_hash,"adapter":ContentDigest::of_bytes(SCRIPT),"prlimit":value.prlimit_hash,"probe":probe}),
        )?);
        record(
            &value.root.join("qualification.json"),
            &json!({"runtime":value.identity,"probe":probe,"assurance":"operator_trusted_fixed_model_adapter_not_hostile_gym_sandbox"}),
        )?;
        Ok(value)
    }
    pub fn identity(&self) -> &ContentDigest {
        &self.identity
    }
    fn new_job(&self) -> Result<PathBuf> {
        private_root(&self.root)?;
        let path = self.root.join(format!("job-{}", name()?));
        fs::DirBuilder::new()
            .mode(0o700)
            .create(&path)
            .map_err(io)?;
        fs::DirBuilder::new()
            .mode(0o700)
            .create(path.join("tmp"))
            .map_err(io)?;
        Ok(path)
    }
    fn verify_runtime(&self) -> Result<()> {
        private_root(&self.root)?;
        if hash_file(&self.python, 33554432)? != self.python_hash
            || hash_file(&self.extension, 134217728)? != self.extension_hash
            || hash_file(&self.root.join("worker.py"), 65536)? != ContentDigest::of_bytes(SCRIPT)
            || hash_file(Path::new("/usr/bin/prlimit"), 1048576)? != self.prlimit_hash
        {
            return Err(error(
                ErrorCode::IntegrityMismatch,
                "qualified model runtime files changed",
            ));
        }
        Ok(())
    }
    fn launch(&self, job: &Path, limits: &Limits, cancel: &CancellationToken) -> Result<Captured> {
        self.launch_registered(job, limits, cancel, &mut |_| Ok(()))
    }
    /// Register the actual, still-gated exec child using its owned kernel handle.
    /// The callback is trusted native code, never a PID/report supplied by a job.
    fn launch_registered(
        &self,
        job: &Path,
        limits: &Limits,
        cancel: &CancellationToken,
        register: &mut dyn FnMut(&OwnedFd) -> Result<()>,
    ) -> Result<Captured> {
        limits.validate()?;
        self.verify_runtime()?;
        cancel.check()?;
        let parent = std::process::id();
        let mut command = Command::new("/usr/bin/prlimit");
        command.args([
            format!("--cpu={}:{}", limits.cpu_seconds, limits.cpu_seconds),
            format!(
                "--as={}:{}",
                limits.address_space_bytes, limits.address_space_bytes
            ),
            "--nofile=64:64".into(),
            "--core=0:0".into(),
            "--fsize=1048576:1048576".into(),
            "--".into(),
        ]);
        command
            .arg(&self.python)
            .args(["-I", "-S", "-B"])
            .arg(self.root.join("worker.py"))
            .arg(&self.extension)
            .arg(&self.site)
            .arg(job)
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("OMP_NUM_THREADS", "1")
            .env("MKL_NUM_THREADS", "1")
            .env("OPENBLAS_NUM_THREADS", "1")
            .env("NUMEXPR_NUM_THREADS", "1")
            .env("TMPDIR", job.join("tmp"))
            .current_dir(job)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .process_group(0);
        // SAFETY: fixed async-signal-safe syscalls only, no Python/callback/alloc.
        unsafe {
            command.pre_exec(move || {
                if libc::syscall(
                    libc::SYS_close_range,
                    3_u32,
                    u32::MAX,
                    libc::CLOSE_RANGE_CLOEXEC,
                ) != 0
                {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::getppid() != parent as i32 {
                    return Err(std::io::Error::from_raw_os_error(libc::ESRCH));
                }
                Ok(())
            });
        }
        let start = boot_ms();
        let mut owned = OwnedChild {
            child: command.spawn().map_err(io)?,
            pidfd: None,
            reaped: false,
            cleanup_deadline: None,
        };
        let descriptor = match pidfd(owned.child.id()) {
            Ok(fd) => fd,
            Err(e) => {
                // Drop closes the gate and performs bounded cleanup even if
                // kill/wait are denied. No measured result or witness escapes.
                drop(owned.child.stdin.take());
                return Err(e);
            }
        };
        owned.pidfd = Some(descriptor);
        let pid = owned.child.id() as i32;
        // Command::spawn has completed exec (no pre_exec pipe deadlock). The
        // fixed Python bootstrap blocks on stdin BEFORE imports or job reads.
        // prlimit/Python/bootstrap are this same child, not a synthetic proxy.
        let gate_result = (|| {
            record(
                &job.join("process.json"),
                &json!({"pid":pid,"group":pid,"runtime":self.identity,"limits":limits,"start_boot_ms":start}),
            )?;
            cancel.check()?;
            register(owned.pidfd.as_ref().unwrap())?;
            cancel.check()?;
            if boot_ms() - start >= limits.wall_ms as i64 {
                return Err(error(
                    ErrorCode::LimitExceeded,
                    "deadline before work gate release",
                ));
            }
            owned
                .child
                .stdin
                .take()
                .ok_or_else(|| error(ErrorCode::Io, "missing work gate"))?
                .write_all(b"G")
                .map_err(io)
        })();
        // Failed registration/gate release follows the SAME wait4 measurement
        // path as cancellation. Never discard real CPU/RSS or claim no spawn.
        let gate_failure = gate_result.err().map(|e| e.to_string());
        drop(owned.child.stdin.take());
        let overflow = Arc::new(AtomicBool::new(false));
        let o = overflow.clone();
        let e = overflow.clone();
        let limit = limits.output_bytes;
        let stdout = owned.child.stdout.take().unwrap();
        let stderr = owned.child.stderr.take().unwrap();
        nonblocking(&stdout)?;
        nonblocking(&stderr)?;
        let capture_stop = CaptureStop(Arc::new(AtomicBool::new(false)));
        let stop_out = capture_stop.0.clone();
        let stop_err = capture_stop.0.clone();
        let out = std::thread::spawn(move || capture(stdout, limit, o, stop_out));
        let err = std::thread::spawn(move || capture(stderr, 65536, e, stop_err));
        let mut reason = "completed";
        let (status, usage) = loop {
            match owned.try_reap() {
                Ok(Some(measured)) => break measured,
                Err(cause) if cause.kind() != std::io::ErrorKind::Interrupted => {
                    return Err(io(cause))
                }
                // EINTR still passes through cancellation/deadline checks.
                _ => {}
            }
            if gate_failure.is_some()
                || cancel.is_cancelled()
                || boot_ms() - start >= limits.wall_ms as i64
                || overflow.load(Ordering::Acquire)
            {
                reason = if gate_failure.is_some() {
                    "launch_gate_failed"
                } else if cancel.is_cancelled() {
                    "cancelled"
                } else if overflow.load(Ordering::Acquire) {
                    "output_limit"
                } else {
                    "wall_limit"
                };
                break owned.cleanup().map_err(io)?;
            }
            std::thread::sleep(Duration::from_millis(5));
        };
        // Nonblocking capture can always stop, even if an unexpected inherited
        // writer keeps a pipe open. Bound joins too; never mint a partial witness.
        drop(capture_stop);
        let until = Instant::now() + CLEANUP_LIMIT;
        let stdout = finish_capture(out, until)?;
        let stderr = finish_capture(err, until)?;
        let cpu = (usage.ru_utime.tv_sec as u64 + usage.ru_stime.tv_sec as u64) * 1000
            + (usage.ru_utime.tv_usec as u64 + usage.ru_stime.tv_usec as u64) / 1000;
        let wall = (boot_ms() - start).max(0) as u64;
        if gate_failure.is_some() {
            reason = "launch_gate_failed";
        }
        if reason == "completed" && overflow.load(Ordering::Acquire) {
            reason = "output_limit";
        }
        if reason == "completed" && cpu > limits.cpu_seconds * 1000 {
            reason = "cpu_limit";
        }
        let success = reason == "completed"
            && libc::WIFEXITED(status)
            && libc::WEXITSTATUS(status) == 0
            && !overflow.load(Ordering::Acquire)
            && wall <= limits.wall_ms;
        Ok(Captured {
            success,
            stdout,
            stderr,
            cpu_ms: cpu,
            rss_bytes: usage.ru_maxrss.max(0) as u64 * 1024,
            wall_ms: wall,
            status: if let Some(failure) = gate_failure {
                format!("launch_gate_failed: {failure}")
            } else if success {
                "completed".into()
            } else if reason == "completed" {
                "worker_failed".into()
            } else {
                reason.into()
            },
        })
    }
    fn stage_source(&self, source: &SourceState) -> Result<(PathBuf, BTreeMap<String, String>)> {
        let job = self.new_job()?;
        save(
            &job.join("manifest.json"),
            &canonical(&**source.manifest())?,
            0o400,
        )?;
        save(
            &job.join("checkpoint.safetensors"),
            source.checkpoint_bytes(),
            0o400,
        )?;
        fs::DirBuilder::new()
            .mode(0o700)
            .create(job.join("assets"))
            .map_err(io)?;
        let mut assets = BTreeMap::new();
        for (name, data) in source.assets() {
            let filename = ContentDigest::of_bytes(name.as_bytes()).as_str()[7..].to_owned();
            save(&job.join("assets").join(&filename), data, 0o400)?;
            assets.insert(name.clone(), filename);
        }
        Ok((job, assets))
    }
    pub fn infer(
        &self,
        source: &SourceState,
        inputs: Vec<Vec<f64>>,
        limits: Limits,
        cancel: &CancellationToken,
    ) -> Result<Inference> {
        let seeds = vec![0; inputs.len()];
        self.infer_seeded(source, inputs, seeds, limits, cancel)
    }
    pub fn infer_seeded(
        &self,
        source: &SourceState,
        inputs: Vec<Vec<f64>>,
        seeds: Vec<u64>,
        limits: Limits,
        cancel: &CancellationToken,
    ) -> Result<Inference> {
        limits.validate()?;
        cancel.check()?;
        source.verify_integrity()?;
        let expected = preflight(source, &inputs)?;
        if seeds.len() != inputs.len() {
            return Err(error(
                ErrorCode::InvalidValue,
                "one explicit seed per matched task required",
            ));
        }
        let (job, assets) = self.stage_source(source)?;
        let job_name = job.file_name().unwrap().to_string_lossy().into_owned();
        record(
            &job.join("request.json"),
            &json!({"schema_version":1,"mode":"infer","assets":assets,"inputs":inputs,"seeds":seeds}),
        )?;
        let mut receipt=Receipt{schema_version:2,job:job_name,manifest:source.manifest_digest().clone(),checkpoint:ContentDigest::of_bytes(source.checkpoint_bytes()),inputs:ContentDigest::of_bytes(&canonical(&inputs)?),seed_schedule:ContentDigest::of_bytes(&canonical(&seeds)?),parameters:source.manifest().parameter_digest.clone(),runtime:self.identity.clone(),status:"incomplete".into(),cleanup_confirmed:false,outputs_eligible:false,promotion_status:"not_evaluated".into(),outputs:Vec::new(),child_report:None,cpu_ms:None,peak_rss_bytes:None,wall_ms:0,measurement_scope:"kernel_wait4_whole_single_trusted_process_all_threads;AS_hard_cap_not_RSS_limit;runtime_setup_outside_process_unknown".into()};
        record(&job.join("intent.json"), &receipt)?;
        let captured = self.launch(&job, &limits, cancel)?;
        receipt.cleanup_confirmed = true;
        receipt.cpu_ms = Some(captured.cpu_ms);
        receipt.peak_rss_bytes = Some(captured.rss_bytes);
        receipt.wall_ms = captured.wall_ms;
        receipt.status = captured.status;
        if captured.success {
            let doc = CanonicalDocument::parse(
                &captured.stdout,
                JsonLimits {
                    max_bytes: limits.output_bytes,
                    max_depth: 32,
                    max_nodes: 20000,
                },
            )?;
            verify_report(source, doc.value(), &inputs, &seeds, &expected)?;
            receipt.outputs = serde_json::from_value(doc.value()["outputs"].clone())
                .map_err(|_| error(ErrorCode::InvalidValue, "malformed worker outputs"))?;
            receipt.child_report = Some(doc.value().clone());
            receipt.status = "verified_inference".into();
        }
        source.verify_integrity()?;
        self.verify_runtime()?;
        record(&job.join("completion.json"), &receipt)?;
        record(
            &job.join("stderr.json"),
            &json!({"digest":ContentDigest::of_bytes(&captured.stderr),"bytes":captured.stderr.len()}),
        )?;
        Ok(Inference { receipt })
    }
}
fn scalar(source: &SourceState, name: &str, index: usize) -> Result<f64> {
    let t = source.tensor(name).ok_or_else(|| {
        error(
            ErrorCode::UnsupportedCapability,
            "floating model tensor missing",
        )
    })?;
    t.f64_values()
        .and_then(|v| v.get(index).copied())
        .or_else(|| {
            t.f32_values()
                .and_then(|v| v.get(index).map(|v| f64::from(*v)))
        })
        .ok_or_else(|| {
            error(
                ErrorCode::TypeMismatch,
                "floating tensor index/type mismatch",
            )
        })
}
fn preflight(source: &SourceState, inputs: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    if source.payload_bytes()? > 1048576
        || inputs.is_empty()
        || inputs.len() > 64
        || inputs.iter().any(|v| {
            v.is_empty() || v.len() > 64 || v.iter().any(|n| !n.is_finite() || n.abs() > 1e6)
        })
    {
        return Err(error(
            ErrorCode::LimitExceeded,
            "bounded small CPU source/input inventory required",
        ));
    }
    let manifest = source.manifest();
    let config = CanonicalDocument::parse(&source.assets()["config"], JsonLimits::default())?;
    let factory = manifest.architecture.id.as_str();
    if manifest.architecture.version != "1" {
        return Err(error(
            ErrorCode::UnsupportedCapability,
            "pinned factory version required",
        ));
    }
    if factory == "nf.arithmetic.relu" {
        return inputs
            .iter()
            .map(|v| source.infer_arithmetic_fixture(v, ExecutionLimits::default()))
            .collect();
    }
    if !matches!(factory, "fixture.cpu" | "nf.fixture.linear_buffered") {
        return Err(error(
            ErrorCode::UnsupportedCapability,
            "remote/unknown model factory disabled",
        ));
    }
    let buffered = factory == "nf.fixture.linear_buffered";
    if (!buffered && config.value() != &json!({}))
        || (buffered
            && config.value()
                != &json!({"schema_version":1,"rows":2,"columns":2,"tied":true,"buffer":"offset"}))
    {
        return Err(error(
            ErrorCode::UnsupportedCapability,
            "factory config/layout must match registered schema",
        ));
    }
    if manifest
        .parameters
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>()
        != BTreeSet::from(["w", "tied"])
        || (!buffered && !manifest.buffers.is_empty())
        || (buffered
            && manifest
                .buffers
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>()
                != BTreeSet::from(["offset", "offset_alias"]))
    {
        return Err(error(
            ErrorCode::UnsupportedCapability,
            "actual trusted factory parameter/buffer inventory required",
        ));
    }
    let w = &manifest.parameters["w"];
    let tied = &manifest.parameters["tied"];
    if w.storage_group != tied.storage_group
        || w.shape != tied.shape
        || w.dtype != tied.dtype
        || w.shape.len() != 2
        || w.shape.iter().any(|n| *n == 0 || *n > 64)
        || !matches!(w.dtype, DType::F32 | DType::F64)
    {
        return Err(error(
            ErrorCode::UnsupportedCapability,
            "factory requires actual same-span tied dense floating matrices",
        ));
    }
    if buffered {
        let b = &manifest.buffers["offset"];
        let alias = &manifest.buffers["offset_alias"];
        if b.shape != vec![w.shape[0]]
            || b.dtype != w.dtype
            || b.storage_group != alias.storage_group
            || b.shape != alias.shape
            || b.dtype != alias.dtype
            || b.storage_group == w.storage_group
            || w.shape != vec![2, 2]
        {
            return Err(error(
                ErrorCode::UnsupportedCapability,
                "unsupported buffer/alias layout",
            ));
        }
    }
    let rows = w.shape[0] as usize;
    let cols = w.shape[1] as usize;
    inputs
        .iter()
        .map(|input| {
            if input.len() != cols {
                return Err(error(
                    ErrorCode::TypeMismatch,
                    "model input dimension mismatch",
                ));
            }
            let mut output = Vec::new();
            for row in 0..rows {
                let mut sum = 0_f64;
                for (col, x) in input.iter().enumerate() {
                    let value = scalar(source, "w", row * cols + col)?;
                    sum = if w.dtype == DType::F32 {
                        ((sum as f32) + (value as f32) * (*x as f32)) as f64
                    } else {
                        sum + value * x
                    };
                }
                if buffered {
                    let b = scalar(source, "offset", row)?;
                    sum = if w.dtype == DType::F32 {
                        ((sum as f32) + (b as f32)) as f64
                    } else {
                        sum + b
                    };
                }
                if !sum.is_finite() {
                    return Err(error(
                        ErrorCode::NonFinite,
                        "native model reference became nonfinite",
                    ));
                }
                output.push(sum);
            }
            Ok(output)
        })
        .collect()
}
#[cfg(all(test, target_os = "linux"))]
#[path = "model_worker/cleanup_tests.rs"]
mod cleanup_tests;
pub mod edit;
pub mod measurement;
pub mod reload;

fn verify_report(
    source: &SourceState,
    report: &Value,
    inputs: &[Vec<f64>],
    seeds: &[u64],
    expected: &[Vec<f64>],
) -> Result<()> {
    if report["schema_version"] != 2
        || report["kind"] != "trusted_torch_cpu_inference"
        || report["manifest"].as_str() != Some(source.manifest_digest().as_str())
        || report["parameters"].as_str() != Some(source.manifest().parameter_digest.as_str())
        || report["torch_version"] != "2.9.1+cpu"
        || report["inference_mode"] != true
        || report["gradient_enabled"] != false
        || report["training"] != false
        || report["threads"] != 1
        || report["interop_threads"] != 1
        || report["kernel_threads"] != 1
        || report["before"] != report["after"]
    {
        return Err(error(
            ErrorCode::IntegrityMismatch,
            "trusted worker state/mode/version receipt mismatch",
        ));
    }
    if report["factory"] != source.manifest().architecture.id
        || report["checkpoint"].as_str()
            != Some(ContentDigest::of_bytes(source.checkpoint_bytes()).as_str())
        || report["inputs"].as_str() != Some(ContentDigest::of_bytes(&canonical(&inputs)?).as_str())
        || report["seeds"] != json!(seeds)
        || report["copies"]["zero_copy"] != false
        || report["assurance"] != "trusted_fixed_model_worker_not_untrusted_gym_sandbox"
    {
        return Err(error(
            ErrorCode::IntegrityMismatch,
            "model input/checkpoint/copy/assurance receipt mismatch",
        ));
    }
    let mut group_bytes = BTreeMap::new();
    for spec in source
        .manifest()
        .parameters
        .values()
        .chain(source.manifest().buffers.values())
    {
        group_bytes.insert(&spec.storage_group, spec.byte_len()?);
    }
    let bytes: u64 = group_bytes.into_values().sum();
    for key in [
        "logical_group_bytes",
        "factory_tensor_bytes",
        "native_tensor_bytes_copy",
        "mutable_staging_bytes",
        "torch_clone_bytes",
    ] {
        if report["copies"][key].as_u64() != Some(bytes) {
            return Err(error(
                ErrorCode::IntegrityMismatch,
                "explicit model copy/allocation byte count mismatch",
            ));
        }
    }
    verify_state(source, &report["before"], None)?;
    let outputs: Vec<Vec<f64>> = serde_json::from_value(report["outputs"].clone())
        .map_err(|_| error(ErrorCode::InvalidValue, "bounded output matrix required"))?;
    let f32 = source
        .manifest()
        .parameters
        .values()
        .any(|s| s.dtype == DType::F32);
    let tolerance = if f32 { 1e-5 } else { 1e-10 };
    if outputs.len() != expected.len()
        || outputs.iter().zip(expected).any(|(a, b)| {
            a.len() != b.len()
                || a.iter()
                    .zip(b)
                    .any(|(a, b)| !a.is_finite() || (a - b).abs() > tolerance)
        })
    {
        return Err(error(
            ErrorCode::SmokeFailed,
            "actual Torch forward differs from native reference",
        ));
    }
    Ok(())
}
fn verify_state(
    source: &SourceState,
    state: &Value,
    hashes: Option<&BTreeMap<String, ContentDigest>>,
) -> Result<()> {
    let before = state
        .as_array()
        .ok_or_else(|| error(ErrorCode::InvalidValue, "actual runtime inventory missing"))?;
    let mut observed = BTreeMap::new();
    let mut actual_groups: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut expected_groups: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut group_hashes: BTreeMap<&str, &ContentDigest> = BTreeMap::new();
    for (role, specs) in [
        ("parameter", &source.manifest().parameters),
        ("buffer", &source.manifest().buffers),
    ] {
        for (name, spec) in specs {
            expected_groups
                .entry(spec.storage_group.clone())
                .or_default()
                .insert(name.clone());
            let value = before
                .iter()
                .find(|v| v["name"] == *name)
                .ok_or_else(|| error(ErrorCode::IntegrityMismatch, "runtime tensor missing"))?;
            let expected_hash = hashes.and_then(|h| h.get(name)).unwrap_or(&spec.digest);
            if group_hashes
                .insert(&spec.storage_group, expected_hash)
                .is_some_and(|h| h != expected_hash)
            {
                return Err(error(
                    ErrorCode::AliasScopeEscape,
                    "alias observations disagree on shared bytes",
                ));
            }
            let dtype = serde_json::to_value(spec.dtype).unwrap();
            let mut stride = 1;
            let mut strides = Vec::new();
            for n in spec.shape.iter().rev() {
                strides.push(stride);
                stride *= n;
            }
            strides.reverse();
            if value["role"] != role
                || value["dtype"] != dtype
                || value["shape"] != json!(spec.shape)
                || value["strides"] != json!(strides)
                || value["device"] != "cpu"
                || value["requires_grad"] != false
                || value["grad_is_none"] != true
                || value["digest"].as_str()
                    != Some(
                        hashes
                            .and_then(|h| h.get(name))
                            .unwrap_or(&spec.digest)
                            .as_str(),
                    )
            {
                return Err(error(
                    ErrorCode::IntegrityMismatch,
                    "actual Torch tensor/state differs from frozen bytes/layout",
                ));
            }
            let group = value["group"]
                .as_str()
                .ok_or_else(|| error(ErrorCode::InvalidValue, "physical group missing"))?;
            actual_groups
                .entry(group.into())
                .or_default()
                .insert(name.clone());
            observed.insert(name, value);
        }
    }
    if before.len() != observed.len()
        || actual_groups.into_values().collect::<BTreeSet<_>>()
            != expected_groups.into_values().collect::<BTreeSet<_>>()
    {
        return Err(error(
            ErrorCode::AliasScopeEscape,
            "actual Torch storage sharing differs from source inventory",
        ));
    }
    Ok(())
}
