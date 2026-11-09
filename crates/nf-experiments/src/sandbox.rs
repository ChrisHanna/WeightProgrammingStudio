//! Opt-in, fixed-image local OCI isolation. A capability is minted only after
//! actual native/guest checks; it is not gym approval or a budget/lease grant.

use nf_core::canonical::ContentDigest;
use nf_core::error::{CoreError, ErrorCode, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub const IMAGE: &str =
    "python@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c";
const GUARDIAN: &[u8] = include_bytes!("sandbox/guardian.py");
const DOCKER_CONTROL: &str = include_str!("sandbox/docker_control.py");
const MAX_CONTROL: usize = 262_144;
/// Enforced cgroup envelope, including guest tmpfs (not caller-reported RSS).
pub const HOST_MEMORY_BYTES: u64 = 134_217_728;
/// Conservative admission allowance for controller assets/JSON/control buffers.
/// This is reservation sizing, NOT an enforced controller-process RSS limit.
pub const CONTROLLER_MEMORY_ALLOWANCE_BYTES: u64 = 268_435_456;
mod storage;
pub use storage::StorageReceipt;
/// Managed journal allocation, including live reservations and retained evidence.
pub const STORAGE_ROOT_BYTES: u64 = storage::ROOT_BYTES;
pub const STORAGE_JOB_BYTES: u64 = storage::JOB_BYTES;
pub const STORAGE_JOURNALS: usize = storage::JOURNALS;
const PROBE: &[u8] = br#"import ctypes,json,os,signal,socket,resource
from pathlib import Path
s=Path('/proc/self/status').read_text()
assert os.getuid()==65534 and os.getgid()==65534 and os.getgroups()==[]
assert 'CapEff:\t0000000000000000' in s and 'CapPrm:\t0000000000000000' in s
assert 'NoNewPrivs:\t1' in s and 'Seccomp:\t2' in s
assert Path('/sys/fs/cgroup/memory.max').read_text().strip()=='134217728'
assert Path('/sys/fs/cgroup/pids.max').read_text().strip()=='16'
assert Path('/sys/fs/cgroup/cpu.max').read_text().strip()=='100000 100000'
assert resource.getrlimit(resource.RLIMIT_NOFILE)==(64,64)
assert resource.getrlimit(resource.RLIMIT_CORE)==(0,0)
assert resource.getrlimit(resource.RLIMIT_FSIZE)==(65536,65536)
for target in ['/etc/nf-write','/job/main.py','/guard/policy.json','/sys/fs/cgroup/memory.max']:
 try: open(target,'wb').close(); raise AssertionError('write escaped')
 except OSError: pass
for target in ['/var/run/docker.sock','/controller-secret','/dev/nvidia0']:
 assert not Path(target).exists()
try: os.setuid(0); raise AssertionError('uid escape')
except PermissionError: pass
try: os.kill(os.getppid(),signal.SIGKILL); raise AssertionError('guardian signal escape')
except PermissionError: pass
libc=ctypes.CDLL(None,use_errno=True)
assert libc.mount(b'none',b'/out',b'tmpfs',0,None)==-1
assert ctypes.get_errno() in [1,13]
assert libc.unshare(0x10000000)==-1
sock=socket.socket();sock.settimeout(.1)
try: sock.connect(('198.51.100.1',9));raise AssertionError('network escape')
except OSError: pass
assert len(Path('/proc/net/route').read_text().splitlines())==1
# AF_VSOCK is not confined by network namespaces; raw packet sockets also deny.
for family,kind in [(40,socket.SOCK_STREAM),(17,socket.SOCK_RAW)]:
 try: bypass=socket.socket(family,kind)
 except OSError: pass
 else:
  bypass.close();raise AssertionError('alternate network socket allowed')
assert not any(k in os.environ for k in ['AWS_ACCESS_KEY_ID','AWS_SECRET_ACCESS_KEY','COORDINATOR_TOKEN','NF_HOST_SECRET'])
print(json.dumps({'qualified':True,'uid':os.getuid()}))
"#;

fn error(code: ErrorCode, text: &str) -> CoreError {
    CoreError::new(code, "/sandbox", text)
}
fn io(e: std::io::Error) -> CoreError {
    error(ErrorCode::Io, &e.to_string())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    pub wall_ms: u64,
    pub cpu_ms: u64,
    pub output_bytes: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            wall_ms: 2000,
            cpu_ms: 2000,
            output_bytes: 4096,
        }
    }
}
impl Limits {
    fn validate(&self) -> Result<()> {
        if !(100..=5000).contains(&self.wall_ms)
            || !(100..=5000).contains(&self.cpu_ms)
            || !(1..=65536).contains(&self.output_bytes)
        {
            return Err(error(
                ErrorCode::LimitExceeded,
                "unsupported finite sandbox limits",
            ));
        }
        Ok(())
    }
}

/// Private capability. No Deserialize, no user-supplied runtime command/image.
pub struct QualifiedDocker {
    root: PathBuf,
    proof: ContentDigest,
    environment: ContentDigest,
}
/// Native cleanup proof keeps the attested root locked through ledger release.
/// No Deserialize or public constructor: a caller cannot declare cleanup.
pub(crate) struct AttestedCleanup {
    _lock: storage::Lock,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Receipt {
    pub schema_version: u32,
    pub job: String,
    pub source: ContentDigest,
    pub input: ContentDigest,
    #[serde(default)]
    pub asset_identity: Option<ContentDigest>,
    pub capability: ContentDigest,
    pub status: String,
    pub cleanup_confirmed: bool,
    pub outputs_eligible: bool,
    pub promotion_status: String,
    pub report: Option<Value>,
    #[serde(default)]
    pub storage: Option<StorageReceipt>,
}
struct Captured {
    ok: bool,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    interrupted: bool,
}

fn command(args: &[String], duration: Duration, cancel: &AtomicBool) -> Result<Captured> {
    command_authorized(args, duration, cancel, None)
}

/// The callback is trusted controller code, never a guest/caller JSON assertion.
/// It is polled on this thread, so SQLite-backed authority need not be Sync.
fn command_authorized(
    args: &[String],
    duration: Duration,
    cancel: &AtomicBool,
    authorize: Option<&dyn Fn() -> Result<()>>,
) -> Result<Captured> {
    if let Some(check) = authorize {
        check()?;
    }
    let mut child = Command::new("/usr/bin/python3")
        .args(["-I", "-S", "-B", "-c", DOCKER_CONTROL])
        .args(args)
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(io)?;
    let out = child.stdout.take().unwrap();
    let err = child.stderr.take().unwrap();
    let stdout = std::thread::spawn(move || {
        let mut b = Vec::new();
        out.take((MAX_CONTROL + 1) as u64)
            .read_to_end(&mut b)
            .map(|_| b)
    });
    let stderr = std::thread::spawn(move || {
        let mut b = Vec::new();
        err.take((MAX_CONTROL + 1) as u64)
            .read_to_end(&mut b)
            .map(|_| b)
    });
    let start = Instant::now();
    let creating = args.first().is_some_and(|arg| arg == "create");
    let mut interrupted = false;
    let mut revoked = None;
    let status = loop {
        if revoked.is_none() {
            if let Some(check) = authorize {
                if let Err(e) = check() {
                    revoked = Some(e);
                    // A submitted create may finish after a client disconnects.
                    // Drain its terminal response before native cleanup checks
                    // absence. On timeout its durable marker remains uncertain.
                    if !creating {
                        let _ = child.kill();
                        break child.wait().map_err(io)?;
                    }
                }
            }
        }
        if let Some(status) = child.try_wait().map_err(io)? {
            break status;
        }
        if start.elapsed() >= duration {
            interrupted = true;
            let _ = child.kill();
            break child.wait().map_err(io)?;
        }
        if cancel.load(Ordering::Acquire) {
            interrupted = true;
            if !creating {
                let _ = child.kill();
                break child.wait().map_err(io)?;
            }
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let stdout = stdout
        .join()
        .map_err(|_| error(ErrorCode::InternalPanic, "docker output reader"))?
        .map_err(io)?;
    let stderr = stderr
        .join()
        .map_err(|_| error(ErrorCode::InternalPanic, "docker error reader"))?
        .map_err(io)?;
    if let Some(e) = revoked {
        return Err(e);
    }
    if let Some(check) = authorize {
        check()?;
    }
    if stdout.len() > MAX_CONTROL || stderr.len() > MAX_CONTROL {
        return Err(error(
            ErrorCode::LimitExceeded,
            "docker control output exceeded",
        ));
    }
    Ok(Captured {
        ok: status.success(),
        stdout,
        stderr,
        interrupted,
    })
}
fn control(args: &[String]) -> Result<Captured> {
    command(args, Duration::from_secs(10), &AtomicBool::new(false))
}
fn json_control(args: &[String]) -> Result<Value> {
    let capture = control(args)?;
    if !capture.ok {
        return Err(error(
            ErrorCode::UnsupportedCapability,
            "local Docker control unavailable",
        ));
    }
    serde_json::from_slice(&capture.stdout)
        .map_err(|_| error(ErrorCode::InvalidJson, "invalid Docker control JSON"))
}
fn strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|s| (*s).into()).collect()
}
/// Docker capability order and the optional CAP_ prefix are representations,
/// not permissions. Reject duplicates (including aliases), unknown names and
/// malformed values; the normalized membership must be exactly the frozen set.
fn exact_capabilities(value: &Value, expected: &[&str]) -> bool {
    let Some(values) = value.as_array() else {
        return false;
    };
    if values.len() != expected.len() {
        return false;
    }
    let mut seen = std::collections::BTreeSet::new();
    values.iter().all(|value| {
        value.as_str().is_some_and(|name| {
            let normalized = name.strip_prefix("CAP_").unwrap_or(name);
            expected.contains(&normalized) && seen.insert(normalized)
        })
    })
}
fn save(job: &storage::Job, path: &Path, bytes: &[u8]) -> Result<()> {
    job.before_save(path, bytes.len())?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o444)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
        .map_err(io)?;
    file.write_all(bytes).map_err(io)?;
    file.sync_all().map_err(io)?;
    job.observe()
}
fn record(job: &storage::Job, path: &Path, value: &impl Serialize) -> Result<()> {
    save(
        job,
        path,
        &serde_json::to_vec(value)
            .map_err(|_| error(ErrorCode::InvalidValue, "invalid journal record"))?,
    )
}
fn random_name() -> Result<String> {
    let mut b = [0_u8; 16];
    File::open("/dev/urandom")
        .map_err(io)?
        .read_exact(&mut b)
        .map_err(io)?;
    Ok(format!(
        "nf-sandbox-{}",
        b.iter().map(|b| format!("{b:02x}")).collect::<String>()
    ))
}
fn checked_root(path: &Path) -> Result<PathBuf> {
    if !path.is_absolute()
        || path
            .as_os_str()
            .as_encoded_bytes()
            .iter()
            .any(|b| matches!(*b, b',' | b'\n' | b'\r' | b':'))
    {
        return Err(error(
            ErrorCode::InvalidValue,
            "absolute safe controller root required",
        ));
    }
    let mut current = PathBuf::new();
    for component in path.components() {
        if !matches!(component, Component::RootDir | Component::Normal(_)) {
            return Err(error(ErrorCode::ScopeViolation, "nonphysical root"));
        }
        current.push(component);
        let meta = fs::symlink_metadata(&current).map_err(io)?;
        if !meta.is_dir() || meta.file_type().is_symlink() {
            return Err(error(
                ErrorCode::ScopeViolation,
                "symlink or non-directory root",
            ));
        }
    }
    let meta = fs::symlink_metadata(path).map_err(io)?;
    // Controller-private journal; it is NEVER mounted into the container.
    // SAFETY: geteuid has no pointer or mutable-memory preconditions.
    let owner = unsafe { libc::geteuid() };
    if meta.uid() != owner || meta.mode() & 0o077 != 0 {
        return Err(error(
            ErrorCode::PermissionDenied,
            "controller root must be private and owner-held",
        ));
    }
    Ok(path.to_path_buf())
}

impl QualifiedDocker {
    pub fn qualify(enabled: bool, root: &Path) -> Result<Self> {
        if !enabled {
            return Err(error(
                ErrorCode::PermissionDenied,
                "UNTRUSTED_GYM_EXECUTION_DISABLED",
            ));
        }
        let root = checked_root(root)?;
        let version = json_control(&strings(&["version", "--format", "{{json .Server}} "]))?;
        if version["Os"] != "linux" {
            return Err(error(
                ErrorCode::UnsupportedCapability,
                "Linux OCI required",
            ));
        }
        let info = json_control(&strings(&[
            "info",
            "--format",
            r#"{"CgroupVersion":{{json .CgroupVersion}},"MemoryLimit":{{json .MemoryLimit}},"PidsLimit":{{json .PidsLimit}},"SecurityOptions":{{json .SecurityOptions}}}"#,
        ]))?;
        let security = info["SecurityOptions"]
            .as_array()
            .ok_or_else(|| error(ErrorCode::UnsupportedCapability, "missing security options"))?;
        if info["CgroupVersion"] != "2"
            || info["MemoryLimit"] != true
            || info["PidsLimit"] != true
            || !security
                .iter()
                .any(|v| v.as_str().is_some_and(|s| s.starts_with("name=seccomp,")))
        {
            return Err(error(
                ErrorCode::UnsupportedCapability,
                "enforced cgroupv2 and seccomp required",
            ));
        }
        let image = json_control(&strings(&["image", "inspect", IMAGE]))?;
        if !image[0]["RepoDigests"]
            .as_array()
            .is_some_and(|v| v.iter().any(|d| d == IMAGE))
        {
            return Err(error(
                ErrorCode::IntegrityMismatch,
                "immutable approved Python image missing",
            ));
        }
        let environment = ContentDigest::of_bytes(
            &serde_json::to_vec(&(Self::profile_identity(), &version, &info, &image[0]["Id"]))
                .unwrap(),
        );
        let mut capability = Self {
            root,
            proof: ContentDigest::of_bytes(b"pending sandbox proof"),
            environment,
        };
        // Reconcile crashes and legacy retained workspaces before the probe
        // itself needs storage. Recovery cannot execute untrusted code.
        capability.recover_pending()?;
        // Qualification must surface infrastructure/configuration failures,
        // not collapse them into a receipt with no guardian report.
        let receipt = capability.execute_assets_authorized(
            PROBE,
            b"{}",
            &BTreeMap::new(),
            Limits::default(),
            STORAGE_JOB_BYTES,
            &AtomicBool::new(false),
            Instant::now() + Duration::from_secs(15),
            &|| Ok(()),
        )?;
        let report = receipt.report.as_ref().ok_or_else(|| {
            error(
                ErrorCode::UnsupportedCapability,
                "sandbox probe produced no report",
            )
        })?;
        if receipt.status != "completed"
            || !receipt.cleanup_confirmed
            || decode_hex(report["stdout_hex"].as_str().unwrap_or(""))?
                != b"{\"qualified\": true, \"uid\": 65534}\n"
        {
            return Err(error(
                ErrorCode::UnsupportedCapability,
                "real sandbox capability probe failed",
            ));
        }
        capability.proof=ContentDigest::of_bytes(&serde_json::to_vec(&json!({"guardian":ContentDigest::of_bytes(GUARDIAN),"image":IMAGE,"server":version,"receipt":receipt})).unwrap());
        Ok(capability)
    }
    /// Native ledger binding includes the directory inode, not only its name.
    /// A replacement directory cannot stand in for a crashed sandbox root.
    pub(crate) fn durable_root_identity(&self) -> Result<ContentDigest> {
        use std::os::unix::fs::MetadataExt;
        checked_root(&self.root)?;
        let metadata = fs::symlink_metadata(&self.root).map_err(io)?;
        Ok(ContentDigest::of_bytes(
            &serde_json::to_vec(&(
                &self.root,
                metadata.dev(),
                metadata.ino(),
                Self::profile_identity(),
            ))
            .map_err(|_| error(ErrorCode::InvalidValue, "invalid sandbox root identity"))?,
        ))
    }
    pub fn proof(&self) -> &ContentDigest {
        &self.proof
    }
    /// Snapshot of managed nf-sandbox-* storage; unrelated controller/CAS data
    /// is not included. Busy roots fail closed rather than racing an execution.
    pub fn storage_usage(&self) -> Result<u64> {
        let _lock = storage::Lock::acquire(&self.root)?;
        storage::root_usage(&self.root)
    }
    pub fn environment_identity(&self) -> &ContentDigest {
        &self.environment
    }
    pub fn profile_identity() -> ContentDigest {
        ContentDigest::of_bytes(
            &serde_json::to_vec(&(
                IMAGE,
                ContentDigest::of_bytes(GUARDIAN),
                ContentDigest::of_bytes(DOCKER_CONTROL.as_bytes()),
                ContentDigest::of_bytes(include_bytes!("sandbox.rs")),
                ContentDigest::of_bytes(include_bytes!("sandbox/storage.rs")),
            ))
            .unwrap(),
        )
    }
    pub fn execute(
        &self,
        source: &[u8],
        input: &[u8],
        limits: Limits,
        cancel: &AtomicBool,
    ) -> Result<Receipt> {
        self.execute_assets(source, input, &BTreeMap::new(), limits, cancel)
    }
    pub(crate) fn execute_assets(
        &self,
        source: &[u8],
        input: &[u8],
        assets: &BTreeMap<String, Vec<u8>>,
        limits: Limits,
        cancel: &AtomicBool,
    ) -> Result<Receipt> {
        self.execute_assets_with_staging_limit(
            source,
            input,
            assets,
            limits,
            STORAGE_JOB_BYTES,
            cancel,
        )
    }
    /// Controller-supplied study allowance for the charged read-only workspace
    /// (source, input, assets, guardian, policy and directories). Journal evidence
    /// and the admission capacity probe remain under the fixed job/root ceilings.
    /// Admission uses a conservative metadata-inclusive bound; actual charged
    /// bytes are checked at allocation boundaries. This does not reserve
    /// underlying disk capacity or bound external CAS data.
    pub fn execute_assets_with_staging_limit(
        &self,
        source: &[u8],
        input: &[u8],
        assets: &BTreeMap<String, Vec<u8>>,
        limits: Limits,
        workspace_limit_bytes: u64,
        cancel: &AtomicBool,
    ) -> Result<Receipt> {
        self.execute_assets_inner(
            source,
            input,
            assets,
            limits,
            workspace_limit_bytes,
            cancel,
            None,
            None,
        )
    }

    /// Live, controller-owned authorization plus an absolute monotonic deadline.
    /// Revocation stops Docker control, then removes the owned container before
    /// returning an error. Cleanup deliberately does not require live authority.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_assets_authorized(
        &self,
        source: &[u8],
        input: &[u8],
        assets: &BTreeMap<String, Vec<u8>>,
        limits: Limits,
        workspace_limit_bytes: u64,
        cancel: &AtomicBool,
        deadline: Instant,
        authorize: &dyn Fn() -> Result<()>,
    ) -> Result<Receipt> {
        if deadline.saturating_duration_since(Instant::now())
            < Duration::from_millis(limits.wall_ms)
        {
            return Err(error(
                ErrorCode::LimitExceeded,
                "insufficient wall allowance before staging",
            ));
        }
        let check = || {
            // Caller cancellation is a bounded execution outcome, handled by
            // command polling and returned with native cleanup evidence. Lease
            // revocation/deadline failure remain errors and retain ledger holds.
            if Instant::now() >= deadline {
                return Err(error(ErrorCode::LimitExceeded, "sandbox authority expired"));
            }
            authorize()?;
            if Instant::now() >= deadline {
                return Err(error(
                    ErrorCode::LimitExceeded,
                    "sandbox authority expired during check",
                ));
            }
            Ok(())
        };
        let admit = || {
            check()?;
            if deadline.saturating_duration_since(Instant::now())
                < Duration::from_millis(limits.wall_ms)
            {
                return Err(error(
                    ErrorCode::LimitExceeded,
                    "insufficient wall allowance before staging",
                ));
            }
            Ok(())
        };
        self.execute_assets_inner(
            source,
            input,
            assets,
            limits.clone(),
            workspace_limit_bytes,
            cancel,
            Some(&check),
            Some(&admit),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_assets_inner(
        &self,
        source: &[u8],
        input: &[u8],
        assets: &BTreeMap<String, Vec<u8>>,
        limits: Limits,
        workspace_limit_bytes: u64,
        cancel: &AtomicBool,
        authorize: Option<&dyn Fn() -> Result<()>>,
        admit: Option<&dyn Fn() -> Result<()>>,
    ) -> Result<Receipt> {
        if let Some(check) = authorize {
            check()?;
        }
        limits.validate()?;
        if workspace_limit_bytes == 0 || workspace_limit_bytes > STORAGE_JOB_BYTES {
            return Err(error(
                ErrorCode::LimitExceeded,
                "invalid study staging quota",
            ));
        }
        let mut asset_bytes = 0_usize;
        if assets.len() > 64 {
            return Err(error(
                ErrorCode::LimitExceeded,
                "sandbox asset count exceeded",
            ));
        }
        let mut identities = BTreeMap::new();
        for (name, bytes) in assets {
            if let Some(check) = authorize {
                check()?;
            }
            crate::intake::relative(name)?;
            asset_bytes = asset_bytes
                .checked_add(bytes.len())
                .ok_or_else(|| error(ErrorCode::LimitExceeded, "asset byte overflow"))?;
            if bytes.len() > 1048576 || asset_bytes > 8388608 {
                return Err(error(
                    ErrorCode::LimitExceeded,
                    "sandbox asset bytes exceeded",
                ));
            }
            identities.insert(name, ContentDigest::of_bytes(bytes));
        }
        let asset_identity = if assets.is_empty() {
            None
        } else {
            Some(ContentDigest::of_bytes(
                &serde_json::to_vec(&identities).unwrap(),
            ))
        };
        let _storage_lock = storage::Lock::acquire(&self.root)?;
        if source.is_empty() || source.len() > 65536 || input.len() > 65536 {
            return Err(error(
                ErrorCode::LimitExceeded,
                "bounded source/input required",
            ));
        }
        if storage::workspace_bound(source.len(), input.len(), assets, &limits)?
            > workspace_limit_bytes
        {
            return Err(error(
                ErrorCode::LimitExceeded,
                "study staging admission quota exceeded",
            ));
        }
        if let Some(check) = authorize {
            check()?;
        }
        if let Some(check) = admit {
            check()?;
        }
        let name = random_name()?;
        let path = self.root.join(&name);
        let job = storage::reserve(&self.root, &path, workspace_limit_bytes)?;
        // Persist intent before staging: recovery also handles partial staging.
        let mut receipt = Receipt {
            schema_version: 3,
            job: name.clone(),
            source: ContentDigest::of_bytes(source),
            input: ContentDigest::of_bytes(input),
            asset_identity,
            capability: self.proof.clone(),
            status: "incomplete".into(),
            cleanup_confirmed: false,
            outputs_eligible: false,
            promotion_status: "not_evaluated".into(),
            report: None,
            storage: Some(job.initial_receipt()),
        };
        let result = (|| {
            if let Some(check) = authorize {
                check()?;
            }
            record(&job, &path.join("intent.tmp"), &receipt)?;
            fs::rename(path.join("intent.tmp"), path.join("intent.json")).map_err(io)?;
            File::open(&path).map_err(io)?.sync_all().map_err(io)?;
            for folder in ["guard", "job"] {
                if let Some(check) = authorize {
                    check()?;
                }
                job.mkdir(&path.join(folder))?;
            }
            if let Some(check) = authorize {
                check()?;
            }
            save(&job, &path.join("guard/guardian.py"), GUARDIAN)?;
            if let Some(check) = authorize {
                check()?;
            }
            record(&job, &path.join("guard/policy.json"), &limits)?;
            if let Some(check) = authorize {
                check()?;
            }
            save(&job, &path.join("job/main.py"), source)?;
            if let Some(check) = authorize {
                check()?;
            }
            save(&job, &path.join("job/input.json"), input)?;
            if !assets.is_empty() {
                let base = path.join("job/assets");
                job.mkdir(&base)?;
                for (name, bytes) in assets {
                    if let Some(check) = authorize {
                        check()?;
                    }
                    let target = base.join(name);
                    if let Some(parent) = target.parent() {
                        job.mkdir(parent)?;
                    }
                    save(&job, &target, bytes)?;
                }
            }
            self.run_container(&job, &name, &limits, cancel, authorize)
        })();
        // Cleanup is required even if create/start/inspect/parse failed. Only
        // our unpredictable persisted name and exact ownership label may be removed.
        receipt.cleanup_confirmed = self.cleanup(&job, &name)?;
        let mut failure = None;
        match result {
            Ok((status, report)) => {
                receipt.status = status;
                receipt.report = report;
            }
            Err(e) => {
                record(&job, &path.join("failure.json"), &e)?;
                failure = Some(e);
            }
        }
        if !receipt.cleanup_confirmed {
            receipt.status = "incomplete".into();
            receipt.report = None;
        }
        storage::finish(&job, &mut receipt, false)?;
        if authorize.is_some() {
            if let Some(e) = failure {
                return Err(e);
            }
            if let Some(check) = authorize {
                check()?;
            }
        }
        Ok(receipt)
    }
    fn run_container(
        &self,
        job: &storage::Job,
        name: &str,
        limits: &Limits,
        cancel: &AtomicBool,
        authorize: Option<&dyn Fn() -> Result<()>>,
    ) -> Result<(String, Option<Value>)> {
        let path = &job.path;
        let mut args = strings(&[
            "create",
            "--pull=never",
            "--init",
            "--name",
            name,
            "--label",
            "org.neural-foundry.sandbox=v1",
            "--label",
            &format!("org.neural-foundry.owner={name}"),
            "--network",
            "none",
            "--read-only",
            "--cap-drop=ALL",
            "--cap-add=KILL",
            "--cap-add=SETUID",
            "--cap-add=SETGID",
            "--security-opt=no-new-privileges",
            "--pids-limit=16",
            "--memory=134217728",
            "--memory-swap=134217728",
            "--cpus=1",
            "--ulimit",
            "cpu=6:6",
            "--ulimit",
            "fsize=65536:65536",
            "--ulimit",
            "nofile=64:64",
            "--ulimit",
            "core=0:0",
            "--stop-timeout=0",
            "--log-driver=none",
            "--shm-size=65536",
            "--ulimit",
            "msgqueue=0:0",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=8388608,mode=1777",
            "--tmpfs",
            "/out:rw,noexec,nosuid,nodev,size=8388608,uid=65534,gid=0,mode=0770",
        ]);
        args.extend(strings(&[
            "--mount",
            &format!(
                "type=bind,src={},dst=/guard,readonly",
                path.join("guard").display()
            ),
            "--mount",
            &format!(
                "type=bind,src={},dst=/job,readonly",
                path.join("job").display()
            ),
            "--env",
            "NF_CONTAINER_GUARDIAN=v1",
            "--workdir",
            "/out",
            "--entrypoint",
            "/usr/local/bin/python3",
            IMAGE,
            "-I",
            "-S",
            "-B",
            "/guard/guardian.py",
        ]));
        record(
            job,
            &path.join("create_started.json"),
            &json!({"schema_version":1,"job":name}),
        )?;
        let created = command_authorized(&args, Duration::from_secs(10), cancel, authorize)?;
        if created.interrupted {
            return Ok(("cancelled".into(), None));
        }
        if !created.ok {
            return Err(error(
                ErrorCode::UnsupportedCapability,
                "sandbox creation failed",
            ));
        }
        let Some(inspect) = self.inspect_authorized(name, cancel, authorize)? else {
            return Ok(("cancelled".into(), None));
        };
        record(job, &path.join("observed.json"), &inspect)?;
        Self::verify_configuration(&inspect, name, path)?;
        record(job, &path.join("configured.json"), &inspect)?;
        if cancel.load(Ordering::Acquire) {
            return Ok(("cancelled".into(), None));
        }
        let capture = command_authorized(
            &strings(&["start", "--attach", name]),
            Duration::from_millis(limits.wall_ms + 10000),
            cancel,
            authorize,
        )?;
        if capture.interrupted {
            return Ok((
                if cancel.load(Ordering::Acquire) {
                    "cancelled"
                } else {
                    "control_timeout"
                }
                .into(),
                None,
            ));
        }
        let Some(inspect) = self.inspect_authorized(name, cancel, authorize)? else {
            return Ok(("cancelled".into(), None));
        };
        record(job, &path.join("exit.json"), &inspect)?;
        if inspect["State"]["Running"] != false
            || inspect["State"]["OOMKilled"] != false
            || !capture.ok
            || !capture.stderr.is_empty()
        {
            return Ok(("failed".into(), None));
        }
        let value: Value = serde_json::from_slice(&capture.stdout)
            .map_err(|_| error(ErrorCode::InvalidJson, "invalid guardian report"))?;
        if value["schema_version"] != 1
            || value["kind"] != "sandbox_execution"
            || value["measurement_scope"] != "private_container_cgroup_v2"
        {
            return Err(error(
                ErrorCode::IntegrityMismatch,
                "guardian report mismatch",
            ));
        }
        let stdout = decode_hex(
            value["stdout_hex"]
                .as_str()
                .ok_or_else(|| error(ErrorCode::InvalidJson, "stdout missing"))?,
        )?;
        let stderr = decode_hex(
            value["stderr_hex"]
                .as_str()
                .ok_or_else(|| error(ErrorCode::InvalidJson, "stderr missing"))?,
        )?;
        if stdout.len() + stderr.len() > limits.output_bytes {
            return Err(error(
                ErrorCode::LimitExceeded,
                "guardian output exceeded quota",
            ));
        }
        let status = value["status"]
            .as_str()
            .ok_or_else(|| error(ErrorCode::InvalidJson, "guardian status missing"))?
            .to_owned();
        Ok((status, Some(value)))
    }
    fn inspect_authorized(
        &self,
        name: &str,
        cancel: &AtomicBool,
        authorize: Option<&dyn Fn() -> Result<()>>,
    ) -> Result<Option<Value>> {
        let capture = command_authorized(
            &strings(&["inspect", name]),
            Duration::from_secs(10),
            cancel,
            authorize,
        )?;
        if capture.interrupted {
            if cancel.load(Ordering::Acquire) {
                return Ok(None);
            }
            return Err(error(ErrorCode::LimitExceeded, "Docker inspect timed out"));
        }
        if !capture.ok {
            return Err(error(
                ErrorCode::UnsupportedCapability,
                "authorized Docker inspect failed",
            ));
        }
        let value: Value = serde_json::from_slice(&capture.stdout)
            .map_err(|_| error(ErrorCode::InvalidJson, "invalid Docker control JSON"))?;
        Ok(Some(value[0].clone()))
    }
    /// Controller-only observation for real lifecycle tests and supervision.
    #[cfg(test)]
    pub(crate) fn container_running(name: &str) -> Result<bool> {
        if !valid_name(name) {
            return Err(error(ErrorCode::ScopeViolation, "invalid sandbox name"));
        }
        let value = json_control(&strings(&["inspect", name]))?;
        Ok(value[0]["State"]["Running"] == true)
    }
    #[allow(dead_code)]
    fn inspect(&self, name: &str) -> Result<Value> {
        let value = json_control(&strings(&["inspect", name]))?;
        Ok(value[0].clone())
    }
    fn verify_configuration(v: &Value, name: &str, path: &Path) -> Result<()> {
        let h = &v["HostConfig"];
        let exact = [
            ("NetworkMode", json!("none")),
            ("ReadonlyRootfs", json!(true)),
            ("Privileged", json!(false)),
            ("PidMode", json!("")),
            ("IpcMode", json!("private")),
            ("Memory", json!(MEMORY_BYTES)),
            ("MemorySwap", json!(MEMORY_BYTES)),
            ("PidsLimit", json!(16)),
            ("NanoCpus", json!(1000000000)),
            ("ShmSize", json!(65536)),
            ("CgroupnsMode", json!("private")),
            ("CapDrop", json!(["ALL"])),
        ];
        if exact.iter().any(|(k, want)| h[k] != *want)
            || !exact_capabilities(&h["CapAdd"], &["KILL", "SETGID", "SETUID"])
            || v["Config"]["Image"] != IMAGE
            || v["Config"]["Labels"]["org.neural-foundry.owner"] != name
            || !h["SecurityOpt"]
                .as_array()
                .is_some_and(|a| a.iter().any(|s| s == "no-new-privileges"))
            || v["Mounts"].as_array().is_none_or(|a| {
                a.len() != 2
                    || a.iter().any(|m| {
                        let expected = match m["Destination"].as_str() {
                            Some("/guard") => path.join("guard"),
                            Some("/job") => path.join("job"),
                            _ => return true,
                        };
                        m["RW"] != false
                            || m["Type"] != "bind"
                            || m["Source"].as_str() != expected.to_str()
                    })
            })
        {
            return Err(error(
                ErrorCode::UnsupportedCapability,
                "Docker did not enforce frozen sandbox configuration",
            ));
        }
        Ok(())
    }
    fn cleanup(&self, job: &storage::Job, name: &str) -> Result<bool> {
        let path = self.root.join(name);
        let started = path.join("create_started.json").try_exists().map_err(io)?;
        let terminal = path.join("create_terminal.json");
        let confirmed = if started && terminal.try_exists().map_err(io)? {
            let file = OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NOFOLLOW)
                .open(terminal)
                .map_err(io)?;
            let value: Value = serde_json::from_reader(file.take(1024)).map_err(|_| {
                error(
                    ErrorCode::IntegrityMismatch,
                    "invalid create completion evidence",
                )
            })?;
            value["schema_version"] == 1
                && value["id"]
                    .as_str()
                    .is_some_and(|id| id.len() == 64 && id.bytes().all(|b| b.is_ascii_hexdigit()))
        } else {
            !started
        };
        let uncertain = || {
            error(ErrorCode::UnsupportedCapability,
            "in-flight Engine create has no durable terminal response; hold retained even if currently absent; retry after helper completes, otherwise operator Engine reconciliation required")
        };
        let existing = control(&strings(&["inspect", name]))?;
        if !existing.ok {
            if !confirmed {
                return Err(uncertain());
            }
            return self.absent(name);
        }
        let value: Value = serde_json::from_slice(&existing.stdout)
            .map_err(|_| error(ErrorCode::InvalidJson, "cleanup inspect failed"))?;
        if value[0]["Config"]["Labels"]["org.neural-foundry.owner"] != name
            || value[0]["Config"]["Labels"]["org.neural-foundry.sandbox"] != "v1"
            || value[0]["Config"]["Image"] != IMAGE
        {
            return Err(error(
                ErrorCode::ScopeViolation,
                "foreign container cleanup refused",
            ));
        }
        let evidence = self.root.join(name).join("pre_cleanup.json");
        let saved = if !evidence.exists() {
            record(job, &evidence, &value[0])
        } else {
            Ok(())
        };
        // Disk pressure must never prevent attempting container teardown.
        let removed = control(&strings(&["rm", "--force", name]));
        let absent = self.absent(name);
        saved?;
        removed?;
        if !confirmed {
            return Err(uncertain());
        }
        absent
    }
    fn absent(&self, name: &str) -> Result<bool> {
        let capture = control(&strings(&[
            "ps",
            "--all",
            "--filter",
            &format!("name=^/{name}$"),
            "--format",
            "{{.ID}} ",
        ]))?;
        Ok(capture.ok && capture.stdout.iter().all(u8::is_ascii_whitespace))
    }
    /// Controller-only crash reconciliation. Records stay private; no worker
    /// can select a target container or produce an authenticated intent.
    pub fn recover_pending(&self) -> Result<usize> {
        let _storage_lock = storage::Lock::acquire(&self.root)?;
        self.recover_locked()
    }
    pub(crate) fn recover_attested(&self, expected: &ContentDigest) -> Result<AttestedCleanup> {
        let lock = storage::Lock::acquire(&self.root)?;
        if self.durable_root_identity()? != *expected {
            return Err(error(
                ErrorCode::PermissionDenied,
                "recovery requires exact attested sandbox root",
            ));
        }
        self.recover_locked()?;
        if self.durable_root_identity()? != *expected {
            return Err(error(
                ErrorCode::IntegrityMismatch,
                "attested sandbox root changed during recovery",
            ));
        }
        Ok(AttestedCleanup { _lock: lock })
    }
    fn recover_locked(&self) -> Result<usize> {
        checked_root(&self.root)?;
        let mut recovered = 0;
        for path in storage::jobs(&self.root)? {
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            // No intent is possible if the controller crashed during reservation.
            // Still reconcile the unpredictable owned name before reclaiming.
            let mut receipt = None;
            for file in ["recovery.json", "completion.json", "intent.json"] {
                let p = path.join(file);
                if p.try_exists().map_err(io)? {
                    if file == "recovery.json" && storage::legacy_recovery(&p)? {
                        // Old markers lacked a full receipt. Reconfirm absence
                        // and migrate the original intent/completion instead.
                        continue;
                    }
                    receipt = Some(storage::read_receipt(&p)?);
                    break;
                }
            }
            if let Some(done) = &receipt {
                if done.job != name {
                    return Err(error(
                        ErrorCode::IntegrityMismatch,
                        "journal target mismatch",
                    ));
                }
                if storage::finalized(&path, done)? && self.absent(&name)? {
                    continue;
                }
            }
            let job = storage::Job::recovery(&self.root, &path)?;
            let clean = self.cleanup(&job, &name)?;
            if !clean {
                return Err(error(
                    ErrorCode::UnsupportedCapability,
                    "unconfirmed sandbox cleanup",
                ));
            }
            let Some(mut receipt) = receipt else {
                storage::reclaim(&path)?;
                fs::remove_dir_all(&path).map_err(io)?;
                File::open(&self.root).map_err(io)?.sync_all().map_err(io)?;
                recovered += 1;
                continue;
            };
            // Preserve a completed result, but never upgrade an interrupted job.
            receipt.schema_version = 3;
            receipt.cleanup_confirmed = true;
            storage::finish(&job, &mut receipt, true)?;
            recovered += 1;
        }
        storage::prune(&self.root, false)?;
        Ok(recovered)
    }
}
fn valid_name(s: &str) -> bool {
    s.strip_prefix("nf-sandbox-")
        .is_some_and(|s| s.len() == 32 && s.bytes().all(|c| c.is_ascii_hexdigit()))
}
#[cfg(test)]
#[path = "sandbox/authority_tests.rs"]
mod authority_tests;
#[cfg(test)]
#[path = "sandbox/configuration_tests.rs"]
mod configuration_tests;
#[cfg(test)]
#[path = "sandbox/storage_tests.rs"]
mod storage_tests;

/// Frozen cgroup memory envelope, shared with durable admission.
pub const MEMORY_BYTES: u64 = 134_217_728;

pub fn decode_hex(s: &str) -> Result<Vec<u8>> {
    if s.len() > 131072 || !s.len().is_multiple_of(2) {
        return Err(error(ErrorCode::InvalidValue, "invalid bounded hex"));
    }
    s.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| {
            let h = (p[0] as char).to_digit(16);
            let l = (p[1] as char).to_digit(16);
            h.zip(l)
                .map(|(h, l)| ((h << 4) | l) as u8)
                .ok_or_else(|| error(ErrorCode::InvalidValue, "invalid hex"))
        })
        .collect()
}
