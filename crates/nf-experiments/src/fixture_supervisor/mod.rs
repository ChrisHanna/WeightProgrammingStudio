//! Static-native, FD-only Linux fixture sandbox; not the T038 integration.
//!
//! Only anonymous memory, local threads and stdin/stdout/stderr are available.
//! No Python, dynamic ELF, host filesystem, network, GPU, ledger or publication.
//! Sealed executable bytes include the whole static runtime. Kernel/admin and
//! the admin-protected Python/controller deployment are trusted. Unrelated
//! malicious same-UID host processes and compromised controller are NOT covered.
//!
//! Protocol: persist intent/absolute deadline -> prepare -> poll_prepared ->
//! persist returned identity -> recheck current lease/fence/admission -> authorize.
//! A receipt never grants authority. Recovery must reconcile this exact launch,
//! never reissue its nonce or reset a deadline. Rust cannot prove caller fsync.
//! Stop: close admission, persist stop, cancel. EOF/deadline are emergency fences.
//! CPU limit is integer seconds, AS is virtual address space (not total memory).
//! Guardian costs are separate snapshots; controller CPU/RSS observations are
//! process-wide and may include other threads/work. Receipt serialization,
//! transport, persistence and Drop tails need separate reservations/accounting.
//! No exact cutoff, aggregate scratch quota, fsync bound or scientific claim.
//! Unknown cleanup retains ALL reservations.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::{self, Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::MetadataExt;
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::time::{Duration, Instant};

const GUARDIAN: &str = concat!(
    include_str!("boundary.py"),
    "\n",
    include_str!("guardian.py")
);
const MAX_RECEIPT: usize = 180_000;
pub const PROFILE: &str = "static_elf_fd_cpu_linux_v2";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct LaunchBinding {
    pub contract: String,
    pub reservation: String,
    pub attempt: String,
    pub lease: String,
    pub fencing_epoch: u64,
    /// Controller-generated, unique 256-bit lowercase hex value; never reused.
    pub nonce: String,
    pub lease_deadline_boottime_ns: u64,
}

/// Construct in trusted controller code, never from worker/gym/receipt metadata.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    pub profile: String,
    pub boot_id: String,
    pub deadline_boottime_ns: u64,
    pub memory_bytes: u64,
    pub cpu_seconds: u64,
    pub output_bytes: u64,
    pub grace_ms: u64,
    pub reap_ms: u64,
    pub executable_sha256: String,
    pub argv: Vec<String>,
    pub input_hex: String,
    pub binding: LaunchBinding,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PreparedIdentity {
    pub policy: Policy,
    pub guardian_pid: u32,
    pub guardian_start_ticks: u64,
    pub policy_sha256: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Credentials {
    pub uid: u32,
    pub gid: u32,
    pub groups: Vec<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Receipt {
    pub identity: PreparedIdentity,
    pub qualified_credentials: Credentials,
    pub profile: String,
    pub boot_id: String,
    pub deadline_boottime_ns: u64,
    pub status: String,
    pub cleanup_confirmed: bool,
    pub exit_code: Option<i32>,
    pub memory_scope: String,
    pub cpu_used_ns: Option<u64>,
    pub peak_rss_bytes: Option<u64>,
    pub guardian_cpu_used_ns: u64,
    pub guardian_peak_rss_bytes: u64,
    pub all_children_cpu_used_ns: u64,
    pub accounting_scope: String,
    pub events: BTreeMap<String, u64>,
    pub stdout_hex: String,
    pub stderr_hex: String,
    pub cleanup_overshoot_ns: Option<u64>,
    pub cancellation_latency_ns: Option<u64>,
    pub receipt_overshoot_ns: u64,
    pub error: Option<String>,
    /// Added by this controller, not supplied by the guardian.
    #[serde(skip)]
    pub controller_observation: Option<ControllerObservation>,
}

#[derive(Debug, Serialize)]
pub struct ControllerObservation {
    pub process_cpu_delta_ns: u64,
    pub process_peak_rss_bytes: u64,
    pub elapsed_ns: u128,
}

pub struct Supervisor {
    child: Child,
    control: Option<ChildStdin>,
    bytes: Vec<u8>,
    policy: Policy,
    prepared: Option<PreparedIdentity>,
    authorized: bool,
    finished: bool,
    started: Instant,
    controller_cpu_start: u64,
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn usage() -> (u64, u64) {
    let mut value = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage writes the supplied live storage on success.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, value.as_mut_ptr()) } != 0 {
        return (0, 0);
    }
    let value = unsafe { value.assume_init() };
    let cpu = (value.ru_utime.tv_sec + value.ru_stime.tv_sec) as u64 * 1_000_000_000
        + (value.ru_utime.tv_usec + value.ru_stime.tv_usec) as u64 * 1000;
    (cpu, value.ru_maxrss as u64 * 1024)
}

fn qualify_interpreter(python: &Path) -> io::Result<()> {
    let mut path = python.canonicalize()?;
    loop {
        let meta = path.metadata()?;
        // SAFETY: getuid has no memory or ownership preconditions.
        if meta.uid() == unsafe { libc::getuid() } || meta.mode() & 0o022 != 0 {
            return Err(invalid(
                "interpreter/runtime ancestry must be admin-protected",
            ));
        }
        if !path.pop() {
            break;
        }
    }
    Ok(())
}

fn hex(value: &str) -> bool {
    value.len().is_multiple_of(2)
        && value
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

impl Supervisor {
    /// No worker starts here. The interpreter must be operator-approved Python
    /// >=3.11 in an admin-protected deployment. Its loaded runtime is checked too.
    pub fn prepare(python: &Path, policy: Policy) -> io::Result<Self> {
        if !cfg!(target_os = "linux") || !python.is_absolute() || policy.profile != PROFILE {
            return Err(invalid(
                "unsupported platform/interpreter/profile; no sandbox fallback",
            ));
        }
        qualify_interpreter(python)?;
        let encoded = serde_json::to_string(&policy).map_err(io::Error::other)?;
        if encoded.len() > 16384 {
            return Err(invalid("policy exceeds bounded IPC schema"));
        }
        let started = Instant::now();
        let controller_cpu_start = usage().0;
        let mut child = Command::new(python)
            .args(["-I", "-S", "-B", "-c", GUARDIAN, &encoded])
            .env_clear()
            .current_dir("/")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let control = child.stdin.take();
        let supervisor = Self {
            child,
            control,
            bytes: Vec::new(),
            policy,
            prepared: None,
            authorized: false,
            finished: false,
            started,
            controller_cpu_start,
        };
        let fd = supervisor.child.stdout.as_ref().unwrap().as_raw_fd();
        // SAFETY: fd is a live owned pipe; fcntl does not retain its integer.
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
        if flags < 0 || unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(supervisor)
    }

    /// Deliberately disabled legacy one-step API: it cannot establish persistence
    /// of guardian identity before execution. Call prepare/poll_prepared/authorize.
    pub fn start_after_durable_intent(_python: &Path, _policy: Policy) -> io::Result<Self> {
        Err(invalid("two-phase persisted identity handshake required"))
    }

    pub fn poll_prepared(&mut self) -> io::Result<Option<PreparedIdentity>> {
        if let Some(identity) = &self.prepared {
            return Ok(Some(identity.clone()));
        }
        self.drain()?;
        if let Some(newline) = self.bytes.iter().position(|byte| *byte == b'\n') {
            #[derive(Deserialize)]
            #[serde(deny_unknown_fields)]
            struct Prepared {
                kind: String,
                identity: PreparedIdentity,
            }
            let line: Vec<u8> = self.bytes.drain(..=newline).collect();
            let response: Prepared = match serde_json::from_slice(&line) {
                Ok(response) => response,
                Err(_) => {
                    let failure: serde_json::Value = serde_json::from_slice(&line)
                        .map_err(|_| invalid("qualification response is malformed"))?;
                    let requirement = failure
                        .get("requirement")
                        .and_then(serde_json::Value::as_str)
                        .filter(|value| value.len() <= 256)
                        .unwrap_or("unsupported host/runtime/static ELF");
                    return Err(invalid(&format!("qualification failed: {requirement}")));
                }
            };
            if response.kind != "prepared"
                || response.identity.policy != self.policy
                || response.identity.guardian_pid != self.child.id()
                || response.identity.guardian_start_ticks == 0
                || response.identity.policy_sha256.len() != 64
                || !hex(&response.identity.policy_sha256)
            {
                return Err(invalid("prepared identity mismatch"));
            }
            self.prepared = Some(response.identity.clone());
            return Ok(Some(response.identity));
        }
        if self.child.try_wait()?.is_some() {
            return Err(invalid(
                "qualification failed before preparation; no worker authorized",
            ));
        }
        Ok(None)
    }

    /// Caller MUST persist this exact identity, then recheck its live lease/fence,
    /// admission and minimum applicable deadline. This method cannot prove fsync
    /// or query a coordinator's current fence; it rejects mismatched/reused tokens.
    pub fn authorize_after_persist(&mut self, persisted: &PreparedIdentity) -> io::Result<()> {
        if self.authorized || self.prepared.as_ref() != Some(persisted) {
            return Err(invalid(
                "not prepared, mismatched identity, or authorization already consumed",
            ));
        }
        let mut encoded = serde_json::to_vec(&serde_json::json!({"authorize": persisted}))
            .map_err(io::Error::other)?;
        encoded.push(b'\n');
        self.control
            .as_mut()
            .ok_or_else(|| invalid("control closed"))?
            .write_all(&encoded)?;
        self.authorized = true;
        Ok(())
    }

    pub fn cancel_after_durable_stop(&mut self) -> io::Result<()> {
        if let Some(mut control) = self.control.take() {
            // Completion can close the pipe between the last fence and stop.
            // Still drain/reap its receipt; this does not undo the durable stop
            // or authorize publication. A missing receipt keeps cleanup unknown.
            if let Err(error) = control.write_all(b"S") {
                if error.kind() != io::ErrorKind::BrokenPipe {
                    return Err(error);
                }
            }
        }
        Ok(())
    }

    /// Missing/invalid receipt means incomplete, unknown cleanup, retained holds.
    pub fn poll(&mut self) -> io::Result<Option<Receipt>> {
        if self.finished {
            return Err(invalid("receipt already consumed"));
        }
        if self.prepared.is_none() && self.poll_prepared()?.is_none() {
            return Ok(None);
        }
        self.drain()?;
        let Some(status) = self.child.try_wait()? else {
            return Ok(None);
        };
        self.drain()?;
        self.finished = true;
        if !status.success() {
            return Err(invalid("supervisor lost; cleanup unknown"));
        }
        let mut receipt: Receipt = serde_json::from_slice(&self.bytes).map_err(io::Error::other)?;
        let events = &receipt.events;
        if Some(&receipt.identity) != self.prepared.as_ref()
            || receipt.profile != PROFILE
            || receipt.boot_id != self.policy.boot_id
            || receipt.deadline_boottime_ns != self.policy.deadline_boottime_ns
            || !hex(&receipt.stdout_hex)
            || !hex(&receipt.stderr_hex)
            || receipt.stdout_hex.len() + receipt.stderr_hex.len()
                > self.policy.output_bytes as usize * 2
            || ![
                "completed",
                "failed",
                "timed_out",
                "cancelled",
                "control_lost",
                "invalid_control",
                "output_exceeded",
                "supervisor_error",
                "cpu_exceeded",
            ]
            .contains(&receipt.status.as_str())
            || (receipt.status == "completed"
                && (!self.authorized
                    || !receipt.cleanup_confirmed
                    || receipt.exit_code != Some(0)
                    || receipt.cleanup_overshoot_ns != Some(0)
                    || receipt
                        .cpu_used_ns
                        .is_none_or(|cpu| cpu >= self.policy.cpu_seconds * 1_000_000_000)
                    || receipt.peak_rss_bytes.is_none()
                    || events.get("worker_started_ns").is_none()
                    || events.get("cleanup_confirmed_ns").is_none()))
        {
            return Err(invalid("receipt identity/resource/completion mismatch"));
        }
        let (cpu, rss) = usage();
        receipt.controller_observation = Some(ControllerObservation {
            process_cpu_delta_ns: cpu.saturating_sub(self.controller_cpu_start),
            process_peak_rss_bytes: rss,
            elapsed_ns: self.started.elapsed().as_nanos(),
        });
        Ok(Some(receipt))
    }

    fn drain(&mut self) -> io::Result<()> {
        let stdout = self.child.stdout.as_mut().unwrap();
        let mut buffer = [0; 8192];
        for _ in 0..24 {
            match stdout.read(&mut buffer) {
                Ok(0) => break,
                Ok(count) => {
                    if self.bytes.len() + count > MAX_RECEIPT {
                        self.control.take();
                        return Err(invalid("oversized supervisor receipt"));
                    }
                    self.bytes.extend_from_slice(&buffer[..count]);
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => break,
                Err(error) => {
                    self.control.take();
                    return Err(error);
                }
            }
        }
        Ok(())
    }
}

impl Drop for Supervisor {
    fn drop(&mut self) {
        self.control.take();
        if !self.finished {
            let until = Instant::now() + Duration::from_millis(2200);
            while Instant::now() < until {
                if matches!(self.child.try_wait(), Ok(Some(_))) {
                    return;
                }
                std::thread::sleep(Duration::from_millis(2));
            }
            let _ = self.child.kill();
            let reap_until = Instant::now() + Duration::from_millis(200);
            while Instant::now() < reap_until {
                if matches!(self.child.try_wait(), Ok(Some(_))) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(2));
            }
        }
    }
}
