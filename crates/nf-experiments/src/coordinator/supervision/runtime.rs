use super::*;
use std::io::{Read, Write};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
/// Controller/operator-selected trusted native fixture. Never construct from
/// untrusted gym metadata; this is resource qualification, not code authentication.
pub struct TrustedNativeWorker {
    path: PathBuf,
    digest: ContentDigest,
    limiter: PathBuf,
    limiter_digest: ContentDigest,
}
fn file_digest(path: &Path) -> Result<ContentDigest> {
    use rustix::fs::{fstat, open, FileType, Mode, OFlags};
    if !path.is_absolute() {
        return Err(failure(
            ErrorCode::InvalidValue,
            "trusted worker path must be absolute",
        ));
    }
    let fd = open(
        path,
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(|error| failure(ErrorCode::Io, &error.to_string()))?;
    let stat = fstat(&fd).map_err(|error| failure(ErrorCode::Io, &error.to_string()))?;
    if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile
        || stat.st_nlink != 1
        || stat.st_mode & 0o022 != 0
        || stat.st_size < 0
        || stat.st_size > 67_108_864
    {
        return Err(failure(
            ErrorCode::PermissionDenied,
            "trusted executable type/link/permission/size invalid",
        ));
    }
    let mut bytes = Vec::new();
    std::fs::File::from(fd)
        .take(67_108_865)
        .read_to_end(&mut bytes)
        .map_err(|error| failure(ErrorCode::Io, &error.to_string()))?;
    if bytes.len() > 67_108_864 {
        return Err(failure(ErrorCode::LimitExceeded, "executable grew"));
    }
    Ok(ContentDigest::of_bytes(&bytes))
}
impl TrustedNativeWorker {
    pub fn qualify(path: &Path, operator_approved_digest: ContentDigest) -> Result<Self> {
        let digest = file_digest(path)?;
        if digest != operator_approved_digest {
            return Err(failure(
                ErrorCode::SourceMismatch,
                "trusted worker differs from operator approval",
            ));
        }
        let limiter = PathBuf::from("/usr/bin/prlimit");
        let limiter_digest = file_digest(&limiter)?;
        let worker = Self {
            path: path.to_owned(),
            digest,
            limiter,
            limiter_digest,
        };
        let mut child = worker
            .command(134_217_728, 1, 4096)
            .arg("--probe")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| failure(ErrorCode::Io, &error.to_string()))?;
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            if child
                .try_wait()
                .map_err(|error| failure(ErrorCode::Io, &error.to_string()))?
                .is_some()
            {
                break;
            }
            if Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                return Err(failure(
                    ErrorCode::UnsupportedCapability,
                    "native resource probe did not terminate",
                ));
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        let mut output = Vec::new();
        child
            .stdout
            .take()
            .unwrap()
            .take(4097)
            .read_to_end(&mut output)
            .map_err(|error| failure(ErrorCode::Io, &error.to_string()))?;
        let proof = CanonicalDocument::parse(
            &output,
            Limits {
                max_bytes: 4096,
                ..Limits::default()
            },
        )?;
        let value = proof.value();
        if value["protocol"] != "nf.trusted-fixture-worker.v1"
            || value["address_space"] != 134_217_728u64
            || value["cpu_seconds"] != 1u64
            || value["file_bytes"] != 4096u64
            || value["pid"] != child.id()
            || value["pgrp"] != child.id()
        {
            return Err(failure(
                ErrorCode::UnsupportedCapability,
                "actual child resource/group probe failed",
            ));
        }
        Ok(worker)
    }
    pub(super) fn digest(&self) -> &ContentDigest {
        &self.digest
    }
    pub(super) fn verify(&self) -> Result<()> {
        if file_digest(&self.path)? != self.digest
            || file_digest(&self.limiter)? != self.limiter_digest
        {
            return Err(failure(
                ErrorCode::SourceMismatch,
                "trusted worker/limiter changed since qualification",
            ));
        }
        Ok(())
    }
    fn command(&self, memory: u64, cpu_seconds: u64, output: u64) -> Command {
        let mut command = Command::new(&self.limiter);
        command
            .args([
                format!("--as={memory}:{memory}"),
                format!("--cpu={cpu_seconds}:{cpu_seconds}"),
                format!("--fsize={output}:{output}"),
                "--core=0:0".to_owned(),
                "--nofile=64:64".to_owned(),
                "--".to_owned(),
            ])
            .arg(&self.path)
            .env_clear()
            .current_dir("/")
            .process_group(0);
        command
    }
}
fn proc_stat(pid: u32) -> Result<Option<(u32, u64, bool)>> {
    let file = match std::fs::File::open(format!("/proc/{pid}/stat")) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(failure(ErrorCode::Io, &error.to_string())),
    };
    let mut text = String::new();
    file.take(8193)
        .read_to_string(&mut text)
        .map_err(|error| failure(ErrorCode::Io, &error.to_string()))?;
    if text.len() > 8192 {
        return Err(failure(ErrorCode::LimitExceeded, "process stat bound"));
    }
    let end = text
        .rfind(')')
        .ok_or_else(|| failure(ErrorCode::IntegrityMismatch, "process stat malformed"))?;
    let fields: Vec<_> = text[end + 1..].split_whitespace().collect();
    let group = fields
        .get(2)
        .and_then(|value| value.parse().ok())
        .ok_or_else(|| failure(ErrorCode::IntegrityMismatch, "process group missing"))?;
    let start = fields
        .get(19)
        .and_then(|value| value.parse().ok())
        .ok_or_else(|| {
            failure(
                ErrorCode::IntegrityMismatch,
                "process start identity missing",
            )
        })?;
    Ok(Some((group, start, fields[0] == "Z")))
}
fn identity(pid: u32) -> Result<ProcessIdentity> {
    let (group, start_ticks, _) = proc_stat(pid)?.ok_or_else(|| {
        failure(
            ErrorCode::TransactionAborted,
            "worker exited before identity registration",
        )
    })?;
    if pid <= 1 || group != pid {
        return Err(failure(
            ErrorCode::UnsupportedCapability,
            "worker is not its dedicated process-group leader",
        ));
    }
    Ok(ProcessIdentity {
        pid,
        group,
        start_ticks,
        boot: files::moment()?.boot,
    })
}
fn matches(identity: &ProcessIdentity) -> Result<bool> {
    Ok(identity.boot == files::moment()?.boot
        && proc_stat(identity.pid)?.is_some_and(|(group, start, _)| {
            group == identity.group && start == identity.start_ticks
        }))
}
fn signal(identity: &ProcessIdentity, signal: rustix::process::Signal) -> Result<()> {
    if !matches(identity)? {
        return Err(failure(
            ErrorCode::SourceMismatch,
            "refusing signal to reused/unconfirmed process identity",
        ));
    }
    let pid = rustix::process::Pid::from_raw(identity.group as i32)
        .ok_or_else(|| failure(ErrorCode::InvalidValue, "invalid process group"))?;
    rustix::process::kill_process_group(pid, signal)
        .map_err(|error| failure(ErrorCode::Io, &error.to_string()))
}
fn live_group(group: u32) -> Result<bool> {
    let mut count = 0;
    for entry in
        std::fs::read_dir("/proc").map_err(|error| failure(ErrorCode::Io, &error.to_string()))?
    {
        count += 1;
        if count > 65536 {
            return Err(failure(ErrorCode::LimitExceeded, "process inventory bound"));
        }
        let entry = entry.map_err(|error| failure(ErrorCode::Io, &error.to_string()))?;
        if let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|name| name.parse::<u32>().ok())
        {
            if proc_stat(pid)?.is_some_and(|(actual, _, zombie)| actual == group && !zombie) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}
pub(super) fn recover_group(process: &ProcessIdentity) -> Result<bool> {
    // Never use a reused PID as a new authority. Missing leader is deliberately
    // unconfirmed unless the whole dedicated group is observed absent/dead.
    if matches(process)? {
        let _ = signal(process, rustix::process::Signal::TERM);
        std::thread::sleep(Duration::from_millis(20));
        if matches(process)? {
            let _ = signal(process, rustix::process::Signal::KILL);
        }
    } else if proc_stat(process.pid)?.is_some() {
        return Ok(false);
    }
    let deadline = Instant::now() + Duration::from_millis(500);
    loop {
        if !live_group(process.group)? {
            return Ok(true);
        }
        if Instant::now() >= deadline {
            return Ok(false);
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}
fn capture(
    mut input: impl Read + Send + 'static,
    limit: usize,
    overflow: Arc<AtomicBool>,
) -> JoinHandle<Vec<u8>> {
    std::thread::spawn(move || {
        let mut output = Vec::new();
        let mut buffer = [0; 1024];
        loop {
            match input.read(&mut buffer) {
                Ok(0) => break,
                Ok(count) => {
                    let remaining = limit.saturating_sub(output.len());
                    output.extend_from_slice(&buffer[..count.min(remaining)]);
                    if count > remaining {
                        overflow.store(true, Ordering::Release);
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => break,
            }
        }
        output
    })
}
/// Owned live process; dropping it kills only its verified dedicated group.
pub struct RunningNative {
    id: ContentDigest,
    child: Child,
    identity: ProcessIdentity,
    stdout: Option<JoinHandle<Vec<u8>>>,
    stderr: Option<JoinHandle<Vec<u8>>>,
    overflow: Arc<AtomicBool>,
    deadline: i64,
    grace: u64,
    reaped: bool,
    delivered: bool,
}
impl RunningNative {
    pub fn id(&self) -> &ContentDigest {
        &self.id
    }
    pub fn identity(&self) -> &ProcessIdentity {
        &self.identity
    }
    pub(super) fn spawn(worker: &TrustedNativeWorker, intent: &Intent) -> Result<Self> {
        let mut child = worker
            .command(
                intent.host_bytes,
                intent.cpu_seconds,
                intent.job.maximum_output_bytes,
            )
            .args(["--worker", &intent.wall_deadline_ms.to_string()])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| failure(ErrorCode::Io, &error.to_string()))?;
        let process = match identity(child.id()) {
            Ok(identity) => identity,
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(error);
            }
        };
        let overflow = Arc::new(AtomicBool::new(false));
        let stdout = capture(
            child.stdout.take().unwrap(),
            intent.job.maximum_output_bytes as usize,
            overflow.clone(),
        );
        let stderr = capture(
            child.stderr.take().unwrap(),
            intent.job.maximum_output_bytes as usize,
            overflow.clone(),
        );
        Ok(Self {
            id: intent.id.clone(),
            child,
            identity: process,
            stdout: Some(stdout),
            stderr: Some(stderr),
            overflow,
            deadline: intent.wall_deadline_ms,
            grace: intent.job.termination_grace_ms,
            reaped: false,
            delivered: false,
        })
    }
    pub(super) fn start(&mut self, intent: &Intent) -> Result<()> {
        let message = canonical(&serde_json::json!({"mode":intent.job.mode.name()}))?;
        let mut pipe = self.child.stdin.take().ok_or_else(|| {
            failure(
                ErrorCode::IntegrityMismatch,
                "worker start already delivered",
            )
        })?;
        pipe.write_all(&message)
            .map_err(|error| failure(ErrorCode::Io, &error.to_string()))?;
        drop(pipe);
        Ok(())
    }
    pub(super) fn stop_cause(&self) -> Result<Option<StopReason>> {
        if self.overflow.load(Ordering::Acquire) {
            Ok(Some(StopReason::OutputLimit))
        } else if files::moment()?.ms >= self.deadline {
            Ok(Some(StopReason::Deadline))
        } else {
            Ok(None)
        }
    }
    pub(super) fn poll(&mut self, stop: bool) -> Result<Option<WorkerReport>> {
        if self.delivered {
            return Ok(None);
        }
        if !matches(&self.identity)? {
            return Err(failure(
                ErrorCode::SourceMismatch,
                "owned worker identity disappeared/reused before reap",
            ));
        }
        let exited = proc_stat(self.identity.pid)?.is_some_and(|(_, _, zombie)| zombie);
        let reason = self.stop_cause()?;
        if !stop && !exited && reason.is_none() {
            return Ok(None);
        }
        if stop || reason.is_some() {
            let _ = signal(&self.identity, rustix::process::Signal::TERM);
            let end = Instant::now()
                + Duration::from_millis(
                    self.grace
                        .min(self.deadline.saturating_sub(files::moment()?.ms).max(0) as u64),
                );
            while Instant::now() < end {
                if !live_group(self.identity.group)? {
                    break;
                }
                std::thread::sleep(Duration::from_millis(2));
            }
        }
        // Reap only after group cleanup, so the leader PID cannot be reused
        // while it still authorizes a signal to lingering descendants.
        if matches(&self.identity)? {
            let _ = signal(&self.identity, rustix::process::Signal::KILL);
        }
        let end = Instant::now() + Duration::from_millis(500);
        let mut status = 0;
        // SAFETY: rusage/status are valid POD output buffers, and this handle
        // owns the unreaped child PID; no other wait operation runs on it.
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        loop {
            let waited = unsafe {
                libc::wait4(
                    self.identity.pid as i32,
                    &mut status,
                    libc::WNOHANG,
                    &mut usage,
                )
            };
            if waited == self.identity.pid as i32 {
                self.reaped = true;
                break;
            }
            if waited < 0 {
                return Err(failure(ErrorCode::Io, "wait4 failed for owned child"));
            }
            if Instant::now() >= end {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        let cleanup = self.reaped && !live_group(self.identity.group)?;
        let (stdout, stderr) = if cleanup {
            (
                self.stdout.take().unwrap().join().unwrap_or_default(),
                self.stderr.take().unwrap().join().unwrap_or_default(),
            )
        } else {
            (vec![], vec![])
        };
        let exit_code = if self.reaped && libc::WIFEXITED(status) {
            Some(libc::WEXITSTATUS(status))
        } else {
            None
        };
        let signal = if self.reaped && libc::WIFSIGNALED(status) {
            Some(libc::WTERMSIG(status))
        } else {
            None
        };
        let outcome = if !cleanup {
            WorkerOutcome::Incomplete
        } else if self.overflow.load(Ordering::Acquire) {
            WorkerOutcome::OutputExceeded
        } else if reason == Some(StopReason::Deadline) {
            WorkerOutcome::TimedOut
        } else if stop {
            WorkerOutcome::Cancelled
        } else if exit_code == Some(0) {
            WorkerOutcome::Completed
        } else {
            WorkerOutcome::ResourceFailure
        };
        let cpu = if self.reaped {
            Some(
                (usage.ru_utime.tv_sec as u64 + usage.ru_stime.tv_sec as u64) * 1000
                    + (usage.ru_utime.tv_usec as u64 + usage.ru_stime.tv_usec as u64) / 1000,
            )
        } else {
            None
        };
        self.delivered = true;
        Ok(Some(WorkerReport {
            outputs_eligible: outcome == WorkerOutcome::Completed,
            cleanup_confirmed: cleanup,
            outcome,
            stdout,
            stderr,
            exit_code,
            signal,
            cpu_ms: cpu,
            leader_peak_rss_bytes: if self.reaped {
                Some(usage.ru_maxrss as u64 * 1024)
            } else {
                None
            },
            measurement_scope: "kernel wait4 leader only; descendant CPU remains unknown"
                .to_owned(),
        }))
    }
    pub(super) fn abort_owned(&mut self) {
        if !self.reaped {
            let _ = signal(&self.identity, rustix::process::Signal::KILL);
            let _ = self.poll(true);
        }
    }
}
impl Drop for RunningNative {
    fn drop(&mut self) {
        self.abort_owned();
    }
}
