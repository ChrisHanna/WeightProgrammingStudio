//! Native registration/recovery for the fixed NON-FORKING reload child.
//! The worker must call registration with its owned pidfd while work is gated.
//! This is cleanup authority only: it never constructs a reload witness.
use super::*;
use std::io::Read;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Identity {
    pid: u32,
    parent: u32,
    group: u32,
    start_ticks: u64,
    boot: String,
}
fn stat(pid: u32) -> Result<Option<Identity>> {
    let file = match std::fs::File::open(format!("/proc/{pid}/stat")) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(failure(ErrorCode::Io, &e.to_string())),
    };
    let mut text = String::new();
    file.take(8193)
        .read_to_string(&mut text)
        .map_err(|e| failure(ErrorCode::Io, &e.to_string()))?;
    if text.len() > 8192 {
        return Err(deny("bounded child identity required"));
    }
    let (_, tail) = text
        .rsplit_once(')')
        .ok_or_else(|| deny("malformed child identity"))?;
    let fields = tail.split_whitespace().collect::<Vec<_>>();
    let field = |n: usize| {
        fields
            .get(n)
            .ok_or_else(|| deny("missing child identity field"))
    };
    Ok(Some(Identity {
        pid,
        parent: field(1)?
            .parse()
            .map_err(|_| deny("invalid child parent"))?,
        group: field(2)?.parse().map_err(|_| deny("invalid child group"))?,
        start_ticks: field(19)?
            .parse()
            .map_err(|_| deny("invalid child start"))?,
        boot: files::moment()?.boot,
    }))
}
fn exited(fd: &OwnedFd) -> Result<bool> {
    let mut poll = libc::pollfd {
        fd: fd.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    let result = unsafe { libc::poll(&mut poll, 1, 0) };
    if result < 0 || poll.revents & libc::POLLNVAL != 0 {
        return Err(deny("child pidfd poll failed"));
    }
    Ok(poll.revents & (libc::POLLIN | libc::POLLHUP) != 0)
}
fn pid(fd: &OwnedFd) -> Result<u32> {
    let mut text = String::new();
    std::fs::File::open(format!("/proc/self/fdinfo/{}", fd.as_raw_fd()))
        .and_then(|f| f.take(8193).read_to_string(&mut text))
        .map_err(|e| failure(ErrorCode::Io, &e.to_string()))?;
    if text.len() > 8192 {
        return Err(deny("bounded pidfd metadata required"));
    }
    text.lines()
        .find_map(|line| {
            line.strip_prefix("Pid:")
                .and_then(|s| s.trim().parse::<u32>().ok())
        })
        .filter(|pid| *pid > 1)
        .ok_or_else(|| deny("live native pidfd required"))
}
fn binding(registration: &Value, intent_id: &ContentDigest, intent: &Value) -> Result<Identity> {
    let identity: Identity = serde_json::from_value(registration["process"].clone())
        .map_err(|_| deny("invalid registered child"))?;
    if registration["intent"] != json!(intent_id)
        || registration["reservation"] != intent["reservation"]
        || registration["epoch"] != intent["epoch"]
        || registration["scope"] != "fixed_nonforking_reload_child"
        || identity.pid <= 1
        || identity.group != identity.pid
        || identity.start_ticks == 0
        || json!(identity.boot) != intent["boot"]
        || json!(identity.parent) != intent["native_process"]["pid"]
    {
        return Err(deny("registered child binding differs"));
    }
    Ok(identity)
}
impl Ledger {
    /// Call ONLY from the trusted reload launch handshake, before releasing its
    /// work gate. Failed registration must kill/reap the still-owned child.
    /// No pid, scratch JSON or caller-selected process identity is accepted.
    pub(crate) fn register_fixture_reload_child(
        &mut self,
        lease: &Lease,
        fd: &OwnedFd,
    ) -> Result<ContentDigest> {
        self.files.verify(self.limits.max_database_bytes)?;
        let identity = stat(pid(fd)?)?.ok_or_else(|| deny("child vanished before registration"))?;
        if identity.parent != std::process::id() || identity.group != identity.pid || exited(fd)? {
            return Err(deny("owned live dedicated reload child required"));
        }
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sql)?;
        let now = files::moment()?;
        check_lease(&tx, &self.store, lease, &now)?;
        let (intent_id, intent) = read(&tx, &lease.contract, "intent")?
            .ok_or_else(|| deny("missing promotion intent"))?;
        let child = read(&tx, &lease.contract, "child_intent")?
            .ok_or_else(|| deny("missing child admission"))?
            .1;
        let reservation: ContentDigest = serde_json::from_value(intent["reservation"].clone())
            .map_err(|_| deny("missing child reservation"))?;
        if child["intent"] != json!(intent_id)
            || intent["native_process"] != json!(process::Owner::current()?)
            || intent["store"] != self.store
            || intent["contract"] != json!(lease.contract)
            || intent["epoch"] != lease.epoch
            || intent["deadline"].as_i64().is_none_or(|d| now.ms >= d)
            || supervision::stopped(&tx, &lease.contract)?
            || budgets::snapshot_in(&tx, &lease.contract)?.accounting_halted
            || budgets::native_released(&tx, &reservation)?
            || budgets::native_kind(&tx, &reservation)? != WorkKind::Reload
            || history::budget_row(&tx, "budget_requests", "id", &reservation)?["contract"]
                != json!(lease.contract)
            || read(&tx, &lease.contract, "attempt_end")?.is_some()
        {
            return Err(deny("stale or foreign reload registration"));
        }
        let value = json!({"intent":intent_id,"reservation":reservation,"epoch":lease.epoch,
            "process":identity,"scope":"fixed_nonforking_reload_child"});
        binding(&value, &intent_id, &intent)?;
        let id = record(&tx, &lease.contract, "child_registered", &value)?;
        if exited(fd)? {
            return Err(deny("child exited during registration"));
        }
        ensure_live(lease)?;
        tx.commit().map_err(sql)?;
        Ok(id)
    }
}
/// Read-only native cross-check: a report alone cannot clean a registered live
/// child. Boot/start mismatch proves the original fixed nonforking child gone.
pub(super) fn registered_gone(
    c: &Connection,
    contract: &ContentDigest,
    intent_id: &ContentDigest,
    intent: &Value,
) -> Result<Option<bool>> {
    let Some((_, registration)) = read(c, contract, "child_registered")? else {
        return Ok(None);
    };
    let identity = binding(&registration, intent_id, intent)?;
    if identity.boot != files::moment()?.boot {
        return Ok(Some(true));
    }
    let raw = unsafe { libc::syscall(libc::SYS_pidfd_open, identity.pid, 0) };
    if raw < 0 {
        return if std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH) {
            Ok(Some(true))
        } else {
            Err(deny("inventory pidfd unavailable"))
        };
    }
    let fd = unsafe { OwnedFd::from_raw_fd(raw as i32) };
    let same = stat(identity.pid)?.is_some_and(|p| p.start_ticks == identity.start_ticks);
    Ok(Some(!same || exited(&fd)?))
}

/// A registered child may be killed only through a pinned kernel handle after
/// verifying boot/start identity. Never signal a process group or recycled PID.
pub(super) fn recover(
    tx: &Transaction<'_>,
    contract: &ContentDigest,
    intent_id: &ContentDigest,
    intent: &Value,
) -> Result<bool> {
    let Some((registration_id, registration)) = read(tx, contract, "child_registered")? else {
        return Ok(false);
    };
    let identity = binding(&registration, intent_id, intent)?;
    let mut reaped = false;
    if identity.boot == files::moment()?.boot {
        let raw = unsafe { libc::syscall(libc::SYS_pidfd_open, identity.pid, 0) };
        if raw < 0 {
            if std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH) {
                return Err(deny("recovery pidfd unavailable"));
            }
        } else {
            let fd = unsafe { OwnedFd::from_raw_fd(raw as i32) };
            // Open first, inspect second: disappearance/reuse cannot redirect a
            // signal after the check. The child is pinned by fd, not by PID.
            let same = stat(identity.pid)?.is_some_and(|p| p.start_ticks == identity.start_ticks);
            if same {
                if !exited(&fd)?
                    && unsafe {
                        libc::syscall(
                            libc::SYS_pidfd_send_signal,
                            fd.as_raw_fd(),
                            libc::SIGKILL,
                            std::ptr::null::<libc::siginfo_t>(),
                            0,
                        )
                    } < 0
                    && std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
                {
                    return Err(deny("registered child termination failed"));
                }
                let until = std::time::Instant::now() + Duration::from_millis(500);
                while !exited(&fd)? {
                    if std::time::Instant::now() >= until {
                        return Err(deny("registered child exit unconfirmed"));
                    }
                    std::thread::sleep(Duration::from_millis(2));
                }
                // Reap if recovery owns/adopted the child. ECHILD means its
                // actual parent/reaper owns the zombie; pidfd still proves exit.
                loop {
                    let mut info = std::mem::MaybeUninit::<libc::siginfo_t>::zeroed();
                    let result = unsafe {
                        libc::waitid(
                            libc::P_PIDFD,
                            fd.as_raw_fd() as u32,
                            info.as_mut_ptr(),
                            libc::WEXITED | libc::WNOHANG,
                        )
                    };
                    if result == 0 && unsafe { info.assume_init().si_pid() } != 0 {
                        reaped = true;
                        break;
                    }
                    if result < 0
                        && std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted
                    {
                        if std::time::Instant::now() >= until {
                            return Err(deny("registered child reap deadline exceeded"));
                        }
                        std::thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    if result < 0
                        && std::io::Error::last_os_error().raw_os_error() == Some(libc::ECHILD)
                    {
                        break;
                    }
                    return Err(deny("registered child reap unconfirmed"));
                }
            }
        }
    }
    record(
        tx,
        contract,
        "child_recovery",
        &json!({"intent":intent_id,"registration":registration_id,
        "status":"registered_nonforking_child_gone","cleanup_confirmed":true,"reaped_by_recovery":reaped}),
    )?;
    Ok(true)
}
pub(super) fn recovery_clean(
    c: &Connection,
    contract: &ContentDigest,
    intent_id: &ContentDigest,
    intent: &Value,
    recovery: &Value,
) -> Result<bool> {
    let (id, registration) = read(c, contract, "child_registered")?
        .ok_or_else(|| deny("missing recovered registration"))?;
    binding(&registration, intent_id, intent)?;
    if recovery["intent"] != json!(intent_id)
        || recovery["registration"] != json!(id)
        || recovery["cleanup_confirmed"] != true
        || recovery["reaped_by_recovery"].as_bool().is_none()
    {
        return Err(deny("registered child recovery binding differs"));
    }
    Ok(true)
}
