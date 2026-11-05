//! Local operator-owned consume-once publisher. No TCP, shell, callbacks, code
//! qualification or researcher execution. Deployment is an explicit root action.
//! Every launch consumes outside researcher-owned SQLite, BEFORE any work.
use super::*;
use std::net::Shutdown;
use std::os::unix::fs::{FileTypeExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::time::Duration;
const SOCKET: &str = "/etc/neural-foundry/manual-execution/consume.sock";
const MAX: u64 = 65_536;

pub(super) fn consumer() -> Result<()> {
    if unsafe { libc::getuid() } == 0 || unsafe { libc::geteuid() } == 0 {
        return Err(denied());
    }
    // Credentials/capabilities are per THREAD, not necessarily the leader's.
    unprivileged_status(&std::fs::read_to_string("/proc/thread-self/status").map_err(io)?)
}
pub(super) fn unprivileged_status(status: &str) -> Result<()> {
    for key in ["CapPrm:", "CapEff:", "CapAmb:"] {
        if status
            .lines()
            .find_map(|line| line.strip_prefix(key))
            .and_then(|v| u64::from_str_radix(v.trim(), 16).ok())
            != Some(0)
        {
            return Err(denied());
        }
    }
    let uids = status
        .lines()
        .find_map(|line| line.strip_prefix("Uid:"))
        .ok_or_else(denied)?;
    let uids = uids
        .split_whitespace()
        .map(str::parse::<u32>)
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| denied())?;
    if uids.len() != 4 || uids.contains(&0) {
        return Err(denied());
    }
    Ok(())
}
fn peer(stream: &UnixStream) -> Result<libc::ucred> {
    let mut credentials: libc::ucred = unsafe { std::mem::zeroed() };
    let mut size = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    if unsafe {
        libc::getsockopt(
            stream.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_PEERCRED,
            (&mut credentials as *mut libc::ucred).cast(),
            &mut size,
        )
    } != 0
        || size as usize != std::mem::size_of::<libc::ucred>()
    {
        return Err(denied());
    }
    Ok(credentials)
}
fn bounded(stream: &UnixStream) -> Result<()> {
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .map_err(io)?;
    stream
        .set_write_timeout(Some(Duration::from_secs(2)))
        .map_err(io)
}
fn read(stream: &mut UnixStream) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    loop {
        let remaining = deadline
            .checked_duration_since(std::time::Instant::now())
            .ok_or_else(denied)?;
        stream.set_read_timeout(Some(remaining)).map_err(io)?;
        let mut buffer = [0u8; 8192];
        let n = stream.read(&mut buffer).map_err(io)?;
        if n == 0 {
            return Ok(bytes);
        }
        bytes.extend_from_slice(&buffer[..n]);
        if bytes.len() as u64 > MAX {
            return Err(denied());
        }
    }
}
pub(super) fn consume_local(request: &ExistingRequest, authority: &Authority) -> Result<()> {
    request.validate()?;
    let id = request.authorization.as_ref().ok_or_else(denied)?;
    let _lock = authority.lock(id, false)?;
    let contract = frozen_contract(request)?;
    authority.verify(request, &contract)?;
    // NOREPLACE makes consumption monotonic even across crashes and same-inode
    // database rollback. An unknown response burns the grant; never auto-retry.
    authority.publish(
        &format!("{}.consumed", id.as_str()),
        canonical(request)?.bytes(),
    )
}
pub(super) fn handle(mut stream: UnixStream, authority: &Authority) -> Result<()> {
    bounded(&stream)?;
    let uid = peer(&stream)?.uid;
    let request: ExistingRequest = decode(&read(&mut stream)?)?;
    request.validate()?;
    // A foreign local uid cannot burn somebody else's grant.
    if uid == 0
        || std::fs::metadata(&request.store).map_err(io)?.uid() != uid
        || std::fs::metadata(&request.ledger).map_err(io)?.uid() != uid
    {
        return Err(denied());
    }
    consume_local(&request, authority)?;
    stream
        .write_all(
            canonical(&json!({"consumed":true,"authorization":request.authorization}))?.bytes(),
        )
        .map_err(io)
}
/// Explicit operator entrypoint; never creates/relaxes the registry or removes a
/// stale socket. A root service restart requires operator inspection/removal of
/// that stale socket, not a researcher-selected path or environment override.
pub(super) fn serve(bytes: &[u8]) -> Result<Value> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Request {
        schema_version: u32,
    }
    if unsafe { libc::getuid() } != 0 || unsafe { libc::geteuid() } != 0 {
        return Err(denied());
    }
    if decode::<Request>(bytes)?.schema_version != 1 {
        return Err(denied());
    }
    // A long-running privileged publisher must itself be an operator-installed
    // immutable executable, not the researcher's mutable build output.
    let executable = std::fs::read_link("/proc/self/exe").map_err(io)?;
    let image = std::fs::metadata(&executable).map_err(io)?;
    if !image.is_file() || image.uid() != 0 || image.mode() & 0o022 != 0 {
        return Err(denied());
    }
    Authority::open(executable.parent().ok_or_else(denied)?, 0)?;
    let authority = Authority::open(Path::new(AUTHORITY), 0)?;
    let listener = UnixListener::bind(SOCKET).map_err(io)?;
    std::fs::set_permissions(SOCKET, std::fs::Permissions::from_mode(0o666)).map_err(io)?;
    for stream in listener.incoming() {
        // Bad/abandoned requests cannot stop service or roll back consumed grants.
        match stream {
            Ok(stream) => {
                let _ = handle(stream, &authority);
            }
            Err(e) => return Err(io(e)), // Never spin on an exhausted listener.
        }
    }
    Err(denied())
}
fn connect() -> Result<UnixStream> {
    // Nonblocking connect avoids unbounded waits on a full Unix accept queue.
    let fd = unsafe {
        libc::socket(
            libc::AF_UNIX,
            libc::SOCK_STREAM | libc::SOCK_CLOEXEC | libc::SOCK_NONBLOCK,
            0,
        )
    };
    if fd < 0 {
        return Err(denied());
    }
    let stream = unsafe { UnixStream::from_raw_fd(fd) };
    let mut address: libc::sockaddr_un = unsafe { std::mem::zeroed() };
    address.sun_family = libc::AF_UNIX as libc::sa_family_t;
    for (out, byte) in address.sun_path.iter_mut().zip(SOCKET.bytes()) {
        *out = byte as libc::c_char;
    }
    if unsafe {
        libc::connect(
            fd,
            (&address as *const libc::sockaddr_un).cast(),
            std::mem::size_of::<libc::sockaddr_un>() as libc::socklen_t,
        )
    } != 0
    {
        return Err(denied());
    }
    stream.set_nonblocking(false).map_err(io)?;
    Ok(stream)
}
pub(super) fn consume(request: &ExistingRequest, authority: &Authority) -> Result<()> {
    // Private compile-time seam ONLY. It does not claim uid-0 authentication.
    #[cfg(test)]
    if authority.uid == unsafe { libc::getuid() } {
        return consume_local(request, authority);
    }
    consumer()?;
    let _ = authority;
    let metadata = std::fs::symlink_metadata(SOCKET).map_err(|_| {
        error(
            ErrorCode::UnsupportedCapability,
            "operator consume-once service is not installed; no execution admitted",
        )
    })?;
    if metadata.uid() != 0 || !metadata.file_type().is_socket() {
        return Err(denied());
    }
    exchange(connect()?, request, 0)
}
// Production always supplies uid 0. Only private tests pass their isolated
// operator principal; no public operation can select a peer or socket path.
pub(super) fn exchange(
    mut stream: UnixStream,
    request: &ExistingRequest,
    operator_uid: u32,
) -> Result<()> {
    if peer(&stream)?.uid != operator_uid {
        return Err(denied());
    }
    bounded(&stream)?;
    stream.write_all(canonical(request)?.bytes()).map_err(io)?;
    stream.shutdown(Shutdown::Write).map_err(io)?;
    let response: Value = decode(&read(&mut stream)?)?;
    if response != json!({"consumed":true,"authorization":request.authorization}) {
        return Err(denied());
    }
    Ok(())
}
#[cfg(test)]
thread_local! {
    static FAULT: std::cell::Cell<Option<&'static str>> = const { std::cell::Cell::new(None) };
    static CRASH: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
pub(super) fn checkpoint(_point: &'static str) -> Result<()> {
    #[cfg(test)]
    if FAULT.with(|slot| slot.get() == Some(_point)) {
        if CRASH.with(|slot| slot.get()) {
            std::process::exit(74);
        }
        return Err(error(ErrorCode::Io, "injected registry crash"));
    }
    Ok(())
}
#[cfg(test)]
pub(super) fn test_fault(point: Option<&'static str>) {
    FAULT.with(|slot| slot.set(point));
}
#[cfg(test)]
pub(super) fn test_crash(point: &'static str) {
    test_fault(Some(point));
    CRASH.with(|slot| slot.set(true));
}
