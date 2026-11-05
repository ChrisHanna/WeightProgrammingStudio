//! OS-separated operator authority. The researcher uid cannot mint grants.
//! Root is trusted; a process running as root is outside this threat model.
use super::*;
use std::fs::File;
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::fs::MetadataExt;
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(test)]
#[path = "authorization_tests.rs"]
mod tests;

#[path = "authority_service.rs"]
mod authority_service;
const AUTHORITY: &str = "/etc/neural-foundry/manual-execution";
fn denied() -> CoreError {
    error(
        ErrorCode::PermissionDenied,
        "missing, forged, revoked, expired or out-of-scope operator grant",
    )
}
fn io(e: std::io::Error) -> CoreError {
    error(ErrorCode::Io, &e.to_string())
}
fn now() -> Result<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .map_err(|_| denied())
}
#[derive(Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Identity {
    path: PathBuf,
    device: u64,
    inode: u64,
}
fn identity(path: &Path) -> Result<Identity> {
    explicit(path)?;
    let canonical = std::fs::canonicalize(path).map_err(io)?;
    if canonical != path {
        return Err(denied());
    }
    let m = std::fs::symlink_metadata(path).map_err(io)?;
    Ok(Identity {
        path: canonical,
        device: m.dev(),
        inode: m.ino(),
    })
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Grant {
    schema_version: u32,
    pub(super) operation: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) supplied: Option<supplied::Binding>,
    store: Identity,
    ledger: Identity,
    database: Identity,
    contract: ContentDigest,
    permissions: ContentDigest,
    pub(super) worker: PathBuf,
    pub(super) worker_digest: ContentDigest,
    expires_unix_seconds: u64,
}
/// Pinned root-owned directory, opened component by component without symlinks.
/// No caller path, environment override, boolean, CAS record or agent text can
/// select the production trust root. Revocations are permanent marker files.
struct Authority {
    directory: File,
    uid: u32,
}
impl Authority {
    fn open(path: &Path, uid: u32) -> Result<Self> {
        use std::ffi::CString;
        let mut directory = File::open("/").map_err(io)?;
        let root = directory.metadata().map_err(io)?;
        if root.uid() != uid || root.mode() & 0o022 != 0 {
            return Err(denied());
        }
        for part in path.components() {
            match part {
                std::path::Component::RootDir => continue,
                std::path::Component::Normal(name) => {
                    use std::os::unix::ffi::OsStrExt;
                    let name = CString::new(name.as_bytes()).map_err(|_| denied())?;
                    // SAFETY: name is NUL terminated; returned fd is exclusively owned.
                    let fd = unsafe {
                        libc::openat(
                            directory.as_raw_fd(),
                            name.as_ptr(),
                            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                        )
                    };
                    if fd < 0 {
                        return Err(denied());
                    }
                    directory = unsafe { File::from_raw_fd(fd) };
                    let m = directory.metadata().map_err(io)?;
                    if m.uid() != uid || m.mode() & 0o022 != 0 {
                        return Err(denied());
                    }
                }
                _ => return Err(denied()),
            }
        }
        Ok(Self { directory, uid })
    }
    fn file(&self, name: &str, create: bool) -> Result<File> {
        self.file_mode(name, create, 0o644)
    }
    fn file_mode(&self, name: &str, create: bool, mode: libc::mode_t) -> Result<File> {
        let name = std::ffi::CString::new(name).map_err(|_| denied())?;
        let flags = libc::O_NOFOLLOW
            | libc::O_CLOEXEC
            | libc::O_NONBLOCK
            | if create {
                libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL
            } else {
                libc::O_RDONLY
            };
        let fd = unsafe { libc::openat(self.directory.as_raw_fd(), name.as_ptr(), flags, mode) };
        if fd < 0 {
            return Err(denied());
        }
        let file = unsafe { File::from_raw_fd(fd) };
        let m = file.metadata().map_err(io)?;
        if !m.is_file() || m.uid() != self.uid || m.mode() & 0o022 != 0 || m.nlink() != 1 {
            return Err(denied());
        }
        if create && unsafe { libc::fchmod(file.as_raw_fd(), mode) } != 0 {
            return Err(io(std::io::Error::last_os_error()));
        }
        Ok(file)
    }
    // Serialize issue/revoke/consume on a stable inode that is never replaced.
    fn lock(&self, id: &ContentDigest, create: bool) -> Result<File> {
        let name = format!("{}.lock", id.as_str());
        let file = match self.file(&name, false) {
            Ok(file) => file,
            Err(_) if create => self.file_mode(&name, true, 0o600)?,
            Err(e) => return Err(e),
        };
        // Readable flock files let a researcher block operator revocation even
        // without write access. Only the operator service needs this inode.
        if file.metadata().map_err(io)?.mode() & 0o077 != 0 {
            return Err(denied());
        }
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            return Err(denied());
        }
        file.sync_all().map_err(io)?;
        self.directory.sync_all().map_err(io)?;
        Ok(file)
    }
    // Publish complete, synced bytes atomically, never replacing an old record.
    // A crash leaves at most a non-authoritative staging file.
    fn publish(&self, target: &str, bytes: &[u8]) -> Result<()> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let temporary = format!(
            ".stage-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let result = (|| {
            let mut file = self.file(&temporary, true)?;
            let split = bytes.len() / 2;
            file.write_all(&bytes[..split]).map_err(io)?;
            authority_service::checkpoint("partial_write")?;
            file.write_all(&bytes[split..]).map_err(io)?;
            file.sync_all().map_err(io)?;
            authority_service::checkpoint("before_publish")?;
            let from = std::ffi::CString::new(temporary.as_str()).map_err(|_| denied())?;
            let to = std::ffi::CString::new(target).map_err(|_| denied())?;
            if unsafe {
                libc::renameat2(
                    self.directory.as_raw_fd(),
                    from.as_ptr(),
                    self.directory.as_raw_fd(),
                    to.as_ptr(),
                    libc::RENAME_NOREPLACE,
                )
            } != 0
            {
                return Err(io(std::io::Error::last_os_error()));
            }
            authority_service::checkpoint("after_publish")?;
            self.directory.sync_all().map_err(io)
        })();
        let temporary = std::ffi::CString::new(temporary).map_err(|_| denied())?;
        unsafe {
            libc::unlinkat(self.directory.as_raw_fd(), temporary.as_ptr(), 0);
        }
        result
    }
    fn revoked(&self, id: &ContentDigest) -> Result<bool> {
        let name =
            std::ffi::CString::new(format!("{}.revoked", id.as_str())).map_err(|_| denied())?;
        let mut stat: libc::stat = unsafe { std::mem::zeroed() };
        if unsafe {
            libc::fstatat(
                self.directory.as_raw_fd(),
                name.as_ptr(),
                &mut stat,
                libc::AT_SYMLINK_NOFOLLOW,
            )
        } == 0
        {
            return Ok(true);
        }
        if std::io::Error::last_os_error().raw_os_error() == Some(libc::ENOENT) {
            Ok(false)
        } else {
            Err(denied())
        }
    }
    fn verify(
        &self,
        request: &ExistingRequest,
        contract: &Validated<StudyContract>,
    ) -> Result<(Grant, CanonicalDocument)> {
        let id = request.authorization.as_ref().ok_or_else(denied)?;
        if self.revoked(id)? {
            return Err(denied());
        }
        let mut bytes = Vec::new();
        self.file(id.as_str(), false)?
            .take(65_537)
            .read_to_end(&mut bytes)
            .map_err(io)?;
        if bytes.len() > 65_536 {
            return Err(denied());
        }
        let doc = CanonicalDocument::parse(&bytes, Limits::default())?;
        if doc.digest() != *id {
            return Err(denied());
        }
        let grant: Grant = decode(doc.bytes())?;
        if grant.schema_version != 1
            || !matches!(grant.operation.as_str(), "nf.native.complete_fixture.v1")
                && !supplied::is_supplied_operation(&grant.operation)
            || supplied::is_supplied_operation(&grant.operation) != grant.supplied.is_some()
            || grant.expires_unix_seconds <= now()?
            || grant.contract != request.contract_digest
            || contract.digest()? != grant.contract
            || grant.permissions != canonical(&contract.permissions)?.digest()
            || grant.store != identity(&request.store)?
            || grant.ledger != identity(&request.ledger)?
            || grant.database != identity(&request.ledger.join("events.sqlite3"))?
        {
            return Err(denied());
        }
        if let Some(binding) = &grant.supplied {
            supplied::verify_binding(request, contract, binding)?;
        }
        // Static input reads can be slow; check withdrawal/expiry again afterward.
        if self.revoked(id)? || grant.expires_unix_seconds <= now()? {
            return Err(denied());
        }
        Ok((grant, doc))
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct IssueRequest {
    request: ExistingRequest,
    #[serde(default = "native_operation")]
    operation: String,
    #[serde(default)]
    supplied: Option<supplied::Binding>,
    worker: PathBuf,
    worker_digest: ContentDigest,
    expires_unix_seconds: u64,
}
fn native_operation() -> String {
    "nf.native.complete_fixture.v1".into()
}
/// Only the OS operator can issue/revoke. Provision AUTHORITY root:root 0755
/// outside the research process; no library call creates or relaxes it.
pub(super) fn serve(bytes: &[u8]) -> Result<Value> {
    authority_service::serve(bytes)
}
pub(super) fn operator(operation: &str, bytes: &[u8]) -> Result<Value> {
    if unsafe { libc::getuid() } != 0 || unsafe { libc::geteuid() } != 0 {
        return Err(denied());
    }
    let authority = Authority::open(Path::new(AUTHORITY), 0)?;
    operator_with_authority(operation, bytes, &authority)
}
fn operator_with_authority(operation: &str, bytes: &[u8], authority: &Authority) -> Result<Value> {
    if operation == "revoke" {
        let request: ExistingRequest = decode(bytes)?;
        request.validate()?;
        let id = request.authorization.ok_or_else(denied)?;
        let _lock = authority.lock(&id, true)?;
        if !authority.revoked(&id)? {
            authority.publish(&format!("{}.revoked", id.as_str()), b"revoked\n")?;
        } else {
            // A previous response may have failed after publication but before
            // directory fsync. Idempotent success must still establish durability.
            authority
                .file(&format!("{}.revoked", id.as_str()), false)?
                .sync_all()
                .map_err(io)?;
            authority.directory.sync_all().map_err(io)?;
        }
        return Ok(json!({"authorization":id,"revoked":true}));
    }
    if operation != "authorize" {
        return Err(denied());
    }
    let issue: IssueRequest = decode(bytes)?;
    issue.request.validate()?;
    explicit(&issue.worker)?;
    issue.request.snapshot()?;
    if issue.expires_unix_seconds <= now()?
        || issue.expires_unix_seconds > now()?.saturating_add(86400)
    {
        return Err(denied());
    }
    let contract = frozen_contract(&issue.request)?;
    if (!matches!(issue.operation.as_str(), "nf.native.complete_fixture.v1")
        && !supplied::is_supplied_operation(&issue.operation))
        || supplied::is_supplied_operation(&issue.operation) != issue.supplied.is_some()
    {
        return Err(denied());
    }
    if let Some(binding) = &issue.supplied {
        // Static content verification only: never invoke runtime qualification,
        // smoke tests, compilation or model inference in the privileged issuer.
        supplied::verify_binding(&issue.request, &contract, binding)?;
    }
    // The privileged issuer hashes bytes only: never execute researcher-owned
    // code as root. Actual executable/resource qualification occurs later in
    // the unprivileged run process, after verifying the grant.
    use std::os::unix::fs::OpenOptionsExt;
    let binary = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(&issue.worker)
        .map_err(io)?;
    let metadata = binary.metadata().map_err(io)?;
    if !metadata.is_file() || metadata.len() > 67_108_864 || metadata.mode() & 0o111 == 0 {
        return Err(denied());
    }
    let mut code = Vec::new();
    binary.take(67_108_865).read_to_end(&mut code).map_err(io)?;
    if code.len() > 67_108_864 || ContentDigest::of_bytes(&code) != issue.worker_digest {
        return Err(denied());
    }
    let grant = Grant {
        schema_version: 1,
        operation: issue.operation,
        supplied: issue.supplied,
        store: identity(&issue.request.store)?,
        ledger: identity(&issue.request.ledger)?,
        database: identity(&issue.request.ledger.join("events.sqlite3"))?,
        contract: issue.request.contract_digest,
        permissions: canonical(&contract.permissions)?.digest(),
        worker: issue.worker,
        worker_digest: issue.worker_digest,
        expires_unix_seconds: issue.expires_unix_seconds,
    };
    let doc = canonical(&grant)?;
    let _lock = authority.lock(&doc.digest(), true)?;
    if authority.revoked(&doc.digest())? {
        return Err(denied());
    }
    authority.publish(doc.digest().as_str(), doc.bytes())?;
    Ok(json!({"authorization":doc.digest(),"grant":doc.value(),"scope":grant.operation}))
}
pub(super) fn run(
    request: &ExistingRequest,
    cancel: &nf_surgeon::cancellation::CancellationToken,
) -> Result<Value> {
    // Operator issuance is privileged; execution must never run the approved
    // researcher fixture with operator/root privileges.
    authority_service::consumer()?;
    if request.authorization.is_none() {
        return Err(denied());
    }
    run_with_cancel(request, &Authority::open(Path::new(AUTHORITY), 0)?, cancel)
}
/// The child endpoint is not an approval switch. It verifies the same OS grant,
/// the approved executable bytes, and observed irreversible kernel limits.
pub(super) fn execute_supplied(bytes: &[u8]) -> Result<Value> {
    authority_service::consumer()?;
    let request: ExistingRequest = decode(bytes)?;
    request.validate()?;
    execute_supplied_with(&request, &Authority::open(Path::new(AUTHORITY), 0)?)
}
fn execute_supplied_with(request: &ExistingRequest, authority: &Authority) -> Result<Value> {
    let contract = frozen_contract(request)?;
    let (grant, _) = authority.verify(request, &contract)?;
    if !supplied::is_supplied_operation(&grant.operation) {
        return Err(denied());
    }
    let memory = rustix::process::getrlimit(rustix::process::Resource::As);
    let cpu = rustix::process::getrlimit(rustix::process::Resource::Cpu);
    if !memory
        .maximum
        .is_some_and(|n| n <= contract.limits.host_memory_bytes)
        || !cpu
            .maximum
            .is_some_and(|n| n <= contract.limits.cpu_ms.div_ceil(1000))
    {
        return Err(denied());
    }
    let mut image = Vec::new();
    File::open("/proc/self/exe")
        .map_err(io)?
        .take(67_108_865)
        .read_to_end(&mut image)
        .map_err(io)?;
    if image.len() > 67_108_864 || ContentDigest::of_bytes(&image) != grant.worker_digest {
        return Err(denied());
    }
    authority_service::consume(request, authority)?;
    use std::sync::atomic::{AtomicBool, Ordering};
    let done = AtomicBool::new(false);
    let withdrawn = AtomicBool::new(false);
    let cancel = nf_surgeon::cancellation::CancellationToken::default();
    let result = std::thread::scope(|scope| {
        let watcher = scope.spawn(|| {
            while !done.load(Ordering::Acquire) {
                if authority.verify(request, &contract).is_err() {
                    withdrawn.store(true, Ordering::Release);
                    cancel.request();
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        });
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            supplied::execute(request, &grant, &cancel)
        }));
        done.store(true, Ordering::Release);
        watcher
            .join()
            .map_err(|_| error(ErrorCode::InternalPanic, "authority watcher failed"))?;
        result.map_err(|_| {
            error(
                ErrorCode::InternalPanic,
                "supplied dispatcher panicked; reconcile retained intent",
            )
        })?
    });
    if withdrawn.load(Ordering::Acquire) || authority.verify(request, &contract).is_err() {
        request.open()?.request_stop(
            &request.contract_digest,
            canonical(&(request.authorization.as_ref(), "supplied-grant-withdrawn"))?.digest(),
            StopReason::OwnerStop,
        )?;
        return Err(denied());
    }
    result
}
fn frozen_contract(request: &ExistingRequest) -> Result<Validated<StudyContract>> {
    let snapshot = request.snapshot()?;
    let value: Value = serde_json::from_str(&snapshot.render_json()).map_err(|_| denied())?;
    Validated::<StudyContract>::parse(
        &serde_json::to_vec(&value["frozen_contract"]).map_err(|_| denied())?,
    )
}
/// Last controller return fence: a decoded child response is not authority.
fn finish_authorized(
    request: &ExistingRequest,
    authority: &Authority,
    contract: &Validated<StudyContract>,
    cancel: &nf_surgeon::cancellation::CancellationToken,
    result: Result<Value>,
) -> Result<Value> {
    let live = cancel
        .check()
        .and_then(|_| authority.verify(request, contract).map(|_| ()));
    if let Err(refusal) = live {
        request.open()?.request_stop(
            &request.contract_digest,
            canonical(&(
                request.authorization.as_ref(),
                "controller-return-withdrawn",
            ))?
            .digest(),
            StopReason::OwnerStop,
        )?;
        return Err(refusal);
    }
    result
}
#[cfg(test)]
fn run_with(request: &ExistingRequest, authority: &Authority) -> Result<Value> {
    run_with_cancel(
        request,
        authority,
        &nf_surgeon::cancellation::CancellationToken::default(),
    )
}
fn run_with_cancel(
    request: &ExistingRequest,
    authority: &Authority,
    cancel: &nf_surgeon::cancellation::CancellationToken,
) -> Result<Value> {
    cancel.check()?;
    use crate::coordinator::budgets::{
        OutcomeKind, Request, Resources, Settlement, Usage, WorkKind,
    };
    use crate::coordinator::supervision::{FixtureMode, NativeJob, TrustedNativeWorker};
    // All grant and identity checks precede writable ledger/CAS opens and probe.
    let contract = frozen_contract(request)?;
    let (grant, doc) = authority.verify(request, &contract)?;
    if supplied::is_supplied_operation(&grant.operation) {
        let result = supplied::launch(request, &grant, &contract, cancel);
        return finish_authorized(request, authority, &contract, cancel, result);
    }
    authority_service::consume(request, authority)?;
    let worker = TrustedNativeWorker::qualify(&grant.worker, grant.worker_digest.clone())?;
    let mut ledger = request.open()?;
    authority.verify(request, &ledger.contract(&request.contract_digest)?)?;
    if ledger.stop_record(&request.contract_digest)?.is_some() {
        return Err(denied());
    }
    let lease = ledger.acquire(
        &request.contract_digest,
        "nf.manual.authorized_fixture.v1",
        30_000,
    )?;
    let result = (|| {
        let store = ArtifactStore::create(&request.store, store_limits())?;
        // CAS copy is an audit reference, NEVER the source of authority. The
        // reservation operation commits to this exact verified grant digest.
        let reservation = ledger.reserve(
            &lease,
            Request {
                schema_version: 1,
                operation: canonical(&("nf.manual.dispatch.v1", doc.digest()))?.digest(),
                attempt: 0,
                parent: None,
                kind: WorkKind::Metadata,
                resources: Resources {
                    cpu_ms: 4000,
                    host_bytes: 67_108_864,
                    storage_bytes: 1_048_576,
                    workers: 1,
                    ..Resources::default()
                },
                wall_ms: 3000,
                external: None,
            },
        )?;
        if let Err(refusal) = store
            .put(doc.bytes())
            .and_then(|_| authority.verify(request, &contract).map(|_| ()))
        {
            // No process was launched: release only this known prelaunch case.
            ledger.settle(
                &lease,
                &reservation,
                Settlement {
                    operation: canonical(&(doc.digest(), "revoked-before-launch"))?.digest(),
                    outcome: OutcomeKind::Cancelled,
                    usage: Usage {
                        cpu_ms: Some(0),
                        storage_bytes: None,
                        ..Usage::default()
                    },
                    resources_released: true,
                    evidence: doc.digest(),
                },
            )?;
            return Err(refusal);
        }
        let mut running = ledger.launch_native(
            &lease,
            &reservation,
            &worker,
            NativeJob {
                mode: FixtureMode::Complete,
                maximum_output_bytes: 1024,
                termination_grace_ms: 30,
            },
        )?;
        let worker_id = running.id().clone();
        loop {
            if cancel.check().is_err() || authority.verify(request, &contract).is_err() {
                ledger.request_stop(
                    &request.contract_digest,
                    canonical(&(doc.digest(), "grant-withdrawn"))?.digest(),
                    StopReason::OwnerStop,
                )?;
            }
            if let Some(report) = ledger.poll_native(&lease, &reservation, &mut running)? {
                return finish_authorized(
                    request,
                    authority,
                    &contract,
                    cancel,
                    Ok(json!({"schema_version":1,"authorization_verified":true,
                    "authorization":doc.digest(),"worker":worker_id,"report":report,
                    "execution":"native_complete_fixture","supplied_ir_executed":false,
                    "scientific_acceptance":"not_evaluated","promotion":false,
                    "strict_publication":false})),
                );
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    })();
    let released = ledger.release(&lease);
    let value = result?;
    released?;
    Ok(value)
}
