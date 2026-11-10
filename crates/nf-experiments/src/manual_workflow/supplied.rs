//! Explicit no-cycle v2 input binding and dedicated coordinator dispatch.
//! No user program is synthesized, patched, or replaced by a metadata fixture.
use super::*;
use authorization::Grant;
use nf_surgeon::cancellation::CancellationToken;
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::process::CommandExt;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

pub(super) const OPERATION: &str = "nf.manual.supplied_ir.linear_cpu.v2";
pub(super) const ARITHMETIC_OPERATION: &str = "nf.manual.supplied_ir.arithmetic_mixed_five.v3";
pub(super) fn is_supplied_operation(op: &str) -> bool {
    matches!(op, OPERATION | ARITHMETIC_OPERATION)
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Binding {
    pub(super) revision: u32,
    pub(super) manifest: ContentDigest,
    pub(super) source_export: ContentDigest,
    pub(super) source_identity: ContentDigest,
}
/// Exact manifest schema: metadata cannot smuggle an authority/path/callback.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: u32,
    protocol: String,
    source_export: ContentDigest,
    program: ContentDigest,
    search_space: ContentDigest,
    policy: ContentDigest,
    smoke_suite: ContentDigest,
    compiled_plan: ContentDigest,
    input_bundle: ContentDigest,
    frozen_contract: ContentDigest,
    method_protocol: ContentDigest,
    control_program: ContentDigest,
    compiled_control: ContentDigest,
}
fn mismatch() -> CoreError {
    error(
        ErrorCode::SourceMismatch,
        "approved supplied-IR revision/source/frozen input binding differs",
    )
}
/// Static verification is also used by the privileged issuer. Never compile,
/// qualify a runtime, execute probes, or follow a caller-selected code callback.
pub(super) fn verify_binding(
    request: &ExistingRequest,
    contract: &Validated<StudyContract>,
    binding: &Binding,
) -> Result<()> {
    if binding.revision != 2 || binding.source_identity != canonical(&contract.source)?.digest() {
        return Err(mismatch());
    }
    let store = ArtifactStore::open_read_only(&request.store, store_limits())?;
    let manifest = CanonicalDocument::parse(
        &store.read_bounded(&binding.manifest, 65536)?,
        Limits::default(),
    )?;
    let m: Manifest = decode(manifest.bytes())?;
    if manifest.digest() != binding.manifest
        || m.schema_version != 2
        || m.protocol != "nf.manual.supplied_ir.v2"
        || m.frozen_contract != request.contract_digest
        || m.source_export != binding.source_export
        || m.method_protocol != contract.intervention.method_protocol
    {
        return Err(mismatch());
    }
    // Compiled digests are syntax-checked here; the unprivileged retained-plan
    // path independently recompiles and checks the WHOLE manifest before work.
    let _ = (&m.compiled_plan, &m.compiled_control);
    let bundle_id = m.input_bundle;
    let bundle = CanonicalDocument::parse(
        &store.read_bounded(&bundle_id, 1_048_576)?,
        Limits::default(),
    )?;
    let plan: PlanRequest = decode(bundle.bytes())?;
    if bundle.digest() != bundle_id
        || plan.schema_version != 2
        || plan.store != request.store
        || canonical(&plan.contract)?.digest() != request.contract_digest
        || plan.source_export != binding.source_export
        || m.program != plan.program.digest()?
        || m.search_space != plan.search_space.digest()?
        || m.policy != canonical(&plan.policy)?.digest()
        || m.smoke_suite != plan.smoke_suite.digest()?
        || m.control_program
            != plan
                .control_program
                .as_ref()
                .ok_or_else(mismatch)?
                .digest()?
        || plan.program.objective.experiment_protocol != contract.intervention.method_protocol
    {
        return Err(mismatch());
    }
    Ok(())
}
pub(super) fn retained(request: &ExistingRequest, binding: &Binding) -> Result<Planned> {
    let store = ArtifactStore::open_read_only(&request.store, store_limits())?;
    let manifest: Value = decode(&store.read_bounded(&binding.manifest, 65536)?)?;
    let id: ContentDigest =
        serde_json::from_value(manifest["input_bundle"].clone()).map_err(|_| mismatch())?;
    let planned = prepare(decode(&store.read_bounded(&id, 1_048_576)?)?)?;
    if planned.manifest.digest() != binding.manifest
        || planned.result["contract_binding_matches"] != true
    {
        return Err(mismatch());
    }
    Ok(planned)
}
/// Read and seal the approved executable before exec, avoiding path/digest races.
/// The privileged operator never executes these bytes; the researcher does.
fn sealed_executable(grant: &Grant) -> Result<std::fs::File> {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(&grant.worker)
        .map_err(|e| error(ErrorCode::Io, &e.to_string()))?;
    if !file.metadata().map_err(|_| mismatch())?.is_file() {
        return Err(mismatch());
    }
    let mut bytes = Vec::new();
    file.take(67_108_865)
        .read_to_end(&mut bytes)
        .map_err(|_| mismatch())?;
    if bytes.len() > 67_108_864 || ContentDigest::of_bytes(&bytes) != grant.worker_digest {
        return Err(mismatch());
    }
    sealed_bytes(&bytes, c"nf-manual-approved")
}
fn sealed_bytes(bytes: &[u8], name: &std::ffi::CStr) -> Result<std::fs::File> {
    let fd =
        unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING) };
    if fd < 0 {
        return Err(error(
            ErrorCode::UnsupportedCapability,
            "sealed executable unavailable",
        ));
    }
    let mut image = unsafe { std::fs::File::from_raw_fd(fd) };
    image
        .write_all(bytes)
        .map_err(|e| error(ErrorCode::Io, &e.to_string()))?;
    image
        .seek(SeekFrom::Start(0))
        .map_err(|e| error(ErrorCode::Io, &e.to_string()))?;
    if unsafe {
        libc::fcntl(
            fd,
            libc::F_ADD_SEALS,
            libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL,
        )
    } < 0
    {
        return Err(error(
            ErrorCode::UnsupportedCapability,
            "executable sealing failed",
        ));
    }
    Ok(image)
}
pub(super) fn launch(
    request: &ExistingRequest,
    grant: &Grant,
    contract: &Validated<StudyContract>,
    cancel: &CancellationToken,
) -> Result<Value> {
    launch_with(
        request,
        grant,
        contract,
        cancel,
        &["study", "execute-supplied", "-"],
        &[],
        |bytes| decode(bytes),
    )
}
fn launch_with(
    request: &ExistingRequest,
    grant: &Grant,
    contract: &Validated<StudyContract>,
    cancel: &CancellationToken,
    arguments: &[&str],
    child_environment: &[(&str, &str)],
    response: impl FnOnce(&[u8]) -> Result<Value>,
) -> Result<Value> {
    cancel.check()?;
    let image = sealed_executable(grant)?;
    // Finite sealed input avoids a blocking pipe write before the wall deadline
    // is active, even if an approved executable never reads its request.
    let input = sealed_bytes(canonical(request)?.bytes(), c"nf-manual-request")?;
    let is_arith = grant.operation == ARITHMETIC_OPERATION
        || contract.runtime.backend.id == "nf.final.arithmetic_mixed_five";
    let memory = if is_arith {
        (512_u64 << 20).min(contract.limits.host_memory_bytes)
    } else {
        contract.limits.host_memory_bytes
    };
    let max_cpu = if is_arith { 60 } else { 35 };
    let cpu = contract.limits.cpu_ms.div_ceil(1000).min(max_cpu).max(1);
    let storage = contract.limits.storage_bytes;
    let parent = unsafe { libc::getpid() };
    let mut command = Command::new(format!("/proc/self/fd/{}", image.as_raw_fd()));
    command
        .args(arguments)
        .env_clear()
        .envs(child_environment.iter().copied())
        .current_dir("/")
        .stdin(Stdio::from(input))
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .process_group(0);
    // Only async-signal-safe syscalls between fork and exec. Never narrow the
    // Python/shared controller's resource limits or trust an isolation flag.
    unsafe {
        command.pre_exec(move || {
            for (resource, maximum) in [
                (libc::RLIMIT_AS, memory),
                (libc::RLIMIT_CPU, cpu),
                (libc::RLIMIT_FSIZE, storage),
                (libc::RLIMIT_CORE, 0),
                (libc::RLIMIT_NOFILE, 64),
            ] {
                let limit = libc::rlimit {
                    rlim_cur: maximum,
                    rlim_max: maximum,
                };
                if libc::setrlimit(resource, &limit) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
            }
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::getppid() != parent {
                return Err(std::io::Error::from_raw_os_error(libc::ECHILD));
            }
            Ok(())
        });
    }
    let mut child = command
        .spawn()
        .map_err(|e| error(ErrorCode::Io, &e.to_string()))?;
    let result = (|| {
        let mut output = child.stdout.take().ok_or_else(mismatch)?;
        if unsafe { libc::fcntl(output.as_raw_fd(), libc::F_SETFL, libc::O_NONBLOCK) } < 0 {
            return Err(error(
                ErrorCode::UnsupportedCapability,
                "nonblocking bounded output unavailable",
            ));
        }
        let max_wall = if is_arith { 80_000 } else { 30_000 };
        let deadline =
            Instant::now() + Duration::from_millis(contract.limits.wall_ms.min(max_wall) + 5000);
        let mut bytes = Vec::new();
        let mut cancel_recorded = false;
        loop {
            if !cancel_recorded && cancel.check().is_err() {
                request.open()?.request_stop(
                    &request.contract_digest,
                    canonical(&(request.authorization.as_ref(), "controller-cancel"))?.digest(),
                    StopReason::OwnerStop,
                )?;
                cancel_recorded = true;
            }
            let mut buffer = [0u8; 8192];
            match output.read(&mut buffer) {
                Ok(n) => bytes.extend_from_slice(&buffer[..n]),
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => (),
                Err(e) => return Err(error(ErrorCode::Io, &e.to_string())),
            }
            if bytes.len() > 1_048_576 {
                return Err(error(
                    ErrorCode::LimitExceeded,
                    "supplied output bound exceeded",
                ));
            }
            if let Some(status) = child
                .try_wait()
                .map_err(|e| error(ErrorCode::Io, &e.to_string()))?
            {
                // Drain the bounded tail after exit; all descendants use their
                // own coordinator channels, not this stdout pipe.
                output
                    .take((1_048_577 - bytes.len()) as u64)
                    .read_to_end(&mut bytes)
                    .map_err(|_| mismatch())?;
                if !status.success() {
                    return Err(error(ErrorCode::UnsupportedCapability, "isolated supplied dispatcher failed; inspect retained ledger; never retry uncertain work"));
                }
                let value = response(&bytes)?;
                // Cancellation can arrive while draining/decoding a successful
                // child response, after the last poll. Persist it before returning
                // and never report a cancelled invocation as successful.
                if let Err(cancelled) = cancel.check() {
                    if !cancel_recorded {
                        request.open()?.request_stop(
                            &request.contract_digest,
                            canonical(&(request.authorization.as_ref(), "controller-cancel"))?
                                .digest(),
                            StopReason::OwnerStop,
                        )?;
                    }
                    return Err(cancelled);
                }
                if value["ok"] != true {
                    return Err(mismatch());
                }
                return Ok(value["result"].clone());
            }
            if Instant::now() >= deadline {
                return Err(error(ErrorCode::LimitExceeded, "isolated supplied dispatcher timed out; reconcile retained intent before further work"));
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    })();
    if result.is_err() {
        let _ = child.kill();
    }
    let _ = child.wait();
    result
}
#[cfg(test)]
pub(super) fn launch_test(
    request: &ExistingRequest,
    grant: &Grant,
    contract: &Validated<StudyContract>,
) -> Result<Value> {
    launch_test_with_cancel(
        request,
        grant,
        contract,
        &CancellationToken::default(),
        false,
    )
}
#[cfg(test)]
pub(super) fn launch_test_with_cancel(
    request: &ExistingRequest,
    grant: &Grant,
    contract: &Validated<StudyContract>,
    cancel: &CancellationToken,
    cancel_on_response: bool,
) -> Result<Value> {
    // Only the test build adapts libtest framing. The production child accepts
    // exact bounded JSON, and has no selectable authority path or test argument.
    launch_with(
        request,
        grant,
        contract,
        cancel,
        &[
            "--exact",
            "manual_workflow::authorization::tests::supplied_tests::supplied_ir_child_entry",
            "--nocapture",
            "--test-threads=1",
        ],
        &[("NF_TEST_SUPPLIED_IR_CHILD", "1")],
        |bytes| {
            let text = std::str::from_utf8(bytes).map_err(|_| mismatch())?;
            let line = text
                .lines()
                .find_map(|line| {
                    line.split_once("NF_SUPPLIED_RESULT=")
                        .map(|(_, value)| value)
                })
                .ok_or_else(mismatch)?;
            let value = decode(line.as_bytes())?;
            if cancel_on_response {
                cancel.request();
            }
            Ok(value)
        },
    )
}
/// Called only after the independent OS grant and executable/limit checks. The
/// actual coordinator owns reservations, process identities, controls, outputs,
/// paired measurements, export/reload and settlement. No receipt is fabricated.
pub(super) fn execute(
    request: &ExistingRequest,
    grant: &Grant,
    cancel: &CancellationToken,
) -> Result<Value> {
    cancel.check()?;
    let planned = retained(request, grant.supplied.as_ref().ok_or_else(mismatch)?)?;
    let is_arith =
        grant.operation == ARITHMETIC_OPERATION || planned.smoke.adapter == "nf.arithmetic_relu.v1";
    let mut ledger = if is_arith {
        request.snapshot()?;
        Ledger::open(
            &request.ledger,
            LedgerLimits {
                max_database_bytes: 16 << 20,
                ..Default::default()
            },
        )?
    } else {
        request.open()?
    };
    let op_name = grant.operation.as_str();
    let lease_ms = if is_arith { 120_000 } else { 60_000 };
    let lease = ledger.acquire(&request.contract_digest, op_name, lease_ms)?;
    let result = (|| {
        if ledger.stop_record(&request.contract_digest)?.is_some() {
            return Err(error(
                ErrorCode::TransactionAborted,
                "supplied execution stopped",
            ));
        }
        // These are software admission phases, not invented scientific evidence.
        // The coordinator dispatcher performs the real controls and evaluations.
        let state = ledger.state(&request.contract_digest)?;
        if state.stage != crate::coordinator::Stage::Frozen {
            return Err(error(
                ErrorCode::RevisionConflict,
                "supplied dispatch is one-shot from Frozen; resume only reconciles",
            ));
        }
        let store = ArtifactStore::create(&request.store, store_limits())?;
        for binding in &planned.contract.portfolio.bindings {
            ledger.bind_split_catalog(&lease, &store, &binding.id)?;
        }
        use crate::coordinator::{Stage, Transition};
        let mut previous = Stage::Frozen;
        for next in [
            Stage::Preflight,
            Stage::Profile,
            Stage::Baseline,
            Stage::Design,
            Stage::Execute,
            Stage::Evaluate,
            Stage::Verify,
        ] {
            cancel.check()?;
            ledger.advance(&lease, Transition {
                operation:canonical(&(op_name, &request.contract_digest, format!("{next:?}")))?.digest(),
                expected:previous, next, protocol:planned.contract.intervention.method_protocol.clone(),
                evidence:planned.manifest.digest(), message:format!("supplied-IR admission; grant {}; actual science delegated to final dispatcher", request.authorization.as_ref().ok_or_else(mismatch)?),
            })?;
            previous = next;
        }
        if is_arith {
            use std::os::unix::fs::DirBuilderExt;
            let b0 = planned
                .contract
                .portfolio
                .bindings
                .first()
                .ok_or_else(mismatch)?;
            let g0 = planned
                .contract
                .portfolio
                .gyms
                .get(&b0.gym)
                .ok_or_else(mismatch)?;
            let sandbox_root = request
                .ledger
                .parent()
                .ok_or_else(mismatch)?
                .join("supplied-sandbox");
            if !sandbox_root.exists() {
                std::fs::DirBuilder::new()
                    .mode(0o700)
                    .create(&sandbox_root)
                    .map_err(|e| error(ErrorCode::Io, &e.to_string()))?;
            }
            let approval_id = g0.approval.as_ref().ok_or_else(mismatch)?;
            let run = ledger.run_arithmetic_final_from_store(
                &lease,
                &store,
                &planned.edit,
                planned.control.as_ref().ok_or_else(mismatch)?,
                &planned.smoke,
                &g0.code,
                approval_id,
                &sandbox_root,
                cancel,
            )?;
            let decision = run.decision();
            let executed = decision.report().checks.iter().any(|check| {
                check.name == "final_execution"
                    && check.status == crate::coordinator::acceptance::Status::Passed
            });
            return Ok(
                json!({"schema_version":2,"authorization_verified":true,"authorization":request.authorization,
                "execution":op_name,"supplied_ir_executed":executed,"manifest":planned.manifest.digest(),
                "program":planned.manifest.value()["program"],"source_export":planned.manifest.value()["source_export"],
                "compiled_plan":planned.edit.digest(),"export_root":run.export_root(),
                "decision":serde_json::from_slice::<Value>(decision.bytes()).map_err(|_| mismatch())?,
                "budgets":ledger.budget_snapshot(&request.contract_digest)?,
                "scientific_scope":"supported_arithmetic_mixed_five_fixture_only","strict_publication":false,"promotion":false}),
            );
        }
        let decision = ledger.run_fixture_final(
            &lease,
            &store,
            &planned.edit,
            planned.control.as_ref().ok_or_else(mismatch)?,
            &planned.smoke,
            cancel,
        )?;
        let executed = decision.report().checks.iter().any(|check| {
            check.name == "final_execution"
                && check.status == crate::coordinator::acceptance::Status::Passed
        });
        Ok(
            json!({"schema_version":2,"authorization_verified":true,"authorization":request.authorization,
            "execution":OPERATION,"supplied_ir_executed":executed,"manifest":planned.manifest.digest(),
            "program":planned.manifest.value()["program"],"source_export":planned.manifest.value()["source_export"],
            "compiled_plan":planned.edit.digest(),"decision":serde_json::from_slice::<Value>(decision.bytes()).map_err(|_| mismatch())?,
            "budgets":ledger.budget_snapshot(&request.contract_digest)?,
            "scientific_scope":"supported_linear_cpu_fixture_only","strict_publication":false,"promotion":false}),
        )
    })();
    let release = ledger.release(&lease);
    let value = result?;
    release?;
    Ok(value)
}
