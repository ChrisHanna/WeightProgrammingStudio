//! Real syscalls, isolated seccomp controllers, and an unfiltered kill-safe watchdog.
//! No production fault switches; raw SQL below is test-only private admission setup.
use super::reload::{reload_graph, FrozenReloadProbes};
use super::*;
use crate::coordinator::budgets::{Request, Resources, WorkKind};
use crate::coordinator::{Lease, Ledger, LedgerLimits, Stage, Transition};
use nf_checkpoint::{
    export::{publish_export, ExportLimits},
    store::{ArtifactStore, StoreLimits},
};
use nf_core::{
    schemas::{ModelManifest, Validated},
    study::{ModelIdentity, StudyContract},
};
use nf_graph::inventory::tensor_set_digest;
use nf_surgeon::transaction::StateLoadLimits;
use std::sync::mpsc;
#[allow(dead_code)]
mod fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../nf-core/tests/common/study_fixture.rs"
    ));
}

fn signal(fd: &OwnedFd, signal: i32) {
    assert_eq!(
        unsafe {
            libc::syscall(
                libc::SYS_pidfd_send_signal,
                fd.as_raw_fd(),
                signal,
                std::ptr::null::<libc::siginfo_t>(),
                0,
            )
        },
        0
    );
}
fn live(fd: &OwnedFd) -> bool {
    let mut p = libc::pollfd {
        fd: fd.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    assert!(unsafe { libc::poll(&mut p, 1, 0) } >= 0);
    p.revents == 0
}
fn assert_reaped(fd: &OwnedFd) {
    let mut info = std::mem::MaybeUninit::<libc::siginfo_t>::zeroed();
    assert_eq!(
        unsafe {
            libc::waitid(
                libc::P_PIDFD,
                fd.as_raw_fd() as u32,
                info.as_mut_ptr(),
                libc::WEXITED | libc::WNOHANG,
            )
        },
        -1
    );
    assert_eq!(
        std::io::Error::last_os_error().raw_os_error(),
        Some(libc::ECHILD)
    );
}
fn watchdog_reap(fd: &OwnedFd) {
    if live(fd) {
        signal(fd, libc::SIGKILL);
    }
    let until = Instant::now() + Duration::from_secs(2);
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
            break;
        }
        if result < 0 && std::io::Error::last_os_error().raw_os_error() == Some(libc::ECHILD) {
            break;
        }
        assert!(Instant::now() < until, "watchdog could not reap child");
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(!live(fd));
    assert_reaped(fd);
}
fn deny_syscalls(calls: &[(libc::c_long, i32)]) {
    // Calling thread only, no TSYNC: watchdog remains unfiltered. Never execute
    // this in the outer test runner or install filters before qualifying Torch.
    let mut filter = vec![libc::sock_filter {
        code: 0x20,
        jt: 0,
        jf: 0,
        k: 0,
    }];
    for &(number, errno) in calls {
        filter.push(libc::sock_filter {
            code: 0x15,
            jt: 0,
            jf: 1,
            k: number as u32,
        });
        filter.push(libc::sock_filter {
            code: 0x06,
            jt: 0,
            jf: 0,
            k: libc::SECCOMP_RET_ERRNO | errno as u32,
        });
    }
    filter.push(libc::sock_filter {
        code: 0x06,
        jt: 0,
        jf: 0,
        k: libc::SECCOMP_RET_ALLOW,
    });
    let program = libc::sock_fprog {
        len: filter.len() as u16,
        filter: filter.as_mut_ptr(),
    };
    assert_eq!(
        unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) },
        0
    );
    assert_eq!(
        unsafe { libc::prctl(libc::PR_SET_SECCOMP, libc::SECCOMP_MODE_FILTER, &program) },
        0
    );
}
fn source() -> SourceState {
    let mut f: Value = serde_json::from_str(include_str!(
        "../../../../python/neural_foundry/data/m1_fixture.json"
    ))
    .unwrap();
    let w: Vec<u8> = [1.0_f64, 2.0, 3.0, 4.0]
        .into_iter()
        .flat_map(f64::to_le_bytes)
        .collect();
    let mut header =
        canonical(&json!({"w":{"dtype":"F64","shape":[2,2],"data_offsets":[0,32]}})).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut checkpoint = (header.len() as u64).to_le_bytes().to_vec();
    checkpoint.extend(header);
    checkpoint.extend(&w);
    for name in ["w", "tied"] {
        f["manifest"]["parameters"][name]["digest"] = json!(ContentDigest::of_bytes(&w));
    }
    let parameters = serde_json::from_value(f["manifest"]["parameters"].clone()).unwrap();
    f["manifest"]["parameter_digest"] = json!(tensor_set_digest(&parameters).unwrap());
    f["manifest"]["architecture"]["config"] = json!(ContentDigest::of_bytes(b"{}"));
    f["manifest"]["assets"] = json!({"checkpoint":ContentDigest::of_bytes(&checkpoint), "config":ContentDigest::of_bytes(b"{}")});
    SourceState::load(
        Validated::<ModelManifest>::parse(&canonical(&f["manifest"]).unwrap()).unwrap(),
        Arc::from(checkpoint),
        BTreeMap::from([("config".into(), Arc::from(b"{}".as_slice()))]),
        StateLoadLimits::default(),
    )
    .unwrap()
}
fn probes(s: &SourceState) -> FrozenReloadProbes {
    let model = ModelIdentity {
        checkpoint: ContentDigest::of_bytes(s.checkpoint_bytes()),
        manifest: s.manifest_digest().clone(),
        parameters: s.manifest().parameter_digest.clone(),
        buffers: tensor_set_digest(&s.manifest().buffers).unwrap(),
        graph: ContentDigest::of_bytes(&reload_graph(s).unwrap()),
        config: s.manifest().architecture.config.clone(),
        tokenizer: None,
        system_state: None,
        runtime_artifact: None,
    };
    FrozenReloadProbes::freeze(s, model, vec![vec![1.0, 2.0]], vec![17]).unwrap()
}
// Keep the spawning thread alive until watchdog inspection: Linux PDEATHSIG
// fires when that thread exits, not only when the whole controller exits.
struct ControllerBoundary(mpsc::Sender<()>, mpsc::Receiver<()>);
impl Drop for ControllerBoundary {
    fn drop(&mut self) {
        let _ = self.0.send(());
        let _ = self.1.recv_timeout(Duration::from_secs(6));
    }
}
struct Root(PathBuf);
impl Drop for Root {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn record_test(
    db: &rusqlite::Connection,
    lease: &Lease,
    kind: &str,
    value: &impl Serialize,
) -> ContentDigest {
    let bytes = canonical(value).unwrap();
    let digest = ContentDigest::of_bytes(&bytes);
    db.execute(
        "INSERT INTO promotion_records VALUES(?1,?2,?3,?4)",
        rusqlite::params![lease.contract().as_str(), kind, bytes, digest.as_str()],
    )
    .unwrap();
    digest
}
fn admitted(root: &Path) -> (Ledger, Lease, rusqlite::Connection) {
    // Test-only private admission, not public promotion or a claimed T038 run.
    let mut ledger = Ledger::open(&root.join("ledger"), LedgerLimits::default()).unwrap();
    let mut contract = fixture::standard();
    // The shared metadata fixture only allows 100 MB, below a real Torch
    // reservation. Freeze an explicit test-only cap before reserving; never
    // bypass admission or alter production budget enforcement.
    contract["limits"]["host_memory_bytes"] = json!(8_589_934_592_u64);
    let contract =
        Validated::<StudyContract>::parse(&serde_json::to_vec(&contract).unwrap()).unwrap();
    let id = ledger.freeze(&contract).unwrap();
    let lease = ledger.acquire(&id, "worker-fault-test", 300000).unwrap();
    let mut previous = Stage::Frozen;
    for (i, next) in [
        Stage::Preflight,
        Stage::Profile,
        Stage::Baseline,
        Stage::Design,
        Stage::Execute,
        Stage::Evaluate,
        Stage::Verify,
    ]
    .into_iter()
    .enumerate()
    {
        ledger
            .advance(
                &lease,
                Transition {
                    operation: fixture::d(&format!("fault-stage-{i}")),
                    expected: previous,
                    next,
                    protocol: fixture::d("test-only"),
                    evidence: fixture::d("test-only"),
                    message: "worker cleanup fixture only".into(),
                },
            )
            .unwrap();
        previous = next;
    }
    let reservation = ledger
        .reserve(
            &lease,
            Request {
                schema_version: 1,
                operation: fixture::d("worker-fault"),
                attempt: 0,
                parent: None,
                kind: WorkKind::Reload,
                resources: Resources {
                    cpu_ms: 30000,
                    host_bytes: 6442450944,
                    storage_bytes: 4194304,
                    workers: 1,
                    ..Default::default()
                },
                wall_ms: 30000,
                external: None,
            },
        )
        .unwrap();
    let db = rusqlite::Connection::open(root.join("ledger/events.sqlite3")).unwrap();
    let store: String = db
        .query_row("SELECT value FROM meta WHERE key='store'", [], |r| r.get(0))
        .unwrap();
    let boot = fs::read_to_string("/proc/sys/kernel/random/boot_id")
        .unwrap()
        .trim()
        .to_owned();
    let stat = fs::read_to_string("/proc/self/stat").unwrap();
    let start: u64 = stat
        .rsplit_once(')')
        .unwrap()
        .1
        .split_whitespace()
        .nth(19)
        .unwrap()
        .parse()
        .unwrap();
    let intent = record_test(
        &db,
        &lease,
        "intent",
        &json!({"contract":id, "store":store, "reservation":reservation.id(),
        "native_process":{"pid":std::process::id(), "boot":boot, "start_ticks":start}, "boot":boot,
        "epoch":lease.fencing_token(), "deadline":boot_ms()+30000}),
    );
    record_test(&db, &lease, "child_intent", &json!({"intent":intent}));
    (ledger, lease, db)
}

#[test]
fn isolated_worker_case() {
    let Ok(mode) = std::env::var("NF_WORKER_CLEANUP_CASE") else {
        return;
    };
    let root = Root(std::env::temp_dir().join(format!("worker-cleanup-{}", std::process::id())));
    fs::DirBuilder::new().mode(0o700).create(&root.0).unwrap();
    let project = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    // Cargo's library is hard-linked; qualification correctly rejects it.
    let extension = root.0.join("_native.so");
    fs::copy(project.join("target/debug/lib_native.so"), &extension).unwrap();
    let runtime = QualifiedTorch::qualify(
        true,
        Path::new("/usr/bin/python3.12"),
        &project.join(".venv/lib/python3.12/site-packages"),
        &extension,
        &root.0,
    )
    .unwrap();
    let store = ArtifactStore::create(
        &root.0.join("store"),
        StoreLimits {
            total_bytes: 8388608,
            max_blob_bytes: 1048576,
            max_entries: 128,
        },
    )
    .unwrap();
    let s = source();
    let probes = probes(&s);
    let export = publish_export(
        &store,
        s.manifest(),
        Arc::from(s.checkpoint_bytes()),
        s.assets().clone(),
        ExportLimits::default(),
    )
    .unwrap();
    let (mut ledger, lease, db) = admitted(&root.0);
    let (pin_tx, pin_rx) = mpsc::channel();
    let (ready_tx, ready_rx) = mpsc::channel();
    let (rescue_tx, rescue_rx) = mpsc::channel();
    let worker_mode = mode.clone();
    let fault_installed = Arc::new(AtomicBool::new(false));
    let callback_fault = fault_installed.clone();
    let controller = std::thread::spawn(move || {
        let _boundary = ControllerBoundary(ready_tx, rescue_rx);
        let mode = worker_mode.as_str();
        if mode == "pidfd_open" {
            deny_syscalls(&[(libc::SYS_pidfd_open, libc::EACCES)]);
            let a = runtime.verify_reload_registered(
                &store,
                &export,
                &probes,
                Limits::default(),
                &CancellationToken::default(),
                &mut |_| panic!("pidfd failure must not register/release work"),
            );
            assert!(a.witness().is_none());
            assert!(!a.report().cleanup_confirmed);
            // PID acquisition failed before process.json; inspect native children,
            // not worker reports. All must already be reaped by bounded fallback.
            let tid = unsafe { libc::syscall(libc::SYS_gettid) };
            assert!(
                fs::read_to_string(format!("/proc/self/task/{tid}/children"))
                    .unwrap()
                    .trim()
                    .is_empty()
            );
            return;
        }
        if mode.starts_with("drop_") {
            let child = Command::new("/usr/bin/sleep").arg("30").spawn().unwrap();
            let fd = pidfd(child.id()).unwrap();
            signal(&fd, libc::SIGSTOP);
            pin_tx.send(fd.try_clone().unwrap()).unwrap();
            let mut owned = OwnedChild {
                child,
                pidfd: Some(fd),
                reaped: false,
                cleanup_deadline: None,
            };
            let calls = if mode == "drop_denied" {
                vec![
                    (libc::SYS_pidfd_send_signal, libc::EPERM),
                    (libc::SYS_wait4, libc::EACCES),
                    (libc::SYS_waitid, libc::EACCES),
                ]
            } else {
                vec![(libc::SYS_wait4, libc::EACCES)]
            };
            deny_syscalls(&calls);
            assert!(owned.try_reap().is_err());
            assert!(!owned.reaped, "wait failure is not reaping");
            let start = Instant::now();
            drop(owned);
            assert!(start.elapsed() < Duration::from_millis(900));
            return;
        }
        let cancel = CancellationToken::default();
        if mode == "pre_cancel" {
            cancel.request();
        }
        let mut registered = false;
        let start = Instant::now();
        let mut fault_start = None;
        let limits = if matches!(mode, "eintr" | "signal_wall") {
            Limits {
                wall_ms: 100,
                ..Limits::default()
            }
        } else {
            Limits::default()
        };
        let attempt = runtime.verify_reload_registered(
            &store,
            &export,
            &probes,
            limits,
            &cancel,
            &mut |fd| {
                ledger.register_fixture_reload_child(&lease, fd)?;
                registered = true;
                fault_start = Some(Instant::now());
                pin_tx.send(fd.try_clone().unwrap()).unwrap();
                if mode != "success" {
                    signal(fd, libc::SIGSTOP);
                }
                let calls = match mode {
                    "wait4" => vec![(libc::SYS_wait4, libc::EACCES)],
                    "wait_both" => vec![
                        (libc::SYS_wait4, libc::EACCES),
                        (libc::SYS_waitid, libc::EACCES),
                    ],
                    "eintr" => vec![
                        (libc::SYS_wait4, libc::EINTR),
                        (libc::SYS_waitid, libc::EINTR),
                    ],
                    "signal_cancel" | "signal_refusal" | "signal_panic" | "signal_wall" => {
                        vec![(libc::SYS_pidfd_send_signal, libc::EPERM)]
                    }
                    _ => vec![],
                };
                deny_syscalls(&calls);
                callback_fault.store(true, Ordering::Release);
                if matches!(mode, "signal_panic" | "panic") {
                    panic!("test callback unwind");
                }
                if mode == "refusal" || mode == "signal_refusal" {
                    return Err(error(
                        ErrorCode::PermissionDenied,
                        "test registration refusal",
                    ));
                }
                if !matches!(
                    mode,
                    "success" | "wait4" | "wait_both" | "eintr" | "signal_wall"
                ) {
                    cancel.request();
                }
                Ok(())
            },
        );
        let bound = if mode == "success" { 20 } else { 3 };
        assert!(
            fault_start.unwrap_or(start).elapsed() < Duration::from_secs(bound),
            "cleanup exceeded controller bound"
        );
        if mode == "success" {
            assert!(attempt.witness().is_some(), "{:?}", attempt.report());
            assert!(attempt.report().cleanup_confirmed);
            assert!(attempt.report().cpu_ms.is_some());
            assert_eq!(
                attempt.report().child_report.as_ref().unwrap()["outputs"][0],
                json!([5.0, 11.0])
            );
        } else {
            assert!(attempt.witness().is_none(), "{:?}", attempt.report());
            let unknown = matches!(
                mode,
                "wait4"
                    | "wait_both"
                    | "eintr"
                    | "signal_cancel"
                    | "signal_refusal"
                    | "signal_wall"
            );
            if matches!(mode, "eintr" | "signal_wall") {
                assert!(
                    !cancel.is_cancelled(),
                    "deadline test must not use cancellation"
                );
                assert!(fault_start.unwrap().elapsed() >= Duration::from_millis(100));
            }
            assert_eq!(attempt.report().cleanup_confirmed, !unknown);
            assert_eq!(
                attempt.report().cpu_ms.is_none(),
                unknown || mode == "pre_cancel"
            );
            assert!(attempt.report().child_report.is_none());
            assert_eq!(registered, mode != "pre_cancel");
            if unknown {
                record_test(&db, &lease, "reload", attempt.report());
                let error = ledger.recover_fixture_promotion(&lease).unwrap_err();
                assert!(error.message.contains("owner still alive"), "{error}");
                if mode != "wait4" {
                    // Adversarial terminal marker passes the live-owner gate,
                    // but must NOT substitute for native child cleanup proof.
                    let intent: String = db
                        .query_row(
                            "SELECT digest FROM promotion_records WHERE kind='intent'",
                            [],
                            |r| r.get(0),
                        )
                        .unwrap();
                    record_test(
                        &db,
                        &lease,
                        "attempt_end",
                        &json!({"intent":intent,"cleanup_confirmed":true,
                        "child_entered":true,"status":"incomplete_no_rerun"}),
                    );
                    let error = ledger.recover_fixture_promotion(&lease).unwrap_err();
                    let expected = match mode {
                        "eintr" => "registered child reap deadline exceeded",
                        "wait_both" => "registered child reap unconfirmed",
                        _ => "registered child termination failed",
                    };
                    assert!(error.message.contains(expected), "{mode}: {error}");
                }
                assert_eq!(
                    ledger
                        .budget_snapshot(lease.contract())
                        .unwrap()
                        .held
                        .workers,
                    1
                );
                assert!(ledger
                    .accepted_fixture_head(&contract_family())
                    .unwrap()
                    .is_none());
            }
        }
    });
    // This thread existed before seccomp. It independently owns a pidfd and
    // rescues a stopped/unsignalable child even if the filtered controller hangs.
    let mut pin = None;
    let mut until = Instant::now() + Duration::from_secs(30);
    let finished = loop {
        if pin.is_none() {
            if let Ok(fd) = pin_rx.try_recv() {
                pin = Some(fd);
                until =
                    Instant::now() + Duration::from_secs(if mode == "success" { 20 } else { 4 });
            }
        }
        if ready_rx.try_recv().is_ok() {
            break true;
        }
        if Instant::now() >= until {
            break false;
        }
        std::thread::sleep(Duration::from_millis(2));
    };
    // Readiness and pidfd can arrive between the two receiver probes above.
    if pin.is_none() {
        pin = pin_rx.try_recv().ok();
    }
    let received_pin = pin.is_some();
    let inspections = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if let Some(fd) = &pin {
            if finished
                && matches!(
                    mode.as_str(),
                    "signal_cancel"
                        | "signal_refusal"
                        | "signal_panic"
                        | "signal_wall"
                        | "drop_denied"
                )
            {
                assert!(
                    live(&fd),
                    "denied signal must preserve live-child uncertainty"
                );
            }
            if finished
                && matches!(
                    mode.as_str(),
                    "wait4" | "drop_wait4" | "success" | "cancel" | "refusal" | "panic"
                )
            {
                assert_reaped(fd);
            }
        }
    }));
    // Rescue and acknowledge even when a diagnostic assertion fails.
    if let Some(fd) = &pin {
        watchdog_reap(fd);
    }
    let _ = rescue_tx.send(());
    let result = controller.join();
    assert!(finished, "watchdog rescued hung controller");
    if !matches!(mode.as_str(), "pre_cancel" | "pidfd_open") {
        assert!(received_pin, "missing child pidfd");
    }
    if matches!(mode.as_str(), "signal_panic" | "panic") {
        assert!(
            fault_installed.load(Ordering::Acquire),
            "callback fault was not installed"
        );
        let payload = result.expect_err("callback must unwind");
        let text = payload
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| payload.downcast_ref::<String>().map(String::as_str));
        assert_eq!(text, Some("test callback unwind"));
    } else {
        result.unwrap();
    }
    if let Err(payload) = inspections {
        std::panic::resume_unwind(payload);
    }
    // Includes both qualification and attempted reload children; no dangling
    // live child or zombie is allowed after watchdog cleanup.
    for task in fs::read_dir("/proc/self/task").unwrap().flatten() {
        let children = fs::read_to_string(task.path().join("children")).unwrap_or_default();
        assert!(children.trim().is_empty(), "dangling child: {children}");
    }
}
fn contract_family() -> ContentDigest {
    Validated::<StudyContract>::parse(&serde_json::to_vec(&fixture::standard()).unwrap())
        .unwrap()
        .study_family
        .clone()
}
fn run_case(mode: &str) {
    // Outer process is a subreaper so killing a broken controller cannot leave
    // adopted worker zombies. Pin its children continuously before any timeout.
    assert_eq!(
        unsafe { libc::prctl(libc::PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) },
        0
    );
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "model_worker::cleanup_tests::isolated_worker_case",
            "--nocapture",
        ])
        .env("NF_WORKER_CLEANUP_CASE", mode)
        .spawn()
        .unwrap();
    let until = Instant::now() + Duration::from_secs(45);
    let mut pins = BTreeMap::new();
    let status = loop {
        if let Ok(tasks) = fs::read_dir(format!("/proc/{}/task", child.id())) {
            for task in tasks.flatten() {
                for pid in fs::read_to_string(task.path().join("children"))
                    .unwrap_or_default()
                    .split_whitespace()
                {
                    let pid: u32 = pid.parse().unwrap();
                    if let std::collections::btree_map::Entry::Vacant(entry) = pins.entry(pid) {
                        if let Ok(fd) = pidfd(pid) {
                            entry.insert(fd);
                        }
                    }
                }
            }
        }
        if let Some(status) = child.try_wait().unwrap() {
            break Some(status);
        }
        if Instant::now() >= until {
            child.kill().unwrap();
            break None;
        }
        std::thread::sleep(Duration::from_millis(2));
    };
    if status.is_none() {
        let until = Instant::now() + Duration::from_secs(2);
        while child.try_wait().unwrap().is_none() {
            assert!(
                Instant::now() < until,
                "controller SIGKILL did not complete"
            );
            std::thread::sleep(Duration::from_millis(2));
        }
    }
    for fd in pins.values() {
        watchdog_reap(fd);
    }
    assert!(
        status.is_some_and(|s| s.success()),
        "isolated case {mode}: {status:?}"
    );
}
macro_rules! case {
    ($name:ident,$mode:literal) => {
        #[test]
        fn $name() {
            run_case($mode);
        }
    };
}
case!(real_wait4_error_still_terminates_and_reaps, "wait4");
case!(real_wait_errors_preserve_unknown_cleanup, "wait_both");
case!(real_eintr_is_deadline_bounded, "eintr");
case!(
    real_signal_denied_wall_expiry_retains_reservation,
    "signal_wall"
);
case!(
    real_signal_denied_cancellation_retains_reservation,
    "signal_cancel"
);
case!(
    real_signal_denied_refusal_retains_reservation,
    "signal_refusal"
);
case!(
    real_signal_denied_callback_unwind_is_bounded,
    "signal_panic"
);
case!(real_callback_unwind_reaps_before_watchdog, "panic");
case!(
    real_pidfd_open_failure_reaps_without_registration,
    "pidfd_open"
);
case!(real_destructor_wait4_error_uses_pidfd_reap, "drop_wait4");
case!(real_destructor_all_denied_returns_uncertain, "drop_denied");
case!(
    real_success_still_mints_only_private_reload_witness,
    "success"
);
case!(real_cancellation_still_reaps, "cancel");
case!(real_refusal_still_reaps, "refusal");
case!(pre_cancel_never_starts_reload, "pre_cancel");
