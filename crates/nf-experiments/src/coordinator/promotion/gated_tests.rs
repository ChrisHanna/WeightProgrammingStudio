//! Actual reload-child launch tests; no synthetic process stands in for work.
use super::e2e::Fixture;
use super::*;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

fn probes(f: &Fixture) -> (FrozenReloadProbes, ContentDigest) {
    let c = f.ledger.contract(f.lease.contract()).unwrap();
    let original = source(&f.store, &c.source).unwrap();
    let candidate = execute_cancellable(
        &f.edit,
        &original,
        &f.smoke,
        execution_limits(),
        &CancellationToken::default(),
    )
    .unwrap();
    let reference = candidate
        .materialize_source(export_limits().materialization)
        .unwrap();
    let inputs = f
        .smoke
        .probes
        .iter()
        .map(|p| p.input.clone())
        .collect::<Vec<_>>();
    let frozen = FrozenReloadProbes::freeze(
        &reference,
        model(&reference).unwrap(),
        inputs.clone(),
        vec![c.seeds[0]; inputs.len()],
    )
    .unwrap();
    (
        frozen,
        export_candidate(&f.store, &candidate, &f.smoke, export_limits())
            .unwrap()
            .export_root,
    )
}
pub(super) fn admit(f: &mut Fixture) {
    let store = f.ledger.store.clone();
    let owner = process::Owner::current().unwrap();
    let lease = &f.lease;
    f.ledger.reserve_with(lease, Request {schema_version:1, operation:fixture::d("actual-gated-reload"), attempt:0,
        parent:None, kind:WorkKind::Reload, resources:Resources {cpu_ms:30000, host_bytes:6442450944,
            storage_bytes:4194304, workers:1, ..Default::default()}, wall_ms:30000, external:None}, |tx,r,deadline| {
        let id = record(tx, lease.contract(), "intent", &json!({"contract":lease.contract(), "store":store,
            "native_process":owner, "boot":files::moment()?.boot, "reservation":r.id(), "epoch":lease.epoch, "deadline":deadline}))?;
        record(tx, lease.contract(), "child_intent", &json!({"intent":id}))?;
        Ok(())
    }).unwrap();
}
fn native_pid(fd: &OwnedFd) -> u32 {
    std::fs::read_to_string(format!("/proc/self/fdinfo/{}", fd.as_raw_fd()))
        .unwrap()
        .lines()
        .find_map(|s| s.strip_prefix("Pid:").and_then(|v| v.trim().parse().ok()))
        .unwrap()
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
#[test]
fn actual_reload_gate_failure_is_measured_reaped_and_never_runs_model() {
    if e2e::dedicated_test() {
        return;
    }
    let mut f = Fixture::new(true);
    let runtime = f.runtime();
    let (probes, root) = probes(&f);
    admit(&mut f);
    let mut pinned = None;
    let attempt = runtime.verify_reload_registered(
        &f.store,
        &root,
        &probes,
        WorkerLimits::default(),
        &CancellationToken::default(),
        &mut |fd| {
            let pid = native_pid(fd);
            let until = std::time::Instant::now() + Duration::from_secs(3);
            loop {
                // Kernel-observed blocking read, not a JSON claim about a gate.
                let waiting = std::fs::read_to_string(format!("/proc/{pid}/wchan")).unwrap();
                if waiting.trim() == "pipe_read" {
                    break;
                }
                assert!(
                    std::time::Instant::now() < until,
                    "actual child did not reach stdin gate: {waiting}"
                );
                std::thread::sleep(Duration::from_millis(2));
            }
            let maps = std::fs::read_to_string(format!("/proc/{pid}/maps")).unwrap();
            assert!(
                !maps.contains("libtorch"),
                "dependencies loaded before registration"
            );
            f.ledger.register_fixture_reload_child(&f.lease, fd)?;
            assert!(!supervision::all_cleanup_confirmed(
                &f.ledger.connection,
                f.lease.contract()
            )?);
            pinned = Some(fd.try_clone().unwrap());
            Err(deny("test native registration callback failure"))
        },
    );
    assert!(attempt.witness().is_none());
    let report = attempt.report();
    assert!(report
        .failure
        .as_ref()
        .unwrap()
        .contains("test native registration callback failure"));
    assert_eq!(report.child_started, Some(true));
    assert!(report.cleanup_confirmed && report.cpu_ms.is_some() && report.peak_rss_bytes.is_some());
    assert!(report.child_report.is_none());
    let fd = pinned.unwrap();
    let mut p = libc::pollfd {
        fd: fd.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    assert_eq!(unsafe { libc::poll(&mut p, 1, 0) }, 1);
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
        Some(libc::ECHILD),
        "launch must already have reaped"
    );
    let intent = read(&f.ledger.connection, f.lease.contract(), "intent")
        .unwrap()
        .unwrap()
        .0;
    let tx = f.ledger.connection.transaction().unwrap();
    record(&tx, f.lease.contract(), "reload", report).unwrap();
    record(
        &tx,
        f.lease.contract(),
        "supervision",
        &json!({"intent":intent,"cleanup_confirmed":true}),
    )
    .unwrap();
    record(&tx, f.lease.contract(), "attempt_end", &json!({"intent":intent,"cleanup_confirmed":true,"status":"incomplete_no_rerun","child_entered":true})).unwrap();
    tx.commit().unwrap();
    f.ledger.recover_fixture_promotion(&f.lease).unwrap();
    assert_eq!(
        f.ledger
            .budget_snapshot(f.lease.contract())
            .unwrap()
            .held
            .workers,
        0
    );
    assert!(supervision::all_cleanup_confirmed(&f.ledger.connection, f.lease.contract()).unwrap());
    f.unchanged();
}
#[test]
fn private_registration_commit_failure_kills_reaps_and_retains_costs() {
    if e2e::dedicated_test() {
        return;
    }
    let mut f = Fixture::new(true);
    let runtime = f.runtime();
    f.ledger.connection.execute_batch("CREATE TRIGGER reject_child_registration BEFORE INSERT ON promotion_records WHEN NEW.kind='child_registered' BEGIN SELECT RAISE(ABORT,'test registration commit failure'); END;").unwrap();
    let result = f.ledger.promote_fixture_after_review(
        &f.lease,
        &f.store,
        &f.edit,
        &f.smoke,
        &runtime,
        fixture::d("registration-rollback"),
        None,
        &CancellationToken::default(),
    );
    assert!(result.is_err());
    assert!(
        read(&f.ledger.connection, f.lease.contract(), "child_registered")
            .unwrap()
            .is_none()
    );
    let report = read(&f.ledger.connection, f.lease.contract(), "reload")
        .unwrap()
        .unwrap()
        .1;
    assert_eq!(report["cleanup_confirmed"], true);
    assert_eq!(report["child_started"], true);
    assert!(report["cpu_ms"].as_u64().is_some() && report["peak_rss_bytes"].as_u64().is_some());
    assert!(report["failure"]
        .as_str()
        .unwrap()
        .contains("test registration commit failure"));
    assert!(report["child_report"].is_null());
    assert!(supervision::all_cleanup_confirmed(&f.ledger.connection, f.lease.contract()).unwrap());
    assert_eq!(
        f.ledger
            .budget_snapshot(f.lease.contract())
            .unwrap()
            .held
            .workers,
        1
    );
    f.ledger.recover_fixture_promotion(&f.lease).unwrap();
    assert_eq!(
        f.ledger
            .budget_snapshot(f.lease.contract())
            .unwrap()
            .held
            .workers,
        0
    );
    f.unchanged();
}
struct Controller(std::process::Child);
impl Drop for Controller {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
#[test]
fn actual_registered_reload_controller_crash_before_and_after_gate_has_no_ghost() {
    if let Some(mode) = std::env::var_os("NF_ACTUAL_RELOAD_CRASH") {
        let mut f = Fixture::new(true);
        let runtime = f.runtime();
        let (probes, root) = probes(&f);
        admit(&mut f);
        // Release only the lease; the original process still owns this attempt.
        // Registration must run first, so release in the stop boundary below.
        let before = mode == "before_gate";
        let _attempt = runtime.verify_reload_registered(
            &f.store,
            &root,
            &probes,
            WorkerLimits::default(),
            &CancellationToken::default(),
            &mut |fd| {
                f.ledger.register_fixture_reload_child(&f.lease, fd)?;
                if before {
                    f.ledger.release(&f.lease)?;
                    unsafe {
                        libc::raise(libc::SIGSTOP);
                    }
                } else {
                    // A native mapped libtorch proves gate release and actual
                    // dependency work, independently of any child JSON report.
                    let fd = fd.try_clone().unwrap();
                    let pid = native_pid(&fd);
                    let path = f.root.join("ledger");
                    let lease = f.lease.clone();
                    std::thread::spawn(move || {
                        let until = std::time::Instant::now() + Duration::from_secs(8);
                        loop {
                            let maps = std::fs::read_to_string(format!("/proc/{pid}/maps"))
                                .unwrap_or_default();
                            if maps.contains("libtorch") {
                                break;
                            }
                            assert!(
                                std::time::Instant::now() < until,
                                "child never imported actual Torch"
                            );
                            std::thread::sleep(Duration::from_millis(1));
                        }
                        signal(&fd, libc::SIGSTOP);
                        Ledger::open(&path, LedgerLimits::default())
                            .unwrap()
                            .release(&lease)
                            .unwrap();
                        unsafe {
                            libc::raise(libc::SIGSTOP);
                        }
                    });
                }
                Ok(())
            },
        );
        panic!("controller failed to reach live crash boundary");
    }
    if e2e::dedicated_test() {
        return;
    }
    assert_eq!(unsafe { libc::prctl(libc::PR_SET_CHILD_SUBREAPER, 1) }, 0);
    for mode in ["before_gate", "after_gate"] {
        let mut controller = Controller(std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "coordinator::promotion::tests::gated::actual_registered_reload_controller_crash_before_and_after_gate_has_no_ghost", "--nocapture"])
            .env("NF_ACTUAL_RELOAD_CRASH", mode).spawn().unwrap());
        let until = std::time::Instant::now() + Duration::from_secs(40);
        let mut status = 0;
        loop {
            let got = unsafe {
                libc::waitpid(
                    controller.0.id() as i32,
                    &mut status,
                    libc::WUNTRACED | libc::WNOHANG,
                )
            };
            if got == controller.0.id() as i32 {
                break;
            }
            assert_eq!(got, 0);
            assert!(
                std::time::Instant::now() < until,
                "bounded controller crash wait expired"
            );
            std::thread::sleep(Duration::from_millis(2));
        }
        assert!(libc::WIFSTOPPED(status));
        let root = std::env::temp_dir().join(format!("atomic-promotion-{}-0", controller.0.id()));
        let mut ledger = Ledger::open(&root.join("ledger"), LedgerLimits::default()).unwrap();
        let (contract, body): (String, Vec<u8>) = ledger
            .connection
            .query_row(
                "SELECT contract,body FROM promotion_records WHERE kind='child_registered'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        let contract: ContentDigest = contract.parse().unwrap();
        let registration: Value = serde_json::from_slice(&body).unwrap();
        let pid = registration["process"]["pid"].as_u64().unwrap() as u32;
        let raw = unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0) };
        assert!(raw >= 0);
        let fd = unsafe { OwnedFd::from_raw_fd(raw as i32) };
        assert_eq!(native_pid(&fd), pid);
        // No signal is sent using this stored PID. Recovery checks boot/start
        // against the pinned kernel process before it may send a signal.
        assert!(!supervision::all_cleanup_confirmed(&ledger.connection, &contract).unwrap());
        controller.0.kill().unwrap();
        controller.0.wait().unwrap();
        let lease = ledger
            .acquire(&contract, "actual-child-recovery", 300000)
            .unwrap();
        ledger.recover_fixture_promotion(&lease).unwrap();
        assert!(supervision::all_cleanup_confirmed(&ledger.connection, &contract).unwrap());
        let budget = ledger.budget_snapshot(&contract).unwrap();
        assert_eq!(budget.held.workers, 0);
        assert!(
            budget.unknown_charged.cpu_ms >= 30000,
            "crash CPU is unknown, not measured zero"
        );
        assert!(read(&ledger.connection, &contract, "reload")
            .unwrap()
            .is_none());
        assert_eq!(
            read(&ledger.connection, &contract, "child_recovery")
                .unwrap()
                .unwrap()
                .1["reaped_by_recovery"],
            true
        );
        let mut p = libc::pollfd {
            fd: fd.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        assert_eq!(unsafe { libc::poll(&mut p, 1, 0) }, 1);
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
        let family = ledger.contract(&contract).unwrap().study_family.clone();
        assert!(ledger.accepted_fixture_head(&family).unwrap().is_none());
        ledger.recover_fixture_promotion(&lease).unwrap();
        drop(ledger);
        std::fs::remove_dir_all(root).unwrap();
    }
}
