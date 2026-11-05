//! Real kernel-child tests of registration/recovery, NOT reload-hook integration.
use super::*;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::process::CommandExt;
use std::process::{Child, Command, Stdio};

struct OwnedProcess(Child);
impl Drop for OwnedProcess {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
struct Pinned(OwnedFd);
impl Drop for Pinned {
    fn drop(&mut self) {
        unsafe {
            libc::syscall(
                libc::SYS_pidfd_send_signal,
                self.0.as_raw_fd(),
                libc::SIGKILL,
                std::ptr::null::<libc::siginfo_t>(),
                0,
            );
            let mut info = std::mem::MaybeUninit::<libc::siginfo_t>::zeroed();
            // WNOHANG avoids an unbounded destructor if a kernel exit stalls.
            libc::waitid(
                libc::P_PIDFD,
                self.0.as_raw_fd() as u32,
                info.as_mut_ptr(),
                libc::WEXITED | libc::WNOHANG,
            );
        }
    }
}
fn pin(pid: u32) -> Pinned {
    let fd = unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0) };
    assert!(fd >= 0);
    Pinned(unsafe { OwnedFd::from_raw_fd(fd as i32) })
}
fn admitted(path: &Path) -> (Ledger, Lease) {
    let mut ledger = Ledger::open(path, LedgerLimits::default()).unwrap();
    let c = Validated::<StudyContract>::parse(&serde_json::to_vec(&fixture::standard()).unwrap())
        .unwrap();
    let id = ledger.freeze(&c).unwrap();
    let lease = ledger
        .acquire(&id, "registration-controller", 300000)
        .unwrap();
    let store = ledger.store.clone();
    let owner = process::Owner::current().unwrap();
    ledger.reserve_with(&lease,Request {schema_version:1,operation:fixture::d("registration-cost"),attempt:0,parent:None,kind:WorkKind::Reload,
        resources:Resources {cpu_ms:100,storage_bytes:65536,host_bytes:1024,workers:1,..Default::default()},wall_ms:10000,external:None},|tx,reservation,deadline| {
        let intent=record(tx,&id,"intent",&json!({"contract":id,"store":store,"reservation":reservation.id(),
            "native_process":owner,"boot":files::moment()?.boot,"epoch":lease.epoch,"deadline":deadline}))?;
        record(tx,&id,"child_intent",&json!({"intent":intent}))?;Ok(())
    }).unwrap();
    (ledger, lease)
}
fn sleep() -> OwnedProcess {
    OwnedProcess(
        Command::new("/bin/sleep")
            .arg("120")
            .process_group(0)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    )
}
#[test]
fn native_registration_rejects_non_pidfd_duplicate_and_stopped_admission() {
    let path = std::env::temp_dir().join(format!("promotion-registration-{}", std::process::id()));
    let (mut ledger, lease) = admitted(&path);
    let worker = sleep();
    let fd = pin(worker.0.id());
    let ordinary: OwnedFd = std::fs::File::open("/dev/null").unwrap().into();
    assert!(ledger
        .register_fixture_reload_child(&lease, &ordinary)
        .is_err());
    assert!(
        read(&ledger.connection, lease.contract(), "child_registered")
            .unwrap()
            .is_none()
    );
    ledger.register_fixture_reload_child(&lease, &fd.0).unwrap();
    assert!(ledger.register_fixture_reload_child(&lease, &fd.0).is_err());
    assert!(!supervision::all_cleanup_confirmed(&ledger.connection, lease.contract()).unwrap());
    assert!(
        ledger.recover_fixture_promotion(&lease).is_err(),
        "live controller cannot be recovered"
    );
    let intent = read(&ledger.connection, lease.contract(), "intent")
        .unwrap()
        .unwrap()
        .0;
    let tx = ledger.connection.transaction().unwrap();
    record(
        &tx,
        lease.contract(),
        "reload",
        &json!({"cleanup_confirmed":true}),
    )
    .unwrap();
    record(
        &tx,
        lease.contract(),
        "supervision",
        &json!({"intent":intent,"cleanup_confirmed":true}),
    )
    .unwrap();
    tx.commit().unwrap();
    assert!(
        !supervision::all_cleanup_confirmed(&ledger.connection, lease.contract()).unwrap(),
        "report alone cannot clean a registered live child"
    );
    drop(worker);
    assert!(supervision::all_cleanup_confirmed(&ledger.connection, lease.contract()).unwrap());
    drop(fd);
    drop(ledger);
    std::fs::remove_dir_all(&path).unwrap();
    let (mut ledger, lease) = admitted(&path);
    let worker = sleep();
    let fd = pin(worker.0.id());
    ledger
        .request_stop(
            lease.contract(),
            fixture::d("before-registration"),
            supervision::StopReason::OwnerStop,
        )
        .unwrap();
    assert!(ledger.register_fixture_reload_child(&lease, &fd.0).is_err());
    assert!(
        read(&ledger.connection, lease.contract(), "child_registered")
            .unwrap()
            .is_none()
    );
    assert_eq!(
        ledger
            .budget_snapshot(lease.contract())
            .unwrap()
            .held
            .workers,
        1
    );
    drop(fd);
    drop(worker);
    drop(ledger);
    std::fs::remove_dir_all(path).unwrap();
}
#[test]
fn registered_crash_recovery_reaps_exact_child_and_never_signals_mismatched_start() {
    if let Some(path) = std::env::var_os("NF_REGISTERED_CRASH_DB") {
        let (mut ledger, lease) = admitted(Path::new(&path));
        let worker = sleep();
        let fd = pin(worker.0.id());
        ledger.register_fixture_reload_child(&lease, &fd.0).unwrap();
        let mode = std::env::var("NF_REGISTERED_CRASH_MODE").unwrap();
        if mode != "live" {
            // Synthetic damaged storage models stale identity/foreign binding;
            // this does NOT claim to force actual kernel PID reuse.
            let (_, mut value) = read(&ledger.connection, lease.contract(), "child_registered")
                .unwrap()
                .unwrap();
            if mode == "different_start" {
                value["process"]["start_ticks"] =
                    json!(value["process"]["start_ticks"].as_u64().unwrap() + 1);
            } else {
                value["reservation"] = json!(fixture::d("foreign-registration"));
            }
            ledger
                .connection
                .execute_batch("DROP TRIGGER promotion_no_update;")
                .unwrap();
            ledger
                .connection
                .execute(
                    "UPDATE promotion_records SET body=?1,digest=?2 WHERE kind='child_registered'",
                    params![canonical(&value).unwrap(), hash(&value).unwrap().as_str()],
                )
                .unwrap();
            ledger.connection.execute_batch("CREATE TRIGGER promotion_no_update BEFORE UPDATE ON promotion_records BEGIN SELECT RAISE(ABORT,'immutable promotion evidence');END;").unwrap();
        }
        ledger.release(&lease).unwrap();
        unsafe {
            libc::raise(libc::SIGSTOP);
        }
        unreachable!();
    }
    // Isolate process-global subreaper state in the existing dedicated harness.
    if e2e::dedicated_test() {
        return;
    }
    assert_eq!(unsafe { libc::prctl(libc::PR_SET_CHILD_SUBREAPER, 1) }, 0);
    for mode in ["live", "different_start", "foreign_binding"] {
        let path = std::env::temp_dir().join(format!(
            "promotion-registered-crash-{}-{mode}",
            std::process::id()
        ));
        let mut controller=OwnedProcess(Command::new(std::env::current_exe().unwrap())
            .args(["--exact","coordinator::promotion::tests::registration::registered_crash_recovery_reaps_exact_child_and_never_signals_mismatched_start","--nocapture"])
            .env("NF_REGISTERED_CRASH_DB",&path).env("NF_REGISTERED_CRASH_MODE",mode).spawn().unwrap());
        let until = std::time::Instant::now() + Duration::from_secs(30);
        let mut status = 0;
        loop {
            let observed = unsafe {
                libc::waitpid(
                    controller.0.id() as i32,
                    &mut status,
                    libc::WUNTRACED | libc::WNOHANG,
                )
            };
            if observed == controller.0.id() as i32 {
                break;
            }
            assert_eq!(observed, 0);
            assert!(
                std::time::Instant::now() < until,
                "bounded crash boundary expired"
            );
            std::thread::sleep(Duration::from_millis(2));
        }
        assert!(libc::WIFSTOPPED(status));
        let mut ledger = Ledger::open(&path, LedgerLimits::default()).unwrap();
        let c =
            Validated::<StudyContract>::parse(&serde_json::to_vec(&fixture::standard()).unwrap())
                .unwrap();
        let id = c.digest().unwrap();
        let registration = read(&ledger.connection, &id, "child_registered")
            .unwrap()
            .unwrap()
            .1;
        let fd = pin(registration["process"]["pid"].as_u64().unwrap() as u32);
        controller.0.kill().unwrap();
        controller.0.wait().unwrap();
        let lease = ledger.acquire(&id, "recovery", 300000).unwrap();
        let inventory = supervision::all_cleanup_confirmed(&ledger.connection, &id);
        if mode == "foreign_binding" {
            assert!(inventory.is_err());
        } else {
            assert!(!inventory.unwrap());
        }
        let result = ledger.recover_fixture_promotion(&lease);
        assert_eq!(
            result.is_ok(),
            mode != "foreign_binding",
            "{mode}: {result:?}"
        );
        let mut poll = libc::pollfd {
            fd: fd.0.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        assert_eq!(
            unsafe { libc::poll(&mut poll, 1, 0) },
            if mode == "live" { 1 } else { 0 },
            "must never signal mismatched identity"
        );
        assert_eq!(
            ledger.budget_snapshot(&id).unwrap().held.workers,
            if mode == "foreign_binding" { 1 } else { 0 }
        );
        let inventory = supervision::all_cleanup_confirmed(&ledger.connection, &id);
        if mode == "foreign_binding" {
            assert!(inventory.is_err());
        } else {
            assert!(inventory.unwrap());
        }
        assert!(ledger
            .accepted_fixture_head(&c.study_family)
            .unwrap()
            .is_none());
        if mode == "live" {
            assert_eq!(
                read(&ledger.connection, &id, "child_recovery")
                    .unwrap()
                    .unwrap()
                    .1["reaped_by_recovery"],
                true
            );
        }
        if result.is_ok() {
            ledger.recover_fixture_promotion(&lease).unwrap();
        }
        // Explicit owned cleanup even for the intentionally untouched sentinel.
        unsafe {
            libc::syscall(
                libc::SYS_pidfd_send_signal,
                fd.0.as_raw_fd(),
                libc::SIGKILL,
                std::ptr::null::<libc::siginfo_t>(),
                0,
            );
        }
        let until = std::time::Instant::now() + Duration::from_secs(2);
        while unsafe { libc::poll(&mut poll, 1, 0) } == 0 {
            assert!(std::time::Instant::now() < until);
            std::thread::sleep(Duration::from_millis(2));
        }
        drop(fd);
        drop(controller);
        drop(ledger);
        std::fs::remove_dir_all(path).unwrap();
    }
}
