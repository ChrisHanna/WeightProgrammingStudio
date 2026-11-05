use super::e2e::Fixture;
use super::*;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
fn stopped_child(root: &Path) -> OwnedFd {
    let db = Connection::open_with_flags(
        root.join("ledger/events.sqlite3"),
        OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .unwrap();
    let until = std::time::Instant::now() + Duration::from_secs(30);
    loop {
        for runtime in std::fs::read_dir(root)
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with("runtime-"))
        {
            for job in std::fs::read_dir(runtime.path()).unwrap().flatten() {
                if !job.path().join("reload-intent.json").exists() {
                    continue;
                }
                // Exercise stop/takeover AFTER durable registration, not a race
                // against registration refusal (which has its own gate test).
                let registered=db.query_row("SELECT EXISTS(SELECT 1 FROM promotion_records WHERE kind='child_registered')",[],|r|r.get::<_,bool>(0)).unwrap_or(false);
                if !registered {
                    continue;
                }
                let Ok(bytes) = std::fs::read(job.path().join("process.json")) else {
                    continue;
                };
                let Ok(value) = serde_json::from_slice::<Value>(&bytes) else {
                    continue;
                };
                let pid = value["pid"].as_i64().unwrap() as i32;
                // Pin identity before STOP; never signal a potentially reused PID.
                let fd = unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0) };
                assert!(fd >= 0, "reload child exited before deterministic stop");
                let fd = unsafe { OwnedFd::from_raw_fd(fd as i32) };
                // process.json is not authority: verify the pinned live process
                // is OUR child and its command names this exact private job.
                let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) else {
                    continue;
                };
                let Some((_, tail)) = stat.rsplit_once(')') else {
                    continue;
                };
                let fields = tail.split_whitespace().collect::<Vec<_>>();
                if fields.get(1).and_then(|s| s.parse::<u32>().ok()) != Some(std::process::id())
                    || fields.get(2).and_then(|s| s.parse::<i32>().ok()) != Some(pid)
                {
                    continue;
                }
                use std::io::Read;
                let Ok(file) = std::fs::File::open(format!("/proc/{pid}/cmdline")) else {
                    continue;
                };
                let mut command = Vec::new();
                file.take(16385).read_to_end(&mut command).unwrap();
                let job_path = job.path();
                use std::os::unix::ffi::OsStrExt;
                if command.len() > 16384
                    || !command
                        .split(|b| *b == 0)
                        .any(|arg| arg == job_path.as_os_str().as_bytes())
                {
                    continue;
                }
                let mut poll = libc::pollfd {
                    fd: fd.as_raw_fd(),
                    events: libc::POLLIN,
                    revents: 0,
                };
                if unsafe { libc::poll(&mut poll, 1, 0) } != 0 {
                    continue;
                }
                assert_eq!(
                    unsafe {
                        libc::syscall(
                            libc::SYS_pidfd_send_signal,
                            fd.as_raw_fd(),
                            libc::SIGSTOP,
                            std::ptr::null::<libc::siginfo_t>(),
                            0,
                        )
                    },
                    0
                );
                return fd;
            }
        }
        assert!(
            std::time::Instant::now() < until,
            "reload child not launched"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}
#[test]
fn private_live_child_stop_and_takeover_reap_and_retain_evidence() {
    let Some(case) = e2e::isolated_cases(&["stop", "takeover"]) else {
        return;
    };
    for takeover in [case == "takeover"] {
        let mut f = Fixture::new(true);
        let runtime = f.runtime();
        let qualification = f
            .ledger
            .qualify_fixture_promotion(&f.lease, &f.store, PromotionScope::OperationalFixture)
            .unwrap();
        operator::test_registry(&f.root.join("operator"));
        let request = f
            .ledger
            .fixture_promotion_approval_request(
                &f.lease,
                &qualification,
                &f.edit,
                &f.smoke,
                &runtime,
                fixture::d("live-child"),
                None,
                &CancellationToken::default(),
            )
            .unwrap();
        let id = operator::issue(&request, operator::test_expiration()).unwrap();
        let grant = f
            .ledger
            .approve_fixture_promotion(
                &f.lease,
                &qualification,
                id,
                &f.edit,
                &f.smoke,
                &runtime,
                fixture::d("live-child"),
                None,
                &CancellationToken::default(),
            )
            .unwrap();
        let root = f.root.clone();
        let lease = f.lease.clone();
        let racer = std::thread::spawn(move || {
            let fd = stopped_child(&root);
            let mut other = Ledger::open(&root.join("ledger"), LedgerLimits::default()).unwrap();
            assert!(
                !supervision::all_cleanup_confirmed(&other.connection, lease.contract()).unwrap()
            );
            if takeover {
                other.release(&lease).unwrap();
                other.acquire(lease.contract(), "takeover", 300000).unwrap();
            } else {
                other
                    .request_stop(
                        lease.contract(),
                        fixture::d("live-stop"),
                        supervision::StopReason::OwnerStop,
                    )
                    .unwrap();
            }
            (fd, std::time::Instant::now())
        });
        let result = f.ledger.promote_fixture_with_grant(
            &f.lease,
            &f.store,
            &f.edit,
            &f.smoke,
            &runtime,
            fixture::d("live-child"),
            None,
            &CancellationToken::default(),
            &grant,
        );
        let (fd, stopped) = racer.join().unwrap_or_else(|_| {
            panic!("reload launch observer failed; promotion result: {result:?}")
        });
        assert!(result.is_err());
        assert!(stopped.elapsed() < Duration::from_secs(3));
        let mut poll = libc::pollfd {
            fd: fd.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        assert_eq!(
            unsafe { libc::poll(&mut poll, 1, 0) },
            1,
            "kernel handle must report child exit"
        );
        assert!(
            read(&f.ledger.connection, f.lease.contract(), "child_registered")
                .unwrap()
                .is_some(),
            "real stopped child must have durable registration"
        );
        let report = read(&f.ledger.connection, f.lease.contract(), "reload")
            .unwrap()
            .unwrap()
            .1;
        assert_eq!(report["cleanup_confirmed"], true);
        assert_eq!(report["child_started"], true);
        assert_eq!(report["status"], "failed_reload");
        assert!(report["cpu_ms"].as_u64().is_some());
        assert!(
            read(&f.ledger.connection, f.lease.contract(), "supervision")
                .unwrap()
                .is_some()
        );
        assert!(
            supervision::all_cleanup_confirmed(&f.ledger.connection, f.lease.contract()).unwrap()
        );
        f.unchanged();
        if takeover {
            assert_eq!(
                f.ledger
                    .budget_snapshot(f.lease.contract())
                    .unwrap()
                    .held
                    .workers,
                1
            );
            assert!(
                read(&f.ledger.connection, f.lease.contract(), "attempt_end")
                    .unwrap()
                    .is_some()
            );
        }
    }
}
#[test]
fn root_grant_expiration_or_revocation_during_registered_reload_blocks_final_commit() {
    let Some(case) = e2e::isolated_cases(&["expired", "revoked"]) else {
        return;
    };
    let mut f = Fixture::new(true);
    let runtime = f.runtime();
    let qualification = f
        .ledger
        .qualify_fixture_promotion(&f.lease, &f.store, PromotionScope::OperationalFixture)
        .unwrap();
    operator::test_registry(&f.root.join("operator"));
    let request = f
        .ledger
        .fixture_promotion_approval_request(
            &f.lease,
            &qualification,
            &f.edit,
            &f.smoke,
            &runtime,
            fixture::d("expiring-child"),
            None,
            &CancellationToken::default(),
        )
        .unwrap();
    let expires = if case == "expired" {
        operator::test_expiration() - 295_000
    } else {
        operator::test_expiration()
    };
    let id = operator::issue(&request, expires).unwrap();
    let grant = f
        .ledger
        .approve_fixture_promotion(
            &f.lease,
            &qualification,
            id.clone(),
            &f.edit,
            &f.smoke,
            &runtime,
            fixture::d("expiring-child"),
            None,
            &CancellationToken::default(),
        )
        .unwrap();
    let root = f.root.clone();
    let racer = std::thread::spawn(move || {
        let fd = stopped_child(&root);
        operator::test_registry(&root.join("operator"));
        if case == "revoked" {
            // Durable registration can become visible just before its callback
            // returns and releases the short registration guard. Busy is honest;
            // retry it, without resuming the child, until revocation succeeds.
            let until = std::time::Instant::now() + Duration::from_secs(2);
            while let Err(error) = operator::revoke(&id) {
                assert!(
                    std::time::Instant::now() < until,
                    "revocation remained busy during native work: {error}"
                );
                std::thread::sleep(Duration::from_millis(2));
            }
        } else {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;
            std::thread::sleep(Duration::from_millis(expires.saturating_sub(now) + 30));
        }
        assert_eq!(
            unsafe {
                libc::syscall(
                    libc::SYS_pidfd_send_signal,
                    fd.as_raw_fd(),
                    libc::SIGCONT,
                    std::ptr::null::<libc::siginfo_t>(),
                    0,
                )
            },
            0
        );
        fd
    });
    let result = f.ledger.promote_fixture_with_grant(
        &f.lease,
        &f.store,
        &f.edit,
        &f.smoke,
        &runtime,
        fixture::d("expiring-child"),
        None,
        &CancellationToken::default(),
        &grant,
    );
    let fd = racer.join().unwrap();
    assert!(result.is_err(), "expired/revoked grant published");
    let mut poll = libc::pollfd {
        fd: fd.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    assert_eq!(unsafe { libc::poll(&mut poll, 1, 0) }, 1);
    let report = read(&f.ledger.connection, f.lease.contract(), "reload")
        .unwrap()
        .unwrap()
        .1;
    assert_eq!(report["status"], "verified_fresh_process_reload");
    assert_eq!(report["cleanup_confirmed"], true);
    assert!(read(&f.ledger.connection, f.lease.contract(), "costs")
        .unwrap()
        .is_none());
    assert_eq!(
        f.ledger
            .budget_snapshot(f.lease.contract())
            .unwrap()
            .held
            .workers,
        1
    );
    f.unchanged();
}
#[test]
fn real_reload_child_is_reaped_on_coordinator_deadline() {
    if e2e::dedicated_test() {
        return;
    }
    let mut f = Fixture::new(true);
    let runtime = f.runtime();
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
    let probes = FrozenReloadProbes::freeze(
        &reference,
        model(&reference).unwrap(),
        inputs.clone(),
        vec![c.seeds[0]; inputs.len()],
    )
    .unwrap();
    let export = export_candidate(&f.store, &candidate, &f.smoke, export_limits()).unwrap();
    super::gated::admit(&mut f);
    let cancel = CancellationToken::default();
    let mut guard = None;
    let mut pinned = None;
    let mut started = None;
    let attempt = runtime.verify_reload_registered(
        &f.store,
        &export.export_root,
        &probes,
        WorkerLimits::default(),
        &cancel,
        &mut |fd| {
            f.ledger.register_fixture_reload_child(&f.lease, fd)?;
            assert_eq!(
                unsafe {
                    libc::syscall(
                        libc::SYS_pidfd_send_signal,
                        fd.as_raw_fd(),
                        libc::SIGSTOP,
                        std::ptr::null::<libc::siginfo_t>(),
                        0,
                    )
                },
                0
            );
            pinned = Some(fd.try_clone().unwrap());
            started = Some(std::time::Instant::now());
            guard = Some(supervision_guard::Guard::start(
                f.ledger.files.sqlite_path(),
                f.ledger.store.clone(),
                f.lease.clone(),
                files::moment()?.ms + 100,
                CancellationToken::default(),
                cancel.clone(),
            )?);
            Ok(())
        },
    );
    assert!(guard.unwrap().finish().is_err());
    assert!(started.unwrap().elapsed() < Duration::from_secs(3));
    assert!(attempt.witness().is_none());
    assert!(attempt.report().cleanup_confirmed);
    assert_eq!(attempt.report().child_started, Some(true));
    assert!(attempt.report().cpu_ms.is_some());
    let fd = pinned.unwrap();
    let mut poll = libc::pollfd {
        fd: fd.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    assert_eq!(unsafe { libc::poll(&mut poll, 1, 0) }, 1);
    assert!(
        read(&f.ledger.connection, f.lease.contract(), "child_registered")
            .unwrap()
            .is_some()
    );
    f.unchanged();
}
#[test]
fn crashed_attempt_recovery_releases_only_proven_cleanup() {
    if let Some(path) = std::env::var_os("NF_PROMOTION_RECOVERY_DB") {
        let mode = std::env::var("NF_PROMOTION_RECOVERY_MODE").unwrap();
        let mut ledger = Ledger::open(Path::new(&path), LedgerLimits::default()).unwrap();
        let c =
            Validated::<StudyContract>::parse(&serde_json::to_vec(&fixture::standard()).unwrap())
                .unwrap();
        let id = ledger.freeze(&c).unwrap();
        let lease = ledger.acquire(&id, "crash-owner", 300000).unwrap();
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
                        operation: fixture::d(&format!("recovery-{i}")),
                        expected: previous,
                        next,
                        protocol: fixture::d("recovery"),
                        evidence: fixture::d("recovery"),
                        message: "storage-only crash test".into(),
                    },
                )
                .unwrap();
            previous = next;
        }
        let store = ledger.store.clone();
        let mut owner = json!(process::Owner::current().unwrap());
        // Synthetic prior-boot identity exercises reboot reconciliation, not an
        // assertion that this test actually restarted the host kernel.
        if mode == "previous_boot" {
            owner["boot"] = json!("test-previous-kernel-boot");
        }
        ledger.reserve_with(&lease,Request {schema_version:1,operation:fixture::d("crash-cost"),attempt:0,parent:None,kind:WorkKind::Reload,
            resources:Resources {cpu_ms:100,storage_bytes:65536,host_bytes:1024,workers:1,..Default::default()},wall_ms:10000,external:None},|tx,reservation,deadline| {
            let intent=record(tx,&id,"intent",&json!({"contract":id,"store":store,"reservation":reservation.id(),"native_process":owner,"boot":owner["boot"],"deadline":deadline}))?;
            if mode!="before_child" {record(tx,&id,"child_intent",&json!({"intent":intent}))?;}
            if mode=="reaped" {
                record(tx,&id,"reload",&json!({"cleanup_confirmed":true,"status":"failed_reload"}))?;
                record(tx,&id,"supervision",&json!({"intent":intent,"cleanup_confirmed":true}))?;
            }
            Ok(())
        }).unwrap();
        if mode != "previous_boot" {
            assert!(ledger.recover_fixture_promotion(&lease).is_err());
        }
        ledger.release(&lease).unwrap();
        // Parent kills this stopped controller: no Rust destructors or graceful
        // exit can provide the cleanup evidence inspected after reopening.
        unsafe {
            libc::raise(libc::SIGSTOP);
        }
        unreachable!();
    }
    for mode in ["before_child", "unknown_child", "reaped", "previous_boot"] {
        let path =
            std::env::temp_dir().join(format!("promotion-recovery-{}-{mode}", std::process::id()));
        let mut child=std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact","coordinator::promotion::tests::lifecycle::crashed_attempt_recovery_releases_only_proven_cleanup","--nocapture"])
            .env("NF_PROMOTION_RECOVERY_DB",&path).env("NF_PROMOTION_RECOVERY_MODE",mode).spawn().unwrap();
        let mut status = 0;
        let until = std::time::Instant::now() + Duration::from_secs(30);
        loop {
            let observed = unsafe {
                libc::waitpid(
                    child.id() as i32,
                    &mut status,
                    libc::WUNTRACED | libc::WNOHANG,
                )
            };
            if observed == child.id() as i32 {
                break;
            }
            assert_eq!(observed, 0, "controller wait failed");
            if std::time::Instant::now() >= until {
                let _ = child.kill();
                let _ = child.wait();
                panic!("controller failed to reach bounded crash boundary");
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        assert!(
            libc::WIFSTOPPED(status),
            "controller failed before crash boundary: {status}"
        );
        child.kill().unwrap();
        use std::os::unix::process::ExitStatusExt;
        assert_eq!(child.wait().unwrap().signal(), Some(libc::SIGKILL));
        let mut ledger = Ledger::open(&path, LedgerLimits::default()).unwrap();
        let c =
            Validated::<StudyContract>::parse(&serde_json::to_vec(&fixture::standard()).unwrap())
                .unwrap();
        let id = c.digest().unwrap();
        let lease = ledger.acquire(&id, "recovery-owner", 300000).unwrap();
        assert_eq!(
            supervision::all_cleanup_confirmed(&ledger.connection, &id).unwrap(),
            mode == "before_child" || mode == "reaped"
        );
        let result = ledger.recover_fixture_promotion(&lease);
        assert_eq!(
            result.is_ok(),
            mode != "unknown_child",
            "{mode}: {result:?}"
        );
        assert_eq!(
            supervision::all_cleanup_confirmed(&ledger.connection, &id).unwrap(),
            mode != "unknown_child"
        );
        assert_eq!(
            read(&ledger.connection, &id, "child_recovery")
                .unwrap()
                .is_some(),
            mode == "previous_boot"
        );
        assert_eq!(
            ledger.budget_snapshot(&id).unwrap().held.workers,
            if mode == "unknown_child" { 1 } else { 0 }
        );
        assert!(ledger
            .accepted_fixture_head(&c.study_family)
            .unwrap()
            .is_none());
        assert_eq!(ledger.state(&id).unwrap().stage, Stage::Verify);
        if result.is_ok() {
            ledger.recover_fixture_promotion(&lease).unwrap();
        }
        drop(ledger);
        std::fs::remove_dir_all(path).unwrap();
    }
}
