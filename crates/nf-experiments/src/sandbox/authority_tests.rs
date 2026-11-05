//! Real pinned local OCI execution; no synthetic runtime or guest authority DTOs.
use super::*;
use std::cell::Cell;
use std::os::unix::fs::PermissionsExt;

struct Root(PathBuf);
impl Root {
    fn new() -> Self {
        let path = std::env::temp_dir().join(random_name().unwrap());
        fs::create_dir(&path).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
        Self(path)
    }
}
impl Drop for Root {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn runtime(root: &Root) -> QualifiedDocker {
    QualifiedDocker::qualify(true, &root.0).expect("real pinned OCI qualification required")
}
fn deny() -> CoreError {
    error(ErrorCode::ScopeViolation, "durable grant fenced")
}
fn limits() -> Limits {
    Limits {
        wall_ms: 5000,
        ..Limits::default()
    }
}
fn finalized(root: &Root) {
    for path in storage::jobs(&root.0).unwrap() {
        let receipt = storage::read_receipt(&path.join("completion.json")).unwrap();
        assert!(receipt.cleanup_confirmed);
        assert!(receipt.storage.unwrap().workspace_reclaimed);
        assert!(!path.join("job").exists());
        assert!(!path.join("guard").exists());
    }
}

#[test]
fn denied_and_short_deadline_never_stage() {
    let root = Root::new();
    let backend = runtime(&root);
    let before = storage::jobs(&root.0).unwrap();
    let cancel = AtomicBool::new(false);
    assert!(backend
        .execute_assets_authorized(
            b"print(1)",
            b"{}",
            &BTreeMap::new(),
            limits(),
            STORAGE_JOB_BYTES,
            &cancel,
            Instant::now() + Duration::from_secs(10),
            &|| Err(deny())
        )
        .is_err());
    assert!(backend
        .execute_assets_authorized(
            b"print(1)",
            b"{}",
            &BTreeMap::new(),
            limits(),
            STORAGE_JOB_BYTES,
            &cancel,
            Instant::now() + Duration::from_millis(50),
            &|| Ok(())
        )
        .is_err());
    assert_eq!(storage::jobs(&root.0).unwrap(), before);
}

#[test]
fn staging_revocation_reclaims_without_launching() {
    let root = Root::new();
    let backend = runtime(&root);
    let before = storage::jobs(&root.0).unwrap();
    let calls = Cell::new(0);
    let authorize = || {
        calls.set(calls.get() + 1);
        // Revoke only after a real staging directory exists.
        if storage::jobs(&root.0)
            .unwrap()
            .iter()
            .any(|p| p.join("guard").exists())
        {
            Err(deny())
        } else {
            Ok(())
        }
    };
    assert!(backend
        .execute_assets_authorized(
            b"print(1)",
            b"{}",
            &BTreeMap::new(),
            limits(),
            STORAGE_JOB_BYTES,
            &AtomicBool::new(false),
            Instant::now() + Duration::from_secs(10),
            &authorize
        )
        .is_err());
    assert!(calls.get() > 2);
    finalized(&root);
    for path in storage::jobs(&root.0)
        .unwrap()
        .into_iter()
        .filter(|p| !before.contains(p))
    {
        assert!(!path.join("configured.json").exists());
    }
}

#[test]
fn persisted_stop_and_lease_fence_kill_active_container() {
    for lease_fence in [false, true] {
        let root = Root::new();
        let backend = runtime(&root);
        let database_root = Root::new();
        let database = database_root.0.join("grant.sqlite");
        let db = rusqlite::Connection::open(&database).unwrap();
        let token = random_name().unwrap();
        db.execute_batch("CREATE TABLE grant(token TEXT PRIMARY KEY, stopped INTEGER NOT NULL, epoch INTEGER NOT NULL)").unwrap();
        db.execute("INSERT INTO grant VALUES (?1,0,1)", [&token])
            .unwrap();
        let done = AtomicBool::new(false);
        std::thread::scope(|scope| {
            let root_path = &root.0;
            let database = &database;
            let token = &token;
            let done = &done;
            let writer = scope.spawn(move || {
                let until = Instant::now() + Duration::from_secs(10);
                while Instant::now() < until && !done.load(Ordering::Acquire) {
                    for path in storage::jobs(root_path).unwrap() {
                        if !path.join("configured.json").exists() {
                            continue;
                        }
                        let name = path.file_name().unwrap().to_str().unwrap();
                        if QualifiedDocker::container_running(name).unwrap_or(false) {
                            let writer = rusqlite::Connection::open(database).unwrap();
                            let sql = if lease_fence {
                                "UPDATE grant SET epoch=2 WHERE token=?1"
                            } else {
                                "UPDATE grant SET stopped=1 WHERE token=?1"
                            };
                            assert_eq!(writer.execute(sql, [token]).unwrap(), 1);
                            return true;
                        }
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
                false
            });
            // Exact controller-owned token and lease epoch, not caller JSON.
            let authorize = || {
                let live: bool = db
                    .query_row(
                        "SELECT stopped=0 AND epoch=1 FROM grant WHERE token=?1",
                        [token],
                        |row| row.get(0),
                    )
                    .map_err(|_| deny())?;
                if live {
                    Ok(())
                } else {
                    Err(deny())
                }
            };
            let start = Instant::now();
            let result = backend.execute_assets_authorized(
                b"import time\ntime.sleep(30)",
                b"{}",
                &BTreeMap::new(),
                limits(),
                STORAGE_JOB_BYTES,
                &AtomicBool::new(false),
                start + Duration::from_secs(10),
                &authorize,
            );
            done.store(true, Ordering::Release);
            assert!(
                writer.join().unwrap(),
                "must revoke a genuinely running container"
            );
            assert!(result.is_err());
            assert!(start.elapsed() < Duration::from_secs(5));
        });
        finalized(&root);
        assert_eq!(backend.recover_pending().unwrap(), 0);
    }
}

#[test]
fn cancellation_during_create_drains_terminal_response_before_cleanup() {
    let root = Root::new();
    let backend = runtime(&root);
    let before = storage::jobs(&root.0).unwrap();
    let cancel = AtomicBool::new(false);
    let reached = Cell::new(false);
    let authorize = || {
        if storage::jobs(&root.0)
            .unwrap()
            .iter()
            .filter(|p| !before.contains(p))
            .any(|p| p.join("create_started.json").exists())
        {
            reached.set(true);
            cancel.store(true, Ordering::Release);
        }
        Ok(())
    };
    let receipt = backend
        .execute_assets_authorized(
            b"print('not authorized to start')",
            b"{}",
            &BTreeMap::new(),
            limits(),
            STORAGE_JOB_BYTES,
            &cancel,
            Instant::now() + Duration::from_secs(15),
            &authorize,
        )
        .unwrap();
    assert!(reached.get());
    assert_eq!(receipt.status, "cancelled");
    assert!(receipt.cleanup_confirmed);
    assert!(root
        .0
        .join(&receipt.job)
        .join("create_terminal.json")
        .is_file());
    assert!(backend.absent(&receipt.job).unwrap());
    assert_eq!(backend.recover_pending().unwrap(), 0);
    assert!(backend.absent(&receipt.job).unwrap());
}

#[test]
fn uncertain_create_never_certifies_absence_or_reclaims_reservation() {
    let root = Root::new();
    let backend = runtime(&root);
    let path = root.0.join(random_name().unwrap());
    {
        let _lock = storage::Lock::acquire(&root.0).unwrap();
        let job = storage::reserve(&root.0, &path, STORAGE_JOB_BYTES).unwrap();
        record(
            &job,
            &path.join("create_started.json"),
            &json!({"schema_version":1}),
        )
        .unwrap();
    }
    let error = backend.recover_pending().unwrap_err();
    assert!(error.message.contains("no durable terminal response"));
    assert!(path.join("reservation.bin").exists());
    assert!(!path.join("recovery.json").exists());
    assert!(backend
        .recover_attested(&backend.durable_root_identity().unwrap())
        .is_err());
}

#[test]
fn absolute_deadline_interrupts_active_execution() {
    let root = Root::new();
    let backend = runtime(&root);
    let before = storage::jobs(&root.0).unwrap();
    let start = Instant::now();
    // Consume allowance AFTER admission/staging, before starting the guest.
    // The absolute deadline must then beat the guest's five-second watchdog.
    let delayed = Cell::new(false);
    let active = Cell::new(false);
    let authorize = || {
        for path in storage::jobs(&root.0)
            .unwrap()
            .iter()
            .filter(|p| !before.contains(p))
        {
            if !path.join("configured.json").exists() {
                continue;
            }
            if !delayed.get() {
                delayed.set(true);
                std::thread::sleep(Duration::from_millis(1000));
            }
            let name = path.file_name().unwrap().to_str().unwrap();
            if QualifiedDocker::container_running(name)? {
                active.set(true);
            }
        }
        Ok(())
    };
    let result = backend.execute_assets_authorized(
        b"import time\ntime.sleep(30)",
        b"{}",
        &BTreeMap::new(),
        limits(),
        STORAGE_JOB_BYTES,
        &AtomicBool::new(false),
        start + Duration::from_millis(5500),
        &authorize,
    );
    assert!(result.is_err());
    assert!(
        active.get(),
        "deadline test must reach a genuinely active container"
    );
    assert!(start.elapsed() < Duration::from_secs(7));
    finalized(&root);
}
