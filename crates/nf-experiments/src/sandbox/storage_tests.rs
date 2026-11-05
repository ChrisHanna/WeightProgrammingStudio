//! Real local OCI fixtures plus filesystem accounting tests. No runtime mocks.
use super::*;
use std::os::unix::fs::{symlink, PermissionsExt};
use std::sync::atomic::AtomicU64;
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Root(PathBuf);
impl Root {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "nf-storage-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        fs::set_permissions(&p, fs::Permissions::from_mode(0o700)).unwrap();
        Self(p)
    }
}
impl Drop for Root {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn runtime(root: &Root) -> QualifiedDocker {
    QualifiedDocker::qualify(true, &root.0)
        .expect("real pinned local OCI runtime required; do not skip")
}
fn snapshot_runner(root: &Root) -> PathBuf {
    // Concurrent Cargo builds may unlink/replace current_exe's pathname. Pin
    // this running image through procfs for child processes and Docker binds.
    let binary = root.0.join("fixture-tests");
    fs::copy("/proc/self/exe", &binary).unwrap();
    binary
}

#[test]
fn maximum_jobs_reclaim_and_bound_retained_evidence() {
    let root = Root::new();
    let backend = runtime(&root);
    let assets: BTreeMap<_, _> = (0..64)
        .map(|i| (format!("a{i}/b/c/d/e/f/g/blob"), vec![b'x'; 131072]))
        .collect();
    let mut source = b"from pathlib import Path\nassert sum(p.stat().st_size for p in Path('/job/assets').rglob('blob')) == 8388608\nprint('x'*65535)\n#".to_vec();
    source.resize(65536, b' ');
    let input = vec![b' '; 65536];
    // Exercise maximal path/count/depth once, then retain the SAME maximum
    // 8 MiB payload and 64 KiB output with fewer files for the pruning cycles.
    // Repeating the quadratic per-file physical-tree scan 19 times adds no
    // boundary coverage and exceeded the offline 90-second named-test budget.
    let compact: BTreeMap<_, _> = (0..8)
        .map(|i| (format!("a{i}/blob"), vec![b'x'; 1048576]))
        .collect();
    for index in 0..(STORAGE_JOURNALS + 3) {
        eprintln!(
            "storage retention job {}/{} (8 MiB assets, 64 KiB output)",
            index + 1,
            STORAGE_JOURNALS + 3
        );
        let selected = if index == 0 { &assets } else { &compact };
        let receipt = backend
            .execute_assets(
                &source,
                &input,
                selected,
                Limits {
                    wall_ms: 5000,
                    cpu_ms: 5000,
                    output_bytes: 65536,
                },
                &AtomicBool::new(false),
            )
            .unwrap();
        assert_eq!(receipt.schema_version, 3);
        assert_eq!(receipt.status, "completed");
        assert!(receipt.cleanup_confirmed);
        let path = root.0.join(&receipt.job);
        let measured = receipt.storage.as_ref().unwrap();
        assert!(measured.workspace_reclaimed);
        assert_eq!(measured.reserved_bytes, STORAGE_JOB_BYTES - 4096);
        assert!(measured.accounting_finalized && !measured.capacity_guaranteed);
        assert!(measured.workspace_peak_charged_bytes >= measured.workspace_bytes_before_reclaim);
        assert!(measured.peak_charged_bytes <= STORAGE_JOB_BYTES);
        assert!(measured.workspace_bytes_before_reclaim >= 8388608 + 131072);
        assert!(measured.workspace_bytes_before_reclaim < STORAGE_JOB_BYTES);
        assert_eq!(measured.retained_bytes, storage::usage(&path).unwrap());
        assert_eq!(
            measured.root_retained_bytes,
            backend.storage_usage().unwrap()
        );
        assert!(measured.root_retained_bytes <= STORAGE_ROOT_BYTES);
        assert!(!path.join("job").exists() && !path.join("guard").exists());
        assert!(!path.join("reservation.bin").exists());
        assert!(path.join("pre_cleanup.json").is_file());
        let persisted = storage::read_receipt(&path.join("completion.json")).unwrap();
        assert_eq!(
            persisted.storage.unwrap().retained_bytes,
            measured.retained_bytes
        );
        assert!(storage::jobs(&root.0).unwrap().len() <= STORAGE_JOURNALS);
    }
    assert_eq!(backend.recover_pending().unwrap(), 0);
}

#[test]
fn physical_reservation_quota_and_path_guards() {
    let root = Root::new();
    let _lock = storage::Lock::acquire(&root.0).unwrap();
    assert!(storage::Lock::acquire(&root.0).is_err());
    let mut paths = Vec::new();
    loop {
        let path = root.0.join(random_name().unwrap());
        match storage::reserve(&root.0, &path, STORAGE_JOB_BYTES) {
            Ok(_) => {
                let m = fs::metadata(path.join("reservation.bin")).unwrap();
                assert!(m.blocks() * 512 >= m.len()); // not sparse accounting
                paths.push(path);
            }
            Err(_) => {
                assert!(!path.exists());
                break;
            }
        }
        assert!(paths.len() <= 4);
    }
    assert_eq!(paths.len(), 3); // Full admission probe remains charged until needed.
    assert!(storage::root_usage(&root.0).unwrap() <= STORAGE_ROOT_BYTES);
    let outside = Root::new();
    let sentinel = outside.0.join("sentinel");
    fs::write(&sentinel, b"keep").unwrap();
    symlink(&outside.0, paths[0].join("job")).unwrap();
    assert!(storage::reclaim(&paths[0]).is_err());
    assert_eq!(fs::read(&sentinel).unwrap(), b"keep");
    fs::remove_file(paths[0].join("job")).unwrap();
    fs::hard_link(&sentinel, paths[0].join("aliased")).unwrap();
    assert!(storage::usage(&paths[0]).is_err());
    fs::remove_file(paths[0].join("aliased")).unwrap();
    let alias = outside.0.join("root-alias");
    symlink(&root.0, &alias).unwrap();
    assert!(storage::Lock::acquire(&alias).is_err());
    let hostile = root.0.join(random_name().unwrap());
    symlink(&outside.0, &hostile).unwrap();
    assert!(storage::jobs(&root.0).is_err());
    fs::remove_file(hostile).unwrap();
    for path in paths {
        storage::reclaim(&path).unwrap();
        fs::remove_dir(path).unwrap();
    }
}

// A fresh OS process: controller death must release the flock, not only a Rust mutex.
#[test]
fn crash_child() {
    let Ok(root) = std::env::var("NF_STORAGE_CRASH_ROOT") else {
        return;
    };
    let backend = QualifiedDocker::qualify(true, Path::new(&root)).unwrap();
    backend
        .execute(
            b"import time\ntime.sleep(30)",
            b"{}",
            Limits {
                wall_ms: 5000,
                cpu_ms: 5000,
                output_bytes: 4096,
            },
            &AtomicBool::new(false),
        )
        .unwrap();
    panic!("parent should have killed the controller");
}

#[test]
fn interrupted_real_container_is_recovered_before_new_staging() {
    let root = Root::new();
    let backend = runtime(&root);
    let runner = Root::new();
    let mut child = Command::new(snapshot_runner(&runner))
        .args([
            "--exact",
            "sandbox::storage_tests::crash_child",
            "--nocapture",
        ])
        .env("NF_STORAGE_CRASH_ROOT", &root.0)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let start = Instant::now();
    let pending = loop {
        let found = storage::jobs(&root.0).unwrap().into_iter().find(|p| {
            p.join("configured.json").exists()
                && !p.join("completion.json").exists()
                && storage::read_receipt(&p.join("intent.json")).is_ok_and(|r| {
                    r.source == ContentDigest::of_bytes(b"import time\ntime.sleep(30)")
                })
                && backend
                    .inspect(p.file_name().unwrap().to_str().unwrap())
                    .is_ok_and(|v| v["State"]["Running"] == true)
        });
        if let Some(path) = found {
            break path;
        }
        if start.elapsed() > Duration::from_secs(30) {
            let _ = child.kill();
            let _ = child.wait();
            panic!("real child container never started");
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let count = storage::jobs(&root.0).unwrap().len();
    // Another capability/process cannot stage around the held root lock.
    assert!(backend
        .execute(
            b"print('bypass')",
            b"{}",
            Limits::default(),
            &AtomicBool::new(false)
        )
        .is_err());
    assert_eq!(storage::jobs(&root.0).unwrap().len(), count);
    child.kill().unwrap();
    child.wait().unwrap();
    assert!(pending.join("reservation.bin").is_file());
    let name = pending.file_name().unwrap().to_str().unwrap();
    // Qualification performs recovery before reserving space for its probe.
    let recovered_backend = runtime(&root);
    assert!(recovered_backend.absent(name).unwrap());
    let receipt = storage::read_receipt(&pending.join("recovery.json")).unwrap();
    assert!(receipt.cleanup_confirmed && receipt.storage.unwrap().workspace_reclaimed);
    assert_eq!(receipt.status, "incomplete");
    assert!(!receipt.outputs_eligible && receipt.report.is_none());
    assert!(!pending.join("job").exists());
    assert_eq!(recovered_backend.recover_pending().unwrap(), 0);
    assert!(recovered_backend.storage_usage().unwrap() < STORAGE_ROOT_BYTES);
}

#[test]
fn recovery_handles_pre_intent_and_partial_staging_crashes() {
    let root = Root::new();
    let backend = runtime(&root);
    let paths = {
        let _lock = storage::Lock::acquire(&root.0).unwrap();
        let paths: Vec<_> = (0..2)
            .map(|_| root.0.join(random_name().unwrap()))
            .collect();
        let jobs: Vec<_> = paths
            .iter()
            .map(|path| storage::reserve(&root.0, path, STORAGE_JOB_BYTES).unwrap())
            .collect();
        let job = &jobs[1];
        let prior = storage::jobs(&root.0)
            .unwrap()
            .into_iter()
            .find(|p| p.join("completion.json").exists())
            .unwrap();
        let mut receipt = storage::read_receipt(&prior.join("completion.json")).unwrap();
        receipt.job = paths[1].file_name().unwrap().to_str().unwrap().into();
        receipt.status = "incomplete".into();
        receipt.cleanup_confirmed = false;
        receipt.storage = None;
        receipt.report = None;
        record(&job, &paths[1].join("intent.json"), &receipt).unwrap();
        job.mkdir(&paths[1].join("job")).unwrap();
        save(&job, &paths[1].join("job/partial"), &[1; 65536]).unwrap();
        save(
            &job,
            &paths[1].join("receipt.tmp"),
            b"interrupted atomic receipt",
        )
        .unwrap();
        // Pre-storage-schema recovery markers must migrate, not strand assets.
        record(
            &job,
            &paths[1].join("recovery.json"),
            &json!({"cleanup_confirmed": true, "status": "incomplete", "outputs_eligible": false}),
        )
        .unwrap();
        paths
    };
    assert_eq!(backend.recover_pending().unwrap(), 2);
    assert!(!paths[0].exists());
    assert!(!paths[1].join("job").exists());
    assert!(
        storage::read_receipt(&paths[1].join("recovery.json"))
            .unwrap()
            .storage
            .unwrap()
            .workspace_reclaimed
    );
    assert_eq!(backend.recover_pending().unwrap(), 0);
}

#[test]
fn attacker_named_assets_use_real_job_and_study_quota() {
    let root = Root::new();
    let backend = runtime(&root);
    let nested = "nf-sandbox-00000000000000000000000000000000";
    let assets: BTreeMap<_, _> = (0..8)
        .map(|i| (format!("{nested}/{i}/blob"), vec![b'x'; 1048576]))
        .collect();
    let source = b"from pathlib import Path\nassert sum(p.stat().st_size for p in Path('/job/assets').rglob('blob')) == 8388608\nprint('accounted')";
    let limits = Limits {
        wall_ms: 5000,
        cpu_ms: 5000,
        output_bytes: 4096,
    };
    let bound = storage::workspace_bound(source.len(), 2, &assets, &limits).unwrap();
    let count = storage::jobs(&root.0).unwrap().len();
    assert!(backend
        .execute_assets_with_staging_limit(
            source,
            b"{}",
            &assets,
            limits.clone(),
            bound - 1,
            &AtomicBool::new(false)
        )
        .is_err());
    assert_eq!(storage::jobs(&root.0).unwrap().len(), count); // no staging on rejection
                                                              // Two crashed pre-intent reservations coexist with the reproducer.
    let _pending: Vec<_> = (0..2)
        .map(|_| {
            let _lock = storage::Lock::acquire(&root.0).unwrap();
            let path = root.0.join(random_name().unwrap());
            storage::reserve(&root.0, &path, STORAGE_JOB_BYTES).unwrap();
            path
        })
        .collect();
    let r = backend
        .execute_assets_with_staging_limit(
            source,
            b"{}",
            &assets,
            limits,
            bound,
            &AtomicBool::new(false),
        )
        .unwrap();
    assert_eq!(r.status, "completed");
    assert_eq!(
        decode_hex(r.report.as_ref().unwrap()["stdout_hex"].as_str().unwrap()).unwrap(),
        b"accounted\n"
    );
    let s = r.storage.unwrap();
    assert!(s.accounting_finalized && !s.capacity_guaranteed);
    assert!(s.workspace_peak_charged_bytes >= 8388608);
    assert!(s.workspace_peak_charged_bytes <= bound);
    assert!(s.peak_charged_bytes <= STORAGE_JOB_BYTES);
    assert!(backend.storage_usage().unwrap() <= STORAGE_ROOT_BYTES);
    assert_eq!(backend.recover_pending().unwrap(), 2);
}

#[test]
fn explicit_context_requires_reservation_and_enforces_exact_boundary() {
    let root = Root::new();
    let _lock = storage::Lock::acquire(&root.0).unwrap();
    let path = root.0.join(random_name().unwrap());
    let job = storage::reserve(&root.0, &path, 12288).unwrap();
    job.mkdir(&path.join("job")).unwrap();
    save(&job, &path.join("job/blob"), &[1; 4096]).unwrap();
    assert!(save(&job, &path.join("job/over"), &[1; 4097]).is_err());
    assert!(!path.join("job/over").exists());
    let outside = root.0.join("foreign");
    assert!(save(&job, &outside, b"x").is_err());
    assert!(save(&job, &path.join("job/../escape"), b"x").is_err());
    assert!(storage::Job::recovery(&root.0, &path.join("job")).is_err());
    fs::remove_file(path.join("reservation.bin")).unwrap();
    assert!(save(&job, &path.join("missing-reservation"), b"x").is_err());
    assert!(!path.join("missing-reservation").exists());
    let recovery = storage::Job::recovery(&root.0, &path).unwrap();
    assert!(save(&recovery, &path.join("job/recovery-stage"), b"forbidden").is_err());
    fs::write(path.join("reservation.bin"), []).unwrap();
    assert!(save(&job, &path.join("exhausted-stage"), b"forbidden").is_err());
    record(
        &recovery,
        &path.join("recovery-evidence.json"),
        &json!({"fixture":true}),
    )
    .unwrap();
}

fn fixture_receipt(path: &Path) -> Receipt {
    Receipt {
        schema_version: 3,
        job: path.file_name().unwrap().to_str().unwrap().into(),
        source: ContentDigest::of_bytes(b"fixture"),
        input: ContentDigest::of_bytes(b"{}"),
        asset_identity: None,
        capability: ContentDigest::of_bytes(b"fixture capability"),
        status: "incomplete".into(),
        cleanup_confirmed: true,
        outputs_eligible: false,
        promotion_status: "not_evaluated".into(),
        report: None,
        storage: None,
    }
}

#[test]
fn receipt_crash_child() {
    let Ok(root) = std::env::var("NF_RECEIPT_CRASH_ROOT") else {
        return;
    };
    let root = Path::new(&root);
    let _lock = storage::Lock::acquire(root).unwrap();
    let path = root.join(random_name().unwrap());
    let job = storage::reserve(root, &path, STORAGE_JOB_BYTES).unwrap();
    let mut receipt = fixture_receipt(&path);
    receipt.storage = Some(job.initial_receipt());
    record(&job, &path.join("intent.json"), &receipt).unwrap();
    job.mkdir(&path.join("job")).unwrap();
    save(&job, &path.join("job/blob"), &[7; 65536]).unwrap();
    let step = std::env::var("NF_RECEIPT_CRASH_STEP")
        .unwrap()
        .parse()
        .unwrap();
    storage::CRASH_STEP.with(|target| target.set(step));
    storage::finish(
        &job,
        &mut receipt,
        std::env::var("NF_RECEIPT_RECOVERY").unwrap() == "true",
    )
    .unwrap();
    panic!("crash hook not reached");
}

#[test]
fn every_receipt_step_is_crash_recoverable() {
    let runner = Root::new();
    let binary = snapshot_runner(&runner);
    for recovery in [false, true] {
        for step in 1..=9 {
            let root = Root::new();
            let backend = runtime(&root);
            let result = Command::new(&binary)
                .args([
                    "--exact",
                    "sandbox::storage_tests::receipt_crash_child",
                    "--nocapture",
                ])
                .env("NF_RECEIPT_CRASH_ROOT", &root.0)
                .env("NF_RECEIPT_CRASH_STEP", step.to_string())
                .env("NF_RECEIPT_RECOVERY", recovery.to_string())
                .output()
                .unwrap();
            assert_eq!(
                result.status.code(),
                Some(87),
                "step {step}: {}",
                String::from_utf8_lossy(&result.stderr)
            );
            assert!(storage::Lock::acquire(&root.0).is_ok()); // real process death released flock
            backend.recover_pending().unwrap();
            for path in storage::jobs(&root.0).unwrap() {
                let target = if path.join("recovery.json").exists() {
                    "recovery.json"
                } else {
                    "completion.json"
                };
                let receipt = storage::read_receipt(&path.join(target)).unwrap();
                assert!(storage::finalized(&path, &receipt).unwrap(), "step {step}");
                let s = receipt.storage.as_ref().unwrap();
                assert_eq!(s.retained_bytes, storage::usage(&path).unwrap());
                assert!(s.root_retained_bytes > 0 && s.root_retained_bytes <= STORAGE_ROOT_BYTES);
                if receipt.source == ContentDigest::of_bytes(b"fixture") {
                    assert!(!receipt.outputs_eligible && receipt.report.is_none());
                    assert_eq!(receipt.status, "incomplete");
                }
                assert!(!path.join("receipt.tmp").exists());
            }
            assert_eq!(backend.recover_pending().unwrap(), 0);
        }
    }
}

#[test]
fn same_device_mount_is_refused_before_any_unlink() {
    // Mount creation requires a capability unavailable to the ordinary worker.
    // Use a separate, networkless controller-owned fixture container, NOT the
    // untrusted execution profile. Failure remains a failed test, never a skip.
    if std::env::var_os("NF_STORAGE_MOUNT_CHILD").is_none() {
        isolated_storage_fixture(
            "sandbox::storage_tests::same_device_mount_is_refused_before_any_unlink",
        );
        return;
    }
    let root = Root::new();
    let outside = Root::new();
    let _lock = storage::Lock::acquire(&root.0).unwrap();
    let path = root.0.join(random_name().unwrap());
    let job = storage::reserve(&root.0, &path, STORAGE_JOB_BYTES).unwrap();
    job.mkdir(&path.join("job")).unwrap();
    fs::write(outside.0.join("sentinel"), b"keep").unwrap();
    let target = path.join("job/mounted");
    job.mkdir(&target).unwrap();
    assert!(Command::new("mount")
        .arg("--bind")
        .arg(&outside.0)
        .arg(&target)
        .status()
        .unwrap()
        .success());
    assert_eq!(
        fs::metadata(&outside.0).unwrap().dev(),
        fs::metadata(&target).unwrap().dev()
    );
    let refused = storage::reclaim(&path).is_err();
    assert!(storage::jobs(&root.0).is_err());
    assert!(storage::prune(&root.0, true).is_err());
    let kept = fs::read(outside.0.join("sentinel")).unwrap();
    assert!(Command::new("umount")
        .arg(&target)
        .status()
        .unwrap()
        .success());
    assert!(refused);
    assert_eq!(kept, b"keep");
}

fn isolated_storage_fixture(test: &str) {
    let runner = Root::new();
    let binary = snapshot_runner(&runner);
    let result = Command::new("/usr/bin/docker")
        .args([
            "--host",
            "unix:///var/run/docker.sock",
            "run",
            "--rm",
            "--pull=never",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--cap-add=SYS_ADMIN",
            "--security-opt=seccomp=unconfined",
            "--security-opt=apparmor=unconfined",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=67108864",
            "--env",
            "NF_STORAGE_MOUNT_CHILD=1",
            "--mount",
            &format!(
                "type=bind,src={},dst=/fixture-tests,readonly",
                binary.display()
            ),
            "--entrypoint",
            "/fixture-tests",
            IMAGE,
            "--exact",
            test,
            "--nocapture",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "storage fixture unavailable/failed: {} {}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
}

#[test]
fn enospc_capacity_probe_does_not_promise_future_writes() {
    if std::env::var_os("NF_STORAGE_MOUNT_CHILD").is_none() {
        isolated_storage_fixture(
            "sandbox::storage_tests::enospc_capacity_probe_does_not_promise_future_writes",
        );
        return;
    }
    let root = Root::new();
    let outside = Root::new();
    let _lock = storage::Lock::acquire(&root.0).unwrap();
    let path = root.0.join(random_name().unwrap());
    let job = storage::reserve(&root.0, &path, STORAGE_JOB_BYTES).unwrap();
    let mut receipt = fixture_receipt(&path);
    receipt.storage = Some(job.initial_receipt());
    record(&job, &path.join("intent.json"), &receipt).unwrap();
    job.mkdir(&path.join("job")).unwrap();
    let target = path.join("job/blob");
    // Exact production pre-write handoff, followed by a competing filesystem
    // allocation before destination write. Confine exhaustion to this tmpfs.
    job.before_save(&target, 65536).unwrap();
    let filler = outside.0.join("competing-writer");
    let mut f = File::create(&filler).unwrap();
    loop {
        match f.write_all(&[0; 4096]) {
            Ok(()) => (),
            Err(e) => {
                assert_eq!(e.raw_os_error(), Some(libc::ENOSPC));
                break;
            }
        }
    }
    let e = fs::write(&target, [1; 65536]).unwrap_err();
    assert_eq!(e.raw_os_error(), Some(libc::ENOSPC));
    assert!(!job.initial_receipt().capacity_guaranteed);
    let denied = root.0.join(random_name().unwrap());
    assert!(storage::reserve(&root.0, &denied, STORAGE_JOB_BYTES).is_err());
    assert!(!denied.exists());
    drop(f);
    fs::remove_file(filler).unwrap();
    storage::finish(&job, &mut receipt, false).unwrap();
    assert!(storage::finalized(&path, &receipt).unwrap());
    assert!(!path.join("job").exists());
}
