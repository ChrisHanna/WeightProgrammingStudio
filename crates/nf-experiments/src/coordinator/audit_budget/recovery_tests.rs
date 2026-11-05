//! Real two-root OCI recovery interleavings, not simulated root identifiers.
use super::tests::Harness;
use super::*;
use crate::sandbox::{Limits, QualifiedDocker, STORAGE_JOB_BYTES};
use std::{
    cell::Cell,
    fs,
    os::unix::fs::PermissionsExt,
    sync::{atomic::AtomicBool, Barrier},
    time::{Duration, Instant},
};

fn second_root(h: &Harness) -> QualifiedDocker {
    let root = h.root.join("other-sandbox");
    fs::create_dir(&root).unwrap();
    fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
    QualifiedDocker::qualify(true, &root).unwrap()
}

#[test]
fn binding_wins_recovery_race_only_exact_root_cleanup_can_release() {
    if enforcement_tests::bounded_controller_test() {
        return;
    }
    let mut h = Harness::new();
    h.claim();
    let runtime_a = QualifiedDocker::qualify(true, &h.root).unwrap();
    let runtime_b = second_root(&h);
    let begin_bind = Barrier::new(2);
    let active = Barrier::new(2);
    let release_callback = Barrier::new(2);
    let root = h.root.clone();
    let lease = h.lease.clone();
    let reservation = h.reservation.clone();
    let budget = h.budget.clone();
    std::thread::scope(|scope| {
        let worker = scope.spawn(|| {
            let mut other = enforcement_tests::open_component(&root.join("ledger"));
            begin_bind.wait();
            other
                .bind_audit_runtime(&lease, &reservation, &runtime_a)
                .unwrap();
            let observed = Cell::new(false);
            let authorize = || {
                // Stop the real execution at a known active-container boundary,
                // retaining root A's OS lock while recovery examines root B.
                if !observed.get() {
                    for entry in fs::read_dir(&root).unwrap().filter_map(|e| e.ok()) {
                        if !entry.path().join("configured.json").exists()
                            || entry.path().join("completion.json").exists()
                        {
                            continue;
                        }
                        if QualifiedDocker::container_running(entry.file_name().to_str().unwrap())
                            .unwrap_or(false)
                        {
                            observed.set(true);
                            active.wait();
                            release_callback.wait();
                            break;
                        }
                    }
                }
                other.check_audit(&lease, &reservation, &budget)
            };
            let result = runtime_a.execute_assets_authorized(
                b"import time\ntime.sleep(30)",
                b"{}",
                &BTreeMap::new(),
                Limits {
                    wall_ms: 5000,
                    cpu_ms: 2000,
                    output_bytes: 4096,
                },
                STORAGE_JOB_BYTES,
                &AtomicBool::new(false),
                Instant::now() + Duration::from_secs(12),
                &authorize,
            );
            assert!(
                observed.get(),
                "race must reach a genuinely running container"
            );
            assert!(
                result.is_err(),
                "committed recovery stop must revoke execution"
            );
        });
        let result = h
            .ledger
            .recover_audit_fenced(&h.lease, &h.reservation, &runtime_b, || {
                // Recovery has begun with NO binding. The binder now wins before
                // stop: a pre-stop binding snapshot would incorrectly sweep B.
                begin_bind.wait();
                active.wait();
            });
        assert!(result.unwrap_err().message.contains("exact attested"));
        assert!(h.ledger.stop_record(&h.lease.contract).unwrap().is_some());
        assert_eq!(
            h.ledger
                .budget_snapshot(&h.lease.contract)
                .unwrap()
                .held
                .workers,
            1
        );
        assert_eq!(
            h.ledger
                .connection
                .query_row("SELECT count(*) FROM audit_cleanup", [], |r| r
                    .get::<_, u64>(0))
                .unwrap(),
            0
        );
        // Even the correct identity cannot release before its lock is free.
        assert!(h
            .ledger
            .recover_reference_audit(&h.lease, &h.reservation, &runtime_a)
            .is_err());
        assert_eq!(
            h.ledger
                .budget_snapshot(&h.lease.contract)
                .unwrap()
                .held
                .workers,
            1
        );
        release_callback.wait();
        worker.join().unwrap();
    });
    h.ledger
        .recover_reference_audit(&h.lease, &h.reservation, &runtime_a)
        .unwrap();
    let after = h.ledger.budget_snapshot(&h.lease.contract).unwrap();
    assert_eq!(after.held.workers, 0);
    assert_eq!(after.spent.storage_bytes, h.budget.storage_bytes);
    assert_eq!(after.spent.cpu_ms, h.request.resources.cpu_ms);
    assert_eq!(after.unknown_charged.storage_bytes, h.budget.storage_bytes);
    h.reopen();
    h.ledger
        .recover_reference_audit(&h.lease, &h.reservation, &runtime_a)
        .unwrap();
    assert_eq!(
        h.ledger.budget_snapshot(&h.lease.contract).unwrap().spent,
        after.spent
    );
    for entry in fs::read_dir(&h.root).unwrap().filter_map(|e| e.ok()) {
        if !entry
            .file_name()
            .to_string_lossy()
            .starts_with("nf-sandbox-")
        {
            continue;
        }
        assert!(!entry.path().join("job").exists());
        assert!(!entry.path().join("guard").exists());
    }
}

#[test]
fn stop_wins_binding_race_unbound_recovery_never_sweeps_supplied_root() {
    let mut h = Harness::new();
    h.claim();
    let runtime = second_root(&h);
    let clean = runtime
        .recover_attested(&runtime.durable_root_identity().unwrap())
        .unwrap();
    // Holding this unrelated root lock proves unbound recovery never sweeps it.
    h.ledger
        .recover_reference_audit(&h.lease, &h.reservation, &runtime)
        .unwrap();
    drop(clean);
    assert!(h
        .ledger
        .bind_audit_runtime(&h.lease, &h.reservation, &runtime)
        .is_err());
    assert_eq!(
        h.ledger
            .budget_snapshot(&h.lease.contract)
            .unwrap()
            .held
            .workers,
        0
    );
    assert!(h
        .ledger
        .connection
        .query_row(
            "SELECT root FROM audit_runtime_bindings WHERE request=?1",
            [h.reservation.id().as_str()],
            |r| r.get::<_, String>(0)
        )
        .optional()
        .unwrap()
        .is_none());
}

#[test]
fn unlimited_controller_denied_before_claim_or_qualifier() {
    // Do not mistake a small current RSS for an irreversible kernel bound.
    if rustix::process::getrlimit(rustix::process::Resource::As)
        .maximum
        .is_some()
    {
        return;
    }
    let mut h = Harness::new();
    assert!(memory::qualify_controller(AUDIT_HOST_BYTES).is_err());
    assert!(!claimed(&h.ledger.connection, h.reservation.id()).unwrap());
    // No public execution inputs are needed to exercise this pre-claim check.
    assert!(memory::qualify_controller(crate::sandbox::MEMORY_BYTES).is_err());
    h.claim();
    assert_eq!(h.held(), h.budget.storage_bytes);
}

#[test]
fn kernel_controller_ceiling_cannot_be_raised_and_denies_mapping() {
    if enforcement_tests::bounded_controller_test() {
        return;
    }
    let hard = memory::qualify_controller(AUDIT_HOST_BYTES).unwrap();
    assert_eq!(hard, crate::sandbox::CONTROLLER_MEMORY_ALLOWANCE_BYTES);
    assert!(memory::qualify_controller(AUDIT_HOST_BYTES - 1).is_err());
    let raised = libc::rlimit {
        rlim_cur: hard + 4096,
        rlim_max: hard + 4096,
    };
    // SAFETY: valid limit pointer and anonymous mapping, never dereferenced.
    unsafe {
        assert_ne!(libc::setrlimit(libc::RLIMIT_AS, &raised), 0);
        let mapping = libc::mmap(
            std::ptr::null_mut(),
            hard as usize,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        );
        assert_eq!(mapping, libc::MAP_FAILED);
    }
}
