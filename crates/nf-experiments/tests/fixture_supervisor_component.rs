// Path inclusion deliberately requires no coordinator/crate-root changes.
#[path = "../src/fixture_supervisor/mod.rs"]
#[allow(dead_code)]
mod fixture_supervisor;

use fixture_supervisor::{LaunchBinding, Policy, PreparedIdentity, Receipt, Supervisor, PROFILE};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Fixture {
    directory: PathBuf,
    binary: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let directory = std::env::temp_dir().join(format!(
            "nf-t038-api-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        let binary = directory.join("worker");
        let output = Command::new("/usr/bin/cc")
            .args(["-static", "-O2", "-pthread", "-o"])
            .arg(&binary)
            .arg(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixture_supervisor_static_worker.c"
            ))
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        Self { directory, binary }
    }
    fn policy(&self, mode: &str, wall_ns: u64) -> Policy {
        let output = Command::new("/usr/bin/python3")
            .args(["-I", "-S", "-c", "import hashlib,json,secrets,sys,time; print(json.dumps([open('/proc/sys/kernel/random/boot_id').read().strip(),time.clock_gettime_ns(time.CLOCK_BOOTTIME),hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest(),secrets.token_hex(32)]))"])
            .arg(&self.binary).output().unwrap();
        assert!(output.status.success());
        let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        let deadline = value[1].as_u64().unwrap() + wall_ns;
        Policy {
            profile: PROFILE.to_owned(),
            input_hex: String::new(),
            boot_id: value[0].as_str().unwrap().to_owned(),
            deadline_boottime_ns: deadline,
            memory_bytes: 96 << 20,
            cpu_seconds: 1,
            output_bytes: 65536,
            grace_ms: 40,
            reap_ms: 200,
            executable_sha256: value[2].as_str().unwrap().to_owned(),
            argv: vec![
                self.binary.to_str().unwrap().to_owned(),
                mode.into(),
                std::process::id().to_string(),
            ],
            binding: LaunchBinding {
                contract: "contract-1".into(),
                reservation: "reservation-1".into(),
                attempt: "attempt-1".into(),
                lease: "lease-1".into(),
                fencing_epoch: 7,
                nonce: value[3].as_str().unwrap().to_owned(),
                lease_deadline_boottime_ns: deadline,
            },
        }
    }
    fn start(&self, mode: &str, wall_ns: u64) -> Supervisor {
        let mut supervisor =
            Supervisor::prepare(Path::new("/usr/bin/python3"), self.policy(mode, wall_ns)).unwrap();
        let identity = prepared(&mut supervisor);
        // Protocol test only. Durable coordinator persistence is NOT claimed.
        supervisor.authorize_after_persist(&identity).unwrap();
        supervisor
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.directory).unwrap();
    }
}
fn prepared(supervisor: &mut Supervisor) -> PreparedIdentity {
    let until = Instant::now() + Duration::from_secs(3);
    loop {
        if let Some(identity) = supervisor.poll_prepared().unwrap() {
            return identity;
        }
        assert!(Instant::now() < until, "preparation exceeded bounded wait");
        std::thread::sleep(Duration::from_millis(2));
    }
}
fn finish(supervisor: &mut Supervisor) -> Receipt {
    let until = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(receipt) = supervisor.poll().unwrap() {
            return receipt;
        }
        assert!(Instant::now() < until, "receipt exceeded bounded wait");
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn static_positive_with_real_thread_and_separate_resource_observations() {
    let fixture = Fixture::new();
    let mut supervisor = fixture.start("positive", 2_000_000_000);
    let receipt = finish(&mut supervisor);
    assert_eq!(receipt.status, "completed");
    assert_eq!(receipt.stdout_hex, "7468726561640a");
    assert!(receipt.cleanup_confirmed);
    assert!(receipt.guardian_cpu_used_ns > 0);
    assert!(receipt.guardian_peak_rss_bytes > 0);
    assert!(
        receipt
            .controller_observation
            .as_ref()
            .unwrap()
            .process_peak_rss_bytes
            > 0
    );
    assert!(receipt.accounting_scope.contains("tail_unmeasured"));
    assert!(supervisor.poll().is_err());
}

#[test]
fn prepared_identity_is_required_and_single_use() {
    let fixture = Fixture::new();
    let mut supervisor = Supervisor::prepare(
        Path::new("/usr/bin/python3"),
        fixture.policy("positive", 2_000_000_000),
    )
    .unwrap();
    let identity = prepared(&mut supervisor);
    let mut wrong = identity.clone();
    wrong.policy.binding.fencing_epoch += 1;
    assert!(supervisor.authorize_after_persist(&wrong).is_err());
    supervisor.authorize_after_persist(&identity).unwrap();
    assert!(supervisor.authorize_after_persist(&identity).is_err());
    assert_eq!(finish(&mut supervisor).status, "completed");
}

#[test]
fn legacy_api_cannot_bypass_persisted_identity() {
    let fixture = Fixture::new();
    assert!(Supervisor::start_after_durable_intent(
        Path::new("/usr/bin/python3"),
        fixture.policy("positive", 2_000_000_000)
    )
    .is_err());
}

#[test]
fn native_attacks_are_denied_by_kernel_not_worker_assertions() {
    let fixture = Fixture::new();
    let mut supervisor = fixture.start("attack", 2_000_000_000);
    let receipt = finish(&mut supervisor);
    assert_eq!(receipt.status, "completed", "{receipt:?}");
    assert_eq!(receipt.stdout_hex, "64656e6965640a");
    assert!(receipt.cleanup_confirmed);
}

#[test]
fn no_work_before_identity_authorization_and_no_deadline_reset() {
    let fixture = Fixture::new();
    let policy = fixture.policy("positive", 350_000_000);
    let absolute = policy.deadline_boottime_ns;
    let mut supervisor = Supervisor::prepare(Path::new("/usr/bin/python3"), policy).unwrap();
    prepared(&mut supervisor);
    let receipt = finish(&mut supervisor);
    assert_eq!(receipt.status, "timed_out");
    assert_eq!(receipt.deadline_boottime_ns, absolute);
    assert!(!receipt.events.contains_key("worker_started_ns"));
}

#[test]
fn cancellation_is_terminal_and_idempotent() {
    let fixture = Fixture::new();
    let mut supervisor = fixture.start("sleep", 2_000_000_000);
    std::thread::sleep(Duration::from_millis(80));
    supervisor.cancel_after_durable_stop().unwrap();
    supervisor.cancel_after_durable_stop().unwrap();
    assert_eq!(finish(&mut supervisor).status, "cancelled");
}

#[test]
fn coalesced_authorization_and_stop_never_starts_worker() {
    let fixture = Fixture::new();
    let mut supervisor = Supervisor::prepare(
        Path::new("/usr/bin/python3"),
        fixture.policy("positive", 2_000_000_000),
    )
    .unwrap();
    let identity = prepared(&mut supervisor);
    // The unreaped direct child's PID cannot be reused. Pause only the trusted
    // guardian so both protocol messages are queued before its next read.
    assert_eq!(
        unsafe { libc::kill(identity.guardian_pid as i32, libc::SIGSTOP) },
        0
    );
    supervisor.authorize_after_persist(&identity).unwrap();
    supervisor.cancel_after_durable_stop().unwrap();
    assert_eq!(
        unsafe { libc::kill(identity.guardian_pid as i32, libc::SIGCONT) },
        0
    );
    let receipt = finish(&mut supervisor);
    assert_eq!(receipt.status, "cancelled");
    assert!(receipt.cleanup_confirmed);
    assert!(!receipt.events.contains_key("worker_started_ns"));
}

#[test]
fn stop_after_guardian_exit_still_drains_real_cleanup_receipt() {
    let fixture = Fixture::new();
    let mut supervisor = fixture.start("positive", 2_000_000_000);
    std::thread::sleep(Duration::from_millis(500));
    supervisor.cancel_after_durable_stop().unwrap();
    let receipt = finish(&mut supervisor);
    assert!(receipt.cleanup_confirmed);
    assert_eq!(receipt.status, "completed");
}

#[test]
fn large_output_receipt_drain_is_bounded() {
    let fixture = Fixture::new();
    let mut supervisor = fixture.start("flood", 2_000_000_000);
    let receipt = finish(&mut supervisor);
    assert_eq!(receipt.status, "output_exceeded");
    assert_eq!(receipt.stdout_hex.len(), 131072);
    assert!(receipt.cleanup_confirmed);
}

#[test]
fn dynamic_python_and_bad_digest_fail_qualification_not_fallback() {
    let fixture = Fixture::new();
    for python in [true, false] {
        let mut policy = fixture.policy("positive", 2_000_000_000);
        if python {
            policy.argv = vec!["/usr/bin/python3".into()];
        } else {
            policy.executable_sha256 = "0".repeat(64);
        }
        let mut supervisor = Supervisor::prepare(Path::new("/usr/bin/python3"), policy).unwrap();
        let until = Instant::now() + Duration::from_secs(3);
        loop {
            match supervisor.poll_prepared() {
                Err(_) => break,
                Ok(Some(_)) => panic!("unsafe runtime was qualified"),
                Ok(None) => (),
            }
            assert!(Instant::now() < until);
            std::thread::sleep(Duration::from_millis(2));
        }
    }
}
