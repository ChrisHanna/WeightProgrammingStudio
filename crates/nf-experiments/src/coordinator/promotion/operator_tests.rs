use super::*;
use std::os::unix::fs::PermissionsExt;
static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
struct Registry(PathBuf);
impl Registry {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "nf-operator-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        test_registry(&path);
        Self(path)
    }
}
impl Drop for Registry {
    fn drop(&mut self) {
        TEST_ROOT.with(|v| *v.borrow_mut() = None);
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}
fn request() -> Value {
    json!({"schema_version":1,"scope":"operational_fixture","store":"store-A",
        "invocation":{"contract":"frozen-A","epoch":4,"lease_deadline":100,
            "operation":"promotion-A","expected_head":null,"edit":"edit-A","smoke":"smoke-A","runtime":"runtime-A"}})
}
fn isolated_registry_test() -> bool {
    if std::env::var_os("NF_OPERATOR_TEST_ISOLATED").is_some() {
        return false;
    }
    let name = std::thread::current().name().unwrap().to_owned();
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", &name, "--test-threads=1"])
        .env("NF_OPERATOR_TEST_ISOLATED", "1")
        .status()
        .unwrap();
    assert!(status.success(), "isolated operator test failed: {status}");
    true
}
#[test]
fn issuance_resolution_binding_revocation_and_cross_process_lock() {
    if isolated_registry_test() { return; }
    let registry = Registry::new();
    let request = request();
    let id = issue(&request, test_expiration()).unwrap();
    let guard = verify(&id, &request).unwrap();
    let descriptor_flags = unsafe { libc::fcntl(guard._file.as_raw_fd(), libc::F_GETFD) };
    assert!(descriptor_flags >= 0);
    assert_ne!(descriptor_flags & libc::FD_CLOEXEC, 0);
    let file = registry.0.join(sidecar(&id, "lock"));
    let status = std::process::Command::new("/usr/bin/flock")
        .args(["-xn"])
        .arg(&file)
        .args(["-c", "true"])
        .status()
        .unwrap();
    assert_eq!(
        status.code(),
        Some(1),
        "independent process must not acquire exclusive lock"
    );
    assert!(revoke(&id).is_err(), "busy is NOT a successful revocation");
    assert!(verify(&id, &request).is_ok());
    for pointer in [
        "/store",
        "/invocation/contract",
        "/invocation/epoch",
        "/invocation/lease_deadline",
        "/invocation/operation",
        "/invocation/expected_head",
        "/invocation/edit",
        "/invocation/smoke",
        "/invocation/runtime",
    ] {
        let mut wrong = request.clone();
        *wrong.pointer_mut(pointer).unwrap() = json!("other");
        assert!(verify(&id, &wrong).is_err(), "{pointer}");
    }
    let fake = ContentDigest::of_bytes(b"valid-looking digest or agent text");
    assert!(verify(&fake, &request).is_err());
    drop(guard);
    assert!(std::process::Command::new("/usr/bin/flock")
        .args(["-xn"])
        .arg(&file)
        .args(["-c", "true"])
        .status()
        .unwrap()
        .success());
    revoke(&id).unwrap();
    assert!(verify(&id, &request).is_err());
}
#[test]
fn expiration_is_live_and_not_renewable_by_reissuing_same_identifier() {
    if isolated_registry_test() { return; }
    let _registry = Registry::new();
    let request = request();
    let expires = now().unwrap() + 100;
    let id = issue(&request, expires).unwrap();
    assert!(verify(&id, &request).is_ok());
    std::thread::sleep(std::time::Duration::from_millis(120));
    assert!(verify(&id, &request).is_err());
    assert!(issue(&request, expires).is_err());
    let new_id = issue(&request, test_expiration()).unwrap();
    assert_ne!(id, new_id);
    assert!(verify(&id, &request).is_err());
    assert!(verify(&new_id, &request).is_ok());
}
#[test]
fn unsafe_modes_links_owners_and_symlinked_registry_are_denied() {
    if isolated_registry_test() { return; }
    let registry = Registry::new();
    let request = request();
    let id = issue(&request, test_expiration()).unwrap();
    let path = registry.0.join(name(&id));
    let extra = registry.0.join("extra");
    let file = File::open(&path).unwrap();
    assert!(checked(&file, location().1 + 1, false).is_err());
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o666)).unwrap();
    assert!(verify(&id, &request).is_err());
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
    std::fs::hard_link(&path, &extra).unwrap();
    assert!(verify(&id, &request).is_err());
    std::fs::remove_file(&extra).unwrap();
    assert!(verify(&id, &request).is_ok());
    std::fs::rename(&path, &extra).unwrap();
    std::os::unix::fs::symlink(&extra, &path).unwrap();
    assert!(verify(&id, &request).is_err());
    std::fs::remove_file(&path).unwrap();
    std::fs::rename(&extra, &path).unwrap();
    std::fs::set_permissions(&registry.0, std::fs::Permissions::from_mode(0o777)).unwrap();
    assert!(verify(&id, &request).is_err());
    std::fs::set_permissions(&registry.0, std::fs::Permissions::from_mode(0o700)).unwrap();
    let alias = registry.0.with_extension("alias");
    std::os::unix::fs::symlink(&registry.0, &alias).unwrap();
    TEST_ROOT.with(|v| *v.borrow_mut() = Some(alias.clone()));
    assert!(verify(&id, &request).is_err());
    std::fs::remove_file(alias).unwrap();
}
#[test]
fn crashed_consumer_releases_os_lock_without_destructor_or_fake_revocation() {
    if isolated_registry_test() { return; }
    let registry = Registry::new();
    let request = request();
    let id = issue(&request, test_expiration()).unwrap();
    let mut child=std::process::Command::new("/usr/bin/python3.12").args(["-I","-S","-c",
        "import fcntl,os,signal,sys; f=open(sys.argv[1]); fcntl.flock(f,fcntl.LOCK_SH); os.kill(os.getpid(),signal.SIGSTOP)"])
        .arg(registry.0.join(sidecar(&id,"lock"))).spawn().unwrap();
    let until = std::time::Instant::now() + std::time::Duration::from_secs(2);
    let mut status = 0;
    loop {
        let observed = unsafe {
            libc::waitpid(
                child.id() as i32,
                &mut status,
                libc::WNOHANG | libc::WUNTRACED,
            )
        };
        if observed == child.id() as i32 {
            break;
        }
        if observed < 0 || std::time::Instant::now() >= until {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("lock holder did not stop");
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let busy = revoke(&id).is_err();
    child.kill().unwrap();
    child.wait().unwrap();
    assert!(libc::WIFSTOPPED(status));
    assert!(busy);
    revoke(&id).unwrap();
    assert!(verify(&id, &request).is_err());
}
#[test]
fn same_uid_production_operator_cannot_mint_and_no_environment_override_exists() {
    TEST_ROOT.with(|v| *v.borrow_mut() = None);
    assert_eq!(location().0, PathBuf::from("/etc/nf-promotion"));
    if unsafe { libc::geteuid() } != 0 {
        assert!(operator().is_err());
    }
    // Never invoke production issue/revoke or write actual /etc in this suite.
    let error = match consume(&ContentDigest::of_bytes(b"not authority"), &request()) {
        Ok(_) => panic!("production must not substitute a local hash for monotonic authority"),
        Err(error) => error,
    };
    assert_eq!(error.code, ErrorCode::UnsupportedCapability);
    assert!(error
        .message
        .contains("privileged monotonic consumption publisher"));
}
const WRITE_POINTS: [&str; 9] = [
    "before_write",
    "partial_write",
    "after_write",
    "before_file_fsync",
    "after_file_fsync",
    "before_rename",
    "after_rename",
    "before_directory_fsync",
    "after_directory_fsync",
];
#[test]
fn every_writer_fault_preserves_immutable_provenance_and_burns_published_consumption() {
    if isolated_registry_test() { return; }
    for point in WRITE_POINTS {
        for action in ["issue", "revoke", "consume"] {
            let registry = Registry::new();
            let request = request();
            let expires = test_expiration();
            let id = hash(&json!({"request":request,"expires_unix_ms":expires})).unwrap();
            if action != "issue" {
                assert_eq!(issue(&request, expires).unwrap(), id);
            }
            let before = std::fs::read(registry.0.join(name(&id))).ok();
            FAULT.with(|v| *v.borrow_mut() = Some(point));
            let result = match action {
                "issue" => issue(&request, expires).map(|_| ()),
                "revoke" => revoke(&id),
                _ => consume(&id, &request).map(|_| ()),
            };
            FAULT.with(|v| *v.borrow_mut() = None);
            assert!(result.is_err(), "{action}/{point}");
            if action == "issue" {
                if registry.0.join(name(&id)).exists() {
                    audit(&id, &request).unwrap();
                } else {
                    issue(&request, expires).unwrap();
                }
            } else {
                assert_eq!(
                    std::fs::read(registry.0.join(name(&id))).unwrap(),
                    before.unwrap()
                );
                audit(&id, &request).unwrap();
                if action == "revoke" {
                    revoke(&id).unwrap();
                    assert!(verify(&id, &request).is_err());
                    audit(&id, &request).unwrap();
                } else if registry.0.join(sidecar(&id, "consumed")).exists() {
                    assert!(
                        consume(&id, &request).is_err(),
                        "published consumption cannot be retried"
                    );
                    audit_consumed(&id).unwrap();
                } else {
                    drop(consume(&id, &request).unwrap());
                }
            }
        }
    }
    for point in [
        "before_lock_fsync",
        "after_lock_fsync",
        "before_lock_directory_fsync",
        "after_lock_directory_fsync",
    ] {
        let registry = Registry::new();
        let request = request();
        let expires = test_expiration();
        FAULT.with(|v| *v.borrow_mut() = Some(point));
        assert!(issue(&request, expires).is_err());
        FAULT.with(|v| *v.borrow_mut() = None);
        assert_eq!(
            std::fs::read_dir(&registry.0).unwrap().count(),
            1,
            "only stable lock may remain"
        );
        let id = issue(&request, expires).unwrap();
        audit(&id, &request).unwrap();
    }
}
#[test]
fn consumption_is_external_one_shot_and_revocation_does_not_destroy_history() {
    if isolated_registry_test() { return; }
    let registry = Registry::new();
    let request = request();
    let id = issue(&request, test_expiration()).unwrap();
    assert!(audit_consumed(&id).is_err());
    let guard = consume(&id, &request).unwrap();
    assert!(consume(&id, &request).is_err());
    assert!(revoke(&id).is_err());
    drop(guard);
    audit_consumed(&id).unwrap();
    assert!(consume(&id, &request).is_err());
    let original = std::fs::read(registry.0.join(name(&id))).unwrap();
    revoke(&id).unwrap();
    audit(&id, &request).unwrap();
    audit_consumed(&id).unwrap();
    assert_eq!(original, std::fs::read(registry.0.join(name(&id))).unwrap());
    assert!(verify(&id, &request).is_err());
}
#[test]
fn calling_thread_status_is_not_the_leader_and_privilege_fields_fail_closed() {
    let clean="Uid:\t1000\t1000\t1000\t1000\nCapInh:\t0000000000000000\nCapPrm:\t0000000000000000\nCapEff:\t0000000000000000\nCapAmb:\t0000000000000000\n";
    for ids in [
        "0 1000 1000 1000",
        "1000 0 1000 1000",
        "1000 1000 0 1000",
        "1000 1000 1000 0",
        "invalid",
        "1000 1000 1000",
    ] {
        assert!(unprivileged_status(&clean.replace("1000\t1000\t1000\t1000", ids)).is_err());
    }
    for field in ["CapInh:", "CapPrm:", "CapEff:", "CapAmb:"] {
        assert!(unprivileged_status(clean).is_ok());
        assert!(unprivileged_status(&clean.replace(
            &format!("{field}\t0000000000000000"),
            &format!("{field}\t0000000000000001")
        ))
        .is_err());
        assert!(unprivileged_status(&clean.replace(
            &format!("{field}\t0000000000000000"),
            &format!("{field}\tinvalid")
        ))
        .is_err());
        assert!(
            unprivileged_status(&clean.replace(&format!("{field}\t0000000000000000\n"), ""))
                .is_err()
        );
    }
    std::thread::spawn(|| {
        let caller=std::fs::read_to_string("/proc/thread-self/status").unwrap();
        let leader=std::fs::read_to_string("/proc/self/status").unwrap();
        let tid=unsafe {libc::syscall(libc::SYS_gettid)};
        assert!(caller.lines().any(|line|line==format!("Pid:\t{tid}")));
        assert!(leader.lines().any(|line|line==format!("Pid:\t{}",std::process::id())));
        assert_ne!(tid,std::process::id() as libc::c_long);
        let has_caps=unprivileged_status(&caller).is_err();
        TEST_ROOT.with(|v|*v.borrow_mut()=None);
        if has_caps { assert!(consumer().is_err()); }
        else if unsafe {libc::getuid()}!=0 && unsafe {libc::geteuid()}!=0 { consumer().unwrap(); }
        if !has_caps { eprintln!("per-thread capability-bearing live reproduction unavailable: no inherited capabilities; no privilege escalation attempted"); }
    }).join().unwrap();
}
#[test]
fn concurrent_issuers_and_consumers_use_one_stable_identity() {
    if isolated_registry_test() { return; }
    let registry = Registry::new();
    let request = request();
    let expires = test_expiration();
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
    let run = |consume_only: bool| {
        let mut threads = Vec::new();
        for _ in 0..2 {
            let path = registry.0.clone();
            let request = request.clone();
            let barrier = barrier.clone();
            threads.push(std::thread::spawn(move || {
                test_registry(&path);
                barrier.wait();
                if consume_only {
                    let id = hash(&json!({"request":request,"expires_unix_ms":expires})).unwrap();
                    consume(&id, &request).is_ok()
                } else {
                    issue(&request, expires).is_ok()
                }
            }));
        }
        threads
            .into_iter()
            .map(|t| usize::from(t.join().unwrap()))
            .sum::<usize>()
    };
    assert_eq!(run(false), 1);
    assert_eq!(run(true), 1);
    let id = hash(&json!({"request":request,"expires_unix_ms":expires})).unwrap();
    audit(&id, &request).unwrap();
    audit_consumed(&id).unwrap();
}
#[test]
fn issuer_and_consumer_race_cannot_admit_partial_authorization() {
    if isolated_registry_test() { return; }
    let registry = Registry::new();
    let request = request();
    let expires = test_expiration();
    let id = hash(&json!({"request":request,"expires_unix_ms":expires})).unwrap();
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
    let path = registry.0.clone();
    let other_request = request.clone();
    let other_id = id.clone();
    let other_barrier = barrier.clone();
    let child = std::thread::spawn(move || {
        test_registry(&path);
        other_barrier.wait();
        let until = std::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            match consume(&other_id, &other_request) {
                Ok(guard) => {
                    drop(guard);
                    break;
                }
                Err(error) => {
                    assert!(
                        std::time::Instant::now() < until,
                        "issuer/consumer race: {error}"
                    );
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }
        }
        audit(&other_id, &other_request).unwrap();
        audit_consumed(&other_id).unwrap();
    });
    barrier.wait();
    let until = std::time::Instant::now() + std::time::Duration::from_secs(2);
    loop {
        match issue(&request, expires) {
            Ok(issued) => {
                assert_eq!(issued, id);
                break;
            }
            Err(error) => {
                assert!(std::time::Instant::now() < until, "racing issuer: {error}");
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }
    child.join().unwrap();
    assert!(consume(&id, &request).is_err());
}
#[test]
fn actual_writer_crashes_leave_only_complete_authority_and_readable_history() {
    if isolated_registry_test() { return; }
    if let Ok(path) = std::env::var("NF_OPERATOR_CRASH_ROOT") {
        test_registry(std::path::Path::new(&path));
        let point = WRITE_POINTS
            .into_iter()
            .find(|p| Some(*p) == std::env::var("NF_OPERATOR_CRASH_POINT").ok().as_deref())
            .unwrap();
        let expires = std::env::var("NF_OPERATOR_CRASH_EXPIRES")
            .unwrap()
            .parse::<u64>()
            .unwrap();
        let id = std::env::var("NF_OPERATOR_CRASH_ID")
            .ok()
            .map(|id| id.parse::<ContentDigest>().unwrap())
            .unwrap_or_else(|| {
                hash(&json!({"request":request(),"expires_unix_ms":expires})).unwrap()
            });
        FAULT.with(|v| *v.borrow_mut() = Some(point));
        FAULT_CRASH.with(|v| v.set(true));
        match std::env::var("NF_OPERATOR_CRASH_ACTION").unwrap().as_str() {
            "issue" => {
                issue(&request(), expires).unwrap();
            }
            "revoke" => revoke(&id).unwrap(),
            "consume" => {
                drop(consume(&id, &request()).unwrap());
            }
            _ => panic!("unknown test action"),
        }
        panic!("writer did not reach crash point");
    }
    for action in ["issue", "revoke", "consume"] {
        for point in WRITE_POINTS {
            let registry = Registry::new();
            let request = request();
            let expires = test_expiration();
            let id = hash(&json!({"request":request,"expires_unix_ms":expires})).unwrap();
            if action != "issue" {
                issue(&request, expires).unwrap();
            }
            let original = std::fs::read(registry.0.join(name(&id))).ok();
            let status=std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact","coordinator::promotion::operator::tests::actual_writer_crashes_leave_only_complete_authority_and_readable_history","--test-threads=1"])
                .env("NF_OPERATOR_CRASH_ROOT",&registry.0).env("NF_OPERATOR_CRASH_ACTION",action)
                .env("NF_OPERATOR_CRASH_POINT",point).env("NF_OPERATOR_CRASH_EXPIRES",expires.to_string())
                .status().unwrap();
            assert_eq!(status.code(), Some(86), "{action}/{point}");
            if let Some(original) = original {
                assert_eq!(std::fs::read(registry.0.join(name(&id))).unwrap(), original);
                audit(&id, &request).unwrap();
                if action == "revoke" {
                    revoke(&id).unwrap();
                    assert!(verify(&id, &request).is_err());
                    audit(&id, &request).unwrap();
                } else if registry.0.join(sidecar(&id, "consumed")).exists() {
                    assert!(consume(&id, &request).is_err());
                    audit_consumed(&id).unwrap();
                } else {
                    drop(consume(&id, &request).unwrap());
                }
            } else if registry.0.join(name(&id)).exists() {
                audit(&id, &request).unwrap();
            } else {
                issue(&request, expires).unwrap();
            }
        }
    }
}
