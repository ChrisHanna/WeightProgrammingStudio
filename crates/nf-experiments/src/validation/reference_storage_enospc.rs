//! Real ENOSPC in a separate networkless fixture tmpfs, never the host disk.
use super::*;
use nf_checkpoint::store::StoreLimits;
use std::{fs, io::Write, os::unix::fs::PermissionsExt, process::Command};

struct Scratch(std::path::PathBuf);
impl Scratch {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("nf-reference-enospc-{}", std::process::id()));
        fs::create_dir(&path).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn cas_enospc_cleanup_recovery_keeps_full_charge() {
    let scratch = Scratch::new();
    if std::env::var_os("NF_REFERENCE_ENOSPC_CHILD").is_none() {
        // Pin the running image: a concurrent Cargo build may replace its path.
        let binary = scratch.0.join("fixture-tests");
        fs::copy("/proc/self/exe", &binary).unwrap();
        let output = Command::new("/usr/bin/docker")
            .args(["--host", "unix:///var/run/docker.sock", "run", "--rm", "--pull=never",
                "--network=none", "--read-only", "--cap-drop=ALL",
                "--tmpfs", "/tmp:rw,nosuid,nodev,size=33554432",
                "--env", "NF_REFERENCE_ENOSPC_CHILD=1", "--mount",
                &format!("type=bind,src={},dst=/fixture-tests,readonly", binary.display()),
                "--entrypoint", "/fixture-tests", sandbox::IMAGE, "--exact",
                "validation::reference::storage::enospc::cas_enospc_cleanup_recovery_keeps_full_charge",
                "--nocapture"])
            .output().unwrap();
        assert!(
            output.status.success(),
            "real ENOSPC fixture unavailable/failed: {} {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    let store = ArtifactStore::create(
        &scratch.0.join("store"),
        StoreLimits {
            total_bytes: 8 * 1024 * 1024,
            max_blob_bytes: 1048576,
            max_entries: 100,
        },
    )
    .unwrap();
    let mut authority =
        crate::coordinator::budgets::audit_budget::tests::Harness::at(scratch.0.join("authority"));
    authority.claim();
    let held = authority.held();
    let mut accounting = Accounting::admit(2, 64 * 1024 * 1024).unwrap();
    let charge = accounting.charged_bytes;
    let filler_path = scratch.0.join("external-writer");
    let mut filler = fs::File::create(&filler_path).unwrap();
    loop {
        if let Err(e) = filler.write_all(&[0; 4096]) {
            assert_eq!(e.raw_os_error(), Some(libc::ENOSPC));
            break;
        }
    }
    // Leave room to create the CAS staging entry, but not its payload. The
    // failed production put must remove partial staging and keep its charge.
    let len = filler.metadata().unwrap().len();
    filler.set_len(len - 8192).unwrap();
    let failure = accounting.put(&store, &[7; 65536]).unwrap_err();
    assert_eq!(failure.code, ErrorCode::Io);
    assert!(
        failure.message.contains("28"),
        "expected actual ENOSPC: {failure:?}"
    );
    assert_eq!(store.used_bytes().unwrap(), 0);
    assert_eq!(
        fs::read_dir(scratch.0.join("store/staging"))
            .unwrap()
            .count(),
        0
    );
    assert_eq!(accounting.publications_claimed, 1);
    assert_eq!(accounting.charged_bytes, charge);
    assert_eq!(accounting.measured_study_storage_bytes, None);
    assert!(!accounting.capacity_guaranteed);
    drop(filler);
    fs::remove_file(filler_path).unwrap();
    // Actual disk capacity restored; same accounting object, no budget reset.
    let id = accounting.put(&store, &[7; 65536]).unwrap();
    assert_eq!(store.read(&id).unwrap(), [7; 65536]);
    assert_eq!(store.used_bytes().unwrap(), 65536);
    assert_eq!(accounting.publications_claimed, 2);
    assert_eq!(accounting.charged_bytes, charge);
    // Capacity loss cannot erase a committed family hold. Reopen the real
    // ledger after capacity returns: same-attempt redispatch remains denied.
    authority.reopen();
    assert_eq!(authority.held(), held);
    let retry = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| authority.claim()));
    assert!(retry.is_err());
    assert_eq!(authority.held(), held);
}
