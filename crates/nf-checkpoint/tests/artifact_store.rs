#![cfg(target_os = "linux")]

use nf_checkpoint::store::{ArtifactStore, StoreLimits};
use nf_core::error::ErrorCode;
use std::fs::{self, File};
use std::io::{self, Cursor, Read};
use std::os::unix::fs::{symlink, PermissionsExt};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct TestRoot(PathBuf);
impl TestRoot {
    fn new() -> Self {
        let parent = std::env::var_os("KIROCREW_SCRATCH")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        let path = parent.join(format!(
            "artifact-tests-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn store(&self, quota: u64) -> ArtifactStore {
        ArtifactStore::create(
            &self.0.join("store"),
            StoreLimits {
                total_bytes: quota,
                max_blob_bytes: quota,
                max_entries: 100,
            },
        )
        .unwrap()
    }
}
impl Drop for TestRoot {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn artifact_store_verified_publication_and_deduplication() {
    let root = TestRoot::new();
    let store = root.store(1024);
    let first = store.put(Cursor::new(b"immutable bytes")).unwrap();
    assert!(!first.deduplicated);
    assert_eq!(store.read(&first.digest).unwrap(), b"immutable bytes");
    let again = store.put(Cursor::new(b"immutable bytes")).unwrap();
    assert!(again.deduplicated);
    assert_eq!(first.digest, again.digest);
    assert_eq!(store.used_bytes().unwrap(), 15);
    assert_eq!(
        fs::read_dir(root.0.join("store/staging")).unwrap().count(),
        0
    );
    let path = root.0.join("store/blobs").join(&first.digest.as_str()[7..]);
    assert_eq!(fs::metadata(path).unwrap().permissions().mode() & 0o222, 0);
}

#[test]
fn artifact_store_detects_corruption_and_refuses_overwrite() {
    let root = TestRoot::new();
    let store = root.store(1024);
    let original = store.put(Cursor::new(b"valid")).unwrap();
    let path = root
        .0
        .join("store/blobs")
        .join(&original.digest.as_str()[7..]);
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
    fs::write(&path, b"wrong").unwrap();
    assert_eq!(
        store.read(&original.digest).unwrap_err().code,
        ErrorCode::IntegrityMismatch
    );
    assert!(store.put(Cursor::new(b"valid")).is_err());
    assert_eq!(fs::read(path).unwrap(), b"wrong");
}

#[test]
fn artifact_store_quota_and_entry_limits_cleanup_partial_data() {
    let root = TestRoot::new();
    let store = root.store(5);
    assert_eq!(
        store.put(Cursor::new(b"too large")).unwrap_err().code,
        ErrorCode::QuotaExceeded
    );
    assert_eq!(store.used_bytes().unwrap(), 0);
    let one = ArtifactStore::create(
        &root.0.join("one-entry"),
        StoreLimits {
            total_bytes: 100,
            max_blob_bytes: 100,
            max_entries: 1,
        },
    )
    .unwrap();
    one.put(Cursor::new(b"a")).unwrap();
    assert_eq!(
        one.put(Cursor::new(b"b")).unwrap_err().code,
        ErrorCode::LimitExceeded
    );
}

struct FailingRead(bool);
impl Read for FailingRead {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.0 {
            return Err(io::Error::other("injected interrupted source"));
        }
        self.0 = true;
        out[0] = 42;
        Ok(1)
    }
}

#[test]
fn artifact_store_failed_source_never_publishes() {
    let root = TestRoot::new();
    let store = root.store(1024);
    assert!(store.put(FailingRead(false)).is_err());
    assert_eq!(store.used_bytes().unwrap(), 0);
    assert_eq!(fs::read_dir(root.0.join("store/blobs")).unwrap().count(), 0);
    assert_eq!(
        fs::read_dir(root.0.join("store/staging")).unwrap().count(),
        0
    );
}

#[test]
fn artifact_store_rejects_root_traversal_and_symlink_escape() {
    let root = TestRoot::new();
    let outside = root.0.join("outside");
    fs::create_dir(&outside).unwrap();
    symlink(&outside, root.0.join("link")).unwrap();
    let limits = StoreLimits {
        total_bytes: 100,
        max_blob_bytes: 100,
        max_entries: 10,
    };
    assert!(ArtifactStore::create(&root.0.join("link/nested"), limits).is_err());
    assert!(!outside.join("nested").exists());
    assert!(ArtifactStore::create(&root.0.join("new/../escape"), limits).is_err());
    assert!(!root.0.join("new").exists());
    assert!(ArtifactStore::create(std::path::Path::new("relative"), limits).is_err());
}

#[test]
fn artifact_store_rejects_internal_symlinks_and_hardlinks() {
    let root = TestRoot::new();
    let store = root.store(1024);
    let receipt = store.put(Cursor::new(b"data")).unwrap();
    let path = root
        .0
        .join("store/blobs")
        .join(&receipt.digest.as_str()[7..]);
    fs::hard_link(&path, root.0.join("alias")).unwrap();
    assert_eq!(
        store.read(&receipt.digest).unwrap_err().code,
        ErrorCode::IntegrityMismatch
    );
    fs::remove_file(root.0.join("alias")).unwrap();
    fs::remove_file(&path).unwrap();
    fs::write(root.0.join("external"), b"data").unwrap();
    symlink(root.0.join("external"), &path).unwrap();
    assert!(store.read(&receipt.digest).is_err());
    assert!(store.put(Cursor::new(b"new data")).is_err());
}

#[test]
fn artifact_store_snapshot_does_not_alias_mutable_source() {
    let root = TestRoot::new();
    fs::write(root.0.join("source"), b"original").unwrap();
    let store = root.store(1024);
    let receipt = store
        .put(File::open(root.0.join("source")).unwrap())
        .unwrap();
    fs::write(root.0.join("source"), b"changed").unwrap();
    assert_eq!(store.read(&receipt.digest).unwrap(), b"original");
}

#[test]
fn artifact_store_quota_is_serialized_across_store_handles() {
    let root = TestRoot::new();
    let left = root.store(100);
    let right = root.store(100);
    std::thread::scope(|scope| {
        let a = scope.spawn(|| left.put(Cursor::new([1u8; 60])));
        let b = scope.spawn(|| right.put(Cursor::new([2u8; 60])));
        let outcomes = [a.join().unwrap(), b.join().unwrap()];
        assert_eq!(outcomes.iter().filter(|value| value.is_ok()).count(), 1);
        assert_eq!(
            outcomes
                .iter()
                .filter(
                    |value| matches!(value, Err(error) if error.code == ErrorCode::QuotaExceeded)
                )
                .count(),
            1
        );
    });
    assert_eq!(left.used_bytes().unwrap(), 60);
}

#[test]
fn artifact_store_orphan_staging_remains_accounted_and_unpublished() {
    let root = TestRoot::new();
    let store = root.store(10);
    fs::write(root.0.join("store/staging/orphan"), b"12345").unwrap();
    assert_eq!(store.used_bytes().unwrap(), 5);
    assert_eq!(
        store.put(Cursor::new(b"123456")).unwrap_err().code,
        ErrorCode::QuotaExceeded
    );
    assert_eq!(store.used_bytes().unwrap(), 5);
}
