use super::*;
use crate::store::StoreLimits;
use std::sync::atomic::{AtomicU64, Ordering};

#[test]
fn sharded_io_equal_metadata_stamps_cannot_hide_reused_source_changes() {
    let scratch = std::env::var_os("KIROCREW_SCRATCH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let path = scratch.join(format!(
        "shard-same-stamp-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&path).unwrap();
    let header = br#"{"x":{"dtype":"U8","shape":[8],"data_offsets":[0,8]}}"#;
    let mut raw = (header.len() as u64).to_le_bytes().to_vec();
    raw.extend_from_slice(header);
    raw.extend_from_slice(b"abcdefgh");
    std::fs::write(path.join("x.safetensors"), &raw).unwrap();
    std::fs::write(
        path.join("model.index.json"),
        br#"{"metadata":{"total_size":8},"weight_map":{"x":"x.safetensors"}}"#,
    )
    .unwrap();
    let mut source = inspect_shards(&path, "model.index.json", ShardLimits::default()).unwrap();
    let store = ArtifactStore::create(
        &path.join("store"),
        StoreLimits {
            total_bytes: 4096,
            max_blob_bytes: 4096,
            max_entries: 20,
        },
    )
    .unwrap();
    source.snapshot(&store).unwrap();
    let before = store.used_bytes().unwrap();
    *raw.last_mut().unwrap() = b'X';
    std::fs::write(path.join("x.safetensors"), &raw).unwrap();
    // Simulate a filesystem whose stat timestamps did not distinguish this
    // same-size rewrite. Digest fences must still reject the changed bytes.
    source.shards.get_mut("x.safetensors").unwrap().source.stamp =
        Stamp::from(&std::fs::metadata(path.join("x.safetensors")).unwrap());
    assert_eq!(
        source.snapshot(&store).unwrap_err().code,
        ErrorCode::IntegrityMismatch
    );
    assert_eq!(store.used_bytes().unwrap(), before);
    std::fs::remove_dir_all(path).unwrap();
}

static NEXT: AtomicU64 = AtomicU64::new(0);

struct ChangeDuringCopy<'a> {
    reader: VerifiedRead<'a>,
    path: std::path::PathBuf,
    changed: bool,
}
impl Read for ChangeDuringCopy<'_> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        let count = self.reader.read(out)?;
        if count != 0 && !self.changed {
            self.changed = true;
            std::fs::write(&self.path, b"abcdEFGH")?;
        }
        Ok(count)
    }
}

#[test]
fn sharded_io_mid_copy_changes_never_publish() {
    let scratch = std::env::var_os("KIROCREW_SCRATCH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let path = scratch.join(format!(
        "shard-copy-change-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&path).unwrap();
    std::fs::write(path.join("a.safetensors"), b"abcdefgh").unwrap();
    let root = SourceRoot::open(&path).unwrap();
    let file = root.open_file("a.safetensors").unwrap();
    let source = SourceBlob {
        name: "a.safetensors".to_owned(),
        stamp: Stamp::from(&file.metadata().unwrap()),
        digest: ContentDigest::of_bytes(b"abcdefgh"),
    };
    let reader = VerifiedRead {
        file,
        root: &root,
        source: &source,
        hash: Sha256::new(),
        bytes: 0,
        chunk: 4,
        failure: None,
    };
    let mut changing = ChangeDuringCopy {
        reader,
        path: path.join("a.safetensors"),
        changed: false,
    };
    let store = ArtifactStore::create(
        &path.join("store"),
        StoreLimits {
            total_bytes: 4096,
            max_blob_bytes: 4096,
            max_entries: 20,
        },
    )
    .unwrap();
    assert!(store.put(&mut changing).is_err());
    assert_eq!(
        changing.reader.failure.as_ref().unwrap().code,
        ErrorCode::IntegrityMismatch
    );
    assert_eq!(store.used_bytes().unwrap(), 0);
    std::fs::remove_dir_all(path).unwrap();
}
