#![cfg(target_os = "linux")]

use nf_checkpoint::sharded::{inspect_shards, ShardLimits};
use nf_checkpoint::store::{ArtifactStore, StoreLimits};
use nf_checkpoint::tensor_file::{inspect_header, TensorFileLimits};
use nf_core::canonical::ContentDigest;
use nf_core::error::ErrorCode;
use safetensors::SafeTensors;
use serde_json::json;
use std::fs;
use std::os::unix::fs::symlink;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Fixture {
    path: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let parent = std::env::var_os("KIROCREW_SCRATCH")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        let path = parent.join(format!(
            "shard-tests-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        fs::create_dir(path.join("source")).unwrap();
        let value = Self { path };
        value.shard("a.safetensors", "a", &[1, 2, 3]);
        value.shard("b.safetensors", "b", &[4, 5]);
        value.index(json!({"a":"a.safetensors", "b":"b.safetensors"}), 5);
        value
    }
    fn source(&self) -> PathBuf {
        self.path.join("source")
    }
    fn shard(&self, file: &str, name: &str, data: &[u8]) {
        let header =
            json!({(name):{"dtype":"U8","shape":[data.len()],"data_offsets":[0,data.len()]}})
                .to_string();
        fs::write(self.source().join(file), tensor_file(&header, data)).unwrap();
    }
    fn index(&self, map: serde_json::Value, total: u64) {
        fs::write(
            self.source().join("model.index.json"),
            json!({"metadata":{"total_size":total},"weight_map":map}).to_string(),
        )
        .unwrap();
    }
    fn inspect(&self) -> Result<nf_checkpoint::sharded::ShardedSource, nf_core::error::CoreError> {
        inspect_shards(&self.source(), "model.index.json", ShardLimits::default())
    }
    fn store(&self) -> ArtifactStore {
        ArtifactStore::create(
            &self.path.join("store"),
            StoreLimits {
                total_bytes: 4_194_304,
                max_blob_bytes: 2_097_152,
                max_entries: 100,
            },
        )
        .unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.path).unwrap();
    }
}
fn tensor_file(header: &str, data: &[u8]) -> Vec<u8> {
    let mut raw = header.as_bytes().to_vec();
    raw.resize(raw.len().div_ceil(8) * 8, b' ');
    let mut out = (raw.len() as u64).to_le_bytes().to_vec();
    out.extend(raw);
    out.extend_from_slice(data);
    out
}

#[test]
fn sharded_io_snapshot_replay_and_only_changed_shards_are_staged() {
    let fixture = Fixture::new();
    let source = fixture.inspect().unwrap();
    assert_eq!(
        source.tensor_digests()["a"],
        ContentDigest::of_bytes(&[1, 2, 3])
    );
    let store = fixture.store();
    let first = source.snapshot(&store).unwrap();
    assert_eq!(first.stats.new_shards, 2);
    assert_eq!(first.stats.reused_shards, 0);
    let again = source.snapshot(&store).unwrap();
    assert_eq!(again.manifest_digest, first.manifest_digest);
    assert_eq!(again.stats.new_shards, 0);
    assert_eq!(again.stats.reused_shards, 2);
    assert_eq!(again.stats.new_shard_bytes, 0);
    fixture.shard("a.safetensors", "a", &[9, 2, 3]);
    let changed = fixture.inspect().unwrap().snapshot(&store).unwrap();
    assert_eq!(changed.stats.new_shards, 1);
    assert_eq!(changed.stats.reused_shards, 1);
    assert_eq!(
        changed.manifest.shards["b.safetensors"],
        first.manifest.shards["b.safetensors"]
    );
    assert_ne!(
        changed.manifest.shards["a.safetensors"],
        first.manifest.shards["a.safetensors"]
    );
    assert_eq!(
        changed.stats.new_shard_bytes,
        fs::metadata(fixture.source().join("a.safetensors"))
            .unwrap()
            .len()
    );
}

#[test]
fn sharded_io_missing_extra_and_misindexed_tensors_are_rejected() {
    let fixture = Fixture::new();
    fs::remove_file(fixture.source().join("b.safetensors")).unwrap();
    assert!(fixture.inspect().is_err());
    fixture.shard("b.safetensors", "b", &[4, 5]);
    fixture.shard("extra.safetensors", "extra", &[0]);
    assert!(fixture.inspect().is_err());
    fs::remove_file(fixture.source().join("extra.safetensors")).unwrap();
    fixture.index(json!({"a":"b.safetensors", "b":"a.safetensors"}), 5);
    assert!(fixture.inspect().is_err());
    fixture.index(
        json!({"a":"a.safetensors", "b":"b.safetensors", "ghost":"b.safetensors"}),
        5,
    );
    assert!(fixture.inspect().is_err());
    fixture.index(json!({"a":"a.safetensors", "b":"b.safetensors"}), 999);
    assert!(fixture.inspect().is_err());
}

#[test]
fn sharded_io_duplicate_index_and_unknown_fields_are_rejected() {
    let fixture = Fixture::new();
    fs::write(
        fixture.source().join("model.index.json"),
        r#"{"metadata":{"total_size":5},"weight_map":{"a":"a.safetensors","a":"b.safetensors"}}"#,
    )
    .unwrap();
    assert_eq!(fixture.inspect().unwrap_err().code, ErrorCode::DuplicateKey);
    fs::write(
        fixture.source().join("model.index.json"),
        r#"{"metadata":{"total_size":5},"weight_map":{},"execute":"not allowed"}"#,
    )
    .unwrap();
    assert!(fixture.inspect().is_err());
}

#[test]
fn sharded_io_rejects_traversal_symlinks_and_hardlinks() {
    let fixture = Fixture::new();
    fixture.index(json!({"a":"../a.safetensors"}), 3);
    assert!(fixture.inspect().is_err());
    fixture.index(json!({"a":"a.safetensors", "b":"b.safetensors"}), 5);
    let b = fixture.source().join("b.safetensors");
    fs::rename(&b, fixture.path.join("outside")).unwrap();
    symlink(fixture.path.join("outside"), &b).unwrap();
    assert!(fixture.inspect().is_err());
    fs::remove_file(&b).unwrap();
    fs::hard_link(fixture.path.join("outside"), &b).unwrap();
    assert!(fixture.inspect().is_err());
    assert!(inspect_shards(
        &fixture.source(),
        "../model.index.json",
        ShardLimits::default()
    )
    .is_err());
}

#[test]
fn sharded_io_source_changes_prevent_checkpoint_publication() {
    let fixture = Fixture::new();
    let source = fixture.inspect().unwrap();
    let store = fixture.store();
    fixture.shard("a.safetensors", "a", &[0, 2, 3]);
    assert_eq!(
        source.snapshot(&store).unwrap_err().code,
        ErrorCode::IntegrityMismatch
    );
    assert_eq!(store.used_bytes().unwrap(), 0);
    let source = fixture.inspect().unwrap();
    fixture.index(json!({"a":"a.safetensors", "b":"b.safetensors"}), 6);
    assert_eq!(
        source.snapshot(&store).unwrap_err().code,
        ErrorCode::IntegrityMismatch
    );
    assert_eq!(store.used_bytes().unwrap(), 0);
}

#[test]
fn sharded_io_payload_buffers_and_io_receipts_are_bounded() {
    let fixture = Fixture::new();
    fs::remove_file(fixture.source().join("b.safetensors")).unwrap();
    let data = vec![7; 200_000];
    fixture.shard("a.safetensors", "a", &data);
    fixture.index(json!({"a":"a.safetensors"}), data.len() as u64);
    let source = inspect_shards(
        &fixture.source(),
        "model.index.json",
        ShardLimits {
            chunk_bytes: 17,
            ..ShardLimits::default()
        },
    )
    .unwrap();
    assert_eq!(source.inspection_stats().payload_buffer_bytes, 17);
    assert!(source.inspection_stats().largest_header_bytes < 1024);
    let actual_bytes = fs::metadata(fixture.source().join("a.safetensors"))
        .unwrap()
        .len()
        + fs::metadata(fixture.source().join("model.index.json"))
            .unwrap()
            .len();
    assert_eq!(source.inspection_stats().source_bytes_read, actual_bytes);
    let receipt = source.snapshot(&fixture.store()).unwrap();
    assert_eq!(receipt.stats.source_bytes_read, actual_bytes * 4);
    assert_eq!(receipt.stats.snapshot_source_bytes_read, actual_bytes * 3);
    assert_eq!(receipt.stats.payload_buffer_bytes, 17);
    assert_eq!(receipt.stats.staging_buffer_bytes, 65_536);
    assert_eq!(source.tensor_digests()["a"], ContentDigest::of_bytes(&data));
}

#[test]
fn sharded_io_aggregate_index_shard_and_header_bounds_are_enforced() {
    let fixture = Fixture::new();
    for limits in [
        ShardLimits {
            max_shards: 1,
            ..ShardLimits::default()
        },
        ShardLimits {
            max_index_bytes: 1,
            ..ShardLimits::default()
        },
        ShardLimits {
            max_total_file_bytes: 1,
            ..ShardLimits::default()
        },
        ShardLimits {
            max_total_header_bytes: 1,
            ..ShardLimits::default()
        },
        ShardLimits {
            max_directory_entries: 1,
            ..ShardLimits::default()
        },
        ShardLimits {
            chunk_bytes: 0,
            ..ShardLimits::default()
        },
    ] {
        let expected = if limits.chunk_bytes == 0 {
            ErrorCode::InvalidValue
        } else {
            ErrorCode::LimitExceeded
        };
        assert_eq!(
            inspect_shards(&fixture.source(), "model.index.json", limits)
                .unwrap_err()
                .code,
            expected
        );
    }
}

#[test]
fn sharded_io_streaming_metadata_matches_upstream_payload_rules() {
    for start in 0..4 {
        for end in start..5 {
            for payload in 0..5 {
                let header = format!(
                    r#"{{"x":{{"dtype":"U8","shape":[{}],"data_offsets":[{start},{end}]}}}}"#,
                    end - start
                );
                let raw = tensor_file(&header, &vec![0; payload]);
                let size = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
                let streaming =
                    inspect_header(&raw[8..8 + size], raw.len(), TensorFileLimits::default());
                assert_eq!(
                    streaming.is_ok(),
                    SafeTensors::deserialize(&raw).is_ok(),
                    "start={start} end={end} payload={payload}"
                );
            }
        }
    }
    let raw = tensor_file(
        r#"{"empty":{"dtype":"U8","shape":[0],"data_offsets":[0,0]},"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
        &[1],
    );
    let size = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
    assert!(inspect_header(&raw[8..8 + size], raw.len(), TensorFileLimits::default()).is_ok());
    assert!(SafeTensors::deserialize(&raw).is_ok());
}
