#![cfg(target_os = "linux")]
use nf_checkpoint::export::{
    publish_export, restore_export, restore_export_with_reader, ExportBlob, ExportLimits,
    ExportManifest, LoadedExport,
};
use nf_checkpoint::serialization::{serialize_tensors, SerializationLimits, TensorPayload};
use nf_checkpoint::store::{ArtifactStore, StoreLimits};
use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use nf_core::error::ErrorCode;
use nf_core::schemas::{DType, FactoryRef, ModelManifest, TensorFormat, TensorSpec, Validated};
use nf_graph::inventory::tensor_set_digest;
use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Root(PathBuf);
impl Root {
    fn new() -> Self {
        let parent =
            PathBuf::from(std::env::var_os("KIROCREW_SCRATCH").expect("session scratch required"));
        let path = parent.join(format!(
            "export-tests-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn store(&self) -> ArtifactStore {
        open(&self.0)
    }
}
impl Drop for Root {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn open(path: &std::path::Path) -> ArtifactStore {
    ArtifactStore::create(
        &path.join("store"),
        StoreLimits {
            total_bytes: 1_000_000,
            max_blob_bytes: 500_000,
            max_entries: 100,
        },
    )
    .unwrap()
}
type Assets = BTreeMap<String, Arc<[u8]>>;
fn fixture() -> (Validated<ModelManifest>, Arc<[u8]>, Assets) {
    let data: Vec<u8> = [1.0f64, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let buffer = 7i64.to_le_bytes();
    let checkpoint = serialize_tensors(
        &BTreeMap::from([
            (
                "w".to_owned(),
                TensorPayload {
                    dtype: DType::F64,
                    shape: &[2, 2],
                    bytes: &data,
                },
            ),
            (
                "cache".to_owned(),
                TensorPayload {
                    dtype: DType::I64,
                    shape: &[1],
                    bytes: &buffer,
                },
            ),
        ]),
        SerializationLimits::default(),
    )
    .unwrap();
    let config = ContentDigest::of_bytes(b"{}");
    let spec = TensorSpec {
        shape: vec![2, 2],
        dtype: DType::F64,
        digest: ContentDigest::of_bytes(&data),
        storage_group: "weights".to_owned(),
    };
    let parameters = BTreeMap::from([("w".to_owned(), spec.clone()), ("tied".to_owned(), spec)]);
    let model = Validated::new(ModelManifest {
        schema_version: 1,
        architecture: FactoryRef {
            id: "fixture.cpu".to_owned(),
            version: "1".to_owned(),
            config: config.clone(),
        },
        parameter_digest: tensor_set_digest(&parameters).unwrap(),
        parameters,
        buffers: BTreeMap::from([(
            "cache".to_owned(),
            TensorSpec {
                shape: vec![1],
                dtype: DType::I64,
                digest: ContentDigest::of_bytes(&buffer),
                storage_group: "cache".to_owned(),
            },
        )]),
        format: TensorFormat::DenseSafetensors,
        packing_policy: None,
        assets: BTreeMap::from([
            ("config".to_owned(), config),
            ("tokenizer".to_owned(), ContentDigest::of_bytes(b"toy")),
            ("checkpoint".to_owned(), checkpoint.digest),
        ]),
        usage_rights: "synthetic fixture only".to_owned(),
    })
    .unwrap();
    (
        model,
        Arc::from(checkpoint.bytes),
        BTreeMap::from([
            ("config".to_owned(), Arc::from(b"{}".as_slice())),
            ("tokenizer".to_owned(), Arc::from(b"toy".as_slice())),
        ]),
    )
}
#[test]
fn export_restore_fresh_process_standard_bytes_ties_and_forward_parity() {
    if let Some(path) = std::env::var_os("NF_EXPORT_TEST_ROOT") {
        let store = ArtifactStore::open_read_only(
            &PathBuf::from(path).join("store"),
            StoreLimits {
                total_bytes: 1_000_000,
                max_blob_bytes: 500_000,
                max_entries: 100,
            },
        )
        .unwrap();
        let root = std::env::var("NF_EXPORT_TEST_ID").unwrap().parse().unwrap();
        let expected = std::env::var("NF_EXPORT_TEST_MODEL")
            .unwrap()
            .parse()
            .unwrap();
        let loaded = restore_export_with_reader(
            &root,
            Some(&expected),
            ExportLimits::default(),
            |digest, bound| store.read_bounded(digest, bound),
        )
        .unwrap();
        assert_eq!(loaded.manifest_digest(), &expected);
        let w = loaded.tensor_bytes("w").unwrap();
        assert_eq!(w.as_ptr(), loaded.tensor_bytes("tied").unwrap().as_ptr());
        let weights: Vec<f64> = w
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| f64::from_le_bytes(*b))
            .collect();
        let output: Vec<f64> = weights
            .as_chunks::<2>()
            .0
            .iter()
            .map(|row| row[0] * 2.0 + row[1] * 3.0)
            .collect();
        assert_eq!(output, [8.0, 18.0]);
        assert_eq!(loaded.tensor_bytes("cache").unwrap(), 7i64.to_le_bytes());
        assert_eq!(
            loaded.model().parameter_digest,
            fixture().0.parameter_digest
        );
        assert_eq!(loaded.assets()["tokenizer"].as_ref(), b"toy");
        return;
    }
    let root = Root::new();
    let store = root.store();
    let (model, checkpoint, assets) = fixture();
    let id = publish_export(
        &store,
        &model,
        checkpoint.clone(),
        assets.clone(),
        ExportLimits::default(),
    )
    .unwrap();
    assert_eq!(
        publish_export(
            &store,
            &model,
            checkpoint.clone(),
            assets,
            ExportLimits::default()
        )
        .unwrap(),
        id
    );
    let loaded = restore_export(&store, &id, ExportLimits::default()).unwrap();
    assert_eq!(loaded.checkpoint().as_ref(), checkpoint.as_ref());
    assert_eq!(&**loaded.model(), &*model);
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "export_restore_fresh_process_standard_bytes_ties_and_forward_parity",
            "--nocapture",
        ])
        .env("NF_EXPORT_TEST_ROOT", &root.0)
        .env("NF_EXPORT_TEST_ID", id.as_str())
        .env("NF_EXPORT_TEST_MODEL", loaded.manifest_digest().as_str())
        .status()
        .unwrap();
    assert!(status.success());
}
#[test]
fn export_restore_rejects_bad_metadata_and_missing_assets_before_writes() {
    let root = Root::new();
    let store = root.store();
    let (model, bytes, mut assets) = fixture();
    assets.remove("tokenizer");
    assert!(publish_export(
        &store,
        &model,
        bytes.clone(),
        assets,
        ExportLimits::default()
    )
    .is_err());
    assert_eq!(store.used_bytes().unwrap(), 0);
    let (_, _, assets) = fixture();
    let mut bad = (*model).clone();
    bad.parameter_digest = ContentDigest::of_bytes(b"false");
    assert!(publish_export(
        &store,
        &Validated::new(bad).unwrap(),
        bytes.clone(),
        assets.clone(),
        ExportLimits::default()
    )
    .is_err());
    let mut bad = (*model).clone();
    bad.buffers.get_mut("cache").unwrap().dtype = DType::F64;
    assert!(publish_export(
        &store,
        &Validated::new(bad).unwrap(),
        bytes,
        assets,
        ExportLimits::default()
    )
    .is_err());
    assert_eq!(store.used_bytes().unwrap(), 0);
}
#[test]
fn export_restore_corrupt_missing_and_incomplete_blobs_fail_without_source_changes() {
    let root = Root::new();
    let store = root.store();
    let (model, bytes, assets) = fixture();
    let original = bytes.clone();
    let id = publish_export(
        &store,
        &model,
        bytes.clone(),
        assets,
        ExportLimits::default(),
    )
    .unwrap();
    let path = root
        .0
        .join("store/blobs")
        .join(&model.assets["checkpoint"].as_str()[7..]);
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
    let mut corrupt = bytes.to_vec();
    *corrupt.last_mut().unwrap() ^= 1;
    fs::write(&path, corrupt).unwrap();
    assert_eq!(
        restore_export(&store, &id, ExportLimits::default())
            .unwrap_err()
            .code,
        ErrorCode::IntegrityMismatch
    );
    fs::remove_file(&path).unwrap();
    assert!(restore_export(&store, &id, ExportLimits::default()).is_err());
    let mut manifest: ExportManifest = serde_json::from_slice(&store.read(&id).unwrap()).unwrap();
    manifest.assets.remove("config");
    let bad_root = store
        .put(serde_json::to_vec(&manifest).unwrap().as_slice())
        .unwrap()
        .digest;
    assert!(restore_export(&store, &bad_root, ExportLimits::default()).is_err());
    assert_eq!(bytes, original);
    assert!(store.read(&id).is_ok());
}
#[test]
fn export_restore_limits_alias_contract_and_unknown_fields_are_closed() {
    let root = Root::new();
    let store = root.store();
    let (model, bytes, assets) = fixture();
    let limits = ExportLimits {
        max_checkpoint_bytes: 1,
        ..ExportLimits::default()
    };
    assert_eq!(
        publish_export(&store, &model, bytes.clone(), assets.clone(), limits)
            .unwrap_err()
            .code,
        ErrorCode::LimitExceeded
    );
    assert_eq!(store.used_bytes().unwrap(), 0);
    let id = publish_export(&store, &model, bytes, assets, ExportLimits::default()).unwrap();
    assert!(restore_export(&store, &id, limits).is_err());
    assert!(store.read_bounded(&id, 1).is_err());
    let mut value: serde_json::Value = serde_json::from_slice(&store.read(&id).unwrap()).unwrap();
    value["alias_contract"] = serde_json::json!("guess-from-equal-values");
    let bad = store
        .put(serde_json::to_vec(&value).unwrap().as_slice())
        .unwrap()
        .digest;
    assert_eq!(
        restore_export(&store, &bad, ExportLimits::default())
            .unwrap_err()
            .code,
        ErrorCode::UnsupportedVersion
    );
    value["alias_contract"] = serde_json::json!("canonical_group_shared_storage_v1");
    value["accepted"] = serde_json::json!(true);
    let bad = store
        .put(serde_json::to_vec(&value).unwrap().as_slice())
        .unwrap()
        .digest;
    assert!(restore_export(&store, &bad, ExportLimits::default()).is_err());
}
#[test]
fn export_restore_interrupted_publication_has_no_root_or_active_pointer() {
    let root = Root::new();
    let (model, bytes, assets) = fixture();
    let store = ArtifactStore::create(
        &root.0.join("store"),
        StoreLimits {
            total_bytes: bytes.len() as u64 + 5,
            max_blob_bytes: bytes.len() as u64 + 5,
            max_entries: 100,
        },
    )
    .unwrap();
    assert!(publish_export(
        &store,
        &model,
        bytes.clone(),
        assets,
        ExportLimits::default()
    )
    .is_err());
    let blobs: Vec<_> = fs::read_dir(root.0.join("store/blobs")).unwrap().collect();
    assert!(blobs.len() <= 3);
    for blob in blobs {
        let data = fs::read(blob.unwrap().path()).unwrap();
        assert!(serde_json::from_slice::<ExportManifest>(&data).is_err());
    }
    assert!(!root.0.join("active").exists());
    assert_eq!(ContentDigest::of_bytes(&bytes), model.assets["checkpoint"]);
}

// A reader adapter independent of ArtifactStore must get the same native checks.
type Blobs = BTreeMap<ContentDigest, Vec<u8>>;
fn detached_bundle() -> (ExportManifest, Blobs) {
    let (model, checkpoint, assets) = fixture();
    let mut blobs = Blobs::new();
    let manifest = ExportManifest {
        schema_version: 1,
        kind: "standalone_dense_checkpoint".to_owned(),
        alias_contract: "canonical_group_shared_storage_v1".to_owned(),
        model: (*model).clone(),
        checkpoint: ExportBlob {
            digest: ContentDigest::of_bytes(&checkpoint),
            bytes: checkpoint.len() as u64,
        },
        assets: assets
            .iter()
            .map(|(name, bytes)| {
                let digest = ContentDigest::of_bytes(bytes);
                blobs.insert(digest.clone(), bytes.to_vec());
                (
                    name.clone(),
                    ExportBlob {
                        digest,
                        bytes: bytes.len() as u64,
                    },
                )
            })
            .collect(),
    };
    blobs.insert(manifest.checkpoint.digest.clone(), checkpoint.to_vec());
    (manifest, blobs)
}
fn model_identity(manifest: &ExportManifest) -> ContentDigest {
    CanonicalDocument::parse(
        &serde_json::to_vec(&manifest.model).unwrap(),
        Limits::default(),
    )
    .unwrap()
    .digest()
}
fn detached_restore(
    manifest: &ExportManifest,
    mut blobs: Blobs,
) -> nf_core::error::Result<LoadedExport> {
    let raw = serde_json::to_vec(manifest).unwrap();
    let root = ContentDigest::of_bytes(&raw);
    blobs.insert(root.clone(), raw);
    restore_export_with_reader(
        &root,
        Some(&model_identity(manifest)),
        ExportLimits::default(),
        |digest, _bound| Ok(blobs[digest].clone()),
    )
}
fn replace_checkpoint(
    manifest: &mut ExportManifest,
    blobs: &mut Blobs,
    tensors: &BTreeMap<String, TensorPayload<'_>>,
) {
    let checkpoint = serialize_tensors(tensors, SerializationLimits::default()).unwrap();
    manifest.checkpoint = ExportBlob {
        digest: checkpoint.digest.clone(),
        bytes: checkpoint.bytes.len() as u64,
    };
    manifest
        .model
        .assets
        .insert("checkpoint".to_owned(), checkpoint.digest.clone());
    blobs.insert(checkpoint.digest, checkpoint.bytes);
}

#[test]
fn reader_restore_pins_model_and_transfers_verified_runtime_inputs() {
    let (manifest, blobs) = detached_bundle();
    let expected = model_identity(&manifest);
    let loaded = detached_restore(&manifest, blobs).unwrap();
    assert_eq!(loaded.manifest_digest(), &expected);
    assert_eq!(loaded.inventory().entries().len(), 3);
    assert_eq!(loaded.inventory().alias_groups()["weights"], ["tied", "w"]);
    assert_eq!(
        loaded.tensor_bytes("w").unwrap().as_ptr(),
        loaded.tensor_bytes("tied").unwrap().as_ptr()
    );
    let checkpoint = loaded.checkpoint().clone();
    let (model, transferred, assets) = loaded.into_parts();
    assert!(Arc::ptr_eq(&checkpoint, &transferred));
    assert_eq!(*model, manifest.model);
    assert_eq!(assets["config"].as_ref(), b"{}");
}

#[test]
fn reader_restore_rejects_wrong_identity_root_and_limits_before_payload_reads() {
    let (manifest, _) = detached_bundle();
    let raw = serde_json::to_vec(&manifest).unwrap();
    let root = ContentDigest::of_bytes(&raw);
    let wrong = ContentDigest::of_bytes(b"not the frozen identity");
    let mut calls = 0;
    let error =
        restore_export_with_reader(&root, Some(&wrong), ExportLimits::default(), |digest, _| {
            calls += 1;
            assert_eq!(digest, &root);
            Ok(raw.clone())
        })
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::SourceMismatch);
    assert_eq!(calls, 1);
    let error =
        restore_export_with_reader(
            &wrong,
            None,
            ExportLimits::default(),
            |_, _| Ok(raw.clone()),
        )
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::IntegrityMismatch);
    let error = restore_export_with_reader(
        &root,
        None,
        ExportLimits {
            max_tensors: 0,
            ..ExportLimits::default()
        },
        |_, _| panic!("invalid limits must not invoke reader"),
    )
    .unwrap_err();
    assert_eq!(error.code, ErrorCode::InvalidValue);
    let error = restore_export_with_reader(
        &root,
        None,
        ExportLimits {
            max_total_bytes: 1,
            ..ExportLimits::default()
        },
        |_, bound| {
            assert_eq!(bound, 1);
            Ok(raw.clone()) // Deliberately nonconforming reader.
        },
    )
    .unwrap_err();
    assert_eq!(error.code, ErrorCode::LimitExceeded);
}

#[test]
fn reader_restore_checks_each_payload_even_when_reader_claims_success() {
    let (manifest, blobs) = detached_bundle();
    for digest in [
        &manifest.checkpoint.digest,
        &manifest.assets["config"].digest,
        &manifest.assets["tokenizer"].digest,
    ] {
        for truncate in [false, true] {
            let mut corrupt = blobs.clone();
            let data = corrupt.get_mut(digest).unwrap();
            if truncate {
                data.pop();
            } else {
                data[0] ^= 1;
            }
            assert_eq!(
                detached_restore(&manifest, corrupt).unwrap_err().code,
                ErrorCode::IntegrityMismatch
            );
        }
    }
}

#[test]
fn reader_restore_rejects_conflicting_and_invalid_config_but_allows_empty_auxiliary_assets() {
    let (mut manifest, mut blobs) = detached_bundle();
    manifest
        .model
        .assets
        .insert("config".to_owned(), ContentDigest::of_bytes(b"other"));
    assert_eq!(
        detached_restore(&manifest, blobs.clone()).unwrap_err().code,
        ErrorCode::SourceMismatch
    );
    let invalid = b"not json";
    let digest = ContentDigest::of_bytes(invalid);
    manifest.model.architecture.config = digest.clone();
    manifest
        .model
        .assets
        .insert("config".to_owned(), digest.clone());
    manifest.assets.insert(
        "config".to_owned(),
        ExportBlob {
            digest: digest.clone(),
            bytes: invalid.len() as u64,
        },
    );
    blobs.insert(digest, invalid.to_vec());
    assert!(detached_restore(&manifest, blobs).is_err());
    let (mut manifest, mut blobs) = detached_bundle();
    let empty = ContentDigest::of_bytes(b"");
    manifest
        .model
        .assets
        .insert("tokenizer".to_owned(), empty.clone());
    manifest.assets.insert(
        "tokenizer".to_owned(),
        ExportBlob {
            digest: empty.clone(),
            bytes: 0,
        },
    );
    blobs.insert(empty, vec![]);
    assert!(detached_restore(&manifest, blobs).unwrap().assets()["tokenizer"].is_empty());
}

#[test]
fn reader_restore_serialized_aliases_must_match_and_equal_independent_groups_stay_separate() {
    let data: Vec<_> = [1.0f64, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let buffer = 7i64.to_le_bytes();
    let mut tensors = BTreeMap::from([
        (
            "w".to_owned(),
            TensorPayload {
                dtype: DType::F64,
                shape: &[2, 2],
                bytes: &data,
            },
        ),
        (
            "tied".to_owned(),
            TensorPayload {
                dtype: DType::F64,
                shape: &[2, 2],
                bytes: &data,
            },
        ),
        (
            "cache".to_owned(),
            TensorPayload {
                dtype: DType::I64,
                shape: &[1],
                bytes: &buffer,
            },
        ),
    ]);
    let (mut manifest, mut blobs) = detached_bundle();
    replace_checkpoint(&mut manifest, &mut blobs, &tensors);
    let loaded = detached_restore(&manifest, blobs.clone()).unwrap();
    assert_eq!(
        loaded.tensor_bytes("w").unwrap().as_ptr(),
        loaded.tensor_bytes("tied").unwrap().as_ptr()
    );
    manifest
        .model
        .parameters
        .get_mut("tied")
        .unwrap()
        .storage_group = "independent".to_owned();
    let loaded = detached_restore(&manifest, blobs.clone()).unwrap();
    assert_ne!(
        loaded.tensor_bytes("w").unwrap().as_ptr(),
        loaded.tensor_bytes("tied").unwrap().as_ptr()
    );
    manifest
        .model
        .parameters
        .get_mut("tied")
        .unwrap()
        .storage_group = "weights".to_owned();
    let wrong = vec![0; data.len()];
    tensors.get_mut("tied").unwrap().bytes = &wrong;
    replace_checkpoint(&mut manifest, &mut blobs, &tensors);
    assert_eq!(
        detached_restore(&manifest, blobs.clone()).unwrap_err().code,
        ErrorCode::SourceMismatch
    );
    tensors.remove("tied");
    tensors.remove("w");
    replace_checkpoint(&mut manifest, &mut blobs, &tensors);
    assert_eq!(
        detached_restore(&manifest, blobs.clone()).unwrap_err().code,
        ErrorCode::SourceMismatch
    );
    tensors.insert(
        "rogue".to_owned(),
        TensorPayload {
            dtype: DType::F64,
            shape: &[2, 2],
            bytes: &data,
        },
    );
    replace_checkpoint(&mut manifest, &mut blobs, &tensors);
    assert_eq!(
        detached_restore(&manifest, blobs).unwrap_err().code,
        ErrorCode::SourceMismatch
    );
}

#[test]
fn reader_restore_charges_actual_noncanonical_root_bytes_to_total() {
    let (manifest, mut blobs) = detached_bundle();
    let canonical =
        CanonicalDocument::parse(&serde_json::to_vec(&manifest).unwrap(), Limits::default())
            .unwrap();
    let total = canonical.bytes().len() as u64
        + manifest.checkpoint.bytes
        + manifest.assets.values().map(|blob| blob.bytes).sum::<u64>();
    let mut raw = canonical.bytes().to_vec();
    raw.extend_from_slice(b"  ");
    let root = ContentDigest::of_bytes(&raw);
    blobs.insert(root.clone(), raw);
    let mut calls = 0;
    let error = restore_export_with_reader(
        &root,
        None,
        ExportLimits {
            max_total_bytes: total,
            ..ExportLimits::default()
        },
        |digest, _| {
            calls += 1;
            Ok(blobs[digest].clone())
        },
    )
    .unwrap_err();
    assert_eq!(error.code, ErrorCode::LimitExceeded);
    assert_eq!(calls, 1);
}
