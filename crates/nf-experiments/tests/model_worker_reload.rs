#![cfg(target_os = "linux")]
//! Bounded real CPU fixture children; no downloads or study/promotion execution.
use nf_checkpoint::export::{publish_export, ExportLimits};
use nf_checkpoint::store::{ArtifactStore, StoreLimits};
use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits as JsonLimits};
use nf_core::schemas::{ModelManifest, Validated};
use nf_core::study::ModelIdentity;
use nf_experiments::model_worker as worker;
use nf_graph::inventory::tensor_set_digest;
use nf_surgeon::cancellation::CancellationToken;
use nf_surgeon::transaction::{SourceState, StateLoadLimits};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use worker::reload::{reload_graph, FrozenReloadProbes};
use worker::{Limits, QualifiedTorch};
static NEXT: AtomicU64 = AtomicU64::new(0);
fn bytes(v: &Value) -> Vec<u8> {
    CanonicalDocument::parse(&serde_json::to_vec(v).unwrap(), JsonLimits::default())
        .unwrap()
        .bytes()
        .to_vec()
}
struct Fixture {
    root: PathBuf,
    store: ArtifactStore,
}
impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "reload-witness-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::DirBuilder::new()
            .mode(0o700)
            .create(&root)
            .unwrap();
        let store = ArtifactStore::create(
            &root.join("store"),
            StoreLimits {
                total_bytes: 8_388_608,
                max_blob_bytes: 1_048_576,
                max_entries: 128,
            },
        )
        .unwrap();
        Self { root, store }
    }
    fn runtime(&self) -> QualifiedTorch {
        let project = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        // Cargo's library is hard-linked; qualification correctly rejects it.
        let extension = self.root.join("_native.so");
        std::fs::copy(project.join("target/debug/lib_native.so"), &extension).unwrap();
        QualifiedTorch::qualify(
            true,
            Path::new("/usr/bin/python3.12"),
            &project.join(".venv/lib/python3.12/site-packages"),
            &extension,
            &self.root,
        )
        .unwrap()
    }
    fn publish(&self, source: &SourceState) -> ContentDigest {
        publish_export(
            &self.store,
            source.manifest(),
            Arc::from(source.checkpoint_bytes()),
            source.assets().clone(),
            ExportLimits::default(),
        )
        .unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}
fn source(buffered: bool, weights: [f64; 4]) -> SourceState {
    let mut f: Value = serde_json::from_str(include_str!(
        "../../../python/neural_foundry/data/m1_fixture.json"
    ))
    .unwrap();
    let w: Vec<u8> = weights.into_iter().flat_map(f64::to_le_bytes).collect();
    let offsets: Vec<u8> = [0.5_f64, -1.0]
        .into_iter()
        .flat_map(f64::to_le_bytes)
        .collect();
    let mut header = json!({"w":{"dtype":"F64","shape":[2,2],"data_offsets":[0,32]}});
    let mut config = b"{}".to_vec();
    if buffered {
        header["offset"] = json!({"dtype":"F64","shape":[2],"data_offsets":[32,48]});
        config =
            bytes(&json!({"schema_version":1,"rows":2,"columns":2,"tied":true,"buffer":"offset"}));
        f["manifest"]["architecture"]["id"] = json!("nf.fixture.linear_buffered");
        let spec = json!({"dtype":"f64","shape":[2],"storage_group":"offset","digest":ContentDigest::of_bytes(&offsets)});
        f["manifest"]["buffers"] = json!({"offset":spec,"offset_alias":spec});
    }
    let mut h = bytes(&header);
    while !h.len().is_multiple_of(8) {
        h.push(b' ');
    }
    let mut checkpoint = (h.len() as u64).to_le_bytes().to_vec();
    checkpoint.extend(h);
    checkpoint.extend(&w);
    if buffered {
        checkpoint.extend(offsets);
    }
    for name in ["w", "tied"] {
        f["manifest"]["parameters"][name]["digest"] = json!(ContentDigest::of_bytes(&w));
    }
    let parameters = serde_json::from_value(f["manifest"]["parameters"].clone()).unwrap();
    f["manifest"]["parameter_digest"] = json!(tensor_set_digest(&parameters).unwrap());
    f["manifest"]["architecture"]["config"] = json!(ContentDigest::of_bytes(&config));
    f["manifest"]["assets"] = json!({"checkpoint":ContentDigest::of_bytes(&checkpoint),"config":ContentDigest::of_bytes(&config)});
    SourceState::load(
        Validated::<ModelManifest>::parse(&bytes(&f["manifest"])).unwrap(),
        Arc::from(checkpoint),
        BTreeMap::from([("config".into(), Arc::from(config))]),
        StateLoadLimits::default(),
    )
    .unwrap()
}
fn identity(s: &SourceState) -> ModelIdentity {
    ModelIdentity {
        checkpoint: ContentDigest::of_bytes(s.checkpoint_bytes()),
        manifest: s.manifest_digest().clone(),
        parameters: s.manifest().parameter_digest.clone(),
        buffers: tensor_set_digest(&s.manifest().buffers).unwrap(),
        graph: ContentDigest::of_bytes(&reload_graph(s).unwrap()),
        config: s.manifest().architecture.config.clone(),
        tokenizer: None,
        system_state: None,
        runtime_artifact: None,
    }
}
fn freeze(s: &SourceState) -> FrozenReloadProbes {
    FrozenReloadProbes::freeze(
        s,
        identity(s),
        vec![vec![1.0, 2.0], vec![0.0, 0.0]],
        vec![17, 29],
    )
    .unwrap()
}
#[test]
fn ordinary_inference_still_has_kernel_limits_parity_and_reaped_child() {
    let f = Fixture::new();
    let runtime = f.runtime();
    let s = source(false, [1.0, 2.0, 3.0, 4.0]);
    let run = runtime
        .infer(
            &s,
            vec![vec![1.0, 2.0]],
            Limits::default(),
            &CancellationToken::default(),
        )
        .unwrap();
    let r = run.receipt();
    assert_eq!(r.status, "verified_inference", "{r:?}");
    assert_eq!(r.outputs, vec![vec![5.0, 11.0]]);
    assert!(r.cleanup_confirmed);
    assert!(r.cpu_ms.unwrap() > 0);
    assert!(r.peak_rss_bytes.unwrap() > 0);
    assert!(!r.outputs_eligible);
    assert_eq!(r.promotion_status, "not_evaluated");
    let runtime_dir = std::fs::read_dir(&f.root)
        .unwrap()
        .flatten()
        .map(|p| p.path())
        .find(|p| {
            p.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("runtime-")
        })
        .unwrap();
    let process: Value = serde_json::from_slice(
        &std::fs::read(runtime_dir.join(&r.job).join("process.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(process["limits"]["wall_ms"], json!(15000));
    assert_eq!(process["limits"]["cpu_seconds"], json!(10));
    assert_eq!(
        process["limits"]["address_space_bytes"],
        json!(2147483648_u64)
    );
    assert!(!Path::new(&format!("/proc/{}", process["pid"])).exists());
    let probe_dir = std::fs::read_dir(&runtime_dir)
        .unwrap()
        .flatten()
        .map(|p| p.path())
        .find(|p| {
            std::fs::read(p.join("request.json"))
                .ok()
                .is_some_and(|b| serde_json::from_slice::<Value>(&b).unwrap()["mode"] == "probe")
        })
        .unwrap();
    let probe_process: Value =
        serde_json::from_slice(&std::fs::read(probe_dir.join("process.json")).unwrap()).unwrap();
    assert_eq!(probe_process["limits"]["wall_ms"], json!(30000));
    assert_eq!(
        probe_process["limits"]["cpu_seconds"],
        process["limits"]["cpu_seconds"]
    );
    assert_eq!(
        probe_process["limits"]["address_space_bytes"],
        process["limits"]["address_space_bytes"]
    );
}
#[test]
fn actual_standalone_reload_has_fresh_reaped_child_and_no_authority() {
    for buffered in [false, true] {
        let f = Fixture::new();
        let runtime = f.runtime();
        let s = source(buffered, [1.0, 2.0, 3.0, 4.0]);
        let probes = freeze(&s);
        let root = f.publish(&s);
        drop(s); // No candidate or reference survives into verification.
        let attempt = runtime.verify_reload(
            &f.store,
            &root,
            &probes,
            Limits::default(),
            &CancellationToken::default(),
        );
        assert!(attempt.witness().is_some(), "{:?}", attempt.report());
        let r = attempt.witness().unwrap().report();
        assert!(r.cleanup_confirmed);
        assert_eq!(r.child_started, Some(true));
        assert!(!r.outputs_eligible);
        assert_eq!(r.promotion_status, "not_evaluated");
        assert!(r.cpu_ms.is_some());
        assert!(r.peak_rss_bytes.unwrap() > 0);
        assert!(r.staged_bytes > r.restored_payload_bytes);
        let child = r.child_report.as_ref().unwrap();
        assert_eq!(
            child["outputs"][0],
            if buffered {
                json!([5.5, 10.0])
            } else {
                json!([5.0, 11.0])
            }
        );
        let inventory = child["before"].as_array().unwrap();
        let group = |name| inventory.iter().find(|t| t["name"] == name).unwrap()["group"].clone();
        assert_eq!(group("w"), group("tied"));
        if buffered {
            assert_eq!(group("offset"), group("offset_alias"));
            assert_ne!(group("w"), group("offset"));
        }
        let runtime_dir = std::fs::read_dir(&f.root)
            .unwrap()
            .map(|p| p.unwrap().path())
            .find(|p| {
                p.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("runtime-")
            })
            .unwrap();
        let process: Value = serde_json::from_slice(
            &std::fs::read(
                runtime_dir
                    .join(r.job.as_ref().unwrap())
                    .join("process.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let pid = process["pid"].as_u64().unwrap();
        assert_ne!(pid, std::process::id() as u64);
        assert!(!Path::new(&format!("/proc/{pid}")).exists());
        assert!(!f.root.join("store/accepted").exists());
        assert!(serde_json::to_value(r).unwrap()["failure"].is_null());
    }
}
#[test]
fn independent_freeze_denies_self_consistent_substitute_and_invalid_identity() {
    let f = Fixture::new();
    let runtime = f.runtime();
    let s = source(false, [1.0, 2.0, 3.0, 4.0]);
    let probes = freeze(&s);
    let other = source(false, [2.0, 4.0, 6.0, 8.0]);
    let root = f.publish(&other);
    let a = runtime.verify_reload(
        &f.store,
        &root,
        &probes,
        Limits::default(),
        &CancellationToken::default(),
    );
    assert!(a.witness().is_none());
    assert!(a.report().failure.as_ref().unwrap().contains("identity"));
    assert_eq!(a.report().child_started, Some(false));
    let mut model = identity(&s);
    model.system_state = Some(ContentDigest::of_bytes(b"unsupported"));
    assert!(FrozenReloadProbes::freeze(&s, model, vec![vec![1.0, 2.0]], vec![17]).is_err());
    assert!(FrozenReloadProbes::freeze(&s, identity(&s), vec![], vec![]).is_err());
    assert!(FrozenReloadProbes::freeze(&s, identity(&s), vec![vec![1.0, 2.0]], vec![]).is_err());
}
#[test]
fn tampered_artifact_and_unknown_root_never_mint_witness() {
    let f = Fixture::new();
    let runtime = f.runtime();
    let s = source(true, [1.0, 2.0, 3.0, 4.0]);
    let probes = freeze(&s);
    let root = f.publish(&s);
    let original: Value = serde_json::from_slice(&f.store.read(&root).unwrap()).unwrap();
    for damage in [
        "alias",
        "dtype",
        "buffer",
        "config",
        "packing",
        "missing_asset",
    ] {
        let mut bad = original.clone();
        match damage {
            "alias" => bad["model"]["parameters"]["tied"]["storage_group"] = json!("untied"),
            "dtype" => bad["model"]["parameters"]["w"]["dtype"] = json!("f32"),
            "buffer" => {
                bad["model"]["buffers"]["offset_alias"]["digest"] =
                    json!(ContentDigest::of_bytes(b"bad"))
            }
            "config" => {
                bad["model"]["architecture"]["config"] = json!(ContentDigest::of_bytes(b"bad"))
            }
            "packing" => bad["model"]["format"] = json!("unknown_packing"),
            "missing_asset" => {
                bad["assets"].as_object_mut().unwrap().remove("config");
            }
            _ => unreachable!(),
        }
        let bad_root = f.store.put(bytes(&bad).as_slice()).unwrap().digest;
        let a = runtime.verify_reload(
            &f.store,
            &bad_root,
            &probes,
            Limits::default(),
            &CancellationToken::default(),
        );
        assert!(a.witness().is_none(), "{damage}");
        assert!(a.report().failure.is_some());
        assert_eq!(a.report().child_started, Some(false));
    }
    let path = f
        .root
        .join("store/blobs")
        .join(&ContentDigest::of_bytes(s.checkpoint_bytes()).as_str()[7..]);
    let mut damaged = s.checkpoint_bytes().to_vec();
    damaged.pop();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    std::fs::write(path, damaged).unwrap();
    for root in [root, ContentDigest::of_bytes(b"missing")] {
        let a = runtime.verify_reload(
            &f.store,
            &root,
            &probes,
            Limits::default(),
            &CancellationToken::default(),
        );
        assert!(a.witness().is_none());
        assert!(a.report().failure.is_some());
        assert_eq!(a.report().child_started, Some(false));
        assert!(a.report().cleanup_confirmed);
    }
}
#[test]
fn tampered_parity_alias_buffer_and_config_evidence_fails_audit() {
    let f = Fixture::new();
    let runtime = f.runtime();
    let s = source(true, [1.0, 2.0, 3.0, 4.0]);
    let probes = freeze(&s);
    let root = f.publish(&s);
    let a = runtime.verify_reload(
        &f.store,
        &root,
        &probes,
        Limits::default(),
        &CancellationToken::default(),
    );
    assert!(a.witness().is_some(), "{:?}", a.report());
    let original = a.report().child_report.as_ref().unwrap();
    probes.audit_child_report(&s, original).unwrap();
    let mut parity = original.clone();
    parity["outputs"][0][0] = json!(999.0);
    assert!(probes
        .audit_child_report(&s, &parity)
        .unwrap_err()
        .to_string()
        .contains("forward differs"));
    for field in ["group", "digest", "dtype", "shape"] {
        let mut bad = original.clone();
        let index = bad["before"]
            .as_array()
            .unwrap()
            .iter()
            .position(|v| v["name"] == "offset_alias")
            .unwrap();
        bad["before"][index][field] = json!("tampered");
        bad["after"] = bad["before"].clone();
        assert!(probes.audit_child_report(&s, &bad).is_err());
    }
    let mut bad = original.clone();
    bad["manifest"] = json!(ContentDigest::of_bytes(b"changed_config"));
    assert!(probes.audit_child_report(&s, &bad).is_err());
    // This diagnostic returns only (), never a witness from the supplied JSON.
    assert!(a.report().cleanup_confirmed);
    assert!(a.report().cpu_ms.is_some());
}
#[test]
fn cancellation_before_and_during_child_never_mints_witness() {
    let f = Fixture::new();
    let runtime = f.runtime();
    let s = source(false, [1.0, 2.0, 3.0, 4.0]);
    let probes = freeze(&s);
    let root = f.publish(&s);
    let token = CancellationToken::default();
    token.request();
    let a = runtime.verify_reload(&f.store, &root, &probes, Limits::default(), &token);
    assert!(a.witness().is_none());
    assert_eq!(a.report().child_started, Some(false));
    let token = CancellationToken::default();
    let signal = token.clone();
    let dir = f.root.clone();
    let watcher = std::thread::spawn(move || {
        for _ in 0..3000 {
            for runtime in std::fs::read_dir(&dir).unwrap().flatten() {
                if runtime
                    .file_name()
                    .to_string_lossy()
                    .starts_with("runtime-")
                {
                    for job in std::fs::read_dir(runtime.path()).unwrap().flatten() {
                        if job.path().join("reload-intent.json").exists()
                            && job.path().join("process.json").exists()
                        {
                            signal.request();
                            return true;
                        }
                    }
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        false
    });
    let a = runtime.verify_reload(&f.store, &root, &probes, Limits::default(), &token);
    assert!(watcher.join().unwrap());
    assert!(a.witness().is_none(), "{:?}", a.report());
    assert!(a.report().cleanup_confirmed);
    assert_eq!(a.report().child_started, Some(true));
    assert!(a.report().cpu_ms.is_some());
    let runtime_dir = std::fs::read_dir(&f.root)
        .unwrap()
        .flatten()
        .map(|p| p.path())
        .find(|p| {
            p.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("runtime-")
        })
        .unwrap();
    let process: Value = serde_json::from_slice(
        &std::fs::read(
            runtime_dir
                .join(a.report().job.as_ref().unwrap())
                .join("process.json"),
        )
        .unwrap(),
    )
    .unwrap();
    assert!(!Path::new(&format!("/proc/{}", process["pid"])).exists());
    let limits = Limits {
        wall_ms: 100,
        ..Limits::default()
    };
    let timed = runtime.verify_reload(
        &f.store,
        &root,
        &probes,
        limits,
        &CancellationToken::default(),
    );
    assert!(timed.witness().is_none());
    assert!(timed.report().cleanup_confirmed);
    assert_eq!(timed.report().child_started, Some(true));
    assert!(timed.report().cpu_ms.is_some());
    assert!(timed
        .report()
        .failure
        .as_ref()
        .unwrap()
        .contains("wall_limit"));
}
