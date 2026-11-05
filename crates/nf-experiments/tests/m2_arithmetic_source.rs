//! T043 artifact integration only, explicitly NOT joined study acceptance.
use nf_checkpoint::store::{ArtifactStore, StoreLimits};
use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use nf_core::schemas::{ModelManifest, Validated};
use nf_experiments::arithmetic::{build, Mode};
use nf_experiments::coordinator::final_fixture::{arithmetic_source, graph_bytes};
use nf_graph::inventory::{verify_inventory, InventoryLimits};
use nf_surgeon::cancellation::CancellationToken;
use nf_surgeon::export::{export_candidate, restore_source, CoreExportLimits};
use nf_surgeon::transaction::{execute_cancellable, ExecutionLimits, SmokeSuite, SourceState};
use nf_weight_ir::compilation::{compile, CompilePolicy, SourceSnapshot};
use nf_weight_ir::program::parse_program;
use nf_weight_ir::search_space::{parse_search_space, resolve_space};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

fn bytes(value: &impl Serialize) -> Vec<u8> {
    CanonicalDocument::parse(&serde_json::to_vec(value).unwrap(), Limits::default())
        .unwrap()
        .bytes()
        .to_vec()
}
struct Store {
    path: PathBuf,
    cas: ArtifactStore,
}
impl Store {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = PathBuf::from(format!(
            "/tmp/nf-t043-joined-source-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let cas = ArtifactStore::create(
            &path,
            StoreLimits {
                total_bytes: 1_048_576,
                max_blob_bytes: 65_536,
                max_entries: 128,
            },
        )
        .unwrap();
        Self { path, cas }
    }
    fn publish(&self, source: &SourceState) {
        self.cas
            .put(bytes(&**source.manifest()).as_slice())
            .unwrap();
        self.cas.put(source.checkpoint_bytes()).unwrap();
        for asset in source.assets().values() {
            self.cas.put(asset.as_ref()).unwrap();
        }
    }
}
impl Drop for Store {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[test]
fn m2_arithmetic_source_supplied_ir_export_resolves_repair_harm_and_noop() {
    // Real compiler, direct edit and standalone CAS export. No Python DTO or
    // linear fixture decision is used as evidence for this arithmetic candidate.
    for (factor, harmful) in [(2.0, false), (2.0, true), (1.0, false)] {
        let fixture = build(Mode::CorruptSum).unwrap();
        let source = &fixture.source;
        let original = source.checkpoint_bytes().to_vec();
        let id = arithmetic_source::identity(source).unwrap();
        let store = Store::new();
        store.publish(source);
        let template: Value = serde_json::from_str(include_str!(
            "../../../python/neural_foundry/data/m1_fixture.json"
        ))
        .unwrap();
        let mut program = template["program"].clone();
        let mut space = template["space"].clone();
        let mut policy = template["policy"].clone();
        let base = source.binding(id.graph.clone());
        let smoke: Validated<SmokeSuite> = Validated::parse(&bytes(&json!({
            "schema_version":1,"adapter":"nf.arithmetic_relu.v1",
            "probes":[{"input":[8.0,2.0],"expected":null,
                "absolute_tolerance":0.0,"max_abs_output":1000.0}]
        })))
        .unwrap();
        let mut regions = vec![json!({"tensor":"heads.weight","start":[0,0],"shape":[1,1]})];
        if harmful {
            regions.push(json!({"tensor":"heads.weight","start":[1,0],"shape":[1,1]}));
        }
        space["base_model"] = json!(base);
        space["sites"] = json!(["hidden.weight", "heads.weight"].map(|name| json!({
            "selector":{"type":"tensor","name":name},"role":"parameter",
            "regions":{"type":"whole_tensor"},"allowed_axes":[0,1]})));
        space["constraints"]["max_changed_parameters"] = json!(regions.len());
        let space = parse_search_space(&bytes(&space)).unwrap();
        program["base_model"] = json!(base);
        program["objective"]["search_space"] = json!(space.digest().unwrap());
        program["verification"]["smoke_suite"] = json!(smoke.digest().unwrap());
        program["operations"] = json!(regions
            .iter()
            .enumerate()
            .map(|(i, region)| json!({
            "id":format!("op{i}"),"depends_on":[],"operation":{"type":"scale",
            "target":region,"factor":if i==0 {factor} else {0.0}}}))
            .collect::<Vec<_>>());
        program["preconditions"] = json!(source
            .manifest()
            .parameters
            .iter()
            .map(|(name, spec)| json!({
            "tensor":name,"role":"parameter","shape":spec.shape,"dtype":"f64",
            "before_digest":spec.digest,"storage_group":spec.storage_group}))
            .collect::<Vec<_>>());
        program["constraints"]["max_changed_parameters"] = json!(regions.len());
        program["effects"]["reads"] = json!(regions);
        program["effects"]["writes"] = json!(regions);
        program["lineage"]["method"]["changed_parameters"] = json!(regions.len());
        policy["source"] = json!(base);
        policy["search_space"] = json!(space.digest().unwrap());
        policy["verification"] = program["verification"].clone();
        let program = parse_program(&bytes(&program)).unwrap();
        let policy: Validated<CompilePolicy> = Validated::parse(&bytes(&policy)).unwrap();
        let observations = source.observations();
        let inventory =
            verify_inventory(source.manifest(), &observations, InventoryLimits::default()).unwrap();
        let resolved = resolve_space(&space, &inventory).unwrap();
        let plan = compile(
            &program,
            SourceSnapshot {
                manifest: source.manifest(),
                tensors: &observations,
                checkpoint: source.checkpoint_bytes(),
                assets: source.assets(),
            },
            &resolved,
            &BTreeMap::new(),
            &policy,
            &[],
        )
        .unwrap();
        let candidate = execute_cancellable(
            &plan,
            source,
            &smoke,
            ExecutionLimits::default(),
            &CancellationToken::default(),
        )
        .unwrap();
        let export =
            export_candidate(&store.cas, &candidate, &smoke, CoreExportLimits::default()).unwrap();
        let edited =
            restore_source(&store.cas, &export.export_root, CoreExportLimits::default()).unwrap();
        let edited_id = arithmetic_source::identity(&edited).unwrap();
        store.publish(&edited);
        assert_eq!(export.exported_model, edited_id.manifest);
        assert_eq!(source.checkpoint_bytes(), original);
        assert_eq!(
            edited.tensor("heads.weight").unwrap().f64_values().unwrap()[0],
            factor * 0.5
        );
        if harmful {
            assert_eq!(
                edited.tensor("heads.weight").unwrap().f64_values().unwrap()[6],
                0.0
            );
        }
        if factor == 1.0 {
            assert_eq!(edited_id.parameters, id.parameters);
        } else {
            assert_ne!(edited_id.parameters, id.parameters);
        }
        drop(edited);
        drop(candidate);
        drop(fixture);
        let reopened = arithmetic_source::resolve(&store.cas, &edited_id).unwrap();
        assert_eq!(reopened.identity(), &edited_id);
        reopened
            .state()
            .verify_smoke(&smoke, ExecutionLimits::default())
            .unwrap();
        let output = reopened
            .state()
            .infer_arithmetic_fixture(&[8.0, 2.0], ExecutionLimits::default())
            .unwrap();
        assert_eq!(
            output,
            vec![
                if factor == 1.0 { 6.0 } else { 10.0 },
                if harmful { -2.0 } else { 6.0 },
                8.0,
                2.0,
                6.0
            ]
        );
        assert_ne!(edited_id.graph, ContentDigest::of_bytes(graph_bytes()));
    }
}

#[test]
fn m2_arithmetic_source_rejects_each_forged_axis_and_missing_payload() {
    let source = build(Mode::Reference).unwrap();
    let store = Store::new();
    store.publish(&source.source);
    let id = arithmetic_source::identity(&source.source).unwrap();
    for axis in [
        "checkpoint",
        "manifest",
        "parameters",
        "buffers",
        "graph",
        "config",
        "tokenizer",
        "system_state",
        "runtime_artifact",
    ] {
        let mut forged = json!(id);
        forged[axis] = json!(ContentDigest::of_bytes(b"not-the-frozen-model"));
        let forged = serde_json::from_value(forged).unwrap();
        assert!(
            arithmetic_source::resolve(&store.cas, &forged).is_err(),
            "{axis}"
        );
    }
    std::fs::remove_file(store.path.join("blobs").join(&id.checkpoint.as_str()[7..])).unwrap();
    assert!(arithmetic_source::resolve(&store.cas, &id).is_err());
}

#[test]
fn m2_arithmetic_source_rejects_valid_cas_unregistered_graph_and_extra_assets() {
    let fixture = build(Mode::EmptySum).unwrap();
    let store = Store::new();
    store.publish(&fixture.source);
    let id = arithmetic_source::identity(&fixture.source).unwrap();
    let mut manifest: ModelManifest =
        serde_json::from_slice(&bytes(&**fixture.source.manifest())).unwrap();
    let wrong = store.cas.put(b"{}".as_slice()).unwrap().digest;
    manifest.assets.insert("graph".into(), wrong.clone());
    let mut forged = id.clone();
    forged.graph = wrong;
    forged.manifest = store.cas.put(bytes(&manifest).as_slice()).unwrap().digest;
    assert!(arithmetic_source::resolve(&store.cas, &forged).is_err());
    manifest
        .assets
        .insert("unexpected".into(), ContentDigest::of_bytes(b"missing"));
    forged.manifest = store.cas.put(bytes(&manifest).as_slice()).unwrap().digest;
    let error = arithmetic_source::resolve(&store.cas, &forged)
        .err()
        .unwrap();
    assert!(error.message.contains("fanout"));
}
