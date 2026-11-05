//! Self-contained native M1 example: no Python, model download or training.
use nf_checkpoint::export::{publish_export, ExportLimits};
use nf_checkpoint::store::{ArtifactStore, StoreLimits};
use nf_core::canonical::{CanonicalDocument, Limits};
use nf_core::error::{CoreError, ErrorCode, Result};
use nf_core::schemas::{ModelManifest, Validated};
use nf_graph::inventory::{verify_inventory, InventoryLimits};
use nf_patches::replay::{replay, PatchGraph, PatchRecord, ReplayLimits};
use nf_surgeon::export::{export_history, restore_source, CoreExportLimits};
use nf_surgeon::history::{certify_history, HistoryLimits, HistoryRequest};
use nf_surgeon::transaction::{SmokeSuite, SourceState, StateLoadLimits};
use nf_weight_ir::compilation::{compile, CompilePolicy, SourceSnapshot};
use nf_weight_ir::program::Program;
use nf_weight_ir::search_space::{resolve_space, SearchSpace};
use serde::Deserialize;
use serde_json::json;
use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    schema_version: u32,
    manifest: ModelManifest,
    checkpoint_hex: String,
    assets_hex: BTreeMap<String, String>,
    program: Program,
    space: SearchSpace,
    policy: CompilePolicy,
    smoke_suite: SmokeSuite,
}
fn fail(message: &str) -> CoreError {
    CoreError::new(ErrorCode::IntegrityMismatch, "", message)
}
fn decode_hex(value: &str) -> Result<Arc<[u8]>> {
    if value.len() > 16384
        || !value.len().is_multiple_of(2)
        || !value.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Err(fail("bounded hex fixture required"));
    }
    let bytes = value
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect::<Vec<_>>();
    Ok(Arc::from(bytes))
}
fn run(fixture_path: &Path, output: &Path) -> Result<serde_json::Value> {
    let mut bytes = Vec::new();
    std::fs::File::open(fixture_path)
        .map_err(|error| CoreError::new(ErrorCode::Io, "", error.to_string()))?
        .take(1_048_577)
        .read_to_end(&mut bytes)
        .map_err(|error| CoreError::new(ErrorCode::Io, "", error.to_string()))?;
    let document = CanonicalDocument::parse(&bytes, Limits::default())?;
    let fixture: Fixture = serde_json::from_slice(document.bytes())
        .map_err(|error| CoreError::new(ErrorCode::InvalidValue, "", error.to_string()))?;
    if fixture.schema_version != 1 {
        return Err(fail("fixture version mismatch"));
    }
    let assets = fixture
        .assets_hex
        .iter()
        .map(|(name, data)| Ok((name.clone(), decode_hex(data)?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let source = SourceState::load(
        Validated::new(fixture.manifest)?,
        decode_hex(&fixture.checkpoint_hex)?,
        assets,
        StateLoadLimits::default(),
    )?;
    let original = source.tensor_bytes("w")?;
    let observations = source.observations();
    let inventory = verify_inventory(source.manifest(), &observations, InventoryLimits::default())?;
    let program = Validated::new(fixture.program)?;
    let policy = Validated::new(fixture.policy)?;
    let space = Validated::new(fixture.space)?;
    let smoke = Validated::new(fixture.smoke_suite)?;
    let resolved = resolve_space(&space, &inventory)?;
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
    )?;
    let history = certify_history(
        HistoryRequest {
            plan: &plan,
            source: &source,
            smoke: &smoke,
            space: &resolved,
            parent: None,
            expected_candidate: None,
        },
        HistoryLimits::default(),
    )?;
    let record = PatchRecord::from_verified(vec![], &plan, &policy, &space, &smoke, &history)?;
    let graph = PatchGraph::from_records(vec![record.clone()])?;
    let replayed = replay(
        &graph,
        &record.id,
        &source,
        &BTreeMap::new(),
        ReplayLimits::default(),
    )?;
    if replayed.history.candidate_id() != history.candidate_id() {
        return Err(fail("exact replay identity mismatch"));
    }
    let store = ArtifactStore::create(
        output,
        StoreLimits {
            total_bytes: 2_097_152,
            max_blob_bytes: 1_048_576,
            max_entries: 128,
        },
    )?;
    let source_root = publish_export(
        &store,
        source.manifest(),
        Arc::from(source.checkpoint_bytes()),
        source.assets().clone(),
        ExportLimits::default(),
    )?;
    let exported = export_history(&store, &history, &smoke, CoreExportLimits::default())?;
    let loaded = restore_source(&store, &exported.export_root, CoreExportLimits::default())?;
    let restored = restore_source(&store, &source_root, CoreExportLimits::default())?;
    let expected: Vec<u8> = [2.0f64, 4.0, 6.0, 8.0]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    if loaded.tensor_bytes("w")? != expected
        || restored.tensor_bytes("w")? != original
        || source.tensor_bytes("w")? != original
        || loaded.manifest().parameter_digest != history.resulting_model().parameter_digest
    {
        return Err(fail("export/reload/restore values differ"));
    }
    let record_bytes = serde_json::to_vec(&record)
        .map_err(|error| CoreError::new(ErrorCode::InvalidValue, "", error.to_string()))?;
    let record_blob =
        store.put(CanonicalDocument::parse(&record_bytes, Limits::default())?.bytes())?;
    Ok(
        json!({"schema_version":1,"example":"m1_offline_direct","source_export":source_root,"export":exported,"patch_blob":record_blob.digest,"patch":record.id,"replayed_candidate":replayed.history.candidate_id(),"changed_parameters":history.receipt().changed_parameters,"source_preserved":true,"restored_source":restored.manifest().parameter_digest,"values":[2,4,6,8],"capabilities":nf_core::capabilities::m1_manifest(),"promotion_status":"not_evaluated"}),
    )
}
fn main() {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 2 {
        eprintln!("Usage: offline_direct FIXTURE.json ABSOLUTE_STORE_DIRECTORY");
        std::process::exit(2);
    }
    match run(Path::new(&args[0]), Path::new(&args[1])) {
        Ok(result) => println!("{result}"),
        Err(error) => {
            eprintln!("{}", serde_json::to_string(&error).unwrap());
            std::process::exit(1);
        }
    }
}
