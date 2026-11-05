//! Bounded native command dispatch. No Python, callbacks or study execution.
use nf_checkpoint::store::{ArtifactStore, StoreLimits};
use nf_checkpoint::tensor_file::{inspect_safetensors, TensorFileLimits};
use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use nf_core::error::{CoreError, ErrorCode, Result};
use nf_core::schemas::{Validate, Validated};
use nf_graph::inventory::{verify_inventory, InventoryLimits};
use nf_patches::replay::{replay, PatchGraph, PatchRecord, ReplayLimits};
use nf_surgeon::export::{export_history, restore_source, CoreExportLimits};
use nf_surgeon::history::{certify_history, HistoryLimits, HistoryRequest};
use nf_surgeon::transaction::SmokeSuite;
use nf_weight_ir::compilation::{compile, CompilePolicy, SourceSnapshot};
use nf_weight_ir::program::{parse_program, Program};
use nf_weight_ir::search_space::{parse_search_space, resolve_space, SearchSpace};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn error(code: ErrorCode, message: impl Into<String>) -> CoreError {
    CoreError::new(code, "", message)
}
pub fn read_file(path: &str, max_bytes: usize) -> Result<Vec<u8>> {
    use rustix::fs::{fstat, open, FileType, Mode, OFlags};
    let fd = open(
        Path::new(path),
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(|e| error(ErrorCode::Io, e.to_string()))?;
    let stat = fstat(&fd).map_err(|e| error(ErrorCode::Io, e.to_string()))?;
    if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile || stat.st_nlink != 1 {
        return Err(error(
            ErrorCode::InvalidValue,
            "input requires a regular single-link file",
        ));
    }
    if stat.st_size < 0 || stat.st_size as u64 > max_bytes as u64 {
        return Err(error(
            ErrorCode::LimitExceeded,
            "input file exceeds byte limit",
        ));
    }
    let mut bytes = Vec::new();
    std::fs::File::from(fd)
        .take(max_bytes as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| error(ErrorCode::Io, e.to_string()))?;
    if bytes.len() > max_bytes {
        return Err(error(
            ErrorCode::LimitExceeded,
            "input grew beyond byte limit",
        ));
    }
    Ok(bytes)
}
fn decode<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    let document = CanonicalDocument::parse(bytes, Limits::default())?;
    serde_json::from_value(document.value().clone())
        .map_err(|e| error(ErrorCode::InvalidValue, e.to_string()))
}
fn store(path: &Path, write: bool) -> Result<ArtifactStore> {
    let limits = StoreLimits {
        total_bytes: 536_870_912,
        max_blob_bytes: 67_108_864,
        max_entries: 4096,
    };
    if write {
        ArtifactStore::create(path, limits)
    } else {
        ArtifactStore::open_read_only(path, limits)
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CompileRequest {
    schema_version: u32,
    store: PathBuf,
    source_export: ContentDigest,
    program: Program,
    search_space: SearchSpace,
    policy: CompilePolicy,
    smoke_suite: SmokeSuite,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RestoreRequest {
    schema_version: u32,
    store: PathBuf,
    export_root: ContentDigest,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReplayRequest {
    schema_version: u32,
    store: PathBuf,
    source_export: ContentDigest,
    records: Vec<PatchRecord>,
    target: ContentDigest,
}
fn version(version: u32) -> Result<()> {
    if version == 1 {
        Ok(())
    } else {
        Err(error(
            ErrorCode::UnsupportedVersion,
            "only command request v1 supported",
        ))
    }
}
fn compile_command(bytes: &[u8], apply: bool) -> Result<Value> {
    let request: CompileRequest = decode(bytes)?;
    version(request.schema_version)?;
    let program = Validated::new(request.program)?;
    let policy = Validated::new(request.policy)?;
    let space = Validated::new(request.search_space)?;
    let smoke = Validated::new(request.smoke_suite)?;
    if smoke.digest()? != program.verification.smoke_suite {
        return Err(error(
            ErrorCode::SourceMismatch,
            "request smoke suite differs from program",
        ));
    }
    let reader = store(&request.store, false)?;
    let source = restore_source(&reader, &request.source_export, CoreExportLimits::default())?;
    let observations = source.observations();
    let inventory = verify_inventory(source.manifest(), &observations, InventoryLimits::default())?;
    let resolved = resolve_space(&space, &inventory)?;
    let mut operands = BTreeMap::new();
    let mut total = 0u64;
    for (name, operand) in &program.artifacts {
        total = total
            .checked_add(operand.artifact_bytes)
            .filter(|n| *n <= program.constraints.max_artifact_bytes && *n <= 67_108_864)
            .ok_or_else(|| {
                error(
                    ErrorCode::LimitExceeded,
                    "operand admission budget exceeded",
                )
            })?;
        operands.insert(
            name.clone(),
            Arc::from(reader.read_bounded(&operand.digest, operand.artifact_bytes)?),
        );
    }
    let plan = compile(
        &program,
        SourceSnapshot {
            manifest: source.manifest(),
            tensors: &observations,
            checkpoint: source.checkpoint_bytes(),
            assets: source.assets(),
        },
        &resolved,
        &operands,
        &policy,
        &[],
    )?;
    if !apply {
        return Ok(
            json!({"schema_version":1,"command":"ir.compile","plan":plan.summary(),"plan_digest":plan.digest(),"execution":"not_run","promotion_status":"not_evaluated"}),
        );
    }
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
    let writer = store(&request.store, true)?;
    let receipt = export_history(&writer, &history, &smoke, CoreExportLimits::default())?;
    Ok(
        json!({"schema_version":1,"command":"ir.apply","patch":record,"export":receipt,"certificate_execution_passes":1,"promotion_status":"not_evaluated"}),
    )
}
pub fn dispatch(args: &[String]) -> Result<Value> {
    if matches!(args, [group,action] if group=="core" && action=="capabilities") {
        return Ok(nf_core::capabilities::m1_manifest());
    }
    let [group, action, input] = args else {
        return Err(error(
            ErrorCode::UnsupportedOperation,
            "unsupported command; use nf --help",
        ));
    };
    // Unknown commands are denied before any file or store access.
    match (group.as_str(), action.as_str()) {
        ("json", "canonicalize") => {
            let document = CanonicalDocument::parse(
                &read_file(input, Limits::default().max_bytes)?,
                Limits::default(),
            )?;
            Ok(
                json!({"schema_version":1,"canonical":String::from_utf8_lossy(document.bytes()),"digest":document.digest(),"execution":"not_run"}),
            )
        }
        ("blob", "digest") => Ok(
            json!({"schema_version":1,"digest":ContentDigest::of_bytes(&read_file(input,67_108_864)?),"execution":"not_run"}),
        ),
        ("protocol", "negotiate") => {
            let bytes =
                nf_core::protocol::negotiate(&read_file(input, Limits::default().max_bytes)?)?;
            serde_json::from_slice(&bytes)
                .map_err(|error| CoreError::new(ErrorCode::InvalidJson, "", error.to_string()))
        }
        ("model", "inspect") => {
            let bytes = read_file(input, 67_108_864)?;
            let inventory = inspect_safetensors(
                &bytes,
                TensorFileLimits {
                    max_file_bytes: 67_108_864,
                    ..TensorFileLimits::default()
                },
            )?;
            let tensors: BTreeMap<_,_> = inventory.tensors.iter().map(|(name,entry)| (name,json!({"shape":entry.shape,"dtype":entry.dtype,"digest":entry.digest,"data_range":[entry.data_range.start,entry.data_range.end]}))).collect();
            Ok(
                json!({"schema_version":1,"command":"model.inspect","checkpoint_digest":inventory.file_digest,"header_digest":inventory.header_digest,"tensors":tensors,"execution":"not_run"}),
            )
        }
        ("ir", "validate") => {
            let program = parse_program(&read_file(input, Limits::default().max_bytes)?)?;
            Ok(
                json!({"schema_version":1,"valid":true,"program_digest":program.digest()?,"execution":"not_run"}),
            )
        }
        ("search-space", "validate" | "show") => {
            let space = parse_search_space(&read_file(input, Limits::default().max_bytes)?)?;
            Ok(
                json!({"schema_version":1,"valid":true,"search_space_digest":space.digest()?,"search_space":&*space,"quality_status":"not_evaluated","execution":"not_run"}),
            )
        }
        ("ir", "compile" | "explain" | "apply") => compile_command(
            &read_file(input, Limits::default().max_bytes)?,
            action == "apply",
        ),
        ("checkpoint", "restore") => {
            let request: RestoreRequest = decode(&read_file(input, Limits::default().max_bytes)?)?;
            version(request.schema_version)?;
            let reader = store(&request.store, false)?;
            let source =
                restore_source(&reader, &request.export_root, CoreExportLimits::default())?;
            Ok(
                json!({"schema_version":1,"command":"checkpoint.restore","export_root":request.export_root,"manifest_digest":source.manifest_digest(),"parameter_digest":source.manifest().parameter_digest,"active_reference":"unchanged","promotion_status":"not_evaluated"}),
            )
        }
        ("patch", "replay") => {
            let request: ReplayRequest = decode(&read_file(input, Limits::default().max_bytes)?)?;
            version(request.schema_version)?;
            let graph = PatchGraph::from_records(request.records.clone())?;
            let reader = store(&request.store, false)?;
            let source =
                restore_source(&reader, &request.source_export, CoreExportLimits::default())?;
            let mut blobs = BTreeMap::new();
            let mut total = 0u64;
            for record in &request.records {
                record.body.program.validate()?;
                for operand in record.body.program.artifacts.values() {
                    if blobs.contains_key(&operand.digest) {
                        continue;
                    }
                    total = total
                        .checked_add(operand.artifact_bytes)
                        .filter(|n| *n <= 67_108_864)
                        .ok_or_else(|| {
                            error(
                                ErrorCode::LimitExceeded,
                                "replay operand pool budget exceeded",
                            )
                        })?;
                    blobs.insert(
                        operand.digest.clone(),
                        Arc::from(reader.read_bounded(&operand.digest, operand.artifact_bytes)?),
                    );
                }
            }
            let result = replay(
                &graph,
                &request.target,
                &source,
                &blobs,
                ReplayLimits::default(),
            )?;
            Ok(
                json!({"schema_version":1,"command":"patch.replay","candidate":result.history.candidate_id(),"result_model":result.history.resulting_model(),"receipt":result.history.receipt(),"nodes_replayed":result.nodes_replayed,"promotion_status":"not_evaluated"}),
            )
        }
        _ => Err(error(
            ErrorCode::UnsupportedOperation,
            "unsupported command; use nf --help",
        )),
    }
}
