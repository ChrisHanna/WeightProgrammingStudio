//! Manual supplied-IR workflow. Planning is inert, bounded, and read-only.
//!
//! `freeze` persists a structurally validated contract, NOT verified approval or
//! execution authority. Legacy v1 intervention.method_protocol binds the input
//! manifest digest returned by plan; existing freezes are never reinterpreted.
//! v2 keeps the program protocol intact and separately grants the exact frozen
//! input bundle. Supplied-IR execution uses the supported final-fixture dispatcher
//! in a dedicated bounded process; it never grants strict publication or promotion.
//! Resume is explicitly recovery-only: it never restarts uncertain work.
#[path = "manual_workflow/authorization.rs"]
mod authorization;
#[path = "manual_workflow/supplied.rs"]
mod supplied;
use crate::coordinator::supervision::StopReason;
use crate::coordinator::{Ledger, LedgerLimits};
use crate::status_reports::CommittedSnapshot;
use nf_checkpoint::export::{restore_export, ExportLimits};
use nf_checkpoint::store::{ArtifactStore, StoreLimits};
use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use nf_core::error::{CoreError, ErrorCode, Result};
use nf_core::schemas::Validated;
use nf_core::study::StudyContract;
use nf_graph::inventory::{tensor_set_digest, TensorObservation};
use nf_surgeon::transaction::SmokeSuite;
use nf_weight_ir::compilation::{compile, CompilePolicy, SourceSnapshot};
use nf_weight_ir::program::Program;
use nf_weight_ir::search_space::{resolve_space, SearchSpace};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn error(code: ErrorCode, message: &str) -> CoreError {
    CoreError::new(code, "/manual_workflow", message)
}
fn canonical(value: &impl Serialize) -> Result<CanonicalDocument> {
    let raw = serde_json::to_vec(value)
        .map_err(|_| error(ErrorCode::InvalidValue, "serialization failed"))?;
    CanonicalDocument::parse(&raw, Limits::default())
}
fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    let doc = CanonicalDocument::parse(bytes, Limits::default())?;
    serde_json::from_slice(doc.bytes()).map_err(|e| error(ErrorCode::InvalidValue, &e.to_string()))
}
fn explicit(path: &Path) -> Result<()> {
    if !path.is_absolute()
        || path
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return Err(error(
            ErrorCode::InvalidValue,
            "explicit absolute paths without parent traversal required",
        ));
    }
    Ok(())
}
fn store_limits() -> StoreLimits {
    StoreLimits {
        total_bytes: 536_870_912,
        max_blob_bytes: 67_108_864,
        max_entries: 4096,
    }
}
/// Local immutable source only. Hub names are not resolved or downloaded here.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanRequest {
    pub schema_version: u32,
    pub store: PathBuf,
    pub source_export: ContentDigest,
    pub contract: StudyContract,
    pub program: Program,
    pub search_space: SearchSpace,
    pub policy: CompilePolicy,
    pub smoke_suite: SmokeSuite,
    /// v2 requires an explicit, independently compiled no-op control. Never
    /// rewrite the supplied edit to manufacture a control or break a hash cycle.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control_program: Option<Program>,
}
struct Planned {
    result: Value,
    manifest: CanonicalDocument,
    records: Vec<CanonicalDocument>,
    contract: Validated<StudyContract>,
    store: PathBuf,
    edit: nf_weight_ir::compilation::CompiledPlan,
    control: Option<nf_weight_ir::compilation::CompiledPlan>,
    smoke: Validated<SmokeSuite>,
}
fn prepare(request: PlanRequest) -> Result<Planned> {
    if !matches!(request.schema_version, 1 | 2) {
        return Err(error(
            ErrorCode::UnsupportedVersion,
            "manual workflow v1 or explicit supplied-IR v2 required",
        ));
    }
    if (request.schema_version == 2) != request.control_program.is_some() {
        return Err(error(
            ErrorCode::InvalidValue,
            "only v2 requires an explicit no-op control program",
        ));
    }
    let bundle = if request.schema_version == 2 {
        Some(canonical(&request)?)
    } else {
        None
    };
    explicit(&request.store)?;
    let contract = Validated::new(request.contract)?;
    let program = Validated::new(request.program)?;
    let space = Validated::new(request.search_space)?;
    let policy = Validated::new(request.policy)?;
    let smoke = Validated::new(request.smoke_suite)?;
    let source = &contract.source;
    let model = &program.base_model;
    if model.checkpoint != source.checkpoint
        || model.manifest != source.manifest
        || model.parameter_digest != source.parameters
        || model.architecture_graph != source.graph
        || model.config != source.config
        || model.tokenizer != source.tokenizer
        || source.system_state.is_some()
        || source.runtime_artifact.is_some()
        || program.objective.contract != contract.contract_key
        || program.objective.objective != contract.objective.id
        || space.contract != contract.contract_key
        || space.objective != contract.objective.id
        || space.digest()? != contract.intervention.transformation_space
        || program.verification.portfolio != contract.portfolio_digest
        || canonical(&program.effects)?.digest() != contract.intervention.scope
        || canonical(&policy.permissions)?.bytes() != canonical(&contract.permissions)?.bytes()
        || smoke.digest()? != program.verification.smoke_suite
        || !policy.allowed_extra_gyms.is_empty()
        || !contract.intervention.allowed_engines.iter().any(|e| {
            e.id == policy.backend.id
                && e.version == policy.backend.version
                && Some(e.config.clone()) == canonical(&policy.backend).ok().map(|d| d.digest())
        })
    {
        return Err(error(
            ErrorCode::SourceMismatch,
            "supplied IR/source/space/permissions/portfolio/backend differ from contract",
        ));
    }
    let mut seen = BTreeSet::new();
    for gym in &program.verification.gyms {
        let binding = contract
            .portfolio
            .bindings
            .iter()
            .find(|b| b.id == gym.binding)
            .ok_or_else(|| {
                error(
                    ErrorCode::SourceMismatch,
                    "IR gym binding absent from contract",
                )
            })?;
        let manifest = &contract.portfolio.gyms[&binding.gym];
        let role = serde_json::to_value(gym.role).unwrap();
        if !seen.insert(gym.binding.clone())
            || gym.gym != binding.gym
            || gym.validation != binding.validation
            || gym.scorer != manifest.scorer
            || gym.split != binding.splits.selection.digest()?
            || !binding
                .roles
                .iter()
                .any(|r| serde_json::to_value(r).unwrap() == role)
        {
            return Err(error(
                ErrorCode::SourceMismatch,
                "IR gym identity/role/selection split differs",
            ));
        }
    }
    if contract
        .portfolio
        .bindings
        .iter()
        .any(|b| b.required && !seen.contains(&b.id))
    {
        return Err(error(
            ErrorCode::SourceMismatch,
            "required portfolio binding missing from supplied IR",
        ));
    }
    for quality in &space.constraints.quality_limits {
        let metric = contract
            .portfolio
            .bindings
            .iter()
            .find(|b| b.id == quality.binding)
            .and_then(|b| b.metrics.iter().find(|m| m.metric == quality.metric))
            .ok_or_else(|| {
                error(
                    ErrorCode::SourceMismatch,
                    "space quality binding/metric absent from contract",
                )
            })?;
        if quality.baseline != source.checkpoint
            || metric
                .threshold
                .max_regression
                .is_some_and(|max| quality.max_regression > max)
        {
            return Err(error(
                ErrorCode::SourceMismatch,
                "space quality baseline or regression weakens contract",
            ));
        }
    }
    // Inert tensor/asset decoding only; no architecture factory, smoke execution,
    // runtime qualification, model worker, provider or download is invoked.
    let reader = ArtifactStore::open_read_only(&request.store, store_limits())?;
    let state = restore_export(&reader, &request.source_export, ExportLimits::default())?;
    reader.read_bounded(&source.graph, 1_048_576)?;
    if tensor_set_digest(&state.model().buffers)? != source.buffers {
        return Err(error(
            ErrorCode::SourceMismatch,
            "source buffer identity differs",
        ));
    }
    let observations = state
        .inventory()
        .entries()
        .iter()
        .map(|(name, entry)| {
            Ok(TensorObservation {
                name,
                role: entry.role,
                shape: &entry.shape,
                dtype: entry.dtype,
                bytes: state.tensor_bytes(name).ok_or_else(|| {
                    error(ErrorCode::IntegrityMismatch, "source tensor bytes missing")
                })?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let resolved = resolve_space(&space, state.inventory())?;
    let mut operands = BTreeMap::new();
    let mut total = 0u64;
    for (name, operand) in &program.artifacts {
        total = total
            .checked_add(operand.artifact_bytes)
            .filter(|n| *n <= program.constraints.max_artifact_bytes && *n <= 67_108_864)
            .ok_or_else(|| error(ErrorCode::LimitExceeded, "operand pool bound exceeded"))?;
        operands.insert(
            name.clone(),
            Arc::from(reader.read_bounded(&operand.digest, operand.artifact_bytes)?),
        );
    }
    let compiled = compile(
        &program,
        SourceSnapshot {
            manifest: state.model(),
            tensors: &observations,
            checkpoint: state.checkpoint(),
            assets: state.assets(),
        },
        &resolved,
        &operands,
        &policy,
        &[],
    )?;
    if compiled.summary().allocations.estimated_working_bytes > contract.limits.host_memory_bytes {
        return Err(error(
            ErrorCode::LimitExceeded,
            "compiled temporary memory exceeds study limit",
        ));
    }
    let control = request.control_program.map(Validated::new).transpose()?;
    let compiled_control = control
        .as_ref()
        .map(|control| {
            if !control.operations.is_empty()
                || control.base_model != program.base_model
                || canonical(&control.objective)?.bytes() != canonical(&program.objective)?.bytes()
            {
                return Err(error(
                    ErrorCode::SourceMismatch,
                    "explicit control must be a no-op on the same source/objective/protocol",
                ));
            }
            compile(
                control,
                SourceSnapshot {
                    manifest: state.model(),
                    tensors: &observations,
                    checkpoint: state.checkpoint(),
                    assets: state.assets(),
                },
                &resolved,
                &operands,
                &policy,
                &[],
            )
        })
        .transpose()?;
    let mut manifest_value = json!({"schema_version":request.schema_version,
        "protocol":if bundle.is_some() { "nf.manual.supplied_ir.v2" } else { "nf.manual.supplied_ir.v1" },
        "source_export":request.source_export,"program":program.digest()?,
        "search_space":space.digest()?,"policy":canonical(&*policy)?.digest(),
        "smoke_suite":smoke.digest()?,"compiled_plan":compiled.digest()});
    let id = contract.digest()?;
    if let Some(bundle) = &bundle {
        manifest_value["input_bundle"] = json!(bundle.digest());
        manifest_value["frozen_contract"] = json!(id);
        manifest_value["method_protocol"] = json!(program.objective.experiment_protocol);
        manifest_value["control_program"] = json!(control.as_ref().unwrap().digest()?);
        manifest_value["compiled_control"] = json!(compiled_control.as_ref().unwrap().digest());
    }
    let manifest = canonical(&manifest_value)?;
    let binding_matches = if bundle.is_some() {
        contract.intervention.method_protocol == program.objective.experiment_protocol
    } else {
        contract.intervention.method_protocol == manifest.digest()
    };
    let plan_digest =
        canonical(&json!({"contract":id,"manifest":manifest.digest(),"store":request.store}))?
            .digest();
    let result = json!({"schema_version":1,"valid":true,"validation_scope":"offline_static_only",
        "contract_digest":id,"plan_digest":plan_digest,"manifest_digest":manifest.digest(),
        "manifest":manifest.value(),"compiled_plan":compiled.summary(),
        "contract_binding_matches":binding_matches,"approval_reference":contract.approval_receipt,
        "operator_binding":if bundle.is_some() { json!({"revision":2,"manifest":manifest.digest(),
            "source_export":request.source_export,"source_identity":canonical(&contract.source)?.digest()}) } else { Value::Null },
        "authorization_verified":false,"execution_ready":false,"execution":"not_run",
        "scientific_acceptance":"not_evaluated",
        "execution_graph":["preflight","profile","baseline","diagnose","design","execute","evaluate","search","verify","learn"],
        "limits":contract.limits,"portfolio_digest":contract.portfolio_digest,
        "coverage_gaps":["gym validation and approval references not resolved by static planning",
            "source/no-op/best-known baselines not executed","runtime and isolation not qualified"],
        "supported_dispatch":["nf.native.complete_fixture.v1", supplied::OPERATION],
        "authorization_source":"root-owned /etc/neural-foundry/manual-execution registry",
        "unavailable_capabilities":["general_model_dispatch_outside_supported_linear_cpu_profile",
            "strict_fixture_publication","authoritative_promotion"]});
    let mut records = vec![
        canonical(&*program)?,
        canonical(&*space)?,
        canonical(&*policy)?,
        canonical(&*smoke)?,
    ];
    if let Some(control) = &control {
        records.push(canonical(&**control)?);
    }
    if let Some(bundle) = bundle {
        // Standalone exports embed the source manifest; the coordinator resolves
        // that exact manifest by CAS identity. Freeze retains it, plan never writes.
        records.push(canonical(&**state.model())?);
        records.push(bundle);
    }
    Ok(Planned {
        result,
        manifest,
        records,
        contract,
        store: request.store,
        edit: compiled,
        control: compiled_control,
        smoke,
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FreezeRequest {
    plan: PlanRequest,
    ledger: PathBuf,
    expected_plan_digest: ContentDigest,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExistingRequest {
    schema_version: u32,
    store: PathBuf,
    ledger: PathBuf,
    contract_digest: ContentDigest,
    #[serde(default)]
    authorization: Option<ContentDigest>,
    #[serde(default)]
    workers: Vec<ContentDigest>,
}
impl ExistingRequest {
    fn validate(&self) -> Result<()> {
        if self.schema_version != 1 {
            return Err(error(
                ErrorCode::UnsupportedVersion,
                "manual workflow v1 required",
            ));
        }
        explicit(&self.store)?;
        explicit(&self.ledger)?;
        if self.workers.len() > 32
            || self.workers.iter().collect::<BTreeSet<_>>().len() != self.workers.len()
        {
            return Err(error(
                ErrorCode::LimitExceeded,
                "at most 32 distinct recovery worker IDs required",
            ));
        }
        Ok(())
    }
    fn snapshot(&self) -> Result<CommittedSnapshot> {
        CommittedSnapshot::capture(&self.ledger.join("events.sqlite3"), &self.contract_digest)
    }
    fn recovery_workers(&self) -> Result<Vec<ContentDigest>> {
        if !self.workers.is_empty() {
            return Ok(self.workers.clone());
        }
        // A crash can occur before run returns its worker ID. Discover retained
        // native intents from the committed snapshot, not caller scratch files.
        let value: Value = serde_json::from_str(&self.snapshot()?.render_json())
            .map_err(|_| error(ErrorCode::IntegrityMismatch, "snapshot JSON differs"))?;
        let intents = value["supervision"]["worker_intents"]
            .as_array()
            .ok_or_else(|| {
                error(
                    ErrorCode::IntegrityMismatch,
                    "native intent inventory missing",
                )
            })?;
        let mut ids = Vec::new();
        for intent in intents {
            if intent["record"]["contract"] == json!(self.contract_digest) {
                ids.push(
                    serde_json::from_value(intent["record"]["id"].clone()).map_err(|_| {
                        error(
                            ErrorCode::IntegrityMismatch,
                            "native intent identity missing",
                        )
                    })?,
                );
                if ids.len() > 4096 {
                    return Err(error(
                        ErrorCode::LimitExceeded,
                        "native recovery inventory bound",
                    ));
                }
            }
        }
        Ok(ids)
    }
    fn open(&self) -> Result<Ledger> {
        // Pre-existing, validated snapshot required before any writable/migrating
        // open. Never initialize a misspelled store/ledger for stop or recovery.
        self.snapshot()?;
        // Stop/recovery must still work when source artifacts are missing or
        // damaged. They address a frozen ledger identity, never dispatch work.
        Ledger::open(&self.ledger, LedgerLimits::default())
    }
}
/// One shared bounded wire service for native CLI and PyO3. Unknown operations
/// fail before parsing a path. This API is host/controller-side, never gym IPC.
pub fn dispatch(operation: &str, bytes: &[u8]) -> Result<Value> {
    dispatch_cancellable(
        operation,
        bytes,
        &nf_surgeon::cancellation::CancellationToken::default(),
    )
}
/// Controller cancellation is separate from execution authority and persists a
/// stop request. It can stop approved work, never grant or broaden that work.
pub fn dispatch_cancellable(
    operation: &str,
    bytes: &[u8],
    cancel: &nf_surgeon::cancellation::CancellationToken,
) -> Result<Value> {
    match operation {
        // Explicit opt-in runtime observation, separate from inert planning.
        // Run this with the reviewed CLI executable before freezing its profile.
        "runtime-profile" => {
            #[derive(Deserialize)]
            #[serde(deny_unknown_fields)]
            struct ProfileRequest {
                schema_version: u32,
            }
            let request: ProfileRequest = decode(bytes)?;
            if request.schema_version != 1 {
                return Err(error(
                    ErrorCode::UnsupportedVersion,
                    "profile request v1 required",
                ));
            }
            Ok(
                json!({"runtime":crate::coordinator::final_fixture::runtime_profile()?,
                "scope":"observed_supported_linear_cpu_runtime_only","execution_authority":false}),
            )
        }
        "authorize" | "revoke" => authorization::operator(operation, bytes),
        "serve-authority" => authorization::serve(bytes),
        // Safe to call directly: independently verifies OS authority, exact
        // executable and irreversible hard limits; not a caller approval flag.
        "execute-supplied" => authorization::execute_supplied(bytes),
        "plan" | "validate" => Ok(prepare(decode(bytes)?)?.result),
        "freeze" => {
            let request: FreezeRequest = decode(bytes)?;
            explicit(&request.ledger)?;
            let planned = prepare(request.plan)?;
            if planned.result["plan_digest"] != json!(request.expected_plan_digest) {
                return Err(error(
                    ErrorCode::SourceMismatch,
                    "expected plan digest differs; replan changed inputs",
                ));
            }
            if planned.result["contract_binding_matches"] != true {
                return Err(error(ErrorCode::SourceMismatch, "contract protocol differs from explicit revision binding; frozen inputs are never rewritten"));
            }
            // Store already exists; create enables writes but never invents a root.
            let writer = ArtifactStore::create(&planned.store, store_limits())?;
            for record in &planned.records {
                writer.put(record.bytes())?;
            }
            let receipt = writer.put(planned.manifest.bytes())?;
            let mut ledger = Ledger::open(&request.ledger, LedgerLimits::default())?;
            let id = ledger.freeze(&planned.contract)?;
            Ok(
                json!({"schema_version":1,"contract_digest":id,"manifest_digest":receipt.digest,
                "persistence":"frozen_contract_only","authorization_verified":false,
                "execution":"not_run","scientific_acceptance":"not_evaluated"}),
            )
        }
        "status" | "report" | "stop" | "resume" | "run" => {
            let request: ExistingRequest = decode(bytes)?;
            request.validate()?;
            if operation == "run" {
                return authorization::run(&request, cancel);
            }
            if operation == "status" || operation == "report" {
                let snapshot = request.snapshot()?;
                return if operation == "status" {
                    serde_json::from_str(&snapshot.render_json())
                        .map_err(|_| error(ErrorCode::IntegrityMismatch, "snapshot JSON differs"))
                } else {
                    Ok(
                        json!({"json":snapshot.render_json(),"markdown":snapshot.render_markdown()?}),
                    )
                };
            }
            let mut ledger = request.open()?;
            if operation == "stop" {
                let stop = ledger.request_stop(
                    &request.contract_digest,
                    canonical(&("nf.manual.owner_stop.v1", &request.contract_digest))?.digest(),
                    StopReason::OwnerStop,
                )?;
                return Ok(
                    json!({"schema_version":1,"stop_request":stop,"request_status":"persisted",
                    "cleanup_status":"not_confirmed_by_request","execution":"not_dispatched"}),
                );
            }
            let workers = request.recovery_workers()?;
            let lease =
                ledger.acquire(&request.contract_digest, "nf.manual.recovery.v1", 300_000)?;
            // Precheck the entire selected set before recovery mutates any worker.
            let recovery = (|| -> Result<Vec<Value>> {
                for id in &workers {
                    if ledger.worker_state(id)?.contract != request.contract_digest {
                        return Err(error(
                            ErrorCode::PermissionDenied,
                            "worker belongs to another contract",
                        ));
                    }
                }
                // The coordinator owns final-fixture intent identities and
                // unknown charges. Reconcile only after validating the entire
                // selected set; never re-run an interrupted supplied edit.
                if ledger
                    .contract(&request.contract_digest)?
                    .runtime
                    .backend
                    .id
                    == "nf.final.arithmetic_mixed_five"
                    && request
                        .ledger
                        .parent()
                        .is_some_and(|p| p.join("supplied-sandbox").exists())
                {
                    let runtime = crate::sandbox::QualifiedDocker::qualify(
                        true,
                        &request.ledger.parent().unwrap().join("supplied-sandbox"),
                    )?;
                    ledger.reconcile_arithmetic_final(&lease, &runtime)?;
                } else {
                    ledger.reconcile_fixture_final(&lease)?;
                }
                let mut reports = Vec::new();
                for id in &workers {
                    let report = ledger.recover_native(&lease, id)?;
                    reports.push(json!({"worker":id,"report":report}));
                }
                Ok(reports)
            })();
            let released = ledger.release(&lease);
            let reports = recovery?;
            released?;
            Ok(
                json!({"schema_version":1,"mode":"recovery_only","recovered":reports,
                "inventory_scope":if request.workers.is_empty() {
                    "all retained native intents for exact frozen contract; not final-fixture/reload cleanup proof"
                } else { "caller-selected native intents; not whole-study cleanup proof" },
                "stop_request":ledger.stop_record(&request.contract_digest)?,
                "budgets":ledger.budget_snapshot(&request.contract_digest)?,
                "execution":"not_resumed","costs_and_exposure":"retained",
                "reason":"recovery never redispatches uncertain work; run requires a live scoped operator grant"}),
            )
        }
        _ => Err(error(
            ErrorCode::UnsupportedOperation,
            "unsupported manual workflow operation",
        )),
    }
}
