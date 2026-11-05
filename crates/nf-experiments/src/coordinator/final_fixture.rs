//! Coordinator-owned, finite 2x2 f64 linear CPU fixture acceptance. Not T039.
//! No callbacks, imported observations, receipt DTOs or scientific generalization.
//! Authority is the live ledger plus this fixed executor, not a serialized flag.
use super::acceptance::{
    mark_verified_execution, model_identity, Check, Decision, Evidence, Report, Status,
};
use super::budgets::{OutcomeKind, Request, Resources, Settlement, Usage, WorkKind};
use super::*;
use crate::statistics::{self, Arm, Contrast, Data, Observation, Plan, Protocol};
use nf_checkpoint::store::ArtifactStore;
use nf_core::schemas::{DType, ModelManifest, ToolKind};
use nf_core::study::*;
use nf_graph::inventory::tensor_set_digest;
use nf_surgeon::cancellation::CancellationToken;
use nf_surgeon::export::{export_candidate, restore_source, CoreExportLimits};
use nf_surgeon::transaction::{
    execute_cancellable, ExecutionLimits, SmokeSuite, SourceState, StateLoadLimits,
};
use nf_weight_ir::compilation::CompiledPlan;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Instant;
pub mod arithmetic_final;
/// Artifact resolution only; does not broaden the linear execution authority.
pub mod arithmetic_source;
mod bounds;
mod native_process;
mod supervised;
#[cfg(test)]
mod tests;

// Debug-only deterministic boundary testing. Actions can interrupt/pause work,
// never skip a fence or manufacture evidence. No environment-controlled hooks.
#[cfg(debug_assertions)]
thread_local! {
    static FAULT: std::cell::RefCell<Option<(&'static str, Box<dyn FnOnce() -> Result<()>>)>> = const { std::cell::RefCell::new(None) };
}
#[cfg(debug_assertions)]
#[doc(hidden)]
pub fn inject_fixture_fault(point: &'static str, action: impl FnOnce() -> Result<()> + 'static) {
    FAULT.with(|slot| *slot.borrow_mut() = Some((point, Box::new(action))));
}
fn fault(point: &str) -> Result<()> {
    #[cfg(debug_assertions)]
    {
        let action = FAULT.with(|slot| {
            let mut slot = slot.borrow_mut();
            if slot.as_ref().is_some_and(|(target, _)| *target == point) {
                slot.take()
            } else {
                None
            }
        });
        if let Some((_, action)) = action {
            return action();
        }
    }
    let _ = point;
    Ok(())
}

pub(super) const MIGRATION_V6: &str = r#"
CREATE TABLE final_fixture_records(contract TEXT NOT NULL REFERENCES contracts(digest),kind TEXT NOT NULL,body BLOB NOT NULL,digest TEXT NOT NULL,PRIMARY KEY(contract,kind));
CREATE TRIGGER final_fixture_no_update BEFORE UPDATE ON final_fixture_records BEGIN SELECT RAISE(ABORT,'immutable final evidence');END;
CREATE TRIGGER final_fixture_no_delete BEFORE DELETE ON final_fixture_records BEGIN SELECT RAISE(ABORT,'retained final evidence');END;
INSERT INTO budget_migrations VALUES(6,(SELECT value FROM meta WHERE key='schema'));
PRAGMA user_version=6;
"#;
pub(super) fn schema_v6() -> ContentDigest {
    ContentDigest::of_bytes(
        &[
            SCHEMA.as_bytes(),
            budgets::MIGRATION_V2.as_bytes(),
            supervision::MIGRATION_V3.as_bytes(),
            splits::MIGRATION_V4.as_bytes(),
            exposure::MIGRATION_V5.as_bytes(),
            MIGRATION_V6.as_bytes(),
        ]
        .concat(),
    )
}
/// Register these exact bytes in the CAS as the supported architecture graph.
pub fn graph_bytes() -> &'static [u8] {
    br#"{"factory":"fixture.cpu","operation":"dense_linear","shape":[2,2],"version":1}"#
}
/// Native scorer/runtime code identity, used in approved fixture gym manifests.
pub fn profile_id() -> ContentDigest {
    ContentDigest::of_bytes(
        &[
            include_bytes!("final_fixture.rs").as_slice(),
            include_bytes!("final_fixture/bounds.rs").as_slice(),
            include_bytes!("final_fixture/native_process.rs").as_slice(),
            supervised::profile().as_str().as_bytes(),
        ]
        .concat(),
    )
}
fn executable_identity() -> Result<ContentDigest> {
    use std::io::Read;
    static ID: std::sync::OnceLock<ContentDigest> = std::sync::OnceLock::new();
    if let Some(id) = ID.get() {
        return Ok(id.clone());
    }
    let mut file = std::fs::File::open("/proc/self/exe")
        .map_err(|e| failure(ErrorCode::Io, &e.to_string()))?;
    let mut buffer = [0u8; 65536];
    let mut chunks = Vec::new();
    let mut total = 0usize;
    loop {
        let mut n = 0;
        while n < buffer.len() {
            match file.read(&mut buffer[n..]) {
                Ok(0) => break,
                Ok(count) => n += count,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(failure(ErrorCode::Io, &e.to_string())),
            }
        }
        if n == 0 {
            break;
        }
        total += n;
        if total > 268_435_456 {
            return Err(deny("bounded native executable identity required"));
        }
        chunks.push(ContentDigest::of_bytes(&buffer[..n]));
    }
    let id = hash(&("nf.native.executable.chunked_read.v1", total, chunks))?;
    let _ = ID.set(id.clone());
    Ok(id)
}
/// Observed local runtime, not a caller-declared environment identity.
/// This supported static native profile is bound to the executable and boot.
pub fn runtime_profile() -> Result<RuntimeProfile> {
    use std::io::Read;
    let mut cpu = String::new();
    std::fs::File::open("/proc/cpuinfo")
        .and_then(|f| f.take(65536).read_to_string(&mut cpu))
        .map_err(|e| failure(ErrorCode::Io, &e.to_string()))?;
    let model = cpu
        .lines()
        .find(|l| l.starts_with("model name"))
        .ok_or_else(|| deny("CPU model identity unavailable"))?;
    let boot = std::fs::read_to_string("/proc/sys/kernel/random/boot_id")
        .map_err(|e| failure(ErrorCode::Io, &e.to_string()))?;
    let physical = hash(&(std::env::consts::ARCH, model, boot.trim()))?;
    let code_revision = hash(&(profile_id(), executable_identity()?))?;
    let kernel = std::fs::read_to_string("/proc/sys/kernel/osrelease")
        .map_err(|e| failure(ErrorCode::Io, &e.to_string()))?;
    let driver = ContentDigest::of_bytes(kernel.trim().as_bytes());
    let dependencies = ContentDigest::of_bytes(
        &[
            include_bytes!("../../../nf-surgeon/src/transaction.rs").as_slice(),
            include_bytes!("../../../nf-surgeon/src/export.rs").as_slice(),
            include_bytes!("../../../nf-weight-ir/src/compilation.rs").as_slice(),
        ]
        .concat(),
    );
    Ok(RuntimeProfile {
        schema_version: 1,
        device: Device::Cpu,
        physical_device: physical.clone(),
        backend: nf_core::schemas::FactoryRef {
            id: "fixture.cpu".into(),
            version: "1".into(),
            config: profile_id(),
        },
        precision: DType::F64,
        driver: driver.clone(),
        dependencies: dependencies.clone(),
        code_revision: code_revision.clone(),
        deterministic: true,
        batch_size: 1,
        input_length: 2,
        max_output_length: 2,
        length_unit: LengthUnit::TensorElements,
        concurrency: 1,
        cache: CacheMode::Disabled,
        synchronization: Synchronization::CpuMonotonic,
        state_reset_protocol: profile_id(),
        environment: hash(&(physical, driver, dependencies, code_revision))?,
    })
}
fn hash(v: &impl Serialize) -> Result<ContentDigest> {
    Ok(ContentDigest::of_bytes(&canonical(v)?))
}
fn deny(text: &str) -> CoreError {
    failure(ErrorCode::UnsupportedCapability, text)
}
fn row(name: &str, status: Status, reason: impl Into<String>) -> Check {
    Check {
        name: name.into(),
        candidate: None,
        binding: None,
        metric: None,
        required: true,
        status,
        reason: reason.into(),
    }
}
fn record(
    tx: &Transaction<'_>,
    lease: &Lease,
    kind: &str,
    value: &impl Serialize,
) -> Result<ContentDigest> {
    let body = canonical(value)?;
    if body.len() > 1_048_576 {
        return Err(deny("bounded final evidence record required"));
    }
    let id = ContentDigest::of_bytes(&body);
    tx.execute(
        "INSERT INTO final_fixture_records VALUES(?1,?2,?3,?4)",
        params![lease.contract.as_str(), kind, body, id.as_str()],
    )
    .map_err(sql)?;
    Ok(id)
}
fn read(
    connection: &Connection,
    contract: &ContentDigest,
    kind: &str,
) -> Result<Option<(ContentDigest, Value)>> {
    let r: Option<(Vec<u8>, String)> = connection
        .query_row(
            "SELECT body,digest FROM final_fixture_records WHERE contract=?1 AND kind=?2",
            params![contract.as_str(), kind],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()
        .map_err(sql)?;
    r.map(|(b, d)| {
        let doc = CanonicalDocument::parse(&b, Limits::default())?;
        if doc.digest().as_str() != d || doc.bytes() != b {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "final evidence content changed",
            ));
        }
        Ok((d.parse()?, doc.value().clone()))
    })
    .transpose()
}
fn resolve(store: &ArtifactStore, model: &ModelIdentity) -> Result<SourceState> {
    if model.graph != ContentDigest::of_bytes(graph_bytes())
        || store.read_bounded(&model.graph, 1024)? != graph_bytes()
        || model.system_state.is_some()
        || model.runtime_artifact.is_some()
        || model.tokenizer.is_some()
    {
        return Err(deny("unsupported graph/tokenizer/external state"));
    }
    let m = Validated::<ModelManifest>::parse(&store.read_bounded(&model.manifest, 65536)?)?;
    // Reject metadata fanout before allocating/reading asset payloads.
    if m.assets.len() != 2
        || !m.assets.contains_key("config")
        || !m.assets.contains_key("checkpoint")
        || m.parameters.len() != 2
        || !m.buffers.is_empty()
    {
        return Err(deny("bounded fixture manifest required"));
    }
    let mut assets = BTreeMap::new();
    for (name, id) in &m.assets {
        if name != "checkpoint" {
            assets.insert(name.clone(), Arc::from(store.read_bounded(id, 2)?));
        }
    }
    let source = SourceState::load(
        m,
        Arc::from(store.read_bounded(&model.checkpoint, 4096)?),
        assets,
        bounds::LOAD,
    )?;
    let m = source.manifest();
    if m.architecture.id != "fixture.cpu"
        || m.architecture.version != "1"
        || !m.buffers.is_empty()
        || m.parameters
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
            != BTreeSet::from(["tied", "w"])
        || m.parameters["w"].shape != vec![2, 2]
        || m.parameters["w"].dtype != DType::F64
        || m.parameters["w"] != m.parameters["tied"]
        || source.assets().len() != 1
        || source.assets().get("config").map(AsRef::as_ref) != Some(b"{}".as_slice())
        || source.manifest_digest() != &model.manifest
        || m.parameter_digest != model.parameters
        || tensor_set_digest(&m.buffers)? != model.buffers
        || m.architecture.config != model.config
    {
        return Err(deny(
            "artifact identity or supported fixture layout mismatch",
        ));
    }
    source.verify_integrity()?;
    Ok(source)
}
fn identity(source: &SourceState) -> Result<ModelIdentity> {
    Ok(ModelIdentity {
        checkpoint: ContentDigest::of_bytes(source.checkpoint_bytes()),
        manifest: source.manifest_digest().clone(),
        parameters: source.manifest().parameter_digest.clone(),
        buffers: tensor_set_digest(&source.manifest().buffers)?,
        graph: ContentDigest::of_bytes(graph_bytes()),
        config: source.manifest().architecture.config.clone(),
        tokenizer: None,
        system_state: None,
        runtime_artifact: None,
    })
}
fn supported(c: &StudyContract, edit: &CompiledPlan, noop: &CompiledPlan) -> Result<()> {
    let p = edit.program();
    if canonical(&c.runtime)? != canonical(&runtime_profile()?)?
        || c.portfolio.independent.scope != ClaimScope::FixtureOnly
        || c.portfolio.independent.isolation != Isolation::TrustedFixtureBroker
        || c.statistics.estimator != Estimator::DeterministicExact
        || c.seeds.len() != 1
        || c.statistics.repeats != 1
        || c.runtime.device != Device::Cpu
        || c.runtime.backend.id != "fixture.cpu"
        || c.runtime.backend.version != "1"
        || c.runtime.backend.config != profile_id()
        || c.runtime.precision != DType::F64
        || !c.runtime.deterministic
        || c.runtime.batch_size != 1
        || c.runtime.concurrency != 1
        || c.runtime.cache != CacheMode::Disabled
        || c.runtime.synchronization != Synchronization::CpuMonotonic
        || c.runtime.input_length != 2
        || c.runtime.max_output_length != 2
        || c.objective.kind == ObjectiveKind::CausalClaim
        || c.intervention.allow_buffer_write
        || c.intervention.allow_graph_change
        || p.objective.contract != c.contract_key
        || p.objective.objective != c.objective.id
        || p.objective.search_space != c.intervention.transformation_space
        || p.objective.experiment_protocol != c.intervention.method_protocol
        || hash(&p.effects)? != c.intervention.scope
        || p.verification.portfolio != c.portfolio_digest
        || edit.backend().id != "nf.native_cpu"
        || noop.backend().id != "nf.native_cpu"
        || !c.intervention.allowed_engines.iter().any(|e| {
            e.id == edit.backend().id
                && e.version == edit.backend().version
                && Some(e.config.clone()) == hash(edit.backend()).ok()
        })
        || !noop.program().operations.is_empty()
        || !p.lineage.parent_candidates.is_empty()
        || p.base_model != noop.program().base_model
    {
        return Err(deny(
            "unsupported fixture profile or contract-bound edit/scope/backend",
        ));
    }
    c.permissions.check_tool(ToolKind::DirectEdit)?;
    p.lineage.method.require_fully_gradient_free()?;
    if c.baselines.iter().any(|b| {
        b.enabled
            && (!matches!(
                b.kind,
                BaselineKind::Original | BaselineKind::NoOp | BaselineKind::BestKnown
            ) || b.backend.id != "fixture.cpu"
                || b.backend.version != "1"
                || b.backend.config != profile_id())
    }) {
        return Err(deny("additional enabled baseline is unsupported"));
    }
    for b in &c.portfolio.bindings {
        if !b.schedule.final_required {
            continue;
        }
        let g = &c.portfolio.gyms[&b.gym];
        if g.execution_trust != ExecutionTrust::TrustedFixture
            || g.adapter.id != "nf.final.linear_exact"
            || g.adapter.version != "1"
            || g.adapter.config != profile_id()
            || g.code != profile_id()
            || g.scorer != profile_id()
            || g.oracle != profile_id()
            || g.dependencies != profile_id()
            || g.input_schema != profile_id()
            || g.output_schema != profile_id()
            || g.scoring_protocol != profile_id()
            || b.validation != profile_id()
            || g.validation != profile_id()
            || b.splits.holdout.items.len() > 32
            || b.schedule.holdout_episodes as usize != b.splits.holdout.items.len()
        {
            return Err(deny(
                "unregistered scorer/oracle/validation or unsupported episode schedule",
            ));
        }
        for m in &b.metrics {
            let schema = &g.metrics[&m.metric];
            match m.metric.as_str() {
                "quality" | "safety"
                    if schema.unit == MetricUnit::Fraction
                        && schema.direction == Direction::HigherIsBetter
                        && m.statistic == Statistic::Exact
                        && m.measurement.is_none() => {}
                "latency" | "memory" => {
                    let q = m
                        .measurement
                        .as_ref()
                        .ok_or_else(|| deny("resource protocol absent"))?;
                    if q.statistic != Statistic::Maximum
                        || m.statistic != Statistic::Maximum
                        || q.repetitions != 3
                        || q.warmup != 1
                        || q.order != MeasurementOrder::PairedInterleaved
                        || q.synchronization != Synchronization::CpuMonotonic
                        || q.timer.id != "nf.final.monotonic"
                        || q.timer.version != "1"
                        || q.timer.config != profile_id()
                        || schema.direction != Direction::LowerIsBetter
                        || m.threshold.min_gain.is_some()
                        || m.threshold.max_regression.is_some()
                        || (m.metric == "latency"
                            && (q.scope != MeasurementScope::ForwardLatency
                                || schema.unit != MetricUnit::Milliseconds))
                        || (m.metric == "memory"
                            && (q.scope != MeasurementScope::ProcessRss
                                || schema.unit != MetricUnit::Bytes
                                || m.threshold.min_gain.is_some()
                                || m.threshold.max_regression.is_some()))
                    {
                        return Err(deny("resource scope/statistic differs; process-wide RSS cannot prove per-arm gain"));
                    }
                }
                _ => return Err(deny("unsupported frozen metric")),
            }
        }
    }
    Ok(())
}
fn cpu_ns() -> u64 {
    let t = rustix::time::clock_gettime(rustix::time::ClockId::ProcessCPUTime);
    t.tv_sec as u64 * 1_000_000_000 + t.tv_nsec as u64
}
fn rss() -> Result<u64> {
    // Whole controller high-water mark, never a per-model allocation estimate.
    let mut u = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: getrusage writes a valid rusage output pointer.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, u.as_mut_ptr()) } != 0 {
        return Err(deny("kernel RSS unavailable"));
    }
    // SAFETY: successful getrusage initialized u.
    Ok(unsafe { u.assume_init() }.ru_maxrss.max(0) as u64 * 1024)
}
/// Eligibility covers the operation/family deadline captured by the reservation,
/// not only the lease. A missing completion receipt is deliberately fail-closed.
fn check_completion(
    lease: &Lease,
    deadline: i64,
    cpu_start: u64,
    resources: &Resources,
    cancel: Option<&CancellationToken>,
) -> Result<files::Moment> {
    let now = files::moment()?;
    if now.boot != lease.boot
        || now.ms >= deadline
        || cpu_ns().saturating_sub(cpu_start).div_ceil(1_000_000) > resources.cpu_ms
        || rss()? > resources.host_bytes
    {
        return Err(failure(
            ErrorCode::QuotaExceeded,
            "finalization crossed native resource/deadline envelope",
        ));
    }
    if let Some(cancel) = cancel {
        cancel.check()?;
    }
    Ok(now)
}
fn commit_decision(
    tx: Transaction<'_>,
    lease: &Lease,
    deadline: i64,
    cpu: u64,
    resources: &Resources,
    cancel: &CancellationToken,
) -> Result<()> {
    check_completion(lease, deadline, cpu, resources, Some(cancel))?;
    ensure_live(lease)?;
    tx.commit().map_err(sql)
}
struct FixtureInputs<'a> {
    store: &'a ArtifactStore,
    edit: &'a CompiledPlan,
    noop: &'a CompiledPlan,
    smoke: &'a Validated<SmokeSuite>,
    cancel: &'a CancellationToken,
}
struct Executed {
    plan: Plan,
    data: Data,
    panel: ContentDigest,
    receipt: Value,
    evaluations: u64,
}
impl Ledger {
    fn final_fence(&self, lease: &Lease, cancel: &CancellationToken) -> Result<()> {
        cancel.check()?;
        let tx = self.connection.unchecked_transaction().map_err(sql)?;
        check_lease(&tx, &self.store, lease, &files::moment()?)?;
        if state(&tx, &lease.contract)?.stage != Stage::Verify
            || supervision::stopped(&tx, &lease.contract)?
        {
            return Err(deny("final execution stopped or wrong phase"));
        }
        Ok(())
    }
    fn allocation_fence(
        &self,
        lease: &Lease,
        reservation: &budgets::Reservation,
        cancel: &CancellationToken,
        cpu_start: u64,
    ) -> Result<()> {
        self.final_fence(lease, cancel)?;
        let tx = self.connection.unchecked_transaction().map_err(sql)?;
        let allocation = budgets::native_allocation(&tx, &self.store, lease, reservation)?;
        bounds::hard_memory_limit(allocation.resources.host_bytes)?;
        if rss()? > allocation.resources.host_bytes
            || cpu_ns()
                .saturating_sub(cpu_start)
                .div_ceil(1_000_000)
                .saturating_add(supervised::cpu_ms(&self.connection, &lease.contract)?)
                >= allocation.resources.cpu_ms
        {
            return Err(failure(
                ErrorCode::QuotaExceeded,
                "native resource envelope exhausted before allocation",
            ));
        }
        Ok(())
    }
    /// Capability prerequisite for callers that require T038's strict deadline
    /// through authoritative durable publication. SQLite FULL synchronous mode,
    /// busy timeouts, progress/commit hooks and process kill timers cannot bound
    /// filesystem sync or scheduler latency, nor revoke a completed late write.
    /// No strict execution is qualified on this backend. Never silently fall
    /// back from this error to `run_fixture_final` and claim strict compliance.
    pub fn require_strict_fixture_publication(&self) -> Result<()> {
        Err(deny("strict durable-publication wall/CPU enforcement unavailable on SQLite/OS; native fixture evidence is not T038 sign-off"))
    }
    /// A one-shot final study for the fixed local fixture. Runs actual edit,
    /// no-op apply/export/core reload, four matched arms, scoring and statistics.
    /// Launch the dedicated native caller under a finite hard Linux RLIMIT_AS
    /// no greater than the contract host-memory cap (e.g. via prlimit). Unlimited
    /// callers fail closed. This dedicated-process API temporarily narrows the
    /// controller soft AS limit to reserve guardian+worker capacity; it must not
    /// be invoked in a shared controller. Unknown cleanup retains that limit.
    /// A started attempt is never silently retried after interruption/feedback.
    /// Positive fixture evidence is NOT strict T038 wall/CPU enforcement; callers
    /// needing it must first pass `require_strict_fixture_publication`.
    pub fn run_fixture_final(
        &mut self,
        lease: &Lease,
        store: &ArtifactStore,
        edit: &CompiledPlan,
        noop: &CompiledPlan,
        smoke: &Validated<SmokeSuite>,
        cancel: &CancellationToken,
    ) -> Result<Decision> {
        self.final_fence(lease, cancel)?;
        let contract = self.contract(&lease.contract)?;
        bounds::preflight(self, &contract, edit, noop, smoke)?;
        supported(&contract, edit, noop)?;
        let mut memory_envelope =
            supervised::MemoryEnvelope::enter(contract.limits.host_memory_bytes)?;
        let had_intent = read(&self.connection, &lease.contract, "intent")?.is_some();
        let old_cap = bounds::cap_database(&self.connection, self.limits.max_database_bytes)?;
        let cleanup_cpu = cpu_ns();
        let result = self.run_fixture_final_bounded(lease, store, edit, noop, smoke, cancel);
        // Synchronous native work has unwound; there are no surviving workers.
        // Reconcile ONLY an attempt started by this invocation, never an old
        // interrupted attempt whose resources this invocation cannot certify.
        let cleanup = if result.is_err() && !had_intent {
            self.cleanup_fixture_abort(lease, cleanup_cpu)
        } else {
            Ok(())
        };
        self.connection
            .pragma_update(None, "max_page_count", old_cap)
            .map_err(sql)?;
        cleanup?;
        if supervised::cleanup_known(&self.connection, &lease.contract)? {
            memory_envelope.cleanup_confirmed();
        }
        result
    }
    /// Reconcile an interrupted synchronous fixture after acquiring a new lease.
    /// A matching live process may still be using its allocation: never refund
    /// it merely because its lease expired. Reboot/PID-reuse/disappearance prove
    /// the old native executor is gone; unknown costs are charged in full.
    pub fn reconcile_fixture_final(&mut self, lease: &Lease) -> Result<()> {
        let tx = self.connection.unchecked_transaction().map_err(sql)?;
        check_lease(&tx, &self.store, lease, &files::moment()?)?;
        drop(tx);
        // A provisional completion is not final settlement. A crash during its
        // observation tail must retain the allocation until native reconciliation.
        if read(&self.connection, &lease.contract, "commit_observation")?.is_some() {
            return Ok(());
        }
        let Some((_, intent)) = read(&self.connection, &lease.contract, "intent")? else {
            return Ok(());
        };
        let owner: native_process::Owner = serde_json::from_value(intent["native_process"].clone())
            .map_err(|_| deny("native process recovery identity missing"))?;
        if !owner.gone()? {
            return Err(deny("native process still live; allocation retained"));
        }
        self.cleanup_fixture_abort(lease, cpu_ns())
    }
    fn cleanup_fixture_abort(&mut self, lease: &Lease, cpu_start: u64) -> Result<()> {
        let Some((intent, value)) = read(&self.connection, &lease.contract, "intent")? else {
            return Ok(());
        };
        if value["profile"] != json!(profile_id()) {
            return Err(deny("nonlinear intent requires its own attested recovery; never release via linear cleanup"));
        }
        let id: ContentDigest = serde_json::from_value(value["reservation"].clone())
            .map_err(|_| deny("invalid native reservation"))?;
        if budgets::native_released(&self.connection, &id)? {
            return Ok(());
        }
        // Coordinator death/disappearance does not prove a separate guardian or
        // worker was reaped. Unknown supervised cleanup retains the entire hold.
        if !supervised::cleanup_known(&self.connection, &lease.contract)? {
            return Ok(());
        }
        let reservation = self.recover_reservation(lease, &id)?;
        let used = cpu_ns().saturating_sub(cpu_start).div_ceil(1_000_000);
        let ceiling = value["resources"]["cpu_ms"].as_u64().unwrap_or(0);
        self.settle(
            lease,
            &reservation,
            Settlement {
                operation: hash(&(&intent, "native_abort_cleanup"))?,
                outcome: OutcomeKind::Incomplete,
                usage: Usage {
                    attempts: Some(1),
                    cpu_ms: (used > ceiling).then_some(used),
                    ..Default::default()
                },
                resources_released: true,
                evidence: intent,
            },
        )
    }
    fn run_fixture_final_bounded(
        &mut self,
        lease: &Lease,
        store: &ArtifactStore,
        edit: &CompiledPlan,
        noop: &CompiledPlan,
        smoke: &Validated<SmokeSuite>,
        cancel: &CancellationToken,
    ) -> Result<Decision> {
        let start = Instant::now();
        let cpu = cpu_ns();
        let contract = self.contract(&lease.contract)?;
        if read(&self.connection, &lease.contract, "intent")?.is_some()
            || self.frozen_final_panel(&lease.contract)?.is_some()
        {
            return Err(failure(
                ErrorCode::RevisionConflict,
                "final attempt/panel already exists; no adaptive retry",
            ));
        }
        let evaluations = bounds::calls(&contract, smoke.probes.len())?;
        let resources = Resources {
            cpu_ms: 30_000,
            attempts: 1,
            evaluations,
            storage_bytes: bounds::STORAGE,
            host_bytes: contract.limits.host_memory_bytes,
            workers: 1,
            ..Default::default()
        };
        let store_id = self.store.clone();
        let native_process = native_process::Owner::current()?;
        let reservation = self.reserve_with(
            lease,
            Request {
                schema_version: 1,
                operation: hash(&("final_fixture", &lease.contract))?,
                attempt: 0,
                parent: None,
                kind: WorkKind::Verification,
                resources: resources.clone(),
                wall_ms: 30_000.min(contract.limits.wall_ms),
                external: None,
            },
            |tx, reservation, allocation_deadline| {
                fault("admission_before_intent")?;
                record(tx, lease, "intent",
                    &json!({"contract":lease.contract,"store":store_id,"epoch":lease.epoch,"boot":lease.boot,"deadline":lease.deadline,"profile":profile_id(),"supervised_profile":supervised::profile(),"worker_image":supervised::image_digest(),"edit":edit.digest(),"noop":noop.digest(),"smoke":smoke.digest()?,"reservation":reservation.id(),"resources":resources,"hard_address_space_bytes":bounds::hard_memory_limit(resources.host_bytes)?,"native_process":native_process,"operation_deadline":allocation_deadline.min(lease.deadline),"status":"started_not_complete"}))?;
                fault("intent_before_admission_commit")
            },
        )?;
        fault("after_admission_commit")?;
        let (intent, intent_value) = read(&self.connection, &lease.contract, "intent")?
            .ok_or_else(|| deny("atomic native intent missing"))?;
        let deadline = intent_value["operation_deadline"]
            .as_i64()
            .ok_or_else(|| deny("native operation deadline missing"))?;
        let before_storage = store.used_bytes()?;
        let result = self.execute_fixture(
            lease,
            &reservation,
            FixtureInputs {
                store,
                edit,
                noop,
                smoke,
                cancel,
            },
            cpu,
        );
        fault("after_native_execution")?;
        if !supervised::cleanup_known(&self.connection, &lease.contract)? {
            return Err(deny(
                "supervised cleanup unknown; full reservation retained",
            ));
        }
        let used_cpu_ms = cpu_ns()
            .saturating_sub(cpu)
            .div_ceil(1_000_000)
            .saturating_add(supervised::cpu_ms(&self.connection, &lease.contract)?);
        let wall_ms = start.elapsed().as_millis() as u64;
        // If authority was lost, leave the durable intent and reservation open.
        self.final_fence(lease, &CancellationToken::default())?;
        let mut report;
        let execution;
        let count;
        match result {
            Ok(e) => {
                let analysis = e.plan.analyze(&canonical(&e.data)?, cancel)?;
                report = self
                    .acceptance_gate(
                        lease,
                        Some(Evidence {
                            plan: &e.plan,
                            analysis: &analysis,
                        }),
                    )?
                    .into_report();
                execution = json!({"status":"complete","intent":intent,"store":self.store,"contract":lease.contract,"epoch":lease.epoch,"boot":lease.boot,"panel":e.panel,"plan":e.plan.record()?,"analysis":analysis.id(),"analysis_report":analysis.report(),"data":e.data,"receipt":e.receipt,"cpu_ms":used_cpu_ms,"wall_ms":wall_ms,"artifact_storage_delta_bytes":store.used_bytes()?.saturating_sub(before_storage),"cost_scope":"process_CPU_and_storage_charged_at_full_reserved_ceiling_including_analysis_and_persistence;measured_execution_sample_excludes_analysis_and_persistence;atomic_completion_and_settlement_fenced_after_serialization_precommit_not_fsync_completion","peak_controller_rss_bytes":rss()?});
                let execution_id = hash(&execution)?;
                mark_verified_execution(&mut report.checks, &contract, &execution_id);
                count = Some(e.evaluations);
            }
            Err(e) => {
                report = self.acceptance_gate(lease, None)?.into_report();
                for c in &mut report.checks {
                    if c.status != Status::Passed {
                        c.status = Status::Incomplete;
                    }
                }
                report.checks.push(row(
                    "interrupted_execution",
                    Status::Incomplete,
                    format!("{:?}: {}", e.code, e.message),
                ));
                execution = json!({"status":"incomplete","intent":intent,"store":self.store,"contract":lease.contract,"epoch":lease.epoch,"error":format!("{:?}",e.code),"cpu_ms":used_cpu_ms,"wall_ms":wall_ms});
                count = None;
            }
        }
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sql)?;
        check_lease(&tx, &self.store, lease, &files::moment()?)?;
        let execution_id = record(&tx, lease, "execution", &execution)?;
        fault("execution_before_commit")?;
        ensure_live(lease)?;
        tx.commit().map_err(sql)?;
        fault("after_execution_commit")?;
        let settlement = Settlement {
            operation: hash(&(&intent, "settled"))?,
            outcome: if count.is_some() {
                OutcomeKind::Completed
            } else {
                OutcomeKind::Incomplete
            },
            usage: Usage {
                cpu_ms: if used_cpu_ms > resources.cpu_ms {
                    Some(used_cpu_ms)
                } else {
                    None
                },
                gpu_ms: Some(0),
                attempts: Some(1),
                evaluations: count,
                reasoning_tokens: Some(0),
                teacher_tokens: Some(0),
                spend_microusd: Some(0),
                storage_bytes: None,
            },
            resources_released: true,
            evidence: execution_id,
        };
        let clean = {
            let tx = self.connection.unchecked_transaction().map_err(sql)?;
            budgets::contract_clean_except(&tx, &lease.contract, Some(&reservation))?
        };
        report.checks.push(row("resource_budget_and_cleanup", if clean && used_cpu_ms <= resources.cpu_ms && wall_ms <= 30_000.min(contract.limits.wall_ms) && rss()? <= resources.host_bytes { Status::Passed } else { Status::Incomplete }, "whole-process CPU and wall checks; kernel-enforced hard RLIMIT_AS through publication; RSS is measurement only; SQLite pre-write cap"));
        report.status = [
            Status::Failed,
            Status::Incomplete,
            Status::Missing,
            Status::Inconclusive,
        ]
        .into_iter()
        .find(|s| report.checks.iter().any(|c| c.required && c.status == *s))
        .unwrap_or(Status::Passed);
        report.accepted = report.status == Status::Passed;
        report.promotion_authorized = false;
        report.scope =
            "trusted_native_linear_fixture_only_not_generalization_or_strict_T038_or_T039".into();
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sql)?;
        let now = files::moment()?;
        check_lease(&tx, &self.store, lease, &now)?;
        let halted: bool = tx
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM budget_faults WHERE family=?1)",
                [contract.study_family.as_str()],
                |r| r.get(0),
            )
            .map_err(sql)?;
        // budget_epochs.wall_ms is a wall-clock origin, NOT a duration.
        let (boot, origin): (String, i64) = tx
            .query_row(
                "SELECT boot,start_ms FROM budget_epochs WHERE family=?1",
                [contract.study_family.as_str()],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .map_err(sql)?;
        let final_cpu_ms = cpu_ns()
            .saturating_sub(cpu)
            .div_ceil(1_000_000)
            .saturating_add(supervised::cpu_ms(&tx, &lease.contract)?);
        let final_wall_ms = start.elapsed().as_millis() as u64;
        let live = !cancel.is_cancelled()
            && !supervision::stopped(&tx, &lease.contract)?
            && state(&tx, &lease.contract)?.stage == Stage::Verify
            && !halted
            && budgets::contract_clean_except(&tx, &lease.contract, Some(&reservation))?
            && supervision::all_cleanup_confirmed(&tx, &lease.contract)?
            && now.boot == boot
            && now.ms < origin.saturating_add(contract.limits.wall_ms as i64)
            && final_cpu_ms <= resources.cpu_ms
            && final_wall_ms <= 30_000.min(contract.limits.wall_ms);
        report.checks.push(row("final_fence_cleanup_and_accounting",if live {Status::Passed}else{Status::Incomplete},format!("serialized stop/fence/cleanup/accounting check; process_cpu_ms={final_cpu_ms}, wall_ms={final_wall_ms}; cancellation linearizes at last precommit check; authoritative settlement+completion linearizes at its last fenced precommit check, not at fsync return")));
        if !live {
            if report.status != Status::Failed {
                report.status = Status::Incomplete;
            }
            report.accepted = false;
        }
        let decision = Decision::from_report(report)?;
        let decision_id = record(&tx, lease, "decision", decision.report())?;
        // Stop is serialized by this IMMEDIATE transaction. Cancellation before
        // this final acquire-load wins and rolls the prospective decision back;
        // cancellation after it is still checked by authoritative publication.
        // Serialization/SQLite insertion above are deliberately BEFORE this point.
        fault("decision_before_commit")?;
        commit_decision(tx, lease, deadline, cpu, &resources, cancel)?;
        fault("after_decision_commit")?;
        // Provisional completion is NOT authority and releases no admission.
        // The postcommit observation and final settlement share the LAST commit;
        // its physical fsync tail is explicitly not a hard deadline guarantee.
        let store_id = self.store.clone();
        let supervised_cpu = supervised::cpu_ms(&self.connection, &lease.contract)?;
        let mut controller_resources = resources.clone();
        controller_resources.cpu_ms = controller_resources.cpu_ms.saturating_sub(supervised_cpu);
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sql)?;
        {
            fault("settlement_before_completion")?;
            let completed =
                check_completion(lease, deadline, cpu, &controller_resources, Some(cancel))?;
            record(
                &tx,
                lease,
                "completion",
                &json!({"decision":decision_id,
                "boot":completed.boot,"completed_ms":completed.ms,"deadline_ms":deadline,
                "cpu_ms":cpu_ns().saturating_sub(cpu).div_ceil(1_000_000).saturating_add(supervised_cpu),
                "linearization":"last_fenced_precommit_check_not_fsync_completion"}),
            )?;
            fault("completion_before_publication_fence")?;
            check_lease(&tx, &store_id, lease, &files::moment()?)?;
            if supervision::stopped(&tx, &lease.contract)?
                || state(&tx, &lease.contract)?.stage != Stage::Verify
            {
                return Err(deny("publication stopped or wrong phase"));
            }
            check_completion(lease, deadline, cpu, &controller_resources, Some(cancel))?;
            bounds::hard_memory_limit(resources.host_bytes)?;
            fault("publication_after_fence")?;
            ensure_live(lease)?;
            tx.commit().map_err(sql)?;
        }
        fault("after_publication_commit")?;
        // Sample AFTER SQLite's FULL synchronous commit returned. Never infer
        // this observation from a persisted precommit timestamp on recovery.
        let observed = check_completion(lease, deadline, cpu, &controller_resources, Some(cancel))?;
        let observed_cpu = cpu_ns()
            .saturating_sub(cpu)
            .div_ceil(1_000_000)
            .saturating_add(supervised_cpu);
        self.finish_budget_with(lease, &reservation, settlement, false, |tx| {
            check_lease(tx, &store_id, lease, &files::moment()?)?;
            if supervision::stopped(&tx, &lease.contract)?
                || state(&tx, &lease.contract)?.stage != Stage::Verify
            {
                return Err(deny("commit observation stopped or wrong phase"));
            }
            let (completion_id, _) = read(&tx, &lease.contract, "completion")?
                .ok_or_else(|| deny("committed completion missing"))?;
            record(
                &tx,
                lease,
                "commit_observation",
                &json!({
                    "completion":completion_id,"decision":decision_id,"epoch":lease.epoch,
                    "boot":observed.boot,"observed_ms":observed.ms,"deadline_ms":deadline,
                    "cpu_ms":observed_cpu,"semantics":"postcommit_observation_not_hard_enforcement"
                }),
            )?;
            fault("observation_before_commit")?;
            check_completion(lease, deadline, cpu, &controller_resources, Some(cancel))?;
            ensure_live(lease)?;
            fault("observation_after_fence")?;
            Ok(())
        })?;
        fault("after_observation_commit")?;
        self.files.verify(self.limits.max_database_bytes)?;
        Ok(decision)
    }
    fn execute_fixture(
        &mut self,
        lease: &Lease,
        reservation: &budgets::Reservation,
        inputs: FixtureInputs<'_>,
        cpu_start: u64,
    ) -> Result<Executed> {
        let FixtureInputs {
            store,
            edit,
            noop,
            smoke,
            cancel,
        } = inputs;
        let c = self.contract(&lease.contract)?;
        self.allocation_fence(lease, reservation, cancel, cpu_start)?;
        let source = resolve(store, &c.source)?;
        let best = resolve(store, &c.best_known)?;
        if source.binding(c.source.graph.clone()) != edit.program().base_model {
            return Err(deny(
                "compiled edit source differs from resolved contract source",
            ));
        }
        self.allocation_fence(lease, reservation, cancel, cpu_start)?;
        let candidate = execute_cancellable(edit, &source, smoke, bounds::EXECUTION, cancel)?;
        self.allocation_fence(lease, reservation, cancel, cpu_start)?;
        let control = execute_cancellable(noop, &source, smoke, bounds::EXECUTION, cancel)?;
        if control.receipt().changed_parameters != 0
            || control.receipt().before_parameters != control.receipt().after_parameters
            || control.receipt().before_buffers != control.receipt().after_buffers
        {
            return Err(deny("no-op changed source"));
        }
        self.allocation_fence(lease, reservation, cancel, cpu_start)?;
        let a = export_candidate(store, &candidate, smoke, bounds::export_limits())?;
        self.allocation_fence(lease, reservation, cancel, cpu_start)?;
        let n = export_candidate(store, &control, smoke, bounds::export_limits())?;
        self.allocation_fence(lease, reservation, cancel, cpu_start)?;
        let final_model = restore_source(store, &a.export_root, bounds::export_limits())?;
        let no_op = restore_source(store, &n.export_root, bounds::export_limits())?;
        if no_op.manifest().parameters != source.manifest().parameters
            || no_op.manifest().buffers != source.manifest().buffers
            || no_op.assets() != source.assets()
        {
            return Err(deny("no-op reload identity changed"));
        }
        let candidate_id = identity(&final_model)?;
        let noop_id = identity(&no_op)?;
        // Freeze the identity actually executed, not the pre-export source.
        let tx = self.connection.unchecked_transaction().map_err(sql)?;
        check_lease(&tx, &self.store, lease, &files::moment()?)?;
        record(
            &tx,
            lease,
            "controls",
            &json!({"source":c.source,"no_op":noop_id,"best_known":c.best_known}),
        )?;
        tx.commit().map_err(sql)?;
        // Trusted edits and export validation retain coordinator authority. All
        // matched holdout forward computation crosses the qualified FD broker.
        let models = [&source, &no_op, &best, &final_model];
        // Bundle exports embed the model manifest; publish its verified bytes
        // separately so every frozen identity axis resolves directly from CAS.
        store.put(canonical(&**final_model.manifest())?.as_slice())?;
        store.put(canonical(&**no_op.manifest())?.as_slice())?;
        resolve(store, &candidate_id)?;
        resolve(store, &noop_id)?;
        let mut contrasts = Vec::new();
        let mut catalogs = BTreeMap::new();
        for b in c
            .portfolio
            .bindings
            .iter()
            .filter(|b| b.schedule.final_required)
        {
            let items = splits::final_items(&self.connection, &lease.contract, &b.id)?;
            for m in &b.metrics {
                let base = match m.comparator {
                    Comparator::BestKnown => &c.best_known,
                    Comparator::NoOp => &noop_id,
                    Comparator::Original => &c.source,
                };
                contrasts.push(Contrast {
                    id: hash(&(&candidate_id, &b.id, &m.metric))?,
                    candidate: model_identity(&candidate_id)?,
                    baseline: model_identity(base)?,
                    comparator: m.comparator,
                    binding: b.id.clone(),
                    metric: m.metric.clone(),
                    schema: c.portfolio.gyms[&b.gym].metrics[&m.metric].clone(),
                    value_statistic: m.statistic,
                    context: hash(&c.runtime)?,
                    workload: b.splits.holdout.digest()?,
                    measurement_protocol: hash(&m.measurement)?,
                    tasks: items
                        .iter()
                        .map(|t| statistics::Task {
                            item: t.identity.item.clone(),
                            group: t.identity.group.clone(),
                            input: t.input.clone(),
                        })
                        .collect(),
                    threshold: m.threshold.clone(),
                    required: m.required,
                    no_op_control: false,
                });
            }
            catalogs.insert(b.id.clone(), items);
        }
        let plan = Plan::freeze(&canonical(&Protocol {
            schema_version: 1,
            family: c.statistics.multiplicity_family.clone(),
            estimator: c.statistics.estimator,
            scope: "deterministic_fixture_only".into(),
            estimand: "equal_weight_group_mean_of_task_seed_means".into(),
            minimum_groups: c.statistics.minimum_groups,
            minimum_tasks: 1,
            seeds: c.seeds.clone(),
            confidence: c.statistics.confidence,
            resamples: 0,
            bootstrap_seed: 0,
            maximum_comparisons: c.statistics.maximum_comparisons,
            contrasts,
        })?)?;
        self.final_fence(lease, cancel)?;
        let panel =
            self.freeze_final_panel_with_analysis(lease, vec![candidate_id.clone()], &plan)?;
        let dispatch = json!({"contract":lease.contract,"panel":panel,"epoch":lease.epoch,"boot":lease.boot,"deadline":lease.deadline,"candidate":candidate_id,"original":c.source,"best_known":c.best_known,"plan":plan.record()?,"catalogs":catalogs,"edit":candidate.receipt(),"noop":control.receipt(),"candidate_export":a,"noop_export":n});
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sql)?;
        check_lease(&tx, &self.store, lease, &files::moment()?)?;
        let dispatch_id = record(&tx, lease, "dispatch", &dispatch)?;
        splits::mark_open(&tx, &lease.contract, &panel, "trusted-native-final-fixture")?;
        ensure_live(lease)?;
        tx.commit().map_err(sql)?;
        let mut observations = Vec::new();
        let mut receipts = Vec::new();
        let mut jobs = Vec::new();
        // Both transactions and both materialized/reload export checks ran.
        let mut evaluations = smoke.probes.len() as u64 * 6;
        for b in c
            .portfolio
            .bindings
            .iter()
            .filter(|b| b.schedule.final_required)
        {
            for item in &catalogs[&b.id] {
                self.allocation_fence(lease, reservation, cancel, cpu_start)?;
                let tx = self.connection.unchecked_transaction().map_err(sql)?;
                let subject = exposure::resolved_subject(
                    &tx,
                    &c,
                    &lease.contract,
                    &b.id,
                    SplitRole::Holdout,
                    item,
                    store,
                )?;
                exposure::admit(&tx, &c, &subject)?;
                let ticket = exposure::begin(
                    &tx,
                    lease,
                    &c,
                    exposure::Start {
                        subject,
                        recipient: "trusted-native-final-fixture".into(),
                        role: "final_evaluator".into(),
                        operation: hash(&(&dispatch_id, &b.id, &item.identity.item))?,
                        context: exposure::Context {
                            seed: Some(c.seeds[0]),
                            model: Some(model_identity(&candidate_id)?),
                            candidate: Some(model_identity(&candidate_id)?),
                            protocol: Some(plan.id().clone()),
                            ..Default::default()
                        },
                        fields: [
                            exposure::Field::Input,
                            exposure::Field::Label,
                            exposure::Field::Output,
                        ]
                        .into_iter()
                        .collect(),
                    },
                )?;
                ensure_live(lease)?;
                tx.commit().map_err(sql)?;
                let input: Vec<f64> =
                    serde_json::from_slice(&store.read_bounded(&item.input, 1024)?)
                        .map_err(|_| deny("exact numeric input required"))?;
                let label: Vec<f64> =
                    serde_json::from_slice(&store.read_bounded(&item.label, 1024)?)
                        .map_err(|_| deny("exact numeric label required"))?;
                if input.len() != 2
                    || label.len() != 2
                    || input
                        .iter()
                        .chain(&label)
                        .any(|x| !x.is_finite() || x.abs() > 1e6)
                {
                    return Err(deny("bounded 2-vector task/label required"));
                }
                let arms = [
                    identity(&source)?,
                    identity(&no_op)?,
                    identity(&best)?,
                    identity(&final_model)?,
                ];
                let cell = hash(&(&dispatch_id, &b.id, item, &arms, c.seeds[0]))?;
                jobs.push((b.id.clone(), item.clone(), input, label, ticket, cell));
            }
        }
        let requests: Vec<_> = jobs
            .iter()
            .map(|(_, _, input, _, _, key)| (key, input.as_slice()))
            .collect();
        let measured = self.supervised_forwards(
            lease,
            reservation,
            cancel,
            cpu_start,
            &dispatch_id,
            models,
            &requests,
        )?;
        for ((binding, item, _input, label, ticket, _cell), measurement) in
            jobs.into_iter().zip(measured)
        {
            let b = c
                .portfolio
                .bindings
                .iter()
                .find(|b| b.id == binding)
                .ok_or_else(|| deny("supervised binding absent"))?;
            let supervised::Measured {
                outputs,
                samples,
                latencies,
                evidence,
                ..
            } = measurement;
            evaluations += 16;
            let peak = rss()? as f64;
            for contrast in plan
                .protocol()
                .contrasts
                .iter()
                .filter(|v| v.binding == b.id)
            {
                let baseline = match contrast.comparator {
                    Comparator::Original => 0,
                    Comparator::NoOp => 1,
                    Comparator::BestKnown => 2,
                };
                for (arm, index) in [(Arm::Baseline, baseline), (Arm::Candidate, 3)] {
                    let value = match contrast.metric.as_str() {
                        "latency" => latencies[index],
                        "memory" => peak,
                        _ => {
                            if label == outputs[index] {
                                1.0
                            } else {
                                0.0
                            }
                        }
                    };
                    observations.push(Observation {
                        contrast: contrast.id.clone(),
                        item: item.identity.item.clone(),
                        group: item.identity.group.clone(),
                        input: item.input.clone(),
                        seed: c.seeds[0],
                        arm,
                        model: if arm == Arm::Candidate {
                            contrast.candidate.clone()
                        } else {
                            contrast.baseline.clone()
                        },
                        context: contrast.context.clone(),
                        workload: contrast.workload.clone(),
                        measurement_protocol: contrast.measurement_protocol.clone(),
                        unit: contrast.schema.unit,
                        value_statistic: contrast.value_statistic,
                        status: statistics::Status::Complete,
                        value: Some(value),
                    });
                }
            }
            self.finish_exposure(
                lease,
                &ticket,
                exposure::Stage::Completed,
                ticket.body.intended.clone(),
                "native_matched_execution_scored",
                exposure::Cost {
                    model_calls: Some(16),
                    ..Default::default()
                },
            )?;
            receipts.push(json!({"dispatch":dispatch_id,"binding":b.id,"gym":b.gym,"item":item,"seed":c.seeds[0],"scorer":profile_id(),"arm_identities":[identity(&source)?,identity(&no_op)?,identity(&best)?,identity(&final_model)?],"outputs":outputs,"samples":samples,"peak_controller_rss_bytes":peak,"exposure":ticket.started,"supervisor":evidence}));
        }
        self.allocation_fence(lease, reservation, cancel, cpu_start)?;
        for model in [&source, &no_op, &best, &final_model] {
            model.verify_integrity()?;
        }
        Ok(Executed {
            data: Data {
                schema_version: 1,
                plan: plan.id().clone(),
                observations,
            },
            plan,
            panel,
            receipt: json!({"dispatch":dispatch_id,"cells":receipts,"forward_accounting":{"transaction_smoke":smoke.probes.len()*2,"export_and_reload_smoke":smoke.probes.len()*4,"holdout":evaluations-smoke.probes.len() as u64*6,"total":evaluations},"resource_scope":"2x2_f64_forward_only_no_validation_hashing_or_allocation;per_task_max_of_three_excluding_one_warmup_then_equal_group_mean;whole_controller_getrusage_RSS_not_per_arm;controller_plus_guardian_and_children_CPU_envelope;qualified_static_worker_forward;guardian_delivery_and_SQLite_tail_observed_not_exact_hard_cutoff;no_device_metrics"}),
            evaluations,
        })
    }
    pub(super) fn fixture_noop_identity(
        &self,
        contract: &ContentDigest,
    ) -> Result<Option<ModelIdentity>> {
        read_noop_in(&self.connection, contract)
    }
}
fn read_noop_in(
    connection: &Connection,
    contract: &ContentDigest,
) -> Result<Option<ModelIdentity>> {
    read(connection, contract, "controls")?
        .map(|(_, v)| {
            let c = load_contract(connection, contract)?;
            let model: ModelIdentity = serde_json::from_value(v["no_op"].clone())
                .map_err(|_| deny("invalid native no-op identity"))?;
            if v["source"] != json!(c.source)
                || v["best_known"] != json!(c.best_known)
                || model.parameters != c.source.parameters
                || model.buffers != c.source.buffers
                || model.graph != c.source.graph
                || model.config != c.source.config
                || model.tokenizer != c.source.tokenizer
                || model.system_state != c.source.system_state
                || model.runtime_artifact != c.source.runtime_artifact
            {
                return Err(deny("native no-op control binding differs"));
            }
            Ok(model)
        })
        .transpose()
}
/// Explicit timing scope, not an interchangeable "hard wall qualified" flag.
#[derive(Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FixtureTimingScope {
    /// Absolute worker deadline, cancellation and conservative unknown cleanup;
    /// coordinator admission/fencing checks. Physical persistence tail unbounded.
    OperationalCancellation,
}

/// Native, non-deserializable witness of verified local operational evidence.
/// Private fields prevent caller/reviewer booleans from constructing authority.
/// This is NOT a promotion grant or current lease: consumers must recheck their
/// own frozen timing requirements, runtime, fencing, stop and release policy.
#[derive(Debug, Serialize)]
pub struct NativeFixtureQualification {
    contract: ContentDigest,
    decision: ContentDigest,
    profile: ContentDigest,
    supervisor_profile: ContentDigest,
    worker_image: ContentDigest,
    supervisor_receipts: Vec<ContentDigest>,
    timing_scope: FixtureTimingScope,
    physical_publication: &'static str,
}
impl NativeFixtureQualification {
    pub fn contract(&self) -> &ContentDigest {
        &self.contract
    }
    pub fn decision(&self) -> &ContentDigest {
        &self.decision
    }
    pub fn timing_scope(&self) -> &FixtureTimingScope {
        &self.timing_scope
    }
    pub fn digest(&self) -> Result<ContentDigest> {
        hash(self)
    }
}

pub(super) fn supervised_profile() -> ContentDigest {
    supervised::profile()
}
pub(super) fn worker_image_digest() -> ContentDigest {
    supervised::image_digest()
}

impl Ledger {
    /// Derive a qualification from immutable evidence plus actual CAS artifacts,
    /// never a caller-supplied receipt. Only the exact current frozen native
    /// profile is supported; historical contracts are not silently re-scoped.
    /// No strict durable-publication capability or release gate is changed.
    pub fn qualify_fixture_final(
        &self,
        contract: &ContentDigest,
        store: &ArtifactStore,
    ) -> Result<NativeFixtureQualification> {
        self.files.verify(self.limits.max_database_bytes)?;
        if read(&self.connection, contract, "arithmetic_decision")?.is_some() {
            let qualified = self.qualify_arithmetic_final(contract, store)?;
            if !qualified.decision().report().accepted
                || qualified.decision().report().status != Status::Passed
            {
                return Err(deny("failed arithmetic final evidence cannot qualify"));
            }
            let mut statement = self
                .connection
                .prepare("SELECT digest FROM final_fixture_records WHERE contract=?1 AND kind LIKE 'worker_receipt_%' ORDER BY kind")
                .map_err(sql)?;
            let supervisor_receipts = statement
                .query_map([contract.as_str()], |r| r.get::<_, String>(0))
                .map_err(sql)?
                .map(|r| r.map_err(sql)?.parse())
                .collect::<Result<Vec<ContentDigest>>>()?;
            if supervisor_receipts.is_empty() {
                return Err(deny("arithmetic supervisor receipts absent"));
            }
            self.files.verify(self.limits.max_database_bytes)?;
            return Ok(NativeFixtureQualification {
                contract: contract.clone(),
                decision: qualified.decision().id().clone(),
                profile: arithmetic_final::profile_id(),
                supervisor_profile: supervised::profile(),
                worker_image: supervised::image_digest(),
                supervisor_receipts,
                timing_scope: FixtureTimingScope::OperationalCancellation,
                physical_publication: "unsupported_unbounded_SQLite_fsync_and_scheduler_tail",
            });
        }
        let tx = self.connection.unchecked_transaction().map_err(sql)?;
        let decision = read_final_decision_in(&tx, &self.store, contract)?
            .ok_or_else(|| deny("operational qualification requires complete final evidence"))?;
        if !decision.report().accepted {
            return Err(deny("failed final evidence cannot qualify"));
        }
        let body = load_contract(&tx, contract)?;
        let (_, intent) =
            read(&tx, contract, "intent")?.ok_or_else(|| deny("qualification intent absent"))?;
        if body.runtime.backend.config != profile_id()
            || intent["profile"] != json!(profile_id())
            || intent["supervised_profile"] != json!(supervised::profile())
            || intent["worker_image"] != json!(supervised::image_digest())
        {
            return Err(deny(
                "frozen native profile differs; no implicit contract revision",
            ));
        }
        if !supervised::cleanup_known(&tx, contract)? {
            return Err(deny("qualification cleanup unknown"));
        }
        let (_, dispatch) = read(&tx, contract, "dispatch")?
            .ok_or_else(|| deny("qualification dispatch absent"))?;
        let noop = read_noop_in(&tx, contract)?.ok_or_else(|| deny("qualification noop absent"))?;
        let candidate: ModelIdentity = serde_json::from_value(dispatch["candidate"].clone())
            .map_err(|_| deny("qualification candidate absent"))?;
        let states = [
            resolve(store, &body.source)?,
            resolve(store, &noop)?,
            resolve(store, &body.best_known)?,
            resolve(store, &candidate)?,
        ];
        let supervisor_receipts = supervised::qualify_inputs(
            &tx,
            contract,
            [&states[0], &states[1], &states[2], &states[3]],
        )?;
        self.files.verify(self.limits.max_database_bytes)?;
        Ok(NativeFixtureQualification {
            contract: contract.clone(),
            decision: decision.id().clone(),
            profile: profile_id(),
            supervisor_profile: supervised::profile(),
            worker_image: supervised::image_digest(),
            supervisor_receipts,
            timing_scope: FixtureTimingScope::OperationalCancellation,
            physical_publication: "unsupported_unbounded_SQLite_fsync_and_scheduler_tail",
        })
    }

    /// Read immutable local fixture evidence, including a postcommit observation.
    /// Missing observation (even with settled completion) is incomplete forever;
    /// recovery cannot manufacture a past timing observation. This is NOT a hard
    /// wall/CPU enforcement certificate: the observation's own commit is unbounded.
    /// This does not accept a report/receipt supplied by a caller.
    pub fn fixture_final_decision(&self, contract: &ContentDigest) -> Result<Option<Decision>> {
        self.files.verify(self.limits.max_database_bytes)?;
        let tx = self.connection.unchecked_transaction().map_err(sql)?;
        let decision = read_final_decision_in(&tx, &self.store, contract)?;
        self.files.verify(self.limits.max_database_bytes)?;
        Ok(decision)
    }
    /// Started with no committed decision is incomplete, including process death.
    pub fn fixture_final_status(&self, contract: &ContentDigest) -> Result<Option<Status>> {
        if let Some(d) = self.fixture_final_decision(contract)? {
            return Ok(Some(d.report().status));
        }
        Ok(read(&self.connection, contract, "intent")?.map(|_| Status::Incomplete))
    }
}
pub(super) fn read_any_final_decision_in(
    connection: &Connection,
    store: &str,
    contract: &ContentDigest,
) -> Result<Option<Decision>> {
    if let Some(d) = read_final_decision_in(connection, store, contract)? {
        return Ok(Some(d));
    }
    let Some((decision_id, report_val)) = read(connection, contract, "arithmetic_decision")? else {
        return Ok(None);
    };
    let c = load_contract(connection, contract)?;
    let (_, intent) =
        read(connection, contract, "intent")?.ok_or_else(|| deny("arithmetic intent missing"))?;
    let (dispatch_id, dispatch) = read(connection, contract, "arithmetic_dispatch")?
        .ok_or_else(|| deny("arithmetic dispatch missing"))?;
    let (execution_id, execution) = read(connection, contract, "arithmetic_execution")?
        .ok_or_else(|| deny("arithmetic execution missing"))?;
    let report: Report = serde_json::from_value(report_val)
        .map_err(|_| deny("arithmetic decision schema invalid"))?;
    if intent["profile"] != json!(arithmetic_final::profile_id())
        || intent["scope"] != arithmetic_final::SCOPE
        || report.schema_version != 1
        || report.contract != *contract
        || report.revision != c.revision
        || report.scope != arithmetic_final::SCOPE
        || report.promotion_authorized
        || report.accepted != (report.status == Status::Passed)
        || execution["dispatch"] != json!(dispatch_id)
        || execution["scope"] != arithmetic_final::SCOPE
        || dispatch["profile"] != json!(arithmetic_final::profile_id())
        || dispatch["snapshot"] != intent["snapshot"]
        || dispatch["approval"] != intent["approval"]
    {
        return Err(deny("arithmetic durable decision binding differs"));
    }
    let reservation: ContentDigest = serde_json::from_value(intent["reservation"].clone())
        .map_err(|_| deny("reservation missing"))?;
    budgets::snapshot_in(connection, contract)?;
    if !budgets::native_released(connection, &reservation)? {
        return Err(deny("arithmetic allocation unsettled"));
    }
    let request_bytes: Vec<u8> = connection
        .query_row(
            "SELECT body FROM budget_requests WHERE id=?1 AND contract=?2",
            params![reservation.as_str(), contract.as_str()],
            |row| row.get(0),
        )
        .map_err(sql)?;
    let request = CanonicalDocument::parse(&request_bytes, Limits::default())?;
    let anchor: Option<Vec<u8>> = connection
        .query_row(
            "SELECT body FROM events WHERE contract=?1 AND digest=?2",
            params![contract.as_str(), report.event_head.as_str()],
            |row| row.get(0),
        )
        .optional()
        .map_err(sql)?;
    let anchor = anchor
        .map(|bytes| decode_event(&bytes, report.event_head.as_str()))
        .transpose()?;
    if anchor.as_ref().is_none_or(|event| {
        event.body.contract != *contract
            || event.body.transition.next != Stage::Verify
            || json!(event.body.epoch) != request.value()["epoch"]
            || json!(event.body.boot) != request.value()["boot"]
    }) {
        return Err(deny("arithmetic final event anchor differs"));
    }
    let outcome_bytes: Vec<u8> = connection
        .query_row(
            "SELECT body FROM budget_outcomes WHERE request=?1 ORDER BY version DESC LIMIT 1",
            [reservation.as_str()],
            |row| row.get(0),
        )
        .map_err(sql)?;
    let outcome = CanonicalDocument::parse(&outcome_bytes, Limits::default())?;
    if outcome.value()["request"] != json!(reservation)
        || outcome.value()["settlement"]["evidence"] != json!(execution_id)
        || outcome.value()["settlement"]["outcome"] != "completed"
        || outcome.value()["settlement"]["resources_released"] != true
    {
        return Err(deny("arithmetic settlement binding differs"));
    }
    let decision = Decision::from_report(report)?;
    if decision.id() != &decision_id {
        return Err(deny("arithmetic decision digest mismatch"));
    }
    Ok(Some(decision))
}
/// Uses the caller's existing transaction; never opens a transaction or probes
/// the current executable/process. Historical records are not current runtime
/// qualification, and strict T038 publication remains unsupported.
pub(super) fn read_final_decision_in(
    connection: &Connection,
    store: &str,
    contract: &ContentDigest,
) -> Result<Option<Decision>> {
    let body = load_contract(connection, contract)?;
    let Some((decision_id, value)) = read(connection, contract, "decision")? else {
        return Ok(None);
    };
    let Some((completion_id, completion)) = read(connection, contract, "completion")? else {
        return Ok(None);
    };
    let Some((_, observation)) = read(connection, contract, "commit_observation")? else {
        return Ok(None);
    };
    let report: Report = serde_json::from_value(value).map_err(|_| {
        failure(
            ErrorCode::IntegrityMismatch,
            "invalid durable final decision",
        )
    })?;
    let (intent_id, intent) =
        read(connection, contract, "intent")?.ok_or_else(|| deny("missing final intent"))?;
    let (execution_id, execution) = read(connection, contract, "execution")?
        .ok_or_else(|| deny("missing execution receipt"))?;
    if observation["completion"] != json!(completion_id)
        || observation["decision"] != json!(decision_id)
        || observation["epoch"] != intent["epoch"]
        || observation["boot"] != intent["boot"]
        || observation["deadline_ms"] != intent["operation_deadline"]
        || observation["semantics"] != "postcommit_observation_not_hard_enforcement"
        || observation["observed_ms"].as_i64().is_none_or(|t| {
            t < completion["completed_ms"].as_i64().unwrap_or(i64::MAX)
                || t >= intent["operation_deadline"].as_i64().unwrap_or(i64::MIN)
        })
        || observation["cpu_ms"].as_u64().is_none_or(|t| {
            t < completion["cpu_ms"].as_u64().unwrap_or(u64::MAX)
                || t > intent["resources"]["cpu_ms"].as_u64().unwrap_or(0)
        })
        || completion["decision"] != json!(decision_id)
        || completion["linearization"] != "last_fenced_precommit_check_not_fsync_completion"
        || intent["hard_address_space_bytes"]
            .as_u64()
            .is_none_or(|hard| {
                hard < bounds::MEMORY
                    || hard > body.limits.host_memory_bytes
                    || hard > intent["resources"]["host_bytes"].as_u64().unwrap_or(0)
            })
        || intent["profile"] != json!(body.runtime.backend.config)
        || completion["boot"] != intent["boot"]
        || completion["deadline_ms"] != intent["operation_deadline"]
        || completion["completed_ms"]
            .as_i64()
            .is_none_or(|t| t >= completion["deadline_ms"].as_i64().unwrap_or(i64::MIN))
        || completion["cpu_ms"]
            .as_u64()
            .is_none_or(|t| t > intent["resources"]["cpu_ms"].as_u64().unwrap_or(0))
        || report.contract != *contract
        || report.promotion_authorized
        || report.accepted != (report.status == Status::Passed)
        || (execution["status"] != "complete" && report.status == Status::Passed)
        || report.schema_version != 1
        || report.revision != body.revision
        || intent["contract"] != json!(contract)
        || intent["store"] != store
        || execution["store"] != store
        || execution["intent"] != json!(intent_id)
        || execution["contract"] != json!(contract)
        || (report.accepted
            && (report.status != Status::Passed
                || execution["status"] != "complete"
                || execution["analysis"] != json!(report.analysis)
                || !report
                    .checks
                    .iter()
                    .any(|c| c.name == "final_execution" && c.reason == execution_id.to_string())))
    {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "durable final decision links differ",
        ));
    }
    let current = state(connection, contract)?;
    let anchor: Option<Vec<u8>> = connection
        .query_row(
            "SELECT body FROM events WHERE contract=?1 AND digest=?2",
            params![contract.as_str(), report.event_head.as_str()],
            |r| r.get(0),
        )
        .optional()
        .map_err(sql)?;
    let anchor = anchor
        .map(|bytes| decode_event(&bytes, report.event_head.as_str()))
        .transpose()?;
    if anchor.as_ref().is_none_or(|e| {
        e.body.transition.next != Stage::Verify
            || json!(e.body.epoch) != intent["epoch"]
            || json!(e.body.boot) != intent["boot"]
    }) || current.sequence == 0
    {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "final decision historical event anchor absent",
        ));
    }
    budgets::snapshot_in(connection, contract)?;
    if execution["status"] == "complete" {
        let no_op = read_noop_in(connection, contract)?
            .ok_or_else(|| deny("missing native control identities"))?;
        let (id, dispatch) = read(connection, contract, "dispatch")?
            .ok_or_else(|| deny("missing frozen dispatch"))?;
        let (panel_id, panel) =
            read_panel_in(connection, contract)?.ok_or_else(|| deny("missing frozen panel"))?;
        let candidate = panel
            .finalists
            .first()
            .ok_or_else(|| deny("missing native finalist"))?;
        let arms = json!([body.source, no_op, body.best_known, candidate]);
        if execution["receipt"]["cells"]
            .as_array()
            .is_none_or(|cells| {
                cells.is_empty() || cells.iter().any(|cell| cell["arm_identities"] != arms)
            })
            || dispatch["noop_export"]["exported_model"] != json!(no_op.manifest)
        {
            return Err(deny("executed control/model identities differ"));
        }
        verify_execution_links(
            connection, &body, &panel_id, &panel, &dispatch, &id, &execution, &report, &no_op,
        )?;
        if execution["receipt"]["dispatch"] != json!(id)
            || dispatch["contract"] != json!(contract)
            || dispatch["panel"] != json!(panel_id)
            || report.panel.as_ref() != Some(&panel_id)
            || panel.analysis_plan != report.plan
            || execution["panel"] != json!(report.panel)
            || execution["epoch"] != intent["epoch"]
            || dispatch["epoch"] != intent["epoch"]
            || dispatch["boot"] != intent["boot"]
            || execution["boot"] != intent["boot"]
            || dispatch["deadline"] != intent["deadline"]
            || execution["plan"]["id"] != json!(report.plan)
            || hash(&execution["analysis_report"])?
                != *report
                    .analysis
                    .as_ref()
                    .ok_or_else(|| deny("missing native analysis"))?
            || execution["analysis_report"]["data_digest"] != json!(hash(&execution["data"])?)
            || (report.accepted
                && (!report.analysis_conditions_satisfied
                    || report
                        .checks
                        .iter()
                        .any(|c| c.required && c.status != Status::Passed)))
        {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "execution dispatch/panel/analysis/budget closure differs",
            ));
        }
    } else {
        if execution["status"] != "incomplete" {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "unknown native execution status",
            ));
        }
        verify_decision_checks(connection, &body, &execution, &report, None)?;
    }
    Ok(Some(Decision::from_report(report)?))
}

/// Rebuild the complete gate inventory from frozen requirements and verified
/// native evidence. Submitted booleans, statuses, omitted/optionalized gates and
/// duplicate gates are never substitutes for this calculation.
fn verify_decision_checks(
    connection: &Connection,
    body: &StudyContract,
    execution: &Value,
    report: &Report,
    analysis: Option<&Value>,
) -> Result<()> {
    let invalid = || {
        failure(
            ErrorCode::IntegrityMismatch,
            "native decision gate coverage/outcome differs",
        )
    };
    let complete = analysis.is_some();
    let mut expected = Vec::new();
    let panel = read_panel_in(connection, &report.contract)?;
    if report.panel.as_ref() != panel.as_ref().map(|(id, _)| id) {
        return Err(invalid());
    }
    if let Some((id, panel)) = &panel {
        expected.push(row("frozen_panel", Status::Passed, ""));
        let opening: Option<(Vec<u8>, String, String)> = connection
            .query_row(
                "SELECT body,digest,panel FROM holdout_openings WHERE contract=?1",
                [report.contract.as_str()],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()
            .map_err(sql)?;
        let opened = if let Some((bytes, digest, indexed_panel)) = opening {
            let doc = CanonicalDocument::parse(&bytes, Limits::default())?;
            if doc.bytes() != bytes
                || doc.digest().as_str() != digest
                || doc.value()["contract"] != json!(report.contract)
                || doc.value()["panel"] != json!(id)
                || indexed_panel != id.as_str()
            {
                return Err(invalid());
            }
            true
        } else {
            false
        };
        if complete && !opened {
            return Err(invalid());
        }
        expected.push(row(
            "holdout_opening",
            if opened {
                Status::Passed
            } else {
                Status::Incomplete
            },
            "",
        ));
        if let Some(analysis) = analysis {
            for name in ["prefrozen_analysis_plan", "analysis_binding"] {
                expected.push(row(name, Status::Passed, ""));
            }
            let results = analysis["results"].as_array().ok_or_else(invalid)?;
            for candidate in &panel.finalists {
                for b in body
                    .portfolio
                    .bindings
                    .iter()
                    .filter(|b| b.schedule.final_required)
                {
                    for m in &b.metrics {
                        let contrast = hash(&(candidate, &b.id, &m.metric))?;
                        let result = results
                            .iter()
                            .find(|r| r["contrast"] == json!(contrast))
                            .ok_or_else(invalid)?;
                        let status = match result["status"].as_str() {
                            Some("passed") => Status::Passed,
                            Some("failed" | "no_op_cannot_satisfy_positive_gain") => Status::Failed,
                            Some("inconclusive") => Status::Inconclusive,
                            _ => return Err(invalid()),
                        };
                        let mut check = row("frozen_metric", status, "");
                        check.candidate = Some(model_identity(candidate)?);
                        check.binding = Some(b.id.clone());
                        check.metric = Some(m.metric.clone());
                        check.required = m.required;
                        expected.push(check);
                    }
                }
            }
            expected.push(row("contrast_coverage", Status::Passed, ""));
            expected.push(row(
                "full_analysis_panel",
                if acceptance::has_resource_ceilings(body) {
                    Status::Passed
                } else {
                    Status::Missing
                },
                "",
            ));
        } else {
            expected.push(row("analysis", Status::Incomplete, ""));
            for candidate in &panel.finalists {
                for b in body
                    .portfolio
                    .bindings
                    .iter()
                    .filter(|b| b.schedule.final_required)
                {
                    for m in &b.metrics {
                        let mut check = row("frozen_metric", Status::Incomplete, "");
                        check.candidate = Some(model_identity(candidate)?);
                        check.binding = Some(b.id.clone());
                        check.metric = Some(m.metric.clone());
                        check.required = m.required;
                        expected.push(check);
                    }
                }
            }
            expected.push(row("full_analysis_panel", Status::Incomplete, ""));
        }
    } else {
        if complete {
            return Err(invalid());
        }
        expected.push(row("frozen_panel", Status::Incomplete, ""));
    }
    let analysis_passed = expected
        .iter()
        .all(|c| !c.required || c.status == Status::Passed);
    let metric_analysis_passed = expected
        .iter()
        .filter(|c| c.name == "frozen_metric")
        .all(|c| !c.required || c.status == Status::Passed);
    if report.analysis_conditions_satisfied != analysis_passed
        || analysis.is_some_and(|a| {
            a["all_required_analysis_conditions_satisfied"] != json!(metric_analysis_passed)
        })
        || (!complete && (report.plan.is_some() || report.analysis.is_some()))
    {
        return Err(invalid());
    }
    let native_status = if complete {
        Status::Passed
    } else {
        Status::Incomplete
    };
    let execution_id = hash(execution)?.to_string();
    if let Some((_, panel)) = &panel {
        for candidate in &panel.finalists {
            let mut check = row(
                "candidate_artifact_resolution",
                native_status,
                &execution_id,
            );
            check.candidate = Some(model_identity(candidate)?);
            expected.push(check);
        }
    }
    for name in [
        "final_execution",
        "source_noop_best_known_controls",
        "resource_measurements",
        "method_and_edit_integrity",
    ] {
        let status = if complete
            && name == "resource_measurements"
            && !acceptance::has_resource_ceilings(body)
        {
            Status::Missing
        } else {
            native_status
        };
        expected.push(row(name, status, &execution_id));
    }
    if body.objective.kind == ObjectiveKind::CausalClaim {
        expected.push(row(
            "mechanism_confirmation",
            if complete {
                Status::Missing
            } else {
                Status::Incomplete
            },
            "",
        ));
    }
    if !complete {
        expected.push(row("interrupted_execution", Status::Incomplete, ""));
    }
    // Observed completion proves the bounded publication checks ran. Recheck
    // the recorded execution measurements too; never use their gate booleans.
    let (_, intent) = read(connection, &report.contract, "intent")?.ok_or_else(invalid)?;
    let reservation: ContentDigest =
        serde_json::from_value(intent["reservation"].clone()).map_err(|_| invalid())?;
    let request_bytes: Vec<u8> = connection
        .query_row(
            "SELECT body FROM budget_requests WHERE id=?1 AND contract=?2",
            params![reservation.as_str(), report.contract.as_str()],
            |r| r.get(0),
        )
        .map_err(sql)?;
    let request = CanonicalDocument::parse(&request_bytes, Limits::default())?;
    let request = request.value();
    let outcome_bytes: Vec<u8> = connection
        .query_row(
            "SELECT body FROM budget_outcomes WHERE request=?1 ORDER BY version DESC LIMIT 1",
            [reservation.as_str()],
            |r| r.get(0),
        )
        .map_err(sql)?;
    let outcome = CanonicalDocument::parse(&outcome_bytes, Limits::default())?;
    if request["request"]["operation"] != json!(hash(&("final_fixture", &report.contract))?)
        || request["request"]["resources"] != intent["resources"]
        || request["epoch"] != intent["epoch"]
        || request["boot"] != intent["boot"]
        || outcome.value()["settlement"]["evidence"] != json!(hash(execution)?)
        || outcome.value()["settlement"]["outcome"]
            != if complete { "completed" } else { "incomplete" }
        || !budgets::native_released(connection, &reservation)?
    {
        return Err(invalid());
    }
    if execution["cpu_ms"]
        .as_u64()
        .is_none_or(|n| n > intent["resources"]["cpu_ms"].as_u64().unwrap_or(0))
        || execution["wall_ms"]
            .as_u64()
            .is_none_or(|n| n > 30_000.min(body.limits.wall_ms))
        || (complete
            && execution["peak_controller_rss_bytes"]
                .as_u64()
                .is_none_or(|n| n > intent["resources"]["host_bytes"].as_u64().unwrap_or(0)))
    {
        return Err(invalid());
    }
    for name in [
        "resource_budget_and_cleanup",
        "final_fence_cleanup_and_accounting",
    ] {
        expected.push(row(name, Status::Passed, ""));
    }
    acceptance::verify_check_inventory(&report.checks, &expected)?;
    let status = [
        Status::Failed,
        Status::Incomplete,
        Status::Missing,
        Status::Inconclusive,
    ]
    .into_iter()
    .find(|s| expected.iter().any(|c| c.required && c.status == *s))
    .unwrap_or(Status::Passed);
    if report.status != status
        || report.accepted != (status == Status::Passed)
        || report.scope
            != "trusted_native_linear_fixture_only_not_generalization_or_strict_T038_or_T039"
    {
        return Err(invalid());
    }
    Ok(())
}

fn read_panel_in(
    connection: &Connection,
    contract: &ContentDigest,
) -> Result<Option<(ContentDigest, splits::Panel)>> {
    let data: Option<(Vec<u8>, String)> = connection
        .query_row(
            "SELECT body,digest FROM final_panels WHERE contract=?1",
            [contract.as_str()],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()
        .map_err(sql)?;
    data.map(|(bytes, id)| {
        let doc = CanonicalDocument::parse(&bytes, Limits::default())?;
        if doc.digest().as_str() != id || doc.bytes() != bytes {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "final panel hash differs",
            ));
        }
        let panel = serde_json::from_value(doc.value().clone())
            .map_err(|_| failure(ErrorCode::IntegrityMismatch, "final panel schema differs"))?;
        Ok((doc.digest(), panel))
    })
    .transpose()
}

/// Cross-check complete native executions, including negative metric outcomes.
/// Only references are read; this never opens CAS, reexecutes a model or exposes
/// private tasks, labels, outputs, per-task scores or sample hashes to reports.
fn verify_execution_links(
    connection: &Connection,
    body: &StudyContract,
    panel_id: &ContentDigest,
    panel: &splits::Panel,
    dispatch: &Value,
    dispatch_id: &ContentDigest,
    execution: &Value,
    report: &Report,
    noop: &ModelIdentity,
) -> Result<()> {
    let invalid = || {
        failure(
            ErrorCode::IntegrityMismatch,
            "native final execution coverage/bindings differ",
        )
    };
    let contract = body.digest()?;
    let candidate = panel.finalists.first().ok_or_else(invalid)?;
    let bindings: BTreeSet<_> = body
        .portfolio
        .bindings
        .iter()
        .filter(|b| b.schedule.final_required)
        .map(|b| b.id.clone())
        .collect();
    if panel.schema_version != 1
        || panel.contract != contract
        || panel.revision != body.revision
        || panel.protocol != body.portfolio.independent.frozen_panel_protocol
        || panel.finalists.len() != 1
        || panel.original != body.source
        || panel.best_known != body.best_known
        || panel.bindings != bindings
        || panel.seeds != body.seeds
        || panel.repeats != body.statistics.repeats
        || panel.comparison_family != body.statistics.multiplicity_family
        || panel.status != "frozen_identities_not_model_execution_evidence"
        || body.statistics.estimator != Estimator::DeterministicExact
        || body.seeds.len() != 1
        || body.statistics.repeats != 1
        || body.runtime.device != Device::Cpu
        || body.runtime.precision != DType::F64
        || body.runtime.backend.id != "fixture.cpu"
        || body.runtime.backend.version != "1"
        || body.runtime.batch_size != 1
        || body.runtime.concurrency != 1
        || body.runtime.cache != CacheMode::Disabled
        || !body.runtime.deterministic
        || body.runtime.synchronization != Synchronization::CpuMonotonic
        || body.runtime.input_length != 2
        || body.runtime.max_output_length != 2
        || body.portfolio.independent.scope != ClaimScope::FixtureOnly
        || dispatch["candidate"] != json!(candidate)
        || dispatch["original"] != json!(body.source)
        || dispatch["best_known"] != json!(body.best_known)
        || dispatch["plan"] != execution["plan"]
        || panel.analysis_plan != report.plan
        || report.panel.as_ref() != Some(panel_id)
    {
        return Err(invalid());
    }
    let plan = Plan::freeze(&canonical(&execution["plan"]["protocol"])?)?;
    if plan.record()? != execution["plan"]
        || Some(plan.id()) != report.plan.as_ref()
        || plan.protocol().family != body.statistics.multiplicity_family
        || plan.protocol().seeds != body.seeds
        || plan.protocol().estimator != body.statistics.estimator
        || plan.protocol().minimum_groups != body.statistics.minimum_groups
        || plan.protocol().minimum_tasks != 1
        || plan.protocol().confidence != body.statistics.confidence
        || plan.protocol().maximum_comparisons != body.statistics.maximum_comparisons
    {
        return Err(invalid());
    }
    let (_, intent) = read(connection, &contract, "intent")?.ok_or_else(invalid)?;
    for (name, export, model) in [
        ("edit", "candidate_export", candidate),
        ("noop", "noop_export", noop),
    ] {
        let receipt = &dispatch[name];
        let _: ContentDigest = serde_json::from_value(dispatch[export]["export_root"].clone())
            .map_err(|_| invalid())?;
        if receipt["schema_version"] != 1
            || receipt["stage"] != "tensor_scope_and_smoke"
            || receipt["plan"] != intent[if name == "edit" { "edit" } else { "noop" }]
            || receipt["smoke_suite"] != intent["smoke"]
            || dispatch[export]["schema_version"] != 1
            || dispatch[export]["stage"] != "standalone_export_and_core_reload"
            || dispatch[export]["reload_instance"] != "fresh_native_cpu_instance_in_process"
            || dispatch[export]["smoke_suite"] != receipt["smoke_suite"]
            || dispatch[export]["smoke_outputs"] != receipt["smoke_outputs"]
            || dispatch[export]["candidate"] != json!(hash(receipt)?)
            || dispatch[export]["transaction"] != *receipt
            || dispatch[export]["exported_model"] != json!(model.manifest)
            || receipt["source_manifest"] != json!(body.source.manifest)
            || receipt["source_checkpoint"] != json!(body.source.checkpoint)
            || receipt["before_parameters"] != json!(body.source.parameters)
            || receipt["before_buffers"] != json!(body.source.buffers)
            || receipt["after_parameters"] != json!(model.parameters)
            || receipt["after_buffers"] != json!(model.buffers)
            || (name == "noop" && receipt["changed_parameters"] != 0)
        {
            return Err(invalid());
        }
    }
    let cells = execution["receipt"]["cells"]
        .as_array()
        .ok_or_else(invalid)?;
    let mut expected_cells = 0;
    let mut expected_contrasts = 0;
    for b in body
        .portfolio
        .bindings
        .iter()
        .filter(|b| b.schedule.final_required)
    {
        let items = splits::final_items(connection, &contract, &b.id)?;
        if dispatch["catalogs"][b.id.as_str()] != json!(items)
            || items.len() != b.schedule.holdout_episodes as usize
            || canonical(&items.iter().map(|t| &t.identity).collect::<Vec<_>>())?
                != canonical(&b.splits.holdout.items)?
        {
            return Err(invalid());
        }
        for item in &items {
            expected_cells += 1;
            let matches: Vec<_> = cells
                .iter()
                .filter(|cell| {
                    cell["binding"] == json!(b.id)
                        && cell["item"] == json!(item)
                        && cell["seed"] == json!(body.seeds[0])
                })
                .collect();
            if matches.len() != 1 {
                return Err(invalid());
            }
            let cell = matches[0];
            supervised::verify(connection, &contract, cell, &intent, dispatch_id)?;
            if cell["dispatch"] != json!(dispatch_id)
                || cell["gym"] != json!(b.gym)
                || cell["scorer"] != json!(body.runtime.backend.config)
                || cell["arm_identities"] != json!([body.source, noop, body.best_known, candidate])
                || cell["outputs"].as_array().is_none_or(|a| a.len() != 4)
                || cell["samples"].as_array().is_none_or(|a| a.len() != 16)
            {
                return Err(invalid());
            }
            if cell["outputs"][0] != cell["outputs"][1] {
                return Err(invalid());
            }
            let exposure_id: ContentDigest =
                serde_json::from_value(cell["exposure"].clone()).map_err(|_| invalid())?;
            let started: Vec<u8> = connection.query_row(
                "SELECT body FROM exposure_events WHERE id=?1 AND family=?2 AND stage='started'",
                params![exposure_id.as_str(), body.study_family.as_str()], |r| r.get(0),
            ).map_err(sql)?;
            let started_doc = CanonicalDocument::parse(&started, Limits::default())?;
            let start: exposure::EventBody =
                serde_json::from_value(started_doc.value().clone()).map_err(|_| invalid())?;
            if started_doc.digest() != exposure_id
                || start.contract != contract
                || start.revision != body.revision
                || start.subject.binding != b.id
                || start.subject.package_version != b.gym
                || canonical(&start.subject.task)? != canonical(&item.identity)?
                || start.subject.input != item.input
                || start.subject.label != item.label
                || start.subject.split != Some(SplitRole::Holdout)
                || start.context.seed != Some(body.seeds[0])
                || start.context.model != Some(model_identity(candidate)?)
                || start.context.candidate != Some(model_identity(candidate)?)
                || start.context.protocol.as_ref() != Some(plan.id())
                || json!(start.epoch) != dispatch["epoch"]
                || json!(start.boot) != dispatch["boot"]
            {
                return Err(invalid());
            }
            let (finished, finished_id): (Vec<u8>, String) = connection.query_row(
                "SELECT body,id FROM exposure_events WHERE family=?1 AND operation=?2 AND stage='completed'",
                params![body.study_family.as_str(), start.operation.as_str()], |r| Ok((r.get(0)?, r.get(1)?)),
            ).map_err(sql)?;
            let finished_doc = CanonicalDocument::parse(&finished, Limits::default())?;
            let finish: exposure::EventBody =
                serde_json::from_value(finished_doc.value().clone()).map_err(|_| invalid())?;
            if finished_doc.digest().as_str() != finished_id
                || finish.parent.as_ref() != Some(&exposure_id)
                || finish.contract != contract
                || finish.revision != body.revision
                || canonical(&finish.subject)? != canonical(&start.subject)?
                || canonical(&finish.context)? != canonical(&start.context)?
                || finish.shown != start.intended
                || finish.cost.model_calls != Some(16)
            {
                return Err(invalid());
            }
            for (index, sample) in cell["samples"].as_array().unwrap().iter().enumerate() {
                if sample["arm"] != json!(index % 4)
                    || sample["repeat"] != json!(index / 4)
                    || sample["warmup"] != json!(index < 4)
                    || sample["output"] != json!(hash(&cell["outputs"][index % 4])?)
                    || sample["wall_ms"]
                        .as_f64()
                        .is_none_or(|v| !v.is_finite() || v < 0.0)
                {
                    return Err(invalid());
                }
            }
        }
        for metric in &b.metrics {
            match metric.metric.as_str() {
                "quality" | "safety"
                    if metric.measurement.is_none() && metric.statistic == Statistic::Exact => {}
                "latency" | "memory" => {
                    let q = metric.measurement.as_ref().ok_or_else(invalid)?;
                    if q.repetitions != 3
                        || q.warmup != 1
                        || q.statistic != Statistic::Maximum
                        || metric.statistic != Statistic::Maximum
                        || q.order != MeasurementOrder::PairedInterleaved
                        || q.synchronization != Synchronization::CpuMonotonic
                        || q.timer.id != "nf.final.monotonic"
                        || q.timer.version != "1"
                        || q.timer.config != body.runtime.backend.config
                        || (metric.metric == "latency"
                            && q.scope != MeasurementScope::ForwardLatency)
                        || (metric.metric == "memory" && q.scope != MeasurementScope::ProcessRss)
                    {
                        return Err(invalid());
                    }
                }
                _ => return Err(invalid()),
            }
            expected_contrasts += 1;
            let matches: Vec<_> = plan
                .protocol()
                .contrasts
                .iter()
                .filter(|c| c.binding == b.id && c.metric == metric.metric)
                .collect();
            if matches.len() != 1 {
                return Err(invalid());
            }
            let c = matches[0];
            let baseline = match metric.comparator {
                Comparator::Original => &body.source,
                Comparator::NoOp => noop,
                Comparator::BestKnown => &body.best_known,
            };
            let tasks: Vec<_> = items
                .iter()
                .map(|t| statistics::Task {
                    item: t.identity.item.clone(),
                    group: t.identity.group.clone(),
                    input: t.input.clone(),
                })
                .collect();
            if c.id != hash(&(candidate, &b.id, &metric.metric))?
                || c.candidate != model_identity(candidate)?
                || c.baseline != model_identity(baseline)?
                || c.comparator != metric.comparator
                || c.context != hash(&body.runtime)?
                || c.workload != b.splits.holdout.digest()?
                || c.measurement_protocol != hash(&metric.measurement)?
                || canonical(&c.tasks)? != canonical(&tasks)?
                || canonical(&c.threshold)? != canonical(&metric.threshold)?
                || canonical(&c.schema)?
                    != canonical(&body.portfolio.gyms[&b.gym].metrics[&metric.metric])?
                || c.value_statistic != metric.statistic
                || c.required != metric.required
                || c.no_op_control
            {
                return Err(invalid());
            }
        }
    }
    let data: Data = serde_json::from_value(execution["data"].clone()).map_err(|_| invalid())?;
    for o in &data.observations {
        let c = plan
            .protocol()
            .contrasts
            .iter()
            .find(|c| c.id == o.contrast)
            .ok_or_else(invalid)?;
        let cell = cells
            .iter()
            .find(|cell| {
                cell["binding"] == json!(c.binding)
                    && cell["item"]["identity"]["item"] == json!(o.item)
                    && cell["seed"] == json!(o.seed)
            })
            .ok_or_else(invalid)?;
        let arm = match o.arm {
            Arm::Candidate => 3,
            Arm::Baseline => match c.comparator {
                Comparator::Original => 0,
                Comparator::NoOp => 1,
                Comparator::BestKnown => 2,
            },
        };
        let expected = match c.metric.as_str() {
            "latency" => cell["samples"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|s| s["arm"] == json!(arm) && s["warmup"] == false)
                .map(|s| s["wall_ms"].as_f64().unwrap())
                .fold(0.0, f64::max),
            "memory" => cell["peak_controller_rss_bytes"]
                .as_f64()
                .ok_or_else(invalid)?,
            "quality" | "safety" => {
                if json!(hash(&cell["outputs"][arm])?) == cell["item"]["label"] {
                    1.0
                } else {
                    0.0
                }
            }
            _ => return Err(invalid()),
        };
        if o.status != statistics::Status::Complete || o.value != Some(expected) {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "native observation differs from its executed arm cell",
            ));
        }
    }
    let expected_rows: usize = plan
        .protocol()
        .contrasts
        .iter()
        .map(|c| c.tasks.len() * body.seeds.len() * 2)
        .sum();
    if cells.len() != expected_cells
        || plan.protocol().contrasts.len() != expected_contrasts
        || data.observations.len() != expected_rows
    {
        return Err(invalid());
    }
    let analysis = plan.analyze(&canonical(&data)?, &CancellationToken::default())?;
    if hash(analysis.report())? != hash(&execution["analysis_report"])?
        || Some(analysis.id()) != report.analysis.as_ref()
        || analysis.report()["duplicates_dropped"] != 0
        || analysis.report()["results"].as_array().is_none_or(|rows| {
            rows.iter()
                .any(|r| r["status"] == "incomplete" || r["status"] == "missing")
        })
    {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "native analysis recomputation differs",
        ));
    }
    verify_decision_checks(connection, body, execution, report, Some(analysis.report()))
}

/// Non-deserializable report-only witness. No raw final record bodies cross
/// this boundary; none of these facts authorizes acceptance or promotion.
#[derive(Serialize)]
pub(crate) struct FinalEvidenceProjection {
    contract: ContentDigest,
    revision: u32,
    event_head: ContentDigest,
    decision: ContentDigest,
    recorded_status: Status,
    execution_status: String,
    panel: Option<ContentDigest>,
    plan: Option<ContentDigest>,
    receipt_links: Vec<FinalReceiptLink>,
    /// Group-aggregate results only, never per-task observations or outputs.
    metric_results: Vec<Value>,
    measurements: Value,
    export_reload_links: Vec<Value>,
    checks: Vec<Value>,
    strict_t038_publication: &'static str,
}
#[derive(Serialize)]
struct FinalReceiptLink {
    contract: ContentDigest,
    revision: u32,
    event_head: ContentDigest,
    candidate: ModelIdentity,
    arm: &'static str,
    model: ModelIdentity,
    kind: &'static str,
    digest: ContentDigest,
    panel: ContentDigest,
    plan: ContentDigest,
    task_count: usize,
    seeds: Vec<u64>,
}
pub(super) fn read_final_evidence_in(
    connection: &Connection,
    store: &str,
    contract: &ContentDigest,
) -> Result<Option<FinalEvidenceProjection>> {
    let Some(decision) = read_final_decision_in(connection, store, contract)? else {
        return Ok(None);
    };
    let report = decision.report();
    let (execution_id, execution) = read(connection, contract, "execution")?
        .ok_or_else(|| deny("missing verified execution"))?;
    let mut links = Vec::new();
    let mut metric_results = Vec::new();
    let mut measurements = Value::Null;
    let mut export_reload_links = Vec::new();
    if execution["status"] == "complete" {
        let (_, panel) =
            read_panel_in(connection, contract)?.ok_or_else(|| deny("missing verified panel"))?;
        let body = load_contract(connection, contract)?;
        let noop =
            read_noop_in(connection, contract)?.ok_or_else(|| deny("missing verified noop"))?;
        let (dispatch_id, dispatch) = read(connection, contract, "dispatch")?
            .ok_or_else(|| deny("missing verified dispatch"))?;
        for (arm, name) in [("candidate", "candidate_export"), ("no_op", "noop_export")] {
            let export = &dispatch[name];
            export_reload_links.push(json!({"arm":arm,"dispatch":dispatch_id,
                "receipt_digest":hash(export)?,"transaction":hash(&export["transaction"])?,
                "exported_model":export["exported_model"],"export_root":export["export_root"],
                "scope":"verified recorded native export/core reload links; not independent CAS resolution at capture"}));
        }
        measurements = json!({"runtime":body.runtime,"seeds":body.seeds,
            "warmup":1,"repetitions":3,"order":"paired_interleaved",
            "arm_order":["original","no_op","best_known","candidate"],
            "latency_statistic":"per_task_maximum_excluding_warmup_then_equal_group_mean",
            "memory_statistic":"whole_controller_process_rss_not_per_arm",
            "scope":"recorded native fixture; RSS is whole controller, not per-arm allocation",
            "hard_wall_cpu_qualified":false});
        let protocol: Protocol = serde_json::from_value(execution["plan"]["protocol"].clone())
            .map_err(|_| deny("missing verified protocol"))?;
        for (contrast, result) in protocol
            .contrasts
            .iter()
            .zip(execution["analysis_report"]["results"].as_array().unwrap())
        {
            metric_results.push(json!({"contrast":contrast.id,"binding":contrast.binding,
                "metric":contrast.metric,"candidate":contrast.candidate,"baseline":contrast.baseline,
                "comparator":contrast.comparator,"measurement":contrast.measurement_protocol,
                "unit":contrast.schema.unit,"statistic":contrast.value_statistic,
                "estimate":result["estimate"],"recorded_status":result["status"],
                "scope":"verified recorded finite-fixture aggregate; not scientific acceptance"}));
        }
        for (arm, model) in [
            ("original", body.source.clone()),
            ("no_op", noop),
            ("best_known", body.best_known.clone()),
            ("candidate", panel.finalists[0].clone()),
        ] {
            links.push(FinalReceiptLink {
                contract: contract.clone(),
                revision: report.revision,
                event_head: report.event_head.clone(),
                candidate: panel.finalists[0].clone(),
                arm,
                model,
                kind: "native_fixture_execution",
                digest: execution_id.clone(),
                panel: report
                    .panel
                    .clone()
                    .ok_or_else(|| deny("missing panel link"))?,
                plan: report
                    .plan
                    .clone()
                    .ok_or_else(|| deny("missing plan link"))?,
                task_count: execution["receipt"]["cells"].as_array().unwrap().len(),
                seeds: body.seeds.clone(),
            });
        }
    }
    Ok(Some(FinalEvidenceProjection {
        contract: contract.clone(), revision: report.revision, event_head: report.event_head.clone(),
        decision: decision.id().clone(), recorded_status: report.status,
        execution_status: if links.is_empty() { "incomplete" } else { "complete" }.into(),
        panel: report.panel.clone(), plan: report.plan.clone(), receipt_links: links,
        metric_results, measurements, export_reload_links,
        checks: report.checks.iter().map(|c| json!({"name":if matches!(c.name.as_str(),
            "final_execution" | "candidate_artifact_resolution" | "source_noop_best_known_controls"
            | "resource_measurements" | "method_and_edit_integrity" | "resource_budget_and_cleanup"
            | "final_fence_cleanup_and_accounting") {c.name.as_str()} else {"other_recorded_check"},"status":c.status,
            "required":c.required})).collect(),
        strict_t038_publication: "unresolved_hard_wall_and_cpu_through_observation_commit",
    }))
}
