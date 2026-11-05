//! Real paired original/no-op fixture controls. No study dispatch/promotion grant.
use crate::arithmetic::{self, Case};
use crate::model_worker::{Inference, Limits as WorkerLimits, QualifiedTorch};
use nf_checkpoint::store::{ArtifactStore, StoreLimits};
use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use nf_core::error::{CoreError, ErrorCode, Result};
use nf_core::schemas::Validated;
use nf_surgeon::cancellation::CancellationToken;
use nf_surgeon::export::{export_candidate, restore_source, CoreExportLimits};
use nf_surgeon::transaction::{execute_cancellable, ExecutionLimits, SmokeSuite, SourceState};
use nf_weight_ir::compilation::{CompiledPlan, NATIVE_CPU_BACKEND_ID, TORCH_CPU_BACKEND_ID};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::{Arc, Mutex, TryLockError};
use std::time::{Duration, Instant};

fn error(code: ErrorCode, text: &str) -> CoreError {
    CoreError::new(code, "/paired_controls", text)
}
fn doc(value: &impl Serialize) -> Result<CanonicalDocument> {
    CanonicalDocument::parse(
        &serde_json::to_vec(value).map_err(|_| error(ErrorCode::InvalidJson, "encoding failed"))?,
        Limits::default(),
    )
}
fn hash(value: &impl Serialize) -> Result<ContentDigest> {
    Ok(doc(value)?.digest())
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    pub device: String,
    pub precision: String,
    pub batch_size: u32,
    pub concurrency: u32,
    pub model_cache: String,
    pub state_reset: String,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            device: "cpu".into(),
            precision: "f64".into(),
            batch_size: 1,
            concurrency: 1,
            model_cache: "disabled".into(),
            state_reset: "fresh_process_per_arm_seeded_each_task".into(),
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Protocol {
    pub schema_version: u32,
    pub scope: String,
    pub protocol: ContentDigest,
    pub context: ContentDigest,
    pub head: String,
    pub tasks: Vec<Case>,
    pub seeds: Vec<u64>,
    pub settings: Settings,
    pub order: Order,
    pub worker_limits: WorkerLimits,
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Order {
    OriginalFirst,
    NoOpFirst,
}
#[derive(Default)]
pub struct SourceCache {
    values: Mutex<BTreeMap<ContentDigest, Arc<Inference>>>,
}
impl SourceCache {
    pub fn entries(&self) -> Result<usize> {
        Ok(self
            .values
            .lock()
            .map_err(|_| error(ErrorCode::TransactionAborted, "cache poisoned"))?
            .len())
    }
}
/// Only real completed paired runs create this witness. No DTO restoration.
pub struct PairedControls {
    id: ContentDigest,
    report: Value,
}
impl PairedControls {
    pub fn id(&self) -> &ContentDigest {
        &self.id
    }
    pub fn report(&self) -> &Value {
        &self.report
    }
}
fn normalize(mut p: Protocol) -> Result<(Protocol, usize)> {
    if p.schema_version != 1
        || p.scope != "public_synthetic_fixture_audit"
        || p.settings != Settings::default()
        || p.tasks.is_empty()
        || p.tasks.len() > 128
        || p.seeds.is_empty()
        || p.seeds.len() > 4
        || p.seeds.iter().collect::<BTreeSet<_>>().len() != p.seeds.len()
        || !arithmetic::HEADS.contains(&p.head.as_str())
    {
        return Err(error(ErrorCode::UnsupportedCapability,"bounded registered public fixture tasks/seeds and exact supported runtime settings required"));
    }
    let inventory = arithmetic::inventory()?;
    let mut tasks = Vec::new();
    let mut seen = BTreeSet::new();
    let raw_count = p.tasks.len();
    for case in p.tasks {
        let actual = inventory
            .cases
            .iter()
            .find(|c| c.item == case.item)
            .ok_or_else(|| {
                error(
                    ErrorCode::SourceMismatch,
                    "unknown registered task identity",
                )
            })?;
        if doc(actual)?.bytes() != doc(&case)?.bytes() {
            return Err(error(
                ErrorCode::SourceMismatch,
                "task/group/family/input/split was relabeled",
            ));
        }
        if seen.insert(case.item.clone()) {
            tasks.push(case);
        }
    }
    if tasks.len() > 32 || tasks.len() * p.seeds.len() > 64 {
        return Err(error(
            ErrorCode::LimitExceeded,
            "at most 32 distinct tasks and 64 task/seed forwards",
        ));
    }
    p.tasks = tasks;
    let duplicates = raw_count - p.tasks.len();
    Ok((p, duplicates))
}
fn key(
    source: &SourceState,
    runtime: &QualifiedTorch,
    plan: &CompiledPlan,
    smoke: &Validated<SmokeSuite>,
    p: &Protocol,
) -> Result<ContentDigest> {
    hash(
        &json!({"schema_version":1,"kind":"paired_source_evidence","source_manifest":source.manifest_digest(),"checkpoint":ContentDigest::of_bytes(source.checkpoint_bytes()),"assets":source.manifest().assets,"runtime":runtime.identity(),"control_plan":plan.digest(),"smoke":smoke.digest()?,"protocol":p,"controller_code":ContentDigest::of_bytes(include_bytes!("controls.rs"))}),
    )
}
fn verify_identity(source: &SourceState, reloaded: &SourceState) -> Result<()> {
    let a = source.manifest();
    let b = reloaded.manifest();
    if a.parameters != b.parameters
        || a.buffers != b.buffers
        || a.parameter_digest != b.parameter_digest
        || a.architecture != b.architecture
        || source.assets() != reloaded.assets()
    {
        return Err(error(
            ErrorCode::IntegrityMismatch,
            "no-op changed parameter/buffer/config/graph/assets or alias identities",
        ));
    }
    for name in a.parameters.keys().chain(a.buffers.keys()) {
        if source.tensor_bytes(name)? != reloaded.tensor_bytes(name)? {
            return Err(error(
                ErrorCode::IntegrityMismatch,
                "no-op changed exact tensor bytes",
            ));
        }
    }
    Ok(())
}
fn confirmed(value: &Inference) -> Result<()> {
    let r = value.receipt();
    if r.status != "verified_inference" || !r.cleanup_confirmed {
        return Err(error(
            ErrorCode::TransactionAborted,
            "control worker incomplete/failed; no paired evidence or cache insertion",
        ));
    }
    Ok(())
}
fn truth(head: &str, c: &Case) -> f64 {
    let a = f64::from(c.x1);
    let b = f64::from(c.x2);
    match head {
        "sum" => a + b,
        "difference" => a - b,
        "maximum" => a.max(b),
        "minimum" => a.min(b),
        _ => (a - b).abs(),
    }
}
pub struct RunContext<'a> {
    pub cache: &'a SourceCache,
    pub runtime: &'a QualifiedTorch,
    pub plan: &'a CompiledPlan,
    pub source: &'a SourceState,
    pub smoke: &'a Validated<SmokeSuite>,
    pub store_root: &'a Path,
    pub cancel: &'a CancellationToken,
}
pub fn run(context: RunContext<'_>, bytes: &[u8]) -> Result<PairedControls> {
    let RunContext {
        cache,
        runtime,
        plan,
        source,
        smoke,
        store_root,
        cancel,
    } = context;
    cancel.check()?;
    source.verify_integrity()?;
    let parsed = CanonicalDocument::parse(bytes, Limits::default())?;
    let p: Protocol = serde_json::from_value(parsed.value().clone())
        .map_err(|_| error(ErrorCode::InvalidValue, "invalid paired control protocol"))?;
    let (p, duplicates) = normalize(p)?;
    if source.manifest().architecture.id != "nf.arithmetic.relu"
        || source.manifest().architecture.version != "1"
    {
        return Err(error(
            ErrorCode::UnsupportedCapability,
            "paired accuracy runner currently supports the registered arithmetic model only",
        ));
    }
    // Serializes identical concurrent cache requests. A bounded cancellable wait
    // is not an implicit budget/lease authority or an unbounded queue.
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut cached = loop {
        cancel.check()?;
        match cache.values.try_lock() {
            Ok(value) => break value,
            Err(TryLockError::Poisoned(_)) => {
                return Err(error(ErrorCode::TransactionAborted, "cache poisoned"))
            }
            Err(TryLockError::WouldBlock) => {
                if Instant::now() >= deadline {
                    return Err(error(
                        ErrorCode::LimitExceeded,
                        "bounded control cache wait exceeded",
                    ));
                }
                std::thread::sleep(Duration::from_millis(2));
            }
        }
    };
    let evidence_key = key(source, runtime, plan, smoke, &p)?;
    let (noop, application) = match plan.backend().id.as_str() {
        NATIVE_CPU_BACKEND_ID => (
            execute_cancellable(plan, source, smoke, ExecutionLimits::default(), cancel)?,
            json!({"backend":"nf.native_cpu","kind":"actual_native_transaction"}),
        ),
        TORCH_CPU_BACKEND_ID => {
            let (candidate, receipt) = runtime
                .edit(plan, source, smoke, p.worker_limits.clone(), cancel)?
                .into_parts();
            (
                candidate,
                serde_json::to_value(receipt)
                    .map_err(|_| error(ErrorCode::InvalidJson, "application receipt failed"))?,
            )
        }
        _ => {
            return Err(error(
                ErrorCode::UnsupportedCapability,
                "unqualified no-op apply backend",
            ))
        }
    };
    if noop.receipt().changed_parameters != 0
        || noop.receipt().before_parameters != noop.receipt().after_parameters
        || noop.receipt().before_buffers != noop.receipt().after_buffers
    {
        return Err(error(
            ErrorCode::IntegrityMismatch,
            "control plan is not a no-op; source/buffer identity changed",
        ));
    }
    let store = ArtifactStore::create(
        store_root,
        StoreLimits {
            total_bytes: 16_777_216,
            max_blob_bytes: 1_048_576,
            max_entries: 1024,
        },
    )?;
    let exported = export_candidate(&store, &noop, smoke, CoreExportLimits::default())?;
    let restored = restore_source(&store, &exported.export_root, CoreExportLimits::default())?;
    verify_identity(source, &restored)?;
    cancel.check()?;
    let mut inputs = Vec::new();
    let mut seeds = Vec::new();
    for seed in &p.seeds {
        for case in &p.tasks {
            inputs.push(vec![f64::from(case.x1), f64::from(case.x2)]);
            seeds.push(*seed);
        }
    }
    let hit = cached.get(&evidence_key).cloned();
    let source_arm = || -> Result<Arc<Inference>> {
        if let Some(value) = &hit {
            confirmed(value)?;
            Ok(value.clone())
        } else {
            let value = runtime.infer_seeded(
                source,
                inputs.clone(),
                seeds.clone(),
                p.worker_limits.clone(),
                cancel,
            )?;
            confirmed(&value)?;
            Ok(Arc::new(value))
        }
    };
    let noop_arm = || -> Result<Inference> {
        let value = runtime.infer_seeded(
            &restored,
            inputs.clone(),
            seeds.clone(),
            p.worker_limits.clone(),
            cancel,
        )?;
        confirmed(&value)?;
        Ok(value)
    };
    let (original, no_op) = match p.order {
        Order::OriginalFirst => (source_arm()?, noop_arm()?),
        Order::NoOpFirst => {
            let b = noop_arm()?;
            (source_arm()?, b)
        }
    };
    let a = &original.receipt().outputs;
    let b = &no_op.receipt().outputs;
    if a.len() != inputs.len()
        || b.len() != a.len()
        || a.iter().zip(b).any(|(a, b)| {
            a.len() != b.len() || a.iter().zip(b).any(|(a, b)| a.to_bits() != b.to_bits())
        })
    {
        return Err(error(
            ErrorCode::SmokeFailed,
            "no-op actual fresh-process behavior is not exact paired source parity",
        ));
    }
    let head = arithmetic::HEADS.iter().position(|v| *v == p.head).unwrap();
    let mut rows = Vec::new();
    let mut source_correct = 0;
    let mut noop_correct = 0;
    for (index, case) in p.tasks.iter().enumerate() {
        let expected = truth(&p.head, case);
        let mut ac = true;
        let mut bc = true;
        for (repeat, seed) in p.seeds.iter().enumerate() {
            let ordinal = repeat * p.tasks.len() + index;
            ac &= (a[ordinal][head] - expected).abs() <= 1e-10;
            bc &= (b[ordinal][head] - expected).abs() <= 1e-10;
            rows.push(json!({"task":case.item,"group":case.group,"seed":seed,"expected":expected,"original":a[ordinal][head],"no_op":b[ordinal][head],"original_correct":(a[ordinal][head]-expected).abs()<=1e-10,"no_op_correct":(b[ordinal][head]-expected).abs()<=1e-10}));
        }
        source_correct += u64::from(ac);
        noop_correct += u64::from(bc);
    }
    if source_correct != noop_correct {
        return Err(error(
            ErrorCode::IntegrityMismatch,
            "no-op accuracy changed despite exact pairing",
        ));
    }
    source.verify_integrity()?;
    cancel.check()?;
    // Cache success only AFTER no-op apply/export/reload and actual paired parity.
    if hit.is_none() {
        if cached.len() >= 16 {
            let first = cached.keys().next().unwrap().clone();
            cached.remove(&first);
        }
        cached.insert(evidence_key.clone(), original.clone());
    }
    let count = p.tasks.len();
    let report = json!({"schema_version":1,"kind":"paired_original_no_op_controls","scope":"public_synthetic_fixture_audit_not_scientific_acceptance_or_holdout_authority","evidence_key":evidence_key,"protocol":p,"source_cache_hit":hit.is_some(),"cache_entries":cached.len(),"source_execution_receipt":hash(original.receipt())?,"source_execution":original.receipt(),"no_op_execution":no_op.receipt(),"application":application,"export":exported,"fresh_process_no_op_reload":true,"exact_parameters_buffers_aliases_assets":true,"exact_behavior":true,"observations":rows,"unique_tasks":count,"unique_groups":p.tasks.iter().map(|c|&c.group).collect::<BTreeSet<_>>().len(),"task_seed_pairs":inputs.len(),"duplicates_dropped":duplicates,"original_correct_unique_tasks":source_correct,"no_op_correct_unique_tasks":noop_correct,"original_accuracy":source_correct as f64/count as f64,"no_op_accuracy":noop_correct as f64/count as f64,"no_op_accuracy_gain":0.0,"no_op_positive_gain_eligible":false,"new_source_forward_calls":if hit.is_some(){0}else{inputs.len()},"new_no_op_forward_calls":inputs.len(),"additional_cost_scope":"native_apply_export_reload_and_parent_runtime_setup_cpu_memory_unmeasured_additional_unknown;cache_hit_reuses_prior_source_evidence_not_free_total_execution","outputs_eligible":false,"promotion_status":"not_evaluated"});
    let id = hash(&report)?;
    Ok(PairedControls { id, report })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn protocol() -> Protocol {
        Protocol {
            schema_version: 1,
            scope: "public_synthetic_fixture_audit".into(),
            protocol: hash(&"protocol").unwrap(),
            context: hash(&"context").unwrap(),
            head: "sum".into(),
            tasks: arithmetic::inventory().unwrap().cases[..2].to_vec(),
            seeds: vec![17, 42],
            settings: Settings::default(),
            order: Order::OriginalFirst,
            worker_limits: WorkerLimits::default(),
        }
    }
    #[test]
    fn paired_controls_deduplicate_exact_tasks_but_reject_relabels_and_seeds() {
        let mut p = protocol();
        p.tasks.push(p.tasks[0].clone());
        let (p, dropped) = normalize(p).unwrap();
        assert_eq!(p.tasks.len(), 2);
        assert_eq!(dropped, 1);
        let mut bad = p.clone();
        bad.tasks[0].x1 = 7;
        assert!(normalize(bad).is_err());
        let mut bad = p;
        bad.seeds = vec![17, 17];
        assert!(normalize(bad).is_err());
    }
    #[test]
    fn paired_controls_protocol_hash_binds_every_changed_evidence_dimension() {
        let p = protocol();
        let before = hash(&p).unwrap();
        let mut variants = Vec::new();
        let mut q = p.clone();
        q.head = "maximum".into();
        variants.push(q);
        let mut q = p.clone();
        q.context = hash(&"different context").unwrap();
        variants.push(q);
        let mut q = p.clone();
        q.protocol = hash(&"different protocol").unwrap();
        variants.push(q);
        let mut q = p.clone();
        q.tasks.reverse();
        variants.push(q);
        let mut q = p.clone();
        q.order = Order::NoOpFirst;
        variants.push(q);
        let mut q = p.clone();
        q.seeds.reverse();
        variants.push(q);
        let mut q = p;
        q.worker_limits.wall_ms += 1;
        variants.push(q);
        for q in variants {
            assert_ne!(before, hash(&q).unwrap());
        }
    }
    #[test]
    fn paired_controls_runtime_settings_and_audit_scope_cannot_silently_change() {
        let mut p = protocol();
        p.settings.precision = "f32".into();
        assert!(normalize(p).is_err());
        let mut p = protocol();
        p.scope = "private_holdout".into();
        assert!(normalize(p).is_err());
    }
}
