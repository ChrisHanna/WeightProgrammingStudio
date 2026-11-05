//! Fresh-process component evidence only; never acceptance or promotion authority.
//! The coordinator must freeze before export and bind this capability to its intent.
use super::*;
use nf_checkpoint::store::ArtifactStore;
use nf_core::study::ModelIdentity;
use nf_graph::inventory::tensor_set_digest;
use nf_surgeon::export::{restore_source, CoreExportLimits};

fn thread_cpu_ns() -> Option<u64> {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: a valid writable timespec; this only reads the current thread clock.
    if unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut time) } != 0 {
        return None;
    }
    u64::try_from(time.tv_sec)
        .ok()?
        .checked_mul(1_000_000_000)?
        .checked_add(u64::try_from(time.tv_nsec).ok()?)
}
/// Registry graph identity, not a caller-selected graph assertion.
pub fn reload_graph(source: &SourceState) -> Result<Vec<u8>> {
    match source.manifest().architecture.id.as_str() {
        "fixture.cpu" if source.manifest().parameters.get("w").is_some_and(|s| s.shape == vec![2, 2]) =>
            Ok(br#"{"factory":"fixture.cpu","operation":"dense_linear","shape":[2,2],"version":1}"#.to_vec()),
        "nf.fixture.linear_buffered" => canonical(&json!({"factory":"nf.fixture.linear_buffered","operation":"dense_linear_with_offset","shape":[2,2],"version":1})),
        "nf.arithmetic.relu" => nf_surgeon::transaction::arithmetic_graph(),
        _ => Err(error(ErrorCode::UnsupportedCapability, "reload graph not registered")),
    }
}
fn identity(source: &SourceState) -> Result<ModelIdentity> {
    Ok(ModelIdentity {
        checkpoint: ContentDigest::of_bytes(source.checkpoint_bytes()),
        manifest: source.manifest_digest().clone(),
        parameters: source.manifest().parameter_digest.clone(),
        buffers: tensor_set_digest(&source.manifest().buffers)?,
        graph: ContentDigest::of_bytes(&reload_graph(source)?),
        config: source.manifest().architecture.config.clone(),
        tokenizer: source.manifest().assets.get("tokenizer").cloned(),
        system_state: None,
        runtime_artifact: None,
    })
}
/// Native executed expectations; no Deserialize, JSON constructor or live model handle.
///
/// ```compile_fail
/// use nf_experiments::model_worker::reload::FrozenReloadProbes;
/// let _: FrozenReloadProbes = serde_json::from_str("{}").unwrap();
/// ```
pub struct FrozenReloadProbes {
    id: ContentDigest,
    model: ModelIdentity,
    inputs: Vec<Vec<f64>>,
    seeds: Vec<u64>,
    expected: Vec<Vec<f64>>,
    freeze_wall_ms: u64,
    freeze_cpu_ns: Option<u64>,
    reference_payload_bytes: u64,
}
impl FrozenReloadProbes {
    /// Execute the fixed native reference NOW, independently of subsequent reload.
    /// No expected-output override or tolerance override is accepted.
    pub fn freeze(
        reference: &SourceState,
        model: ModelIdentity,
        inputs: Vec<Vec<f64>>,
        seeds: Vec<u64>,
    ) -> Result<Self> {
        let start = boot_ms();
        let cpu = thread_cpu_ns();
        reference.verify_integrity()?;
        let expected = preflight(reference, &inputs)?;
        if seeds.len() != inputs.len() || model != identity(reference)? {
            return Err(error(
                ErrorCode::SourceMismatch,
                "frozen model identity/seed schedule mismatch",
            ));
        }
        let id = ContentDigest::of_bytes(&canonical(&json!({
            "kind":"native_pre_export_reload_probes_v1", "model":model,
            "inputs":inputs, "seeds":seeds, "expected":expected,
            "reference_code":ContentDigest::of_bytes(include_bytes!("../model_worker.rs")),
            "arithmetic_code":ContentDigest::of_bytes(include_bytes!("../../../nf-surgeon/src/transaction/arithmetic.rs"))
        }))?);
        Ok(Self {
            id,
            model,
            inputs,
            seeds,
            expected,
            freeze_wall_ms: (boot_ms() - start).max(0) as u64,
            freeze_cpu_ns: cpu.and_then(|start| thread_cpu_ns()?.checked_sub(start)),
            reference_payload_bytes: reference.payload_bytes()?,
        })
    }
    pub fn id(&self) -> &ContentDigest {
        &self.id
    }
    pub fn model(&self) -> &ModelIdentity {
        &self.model
    }
    /// Untrusted report replay diagnostic ONLY: never creates a process witness.
    /// Useful for detecting damaged archived parity/inventory evidence.
    pub fn audit_child_report(&self, restored: &SourceState, child: &Value) -> Result<()> {
        preflight(restored, &self.inputs)?;
        if identity(restored)? != self.model {
            return Err(error(
                ErrorCode::SourceMismatch,
                "audit source differs from freeze",
            ));
        }
        verify_report(restored, child, &self.inputs, &self.seeds, &self.expected)
    }
}
/// Audit data, not a capability. Missing CPU/RSS means no child was measured.
#[derive(Debug, Clone, Serialize)]
pub struct ReloadReport {
    pub schema_version: u32,
    pub export_root: ContentDigest,
    pub probes: ContentDigest,
    pub model: ModelIdentity,
    pub runtime: ContentDigest,
    pub verifier: ContentDigest,
    pub limits: Limits,
    pub status: String,
    pub failure: Option<String>,
    pub job: Option<String>,
    pub child_report: Option<Value>,
    pub child_started: Option<bool>,
    pub cleanup_confirmed: bool,
    pub cpu_ms: Option<u64>,
    pub peak_rss_bytes: Option<u64>,
    pub child_wall_ms: Option<u64>,
    pub total_wall_ms: u64,
    pub freeze_wall_ms: u64,
    pub freeze_cpu_ns: Option<u64>,
    pub parent_thread_cpu_ns: Option<u64>,
    pub restored_payload_bytes: u64,
    pub reference_payload_bytes: u64,
    pub staged_bytes: u64,
    pub accounting_scope: String,
    pub outputs_eligible: bool,
    pub promotion_status: String,
}
/// Only native verification can mint this; report JSON cannot reconstruct it.
///
/// ```compile_fail
/// use nf_experiments::model_worker::reload::ReloadWitness;
/// let _: ReloadWitness = serde_json::from_str("{}").unwrap();
/// ```
pub struct ReloadWitness {
    report: ReloadReport,
}
impl ReloadWitness {
    pub fn report(&self) -> &ReloadReport {
        &self.report
    }
}
/// Failed attempts retain accounting and cleanup evidence but carry no witness.
pub struct ReloadAttempt {
    report: ReloadReport,
    witness: Option<ReloadWitness>,
}
impl ReloadAttempt {
    pub fn report(&self) -> &ReloadReport {
        &self.report
    }
    pub fn witness(&self) -> Option<&ReloadWitness> {
        self.witness.as_ref()
    }
    pub fn into_witness(self) -> Option<ReloadWitness> {
        self.witness
    }
}
impl QualifiedTorch {
    /// Restore only CAS bytes. No candidate/source handle or expected JSON enters here.
    /// Every attempted verification returns audit data, including cancellation/failure.
    pub fn verify_reload(
        &self,
        store: &ArtifactStore,
        root: &ContentDigest,
        probes: &FrozenReloadProbes,
        limits: Limits,
        cancel: &CancellationToken,
    ) -> ReloadAttempt {
        self.verify_reload_registered(store, root, probes, limits, cancel, &mut |_| Ok(()))
    }
    /// Crate-private native launch handshake. Only a live successful execution
    /// can produce a witness; registration and archived JSON never produce one.
    pub(crate) fn verify_reload_registered(
        &self,
        store: &ArtifactStore,
        root: &ContentDigest,
        probes: &FrozenReloadProbes,
        limits: Limits,
        cancel: &CancellationToken,
        register: &mut dyn FnMut(&OwnedFd) -> Result<()>,
    ) -> ReloadAttempt {
        let start = boot_ms();
        let cpu = thread_cpu_ns();
        let mut report = ReloadReport {
            schema_version: 1, export_root: root.clone(), probes: probes.id.clone(),
            model: probes.model.clone(), runtime: self.identity.clone(),
            verifier: ContentDigest::of_bytes(include_bytes!("reload.rs")), limits: limits.clone(),
            status: "incomplete".into(),
            failure: None, job: None, child_report: None, child_started: Some(false), cleanup_confirmed: true,
            cpu_ms: None, peak_rss_bytes: None, child_wall_ms: None, total_wall_ms: 0,
            freeze_wall_ms: probes.freeze_wall_ms, freeze_cpu_ns: probes.freeze_cpu_ns,
            parent_thread_cpu_ns: None, restored_payload_bytes: 0,
            reference_payload_bytes: probes.reference_payload_bytes, staged_bytes: 0,
            accounting_scope: "wall:restore+staging+verify+child;child_cpu/rss:wait4;parent_cpu:calling_thread;bytes:payload_and_staging_not_allocator_peak;helper_thread_cpu_parent_rss_partial_staging_and_store_io_cost_unknown;completion_record_io_excluded;no_budget_settlement".into(),
            outputs_eligible: false, promotion_status: "not_evaluated".into(),
        };
        let outcome = (|| -> Result<()> {
            limits.validate()?;
            cancel.check()?;
            self.verify_runtime()?;
            let mut bounds = CoreExportLimits::default();
            bounds.export.max_checkpoint_bytes = 1_048_576;
            bounds.export.max_asset_bytes = 1_048_576;
            bounds.export.max_total_bytes = 2_097_152;
            bounds.export.max_tensors = 64;
            bounds.reload.max_checkpoint_bytes = 1_048_576;
            bounds.reload.max_tensor_bytes = 1_048_576;
            bounds.reload.max_tensors = 64;
            bounds.reload.max_asset_bytes = 1_048_576;
            let restored = restore_source(store, root, bounds)?;
            report.restored_payload_bytes = restored.payload_bytes()?;
            cancel.check()?;
            // Validate factory/layout before looking up its registered graph.
            preflight(&restored, &probes.inputs)?;
            if identity(&restored)? != probes.model {
                return Err(error(
                    ErrorCode::SourceMismatch,
                    "standalone identity differs from independent freeze",
                ));
            }
            let (job, assets) = self.stage_source(&restored)?;
            report.job = Some(job.file_name().unwrap().to_string_lossy().into_owned());
            let request = json!({"schema_version":1,"mode":"infer","assets":assets,"inputs":probes.inputs,"seeds":probes.seeds});
            report.staged_bytes = canonical(&**restored.manifest())?.len() as u64
                + restored.checkpoint_bytes().len() as u64
                + restored
                    .assets()
                    .values()
                    .map(|b| b.len() as u64)
                    .sum::<u64>()
                + canonical(&request)?.len() as u64;
            record(&job.join("request.json"), &request)?;
            record(&job.join("reload-intent.json"), &report)?;
            // If launch fails without returning a reaped measurement, cleanup is unknown.
            report.cleanup_confirmed = false;
            report.child_started = None;
            let run = self.launch_registered(&job, &limits, cancel, register)?;
            report.child_started = Some(true);
            report.cleanup_confirmed = true;
            report.cpu_ms = Some(run.cpu_ms);
            report.peak_rss_bytes = Some(run.rss_bytes);
            report.child_wall_ms = Some(run.wall_ms);
            if !run.success {
                return Err(error(ErrorCode::SmokeFailed, &run.status));
            }
            let doc = CanonicalDocument::parse(
                &run.stdout,
                JsonLimits {
                    max_bytes: limits.output_bytes,
                    max_depth: 32,
                    max_nodes: 20000,
                },
            )?;
            report.child_report = Some(doc.value().clone());
            // Existing verifier checks actual runtime buffers/aliases/dtypes/bytes/modes,
            // and compares outputs to expectations from BEFORE this reload.
            probes.audit_child_report(&restored, doc.value())?;
            restored.verify_integrity()?;
            self.verify_runtime()?;
            cancel.check()?;
            Ok(())
        })();
        report.total_wall_ms = (boot_ms() - start).max(0) as u64;
        report.parent_thread_cpu_ns = cpu.and_then(|start| thread_cpu_ns()?.checked_sub(start));
        report.status = if outcome.is_ok() {
            "verified_fresh_process_reload"
        } else {
            "failed_reload"
        }
        .into();
        report.failure = outcome.err().map(|e| e.to_string());
        // Persist failures too. A failed evidence write must never produce a witness.
        if let Some(job) = &report.job {
            if let Err(e) = record(&self.root.join(job).join("reload-completion.json"), &report) {
                report.status = "failed_reload".into();
                report.failure = Some(e.to_string());
            }
        }
        // Final cancellation check also fences completion-record I/O.
        if let Err(e) = cancel.check() {
            report.status = "failed_reload".into();
            report.failure = Some(e.to_string());
        }
        let witness = (report.status == "verified_fresh_process_reload").then(|| ReloadWitness {
            report: report.clone(),
        });
        ReloadAttempt { report, witness }
    }
}
