//! Versioned joined synthetic mixed-five executor. Never a linear qualification
//! or an operator grant. Publication still requires the protected release owner.
use super::*;
use crate::{arithmetic, intake, sandbox, validation};
use std::sync::atomic::AtomicBool;
use std::time::Duration;

pub const SCOPE: &str = "native_arithmetic_mixed_five_fixture_v1_not_strict_publication";
const STORAGE: u64 = 128 << 20;
const EXEC: ExecutionLimits = ExecutionLimits {
    max_verification_bytes: 16384,
    max_smoke_work: 65536,
    max_smoke_vector_elements: 6,
};
const SCORER: &[u8] = include_bytes!("../../validation/reference_worker.py");
pub fn profile_id() -> ContentDigest {
    hash(&(
        ContentDigest::of_bytes(include_bytes!("arithmetic_final.rs")),
        ContentDigest::of_bytes(include_bytes!("arithmetic_source.rs")),
        supervised::profile(),
        ContentDigest::of_bytes(SCORER),
    ))
    .expect("finite profile")
}
pub fn runtime_profile() -> Result<RuntimeProfile> {
    let mut r = super::runtime_profile()?;
    r.backend.id = "nf.final.arithmetic_mixed_five".into();
    r.backend.config = profile_id();
    r.max_output_length = 5;
    r.state_reset_protocol = profile_id();
    r.code_revision = hash(&(r.code_revision, profile_id()))?;
    r.environment = hash(&(
        &r.physical_device,
        &r.driver,
        &r.dependencies,
        &r.code_revision,
    ))?;
    Ok(r)
}
fn export_limits() -> CoreExportLimits {
    let mut b = CoreExportLimits::default();
    b.materialization.load.max_checkpoint_bytes = 4096;
    b.materialization.load.max_tensor_bytes = 336;
    b.materialization.load.max_tensors = 2;
    b.materialization.load.max_asset_bytes = 2048;
    b.materialization.serialization.max_checkpoint_bytes = 4096;
    b.materialization.serialization.max_tensor_bytes = 336;
    b.materialization.serialization.max_header_bytes = 2048;
    b.materialization.serialization.max_staging_bytes = 8192;
    b.export.max_checkpoint_bytes = 4096;
    b.export.max_asset_bytes = 2048;
    b.export.max_total_bytes = 16384;
    b.export.max_tensors = 2;
    b.reload = b.materialization.load;
    b.smoke = EXEC;
    b
}
fn truth(x: &[f64]) -> [f64; 5] {
    [
        x[0] + x[1],
        x[0] - x[1],
        x[0].max(x[1]),
        x[0].min(x[1]),
        (x[0] - x[1]).abs(),
    ]
}
fn entries(snapshot: &intake::Snapshot) -> Result<Value> {
    match &snapshot.manifest().draft.interface {
        intake::Interface::SingleShot { score, oracle, .. } => {
            Ok(json!({"score":score,"oracle":oracle}))
        }
        _ => Err(deny("arithmetic sum requires single-shot scorer")),
    }
}
/// Exact native capabilities, not restored approval/receipt JSON.
pub struct SumScorer<'a> {
    pub snapshot: &'a intake::Snapshot,
    pub approval: &'a validation::Approval,
    pub runtime: &'a sandbox::QualifiedDocker,
}
/// Native run result, no Deserialize. Not publication authority.
pub struct ArithmeticRun {
    decision: Decision,
    export: ContentDigest,
}
/// Re-derived artifact/native/scorer/statistics witness, not a release grant.
/// The protected promotion owner must explicitly admit this versioned profile.
pub struct ArithmeticQualification {
    decision: Decision,
    export: ContentDigest,
}
impl ArithmeticQualification {
    pub fn decision(&self) -> &Decision {
        &self.decision
    }
    pub fn export_root(&self) -> &ContentDigest {
        &self.export
    }
}
impl ArithmeticRun {
    pub fn decision(&self) -> &Decision {
        &self.decision
    }
    pub fn export_root(&self) -> &ContentDigest {
        &self.export
    }
}
fn supported(
    c: &StudyContract,
    edit: &CompiledPlan,
    noop: &CompiledPlan,
    smoke: &Validated<SmokeSuite>,
    scorer: &SumScorer<'_>,
) -> Result<()> {
    scorer
        .approval
        .check_environment(scorer.snapshot, scorer.runtime)?;
    entries(scorer.snapshot)?;
    let p = edit.program();
    if canonical(&c.runtime)? != canonical(&runtime_profile()?)?
        || c.portfolio.independent.scope != ClaimScope::FixtureOnly
        || c.portfolio.independent.isolation != Isolation::TrustedFixtureBroker
        || c.statistics.estimator != Estimator::DeterministicExact
        || c.statistics.repeats != 1
        || c.seeds.len() != 1
        || c.objective.kind == ObjectiveKind::CausalClaim
        || c.intervention.allow_buffer_write
        || c.intervention.allow_graph_change
        || !acceptance::has_resource_ceilings(c)
        || c.portfolio.bindings.len() != 5
        || c.portfolio.gyms.len() != 5
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
        || p.base_model != noop.program().base_model
        || !p.lineage.parent_candidates.is_empty()
        || smoke.adapter != "nf.arithmetic_relu.v1"
        || smoke.probes.is_empty()
        || smoke.probes.len() > 8
        || smoke
            .probes
            .iter()
            .any(|p| p.input.len() != 2 || p.expected.as_ref().is_some_and(|v| v.len() != 5))
        || scorer.snapshot.manifest().draft.name != "user.arithmetic.sum"
    {
        return Err(deny(
            "exact arithmetic mixed-five contract/program/snapshot profile required",
        ));
    }
    c.permissions.check_tool(ToolKind::DirectEdit)?;
    p.lineage.method.require_fully_gradient_free()?;
    for (head, b) in c.portfolio.bindings.iter().enumerate() {
        let g = &c.portfolio.gyms[&b.gym];
        let expected = if head == 0 {
            "user.arithmetic.sum".into()
        } else {
            format!("arithmetic.{}.v1", arithmetic::HEADS[head])
        };
        let validation = if head == 0 {
            scorer.approval.id()
        } else {
            &profile_id()
        };
        if g.name != expected
            || !b.required
            || !b.schedule.final_required
            || b.metrics.len() != if head == 0 { 3 } else { 1 }
            || g.adapter.id != "nf.final.arithmetic_mixed_five"
            || g.adapter.version != "1"
            || g.adapter.config != profile_id()
            || g.scoring_protocol != profile_id()
            || b.validation != *validation
            || g.validation != *validation
            || (head == 0
                && (g.code != *scorer.snapshot.id()
                    || g.approval.as_ref() != Some(scorer.approval.id())))
            || (head > 0 && (g.code != profile_id() || !b.roles.contains(&GymRole::Retention)))
        {
            return Err(deny(
                "all five exact required gym bindings/approval identities required",
            ));
        }
        let m = &b.metrics[0];
        let schema = &g.metrics[&m.metric];
        if !m.required
            || m.metric != "accuracy"
            || m.statistic != Statistic::Exact
            || m.measurement.is_some()
            || schema.unit != MetricUnit::Fraction
            || schema.direction != Direction::HigherIsBetter
            || m.comparator != Comparator::Original
            || (head == 0 && m.threshold.min_gain.is_none_or(|v| v <= 0.0))
            || (head > 0 && m.threshold.max_regression != Some(0.0))
        {
            return Err(deny(
                "positive sum gain and four individually zero-regression accuracy gates required",
            ));
        }
        if head == 0 {
            for (name, domain, unit, scope) in [
                (
                    "latency",
                    MetricDomain::Latency,
                    MetricUnit::Milliseconds,
                    MeasurementScope::ForwardLatency,
                ),
                (
                    "memory",
                    MetricDomain::Memory,
                    MetricUnit::Bytes,
                    MeasurementScope::ProcessRss,
                ),
            ] {
                let m = b
                    .metrics
                    .iter()
                    .find(|metric| metric.metric == name)
                    .ok_or_else(|| deny("frozen resource metric missing"))?;
                let schema = g
                    .metrics
                    .get(name)
                    .ok_or_else(|| deny("resource schema missing"))?;
                let q = m
                    .measurement
                    .as_ref()
                    .ok_or_else(|| deny("resource protocol missing"))?;
                if !m.required
                    || m.comparator != Comparator::Original
                    || m.statistic != Statistic::Maximum
                    || m.threshold.ceiling.is_none()
                    || m.threshold.min_gain.is_some()
                    || m.threshold.max_regression.is_some()
                    || schema.domain != domain
                    || schema.unit != unit
                    || schema.direction != Direction::LowerIsBetter
                    || q.scope != scope
                    || q.statistic != Statistic::Maximum
                    || q.warmup != 1
                    || q.repetitions != 3
                    || q.order != MeasurementOrder::PairedInterleaved
                    || q.synchronization != Synchronization::CpuMonotonic
                    || q.timer.id != "nf.final.monotonic"
                    || q.timer.version != "1"
                    || q.timer.config != profile_id()
                {
                    return Err(deny("unsupported frozen resource scope or protocol"));
                }
            }
        }
    }
    if c.baselines.iter().filter(|b| b.enabled).count() != 3
        || c.baselines.iter().any(|b| {
            b.enabled
                && (!matches!(
                    b.kind,
                    BaselineKind::Original | BaselineKind::NoOp | BaselineKind::BestKnown
                ) || b.backend.id != c.runtime.backend.id
                    || b.backend.config != profile_id())
        })
    {
        return Err(deny("exact three matched controls required"));
    }
    for plan in [edit, noop] {
        let a = &plan.summary().allocations;
        if plan.program().operations.len() > 8
            || canonical(plan.program())?.len() > 65536
            || a.estimated_working_bytes > 1 << 20
            || a.kernel_work_units > 65536
        {
            return Err(deny("bounded supplied arithmetic IR required"));
        }
    }
    Ok(())
}
fn persist(
    ledger: &mut Ledger,
    lease: &Lease,
    name: &str,
    body: &impl Serialize,
) -> Result<ContentDigest> {
    let tx = ledger
        .connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(sql)?;
    check_lease(&tx, &ledger.store, lease, &files::moment()?)?;
    let id = record(&tx, lease, name, body)?;
    ensure_live(lease)?;
    tx.commit().map_err(sql)?;
    Ok(id)
}
impl Ledger {
    /// Restores the validated scorer snapshot and approval from the store and executes
    /// one-shot native arithmetic evaluation.
    pub fn run_arithmetic_final_from_store(
        &mut self,
        lease: &Lease,
        store: &ArtifactStore,
        edit: &CompiledPlan,
        noop: &CompiledPlan,
        smoke: &Validated<SmokeSuite>,
        code_id: &ContentDigest,
        approval_id: &ContentDigest,
        sandbox_root: &std::path::Path,
        cancel: &CancellationToken,
    ) -> Result<ArithmeticRun> {
        let snapshot = intake::restore(store, code_id)?;
        let runtime = sandbox::QualifiedDocker::qualify(true, sandbox_root)?;
        let approval = validation::restore_approval(store, approval_id, &snapshot, &runtime)?;
        self.run_arithmetic_final(
            lease,
            store,
            edit,
            noop,
            smoke,
            SumScorer {
                snapshot: &snapshot,
                approval: &approval,
                runtime: &runtime,
            },
            cancel,
        )
    }

    /// One-shot native arithmetic execution. Operator routing and promotion are
    /// deliberately NOT inferred from this computation capability.
    pub fn run_arithmetic_final(
        &mut self,
        lease: &Lease,
        store: &ArtifactStore,
        edit: &CompiledPlan,
        noop: &CompiledPlan,
        smoke: &Validated<SmokeSuite>,
        scorer: SumScorer<'_>,
        cancel: &CancellationToken,
    ) -> Result<ArithmeticRun> {
        self.final_fence(lease, cancel)?;
        let c = self.contract(&lease.contract)?;
        supported(&c, edit, noop, smoke, &scorer)?;
        let hard = bounds::hard_memory_limit(c.limits.host_memory_bytes)?;
        // Controller and its one Docker control helper inherit this hard AS cap;
        // guest cgroup, guardian and static worker fit the remaining envelope.
        if hard > 512 << 20
            || c.limits.host_memory_bytes < (2 << 30)
            || c.limits.storage_bytes < STORAGE
            || c.limits.cpu_ms < 60_000
            || c.limits.wall_ms < 60_000
            || self.limits.max_database_bytes > 16 << 20
        {
            return Err(deny("arithmetic requires dedicated <=512MiB AS controller and frozen 2GiB/128MiB/60s envelope"));
        }
        if read(&self.connection, &lease.contract, "intent")?.is_some()
            || self.frozen_final_panel(&lease.contract)?.is_some()
        {
            return Err(deny(
                "one-shot arithmetic attempt; reconcile interrupted intent, never redispatch",
            ));
        }
        let resources = Resources {
            cpu_ms: 60_000,
            attempts: 1,
            evaluations: 10_000,
            storage_bytes: STORAGE,
            host_bytes: c.limits.host_memory_bytes,
            workers: 1,
            ..Default::default()
        };
        let page: u64 = self
            .connection
            .query_row("PRAGMA page_size", [], |r| r.get(0))
            .map_err(sql)?;
        self.connection
            .pragma_update(
                None,
                "max_page_count",
                self.limits.max_database_bytes / page,
            )
            .map_err(sql)?;
        let cpu = cpu_ns();
        let owner = native_process::Owner::current()?;
        let reservation = self.reserve_with(lease, Request { schema_version:1,
            operation:hash(&(SCOPE, &lease.contract))?, attempt:0, parent:None,
            kind:WorkKind::Verification, resources:resources.clone(), wall_ms:60_000, external:None },
            |tx, r, deadline| { record(tx, lease, "intent", &json!({"profile":profile_id(),"scope":SCOPE,
                "reservation":r.id(),"resources":resources,"operation_deadline":deadline.min(lease.deadline),
                "native_process":owner,"snapshot":scorer.snapshot.id(),"approval":scorer.approval.id(),
                "sandbox_environment":scorer.runtime.environment_identity(),"sandbox_root":scorer.runtime.durable_root_identity()?,
                "edit":edit.digest(),"noop":noop.digest(),
                "smoke":smoke.digest()?,"qualification_cost":"charged_full_reserved_ceiling_not_measured_per_run"}))?; Ok(()) })?;
        fault("arithmetic_after_intent")?;
        let result = self.execute_arithmetic(
            lease,
            store,
            edit,
            noop,
            smoke,
            &scorer,
            cancel,
            &reservation,
            cpu,
        );
        // Errors remain explicit incomplete intents; recovery never retries work.
        result
    }
    fn execute_arithmetic(
        &mut self,
        lease: &Lease,
        store: &ArtifactStore,
        edit: &CompiledPlan,
        noop: &CompiledPlan,
        smoke: &Validated<SmokeSuite>,
        scorer: &SumScorer<'_>,
        cancel: &CancellationToken,
        reservation: &budgets::Reservation,
        cpu: u64,
    ) -> Result<ArithmeticRun> {
        let c = self.contract(&lease.contract)?;
        self.allocation_fence(lease, reservation, cancel, cpu)?;
        let source = arithmetic_source::resolve(store, &c.source)?;
        let best = arithmetic_source::resolve(store, &c.best_known)?;
        if source.state().binding(c.source.graph.clone()) != edit.program().base_model {
            return Err(deny("arithmetic supplied IR source mismatch"));
        }
        let candidate = execute_cancellable(edit, source.state(), smoke, EXEC, cancel)?;
        self.allocation_fence(lease, reservation, cancel, cpu)?;
        let control = execute_cancellable(noop, source.state(), smoke, EXEC, cancel)?;
        let a = export_candidate(store, &candidate, smoke, export_limits())?;
        let n = export_candidate(store, &control, smoke, export_limits())?;
        let model = restore_source(store, &a.export_root, export_limits())?;
        let no_op = restore_source(store, &n.export_root, export_limits())?;
        let finalist = arithmetic_source::identity(&model)?;
        if control.receipt().changed_parameters != 0
            || arithmetic_source::identity(&no_op)? != c.source
        {
            return Err(deny("arithmetic no-op changed exact source identity"));
        }
        for state in [&model, &no_op] {
            store.put(canonical(&**state.manifest())?.as_slice())?;
        }
        // Resolve three-role native catalogues. No label bytes are sent to any
        // worker before the actual finalist AND analysis plan are frozen.
        let inventory = arithmetic::inventory()?;
        let mut catalogs = BTreeMap::new();
        let mut contrasts = Vec::new();
        for (head, b) in c.portfolio.bindings.iter().enumerate() {
            for (role, split) in [
                (SplitRole::Development, &b.splits.development),
                (SplitRole::Selection, &b.splits.selection),
                (SplitRole::Holdout, &b.splits.holdout),
            ] {
                let expected: Vec<_> = inventory.cases.iter().filter(|v| v.split == role).collect();
                let dataset: splits::Dataset =
                    serde_json::from_slice(&store.read_bounded(&split.dataset, 131072)?)
                        .map_err(|_| deny("arithmetic catalogue missing/invalid"))?;
                if dataset.role != role
                    || dataset.items.len() != expected.len()
                    || split.items.len() != expected.len()
                {
                    return Err(deny(
                        "complete frozen arithmetic split required; no partial gym",
                    ));
                }
                for ((item, identity), case) in dataset.items.iter().zip(&split.items).zip(expected)
                {
                    let x = vec![case.x1 as f64, case.x2 as f64];
                    if canonical(&item.identity)? != canonical(identity)?
                        || identity.item != case.item
                        || identity.group != case.group
                        || identity.family != case.family
                        || item.input != hash(&x)?
                        || item.label != hash(&truth(&x)[head])?
                    {
                        return Err(deny(
                            "arithmetic catalogue differs from independently grouped inventory",
                        ));
                    }
                }
            }
            let items = splits::final_items(&self.connection, &lease.contract, &b.id)?;
            for m in &b.metrics {
                contrasts.push(Contrast {
                    id: hash(&(&finalist, &b.id, &m.metric))?,
                    candidate: model_identity(&finalist)?,
                    baseline: model_identity(&c.source)?,
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
        let panel = self.freeze_final_panel_with_analysis(lease, vec![finalist.clone()], &plan)?;
        let dispatch = persist(
            self,
            lease,
            "arithmetic_dispatch",
            &json!({"profile":profile_id(),"panel":panel,
            "plan":plan.record()?,"source":c.source,"noop":c.source,"best":c.best_known,"candidate":finalist,
            "edit":candidate.receipt(),"noop_edit":control.receipt(),"candidate_export":a,"noop_export":n,
            "snapshot":scorer.snapshot.id(),"approval":scorer.approval.id()}),
        )?;
        let tx = self.connection.unchecked_transaction().map_err(sql)?;
        splits::mark_open(&tx, &lease.contract, &panel, SCOPE)?;
        ensure_live(lease)?;
        tx.commit().map_err(sql)?;
        fault("arithmetic_after_panel")?;
        let sum = &c.portfolio.bindings[0];
        let items = &catalogs[&sum.id];
        let mut observations = Vec::new();
        let mut batches = Vec::new();
        let models = [source.state(), &no_op, best.state(), &model];
        for (batch_index, chunk) in items.chunks(32).enumerate() {
            self.allocation_fence(lease, reservation, cancel, cpu)?;
            let mut jobs = Vec::new();
            let mut tickets = Vec::new();
            let tx = self.connection.unchecked_transaction().map_err(sql)?;
            check_lease(&tx, &self.store, lease, &files::moment()?)?;
            for (index, item) in chunk.iter().enumerate() {
                // Every gym has an independent exposure receipt, even when its
                // model input is deduplicated across five heads.
                for b in &c.portfolio.bindings {
                    let t = &catalogs[&b.id][batch_index * 32 + index];
                    cancel.check()?;
                    let subject = exposure::subject(
                        &c,
                        &lease.contract,
                        &b.id,
                        SplitRole::Holdout,
                        t,
                        store,
                    )?;
                    exposure::admit(&tx, &c, &subject)?;
                    let ticket = exposure::begin(
                        &tx,
                        lease,
                        &c,
                        exposure::Start {
                            subject,
                            recipient: SCOPE.into(),
                            role: "final_evaluator".into(),
                            operation: hash(&(&dispatch, &b.id, &t.identity.item))?,
                            context: exposure::Context {
                                seed: Some(c.seeds[0]),
                                model: Some(model_identity(&finalist)?),
                                candidate: Some(model_identity(&finalist)?),
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
                    tickets.push(ticket);
                }
                let input: Vec<f64> =
                    serde_json::from_slice(&store.read_bounded(&item.input, 1024)?)
                        .map_err(|_| deny("numeric arithmetic input required"))?;
                jobs.push((hash(&(&dispatch, &item.identity.item))?, input));
            }
            check_lease(&tx, &self.store, lease, &files::moment()?)?;
            cancel.check()?;
            ensure_live(lease)?;
            tx.commit().map_err(sql)?;
            let requests: Vec<_> = jobs.iter().map(|(key, x)| (key, x.as_slice())).collect();
            let measured = self.supervised_arithmetic_forwards(
                lease,
                reservation,
                cancel,
                cpu,
                &dispatch,
                models,
                &requests,
            )?;
            fault("arithmetic_after_native_before_score")?;
            let mut cells = Vec::new();
            for ((_, x), measurement) in jobs.iter().zip(&measured) {
                for output in measurement
                    .arithmetic_outputs
                    .as_ref()
                    .ok_or_else(|| deny("arithmetic worker output absent"))?
                {
                    cells.push(json!({"input":{"x1":x[0],"x2":x[1]},"expected":truth(x)[0],
                        "wrong":[truth(x)[0]+1.0],"output":output[0]}));
                }
            }
            let mut request = entries(scorer.snapshot)?;
            request["mode"] = json!("gym");
            request["cells"] = json!(cells);
            let mut assets = BTreeMap::new();
            for path in scorer.snapshot.manifest().assets.keys() {
                assets.insert(path.clone(), scorer.snapshot.asset(store, path)?);
            }
            let (_, intent) = read(&self.connection, &lease.contract, "intent")?
                .ok_or_else(|| deny("intent missing"))?;
            let remaining =
                intent["operation_deadline"].as_i64().unwrap_or(0) - files::moment()?.ms;
            let deadline = Instant::now() + Duration::from_millis(remaining.max(0) as u64);
            persist(
                self,
                lease,
                &format!("arithmetic_scorer_intent_{batch_index}"),
                &json!({
                "input":hash(&request)?,"snapshot":scorer.snapshot.id(),"profile":ContentDigest::of_bytes(SCORER)}),
            )?;
            let receipt = scorer.runtime.execute_assets_authorized(
                SCORER,
                &canonical(&request)?,
                &assets,
                sandbox::Limits {
                    wall_ms: 5000,
                    cpu_ms: 5000,
                    output_bytes: 65536,
                },
                sandbox::STORAGE_JOB_BYTES,
                &AtomicBool::new(false),
                deadline,
                &|| self.allocation_fence(lease, reservation, cancel, cpu),
            )?;
            persist(
                self,
                lease,
                &format!("arithmetic_scorer_receipt_{batch_index}"),
                &receipt,
            )?;
            if receipt.status != "completed"
                || !receipt.cleanup_confirmed
                || receipt.source != ContentDigest::of_bytes(SCORER)
                || receipt.input != hash(&request)?
            {
                return Err(deny("actual complete scoped scorer receipt required"));
            }
            let raw = receipt
                .report
                .as_ref()
                .and_then(|v| v["stdout_hex"].as_str())
                .ok_or_else(|| deny("scorer stdout missing"))?;
            let result: Value = serde_json::from_slice(&sandbox::decode_hex(raw)?)
                .map_err(|_| deny("scorer report invalid"))?;
            let samples = result["samples"]
                .as_array()
                .ok_or_else(|| deny("scorer samples missing"))?;
            if samples.len() != chunk.len() * 4 {
                return Err(deny("partial arithmetic scoring rejected"));
            }
            for (index, ((item, (_, x)), measurement)) in
                chunk.iter().zip(&jobs).zip(&measured).enumerate()
            {
                let outputs = measurement.arithmetic_outputs.as_ref().unwrap();
                for arm in 0..4 {
                    let s = &samples[index * 4 + arm];
                    let expected = truth(x)[0];
                    let score = if (outputs[arm][0] - expected).abs() <= 1e-10 {
                        1.0
                    } else {
                        0.0
                    };
                    if s["index"] != json!(index * 4 + arm)
                        || s["oracle"] != json!({"ok":true,"value":expected})
                        || s["score"] != json!({"ok":true,"value":score})
                        || s["good_score"] != json!({"ok":true,"value":1.0})
                        || s["invalid_score"] != json!({"ok":true,"value":0.0})
                        || s["wrong_scores"] != json!([{"ok":true,"value":0.0}])
                    {
                        return Err(deny(
                            "user scorer trace disagrees with independent arithmetic truth",
                        ));
                    }
                }
                for (head, b) in c.portfolio.bindings.iter().enumerate() {
                    let t = &catalogs[&b.id][batch_index * 32 + index];
                    let label: f64 = serde_json::from_slice(&store.read_bounded(&t.label, 128)?)
                        .map_err(|_| deny("scalar gym label required"))?;
                    if label != truth(x)[head] {
                        return Err(deny("actual holdout label differs from frozen oracle"));
                    }
                    for metric in &b.metrics {
                        let contrast = plan
                            .protocol()
                            .contrasts
                            .iter()
                            .find(|contrast| {
                                contrast.binding == b.id && contrast.metric == metric.metric
                            })
                            .ok_or_else(|| deny("frozen arithmetic contrast missing"))?;
                        for (arm, k) in [(Arm::Baseline, 0), (Arm::Candidate, 3)] {
                            let value = match metric.metric.as_str() {
                                "accuracy" => {
                                    if (outputs[k][head] - label).abs() <= 1e-10 {
                                        1.0
                                    } else {
                                        0.0
                                    }
                                }
                                "latency" => measurement.latencies[k],
                                // The same whole-controller high-water RSS applies to both
                                // arms. It is not a per-model memory improvement claim.
                                "memory" => measurement.process_rss_bytes as f64,
                                _ => return Err(deny("unsupported arithmetic metric")),
                            };
                            observations.push(Observation {
                                contrast: contrast.id.clone(),
                                item: item.identity.item.clone(),
                                group: item.identity.group.clone(),
                                input: item.input.clone(),
                                seed: c.seeds[0],
                                arm,
                                model: if k == 3 {
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
                }
            }
            self.files.verify(self.limits.max_database_bytes)?;
            let tx = self.connection.unchecked_transaction().map_err(sql)?;
            check_lease(&tx, &self.store, lease, &files::moment()?)?;
            for (index, ticket) in tickets.into_iter().enumerate() {
                cancel.check()?;
                if ticket.body.contract != lease.contract
                    || ticket.body.family != c.study_family
                    || ticket.body.epoch != lease.epoch
                {
                    return Err(deny("arithmetic exposure ticket fence differs"));
                }
                exposure::finish(
                    &tx,
                    &ticket,
                    exposure::Stage::Completed,
                    ticket.body.intended.clone(),
                    "native_arithmetic_and_scoped_user_score",
                    exposure::Cost {
                        model_calls: Some(if index % 5 == 0 { 20 } else { 0 }),
                        ..Default::default()
                    },
                )?;
            }
            check_lease(&tx, &self.store, lease, &files::moment()?)?;
            cancel.check()?;
            ensure_live(lease)?;
            tx.commit().map_err(sql)?;
            self.files.verify(self.limits.max_database_bytes)?;
            batches.push(json!({"items":chunk,"outputs":measured.iter().map(|v|&v.arithmetic_outputs).collect::<Vec<_>>(),
                "native":measured.iter().map(|v|&v.evidence).collect::<Vec<_>>(),"scorer":hash(&receipt)?}));
        }
        for model in models {
            model.verify_integrity()?;
        }
        self.allocation_fence(lease, reservation, cancel, cpu)?;
        let data = Data {
            schema_version: 1,
            plan: plan.id().clone(),
            observations,
        };
        let compact = statistics::CompactData::from_data(&plan, &data, cancel)?;
        drop(data);
        let data_bytes = canonical(&compact).map_err(|mut error| {
            error.path = "/arithmetic/data".into();
            error
        })?;
        let analysis = plan.analyze_compact(&data_bytes, cancel)?;
        let mut report = self
            .acceptance_gate(
                lease,
                Some(Evidence {
                    plan: &plan,
                    analysis: &analysis,
                }),
            )?
            .into_report();
        // Observations are a separately verified CAS object. Repeating the
        // whole dataset inside the execution record exceeds its canonical bound
        // once resource metrics are included. The analysis retains its digest.
        self.allocation_fence(lease, reservation, cancel, cpu)?;
        let data_digest = store.put(data_bytes.as_slice())?.digest;
        if analysis.report()["data_digest"] != json!(data_digest) {
            return Err(deny("arithmetic analysis data digest differs"));
        }
        self.allocation_fence(lease, reservation, cancel, cpu)?;
        let evidence = persist(self,lease,"arithmetic_execution",&json!({"dispatch":dispatch,"batches":batches,
            "data_format":"plan_compact_v1","data_digest":data_digest,"analysis":analysis.report(),"plan":plan.record()?,"scope":SCOPE}))
            .map_err(|mut error| { error.path = "/arithmetic/execution".into(); error })?;
        mark_verified_execution(&mut report.checks, &c, &evidence);
        report.status = [
            Status::Failed,
            Status::Missing,
            Status::Incomplete,
            Status::Inconclusive,
        ]
        .into_iter()
        .find(|s| report.checks.iter().any(|c| c.required && c.status == *s))
        .unwrap_or(Status::Passed);
        report.accepted = report.status == Status::Passed;
        report.promotion_authorized = false;
        report.scope = SCOPE.into();
        let decision = Decision::from_report(report)?;
        self.allocation_fence(lease, reservation, cancel, cpu)?;
        if !supervised::cleanup_known(&self.connection, &lease.contract)? {
            return Err(deny("arithmetic native cleanup unknown"));
        }
        let _cleanup = scorer
            .runtime
            .recover_attested(&scorer.runtime.durable_root_identity()?)?;
        let store_id = self.store.clone();
        let (_, intent) = read(&self.connection, &lease.contract, "intent")?
            .ok_or_else(|| deny("intent missing"))?;
        let deadline = intent["operation_deadline"]
            .as_i64()
            .ok_or_else(|| deny("deadline missing"))?;
        let mut resources: Resources = serde_json::from_value(intent["resources"].clone())
            .map_err(|_| deny("resources missing"))?;
        let mut guest_cpu = 0_u64;
        for i in 0..batches.len() {
            let (_, r) = read(
                &self.connection,
                &lease.contract,
                &format!("arithmetic_scorer_receipt_{i}"),
            )?
            .ok_or_else(|| deny("scorer receipt missing"))?;
            guest_cpu = guest_cpu
                .checked_add(
                    r["report"]["cpu_ms"]
                        .as_u64()
                        .ok_or_else(|| deny("scorer CPU unknown"))?,
                )
                .ok_or_else(|| deny("scorer CPU overflow"))?;
        }
        resources.cpu_ms = resources
            .cpu_ms
            .checked_sub(
                guest_cpu.saturating_add(supervised::cpu_ms(&self.connection, &lease.contract)?),
            )
            .ok_or_else(|| deny("aggregate CPU envelope exhausted"))?;
        self.finish_budget_with(
            lease,
            reservation,
            Settlement {
                operation: hash(&(&dispatch, "settled"))?,
                outcome: OutcomeKind::Completed,
                usage: Usage {
                    attempts: Some(1),
                    ..Default::default()
                },
                resources_released: true,
                evidence,
            },
            false,
            |tx| {
                check_lease(tx, &store_id, lease, &files::moment()?)?;
                cancel.check()?;
                if supervision::stopped(tx, &lease.contract)?
                    || state(tx, &lease.contract)?.stage != Stage::Verify
                {
                    return Err(deny("arithmetic finalization stopped"));
                }
                record(tx, lease, "arithmetic_decision", decision.report())?;
                check_completion(lease, deadline, cpu, &resources, Some(cancel))?;
                ensure_live(lease)
            },
        )?;
        self.qualify_arithmetic_final(&lease.contract, store)?;
        Ok(ArithmeticRun {
            decision,
            export: a.export_root,
        })
    }
    /// Reopen performs fresh CAS restoration, native launch/output verification
    /// and statistical recomputation. Serialized success is never sufficient.
    pub fn qualify_arithmetic_final(
        &self,
        contract: &ContentDigest,
        store: &ArtifactStore,
    ) -> Result<ArithmeticQualification> {
        self.files.verify(self.limits.max_database_bytes)?;
        let c = self.contract(contract)?;
        let (_, intent) = read(&self.connection, contract, "intent")?
            .ok_or_else(|| deny("arithmetic intent missing"))?;
        if intent["profile"] != json!(profile_id()) || intent["scope"] != SCOPE {
            return Err(deny("arithmetic profile differs"));
        }
        let reservation: ContentDigest = serde_json::from_value(intent["reservation"].clone())
            .map_err(|_| deny("reservation missing"))?;
        budgets::snapshot_in(&self.connection, contract)?;
        if !budgets::native_released(&self.connection, &reservation)? {
            return Err(deny("arithmetic allocation unsettled"));
        }
        let request_bytes: Vec<u8> = self
            .connection
            .query_row(
                "SELECT body FROM budget_requests WHERE id=?1 AND contract=?2",
                params![reservation.as_str(), contract.as_str()],
                |row| row.get(0),
            )
            .map_err(sql)?;
        let request = CanonicalDocument::parse(&request_bytes, Limits::default())?;
        if request.value()["id"] != json!(reservation)
            || request.value()["contract"] != json!(contract)
            || request.value()["request"]["operation"] != json!(hash(&(SCOPE, contract))?)
            || request.value()["request"]["resources"] != intent["resources"]
            || request.value()["boot"] != intent["native_process"]["boot"]
        {
            return Err(deny("arithmetic reservation binding differs"));
        }
        let (dispatch_id, dispatch) = read(&self.connection, contract, "arithmetic_dispatch")?
            .ok_or_else(|| deny("dispatch missing"))?;
        let (execution_id, execution) = read(&self.connection, contract, "arithmetic_execution")?
            .ok_or_else(|| deny("execution missing"))?;
        let (_, report) = read(&self.connection, contract, "arithmetic_decision")?
            .ok_or_else(|| deny("decision missing"))?;
        let report: Report =
            serde_json::from_value(report).map_err(|_| deny("decision schema invalid"))?;
        if report.schema_version != 1
            || report.contract != *contract
            || report.revision != c.revision
            || report.scope != SCOPE
            || report.promotion_authorized
            || execution["dispatch"] != json!(dispatch_id)
            || execution["scope"] != SCOPE
            || dispatch["profile"] != json!(profile_id())
            || dispatch["snapshot"] != intent["snapshot"]
            || dispatch["approval"] != intent["approval"]
        {
            return Err(deny("arithmetic durable binding differs"));
        }
        // A self-consistent decision hash is not a historical event. Verify
        // the full journal and bind the Verify-stage anchor to the reservation
        // epoch and boot before accepting any derived evidence.
        state(&self.connection, contract)?;
        let anchor: Option<Vec<u8>> = self
            .connection
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
        let outcome_bytes: Vec<u8> = self
            .connection
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
        let export: ContentDigest =
            serde_json::from_value(dispatch["candidate_export"]["export_root"].clone())
                .map_err(|_| deny("export absent"))?;
        let noop_export = serde_json::from_value(dispatch["noop_export"]["export_root"].clone())
            .map_err(|_| deny("noop export absent"))?;
        let model = restore_source(store, &export, export_limits())?;
        let noop = restore_source(store, &noop_export, export_limits())?;
        let source = arithmetic_source::resolve(store, &c.source)?;
        let best = arithmetic_source::resolve(store, &c.best_known)?;
        let finalist = arithmetic_source::identity(&model)?;
        if dispatch["candidate"] != json!(finalist)
            || dispatch["source"] != json!(c.source)
            || dispatch["best"] != json!(c.best_known)
            || arithmetic_source::identity(&noop)? != c.source
        {
            return Err(deny("reopened arithmetic identities differ"));
        }
        let native = supervised::qualify_arithmetic_inputs(
            &self.connection,
            contract,
            [source.state(), &noop, best.state(), &model],
        )?;
        let plan = Plan::freeze(&canonical(&execution["plan"]["protocol"])?)?;
        if execution["plan"] != plan.record()? || dispatch["plan"] != plan.record()? {
            return Err(deny("reopened plan differs"));
        }
        let protocol = plan.protocol();
        if protocol.family != c.statistics.multiplicity_family
            || protocol.estimator != c.statistics.estimator
            || protocol.minimum_groups != c.statistics.minimum_groups
            || protocol.seeds != c.seeds
            || protocol.confidence != c.statistics.confidence
            || protocol.maximum_comparisons != c.statistics.maximum_comparisons
            || protocol.scope != "deterministic_fixture_only"
            || protocol.resamples != 0
            || protocol.contrasts.len()
                != c.portfolio
                    .bindings
                    .iter()
                    .map(|b| b.metrics.len())
                    .sum::<usize>()
        {
            return Err(deny(
                "arithmetic analysis differs from frozen statistical contract",
            ));
        }
        let (panel_id, panel) = self
            .frozen_final_panel(contract)?
            .ok_or_else(|| deny("panel absent"))?;
        let bindings: BTreeSet<_> = c
            .portfolio
            .bindings
            .iter()
            .filter(|b| b.schedule.final_required)
            .map(|b| b.id.clone())
            .collect();
        if panel.schema_version != 1
            || panel.contract != *contract
            || panel.revision != c.revision
            || panel.protocol != c.portfolio.independent.frozen_panel_protocol
            || panel.original != c.source
            || panel.best_known != c.best_known
            || panel.bindings != bindings
            || panel.seeds != c.seeds
            || panel.repeats != c.statistics.repeats
            || panel.comparison_family != c.statistics.multiplicity_family
            || panel.finalists != vec![finalist.clone()]
            || panel.analysis_plan.as_ref() != Some(plan.id())
            || report.panel.as_ref() != Some(&panel_id)
            || dispatch["panel"] != json!(panel_id)
            || report.plan.as_ref() != Some(plan.id())
        {
            return Err(deny("frozen arithmetic panel differs"));
        }
        let opening: String = self
            .connection
            .query_row(
                "SELECT panel FROM holdout_openings WHERE contract=?1",
                [contract.as_str()],
                |r| r.get(0),
            )
            .map_err(sql)?;
        if opening != panel_id.as_str() {
            return Err(deny("holdout opening absent"));
        }
        let mut compact_bytes = None;
        let data: Data = if let Some(reference) = execution.get("data_digest") {
            let id: ContentDigest = serde_json::from_value(reference.clone())
                .map_err(|_| deny("data digest invalid"))?;
            let bytes = store.read_bounded(&id, 1_048_576)?;
            let document = CanonicalDocument::parse(&bytes, Limits::default())?;
            if document.bytes() != bytes
                || document.digest() != id
                || execution["analysis"]["data_digest"] != json!(id)
            {
                return Err(deny("arithmetic observation artifact differs"));
            }
            match execution.get("data_format") {
                Some(Value::String(format)) if format == "plan_compact_v1" => {
                    let compact: statistics::CompactData =
                        serde_json::from_value(document.value().clone())
                            .map_err(|_| deny("compact observation schema invalid"))?;
                    let data = plan.expand_compact(&compact, &CancellationToken::default())?;
                    compact_bytes = Some(bytes);
                    data
                }
                None => serde_json::from_value(document.value().clone())
                    .map_err(|_| deny("data absent"))?,
                _ => return Err(deny("unsupported arithmetic observation format")),
            }
        } else {
            // Legacy fixture records embedded data directly. Never invent
            // timing evidence for them, but retain their original format.
            if execution.get("data_format").is_some() {
                return Err(deny("observation artifact missing"));
            }
            serde_json::from_value(execution["data"].clone()).map_err(|_| deny("data absent"))?
        };
        for observation in &data.observations {
            let contrast = plan
                .protocol()
                .contrasts
                .iter()
                .find(|value| value.id == observation.contrast)
                .ok_or_else(|| deny("contrast absent"))?;
            let head = c
                .portfolio
                .bindings
                .iter()
                .position(|binding| binding.id == contrast.binding)
                .ok_or_else(|| deny("contrast binding absent"))?;
            let input: Vec<f64> =
                serde_json::from_slice(&store.read_bounded(&observation.input, 128)?)
                    .map_err(|_| deny("input invalid"))?;
            if input.len() != 2 || head >= 5 {
                return Err(deny("arithmetic observation outside profile"));
            }
            let key = hash(&(&dispatch_id, &observation.item))?;
            let measured = native
                .get(&key)
                .ok_or_else(|| deny("observation lacks real native cell"))?;
            let arm = if observation.arm == Arm::Candidate {
                3
            } else {
                0
            };
            let expected = match contrast.metric.as_str() {
                "accuracy" => {
                    if (measured.outputs[arm][head] - truth(&input)[head]).abs() <= 1e-10 {
                        1.0
                    } else {
                        0.0
                    }
                }
                "latency" => measured
                    .latencies
                    .ok_or_else(|| deny("native latency evidence missing"))?[arm],
                "memory" => measured
                    .process_rss_bytes
                    .ok_or_else(|| deny("native process RSS evidence missing"))?
                    as f64,
                _ => return Err(deny("unsupported arithmetic observation metric")),
            };
            if observation.value != Some(expected) {
                return Err(deny("observation differs from native evidence"));
            }
        }
        let batches = execution["batches"]
            .as_array()
            .ok_or_else(|| deny("batches absent"))?;
        let mut count = 0;
        for (index, batch) in batches.iter().enumerate() {
            let (_, started) = read(
                &self.connection,
                contract,
                &format!("arithmetic_scorer_intent_{index}"),
            )?
            .ok_or_else(|| deny("scorer intent missing"))?;
            let (id, receipt) = read(
                &self.connection,
                contract,
                &format!("arithmetic_scorer_receipt_{index}"),
            )?
            .ok_or_else(|| deny("scorer receipt missing"))?;
            if batch["scorer"] != json!(id)
                || receipt["source"] != json!(ContentDigest::of_bytes(SCORER))
                || receipt["input"] != started["input"]
                || receipt["status"] != "completed"
                || receipt["cleanup_confirmed"] != true
                || started["snapshot"] != intent["snapshot"]
            {
                return Err(deny("scorer native chain differs"));
            }
            count += batch["items"]
                .as_array()
                .ok_or_else(|| deny("scorer items missing"))?
                .len();
        }
        let required_observations = c
            .portfolio
            .bindings
            .iter()
            .map(|binding| binding.metrics.len())
            .sum::<usize>()
            * 2;
        if count != native.len() || data.observations.len() != count * required_observations {
            return Err(deny("incomplete/extra native coverage"));
        }
        let analysis = if let Some(bytes) = compact_bytes {
            plan.analyze_compact(&bytes, &CancellationToken::default())?
        } else {
            plan.analyze(&canonical(&data)?, &CancellationToken::default())?
        };
        if canonical(&execution["analysis"])? != canonical(analysis.report())?
            || report.analysis.as_ref() != Some(analysis.id())
        {
            let different: Vec<_> = analysis
                .report()
                .as_object()
                .into_iter()
                .flatten()
                .filter(|(k, v)| execution["analysis"][*k] != **v)
                .map(|(k, _)| k.as_str())
                .collect();
            return Err(deny(&format!("reopened arithmetic analysis differs: fields={different:?}; retained={:?}; derived={}",report.analysis,analysis.id())));
        }
        let results = analysis.report()["results"]
            .as_array()
            .ok_or_else(|| deny("analysis results absent"))?;
        let mut expected = Vec::new();
        for name in [
            "frozen_panel",
            "holdout_opening",
            "prefrozen_analysis_plan",
            "analysis_binding",
        ] {
            expected.push(row(name, Status::Passed, ""));
        }
        for b in &c.portfolio.bindings {
            let items = splits::final_items(&self.connection, contract, &b.id)?;
            let tasks: Vec<_> = items
                .iter()
                .map(|item| statistics::Task {
                    item: item.identity.item.clone(),
                    group: item.identity.group.clone(),
                    input: item.input.clone(),
                })
                .collect();
            for metric in &b.metrics {
                let contrast = plan
                    .protocol()
                    .contrasts
                    .iter()
                    .find(|value| value.binding == b.id && value.metric == metric.metric)
                    .ok_or_else(|| deny("frozen contrast absent"))?;
                if canonical(&contrast.threshold)? != canonical(&metric.threshold)?
                    || contrast.required != metric.required
                    || contrast.value_statistic != metric.statistic
                    || contrast.comparator != metric.comparator
                    || contrast.no_op_control
                    || contrast.candidate != model_identity(&finalist)?
                    || contrast.baseline != model_identity(&c.source)?
                    || contrast.context != hash(&c.runtime)?
                    || contrast.workload != b.splits.holdout.digest()?
                    || canonical(&contrast.tasks)? != canonical(&tasks)?
                    || canonical(&contrast.schema)?
                        != canonical(&c.portfolio.gyms[&b.gym].metrics[&metric.metric])?
                    || contrast.measurement_protocol != hash(&metric.measurement)?
                {
                    return Err(deny("contrast bypassed frozen gate"));
                }
                let result = results
                    .iter()
                    .find(|r| r["contrast"] == json!(contrast.id))
                    .ok_or_else(|| deny("metric result absent"))?;
                let status = match result["status"].as_str() {
                    Some("passed") => Status::Passed,
                    Some("failed" | "no_op_cannot_satisfy_positive_gain") => Status::Failed,
                    Some("inconclusive") => Status::Inconclusive,
                    _ => Status::Incomplete,
                };
                let mut check = row("frozen_metric", status, "");
                check.candidate = Some(model_identity(&finalist)?);
                check.binding = Some(b.id.clone());
                check.metric = Some(metric.metric.clone());
                check.required = metric.required;
                expected.push(check);
            }
        }
        expected.push(row("contrast_coverage", Status::Passed, ""));
        let panel_status = if !acceptance::has_resource_ceilings(&c) {
            Status::Missing
        } else if expected
            .iter()
            .any(|check| check.name == "frozen_metric" && check.status == Status::Incomplete)
        {
            Status::Incomplete
        } else {
            Status::Passed
        };
        expected.push(row("full_analysis_panel", panel_status, ""));
        let analysis_passed = expected
            .iter()
            .all(|check| !check.required || check.status == Status::Passed);
        let metrics_passed = expected
            .iter()
            .filter(|check| check.name == "frozen_metric")
            .all(|check| !check.required || check.status == Status::Passed);
        if report.analysis_conditions_satisfied != analysis_passed
            || analysis.report()["all_required_analysis_conditions_satisfied"]
                != json!(metrics_passed)
        {
            return Err(deny("arithmetic analysis gate aggregate differs"));
        }
        let mut artifact = row(
            "candidate_artifact_resolution",
            Status::Passed,
            execution_id.to_string(),
        );
        artifact.candidate = Some(model_identity(&finalist)?);
        expected.push(artifact);
        for name in [
            "final_execution",
            "source_noop_best_known_controls",
            "resource_measurements",
            "method_and_edit_integrity",
        ] {
            let status =
                if name == "resource_measurements" && !acceptance::has_resource_ceilings(&c) {
                    Status::Missing
                } else {
                    Status::Passed
                };
            expected.push(row(name, status, execution_id.to_string()));
        }
        acceptance::verify_check_inventory(&report.checks, &expected)?;
        let status = [
            Status::Failed,
            Status::Missing,
            Status::Incomplete,
            Status::Inconclusive,
        ]
        .into_iter()
        .find(|s| {
            expected
                .iter()
                .any(|check| check.required && check.status == *s)
        })
        .unwrap_or(Status::Passed);
        if report.accepted != (status == Status::Passed) || report.status != status {
            return Err(deny("arithmetic gate aggregate differs"));
        }
        Ok(ArithmeticQualification {
            decision: Decision::from_report(report)?,
            export,
        })
    }
    /// Reconcile, never redispatch. Unknown native cleanup retains all holds.
    pub fn reconcile_arithmetic_final(
        &mut self,
        lease: &Lease,
        runtime: &sandbox::QualifiedDocker,
    ) -> Result<()> {
        let tx = self.connection.unchecked_transaction().map_err(sql)?;
        check_lease(&tx, &self.store, lease, &files::moment()?)?;
        drop(tx);
        let Some((intent_id, intent)) = read(&self.connection, &lease.contract, "intent")? else {
            return Ok(());
        };
        if intent["profile"] != json!(profile_id()) {
            return Err(deny("not this arithmetic recovery profile"));
        }
        let id: ContentDigest = serde_json::from_value(intent["reservation"].clone())
            .map_err(|_| deny("reservation absent"))?;
        if budgets::native_released(&self.connection, &id)? {
            return Ok(());
        }
        let owner: native_process::Owner = serde_json::from_value(intent["native_process"].clone())
            .map_err(|_| deny("owner absent"))?;
        if !owner.gone()? {
            return Err(deny("arithmetic controller still live; keep allocation"));
        }
        let root = serde_json::from_value(intent["sandbox_root"].clone())
            .map_err(|_| deny("sandbox root absent"))?;
        let _cleanup = runtime.recover_attested(&root)?;
        if !supervised::cleanup_known(&self.connection, &lease.contract)? {
            return Err(deny("arithmetic guardian cleanup unknown; keep allocation"));
        }
        let r = self.recover_reservation(lease, &id)?;
        self.settle(
            lease,
            &r,
            Settlement {
                operation: hash(&(&intent_id, "reconciled"))?,
                outcome: OutcomeKind::Incomplete,
                usage: Usage {
                    attempts: Some(1),
                    ..Default::default()
                },
                resources_released: true,
                evidence: intent_id,
            },
        )
    }
}
