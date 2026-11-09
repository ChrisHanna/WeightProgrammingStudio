//! Bounded reference-policy execution over ONE unchanged gym snapshot.
//! Controller-selected policy snapshots are not discovered or downloaded. This
//! audit exposes development/selection inputs only and cannot mint Approval.
use super::*;

#[path = "reference_episode.rs"]
mod episode;
#[path = "reference_storage.rs"]
mod storage;

const SHIM: &[u8] = include_bytes!("reference_worker.py");

/// Controller declarations are coverage targets, not authenticated provenance.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Coverage {
    pub required: BTreeSet<String>,
    pub case_groups: BTreeMap<String, String>,
}
impl Coverage {
    fn complete(&self, plan: &Plan) -> Result<bool> {
        if self.required.len() > 32
            || self.case_groups.len() > 8
            || self
                .required
                .iter()
                .chain(self.case_groups.values())
                .any(|s| s.is_empty() || s.len() > 128 || s.chars().any(char::is_control))
            || self
                .case_groups
                .keys()
                .any(|id| !plan.cases.iter().any(|c| &c.id == id))
        {
            return Err(error(
                ErrorCode::InvalidValue,
                "invalid bounded coverage declaration",
            ));
        }
        Ok(!self.required.is_empty()
            && self.case_groups.len() == plan.cases.len()
            && self
                .required
                .iter()
                .all(|g| self.case_groups.values().any(|v| v == g)))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Valid,
    Invalid,
    Constant,
    MetadataOnly,
    Weak,
    Strong,
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Exposure {
    Development,
    Selection,
}
/// Explicit controller designation; the source itself remains sandboxed.
pub struct Policy<'a> {
    pub snapshot: &'a Snapshot,
    pub role: Role,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Budget {
    pub max_executions: u64,
    pub cpu_ms: u64,
    pub wall_ms: u64,
    /// Explicit controller-allocated allowance for this invocation. Not a lease
    /// or a new family budget: callers must debit their durable study allocation.
    pub storage_bytes: u64,
}
impl Budget {
    /// Run before sandbox qualification (which may recover/write old journals).
    pub fn validate_storage(&self) -> Result<()> {
        storage::Accounting::admit(self.max_executions, self.storage_bytes).map(|_| ())
    }
}
#[derive(Debug, Serialize)]
pub struct PolicyReport {
    pub snapshot: ContentDigest,
    pub role: Role,
    pub samples: usize,
    pub correct: usize,
    pub passing: usize,
}
#[derive(Debug, Default, Clone, Serialize)]
pub struct PreflightTimings {
    pub binding_decode_ns: u64,
    pub request_validation_ns: u64,
    pub snapshot_restore_ns: u64,
    pub coverage_parse_ns: u64,
    pub docker_qualification_ns: u64,
    pub before_audit_ns: u64,
    pub audit_preflight_ns: u64,
}
#[derive(Debug, Serialize)]
pub struct AuditReport {
    pub schema_version: u32,
    pub snapshot: ContentDigest,
    pub plan: ContentDigest,
    pub protocol: ContentDigest,
    pub runtime_environment: ContentDigest,
    pub exposure: Exposure,
    pub budget: Budget,
    pub status: Status,
    pub requested_policies: Vec<(ContentDigest, Role)>,
    pub policies: Vec<PolicyReport>,
    pub sandbox_receipts: Vec<ContentDigest>,
    pub observations: Vec<ContentDigest>,
    pub findings: BTreeSet<String>,
    pub executions: u64,
    pub cpu_ms: u64,
    pub wall_ms: u64,
    pub brokered_model_calls: u64,
    pub native_approval: bool,
    pub scope: String,
    pub audit_complete: bool,
    pub missing_prerequisites: BTreeSet<String>,
    pub coverage: Coverage,
    pub reserved_asset_bytes: u64,
    pub storage: storage::Accounting,
    pub controller_wall_ms: u64,
    pub preflight_timings: PreflightTimings,
    pub memory_peak_bytes: u64,
    /// Present only for the durable, qualified dedicated controller path.
    pub controller_hard_address_space_bytes: Option<u64>,
    pub measurement_scope: String,
    pub observation_boundary: String,
}
pub struct Audit {
    report: AuditReport,
    id: ContentDigest,
}
impl Audit {
    pub fn report(&self) -> &AuditReport {
        &self.report
    }
    pub fn id(&self) -> &ContentDigest {
        &self.id
    }
}
pub fn protocol_identity() -> ContentDigest {
    ContentDigest::of_bytes(
        &serde_json::to_vec(&(
            super::protocol_identity(),
            ContentDigest::of_bytes(SHIM),
            ContentDigest::of_bytes(include_bytes!("reference.rs")),
            ContentDigest::of_bytes(include_bytes!("reference_episode.rs")),
            ContentDigest::of_bytes(include_bytes!("reference_storage.rs")),
        ))
        .unwrap(),
    )
}
fn assets(snapshot: &Snapshot, store: &ArtifactStore) -> Result<BTreeMap<String, Vec<u8>>> {
    // Reverify CAS, including the manifest, before executing the frozen assets.
    intake::restore(store, snapshot.id())?;
    let image: ContentDigest = sandbox::IMAGE.split('@').nth(1).unwrap().parse()?;
    if snapshot.manifest().dependency.image != image
        || !snapshot.manifest().dependency.packages.is_empty()
    {
        return Err(error(
            ErrorCode::UnsupportedCapability,
            "reference audit requires pinned stdlib image",
        ));
    }
    snapshot
        .manifest()
        .assets
        .keys()
        .map(|name| Ok((name.clone(), snapshot.asset(store, name)?)))
        .collect()
}
/// Controller-only live grant; not deserializable or derived from guest output.
struct LiveAuthority<'a> {
    deadline: std::time::Instant,
    authorize: &'a dyn Fn() -> Result<()>,
}
impl LiveAuthority<'_> {
    fn check(&self) -> Result<()> {
        if std::time::Instant::now() >= self.deadline {
            return Err(error(
                ErrorCode::LimitExceeded,
                "absolute audit deadline expired",
            ));
        }
        (self.authorize)()?;
        if std::time::Instant::now() >= self.deadline {
            return Err(error(
                ErrorCode::LimitExceeded,
                "absolute audit deadline expired during authorization",
            ));
        }
        Ok(())
    }
    fn admit_job(&self, limits: &Limits) -> Result<()> {
        self.check()?;
        if self
            .deadline
            .saturating_duration_since(std::time::Instant::now())
            < std::time::Duration::from_millis(limits.wall_ms)
        {
            return Err(error(
                ErrorCode::LimitExceeded,
                "insufficient remaining audit wall allowance before staging",
            ));
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)] // Explicit execution boundary and shared cancellation.
fn execute(
    runtime: &QualifiedDocker,
    store: &ArtifactStore,
    assets: &BTreeMap<String, Vec<u8>>,
    request: Value,
    limits: &Limits,
    report: &mut AuditReport,
    fields: &[&str],
    count: usize,
    cancel: &AtomicBool,
    authorize: Option<&LiveAuthority<'_>>,
) -> Result<Option<Vec<Value>>> {
    if let Some(check) = authorize {
        check.admit_job(limits)?;
    }
    let request = document(&request)?;
    if request.bytes().len() > 65536 {
        return Err(error(
            ErrorCode::LimitExceeded,
            "reference request exceeds finite bound",
        ));
    }
    if cancel.load(std::sync::atomic::Ordering::Acquire) {
        report.status = Status::Inconclusive;
        report.findings.insert("audit_cancelled".into());
        return Ok(None);
    }
    // A non-refundable slice was admitted for EVERY possible job before the
    // first CAS publication or sandbox job. Claim before staging, including on
    // errors/cancellation; neither cleanup nor CAS deduplication refunds it.
    let staging_limit = report.storage.claim_job()?;
    let receipt = if let Some(live) = authorize {
        runtime.execute_assets_authorized(
            SHIM,
            request.bytes(),
            assets,
            limits.clone(),
            staging_limit,
            cancel,
            live.deadline,
            &|| live.check(),
        )?
    } else {
        runtime.execute_assets_with_staging_limit(
            SHIM,
            request.bytes(),
            assets,
            limits.clone(),
            staging_limit,
            cancel,
        )?
    };
    report.executions += 1;
    if let Some(check) = authorize {
        check.check()?;
    }
    report.storage.observe(&receipt);
    report
        .sandbox_receipts
        .push(report.storage.put(store, document(&receipt)?.bytes())?);
    let raw = receipt.report.as_ref();
    let cpu = raw.and_then(|r| r["cpu_ms"].as_u64());
    let wall = raw.and_then(|r| r["wall_ms"].as_u64());
    let memory = raw.and_then(|r| r["memory_peak_bytes"].as_u64());
    if let Some(n) = cpu {
        report.cpu_ms += n;
    }
    if let Some(n) = wall {
        report.wall_ms += n;
    }
    if let Some(n) = memory {
        report.memory_peak_bytes = report.memory_peak_bytes.max(n);
    }
    if cancel.load(std::sync::atomic::Ordering::Acquire) {
        report.findings.insert("audit_cancelled".into());
    }
    if receipt.status != "completed"
        || !receipt.cleanup_confirmed
        || !cpu.is_some_and(|n| n <= limits.cpu_ms)
        || !wall.is_some_and(|n| n <= limits.wall_ms)
        || !memory.is_some_and(|n| n <= 134217728)
    {
        report.status = Status::Inconclusive;
        report
            .findings
            .insert("bounded_execution_incomplete".into());
        return Ok(None);
    }
    let observed = (|| {
        let bytes = sandbox::decode_hex(raw?["stdout_hex"].as_str()?).ok()?;
        CanonicalDocument::parse(
            &bytes,
            JsonLimits {
                max_bytes: 65536,
                max_depth: 32,
                max_nodes: 10000,
            },
        )
        .ok()
    })();
    if let Some(doc) = observed {
        report
            .observations
            .push(report.storage.put(store, doc.bytes())?);
        let value = doc.value();
        if value["schema_version"] == 1 && value.as_object().is_some_and(|m| m.len() == 2) {
            if let Some(rows) = value["samples"].as_array() {
                if rows.len() == count
                    && rows.iter().enumerate().all(|(i, row)| {
                        row["index"].as_u64() == Some(i as u64)
                            && row.as_object().is_some_and(|m| m.len() == fields.len() + 1)
                            && fields.iter().all(|f| row.get(*f).is_some())
                    })
                {
                    return Ok(Some(rows.clone()));
                }
            }
        }
    }
    report.findings.insert("observation_schema".into());
    report.status = Status::Rejected;
    Ok(None)
}
/// No supplied observation DTO is accepted. `Plan` contains only independent
/// controller truth. Policy bytes and gym bytes come from verified snapshots.
/// Each policy runs separately from the scorer: it cannot read fixture truths,
/// overwrite the gym, or impersonate another policy through module globals.
/// No model/network/GPU calls are supported; all execution is validation-domain
/// exposure, not a holdout, research acceptance, or contract execution grant.
pub fn run(
    snapshot: &Snapshot,
    store: &ArtifactStore,
    plan: Plan,
    policies: &[Policy<'_>],
    exposure: Exposure,
    budget: Budget,
    runtime: &QualifiedDocker,
) -> Result<Audit> {
    run_cancellable(
        snapshot,
        store,
        plan,
        policies,
        exposure,
        budget,
        runtime,
        Coverage::default(),
        &AtomicBool::new(false),
    )
}

/// Full audits require six executed control roles and complete declared coverage.
/// Missing controls yield a narrow probe, never a full T042 success.
#[allow(clippy::too_many_arguments)]
pub fn run_cancellable(
    snapshot: &Snapshot,
    store: &ArtifactStore,
    plan: Plan,
    policies: &[Policy<'_>],
    exposure: Exposure,
    budget: Budget,
    runtime: &QualifiedDocker,
    coverage: Coverage,
    cancel: &AtomicBool,
) -> Result<Audit> {
    let mut preflight_timings = PreflightTimings::default();
    run_cancellable_with_preflight(
        snapshot,
        store,
        plan,
        policies,
        exposure,
        budget,
        runtime,
        coverage,
        cancel,
        &mut preflight_timings,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_cancellable_with_preflight(
    snapshot: &Snapshot,
    store: &ArtifactStore,
    plan: Plan,
    policies: &[Policy<'_>],
    exposure: Exposure,
    budget: Budget,
    runtime: &QualifiedDocker,
    coverage: Coverage,
    cancel: &AtomicBool,
    preflight_timings: &mut PreflightTimings,
) -> Result<Audit> {
    run_inner(
        snapshot,
        store,
        plan,
        policies,
        exposure,
        budget,
        runtime,
        coverage,
        cancel,
        None,
        None,
        preflight_timings,
    )
}

/// Compatibility entry point: live checks still run throughout staging/execution.
/// Durable callers should use `run_authorized_until` to admit against the remaining
/// reservation/lease deadline rather than just the audit's own wall allowance.
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn run_authorized(
    snapshot: &Snapshot,
    store: &ArtifactStore,
    plan: Plan,
    policies: &[Policy<'_>],
    exposure: Exposure,
    budget: Budget,
    runtime: &QualifiedDocker,
    coverage: Coverage,
    cancel: &AtomicBool,
    authorize: &dyn Fn() -> Result<()>,
) -> Result<Audit> {
    let mut preflight_timings = PreflightTimings::default();
    run_inner(
        snapshot,
        store,
        plan,
        policies,
        exposure,
        budget,
        runtime,
        coverage,
        cancel,
        Some(authorize),
        None,
        &mut preflight_timings,
    )
}

/// Crate-private: only the native coordinator can attach durable authority.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_authorized_until(
    snapshot: &Snapshot,
    store: &ArtifactStore,
    plan: Plan,
    policies: &[Policy<'_>],
    exposure: Exposure,
    budget: Budget,
    runtime: &QualifiedDocker,
    coverage: Coverage,
    cancel: &AtomicBool,
    authorize: &dyn Fn() -> Result<()>,
    deadline: std::time::Instant,
) -> Result<Audit> {
    let mut preflight_timings = PreflightTimings::default();
    run_inner(
        snapshot,
        store,
        plan,
        policies,
        exposure,
        budget,
        runtime,
        coverage,
        cancel,
        Some(authorize),
        Some(deadline),
        &mut preflight_timings,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_inner(
    snapshot: &Snapshot,
    store: &ArtifactStore,
    plan: Plan,
    policies: &[Policy<'_>],
    exposure: Exposure,
    budget: Budget,
    runtime: &QualifiedDocker,
    coverage: Coverage,
    cancel: &AtomicBool,
    authorize: Option<&dyn Fn() -> Result<()>>,
    deadline: Option<std::time::Instant>,
    preflight_timings: &mut PreflightTimings,
) -> Result<Audit> {
    let started = std::time::Instant::now();
    let durable = authorize.is_some();
    let audit_deadline = started
        .checked_add(std::time::Duration::from_millis(budget.wall_ms))
        .ok_or_else(|| error(ErrorCode::LimitExceeded, "audit deadline overflow"))?;
    // A pre-cancelled invocation is sticky and must never stage. Once running,
    // cancellation is separately polled by the sandbox and produces an
    // inconclusive report only after native teardown; it is not lease authority.
    if cancel.load(std::sync::atomic::Ordering::Acquire) {
        return Err(error(ErrorCode::LimitExceeded, "audit cancelled"));
    }
    let check = || {
        if let Some(check) = authorize {
            check()?;
        }
        Ok(())
    };
    let live = LiveAuthority {
        deadline: deadline.unwrap_or(audit_deadline).min(audit_deadline),
        authorize: &check,
    };
    let authorize = Some(&live);
    live.check()?;
    budget.validate_storage()?;
    plan.validate(snapshot)?;
    let covered = coverage.complete(&plan)?;
    let (scorer, oracle, rounds, execution_cap) = match &snapshot.manifest().draft.interface {
        Interface::SingleShot { score, oracle, .. } => (score, oracle, 2, 16),
        Interface::Episode {
            score,
            oracle,
            max_episode_ms,
            ..
        } => {
            // Prefix replay keeps policy and gym in distinct sandboxes. Reserve
            // every reset/replay, decision and scoring job, not just final runs.
            let rounds = 2 + 2 * plan
                .cases
                .iter()
                .map(|c| c.actions.len() as u64)
                .max()
                .unwrap();
            if rounds * plan.limits.wall_ms > *max_episode_ms {
                return Err(error(ErrorCode::VerificationPolicyWeakened,
                    "aggregate episode replay/decision/scoring wall reservation exceeds frozen episode limit"));
            }
            (score, oracle, rounds, 528)
        }
    };
    let jobs = policies.len() as u64 * rounds;
    if policies.is_empty()
        || policies.len() > 8
        || policies
            .iter()
            .map(|p| p.snapshot.id())
            .collect::<BTreeSet<_>>()
            .len()
            != policies.len()
        || jobs > budget.max_executions
        || budget.max_executions > execution_cap
        || jobs * plan.limits.cpu_ms > budget.cpu_ms
        || budget.cpu_ms > 80000
        || jobs * plan.limits.wall_ms > budget.wall_ms
        || budget.wall_ms > 80000
    {
        return Err(error(
            ErrorCode::LimitExceeded,
            "explicit aggregate reference execution budget exceeded",
        ));
    }
    // Include source/input/guardian/policy/directories and all journal/capacity
    // probe bytes via the full sandbox job ceiling, not just asset payloads.
    // Reserve all CAS writes before any publication as well.
    let mut storage = storage::Accounting::admit(budget.max_executions, budget.storage_bytes)?;
    if durable {
        storage.durable_family_accounting = true;
        storage.scope = "native_coordinator_single_use_reservation;full_reserved_storage_charged_unknown_on_settlement;retries_require_new_sequential_attempt;excludes_existing_CAS_SQLite_qualification;not_physical_capacity".into();
    }
    let gym_assets = assets(snapshot, store)?;
    // Preflight every policy before launching ANY container.
    let mut supplied = Vec::new();
    for policy in policies {
        let Interface::SingleShot {
            evaluate, input, ..
        } = &policy.snapshot.manifest().draft.interface
        else {
            return Err(error(
                ErrorCode::UnsupportedCapability,
                "policy requires single-shot evaluate(inputs, seed)",
            ));
        };
        if matches!(
            snapshot.manifest().draft.interface,
            Interface::SingleShot { .. }
        ) {
            for case in &plan.cases {
                input.check(&case.input)?;
            }
        }
        supplied.push((evaluate, input, assets(policy.snapshot, store)?));
    }
    // Retain the old payload-amplification diagnostic, but do not use it as
    // storage admission: it excludes source/input/guardian/journals and CAS.
    let gym_bytes: u64 = gym_assets.values().map(|b| b.len() as u64).sum();
    let reserved_asset_bytes = supplied
        .iter()
        .map(|(_, _, a)| rounds * (gym_bytes + a.values().map(|b| b.len() as u64).sum::<u64>()))
        .sum();
    let mut missing_prerequisites = BTreeSet::new();
    for role in [
        Role::Valid,
        Role::Invalid,
        Role::Constant,
        Role::MetadataOnly,
        Role::Weak,
        Role::Strong,
    ] {
        if !policies.iter().any(|p| p.role == role) {
            missing_prerequisites.insert(format!("missing_{role:?}_control").to_lowercase());
        }
    }
    if !covered {
        missing_prerequisites.insert("missing_required_coverage".into());
    }
    if let Some(check) = authorize {
        check.check()?;
    }
    let plan_id = storage.put(store, document(&plan)?.bytes())?;
    let mut report = AuditReport {
        schema_version: 1, snapshot: snapshot.id().clone(), plan: plan_id,
        protocol: protocol_identity(), runtime_environment: runtime.environment_identity().clone(),
        exposure, budget, status: Status::Passed,
        requested_policies: policies.iter().map(|p| (p.snapshot.id().clone(), p.role)).collect(),
        policies: vec![], sandbox_receipts: vec![],
        observations: vec![], findings: BTreeSet::new(), executions: 0, cpu_ms: 0, wall_ms: 0,
        brokered_model_calls: 0, native_approval: false,
        scope: "finite_exposed_reference_audit_not_validation_approval_or_scientific_acceptance".into(),
        audit_complete: false, missing_prerequisites, coverage, reserved_asset_bytes, storage,
        controller_wall_ms: 0, memory_peak_bytes: 0,
        preflight_timings: preflight_timings.clone(),
        controller_hard_address_space_bytes: if durable {
            rustix::process::getrlimit(rustix::process::Resource::As).maximum
        } else { None },
        measurement_scope: if durable {
            "controller_memory_including_baseline_and_qualification_hard_RLIMIT_AS;guest_memory_hard_cgroup;excludes_Docker_daemon_CLI_helpers_other_host_processes_and_operator_recovery;not_whole_process_tree;cpu_wall_memory_metrics_are_guest_cgroup_only;memory_metric_is_max_job_peak_not_aggregate;controller_wall_including_qualification_is_bounded_by_absolute_deadline_but_report_excludes_qualification;storage_is_conservative_ceiling_charged_not_physical_usage"
        } else {
            "cpu_wall_memory_are_container_cgroup_only;no_hard_controller_memory_guarantee;controller_wall_excludes_qualification;memory_is_max_job_peak_not_aggregate;storage_is_conservative_ceiling_charged_not_physical_usage"
        }.into(),
        observation_boundary: "non_dumpable_supervisor_dispatches_isolated_callbacks;values_are_untrusted;finite_probes_do_not_prove_malicious_scorer_semantics".into(),
    };
    let cells: Vec<Value> = plan
        .cases
        .iter()
        .flat_map(|case| {
            plan.seeds
                .iter()
                .flat_map(move |seed| (0..2).map(move |_| json!({"input":case.input,"seed":seed})))
        })
        .collect();
    preflight_timings.audit_preflight_ns = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
    report.preflight_timings = preflight_timings.clone();
    for (policy, (entry, input, policy_assets)) in policies.iter().zip(supplied) {
        let outputs = if matches!(
            snapshot.manifest().draft.interface,
            Interface::Episode { .. }
        ) {
            episode::run(
                snapshot,
                &plan,
                entry,
                input,
                policy.role,
                &policy_assets,
                &gym_assets,
                &cells,
                runtime,
                store,
                &mut report,
                cancel,
                authorize,
            )?
        } else {
            execute(
                runtime,
                store,
                &policy_assets,
                json!({"mode":"policy","entry":entry,"cells":cells}),
                &plan.limits,
                &mut report,
                &["output"],
                cells.len(),
                cancel,
                authorize,
            )?
        };
        let Some(outputs) = outputs else {
            break;
        };
        let mut scoring_cells = Vec::new();
        let mut summary = PolicyReport {
            snapshot: policy.snapshot.id().clone(),
            role: policy.role,
            samples: outputs.len(),
            correct: 0,
            passing: 0,
        };
        let mut distinct = BTreeSet::new();
        for (i, row) in outputs.iter().enumerate() {
            let case = &plan.cases[i / (plan.seeds.len() * 2)];
            let output = value(&row["output"]);
            if output.is_none() {
                report.findings.insert("policy_execution_failed".into());
            }
            if output == Some(&case.expected) {
                summary.correct += 1;
            }
            distinct.insert(document(&row["output"])?.digest());
            if i % 2 == 1 && row["output"] != outputs[i - 1]["output"] {
                report.findings.insert("policy_replay_drift".into());
            }
            scoring_cells.push(json!({"input":case.input,"expected":case.expected,"wrong":case.wrong,"output":output}));
        }
        let Some(scores) = execute(
            runtime,
            store,
            &gym_assets,
            json!({"mode":"gym","score":scorer,"oracle":oracle,"cells":scoring_cells}),
            &plan.limits,
            &mut report,
            &[
                "oracle",
                "score",
                "invalid_score",
                "good_score",
                "wrong_scores",
            ],
            cells.len(),
            cancel,
            authorize,
        )?
        else {
            break;
        };
        for (i, row) in scores.iter().enumerate() {
            let case = &plan.cases[i / (plan.seeds.len() * 2)];
            let output = value(&outputs[i]["output"]);
            let correct = output == Some(&case.expected);
            if output.is_some_and(|v| output_schema(snapshot).check(v).is_err())
                && policy.role != Role::Invalid
            {
                report.findings.insert("policy_output_schema".into());
            }
            if value(&row["oracle"]) != Some(&case.expected) {
                report.findings.insert("oracle_disagreement".into());
            }
            if !score(&row["score"], &plan.scoring, correct) {
                report.findings.insert("scorer_disagreement".into());
            }
            if !score(&row["good_score"], &plan.scoring, true) {
                report.findings.insert("known_good_score".into());
            }
            if !row["wrong_scores"].as_array().is_some_and(|scores| {
                scores.len() == case.wrong.len()
                    && scores.iter().all(|s| score(s, &plan.scoring, false))
            }) {
                report.findings.insert("known_wrong_score".into());
            }
            if score(&row["score"], &plan.scoring, true) {
                summary.passing += 1;
            }
            if !(row["invalid_score"]["ok"] == false
                || score(&row["invalid_score"], &plan.scoring, false))
            {
                report.findings.insert("invalid_output_rewarded".into());
            }
            if i % 2 == 1
                && [
                    "oracle",
                    "score",
                    "invalid_score",
                    "good_score",
                    "wrong_scores",
                ]
                .iter()
                .any(|k| row[*k] != scores[i - 1][*k])
            {
                report.findings.insert("gym_replay_drift".into());
            }
        }
        match policy.role {
            Role::Valid if summary.correct != summary.samples => {
                report.findings.insert("valid_policy_failed".into());
            }
            Role::Strong if summary.correct != summary.samples => {
                report.findings.insert("strong_policy_failed".into());
            }
            Role::Invalid if summary.correct > 0 || summary.passing > 0 => {
                report
                    .findings
                    .insert("invalid_policy_not_separated".into());
            }
            Role::Constant
                if matches!(
                    snapshot.manifest().draft.interface,
                    Interface::SingleShot { .. }
                ) && distinct.len() != 1 =>
            {
                report
                    .findings
                    .insert("constant_policy_role_mismatch".into());
            }
            Role::Weak if summary.correct == summary.samples => {
                report.findings.insert("weak_policy_saturation".into());
            }
            _ => {}
        }
        if matches!(policy.role, Role::Constant | Role::MetadataOnly) && summary.passing > 0 {
            report.findings.insert("shortcut_rewarded".into());
        }
        report.policies.push(summary);
    }
    if cancel.load(std::sync::atomic::Ordering::Acquire) {
        report.status = Status::Inconclusive;
        report.findings.insert("audit_cancelled".into());
    }
    if report.policies.len() != policies.len() {
        report
            .missing_prerequisites
            .insert("incomplete_policy_execution".into());
    }
    report.audit_complete = report.missing_prerequisites.is_empty();
    if report.status == Status::Passed {
        report.status = if !report.findings.is_empty() {
            Status::Rejected
        } else if report.audit_complete {
            Status::Passed
        } else {
            Status::ProbePassed
        };
    }
    report.controller_wall_ms = started.elapsed().as_millis().min(u64::MAX as u128) as u64;
    // Account for the report's own CAS publication before serializing it.
    if let Some(check) = authorize {
        check.check()?;
    }
    report.storage.claim_publication()?;
    let id = store.put(document(&report)?.bytes())?.digest;
    Ok(Audit { report, id })
}
