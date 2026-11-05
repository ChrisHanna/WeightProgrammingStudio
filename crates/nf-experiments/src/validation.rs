//! Finite controller-owned gym probes. Native checks decide receipts; untrusted
//! scorers never attest their own success or mint approval/execution grants.
use crate::intake::schema::Schema;
use crate::intake::{self, Interface, Snapshot};
use crate::sandbox::{self, Limits, QualifiedDocker};
use nf_checkpoint::store::ArtifactStore;
use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits as JsonLimits};
use nf_core::error::{CoreError, ErrorCode, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::AtomicBool;

pub mod reference;

// Validation and reference audits share the same non-importing supervisor.
const WORKER: &[u8] = include_bytes!("validation/reference_worker.py");
fn error(code: ErrorCode, message: &str) -> CoreError {
    CoreError::new(code, "/gym_validation", message)
}
fn document(value: &impl Serialize) -> Result<CanonicalDocument> {
    CanonicalDocument::parse(
        &serde_json::to_vec(value)
            .map_err(|_| error(ErrorCode::InvalidJson, "record encoding failed"))?,
        JsonLimits {
            max_bytes: 1048576,
            max_depth: 48,
            max_nodes: 20000,
        },
    )
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Termination {
    Terminated,
    Truncated,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    pub id: String,
    pub input: Value,
    pub expected: Value,
    pub wrong: Vec<Value>,
    #[serde(default)]
    pub actions: Vec<Value>,
    pub termination: Option<Termination>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScoreRule {
    pub minimum: f64,
    pub maximum: f64,
    pub good_min: f64,
    pub bad_max: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub schema_version: u32,
    pub snapshot: ContentDigest,
    pub truth_provenance: String,
    pub cases: Vec<Case>,
    pub seeds: Vec<u64>,
    pub scoring: ScoreRule,
    pub limits: Limits,
}
fn output_schema(snapshot: &Snapshot) -> &Schema {
    match &snapshot.manifest().draft.interface {
        Interface::SingleShot { output, .. } => output,
        Interface::Episode { result, .. } => result,
    }
}
impl Plan {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let doc = CanonicalDocument::parse(
            bytes,
            JsonLimits {
                max_bytes: 65536,
                max_depth: 32,
                max_nodes: 8000,
            },
        )?;
        serde_json::from_value(doc.value().clone())
            .map_err(|e| error(ErrorCode::InvalidValue, &e.to_string()))
    }
    fn validate(&self, snapshot: &Snapshot) -> Result<()> {
        if self.schema_version != 1 || self.snapshot != *snapshot.id() {
            return Err(error(
                ErrorCode::SourceMismatch,
                "plan requires exact snapshot version1",
            ));
        }
        if self.truth_provenance.is_empty()
            || self.truth_provenance.len() > 256
            || self.truth_provenance.chars().any(char::is_control)
        {
            return Err(error(
                ErrorCode::InvalidValue,
                "bounded independent controller-fixture truth reference required",
            ));
        }
        let r = &self.scoring;
        if [r.minimum, r.maximum, r.good_min, r.bad_max]
            .iter()
            .any(|v| !v.is_finite())
            || !(r.minimum <= r.bad_max && r.bad_max < r.good_min && r.good_min <= r.maximum)
        {
            return Err(error(
                ErrorCode::InvalidValue,
                "finite uncompensated positive/negative score gates required",
            ));
        }
        if !(2..=8).contains(&self.cases.len())
            || !(2..=4).contains(&self.seeds.len())
            || self.cases.len() * self.seeds.len() * 2 > 32
            || self.seeds.iter().collect::<BTreeSet<_>>().len() != self.seeds.len()
        {
            return Err(error(
                ErrorCode::LimitExceeded,
                "distinct seeds and bounded repeated independent fixture cases required",
            ));
        }
        if !(100..=5000).contains(&self.limits.wall_ms)
            || !(100..=5000).contains(&self.limits.cpu_ms)
            || !(1024..=65536).contains(&self.limits.output_bytes)
        {
            return Err(error(
                ErrorCode::LimitExceeded,
                "finite sandbox-compatible validation limits required",
            ));
        }
        let mut ids = BTreeSet::new();
        let mut inputs = BTreeSet::new();
        let mut truths = BTreeSet::new();
        for case in &self.cases {
            if case.id.is_empty()
                || case.id.len() > 128
                || !ids.insert(&case.id)
                || !inputs.insert(document(&case.input)?.digest())
            {
                return Err(error(
                    ErrorCode::InvalidValue,
                    "distinct bounded fixture ids/inputs required",
                ));
            }
            output_schema(snapshot).check(&case.expected)?;
            truths.insert(document(&case.expected)?.digest());
            if case.wrong.is_empty() || case.wrong.len() > 4 {
                return Err(error(
                    ErrorCode::InvalidValue,
                    "explicit known-wrong outputs required per case",
                ));
            }
            for wrong in &case.wrong {
                output_schema(snapshot).check(wrong)?;
                if wrong == &case.expected {
                    return Err(error(
                        ErrorCode::InvalidValue,
                        "wrong output cannot equal independent truth",
                    ));
                }
            }
            match &snapshot.manifest().draft.interface {
                Interface::SingleShot { input, .. } => {
                    input.check(&case.input)?;
                    if !case.actions.is_empty() || case.termination.is_some() {
                        return Err(error(
                            ErrorCode::InvalidValue,
                            "single-shot case cannot request episode actions",
                        ));
                    }
                }
                Interface::Episode {
                    reset,
                    action,
                    max_steps,
                    max_episode_ms,
                    ..
                } => {
                    reset.check(&case.input)?;
                    if self.limits.wall_ms > *max_episode_ms {
                        return Err(error(
                            ErrorCode::VerificationPolicyWeakened,
                            "validation cannot relax the frozen episode wall limit",
                        ));
                    }
                    if case.actions.is_empty()
                        || case.actions.len() > 32
                        || case.actions.len() > *max_steps as usize
                        || case.termination.is_none()
                    {
                        return Err(error(
                            ErrorCode::InvalidValue,
                            "bounded actions and expected termination required",
                        ));
                    }
                    for value in &case.actions {
                        action.check(value)?;
                    }
                }
            }
        }
        if truths.len() < 2 {
            return Err(error(
                ErrorCode::InvalidValue,
                "distinct independent truths required to probe constant oracles",
            ));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Passed,
    ProbePassed,
    Rejected,
    Inconclusive,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Report {
    pub schema_version: u32,
    pub snapshot: ContentDigest,
    pub plan: ContentDigest,
    pub protocol: ContentDigest,
    pub image: ContentDigest,
    pub runtime_profile: ContentDigest,
    pub runtime_environment: ContentDigest,
    pub sandbox_receipt: ContentDigest,
    pub observations: Option<ContentDigest>,
    pub status: Status,
    pub checks: BTreeMap<String, bool>,
    pub measurement_scope: String,
    pub cpu_ms: Option<u64>,
    pub wall_ms: Option<u64>,
    pub memory_peak_bytes: Option<u64>,
    pub brokered_model_calls: u64,
    pub scope: String,
}
/// Only actual validation execution mints this witness. DTOs cannot restore it.
pub struct Validation {
    report: Report,
    id: ContentDigest,
}
impl Validation {
    pub fn report(&self) -> &Report {
        &self.report
    }
    pub fn id(&self) -> &ContentDigest {
        &self.id
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApprovalRecord {
    pub schema_version: u32,
    pub snapshot: ContentDigest,
    pub validation: ContentDigest,
    pub plan: ContentDigest,
    pub runtime_profile: ContentDigest,
    pub runtime_environment: ContentDigest,
    pub protocol: ContentDigest,
    pub approval_reference: String,
    pub scope: String,
}
pub struct Approval {
    record: ApprovalRecord,
    id: ContentDigest,
}
impl Approval {
    pub fn record(&self) -> &ApprovalRecord {
        &self.record
    }
    pub fn id(&self) -> &ContentDigest {
        &self.id
    }
    pub fn check_environment(&self, snapshot: &Snapshot, runtime: &QualifiedDocker) -> Result<()> {
        if self.record.snapshot != *snapshot.id()
            || self.record.runtime_environment != *runtime.environment_identity()
            || self.record.runtime_profile != QualifiedDocker::profile_identity()
            || self.record.protocol != protocol_identity()
        {
            return Err(error(
                ErrorCode::PermissionDenied,
                "approval does not cover this snapshot/runtime/protocol",
            ));
        }
        Ok(())
    }
}
pub(crate) fn restore_approval(
    store: &ArtifactStore,
    id: &ContentDigest,
    snapshot: &Snapshot,
    runtime: &QualifiedDocker,
) -> Result<Approval> {
    let raw = store.read_bounded(id, 65536)?;
    let doc = CanonicalDocument::parse(
        &raw,
        JsonLimits {
            max_bytes: 65536,
            max_depth: 16,
            max_nodes: 2048,
        },
    )?;
    if doc.bytes() != raw || doc.digest() != *id {
        return Err(error(
            ErrorCode::IntegrityMismatch,
            "stored approval record digest mismatch",
        ));
    }
    let record: ApprovalRecord = serde_json::from_value(doc.value().clone())
        .map_err(|_| error(ErrorCode::InvalidValue, "invalid stored approval schema"))?;
    if record.schema_version != 1
        || record.approval_reference.is_empty()
        || record.approval_reference.len() > 256
        || record.approval_reference.chars().any(char::is_control)
        || record.scope != "exact_snapshot_only;contract_permissions_and_budgets_still_required"
    {
        return Err(error(
            ErrorCode::PermissionDenied,
            "stored approval record fields invalid",
        ));
    }
    let plan = Plan::parse(&store.read_bounded(&record.plan, 65536)?)?;
    plan.validate(snapshot)?;
    let report_raw = store.read_bounded(&record.validation, 1048576)?;
    let report_doc = CanonicalDocument::parse(
        &report_raw,
        JsonLimits {
            max_bytes: 1048576,
            max_depth: 48,
            max_nodes: 20000,
        },
    )?;
    if report_doc.bytes() != report_raw || report_doc.digest() != record.validation {
        return Err(error(
            ErrorCode::IntegrityMismatch,
            "stored validation report digest mismatch",
        ));
    }
    let report: Report = serde_json::from_value(report_doc.value().clone())
        .map_err(|_| error(ErrorCode::InvalidValue, "invalid stored validation report"))?;
    store.read_bounded(&report.sandbox_receipt, 1048576)?;
    let obs_id = report.observations.as_ref().ok_or_else(|| {
        error(
            ErrorCode::IntegrityMismatch,
            "stored approval lacks observations",
        )
    })?;
    let obs_raw = store.read_bounded(obs_id, 65536)?;
    let obs_doc = CanonicalDocument::parse(
        &obs_raw,
        JsonLimits {
            max_bytes: 65536,
            max_depth: 32,
            max_nodes: 10000,
        },
    )?;
    if report.status != Status::Passed
        || report.snapshot != *snapshot.id()
        || report.plan != record.plan
        || !inspect(obs_doc.value(), &plan, snapshot)
            .values()
            .all(|v| *v)
    {
        return Err(error(
            ErrorCode::PermissionDenied,
            "stored validation observations do not pass",
        ));
    }
    let approval = Approval {
        record,
        id: id.clone(),
    };
    approval.check_environment(snapshot, runtime)?;
    Ok(approval)
}
pub fn protocol_identity() -> ContentDigest {
    ContentDigest::of_bytes(
        &serde_json::to_vec(&(
            ContentDigest::of_bytes(WORKER),
            ContentDigest::of_bytes(include_bytes!("validation.rs")),
            ContentDigest::of_bytes(include_bytes!("intake/schema.rs")),
        ))
        .unwrap(),
    )
}
fn value(call: &Value) -> Option<&Value> {
    if call["ok"] == true {
        call.get("value")
    } else {
        None
    }
}
fn score(call: &Value, rule: &ScoreRule, good: bool) -> bool {
    value(call).and_then(Value::as_f64).is_some_and(|v| {
        v.is_finite()
            && v >= rule.minimum
            && v <= rule.maximum
            && if good {
                v >= rule.good_min
            } else {
                v <= rule.bad_max
            }
    })
}
fn episode_check(sample: &Value, case: &Case, snapshot: &Snapshot) -> bool {
    let Interface::Episode {
        observation,
        max_steps,
        ..
    } = &snapshot.manifest().draft.interface
    else {
        return true;
    };
    let episode = &sample["episode"];
    if !value(&episode["reset"]).is_some_and(|v| observation.check(v).is_ok())
        || value(&episode["close"]) != Some(&Value::Null)
    {
        return false;
    }
    let Some(steps) = episode["steps"].as_array() else {
        return false;
    };
    if steps.is_empty() || steps.len() > case.actions.len() || steps.len() > *max_steps as usize {
        return false;
    }
    for (i, step) in steps.iter().enumerate() {
        let Some(v) = value(step) else { return false };
        if !v.as_object().is_some_and(|m| {
            m.len() == 4
                && ["observation", "reward", "terminated", "truncated"]
                    .iter()
                    .all(|k| m.contains_key(*k))
        }) {
            return false;
        }
        if observation.check(&v["observation"]).is_err()
            || !v["reward"]
                .as_f64()
                .is_some_and(|n| n.is_finite() && n.abs() <= 1e12)
        {
            return false;
        }
        let (Some(terminated), Some(truncated)) =
            (v["terminated"].as_bool(), v["truncated"].as_bool())
        else {
            return false;
        };
        if i + 1 == steps.len() {
            if !(match case.termination {
                Some(Termination::Terminated) => terminated && !truncated,
                Some(Termination::Truncated) => truncated && !terminated,
                None => false,
            }) {
                return false;
            }
        } else if terminated || truncated {
            return false;
        }
    }
    true
}
fn inspect(observed: &Value, plan: &Plan, snapshot: &Snapshot) -> BTreeMap<String, bool> {
    let mut checks: BTreeMap<String, bool> = [
        "observation_schema",
        "known_good",
        "oracle_agreement",
        "good_score",
        "wrong_score",
        "invalid_score",
        "seed_replay",
        "episode_lifecycle",
    ]
    .into_iter()
    .map(|name| (name.into(), true))
    .collect();
    let Some(samples) = observed["samples"].as_array() else {
        checks.insert("observation_schema".into(), false);
        return checks;
    };
    if observed["schema_version"] != 1
        || !observed.as_object().is_some_and(|m| m.len() == 2)
        || samples.len() != plan.cases.len() * plan.seeds.len() * 2
    {
        checks.insert("observation_schema".into(), false);
        return checks;
    }
    let mut index = 0;
    for (ci, case) in plan.cases.iter().enumerate() {
        for seed in &plan.seeds {
            let mut first = None;
            for repeat in 0..2 {
                let sample = &samples[index];
                index += 1;
                if sample["case"].as_u64() != Some(ci as u64)
                    || sample["seed"].as_u64() != Some(*seed)
                    || sample["repeat"].as_u64() != Some(repeat)
                {
                    checks.insert("observation_schema".into(), false);
                }
                let evaluation = value(&sample["evaluation"]);
                if !evaluation.is_some_and(|v| output_schema(snapshot).check(v).is_ok()) {
                    checks.insert("observation_schema".into(), false);
                }
                if evaluation != Some(&case.expected) {
                    checks.insert("known_good".into(), false);
                }
                if value(&sample["oracle"]) != Some(&case.expected) {
                    checks.insert("oracle_agreement".into(), false);
                }
                if !score(&sample["good_score"], &plan.scoring, true)
                    || !score(&sample["observed_score"], &plan.scoring, true)
                {
                    checks.insert("good_score".into(), false);
                }
                if !sample["wrong_scores"].as_array().is_some_and(|scores| {
                    scores.len() == case.wrong.len()
                        && scores.iter().all(|s| score(s, &plan.scoring, false))
                }) {
                    checks.insert("wrong_score".into(), false);
                }
                let invalid = &sample["invalid_score"];
                if !(invalid["ok"] == false || score(invalid, &plan.scoring, false)) {
                    checks.insert("invalid_score".into(), false);
                }
                if !episode_check(sample, case, snapshot) {
                    checks.insert("episode_lifecycle".into(), false);
                }
                let mut normalized = sample.clone();
                if let Some(m) = normalized.as_object_mut() {
                    m.remove("repeat");
                }
                if let Some(first) = &first {
                    if first != &normalized {
                        checks.insert("seed_replay".into(), false);
                    }
                } else {
                    first = Some(normalized);
                }
            }
        }
    }
    checks
}
/// The trusted controller supplies fixture truths independently of the gym.
/// Those labels are validation-domain exposure, never a proposer holdout grant.
pub fn validate(
    snapshot: &Snapshot,
    store: &ArtifactStore,
    plan: Plan,
    sandbox: &QualifiedDocker,
) -> Result<Validation> {
    plan.validate(snapshot)?;
    let manifest = snapshot.manifest();
    let image_id: ContentDigest = sandbox::IMAGE.split('@').nth(1).unwrap().parse()?;
    if manifest.dependency.image != image_id || !manifest.dependency.packages.is_empty() {
        return Err(error(
            ErrorCode::UnsupportedCapability,
            "validation supports only the qualified immutable stdlib CPU image",
        ));
    }
    let plan_doc = document(&plan)?;
    let plan_id = store.put(plan_doc.bytes())?.digest;
    let request =
        json!({"interface":manifest.draft.interface,"cases":plan.cases,"seeds":plan.seeds});
    let request = document(&request)?;
    if request.bytes().len() > 65536 {
        return Err(error(
            ErrorCode::LimitExceeded,
            "validation request exceeds sandbox bound",
        ));
    }
    let mut assets = BTreeMap::new();
    for name in manifest.assets.keys() {
        assets.insert(name.clone(), snapshot.asset(store, name)?);
    }
    let runtime = sandbox.execute_assets(
        WORKER,
        request.bytes(),
        &assets,
        plan.limits.clone(),
        &AtomicBool::new(false),
    )?;
    let sandbox_id = store.put(document(&runtime)?.bytes())?.digest;
    let raw = runtime.report.as_ref();
    let mut report = Report {
        schema_version: 1,
        snapshot: snapshot.id().clone(),
        plan: plan_id,
        protocol: protocol_identity(),
        image: image_id,
        runtime_profile: QualifiedDocker::profile_identity(),
        runtime_environment: sandbox.environment_identity().clone(),
        sandbox_receipt: sandbox_id,
        observations: None,
        status: Status::Inconclusive,
        checks: BTreeMap::new(),
        measurement_scope: "private_container_cgroup_v2;startup_overhead_unmeasured".into(),
        cpu_ms: raw.and_then(|v| v["cpu_ms"].as_u64()),
        wall_ms: raw.and_then(|v| v["wall_ms"].as_u64()),
        memory_peak_bytes: raw.and_then(|v| v["memory_peak_bytes"].as_u64()),
        brokered_model_calls: 0,
        scope: "finite_controller_supplied_fixture_probes_not_scientific_acceptance".into(),
    };
    let bounded_cost = report.cpu_ms.is_some_and(|n| n <= plan.limits.cpu_ms)
        && report.wall_ms.is_some_and(|n| n <= plan.limits.wall_ms)
        && report.memory_peak_bytes.is_some_and(|n| n <= 134217728);
    if runtime.status == "completed" && runtime.cleanup_confirmed && bounded_cost {
        if let Some(raw) = raw {
            if let Some(stdout) = raw["stdout_hex"].as_str() {
                let stdout = sandbox::decode_hex(stdout)?;
                match CanonicalDocument::parse(
                    &stdout,
                    JsonLimits {
                        max_bytes: 65536,
                        max_depth: 32,
                        max_nodes: 10000,
                    },
                ) {
                    Ok(observed) => {
                        report.observations = Some(store.put(observed.bytes())?.digest);
                        report.checks = inspect(observed.value(), &plan, snapshot);
                        report.status = if report.checks.values().all(|v| *v) {
                            Status::Passed
                        } else {
                            Status::Rejected
                        };
                    }
                    Err(_) => {
                        report.status = Status::Rejected;
                        report.checks.insert("observation_schema".into(), false);
                    }
                }
            }
        }
    }
    let receipt = store.put(document(&report)?.bytes())?;
    Ok(Validation {
        report,
        id: receipt.digest,
    })
}
/// Approval is a separate explicit controller action on a LIVE native witness.
/// Persisted DTOs and untrusted success JSON cannot reconstruct this authority.
pub fn approve(
    snapshot: &Snapshot,
    validation: &Validation,
    store: &ArtifactStore,
    reference: String,
) -> Result<Approval> {
    let report = &validation.report;
    if reference.is_empty() || reference.len() > 256 || reference.chars().any(char::is_control) {
        return Err(error(
            ErrorCode::InvalidValue,
            "explicit bounded approval reference required",
        ));
    }
    if report.status != Status::Passed
        || report.snapshot != *snapshot.id()
        || report.runtime_profile != QualifiedDocker::profile_identity()
        || report.protocol != protocol_identity()
    {
        return Err(error(
            ErrorCode::PermissionDenied,
            "rejected, inconclusive or stale validation cannot approve this exact version",
        ));
    }
    intake::restore(store, snapshot.id())?;
    for digest in [&validation.id, &report.plan, &report.sandbox_receipt] {
        store.read_bounded(digest, 1048576)?;
    }
    if let Some(id) = &report.observations {
        store.read_bounded(id, 65536)?;
    } else {
        return Err(error(
            ErrorCode::IntegrityMismatch,
            "approval requires actual observations",
        ));
    }
    let record = ApprovalRecord {
        schema_version: 1,
        snapshot: snapshot.id().clone(),
        validation: validation.id.clone(),
        plan: report.plan.clone(),
        runtime_profile: report.runtime_profile.clone(),
        runtime_environment: report.runtime_environment.clone(),
        protocol: report.protocol.clone(),
        approval_reference: reference,
        scope: "exact_snapshot_only;contract_permissions_and_budgets_still_required".into(),
    };
    let id = store.put(document(&record)?.bytes())?.digest;
    Ok(Approval { record, id })
}
