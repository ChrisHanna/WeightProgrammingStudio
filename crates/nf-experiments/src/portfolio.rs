//! Immutable, live-source-backed portfolio planning. Never a dispatch/promotion grant.
use crate::{arithmetic, intake, validation};
use nf_checkpoint::store::ArtifactStore;
use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use nf_core::error::{CoreError, ErrorCode, Result};
use nf_core::schemas::{FactoryRef, Validate, Validated};
use nf_core::study::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

fn error(text: &str) -> CoreError {
    CoreError::new(ErrorCode::InvalidValue, "/portfolio", text)
}
fn doc(value: &impl Serialize) -> Result<CanonicalDocument> {
    CanonicalDocument::parse(
        &serde_json::to_vec(value).map_err(|_| error("encoding failed"))?,
        Limits::default(),
    )
}
fn hash(value: &impl Serialize) -> Result<ContentDigest> {
    Ok(doc(value)?.digest())
}
fn factory(id: &str, config: ContentDigest) -> FactoryRef {
    FactoryRef {
        id: id.into(),
        version: "1".into(),
        config,
    }
}

/// Private capability: constructed from actual local probes or T028 approval, not DTOs.
pub struct GymSource {
    manifest: GymManifest,
    id: ContentDigest,
    evidence: ContentDigest,
    dimensions: BTreeSet<Dimension>,
    validation_state: String,
    validation_bytes: Vec<u8>,
}
impl GymSource {
    pub fn id(&self) -> &ContentDigest {
        &self.id
    }
    pub fn manifest(&self) -> &GymManifest {
        &self.manifest
    }
    pub fn validation_bytes(&self) -> &[u8] {
        &self.validation_bytes
    }
    pub fn builtin(head: &str) -> Result<Self> {
        let index = arithmetic::HEADS
            .iter()
            .position(|v| *v == head)
            .ok_or_else(|| error("unknown registered Foundry gym"))?;
        let fixture = arithmetic::build(arithmetic::Mode::Reference)?;
        let descriptor = &fixture.description["gyms"][index];
        let mut observations = Vec::new();
        for case in &fixture.inventory.cases {
            let a = case.x1 as f64;
            let b = case.x2 as f64;
            let truth = match index {
                0 => a + b,
                1 => a - b,
                2 => a.max(b),
                3 => a.min(b),
                _ => (a - b).abs(),
            };
            let output = fixture
                .source
                .infer_arithmetic_fixture(&[a, b], Default::default())?[index];
            if !output.is_finite() || (output - truth).abs() > 1e-10 {
                return Err(error(
                    "actual builtin reference failed independent arithmetic probe",
                ));
            }
            // Deliberately wrong and invalid/empty policies cannot earn accuracy.
            let score = |value: Option<f64>| {
                value.is_some_and(|v| v.is_finite() && (v - truth).abs() <= 1e-10)
            };
            if !score(Some(output))
                || score(Some(truth + 1.0))
                || score(None)
                || score(Some(f64::NAN))
            {
                return Err(error("builtin positive/negative scorer probes failed"));
            }
            observations.push((case.item.clone(), output, truth));
        }
        let validation_doc = doc(
            &json!({"protocol":"nf.portfolio.arithmetic.local_probe.v1","probe_code":ContentDigest::of_bytes(include_bytes!("portfolio.rs")),"descriptor":descriptor,"observations":observations,"checks":["actual_forward","independent_truth","wrong","empty","nonfinite"],"scope":"synthetic_fixture_only_not_general_scorer_proof"}),
        )?;
        let evidence = validation_doc.digest();
        let descriptor_id: ContentDigest = serde_json::from_value(descriptor["id"].clone())
            .map_err(|_| error("builtin registry identity absent"))?;
        let protocol = hash(&"nf.portfolio.builtin.arithmetic.v1")?;
        let dataset = hash(&fixture.inventory)?;
        let oracle: ContentDigest = serde_json::from_value(descriptor["oracle_code"].clone())
            .map_err(|_| error("oracle identity absent"))?;
        let manifest = GymManifest {
            schema_version: 1,
            package: "nf.arithmetic".into(),
            name: format!("arithmetic.{head}.v1"),
            version: "1".into(),
            origin: GymOrigin::Foundry,
            adapter: factory("nf.arithmetic.local", descriptor_id.clone()),
            code: oracle.clone(),
            dependencies: hash(&"rust_native_no_extra_packages")?,
            image: None,
            dataset,
            input_schema: hash(&"integer_pair_-8_8")?,
            output_schema: hash(&"finite_arithmetic_number")?,
            scorer: oracle.clone(),
            oracle,
            lifecycle_protocol: protocol.clone(),
            model_interface: factory(
                "nf.arithmetic.relu",
                fixture.source.manifest().architecture.config.clone(),
            ),
            data_policy: protocol.clone(),
            scoring_protocol: descriptor_id,
            privacy_policy: protocol.clone(),
            metrics: BTreeMap::from([(
                "accuracy".into(),
                MetricSchema {
                    domain: MetricDomain::Quality,
                    direction: Direction::HigherIsBetter,
                    unit: MetricUnit::Fraction,
                    lower: Some(0.0),
                    upper: Some(1.0),
                },
            )]),
            execution_trust: ExecutionTrust::TrustedFixture,
            isolation_policy: protocol,
            validation: evidence.clone(),
            approval: None,
            usage_rights: "Apache-2.0; synthetic public arithmetic fixture".into(),
            confidentiality: nf_core::schemas::AccessClass::Public,
            max_steps: 1,
            max_episode_ms: 1000,
            max_output_bytes: 65536,
        };
        manifest.validate()?;
        Ok(Self {
            id: manifest.digest()?,
            manifest,
            evidence,
            dimensions: BTreeSet::from([Dimension::TargetTasks, Dimension::ProtectedBehaviors]),
            validation_state: "locally_probed_fixture_only".into(),
            validation_bytes: validation_doc.bytes().to_vec(),
        })
    }
    pub fn user(
        snapshot: &intake::Snapshot,
        approval: &validation::Approval,
        store: &ArtifactStore,
    ) -> Result<Self> {
        if approval.record().snapshot != *snapshot.id()
            || approval.record().protocol != validation::protocol_identity()
        {
            return Err(error(
                "live approval does not bind this exact snapshot/protocol",
            ));
        }
        // CAS verifies the original frozen plan, not caller-proposed metric names/ranges.
        let plan = validation::Plan::parse(&store.read_bounded(&approval.record().plan, 65536)?)?;
        if plan.snapshot != *snapshot.id() {
            return Err(error("approved validation plan snapshot differs"));
        }
        let m = snapshot.manifest();
        for name in m.assets.keys() {
            snapshot.asset(store, name)?;
        }
        let (input, output, score, oracle, steps, wall) = match &m.draft.interface {
            intake::Interface::SingleShot {
                input,
                output,
                score,
                oracle,
                ..
            } => (
                hash(input)?,
                hash(output)?,
                hash(&(snapshot.id(), score))?,
                hash(&(snapshot.id(), oracle))?,
                1,
                plan.limits.wall_ms,
            ),
            intake::Interface::Episode {
                reset,
                result,
                score,
                oracle,
                max_steps,
                max_episode_ms,
                ..
            } => (
                hash(reset)?,
                hash(result)?,
                hash(&(snapshot.id(), score))?,
                hash(&(snapshot.id(), oracle))?,
                *max_steps,
                *max_episode_ms,
            ),
        };
        let policy = hash(&("nf.portfolio.approved_custom.v1", snapshot.id()))?;
        let manifest = GymManifest {
            schema_version: 1,
            package: "local.custom".into(),
            name: m.draft.name.clone(),
            version: m.draft.version.clone(),
            origin: GymOrigin::User,
            adapter: factory("nf.custom.python", snapshot.id().clone()),
            code: m.code_identity.clone(),
            dependencies: hash(&m.dependency)?,
            image: Some(m.dependency.image.clone()),
            dataset: m.data_identity.clone(),
            input_schema: input,
            output_schema: output,
            scorer: score,
            oracle,
            lifecycle_protocol: hash(&m.draft.interface)?,
            model_interface: factory("nf.scoped.bytes", policy.clone()),
            data_policy: policy.clone(),
            scoring_protocol: approval.record().plan.clone(),
            privacy_policy: policy.clone(),
            metrics: BTreeMap::from([(
                "score".into(),
                MetricSchema {
                    domain: MetricDomain::Quality,
                    direction: Direction::HigherIsBetter,
                    unit: MetricUnit::Score,
                    lower: Some(plan.scoring.minimum),
                    upper: Some(plan.scoring.maximum),
                },
            )]),
            execution_trust: ExecutionTrust::UntrustedSubprocess,
            isolation_policy: approval.record().runtime_profile.clone(),
            validation: approval.record().validation.clone(),
            approval: Some(approval.id().clone()),
            usage_rights: m.draft.usage_rights.clone(),
            confidentiality: m.draft.confidentiality,
            max_steps: steps,
            max_episode_ms: wall,
            max_output_bytes: plan.limits.output_bytes as u64,
        };
        manifest.validate()?;
        Ok(Self {
            id: manifest.digest()?,
            manifest,
            evidence: approval.id().clone(),
            dimensions: BTreeSet::from([Dimension::TargetTasks, Dimension::ProtectedBehaviors]),
            validation_state: "approved_exact_snapshot_historical_runtime_recheck_required".into(),
            validation_bytes: doc(approval.record())?.bytes().to_vec(),
        })
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Dimension {
    TargetTasks,
    ProtectedBehaviors,
    Mechanism,
    ContextBoundaries,
    CacheCorrectness,
    Tools,
    Latency,
    Memory,
    Safety,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoverageLink {
    pub binding: ContentDigest,
    pub metric: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoverageNeed {
    pub dimension: Dimension,
    pub mandatory: bool,
    pub measures: Vec<CoverageLink>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoveragePlan {
    pub schema_version: u32,
    pub objective: ContentDigest,
    pub architecture_surgery: bool,
    pub needs: Vec<CoverageNeed>,
}
#[derive(Debug, Clone, Serialize)]
pub struct CoverageRow {
    pub dimension: Dimension,
    pub mandatory: bool,
    pub status: String,
    pub bindings: Vec<ContentDigest>,
    pub recommendation: String,
}
#[derive(Debug, Clone, Serialize)]
pub struct CoverageReport {
    pub rows: Vec<CoverageRow>,
    pub mandatory_gaps: usize,
    pub duplicate_data_sources: usize,
    pub cost_envelope: Value,
    pub standard_five_gym_declared: bool,
    pub standard_five_gym_passed: bool,
    pub validation_states: BTreeMap<ContentDigest, String>,
    pub scope: String,
}
/// Private executable *planning* witness, not budget/lease/split custody authority.
pub struct Portfolio {
    value: Validated<GymPortfolio>,
    coverage: CoveragePlan,
    report: CoverageReport,
    id: ContentDigest,
    _sources: Vec<Arc<GymSource>>,
}
impl Portfolio {
    pub fn compose(bytes: &[u8], coverage: &[u8], sources: Vec<Arc<GymSource>>) -> Result<Self> {
        let value = Validated::<GymPortfolio>::parse(bytes)?;
        if value.independent.scope != ClaimScope::FixtureOnly {
            return Err(error("current live sources certify finite fixture probes only, not external holdout custody or generalization"));
        }
        let coverage_doc = CanonicalDocument::parse(coverage, Limits::default())?;
        let plan: CoveragePlan = serde_json::from_value(coverage_doc.value().clone())
            .map_err(|_| error("invalid bounded coverage plan"))?;
        if plan.schema_version != 1
            || plan.needs.is_empty()
            || plan.needs.len() > 32
            || value.coverage != hash(&plan)?
        {
            return Err(error("exact nonempty versioned coverage identity required"));
        }
        if sources.len() != value.gyms.len() || sources.len() > 64 {
            return Err(error("one live exact source witness per gym required"));
        }
        let resolved: BTreeMap<_, _> = sources.iter().map(|s| (s.id.clone(), s)).collect();
        if resolved.len() != sources.len() {
            return Err(error("duplicate live gym source"));
        }
        for (id, m) in &value.gyms {
            let source = resolved
                .get(id)
                .ok_or_else(|| error("unresolved gym or DTO-only validation/approval"))?;
            if doc(m)?.bytes() != doc(&source.manifest)?.bytes() {
                return Err(error("source witness manifest changed"));
            }
        }
        if value.bindings.iter().any(|b| b.objective != plan.objective) {
            return Err(error("coverage and constituent objectives differ"));
        }
        let schedule: Vec<_> = value
            .bindings
            .iter()
            .map(|b| (&b.id, &b.schedule))
            .collect();
        if value.curriculum != hash(&schedule)? {
            return Err(error(
                "curriculum must bind exact immutable constituent schedules",
            ));
        }
        let mut needs = BTreeMap::new();
        for need in &plan.needs {
            if need.measures.len() > 64 || needs.insert(need.dimension, need.clone()).is_some() {
                return Err(error("duplicate/oversized coverage dimension"));
            }
        }
        for dimension in [Dimension::TargetTasks, Dimension::ProtectedBehaviors] {
            let row = needs.entry(dimension).or_insert(CoverageNeed {
                dimension,
                mandatory: true,
                measures: vec![],
            });
            row.mandatory = true;
        }
        if plan.architecture_surgery {
            for dimension in [
                Dimension::ContextBoundaries,
                Dimension::CacheCorrectness,
                Dimension::Latency,
            ] {
                let row = needs.entry(dimension).or_insert(CoverageNeed {
                    dimension,
                    mandatory: true,
                    measures: vec![],
                });
                row.mandatory = true;
            }
        }
        let mut rows = Vec::new();
        let mut gaps = 0;
        for need in needs.values() {
            let mut ids = BTreeSet::new();
            let mut supported = true;
            for link in &need.measures {
                if link.metric.is_empty()
                    || link.metric.len() > 128
                    || !ids.insert(link.binding.clone())
                {
                    return Err(error("duplicate or invalid coverage measure"));
                }
                let b = value
                    .bindings
                    .iter()
                    .find(|b| b.id == link.binding)
                    .ok_or_else(|| error("coverage references unknown binding"))?;
                if !b.metrics.iter().any(|m| m.metric == link.metric) {
                    return Err(error("coverage references undeclared bound metric"));
                }
                let s = resolved[&b.gym];
                // A claim written in JSON cannot upgrade a fixture into a cache,
                // safety, latency, or causal measurement protocol.
                supported &= s.dimensions.contains(&need.dimension)
                    && b.required
                    && b.metrics
                        .iter()
                        .any(|m| m.metric == link.metric && m.required);
                supported &= match need.dimension {
                    Dimension::TargetTasks => b.roles.contains(&GymRole::Target),
                    Dimension::ProtectedBehaviors => b.roles.contains(&GymRole::Retention),
                    _ => true,
                };
            }
            let status = if ids.is_empty() {
                "missing"
            } else if !supported {
                "weak_or_unsupported"
            } else {
                "fixture_observed"
            };
            if need.mandatory && status != "fixture_observed" {
                gaps += 1;
            }
            rows.push(CoverageRow{dimension:need.dimension,mandatory:need.mandatory,status:status.into(),bindings:ids.into_iter().collect(),recommendation:if status=="fixture_observed"{"retain fixture-only scope".into()}else{format!("add an independently validated {:?} gym/protocol; revise portfolio before execution",need.dimension)}});
        }
        let distinct_datasets: BTreeSet<_> = sources.iter().map(|s| &s.manifest.dataset).collect();
        let mut wall_bound = 0u64;
        let mut output_bound = 0u64;
        for b in &value.bindings {
            let count = u64::from(b.schedule.development_episodes)
                + u64::from(b.schedule.selection_episodes)
                + u64::from(b.schedule.holdout_episodes);
            let m = &value.gyms[&b.gym];
            wall_bound = wall_bound
                .checked_add(
                    count
                        .checked_mul(m.max_episode_ms)
                        .ok_or_else(|| error("schedule cost overflow"))?,
                )
                .ok_or_else(|| error("schedule cost overflow"))?;
            output_bound = output_bound
                .checked_add(
                    count
                        .checked_mul(m.max_output_bytes)
                        .ok_or_else(|| error("schedule output overflow"))?,
                )
                .ok_or_else(|| error("schedule output overflow"))?;
        }
        let report = CoverageReport {
            rows,
            mandatory_gaps: gaps,
            duplicate_data_sources: sources.len() - distinct_datasets.len(),
            cost_envelope: json!({"scheduled_episode_wall_ms_upper_bound":wall_bound,"scheduled_output_bytes_upper_bound":output_bound,"cpu_ms":null,"memory_peak_bytes":null,"setup_and_runtime_overhead":"additional_unknown_not_zero","budget_admission":false}),
            standard_five_gym_declared: value.profile == AssuranceProfile::StandardFiveGym,
            standard_five_gym_passed: false,
            validation_states: sources
                .iter()
                .map(|s| {
                    (
                        s.id.clone(),
                        format!("{};evidence={}", s.validation_state, s.evidence),
                    )
                })
                .collect(),
            scope: "live_source_backed_fixture_planning_not_generalization_or_execution_authority"
                .into(),
        };
        let id = hash(&(value.digest()?, &plan, &report))?;
        Ok(Self {
            value,
            coverage: plan,
            report,
            id,
            _sources: sources,
        })
    }
    pub fn id(&self) -> &ContentDigest {
        &self.id
    }
    pub fn record(&self) -> Result<Value> {
        Ok(
            json!({"schema_version":1,"id":self.id,"contract_portfolio_digest":self.value.digest()?,"portfolio":&*self.value,"coverage_plan":self.coverage,"coverage_report":self.report,"status":if self.report.mandatory_gaps==0{"bound_for_planning"}else{"blocked_missing_coverage"},"dispatch_allowed":false,"split_custody_verified":false,"promotion_status":"not_evaluated"}),
        )
    }
    pub fn schedule(&self, stage: Stage, candidate: u64) -> Result<Value> {
        if candidate == 0 || candidate > 1_000_000 {
            return Err(error("candidate ordinal must be in 1..1000000"));
        }
        let mut entries = Vec::new();
        let mut tasks = BTreeSet::new();
        let mut groups = BTreeSet::new();
        for b in &self.value.bindings {
            let (split, count, enabled) = match stage {
                Stage::Development => {
                    (&b.splits.development, b.schedule.development_episodes, true)
                }
                Stage::Selection => (&b.splits.selection, b.schedule.selection_episodes, true),
                Stage::Final => (
                    &b.splits.holdout,
                    b.schedule.holdout_episodes,
                    b.schedule.final_required,
                ),
            };
            let due = enabled
                && (stage == Stage::Final
                    || (b.schedule.every_n_candidates != 0
                        && (candidate - 1)
                            .is_multiple_of(u64::from(b.schedule.every_n_candidates))));
            let chosen: Vec<_> = if due {
                split.items.iter().take(count as usize).collect()
            } else {
                vec![]
            };
            for item in &chosen {
                tasks.insert(item.item.clone());
                groups.insert(item.group.clone());
            }
            entries.push(json!({"binding":b.id,"gym":b.gym,"due":due,"required":b.required,"deferred_required":b.required && !due,"split":split.role,"items":chosen,"weights":b.metrics.iter().map(|m|json!({"metric":m.metric,"priority":m.priority})).collect::<Vec<_>>()}));
        }
        Ok(
            json!({"portfolio":self.id,"stage":stage,"candidate":candidate,"entries":entries,"unique_tasks":tasks.len(),"unique_groups":groups.len(),"dispatch_allowed":false,"scope":"fixed_schedule_metadata_not_split_access"}),
        )
    }
    pub fn assess(&self, bytes: &[u8]) -> Result<Value> {
        let observations_doc = CanonicalDocument::parse(bytes, Limits::default())?;
        let input: Assessment = serde_json::from_value(observations_doc.value().clone())
            .map_err(|_| error("invalid assessment data"))?;
        if input.schema_version != 1
            || input.portfolio != self.id
            || input.observations.len() > 4096
        {
            return Err(error(
                "assessment requires exact portfolio identity and bounded observations",
            ));
        }
        let mut seen = BTreeMap::new();
        for o in &input.observations {
            if seen
                .insert((&o.binding, &o.metric, o.comparator), o)
                .is_some()
            {
                return Err(error(
                    "duplicate observation cannot count as extra evidence",
                ));
            }
            let b = self
                .value
                .bindings
                .iter()
                .find(|b| b.id == o.binding)
                .ok_or_else(|| error("unknown observation binding"))?;
            if !b
                .metrics
                .iter()
                .any(|m| m.metric == o.metric && m.comparator == o.comparator)
            {
                return Err(error("unknown observation metric/comparator"));
            }
            let schema = &self.value.gyms[&b.gym].metrics[&o.metric];
            if o.unit != schema.unit
                || b.metrics
                    .iter()
                    .find(|m| m.metric == o.metric && m.comparator == o.comparator)
                    .unwrap()
                    .statistic
                    != o.statistic
            {
                return Err(error(
                    "observation units/statistic differ from frozen metric",
                ));
            }
            if [o.value, o.baseline].into_iter().flatten().any(|n| {
                !n.is_finite()
                    || schema.lower.is_some_and(|v| n < v)
                    || schema.upper.is_some_and(|v| n > v)
            }) {
                return Err(error("observation outside finite metric domain"));
            }
        }
        let mut rows = Vec::new();
        let mut failed = Vec::new();
        for b in &self.value.bindings {
            for m in &b.metrics {
                let o = seen.get(&(&b.id, &m.metric, m.comparator));
                let schema = &self.value.gyms[&b.gym].metrics[&m.metric];
                let value = o.and_then(|o| o.value);
                let baseline = o.and_then(|o| o.baseline);
                let needs_baseline =
                    m.threshold.max_regression.is_some() || m.threshold.min_gain.is_some();
                let gain = value.zip(baseline).map(|(a, b)| {
                    if schema.direction == Direction::HigherIsBetter {
                        a - b
                    } else {
                        b - a
                    }
                });
                let complete = o.is_some_and(|o| o.status == ObservationStatus::Complete)
                    && value.is_some()
                    && (!needs_baseline || baseline.is_some());
                let pass = complete
                    && value.is_some_and(|v| {
                        m.threshold.floor.is_none_or(|n| v >= n)
                            && m.threshold.ceiling.is_none_or(|n| v <= n)
                    })
                    && m.threshold
                        .max_regression
                        .is_none_or(|n| gain.is_some_and(|v| v >= -n))
                    && m.threshold
                        .min_gain
                        .is_none_or(|n| gain.is_some_and(|v| v >= n));
                let hard = b.required && m.required;
                if hard && !pass {
                    failed.push(json!({"binding":b.id,"metric":m.metric,"status":if complete{"failed"}else{"missing_or_incomplete"}}));
                }
                rows.push(json!({"binding":b.id,"metric":m.metric,"comparator":m.comparator,"priority":m.priority,"threshold":m.threshold,"direction":schema.direction,"observation_status":o.map(|o|o.status),"hard":hard,"value":value,"baseline":baseline,"gain":gain,"unit":schema.unit,"statistic":m.statistic,"passed":pass,"complete":complete}));
            }
        }
        Ok(
            json!({"schema_version":1,"portfolio":self.id,"rows":rows,"failed_required":failed,"hard_constraints_satisfied":failed.is_empty(),"mandatory_coverage_gaps":self.report.mandatory_gaps,"constraints_satisfied":failed.is_empty() && self.report.mandatory_gaps==0,"weighted_compensation_allowed":false,"scientific_acceptance":false,"outputs_eligible":false,"scope":"unverified_supplied_measurement_constraint_assessment_only"}),
        )
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    Development,
    Selection,
    Final,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationStatus {
    Complete,
    TimedOut,
    Missing,
    Failed,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    pub binding: ContentDigest,
    pub metric: String,
    pub comparator: Comparator,
    pub status: ObservationStatus,
    pub value: Option<f64>,
    pub baseline: Option<f64>,
    pub unit: MetricUnit,
    pub statistic: Statistic,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Assessment {
    pub schema_version: u32,
    pub portfolio: ContentDigest,
    pub observations: Vec<Observation>,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn portfolio_coverage_actual_builtin_receipts_bind_repeatable_observations() {
        let a = GymSource::builtin("sum").unwrap();
        let b = GymSource::builtin("sum").unwrap();
        assert_eq!(a.id(), b.id());
        assert_eq!(
            ContentDigest::of_bytes(a.validation_bytes()),
            a.manifest().validation
        );
        let report: Value = serde_json::from_slice(a.validation_bytes()).unwrap();
        assert_eq!(report["observations"].as_array().unwrap().len(), 289);
        assert!(GymSource::builtin("unregistered.eval_python").is_err());
    }
    #[test]
    fn portfolio_coverage_arithmetic_observations_do_not_qualify_surgery_or_performance() {
        let source = GymSource::builtin("maximum").unwrap();
        assert!(source.dimensions.contains(&Dimension::TargetTasks));
        for dimension in [
            Dimension::CacheCorrectness,
            Dimension::ContextBoundaries,
            Dimension::Latency,
            Dimension::Safety,
            Dimension::Mechanism,
        ] {
            assert!(!source.dimensions.contains(&dimension));
        }
    }
}
