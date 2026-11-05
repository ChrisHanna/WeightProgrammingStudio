//! Frozen finite-fixture / paired group bootstrap analysis, not an evidence importer.
use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use nf_core::error::{CoreError, ErrorCode, Result};
use nf_core::study::{
    Comparator, Direction, Estimator, MetricSchema, MetricUnit, Statistic, Threshold,
};
use nf_surgeon::cancellation::CancellationToken;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

fn error(text: &str) -> CoreError {
    CoreError::new(ErrorCode::InvalidValue, "/paired_statistics", text)
}
fn doc(v: &impl Serialize) -> Result<CanonicalDocument> {
    CanonicalDocument::parse(
        &serde_json::to_vec(v).map_err(|_| error("encoding failed"))?,
        Limits::default(),
    )
}
fn hash(v: &impl Serialize) -> Result<ContentDigest> {
    Ok(doc(v)?.digest())
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Task {
    pub item: ContentDigest,
    pub group: ContentDigest,
    pub input: ContentDigest,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Contrast {
    pub id: ContentDigest,
    pub candidate: ContentDigest,
    pub baseline: ContentDigest,
    pub comparator: Comparator,
    pub binding: ContentDigest,
    pub metric: String,
    pub schema: MetricSchema,
    pub value_statistic: Statistic,
    pub context: ContentDigest,
    pub workload: ContentDigest,
    pub measurement_protocol: ContentDigest,
    pub tasks: Vec<Task>,
    pub threshold: Threshold,
    pub required: bool,
    pub no_op_control: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Protocol {
    pub schema_version: u32,
    pub family: ContentDigest,
    pub estimator: Estimator,
    pub scope: String,
    pub estimand: String,
    pub minimum_groups: u32,
    pub minimum_tasks: u32,
    pub seeds: Vec<u64>,
    pub confidence: f64,
    pub resamples: u32,
    pub bootstrap_seed: u64,
    pub maximum_comparisons: u32,
    pub contrasts: Vec<Contrast>,
}
/// Private frozen protocol witness; its record alone is not a prefrozen final panel.
pub struct Plan {
    value: Protocol,
    id: ContentDigest,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Arm {
    Baseline,
    Candidate,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Complete,
    Missing,
    Failed,
    TimedOut,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    pub contrast: ContentDigest,
    pub item: ContentDigest,
    pub group: ContentDigest,
    pub input: ContentDigest,
    pub seed: u64,
    pub arm: Arm,
    pub model: ContentDigest,
    pub context: ContentDigest,
    pub workload: ContentDigest,
    pub measurement_protocol: ContentDigest,
    pub unit: MetricUnit,
    pub value_statistic: Statistic,
    pub status: Status,
    pub value: Option<f64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Data {
    pub schema_version: u32,
    pub plan: ContentDigest,
    pub observations: Vec<Observation>,
}
/// Compact on-disk observations refer to the immutable plan for all shared
/// model, task, metric and runtime identity fields. They carry no authority.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CompactObservation {
    pub contrast: ContentDigest,
    pub item: ContentDigest,
    pub seed: u64,
    pub arm: Arm,
    pub status: Status,
    pub value: Option<f64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CompactData {
    pub schema_version: u32,
    pub plan: ContentDigest,
    pub observations: Vec<CompactObservation>,
}
impl CompactData {
    pub(crate) fn from_data(plan: &Plan, data: &Data, cancel: &CancellationToken) -> Result<Self> {
        if data.schema_version != 1 || data.plan != plan.id || data.observations.len() > 8192 {
            return Err(error(
                "exact frozen statistical plan and bounded data required",
            ));
        }
        let mut observations = Vec::with_capacity(data.observations.len());
        for value in &data.observations {
            cancel.check()?;
            let contrast = plan
                .value
                .contrasts
                .iter()
                .find(|contrast| contrast.id == value.contrast)
                .ok_or_else(|| error("unknown contrast"))?;
            let task = contrast
                .tasks
                .iter()
                .find(|task| task.item == value.item)
                .ok_or_else(|| error("observation outside frozen task support"))?;
            validate_observation(&plan.value, contrast, task, value)?;
            observations.push(CompactObservation {
                contrast: value.contrast.clone(),
                item: value.item.clone(),
                seed: value.seed,
                arm: value.arm,
                status: value.status,
                value: value.value,
            });
        }
        Ok(Self {
            schema_version: data.schema_version,
            plan: data.plan.clone(),
            observations,
        })
    }
}
/// Shared validation prevents compaction from dropping mismatched identity or
/// numeric fields and preserves the original full-observation checks.
fn validate_observation(p: &Protocol, c: &Contrast, t: &Task, o: &Observation) -> Result<()> {
    let model = match o.arm {
        Arm::Baseline => &c.baseline,
        Arm::Candidate => &c.candidate,
    };
    if o.group != t.group
        || o.input != t.input
        || !p.seeds.contains(&o.seed)
        || &o.model != model
        || o.context != c.context
        || o.workload != c.workload
        || o.measurement_protocol != c.measurement_protocol
        || o.unit != c.schema.unit
        || o.value_statistic != c.value_statistic
    {
        return Err(error(
            "unmatched model/task/group/seed/units/statistic/workload/context/protocol",
        ));
    }
    if o.value.is_some_and(|v| {
        !v.is_finite()
            || v.abs() > 1e12
            || c.schema.lower.is_some_and(|x| v < x)
            || c.schema.upper.is_some_and(|x| v > x)
    }) {
        return Err(error("nonfinite or out-of-domain paired outcome"));
    }
    Ok(())
}
pub struct Analysis {
    id: ContentDigest,
    report: Value,
}
impl Analysis {
    pub fn id(&self) -> &ContentDigest {
        &self.id
    }
    pub fn report(&self) -> &Value {
        &self.report
    }
}
impl Plan {
    pub fn freeze(bytes: &[u8]) -> Result<Self> {
        let d = CanonicalDocument::parse(bytes, Limits::default())?;
        let p: Protocol = serde_json::from_value(d.value().clone())
            .map_err(|_| error("invalid bounded statistical protocol"))?;
        if p.schema_version != 1
            || p.estimand != "equal_weight_group_mean_of_task_seed_means"
            || p.minimum_groups == 0
            || p.minimum_tasks == 0
            || p.seeds.is_empty()
            || p.seeds.len() > 16
            || p.seeds.iter().collect::<BTreeSet<_>>().len() != p.seeds.len()
            || p.contrasts.is_empty()
            || p.contrasts.len() > 64
            || p.maximum_comparisons < p.contrasts.len() as u32
            || p.maximum_comparisons > 256
            || !p.confidence.is_finite()
            || p.confidence <= 0.0
            || p.confidence > 1.0
        {
            return Err(error("finite support, exact estimand, unique paired seeds and frozen multiplicity family required"));
        }
        match p.estimator {
            Estimator::DeterministicExact => {
                if p.scope != "deterministic_fixture_only"
                    || p.confidence != 1.0
                    || p.resamples != 0
                {
                    return Err(error("exact finite fixture analysis needs confidence one and no bootstrap/generalization claim"));
                }
            }
            Estimator::PairedGroupBootstrap => {
                if p.scope != "paired_empirical_analysis"
                    || p.confidence >= 1.0
                    || p.minimum_groups < 2
                    || p.seeds.len() < 2
                    || !(1000..=50_000).contains(&p.resamples)
                {
                    return Err(error("paired bootstrap requires multiple groups/repeats and finite empirical confidence/resamples"));
                }
            }
        }
        let mut ids = BTreeSet::new();
        let mut dimensions = BTreeSet::new();
        let mut baselines = BTreeMap::new();
        let mut controls = BTreeMap::new();
        let mut all_tasks = BTreeMap::new();
        let mut input_groups = BTreeMap::new();
        let mut rows = 0u64;
        let mut draws = 0u64;
        for c in &p.contrasts {
            c.schema.validate()?;
            c.threshold.validate(&c.schema)?;
            if baselines
                .insert(c.comparator, &c.baseline)
                .is_some_and(|id| id != &c.baseline)
                || controls
                    .insert(&c.candidate, c.no_op_control)
                    .is_some_and(|flag| flag != c.no_op_control)
            {
                return Err(error(
                    "baseline/control identity changed across frozen family metrics",
                ));
            }
            if !ids.insert(&c.id)
                || !dimensions.insert((
                    &c.candidate,
                    &c.baseline,
                    c.comparator,
                    &c.binding,
                    &c.metric,
                ))
                || c.metric.is_empty()
                || c.metric.len() > 128
                || c.metric.chars().any(char::is_control)
                || c.tasks.is_empty()
                || c.tasks.len() > 256
                || (c.required
                    && c.threshold.floor.is_none()
                    && c.threshold.ceiling.is_none()
                    && c.threshold.min_gain.is_none()
                    && c.threshold.max_regression.is_none())
            {
                return Err(error("distinct bounded candidates/metrics/contrasts and required thresholds must be frozen"));
            }
            let mut items = BTreeSet::new();
            let mut inputs = BTreeSet::new();
            let mut groups = BTreeSet::new();
            for t in &c.tasks {
                if !items.insert(&t.item) || !inputs.insert(&t.input) {
                    return Err(error(
                        "duplicate task/input cannot inflate independent support",
                    ));
                }
                if all_tasks
                    .insert(&t.item, (&t.group, &t.input))
                    .is_some_and(|old| old != (&t.group, &t.input))
                    || input_groups
                        .insert(&t.input, &t.group)
                        .is_some_and(|old| old != &t.group)
                {
                    return Err(error(
                        "task/input group relabeling across comparisons denied",
                    ));
                }
                groups.insert(&t.group);
            }
            rows += c.tasks.len() as u64 * p.seeds.len() as u64 * 2;
            draws += u64::from(p.resamples) * groups.len() as u64;
        }
        if rows > 4096 || draws > 10_000_000 {
            return Err(error("paired row/bootstrap draw admission limit exceeded"));
        }
        let id = hash(&p)?;
        Ok(Self { value: p, id })
    }
    pub fn id(&self) -> &ContentDigest {
        &self.id
    }
    /// Internal inspection only; a plan is not execution evidence.
    pub(crate) fn protocol(&self) -> &Protocol {
        &self.value
    }
    pub fn record(&self) -> Result<Value> {
        Ok(
            json!({"schema_version":1,"id":self.id,"protocol":self.value,"correction":"two_sided_bonferroni_over_maximum_planned_comparisons","quantiles":"conservative_empirical_floor_lower_ceil_upper","bootstrap_rng":"xorshift64_rejection_index_v1","scope":"analysis_protocol_not_execution_or_prefrozen_final_panel_authority"}),
        )
    }
    pub fn analyze(&self, bytes: &[u8], cancel: &CancellationToken) -> Result<Analysis> {
        cancel.check()?;
        let document = CanonicalDocument::parse(bytes, Limits::default())?;
        let data: Data = serde_json::from_value(document.value().clone())
            .map_err(|_| error("invalid finite paired observation record"))?;
        self.analyze_data(data, document.digest(), cancel)
    }
    /// Restore shared identities from the frozen plan before running the exact
    /// same validation and statistics used by the full observation format.
    pub(crate) fn analyze_compact(
        &self,
        bytes: &[u8],
        cancel: &CancellationToken,
    ) -> Result<Analysis> {
        cancel.check()?;
        let document = CanonicalDocument::parse(bytes, Limits::default())?;
        let compact: CompactData = serde_json::from_value(document.value().clone())
            .map_err(|_| error("invalid compact paired observation record"))?;
        self.analyze_data(
            self.expand_compact(&compact, cancel)?,
            document.digest(),
            cancel,
        )
    }
    pub(crate) fn expand_compact(
        &self,
        compact: &CompactData,
        cancel: &CancellationToken,
    ) -> Result<Data> {
        if compact.schema_version != 1
            || compact.plan != self.id
            || compact.observations.len() > 8192
        {
            return Err(error(
                "exact frozen statistical plan and bounded data required",
            ));
        }
        let mut observations = Vec::with_capacity(compact.observations.len());
        for row in &compact.observations {
            cancel.check()?;
            let contrast = self
                .value
                .contrasts
                .iter()
                .find(|value| value.id == row.contrast)
                .ok_or_else(|| error("unknown contrast"))?;
            let task = contrast
                .tasks
                .iter()
                .find(|value| value.item == row.item)
                .ok_or_else(|| error("observation outside frozen task support"))?;
            let model = match row.arm {
                Arm::Baseline => &contrast.baseline,
                Arm::Candidate => &contrast.candidate,
            };
            observations.push(Observation {
                contrast: row.contrast.clone(),
                item: row.item.clone(),
                group: task.group.clone(),
                input: task.input.clone(),
                seed: row.seed,
                arm: row.arm,
                model: model.clone(),
                context: contrast.context.clone(),
                workload: contrast.workload.clone(),
                measurement_protocol: contrast.measurement_protocol.clone(),
                unit: contrast.schema.unit,
                value_statistic: contrast.value_statistic,
                status: row.status,
                value: row.value,
            });
        }
        Ok(Data {
            schema_version: compact.schema_version,
            plan: compact.plan.clone(),
            observations,
        })
    }
    fn analyze_data(
        &self,
        data: Data,
        data_digest: ContentDigest,
        cancel: &CancellationToken,
    ) -> Result<Analysis> {
        if data.schema_version != 1 || data.plan != self.id || data.observations.len() > 8192 {
            return Err(error(
                "exact frozen statistical plan and bounded data required",
            ));
        }
        let p = &self.value;
        let mut observations = BTreeMap::new();
        let mut dropped = 0usize;
        for o in &data.observations {
            let c = p
                .contrasts
                .iter()
                .find(|c| c.id == o.contrast)
                .ok_or_else(|| error("unknown contrast"))?;
            let t = c
                .tasks
                .iter()
                .find(|t| t.item == o.item)
                .ok_or_else(|| error("observation outside frozen task support"))?;
            validate_observation(p, c, t, o)?;
            let key = (&o.contrast, &o.item, o.seed, o.arm);
            if let Some(old) = observations.insert(key, o) {
                if doc(old)?.bytes() != doc(o)?.bytes() {
                    return Err(error("conflicting duplicate observation"));
                }
                dropped += 1;
            }
        }
        let alpha = 1.0 - p.confidence;
        let tail = alpha / (2.0 * f64::from(p.maximum_comparisons));
        let mut results = Vec::new();
        for c in &p.contrasts {
            cancel.check()?;
            let mut groups: BTreeMap<&ContentDigest, Vec<(f64, f64)>> = BTreeMap::new();
            let mut incomplete = Vec::new();
            let mut complete_pairs = 0;
            let mut nondeterministic = false;
            for t in &c.tasks {
                let mut left = Vec::new();
                let mut right = Vec::new();
                for seed in &p.seeds {
                    let a = observations.get(&(&c.id, &t.item, *seed, Arm::Baseline));
                    let b = observations.get(&(&c.id, &t.item, *seed, Arm::Candidate));
                    match (a,b) {
                        (Some(a),Some(b)) if a.status==Status::Complete && b.status==Status::Complete && a.value.is_some() && b.value.is_some()=>{left.push(a.value.unwrap());right.push(b.value.unwrap());complete_pairs+=1;},
                        _=>incomplete.push(json!({"item":t.item,"seed":seed,"status":"missing_failed_or_incomplete_pair"})),
                    }
                }
                if left.len() == p.seeds.len() {
                    if matches!(p.estimator, Estimator::DeterministicExact)
                        && (left.iter().any(|v| v.to_bits() != left[0].to_bits())
                            || right.iter().any(|v| v.to_bits() != right[0].to_bits()))
                    {
                        nondeterministic = true;
                    }
                    groups
                        .entry(&t.group)
                        .or_default()
                        .push((mean(&left), mean(&right)));
                }
            }
            let mut base = json!({"contrast":c.id,"candidate":c.candidate,"baseline":c.baseline,"comparator":c.comparator,"binding":c.binding,"metric":c.metric,"unit":c.schema.unit,"direction":c.schema.direction,"threshold":c.threshold,"required":c.required,"unique_tasks":c.tasks.len(),"unique_groups":groups.len(),"seed_repeats":p.seeds.len(),"expected_pairs":c.tasks.len()*p.seeds.len(),"complete_pairs":complete_pairs,"incomplete_pairs":incomplete,"estimate":null,"interval":null,"candidate_interval":null,"status":"incomplete","reason":"frozen_pair_panel_incomplete","resamples":0,"positive_gain_eligible":false});
            if !base["incomplete_pairs"].as_array().unwrap().is_empty() {
                results.push(base);
                continue;
            }
            if groups.len() < p.minimum_groups as usize || c.tasks.len() < p.minimum_tasks as usize
            {
                base["status"] = json!("inconclusive");
                base["reason"] = json!("insufficient_unique_group_or_task_support");
                results.push(base);
                continue;
            }
            if nondeterministic {
                base["status"] = json!("inconclusive");
                base["reason"] =
                    json!("repeat_variation_cannot_establish_deterministic_exact_claim");
                results.push(base);
                continue;
            }
            let group_means: Vec<_> = groups
                .values()
                .map(|tasks| {
                    (
                        mean(&tasks.iter().map(|v| v.0).collect::<Vec<_>>()),
                        mean(&tasks.iter().map(|v| v.1).collect::<Vec<_>>()),
                    )
                })
                .collect();
            let original = mean(&group_means.iter().map(|p| p.0).collect::<Vec<_>>());
            let candidate = mean(&group_means.iter().map(|p| p.1).collect::<Vec<_>>());
            let gain = directional(c.schema.direction, original, candidate);
            base["estimate"] = json!({"baseline":original,"candidate":candidate,"gain":gain,"estimand":p.estimand});
            let (gain_ci, candidate_ci) = match p.estimator {
                Estimator::DeterministicExact => ([gain, gain], [candidate, candidate]),
                Estimator::PairedGroupBootstrap => {
                    if f64::from(p.resamples) * tail < 5.0 {
                        base["status"] = json!("inconclusive");
                        base["reason"] =
                            json!("insufficient_resampling_resolution_for_corrected_tail");
                        results.push(base);
                        continue;
                    }
                    let mut rng = Rng(seed(p.bootstrap_seed, &p.family, &c.id)?);
                    let mut gains = Vec::new();
                    let mut candidates = Vec::new();
                    let mut left = Vec::with_capacity(group_means.len());
                    let mut right = Vec::with_capacity(group_means.len());
                    for n in 0..p.resamples {
                        if n % 256 == 0 {
                            cancel.check()?;
                        }
                        left.clear();
                        right.clear();
                        for _ in 0..group_means.len() {
                            let pair = group_means[rng.index(group_means.len())];
                            left.push(pair.0);
                            right.push(pair.1);
                        }
                        let a = mean(&left);
                        let b = mean(&right);
                        gains.push(directional(c.schema.direction, a, b));
                        candidates.push(b);
                    }
                    base["resamples"] = json!(p.resamples);
                    (interval(&mut gains, tail), interval(&mut candidates, tail))
                }
            };
            base["interval"] = json!(gain_ci);
            base["candidate_interval"] = json!(candidate_ci);
            let noop = c.no_op_control || c.candidate == c.baseline;
            let status = if noop && c.threshold.min_gain.is_some() {
                "no_op_cannot_satisfy_positive_gain"
            } else {
                verdict(&c.threshold, gain_ci, candidate_ci)
            };
            base["status"] = json!(status);
            base["reason"] = json!(if matches!(p.estimator, Estimator::DeterministicExact) {
                "complete_declared_finite_fixture_only"
            } else {
                "finite_sample_paired_group_percentile_interval_not_guaranteed_population_coverage"
            });
            base["positive_gain_eligible"] =
                json!(status == "passed" && !noop && c.threshold.min_gain.is_some());
            results.push(base);
        }
        cancel.check()?;
        let all = results
            .iter()
            .all(|r| r["required"] != true || r["status"] == "passed");
        let report = json!({"schema_version":1,"plan":self.id,"family":p.family,"data_digest":data_digest,"estimator":p.estimator,"estimand":p.estimand,"confidence":p.confidence,"maximum_planned_comparisons":p.maximum_comparisons,"frozen_contrasts":p.contrasts.len(),"per_comparison_two_sided_alpha":if matches!(p.estimator,Estimator::DeterministicExact){0.0}else{alpha/f64::from(p.maximum_comparisons)},"correction":"bonferroni_reserved_family_including_metrics_candidates_baseline_and_confirmatory_contrasts","duplicates_dropped":dropped,"results":results,"all_required_analysis_conditions_satisfied":all,"scientific_acceptance":false,"outputs_eligible":false,"scope":"analysis_of_supplied_observations_not_execution_model_or_prefrozen_final_panel_authority"});
        Ok(Analysis {
            id: hash(&report)?,
            report,
        })
    }
}
fn mean(values: &[f64]) -> f64 {
    let mut sum: f64 = 0.0;
    let mut correction = 0.0;
    for value in values {
        let next = sum + value;
        correction += if sum.abs() >= value.abs() {
            (sum - next) + value
        } else {
            (value - next) + sum
        };
        sum = next;
    }
    (sum + correction) / values.len() as f64
}
fn directional(direction: Direction, a: f64, b: f64) -> f64 {
    if direction == Direction::HigherIsBetter {
        b - a
    } else {
        a - b
    }
}
fn interval(values: &mut [f64], tail: f64) -> [f64; 2] {
    values.sort_by(f64::total_cmp);
    let n = (values.len() - 1) as f64;
    [
        values[(tail * n).floor() as usize],
        values[((1.0 - tail) * n).ceil() as usize],
    ]
}
fn verdict(t: &Threshold, g: [f64; 2], c: [f64; 2]) -> &'static str {
    if t.floor.is_some_and(|x| c[1] < x)
        || t.ceiling.is_some_and(|x| c[0] > x)
        || t.min_gain.is_some_and(|x| g[1] < x)
        || t.max_regression.is_some_and(|x| g[1] < -x)
    {
        "failed"
    } else if t.floor.is_none_or(|x| c[0] >= x)
        && t.ceiling.is_none_or(|x| c[1] <= x)
        && t.min_gain.is_none_or(|x| g[0] >= x)
        && t.max_regression.is_none_or(|x| g[0] >= -x)
    {
        "passed"
    } else {
        "inconclusive"
    }
}
fn seed(value: u64, family: &ContentDigest, contrast: &ContentDigest) -> Result<u64> {
    let h = hash(&("nf.paired_group_rng.v1", value, family, contrast))?;
    let value =
        u64::from_str_radix(&h.as_str()[7..23], 16).map_err(|_| error("invalid seed digest"))?;
    Ok(if value == 0 {
        0x9e3779b97f4a7c15
    } else {
        value
    })
}
struct Rng(u64);
impl Rng {
    fn index(&mut self, n: usize) -> usize {
        let n = n as u64;
        let zone = u64::MAX - u64::MAX % n;
        loop {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            if self.0 < zone {
                return (self.0 % n) as usize;
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn paired_statistics_controlled_intervals_use_lower_bounds_not_point_gains() {
        let t = Threshold {
            floor: None,
            ceiling: None,
            max_regression: None,
            min_gain: Some(0.1),
        };
        assert_eq!(verdict(&t, [-0.2, 0.5], [0.0, 1.0]), "inconclusive");
        assert_eq!(verdict(&t, [0.2, 0.5], [0.0, 1.0]), "passed");
        assert_eq!(verdict(&t, [-0.5, 0.05], [0.0, 1.0]), "failed");
    }
    #[test]
    fn paired_statistics_seeded_uniform_group_draws_and_quantiles_are_repeatable() {
        let mut a = Rng(17);
        let mut b = Rng(17);
        for _ in 0..1000 {
            assert_eq!(a.index(7), b.index(7));
        }
        assert_eq!(interval(&mut [3.0, 1.0, 4.0, 2.0], 0.1), [1.0, 4.0]);
        assert_eq!(directional(Direction::LowerIsBetter, 10.0, 8.0), 2.0);
        assert_eq!(mean(&[0.75; 22]), 0.75);
        assert_eq!(mean(&[1e12, 0.125, -1e12]), 0.125 / 3.0);
    }
}
