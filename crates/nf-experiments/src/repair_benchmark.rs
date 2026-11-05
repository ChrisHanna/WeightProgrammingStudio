//! WeightRepairBench v1 component records. Validation is not execution authority.
use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const METHODS: &[&str] = &[
    "original",
    "no_op",
    "best_known",
    "direct",
    "decoding",
    "activation",
    "lora",
    "sft",
    "distillation",
    "rl",
];
pub const COSTS: &[(&str, &str)] = &[
    ("wall", "seconds"),
    ("cpu", "seconds"),
    ("gpu", "seconds"),
    ("host_peak", "bytes"),
    ("device_peak", "bytes"),
    ("bytes_read", "bytes"),
    ("bytes_written", "bytes"),
    ("bytes_stored", "bytes"),
    ("local_tokens", "tokens"),
    ("remote_tokens", "tokens"),
    ("quoted_spend", "USD"),
    ("reserved_spend", "USD"),
    ("actual_spend", "USD"),
    ("forwards", "count"),
    ("backwards", "count"),
    ("optimizer_steps", "count"),
    ("fitting_steps", "count"),
    ("inspected_parameters", "count"),
    ("changed_parameters", "count"),
    ("trainable_parameters", "count"),
    ("proposal_attempts", "count"),
];
pub const PHASES: &[&str] = &[
    "setup",
    "diagnosis",
    "failed_edits",
    "search",
    "controls",
    "verification",
    "export",
    "reload",
    "recovery",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Failure {
    pub kind: String,
    pub provenance: ContentDigest,
    pub ground_truth: Option<ContentDigest>,
    pub solution_exposure: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Task {
    pub schema_version: u32,
    pub protocol: String,
    pub task_version: String,
    pub source: ContentDigest,
    pub model: ContentDigest,
    pub gyms: BTreeMap<String, ContentDigest>,
    pub splits: BTreeMap<String, ContentDigest>,
    pub failure: Failure,
    pub programs: BTreeMap<String, ContentDigest>,
    pub permissions: Vec<String>,
    pub budget: BTreeMap<String, f64>,
    pub runtime: ContentDigest,
    pub seed: u64,
    pub comparisons: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Quantity {
    pub status: String,
    pub value: Option<f64>,
    pub unit: String,
    pub reason: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cost {
    pub operation_id: String,
    pub scope: String,
    pub phases: Vec<String>,
    pub evidence: ContentDigest,
    pub quantities: BTreeMap<String, Quantity>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Score {
    pub correct: u32,
    pub count: u32,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Attempt {
    pub id: String,
    pub method: String,
    pub status: String,
    pub reason: String,
    pub task: ContentDigest,
    pub source: ContentDigest,
    pub candidate: Option<ContentDigest>,
    pub program: Option<ContentDigest>,
    pub scores: BTreeMap<String, Score>,
    pub behavioral: String,
    pub localization: String,
    pub mechanism: String,
    pub gradient_inspection: bool,
    pub gradient_update: bool,
    pub evidence: BTreeMap<String, ContentDigest>,
    pub costs: Vec<Cost>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResultRecord {
    pub schema_version: u32,
    pub protocol: String,
    pub task: ContentDigest,
    pub scope: String,
    pub evaluation_split: String,
    pub attempts: Vec<Attempt>,
    pub failed_direct_attempts: usize,
    pub shared_costs: Vec<Cost>,
    pub phase_coverage: BTreeMap<String, String>,
    pub efficiency_claim: String,
    pub budget_verdict: String,
    pub workflow_linkage: String,
}

fn require(ok: bool, message: &str) -> Result<(), String> {
    if ok {
        Ok(())
    } else {
        Err(message.to_owned())
    }
}
fn exact_keys<'a>(keys: impl Iterator<Item = &'a String>, expected: &[&str]) -> bool {
    keys.map(String::as_str).collect::<BTreeSet<_>>() == expected.iter().copied().collect()
}
fn parse<T: for<'a> Deserialize<'a>>(bytes: &[u8]) -> Result<T, String> {
    // Reject duplicate keys, nonfinite values, oversized and deeply nested inputs first.
    let doc = CanonicalDocument::parse(bytes, Limits::default()).map_err(|e| e.to_string())?;
    serde_json::from_slice(doc.bytes()).map_err(|e| e.to_string())
}

impl Task {
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let task: Self = parse(bytes)?;
        require(
            task.schema_version == 1 && task.protocol == "weight-repair-benchmark-v1",
            "version",
        )?;
        require(!task.task_version.is_empty(), "task version")?;
        require(
            matches!(
                task.failure.kind.as_str(),
                "injected_known" | "natural" | "adversarial"
            ),
            "failure stratum",
        )?;
        require(
            !task.failure.solution_exposure.is_empty(),
            "solution exposure disclosure",
        )?;
        require(
            task.failure.kind != "injected_known" || task.failure.ground_truth.is_some(),
            "injected ground truth",
        )?;
        require(
            task.failure.kind != "natural" || task.failure.ground_truth.is_none(),
            "natural failure cannot claim planted ground truth",
        )?;
        require(
            exact_keys(
                task.gyms.keys(),
                &[
                    "sum",
                    "difference",
                    "maximum",
                    "minimum",
                    "absolute_difference",
                ],
            ),
            "five fixture gyms",
        )?;
        require(
            exact_keys(task.splits.keys(), &["development", "selection", "holdout"]),
            "three splits",
        )?;
        require(
            task.splits.values().collect::<BTreeSet<_>>().len() == 3,
            "distinct split identities",
        )?;
        require(
            exact_keys(task.programs.keys(), &["no_op", "direct", "direct_failed"]),
            "frozen programs including failed direct",
        )?;
        require(
            task.programs.values().collect::<BTreeSet<_>>().len() == 3,
            "distinct no-op/direct/failed-direct compiled programs",
        )?;
        require(
            task.comparisons.len() == METHODS.len() && exact_keys(task.comparisons.iter(), METHODS),
            "complete frozen comparison panel",
        )?;
        require(
            task.permissions == ["inspect", "direct_edit"],
            "v1 component permissions; no training or network",
        )?;
        require(
            exact_keys(
                task.budget.keys(),
                &[
                    "max_cases",
                    "max_attempts",
                    "max_forwards",
                    "max_wall_seconds",
                ],
            ),
            "finite budget keys",
        )?;
        require(
            task.budget.values().all(|v| v.is_finite() && *v > 0.0),
            "positive finite budgets",
        )?;
        require(
            ["max_cases", "max_attempts", "max_forwards"]
                .iter()
                .all(|k| task.budget[*k].fract() == 0.0),
            "integer count budgets",
        )?;
        require(
            task.budget["max_cases"] <= 289.0
                && task.budget["max_attempts"] <= 16.0
                && task.budget["max_forwards"] <= 10_000.0
                && task.budget["max_wall_seconds"] <= 60.0,
            "bounded fixture limits",
        )?;
        Ok(task)
    }
}

fn check_cost(cost: &Cost, scope: &str, ids: &mut BTreeSet<String>) -> Result<(), String> {
    require(
        !cost.operation_id.is_empty() && ids.insert(cost.operation_id.clone()),
        "duplicate cost operation / shared double charge",
    )?;
    require(
        cost.scope == scope
            && !cost.phases.is_empty()
            && cost.phases.iter().all(|p| PHASES.contains(&p.as_str())),
        "cost scope/phases",
    )?;
    require(
        cost.quantities.len() == COSTS.len(),
        "complete cost quantities",
    )?;
    for (name, unit) in COSTS {
        let q = cost
            .quantities
            .get(*name)
            .ok_or_else(|| format!("missing cost {name}"))?;
        require(
            q.unit == *unit && !q.reason.is_empty(),
            "units and measurement provenance",
        )?;
        match (q.status.as_str(), q.value) {
            ("known", Some(v)) => require(
                v.is_finite()
                    && v >= 0.0
                    && (*unit != "count" && *unit != "tokens" && *unit != "bytes"
                        || v.fract() == 0.0),
                "nonnegative finite quantity",
            )?,
            ("unknown" | "unavailable", None) => (),
            _ => return Err("unknown/unavailable must be null; known must have value".to_owned()),
        }
    }
    Ok(())
}

impl ResultRecord {
    pub fn parse(task_bytes: &[u8], result_bytes: &[u8]) -> Result<Self, String> {
        let task = Task::parse(task_bytes)?;
        let doc =
            CanonicalDocument::parse(task_bytes, Limits::default()).map_err(|e| e.to_string())?;
        let task_id = ContentDigest::of_bytes(doc.bytes());
        let result: Self = parse(result_bytes)?;
        require(
            result.schema_version == 1
                && result.protocol == task.protocol
                && result.task == task_id,
            "result version/task binding",
        )?;
        require(
            result.scope == "synthetic_component_fixture_not_scientific_acceptance"
                && result.evaluation_split == "development",
            "component scope; no holdout authority",
        )?;
        require(
            result.efficiency_claim == "not_ranked"
                && result.budget_verdict == "incomplete"
                && result.workflow_linkage == "unavailable_pending_authorized_workflow",
            "no efficiency/acceptance/complete-budget claim",
        )?;
        require(
            task.failure.kind == "injected_known",
            "this fixture cannot relabel injected failure as natural/adversarial",
        )?;
        require(
            exact_keys(result.phase_coverage.keys(), PHASES)
                && result
                    .phase_coverage
                    .values()
                    .all(|s| matches!(s.as_str(), "measured_or_unknown" | "not_run")),
            "phase coverage",
        )?;
        let mut ids = BTreeSet::new();
        let mut operations = BTreeSet::new();
        let mut methods = BTreeSet::new();
        let mut failed_direct = false;
        let mut forwards = 0.0;
        let mut wall = 0.0;
        for cost in &result.shared_costs {
            check_cost(cost, "shared", &mut operations)?;
        }
        require(
            !result.shared_costs.is_empty(),
            "setup and shared accounting required",
        )?;
        for arm in &result.attempts {
            require(
                !arm.id.is_empty() && ids.insert(arm.id.clone()),
                "unique attempt id",
            )?;
            require(
                arm.id == arm.method || arm.id == "direct_failed" && arm.method == "direct",
                "fixed attempt identities; cannot replace direct by fallback",
            )?;
            require(METHODS.contains(&arm.method.as_str()), "method kind")?;
            methods.insert(arm.method.clone());
            require(
                arm.task == task_id && arm.source == task.source,
                "matched task/source",
            )?;
            require(!arm.reason.is_empty(), "attempt disclosure")?;
            require(
                !arm.gradient_inspection && !arm.gradient_update,
                "no gradient authorization",
            )?;
            require(
                matches!(arm.localization.as_str(), "not_evaluated" | "unsupported")
                    && matches!(arm.mechanism.as_str(), "not_evaluated" | "unsupported"),
                "no localization/mechanism proof in fixture",
            )?;
            require(
                match arm.method.as_str() {
                    "decoding" | "activation" => arm.status == "unsupported",
                    "lora" | "sft" | "distillation" | "rl" => arm.status == "not_run",
                    _ => matches!(arm.status.as_str(), "completed" | "failed"),
                },
                "method lifecycle; unsupported runtime slots and unrun training",
            )?;
            match arm.status.as_str() {
                "not_run" | "unsupported" => {
                    require(
                        arm.scores.is_empty()
                            && arm.costs.is_empty()
                            && arm.evidence.is_empty()
                            && arm.candidate.is_none()
                            && arm.program.is_none()
                            && arm.behavioral == "not_evaluated",
                        "unrun arms have no measurements or zero-cost inventions",
                    )?;
                }
                "completed" | "failed" => {
                    require(
                        matches!(
                            arm.method.as_str(),
                            "original" | "no_op" | "best_known" | "direct"
                        ),
                        "unsupported fixture execution method",
                    )?;
                    require(
                        arm.candidate.is_some() && !arm.costs.is_empty(),
                        "executed arm evidence/cost",
                    )?;
                    require(
                        exact_keys(
                            arm.scores.keys(),
                            &[
                                "sum",
                                "difference",
                                "maximum",
                                "minimum",
                                "absolute_difference",
                            ],
                        ),
                        "complete retention panel",
                    )?;
                    let counts: BTreeSet<_> = arm.scores.values().map(|s| s.count).collect();
                    require(
                        counts.len() == 1
                            && arm.scores.values().all(|s| {
                                s.count > 0
                                    && s.count as f64 <= task.budget["max_cases"]
                                    && s.correct <= s.count
                            }),
                        "matched nonempty case counts",
                    )?;
                    let passed = arm.scores.values().all(|s| s.correct == s.count);
                    require(
                        arm.behavioral == (if passed { "passed" } else { "failed" })
                            && arm.status == (if passed { "completed" } else { "failed" }),
                        "behavior/status derived from complete panel",
                    )?;
                    for key in ["observations", "source_integrity"] {
                        require(
                            arm.evidence.contains_key(key),
                            "observation/source evidence",
                        )?;
                    }
                    if matches!(arm.method.as_str(), "no_op" | "direct") {
                        let key = if arm.id == "direct_failed" {
                            "direct_failed"
                        } else {
                            arm.method.as_str()
                        };
                        require(
                            arm.program.as_ref() == task.programs.get(key)
                                && arm.evidence.contains_key("apply_receipt"),
                            "compiled program/apply evidence",
                        )?;
                    } else {
                        require(arm.program.is_none(), "original/reference are not edits")?;
                    }
                    if matches!(arm.method.as_str(), "original" | "no_op") {
                        require(
                            arm.candidate.as_ref() == Some(&task.source),
                            "exact original/no-op parameter identity",
                        )?;
                    }
                    if arm.method == "direct" {
                        require(
                            arm.candidate.as_ref() != Some(&task.source),
                            "direct comparison must change actual parameters",
                        )?;
                    }
                    if arm.id == "direct_failed" {
                        failed_direct = arm.method == "direct" && arm.status == "failed";
                    }
                    for cost in &arm.costs {
                        check_cost(cost, &arm.id, &mut operations)?;
                        for key in [
                            "backwards",
                            "optimizer_steps",
                            "fitting_steps",
                            "trainable_parameters",
                        ] {
                            require(
                                cost.quantities[key].status == "known"
                                    && cost.quantities[key].value == Some(0.0),
                                "unapproved gradient/training execution",
                            )?;
                        }
                    }
                }
                _ => return Err("unknown attempt lifecycle".to_owned()),
            }
        }
        require(
            methods.len() == METHODS.len()
                && result.attempts.len() == METHODS.len() + 1
                && failed_direct,
            "required slots and preserved failed direct attempt",
        )?;
        let direct_candidates: BTreeSet<_> = result
            .attempts
            .iter()
            .filter(|a| a.method == "direct")
            .filter_map(|a| a.candidate.as_ref())
            .collect();
        require(
            direct_candidates.len() == 2,
            "distinct direct and failed-direct candidates",
        )?;
        require(
            result.failed_direct_attempts
                == result
                    .attempts
                    .iter()
                    .filter(|a| a.method == "direct" && a.status == "failed")
                    .count(),
            "complete failed-direct count",
        )?;
        let executed: Vec<_> = result
            .attempts
            .iter()
            .filter(|a| !a.scores.is_empty())
            .collect();
        let counts: BTreeSet<_> = executed
            .iter()
            .flat_map(|a| a.scores.values().map(|s| s.count))
            .collect();
        require(
            counts.len() == 1,
            "same case count across all executed arms",
        )?;
        let original = result
            .attempts
            .iter()
            .find(|a| a.id == "original")
            .ok_or("original missing")?;
        let noop = result
            .attempts
            .iter()
            .find(|a| a.id == "no_op")
            .ok_or("no-op missing")?;
        require(
            original.scores.iter().all(|(k, s)| {
                noop.scores
                    .get(k)
                    .is_some_and(|n| n.correct == s.correct && n.count == s.count)
            }),
            "matched original/no-op scores",
        )?;
        for method in ["original", "no_op", "best_known", "direct"] {
            require(
                result.attempts.iter().any(|a| {
                    a.id == method
                        && a.method == method
                        && matches!(a.status.as_str(), "completed" | "failed")
                }),
                "real required fixture comparison",
            )?;
        }
        require(
            result.attempts.len() as f64 <= task.budget["max_attempts"],
            "attempt budget",
        )?;
        // Shared preparation can be unknown. Never report that partial sums are totals.
        for cost in result
            .shared_costs
            .iter()
            .chain(result.attempts.iter().flat_map(|a| a.costs.iter()))
        {
            for (name, total) in [("forwards", &mut forwards), ("wall", &mut wall)] {
                if let Some(value) = cost.quantities[name].value {
                    *total += value;
                } else {
                    require(
                        cost.scope == "shared",
                        "runner budget requires measured arm forwards/wall",
                    )?;
                }
            }
        }
        require(
            forwards <= task.budget["max_forwards"] && wall <= task.budget["max_wall_seconds"],
            "measured fixture budgets",
        )?;
        let covered: BTreeSet<_> = result
            .shared_costs
            .iter()
            .chain(result.attempts.iter().flat_map(|a| a.costs.iter()))
            .flat_map(|c| c.phases.iter().map(String::as_str))
            .collect();
        require(
            result
                .phase_coverage
                .iter()
                .all(|(p, s)| covered.contains(p.as_str()) == (s == "measured_or_unknown")),
            "phase coverage must agree with cost operations",
        )?;
        Ok(result)
    }
}
