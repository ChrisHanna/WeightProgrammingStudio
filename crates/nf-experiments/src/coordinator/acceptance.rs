//! Immutable contract/panel-bound evidence audit, not a promotion capability.
//!
//! Supplied statistical observations are not authenticated study execution.
//! This diagnostic API remains incomplete even for passing supplied analyses.
//! `Ledger::run_fixture_final` is the separate native trusted-evidence path;
//! no DTO can waive its actual dispatch/execution/control requirements.
use super::*;
use crate::statistics::{Analysis, Plan};
use nf_core::study::{MetricDomain, ModelIdentity, ObjectiveKind, StudyContract};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Passed,
    Failed,
    Inconclusive,
    Missing,
    Incomplete,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Check {
    pub name: String,
    pub candidate: Option<ContentDigest>,
    pub binding: Option<ContentDigest>,
    pub metric: Option<String>,
    pub required: bool,
    pub status: Status,
    pub reason: String,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Report {
    pub schema_version: u32,
    pub contract: ContentDigest,
    pub revision: u32,
    pub event_head: ContentDigest,
    pub panel: Option<ContentDigest>,
    pub plan: Option<ContentDigest>,
    pub analysis: Option<ContentDigest>,
    pub checks: Vec<Check>,
    pub status: Status,
    pub analysis_conditions_satisfied: bool,
    pub accepted: bool,
    pub promotion_authorized: bool,
    pub scope: String,
}
/// Private construction and no deserialization: reports cannot be submitted as
/// acceptance witnesses. Canonical bytes include every denial and evidence ID.
pub struct Decision {
    report: Report,
    bytes: Vec<u8>,
    id: ContentDigest,
}
impl Decision {
    pub(super) fn from_report(report: Report) -> Result<Self> {
        let bytes = canonical(&report)?;
        let id = ContentDigest::of_bytes(&bytes);
        Ok(Self { report, bytes, id })
    }
    pub(super) fn into_report(self) -> Report {
        self.report
    }
    pub fn report(&self) -> &Report {
        &self.report
    }
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn id(&self) -> &ContentDigest {
        &self.id
    }
}
/// Native calculation witnesses over supplied observations, NOT authenticated
/// executions. Neither these objects nor declared panel identities resolve model
/// artifacts, prove scorer correctness, or authorize scientific acceptance.
pub struct Evidence<'a> {
    pub plan: &'a Plan,
    pub analysis: &'a Analysis,
}
/// All model identity axes, not merely parameter count or a manifest alias.
pub fn model_identity(model: &ModelIdentity) -> Result<ContentDigest> {
    Ok(ContentDigest::of_bytes(&canonical(model)?))
}
fn digest(value: &impl Serialize) -> Result<ContentDigest> {
    Ok(ContentDigest::of_bytes(&canonical(value)?))
}
fn check(name: &str, status: Status, reason: &str) -> Check {
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
fn aggregate(checks: &[Check]) -> Status {
    // Preserve every row; this precedence is only the summary, never a filter.
    for status in [
        Status::Failed,
        Status::Missing,
        Status::Incomplete,
        Status::Inconclusive,
    ] {
        if checks.iter().any(|c| c.required && c.status == status) {
            return status;
        }
    }
    Status::Passed
}
/// Acceptance requires frozen latency and memory ceilings even when the
/// portfolio is exploratory. Study budgets are not inference measurements.
pub(super) fn has_resource_ceilings(contract: &StudyContract) -> bool {
    let mut latency = false;
    let mut memory = false;
    for binding in &contract.portfolio.bindings {
        if !binding.required || !binding.schedule.final_required {
            continue;
        }
        let Some(gym) = contract.portfolio.gyms.get(&binding.gym) else {
            return false;
        };
        for metric in &binding.metrics {
            if !metric.required
                || metric.threshold.ceiling.is_none()
                || metric.measurement.is_none()
            {
                continue;
            }
            match gym.metrics.get(&metric.metric).map(|schema| schema.domain) {
                Some(MetricDomain::Latency) => latency = true,
                Some(MetricDomain::Memory) => memory = true,
                _ => {}
            }
        }
    }
    latency && memory
}
/// Compare every required and optional check with independently reconstructed
/// expectations. A stored report cannot omit, duplicate or relabel a gate.
pub(super) fn verify_check_inventory(actual: &[Check], expected: &[Check]) -> Result<()> {
    let invalid = || {
        failure(
            ErrorCode::IntegrityMismatch,
            "native decision gate coverage/outcome differs",
        )
    };
    if actual.len() != expected.len() {
        return Err(invalid());
    }
    let mut keys = BTreeSet::new();
    for check in expected {
        if !keys.insert((
            check.name.as_str(),
            check.candidate.as_ref(),
            check.binding.as_ref(),
            check.metric.as_deref(),
        )) {
            return Err(invalid());
        }
        let matches: Vec<_> = actual
            .iter()
            .filter(|value| {
                value.name == check.name
                    && value.candidate == check.candidate
                    && value.binding == check.binding
                    && value.metric == check.metric
            })
            .collect();
        if matches.len() != 1
            || matches[0].required != check.required
            || matches[0].status != check.status
            || (check.status == Status::Passed
                && !check.reason.is_empty()
                && matches[0].reason != check.reason)
        {
            return Err(invalid());
        }
    }
    Ok(())
}
/// Attach evidence only after the coordinator verifies the actual execution.
/// A budget or a successful computation does not invent resource ceilings.
pub(super) fn mark_verified_execution(
    checks: &mut [Check],
    contract: &StudyContract,
    evidence: &ContentDigest,
) {
    for check in checks {
        if matches!(
            check.name.as_str(),
            "candidate_artifact_resolution"
                | "final_execution"
                | "source_noop_best_known_controls"
                | "resource_measurements"
                | "method_and_edit_integrity"
        ) {
            if check.name == "resource_measurements" && !has_resource_ceilings(contract) {
                check.status = Status::Missing;
                check.reason = "frozen latency and memory ceilings are required".into();
            } else {
                check.status = Status::Passed;
                check.reason = evidence.to_string();
            }
        }
    }
}
fn analysis_status(value: &serde_json::Value) -> Status {
    match value.as_str() {
        Some("passed") => Status::Passed,
        Some("failed" | "no_op_cannot_satisfy_positive_gain") => Status::Failed,
        Some("inconclusive") => Status::Inconclusive,
        _ => Status::Incomplete,
    }
}
impl Ledger {
    /// Read-only, live-fenced audit of the actual persisted contract and panel.
    /// Does not open holdout, advance stages, mutate pointers, or run a study.
    pub fn acceptance_gate(
        &self,
        lease: &Lease,
        evidence: Option<Evidence<'_>>,
    ) -> Result<Decision> {
        self.files.verify(self.limits.max_database_bytes)?;
        let tx = self.connection.unchecked_transaction().map_err(sql)?;
        let contract = check_lease(&tx, &self.store, lease, &files::moment()?)?;
        let current = state(&tx, &lease.contract)?;
        if current.stage != Stage::Verify || supervision::stopped(&tx, &lease.contract)? {
            return Err(failure(
                ErrorCode::PermissionDenied,
                "acceptance audit requires live final verification",
            ));
        }
        let panel = self.frozen_final_panel(&lease.contract)?;
        let mut checks = Vec::new();
        let mut evidence_ids = (None, None);
        if let Some((id, panel)) = &panel {
            checks.push(check(
                "frozen_panel",
                Status::Passed,
                "actual coordinator-frozen identities",
            ));
            // Opening is retained before any final metadata disclosure. A plan
            // plus observations alone cannot claim this ordering took place.
            let opened: Option<String> = tx
                .query_row(
                    "SELECT panel FROM holdout_openings WHERE contract=?1",
                    [lease.contract.as_str()],
                    |row| row.get(0),
                )
                .optional()
                .map_err(sql)?;
            checks.push(check(
                "holdout_opening",
                if opened.as_deref() == Some(id.as_str()) {
                    Status::Passed
                } else {
                    Status::Missing
                },
                "recorded opening must name this exact frozen panel",
            ));
            if let Some(e) = &evidence {
                let p = e.plan.protocol();
                evidence_ids = (Some(e.plan.id().clone()), Some(e.analysis.id().clone()));
                checks.push(check("prefrozen_analysis_plan", match &panel.analysis_plan {
                    None => Status::Missing,
                    Some(id) if id == e.plan.id() => Status::Passed,
                    Some(_) => Status::Failed,
                }, "analysis plan must be bound with finalist identities before holdout disclosure"));
                let bound = e.analysis.report()["plan"] == serde_json::json!(e.plan.id())
                    && panel.contract == lease.contract
                    && panel.revision == contract.revision
                    && panel.protocol == contract.portfolio.independent.frozen_panel_protocol
                    && p.family == contract.statistics.multiplicity_family
                    && p.estimator == contract.statistics.estimator
                    && p.minimum_groups == contract.statistics.minimum_groups
                    && p.confidence == contract.statistics.confidence
                    && p.maximum_comparisons == contract.statistics.maximum_comparisons
                    && p.seeds == contract.seeds
                    && p.seeds.len() == contract.statistics.repeats as usize
                    && digest(e.analysis.report())? == *e.analysis.id();
                checks.push(check("analysis_binding", if bound { Status::Passed } else { Status::Failed },
                    "exact frozen protocol, revision, family, estimator, support, seeds and multiplicity required"));
                let mut consumed = BTreeSet::new();
                let results = e.analysis.report()["results"].as_array().ok_or_else(|| {
                    failure(
                        ErrorCode::IntegrityMismatch,
                        "native analysis has no result panel",
                    )
                })?;
                for candidate in &panel.finalists {
                    let candidate_id = model_identity(candidate)?;
                    for b in contract
                        .portfolio
                        .bindings
                        .iter()
                        .filter(|b| b.schedule.final_required)
                    {
                        for m in &b.metrics {
                            let mut row = check(
                                "frozen_metric",
                                Status::Missing,
                                "no exact final candidate/metric contrast",
                            );
                            row.candidate = Some(candidate_id.clone());
                            row.binding = Some(b.id.clone());
                            row.metric = Some(m.metric.clone());
                            row.required = m.required;
                            let contrast = p.contrasts.iter().find(|c| {
                                c.candidate == candidate_id
                                    && c.binding == b.id
                                    && c.metric == m.metric
                                    && c.comparator == m.comparator
                            });
                            if let Some(c) = contrast {
                                consumed.insert(c.id.clone());
                                let noop = self.fixture_noop_identity(&lease.contract)?;
                                let baseline = match m.comparator {
                                    nf_core::study::Comparator::Original => Some(&contract.source),
                                    nf_core::study::Comparator::NoOp => noop.as_ref(),
                                    nf_core::study::Comparator::BestKnown => {
                                        Some(&contract.best_known)
                                    }
                                };
                                let Some(baseline) = baseline else {
                                    row.reason = "executed no-op identity is missing".into();
                                    checks.push(row);
                                    continue;
                                };
                                let tasks: BTreeSet<_> =
                                    c.tasks.iter().map(|t| (&t.item, &t.group)).collect();
                                let expected: BTreeSet<_> = b
                                    .splits
                                    .holdout
                                    .items
                                    .iter()
                                    .map(|t| (&t.item, &t.group))
                                    .collect();
                                let exact = bound
                                    && c.baseline == model_identity(baseline)?
                                    && !c.no_op_control
                                    && c.required == m.required
                                    && canonical(&c.threshold)? == canonical(&m.threshold)?
                                    && canonical(&c.schema)?
                                        == canonical(
                                            &contract.portfolio.gyms[&b.gym].metrics[&m.metric],
                                        )?
                                    && c.value_statistic == m.statistic
                                    && c.context == digest(&contract.runtime)?
                                    && c.workload == b.splits.holdout.digest()?
                                    && c.measurement_protocol == digest(&m.measurement)?
                                    && tasks == expected
                                    && c.tasks.len() == expected.len();
                                if !exact {
                                    row.status = Status::Failed;
                                    row.reason = "changed threshold/schema/statistic/runtime/split/baseline or protocol binding".into();
                                } else if let Some(result) = results
                                    .iter()
                                    .find(|r| r["contrast"] == serde_json::json!(c.id))
                                {
                                    row.status = analysis_status(&result["status"]);
                                    row.reason = result["reason"]
                                        .as_str()
                                        .unwrap_or("unresolved analysis")
                                        .into();
                                } else {
                                    row.status = Status::Incomplete;
                                    row.reason = "native analysis omitted planned contrast".into();
                                }
                            }
                            checks.push(row);
                        }
                    }
                }
                checks.push(check(
                    "contrast_coverage",
                    if consumed.len() == p.contrasts.len() {
                        Status::Passed
                    } else {
                        Status::Failed
                    },
                    "unbound or extra candidate/metric contrasts cannot change the final family",
                ));
                let panel_status = if !has_resource_ceilings(&contract)
                    || checks
                        .iter()
                        .any(|c| c.name == "frozen_metric" && c.status == Status::Missing)
                {
                    Status::Missing
                } else if checks
                    .iter()
                    .any(|c| c.name == "frozen_metric" && c.status == Status::Incomplete)
                {
                    Status::Incomplete
                } else {
                    Status::Passed
                };
                checks.push(check(
                    "full_analysis_panel",
                    panel_status,
                    "every finalist needs a complete panel with frozen latency and memory ceilings",
                ));
            } else {
                checks.push(check(
                    "analysis",
                    Status::Missing,
                    "native frozen plan and analysis witnesses absent",
                ));
                for candidate in &panel.finalists {
                    for b in contract
                        .portfolio
                        .bindings
                        .iter()
                        .filter(|b| b.schedule.final_required)
                    {
                        for m in &b.metrics {
                            let mut row = check(
                                "frozen_metric",
                                Status::Missing,
                                "native analysis witness absent",
                            );
                            row.candidate = Some(model_identity(candidate)?);
                            row.binding = Some(b.id.clone());
                            row.metric = Some(m.metric.clone());
                            row.required = m.required;
                            checks.push(row);
                        }
                    }
                }
                checks.push(check(
                    "full_analysis_panel",
                    Status::Missing,
                    "entire final analysis panel absent",
                ));
            }
        } else {
            checks.push(check(
                "frozen_panel",
                Status::Missing,
                "freeze actual candidate identities before opening holdout",
            ));
        }
        let analysis_conditions_satisfied = aggregate(&checks) == Status::Passed;
        if let Some((_, panel)) = &panel {
            for candidate in &panel.finalists {
                let mut unresolved = check(
                    "candidate_artifact_resolution",
                    Status::Incomplete,
                    "declared identity only; no verified checkpoint/manifest/parameters/buffers/graph/config/tokenizer/system-state/runtime-artifact resolution witness",
                );
                unresolved.candidate = Some(model_identity(candidate)?);
                checks.push(unresolved);
            }
        }
        // These are deliberate fail-closed integration boundaries, not booleans
        // that callers can set. T035/T036/T037 witnesses currently have no final
        // coordinator dispatch/scorer provenance and cannot establish these.
        for (name, reason) in [
            ("final_execution", "missing coordinator-bound final scorer/model/task/input/seed/repeat receipts"),
            ("source_noop_best_known_controls", "missing final matched original/best-known and apply/export/reload no-op execution receipts"),
            ("resource_measurements", "missing final runtime-bound measured resource evidence; analysis numbers are not measurements"),
            ("method_and_edit_integrity", "missing candidate-bound method/scope and actual edit integrity receipts"),
        ] {
            checks.push(check(name, Status::Incomplete, reason));
        }
        if contract.objective.kind == ObjectiveKind::CausalClaim {
            checks.push(check(
                "mechanism_confirmation",
                Status::Missing,
                "causal primary claims require independent mechanism confirmation",
            ));
        }
        let report = Report {
            schema_version: 1,
            contract: lease.contract.clone(),
            revision: contract.revision,
            event_head: current.head,
            panel: panel.map(|p| p.0),
            plan: evidence_ids.0,
            analysis: evidence_ids.1,
            status: aggregate(&checks),
            checks,
            analysis_conditions_satisfied,
            accepted: false,
            promotion_authorized: false,
            scope: "contract_bound_evidence_audit_not_authenticated_execution_or_T039_promotion"
                .into(),
        };
        let bytes = canonical(&report)?;
        let id = ContentDigest::of_bytes(&bytes);
        check_lease(&tx, &self.store, lease, &files::moment()?)?;
        ensure_live(lease)?;
        tx.commit().map_err(sql)?;
        self.files.verify(self.limits.max_database_bytes)?;
        Ok(Decision { report, bytes, id })
    }
}

#[cfg(test)]
mod tests;
