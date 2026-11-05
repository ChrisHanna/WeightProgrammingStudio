use super::*;
use crate::statistics::{self, Arm, Data, Observation, Protocol};
use nf_core::study::{MetricUnit, Statistic};
use nf_surgeon::cancellation::CancellationToken;
#[path = "../../../../nf-core/tests/common/study_fixture.rs"]
#[allow(dead_code)]
mod fixture;
use fixture::{d, standard};
use std::sync::atomic::{AtomicU64, Ordering};
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Root(std::path::PathBuf);
impl Drop for Root {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}
fn bytes(value: &impl Serialize) -> Vec<u8> {
    canonical(value).unwrap()
}
// Deliberately supplied numbers/identities, not a model execution fixture. The
// tests prove that these can exercise analysis but can NEVER grant acceptance.
fn setup(change: impl FnOnce(&mut Protocol)) -> (Root, Ledger, Lease, Plan, Data) {
    setup_contract(|_| {}, change)
}
fn setup_contract(
    change_contract: impl FnOnce(&mut serde_json::Value),
    change: impl FnOnce(&mut Protocol),
) -> (Root, Ledger, Lease, Plan, Data) {
    setup_contract_with_holdout(change_contract, change, true)
}
fn setup_contract_with_holdout(
    change_contract: impl FnOnce(&mut serde_json::Value),
    change: impl FnOnce(&mut Protocol),
    open_holdout: bool,
) -> (Root, Ledger, Lease, Plan, Data) {
    let mut value = standard();
    value["statistics"]["repeats"] = serde_json::json!(2);
    change_contract(&mut value);
    fixture::refresh(&mut value);
    let contract: StudyContract = serde_json::from_value(value.clone()).unwrap();
    let mut candidate = contract.source.clone();
    candidate.parameters = d("analysis-only-candidate-parameters");
    candidate.manifest = d("analysis-only-candidate-manifest");
    let candidate_id = model_identity(&candidate).unwrap();
    let mut contrasts = Vec::new();
    for b in &contract.portfolio.bindings {
        for m in &b.metrics {
            contrasts.push(statistics::Contrast {
                id: digest(&(&b.id, &m.metric)).unwrap(),
                candidate: candidate_id.clone(),
                baseline: model_identity(&contract.source).unwrap(),
                comparator: m.comparator,
                binding: b.id.clone(),
                metric: m.metric.clone(),
                schema: contract.portfolio.gyms[&b.gym].metrics[&m.metric].clone(),
                value_statistic: m.statistic,
                context: digest(&contract.runtime).unwrap(),
                workload: b.splits.holdout.digest().unwrap(),
                measurement_protocol: digest(&m.measurement).unwrap(),
                tasks: b
                    .splits
                    .holdout
                    .items
                    .iter()
                    .map(|t| statistics::Task {
                        item: t.item.clone(),
                        group: t.group.clone(),
                        input: digest(&("unexecuted-input", &t.item)).unwrap(),
                    })
                    .collect(),
                threshold: m.threshold.clone(),
                required: m.required,
                no_op_control: false,
            });
        }
    }
    let mut protocol = Protocol {
        schema_version: 1,
        family: contract.statistics.multiplicity_family.clone(),
        estimator: contract.statistics.estimator,
        scope: "deterministic_fixture_only".into(),
        estimand: "equal_weight_group_mean_of_task_seed_means".into(),
        minimum_groups: 1,
        minimum_tasks: 1,
        seeds: contract.seeds.clone(),
        confidence: 1.0,
        resamples: 0,
        bootstrap_seed: 17,
        maximum_comparisons: contract.statistics.maximum_comparisons,
        contrasts,
    };
    change(&mut protocol);
    let plan = Plan::freeze(&bytes(&protocol)).unwrap();
    let root = Root(
        std::path::PathBuf::from(std::env::var_os("KIROCREW_SCRATCH").unwrap()).join(format!(
            "acceptance-gate-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )),
    );
    std::fs::create_dir(&root.0).unwrap();
    let mut ledger = Ledger::open(&root.0.join("ledger"), LedgerLimits::default()).unwrap();
    let id = ledger
        .freeze(&Validated::<StudyContract>::parse(&bytes(&value)).unwrap())
        .unwrap();
    let lease = ledger.acquire(&id, "acceptance-tests", 60000).unwrap();
    let mut previous = Stage::Frozen;
    for (i, next) in [
        Stage::Preflight,
        Stage::Profile,
        Stage::Baseline,
        Stage::Design,
        Stage::Execute,
        Stage::Evaluate,
        Stage::Verify,
    ]
    .into_iter()
    .enumerate()
    {
        ledger
            .advance(
                &lease,
                Transition {
                    operation: d(&format!("transition{i}")),
                    expected: previous,
                    next,
                    protocol: d("fixture-workflow"),
                    evidence: d("software-transition-only"),
                    message: "software state only, not scientific acceptance".into(),
                },
            )
            .unwrap();
        previous = next;
    }
    ledger
        .freeze_final_panel_with_analysis(&lease, vec![candidate], &plan)
        .unwrap();
    if open_holdout {
        ledger
            .open_fixture_holdout(&lease, "fixture-test-metadata-access")
            .unwrap();
    }
    let mut observations = Vec::new();
    for c in &protocol.contrasts {
        for t in &c.tasks {
            for seed in &protocol.seeds {
                for arm in [Arm::Baseline, Arm::Candidate] {
                    observations.push(Observation {
                        contrast: c.id.clone(),
                        item: t.item.clone(),
                        group: t.group.clone(),
                        input: t.input.clone(),
                        seed: *seed,
                        arm,
                        model: if arm == Arm::Baseline {
                            c.baseline.clone()
                        } else {
                            c.candidate.clone()
                        },
                        context: c.context.clone(),
                        workload: c.workload.clone(),
                        measurement_protocol: c.measurement_protocol.clone(),
                        unit: c.schema.unit,
                        value_statistic: c.value_statistic,
                        status: statistics::Status::Complete,
                        value: Some(if c.metric == "latency" || c.metric == "memory" {
                            1.0
                        } else if arm == Arm::Candidate {
                            0.9
                        } else {
                            0.8
                        }),
                    });
                }
            }
        }
    }
    let data = Data {
        schema_version: 1,
        plan: plan.id().clone(),
        observations,
    };
    (root, ledger, lease, plan, data)
}
fn audit(ledger: &Ledger, lease: &Lease, plan: &Plan, data: &Data) -> Decision {
    let analysis = plan
        .analyze(&bytes(data), &CancellationToken::default())
        .unwrap();
    ledger
        .acceptance_gate(
            lease,
            Some(Evidence {
                plan,
                analysis: &analysis,
            }),
        )
        .unwrap()
}
fn row<'a>(decision: &'a Decision, metric: &str, binding: &ContentDigest) -> &'a Check {
    decision
        .report()
        .checks
        .iter()
        .find(|c| c.metric.as_deref() == Some(metric) && c.binding.as_ref() == Some(binding))
        .unwrap()
}
#[test]
fn acceptance_gate_reopened_inventory_rejects_missing_or_changed_checks() {
    let expected = vec![check(
        "resource_measurements",
        Status::Passed,
        "execution-id",
    )];
    assert!(verify_check_inventory(&expected, &expected).is_ok());
    assert!(verify_check_inventory(&[], &expected).is_err());
    let mut changed = expected.clone();
    changed[0].required = false;
    assert!(verify_check_inventory(&changed, &expected).is_err());
    changed[0].required = true;
    changed[0].reason = "different-evidence".into();
    assert!(verify_check_inventory(&changed, &expected).is_err());
    changed = expected.clone();
    changed.push(expected[0].clone());
    assert!(verify_check_inventory(&changed, &expected).is_err());
    let duplicate_expected = vec![expected[0].clone(), expected[0].clone()];
    let mixed = vec![expected[0].clone(), check("unrelated", Status::Passed, "")];
    assert!(verify_check_inventory(&mixed, &duplicate_expected).is_err());
}
#[test]
fn acceptance_gate_execution_evidence_cannot_pass_missing_resource_ceilings() {
    let mut contract: StudyContract = serde_json::from_value(standard()).unwrap();
    contract.portfolio.bindings[3].metrics.clear();
    let mut checks = vec![
        check("final_execution", Status::Incomplete, "not verified"),
        check("resource_measurements", Status::Incomplete, "not verified"),
    ];
    mark_verified_execution(&mut checks, &contract, &d("execution"));
    assert_eq!(checks[0].status, Status::Passed);
    assert_eq!(checks[1].status, Status::Missing);
}
#[test]
fn acceptance_gate_resource_ceilings_must_be_required_and_measured() {
    let mut contract: StudyContract = serde_json::from_value(standard()).unwrap();
    assert!(has_resource_ceilings(&contract));
    contract.portfolio.bindings[3].metrics[0].required = false;
    assert!(!has_resource_ceilings(&contract));
    contract.portfolio.bindings[3].metrics[0].required = true;
    contract.portfolio.bindings[3].metrics[1].threshold.ceiling = None;
    assert!(!has_resource_ceilings(&contract));
    contract.portfolio.bindings[3].metrics[1].threshold.ceiling = Some(1_000_000.0);
    contract.portfolio.bindings[3].metrics[0].measurement = None;
    assert!(!has_resource_ceilings(&contract));
}
#[test]
fn acceptance_gate_missing_resource_ceilings_cannot_satisfy_the_panel() {
    let (_root, mut ledger, lease, plan, data) = setup_contract_with_holdout(
        |contract| {
            contract["portfolio"]["profile"] = serde_json::json!("exploratory");
            let quality_gym = contract["portfolio"]["bindings"][2]["gym"]
                .as_str()
                .unwrap()
                .to_owned();
            let efficiency_gym = contract["portfolio"]["bindings"][3]["gym"]
                .as_str()
                .unwrap()
                .to_owned();
            contract["portfolio"]["gyms"][&efficiency_gym]["metrics"] =
                contract["portfolio"]["gyms"][&quality_gym]["metrics"].clone();
            contract["portfolio"]["bindings"][3]["metrics"] =
                contract["portfolio"]["bindings"][2]["metrics"].clone();
            contract["portfolio"]["bindings"][3]["roles"] = serde_json::json!(["diagnostic"]);
            fixture::rebind_gym(contract, 3);
        },
        |_| {},
        false,
    );
    let error = ledger
        .open_fixture_holdout(&lease, "fixture-test-metadata-access")
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::VerificationPolicyWeakened);
    let openings: u64 = ledger
        .connection
        .query_row(
            "SELECT count(*) FROM holdout_openings WHERE contract=?1",
            [lease.contract().as_str()],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(openings, 0);
    let result = audit(&ledger, &lease, &plan, &data);
    let resources = result
        .report()
        .checks
        .iter()
        .find(|check| check.name == "full_analysis_panel")
        .expect("full analysis panel must be reported");
    assert_eq!(resources.status, Status::Missing);
    assert!(!result.report().analysis_conditions_satisfied);
    assert!(!result.report().accepted);
}
#[test]
fn acceptance_gate_missing_noop_identity_does_not_substitute_original() {
    let (_root, ledger, lease, plan, data) = setup_contract(
        |contract| {
            contract["portfolio"]["bindings"][1]["metrics"][0]["comparator"] =
                serde_json::json!("no_op");
        },
        |_| {},
    );
    let result = audit(&ledger, &lease, &plan, &data);
    let retention = row(&result, "quality", &d("binding1"));
    assert_eq!(retention.status, Status::Missing);
    assert!(!result.report().analysis_conditions_satisfied);
    assert!(!result.report().accepted);
}
#[test]
fn acceptance_gate_noop_comparison_uses_recorded_export_identity() {
    let source: ModelIdentity = serde_json::from_value(standard()["source"].clone()).unwrap();
    let mut noop = source.clone();
    noop.checkpoint = d("separate-noop-export-checkpoint");
    noop.manifest = d("separate-noop-export-manifest");
    let noop_id = model_identity(&noop).unwrap();
    let (_root, ledger, lease, plan, data) = setup_contract(
        |contract| {
            contract["portfolio"]["bindings"][1]["metrics"][0]["comparator"] =
                serde_json::json!("no_op");
        },
        |protocol| {
            let retention = protocol
                .contrasts
                .iter_mut()
                .find(|contrast| contrast.binding == d("binding1"))
                .unwrap();
            retention.baseline = noop_id;
        },
    );
    let contract = ledger.contract(lease.contract()).unwrap();
    let controls = bytes(&serde_json::json!({
        "source": contract.source,
        "no_op": noop,
        "best_known": contract.best_known,
    }));
    let digest = ContentDigest::of_bytes(&controls);
    ledger.connection.execute(
        "INSERT INTO final_fixture_records(contract,kind,body,digest) VALUES(?1,'controls',?2,?3)",
        rusqlite::params![lease.contract().as_str(), controls, digest.as_str()],
    ).unwrap();
    let result = audit(&ledger, &lease, &plan, &data);
    assert_eq!(
        row(&result, "quality", &d("binding1")).status,
        Status::Passed
    );
    assert!(result.report().analysis_conditions_satisfied);
    // These are synthetic analysis inputs, never authenticated execution.
    assert!(!result.report().accepted);
}
#[test]
fn acceptance_gate_noop_comparison_rejects_source_identity_when_export_differs() {
    let source: ModelIdentity = serde_json::from_value(standard()["source"].clone()).unwrap();
    let mut noop = source.clone();
    noop.checkpoint = d("different-noop-export-checkpoint");
    noop.manifest = d("different-noop-export-manifest");
    let (_root, ledger, lease, plan, data) = setup_contract(
        |contract| {
            contract["portfolio"]["bindings"][1]["metrics"][0]["comparator"] =
                serde_json::json!("no_op");
        },
        |_| {},
    );
    let contract = ledger.contract(lease.contract()).unwrap();
    let controls = bytes(&serde_json::json!({
        "source": contract.source,
        "no_op": noop,
        "best_known": contract.best_known,
    }));
    let digest = ContentDigest::of_bytes(&controls);
    ledger.connection.execute(
        "INSERT INTO final_fixture_records(contract,kind,body,digest) VALUES(?1,'controls',?2,?3)",
        rusqlite::params![lease.contract().as_str(), controls, digest.as_str()],
    ).unwrap();
    let result = audit(&ledger, &lease, &plan, &data);
    assert_eq!(
        row(&result, "quality", &d("binding1")).status,
        Status::Failed
    );
    assert!(!result.report().analysis_conditions_satisfied);
    assert!(!result.report().accepted);
}
#[test]
fn acceptance_gate_compact_observations_preserve_full_analysis_and_binding() {
    let (_root, _ledger, _lease, plan, data) = setup(|_| {});
    let full = plan
        .analyze(&bytes(&data), &CancellationToken::default())
        .unwrap();
    let mut compact =
        statistics::CompactData::from_data(&plan, &data, &CancellationToken::default()).unwrap();
    let reduced = plan
        .analyze_compact(&bytes(&compact), &CancellationToken::default())
        .unwrap();
    let mut expected = full.report().clone();
    let mut actual = reduced.report().clone();
    expected.as_object_mut().unwrap().remove("data_digest");
    actual.as_object_mut().unwrap().remove("data_digest");
    assert_eq!(actual, expected);
    assert!(bytes(&compact).len() < bytes(&data).len());
    for field in 0..8 {
        let mut changed = data.clone();
        let row = &mut changed.observations[0];
        match field {
            0 => row.group = d("different-group"),
            1 => row.input = d("different-input"),
            2 => row.model = d("different-model"),
            3 => row.context = d("different-context"),
            4 => row.workload = d("different-workload"),
            5 => row.measurement_protocol = d("different-measurement"),
            6 => row.unit = MetricUnit::Count,
            _ => row.value_statistic = Statistic::Mean,
        }
        assert!(
            statistics::CompactData::from_data(&plan, &changed, &CancellationToken::default())
                .is_err(),
            "compact conversion must not discard mismatched field {field}"
        );
    }
    let mut wrong_seed = data.clone();
    wrong_seed.observations[0].seed = u64::MAX;
    assert!(
        statistics::CompactData::from_data(&plan, &wrong_seed, &CancellationToken::default())
            .is_err()
    );
    compact.observations[0].item = d("unknown-task");
    assert!(plan
        .analyze_compact(&bytes(&compact), &CancellationToken::default())
        .is_err());
}
#[test]
fn acceptance_gate_complete_supplied_numbers_never_authenticate_execution() {
    let (_root, ledger, lease, plan, data) = setup(|_| {});
    let before = ledger.state(lease.contract()).unwrap();
    let a = audit(&ledger, &lease, &plan, &data);
    let b = audit(&ledger, &lease, &plan, &data);
    assert!(a.report().analysis_conditions_satisfied);
    assert_eq!(a.report().status, Status::Incomplete);
    assert!(!a.report().accepted && !a.report().promotion_authorized);
    assert_eq!(a.bytes(), b.bytes());
    assert_eq!(a.id(), &ContentDigest::of_bytes(a.bytes()));
    assert_eq!(ledger.state(lease.contract()).unwrap().head, before.head);
}
#[test]
fn acceptance_gate_calculation_witness_and_declared_identity_do_not_resolve_artifacts() {
    let (_root, ledger, lease, plan, data) = setup(|_| {});
    let result = audit(&ledger, &lease, &plan, &data);
    let (_, panel) = ledger
        .frozen_final_panel(lease.contract())
        .unwrap()
        .unwrap();
    assert!(result.report().analysis_conditions_satisfied);
    for candidate in &panel.finalists {
        let id = model_identity(candidate).unwrap();
        assert!(result.report().checks.iter().any(|c| {
            c.name == "candidate_artifact_resolution"
                && c.candidate.as_ref() == Some(&id)
                && c.required
                && c.status == Status::Incomplete
        }));
    }
    assert!(result
        .report()
        .checks
        .iter()
        .any(|c| { c.name == "final_execution" && c.required && c.status == Status::Incomplete }));
    assert!(!result.report().accepted);
    assert!(!result.report().promotion_authorized);
}
#[test]
fn acceptance_gate_submitted_optional_contrast_cannot_waive_frozen_required_check() {
    let (_root, ledger, lease, plan, data) = setup(|p| {
        p.contrasts[0].required = false;
    });
    let result = audit(&ledger, &lease, &plan, &data);
    let target = row(&result, "quality", &d("binding0"));
    assert!(target.required);
    assert_eq!(target.status, Status::Failed);
    assert!(!result.report().accepted);
}
#[test]
fn acceptance_gate_missing_analysis_retains_every_frozen_metric() {
    let (_root, ledger, lease, _plan, _data) = setup(|_| {});
    let result = ledger.acceptance_gate(&lease, None).unwrap();
    assert_eq!(result.report().status, Status::Missing);
    assert_eq!(
        result
            .report()
            .checks
            .iter()
            .filter(|c| c.name == "frozen_metric")
            .count(),
        6
    );
    assert!(!result.report().accepted);
}
#[test]
fn acceptance_gate_missing_gym_blocks_full_panel() {
    let (_root, ledger, lease, plan, data) = setup(|p| {
        p.contrasts.retain(|c| c.binding != d("binding1"));
    });
    let result = audit(&ledger, &lease, &plan, &data);
    assert_eq!(
        row(&result, "quality", &d("binding1")).status,
        Status::Missing
    );
    assert_eq!(result.report().status, Status::Missing);
}
#[test]
fn acceptance_gate_retention_damage_cannot_be_offset_by_target_gain() {
    let (_root, ledger, lease, plan, mut data) = setup(|_| {});
    let retention = plan
        .protocol()
        .contrasts
        .iter()
        .find(|c| c.binding == d("binding1"))
        .unwrap();
    for o in &mut data.observations {
        if o.contrast == retention.id && o.arm == Arm::Candidate {
            o.value = Some(0.1);
        }
    }
    let result = audit(&ledger, &lease, &plan, &data);
    assert_eq!(
        row(&result, "quality", &d("binding0")).status,
        Status::Passed
    );
    assert_eq!(
        row(&result, "quality", &d("binding1")).status,
        Status::Failed
    );
    assert_eq!(result.report().status, Status::Failed);
}
#[test]
fn acceptance_gate_changed_threshold_is_not_a_new_acceptance_rule() {
    let (_root, ledger, lease, plan, data) = setup(|p| {
        p.contrasts[0].threshold.min_gain = Some(0.001);
    });
    let result = audit(&ledger, &lease, &plan, &data);
    assert_eq!(
        row(&result, "quality", &d("binding0")).status,
        Status::Failed
    );
}
#[test]
fn acceptance_gate_timeout_and_absent_pair_are_incomplete_not_scientific_failure() {
    let (_root, ledger, lease, plan, mut data) = setup(|_| {});
    data.observations[0].status = statistics::Status::TimedOut;
    data.observations[0].value = None;
    data.observations.pop();
    let result = audit(&ledger, &lease, &plan, &data);
    assert_eq!(
        row(&result, "quality", &d("binding0")).status,
        Status::Incomplete
    );
    assert_eq!(
        row(&result, "safety", &d("binding4")).status,
        Status::Incomplete
    );
}
#[test]
fn acceptance_gate_insufficient_support_is_inconclusive_not_missing() {
    let (_root, ledger, lease, plan, data) = setup(|p| p.minimum_tasks = 2);
    let result = audit(&ledger, &lease, &plan, &data);
    assert_eq!(
        row(&result, "quality", &d("binding0")).status,
        Status::Inconclusive
    );
    assert!(!result.report().analysis_conditions_satisfied);
}
#[test]
fn acceptance_gate_resource_scope_change_and_extra_finalist_fail_closed() {
    for change in [0, 1] {
        let (_root, ledger, lease, plan, data) = setup(|p| {
            if change == 0 {
                p.contrasts[3].measurement_protocol = d("different-memory-scope");
            } else {
                let mut extra = p.contrasts[0].clone();
                extra.id = d("extra-contrast");
                extra.candidate = d("unfrozen-finalist");
                p.contrasts.push(extra);
            }
        });
        let result = audit(&ledger, &lease, &plan, &data);
        assert_eq!(result.report().status, Status::Failed);
        assert!(!result.report().accepted);
    }
}
#[test]
fn acceptance_gate_panel_plan_is_immutable_after_holdout_and_cannot_be_substituted() {
    let (_root, mut ledger, lease, plan, mut data) = setup(|_| {});
    let (panel_id, panel) = ledger
        .frozen_final_panel(lease.contract())
        .unwrap()
        .unwrap();
    assert_eq!(
        ledger
            .freeze_final_panel_with_analysis(&lease, panel.finalists.clone(), &plan)
            .unwrap(),
        panel_id
    );
    let mut changed = plan.protocol().clone();
    changed.bootstrap_seed += 1;
    let changed = Plan::freeze(&bytes(&changed)).unwrap();
    assert_eq!(
        ledger
            .freeze_final_panel_with_analysis(&lease, panel.finalists.clone(), &changed)
            .unwrap_err()
            .code,
        ErrorCode::RevisionConflict
    );
    assert_eq!(
        ledger
            .freeze_final_panel(&lease, panel.finalists)
            .unwrap_err()
            .code,
        ErrorCode::RevisionConflict
    );
    data.plan = changed.id().clone();
    let result = audit(&ledger, &lease, &changed, &data);
    assert!(result
        .report()
        .checks
        .iter()
        .any(|c| c.name == "prefrozen_analysis_plan" && c.status == Status::Failed));
    assert!(!result.report().accepted);
}
#[test]
fn acceptance_gate_stale_lease_cannot_issue_decisions() {
    let (_root, mut ledger, lease, _plan, _data) = setup(|_| {});
    ledger.release(&lease).unwrap();
    let result = ledger.acceptance_gate(&lease, None);
    assert!(matches!(
        result,
        Err(CoreError {
            code: ErrorCode::StaleLease,
            ..
        })
    ));
}
#[test]
fn acceptance_gate_legacy_panel_serialization_does_not_invent_an_analysis_plan() {
    let (_root, ledger, lease, _plan, _data) = setup(|_| {});
    let (_, mut panel) = ledger
        .frozen_final_panel(lease.contract())
        .unwrap()
        .unwrap();
    panel.analysis_plan = None;
    let old = serde_json::to_value(&panel).unwrap();
    assert!(old.get("analysis_plan").is_none());
    let restored: splits::Panel = serde_json::from_value(old).unwrap();
    assert!(restored.analysis_plan.is_none());
}
#[test]
fn acceptance_gate_status_summary_preserves_distinct_denials() {
    assert_eq!(
        analysis_status(&serde_json::json!("inconclusive")),
        Status::Inconclusive
    );
    assert_eq!(
        analysis_status(&serde_json::json!("unknown")),
        Status::Incomplete
    );
    let mut checks = vec![check("a", Status::Inconclusive, "noise")];
    assert_eq!(aggregate(&checks), Status::Inconclusive);
    checks.push(check("b", Status::Missing, "absent"));
    assert_eq!(aggregate(&checks), Status::Missing);
    checks.push(check("c", Status::Failed, "damage"));
    assert_eq!(aggregate(&checks), Status::Failed);
    assert_eq!(checks.len(), 3);
}
