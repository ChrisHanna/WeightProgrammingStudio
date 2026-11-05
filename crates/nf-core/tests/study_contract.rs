#[path = "common/study_fixture.rs"]
mod fixture;
use fixture::{d, hash, refresh, standard};
use nf_core::error::ErrorCode;
use nf_core::schemas::{Validate, Validated};
use nf_core::study::{AssuranceProfile, ClaimScope, StudyContract};
use serde_json::{json, Value};
fn parse(value: &Value) -> nf_core::error::Result<Validated<StudyContract>> {
    Validated::parse(&serde_json::to_vec(value).unwrap())
}
fn rejected(value: &mut Value) {
    refresh(value);
    assert!(parse(value).is_err());
}
#[test]
fn study_contract_standard_five_gym_mixed_origins_and_canonical_identity() {
    let value = standard();
    let contract = parse(&value).unwrap();
    assert_eq!(
        contract.portfolio.profile,
        AssuranceProfile::StandardFiveGym
    );
    assert_eq!(
        contract.portfolio.independent.scope,
        ClaimScope::FixtureOnly
    );
    assert_eq!(contract.limits.external_spend_microusd, 0);
    assert_eq!(contract.digest().unwrap(), hash(&value));
    let bytes = serde_json::to_vec(&*contract).unwrap();
    assert_eq!(
        Validated::<StudyContract>::parse(&bytes)
            .unwrap()
            .digest()
            .unwrap(),
        contract.digest().unwrap()
    );
}
#[test]
fn study_contract_exploratory_three_gyms_is_not_standard_assurance() {
    let mut value = standard();
    value["portfolio"]["profile"] = json!("exploratory");
    value["portfolio"]["label"] = json!("limited three-gym exploration");
    let removed: Vec<_> = [2, 3]
        .iter()
        .map(|i| {
            value["portfolio"]["bindings"][i]["gym"]
                .as_str()
                .unwrap()
                .to_owned()
        })
        .collect();
    let bindings = value["portfolio"]["bindings"].as_array_mut().unwrap();
    bindings.remove(3);
    bindings.remove(2);
    for id in removed {
        value["portfolio"]["gyms"]
            .as_object_mut()
            .unwrap()
            .remove(&id);
    }
    refresh(&mut value);
    let contract = parse(&value).unwrap();
    assert_eq!(contract.portfolio.profile, AssuranceProfile::Exploratory);
    value["portfolio"]["profile"] = json!("standard_five_gym");
    rejected(&mut value);
}
#[test]
fn study_contract_missing_nonfinite_compensated_or_unknown_metric_gates_fail() {
    for field in ["floor", "ceiling", "max_regression", "min_gain"] {
        let mut value = standard();
        value["portfolio"]["bindings"][0]["metrics"][0]["threshold"][field] = Value::Null;
        if field == "min_gain" {
            rejected(&mut value);
        } else {
            assert!(parse(&value).is_ok());
        }
    }
    let mut value = standard();
    value["portfolio"]["bindings"][1]["metrics"][0]["threshold"]["max_regression"] = Value::Null;
    rejected(&mut value);
    let mut value = standard();
    value["portfolio"]["bindings"][1]["metrics"][0]["priority"] = json!(0.0);
    value["portfolio"]["bindings"][0]["metrics"][0]["priority"] = json!(1_000_000.0);
    value["portfolio"]["bindings"][1]["metrics"][0]["threshold"]["max_regression"] = Value::Null;
    rejected(&mut value);
    let mut contract = (*parse(&standard()).unwrap()).clone();
    contract.portfolio.bindings[0].metrics[0].priority = f64::INFINITY;
    assert!(contract.validate().is_err());
    let mut value = standard();
    value["portfolio"]["bindings"][0]["metrics"][0]["metric"] = json!("unknown");
    rejected(&mut value);
}
#[test]
fn study_contract_split_roles_groups_exposure_and_feedback_cannot_be_laundered() {
    for role in ["edit", "sealed", "evaluation"] {
        let mut value = standard();
        value["portfolio"]["bindings"][0]["splits"]["holdout"]["role"] = json!(role);
        rejected(&mut value);
    }
    for state in ["unknown", "known_exposed"] {
        let mut value = standard();
        value["portfolio"]["bindings"][0]["splits"]["holdout"]["exposure"] = json!(state);
        rejected(&mut value);
    }
    let mut value = standard();
    let group =
        value["portfolio"]["bindings"][0]["splits"]["development"]["items"][0]["group"].clone();
    value["portfolio"]["bindings"][4]["splits"]["holdout"]["items"][0]["group"] = group;
    rejected(&mut value);
    let mut value = standard();
    value["portfolio"]["independent"]["close_revision_after_feedback"] = json!(false);
    rejected(&mut value);
    let mut value = standard();
    value["portfolio"]["independent"]["scope"] = json!("held_out_generalization");
    rejected(&mut value);
}
#[test]
fn study_contract_zero_unlimited_and_incompatible_runtime_limits_are_rejected() {
    for field in [
        "wall_ms",
        "cpu_ms",
        "candidate_attempts",
        "model_evaluations",
        "host_memory_bytes",
        "storage_bytes",
        "concurrent_workers",
    ] {
        let mut value = standard();
        value["limits"][field] = json!(0);
        assert!(parse(&value).is_err(), "{field}");
    }
    for field in [
        "wall_ms",
        "cpu_ms",
        "model_evaluations",
        "host_memory_bytes",
        "storage_bytes",
        "external_spend_microusd",
    ] {
        let mut value = standard();
        value["limits"][field] = json!(u64::MAX);
        assert!(parse(&value).is_err(), "{field}");
    }
    let mut value = standard();
    value["runtime"]["synchronization"] = json!("cuda_synchronize");
    assert!(parse(&value).is_err());
    let mut value = standard();
    value["runtime"]["device"] = json!("cuda");
    assert!(parse(&value).is_err());
    let mut value = standard();
    value["limits"]["model_evaluations"] = json!(1);
    assert!(parse(&value).is_err());
    let mut value = standard();
    value["statistics"]["minimum_groups"] = json!(2);
    assert!(parse(&value).is_err());
}
#[test]
fn study_contract_controls_versions_roles_and_missing_critical_fields_are_strict() {
    for field in [
        "original_required",
        "noop_required",
        "best_known_required",
        "matched_tasks_and_seeds",
        "noop_apply_export_reload",
        "noop_exact_parameters_and_parity",
        "positive_gain_required",
    ] {
        let mut value = standard();
        value["controls"][field] = json!(false);
        assert!(parse(&value).is_err());
    }
    let mut value = standard();
    value["baselines"].as_array_mut().unwrap().remove(1);
    assert!(parse(&value).is_err());
    let mut value = standard();
    value["baselines"][1]["input"]["parameters"] = json!(d("stale-noop"));
    assert!(parse(&value).is_err());
    let mut value = standard();
    value["portfolio"]["bindings"][0]["roles"] = json!(["accuracy"]);
    rejected(&mut value);
    for field in [
        "limits",
        "statistics",
        "runtime",
        "approval_receipt",
        "export_parity_protocol",
    ] {
        let mut value = standard();
        value.as_object_mut().unwrap().remove(field);
        assert!(parse(&value).is_err());
    }
    let mut value = standard();
    value["schema_version"] = json!(2);
    assert_eq!(
        parse(&value).unwrap_err().code,
        ErrorCode::UnsupportedVersion
    );
    let mut value = standard();
    value["ignore_hard_limits"] = json!(true);
    assert_eq!(parse(&value).unwrap_err().code, ErrorCode::UnknownField);
}
#[test]
fn study_contract_exact_gym_validation_and_portfolio_hashes_are_not_names() {
    let mut value = standard();
    let key = value["portfolio"]["bindings"][0]["gym"]
        .as_str()
        .unwrap()
        .to_owned();
    value["portfolio"]["gyms"][&key]["scorer"] = json!(d("other-scorer"));
    refresh(&mut value);
    assert_eq!(
        parse(&value).unwrap_err().code,
        ErrorCode::IntegrityMismatch
    );
    let mut value = standard();
    value["portfolio"]["bindings"][0]["validation"] = json!(d("different-validation"));
    rejected(&mut value);
    let mut value = standard();
    value["portfolio_digest"] = json!(d("not-the-portfolio"));
    assert_eq!(
        parse(&value).unwrap_err().code,
        ErrorCode::IntegrityMismatch
    );
    let mut value = standard();
    let duplicate = value["portfolio"]["bindings"][0].clone();
    value["portfolio"]["bindings"]
        .as_array_mut()
        .unwrap()
        .push(duplicate);
    rejected(&mut value);
}
#[test]
fn study_contract_latency_memory_statistic_and_runtime_are_frozen_not_proxies() {
    let mut value = standard();
    value["portfolio"]["bindings"][3]["metrics"]
        .as_array_mut()
        .unwrap()
        .remove(1);
    rejected(&mut value);
    let mut value = standard();
    value["portfolio"]["bindings"][3]["metrics"][0]["measurement"] = Value::Null;
    rejected(&mut value);
    let mut value = standard();
    value["portfolio"]["bindings"][3]["metrics"][0]["measurement"]["scope"] = json!("process_rss");
    rejected(&mut value);
    let mut value = standard();
    value["portfolio"]["bindings"][3]["metrics"][1]["measurement"]["statistic"] = json!("median");
    rejected(&mut value);
    let mut value = standard();
    value["portfolio"]["bindings"][3]["metrics"][1]["measurement"]["repetitions"] = json!(1);
    rejected(&mut value);
}
#[test]
fn study_contract_efficiency_gain_does_not_require_positive_accuracy_gain() {
    let mut value = standard();
    value["objective"]["kind"] = json!("efficiency_gain");
    value["objective"]["primary_binding"] = json!(d("binding3"));
    value["objective"]["primary_metric"] = json!("memory");
    value["portfolio"]["bindings"][3]["roles"] = json!(["target", "efficiency"]);
    value["portfolio"]["bindings"][3]["metrics"][1]["threshold"]["min_gain"] = json!(16.0);
    value["portfolio"]["bindings"][0]["metrics"][0]["threshold"]["min_gain"] = Value::Null;
    value["portfolio"]["bindings"][0]["metrics"][0]["threshold"]["max_regression"] = json!(0.0);
    refresh(&mut value);
    assert!(parse(&value).is_ok());
    value["portfolio"]["bindings"][3]["metrics"][1]["threshold"]["min_gain"] = json!(0.0);
    rejected(&mut value);
}
#[test]
fn study_contract_empirical_heldout_protocol_can_be_declared_without_claiming_execution() {
    let mut value = standard();
    value["portfolio"]["independent"]["scope"] = json!("held_out_generalization");
    value["portfolio"]["independent"]["isolation"] = json!("separate_process");
    value["statistics"]["estimator"] = json!("paired_group_bootstrap");
    value["statistics"]["minimum_groups"] = json!(2);
    value["statistics"]["confidence"] = json!(0.95);
    for (i, binding) in value["portfolio"]["bindings"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .enumerate()
    {
        for role in ["development", "selection", "holdout"] {
            let mut item = binding["splits"][role]["items"][0].clone();
            item["item"] = json!(d(&format!("second-item-{i}-{role}")));
            item["group"] = json!(d(&format!("second-group-{i}-{role}")));
            binding["splits"][role]["items"]
                .as_array_mut()
                .unwrap()
                .push(item);
        }
        binding["schedule"]["selection_episodes"] = json!(2);
        binding["schedule"]["holdout_episodes"] = json!(2);
    }
    refresh(&mut value);
    assert!(parse(&value).is_ok());
    value["portfolio"]["bindings"][0]["schedule"]["holdout_episodes"] = json!(1);
    rejected(&mut value);
}

#[test]
fn study_contract_optional_training_sources_and_generated_gyms_stay_explicit() {
    let mut value = standard();
    let mut arm = value["baselines"][0].clone();
    arm["kind"] = json!("lora");
    arm["required"] = json!(false);
    arm["enabled"] = json!(false);
    value["baselines"].as_array_mut().unwrap().push(arm);
    assert!(parse(&value).is_ok());
    value["baselines"][3]["enabled"] = json!(true);
    assert_eq!(parse(&value).unwrap_err().code, ErrorCode::PermissionDenied);
    let mut value = standard();
    value["sources"] = json!([{"kind":"teacher","artifact":d("teacher"),"destination":null,"payloads":["summaries"],"access":"development"}]);
    assert!(parse(&value).is_err());
    value["limits"]["teacher_tokens"] = json!(100);
    assert!(parse(&value).is_ok());
    value["sources"][0]["access"] = json!("holdout");
    assert_eq!(parse(&value).unwrap_err().code, ErrorCode::PermissionDenied);
    let mut value = standard();
    let id = value["portfolio"]["bindings"][0]["gym"]
        .as_str()
        .unwrap()
        .to_owned();
    value["portfolio"]["gyms"][&id]["origin"] = json!("generated");
    fixture::rebind_gym(&mut value, 0);
    assert!(parse(&value).is_err());
}

#[test]
fn study_contract_role_laundering_and_multiplicity_understatement_fail() {
    let mut value = standard();
    value["portfolio"]["bindings"][3]["roles"] = json!(["retention", "efficiency"]);
    rejected(&mut value);
    let mut value = standard();
    value["portfolio"]["bindings"][2]["roles"] = json!(["safety"]);
    rejected(&mut value);
    let mut value = standard();
    value["statistics"]["maximum_finalists"] = json!(10);
    value["statistics"]["maximum_comparisons"] = json!(6);
    assert!(parse(&value).is_err());
    let mut value = standard();
    value["statistics"]["maximum_finalists"] = json!(11);
    assert!(parse(&value).is_err());
}

#[test]
fn study_contract_revisions_and_changed_inputs_never_reuse_the_same_digest() {
    let original = standard();
    let id = parse(&original).unwrap().digest().unwrap();
    for field in ["seeds", "limits", "runtime", "portfolio"] {
        let mut value = original.clone();
        match field {
            "seeds" => value["seeds"] = json!([7, 8]),
            "limits" => value["limits"]["wall_ms"] = json!(30000),
            "runtime" => value["runtime"]["cache"] = json!("warm"),
            _ => {
                value["portfolio"]["bindings"][0]["metrics"][0]["priority"] = json!(2.0);
                refresh(&mut value);
            }
        }
        assert_ne!(parse(&value).unwrap().digest().unwrap(), id);
    }
    let mut value = original;
    value["revision"] = json!(2);
    assert!(parse(&value).is_err());
    value["parent_contract"] = json!(id);
    assert!(parse(&value).is_ok());
}
