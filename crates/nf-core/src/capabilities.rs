//! Declared software scope, not runtime qualification or model-quality evidence.
//! M1 metadata is retained verbatim; M2 is a separate, conservative inventory.
use serde_json::{json, Value};
pub fn m1_manifest() -> Value {
    json!({
        "schema_version":1,"kind":"core_capability","core_version":crate::VERSION,"milestone":"M1",
        "qualified_platforms":["linux-x86_64"],"qualified_on_this_host":cfg!(all(target_os="linux",target_arch="x86_64")),
        "backend":{"id":"nf.native_cpu","version":"1","dtypes":["f32","f64"],"device":"cpu","format":"dense_safetensors"},
        "model_factories":[{"id":"fixture.cpu","version":"1","smoke_adapter":"nf.linear_fixture.v1"},{"id":"nf.arithmetic.relu","version":"1","smoke_adapter":"nf.arithmetic_relu.v1","dtypes":["f64"],"scope":"synthetic_integer_pairs_minus8_to8"}],
        "operations":["replace_slice","add_delta","scale","mask","permute_axis","linear_map","subspace_projection","low_rank_update"],
        "checkpoint_inspection":true,"bounded_sharded_io":true,"isolated_transactions":true,"exact_single_parent_replay":true,
        "standalone_dense_export":true,"immutable_core_restore":true,"python_control_boundary":true,
        "unavailable":["construct_recipe","multi_parent_reconciliation","structural_edit","quantized_edit","gpu_execution","generated_gym_execution","study_execution","scientific_acceptance"],
        "evidence_scope":"software_fixture_tests_only","promotion_status":"not_evaluated","network_required_for_example":false
    })
}

/// Static inventory only. Run `scripts/check_m2_release.py` for measured checks.
/// OS/architecture, installed runners, and component tests do not qualify M2.
pub fn m2_manifest() -> Value {
    json!({
        "schema_version":1,"kind":"m2_capability_inventory","core_version":crate::VERSION,
        "milestone":"M2","evidence_scope":"source_inventory_not_execution_evidence",
        "states":{
            "implemented":"component_subset",
            "host_qualified":"not_evaluated",
            "workflow_integrated":"incomplete",
            "release_enabled":false,
            "scientific_evidence":"not_established"
        },
        "components":[
            "custom_python_gym_intake","sandbox_isolation_protocol","frozen_contracts",
            "three_way_split_broker","paired_controls","independent_evaluator_protocol",
            "standalone_reload_component","committed_json_markdown_reports",
            "operational_fixture_promotion_with_live_grant","operator_scoped_native_manual_dispatch"
        ],
        "timing_policy":{
            "supported_scope":"operational_cancellation",
            "m2_requires":"finite_budgets_cancellation_incomplete_and_cleanup_disclosure",
            "strict_physical_publication_required_by_m2":false,
            "strict_physical_publication":"unsupported_refuse_if_requested",
            "frozen_contract_scope":"never_silently_reinterpreted"
        },
        "measurement_policy":"checker_runs_existing_m1_and_m2_paths_no_historical_receipts",
        "scope_limits":{
            "native_fixture_only":true,"strict_fsync_publication":"unsupported",
            "independent_engineering_review":"not_established",
            "mixed_five_gym_dispatch":"not_established"
        },
        "release_blockers":[
            "T038_bounded_work_cancellation_cleanup_and_independent_review",
            "T039_authoritative_atomic_promotion_and_reviewed_release",
            "T041_integrated_authorized_custom_gym_dispatch",
            "T043_integrated_mixed_five_gym_evidence",
            "fresh_installed_wheel_and_matching_native_build",
            "actual_host_sandbox_and_trusted_worker_qualification"
        ],
        "incomplete_advanced_capabilities":[
            "nemo_gym_adapter","prime_intellect_verifiers_adapter","gymnasium_adapter",
            "autonomous_research_agents","rich_evolution_engine","general_transfer",
            "structural_edits","quantized_edits","confidential_deployment",
            "gpu_qualification","serving_qualification"
        ],
        "required_compliance_gates":[
            "CG-01","CG-02","CG-03","CG-04","CG-05","CG-06","CG-07","CG-08",
            "CG-11","CG-12","CG-14"
        ],
        "qualification_policy":"missing_failed_skipped_stale_or_component_only_is_not_release",
        "checker":"scripts/check_m2_release.py",
        "protocol":"protocols/m2-capabilities-v1.md",
        "scientific_scope":"synthetic_known_target_tests_do_not_establish_discovery_or_generalization"
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capabilities_m1_metadata_remains_m1() {
        let manifest = m1_manifest();
        assert_eq!(manifest["milestone"], "M1");
        assert_eq!(manifest["promotion_status"], "not_evaluated");
        assert_eq!(manifest["evidence_scope"], "software_fixture_tests_only");
        let unavailable = manifest["unavailable"].as_array().unwrap();
        assert!(unavailable.contains(&json!("study_execution")));
        assert!(unavailable.contains(&json!("scientific_acceptance")));
        assert_eq!(manifest["backend"]["id"], "nf.native_cpu");
    }

    #[test]
    fn capabilities_m2_inventory_never_qualifies_by_host_or_runner_presence() {
        let manifest = m2_manifest();
        assert_eq!(manifest["schema_version"], 1);
        assert_eq!(manifest["states"]["implemented"], "component_subset");
        assert_eq!(manifest["states"]["host_qualified"], "not_evaluated");
        assert_eq!(manifest["states"]["workflow_integrated"], "incomplete");
        assert_eq!(manifest["states"]["release_enabled"], false);
        assert_eq!(manifest["states"]["scientific_evidence"], "not_established");
        assert_eq!(
            manifest["timing_policy"]["strict_physical_publication_required_by_m2"],
            false
        );
        assert_eq!(
            manifest["timing_policy"]["strict_physical_publication"],
            "unsupported_refuse_if_requested"
        );
        assert_eq!(
            manifest["timing_policy"]["supported_scope"],
            "operational_cancellation"
        );
        assert_eq!(
            manifest["required_compliance_gates"]
                .as_array()
                .unwrap()
                .len(),
            11
        );
        assert_eq!(manifest["release_blockers"].as_array().unwrap().len(), 6);
        assert!(manifest["incomplete_advanced_capabilities"]
            .as_array()
            .unwrap()
            .contains(&json!("gymnasium_adapter")));
    }
}
