mod common;
#[path = "common/compiler_fixture.rs"]
mod fixture;

use common::{digest, target};
use fixture::{floats, Fixture};
use nf_core::error::ErrorCode;
use nf_core::schemas::{DType, ToolKind};
use nf_weight_ir::compilation::identity_rule_specification;
use nf_weight_ir::program::OperationKind;
use serde_json::json;
use std::sync::Arc;

#[test]
fn compilation_emits_immutable_replayable_dry_run_without_mutation() {
    let mut fixture = Fixture::new(false);
    let before = fixture.data.clone();
    let checkpoint = fixture.checkpoint.clone();
    let first = fixture.run().unwrap();
    let again = fixture.run().unwrap();
    assert_eq!(first.digest(), again.digest());
    assert_eq!(fixture.data, before);
    assert_eq!(fixture.checkpoint, checkpoint);
    let summary = first.summary();
    assert_eq!(summary.effective_writes.len(), 1);
    assert_eq!(summary.effective_writes[0].tensor, "w");
    assert_eq!(summary.allocations.candidate_copy_bytes, 32);
    assert_eq!(summary.allocations.output_tensor_bytes, 32);
    assert_eq!(summary.pending_quality_constraints, 1);
    assert!(!summary.identity_preservation_verified);
    assert_eq!(
        summary.allocations.estimated_working_bytes,
        summary.allocations.retained_operand_bytes
            + 32
            + summary.allocations.tracking_bytes
            + summary.allocations.peak_kernel_scratch_bytes
    );
    let old_operand = first.operands()["operand"].data().to_vec();
    Arc::make_mut(fixture.operands.get_mut("operand").unwrap()).fill(0);
    assert_eq!(first.operands()["operand"].data(), old_operand);
    assert!(fixture.run().is_err());
    assert_eq!(first.instructions().len(), 1);
    assert_eq!(
        first.inventory().parameter_digest(),
        &fixture.policy.source.parameter_digest
    );
    assert_eq!(first.numerical().accumulation_dtype, DType::F64);
    assert_eq!(first.program().ir_version, 1);
}

#[test]
fn compilation_stale_source_content_config_and_signature_fail() {
    let mut fixture = Fixture::new(false);
    fixture.data[0] ^= 1;
    assert!(fixture.run().is_err());
    let mut fixture = Fixture::new(false);
    *fixture.checkpoint.last_mut().unwrap() ^= 1;
    assert_eq!(fixture.run().unwrap_err().code, ErrorCode::SourceMismatch);
    let mut fixture = Fixture::new(false);
    fixture.program["preconditions"][0]["before_digest"] = json!(digest("stale"));
    assert_eq!(fixture.run().unwrap_err().code, ErrorCode::SourceMismatch);
    let mut fixture = Fixture::new(false);
    fixture.assets.insert(
        "config".to_owned(),
        Arc::from(b"{\"different\":true}".as_slice()),
    );
    assert_eq!(fixture.run().unwrap_err().code, ErrorCode::SourceMismatch);
}

#[test]
fn compilation_operand_hash_shape_dtype_and_finiteness_are_real_checks() {
    let mut fixture = Fixture::new(false);
    *Arc::make_mut(fixture.operands.get_mut("operand").unwrap())
        .last_mut()
        .unwrap() ^= 1;
    assert_eq!(
        fixture.run().unwrap_err().code,
        ErrorCode::IntegrityMismatch
    );
    let mut fixture = Fixture::new(false);
    fixture.program["artifacts"]["operand"]["shape"] = json!([4]);
    assert_eq!(
        fixture.run().unwrap_err().code,
        ErrorCode::IntegrityMismatch
    );
    let mut fixture = Fixture::new(false);
    let f32_data: Vec<_> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    fixture.set_operand("F32", DType::F32, &[2, 2], &f32_data);
    assert_eq!(
        fixture.run().unwrap_err().code,
        ErrorCode::ShapeUnsatisfiable
    );
    let mut fixture = Fixture::new(false);
    fixture.set_operand(
        "F64",
        DType::F64,
        &[2, 2],
        &floats(&[f64::NAN, 0.0, 0.0, 0.0]),
    );
    assert_eq!(fixture.run().unwrap_err().code, ErrorCode::NonFinite);
}

#[test]
fn compilation_effect_masks_include_actual_aliases_not_only_declared_names() {
    let mut fixture = Fixture::new(true);
    let complete = fixture.run().unwrap();
    assert_eq!(complete.summary().effective_writes.len(), 2);
    assert_eq!(complete.summary().allocations.candidate_copy_bytes, 32);
    assert_eq!(complete.summary().affected_parameter_upper_bound, 4);
    fixture.program["effects"]["writes"] = json!([target()]);
    assert_eq!(fixture.run().unwrap_err().code, ErrorCode::AliasScopeEscape);
}

#[test]
fn compilation_derived_mask_is_narrower_than_conservative_declarations() {
    let mut fixture = Fixture::new(false);
    let row = json!({"tensor":"w","start":[0,0],"shape":[1,2]});
    fixture.program["operations"][0]["operation"]["target"] = row.clone();
    fixture.program["effects"]["writes"] = json!([target(), row]);
    fixture.set_operand("F64", DType::F64, &[1, 2], &floats(&[10.0, 20.0]));
    let plan = fixture.run().unwrap();
    assert_eq!(plan.summary().effective_writes.len(), 1);
    assert_eq!(plan.summary().effective_writes[0].shape, [1, 2]);
    assert_eq!(plan.summary().affected_parameter_upper_bound, 4);
}

#[test]
fn compilation_resource_plan_and_unresolved_constructor_bounds_fail_closed() {
    let mut fixture = Fixture::new(false);
    fixture.policy.bounds.max_temporary_bytes = 100;
    assert_eq!(fixture.run().unwrap_err().code, ErrorCode::LimitExceeded);
    let mut fixture = Fixture::new(false);
    fixture.policy.bounds.max_output_tensor_bytes = 31;
    assert_eq!(fixture.run().unwrap_err().code, ErrorCode::LimitExceeded);
    let mut fixture = Fixture::new(false);
    fixture.policy.bounds.max_kernel_work = 3;
    assert_eq!(fixture.run().unwrap_err().code, ErrorCode::LimitExceeded);
    let mut fixture = Fixture::new(false);
    fixture.program["operations"][0]["operation"] = json!({"type":"construct_recipe","target":target(),"recipe":{"id":"unregistered.recipe","version":1,"specification":digest("recipe")},"inputs":["operand"]});
    assert_eq!(
        fixture.run().unwrap_err().code,
        ErrorCode::UnsupportedOperation
    );
}

#[test]
fn compilation_frozen_verification_cannot_be_weakened_or_repointed() {
    for field in ["scorer", "split", "validation", "gym"] {
        let mut fixture = Fixture::new(false);
        fixture.program["verification"]["gyms"][0][field] = json!(digest("replacement"));
        assert_eq!(
            fixture.run().unwrap_err().code,
            ErrorCode::VerificationPolicyWeakened
        );
    }
    let mut fixture = Fixture::new(false);
    let mut extra = fixture.policy.verification.gyms[0].clone();
    extra.binding = digest("extra-binding").parse().unwrap();
    fixture.program["verification"]["gyms"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::to_value(&extra).unwrap());
    assert_eq!(
        fixture.run().unwrap_err().code,
        ErrorCode::VerificationPolicyWeakened
    );
    fixture.policy.allowed_extra_gyms.push(extra);
    assert!(fixture.run().is_ok());
}

#[test]
fn compilation_backend_capability_and_space_binding_are_not_claimed_by_agent() {
    let mut fixture = Fixture::new(false);
    fixture.policy.backend.capabilities.clear();
    assert_eq!(
        fixture.run().unwrap_err().code,
        ErrorCode::UnsupportedCapability
    );
    let mut fixture = Fixture::new(false);
    fixture.policy.backend.id = "vllm".to_owned();
    assert_eq!(
        fixture.run().unwrap_err().code,
        ErrorCode::UnsupportedCapability
    );
    let mut fixture = Fixture::new(false);
    fixture
        .space
        .transformations
        .remove(&OperationKind::ReplaceSlice);
    fixture.rebind_space();
    assert_eq!(
        fixture.run().unwrap_err().code,
        ErrorCode::UnsupportedOperation
    );
    let mut fixture = Fixture::new(false);
    fixture.program["base_model"]["config"] = json!(digest("unapproved-config"));
    assert_eq!(fixture.run().unwrap_err().code, ErrorCode::SourceMismatch);
}

#[test]
fn compilation_grad_inspection_permission_and_numerical_policy_stay_explicit() {
    let mut fixture = Fixture::new(false);
    fixture.program["lineage"]["method"]["inspection_uses_backprop"] = json!(true);
    assert_eq!(fixture.run().unwrap_err().code, ErrorCode::PermissionDenied);
    fixture.policy.permissions.allow_gradient_inspection = true;
    fixture
        .policy
        .permissions
        .tools
        .insert(ToolKind::Attribution);
    assert!(fixture.run().is_ok());
    fixture.program["constraints"]["numerical"]["max_abs_parameter"] = json!(101.0);
    assert_eq!(fixture.run().unwrap_err().code, ErrorCode::PermissionDenied);
    let mut fixture = Fixture::new(false);
    fixture
        .policy
        .permissions
        .tools
        .remove(&ToolKind::DirectEdit);
    assert_eq!(fixture.run().unwrap_err().code, ErrorCode::PermissionDenied);
}

#[test]
fn compilation_mask_values_and_matrix_axes_are_checked_before_mutation() {
    let mut fixture = Fixture::new(false);
    fixture.program["operations"][0]["operation"] =
        json!({"type":"mask","target":target(),"mask":"operand"});
    fixture.set_operand("U8", DType::U8, &[2, 2], &[1, 0, 1, 1]);
    assert!(fixture.run().is_ok());
    fixture.set_operand("U8", DType::U8, &[2, 2], &[1, 2, 1, 1]);
    assert_eq!(fixture.run().unwrap_err().code, ErrorCode::InvalidValue);
    let mut fixture = Fixture::new(false);
    fixture.program["operations"][0]["operation"] =
        json!({"type":"linear_map","target":target(),"matrix":"operand","side":"left"});
    fixture.set_operand("F64", DType::F64, &[2, 1], &floats(&[1.0, 2.0]));
    assert_eq!(
        fixture.run().unwrap_err().code,
        ErrorCode::ShapeUnsatisfiable
    );
}

#[test]
fn compilation_projection_preconditions_and_validation_work_are_enforced() {
    let mut fixture = Fixture::new(false);
    fixture.program["operations"][0]["operation"] = json!({"type":"subspace_projection","target":target(),"basis":"operand","side":"left","strength":0.7,"orthonormal_tolerance":0.0});
    fixture.set_operand("F64", DType::F64, &[2, 2], &floats(&[0.0, 0.0, 0.0, 0.0]));
    assert_eq!(fixture.run().unwrap_err().code, ErrorCode::InvalidValue);
    fixture.set_operand("F64", DType::F64, &[2, 2], &floats(&[1.0, 0.0, 0.0, 1.0]));
    let plan = fixture.run().unwrap();
    assert!(plan.summary().allocations.validation_scalar_work > 4);
    fixture.policy.bounds.max_validation_work = 4;
    assert_eq!(fixture.run().unwrap_err().code, ErrorCode::LimitExceeded);
}

#[test]
fn compilation_exact_claims_require_supported_checked_identity_rule() {
    let mut fixture = Fixture::new(false);
    fixture.program["preservation"] = json!({"type":"exact_under_conditions","rule":{"id":"identity_noop","version":1,"specification":identity_rule_specification()},"conditions":[fixture.policy.source.manifest]});
    assert_eq!(
        fixture.run().unwrap_err().code,
        ErrorCode::UnsupportedCapability
    );
    fixture.program["operations"] = json!([]);
    fixture.program["artifacts"] = json!({});
    fixture.operands.clear();
    fixture.program["effects"]["reads"] = json!([]);
    fixture.program["effects"]["writes"] = json!([]);
    fixture.program["lineage"]["method"]["target_update"] = json!("no_change");
    fixture.program["lineage"]["method"]["changed_parameters"] = json!(0);
    fixture.program["constraints"]["max_changed_parameters"] = json!(0);
    let plan = fixture.run().unwrap();
    assert!(plan.summary().identity_preservation_verified);
    assert!(plan.summary().effective_writes.is_empty());
    assert_eq!(plan.summary().pending_quality_constraints, 1);
}
