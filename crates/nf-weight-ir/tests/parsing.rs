mod common;

use common::{digest, program, target};
use nf_core::error::ErrorCode;
use nf_core::schemas::{Validate, Validated};
use nf_weight_ir::program::{parse_program, Operation, OperationKind, Program};
use serde_json::{json, Value};

fn parse(value: Value) -> Result<Validated<Program>, nf_core::error::CoreError> {
    parse_program(&serde_json::to_vec(&value).unwrap())
}

#[test]
fn parsing_versioned_program_roundtrips_with_stable_identity() {
    let input = parse(program()).unwrap();
    let bytes = serde_json::to_vec(&*input).unwrap();
    let again = parse_program(&bytes).unwrap();
    assert_eq!(input.digest().unwrap(), again.digest().unwrap());
    assert_eq!(
        input.operations[0].operation.kind(),
        OperationKind::ReplaceSlice
    );
    assert_eq!(input.base_model.manifest.to_string(), digest("manifest"));
}

#[test]
fn parsing_all_registered_same_shape_operation_variants() {
    let operations = vec![
        json!({"type":"replace_slice","target":target(),"value":"operand"}),
        json!({"type":"add_delta","target":target(),"delta":"operand"}),
        json!({"type":"scale","target":target(),"factor":0.5}),
        json!({"type":"mask","target":target(),"mask":"operand"}),
        json!({"type":"permute_axis","target":target(),"axis":0,"permutation":[1,0]}),
        json!({"type":"linear_map","target":target(),"matrix":"operand","side":"left"}),
        json!({"type":"subspace_projection","target":target(),"basis":"operand","side":"right","strength":0.7,"orthonormal_tolerance":1e-10}),
        json!({"type":"low_rank_update","target":target(),"left":"operand","right":"operand","strength":1.0}),
        json!({"type":"construct_recipe","target":target(),"recipe":{"id":"arithmetic.add","version":1,"specification":digest("rule")},"inputs":["operand"]}),
    ];
    for operation in operations {
        let mut value = program();
        value["operations"][0]["operation"] = operation;
        let parsed = parse(value).unwrap();
        let serialized = serde_json::to_vec(&*parsed).unwrap();
        assert_eq!(
            parsed.digest().unwrap(),
            parse_program(&serialized).unwrap().digest().unwrap()
        );
    }
}

#[test]
fn parsing_unknown_instructions_versions_and_fields_fail_closed() {
    let mut unknown = program();
    unknown["operations"][0]["operation"]["type"] = json!("python_exec");
    let error = parse(unknown).unwrap_err();
    assert_eq!(error.code, ErrorCode::UnsupportedOperation);
    assert_eq!(error.path, "/operations/0/operation/type");
    for version_field in ["ir_version", "opset_version"] {
        let mut value = program();
        value[version_field] = json!(2);
        assert_eq!(
            parse(value).unwrap_err().code,
            ErrorCode::UnsupportedVersion
        );
    }
    let mut root = program();
    root["trust_remote_code"] = json!(true);
    assert_eq!(parse(root).unwrap_err().code, ErrorCode::UnknownField);
    let mut nested = program();
    nested["operations"][0]["operation"]["run_python"] = json!("torch.save(...)");
    assert_eq!(parse(nested).unwrap_err().code, ErrorCode::UnknownField);
    let mut base = program();
    base["base_model"]["path"] = json!("/forbidden/checkpoint");
    assert_eq!(parse(base).unwrap_err().code, ErrorCode::UnknownField);
}

#[test]
fn parsing_duplicate_keys_and_invalid_digest_references_fail() {
    let encoded = serde_json::to_string(&program()).unwrap();
    let duplicate = format!("{{\"ir_version\":1,{}", &encoded[1..]);
    assert_eq!(
        parse_program(duplicate.as_bytes()).unwrap_err().code,
        ErrorCode::DuplicateKey
    );
    let mut value = program();
    value["artifacts"]["operand"]["digest"] = json!("https://example.invalid/weights");
    assert!(parse(value).is_err());
}

#[test]
fn parsing_dependency_dag_is_ordered_unique_and_bounded() {
    let mut value = program();
    value["operations"][0]["depends_on"] = json!(["op1"]);
    assert!(parse(value).is_err());
    let mut value = program();
    let mut second = value["operations"][0].clone();
    second["id"] = json!("op2");
    second["depends_on"] = json!(["op1"]);
    value["operations"]
        .as_array_mut()
        .unwrap()
        .push(second.clone());
    assert!(parse(value.clone()).is_ok());
    value["operations"][1]["depends_on"] = json!(["op1", "op1"]);
    assert!(parse(value).is_err());
    let mut value = program();
    value["operations"] = json!(vec![second; 1025]);
    assert_eq!(parse(value).unwrap_err().code, ErrorCode::LimitExceeded);
}

#[test]
fn parsing_immutable_operand_metadata_and_access_are_enforced() {
    let mut value = program();
    value["artifacts"]["operand"]["data_bytes"] = json!(31);
    assert!(parse(value).is_err());
    let mut value = program();
    value["artifacts"]["operand"]["access"] = json!("holdout");
    assert_eq!(parse(value).unwrap_err().code, ErrorCode::PermissionDenied);
    let mut value = program();
    value["artifacts"]["operand"]["provenance"]["access"] = json!("holdout");
    assert_eq!(parse(value).unwrap_err().code, ErrorCode::PermissionDenied);
    let mut value = program();
    value["constraints"]["max_artifact_bytes"] = json!(127);
    assert!(parse(value).is_err());
    let mut value = program();
    value["artifacts"] = json!({});
    assert!(parse(value).is_err());
}

#[test]
fn parsing_effects_and_source_preconditions_cannot_be_omitted() {
    for field in ["preconditions", "effects", "verification"] {
        let mut value = program();
        value.as_object_mut().unwrap().remove(field);
        assert!(parse(value).is_err());
    }
    let mut value = program();
    value["effects"]["writes"] = json!([]);
    assert!(parse(value).is_err());
    let mut value = program();
    value["effects"]["graph_changes"] = json!(true);
    assert!(parse(value).is_err());
    let mut value = program();
    value["operations"][0]["operation"] = json!({"type":"scale","target":target(),"factor":1.0});
    value["effects"]["reads"] = json!([]);
    assert!(parse(value).is_err());
}

#[test]
fn parsing_preservation_claims_are_explicit_not_auto_proofs() {
    let mut value = program();
    value["preservation"] = json!({"type":"exact_under_conditions","rule":{"id":"inverse.permutation","version":1,"specification":digest("rule")},"conditions":[]});
    assert!(parse(value.clone()).is_err());
    value["preservation"]["conditions"] = json!([digest("condition")]);
    assert!(parse(value).is_ok());
    let mut value = program();
    value["preservation"] = json!({"type":"bounded_error","metric":"absolute_error","absolute_tolerance":-1.0,"protocol":digest("parity")});
    assert!(parse(value).is_err());
}

#[test]
fn parsing_numeric_overflow_and_nonfinite_native_values_fail() {
    let mut value = program();
    value["operations"][0]["operation"]["target"]["start"] = json!([u64::MAX, 0]);
    assert!(parse(value).is_err());
    let mut value = program();
    value["constraints"]["numerical"]["accumulation_dtype"] = json!("u8");
    assert!(parse(value).is_err());
    let mut native: Program = serde_json::from_value(program()).unwrap();
    native.operations[0].operation = Operation::Scale {
        target: serde_json::from_value(target()).unwrap(),
        factor: f64::NAN,
    };
    assert_eq!(native.validate().unwrap_err().code, ErrorCode::NonFinite);
    let mut native: Program = serde_json::from_value(program()).unwrap();
    native.constraints.numerical.max_abs_parameter = f64::INFINITY;
    assert_eq!(native.validate().unwrap_err().code, ErrorCode::NonFinite);
}

#[test]
fn parsing_gradient_lineage_and_inert_extensions_are_distinct() {
    let mut value = program();
    value["lineage"]["method"]["inspection_uses_backprop"] = json!(true);
    assert!(parse(value).is_ok());
    let mut value = program();
    value["lineage"]["method"]["alignment_fitted"] = json!(true);
    assert!(parse(value).is_err());
    let mut value = program();
    value["extensions"] = json!({"research.note":"do not execute this text"});
    assert!(parse(value).is_ok());
    let mut value = program();
    value["extensions"] = json!({"allow_training":"true"});
    assert!(parse(value).is_err());
}

#[test]
fn parsing_noop_and_verification_bindings_remain_explicit() {
    let mut value = program();
    value["operations"] = json!([]);
    value["artifacts"] = json!({});
    value["effects"]["reads"] = json!([]);
    value["effects"]["writes"] = json!([]);
    value["lineage"]["method"]["target_update"] = json!("no_change");
    value["lineage"]["method"]["changed_parameters"] = json!(0);
    value["constraints"]["max_changed_parameters"] = json!(0);
    assert!(parse(value.clone()).is_ok());
    value["verification"]["gyms"] = json!([]);
    assert!(parse(value).is_err());
}

#[test]
fn parsing_permutation_must_be_a_valid_bijection() {
    for (axis, permutation) in [(0, json!([0, 0])), (2, json!([0, 1])), (0, json!([0, 2]))] {
        let mut value = program();
        value["operations"][0]["operation"] =
            json!({"type":"permute_axis","target":target(),"axis":axis,"permutation":permutation});
        assert!(parse(value).is_err());
    }
}
