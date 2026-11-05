use nf_core::canonical::ContentDigest;
use nf_core::error::ErrorCode;
use nf_core::schemas::*;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::{json, Value};
use std::fmt::Debug;

fn validated<T: Validate + DeserializeOwned>(
    value: Value,
) -> Result<Validated<T>, nf_core::error::CoreError> {
    Validated::parse(&serde_json::to_vec(&value).unwrap())
}

fn permissions() -> Value {
    json!({
        "schema_version":1, "tools":["inspect","direct_edit"], "external_destinations":{},
        "allow_gradient_inspection":false, "allow_parameter_training":false,
        "require_fully_gradient_free":true, "allow_generated_gym":false, "allow_downloads":false
    })
}

fn model() -> Value {
    let digest = ContentDigest::of_bytes(b"fixture").to_string();
    json!({
        "schema_version":1, "architecture":{"id":"fixture.relu","version":"1","config":digest},
        "parameters":{"head.weight":{"shape":[2,3],"dtype":"f64","digest":digest,"storage_group":"head"}},
        "buffers":{}, "parameter_digest":digest,"format":"dense_safetensors","packing_policy":null,
        "assets":{},"usage_rights":"synthetic local fixture"
    })
}

fn method() -> Value {
    json!({
        "schema_version":1,"target_update":"direct_tensor_edit","proposal_origin":"scripted",
        "adaptation":"none","production_history":"direct_only","inspection_uses_backprop":false,
        "parameter_update_uses_backprop":false,"gradient_derived_update_values":false,
        "alignment_fitted":false,"optimizer_steps":0,"changed_parameters":1
    })
}

fn capability() -> Value {
    json!({
        "schema_version":1,"id":"fixture.cpu","version":"1","model_factories":["fixture.relu"],
        "operations":["replace_slice"],"required_tools":["direct_edit"],"isolation":"trusted_builtin",
        "uses_backprop":false,"fits_parameters":false,"max_input_bytes":1024,
        "max_output_bytes":1024,"max_wall_millis":500
    })
}

#[test]
fn schemas_permissions_default_deny_and_exact_payload_classes() {
    let base = validated::<Permissions>(permissions()).unwrap();
    assert!(base.check_tool(ToolKind::DirectEdit).is_ok());
    assert_eq!(
        base.check_tool(ToolKind::Training).unwrap_err().code,
        ErrorCode::PermissionDenied
    );
    let mut remote = permissions();
    remote["tools"] = json!(["inspect", "external_request"]);
    remote["external_destinations"] = json!({"teacher-A":["summaries"]});
    let policy = validated::<Permissions>(remote).unwrap();
    assert!(policy
        .check_external("teacher-A", &[DataClass::Summaries].into())
        .is_ok());
    assert!(policy
        .check_external("teacher-A", &[DataClass::Weights].into())
        .is_err());
    assert!(policy
        .check_external("teacher-B", &[DataClass::Summaries].into())
        .is_err());
    assert!(policy
        .check_external("teacher-A", &Default::default())
        .is_err());
}

#[test]
fn schemas_reject_contradictory_permissions() {
    for (tools, flag) in [
        (json!(["training"]), "allow_parameter_training"),
        (json!(["attribution"]), "allow_gradient_inspection"),
        (json!(["download"]), "allow_downloads"),
        (json!(["generated_gym"]), "allow_generated_gym"),
    ] {
        let mut policy = permissions();
        policy["tools"] = tools;
        let error = validated::<Permissions>(policy).unwrap_err();
        assert_eq!(error.path, format!("/{flag}"));
    }
    let mut policy = permissions();
    policy["allow_gradient_inspection"] = json!(true);
    assert!(validated::<Permissions>(policy).is_err());
    let mut policy = permissions();
    policy["external_destinations"] = json!({"teacher-A":["weights"]});
    assert!(validated::<Permissions>(policy).is_err());
}

#[test]
fn schemas_model_inventory_alias_and_overflow_contracts() {
    let manifest = validated::<ModelManifest>(model()).unwrap();
    assert_eq!(manifest.parameters["head.weight"].byte_len().unwrap(), 48);
    let mut tied = model();
    tied["parameters"]["alias"] = tied["parameters"]["head.weight"].clone();
    assert!(validated::<ModelManifest>(tied.clone()).is_ok());
    tied["parameters"]["alias"]["dtype"] = json!("f32");
    assert!(validated::<ModelManifest>(tied).is_err());
    let mut overflow = model();
    overflow["parameters"]["head.weight"]["shape"] = json!([u64::MAX, 2]);
    let error = validated::<ModelManifest>(overflow).unwrap_err();
    assert_eq!(error.path, "/parameters/head.weight/shape");
    let mut packed = model();
    packed["format"] = json!("quantized_safetensors");
    assert!(validated::<ModelManifest>(packed).is_err());
}

#[test]
fn schemas_unknown_fields_enums_versions_are_rejected() {
    let mut unknown = permissions();
    unknown["root_access"] = json!(true);
    assert_eq!(
        validated::<Permissions>(unknown).unwrap_err().code,
        ErrorCode::UnknownField
    );
    let mut nested = model();
    nested["architecture"]["python_code"] = json!("execute me");
    assert_eq!(
        validated::<ModelManifest>(nested).unwrap_err().code,
        ErrorCode::UnknownField
    );
    let mut unknown_enum = model();
    unknown_enum["format"] = json!("pickle");
    assert!(validated::<ModelManifest>(unknown_enum).is_err());
    let mut future = permissions();
    future["schema_version"] = json!(2);
    assert_eq!(
        validated::<Permissions>(future).unwrap_err().code,
        ErrorCode::UnsupportedVersion
    );
}

#[test]
fn schemas_attribution_is_inspection_but_not_fully_gradient_free() {
    assert!(validated::<MethodRecord>(method())
        .unwrap()
        .require_fully_gradient_free()
        .is_ok());
    let mut attributed = method();
    attributed["inspection_uses_backprop"] = json!(true);
    let record = validated::<MethodRecord>(attributed).unwrap();
    assert_eq!(record.production_history, ProductionHistory::DirectOnly);
    assert!(record.require_fully_gradient_free().is_err());
    for field in [
        "parameter_update_uses_backprop",
        "gradient_derived_update_values",
        "alignment_fitted",
    ] {
        let mut fitted = method();
        fitted[field] = json!(true);
        assert!(validated::<MethodRecord>(fitted.clone()).is_err());
        fitted["production_history"] = json!("hybrid");
        assert!(validated::<MethodRecord>(fitted).is_ok());
    }
    let mut trained = method();
    trained["optimizer_steps"] = json!(1);
    assert!(validated::<MethodRecord>(trained).is_err());
}

#[test]
fn schemas_unknown_history_and_noop_cannot_launder_changes() {
    let mut unknown = method();
    unknown["proposal_origin"] = json!("external_unknown");
    assert!(validated::<MethodRecord>(unknown.clone()).is_err());
    unknown["production_history"] = json!("external_unverified");
    assert!(validated::<MethodRecord>(unknown)
        .unwrap()
        .require_fully_gradient_free()
        .is_err());
    let mut noop = method();
    noop["target_update"] = json!("no_change");
    assert!(validated::<MethodRecord>(noop.clone()).is_err());
    noop["changed_parameters"] = json!(0);
    assert!(validated::<MethodRecord>(noop).is_ok());
}

#[test]
fn schemas_evidence_origin_and_causal_support_are_separate() {
    let mut label = json!({"schema_version":1,"origin":"model_claim","support":"suggestive",
        "provenance":"agent-A","evidence":[],"access":"development"});
    assert!(validated::<EvidenceLabel>(label.clone()).is_ok());
    label["support"] = json!("intervention_supported");
    assert!(validated::<EvidenceLabel>(label.clone()).is_err());
    label["evidence"] = json!([ContentDigest::of_bytes(b"causal-protocol")]);
    assert!(validated::<EvidenceLabel>(label.clone()).is_err());
    label["origin"] = json!("measured_result");
    assert!(validated::<EvidenceLabel>(label).is_ok());
}

#[test]
fn schemas_capability_limits_training_and_isolation() {
    assert!(validated::<CapabilityDescriptor>(capability()).is_ok());
    for field in ["max_input_bytes", "max_output_bytes", "max_wall_millis"] {
        let mut descriptor = capability();
        descriptor[field] = json!(0);
        assert!(validated::<CapabilityDescriptor>(descriptor).is_err());
    }
    let mut descriptor = capability();
    descriptor["fits_parameters"] = json!(true);
    assert!(validated::<CapabilityDescriptor>(descriptor).is_err());
    let mut descriptor = capability();
    descriptor["required_tools"] = json!(["generated_gym"]);
    assert!(validated::<CapabilityDescriptor>(descriptor.clone()).is_err());
    descriptor["isolation"] = json!("restricted_sandbox");
    assert!(validated::<CapabilityDescriptor>(descriptor).is_ok());
}

fn roundtrip<T: Serialize + DeserializeOwned + PartialEq + Debug>(value: T) {
    let decoded: T = serde_json::from_slice(&serde_json::to_vec(&value).unwrap()).unwrap();
    assert_eq!(decoded, value);
}

#[test]
fn schemas_enum_wire_roundtrips() {
    for item in [
        AccessClass::Public,
        AccessClass::Development,
        AccessClass::Selection,
        AccessClass::Holdout,
        AccessClass::Confidential,
    ] {
        roundtrip(item);
    }
    for item in [
        DataClass::Weights,
        DataClass::Activations,
        DataClass::Tasks,
        DataClass::Labels,
        DataClass::Feedback,
        DataClass::ToolTraces,
        DataClass::Summaries,
    ] {
        roundtrip(item);
    }
    for item in [
        ToolKind::Inspect,
        ToolKind::Capture,
        ToolKind::Attribution,
        ToolKind::DirectEdit,
        ToolKind::Training,
        ToolKind::ExternalRequest,
        ToolKind::GeneratedGym,
        ToolKind::Download,
    ] {
        roundtrip(item);
    }
    for item in [
        DType::F32,
        DType::F64,
        DType::F16,
        DType::Bf16,
        DType::I8,
        DType::U8,
        DType::I32,
        DType::I64,
    ] {
        roundtrip(item);
    }
    for item in [
        TargetUpdate::DirectTensorEdit,
        TargetUpdate::GradientTraining,
        TargetUpdate::NoChange,
        TargetUpdate::ArchitectureTransformation,
    ] {
        roundtrip(item);
    }
    for item in [
        ProposalOrigin::Scripted,
        ProposalOrigin::ReasoningModel,
        ProposalOrigin::LearnedProposer,
        ProposalOrigin::ExternalUnknown,
    ] {
        roundtrip(item);
    }
    for item in [
        Adaptation::None,
        Adaptation::ExternalMemory,
        Adaptation::AdaptiveMemory,
    ] {
        roundtrip(item);
    }
    for item in [
        ProductionHistory::DirectOnly,
        ProductionHistory::Trained,
        ProductionHistory::Hybrid,
        ProductionHistory::ExternalUnverified,
    ] {
        roundtrip(item);
    }
    for item in [
        EvidenceOrigin::SuppliedFact,
        EvidenceOrigin::ModelClaim,
        EvidenceOrigin::MeasuredResult,
        EvidenceOrigin::Assumption,
    ] {
        roundtrip(item);
    }
    for item in [
        SupportGrade::Untested,
        SupportGrade::Suggestive,
        SupportGrade::InterventionSupported,
        SupportGrade::IndependentlyConfirmedUnderProtocol,
        SupportGrade::Contradicted,
        SupportGrade::Inconclusive,
    ] {
        roundtrip(item);
    }
    for item in [
        IsolationRequirement::TrustedBuiltin,
        IsolationRequirement::RestrictedSandbox,
    ] {
        roundtrip(item);
    }
    for item in [
        TensorFormat::DenseSafetensors,
        TensorFormat::QuantizedSafetensors,
    ] {
        roundtrip(item);
    }
}
