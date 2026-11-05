use nf_core::canonical::ContentDigest;
use serde_json::{json, Value};

pub fn digest(value: &str) -> String {
    ContentDigest::of_bytes(value.as_bytes()).to_string()
}

pub fn target() -> Value {
    json!({"tensor":"w","start":[0,0],"shape":[2,2]})
}

pub fn program() -> Value {
    json!({
        "ir_version":1,"opset_version":1,
        "base_model":{"checkpoint":digest("checkpoint"),"manifest":digest("manifest"),
            "parameter_digest":digest("parameters"),"architecture_graph":digest("graph"),
            "config":digest("config"),"tokenizer":null},
        "objective":{"contract":digest("contract"),"objective":digest("objective"),
            "experiment_protocol":digest("experiment"),"search_space":digest("space")},
        "operations":[{"id":"op1","depends_on":[],"operation":{"type":"replace_slice","target":target(),"value":"operand"}}],
        "artifacts":{"operand":{"digest":digest("operand"),"tensor":"value","dtype":"f64","shape":[2,2],
            "data_bytes":32,"artifact_bytes":128,"access":"development", "provenance":{
                "schema_version":1,"origin":"supplied_fact","support":"untested","provenance":"fixture",
                "evidence":[],"access":"development"}}},
        "preconditions":[{"tensor":"w","role":"parameter","shape":[2,2],"dtype":"f64",
            "before_digest":digest("before"),"storage_group":"w"}],
        "required_capabilities":["dense.cpu"],
        "constraints":{"gradient_free":true,"preserve_tensor_shapes":true,
            "max_changed_parameters":4,"max_artifact_bytes":128,"max_temporary_bytes":4096,
            "numerical":{"accumulation_dtype":"f64","output_rounding":"nearest_ties_even",
                "non_finite":"reject","max_abs_parameter":100.0,"backend_abs_tolerance":0.0}},
        "effects":{"reads":[target()],"writes":[target()],"buffer_writes":[],
            "graph_changes":false,"config_changes":[],"tokenizer_changes":false},
        "preservation":{"type":"experimental"},
        "verification":{"portfolio":digest("portfolio"),"protocol":digest("verification"),
            "gyms":[{"binding":digest("binding"),"gym":digest("gym"),"validation":digest("validation"),
                "scorer":digest("scorer"),"split":digest("selection"),"role":"target"}],
            "smoke_suite":digest("smoke"),"parity_suite":digest("parity")},
        "lineage":{"parent_candidates":[],"hypothesis":digest("hypothesis"),"method":{
            "schema_version":1,"target_update":"direct_tensor_edit","proposal_origin":"scripted",
            "adaptation":"none","production_history":"direct_only","inspection_uses_backprop":false,
            "parameter_update_uses_backprop":false,"gradient_derived_update_values":false,
            "alignment_fitted":false,"optimizer_steps":0,"changed_parameters":4}},
        "extensions":{}
    })
}
