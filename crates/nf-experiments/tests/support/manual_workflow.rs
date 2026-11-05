//! Real local checkpoint/contract input shared by Rust and CLI integration tests.
use crate::{compiler_fixture, fixture};
use nf_checkpoint::export::{publish_export, ExportLimits};
use nf_checkpoint::store::{ArtifactStore, StoreLimits};
use nf_core::schemas::Validated;
use nf_graph::inventory::tensor_set_digest;
use nf_surgeon::transaction::{SmokeProbe, SmokeSuite};
use serde_json::{json, Value};
use std::path::Path;
use std::sync::Arc;

pub fn setup(root: &Path) -> Value {
    let mut f = compiler_fixture::Fixture::new(false);
    let mut c = fixture::standard();
    let model = json!({"checkpoint":f.program["base_model"]["checkpoint"],
        "manifest":f.program["base_model"]["manifest"],"parameters":f.manifest.parameter_digest,
        "buffers":tensor_set_digest(&f.manifest.buffers).unwrap(),
        "graph":f.program["base_model"]["architecture_graph"],"config":f.program["base_model"]["config"],
        "tokenizer":null,"system_state":null,"runtime_artifact":null});
    c["source"] = model.clone();
    c["best_known"] = model.clone();
    for arm in c["baselines"].as_array_mut().unwrap() {
        arm["input"] = model.clone();
    }
    c["contract_key"] = json!(f.space.contract);
    c["permissions"] = json!(f.policy.permissions);
    f.space.constraints.quality_limits[0].binding = fixture::d("binding1");
    f.space.constraints.quality_limits[0].metric = "quality".into();
    f.rebind_space();
    c["intervention"]["transformation_space"] = json!(f.space.digest().unwrap());
    c["intervention"]["scope"] = json!(fixture::hash(&f.program["effects"]));
    c["intervention"]["allowed_engines"][0]["config"] =
        json!(fixture::hash(&json!(f.policy.backend)));
    let smoke = Validated::new(SmokeSuite {
        schema_version: 1,
        adapter: "nf.linear_fixture.v1".into(),
        // Deliberately wrong expected output: planning must NOT execute probes.
        probes: vec![SmokeProbe {
            input: vec![1.0, 2.0],
            expected: Some(vec![999.0, 999.0]),
            absolute_tolerance: 0.0,
            max_abs_output: 1000.0,
        }],
    })
    .unwrap();
    f.program["verification"]["smoke_suite"] = json!(smoke.digest().unwrap());
    f.program["verification"]["portfolio"] = c["portfolio_digest"].clone();
    let gyms: Vec<_> = c["portfolio"]["bindings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| {
            let gym = &c["portfolio"]["gyms"][b["gym"].as_str().unwrap()];
            json!({"binding":b["id"],"gym":b["gym"],"validation":b["validation"],
            "scorer":gym["scorer"],"split":fixture::hash(&b["splits"]["selection"]),
            "role":b["roles"][0]})
        })
        .collect();
    f.program["verification"]["gyms"] = json!(gyms);
    f.policy.verification = serde_json::from_value(f.program["verification"].clone()).unwrap();
    let store = ArtifactStore::create(
        &root.join("store"),
        StoreLimits {
            total_bytes: 536_870_912,
            max_blob_bytes: 67_108_864,
            max_entries: 4096,
        },
    )
    .unwrap();
    let export = publish_export(
        &store,
        &f.manifest,
        Arc::from(f.checkpoint),
        f.assets,
        ExportLimits::default(),
    )
    .unwrap();
    for operand in f.operands.values() {
        store.put(operand.as_ref()).unwrap();
    }
    store.put(b"graph".as_slice()).unwrap();
    json!({"schema_version":1,"store":root.join("store"),"source_export":export,
        "contract":c,"program":f.program,"search_space":f.space,"policy":f.policy,"smoke_suite":&*smoke})
}

pub fn call(operation: &str, value: &Value) -> nf_core::error::Result<Value> {
    nf_experiments::manual_workflow::dispatch(operation, &serde_json::to_vec(value).unwrap())
}
pub fn bound(mut request: Value) -> Value {
    let plan = call("plan", &request).unwrap();
    request["contract"]["intervention"]["method_protocol"] = plan["manifest_digest"].clone();
    request
}
pub fn freeze(root: &Path, request: &Value) -> Value {
    let plan = call("plan", request).unwrap();
    call("freeze", &json!({"plan":request,"ledger":root.join("ledger"),"expected_plan_digest":plan["plan_digest"]})).unwrap()
}
pub fn existing(root: &Path, frozen: &Value) -> Value {
    json!({"schema_version":1,"store":root.join("store"),"ledger":root.join("ledger"),"contract_digest":frozen["contract_digest"]})
}
