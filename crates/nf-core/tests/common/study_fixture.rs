use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use serde_json::{json, Value};
pub fn d(value: &str) -> ContentDigest {
    ContentDigest::of_bytes(value.as_bytes())
}
pub fn hash(value: &Value) -> ContentDigest {
    CanonicalDocument::parse(&serde_json::to_vec(value).unwrap(), Limits::default())
        .unwrap()
        .digest()
}
pub fn refresh(value: &mut Value) {
    value["portfolio_digest"] = json!(hash(&value["portfolio"]));
}
pub fn rebind_gym(value: &mut Value, index: usize) {
    let old = value["portfolio"]["bindings"][index]["gym"]
        .as_str()
        .unwrap()
        .to_owned();
    let gym = value["portfolio"]["gyms"]
        .as_object_mut()
        .unwrap()
        .remove(&old)
        .unwrap();
    let id = hash(&gym);
    value["portfolio"]["gyms"][id.as_str()] = gym;
    value["portfolio"]["bindings"][index]["gym"] = json!(id);
    refresh(value);
}
fn factory(id: &str) -> Value {
    json!({"id":id,"version":"1","config":d(id)})
}
fn metric(domain: &str, unit: &str, direction: &str) -> Value {
    json!({"domain":domain,"unit":unit,"direction":direction,"lower":if unit=="fraction"{json!(0.0)}else{Value::Null},"upper":if unit=="fraction"{json!(1.0)}else{Value::Null}})
}
fn threshold(
    floor: Option<f64>,
    ceiling: Option<f64>,
    regression: Option<f64>,
    gain: Option<f64>,
) -> Value {
    json!({"floor":floor,"ceiling":ceiling,"max_regression":regression,"min_gain":gain})
}
fn metric_binding(metric: &str, threshold: Value, scope: Option<&str>) -> Value {
    json!({"metric":metric,"required":true,"priority":1.0,"comparator":"original","threshold":threshold,"statistic":if scope.is_some(){"maximum"}else{"exact"},"measurement":scope.map(|scope|json!({"scope":scope,"statistic":"maximum","warmup":1,"repetitions":3,"order":"paired_interleaved","synchronization":"cpu_monotonic","timer":factory("fixture.timer")}))})
}
fn split(gym: usize, role: &str) -> Value {
    json!({"schema_version":1,"role":role,"dataset":d(&format!("dataset{gym}{role}")),"items":[{"item":d(&format!("item{gym}{role}")),"group":d(&format!("group{gym}{role}")),"family":d("task-family")}],"exposure":"known_unexposed","exposure_ledger":d("exposure-ledger"),"access_policy":d(role)})
}
pub fn standard() -> Value {
    let mut gyms = serde_json::Map::new();
    let mut bindings = Vec::new();
    for i in 0..5 {
        let metrics = if i == 3 {
            json!({"latency":metric("latency","milliseconds","lower_is_better"),"memory":metric("memory","bytes","lower_is_better")})
        } else if i == 4 {
            json!({"safety":metric("safety","fraction","higher_is_better")})
        } else {
            json!({"quality":metric("quality","fraction","higher_is_better")})
        };
        let gym = json!({"schema_version":1,"package":"fixture.suite","name":format!("gym{i}"),"version":"1","origin":if i%2==0{"foundry"}else{"user"},"adapter":factory("fixture.gym"),"code":d(&format!("code{i}")),"dependencies":d("gym-lock"),"image":null,"dataset":d(&format!("dataset{i}")),"input_schema":d("input-schema"),"output_schema":d("output-schema"),"scorer":d(&format!("scorer{i}")),"oracle":d("reference-oracle"),"lifecycle_protocol":d("reset-termination-cleanup"),"model_interface":factory("fixture.model-interface"),"data_policy":d("dataset-generation-policy"),"scoring_protocol":d("rubric-invalid-output-policy"),"privacy_policy":d("retention-export-policy"),"metrics":metrics,"execution_trust":"trusted_fixture","isolation_policy":d("fixture-policy"),"validation":d(&format!("validation{i}")),"approval":null,"usage_rights":"synthetic local fixtures","confidentiality":"public","max_steps":1,"max_episode_ms":1000,"max_output_bytes":4096});
        let id = hash(&gym);
        gyms.insert(id.to_string(), gym);
        let metric_bindings = match i {
            0 => vec![metric_binding(
                "quality",
                threshold(None, None, None, Some(0.01)),
                None,
            )],
            1 => vec![metric_binding(
                "quality",
                threshold(None, None, Some(0.01), None),
                None,
            )],
            2 => vec![metric_binding(
                "quality",
                threshold(Some(0.0), None, None, None),
                None,
            )],
            3 => vec![
                metric_binding(
                    "latency",
                    threshold(None, Some(100.0), None, None),
                    Some("forward_latency"),
                ),
                metric_binding(
                    "memory",
                    threshold(None, Some(1_000_000.0), None, None),
                    Some("process_rss"),
                ),
            ],
            _ => vec![metric_binding(
                "safety",
                threshold(Some(0.8), None, None, None),
                None,
            )],
        };
        let roles = match i {
            0 => json!(["target"]),
            1 => json!(["retention"]),
            2 => json!(["diagnostic"]),
            3 => json!(["efficiency"]),
            _ => json!(["safety", "independent_verification"]),
        };
        bindings.push(json!({"id":d(&format!("binding{i}")),"objective":d("objective"),"gym":id,"validation":d(&format!("validation{i}")),"roles":roles,"required":true,"metrics":metric_bindings,"splits":{"development":split(i,"development"),"selection":split(i,"selection"),"holdout":split(i,"holdout")},"schedule":{"development_episodes":1,"selection_episodes":1,"holdout_episodes":1,"every_n_candidates":1,"final_required":true}}));
    }
    let portfolio = json!({"schema_version":1,"profile":"standard_five_gym","label":"standard software fixture assurance","gyms":gyms,"bindings":bindings,"coverage":d("coverage"),"curriculum":d("curriculum"),"deduplicate_shared_tasks":true,"independent":{"scope":"fixture_only","evaluator":factory("fixture.evaluator"),"administration":d("evaluation-domain"),"isolation":"trusted_fixture_broker","frozen_panel_protocol":d("final-panel"),"close_revision_after_feedback":true,"require_group_disjoint":true}});
    let model = json!({"checkpoint":d("checkpoint"),"manifest":d("model-manifest"),"parameters":d("parameters"),"buffers":d("buffers"),"graph":d("graph"),"config":d("config"),"tokenizer":null,"system_state":null,"runtime_artifact":null});
    json!({"schema_version":1,"contract_key":d("study-contract-key"),"study_family":d("study-family"),"revision":1,"parent_contract":null,"approval_receipt":d("owner-approval"),"source":model,"best_known":model,"objective":{"id":d("objective"),"kind":"capability_gain","description":"repair the bounded arithmetic fixture","primary_binding":d("binding0"),"primary_metric":"quality","primary_comparator":"original"},"intervention":{"scope":d("tensor-scope"),"transformation_space":d("frozen-search-space"),"method_protocol":d("direct-edit-protocol"),"allowed_engines":[factory("nf.native_cpu")],"allow_graph_change":false,"allow_buffer_write":false,"policy_checks":d("policy-checks")},"permissions":{"schema_version":1,"tools":["inspect","direct_edit"],"external_destinations":{},"allow_gradient_inspection":false,"allow_parameter_training":false,"require_fully_gradient_free":true,"allow_generated_gym":false,"allow_downloads":false},"sources":[],"portfolio_digest":hash(&portfolio),"portfolio":portfolio,"baselines":[{"kind":"original","required":true,"enabled":true,"input":model,"backend":factory("fixture.source")},{"kind":"no_op","required":true,"enabled":true,"input":model,"backend":factory("fixture.noop")},{"kind":"best_known","required":true,"enabled":true,"input":model,"backend":factory("fixture.best")}],"controls":{"original_required":true,"noop_required":true,"best_known_required":true,"matched_tasks_and_seeds":true,"noop_apply_export_reload":true,"noop_exact_parameters_and_parity":true,"positive_gain_required":true},"seeds":[17,42],"statistics":{"estimator":"deterministic_exact","minimum_groups":1,"repeats":3,"confidence":1.0,"multiplicity_family":d("comparison-family"),"maximum_comparisons":100,"maximum_finalists":1,"stopping_rule":d("stopping-rule")},"runtime":{"schema_version":1,"device":"cpu","physical_device":d("physical-cpu"),"backend":factory("fixture.cpu"),"precision":"f64","driver":d("cpu-driver"),"dependencies":d("model-runtime-lock"),"code_revision":d("code-revision"),"deterministic":true,"batch_size":1,"input_length":2,"max_output_length":2,"length_unit":"tensor_elements","concurrency":1,"cache":"disabled","synchronization":"cpu_monotonic","state_reset_protocol":d("state-reset"),"environment":d("environment")},"limits":{"wall_ms":60000,"cpu_ms":60000,"gpu_ms":0,"candidate_attempts":10,"model_evaluations":10000,"reasoning_tokens":0,"teacher_tokens":0,"external_spend_microusd":0,"host_memory_bytes":100_000_000,"device_memory_bytes":0,"storage_bytes":10_000_000,"concurrent_workers":1,"retries_per_operation":2},"promotion_protocol":d("promotion"),"export_parity_protocol":d("export-parity")})
}
