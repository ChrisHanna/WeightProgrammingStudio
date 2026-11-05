//! Conformance tests exercise the registered authoritative library module.
use nf_experiments::repair_benchmark::{ResultRecord, Task, COSTS, METHODS, PHASES};
use serde_json::{json, Value};

fn digest(value: &Value) -> String {
    let doc = nf_core::canonical::CanonicalDocument::parse(
        &serde_json::to_vec(value).unwrap(),
        Default::default(),
    )
    .unwrap();
    nf_core::canonical::ContentDigest::of_bytes(doc.bytes()).to_string()
}
fn example() -> (Value, Value) {
    let d = digest(&json!("test-only identity; not execution evidence"));
    let mut task = json!({
        "schema_version": 1, "protocol": "weight-repair-benchmark-v1", "task_version": "test-v1",
        "source": d, "model": d, "gyms": {}, "splits": {},
        "failure": {"kind": "injected_known", "provenance": d, "ground_truth": d, "solution_exposure": "known"},
        "programs": {"no_op": digest(&json!("no_op")), "direct": digest(&json!("direct")), "direct_failed": digest(&json!("direct_failed"))},
        "permissions": ["inspect", "direct_edit"],
        "budget": {"max_cases": 289, "max_attempts": 16, "max_forwards": 10000, "max_wall_seconds": 60},
        "runtime": d, "seed": 17, "comparisons": METHODS,
    });
    let heads = [
        "sum",
        "difference",
        "maximum",
        "minimum",
        "absolute_difference",
    ];
    for head in heads {
        task["gyms"][head] = json!(digest(&json!(head)));
    }
    for role in ["development", "selection", "holdout"] {
        task["splits"][role] = json!(digest(&json!(role)));
    }
    let id = digest(&task);
    let cost = |scope: &str| {
        let quantities: serde_json::Map<String, Value> = COSTS.iter().map(|(name, unit)| {
            (name.to_string(), json!({"status": "known", "value": 0, "unit": unit, "reason": "test vector, not measured"}))
        }).collect();
        json!({"operation_id": format!("op-{scope}"), "scope": scope, "phases": ["setup"], "evidence": d, "quantities": quantities})
    };
    let mut attempts = Vec::new();
    for name in METHODS.iter().copied().chain(["direct_failed"]) {
        let method = if name == "direct_failed" {
            "direct"
        } else {
            name
        };
        let run = matches!(method, "original" | "no_op" | "best_known" | "direct");
        let fail = matches!(name, "original" | "no_op" | "direct_failed");
        let mut arm = json!({
            "id": name, "method": method, "status": if matches!(method, "decoding" | "activation") {"unsupported"} else if !run {"not_run"} else if fail {"failed"} else {"completed"},
            "reason": "schema conformance vector only", "task": id, "source": d,
            "candidate": null, "program": null, "scores": {},
            "behavioral": if !run {"not_evaluated"} else if fail {"failed"} else {"passed"},
            "localization": "not_evaluated", "mechanism": "not_evaluated",
            "gradient_inspection": false, "gradient_update": false, "evidence": {}, "costs": [],
        });
        if run {
            arm["candidate"] = if method == "direct" {
                json!(digest(&json!(name)))
            } else {
                json!(d)
            };
            for head in heads {
                arm["scores"][head] =
                    json!({"count": 2, "correct": if fail && head == "sum" {1} else {2}});
            }
            arm["evidence"] = json!({"observations": d, "source_integrity": d});
            if matches!(method, "no_op" | "direct") {
                arm["program"] = task["programs"][name].clone();
                arm["evidence"]["apply_receipt"] = json!(d);
            }
            arm["costs"] = json!([cost(name)]);
        }
        attempts.push(arm);
    }
    let coverage: serde_json::Map<String, Value> = PHASES
        .iter()
        .map(|p| {
            (
                p.to_string(),
                json!(if *p == "setup" {
                    "measured_or_unknown"
                } else {
                    "not_run"
                }),
            )
        })
        .collect();
    let result = json!({"schema_version": 1, "protocol": task["protocol"], "task": id,
        "scope": "synthetic_component_fixture_not_scientific_acceptance", "evaluation_split": "development",
        "attempts": attempts, "failed_direct_attempts": 1, "shared_costs": [cost("shared")], "phase_coverage": coverage,
        "efficiency_claim": "not_ranked", "budget_verdict": "incomplete",
        "workflow_linkage": "unavailable_pending_authorized_workflow"});
    (task, result)
}
fn check(task: &Value, result: &Value) -> Result<ResultRecord, String> {
    ResultRecord::parse(
        &serde_json::to_vec(task).unwrap(),
        &serde_json::to_vec(result).unwrap(),
    )
}

#[test]
fn repair_benchmark_strict_versions_provenance_and_json() {
    let (task, result) = example();
    check(&task, &result).unwrap();
    for field in [
        "schema_version",
        "programs",
        "failure",
        "budget",
        "comparisons",
        "splits",
    ] {
        let mut bad = task.clone();
        bad.as_object_mut().unwrap().remove(field);
        assert!(check(&bad, &result).is_err(), "{field}");
    }
    let mut bad = task.clone();
    bad["schema_version"] = json!(2);
    assert!(check(&bad, &result).is_err());
    let mut bad = task.clone();
    bad["extra"] = json!(true);
    assert!(check(&bad, &result).is_err());
    let mut natural = task.clone();
    natural["failure"]["kind"] = json!("natural");
    assert!(Task::parse(&serde_json::to_vec(&natural).unwrap()).is_err());
    natural["failure"]["ground_truth"] = Value::Null;
    Task::parse(&serde_json::to_vec(&natural).unwrap()).unwrap();
    let mut relabeled = result.clone();
    relabeled["task"] = json!(digest(&natural));
    assert!(check(&natural, &relabeled).is_err());
    assert!(Task::parse(br#"{"schema_version":1,"schema_version":1}"#).is_err());
    assert!(Task::parse(br#"{"schema_version":NaN}"#).is_err());
}
#[test]
fn repair_benchmark_missing_slots_evidence_failed_direct_and_fallback_reject() {
    let (task, result) = example();
    for index in 0..result["attempts"].as_array().unwrap().len() {
        let mut bad = result.clone();
        bad["attempts"].as_array_mut().unwrap().remove(index);
        assert!(check(&task, &bad).is_err(), "missing slot {index}");
    }
    for field in ["observations", "source_integrity", "apply_receipt"] {
        let mut bad = result.clone();
        bad["attempts"][1]["evidence"]
            .as_object_mut()
            .unwrap()
            .remove(field);
        assert!(check(&task, &bad).is_err(), "missing evidence {field}");
    }
    let mut bad = result.clone();
    bad["attempts"][3]["method"] = json!("lora");
    assert!(check(&task, &bad).is_err());
    let mut bad = result.clone();
    bad["attempts"][10]["status"] = json!("completed");
    assert!(check(&task, &bad).is_err());
    let mut bad = result.clone();
    bad["attempts"][1]["scores"]["sum"]["correct"] = json!(2);
    assert!(check(&task, &bad).is_err());
    let mut bad = result.clone();
    bad["attempts"][0]["gradient_update"] = json!(true);
    assert!(check(&task, &bad).is_err());
    for (index, field, value) in [
        (0, "candidate", json!(digest(&json!("changed original")))),
        (3, "candidate", task["source"].clone()),
        (4, "status", json!("not_run")),
        (6, "status", json!("unsupported")),
    ] {
        let mut bad = result.clone();
        bad["attempts"][index][field] = value;
        assert!(check(&task, &bad).is_err(), "{index}/{field}");
    }
    let mut bad = result.clone();
    bad["attempts"][10]["candidate"] = bad["attempts"][3]["candidate"].clone();
    assert!(check(&task, &bad).is_err());
}
#[test]
fn repair_benchmark_units_unknowns_shared_double_count_and_budget() {
    let (task, result) = example();
    let mut good = result.clone();
    good["shared_costs"][0]["quantities"]["host_peak"]["status"] = json!("unknown");
    good["shared_costs"][0]["quantities"]["host_peak"]["value"] = Value::Null;
    check(&task, &good).unwrap();
    good["shared_costs"][0]["quantities"]["host_peak"]["value"] = json!(0);
    assert!(check(&task, &good).is_err());
    for (field, value) in [
        ("unit", json!("milliseconds")),
        ("value", json!(-1)),
        ("status", json!("missing")),
    ] {
        let mut bad = result.clone();
        bad["attempts"][0]["costs"][0]["quantities"]["wall"][field] = value;
        assert!(check(&task, &bad).is_err());
    }
    let mut bad = result.clone();
    bad["attempts"][0]["costs"][0]["quantities"]
        .as_object_mut()
        .unwrap()
        .remove("gpu");
    assert!(check(&task, &bad).is_err());
    let mut bad = result.clone();
    bad["shared_costs"]
        .as_array_mut()
        .unwrap()
        .push(result["shared_costs"][0].clone());
    assert!(check(&task, &bad).is_err());
    let mut bad = result.clone();
    bad["attempts"][0]["costs"][0]["quantities"]["forwards"]["value"] = json!(10001);
    assert!(check(&task, &bad).is_err());
    let mut bad = result.clone();
    bad["efficiency_claim"] = json!("best_total_cost");
    assert!(check(&task, &bad).is_err());
    let mut bad = result.clone();
    bad["attempts"][4]["scores"]["sum"] = json!({"count": 1, "correct": 1});
    assert!(check(&task, &bad).is_err());
}
#[test]
fn repair_benchmark_actual_python_record_when_supplied() {
    // The Python test sets this in a bounded subprocess and requires this exact
    // test name. Ordinary Rust conformance tests do not claim Python execution.
    if let Ok(bytes) = std::env::var("NF_REPAIR_BENCH_RECORD") {
        let bundle: Value = serde_json::from_str(&bytes).unwrap();
        check(&bundle["task"], &bundle["result"]).unwrap();
    }
}
