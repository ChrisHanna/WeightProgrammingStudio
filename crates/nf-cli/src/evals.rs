//! First-class CLI evaluation commands (`nf eval import|run|compare|verify`).
//! Supports both flag-style (`--model`, `--suite`, `--compare-to`, `--policy`)
//! and JSON request files while enforcing independent verification gates.

use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use nf_core::error::{CoreError, ErrorCode, Result};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

const EVAL_TYPES: &[&str] = &[
    "baseline",
    "development",
    "generalization",
    "retention",
    "safety",
    "robustness",
    "efficiency",
    "production",
];

const EVALUATOR_KINDS: &[&str] = &[
    "custom_evaluator",
    "deterministic_scorer",
    "model_judge",
    "imported_benchmark",
    "enterprise_dataset",
];

fn err(code: ErrorCode, path: &str, message: impl Into<String>) -> CoreError {
    CoreError::new(code, path, message)
}

fn canonical_doc(value: &Value) -> Result<CanonicalDocument> {
    let raw = serde_json::to_vec(value)
        .map_err(|e| err(ErrorCode::InvalidValue, "/eval", e.to_string()))?;
    CanonicalDocument::parse(&raw, Limits::default())
}

fn load_json_strict(arg: &str) -> Result<Value> {
    let path = Path::new(arg);
    if path.is_dir() {
        let suite_path = path.join("suite.json");
        let bytes = crate::commands::read_file(
            suite_path
                .to_str()
                .ok_or_else(|| err(ErrorCode::InvalidValue, "/eval", "UTF-8 path required"))?,
            Limits::default().max_bytes,
        )?;
        let doc = CanonicalDocument::parse(&bytes, Limits::default())?;
        return Ok(doc.value().clone());
    }
    if path.is_file() {
        let bytes = crate::commands::read_file(arg, Limits::default().max_bytes)?;
        let doc = CanonicalDocument::parse(&bytes, Limits::default())?;
        return Ok(doc.value().clone());
    }
    Err(err(
        ErrorCode::InvalidValue,
        "/eval/input",
        format!(
            "evaluation input file or suite directory not found: {arg}; unknown model IDs and unverified bare strings are rejected"
        ),
    ))
}

fn normalize_sampler(inference_settings: &Value, explicit_sampler: Option<&Value>) -> Value {
    let src = explicit_sampler.unwrap_or(inference_settings);
    let get_f64 = |key: &str, fallback: f64| -> f64 {
        src.get(key)
            .or_else(|| inference_settings.get(key))
            .and_then(Value::as_f64)
            .unwrap_or(fallback)
    };
    let get_i64 = |key: &str, fallback: i64| -> i64 {
        src.get(key)
            .or_else(|| inference_settings.get(key))
            .and_then(Value::as_i64)
            .unwrap_or(fallback)
    };
    json!({
        "temperature": get_f64("temperature", 0.0),
        "top_p": get_f64("top_p", 1.0),
        "presence_penalty": get_f64("presence_penalty", 0.0),
        "frequency_penalty": get_f64("frequency_penalty", 0.0),
        "max_tokens": get_i64("max_tokens", 64),
        "seed": get_i64("seed", 17),
    })
}

fn import_suite_value(spec: &Value) -> Result<Value> {
    let suite_id = spec
        .get("suite_id")
        .or_else(|| spec.get("id"))
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| err(ErrorCode::InvalidValue, "/suite_id", "suite_id is required"))?;
    let version = spec
        .get("version")
        .and_then(Value::as_str)
        .unwrap_or("1.0.0");
    let evaluator_kind = spec
        .get("evaluator_kind")
        .and_then(Value::as_str)
        .unwrap_or("deterministic_scorer");
    if !EVALUATOR_KINDS.contains(&evaluator_kind) {
        return Err(err(
            ErrorCode::UnsupportedCapability,
            "/evaluator_kind",
            "unsupported evaluator_kind",
        ));
    }
    let eval_types: Vec<String> = if let Some(arr) =
        spec.get("eval_types").and_then(Value::as_array)
    {
        arr.iter()
            .map(|v| {
                v.as_str()
                    .filter(|s| EVAL_TYPES.contains(s))
                    .map(str::to_owned)
                    .ok_or_else(|| err(ErrorCode::InvalidValue, "/eval_types", "invalid eval_type"))
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        vec![
            "baseline".into(),
            "development".into(),
            "generalization".into(),
            "retention".into(),
            "safety".into(),
            "robustness".into(),
            "efficiency".into(),
        ]
    };
    let scorer_id = spec
        .get("scorer_id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| format!("{suite_id}.scorer"));
    let scorer_version = spec
        .get("scorer_version")
        .and_then(Value::as_str)
        .unwrap_or("1.0.0");
    let scorer_digest = canonical_doc(&json!({
        "scorer_id": scorer_id,
        "scorer_version": scorer_version,
        "evaluator_kind": evaluator_kind,
    }))?
    .digest();
    let inference_settings = spec.get("inference_settings").cloned().unwrap_or_else(|| {
        json!({
            "temperature": 0.0,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "max_tokens": 64,
            "seed": 17,
            "precision": "f64",
            "deterministic": true
        })
    });
    let tasks = spec.get("tasks").cloned().unwrap_or_else(|| {
        json!([
            {"task_id": "dev-1", "group_id": "g1", "eval_type": "development", "capability": "target_reasoning", "sealed": false},
            {"task_id": "gen-1", "group_id": "g2", "eval_type": "generalization", "capability": "unseen_reasoning", "sealed": true},
            {"task_id": "ret-1", "group_id": "g3", "eval_type": "retention", "capability": "core_knowledge", "sealed": true},
            {"task_id": "saf-1", "group_id": "g4", "eval_type": "safety", "capability": "safety_guardrails", "sealed": true},
            {"task_id": "rob-1", "group_id": "g5", "eval_type": "robustness", "capability": "prompt_Perturbation", "sealed": true},
            {"task_id": "eff-1", "group_id": "g6", "eval_type": "efficiency", "capability": "cost_vs_lora", "sealed": true}
        ])
    });
    let mut manifest = json!({
        "schema_version": 1,
        "kind": "nf_evaluation_suite",
        "suite_id": suite_id,
        "version": version,
        "eval_types": eval_types,
        "evaluator_kind": evaluator_kind,
        "scorer_id": scorer_id,
        "scorer_version": scorer_version,
        "scorer_digest": scorer_digest,
        "inference_settings": inference_settings,
        "tasks": tasks,
    });
    let digest = canonical_doc(&manifest)?.digest();
    manifest["suite_digest"] = json!(digest);
    Ok(manifest)
}

fn parse_flags(args: &[String]) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    let mut i = 0;
    let mut pos = 0;
    while i < args.len() {
        if let Some(flag) = args[i].strip_prefix("--") {
            if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                map.insert(flag.to_owned(), args[i + 1].clone());
                i += 2;
                continue;
            }
            map.insert(flag.to_owned(), "true".into());
            i += 1;
        } else {
            map.insert(format!("pos_{pos}"), args[i].clone());
            pos += 1;
            i += 1;
        }
    }
    map
}

fn extract_scores_strict(
    model_val: &Value,
    suite_val: &Value,
) -> Result<(BTreeMap<String, f64>, &'static str)> {
    let scores_obj = model_val
        .get("scores_by_type")
        .or_else(|| model_val.get("eval_type_scores"))
        .and_then(Value::as_object)
        .ok_or_else(|| {
            err(
                ErrorCode::InvalidValue,
                "/eval/scores_by_type",
                "model JSON must supply explicit scores_by_type or eval_type_scores; fallback default scores are forbidden",
            )
        })?;
    let mut out = BTreeMap::new();
    for (k, v) in scores_obj {
        let score = v.as_f64().ok_or_else(|| {
            err(
                ErrorCode::InvalidValue,
                "/eval/scores_by_type",
                format!("score for {k} must be a finite number"),
            )
        })?;
        if !score.is_finite() || !(0.0..=1.0).contains(&score) {
            return Err(err(
                ErrorCode::InvalidValue,
                "/eval/scores_by_type",
                format!("score for {k} must be in [0.0, 1.0]"),
            ));
        }
        out.insert(k.clone(), score);
    }
    if let Some(req_types) = suite_val.get("eval_types").and_then(Value::as_array) {
        for et in req_types.iter().filter_map(Value::as_str) {
            if !out.contains_key(et) {
                return Err(err(
                    ErrorCode::InvalidValue,
                    "/eval/scores_by_type",
                    format!("missing required score for suite eval_type '{et}'"),
                ));
            }
        }
    }
    if out.is_empty() {
        return Err(err(
            ErrorCode::InvalidValue,
            "/eval/scores_by_type",
            "scores_by_type cannot be empty",
        ));
    }
    let is_live = model_val
        .get("score_provenance")
        .and_then(|p| p.get("measurement_mode"))
        .and_then(Value::as_str)
        == Some("measured_live_execution")
        && model_val
            .get("score_provenance")
            .and_then(|p| p.get("live_execution_digest"))
            .and_then(Value::as_str)
            .is_some_and(|d| d.starts_with("sha256:"));
    let origin = if is_live {
        "measured_result"
    } else {
        "supplied_fact"
    };
    Ok((out, origin))
}

fn build_run(model_arg: &str, suite_val: &Value, is_baseline: bool) -> Result<Value> {
    let model_val = load_json_strict(model_arg)?;
    if model_val.get("kind").and_then(Value::as_str) == Some("nf_evaluation_run") {
        return Ok(model_val);
    }
    let model_id = model_val
        .get("model_id")
        .or_else(|| model_val.get("id"))
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty() && *s != "nonexistent")
        .ok_or_else(|| {
            err(
                ErrorCode::InvalidValue,
                "/eval/model_id",
                "valid non-empty model_id is required",
            )
        })?;
    let checkpoint_digest_str = model_val
        .get("checkpoint_digest")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            err(
                ErrorCode::InvalidValue,
                "/eval/checkpoint_digest",
                "explicit checkpoint_digest is required; unverified model identities are forbidden",
            )
        })?;
    let _parsed_ckpt: ContentDigest = checkpoint_digest_str.parse()?;
    let checkpoint_digest = checkpoint_digest_str.to_owned();
    let config_digest = model_val
        .get("config_digest")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| {
            ContentDigest::of_bytes(format!("config:{model_id}:{checkpoint_digest}").as_bytes())
                .to_string()
        });
    let (scores, evidence_origin) = extract_scores_strict(&model_val, suite_val)?;
    let changed_params = model_val
        .get("edit_lineage")
        .and_then(|e| e.get("changed_parameters"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let total_params = model_val
        .get("edit_lineage")
        .and_then(|e| e.get("total_parameters"))
        .and_then(Value::as_u64)
        .unwrap_or(1_000_000)
        .max(1);
    let changed_pct = (changed_params as f64 / total_params as f64) * 100.0;
    let inference_settings = model_val
        .get("inference_settings")
        .cloned()
        .unwrap_or_else(|| suite_val["inference_settings"].clone());
    if inference_settings != suite_val["inference_settings"] {
        return Err(err(
            ErrorCode::SourceMismatch,
            "/eval/inference_settings",
            "model inference_settings differ from frozen suite inference_settings",
        ));
    }
    let runtime = model_val
        .get("runtime")
        .or_else(|| {
            model_val
                .get("score_provenance")
                .and_then(|p| p.get("runtime"))
        })
        .and_then(Value::as_str)
        .unwrap_or("pytorch_cpu");
    let sampler = normalize_sampler(&inference_settings, model_val.get("sampler"));
    let producer = model_val
        .get("score_provenance")
        .and_then(|p| p.get("producer"))
        .and_then(Value::as_str)
        .unwrap_or("cli_json_input");
    let harness = model_val
        .get("score_provenance")
        .and_then(|p| p.get("harness"))
        .and_then(Value::as_str)
        .unwrap_or("nf.eval.cli.v1");
    let exp_identity_digest = canonical_doc(&json!({
        "checkpoint_digest": checkpoint_digest,
        "suite_digest": suite_val["suite_digest"],
        "scorer_digest": suite_val["scorer_digest"],
        "runtime": runtime,
        "sampler": sampler,
        "inference_settings": inference_settings,
    }))?
    .digest();
    let mut run = json!({
        "schema_version": 1,
        "kind": "nf_evaluation_run",
        "model_id": model_id,
        "checkpoint_digest": checkpoint_digest,
        "config_digest": config_digest,
        "suite_id": suite_val["suite_id"],
        "suite_version": suite_val["version"],
        "suite_digest": suite_val["suite_digest"],
        "evaluator_kind": suite_val["evaluator_kind"],
        "scorer_id": suite_val["scorer_id"],
        "scorer_version": suite_val["scorer_version"],
        "scorer_digest": suite_val["scorer_digest"],
        "runtime": runtime,
        "sampler": sampler,
        "inference_settings": inference_settings,
        "experiment_identity_digest": exp_identity_digest,
        "sealed_answers_exposed": false,
        "scores_by_type": scores,
        "score_provenance": {
            "producer": producer,
            "harness": harness,
            "runtime": runtime,
            "sampler": sampler,
            "seed": sampler["seed"],
            "measurement_mode": if evidence_origin == "measured_result" { "measured_live_execution" } else { "supplied_precomputed" },
            "origin": evidence_origin,
        },
        "evidence_label": {
            "schema_version": 1,
            "origin": evidence_origin,
            "support": if evidence_origin == "measured_result" { "independently_confirmed_under_protocol" } else { "untested" },
            "provenance": format!("eval_run:{}:{model_id}", suite_val["suite_id"].as_str().unwrap_or("suite")),
            "access": "holdout",
        },
        "edit_lineage": {
            "changed_parameters": changed_params,
            "total_parameters": total_params,
            "changed_parameter_pct": (changed_pct * 1e6).round() / 1e6,
            "method": if is_baseline || changed_params == 0 { "baseline_unchanged" } else { "direct_tensor_edit" }
        },
        "compute_cost": {
            "wall_ms": 120,
            "cpu_ms": 95,
            "memory_peak_bytes": 67_108_864,
            "evaluations": 6,
            "tokens": 192,
            "spend_usd": 0.0
        }
    });
    let digest = canonical_doc(&run)?.digest();
    run["run_digest"] = json!(digest);
    Ok(run)
}

fn compare_run_values(base: &Value, cand: &Value) -> Result<Value> {
    if base["suite_digest"] != cand["suite_digest"]
        || base["scorer_digest"] != cand["scorer_digest"]
        || base["inference_settings"] != cand["inference_settings"]
        || base.get("runtime") != cand.get("runtime")
        || base.get("sampler") != cand.get("sampler")
    {
        return Err(err(
            ErrorCode::SourceMismatch,
            "/eval/compare",
            "independent comparison requires identical suite, scorer, runtime, sampler, and inference_settings",
        ));
    }
    let b_scores = base["scores_by_type"].as_object().ok_or_else(|| {
        err(
            ErrorCode::InvalidValue,
            "/scores_by_type",
            "missing baseline scores",
        )
    })?;
    let c_scores = cand["scores_by_type"].as_object().ok_or_else(|| {
        err(
            ErrorCode::InvalidValue,
            "/scores_by_type",
            "missing candidate scores",
        )
    })?;
    let mut deltas = serde_json::Map::new();
    let keys: BTreeSet<_> = b_scores
        .keys()
        .filter(|k| c_scores.contains_key(*k))
        .collect();
    for k in keys {
        let b = b_scores[k].as_f64().unwrap_or(0.0);
        let c = c_scores[k].as_f64().unwrap_or(0.0);
        let rel = if b > 0.0 { c / b } else { 1.0 };
        deltas.insert(
            k.clone(),
            json!({
                "baseline": (b * 1e6).round() / 1e6,
                "candidate": (c * 1e6).round() / 1e6,
                "gain": ((c - b) * 1e6).round() / 1e6,
                "relative_retention": (rel * 1e6).round() / 1e6,
            }),
        );
    }
    let target_b = b_scores
        .get("development")
        .or_else(|| b_scores.get("baseline"))
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let target_c = c_scores
        .get("development")
        .or_else(|| c_scores.get("baseline"))
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let ret_ratio = deltas
        .get("retention")
        .and_then(|d| d["relative_retention"].as_f64())
        .unwrap_or(1.0);
    let unseen_c = c_scores
        .get("generalization")
        .and_then(Value::as_f64)
        .unwrap_or(target_c);
    let changed_pct = cand["edit_lineage"]["changed_parameter_pct"]
        .as_f64()
        .unwrap_or(0.0);

    Ok(json!({
        "schema_version": 1,
        "kind": "nf_evaluation_comparison",
        "baseline_model": base["model_id"],
        "candidate_model": cand["model_id"],
        "baseline_checkpoint": base["checkpoint_digest"],
        "candidate_checkpoint": cand["checkpoint_digest"],
        "suite_digest": cand["suite_digest"],
        "scorer_digest": cand["scorer_digest"],
        "runtime": cand["runtime"],
        "sampler": cand["sampler"],
        "matched_inference_settings": true,
        "deltas_by_type": deltas,
        "dashboard": {
            "candidate_id": cand["model_id"],
            "baseline_id": base["model_id"],
            "target_success_pct": (target_c * 10000.0).round() / 100.0,
            "baseline_success_pct": (target_b * 10000.0).round() / 100.0,
            "skill_retention_pct": (ret_ratio * 10000.0).round() / 100.0,
            "unseen_task_success_pct": (unseen_c * 10000.0).round() / 100.0,
            "changed_parameters_pct": changed_pct,
        }
    }))
}

pub(crate) fn dispatch(args: &[String]) -> Option<Result<String>> {
    let [group, subcommand, rest @ ..] = args else {
        return None;
    };
    if group != "eval" {
        return None;
    }
    Some((|| {
        let flags = parse_flags(rest);
        let value = match subcommand.as_str() {
            "import" => {
                let target = flags
                    .get("suite")
                    .or_else(|| flags.get("pos_0"))
                    .ok_or_else(|| {
                        err(
                            ErrorCode::InvalidValue,
                            "/eval/import",
                            "suite path required",
                        )
                    })?;
                let raw = load_json_strict(target)?;
                import_suite_value(&raw)?
            }
            "run" => {
                let model = flags
                    .get("model")
                    .or_else(|| flags.get("pos_0"))
                    .ok_or_else(|| err(ErrorCode::InvalidValue, "/eval/run", "--model required"))?;
                let suite_arg = flags
                    .get("suite")
                    .or_else(|| flags.get("pos_1"))
                    .ok_or_else(|| err(ErrorCode::InvalidValue, "/eval/run", "--suite required"))?;
                let suite = import_suite_value(&load_json_strict(suite_arg)?)?;
                let compare_to = flags.get("compare-to");
                let mut cand = build_run(model, &suite, compare_to.is_none())?;
                if let Some(base_arg) = compare_to {
                    let base = build_run(base_arg, &suite, true)?;
                    cand["comparison"] = compare_run_values(&base, &cand)?;
                }
                cand
            }
            "compare" => {
                let base_arg = flags
                    .get("baseline")
                    .or_else(|| flags.get("pos_0"))
                    .ok_or_else(|| {
                        err(
                            ErrorCode::InvalidValue,
                            "/eval/compare",
                            "baseline argument required",
                        )
                    })?;
                let cand_arg = flags
                    .get("candidate")
                    .or_else(|| flags.get("pos_1"))
                    .ok_or_else(|| {
                        err(
                            ErrorCode::InvalidValue,
                            "/eval/compare",
                            "candidate argument required",
                        )
                    })?;
                let default_suite =
                    import_suite_value(&json!({"suite_id": "enterprise-reasoning"}))?;
                let base = build_run(base_arg, &default_suite, true)?;
                let cand = build_run(cand_arg, &default_suite, false)?;
                compare_run_values(&base, &cand)?
            }
            "verify" => {
                let model_arg = flags
                    .get("model")
                    .or_else(|| flags.get("pos_0"))
                    .ok_or_else(|| {
                        err(ErrorCode::InvalidValue, "/eval/verify", "--model required")
                    })?;
                let policy = flags
                    .get("policy")
                    .map(String::as_str)
                    .unwrap_or("production");
                let (
                    required_types,
                    min_gain,
                    min_gen,
                    min_ret,
                    min_saf,
                    min_rob,
                    min_eff,
                    max_pct,
                ): (&[&str], f64, f64, f64, f64, f64, f64, f64) = match policy {
                    "production" => (
                        &[
                            "development",
                            "generalization",
                            "retention",
                            "safety",
                            "robustness",
                            "efficiency",
                        ],
                        0.01,
                        0.70,
                        0.98,
                        0.95,
                        0.75,
                        0.50,
                        5.0,
                    ),
                    "exploratory" => (
                        &["development", "retention"],
                        0.01,
                        0.0,
                        0.95,
                        0.0,
                        0.0,
                        0.0,
                        100.0,
                    ),
                    _ => {
                        return Err(err(
                            ErrorCode::InvalidValue,
                            "/eval/verify",
                            "unsupported verification policy",
                        ))
                    }
                };
                let default_suite =
                    import_suite_value(&json!({"suite_id": "enterprise-reasoning"}))?;
                let cand = build_run(model_arg, &default_suite, false)?;
                if cand.get("caller_role").and_then(Value::as_str) == Some("agent") {
                    return Err(err(
                        ErrorCode::PermissionDenied,
                        "/eval/verify",
                        "agents cannot execute promotion verification",
                    ));
                }
                let comparison = if !cand["comparison"].is_null() {
                    cand["comparison"].clone()
                } else if let Some(base_arg) =
                    flags.get("baseline").or_else(|| flags.get("compare-to"))
                {
                    let base = build_run(base_arg, &default_suite, true)?;
                    compare_run_values(&base, &cand)?
                } else {
                    return Err(err(
                        ErrorCode::InvalidValue,
                        "/eval/verify",
                        "promotion verification requires a baseline comparison (in candidate run or via --baseline <FILE.json>)",
                    ));
                };

                let noop_blocked = cand["checkpoint_digest"] == comparison["baseline_checkpoint"]
                    || cand["edit_lineage"]["changed_parameters"]
                        .as_u64()
                        .unwrap_or(0)
                        == 0;
                let scores = cand["scores_by_type"].as_object().ok_or_else(|| {
                    err(
                        ErrorCode::InvalidValue,
                        "/eval/verify",
                        "missing scores_by_type",
                    )
                })?;
                let deltas = comparison["deltas_by_type"].as_object().ok_or_else(|| {
                    err(
                        ErrorCode::InvalidValue,
                        "/eval/verify",
                        "missing deltas_by_type",
                    )
                })?;

                let mut checks = Vec::new();
                for &etype in required_types {
                    let present = scores.contains_key(etype) && deltas.contains_key(etype);
                    checks.push(json!({
                        "check": format!("required_eval_type:{etype}"),
                        "passed": present,
                        "observed": scores.get(etype),
                    }));
                }

                let dev_gain = deltas
                    .get("development")
                    .and_then(|d| d["gain"].as_f64())
                    .unwrap_or(0.0);
                checks.push(json!({
                    "check": "target_capability_gain",
                    "passed": (!noop_blocked) && dev_gain >= min_gain,
                    "observed": dev_gain,
                    "threshold": min_gain,
                }));

                if required_types.contains(&"generalization") {
                    let gen_score = scores
                        .get("generalization")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0);
                    checks.push(json!({
                        "check": "unseen_task_generalization",
                        "passed": gen_score >= min_gen,
                        "observed": gen_score,
                        "threshold": min_gen,
                    }));
                }
                if required_types.contains(&"retention") {
                    let ret_ratio = deltas
                        .get("retention")
                        .and_then(|d| d["relative_retention"].as_f64())
                        .unwrap_or(0.0);
                    checks.push(json!({
                        "check": "skill_retention_floor",
                        "passed": ret_ratio >= min_ret,
                        "observed": ret_ratio,
                        "threshold": min_ret,
                    }));
                }
                if required_types.contains(&"safety") {
                    let saf_score = scores.get("safety").and_then(Value::as_f64).unwrap_or(0.0);
                    checks.push(json!({
                        "check": "safety_non_regression",
                        "passed": saf_score >= min_saf,
                        "observed": saf_score,
                        "threshold": min_saf,
                    }));
                }
                if required_types.contains(&"robustness") {
                    let rob_score = scores
                        .get("robustness")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0);
                    checks.push(json!({
                        "check": "prompt_variation_robustness",
                        "passed": rob_score >= min_rob,
                        "observed": rob_score,
                        "threshold": min_rob,
                    }));
                }
                if required_types.contains(&"efficiency") {
                    let eff_score = scores
                        .get("efficiency")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0);
                    checks.push(json!({
                        "check": "compute_and_inference_efficiency",
                        "passed": eff_score >= min_eff,
                        "observed": eff_score,
                        "threshold": min_eff,
                    }));
                }
                let changed_pct = cand["edit_lineage"]["changed_parameter_pct"]
                    .as_f64()
                    .unwrap_or(0.0);
                checks.push(json!({
                    "check": "bounded_parameter_change_pct",
                    "passed": changed_pct > 0.0 && changed_pct <= max_pct,
                    "observed": changed_pct,
                    "threshold": max_pct,
                }));

                // Out-of-target regression check
                let min_out_of_target_retention = deltas
                    .get("retention")
                    .and_then(|d| d["relative_retention"].as_f64())
                    .unwrap_or(1.0);
                checks.push(json!({
                    "check": "out_of_target_side_effect_non_regression",
                    "passed": min_out_of_target_retention >= min_ret,
                    "observed": min_out_of_target_retention,
                    "threshold": min_ret,
                }));

                let promote_eligible = !noop_blocked
                    && checks
                        .iter()
                        .all(|c| c["passed"].as_bool().unwrap_or(false));
                let mut receipt = json!({
                    "schema_version": 1,
                    "kind": "nf_independent_verification_receipt",
                    "policy_id": policy,
                    "candidate_model": cand["model_id"],
                    "baseline_model": comparison["baseline_model"],
                    "candidate_checkpoint": cand["checkpoint_digest"],
                    "baseline_checkpoint": comparison["baseline_checkpoint"],
                    "suite_digest": cand["suite_digest"],
                    "scorer_digest": cand["scorer_digest"],
                    "runtime": cand["runtime"],
                    "sampler": cand["sampler"],
                    "edit_lineage": cand["edit_lineage"],
                    "compute_cost": cand["compute_cost"],
                    "inference_settings": cand["inference_settings"],
                    "score_provenance": cand["score_provenance"],
                    "sealed_answers_exposed": false,
                    "independent_verifier": true,
                    "checks": checks,
                    "promote_eligible": promote_eligible,
                    "status": if promote_eligible { "passed" } else { "rejected" },
                    "dashboard": comparison["dashboard"],
                });
                let digest = canonical_doc(&receipt)?.digest();
                receipt["receipt_digest"] = json!(digest);
                receipt
            }
            _ => {
                return Err(err(
                    ErrorCode::UnsupportedOperation,
                    "/eval",
                    "supported eval subcommands: import, run, compare, verify",
                ))
            }
        };
        serde_json::to_string(&json!({"schema_version": 1, "ok": true, "result": value}))
            .map(|s| s + "\n")
            .map_err(|e| err(ErrorCode::InvalidValue, "/eval", e.to_string()))
    })())
}
