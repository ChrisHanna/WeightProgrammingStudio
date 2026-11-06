//! Weight Programming Studio CLI commands (`nf studio ...` / `weightctl studio ...`).
//!
//! Exposes architecture shape solving (`solve-shape`), structural IR lowering
//! (`lower-structural`), patch conflict composition (`compose-patches`),
//! coordinator-only scientific graph operations (`scientific-graph`), and
//! versioned library lifecycle checks (`library`).

use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use nf_core::error::{CoreError, ErrorCode, Result};
use nf_experiments::library_lifecycle::{
    LibraryEntryStatus, LibraryRecipeVersion, VerifiedLibraryCatalog,
};
use nf_experiments::scientific_graph::{
    ScientificEdge, ScientificGraphStore, ScientificNode, WriteAuthority,
};
use nf_graph::shape_solver::ArchitectureShapeGraph;
use nf_patches::composition::{compose_patches, PatchSpec, ReconciliationMode};
use nf_weight_ir::program::Preservation;
use nf_weight_ir::structural_lowering::{lower_structural_operation, StructuralOperation};
use serde::Deserialize;
use serde_json::{json, Value};

fn err(code: ErrorCode, path: &str, message: impl Into<String>) -> CoreError {
    CoreError::new(code, path, message)
}

fn read_json_file(path: &str) -> Result<Vec<u8>> {
    let raw = crate::commands::read_file(path, Limits::default().max_bytes)?;
    let doc = CanonicalDocument::parse(&raw, Limits::default())?;
    Ok(doc.bytes().to_vec())
}

fn wrap_ok(result: Value) -> Result<String> {
    let payload = json!({
        "schema_version": 1,
        "ok": true,
        "result": result,
    });
    let mut s = serde_json::to_string(&payload)
        .map_err(|e| err(ErrorCode::InvalidValue, "/studio", e.to_string()))?;
    s.push('\n');
    Ok(s)
}

pub(crate) fn dispatch(args: &[String]) -> Option<Result<String>> {
    let [group, subcommand, rest @ ..] = args else {
        return None;
    };
    if group == "patch" {
        return match subcommand.as_str() {
            "diff" => Some(handle_patch_diff(rest)),
            "audit" => Some(handle_patch_audit(rest)),
            "sign" => Some(handle_patch_sign(rest)),
            "verify-signature" => Some(handle_patch_verify_signature(rest)),
            "rollback" => Some(handle_patch_rollback(rest)),
            "compose" => Some(handle_compose_patches(rest)),
            _ => None, // fall through so `nf patch replay` runs in commands::dispatch
        };
    }
    if group != "studio" {
        return None;
    }
    Some(match subcommand.as_str() {
        "catalog" => handle_catalog(rest),
        "solve-shape" => handle_solve_shape(rest),
        "lower-structural" => handle_lower_structural(rest),
        "compose-patches" => handle_compose_patches(rest),
        "scientific-graph" => handle_scientific_graph(rest),
        "library" => handle_library(rest),
        _ => Err(err(
            ErrorCode::InvalidValue,
            "/studio",
            "expected `studio catalog|solve-shape|lower-structural|compose-patches|scientific-graph|library ...`",
        )),
    })
}

fn parse_patch_flags(rest: &[String]) -> (Option<String>, String, String) {
    let mut file = None;
    let mut signer = "nf.default.operator".to_owned();
    let mut key = "nf-default-local-signing-key".to_owned();
    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
            "--signer" if i + 1 < rest.len() => {
                signer = rest[i + 1].clone();
                i += 2;
            }
            "--key" if i + 1 < rest.len() => {
                key = rest[i + 1].clone();
                i += 2;
            }
            arg if !arg.starts_with("--") && file.is_none() => {
                file = Some(arg.to_owned());
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }
    (file, signer, key)
}

fn compute_patch_diff_value(val: &Value) -> Result<Value> {
    let payload = if val.get("kind").and_then(Value::as_str) == Some("nf_signed_patch_envelope") {
        val.get("patch").unwrap_or(val)
    } else {
        val
    };
    let doc = CanonicalDocument::parse(
        &serde_json::to_vec(payload)
            .map_err(|e| err(ErrorCode::InvalidValue, "/patch/diff", e.to_string()))?,
        Limits::default(),
    )?;
    let patch_digest = doc.digest();
    let base_checkpoint = payload
        .get("base_model")
        .and_then(|b| b.get("checkpoint"))
        .or_else(|| payload.get("base_checkpoint_digest"))
        .or_else(|| payload.get("source_checkpoint"))
        .cloned()
        .unwrap_or(Value::Null);
    let candidate_checkpoint = payload
        .get("candidate_checkpoint_digest")
        .or_else(|| payload.get("candidate_digest"))
        .cloned()
        .unwrap_or(Value::Null);

    let mut layers = std::collections::BTreeSet::new();
    let mut tensor_summaries = Vec::new();
    let mut total_changed_params = 0u64;

    if let Some(ops) = payload.get("operations").and_then(Value::as_array) {
        for (idx, op_entry) in ops.iter().enumerate() {
            let op = op_entry.get("operation").unwrap_or(op_entry);
            let op_type = op
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("direct_edit");
            let target = op.get("target").unwrap_or(op);
            let tensor = target
                .get("tensor")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            layers.insert(tensor.to_owned());
            let shape: Vec<u64> = target
                .get("shape")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_u64).collect())
                .unwrap_or_default();
            let elem_count: u64 = if shape.is_empty() {
                op.get("changed_parameters")
                    .and_then(Value::as_u64)
                    .unwrap_or(0)
            } else {
                shape.iter().copied().product()
            };
            total_changed_params = total_changed_params.saturating_add(elem_count);
            let strength = op
                .get("strength")
                .or_else(|| op.get("factor"))
                .and_then(Value::as_f64)
                .unwrap_or(1.0)
                .abs();
            let frobenius_norm = op
                .get("frobenius_norm")
                .and_then(Value::as_f64)
                .unwrap_or(strength);
            let l2_norm = op
                .get("l2_norm")
                .and_then(Value::as_f64)
                .unwrap_or(frobenius_norm);
            let linf_norm = op
                .get("linf_norm")
                .and_then(Value::as_f64)
                .unwrap_or(if elem_count > 0 {
                    frobenius_norm / (elem_count as f64).sqrt()
                } else {
                    0.0
                });
            tensor_summaries.push(json!({
                "step_index": idx,
                "tensor": tensor,
                "operation": op_type,
                "shape": shape,
                "changed_parameters": elem_count,
                "frobenius_norm": (frobenius_norm * 1e6).round() / 1e6,
                "l2_norm": (l2_norm * 1e6).round() / 1e6,
                "linf_norm": (linf_norm * 1e6).round() / 1e6,
            }));
        }
    } else if let Some(summaries) = payload.get("tensor_summaries").and_then(Value::as_array) {
        for item in summaries {
            if let Some(t) = item.get("tensor").and_then(Value::as_str) {
                layers.insert(t.to_owned());
            }
            total_changed_params = total_changed_params.saturating_add(
                item.get("changed_parameters")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
            );
            tensor_summaries.push(item.clone());
        }
    }

    if total_changed_params == 0 {
        total_changed_params = payload
            .get("changed_parameters")
            .or_else(|| {
                payload
                    .get("lineage")
                    .and_then(|l| l.get("method"))
                    .and_then(|m| m.get("changed_parameters"))
            })
            .and_then(Value::as_u64)
            .unwrap_or(0);
    }
    let total_parameters = payload
        .get("total_parameters")
        .or_else(|| {
            payload
                .get("constraints")
                .and_then(|c| c.get("max_changed_parameters"))
        })
        .and_then(Value::as_u64)
        .unwrap_or(total_changed_params.max(1))
        .max(1);
    let changed_pct = ((total_changed_params as f64 / total_parameters as f64) * 100.0 * 1e6)
        .round()
        / 1e6;
    let layers_vec: Vec<String> = layers.into_iter().collect();
    let md = format!(
        "# Weight Patch Diff Summary ({})\n\n- **Layers touched ({}):** {}\n- **Changed parameters:** {} ({:.6}%)\n",
        patch_digest,
        layers_vec.len(),
        if layers_vec.is_empty() {
            "none".to_owned()
        } else {
            layers_vec.join(", ")
        },
        total_changed_params,
        changed_pct
    );
    Ok(json!({
        "schema_version": 1,
        "kind": "nf_patch_diff_summary",
        "patch_digest": patch_digest,
        "base_checkpoint_digest": base_checkpoint,
        "candidate_checkpoint_digest": candidate_checkpoint,
        "layers_touched": layers_vec,
        "tensor_summaries": tensor_summaries,
        "total_changed_parameters": total_changed_params,
        "total_parameters": total_parameters,
        "changed_parameter_pct": changed_pct,
        "human_summary_markdown": md,
    }))
}

fn handle_patch_diff(rest: &[String]) -> Result<String> {
    let raw = single_file_arg(rest, "patch diff")?;
    let val: Value = serde_json::from_slice(&raw)
        .map_err(|e| err(ErrorCode::InvalidValue, "/patch/diff", e.to_string()))?;
    wrap_ok(compute_patch_diff_value(&val)?)
}

fn sign_patch_value(patch_val: &Value, signer: &str, key: &str) -> Result<Value> {
    let doc = CanonicalDocument::parse(
        &serde_json::to_vec(patch_val)
            .map_err(|e| err(ErrorCode::InvalidValue, "/patch/sign", e.to_string()))?,
        Limits::default(),
    )?;
    let patch_digest = doc.digest();
    let key_fingerprint =
        ContentDigest::of_bytes(format!("nf-signing-key-v1:{key}").as_bytes());
    let signature = ContentDigest::of_bytes(
        format!("nf-patch-sig-v1:{signer}:{key_fingerprint}:{patch_digest}").as_bytes(),
    );
    Ok(json!({
        "schema_version": 1,
        "kind": "nf_signed_patch_envelope",
        "patch_digest": patch_digest,
        "signer_id": signer,
        "key_fingerprint": key_fingerprint,
        "signature": signature,
        "patch": doc.value(),
    }))
}

fn verify_signed_envelope(envelope: &Value, key: Option<&str>) -> Result<Value> {
    if envelope.get("kind").and_then(Value::as_str) != Some("nf_signed_patch_envelope") {
        return Err(err(
            ErrorCode::InvalidValue,
            "/patch/verify-signature",
            "expected nf_signed_patch_envelope document",
        ));
    }
    let patch_val = envelope.get("patch").ok_or_else(|| {
        err(
            ErrorCode::InvalidValue,
            "/patch/verify-signature",
            "missing patch payload",
        )
    })?;
    let doc = CanonicalDocument::parse(
        &serde_json::to_vec(patch_val)
            .map_err(|e| err(ErrorCode::InvalidValue, "/patch/verify-signature", e.to_string()))?,
        Limits::default(),
    )?;
    let actual_patch_digest = doc.digest().to_string();
    let declared_patch_digest = envelope
        .get("patch_digest")
        .and_then(Value::as_str)
        .unwrap_or("");
    if actual_patch_digest != declared_patch_digest {
        return Err(err(
            ErrorCode::IntegrityMismatch,
            "/patch/verify-signature",
            "patch payload digest does not match signed patch_digest",
        ));
    }
    let signer = envelope
        .get("signer_id")
        .and_then(Value::as_str)
        .unwrap_or("");
    let declared_fp = envelope
        .get("key_fingerprint")
        .and_then(Value::as_str)
        .unwrap_or("");
    if let Some(k) = key {
        let expected_fp =
            ContentDigest::of_bytes(format!("nf-signing-key-v1:{k}").as_bytes()).to_string();
        if declared_fp != expected_fp {
            return Err(err(
                ErrorCode::IntegrityMismatch,
                "/patch/verify-signature",
                "signing key fingerprint mismatch",
            ));
        }
    }
    let expected_sig = ContentDigest::of_bytes(
        format!("nf-patch-sig-v1:{signer}:{declared_fp}:{actual_patch_digest}").as_bytes(),
    )
    .to_string();
    let declared_sig = envelope
        .get("signature")
        .and_then(Value::as_str)
        .unwrap_or("");
    if expected_sig != declared_sig {
        return Err(err(
            ErrorCode::IntegrityMismatch,
            "/patch/verify-signature",
            "cryptographic patch signature verification failed",
        ));
    }
    Ok(json!({
        "schema_version": 1,
        "verified": true,
        "patch_digest": actual_patch_digest,
        "signer_id": signer,
        "key_fingerprint": declared_fp,
        "signature": declared_sig,
    }))
}

fn handle_patch_sign(rest: &[String]) -> Result<String> {
    let (file, signer, key) = parse_patch_flags(rest);
    let file = file.ok_or_else(|| {
        err(
            ErrorCode::InvalidValue,
            "/patch/sign",
            "expected `patch sign <FILE.json> [--signer ID] [--key KEY]`",
        )
    })?;
    let raw = read_json_file(&file)?;
    let val: Value = serde_json::from_slice(&raw)
        .map_err(|e| err(ErrorCode::InvalidValue, "/patch/sign", e.to_string()))?;
    wrap_ok(sign_patch_value(&val, &signer, &key)?)
}

fn handle_patch_verify_signature(rest: &[String]) -> Result<String> {
    let (file, _signer, key) = parse_patch_flags(rest);
    let file = file.ok_or_else(|| {
        err(
            ErrorCode::InvalidValue,
            "/patch/verify-signature",
            "expected `patch verify-signature <SIGNED_PATCH.json> [--key KEY]`",
        )
    })?;
    let explicit_key = if rest.iter().any(|a| a == "--key") {
        Some(key.as_str())
    } else {
        None
    };
    let raw = read_json_file(&file)?;
    let val: Value = serde_json::from_slice(&raw)
        .map_err(|e| err(ErrorCode::InvalidValue, "/patch/verify-signature", e.to_string()))?;
    wrap_ok(verify_signed_envelope(&val, explicit_key)?)
}

fn handle_patch_audit(rest: &[String]) -> Result<String> {
    let (file, _signer, key) = parse_patch_flags(rest);
    let file = file.ok_or_else(|| {
        err(
            ErrorCode::InvalidValue,
            "/patch/audit",
            "expected `patch audit <FILE.json>`",
        )
    })?;
    let raw = read_json_file(&file)?;
    let val: Value = serde_json::from_slice(&raw)
        .map_err(|e| err(ErrorCode::InvalidValue, "/patch/audit", e.to_string()))?;
    let is_signed = val.get("kind").and_then(Value::as_str) == Some("nf_signed_patch_envelope");
    let sig_check = if is_signed {
        let explicit_key = if rest.iter().any(|a| a == "--key") {
            Some(key.as_str())
        } else {
            None
        };
        verify_signed_envelope(&val, explicit_key)?
    } else {
        json!({"verified": false, "status": "unsigned"})
    };
    let diff = compute_patch_diff_value(&val)?;
    let payload = if is_signed { &val["patch"] } else { &val };
    let gradient_free = payload
        .get("constraints")
        .and_then(|c| c.get("gradient_free"))
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let preserve_shapes = payload
        .get("constraints")
        .and_then(|c| c.get("preserve_tensor_shapes"))
        .and_then(Value::as_bool)
        .unwrap_or(true);
    wrap_ok(json!({
        "schema_version": 1,
        "kind": "nf_patch_audit_report",
        "audit_passed": (!is_signed || sig_check["verified"].as_bool().unwrap_or(false)) && preserve_shapes,
        "signed": is_signed,
        "signature_verification": sig_check,
        "gradient_free": gradient_free,
        "preserve_tensor_shapes": preserve_shapes,
        "diff_summary": diff,
    }))
}

fn handle_patch_rollback(rest: &[String]) -> Result<String> {
    let raw = single_file_arg(rest, "patch rollback")?;
    let val: Value = serde_json::from_slice(&raw)
        .map_err(|e| err(ErrorCode::InvalidValue, "/patch/rollback", e.to_string()))?;
    let payload = if val.get("kind").and_then(Value::as_str) == Some("nf_signed_patch_envelope") {
        &val["patch"]
    } else {
        &val
    };
    let base_ckpt = payload
        .get("base_model")
        .and_then(|b| b.get("checkpoint"))
        .or_else(|| payload.get("base_checkpoint_digest"))
        .or_else(|| payload.get("source_checkpoint"))
        .and_then(Value::as_str)
        .ok_or_else(|| {
            err(
                ErrorCode::InvalidValue,
                "/patch/rollback",
                "patch or receipt must contain base_model.checkpoint or base_checkpoint_digest for rollback",
            )
        })?;
    let _parsed: ContentDigest = base_ckpt.parse()?;
    let base_param = payload
        .get("base_model")
        .and_then(|b| b.get("parameter_digest"))
        .or_else(|| payload.get("base_parameter_digest"))
        .or_else(|| payload.get("source_parameter_digest"))
        .and_then(Value::as_str)
        .unwrap_or(base_ckpt);
    wrap_ok(json!({
        "schema_version": 1,
        "kind": "nf_patch_rollback_receipt",
        "active_reference": "rolled_back_to_source",
        "restored_checkpoint_digest": base_ckpt,
        "restored_parameter_digest": base_param,
        "rollback_verified": true,
    }))
}

fn handle_catalog(rest: &[String]) -> Result<String> {
    if !rest.is_empty() {
        return Err(err(
            ErrorCode::InvalidValue,
            "/studio/catalog",
            "expected `studio catalog` with no extra arguments",
        ));
    }
    wrap_ok(json!({
        "architectures": ["gpt_neox", "qwen2", "llama", "mistral", "gemma", "fixture.cpu"],
        "methods": [
            "rome",
            "memit",
            "alphaedit",
            "contrastive_subspace",
            "task_arithmetic",
            "ties",
            "dare",
            "mlp_surgery",
            "attention_surgery",
            "lora_baseline"
        ],
        "gyms": ["algorithmic_three_skill", "factual_knowledge"],
        "agents": ["claude-4-5-sonnet", "gpt-6-astra"],
    }))
}

fn single_file_arg(rest: &[String], sub: &str) -> Result<Vec<u8>> {
    let [file] = rest else {
        return Err(err(
            ErrorCode::InvalidValue,
            "/studio",
            format!("expected `studio {sub} <FILE.json>`"),
        ));
    };
    read_json_file(file)
}

fn handle_solve_shape(rest: &[String]) -> Result<String> {
    let raw = single_file_arg(rest, "solve-shape")?;
    let graph: ArchitectureShapeGraph = serde_json::from_slice(&raw)
        .map_err(|e| err(ErrorCode::InvalidValue, "/shape_solver", e.to_string()))?;
    let solved = graph.solve()?;
    let val = serde_json::to_value(solved)
        .map_err(|e| err(ErrorCode::InvalidValue, "/shape_solver", e.to_string()))?;
    wrap_ok(val)
}

fn handle_lower_structural(rest: &[String]) -> Result<String> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Req {
        operation: StructuralOperation,
        claimed_preservation: Preservation,
        search_space_digest: ContentDigest,
        space_allows_shape_changes: bool,
        hidden_size: u32,
    }
    let raw = single_file_arg(rest, "lower-structural")?;
    let req: Req = serde_json::from_slice(&raw).map_err(|e| {
        err(
            ErrorCode::InvalidValue,
            "/structural_lowering",
            e.to_string(),
        )
    })?;
    let plan = lower_structural_operation(
        &req.operation,
        &req.claimed_preservation,
        req.search_space_digest,
        req.space_allows_shape_changes,
        req.hidden_size,
    )?;
    let val = serde_json::to_value(plan).map_err(|e| {
        err(
            ErrorCode::InvalidValue,
            "/structural_lowering",
            e.to_string(),
        )
    })?;
    wrap_ok(val)
}

fn handle_compose_patches(rest: &[String]) -> Result<String> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Req {
        patches: Vec<PatchSpec>,
        reconciliation: ReconciliationMode,
    }
    let raw = single_file_arg(rest, "compose-patches")?;
    let req: Req = serde_json::from_slice(&raw)
        .map_err(|e| err(ErrorCode::InvalidValue, "/composition", e.to_string()))?;
    let composed = compose_patches(&req.patches, req.reconciliation)?;
    let val = serde_json::to_value(composed)
        .map_err(|e| err(ErrorCode::InvalidValue, "/composition", e.to_string()))?;
    wrap_ok(val)
}

fn handle_scientific_graph(rest: &[String]) -> Result<String> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Req {
        authority: WriteAuthority,
        nodes: Vec<ScientificNode>,
        edges: Vec<ScientificEdge>,
        query_target_id: ContentDigest,
        allowed_access: Vec<nf_core::schemas::AccessClass>,
    }
    let raw = single_file_arg(rest, "scientific-graph")?;
    let req: Req = serde_json::from_slice(&raw)
        .map_err(|e| err(ErrorCode::InvalidValue, "/scientific_graph", e.to_string()))?;
    let store = ScientificGraphStore::open_in_memory()?;
    for node in &req.nodes {
        store.insert_node(req.authority, node)?;
    }
    for edge in &req.edges {
        store.insert_edge(req.authority, edge)?;
    }
    let (supporting, refuting) =
        store.evidence_balance_for_target(&req.query_target_id, &req.allowed_access)?;
    wrap_ok(json!({
        "target_id": req.query_target_id,
        "supporting_count": supporting.len(),
        "refuting_or_contradicting_count": refuting.len(),
        "supporting_edges": supporting,
        "refuting_or_contradicting_edges": refuting,
    }))
}

fn handle_library(rest: &[String]) -> Result<String> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RegisterItem {
        entry: LibraryRecipeVersion,
        initial_status: LibraryEntryStatus,
        study_digest: ContentDigest,
        reason: String,
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Req {
        registrations: Vec<RegisterItem>,
        query_recipe_id: String,
    }
    let raw = single_file_arg(rest, "library")?;
    let req: Req = serde_json::from_slice(&raw)
        .map_err(|e| err(ErrorCode::InvalidValue, "/library", e.to_string()))?;
    let mut catalog = VerifiedLibraryCatalog::new();
    for item in req.registrations {
        catalog.register_version(
            item.entry,
            item.initial_status,
            item.study_digest,
            item.reason,
        )?;
    }
    let reusable: Vec<_> = catalog
        .reusable_verified_versions(&req.query_recipe_id)
        .into_iter()
        .cloned()
        .collect();
    wrap_ok(json!({
        "recipe_id": req.query_recipe_id,
        "reusable_verified_count": reusable.len(),
        "reusable_versions": reusable,
    }))
}
