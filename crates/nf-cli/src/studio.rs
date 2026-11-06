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
use nf_patches::signing;
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
    match args {
        [cmd, rest @ ..] if cmd == "verify" => return Some(handle_open_patch_verify(rest)),
        [cmd, rest @ ..] if cmd == "repair" => return Some(handle_top_level_repair(rest)),
        _ => {}
    }
    let [group, subcommand, rest @ ..] = args else {
        return None;
    };
    if group == "patch" {
        return match subcommand.as_str() {
            "diff" => Some(handle_patch_diff(rest)),
            "audit" => Some(handle_patch_audit(rest)),
            "keygen" => Some(handle_patch_keygen(rest)),
            "sign" => Some(handle_patch_sign(rest)),
            "verify" => Some(handle_open_patch_verify(rest)),
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
    let mut key = String::new();
    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
            "--signer" if i + 1 < rest.len() => {
                signer = rest[i + 1].clone();
                i += 2;
            }
            "--key" | "--pubkey" if i + 1 < rest.len() => {
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

fn handle_patch_keygen(rest: &[String]) -> Result<String> {
    if !rest.is_empty() {
        return Err(err(
            ErrorCode::InvalidValue,
            "/patch/keygen",
            "keygen takes no arguments: keys come from the OS CSPRNG and cannot be derived from a seed",
        ));
    }
    let kp = signing::generate_keypair()?;
    wrap_ok(json!({
        "schema_version": 1,
        "scheme": signing::SIGNATURE_SCHEME,
        "private_key": kp.private_key,
        "public_key": kp.public_key,
        "key_fingerprint": kp.key_fingerprint,
    }))
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
    let patch_digest_str = patch_digest.to_string();

    let (signature, public_key) = signing::sign_digest(key, signer, &patch_digest_str)?;
    Ok(json!({
        "schema_version": 1,
        "kind": "nf_signed_patch_envelope",
        "signature_scheme": signing::SIGNATURE_SCHEME,
        "patch_digest": patch_digest,
        "signer_id": signer,
        "public_key": public_key,
        "key_fingerprint": signing::key_fingerprint(&public_key),
        "signature": signature,
        "patch": doc.value(),
    }))
}

fn verify_signed_envelope(envelope: &Value, trusted_public_key: &str) -> Result<Value> {
    if envelope.get("kind").and_then(Value::as_str) != Some("nf_signed_patch_envelope") {
        return Err(err(
            ErrorCode::InvalidValue,
            "/patch/verify-signature",
            "expected nf_signed_patch_envelope document",
        ));
    }
    // Mirrors `patch_ux.verify_patch_signature`: an envelope field nobody signed can't be allowed to mean anything.
    if envelope.get("schema_version").and_then(Value::as_u64) != Some(1) {
        return Err(err(
            ErrorCode::IntegrityMismatch,
            "/patch/verify-signature",
            "unsupported signed patch envelope schema_version",
        ));
    }
    if envelope.as_object().is_some_and(|o| o.keys().any(|k| !ENVELOPE_KEYS.contains(&k.as_str()))) {
        return Err(err(
            ErrorCode::IntegrityMismatch,
            "/patch/verify-signature",
            "signed patch envelope has unknown fields",
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
    let declared_sig = envelope
        .get("signature")
        .and_then(Value::as_str)
        .unwrap_or("");
    if envelope.get("signature_scheme").and_then(Value::as_str) != Some(signing::SIGNATURE_SCHEME) {
        return Err(err(
            ErrorCode::IntegrityMismatch,
            "/patch/verify-signature",
            "unsupported signature_scheme; re-sign with an ed25519_v1 key",
        ));
    }
    signing::verify_digest(trusted_public_key, signer, &actual_patch_digest, declared_sig)?;
    Ok(json!({
        "schema_version": 1,
        "verified": true,
        "signature_scheme": signing::SIGNATURE_SCHEME,
        "patch_digest": actual_patch_digest,
        "signer_id": signer,
        "public_key": trusted_public_key,
        "key_fingerprint": signing::key_fingerprint(trusted_public_key),
    }))
}

const OPEN_PATCH_MAX_BYTES: usize = 64 * 1024 * 1024;
const OPEN_PATCH_FORMAT_VERSION: &str = "1.0.0";
const ENVELOPE_KEYS: [&str; 9] = [
    "schema_version", "kind", "signature_scheme", "patch_digest", "signer_id", "public_key", "key_fingerprint",
    "signature", "patch",
];
const OPEN_PATCH_KEYS: [&str; 11] = [
    "schema_version", "kind", "format_version", "base_model", "candidate_checkpoint_digest",
    "total_parameters", "operations", "sbom", "rounding_contract", "evaluation_provenance", "security_scan",
];

fn is_sha256_digest(value: Option<&Value>) -> bool {
    value
        .and_then(Value::as_str)
        .and_then(|v| v.strip_prefix("sha256:"))
        .is_some_and(|h| h.len() == 64 && h.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)))
}

/// Mirrors `patch_ux.validate_open_patch_schema`.
fn validate_open_patch_schema(payload: &Value) -> Result<()> {
    let fail = |message: &str| err(ErrorCode::IntegrityMismatch, "/verify/schema", message.to_owned());
    let object = payload.as_object().ok_or_else(|| fail("payload must be an object"))?;
    if object.len() != OPEN_PATCH_KEYS.len() || !OPEN_PATCH_KEYS.iter().all(|k| object.contains_key(*k)) {
        return Err(fail("open patch has missing or unknown top-level keys"));
    }
    if payload["kind"].as_str() != Some("nf_open_weight_patch_v1") || payload["schema_version"].as_u64() != Some(1) {
        return Err(fail("unsupported kind or schema_version"));
    }
    if payload["format_version"].as_str() != Some(OPEN_PATCH_FORMAT_VERSION) {
        return Err(fail("unsupported format_version"));
    }
    let base = &payload["base_model"]["parameter_digest"];
    let sbom = &payload["sbom"];
    if !is_sha256_digest(Some(base))
        || !is_sha256_digest(Some(&payload["candidate_checkpoint_digest"]))
        || !is_sha256_digest(Some(&sbom["base_checkpoint_digest"]))
        || !is_sha256_digest(Some(&sbom["candidate_checkpoint_digest"]))
    {
        return Err(fail("digest fields must be sha256 digests"));
    }
    if sbom["base_checkpoint_digest"] != *base || sbom["candidate_checkpoint_digest"] != payload["candidate_checkpoint_digest"] {
        return Err(fail("SBOM digests differ from the patch"));
    }
    let total = payload["total_parameters"].as_u64().unwrap_or(0);
    let changed = sbom["changed_parameters"].as_u64().unwrap_or(0);
    if changed == 0 || changed > total || sbom["total_parameters"].as_u64() != Some(total) {
        return Err(fail("parameter counts are inconsistent"));
    }
    let operations = payload["operations"].as_array().filter(|o| !o.is_empty()).ok_or_else(|| fail("operations must be a non-empty list"))?;
    for op in operations {
        let body = &op["operation"];
        let named = |value: &Value| value.as_str().is_some_and(|text| !text.is_empty());
        if !named(&body["type"]) || !named(&body["target"]["tensor"]) {
            return Err(fail("each operation needs a type and target"));
        }
        for metric in ["frobenius_norm", "l2_norm", "linf_norm"] {
            // Absent is allowed; present has to be a number, so a null doesn't count as absent.
            if body.get(metric).is_some_and(|v| !v.as_f64().is_some_and(|x| x.is_finite() && x >= 0.0)) {
                return Err(fail("operation metrics must be finite and non-negative"));
            }
        }
    }
    if let Some(controls) = payload["evaluation_provenance"]["controls"].as_object() {
        if !controls.values().all(|v| v.as_f64().is_some_and(|x| (0.0..=1.0).contains(&x))) {
            return Err(fail("controls must be in [0, 1]"));
        }
    }
    Ok(())
}

fn required_trusted_key(key: &str, path: &str) -> Result<String> {
    if key.is_empty() {
        return Err(err(
            ErrorCode::InvalidValue,
            path,
            "--pubkey ed25519_pub:... is required: a key inside the document cannot authenticate it",
        ));
    }
    Ok(key.to_owned())
}

fn handle_patch_sign(rest: &[String]) -> Result<String> {
    let (file, signer, key) = parse_patch_flags(rest);
    let file = file.ok_or_else(|| {
        err(
            ErrorCode::InvalidValue,
            "/patch/sign",
            "expected `patch sign <FILE.json> --key ed25519_priv:... [--signer ID]`",
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
            "expected `patch verify-signature <SIGNED_PATCH.json> --pubkey ed25519_pub:...`",
        )
    })?;
    let trusted = required_trusted_key(&key, "/patch/verify-signature")?;
    let raw = read_json_file(&file)?;
    let val: Value = serde_json::from_slice(&raw)
        .map_err(|e| err(ErrorCode::InvalidValue, "/patch/verify-signature", e.to_string()))?;
    wrap_ok(verify_signed_envelope(&val, &trusted)?)
}

fn handle_open_patch_verify(rest: &[String]) -> Result<String> {
    let (file, _signer, key) = parse_patch_flags(rest);
    let file = file.ok_or_else(|| {
        err(
            ErrorCode::InvalidValue,
            "/verify",
            "expected `nf verify <PATCH.json> --pubkey ed25519_pub:...`",
        )
    })?;
    let trusted = required_trusted_key(&key, "/verify")?;
    let raw = read_json_file(&file)?;
    if raw.len() > OPEN_PATCH_MAX_BYTES {
        return Err(err(ErrorCode::LimitExceeded, "/verify", "open patch exceeds the maximum artifact size"));
    }
    let val: Value = serde_json::from_slice(&raw)
        .map_err(|e| err(ErrorCode::InvalidValue, "/verify", e.to_string()))?;

    let envelope = val.get("signature_envelope").unwrap_or(&val);
    let sig_res = verify_signed_envelope(envelope, &trusted)?;
    // Only the signed payload is evidence; an unsigned top-level copy must match it exactly.
    let payload = envelope.get("patch").ok_or_else(|| {
        err(ErrorCode::IntegrityMismatch, "/verify", "signed envelope has no patch payload")
    })?;
    if let Some(top) = val.as_object().filter(|_| val.get("signature_envelope").is_some()) {
        let mut body = top.clone();
        for unsigned in ["signature_envelope", "patch_digest", "public_key"] {
            body.remove(unsigned);
        }
        if Value::Object(body) != *payload {
            return Err(err(
                ErrorCode::IntegrityMismatch,
                "/verify",
                "open patch body differs from its signed payload",
            ));
        }
        if top.get("patch_digest").is_some_and(|digest| *digest != sig_res["patch_digest"]) {
            return Err(err(
                ErrorCode::IntegrityMismatch,
                "/verify",
                "open patch's patch_digest differs from its signed digest",
            ));
        }
    }
    validate_open_patch_schema(payload)?;
    let section = |name: &str| {
        payload.get(name).and_then(Value::as_object).ok_or_else(|| {
            err(
                ErrorCode::IntegrityMismatch,
                "/verify",
                format!("open patch is missing required section `{name}`"),
            )
        })
    };

    let sbom = section("sbom")?;
    let base_d = sbom.get("base_checkpoint_digest").and_then(Value::as_str).unwrap_or("");
    let cand_d = sbom.get("candidate_checkpoint_digest").and_then(Value::as_str).unwrap_or("");
    let changed = sbom.get("changed_parameters").and_then(Value::as_u64).unwrap_or(0);
    if !base_d.starts_with("sha256:") || !cand_d.starts_with("sha256:") || changed == 0 {
        return Err(err(
            ErrorCode::IntegrityMismatch,
            "/verify/sbom",
            "invalid or empty SBOM digests/changed_parameters in open patch",
        ));
    }

    let rc = section("rounding_contract")?;
    let rounding_field = |name: &str| rc.get(name).and_then(Value::as_str).is_some_and(|text| !text.is_empty());
    if !["source_dtype", "accumulation_dtype", "rounding_mode"].iter().all(|name| rounding_field(name)) {
        return Err(err(
            ErrorCode::IntegrityMismatch,
            "/verify/rounding_contract",
            "incomplete rounding_contract in open patch",
        ));
    }

    let sec = section("security_scan")?;
    if sec.get("backdoor_scan_passed").and_then(Value::as_bool) != Some(true) {
        return Err(err(
            ErrorCode::IntegrityMismatch,
            "/verify/security_scan",
            "supply-chain backdoor scan failed on weight patch",
        ));
    }

    let ev = section("evaluation_provenance")?;
    if ev.get("promote_eligible").and_then(Value::as_bool) != Some(true) {
        return Err(err(
            ErrorCode::IntegrityMismatch,
            "/verify/evaluation_provenance",
            "patch evaluation_provenance is not marked promote_eligible=true",
        ));
    }
    let ctrls = ev.get("controls").and_then(Value::as_object).ok_or_else(|| {
        err(ErrorCode::IntegrityMismatch, "/verify/controls", "missing evaluation controls")
    })?;
    let control = |name: &str| {
        ctrls.get(name).and_then(Value::as_f64).ok_or_else(|| {
            err(ErrorCode::IntegrityMismatch, "/verify/controls", format!("missing control `{name}`"))
        })
    };
    let cand_acc = control("candidate_accuracy")?;
    if cand_acc <= control("noop_accuracy")? || cand_acc <= control("random_accuracy")? {
        return Err(err(
            ErrorCode::IntegrityMismatch,
            "/verify/controls",
            "candidate failed to beat mandatory no-op or random controls",
        ));
    }

    wrap_ok(json!({
        "schema_version": 1,
        "kind": "nf_open_patch_verification_receipt",
        "verified": true,
        "patch_digest": sig_res["patch_digest"],
        "signature_scheme": sig_res["signature_scheme"],
        "signer_id": sig_res["signer_id"],
        "sbom_verified": true,
        "rounding_contract_verified": true,
        "backdoor_scan_verified": true,
        "controls_verified": true,
    }))
}

fn handle_top_level_repair(_rest: &[String]) -> Result<String> {
    // The native CLI cannot load or evaluate a model, so it has nothing truthful to sign.
    Err(err(
        ErrorCode::UnsupportedCapability,
        "/repair",
        "`nf repair` cannot execute a model; run `python -m neural_foundry.studio repair --model MODEL --failure MODE`",
    ))
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
        if key.is_empty() {
            json!({"verified": false, "status": "not_verified_no_trusted_key"})
        } else {
            verify_signed_envelope(&val, &key)?
        }
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
        // Unsigned means unauthenticated, so an audit can't pass it, whatever the document says about itself.
        "audit_passed": is_signed && sig_check["verified"].as_bool().unwrap_or(false) && preserve_shapes,
        "signed": is_signed,
        "signature_verification": sig_check,
        "gradient_free": gradient_free,
        "preserve_tensor_shapes": preserve_shapes,
        "diff_summary": diff,
    }))
}

fn handle_patch_rollback(_rest: &[String]) -> Result<String> {
    // A patch document holds digests, not weights, so the native CLI has nothing to restore or check.
    Err(err(
        ErrorCode::UnsupportedCapability,
        "/patch/rollback",
        "`nf patch rollback` cannot restore weights from a patch file; use `neural_foundry.patch_ux.create_rollback_handle` and `rollback_model_patch` in Python",
    ))
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
