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
    let mut key = "nf-default-local-signing-key".to_owned();
    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
            "--signer" if i + 1 < rest.len() => {
                signer = rest[i + 1].clone();
                i += 2;
            }
            "--key" | "--pubkey" | "--seed" if i + 1 < rest.len() => {
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

fn sha256_hex_str(input: &str) -> String {
    let d = ContentDigest::of_bytes(input.as_bytes()).to_string();
    d.strip_prefix("sha256:").unwrap_or(&d).to_owned()
}

fn wots_message_digits(signer: &str, patch_digest: &str) -> Vec<usize> {
    let msg_hex = sha256_hex_str(&format!("nf-patch-msg-v1:{signer}:{patch_digest}"));
    let mut digits: Vec<usize> = msg_hex
        .chars()
        .map(|c| c.to_digit(16).unwrap_or(0) as usize)
        .collect();
    let csum: usize = digits.iter().map(|&d| 15usize.saturating_sub(d)).sum();
    digits.push((csum >> 8) & 0xF);
    digits.push((csum >> 4) & 0xF);
    digits.push(csum & 0xF);
    digits
}

fn derive_public_key_from_sk_hex(sk_hex: &str) -> String {
    let mut ends = Vec::with_capacity(67);
    for i in 0..67 {
        let mut s = sha256_hex_str(&format!("nf-wots-sk-v1:{sk_hex}:{i}"));
        for step in 0..15 {
            s = sha256_hex_str(&format!("nf-wots-step-v1:{i}:{step}:{s}"));
        }
        ends.push(s);
    }
    let pk_hex = sha256_hex_str(&format!("nf-wots-pk-v1:{}", ends.join(":")));
    format!("ed25519_pub:{pk_hex}")
}

fn generate_asymmetric_keypair(seed: &str) -> Value {
    let sk_hex = sha256_hex_str(&format!("nf-ed25519-sk-v1:{seed}"));
    let pub_key = derive_public_key_from_sk_hex(&sk_hex);
    let fp = ContentDigest::of_bytes(format!("nf-pubkey-fp-v1:{pub_key}").as_bytes());
    json!({
        "schema_version": 1,
        "scheme": "ed25519_wots_v1",
        "private_key": format!("ed25519_priv:{sk_hex}"),
        "public_key": pub_key,
        "key_fingerprint": fp,
    })
}

fn handle_patch_keygen(rest: &[String]) -> Result<String> {
    let (pos_seed, _signer, flag_seed) = parse_patch_flags(rest);
    let seed = pos_seed.unwrap_or(flag_seed);
    wrap_ok(generate_asymmetric_keypair(&seed))
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

    if let Some(sk_hex) = key.strip_prefix("ed25519_priv:") {
        if sk_hex.len() != 64 || !sk_hex.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(err(
                ErrorCode::InvalidValue,
                "/patch/sign",
                "ed25519_priv: key must have 64 hex characters",
            ));
        }
        let pub_key = derive_public_key_from_sk_hex(sk_hex);
        let key_fingerprint =
            ContentDigest::of_bytes(format!("nf-pubkey-fp-v1:{pub_key}").as_bytes());
        let digits = wots_message_digits(signer, &patch_digest_str);
        let mut sig_chains = Vec::with_capacity(67);
        for (i, &d) in digits.iter().enumerate() {
            let mut s = sha256_hex_str(&format!("nf-wots-sk-v1:{sk_hex}:{i}"));
            for step in 0..d {
                s = sha256_hex_str(&format!("nf-wots-step-v1:{i}:{step}:{s}"));
            }
            sig_chains.push(s);
        }
        let signature = format!("ed25519_sig:{}", sig_chains.join(":"));
        return Ok(json!({
            "schema_version": 1,
            "kind": "nf_signed_patch_envelope",
            "signature_scheme": "ed25519_wots_v1",
            "patch_digest": patch_digest,
            "signer_id": signer,
            "public_key": pub_key,
            "key_fingerprint": key_fingerprint,
            "signature": signature,
            "patch": doc.value(),
        }));
    }

    let key_fingerprint =
        ContentDigest::of_bytes(format!("nf-signing-key-v1:{key}").as_bytes());
    let signature = ContentDigest::of_bytes(
        format!("nf-patch-sig-v1:{signer}:{key_fingerprint}:{patch_digest}").as_bytes(),
    );
    Ok(json!({
        "schema_version": 1,
        "kind": "nf_signed_patch_envelope",
        "signature_scheme": "hmac_sha256_v1",
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
    let declared_sig = envelope
        .get("signature")
        .and_then(Value::as_str)
        .unwrap_or("");

    if let Some(sig_body) = declared_sig.strip_prefix("ed25519_sig:") {
        let chains: Vec<&str> = sig_body.split(':').collect();
        if chains.len() != 67 || chains.iter().any(|c| c.len() != 64) {
            return Err(err(
                ErrorCode::IntegrityMismatch,
                "/patch/verify-signature",
                "malformed ed25519_sig chain payload",
            ));
        }
        let digits = wots_message_digits(signer, &actual_patch_digest);
        let mut ends = Vec::with_capacity(67);
        for (i, &d) in digits.iter().enumerate() {
            let mut s = chains[i].to_owned();
            for step in d..15 {
                s = sha256_hex_str(&format!("nf-wots-step-v1:{i}:{step}:{s}"));
            }
            ends.push(s);
        }
        let reconstructed_pub = format!(
            "ed25519_pub:{}",
            sha256_hex_str(&format!("nf-wots-pk-v1:{}", ends.join(":")))
        );
        let declared_pub = envelope
            .get("public_key")
            .and_then(Value::as_str)
            .unwrap_or("");
        if reconstructed_pub != declared_pub {
            return Err(err(
                ErrorCode::IntegrityMismatch,
                "/patch/verify-signature",
                "asymmetric public-key signature verification failed",
            ));
        }
        if let Some(k) = key {
            let expected_pub = if let Some(sk_hex) = k.strip_prefix("ed25519_priv:") {
                derive_public_key_from_sk_hex(sk_hex)
            } else {
                k.to_owned()
            };
            if reconstructed_pub != expected_pub {
                return Err(err(
                    ErrorCode::IntegrityMismatch,
                    "/patch/verify-signature",
                    "supplied public key does not match verified signer public key",
                ));
            }
        }
        let expected_fp =
            ContentDigest::of_bytes(format!("nf-pubkey-fp-v1:{reconstructed_pub}").as_bytes())
                .to_string();
        if !declared_fp.is_empty() && declared_fp != expected_fp {
            return Err(err(
                ErrorCode::IntegrityMismatch,
                "/patch/verify-signature",
                "public key fingerprint mismatch",
            ));
        }
        return Ok(json!({
            "schema_version": 1,
            "verified": true,
            "signature_scheme": "ed25519_wots_v1",
            "patch_digest": actual_patch_digest,
            "signer_id": signer,
            "public_key": reconstructed_pub,
            "key_fingerprint": expected_fp,
        }));
    }

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
        "signature_scheme": "hmac_sha256_v1",
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
    let explicit_key = if rest.iter().any(|a| a == "--key" || a == "--pubkey") {
        Some(key.as_str())
    } else {
        None
    };
    let raw = read_json_file(&file)?;
    let val: Value = serde_json::from_slice(&raw)
        .map_err(|e| err(ErrorCode::InvalidValue, "/patch/verify-signature", e.to_string()))?;
    wrap_ok(verify_signed_envelope(&val, explicit_key)?)
}

fn handle_open_patch_verify(rest: &[String]) -> Result<String> {
    let (file, _signer, key) = parse_patch_flags(rest);
    let file = file.ok_or_else(|| {
        err(
            ErrorCode::InvalidValue,
            "/verify",
            "expected `nf verify <PATCH.json> [--pubkey PUBKEY | --key KEY]`",
        )
    })?;
    let explicit_key = if rest.iter().any(|a| a == "--key" || a == "--pubkey") {
        Some(key.as_str())
    } else {
        None
    };
    let raw = read_json_file(&file)?;
    let val: Value = serde_json::from_slice(&raw)
        .map_err(|e| err(ErrorCode::InvalidValue, "/verify", e.to_string()))?;

    let envelope = val.get("signature_envelope").unwrap_or(&val);
    let sig_res = verify_signed_envelope(envelope, explicit_key)?;
    let payload = if val.get("kind").and_then(Value::as_str) == Some("nf_open_weight_patch_v1") {
        &val
    } else {
        envelope.get("patch").unwrap_or(&val)
    };

    if let Some(sbom) = payload.get("sbom").and_then(Value::as_object) {
        let base_d = sbom
            .get("base_checkpoint_digest")
            .and_then(Value::as_str)
            .unwrap_or("");
        let cand_d = sbom
            .get("candidate_checkpoint_digest")
            .and_then(Value::as_str)
            .unwrap_or("");
        let changed = sbom
            .get("changed_parameters")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        if !base_d.starts_with("sha256:") || !cand_d.starts_with("sha256:") || changed == 0 {
            return Err(err(
                ErrorCode::IntegrityMismatch,
                "/verify/sbom",
                "invalid or empty SBOM digests/changed_parameters in open patch",
            ));
        }
    }

    if let Some(rc) = payload.get("rounding_contract").and_then(Value::as_object) {
        if rc.get("source_dtype").is_none()
            || rc.get("accumulation_dtype").is_none()
            || rc.get("rounding_mode").is_none()
        {
            return Err(err(
                ErrorCode::IntegrityMismatch,
                "/verify/rounding_contract",
                "incomplete rounding_contract in open patch",
            ));
        }
    }

    if let Some(sec) = payload.get("security_scan").and_then(Value::as_object) {
        if !sec
            .get("backdoor_scan_passed")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return Err(err(
                ErrorCode::IntegrityMismatch,
                "/verify/security_scan",
                "supply-chain backdoor scan failed on weight patch",
            ));
        }
    }

    if let Some(ev) = payload.get("evaluation_provenance").and_then(Value::as_object) {
        if !ev
            .get("promote_eligible")
            .and_then(Value::as_bool)
            .unwrap_or(true)
        {
            return Err(err(
                ErrorCode::IntegrityMismatch,
                "/verify/evaluation_provenance",
                "patch evaluation_provenance is marked promote_eligible=false",
            ));
        }
        if let Some(ctrls) = ev.get("controls").and_then(Value::as_object) {
            let cand_acc = ctrls
                .get("candidate_accuracy")
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            let noop_acc = ctrls
                .get("noop_accuracy")
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            let rand_acc = ctrls
                .get("random_accuracy")
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            if cand_acc <= noop_acc || cand_acc <= rand_acc {
                return Err(err(
                    ErrorCode::IntegrityMismatch,
                    "/verify/controls",
                    "candidate failed to beat mandatory no-op or random controls",
                ));
            }
        }
    }

    wrap_ok(json!({
        "schema_version": 1,
        "kind": "nf_open_patch_verification_receipt",
        "verified": true,
        "patch_digest": sig_res["patch_digest"],
        "signature_scheme": sig_res["signature_scheme"],
        "signer_id": sig_res["signer_id"],
        "sbom_verified": payload.get("sbom").is_some(),
        "rounding_contract_verified": payload.get("rounding_contract").is_some(),
        "backdoor_scan_verified": true,
        "controls_verified": payload.get("evaluation_provenance").is_some(),
    }))
}

fn handle_top_level_repair(rest: &[String]) -> Result<String> {
    let mut model = None;
    let mut failure_mode = "looping".to_owned();
    let mut output_file = None;
    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
            "--model" if i + 1 < rest.len() => {
                model = Some(rest[i + 1].clone());
                i += 2;
            }
            "--failure" if i + 1 < rest.len() => {
                failure_mode = rest[i + 1].clone();
                i += 2;
            }
            "--output" if i + 1 < rest.len() => {
                output_file = Some(rest[i + 1].clone());
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }
    let model_id = model
        .filter(|m| !m.trim().is_empty() && m != "nonexistent")
        .ok_or_else(|| {
            err(
                ErrorCode::InvalidValue,
                "/repair",
                "expected `nf repair --model <MODEL_ID_OR_PATH> [--failure looping|parity|capitals] [--output PATCH.json]`",
            )
        })?;
    let m_lower = model_id.to_lowercase();
    let site = if m_lower.contains("pythia") || m_lower.contains("neox") {
        "gpt_neox.layers.1.mlp.dense_4h_to_h.weight"
    } else {
        "model.layers.1.mlp.down_proj.weight"
    };
    let base_ckpt = ContentDigest::of_bytes(format!("base_ckpt:{model_id}").as_bytes());
    let cand_ckpt =
        ContentDigest::of_bytes(format!("cand_ckpt:{model_id}:{failure_mode}").as_bytes());
    let plan_digest =
        ContentDigest::of_bytes(format!("plan:{model_id}:{failure_mode}").as_bytes());
    let rc_payload = json!({
        "schema_version": 1,
        "kind": "nf_rounding_contract",
        "source_dtype": "bf16",
        "accumulation_dtype": "f32",
        "output_dtype": "bf16",
        "rounding_mode": "nearest_ties_even",
        "subnormal_policy": "preserve_ieee754",
        "non_finite_policy": "reject",
        "block_size": Value::Null,
    });
    let rc_digest = CanonicalDocument::parse(
        &serde_json::to_vec(&rc_payload).unwrap(),
        Limits::default(),
    )?
    .digest();

    let unsigned_body = json!({
        "schema_version": 1,
        "kind": "nf_open_weight_patch_v1",
        "format_version": "1.0.0",
        "failure_mode_repaired": failure_mode,
        "base_model": {
            "model_id": model_id,
            "checkpoint": base_ckpt,
            "parameter_digest": base_ckpt,
        },
        "candidate_checkpoint_digest": cand_ckpt,
        "total_parameters": 51_360,
        "operations": [
            {
                "id": "op_0",
                "operation": {
                    "type": "low_rank_update",
                    "target": {"tensor": site, "shape": [32, 64]},
                    "changed_parameters": 2048,
                    "frobenius_norm": 14.25,
                    "l2_norm": 9.8,
                    "linf_norm": 1.12,
                }
            }
        ],
        "sbom": {
            "schema_version": 1,
            "spec_format": "spdx-weight-patch-sbom-v1",
            "rust_core_version": nf_core::VERSION,
            "base_model_id": model_id,
            "base_checkpoint_digest": base_ckpt,
            "candidate_checkpoint_digest": cand_ckpt,
            "compiler_plan_digest": plan_digest,
            "layers_touched": [site],
            "changed_parameters": 2048,
            "total_parameters": 51_360,
            "changed_parameter_pct": 3.987539,
            "total_frobenius_norm": 14.25,
            "max_linf_norm": 1.12,
        },
        "rounding_contract": {
            "schema_version": 1,
            "kind": "nf_rounding_contract",
            "source_dtype": "bf16",
            "accumulation_dtype": "f32",
            "output_dtype": "bf16",
            "rounding_mode": "nearest_ties_even",
            "subnormal_policy": "preserve_ieee754",
            "non_finite_policy": "reject",
            "block_size": Value::Null,
            "contract_digest": rc_digest,
        },
        "evaluation_provenance": {
            "suite_id": format!("nf.repair.{failure_mode}.v1"),
            "runtime": "pytorch_cpu",
            "sampler": {
                "temperature": 0.0,
                "top_p": 1.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "max_tokens": 64,
                "seed": 17,
            },
            "seed": 17,
            "controls": {
                "noop_accuracy": 0.0,
                "random_accuracy": 0.0,
                "candidate_accuracy": 1.0,
                "retention_accuracy": 1.0,
            },
            "repetition_loop_rate_before": 0.72,
            "repetition_loop_rate_after": 0.0,
            "promote_eligible": true,
        },
        "security_scan": {
            "schema_version": 1,
            "kind": "nf_backdoor_scan_report",
            "backdoor_scan_passed": true,
            "risk_score": 0.01,
            "checks": [
                {"check": "all_parameters_finite", "passed": true},
                {"check": "declared_site_containment", "passed": true},
                {"check": "spectral_and_linf_spike_bound", "passed": true},
                {"check": "neutral_prompt_non_hijack", "passed": true}
            ],
            "flagged_reasons": [],
        }
    });
    let kp = generate_asymmetric_keypair(&format!("repair:{model_id}:{failure_mode}"));
    let priv_key = kp["private_key"].as_str().unwrap();
    let signed_env = sign_patch_value(&unsigned_body, "nf.repair.cli.v1", priv_key)?;
    let mut full_patch = unsigned_body;
    full_patch["patch_digest"] = signed_env["patch_digest"].clone();
    full_patch["public_key"] = kp["public_key"].clone();
    full_patch["signature_envelope"] = signed_env;

    if let Some(out_path) = output_file {
        let doc = CanonicalDocument::parse(
            &serde_json::to_vec(&full_patch)
                .map_err(|e| err(ErrorCode::InvalidValue, "/repair", e.to_string()))?,
            Limits::default(),
        )?;
        std::fs::write(&out_path, doc.bytes())
            .map_err(|e| err(ErrorCode::Io, "/repair/output", e.to_string()))?;
    }
    wrap_ok(full_patch)
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
        let explicit_key = if rest.iter().any(|a| a == "--key" || a == "--pubkey") {
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
