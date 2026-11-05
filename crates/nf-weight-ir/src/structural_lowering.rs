//! Structural IR lowering and preservation rules (T101).
//!
//! Expands structural architecture surgery (`slice_mlp_intermediate`, `prune_attention_heads`,
//! `remove_layer`, `insert_identity_block`) into coupled tensor/config/metadata rewrites,
//! enforces `preserve_tensor_shapes` space bindings, classifies `ExactUnderConditions`
//! vs `Experimental`, and requires rebuilt runtime and parity checks.

use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use nf_core::error::{CoreError, ErrorCode, Result};
use nf_weight_ir_self::program::Preservation;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use crate as nf_weight_ir_self;

fn err(code: ErrorCode, path: &str, message: impl Into<String>) -> CoreError {
    CoreError::new(code, path, message)
}

fn digest_json<T: Serialize>(value: &T) -> Result<ContentDigest> {
    let raw = serde_json::to_vec(value).map_err(|e| {
        err(
            ErrorCode::InvalidValue,
            "/structural_lowering",
            e.to_string(),
        )
    })?;
    Ok(CanonicalDocument::parse(&raw, Limits::default())?.digest())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum StructuralOperation {
    SliceMlpIntermediate {
        layer_index: u32,
        keep_channels: Vec<u32>,
        original_intermediate_size: u32,
    },
    PruneAttentionHeads {
        layer_index: u32,
        keep_q_heads: Vec<u32>,
        keep_kv_heads: Vec<u32>,
        head_dim: u32,
    },
    RemoveLayer {
        layer_index: u32,
        total_layers_before: u32,
    },
    InsertIdentityProjection {
        layer_index: u32,
        hidden_size: u32,
        zero_initialized_residual_branch: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralRewriteStep {
    pub tensor_name: String,
    pub old_shape: Vec<u32>,
    pub new_shape: Vec<u32>,
    pub action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoweredStructuralPlan {
    pub schema_version: u32,
    pub search_space_digest: ContentDigest,
    pub rewrites: Vec<StructuralRewriteStep>,
    pub config_updates: BTreeMap<String, u32>,
    pub changes_tensor_shapes: bool,
    pub preservation: Preservation,
    pub requires_runtime_rebuild: bool,
    pub requires_parity_reverification: bool,
    pub plan_digest: ContentDigest,
}

pub fn lower_structural_operation(
    op: &StructuralOperation,
    claimed_preservation: &Preservation,
    search_space_digest: ContentDigest,
    space_allows_shape_changes: bool,
    hidden_size: u32,
) -> Result<LoweredStructuralPlan> {
    if hidden_size == 0 {
        return Err(err(
            ErrorCode::InvalidValue,
            "/hidden_size",
            "positive hidden_size required",
        ));
    }

    let mut rewrites = Vec::new();
    let mut config_updates = BTreeMap::new();
    let mut changes_shapes = true;
    let mut algebraically_exact = false;

    match op {
        StructuralOperation::SliceMlpIntermediate {
            layer_index,
            keep_channels,
            original_intermediate_size,
        } => {
            if keep_channels.is_empty() || keep_channels.len() as u32 >= *original_intermediate_size
            {
                return Err(err(
                    ErrorCode::InvalidValue,
                    "/keep_channels",
                    "keep_channels must be non-empty and smaller than original_intermediate_size",
                ));
            }
            let unique: BTreeSet<u32> = keep_channels.iter().copied().collect();
            if unique.len() != keep_channels.len()
                || unique.iter().any(|c| *c >= *original_intermediate_size)
            {
                return Err(err(
                    ErrorCode::InvalidValue,
                    "/keep_channels",
                    "keep_channels must be unique valid indices",
                ));
            }
            let new_inter = keep_channels.len() as u32;
            for proj in ["gate_proj", "up_proj"] {
                rewrites.push(StructuralRewriteStep {
                    tensor_name: format!("model.layers.{layer_index}.mlp.{proj}.weight"),
                    old_shape: vec![*original_intermediate_size, hidden_size],
                    new_shape: vec![new_inter, hidden_size],
                    action: "slice_rows".into(),
                });
            }
            rewrites.push(StructuralRewriteStep {
                tensor_name: format!("model.layers.{layer_index}.mlp.down_proj.weight"),
                old_shape: vec![hidden_size, *original_intermediate_size],
                new_shape: vec![hidden_size, new_inter],
                action: "slice_cols".into(),
            });
            config_updates.insert("intermediate_size".into(), new_inter);
        }
        StructuralOperation::PruneAttentionHeads {
            layer_index,
            keep_q_heads,
            keep_kv_heads,
            head_dim,
        } => {
            if keep_q_heads.is_empty()
                || keep_kv_heads.is_empty()
                || *head_dim == 0
                || keep_q_heads.len() % keep_kv_heads.len() != 0
            {
                return Err(err(
                    ErrorCode::ScopeViolation,
                    "/prune_attention_heads",
                    "retained q_heads must be non-empty and divisible by retained kv_heads",
                ));
            }
            let new_q_dim = (keep_q_heads.len() as u32) * head_dim;
            let new_kv_dim = (keep_kv_heads.len() as u32) * head_dim;
            rewrites.push(StructuralRewriteStep {
                tensor_name: format!("model.layers.{layer_index}.self_attn.q_proj.weight"),
                old_shape: vec![hidden_size, hidden_size],
                new_shape: vec![new_q_dim, hidden_size],
                action: "slice_q_heads".into(),
            });
            for kv_proj in ["k_proj", "v_proj"] {
                rewrites.push(StructuralRewriteStep {
                    tensor_name: format!("model.layers.{layer_index}.self_attn.{kv_proj}.weight"),
                    old_shape: vec![hidden_size, hidden_size],
                    new_shape: vec![new_kv_dim, hidden_size],
                    action: "slice_kv_heads".into(),
                });
            }
            rewrites.push(StructuralRewriteStep {
                tensor_name: format!("model.layers.{layer_index}.self_attn.o_proj.weight"),
                old_shape: vec![hidden_size, hidden_size],
                new_shape: vec![hidden_size, new_q_dim],
                action: "slice_o_cols".into(),
            });
            config_updates.insert("num_attention_heads".into(), keep_q_heads.len() as u32);
            config_updates.insert("num_key_value_heads".into(), keep_kv_heads.len() as u32);
        }
        StructuralOperation::RemoveLayer {
            layer_index,
            total_layers_before,
        } => {
            if *total_layers_before <= 1 || *layer_index >= *total_layers_before {
                return Err(err(
                    ErrorCode::InvalidValue,
                    "/remove_layer",
                    "cannot remove last remaining layer or out-of-range layer index",
                ));
            }
            rewrites.push(StructuralRewriteStep {
                tensor_name: format!("model.layers.{layer_index}.*"),
                old_shape: vec![hidden_size, hidden_size],
                new_shape: vec![0, 0],
                action: "remove_and_reindex_subsequent_layers".into(),
            });
            config_updates.insert("num_hidden_layers".into(), total_layers_before - 1);
        }
        StructuralOperation::InsertIdentityProjection {
            layer_index,
            hidden_size: h,
            zero_initialized_residual_branch,
        } => {
            if !zero_initialized_residual_branch {
                return Err(err(
                    ErrorCode::ScopeViolation,
                    "/insert_identity_projection",
                    "identity-preserving residual insertion requires zero_initialized_residual_branch=true",
                ));
            }
            changes_shapes = true;
            algebraically_exact = true;
            rewrites.push(StructuralRewriteStep {
                tensor_name: format!("model.layers.{layer_index}.residual_adapter.weight"),
                old_shape: vec![0, 0],
                new_shape: vec![*h, *h],
                action: "insert_zero_residual_projection".into(),
            });
        }
    }

    if changes_shapes && !space_allows_shape_changes {
        return Err(err(
            ErrorCode::ScopeViolation,
            "/preserve_tensor_shapes",
            "structural shape change rejected because frozen SearchSpace requires preserve_tensor_shapes=true",
        ));
    }

    if matches!(
        claimed_preservation,
        Preservation::ExactUnderConditions { .. }
    ) && !algebraically_exact
    {
        return Err(err(
            ErrorCode::ScopeViolation,
            "/preservation",
            "lossy pruning or layer removal cannot claim exact_under_conditions preservation",
        ));
    }

    let body = serde_json::json!({
        "search_space_digest": search_space_digest,
        "rewrites": rewrites,
        "config_updates": config_updates,
        "preservation": claimed_preservation,
    });
    let plan_digest = digest_json(&body)?;

    Ok(LoweredStructuralPlan {
        schema_version: 1,
        search_space_digest,
        rewrites,
        config_updates,
        changes_tensor_shapes: changes_shapes,
        preservation: claimed_preservation.clone(),
        requires_runtime_rebuild: true,
        requires_parity_reverification: true,
        plan_digest,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(tag: &str) -> ContentDigest {
        ContentDigest::of_bytes(tag.as_bytes())
    }

    #[test]
    fn structural_lowering_expands_coupled_rewrites_and_rejects_false_exactness_or_shape_violations(
    ) {
        // 1. Valid MLP channel slicing under experimental preservation when shape changes are allowed
        let mlp_op = StructuralOperation::SliceMlpIntermediate {
            layer_index: 0,
            keep_channels: vec![0, 2, 3],
            original_intermediate_size: 8,
        };
        let plan = lower_structural_operation(
            &mlp_op,
            &Preservation::Experimental,
            d("space-structural"),
            true,
            4,
        )
        .unwrap();
        assert_eq!(plan.rewrites.len(), 3);
        assert_eq!(plan.config_updates.get("intermediate_size"), Some(&3));
        assert!(plan.requires_runtime_rebuild);
        assert!(plan.requires_parity_reverification);

        // 2. Same-shape search space rejects structural surgery
        assert_eq!(
            lower_structural_operation(
                &mlp_op,
                &Preservation::Experimental,
                d("space-same-shape"),
                false,
                4,
            )
            .unwrap_err()
            .code,
            ErrorCode::ScopeViolation
        );

        // 3. False ExactUnderConditions claim on lossy MLP slicing is rejected
        let exact_claim = Preservation::ExactUnderConditions {
            rule: crate::program::RecipeRef {
                id: "zero_residual_identity".into(),
                version: 1,
                specification: d("spec-zero-res"),
            },
            conditions: vec![d("exact_real_arithmetic")],
        };
        assert_eq!(
            lower_structural_operation(&mlp_op, &exact_claim, d("space-structural"), true, 4)
                .unwrap_err()
                .code,
            ErrorCode::ScopeViolation
        );

        // 4. Checked zero-residual identity insertion succeeds with ExactUnderConditions
        let id_op = StructuralOperation::InsertIdentityProjection {
            layer_index: 0,
            hidden_size: 4,
            zero_initialized_residual_branch: true,
        };
        let exact_plan =
            lower_structural_operation(&id_op, &exact_claim, d("space-structural"), true, 4)
                .unwrap();
        assert!(exact_plan.changes_tensor_shapes);
        assert!(exact_plan.requires_parity_reverification);
    }
}
