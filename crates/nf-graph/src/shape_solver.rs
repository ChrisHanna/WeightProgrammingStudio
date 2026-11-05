//! Architecture dependency graph and symbolic shape solver (T100).
//!
//! Models modules, ports, symbolic dimensions, residual stream constraints,
//! tied-weight aliases, grouped-query attention (GQA) invariants, and cache/runtime
//! layouts. Rejects overflow, unsupported symbolic expressions, and contradictory
//! dimension bindings.

use nf_core::error::{CoreError, ErrorCode, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

fn err(code: ErrorCode, path: &str, message: impl Into<String>) -> CoreError {
    CoreError::new(code, path, message)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DimExpr {
    Literal { value: u64 },
    Symbol { name: String },
    Product { left: String, right: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorPortSpec {
    pub tensor_name: String,
    pub dims: Vec<DimExpr>,
    pub alias_group: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ArchitectureConstraint {
    ResidualStreamEquality {
        left_symbol: String,
        right_symbol: String,
    },
    TiedWeightAlias {
        source_tensor: String,
        target_tensor: String,
    },
    AttentionGroupDivisibility {
        num_q_heads_symbol: String,
        num_kv_heads_symbol: String,
        head_dim_symbol: String,
        hidden_size_symbol: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArchitectureShapeGraph {
    pub family_id: String,
    pub symbols: BTreeMap<String, u64>,
    pub ports: Vec<TensorPortSpec>,
    pub constraints: Vec<ArchitectureConstraint>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolvedArchitectureLayout {
    pub family_id: String,
    pub resolved_tensors: BTreeMap<String, Vec<u64>>,
    pub total_parameter_elements: u64,
    pub kv_group_ratio: Option<u64>,
}

impl ArchitectureShapeGraph {
    pub fn solve(&self) -> Result<SolvedArchitectureLayout> {
        if self.family_id.trim().is_empty() || self.ports.is_empty() {
            return Err(err(
                ErrorCode::InvalidValue,
                "/shape_solver",
                "non-empty family_id and tensor ports required",
            ));
        }
        for (sym, val) in &self.symbols {
            if sym.trim().is_empty() || *val == 0 || *val > (1 << 30) {
                return Err(err(
                    ErrorCode::InvalidValue,
                    "/symbols",
                    "symbolic dimensions must be positive and bounded",
                ));
            }
        }

        // Resolve each tensor port shape with overflow checking
        let mut resolved_tensors: BTreeMap<String, Vec<u64>> = BTreeMap::new();
        let mut alias_shapes: BTreeMap<String, Vec<u64>> = BTreeMap::new();
        let mut total_elements: u64 = 0;

        for port in &self.ports {
            if port.dims.is_empty() || port.dims.len() > 8 {
                return Err(err(
                    ErrorCode::InvalidValue,
                    "/ports/dims",
                    "port rank must be between 1 and 8",
                ));
            }
            let mut shape = Vec::with_capacity(port.dims.len());
            let mut port_elems: u64 = 1;
            for dim in &port.dims {
                let d_val = match dim {
                    DimExpr::Literal { value } => {
                        if *value == 0 {
                            return Err(err(
                                ErrorCode::InvalidValue,
                                "/ports/dims",
                                "zero literal dimension is invalid",
                            ));
                        }
                        *value
                    }
                    DimExpr::Symbol { name } => *self.symbols.get(name).ok_or_else(|| {
                        err(
                            ErrorCode::SourceMismatch,
                            "/ports/dims",
                            format!("unbound symbolic dimension '{name}'"),
                        )
                    })?,
                    DimExpr::Product { left, right } => {
                        let l = *self.symbols.get(left).ok_or_else(|| {
                            err(
                                ErrorCode::SourceMismatch,
                                "/ports/dims",
                                format!("unbound symbolic dimension '{left}'"),
                            )
                        })?;
                        let r = *self.symbols.get(right).ok_or_else(|| {
                            err(
                                ErrorCode::SourceMismatch,
                                "/ports/dims",
                                format!("unbound symbolic dimension '{right}'"),
                            )
                        })?;
                        l.checked_mul(r).ok_or_else(|| {
                            err(
                                ErrorCode::LimitExceeded,
                                "/ports/dims",
                                "symbolic dimension product overflow",
                            )
                        })?
                    }
                };
                port_elems = port_elems.checked_mul(d_val).ok_or_else(|| {
                    err(
                        ErrorCode::LimitExceeded,
                        "/ports/dims",
                        "tensor element count overflow",
                    )
                })?;
                shape.push(d_val);
            }

            if let Some(group) = &port.alias_group {
                if let Some(existing) = alias_shapes.get(group) {
                    if existing != &shape {
                        return Err(err(
                            ErrorCode::ScopeViolation,
                            "/alias_group",
                            format!("alias group '{group}' has contradictory member shapes"),
                        ));
                    }
                } else {
                    alias_shapes.insert(group.clone(), shape.clone());
                    total_elements = total_elements.checked_add(port_elems).ok_or_else(|| {
                        err(
                            ErrorCode::LimitExceeded,
                            "/total_parameter_elements",
                            "total parameter element count overflow",
                        )
                    })?;
                }
            } else {
                total_elements = total_elements.checked_add(port_elems).ok_or_else(|| {
                    err(
                        ErrorCode::LimitExceeded,
                        "/total_parameter_elements",
                        "total parameter element count overflow",
                    )
                })?;
            }

            if resolved_tensors
                .insert(port.tensor_name.clone(), shape)
                .is_some()
            {
                return Err(err(
                    ErrorCode::DuplicateKey,
                    "/ports",
                    format!("duplicate tensor port '{}'", port.tensor_name),
                ));
            }
        }

        let mut kv_ratio = None;
        for constraint in &self.constraints {
            match constraint {
                ArchitectureConstraint::ResidualStreamEquality {
                    left_symbol,
                    right_symbol,
                } => {
                    let l = self.symbols.get(left_symbol).ok_or_else(|| {
                        err(
                            ErrorCode::SourceMismatch,
                            "/constraints",
                            "missing left residual symbol",
                        )
                    })?;
                    let r = self.symbols.get(right_symbol).ok_or_else(|| {
                        err(
                            ErrorCode::SourceMismatch,
                            "/constraints",
                            "missing right residual symbol",
                        )
                    })?;
                    if l != r {
                        return Err(err(
                            ErrorCode::ScopeViolation,
                            "/constraints/residual",
                            format!("residual stream dimension mismatch: {left_symbol}={l} != {right_symbol}={r}"),
                        ));
                    }
                }
                ArchitectureConstraint::TiedWeightAlias {
                    source_tensor,
                    target_tensor,
                } => {
                    let s = resolved_tensors.get(source_tensor).ok_or_else(|| {
                        err(
                            ErrorCode::SourceMismatch,
                            "/constraints/tied",
                            "missing tied source tensor",
                        )
                    })?;
                    let t = resolved_tensors.get(target_tensor).ok_or_else(|| {
                        err(
                            ErrorCode::SourceMismatch,
                            "/constraints/tied",
                            "missing tied target tensor",
                        )
                    })?;
                    if s != t {
                        return Err(err(
                            ErrorCode::ScopeViolation,
                            "/constraints/tied",
                            format!("tied weight tensors '{source_tensor}' and '{target_tensor}' have mismatched shapes"),
                        ));
                    }
                }
                ArchitectureConstraint::AttentionGroupDivisibility {
                    num_q_heads_symbol,
                    num_kv_heads_symbol,
                    head_dim_symbol,
                    hidden_size_symbol,
                } => {
                    let q = *self.symbols.get(num_q_heads_symbol).ok_or_else(|| {
                        err(
                            ErrorCode::SourceMismatch,
                            "/constraints/gqa",
                            "missing q_heads",
                        )
                    })?;
                    let kv = *self.symbols.get(num_kv_heads_symbol).ok_or_else(|| {
                        err(
                            ErrorCode::SourceMismatch,
                            "/constraints/gqa",
                            "missing kv_heads",
                        )
                    })?;
                    let hd = *self.symbols.get(head_dim_symbol).ok_or_else(|| {
                        err(
                            ErrorCode::SourceMismatch,
                            "/constraints/gqa",
                            "missing head_dim",
                        )
                    })?;
                    let hs = *self.symbols.get(hidden_size_symbol).ok_or_else(|| {
                        err(
                            ErrorCode::SourceMismatch,
                            "/constraints/gqa",
                            "missing hidden_size",
                        )
                    })?;
                    if kv == 0 || q % kv != 0 {
                        return Err(err(
                            ErrorCode::ScopeViolation,
                            "/constraints/gqa",
                            format!("num_q_heads ({q}) must be divisible by num_kv_heads ({kv})"),
                        ));
                    }
                    let expected_hs = q.checked_mul(hd).ok_or_else(|| {
                        err(
                            ErrorCode::LimitExceeded,
                            "/constraints/gqa",
                            "q*head_dim overflow",
                        )
                    })?;
                    if expected_hs != hs {
                        return Err(err(
                            ErrorCode::ScopeViolation,
                            "/constraints/gqa",
                            format!("q_heads * head_dim ({expected_hs}) != hidden_size ({hs})"),
                        ));
                    }
                    kv_ratio = Some(q / kv);
                }
            }
        }

        Ok(SolvedArchitectureLayout {
            family_id: self.family_id.clone(),
            resolved_tensors,
            total_parameter_elements: total_elements,
            kv_group_ratio: kv_ratio,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_qwen2_graph() -> ArchitectureShapeGraph {
        ArchitectureShapeGraph {
            family_id: "hf.qwen2.dense".into(),
            symbols: BTreeMap::from([
                ("vocab_size".into(), 32),
                ("hidden_size".into(), 16),
                ("intermediate_size".into(), 32),
                ("num_q_heads".into(), 4),
                ("num_kv_heads".into(), 2),
                ("head_dim".into(), 4),
                ("out_hidden_size".into(), 16),
            ]),
            ports: vec![
                TensorPortSpec {
                    tensor_name: "model.embed_tokens.weight".into(),
                    dims: vec![
                        DimExpr::Symbol {
                            name: "vocab_size".into(),
                        },
                        DimExpr::Symbol {
                            name: "hidden_size".into(),
                        },
                    ],
                    alias_group: Some("tied_embed".into()),
                },
                TensorPortSpec {
                    tensor_name: "lm_head.weight".into(),
                    dims: vec![
                        DimExpr::Symbol {
                            name: "vocab_size".into(),
                        },
                        DimExpr::Symbol {
                            name: "out_hidden_size".into(),
                        },
                    ],
                    alias_group: Some("tied_embed".into()),
                },
                TensorPortSpec {
                    tensor_name: "model.layers.0.self_attn.q_proj.weight".into(),
                    dims: vec![
                        DimExpr::Product {
                            left: "num_q_heads".into(),
                            right: "head_dim".into(),
                        },
                        DimExpr::Symbol {
                            name: "hidden_size".into(),
                        },
                    ],
                    alias_group: None,
                },
                TensorPortSpec {
                    tensor_name: "model.layers.0.self_attn.k_proj.weight".into(),
                    dims: vec![
                        DimExpr::Product {
                            left: "num_kv_heads".into(),
                            right: "head_dim".into(),
                        },
                        DimExpr::Symbol {
                            name: "hidden_size".into(),
                        },
                    ],
                    alias_group: None,
                },
                TensorPortSpec {
                    tensor_name: "model.layers.0.mlp.up_proj.weight".into(),
                    dims: vec![
                        DimExpr::Symbol {
                            name: "intermediate_size".into(),
                        },
                        DimExpr::Symbol {
                            name: "hidden_size".into(),
                        },
                    ],
                    alias_group: None,
                },
            ],
            constraints: vec![
                ArchitectureConstraint::ResidualStreamEquality {
                    left_symbol: "hidden_size".into(),
                    right_symbol: "out_hidden_size".into(),
                },
                ArchitectureConstraint::TiedWeightAlias {
                    source_tensor: "model.embed_tokens.weight".into(),
                    target_tensor: "lm_head.weight".into(),
                },
                ArchitectureConstraint::AttentionGroupDivisibility {
                    num_q_heads_symbol: "num_q_heads".into(),
                    num_kv_heads_symbol: "num_kv_heads".into(),
                    head_dim_symbol: "head_dim".into(),
                    hidden_size_symbol: "hidden_size".into(),
                },
            ],
        }
    }

    #[test]
    fn shape_solver_resolves_valid_gqa_and_tied_head_layout_without_double_counting_aliases() {
        let g = valid_qwen2_graph();
        let solved = g.solve().unwrap();
        assert_eq!(solved.kv_group_ratio, Some(2));
        assert_eq!(
            solved.resolved_tensors["model.layers.0.self_attn.k_proj.weight"],
            vec![8, 16]
        );
        // tied_embed (32*16 = 512) counted once + q_proj (16*16 = 256) + k_proj (8*16 = 128) + up_proj (32*16 = 512) = 1408
        assert_eq!(solved.total_parameter_elements, 1408);
    }

    #[test]
    fn shape_solver_rejects_residual_mismatch_tied_head_contradiction_gqa_indivisibility_and_overflow(
    ) {
        // 1. Residual and tied-head mismatch
        let mut bad_res = valid_qwen2_graph();
        bad_res.symbols.insert("out_hidden_size".into(), 12);
        assert_eq!(bad_res.solve().unwrap_err().code, ErrorCode::ScopeViolation);

        // 2. GQA indivisibility (num_q_heads=5 not divisible by num_kv_heads=2)
        let mut bad_gqa = valid_qwen2_graph();
        bad_gqa.symbols.insert("num_q_heads".into(), 5);
        assert_eq!(bad_gqa.solve().unwrap_err().code, ErrorCode::ScopeViolation);

        // 3. Overflow in dimension product
        let mut overflow = valid_qwen2_graph();
        overflow.ports.push(TensorPortSpec {
            tensor_name: "huge.weight".into(),
            dims: vec![
                DimExpr::Literal {
                    value: u64::MAX / 2,
                },
                DimExpr::Literal { value: 8 },
            ],
            alias_group: None,
        });
        assert_eq!(overflow.solve().unwrap_err().code, ErrorCode::LimitExceeded);
    }
}
