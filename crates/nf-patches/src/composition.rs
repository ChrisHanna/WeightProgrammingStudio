//! Patch dependency and conflict composition (T069).
//!
//! Checks common bases, dependency cycles, implicit rebases, read/write/alias/packing
//! groups, ordering, graph/runtime effects, and noncommutation. Explicit reconciliation
//! creates a new program; syntactically disjoint writes still require behavioral
//! evaluation (`behavioral_evaluation_required = true`).

use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use nf_core::error::{CoreError, ErrorCode, Result};
use nf_weight_ir::program::{Instruction, ModelBinding, OperationKind, Program, TensorRegion};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

fn err(code: ErrorCode, path: &str, message: impl Into<String>) -> CoreError {
    CoreError::new(code, path, message)
}

fn canonical_digest<T: Serialize>(value: &T) -> Result<ContentDigest> {
    let raw = serde_json::to_vec(value)
        .map_err(|e| err(ErrorCode::InvalidValue, "/composition", e.to_string()))?;
    Ok(CanonicalDocument::parse(&raw, Limits::default())?.digest())
}

pub fn regions_overlap(a: &TensorRegion, b: &TensorRegion) -> bool {
    if a.tensor != b.tensor || a.start.len() != b.start.len() {
        return false;
    }
    for ((sa, wa), (sb, wb)) in a
        .start
        .iter()
        .zip(&a.shape)
        .zip(b.start.iter().zip(&b.shape))
    {
        let ea = sa.saturating_add(*wa);
        let eb = sb.saturating_add(*wb);
        if ea <= *sb || eb <= *sa {
            return false;
        }
    }
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PatchSpec {
    pub patch_id: ContentDigest,
    pub base_model: ModelBinding,
    pub depends_on: Vec<ContentDigest>,
    pub program: Program,
    pub alias_groups: BTreeMap<String, String>,
    pub packing_groups: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReconciliationMode {
    None,
    OrderedSequential,
    ExplicitMergeRecipe,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConflictKind {
    pub left_patch: ContentDigest,
    pub right_patch: ContentDigest,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComposedPatchPlan {
    pub schema_version: u32,
    pub composed_program: Program,
    pub composed_program_digest: ContentDigest,
    pub execution_order: Vec<ContentDigest>,
    pub syntactic_conflicts: Vec<ConflictKind>,
    pub reconciled: bool,
    pub disjoint_writes: bool,
    pub behavioral_evaluation_required: bool,
    pub quantization_rebuild_required: bool,
}

fn ops_commute(a: OperationKind, b: OperationKind) -> bool {
    matches!(
        (a, b),
        (OperationKind::AddDelta, OperationKind::AddDelta)
            | (OperationKind::Scale, OperationKind::Scale)
    )
}

pub fn compose_patches(
    patches: &[PatchSpec],
    reconciliation: ReconciliationMode,
) -> Result<ComposedPatchPlan> {
    if patches.len() < 2 || patches.len() > 32 {
        return Err(err(
            ErrorCode::InvalidValue,
            "/patches",
            "composition requires between 2 and 32 patches",
        ));
    }

    let first_base = &patches[0].base_model;
    let by_id: BTreeMap<_, _> = patches.iter().map(|p| (p.patch_id.clone(), p)).collect();
    if by_id.len() != patches.len() {
        return Err(err(
            ErrorCode::DuplicateKey,
            "/patches",
            "duplicate patch_id in composition set",
        ));
    }

    // 1. Check common base (reject implicit rebases across different checkpoints/tokenizers/configs)
    for patch in &patches[1..] {
        if &patch.base_model != first_base {
            return Err(err(
                ErrorCode::SourceMismatch,
                "/base_model",
                "implicit rebase across different base models or tokenizers is forbidden",
            ));
        }
        if patch.program.effects.graph_changes != patches[0].program.effects.graph_changes
            || patch.program.effects.tokenizer_changes
                != patches[0].program.effects.tokenizer_changes
            || patch.program.effects.config_changes != patches[0].program.effects.config_changes
        {
            return Err(err(
                ErrorCode::UnsupportedCapability,
                "/effects",
                "incompatible graph/config/tokenizer effects across composed patches",
            ));
        }
    }

    // 2. Topological sort & cycle check
    let mut active = BTreeSet::new();
    let mut visited = BTreeSet::new();
    let mut order = Vec::new();

    fn dfs(
        id: &ContentDigest,
        by_id: &BTreeMap<ContentDigest, &PatchSpec>,
        active: &mut BTreeSet<ContentDigest>,
        visited: &mut BTreeSet<ContentDigest>,
        order: &mut Vec<ContentDigest>,
    ) -> Result<()> {
        if visited.contains(id) {
            return Ok(());
        }
        let spec = by_id.get(id).ok_or_else(|| {
            err(
                ErrorCode::SourceMismatch,
                "/depends_on",
                "missing prerequisite patch in composition set",
            )
        })?;
        if !active.insert(id.clone()) {
            return Err(err(
                ErrorCode::InvalidValue,
                "/depends_on",
                "cyclic patch dependency detected",
            ));
        }
        for dep in &spec.depends_on {
            dfs(dep, by_id, active, visited, order)?;
        }
        active.remove(id);
        visited.insert(id.clone());
        order.push(id.clone());
        Ok(())
    }

    for patch in patches {
        dfs(
            &patch.patch_id,
            &by_id,
            &mut active,
            &mut visited,
            &mut order,
        )?;
    }

    // 3. Pairwise conflict detection (write/write, read/write, shared alias group, coupled packing group, noncommuting ops)
    let mut conflicts = Vec::new();
    let mut quantization_rebuild = false;

    for i in 0..order.len() {
        for j in (i + 1)..order.len() {
            let left = by_id[&order[i]];
            let right = by_id[&order[j]];

            for lw in &left.program.effects.writes {
                let l_alias = left
                    .alias_groups
                    .get(&lw.tensor)
                    .cloned()
                    .unwrap_or_else(|| lw.tensor.clone());
                let l_pack = left.packing_groups.get(&lw.tensor);

                for rw in &right.program.effects.writes {
                    let r_alias = right
                        .alias_groups
                        .get(&rw.tensor)
                        .cloned()
                        .unwrap_or_else(|| rw.tensor.clone());
                    let r_pack = right.packing_groups.get(&rw.tensor);

                    let same_alias_overlap = l_alias == r_alias && regions_overlap(lw, rw);
                    let coupled_packing = l_pack.is_some() && l_pack == r_pack;
                    if coupled_packing {
                        quantization_rebuild = true;
                    }
                    if same_alias_overlap || coupled_packing {
                        let l_op = left
                            .program
                            .operations
                            .first()
                            .map(|ins| ins.operation.kind())
                            .unwrap_or(OperationKind::ReplaceSlice);
                        let r_op = right
                            .program
                            .operations
                            .first()
                            .map(|ins| ins.operation.kind())
                            .unwrap_or(OperationKind::ReplaceSlice);
                        let reason = if coupled_packing {
                            "coupled_quantization_packing_group".to_owned()
                        } else if !ops_commute(l_op, r_op) {
                            "noncommuting_overlapping_write".to_owned()
                        } else {
                            "overlapping_write_region".to_owned()
                        };
                        conflicts.push(ConflictKind {
                            left_patch: left.patch_id.clone(),
                            right_patch: right.patch_id.clone(),
                            reason,
                        });
                    }
                }

                for rr in &right.program.effects.reads {
                    let r_alias = right
                        .alias_groups
                        .get(&rr.tensor)
                        .cloned()
                        .unwrap_or_else(|| rr.tensor.clone());
                    if l_alias == r_alias && regions_overlap(lw, rr) {
                        conflicts.push(ConflictKind {
                            left_patch: left.patch_id.clone(),
                            right_patch: right.patch_id.clone(),
                            reason: "read_invalidated_by_earlier_write".to_owned(),
                        });
                    }
                }
            }
        }
    }

    let disjoint_writes = conflicts.is_empty();
    if !disjoint_writes && reconciliation == ReconciliationMode::None {
        return Err(err(
            ErrorCode::ScopeViolation,
            "/reconciliation",
            "overlapping, alias-coupled, or noncommuting patches require an explicit reconciliation mode",
        ));
    }

    // 4. Construct a new reconciled Program preserving lineage and combined effects
    let mut composed = by_id[&order[0]].program.clone();
    composed.operations.clear();
    composed.artifacts.clear();
    composed.effects.reads.clear();
    composed.effects.writes.clear();
    composed.lineage.parent_candidates = order.clone();

    let mut prev_inst_id: Option<String> = None;
    let mut total_changed_bound = 0u64;
    for (p_idx, patch_id) in order.iter().enumerate() {
        let spec = by_id[patch_id];
        total_changed_bound =
            total_changed_bound.saturating_add(spec.program.constraints.max_changed_parameters);
        for (art_id, art) in &spec.program.artifacts {
            composed
                .artifacts
                .insert(format!("p{p_idx}_{art_id}"), art.clone());
        }
        for r in &spec.program.effects.reads {
            if !composed.effects.reads.contains(r) {
                composed.effects.reads.push(r.clone());
            }
        }
        for w in &spec.program.effects.writes {
            if !composed.effects.writes.contains(w) {
                composed.effects.writes.push(w.clone());
            }
        }
        for inst in &spec.program.operations {
            let new_id = format!("p{p_idx}_{}", inst.id);
            let mut deps: Vec<String> = inst
                .depends_on
                .iter()
                .map(|d| format!("p{p_idx}_{d}"))
                .collect();
            if deps.is_empty() {
                if let Some(prev) = &prev_inst_id {
                    deps.push(prev.clone());
                }
            }
            prev_inst_id = Some(new_id.clone());
            composed.operations.push(Instruction {
                id: new_id,
                depends_on: deps,
                operation: inst.operation.clone(),
            });
        }
    }
    composed.constraints.max_changed_parameters = total_changed_bound;
    composed.lineage.method.changed_parameters = total_changed_bound;
    let composed_digest = canonical_digest(&composed)?;

    Ok(ComposedPatchPlan {
        schema_version: 1,
        composed_program: composed,
        composed_program_digest: composed_digest,
        execution_order: order,
        syntactic_conflicts: conflicts,
        reconciled: !disjoint_writes,
        disjoint_writes,
        // Even disjoint writes are only syntactically compatible; behavioral evaluation is always mandatory!
        behavioral_evaluation_required: true,
        quantization_rebuild_required: quantization_rebuild,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nf_weight_ir::program::parse_program;
    use serde_json::json;

    fn d(tag: &str) -> ContentDigest {
        ContentDigest::of_bytes(tag.as_bytes())
    }

    fn base_binding(tag: &str) -> ModelBinding {
        ModelBinding {
            checkpoint: d(tag),
            manifest: d("manifest"),
            parameter_digest: d("params"),
            architecture_graph: d("graph"),
            config: d("config"),
            tokenizer: None,
        }
    }

    fn sample_program(base: ModelBinding, region: TensorRegion, factor: f64) -> Program {
        let raw = json!({
            "ir_version": 1,
            "opset_version": 1,
            "base_model": base,
            "objective": {
                "contract": d("contract"),
                "objective": d("obj"),
                "experiment_protocol": d("proto"),
                "search_space": d("space")
            },
            "operations": [{
                "id": "op0",
                "depends_on": [],
                "operation": {"type": "scale", "target": region, "factor": factor}
            }],
            "artifacts": {},
            "preconditions": [{
                "tensor": region.tensor,
                "role": "parameter",
                "shape": [4, 4],
                "dtype": "f64",
                "before_digest": d("before"),
                "storage_group": region.tensor
            }],
            "required_capabilities": ["dense.cpu"],
            "constraints": {
                "gradient_free": true,
                "preserve_tensor_shapes": true,
                "max_changed_parameters": 4,
                "max_artifact_bytes": 1024,
                "max_temporary_bytes": 4096,
                "numerical": {
                    "accumulation_dtype": "f64",
                    "output_rounding": "nearest_ties_even",
                    "non_finite": "reject",
                    "max_abs_parameter": 100.0,
                    "backend_abs_tolerance": 0.0
                }
            },
            "effects": {
                "reads": [region],
                "writes": [region],
                "buffer_writes": [],
                "graph_changes": false,
                "config_changes": [],
                "tokenizer_changes": false
            },
            "preservation": {"type": "experimental"},
            "verification": {
                "portfolio": d("portfolio"),
                "protocol": d("vproto"),
                "gyms": [{
                    "binding": d("b0"),
                    "gym": d("g0"),
                    "validation": d("val0"),
                    "scorer": d("sc0"),
                    "split": d("sel0"),
                    "role": "target"
                }],
                "smoke_suite": d("smoke"),
                "parity_suite": d("parity")
            },
            "lineage": {
                "parent_candidates": [],
                "hypothesis": d("hyp"),
                "method": {
                    "schema_version": 1,
                    "target_update": "direct_tensor_edit",
                    "proposal_origin": "scripted",
                    "adaptation": "none",
                    "production_history": "direct_only",
                    "inspection_uses_backprop": false,
                    "parameter_update_uses_backprop": false,
                    "gradient_derived_update_values": false,
                    "alignment_fitted": false,
                    "optimizer_steps": 0,
                    "changed_parameters": 4
                }
            },
            "extensions": {}
        });
        (*parse_program(&serde_json::to_vec(&raw).unwrap()).unwrap()).clone()
    }

    #[test]
    fn composition_disjoint_writes_succeed_and_still_require_behavioral_evaluation() {
        let base = base_binding("ckpt-1");
        let p1 = PatchSpec {
            patch_id: d("patch-1"),
            base_model: base.clone(),
            depends_on: vec![],
            program: sample_program(
                base.clone(),
                TensorRegion {
                    tensor: "w".into(),
                    start: vec![0, 0],
                    shape: vec![2, 2],
                },
                0.5,
            ),
            alias_groups: BTreeMap::new(),
            packing_groups: BTreeMap::new(),
        };
        let p2 = PatchSpec {
            patch_id: d("patch-2"),
            base_model: base,
            depends_on: vec![d("patch-1")],
            program: sample_program(
                p1.base_model.clone(),
                TensorRegion {
                    tensor: "w".into(),
                    start: vec![2, 2],
                    shape: vec![2, 2],
                },
                1.5,
            ),
            alias_groups: BTreeMap::new(),
            packing_groups: BTreeMap::new(),
        };

        let plan = compose_patches(&[p1, p2], ReconciliationMode::None).unwrap();
        assert!(plan.disjoint_writes);
        assert!(plan.syntactic_conflicts.is_empty());
        assert!(plan.behavioral_evaluation_required);
        assert_eq!(plan.composed_program.operations.len(), 2);
    }

    #[test]
    fn composition_rejects_cycles_implicit_rebases_and_unreconciled_overlaps() {
        let base_a = base_binding("ckpt-a");
        let base_b = base_binding("ckpt-b");
        let reg = TensorRegion {
            tensor: "w".into(),
            start: vec![0, 0],
            shape: vec![2, 2],
        };

        // 1. Implicit rebase across different base checkpoints rejected
        let p_a = PatchSpec {
            patch_id: d("p-a"),
            base_model: base_a.clone(),
            depends_on: vec![],
            program: sample_program(base_a.clone(), reg.clone(), 0.5),
            alias_groups: BTreeMap::new(),
            packing_groups: BTreeMap::new(),
        };
        let p_b = PatchSpec {
            patch_id: d("p-b"),
            base_model: base_b.clone(),
            depends_on: vec![],
            program: sample_program(base_b, reg.clone(), 0.8),
            alias_groups: BTreeMap::new(),
            packing_groups: BTreeMap::new(),
        };
        assert_eq!(
            compose_patches(&[p_a.clone(), p_b], ReconciliationMode::OrderedSequential)
                .unwrap_err()
                .code,
            ErrorCode::SourceMismatch
        );

        // 2. Dependency cycle rejected
        let mut cyc1 = p_a.clone();
        cyc1.depends_on = vec![d("p-2")];
        let cyc2 = PatchSpec {
            patch_id: d("p-2"),
            base_model: base_a.clone(),
            depends_on: vec![d("p-a")],
            program: sample_program(base_a.clone(), reg.clone(), 0.8),
            alias_groups: BTreeMap::new(),
            packing_groups: BTreeMap::new(),
        };
        assert_eq!(
            compose_patches(&[cyc1, cyc2], ReconciliationMode::OrderedSequential)
                .unwrap_err()
                .code,
            ErrorCode::InvalidValue
        );

        // 3. Overlapping writes or coupled packing groups fail without reconciliation, succeed with OrderedSequential
        let mut pack2 = p_a.clone();
        pack2.patch_id = d("p-overlap");
        pack2
            .packing_groups
            .insert("w".into(), "quant_group_0".into());
        let mut pack1 = p_a;
        pack1
            .packing_groups
            .insert("w".into(), "quant_group_0".into());

        assert_eq!(
            compose_patches(&[pack1.clone(), pack2.clone()], ReconciliationMode::None)
                .unwrap_err()
                .code,
            ErrorCode::ScopeViolation
        );
        let reconciled =
            compose_patches(&[pack1, pack2], ReconciliationMode::OrderedSequential).unwrap();
        assert!(reconciled.reconciled);
        assert!(reconciled.quantization_rebuild_required);
        assert!(reconciled.behavioral_evaluation_required);
    }
}
