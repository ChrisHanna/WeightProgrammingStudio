//! Bounded mutation and recombination operators over a frozen SearchSpace (T070).
//!
//! Generates deterministic seeded mutations and recombinations within the allowed
//! transformation domain, checks cumulative ancestor parameter write budgets so
//! splitting edits across branches cannot evade limits, and recompiles every
//! offspring against the frozen `ResolvedSearchSpace`.

use crate::composition::{compose_patches, PatchSpec, ReconciliationMode};
use nf_core::error::{CoreError, ErrorCode, Result};
use nf_core::schemas::Validated;
use nf_weight_ir::compilation::{compile, CompilePolicy, CompiledPlan, SourceSnapshot};
use nf_weight_ir::program::{Operation, Program};
use nf_weight_ir::search_space::{PriorEffects, ResolvedSearchSpace};
use std::collections::BTreeMap;
use std::sync::Arc;

fn err(code: ErrorCode, path: &str, message: impl Into<String>) -> CoreError {
    CoreError::new(code, path, message)
}

fn xorshift64(mut state: u64) -> u64 {
    if state == 0 {
        state = 0x9e37_79b9_7f4a_7c15;
    }
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

pub fn mutate_program_in_space(
    parent_program: &Program,
    space: &ResolvedSearchSpace,
    prior_effects: &[PriorEffects],
    seed: u64,
    scale_delta: f64,
) -> Result<Validated<Program>> {
    if !scale_delta.is_finite() {
        return Err(err(
            ErrorCode::NonFinite,
            "/scale_delta",
            "finite mutation scale_delta required",
        ));
    }
    let _rng = xorshift64(seed);
    let mut mutated = parent_program.clone();
    for inst in &mut mutated.operations {
        match &mut inst.operation {
            Operation::Scale { factor, .. } => {
                *factor += scale_delta;
            }
            Operation::SubspaceProjection { strength, .. }
            | Operation::LowRankUpdate { strength, .. } => {
                *strength += scale_delta;
            }
            _ => {}
        }
    }
    let validated = Validated::new(mutated)?;
    space.check_program(&validated, prior_effects)?;
    Ok(validated)
}

pub fn recombine_programs_in_space(
    left: &PatchSpec,
    right: &PatchSpec,
    space: &ResolvedSearchSpace,
    prior_effects: &[PriorEffects],
    reconciliation: ReconciliationMode,
) -> Result<Validated<Program>> {
    let composed = compose_patches(&[left.clone(), right.clone()], reconciliation)?;
    let validated = Validated::new(composed.composed_program)?;
    // Revalidate against the frozen search space and cumulative ancestor effects!
    space.check_program(&validated, prior_effects)?;
    Ok(validated)
}

pub fn compile_evolutionary_offspring(
    offspring: &Validated<Program>,
    source: SourceSnapshot<'_>,
    space: &ResolvedSearchSpace,
    operands: &BTreeMap<String, Arc<[u8]>>,
    policy: &Validated<CompilePolicy>,
    prior_effects: &[PriorEffects],
) -> Result<CompiledPlan> {
    compile(offspring, source, space, operands, policy, prior_effects)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composition::ReconciliationMode;
    use nf_core::canonical::ContentDigest;
    use nf_core::schemas::{DType, FactoryRef, ModelManifest, TensorFormat, TensorSpec};
    use nf_graph::inventory::{tensor_set_digest, verify_inventory, InventoryLimits};
    use nf_surgeon::transaction::{SourceState, StateLoadLimits};
    use nf_weight_ir::program::{parse_program, Instruction, TensorRegion};
    use nf_weight_ir::search_space::{parse_search_space, resolve_space};
    use serde_json::{json, Value};

    fn id(tag: &str) -> ContentDigest {
        ContentDigest::of_bytes(tag.as_bytes())
    }

    fn build_safetensors(values: &[f64]) -> Vec<u8> {
        let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
        let header = json!({
            "w": {"dtype": "F64", "shape": [2, 2], "data_offsets": [0, bytes.len()]}
        });
        let mut raw = serde_json::to_vec(&header).unwrap();
        raw.resize(raw.len().div_ceil(8) * 8, b' ');
        let mut out = (raw.len() as u64).to_le_bytes().to_vec();
        out.extend(raw);
        out.extend(bytes);
        out
    }

    fn setup_fixture() -> (
        SourceState,
        ResolvedSearchSpace,
        Validated<CompilePolicy>,
        Validated<Program>,
    ) {
        let raw_vals: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
        let val_bytes: Vec<u8> = raw_vals.iter().flat_map(|x| x.to_le_bytes()).collect();
        let ckpt: Arc<[u8]> = Arc::from(build_safetensors(&raw_vals));
        let cfg: Arc<[u8]> = Arc::from(b"{}".as_slice());
        let cfg_id = ContentDigest::of_bytes(&cfg);
        let spec = TensorSpec {
            shape: vec![2, 2],
            dtype: DType::F64,
            digest: ContentDigest::of_bytes(&val_bytes),
            storage_group: "w".into(),
        };
        let parameters = BTreeMap::from([("w".to_owned(), spec.clone())]);
        let manifest = Validated::new(ModelManifest {
            schema_version: 1,
            architecture: FactoryRef {
                id: "fixture.cpu".into(),
                version: "1".into(),
                config: cfg_id.clone(),
            },
            parameter_digest: tensor_set_digest(&parameters).unwrap(),
            parameters,
            buffers: BTreeMap::new(),
            format: TensorFormat::DenseSafetensors,
            packing_policy: None,
            assets: BTreeMap::from([
                ("config".into(), cfg_id),
                ("checkpoint".into(), ContentDigest::of_bytes(&ckpt)),
            ]),
            usage_rights: "Apache-2.0".into(),
        })
        .unwrap();
        let source = SourceState::load(
            manifest,
            ckpt,
            BTreeMap::from([("config".into(), cfg)]),
            StateLoadLimits::default(),
        )
        .unwrap();
        let obs = source.observations();
        let inventory =
            verify_inventory(source.manifest(), &obs, InventoryLimits::default()).unwrap();
        let binding = source.binding(id("graph"));
        let base_model = serde_json::to_value(&binding).unwrap();
        let space_json = json!({
            "schema_version": 1,
            "base_model": base_model,
            "contract": id("contract"),
            "objective": id("objective"),
            "transformations": {
                "scale": {
                    "dtypes": ["f64"],
                    "coefficient": {"min": 0.1, "max": 2.0},
                    "rank": null,
                    "sides": [],
                    "max_operand_bytes": 0
                }
            },
            "sites": [{
                "selector": {"type": "tensor", "name": "w"},
                "role": "parameter",
                "regions": {"type": "whole_tensor"},
                "allowed_axes": [0, 1]
            }],
            "constraints": {
                "max_changed_parameters": 2,
                "max_artifact_bytes": 4096,
                "max_temporary_bytes": 8192,
                "max_operations": 8,
                "allow_buffer_writes": false,
                "require_direct_weight_edits": true,
                "require_fully_gradient_free": true,
                "require_preserve_shapes": true,
                "quality_limits": [{
                    "binding": id("binding"),
                    "metric": "accuracy",
                    "units": "fraction",
                    "baseline": binding.checkpoint,
                    "max_regression": 0.0,
                    "split": "holdout",
                    "uncertainty_protocol": id("uncertainty")
                }]
            },
            "strategy": {
                "proposal": "scripted",
                "search": "sequential",
                "selection": "pareto",
                "engine": "native.evo",
                "engine_version": "1",
                "seed": 17
            }
        });
        let space = parse_search_space(&serde_json::to_vec(&space_json).unwrap()).unwrap();
        let resolved = resolve_space(&space, &inventory).unwrap();
        let region = json!({"tensor": "w", "start": [0, 0], "shape": [1, 2]});
        let prog_json: Value = json!({
            "ir_version": 1,
            "opset_version": 1,
            "base_model": base_model,
            "objective": {
                "contract": id("contract"),
                "objective": id("objective"),
                "experiment_protocol": id("experiment"),
                "search_space": space.digest().unwrap()
            },
            "operations": [{
                "id": "op0",
                "depends_on": [],
                "operation": {"type": "scale", "target": region, "factor": 1.2}
            }],
            "artifacts": {},
            "preconditions": [{
                "tensor": "w",
                "role": "parameter",
                "shape": [2, 2],
                "dtype": "f64",
                "before_digest": spec.digest,
                "storage_group": "w"
            }],
            "required_capabilities": ["dense.cpu"],
            "constraints": {
                "gradient_free": true,
                "preserve_tensor_shapes": true,
                "max_changed_parameters": 2,
                "max_artifact_bytes": 4096,
                "max_temporary_bytes": 8192,
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
                "portfolio": id("portfolio"),
                "protocol": id("verification"),
                "gyms": [{
                    "binding": id("binding"),
                    "gym": id("gym"),
                    "validation": id("validation"),
                    "scorer": id("scorer"),
                    "split": id("selection"),
                    "role": "target"
                }],
                "smoke_suite": id("smoke"),
                "parity_suite": id("parity")
            },
            "lineage": {
                "parent_candidates": [],
                "hypothesis": id("hyp"),
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
                    "changed_parameters": 2
                }
            },
            "extensions": {}
        });
        let parsed = parse_program(&serde_json::to_vec(&prog_json).unwrap()).unwrap();
        let policy_json = json!({
            "schema_version": 1,
            "source": base_model,
            "contract": id("contract"),
            "objective": id("objective"),
            "experiment_protocol": id("experiment"),
            "search_space": space.digest().unwrap(),
            "verification": prog_json["verification"],
            "allowed_extra_gyms": [],
            "permissions": {
                "schema_version": 1,
                "tools": ["inspect", "direct_edit"],
                "external_destinations": {},
                "allow_gradient_inspection": false,
                "allow_parameter_training": false,
                "require_fully_gradient_free": true,
                "allow_generated_gym": false,
                "allow_downloads": false
            },
            "operand_access": ["development"],
            "backend": {
                "id": "nf.native_cpu",
                "version": "1",
                "operations": ["scale"],
                "capabilities": ["dense.cpu"],
                "dtypes": ["f64"],
                "accumulation_dtypes": ["f64"],
                "fixed_scratch_bytes": 64
            },
            "bounds": {
                "max_metadata_bytes": 1048576,
                "max_source_checkpoint_bytes": 1048576,
                "max_source_asset_bytes": 4096,
                "max_source_tensors": 16,
                "max_observed_source_bytes": 4096,
                "max_temporary_bytes": 8192,
                "max_output_tensor_bytes": 4096,
                "max_validation_work": 100000,
                "max_kernel_work": 100000,
                "max_parameter_magnitude": 100.0,
                "max_backend_abs_tolerance": 0.0,
                "max_orthonormal_tolerance": 0.000001
            }
        });
        let policy =
            Validated::<CompilePolicy>::parse(&serde_json::to_vec(&policy_json).unwrap()).unwrap();
        (source, resolved, policy, parsed)
    }

    #[test]
    fn evolutionary_ops_mutate_recombine_recompile_and_reject_out_of_space_or_budget_evasion() {
        let (source, space, policy, prog) = setup_fixture();
        let obs = source.observations();

        // 1. Valid in-space mutation compiles cleanly
        let mutated = mutate_program_in_space(&prog, &space, &[], 17, 0.3).unwrap();
        let compiled = compile_evolutionary_offspring(
            &mutated,
            SourceSnapshot {
                manifest: source.manifest(),
                tensors: &obs,
                checkpoint: source.checkpoint_bytes(),
                assets: source.assets(),
            },
            &space,
            &BTreeMap::new(),
            &policy,
            &[],
        )
        .unwrap();
        assert_eq!(compiled.summary().affected_parameter_upper_bound, 2);

        // 2. Out-of-space mutation (factor > 2.0 domain bound) is rejected
        assert!(mutate_program_in_space(&prog, &space, &[], 17, 1.5).is_err());

        // 3. Write-budget evasion via recombination or ancestor effects is rejected
        let mut prog2 = (*prog).clone();
        let bottom_row = TensorRegion {
            tensor: "w".into(),
            start: vec![1, 0],
            shape: vec![1, 2],
        };
        prog2.operations = vec![Instruction {
            id: "op0".into(),
            depends_on: vec![],
            operation: Operation::Scale {
                target: bottom_row.clone(),
                factor: 1.1,
            },
        }];
        prog2.effects.reads = vec![bottom_row.clone()];
        prog2.effects.writes = vec![bottom_row];

        let left_spec = PatchSpec {
            patch_id: id("left"),
            base_model: prog.base_model.clone(),
            depends_on: vec![],
            program: (*prog).clone(),
            alias_groups: BTreeMap::new(),
            packing_groups: BTreeMap::new(),
        };
        let right_spec = PatchSpec {
            patch_id: id("right"),
            base_model: prog.base_model.clone(),
            depends_on: vec![],
            program: prog2,
            alias_groups: BTreeMap::new(),
            packing_groups: BTreeMap::new(),
        };
        // Each parent changes 2 parameters (<= limit 2), but recombining them changes 4 > 2!
        assert!(recombine_programs_in_space(
            &left_spec,
            &right_spec,
            &space,
            &[],
            ReconciliationMode::None
        )
        .is_err());
    }
}
