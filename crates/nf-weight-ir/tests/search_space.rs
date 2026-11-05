mod common;

use common::{digest, program, target};
use nf_core::canonical::ContentDigest;
use nf_core::error::ErrorCode;
use nf_core::schemas::{DType, FactoryRef, ModelManifest, TensorFormat, TensorSpec, Validated};
use nf_graph::inventory::{
    tensor_set_digest, verify_inventory, InventoryLimits, TensorObservation,
    TensorRole as ObservedRole, VerifiedInventory,
};
use nf_weight_ir::program::{parse_program, ModelBinding, OperationKind, TensorRole};
use nf_weight_ir::search_space::*;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

struct Fixture {
    space: SearchSpace,
    program: Value,
    inventory: VerifiedInventory,
}
impl Fixture {
    fn new(tied: bool) -> Self {
        let data = vec![0u8; 32];
        let spec = TensorSpec {
            shape: vec![2, 2],
            dtype: DType::F64,
            digest: ContentDigest::of_bytes(&data),
            storage_group: "shared".to_owned(),
        };
        let mut parameters = BTreeMap::from([("w".to_owned(), spec.clone())]);
        if tied {
            parameters.insert("alias".to_owned(), spec);
        }
        let manifest = Validated::new(ModelManifest {
            schema_version: 1,
            architecture: FactoryRef {
                id: "fixture.cpu".to_owned(),
                version: "1".to_owned(),
                config: digest("config").parse().unwrap(),
            },
            parameter_digest: tensor_set_digest(&parameters).unwrap(),
            parameters,
            buffers: BTreeMap::new(),
            format: TensorFormat::DenseSafetensors,
            packing_policy: None,
            assets: BTreeMap::new(),
            usage_rights: "synthetic".to_owned(),
        })
        .unwrap();
        let mut observations = vec![TensorObservation {
            name: "w",
            role: ObservedRole::Parameter,
            shape: &[2, 2],
            dtype: DType::F64,
            bytes: &data,
        }];
        if tied {
            observations.push(TensorObservation {
                name: "alias",
                role: ObservedRole::Parameter,
                shape: &[2, 2],
                dtype: DType::F64,
                bytes: &data,
            });
        }
        let inventory =
            verify_inventory(&manifest, &observations, InventoryLimits::default()).unwrap();
        let mut value = program();
        value["base_model"]["manifest"] = json!(inventory.source_manifest_digest());
        value["base_model"]["parameter_digest"] = json!(inventory.parameter_digest());
        value["preconditions"] = json!(inventory
            .entries()
            .iter()
            .map(|(name, entry)| json!({
                "tensor":name,"role":"parameter","shape":entry.shape,"dtype":entry.dtype,
                "before_digest":entry.content_digest,"storage_group":entry.storage_group,
            }))
            .collect::<Vec<_>>());
        let base: ModelBinding = serde_json::from_value(value["base_model"].clone()).unwrap();
        let space = SearchSpace {
            schema_version: 1,
            base_model: base,
            contract: digest("contract").parse().unwrap(),
            objective: digest("objective").parse().unwrap(),
            transformations: BTreeMap::from([
                (
                    OperationKind::ReplaceSlice,
                    OperationDomain {
                        dtypes: vec![DType::F64],
                        coefficient: None,
                        rank: None,
                        sides: BTreeSet::new(),
                        max_operand_bytes: 128,
                    },
                ),
                (
                    OperationKind::Scale,
                    OperationDomain {
                        dtypes: vec![DType::F64],
                        coefficient: Some(FloatRange {
                            min: -1.0,
                            max: 1.0,
                        }),
                        rank: None,
                        sides: BTreeSet::new(),
                        max_operand_bytes: 0,
                    },
                ),
            ]),
            sites: vec![site("w")],
            constraints: SpaceConstraints {
                max_changed_parameters: 4,
                max_artifact_bytes: 128,
                max_temporary_bytes: 4096,
                max_operations: 8,
                allow_buffer_writes: false,
                require_direct_weight_edits: true,
                require_fully_gradient_free: false,
                require_preserve_shapes: true,
                quality_limits: vec![QualityLimit {
                    binding: digest("binding").parse().unwrap(),
                    metric: "accuracy".to_owned(),
                    units: "fraction".to_owned(),
                    baseline: digest("checkpoint").parse().unwrap(),
                    max_regression: 0.01,
                    split: EvaluationSplit::Holdout,
                    uncertainty_protocol: digest("uncertainty").parse().unwrap(),
                }],
            },
            strategy: StrategyBinding {
                proposal: ProposalStrategy::CausalHypothesis,
                search: SearchStrategy::Branch,
                selection: SelectionStrategy::Pareto,
                engine: "bounded.branch".to_owned(),
                engine_version: "1".to_owned(),
                seed: 17,
            },
        };
        let mut fixture = Self {
            space,
            program: value,
            inventory,
        };
        fixture.bind();
        fixture
    }
    fn bind(&mut self) {
        self.program["objective"]["search_space"] = json!(self.space.digest().unwrap());
    }
    fn resolve(&self) -> ResolvedSearchSpace {
        resolve_space(
            &Validated::new(self.space.clone()).unwrap(),
            &self.inventory,
        )
        .unwrap()
    }
    fn parsed(&self) -> Validated<nf_weight_ir::program::Program> {
        parse_program(&serde_json::to_vec(&self.program).unwrap()).unwrap()
    }
}
fn site(name: &str) -> SiteRule {
    SiteRule {
        selector: Selector::Tensor {
            name: name.to_owned(),
        },
        role: TensorRole::Parameter,
        regions: RegionDomain::WholeTensor,
        allowed_axes: [0, 1].into(),
    }
}
fn rectangle(start: [u64; 2], shape: [u64; 2]) -> Value {
    json!({"tensor":"w","start":start,"shape":shape})
}
fn set_scale(fixture: &mut Fixture, regions: &[Value]) {
    fixture.program["operations"] = json!(regions.iter().enumerate().map(|(i,region)| json!({"id":format!("op{i}"),"depends_on":[],"operation":{"type":"scale","target":region,"factor":0.5}})).collect::<Vec<_>>());
    fixture.program["effects"]["reads"] = json!(regions);
    fixture.program["effects"]["writes"] = json!(regions);
    fixture.bind();
}

#[test]
fn search_space_same_shape_parent_cannot_change_frozen_alias_layout() {
    let tied = Fixture::new(true);
    let untied = Fixture::new(false);
    let resolved = tied.resolve();
    assert!(resolved.check_inventory_layout(&tied.inventory).is_ok());
    assert_eq!(
        resolved
            .check_inventory_layout(&untied.inventory)
            .unwrap_err()
            .code,
        ErrorCode::SourceMismatch
    );
}

#[test]
fn search_space_resolves_actual_inventory_and_keeps_quality_pending() {
    let fixture = Fixture::new(false);
    let space = fixture.resolve();
    assert_eq!(space.resolved_tensor_names(), ["w"]);
    let result = space.check_program(&fixture.parsed(), &[]).unwrap();
    assert_eq!(result.affected_parameter_upper_bound, 4);
    assert_eq!(result.pending_quality_constraints, 1);
    assert_eq!(&result.space_digest, space.digest());
}

#[test]
fn search_space_valid_ir_can_still_be_outside_operation_or_numeric_domain() {
    let mut fixture = Fixture::new(false);
    fixture.program["operations"][0]["operation"] =
        json!({"type":"mask","target":target(),"mask":"operand"});
    assert_eq!(
        fixture
            .resolve()
            .check_program(&fixture.parsed(), &[])
            .unwrap_err()
            .code,
        ErrorCode::UnsupportedOperation
    );
    set_scale(&mut fixture, &[target()]);
    fixture.program["operations"][0]["operation"]["factor"] = json!(2.0);
    assert!(fixture
        .resolve()
        .check_program(&fixture.parsed(), &[])
        .is_err());
}

#[test]
fn search_space_selectors_regions_and_axes_fail_closed() {
    let mut fixture = Fixture::new(false);
    fixture.space.sites[0].selector = Selector::Prefix {
        prefix: "missing.".to_owned(),
    };
    let validated = Validated::new(fixture.space.clone()).unwrap();
    assert!(resolve_space(&validated, &fixture.inventory).is_err());
    fixture.space.sites[0].selector = Selector::Prefix {
        prefix: "w".to_owned(),
    };
    fixture.space.sites[0].regions = RegionDomain::Boxes {
        regions: vec![SliceBox {
            start: vec![0, 0],
            shape: vec![1, 2],
        }],
    };
    fixture.bind();
    assert!(fixture
        .resolve()
        .check_program(&fixture.parsed(), &[])
        .is_err());
    fixture.space.sites[0].regions = RegionDomain::WholeTensor;
    fixture.space.sites[0].allowed_axes = [1].into();
    fixture.space.transformations.insert(
        OperationKind::PermuteAxis,
        OperationDomain {
            dtypes: vec![DType::F64],
            coefficient: None,
            rank: None,
            sides: BTreeSet::new(),
            max_operand_bytes: 0,
        },
    );
    fixture.bind();
    fixture.program["operations"][0]["operation"] =
        json!({"type":"permute_axis","target":target(),"axis":0,"permutation":[1,0]});
    assert!(fixture
        .resolve()
        .check_program(&fixture.parsed(), &[])
        .is_err());
}

#[test]
fn search_space_aliases_need_coordinate_and_axis_permission_without_double_counting() {
    let mut fixture = Fixture::new(true);
    assert_eq!(
        fixture
            .resolve()
            .check_program(&fixture.parsed(), &[])
            .unwrap_err()
            .code,
        ErrorCode::AliasScopeEscape
    );
    fixture.space.sites.push(site("alias"));
    fixture.bind();
    assert_eq!(
        fixture
            .resolve()
            .check_program(&fixture.parsed(), &[])
            .unwrap()
            .affected_parameter_upper_bound,
        4
    );
    fixture.space.sites[1].allowed_axes = [1].into();
    fixture.space.transformations.insert(
        OperationKind::PermuteAxis,
        OperationDomain {
            dtypes: vec![DType::F64],
            coefficient: None,
            rank: None,
            sides: BTreeSet::new(),
            max_operand_bytes: 0,
        },
    );
    fixture.bind();
    fixture.program["operations"][0]["operation"] =
        json!({"type":"permute_axis","target":target(),"axis":0,"permutation":[1,0]});
    assert_eq!(
        fixture
            .resolve()
            .check_program(&fixture.parsed(), &[])
            .unwrap_err()
            .code,
        ErrorCode::AliasScopeEscape
    );
}

#[test]
fn search_space_union_counts_overlaps_without_parameter_enumeration() {
    let mut fixture = Fixture::new(false);
    let boxes = [rectangle([0, 0], [1, 2]), rectangle([0, 1], [2, 1])];
    set_scale(&mut fixture, &boxes);
    assert_eq!(
        fixture
            .resolve()
            .check_program(&fixture.parsed(), &[])
            .unwrap()
            .affected_parameter_upper_bound,
        3
    );
    let mut rectangles = Vec::new();
    for x in 0..2 {
        for y in 0..2 {
            for width in 1..=2 - x {
                for height in 1..=2 - y {
                    rectangles.push((rectangle([x, y], [width, height]), x, y, width, height));
                }
            }
        }
    }
    for a in &rectangles {
        for b in &rectangles {
            set_scale(&mut fixture, &[a.0.clone(), b.0.clone()]);
            let mut cells = BTreeSet::new();
            for item in [a, b] {
                for x in item.1..item.1 + item.3 {
                    for y in item.2..item.2 + item.4 {
                        cells.insert((x, y));
                    }
                }
            }
            assert_eq!(
                fixture
                    .resolve()
                    .check_program(&fixture.parsed(), &[])
                    .unwrap()
                    .affected_parameter_upper_bound,
                cells.len() as u64
            );
        }
    }
}

#[test]
fn search_space_committed_parent_scope_is_cumulative_and_source_bound() {
    let mut fixture = Fixture::new(false);
    set_scale(&mut fixture, &[rectangle([1, 0], [1, 2])]);
    let resolved = fixture.resolve();
    let mut resulting = fixture.space.base_model.clone();
    resulting.checkpoint = digest("parent-checkpoint").parse().unwrap();
    resulting.manifest = digest("parent-manifest").parse().unwrap();
    resulting.parameter_digest = digest("parent-parameters").parse().unwrap();
    let parent = PriorEffects {
        candidate: digest("parent").parse().unwrap(),
        root_manifest: fixture.space.base_model.manifest.clone(),
        search_space: resolved.digest().clone(),
        resulting_model: resulting.clone(),
        cumulative_writes: vec![serde_json::from_value(rectangle([0, 0], [1, 2])).unwrap()],
    };
    fixture.program["base_model"] = serde_json::to_value(resulting).unwrap();
    fixture.program["lineage"]["parent_candidates"] = json!([parent.candidate]);
    assert!(resolved.check_program(&fixture.parsed(), &[]).is_err());
    assert_eq!(
        resolved
            .check_program(&fixture.parsed(), std::slice::from_ref(&parent))
            .unwrap()
            .affected_parameter_upper_bound,
        4
    );
    fixture.program["constraints"]["max_changed_parameters"] = json!(3);
    assert_eq!(
        resolved
            .check_program(&fixture.parsed(), std::slice::from_ref(&parent))
            .unwrap_err()
            .code,
        ErrorCode::ParameterBudgetExceeded
    );
    let mut wrong = parent;
    wrong.root_manifest = digest("wrong-root").parse().unwrap();
    assert_eq!(
        resolved
            .check_program(&fixture.parsed(), &[wrong])
            .unwrap_err()
            .code,
        ErrorCode::SourceMismatch
    );
}

#[test]
fn search_space_rank_dtype_and_artifact_cost_domains_are_enforced() {
    let mut fixture = Fixture::new(false);
    fixture.space.transformations.insert(
        OperationKind::LowRankUpdate,
        OperationDomain {
            dtypes: vec![DType::F64],
            coefficient: Some(FloatRange { min: 0.0, max: 1.0 }),
            rank: Some(RankRange { min: 1, max: 1 }),
            sides: BTreeSet::new(),
            max_operand_bytes: 128,
        },
    );
    fixture.bind();
    fixture.program["operations"][0]["operation"] = json!({"type":"low_rank_update","target":target(),"left":"operand","right":"operand","strength":1.0});
    assert!(fixture
        .resolve()
        .check_program(&fixture.parsed(), &[])
        .is_err());
    fixture
        .space
        .transformations
        .get_mut(&OperationKind::LowRankUpdate)
        .unwrap()
        .rank
        .as_mut()
        .unwrap()
        .max = 2;
    fixture.bind();
    assert!(fixture
        .resolve()
        .check_program(&fixture.parsed(), &[])
        .is_ok());
    fixture
        .space
        .transformations
        .get_mut(&OperationKind::LowRankUpdate)
        .unwrap()
        .max_operand_bytes = 127;
    fixture.bind();
    assert!(fixture
        .resolve()
        .check_program(&fixture.parsed(), &[])
        .is_err());
    let mut fixture = Fixture::new(false);
    fixture
        .space
        .transformations
        .get_mut(&OperationKind::ReplaceSlice)
        .unwrap()
        .dtypes = vec![DType::F32];
    fixture.bind();
    assert!(fixture
        .resolve()
        .check_program(&fixture.parsed(), &[])
        .is_err());
}

#[test]
fn search_space_directness_and_inspection_are_not_conflated() {
    let mut fixture = Fixture::new(false);
    fixture.program["lineage"]["method"]["inspection_uses_backprop"] = json!(true);
    assert!(fixture
        .resolve()
        .check_program(&fixture.parsed(), &[])
        .is_ok());
    fixture.space.constraints.require_fully_gradient_free = true;
    fixture.bind();
    assert!(fixture
        .resolve()
        .check_program(&fixture.parsed(), &[])
        .is_err());
    fixture.space.constraints.require_fully_gradient_free = false;
    fixture.bind();
    fixture.program["constraints"]["gradient_free"] = json!(false);
    fixture.program["lineage"]["method"]["production_history"] = json!("hybrid");
    fixture.program["lineage"]["method"]["alignment_fitted"] = json!(true);
    assert!(fixture
        .resolve()
        .check_program(&fixture.parsed(), &[])
        .is_err());
}

#[test]
fn search_space_revisions_and_required_quality_bindings_cannot_be_substituted() {
    let mut fixture = Fixture::new(false);
    let frozen = fixture.resolve();
    fixture.space.constraints.max_changed_parameters = 8;
    fixture.bind();
    assert_eq!(
        frozen
            .check_program(&fixture.parsed(), &[])
            .unwrap_err()
            .code,
        ErrorCode::SourceMismatch
    );
    let mut fixture = Fixture::new(false);
    fixture.program["verification"]["gyms"][0]["binding"] = json!(digest("easier-gym"));
    assert!(fixture
        .resolve()
        .check_program(&fixture.parsed(), &[])
        .is_err());
}

#[test]
fn search_space_projection_side_and_strength_remain_bounded() {
    let mut fixture = Fixture::new(false);
    fixture.space.transformations.insert(
        OperationKind::SubspaceProjection,
        OperationDomain {
            dtypes: vec![DType::F64],
            coefficient: Some(FloatRange { min: 0.0, max: 1.0 }),
            rank: Some(RankRange { min: 1, max: 2 }),
            sides: [nf_weight_ir::program::MapSide::Right].into(),
            max_operand_bytes: 128,
        },
    );
    fixture.bind();
    fixture.program["operations"][0]["operation"] = json!({"type":"subspace_projection","target":target(),"basis":"operand","side":"left","strength":0.7,"orthonormal_tolerance":0.0});
    assert!(fixture
        .resolve()
        .check_program(&fixture.parsed(), &[])
        .is_err());
    fixture.program["operations"][0]["operation"]["side"] = json!("right");
    assert!(fixture
        .resolve()
        .check_program(&fixture.parsed(), &[])
        .is_ok());
    fixture.program["operations"][0]["operation"]["strength"] = json!(1.5);
    assert!(fixture
        .resolve()
        .check_program(&fixture.parsed(), &[])
        .is_err());
}

#[test]
fn search_space_resolved_expansion_is_globally_bounded() {
    let fixture = Fixture::new(false);
    let mut repeated = site("w");
    repeated.regions = RegionDomain::Boxes {
        regions: vec![
            SliceBox {
                start: vec![0, 0],
                shape: vec![2, 2]
            };
            5
        ],
    };
    let mut space = fixture.space;
    space.sites = vec![repeated; 1024];
    assert_eq!(
        resolve_space(&Validated::new(space).unwrap(), &fixture.inventory)
            .unwrap_err()
            .code,
        ErrorCode::LimitExceeded
    );
}

#[test]
fn search_space_wire_schema_and_finite_quality_constraints_are_strict() {
    let fixture = Fixture::new(false);
    let mut wire = serde_json::to_value(&fixture.space).unwrap();
    assert!(parse_search_space(&serde_json::to_vec(&wire).unwrap()).is_ok());
    wire["strategy"]["selection"] = json!("weighted_sum");
    assert!(parse_search_space(&serde_json::to_vec(&wire).unwrap()).is_err());
    let mut space = fixture.space.clone();
    space.constraints.quality_limits[0].max_regression = f64::NAN;
    assert!(Validated::new(space).is_err());
    let mut space = fixture.space;
    space.sites[0].allowed_axes = [31].into();
    let validated = Validated::new(space).unwrap();
    assert!(resolve_space(&validated, &fixture.inventory).is_err());
}
