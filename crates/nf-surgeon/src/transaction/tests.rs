use super::*;
use crate::basic::Scalar;
use nf_core::schemas::{DType, FactoryRef, TensorSpec};
use nf_graph::inventory::{verify_inventory, InventoryLimits};
use nf_weight_ir::compilation::{compile, CompilePolicy, CompiledPlan, SourceSnapshot};
use nf_weight_ir::program::parse_program;
use nf_weight_ir::search_space::{parse_search_space, resolve_space};
use serde_json::{json, Value};

fn id(value: &str) -> ContentDigest {
    ContentDigest::of_bytes(value.as_bytes())
}
fn floats(values: &[f64]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn file(tensors: BTreeMap<&str, (&str, Vec<u64>, Vec<u8>)>) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, (dtype, shape, bytes)) in tensors {
        let start = data.len();
        data.extend(bytes);
        header.insert(
            name.to_owned(),
            json!({"dtype":dtype,"shape":shape,"data_offsets":[start,data.len()]}),
        );
    }
    let mut raw = serde_json::to_vec(&header).unwrap();
    raw.resize(raw.len().div_ceil(8) * 8, b' ');
    let mut out = (raw.len() as u64).to_le_bytes().to_vec();
    out.extend(raw);
    out.extend(data);
    out
}
fn target() -> Value {
    json!({"tensor":"w","start":[0,0],"shape":[2,2]})
}

struct Fixture {
    source: SourceState,
    smoke: Validated<SmokeSuite>,
    program: Value,
    operands: BTreeMap<String, Arc<[u8]>>,
}
impl Fixture {
    fn new(tied: bool) -> Self {
        let values = floats(&[1.0, 2.0, 3.0, 4.0]);
        let cache = 7i64.to_le_bytes().to_vec();
        let checkpoint: Arc<[u8]> = Arc::from(file(BTreeMap::from([
            ("cache", ("I64", vec![1], cache.clone())),
            ("w", ("F64", vec![2, 2], values.clone())),
        ])));
        let config: Arc<[u8]> = Arc::from(b"{}".as_slice());
        let config_id = ContentDigest::of_bytes(&config);
        let spec = TensorSpec {
            shape: vec![2, 2],
            dtype: DType::F64,
            digest: ContentDigest::of_bytes(&values),
            storage_group: "weights".to_owned(),
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
                config: config_id.clone(),
            },
            parameter_digest: tensor_set_digest(&parameters).unwrap(),
            parameters,
            buffers: BTreeMap::from([(
                "cache".to_owned(),
                TensorSpec {
                    shape: vec![1],
                    dtype: DType::I64,
                    digest: ContentDigest::of_bytes(&cache),
                    storage_group: "cache".to_owned(),
                },
            )]),
            format: TensorFormat::DenseSafetensors,
            packing_policy: None,
            assets: BTreeMap::from([
                ("config".to_owned(), config_id),
                (
                    "checkpoint".to_owned(),
                    ContentDigest::of_bytes(&checkpoint),
                ),
            ]),
            usage_rights: "synthetic local fixture".to_owned(),
        })
        .unwrap();
        let source = SourceState::load(
            manifest,
            checkpoint,
            BTreeMap::from([("config".to_owned(), config)]),
            StateLoadLimits::default(),
        )
        .unwrap();
        let smoke = Validated::new(SmokeSuite {
            schema_version: 1,
            adapter: "nf.linear_fixture.v1".to_owned(),
            probes: vec![SmokeProbe {
                input: vec![1.0, 2.0],
                expected: None,
                absolute_tolerance: 0.0,
                max_abs_output: 1000.0,
            }],
        })
        .unwrap();
        let operand: Arc<[u8]> = Arc::from(file(BTreeMap::from([(
            "value",
            ("F64", vec![2, 2], floats(&[10.0, 20.0, 30.0, 40.0])),
        )])));
        let regions: Vec<_> = source
            .manifest
            .parameters
            .keys()
            .map(|name| json!({"tensor":name,"start":[0,0],"shape":[2,2]}))
            .collect();
        let preconditions:Vec<_>=source.manifest.parameters.iter().chain(&source.manifest.buffers).map(|(name,spec)|json!({"tensor":name,"role":if source.manifest.parameters.contains_key(name){"parameter"}else{"buffer"},"shape":spec.shape,"dtype":spec.dtype,"before_digest":spec.digest,"storage_group":spec.storage_group})).collect();
        let program = json!({"ir_version":1,"opset_version":1,
            "base_model":{"checkpoint":source.checkpoint_digest,"manifest":source.manifest_digest,"parameter_digest":source.manifest.parameter_digest,"architecture_graph":id("graph"),"config":source.manifest.architecture.config,"tokenizer":null},
            "objective":{"contract":id("contract"),"objective":id("objective"),"experiment_protocol":id("experiment"),"search_space":id("unbound-space")},
            "operations":[{"id":"op0","depends_on":[],"operation":{"type":"replace_slice","target":target(),"value":"operand"}}],
            "artifacts":{"operand":{"digest":ContentDigest::of_bytes(&operand),"tensor":"value","dtype":"f64","shape":[2,2],"data_bytes":32,"artifact_bytes":operand.len(),"access":"development",
                "provenance":{"schema_version":1,"origin":"supplied_fact","support":"untested","provenance":"fixture","evidence":[],"access":"development"}}},
            "preconditions":preconditions,"required_capabilities":["dense.cpu"],
            "constraints":{"gradient_free":true,"preserve_tensor_shapes":true,"max_changed_parameters":4,"max_artifact_bytes":4096,"max_temporary_bytes":8192,
                "numerical":{"accumulation_dtype":"f64","output_rounding":"nearest_ties_even","non_finite":"reject","max_abs_parameter":100.0,"backend_abs_tolerance":0.0}},
            "effects":{"reads":regions,"writes":regions,"buffer_writes":[],"graph_changes":false,"config_changes":[],"tokenizer_changes":false},
            "preservation":{"type":"experimental"},
            "verification":{"portfolio":id("portfolio"),"protocol":id("verification"),"gyms":[{"binding":id("binding"),"gym":id("gym"),"validation":id("validation"),"scorer":id("scorer"),"split":id("selection"),"role":"target"}],"smoke_suite":smoke.digest().unwrap(),"parity_suite":id("parity")},
            "lineage":{"parent_candidates":[],"hypothesis":id("hypothesis"),"method":{"schema_version":1,"target_update":"direct_tensor_edit","proposal_origin":"scripted","adaptation":"none","production_history":"direct_only","inspection_uses_backprop":false,"parameter_update_uses_backprop":false,"gradient_derived_update_values":false,"alignment_fitted":false,"optimizer_steps":0,"changed_parameters":4}},"extensions":{}});
        Self {
            source,
            smoke,
            program,
            operands: BTreeMap::from([("operand".to_owned(), operand)]),
        }
    }
    fn plan(&self) -> CompiledPlan {
        let observations = self.source.observations();
        let inventory = verify_inventory(
            self.source.manifest(),
            &observations,
            InventoryLimits::default(),
        )
        .unwrap();
        let mut program = self.program.clone();
        program["verification"]["smoke_suite"] = json!(self.smoke.digest().unwrap());
        program["base_model"]["checkpoint"] = json!(self.source.checkpoint_digest);
        program["base_model"]["manifest"] = json!(self.source.manifest_digest);
        program["base_model"]["parameter_digest"] = json!(self.source.manifest.parameter_digest);
        program["base_model"]["config"] = json!(self.source.manifest.architecture.config);
        program["preconditions"]=json!(self.source.manifest.parameters.iter().chain(&self.source.manifest.buffers).map(|(name,spec)|json!({"tensor":name,"role":if self.source.manifest.parameters.contains_key(name){"parameter"}else{"buffer"},"shape":spec.shape,"dtype":spec.dtype,"before_digest":spec.digest,"storage_group":spec.storage_group})).collect::<Vec<_>>());
        let dtype = self.source.manifest.parameters["w"].dtype;
        let mut domains = json!({"replace_slice":{"dtypes":["f64"],"coefficient":null,"rank":null,"sides":[],"max_operand_bytes":4096},
            "scale":{"dtypes":["f64"],"coefficient":{"min":0.0,"max":2.0},"rank":null,"sides":[],"max_operand_bytes":0},
            "add_delta":{"dtypes":["f64"],"coefficient":null,"rank":null,"sides":[],"max_operand_bytes":4096},
            "permute_axis":{"dtypes":["f64"],"coefficient":null,"rank":null,"sides":[],"max_operand_bytes":0}});
        for domain in domains.as_object_mut().unwrap().values_mut() {
            domain["dtypes"] = json!([dtype]);
        }
        let space = json!({"schema_version":1,"base_model":program["base_model"],"contract":id("contract"),"objective":id("objective"),"transformations":domains,
            "sites":self.source.manifest.parameters.keys().map(|name|json!({"selector":{"type":"tensor","name":name},"role":"parameter","regions":{"type":"whole_tensor"},"allowed_axes":[0,1]})).collect::<Vec<_>>(),
            "constraints":{"max_changed_parameters":4,"max_artifact_bytes":4096,"max_temporary_bytes":8192,"max_operations":16,"allow_buffer_writes":false,"require_direct_weight_edits":true,"require_fully_gradient_free":false,"require_preserve_shapes":true,
                "quality_limits":[{"binding":id("binding"),"metric":"accuracy","units":"fraction","baseline":self.source.checkpoint_digest,"max_regression":0.0,"split":"holdout","uncertainty_protocol":id("uncertainty")}]},
            "strategy":{"proposal":"scripted","search":"sequential","selection":"pareto","engine":"native.test","engine_version":"1","seed":17}});
        let space = parse_search_space(&serde_json::to_vec(&space).unwrap()).unwrap();
        program["objective"]["search_space"] = json!(space.digest().unwrap());
        let parsed = parse_program(&serde_json::to_vec(&program).unwrap()).unwrap();
        let policy = json!({"schema_version":1,"source":program["base_model"],"contract":id("contract"),"objective":id("objective"),"experiment_protocol":id("experiment"),"search_space":space.digest().unwrap(),
            "verification":program["verification"],"allowed_extra_gyms":[],"permissions":{"schema_version":1,"tools":["inspect","direct_edit"],"external_destinations":{},"allow_gradient_inspection":false,"allow_parameter_training":false,"require_fully_gradient_free":false,"allow_generated_gym":false,"allow_downloads":false},
            "operand_access":["development"],"backend":{"id":"nf.native_cpu","version":"1","operations":["replace_slice","add_delta","scale","permute_axis"],"capabilities":["dense.cpu"],"dtypes":["f32","f64"],"accumulation_dtypes":["f64"],"fixed_scratch_bytes":64},
            "bounds":{"max_metadata_bytes":1048576,"max_source_checkpoint_bytes":1048576,"max_source_asset_bytes":4096,"max_source_tensors":16,"max_observed_source_bytes":4096,"max_temporary_bytes":8192,"max_output_tensor_bytes":4096,"max_validation_work":100000,"max_kernel_work":100000,"max_parameter_magnitude":100.0,"max_backend_abs_tolerance":0.0,"max_orthonormal_tolerance":0.000001}});
        let policy =
            Validated::<CompilePolicy>::parse(&serde_json::to_vec(&policy).unwrap()).unwrap();
        compile(
            &parsed,
            SourceSnapshot {
                manifest: self.source.manifest(),
                tensors: &observations,
                checkpoint: self.source.checkpoint_bytes(),
                assets: self.source.assets(),
            },
            &resolve_space(&space, &inventory).unwrap(),
            &self.operands,
            &policy,
            &[],
        )
        .unwrap()
    }
    fn two_steps(&mut self) {
        let second = json!({"id":"op1","depends_on":["op0"],"operation":{"type":"scale","target":target(),"factor":0.5}});
        self.program["operations"]
            .as_array_mut()
            .unwrap()
            .push(second);
    }
}

pub(super) fn inject(candidate: &mut ModelState, index: usize, fault: Fault) -> Result<()> {
    match fault {
        Fault::Abort(step) if step == index => {
            return Err(failure(
                ErrorCode::TransactionAborted,
                "test fault after instruction",
            ))
        }
        Fault::Outside(step) if step == index => {
            let id = candidate.names["w"].clone();
            if let Group::Float(tensor) = Arc::make_mut(candidate.groups.get_mut(&id).unwrap()) {
                tensor.set(3, Scalar::F64(99.0));
            }
        }
        Fault::Buffer(step) if step == index => {
            if let Group::Opaque { bytes, .. } =
                Arc::make_mut(candidate.groups.get_mut("cache").unwrap())
            {
                Arc::make_mut(bytes)[0] ^= 1;
            }
        }
        Fault::Alias(step) if step == index => {
            candidate
                .names
                .insert("alias".to_owned(), "cache".to_owned());
        }
        Fault::Metadata(step) if step == index => {
            candidate.manifest.architecture.config = id("changed-config");
        }
        Fault::Shape(step) if step == index => {
            let id = candidate.names["w"].clone();
            let name = candidate.tensor("w").unwrap().name().to_owned();
            *Arc::make_mut(candidate.groups.get_mut(&id).unwrap()) = Group::Float(
                CpuTensor::from_f64(name, vec![4], vec![1.0, 2.0, 3.0, 4.0])?,
            );
        }
        Fault::DType(step) if step == index => {
            let id = candidate.names["w"].clone();
            let name = candidate.tensor("w").unwrap().name().to_owned();
            *Arc::make_mut(candidate.groups.get_mut(&id).unwrap()) = Group::Float(
                CpuTensor::from_f32(name, vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])?,
            );
        }
        _ => (),
    }
    Ok(())
}

#[test]
fn transaction_executes_real_plan_with_shared_aliases_and_complete_receipt() {
    let fixture = Fixture::new(true);
    let plan = fixture.plan();
    let source_before = fixture.source.tensor_bytes("w").unwrap();
    let result = execute(
        &plan,
        &fixture.source,
        &fixture.smoke,
        ExecutionLimits::default(),
    )
    .unwrap();
    assert_eq!(
        result.tensor("w").unwrap().f64_values().unwrap(),
        [10.0, 20.0, 30.0, 40.0]
    );
    assert!(std::ptr::eq(
        result.tensor("w").unwrap(),
        result.tensor("alias").unwrap()
    ));
    assert_eq!(result.receipt().changed_parameters, 4);
    assert_eq!(result.receipt().copied_group_bytes, 32);
    assert_eq!(
        result.receipt().before_buffers,
        result.receipt().after_buffers
    );
    assert_eq!(result.receipt().promotion_status, "not_evaluated");
    assert_eq!(result.receipt().steps.len(), 1);
    assert_eq!(fixture.source.tensor_bytes("w").unwrap(), source_before);
    assert!(!result.manifest().assets.contains_key("checkpoint"));
    assert_eq!(result.assets()["config"].as_ref(), b"{}");
    let replay = execute(
        &plan,
        &fixture.source,
        &fixture.smoke,
        ExecutionLimits::default(),
    )
    .unwrap();
    assert_eq!(result.digest(), replay.digest());
    assert_eq!(
        result.tensor_bytes("cache").unwrap(),
        fixture.source.tensor_bytes("cache").unwrap()
    );
}

#[test]
fn transaction_multi_step_copy_on_write_is_once_per_physical_group() {
    let mut fixture = Fixture::new(false);
    fixture.two_steps();
    let result = execute(
        &fixture.plan(),
        &fixture.source,
        &fixture.smoke,
        ExecutionLimits::default(),
    )
    .unwrap();
    assert_eq!(
        result.tensor("w").unwrap().f64_values().unwrap(),
        [5.0, 10.0, 15.0, 20.0]
    );
    assert_eq!(result.receipt().copied_group_bytes, 32);
    assert_eq!(result.receipt().steps.len(), 2);
}

#[test]
fn transaction_abort_after_each_instruction_never_changes_source() {
    let mut fixture = Fixture::new(false);
    fixture.two_steps();
    let plan = fixture.plan();
    let before = fixture.source.tensor_bytes("w").unwrap();
    for step in 0..2 {
        assert_eq!(
            execute_inner(
                &plan,
                &fixture.source,
                &fixture.smoke,
                ExecutionLimits::default(),
                Fault::Abort(step)
            )
            .unwrap_err()
            .code,
            ErrorCode::TransactionAborted
        );
        assert_eq!(fixture.source.tensor_bytes("w").unwrap(), before);
    }
}

#[test]
fn transaction_outside_mask_buffer_alias_and_metadata_faults_are_rejected() {
    let mut fixture = Fixture::new(true);
    fixture.program["operations"][0]["operation"] =
        json!({"type":"scale","target":{"tensor":"w","start":[0,0],"shape":[1,1]},"factor":0.5});
    fixture.program["effects"]["writes"]
        .as_array_mut()
        .unwrap()
        .push(json!({"tensor":"w","start":[0,0],"shape":[1,1]}));
    fixture.program["effects"]["reads"]
        .as_array_mut()
        .unwrap()
        .push(json!({"tensor":"w","start":[0,0],"shape":[1,1]}));
    let plan = fixture.plan();
    let before = fixture.source.tensor_bytes("w").unwrap();
    let cache = fixture.source.tensor_bytes("cache").unwrap();
    for fault in [
        Fault::Outside(0),
        Fault::Buffer(0),
        Fault::Alias(0),
        Fault::Metadata(0),
        Fault::Shape(0),
        Fault::DType(0),
    ] {
        assert_eq!(
            execute_inner(
                &plan,
                &fixture.source,
                &fixture.smoke,
                ExecutionLimits::default(),
                fault
            )
            .unwrap_err()
            .code,
            ErrorCode::ScopeViolation
        );
        assert_eq!(fixture.source.tensor_bytes("w").unwrap(), before);
        assert_eq!(fixture.source.tensor_bytes("cache").unwrap(), cache);
    }
}

#[test]
fn transaction_late_numeric_failure_drops_an_earlier_successful_step() {
    let mut fixture = Fixture::new(false);
    fixture.program["operations"].as_array_mut().unwrap().push(json!({"id":"op1","depends_on":["op0"],"operation":{"type":"scale","target":target(),"factor":2.0}}));
    fixture.program["operations"].as_array_mut().unwrap().push(json!({"id":"op2","depends_on":["op1"],"operation":{"type":"scale","target":target(),"factor":2.0}}));
    let before = fixture.source.tensor_bytes("w").unwrap();
    assert_eq!(
        execute(
            &fixture.plan(),
            &fixture.source,
            &fixture.smoke,
            ExecutionLimits::default()
        )
        .unwrap_err()
        .code,
        ErrorCode::InvalidValue
    );
    assert_eq!(fixture.source.tensor_bytes("w").unwrap(), before);
}

#[test]
fn transaction_smoke_binding_forward_failure_and_work_limits_prevent_publication() {
    let mut fixture = Fixture::new(false);
    let old_plan = fixture.plan();
    fixture.smoke = Validated::new(SmokeSuite {
        probes: vec![SmokeProbe {
            input: vec![1.0, 2.0],
            expected: Some(vec![5.0, 11.0]),
            absolute_tolerance: 0.0,
            max_abs_output: 1000.0,
        }],
        ..(*fixture.smoke).clone()
    })
    .unwrap();
    assert_eq!(
        execute(
            &old_plan,
            &fixture.source,
            &fixture.smoke,
            ExecutionLimits::default()
        )
        .unwrap_err()
        .code,
        ErrorCode::VerificationPolicyWeakened
    );
    assert_eq!(
        execute(
            &fixture.plan(),
            &fixture.source,
            &fixture.smoke,
            ExecutionLimits::default()
        )
        .unwrap_err()
        .code,
        ErrorCode::SmokeFailed
    );
    assert_eq!(
        execute(
            &fixture.plan(),
            &fixture.source,
            &fixture.smoke,
            ExecutionLimits {
                max_smoke_work: 1,
                ..ExecutionLimits::default()
            }
        )
        .unwrap_err()
        .code,
        ErrorCode::LimitExceeded
    );
    assert_eq!(
        fixture.source.tensor("w").unwrap().f64_values().unwrap(),
        [1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn transaction_fresh_source_bytes_and_runtime_verification_budget_are_checked() {
    let mut fixture = Fixture::new(false);
    let plan = fixture.plan();
    assert_eq!(
        execute(
            &plan,
            &fixture.source,
            &fixture.smoke,
            ExecutionLimits {
                max_verification_bytes: 1,
                ..ExecutionLimits::default()
            }
        )
        .unwrap_err()
        .code,
        ErrorCode::LimitExceeded
    );
    if let Group::Float(tensor) =
        Arc::make_mut(fixture.source.state.groups.get_mut("weights").unwrap())
    {
        tensor.set(0, Scalar::F64(9.0));
    }
    assert_eq!(
        execute(
            &plan,
            &fixture.source,
            &fixture.smoke,
            ExecutionLimits::default()
        )
        .unwrap_err()
        .code,
        ErrorCode::SourceMismatch
    );
}

#[test]
fn transaction_representation_sequence_commits_atomically_and_can_restore_identity() {
    let mut fixture = Fixture::new(false);
    fixture.program["operations"] = json!([
        {"id":"p0","depends_on":[],"operation":{"type":"permute_axis","target":target(),"axis":0,"permutation":[1,0]}},
        {"id":"p1","depends_on":["p0"],"operation":{"type":"permute_axis","target":target(),"axis":1,"permutation":[1,0]}},
        {"id":"p2","depends_on":["p1"],"operation":{"type":"permute_axis","target":target(),"axis":0,"permutation":[1,0]}},
        {"id":"p3","depends_on":["p2"],"operation":{"type":"permute_axis","target":target(),"axis":1,"permutation":[1,0]}}
    ]);
    let plan = fixture.plan();
    let before = fixture.source.tensor_bytes("w").unwrap();
    let result = execute(
        &plan,
        &fixture.source,
        &fixture.smoke,
        ExecutionLimits::default(),
    )
    .unwrap();
    assert_eq!(result.tensor_bytes("w").unwrap(), before);
    assert_eq!(result.receipt().changed_parameters, 0);
    assert_eq!(result.receipt().copied_group_bytes, 32);
    assert_eq!(result.receipt().steps.len(), 4);
    assert_eq!(
        result.receipt().method.target_update,
        TargetUpdate::NoChange
    );
    for index in 0..4 {
        assert!(execute_inner(
            &plan,
            &fixture.source,
            &fixture.smoke,
            ExecutionLimits::default(),
            Fault::Abort(index)
        )
        .is_err());
        assert_eq!(fixture.source.tensor_bytes("w").unwrap(), before);
    }
}

#[test]
fn transaction_f32_checkpoint_observations_and_operand_execute_without_implicit_casts() {
    let mut fixture = Fixture::new(false);
    let values: Vec<_> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let checkpoint: Arc<[u8]> = Arc::from(file(BTreeMap::from([
        ("cache", ("I64", vec![1], 7i64.to_le_bytes().to_vec())),
        ("w", ("F32", vec![2, 2], values.clone())),
    ])));
    let mut manifest = (*fixture.source.manifest).clone();
    let spec = manifest.parameters.get_mut("w").unwrap();
    spec.dtype = DType::F32;
    spec.digest = ContentDigest::of_bytes(&values);
    manifest.parameter_digest = tensor_set_digest(&manifest.parameters).unwrap();
    manifest.assets.insert(
        "checkpoint".to_owned(),
        ContentDigest::of_bytes(&checkpoint),
    );
    fixture.source = SourceState::load(
        Validated::new(manifest).unwrap(),
        checkpoint,
        fixture.source.state.assets.clone(),
        StateLoadLimits::default(),
    )
    .unwrap();
    let values: Vec<_> = [10.0f32, 20.0, 30.0, 40.0]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let operand: Arc<[u8]> = Arc::from(file(BTreeMap::from([(
        "value",
        ("F32", vec![2, 2], values),
    )])));
    let spec = &mut fixture.program["artifacts"]["operand"];
    spec["dtype"] = json!("f32");
    spec["data_bytes"] = json!(16);
    spec["artifact_bytes"] = json!(operand.len());
    spec["digest"] = json!(ContentDigest::of_bytes(&operand));
    fixture.operands.insert("operand".to_owned(), operand);
    let result = execute(
        &fixture.plan(),
        &fixture.source,
        &fixture.smoke,
        ExecutionLimits::default(),
    )
    .unwrap();
    assert_eq!(result.tensor("w").unwrap().dtype(), DType::F32);
    assert_eq!(
        result.tensor("w").unwrap().f32_values().unwrap(),
        [10.0, 20.0, 30.0, 40.0]
    );
    assert_eq!(result.receipt().copied_group_bytes, 16);
}

#[test]
fn transaction_missing_smoke_adapter_and_source_load_limits_fail_closed() {
    let mut fixture = Fixture::new(false);
    assert_eq!(
        SourceState::load(
            fixture.source.manifest.clone(),
            fixture.source.checkpoint.clone(),
            fixture.source.state.assets.clone(),
            StateLoadLimits {
                max_tensor_bytes: 1,
                ..StateLoadLimits::default()
            }
        )
        .unwrap_err()
        .code,
        ErrorCode::LimitExceeded
    );
    let mut manifest = (*fixture.source.manifest).clone();
    manifest.architecture.id = "unregistered.model".to_owned();
    fixture.source = SourceState::load(
        Validated::new(manifest).unwrap(),
        fixture.source.checkpoint.clone(),
        fixture.source.state.assets.clone(),
        StateLoadLimits::default(),
    )
    .unwrap();
    assert_eq!(
        execute(
            &fixture.plan(),
            &fixture.source,
            &fixture.smoke,
            ExecutionLimits::default()
        )
        .unwrap_err()
        .code,
        ErrorCode::UnsupportedCapability
    );
    assert_eq!(
        fixture.source.tensor("w").unwrap().f64_values().unwrap(),
        [1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn transaction_noop_preserves_parameter_identity_and_returns_no_acceptance() {
    let mut fixture = Fixture::new(false);
    fixture.program["operations"] = json!([]);
    fixture.program["artifacts"] = json!({});
    fixture.operands.clear();
    fixture.program["effects"]["reads"] = json!([]);
    fixture.program["effects"]["writes"] = json!([]);
    fixture.program["lineage"]["method"]["target_update"] = json!("no_change");
    fixture.program["lineage"]["method"]["changed_parameters"] = json!(0);
    fixture.program["constraints"]["max_changed_parameters"] = json!(0);
    let result = execute(
        &fixture.plan(),
        &fixture.source,
        &fixture.smoke,
        ExecutionLimits::default(),
    )
    .unwrap();
    assert_eq!(result.receipt().changed_parameters, 0);
    assert_eq!(result.receipt().copied_group_bytes, 0);
    assert_eq!(
        result.receipt().before_parameters,
        result.receipt().after_parameters
    );
    assert_eq!(
        result.receipt().method.target_update,
        TargetUpdate::NoChange
    );
    assert_eq!(result.receipt().promotion_status, "not_evaluated");
}

#[test]
fn export_restore_core_candidate_keeps_ties_scope_buffers_and_actual_smoke() {
    use crate::export::{export_candidate, restore_source, CoreExportLimits};
    use nf_checkpoint::store::{ArtifactStore, StoreLimits};
    let fixture = Fixture::new(true);
    let candidate = execute(
        &fixture.plan(),
        &fixture.source,
        &fixture.smoke,
        ExecutionLimits::default(),
    )
    .unwrap();
    let path = std::env::var_os("KIROCREW_SCRATCH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join(format!("core-export-{}", std::process::id()));
    std::fs::create_dir(&path).unwrap();
    let store = ArtifactStore::create(
        &path.join("store"),
        StoreLimits {
            total_bytes: 1_000_000,
            max_blob_bytes: 500_000,
            max_entries: 100,
        },
    )
    .unwrap();
    let receipt = export_candidate(
        &store,
        &candidate,
        &fixture.smoke,
        CoreExportLimits::default(),
    )
    .unwrap();
    assert_eq!(receipt.promotion_status, "not_evaluated");
    assert_eq!(
        receipt.transaction.after_parameters,
        candidate.manifest().parameter_digest
    );
    assert_eq!(receipt.smoke_outputs, candidate.receipt().smoke_outputs);
    assert_eq!(receipt.transaction.effective_writes.len(), 2);
    let coordinates = &receipt.transaction.actual_changed_coordinates["weights"];
    assert_eq!(coordinates.len(), 1);
    assert_eq!(
        (coordinates[0].start_flat_index, coordinates[0].elements),
        (0, 4)
    );
    let restored =
        restore_source(&store, &receipt.export_root, CoreExportLimits::default()).unwrap();
    assert_eq!(
        restored.tensor_bytes("w").unwrap(),
        candidate.tensor_bytes("w").unwrap()
    );
    assert_eq!(
        restored.tensor_bytes("cache").unwrap(),
        fixture.source.tensor_bytes("cache").unwrap()
    );
    assert!(std::ptr::eq(
        restored.tensor("w").unwrap(),
        restored.tensor("alias").unwrap()
    ));
    assert_eq!(
        fixture.source.tensor("w").unwrap().f64_values().unwrap(),
        [1.0, 2.0, 3.0, 4.0]
    );
    let mut wrong = (*fixture.smoke).clone();
    wrong.probes[0].input[0] += 1.0;
    assert_eq!(
        export_candidate(
            &store,
            &candidate,
            &Validated::new(wrong).unwrap(),
            CoreExportLimits::default()
        )
        .unwrap_err()
        .code,
        ErrorCode::SourceMismatch
    );
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn ffi_cancellation_before_and_between_instructions_never_changes_source() {
    let mut fixture = Fixture::new(true);
    fixture.two_steps();
    let plan = fixture.plan();
    let token = CancellationToken::default();
    token.request();
    assert_eq!(
        execute_cancellable(
            &plan,
            &fixture.source,
            &fixture.smoke,
            ExecutionLimits::default(),
            &token
        )
        .unwrap_err()
        .code,
        ErrorCode::TransactionAborted
    );
    for index in 0..2 {
        let token = CancellationToken::default();
        assert_eq!(
            execute_verified_inner(
                &plan,
                &fixture.source,
                &fixture.smoke,
                ExecutionLimits::default(),
                None,
                Fault::Cancel(index),
                Some(&token)
            )
            .unwrap_err()
            .code,
            ErrorCode::TransactionAborted
        );
        assert!(token.is_cancelled());
        assert_eq!(
            fixture.source.tensor("w").unwrap().f64_values().unwrap(),
            [1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            fixture.source.tensor_bytes("cache").unwrap(),
            7i64.to_le_bytes()
        );
    }
}

#[test]
fn worker_broker_registered_forward_is_readonly_bounded_and_matches_smoke() {
    let fixture = Fixture::new(true);
    let original = fixture.source.tensor_bytes("w").unwrap();
    let output = fixture
        .source
        .infer_linear_fixture(&[1.0, 2.0], ExecutionLimits::default())
        .unwrap();
    assert_eq!(output, [5.0, 11.0]);
    assert_eq!(
        fixture
            .source
            .verify_smoke(&fixture.smoke, ExecutionLimits::default())
            .unwrap()
            .0,
        ContentDigest::of_bytes(&floats(&output))
    );
    assert!(fixture
        .source
        .infer_linear_fixture(&[1.0], ExecutionLimits::default())
        .is_err());
    assert!(fixture
        .source
        .infer_linear_fixture(
            &[1.0, 2.0],
            ExecutionLimits {
                max_smoke_work: 0,
                ..ExecutionLimits::default()
            }
        )
        .is_err());
    assert!(fixture
        .source
        .infer_linear_fixture(&[f64::MAX, f64::MAX], ExecutionLimits::default())
        .is_err());
    assert_eq!(fixture.source.tensor_bytes("w").unwrap(), original);
    assert_eq!(fixture.source.tensor_bytes("alias").unwrap(), original);
}

#[test]
fn export_restore_coordinate_receipt_records_actual_cells_not_whole_declared_masks() {
    let mut fixture = Fixture::new(false);
    let operand: Arc<[u8]> = Arc::from(file(BTreeMap::from([(
        "value",
        ("F64", vec![2, 2], floats(&[1.0, 20.0, 3.0, 40.0])),
    )])));
    fixture.program["artifacts"]["operand"]["digest"] = json!(ContentDigest::of_bytes(&operand));
    fixture.program["artifacts"]["operand"]["artifact_bytes"] = json!(operand.len());
    fixture.operands.insert("operand".to_owned(), operand);
    let result = execute(
        &fixture.plan(),
        &fixture.source,
        &fixture.smoke,
        ExecutionLimits::default(),
    )
    .unwrap();
    let spans = &result.receipt().actual_changed_coordinates["weights"];
    assert_eq!(
        spans
            .iter()
            .map(|span| (span.start_flat_index, span.elements))
            .collect::<Vec<_>>(),
        [(1, 1), (3, 1)]
    );
    assert_eq!(result.receipt().changed_parameters, 2);
    assert_eq!(result.receipt().effective_writes[0].shape, [2, 2]);
    assert!(fixture
        .source
        .verify_smoke(
            &fixture.smoke,
            ExecutionLimits {
                max_verification_bytes: 1,
                ..ExecutionLimits::default()
            }
        )
        .is_err());
}
