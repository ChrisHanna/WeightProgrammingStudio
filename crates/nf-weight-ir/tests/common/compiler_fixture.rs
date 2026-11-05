use crate::common::{digest, program};
use nf_core::canonical::ContentDigest;
use nf_core::schemas::{DType, FactoryRef, ModelManifest, TensorFormat, TensorSpec, Validated};
use nf_graph::inventory::{
    tensor_set_digest, verify_inventory, InventoryLimits, TensorObservation, TensorRole,
    VerifiedInventory,
};
use nf_weight_ir::compilation::*;
use nf_weight_ir::program::{parse_program, Program};
use nf_weight_ir::search_space::{
    parse_search_space, resolve_space, ResolvedSearchSpace, SearchSpace,
};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

pub fn tensor_file(dtype: &str, shape: &[u64], data: &[u8]) -> Vec<u8> {
    let mut header = json!({"value":{"dtype":dtype,"shape":shape,"data_offsets":[0,data.len()]}})
        .to_string()
        .into_bytes();
    header.resize(header.len().div_ceil(8) * 8, b' ');
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend_from_slice(data);
    bytes
}
pub fn floats(values: &[f64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

pub struct Fixture {
    pub program: Value,
    pub space: SearchSpace,
    pub policy: CompilePolicy,
    pub manifest: Validated<ModelManifest>,
    pub data: Vec<u8>,
    pub checkpoint: Vec<u8>,
    pub assets: BTreeMap<String, Arc<[u8]>>,
    pub operands: BTreeMap<String, Arc<[u8]>>,
    pub root_inventory: VerifiedInventory,
    pub tied: bool,
}
impl Fixture {
    pub fn new(tied: bool) -> Self {
        let data = floats(&[1.0, 2.0, 3.0, 4.0]);
        let mut header = json!({"w":{"dtype":"F64","shape":[2,2],"data_offsets":[0,32]}})
            .to_string()
            .into_bytes();
        header.resize(header.len().div_ceil(8) * 8, b' ');
        let mut checkpoint = (header.len() as u64).to_le_bytes().to_vec();
        checkpoint.extend(header);
        checkpoint.extend(&data);
        let config: Arc<[u8]> = Arc::from(b"{}".as_slice());
        let config_id = ContentDigest::of_bytes(&config);
        let checkpoint_id = ContentDigest::of_bytes(&checkpoint);
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
                config: config_id.clone(),
            },
            parameter_digest: tensor_set_digest(&parameters).unwrap(),
            parameters,
            buffers: BTreeMap::new(),
            format: TensorFormat::DenseSafetensors,
            packing_policy: None,
            assets: BTreeMap::from([
                ("config".to_owned(), config_id.clone()),
                ("checkpoint".to_owned(), checkpoint_id.clone()),
            ]),
            usage_rights: "synthetic".to_owned(),
        })
        .unwrap();
        let mut observed = vec![TensorObservation {
            name: "w",
            role: TensorRole::Parameter,
            shape: &[2, 2],
            dtype: DType::F64,
            bytes: &data,
        }];
        if tied {
            observed.push(TensorObservation {
                name: "alias",
                role: TensorRole::Parameter,
                shape: &[2, 2],
                dtype: DType::F64,
                bytes: &data,
            });
        }
        let inventory = verify_inventory(&manifest, &observed, InventoryLimits::default()).unwrap();
        let mut value = program();
        value["base_model"]["checkpoint"] = json!(checkpoint_id);
        value["base_model"]["manifest"] = json!(inventory.source_manifest_digest());
        value["base_model"]["parameter_digest"] = json!(inventory.parameter_digest());
        value["base_model"]["config"] = json!(config_id);
        value["preconditions"]=json!(inventory.entries().iter().map(|(name,entry)|json!({"tensor":name,"role":"parameter","shape":entry.shape,"dtype":entry.dtype,"before_digest":entry.content_digest,"storage_group":entry.storage_group})).collect::<Vec<_>>());
        let regions: Vec<_> = inventory
            .entries()
            .keys()
            .map(|name| json!({"tensor":name,"start":[0,0],"shape":[2,2]}))
            .collect();
        value["effects"]["reads"] = json!(regions);
        value["effects"]["writes"] = json!(regions);
        value["constraints"]["max_artifact_bytes"] = json!(4096);
        let domain = |coefficient: bool, rank: bool, sides: bool| {
            json!({"dtypes":["f64"],"coefficient":if coefficient {json!({"min":-1.0,"max":1.0})}else{Value::Null},
            "rank":if rank{json!({"min":1,"max":2})}else{Value::Null},"sides":if sides{json!(["left","right"])}else{json!([])},"max_operand_bytes":4096})
        };
        let space_value = json!({"schema_version":1,"base_model":value["base_model"],"contract":digest("contract"),"objective":digest("objective"),
            "transformations":{"replace_slice":domain(false,false,false),"add_delta":domain(false,false,false),"scale":domain(true,false,false),"mask":domain(false,false,false),
                "permute_axis":domain(false,false,false),"linear_map":domain(false,false,true),"subspace_projection":domain(true,true,true),"low_rank_update":domain(true,true,false),"construct_recipe":domain(false,false,false)},
            "sites":inventory.entries().keys().map(|name|json!({"selector":{"type":"tensor","name":name},"role":"parameter","regions":{"type":"whole_tensor"},"allowed_axes":[0,1]})).collect::<Vec<_>>(),
            "constraints":{"max_changed_parameters":4,"max_artifact_bytes":4096,"max_temporary_bytes":4096,"max_operations":16,"allow_buffer_writes":false,"require_direct_weight_edits":true,"require_fully_gradient_free":false,"require_preserve_shapes":true,
                "quality_limits":[{"binding":digest("binding"),"metric":"accuracy","units":"fraction","baseline":checkpoint_id,"max_regression":0.01,"split":"holdout","uncertainty_protocol":digest("uncertainty")}]},
            "strategy":{"proposal":"causal_hypothesis","search":"branch","selection":"pareto","engine":"bounded.branch","engine_version":"1","seed":17}});
        let space =
            (*parse_search_space(&serde_json::to_vec(&space_value).unwrap()).unwrap()).clone();
        value["objective"]["search_space"] = json!(space.digest().unwrap());
        let parsed: Program = serde_json::from_value(value.clone()).unwrap();
        let permissions=serde_json::from_value(json!({"schema_version":1,"tools":["inspect","direct_edit"],"external_destinations":{},"allow_gradient_inspection":false,
            "allow_parameter_training":false,"require_fully_gradient_free":false,"allow_generated_gym":false,"allow_downloads":false})).unwrap();
        let policy = CompilePolicy {
            schema_version: 1,
            source: parsed.base_model,
            contract: parsed.objective.contract,
            objective: parsed.objective.objective,
            experiment_protocol: parsed.objective.experiment_protocol,
            search_space: space.digest().unwrap(),
            verification: parsed.verification,
            allowed_extra_gyms: vec![],
            permissions,
            operand_access: [nf_core::schemas::AccessClass::Development].into(),
            backend: BackendProfile {
                id: NATIVE_CPU_BACKEND_ID.to_owned(),
                version: "1".to_owned(),
                operations: space.transformations.keys().copied().collect(),
                capabilities: BTreeSet::from(["dense.cpu".to_owned()]),
                dtypes: vec![DType::F32, DType::F64],
                accumulation_dtypes: vec![DType::F32, DType::F64],
                fixed_scratch_bytes: 64,
            },
            bounds: CompilerBounds {
                max_metadata_bytes: 1_048_576,
                max_source_checkpoint_bytes: 1_048_576,
                max_source_asset_bytes: 4096,
                max_source_tensors: 16,
                max_observed_source_bytes: 4096,
                max_temporary_bytes: 4096,
                max_output_tensor_bytes: 4096,
                max_validation_work: 100_000,
                max_kernel_work: 100_000,
                max_parameter_magnitude: 100.0,
                max_backend_abs_tolerance: 0.0,
                max_orthonormal_tolerance: 1e-6,
            },
        };
        let mut fixture = Self {
            program: value,
            space,
            policy,
            manifest,
            data,
            checkpoint,
            assets: BTreeMap::from([("config".to_owned(), config)]),
            operands: BTreeMap::new(),
            root_inventory: inventory,
            tied,
        };
        fixture.set_operand(
            "F64",
            DType::F64,
            &[2, 2],
            &floats(&[10.0, 20.0, 30.0, 40.0]),
        );
        fixture
    }
    pub fn set_operand(&mut self, wire_dtype: &str, dtype: DType, shape: &[u64], data: &[u8]) {
        let bytes = tensor_file(wire_dtype, shape, data);
        let descriptor = &mut self.program["artifacts"]["operand"];
        descriptor["digest"] = json!(ContentDigest::of_bytes(&bytes));
        descriptor["dtype"] = json!(dtype);
        descriptor["shape"] = json!(shape);
        descriptor["data_bytes"] = json!(data.len());
        descriptor["artifact_bytes"] = json!(bytes.len());
        self.operands.insert("operand".to_owned(), Arc::from(bytes));
    }
    pub fn parsed(&self) -> Validated<Program> {
        parse_program(&serde_json::to_vec(&self.program).unwrap()).unwrap()
    }
    pub fn resolved(&self) -> ResolvedSearchSpace {
        resolve_space(
            &Validated::new(self.space.clone()).unwrap(),
            &self.root_inventory,
        )
        .unwrap()
    }
    pub fn run(&self) -> nf_core::error::Result<CompiledPlan> {
        let mut observed = vec![TensorObservation {
            name: "w",
            role: TensorRole::Parameter,
            shape: &[2, 2],
            dtype: DType::F64,
            bytes: &self.data,
        }];
        if self.tied {
            observed.push(TensorObservation {
                name: "alias",
                role: TensorRole::Parameter,
                shape: &[2, 2],
                dtype: DType::F64,
                bytes: &self.data,
            });
        }
        let source = SourceSnapshot {
            manifest: &self.manifest,
            tensors: &observed,
            checkpoint: &self.checkpoint,
            assets: &self.assets,
        };
        compile(
            &self.parsed(),
            source,
            &self.resolved(),
            &self.operands,
            &Validated::new(self.policy.clone())?,
            &[],
        )
    }
    pub fn rebind_space(&mut self) {
        self.program["objective"]["search_space"] = json!(self.space.digest().unwrap());
        self.policy.search_space = self.space.digest().unwrap();
    }
}
