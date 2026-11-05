//! Hand-constructed synthetic ReLU checkpoint and versioned grouped task inventory.
//! No optimizer, training data, pretrained target or research success is implied.
use nf_checkpoint::serialization::{serialize_tensors, SerializationLimits, TensorPayload};
use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use nf_core::error::{CoreError, ErrorCode, Result};
use nf_core::schemas::{
    Adaptation, DType, FactoryRef, MethodRecord, ModelManifest, ProductionHistory, ProposalOrigin,
    TargetUpdate, TensorFormat, TensorSpec, Validate, Validated,
};
use nf_core::study::SplitRole;
use nf_graph::inventory::tensor_set_digest;
use nf_surgeon::transaction::{arithmetic_graph, SourceState, StateLoadLimits};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

pub const HEADS: [&str; 5] = [
    "sum",
    "difference",
    "maximum",
    "minimum",
    "absolute_difference",
];
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    CorruptSum,
    Reference,
    EmptySum,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    pub item: ContentDigest,
    pub group: ContentDigest,
    pub family: ContentDigest,
    pub split: SplitRole,
    pub x1: i32,
    pub x2: i32,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Inventory {
    pub schema_version: u32,
    pub protocol: String,
    pub domain_min: i32,
    pub domain_max: i32,
    pub tolerance: f64,
    pub cases: Vec<Case>,
}
fn canonical(value: &impl Serialize) -> Result<Vec<u8>> {
    Ok(CanonicalDocument::parse(
        &serde_json::to_vec(value).map_err(|_| {
            CoreError::new(ErrorCode::InvalidJson, "/arithmetic", "encoding failed")
        })?,
        Limits::default(),
    )?
    .bytes()
    .to_vec())
}
pub fn inventory() -> Result<Inventory> {
    let mut cases = Vec::new();
    let family = ContentDigest::of_bytes(b"nf.synthetic.arithmetic-pairs.family.v1");
    for x1 in -8..=8 {
        for x2 in -8..=8 {
            // Mirrored pairs share a group across ALL heads; symmetry cannot place
            // a twin in development and its counterpart in final verification.
            let group = ContentDigest::of_bytes(&canonical(&(
                "nf.arithmetic.unordered-pair-group.v1",
                x1.min(x2),
                x1.max(x2),
            ))?);
            let partition =
                ContentDigest::of_bytes(&canonical(&("nf.arithmetic.three-way.v1", &group))?);
            let bucket = u64::from_str_radix(&partition.as_str()[7..23], 16).unwrap() % 3;
            let split = match bucket {
                0 => SplitRole::Development,
                1 => SplitRole::Selection,
                _ => SplitRole::Holdout,
            };
            let item =
                ContentDigest::of_bytes(&canonical(&("nf.arithmetic.ordered-pair.v1", x1, x2))?);
            cases.push(Case {
                item,
                group,
                family: family.clone(),
                split,
                x1,
                x2,
            });
        }
    }
    for role in [
        SplitRole::Development,
        SplitRole::Selection,
        SplitRole::Holdout,
    ] {
        if !cases.iter().any(|c| c.split == role) {
            return Err(CoreError::new(
                ErrorCode::IntegrityMismatch,
                "/arithmetic",
                "nonempty frozen splits required",
            ));
        }
    }
    Ok(Inventory {
        schema_version: 1,
        protocol: "nf.arithmetic.three-way.unordered-groups.v1".into(),
        domain_min: -8,
        domain_max: 8,
        tolerance: 1e-10,
        cases,
    })
}
/// Owned immutable model bytes and source; no DTO-only model witness.
pub struct Fixture {
    pub source: Arc<SourceState>,
    pub inventory: Inventory,
    pub description: serde_json::Value,
}
pub fn build(mode: Mode) -> Result<Fixture> {
    let hidden: [f64; 12] = [1., 0., -1., 0., 0., 1., 0., -1., 1., -1., -1., 1.];
    let mut heads: [f64; 30] = [
        1., -1., 1., -1., 0., 0., 1., -1., -1., 1., 0., 0., 0., 0., 1., -1., 1., 0., 1., -1., 0.,
        0., -1., 0., 0., 0., 0., 0., 1., 1.,
    ];
    match mode {
        Mode::CorruptSum => heads[0] = 0.5,
        Mode::Reference => (),
        Mode::EmptySum => heads[..6].fill(0.0),
    }
    let hb: Vec<u8> = hidden.iter().flat_map(|x| x.to_le_bytes()).collect();
    let ob: Vec<u8> = heads.iter().flat_map(|x| x.to_le_bytes()).collect();
    let tensors = BTreeMap::from([
        (
            "hidden.weight".into(),
            TensorPayload {
                dtype: DType::F64,
                shape: &[6, 2],
                bytes: &hb,
            },
        ),
        (
            "heads.weight".into(),
            TensorPayload {
                dtype: DType::F64,
                shape: &[5, 6],
                bytes: &ob,
            },
        ),
    ]);
    let checkpoint = serialize_tensors(&tensors, SerializationLimits::default())?;
    let config = canonical(
        &serde_json::json!({"schema_version":1,"input_dimension":2,"hidden_dimension":6,"heads":HEADS,"domain_min":-8,"domain_max":8,"integer_inputs":true}),
    )?;
    let graph = arithmetic_graph()?;
    let graph_id = ContentDigest::of_bytes(&graph);
    let method = MethodRecord {
        schema_version: 1,
        target_update: TargetUpdate::DirectTensorEdit,
        proposal_origin: ProposalOrigin::Scripted,
        adaptation: Adaptation::None,
        production_history: ProductionHistory::DirectOnly,
        inspection_uses_backprop: false,
        parameter_update_uses_backprop: false,
        gradient_derived_update_values: false,
        alignment_fitted: false,
        optimizer_steps: 0,
        changed_parameters: 42,
    };
    method.validate()?;
    let provenance = canonical(
        &serde_json::json!({"schema_version":1,"origin":"hand_constructed_synthetic_fixture","known_target":true,"mode":mode,"method":method,"evidence_scope":"software_fixture_only_not_pretrained_research","training_or_pretrained_source":null}),
    )?;
    let assets: BTreeMap<String, Arc<[u8]>> = BTreeMap::from([
        ("config".into(), Arc::from(config)),
        ("graph".into(), Arc::from(graph)),
        ("provenance".into(), Arc::from(provenance)),
    ]);
    let parameters = BTreeMap::from([
        (
            "hidden.weight".into(),
            TensorSpec {
                dtype: DType::F64,
                shape: vec![6, 2],
                digest: ContentDigest::of_bytes(&hb),
                storage_group: "hidden".into(),
            },
        ),
        (
            "heads.weight".into(),
            TensorSpec {
                dtype: DType::F64,
                shape: vec![5, 6],
                digest: ContentDigest::of_bytes(&ob),
                storage_group: "heads".into(),
            },
        ),
    ]);
    let mut refs: BTreeMap<_, _> = assets
        .iter()
        .map(|(name, data)| (name.clone(), ContentDigest::of_bytes(data)))
        .collect();
    refs.insert("checkpoint".into(), checkpoint.digest.clone());
    let manifest = ModelManifest {
        schema_version: 1,
        architecture: FactoryRef {
            id: "nf.arithmetic.relu".into(),
            version: "1".into(),
            config: ContentDigest::of_bytes(&assets["config"]),
        },
        parameter_digest: tensor_set_digest(&parameters)?,
        parameters,
        buffers: BTreeMap::new(),
        format: TensorFormat::DenseSafetensors,
        packing_policy: None,
        assets: refs,
        usage_rights: "Apache-2.0; synthetic hand-constructed fixture".into(),
    };
    let source = Arc::new(SourceState::load(
        Validated::new(manifest)?,
        Arc::from(checkpoint.bytes),
        assets,
        StateLoadLimits::default(),
    )?);
    let inventory = inventory()?;
    let oracle_code = ContentDigest::of_bytes(include_bytes!(
        "../../../python/neural_foundry/arithmetic.py"
    ));
    let inventory_id = ContentDigest::of_bytes(&canonical(&inventory)?);
    let mut gyms = Vec::new();
    for (index, name) in HEADS.iter().enumerate() {
        let mut descriptor = serde_json::json!({"schema_version":1,"name":format!("arithmetic.{name}.v1"),"head":name,"head_index":index,"origin":"foundry_synthetic_builtin","metric":"accuracy","units":"fraction","direction":"higher_is_better","absolute_tolerance":1e-10,"oracle_protocol":"python_direct_arithmetic_v1","oracle_code":oracle_code,"dataset":inventory_id,"fixture_only":true});
        let id = ContentDigest::of_bytes(&canonical(&descriptor)?);
        descriptor["id"] = serde_json::json!(id);
        gyms.push(descriptor);
    }
    let description = serde_json::json!({"schema_version":1,"mode":mode,"architecture":"nf.arithmetic.relu","architecture_version":"1","smoke_adapter":"nf.arithmetic_relu.v1","graph":graph_id,"config":source.manifest().architecture.config,"manifest":source.manifest_digest(),"checkpoint":ContentDigest::of_bytes(source.checkpoint_bytes()),"parameters":source.manifest().parameter_digest,"dtype":"f64","device":"cpu","parameter_count":42,"inventory":inventory,"inventory_id":ContentDigest::of_bytes(&canonical(&inventory)?),"gyms":gyms,"provenance":source.manifest().assets["provenance"],"promotion_status":"not_evaluated"});
    Ok(Fixture {
        source,
        inventory,
        description,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    #[test]
    fn arithmetic_gyms_frozen_groups_are_nonempty_unique_and_mirror_consistent() {
        let inventory = inventory().unwrap();
        assert_eq!(inventory.cases.len(), 289);
        let mut ids = BTreeSet::new();
        let mut groups = BTreeMap::new();
        for c in inventory.cases {
            assert!(ids.insert(c.item));
            if let Some(role) = groups.insert(c.group, c.split) {
                assert_eq!(role, c.split);
            }
        }
        assert_eq!(groups.len(), 153);
    }
}
