use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use nf_core::error::ErrorCode;
use nf_core::schemas::{DType, FactoryRef, ModelManifest, TensorFormat, TensorSpec, Validated};
use nf_graph::inventory::{tensor_set_digest, verify_inventory, InventoryLimits};
use nf_patches::replay::{replay, PatchGraph, PatchRecord, ReplayLimits};
use nf_surgeon::history::{certify_history, HistoryLimits, HistoryRequest, VerifiedHistory};
use nf_surgeon::transaction::{
    execute, execute_with_history, ExecutionLimits, SmokeProbe, SmokeSuite, SourceState,
    StateLoadLimits,
};
use nf_weight_ir::compilation::{compile, CompilePolicy, CompiledPlan, SourceSnapshot};
use nf_weight_ir::program::parse_program;
use nf_weight_ir::search_space::{parse_search_space, resolve_space, PriorEffects, SearchSpace};
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::Arc;

fn id(value: &str) -> ContentDigest {
    ContentDigest::of_bytes(value.as_bytes())
}
fn rehash(record: &mut PatchRecord) {
    let raw = serde_json::to_vec(&record.body).unwrap();
    record.id = CanonicalDocument::parse(&raw, Limits::default())
        .unwrap()
        .digest();
}
struct Fixture {
    root: SourceState,
    space: Validated<SearchSpace>,
    smoke: Validated<SmokeSuite>,
    gradient_inspection: bool,
}
impl Fixture {
    fn new(values: &[f64]) -> Self {
        let data: Vec<_> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let mut header = json!({"w":{"dtype":"F64","shape":[2,2],"data_offsets":[0,32]}})
            .to_string()
            .into_bytes();
        header.resize(header.len().div_ceil(8) * 8, b' ');
        let mut raw = (header.len() as u64).to_le_bytes().to_vec();
        raw.extend(header);
        raw.extend(&data);
        let config: Arc<[u8]> = Arc::from(b"{}".as_slice());
        let config_id = ContentDigest::of_bytes(&config);
        let parameters = BTreeMap::from([(
            "w".to_owned(),
            TensorSpec {
                shape: vec![2, 2],
                dtype: DType::F64,
                digest: ContentDigest::of_bytes(&data),
                storage_group: "w".to_owned(),
            },
        )]);
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
                ("config".to_owned(), config_id),
                ("checkpoint".to_owned(), ContentDigest::of_bytes(&raw)),
            ]),
            usage_rights: "synthetic".to_owned(),
        })
        .unwrap();
        let root = SourceState::load(
            manifest,
            Arc::from(raw),
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
        let space = json!({"schema_version":1,"base_model":root.binding(id("graph")),"contract":id("contract"),"objective":id("objective"),
            "transformations":{"scale":{"dtypes":["f64"],"coefficient":{"min":0.0,"max":2.0},"rank":null,"sides":[],"max_operand_bytes":0}},
            "sites":[{"selector":{"type":"tensor","name":"w"},"role":"parameter","regions":{"type":"whole_tensor"},"allowed_axes":[0,1]}],
            "constraints":{"max_changed_parameters":4,"max_artifact_bytes":0,"max_temporary_bytes":4096,"max_operations":8,"allow_buffer_writes":false,"require_direct_weight_edits":true,"require_fully_gradient_free":false,"require_preserve_shapes":true,
                "quality_limits":[{"binding":id("binding"),"metric":"accuracy","units":"fraction","baseline":root.binding(id("graph")).checkpoint,"max_regression":0.0,"split":"holdout","uncertainty_protocol":id("uncertainty")}]},
            "strategy":{"proposal":"scripted","search":"branch","selection":"pareto","engine":"native.replay","engine_version":"1","seed":17}});
        Self {
            root,
            space: parse_search_space(&serde_json::to_vec(&space).unwrap()).unwrap(),
            smoke,
            gradient_inspection: false,
        }
    }
    fn plan(
        &self,
        source: &SourceState,
        parent: Option<&VerifiedHistory>,
        factor: f64,
        partial: bool,
        forged: bool,
    ) -> (CompiledPlan, Validated<CompilePolicy>) {
        let target = if partial {
            json!({"tensor":"w","start":[0,0],"shape":[1,1]})
        } else {
            json!({"tensor":"w","start":[0,0],"shape":[2,2]})
        };
        let program = json!({"ir_version":1,"opset_version":1,"base_model":source.binding(id("graph")),
            "objective":{"contract":id("contract"),"objective":id("objective"),"experiment_protocol":id("experiment"),"search_space":self.space.digest().unwrap()},
            "operations":[{"id":"op0","depends_on":[],"operation":{"type":"scale","target":target,"factor":factor}}],"artifacts":{},
            "preconditions":[{"tensor":"w","role":"parameter","shape":[2,2],"dtype":"f64","before_digest":source.manifest().parameters["w"].digest,"storage_group":"w"}],"required_capabilities":["dense.cpu"],
            "constraints":{"gradient_free":true,"preserve_tensor_shapes":true,"max_changed_parameters":if partial{1}else{4},"max_artifact_bytes":0,"max_temporary_bytes":4096,
                "numerical":{"accumulation_dtype":"f64","output_rounding":"nearest_ties_even","non_finite":"reject","max_abs_parameter":100.0,"backend_abs_tolerance":0.0}},
            "effects":{"reads":[target],"writes":[target],"buffer_writes":[],"graph_changes":false,"config_changes":[],"tokenizer_changes":false},"preservation":{"type":"experimental"},
            "verification":{"portfolio":id("portfolio"),"protocol":id("verification"),"gyms":[{"binding":id("binding"),"gym":id("gym"),"validation":id("validation"),"scorer":id("scorer"),"split":id("selection"),"role":"target"}],"smoke_suite":self.smoke.digest().unwrap(),"parity_suite":id("parity")},
            "lineage":{"parent_candidates":parent.map(|parent|vec![parent.candidate_id().clone()]).unwrap_or_default(),"hypothesis":id("hypothesis"),
                "method":{"schema_version":1,"target_update":"direct_tensor_edit","proposal_origin":"scripted","adaptation":"none","production_history":"direct_only","inspection_uses_backprop":self.gradient_inspection,"parameter_update_uses_backprop":false,"gradient_derived_update_values":false,"alignment_fitted":false,"optimizer_steps":0,"changed_parameters":if partial{1}else{4}}},"extensions":{}});
        let parsed = parse_program(&serde_json::to_vec(&program).unwrap()).unwrap();
        let policy = json!({"schema_version":1,"source":program["base_model"],"contract":id("contract"),"objective":id("objective"),"experiment_protocol":id("experiment"),"search_space":self.space.digest().unwrap(),
            "verification":program["verification"],"allowed_extra_gyms":[],"permissions":{"schema_version":1,"tools":if self.gradient_inspection{json!(["inspect","direct_edit","attribution"])}else{json!(["inspect","direct_edit"])},"external_destinations":{},"allow_gradient_inspection":self.gradient_inspection,"allow_parameter_training":false,"require_fully_gradient_free":false,"allow_generated_gym":false,"allow_downloads":false},
            "operand_access":[],"backend":{"id":"nf.native_cpu","version":"1","operations":["scale"],"capabilities":["dense.cpu"],"dtypes":["f64"],"accumulation_dtypes":["f64"],"fixed_scratch_bytes":64},
            "bounds":{"max_metadata_bytes":1048576,"max_source_checkpoint_bytes":1048576,"max_source_asset_bytes":4096,"max_source_tensors":16,"max_observed_source_bytes":4096,"max_temporary_bytes":4096,"max_output_tensor_bytes":4096,"max_validation_work":100000,"max_kernel_work":100000,"max_parameter_magnitude":100.0,"max_backend_abs_tolerance":0.0,"max_orthonormal_tolerance":0.000001}});
        let policy =
            Validated::<CompilePolicy>::parse(&serde_json::to_vec(&policy).unwrap()).unwrap();
        let root_observations = self.root.observations();
        let root_inventory = verify_inventory(
            self.root.manifest(),
            &root_observations,
            InventoryLimits::default(),
        )
        .unwrap();
        let resolved = resolve_space(&self.space, &root_inventory).unwrap();
        let observations = source.observations();
        let mut prior: Vec<PriorEffects> = parent
            .map(|parent| vec![parent.prior_effects()])
            .unwrap_or_default();
        if forged {
            prior[0].cumulative_writes.clear();
        }
        let plan = compile(
            &parsed,
            SourceSnapshot {
                manifest: source.manifest(),
                tensors: &observations,
                checkpoint: source.checkpoint_bytes(),
                assets: source.assets(),
            },
            &resolved,
            &BTreeMap::new(),
            &policy,
            &prior,
        )
        .unwrap();
        (plan, policy)
    }
    fn record(
        &self,
        parent: Option<(&PatchRecord, &VerifiedHistory)>,
        factor: f64,
    ) -> (PatchRecord, VerifiedHistory) {
        let source = parent.map_or(&self.root, |(_, history)| history.source_state());
        let parent_history = parent.map(|(_, history)| history);
        let (plan, policy) = self.plan(source, parent_history, factor, false, false);
        let candidate = if let Some(history) = parent_history {
            execute_with_history(
                &plan,
                source,
                &self.smoke,
                ExecutionLimits::default(),
                history,
            )
            .unwrap()
        } else {
            execute(&plan, source, &self.smoke, ExecutionLimits::default()).unwrap()
        };
        let root_observations = self.root.observations();
        let inventory = verify_inventory(
            self.root.manifest(),
            &root_observations,
            InventoryLimits::default(),
        )
        .unwrap();
        let resolved = resolve_space(&self.space, &inventory).unwrap();
        let history = certify_history(
            HistoryRequest {
                plan: &plan,
                source,
                smoke: &self.smoke,
                space: &resolved,
                parent: parent_history,
                expected_candidate: Some(candidate.digest()),
            },
            HistoryLimits::default(),
        )
        .unwrap();
        let record = PatchRecord::from_verified(
            parent
                .map(|(record, _)| vec![record.id.clone()])
                .unwrap_or_default(),
            &plan,
            &policy,
            &self.space,
            &self.smoke,
            &history,
        )
        .unwrap();
        (record, history)
    }
}

#[test]
fn replay_root_child_identity_and_immutable_checkpoint_restore() {
    let fixture = Fixture::new(&[1.0, 2.0, 3.0, 4.0]);
    let (first, parent) = fixture.record(None, 0.5);
    let (second, expected) = fixture.record(Some((&first, &parent)), 2.0);
    let records = vec![second.clone(), first.clone()];
    let graph = PatchGraph::parse(&serde_json::to_vec(&records).unwrap()).unwrap();
    let result = replay(
        &graph,
        &second.id,
        &fixture.root,
        &BTreeMap::new(),
        ReplayLimits::default(),
    )
    .unwrap();
    assert_eq!(result.history.candidate_id(), expected.candidate_id());
    assert_eq!(result.history.resulting_model(), expected.resulting_model());
    assert_eq!(result.nodes_replayed, 2);
    assert_eq!(
        result
            .history
            .source_state()
            .tensor("w")
            .unwrap()
            .f64_values()
            .unwrap(),
        [1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(result.history.costs().kernel_work_upper_bound, 16);
    let restored = graph
        .restore_root(
            &second.id,
            &fixture.root,
            &fixture.space.base_model.checkpoint,
        )
        .unwrap();
    assert_eq!(restored.checkpoint_bytes(), fixture.root.checkpoint_bytes());
    assert!(graph
        .restore_root(&second.id, &fixture.root, &id("wrong-checkpoint"))
        .is_err());
    assert_eq!(
        fixture.root.tensor("w").unwrap().f64_values().unwrap(),
        [1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn replay_missing_cycles_duplicate_and_tampered_records_fail_closed() {
    let fixture = Fixture::new(&[1.0, 2.0, 3.0, 4.0]);
    let (first, parent) = fixture.record(None, 0.5);
    let (second, _) = fixture.record(Some((&first, &parent)), 2.0);
    assert!(PatchGraph::from_records(vec![second.clone()]).is_err());
    assert!(PatchGraph::from_records(vec![first.clone(), first.clone()]).is_err());
    let mut cyclic = first.clone();
    cyclic.body.parents = vec![second.id.clone()];
    assert_eq!(
        PatchGraph::from_records(vec![cyclic, second])
            .unwrap_err()
            .code,
        ErrorCode::InvalidValue
    );
    let mut tampered = first;
    tampered.body.compiler_version = "unrecorded".to_owned();
    assert_eq!(
        PatchGraph::from_records(vec![tampered]).unwrap_err().code,
        ErrorCode::IntegrityMismatch
    );
}

#[test]
fn replay_content_addressed_false_result_cost_and_backend_claims_are_not_proofs() {
    let fixture = Fixture::new(&[1.0, 2.0, 3.0, 4.0]);
    let (record, _) = fixture.record(None, 0.5);
    for variant in 0..3 {
        let mut tampered = record.clone();
        match variant {
            0 => tampered.body.expected_model.parameter_digest = id("fake-parameters"),
            1 => tampered.body.costs.kernel_work_upper_bound += 1,
            _ => tampered.body.backend_digest = id("fake-backend"),
        }
        rehash(&mut tampered);
        let target = tampered.id.clone();
        let graph = PatchGraph::from_records(vec![tampered]).unwrap();
        assert_eq!(
            replay(
                &graph,
                &target,
                &fixture.root,
                &BTreeMap::new(),
                ReplayLimits::default()
            )
            .unwrap_err()
            .code,
            ErrorCode::IntegrityMismatch
        );
    }
}

#[test]
fn replay_wrong_base_implicit_rebase_and_runtime_budgets_are_rejected() {
    let fixture = Fixture::new(&[1.0, 2.0, 3.0, 4.0]);
    let (first, parent) = fixture.record(None, 0.5);
    let (second, _) = fixture.record(Some((&first, &parent)), 2.0);
    let graph = PatchGraph::from_records(vec![first.clone(), second.clone()]).unwrap();
    let wrong = Fixture::new(&[2.0, 3.0, 4.0, 5.0]);
    assert_eq!(
        replay(
            &graph,
            &second.id,
            &wrong.root,
            &BTreeMap::new(),
            ReplayLimits::default()
        )
        .unwrap_err()
        .code,
        ErrorCode::SourceMismatch
    );
    for bounds in [
        ReplayLimits {
            max_nodes: 1,
            ..ReplayLimits::default()
        },
        ReplayLimits {
            max_materialized_bytes: 1,
            ..ReplayLimits::default()
        },
        ReplayLimits {
            max_kernel_work: 7,
            ..ReplayLimits::default()
        },
    ] {
        assert_eq!(
            replay(&graph, &second.id, &fixture.root, &BTreeMap::new(), bounds)
                .unwrap_err()
                .code,
            ErrorCode::LimitExceeded
        );
    }
    let mut rebased = second;
    rebased.body.base_model = fixture.space.base_model.clone();
    rebased.body.program.base_model = rebased.body.base_model.clone();
    rebased.body.policy.source = rebased.body.base_model.clone();
    rehash(&mut rebased);
    assert_eq!(
        PatchGraph::from_records(vec![first, rebased])
            .unwrap_err()
            .code,
        ErrorCode::SourceMismatch
    );
}

#[test]
fn replay_parent_inspection_history_cannot_be_erased_by_a_direct_child() {
    let mut fixture = Fixture::new(&[1.0, 2.0, 3.0, 4.0]);
    fixture.gradient_inspection = true;
    let (_, parent) = fixture.record(None, 0.5);
    fixture.gradient_inspection = false;
    let (plan, _) = fixture.plan(parent.source_state(), Some(&parent), 2.0, false, false);
    assert_eq!(
        execute_with_history(
            &plan,
            parent.source_state(),
            &fixture.smoke,
            ExecutionLimits::default(),
            &parent
        )
        .unwrap_err()
        .code,
        ErrorCode::PermissionDenied
    );
}

#[test]
fn replay_missing_operand_artifacts_are_explicit_failures_not_empty_results() {
    let fixture = Fixture::new(&[1.0, 2.0, 3.0, 4.0]);
    let (mut record, _) = fixture.record(None, 0.5);
    let artifact = nf_weight_ir::program::TensorArtifact {
        digest: id("missing-artifact"),
        tensor: "value".to_owned(),
        dtype: DType::F64,
        shape: vec![1],
        data_bytes: 8,
        artifact_bytes: 80,
        access: nf_core::schemas::AccessClass::Development,
        provenance: nf_core::schemas::EvidenceLabel {
            schema_version: 1,
            origin: nf_core::schemas::EvidenceOrigin::SuppliedFact,
            support: nf_core::schemas::SupportGrade::Untested,
            provenance: "fixture".to_owned(),
            evidence: vec![],
            access: nf_core::schemas::AccessClass::Development,
        },
    };
    record
        .body
        .program
        .artifacts
        .insert("operand".to_owned(), artifact);
    record
        .body
        .operands
        .insert("operand".to_owned(), id("missing-artifact"));
    record.body.program.constraints.max_artifact_bytes = 80;
    record.body.search_space.constraints.max_artifact_bytes = 80;
    let space = record.body.search_space.digest().unwrap();
    record.body.program.objective.search_space = space.clone();
    record.body.policy.search_space = space;
    rehash(&mut record);
    let target = record.id.clone();
    let graph = PatchGraph::from_records(vec![record]).unwrap();
    let error = replay(
        &graph,
        &target,
        &fixture.root,
        &BTreeMap::new(),
        ReplayLimits::default(),
    )
    .unwrap_err();
    assert_eq!(error.code, ErrorCode::IntegrityMismatch);
    assert!(error.message.contains("artifact missing"));
}

#[test]
fn replay_verified_parent_cannot_be_replaced_by_a_forged_effect_dto() {
    let fixture = Fixture::new(&[1.0, 2.0, 3.0, 4.0]);
    let (_, parent) = fixture.record(None, 0.5);
    let (plan, _) = fixture.plan(parent.source_state(), Some(&parent), 0.5, true, true);
    assert_eq!(
        execute(
            &plan,
            parent.source_state(),
            &fixture.smoke,
            ExecutionLimits::default()
        )
        .unwrap_err()
        .code,
        ErrorCode::UnsupportedCapability
    );
    assert_eq!(
        execute_with_history(
            &plan,
            parent.source_state(),
            &fixture.smoke,
            ExecutionLimits::default(),
            &parent
        )
        .unwrap_err()
        .code,
        ErrorCode::ParameterBudgetExceeded
    );
}
