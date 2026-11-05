#![cfg(target_os = "linux")]
//! T038 prerequisite regressions, not evidence acceptance or promotion tests.
#[path = "../../nf-core/tests/common/study_fixture.rs"]
#[allow(dead_code)]
mod fixture;
use fixture::{d, refresh, standard};
use nf_checkpoint::store::{ArtifactStore, StoreLimits};
use nf_core::canonical::ContentDigest;
use nf_core::error::ErrorCode;
use nf_core::schemas::{ModelManifest, Validated};
use nf_core::study::{ModelIdentity, StudyContract};
use nf_experiments::coordinator::broker::{Actor, Quotas, Scope, Session, Task};
use nf_experiments::coordinator::budgets::{Request, Resources, WorkKind};
use nf_experiments::coordinator::{Lease, Ledger, LedgerLimits, Stage, Transition};
use nf_graph::inventory::tensor_set_digest;
use nf_surgeon::transaction::{SourceState, StateLoadLimits};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Root(PathBuf);
impl Root {
    fn new() -> Self {
        let parent = std::env::var_os("KIROCREW_SCRATCH")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        let path = parent.join(format!(
            "acceptance-panel-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn ledger(&self) -> Ledger {
        Ledger::open(&self.0.join("ledger"), LedgerLimits::default()).unwrap()
    }
}
impl Drop for Root {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}
fn contract(value: &Value) -> Validated<StudyContract> {
    Validated::parse(&serde_json::to_vec(value).unwrap()).unwrap()
}
fn verify_phase(ledger: &mut Ledger, lease: &Lease) {
    let stages = [
        Stage::Frozen,
        Stage::Preflight,
        Stage::Profile,
        Stage::Baseline,
        Stage::Design,
        Stage::Execute,
        Stage::Evaluate,
        Stage::Verify,
    ];
    for (i, pair) in stages.windows(2).enumerate() {
        ledger
            .advance(
                lease,
                Transition {
                    operation: d(&format!("panel-phase-{i}")),
                    expected: pair[0],
                    next: pair[1],
                    protocol: d("panel-software-fixture"),
                    evidence: d("not-model-evidence"),
                    message: "software fixture only".into(),
                },
            )
            .unwrap();
    }
}
fn snapshot(ledger: &Ledger, id: &ContentDigest) -> Value {
    serde_json::to_value(ledger.frozen_final_panel(id).unwrap()).unwrap()
}
fn hex(text: &str) -> Vec<u8> {
    text.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}
// Same committed CPU source fixture and identity construction as fixture_broker.
fn source_contract() -> (Arc<SourceState>, Value) {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../python/neural_foundry/data/m1_fixture.json"
    ))
    .unwrap();
    let manifest: ModelManifest = serde_json::from_value(fixture["manifest"].clone()).unwrap();
    let assets: BTreeMap<String, Arc<[u8]>> = fixture["assets_hex"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, value)| (name.clone(), Arc::from(hex(value.as_str().unwrap()))))
        .collect();
    let source = Arc::new(
        SourceState::load(
            Validated::new(manifest).unwrap(),
            Arc::from(hex(fixture["checkpoint_hex"].as_str().unwrap())),
            assets,
            StateLoadLimits::default(),
        )
        .unwrap(),
    );
    let model = json!({"checkpoint":ContentDigest::of_bytes(source.checkpoint_bytes()),
        "manifest":source.manifest_digest(),"parameters":source.manifest().parameter_digest,
        "buffers":tensor_set_digest(&source.manifest().buffers).unwrap(),
        "graph":fixture["program"]["base_model"]["architecture_graph"],
        "config":source.manifest().architecture.config,"tokenizer":null,
        "system_state":null,"runtime_artifact":null});
    let mut value = standard();
    value["source"] = model.clone();
    value["best_known"] = model.clone();
    for baseline in value["baselines"].as_array_mut().unwrap() {
        baseline["input"] = model.clone();
    }
    (source, value)
}
fn scope(c: &StudyContract, binding: ContentDigest) -> Scope {
    Scope {
        binding,
        actor: Actor::FinalEvaluator,
        tasks: c.portfolio.bindings[0]
            .splits
            .holdout
            .items
            .iter()
            .map(|item| Task {
                id: item.item.clone(),
                group: item.group.clone(),
                input: vec![1.0, 2.0],
                labels: Some(json!({"expected":[5.0,11.0]})),
            })
            .collect(),
        quotas: Quotas {
            messages: 8,
            forwards: 1,
            tool_calls: 0,
            input_units: 8,
            response_bytes: 65536,
            staged_bytes: 256,
        },
    }
}

#[test]
fn acceptance_gate_panel_requires_verify_and_bounded_unique_finalists() {
    let root = Root::new();
    let c = contract(&standard());
    let mut ledger = root.ledger();
    let id = ledger.freeze(&c).unwrap();
    let lease = ledger.acquire(&id, "panel-tests", 300000).unwrap();
    assert_eq!(
        ledger
            .freeze_final_panel(&lease, vec![c.source.clone()])
            .unwrap_err()
            .code,
        ErrorCode::PermissionDenied
    );
    assert!(ledger.frozen_final_panel(&id).unwrap().is_none());
    verify_phase(&mut ledger, &lease);
    let mut other = c.source.clone();
    other.parameters = d("second-candidate");
    for finalists in [vec![], vec![c.source.clone(), other]] {
        assert_eq!(
            ledger
                .freeze_final_panel(&lease, finalists)
                .unwrap_err()
                .code,
            ErrorCode::QuotaExceeded
        );
        assert!(ledger.frozen_final_panel(&id).unwrap().is_none());
    }
    // Raise the declared bound in a separate store to reach duplicate checking.
    let second = Root::new();
    let mut value = standard();
    value["statistics"]["maximum_finalists"] = json!(2);
    let c = contract(&value);
    let mut ledger = second.ledger();
    let id = ledger.freeze(&c).unwrap();
    let lease = ledger.acquire(&id, "duplicate-tests", 300000).unwrap();
    verify_phase(&mut ledger, &lease);
    assert_eq!(
        ledger
            .freeze_final_panel(&lease, vec![c.source.clone(); 2])
            .unwrap_err()
            .code,
        ErrorCode::InvalidValue
    );
    assert!(ledger.frozen_final_panel(&id).unwrap().is_none());
}

#[test]
fn acceptance_gate_panel_freezes_every_model_identity_coordinate_and_survives_reopen() {
    let root = Root::new();
    let c = contract(&standard());
    let mut ledger = root.ledger();
    let id = ledger.freeze(&c).unwrap();
    let lease = ledger.acquire(&id, "identity-tests", 300000).unwrap();
    verify_phase(&mut ledger, &lease);
    let panel_id = ledger
        .freeze_final_panel(&lease, vec![c.source.clone()])
        .unwrap();
    let frozen = snapshot(&ledger, &id);
    let (_, panel) = ledger.frozen_final_panel(&id).unwrap().unwrap();
    assert_eq!(panel.contract, id);
    assert_eq!(panel.revision, c.revision);
    assert_eq!(
        panel.protocol,
        c.portfolio.independent.frozen_panel_protocol
    );
    assert_eq!(
        serde_json::to_value(&panel.finalists).unwrap(),
        json!([c.source])
    );
    assert_eq!(
        serde_json::to_value(&panel.original).unwrap(),
        json!(c.source)
    );
    assert_eq!(
        serde_json::to_value(&panel.best_known).unwrap(),
        json!(c.best_known)
    );
    assert_eq!(
        panel.bindings,
        c.portfolio
            .bindings
            .iter()
            .filter(|b| b.schedule.final_required)
            .map(|b| b.id.clone())
            .collect()
    );
    assert_eq!(panel.seeds, c.seeds);
    assert_eq!(panel.repeats, c.statistics.repeats);
    assert_eq!(panel.comparison_family, c.statistics.multiplicity_family);
    assert_eq!(
        panel.status,
        "frozen_identities_not_model_execution_evidence"
    );
    for field in [
        "checkpoint",
        "manifest",
        "parameters",
        "buffers",
        "graph",
        "config",
        "tokenizer",
        "system_state",
        "runtime_artifact",
    ] {
        let mut changed = serde_json::to_value(&c.source).unwrap();
        changed[field] = json!(d(&format!("changed-{field}")));
        let candidate: ModelIdentity = serde_json::from_value(changed).unwrap();
        assert_eq!(
            ledger
                .freeze_final_panel(&lease, vec![candidate])
                .unwrap_err()
                .code,
            ErrorCode::RevisionConflict,
            "{field}"
        );
        assert_eq!(snapshot(&ledger, &id), frozen);
    }
    assert_eq!(
        ledger
            .freeze_final_panel(&lease, vec![c.source.clone()])
            .unwrap(),
        panel_id
    );
    ledger.release(&lease).unwrap();
    drop(ledger);
    let mut ledger = root.ledger();
    assert_eq!(snapshot(&ledger, &id), frozen);
    let lease = ledger.acquire(&id, "reopened", 300000).unwrap();
    assert_eq!(
        ledger
            .freeze_final_panel(&lease, vec![c.source.clone()])
            .unwrap(),
        panel_id
    );
}

#[test]
fn acceptance_gate_panel_binding_threshold_and_candidate_changes_fail_before_and_after_holdout() {
    let root = Root::new();
    let (source, value) = source_contract();
    let c = contract(&value);
    let mut ledger = root.ledger();
    let id = ledger.freeze(&c).unwrap();
    let lease = ledger.acquire(&id, "holdout-tests", 300000).unwrap();
    verify_phase(&mut ledger, &lease);
    let store = ArtifactStore::create(
        &root.0.join("store"),
        StoreLimits {
            total_bytes: 2097152,
            max_blob_bytes: 65536,
            max_entries: 128,
        },
    )
    .unwrap();
    let reserve_pool = |ledger: &mut Ledger, operation: &str| {
        ledger
            .reserve(
                &lease,
                Request {
                    schema_version: 1,
                    operation: d(operation),
                    attempt: 0,
                    parent: None,
                    kind: WorkKind::Branch,
                    resources: Resources {
                        cpu_ms: 10000,
                        evaluations: 4,
                        host_bytes: 16777216,
                        storage_bytes: 4194304,
                        workers: 1,
                        ..Resources::default()
                    },
                    wall_ms: 30000,
                    external: None,
                },
            )
            .unwrap()
    };
    let mut pool = reserve_pool(&mut ledger, "holdout-pool");
    let binding = c.portfolio.bindings[0].id.clone();
    let error = Session::trusted_fixture(
        &mut ledger,
        lease.clone(),
        pool.clone(),
        source.clone(),
        &store,
        scope(&c, binding.clone()),
    )
    .err()
    .expect("holdout must require an actual frozen panel");
    assert_eq!(error.code, ErrorCode::PermissionDenied);
    ledger
        .freeze_final_panel(&lease, vec![c.source.clone()])
        .unwrap();
    let frozen = snapshot(&ledger, &id);
    for opened in [false, true] {
        if opened {
            // Constructor records opening before it discloses holdout metadata.
            let session = Session::trusted_fixture(
                &mut ledger,
                lease.clone(),
                pool.clone(),
                source.clone(),
                &store,
                scope(&c, binding.clone()),
            )
            .unwrap();
            assert_eq!(session.welcome().contract, id);
            assert_eq!(session.welcome().split, nf_core::study::SplitRole::Holdout);
            assert_eq!(session.welcome().binding, binding);
            drop(session);
            // Session drop closes its allocation; use a live pool so subsequent
            // scope errors exercise binding checks rather than budget fencing.
            pool = reserve_pool(&mut ledger, "post-holdout-pool");
        }
        let mut candidate = c.source.clone();
        candidate.parameters = d("post-freeze-candidate");
        assert_eq!(
            ledger
                .freeze_final_panel(&lease, vec![candidate])
                .unwrap_err()
                .code,
            ErrorCode::RevisionConflict
        );
        for (pointer, replacement) in [
            ("/portfolio/bindings/0/id", json!(d("altered-binding"))),
            (
                "/portfolio/bindings/0/metrics/0/threshold/min_gain",
                json!(0.02),
            ),
            (
                "/portfolio/bindings/1/metrics/0/threshold/max_regression",
                json!(0.02),
            ),
            (
                "/portfolio/bindings/3/metrics/0/threshold/ceiling",
                json!(200.0),
            ),
            ("/seeds", json!([18, 43])),
            ("/statistics/repeats", json!(2)),
        ] {
            let mut changed = value.clone();
            *changed.pointer_mut(pointer).unwrap() = replacement;
            if pointer.ends_with("/id") {
                changed["objective"]["primary_binding"] = json!(d("altered-binding"));
            }
            refresh(&mut changed);
            // Valid schema and refreshed portfolio digest: this reaches ledger immutability.
            let changed = contract(&changed);
            assert_eq!(
                ledger.freeze(&changed).unwrap_err().code,
                ErrorCode::RevisionConflict,
                "{pointer}, opened={opened}"
            );
            assert_eq!(ledger.contract(&id).unwrap().digest().unwrap(), id);
            assert_eq!(snapshot(&ledger, &id), frozen);
        }
        // A genuine other binding cannot borrow this binding's holdout tasks.
        for (other, code) in [
            (d("unknown-binding"), ErrorCode::SourceMismatch),
            (
                c.portfolio.bindings[1].id.clone(),
                ErrorCode::SourceMismatch,
            ),
        ] {
            let error = Session::trusted_fixture(
                &mut ledger,
                lease.clone(),
                pool.clone(),
                source.clone(),
                &store,
                scope(&c, other),
            )
            .err()
            .expect("altered scope must fail closed");
            assert_eq!(error.code, code);
            assert_eq!(snapshot(&ledger, &id), frozen);
        }
    }
    ledger.release(&lease).unwrap();
    drop(ledger);
    assert_eq!(snapshot(&root.ledger(), &id), frozen);
}
