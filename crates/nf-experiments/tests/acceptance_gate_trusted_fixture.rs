#![cfg(target_os = "linux")]
//! Real local native final acceptance, not supplied score/receipt acceptance.
#[path = "../../nf-core/tests/common/study_fixture.rs"]
#[allow(dead_code)]
mod fixture;
use fixture::{d, hash, refresh, standard};
use nf_checkpoint::store::{ArtifactStore, StoreLimits};
use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use nf_core::schemas::{ModelManifest, Validated};
use nf_core::study::{ModelIdentity, SplitRole, StudyContract};
use nf_experiments::coordinator::acceptance::Status;
use nf_experiments::coordinator::final_fixture::{graph_bytes, profile_id, runtime_profile};
use nf_experiments::coordinator::splits::{publish_dataset, Dataset, Item};
use nf_experiments::coordinator::{Lease, Ledger, LedgerLimits, Stage, Transition};
use nf_graph::inventory::{tensor_set_digest, verify_inventory, InventoryLimits};
use nf_surgeon::cancellation::CancellationToken;
use nf_surgeon::transaction::{SmokeSuite, SourceState, StateLoadLimits};
use nf_weight_ir::compilation::{compile, CompilePolicy, CompiledPlan, SourceSnapshot};
use nf_weight_ir::program::parse_program;
use nf_weight_ir::search_space::{parse_search_space, resolve_space};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
static NEXT: AtomicU64 = AtomicU64::new(0);
#[path = "support/t038_security.rs"]
mod t038_security;
#[path = "support/t038_supervised.rs"]
mod t038_supervised;
/// Each fixture test runs in a fresh native process with an irreversible kernel
/// address-space limit. Never mutate the shared cargo/test controller's limit.
fn isolated() -> bool {
    let limit = rustix::process::getrlimit(rustix::process::Resource::As);
    if limit.maximum.is_some_and(|n| n <= 1_000_000_000) {
        return false;
    }
    let status = std::process::Command::new("/usr/bin/prlimit")
        .args(["--as=1000000000:1000000000", "--core=0:0", "--"])
        .arg(std::env::current_exe().unwrap())
        .args([
            "--exact",
            std::thread::current().name().unwrap(),
            "--test-threads=1",
            "--include-ignored",
        ])
        .status()
        .unwrap();
    assert!(
        status.success(),
        "isolated fixture subprocess failed: {status}"
    );
    true
}
fn bytes(v: &impl Serialize) -> Vec<u8> {
    CanonicalDocument::parse(&serde_json::to_vec(v).unwrap(), Limits::default())
        .unwrap()
        .bytes()
        .to_vec()
}
fn unhex(s: &str) -> Vec<u8> {
    s.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
        .collect()
}
struct Setup {
    root: PathBuf,
    ledger: Ledger,
    lease: Lease,
    store: ArtifactStore,
    edit: CompiledPlan,
    noop: CompiledPlan,
    smoke: Validated<SmokeSuite>,
}
impl Drop for Setup {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}
impl Setup {
    fn new(retention_damage: bool, missing_catalog: bool) -> Self {
        Self::custom(retention_damage, missing_catalog, |_| {})
    }
    fn custom(
        retention_damage: bool,
        missing_catalog: bool,
        change: impl FnOnce(&mut Value),
    ) -> Self {
        Self::custom_artifacts(retention_damage, missing_catalog, |c, _| change(c))
    }
    fn custom_artifacts(
        retention_damage: bool,
        missing_catalog: bool,
        change: impl FnOnce(&mut Value, &ArtifactStore),
    ) -> Self {
        Self::custom_probes(retention_damage, missing_catalog, 1, change)
    }
    fn custom_probes(
        retention_damage: bool,
        missing_catalog: bool,
        probes: usize,
        change: impl FnOnce(&mut Value, &ArtifactStore),
    ) -> Self {
        let root = std::env::temp_dir().join(format!(
            "trusted-final-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).unwrap();
        let store = ArtifactStore::create(
            &root.join("store"),
            StoreLimits {
                total_bytes: 16_777_216,
                max_blob_bytes: 1_048_576,
                max_entries: 1024,
            },
        )
        .unwrap();
        let mut f: Value = serde_json::from_str(include_str!(
            "../../../python/neural_foundry/data/m1_fixture.json"
        ))
        .unwrap();
        f["smoke_suite"]["probes"] = json!(vec![f["smoke_suite"]["probes"][0].clone(); probes]);
        f["program"]["verification"]["smoke_suite"] = json!(hash(&f["smoke_suite"]));
        let source = SourceState::load(
            Validated::<ModelManifest>::parse(&bytes(&f["manifest"])).unwrap(),
            Arc::from(unhex(f["checkpoint_hex"].as_str().unwrap())),
            BTreeMap::from([("config".into(), Arc::from(b"{}".as_slice()))]),
            StateLoadLimits::default(),
        )
        .unwrap();
        store.put(bytes(&**source.manifest()).as_slice()).unwrap();
        store.put(source.checkpoint_bytes()).unwrap();
        store.put(b"{}".as_slice()).unwrap();
        store.put(graph_bytes()).unwrap();
        let model = ModelIdentity {
            checkpoint: ContentDigest::of_bytes(source.checkpoint_bytes()),
            manifest: source.manifest_digest().clone(),
            parameters: source.manifest().parameter_digest.clone(),
            buffers: tensor_set_digest(&source.manifest().buffers).unwrap(),
            graph: ContentDigest::of_bytes(graph_bytes()),
            config: source.manifest().architecture.config.clone(),
            tokenizer: None,
            system_state: None,
            runtime_artifact: None,
        };
        let mut c = standard();
        c["source"] = json!(model);
        c["best_known"] = json!(model);
        for b in c["baselines"].as_array_mut().unwrap() {
            b["input"] = json!(model);
            b["backend"] = json!({"id":"fixture.cpu","version":"1","config":profile_id()});
        }
        c["statistics"]["repeats"] = json!(1);
        c["seeds"] = json!([17]);
        c["runtime"] = json!(runtime_profile().unwrap());
        c["limits"]["host_memory_bytes"] = json!(1_000_000_000_u64);
        let old = c["portfolio"]["gyms"].as_object().unwrap().clone();
        c["portfolio"]["gyms"] = json!({});
        for (i, b) in c["portfolio"]["bindings"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .enumerate()
        {
            for m in b["metrics"].as_array_mut().unwrap() {
                if !m["measurement"].is_null() {
                    m["measurement"]["timer"] =
                        json!({"id":"nf.final.monotonic","version":"1","config":profile_id()});
                }
                if m["metric"] == "memory" {
                    m["threshold"]["ceiling"] = json!(1_000_000_000.0);
                }
            }
            for (j, role) in ["development", "selection", "holdout"]
                .into_iter()
                .enumerate()
            {
                let identity =
                    serde_json::from_value(b["splits"][role]["items"][0].clone()).unwrap();
                let input = if i == 1 && j == 2 && !retention_damage {
                    vec![0.0, 0.0]
                } else {
                    vec![(i * 10 + j + 1) as f64, 2.0]
                };
                let original = vec![input[0] + 2.0 * input[1], 3.0 * input[0] + 4.0 * input[1]];
                let label = if i == 1 {
                    original
                } else {
                    original.iter().map(|x| x * 0.5).collect()
                };
                let input = store.put(bytes(&input).as_slice()).unwrap().digest;
                let label = store.put(bytes(&label).as_slice()).unwrap().digest;
                let dataset = Dataset {
                    schema_version: 1,
                    role: match j {
                        0 => SplitRole::Development,
                        1 => SplitRole::Selection,
                        _ => SplitRole::Holdout,
                    },
                    items: vec![Item {
                        identity,
                        input,
                        label,
                    }],
                };
                b["splits"][role]["dataset"] = json!(publish_dataset(&store, &dataset).unwrap());
            }
        }
        for i in 0..5 {
            let b = &c["portfolio"]["bindings"][i];
            let mut g = old[b["gym"].as_str().unwrap()].clone();
            g["adapter"] =
                json!({"id":"nf.final.linear_exact","version":"1","config":profile_id()});
            for k in [
                "code",
                "scorer",
                "oracle",
                "scoring_protocol",
                "validation",
                "dependencies",
                "input_schema",
                "output_schema",
            ] {
                g[k] = json!(profile_id());
            }
            let id = hash(&g);
            c["portfolio"]["gyms"][id.as_str()] = g;
            c["portfolio"]["bindings"][i]["gym"] = json!(id);
            c["portfolio"]["bindings"][i]["validation"] = json!(profile_id());
        }
        c["portfolio"]["bindings"][2]["metrics"][0]["comparator"] = json!("no_op");
        c["portfolio"]["bindings"][4]["metrics"][0]["comparator"] = json!("best_known");
        refresh(&mut c);
        f["program"]["operations"][0]["operation"]["factor"] = json!(0.5);
        f["program"]["base_model"]["architecture_graph"] = json!(model.graph);
        f["space"]["base_model"] = f["program"]["base_model"].clone();
        let space = parse_search_space(&bytes(&f["space"])).unwrap();
        f["program"]["objective"]["search_space"] = json!(space.digest().unwrap());
        f["program"]["verification"]["portfolio"] = c["portfolio_digest"].clone();
        f["policy"]["source"] = f["program"]["base_model"].clone();
        f["policy"]["search_space"] = json!(space.digest().unwrap());
        f["policy"]["verification"] = f["program"]["verification"].clone();
        c["contract_key"] = f["program"]["objective"]["contract"].clone();
        c["objective"]["id"] = f["program"]["objective"]["objective"].clone();
        let objective = c["objective"]["id"].clone();
        for b in c["portfolio"]["bindings"].as_array_mut().unwrap() {
            b["objective"] = objective.clone();
        }
        refresh(&mut c);
        f["program"]["verification"]["portfolio"] = c["portfolio_digest"].clone();
        f["policy"]["verification"] = f["program"]["verification"].clone();
        c["intervention"]["scope"] = json!(hash(&f["program"]["effects"]));
        c["intervention"]["transformation_space"] = json!(space.digest().unwrap());
        c["intervention"]["method_protocol"] =
            f["program"]["objective"]["experiment_protocol"].clone();
        c["intervention"]["allowed_engines"] =
            json!([{"id":"nf.native_cpu","version":"1","config":hash(&f["policy"]["backend"])}]);
        change(&mut c, &store);
        refresh(&mut c);
        f["program"]["verification"]["portfolio"] = c["portfolio_digest"].clone();
        f["policy"]["verification"] = f["program"]["verification"].clone();
        let observations = source.observations();
        let inventory =
            verify_inventory(source.manifest(), &observations, InventoryLimits::default()).unwrap();
        let resolved = resolve_space(&space, &inventory).unwrap();
        let policy = Validated::<CompilePolicy>::parse(&bytes(&f["policy"])).unwrap();
        let compile_plan = |v: &Value| {
            compile(
                &parse_program(&bytes(v)).unwrap(),
                SourceSnapshot {
                    manifest: source.manifest(),
                    tensors: &observations,
                    checkpoint: source.checkpoint_bytes(),
                    assets: source.assets(),
                },
                &resolved,
                &BTreeMap::new(),
                &policy,
                &[],
            )
            .unwrap()
        };
        let edit = compile_plan(&f["program"]);
        let mut n = f["program"].clone();
        n["operations"] = json!([]);
        n["effects"]["reads"] = json!([]);
        n["effects"]["writes"] = json!([]);
        n["lineage"]["method"]["target_update"] = json!("no_change");
        n["lineage"]["method"]["changed_parameters"] = json!(0);
        n["constraints"]["max_changed_parameters"] = json!(0);
        let noop = compile_plan(&n);
        let smoke = Validated::<SmokeSuite>::parse(&bytes(&f["smoke_suite"])).unwrap();
        let c = Validated::<StudyContract>::parse(&bytes(&c)).unwrap();
        let mut ledger = Ledger::open(&root.join("ledger"), LedgerLimits::default()).unwrap();
        let id = ledger.freeze(&c).unwrap();
        let lease = ledger.acquire(&id, "native-final-tests", 300000).unwrap();
        for b in c
            .portfolio
            .bindings
            .iter()
            .take(if missing_catalog { 4 } else { 5 })
        {
            ledger.bind_split_catalog(&lease, &store, &b.id).unwrap();
        }
        let mut previous = Stage::Frozen;
        for (i, next) in [
            Stage::Preflight,
            Stage::Profile,
            Stage::Baseline,
            Stage::Design,
            Stage::Execute,
            Stage::Evaluate,
            Stage::Verify,
        ]
        .into_iter()
        .enumerate()
        {
            ledger
                .advance(
                    &lease,
                    Transition {
                        operation: d(&format!("transition{i}")),
                        expected: previous,
                        next,
                        protocol: d("test-phase"),
                        evidence: d("not-scientific-evidence"),
                        message: "software fixture phase".into(),
                    },
                )
                .unwrap();
            previous = next;
        }
        Self {
            root,
            ledger,
            lease,
            store,
            edit,
            noop,
            smoke,
        }
    }
    fn run(&mut self) -> nf_experiments::coordinator::acceptance::Decision {
        self.ledger
            .run_fixture_final(
                &self.lease,
                &self.store,
                &self.edit,
                &self.noop,
                &self.smoke,
                &CancellationToken::default(),
            )
            .unwrap()
    }
}
#[test]
fn acceptance_gate_trusted_fixture_accepts_actual_execution_and_durable_reopen() {
    if isolated() {
        return;
    }
    let mut s = Setup::new(false, false);
    let result = s.run();
    assert!(
        result.report().accepted,
        "{}",
        String::from_utf8_lossy(result.bytes())
    );
    assert_eq!(result.report().status, Status::Passed);
    assert!(!result.report().promotion_authorized);
    let snapshot = s.ledger.budget_snapshot(s.lease.contract()).unwrap();
    assert_eq!(snapshot.spent.evaluations, 86);
    assert_eq!(snapshot.held.evaluations, 0);
    assert_eq!(snapshot.spent.attempts, 1);
    assert!(snapshot.spent.storage_bytes > 0);
    assert_eq!(snapshot.unknown_charged.storage_bytes, 4_194_304);
    assert_eq!(snapshot.unknown_charged.cpu_ms, 30_000);
    let reopened = Ledger::open(&s.root.join("ledger"), LedgerLimits::default()).unwrap();
    assert_eq!(
        reopened
            .fixture_final_decision(s.lease.contract())
            .unwrap()
            .unwrap()
            .bytes(),
        result.bytes()
    );
    assert!(s
        .ledger
        .run_fixture_final(
            &s.lease,
            &s.store,
            &s.edit,
            &s.noop,
            &s.smoke,
            &CancellationToken::default()
        )
        .is_err());
    let diagnostic = s.ledger.acceptance_gate(&s.lease, None).unwrap();
    assert!(!diagnostic.report().accepted);
}
#[test]
fn acceptance_gate_trusted_fixture_retention_damage_never_accepts() {
    if isolated() {
        return;
    }
    let mut s = Setup::new(true, false);
    let d = s.run();
    assert!(!d.report().accepted);
    assert_eq!(d.report().status, Status::Failed);
    assert!(d
        .report()
        .checks
        .iter()
        .any(|c| c.binding == Some(fixture::d("binding0")) && c.status == Status::Passed));
    assert!(d
        .report()
        .checks
        .iter()
        .any(|c| c.binding == Some(fixture::d("binding1")) && c.status == Status::Failed));
    let reopened = Ledger::open(&s.root.join("ledger"), LedgerLimits::default()).unwrap();
    let durable = reopened
        .fixture_final_decision(s.lease.contract())
        .unwrap()
        .unwrap();
    assert_eq!(durable.bytes(), d.bytes());
    assert!(!durable.report().accepted);
}
#[test]
fn acceptance_gate_trusted_fixture_missing_catalog_is_durable_incomplete_not_acceptance() {
    if isolated() {
        return;
    }
    let mut s = Setup::new(false, true);
    let d = s.run();
    assert!(!d.report().accepted);
    assert_eq!(d.report().status, Status::Incomplete);
    assert_eq!(
        s.ledger.fixture_final_status(s.lease.contract()).unwrap(),
        Some(Status::Incomplete)
    );
    assert!(s
        .ledger
        .run_fixture_final(
            &s.lease,
            &s.store,
            &s.edit,
            &s.noop,
            &s.smoke,
            &CancellationToken::default()
        )
        .is_err());
}
#[test]
fn acceptance_gate_trusted_fixture_serialized_success_and_stale_lease_cannot_authorize() {
    if isolated() {
        return;
    }
    let mut s = Setup::new(false, false);
    let mut forged = json!({"accepted":true,"status":"passed","promotion_authorized":true,"execution":"complete"});
    forged["contract"] = json!(s.lease.contract());
    s.store.put(bytes(&forged).as_slice()).unwrap();
    assert!(s
        .ledger
        .fixture_final_decision(s.lease.contract())
        .unwrap()
        .is_none());
    assert!(
        !s.ledger
            .acceptance_gate(&s.lease, None)
            .unwrap()
            .report()
            .accepted
    );
    s.ledger.release(&s.lease).unwrap();
    assert!(s
        .ledger
        .run_fixture_final(
            &s.lease,
            &s.store,
            &s.edit,
            &s.noop,
            &s.smoke,
            &CancellationToken::default()
        )
        .is_err());
    assert!(s
        .ledger
        .fixture_final_status(s.lease.contract())
        .unwrap()
        .is_none());
}
#[test]
fn acceptance_gate_trusted_fixture_resource_damage_never_accepts_target_gain() {
    if isolated() {
        return;
    }
    let mut s = Setup::custom(false, false, |c| {
        c["portfolio"]["bindings"][3]["metrics"][1]["threshold"]["ceiling"] = json!(1.0)
    });
    let result = s.run();
    assert!(!result.report().accepted);
    assert_eq!(result.report().status, Status::Failed);
    assert!(result
        .report()
        .checks
        .iter()
        .any(|c| c.metric.as_deref() == Some("memory") && c.status == Status::Failed));
    assert!(result
        .report()
        .checks
        .iter()
        .any(|c| c.binding == Some(d("binding0")) && c.status == Status::Passed));
}
#[test]
fn acceptance_gate_trusted_fixture_runtime_forgery_and_insufficient_budget_never_dispatch() {
    if isolated() {
        return;
    }
    for variant in [0, 1] {
        let mut s = Setup::custom(false, false, |c| {
            if variant == 0 {
                c["runtime"]["environment"] = json!(d("forged-host"));
            } else {
                c["limits"]["model_evaluations"] = json!(40);
            }
        });
        assert!(s
            .ledger
            .run_fixture_final(
                &s.lease,
                &s.store,
                &s.edit,
                &s.noop,
                &s.smoke,
                &CancellationToken::default()
            )
            .is_err());
        assert!(s
            .ledger
            .fixture_final_status(s.lease.contract())
            .unwrap()
            .is_none());
    }
}
#[test]
fn acceptance_gate_trusted_fixture_missing_model_artifact_never_accepts() {
    if isolated() {
        return;
    }
    let mut s = Setup::custom(false, false, |c| {
        let bad = json!(d("unresolved-checkpoint"));
        c["source"]["checkpoint"] = bad.clone();
        c["best_known"]["checkpoint"] = bad.clone();
        for b in c["baselines"].as_array_mut().unwrap() {
            b["input"]["checkpoint"] = bad.clone();
        }
    });
    let result = s.run();
    assert!(!result.report().accepted);
    assert_eq!(result.report().status, Status::Incomplete);
    assert!(s
        .ledger
        .frozen_final_panel(s.lease.contract())
        .unwrap()
        .is_none());
}
#[test]
fn acceptance_gate_trusted_fixture_interruption_after_durable_intent_stays_incomplete() {
    if isolated() {
        return;
    }
    let mut s = Setup::new(false, false);
    let cancel = CancellationToken::default();
    let stop = cancel.clone();
    let path = s.root.join("ledger/events.sqlite3");
    let watcher = std::thread::spawn(move || {
        let db =
            rusqlite::Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
                .unwrap();
        for _ in 0..5000 {
            let started = db
                .query_row(
                    "SELECT count(*) FROM final_fixture_records WHERE kind='intent'",
                    [],
                    |r| r.get::<_, u64>(0),
                )
                .unwrap_or(0);
            if started > 0 {
                stop.request();
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        panic!("final intent not observed");
    });
    let result = s
        .ledger
        .run_fixture_final(&s.lease, &s.store, &s.edit, &s.noop, &s.smoke, &cancel);
    watcher.join().unwrap();
    if let Ok(d) = result {
        assert!(!d.report().accepted);
        assert_eq!(d.report().status, Status::Incomplete);
    }
    let budget = s.ledger.budget_snapshot(s.lease.contract()).unwrap();
    assert_eq!(budget.held.evaluations, 0);
    assert_eq!(budget.spent.evaluations, 86);
    let reopened = Ledger::open(&s.root.join("ledger"), LedgerLimits::default()).unwrap();
    assert_eq!(
        reopened.fixture_final_status(s.lease.contract()).unwrap(),
        Some(Status::Incomplete)
    );
    assert!(s
        .ledger
        .run_fixture_final(
            &s.lease,
            &s.store,
            &s.edit,
            &s.noop,
            &s.smoke,
            &CancellationToken::default()
        )
        .is_err());
}
#[test]
fn acceptance_gate_trusted_fixture_receipts_are_immutable_and_missing_or_forged_never_accept() {
    if isolated() {
        return;
    }
    for corrupt in 0..4 {
        let mut s = Setup::new(false, false);
        assert!(s.run().report().accepted);
        let db = rusqlite::Connection::open(s.root.join("ledger/events.sqlite3")).unwrap();
        assert!(db
            .execute(
                "DELETE FROM final_fixture_records WHERE kind='execution'",
                []
            )
            .is_err());
        assert!(db
            .execute(
                "UPDATE final_fixture_records SET body=x'7b7d' WHERE kind='execution'",
                []
            )
            .is_err());
        // Simulate out-of-band disk corruption, not an authorized receipt API.
        db.execute_batch(match corrupt {
            0 => "DROP TRIGGER final_fixture_no_update;UPDATE final_fixture_records SET body=x'7b7d' WHERE kind='execution';",
            1 => "DROP TRIGGER final_fixture_no_delete;DELETE FROM final_fixture_records WHERE kind='execution';",
            2 => "DROP TRIGGER panels_no_delete;DELETE FROM final_panels;",
            _ => "DROP TRIGGER outcomes_no_delete;DELETE FROM budget_outcomes;",
        }).unwrap();
        assert!(s.ledger.fixture_final_decision(s.lease.contract()).is_err());
    }
}

fn native_record(s: &Setup, kind: &str) -> Value {
    let db = rusqlite::Connection::open_with_flags(
        s.root.join("ledger/events.sqlite3"),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .unwrap();
    let body: Vec<u8> = db
        .query_row(
            "SELECT body FROM final_fixture_records WHERE kind=?1",
            [kind],
            |r| r.get(0),
        )
        .unwrap();
    serde_json::from_slice(&body).unwrap()
}
#[test]
fn acceptance_gate_trusted_fixture_reserves_every_smoke_forward_at_exact_caps() {
    if isolated() {
        return;
    }
    for probes in [1, 64] {
        let total = 80 + 6 * probes;
        for cap in [80, total - 1, total] {
            let mut s = Setup::custom_probes(false, false, probes, |c, _| {
                c["limits"]["model_evaluations"] = json!(cap)
            });
            let result = s.ledger.run_fixture_final(
                &s.lease,
                &s.store,
                &s.edit,
                &s.noop,
                &s.smoke,
                &CancellationToken::default(),
            );
            if cap < total {
                assert!(result.is_err());
                assert!(s
                    .ledger
                    .fixture_final_status(s.lease.contract())
                    .unwrap()
                    .is_none());
                assert_eq!(
                    s.ledger
                        .budget_snapshot(s.lease.contract())
                        .unwrap()
                        .held
                        .evaluations,
                    0
                );
            } else {
                let result = result.unwrap();
                assert!(
                    result.report().accepted,
                    "{}",
                    String::from_utf8_lossy(result.bytes())
                );
                assert_eq!(
                    s.ledger
                        .budget_snapshot(s.lease.contract())
                        .unwrap()
                        .spent
                        .evaluations,
                    total as u64
                );
                let accounting = &native_record(&s, "execution")["receipt"]["forward_accounting"];
                assert_eq!(accounting["transaction_smoke"], json!(2 * probes));
                assert_eq!(accounting["export_and_reload_smoke"], json!(4 * probes));
                assert_eq!(accounting["holdout"], json!(80));
                assert_eq!(accounting["total"], json!(total));
            }
        }
    }
}
#[test]
fn acceptance_gate_trusted_fixture_small_resource_limits_refuse_before_native_work() {
    if isolated() {
        return;
    }
    for key in ["host_memory_bytes", "storage_bytes", "cpu_ms", "wall_ms"] {
        let mut s = Setup::custom(false, false, |c| c["limits"][key] = json!(1));
        let before = s.store.used_bytes().unwrap();
        assert!(s
            .ledger
            .run_fixture_final(
                &s.lease,
                &s.store,
                &s.edit,
                &s.noop,
                &s.smoke,
                &CancellationToken::default()
            )
            .is_err());
        assert_eq!(s.store.used_bytes().unwrap(), before);
        assert!(s
            .ledger
            .fixture_final_status(s.lease.contract())
            .unwrap()
            .is_none());
        assert!(s
            .ledger
            .frozen_final_panel(s.lease.contract())
            .unwrap()
            .is_none());
        let budget = s.ledger.budget_snapshot(s.lease.contract()).unwrap();
        assert_eq!(budget.reservation_count, 0);
    }
}
#[test]
fn acceptance_gate_trusted_fixture_actual_noop_and_each_observation_match_executed_arm() {
    if isolated() {
        return;
    }
    let mut s = Setup::new(false, false);
    assert!(s.run().report().accepted);
    let execution = native_record(&s, "execution");
    let dispatch = native_record(&s, "dispatch");
    let controls = native_record(&s, "controls");
    let source = &controls["source"];
    let noop = &controls["no_op"];
    assert_ne!(
        source["checkpoint"], noop["checkpoint"],
        "noncanonical alias representation must change on export"
    );
    assert_eq!(source["parameters"], noop["parameters"]);
    for contrast in dispatch["plan"]["protocol"]["contrasts"]
        .as_array()
        .unwrap()
    {
        let index = match contrast["comparator"].as_str().unwrap() {
            "original" => 0,
            "no_op" => 1,
            "best_known" => 2,
            _ => unreachable!(),
        };
        for obs in execution["data"]["observations"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|o| o["contrast"] == contrast["id"])
        {
            let cell = execution["receipt"]["cells"]
                .as_array()
                .unwrap()
                .iter()
                .find(|cell| {
                    cell["binding"] == contrast["binding"]
                        && cell["item"]["identity"]["item"] == obs["item"]
                })
                .unwrap();
            let arm = if obs["arm"] == "candidate" { 3 } else { index };
            assert_eq!(obs["model"], json!(hash(&cell["arm_identities"][arm])));
            assert_eq!(cell["samples"].as_array().unwrap().len(), 16);
        }
    }
    assert!(execution["receipt"]["resource_scope"]
        .as_str()
        .unwrap()
        .contains("no_validation_hashing_or_allocation"));
}
#[test]
fn acceptance_gate_trusted_fixture_distinct_best_known_has_distinct_executed_outputs() {
    if isolated() {
        return;
    }
    let mut s = Setup::custom_artifacts(false, false, |c, store| {
        let f: Value = serde_json::from_str(include_str!(
            "../../../python/neural_foundry/data/m1_fixture.json"
        ))
        .unwrap();
        let mut checkpoint = unhex(f["checkpoint_hex"].as_str().unwrap());
        let weights: Vec<u8> = [0.25_f64, 0.5, 0.75, 1.0]
            .into_iter()
            .flat_map(f64::to_le_bytes)
            .collect();
        let offset = checkpoint.len() - weights.len();
        checkpoint[offset..].copy_from_slice(&weights);
        let checkpoint = store.put(checkpoint.as_slice()).unwrap().digest;
        let mut manifest: ModelManifest = serde_json::from_value(f["manifest"].clone()).unwrap();
        for tensor in manifest.parameters.values_mut() {
            tensor.digest = ContentDigest::of_bytes(&weights);
        }
        manifest.parameter_digest = tensor_set_digest(&manifest.parameters).unwrap();
        manifest
            .assets
            .insert("checkpoint".into(), checkpoint.clone());
        let id = store.put(bytes(&manifest).as_slice()).unwrap().digest;
        c["best_known"]["checkpoint"] = json!(checkpoint);
        c["best_known"]["manifest"] = json!(id);
        c["best_known"]["parameters"] = json!(manifest.parameter_digest);
        let best = c["best_known"].clone();
        for b in c["baselines"].as_array_mut().unwrap() {
            if b["kind"] == "best_known" {
                b["input"] = best.clone();
            }
        }
        c["portfolio"]["bindings"][4]["metrics"][0]["threshold"]["min_gain"] = json!(0.5);
    });
    assert!(s.run().report().accepted);
    let execution = native_record(&s, "execution");
    let cell = &execution["receipt"]["cells"][4];
    assert_ne!(cell["arm_identities"][0], cell["arm_identities"][2]);
    assert_ne!(cell["outputs"][0], cell["outputs"][2]);
    assert_ne!(cell["outputs"][2], cell["outputs"][3]);
    let observations = execution["data"]["observations"].as_array().unwrap();
    assert!(observations
        .iter()
        .any(|o| o["model"] == json!(hash(&cell["arm_identities"][2])) && o["value"] == 0.0));
}
#[test]
fn acceptance_gate_trusted_fixture_unsealed_decision_is_incomplete_not_authority() {
    if isolated() {
        return;
    }
    let mut s = Setup::new(false, false);
    assert!(s.run().report().accepted);
    let db = rusqlite::Connection::open(s.root.join("ledger/events.sqlite3")).unwrap();
    db.execute_batch("DROP TRIGGER final_fixture_no_delete; DELETE FROM final_fixture_records WHERE kind='completion';").unwrap();
    assert!(s
        .ledger
        .fixture_final_decision(s.lease.contract())
        .unwrap()
        .is_none());
    assert_eq!(
        s.ledger.fixture_final_status(s.lease.contract()).unwrap(),
        Some(Status::Incomplete)
    );
}
#[test]
fn acceptance_gate_trusted_fixture_native_graph_alias_and_config_corruption_never_accept() {
    if isolated() {
        return;
    }
    for variant in [0, 1, 2] {
        let mut s = Setup::custom_artifacts(false, false, |c, store| {
            if variant == 0 {
                c["source"]["graph"] = json!(d("corrupt-graph"));
            }
            if variant == 1 {
                c["source"]["config"] = json!(d("corrupt-config"));
            }
            if variant == 2 {
                let f: Value = serde_json::from_str(include_str!(
                    "../../../python/neural_foundry/data/m1_fixture.json"
                ))
                .unwrap();
                let mut m: ModelManifest = serde_json::from_value(f["manifest"].clone()).unwrap();
                m.parameters.get_mut("tied").unwrap().storage_group = "untied".into();
                m.parameter_digest = tensor_set_digest(&m.parameters).unwrap();
                c["source"]["manifest"] = json!(store.put(bytes(&m).as_slice()).unwrap().digest);
                c["source"]["parameters"] = json!(m.parameter_digest);
            }
            let source = c["source"].clone();
            for b in c["baselines"].as_array_mut().unwrap() {
                if b["kind"] != "best_known" {
                    b["input"] = source.clone();
                }
            }
        });
        assert!(!s.run().report().accepted);
        assert!(s
            .ledger
            .frozen_final_panel(s.lease.contract())
            .unwrap()
            .is_none());
    }
}

#[test]
fn acceptance_gate_trusted_fixture_largest_admitted_panel_stays_within_storage_cap() {
    if isolated() {
        return;
    }
    let mut s = Setup::custom_artifacts(false, false, |c, store| {
        let b = &mut c["portfolio"]["bindings"][3];
        let mut items = Vec::new();
        for i in 0..28 {
            let identity = nf_core::study::TaskIdentity {
                item: d(&format!("bounded-panel-item-{i}")),
                group: d(&format!("bounded-panel-group-{i}")),
                family: d("task-family"),
            };
            let input = store
                .put(bytes(&vec![i as f64 + 100.0, 2.0]).as_slice())
                .unwrap()
                .digest;
            let label = store.put(bytes(&vec![0.0, 0.0]).as_slice()).unwrap().digest;
            items.push(Item {
                identity,
                input,
                label,
            });
        }
        b["splits"]["holdout"]["items"] =
            json!(items.iter().map(|t| &t.identity).collect::<Vec<_>>());
        b["splits"]["holdout"]["dataset"] = json!(publish_dataset(
            store,
            &Dataset {
                schema_version: 1,
                role: SplitRole::Holdout,
                items
            }
        )
        .unwrap());
        b["schedule"]["holdout_episodes"] = json!(28);
    });
    let before = s.store.used_bytes().unwrap();
    let result = s.run();
    assert!(
        result.report().accepted,
        "{}",
        String::from_utf8_lossy(result.bytes())
    );
    let retained = s.store.used_bytes().unwrap() - before
        + std::fs::metadata(s.root.join("ledger/events.sqlite3"))
            .unwrap()
            .len();
    assert!(
        retained < 4_194_304,
        "actual retained CAS + whole DB: {retained}"
    );
    assert_eq!(
        native_record(&s, "execution")["receipt"]["cells"]
            .as_array()
            .unwrap()
            .len(),
        32
    );
}
#[test]
fn acceptance_gate_trusted_fixture_durable_stop_after_intent_closes_native_reservation() {
    if isolated() {
        return;
    }
    let mut s = Setup::new(false, false);
    let root = s.root.join("ledger");
    let contract = s.lease.contract().clone();
    let (ready, wait) = std::sync::mpsc::channel();
    let watcher = std::thread::spawn(move || {
        let mut other = Ledger::open(&root, LedgerLimits::default()).unwrap();
        let db = rusqlite::Connection::open_with_flags(
            root.join("events.sqlite3"),
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        )
        .unwrap();
        ready.send(()).unwrap();
        for _ in 0..5000 {
            let started = db
                .query_row(
                    "SELECT count(*) FROM final_fixture_records WHERE kind='intent'",
                    [],
                    |r| r.get::<_, u64>(0),
                )
                .unwrap_or(0);
            if started > 0 {
                other
                    .request_stop(
                        &contract,
                        d("stop-final-fixture"),
                        nf_experiments::coordinator::supervision::StopReason::OwnerStop,
                    )
                    .unwrap();
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        panic!("final intent not observed");
    });
    wait.recv().unwrap();
    let result = s.ledger.run_fixture_final(
        &s.lease,
        &s.store,
        &s.edit,
        &s.noop,
        &s.smoke,
        &CancellationToken::default(),
    );
    watcher.join().unwrap();
    assert!(result.is_err() || !result.unwrap().report().accepted);
    assert_eq!(
        s.ledger.fixture_final_status(s.lease.contract()).unwrap(),
        Some(Status::Incomplete)
    );
    let budget = s.ledger.budget_snapshot(s.lease.contract()).unwrap();
    assert_eq!(budget.held.evaluations, 0);
    assert_eq!(budget.spent.evaluations, 86);
}
