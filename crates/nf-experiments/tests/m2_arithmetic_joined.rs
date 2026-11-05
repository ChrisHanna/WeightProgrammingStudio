//! Real coordinator/IR/catalogue/native-worker/OCI-scorer join. Never skip a
//! failed qualification and never count linear fixture evidence as this result.
#[path = "../../nf-core/tests/common/study_fixture.rs"]
#[allow(dead_code)]
mod fixture;
use fixture::{d, hash, refresh, standard};
use nf_checkpoint::store::{ArtifactStore, StoreLimits};
use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use nf_core::schemas::Validated;
use nf_core::study::{SplitRole, StudyContract, TaskIdentity};
use nf_experiments::coordinator::final_fixture::{arithmetic_final as joined, arithmetic_source};
use nf_experiments::coordinator::splits::{publish_dataset, Dataset, Item};
use nf_experiments::coordinator::{Ledger, LedgerLimits, Stage, Transition};
use nf_experiments::{arithmetic, intake, sandbox, validation};
use nf_graph::inventory::{verify_inventory, InventoryLimits};
use nf_surgeon::cancellation::CancellationToken;
use nf_surgeon::transaction::SmokeSuite;
use nf_weight_ir::compilation::{compile, CompilePolicy, SourceSnapshot};
use nf_weight_ir::program::parse_program;
use nf_weight_ir::search_space::{parse_search_space, resolve_space};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
fn bytes(v: &impl Serialize) -> Vec<u8> {
    CanonicalDocument::parse(&serde_json::to_vec(v).unwrap(), Limits::default())
        .unwrap()
        .bytes()
        .to_vec()
}
fn isolated(mode: u8) -> bool {
    if rustix::process::getrlimit(rustix::process::Resource::As).maximum == Some(512 << 20) {
        return false;
    }
    let root = std::env::temp_dir().join(format!(
        "nf-t043-joined-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap()
    ));
    let mut child = std::process::Command::new("/usr/bin/timeout")
        .args([
            "85",
            "/usr/bin/prlimit",
            "--as=536870912:536870912",
            "--core=0:0",
            "--",
        ])
        .arg(std::env::current_exe().unwrap())
        .args([
            "--exact",
            std::thread::current().name().unwrap(),
            "--test-threads=1",
            "--nocapture",
        ])
        .env("NF_T043_TEST_ROOT", &root)
        .spawn()
        .unwrap();
    let status = child.wait().unwrap();
    if mode == 2 {
        assert_eq!(status.code(), Some(73), "actual crash sentinel required");
        let connection = rusqlite::Connection::open_with_flags(
            root.join("ledger/events.sqlite3"),
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        )
        .unwrap();
        let id: String = connection
            .query_row("SELECT digest FROM contracts", [], |r| r.get(0))
            .unwrap();
        drop(connection);
        let id = id.parse().unwrap();
        let limits = LedgerLimits {
            max_database_bytes: 16 << 20,
            ..Default::default()
        };
        let mut ledger = Ledger::open(&root.join("ledger"), limits).unwrap();
        let runtime = sandbox::QualifiedDocker::qualify(true, &root.join("sandbox")).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2100));
        let lease = ledger.acquire(&id, "crash-recovery", 60000).unwrap();
        assert!(ledger.budget_snapshot(&id).unwrap().held.storage_bytes > 0);
        ledger.reconcile_arithmetic_final(&lease, &runtime).unwrap();
        let once = ledger.budget_snapshot(&id).unwrap();
        ledger.reconcile_arithmetic_final(&lease, &runtime).unwrap();
        let twice = ledger.budget_snapshot(&id).unwrap();
        assert_eq!(once.spent, twice.spent);
        assert_eq!(once.held, twice.held);
        assert_eq!(once.unknown_charged, twice.unknown_charged);
        assert_eq!(once.reservation_count, twice.reservation_count);
        let budget = ledger.budget_snapshot(&id).unwrap();
        assert_eq!(budget.held.storage_bytes, 0);
        assert!(budget.spent.storage_bytes > 0);
        let store = ArtifactStore::open_read_only(
            &root.join("store"),
            StoreLimits {
                total_bytes: 32 << 20,
                max_blob_bytes: 1 << 20,
                max_entries: 4096,
            },
        )
        .unwrap();
        assert!(ledger.qualify_arithmetic_final(&id, &store).is_err());
        drop(ledger);
        let reopened = Ledger::open(&root.join("ledger"), limits).unwrap();
        assert_eq!(reopened.budget_snapshot(&id).unwrap().held.storage_bytes, 0);
        std::fs::remove_dir_all(root).unwrap();
    } else {
        assert!(status.success(), "bounded joined child failed: {status}");
    }
    true
}
fn run(harmful: bool, no_gain: bool, missing: bool, mode: u8) {
    if isolated(mode) {
        return;
    }
    let root = std::path::PathBuf::from(
        std::env::var_os("NF_T043_TEST_ROOT").expect("isolated test root"),
    );
    std::fs::DirBuilder::new()
        .mode(0o700)
        .create(&root)
        .unwrap();
    let store = ArtifactStore::create(
        &root.join("store"),
        StoreLimits {
            total_bytes: 32 << 20,
            max_blob_bytes: 1 << 20,
            max_entries: 4096,
        },
    )
    .unwrap();
    let fixture = arithmetic::build(arithmetic::Mode::CorruptSum).unwrap();
    let source = &fixture.source;
    let original = source.checkpoint_bytes().to_vec();
    let model = arithmetic_source::identity(source).unwrap();
    let best = arithmetic::build(arithmetic::Mode::Reference).unwrap();
    let best_id = arithmetic_source::identity(&best.source).unwrap();
    for state in [source, &best.source] {
        store.put(bytes(&**state.manifest()).as_slice()).unwrap();
        store.put(state.checkpoint_bytes()).unwrap();
        for asset in state.assets().values() {
            store.put(asset.as_ref()).unwrap();
        }
    }
    let package = root.join("sum");
    std::fs::create_dir(&package).unwrap();
    let entry = json!({"file":"sum_gym.py","symbol":"evaluate"});
    let draft = json!({"schema_version":1,"name":"user.arithmetic.sum","version":"1",
        "usage_rights":"Apache-2.0 synthetic fixture","confidentiality":"development",
        "assets":{"sum_gym.py":{"kind":"code","expected":null},"LICENSE":{"kind":"license","expected":null},
            "lock.json":{"kind":"dependency_lock","expected":null},"development.json":{"kind":"data","expected":null},
            "manifest.json":{"kind":"manifest","expected":null}},"dependency_lock":"lock.json","licenses":["LICENSE"],
        "interface":{"kind":"single_shot","evaluate":entry,"oracle":{"file":"sum_gym.py","symbol":"oracle"},
            "score":{"file":"sum_gym.py","symbol":"score"},"input":{"type":"object","properties":{
                "x1":{"type":"integer","min":-8,"max":8},"x2":{"type":"integer","min":-8,"max":8}},
                "required":["x1","x2"]},"output":{"type":"number","min":-16.0,"max":16.0}},
        "dataset":{"kind":"frozen","files":["development.json"],"task_key":"task","group_key":"group"}});
    let development: Vec<_> = fixture
        .inventory
        .cases
        .iter()
        .filter(|v| v.split == SplitRole::Development)
        .map(|v| json!({"task":v.item,"group":v.group,"inputs":{"x1":v.x1,"x2":v.x2}}))
        .collect();
    std::fs::write(
        package.join("sum_gym.py"),
        include_bytes!("../../../python/neural_foundry/data/arithmetic_sum.py"),
    )
    .unwrap();
    std::fs::write(package.join("LICENSE"), b"Apache-2.0 synthetic fixture").unwrap();
    std::fs::write(package.join("manifest.json"), bytes(&draft)).unwrap();
    std::fs::write(
        package.join("development.json"),
        bytes(&json!({"schema_version":1,"items":development})),
    )
    .unwrap();
    std::fs::write(package.join("lock.json"),bytes(&json!({"schema_version":1,"image":sandbox::IMAGE.split('@').nth(1).unwrap(),"packages":{}}))).unwrap();
    let snapshot = intake::snapshot(
        &package,
        &store,
        intake::Draft::parse(&bytes(&draft)).unwrap(),
    )
    .unwrap();
    let sandbox_root = root.join("sandbox");
    std::fs::DirBuilder::new()
        .mode(0o700)
        .create(&sandbox_root)
        .unwrap();
    // Mandatory actual OCI capability. A failure is an external qualification
    // failure, never an ignored test or a fabricated positive witness.
    let runtime = sandbox::QualifiedDocker::qualify(true, &sandbox_root)
        .expect("actual OCI qualification required for joined T043");
    let validation_plan =
        validation::Plan::parse(&bytes(&json!({"schema_version":1,"snapshot":snapshot.id(),
        "truth_provenance":"controller independent integer addition","cases":[
            {"id":"a","input":{"x1":2,"x2":3},"expected":5.0,"wrong":[6.0],"termination":null},
            {"id":"b","input":{"x1":-8,"x2":7},"expected":-1.0,"wrong":[0.0],"termination":null}],
        "seeds":[17,42],"scoring":{"minimum":0.0,"maximum":1.0,"good_min":1.0,"bad_max":0.0},
        "limits":{"wall_ms":5000,"cpu_ms":5000,"output_bytes":65536}})))
        .unwrap();
    let validated = validation::validate(&snapshot, &store, validation_plan, &runtime).unwrap();
    let approval = validation::approve(
        &snapshot,
        &validated,
        &store,
        "isolated test owner exact snapshot".into(),
    )
    .unwrap();
    let mut c = standard();
    c["portfolio"]["profile"] = json!(if mode == 6 {
        "exploratory"
    } else {
        "standard_five_gym"
    });
    c["source"] = json!(model);
    c["best_known"] = json!(best_id);
    c["seeds"] = json!([17]);
    c["statistics"]["repeats"] = json!(1);
    c["runtime"] = json!(joined::runtime_profile().unwrap());
    c["limits"]["host_memory_bytes"] = json!(2_u64 << 30);
    c["limits"]["storage_bytes"] = json!(128_u64 << 20);
    let backend = c["runtime"]["backend"].clone();
    for b in c["baselines"].as_array_mut().unwrap() {
        b["input"] = if b["kind"] == "best_known" {
            json!(best_id)
        } else {
            json!(model)
        };
        b["backend"] = backend.clone();
    }
    let old = c["portfolio"]["gyms"].as_object().unwrap().clone();
    c["portfolio"]["gyms"] = json!({});
    for head in 0..5 {
        let mut b = c["portfolio"]["bindings"][head].clone();
        let mut g = old[b["gym"].as_str().unwrap()].clone();
        g["name"] = json!(if head == 0 {
            "user.arithmetic.sum".into()
        } else {
            format!("arithmetic.{}.v1", arithmetic::HEADS[head])
        });
        g["adapter"] = c["runtime"]["backend"].clone();
        for key in [
            "code",
            "scorer",
            "oracle",
            "scoring_protocol",
            "validation",
            "dependencies",
            "input_schema",
            "output_schema",
        ] {
            g[key] = json!(joined::profile_id());
        }
        let validation = if head == 0 {
            approval.id().clone()
        } else {
            joined::profile_id()
        };
        g["validation"] = json!(validation);
        b["validation"] = json!(validation);
        if head == 0 {
            g["code"] = json!(snapshot.id());
            g["approval"] = json!(approval.id());
        }
        g["metrics"] = json!({"accuracy":{"domain":"quality","unit":"fraction","direction":"higher_is_better","lower":0.0,"upper":1.0}});
        b["roles"] = if head == 0 {
            json!(["target"])
        } else if head == 4 {
            json!(["retention", "independent_verification"])
        } else {
            json!(["retention"])
        };
        b["metrics"] = json!([{"metric":"accuracy","required":true,"priority":1.0,"comparator":"original",
            "threshold":{"floor":null,"ceiling":null,"max_regression":if head==0 {Value::Null}else{json!(0.0)},
                "min_gain":if head==0 {json!(0.01)}else{Value::Null}},"statistic":"exact","measurement":null}]);
        if head == 0 && mode != 6 {
            for (name, domain, unit, scope, ceiling) in [
                (
                    "latency",
                    "latency",
                    "milliseconds",
                    "forward_latency",
                    json!(if mode == 7 { 0.0 } else { 1000.0 }),
                ),
                (
                    "memory",
                    "memory",
                    "bytes",
                    "process_rss",
                    json!(if mode == 5 { 1.0 } else { 2_147_483_648.0 }),
                ),
            ] {
                g["metrics"][name] = json!({"domain":domain,"unit":unit,
                    "direction":"lower_is_better","lower":null,"upper":null});
                b["metrics"].as_array_mut().unwrap().push(json!({
                    "metric":name,"required":true,"priority":1.0,"comparator":"original",
                    "threshold":{"floor":null,"ceiling":ceiling,"max_regression":null,"min_gain":null},
                    "statistic":"maximum","measurement":{"scope":scope,"statistic":"maximum",
                        "warmup":1,"repetitions":3,"order":"paired_interleaved",
                        "synchronization":"cpu_monotonic",
                        "timer":{"id":"nf.final.monotonic","version":"1","config":joined::profile_id()}}
                }));
            }
        }
        for role in [
            SplitRole::Development,
            SplitRole::Selection,
            SplitRole::Holdout,
        ] {
            let name = match role {
                SplitRole::Development => "development",
                SplitRole::Selection => "selection",
                _ => "holdout",
            };
            let mut items = Vec::new();
            for case in fixture.inventory.cases.iter().filter(|v| v.split == role) {
                let x = [case.x1 as f64, case.x2 as f64];
                let y = [
                    x[0] + x[1],
                    x[0] - x[1],
                    x[0].max(x[1]),
                    x[0].min(x[1]),
                    (x[0] - x[1]).abs(),
                ];
                items.push(Item {
                    identity: TaskIdentity {
                        item: case.item.clone(),
                        group: case.group.clone(),
                        family: case.family.clone(),
                    },
                    input: store.put(bytes(&x).as_slice()).unwrap().digest,
                    label: store.put(bytes(&y[head]).as_slice()).unwrap().digest,
                });
            }
            b["splits"][name]["dataset"] = json!(publish_dataset(
                &store,
                &Dataset {
                    schema_version: 1,
                    role,
                    items: items.clone()
                }
            )
            .unwrap());
            b["splits"][name]["items"] =
                json!(items.iter().map(|v| &v.identity).collect::<Vec<_>>());
            b["schedule"][format!("{name}_episodes")] = json!(items.len());
        }
        let id = hash(&g);
        c["portfolio"]["gyms"][id.as_str()] = g;
        b["gym"] = json!(id);
        c["portfolio"]["bindings"][head] = b;
    }
    let mut f: Value = serde_json::from_str(include_str!(
        "../../../python/neural_foundry/data/m1_fixture.json"
    ))
    .unwrap();
    let smoke: Validated<SmokeSuite>=Validated::parse(&bytes(&json!({"schema_version":1,"adapter":"nf.arithmetic_relu.v1",
        "probes":[{"input":[8,2],"expected":null,"absolute_tolerance":0.0,"max_abs_output":1000.0}]}))).unwrap();
    let binding = source.binding(model.graph.clone());
    f["space"]["base_model"] = json!(binding);
    f["space"]["sites"] = json!(["hidden.weight", "heads.weight"]
        .map(|name| json!({"selector":{"type":"tensor","name":name},
        "role":"parameter","regions":{"type":"whole_tensor"},"allowed_axes":[0,1]})));
    f["space"]["constraints"]["max_changed_parameters"] = json!(2);
    let space = parse_search_space(&bytes(&f["space"])).unwrap();
    let p = &mut f["program"];
    p["base_model"] = json!(binding);
    p["objective"]["search_space"] = json!(space.digest().unwrap());
    p["verification"]["smoke_suite"] = json!(smoke.digest().unwrap());
    let mut regions = vec![json!({"tensor":"heads.weight","start":[0,0],"shape":[1,1]})];
    if harmful {
        regions.push(json!({"tensor":"heads.weight","start":[1,0],"shape":[1,1]}));
    }
    p["operations"] = json!(regions
        .iter()
        .enumerate()
        .map(
            |(i, r)| json!({"id":format!("op{i}"),"depends_on":[],"operation":{
        "type":"scale","target":r,"factor":if i>0 {0.0}else if no_gain {1.0}else{2.0}}})
        )
        .collect::<Vec<_>>());
    p["preconditions"]=json!(source.manifest().parameters.iter().map(|(name,spec)|json!({"tensor":name,"role":"parameter",
        "shape":spec.shape,"dtype":"f64","before_digest":spec.digest,"storage_group":spec.storage_group})).collect::<Vec<_>>());
    p["effects"]["reads"] = json!(regions);
    p["effects"]["writes"] = json!(regions);
    p["constraints"]["max_changed_parameters"] = json!(regions.len());
    p["lineage"]["method"]["changed_parameters"] = json!(regions.len());
    c["contract_key"] = p["objective"]["contract"].clone();
    c["objective"]["id"] = p["objective"]["objective"].clone();
    let objective = c["objective"]["id"].clone();
    c["objective"]["primary_metric"] = json!("accuracy");
    for b in c["portfolio"]["bindings"].as_array_mut().unwrap() {
        b["objective"] = objective.clone();
    }
    c["intervention"]["scope"] = json!(hash(&p["effects"]));
    c["intervention"]["transformation_space"] = json!(space.digest().unwrap());
    c["intervention"]["method_protocol"] = p["objective"]["experiment_protocol"].clone();
    c["intervention"]["allowed_engines"] =
        json!([{"id":"nf.native_cpu","version":"1","config":hash(&f["policy"]["backend"])}]);
    refresh(&mut c);
    f["program"]["verification"]["portfolio"] = c["portfolio_digest"].clone();
    f["policy"]["source"] = json!(binding);
    f["policy"]["search_space"] = json!(space.digest().unwrap());
    f["policy"]["verification"] = f["program"]["verification"].clone();
    let policy: Validated<CompilePolicy> = Validated::parse(&bytes(&f["policy"])).unwrap();
    let observations = source.observations();
    let inventory =
        verify_inventory(source.manifest(), &observations, InventoryLimits::default()).unwrap();
    let resolved = resolve_space(&space, &inventory).unwrap();
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
    let c = Validated::<StudyContract>::parse(&bytes(&c)).unwrap();
    let mut ledger = Ledger::open(
        &root.join("ledger"),
        LedgerLimits {
            max_database_bytes: 16 << 20,
            ..Default::default()
        },
    )
    .unwrap();
    let id = ledger.freeze(&c).unwrap();
    let mut lease = ledger.acquire(&id, "joined-test", 300000).unwrap();
    for b in c
        .portfolio
        .bindings
        .iter()
        .take(if missing { 4 } else { 5 })
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
                    protocol: d("fixture-test"),
                    evidence: d("not-scientific-evidence"),
                    message: "joined synthetic test".into(),
                },
            )
            .unwrap();
        previous = next;
    }
    use nf_experiments::coordinator::final_fixture::inject_fixture_fault;
    use nf_experiments::coordinator::splits::Recipient;
    assert!(ledger
        .grant_split_worker(
            &lease,
            &c.portfolio.bindings[0].id,
            Recipient::FinalEvaluator,
            &runtime,
            1000
        )
        .is_err());
    assert!(ledger
        .grant_split_worker(
            &lease,
            &c.portfolio.bindings[0].id,
            Recipient::Proposer,
            &runtime,
            1000
        )
        .is_err());
    let cancel = CancellationToken::default();
    if mode == 1 {
        let stop = cancel.clone();
        inject_fixture_fault("arithmetic_after_native_before_score", move || {
            stop.request();
            Ok(())
        });
    }
    if mode == 2 {
        lease = ledger.renew(&lease, 2000).unwrap();
        inject_fixture_fault("arithmetic_after_intent", || std::process::exit(73));
    }
    if mode == 3 {
        inject_fixture_fault("arithmetic_after_native_before_score", || {
            Err(nf_core::error::CoreError::new(
                nf_core::error::ErrorCode::TransactionAborted,
                "/joined_test",
                "interruption after actual model output before scorer",
            ))
        });
    }
    let cas_before = store.used_bytes().unwrap();
    let run_started = std::time::Instant::now();
    let result = ledger.run_arithmetic_final(
        &lease,
        &store,
        &edit,
        &noop,
        &smoke,
        joined::SumScorer {
            snapshot: &snapshot,
            approval: &approval,
            runtime: &runtime,
        },
        &cancel,
    );
    let run_wall_ms = run_started.elapsed().as_millis();
    let cas_delta = store.used_bytes().unwrap().saturating_sub(cas_before);
    assert_eq!(source.checkpoint_bytes(), original);
    if mode == 6 {
        let error = result
            .err()
            .expect("missing performance ceilings must be refused");
        assert_eq!(error.code, nf_core::error::ErrorCode::UnsupportedCapability);
        assert!(ledger.frozen_final_panel(&id).unwrap().is_none());
        let connection = rusqlite::Connection::open(root.join("ledger/events.sqlite3")).unwrap();
        let openings: u64 = connection
            .query_row(
                "SELECT count(*) FROM holdout_openings WHERE contract=?1",
                [id.as_str()],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            openings, 0,
            "missing ceilings must be refused before holdout disclosure"
        );
    } else if missing {
        assert!(result.is_err());
        assert!(ledger.frozen_final_panel(&id).unwrap().is_none());
    } else if mode == 1 || mode == 3 {
        assert!(result.is_err());
        assert!(ledger.frozen_final_panel(&id).unwrap().is_some());
        assert!(ledger.qualify_arithmetic_final(&id, &store).is_err());
        assert!(ledger.budget_snapshot(&id).unwrap().held.storage_bytes > 0);
    } else {
        let result = result.unwrap();
        assert_eq!(
            result.decision().report().accepted,
            !harmful && !no_gain && mode != 5 && mode != 7,
            "{}",
            String::from_utf8_lossy(result.decision().bytes())
        );
        assert!(!result.decision().report().promotion_authorized);
        if harmful || no_gain || mode == 5 || mode == 7 {
            assert!(result
                .decision()
                .report()
                .checks
                .iter()
                .any(|v| v.name == "frozen_metric"
                    && v.status == nf_experiments::coordinator::acceptance::Status::Failed));
        } else {
            assert_eq!(
                result
                    .decision()
                    .report()
                    .checks
                    .iter()
                    .filter(|v| v.name == "frozen_metric"
                        && v.status == nf_experiments::coordinator::acceptance::Status::Passed)
                    .count(),
                7
            );
        }
        if mode == 5 || mode == 7 {
            assert!(result
                .decision()
                .report()
                .checks
                .iter()
                .any(|v| v.metric.as_deref()
                    == Some(if mode == 5 { "memory" } else { "latency" })
                    && v.status == nf_experiments::coordinator::acceptance::Status::Failed));
        }
        if mode == 10 {
            let connection =
                rusqlite::Connection::open(root.join("ledger/events.sqlite3")).unwrap();
            let body:Vec<u8>=connection.query_row("SELECT body FROM final_fixture_records WHERE contract=?1 AND kind='arithmetic_execution'",[id.as_str()],|r|r.get(0)).unwrap();
            let execution: Value = serde_json::from_slice(&body).unwrap();
            let data_id: ContentDigest =
                serde_json::from_value(execution["data_digest"].clone()).unwrap();
            let path = root.join("store/blobs").join(&data_id.as_str()[7..]);
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
            let mut corrupted = std::fs::read(&path).unwrap();
            corrupted[0] ^= 1;
            std::fs::write(&path, corrupted).unwrap();
            drop(connection);
            drop(result);
            drop(ledger);
            let reopened = Ledger::open(
                &root.join("ledger"),
                LedgerLimits {
                    max_database_bytes: 16 << 20,
                    ..Default::default()
                },
            )
            .unwrap();
            assert!(
                reopened.qualify_arithmetic_final(&id, &store).is_err(),
                "corrupted observation artifact must be rejected after reopen"
            );
            drop(reopened);
            std::fs::remove_dir_all(root).unwrap();
            return;
        }
        if mode == 8 || mode == 9 {
            let connection =
                rusqlite::Connection::open(root.join("ledger/events.sqlite3")).unwrap();
            connection
                .execute_batch("DROP TRIGGER final_fixture_no_update")
                .unwrap();
            let body:Vec<u8>=connection.query_row("SELECT body FROM final_fixture_records WHERE contract=?1 AND kind='arithmetic_decision'",[id.as_str()],|r|r.get(0)).unwrap();
            let mut decision: Value = serde_json::from_slice(&body).unwrap();
            if mode == 8 {
                decision["checks"]
                    .as_array_mut()
                    .unwrap()
                    .retain(|check| check["name"] != "resource_measurements");
            } else {
                decision["event_head"] = json!(d("foreign-historical-event"));
            }
            let forged = bytes(&decision);
            connection.execute("UPDATE final_fixture_records SET body=?1,digest=?2 WHERE contract=?3 AND kind='arithmetic_decision'",
                rusqlite::params![forged,ContentDigest::of_bytes(&forged).as_str(),id.as_str()]).unwrap();
            assert!(
                ledger.qualify_arithmetic_final(&id, &store).is_err(),
                "rehashed decision missing a required check or event anchor must be rejected"
            );
            drop(connection);
            drop(ledger);
            std::fs::remove_dir_all(root).unwrap();
            return;
        }
        if !harmful && !no_gain && mode == 0 {
            let connection =
                rusqlite::Connection::open(root.join("ledger/events.sqlite3")).unwrap();
            let body:Vec<u8>=connection.query_row("SELECT body FROM final_fixture_records WHERE contract=?1 AND kind='arithmetic_execution'",[id.as_str()],|r|r.get(0)).unwrap();
            let execution: Value = serde_json::from_slice(&body).unwrap();
            let results = execution["analysis"]["results"].as_array().unwrap();
            let metric = |name: &str| {
                results.iter().find(|r| r["metric"] == name).unwrap()["estimate"]["candidate"]
                    .as_f64()
                    .unwrap()
            };
            let data_id: ContentDigest =
                serde_json::from_value(execution["data_digest"].clone()).unwrap();
            let observation_bytes = store.read_bounded(&data_id, 1_048_576).unwrap().len();
            println!("T038_MEASUREMENTS candidate_mean_group_forward_max_ms={} mean_group_controller_highwater_rss_bytes={} run_wall_ms={} cas_delta_bytes={} observation_bytes={} execution_record_bytes={} scope=per_task_max_then_equal_group_mean,co_resident_controller_rss,cas_only_excludes_sandbox_and_db",metric("latency"),metric("memory"),run_wall_ms,cas_delta,observation_bytes,body.len());
            assert!(metric("memory") > 0.0);
        }
        assert!(ledger
            .grant_split_worker(
                &lease,
                &c.portfolio.bindings[0].id,
                Recipient::Proposer,
                &runtime,
                1000
            )
            .is_err());
        let decision = result.decision().id().clone();
        let export = result.export_root().clone();
        drop(result);
        drop(ledger);
        let reopened = Ledger::open(
            &root.join("ledger"),
            LedgerLimits {
                max_database_bytes: 16 << 20,
                ..Default::default()
            },
        )
        .unwrap();
        let qualified = reopened.qualify_arithmetic_final(&id, &store).unwrap();
        assert_eq!(qualified.decision().id(), &decision);
        assert_eq!(qualified.export_root(), &export);
        if !harmful && !no_gain && mode == 0 {
            reopened
                .qualify_fixture_promotion(
                    &lease,
                    &store,
                    nf_experiments::coordinator::promotion::PromotionScope::OperationalFixture,
                )
                .unwrap();
        }
    }
    std::fs::remove_dir_all(root).unwrap();
}
#[test]
fn m2_arithmetic_joined_positive_five_gym() {
    run(false, false, false, 0);
}
#[test]
fn m2_arithmetic_joined_harmful_retention() {
    run(true, false, false, 0);
}
#[test]
fn m2_arithmetic_joined_noop_has_no_gain() {
    run(false, true, false, 0);
}
#[test]
fn m2_arithmetic_joined_missing_gym() {
    run(false, false, true, 0);
}
#[test]
fn m2_arithmetic_joined_cancel_after_actual_model() {
    run(false, false, false, 1);
}
#[test]
fn m2_arithmetic_joined_crash_reopen_recovery() {
    run(false, false, false, 2);
}
#[test]
fn m2_arithmetic_joined_missing_score_is_incomplete() {
    run(false, false, false, 3);
}
#[test]
fn m2_arithmetic_joined_memory_ceiling_failure() {
    run(false, false, false, 5);
}
#[test]
fn m2_arithmetic_joined_missing_resource_ceilings_prevent_holdout() {
    run(false, false, false, 6);
}
#[test]
fn m2_arithmetic_joined_latency_ceiling_failure() {
    run(false, false, false, 7);
}
#[test]
fn m2_arithmetic_joined_forged_decision_missing_resource_check_is_rejected() {
    run(false, false, false, 8);
}
#[test]
fn m2_arithmetic_joined_forged_decision_event_anchor_is_rejected() {
    run(false, false, false, 9);
}
#[test]
fn m2_arithmetic_joined_corrupted_observation_artifact_is_rejected() {
    run(false, false, false, 10);
}
