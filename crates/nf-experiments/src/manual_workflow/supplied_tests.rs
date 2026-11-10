//! Positive supplied model program tests, not the native metadata fixture.
use super::*;
use crate::coordinator::final_fixture::{graph_bytes, profile_id, runtime_profile};
use crate::coordinator::splits::{publish_dataset, Dataset, Item};
use nf_checkpoint::export::{publish_export, restore_export, ExportLimits};
use nf_core::schemas::ModelManifest;
use nf_core::study::SplitRole;
use nf_surgeon::transaction::{SourceState, StateLoadLimits};

fn isolated() -> bool {
    if rustix::process::getrlimit(rustix::process::Resource::As)
        .maximum
        .is_some_and(|n| n <= 1_000_000_000)
    {
        return false;
    }
    let status = std::process::Command::new("/usr/bin/prlimit")
        .args([
            "--as=1000000000:1000000000",
            "--cpu=60:60",
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
        .status()
        .unwrap();
    assert!(status.success(), "isolated supplied test failed: {status}");
    true
}
fn setup() -> (Setup, Value, Value) {
    setup_with_factor(0.5)
}
fn setup_with_factor(factor: f64) -> (Setup, Value, Value) {
    let mut s = Setup::new();
    let store = ArtifactStore::create(&s.request.store, store_limits()).unwrap();
    let mut f: Value = serde_json::from_str(include_str!(
        "../../../../python/neural_foundry/data/m1_fixture.json"
    ))
    .unwrap();
    let hex = f["checkpoint_hex"].as_str().unwrap();
    let bytes: Vec<u8> = (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect();
    let manifest =
        Validated::<ModelManifest>::parse(canonical(&f["manifest"]).unwrap().bytes()).unwrap();
    let state = SourceState::load(
        manifest.clone(),
        Arc::from(bytes.clone()),
        BTreeMap::from([("config".into(), Arc::from(b"{}".as_slice()))]),
        StateLoadLimits::default(),
    )
    .unwrap();
    let export = publish_export(
        &store,
        &manifest,
        Arc::from(bytes),
        state.assets().clone(),
        ExportLimits::default(),
    )
    .unwrap();
    store.put(graph_bytes()).unwrap();
    f["program"]["base_model"]["architecture_graph"] =
        json!(ContentDigest::of_bytes(graph_bytes()));
    f["program"]["operations"][0]["operation"]["factor"] = json!(factor);
    f["space"]["base_model"] = f["program"]["base_model"].clone();
    f["space"]["constraints"]["quality_limits"][0]["binding"] = json!(fixture::d("binding1"));
    f["space"]["constraints"]["quality_limits"][0]["metric"] = json!("quality");
    f["program"]["objective"]["search_space"] = json!(fixture::hash(&f["space"]));
    let model = json!({"checkpoint":f["program"]["base_model"]["checkpoint"],
        "manifest":state.manifest_digest(),"parameters":manifest.parameter_digest,
        "buffers":tensor_set_digest(&manifest.buffers).unwrap(),"graph":ContentDigest::of_bytes(graph_bytes()),
        "config":manifest.architecture.config,"tokenizer":null,"system_state":null,"runtime_artifact":null});
    let mut c = fixture::standard();
    c["study_family"] = json!(fixture::d("supplied-ir-distinct-family"));
    c["source"] = model.clone();
    c["best_known"] = model.clone();
    c["contract_key"] = f["program"]["objective"]["contract"].clone();
    c["objective"]["id"] = f["program"]["objective"]["objective"].clone();
    c["runtime"] = json!(runtime_profile().unwrap());
    c["limits"]["host_memory_bytes"] = json!(1_000_000_000_u64);
    c["statistics"]["repeats"] = json!(1);
    c["seeds"] = json!([17]);
    for baseline in c["baselines"].as_array_mut().unwrap() {
        baseline["input"] = model.clone();
        baseline["backend"] = json!({"id":"fixture.cpu","version":"1","config":profile_id()});
    }
    let gyms = c["portfolio"]["gyms"].as_object().unwrap().clone();
    c["portfolio"]["gyms"] = json!({});
    let objective = c["objective"]["id"].clone();
    for i in 0..5 {
        let b = &mut c["portfolio"]["bindings"][i];
        b["objective"] = objective.clone();
        for metric in b["metrics"].as_array_mut().unwrap() {
            if !metric["measurement"].is_null() {
                metric["measurement"]["timer"] =
                    json!({"id":"nf.final.monotonic","version":"1","config":profile_id()});
            }
            if metric["metric"] == "memory" {
                metric["threshold"]["ceiling"] = json!(1_000_000_000.0);
            }
        }
        for (j, role) in ["development", "selection", "holdout"]
            .into_iter()
            .enumerate()
        {
            let input = if i == 1 && j == 2 {
                vec![0.0, 0.0]
            } else {
                vec![(i * 10 + j + 1) as f64, 2.0]
            };
            let label: Vec<f64> = vec![
                (input[0] + 2.0 * input[1]) * factor,
                (3.0 * input[0] + 4.0 * input[1]) * factor,
            ];
            let dataset = Dataset {
                schema_version: 1,
                role: match j {
                    0 => SplitRole::Development,
                    1 => SplitRole::Selection,
                    _ => SplitRole::Holdout,
                },
                items: vec![Item {
                    identity: serde_json::from_value(b["splits"][role]["items"][0].clone())
                        .unwrap(),
                    input: store
                        .put(canonical(&input).unwrap().bytes())
                        .unwrap()
                        .digest,
                    label: store
                        .put(canonical(&label).unwrap().bytes())
                        .unwrap()
                        .digest,
                }],
            };
            b["splits"][role]["dataset"] = json!(publish_dataset(&store, &dataset).unwrap());
        }
        let mut gym = gyms[b["gym"].as_str().unwrap()].clone();
        gym["adapter"] = json!({"id":"nf.final.linear_exact","version":"1","config":profile_id()});
        for field in [
            "code",
            "scorer",
            "oracle",
            "scoring_protocol",
            "validation",
            "dependencies",
            "input_schema",
            "output_schema",
        ] {
            gym[field] = json!(profile_id());
        }
        let id = fixture::hash(&gym);
        b["gym"] = json!(id);
        b["validation"] = json!(profile_id());
        c["portfolio"]["gyms"][id.as_str()] = gym;
    }
    fixture::refresh(&mut c);
    f["program"]["verification"]["portfolio"] = c["portfolio_digest"].clone();
    f["program"]["verification"]["gyms"] =
        json!(c["portfolio"]["bindings"].as_array().unwrap().iter().map(|b| {
        json!({"binding":b["id"],"gym":b["gym"],"validation":b["validation"],"scorer":profile_id(),
            "split":fixture::hash(&b["splits"]["selection"]),"role":b["roles"][0]})
    }).collect::<Vec<_>>());
    f["policy"]["source"] = f["program"]["base_model"].clone();
    f["policy"]["search_space"] = f["program"]["objective"]["search_space"].clone();
    f["policy"]["verification"] = f["program"]["verification"].clone();
    c["intervention"]["method_protocol"] = f["program"]["objective"]["experiment_protocol"].clone();
    c["intervention"]["transformation_space"] = f["program"]["objective"]["search_space"].clone();
    c["intervention"]["scope"] = json!(fixture::hash(&f["program"]["effects"]));
    c["intervention"]["allowed_engines"] = json!([{"id":"nf.native_cpu","version":"1","config":fixture::hash(&f["policy"]["backend"])}]);
    let mut noop = f["program"].clone();
    noop["operations"] = json!([]);
    noop["effects"]["reads"] = json!([]);
    noop["effects"]["writes"] = json!([]);
    noop["lineage"]["method"]["target_update"] = json!("no_change");
    noop["lineage"]["method"]["changed_parameters"] = json!(0);
    noop["constraints"]["max_changed_parameters"] = json!(0);
    let request = json!({"schema_version":2,"store":s.request.store,"source_export":export,"contract":c,
        "program":f["program"],"control_program":noop,"search_space":f["space"],"policy":f["policy"],"smoke_suite":f["smoke_suite"]});
    let planned = dispatch("plan", canonical(&request).unwrap().bytes()).unwrap();
    let frozen = dispatch(
        "freeze",
        canonical(&json!({"plan":request,"ledger":s.request.ledger,
        "expected_plan_digest":planned["plan_digest"]}))
        .unwrap()
        .bytes(),
    )
    .unwrap();
    s.request.contract_digest = serde_json::from_value(frozen["contract_digest"].clone()).unwrap();
    s.contract = frozen_contract(&s.request).unwrap();
    (s, request, planned)
}
fn authorize(s: &mut Setup, planned: &Value) -> Grant {
    authorize_until(s, planned, now().unwrap() + 60)
}
fn authorize_until(s: &mut Setup, planned: &Value, expires: u64) -> Grant {
    let image = std::env::current_exe().unwrap();
    let issue = json!({"request":s.request,"operation":supplied::OPERATION,"supplied":planned["operator_binding"],
        "worker":image,"worker_digest":ContentDigest::of_bytes(&std::fs::read(&image).unwrap()),
        "expires_unix_seconds":expires});
    let issued = operator_with_authority(
        "authorize",
        canonical(&issue).unwrap().bytes(),
        &s.authority,
    )
    .unwrap();
    s.request.authorization =
        Some(serde_json::from_value(issued["authorization"].clone()).unwrap());
    decode(canonical(&issued["grant"]).unwrap().bytes()).unwrap()
}
#[test]
fn supplied_ir_child_entry() {
    if std::env::var_os(SUPPLIED_IR_CHILD_ENV).is_none() {
        return;
    }
    // Test-only IPC adapter: no test authority selector exists in production.
    let mut bytes = Vec::new();
    std::io::stdin().read_to_end(&mut bytes).unwrap();
    if bytes.is_empty() {
        return;
    }
    assert!(std::fs::metadata("/proc/self/fd/0").unwrap().is_file());
    let seals = unsafe { libc::fcntl(0, libc::F_GET_SEALS) };
    assert_eq!(
        seals & libc::F_SEAL_WRITE,
        libc::F_SEAL_WRITE,
        "request stdin must be immutable and finite"
    );
    let request: ExistingRequest = decode(&bytes).unwrap();
    let root = request.ledger.parent().unwrap();
    let authority = Authority {
        directory: File::open(root.join("authority")).unwrap(),
        uid: unsafe { libc::getuid() },
    };
    if root.join("test-revoke-on-admission").exists() {
        let marker = root.join("authority").join(format!(
            "{}.revoked",
            request.authorization.as_ref().unwrap().as_str()
        ));
        crate::coordinator::final_fixture::inject_fixture_fault(
            "after_admission_commit",
            move || {
                std::fs::write(&marker, b"").unwrap();
                std::thread::sleep(std::time::Duration::from_millis(100));
                Ok(())
            },
        );
    }
    if root.join("test-revoke-before-publication").exists() {
        let marker = root.join("authority").join(format!(
            "{}.revoked",
            request.authorization.as_ref().unwrap().as_str()
        ));
        crate::coordinator::final_fixture::inject_fixture_fault(
            "completion_before_publication_fence",
            move || {
                std::fs::write(&marker, b"").unwrap();
                std::thread::sleep(std::time::Duration::from_millis(100));
                Ok(())
            },
        );
    }
    if root.join("test-expire-on-admission").exists() {
        let expires = authority
            .verify(&request, &frozen_contract(&request).unwrap())
            .unwrap()
            .0
            .expires_unix_seconds;
        crate::coordinator::final_fixture::inject_fixture_fault(
            "after_admission_commit",
            move || {
                while now().unwrap() < expires {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
                Ok(())
            },
        );
    }
    if root.join("test-crash-on-admission").exists() {
        crate::coordinator::final_fixture::inject_fixture_fault("after_admission_commit", || {
            std::process::exit(73)
        });
    }
    let result = execute_supplied_with(&request, &authority).unwrap();
    println!("NF_SUPPLIED_RESULT={}", json!({"ok":true,"result":result}));
}
#[test]
fn supplied_ir_real_program_controls_export_reload_and_retained_source() {
    if isolated() {
        return;
    }
    let (mut s, request, planned) = setup_with_factor(0.375);
    let original = canonical(&request["program"]).unwrap();
    let grant = authorize(&mut s, &planned);
    let database = s.request.ledger.join("events.sqlite3");
    let checkpoint = || {
        let db = rusqlite::Connection::open(&database).unwrap();
        db.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)").unwrap();
    };
    checkpoint();
    let frozen_bytes = std::fs::read(&database).unwrap();
    let frozen_identity = identity(&database).unwrap();
    let result = supplied::launch_test(&s.request, &grant, &s.contract).unwrap();
    assert_eq!(result["supplied_ir_executed"], true);
    assert_eq!(result["program"], json!(original.digest()));
    assert_eq!(result["promotion"], false);
    assert_eq!(result["strict_publication"], false);
    let db = rusqlite::Connection::open(s.request.ledger.join("events.sqlite3")).unwrap();
    let body: Vec<u8> = db
        .query_row(
            "SELECT body FROM final_fixture_records WHERE contract=?1 AND kind='execution'",
            [s.request.contract_digest.as_str()],
            |r| r.get(0),
        )
        .unwrap();
    let evidence: Value = decode(&body).unwrap();
    assert_eq!(evidence["status"], "complete", "{evidence}");
    let body: Vec<u8> = db
        .query_row(
            "SELECT body FROM final_fixture_records WHERE contract=?1 AND kind='dispatch'",
            [s.request.contract_digest.as_str()],
            |r| r.get(0),
        )
        .unwrap();
    let dispatch: Value = decode(&body).unwrap();
    assert_eq!(dispatch["edit"]["changed_parameters"], 4);
    assert_eq!(dispatch["noop"]["changed_parameters"], 0);
    assert_ne!(
        dispatch["candidate"]["parameters"],
        request["contract"]["source"]["parameters"]
    );
    assert_eq!(dispatch["original"], request["contract"]["source"]);
    let store = ArtifactStore::open_read_only(&s.request.store, store_limits()).unwrap();
    assert_eq!(
        store.read_bounded(&original.digest(), 65536).unwrap(),
        original.bytes()
    );
    let candidate_export: ContentDigest =
        serde_json::from_value(dispatch["candidate_export"]["export_root"].clone()).unwrap();
    let candidate = restore_export(&store, &candidate_export, ExportLimits::default()).unwrap();
    assert_eq!(
        candidate.tensor_bytes("w").unwrap(),
        [0.375f64, 0.75, 1.125, 1.5]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>()
    );
    let source = restore_export(
        &store,
        &grant.supplied.as_ref().unwrap().source_export,
        ExportLimits::default(),
    )
    .unwrap();
    assert_eq!(
        source.tensor_bytes("w").unwrap(),
        [1.0f64, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>()
    );
    let budgets = s
        .request
        .open()
        .unwrap()
        .budget_snapshot(&s.request.contract_digest)
        .unwrap();
    assert_eq!(budgets.spent.attempts, 1);
    assert!(budgets.spent.evaluations > 0);
    assert_eq!(result["decision"]["accepted"], true, "{result}");
    assert!(
        execute_supplied_with(&s.request, &s.authority).is_err(),
        "no rerun after completion"
    );
    drop(db);
    checkpoint();
    std::fs::write(&database, frozen_bytes).unwrap();
    assert!(identity(&database).unwrap() == frozen_identity);
    assert_eq!(
        s.request
            .open()
            .unwrap()
            .state(&s.request.contract_digest)
            .unwrap()
            .stage,
        crate::coordinator::Stage::Frozen
    );
    s.authority.verify(&s.request, &s.contract).unwrap();
    assert!(
        supplied::launch_test(&s.request, &grant, &s.contract).is_err(),
        "fresh child must reject grant replay even after same-inode rollback to Frozen"
    );
}
#[test]
fn supplied_ir_grants_reject_changed_bindings_and_preserve_frozen_program() {
    if isolated() {
        return;
    }
    let (mut s, request, planned) = setup();
    let mut grant = authorize(&mut s, &planned);
    grant.worker_digest = fixture::d("unapproved executable bytes");
    assert!(supplied::launch_test(&s.request, &grant, &s.contract).is_err());
    let approved = s.request.authorization.clone();
    s.request.authorization = Some(fixture::d("forged supplied grant"));
    assert!(execute_supplied_with(&s.request, &s.authority).is_err());
    s.request.authorization = approved.clone();
    grant.permissions = fixture::d("different permissions");
    s.install(&grant);
    assert!(execute_supplied_with(&s.request, &s.authority).is_err());
    grant.permissions = canonical(&s.contract.permissions).unwrap().digest();
    grant.expires_unix_seconds = now().unwrap() - 1;
    s.install(&grant);
    assert!(execute_supplied_with(&s.request, &s.authority).is_err());
    s.request.authorization = approved;
    let binding = grant.supplied.as_mut().unwrap();
    binding.source_export = fixture::d("different source");
    assert!(supplied::verify_binding(&s.request, &s.contract, binding).is_err());
    binding.source_export = serde_json::from_value(request["source_export"].clone()).unwrap();
    binding.manifest = fixture::d("different manifest");
    assert!(supplied::verify_binding(&s.request, &s.contract, binding).is_err());
    assert_eq!(
        frozen_contract(&s.request)
            .unwrap()
            .intervention
            .method_protocol,
        serde_json::from_value::<ContentDigest>(
            request["program"]["objective"]["experiment_protocol"].clone()
        )
        .unwrap()
    );
    assert_ne!(
        request["contract"]["intervention"]["method_protocol"],
        planned["manifest_digest"]
    );
    let id = s.request.authorization.as_ref().unwrap();
    s.authority
        .file(&format!("{}.revoked", id.as_str()), true)
        .unwrap();
    assert!(execute_supplied_with(&s.request, &s.authority).is_err());
    assert_eq!(
        s.request
            .open()
            .unwrap()
            .budget_snapshot(&s.request.contract_digest)
            .unwrap()
            .spent
            .attempts,
        0
    );
}
#[test]
fn supplied_ir_malicious_manifests_cannot_change_exact_inputs_or_execute_callbacks() {
    if isolated() {
        return;
    }
    let (s, request, planned) = setup();
    let store = ArtifactStore::create(&s.request.store, store_limits()).unwrap();
    let original = canonical(&request["program"]).unwrap();
    for (field, value) in [
        ("operator_callback", json!("/tmp/not-an-authority")),
        ("program", json!(fixture::d("substituted program"))),
        ("compiled_plan", json!({"execute":"not-a-digest"})),
        ("frozen_contract", json!(fixture::d("other contract"))),
        ("input_bundle", json!(fixture::d("missing input bundle"))),
    ] {
        let mut manifest = planned["manifest"].clone();
        manifest[field] = value;
        let mut binding: supplied::Binding =
            decode(canonical(&planned["operator_binding"]).unwrap().bytes()).unwrap();
        binding.manifest = store
            .put(canonical(&manifest).unwrap().bytes())
            .unwrap()
            .digest;
        assert!(
            supplied::verify_binding(&s.request, &s.contract, &binding).is_err(),
            "{field}"
        );
    }
    // A plausible but false compiled digest cannot bypass independent recompile
    // in the unprivileged process; the privileged issuer never compiles code.
    let mut manifest = planned["manifest"].clone();
    manifest["compiled_plan"] = json!(fixture::d("forged compiled plan"));
    let mut binding: supplied::Binding =
        decode(canonical(&planned["operator_binding"]).unwrap().bytes()).unwrap();
    binding.manifest = store
        .put(canonical(&manifest).unwrap().bytes())
        .unwrap()
        .digest;
    supplied::verify_binding(&s.request, &s.contract, &binding).unwrap();
    assert!(supplied::retained(&s.request, &binding).is_err());
    assert_eq!(
        store.read_bounded(&original.digest(), 65536).unwrap(),
        original.bytes()
    );
    assert_eq!(
        s.request
            .open()
            .unwrap()
            .budget_snapshot(&s.request.contract_digest)
            .unwrap()
            .spent
            .attempts,
        0
    );
}
#[test]
fn supplied_ir_plan_is_read_only_even_with_failing_smoke() {
    if isolated() {
        return;
    }
    let (s, mut request, _) = setup();
    fn files(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
        let mut result = BTreeMap::new();
        for entry in std::fs::read_dir(root).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                result.extend(files(&path));
            } else {
                result.insert(path.clone(), std::fs::read(path).unwrap());
            }
        }
        result
    }
    let before = files(&s.root);
    request["smoke_suite"]["probes"][0]["expected"] = json!([999.0, 999.0]);
    request["program"]["verification"]["smoke_suite"] =
        json!(fixture::hash(&request["smoke_suite"]));
    request["control_program"]["verification"] = request["program"]["verification"].clone();
    request["policy"]["verification"] = request["program"]["verification"].clone();
    let plan = dispatch("plan", canonical(&request).unwrap().bytes()).unwrap();
    assert_eq!(plan["execution"], "not_run");
    assert_eq!(plan["contract_binding_matches"], true);
    assert_eq!(
        before,
        files(&s.root),
        "planning must not write or execute the failing smoke"
    );
}
#[test]
fn supplied_ir_revocation_during_reserved_work_cancels_and_recovers_without_retry() {
    if isolated() {
        return;
    }
    let (mut s, _, planned) = setup();
    let grant = authorize(&mut s, &planned);
    std::fs::write(s.root.join("test-revoke-on-admission"), b"").unwrap();
    assert!(supplied::launch_test(&s.request, &grant, &s.contract).is_err());
    let ledger = s.request.open().unwrap();
    assert!(ledger
        .stop_record(&s.request.contract_digest)
        .unwrap()
        .is_some());
    let before = ledger.budget_snapshot(&s.request.contract_digest).unwrap();
    assert_eq!(before.spent.attempts, 1);
    let result = dispatch("resume", canonical(&s.request).unwrap().bytes()).unwrap();
    assert_eq!(result["execution"], "not_resumed");
    let after = ledger.budget_snapshot(&s.request.contract_digest).unwrap();
    assert_eq!(before.spent.attempts, after.spent.attempts);
    assert!(after.unknown_charged.cpu_ms >= before.unknown_charged.cpu_ms);
    assert!(after.unknown_charged.storage_bytes >= before.unknown_charged.storage_bytes);
    assert!(execute_supplied_with(&s.request, &s.authority).is_err());
}
#[test]
fn supplied_ir_revocation_before_publication_cancels_authoritative_completion() {
    if isolated() {
        return;
    }
    let (mut s, _, planned) = setup();
    let grant = authorize(&mut s, &planned);
    std::fs::write(s.root.join("test-revoke-before-publication"), b"").unwrap();
    assert!(supplied::launch_test(&s.request, &grant, &s.contract).is_err());
    let ledger = s.request.open().unwrap();
    assert!(ledger
        .stop_record(&s.request.contract_digest)
        .unwrap()
        .is_some());
    let db = rusqlite::Connection::open(s.request.ledger.join("events.sqlite3")).unwrap();
    for kind in ["completion", "commit_observation"] {
        let count: u64 = db
            .query_row(
                "SELECT count(*) FROM final_fixture_records WHERE contract=?1 AND kind=?2",
                [s.request.contract_digest.as_str(), kind],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 0, "withdrawal must roll back authoritative {kind}");
    }
    let status = dispatch("status", canonical(&s.request).unwrap().bytes()).unwrap();
    assert_eq!(status["authority"]["accepted"], false);
    let before = ledger.budget_snapshot(&s.request.contract_digest).unwrap();
    assert_eq!(
        dispatch("resume", canonical(&s.request).unwrap().bytes()).unwrap()["execution"],
        "not_resumed"
    );
    let after = ledger.budget_snapshot(&s.request.contract_digest).unwrap();
    assert!(after.spent.attempts >= before.spent.attempts);
    assert!(after.unknown_charged.cpu_ms >= before.unknown_charged.cpu_ms);
    assert!(supplied::launch_test(&s.request, &grant, &s.contract).is_err());
}
#[test]
fn supplied_ir_late_controller_cancel_is_durable_and_not_success() {
    if isolated() {
        return;
    }
    let (mut s, _, planned) = setup();
    let grant = authorize(&mut s, &planned);
    let cancel = nf_surgeon::cancellation::CancellationToken::default();
    // Deterministically cancel after decoding the actual successful response.
    let error = supplied::launch_test_with_cancel(&s.request, &grant, &s.contract, &cancel, true)
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::TransactionAborted);
    let ledger = s.request.open().unwrap();
    assert!(ledger
        .stop_record(&s.request.contract_digest)
        .unwrap()
        .is_some());
    let before = ledger.budget_snapshot(&s.request.contract_digest).unwrap();
    assert_eq!(before.spent.attempts, 1);
    assert!(supplied::launch_test(&s.request, &grant, &s.contract).is_err());
    let recovered = dispatch("resume", canonical(&s.request).unwrap().bytes()).unwrap();
    assert_eq!(recovered["execution"], "not_resumed");
    assert_eq!(
        before.spent.attempts,
        ledger
            .budget_snapshot(&s.request.contract_digest)
            .unwrap()
            .spent
            .attempts
    );
}
#[test]
fn supplied_ir_expiration_during_admission_stops_without_refunding_or_replay() {
    if isolated() {
        return;
    }
    let (mut s, _, planned) = setup();
    let grant = authorize_until(&mut s, &planned, now().unwrap() + 10);
    std::fs::write(s.root.join("test-expire-on-admission"), b"").unwrap();
    assert!(supplied::launch_test(&s.request, &grant, &s.contract).is_err());
    assert!(now().unwrap() >= grant.expires_unix_seconds);
    let ledger = s.request.open().unwrap();
    assert!(ledger
        .stop_record(&s.request.contract_digest)
        .unwrap()
        .is_some());
    let before = ledger.budget_snapshot(&s.request.contract_digest).unwrap();
    assert_eq!(before.spent.attempts, 1);
    assert_eq!(
        dispatch("resume", canonical(&s.request).unwrap().bytes()).unwrap()["execution"],
        "not_resumed"
    );
    assert!(execute_supplied_with(&s.request, &s.authority).is_err());
    let after = ledger.budget_snapshot(&s.request.contract_digest).unwrap();
    assert_eq!(before.spent.attempts, after.spent.attempts);
    assert!(after.unknown_charged.cpu_ms >= before.unknown_charged.cpu_ms);
}
#[test]
fn supplied_ir_same_uid_cas_forgery_and_wrong_store_cannot_grant_authority() {
    if isolated() {
        return;
    }
    let (mut s, _, planned) = setup();
    let grant = authorize(&mut s, &planned);
    let doc = canonical(&grant).unwrap();
    let store = ArtifactStore::create(&s.request.store, store_limits()).unwrap();
    assert_eq!(
        store.put(doc.bytes()).unwrap().digest,
        s.request.authorization.clone().unwrap()
    );
    if unsafe { libc::getuid() } != 0 {
        // Bypass only directory traversal to isolate the record-owner check:
        // even byte-identical same-UID records cannot satisfy production uid 0.
        let root_authority = Authority {
            directory: s.authority.directory.try_clone().unwrap(),
            uid: 0,
        };
        assert!(root_authority.verify(&s.request, &s.contract).is_err());
        assert!(operator(
            "authorize",
            canonical(&json!({"request":s.request})).unwrap().bytes()
        )
        .is_err());
    }
    let original = s.request.store.clone();
    s.request.store = s.root.join("other-store");
    ArtifactStore::create(&s.request.store, store_limits())
        .unwrap()
        .put(doc.bytes())
        .unwrap();
    assert!(s.authority.verify(&s.request, &s.contract).is_err());
    s.request.store = original;
    s.authority.verify(&s.request, &s.contract).unwrap();
    assert_eq!(
        s.request
            .open()
            .unwrap()
            .budget_snapshot(&s.request.contract_digest)
            .unwrap()
            .spent
            .attempts,
        0
    );
}
#[test]
fn supplied_ir_crashed_child_retains_charge_and_recovery_never_reexecutes() {
    if isolated() {
        return;
    }
    let (mut s, _, planned) = setup();
    let grant = authorize_until(&mut s, &planned, now().unwrap() + 180);
    std::fs::write(s.root.join("test-crash-on-admission"), b"").unwrap();
    assert!(supplied::launch_test(&s.request, &grant, &s.contract).is_err());
    let ledger = s.request.open().unwrap();
    let before = ledger.budget_snapshot(&s.request.contract_digest).unwrap();
    assert_eq!(before.spent.attempts, 0);
    assert_eq!(before.held.attempts, 1);
    // Real crash leaves its 60-second lease behind. Do not weaken the fence or
    // edit SQLite to force takeover. The enclosing command has a 90-second cap.
    std::thread::sleep(std::time::Duration::from_secs(61));
    let recovered = dispatch("resume", canonical(&s.request).unwrap().bytes()).unwrap();
    assert_eq!(recovered["execution"], "not_resumed");
    let after = ledger.budget_snapshot(&s.request.contract_digest).unwrap();
    assert_eq!(after.spent.attempts, 1);
    assert_eq!(after.held.attempts, 0);
    assert_eq!(after.reservation_count, before.reservation_count);
    assert!(after.unknown_charged.cpu_ms >= before.held.cpu_ms);
    assert!(after.unknown_charged.storage_bytes >= before.held.storage_bytes);
    s.authority.verify(&s.request, &s.contract).unwrap(); // Still live: replay is not merely denied by expiry.
    assert!(execute_supplied_with(&s.request, &s.authority).is_err());
}
#[test]
fn supplied_ir_stop_cancel_and_resume_do_not_redispatch() {
    if isolated() {
        return;
    }
    let (mut s, _, planned) = setup();
    let grant = authorize(&mut s, &planned);
    let token = nf_surgeon::cancellation::CancellationToken::default();
    token.request();
    assert!(supplied::execute(&s.request, &grant, &token).is_err());
    let stopped = dispatch("stop", canonical(&s.request).unwrap().bytes()).unwrap();
    assert_eq!(stopped["request_status"], "persisted");
    assert!(execute_supplied_with(&s.request, &s.authority).is_err());
    let recovery = dispatch("resume", canonical(&s.request).unwrap().bytes()).unwrap();
    assert_eq!(recovery["execution"], "not_resumed");
    assert_eq!(
        s.request
            .open()
            .unwrap()
            .budget_snapshot(&s.request.contract_digest)
            .unwrap()
            .spent
            .attempts,
        0
    );
}
