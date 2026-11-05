//! Real committed fixture ledger; no science, models, network or evidence DTOs.
use nf_core::schemas::Validated;
use nf_core::study::StudyContract;
use nf_experiments::coordinator::supervision::StopReason;
use nf_experiments::coordinator::{Ledger, LedgerLimits};
use nf_experiments::status_reports::CommittedSnapshot;
use serde_json::{json, Value};

#[allow(dead_code)]
mod fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../nf-core/tests/common/study_fixture.rs"
    ));
}

#[test]
fn status_reports_live_stop_request_and_disabled_training_are_not_execution_or_acceptance() {
    let root = std::env::temp_dir().join(format!("nf-report-stop-{}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(root.clone());
    let mut value = fixture::standard();
    for kind in ["lora", "sft", "distillation", "rl"] {
        let mut arm = value["baselines"][0].clone();
        arm["kind"] = json!(kind);
        arm["enabled"] = json!(false);
        arm["required"] = json!(false);
        value["baselines"].as_array_mut().unwrap().push(arm);
    }
    let contract = Validated::<StudyContract>::parse(&serde_json::to_vec(&value).unwrap()).unwrap();
    let mut ledger = Ledger::open(&root.join("ledger"), LedgerLimits::default()).unwrap();
    let id = ledger.freeze(&contract).unwrap();
    let database = root.join("ledger/events.sqlite3");
    let before = CommittedSnapshot::capture(&database, &id).unwrap();
    ledger
        .request_stop(&id, fixture::d("stop"), StopReason::OwnerStop)
        .unwrap();
    let bytes = std::fs::read(&database).unwrap();
    let after = CommittedSnapshot::capture(&database, &id).unwrap();
    assert_eq!(std::fs::read(&database).unwrap(), bytes);
    let data: Value = serde_json::from_str(&after.render_json()).unwrap();
    assert_eq!(data["status"]["stage"], "frozen");
    assert_eq!(data["status"]["study_stop_reason"]["reason"], "owner_stop");
    assert_eq!(data["status"]["study_stop_reason"]["status"], "recorded");
    assert_eq!(data["authority"]["accepted"], false);
    assert!(data["status"]["attempt_counts_by_candidate_verdict"].is_null());
    let old: Value = serde_json::from_str(&before.render_json()).unwrap();
    assert_eq!(old["status"]["study_stop_reason"]["status"], "unavailable");
    let disabled: Vec<_> = data["baselines"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|arm| arm["configuration"]["enabled"] == false)
        .collect();
    assert!(disabled.len() >= 4);
    for arm in disabled {
        assert_eq!(arm["status"], "not_run");
        assert_eq!(arm["execution_disclosure"]["status"], "not_run");
        assert!(arm["scores"].is_null() && arm["costs"].is_null());
    }
    let markdown = after.render_markdown().unwrap();
    for (name, value) in data.as_object().unwrap() {
        let section = markdown
            .split_once(&format!("## {name}\n\n```json\n"))
            .unwrap()
            .1;
        let body = section.split_once("\n```\n").unwrap().0;
        assert_eq!(serde_json::from_str::<Value>(body).unwrap(), *value);
    }
}

#[cfg(target_os = "linux")]
mod final_evidence_tests {
    use super::*;
    use nf_checkpoint::store::{ArtifactStore, StoreLimits};
    use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
    use nf_core::schemas::ModelManifest;
    use nf_core::study::{ModelIdentity, SplitRole};
    use nf_experiments::coordinator::final_fixture::{graph_bytes, profile_id, runtime_profile};
    use nf_experiments::coordinator::splits::{publish_dataset, Dataset, Item};
    use nf_experiments::coordinator::{Stage, Transition};
    use nf_graph::inventory::{tensor_set_digest, verify_inventory, InventoryLimits};
    use nf_surgeon::cancellation::CancellationToken;
    use nf_surgeon::transaction::{SmokeSuite, SourceState, StateLoadLimits};
    use nf_weight_ir::compilation::{compile, CompilePolicy, SourceSnapshot};
    use nf_weight_ir::program::parse_program;
    use nf_weight_ir::search_space::{parse_search_space, resolve_space};
    use std::{collections::BTreeMap, sync::Arc};

    fn bytes(v: &impl serde::Serialize) -> Vec<u8> {
        CanonicalDocument::parse(&serde_json::to_vec(v).unwrap(), Limits::default())
            .unwrap()
            .bytes()
            .to_vec()
    }
    fn hash(v: &impl serde::Serialize) -> ContentDigest {
        ContentDigest::of_bytes(&bytes(v))
    }
    fn isolated() -> bool {
        if rustix::process::getrlimit(rustix::process::Resource::As)
            .maximum
            .is_some_and(|n| n <= 1_000_000_000)
        {
            return false;
        }
        let status = std::process::Command::new("/usr/bin/prlimit")
            .args(["--as=1000000000:1000000000", "--core=0:0", "--"])
            .arg(std::env::current_exe().unwrap())
            .args([
                "--exact",
                std::thread::current().name().unwrap(),
                "--test-threads=1",
            ])
            .status()
            .unwrap();
        assert!(status.success());
        true
    }
    /// A real edit/export/reload/four-arm native run, not inserted evidence DTOs.
    fn run(
        damage: bool,
        missing_catalog: bool,
        inspect: impl FnOnce(&std::path::Path, &ContentDigest),
    ) {
        run_with_invalid_input(damage, missing_catalog, false, inspect);
    }
    fn run_with_invalid_input(
        damage: bool,
        missing_catalog: bool,
        invalid_input: bool,
        inspect: impl FnOnce(&std::path::Path, &ContentDigest),
    ) {
        let root = std::env::temp_dir().join(format!(
            "t040-native-{}-{damage}-{missing_catalog}",
            std::process::id()
        ));
        std::fs::create_dir(&root).unwrap();
        struct Cleanup(std::path::PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let _cleanup = Cleanup(root.clone());
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
        let checkpoint = f["checkpoint_hex"]
            .as_str()
            .unwrap()
            .as_bytes()
            .chunks_exact(2)
            .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
            .collect::<Vec<_>>();
        let source = SourceState::load(
            Validated::<ModelManifest>::parse(&bytes(&f["manifest"])).unwrap(),
            Arc::from(checkpoint),
            BTreeMap::from([("config".into(), Arc::from(b"{}".as_slice()))]),
            StateLoadLimits::default(),
        )
        .unwrap();
        for blob in [
            bytes(&**source.manifest()),
            source.checkpoint_bytes().to_vec(),
            b"{}".to_vec(),
            graph_bytes().to_vec(),
        ] {
            store.put(blob.as_slice()).unwrap();
        }
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
        let mut c = fixture::standard();
        c["source"] = json!(model);
        c["best_known"] = json!(model);
        for b in c["baselines"].as_array_mut().unwrap() {
            b["input"] = json!(model);
            b["backend"] = json!({"id":"fixture.cpu","version":"1","config":profile_id()});
        }
        c["statistics"]["repeats"] = json!(1);
        c["seeds"] = json!([17]);
        c["runtime"] = json!(runtime_profile().unwrap());
        c["limits"]["host_memory_bytes"] = json!(1_000_000_000u64);
        let gyms = c["portfolio"]["gyms"].as_object().unwrap().clone();
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
                // A valid published JSON payload that the native executor cannot
                // score: interruption occurs AFTER panel freeze and opening.
                let input = if invalid_input && i == 0 && j == 2 {
                    vec![1.0, 2.0, 3.0]
                } else if i == 1 && j == 2 && !damage {
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
                        input: store.put(bytes(&input).as_slice()).unwrap().digest,
                        label: store.put(bytes(&label).as_slice()).unwrap().digest,
                    }],
                };
                b["splits"][role]["dataset"] = json!(publish_dataset(&store, &dataset).unwrap());
            }
        }
        for i in 0..5 {
            let b = &c["portfolio"]["bindings"][i];
            let mut g = gyms[b["gym"].as_str().unwrap()].clone();
            g["adapter"] =
                json!({"id":"nf.final.linear_exact","version":"1","config":profile_id()});
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
                g[key] = json!(profile_id());
            }
            let id = hash(&g);
            c["portfolio"]["gyms"][id.as_str()] = g;
            c["portfolio"]["bindings"][i]["gym"] = json!(id);
            c["portfolio"]["bindings"][i]["validation"] = json!(profile_id());
        }
        c["portfolio"]["bindings"][2]["metrics"][0]["comparator"] = json!("no_op");
        c["portfolio"]["bindings"][4]["metrics"][0]["comparator"] = json!("best_known");
        fixture::refresh(&mut c);
        f["program"]["operations"][0]["operation"]["factor"] = json!(0.5);
        f["program"]["base_model"]["architecture_graph"] = json!(model.graph);
        f["space"]["base_model"] = f["program"]["base_model"].clone();
        let space = parse_search_space(&bytes(&f["space"])).unwrap();
        f["program"]["objective"]["search_space"] = json!(space.digest().unwrap());
        f["program"]["verification"]["portfolio"] = c["portfolio_digest"].clone();
        f["policy"]["source"] = f["program"]["base_model"].clone();
        f["policy"]["search_space"] = json!(space.digest().unwrap());
        c["contract_key"] = f["program"]["objective"]["contract"].clone();
        c["objective"]["id"] = f["program"]["objective"]["objective"].clone();
        let objective = c["objective"]["id"].clone();
        for b in c["portfolio"]["bindings"].as_array_mut().unwrap() {
            b["objective"] = objective.clone();
        }
        c["intervention"]["scope"] = json!(hash(&f["program"]["effects"]));
        c["intervention"]["transformation_space"] = json!(space.digest().unwrap());
        c["intervention"]["method_protocol"] =
            f["program"]["objective"]["experiment_protocol"].clone();
        c["intervention"]["allowed_engines"] =
            json!([{"id":"nf.native_cpu","version":"1","config":hash(&f["policy"]["backend"])}]);
        fixture::refresh(&mut c);
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
        let lease = ledger.acquire(&id, "report-native", 300000).unwrap();
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
                        operation: fixture::d(&format!("transition{i}")),
                        expected: previous,
                        next,
                        protocol: fixture::d("test"),
                        evidence: fixture::d("not-science"),
                        message: "software fixture".into(),
                    },
                )
                .unwrap();
            previous = next;
        }
        let decision = ledger
            .run_fixture_final(
                &lease,
                &store,
                &edit,
                &noop,
                &smoke,
                &CancellationToken::default(),
            )
            .unwrap();
        assert_eq!(
            decision.report().accepted,
            !damage && !missing_catalog && !invalid_input
        );
        // Leave an additional journal descendant: final evidence anchors a real
        // historical ancestor, never a manufactured current event head.
        ledger
            .advance(
                &lease,
                Transition {
                    operation: fixture::d("after-final"),
                    expected: Stage::Verify,
                    next: Stage::Learn,
                    protocol: fixture::d("test"),
                    evidence: fixture::d("not-science"),
                    message: "later event".into(),
                },
            )
            .unwrap();
        inspect(&root.join("ledger/events.sqlite3"), &id);
    }
    fn record(connection: &rusqlite::Connection, id: &ContentDigest, kind: &str) -> Value {
        let body: Vec<u8> = connection
            .query_row(
                "SELECT body FROM final_fixture_records WHERE contract=?1 AND kind=?2",
                rusqlite::params![id.as_str(), kind],
                |r| r.get(0),
            )
            .unwrap();
        serde_json::from_slice(&body).unwrap()
    }
    fn replace_record(
        connection: &rusqlite::Connection,
        id: &ContentDigest,
        kind: &str,
        value: &Value,
    ) -> ContentDigest {
        let body = bytes(value);
        let digest = ContentDigest::of_bytes(&body);
        connection
            .execute(
                "UPDATE final_fixture_records SET body=?1,digest=?2 WHERE contract=?3 AND kind=?4",
                rusqlite::params![body, digest.as_str(), id.as_str(), kind],
            )
            .unwrap();
        digest
    }
    // Repair every dependent hash, so rejection tests semantic verification,
    // not merely the decision/completion/observation content-hash links.
    fn replace_decision(connection: &rusqlite::Connection, id: &ContentDigest, value: &Value) {
        let decision = replace_record(connection, id, "decision", value);
        let mut completion = record(connection, id, "completion");
        completion["decision"] = json!(decision);
        let completion = replace_record(connection, id, "completion", &completion);
        let mut observation = record(connection, id, "commit_observation");
        observation["decision"] = json!(decision);
        observation["completion"] = json!(completion);
        replace_record(connection, id, "commit_observation", &observation);
    }
    fn assert_integrity_rejection(path: &std::path::Path, id: &ContentDigest) {
        let before = std::fs::read(path).unwrap();
        let result = std::panic::catch_unwind(|| CommittedSnapshot::capture(path, id));
        assert_eq!(
            result.expect("capture must not panic").unwrap_err().code,
            nf_core::error::ErrorCode::IntegrityMismatch
        );
        assert_eq!(std::fs::read(path).unwrap(), before);
        // The underlying reader must reject it too, not only the report facade.
        let ledger = Ledger::open(path.parent().unwrap(), LedgerLimits::default()).unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ledger.fixture_final_decision(id)
        }));
        assert_eq!(
            result
                .expect("decision reader must not panic")
                .err()
                .unwrap()
                .code,
            nf_core::error::ErrorCode::IntegrityMismatch
        );
    }
    #[test]
    fn status_reports_rehashed_catalogs_remain_bound_to_frozen_payloads_and_manifests() {
        if isolated() {
            return;
        }
        run(false, false, |path, id| {
            let connection = rusqlite::Connection::open(path).unwrap();
            connection
                .execute_batch("DROP TRIGGER catalogs_no_update;")
                .unwrap();
            let (binding, original): (String, Vec<u8>) = connection.query_row(
                "SELECT binding,body FROM split_catalogs WHERE contract=?1 ORDER BY binding LIMIT 1",
                [id.as_str()], |r| Ok((r.get(0)?, r.get(1)?))).unwrap();
            let original: Value = serde_json::from_slice(&original).unwrap();
            for (pointer, replacement) in [
                ("/schema_version", json!(2)),
                ("/datasets/holdout/schema_version", json!(2)),
                ("/datasets/holdout/role", json!("development")),
                ("/datasets/holdout/role", json!("unknown_split")),
                ("/datasets/holdout/items", json!("not_an_item_array")),
                ("/datasets/holdout/items/0/input", json!("not_a_digest")),
                (
                    "/datasets/holdout/items/0/input",
                    json!(fixture::d("rewritten-input")),
                ),
                (
                    "/datasets/holdout/items/0/label",
                    json!(fixture::d("rewritten-label")),
                ),
                (
                    "/manifests/holdout",
                    json!(fixture::d("rewritten-manifest")),
                ),
                (
                    "/manifests/selection",
                    json!(fixture::d("rewritten-selection")),
                ),
                ("/datasets/selection/schema_version", json!(2)),
                ("/datasets/holdout", Value::Null),
                ("/datasets/development", Value::Null),
                ("/datasets/selection", Value::Null),
                ("/datasets", json!({})),
                ("/manifests/holdout", Value::Null),
                ("/manifests/development", Value::Null),
                ("/manifests/selection", Value::Null),
                ("/manifests", json!({})),
                ("/datasets/development/role", json!("holdout")),
                (
                    "/datasets/selection/items/0/input",
                    json!(fixture::d("rewritten-selection-input")),
                ),
            ] {
                let mut tampered = original.clone();
                if replacement.is_null() {
                    let (parent, key) = pointer.rsplit_once('/').unwrap();
                    tampered
                        .pointer_mut(parent)
                        .unwrap()
                        .as_object_mut()
                        .unwrap()
                        .remove(key);
                } else {
                    *tampered.pointer_mut(pointer).unwrap() = replacement;
                }
                let body = bytes(&tampered);
                connection.execute("UPDATE split_catalogs SET body=?1,digest=?2 WHERE contract=?3 AND binding=?4",
                    rusqlite::params![body, ContentDigest::of_bytes(&body).as_str(), id.as_str(), binding]).unwrap();
                assert_integrity_rejection(path, id);
                let frozen: Vec<u8> = connection
                    .query_row(
                        "SELECT canonical FROM contracts WHERE digest=?1",
                        [id.as_str()],
                        |r| r.get(0),
                    )
                    .unwrap();
                assert_eq!(
                    ContentDigest::of_bytes(&frozen),
                    *id,
                    "frozen requirements never change"
                );
            }
            let body = bytes(&original);
            connection
                .execute(
                    "UPDATE split_catalogs SET body=?1,digest=?2 WHERE contract=?3 AND binding=?4",
                    rusqlite::params![
                        body,
                        ContentDigest::of_bytes(&body).as_str(),
                        id.as_str(),
                        binding
                    ],
                )
                .unwrap();
            assert!(CommittedSnapshot::capture(path, id).is_ok());
        });
    }
    #[test]
    fn status_reports_rehashed_failed_decision_cannot_launder_native_analysis_or_gate_coverage() {
        if isolated() {
            return;
        }
        run(true, false, |path, id| {
            let connection = rusqlite::Connection::open(path).unwrap();
            connection
                .execute_batch("DROP TRIGGER final_fixture_no_update;")
                .unwrap();
            let original = record(&connection, id, "decision");
            let execution = record(&connection, id, "execution");
            assert_eq!(
                execution["analysis_report"]["all_required_analysis_conditions_satisfied"],
                false
            );
            for variant in 0..6 {
                let mut forged = original.clone();
                match variant {
                    0 => {
                        forged["accepted"] = json!(true);
                        forged["status"] = json!("passed");
                        forged["analysis_conditions_satisfied"] = json!(true);
                        for c in forged["checks"].as_array_mut().unwrap() {
                            c["status"] = json!("passed");
                        }
                    }
                    1 => {
                        forged["analysis_conditions_satisfied"] = json!(true);
                    }
                    2 => {
                        forged["checks"]
                            .as_array_mut()
                            .unwrap()
                            .retain(|c| c["status"] != "failed");
                    }
                    3 => {
                        for c in forged["checks"].as_array_mut().unwrap() {
                            if c["status"] == "failed" {
                                c["required"] = json!(false);
                            }
                        }
                    }
                    4 => {
                        let duplicate = forged["checks"][0].clone();
                        forged["checks"].as_array_mut().unwrap().push(duplicate);
                    }
                    _ => {
                        forged["status"] = json!("incomplete");
                    }
                }
                replace_decision(&connection, id, &forged);
                assert_eq!(
                    record(&connection, id, "execution"),
                    execution,
                    "native failure is unchanged"
                );
                assert_integrity_rejection(path, id);
                replace_decision(&connection, id, &original);
                assert!(CommittedSnapshot::capture(path, id).is_ok());
            }
        });
    }
    #[test]
    fn status_reports_rehashed_passing_decision_requires_every_exact_native_gate() {
        if isolated() {
            return;
        }
        run(false, false, |path, id| {
            let connection = rusqlite::Connection::open(path).unwrap();
            connection
                .execute_batch("DROP TRIGGER final_fixture_no_update;")
                .unwrap();
            let original = record(&connection, id, "decision");
            for index in 0..original["checks"].as_array().unwrap().len() {
                for variant in 0..4 {
                    let mut forged = original.clone();
                    match variant {
                        0 => {
                            forged["checks"].as_array_mut().unwrap().remove(index);
                        }
                        1 => {
                            forged["checks"][index]["required"] = json!(false);
                        }
                        2 => {
                            forged["checks"][index]["status"] = json!("incomplete");
                        }
                        _ => {
                            forged["checks"][index]["binding"] =
                                json!(fixture::d("foreign-binding"));
                        }
                    }
                    replace_decision(&connection, id, &forged);
                    assert_integrity_rejection(path, id);
                }
            }
            replace_decision(&connection, id, &original);
            assert!(CommittedSnapshot::capture(path, id).is_ok());
        });
    }
    #[test]
    fn status_reports_incomplete_before_and_after_panel_require_exact_gates_and_settlement() {
        if isolated() {
            return;
        }
        for after_panel in [false, true] {
            run_with_invalid_input(false, !after_panel, after_panel, |path, id| {
                let connection = rusqlite::Connection::open(path).unwrap();
                connection
                    .execute_batch(
                        "DROP TRIGGER final_fixture_no_update;DROP TRIGGER outcomes_no_update;",
                    )
                    .unwrap();
                let original = record(&connection, id, "decision");
                let execution = record(&connection, id, "execution");
                assert_eq!(execution["status"], "incomplete");
                assert_eq!(!original["panel"].is_null(), after_panel);
                assert_eq!(original["analysis_conditions_satisfied"], false);
                let snapshot = CommittedSnapshot::capture(path, id).unwrap();
                let value: Value = serde_json::from_str(&snapshot.render_json()).unwrap();
                assert_eq!(
                    value["final_evidence"]["records"][0]["recorded_status"],
                    "incomplete"
                );
                assert!(value["final_evidence"]["records"][0]["receipt_links"]
                    .as_array()
                    .unwrap()
                    .is_empty());
                for index in 0..original["checks"].as_array().unwrap().len() {
                    for variant in 0..3 {
                        let mut forged = original.clone();
                        match variant {
                            0 => {
                                forged["checks"].as_array_mut().unwrap().remove(index);
                            }
                            1 => {
                                forged["checks"][index]["required"] = json!(false);
                            }
                            _ => {
                                forged["checks"][index]["status"] = json!("passed");
                            }
                        }
                        // Some gates really passed before the interruption.
                        if forged == original {
                            continue;
                        }
                        replace_decision(&connection, id, &forged);
                        assert_integrity_rejection(path, id);
                    }
                }
                replace_decision(&connection, id, &original);
                let intent = record(&connection, id, "intent");
                let (outcome_id, outcome_bytes): (String, Vec<u8>) = connection.query_row(
                    "SELECT id,body FROM budget_outcomes WHERE request=?1 ORDER BY version DESC LIMIT 1",
                    [intent["reservation"].as_str().unwrap()], |r| Ok((r.get(0)?, r.get(1)?))).unwrap();
                let outcome: Value = serde_json::from_slice(&outcome_bytes).unwrap();
                assert_eq!(outcome["settlement"]["outcome"], "incomplete");
                assert_eq!(outcome["settlement"]["evidence"], json!(hash(&execution)));
                for (field, replacement) in [
                    ("evidence", json!(fixture::d("foreign-incomplete-evidence"))),
                    ("outcome", json!("completed")),
                    ("resources_released", json!(false)),
                ] {
                    let mut forged = outcome.clone();
                    forged["settlement"][field] = replacement;
                    let body = bytes(&forged);
                    connection
                        .execute(
                            "UPDATE budget_outcomes SET body=?1,content=?2 WHERE id=?3",
                            rusqlite::params![
                                body,
                                ContentDigest::of_bytes(&body).as_str(),
                                outcome_id
                            ],
                        )
                        .unwrap();
                    assert_integrity_rejection(path, id);
                }
                connection
                    .execute(
                        "UPDATE budget_outcomes SET body=?1,content=?2 WHERE id=?3",
                        rusqlite::params![
                            outcome_bytes,
                            ContentDigest::of_bytes(&outcome_bytes).as_str(),
                            outcome_id
                        ],
                    )
                    .unwrap();
                assert!(CommittedSnapshot::capture(path, id).is_ok());
            });
        }
    }
    #[test]
    fn status_reports_rehashed_native_tamper_and_unobserved_completion_never_project_success() {
        if isolated() {
            return;
        }
        run(false, false, |path, id| {
            let connection = rusqlite::Connection::open(path).unwrap();
            connection
                .execute_batch(
                    "DROP TRIGGER final_fixture_no_update;DROP TRIGGER final_fixture_no_delete;",
                )
                .unwrap();
            for (kind, pointer, replacement) in [
                ("decision", "/event_head", json!(fixture::d("foreign-head"))),
                ("decision", "/revision", json!(999)),
                (
                    "dispatch",
                    "/candidate/checkpoint",
                    json!(fixture::d("foreign-candidate")),
                ),
                (
                    "dispatch",
                    "/noop_export/exported_model",
                    json!(fixture::d("foreign-noop")),
                ),
                ("execution", "/receipt/cells/0/seed", json!(999)),
                (
                    "execution",
                    "/receipt/cells/0/arm_identities/3/checkpoint",
                    json!(fixture::d("foreign-arm")),
                ),
                (
                    "execution",
                    "/receipt/cells/0/outputs/0",
                    json!(["HOLDOUT_CONTENT_SENTINEL"]),
                ),
                ("execution", "/data/observations/0/value", json!(0.12345)),
                ("commit_observation", "/semantics", json!("qualified_T038")),
            ] {
                let (original, digest): (Vec<u8>, String) = connection.query_row(
                    "SELECT body,digest FROM final_fixture_records WHERE contract=?1 AND kind=?2",
                    rusqlite::params![id.as_str(), kind], |r| Ok((r.get(0)?, r.get(1)?))).unwrap();
                let mut value: Value = serde_json::from_slice(&original).unwrap();
                *value.pointer_mut(pointer).unwrap() = replacement;
                let tampered = bytes(&value);
                connection.execute("UPDATE final_fixture_records SET body=?1,digest=?2 WHERE contract=?3 AND kind=?4",
                    rusqlite::params![tampered, ContentDigest::of_bytes(&tampered).as_str(), id.as_str(), kind]).unwrap();
                assert!(
                    CommittedSnapshot::capture(path, id).is_err(),
                    "{kind}{pointer}"
                );
                connection.execute("UPDATE final_fixture_records SET body=?1,digest=?2 WHERE contract=?3 AND kind=?4",
                    rusqlite::params![original, digest, id.as_str(), kind]).unwrap();
            }
            // Valid but unobserved provisional completion remains unknown, even
            // if a completion and serialized accepted decision still exist.
            connection.execute("DELETE FROM final_fixture_records WHERE contract=?1 AND kind='commit_observation'", [id.as_str()]).unwrap();
            let snapshot = CommittedSnapshot::capture(path, id).unwrap();
            let value: Value = serde_json::from_str(&snapshot.render_json()).unwrap();
            assert_eq!(value["final_evidence"]["status"], "unknown");
            assert!(value["candidates"]["execution_receipt_links"].is_null());
            assert_eq!(value["authority"]["accepted"], false);
            assert!(!snapshot.render_json().contains("HOLDOUT_CONTENT_SENTINEL"));
        });
    }
    #[test]
    fn status_reports_real_final_positive_failed_and_incomplete_are_distinct() {
        if isolated() {
            return;
        }
        for (damage, missing) in [(false, false), (true, false), (false, true)] {
            run(damage, missing, |path, id| {
                let before = std::fs::read(path).unwrap();
                let snapshot = CommittedSnapshot::capture(path, id).unwrap();
                assert_eq!(std::fs::read(path).unwrap(), before);
                let value: Value = serde_json::from_str(&snapshot.render_json()).unwrap();
                let e = &value["final_evidence"]["records"][0];
                assert_eq!(value["authority"]["accepted"], false);
                assert_ne!(e["event_head"], value["snapshot"]["event_head"]);
                assert_eq!(
                    e["recorded_status"],
                    if missing {
                        "incomplete"
                    } else if damage {
                        "failed"
                    } else {
                        "passed"
                    }
                );
                assert_eq!(
                    e["receipt_links"].as_array().unwrap().len(),
                    if missing { 0 } else { 4 }
                );
                assert_eq!(
                    e["execution_status"],
                    if missing { "incomplete" } else { "complete" }
                );
                if !missing {
                    assert!(!e["metric_results"].as_array().unwrap().is_empty());
                    for link in e["receipt_links"].as_array().unwrap() {
                        assert_eq!(link["contract"], json!(id));
                        assert_eq!(link["event_head"], e["event_head"]);
                        assert_eq!(link["plan"], e["plan"]);
                        assert_eq!(link["panel"], e["panel"]);
                        assert_eq!(link["seeds"], json!([17]));
                    }
                }
                // Test-only interchange for SDK/CLI cross-process capture. A
                // copied completed ledger is historical data, not a qualification
                // of the Python/CLI executable or its current process limits.
                if let Some(destination) = std::env::var_os("NF_T040_TEST_FIXTURE_OUTPUT") {
                    let destination = std::path::PathBuf::from(destination).join(if missing {
                        "incomplete"
                    } else if damage {
                        "failed"
                    } else {
                        "positive"
                    });
                    std::fs::create_dir_all(&destination).unwrap();
                    std::fs::copy(path, destination.join("events.sqlite3")).unwrap();
                    std::fs::write(destination.join("contract.txt"), id.as_str()).unwrap();
                }
                let markdown = snapshot.render_markdown().unwrap();
                for (name, section) in value.as_object().unwrap() {
                    let tail = markdown.split_once(&format!("## {name}\n\n")).unwrap().1;
                    let (opening, rest) = tail.split_once('\n').unwrap();
                    let fence = opening.strip_suffix("json").unwrap();
                    let body = rest.split_once(&format!("\n{fence}\n")).unwrap().0;
                    assert_eq!(serde_json::from_str::<Value>(body).unwrap(), *section);
                }
                for rendered in [snapshot.render_json(), markdown] {
                    assert!(!rendered.contains("\"outputs\""));
                    assert!(!rendered.contains("\"samples\""));
                    assert!(!rendered.contains("\"observations\""));
                }
                let text = snapshot.render_json();
                assert!(!text.contains("\"outputs\""));
                assert!(!text.contains("\"samples\""));
                assert!(!text.contains("\"observations\""));
                assert!(text.contains("unresolved_hard_wall_and_cpu"));
            });
        }
    }
}
