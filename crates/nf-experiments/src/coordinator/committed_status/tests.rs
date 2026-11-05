use super::*;
use crate::status_reports::CommittedSnapshot;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    mpsc,
};
#[allow(dead_code)]
mod fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../nf-core/tests/common/study_fixture.rs"
    ));
}
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Root(PathBuf);
impl Root {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "committed-status-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn db(&self) -> PathBuf {
        self.0.join("ledger/events.sqlite3")
    }
    fn setup(&self) -> (Ledger, ContentDigest, Lease) {
        let mut ledger = Ledger::open(&self.0.join("ledger"), LedgerLimits::default()).unwrap();
        let contract =
            Validated::<StudyContract>::parse(&serde_json::to_vec(&fixture::standard()).unwrap())
                .unwrap();
        let id = ledger.freeze(&contract).unwrap();
        let lease = ledger.acquire(&id, "status-tests", 300000).unwrap();
        (ledger, id, lease)
    }
}
impl Drop for Root {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn request() -> budgets::Request {
    budgets::Request {
        schema_version: 1,
        operation: fixture::d("work"),
        attempt: 0,
        parent: None,
        kind: budgets::WorkKind::Metadata,
        resources: budgets::Resources {
            cpu_ms: 100,
            host_bytes: 1000,
            workers: 1,
            ..Default::default()
        },
        wall_ms: 10000,
        external: None,
    }
}
#[test]
fn status_reports_rejects_forged_v8_audit_objects_and_provenance_without_writing() {
    for mutation in [
        "DROP TRIGGER audit_claim_no_update;",
        "CREATE TRIGGER shadow_insert BEFORE INSERT ON execution_owners BEGIN SELECT RAISE(IGNORE); END;",
        "DELETE FROM budget_migrations WHERE version=8;",
        "UPDATE budget_migrations SET previous='forged' WHERE version=8;",
    ] {
        let root = Root::new();
        let (ledger, id, _) = root.setup();
        ledger.connection.execute_batch(mutation).unwrap();
        drop(ledger);
        let before = std::fs::read(root.db()).unwrap();
        assert!(CommittedSnapshot::capture(&root.db(), &id).is_err(), "accepted forged v8: {mutation}");
        assert_eq!(std::fs::read(root.db()).unwrap(), before, "diagnostic reads never repair or migrate");
    }
}

#[test]
fn status_reports_capture_retains_all_cost_versions_without_writing_or_inventing_authority() {
    let root = Root::new();
    let (mut ledger, id, lease) = root.setup();
    let reservation = ledger.reserve(&lease, request()).unwrap();
    ledger
        .settle(
            &lease,
            &reservation,
            budgets::Settlement {
                operation: fixture::d("unknown"),
                outcome: budgets::OutcomeKind::Incomplete,
                usage: budgets::Usage::default(),
                resources_released: true,
                evidence: fixture::d("unresolved"),
            },
        )
        .unwrap();
    let before = std::fs::read(root.db()).unwrap();
    let names = std::fs::read_dir(root.db().parent().unwrap())
        .unwrap()
        .map(|r| r.unwrap().file_name())
        .collect::<std::collections::BTreeSet<_>>();
    let snapshot = CommittedSnapshot::capture(&root.db(), &id).unwrap();
    assert_eq!(before, std::fs::read(root.db()).unwrap());
    assert_eq!(
        names,
        std::fs::read_dir(root.db().parent().unwrap())
            .unwrap()
            .map(|r| r.unwrap().file_name())
            .collect()
    );
    let value: Value = serde_json::from_str(&snapshot.render_json()).unwrap();
    assert_eq!(value["snapshot"]["database_schema_version"], 8);
    assert_eq!(
        value["budgets"]["aggregate"]["unknown_charged"]["cpu_ms"],
        100
    );
    assert_eq!(
        value["budgets"]["records"]["outcomes"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(value["authority"]["accepted"], false);
    assert_eq!(value["controls"]["status"], "unknown");
    ledger
        .settle(
            &lease,
            &reservation,
            budgets::Settlement {
                operation: fixture::d("known"),
                outcome: budgets::OutcomeKind::Completed,
                usage: budgets::Usage::known(&budgets::Resources {
                    cpu_ms: 40,
                    ..Default::default()
                }),
                resources_released: true,
                evidence: fixture::d("cost"),
            },
        )
        .unwrap();
    let value: Value = serde_json::from_str(
        &CommittedSnapshot::capture(&root.db(), &id)
            .unwrap()
            .render_json(),
    )
    .unwrap();
    assert_eq!(
        value["budgets"]["records"]["outcomes"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(value["budgets"]["aggregate"]["spent"]["cpu_ms"], 40);
    assert_eq!(
        value["budgets"]["aggregate"]["unknown_charged"]["cpu_ms"],
        0
    );
    assert!(snapshot
        .render_json()
        .contains("single committed coordinator read transaction"));
}
#[test]
fn status_reports_single_transaction_holds_old_journal_while_writer_commits_next_state() {
    let root = Root::new();
    let (mut ledger, id, lease) = root.setup();
    let (begin_tx, begin_rx) = mpsc::channel();
    let (started_tx, started_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    let writer = std::thread::spawn(move || {
        begin_rx.recv().unwrap();
        started_tx.send(()).unwrap();
        let event = ledger.advance(
            &lease,
            Transition {
                operation: fixture::d("concurrent"),
                expected: Stage::Frozen,
                next: Stage::Preflight,
                protocol: fixture::d("protocol"),
                evidence: fixture::d("unresolved"),
                message: "next committed state".into(),
            },
        );
        done_tx.send(event.map(|e| e.digest)).unwrap();
    });
    let read = read(&root.db(), &id, || {
        begin_tx.send(()).unwrap();
        started_rx.recv().unwrap();
        assert!(
            done_rx.recv_timeout(Duration::from_millis(50)).is_err(),
            "writer cannot commit while snapshot read lock is held"
        );
    })
    .unwrap();
    assert_eq!(read.state.stage, Stage::Frozen);
    assert_eq!(read.events.len(), 1);
    assert_eq!(read.budgets.reservation_count, 0);
    done_rx
        .recv_timeout(Duration::from_secs(3))
        .unwrap()
        .unwrap();
    writer.join().unwrap();
    let next = Ledger::read_committed_status(&root.db(), &id).unwrap();
    assert_eq!(next.state.stage, Stage::Preflight);
    assert_eq!(next.events.len(), 2);
}
#[test]
fn status_reports_missing_unsafe_foreign_and_future_stores_never_initialize_or_migrate() {
    let root = Root::new();
    let missing = root.0.join("absent/events.sqlite3");
    assert!(CommittedSnapshot::capture(&missing, &fixture::d("unknown")).is_err());
    assert!(!missing.parent().unwrap().exists());
    let (ledger, id, _) = root.setup();
    let link = root.0.join("linked");
    std::os::unix::fs::symlink(root.db().parent().unwrap(), &link).unwrap();
    assert!(CommittedSnapshot::capture(&link.join("events.sqlite3"), &id).is_err());
    let alias = root.0.join("alias");
    std::fs::hard_link(root.db(), &alias).unwrap();
    assert!(CommittedSnapshot::capture(&root.db(), &id).is_err());
    std::fs::remove_file(alias).unwrap();
    let journal = root.db().with_file_name("events.sqlite3-journal");
    std::os::unix::fs::symlink(root.db(), &journal).unwrap();
    assert!(CommittedSnapshot::capture(&root.db(), &id).is_err());
    std::fs::remove_file(journal).unwrap();
    ledger
        .connection
        .execute_batch("PRAGMA user_version=9;")
        .unwrap();
    let before = std::fs::read(root.db()).unwrap();
    assert_eq!(
        CommittedSnapshot::capture(&root.db(), &id)
            .unwrap_err()
            .code,
        ErrorCode::UnsupportedVersion
    );
    assert_eq!(before, std::fs::read(root.db()).unwrap());
}
#[test]
fn status_reports_foreign_layout_is_rejected_before_selecting_substituted_views() {
    let root = Root::new();
    let (ledger, id, _) = root.setup();
    ledger.connection.execute_batch("DROP TABLE budget_requests;CREATE VIEW budget_requests AS SELECT zeroblob(999999999) AS body;").unwrap();
    let before = std::fs::read(root.db()).unwrap();
    assert_eq!(
        CommittedSnapshot::capture(&root.db(), &id)
            .unwrap_err()
            .code,
        ErrorCode::UnsupportedVersion
    );
    assert_eq!(before, std::fs::read(root.db()).unwrap());
}
#[test]
fn status_reports_read_only_v5_capture_and_unresolved_native_records_do_not_grant_acceptance() {
    let root = Root::new();
    let (ledger, id, _) = root.setup();
    let body = canonical(&json!({"accepted":true,"promotion_authorized":true,
        "receipt":{"cells":[{"outputs":["HOLDOUT_CONTENT_SENTINEL"],
            "samples":[{"output":"HOLDOUT_SAMPLE_SENTINEL"}]}]}}))
    .unwrap();
    ledger
        .connection
        .execute(
            "INSERT INTO final_fixture_records VALUES(?1,'untrusted_claim',?2,?3)",
            params![id.as_str(), body, ContentDigest::of_bytes(&body).as_str()],
        )
        .unwrap();
    let value: Value = serde_json::from_str(
        &CommittedSnapshot::capture(&root.db(), &id)
            .unwrap()
            .render_json(),
    )
    .unwrap();
    let retained = &value["split_evidence"]["native_final_record_identities"][0];
    assert_eq!(retained["digest"], json!(ContentDigest::of_bytes(&body)));
    assert!(retained.get("record").is_none());
    assert_eq!(value["authority"]["accepted"], false);
    let snapshot = CommittedSnapshot::capture(&root.db(), &id).unwrap();
    for rendered in [snapshot.render_json(), snapshot.render_markdown().unwrap()] {
        assert!(!rendered.contains("HOLDOUT_CONTENT_SENTINEL"));
        assert!(!rendered.contains("HOLDOUT_SAMPLE_SENTINEL"));
    }
    let markdown = snapshot.render_markdown().unwrap();
    for (name, section) in value.as_object().unwrap() {
        let tail = markdown.split_once(&format!("## {name}\n\n")).unwrap().1;
        let (opening, rest) = tail.split_once('\n').unwrap();
        let fence = opening.strip_suffix("json").unwrap();
        let body = rest.split_once(&format!("\n{fence}\n")).unwrap().0;
        assert_eq!(serde_json::from_str::<Value>(body).unwrap(), *section);
    }
    // Build an exact old schema, then assert capture reads but never migrates it.
    ledger
        .connection
        .execute_batch(budgets::audit_budget::STRIP_V8)
        .unwrap();
    ledger.connection.execute_batch("DROP TABLE accepted_heads;DROP TABLE verified_frontier;DROP TABLE accepted_records;DROP TABLE promotion_records;DELETE FROM budget_migrations WHERE version=7;DROP TABLE final_fixture_records;DELETE FROM budget_migrations WHERE version=6;PRAGMA user_version=5;").unwrap();
    ledger
        .connection
        .execute(
            "UPDATE meta SET value=?1 WHERE key='schema'",
            [exposure::schema_v5().as_str()],
        )
        .unwrap();
    let before = std::fs::read(root.db()).unwrap();
    let value: Value = serde_json::from_str(
        &CommittedSnapshot::capture(&root.db(), &id)
            .unwrap()
            .render_json(),
    )
    .unwrap();
    assert_eq!(value["snapshot"]["database_schema_version"], 5);
    assert_eq!(before, std::fs::read(root.db()).unwrap());
}
#[test]
fn status_reports_family_scope_retains_prior_revision_stops_and_budgets() {
    let root = Root::new();
    let (mut ledger, first, lease) = root.setup();
    let reservation = ledger.reserve(&lease, request()).unwrap();
    ledger
        .settle(
            &lease,
            &reservation,
            budgets::Settlement {
                operation: fixture::d("first-cost"),
                outcome: budgets::OutcomeKind::Completed,
                usage: budgets::Usage::known(&budgets::Resources {
                    cpu_ms: 40,
                    ..Default::default()
                }),
                resources_released: true,
                evidence: fixture::d("cost"),
            },
        )
        .unwrap();
    ledger
        .request_stop(
            &first,
            fixture::d("stop-old"),
            supervision::StopReason::OwnerStop,
        )
        .unwrap();
    ledger.release(&lease).unwrap();
    let mut value = fixture::standard();
    value["revision"] = json!(2);
    value["parent_contract"] = json!(first);
    value["approval_receipt"] = json!(fixture::d("new-approval"));
    let contract = Validated::<StudyContract>::parse(&serde_json::to_vec(&value).unwrap()).unwrap();
    let second = ledger.freeze(&contract).unwrap();
    let lease = ledger.acquire(&second, "second", 300000).unwrap();
    let mut next = request();
    next.operation = fixture::d("second-work");
    ledger.reserve(&lease, next).unwrap();
    let value: Value = serde_json::from_str(
        &CommittedSnapshot::capture(&root.db(), &second)
            .unwrap()
            .render_json(),
    )
    .unwrap();
    assert_eq!(value["snapshot"]["revision"], 2);
    assert_eq!(value["budgets"]["origin"], json!(first));
    assert_eq!(
        value["budgets"]["records"]["requests"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(value["budgets"]["aggregate"]["spent"]["cpu_ms"], 40);
    assert_eq!(value["budgets"]["aggregate"]["held"]["cpu_ms"], 100);
    assert_eq!(
        value["supervision"]["stop_requests"][0]["record"]["contract"],
        json!(first)
    );
    assert_eq!(value["status"]["stage"], "frozen");
    let journals = value["family_journal"]["revisions"].as_array().unwrap();
    assert_eq!(journals.len(), 2);
    assert_eq!(journals[0]["contract"], json!(first));
    assert_eq!(journals[1]["contract"], json!(second));
    assert_eq!(journals[1]["parent"], json!(first));
    assert_eq!(journals[0]["events"][0]["record"]["contract"], json!(first));
    assert_eq!(
        journals[1]["events"][0]["record"]["contract"],
        json!(second)
    );
    // Corruption of an ancestor journal must fail even when the current
    // revision and family budget fold are individually valid.
    ledger
        .connection
        .execute_batch("DROP TRIGGER events_no_update;")
        .unwrap();
    ledger
        .connection
        .execute(
            "UPDATE events SET body=x'7b7d' WHERE contract=?1",
            [first.as_str()],
        )
        .unwrap();
    assert_eq!(
        CommittedSnapshot::capture(&root.db(), &second)
            .unwrap_err()
            .code,
        ErrorCode::IntegrityMismatch
    );
}
#[test]
fn status_reports_retains_actual_panel_opening_and_exposure_chain_as_diagnostics() {
    let root = Root::new();
    let (mut ledger, id, lease) = root.setup();
    let contract = ledger.contract(&id).unwrap();
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
                    operation: fixture::d(&format!("phase{i}")),
                    expected: previous,
                    next,
                    protocol: fixture::d("protocol"),
                    evidence: fixture::d("unresolved"),
                    message: "software test only".into(),
                },
            )
            .unwrap();
        previous = next;
    }
    let panel = ledger
        .freeze_final_panel(&lease, vec![contract.source.clone()])
        .unwrap();
    ledger
        .open_fixture_holdout(&lease, "fixture-metadata-only")
        .unwrap();
    let binding = &contract.portfolio.bindings[0];
    let subject = exposure::Subject {
        contract: id.clone(),
        binding: binding.id.clone(),
        package_version: binding.gym.clone(),
        dataset_version: binding.splits.holdout.dataset.clone(),
        task: binding.splits.holdout.items[0].clone(),
        input: fixture::d("input"),
        semantic: fixture::d("semantic"),
        identity_protocol: exposure::identity_protocol(),
        label: fixture::d("label"),
        split: Some(nf_core::study::SplitRole::Holdout),
        legacy: false,
    };
    let ticket = exposure::begin(
        &ledger.connection,
        &lease,
        &contract,
        exposure::Start {
            subject,
            recipient: "test-controller".into(),
            role: "metadata_fixture".into(),
            operation: fixture::d("exposure"),
            context: Default::default(),
            fields: [exposure::Field::Identity].into_iter().collect(),
        },
    )
    .unwrap();
    ledger
        .finish_exposure(
            &lease,
            &ticket,
            exposure::Stage::Incomplete,
            Default::default(),
            "interrupted_fixture_no_model_execution",
            Default::default(),
        )
        .unwrap();
    ledger
        .connection
        .execute(
            "INSERT INTO exposure_gaps VALUES(?1,'unknown legacy history')",
            [contract.study_family.as_str()],
        )
        .unwrap();
    let value: Value = serde_json::from_str(
        &CommittedSnapshot::capture(&root.db(), &id)
            .unwrap()
            .render_json(),
    )
    .unwrap();
    assert_eq!(value["split_evidence"]["panels"][0]["digest"], json!(panel));
    assert_eq!(
        value["split_evidence"]["holdout_openings"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        value["split_evidence"]["exposure_events"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
    assert_eq!(
        value["split_evidence"]["exposure_events"][2]["record"]["stage"],
        "incomplete"
    );
    assert_eq!(
        value["split_evidence"]["exposure_gap"],
        "unknown legacy history"
    );
    assert_eq!(value["authority"]["accepted"], false);
    assert_eq!(value["gym_gates"][0]["status"], "unknown");
    assert_eq!(value["candidates"]["status"], "frozen_identities_only");
    assert_eq!(value["candidates"]["panels"][0]["digest"], json!(panel));
    assert_eq!(
        value["candidates"]["panels"][0]["panel"]["original"],
        json!(contract.source)
    );
    assert_eq!(
        value["candidates"]["panels"][0]["panel"]["finalists"][0],
        json!(contract.source)
    );
    assert!(value["candidates"]["execution_receipt_links"].is_null());
    assert!(value["status"]["attempt_counts_by_candidate_verdict"].is_null());
    assert_eq!(value["authority"]["independently_verified"], false);
    // Rehashing a changed panel does not repair its contract bindings. No
    // serialized status, plan or candidate identity can upgrade frozen metadata
    // into execution evidence.
    let frozen = value["candidates"]["panels"][0]["panel"].clone();
    ledger
        .connection
        .execute_batch("DROP TRIGGER panels_no_update;")
        .unwrap();
    for (pointer, replacement) in [
        ("/schema_version", json!(2)),
        ("/contract", json!(fixture::d("foreign-contract"))),
        ("/revision", json!(99)),
        ("/protocol", json!(fixture::d("foreign-protocol"))),
        ("/original/checkpoint", json!(fixture::d("foreign-source"))),
        ("/best_known/checkpoint", json!(fixture::d("foreign-best"))),
        ("/bindings", json!([])),
        ("/seeds", json!([99999])),
        ("/repeats", json!(99)),
        (
            "/comparison_family",
            json!(fixture::d("foreign-comparison")),
        ),
        ("/finalists", json!([])),
        ("/finalists", json!([contract.source, contract.source])),
        ("/status", json!("accepted")),
    ] {
        let mut tampered = frozen.clone();
        *tampered.pointer_mut(pointer).unwrap() = replacement;
        let body = canonical(&tampered).unwrap();
        ledger
            .connection
            .execute(
                "UPDATE final_panels SET body=?1,digest=?2 WHERE contract=?3",
                params![body, ContentDigest::of_bytes(&body).as_str(), id.as_str()],
            )
            .unwrap();
        assert_eq!(
            CommittedSnapshot::capture(&root.db(), &id)
                .unwrap_err()
                .code,
            ErrorCode::IntegrityMismatch,
            "{pointer}"
        );
    }
}
#[test]
fn status_reports_family_indices_cannot_substitute_for_validated_contracts() {
    for tamper in [
        "UPDATE families SET latest='not-a-contract'",
        "UPDATE families SET revision=2",
        "DROP TRIGGER contracts_no_update;UPDATE contracts SET revision=2",
        "DROP TRIGGER contracts_no_update;UPDATE contracts SET parent='forged-parent'",
        "DROP TRIGGER contracts_no_update;UPDATE contracts SET family='foreign-family'",
    ] {
        let root = Root::new();
        let (ledger, id, _) = root.setup();
        ledger.connection.execute_batch(tamper).unwrap();
        let before = std::fs::read(root.db()).unwrap();
        assert_eq!(
            CommittedSnapshot::capture(&root.db(), &id)
                .unwrap_err()
                .code,
            ErrorCode::IntegrityMismatch,
            "{tamper}"
        );
        assert_eq!(std::fs::read(root.db()).unwrap(), before);
    }
}

#[test]
fn status_reports_withheld_native_bodies_still_require_valid_content_hashes() {
    let root = Root::new();
    let (ledger, id, _) = root.setup();
    let body = canonical(&json!({"outputs":[1,2],"accepted":true})).unwrap();
    ledger
        .connection
        .execute(
            "INSERT INTO final_fixture_records VALUES(?1,'execution',?2,?3)",
            params![id.as_str(), body, fixture::d("forged-hash").as_str()],
        )
        .unwrap();
    assert_eq!(
        CommittedSnapshot::capture(&root.db(), &id)
            .unwrap_err()
            .code,
        ErrorCode::IntegrityMismatch
    );
}

#[test]
fn status_reports_hash_corruption_and_oversized_retention_fail_without_partial_reports() {
    let root = Root::new();
    let (ledger, id, _) = root.setup();
    for i in 0..10 {
        let body = canonical(&json!({"padding":"x".repeat(900_000)})).unwrap();
        ledger
            .connection
            .execute(
                "INSERT INTO final_fixture_records VALUES(?1,?2,?3,?4)",
                params![
                    id.as_str(),
                    format!("record{i}"),
                    body,
                    ContentDigest::of_bytes(&body).as_str()
                ],
            )
            .unwrap();
    }
    assert_eq!(
        CommittedSnapshot::capture(&root.db(), &id)
            .unwrap_err()
            .code,
        ErrorCode::LimitExceeded
    );
    let root = Root::new();
    let (ledger, id, _) = root.setup();
    ledger
        .connection
        .execute_batch("DROP TRIGGER events_no_update;UPDATE events SET body=x'7b7d';")
        .unwrap();
    assert_eq!(
        CommittedSnapshot::capture(&root.db(), &id)
            .unwrap_err()
            .code,
        ErrorCode::IntegrityMismatch
    );
}
