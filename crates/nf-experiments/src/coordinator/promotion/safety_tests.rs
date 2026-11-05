//! PRIVATE positive tests are not evidence that the disabled public route works.
use super::e2e::Fixture;
use super::*;
fn promote(f: &mut Fixture, runtime: &QualifiedTorch) -> Result<Acceptance> {
    f.ledger.promote_fixture_after_review(
        &f.lease,
        &f.store,
        &f.edit,
        &f.smoke,
        runtime,
        fixture::d("private-promotion"),
        None,
        &CancellationToken::default(),
    )
}
#[test]
fn private_positive_reopen_and_rehashed_semantic_corruption() {
    if e2e::dedicated_test() {
        return;
    }
    let mut f = Fixture::new(true);
    let runtime = f.runtime();
    let mut other = Ledger::open(&f.root.join("ledger"), LedgerLimits::default()).unwrap();
    let barrier = std::sync::Barrier::new(2);
    let lease = &f.lease;
    let store = &f.store;
    let edit = &f.edit;
    let smoke = &f.smoke;
    let (a, b) = std::thread::scope(|scope| {
        let racer = scope.spawn(|| {
            barrier.wait();
            other.promote_fixture_after_review(
                lease,
                store,
                edit,
                smoke,
                &runtime,
                fixture::d("private-promotion"),
                None,
                &CancellationToken::default(),
            )
        });
        barrier.wait();
        let first = f.ledger.promote_fixture_after_review(
            lease,
            store,
            edit,
            smoke,
            &runtime,
            fixture::d("private-promotion"),
            None,
            &CancellationToken::default(),
        );
        (first, racer.join().unwrap())
    });
    assert!(
        a.is_ok() || b.is_ok(),
        "both private racers failed: {a:?}, {b:?}; reload={:?}; terminal={:?}",
        read(&f.ledger.connection, f.lease.contract(), "reload"),
        read(&f.ledger.connection, f.lease.contract(), "attempt_end")
    );
    let winners = [a, b].into_iter().flatten().collect::<Vec<_>>();
    let a = winners[0].clone();
    assert!(
        read(&f.ledger.connection, f.lease.contract(), "child_registered")
            .unwrap()
            .is_some()
    );
    assert!(supervision::all_cleanup_confirmed(&f.ledger.connection, f.lease.contract()).unwrap());
    for winner in &winners {
        assert_eq!(winner.digest(), a.digest());
    }
    assert_eq!(
        f.ledger.events(f.lease.contract(), 0, 100).unwrap().len(),
        9
    );
    assert!(
        read(&f.ledger.connection, f.lease.contract(), "attempt_end")
            .unwrap()
            .is_none()
    );
    assert_eq!(
        f.ledger.state(f.lease.contract()).unwrap().stage,
        Stage::Completed
    );
    assert_eq!(
        f.ledger
            .accepted_fixture_head(&f.family)
            .unwrap()
            .unwrap()
            .digest(),
        a.digest()
    );
    let reopened = Ledger::open(&f.root.join("ledger"), LedgerLimits::default()).unwrap();
    assert_eq!(
        reopened
            .accepted_fixture_head(&f.family)
            .unwrap()
            .unwrap()
            .digest(),
        a.digest()
    );
    assert_eq!(promote(&mut f, &runtime).unwrap().digest(), a.digest());
    let snapshot = f.ledger.budget_snapshot(f.lease.contract()).unwrap();
    assert_eq!(snapshot.held.workers, 0);
    assert_eq!(snapshot.held.host_bytes, 0);
    assert_eq!(snapshot.reservation_count, 2); // one T038 execution, one promotion
    assert!(memory_envelope(2_147_483_648, 3_221_225_472).is_err());
    assert!(memory_envelope(2_147_483_648, 6_442_450_944).is_ok());
    // A rehashed failed final decision with rehashed completion/observation
    // links must fail semantic acceptance, not merely canonical decoding.
    let db = &f.ledger.connection;
    db.execute_batch("DROP TRIGGER final_fixture_no_update;")
        .unwrap();
    let final_row = |kind: &str| {
        let (b, d): (Vec<u8>, String) = db
            .query_row(
                "SELECT body,digest FROM final_fixture_records WHERE contract=?1 AND kind=?2",
                params![f.lease.contract().as_str(), kind],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        decode(&b, &d).unwrap().1
    };
    let replace = |kind: &str, value: &Value| {
        let id = hash(value).unwrap();
        db.execute(
            "UPDATE final_fixture_records SET body=?1,digest=?2 WHERE contract=?3 AND kind=?4",
            params![
                canonical(value).unwrap(),
                id.as_str(),
                f.lease.contract().as_str(),
                kind
            ],
        )
        .unwrap();
        id
    };
    let original_decision = final_row("decision");
    let original_completion = final_row("completion");
    let original_observation = final_row("commit_observation");
    let mut decision = original_decision.clone();
    decision["accepted"] = json!(false);
    decision["status"] = json!("failed");
    let decision_id = replace("decision", &decision);
    let mut completion = original_completion.clone();
    completion["decision"] = json!(decision_id);
    let completion_id = replace("completion", &completion);
    let mut observation = original_observation.clone();
    observation["completion"] = json!(completion_id);
    observation["decision"] = json!(decision_id);
    replace("commit_observation", &observation);
    let error = f.ledger.accepted_fixture_head(&f.family).unwrap_err();
    assert!(
        error.code == ErrorCode::IntegrityMismatch || error.message.contains("not passing"),
        "{error}"
    );
    replace("decision", &original_decision);
    replace("completion", &original_completion);
    replace("commit_observation", &original_observation);
    assert_eq!(
        f.ledger
            .accepted_fixture_head(&f.family)
            .unwrap()
            .unwrap()
            .digest(),
        a.digest()
    );
    // Rehash the reload and relink its settlement. The archival component
    // validator must reach actual parity checking, not stop at a stale hash.
    let db = &f.ledger.connection;
    db.execute_batch("DROP TRIGGER promotion_no_update; DROP TRIGGER accepted_no_update; DROP TRIGGER frontier_no_update;").unwrap();
    let (old, mut reload) = read(db, f.lease.contract(), "reload").unwrap().unwrap();
    reload["child_report"]["outputs"][0][0] = json!(123456.0);
    let changed = hash(&reload).unwrap();
    db.execute(
        "UPDATE promotion_records SET body=?1,digest=?2 WHERE kind='reload' AND contract=?3",
        params![
            canonical(&reload).unwrap(),
            changed.as_str(),
            f.lease.contract().as_str()
        ],
    )
    .unwrap();
    let mut body = a.body().clone();
    body["reload"] = json!(changed);
    let intent = read(db, f.lease.contract(), "intent").unwrap().unwrap().1;
    let probes = read(db, f.lease.contract(), "probes").unwrap().unwrap().1;
    let reservation: ContentDigest = serde_json::from_value(intent["reservation"].clone()).unwrap();
    let mut outcome = history::budget_row(db, "budget_outcomes", "request", &reservation).unwrap();
    outcome["settlement"]["evidence"] = json!(changed);
    db.execute_batch("DROP TRIGGER outcomes_no_update;")
        .unwrap();
    db.execute(
        "UPDATE budget_outcomes SET body=?1,content=?2 WHERE request=?3",
        params![
            canonical(&outcome).unwrap(),
            hash(&outcome).unwrap().as_str(),
            reservation.as_str()
        ],
    )
    .unwrap();
    let error =
        history::validate(db, f.lease.contract(), &intent, &probes, &reload, &body).unwrap_err();
    assert!(error.message.contains("actual Torch forward"), "{error}");
    assert_ne!(old, changed);
    assert!(f.ledger.accepted_fixture_head(&f.family).is_err());
}
#[test]
fn real_acceptance_requires_bound_successful_supervision_and_registered_cleanup() {
    if e2e::dedicated_test() {
        return;
    }
    let mut f = Fixture::new(true);
    let runtime = f.runtime();
    let accepted = promote(&mut f, &runtime).unwrap();
    let db = &f.ledger.connection;
    db.execute_batch("DROP TRIGGER promotion_no_update; DROP TRIGGER promotion_no_delete;")
        .unwrap();
    for kind in ["supervision", "child_intent", "child_registered"] {
        let (_, original) = read(db, f.lease.contract(), kind).unwrap().unwrap();
        db.execute(
            "DELETE FROM promotion_records WHERE contract=?1 AND kind=?2",
            params![f.lease.contract().as_str(), kind],
        )
        .unwrap();
        let error = f.ledger.accepted_fixture_head(&f.family).expect_err(kind);
        assert!(
            error.message.contains("supervision")
                || error.message.contains("registered")
                || error.message.contains("child admission"),
            "{kind}: {error}"
        );
        let tx = db.unchecked_transaction().unwrap();
        record(&tx, f.lease.contract(), kind, &original).unwrap();
        tx.commit().unwrap();
        let mut corrupt = original.clone();
        corrupt["intent"] = json!(fixture::d("foreign-promotion-intent"));
        db.execute(
            "UPDATE promotion_records SET body=?1,digest=?2 WHERE contract=?3 AND kind=?4",
            params![
                canonical(&corrupt).unwrap(),
                hash(&corrupt).unwrap().as_str(),
                f.lease.contract().as_str(),
                kind
            ],
        )
        .unwrap();
        assert!(
            f.ledger.accepted_fixture_head(&f.family).is_err(),
            "foreign {kind} accepted"
        );
        db.execute(
            "UPDATE promotion_records SET body=?1,digest=?2 WHERE contract=?3 AND kind=?4",
            params![
                canonical(&original).unwrap(),
                hash(&original).unwrap().as_str(),
                f.lease.contract().as_str(),
                kind
            ],
        )
        .unwrap();
        assert_eq!(
            f.ledger
                .accepted_fixture_head(&f.family)
                .unwrap()
                .unwrap()
                .digest(),
            accepted.digest()
        );
    }
    let (_, original) = read(db, f.lease.contract(), "supervision")
        .unwrap()
        .unwrap();
    for field in ["result", "cleanup_confirmed"] {
        let mut corrupt = original.clone();
        corrupt[field] = if field == "result" {
            json!("durable stop observed")
        } else {
            json!(false)
        };
        db.execute("UPDATE promotion_records SET body=?1,digest=?2 WHERE contract=?3 AND kind='supervision'",params![canonical(&corrupt).unwrap(),hash(&corrupt).unwrap().as_str(),f.lease.contract().as_str()]).unwrap();
        assert!(
            f.ledger.accepted_fixture_head(&f.family).is_err(),
            "failed supervision {field} accepted"
        );
    }
    db.execute(
        "UPDATE promotion_records SET body=?1,digest=?2 WHERE contract=?3 AND kind='supervision'",
        params![
            canonical(&original).unwrap(),
            hash(&original).unwrap().as_str(),
            f.lease.contract().as_str()
        ],
    )
    .unwrap();
    let reopened = Ledger::open(&f.root.join("ledger"), LedgerLimits::default()).unwrap();
    assert_eq!(
        reopened
            .accepted_fixture_head(&f.family)
            .unwrap()
            .unwrap()
            .digest(),
        accepted.digest()
    );
}
#[test]
fn private_admission_and_publication_failures_do_not_release_early() {
    if e2e::dedicated_test() {
        return;
    }
    let mut f = Fixture::new(true);
    let runtime = f.runtime();
    let before = f
        .ledger
        .budget_snapshot(f.lease.contract())
        .unwrap()
        .reservation_count;
    f.ledger.connection.execute_batch("CREATE TRIGGER reject_promotion_intent BEFORE INSERT ON promotion_records WHEN NEW.kind='intent' BEGIN SELECT RAISE(ABORT,'test admission rollback'); END;").unwrap();
    assert!(promote(&mut f, &runtime).is_err());
    f.unchanged();
    assert_eq!(
        f.ledger
            .budget_snapshot(f.lease.contract())
            .unwrap()
            .reservation_count,
        before
    );
    f.ledger.connection.execute_batch("DROP TRIGGER reject_promotion_intent; CREATE TRIGGER reject_promotion_event BEFORE INSERT ON events WHEN NEW.sequence=9 BEGIN SELECT RAISE(ABORT,'test final commit rollback'); END;").unwrap();
    assert!(promote(&mut f, &runtime).is_err());
    f.unchanged();
    let snapshot = f.ledger.budget_snapshot(f.lease.contract()).unwrap();
    assert_eq!(snapshot.held.workers, 1); // rolled-back completion retains allocation
    let intent = read(&f.ledger.connection, f.lease.contract(), "intent")
        .unwrap()
        .unwrap()
        .1;
    let id: ContentDigest = serde_json::from_value(intent["reservation"].clone()).unwrap();
    assert!(!budgets::native_released(&f.ledger.connection, &id).unwrap());
    f.ledger.recover_fixture_promotion(&f.lease).unwrap(); // durable attempt_end, not merely a live owner
    assert!(budgets::native_released(&f.ledger.connection, &id).unwrap());
    f.unchanged();
    f.ledger
        .connection
        .execute_batch("DROP TRIGGER reject_promotion_event;")
        .unwrap();
    assert!(promote(&mut f, &runtime).is_err()); // no archived witness replay
}
#[test]
fn private_guard_durable_stop_takeover_deadline_and_token() {
    if e2e::dedicated_test() {
        return;
    }
    for mode in 0..4 {
        let f = Fixture::new(true);
        let external = CancellationToken::default();
        let deadline = if mode == 2 {
            files::moment().unwrap().ms + 30
        } else {
            f.lease.deadline
        };
        let guard = supervision_guard::Guard::start(
            f.ledger.files.sqlite_path(),
            f.ledger.store.clone(),
            f.lease.clone(),
            deadline,
            external.clone(),
            CancellationToken::default(),
        )
        .unwrap();
        let mut other = Ledger::open(&f.root.join("ledger"), LedgerLimits::default()).unwrap();
        match mode {
            0 => {
                other
                    .request_stop(
                        f.lease.contract(),
                        fixture::d("guard-stop"),
                        supervision::StopReason::OwnerStop,
                    )
                    .unwrap();
            }
            1 => {
                other.release(&f.lease).unwrap();
                other
                    .acquire(f.lease.contract(), "replacement", 300000)
                    .unwrap();
            }
            2 => {}
            _ => external.request(),
        }
        let until = std::time::Instant::now() + Duration::from_secs(2);
        while !guard.cancel.is_cancelled() && std::time::Instant::now() < until {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(guard.cancel.is_cancelled(), "mode {mode}");
        assert!(guard.finish().is_err());
        f.unchanged();
    }
}
#[test]
fn family_tip_detects_rollback_missing_head_fork_and_disconnected_records() {
    let path = std::env::temp_dir().join(format!("promotion-lineage-{}", std::process::id()));
    let mut ledger = Ledger::open(&path, LedgerLimits::default()).unwrap();
    let c = Validated::<StudyContract>::parse(&serde_json::to_vec(&fixture::standard()).unwrap())
        .unwrap();
    let contract = ledger.freeze(&c).unwrap();
    let a = json!({"family":c.study_family,"predecessor":null});
    let aid = hash(&a).unwrap();
    ledger
        .connection
        .execute(
            "INSERT INTO accepted_records VALUES(?1,?2,?3,?4)",
            params![
                aid.as_str(),
                contract.as_str(),
                c.study_family.as_str(),
                canonical(&a).unwrap()
            ],
        )
        .unwrap();
    // A second frozen revision supplies the next storage lineage node.
    let mut second = fixture::standard();
    second["revision"] = json!(2);
    second["parent_contract"] = json!(contract);
    second["approval_receipt"] = json!(fixture::d("second-lineage-approval"));
    let second = Validated::<StudyContract>::parse(&serde_json::to_vec(&second).unwrap()).unwrap();
    let second_id = ledger.freeze(&second).unwrap();
    let b = json!({"family":c.study_family,"predecessor":aid});
    let bid = hash(&b).unwrap();
    ledger
        .connection
        .execute(
            "INSERT INTO accepted_records VALUES(?1,?2,?3,?4)",
            params![
                bid.as_str(),
                second_id.as_str(),
                c.study_family.as_str(),
                canonical(&b).unwrap()
            ],
        )
        .unwrap();
    assert!(history::check_tip(&ledger.connection, &c.study_family, Some(&bid)).is_ok());
    assert!(history::check_tip(&ledger.connection, &c.study_family, Some(&aid)).is_err());
    assert!(history::check_tip(&ledger.connection, &c.study_family, None).is_err());
    ledger
        .connection
        .execute_batch("DROP TRIGGER accepted_no_update;")
        .unwrap();
    let fork = json!({"family":c.study_family,"predecessor":null,"operation":fixture::d("fork")});
    let fork_id = hash(&fork).unwrap();
    ledger
        .connection
        .execute(
            "UPDATE accepted_records SET body=?1,digest=?2 WHERE digest=?3",
            params![canonical(&fork).unwrap(), fork_id.as_str(), bid.as_str()],
        )
        .unwrap();
    assert!(history::check_tip(&ledger.connection, &c.study_family, Some(&fork_id)).is_err());
    let disconnected = json!({"family":c.study_family,"predecessor":fixture::d("absent")});
    ledger
        .connection
        .execute(
            "UPDATE accepted_records SET body=?1,digest=?2 WHERE digest=?3",
            params![
                canonical(&disconnected).unwrap(),
                hash(&disconnected).unwrap().as_str(),
                fork_id.as_str()
            ],
        )
        .unwrap();
    assert!(history::check_tip(
        &ledger.connection,
        &c.study_family,
        Some(&hash(&disconnected).unwrap())
    )
    .is_err());
    drop(ledger);
    std::fs::remove_dir_all(path).unwrap();
}
