//! Storage-only distinct-operation CAS. Native positive behavior is tested separately.
use super::*;

#[test]
fn competing_operations_cannot_both_publish_from_the_same_family_head() {
    let path = std::env::temp_dir().join(format!("promotion-head-race-{}", std::process::id()));
    let mut first = Ledger::open(&path, LedgerLimits::default()).unwrap();
    let base =
        Validated::<StudyContract>::parse(&serde_json::to_vec(&fixture::standard()).unwrap())
            .unwrap();
    let a = first.freeze(&base).unwrap();
    let advance = |ledger: &mut Ledger, lease: &Lease| {
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
                    lease,
                    Transition {
                        operation: fixture::d(&format!("head-race-{i}")),
                        expected: previous,
                        next,
                        protocol: fixture::d("storage-race"),
                        evidence: fixture::d("storage-race"),
                        message: "storage-only competing publication".into(),
                    },
                )
                .unwrap();
            previous = next;
        }
    };
    let first_lease = first.acquire(&a, "first", 300000).unwrap();
    advance(&mut first, &first_lease);
    let leases = [first_lease.clone(), first_lease];
    let mut second = Ledger::open(&path, LedgerLimits::default()).unwrap();
    let barrier = std::sync::Barrier::new(2);
    let family = &base.study_family;
    let attempt = |ledger: &mut Ledger, lease: &Lease, operation: &str| -> Result<ContentDigest> {
        barrier.wait();
        let store = ledger.store.clone();
        let tx = ledger
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sql)?;
        check_lease(&tx, &store, lease, &files::moment()?)?;
        let body = json!({"operation":fixture::d(operation),"contract":lease.contract(),"predecessor":null});
        let digest = hash(&body)?;
        publish(&tx, lease, family, &None, &digest, &body, 100)?;
        tx.commit().map_err(sql)?;
        Ok(digest)
    };
    let (a_result, b_result) = std::thread::scope(|scope| {
        let worker = scope.spawn(|| attempt(&mut second, &leases[1], "second-publication"));
        (
            attempt(&mut first, &leases[0], "first-publication"),
            worker.join().unwrap(),
        )
    });
    assert_ne!(
        a_result.is_ok(),
        b_result.is_ok(),
        "exactly one distinct-operation CAS must win"
    );
    let winner = a_result.or(b_result).unwrap();
    assert_eq!(head(&first.connection, family).unwrap(), Some(winner));
    for table in ["accepted_heads", "accepted_records", "verified_frontier"] {
        assert_eq!(
            first
                .connection
                .query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r
                    .get::<_, u64>(0))
                .unwrap(),
            1
        );
    }
    assert_eq!(first.state(&a).unwrap().stage, Stage::Completed);
    assert_eq!(first.events(&a, 0, 100).unwrap().len(), 9);
    drop(first);
    drop(second);
    std::fs::remove_dir_all(path).unwrap();
}
