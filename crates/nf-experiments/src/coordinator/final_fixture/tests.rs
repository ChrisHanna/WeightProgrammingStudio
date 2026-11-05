use super::*;
fn envelope() -> (Lease, Resources) {
    let now = files::moment().unwrap();
    (
        Lease {
            store: "test".into(),
            contract: profile_id(),
            owner: "test".into(),
            epoch: 1,
            boot: now.boot,
            deadline: now.ms + 60_000,
        },
        Resources {
            cpu_ms: 60_000,
            host_bytes: u64::MAX,
            ..Default::default()
        },
    )
}
fn prospective(db: &mut Connection) -> Transaction<'_> {
    db.execute_batch("CREATE TABLE decision(value INTEGER)")
        .unwrap();
    let tx = db.transaction().unwrap();
    tx.execute("INSERT INTO decision VALUES(1)", []).unwrap();
    tx
}
#[test]
fn acceptance_gate_finalization_cancel_after_early_sample_rolls_back() {
    let (lease, resources) = envelope();
    let cancel = CancellationToken::default();
    let cpu = cpu_ns();
    check_completion(&lease, lease.deadline, cpu, &resources, Some(&cancel)).unwrap();
    let mut db = Connection::open_in_memory().unwrap();
    let tx = prospective(&mut db);
    // Deterministic barrier at the actual helper used after serialization and
    // insertion, rather than a timing-sensitive sleep/poll of the whole study.
    let barrier = Arc::new(std::sync::Barrier::new(2));
    let stop = cancel.clone();
    let other = barrier.clone();
    let thread = std::thread::spawn(move || {
        stop.request();
        other.wait();
    });
    barrier.wait();
    assert!(commit_decision(tx, &lease, lease.deadline, cpu, &resources, &cancel).is_err());
    thread.join().unwrap();
    assert_eq!(
        db.query_row("SELECT count(*) FROM decision", [], |r| r.get::<_, u64>(0))
            .unwrap(),
        0
    );
}
#[test]
fn acceptance_gate_finalization_deadline_after_early_sample_rolls_back() {
    let (lease, resources) = envelope();
    let cpu = cpu_ns();
    check_completion(&lease, lease.deadline, cpu, &resources, None).unwrap();
    let mut db = Connection::open_in_memory().unwrap();
    let tx = prospective(&mut db);
    let expired = files::moment().unwrap().ms;
    assert!(commit_decision(
        tx,
        &lease,
        expired,
        cpu,
        &resources,
        &CancellationToken::default()
    )
    .is_err());
    assert_eq!(
        db.query_row("SELECT count(*) FROM decision", [], |r| r.get::<_, u64>(0))
            .unwrap(),
        0
    );
    // Post-commit tail uses the SAME deadline check before publishing authority.
    assert!(check_completion(&lease, expired, cpu, &resources, None).is_err());
}
#[test]
fn acceptance_gate_provisional_decision_does_not_override_late_cancellation() {
    let (lease, resources) = envelope();
    let cancel = CancellationToken::default();
    let mut db = Connection::open_in_memory().unwrap();
    let tx = prospective(&mut db);
    commit_decision(tx, &lease, lease.deadline, cpu_ns(), &resources, &cancel).unwrap();
    cancel.request();
    assert_eq!(
        db.query_row("SELECT count(*) FROM decision", [], |r| r.get::<_, u64>(0))
            .unwrap(),
        1
    );
    // A provisional decision remains retained, but is not authority. The actual
    // completion transaction checks cancellation again before publication.
    assert!(check_completion(&lease, lease.deadline, cpu_ns(), &resources, Some(&cancel)).is_err());
}
#[test]
fn acceptance_gate_finalization_cpu_envelope_is_not_lease_only() {
    let (lease, mut resources) = envelope();
    resources.cpu_ms = 0;
    assert!(check_completion(&lease, lease.deadline, 0, &resources, None).is_err());
}
#[test]
fn acceptance_gate_image_scratch_cap_is_enforced_before_file_creation() {
    let db = Connection::open_in_memory().unwrap();
    db.execute_batch("CREATE TABLE payload(bytes BLOB)")
        .unwrap();
    bounds::cap_database(&db, 4 * 1024 * 1024).unwrap();
    bounds::image_scratch_cap(&db).unwrap();
    let cap: u64 = db
        .query_row(
            "SELECT page_size*max_page_count FROM pragma_page_size(),pragma_max_page_count()",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(cap <= 1024 * 1024 + 384 * 1024);
    assert!(supervised::image_storage_bytes() <= 1024 * 1024);
    assert!(db
        .execute("INSERT INTO payload VALUES(zeroblob(2097152))", [])
        .is_err());
    let large = Connection::open_in_memory().unwrap();
    large
        .execute_batch(
            "CREATE TABLE payload(bytes BLOB); INSERT INTO payload VALUES(zeroblob(2097152))",
        )
        .unwrap();
    assert!(bounds::image_scratch_cap(&large).is_err());
}
#[test]
fn acceptance_gate_fixture_sqlite_cap_prevents_write_before_growth() {
    let db = Connection::open_in_memory().unwrap();
    db.execute_batch("CREATE TABLE payload(bytes BLOB)")
        .unwrap();
    bounds::cap_database(&db, 128 * 1024).unwrap();
    assert!(db
        .execute("INSERT INTO payload VALUES(zeroblob(2097152))", [])
        .is_err());
    let size: u64 = db
        .query_row(
            "SELECT page_size*page_count FROM pragma_page_size(),pragma_page_count()",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(size <= 128 * 1024);
}
