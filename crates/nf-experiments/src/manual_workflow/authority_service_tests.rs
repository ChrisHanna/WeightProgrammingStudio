//! Isolated operator-principal seam, NOT root-authentication/provisioning proof.
use super::super::authority_service as service;
use super::*;
use std::os::unix::net::UnixStream;
fn issue(s: &mut Setup) -> Value {
    let grant = s.grant(now().unwrap() + 180);
    json!({"request":s.request,"worker":grant.worker,"worker_digest":grant.worker_digest,
        "expires_unix_seconds":grant.expires_unix_seconds})
}
fn install(s: &mut Setup, issue: &Value) -> ContentDigest {
    let issued =
        operator_with_authority("authorize", canonical(issue).unwrap().bytes(), &s.authority)
            .unwrap();
    let id = serde_json::from_value(issued["authorization"].clone()).unwrap();
    s.request.authorization = Some(id);
    s.request.authorization.clone().unwrap()
}
#[test]
fn registry_crash_child_entry() {
    let mut bytes = Vec::new();
    std::io::stdin().read_to_end(&mut bytes).unwrap();
    if bytes.is_empty() {
        return;
    }
    let input: Value = decode(&bytes).unwrap();
    let directory = PathBuf::from(input["directory"].as_str().unwrap());
    let authority = Authority {
        directory: File::open(directory).unwrap(),
        uid: unsafe { libc::getuid() },
    };
    service::test_crash(match input["point"].as_str().unwrap() {
        "partial_write" => "partial_write",
        "after_publish" => "after_publish",
        _ => panic!("test fault"),
    });
    if input["operation"] == "consume" {
        service::consume_local(
            &decode(canonical(&input["request"]).unwrap().bytes()).unwrap(),
            &authority,
        )
        .unwrap();
    } else {
        operator_with_authority(
            input["operation"].as_str().unwrap(),
            canonical(&input["request"]).unwrap().bytes(),
            &authority,
        )
        .unwrap();
    }
    panic!("crash checkpoint not reached");
}
#[test]
fn registry_actual_process_crashes_leave_staging_inert_and_consumption_monotonic() {
    fn crash(s: &Setup, operation: &str, request: &Value, point: &str) {
        use std::process::{Command, Stdio};
        let mut child = Command::new(std::env::current_exe().unwrap()).args(["--exact",
            "manual_workflow::authorization::tests::authority_service_tests::registry_crash_child_entry", "--test-threads=1"])
            .stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::null()).spawn().unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(
                canonical(&json!({"directory":s.root.join("authority"),
            "operation":operation,"request":request,"point":point}))
                .unwrap()
                .bytes(),
            )
            .unwrap();
        assert_eq!(child.wait().unwrap().code(), Some(74));
    }
    let mut s = Setup::new();
    let issue = issue(&mut s);
    crash(&s, "authorize", &issue, "partial_write");
    assert!(std::fs::read_dir(s.root.join("authority"))
        .unwrap()
        .any(|e| e
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".stage-")));
    let id = install(&mut s, &issue); // Orphan staging cannot authorize or block this issuance.
    s.authority.verify(&s.request, &s.contract).unwrap();
    crash(
        &s,
        "consume",
        &canonical(&s.request).unwrap().value().clone(),
        "after_publish",
    );
    assert!(service::consume_local(&s.request, &s.authority).is_err());
    crash(
        &s,
        "revoke",
        &canonical(&s.request).unwrap().value().clone(),
        "after_publish",
    );
    operator_with_authority(
        "revoke",
        canonical(&s.request).unwrap().bytes(),
        &s.authority,
    )
    .unwrap();
    assert!(s.authority.revoked(&id).unwrap());
    assert!(s.authority.verify(&s.request, &s.contract).is_err());
}
#[test]
fn registry_publication_crashes_never_publish_partial_grants_or_replace_existing() {
    for point in ["partial_write", "before_publish", "after_publish"] {
        let mut s = Setup::new();
        let issue = issue(&mut s);
        service::test_fault(Some(point));
        assert!(operator_with_authority(
            "authorize",
            canonical(&issue).unwrap().bytes(),
            &s.authority
        )
        .is_err());
        service::test_fault(None);
        let records = std::fs::read_dir(s.root.join("authority"))
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| {
                p.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("sha256:")
                    && p.extension().is_none()
            })
            .collect::<Vec<_>>();
        if point == "after_publish" {
            assert_eq!(records.len(), 1);
            let bytes = std::fs::read(&records[0]).unwrap();
            let doc = CanonicalDocument::parse(&bytes, Limits::default()).unwrap();
            s.request.authorization = Some(doc.digest());
            s.authority.verify(&s.request, &s.contract).unwrap();
            assert!(operator_with_authority(
                "authorize",
                canonical(&issue).unwrap().bytes(),
                &s.authority
            )
            .is_err());
            assert_eq!(std::fs::read(&records[0]).unwrap(), bytes);
        } else {
            assert!(records.is_empty());
            install(&mut s, &issue);
        }
        assert!(!std::fs::read_dir(s.root.join("authority"))
            .unwrap()
            .any(|e| e
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".stage-")));
    }
}
#[test]
fn registry_revocation_unknown_outcome_is_retryable_and_never_resurrects_grant() {
    let mut s = Setup::new();
    let issue = issue(&mut s);
    let id = install(&mut s, &issue);
    let before = std::fs::read(s.root.join("authority").join(id.as_str())).unwrap();
    service::test_fault(Some("after_publish"));
    assert!(operator_with_authority(
        "revoke",
        canonical(&s.request).unwrap().bytes(),
        &s.authority
    )
    .is_err());
    service::test_fault(None);
    assert!(s.authority.verify(&s.request, &s.contract).is_err());
    assert!(operator_with_authority(
        "revoke",
        canonical(&s.request).unwrap().bytes(),
        &s.authority
    )
    .is_ok());
    assert!(service::consume_local(&s.request, &s.authority).is_err());
    assert_eq!(
        std::fs::read(s.root.join("authority").join(id.as_str())).unwrap(),
        before
    );
}
#[test]
fn registry_consumption_survives_same_inode_sqlite_rollback_and_uncertain_response() {
    let mut s = Setup::new();
    let issue = issue(&mut s);
    let id = install(&mut s, &issue);
    let path = s.request.ledger.join("events.sqlite3");
    let checkpoint = || {
        let db = rusqlite::Connection::open(&path).unwrap();
        db.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)").unwrap();
    };
    checkpoint();
    let bytes = std::fs::read(&path).unwrap();
    let inode = identity(&path).unwrap();
    service::test_fault(Some("after_publish"));
    assert!(service::consume_local(&s.request, &s.authority).is_err());
    service::test_fault(None);
    dispatch("stop", canonical(&s.request).unwrap().bytes()).unwrap();
    checkpoint();
    std::fs::write(&path, bytes).unwrap();
    assert!(identity(&path).unwrap() == inode);
    assert!(s
        .request
        .open()
        .unwrap()
        .stop_record(&s.request.contract_digest)
        .unwrap()
        .is_none());
    s.authority.verify(&s.request, &s.contract).unwrap(); // Still live and same scope.
    let reopened = Authority {
        directory: File::open(s.root.join("authority")).unwrap(),
        uid: unsafe { libc::getuid() },
    };
    assert!(
        service::consume_local(&s.request, &reopened).is_err(),
        "rollback must not restore launch authority"
    );
    assert!(s
        .root
        .join("authority")
        .join(format!("{}.consumed", id.as_str()))
        .is_file());
}
#[test]
fn registry_local_ipc_consumes_once_and_authenticates_actual_peer_uid() {
    if unsafe { libc::getuid() } == 0 {
        return;
    }
    let mut s = Setup::new();
    let issue = issue(&mut s);
    install(&mut s, &issue);
    for expected in [true, false] {
        let (client, server) = UnixStream::pair().unwrap();
        let authority = Authority {
            directory: s.authority.directory.try_clone().unwrap(),
            uid: s.authority.uid,
        };
        let task = std::thread::spawn(move || service::handle(server, &authority));
        assert_eq!(
            service::exchange(client, &s.request, unsafe { libc::getuid() }).is_ok(),
            expected
        );
        assert_eq!(task.join().unwrap().is_ok(), expected);
    }
}
#[test]
fn registry_incomplete_ipc_is_bounded_and_does_not_consume_authority() {
    if unsafe { libc::getuid() } == 0 {
        return;
    }
    let mut s = Setup::new();
    let issue = issue(&mut s);
    install(&mut s, &issue);
    let (mut client, server) = UnixStream::pair().unwrap();
    let authority = Authority {
        directory: s.authority.directory.try_clone().unwrap(),
        uid: s.authority.uid,
    };
    let task = std::thread::spawn(move || service::handle(server, &authority));
    let start = std::time::Instant::now();
    client.write_all(b"{").unwrap();
    assert!(task.join().unwrap().is_err());
    assert!(start.elapsed() < std::time::Duration::from_secs(5));
    service::consume_local(&s.request, &s.authority).unwrap();
}
#[test]
fn registry_client_rejects_same_uid_impersonating_root_before_sending_request() {
    if unsafe { libc::getuid() } == 0 {
        return;
    }
    let s = Setup::new();
    let (client, mut impostor) = UnixStream::pair().unwrap();
    assert!(service::exchange(client, &s.request, 0).is_err());
    let mut bytes = Vec::new();
    impostor.read_to_end(&mut bytes).unwrap();
    assert!(bytes.is_empty());
}
#[test]
fn registry_revoke_and_consume_share_one_stable_nonblocking_lock() {
    let mut s = Setup::new();
    let issue = issue(&mut s);
    let id = install(&mut s, &issue);
    let lock_path = s
        .root
        .join("authority")
        .join(format!("{}.lock", id.as_str()));
    assert_eq!(std::fs::metadata(&lock_path).unwrap().mode() & 0o077, 0);
    std::fs::set_permissions(&lock_path, std::fs::Permissions::from_mode(0o644)).unwrap();
    assert!(
        s.authority.lock(&id, false).is_err(),
        "researcher-readable locks cannot protect revocation"
    );
    std::fs::set_permissions(&lock_path, std::fs::Permissions::from_mode(0o600)).unwrap();
    let lock = s.authority.lock(&id, false).unwrap();
    assert!(service::consume_local(&s.request, &s.authority).is_err());
    assert!(operator_with_authority(
        "revoke",
        canonical(&s.request).unwrap().bytes(),
        &s.authority
    )
    .is_err());
    assert!(
        !s.authority.revoked(&id).unwrap(),
        "busy is NOT successful revocation"
    );
    drop(lock);
    operator_with_authority(
        "revoke",
        canonical(&s.request).unwrap().bytes(),
        &s.authority,
    )
    .unwrap();
    assert!(service::consume_local(&s.request, &s.authority).is_err());
}
#[test]
fn controller_return_fence_rejects_late_withdrawal_and_persists_stop() {
    for cancelled in [false, true] {
        let mut s = Setup::new();
        let issue = issue(&mut s);
        install(&mut s, &issue);
        let cancel = nf_surgeon::cancellation::CancellationToken::default();
        if cancelled {
            cancel.request();
        } else {
            operator_with_authority(
                "revoke",
                canonical(&s.request).unwrap().bytes(),
                &s.authority,
            )
            .unwrap();
        }
        assert!(finish_authorized(
            &s.request,
            &s.authority,
            &s.contract,
            &cancel,
            Ok(json!({"ok":true}))
        )
        .is_err());
        assert!(s
            .request
            .open()
            .unwrap()
            .stop_record(&s.request.contract_digest)
            .unwrap()
            .is_some());
    }
}
#[test]
fn registry_per_thread_capability_parser_fails_closed() {
    let clean = "Uid:\t1000\t1000\t1000\t1000\nCapPrm:\t0000000000000000\nCapEff:\t0000000000000000\nCapAmb:\t0000000000000000\n";
    service::unprivileged_status(clean).unwrap();
    for field in ["CapPrm:", "CapEff:", "CapAmb:"] {
        assert!(service::unprivileged_status(&clean.replace(
            &format!("{field}\t0000000000000000"),
            &format!("{field}\t0000000000000001")
        ))
        .is_err());
    }
    assert!(service::unprivileged_status(
        &clean.replace("1000\t1000\t1000\t1000", "1000\t1000\t0\t1000")
    )
    .is_err());
    assert!(service::unprivileged_status("CapEff: 0").is_err());
    if unsafe { libc::getuid() } != 0 {
        service::consumer().unwrap();
        assert_eq!(
            dispatch("serve-authority", b"{\"schema_version\":1}")
                .unwrap_err()
                .code,
            ErrorCode::PermissionDenied
        );
    }
}
