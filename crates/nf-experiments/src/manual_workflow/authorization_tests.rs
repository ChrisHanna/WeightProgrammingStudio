//! Private OS-authority seam tests: production always pins uid 0 and /etc.
//! No test path/uid override is compiled into the public service.
use super::*;
use std::os::unix::fs::PermissionsExt;
use std::sync::atomic::{AtomicU64, Ordering};
#[path = "authority_service_tests.rs"]
mod authority_service_tests;
#[path = "../../../nf-core/tests/common/study_fixture.rs"]
#[allow(dead_code)]
mod fixture;
#[path = "supplied_tests.rs"]
mod supplied_tests;
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Setup {
    root: PathBuf,
    authority: Authority,
    request: ExistingRequest,
    contract: Validated<StudyContract>,
}
impl Setup {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "nf-t041-grant-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).unwrap();
        std::fs::create_dir(root.join("authority")).unwrap();
        let authority = Authority {
            directory: File::open(root.join("authority")).unwrap(),
            uid: unsafe { libc::getuid() },
        };
        let mut c = fixture::standard();
        c["limits"]["host_memory_bytes"] = json!(268435456);
        let contract = Validated::<StudyContract>::parse(&serde_json::to_vec(&c).unwrap()).unwrap();
        ArtifactStore::create(&root.join("store"), store_limits()).unwrap();
        let mut ledger = Ledger::open(&root.join("ledger"), LedgerLimits::default()).unwrap();
        let id = ledger.freeze(&contract).unwrap();
        let request = ExistingRequest {
            schema_version: 1,
            store: root.join("store"),
            ledger: root.join("ledger"),
            contract_digest: id,
            authorization: None,
            workers: vec![],
        };
        Self {
            root,
            authority,
            request,
            contract,
        }
    }
    fn grant(&mut self, expires: u64) -> Grant {
        let binary = self.root.join("fixture-worker");
        if !binary.exists() {
            let target = std::env::current_exe()
                .unwrap()
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .join("nf-fixture-worker");
            assert!(std::process::Command::new("/usr/bin/strip")
                .args(["--strip-debug", "-o"])
                .arg(&binary)
                .arg(target)
                .status()
                .unwrap()
                .success());
            std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o500)).unwrap();
        }
        Grant {
            schema_version: 1,
            operation: "nf.native.complete_fixture.v1".into(),
            supplied: None,
            store: identity(&self.request.store).unwrap(),
            ledger: identity(&self.request.ledger).unwrap(),
            database: identity(&self.request.ledger.join("events.sqlite3")).unwrap(),
            contract: self.request.contract_digest.clone(),
            permissions: canonical(&self.contract.permissions).unwrap().digest(),
            worker_digest: ContentDigest::of_bytes(&std::fs::read(&binary).unwrap()),
            worker: binary,
            expires_unix_seconds: expires,
        }
    }
    fn install(&mut self, grant: &Grant) {
        let doc = canonical(grant).unwrap();
        self.authority
            .file(doc.digest().as_str(), true)
            .unwrap()
            .write_all(doc.bytes())
            .unwrap();
        self.request.authorization = Some(doc.digest());
    }
}
impl Drop for Setup {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}
#[test]
fn manual_workflow_trusted_authority_dispatches_real_local_fixture_and_retains_costs() {
    let mut s = Setup::new();
    let grant = s.grant(now().unwrap() + 60);
    let issue = json!({"request":{"schema_version":1,"store":s.request.store,
        "ledger":s.request.ledger,"contract_digest":s.request.contract_digest},
        "worker":grant.worker,"worker_digest":grant.worker_digest,
        "expires_unix_seconds":grant.expires_unix_seconds});
    let issued = operator_with_authority(
        "authorize",
        &serde_json::to_vec(&issue).unwrap(),
        &s.authority,
    )
    .unwrap();
    s.request.authorization =
        Some(serde_json::from_value(issued["authorization"].clone()).unwrap());
    assert_eq!(
        s.request.authorization,
        Some(canonical(&grant).unwrap().digest())
    );
    let result = run_with(&s.request, &s.authority).unwrap();
    assert_eq!(result["authorization_verified"], true);
    assert_eq!(result["execution"], "native_complete_fixture");
    assert_eq!(result["report"]["outcome"], "completed");
    assert_eq!(result["report"]["cleanup_confirmed"], true);
    assert!(!result["report"]["stdout"].as_array().unwrap().is_empty());
    assert_eq!(result["supplied_ir_executed"], false);
    assert_eq!(result["promotion"], false);
    assert_eq!(result["strict_publication"], false);
    let mut ledger = s.request.open().unwrap();
    let worker: ContentDigest = serde_json::from_value(result["worker"].clone()).unwrap();
    assert!(
        ledger
            .worker_state(&worker)
            .unwrap()
            .report
            .unwrap()
            .cleanup_confirmed
    );
    let before = ledger.budget_snapshot(&s.request.contract_digest).unwrap();
    assert!(
        run_with(&s.request, &s.authority).is_err(),
        "one-shot operation must not silently rerun"
    );
    let after = ledger.budget_snapshot(&s.request.contract_digest).unwrap();
    let before = serde_json::to_value(before).unwrap();
    let after = serde_json::to_value(after).unwrap();
    for key in ["spent", "held", "unknown_charged", "reservation_count"] {
        assert_eq!(before[key], after[key]);
    }
    ledger
        .request_stop(
            &s.request.contract_digest,
            fixture::d("stop"),
            StopReason::OwnerStop,
        )
        .unwrap();
    assert!(run_with(&s.request, &s.authority).is_err());
}
#[test]
fn manual_workflow_grants_refuse_forgery_wrong_store_permissions_expiry_and_revocation() {
    let mut s = Setup::new();
    let grant = s.grant(now().unwrap() + 60);
    s.install(&grant);
    s.authority.verify(&s.request, &s.contract).unwrap();
    let original = s.request.authorization.clone();
    s.request.authorization = Some(fixture::d("caller forged digest"));
    assert!(s.authority.verify(&s.request, &s.contract).is_err());
    s.request.authorization = original;
    // Copying an otherwise valid authority record into the CAS grants nothing.
    let fake = s.root.join("fake-authority");
    std::fs::create_dir(&fake).unwrap();
    let real_store = s.request.store.clone();
    s.request.store = fake;
    assert!(s.authority.verify(&s.request, &s.contract).is_err());
    s.request.store = real_store;
    let mut changed = serde_json::to_value(&*s.contract).unwrap();
    changed["permissions"]["allow_gradient_inspection"] = json!(true);
    changed["permissions"]["require_fully_gradient_free"] = json!(false);
    let changed =
        Validated::<StudyContract>::parse(&serde_json::to_vec(&changed).unwrap()).unwrap();
    assert!(s.authority.verify(&s.request, &changed).is_err());
    let id = s.request.authorization.clone().unwrap();
    let record = s.root.join("authority").join(id.as_str());
    std::fs::set_permissions(&record, std::fs::Permissions::from_mode(0o666)).unwrap();
    assert!(s.authority.verify(&s.request, &s.contract).is_err());
    std::fs::set_permissions(&record, std::fs::Permissions::from_mode(0o644)).unwrap();
    let revoke = json!({"schema_version":1,"store":s.request.store,"ledger":s.request.ledger,
        "contract_digest":s.request.contract_digest,"authorization":id});
    let revoke = serde_json::to_vec(&revoke).unwrap();
    let revoked = operator_with_authority("revoke", &revoke, &s.authority).unwrap();
    assert_eq!(revoked["revoked"], true);
    assert_eq!(
        revoked,
        operator_with_authority("revoke", &revoke, &s.authority).unwrap()
    );
    assert!(run_with(&s.request, &s.authority).is_err());
    let expired = s.grant(now().unwrap() - 1);
    s.install(&expired);
    assert!(run_with(&s.request, &s.authority).is_err());
    assert_eq!(
        s.request
            .open()
            .unwrap()
            .state(&s.request.contract_digest)
            .unwrap()
            .stage,
        crate::coordinator::Stage::Frozen
    );
}
#[test]
fn manual_workflow_production_operator_boundary_rejects_researcher_uid_and_untrusted_tree() {
    if unsafe { libc::getuid() } != 0 {
        assert_eq!(
            operator("authorize", b"{}").unwrap_err().code,
            ErrorCode::PermissionDenied
        );
        assert_eq!(
            operator("revoke", b"{}").unwrap_err().code,
            ErrorCode::PermissionDenied
        );
    }
    let s = Setup::new();
    assert!(Authority::open(&s.root.join("authority"), 0).is_err());
    // A symlink cannot select even a correctly owned authority directory.
    std::os::unix::fs::symlink("/etc", s.root.join("link")).unwrap();
    assert!(Authority::open(&s.root.join("link"), unsafe { libc::getuid() }).is_err());
}
#[test]
fn manual_workflow_operator_issuer_only_hashes_code_and_rejects_digest_substitution() {
    let mut s = Setup::new();
    let grant = s.grant(now().unwrap() + 60);
    let script = s.root.join("operator-must-not-execute");
    let marker = s.root.join("executed-as-operator");
    std::fs::write(
        &script,
        format!("#!/bin/sh\ntouch '{}'\n", marker.display()),
    )
    .unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o500)).unwrap();
    let digest = ContentDigest::of_bytes(&std::fs::read(&script).unwrap());
    let mut issue = json!({"request":{"schema_version":1,"store":s.request.store,
        "ledger":s.request.ledger,"contract_digest":s.request.contract_digest},
        "worker":script,"worker_digest":grant.worker_digest,
        "expires_unix_seconds":grant.expires_unix_seconds});
    assert!(operator_with_authority(
        "authorize",
        &serde_json::to_vec(&issue).unwrap(),
        &s.authority
    )
    .is_err());
    issue["worker_digest"] = json!(digest);
    assert!(operator_with_authority(
        "authorize",
        &serde_json::to_vec(&issue).unwrap(),
        &s.authority
    )
    .is_ok());
    assert!(
        !marker.exists(),
        "privileged issuer must never probe researcher code"
    );
}
