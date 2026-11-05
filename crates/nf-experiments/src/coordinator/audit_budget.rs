//! Single-use native reference-audit authority backed by the family ledger.
//! No JSON DTO implements this authority. Claims survive process loss; a retry
//! needs a new sequential reservation, after conservative reconciliation.
use super::*;
use crate::validation::reference::{self, Budget};
#[cfg(test)]
#[path = "audit_budget/tests.rs"]
mod enforcement_tests;
#[path = "audit_budget/lifecycle.rs"]
mod lifecycle;
#[path = "audit_budget/memory.rs"]
mod memory;
#[cfg(test)]
#[path = "audit_budget/recovery_tests.rs"]
mod recovery_tests;
#[path = "audit_budget/schema.rs"]
mod schema;
pub(crate) use schema::claim_execution;
pub(crate) use schema::{migrate_v8, schema_v8, verify_v8};
/// 128 MiB guest cgroup plus 256 MiB dedicated controller hard address space.
/// Includes controller baseline/qualification; excludes Docker daemon/CLI helpers
/// and unrelated host processes. This is not a whole-host/process-tree cap.
pub const AUDIT_HOST_BYTES: u64 =
    crate::sandbox::MEMORY_BYTES + crate::sandbox::CONTROLLER_MEMORY_ALLOWANCE_BYTES;
// Test fixtures must remove the entire newer schema before claiming an older
// version. Never expose a production downgrade or runtime schema repair path.
#[cfg(test)]
pub(crate) const STRIP_V8: &str = "DROP TABLE audit_cleanup;DROP TABLE audit_runtime_bindings;DROP TABLE execution_owners;DROP TABLE audit_storage_claims;DELETE FROM budget_migrations WHERE version=8;";
#[cfg(test)]
#[path = "../../../nf-core/tests/common/study_fixture.rs"]
#[allow(dead_code)]
mod fixture;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::{fs, os::unix::fs::PermissionsExt, path::PathBuf};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    pub(crate) struct Harness {
        pub root: PathBuf,
        pub ledger: Ledger,
        pub lease: Lease,
        pub reservation: Reservation,
        pub request: Request,
        pub budget: Budget,
    }
    impl Harness {
        pub(crate) fn at(root: PathBuf) -> Self {
            fs::create_dir(&root).unwrap();
            fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
            let mut value = fixture::standard();
            value["limits"]["storage_bytes"] = serde_json::json!(512 * 1024 * 1024);
            value["limits"]["host_memory_bytes"] = serde_json::json!(512 * 1024 * 1024);
            let contract = nf_core::schemas::Validated::<StudyContract>::parse(
                &serde_json::to_vec(&value).unwrap(),
            )
            .unwrap();
            let mut ledger = Ledger::open(&root.join("ledger"), LedgerLimits::default()).unwrap();
            let id = ledger.freeze(&contract).unwrap();
            let lease = ledger.acquire(&id, "audit-fixture", 300000).unwrap();
            let budget = Budget {
                max_executions: 2,
                cpu_ms: 4000,
                wall_ms: 15000,
                storage_bytes: 64 * 1024 * 1024,
            };
            let request = Request {
                schema_version: 1,
                operation: fixture::d("audit"),
                attempt: 0,
                parent: None,
                kind: WorkKind::Inspection,
                external: None,
                wall_ms: 20000,
                resources: Resources {
                    cpu_ms: 5000,
                    storage_bytes: budget.storage_bytes,
                    workers: 1,
                    host_bytes: AUDIT_HOST_BYTES,
                    ..Resources::default()
                },
            };
            let reservation = ledger.reserve(&lease, request.clone()).unwrap();
            Self {
                root,
                ledger,
                lease,
                reservation,
                request,
                budget,
            }
        }
        pub(crate) fn new() -> Self {
            Self::at(std::env::temp_dir().join(format!(
                "nf-audit-authority-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            )))
        }
        pub(crate) fn claim(&mut self) {
            self.ledger
                .claim_audit(&self.lease, &self.reservation, &self.budget)
                .unwrap();
        }
        pub(crate) fn reopen(&mut self) {
            self.ledger = enforcement_tests::open_component(&self.root.join("ledger"));
        }
        pub(crate) fn held(&self) -> u64 {
            self.ledger
                .budget_snapshot(&self.lease.contract)
                .unwrap()
                .held
                .storage_bytes
        }
    }
    impl Drop for Harness {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.root).unwrap();
        }
    }

    #[test]
    fn crash_reopen_retry_and_unknown_settlement_never_reset_storage() {
        let mut h = Harness::new();
        h.claim();
        let runtime = crate::sandbox::QualifiedDocker::qualify(true, &h.root).unwrap();
        h.ledger
            .bind_audit_runtime(&h.lease, &h.reservation, &runtime)
            .unwrap();
        let full = h.budget.storage_bytes;
        assert_eq!(h.held(), full);
        h.reopen();
        assert!(h
            .ledger
            .claim_audit(&h.lease, &h.reservation, &h.budget)
            .is_err());
        assert_eq!(h.held(), full);
        let mut retry = h.request.clone();
        retry.attempt = 1;
        assert!(h.ledger.reserve(&h.lease, retry.clone()).is_err());
        let mut settlement = Settlement {
            operation: fixture::d("cleanup"),
            outcome: OutcomeKind::Incomplete,
            usage: Usage::default(),
            resources_released: true,
            evidence: fixture::d("recovery-cleanup"),
        };
        settlement.usage.storage_bytes = Some(0);
        assert!(h
            .ledger
            .settle(&h.lease, &h.reservation, settlement.clone())
            .is_err());
        settlement.usage.storage_bytes = None;
        assert!(h
            .ledger
            .settle(&h.lease, &h.reservation, settlement.clone())
            .is_err());
        h.ledger
            .confirm_audit_cleanup(&h.lease, &h.reservation, &runtime, &settlement.evidence)
            .unwrap();
        h.ledger
            .settle(&h.lease, &h.reservation, settlement)
            .unwrap();
        let snapshot = h.ledger.budget_snapshot(&h.lease.contract).unwrap();
        assert_eq!(snapshot.spent.storage_bytes, full);
        assert_eq!(snapshot.unknown_charged.storage_bytes, full);
        let reservation = h.ledger.reserve(&h.lease, retry).unwrap();
        h.ledger
            .claim_audit(&h.lease, &reservation, &h.budget)
            .unwrap();
        h.reopen();
        let snapshot = h.ledger.budget_snapshot(&h.lease.contract).unwrap();
        assert_eq!(
            snapshot.spent.storage_bytes + snapshot.held.storage_bytes,
            2 * full
        );
        assert!(h
            .ledger
            .claim_audit(&h.lease, &reservation, &h.budget)
            .is_err());
    }

    #[test]
    fn insufficient_foreign_and_stop_deny_before_claim() {
        let mut h = Harness::new();
        let mut too_big = h.budget.clone();
        too_big.storage_bytes += 1;
        assert!(h
            .ledger
            .claim_audit(&h.lease, &h.reservation, &too_big)
            .is_err());
        let mut too_small = h.budget.clone();
        too_small.storage_bytes = 1;
        assert!(h
            .ledger
            .claim_audit(&h.lease, &h.reservation, &too_small)
            .is_err());
        let other = Harness::new();
        assert!(h
            .ledger
            .claim_audit(&h.lease, &other.reservation, &h.budget)
            .is_err());
        assert!(!claimed(&h.ledger.connection, h.reservation.id()).unwrap());
        h.claim();
        h.ledger
            .request_stop(
                &h.lease.contract,
                fixture::d("stop"),
                super::super::super::supervision::StopReason::OwnerStop,
            )
            .unwrap();
        assert!(h
            .ledger
            .check_audit(&h.lease, &h.reservation, &h.budget)
            .is_err());
        h.reopen();
        assert!(h
            .ledger
            .check_audit(&h.lease, &h.reservation, &h.budget)
            .is_err());
        assert_eq!(h.held(), h.budget.storage_bytes);
    }

    // Actual OCI execution: no mock runtime and no host Python fallback.
    #[test]
    fn sufficient_native_budget_runs_real_reference_and_charges_unknown() {
        if enforcement_tests::bounded_controller_test() {
            return;
        }
        use crate::{
            intake::{self, Draft},
            sandbox::QualifiedDocker,
            validation::{
                reference::{Coverage, Exposure, Policy, Role},
                Plan,
            },
        };
        use nf_checkpoint::store::{ArtifactStore, StoreLimits};
        use serde_json::json;
        let mut h = Harness::new();
        let store = ArtifactStore::create(
            &h.root.join("cas"),
            StoreLimits {
                total_bytes: 16 * 1024 * 1024,
                max_blob_bytes: 1048576,
                max_entries: 100,
            },
        )
        .unwrap();
        let source = h.root.join("source");
        fs::create_dir(&source).unwrap();
        fs::write(source.join("gym.py"), "def evaluate(x,seed):\n allocation=bytearray(48*1024*1024)\n allocation[-1]=1\n return x*2\ndef oracle(x): return x*2\ndef score(x,out,reference): return float(out==reference)\n").unwrap();
        fs::write(source.join("LICENSE"), "fixture").unwrap();
        fs::write(source.join("data.json"), "[]").unwrap();
        let image = crate::sandbox::IMAGE.split('@').nth(1).unwrap();
        fs::write(
            source.join("lock.json"),
            json!({"schema_version":1,"image":image,"packages":{}}).to_string(),
        )
        .unwrap();
        let ep = |symbol| json!({"file":"gym.py","symbol":symbol});
        let integer = json!({"type":"integer","min":0,"max":1000});
        let draft = json!({"schema_version":1,"name":"fixture","version":"1","usage_rights":"fixture",
            "confidentiality":"confidential","assets":{
                "gym.py":{"kind":"code","expected":null},"LICENSE":{"kind":"license","expected":null},
                "lock.json":{"kind":"dependency_lock","expected":null},"data.json":{"kind":"data","expected":null}},
            "dependency_lock":"lock.json","licenses":["LICENSE"],
            "interface":{"kind":"single_shot","evaluate":ep("evaluate"),"score":ep("score"),"oracle":ep("oracle"),"input":integer,"output":integer},
            "dataset":{"kind":"frozen","files":["data.json"],"task_key":"task","group_key":"group"}});
        let gym = intake::snapshot(
            &source,
            &store,
            Draft::parse(draft.to_string().as_bytes()).unwrap(),
        )
        .unwrap();
        let plan = Plan::parse(json!({"schema_version":1,"snapshot":gym.id(),"truth_provenance":"independent arithmetic",
            "cases":[{"id":"two","input":2,"expected":4,"wrong":[3],"termination":null},
                     {"id":"seven","input":7,"expected":14,"wrong":[13],"termination":null}],"seeds":[11,22],
            "scoring":{"minimum":0.0,"maximum":1.0,"good_min":1.0,"bad_max":0.0},
            "limits":{"wall_ms":2000,"cpu_ms":2000,"output_bytes":32768}}).to_string().as_bytes()).unwrap();
        let audit = h
            .ledger
            .run_reference_audit(
                &h.lease,
                &h.reservation,
                &gym,
                &store,
                plan.clone(),
                &[Policy {
                    snapshot: &gym,
                    role: Role::Valid,
                }],
                Exposure::Development,
                h.budget.clone(),
                || QualifiedDocker::qualify(true, &h.root),
                Coverage::default(),
                &AtomicBool::new(false),
            )
            .unwrap();
        assert_eq!(audit.report().executions, 2);
        assert_eq!(audit.report().policies[0].correct, 8);
        assert!(
            audit.report().memory_peak_bytes >= 48 * 1024 * 1024,
            "real policy must exercise its admitted memory envelope"
        );
        assert!(audit.report().storage.durable_family_accounting);
        assert!(!audit.report().storage.capacity_guaranteed);
        assert_eq!(audit.report().storage.measured_study_storage_bytes, None);
        h.reopen();
        let snapshot = h.ledger.budget_snapshot(&h.lease.contract).unwrap();
        assert_eq!(snapshot.spent.storage_bytes, h.budget.storage_bytes);
        assert_eq!(
            snapshot.unknown_charged.storage_bytes,
            h.budget.storage_bytes
        );
        let called = AtomicBool::new(false);
        assert!(h
            .ledger
            .run_reference_audit(
                &h.lease,
                &h.reservation,
                &gym,
                &store,
                plan,
                &[Policy {
                    snapshot: &gym,
                    role: Role::Valid
                }],
                Exposure::Development,
                h.budget.clone(),
                || {
                    called.store(true, Ordering::Relaxed);
                    panic!("must deny before qualifier");
                },
                Coverage::default(),
                &AtomicBool::new(false)
            )
            .is_err());
        assert!(!called.load(Ordering::Relaxed));
    }
}

pub(super) fn claimed(connection: &Connection, id: &ContentDigest) -> Result<bool> {
    verify_v8(connection)?;
    connection
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM audit_storage_claims WHERE request=?1)",
            [id.as_str()],
            |r| r.get(0),
        )
        .map_err(sql)
}

/// DTOs cannot certify audit cleanup or refund any unmeasured resource.
pub(super) fn validate_settlement(
    connection: &Connection,
    id: &ContentDigest,
    value: &Settlement,
) -> Result<()> {
    if !claimed(connection, id)? {
        return Ok(());
    }
    verify_v8(connection)?;
    if value.usage != Usage::default() {
        return Err(failure(
            ErrorCode::PermissionDenied,
            "audit usage must remain unknown at the reserved ceiling",
        ));
    }
    if value.resources_released {
        let evidence: Option<String> = connection
            .query_row(
                "SELECT evidence FROM audit_cleanup WHERE request=?1",
                [id.as_str()],
                |r| r.get(0),
            )
            .optional()
            .map_err(sql)?;
        if evidence.as_deref() != Some(value.evidence.as_str()) {
            return Err(failure(
                ErrorCode::PermissionDenied,
                "audit release requires native verified cleanup",
            ));
        }
    }
    Ok(())
}

// Private, non-Clone, non-Deserialize, and never exposed through Python.
struct Authority {
    reservation: Reservation,
}

impl Ledger {
    fn check_audit(&self, lease: &Lease, reservation: &Reservation, budget: &Budget) -> Result<()> {
        self.files.verify(self.limits.max_database_bytes)?;
        let tx = self.connection.unchecked_transaction().map_err(sql)?;
        verify_v8(&tx)?;
        let backend: Option<String> = tx
            .query_row(
                "SELECT backend FROM execution_owners WHERE request=?1",
                [reservation.id().as_str()],
                |r| r.get(0),
            )
            .optional()
            .map_err(sql)?;
        if backend.as_deref().is_some_and(|value| value != "audit") {
            return Err(failure(
                ErrorCode::PermissionDenied,
                "execution owner is not audit",
            ));
        }
        let contract = check_lease(&tx, &self.store, lease, &files::moment()?)?;
        if super::super::supervision::stopped(&tx, &lease.contract)? {
            return Err(failure(
                ErrorCode::PermissionDenied,
                "persisted stop blocks reference staging",
            ));
        }
        let allocation = native_allocation(&tx, &self.store, lease, reservation)?;
        let nodes = load(&tx, &contract.study_family)?;
        let request = &nodes
            .get(reservation.id())
            .ok_or_else(|| failure(ErrorCode::SourceMismatch, "audit reservation absent"))?
            .stored
            .request;
        policy(&contract, request, state(&tx, &lease.contract)?.stage)?;
        if request.kind != WorkKind::Inspection
            || budget.storage_bytes > allocation.resources.storage_bytes
            || budget.cpu_ms > allocation.resources.cpu_ms
            || allocation.resources.host_bytes < AUDIT_HOST_BYTES
            || allocation.resources.workers != 1
            || allocation.resources.gpu_ms != 0
            || allocation.resources.device_bytes != 0
            || budget.wall_ms == 0
            || budget.cpu_ms == 0
            || budget.wall_ms > request.wall_ms
        {
            return Err(failure(
                ErrorCode::PermissionDenied,
                "audit exceeds native inspection allocation",
            ));
        }
        budget.validate_storage()?;
        Ok(())
    }

    fn claim_audit(
        &mut self,
        lease: &Lease,
        reservation: &Reservation,
        budget: &Budget,
    ) -> Result<Authority> {
        self.check_audit(lease, reservation, budget)?;
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sql)?;
        // Recheck after acquiring the write fence: waiting on SQLite must not
        // turn an expired lease or a winning stop into dispatch permission.
        check_lease(&tx, &self.store, lease, &files::moment()?)?;
        if super::super::supervision::stopped(&tx, &lease.contract)? {
            return Err(failure(
                ErrorCode::PermissionDenied,
                "stop won reference claim race",
            ));
        }
        native_allocation(&tx, &self.store, lease, reservation)?;
        claim_execution(&tx, reservation.id(), "audit")?;
        if claimed(&tx, reservation.id())? {
            return Err(failure(
                ErrorCode::IdempotencyConflict,
                "audit attempt already dispatched; reconcile before retry",
            ));
        }
        if tx
            .execute(
                "INSERT INTO audit_storage_claims VALUES(?1,?2,?3)",
                params![
                    reservation.id().as_str(),
                    budget.max_executions,
                    budget.storage_bytes
                ],
            )
            .map_err(sql)?
            != 1
        {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "audit claim insertion ignored",
            ));
        }
        let stored: (u64, u64) = tx
            .query_row(
                "SELECT jobs,allowance FROM audit_storage_claims WHERE request=?1",
                [reservation.id().as_str()],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .map_err(sql)?;
        if stored != (budget.max_executions, budget.storage_bytes) {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "audit claim readback differs",
            ));
        }
        ensure_live(lease)?;
        tx.commit().map_err(sql)?;
        Ok(Authority {
            reservation: reservation.clone(),
        })
    }

    /// Run an exposed audit under an existing live native inspection allocation.
    /// Storage is held before any audit CAS publication or sandbox staging.
    /// Requires a dedicated controller with preinstalled finite hard RLIMIT_AS.
    /// Import/preexisting CAS, Docker daemon and CLI helpers are outside this scope.
    /// The trusted qualifier is invoked only AFTER the durable rights check.
    /// On ANY error (including ENOSPC), the durable hold remains unreleased.
    /// Recovery must confirm cleanup before settling with Usage::default(); no
    /// whole-study physical measurement exists, so storage remains unknown.
    #[allow(clippy::too_many_arguments)]
    pub fn run_reference_audit(
        &mut self,
        lease: &Lease,
        reservation: &Reservation,
        snapshot: &crate::intake::Snapshot,
        store: &nf_checkpoint::store::ArtifactStore,
        plan: crate::validation::Plan,
        policies: &[reference::Policy<'_>],
        exposure: reference::Exposure,
        budget: Budget,
        qualify: impl FnOnce() -> Result<crate::sandbox::QualifiedDocker>,
        coverage: reference::Coverage,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Result<reference::Audit> {
        let started = std::time::Instant::now();
        self.check_audit(lease, reservation, &budget)?;
        let allocation = native_allocation(&self.connection, &self.store, lease, reservation)?;
        memory::qualify_controller(allocation.resources.host_bytes)?;
        let remaining = allocation
            .deadline_ms
            .min(lease.deadline)
            .saturating_sub(files::moment()?.ms);
        if remaining <= 0 || (remaining as u64) < budget.wall_ms {
            return Err(failure(
                ErrorCode::LimitExceeded,
                "insufficient remaining allocation before qualification",
            ));
        }
        let deadline = started
            .checked_add(std::time::Duration::from_millis(
                budget.wall_ms.min(remaining as u64),
            ))
            .ok_or_else(|| failure(ErrorCode::LimitExceeded, "audit deadline overflow"))?;
        let authority = self.claim_audit(lease, reservation, &budget)?;
        let check_budget = budget.clone();
        self.check_audit(lease, &authority.reservation, &check_budget)?;
        let runtime = qualify()?;
        self.bind_audit_runtime(lease, reservation, &runtime)?;
        let authorize = || self.check_audit(lease, &authority.reservation, &check_budget);
        let audit = reference::run_authorized_until(
            snapshot, store, plan, policies, exposure, budget, &runtime, coverage, cancel,
            &authorize, deadline,
        )?;
        // Even a completed report has no all-inclusive measured study storage.
        // Keep CPU unknown too: receipts cover cgroups, not controller overhead.
        let clean = audit.report().storage.storage_samples_complete;
        if clean {
            self.confirm_audit_cleanup(lease, reservation, &runtime, audit.id())?;
        }
        self.finish_budget_with(
            lease,
            reservation,
            Settlement {
                operation: ContentDigest::of_bytes(&canonical(&(
                    reservation.id(),
                    "reference-audit-complete",
                ))?),
                outcome: if clean {
                    OutcomeKind::Completed
                } else {
                    OutcomeKind::Incomplete
                },
                usage: Usage::default(),
                resources_released: clean,
                evidence: audit.id().clone(),
            },
            false,
            |tx| {
                if super::super::supervision::stopped(tx, &lease.contract)? {
                    return Err(failure(
                        ErrorCode::PermissionDenied,
                        "stop fenced audit publication",
                    ));
                }
                Ok(())
            },
        )?;
        Ok(audit)
    }
}
