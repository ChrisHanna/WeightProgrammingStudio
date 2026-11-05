//! Cleanup authority is produced only by native, bound sandbox recovery.
use super::*;
use crate::sandbox::QualifiedDocker;

fn owner(connection: &Connection, reservation: &Reservation) -> Result<()> {
    verify_v8(connection)?;
    let backend: String = connection
        .query_row(
            "SELECT backend FROM execution_owners WHERE request=?1",
            [reservation.id().as_str()],
            |r| r.get(0),
        )
        .map_err(sql)?;
    if backend == "legacy_audit" {
        return Err(failure(ErrorCode::UnsupportedCapability,
            "legacy audit hold retained: no authenticated reservation-to-root inventory; sweeping an operator-selected root cannot prove absence elsewhere; requires externally verified complete historical inventory and a separately reviewed migration"));
    }
    if backend != "audit" {
        return Err(failure(
            ErrorCode::PermissionDenied,
            "foreign execution requires separate recovery",
        ));
    }
    Ok(())
}
fn binding(connection: &Connection, reservation: &Reservation) -> Result<Option<String>> {
    connection
        .query_row(
            "SELECT root FROM audit_runtime_bindings WHERE request=?1",
            [reservation.id().as_str()],
            |r| r.get(0),
        )
        .optional()
        .map_err(sql)
}
impl Ledger {
    pub(super) fn bind_audit_runtime(
        &mut self,
        lease: &Lease,
        reservation: &Reservation,
        runtime: &QualifiedDocker,
    ) -> Result<()> {
        self.files.verify(self.limits.max_database_bytes)?;
        let identity = runtime.durable_root_identity()?;
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sql)?;
        check_lease(&tx, &self.store, lease, &files::moment()?)?;
        native_allocation(&tx, &self.store, lease, reservation)?;
        owner(&tx, reservation)?;
        if super::super::super::supervision::stopped(&tx, &lease.contract)? {
            return Err(failure(
                ErrorCode::PermissionDenied,
                "stop won audit binding race",
            ));
        }
        if tx
            .execute(
                "INSERT INTO audit_runtime_bindings VALUES(?1,?2)",
                params![reservation.id().as_str(), identity.as_str()],
            )
            .map_err(sql)?
            != 1
            || binding(&tx, reservation)?.as_deref() != Some(identity.as_str())
        {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "audit runtime binding not durable",
            ));
        }
        ensure_live(lease)?;
        tx.commit().map_err(sql)
    }

    pub(super) fn confirm_audit_cleanup(
        &mut self,
        lease: &Lease,
        reservation: &Reservation,
        runtime: &QualifiedDocker,
        evidence: &ContentDigest,
    ) -> Result<()> {
        self.files.verify(self.limits.max_database_bytes)?;
        let identity = runtime.durable_root_identity()?;
        owner(&self.connection, reservation)?;
        if binding(&self.connection, reservation)?.as_deref() != Some(identity.as_str()) {
            return Err(failure(
                ErrorCode::PermissionDenied,
                "cleanup runtime differs from durable binding",
            ));
        }
        // Synchronous execution returned; root recovery takes its exclusive OS
        // lock and verifies container absence before reclaiming any workspace.
        let _clean = runtime.recover_attested(&identity)?;
        self.record_audit_cleanup(lease, reservation, evidence)
    }

    fn record_audit_cleanup(
        &mut self,
        lease: &Lease,
        reservation: &Reservation,
        evidence: &ContentDigest,
    ) -> Result<()> {
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sql)?;
        check_lease(&tx, &self.store, lease, &files::moment()?)?;
        owner(&tx, reservation)?;
        let contract = load_contract(&tx, &lease.contract)?;
        if reservation.store != self.store || reservation.family != contract.study_family {
            return Err(failure(
                ErrorCode::PermissionDenied,
                "foreign cleanup reservation",
            ));
        }
        let actual: String = tx
            .query_row(
                "SELECT contract FROM budget_requests WHERE id=?1",
                [reservation.id().as_str()],
                |r| r.get(0),
            )
            .map_err(sql)?;
        if actual != lease.contract.as_str() {
            return Err(failure(
                ErrorCode::PermissionDenied,
                "cleanup contract differs",
            ));
        }
        let prior: Option<String> = tx
            .query_row(
                "SELECT evidence FROM audit_cleanup WHERE request=?1",
                [reservation.id().as_str()],
                |r| r.get(0),
            )
            .optional()
            .map_err(sql)?;
        if let Some(prior) = prior {
            if prior != evidence.as_str() {
                return Err(failure(
                    ErrorCode::IdempotencyConflict,
                    "cleanup evidence differs",
                ));
            }
        } else if tx
            .execute(
                "INSERT INTO audit_cleanup VALUES(?1,?2)",
                params![reservation.id().as_str(), evidence.as_str()],
            )
            .map_err(sql)?
            != 1
        {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "cleanup evidence insertion ignored",
            ));
        }
        let stored: String = tx
            .query_row(
                "SELECT evidence FROM audit_cleanup WHERE request=?1",
                [reservation.id().as_str()],
                |r| r.get(0),
            )
            .map_err(sql)?;
        if stored != evidence.as_str() {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "cleanup readback differs",
            ));
        }
        ensure_live(lease)?;
        tx.commit().map_err(sql)
    }

    /// Native recovery, not a JSON cleanup declaration. Persist stop BEFORE
    /// taking the bound runtime lock so no prior owner can start another job.
    /// Legacy claims have no authenticated root inventory and remain held.
    pub fn recover_reference_audit(
        &mut self,
        lease: &Lease,
        reservation: &Reservation,
        runtime: &QualifiedDocker,
    ) -> Result<()> {
        self.recover_audit_fenced(lease, reservation, runtime, || {})
    }

    // The hook makes the pre-stop binding interleaving deterministic in tests;
    // production callers cannot supply it or choose a root after the fence.
    pub(super) fn recover_audit_fenced(
        &mut self,
        lease: &Lease,
        reservation: &Reservation,
        runtime: &QualifiedDocker,
        before_stop: impl FnOnce(),
    ) -> Result<()> {
        self.files.verify(self.limits.max_database_bytes)?;
        owner(&self.connection, reservation)?;
        let recovered = self.recover_reservation(lease, reservation.id())?;
        if recovered.store != reservation.store || recovered.family != reservation.family {
            return Err(failure(
                ErrorCode::PermissionDenied,
                "foreign recovery reservation",
            ));
        }
        let actual: String = self
            .connection
            .query_row(
                "SELECT contract FROM budget_requests WHERE id=?1",
                [reservation.id().as_str()],
                |r| r.get(0),
            )
            .map_err(sql)?;
        if actual != lease.contract.as_str() {
            return Err(failure(
                ErrorCode::PermissionDenied,
                "recovery contract differs",
            ));
        }
        before_stop();
        self.request_stop(
            &lease.contract,
            ContentDigest::of_bytes(&canonical(&(reservation.id(), "audit-recovery-stop"))?),
            super::super::super::supervision::StopReason::Recovery,
        )?;
        // Do not read the binding before stop commits. Binding takes the same
        // write fence and checks stop: it either committed first and is visible
        // here, or it can never authorize staging. Revalidate under that fence.
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sql)?;
        check_lease(&tx, &self.store, lease, &files::moment()?)?;
        owner(&tx, reservation)?;
        let root = binding(&tx, reservation)?;
        let clean = match &root {
            Some(root) => Some(runtime.recover_attested(&root.parse()?)?),
            // No binding after the fence proves no staging was authorized. Do
            // NOT sweep an unrelated caller-selected root in this case.
            None => None,
        };
        let prior: Option<String> = tx
            .query_row(
                "SELECT evidence FROM audit_cleanup WHERE request=?1",
                [reservation.id().as_str()],
                |r| r.get(0),
            )
            .optional()
            .map_err(sql)?;
        let evidence = match prior {
            Some(value) => value.parse()?,
            None => ContentDigest::of_bytes(&canonical(&(
                reservation.id(),
                &root,
                "native-audit-cleanup",
            ))?),
        };
        ensure_live(lease)?;
        tx.commit().map_err(sql)?;
        // `clean` keeps the exact attested directory exclusively locked until
        // cleanup evidence and conservative settlement are durably committed.
        let _clean = clean;
        self.record_audit_cleanup(lease, reservation, &evidence)?;
        self.settle(
            lease,
            reservation,
            Settlement {
                operation: ContentDigest::of_bytes(&canonical(&(
                    reservation.id(),
                    "audit-recovery-cost",
                ))?),
                outcome: OutcomeKind::Incomplete,
                usage: Usage::default(),
                resources_released: true,
                evidence,
            },
        )
    }
}
