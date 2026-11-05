//! Durable owner stop and trusted-native process supervision; not a Python sandbox.
#[cfg(test)]
mod migration_tests;
mod runtime;
use super::budgets::{OutcomeKind, Reservation, Settlement, Usage};
use super::*;
pub use runtime::{RunningNative, TrustedNativeWorker};
pub(super) const MIGRATION_V3: &str = r#"
CREATE TABLE stop_requests(contract TEXT PRIMARY KEY,id TEXT UNIQUE NOT NULL,body BLOB NOT NULL,content TEXT NOT NULL);
CREATE TABLE worker_intents(id TEXT PRIMARY KEY,contract TEXT NOT NULL,reservation TEXT NOT NULL UNIQUE,body BLOB NOT NULL,content TEXT NOT NULL);
CREATE TABLE worker_updates(worker TEXT NOT NULL,version INTEGER NOT NULL,body BLOB NOT NULL,content TEXT NOT NULL,PRIMARY KEY(worker,version));
CREATE TRIGGER stop_no_update BEFORE UPDATE ON stop_requests BEGIN SELECT RAISE(ABORT,'retained owner stop');END;
CREATE TRIGGER stop_no_delete BEFORE DELETE ON stop_requests BEGIN SELECT RAISE(ABORT,'retained owner stop');END;
CREATE TRIGGER intents_no_update BEFORE UPDATE ON worker_intents BEGIN SELECT RAISE(ABORT,'immutable worker intent');END;
CREATE TRIGGER intents_no_delete BEFORE DELETE ON worker_intents BEGIN SELECT RAISE(ABORT,'retained worker intent');END;
CREATE TRIGGER updates_no_update BEFORE UPDATE ON worker_updates BEGIN SELECT RAISE(ABORT,'immutable worker observation');END;
CREATE TRIGGER updates_no_delete BEFORE DELETE ON worker_updates BEGIN SELECT RAISE(ABORT,'retained worker observation');END;
INSERT INTO budget_migrations VALUES(3,(SELECT value FROM meta WHERE key='schema'));
PRAGMA user_version=3;
"#;
pub(super) fn schema_v3() -> ContentDigest {
    ContentDigest::of_bytes(
        [
            SCHEMA.as_bytes(),
            budgets::MIGRATION_V2.as_bytes(),
            MIGRATION_V3.as_bytes(),
        ]
        .concat()
        .as_slice(),
    )
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    OwnerStop,
    Deadline,
    LeaseLost,
    OutputLimit,
    ResourceLimit,
    Recovery,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StopRecord {
    pub schema_version: u32,
    pub contract: ContentDigest,
    pub operation: ContentDigest,
    pub stage: Stage,
    pub reason: StopReason,
    pub boot: String,
    pub monotonic_ms: i64,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixtureMode {
    Complete,
    Sleep,
    IgnoreTerm,
    SpawnChild,
    Busy,
    Memory,
    Output,
    Partial,
}
impl FixtureMode {
    fn name(self) -> &'static str {
        match self {
            Self::Complete => "complete",
            Self::Sleep => "sleep",
            Self::IgnoreTerm => "ignore_term",
            Self::SpawnChild => "spawn_child",
            Self::Busy => "busy",
            Self::Memory => "memory",
            Self::Output => "output",
            Self::Partial => "partial",
        }
    }
    fn processes(self) -> u64 {
        if self == Self::SpawnChild {
            2
        } else {
            1
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeJob {
    pub mode: FixtureMode,
    pub maximum_output_bytes: u64,
    pub termination_grace_ms: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessIdentity {
    pub pid: u32,
    pub group: u32,
    pub start_ticks: u64,
    pub boot: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Intent {
    schema_version: u32,
    id: ContentDigest,
    contract: ContentDigest,
    reservation: ContentDigest,
    epoch: i64,
    stage: Stage,
    code: ContentDigest,
    job: NativeJob,
    wall_deadline_ms: i64,
    host_bytes: u64,
    cpu_seconds: u64,
    processes: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerOutcome {
    Completed,
    Cancelled,
    TimedOut,
    OutputExceeded,
    ResourceFailure,
    Incomplete,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerReport {
    pub outcome: WorkerOutcome,
    pub cleanup_confirmed: bool,
    pub outputs_eligible: bool,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub exit_code: Option<i32>,
    pub signal: Option<i32>,
    pub cpu_ms: Option<u64>,
    pub leader_peak_rss_bytes: Option<u64>,
    pub measurement_scope: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Observation {
    Started { identity: ProcessIdentity },
    Finished { report: WorkerReport },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Update {
    schema_version: u32,
    worker: ContentDigest,
    version: u32,
    observation: Observation,
    boot: String,
    monotonic_ms: i64,
}
#[derive(Debug, Clone, Serialize)]
pub struct WorkerState {
    pub id: ContentDigest,
    pub contract: ContentDigest,
    pub reservation: ContentDigest,
    pub process: Option<ProcessIdentity>,
    pub report: Option<WorkerReport>,
    pub observations: u32,
}
fn decode<T: serde::de::DeserializeOwned + Serialize>(bytes: &[u8], content: &str) -> Result<T> {
    if ContentDigest::of_bytes(bytes).as_str() != content {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "worker/stop record identity differs",
        ));
    }
    let doc = CanonicalDocument::parse(bytes, Limits::default())?;
    let value: T = serde_json::from_slice(doc.bytes())
        .map_err(|_| failure(ErrorCode::IntegrityMismatch, "worker/stop schema differs"))?;
    if canonical(&value)? != bytes {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "worker/stop canonical bytes differ",
        ));
    }
    Ok(value)
}
pub(super) fn stopped(connection: &Connection, id: &ContentDigest) -> Result<bool> {
    connection
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM stop_requests WHERE contract=?1)",
            [id.as_str()],
            |row| row.get(0),
        )
        .map_err(sql)
}
pub(super) fn all_cleanup_confirmed(
    connection: &Connection,
    contract: &ContentDigest,
) -> Result<bool> {
    let mut statement = connection
        .prepare("SELECT id FROM worker_intents WHERE contract=?1 LIMIT 4097")
        .map_err(sql)?;
    let mut rows = statement.query([contract.as_str()]).map_err(sql)?;
    let mut count = 0;
    while let Some(row) = rows.next().map_err(sql)? {
        count += 1;
        if count > 4096 {
            return Err(failure(ErrorCode::LimitExceeded, "worker inventory bound"));
        }
        let id: String = row.get(0).map_err(sql)?;
        if !read_state(connection, &id.parse()?)?
            .report
            .is_some_and(|report| report.cleanup_confirmed)
        {
            return Ok(false);
        }
    }
    // Reload children have durable promotion admission even before the model
    // worker can return a reaped report. Absence from worker_intents is not clean.
    super::promotion::child_cleanup_confirmed(connection, contract)
}
fn read_intent(connection: &Connection, id: &ContentDigest) -> Result<Intent> {
    let (body, content): (Vec<u8>, String) = connection
        .query_row(
            "SELECT body,content FROM worker_intents WHERE id=?1",
            [id.as_str()],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .map_err(sql)?;
    let value: Intent = decode(&body, &content)?;
    if value.schema_version != 1 || value.id != *id {
        return Err(failure(
            ErrorCode::IntegrityMismatch,
            "worker intent binding differs",
        ));
    }
    Ok(value)
}
fn read_state(connection: &Connection, id: &ContentDigest) -> Result<WorkerState> {
    let intent = read_intent(connection, id)?;
    let mut result = WorkerState {
        id: id.clone(),
        contract: intent.contract,
        reservation: intent.reservation,
        process: None,
        report: None,
        observations: 0,
    };
    let mut statement=connection.prepare("SELECT version,body,content FROM worker_updates WHERE worker=?1 ORDER BY version LIMIT 17").map_err(sql)?;
    let mut rows = statement.query([id.as_str()]).map_err(sql)?;
    while let Some(row) = rows.next().map_err(sql)? {
        if result.observations >= 16 {
            return Err(failure(ErrorCode::LimitExceeded, "worker journal bound"));
        }
        let version: u32 = row.get(0).map_err(sql)?;
        let body: Vec<u8> = row.get(1).map_err(sql)?;
        let content: String = row.get(2).map_err(sql)?;
        let update: Update = decode(&body, &content)?;
        if update.schema_version != 1
            || update.worker != *id
            || update.version != version
            || version != result.observations + 1
        {
            return Err(failure(
                ErrorCode::IntegrityMismatch,
                "worker journal ordering differs",
            ));
        }
        match update.observation {
            Observation::Started { identity } => {
                if result.process.is_some() || result.report.is_some() {
                    return Err(failure(
                        ErrorCode::IntegrityMismatch,
                        "worker process identity changed",
                    ));
                }
                result.process = Some(identity);
            }
            Observation::Finished { report } => {
                if result
                    .report
                    .as_ref()
                    .is_some_and(|prior| prior.outputs_eligible)
                {
                    return Err(failure(
                        ErrorCode::IntegrityMismatch,
                        "worker success was overwritten",
                    ));
                }
                result.report = Some(report);
            }
        }
        result.observations = version;
    }
    Ok(result)
}
fn append(tx: &Transaction<'_>, id: &ContentDigest, observation: Observation) -> Result<()> {
    let current = read_state(tx, id)?;
    if current.observations >= 16 {
        return Err(failure(
            ErrorCode::LimitExceeded,
            "worker observation bound",
        ));
    }
    let now = files::moment()?;
    let record = Update {
        schema_version: 1,
        worker: id.clone(),
        version: current.observations + 1,
        observation,
        boot: now.boot,
        monotonic_ms: now.ms,
    };
    let body = canonical(&record)?;
    tx.execute(
        "INSERT INTO worker_updates VALUES(?1,?2,?3,?4)",
        params![
            id.as_str(),
            record.version,
            body,
            ContentDigest::of_bytes(&body).as_str()
        ],
    )
    .map_err(sql)?;
    Ok(())
}
impl Ledger {
    /// Host/controller control API. Never expose this to a gym or worker IPC.
    pub fn request_stop(
        &mut self,
        contract: &ContentDigest,
        operation: ContentDigest,
        reason: StopReason,
    ) -> Result<StopRecord> {
        self.files.verify(self.limits.max_database_bytes)?;
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sql)?;
        load_contract(&tx, contract)?;
        let old: Option<(Vec<u8>, String)> = tx
            .query_row(
                "SELECT body,content FROM stop_requests WHERE contract=?1",
                [contract.as_str()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .map_err(sql)?;
        if let Some((body, content)) = old {
            return decode(&body, &content);
        }
        let run = state(&tx, contract)?;
        if run.stage.terminal() {
            return Err(failure(
                ErrorCode::InvalidTransition,
                "terminal procedure cannot reopen via stop",
            ));
        }
        let now = files::moment()?;
        let record = StopRecord {
            schema_version: 1,
            contract: contract.clone(),
            operation: operation.clone(),
            stage: run.stage,
            reason,
            boot: now.boot,
            monotonic_ms: now.ms,
        };
        let body = canonical(&record)?;
        tx.execute(
            "INSERT INTO stop_requests VALUES(?1,?2,?3,?4)",
            params![
                contract.as_str(),
                operation.as_str(),
                body,
                ContentDigest::of_bytes(&body).as_str()
            ],
        )
        .map_err(sql)?;
        tx.commit().map_err(sql)?;
        Ok(record)
    }
    pub fn stop_record(&self, contract: &ContentDigest) -> Result<Option<StopRecord>> {
        self.files.verify(self.limits.max_database_bytes)?;
        let data: Option<(Vec<u8>, String)> = self
            .connection
            .query_row(
                "SELECT body,content FROM stop_requests WHERE contract=?1",
                [contract.as_str()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .map_err(sql)?;
        data.map(|(body, content)| decode(&body, &content))
            .transpose()
    }
    pub fn worker_state(&self, id: &ContentDigest) -> Result<WorkerState> {
        self.files.verify(self.limits.max_database_bytes)?;
        let tx = self.connection.unchecked_transaction().map_err(sql)?;
        read_state(&tx, id)
    }
    pub fn launch_native(
        &mut self,
        lease: &Lease,
        reservation: &Reservation,
        worker: &TrustedNativeWorker,
        job: NativeJob,
    ) -> Result<runtime::RunningNative> {
        self.files.verify(self.limits.max_database_bytes)?;
        let now = files::moment()?;
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sql)?;
        check_lease(&tx, &self.store, lease, &now)?;
        if stopped(&tx, &lease.contract)? {
            return Err(failure(
                ErrorCode::PermissionDenied,
                "persisted stop blocks worker dispatch",
            ));
        }
        let allocation = budgets::native_allocation(&tx, &self.store, lease, reservation)?;
        let processes = job.mode.processes();
        if job.maximum_output_bytes == 0
            || job.maximum_output_bytes > 65536
            || job.termination_grace_ms > 5000
            || allocation.resources.storage_bytes
                < job
                    .maximum_output_bytes
                    .checked_mul(20)
                    .and_then(|bytes| bytes.checked_add(65536))
                    .ok_or_else(|| failure(ErrorCode::LimitExceeded, "output budget overflow"))?
            || allocation.resources.gpu_ms != 0
            || allocation.resources.device_bytes != 0
            || allocation.resources.host_bytes / processes < 33_554_432
            || allocation.resources.cpu_ms / (1000 * processes) < 2
        {
            return Err(failure(
                ErrorCode::UnsupportedCapability,
                "native fixture CPU/address-space/output limit granularity unsupported",
            ));
        }
        worker.verify()?;
        let id = ContentDigest::of_bytes(&canonical(&(
            &lease.contract,
            reservation.id(),
            worker.digest(),
            &job,
        ))?);
        let duplicate: bool = tx
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM worker_intents WHERE reservation=?1)",
                [reservation.id().as_str()],
                |row| row.get(0),
            )
            .map_err(sql)?;
        if duplicate {
            return Err(failure(
                ErrorCode::IdempotencyConflict,
                "reservation already has a worker intent; recover instead of relaunching",
            ));
        }
        let intent = Intent {
            schema_version: 1,
            id: id.clone(),
            contract: lease.contract.clone(),
            reservation: reservation.id().clone(),
            epoch: lease.epoch,
            stage: state(&tx, &lease.contract)?.stage,
            code: worker.digest().clone(),
            job,
            wall_deadline_ms: allocation.deadline_ms,
            host_bytes: allocation.resources.host_bytes / processes,
            cpu_seconds: allocation.resources.cpu_ms / (1000 * processes) - 1,
            processes,
        };
        let body = canonical(&intent)?;
        // Both backends acquire the same unique reservation key under this
        // IMMEDIATE transaction, before any executable is spawned.
        budgets::audit_budget::claim_execution(&tx, reservation.id(), "native")?;
        tx.execute(
            "INSERT INTO worker_intents VALUES(?1,?2,?3,?4,?5)",
            params![
                id.as_str(),
                lease.contract.as_str(),
                reservation.id().as_str(),
                body,
                ContentDigest::of_bytes(&body).as_str()
            ],
        )
        .map_err(sql)?;
        ensure_live(lease)?;
        tx.commit().map_err(sql)?;
        let mut running = runtime::RunningNative::spawn(worker, &intent)?;
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sql)?;
        if let Err(error) = check_lease(&tx, &self.store, lease, &files::moment()?) {
            drop(tx);
            running.abort_owned();
            return Err(error);
        }
        if stopped(&tx, &lease.contract)? {
            drop(tx);
            running.abort_owned();
            return Err(failure(ErrorCode::PermissionDenied, "stop won launch race"));
        }
        append(
            &tx,
            &id,
            Observation::Started {
                identity: running.identity().clone(),
            },
        )?;
        ensure_live(lease)?;
        tx.commit().map_err(sql)?;
        // stdin work authorization is sent only AFTER identity was persisted.
        running.start(&intent)?;
        Ok(running)
    }
    pub fn poll_native(
        &mut self,
        lease: &Lease,
        reservation: &Reservation,
        running: &mut runtime::RunningNative,
    ) -> Result<Option<WorkerReport>> {
        let intent = read_intent(&self.connection, running.id())?;
        if let Err(error) = check_lease_read(self, lease) {
            let _ = self.request_stop(
                &lease.contract,
                ContentDigest::of_bytes(&canonical(&(running.id(), "lease-lost-stop"))?),
                StopReason::LeaseLost,
            );
            running.abort_owned();
            return Err(error);
        }
        let mut stop = self.stop_record(&lease.contract)?.is_some();
        if !stop {
            if let Some(cause) = running.stop_cause()? {
                self.request_stop(
                    &lease.contract,
                    ContentDigest::of_bytes(&canonical(&(running.id(), "automatic-stop"))?),
                    cause,
                )?;
                stop = true;
            }
        }
        let report = running.poll(stop)?;
        let Some(mut report) = report else {
            return Ok(None);
        };
        if matches!(
            report.outcome,
            WorkerOutcome::TimedOut | WorkerOutcome::OutputExceeded
        ) && !stop
        {
            // The runtime has already enforced its fixed trusted watchdog;
            // retain the coordinator stop before any further termination.
            self.request_stop(
                &lease.contract,
                ContentDigest::of_bytes(&canonical(&(running.id(), "automatic-stop"))?),
                if report.outcome == WorkerOutcome::TimedOut {
                    StopReason::Deadline
                } else {
                    StopReason::OutputLimit
                },
            )?;
        }
        if self.stop_record(&lease.contract)?.is_some() {
            report.outputs_eligible = false;
            report.stdout.clear();
            if report.outcome == WorkerOutcome::Completed {
                report.outcome = WorkerOutcome::Cancelled;
            }
        }
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sql)?;
        check_lease(&tx, &self.store, lease, &files::moment()?)?;
        let mut exit_observation = report.clone();
        exit_observation.outputs_eligible = false;
        append(
            &tx,
            running.id(),
            Observation::Finished {
                report: exit_observation,
            },
        )?;
        ensure_live(lease)?;
        tx.commit().map_err(sql)?;
        let usage = Usage {
            cpu_ms: if intent.processes == 1 {
                report.cpu_ms
            } else {
                None
            },
            attempts: if budgets::native_kind(&self.connection, reservation.id())?
                == budgets::WorkKind::Candidate
            {
                Some(1)
            } else {
                None
            },
            ..Usage::default()
        };
        self.settle(
            lease,
            reservation,
            Settlement {
                operation: ContentDigest::of_bytes(&canonical(&(running.id(), "final-cost"))?),
                outcome: if report.outcome == WorkerOutcome::Completed {
                    OutcomeKind::Completed
                } else if report.outcome == WorkerOutcome::Cancelled {
                    OutcomeKind::Cancelled
                } else {
                    OutcomeKind::Incomplete
                },
                usage: if report.cleanup_confirmed {
                    usage
                } else {
                    Usage::default()
                },
                resources_released: report.cleanup_confirmed,
                evidence: ContentDigest::of_bytes(&canonical(&report)?),
            },
        )?;
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sql)?;
        check_lease(&tx, &self.store, lease, &files::moment()?)?;
        if stopped(&tx, &lease.contract)? {
            report.outputs_eligible = false;
            report.stdout.clear();
            if report.outcome == WorkerOutcome::Completed {
                report.outcome = WorkerOutcome::Cancelled;
            }
        }
        append(
            &tx,
            running.id(),
            Observation::Finished {
                report: report.clone(),
            },
        )?;
        ensure_live(lease)?;
        tx.commit().map_err(sql)?;
        // Reports alone never advance workflow or accepted model pointers.
        Ok(Some(report))
    }
    pub fn recover_native(&mut self, lease: &Lease, id: &ContentDigest) -> Result<WorkerReport> {
        check_lease_read(self, lease)?;
        let intent = read_intent(&self.connection, id)?;
        if intent.contract != lease.contract {
            return Err(failure(
                ErrorCode::PermissionDenied,
                "worker belongs to a different revision",
            ));
        }
        let prior = read_state(&self.connection, id)?;
        if let Some(report) = prior
            .report
            .as_ref()
            .filter(|report| report.cleanup_confirmed)
        {
            if budgets::native_released(&self.connection, &intent.reservation)? {
                return Ok(report.clone());
            }
        }
        self.request_stop(
            &lease.contract,
            ContentDigest::of_bytes(&canonical(&(id, "recovery-stop"))?),
            StopReason::Recovery,
        )?;
        let cleanup = if prior
            .report
            .as_ref()
            .is_some_and(|report| report.cleanup_confirmed)
        {
            true
        } else {
            match &prior.process {
                Some(identity) => runtime::recover_group(identity)?,
                None => false,
            }
        };
        let report = WorkerReport {
            outcome: WorkerOutcome::Incomplete,
            cleanup_confirmed: cleanup,
            outputs_eligible: false,
            stdout: vec![],
            stderr: vec![],
            exit_code: None,
            signal: None,
            cpu_ms: None,
            leader_peak_rss_bytes: None,
            measurement_scope: "crash: actual costs unknown; full reservation retained/charged"
                .to_owned(),
        };
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sql)?;
        check_lease(&tx, &self.store, lease, &files::moment()?)?;
        append(
            &tx,
            id,
            Observation::Finished {
                report: report.clone(),
            },
        )?;
        ensure_live(lease)?;
        tx.commit().map_err(sql)?;
        let reservation = self.recover_reservation(lease, &intent.reservation)?;
        self.settle(
            lease,
            &reservation,
            Settlement {
                operation: ContentDigest::of_bytes(&canonical(&(id, "crash-cost"))?),
                outcome: OutcomeKind::Incomplete,
                usage: Usage::default(),
                resources_released: cleanup,
                evidence: ContentDigest::of_bytes(&canonical(&report)?),
            },
        )?;
        Ok(report)
    }
}
fn check_lease_read(ledger: &Ledger, lease: &Lease) -> Result<()> {
    ledger.files.verify(ledger.limits.max_database_bytes)?;
    let tx = ledger.connection.unchecked_transaction().map_err(sql)?;
    check_lease(&tx, &ledger.store, lease, &files::moment()?)?;
    Ok(())
}
