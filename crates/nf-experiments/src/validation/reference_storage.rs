//! Conservative per-invocation accounting; NOT a coordinator family grant.
use super::*;

// Every publication is a canonical document bounded by super::document to 1 MiB.
// Include one block of headroom, but do not claim portable physical CAS quotas.
const PUBLICATION_BYTES: u64 = 1048576 + 4096;
const MAX_ALLOWANCE: u64 = 16 * 1024 * 1024 * 1024;

pub(super) fn validate_allowance(bytes: u64) -> Result<()> {
    if bytes == 0 || bytes > MAX_ALLOWANCE {
        return Err(error(
            ErrorCode::LimitExceeded,
            "explicit finite audit storage_bytes required (1..=16 GiB)",
        ));
    }
    Ok(())
}

#[derive(Debug, Serialize)]
pub struct Accounting {
    pub allowance_bytes: u64,
    /// All possible work is ceiling-charged up front, even on early termination.
    pub charged_bytes: u64,
    pub sandbox_ceiling_bytes: u64,
    pub cas_ceiling_bytes: u64,
    pub workspace_limit_per_job_bytes: u64,
    pub jobs_claimed: u64,
    pub publications_claimed: u64,
    /// Physical all-inclusive study usage is unknown, never a fabricated zero.
    pub measured_study_storage_bytes: Option<u64>,
    pub observed_workspace_peak_bytes: Option<u64>,
    pub observed_journal_retained_bytes: Option<u64>,
    /// All returned job receipts supplied finalized, complete storage samples.
    /// No samples is not a measured zero (both observations remain None).
    pub storage_samples_complete: bool,
    pub capacity_guaranteed: bool,
    pub durable_family_accounting: bool,
    pub scope: String,
    #[serde(skip)]
    jobs: u64,
    #[serde(skip)]
    publications: u64,
}
impl Accounting {
    pub(super) fn admit(jobs: u64, allowance: u64) -> Result<Self> {
        validate_allowance(allowance)?;
        if jobs == 0 || jobs > 528 {
            return Err(error(
                ErrorCode::LimitExceeded,
                "finite audit job storage count required",
            ));
        }
        // Entire managed job ceiling includes workspace (source/input/assets/
        // guardian/policy/directories), journals and initial capacity probe.
        // No reclamation/dedup/early-exit refund is inferred from this bound.
        let sandbox = jobs * sandbox::STORAGE_JOB_BYTES;
        let publications = jobs * 2 + 2; // receipt + observation/job, plan + report
        let cas = publications * PUBLICATION_BYTES;
        let charged = sandbox + cas;
        if charged > allowance {
            return Err(error(
                ErrorCode::LimitExceeded,
                "aggregate study storage allowance insufficient before staging",
            ));
        }
        Ok(Self {
            allowance_bytes: allowance, charged_bytes: charged,
            sandbox_ceiling_bytes: sandbox, cas_ceiling_bytes: cas,
            workspace_limit_per_job_bytes: sandbox::STORAGE_JOB_BYTES,
            jobs_claimed: 0, publications_claimed: 0,
            measured_study_storage_bytes: None,
            observed_workspace_peak_bytes: None, observed_journal_retained_bytes: None,
            storage_samples_complete: true,
            capacity_guaranteed: false, durable_family_accounting: false,
            scope: "controller_allocated_invocation_only;full_job_ceiling_per_attempt;CAS_logical_document_ceiling_plus_headroom;excludes_existing_CAS_SQLite_qualification_and_other_invocations;physical_CAS_metadata_and_future_disk_capacity_unsupported;caller_must_hold_and_settle_durable_family_reservation_including_errors".into(),
            jobs, publications,
        })
    }
    pub(super) fn claim_job(&mut self) -> Result<u64> {
        if self.jobs_claimed >= self.jobs {
            return Err(error(
                ErrorCode::LimitExceeded,
                "audit cumulative staging slices exhausted",
            ));
        }
        self.jobs_claimed += 1;
        Ok(self.workspace_limit_per_job_bytes)
    }
    pub(super) fn claim_publication(&mut self) -> Result<()> {
        if self.publications_claimed >= self.publications {
            return Err(error(
                ErrorCode::LimitExceeded,
                "audit cumulative CAS slices exhausted",
            ));
        }
        self.publications_claimed += 1;
        Ok(())
    }
    pub(super) fn put(&mut self, store: &ArtifactStore, bytes: &[u8]) -> Result<ContentDigest> {
        if bytes.len() > 1048576 {
            return Err(error(
                ErrorCode::LimitExceeded,
                "audit CAS document bound exceeded",
            ));
        }
        self.claim_publication()?;
        Ok(store.put(bytes)?.digest)
    }
    pub(super) fn observe(&mut self, receipt: &sandbox::Receipt) {
        let complete = receipt.storage.as_ref().is_some_and(|s| {
            s.high_water_complete && s.accounting_finalized && receipt.cleanup_confirmed
        });
        self.storage_samples_complete &= complete;
        if !self.storage_samples_complete {
            // Unknown is sticky: later good receipts cannot repair missing
            // historical samples or turn partial totals into complete metrics.
            self.observed_workspace_peak_bytes = None;
            self.observed_journal_retained_bytes = None;
            return;
        }
        if let Some(s) = &receipt.storage {
            self.observed_workspace_peak_bytes = Some(
                self.observed_workspace_peak_bytes
                    .unwrap_or(0)
                    .max(s.workspace_peak_charged_bytes),
            );
            // Cumulative observed per-job retained bytes, not a root delta
            // (root journals may be pruned between jobs).
            self.observed_journal_retained_bytes = self
                .observed_journal_retained_bytes
                .unwrap_or(0)
                .checked_add(s.retained_bytes);
        }
    }
}

#[cfg(test)]
#[path = "reference_storage_enospc.rs"]
mod enospc;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_ceiling_not_payload_and_no_refunds() {
        let mut a = Accounting::admit(2, 64 * 1024 * 1024).unwrap();
        assert_eq!(a.sandbox_ceiling_bytes, 2 * sandbox::STORAGE_JOB_BYTES);
        assert_eq!(
            a.charged_bytes,
            a.sandbox_ceiling_bytes + 6 * PUBLICATION_BYTES
        );
        assert!(Accounting::admit(2, a.charged_bytes - 1).is_err());
        let exact = Accounting::admit(2, a.charged_bytes).unwrap();
        assert_eq!(exact.charged_bytes, a.charged_bytes);
        let charge = a.charged_bytes;
        for _ in 0..2 {
            assert_eq!(a.claim_job().unwrap(), sandbox::STORAGE_JOB_BYTES);
        }
        assert!(a.claim_job().is_err());
        for _ in 0..6 {
            a.claim_publication().unwrap();
        }
        assert!(a.claim_publication().is_err());
        assert_eq!(a.charged_bytes, charge);
        assert_eq!(a.measured_study_storage_bytes, None);
        assert_eq!(a.observed_workspace_peak_bytes, None);
        assert_eq!(a.observed_journal_retained_bytes, None);
        assert!(!a.capacity_guaranteed && !a.durable_family_accounting);
    }
    #[test]
    fn missing_or_incomplete_storage_is_sticky_unknown() {
        let d = ContentDigest::of_bytes(b"fixture");
        let mut r: sandbox::Receipt = serde_json::from_value(json!({
            "schema_version":3,"job":"fixture","source":d,"input":d,
            "capability":d,"status":"completed","cleanup_confirmed":true,
            "outputs_eligible":false,"promotion_status":"not_evaluated","report":null,
            "storage":{"admission_preallocated_bytes":4096,"peak_charged_bytes":8192,
                "workspace_bytes_before_reclaim":4096,"retained_bytes":4096,
                "root_retained_bytes":4096,"workspace_reclaimed":true,
                "accounting_finalized":true,"high_water_complete":true,
                "workspace_peak_charged_bytes":4096,"workspace_limit_bytes":16384}
        }))
        .unwrap();
        for missing in [false, true] {
            let mut a = Accounting::admit(3, 64 * 1024 * 1024).unwrap();
            a.observe(&r);
            assert_eq!(a.observed_journal_retained_bytes, Some(4096));
            let mut unknown = r.clone();
            if missing {
                unknown.storage = None;
            } else {
                unknown.storage.as_mut().unwrap().high_water_complete = false;
            }
            a.observe(&unknown);
            a.observe(&r);
            assert!(!a.storage_samples_complete);
            assert_eq!(a.observed_workspace_peak_bytes, None);
            assert_eq!(a.observed_journal_retained_bytes, None);
            assert_eq!(a.measured_study_storage_bytes, None);
        }
        r.cleanup_confirmed = false;
        let mut a = Accounting::admit(2, 64 * 1024 * 1024).unwrap();
        a.observe(&r);
        assert!(!a.storage_samples_complete);
        assert_eq!(a.observed_workspace_peak_bytes, None);
    }
    #[test]
    fn finite_explicit_allowance_and_job_count() {
        for allowance in [0, u64::MAX, MAX_ALLOWANCE + 1] {
            assert!(Accounting::admit(2, allowance).is_err());
        }
        for jobs in [0, 529, u64::MAX] {
            assert!(Accounting::admit(jobs, MAX_ALLOWANCE).is_err());
        }
        assert!(Accounting::admit(528, MAX_ALLOWANCE).is_ok());
        assert!(serde_json::from_value::<Budget>(
            json!({"max_executions":2,"cpu_ms":1000,"wall_ms":1000})
        )
        .is_err());
    }
}
