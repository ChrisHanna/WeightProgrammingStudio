//! Live native validation/approval witnesses; no DTO-only approval restoration.
use crate::boundary::{canonical, document, python_error, worker};
use crate::intake::GymSnapshotHandle;
use nf_checkpoint::store::ArtifactStore;
use nf_core::canonical::{CanonicalDocument, Limits};
use nf_core::error::{CoreError, ErrorCode};
use nf_experiments::intake;
use nf_experiments::sandbox::QualifiedDocker;
use nf_experiments::validation::{self, reference, Approval, Plan, Validation};
use nf_surgeon::cancellation::CancellationToken;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyString};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Sticky cancellation shared with each active sandbox job. No Python callback.
#[pyclass(frozen, module = "neural_foundry._native")]
pub struct GymAuditCancellationHandle {
    flag: Arc<AtomicBool>,
}
#[pymethods]
impl GymAuditCancellationHandle {
    #[new]
    fn new() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
        }
    }
    fn cancel(&self) {
        self.flag.store(true, Ordering::Release);
    }
    #[getter]
    fn cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }
}

/// Explicit crash reconciliation: the caller must exclusively own this root.
#[pyfunction(signature=(sandbox_root, enabled=false))]
fn recover_gym_reference_audits(
    py: Python<'_>,
    sandbox_root: String,
    enabled: bool,
) -> PyResult<usize> {
    if !enabled {
        return Err(python_error(
            py,
            audit_error(
                ErrorCode::PermissionDenied,
                "UNTRUSTED_GYM_EXECUTION_DISABLED",
            ),
        ));
    }
    worker(py, CancellationToken::default(), move || {
        QualifiedDocker::qualify(true, Path::new(&sandbox_root))?.recover_pending()
    })
}

#[pyclass(frozen, module = "neural_foundry._native")]
pub struct GymValidationHandle {
    value: Arc<Validation>,
    store: Arc<ArtifactStore>,
}
#[pymethods]
impl GymValidationHandle {
    #[getter]
    fn receipt_id(&self) -> String {
        self.value.id().to_string()
    }
    fn report_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let value = self.value.clone();
        let bytes = worker(py, CancellationToken::default(), move || {
            canonical(value.report())
        })?;
        Ok(PyBytes::new(py, &bytes))
    }
}
#[pyclass(frozen, module = "neural_foundry._native")]
pub struct GymApprovalHandle {
    pub(crate) value: Arc<Approval>,
}
#[pymethods]
impl GymApprovalHandle {
    #[getter]
    fn receipt_id(&self) -> String {
        self.value.id().to_string()
    }
    #[getter]
    fn snapshot_id(&self) -> String {
        self.value.record().snapshot.to_string()
    }
    fn record_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let value = self.value.clone();
        let bytes = worker(py, CancellationToken::default(), move || {
            canonical(value.record())
        })?;
        Ok(PyBytes::new(py, &bytes))
    }
}
#[pyfunction]
#[pyo3(signature=(snapshot,plan,sandbox_root,enabled=false))]
fn validate_gym(
    py: Python<'_>,
    snapshot: &GymSnapshotHandle,
    plan: &Bound<'_, PyBytes>,
    sandbox_root: String,
    enabled: bool,
) -> PyResult<GymValidationHandle> {
    if !enabled {
        return Err(python_error(
            py,
            CoreError::new(
                ErrorCode::PermissionDenied,
                "/gym_validation",
                "UNTRUSTED_GYM_EXECUTION_DISABLED",
            ),
        ));
    }
    if sandbox_root.is_empty() || sandbox_root.len() > 4096 || sandbox_root.contains('\0') {
        return Err(python_error(
            py,
            CoreError::new(
                ErrorCode::InvalidValue,
                "/sandbox_root",
                "bounded controller root required",
            ),
        ));
    }
    let bytes = document(py, plan)?;
    let store = snapshot.store.clone();
    let worker_store = store.clone();
    let snapshot = snapshot.snapshot.clone();
    let value = worker(py, CancellationToken::default(), move || {
        let plan = Plan::parse(&bytes)?;
        let sandbox = QualifiedDocker::qualify(true, Path::new(&sandbox_root))?;
        validation::validate(&snapshot, &worker_store, plan, &sandbox)
    })?;
    Ok(GymValidationHandle {
        value: Arc::new(value),
        store,
    })
}
#[pyfunction]
fn approve_gym_snapshot(
    py: Python<'_>,
    snapshot: &GymSnapshotHandle,
    validation: &GymValidationHandle,
    approval_reference: String,
) -> PyResult<GymApprovalHandle> {
    let snapshot = snapshot.snapshot.clone();
    let value = validation.value.clone();
    let store = validation.store.clone();
    let approved = worker(py, CancellationToken::default(), move || {
        validation::approve(&snapshot, &value, &store, approval_reference)
    })?;
    Ok(GymApprovalHandle {
        value: Arc::new(approved),
    })
}
/// No constructor/Deserialize/approval conversion: only native execution mints it.
#[pyclass(frozen, module = "neural_foundry._native")]
pub struct GymReferenceAuditHandle {
    value: Arc<reference::Audit>,
}
#[pymethods]
impl GymReferenceAuditHandle {
    #[getter]
    fn receipt_id(&self) -> String {
        self.value.id().to_string()
    }
    fn report_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let value = self.value.clone();
        let bytes = worker(py, CancellationToken::default(), move || {
            canonical(value.report())
        })?;
        Ok(PyBytes::new(py, &bytes))
    }
}
fn audit_error(code: ErrorCode, message: &str) -> CoreError {
    CoreError::new(code, "/reference_audit", message)
}
fn audit_json(py: Python<'_>, value: &Bound<'_, PyAny>, max: usize) -> PyResult<Vec<u8>> {
    let bytes = if let Ok(value) = value.downcast::<PyBytes>() {
        value.as_bytes()
    } else if let Ok(value) = value.downcast::<PyString>() {
        value.to_str()?.as_bytes()
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "JSON str or bytes required",
        ));
    };
    if bytes.len() > max {
        return Err(python_error(
            py,
            audit_error(ErrorCode::LimitExceeded, "audit JSON exceeds byte bound"),
        ));
    }
    Ok(bytes.to_vec())
}
#[allow(clippy::too_many_arguments)] // Fixed handoff ABI; no ambient execution settings.
#[pyfunction(signature=(snapshot, plan_json, policy_snapshot_handles, roles, exposure, budget_json, sandbox_root, enabled=false, coverage_json=None, cancellation=None))]
fn run_gym_reference_audit(
    py: Python<'_>,
    snapshot: &GymSnapshotHandle,
    plan_json: &Bound<'_, PyAny>,
    policy_snapshot_handles: &Bound<'_, PyList>,
    roles: &Bound<'_, PyList>,
    exposure: String,
    budget_json: &Bound<'_, PyAny>,
    sandbox_root: String,
    enabled: bool,
    coverage_json: Option<&Bound<'_, PyAny>>,
    cancellation: Option<PyRef<'_, GymAuditCancellationHandle>>,
) -> PyResult<GymReferenceAuditHandle> {
    if !enabled {
        return Err(python_error(
            py,
            audit_error(
                ErrorCode::PermissionDenied,
                "UNTRUSTED_GYM_EXECUTION_DISABLED",
            ),
        ));
    }
    if !(1..=8).contains(&policy_snapshot_handles.len())
        || roles.len() != policy_snapshot_handles.len()
    {
        return Err(python_error(
            py,
            audit_error(
                ErrorCode::LimitExceeded,
                "1-8 policy handles with matching roles required",
            ),
        ));
    }
    if sandbox_root.is_empty() || sandbox_root.len() > 4096 || sandbox_root.contains('\0') {
        return Err(python_error(
            py,
            audit_error(ErrorCode::InvalidValue, "bounded sandbox root required"),
        ));
    }
    let plan = audit_json(py, plan_json, 65536)?;
    let budget = audit_json(py, budget_json, 4096)?;
    let coverage = coverage_json
        .map(|value| audit_json(py, value, 4096))
        .transpose()?;
    let cancel = cancellation.map(|c| c.flag.clone()).unwrap_or_default();
    if cancel.load(Ordering::Acquire) {
        return Err(python_error(
            py,
            audit_error(ErrorCode::TransactionAborted, "audit cancelled"),
        ));
    }
    let policies = policy_snapshot_handles
        .iter()
        .map(|value| {
            Ok(value
                .extract::<PyRef<'_, GymSnapshotHandle>>()?
                .snapshot
                .id()
                .clone())
        })
        .collect::<PyResult<Vec<_>>>()?;
    let roles = roles
        .iter()
        .map(|value| value.extract::<String>())
        .collect::<PyResult<Vec<_>>>()?;
    let store = snapshot.store.clone();
    let id = snapshot.snapshot.id().clone();
    let value = worker(py, CancellationToken::default(), move || {
        let parse_enum = |value: &str| serde_json::Value::String(value.to_owned());
        let exposure: reference::Exposure = serde_json::from_value(parse_enum(&exposure))
            .map_err(|_| audit_error(ErrorCode::InvalidValue, "unknown exposed audit domain"))?;
        let roles = roles
            .into_iter()
            .map(|role| {
                serde_json::from_value::<reference::Role>(parse_enum(&role))
                    .map_err(|_| audit_error(ErrorCode::InvalidValue, "unknown reference role"))
            })
            .collect::<nf_core::error::Result<Vec<_>>>()?;
        let plan = Plan::parse(&plan)?;
        let document = CanonicalDocument::parse(
            &budget,
            Limits {
                max_bytes: 4096,
                max_depth: 8,
                max_nodes: 64,
            },
        )?;
        let budget: reference::Budget = serde_json::from_value(document.value().clone())
            .map_err(|_| audit_error(ErrorCode::InvalidValue, "invalid aggregate audit budget"))?;
        // Reject absent/insufficient cumulative storage allowance BEFORE
        // qualification, recovery, CAS publication or any sandbox staging.
        budget.validate_storage()?;
        // Every snapshot and asset must exist and verify in the gym's CAS. Never
        // open a policy's host source or silently copy from a different store.
        let snapshot = intake::restore(&store, &id)?;
        let policies = policies
            .iter()
            .map(|id| intake::restore(&store, id))
            .collect::<nf_core::error::Result<Vec<_>>>()?;
        let policies = policies
            .iter()
            .zip(roles)
            .map(|(snapshot, role)| reference::Policy { snapshot, role })
            .collect::<Vec<_>>();
        let coverage = match coverage {
            Some(bytes) => {
                let doc = CanonicalDocument::parse(
                    &bytes,
                    Limits {
                        max_bytes: 4096,
                        max_depth: 8,
                        max_nodes: 256,
                    },
                )?;
                serde_json::from_value(doc.value().clone()).map_err(|_| {
                    audit_error(ErrorCode::InvalidValue, "invalid coverage declaration")
                })?
            }
            None => reference::Coverage::default(),
        };
        let sandbox = QualifiedDocker::qualify(true, Path::new(&sandbox_root))?;
        reference::run_cancellable(
            &snapshot, &store, plan, &policies, exposure, budget, &sandbox, coverage, &cancel,
        )
    })?;
    Ok(GymReferenceAuditHandle {
        value: Arc::new(value),
    })
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<GymValidationHandle>()?;
    module.add_class::<GymApprovalHandle>()?;
    module.add_class::<GymReferenceAuditHandle>()?;
    module.add_class::<GymAuditCancellationHandle>()?;
    module.add_function(wrap_pyfunction!(recover_gym_reference_audits, module)?)?;
    module.add_function(wrap_pyfunction!(run_gym_reference_audit, module)?)?;
    module.add_function(wrap_pyfunction!(validate_gym, module)?)?;
    module.add_function(wrap_pyfunction!(approve_gym_snapshot, module)?)?;
    Ok(())
}
