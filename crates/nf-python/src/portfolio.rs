//! Opaque live-source portfolio planning; no source code executes here.
use crate::boundary::{canonical, document, python_error, worker};
use crate::intake::GymSnapshotHandle;
use crate::validation::GymApprovalHandle;
use nf_core::error::{CoreError, ErrorCode};
use nf_experiments::portfolio::{GymSource, Portfolio, Stage};
use nf_surgeon::cancellation::CancellationToken;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::Arc;

#[pyclass(frozen, module = "neural_foundry._native")]
pub struct PortfolioGymHandle {
    inner: Arc<GymSource>,
}
#[pymethods]
impl PortfolioGymHandle {
    #[getter]
    fn gym_id(&self) -> String {
        self.inner.id().to_string()
    }
    fn validation_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.inner.validation_bytes())
    }
    fn manifest_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let inner = self.inner.clone();
        let bytes = worker(py, CancellationToken::default(), move || {
            canonical(inner.manifest())
        })?;
        Ok(PyBytes::new(py, &bytes))
    }
}
#[pyclass(frozen, module = "neural_foundry._native")]
pub struct PortfolioHandle {
    inner: Arc<Portfolio>,
}
#[pymethods]
impl PortfolioHandle {
    #[getter]
    fn portfolio_id(&self) -> String {
        self.inner.id().to_string()
    }
    fn record_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let inner = self.inner.clone();
        let bytes = worker(py, CancellationToken::default(), move || {
            canonical(&inner.record()?)
        })?;
        Ok(PyBytes::new(py, &bytes))
    }
    #[pyo3(signature=(stage,candidate=1))]
    fn schedule_bytes<'py>(
        &self,
        py: Python<'py>,
        stage: String,
        candidate: u64,
    ) -> PyResult<Bound<'py, PyBytes>> {
        if stage.len() > 32 {
            return Err(python_error(
                py,
                CoreError::new(
                    ErrorCode::InvalidValue,
                    "/stage",
                    "bounded canonical stage required",
                ),
            ));
        }
        let inner = self.inner.clone();
        let bytes = worker(py, CancellationToken::default(), move || {
            let stage: Stage =
                serde_json::from_value(serde_json::Value::String(stage)).map_err(|_| {
                    CoreError::new(
                        ErrorCode::InvalidValue,
                        "/stage",
                        "development, selection or final required",
                    )
                })?;
            canonical(&inner.schedule(stage, candidate)?)
        })?;
        Ok(PyBytes::new(py, &bytes))
    }
    fn assess_bytes<'py>(
        &self,
        py: Python<'py>,
        observations: &Bound<'_, PyBytes>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = document(py, observations)?;
        let inner = self.inner.clone();
        let result = worker(py, CancellationToken::default(), move || {
            canonical(&inner.assess(&bytes)?)
        })?;
        Ok(PyBytes::new(py, &result))
    }
}
#[pyfunction]
fn portfolio_builtin(py: Python<'_>, head: String) -> PyResult<PortfolioGymHandle> {
    if head.len() > 128 {
        return Err(python_error(
            py,
            CoreError::new(ErrorCode::LimitExceeded, "/head", "head bound exceeded"),
        ));
    }
    let source = worker(py, CancellationToken::default(), move || {
        GymSource::builtin(&head)
    })?;
    Ok(PortfolioGymHandle {
        inner: Arc::new(source),
    })
}
#[pyfunction]
fn portfolio_user(
    py: Python<'_>,
    snapshot: &GymSnapshotHandle,
    approval: &GymApprovalHandle,
) -> PyResult<PortfolioGymHandle> {
    let store = snapshot.store.clone();
    let snapshot = snapshot.snapshot.clone();
    let approval = approval.value.clone();
    let source = worker(py, CancellationToken::default(), move || {
        GymSource::user(&snapshot, &approval, &store)
    })?;
    Ok(PortfolioGymHandle {
        inner: Arc::new(source),
    })
}
#[pyfunction]
fn compose_portfolio(
    py: Python<'_>,
    portfolio: &Bound<'_, PyBytes>,
    coverage: &Bound<'_, PyBytes>,
    sources: Vec<PyRef<'_, PortfolioGymHandle>>,
) -> PyResult<PortfolioHandle> {
    if sources.len() > 64 {
        return Err(python_error(
            py,
            CoreError::new(
                ErrorCode::LimitExceeded,
                "/sources",
                "at most 64 source witnesses",
            ),
        ));
    }
    let portfolio = document(py, portfolio)?;
    let coverage = document(py, coverage)?;
    let sources = sources.iter().map(|s| s.inner.clone()).collect();
    let inner = worker(py, CancellationToken::default(), move || {
        Portfolio::compose(&portfolio, &coverage, sources)
    })?;
    Ok(PortfolioHandle {
        inner: Arc::new(inner),
    })
}
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PortfolioGymHandle>()?;
    module.add_class::<PortfolioHandle>()?;
    module.add_function(wrap_pyfunction!(portfolio_builtin, module)?)?;
    module.add_function(wrap_pyfunction!(portfolio_user, module)?)?;
    module.add_function(wrap_pyfunction!(compose_portfolio, module)?)?;
    Ok(())
}
