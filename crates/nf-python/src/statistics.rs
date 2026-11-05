//! Native bounded analysis handles, never serialized execution-evidence authority.
use crate::boundary::{canonical, document, token_for, worker, CancellationHandle};
use nf_experiments::statistics::{Analysis, Plan};
use nf_surgeon::cancellation::CancellationToken;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::Arc;

#[pyclass(frozen, module = "neural_foundry._native")]
pub struct StatisticalPlanHandle {
    inner: Arc<Plan>,
}
#[pymethods]
impl StatisticalPlanHandle {
    #[getter]
    fn plan_id(&self) -> String {
        self.inner.id().to_string()
    }
    fn record_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let inner = self.inner.clone();
        let bytes = worker(py, CancellationToken::default(), move || {
            canonical(&inner.record()?)
        })?;
        Ok(PyBytes::new(py, &bytes))
    }
}
#[pyclass(frozen, module = "neural_foundry._native")]
pub struct StatisticalAnalysisHandle {
    inner: Arc<Analysis>,
}
#[pymethods]
impl StatisticalAnalysisHandle {
    #[getter]
    fn analysis_id(&self) -> String {
        self.inner.id().to_string()
    }
    fn report_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let inner = self.inner.clone();
        let bytes = worker(py, CancellationToken::default(), move || {
            canonical(inner.report())
        })?;
        Ok(PyBytes::new(py, &bytes))
    }
}
#[pyfunction]
fn freeze_statistics(
    py: Python<'_>,
    protocol: &Bound<'_, PyBytes>,
) -> PyResult<StatisticalPlanHandle> {
    let bytes = document(py, protocol)?;
    let inner = worker(py, CancellationToken::default(), move || {
        Plan::freeze(&bytes)
    })?;
    Ok(StatisticalPlanHandle {
        inner: Arc::new(inner),
    })
}
#[pyfunction]
#[pyo3(signature=(plan,observations,cancellation=None))]
fn analyze_statistics(
    py: Python<'_>,
    plan: &StatisticalPlanHandle,
    observations: &Bound<'_, PyBytes>,
    cancellation: Option<PyRef<'_, CancellationHandle>>,
) -> PyResult<StatisticalAnalysisHandle> {
    let token = token_for(py, cancellation)?;
    let inner_token = token.clone();
    let plan = plan.inner.clone();
    let bytes = document(py, observations)?;
    let inner = worker(py, token, move || plan.analyze(&bytes, &inner_token))?;
    Ok(StatisticalAnalysisHandle {
        inner: Arc::new(inner),
    })
}
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<StatisticalPlanHandle>()?;
    module.add_class::<StatisticalAnalysisHandle>()?;
    module.add_function(wrap_pyfunction!(freeze_statistics, module)?)?;
    module.add_function(wrap_pyfunction!(analyze_statistics, module)?)?;
    Ok(())
}
