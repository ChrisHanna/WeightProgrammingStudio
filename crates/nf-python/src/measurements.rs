//! Read-only fixed trusted CPU measurement. No clock/sample DTO execution witness.
use crate::boundary::{canonical, document, token_for, worker, CancellationHandle, SourceHandle};
use crate::models::TorchRuntimeHandle;
use nf_core::error::{CoreError, ErrorCode};
use nf_experiments::model_worker::{measurement::Measurement, Limits};
use nf_surgeon::cancellation::CancellationToken;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::Arc;

#[pyclass(frozen, module = "neural_foundry._native")]
pub struct MeasurementHandle {
    inner: Arc<Measurement>,
}
#[pymethods]
impl MeasurementHandle {
    #[getter]
    fn receipt_id(&self) -> String {
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
fn measurement_environment<'py>(
    py: Python<'py>,
    runtime: &TorchRuntimeHandle,
) -> PyResult<Bound<'py, PyBytes>> {
    let runtime = runtime.inner.clone();
    let bytes = worker(py, CancellationToken::default(), move || {
        canonical(&runtime.measurement_environment()?)
    })?;
    Ok(PyBytes::new(py, &bytes))
}
#[pyfunction]
#[pyo3(signature=(runtime,original,candidate,protocol,limits,cancellation=None))]
fn measure_protocol(
    py: Python<'_>,
    runtime: &TorchRuntimeHandle,
    original: &SourceHandle,
    candidate: &SourceHandle,
    protocol: &Bound<'_, PyBytes>,
    limits: &Bound<'_, PyBytes>,
    cancellation: Option<PyRef<'_, CancellationHandle>>,
) -> PyResult<MeasurementHandle> {
    let token = token_for(py, cancellation)?;
    let inner_token = token.clone();
    let runtime = runtime.inner.clone();
    let original = original.inner.clone();
    let candidate = candidate.inner.clone();
    let protocol = document(py, protocol)?;
    let limits = document(py, limits)?;
    let inner = worker(py, token, move || {
        let limits: Limits = serde_json::from_slice(&limits).map_err(|_| {
            CoreError::new(
                ErrorCode::InvalidValue,
                "/limits",
                "invalid measurement worker limits",
            )
        })?;
        runtime.measure(&original, &candidate, &protocol, limits, &inner_token)
    })?;
    Ok(MeasurementHandle {
        inner: Arc::new(inner),
    })
}
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<MeasurementHandle>()?;
    module.add_function(wrap_pyfunction!(measurement_environment, module)?)?;
    module.add_function(wrap_pyfunction!(measure_protocol, module)?)?;
    Ok(())
}
