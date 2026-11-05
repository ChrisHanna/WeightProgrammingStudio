//! Immutable hand-constructed fixture handles; no Python model or oracle callback.
use crate::boundary::{canonical, python_error, worker, SourceHandle};
use nf_core::error::{CoreError, ErrorCode};
use nf_experiments::arithmetic::{self, Fixture, Mode};
use nf_surgeon::cancellation::CancellationToken;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::Arc;

#[pyclass(frozen, module = "neural_foundry._native")]
pub struct ArithmeticFixtureHandle {
    inner: Arc<Fixture>,
}
#[pymethods]
impl ArithmeticFixtureHandle {
    #[getter]
    fn source(&self) -> SourceHandle {
        SourceHandle {
            inner: self.inner.source.clone(),
        }
    }
    fn description_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let inner = self.inner.clone();
        let bytes = worker(py, CancellationToken::default(), move || {
            canonical(&inner.description)
        })?;
        Ok(PyBytes::new(py, &bytes))
    }
    fn checkpoint_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.inner.source.checkpoint_bytes())
    }
    fn asset_bytes<'py>(&self, py: Python<'py>, name: String) -> PyResult<Bound<'py, PyBytes>> {
        let data = self.inner.source.assets().get(&name).ok_or_else(|| {
            python_error(
                py,
                CoreError::new(
                    ErrorCode::ScopeViolation,
                    "/asset",
                    "fixture asset is not declared",
                ),
            )
        })?;
        Ok(PyBytes::new(py, data))
    }
}
#[pyfunction]
#[pyo3(signature=(mode="corrupt_sum"))]
fn arithmetic_fixture(py: Python<'_>, mode: &str) -> PyResult<ArithmeticFixtureHandle> {
    let mode: Mode =
        serde_json::from_value(serde_json::Value::String(mode.into())).map_err(|_| {
            python_error(
                py,
                CoreError::new(
                    ErrorCode::UnsupportedCapability,
                    "/mode",
                    "unsupported arithmetic fixture mode",
                ),
            )
        })?;
    let inner = worker(py, CancellationToken::default(), move || {
        arithmetic::build(mode)
    })?;
    Ok(ArithmeticFixtureHandle {
        inner: Arc::new(inner),
    })
}
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ArithmeticFixtureHandle>()?;
    module.add_function(wrap_pyfunction!(arithmetic_fixture, module)?)?;
    Ok(())
}
