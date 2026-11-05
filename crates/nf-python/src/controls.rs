//! Opaque completed paired-control evidence, bounded live source cache.
use crate::boundary::{
    canonical, document, python_error, token_for, worker, CancellationHandle, PlanHandle,
};
use crate::models::TorchRuntimeHandle;
use nf_core::error::{CoreError, ErrorCode};
use nf_core::schemas::Validated;
use nf_experiments::controls::{self, PairedControls, SourceCache};
use nf_surgeon::cancellation::CancellationToken;
use nf_surgeon::transaction::SmokeSuite;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::path::Path;
use std::sync::Arc;

#[pyclass(frozen, module = "neural_foundry._native")]
pub struct ControlCacheHandle {
    inner: Arc<SourceCache>,
}
#[pymethods]
impl ControlCacheHandle {
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(SourceCache::default()),
        }
    }
    #[getter]
    fn entries(&self, py: Python<'_>) -> PyResult<usize> {
        let inner = self.inner.clone();
        worker(py, CancellationToken::default(), move || inner.entries())
    }
}
#[pyclass(frozen, module = "neural_foundry._native")]
pub struct PairedControlsHandle {
    inner: Arc<PairedControls>,
}
#[pymethods]
impl PairedControlsHandle {
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
#[pyo3(signature=(cache,runtime,plan,records,store_root,cancellation=None))]
fn run_paired_controls(
    py: Python<'_>,
    cache: &ControlCacheHandle,
    runtime: &TorchRuntimeHandle,
    plan: &PlanHandle,
    records: (Bound<'_, PyBytes>, Bound<'_, PyBytes>),
    store_root: String,
    cancellation: Option<PyRef<'_, CancellationHandle>>,
) -> PyResult<PairedControlsHandle> {
    if store_root.is_empty() || store_root.len() > 4096 || store_root.contains('\0') {
        return Err(python_error(
            py,
            CoreError::new(
                ErrorCode::InvalidValue,
                "/store_root",
                "bounded explicit store root required",
            ),
        ));
    }
    let token = token_for(py, cancellation)?;
    let inner_token = token.clone();
    let smoke = document(py, &records.0)?;
    let protocol = document(py, &records.1)?;
    let cache = cache.inner.clone();
    let runtime = runtime.inner.clone();
    let source = plan.source.clone();
    let plan = plan.inner.clone();
    let result = worker(py, token, move || {
        controls::run(
            controls::RunContext {
                cache: &cache,
                runtime: &runtime,
                plan: &plan,
                source: &source,
                smoke: &Validated::<SmokeSuite>::parse(&smoke)?,
                store_root: Path::new(&store_root),
                cancel: &inner_token,
            },
            &protocol,
        )
    })?;
    Ok(PairedControlsHandle {
        inner: Arc::new(result),
    })
}
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ControlCacheHandle>()?;
    module.add_class::<PairedControlsHandle>()?;
    module.add_function(wrap_pyfunction!(run_paired_controls, module)?)?;
    Ok(())
}
