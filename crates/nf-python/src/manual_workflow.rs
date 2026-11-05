//! Thin controller-side binding to the same Rust service as `nf study`.
use crate::boundary::{document, worker};
use nf_experiments::manual_workflow;
use nf_surgeon::cancellation::CancellationToken;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyfunction]
fn manual_workflow_request(
    py: Python<'_>,
    operation: String,
    request: &Bound<'_, PyBytes>,
) -> PyResult<String> {
    let bytes = document(py, request)?;
    let cancel = CancellationToken::default();
    worker(py, cancel.clone(), move || {
        let value = manual_workflow::dispatch_cancellable(&operation, &bytes, &cancel)?;
        serde_json::to_string(&value).map_err(|e| {
            nf_core::error::CoreError::new(
                nf_core::error::ErrorCode::InvalidValue,
                "",
                e.to_string(),
            )
        })
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(manual_workflow_request, module)?)?;
    Ok(())
}
