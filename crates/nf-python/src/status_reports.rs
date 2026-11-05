//! Opaque detached reports; Python cannot supply a committed-read DTO.
use crate::boundary::{python_error, worker};
use nf_experiments::status_reports::CommittedSnapshot;
use nf_surgeon::cancellation::CancellationToken;
use pyo3::prelude::*;
use std::path::Path;

#[pyclass(name = "CommittedStatusSnapshot", frozen)]
pub(crate) struct SnapshotHandle {
    snapshot: CommittedSnapshot,
}

#[pymethods]
impl SnapshotHandle {
    fn render_json(&self) -> String {
        self.snapshot.render_json()
    }
    fn render_markdown(&self, py: Python<'_>) -> PyResult<String> {
        self.snapshot
            .render_markdown()
            .map_err(|error| python_error(py, error))
    }
}

#[pyfunction]
fn capture_status_snapshot(
    py: Python<'_>,
    database_path: String,
    contract_digest: String,
) -> PyResult<SnapshotHandle> {
    let snapshot = worker(py, CancellationToken::default(), move || {
        let id = contract_digest.parse()?;
        CommittedSnapshot::capture(Path::new(&database_path), &id)
    })?;
    Ok(SnapshotHandle { snapshot })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<SnapshotHandle>()?;
    module.add_function(wrap_pyfunction!(capture_status_snapshot, module)?)?;
    Ok(())
}
