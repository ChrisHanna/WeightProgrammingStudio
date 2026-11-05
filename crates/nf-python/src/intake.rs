//! Controller-owned bytes-only gym intake bindings. No Python import hooks.
use crate::boundary::{canonical, document, python_error, worker};
use nf_checkpoint::store::{ArtifactStore, StoreLimits};
use nf_core::error::{CoreError, ErrorCode};
use nf_experiments::intake::{self, Draft, Snapshot};
use nf_surgeon::cancellation::CancellationToken;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::path::Path;
use std::sync::Arc;

fn path(py: Python<'_>, value: &str) -> PyResult<()> {
    if value.is_empty() || value.len() > 4096 || value.contains('\0') {
        return Err(python_error(
            py,
            CoreError::new(
                ErrorCode::InvalidValue,
                "/path",
                "bounded absolute paths required",
            ),
        ));
    }
    Ok(())
}
fn limits() -> StoreLimits {
    StoreLimits {
        total_bytes: 134217728,
        max_blob_bytes: 1048576,
        max_entries: 1024,
    }
}
#[pyclass(frozen, module = "neural_foundry._native")]
pub struct GymSnapshotHandle {
    pub(crate) snapshot: Arc<Snapshot>,
    pub(crate) store: Arc<ArtifactStore>,
}
#[pymethods]
impl GymSnapshotHandle {
    #[getter]
    fn snapshot_id(&self) -> String {
        self.snapshot.id().to_string()
    }
    fn manifest_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let snapshot = self.snapshot.clone();
        let bytes = worker(py, CancellationToken::default(), move || {
            canonical(snapshot.manifest())
        })?;
        Ok(PyBytes::new(py, &bytes))
    }
    fn asset_bytes<'py>(&self, py: Python<'py>, name: String) -> PyResult<Bound<'py, PyBytes>> {
        let snapshot = self.snapshot.clone();
        let store = self.store.clone();
        let bytes = worker(py, CancellationToken::default(), move || {
            snapshot.asset(&store, &name)
        })?;
        Ok(PyBytes::new(py, &bytes))
    }
}
#[pyfunction]
#[pyo3(signature=(source_root,store_root,draft,selected_file=None))]
fn snapshot_gym(
    py: Python<'_>,
    source_root: String,
    store_root: String,
    draft: &Bound<'_, PyBytes>,
    selected_file: Option<String>,
) -> PyResult<GymSnapshotHandle> {
    path(py, &source_root)?;
    path(py, &store_root)?;
    let bytes = document(py, draft)?;
    let (snapshot, store) = worker(py, CancellationToken::default(), move || {
        let draft = Draft::parse(&bytes)?;
        if let Some(selected) = selected_file {
            let main = match &draft.interface {
                intake::Interface::SingleShot { evaluate, .. } => &evaluate.file,
                intake::Interface::Episode { factory, .. } => &factory.file,
            };
            if selected != *main {
                return Err(CoreError::new(
                    ErrorCode::ScopeViolation,
                    "/selected_file",
                    "explicit file must match the declared main entry point",
                ));
            }
        }
        let store = ArtifactStore::create(Path::new(&store_root), limits())?;
        let snapshot = intake::snapshot(Path::new(&source_root), &store, draft)?;
        Ok((Arc::new(snapshot), Arc::new(store)))
    })?;
    Ok(GymSnapshotHandle { snapshot, store })
}
#[pyfunction]
fn restore_gym_snapshot(
    py: Python<'_>,
    store_root: String,
    snapshot_id: String,
) -> PyResult<GymSnapshotHandle> {
    path(py, &store_root)?;
    let (snapshot, store) = worker(py, CancellationToken::default(), move || {
        let id = snapshot_id.parse()?;
        let store = ArtifactStore::open_read_only(Path::new(&store_root), limits())?;
        let snapshot = intake::restore(&store, &id)?;
        Ok((Arc::new(snapshot), Arc::new(store)))
    })?;
    Ok(GymSnapshotHandle { snapshot, store })
}
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<GymSnapshotHandle>()?;
    module.add_function(wrap_pyfunction!(snapshot_gym, module)?)?;
    module.add_function(wrap_pyfunction!(restore_gym_snapshot, module)?)?;
    Ok(())
}
