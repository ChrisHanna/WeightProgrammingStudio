//! No Python references enter the numerical worker or opaque native handles.
use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use nf_core::error::{CoreError, ErrorCode, Result};
use nf_core::schemas::{ModelManifest, Validated};
use nf_graph::inventory::{verify_inventory, InventoryLimits};
use nf_surgeon::cancellation::CancellationToken;
use nf_surgeon::transaction::{
    execute_cancellable, ExecutionLimits, MaterializationLimits, SmokeSuite, SourceState,
    StateLoadLimits, VerifiedCandidate,
};
use nf_weight_ir::compilation::{compile, CompilePolicy, CompiledPlan, SourceSnapshot};
use nf_weight_ir::program::parse_program;
use nf_weight_ir::search_space::{parse_search_space, resolve_space};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyString};
use serde::Serialize;
use std::cell::Cell;
use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

pyo3::create_exception!(neural_foundry._native, NativeError, PyException);
pyo3::create_exception!(neural_foundry._native, CancelledError, NativeError);
pyo3::create_exception!(neural_foundry._native, NativePanicError, NativeError);
thread_local! { static ACTIVE: Cell<bool> = const { Cell::new(false) }; }
struct ActiveGuard;
impl Drop for ActiveGuard {
    fn drop(&mut self) {
        ACTIVE.set(false);
    }
}
fn error(code: ErrorCode, message: &str) -> CoreError {
    CoreError::new(code, "", message)
}
fn guarded<T>(token: &CancellationToken, work: impl FnOnce() -> Result<T>) -> Result<T> {
    token.check()?;
    if ACTIVE.replace(true) {
        return Err(error(
            ErrorCode::PermissionDenied,
            "native callback reentry denied",
        ));
    }
    let _active = ActiveGuard;
    let result = catch_unwind(AssertUnwindSafe(work)).unwrap_or_else(|_| {
        Err(error(
            ErrorCode::InternalPanic,
            "native boundary contained a panic",
        ))
    });
    token.check()?;
    result
}
pub(crate) fn python_error(py: Python<'_>, error: CoreError) -> PyErr {
    let code = serde_json::to_value(error.code)
        .expect("error code is serializable")
        .as_str()
        .unwrap()
        .to_owned();
    let exception = match error.code {
        ErrorCode::TransactionAborted => CancelledError::new_err(error.message.clone()),
        ErrorCode::InternalPanic => NativePanicError::new_err(error.message.clone()),
        _ => NativeError::new_err(error.message.clone()),
    };
    let value = exception.value(py);
    for (name, value) in [
        ("code", code),
        ("path", error.path),
        ("message", error.message),
    ] {
        if let Err(error) = exception.value(py).setattr(name, value) {
            return error;
        }
    }
    let _ = value;
    exception
}
pub(crate) fn worker<T: Send>(
    py: Python<'_>,
    token: CancellationToken,
    work: impl FnOnce() -> Result<T> + Send,
) -> PyResult<T> {
    let result = py.allow_threads(move || guarded(&token, work));
    py.check_signals()?;
    result.map_err(|error| python_error(py, error))
}
pub(crate) fn canonical<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let bytes = serde_json::to_vec(value)
        .map_err(|_| error(ErrorCode::InvalidValue, "record serialization failed"))?;
    Ok(CanonicalDocument::parse(&bytes, Limits::default())?
        .bytes()
        .to_vec())
}
pub(crate) fn document(py: Python<'_>, bytes: &Bound<'_, PyBytes>) -> PyResult<Vec<u8>> {
    if bytes.as_bytes().len() > Limits::default().max_bytes {
        return Err(python_error(
            py,
            error(ErrorCode::LimitExceeded, "record exceeds 1 MiB"),
        ));
    }
    Ok(bytes.as_bytes().to_vec())
}
fn blobs(
    py: Python<'_>,
    mapping: &Bound<'_, PyDict>,
    limit: u64,
) -> PyResult<BTreeMap<String, Arc<[u8]>>> {
    if mapping.len() > 1024 {
        return Err(python_error(
            py,
            error(ErrorCode::LimitExceeded, "blob count exceeds 1024"),
        ));
    }
    let mut total = 0u64;
    for (key, value) in mapping.iter() {
        let key = key.downcast::<PyString>()?.to_str()?;
        let bytes = value.downcast::<PyBytes>()?.as_bytes();
        total = total.checked_add(bytes.len() as u64).ok_or_else(|| {
            python_error(py, error(ErrorCode::LimitExceeded, "blob size overflow"))
        })?;
        if key.is_empty() || key.len() > 512 || total > limit {
            return Err(python_error(
                py,
                error(
                    ErrorCode::LimitExceeded,
                    "blob names/bytes exceed boundary limit",
                ),
            ));
        }
    }
    mapping
        .iter()
        .map(|(key, value)| {
            Ok((
                key.downcast::<PyString>()?.to_str()?.to_owned(),
                Arc::from(value.downcast::<PyBytes>()?.as_bytes()),
            ))
        })
        .collect()
}
pub(crate) fn token_for(
    py: Python<'_>,
    token: Option<PyRef<'_, CancellationHandle>>,
) -> PyResult<CancellationToken> {
    let token = token.map(|value| value.inner.clone()).unwrap_or_default();
    token.check().map_err(|error| python_error(py, error))?;
    Ok(token)
}

#[pyclass(frozen, module = "neural_foundry._native")]
pub struct CancellationHandle {
    inner: CancellationToken,
}
#[pymethods]
impl CancellationHandle {
    #[new]
    fn new() -> Self {
        Self {
            inner: CancellationToken::default(),
        }
    }
    fn cancel(&self) {
        self.inner.request();
    }
    #[getter]
    fn cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }
}
#[pyclass(frozen, module = "neural_foundry._native")]
pub struct RecordHandle {
    kind: String,
    bytes: Vec<u8>,
    digest: ContentDigest,
}
#[pymethods]
impl RecordHandle {
    #[getter]
    fn kind(&self) -> &str {
        &self.kind
    }
    #[getter]
    fn digest(&self) -> String {
        self.digest.to_string()
    }
    fn json<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.bytes)
    }
}
#[pyfunction(signature=(kind, record_json, cancellation=None))]
pub fn validate_record(
    py: Python<'_>,
    kind: String,
    record_json: &Bound<'_, PyBytes>,
    cancellation: Option<PyRef<'_, CancellationHandle>>,
) -> PyResult<RecordHandle> {
    let token = token_for(py, cancellation)?;
    let bytes = document(py, record_json)?;
    worker(py, token, move || {
        let bytes = match kind.as_str() {
            "program" => canonical(&*parse_program(&bytes)?)?,
            "search_space" => canonical(&*parse_search_space(&bytes)?)?,
            "manifest" => canonical(&*Validated::<ModelManifest>::parse(&bytes)?)?,
            "compile_policy" => canonical(&*Validated::<CompilePolicy>::parse(&bytes)?)?,
            "smoke_suite" => canonical(&*Validated::<SmokeSuite>::parse(&bytes)?)?,
            _ => {
                return Err(error(
                    ErrorCode::UnsupportedCapability,
                    "unknown record kind",
                ))
            }
        };
        Ok(RecordHandle {
            kind,
            digest: ContentDigest::of_bytes(&bytes),
            bytes,
        })
    })
}
#[pyclass(frozen, module = "neural_foundry._native")]
pub struct SourceHandle {
    pub(crate) inner: Arc<SourceState>,
}
#[pymethods]
impl SourceHandle {
    #[getter]
    fn manifest_digest(&self) -> String {
        self.inner.manifest_digest().to_string()
    }
    #[getter]
    fn parameter_digest(&self) -> String {
        self.inner.manifest().parameter_digest.to_string()
    }
    fn forward(&self, py: Python<'_>, input: Vec<f64>) -> PyResult<Vec<f64>> {
        let source = self.inner.clone();
        worker(py, CancellationToken::default(), move || {
            match source.manifest().architecture.id.as_str() {
                "fixture.cpu" => source.infer_linear_fixture(&input, ExecutionLimits::default()),
                "nf.arithmetic.relu" => {
                    source.infer_arithmetic_fixture(&input, ExecutionLimits::default())
                }
                _ => Err(error(
                    ErrorCode::UnsupportedCapability,
                    "native forward factory is not registered",
                )),
            }
        })
    }
    fn manifest_json<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let source = self.inner.clone();
        let bytes = worker(py, CancellationToken::default(), move || {
            canonical(&**source.manifest())
        })?;
        Ok(PyBytes::new(py, &bytes))
    }
    fn tensor_bytes<'py>(&self, py: Python<'py>, name: String) -> PyResult<Bound<'py, PyBytes>> {
        let source = self.inner.clone();
        let bytes = worker(py, CancellationToken::default(), move || {
            source.tensor_bytes(&name)
        })?;
        Ok(PyBytes::new(py, &bytes))
    }
}
#[pyfunction(signature=(manifest_json, checkpoint, assets, cancellation=None, *, max_checkpoint_bytes=67_108_864, max_tensors=1024))]
pub fn load_source(
    py: Python<'_>,
    manifest_json: &Bound<'_, PyBytes>,
    checkpoint: &Bound<'_, PyBytes>,
    assets: &Bound<'_, PyDict>,
    cancellation: Option<PyRef<'_, CancellationHandle>>,
    max_checkpoint_bytes: usize,
    max_tensors: usize,
) -> PyResult<SourceHandle> {
    let token = token_for(py, cancellation)?;
    let manifest = document(py, manifest_json)?;
    if max_checkpoint_bytes == 0 || max_checkpoint_bytes > StateLoadLimits::large_model().max_checkpoint_bytes {
        return Err(python_error(
            py,
            error(ErrorCode::LimitExceeded, "max_checkpoint_bytes exceeds large-model ceiling"),
        ));
    }
    if checkpoint.as_bytes().len() > max_checkpoint_bytes {
        return Err(python_error(
            py,
            error(ErrorCode::LimitExceeded, "checkpoint exceeds configured max_checkpoint_bytes"),
        ));
    }
    let assets = blobs(py, assets, 4_194_304)?;
    let checkpoint = Arc::from(checkpoint.as_bytes());
    worker(py, token, move || {
        Ok(SourceHandle {
            inner: Arc::new(SourceState::load(
                Validated::parse(&manifest)?,
                checkpoint,
                assets,
                StateLoadLimits {
                    max_checkpoint_bytes,
                    max_tensor_bytes: max_checkpoint_bytes as u64,
                    max_tensors,
                    max_asset_bytes: 4_194_304,
                },
            )?),
        })
    })
}
/// Read-only standalone restore. The expected identity must be frozen before
/// export by the controller; neither a bundle nor its receipt supplies authority.
#[cfg(target_os = "linux")]
#[allow(clippy::too_many_arguments)] // Explicit bounded Python keyword arguments.
#[pyfunction(signature=(store_root, export_root, expected_manifest, *, max_checkpoint_bytes=67_108_864, max_asset_bytes=4_194_304, max_total_bytes=73_400_320, max_tensors=1024, cancellation=None))]
pub fn restore_checkpoint(
    py: Python<'_>,
    store_root: String,
    export_root: String,
    expected_manifest: String,
    max_checkpoint_bytes: usize,
    max_asset_bytes: u64,
    max_total_bytes: u64,
    max_tensors: usize,
    cancellation: Option<PyRef<'_, CancellationHandle>>,
) -> PyResult<SourceHandle> {
    use nf_checkpoint::export::{restore_export_with_reader, ExportLimits};
    use nf_checkpoint::store::{ArtifactStore, StoreLimits};
    let token = token_for(py, cancellation)?;
    if store_root.is_empty() || store_root.len() > 4096 || store_root.contains('\0') {
        return Err(python_error(
            py,
            error(ErrorCode::InvalidValue, "bounded store root required"),
        ));
    }
    let limits = ExportLimits {
        max_checkpoint_bytes,
        max_asset_bytes,
        max_total_bytes,
        max_tensors,
    };
    let cancel = token.clone();
    worker(py, token, move || {
        let ceiling = ExportLimits::default();
        if max_checkpoint_bytes == 0
            || max_checkpoint_bytes > ceiling.max_checkpoint_bytes
            || max_asset_bytes == 0
            || max_asset_bytes > ceiling.max_asset_bytes
            || max_total_bytes == 0
            || max_total_bytes > ceiling.max_total_bytes
            || max_tensors == 0
            || max_tensors > ceiling.max_tensors
        {
            return Err(error(
                ErrorCode::LimitExceeded,
                "restore limits exceed native ceilings",
            ));
        }
        let root: ContentDigest = export_root.parse()?;
        let expected: ContentDigest = expected_manifest.parse()?;
        let store = ArtifactStore::open_read_only(
            std::path::Path::new(&store_root),
            StoreLimits {
                total_bytes: 134_217_728,
                max_blob_bytes: 67_108_864,
                max_entries: 4096,
            },
        )?;
        let restored =
            restore_export_with_reader(&root, Some(&expected), limits, |digest, bound| {
                cancel.check()?;
                let bytes = store.read_bounded(digest, bound)?;
                cancel.check()?;
                Ok(bytes)
            })?;
        cancel.check()?;
        let (model, checkpoint, assets) = restored.into_parts();
        let source = SourceState::load(
            model,
            checkpoint,
            assets,
            StateLoadLimits {
                max_checkpoint_bytes,
                max_asset_bytes,
                max_tensors,
                ..StateLoadLimits::default()
            },
        )?;
        Ok(SourceHandle {
            inner: Arc::new(source),
        })
    })
}

#[pyclass(frozen, module = "neural_foundry._native")]
pub struct PlanHandle {
    pub(crate) inner: Arc<CompiledPlan>,
    pub(crate) source: Arc<SourceState>,
}
#[pymethods]
impl PlanHandle {
    #[getter]
    fn digest(&self) -> String {
        self.inner.digest().to_string()
    }
    fn summary_json<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let plan = self.inner.clone();
        let bytes = worker(py, CancellationToken::default(), move || {
            canonical(plan.summary())
        })?;
        Ok(PyBytes::new(py, &bytes))
    }
}
#[pyfunction(signature=(source, program_json, search_space_json, policy_json, operands, cancellation=None))]
pub fn compile_plan(
    py: Python<'_>,
    source: PyRef<'_, SourceHandle>,
    program_json: &Bound<'_, PyBytes>,
    search_space_json: &Bound<'_, PyBytes>,
    policy_json: &Bound<'_, PyBytes>,
    operands: &Bound<'_, PyDict>,
    cancellation: Option<PyRef<'_, CancellationHandle>>,
) -> PyResult<PlanHandle> {
    let token = token_for(py, cancellation)?;
    let source = source.inner.clone();
    let program = document(py, program_json)?;
    let space = document(py, search_space_json)?;
    let policy = document(py, policy_json)?;
    let operands = blobs(py, operands, 67_108_864)?;
    worker(py, token, move || {
        let observations = source.observations();
        let inventory =
            verify_inventory(source.manifest(), &observations, InventoryLimits::default())?;
        let space = resolve_space(&parse_search_space(&space)?, &inventory)?;
        let plan = compile(
            &parse_program(&program)?,
            SourceSnapshot {
                manifest: source.manifest(),
                tensors: &observations,
                checkpoint: source.checkpoint_bytes(),
                assets: source.assets(),
            },
            &space,
            &operands,
            &Validated::<CompilePolicy>::parse(&policy)?,
            &[],
        )?;
        Ok(PlanHandle {
            inner: Arc::new(plan),
            source,
        })
    })
}
#[pyclass(frozen, module = "neural_foundry._native")]
pub struct CandidateHandle {
    pub(crate) inner: Arc<VerifiedCandidate>,
}
#[pymethods]
impl CandidateHandle {
    #[getter]
    fn digest(&self) -> String {
        self.inner.digest().to_string()
    }
    #[getter]
    fn parameter_digest(&self) -> String {
        self.inner.manifest().parameter_digest.to_string()
    }
    fn receipt_json<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let candidate = self.inner.clone();
        let bytes = worker(py, CancellationToken::default(), move || {
            canonical(candidate.receipt())
        })?;
        Ok(PyBytes::new(py, &bytes))
    }
    fn tensor_bytes<'py>(&self, py: Python<'py>, name: String) -> PyResult<Bound<'py, PyBytes>> {
        let candidate = self.inner.clone();
        let bytes = worker(py, CancellationToken::default(), move || {
            candidate.tensor_bytes(&name)
        })?;
        Ok(PyBytes::new(py, &bytes))
    }
    #[pyo3(signature=(cancellation=None))]
    fn snapshot(
        &self,
        py: Python<'_>,
        cancellation: Option<PyRef<'_, CancellationHandle>>,
    ) -> PyResult<SourceHandle> {
        let token = token_for(py, cancellation)?;
        let candidate = self.inner.clone();
        worker(py, token, move || {
            Ok(SourceHandle {
                inner: Arc::new(candidate.materialize_source(MaterializationLimits::default())?),
            })
        })
    }
}
#[pyfunction(signature=(plan, smoke_json, cancellation=None))]
pub fn apply_plan(
    py: Python<'_>,
    plan: PyRef<'_, PlanHandle>,
    smoke_json: &Bound<'_, PyBytes>,
    cancellation: Option<PyRef<'_, CancellationHandle>>,
) -> PyResult<CandidateHandle> {
    let token = token_for(py, cancellation)?;
    let source = plan.source.clone();
    let plan = plan.inner.clone();
    let smoke = document(py, smoke_json)?;
    let runtime_token = token.clone();
    worker(py, token, move || {
        Ok(CandidateHandle {
            inner: Arc::new(execute_cancellable(
                &plan,
                &source,
                &Validated::<SmokeSuite>::parse(&smoke)?,
                ExecutionLimits::default(),
                &runtime_token,
            )?),
        })
    })
}
#[pyfunction]
pub fn canonical_json<'py>(
    py: Python<'py>,
    json: &Bound<'_, PyBytes>,
) -> PyResult<Bound<'py, PyBytes>> {
    let input = document(py, json)?;
    let output = worker(py, CancellationToken::default(), move || {
        Ok(CanonicalDocument::parse(&input, Limits::default())?
            .bytes()
            .to_vec())
    })?;
    Ok(PyBytes::new(py, &output))
}
#[pyfunction]
pub fn content_digest(py: Python<'_>, data: &Bound<'_, PyBytes>) -> PyResult<String> {
    if data.as_bytes().len() > 67_108_864 {
        return Err(python_error(
            py,
            error(ErrorCode::LimitExceeded, "digest input exceeds 64 MiB"),
        ));
    }
    let data: Arc<[u8]> = Arc::from(data.as_bytes());
    worker(py, CancellationToken::default(), move || {
        Ok(ContentDigest::of_bytes(&data).to_string())
    })
}
#[pyfunction]
pub fn negotiate_protocols<'py>(
    py: Python<'py>,
    request_json: &Bound<'_, PyBytes>,
) -> PyResult<Bound<'py, PyBytes>> {
    let input = document(py, request_json)?;
    let output = worker(py, CancellationToken::default(), move || {
        nf_core::protocol::negotiate(&input)
    })?;
    Ok(PyBytes::new(py, &output))
}

#[pyfunction]
pub fn capabilities_json<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
    let output = worker(py, CancellationToken::default(), || {
        canonical(&nf_core::capabilities::m1_manifest())
    })?;
    Ok(PyBytes::new(py, &output))
}

#[pyfunction]
pub fn m2_capabilities_json<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
    let output = worker(py, CancellationToken::default(), || {
        canonical(&nf_core::capabilities::m2_manifest())
    })?;
    Ok(PyBytes::new(py, &output))
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("NativeError", module.py().get_type::<NativeError>())?;
    module.add("CancelledError", module.py().get_type::<CancelledError>())?;
    module.add(
        "NativePanicError",
        module.py().get_type::<NativePanicError>(),
    )?;
    module.add_class::<CancellationHandle>()?;
    module.add_class::<RecordHandle>()?;
    module.add_class::<SourceHandle>()?;
    module.add_class::<PlanHandle>()?;
    module.add_class::<CandidateHandle>()?;
    module.add_function(wrap_pyfunction!(validate_record, module)?)?;
    module.add_function(wrap_pyfunction!(load_source, module)?)?;
    #[cfg(target_os = "linux")]
    module.add_function(wrap_pyfunction!(restore_checkpoint, module)?)?;
    module.add_function(wrap_pyfunction!(compile_plan, module)?)?;
    module.add_function(wrap_pyfunction!(apply_plan, module)?)?;
    module.add_function(wrap_pyfunction!(canonical_json, module)?)?;
    module.add_function(wrap_pyfunction!(capabilities_json, module)?)?;
    module.add_function(wrap_pyfunction!(m2_capabilities_json, module)?)?;
    module.add_function(wrap_pyfunction!(content_digest, module)?)?;
    module.add_function(wrap_pyfunction!(negotiate_protocols, module)?)?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ffi_contains_panics_and_resets_reentry_guard() {
        let token = CancellationToken::default();
        let result: Result<()> = guarded(&token, || panic!("fixture panic"));
        assert_eq!(result.unwrap_err().code, ErrorCode::InternalPanic);
        assert!(guarded(&token, || Ok(())).is_ok());
    }
    #[test]
    fn ffi_cancel_is_sticky_and_suppresses_results() {
        let token = CancellationToken::default();
        let clone = token.clone();
        assert_eq!(
            guarded(&token, || {
                clone.request();
                Ok(7)
            })
            .unwrap_err()
            .code,
            ErrorCode::TransactionAborted
        );
        assert!(guarded(&token, || Ok(8)).is_err());
    }
    #[test]
    fn ffi_worker_releases_gil_and_native_errors_keep_typed_fields() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let token = CancellationToken::default();
            let clone = token.clone();
            let (send, recv) = std::sync::mpsc::channel();
            let thread = std::thread::spawn(move || {
                Python::with_gil(|_| {
                    clone.request();
                    send.send(()).unwrap();
                })
            });
            let outcome = worker(py, token, move || {
                recv.recv_timeout(std::time::Duration::from_secs(5))
                    .map_err(|_| error(ErrorCode::InvalidValue, "GIL was not released"))?;
                Ok(())
            });
            // Join without the GIL, including the failing no-release case.
            py.allow_threads(move || thread.join().unwrap());
            let error = outcome.unwrap_err();
            assert!(error.is_instance_of::<CancelledError>(py));
            assert_eq!(
                error
                    .value(py)
                    .getattr("code")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "TRANSACTION_ABORTED"
            );
            let panic: PyResult<()> =
                worker(py, CancellationToken::default(), || panic!("fixture"));
            assert!(panic.unwrap_err().is_instance_of::<NativePanicError>(py));
        });
    }
    #[test]
    fn ffi_nested_worker_cannot_reenter_and_outer_guard_remains_active() {
        let token = CancellationToken::default();
        guarded(&token, || {
            assert_eq!(
                guarded(&token, || Ok(())).unwrap_err().code,
                ErrorCode::PermissionDenied
            );
            assert_eq!(
                guarded(&token, || Ok(())).unwrap_err().code,
                ErrorCode::PermissionDenied
            );
            Ok(())
        })
        .unwrap();
        assert!(guarded(&token, || Ok(())).is_ok());
    }
}
