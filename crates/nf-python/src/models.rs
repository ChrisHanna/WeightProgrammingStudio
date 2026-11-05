//! Controller-owned fixed trusted model workers. Only data and receipts cross.
use crate::boundary::{
    canonical, document, python_error, token_for, worker, CancellationHandle, CandidateHandle,
    PlanHandle, SourceHandle,
};
use nf_core::error::{CoreError, ErrorCode};
use nf_experiments::model_worker::{Inference, Limits, QualifiedTorch};
use nf_surgeon::cancellation::CancellationToken;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::path::Path;
use std::sync::Arc;

#[pyclass(frozen, module = "neural_foundry._native")]
pub struct TorchRuntimeHandle {
    pub(crate) inner: Arc<QualifiedTorch>,
}
#[pymethods]
impl TorchRuntimeHandle {
    #[getter]
    fn runtime_id(&self) -> String {
        self.inner.identity().to_string()
    }
}
#[pyclass(frozen, module = "neural_foundry._native")]
pub struct ModelInferenceHandle {
    inner: Arc<Inference>,
}
#[pymethods]
impl ModelInferenceHandle {
    #[getter]
    fn status(&self) -> String {
        self.inner.receipt().status.clone()
    }
    #[getter]
    fn outputs(&self) -> Vec<Vec<f64>> {
        self.inner.receipt().outputs.clone()
    }
    fn receipt_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let inner = self.inner.clone();
        let bytes = worker(py, CancellationToken::default(), move || {
            canonical(inner.receipt())
        })?;
        Ok(PyBytes::new(py, &bytes))
    }
}
#[pyfunction]
#[pyo3(signature=(python,site_packages,native_extension,scratch_root,enabled=false))]
fn qualify_torch_worker(
    py: Python<'_>,
    python: String,
    site_packages: String,
    native_extension: String,
    scratch_root: String,
    enabled: bool,
) -> PyResult<TorchRuntimeHandle> {
    if [
        python.len(),
        site_packages.len(),
        native_extension.len(),
        scratch_root.len(),
    ]
    .iter()
    .any(|n| *n > 4096)
    {
        return Err(python_error(
            py,
            CoreError::new(
                ErrorCode::LimitExceeded,
                "/paths",
                "runtime path bound exceeded",
            ),
        ));
    }
    let value = worker(py, CancellationToken::default(), move || {
        QualifiedTorch::qualify(
            enabled,
            Path::new(&python),
            Path::new(&site_packages),
            Path::new(&native_extension),
            Path::new(&scratch_root),
        )
    })?;
    Ok(TorchRuntimeHandle {
        inner: Arc::new(value),
    })
}
#[pyfunction]
#[pyo3(signature=(runtime,source,inputs,limits,cancellation=None))]
fn infer_model_worker(
    py: Python<'_>,
    runtime: &TorchRuntimeHandle,
    source: &SourceHandle,
    inputs: Vec<Vec<f64>>,
    limits: &Bound<'_, PyBytes>,
    cancellation: Option<PyRef<'_, CancellationHandle>>,
) -> PyResult<ModelInferenceHandle> {
    let token = token_for(py, cancellation)?;
    let limits = document(py, limits)?;
    let runtime = runtime.inner.clone();
    let source = source.inner.clone();
    let inner_token = token.clone();
    let result = worker(py, token, move || {
        let limits: Limits = serde_json::from_slice(&limits).map_err(|_| {
            CoreError::new(
                ErrorCode::InvalidValue,
                "/limits",
                "invalid model worker limits",
            )
        })?;
        runtime.infer(&source, inputs, limits, &inner_token)
    })?;
    Ok(ModelInferenceHandle {
        inner: Arc::new(result),
    })
}
#[pyclass(frozen, module = "neural_foundry._native")]
pub struct TorchEditHandle {
    candidate: Arc<nf_surgeon::transaction::VerifiedCandidate>,
    receipt: nf_experiments::model_worker::edit::EditReceipt,
}
#[pymethods]
impl TorchEditHandle {
    #[getter]
    fn candidate(&self) -> CandidateHandle {
        CandidateHandle {
            inner: self.candidate.clone(),
        }
    }
    fn receipt_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let receipt = self.receipt.clone();
        let bytes = worker(py, CancellationToken::default(), move || {
            canonical(&receipt)
        })?;
        Ok(PyBytes::new(py, &bytes))
    }
}
#[pyfunction]
#[pyo3(signature=(runtime,plan,smoke,limits,cancellation=None))]
fn edit_model_worker(
    py: Python<'_>,
    runtime: &TorchRuntimeHandle,
    plan: &PlanHandle,
    smoke: &Bound<'_, PyBytes>,
    limits: &Bound<'_, PyBytes>,
    cancellation: Option<PyRef<'_, CancellationHandle>>,
) -> PyResult<TorchEditHandle> {
    let token = token_for(py, cancellation)?;
    let runtime = runtime.inner.clone();
    let source = plan.source.clone();
    let plan = plan.inner.clone();
    let smoke = document(py, smoke)?;
    let limits = document(py, limits)?;
    let runtime_token = token.clone();
    let (candidate, receipt) = worker(py, token, move || {
        let limits: Limits = serde_json::from_slice(&limits).map_err(|_| {
            CoreError::new(
                ErrorCode::InvalidValue,
                "/limits",
                "invalid model worker limits",
            )
        })?;
        Ok(runtime
            .edit(
                &plan,
                &source,
                &nf_core::schemas::Validated::<nf_surgeon::transaction::SmokeSuite>::parse(&smoke)?,
                limits,
                &runtime_token,
            )?
            .into_parts())
    })?;
    Ok(TorchEditHandle {
        candidate: Arc::new(candidate),
        receipt,
    })
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<TorchEditHandle>()?;
    module.add_function(wrap_pyfunction!(edit_model_worker, module)?)?;
    module.add_class::<TorchRuntimeHandle>()?;
    module.add_class::<ModelInferenceHandle>()?;
    module.add_function(wrap_pyfunction!(qualify_torch_worker, module)?)?;
    module.add_function(wrap_pyfunction!(infer_model_worker, module)?)?;
    Ok(())
}
