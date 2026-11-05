//! Python interface to the same authoritative Rust core used by the CLI.

mod arithmetic;
mod boundary;
#[cfg(target_os = "linux")]
mod controls;
#[cfg(target_os = "linux")]
mod intake;
#[cfg(target_os = "linux")]
mod manual_workflow;
#[cfg(target_os = "linux")]
mod measurements;
#[cfg(target_os = "linux")]
mod models;
#[cfg(target_os = "linux")]
mod portfolio;
mod statistics;
#[cfg(target_os = "linux")]
mod status_reports;
#[cfg(target_os = "linux")]
mod validation;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Structural validation only; caller bytes never become execution authority.
#[pyfunction(signature = (task_json, result_json=None))]
fn validate_repair_benchmark<'py>(
    py: Python<'py>,
    task_json: &Bound<'_, PyBytes>,
    result_json: Option<&Bound<'_, PyBytes>>,
) -> PyResult<Bound<'py, PyBytes>> {
    use nf_core::error::{CoreError, ErrorCode};
    use nf_experiments::repair_benchmark::{ResultRecord, Task};
    let task = boundary::document(py, task_json)?;
    let result = result_json
        .map(|bytes| boundary::document(py, bytes))
        .transpose()?;
    let bytes = boundary::worker(py, Default::default(), move || {
        let invalid =
            |message| CoreError::new(ErrorCode::InvalidValue, "repair_benchmark", message);
        if let Some(result) = result {
            ResultRecord::parse(&task, &result).map_err(invalid)?;
            // Return the checked canonical document, not a typed reserialization:
            // f64 cost/budget fields must not change caller record identities.
            Ok(
                nf_core::canonical::CanonicalDocument::parse(&result, Default::default())?
                    .bytes()
                    .to_vec(),
            )
        } else {
            Task::parse(&task).map_err(invalid)?;
            Ok(
                nf_core::canonical::CanonicalDocument::parse(&task, Default::default())?
                    .bytes()
                    .to_vec(),
            )
        }
    })?;
    Ok(PyBytes::new(py, &bytes))
}

#[pyfunction]
fn version() -> &'static str {
    nf_core::VERSION
}

#[pyfunction]
fn workspace_crates() -> Vec<&'static str> {
    nf_core::WORKSPACE_CRATES.to_vec()
}

#[pyfunction]
fn solve_architecture_shape<'py>(
    py: Python<'py>,
    graph_json: &Bound<'_, PyBytes>,
) -> PyResult<Bound<'py, PyBytes>> {
    use nf_core::error::{CoreError, ErrorCode};
    use nf_graph::shape_solver::ArchitectureShapeGraph;
    let raw = boundary::document(py, graph_json)?;
    let out = boundary::worker(py, Default::default(), move || {
        let graph: ArchitectureShapeGraph = serde_json::from_slice(&raw)
            .map_err(|e| CoreError::new(ErrorCode::InvalidValue, "/shape_solver", e.to_string()))?;
        let solved = graph.solve()?;
        let bytes = serde_json::to_vec(&solved)
            .map_err(|e| CoreError::new(ErrorCode::InvalidValue, "/shape_solver", e.to_string()))?;
        Ok(
            nf_core::canonical::CanonicalDocument::parse(&bytes, Default::default())?
                .bytes()
                .to_vec(),
        )
    })?;
    Ok(PyBytes::new(py, &out))
}

#[pyfunction]
fn lower_structural_ir<'py>(
    py: Python<'py>,
    request_json: &Bound<'_, PyBytes>,
) -> PyResult<Bound<'py, PyBytes>> {
    use nf_core::canonical::ContentDigest;
    use nf_core::error::{CoreError, ErrorCode};
    use nf_weight_ir::program::Preservation;
    use nf_weight_ir::structural_lowering::{lower_structural_operation, StructuralOperation};
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Req {
        operation: StructuralOperation,
        claimed_preservation: Preservation,
        search_space_digest: ContentDigest,
        space_allows_shape_changes: bool,
        hidden_size: u32,
    }
    let raw = boundary::document(py, request_json)?;
    let out = boundary::worker(py, Default::default(), move || {
        let req: Req = serde_json::from_slice(&raw).map_err(|e| {
            CoreError::new(
                ErrorCode::InvalidValue,
                "/structural_lowering",
                e.to_string(),
            )
        })?;
        let plan = lower_structural_operation(
            &req.operation,
            &req.claimed_preservation,
            req.search_space_digest,
            req.space_allows_shape_changes,
            req.hidden_size,
        )?;
        let bytes = serde_json::to_vec(&plan).map_err(|e| {
            CoreError::new(
                ErrorCode::InvalidValue,
                "/structural_lowering",
                e.to_string(),
            )
        })?;
        Ok(
            nf_core::canonical::CanonicalDocument::parse(&bytes, Default::default())?
                .bytes()
                .to_vec(),
        )
    })?;
    Ok(PyBytes::new(py, &out))
}

#[pyfunction]
fn compose_patch_specs<'py>(
    py: Python<'py>,
    request_json: &Bound<'_, PyBytes>,
) -> PyResult<Bound<'py, PyBytes>> {
    use nf_core::error::{CoreError, ErrorCode};
    use nf_patches::composition::{compose_patches, PatchSpec, ReconciliationMode};
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Req {
        patches: Vec<PatchSpec>,
        reconciliation: ReconciliationMode,
    }
    let raw = boundary::document(py, request_json)?;
    let out = boundary::worker(py, Default::default(), move || {
        let req: Req = serde_json::from_slice(&raw)
            .map_err(|e| CoreError::new(ErrorCode::InvalidValue, "/composition", e.to_string()))?;
        let composed = compose_patches(&req.patches, req.reconciliation)?;
        let bytes = serde_json::to_vec(&composed)
            .map_err(|e| CoreError::new(ErrorCode::InvalidValue, "/composition", e.to_string()))?;
        Ok(
            nf_core::canonical::CanonicalDocument::parse(&bytes, Default::default())?
                .bytes()
                .to_vec(),
        )
    })?;
    Ok(PyBytes::new(py, &out))
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(version, module)?)?;
    module.add_function(wrap_pyfunction!(workspace_crates, module)?)?;
    module.add_function(wrap_pyfunction!(solve_architecture_shape, module)?)?;
    module.add_function(wrap_pyfunction!(lower_structural_ir, module)?)?;
    module.add_function(wrap_pyfunction!(compose_patch_specs, module)?)?;
    boundary::register(module)?;
    module.add_function(wrap_pyfunction!(validate_repair_benchmark, module)?)?;
    statistics::register(module)?;
    #[cfg(target_os = "linux")]
    status_reports::register(module)?;
    #[cfg(target_os = "linux")]
    manual_workflow::register(module)?;
    arithmetic::register(module)?;
    #[cfg(target_os = "linux")]
    models::register(module)?;
    #[cfg(target_os = "linux")]
    measurements::register(module)?;
    #[cfg(target_os = "linux")]
    portfolio::register(module)?;
    #[cfg(target_os = "linux")]
    controls::register(module)?;
    #[cfg(target_os = "linux")]
    intake::register(module)?;
    #[cfg(target_os = "linux")]
    validation::register(module)?;
    Ok(())
}
