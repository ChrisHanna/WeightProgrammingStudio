//! Registered synthetic float64 two-input/six-ReLU/five-head CPU adapter.
use super::*;

pub(super) fn graph() -> Result<Vec<u8>> {
    let value = serde_json::json!({"schema_version":1,"kind":"registered_arithmetic_graph","factory":"nf.arithmetic.relu","factory_version":"1","dtype":"f64","nodes":[{"id":"inputs","op":"input","shape":[2]},{"id":"hidden_linear","op":"linear","input":"inputs","weight":"hidden.weight","shape":[6]},{"id":"hidden_relu","op":"relu","input":"hidden_linear","shape":[6]},{"id":"heads","op":"linear","input":"hidden_relu","weight":"heads.weight","shape":[5]}],"outputs":["sum","difference","maximum","minimum","absolute_difference"]});
    Ok(
        CanonicalDocument::parse(&serde_json::to_vec(&value).unwrap(), Limits::default())?
            .bytes()
            .to_vec(),
    )
}
pub(super) fn shape(
    state: &ModelState,
    suite: &SmokeSuite,
    limits: ExecutionLimits,
) -> Result<(usize, usize, u64)> {
    if state.manifest.architecture.id != "nf.arithmetic.relu"
        || state.manifest.architecture.version != "1"
        || suite.adapter != "nf.arithmetic_relu.v1"
    {
        return Err(failure(
            ErrorCode::UnsupportedCapability,
            "arithmetic adapter/factory mismatch",
        ));
    }
    let config = CanonicalDocument::parse(&state.assets["config"], Limits::default())?;
    if config.value()
        != &serde_json::json!({"schema_version":1,"input_dimension":2,"hidden_dimension":6,"heads":["sum","difference","maximum","minimum","absolute_difference"],"domain_min":-8,"domain_max":8,"integer_inputs":true})
    {
        return Err(failure(
            ErrorCode::UnsupportedCapability,
            "arithmetic fixture requires exact registered config",
        ));
    }
    if state.assets.get("graph").map(AsRef::as_ref) != Some(graph()?.as_slice()) {
        return Err(failure(
            ErrorCode::UnsupportedCapability,
            "arithmetic graph must match its actual registered operators",
        ));
    }
    if !state.manifest.buffers.is_empty() || state.manifest.parameters.len() != 2 {
        return Err(failure(
            ErrorCode::UnsupportedCapability,
            "arithmetic fixture has exactly two dense matrices and no buffers",
        ));
    }
    for (name, shape) in [("hidden.weight", vec![6, 2]), ("heads.weight", vec![5, 6])] {
        let tensor = state.tensor(name).ok_or_else(|| {
            failure(
                ErrorCode::UnsupportedCapability,
                "arithmetic matrix missing",
            )
        })?;
        if tensor.dtype() != DType::F64
            || tensor.shape() != shape
            || !state.manifest.parameters.contains_key(name)
        {
            return Err(failure(
                ErrorCode::UnsupportedCapability,
                "arithmetic matrix must be float64 with registered shape",
            ));
        }
    }
    if limits.max_smoke_vector_elements < 6 || limits.max_smoke_vector_elements > 4096 {
        return Err(failure(
            ErrorCode::LimitExceeded,
            "arithmetic vector bound exceeded",
        ));
    }
    for probe in &suite.probes {
        check_input(&probe.input)?;
        if probe
            .expected
            .as_ref()
            .is_some_and(|expected| expected.len() != 5)
        {
            return Err(failure(
                ErrorCode::SmokeFailed,
                "arithmetic output has five heads",
            ));
        }
    }
    let work = mul(suite.probes.len() as u64, 256)?;
    if work > limits.max_smoke_work {
        return Err(failure(
            ErrorCode::LimitExceeded,
            "arithmetic forward work bound exceeded",
        ));
    }
    Ok((5, 2, work))
}
fn check_input(input: &[f64]) -> Result<()> {
    if input.len() != 2
        || input
            .iter()
            .any(|v| !v.is_finite() || v.fract() != 0.0 || !(-8.0..=8.0).contains(v))
    {
        return Err(failure(
            ErrorCode::InvalidValue,
            "synthetic arithmetic domain is integer pairs in [-8,8]",
        ));
    }
    Ok(())
}
pub(super) fn forward(state: &ModelState, input: &[f64]) -> Result<Vec<f64>> {
    check_input(input)?;
    let mut hidden = linear_vector(state.tensor("hidden.weight").unwrap(), input, 6, 2)?;
    for value in &mut hidden {
        *value = value.max(0.0);
    }
    linear_vector(state.tensor("heads.weight").unwrap(), &hidden, 5, 6)
}
