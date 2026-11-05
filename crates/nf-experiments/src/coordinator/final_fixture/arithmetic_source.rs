//! Bounded CAS resolution for the real five-head arithmetic checkpoint.
//!
//! This is artifact verification, NOT execution, acceptance, qualification or an
//! operator grant. In particular it does not broaden the legacy linear runner.
use super::*;

const LOAD: StateLoadLimits = StateLoadLimits {
    max_checkpoint_bytes: 4096,
    max_tensor_bytes: 336, // Total for both matrices: (6*2 + 5*6) * sizeof(f64).
    max_tensors: 2,
    max_asset_bytes: 2048,
};

/// Live verified artifact witness; cannot be deserialized from receipt JSON.
/// No study or publication authority is carried by this type.
pub struct ArithmeticSource {
    state: SourceState,
    identity: ModelIdentity,
}
impl ArithmeticSource {
    pub fn state(&self) -> &SourceState {
        &self.state
    }
    pub fn identity(&self) -> &ModelIdentity {
        &self.identity
    }
}

/// Derive every identity axis from actual state, including the arithmetic graph
/// asset (never the unrelated 2x2 graph returned by `final_fixture::graph_bytes`).
pub fn identity(state: &SourceState) -> Result<ModelIdentity> {
    validate(state)?;
    Ok(ModelIdentity {
        checkpoint: ContentDigest::of_bytes(state.checkpoint_bytes()),
        manifest: state.manifest_digest().clone(),
        parameters: state.manifest().parameter_digest.clone(),
        buffers: tensor_set_digest(&state.manifest().buffers)?,
        graph: ContentDigest::of_bytes(&state.assets()["graph"]),
        config: state.manifest().architecture.config.clone(),
        tokenizer: None,
        system_state: None,
        runtime_artifact: None,
    })
}

fn validate(state: &SourceState) -> Result<()> {
    state.verify_integrity()?;
    let manifest = state.manifest();
    if manifest.architecture.id != "nf.arithmetic.relu"
        || manifest.architecture.version != "1"
        || manifest.parameters.len() != 2
        || !manifest.buffers.is_empty()
        || manifest
            .assets
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
            != BTreeSet::from(["checkpoint", "config", "graph", "provenance"])
        || state
            .assets()
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
            != BTreeSet::from(["config", "graph", "provenance"])
        || state.checkpoint_bytes().len() > LOAD.max_checkpoint_bytes as usize
        || state
            .assets()
            .values()
            .any(|asset| asset.len() > LOAD.max_asset_bytes as usize)
    {
        return Err(deny(
            "bounded arithmetic layout and exact inert assets required",
        ));
    }
    // Only the three registered synthetic provenance modes are supported. The
    // reference is constructed locally, with no inference, code import or I/O.
    let provenance = CanonicalDocument::parse(&state.assets()["provenance"], Limits::default())?;
    let mode = serde_json::from_value(provenance.value()["mode"].clone())
        .map_err(|_| deny("registered arithmetic provenance mode required"))?;
    let reference = crate::arithmetic::build(mode)?;
    for name in ["config", "graph", "provenance"] {
        if state.assets()[name] != reference.source.assets()[name] {
            return Err(deny(
                "arithmetic graph/config/provenance differs from registered bytes",
            ));
        }
    }
    for name in ["hidden.weight", "heads.weight"] {
        let actual = manifest
            .parameters
            .get(name)
            .ok_or_else(|| deny("arithmetic matrix absent"))?;
        let expected = &reference.source.manifest().parameters[name];
        if actual.dtype != expected.dtype
            || actual.shape != expected.shape
            || actual.storage_group != expected.storage_group
            || state
                .tensor(name)
                .and_then(|tensor| tensor.f64_values())
                .is_none_or(|values| values.iter().any(|value| !value.is_finite()))
        {
            return Err(deny(
                "arithmetic matrices must be finite independent f64 6x2 and 5x6",
            ));
        }
    }
    Ok(())
}

/// Resolve bounded bytes from CAS, verify the registered layout and all identity
/// axes. Accepts edited weights, not just the fixture's original byte sequence.
/// This does not read a dataset, open holdout, apply a program or run a scorer.
pub fn resolve(store: &ArtifactStore, model: &ModelIdentity) -> Result<ArithmeticSource> {
    if model.tokenizer.is_some() || model.system_state.is_some() || model.runtime_artifact.is_some()
    {
        return Err(deny("arithmetic external state/tokenizer is unsupported"));
    }
    let manifest = Validated::<ModelManifest>::parse(&store.read_bounded(&model.manifest, 65536)?)?;
    // Bound fanout before reading any asset payloads. StateLoadLimits alone is
    // insufficient if a hostile manifest can cause an unbounded asset loop.
    if manifest.assets.len() != 4
        || manifest.parameters.len() != 2
        || !manifest.buffers.is_empty()
        || manifest
            .assets
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
            != BTreeSet::from(["checkpoint", "config", "graph", "provenance"])
    {
        return Err(deny("bounded arithmetic manifest fanout required"));
    }
    if manifest.assets["checkpoint"] != model.checkpoint
        || manifest.assets["graph"] != model.graph
        || manifest.assets["config"] != model.config
    {
        return Err(deny("arithmetic asset identity axes differ"));
    }
    let mut assets = BTreeMap::new();
    for name in ["config", "graph", "provenance"] {
        assets.insert(
            name.into(),
            Arc::from(store.read_bounded(&manifest.assets[name], LOAD.max_asset_bytes)?),
        );
    }
    let state = SourceState::load(
        manifest,
        Arc::from(store.read_bounded(&model.checkpoint, LOAD.max_checkpoint_bytes as u64)?),
        assets,
        LOAD,
    )?;
    let identity = identity(&state)?;
    if identity != *model {
        return Err(deny("resolved arithmetic model identity differs"));
    }
    Ok(ArithmeticSource { state, identity })
}
