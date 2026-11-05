//! Core export/reload gates. No scientific acceptance or active-pointer update.
use crate::history::VerifiedHistory;
use crate::transaction::{
    ExecutionLimits, MaterializationLimits, SmokeSuite, SourceState, StateLoadLimits,
    TransactionReceipt, VerifiedCandidate,
};
use nf_checkpoint::export::{publish_export, restore_export, ExportLimits};
use nf_checkpoint::store::ArtifactStore;
use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use nf_core::error::{CoreError, ErrorCode, Result};
use nf_core::schemas::Validated;
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct CoreExportReceipt {
    pub schema_version: u32,
    pub stage: String,
    pub promotion_status: String,
    pub candidate: ContentDigest,
    pub export_root: ContentDigest,
    pub exported_model: ContentDigest,
    pub transaction: TransactionReceipt,
    pub reload_instance: String,
    pub smoke_suite: ContentDigest,
    pub smoke_outputs: ContentDigest,
    pub smoke_work_upper_bound: u64,
}
#[derive(Debug, Clone, Copy, Default)]
pub struct CoreExportLimits {
    pub materialization: MaterializationLimits,
    pub export: ExportLimits,
    pub reload: StateLoadLimits,
    pub smoke: ExecutionLimits,
}
pub fn restore_source(
    store: &ArtifactStore,
    root: &ContentDigest,
    limits: CoreExportLimits,
) -> Result<SourceState> {
    let loaded = restore_export(store, root, limits.export)?;
    SourceState::load(
        loaded.model().clone(),
        loaded.checkpoint().clone(),
        loaded.assets().clone(),
        limits.reload,
    )
}
pub fn export_candidate(
    store: &ArtifactStore,
    candidate: &VerifiedCandidate,
    suite: &Validated<SmokeSuite>,
    limits: CoreExportLimits,
) -> Result<CoreExportReceipt> {
    let source = candidate.materialize_source(limits.materialization)?;
    export_verified_source(
        store,
        &source,
        candidate.digest(),
        candidate.receipt(),
        suite,
        limits,
    )
}

pub fn export_history(
    store: &ArtifactStore,
    history: &VerifiedHistory,
    suite: &Validated<SmokeSuite>,
    limits: CoreExportLimits,
) -> Result<CoreExportReceipt> {
    export_verified_source(
        store,
        history.source_state(),
        history.candidate_id(),
        history.receipt(),
        suite,
        limits,
    )
}

fn export_verified_source(
    store: &ArtifactStore,
    source: &SourceState,
    candidate: &ContentDigest,
    receipt: &TransactionReceipt,
    suite: &Validated<SmokeSuite>,
    limits: CoreExportLimits,
) -> Result<CoreExportReceipt> {
    if suite.digest()? != receipt.smoke_suite {
        return Err(CoreError::new(
            ErrorCode::SourceMismatch,
            "",
            "export smoke suite differs from transaction",
        ));
    }
    // Verify a fresh instance before committing any root.
    let (outputs, _) = source.verify_smoke(suite, limits.smoke)?;
    if outputs != receipt.smoke_outputs {
        return Err(CoreError::new(
            ErrorCode::SmokeFailed,
            "",
            "materialized inference outputs differ",
        ));
    }
    let root = publish_export(
        store,
        source.manifest(),
        source.checkpoint_bytes().into(),
        source.assets().clone(),
        limits.export,
    )?;
    // Read through the store again: no candidate tensor handle enters reload.
    let reloaded = restore_source(store, &root, limits)?;
    let (outputs, work) = reloaded.verify_smoke(suite, limits.smoke)?;
    if **reloaded.manifest() != **source.manifest() || outputs != receipt.smoke_outputs {
        return Err(CoreError::new(
            ErrorCode::SmokeFailed,
            "",
            "export reload identity/parity differs",
        ));
    }
    let raw = serde_json::to_vec(&**source.manifest())
        .map_err(|error| CoreError::new(ErrorCode::InvalidValue, "", error.to_string()))?;
    Ok(CoreExportReceipt {
        schema_version: 1,
        stage: "standalone_export_and_core_reload".to_owned(),
        promotion_status: "not_evaluated".to_owned(),
        candidate: candidate.clone(),
        export_root: root,
        exported_model: CanonicalDocument::parse(&raw, Limits::default())?.digest(),
        transaction: receipt.clone(),
        reload_instance: "fresh_native_cpu_instance_in_process".to_owned(),
        smoke_suite: suite.digest()?,
        smoke_outputs: outputs,
        smoke_work_upper_bound: work,
    })
}
