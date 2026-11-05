//! Versioned verified-library lifecycle (T090 / CG-13).
//!
//! Stores immutable library recipe entries and append-only status transitions
//! (`proposed`, `verified`, `disputed`, `withdrawn`). Disputed or withdrawn
//! versions remain inspectable with full contradiction history, while default
//! reuse queries return only `verified` versions.

use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use nf_core::error::{CoreError, ErrorCode, Result};
use nf_core::schemas::{EvidenceLabel, MethodRecord, Validate};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

fn err(code: ErrorCode, path: &str, message: impl Into<String>) -> CoreError {
    CoreError::new(code, path, message)
}

fn digest_json<T: Serialize>(value: &T) -> Result<ContentDigest> {
    let raw = serde_json::to_vec(value)
        .map_err(|e| err(ErrorCode::InvalidValue, "/library_lifecycle", e.to_string()))?;
    Ok(CanonicalDocument::parse(&raw, Limits::default())?.digest())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LibraryEntryStatus {
    Proposed,
    Verified,
    Disputed,
    Withdrawn,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LibraryRecipeVersion {
    pub recipe_id: String,
    pub version: u32,
    pub version_digest: ContentDigest,
    pub source_model_checkpoint: ContentDigest,
    pub program_digest: ContentDigest,
    pub patch_record_digest: ContentDigest,
    pub required_architecture_family: String,
    pub prerequisites: Vec<String>,
    pub known_failure_modes: Vec<String>,
    pub method: MethodRecord,
    pub initial_evidence: EvidenceLabel,
}

impl Validate for LibraryRecipeVersion {
    fn validate(&self) -> Result<()> {
        if self.recipe_id.trim().is_empty() || self.version == 0 {
            return Err(err(
                ErrorCode::InvalidValue,
                "/recipe_id",
                "non-empty recipe_id and positive version required",
            ));
        }
        self.method.validate()?;
        self.initial_evidence.validate()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LibraryStatusEvent {
    pub event_id: ContentDigest,
    pub recipe_id: String,
    pub version: u32,
    pub status: LibraryEntryStatus,
    pub reason: String,
    pub study_or_contradiction_digest: ContentDigest,
    pub evidence_label: EvidenceLabel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LibraryVersionRecord {
    pub entry: LibraryRecipeVersion,
    pub current_status: LibraryEntryStatus,
    pub status_history: Vec<LibraryStatusEvent>,
}

#[derive(Debug, Default, Clone)]
pub struct VerifiedLibraryCatalog {
    versions: BTreeMap<(String, u32), LibraryVersionRecord>,
}

impl VerifiedLibraryCatalog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_version(
        &mut self,
        entry: LibraryRecipeVersion,
        initial_status: LibraryEntryStatus,
        study_digest: ContentDigest,
        reason: impl Into<String>,
    ) -> Result<ContentDigest> {
        entry.validate()?;
        let key = (entry.recipe_id.clone(), entry.version);
        if self.versions.contains_key(&key) {
            return Err(err(
                ErrorCode::DuplicateKey,
                "/versions",
                "library recipe version already exists; historical entries are immutable",
            ));
        }
        if matches!(
            initial_status,
            LibraryEntryStatus::Disputed | LibraryEntryStatus::Withdrawn
        ) {
            return Err(err(
                ErrorCode::InvalidValue,
                "/initial_status",
                "new recipe version must start as proposed or verified",
            ));
        }
        let reason_str = reason.into();
        let ev_body = serde_json::json!({
            "recipe_id": entry.recipe_id,
            "version": entry.version,
            "status": initial_status,
            "reason": reason_str,
            "study": study_digest,
        });
        let event = LibraryStatusEvent {
            event_id: digest_json(&ev_body)?,
            recipe_id: entry.recipe_id.clone(),
            version: entry.version,
            status: initial_status,
            reason: reason_str,
            study_or_contradiction_digest: study_digest,
            evidence_label: entry.initial_evidence.clone(),
        };
        let digest = entry.version_digest.clone();
        self.versions.insert(
            key,
            LibraryVersionRecord {
                entry,
                current_status: initial_status,
                status_history: vec![event],
            },
        );
        Ok(digest)
    }

    pub fn record_status_event(
        &mut self,
        recipe_id: &str,
        version: u32,
        new_status: LibraryEntryStatus,
        study_or_contradiction_digest: ContentDigest,
        reason: impl Into<String>,
        evidence_label: EvidenceLabel,
    ) -> Result<ContentDigest> {
        evidence_label.validate()?;
        let reason_str = reason.into();
        if reason_str.trim().is_empty() {
            return Err(err(
                ErrorCode::InvalidValue,
                "/reason",
                "non-empty status transition reason required",
            ));
        }
        let key = (recipe_id.to_owned(), version);
        let record = self.versions.get_mut(&key).ok_or_else(|| {
            err(
                ErrorCode::SourceMismatch,
                "/versions",
                "library recipe version not found",
            )
        })?;
        if record.current_status == LibraryEntryStatus::Withdrawn
            && new_status == LibraryEntryStatus::Verified
        {
            return Err(err(
                ErrorCode::ScopeViolation,
                "/status",
                "withdrawn recipe version cannot be silently re-verified in place; publish a new version",
            ));
        }
        let ev_body = serde_json::json!({
            "recipe_id": recipe_id,
            "version": version,
            "prev_events": record.status_history.len(),
            "status": new_status,
            "reason": reason_str,
            "study": study_or_contradiction_digest,
        });
        let event_id = digest_json(&ev_body)?;
        record.current_status = new_status;
        record.status_history.push(LibraryStatusEvent {
            event_id: event_id.clone(),
            recipe_id: recipe_id.to_owned(),
            version,
            status: new_status,
            reason: reason_str,
            study_or_contradiction_digest,
            evidence_label,
        });
        Ok(event_id)
    }

    /// Inspects a specific version (including disputed or withdrawn versions and their full audit history).
    pub fn inspect_version(&self, recipe_id: &str, version: u32) -> Option<&LibraryVersionRecord> {
        self.versions.get(&(recipe_id.to_owned(), version))
    }

    /// Default reuse query: returns only `Verified` versions, never returning `Proposed`, `Disputed`, or `Withdrawn`.
    pub fn reusable_verified_versions(&self, recipe_id: &str) -> Vec<&LibraryVersionRecord> {
        self.versions
            .iter()
            .filter_map(|((id, _), rec)| {
                if id == recipe_id && rec.current_status == LibraryEntryStatus::Verified {
                    Some(rec)
                } else {
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nf_core::schemas::{
        AccessClass, Adaptation, EvidenceOrigin, ProductionHistory, ProposalOrigin, SupportGrade,
        TargetUpdate,
    };

    fn d(tag: &str) -> ContentDigest {
        ContentDigest::of_bytes(tag.as_bytes())
    }

    fn sample_label(support: SupportGrade) -> EvidenceLabel {
        EvidenceLabel {
            schema_version: 1,
            origin: EvidenceOrigin::MeasuredResult,
            support,
            provenance: "library.test".into(),
            evidence: vec![d("receipt")],
            access: AccessClass::Development,
        }
    }

    fn sample_version(recipe_id: &str, version: u32) -> LibraryRecipeVersion {
        LibraryRecipeVersion {
            recipe_id: recipe_id.into(),
            version,
            version_digest: d(&format!("{recipe_id}@{version}")),
            source_model_checkpoint: d("ckpt-1"),
            program_digest: d("prog-1"),
            patch_record_digest: d("patch-1"),
            required_architecture_family: "hf.qwen2.dense".into(),
            prerequisites: vec!["rmsnorm".into(), "swiglu".into()],
            known_failure_modes: vec!["carry_saturation".into()],
            method: MethodRecord {
                schema_version: 1,
                target_update: TargetUpdate::DirectTensorEdit,
                proposal_origin: ProposalOrigin::Scripted,
                adaptation: Adaptation::None,
                production_history: ProductionHistory::DirectOnly,
                inspection_uses_backprop: false,
                parameter_update_uses_backprop: false,
                gradient_derived_update_values: false,
                alignment_fitted: false,
                optimizer_steps: 0,
                changed_parameters: 8,
            },
            initial_evidence: sample_label(SupportGrade::IndependentlyConfirmedUnderProtocol),
        }
    }

    #[test]
    fn library_lifecycle_preserves_immutable_history_and_excludes_disputed_or_withdrawn_from_reuse()
    {
        let mut catalog = VerifiedLibraryCatalog::new();
        let v1 = sample_version("carry.subspace_repair", 1);
        let v2 = sample_version("carry.subspace_repair", 2);

        catalog
            .register_version(
                v1.clone(),
                LibraryEntryStatus::Verified,
                d("study-v1"),
                "Initial verified carry subspace repair",
            )
            .unwrap();

        // Duplicate registration of an existing version is denied (immutability)
        assert_eq!(
            catalog
                .register_version(
                    v1,
                    LibraryEntryStatus::Verified,
                    d("study-dup"),
                    "Attempt overwrite"
                )
                .unwrap_err()
                .code,
            ErrorCode::DuplicateKey
        );

        // Register v2 as verified
        catalog
            .register_version(
                v2,
                LibraryEntryStatus::Verified,
                d("study-v2"),
                "Refined rank-1 projection",
            )
            .unwrap();
        assert_eq!(
            catalog
                .reusable_verified_versions("carry.subspace_repair")
                .len(),
            2
        );

        // Dispute v1 due to contradiction on a new checkpoint, then withdraw it
        catalog
            .record_status_event(
                "carry.subspace_repair",
                1,
                LibraryEntryStatus::Disputed,
                d("contradiction-study"),
                "Failed retention on larger Qwen2 checkpoint",
                sample_label(SupportGrade::Contradicted),
            )
            .unwrap();

        // Disputed v1 is immediately excluded from default reuse, while v2 remains reusable
        let reusable = catalog.reusable_verified_versions("carry.subspace_repair");
        assert_eq!(reusable.len(), 1);
        assert_eq!(reusable[0].entry.version, 2);

        // Withdraw v1 and verify its full status history remains inspectable
        catalog
            .record_status_event(
                "carry.subspace_repair",
                1,
                LibraryEntryStatus::Withdrawn,
                d("withdrawal-study"),
                "Superseded by v2",
                sample_label(SupportGrade::Contradicted),
            )
            .unwrap();

        // Withdrawn version cannot be silently re-verified in place
        assert_eq!(
            catalog
                .record_status_event(
                    "carry.subspace_repair",
                    1,
                    LibraryEntryStatus::Verified,
                    d("illegal-revive"),
                    "Silent revival",
                    sample_label(SupportGrade::IndependentlyConfirmedUnderProtocol),
                )
                .unwrap_err()
                .code,
            ErrorCode::ScopeViolation
        );

        let inspected_v1 = catalog
            .inspect_version("carry.subspace_repair", 1)
            .expect("v1 remains inspectable");
        assert_eq!(inspected_v1.current_status, LibraryEntryStatus::Withdrawn);
        assert_eq!(inspected_v1.status_history.len(), 3);
    }
}
