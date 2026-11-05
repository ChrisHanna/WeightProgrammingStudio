//! Typed scientific graph with SQLite persistence (T064).
//!
//! Stores typed nodes and constrained edges (`tests`, `supports`, `refutes`, `contradicts`,
//! `derived_from`, `reproduces`, `fails_to_reproduce`, `supersedes`, `applies_to`, `explains`)
//! with `EvidenceLabel` provenance, causal confirmation checks for `explains`, contradiction
//! preservation, and coordinator-only write authority (workers cannot write directly).

use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use nf_core::error::{CoreError, ErrorCode, Result};
use nf_core::schemas::{AccessClass, EvidenceLabel, SupportGrade, Validate};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::Path;

fn err(code: ErrorCode, path: &str, message: impl Into<String>) -> CoreError {
    CoreError::new(code, path, message)
}

fn digest_json<T: Serialize>(value: &T) -> Result<ContentDigest> {
    let raw = serde_json::to_vec(value)
        .map_err(|e| err(ErrorCode::InvalidValue, "/scientific_graph", e.to_string()))?;
    Ok(CanonicalDocument::parse(&raw, Limits::default())?.digest())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WriteAuthority {
    Coordinator,
    WorkerProcess,
    UntrustedCaller,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScientificNodeKind {
    ModelCheckpoint,
    ArchitectureSite,
    FeatureBasis,
    CircuitCandidate,
    FailureMode,
    Hypothesis,
    WeightProgram,
    PatchRecord,
    ExperimentRun,
    GymEvaluation,
    LibraryRecipe,
    ExternalClaim,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScientificEdgeKind {
    Tests,
    Supports,
    Refutes,
    Contradicts,
    DerivedFrom,
    Reproduces,
    FailsToReproduce,
    Supersedes,
    AppliesTo,
    Explains,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScientificNode {
    pub id: ContentDigest,
    pub kind: ScientificNodeKind,
    pub title: String,
    pub payload_digest: ContentDigest,
    pub evidence_label: EvidenceLabel,
}

impl Validate for ScientificNode {
    fn validate(&self) -> Result<()> {
        if self.title.trim().is_empty() || self.title.len() > 256 {
            return Err(err(
                ErrorCode::InvalidValue,
                "/title",
                "scientific node title must be 1..=256 characters",
            ));
        }
        self.evidence_label.validate()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScientificEdge {
    pub id: ContentDigest,
    pub kind: ScientificEdgeKind,
    pub source_id: ContentDigest,
    pub target_id: ContentDigest,
    pub confirmation_experiment_id: Option<ContentDigest>,
    pub evidence_label: EvidenceLabel,
}

impl ScientificEdge {
    pub fn new(
        kind: ScientificEdgeKind,
        source_id: ContentDigest,
        target_id: ContentDigest,
        confirmation_experiment_id: Option<ContentDigest>,
        evidence_label: EvidenceLabel,
    ) -> Result<Self> {
        evidence_label.validate()?;
        let body = serde_json::json!({
            "kind": kind,
            "source_id": source_id,
            "target_id": target_id,
            "confirmation_experiment_id": confirmation_experiment_id,
            "evidence_label": evidence_label,
        });
        let id = digest_json(&body)?;
        Ok(Self {
            id,
            kind,
            source_id,
            target_id,
            confirmation_experiment_id,
            evidence_label,
        })
    }
}

fn edge_endpoints_valid(
    edge_kind: ScientificEdgeKind,
    source_kind: ScientificNodeKind,
    target_kind: ScientificNodeKind,
) -> bool {
    use ScientificEdgeKind::*;
    use ScientificNodeKind::*;
    match edge_kind {
        Tests => matches!(
            (source_kind, target_kind),
            (
                ExperimentRun | GymEvaluation,
                Hypothesis | CircuitCandidate | WeightProgram | PatchRecord | LibraryRecipe
            )
        ),
        Supports | Refutes => matches!(
            (source_kind, target_kind),
            (
                ExperimentRun | GymEvaluation,
                Hypothesis | CircuitCandidate | ExternalClaim | LibraryRecipe
            )
        ),
        Contradicts => {
            source_kind == target_kind
                || matches!(
                    (source_kind, target_kind),
                    (
                        ExperimentRun | GymEvaluation | Hypothesis,
                        Hypothesis | ExternalClaim | LibraryRecipe
                    )
                )
        }
        DerivedFrom => true,
        Reproduces | FailsToReproduce => matches!(
            (source_kind, target_kind),
            (
                ExperimentRun | GymEvaluation,
                ExperimentRun | ExternalClaim | LibraryRecipe | PatchRecord
            )
        ),
        Supersedes => source_kind == target_kind,
        AppliesTo => matches!(
            (source_kind, target_kind),
            (
                WeightProgram | PatchRecord | LibraryRecipe | Hypothesis | FeatureBasis,
                ArchitectureSite | ModelCheckpoint | FailureMode
            )
        ),
        Explains => matches!(
            (source_kind, target_kind),
            (
                Hypothesis | CircuitCandidate | FeatureBasis,
                FailureMode | GymEvaluation | ArchitectureSite
            )
        ),
    }
}

pub struct ScientificGraphStore {
    conn: Connection,
}

impl ScientificGraphStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path)
            .map_err(|e| err(ErrorCode::InvalidValue, "/sqlite", e.to_string()))?;
        let store = Self { conn };
        store.init_schema()?;
        Ok(store)
    }

    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()
            .map_err(|e| err(ErrorCode::InvalidValue, "/sqlite", e.to_string()))?;
        let store = Self { conn };
        store.init_schema()?;
        Ok(store)
    }

    fn init_schema(&self) -> Result<()> {
        self.conn
            .execute_batch(
                "PRAGMA foreign_keys = ON;
                 CREATE TABLE IF NOT EXISTS scientific_nodes (
                     id TEXT PRIMARY KEY,
                     kind TEXT NOT NULL,
                     access_class TEXT NOT NULL,
                     support_grade TEXT NOT NULL,
                     node_json TEXT NOT NULL
                 ) STRICT;
                 CREATE TABLE IF NOT EXISTS scientific_edges (
                     id TEXT PRIMARY KEY,
                     kind TEXT NOT NULL,
                     source_id TEXT NOT NULL REFERENCES scientific_nodes(id),
                     target_id TEXT NOT NULL REFERENCES scientific_nodes(id),
                     access_class TEXT NOT NULL,
                     support_grade TEXT NOT NULL,
                     edge_json TEXT NOT NULL
                 ) STRICT;",
            )
            .map_err(|e| err(ErrorCode::InvalidValue, "/sqlite/schema", e.to_string()))
    }

    fn require_coordinator(authority: WriteAuthority) -> Result<()> {
        if authority != WriteAuthority::Coordinator {
            return Err(err(
                ErrorCode::PermissionDenied,
                "/authority",
                "workers and untrusted callers cannot write directly to the scientific graph",
            ));
        }
        Ok(())
    }

    pub fn insert_node(&self, authority: WriteAuthority, node: &ScientificNode) -> Result<()> {
        Self::require_coordinator(authority)?;
        node.validate()?;
        let kind_str = serde_json::to_string(&node.kind).unwrap();
        let access_str = serde_json::to_string(&node.evidence_label.access).unwrap();
        let support_str = serde_json::to_string(&node.evidence_label.support).unwrap();
        let raw = serde_json::to_string(node)
            .map_err(|e| err(ErrorCode::InvalidValue, "/node", e.to_string()))?;
        self.conn
            .execute(
                "INSERT INTO scientific_nodes (id, kind, access_class, support_grade, node_json)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    node.id.as_str(),
                    kind_str.trim_matches('"'),
                    access_str.trim_matches('"'),
                    support_str.trim_matches('"'),
                    raw
                ],
            )
            .map_err(|e| err(ErrorCode::DuplicateKey, "/scientific_nodes", e.to_string()))?;
        Ok(())
    }

    pub fn get_node(&self, id: &ContentDigest) -> Result<Option<ScientificNode>> {
        let mut stmt = self
            .conn
            .prepare("SELECT node_json FROM scientific_nodes WHERE id = ?1")
            .map_err(|e| err(ErrorCode::InvalidValue, "/sqlite", e.to_string()))?;
        let mut rows = stmt
            .query(params![id.as_str()])
            .map_err(|e| err(ErrorCode::InvalidValue, "/sqlite", e.to_string()))?;
        if let Some(row) = rows
            .next()
            .map_err(|e| err(ErrorCode::InvalidValue, "/sqlite", e.to_string()))?
        {
            let raw: String = row
                .get(0)
                .map_err(|e| err(ErrorCode::InvalidValue, "/sqlite", e.to_string()))?;
            let node: ScientificNode = serde_json::from_str(&raw)
                .map_err(|e| err(ErrorCode::InvalidValue, "/node_json", e.to_string()))?;
            Ok(Some(node))
        } else {
            Ok(None)
        }
    }

    pub fn insert_edge(&self, authority: WriteAuthority, edge: &ScientificEdge) -> Result<()> {
        Self::require_coordinator(authority)?;
        edge.evidence_label.validate()?;
        if edge.source_id == edge.target_id {
            return Err(err(
                ErrorCode::InvalidValue,
                "/edge",
                "self-referential scientific graph edges are forbidden",
            ));
        }
        let source = self.get_node(&edge.source_id)?.ok_or_else(|| {
            err(
                ErrorCode::SourceMismatch,
                "/source_id",
                "source node does not exist in scientific graph",
            )
        })?;
        let target = self.get_node(&edge.target_id)?.ok_or_else(|| {
            err(
                ErrorCode::SourceMismatch,
                "/target_id",
                "target node does not exist in scientific graph",
            )
        })?;

        if !edge_endpoints_valid(edge.kind, source.kind, target.kind) {
            return Err(err(
                ErrorCode::ScopeViolation,
                "/edge/kind",
                "illegal node kind pair for scientific edge",
            ));
        }

        // Causal `explains` edges require a confirmed ExperimentRun node and CausalConfirmed/Replicated support
        if edge.kind == ScientificEdgeKind::Explains {
            let exp_id = edge.confirmation_experiment_id.as_ref().ok_or_else(|| {
                err(
                    ErrorCode::InvalidValue,
                    "/confirmation_experiment_id",
                    "explains edge requires a confirmation experiment node",
                )
            })?;
            let exp_node = self.get_node(exp_id)?.ok_or_else(|| {
                err(
                    ErrorCode::SourceMismatch,
                    "/confirmation_experiment_id",
                    "confirmation experiment node missing",
                )
            })?;
            if exp_node.kind != ScientificNodeKind::ExperimentRun {
                return Err(err(
                    ErrorCode::ScopeViolation,
                    "/confirmation_experiment_id",
                    "confirmation node must be an ExperimentRun",
                ));
            }
            if !matches!(
                edge.evidence_label.support,
                SupportGrade::InterventionSupported
                    | SupportGrade::IndependentlyConfirmedUnderProtocol
            ) {
                return Err(err(
                    ErrorCode::ScopeViolation,
                    "/evidence_label/support",
                    "explains edge requires intervention_supported or independently_confirmed_under_protocol support grade (suggestive support is rejected)",
                ));
            }
        }

        let kind_str = serde_json::to_string(&edge.kind).unwrap();
        let access_str = serde_json::to_string(&edge.evidence_label.access).unwrap();
        let support_str = serde_json::to_string(&edge.evidence_label.support).unwrap();
        let raw = serde_json::to_string(edge)
            .map_err(|e| err(ErrorCode::InvalidValue, "/edge", e.to_string()))?;
        self.conn
            .execute(
                "INSERT INTO scientific_edges (id, kind, source_id, target_id, access_class, support_grade, edge_json)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    edge.id.as_str(),
                    kind_str.trim_matches('"'),
                    edge.source_id.as_str(),
                    edge.target_id.as_str(),
                    access_str.trim_matches('"'),
                    support_str.trim_matches('"'),
                    raw
                ],
            )
            .map_err(|e| err(ErrorCode::DuplicateKey, "/scientific_edges", e.to_string()))?;
        Ok(())
    }

    pub fn query_edges_for_node(
        &self,
        node_id: &ContentDigest,
        allowed_access: &[AccessClass],
    ) -> Result<Vec<ScientificEdge>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT edge_json FROM scientific_edges
                 WHERE source_id = ?1 OR target_id = ?1
                 ORDER BY id ASC",
            )
            .map_err(|e| err(ErrorCode::InvalidValue, "/sqlite", e.to_string()))?;
        let rows = stmt
            .query_map(params![node_id.as_str()], |row| row.get::<_, String>(0))
            .map_err(|e| err(ErrorCode::InvalidValue, "/sqlite", e.to_string()))?;
        let mut out = Vec::new();
        for r in rows {
            let raw = r.map_err(|e| err(ErrorCode::InvalidValue, "/sqlite", e.to_string()))?;
            let edge: ScientificEdge = serde_json::from_str(&raw)
                .map_err(|e| err(ErrorCode::InvalidValue, "/edge_json", e.to_string()))?;
            if allowed_access.contains(&edge.evidence_label.access) {
                out.push(edge);
            }
        }
        Ok(out)
    }

    /// Queries both supporting and refuting/contradicting edges for a hypothesis or claim,
    /// ensuring contradictions are preserved alongside supporting evidence.
    pub fn evidence_balance_for_target(
        &self,
        target_id: &ContentDigest,
        allowed_access: &[AccessClass],
    ) -> Result<(Vec<ScientificEdge>, Vec<ScientificEdge>)> {
        let all = self.query_edges_for_node(target_id, allowed_access)?;
        let mut supporting = Vec::new();
        let mut refuting_or_contradicting = Vec::new();
        for edge in all {
            if &edge.target_id == target_id {
                match edge.kind {
                    ScientificEdgeKind::Supports | ScientificEdgeKind::Reproduces => {
                        supporting.push(edge)
                    }
                    ScientificEdgeKind::Refutes
                    | ScientificEdgeKind::Contradicts
                    | ScientificEdgeKind::FailsToReproduce => refuting_or_contradicting.push(edge),
                    _ => {}
                }
            }
        }
        Ok((supporting, refuting_or_contradicting))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nf_core::schemas::EvidenceOrigin;

    fn d(tag: &str) -> ContentDigest {
        ContentDigest::of_bytes(tag.as_bytes())
    }

    fn label(origin: EvidenceOrigin, support: SupportGrade, access: AccessClass) -> EvidenceLabel {
        let evidence = if matches!(
            support,
            SupportGrade::InterventionSupported | SupportGrade::IndependentlyConfirmedUnderProtocol
        ) || origin == EvidenceOrigin::MeasuredResult
        {
            vec![d("evidence-receipt")]
        } else {
            vec![]
        };
        EvidenceLabel {
            schema_version: 1,
            origin,
            support,
            provenance: "unit.test".into(),
            evidence,
            access,
        }
    }

    #[test]
    fn scientific_graph_persists_valid_lineage_preserves_contradictions_and_blocks_illegal_edges() {
        let store = ScientificGraphStore::open_in_memory().unwrap();

        let hyp = ScientificNode {
            id: d("hyp-1"),
            kind: ScientificNodeKind::Hypothesis,
            title: "Head 3 causes carry overflow".into(),
            payload_digest: d("hyp-payload"),
            evidence_label: label(
                EvidenceOrigin::ModelClaim,
                SupportGrade::Suggestive,
                AccessClass::Development,
            ),
        };
        let failure_mode = ScientificNode {
            id: d("fail-1"),
            kind: ScientificNodeKind::FailureMode,
            title: "3-digit carry error".into(),
            payload_digest: d("fail-payload"),
            evidence_label: label(
                EvidenceOrigin::MeasuredResult,
                SupportGrade::InterventionSupported,
                AccessClass::Development,
            ),
        };
        let exp_support = ScientificNode {
            id: d("exp-support"),
            kind: ScientificNodeKind::ExperimentRun,
            title: "Intervention run A".into(),
            payload_digest: d("exp-a"),
            evidence_label: label(
                EvidenceOrigin::MeasuredResult,
                SupportGrade::IndependentlyConfirmedUnderProtocol,
                AccessClass::Development,
            ),
        };
        let exp_refute = ScientificNode {
            id: d("exp-refute"),
            kind: ScientificNodeKind::ExperimentRun,
            title: "Reserve challenge run B".into(),
            payload_digest: d("exp-b"),
            evidence_label: label(
                EvidenceOrigin::MeasuredResult,
                SupportGrade::Contradicted,
                AccessClass::Development,
            ),
        };

        // 1. Worker direct writes must be denied
        assert_eq!(
            store
                .insert_node(WriteAuthority::WorkerProcess, &hyp)
                .unwrap_err()
                .code,
            ErrorCode::PermissionDenied
        );

        // 2. Coordinator inserts nodes
        for n in [&hyp, &failure_mode, &exp_support, &exp_refute] {
            store.insert_node(WriteAuthority::Coordinator, n).unwrap();
        }

        // 3. Both supporting and refuting edges are stored and returned (contradiction preservation)
        let e_sup = ScientificEdge::new(
            ScientificEdgeKind::Supports,
            exp_support.id.clone(),
            hyp.id.clone(),
            Some(exp_support.id.clone()),
            label(
                EvidenceOrigin::MeasuredResult,
                SupportGrade::IndependentlyConfirmedUnderProtocol,
                AccessClass::Development,
            ),
        )
        .unwrap();
        let e_ref = ScientificEdge::new(
            ScientificEdgeKind::Refutes,
            exp_refute.id.clone(),
            hyp.id.clone(),
            Some(exp_refute.id.clone()),
            label(
                EvidenceOrigin::MeasuredResult,
                SupportGrade::Contradicted,
                AccessClass::Development,
            ),
        )
        .unwrap();
        store
            .insert_edge(WriteAuthority::Coordinator, &e_sup)
            .unwrap();
        store
            .insert_edge(WriteAuthority::Coordinator, &e_ref)
            .unwrap();

        let (sup, refut) = store
            .evidence_balance_for_target(&hyp.id, &[AccessClass::Development])
            .unwrap();
        assert_eq!(sup.len(), 1);
        assert_eq!(refut.len(), 1);

        // 4. Unconfirmed suggestive `explains` edge is rejected
        let bad_explains = ScientificEdge::new(
            ScientificEdgeKind::Explains,
            hyp.id.clone(),
            failure_mode.id.clone(),
            Some(exp_support.id.clone()),
            label(
                EvidenceOrigin::ModelClaim,
                SupportGrade::Suggestive,
                AccessClass::Development,
            ),
        )
        .unwrap();
        assert_eq!(
            store
                .insert_edge(WriteAuthority::Coordinator, &bad_explains)
                .unwrap_err()
                .code,
            ErrorCode::ScopeViolation
        );

        // 5. Confirmed `explains` edge with IndependentlyConfirmedUnderProtocol support succeeds
        let good_explains = ScientificEdge::new(
            ScientificEdgeKind::Explains,
            hyp.id.clone(),
            failure_mode.id.clone(),
            Some(exp_support.id.clone()),
            label(
                EvidenceOrigin::MeasuredResult,
                SupportGrade::IndependentlyConfirmedUnderProtocol,
                AccessClass::Development,
            ),
        )
        .unwrap();
        store
            .insert_edge(WriteAuthority::Coordinator, &good_explains)
            .unwrap();
    }
}
