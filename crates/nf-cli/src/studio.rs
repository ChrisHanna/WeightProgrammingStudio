//! Weight Programming Studio CLI commands (`nf studio ...` / `weightctl studio ...`).
//!
//! Exposes architecture shape solving (`solve-shape`), structural IR lowering
//! (`lower-structural`), patch conflict composition (`compose-patches`),
//! coordinator-only scientific graph operations (`scientific-graph`), and
//! versioned library lifecycle checks (`library`).

use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use nf_core::error::{CoreError, ErrorCode, Result};
use nf_experiments::library_lifecycle::{
    LibraryEntryStatus, LibraryRecipeVersion, VerifiedLibraryCatalog,
};
use nf_experiments::scientific_graph::{
    ScientificEdge, ScientificGraphStore, ScientificNode, WriteAuthority,
};
use nf_graph::shape_solver::ArchitectureShapeGraph;
use nf_patches::composition::{compose_patches, PatchSpec, ReconciliationMode};
use nf_weight_ir::program::Preservation;
use nf_weight_ir::structural_lowering::{lower_structural_operation, StructuralOperation};
use serde::Deserialize;
use serde_json::{json, Value};

fn err(code: ErrorCode, path: &str, message: impl Into<String>) -> CoreError {
    CoreError::new(code, path, message)
}

fn read_json_file(path: &str) -> Result<Vec<u8>> {
    let raw = crate::commands::read_file(path, Limits::default().max_bytes)?;
    let doc = CanonicalDocument::parse(&raw, Limits::default())?;
    Ok(doc.bytes().to_vec())
}

fn wrap_ok(result: Value) -> Result<String> {
    let payload = json!({
        "schema_version": 1,
        "ok": true,
        "result": result,
    });
    let mut s = serde_json::to_string(&payload)
        .map_err(|e| err(ErrorCode::InvalidValue, "/studio", e.to_string()))?;
    s.push('\n');
    Ok(s)
}

pub(crate) fn dispatch(args: &[String]) -> Option<Result<String>> {
    let [group, subcommand, rest @ ..] = args else {
        return None;
    };
    if group != "studio" {
        return None;
    }
    Some(match subcommand.as_str() {
        "solve-shape" => handle_solve_shape(rest),
        "lower-structural" => handle_lower_structural(rest),
        "compose-patches" => handle_compose_patches(rest),
        "scientific-graph" => handle_scientific_graph(rest),
        "library" => handle_library(rest),
        _ => Err(err(
            ErrorCode::InvalidValue,
            "/studio",
            "expected `studio solve-shape|lower-structural|compose-patches|scientific-graph|library <FILE.json>`",
        )),
    })
}

fn single_file_arg(rest: &[String], sub: &str) -> Result<Vec<u8>> {
    let [file] = rest else {
        return Err(err(
            ErrorCode::InvalidValue,
            "/studio",
            format!("expected `studio {sub} <FILE.json>`"),
        ));
    };
    read_json_file(file)
}

fn handle_solve_shape(rest: &[String]) -> Result<String> {
    let raw = single_file_arg(rest, "solve-shape")?;
    let graph: ArchitectureShapeGraph = serde_json::from_slice(&raw)
        .map_err(|e| err(ErrorCode::InvalidValue, "/shape_solver", e.to_string()))?;
    let solved = graph.solve()?;
    let val = serde_json::to_value(solved)
        .map_err(|e| err(ErrorCode::InvalidValue, "/shape_solver", e.to_string()))?;
    wrap_ok(val)
}

fn handle_lower_structural(rest: &[String]) -> Result<String> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Req {
        operation: StructuralOperation,
        claimed_preservation: Preservation,
        search_space_digest: ContentDigest,
        space_allows_shape_changes: bool,
        hidden_size: u32,
    }
    let raw = single_file_arg(rest, "lower-structural")?;
    let req: Req = serde_json::from_slice(&raw).map_err(|e| {
        err(
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
    let val = serde_json::to_value(plan).map_err(|e| {
        err(
            ErrorCode::InvalidValue,
            "/structural_lowering",
            e.to_string(),
        )
    })?;
    wrap_ok(val)
}

fn handle_compose_patches(rest: &[String]) -> Result<String> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Req {
        patches: Vec<PatchSpec>,
        reconciliation: ReconciliationMode,
    }
    let raw = single_file_arg(rest, "compose-patches")?;
    let req: Req = serde_json::from_slice(&raw)
        .map_err(|e| err(ErrorCode::InvalidValue, "/composition", e.to_string()))?;
    let composed = compose_patches(&req.patches, req.reconciliation)?;
    let val = serde_json::to_value(composed)
        .map_err(|e| err(ErrorCode::InvalidValue, "/composition", e.to_string()))?;
    wrap_ok(val)
}

fn handle_scientific_graph(rest: &[String]) -> Result<String> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Req {
        authority: WriteAuthority,
        nodes: Vec<ScientificNode>,
        edges: Vec<ScientificEdge>,
        query_target_id: ContentDigest,
        allowed_access: Vec<nf_core::schemas::AccessClass>,
    }
    let raw = single_file_arg(rest, "scientific-graph")?;
    let req: Req = serde_json::from_slice(&raw)
        .map_err(|e| err(ErrorCode::InvalidValue, "/scientific_graph", e.to_string()))?;
    let store = ScientificGraphStore::open_in_memory()?;
    for node in &req.nodes {
        store.insert_node(req.authority, node)?;
    }
    for edge in &req.edges {
        store.insert_edge(req.authority, edge)?;
    }
    let (supporting, refuting) =
        store.evidence_balance_for_target(&req.query_target_id, &req.allowed_access)?;
    wrap_ok(json!({
        "target_id": req.query_target_id,
        "supporting_count": supporting.len(),
        "refuting_or_contradicting_count": refuting.len(),
        "supporting_edges": supporting,
        "refuting_or_contradicting_edges": refuting,
    }))
}

fn handle_library(rest: &[String]) -> Result<String> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RegisterItem {
        entry: LibraryRecipeVersion,
        initial_status: LibraryEntryStatus,
        study_digest: ContentDigest,
        reason: String,
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Req {
        registrations: Vec<RegisterItem>,
        query_recipe_id: String,
    }
    let raw = single_file_arg(rest, "library")?;
    let req: Req = serde_json::from_slice(&raw)
        .map_err(|e| err(ErrorCode::InvalidValue, "/library", e.to_string()))?;
    let mut catalog = VerifiedLibraryCatalog::new();
    for item in req.registrations {
        catalog.register_version(
            item.entry,
            item.initial_status,
            item.study_digest,
            item.reason,
        )?;
    }
    let reusable: Vec<_> = catalog
        .reusable_verified_versions(&req.query_recipe_id)
        .into_iter()
        .cloned()
        .collect();
    wrap_ok(json!({
        "recipe_id": req.query_recipe_id,
        "reusable_verified_count": reusable.len(),
        "reusable_versions": reusable,
    }))
}
