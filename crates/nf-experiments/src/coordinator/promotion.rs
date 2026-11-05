//! Experimental T039 for the trusted native T038 2x2 f64 fixture ONLY.
//! Release requires opaque native qualification AND exact operator execution approval.
//! Native operational qualification is consumed from T038; no independent
//! engineering review or generalization is claimed. Strict-fsync is unsupported.
//! Imported reports and independently supplied reload witnesses are never authority.
use super::budgets::{OutcomeKind, Request, Resources, Settlement, Usage, WorkKind};
use super::*;
use crate::model_worker::{reload::FrozenReloadProbes, Limits as WorkerLimits, QualifiedTorch};
use nf_checkpoint::store::ArtifactStore;
use nf_core::schemas::ModelManifest;
use nf_core::study::{ClaimScope, ModelIdentity};
use nf_graph::inventory::tensor_set_digest;
use nf_surgeon::{
    cancellation::CancellationToken,
    export::{export_candidate, CoreExportLimits},
    transaction::{execute_cancellable, ExecutionLimits, SmokeSuite, SourceState, StateLoadLimits},
};
use nf_weight_ir::compilation::CompiledPlan;
use serde_json::{json, Value};
use std::{collections::BTreeMap, sync::Arc};

pub(super) const MIGRATION_V7: &str = r#"
CREATE TABLE promotion_records(contract TEXT NOT NULL REFERENCES contracts(digest),kind TEXT NOT NULL,body BLOB NOT NULL,digest TEXT NOT NULL,PRIMARY KEY(contract,kind));
CREATE TABLE accepted_records(digest TEXT PRIMARY KEY,contract TEXT NOT NULL UNIQUE REFERENCES contracts(digest),family TEXT NOT NULL,body BLOB NOT NULL);
CREATE TABLE accepted_heads(family TEXT PRIMARY KEY,digest TEXT NOT NULL REFERENCES accepted_records(digest));
CREATE TABLE verified_frontier(digest TEXT PRIMARY KEY REFERENCES accepted_records(digest),family TEXT NOT NULL);
CREATE TRIGGER promotion_no_update BEFORE UPDATE ON promotion_records BEGIN SELECT RAISE(ABORT,'immutable promotion evidence');END;
CREATE TRIGGER promotion_no_delete BEFORE DELETE ON promotion_records BEGIN SELECT RAISE(ABORT,'retained promotion evidence');END;
CREATE TRIGGER accepted_no_update BEFORE UPDATE ON accepted_records BEGIN SELECT RAISE(ABORT,'immutable acceptance');END;
CREATE TRIGGER accepted_no_delete BEFORE DELETE ON accepted_records BEGIN SELECT RAISE(ABORT,'retained acceptance');END;
CREATE TRIGGER frontier_no_update BEFORE UPDATE ON verified_frontier BEGIN SELECT RAISE(ABORT,'immutable frontier');END;
CREATE TRIGGER frontier_no_delete BEFORE DELETE ON verified_frontier BEGIN SELECT RAISE(ABORT,'retained frontier');END;
INSERT INTO budget_migrations VALUES(7,(SELECT value FROM meta WHERE key='schema'));
PRAGMA user_version=7;
"#;
pub(super) fn schema_v7() -> ContentDigest {
    ContentDigest::of_bytes(
        &[
            SCHEMA.as_bytes(),
            budgets::MIGRATION_V2.as_bytes(),
            supervision::MIGRATION_V3.as_bytes(),
            splits::MIGRATION_V4.as_bytes(),
            exposure::MIGRATION_V5.as_bytes(),
            final_fixture::MIGRATION_V6.as_bytes(),
            MIGRATION_V7.as_bytes(),
        ]
        .concat(),
    )
}
const STORAGE: u64 = 4_194_304;
const CPU_MS: u64 = 30_000;
// Exact current T038 fixture scopes, expressly excluding strict T038 and T039.
const FINAL_SCOPE: &str =
    "trusted_native_linear_fixture_only_not_generalization_or_strict_T038_or_T039";
const ARITHMETIC_FINAL_SCOPE: &str = final_fixture::arithmetic_final::SCOPE;
pub(super) fn valid_final_scope(scope: &str) -> bool {
    matches!(scope, FINAL_SCOPE | ARITHMETIC_FINAL_SCOPE)
}
fn hash(value: &impl Serialize) -> Result<ContentDigest> {
    Ok(ContentDigest::of_bytes(&canonical(value)?))
}
fn deny(message: &str) -> CoreError {
    failure(ErrorCode::PermissionDenied, message)
}
fn read(
    c: &Connection,
    contract: &ContentDigest,
    kind: &str,
) -> Result<Option<(ContentDigest, Value)>> {
    let row: Option<(Vec<u8>, String)> = c
        .query_row(
            "SELECT body,digest FROM promotion_records WHERE contract=?1 AND kind=?2",
            params![contract.as_str(), kind],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()
        .map_err(sql)?;
    row.map(|(b, d)| decode(&b, &d)).transpose()
}
fn decode(b: &[u8], d: &str) -> Result<(ContentDigest, Value)> {
    let doc = CanonicalDocument::parse(b, Limits::default())?;
    if doc.bytes() != b || doc.digest().as_str() != d {
        return Err(deny("promotion record integrity mismatch"));
    }
    Ok((d.parse()?, doc.value().clone()))
}
fn record(
    tx: &Transaction<'_>,
    contract: &ContentDigest,
    kind: &str,
    value: &impl Serialize,
) -> Result<ContentDigest> {
    let b = canonical(value)?;
    if b.len() > 524288 {
        return Err(deny("bounded promotion evidence required"));
    }
    let id = ContentDigest::of_bytes(&b);
    tx.execute(
        "INSERT INTO promotion_records VALUES(?1,?2,?3,?4)",
        params![contract.as_str(), kind, b, id.as_str()],
    )
    .map_err(sql)?;
    Ok(id)
}
/// Snapshot of ALL immutable native final rows, never a caller's accepted flag.
/// Rechecked under the publishing writer lock after the native decision resolver.
fn final_chain(c: &Connection, contract: &ContentDigest) -> Result<(ContentDigest, Value)> {
    let mut statement = c
        .prepare(
            "SELECT kind,body,digest FROM final_fixture_records WHERE contract=?1 ORDER BY kind",
        )
        .map_err(sql)?;
    let rows = statement
        .query_map([contract.as_str()], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, Vec<u8>>(1)?,
                r.get::<_, String>(2)?,
            ))
        })
        .map_err(sql)?;
    let mut ids = BTreeMap::new();
    let mut dispatch = None;
    let mut arithmetic_dispatch = None;
    for row in rows {
        let (kind, b, d) = row.map_err(sql)?;
        let (id, v) = decode(&b, &d)?;
        if kind == "dispatch" {
            dispatch = Some(v);
        } else if kind == "arithmetic_dispatch" {
            arithmetic_dispatch = Some(v);
        }
        ids.insert(kind, id);
    }
    if let Some(dispatch) = arithmetic_dispatch {
        for name in [
            "intent",
            "arithmetic_dispatch",
            "arithmetic_execution",
            "arithmetic_decision",
        ] {
            if !ids.contains_key(name) {
                return Err(deny("incomplete native arithmetic final chain"));
            }
        }
        return Ok((hash(&ids)?, dispatch));
    }
    for name in [
        "intent",
        "execution",
        "completion",
        "decision",
        "controls",
        "dispatch",
    ] {
        if !ids.contains_key(name) {
            return Err(deny("incomplete native final chain"));
        }
    }
    Ok((
        hash(&ids)?,
        dispatch.ok_or_else(|| deny("missing native dispatch"))?,
    ))
}
fn head(c: &Connection, family: &ContentDigest) -> Result<Option<ContentDigest>> {
    let value: Option<String> = c
        .query_row(
            "SELECT digest FROM accepted_heads WHERE family=?1",
            [family.as_str()],
            |r| r.get(0),
        )
        .optional()
        .map_err(sql)?;
    value.map(|s| s.parse()).transpose()
}
fn cpu_ms() -> u64 {
    let t = rustix::time::clock_gettime(rustix::time::ClockId::ProcessCPUTime);
    (t.tv_sec as u64 * 1_000_000_000 + t.tv_nsec as u64).div_ceil(1_000_000)
}
fn rss() -> Result<u64> {
    let mut u = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: successful getrusage initializes the valid output pointer.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, u.as_mut_ptr()) } != 0 {
        return Err(deny("controller RSS unavailable"));
    }
    Ok(unsafe { u.assume_init() }.ru_maxrss.max(0) as u64 * 1024)
}
/// The child and controller have independent address spaces. Sampled RSS is
/// diagnostic only: BOTH irreversible hard ceilings must fit the reservation.
fn memory_envelope(child: u64, host: u64) -> Result<u64> {
    let limit = rustix::process::getrlimit(rustix::process::Resource::As);
    let (Some(soft), Some(hard)) = (limit.current, limit.maximum) else {
        return Err(deny(
            "promotion requires a dedicated hard-memory-limited controller",
        ));
    };
    if soft > hard || hard.checked_add(child).is_none_or(|total| total > host) {
        return Err(deny(
            "controller plus child hard address-space ceilings exceed allocation",
        ));
    }
    use std::io::Read;
    let mut stat = String::new();
    std::fs::File::open("/proc/self/statm")
        .and_then(|f| f.take(1024).read_to_string(&mut stat))
        .map_err(|e| failure(ErrorCode::Io, &e.to_string()))?;
    let pages = stat
        .split_whitespace()
        .next()
        .and_then(|n| n.parse::<u64>().ok())
        .ok_or_else(|| deny("missing native mappings"))?;
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page <= 0
        || pages
            .checked_mul(page as u64)
            .is_none_or(|bytes| bytes > soft)
    {
        return Err(deny("controller mappings exceed hard envelope"));
    }
    Ok(hard)
}
fn model(source: &SourceState) -> Result<ModelIdentity> {
    if source.manifest().architecture.id == "nf.arithmetic.relu" {
        return final_fixture::arithmetic_source::identity(source);
    }
    Ok(ModelIdentity {
        checkpoint: ContentDigest::of_bytes(source.checkpoint_bytes()),
        manifest: source.manifest_digest().clone(),
        parameters: source.manifest().parameter_digest.clone(),
        buffers: tensor_set_digest(&source.manifest().buffers)?,
        graph: ContentDigest::of_bytes(final_fixture::graph_bytes()),
        config: source.manifest().architecture.config.clone(),
        tokenizer: None,
        system_state: None,
        runtime_artifact: None,
    })
}
fn source(store: &ArtifactStore, id: &ModelIdentity) -> Result<SourceState> {
    if let Ok(arith_graph) = nf_surgeon::transaction::arithmetic_graph() {
        if id.graph == ContentDigest::of_bytes(&arith_graph) {
            return Ok(final_fixture::arithmetic_source::resolve(store, id)?
                .state()
                .clone());
        }
    }
    if id.graph != ContentDigest::of_bytes(final_fixture::graph_bytes())
        || store.read_bounded(&id.graph, 1024)? != final_fixture::graph_bytes()
        || id.tokenizer.is_some()
        || id.system_state.is_some()
        || id.runtime_artifact.is_some()
    {
        return Err(deny("unsupported source axes"));
    }
    let manifest = Validated::<ModelManifest>::parse(&store.read_bounded(&id.manifest, 65536)?)?;
    if manifest.parameters.len() != 2
        || !manifest.buffers.is_empty()
        || manifest.assets.len() != 2
        || manifest.architecture.id != "fixture.cpu"
        || manifest.architecture.version != "1"
        || manifest.parameters.get("w") != manifest.parameters.get("tied")
        || manifest
            .parameters
            .get("w")
            .is_none_or(|w| w.shape != vec![2, 2] || w.dtype != nf_core::schemas::DType::F64)
    {
        return Err(deny("unsupported source layout"));
    }
    let source = SourceState::load(
        manifest,
        Arc::from(store.read_bounded(&id.checkpoint, 4096)?),
        BTreeMap::from([(
            "config".into(),
            Arc::from(store.read_bounded(&id.config, 2)?),
        )]),
        StateLoadLimits {
            max_checkpoint_bytes: 4096,
            max_tensor_bytes: 32,
            max_tensors: 2,
            max_asset_bytes: 2,
        },
    )?;
    if model(&source)? != *id || source.assets()["config"].as_ref() != b"{}" {
        return Err(deny("source identity mismatch"));
    }
    Ok(source)
}
fn export_limits_for(adapter: &str) -> CoreExportLimits {
    if adapter == "nf.arithmetic_relu.v1" {
        let mut b = CoreExportLimits::default();
        b.materialization.load.max_checkpoint_bytes = 4096;
        b.materialization.load.max_tensor_bytes = 336;
        b.materialization.load.max_tensors = 2;
        b.materialization.load.max_asset_bytes = 2048;
        b.materialization.serialization.max_checkpoint_bytes = 4096;
        b.materialization.serialization.max_tensor_bytes = 336;
        b.materialization.serialization.max_header_bytes = 2048;
        b.materialization.serialization.max_staging_bytes = 8192;
        b.export.max_checkpoint_bytes = 4096;
        b.export.max_asset_bytes = 2048;
        b.export.max_total_bytes = 16384;
        b.export.max_tensors = 2;
        b.reload = b.materialization.load;
        b.smoke = execution_limits_for(adapter);
        return b;
    }
    let mut b = CoreExportLimits::default();
    b.materialization.serialization.max_checkpoint_bytes = 4096;
    b.materialization.serialization.max_tensor_bytes = 32;
    b.materialization.serialization.max_header_bytes = 1024;
    b.materialization.serialization.max_staging_bytes = 4096;
    b.export.max_checkpoint_bytes = 4096;
    b.export.max_asset_bytes = 2;
    b.export.max_total_bytes = 65536;
    b.export.max_tensors = 2;
    b.reload = StateLoadLimits {
        max_checkpoint_bytes: 4096,
        max_tensor_bytes: 32,
        max_tensors: 2,
        max_asset_bytes: 2,
    };
    b.smoke = execution_limits_for(adapter);
    b
}
fn execution_limits_for(adapter: &str) -> ExecutionLimits {
    if adapter == "nf.arithmetic_relu.v1" {
        return ExecutionLimits {
            max_verification_bytes: 16384,
            max_smoke_work: 65536,
            max_smoke_vector_elements: 6,
        };
    }
    ExecutionLimits {
        max_verification_bytes: 4096,
        max_smoke_work: 4096,
        max_smoke_vector_elements: 2,
    }
}
#[cfg(test)]
fn export_limits() -> CoreExportLimits {
    export_limits_for("nf.linear_fixture.v1")
}
#[cfg(test)]
fn execution_limits() -> ExecutionLimits {
    execution_limits_for("nf.linear_fixture.v1")
}
fn fence(
    tx: &Transaction<'_>,
    store: &str,
    lease: &Lease,
    deadline: i64,
    cancel: &CancellationToken,
) -> Result<()> {
    let now = files::moment()?;
    check_lease(tx, store, lease, &now)?;
    cancel.check()?;
    if now.ms >= deadline
        || supervision::stopped(tx, &lease.contract)?
        || !supervision::all_cleanup_confirmed(tx, &lease.contract)?
        || budgets::snapshot_in(tx, &lease.contract)?.accounting_halted
    {
        return Err(deny("promotion stopped, overdue, unclean or budget halted"));
    }
    Ok(())
}
/// A verified immutable local ledger record, not a deserializable credential.
/// ```compile_fail
/// use nf_experiments::coordinator::promotion::Acceptance;
/// let _ = Acceptance { digest: "sha256:0000000000000000000000000000000000000000000000000000000000000000".parse().unwrap(), body: serde_json::json!({}) };
/// ```
#[derive(Debug, Clone, Serialize)]
pub struct Acceptance {
    digest: ContentDigest,
    body: Value,
}
impl Acceptance {
    pub fn digest(&self) -> &ContentDigest {
        &self.digest
    }
    pub fn body(&self) -> &Value {
        &self.body
    }
}
mod history;
pub mod operator;
mod release;
mod supervision_guard;
pub use release::{PromotionExecutionGrant, PromotionQualification, PromotionScope};
mod child_inventory;
mod child_registration;
pub(super) fn child_cleanup_confirmed(c: &Connection, contract: &ContentDigest) -> Result<bool> {
    child_inventory::clean(c, contract)
}
#[path = "final_fixture/native_process.rs"]
mod process;
impl Ledger {
    /// Historical pointer reader checks content, journal, frontier and evidence links.
    /// It does NOT turn archived child JSON back into a live reload witness.
    pub fn accepted_fixture_head(&self, family: &ContentDigest) -> Result<Option<Acceptance>> {
        self.files.verify(self.limits.max_database_bytes)?;
        // The T038 resolver opens its own accounting snapshot. Bind its entire
        // immutable chain before/after, then recheck in our pointer snapshot.
        let expected = head(&self.connection, family)?;
        let audit = if let Some(id) = &expected {
            let contract: String = self
                .connection
                .query_row(
                    "SELECT contract FROM accepted_records WHERE digest=?1",
                    [id.as_str()],
                    |r| r.get(0),
                )
                .map_err(sql)?;
            let contract: ContentDigest = contract.parse()?;
            let chain = final_chain(&self.connection, &contract)?.0;
            let decision = final_fixture::read_any_final_decision_in(
                &self.connection,
                &self.store,
                &contract,
            )?
            .ok_or_else(|| deny("missing native final decision"))?;
            if !decision.report().accepted
                || decision.report().status != acceptance::Status::Passed
                || decision.report().promotion_authorized
                || !valid_final_scope(&decision.report().scope)
                || final_chain(&self.connection, &contract)?.0 != chain
            {
                return Err(deny(
                    "historical native final decision is not passing or differs",
                ));
            }
            Some((contract, chain, decision.id().clone()))
        } else {
            None
        };
        let tx = self.connection.unchecked_transaction().map_err(sql)?;
        if head(&tx, family)? != expected {
            return Err(deny("head changed during historical audit"));
        }
        let result = accepted(&tx, family)?;
        if let Some((contract, chain, decision)) = audit {
            let intent = read(&tx, &contract, "intent")?
                .ok_or_else(|| deny("missing promotion intent"))?
                .1;
            if intent["decision"] != json!(decision) || final_chain(&tx, &contract)?.0 != chain {
                return Err(deny("final chain changed during historical audit"));
            }
        }
        tx.commit().map_err(sql)?;
        Ok(result)
    }
    /// Reconcile an interrupted PRIVATE attempt, never rerun it or publish from
    /// archived reports. Unknown child cleanup deliberately remains reserved.
    pub fn recover_fixture_promotion(&mut self, lease: &Lease) -> Result<()> {
        self.files.verify(self.limits.max_database_bytes)?;
        let (_, intent) = read(&self.connection, &lease.contract, "intent")?
            .ok_or_else(|| deny("missing promotion intent"))?;
        if intent["contract"] != json!(lease.contract) || intent["store"] != self.store {
            return Err(deny("foreign promotion recovery"));
        }
        let id: ContentDigest = serde_json::from_value(intent["reservation"].clone())
            .map_err(|_| deny("missing promotion reservation"))?;
        let request = history::budget_row(&self.connection, "budget_requests", "id", &id)?;
        if request["contract"] != json!(lease.contract) {
            return Err(deny("foreign promotion reservation"));
        }
        if budgets::native_released(&self.connection, &id)? {
            return Ok(());
        }
        let owner: process::Owner = serde_json::from_value(intent["native_process"].clone())
            .map_err(|_| deny("missing native owner"))?;
        let finished = read(&self.connection, &lease.contract, "attempt_end")?;
        let intent_id = hash(&intent)?;
        let terminal = finished.as_ref().filter(|(_, v)| {
            v["intent"] == json!(intent_id)
                && v["cleanup_confirmed"] == true
                && v["status"] == "incomplete_no_rerun"
        });
        if !owner.gone()? && terminal.is_none() {
            return Err(deny(
                "promotion owner still alive; recovery cannot race finalization",
            ));
        }
        let reload = read(&self.connection, &lease.contract, "reload")?;
        // Serialize cleanup proof with the recovery fence. A missing child identity
        // on this boot is NOT evidence of absence; a reboot is native evidence.
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sql)?;
        check_lease(&tx, &self.store, lease, &files::moment()?)?;
        if read(&tx, &lease.contract, "intent")?.map(|v| v.0) != Some(intent_id.clone()) {
            return Err(deny("promotion recovery intent changed"));
        }
        child_inventory::reconcile(&tx, &lease.contract, &intent_id, &intent)?;
        tx.commit().map_err(sql)?;
        let reservation = self.recover_reservation(lease, &id)?;
        self.settle(
            lease,
            &reservation,
            Settlement {
                operation: hash(&(&id, "promotion_recovery"))?,
                outcome: OutcomeKind::Incomplete,
                usage: Usage::default(),
                resources_released: true,
                evidence: reload.map(|v| v.0).unwrap_or(hash(&intent)?),
            },
        )
    }
    /// Legacy ungranted entry point: never infers execution consent from evidence.
    /// Use native qualification, explicit approval, then promote_fixture_with_grant.
    /// A report, environment flag or fixture-only decision is not a qualification.
    #[allow(clippy::too_many_arguments)]
    pub fn promote_fixture(
        &mut self,
        _lease: &Lease,
        _store: &ArtifactStore,
        _edit: &CompiledPlan,
        _smoke: &Validated<SmokeSuite>,
        _runtime: &QualifiedTorch,
        _operation: ContentDigest,
        _expected_head: Option<ContentDigest>,
        _cancel: &CancellationToken,
    ) -> Result<Acceptance> {
        Err(failure(ErrorCode::UnsupportedCapability,
            "T039 requires T038 native qualification and an explicit operator execution grant; independent verification is not implied; use promote_fixture_with_grant"))
    }
    /// Test-only OS-registry seam for low-level adversarial regressions.
    /// Both test and production orchestration enter with a live issued grant.
    /// Requires a dedicated hard-memory-limited controller plus bounded child.
    /// One-shot native orchestration. Interrupted intents cannot be retried as new
    /// evaluations. Only an already committed identical operation is idempotent.
    /// The replay does no holdout work: it reproduces the exact T038 candidate,
    /// freezes approved smoke inputs/seed expectations BEFORE re-export, and
    /// requires that export to equal the original native dispatch byte for byte.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn promote_fixture_after_review(
        &mut self,
        lease: &Lease,
        store: &ArtifactStore,
        edit: &CompiledPlan,
        smoke: &Validated<SmokeSuite>,
        runtime: &QualifiedTorch,
        operation: ContentDigest,
        expected_head: Option<ContentDigest>,
        cancel: &CancellationToken,
    ) -> Result<Acceptance> {
        operator::test_registry(
            &self
                .files
                .sqlite_path()
                .parent()
                .ok_or_else(|| deny("missing test ledger directory"))?
                .join("operator"),
        );
        let grant = if let Some((_, intent)) = read(&self.connection, &lease.contract, "intent")? {
            // Test replay still goes through live registry verification; archived
            // receipt data alone is never sufficient, including idempotence.
            PromotionExecutionGrant::test_receipt(intent["request"]["release"].clone())
        } else {
            let qualification =
                self.qualify_fixture_promotion(lease, store, PromotionScope::OperationalFixture)?;
            let request = self.fixture_promotion_approval_request(
                lease,
                &qualification,
                edit,
                smoke,
                runtime,
                operation.clone(),
                expected_head.clone(),
                cancel,
            )?;
            let id = operator::issue(&request, operator::test_expiration())?;
            self.approve_fixture_promotion(
                lease,
                &qualification,
                id,
                edit,
                smoke,
                runtime,
                operation.clone(),
                expected_head.clone(),
                cancel,
            )?
        };
        self.promote_fixture_with_grant(
            lease,
            store,
            edit,
            smoke,
            runtime,
            operation,
            expected_head,
            cancel,
            &grant,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn promote_fixture_bound(
        &mut self,
        lease: &Lease,
        store: &ArtifactStore,
        edit: &CompiledPlan,
        smoke: &Validated<SmokeSuite>,
        runtime: &QualifiedTorch,
        operation: ContentDigest,
        expected_head: Option<ContentDigest>,
        cancel: &CancellationToken,
        release: Option<&Value>,
    ) -> Result<Acceptance> {
        // Pin admission through its real commit; release during native work so
        // the operator CAN revoke an in-flight attempt. Pin final commit anew.
        let receipt =
            release.ok_or_else(|| deny("promotion orchestration requires root-issued approval"))?;
        let admission_operator_guard = release::live_operator(receipt)?;
        self.files.verify(self.limits.max_database_bytes)?;
        let request = json!({"operation":operation,"expected_head":expected_head,"edit":edit.digest(),
            "smoke":smoke.digest()?,"runtime":runtime.identity(),"protocol":"native_fixture_promotion_v1","release":receipt});
        {
            let tx = self.connection.unchecked_transaction().map_err(sql)?;
            fence(&tx, &self.store, lease, lease.deadline, cancel)?;
            if let Some((_, old)) = read(&tx, &lease.contract, "intent")? {
                if old["request"] != request {
                    return Err(failure(
                        ErrorCode::IdempotencyConflict,
                        "promotion input changed",
                    ));
                }
                let c = load_contract(&tx, &lease.contract)?;
                if let Some(a) = accepted(&tx, &c.study_family)? {
                    if a.body["contract"] == json!(lease.contract)
                        && a.body["operation"] == json!(operation)
                    {
                        drop(tx);
                        return self
                            .accepted_fixture_head(&c.study_family)?
                            .ok_or_else(|| deny("committed acceptance disappeared"));
                    }
                }
                return Err(deny(
                    "interrupted or failed promotion requires explicit recovery; no rerun",
                ));
            }
        }
        let before = final_chain(&self.connection, &lease.contract)?.0;
        let decision = final_fixture::read_any_final_decision_in(
            &self.connection,
            &self.store,
            &lease.contract,
        )?
        .ok_or_else(|| deny("missing trusted T038 decision"))?;
        let (chain, dispatch) = final_chain(&self.connection, &lease.contract)?;
        let c = self.contract(&lease.contract)?;
        let predecessor = self.accepted_fixture_head(&c.study_family)?;
        if predecessor.as_ref().map(|a| a.digest.clone()) != expected_head
            || predecessor
                .as_ref()
                .is_some_and(|a| a.body["model"] != json!(c.best_known))
        {
            return Err(deny(
                "predecessor is not the semantically verified family head",
            ));
        }
        let panel = self
            .frozen_final_panel(&lease.contract)?
            .ok_or_else(|| deny("missing frozen panel"))?
            .1;
        let valid_adapter = (decision.report().scope == FINAL_SCOPE
            && smoke.adapter == "nf.linear_fixture.v1")
            || (decision.report().scope == ARITHMETIC_FINAL_SCOPE
                && smoke.adapter == "nf.arithmetic_relu.v1");
        if chain != before
            || !decision.report().accepted
            || decision.report().status != acceptance::Status::Passed
            || decision.report().promotion_authorized
            || !valid_final_scope(&decision.report().scope)
            || c.portfolio.independent.scope != ClaimScope::FixtureOnly
            || panel.finalists.len() != 1
            || dispatch["candidate"] != json!(panel.finalists[0])
            || dispatch["edit"]["plan"] != json!(edit.digest())
            || dispatch["candidate_export"]["smoke_suite"] != json!(smoke.digest()?)
            || !valid_adapter
            || smoke.probes.is_empty()
            || smoke.probes.len() > 16
        {
            return Err(deny(
                "unsupported or mismatched trusted final decision/candidate/probes",
            ));
        }
        if decision.report().scope == ARITHMETIC_FINAL_SCOPE {
            self.qualify_arithmetic_final(&lease.contract, store)?;
        }
        let target = panel.finalists[0].clone();
        let cpu = cpu_ms();
        let start = files::moment()?;
        // Per input: transaction smoke, independent freeze, two export smoke
        // passes, two reload native preflights, and one fresh-child forward.
        let resources = Resources {
            cpu_ms: CPU_MS,
            evaluations: smoke.probes.len() as u64 * 7,
            storage_bytes: STORAGE,
            host_bytes: c.limits.host_memory_bytes,
            workers: 1,
            ..Default::default()
        };
        // Reserve conservatively for child address space plus whole controller RSS
        // and bounded native staging. Unknown overhead stays charged at ceilings.
        let mut limits = WorkerLimits::default();
        let controller_hard = memory_envelope(limits.address_space_bytes, resources.host_bytes)?;
        if rss()?
            .saturating_add(limits.address_space_bytes)
            .saturating_add(134217728)
            > resources.host_bytes
        {
            return Err(deny(
                "insufficient controller+child+staging memory envelope",
            ));
        }
        // Limit physical DB growth for all intent/evidence/settlement/commit writes.
        let old_cap: u64 = self
            .connection
            .query_row("PRAGMA max_page_count", [], |r| r.get(0))
            .map_err(sql)?;
        let pages: u64 = self
            .connection
            .query_row("PRAGMA page_count", [], |r| r.get(0))
            .map_err(sql)?;
        let page: u64 = self
            .connection
            .query_row("PRAGMA page_size", [], |r| r.get(0))
            .map_err(sql)?;
        self.connection
            .pragma_update(
                None,
                "max_page_count",
                old_cap.min(pages + 1_048_576 / page),
            )
            .map_err(sql)?;
        let mut measured_child_cpu = 0;
        let mut cleanup_confirmed = true;
        let mut child_entered = false;
        let mut reserved = None;
        // Never rely on SQLite's intent row for anti-replay: its owner can
        // restore a pre-admission image into the SAME database inode. Burn the
        // approval in the operator authority first. A crash/failed DB commit
        // may lose availability, but cannot mint a second execution.
        drop(admission_operator_guard);
        let admission_operator_guard = release::consume_operator(receipt)?;
        let result = (|| {
            let owner = process::Owner::current()?;
            let store_id = self.store.clone();
            let mut deadline = lease.deadline;
            let mut intent = None;
            let reservation=self.reserve_with(lease,Request {schema_version:1,operation:hash(&(&operation,"promotion_cost"))?,attempt:0,
                parent:None,kind:WorkKind::Reload,resources:resources.clone(),wall_ms:30_000,external:None}, |tx,reservation,allocation_deadline| {
            deadline=allocation_deadline.min(lease.deadline);
            fence(tx,&store_id,lease,deadline,cancel)?;
            release::validate_in(tx,&store_id,lease,receipt)?;
            if state(&tx,&lease.contract)?.stage!=Stage::Verify || head(&tx,&c.study_family)?!=expected_head
                || final_chain(&tx,&lease.contract)?.0!=chain || !budgets::contract_clean_except(&tx,&lease.contract,Some(&reservation))? {
                return Err(deny("stale predecessor, native final chain or workflow"));
            }
            if let Some(a)=accepted(&tx,&c.study_family)? {
                if a.body["model"]!=json!(c.best_known) { return Err(deny("best-known is not accepted predecessor")); }
            }
            intent=Some(record(tx,&lease.contract,"intent",&json!({"request":request,"store":store_id,
                "contract":lease.contract,"chain":chain,"decision":hash(decision.report())?,"candidate":target,
                "native_export":dispatch["candidate_export"],"reservation":reservation.id(),"resources":resources,
                "hard_controller_address_space_bytes":controller_hard,
                "native_process":owner,"boot":start.boot,"deadline":deadline,"epoch":lease.epoch,
                "execution_owner":lease.owner,"lease_deadline":lease.deadline}))?);
            Ok(()) })?;
            let intent =
                intent.ok_or_else(|| deny("existing allocation cannot restart promotion"))?;
            reserved = Some(reservation.clone());
            drop(admission_operator_guard); // reservation/intent commit finished
            let work = (|| {
                let exp_limits = export_limits_for(&smoke.adapter);
                let exec_limits = execution_limits_for(&smoke.adapter);
                let root: ContentDigest =
                    serde_json::from_value(dispatch["candidate_export"]["export_root"].clone())
                        .map_err(|_| deny("missing native export root"))?;
                let staged = nf_surgeon::export::restore_source(store, &root, exp_limits)?;
                if model(&staged)? != target {
                    return Err(deny("native final export differs from finalist"));
                }
                drop(staged); // NEVER used to generate expected outputs.
                let source = source(store, &c.source)?;
                let candidate = execute_cancellable(edit, &source, smoke, exec_limits, cancel)?;
                if json!(candidate.receipt()) != dispatch["edit"]
                    || json!(candidate.digest()) != dispatch["candidate_export"]["candidate"]
                {
                    return Err(deny(
                        "replayed native transaction differs from accepted candidate",
                    ));
                }
                let reference = candidate.materialize_source(exp_limits.materialization)?;
                if model(&reference)? != target {
                    return Err(deny("replayed model differs from frozen finalist"));
                }
                let inputs = smoke
                    .probes
                    .iter()
                    .map(|p| p.input.clone())
                    .collect::<Vec<_>>();
                let seeds = vec![c.seeds[0]; inputs.len()];
                let probes = FrozenReloadProbes::freeze(
                    &reference,
                    target.clone(),
                    inputs.clone(),
                    seeds.clone(),
                )?;
                let tx = self
                    .connection
                    .transaction_with_behavior(TransactionBehavior::Immediate)
                    .map_err(sql)?;
                fence(&tx, &self.store, lease, deadline, cancel)?;
                let mut archive = json!({"manifest":&**reference.manifest(),"checkpoint":reference.checkpoint_bytes(),"inputs":inputs,"seeds":seeds,"smoke_suite":&**smoke});
                if smoke.adapter == "nf.arithmetic_relu.v1" {
                    let assets: BTreeMap<String, Vec<u8>> = reference
                        .assets()
                        .iter()
                        .map(|(k, v)| (k.clone(), v.to_vec()))
                        .collect();
                    archive["assets"] = json!(assets);
                }
                record(
                    &tx,
                    &lease.contract,
                    "probes",
                    &json!({"intent":intent,"probes":probes.id(),"origin":"native_replay_before_reexport",
                    "transaction":candidate.receipt(),"smoke":smoke.digest()?,"seed":c.seeds[0],"model":target,
                    "archive":archive}),
                )?;
                ensure_live(lease)?;
                tx.commit().map_err(sql)?;
                let export = export_candidate(store, &candidate, smoke, exp_limits)?;
                if json!(export) != dispatch["candidate_export"] {
                    return Err(deny("re-export differs from native final export"));
                }
                drop(reference);
                drop(candidate);
                drop(source);
                // Persist launch intent before entering the worker. A crash without
                // reaped evidence remains held; archived JSON never mints a witness.
                let launch_operator_guard = release::live_operator(receipt)?;
                let tx = self
                    .connection
                    .transaction_with_behavior(TransactionBehavior::Immediate)
                    .map_err(sql)?;
                fence(&tx, &self.store, lease, deadline, cancel)?;
                release::validate_in(&tx, &self.store, lease, receipt)?;
                record(
                    &tx,
                    &lease.contract,
                    "child_intent",
                    &json!({"intent":intent,"status":"cleanup_unknown_until_reload_report"}),
                )?;
                tx.commit().map_err(sql)?;
                drop(launch_operator_guard);
                // The registration writer must finish before the fail-closed
                // read monitor starts: our own commit must not cancel our child
                // as an unreadable fence. Registration itself checks stop/epoch/
                // deadline; the guard then checks again BEFORE gate release.
                let launch_cancel = CancellationToken::default();
                let mut guard = None;
                limits.wall_ms = limits
                    .wall_ms
                    .min(deadline.saturating_sub(files::moment()?.ms).max(1) as u64);
                cancel.check()?;
                child_entered = true;
                let attempt = runtime.verify_reload_registered(
                    store,
                    &export.export_root,
                    &probes,
                    limits,
                    &launch_cancel,
                    &mut |fd| {
                        cancel.check()?;
                        let _registration_operator_guard = release::live_operator(receipt)?;
                        self.register_fixture_reload_child(lease, fd)?;
                        guard = Some(supervision_guard::Guard::start(
                            self.files.sqlite_path(),
                            self.store.clone(),
                            lease.clone(),
                            deadline,
                            cancel.clone(),
                            launch_cancel.clone(),
                        )?);
                        Ok(())
                    },
                );
                let supervised = guard.map_or_else(
                    || Err(deny("reload never reached registered supervision")),
                    |g| g.finish(),
                );
                cleanup_confirmed = attempt.report().cleanup_confirmed;
                measured_child_cpu = attempt.report().cpu_ms.unwrap_or(0);
                // Evidence-only append is allowed after lease loss. No authority,
                // budget release, or journal transition is written by this append.
                let tx = self
                    .connection
                    .transaction_with_behavior(TransactionBehavior::Immediate)
                    .map_err(sql)?;
                if read(&tx, &lease.contract, "intent")?.map(|v| v.0) != Some(intent.clone()) {
                    return Err(deny("promotion intent changed"));
                }
                let evidence = record(&tx, &lease.contract, "reload", attempt.report())?;
                record(
                    &tx,
                    &lease.contract,
                    "supervision",
                    &json!({"intent":intent,"result":supervised.as_ref().err().map(ToString::to_string),"cleanup_confirmed":cleanup_confirmed}),
                )?;
                tx.commit().map_err(sql)?;
                supervised?;
                let witness = attempt.into_witness().ok_or_else(|| {
                    deny("fresh-process reload did not verify; evidence retained")
                })?;
                if !witness.report().cleanup_confirmed
                    || witness.report().child_started != Some(true)
                    || witness.report().model != target
                    || witness.report().probes != *probes.id()
                    || witness.report().runtime != *runtime.identity()
                    || cpu_ms()
                        .saturating_sub(cpu)
                        .saturating_add(witness.report().cpu_ms.unwrap_or(CPU_MS + 1))
                        > CPU_MS
                    || rss()?.saturating_add(witness.report().peak_rss_bytes.unwrap_or(u64::MAX))
                        > resources.host_bytes
                {
                    return Err(deny("reload accounting, cleanup or identity incomplete"));
                }
                Ok((evidence, witness))
            })();
            // Failed attempts stay reserved through all local finalization and
            // explicit recovery. A reaped child alone does not finish promotion.
            let (evidence, live_witness) = work?;
            let store_id = self.store.clone();
            let max_events = self.limits.max_events;
            let mut acceptance = None;
            // Assigned inside the publishing writer, dropped only AFTER the
            // settlement method has actually committed/rolled back SQLite.
            let mut publication_operator_guard = None;
            self.finish_budget_with(lease,&reservation,Settlement {operation:hash(&(&operation,"promotion_settlement"))?,
                outcome:OutcomeKind::Completed,usage:Usage::default(),resources_released:true,evidence:evidence.clone()},false,|tx| {
            publication_operator_guard=Some(release::live_operator(receipt)?);
            fence(tx,&store_id,lease,deadline,cancel)?;
            release::validate_in(tx,&store_id,lease,receipt)?;
            if state(&tx,&lease.contract)?.stage!=Stage::Verify || final_chain(&tx,&lease.contract)?.0!=chain
                || head(&tx,&c.study_family)?!=expected_head || !budgets::contract_clean(&tx,&lease.contract)? {
                return Err(deny("promotion final compare-and-set fence failed"));
            }
            let costs=record(&tx,&lease.contract,"costs",&json!({"reservation":reservation.id(),
                "charged_ceiling":resources,"parent_process_cpu_ms_sample":cpu_ms().saturating_sub(cpu),
                "child_cpu_ms":live_witness.report().cpu_ms,"child_peak_rss_bytes":live_witness.report().peak_rss_bytes,
                "parent_peak_rss_bytes":rss()?,"elapsed_wall_ms_sample":files::moment()?.ms.saturating_sub(start.ms),
                "scope":"full_CPU_storage_evaluations_ceiling_charged;unknown_overhead_not_refunded;SQLite_growth_capped_1MiB;tail_checked_before_commit"}))?;
            let body=json!({"schema_version":1,"scope":"trusted_native_linear_fixture_operational_only_not_generalization_or_strict_fsync",
                "family":c.study_family,"contract":lease.contract,"operation":operation,"predecessor":expected_head,
                "intent":intent,"reload":evidence,"costs":costs,"model":target,"export_root":dispatch["candidate_export"]["export_root"],
                "epoch":lease.epoch,"store":store_id,"release":receipt});
            let id=hash(&body)?;
            publish(tx,lease,&c.study_family,&expected_head,&id,&body,max_events)?;
            // Durable stop and publication serialize on the same IMMEDIATE writer.
            // Token cancellation linearizes at this final check, not retroactively.
            fence(tx,&store_id,lease,deadline,cancel)?;
            if cpu_ms().saturating_sub(cpu).saturating_add(live_witness.report().cpu_ms.unwrap_or(CPU_MS+1))>CPU_MS {
                return Err(deny("promotion parent+child CPU envelope exceeded"));
            }
            ensure_live(lease)?;
            // Last logical authorization/expiration fence; operational only, not
            // a hard real-time guarantee through SQLite's unbounded fsync tail.
            release::validate_in(tx,&store_id,lease,receipt)?;
            acceptance=Some(Acceptance {digest:id,body}); Ok(())
            })?;
            drop(publication_operator_guard);
            acceptance.ok_or_else(|| deny("settlement replay cannot mint acceptance"))
        })();
        // Expanding a connection-local cap is housekeeping, never a second
        // authoritative commit. A failed reset leaves a safer restrictive cap;
        // it must not report a failed promotion after acceptance committed.
        let _ = self
            .connection
            .pragma_update(None, "max_page_count", old_cap);
        let measured = cpu_ms()
            .saturating_sub(cpu)
            .saturating_add(measured_child_cpu);
        if result.is_err() && measured > CPU_MS && cleanup_confirmed {
            let Some(reservation) = reserved.as_ref() else {
                return result;
            };
            // Include failed final-transaction tail work in a durable breach.
            // A lost lease may prevent this write; its reservation remains held.
            let evidence = read(&self.connection, &lease.contract, "reload")?
                .map(|(id, _)| id)
                .unwrap_or(operation.clone());
            if let Err(error) = self.settle(
                lease,
                reservation,
                Settlement {
                    operation: hash(&(&operation, "promotion_tail_overrun"))?,
                    outcome: OutcomeKind::Incomplete,
                    usage: Usage {
                        cpu_ms: Some(measured),
                        ..Default::default()
                    },
                    resources_released: true,
                    evidence,
                },
            ) {
                if error.code == ErrorCode::BudgetBreach {
                    return Err(error);
                }
            }
        }
        if result.is_err() && cleanup_confirmed && reserved.is_some() {
            // Durable terminal marker permits recovery in a still-live controller
            // only AFTER all local work and accounting attempts have ended.
            if let Some((intent, _)) = read(&self.connection, &lease.contract, "intent")? {
                let tx = self
                    .connection
                    .transaction_with_behavior(TransactionBehavior::Immediate)
                    .map_err(sql)?;
                if read(&tx, &lease.contract, "attempt_end")?.is_none() {
                    record(
                        &tx,
                        &lease.contract,
                        "attempt_end",
                        &json!({"intent":intent,"cleanup_confirmed":true,"child_entered":child_entered,"status":"incomplete_no_rerun"}),
                    )?;
                }
                tx.commit().map_err(sql)?;
            }
        }
        result
    }
}
fn publish(
    tx: &Transaction<'_>,
    lease: &Lease,
    family: &ContentDigest,
    expected: &Option<ContentDigest>,
    id: &ContentDigest,
    value: &Value,
    max_events: u32,
) -> Result<()> {
    let current = state(tx, &lease.contract)?;
    if current.stage != Stage::Verify || head(tx, family)? != *expected {
        return Err(deny("publication compare-and-set failed"));
    }
    let count: u64 = tx
        .query_row("SELECT count(*) FROM events", [], |r| r.get(0))
        .map_err(sql)?;
    if count >= max_events as u64 {
        return Err(failure(ErrorCode::QuotaExceeded, "event bound exceeded"));
    }
    tx.execute(
        "INSERT INTO accepted_records VALUES(?1,?2,?3,?4)",
        params![
            id.as_str(),
            lease.contract.as_str(),
            family.as_str(),
            canonical(value)?
        ],
    )
    .map_err(sql)?;
    tx.execute(
        "INSERT INTO verified_frontier VALUES(?1,?2)",
        params![id.as_str(), family.as_str()],
    )
    .map_err(sql)?;
    let changed = if let Some(previous) = expected {
        tx.execute(
            "UPDATE accepted_heads SET digest=?1 WHERE family=?2 AND digest=?3",
            params![id.as_str(), family.as_str(), previous.as_str()],
        )
        .map_err(sql)?
    } else {
        tx.execute(
            "INSERT INTO accepted_heads VALUES(?1,?2)",
            params![family.as_str(), id.as_str()],
        )
        .map_err(sql)?
    };
    if changed != 1 {
        return Err(deny("accepted head compare-and-set lost"));
    }
    let now = files::moment()?;
    let body = EventBody {
        schema_version: 1,
        contract: lease.contract.clone(),
        sequence: current.sequence + 1,
        previous: current.head,
        epoch: lease.epoch,
        owner: lease.owner.clone(),
        boot: now.boot,
        monotonic_ms: now.ms,
        wall_ms: now.wall_ms,
        transition: Transition {
            operation: serde_json::from_value(value["operation"].clone())
                .map_err(|_| deny("missing operation"))?,
            expected: Stage::Verify,
            next: Stage::Completed,
            protocol: ContentDigest::of_bytes(b"nf.fixture.promotion.v1"),
            evidence: id.clone(),
            message: "atomic native fixture acceptance; no deployment/generalization claim".into(),
        },
    };
    let bytes = canonical(&body)?;
    let event = ContentDigest::of_bytes(&bytes);
    tx.execute(
        "INSERT INTO events VALUES(?1,?2,?3,?4,?5)",
        params![
            lease.contract.as_str(),
            body.sequence,
            body.transition.operation.as_str(),
            bytes,
            event.as_str()
        ],
    )
    .map_err(sql)?;
    tx.execute(
        "UPDATE runs SET stage='completed',sequence=?1,head=?2 WHERE contract=?3",
        params![body.sequence, event.as_str(), lease.contract.as_str()],
    )
    .map_err(sql)?;
    Ok(())
}
fn accepted(c: &Connection, family: &ContentDigest) -> Result<Option<Acceptance>> {
    let Some(id) = head(c, family)? else {
        history::check_tip(c, family, None)?;
        return Ok(None);
    };
    history::check_tip(c, family, Some(&id))?;
    let (contract, stored_family, b): (String, String, Vec<u8>) = c
        .query_row(
            "SELECT contract,family,body FROM accepted_records WHERE digest=?1",
            [id.as_str()],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .map_err(sql)?;
    let (_, body) = decode(&b, id.as_str())?;
    let contract: ContentDigest = contract.parse()?;
    let event = find_event(
        c,
        &contract,
        &serde_json::from_value(body["operation"].clone())
            .map_err(|_| deny("invalid accepted operation"))?,
    )?
    .ok_or_else(|| deny("missing acceptance event"))?;
    let frontier: bool = c
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM verified_frontier WHERE digest=?1 AND family=?2)",
            params![id.as_str(), family.as_str()],
            |r| r.get(0),
        )
        .map_err(sql)?;
    let intent = read(c, &contract, "intent")?.ok_or_else(|| deny("missing promotion intent"))?;
    let reload = read(c, &contract, "reload")?.ok_or_else(|| deny("missing reload evidence"))?;
    let probes =
        read(c, &contract, "probes")?.ok_or_else(|| deny("missing native probe provenance"))?;
    let costs =
        read(c, &contract, "costs")?.ok_or_else(|| deny("missing promotion cost record"))?;
    let reservation: ContentDigest = serde_json::from_value(intent.1["reservation"].clone())
        .map_err(|_| deny("missing promotion reservation"))?;
    let store: String = c
        .query_row("SELECT value FROM meta WHERE key='store'", [], |r| r.get(0))
        .map_err(sql)?;
    state(c, &contract)?;
    if stored_family != family.as_str()
        || body["family"] != json!(family)
        || body["contract"] != json!(contract)
        || event.body.transition.evidence != id
        || event.body.transition.next != Stage::Completed
        || event.body.transition.expected != Stage::Verify
        || event.body.transition.protocol != ContentDigest::of_bytes(b"nf.fixture.promotion.v1")
        || state(c, &contract)?.stage != Stage::Completed
        || !frontier
        || body["intent"] != json!(intent.0)
        || body["reload"] != json!(reload.0)
        || body["costs"] != json!(costs.0)
        || body["store"] != store
        || intent.1["store"] != store
        || body["epoch"] != json!(event.body.epoch)
        || body["epoch"] != intent.1["epoch"]
        || body["operation"] != intent.1["request"]["operation"]
        || body["predecessor"] != intent.1["request"]["expected_head"]
        || body["model"] != intent.1["candidate"]
        || probes.1["intent"] != json!(intent.0)
        || probes.1["probes"] != reload.1["probes"]
        || probes.1["model"] != body["model"]
        || reload.1["runtime"] != intent.1["request"]["runtime"]
        || costs.1["reservation"] != json!(reservation)
        || costs.1["charged_ceiling"] != intent.1["resources"]
        || !budgets::native_released(c, &reservation)?
        || budgets::native_kind(c, &reservation)? != WorkKind::Reload
        || reload.1["child_started"] != true
        || reload.1["cpu_ms"].as_u64().is_none()
        || reload.1["peak_rss_bytes"].as_u64().is_none_or(|n| n == 0)
        || body["model"] != reload.1["model"]
        || body["export_root"] != reload.1["export_root"]
        || body["export_root"] != intent.1["native_export"]["export_root"]
        || intent.1["chain"] != json!(final_chain(c, &contract)?.0)
        || reload.1["status"] != "verified_fresh_process_reload"
        || reload.1["cleanup_confirmed"] != true
    {
        return Err(deny("accepted pointer evidence chain differs"));
    }
    history::validate(c, &contract, &intent.1, &probes.1, &reload.1, &body)?;
    Ok(Some(Acceptance { digest: id, body }))
}
#[cfg(test)]
mod tests;
