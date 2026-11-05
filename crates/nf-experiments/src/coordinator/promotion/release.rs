//! Host-only release boundary. Serialized receipts are audit data, never grants.
//! Qualification consumes T038's opaque native operational witness, not a report.
use super::*;

/// Publication boundary requested by the operator. Operational means the last
/// transactional fence, NOT a deadline/CPU guarantee through SQLite fsync.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PromotionScope {
    OperationalFixture,
    StrictFsync,
}

/// Opaque live native qualification; neither Deserialize nor a public constructor.
/// Operational native verification is not independent engineering sign-off.
/// ```compile_fail
/// use nf_experiments::coordinator::promotion::PromotionQualification;
/// let _: PromotionQualification = serde_json::from_str("{}").unwrap();
/// ```
#[derive(Debug)]
pub struct PromotionQualification {
    store: String,
    contract: ContentDigest,
    chain: ContentDigest,
    decision: ContentDigest,
    verifier: ContentDigest,
    native: final_fixture::NativeFixtureQualification,
    scope: PromotionScope,
}

/// Explicit trusted-controller execution approval for ONE exact invocation.
/// Not deserializable: archived approval JSON cannot authorize another launch.
/// ```compile_fail
/// use nf_experiments::coordinator::promotion::PromotionExecutionGrant;
/// let _: PromotionExecutionGrant = serde_json::from_str("{}").unwrap();
/// ```
#[derive(Debug)]
pub struct PromotionExecutionGrant {
    receipt: Value,
}
#[cfg(test)]
impl PromotionExecutionGrant {
    pub(super) fn test_receipt(receipt: Value) -> Self {
        Self { receipt }
    }
}

fn verifier() -> ContentDigest {
    ContentDigest::of_bytes(
        &[
            include_bytes!("../promotion.rs").as_slice(),
            include_bytes!("release.rs").as_slice(),
            include_bytes!("operator.rs").as_slice(),
            include_bytes!("history.rs").as_slice(),
            include_bytes!("../final_fixture.rs").as_slice(),
        ]
        .concat(),
    )
}
fn scope_supported(scope: PromotionScope) -> Result<()> {
    if scope != PromotionScope::OperationalFixture {
        return Err(failure(ErrorCode::UnsupportedCapability,
            "strict-fsync promotion requires qualified hard wall/CPU enforcement through durable commit; operational scope is not a fallback"));
    }
    Ok(())
}
fn ledger_file(c: &Connection) -> Result<Value> {
    use std::os::unix::fs::MetadataExt;
    let path = std::path::Path::new(
        c.path()
            .ok_or_else(|| deny("promotion requires a physical ledger"))?,
    );
    let parent = path
        .parent()
        .ok_or_else(|| deny("missing ledger directory"))?;
    let file =
        std::fs::symlink_metadata(path).map_err(|_| deny("ledger file identity unavailable"))?;
    let directory = std::fs::symlink_metadata(parent)
        .map_err(|_| deny("ledger directory identity unavailable"))?;
    if !file.is_file() || file.nlink() != 1 || !directory.is_dir() {
        return Err(deny("promotion requires an unaliased physical ledger"));
    }
    Ok(
        json!({"path":path,"directory_device":directory.dev(),"directory_inode":directory.ino(),
        "file_device":file.dev(),"file_inode":file.ino()}),
    )
}
fn invocation(
    lease: &Lease,
    edit: &CompiledPlan,
    smoke: &Validated<SmokeSuite>,
    runtime: &QualifiedTorch,
    operation: &ContentDigest,
    expected: &Option<ContentDigest>,
) -> Result<Value> {
    Ok(
        json!({"contract":lease.contract,"epoch":lease.epoch,"owner":lease.owner,
        "boot":lease.boot,"lease_deadline":lease.deadline,"edit":edit.digest(),"smoke":smoke.digest()?,
        "runtime":runtime.identity(),"operation":operation,"expected_head":expected}),
    )
}
impl Ledger {
    /// Resolve from T038's opaque native witness and real CAS artifacts. This
    /// read-only probe never reserves resources or opens another reload child.
    /// Operational scope does not claim independent engineering release review.
    pub fn qualify_fixture_promotion(
        &self,
        lease: &Lease,
        store: &ArtifactStore,
        scope: PromotionScope,
    ) -> Result<PromotionQualification> {
        scope_supported(scope)?;
        self.files.verify(self.limits.max_database_bytes)?;
        let tx = self.connection.unchecked_transaction().map_err(sql)?;
        fence(
            &tx,
            &self.store,
            lease,
            lease.deadline,
            &CancellationToken::default(),
        )?;
        let contract = load_contract(&tx, &lease.contract)?;
        if state(&tx, &lease.contract)?.stage != Stage::Verify
            || contract.portfolio.independent.scope != ClaimScope::FixtureOnly
        {
            return Err(deny(
                "promotion qualification requires live Verify and fixture-only scope",
            ));
        }
        drop(tx);
        let decision = final_fixture::read_any_final_decision_in(
            &self.connection,
            &self.store,
            &lease.contract,
        )?
        .ok_or_else(|| deny("missing native T038 decision"))?;
        if !decision.report().accepted
            || decision.report().status != acceptance::Status::Passed
            || !valid_final_scope(&decision.report().scope)
        {
            return Err(deny("native T038 fixture decision is not passing"));
        }
        let chain = final_chain(&self.connection, &lease.contract)?.0;
        let native = self.qualify_fixture_final(&lease.contract, store)?;
        if native.contract() != &lease.contract
            || native.decision() != decision.id()
            || native.timing_scope() != &final_fixture::FixtureTimingScope::OperationalCancellation
            || final_chain(&self.connection, &lease.contract)?.0 != chain
        {
            return Err(deny("native operational qualification binding changed"));
        }
        Ok(PromotionQualification {
            store: self.store.clone(),
            contract: lease.contract.clone(),
            chain,
            decision: hash(decision.report())?,
            verifier: verifier(),
            native,
            scope,
        })
    }

    /// Read-only request for separate root-operator inspection and issuance via
    /// operator::issue. Returning this JSON does NOT authorize any execution.
    #[allow(clippy::too_many_arguments)]
    pub fn fixture_promotion_approval_request(
        &self,
        lease: &Lease,
        qualification: &PromotionQualification,
        edit: &CompiledPlan,
        smoke: &Validated<SmokeSuite>,
        runtime: &QualifiedTorch,
        operation: ContentDigest,
        expected_head: Option<ContentDigest>,
        cancel: &CancellationToken,
    ) -> Result<Value> {
        scope_supported(qualification.scope)?;
        self.files.verify(self.limits.max_database_bytes)?;
        let tx = self.connection.unchecked_transaction().map_err(sql)?;
        fence(&tx, &self.store, lease, lease.deadline, cancel)?;
        let contract = load_contract(&tx, &lease.contract)?;
        if qualification.store != self.store
            || qualification.contract != lease.contract
            || qualification.native.contract() != &lease.contract
            || qualification.native.decision() != &qualification.decision
            || qualification.native.timing_scope()
                != &final_fixture::FixtureTimingScope::OperationalCancellation
            || qualification.verifier != verifier()
            || qualification.chain != final_chain(&tx, &lease.contract)?.0
            || state(&tx, &lease.contract)?.stage != Stage::Verify
            || contract.portfolio.independent.scope != ClaimScope::FixtureOnly
            || head(&tx, &contract.study_family)? != expected_head
        {
            return Err(deny(
                "stale or foreign promotion qualification/scope/predecessor",
            ));
        }
        drop(tx);
        let decision = final_fixture::read_any_final_decision_in(
            &self.connection,
            &self.store,
            &lease.contract,
        )?
        .ok_or_else(|| deny("missing native T038 decision"))?;
        if hash(decision.report())? != qualification.decision
            || !decision.report().accepted
            || decision.report().status != acceptance::Status::Passed
            || !valid_final_scope(&decision.report().scope)
        {
            return Err(deny(
                "promotion qualification no longer matches native final decision",
            ));
        }
        Ok(
            json!({"schema_version":1,"store":self.store,"ledger_file":ledger_file(&self.connection)?,
            "family":contract.study_family,"scope":qualification.scope,"origin":"native_operational_fixture_not_independent_review",
            "native_qualification":qualification.native,"native_qualification_digest":qualification.native.digest()?,
            "chain":qualification.chain,"decision":qualification.decision,"verifier":qualification.verifier,
            "invocation":invocation(lease,edit,smoke,runtime,&operation,&expected_head)?}),
        )
    }

    /// Resolve a root-issued registry identifier; arbitrary digests and agent
    /// text cannot issue approval. The grant remains live/revocable, not cached.
    #[allow(clippy::too_many_arguments)]
    pub fn approve_fixture_promotion(
        &self,
        lease: &Lease,
        qualification: &PromotionQualification,
        operator_approval: ContentDigest,
        edit: &CompiledPlan,
        smoke: &Validated<SmokeSuite>,
        runtime: &QualifiedTorch,
        operation: ContentDigest,
        expected_head: Option<ContentDigest>,
        cancel: &CancellationToken,
    ) -> Result<PromotionExecutionGrant> {
        let mut receipt = self.fixture_promotion_approval_request(
            lease,
            qualification,
            edit,
            smoke,
            runtime,
            operation,
            expected_head,
            cancel,
        )?;
        let _guard = operator::verify(&operator_approval, &receipt)?;
        receipt["operator_approval"] = json!(operator_approval);
        Ok(PromotionExecutionGrant { receipt })
    }

    /// The authorized path executes the SAME registered fresh-child reload and
    /// atomic budget/pointer/frontier/event transaction as the private tests.
    #[allow(clippy::too_many_arguments)]
    pub fn promote_fixture_with_grant(
        &mut self,
        lease: &Lease,
        store: &ArtifactStore,
        edit: &CompiledPlan,
        smoke: &Validated<SmokeSuite>,
        runtime: &QualifiedTorch,
        operation: ContentDigest,
        expected_head: Option<ContentDigest>,
        cancel: &CancellationToken,
        grant: &PromotionExecutionGrant,
    ) -> Result<Acceptance> {
        self.files.verify(self.limits.max_database_bytes)?;
        let tx = self.connection.unchecked_transaction().map_err(sql)?;
        fence(&tx, &self.store, lease, lease.deadline, cancel)?;
        validate_in(&tx, &self.store, lease, &grant.receipt)?;
        if grant.receipt["invocation"]
            != invocation(lease, edit, smoke, runtime, &operation, &expected_head)?
        {
            return Err(deny(
                "promotion execution grant does not authorize these exact inputs",
            ));
        }
        drop(tx);
        self.promote_fixture_bound(
            lease,
            store,
            edit,
            smoke,
            runtime,
            operation,
            expected_head,
            cancel,
            Some(&grant.receipt),
        )
    }
}

/// Called under admission AND publication writers; a grant is not a cached fence.
pub(super) fn validate_in(
    c: &Connection,
    store: &str,
    lease: &Lease,
    receipt: &Value,
) -> Result<()> {
    let contract = load_contract(c, &lease.contract)?;
    let decision:String=c.query_row("SELECT digest FROM final_fixture_records WHERE contract=?1 AND kind IN ('decision','arithmetic_decision')",
        [lease.contract.as_str()],|r|r.get(0)).map_err(sql)?;
    if receipt["schema_version"] != 1
        || receipt["scope"] != "operational_fixture"
        || receipt["store"] != store
        || receipt["ledger_file"] != ledger_file(c)?
        || receipt["family"] != json!(contract.study_family)
        || receipt["invocation"]["contract"] != json!(lease.contract)
        || receipt["invocation"]["epoch"] != json!(lease.epoch)
        || receipt["invocation"]["owner"] != lease.owner
        || receipt["invocation"]["boot"] != lease.boot
        || receipt["invocation"]["lease_deadline"] != lease.deadline
        || receipt["decision"] != decision
        || receipt["verifier"] != json!(verifier())
        || receipt["chain"] != json!(final_chain(c, &lease.contract)?.0)
        || contract.portfolio.independent.scope != ClaimScope::FixtureOnly
    {
        return Err(deny("promotion execution grant binding/scope differs"));
    }
    validate_origin(receipt)?;
    let _guard = live_operator(receipt)?;
    audit_native(c, &lease.contract, receipt)?;
    Ok(())
}
pub(super) fn consume_operator(receipt: &Value) -> Result<operator::Guard> {
    let (id, request) = operator_request(receipt)?;
    operator::consume(&id, &request)
}
pub(super) fn live_operator(receipt: &Value) -> Result<operator::Guard> {
    let (id, request) = operator_request(receipt)?;
    operator::verify(&id, &request)
}
fn operator_request(receipt: &Value) -> Result<(ContentDigest, Value)> {
    let id: ContentDigest = serde_json::from_value(receipt["operator_approval"].clone())
        .map_err(|_| deny("missing root-issued operator approval identifier"))?;
    let mut request = receipt.clone();
    request
        .as_object_mut()
        .ok_or_else(|| deny("invalid approval request"))?
        .remove("operator_approval");
    Ok((id, request))
}
// Archived root-issued provenance only, NEVER live execution authority.
// Admission/publication additionally require unrevoked, unexpired permission.
fn validate_origin(receipt: &Value) -> Result<()> {
    if receipt["origin"] != "native_operational_fixture_not_independent_review" {
        return Err(deny("unrecognized native promotion qualification origin"));
    }
    let id: ContentDigest = serde_json::from_value(receipt["operator_approval"].clone())
        .map_err(|_| deny("missing exact operator approval"))?;
    let mut request = receipt.clone();
    request
        .as_object_mut()
        .ok_or_else(|| deny("invalid approval request"))?
        .remove("operator_approval");
    operator::audit(&id, &request)
}
fn audit_native(c: &Connection, contract: &ContentDigest, receipt: &Value) -> Result<()> {
    // Audit only: this JSON is NEVER turned back into a native/live witness.
    let native = &receipt["native_qualification"];
    let (b, d): (Vec<u8>, String) = c
        .query_row(
            "SELECT body,digest FROM final_fixture_records WHERE contract=?1 AND kind='intent'",
            [contract.as_str()],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .map_err(sql)?;
    let launch = decode(&b, &d)?.1;
    let mut statement=c.prepare("SELECT digest FROM final_fixture_records WHERE contract=?1 AND kind LIKE 'worker_receipt_%' ORDER BY kind").map_err(sql)?;
    let receipts = statement
        .query_map([contract.as_str()], |r| r.get::<_, String>(0))
        .map_err(sql)?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(sql)?;
    let is_arithmetic = launch["scope"] == json!(ARITHMETIC_FINAL_SCOPE);
    let profile_ok = if is_arithmetic {
        native["profile"] == json!(final_fixture::arithmetic_final::profile_id())
            && native["profile"] == launch["profile"]
            && native["supervisor_profile"] == json!(final_fixture::supervised_profile())
            && native["worker_image"] == json!(final_fixture::worker_image_digest())
            && !receipts.is_empty()
    } else {
        native["profile"] == json!(final_fixture::profile_id())
            && native["profile"] == launch["profile"]
            && native["supervisor_profile"] == launch["supervised_profile"]
            && native["worker_image"] == launch["worker_image"]
            && receipts.len() == 1
    };
    if native.as_object().is_none_or(|v| v.len() != 8)
        || receipt["native_qualification_digest"] != json!(hash(native)?)
        || native["contract"] != json!(contract)
        || native["decision"] != receipt["decision"]
        || !profile_ok
        || native["supervisor_receipts"] != json!(receipts)
        || native["timing_scope"] != "operational_cancellation"
        || native["physical_publication"] != "unsupported_unbounded_SQLite_fsync_and_scheduler_tail"
    {
        return Err(deny(
            "native operational qualification archive binding differs",
        ));
    }
    Ok(())
}
pub(super) fn audit(
    c: &Connection,
    contract: &ContentDigest,
    intent: &Value,
    body: &Value,
) -> Result<()> {
    let receipt = &intent["request"]["release"];
    if receipt.is_null() {
        return Err(deny("acceptance has no root-issued release approval"));
    }
    validate_origin(receipt)?;
    let (id, _) = operator_request(receipt)?;
    operator::audit_consumed(&id)?;
    audit_native(c, contract, receipt)?;
    if body["scope"]
        != "trusted_native_linear_fixture_operational_only_not_generalization_or_strict_fsync"
        || body["release"] != *receipt
        || receipt["schema_version"] != 1
        || receipt["scope"] != "operational_fixture"
        || receipt["verifier"] != json!(verifier())
        || receipt["ledger_file"] != ledger_file(c)?
        || receipt["store"] != body["store"]
        || receipt["family"] != body["family"]
        || receipt["chain"] != intent["chain"]
        || receipt["decision"] != intent["decision"]
        || receipt["invocation"]["contract"] != body["contract"]
        || receipt["invocation"]["epoch"] != body["epoch"]
        || receipt["invocation"]["boot"] != intent["boot"]
        || receipt["invocation"]["owner"] != intent["execution_owner"]
        || receipt["invocation"]["lease_deadline"] != intent["lease_deadline"]
    {
        return Err(deny("archived promotion release qualification differs"));
    }
    for key in ["operation", "expected_head", "runtime", "edit", "smoke"] {
        if receipt["invocation"][key] != intent["request"][key] {
            return Err(deny("archived execution grant inputs differ"));
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "release_tests.rs"]
mod tests;
