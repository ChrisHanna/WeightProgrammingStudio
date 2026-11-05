//! Archival semantic audit, never reconstruction of a live reload witness.
use super::*;

pub(super) fn check_tip(
    c: &Connection,
    family: &ContentDigest,
    tip: Option<&ContentDigest>,
) -> Result<()> {
    let mut statement = c
        .prepare("SELECT digest,body FROM accepted_records WHERE family=?1 ORDER BY digest")
        .map_err(sql)?;
    let rows = statement
        .query_map([family.as_str()], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?))
        })
        .map_err(sql)?;
    let mut successors = BTreeMap::new();
    let mut count = 0;
    for row in rows {
        let (id, bytes) = row.map_err(sql)?;
        let (id, body) = decode(&bytes, &id)?;
        if count >= 4096 || body["family"] != json!(family) {
            return Err(deny("invalid family acceptance lineage"));
        }
        let predecessor: Option<ContentDigest> =
            serde_json::from_value(body["predecessor"].clone())
                .map_err(|_| deny("invalid acceptance predecessor"))?;
        if successors.insert(predecessor, id).is_some() {
            return Err(deny("forked acceptance lineage"));
        }
        count += 1;
    }
    let mut cursor = None;
    let mut visited = 0;
    while let Some(next) = successors.get(&cursor) {
        cursor = Some(next.clone());
        visited += 1;
        if visited > count {
            return Err(deny("cyclic acceptance lineage"));
        }
    }
    if visited != count || cursor.as_ref() != tip {
        return Err(deny(
            "accepted family head rolled back or lineage disconnected",
        ));
    }
    Ok(())
}
pub(super) fn budget_row(
    c: &Connection,
    table: &str,
    key: &str,
    id: &ContentDigest,
) -> Result<Value> {
    let (b, d): (Vec<u8>, String) = c
        .query_row(
            &format!("SELECT body,content FROM {table} WHERE {key}=?1 ORDER BY rowid DESC LIMIT 1"),
            [id.as_str()],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .map_err(sql)?;
    Ok(decode(&b, &d)?.1)
}
pub(super) fn validate(
    c: &Connection,
    contract: &ContentDigest,
    intent: &Value,
    probes: &Value,
    reload: &Value,
    body: &Value,
) -> Result<()> {
    let contract_body = load_contract(c, contract)?;
    release::audit(c, contract, intent, body)?;
    // Reload parity alone is not supervised publication. Require the exact
    // launch admission and successful monitor result, including after reopen.
    let intent_id = hash(intent)?;
    let child = read(c, contract, "child_intent")?
        .ok_or_else(|| deny("missing promotion child admission"))?
        .1;
    let supervision = read(c, contract, "supervision")?
        .ok_or_else(|| deny("missing promotion supervision"))?
        .1;
    if child["intent"] != json!(intent_id)
        || child["status"] != "cleanup_unknown_until_reload_report"
        || supervision["intent"] != json!(intent_id)
        || supervision.get("result") != Some(&Value::Null)
        || supervision["cleanup_confirmed"] != true
    {
        return Err(deny("promotion supervision binding or success differs"));
    }
    if child_registration::registered_gone(c, contract, &intent_id, intent)? != Some(true) {
        return Err(deny("accepted promotion requires registered child cleanup"));
    }
    budgets::snapshot_in(c, contract)?; // Typed accounting replay, not just a release bit.
    let reservation: ContentDigest = serde_json::from_value(intent["reservation"].clone())
        .map_err(|_| deny("invalid reservation"))?;
    let request = budget_row(c, "budget_requests", "id", &reservation)?;
    let outcome = budget_row(c, "budget_outcomes", "request", &reservation)?;
    let costs = read(c, contract, "costs")?
        .ok_or_else(|| deny("missing promotion costs"))?
        .1;
    let limits: WorkerLimits = serde_json::from_value(reload["limits"].clone())
        .map_err(|_| deny("invalid reload limits"))?;
    let defaults = WorkerLimits::default();
    if intent["hard_controller_address_space_bytes"]
        .as_u64()
        .is_none_or(|hard| {
            hard.checked_add(limits.address_space_bytes)
                .is_none_or(|sum| sum > contract_body.limits.host_memory_bytes)
        })
        || limits.cpu_seconds != defaults.cpu_seconds
        || limits.address_space_bytes != defaults.address_space_bytes
        || limits.output_bytes != defaults.output_bytes
        || !(100..=defaults.wall_ms).contains(&limits.wall_ms)
        || reload["cpu_ms"]
            .as_u64()
            .is_none_or(|cpu| cpu > limits.cpu_seconds * 1000)
        || reload["child_wall_ms"]
            .as_u64()
            .is_none_or(|wall| wall > limits.wall_ms)
        || costs["parent_process_cpu_ms_sample"]
            .as_u64()
            .is_none_or(|parent| {
                parent.saturating_add(reload["cpu_ms"].as_u64().unwrap_or(u64::MAX)) > CPU_MS
            })
        || costs["child_cpu_ms"] != reload["cpu_ms"]
        || costs["child_peak_rss_bytes"] != reload["peak_rss_bytes"]
        || costs["parent_peak_rss_bytes"]
            .as_u64()
            .is_none_or(|parent| {
                parent.saturating_add(reload["peak_rss_bytes"].as_u64().unwrap_or(u64::MAX))
                    > contract_body.limits.host_memory_bytes
            })
    {
        return Err(deny("archived reload resource evidence exceeds allocation"));
    }
    let (b,d):(Vec<u8>,String)=c.query_row("SELECT body,digest FROM final_fixture_records WHERE contract=?1 AND kind IN ('dispatch','arithmetic_dispatch')",[contract.as_str()],|r|Ok((r.get(0)?,r.get(1)?))).map_err(sql)?;
    let dispatch = decode(&b, &d)?.1;
    if probes["transaction"] != dispatch["edit"]
        || intent["native_export"] != dispatch["candidate_export"]
        || probes["smoke"] != intent["native_export"]["smoke_suite"]
        || body["model"] != dispatch["candidate"]
    {
        return Err(deny("promotion replay differs from final dispatch"));
    }
    if request["id"] != json!(reservation)
        || request["contract"] != json!(contract)
        || request["family"] != body["family"]
        || request["epoch"] != intent["epoch"]
        || request["boot"] != intent["boot"]
        || request["request"]["kind"] != "reload"
        || request["request"]["resources"] != intent["resources"]
        || request["request"]["operation"] != json!(hash(&(&body["operation"], "promotion_cost"))?)
        || outcome["request"] != json!(reservation)
        || outcome["branch_closed"] != false
        || outcome["settlement"]["outcome"] != "completed"
        || outcome["settlement"]["resources_released"] != true
        || outcome["settlement"]["evidence"] != body["reload"]
        || outcome["settlement"]["operation"]
            != json!(hash(&(&body["operation"], "promotion_settlement"))?)
        || body["schema_version"] != 1
        || intent["request"]["protocol"] != "native_fixture_promotion_v1"
        || probes["origin"] != "native_replay_before_reexport"
        || reload["schema_version"] != 1
        || !reload["failure"].is_null()
        || reload["outputs_eligible"] != false
        || reload["promotion_status"] != "not_evaluated"
        || reload["verifier"]
            != json!(ContentDigest::of_bytes(include_bytes!(
                "../../model_worker/reload.rs"
            )))
    {
        return Err(deny("durable promotion semantics differ"));
    }
    let archive = &probes["archive"];
    let checkpoint: Vec<u8> = serde_json::from_value(archive["checkpoint"].clone())
        .map_err(|_| deny("missing archived checkpoint"))?;
    let manifest = Validated::<ModelManifest>::parse(&canonical(&archive["manifest"])?)?;
    let source = if manifest.architecture.id == "nf.arithmetic.relu" {
        let raw_assets: BTreeMap<String, Vec<u8>> =
            serde_json::from_value(archive["assets"].clone())
                .map_err(|_| deny("missing archived arithmetic assets"))?;
        let assets = raw_assets
            .into_iter()
            .map(|(k, v)| (k, Arc::from(v)))
            .collect();
        SourceState::load(
            manifest,
            Arc::from(checkpoint),
            assets,
            StateLoadLimits {
                max_checkpoint_bytes: 4096,
                max_tensor_bytes: 336,
                max_tensors: 2,
                max_asset_bytes: 2048,
            },
        )?
    } else {
        SourceState::load(
            manifest,
            Arc::from(checkpoint),
            BTreeMap::from([("config".into(), Arc::from(b"{}".as_slice()))]),
            StateLoadLimits {
                max_checkpoint_bytes: 4096,
                max_tensor_bytes: 32,
                max_tensors: 2,
                max_asset_bytes: 2,
            },
        )?
    };
    let target: ModelIdentity = serde_json::from_value(body["model"].clone())
        .map_err(|_| deny("invalid accepted model"))?;
    let inputs: Vec<Vec<f64>> = serde_json::from_value(archive["inputs"].clone())
        .map_err(|_| deny("missing archived inputs"))?;
    let seeds: Vec<u64> = serde_json::from_value(archive["seeds"].clone())
        .map_err(|_| deny("missing archived seeds"))?;
    if inputs.is_empty()
        || inputs.len() > 16
        || inputs.iter().any(|v| v.len() != 2)
        || seeds != vec![contract_body.seeds[0]; inputs.len()]
    {
        return Err(deny("invalid frozen probe schedule"));
    }
    let smoke = Validated::<SmokeSuite>::parse(&canonical(&archive["smoke_suite"])?)?;
    if !matches!(
        smoke.adapter.as_str(),
        "nf.linear_fixture.v1" | "nf.arithmetic_relu.v1"
    ) || smoke.digest()?
        != serde_json::from_value::<ContentDigest>(probes["smoke"].clone())
            .map_err(|_| deny("missing smoke identity"))?
        || smoke
            .probes
            .iter()
            .map(|p| p.input.clone())
            .collect::<Vec<_>>()
            != inputs
        || probes["seed"] != json!(contract_body.seeds[0])
    {
        return Err(deny("archived probes differ from approved smoke"));
    }
    let frozen = FrozenReloadProbes::freeze(&source, target, inputs, seeds)?;
    if probes["probes"] != json!(frozen.id()) {
        return Err(deny("frozen probe digest differs from archive"));
    }
    frozen.audit_child_report(&source, &reload["child_report"])?;
    Ok(())
}
