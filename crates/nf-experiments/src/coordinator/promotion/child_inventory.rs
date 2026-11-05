//! Global cleanup inventory bridge, not a reload witness or spawn authority.
use super::*;

pub(super) fn clean(c: &Connection, contract: &ContentDigest) -> Result<bool> {
    // Supervision also runs while opening pre-v7 ledgers during migration tests.
    let exists: bool = c.query_row("SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='promotion_records')", [], |r| r.get(0)).map_err(sql)?;
    if !exists {
        return Ok(true);
    }
    let Some((_, child)) = read(c, contract, "child_intent")? else {
        if read(c, contract, "child_registered")?.is_some() {
            return Err(deny("orphan promotion child registration"));
        }
        return Ok(true);
    };
    let (intent_id, intent) =
        read(c, contract, "intent")?.ok_or_else(|| deny("orphan promotion child intent"))?;
    if child["intent"] != json!(intent_id) {
        return Err(deny("promotion child intent binding differs"));
    }
    let registered_gone = child_registration::registered_gone(c, contract, &intent_id, &intent)?;
    if let Some((_, terminal)) = read(c, contract, "attempt_end")? {
        if terminal["intent"] == json!(intent_id)
            && terminal["status"] == "incomplete_no_rerun"
            && terminal["cleanup_confirmed"] == true
            && terminal["child_entered"] == false
        {
            if registered_gone.is_some() {
                return Err(deny("prelaunch terminal has a registered child"));
            }
            return Ok(true); // Admission ended before the synchronous worker was entered.
        }
    }
    if let Some((_, recovery)) = read(c, contract, "child_recovery")? {
        if recovery["status"] == "registered_nonforking_child_gone" {
            return child_registration::recovery_clean(c, contract, &intent_id, &intent, &recovery);
        }
        let launch = intent["native_process"]["boot"]
            .as_str()
            .ok_or_else(|| deny("missing launch boot"))?;
        let observed = recovery["observed_boot"]
            .as_str()
            .ok_or_else(|| deny("missing recovery boot"))?;
        if recovery["intent"] != json!(intent_id)
            || recovery["launch_boot"] != launch
            || intent["boot"] != launch
            || observed.is_empty()
            || observed == launch
            || files::moment()?.boot == launch
            || recovery["status"] != "previous_boot_processes_gone"
        {
            return Err(deny("invalid promotion reboot cleanup evidence"));
        }
        return Ok(true); // A different kernel boot cannot retain the old processes.
    }
    let Some((_, reload)) = read(c, contract, "reload")? else {
        return Ok(false);
    };
    let Some((_, supervision)) = read(c, contract, "supervision")? else {
        return Ok(false);
    };
    Ok(reload["cleanup_confirmed"] == true
        && supervision["cleanup_confirmed"] == true
        && supervision["intent"] == json!(intent_id)
        && registered_gone != Some(false))
}

/// Recover a registered exact child, otherwise require native reboot evidence.
/// Missing same-boot registration is never evidence that no child was launched.
pub(super) fn reconcile(
    tx: &Transaction<'_>,
    contract: &ContentDigest,
    intent_id: &ContentDigest,
    intent: &Value,
) -> Result<()> {
    if clean(tx, contract)? {
        return Ok(());
    }
    if child_registration::recover(tx, contract, intent_id, intent)? {
        return Ok(());
    }
    observe_reboot(tx, contract, intent_id, intent)
}

/// Native observation of a changed kernel boot also proves unknown-child cleanup.
pub(super) fn observe_reboot(
    tx: &Transaction<'_>,
    contract: &ContentDigest,
    intent_id: &ContentDigest,
    intent: &Value,
) -> Result<()> {
    if clean(tx, contract)? {
        return Ok(());
    }
    let launch = intent["native_process"]["boot"]
        .as_str()
        .ok_or_else(|| deny("missing native launch boot"))?;
    let now = files::moment()?;
    if launch.is_empty() || intent["boot"] != launch || now.boot == launch {
        return Err(deny(
            "same-boot child cleanup unknown; reservation retained for supervised recovery",
        ));
    }
    record(
        tx,
        contract,
        "child_recovery",
        &json!({"intent":intent_id,
        "launch_boot":launch,"observed_boot":now.boot,"status":"previous_boot_processes_gone"}),
    )?;
    Ok(())
}
