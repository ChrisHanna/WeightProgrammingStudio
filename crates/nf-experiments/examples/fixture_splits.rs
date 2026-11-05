//! Controller-owned actual split broker test support. Never a gym endpoint.
use nf_checkpoint::store::{ArtifactStore, StoreLimits};
use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use nf_core::error::{CoreError, ErrorCode, Result};
use nf_core::schemas::Validated;
use nf_core::study::{SplitRole, StudyContract, TaskIdentity};
use nf_experiments::coordinator::splits::{
    self, Dataset, Grant, Item, ReadRequest, Recipient, ScopedPayload,
};
use nf_experiments::coordinator::{Ledger, LedgerLimits, Stage, Transition};
use nf_experiments::sandbox::{Limits as SandboxLimits, QualifiedDocker};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use std::path::Path;
#[allow(dead_code)]
mod fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../nf-core/tests/common/study_fixture.rs"
    ));
}
fn fail(text: &str) -> CoreError {
    CoreError::new(ErrorCode::InvalidValue, "/fixture", text)
}
fn bytes(v: &impl serde::Serialize) -> Vec<u8> {
    CanonicalDocument::parse(&serde_json::to_vec(v).unwrap(), Limits::default())
        .unwrap()
        .bytes()
        .to_vec()
}
fn setup(
    root: &Path,
    options: &Value,
) -> Result<(
    Ledger,
    ArtifactStore,
    Validated<StudyContract>,
    ContentDigest,
    QualifiedDocker,
)> {
    let store = ArtifactStore::create(
        &root.join("store"),
        StoreLimits {
            total_bytes: 16777216,
            max_blob_bytes: 1048576,
            max_entries: 1024,
        },
    )?;
    let mut value = fixture::standard();
    let mut public = Vec::new();
    for (bi, binding) in value["portfolio"]["bindings"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .enumerate()
    {
        let mut items = BTreeMap::new();
        for (ri, role) in ["development", "selection", "holdout"]
            .into_iter()
            .enumerate()
        {
            let split = &mut binding["splits"][role];
            let identity: TaskIdentity = serde_json::from_value(split["items"][0].clone()).unwrap();
            let input = if options["duplicate_inputs"] == true && bi == 0 {
                json!({"value":0})
            } else if options["cross_binding_alias"] == true
                && ((bi == 0 && ri == 2) || (bi == 1 && ri == 0))
            {
                json!({"alias":"cross-binding"})
            } else {
                json!({"value":bi*10+ri,"task":identity.item,"role":role})
            };
            let input = store.put(bytes(&input).as_slice())?.digest;
            let label = store
                .put(bytes(&json!({"secret":format!("PRIVATE_LABEL_{bi}_{role}")})).as_slice())?
                .digest;
            let mut dataset = Dataset {
                schema_version: 1,
                role: serde_json::from_value(json!(role)).unwrap(),
                items: vec![Item {
                    identity: identity.clone(),
                    input,
                    label,
                }],
            };
            if options["wrong_dataset_role"] == true && bi == 0 && ri == 0 {
                dataset.role = SplitRole::Holdout;
            }
            if options["wrong_identity"] == true && bi == 0 && ri == 0 {
                dataset.items[0].identity.group = fixture::d("wrong-group");
            }
            let digest = splits::publish_dataset(&store, &dataset)?;
            split["dataset"] = json!(if options["unresolved"] == true && bi == 0 && ri == 0 {
                fixture::d("missing")
            } else {
                digest
            });
            items.insert(role, identity.item);
        }
        public.push(json!({"binding":binding["id"],"items":items}));
    }
    fixture::refresh(&mut value);
    let contract = Validated::<StudyContract>::parse(&serde_json::to_vec(&value).unwrap())?;
    let mut ledger = Ledger::open(&root.join("ledger"), LedgerLimits::default())?;
    let id = ledger.freeze(&contract)?;
    let lease = ledger.acquire(&id, "split-controller", 300000)?;
    for binding in &contract.portfolio.bindings {
        ledger.bind_split_catalog(&lease, &store, &binding.id)?;
    }
    // A private child is created; no existing/system permissions are changed.
    use std::os::unix::fs::DirBuilderExt;
    std::fs::DirBuilder::new()
        .mode(0o700)
        .create(root.join("sandbox"))
        .map_err(|e| fail(&e.to_string()))?;
    let runtime = QualifiedDocker::qualify(true, &root.join("sandbox"))?;
    // Only the trusted test controller sees the complete metadata map.
    println!("{}", json!({"ready":true,"contract":id,"bindings":public}));
    std::io::stdout().flush().unwrap();
    ledger.release(&lease)?;
    Ok((ledger, store, contract, id, runtime))
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let root = Path::new(&args[1]);
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    let options: Value = serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();
    let (mut ledger, store, contract, id, runtime) = match setup(root, &options) {
        Ok(v) => v,
        Err(e) => {
            println!("{}", json!({"error":e}));
            return;
        }
    };
    let mut lease = ledger.acquire(&id, "split-controller", 300000).unwrap();
    let mut grants: BTreeMap<String, Grant> = BTreeMap::new();
    let mut prepared: BTreeMap<String, ScopedPayload> = BTreeMap::new();
    for line in lines {
        let line = line.unwrap();
        if line.len() > 65536 {
            break;
        }
        let command: Value = serde_json::from_str(&line).unwrap();
        let result = (|| -> Result<Value> {
            let binding = contract
                .portfolio
                .bindings
                .get(command["binding_index"].as_u64().unwrap_or(0) as usize)
                .ok_or_else(|| fail("unknown binding"))?;
            match command["op"].as_str().unwrap_or("") {
                "grant" => {
                    let recipient: Recipient = serde_json::from_value(command["recipient"].clone())
                        .map_err(|_| fail("bad recipient"))?;
                    let grant = ledger.grant_split_worker(
                        &lease,
                        &binding.id,
                        recipient,
                        &runtime,
                        command["ttl_ms"].as_u64().unwrap_or(60000),
                    )?;
                    let value = serde_json::to_value(grant.welcome()).unwrap();
                    grants.insert(grant.welcome().worker.clone(), grant);
                    Ok(value)
                }
                "read" | "domain" | "prepare" => {
                    let request = ReadRequest::parse(&bytes(&command["request"]))?;
                    let grant = grants
                        .get(&request.worker)
                        .ok_or_else(|| fail("unknown worker"))?;
                    let context = if command["context"].is_null() {
                        Default::default()
                    } else {
                        serde_json::from_value(command["context"].clone())
                            .map_err(|_| fail("invalid exposure context"))?
                    };
                    let payload =
                        ledger.read_split_traced(&lease, grant, &request, &store, context)?;
                    if command["op"] == "prepare" {
                        let id = payload.access_id().to_string();
                        prepared.insert(id.clone(), payload);
                        return Ok(json!({"access":id}));
                    }
                    if command["op"] == "domain" {
                        let source = command["code"]
                            .as_str()
                            .ok_or_else(|| fail("missing source"))?;
                        let receipt = payload.in_domain(
                            &ledger,
                            &lease,
                            &runtime,
                            source.as_bytes(),
                            SandboxLimits::default(),
                        )?;
                        return Ok(serde_json::to_value(receipt).unwrap());
                    }
                    Ok(serde_json::from_slice(payload.bytes()).unwrap())
                }
                "dispatch_prepared" => {
                    let payload = prepared
                        .get(command["access"].as_str().unwrap_or(""))
                        .ok_or_else(|| fail("unknown prepared data"))?;
                    let receipt = payload.in_domain(
                        &ledger,
                        &lease,
                        &runtime,
                        b"print('MUST NOT RUN')",
                        SandboxLimits::default(),
                    )?;
                    Ok(serde_json::to_value(receipt).unwrap())
                }
                "final_phase" => {
                    let stages = [
                        Stage::Frozen,
                        Stage::Preflight,
                        Stage::Profile,
                        Stage::Baseline,
                        Stage::Design,
                        Stage::Execute,
                        Stage::Evaluate,
                        Stage::Verify,
                    ];
                    for (i, pair) in stages.windows(2).enumerate() {
                        ledger.advance(
                            &lease,
                            Transition {
                                operation: fixture::d(&format!("phase{i}")),
                                expected: pair[0],
                                next: pair[1],
                                protocol: fixture::d("phases"),
                                evidence: fixture::d("fixture-evidence"),
                                message: "software access fixture only".into(),
                            },
                        )?;
                    }
                    Ok(json!({"phase":"verify"}))
                }
                "freeze" => {
                    let mut candidate = contract.source.clone();
                    if let Some(salt) = command["candidate_salt"].as_str() {
                        candidate.parameters = fixture::d(salt);
                    }
                    let count = command["candidate_count"].as_u64().unwrap_or(1);
                    if count > 64 {
                        return Err(fail("fixture candidate bound"));
                    }
                    let candidates = vec![candidate; count as usize];
                    let panel = ledger.freeze_final_panel(&lease, candidates)?;
                    Ok(json!({"panel":panel}))
                }
                "events" => {
                    Ok(serde_json::to_value(ledger.exposure_events(&lease, 0, 1000)?).unwrap())
                }
                "feedback" => {
                    use nf_experiments::coordinator::exposure::{Context, Disclosure, Field};
                    let role: SplitRole = serde_json::from_value(command["split"].clone())
                        .map_err(|_| fail("invalid split"))?;
                    let split = match role {
                        SplitRole::Development => &binding.splits.development,
                        SplitRole::Selection => &binding.splits.selection,
                        SplitRole::Holdout => &binding.splits.holdout,
                    };
                    let artifact = store.put(bytes(&command["value"]).as_slice())?.digest;
                    let event = ledger.disclose_score(
                        &lease,
                        &store,
                        Disclosure {
                            operation: fixture::d(
                                command["operation"].as_str().unwrap_or("score-feedback"),
                            ),
                            binding: binding.id.clone(),
                            item: split.items[0].item.clone(),
                            split: role,
                            recipient: "programmer-fixture".into(),
                            role: Recipient::Proposer,
                            fields: [Field::Summary, Field::Feedback].into_iter().collect(),
                            artifact,
                            context: Context {
                                seed: Some(17),
                                ..Default::default()
                            },
                        },
                    )?;
                    Ok(serde_json::to_value(event).unwrap())
                }
                "corrupt_input" => {
                    let dataset: Dataset = serde_json::from_slice(
                        &store.read_bounded(&binding.splits.development.dataset, 1048576)?,
                    )
                    .unwrap();
                    let path = root
                        .join("store/blobs")
                        .join(&dataset.items[0].input.as_str()[7..]);
                    std::fs::remove_file(&path).map_err(|e| fail(&e.to_string()))?;
                    std::fs::write(path, b"corrupt").map_err(|e| fail(&e.to_string()))?;
                    Ok(json!({"corrupted":true}))
                }
                "panel" => Ok(serde_json::to_value(ledger.frozen_final_panel(&id)?).unwrap()),
                "reopen" => {
                    ledger = Ledger::open(&root.join("ledger"), LedgerLimits::default())?;
                    Ok(json!({"reopened":true}))
                }
                "revision_alias" => {
                    let mut next = serde_json::to_value(&*contract).unwrap();
                    next["revision"] = json!(2);
                    next["parent_contract"] = json!(id);
                    next["approval_receipt"] = json!(fixture::d("new-owner-approval"));
                    let first = &contract.portfolio.bindings[0].splits.holdout.dataset;
                    let old: Dataset =
                        serde_json::from_slice(&store.read_bounded(first, 1048576)?).unwrap();
                    let consumed = old.items[0].input.clone();
                    for (bi, binding) in next["portfolio"]["bindings"]
                        .as_array_mut()
                        .unwrap()
                        .iter_mut()
                        .enumerate()
                    {
                        for role in ["development", "selection", "holdout"] {
                            let split = &mut binding["splits"][role];
                            let mut dataset: Dataset =
                                serde_json::from_slice(&store.read_bounded(
                                    &serde_json::from_value(split["dataset"].clone()).unwrap(),
                                    1048576,
                                )?)
                                .unwrap();
                            split["items"][0]["item"] =
                                json!(fixture::d(&format!("renamed-item-{bi}-{role}")));
                            split["items"][0]["group"] =
                                json!(fixture::d(&format!("renamed-group-{bi}-{role}")));
                            dataset.items[0].identity =
                                serde_json::from_value(split["items"][0].clone()).unwrap();
                            if bi == 0 && role == "development" {
                                dataset.items[0].input = consumed.clone();
                            } else {
                                dataset.items[0].input=store.put(bytes(&json!({"revision":2,"task":dataset.items[0].identity.item,"role":role})).as_slice())?.digest;
                            }
                            split["dataset"] = json!(splits::publish_dataset(&store, &dataset)?);
                        }
                    }
                    fixture::refresh(&mut next);
                    let next =
                        Validated::<StudyContract>::parse(&serde_json::to_vec(&next).unwrap())?;
                    ledger.release(&lease)?;
                    let nid = ledger.freeze(&next)?;
                    let nl = ledger.acquire(&nid, "new-controller", 300000)?;
                    ledger.bind_split_catalog(&nl, &store, &next.portfolio.bindings[0].id)?;
                    Ok(json!({"must_not_reach":true}))
                }
                "renew" => {
                    lease = ledger.renew(&lease, 300000)?;
                    Ok(json!({"renewed":true}))
                }
                "search" => Ok(serde_json::to_value(ledger.advance(
                    &lease,
                    Transition {
                        operation: fixture::d("post-holdout"),
                        expected: Stage::Verify,
                        next: Stage::Search,
                        protocol: fixture::d("bad"),
                        evidence: fixture::d("bad"),
                        message: "must deny".into(),
                    },
                )?)
                .unwrap()),
                "reserve_search" => {
                    use nf_experiments::coordinator::budgets::{Request, Resources, WorkKind};
                    let reserved = ledger.reserve(
                        &lease,
                        Request {
                            schema_version: 1,
                            operation: fixture::d("post-final-search"),
                            attempt: 0,
                            parent: None,
                            kind: WorkKind::Search,
                            resources: Resources {
                                cpu_ms: 100,
                                host_bytes: 1000,
                                workers: 1,
                                ..Default::default()
                            },
                            wall_ms: 1000,
                            external: None,
                        },
                    )?;
                    Ok(json!({"reservation":reserved.id()}))
                }
                "stop" => {
                    ledger.request_stop(
                        &id,
                        fixture::d("stop"),
                        nf_experiments::coordinator::supervision::StopReason::OwnerStop,
                    )?;
                    Ok(json!({"stopped":true}))
                }
                _ => Err(fail("unknown test-controller operation")),
            }
        })();
        println!(
            "{}",
            match result {
                Ok(v) => json!({"ok":v}),
                Err(e) => json!({"error":e}),
            }
        );
        std::io::stdout().flush().unwrap();
    }
}
