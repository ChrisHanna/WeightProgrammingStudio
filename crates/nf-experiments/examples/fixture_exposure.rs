//! Explicit controller-owned native legacy intake fixture; never imports code.
use nf_checkpoint::store::{ArtifactStore, StoreLimits};
use nf_core::canonical::{CanonicalDocument, Limits};
use nf_core::schemas::Validated;
use nf_core::study::{SplitRole, StudyContract, TaskIdentity};
use nf_experiments::coordinator::exposure::{Field, LegacyObservation, LegacyRecord};
use nf_experiments::coordinator::splits::{self, Dataset, Item};
use nf_experiments::coordinator::{Ledger, LedgerLimits};
use serde_json::{json, Value};
use std::io::Read;
#[allow(dead_code)]
mod fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../nf-core/tests/common/study_fixture.rs"
    ));
}
fn bytes(value: &impl serde::Serialize) -> Vec<u8> {
    CanonicalDocument::parse(&serde_json::to_vec(value).unwrap(), Limits::default())
        .unwrap()
        .bytes()
        .to_vec()
}
fn main() {
    let root = std::path::PathBuf::from(std::env::args().nth(1).unwrap());
    let mut raw = Vec::new();
    std::io::stdin().take(65537).read_to_end(&mut raw).unwrap();
    if raw.len() > 65536 {
        std::process::exit(2);
    }
    let options: Value = serde_json::from_slice(&raw).unwrap();
    let result = (|| -> nf_core::error::Result<Value> {
        let store = ArtifactStore::create(
            &root.join("store"),
            StoreLimits {
                total_bytes: 16777216,
                max_blob_bytes: 1048576,
                max_entries: 1024,
            },
        )?;
        let make = |name: &str, value: u64| -> nf_core::error::Result<Item> {
            Ok(Item {
                identity: TaskIdentity {
                    item: fixture::d(name),
                    group: fixture::d(&format!("group-{name}")),
                    family: fixture::d("legacy-task-family"),
                },
                input: store
                    .put(bytes(&json!({"task_id":name,"value":value,"nonce":"old"})).as_slice())?
                    .digest,
                label: store.put(bytes(&json!(value * 2)).as_slice())?.digest,
            })
        };
        let edit = make("old-edit", 2)?;
        let sealed = make("old-sealed", 3)?;
        let history = if options["history"] == "missing" {
            None
        } else {
            Some(vec![
                LegacyObservation {
                    item: edit.identity.item.clone(),
                    recipient: "old-programmer".into(),
                    fields: [Field::Input, Field::Label].into_iter().collect(),
                    seed: Some(17),
                    attempt: 0,
                    retry_of: None,
                    outcome: "completed".into(),
                },
                LegacyObservation {
                    item: sealed.identity.item.clone(),
                    recipient: "old-score-feedback".into(),
                    fields: [Field::Summary].into_iter().collect(),
                    seed: Some(42),
                    attempt: 1,
                    retry_of: None,
                    outcome: "failed_or_partial".into(),
                },
                LegacyObservation {
                    item: sealed.identity.item.clone(),
                    recipient: "old-teacher".into(),
                    fields: [Field::Feedback].into_iter().collect(),
                    seed: Some(99),
                    attempt: 2,
                    retry_of: Some(0),
                    outcome: "completed".into(),
                },
            ])
        };
        let legacy = LegacyRecord {
            schema_version: 1,
            edit: vec![edit.clone()],
            sealed: vec![sealed.clone()],
            history,
        };
        let source = store.put(bytes(&legacy).as_slice())?.digest;
        let mut contract = fixture::standard();
        let destination = options["destination"].as_str().unwrap_or("development");
        let selected = if options["bucket"] == "sealed" {
            &sealed
        } else {
            &edit
        };
        for (bi, binding) in contract["portfolio"]["bindings"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .enumerate()
        {
            for (ri, role) in ["development", "selection", "holdout"]
                .into_iter()
                .enumerate()
            {
                let split = &mut binding["splits"][role];
                let mut row = Item {
                    identity: serde_json::from_value(split["items"][0].clone()).unwrap(),
                    input: store
                        .put(bytes(&json!({"value":1000+bi*10+ri})).as_slice())?
                        .digest,
                    label: store.put(bytes(&json!(0)).as_slice())?.digest,
                };
                if bi == 0 && role == destination {
                    if options["renamed"] == true {
                        row.identity.item = fixture::d("renamed-task");
                        row.identity.group = fixture::d("renamed-group");
                        split["items"][0] = serde_json::to_value(&row.identity).unwrap();
                        let val = if options["bucket"] == "sealed" { 3 } else { 2 };
                        row.input=store.put(bytes(&json!({"version":"new","role":role,"task_id":"renamed-task","nonce":"brand-new","value":val})).as_slice())?.digest;
                    } else {
                        row.input = selected.input.clone();
                    }
                }
                split["dataset"] = json!(splits::publish_dataset(
                    &store,
                    &Dataset {
                        schema_version: 1,
                        role: serde_json::from_value(json!(role)).unwrap(),
                        items: vec![row]
                    }
                )?);
            }
        }
        fixture::refresh(&mut contract);
        let contract = Validated::<StudyContract>::parse(&serde_json::to_vec(&contract).unwrap())?;
        let mut ledger = Ledger::open(&root.join("ledger"), LedgerLimits::default())?;
        let id = ledger.freeze(&contract)?;
        let lease = ledger.acquire(&id, "legacy-controller", 300000)?;
        let imported = ledger.import_legacy_split(&lease, &store, &source, legacy.clone())?;
        let again = ledger.import_legacy_split(&lease, &store, &source, legacy)?;
        let report = ledger.legacy_assessment(&lease, &imported)?;
        let mut admissions = serde_json::Map::new();
        for role in [
            SplitRole::Development,
            SplitRole::Selection,
            SplitRole::Holdout,
        ] {
            let result = ledger.admit_legacy_role(&lease, &imported, &selected.identity.item, role);
            admissions.insert(
                serde_json::to_value(role).unwrap().as_str().unwrap().into(),
                json!({"allowed":result.is_ok()}),
            );
        }
        let bound = ledger.bind_split_catalog(&lease, &store, &contract.portfolio.bindings[0].id);
        let history = ledger.exposure_events(&lease, 0, 1000)?;
        let count = history.len();
        drop(ledger);
        let ledger = Ledger::open(&root.join("ledger"), LedgerLimits::default())?;
        let retained = ledger.exposure_events(&lease, 0, 1000)?;
        Ok(
            json!({"source":source,"assessment":report,"assessment_id":imported,"idempotent":imported==again,"admission":admissions,"binding":match bound {Ok(id)=>json!({"ok":id}),Err(e)=>json!({"error":e})},"events":history,"persisted":count==retained.len()}),
        )
    })();
    println!(
        "{}",
        match result {
            Ok(v) => json!({"ok":v}),
            Err(e) => json!({"error":e}),
        }
    );
}
