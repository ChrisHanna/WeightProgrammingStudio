//! Test-support server: trusted Rust fixture broker over private framed stdio.
#[allow(dead_code)]
mod fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../nf-core/tests/common/study_fixture.rs"
    ));
}
use nf_checkpoint::store::{ArtifactStore, StoreLimits};
use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use nf_core::schemas::{ModelManifest, Validated};
use nf_core::study::StudyContract;
use nf_experiments::coordinator::broker::{Actor, Quotas, Scope, Session, Task};
use nf_experiments::coordinator::budgets::{Request, Resources, WorkKind};
use nf_experiments::coordinator::{Ledger, LedgerLimits, Stage, Transition};
use nf_graph::inventory::tensor_set_digest;
use nf_surgeon::transaction::{SourceState, StateLoadLimits};
use std::collections::BTreeMap;
use std::io::Read;
use std::sync::Arc;
fn hex(value: &str) -> Vec<u8> {
    value
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}
fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    assert!((3..=4).contains(&args.len()),"fixture_broker FIXTURE.json ABSOLUTE_ROOT proposer|selection_evaluator|final_evaluator [WALL_MS]");
    let wall_ms: u64 = args
        .get(3)
        .map(|value| value.parse().unwrap())
        .unwrap_or(30000);
    let mut bytes = Vec::new();
    std::fs::File::open(&args[0])
        .unwrap()
        .take(1_048_577)
        .read_to_end(&mut bytes)
        .unwrap();
    let doc = CanonicalDocument::parse(&bytes, Limits::default()).unwrap();
    let value = doc.value();
    let manifest: ModelManifest = serde_json::from_value(value["manifest"].clone()).unwrap();
    let checkpoint: Arc<[u8]> = Arc::from(hex(value["checkpoint_hex"].as_str().unwrap()));
    let assets: BTreeMap<String, Arc<[u8]>> = value["assets_hex"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, data)| (name.clone(), Arc::from(hex(data.as_str().unwrap()))))
        .collect();
    let source = Arc::new(
        SourceState::load(
            Validated::new(manifest).unwrap(),
            checkpoint,
            assets,
            StateLoadLimits::default(),
        )
        .unwrap(),
    );
    let model = serde_json::json!({"checkpoint":ContentDigest::of_bytes(source.checkpoint_bytes()),"manifest":source.manifest_digest(),"parameters":source.manifest().parameter_digest,"buffers":tensor_set_digest(&source.manifest().buffers).unwrap(),"graph":value["program"]["base_model"]["architecture_graph"],"config":source.manifest().architecture.config,"tokenizer":null,"system_state":null,"runtime_artifact":null});
    let mut contract = fixture::standard();
    contract["source"] = model.clone();
    contract["best_known"] = model.clone();
    for baseline in contract["baselines"].as_array_mut().unwrap() {
        baseline["input"] = model.clone();
    }
    contract["limits"]["host_memory_bytes"] = serde_json::json!(268435456);
    let contract =
        Validated::<StudyContract>::parse(&serde_json::to_vec(&contract).unwrap()).unwrap();
    let root = std::path::Path::new(&args[1]);
    let mut ledger = Ledger::open(&root.join("ledger"), LedgerLimits::default()).unwrap();
    let id = ledger.freeze(&contract).unwrap();
    let lease = ledger
        .acquire(&id, "fixture-broker-controller", 300000)
        .unwrap();
    let actor: Actor = serde_json::from_value(serde_json::Value::String(args[2].clone())).unwrap();
    if actor == Actor::FinalEvaluator {
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
            ledger
                .advance(
                    &lease,
                    Transition {
                        operation: fixture::d(&format!("final-phase{i}")),
                        expected: pair[0],
                        next: pair[1],
                        protocol: fixture::d("trusted-final-panel"),
                        evidence: fixture::d("fixture-evidence"),
                        message: "trusted CPU software fixture only".to_owned(),
                    },
                )
                .unwrap();
        }
        ledger
            .freeze_final_panel(&lease, vec![contract.source.clone()])
            .unwrap();
    }
    let pool = ledger
        .reserve(
            &lease,
            Request {
                schema_version: 1,
                operation: fixture::d("worker-pool"),
                attempt: 0,
                parent: None,
                kind: WorkKind::Branch,
                resources: Resources {
                    cpu_ms: 60000,
                    evaluations: 4,
                    host_bytes: 134217728,
                    storage_bytes: 4194304,
                    workers: 1,
                    ..Resources::default()
                },
                wall_ms,
                external: None,
            },
        )
        .unwrap();
    let binding = &contract.portfolio.bindings[0];
    let split = match actor {
        Actor::Proposer => &binding.splits.development,
        Actor::SelectionEvaluator => &binding.splits.selection,
        Actor::FinalEvaluator => &binding.splits.holdout,
    };
    let tasks = split
        .items
        .iter()
        .map(|item| Task {
            id: item.item.clone(),
            group: item.group.clone(),
            input: vec![1.0, 2.0],
            labels: Some(serde_json::json!({"expected":[5.0,11.0]})),
        })
        .collect();
    let store = ArtifactStore::create(
        &root.join("staged"),
        StoreLimits {
            total_bytes: 2097152,
            max_blob_bytes: 65536,
            max_entries: 128,
        },
    )
    .unwrap();
    let mut session = Session::trusted_fixture(
        &mut ledger,
        lease,
        pool,
        source,
        &store,
        Scope {
            binding: binding.id.clone(),
            actor,
            tasks,
            quotas: Quotas {
                messages: 32,
                forwards: 2,
                tool_calls: 2,
                input_units: 8,
                response_bytes: 65536,
                staged_bytes: 256,
            },
        },
    )
    .unwrap();
    use std::os::fd::AsFd;
    let socket =
        std::os::unix::net::UnixStream::from(rustix::io::dup(std::io::stdin().as_fd()).unwrap());
    if let Err(error) = session.serve_unix(&socket) {
        eprintln!("{}", serde_json::to_string(&error).unwrap());
        std::process::exit(2);
    }
}
