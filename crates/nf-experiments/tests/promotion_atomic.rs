// Ungranted promotion stays blocked. Share the real native fixture between
// public grant-path integration tests and private adversarial unit tests.
macro_rules! promotion_atomic_cases { () => {
#[path = "../../nf-core/tests/common/study_fixture.rs"]
#[allow(dead_code)] mod fixture;
use fixture::{d,hash,refresh,standard};
use nf_checkpoint::store::{ArtifactStore,StoreLimits};
use nf_core::{canonical::{CanonicalDocument,ContentDigest,Limits},schemas::{ModelManifest,Validated},study::{ModelIdentity,StudyContract,SplitRole}};
use nf_experiments::{coordinator::{Ledger,LedgerLimits,Lease,Stage,Transition,final_fixture::{graph_bytes,profile_id,runtime_profile},splits::{Dataset,Item,publish_dataset}},model_worker::QualifiedTorch};
use nf_graph::inventory::{tensor_set_digest,verify_inventory,InventoryLimits};
use nf_surgeon::{cancellation::CancellationToken,transaction::{SourceState,StateLoadLimits,SmokeSuite}};
use nf_weight_ir::{compilation::{compile,CompilePolicy,CompiledPlan,SourceSnapshot},program::parse_program,search_space::{parse_search_space,resolve_space}};
use serde::Serialize;
use serde_json::{json,Value};
use std::{collections::BTreeMap,path::{Path,PathBuf},sync::{Arc,atomic::{AtomicU64,Ordering}},os::unix::fs::DirBuilderExt};
static NEXT:AtomicU64=AtomicU64::new(0);
// Never lower a shared test runner's hard limit. Exercise T038 in a dedicated
// process, as its native profile requires. This is a TEST harness only.
pub(crate) fn dedicated_test()->bool {
    if std::env::var_os("NF_PROMOTION_TEST_LIMITED").is_some() { return false; }
    let name=std::thread::current().name().unwrap().to_owned();
    let status=std::process::Command::new("/usr/bin/prlimit")
        .args(["--as=3221225472:3221225472","--"])
        .arg(std::env::current_exe().unwrap()).args(["--exact",&name,"--test-threads=1"])
        .env("NF_PROMOTION_TEST_LIMITED","1").status().unwrap();
    assert!(status.success(),"dedicated promotion test failed: {status}"); true
}
fn bytes(v:&impl Serialize)->Vec<u8>{CanonicalDocument::parse(&serde_json::to_vec(v).unwrap(),Limits::default()).unwrap().bytes().to_vec()}
pub(crate) struct Fixture {pub(crate) root:PathBuf,pub(crate) ledger:Ledger,pub(crate) lease:Lease,pub(crate) store:ArtifactStore,pub(crate) edit:CompiledPlan,pub(crate) smoke:Validated<SmokeSuite>,pub(crate) family:ContentDigest}
impl Drop for Fixture {fn drop(&mut self){let _=std::fs::remove_dir_all(&self.root);}}
impl Fixture {
    pub(crate) fn new(pass:bool)->Self {
        let root=std::env::temp_dir().join(format!("atomic-promotion-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));
        std::fs::DirBuilder::new().mode(0o700).create(&root).unwrap();
        let store=ArtifactStore::create(&root.join("store"),StoreLimits{total_bytes:16777216,max_blob_bytes:1048576,max_entries:1024}).unwrap();
        let mut f:Value=serde_json::from_str(include_str!("../../../python/neural_foundry/data/m1_fixture.json")).unwrap();
        let hex=f["checkpoint_hex"].as_str().unwrap();
        let checkpoint=(0..hex.len()).step_by(2).map(|i|u8::from_str_radix(&hex[i..i+2],16).unwrap()).collect::<Vec<_>>();
        let source=SourceState::load(Validated::<ModelManifest>::parse(&bytes(&f["manifest"])).unwrap(),Arc::from(checkpoint),
            BTreeMap::from([("config".into(),Arc::from(b"{}".as_slice()))]),StateLoadLimits::default()).unwrap();
        for b in [bytes(&**source.manifest()),source.checkpoint_bytes().to_vec(),b"{}".to_vec(),graph_bytes().to_vec()] {store.put(b.as_slice()).unwrap();}
        let model=ModelIdentity{checkpoint:ContentDigest::of_bytes(source.checkpoint_bytes()),manifest:source.manifest_digest().clone(),parameters:source.manifest().parameter_digest.clone(),
            buffers:tensor_set_digest(&source.manifest().buffers).unwrap(),graph:ContentDigest::of_bytes(graph_bytes()),config:source.manifest().architecture.config.clone(),tokenizer:None,system_state:None,runtime_artifact:None};
        let mut c=standard(); c["source"]=json!(model);c["best_known"]=json!(model);
        for b in c["baselines"].as_array_mut().unwrap(){b["input"]=json!(model);b["backend"]=json!({"id":"fixture.cpu","version":"1","config":profile_id()});}
        c["statistics"]["repeats"]=json!(1);c["seeds"]=json!([17]);c["runtime"]=json!(runtime_profile().unwrap());
        c["limits"]["host_memory_bytes"]=json!(6_442_450_944_u64);
        let old=c["portfolio"]["gyms"].as_object().unwrap().clone();c["portfolio"]["gyms"]=json!({});
        for (i,b) in c["portfolio"]["bindings"].as_array_mut().unwrap().iter_mut().enumerate(){
            for m in b["metrics"].as_array_mut().unwrap(){
                if !m["measurement"].is_null(){m["measurement"]["timer"]=json!({"id":"nf.final.monotonic","version":"1","config":profile_id()});}
                if m["metric"]=="memory"{m["threshold"]["ceiling"]=json!(6_442_450_944_f64);}
            }
            for (j,role) in ["development","selection","holdout"].into_iter().enumerate(){
                let input=if i==1 && j==2 && pass{vec![0.0,0.0]}else{vec![(i*10+j+1) as f64,2.0]};
                let original=vec![input[0]+2.0*input[1],3.0*input[0]+4.0*input[1]];
                let label=if i==1{original}else{original.iter().map(|x|x*0.5).collect()};
                let dataset=Dataset{schema_version:1,role:match j{0=>SplitRole::Development,1=>SplitRole::Selection,_=>SplitRole::Holdout},items:vec![Item{
                    identity:serde_json::from_value(b["splits"][role]["items"][0].clone()).unwrap(),input:store.put(bytes(&input).as_slice()).unwrap().digest,label:store.put(bytes(&label).as_slice()).unwrap().digest}]};
                b["splits"][role]["dataset"]=json!(publish_dataset(&store,&dataset).unwrap());
            }
        }
        for i in 0..5 {
            let mut g=old[c["portfolio"]["bindings"][i]["gym"].as_str().unwrap()].clone();
            g["adapter"]=json!({"id":"nf.final.linear_exact","version":"1","config":profile_id()});
            for k in ["code","scorer","oracle","scoring_protocol","validation","dependencies","input_schema","output_schema"]{g[k]=json!(profile_id());}
            let id=hash(&g);c["portfolio"]["gyms"][id.as_str()]=g;c["portfolio"]["bindings"][i]["gym"]=json!(id);c["portfolio"]["bindings"][i]["validation"]=json!(profile_id());
        }
        c["portfolio"]["bindings"][2]["metrics"][0]["comparator"]=json!("no_op");
        c["portfolio"]["bindings"][4]["metrics"][0]["comparator"]=json!("best_known");refresh(&mut c);
        f["program"]["operations"][0]["operation"]["factor"]=json!(0.5);
        f["program"]["base_model"]["architecture_graph"]=json!(model.graph);f["space"]["base_model"]=f["program"]["base_model"].clone();
        let space=parse_search_space(&bytes(&f["space"])).unwrap();f["program"]["objective"]["search_space"]=json!(space.digest().unwrap());
        f["program"]["verification"]["portfolio"]=c["portfolio_digest"].clone();
        f["policy"]["source"]=f["program"]["base_model"].clone();f["policy"]["search_space"]=json!(space.digest().unwrap());
        c["contract_key"]=f["program"]["objective"]["contract"].clone();c["objective"]["id"]=f["program"]["objective"]["objective"].clone();
        let objective=c["objective"]["id"].clone();for b in c["portfolio"]["bindings"].as_array_mut().unwrap(){b["objective"]=objective.clone();}
        c["intervention"]["scope"]=json!(hash(&f["program"]["effects"]));c["intervention"]["transformation_space"]=json!(space.digest().unwrap());
        c["intervention"]["method_protocol"]=f["program"]["objective"]["experiment_protocol"].clone();
        c["intervention"]["allowed_engines"]=json!([{"id":"nf.native_cpu","version":"1","config":hash(&f["policy"]["backend"])}]);refresh(&mut c);
        f["program"]["verification"]["portfolio"]=c["portfolio_digest"].clone();f["policy"]["verification"]=f["program"]["verification"].clone();
        let observations=source.observations();let inventory=verify_inventory(source.manifest(),&observations,InventoryLimits::default()).unwrap();
        let resolved=resolve_space(&space,&inventory).unwrap();let policy=Validated::<CompilePolicy>::parse(&bytes(&f["policy"])).unwrap();
        let build=|p:&Value|compile(&parse_program(&bytes(p)).unwrap(),SourceSnapshot{manifest:source.manifest(),tensors:&observations,checkpoint:source.checkpoint_bytes(),assets:source.assets()},&resolved,&BTreeMap::new(),&policy,&[]).unwrap();
        let edit=build(&f["program"]);let mut n=f["program"].clone();n["operations"]=json!([]);n["effects"]["reads"]=json!([]);n["effects"]["writes"]=json!([]);
        n["lineage"]["method"]["target_update"]=json!("no_change");n["lineage"]["method"]["changed_parameters"]=json!(0);n["constraints"]["max_changed_parameters"]=json!(0);
        let noop=build(&n);let smoke=Validated::<SmokeSuite>::parse(&bytes(&f["smoke_suite"])).unwrap();let c=Validated::<StudyContract>::parse(&bytes(&c)).unwrap();
        let mut ledger=Ledger::open(&root.join("ledger"),LedgerLimits::default()).unwrap();let id=ledger.freeze(&c).unwrap();let lease=ledger.acquire(&id,"atomic-fixture",300000).unwrap();
        for b in &c.portfolio.bindings{ledger.bind_split_catalog(&lease,&store,&b.id).unwrap();}
        let mut previous=Stage::Frozen;
        for (i,next) in [Stage::Preflight,Stage::Profile,Stage::Baseline,Stage::Design,Stage::Execute,Stage::Evaluate,Stage::Verify].into_iter().enumerate(){
            ledger.advance(&lease,Transition{operation:d(&format!("step{i}")),expected:previous,next,protocol:d("fixture"),evidence:d("fixture"),message:"fixture".into()}).unwrap();previous=next;
        }
        let decision=ledger.run_fixture_final(&lease,&store,&edit,&noop,&smoke,&CancellationToken::default()).unwrap();
        assert_eq!(decision.report().accepted,pass,"{}",String::from_utf8_lossy(decision.bytes()));
        Self{root,ledger,lease,store,edit,smoke,family:c.study_family.clone()}
    }
    pub(crate) fn runtime(&self)->QualifiedTorch{
        let p=Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        // Cargo's library is hard-linked; qualification correctly rejects it.
        let extension=self.root.join("_native.so");
        std::fs::copy(p.join("target/debug/lib_native.so"),&extension).unwrap();
        QualifiedTorch::qualify(true,Path::new("/usr/bin/python3.12"),&p.join(".venv/lib/python3.12/site-packages"),&extension,&self.root).unwrap()
    }
    fn promote_public(&mut self,runtime:&QualifiedTorch)->nf_core::error::Result<nf_experiments::coordinator::promotion::Acceptance>{
        self.ledger.promote_fixture(&self.lease,&self.store,&self.edit,&self.smoke,runtime,d("promotion"),None,&CancellationToken::default())
    }
    pub(crate) fn unchanged(&self){
        assert!(self.ledger.accepted_fixture_head(&self.family).unwrap().is_none());
        assert_eq!(self.ledger.state(self.lease.contract()).unwrap().stage,Stage::Verify);
        let db=rusqlite::Connection::open(self.root.join("ledger/events.sqlite3")).unwrap();
        for table in ["accepted_records","accepted_heads","verified_frontier"]{assert_eq!(db.query_row(&format!("SELECT count(*) FROM {table}"),[],|r|r.get::<_,u64>(0)).unwrap(),0);}
        assert_eq!(self.ledger.events(self.lease.contract(),0,100).unwrap().len(),8);
    }
}
#[test]
fn reload_promotion_native_qualification_and_arbitrary_digest_cannot_authorize() {
    if dedicated_test() {return;}
    use nf_experiments::coordinator::promotion::PromotionScope;
    let f=Fixture::new(true); let runtime=f.runtime();
    let original=f.ledger.contract(f.lease.contract()).unwrap().source.clone();
    let source_bytes=f.store.read_bounded(&original.checkpoint,4096).unwrap();
    assert!(f.ledger.qualify_fixture_promotion(&f.lease,&f.store,PromotionScope::StrictFsync).is_err());
    let qualification=f.ledger.qualify_fixture_promotion(&f.lease,&f.store,PromotionScope::OperationalFixture).unwrap();
    let before=f.ledger.budget_snapshot(f.lease.contract()).unwrap().reservation_count;
    let used=f.store.used_bytes().unwrap();
    let error=f.ledger.approve_fixture_promotion(&f.lease,&qualification,d("explicit-integration-operator-approval"),
        &f.edit,&f.smoke,&runtime,d("public-granted-promotion"),None,&CancellationToken::default()).unwrap_err();
    assert_eq!(error.code,nf_core::error::ErrorCode::PermissionDenied);
    f.unchanged(); // A well-formed digest is not OS-authenticated operator approval.
    let db=rusqlite::Connection::open(f.root.join("ledger/events.sqlite3")).unwrap();
    assert_eq!(db.query_row("SELECT count(*) FROM promotion_records",[],|r|r.get::<_,u64>(0)).unwrap(),0);
    assert_eq!(before,f.ledger.budget_snapshot(f.lease.contract()).unwrap().reservation_count);
    assert_eq!(used,f.store.used_bytes().unwrap());
    let reopened=Ledger::open(&f.root.join("ledger"),LedgerLimits::default()).unwrap();
    assert!(reopened.accepted_fixture_head(&f.family).unwrap().is_none());
    assert_eq!(source_bytes,f.store.read_bounded(&original.checkpoint,4096).unwrap());
}
#[test]
fn reload_promotion_review_block_is_unconditional_and_has_no_side_effects() {
    if dedicated_test() { return; }
    let mut f=Fixture::new(true);let runtime=f.runtime();
    // Even a genuine, durable, passing native fixture decision is insufficient.
    assert!(f.ledger.fixture_final_decision(f.lease.contract()).unwrap().unwrap().report().accepted);
    let contract=f.lease.contract().clone();
    let accounting=|ledger:&Ledger| {
        let mut value=serde_json::to_value(ledger.budget_snapshot(&contract).unwrap()).unwrap();
        value.as_object_mut().unwrap().remove("elapsed_wall_ms"); // observation clock, not a budget write
        bytes(&value)
    };
    let before=accounting(&f.ledger);
    let used=f.store.used_bytes().unwrap();
    let source=f.ledger.contract(f.lease.contract()).unwrap().source.checkpoint.clone();
    let original=f.store.read_bounded(&source,4096).unwrap();
    let db=rusqlite::Connection::open(f.root.join("ledger/events.sqlite3")).unwrap();
    for _ in 0..2 {
        let error=f.promote_public(&runtime).unwrap_err();
        assert_eq!(error.code,nf_core::error::ErrorCode::UnsupportedCapability);
        assert!(error.message.contains("T038"));assert!(error.message.contains("independent verification"));
        f.unchanged();
    }
    assert_eq!(before,accounting(&f.ledger));
    assert_eq!(used,f.store.used_bytes().unwrap());
    assert_eq!(original,f.store.read_bounded(&source,4096).unwrap());
    assert_eq!(db.query_row("SELECT count(*) FROM promotion_records",[],|r|r.get::<_,u64>(0)).unwrap(),0);
    let reopened=Ledger::open(&f.root.join("ledger"),LedgerLimits::default()).unwrap();
    assert!(reopened.accepted_fixture_head(&f.family).unwrap().is_none());
    assert_eq!(reopened.state(f.lease.contract()).unwrap().stage,Stage::Verify);
}
}; }
// Expanded only inside the private library module: native positive behavior,
// never a public-refusal assertion masquerading as an exercised inner gate.
#[allow(unused_macros)]
macro_rules! private_promotion_cases { () => {
// A real Torch qualification raises the controller RSS high-water mark. Never
// run a second T038 guardian in that process: fork/exec rusage can inherit it.
// Each scenario gets a fresh dedicated controller, not a relaxed memory check.
pub(crate) fn isolated_cases(cases:&[&str])->Option<String> {
    if let Ok(case)=std::env::var("NF_PROMOTION_SUBCASE") {
        assert!(cases.contains(&case.as_str()));
        return Some(case);
    }
    let name=std::thread::current().name().unwrap().to_owned();
    for case in cases {
        let status=std::process::Command::new("/usr/bin/prlimit")
            .args(["--as=3221225472:3221225472","--"])
            .arg(std::env::current_exe().unwrap()).args(["--exact",&name,"--test-threads=1"])
            .env("NF_PROMOTION_TEST_LIMITED","1").env("NF_PROMOTION_SUBCASE",case).status().unwrap();
        assert!(status.success(),"dedicated {case} failed: {status}");
    }
    None
}
impl Fixture {
    fn promote(&mut self,runtime:&QualifiedTorch)->nf_core::error::Result<nf_experiments::coordinator::promotion::Acceptance>{
        self.ledger.promote_fixture_after_review(&self.lease,&self.store,&self.edit,&self.smoke,runtime,d("promotion"),None,&CancellationToken::default())
    }
}
#[test]
fn reload_promotion_true_success_reopen_and_idempotence(){
    if dedicated_test() { return; }
    let mut f=Fixture::new(true);let runtime=f.runtime();let a=f.promote(&runtime).unwrap();
    assert_eq!(f.ledger.state(f.lease.contract()).unwrap().stage,Stage::Completed);
    assert_eq!(f.promote(&runtime).unwrap().digest(),a.digest());
    assert_eq!(f.ledger.events(f.lease.contract(),0,100).unwrap().len(),9);
    let reopened=Ledger::open(&f.root.join("ledger"),LedgerLimits::default()).unwrap();
    assert_eq!(reopened.accepted_fixture_head(&f.family).unwrap().unwrap().digest(),a.digest());
    let snap=reopened.budget_snapshot(f.lease.contract()).unwrap();assert_eq!(snap.held.cpu_ms,0);assert_eq!(snap.spent.cpu_ms,60000);
    assert!(f.ledger.promote_fixture_after_review(&f.lease,&f.store,&f.edit,&f.smoke,&runtime,d("different"),None,&CancellationToken::default()).is_err());
    assert_eq!(f.ledger.accepted_fixture_head(&f.family).unwrap().unwrap().digest(),a.digest());
}
#[test]
fn reload_promotion_concurrent_identical_operation_has_one_head_and_event() {
    if dedicated_test() { return; }
    let mut f=Fixture::new(true);let runtime=f.runtime();
    let mut other=Ledger::open(&f.root.join("ledger"),LedgerLimits::default()).unwrap();
    let barrier=std::sync::Barrier::new(2);
    let lease=&f.lease;let store=&f.store;let edit=&f.edit;let smoke=&f.smoke;
    let (a,b)=std::thread::scope(|scope| {
        let worker=scope.spawn(|| {
            barrier.wait();other.promote_fixture_after_review(lease,store,edit,smoke,&runtime,d("promotion"),None,&CancellationToken::default())
        });
        barrier.wait();
        let a=f.ledger.promote_fixture_after_review(lease,store,edit,smoke,&runtime,d("promotion"),None,&CancellationToken::default());
        (a,worker.join().unwrap())
    });
    assert!(a.is_ok() || b.is_ok(),"both publication racers failed: {a:?}, {b:?}");
    let accepted=f.ledger.accepted_fixture_head(&f.family).unwrap().unwrap();
    for result in [a,b].into_iter().flatten(){assert_eq!(result.digest(),accepted.digest());}
    assert_eq!(f.ledger.events(f.lease.contract(),0,100).unwrap().len(),9);
}
#[test]
fn reload_promotion_failed_t038_and_changed_smoke_do_not_publish(){
    let Some(case)=isolated_cases(&["failed","smoke"]) else {return;};
    let mut f=Fixture::new(case!="failed");let runtime=f.runtime();
    if case=="smoke" {
        let mut smoke=serde_json::to_value(&*f.smoke).unwrap();smoke["probes"][0]["input"][0]=json!(123.0);
        f.smoke=Validated::<SmokeSuite>::parse(&bytes(&smoke)).unwrap();
    }
    assert!(f.promote(&runtime).is_err());f.unchanged();
}
#[test]
fn reload_promotion_event_failure_rolls_back_pointer_frontier_and_retry(){
    if dedicated_test() { return; }
    let mut f=Fixture::new(true);let runtime=f.runtime();
    let db=rusqlite::Connection::open(f.root.join("ledger/events.sqlite3")).unwrap();
    db.execute_batch("CREATE TRIGGER reject_promotion_event BEFORE INSERT ON events WHEN NEW.sequence=9 BEGIN SELECT RAISE(ABORT,'simulated commit-boundary failure'); END;").unwrap();
    assert!(f.promote(&runtime).is_err());f.unchanged();
    db.execute_batch("DROP TRIGGER reject_promotion_event;").unwrap();
    assert!(f.promote(&runtime).is_err());f.unchanged(); // no witness is reconstructed from archived JSON
    assert!(db.query_row("SELECT count(*) FROM promotion_records WHERE kind='reload'",[],|r|r.get::<_,u64>(0)).unwrap()>0);
}
#[test]
fn reload_promotion_stale_lease_cancel_and_wrong_predecessor_fail_closed(){
    if dedicated_test() { return; }
    let mut f=Fixture::new(true);let runtime=f.runtime();let cancel=CancellationToken::default();cancel.request();
    assert!(f.ledger.promote_fixture_after_review(&f.lease,&f.store,&f.edit,&f.smoke,&runtime,d("promotion"),None,&cancel).is_err());f.unchanged();
    assert!(f.ledger.promote_fixture_after_review(&f.lease,&f.store,&f.edit,&f.smoke,&runtime,d("promotion"),Some(d("other head")),&CancellationToken::default()).is_err());f.unchanged();
    f.ledger.release(&f.lease).unwrap();let _replacement=f.ledger.acquire(f.lease.contract(),"replacement",300000).unwrap();
    assert!(f.promote(&runtime).is_err());f.unchanged();
}
fn wait_for_reload(root:&Path)->PathBuf {
    for _ in 0..10000 {
        for runtime in std::fs::read_dir(root).unwrap().flatten().filter(|e|e.file_name().to_string_lossy().starts_with("runtime-")) {
            for job in std::fs::read_dir(runtime.path()).unwrap().flatten() {
                if job.path().join("reload-intent.json").exists() && job.path().join("process.json").exists() {return job.path();}
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    panic!("real reload child did not start");
}
#[test]
fn reload_promotion_durable_stop_and_lease_takeover_race_live_child() {
    let Some(case)=isolated_cases(&["stop","takeover"]) else {return;};
    for takeover in [case=="takeover"] {
        let mut f=Fixture::new(true);let runtime=f.runtime();let root=f.root.clone();let lease=f.lease.clone();
        let racer=std::thread::spawn(move || {
            let job=wait_for_reload(&root);
            let mut other=Ledger::open(&root.join("ledger"),LedgerLimits::default()).unwrap();
            if takeover {
                other.release(&lease).unwrap();other.acquire(lease.contract(),"racing-owner",300000).unwrap();
            } else {
                other.request_stop(lease.contract(),d("concurrent-stop"),nf_experiments::coordinator::supervision::StopReason::OwnerStop).unwrap();
            }
            job
        });
        assert!(f.promote(&runtime).is_err());let job=racer.join().unwrap();f.unchanged();
        let process:Value=serde_json::from_slice(&std::fs::read(job.join("process.json")).unwrap()).unwrap();
        assert!(!Path::new(&format!("/proc/{}",process["pid"].as_u64().unwrap())).exists());
    }
}
#[test]
fn reload_promotion_corrupt_export_and_runtime_retain_no_accepted_head() {
    let Some(case)=isolated_cases(&["export","runtime"]) else {return;};
    use std::os::unix::fs::PermissionsExt;
    for corrupt_runtime in [case=="runtime"] {
        let mut f=Fixture::new(true);let runtime=f.runtime();
        let path=if corrupt_runtime {
            std::fs::read_dir(&f.root).unwrap().flatten()
                .find(|e|e.file_name().to_string_lossy().starts_with("runtime-")).unwrap().path().join("_native.abi3.so")
        } else {
            let db=rusqlite::Connection::open(f.root.join("ledger/events.sqlite3")).unwrap();
            let b:Vec<u8>=db.query_row("SELECT body FROM final_fixture_records WHERE kind='dispatch'",[],|r|r.get(0)).unwrap();
            let dispatch:Value=serde_json::from_slice(&b).unwrap();let root=dispatch["candidate_export"]["export_root"].as_str().unwrap();
            f.root.join("store/blobs").join(&root[7..])
        };
        std::fs::set_permissions(&path,std::fs::Permissions::from_mode(0o600)).unwrap();
        std::fs::write(path,b"corrupt").unwrap();assert!(f.promote(&runtime).is_err(),"corrupt_runtime={corrupt_runtime}");f.unchanged();
    }
}
}; }
#[cfg(target_os = "linux")]
promotion_atomic_cases!();
