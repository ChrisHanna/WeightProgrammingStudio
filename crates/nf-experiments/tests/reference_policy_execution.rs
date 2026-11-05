#![cfg(target_os = "linux")]
//! Real OCI execution, not supplied observation fixtures. No network/downloads.
#[path = "support/reference_durable.rs"]
mod durable_dispatch;
use nf_checkpoint::store::{ArtifactStore, StoreLimits};
use nf_core::canonical::ContentDigest;
use nf_experiments::{
    intake::{self, Draft, Snapshot},
    sandbox::{Limits, QualifiedDocker},
    validation::{
        reference::{self, Budget, Exposure, Policy, Role},
        Plan, Status,
    },
};
use serde_json::json;
use std::{
    os::unix::fs::PermissionsExt,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT: AtomicU64 = AtomicU64::new(0);
const GOOD: &str = "def evaluate(x, seed): return x*2\ndef oracle(x): return x*2\ndef score(x, out, reference): return float(out==reference)\n";
struct Root(PathBuf);
impl Root {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "nf-reference-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).unwrap();
        std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700)).unwrap();
        Self(root)
    }
}
impl Drop for Root {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}
fn snapshot(root: &Root, name: &str, store: &ArtifactStore, code: &str) -> Snapshot {
    let source = root.0.join(name);
    std::fs::create_dir(&source).unwrap();
    std::fs::write(source.join("gym.py"), code).unwrap();
    std::fs::write(source.join("LICENSE"), "Controller fixture only").unwrap();
    std::fs::write(source.join("data.json"), "[]").unwrap();
    let image = nf_experiments::sandbox::IMAGE.split('@').nth(1).unwrap();
    std::fs::write(
        source.join("lock.json"),
        json!({"schema_version":1,"image":image,"packages":{}}).to_string(),
    )
    .unwrap();
    let ep = |symbol| json!({"file":"gym.py","symbol":symbol});
    let schema = json!({"type":"integer","min":0,"max":1000});
    let draft = json!({"schema_version":1,"name":name,"version":"1","usage_rights":"fixture",
        "confidentiality":"confidential","assets":{
            "gym.py":{"kind":"code","expected":null},"LICENSE":{"kind":"license","expected":null},
            "lock.json":{"kind":"dependency_lock","expected":null},"data.json":{"kind":"data","expected":null}},
        "dependency_lock":"lock.json","licenses":["LICENSE"],
        "interface":{"kind":"single_shot","evaluate":ep("evaluate"),"score":ep("score"),"oracle":ep("oracle"),"input":schema,"output":schema},
        "dataset":{"kind":"frozen","files":["data.json"],"task_key":"task","group_key":"group"}});
    intake::snapshot(
        &source,
        store,
        Draft::parse(draft.to_string().as_bytes()).unwrap(),
    )
    .unwrap()
}
const EPISODE: &str = "class Environment:\n def reset(self,inputs,seed):\n  self.x=inputs; self.n=0; self.total=0; return inputs\n def step(self,action):\n  self.n+=1; self.total+=action\n  return dict(observation=self.x+10,reward=1.0,terminated=self.n==2,truncated=False)\n def result(self): return self.total\n def close(self): pass\ndef oracle(x): return x*2\ndef score(x,out,reference): return float(out==reference)\n";
fn episode_snapshot(root: &Root, store: &ArtifactStore) -> Snapshot {
    let initial = snapshot(root, "episode", store, EPISODE);
    let mut draft = serde_json::to_value(&initial.manifest().draft).unwrap();
    let ep = |symbol| json!({"file":"gym.py","symbol":symbol});
    let integer = json!({"type":"integer","min":0,"max":1000});
    draft["interface"] = json!({"kind":"episode","factory":ep("Environment"),
        "score":ep("score"),"oracle":ep("oracle"),"reset":integer,"observation":integer,
        "action":integer,"result":integer,"max_steps":2,"max_episode_ms":2000});
    intake::snapshot(
        &root.0.join("episode"),
        store,
        Draft::parse(draft.to_string().as_bytes()).unwrap(),
    )
    .unwrap()
}
#[test]
fn episode_policies_act_on_observations_in_separate_real_sandboxes() {
    let root = Root::new();
    let store = store(&root);
    let gym = episode_snapshot(&root, &store);
    let identity = gym.id().clone();
    let policy = snapshot(
        &root,
        "reactive",
        &store,
        &GOOD.replace("return x*2", "return x if x < 10 else x-10"),
    );
    let mut plan = plan(&gym);
    for case in &mut plan.cases {
        case.actions = vec![json!(1), json!(1)];
        case.termination = Some(nf_experiments::validation::Termination::Terminated);
    }
    plan.limits.cpu_ms = 1000;
    plan.limits.wall_ms = 300;
    let runtime = runtime(&root);
    let mut weakened = plan.clone();
    weakened.limits.wall_ms = 400;
    assert!(reference::run(
        &gym,
        &store,
        weakened,
        &[Policy {
            snapshot: &policy,
            role: Role::Valid
        }],
        Exposure::Development,
        budget(),
        &runtime
    )
    .is_err());
    let audit = reference::run(
        &gym,
        &store,
        plan,
        &[Policy {
            snapshot: &policy,
            role: Role::Valid,
        }],
        Exposure::Development,
        budget(),
        &runtime,
    )
    .unwrap();
    let report = audit.report();
    assert_eq!(report.status, Status::ProbePassed, "{report:?}");
    assert!(!report.audit_complete);
    assert_eq!(report.snapshot, identity);
    assert_eq!(gym.id(), &identity);
    assert_eq!(report.executions, 6);
    assert_eq!(report.policies[0].correct, 8);
    assert_eq!(report.sandbox_receipts.len(), 6);
    assert!(!report.native_approval);
    assert!(report.cpu_ms <= 16000 && report.wall_ms <= 16000);
}
fn store(root: &Root) -> ArtifactStore {
    ArtifactStore::create(
        &root.0.join("store"),
        StoreLimits {
            total_bytes: 32_000_000,
            max_blob_bytes: 1_048_576,
            max_entries: 1000,
        },
    )
    .unwrap()
}
fn plan(gym: &Snapshot) -> Plan {
    Plan::parse(json!({"schema_version":1,"snapshot":gym.id(),"truth_provenance":"controller independent arithmetic truths",
        "cases":[{"id":"two","input":2,"expected":4,"wrong":[3],"termination":null},
                 {"id":"seven","input":7,"expected":14,"wrong":[13],"termination":null}],
        "seeds":[11,22],"scoring":{"minimum":0.0,"maximum":1.0,"good_min":1.0,"bad_max":0.0},
        "limits":{"wall_ms":2000,"cpu_ms":2000,"output_bytes":32768}}).to_string().as_bytes()).unwrap()
}
fn budget() -> Budget {
    Budget {
        max_executions: 8,
        cpu_ms: 16000,
        wall_ms: 16000,
        storage_bytes: 256 * 1024 * 1024,
    }
}
fn runtime(root: &Root) -> QualifiedDocker {
    QualifiedDocker::qualify(true, &root.0)
        .expect("actual qualified sandbox required; never skip or substitute host Python")
}
#[test]
fn unchanged_gym_runs_good_bad_constant_and_metadata_policies() {
    let root = Root::new();
    let store = store(&root);
    let gym = snapshot(&root, "gym", &store, GOOD);
    let id = gym.id().clone();
    let good = snapshot(&root, "good", &store, GOOD);
    let bad = snapshot(
        &root,
        "bad",
        &store,
        &GOOD.replace("return x*2", "return 1"),
    );
    let constant = snapshot(
        &root,
        "constant",
        &store,
        &GOOD.replace("return x*2", "return 0"),
    );
    let metadata = snapshot(
        &root,
        "metadata",
        &store,
        &GOOD.replace("return x*2", "return seed"),
    );
    // Live source changes cannot change frozen policy or gym execution.
    std::fs::write(
        root.0.join("good/gym.py"),
        "raise AssertionError('live source used')",
    )
    .unwrap();
    let audit = reference::run(
        &gym,
        &store,
        plan(&gym),
        &[
            Policy {
                snapshot: &good,
                role: Role::Valid,
            },
            Policy {
                snapshot: &bad,
                role: Role::Invalid,
            },
            Policy {
                snapshot: &constant,
                role: Role::Constant,
            },
            Policy {
                snapshot: &metadata,
                role: Role::MetadataOnly,
            },
        ],
        Exposure::Development,
        budget(),
        &runtime(&root),
    )
    .unwrap();
    let report = audit.report();
    assert_eq!(report.status, Status::ProbePassed, "{report:?}");
    assert!(!report.audit_complete);
    assert_eq!(report.snapshot, id);
    assert_eq!(gym.id(), &id);
    assert_eq!(report.executions, 8);
    assert_eq!(report.policies[0].correct, 8);
    assert!(report.policies[1..].iter().all(|p| p.passing == 0));
    assert!(!report.native_approval);
    assert_eq!(report.brokered_model_calls, 0);
    assert_eq!(report.sandbox_receipts.len(), 8);
    assert_eq!(report.observations.len(), 8);
    assert_eq!(report.storage.jobs_claimed, 8);
    assert_eq!(report.storage.publications_claimed, 18);
    assert!(report.storage.charged_bytes <= report.budget.storage_bytes);
    assert!(report.storage.charged_bytes > report.reserved_asset_bytes);
    assert!(!report.storage.capacity_guaranteed);
    assert_eq!(report.storage.measured_study_storage_bytes, None);
    assert!(report.storage.observed_workspace_peak_bytes.unwrap() > 0);
    assert!(report.storage.observed_journal_retained_bytes.unwrap() > 0);
    assert_eq!(
        ContentDigest::of_bytes(&store.read(audit.id()).unwrap()),
        *audit.id()
    );
    for id in report.sandbox_receipts.iter().chain(&report.observations) {
        assert!(store.contains_verified(id).unwrap());
    }
    let malicious = snapshot(&root,"malicious-same-gym",&store,
        "def evaluate(x, seed):\n while True: pass\ndef oracle(x): return 0\ndef score(x,out,reference): return 1\n");
    let bounded = reference::run(
        &gym,
        &store,
        plan(&gym),
        &[Policy {
            snapshot: &malicious,
            role: Role::Invalid,
        }],
        Exposure::Development,
        budget(),
        &runtime(&root),
    )
    .unwrap();
    assert_eq!(bounded.report().snapshot, id);
    assert_eq!(bounded.report().status, Status::Inconclusive);
}
#[test]
fn complete_valid_audit_executes_all_six_controls_under_live_deadline() {
    let root = Root::new();
    let store = store(&root);
    let gym = snapshot(&root, "complete-gym", &store, GOOD);
    let valid = snapshot(&root, "complete-valid", &store, GOOD);
    let strong = snapshot(&root, "complete-strong", &store, GOOD);
    let invalid = snapshot(
        &root,
        "complete-invalid",
        &store,
        &GOOD.replace("return x*2", "return -1"),
    );
    let constant = snapshot(
        &root,
        "complete-constant",
        &store,
        &GOOD.replace("return x*2", "return 0"),
    );
    let metadata = snapshot(
        &root,
        "complete-metadata",
        &store,
        &GOOD.replace("return x*2", "return seed"),
    );
    let weak = snapshot(
        &root,
        "complete-weak",
        &store,
        &GOOD.replace("return x*2", "return 4 if x == 2 else 0"),
    );
    let coverage = reference::Coverage {
        required: ["arithmetic".to_owned()].into(),
        case_groups: [
            ("two".to_owned(), "arithmetic".to_owned()),
            ("seven".to_owned(), "arithmetic".to_owned()),
        ]
        .into(),
    };
    let audit = reference::run_cancellable(
        &gym,
        &store,
        plan(&gym),
        &[
            Policy {
                snapshot: &valid,
                role: Role::Valid,
            },
            Policy {
                snapshot: &invalid,
                role: Role::Invalid,
            },
            Policy {
                snapshot: &constant,
                role: Role::Constant,
            },
            Policy {
                snapshot: &metadata,
                role: Role::MetadataOnly,
            },
            Policy {
                snapshot: &weak,
                role: Role::Weak,
            },
            Policy {
                snapshot: &strong,
                role: Role::Strong,
            },
        ],
        Exposure::Development,
        Budget {
            max_executions: 12,
            cpu_ms: 24000,
            wall_ms: 24000,
            storage_bytes: 384 * 1024 * 1024,
        },
        &runtime(&root),
        coverage,
        &std::sync::atomic::AtomicBool::new(false),
    )
    .unwrap();
    let report = audit.report();
    assert_eq!(report.status, Status::Passed, "{report:?}");
    assert!(report.audit_complete);
    assert_eq!(report.executions, 12);
    assert_eq!(report.policies.len(), 6);
    assert!(!report.native_approval);
    assert!(!report.storage.durable_family_accounting);
    for id in &report.sandbox_receipts {
        let receipt: nf_experiments::sandbox::Receipt =
            serde_json::from_slice(&store.read(id).unwrap()).unwrap();
        assert_eq!(receipt.status, "completed");
        assert!(receipt.cleanup_confirmed);
    }
}

#[test]
fn constant_positive_scorer_cannot_reward_bad_policy() {
    let root = Root::new();
    let store = store(&root);
    let gym = snapshot(
        &root,
        "gym",
        &store,
        &GOOD.replace("float(out==reference)", "1.0"),
    );
    let bad = snapshot(
        &root,
        "bad",
        &store,
        &GOOD.replace("return x*2", "return 0"),
    );
    let audit = reference::run(
        &gym,
        &store,
        plan(&gym),
        &[Policy {
            snapshot: &bad,
            role: Role::Invalid,
        }],
        Exposure::Selection,
        budget(),
        &runtime(&root),
    )
    .unwrap();
    assert_eq!(audit.report().snapshot, *gym.id());
    assert_eq!(audit.report().status, Status::Rejected);
    for finding in [
        "scorer_disagreement",
        "invalid_output_rewarded",
        "invalid_policy_not_separated",
    ] {
        assert!(audit.report().findings.contains(finding));
    }
}
#[test]
fn constant_zero_scorer_fails_even_with_only_a_bad_reference() {
    let root = Root::new();
    let store = store(&root);
    let gym = snapshot(
        &root,
        "gym",
        &store,
        &GOOD.replace("float(out==reference)", "0.0"),
    );
    let bad = snapshot(
        &root,
        "bad",
        &store,
        &GOOD.replace("return x*2", "return 0"),
    );
    let audit = reference::run(
        &gym,
        &store,
        plan(&gym),
        &[Policy {
            snapshot: &bad,
            role: Role::Invalid,
        }],
        Exposure::Development,
        budget(),
        &runtime(&root),
    )
    .unwrap();
    assert_eq!(audit.report().status, Status::Rejected);
    assert!(audit.report().findings.contains("known_good_score"));
}
#[test]
fn malicious_policy_is_bounded_and_host_and_gym_remain_unchanged() {
    let root = Root::new();
    let store = store(&root);
    let gym = snapshot(&root, "gym", &store, GOOD);
    let id = gym.id().clone();
    let marker = root.0.join("host-marker");
    let code = format!("from pathlib import Path\ntry: Path({:?}).touch()\nexcept OSError: pass\ndef evaluate(x, seed):\n while True: pass\ndef oracle(x): return 0\ndef score(x,out,reference): return 1\n", marker.to_str().unwrap());
    let malicious = snapshot(&root, "malicious", &store, &code);
    let audit = reference::run(
        &gym,
        &store,
        plan(&gym),
        &[Policy {
            snapshot: &malicious,
            role: Role::Invalid,
        }],
        Exposure::Development,
        budget(),
        &runtime(&root),
    )
    .unwrap();
    assert_eq!(
        audit.report().status,
        Status::Inconclusive,
        "{:?}",
        audit.report()
    );
    assert_eq!(audit.report().executions, 1);
    assert!(!marker.exists());
    assert_eq!(gym.id(), &id);
    assert!(!audit.report().native_approval);
    assert!(audit
        .report()
        .findings
        .contains("bounded_execution_incomplete"));
}
#[test]
fn budget_and_exact_digest_fail_before_execution() {
    let root = Root::new();
    let store = store(&root);
    let gym = snapshot(&root, "gym", &store, GOOD);
    let policy = snapshot(&root, "policy", &store, GOOD);
    let runtime = runtime(&root);
    let before = store.used_bytes().unwrap();
    let entries = || {
        let mut e = std::fs::read_dir(&root.0)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect::<Vec<_>>();
        e.sort();
        e
    };
    let before_entries = entries();
    let mut storage_tiny = budget();
    storage_tiny.storage_bytes = 1;
    assert!(reference::run(
        &gym,
        &store,
        plan(&gym),
        &[Policy {
            snapshot: &policy,
            role: Role::Valid
        }],
        Exposure::Development,
        storage_tiny,
        &runtime
    )
    .is_err());
    assert_eq!(store.used_bytes().unwrap(), before);
    assert_eq!(
        entries(),
        before_entries,
        "storage rejection must not create jobs/files"
    );
    let mut tiny = budget();
    tiny.max_executions = 1;
    assert!(reference::run(
        &gym,
        &store,
        plan(&gym),
        &[Policy {
            snapshot: &policy,
            role: Role::Valid
        }],
        Exposure::Development,
        tiny,
        &runtime
    )
    .is_err());
    let mut stale = plan(&gym);
    stale.snapshot = ContentDigest::of_bytes(b"not gym");
    assert!(reference::run(
        &gym,
        &store,
        stale,
        &[Policy {
            snapshot: &policy,
            role: Role::Valid
        }],
        Exposure::Development,
        budget(),
        &runtime
    )
    .is_err());
    // Qualification remains default-deny and does not create missing roots.
    assert!(QualifiedDocker::qualify(false, &root.0.join("missing")).is_err());
    assert!(!root.0.join("missing").exists());
    let _limits = Limits::default();
}
