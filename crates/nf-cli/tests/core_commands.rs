#![cfg(target_os = "linux")]
#[path = "../../nf-weight-ir/tests/common/mod.rs"]
#[allow(dead_code)]
mod common;
#[path = "../../nf-weight-ir/tests/common/compiler_fixture.rs"]
#[allow(dead_code)]
mod compiler_fixture;
use nf_checkpoint::export::{publish_export, ExportLimits};
use nf_checkpoint::store::{ArtifactStore, StoreLimits};
use nf_core::schemas::Validated;
use nf_surgeon::transaction::{SmokeProbe, SmokeSuite};
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Root(PathBuf);
impl Root {
    fn new() -> Self {
        let parent = std::env::var_os("KIROCREW_SCRATCH")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        let path = parent.join(format!(
            "cli-core-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn json(&self, name: &str, value: &Value) -> PathBuf {
        let path = self.0.join(name);
        fs::write(&path, serde_json::to_vec(value).unwrap()).unwrap();
        path
    }
}
impl Drop for Root {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn run(args: &[&str], input: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_nf"))
        .args(args)
        .arg(input)
        .env_clear()
        .env("PATH", "")
        .output()
        .unwrap()
}
fn result(output: Output) -> Value {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["ok"], true);
    value["result"].clone()
}
fn setup(root: &Root) -> (compiler_fixture::Fixture, Value) {
    let mut fixture = compiler_fixture::Fixture::new(false);
    let smoke = Validated::new(SmokeSuite {
        schema_version: 1,
        adapter: "nf.linear_fixture.v1".to_owned(),
        probes: vec![SmokeProbe {
            input: vec![1.0, 2.0],
            expected: None,
            absolute_tolerance: 0.0,
            max_abs_output: 1000.0,
        }],
    })
    .unwrap();
    fixture.program["verification"]["smoke_suite"] = json!(smoke.digest().unwrap());
    fixture.policy.verification.smoke_suite = smoke.digest().unwrap();
    let store = ArtifactStore::create(
        &root.0.join("store"),
        StoreLimits {
            total_bytes: 1_000_000,
            max_blob_bytes: 500_000,
            max_entries: 100,
        },
    )
    .unwrap();
    let export = publish_export(
        &store,
        &fixture.manifest,
        Arc::from(fixture.checkpoint.clone()),
        fixture.assets.clone(),
        ExportLimits::default(),
    )
    .unwrap();
    for operand in fixture.operands.values() {
        store.put(operand.as_ref()).unwrap();
    }
    let request = json!({"schema_version":1,"store":root.0.join("store"),"source_export":export,"program":fixture.program,"policy":fixture.policy,"search_space":fixture.space,"smoke_suite":&*smoke});
    (fixture, request)
}
#[test]
fn core_commands_compile_explain_validate_and_inspect_without_python_or_writes() {
    let root = Root::new();
    let (fixture, request) = setup(&root);
    let path = root.json("compile.json", &request);
    let before = fs::read_dir(root.0.join("store/blobs")).unwrap().count();
    let compile = result(run(&["ir", "compile"], &path));
    assert_eq!(
        compile["plan_digest"],
        json!(fixture.run().unwrap().digest())
    );
    assert_eq!(compile["execution"], "not_run");
    assert_eq!(result(run(&["ir", "explain"], &path)), compile);
    let path = root.json("program.json", &fixture.program);
    assert_eq!(result(run(&["ir", "validate"], &path))["valid"], true);
    let path = root.json("space.json", &json!(fixture.space));
    assert_eq!(
        result(run(&["search-space", "validate"], &path))["search_space_digest"],
        json!(fixture.space.digest().unwrap())
    );
    assert_eq!(
        result(run(&["search-space", "show"], &path))["execution"],
        "not_run"
    );
    let path = root.0.join("source.safetensors");
    fs::write(&path, &fixture.checkpoint).unwrap();
    assert_eq!(
        result(run(&["model", "inspect"], &path))["checkpoint_digest"],
        fixture.program["base_model"]["checkpoint"]
    );
    assert_eq!(
        fs::read_dir(root.0.join("store/blobs")).unwrap().count(),
        before
    );
}
#[test]
fn core_commands_apply_replay_and_restore_use_real_weights_and_keep_source() {
    let root = Root::new();
    let (fixture, request) = setup(&root);
    let path = root.json("apply.json", &request);
    let applied = result(run(&["ir", "apply"], &path));
    assert_eq!(applied["export"]["promotion_status"], "not_evaluated");
    assert_eq!(applied["export"]["transaction"]["changed_parameters"], 4);
    assert_ne!(
        applied["export"]["transaction"]["after_parameters"],
        json!(fixture.manifest.parameter_digest)
    );
    let replay_request = json!({"schema_version":1,"store":request["store"],"source_export":request["source_export"],"records":[applied["patch"]],"target":applied["patch"]["id"]});
    let path = root.json("replay.json", &replay_request);
    let replayed = result(run(&["patch", "replay"], &path));
    assert_eq!(replayed["candidate"], applied["export"]["candidate"]);
    assert_eq!(
        replayed["receipt"]["after_parameters"],
        applied["export"]["transaction"]["after_parameters"]
    );
    let restored = json!({"schema_version":1,"store":request["store"],"export_root":applied["export"]["export_root"]});
    let path = root.json("restore.json", &restored);
    let restored = result(run(&["checkpoint", "restore"], &path));
    assert_eq!(
        restored["parameter_digest"],
        applied["export"]["transaction"]["after_parameters"]
    );
    assert_eq!(restored["active_reference"], "unchanged");
    let original =
        json!({"schema_version":1,"store":request["store"],"export_root":request["source_export"]});
    let path = root.json("original.json", &original);
    assert_eq!(
        result(run(&["checkpoint", "restore"], &path))["parameter_digest"],
        json!(fixture.manifest.parameter_digest)
    );
    assert!(!root.0.join("active").exists());
}
#[test]
fn core_commands_errors_are_typed_and_unapproved_studies_never_run() {
    let root = Root::new();
    let missing = root.0.join("absent");
    let output = run(&["study", "run"], &missing);
    // T041 recognizes the command, but an absent explicit request is I/O failure.
    // Valid requests still require authority; see the manual_workflow tests.
    assert_eq!(output.status.code(), Some(4));
    let error: Value = serde_json::from_slice(&output.stderr).unwrap();
    assert_eq!(error["error"]["code"], "IO");
    let output = run(&["ir", "validate"], &missing);
    assert_eq!(output.status.code(), Some(4));
    let path = root.0.join("duplicate.json");
    fs::write(&path, b"{\"ir_version\":1,\"ir_version\":1}").unwrap();
    let output = run(&["ir", "validate"], &path);
    assert_eq!(output.status.code(), Some(2));
    let error: Value = serde_json::from_slice(&output.stderr).unwrap();
    assert_eq!(error["error"]["code"], "DUPLICATE_KEY");
    fs::write(&path, vec![b' '; 1_048_577]).unwrap();
    let output = run(&["ir", "validate"], &path);
    assert_eq!(output.status.code(), Some(6));
    assert!(!root.0.join("store").exists());
}
#[test]
fn core_commands_permissions_bad_source_and_failed_apply_never_publish() {
    let root = Root::new();
    let (_, mut request) = setup(&root);
    let before = fs::read_dir(root.0.join("store/blobs")).unwrap().count();
    request["policy"]["permissions"]["tools"] = json!(["inspect"]);
    let path = root.json("denied.json", &request);
    let output = run(&["ir", "apply"], &path);
    assert_eq!(output.status.code(), Some(3));
    assert_eq!(
        fs::read_dir(root.0.join("store/blobs")).unwrap().count(),
        before
    );
    request["policy"]["permissions"]["tools"] = json!(["inspect", "direct_edit"]);
    request["program"]["base_model"]["parameter_digest"] = json!(common::digest("wrong"));
    let path = root.json("wrong.json", &request);
    let output = run(&["ir", "compile"], &path);
    assert_eq!(output.status.code(), Some(5));
    assert_eq!(
        fs::read_dir(root.0.join("store/blobs")).unwrap().count(),
        before
    );
}
#[test]
fn core_commands_read_only_stores_do_not_create_missing_directories() {
    let root = Root::new();
    let missing = root.0.join("uncreated");
    let request =
        json!({"schema_version":1,"store":missing,"export_root":common::digest("missing")});
    let path = root.json("missing.json", &request);
    let output = run(&["checkpoint", "restore"], &path);
    assert_eq!(output.status.code(), Some(4));
    assert!(!missing.exists());
    let store = ArtifactStore::create(
        &root.0.join("store"),
        StoreLimits {
            total_bytes: 100,
            max_blob_bytes: 100,
            max_entries: 10,
        },
    )
    .unwrap();
    store.put(b"test".as_slice()).unwrap();
    let reader = ArtifactStore::open_read_only(
        &root.0.join("store"),
        StoreLimits {
            total_bytes: 100,
            max_blob_bytes: 100,
            max_entries: 10,
        },
    )
    .unwrap();
    assert!(reader.put(b"denied".as_slice()).is_err());
    assert_eq!(reader.used_bytes().unwrap(), 4);
}
