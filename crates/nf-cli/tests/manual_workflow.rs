#![cfg(target_os = "linux")]
#[path = "../../nf-weight-ir/tests/common/mod.rs"]
#[allow(dead_code)]
mod common;
#[path = "../../nf-weight-ir/tests/common/compiler_fixture.rs"]
#[allow(dead_code)]
mod compiler_fixture;
#[path = "../../nf-core/tests/common/study_fixture.rs"]
#[allow(dead_code)]
mod fixture;
#[path = "../../nf-experiments/tests/support/manual_workflow.rs"]
#[allow(dead_code)]
mod support;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::{fs, path::PathBuf, process::Command};
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Root(PathBuf);
impl Root {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "nf-t041-cli-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn run(&self, action: &str, request: &Value) -> std::process::Output {
        let input = self.0.join("request.json");
        fs::write(&input, serde_json::to_vec(request).unwrap()).unwrap();
        Command::new(env!("CARGO_BIN_EXE_nf"))
            .args(["study", action])
            .arg(input)
            .env_clear()
            .env("PATH", "")
            .output()
            .unwrap()
    }
    fn result(&self, action: &str, request: &Value) -> Value {
        let output = self.run(action, request);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        serde_json::from_slice::<Value>(&output.stdout).unwrap()["result"].clone()
    }
}
impl Drop for Root {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
#[test]
fn manual_workflow_cli_positive_plan_validate_freeze_status_report_stop_recovery() {
    let root = Root::new();
    let request = support::bound(support::setup(&root.0));
    let plan = root.result("plan", &request);
    assert_eq!(plan, support::call("plan", &request).unwrap());
    assert_eq!(plan, root.result("validate", &request));
    assert!(!root.0.join("ledger").exists());
    let freeze = json!({"plan":request,"ledger":root.0.join("ledger"),"expected_plan_digest":plan["plan_digest"]});
    let frozen = root.result("freeze", &freeze);
    assert_eq!(frozen, root.result("freeze", &freeze));
    let mut existing = support::existing(&root.0, &frozen);
    let output = Command::new(env!("CARGO_BIN_EXE_nf"))
        .args(["study", "status"])
        .arg(root.0.join("ledger/events.sqlite3"))
        .arg(frozen["contract_digest"].as_str().unwrap())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let status: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(status["status"]["stage"], "frozen");
    assert_eq!(
        root.result("stop", &existing)["request_status"],
        "persisted"
    );
    assert_eq!(root.result("resume", &existing)["execution"], "not_resumed");
    existing["authorization"] = json!(fixture::d("forged"));
    let output = root.run("run", &existing);
    assert_eq!(output.status.code(), Some(3));
    assert_eq!(
        serde_json::from_slice::<Value>(&output.stderr).unwrap()["error"]["code"],
        "PERMISSION_DENIED"
    );
}
#[test]
fn manual_workflow_cli_errors_match_rust_and_do_not_create_stores() {
    let root = Root::new();
    let mut request = support::setup(&root.0);
    request["program"]["objective"]["contract"] = json!(fixture::d("wrong"));
    let expected = support::call("validate", &request).unwrap_err();
    let output = root.run("validate", &request);
    assert_eq!(output.status.code(), Some(5));
    assert_eq!(
        serde_json::from_slice::<Value>(&output.stderr).unwrap()["error"],
        json!(expected)
    );
    request["store"] = json!(root.0.join("never-created"));
    assert!(!root.run("plan", &request).status.success());
    assert!(!root.0.join("never-created").exists());
    assert!(!root.0.join("ledger").exists());
}
