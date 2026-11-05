//! Explicit controller-owned CPU isolation test support, never a gym API.
use nf_experiments::sandbox::{Limits, QualifiedDocker};
use serde::Deserialize;
use serde_json::json;
use std::io::{self, Read};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    root: String,
    #[serde(default)]
    enabled: bool,
    #[serde(default)]
    source: String,
    #[serde(default = "input")]
    input: String,
    #[serde(default)]
    limits: Limits,
    #[serde(default)]
    qualify_only: bool,
    #[serde(default)]
    recover: bool,
    cancel_after_ms: Option<u64>,
    crash_after_ms: Option<u64>,
}
fn input() -> String {
    "{}".into()
}
fn main() {
    let mut bytes = Vec::new();
    io::stdin().take(262145).read_to_end(&mut bytes).unwrap();
    if bytes.len() > 262144 {
        std::process::exit(2);
    }
    let request: Request = serde_json::from_slice(&bytes).unwrap();
    let result = (|| {
        let backend = QualifiedDocker::qualify(request.enabled, Path::new(&request.root))?;
        if request.recover {
            return backend.recover_pending().map(|n| json!({"recovered":n}));
        }
        if request.qualify_only {
            return Ok(json!({"qualified":true,"proof":backend.proof()}));
        }
        let cancel = Arc::new(AtomicBool::new(false));
        if let Some(ms) = request.cancel_after_ms {
            let flag = cancel.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(ms));
                flag.store(true, Ordering::Release);
            });
        }
        if let Some(ms) = request.crash_after_ms {
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(ms));
                std::process::exit(86);
            });
        }
        backend
            .execute(
                request.source.as_bytes(),
                request.input.as_bytes(),
                request.limits,
                &cancel,
            )
            .map(|v| serde_json::to_value(v).unwrap())
    })();
    println!(
        "{}",
        match result {
            Ok(v) => json!({"ok":v}),
            Err(e) => json!({"error":e}),
        }
    );
}
