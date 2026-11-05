#![cfg(target_os = "linux")]
//! Exercise the exact owned service source without compiling unrelated private
//! promotion unit tests. No production trust-root override is exposed.
pub use nf_experiments::{coordinator, intake, sandbox, status_reports, validation};
#[path = "../src/manual_workflow.rs"]
#[allow(dead_code)]
mod manual_workflow;
