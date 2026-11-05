//! Durable scientific workflow, resource accounting, and evidence lineage.

#[cfg(target_os = "linux")]
pub mod coordinator;

#[cfg(target_os = "linux")]
pub mod intake;
#[cfg(target_os = "linux")]
pub mod sandbox;
#[cfg(target_os = "linux")]
pub mod validation;

pub mod arithmetic;
#[cfg(target_os = "linux")]
pub mod controls;
pub mod engine_registry;
pub mod library_lifecycle;
#[cfg(target_os = "linux")]
pub mod manual_workflow;
#[cfg(target_os = "linux")]
pub mod model_worker;
#[cfg(target_os = "linux")]
pub mod portfolio;
pub mod repair_benchmark;
pub mod scientific_graph;
pub mod statistics;
#[cfg(target_os = "linux")]
pub mod status_reports;

pub use nf_core::VERSION;
