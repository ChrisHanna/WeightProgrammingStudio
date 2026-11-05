//! Safe checkpoint I/O and artifact integrity. No implicit model loading.

#[cfg(target_os = "linux")]
pub mod sharded;
#[cfg(target_os = "linux")]
pub mod store;

#[cfg(target_os = "linux")]
pub mod export;
pub mod serialization;
pub mod tensor_file;

pub use nf_core::VERSION;
