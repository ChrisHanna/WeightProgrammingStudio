//! Sticky cooperative cancellation. No callback or host-language reference.
use nf_core::error::{CoreError, ErrorCode, Result};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
#[derive(Debug, Clone, Default)]
pub struct CancellationToken(Arc<AtomicBool>);
impl CancellationToken {
    pub fn request(&self) {
        self.0.store(true, Ordering::Release);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
    pub fn check(&self) -> Result<()> {
        if self.is_cancelled() {
            Err(CoreError::new(
                ErrorCode::TransactionAborted,
                "",
                "operation cancelled; no candidate committed",
            ))
        } else {
            Ok(())
        }
    }
}
