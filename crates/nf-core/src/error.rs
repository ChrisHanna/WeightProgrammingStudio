//! Stable, serializable errors; paths use RFC 6901 JSON pointers.

use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ErrorCode {
    InvalidJson,
    DuplicateKey,
    NonFinite,
    IntegerOutOfRange,
    LimitExceeded,
    UnsupportedVersion,
    TypeMismatch,
    InvalidValue,
    UnknownField,
    PermissionDenied,
    Io,
    IntegrityMismatch,
    QuotaExceeded,
    UnsupportedCapability,
    UnsupportedOperation,
    SourceMismatch,
    AliasScopeEscape,
    ParameterBudgetExceeded,
    ShapeUnsatisfiable,
    VerificationPolicyWeakened,
    ScopeViolation,
    SmokeFailed,
    TransactionAborted,
    InternalPanic,
    LeaseConflict,
    StaleLease,
    InvalidTransition,
    RevisionConflict,
    IdempotencyConflict,
    BudgetBreach,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoreError {
    pub code: ErrorCode,
    pub path: String,
    pub message: String,
}

impl CoreError {
    pub fn new(code: ErrorCode, path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code,
            path: path.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for CoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let code = serde_json::to_string(&self.code).map_err(|_| fmt::Error)?;
        write!(formatter, "{} at {}: {}", code, self.path, self.message)
    }
}

impl std::error::Error for CoreError {}

pub type Result<T> = std::result::Result<T, CoreError>;

pub fn child_path(parent: &str, component: &str) -> String {
    format!(
        "{}/{}",
        parent,
        component.replace('~', "~0").replace('/', "~1")
    )
}
