//! Manual study requests use the same Rust wire service as the SDK.
use nf_core::error::{CoreError, Result};

pub(crate) fn dispatch(args: &[String]) -> Option<Result<String>> {
    let [group, operation, input] = args else {
        return None;
    };
    if group != "study"
        || !matches!(
            operation.as_str(),
            "plan"
                | "validate"
                | "runtime-profile"
                | "freeze"
                | "authorize"
                | "revoke"
                | "serve-authority"
                | "run"
                | "execute-supplied"
                | "stop"
                | "resume"
        )
    {
        return None;
    }
    Some((|| {
        let limit = nf_core::canonical::Limits::default().max_bytes;
        let bytes = if operation == "execute-supplied" && input == "-" {
            use std::io::Read;
            let mut bytes = Vec::new();
            std::io::stdin()
                .take(limit as u64 + 1)
                .read_to_end(&mut bytes)
                .map_err(|e| CoreError::new(nf_core::error::ErrorCode::Io, "", e.to_string()))?;
            if bytes.len() > limit {
                return Err(CoreError::new(
                    nf_core::error::ErrorCode::LimitExceeded,
                    "",
                    "stdin request bound",
                ));
            }
            bytes
        } else {
            crate::commands::read_file(input, limit)?
        };
        let result = nf_experiments::manual_workflow::dispatch(operation, &bytes)?;
        serde_json::to_string(&serde_json::json!({"schema_version":1,"ok":true,"result":result}))
            .map(|s| s + "\n")
            .map_err(|e| CoreError::new(nf_core::error::ErrorCode::InvalidValue, "", e.to_string()))
    })())
}
