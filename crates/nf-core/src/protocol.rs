//! Version negotiation is shared by CLI and language bindings, not inferred.
use crate::canonical::{CanonicalDocument, Limits};
use crate::error::{CoreError, ErrorCode, Result};
use crate::schemas::{Validate, Validated};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProtocolRequest {
    pub schema_version: u32,
    pub required: BTreeMap<String, u32>,
}
impl Validate for ProtocolRequest {
    fn validate(&self) -> Result<()> {
        if self.schema_version != 1 {
            return Err(CoreError::new(
                ErrorCode::UnsupportedVersion,
                "/schema_version",
                "only negotiation v1 supported",
            ));
        }
        if self.required.is_empty()
            || self.required.len() > 16
            || self.required.keys().any(|name| name.len() > 64)
            || !self.required.contains_key("core_api")
        {
            return Err(CoreError::new(
                ErrorCode::InvalidValue,
                "/required",
                "finite protocol requirements including core_api are required",
            ));
        }
        Ok(())
    }
}
#[derive(Debug, Serialize)]
pub struct ProtocolAgreement {
    pub schema_version: u32,
    pub core_version: &'static str,
    pub agreed: BTreeMap<String, u32>,
    pub supported: BTreeMap<String, u32>,
}
pub fn negotiate(bytes: &[u8]) -> Result<Vec<u8>> {
    let request = Validated::<ProtocolRequest>::parse(bytes)?;
    let supported: BTreeMap<String, u32> = [
        "core_api",
        "canonical_json",
        "model_manifest",
        "weight_ir",
        "opset",
        "search_space",
        "compile_policy",
        "smoke_suite",
    ]
    .into_iter()
    .map(|name| (name.to_owned(), 1))
    .collect();
    for (name, version) in &request.required {
        match supported.get(name) {
            None => {
                return Err(CoreError::new(
                    ErrorCode::UnsupportedCapability,
                    "/required",
                    "unknown requested protocol",
                ))
            }
            Some(known) if known != version => {
                return Err(CoreError::new(
                    ErrorCode::UnsupportedVersion,
                    "/required",
                    "requested protocol version unavailable",
                ))
            }
            _ => (),
        }
    }
    let agreement = ProtocolAgreement {
        schema_version: 1,
        core_version: crate::VERSION,
        agreed: request.required.clone(),
        supported,
    };
    let encoded = serde_json::to_vec(&agreement)
        .map_err(|error| CoreError::new(ErrorCode::InvalidValue, "", error.to_string()))?;
    Ok(CanonicalDocument::parse(&encoded, Limits::default())?
        .bytes()
        .to_vec())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn core_conformance_negotiation_is_exact_not_nearest_version() {
        let result =
            negotiate(br#"{"schema_version":1,"required":{"core_api":1,"weight_ir":1}}"#).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&result).unwrap();
        assert_eq!(value["core_version"], crate::VERSION);
        assert_eq!(
            negotiate(br#"{"schema_version":1,"required":{"core_api":2}}"#)
                .unwrap_err()
                .code,
            ErrorCode::UnsupportedVersion
        );
        assert_eq!(
            negotiate(br#"{"schema_version":1,"required":{"core_api":1,"imaginary":1}}"#)
                .unwrap_err()
                .code,
            ErrorCode::UnsupportedCapability
        );
    }
    #[test]
    fn core_conformance_negotiation_rejects_empty_duplicate_and_unknown_fields() {
        for input in [
            br#"{"schema_version":1,"required":{}}"#.as_slice(),
            br#"{"schema_version":1,"required":{"core_api":1},"execute":true}"#,
            br#"{"schema_version":1,"required":{"core_api":1,"core_api":1}}"#,
        ] {
            assert!(negotiate(input).is_err());
        }
    }
}
