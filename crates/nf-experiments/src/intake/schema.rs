//! Bounded declarative schemas, never Python objects or import discovery.
use super::fail;
use nf_core::error::{ErrorCode, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Schema {
    Object {
        properties: BTreeMap<String, Schema>,
        required: BTreeSet<String>,
    },
    Array {
        items: Box<Schema>,
        max_items: u32,
    },
    String {
        max_bytes: u32,
    },
    Number {
        min: f64,
        max: f64,
    },
    Integer {
        min: i64,
        max: i64,
    },
    Boolean {},
}
impl Schema {
    pub fn validate(&self) -> Result<()> {
        let mut count = 0;
        self.walk(0, &mut count)
    }
    pub fn check(&self, value: &serde_json::Value) -> Result<()> {
        self.validate()?;
        let valid = match self {
            Self::Object {
                properties,
                required,
            } => value.as_object().is_some_and(|object| {
                required.iter().all(|key| object.contains_key(key))
                    && object.iter().all(|(key, value)| {
                        properties
                            .get(key)
                            .is_some_and(|schema| schema.check(value).is_ok())
                    })
            }),
            Self::Array { items, max_items } => value.as_array().is_some_and(|array| {
                array.len() <= *max_items as usize && array.iter().all(|v| items.check(v).is_ok())
            }),
            Self::String { max_bytes } => value
                .as_str()
                .is_some_and(|s| s.len() <= *max_bytes as usize),
            Self::Number { min, max } => value
                .as_f64()
                .is_some_and(|n| n.is_finite() && n >= *min && n <= *max),
            Self::Integer { min, max } => value.as_i64().is_some_and(|n| n >= *min && n <= *max),
            Self::Boolean {} => value.is_boolean(),
        };
        if !valid {
            return Err(fail(
                ErrorCode::TypeMismatch,
                "value violates frozen bounded schema",
            ));
        }
        Ok(())
    }
    fn walk(&self, depth: usize, count: &mut usize) -> Result<()> {
        *count += 1;
        if depth > 16 || *count > 256 {
            return Err(fail(
                ErrorCode::LimitExceeded,
                "bounded schema depth/nodes exceeded",
            ));
        }
        match self {
            Self::Object {
                properties,
                required,
            } => {
                if properties.len() > 64 || !required.iter().all(|key| properties.contains_key(key))
                {
                    return Err(fail(
                        ErrorCode::InvalidValue,
                        "invalid required/property schema",
                    ));
                }
                for (key, value) in properties {
                    super::label(key)?;
                    value.walk(depth + 1, count)?;
                }
            }
            Self::Array { items, max_items } => {
                if *max_items == 0 || *max_items > 4096 {
                    return Err(fail(ErrorCode::LimitExceeded, "invalid array bound"));
                }
                items.walk(depth + 1, count)?;
            }
            Self::String { max_bytes } => {
                if *max_bytes == 0 || *max_bytes > 65536 {
                    return Err(fail(ErrorCode::LimitExceeded, "invalid string bound"));
                }
            }
            Self::Number { min, max } => {
                if !min.is_finite() || !max.is_finite() || min > max {
                    return Err(fail(
                        ErrorCode::NonFinite,
                        "finite ordered number bounds required",
                    ));
                }
            }
            Self::Integer { min, max } => {
                if min > max {
                    return Err(fail(
                        ErrorCode::InvalidValue,
                        "ordered integer bounds required",
                    ));
                }
            }
            Self::Boolean {} => (),
        }
        Ok(())
    }
}
