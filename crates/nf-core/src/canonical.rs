//! Bounded canonical JSON v1. See the public identity protocol.

use crate::error::{child_path, CoreError, ErrorCode, Result};
use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Number, Value};
use sha2::{Digest, Sha256};
use std::cell::{Cell, RefCell};
use std::{fmt, str::FromStr};

#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub max_bytes: usize,
    pub max_depth: usize,
    pub max_nodes: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_bytes: 1_048_576,
            max_depth: 64,
            max_nodes: 100_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ContentDigest(String);

impl ContentDigest {
    pub fn of_bytes(bytes: &[u8]) -> Self {
        Self(format!("sha256:{:x}", Sha256::digest(bytes)))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for ContentDigest {
    type Err = CoreError;

    fn from_str(value: &str) -> Result<Self> {
        let hex = value.strip_prefix("sha256:").ok_or_else(|| {
            CoreError::new(ErrorCode::InvalidValue, "", "digest requires sha256 prefix")
        })?;
        if hex.len() != 64
            || !hex
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(CoreError::new(
                ErrorCode::InvalidValue,
                "",
                "digest requires 64 lowercase hexadecimal digits",
            ));
        }
        Ok(Self(value.to_owned()))
    }
}

impl fmt::Display for ContentDigest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Serialize for ContentDigest {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for ContentDigest {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        String::deserialize(deserializer)?
            .parse()
            .map_err(de::Error::custom)
    }
}

#[derive(Debug, Clone)]
pub struct CanonicalDocument {
    value: Value,
    bytes: Vec<u8>,
}

impl CanonicalDocument {
    pub fn parse(input: &[u8], limits: Limits) -> Result<Self> {
        if limits.max_bytes == 0 || limits.max_depth > 64 || limits.max_nodes == 0 {
            return Err(CoreError::new(
                ErrorCode::InvalidValue,
                "",
                "invalid parsing limits",
            ));
        }
        if input.len() > limits.max_bytes {
            return Err(CoreError::new(
                ErrorCode::LimitExceeded,
                "",
                "input byte limit exceeded",
            ));
        }
        let normalized_input = validate_number_tokens(input)?;
        let budget = Budget {
            limits,
            nodes: Cell::new(0),
            failure: RefCell::new(None),
        };
        let mut parser = serde_json::Deserializer::from_slice(&normalized_input);
        let parsed = Seed {
            budget: &budget,
            depth: 0,
            path: String::new(),
        }
        .deserialize(&mut parser)
        .and_then(|value| parser.end().map(|()| value));
        let value = parsed.map_err(|error| {
            budget
                .failure
                .into_inner()
                .unwrap_or_else(|| CoreError::new(ErrorCode::InvalidJson, "", error.to_string()))
        })?;
        let mut bytes = Vec::new();
        write_canonical(&value, &mut bytes);
        Ok(Self { value, bytes })
    }

    pub fn value(&self) -> &Value {
        &self.value
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn digest(&self) -> ContentDigest {
        ContentDigest::of_bytes(&self.bytes)
    }
}

struct Budget {
    limits: Limits,
    nodes: Cell<usize>,
    failure: RefCell<Option<CoreError>>,
}

impl Budget {
    fn fail<E: de::Error>(&self, code: ErrorCode, path: &str, message: &str) -> E {
        let error = CoreError::new(code, path, message);
        *self.failure.borrow_mut() = Some(error.clone());
        E::custom(error)
    }
}

struct Seed<'a> {
    budget: &'a Budget,
    depth: usize,
    path: String,
}

impl<'de> DeserializeSeed<'de> for Seed<'_> {
    type Value = Value;

    fn deserialize<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> std::result::Result<Value, D::Error> {
        if self.depth > self.budget.limits.max_depth
            || self.budget.nodes.get() >= self.budget.limits.max_nodes
        {
            return Err(self.budget.fail(
                ErrorCode::LimitExceeded,
                &self.path,
                "JSON structure limit exceeded",
            ));
        }
        self.budget.nodes.set(self.budget.nodes.get() + 1);
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for Seed<'_> {
    type Value = Value;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("bounded JSON without duplicate keys")
    }

    fn visit_unit<E: de::Error>(self) -> std::result::Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_bool<E: de::Error>(self, value: bool) -> std::result::Result<Value, E> {
        Ok(Value::Bool(value))
    }

    fn visit_i64<E: de::Error>(self, value: i64) -> std::result::Result<Value, E> {
        Ok(Value::Number(value.into()))
    }

    fn visit_u64<E: de::Error>(self, value: u64) -> std::result::Result<Value, E> {
        Ok(Value::Number(value.into()))
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> std::result::Result<Value, E> {
        Number::from_f64(value).map(Value::Number).ok_or_else(|| {
            self.budget
                .fail(ErrorCode::NonFinite, &self.path, "non-finite float")
        })
    }

    fn visit_str<E: de::Error>(self, value: &str) -> std::result::Result<Value, E> {
        Ok(Value::String(value.to_owned()))
    }

    fn visit_string<E: de::Error>(self, value: String) -> std::result::Result<Value, E> {
        Ok(Value::String(value))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> std::result::Result<Value, A::Error> {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element_seed(Seed {
            budget: self.budget,
            depth: self.depth + 1,
            path: child_path(&self.path, &values.len().to_string()),
        })? {
            values.push(value);
        }
        Ok(Value::Array(values))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> std::result::Result<Value, A::Error> {
        let mut values = Map::new();
        while let Some(key) = map.next_key::<String>()? {
            let path = child_path(&self.path, &key);
            if values.contains_key(&key) {
                return Err(self.budget.fail(
                    ErrorCode::DuplicateKey,
                    &path,
                    "duplicate object key",
                ));
            }
            let value = map.next_value_seed(Seed {
                budget: self.budget,
                depth: self.depth + 1,
                path,
            })?;
            values.insert(key, value);
        }
        Ok(Value::Object(values))
    }
}

// serde_json otherwise converts overflowing integer tokens into f64. Check their
// lexical form first; quoted strings and escaped quote characters are skipped.
fn validate_number_tokens(input: &[u8]) -> Result<Vec<u8>> {
    let mut normalized = input.to_vec();
    let mut offset = 0;
    let mut quoted = false;
    while offset < input.len() {
        let byte = input[offset];
        if quoted {
            if byte == b'\\' {
                offset += 2;
                continue;
            }
            if byte == b'"' {
                quoted = false;
            }
        } else if byte == b'"' {
            quoted = true;
        } else if byte == b'-' || byte.is_ascii_digit() {
            let start = offset;
            offset += 1;
            while offset < input.len()
                && (input[offset].is_ascii_digit() || b"-.+eE".contains(&input[offset]))
            {
                offset += 1;
            }
            let token = std::str::from_utf8(&input[start..offset])
                .map_err(|error| CoreError::new(ErrorCode::InvalidJson, "", error.to_string()))?;
            if token == "-0" {
                // Preserve byte positions while canonicalizing integer negative zero.
                normalized[start] = b' ';
            }
            let bad_integer = if token.contains(['.', 'e', 'E']) {
                if token.parse::<f64>().is_ok_and(|number| !number.is_finite()) {
                    return Err(CoreError::new(
                        ErrorCode::NonFinite,
                        "",
                        format!("float outside finite range at byte {start}"),
                    ));
                }
                false
            } else if token.starts_with('-') {
                token.parse::<i64>().is_err()
            } else {
                token.parse::<u64>().is_err()
            };
            if bad_integer {
                return Err(CoreError::new(
                    ErrorCode::IntegerOutOfRange,
                    "",
                    format!("integer outside i64/u64 range at byte {start}"),
                ));
            }
            continue;
        }
        offset += 1;
    }
    Ok(normalized)
}

fn write_canonical(value: &Value, out: &mut Vec<u8>) {
    match value {
        Value::Array(values) => {
            out.push(b'[');
            for (index, child) in values.iter().enumerate() {
                if index != 0 {
                    out.push(b',');
                }
                write_canonical(child, out);
            }
            out.push(b']');
        }
        Value::Object(values) => {
            out.push(b'{');
            let mut keys: Vec<_> = values.keys().collect();
            keys.sort_unstable();
            for (index, key) in keys.into_iter().enumerate() {
                if index != 0 {
                    out.push(b',');
                }
                out.extend(
                    serde_json::to_vec(key).expect("JSON string serialization is infallible"),
                );
                out.push(b':');
                write_canonical(&values[key], out);
            }
            out.push(b'}');
        }
        _ => out.extend(serde_json::to_vec(value).expect("validated finite JSON value")),
    }
}

/// The envelope provides version/field checks; payload types must also reject
/// unknown fields. Generic canonicalization never invents a domain schema.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Record<T> {
    pub schema_version: u32,
    pub kind: String,
    pub payload: T,
}

pub fn decode_record<T: serde::de::DeserializeOwned>(input: &[u8]) -> Result<Record<T>> {
    let document = CanonicalDocument::parse(input, Limits::default())?;
    let record: Record<T> = serde_json::from_slice(document.bytes()).map_err(|error| {
        let message = error.to_string();
        let code = if message.starts_with("unknown field") {
            ErrorCode::UnknownField
        } else {
            ErrorCode::TypeMismatch
        };
        CoreError::new(code, "", message)
    })?;
    if record.schema_version != 1 {
        return Err(CoreError::new(
            ErrorCode::UnsupportedVersion,
            "/schema_version",
            "only record version 1 is supported",
        ));
    }
    Ok(record)
}
