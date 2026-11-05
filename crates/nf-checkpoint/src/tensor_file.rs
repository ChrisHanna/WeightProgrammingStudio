//! Bounded Safetensors metadata and payload inspection; never loads model code.

use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use nf_core::error::{child_path, CoreError, ErrorCode, Result};
use nf_core::schemas::DType;
use safetensors::SafeTensors;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::ops::Range;

#[derive(Debug, Clone, Copy)]
pub struct TensorFileLimits {
    pub max_file_bytes: usize,
    pub max_header_bytes: usize,
    pub max_tensors: usize,
    pub max_rank: usize,
}

impl Default for TensorFileLimits {
    fn default() -> Self {
        Self {
            max_file_bytes: 1_073_741_824,
            max_header_bytes: 1_048_576,
            max_tensors: 10_000,
            max_rank: 32,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorLayout {
    pub shape: Vec<u64>,
    pub dtype: DType,
    /// Absolute range in the file, after the size prefix and JSON header.
    pub data_range: Range<usize>,
}

#[derive(Debug, Clone)]
pub struct HeaderInventory {
    pub tensors: BTreeMap<String, TensorLayout>,
    pub metadata: BTreeMap<String, String>,
    pub header_digest: ContentDigest,
    pub data_start: usize,
    pub file_bytes: usize,
}

impl HeaderInventory {
    /// Safetensors permits zero-sized tensors sharing an offset. Sorting by
    /// both ends places empty tensors before nonempty tensors at that offset.
    pub fn payload_order(&self) -> Vec<(&str, &TensorLayout)> {
        let mut tensors: Vec<_> = self
            .tensors
            .iter()
            .map(|(name, spec)| (name.as_str(), spec))
            .collect();
        tensors.sort_by(|a, b| {
            (a.1.data_range.start, a.1.data_range.end, a.0).cmp(&(
                b.1.data_range.start,
                b.1.data_range.end,
                b.0,
            ))
        });
        tensors
    }
}

#[derive(Debug, Clone)]
pub struct TensorEntry {
    pub shape: Vec<u64>,
    pub dtype: DType,
    pub data_range: Range<usize>,
    pub digest: ContentDigest,
}

#[derive(Debug, Clone)]
pub struct TensorFileInventory {
    pub file_digest: ContentDigest,
    pub header_digest: ContentDigest,
    pub tensors: BTreeMap<String, TensorEntry>,
    /// Uninterpreted strings, never code or instructions.
    pub metadata: BTreeMap<String, String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HeaderTensor {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

fn invalid(path: &str, message: impl Into<String>) -> CoreError {
    CoreError::new(ErrorCode::InvalidValue, path, message)
}

fn dtype(name: &str, path: &str) -> Result<DType> {
    match name {
        "F64" => Ok(DType::F64),
        "F32" => Ok(DType::F32),
        "F16" => Ok(DType::F16),
        "BF16" => Ok(DType::Bf16),
        "I64" => Ok(DType::I64),
        "I32" => Ok(DType::I32),
        "I8" => Ok(DType::I8),
        "U8" => Ok(DType::U8),
        _ => Err(CoreError::new(
            ErrorCode::UnsupportedCapability,
            path,
            "tensor dtype is not supported by this inspector",
        )),
    }
}

/// Validates metadata against the actual file length without mapping or
/// allocating the payload. Its contiguity rules are differentially tested
/// against upstream SafeTensors::deserialize on small complete files.
pub fn inspect_header(
    header: &[u8],
    file_bytes: usize,
    limits: TensorFileLimits,
) -> Result<HeaderInventory> {
    if limits.max_file_bytes < 8
        || limits.max_header_bytes == 0
        || limits.max_tensors == 0
        || limits.max_rank > 32
    {
        return Err(invalid("", "invalid tensor inspection limits"));
    }
    if file_bytes > limits.max_file_bytes
        || header.is_empty()
        || header.len() > limits.max_header_bytes
    {
        return Err(CoreError::new(
            ErrorCode::LimitExceeded,
            "",
            "file or header byte bound exceeded",
        ));
    }
    let data_start = 8usize
        .checked_add(header.len())
        .filter(|start| *start <= file_bytes)
        .ok_or_else(|| invalid("", "truncated metadata"))?;
    let document = CanonicalDocument::parse(
        header,
        Limits {
            max_bytes: limits.max_header_bytes,
            ..Limits::default()
        },
    )?;
    let entries = document
        .value()
        .as_object()
        .ok_or_else(|| invalid("", "header must be an object"))?;
    if entries
        .keys()
        .filter(|name| name.as_str() != "__metadata__")
        .count()
        > limits.max_tensors
    {
        return Err(CoreError::new(
            ErrorCode::LimitExceeded,
            "",
            "tensor count bound exceeded",
        ));
    }
    // Use the upstream format's metadata representation too; our strict pass
    // additionally rejects duplicate/unknown fields and unsupported dtypes.
    let _: safetensors::tensor::Metadata = serde_json::from_slice(header)
        .map_err(|cause| invalid("", format!("invalid Safetensors metadata: {cause}")))?;
    let mut metadata = BTreeMap::new();
    let mut tensors = BTreeMap::new();
    for (name, value) in entries {
        let path = child_path("", name);
        if name == "__metadata__" {
            metadata = serde_json::from_value(value.clone())
                .map_err(|cause| invalid(&path, cause.to_string()))?;
            continue;
        }
        if name.is_empty() {
            return Err(invalid(&path, "tensor name cannot be empty"));
        }
        let spec: HeaderTensor = serde_json::from_value(value.clone())
            .map_err(|cause| invalid(&path, cause.to_string()))?;
        if spec.shape.len() > limits.max_rank {
            return Err(CoreError::new(
                ErrorCode::LimitExceeded,
                child_path(&path, "shape"),
                "tensor rank bound exceeded",
            ));
        }
        let dtype = dtype(&spec.dtype, &child_path(&path, "dtype"))?;
        let expected = spec
            .shape
            .iter()
            .try_fold(dtype.byte_width(), |count, dimension| {
                count
                    .checked_mul(*dimension)
                    .ok_or_else(|| invalid(&path, "tensor shape byte count overflow"))
            })?;
        let [start, end] = spec.data_offsets;
        if end.checked_sub(start) != Some(expected) {
            return Err(invalid(
                &path,
                "tensor data length does not match shape and dtype",
            ));
        }
        let absolute = |offset: u64| -> Result<usize> {
            usize::try_from(offset)
                .ok()
                .and_then(|offset| data_start.checked_add(offset))
                .filter(|offset| *offset <= file_bytes)
                .ok_or_else(|| invalid(&path, "tensor offset lies outside file"))
        };
        tensors.insert(
            name.clone(),
            TensorLayout {
                shape: spec.shape,
                dtype,
                data_range: absolute(start)?..absolute(end)?,
            },
        );
    }
    let inventory = HeaderInventory {
        tensors,
        metadata,
        header_digest: ContentDigest::of_bytes(header),
        data_start,
        file_bytes,
    };
    let mut cursor = data_start;
    for (name, spec) in inventory.payload_order() {
        if spec.data_range.start != cursor {
            return Err(invalid(
                &child_path("", name),
                "noncontiguous or overlapping tensor payload",
            ));
        }
        cursor = spec.data_range.end;
    }
    if cursor != file_bytes {
        return Err(invalid("", "unexpected trailing or missing tensor payload"));
    }
    Ok(inventory)
}

pub fn inspect_safetensors(bytes: &[u8], limits: TensorFileLimits) -> Result<TensorFileInventory> {
    let prefix: [u8; 8] = bytes
        .get(..8)
        .ok_or_else(|| invalid("", "missing header size"))?
        .try_into()
        .unwrap();
    let header_len = usize::try_from(u64::from_le_bytes(prefix))
        .map_err(|_| invalid("", "header size does not fit host index"))?;
    if header_len > limits.max_header_bytes {
        return Err(CoreError::new(
            ErrorCode::LimitExceeded,
            "",
            "header byte bound exceeded",
        ));
    }
    let stop = 8usize
        .checked_add(header_len)
        .ok_or_else(|| invalid("", "header size overflow"))?;
    let header = bytes
        .get(8..stop)
        .ok_or_else(|| invalid("", "truncated metadata"))?;
    let layout = inspect_header(header, bytes.len(), limits)?;
    SafeTensors::deserialize(bytes)
        .map_err(|cause| invalid("", format!("invalid Safetensors payload: {cause}")))?;
    let tensors = layout
        .tensors
        .into_iter()
        .map(|(name, spec)| {
            let digest = ContentDigest::of_bytes(&bytes[spec.data_range.clone()]);
            (
                name,
                TensorEntry {
                    shape: spec.shape,
                    dtype: spec.dtype,
                    data_range: spec.data_range,
                    digest,
                },
            )
        })
        .collect();
    Ok(TensorFileInventory {
        file_digest: ContentDigest::of_bytes(bytes),
        header_digest: layout.header_digest,
        tensors,
        metadata: layout.metadata,
    })
}
