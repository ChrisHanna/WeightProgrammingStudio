//! Bounded official Safetensors serialization for replay snapshots. No file I/O.

use crate::tensor_file::{inspect_safetensors, TensorFileLimits};
use nf_core::canonical::ContentDigest;
use nf_core::error::{CoreError, ErrorCode, Result};
use nf_core::schemas::DType;
use safetensors::tensor::{serialize, Dtype, TensorView};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy)]
pub struct SerializationLimits {
    pub max_checkpoint_bytes: usize,
    pub max_tensor_bytes: u64,
    pub max_header_bytes: usize,
    pub max_staging_bytes: u64,
}
impl Default for SerializationLimits {
    fn default() -> Self {
        Self {
            max_checkpoint_bytes: 67_108_864,
            max_tensor_bytes: 67_108_864,
            max_header_bytes: 1_048_576,
            max_staging_bytes: 203_423_760,
        }
    }
}
pub struct TensorPayload<'a> {
    pub dtype: DType,
    pub shape: &'a [u64],
    pub bytes: &'a [u8],
}
#[derive(Debug)]
pub struct SerializedCheckpoint {
    pub bytes: Vec<u8>,
    pub digest: ContentDigest,
    pub staging_upper_bound: u64,
}
fn error(code: ErrorCode, message: &str) -> CoreError {
    CoreError::new(code, "", message)
}
fn wire(dtype: DType) -> Dtype {
    match dtype {
        DType::F32 => Dtype::F32,
        DType::F64 => Dtype::F64,
        DType::F16 => Dtype::F16,
        DType::Bf16 => Dtype::BF16,
        DType::I8 => Dtype::I8,
        DType::U8 => Dtype::U8,
        DType::I32 => Dtype::I32,
        DType::I64 => Dtype::I64,
    }
}
pub fn serialize_tensors(
    tensors: &BTreeMap<String, TensorPayload<'_>>,
    limits: SerializationLimits,
) -> Result<SerializedCheckpoint> {
    if limits.max_checkpoint_bytes < 8
        || limits.max_header_bytes == 0
        || limits.max_header_bytes > 1_048_576
        || limits.max_staging_bytes == 0
        || tensors.len() > 1024
    {
        return Err(error(
            ErrorCode::InvalidValue,
            "invalid serialization limits",
        ));
    }
    let mut total = 0u64;
    let mut views = BTreeMap::new();
    for (name, tensor) in tensors {
        if name.is_empty() || name.len() > 512 || name == "__metadata__" || tensor.shape.len() > 32
        {
            return Err(error(
                ErrorCode::InvalidValue,
                "unsupported tensor name/rank",
            ));
        }
        let shape: Vec<usize> = tensor
            .shape
            .iter()
            .map(|value| {
                usize::try_from(*value)
                    .map_err(|_| error(ErrorCode::LimitExceeded, "shape does not fit host index"))
            })
            .collect::<Result<_>>()?;
        let elements = shape.iter().try_fold(1usize, |count, dimension| {
            count
                .checked_mul(*dimension)
                .ok_or_else(|| error(ErrorCode::LimitExceeded, "shape product overflow"))
        })?;
        let expected = elements
            .checked_mul(tensor.dtype.byte_width() as usize)
            .ok_or_else(|| error(ErrorCode::LimitExceeded, "tensor byte product overflow"))?;
        if expected != tensor.bytes.len() {
            return Err(error(
                ErrorCode::IntegrityMismatch,
                "tensor bytes differ from shape/dtype",
            ));
        }
        total = total
            .checked_add(expected as u64)
            .filter(|bytes| *bytes <= limits.max_tensor_bytes)
            .ok_or_else(|| {
                error(
                    ErrorCode::LimitExceeded,
                    "tensor serialization byte bound exceeded",
                )
            })?;
        let view = TensorView::new(wire(tensor.dtype), shape, tensor.bytes)
            .map_err(|_| error(ErrorCode::IntegrityMismatch, "invalid tensor view"))?;
        views.insert(name.as_str(), view);
    }
    if total > limits.max_checkpoint_bytes.saturating_sub(8) as u64 {
        return Err(error(
            ErrorCode::LimitExceeded,
            "tensor payload exceeds checkpoint bound",
        ));
    }
    // Covers encoded input copies, output/Arc transfer and decoded snapshot
    // numeric buffers. It is an admission estimate, not process RSS.
    let staging = total
        .checked_mul(3)
        .and_then(|bytes| bytes.checked_add((limits.max_header_bytes as u64) * 2))
        .and_then(|bytes| bytes.checked_add(16))
        .filter(|bytes| *bytes <= limits.max_staging_bytes)
        .ok_or_else(|| error(ErrorCode::LimitExceeded, "snapshot staging bound exceeded"))?;
    let bytes = serialize(views, &None).map_err(|_| {
        error(
            ErrorCode::IntegrityMismatch,
            "Safetensors serialization failed",
        )
    })?;
    let inventory = inspect_safetensors(
        &bytes,
        TensorFileLimits {
            max_file_bytes: limits.max_checkpoint_bytes,
            max_header_bytes: limits.max_header_bytes,
            max_tensors: 1024,
            ..TensorFileLimits::default()
        },
    )?;
    Ok(SerializedCheckpoint {
        bytes,
        digest: inventory.file_digest,
        staging_upper_bound: staging,
    })
}
