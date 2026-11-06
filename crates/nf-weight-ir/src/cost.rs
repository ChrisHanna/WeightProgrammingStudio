//! Shared checked admission bounds for compiler and CPU representation kernels.
//! Work is an algorithmic upper bound, not a measured time or instruction count.

use crate::program::{element_count, MapSide, OperationKind};
use nf_core::error::{CoreError, ErrorCode, Result};
use nf_core::schemas::DType;

fn error(message: &str) -> CoreError {
    CoreError::new(ErrorCode::ShapeUnsatisfiable, "/operations", message)
}
fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| error("representation work/size overflow"))
}
fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .ok_or_else(|| error("representation work/size overflow"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RepresentationCost {
    pub scratch_bytes: u64,
    pub work_upper_bound: u64,
}

pub fn representation_cost(
    kind: OperationKind,
    shape: &[u64],
    dtype: DType,
    accumulation: DType,
    axis: Option<u32>,
    side: Option<MapSide>,
    rank: Option<u64>,
) -> Result<RepresentationCost> {
    let count = element_count(shape, "/operations")?;
    if !matches!(dtype, DType::F16 | DType::Bf16 | DType::F32 | DType::F64)
        || !matches!(accumulation, DType::F32 | DType::F64)
        || (dtype == DType::F64 && accumulation != DType::F64)
    {
        return Err(error("unsupported representation precision"));
    }
    if kind == OperationKind::PermuteAxis {
        let axis = axis.ok_or_else(|| error("explicit permutation axis required"))? as usize;
        let width = *shape
            .get(axis)
            .ok_or_else(|| error("axis is outside target rank"))?;
        return Ok(RepresentationCost {
            scratch_bytes: mul(count, dtype.byte_width())?,
            // Copy, result validation, commit, and allocation-free bijection checks.
            work_upper_bound: add(mul(count, 3)?, mul(width, width)?)?,
        });
    }
    if shape.len() != 2 {
        return Err(error("matrix representation operation requires rank two"));
    }
    let (rows, cols) = (shape[0], shape[1]);
    let accum = accumulation.byte_width();
    let basic_scratch = mul(mul(count, accum)?, 2)?;
    match kind {
        OperationKind::LinearMap => {
            let dimension = match side.ok_or_else(|| error("explicit map side required"))? {
                MapSide::Left => rows,
                MapSide::Right => cols,
            };
            Ok(RepresentationCost {
                scratch_bytes: basic_scratch,
                work_upper_bound: add(
                    add(mul(mul(count, dimension)?, 2)?, mul(count, 3)?)?,
                    mul(dimension, dimension)?,
                )?,
            })
        }
        OperationKind::LowRankUpdate => {
            let rank = rank
                .filter(|rank| *rank > 0)
                .ok_or_else(|| error("positive factor rank required"))?;
            Ok(RepresentationCost {
                scratch_bytes: basic_scratch,
                work_upper_bound: add(
                    add(mul(mul(count, rank)?, 2)?, mul(count, 5)?)?,
                    mul(add(rows, cols)?, rank)?,
                )?,
            })
        }
        OperationKind::SubspaceProjection => {
            let (dimension, other) =
                match side.ok_or_else(|| error("explicit projection side required"))? {
                    MapSide::Left => (rows, cols),
                    MapSide::Right => (cols, rows),
                };
            let rank = rank
                .filter(|rank| *rank > 0 && *rank <= dimension)
                .ok_or_else(|| error("invalid orthonormal basis rank"))?;
            let coefficients = mul(rank, other)?;
            let gram = mul(mul(dimension, mul(rank, rank)?)?, 2)?;
            Ok(RepresentationCost {
                scratch_bytes: add(basic_scratch, mul(coefficients, accum)?)?,
                work_upper_bound: add(
                    add(add(mul(mul(count, rank)?, 4)?, mul(count, 5)?)?, gram)?,
                    mul(dimension, rank)?,
                )?,
            })
        }
        _ => Err(CoreError::new(
            ErrorCode::UnsupportedOperation,
            "/operations",
            "not a representation operation",
        )),
    }
}
