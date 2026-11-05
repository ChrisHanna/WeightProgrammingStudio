//! Bounded direct CPU representation edits with immutable inputs and atomic
//! candidate mutation. Numeric scratch precision/length matches compiler costs.

use crate::basic::{rounded, validate_numerical, CpuTensor, NumericInput};
use nf_core::error::{CoreError, ErrorCode, Result};
use nf_core::schemas::DType;
use nf_weight_ir::cost::{representation_cost, RepresentationCost};
use nf_weight_ir::program::{element_count, MapSide, NumericalPolicy, OperationKind, TensorRegion};
use serde::Serialize;

fn failure(code: ErrorCode, message: impl Into<String>) -> CoreError {
    CoreError::new(code, "", message)
}
fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b).ok_or_else(|| {
        failure(
            ErrorCode::ShapeUnsatisfiable,
            "representation dimension overflow",
        )
    })
}

#[derive(Clone, Copy)]
pub enum RepresentationEdit<'a> {
    PermuteAxis {
        axis: u32,
        permutation: &'a [u64],
    },
    LinearMap {
        matrix: NumericInput<'a>,
        side: MapSide,
    },
    LowRankUpdate {
        left: NumericInput<'a>,
        right: NumericInput<'a>,
        strength: f64,
    },
    Projection {
        basis: NumericInput<'a>,
        side: MapSide,
        strength: f64,
        orthonormal_tolerance: f64,
    },
}
impl RepresentationEdit<'_> {
    fn kind(&self) -> OperationKind {
        match self {
            Self::PermuteAxis { .. } => OperationKind::PermuteAxis,
            Self::LinearMap { .. } => OperationKind::LinearMap,
            Self::LowRankUpdate { .. } => OperationKind::LowRankUpdate,
            Self::Projection { .. } => OperationKind::SubspaceProjection,
        }
    }
}
#[derive(Debug, Clone, Copy)]
pub struct KernelLimits {
    pub max_changed_parameters: u64,
    pub max_scratch_bytes: u64,
    pub max_work_units: u64,
}
#[derive(Debug, Clone, Serialize)]
pub struct RepresentationReceipt {
    pub operation: OperationKind,
    pub tensor: String,
    pub dtype: DType,
    pub examined_parameters: u64,
    pub changed_parameters: u64,
    pub requested_scratch_bytes: u64,
    pub work_upper_bound: u64,
}

struct Arithmetic(DType);
impl Arithmetic {
    fn cast(&self, x: f64) -> f64 {
        if self.0 == DType::F32 {
            (x as f32) as f64
        } else {
            x
        }
    }
    fn add(&self, a: f64, b: f64) -> f64 {
        if self.0 == DType::F32 {
            ((a as f32) + (b as f32)) as f64
        } else {
            a + b
        }
    }
    fn sub(&self, a: f64, b: f64) -> f64 {
        if self.0 == DType::F32 {
            ((a as f32) - (b as f32)) as f64
        } else {
            a - b
        }
    }
    fn mul(&self, a: f64, b: f64) -> f64 {
        if self.0 == DType::F32 {
            ((a as f32) * (b as f32)) as f64
        } else {
            a * b
        }
    }
    fn coefficient(&self, value: f64) -> Result<()> {
        if !value.is_finite() || !self.cast(value).is_finite() {
            return Err(failure(
                ErrorCode::NonFinite,
                "coefficient is not finite in accumulation precision",
            ));
        }
        Ok(())
    }
}

enum Scratch {
    F32(Vec<f32>),
    F64(Vec<f64>),
}
impl Scratch {
    fn zeros(count: u64, dtype: DType) -> Result<Self> {
        let count = usize::try_from(count).map_err(|_| {
            failure(
                ErrorCode::LimitExceeded,
                "scratch element count exceeds host index",
            )
        })?;
        match dtype {
            DType::F32 => {
                let mut data = Vec::new();
                data.try_reserve_exact(count)
                    .map_err(|_| failure(ErrorCode::LimitExceeded, "scratch reservation failed"))?;
                data.resize(count, 0.0);
                Ok(Self::F32(data))
            }
            DType::F64 => {
                let mut data = Vec::new();
                data.try_reserve_exact(count)
                    .map_err(|_| failure(ErrorCode::LimitExceeded, "scratch reservation failed"))?;
                data.resize(count, 0.0);
                Ok(Self::F64(data))
            }
            _ => Err(failure(
                ErrorCode::UnsupportedCapability,
                "scratch precision unsupported",
            )),
        }
    }
    fn get(&self, index: usize) -> f64 {
        match self {
            Self::F32(data) => data[index] as f64,
            Self::F64(data) => data[index],
        }
    }
    fn set(&mut self, index: usize, value: f64) -> Result<()> {
        match self {
            Self::F32(data) => {
                let stored = value as f32;
                if !stored.is_finite() {
                    return Err(failure(ErrorCode::NonFinite, "non-finite intermediate"));
                }
                data[index] = stored;
            }
            Self::F64(data) => {
                if !value.is_finite() {
                    return Err(failure(ErrorCode::NonFinite, "non-finite intermediate"));
                }
                data[index] = value;
            }
        }
        Ok(())
    }
}
fn finite_input(input: NumericInput<'_>) -> Result<()> {
    let count = element_count(input.shape(), "/input")?;
    for index in 0..count {
        if !input.value(index as usize)?.is_finite() {
            return Err(failure(
                ErrorCode::NonFinite,
                "representation operand must be finite",
            ));
        }
    }
    Ok(())
}
fn matrix(shape: &[u64]) -> Result<(u64, u64)> {
    if shape.len() != 2 {
        return Err(failure(
            ErrorCode::ShapeUnsatisfiable,
            "matrix edit requires rank two",
        ));
    }
    Ok((shape[0], shape[1]))
}

fn validate_edit(
    edit: RepresentationEdit<'_>,
    shape: &[u64],
    dtype: DType,
    arithmetic: &Arithmetic,
) -> Result<(Option<u32>, Option<MapSide>, Option<u64>)> {
    match edit {
        RepresentationEdit::PermuteAxis { axis, permutation } => {
            let width = *shape.get(axis as usize).ok_or_else(|| {
                failure(
                    ErrorCode::ShapeUnsatisfiable,
                    "permutation axis outside target",
                )
            })?;
            if permutation.len() as u64 != width {
                return Err(failure(
                    ErrorCode::ShapeUnsatisfiable,
                    "permutation length differs from axis",
                ));
            }
            Ok((Some(axis), None, None))
        }
        RepresentationEdit::LinearMap {
            matrix: input,
            side,
        } => {
            let (rows, cols) = matrix(shape)?;
            let dimension = if side == MapSide::Left { rows } else { cols };
            input.validate(&[dimension, dimension], dtype)?;
            Ok((None, Some(side), None))
        }
        RepresentationEdit::LowRankUpdate {
            left,
            right,
            strength,
        } => {
            arithmetic.coefficient(strength)?;
            let (rows, cols) = matrix(shape)?;
            let (left_rows, rank) = matrix(left.shape())?;
            if left_rows != rows {
                return Err(failure(
                    ErrorCode::ShapeUnsatisfiable,
                    "factor rows differ from target",
                ));
            }
            left.validate(&[rows, rank], dtype)?;
            right.validate(&[rank, cols], dtype)?;
            Ok((None, None, Some(rank)))
        }
        RepresentationEdit::Projection {
            basis,
            side,
            strength,
            orthonormal_tolerance,
        } => {
            arithmetic.coefficient(strength)?;
            if !orthonormal_tolerance.is_finite() || !(0.0..0.01).contains(&orthonormal_tolerance) {
                return Err(failure(
                    ErrorCode::InvalidValue,
                    "invalid orthonormal tolerance",
                ));
            }
            let (rows, cols) = matrix(shape)?;
            let (dimension, rank) = matrix(basis.shape())?;
            if dimension != if side == MapSide::Left { rows } else { cols } {
                return Err(failure(
                    ErrorCode::ShapeUnsatisfiable,
                    "basis dimension differs from target axis",
                ));
            }
            basis.validate(&[dimension, rank], dtype)?;
            Ok((None, Some(side), Some(rank)))
        }
    }
}
fn validate_values(edit: RepresentationEdit<'_>, arithmetic: &Arithmetic) -> Result<()> {
    match edit {
        RepresentationEdit::PermuteAxis { permutation, .. } => {
            for (index, value) in permutation.iter().enumerate() {
                if *value >= permutation.len() as u64 || permutation[..index].contains(value) {
                    return Err(failure(
                        ErrorCode::InvalidValue,
                        "permutation must be bijective",
                    ));
                }
            }
        }
        RepresentationEdit::LinearMap { matrix, .. } => finite_input(matrix)?,
        RepresentationEdit::LowRankUpdate { left, right, .. } => {
            finite_input(left)?;
            finite_input(right)?;
        }
        RepresentationEdit::Projection {
            basis,
            orthonormal_tolerance,
            ..
        } => {
            finite_input(basis)?;
            let (dimension, rank) = matrix(basis.shape())?;
            for a in 0..rank {
                for b in 0..rank {
                    let mut sum = 0.0;
                    for row in 0..dimension {
                        sum = arithmetic.add(
                            sum,
                            arithmetic.mul(
                                basis.value((row * rank + a) as usize)?,
                                basis.value((row * rank + b) as usize)?,
                            ),
                        );
                    }
                    let expected = if a == b { 1.0 } else { 0.0 };
                    if !sum.is_finite() || (sum - expected).abs() > orthonormal_tolerance {
                        return Err(failure(
                            ErrorCode::InvalidValue,
                            "basis is not orthonormal under declared precision/tolerance",
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum OutputMap<'a> {
    Identity,
    Permutation {
        axis_stride: u64,
        width: u64,
        permutation: &'a [u64],
    },
}
impl OutputMap<'_> {
    fn index(self, linear: u64) -> usize {
        match self {
            Self::Identity => linear as usize,
            Self::Permutation {
                axis_stride,
                width,
                permutation,
            } => {
                let digit = (linear / axis_stride) % width;
                (linear - digit * axis_stride + permutation[digit as usize] * axis_stride) as usize
            }
        }
    }
}
fn commit(
    tensor: &mut CpuTensor,
    region: &TensorRegion,
    result: &Scratch,
    mapping: OutputMap<'_>,
    count: u64,
    numerical: &NumericalPolicy,
    limit: u64,
) -> Result<u64> {
    let dtype = tensor.dtype();
    let mut changed = 0;
    for linear in 0..count {
        let index = tensor.index(region, linear)?;
        let proposed = rounded(result.get(mapping.index(linear)), dtype, numerical)?;
        if !tensor.value(index)?.same_bits(proposed) {
            changed += 1;
        }
        if changed > limit {
            return Err(failure(
                ErrorCode::ParameterBudgetExceeded,
                "representation changed-coordinate limit exceeded",
            ));
        }
    }
    for linear in 0..count {
        let index = tensor
            .index(region, linear)
            .expect("preflighted representation index");
        let value = rounded(result.get(mapping.index(linear)), dtype, numerical)
            .expect("preflighted representation result");
        tensor.set(index, value);
    }
    Ok(changed)
}
fn receipt(
    tensor: &CpuTensor,
    edit: RepresentationEdit<'_>,
    count: u64,
    changed: u64,
    cost: RepresentationCost,
) -> RepresentationReceipt {
    RepresentationReceipt {
        operation: edit.kind(),
        tensor: tensor.name().to_owned(),
        dtype: tensor.dtype(),
        examined_parameters: count,
        changed_parameters: changed,
        requested_scratch_bytes: cost.scratch_bytes,
        work_upper_bound: cost.work_upper_bound,
    }
}

pub fn apply_representation(
    tensor: &mut CpuTensor,
    region: &TensorRegion,
    edit: RepresentationEdit<'_>,
    numerical: &NumericalPolicy,
    limits: KernelLimits,
) -> Result<RepresentationReceipt> {
    let count = tensor.validate_region(region)?;
    let dtype = tensor.dtype();
    validate_numerical(dtype, numerical)?;
    let arithmetic = Arithmetic(numerical.accumulation_dtype);
    let (axis, side, rank) = validate_edit(edit, &region.shape, dtype, &arithmetic)?;
    let cost = representation_cost(
        edit.kind(),
        &region.shape,
        dtype,
        numerical.accumulation_dtype,
        axis,
        side,
        rank,
    )?;
    if cost.scratch_bytes > limits.max_scratch_bytes
        || cost.work_upper_bound > limits.max_work_units
    {
        return Err(failure(
            ErrorCode::LimitExceeded,
            "representation scratch/work bound exceeded",
        ));
    }
    validate_values(edit, &arithmetic)?;
    // Avoid huge outer loops for shapes such as [u64::MAX, 0]. Inputs and
    // numerical preconditions still validate even when there are no cells.
    if count == 0 {
        return Ok(receipt(tensor, edit, 0, 0, cost));
    }
    let scratch_dtype = if matches!(edit, RepresentationEdit::PermuteAxis { .. }) {
        dtype
    } else {
        numerical.accumulation_dtype
    };
    let mut original = Scratch::zeros(count, scratch_dtype)?;
    for linear in 0..count {
        original.set(
            linear as usize,
            tensor.value(tensor.index(region, linear)?)?.as_f64(),
        )?;
    }
    if let RepresentationEdit::PermuteAxis { axis, permutation } = edit {
        let axis = axis as usize;
        let axis_stride = element_count(&region.shape[axis + 1..], "/target")?;
        let mapping = OutputMap::Permutation {
            axis_stride,
            width: region.shape[axis],
            permutation,
        };
        let changed = commit(
            tensor,
            region,
            &original,
            mapping,
            count,
            numerical,
            limits.max_changed_parameters,
        )?;
        return Ok(receipt(tensor, edit, count, changed, cost));
    }
    let (rows, cols) = matrix(&region.shape)?;
    let mut result = Scratch::zeros(count, numerical.accumulation_dtype)?;
    match edit {
        RepresentationEdit::LinearMap { matrix, side } => {
            for row in 0..rows {
                for col in 0..cols {
                    let mut sum = 0.0;
                    if side == MapSide::Left {
                        for k in 0..rows {
                            sum = arithmetic.add(
                                sum,
                                arithmetic.mul(
                                    matrix.value((row * rows + k) as usize)?,
                                    original.get((k * cols + col) as usize),
                                ),
                            );
                        }
                    } else {
                        for k in 0..cols {
                            sum = arithmetic.add(
                                sum,
                                arithmetic.mul(
                                    original.get((row * cols + k) as usize),
                                    matrix.value((k * cols + col) as usize)?,
                                ),
                            );
                        }
                    }
                    result.set((row * cols + col) as usize, sum)?;
                }
            }
        }
        RepresentationEdit::LowRankUpdate {
            left,
            right,
            strength,
        } => {
            let rank = rank.expect("validated low-rank dimensions");
            for row in 0..rows {
                for col in 0..cols {
                    let mut delta = 0.0;
                    for k in 0..rank {
                        delta = arithmetic.add(
                            delta,
                            arithmetic.mul(
                                left.value((row * rank + k) as usize)?,
                                right.value((k * cols + col) as usize)?,
                            ),
                        );
                    }
                    result.set(
                        (row * cols + col) as usize,
                        arithmetic.add(
                            original.get((row * cols + col) as usize),
                            arithmetic.mul(strength, delta),
                        ),
                    )?;
                }
            }
        }
        RepresentationEdit::Projection {
            basis,
            side,
            strength,
            ..
        } => {
            let rank = rank.expect("validated projection dimensions");
            let other = if side == MapSide::Left { cols } else { rows };
            let mut coefficients = Scratch::zeros(mul(rank, other)?, numerical.accumulation_dtype)?;
            for k in 0..rank {
                for j in 0..other {
                    let mut sum = 0.0;
                    let dimension = if side == MapSide::Left { rows } else { cols };
                    for i in 0..dimension {
                        let old = if side == MapSide::Left {
                            original.get((i * cols + j) as usize)
                        } else {
                            original.get((j * cols + i) as usize)
                        };
                        sum = arithmetic.add(
                            sum,
                            arithmetic.mul(basis.value((i * rank + k) as usize)?, old),
                        );
                    }
                    coefficients.set((k * other + j) as usize, sum)?;
                }
            }
            for row in 0..rows {
                for col in 0..cols {
                    let mut projected = 0.0;
                    for k in 0..rank {
                        let i = if side == MapSide::Left { row } else { col };
                        let j = if side == MapSide::Left { col } else { row };
                        projected = arithmetic.add(
                            projected,
                            arithmetic.mul(
                                basis.value((i * rank + k) as usize)?,
                                coefficients.get((k * other + j) as usize),
                            ),
                        );
                    }
                    result.set(
                        (row * cols + col) as usize,
                        arithmetic.sub(
                            original.get((row * cols + col) as usize),
                            arithmetic.mul(strength, projected),
                        ),
                    )?;
                }
            }
        }
        RepresentationEdit::PermuteAxis { .. } => {
            unreachable!("permutation handled with one scratch buffer")
        }
    }
    let changed = commit(
        tensor,
        region,
        &result,
        OutputMap::Identity,
        count,
        numerical,
        limits.max_changed_parameters,
    )?;
    Ok(receipt(tensor, edit, count, changed, cost))
}
