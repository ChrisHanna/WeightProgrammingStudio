//! Direct CPU primitives. Two deterministic passes validate the whole edit
//! before mutation; there are no per-coordinate allocations or callbacks.
//! These primitives operate on caller-owned candidates, not checkpoint files.

use nf_core::error::{CoreError, ErrorCode, Result};
use nf_core::schemas::DType;
use nf_weight_ir::compilation::VerifiedOperand;
use nf_weight_ir::program::{element_count, NumericalPolicy, OperationKind, TensorRegion};
use serde::Serialize;

fn failure(code: ErrorCode, message: impl Into<String>) -> CoreError {
    CoreError::new(code, "", message)
}

#[derive(Debug, Clone)]
pub(crate) enum Storage {
    F32(Vec<f32>),
    F64(Vec<f64>),
}
#[derive(Debug, Clone)]
pub struct CpuTensor {
    name: String,
    shape: Vec<u64>,
    strides: Vec<u64>,
    storage: Storage,
}
#[derive(Debug, Clone, Copy)]
pub(crate) enum Scalar {
    F32(f32),
    F64(f64),
}
impl Scalar {
    pub(crate) fn as_f64(self) -> f64 {
        match self {
            Self::F32(value) => value as f64,
            Self::F64(value) => value,
        }
    }
    pub(crate) fn same_bits(self, other: Self) -> bool {
        match (self, other) {
            (Self::F32(a), Self::F32(b)) => a.to_bits() == b.to_bits(),
            (Self::F64(a), Self::F64(b)) => a.to_bits() == b.to_bits(),
            _ => false,
        }
    }
}
impl CpuTensor {
    pub fn from_f32(name: impl Into<String>, shape: Vec<u64>, values: Vec<f32>) -> Result<Self> {
        Self::new(name.into(), shape, Storage::F32(values))
    }
    pub fn from_f64(name: impl Into<String>, shape: Vec<u64>, values: Vec<f64>) -> Result<Self> {
        Self::new(name.into(), shape, Storage::F64(values))
    }
    fn new(name: String, shape: Vec<u64>, storage: Storage) -> Result<Self> {
        let count = element_count(&shape, "/shape")?;
        let (actual, finite) = match &storage {
            Storage::F32(v) => (v.len(), v.iter().all(|x| x.is_finite())),
            Storage::F64(v) => (v.len(), v.iter().all(|x| x.is_finite())),
        };
        if count != actual as u64 {
            return Err(failure(
                ErrorCode::ShapeUnsatisfiable,
                "CPU data length differs from shape",
            ));
        }
        if !finite {
            return Err(failure(
                ErrorCode::NonFinite,
                "CPU tensor data must be finite",
            ));
        }
        TensorRegion {
            tensor: name.clone(),
            start: vec![0; shape.len()],
            shape: shape.clone(),
        }
        .validate_at("/tensor")?;
        let mut stride = 1u64;
        let mut strides = vec![0; shape.len()];
        for axis in (0..shape.len()).rev() {
            strides[axis] = stride;
            stride = stride
                .checked_mul(shape[axis])
                .ok_or_else(|| failure(ErrorCode::ShapeUnsatisfiable, "CPU stride overflow"))?;
        }
        Ok(Self {
            name,
            shape,
            strides,
            storage,
        })
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }
    pub fn dtype(&self) -> DType {
        match self.storage {
            Storage::F32(_) => DType::F32,
            Storage::F64(_) => DType::F64,
        }
    }
    pub fn f32_values(&self) -> Option<&[f32]> {
        if let Storage::F32(values) = &self.storage {
            Some(values)
        } else {
            None
        }
    }
    pub fn f64_values(&self) -> Option<&[f64]> {
        if let Storage::F64(values) = &self.storage {
            Some(values)
        } else {
            None
        }
    }
    pub fn to_le_bytes(&self) -> Vec<u8> {
        match &self.storage {
            Storage::F32(values) => values.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Storage::F64(values) => values.iter().flat_map(|x| x.to_le_bytes()).collect(),
        }
    }
    pub(crate) fn validate_region(&self, region: &TensorRegion) -> Result<u64> {
        region.validate_at("/target")?;
        if region.tensor != self.name {
            return Err(failure(
                ErrorCode::SourceMismatch,
                "CPU tensor name differs from target",
            ));
        }
        if region.shape.len() != self.shape.len()
            || region.start.iter().zip(&region.shape).zip(&self.shape).any(
                |((start, width), dimension)| {
                    start.checked_add(*width).is_none_or(|end| end > *dimension)
                },
            )
        {
            return Err(failure(
                ErrorCode::ShapeUnsatisfiable,
                "CPU target slice is outside tensor",
            ));
        }
        element_count(&region.shape, "/target")
    }
    pub(crate) fn index(&self, region: &TensorRegion, linear: u64) -> Result<usize> {
        let mut remaining = linear;
        let mut offset = 0u64;
        for axis in (0..region.shape.len()).rev() {
            let width = region.shape[axis];
            if width == 0 {
                return Err(failure(
                    ErrorCode::ShapeUnsatisfiable,
                    "empty region has no coordinates",
                ));
            }
            let coordinate = region.start[axis]
                .checked_add(remaining % width)
                .ok_or_else(|| failure(ErrorCode::ShapeUnsatisfiable, "coordinate overflow"))?;
            remaining /= width;
            offset = offset
                .checked_add(coordinate.checked_mul(self.strides[axis]).ok_or_else(|| {
                    failure(ErrorCode::ShapeUnsatisfiable, "stride product overflow")
                })?)
                .ok_or_else(|| failure(ErrorCode::ShapeUnsatisfiable, "offset overflow"))?;
        }
        if remaining != 0 {
            return Err(failure(
                ErrorCode::ShapeUnsatisfiable,
                "linear coordinate outside region",
            ));
        }
        usize::try_from(offset)
            .map_err(|_| failure(ErrorCode::ShapeUnsatisfiable, "CPU index does not fit host"))
    }
    pub(crate) fn value(&self, index: usize) -> Result<Scalar> {
        match &self.storage {
            Storage::F32(values) => values.get(index).copied().map(Scalar::F32),
            Storage::F64(values) => values.get(index).copied().map(Scalar::F64),
        }
        .ok_or_else(|| failure(ErrorCode::ShapeUnsatisfiable, "CPU index outside data"))
    }
    pub(crate) fn set(&mut self, index: usize, value: Scalar) {
        match (&mut self.storage, value) {
            (Storage::F32(values), Scalar::F32(value)) => values[index] = value,
            (Storage::F64(values), Scalar::F64(value)) => values[index] = value,
            _ => unreachable!("preflighted matching dtype"),
        }
    }
}

#[derive(Clone, Copy)]
pub enum NumericInput<'a> {
    F32 { shape: &'a [u64], values: &'a [f32] },
    F64 { shape: &'a [u64], values: &'a [f64] },
    Operand(&'a VerifiedOperand),
}
impl NumericInput<'_> {
    pub(crate) fn shape(&self) -> &[u64] {
        match self {
            Self::F32 { shape, .. } | Self::F64 { shape, .. } => shape,
            Self::Operand(value) => &value.descriptor().shape,
        }
    }
    pub(crate) fn dtype(&self) -> DType {
        match self {
            Self::F32 { .. } => DType::F32,
            Self::F64 { .. } => DType::F64,
            Self::Operand(value) => value.descriptor().dtype,
        }
    }
    pub(crate) fn validate(&self, shape: &[u64], dtype: DType) -> Result<()> {
        if self.shape() != shape || self.dtype() != dtype {
            return Err(failure(
                ErrorCode::ShapeUnsatisfiable,
                "input shape/dtype mismatch; implicit casting is forbidden",
            ));
        }
        let actual = match self {
            Self::F32 { values, .. } => values.len() as u64,
            Self::F64 { values, .. } => values.len() as u64,
            Self::Operand(value) => value.descriptor().data_bytes / dtype.byte_width(),
        };
        if actual != element_count(shape, "/input")? {
            return Err(failure(
                ErrorCode::ShapeUnsatisfiable,
                "input element count differs",
            ));
        }
        Ok(())
    }
    pub(crate) fn value(&self, index: usize) -> Result<f64> {
        match self {
            Self::F32 { values, .. } => values
                .get(index)
                .map(|value| *value as f64)
                .ok_or_else(|| failure(ErrorCode::ShapeUnsatisfiable, "input index outside data")),
            Self::F64 { values, .. } => values
                .get(index)
                .copied()
                .ok_or_else(|| failure(ErrorCode::ShapeUnsatisfiable, "input index outside data")),
            Self::Operand(value) => value.value_at(index),
        }
    }
}
#[derive(Clone, Copy)]
pub enum MaskInput<'a> {
    Bytes { shape: &'a [u64], values: &'a [u8] },
    Operand(&'a VerifiedOperand),
}
impl MaskInput<'_> {
    fn data(&self) -> Result<&[u8]> {
        match self {
            Self::Bytes { values, .. } => Ok(values),
            Self::Operand(value) if value.descriptor().dtype == DType::U8 => Ok(value.data()),
            _ => Err(failure(
                ErrorCode::ShapeUnsatisfiable,
                "mask dtype must be u8",
            )),
        }
    }
    fn validate(&self, shape: &[u64]) -> Result<()> {
        let declared = match self {
            Self::Bytes { shape, .. } => *shape,
            Self::Operand(value) => &value.descriptor().shape,
        };
        let data = self.data()?;
        if declared != shape || data.len() as u64 != element_count(shape, "/mask")? {
            return Err(failure(
                ErrorCode::ShapeUnsatisfiable,
                "mask shape/length differs",
            ));
        }
        if data.iter().any(|value| *value > 1) {
            return Err(failure(
                ErrorCode::InvalidValue,
                "mask requires binary zero/one values",
            ));
        }
        Ok(())
    }
}
#[derive(Clone, Copy)]
pub enum BasicEdit<'a> {
    Replace(NumericInput<'a>),
    Add(NumericInput<'a>),
    Scale(f64),
    Mask(MaskInput<'a>),
}
impl BasicEdit<'_> {
    fn kind(&self) -> OperationKind {
        match self {
            Self::Replace(_) => OperationKind::ReplaceSlice,
            Self::Add(_) => OperationKind::AddDelta,
            Self::Scale(_) => OperationKind::Scale,
            Self::Mask(_) => OperationKind::Mask,
        }
    }
    fn validate(&self, shape: &[u64], dtype: DType, numerical: &NumericalPolicy) -> Result<()> {
        match self {
            Self::Replace(input) | Self::Add(input) => input.validate(shape, dtype),
            Self::Scale(factor)
                if !factor.is_finite()
                    || (numerical.accumulation_dtype == DType::F32
                        && !(*factor as f32).is_finite()) =>
            {
                Err(failure(
                    ErrorCode::NonFinite,
                    "scale is not finite in accumulation dtype",
                ))
            }
            Self::Mask(mask) => mask.validate(shape),
            _ => Ok(()),
        }
    }
}

pub(crate) fn validate_numerical(dtype: DType, numerical: &NumericalPolicy) -> Result<()> {
    if !numerical.max_abs_parameter.is_finite()
        || numerical.max_abs_parameter <= 0.0
        || !numerical.backend_abs_tolerance.is_finite()
        || numerical.backend_abs_tolerance < 0.0
    {
        return Err(failure(ErrorCode::InvalidValue, "invalid numerical bounds"));
    }
    if !matches!(numerical.accumulation_dtype, DType::F32 | DType::F64)
        || (dtype == DType::F64 && numerical.accumulation_dtype != DType::F64)
    {
        return Err(failure(
            ErrorCode::UnsupportedCapability,
            "unsupported target/accumulation dtype combination",
        ));
    }
    Ok(())
}
pub(crate) fn rounded(value: f64, dtype: DType, numerical: &NumericalPolicy) -> Result<Scalar> {
    let stored = match dtype {
        DType::F32 => Scalar::F32(value as f32),
        DType::F64 => Scalar::F64(value),
        _ => {
            return Err(failure(
                ErrorCode::UnsupportedCapability,
                "unsupported CPU dtype",
            ))
        }
    };
    if !stored.as_f64().is_finite() {
        return Err(failure(
            ErrorCode::NonFinite,
            "edit creates a non-finite parameter",
        ));
    }
    if stored.as_f64().abs() > numerical.max_abs_parameter {
        return Err(failure(
            ErrorCode::InvalidValue,
            "edit exceeds parameter magnitude bound",
        ));
    }
    Ok(stored)
}
fn evaluate(
    old: Scalar,
    edit: BasicEdit<'_>,
    index: usize,
    dtype: DType,
    numerical: &NumericalPolicy,
) -> Result<Scalar> {
    let x = old.as_f64();
    let rhs = match edit {
        BasicEdit::Replace(value) | BasicEdit::Add(value) => value.value(index)?,
        BasicEdit::Scale(value) => value,
        BasicEdit::Mask(value) => value.data()?[index] as f64,
    };
    if !rhs.is_finite() {
        return Err(failure(
            ErrorCode::NonFinite,
            "input contains a non-finite value",
        ));
    }
    let result = if numerical.accumulation_dtype == DType::F32 {
        let (x, rhs) = (x as f32, rhs as f32);
        match edit {
            BasicEdit::Replace(_) => rhs as f64,
            BasicEdit::Add(_) => (x + rhs) as f64,
            _ => (x * rhs) as f64,
        }
    } else {
        match edit {
            BasicEdit::Replace(_) => rhs,
            BasicEdit::Add(_) => x + rhs,
            _ => x * rhs,
        }
    };
    rounded(result, dtype, numerical)
}

#[derive(Debug, Clone, Serialize)]
pub struct BasicReceipt {
    pub operation: OperationKind,
    pub tensor: String,
    pub dtype: DType,
    pub examined_parameters: u64,
    pub changed_parameters: u64,
    pub cell_visits: u64,
}

pub fn apply_basic(
    tensor: &mut CpuTensor,
    region: &TensorRegion,
    edit: BasicEdit<'_>,
    numerical: &NumericalPolicy,
    max_changed_parameters: u64,
) -> Result<BasicReceipt> {
    let count = tensor.validate_region(region)?;
    let dtype = tensor.dtype();
    validate_numerical(dtype, numerical)?;
    edit.validate(&region.shape, dtype, numerical)?;
    let mut changed = 0u64;
    // Preflight visits every affected value before writing even the first one.
    for linear in 0..count {
        let index = tensor.index(region, linear)?;
        let old = tensor.value(index)?;
        let proposed = evaluate(old, edit, linear as usize, dtype, numerical)?;
        if !old.same_bits(proposed) {
            changed += 1;
        }
        if changed > max_changed_parameters {
            return Err(failure(
                ErrorCode::ParameterBudgetExceeded,
                "actual changed-coordinate limit exceeded",
            ));
        }
    }
    let cell_visits = count
        .checked_mul(if matches!(edit, BasicEdit::Mask(_)) {
            3
        } else {
            2
        })
        .ok_or_else(|| failure(ErrorCode::LimitExceeded, "kernel cell-visit count overflow"))?;
    // Inputs are immutable slices/verified blobs and basic edits read only their
    // own unique coordinate. No callback or allocation can change validation.
    for linear in 0..count {
        let index = tensor.index(region, linear).expect("preflighted index");
        let old = tensor.value(index).expect("preflighted value");
        let value = evaluate(old, edit, linear as usize, dtype, numerical)
            .expect("preflighted stable arithmetic");
        tensor.set(index, value);
    }
    Ok(BasicReceipt {
        operation: edit.kind(),
        tensor: tensor.name.clone(),
        dtype,
        examined_parameters: count,
        changed_parameters: changed,
        cell_visits,
    })
}
