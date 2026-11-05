use nf_core::error::ErrorCode;
use nf_core::schemas::DType;
use nf_surgeon::basic::{apply_basic, BasicEdit, CpuTensor, MaskInput, NumericInput};
use nf_weight_ir::program::{NonFinitePolicy, NumericalPolicy, RoundingMode, TensorRegion};

fn policy(dtype: DType) -> NumericalPolicy {
    NumericalPolicy {
        accumulation_dtype: dtype,
        output_rounding: RoundingMode::NearestTiesEven,
        non_finite: NonFinitePolicy::Reject,
        max_abs_parameter: 100.0,
        backend_abs_tolerance: 0.0,
    }
}
fn region(start: &[u64], shape: &[u64]) -> TensorRegion {
    TensorRegion {
        tensor: "w".to_owned(),
        start: start.to_vec(),
        shape: shape.to_vec(),
    }
}

#[test]
fn basic_ops_replace_counts_actual_changes_not_operand_size() {
    let mut tensor = CpuTensor::from_f64("w", vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let edit = BasicEdit::Replace(NumericInput::F64 {
        shape: &[4],
        values: &[1.0, 9.0, 3.0, 8.0],
    });
    let receipt = apply_basic(
        &mut tensor,
        &region(&[0], &[4]),
        edit,
        &policy(DType::F64),
        2,
    )
    .unwrap();
    assert_eq!(tensor.f64_values().unwrap(), [1.0, 9.0, 3.0, 8.0]);
    assert_eq!(receipt.changed_parameters, 2);
    assert_eq!(receipt.examined_parameters, 4);
    assert_eq!(receipt.cell_visits, 8);
    assert_eq!(receipt.dtype, DType::F64);
    assert_eq!(tensor.name(), "w");
    assert_eq!(tensor.shape(), [4]);
}

#[test]
fn basic_ops_budget_failure_preserves_every_original_byte() {
    let mut tensor = CpuTensor::from_f64("w", vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let before = tensor.to_le_bytes();
    let edit = BasicEdit::Replace(NumericInput::F64 {
        shape: &[4],
        values: &[9.0, 9.0, 9.0, 9.0],
    });
    assert_eq!(
        apply_basic(
            &mut tensor,
            &region(&[0], &[4]),
            edit,
            &policy(DType::F64),
            3
        )
        .unwrap_err()
        .code,
        ErrorCode::ParameterBudgetExceeded
    );
    assert_eq!(tensor.to_le_bytes(), before);
}

#[test]
fn basic_ops_multidimensional_slice_updates_only_declared_cells() {
    let mut tensor =
        CpuTensor::from_f64("w", vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let edit = BasicEdit::Replace(NumericInput::F64 {
        shape: &[2, 1],
        values: &[10.0, 20.0],
    });
    apply_basic(
        &mut tensor,
        &region(&[0, 1], &[2, 1]),
        edit,
        &policy(DType::F64),
        2,
    )
    .unwrap();
    assert_eq!(
        tensor.f64_values().unwrap(),
        [1.0, 10.0, 3.0, 4.0, 20.0, 6.0]
    );
}

#[test]
fn basic_ops_add_scale_mask_execute_on_real_f32_data() {
    let mut tensor = CpuTensor::from_f32("w", vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let all = region(&[0], &[4]);
    apply_basic(
        &mut tensor,
        &all,
        BasicEdit::Add(NumericInput::F32 {
            shape: &[4],
            values: &[1.0, -1.0, 0.0, 2.0],
        }),
        &policy(DType::F32),
        4,
    )
    .unwrap();
    assert_eq!(tensor.f32_values().unwrap(), [2.0, 1.0, 3.0, 6.0]);
    apply_basic(
        &mut tensor,
        &all,
        BasicEdit::Scale(0.5),
        &policy(DType::F32),
        4,
    )
    .unwrap();
    assert_eq!(tensor.f32_values().unwrap(), [1.0, 0.5, 1.5, 3.0]);
    let receipt = apply_basic(
        &mut tensor,
        &all,
        BasicEdit::Mask(MaskInput::Bytes {
            shape: &[4],
            values: &[1, 0, 1, 0],
        }),
        &policy(DType::F32),
        4,
    )
    .unwrap();
    assert_eq!(tensor.f32_values().unwrap(), [1.0, 0.0, 1.5, 0.0]);
    assert_eq!(receipt.changed_parameters, 2);
    assert_eq!(receipt.cell_visits, 12);
}

#[test]
fn basic_ops_f64_add_and_mask_preserve_explicit_precision() {
    let mut tensor = CpuTensor::from_f64("w", vec![2], vec![0.1, 0.2]).unwrap();
    let all = region(&[0], &[2]);
    apply_basic(
        &mut tensor,
        &all,
        BasicEdit::Add(NumericInput::F64 {
            shape: &[2],
            values: &[0.2, 0.1],
        }),
        &policy(DType::F64),
        2,
    )
    .unwrap();
    assert_eq!(tensor.f64_values().unwrap(), [0.1 + 0.2, 0.2 + 0.1]);
    apply_basic(
        &mut tensor,
        &all,
        BasicEdit::Mask(MaskInput::Bytes {
            shape: &[2],
            values: &[1, 0],
        }),
        &policy(DType::F64),
        2,
    )
    .unwrap();
    assert_eq!(tensor.f64_values().unwrap()[1], 0.0);
}

#[test]
fn basic_ops_dtype_shape_name_and_slice_errors_do_not_mutate() {
    let mut tensor = CpuTensor::from_f64("w", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let before = tensor.to_le_bytes();
    let cast = BasicEdit::Replace(NumericInput::F32 {
        shape: &[2, 2],
        values: &[0.0; 4],
    });
    assert_eq!(
        apply_basic(
            &mut tensor,
            &region(&[0, 0], &[2, 2]),
            cast,
            &policy(DType::F64),
            4
        )
        .unwrap_err()
        .code,
        ErrorCode::ShapeUnsatisfiable
    );
    assert!(apply_basic(
        &mut tensor,
        &region(&[1, 0], &[2, 2]),
        BasicEdit::Scale(1.0),
        &policy(DType::F64),
        4
    )
    .is_err());
    let mut wrong = region(&[0, 0], &[2, 2]);
    wrong.tensor = "other".to_owned();
    assert_eq!(
        apply_basic(
            &mut tensor,
            &wrong,
            BasicEdit::Scale(1.0),
            &policy(DType::F64),
            4
        )
        .unwrap_err()
        .code,
        ErrorCode::SourceMismatch
    );
    assert!(apply_basic(
        &mut tensor,
        &region(&[0], &[4]),
        BasicEdit::Scale(1.0),
        &policy(DType::F64),
        4
    )
    .is_err());
    assert_eq!(tensor.to_le_bytes(), before);
}

#[test]
fn basic_ops_nonfinite_input_and_late_overflow_fail_before_first_write() {
    assert_eq!(
        CpuTensor::from_f64("w", vec![1], vec![f64::NAN])
            .unwrap_err()
            .code,
        ErrorCode::NonFinite
    );
    assert_eq!(
        CpuTensor::from_f32("w", vec![1], vec![f32::INFINITY])
            .unwrap_err()
            .code,
        ErrorCode::NonFinite
    );
    let mut tensor = CpuTensor::from_f64("w", vec![2], vec![1.0, f64::MAX]).unwrap();
    let before = tensor.to_le_bytes();
    let mut bounds = policy(DType::F64);
    bounds.max_abs_parameter = f64::MAX;
    let edit = BasicEdit::Add(NumericInput::F64 {
        shape: &[2],
        values: &[1.0, f64::MAX],
    });
    assert_eq!(
        apply_basic(&mut tensor, &region(&[0], &[2]), edit, &bounds, 2)
            .unwrap_err()
            .code,
        ErrorCode::NonFinite
    );
    assert_eq!(tensor.to_le_bytes(), before);
    let mut tensor = CpuTensor::from_f64("w", vec![2], vec![1.0, 2.0]).unwrap();
    let before = tensor.to_le_bytes();
    let edit = BasicEdit::Replace(NumericInput::F64 {
        shape: &[2],
        values: &[9.0, f64::NAN],
    });
    assert_eq!(
        apply_basic(
            &mut tensor,
            &region(&[0], &[2]),
            edit,
            &policy(DType::F64),
            2
        )
        .unwrap_err()
        .code,
        ErrorCode::NonFinite
    );
    assert_eq!(tensor.to_le_bytes(), before);
}

#[test]
fn basic_ops_magnitude_and_invalid_mask_fail_atomically() {
    let mut tensor = CpuTensor::from_f64("w", vec![2], vec![1.0, 60.0]).unwrap();
    let before = tensor.to_le_bytes();
    assert_eq!(
        apply_basic(
            &mut tensor,
            &region(&[0], &[2]),
            BasicEdit::Scale(2.0),
            &policy(DType::F64),
            2
        )
        .unwrap_err()
        .code,
        ErrorCode::InvalidValue
    );
    assert!(apply_basic(
        &mut tensor,
        &region(&[0], &[2]),
        BasicEdit::Mask(MaskInput::Bytes {
            shape: &[2],
            values: &[0, 2]
        }),
        &policy(DType::F64),
        2
    )
    .is_err());
    assert!(apply_basic(
        &mut tensor,
        &region(&[0], &[2]),
        BasicEdit::Mask(MaskInput::Bytes {
            shape: &[2],
            values: &[0]
        }),
        &policy(DType::F64),
        2
    )
    .is_err());
    assert_eq!(tensor.to_le_bytes(), before);
}

#[test]
fn basic_ops_signed_zero_counts_byte_changes_and_noop_is_zero_cost_in_changes() {
    let mut tensor = CpuTensor::from_f64("w", vec![2], vec![0.0, -0.0]).unwrap();
    let receipt = apply_basic(
        &mut tensor,
        &region(&[0], &[2]),
        BasicEdit::Replace(NumericInput::F64 {
            shape: &[2],
            values: &[-0.0, 0.0],
        }),
        &policy(DType::F64),
        2,
    )
    .unwrap();
    assert_eq!(receipt.changed_parameters, 2);
    assert_eq!(
        tensor.f64_values().unwrap()[0].to_bits(),
        (-0.0f64).to_bits()
    );
    let receipt = apply_basic(
        &mut tensor,
        &region(&[0], &[2]),
        BasicEdit::Scale(1.0),
        &policy(DType::F64),
        0,
    )
    .unwrap();
    assert_eq!(receipt.changed_parameters, 0);
}

#[test]
fn basic_ops_scalar_empty_and_last_coordinate_boundaries() {
    let mut scalar = CpuTensor::from_f64("w", vec![], vec![3.0]).unwrap();
    apply_basic(
        &mut scalar,
        &region(&[], &[]),
        BasicEdit::Scale(2.0),
        &policy(DType::F64),
        1,
    )
    .unwrap();
    assert_eq!(scalar.f64_values().unwrap(), [6.0]);
    let mut tensor = CpuTensor::from_f64("w", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let empty = apply_basic(
        &mut tensor,
        &region(&[2, 0], &[0, 2]),
        BasicEdit::Scale(9.0),
        &policy(DType::F64),
        0,
    )
    .unwrap();
    assert_eq!(empty.examined_parameters, 0);
    apply_basic(
        &mut tensor,
        &region(&[1, 1], &[1, 1]),
        BasicEdit::Scale(2.0),
        &policy(DType::F64),
        1,
    )
    .unwrap();
    assert_eq!(tensor.f64_values().unwrap(), [1.0, 2.0, 3.0, 8.0]);
}

#[test]
fn basic_ops_accumulation_precision_and_ties_even_are_explicit() {
    let factor = 1.0 + 2.0f64.powi(-24);
    let mut narrow = CpuTensor::from_f32("w", vec![1], vec![3.0]).unwrap();
    let mut wide = narrow.clone();
    let all = region(&[0], &[1]);
    apply_basic(
        &mut narrow,
        &all,
        BasicEdit::Scale(factor),
        &policy(DType::F32),
        1,
    )
    .unwrap();
    apply_basic(
        &mut wide,
        &all,
        BasicEdit::Scale(factor),
        &policy(DType::F64),
        1,
    )
    .unwrap();
    assert_eq!(narrow.f32_values().unwrap()[0].to_bits(), 3.0f32.to_bits());
    assert_eq!(
        wide.f32_values().unwrap()[0].to_bits(),
        3.0f32.to_bits() + 1
    );
    let mut tie = CpuTensor::from_f32("w", vec![1], vec![16_777_216.0]).unwrap();
    let mut bounds = policy(DType::F32);
    bounds.max_abs_parameter = 20_000_000.0;
    assert_eq!(
        apply_basic(
            &mut tie,
            &all,
            BasicEdit::Add(NumericInput::F32 {
                shape: &[1],
                values: &[1.0]
            }),
            &bounds,
            0
        )
        .unwrap()
        .changed_parameters,
        0
    );
}

#[test]
fn basic_ops_invalid_numeric_policy_and_input_lengths_are_rejected() {
    let mut tensor = CpuTensor::from_f64("w", vec![2], vec![1.0, 2.0]).unwrap();
    assert!(apply_basic(
        &mut tensor,
        &region(&[0], &[2]),
        BasicEdit::Scale(1.0),
        &policy(DType::F32),
        2
    )
    .is_err());
    assert!(apply_basic(
        &mut tensor,
        &region(&[0], &[2]),
        BasicEdit::Add(NumericInput::F64 {
            shape: &[2],
            values: &[1.0]
        }),
        &policy(DType::F64),
        2
    )
    .is_err());
    assert!(apply_basic(
        &mut tensor,
        &region(&[0], &[2]),
        BasicEdit::Scale(f64::INFINITY),
        &policy(DType::F64),
        2
    )
    .is_err());
    assert!(CpuTensor::from_f64("w", vec![2], vec![1.0]).is_err());
    assert_eq!(tensor.f64_values().unwrap(), [1.0, 2.0]);
}
