use nf_core::error::ErrorCode;
use nf_core::schemas::DType;
use nf_surgeon::basic::{CpuTensor, NumericInput};
use nf_surgeon::representation::{apply_representation, KernelLimits, RepresentationEdit};
use nf_weight_ir::cost::representation_cost;
use nf_weight_ir::program::{
    MapSide, NonFinitePolicy, NumericalPolicy, OperationKind, RoundingMode, TensorRegion,
};

fn policy(dtype: DType) -> NumericalPolicy {
    NumericalPolicy {
        accumulation_dtype: dtype,
        output_rounding: RoundingMode::NearestTiesEven,
        non_finite: NonFinitePolicy::Reject,
        max_abs_parameter: 100.0,
        backend_abs_tolerance: 0.0,
    }
}
fn limits() -> KernelLimits {
    KernelLimits {
        max_changed_parameters: 100,
        max_scratch_bytes: 4096,
        max_work_units: 100_000,
    }
}
fn region(start: &[u64], shape: &[u64]) -> TensorRegion {
    TensorRegion {
        tensor: "w".to_owned(),
        start: start.to_vec(),
        shape: shape.to_vec(),
    }
}
fn input<'a>(shape: &'a [u64], values: &'a [f64]) -> NumericInput<'a> {
    NumericInput::F64 { shape, values }
}

#[test]
fn representation_ops_coupled_permutation_preserves_a_known_network_computation() {
    fn evaluate(w: &CpuTensor, v: &CpuTensor) -> f64 {
        let w = w.f64_values().unwrap();
        let v = v.f64_values().unwrap();
        v[0] * (w[0] + 2.0 * w[1]) + v[1] * (w[2] + 2.0 * w[3])
    }
    let mut w = CpuTensor::from_f64("w", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let mut v = CpuTensor::from_f64("v", vec![1, 2], vec![5.0, 6.0]).unwrap();
    let downstream_region = TensorRegion {
        tensor: "v".to_owned(),
        start: vec![0, 0],
        shape: vec![1, 2],
    };
    let original = w.to_le_bytes();
    let downstream = v.to_le_bytes();
    let baseline = evaluate(&w, &v);
    apply_representation(
        &mut w,
        &region(&[0, 0], &[2, 2]),
        RepresentationEdit::PermuteAxis {
            axis: 0,
            permutation: &[1, 0],
        },
        &policy(DType::F64),
        limits(),
    )
    .unwrap();
    assert_ne!(evaluate(&w, &v), baseline);
    apply_representation(
        &mut v,
        &downstream_region,
        RepresentationEdit::PermuteAxis {
            axis: 1,
            permutation: &[1, 0],
        },
        &policy(DType::F64),
        limits(),
    )
    .unwrap();
    assert_eq!(evaluate(&w, &v), baseline);
    apply_representation(
        &mut w,
        &region(&[0, 0], &[2, 2]),
        RepresentationEdit::PermuteAxis {
            axis: 0,
            permutation: &[1, 0],
        },
        &policy(DType::F64),
        limits(),
    )
    .unwrap();
    apply_representation(
        &mut v,
        &downstream_region,
        RepresentationEdit::PermuteAxis {
            axis: 1,
            permutation: &[1, 0],
        },
        &policy(DType::F64),
        limits(),
    )
    .unwrap();
    assert_eq!(w.to_le_bytes(), original);
    assert_eq!(v.to_le_bytes(), downstream);
}

#[test]
fn representation_ops_permutation_supports_rank_three_and_signed_zero() {
    let mut tensor =
        CpuTensor::from_f64("w", vec![2, 2, 2], (0..8).map(|x| x as f64).collect()).unwrap();
    let receipt = apply_representation(
        &mut tensor,
        &region(&[0, 0, 0], &[2, 2, 2]),
        RepresentationEdit::PermuteAxis {
            axis: 1,
            permutation: &[1, 0],
        },
        &policy(DType::F64),
        limits(),
    )
    .unwrap();
    assert_eq!(
        tensor.f64_values().unwrap(),
        [2.0, 3.0, 0.0, 1.0, 6.0, 7.0, 4.0, 5.0]
    );
    assert_eq!(receipt.changed_parameters, 8);
    assert_eq!(receipt.requested_scratch_bytes, 64);
    let mut zeros = CpuTensor::from_f32("w", vec![2], vec![-0.0, 0.0]).unwrap();
    let receipt = apply_representation(
        &mut zeros,
        &region(&[0], &[2]),
        RepresentationEdit::PermuteAxis {
            axis: 0,
            permutation: &[1, 0],
        },
        &policy(DType::F32),
        limits(),
    )
    .unwrap();
    assert_eq!(receipt.changed_parameters, 2);
    assert_eq!(receipt.requested_scratch_bytes, 8);
    assert_eq!(
        zeros.f32_values().unwrap()[1].to_bits(),
        (-0.0f32).to_bits()
    );
}

#[test]
fn representation_ops_left_rotation_and_inverse_restore_values() {
    let mut tensor = CpuTensor::from_f64("w", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let before = tensor.to_le_bytes();
    apply_representation(
        &mut tensor,
        &region(&[0, 0], &[2, 2]),
        RepresentationEdit::LinearMap {
            matrix: input(&[2, 2], &[0.0, -1.0, 1.0, 0.0]),
            side: MapSide::Left,
        },
        &policy(DType::F64),
        limits(),
    )
    .unwrap();
    assert_eq!(tensor.f64_values().unwrap(), [-3.0, -4.0, 1.0, 2.0]);
    apply_representation(
        &mut tensor,
        &region(&[0, 0], &[2, 2]),
        RepresentationEdit::LinearMap {
            matrix: input(&[2, 2], &[0.0, 1.0, -1.0, 0.0]),
            side: MapSide::Left,
        },
        &policy(DType::F64),
        limits(),
    )
    .unwrap();
    assert_eq!(tensor.to_le_bytes(), before);
}

#[test]
fn representation_ops_right_map_obeys_partial_target_slice() {
    let mut tensor =
        CpuTensor::from_f64("w", vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let matrix = input(&[3, 3], &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
    apply_representation(
        &mut tensor,
        &region(&[1, 0], &[1, 3]),
        RepresentationEdit::LinearMap {
            matrix,
            side: MapSide::Right,
        },
        &policy(DType::F64),
        limits(),
    )
    .unwrap();
    assert_eq!(tensor.f64_values().unwrap(), [1.0, 2.0, 3.0, 6.0, 4.0, 5.0]);
}

#[test]
fn representation_ops_low_rank_update_matches_an_independent_outer_product() {
    let mut tensor =
        CpuTensor::from_f64("w", vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let receipt = apply_representation(
        &mut tensor,
        &region(&[0, 0], &[2, 3]),
        RepresentationEdit::LowRankUpdate {
            left: input(&[2, 1], &[1.0, 2.0]),
            right: input(&[1, 3], &[3.0, 4.0, 5.0]),
            strength: 0.5,
        },
        &policy(DType::F64),
        limits(),
    )
    .unwrap();
    assert_eq!(
        tensor.f64_values().unwrap(),
        [2.5, 4.0, 5.5, 7.0, 9.0, 11.0]
    );
    assert_eq!(receipt.changed_parameters, 6);
    assert_eq!(receipt.requested_scratch_bytes, 96);
    let expected = representation_cost(
        OperationKind::LowRankUpdate,
        &[2, 3],
        DType::F64,
        DType::F64,
        None,
        None,
        Some(1),
    )
    .unwrap();
    assert_eq!(receipt.work_upper_bound, expected.work_upper_bound);
}

#[test]
fn representation_ops_projection_matches_left_and_right_reference_values() {
    let u = 1.0 / 2.0f64.sqrt();
    for (side, expected) in [
        (MapSide::Left, [-1.0, -1.0, 1.0, 1.0]),
        (MapSide::Right, [-0.5, 0.5, -0.5, 0.5]),
    ] {
        let mut tensor = CpuTensor::from_f64("w", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        apply_representation(
            &mut tensor,
            &region(&[0, 0], &[2, 2]),
            RepresentationEdit::Projection {
                basis: input(&[2, 1], &[u, u]),
                side,
                strength: 1.0,
                orthonormal_tolerance: 1e-12,
            },
            &policy(DType::F64),
            limits(),
        )
        .unwrap();
        for (actual, expected) in tensor.f64_values().unwrap().iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }
}

#[test]
fn representation_ops_f32_rotation_preserves_declared_dtype() {
    let mut tensor = CpuTensor::from_f32("w", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let edit = RepresentationEdit::LinearMap {
        matrix: NumericInput::F32 {
            shape: &[2, 2],
            values: &[0.0, -1.0, 1.0, 0.0],
        },
        side: MapSide::Left,
    };
    let receipt = apply_representation(
        &mut tensor,
        &region(&[0, 0], &[2, 2]),
        edit,
        &policy(DType::F32),
        limits(),
    )
    .unwrap();
    assert_eq!(tensor.f32_values().unwrap(), [-3.0, -4.0, 1.0, 2.0]);
    assert_eq!(receipt.dtype, DType::F32);
    assert_eq!(receipt.requested_scratch_bytes, 32);
}

#[test]
fn representation_ops_finite_magnitude_failure_is_atomic() {
    let mut tensor = CpuTensor::from_f64("w", vec![2, 2], vec![1.0, 2.0, 60.0, 1.0]).unwrap();
    let before = tensor.to_le_bytes();
    let edit = RepresentationEdit::LinearMap {
        matrix: input(&[2, 2], &[2.0, 0.0, 0.0, 2.0]),
        side: MapSide::Left,
    };
    assert_eq!(
        apply_representation(
            &mut tensor,
            &region(&[0, 0], &[2, 2]),
            edit,
            &policy(DType::F64),
            limits()
        )
        .unwrap_err()
        .code,
        ErrorCode::InvalidValue
    );
    assert_eq!(tensor.to_le_bytes(), before);
}

#[test]
fn representation_ops_f32_projection_and_low_rank_are_real_dtype_preserving_edits() {
    let mut tensor = CpuTensor::from_f32("w", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let receipt = apply_representation(
        &mut tensor,
        &region(&[0, 0], &[2, 2]),
        RepresentationEdit::Projection {
            basis: NumericInput::F32 {
                shape: &[2, 1],
                values: &[1.0, 0.0],
            },
            side: MapSide::Left,
            strength: 1.0,
            orthonormal_tolerance: 0.0,
        },
        &policy(DType::F32),
        limits(),
    )
    .unwrap();
    assert_eq!(tensor.f32_values().unwrap(), [0.0, 0.0, 3.0, 4.0]);
    assert_eq!(receipt.changed_parameters, 2);
    assert_eq!(receipt.requested_scratch_bytes, 40);
    apply_representation(
        &mut tensor,
        &region(&[0, 0], &[2, 2]),
        RepresentationEdit::LowRankUpdate {
            left: NumericInput::F32 {
                shape: &[2, 1],
                values: &[1.0, 1.0],
            },
            right: NumericInput::F32 {
                shape: &[1, 2],
                values: &[1.0, 2.0],
            },
            strength: 0.5,
        },
        &policy(DType::F64),
        limits(),
    )
    .unwrap();
    assert_eq!(tensor.dtype(), DType::F32);
    assert_eq!(tensor.f32_values().unwrap(), [0.5, 1.0, 3.5, 5.0]);
}

#[test]
fn representation_ops_bad_basis_rank_dtype_and_tolerance_are_atomic() {
    let mut tensor = CpuTensor::from_f64("w", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let before = tensor.to_le_bytes();
    let bad = RepresentationEdit::Projection {
        basis: input(&[2, 1], &[1.0, 1.0]),
        side: MapSide::Left,
        strength: 1.0,
        orthonormal_tolerance: 0.0,
    };
    assert!(apply_representation(
        &mut tensor,
        &region(&[0, 0], &[2, 2]),
        bad,
        &policy(DType::F64),
        limits()
    )
    .is_err());
    let bad = RepresentationEdit::Projection {
        basis: input(&[2, 3], &[0.0; 6]),
        side: MapSide::Left,
        strength: 1.0,
        orthonormal_tolerance: 0.0,
    };
    assert!(apply_representation(
        &mut tensor,
        &region(&[0, 0], &[2, 2]),
        bad,
        &policy(DType::F64),
        limits()
    )
    .is_err());
    let bad = RepresentationEdit::LinearMap {
        matrix: NumericInput::F32 {
            shape: &[2, 2],
            values: &[1.0, 0.0, 0.0, 1.0],
        },
        side: MapSide::Left,
    };
    assert!(apply_representation(
        &mut tensor,
        &region(&[0, 0], &[2, 2]),
        bad,
        &policy(DType::F64),
        limits()
    )
    .is_err());
    for tolerance in [f64::NAN, -1.0, 0.01] {
        let bad = RepresentationEdit::Projection {
            basis: input(&[2, 1], &[1.0, 0.0]),
            side: MapSide::Left,
            strength: 1.0,
            orthonormal_tolerance: tolerance,
        };
        assert!(apply_representation(
            &mut tensor,
            &region(&[0, 0], &[2, 2]),
            bad,
            &policy(DType::F64),
            limits()
        )
        .is_err());
    }
    assert_eq!(tensor.to_le_bytes(), before);
}

#[test]
fn representation_ops_budget_checks_happen_before_scratch_or_mutation() {
    let mut tensor = CpuTensor::from_f64("w", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let before = tensor.to_le_bytes();
    let edit = RepresentationEdit::LinearMap {
        matrix: input(&[2, 2], &[0.0, 1.0, 1.0, 0.0]),
        side: MapSide::Left,
    };
    let cost = representation_cost(
        OperationKind::LinearMap,
        &[2, 2],
        DType::F64,
        DType::F64,
        None,
        Some(MapSide::Left),
        None,
    )
    .unwrap();
    for bounds in [
        KernelLimits {
            max_scratch_bytes: cost.scratch_bytes - 1,
            ..limits()
        },
        KernelLimits {
            max_work_units: cost.work_upper_bound - 1,
            ..limits()
        },
    ] {
        assert_eq!(
            apply_representation(
                &mut tensor,
                &region(&[0, 0], &[2, 2]),
                edit,
                &policy(DType::F64),
                bounds
            )
            .unwrap_err()
            .code,
            ErrorCode::LimitExceeded
        );
    }
    assert_eq!(
        apply_representation(
            &mut tensor,
            &region(&[0, 0], &[2, 2]),
            edit,
            &policy(DType::F64),
            KernelLimits {
                max_changed_parameters: 0,
                ..limits()
            }
        )
        .unwrap_err()
        .code,
        ErrorCode::ParameterBudgetExceeded
    );
    assert_eq!(tensor.to_le_bytes(), before);
}

#[test]
fn representation_ops_nonfinite_input_and_late_overflow_leave_original_bytes() {
    let mut tensor = CpuTensor::from_f64("w", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let before = tensor.to_le_bytes();
    let bad = RepresentationEdit::LowRankUpdate {
        left: input(&[2, 1], &[1.0, f64::NAN]),
        right: input(&[1, 2], &[1.0, 1.0]),
        strength: 0.0,
    };
    assert_eq!(
        apply_representation(
            &mut tensor,
            &region(&[0, 0], &[2, 2]),
            bad,
            &policy(DType::F64),
            limits()
        )
        .unwrap_err()
        .code,
        ErrorCode::NonFinite
    );
    assert_eq!(tensor.to_le_bytes(), before);
    let mut tensor = CpuTensor::from_f64("w", vec![2, 2], vec![1.0, 2.0, f64::MAX, 1.0]).unwrap();
    let before = tensor.to_le_bytes();
    let mut numerical = policy(DType::F64);
    numerical.max_abs_parameter = f64::MAX;
    let bad = RepresentationEdit::LinearMap {
        matrix: input(&[2, 2], &[2.0, 0.0, 0.0, 2.0]),
        side: MapSide::Left,
    };
    assert_eq!(
        apply_representation(
            &mut tensor,
            &region(&[0, 0], &[2, 2]),
            bad,
            &numerical,
            limits()
        )
        .unwrap_err()
        .code,
        ErrorCode::NonFinite
    );
    assert_eq!(tensor.to_le_bytes(), before);
}

#[test]
fn representation_ops_invalid_permutation_dimensions_and_coefficient_fail_closed() {
    let mut tensor = CpuTensor::from_f64("w", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let before = tensor.to_le_bytes();
    for edit in [
        RepresentationEdit::PermuteAxis {
            axis: 0,
            permutation: &[0, 0],
        },
        RepresentationEdit::PermuteAxis {
            axis: 0,
            permutation: &[0, 2],
        },
        RepresentationEdit::PermuteAxis {
            axis: 2,
            permutation: &[0, 1],
        },
        RepresentationEdit::LowRankUpdate {
            left: input(&[2, 1], &[1.0, 1.0]),
            right: input(&[1, 2], &[1.0, 1.0]),
            strength: f64::INFINITY,
        },
    ] {
        assert!(apply_representation(
            &mut tensor,
            &region(&[0, 0], &[2, 2]),
            edit,
            &policy(DType::F64),
            limits()
        )
        .is_err());
    }
    assert_eq!(tensor.to_le_bytes(), before);
    assert!(representation_cost(
        OperationKind::LinearMap,
        &[u64::MAX, 0],
        DType::F64,
        DType::F64,
        None,
        Some(MapSide::Left),
        None
    )
    .is_err());
}

#[test]
fn representation_ops_empty_extreme_shape_does_not_run_empty_outer_loops() {
    let mut tensor = CpuTensor::from_f64("w", vec![u64::MAX, 0], vec![]).unwrap();
    let edit = RepresentationEdit::LinearMap {
        matrix: input(&[0, 0], &[]),
        side: MapSide::Right,
    };
    let receipt = apply_representation(
        &mut tensor,
        &region(&[0, 0], &[u64::MAX, 0]),
        edit,
        &policy(DType::F64),
        KernelLimits {
            max_changed_parameters: 0,
            max_scratch_bytes: 0,
            max_work_units: 0,
        },
    )
    .unwrap();
    assert_eq!(receipt.examined_parameters, 0);
    assert_eq!(receipt.requested_scratch_bytes, 0);
}
