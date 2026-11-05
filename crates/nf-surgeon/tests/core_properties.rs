//! Exhaustive small domains with fixed cardinality, memory and work caps.
use nf_core::error::ErrorCode;
use nf_core::schemas::DType;
use nf_surgeon::basic::{apply_basic, BasicEdit, CpuTensor};
use nf_surgeon::representation::{apply_representation, KernelLimits, RepresentationEdit};
use nf_weight_ir::program::{NonFinitePolicy, NumericalPolicy, RoundingMode, TensorRegion};
fn policy(dtype: DType) -> NumericalPolicy {
    NumericalPolicy {
        accumulation_dtype: dtype,
        output_rounding: RoundingMode::NearestTiesEven,
        non_finite: NonFinitePolicy::Reject,
        max_abs_parameter: 1000.0,
        backend_abs_tolerance: 0.0,
    }
}
#[test]
fn core_properties_exhaustive_small_shapes_masks_and_atomic_failure() {
    let mut cases = 0;
    for dtype in [DType::F32, DType::F64] {
        for rows in 0..=4u64 {
            for cols in 0..=4u64 {
                let shape = vec![rows, cols];
                let values: Vec<f64> = (0..rows * cols).map(|i| (i + 1) as f64).collect();
                let source = if dtype == DType::F64 {
                    CpuTensor::from_f64("w", shape.clone(), values.clone()).unwrap()
                } else {
                    CpuTensor::from_f32(
                        "w",
                        shape.clone(),
                        values.iter().map(|x| *x as f32).collect(),
                    )
                    .unwrap()
                };
                for row in 0..=rows {
                    for col in 0..=cols {
                        for nr in 0..=rows - row {
                            for nc in 0..=cols - col {
                                cases += 1;
                                assert!(cases <= 10_000);
                                let region = TensorRegion {
                                    tensor: "w".to_owned(),
                                    start: vec![row, col],
                                    shape: vec![nr, nc],
                                };
                                let mut candidate = source.clone();
                                let original = source.to_le_bytes();
                                let receipt = apply_basic(
                                    &mut candidate,
                                    &region,
                                    BasicEdit::Scale(2.0),
                                    &policy(dtype),
                                    nr * nc,
                                )
                                .unwrap();
                                assert_eq!(receipt.changed_parameters, nr * nc);
                                let bytes = candidate.to_le_bytes();
                                let width = dtype.byte_width() as usize;
                                for r in 0..rows {
                                    for c in 0..cols {
                                        let index = (r * cols + c) as usize;
                                        let changed =
                                            r >= row && r < row + nr && c >= col && c < col + nc;
                                        if changed {
                                            let expected = values[index] * 2.0;
                                            let expected = if dtype == DType::F64 {
                                                expected.to_le_bytes().to_vec()
                                            } else {
                                                (expected as f32).to_le_bytes().to_vec()
                                            };
                                            assert_eq!(
                                                &bytes[index * width..(index + 1) * width],
                                                expected
                                            );
                                        } else {
                                            assert_eq!(
                                                &bytes[index * width..(index + 1) * width],
                                                &original[index * width..(index + 1) * width]
                                            );
                                        }
                                    }
                                }
                                if nr * nc > 0 {
                                    let mut rejected = source.clone();
                                    assert_eq!(
                                        apply_basic(
                                            &mut rejected,
                                            &region,
                                            BasicEdit::Scale(2.0),
                                            &policy(dtype),
                                            nr * nc - 1
                                        )
                                        .unwrap_err()
                                        .code,
                                        ErrorCode::ParameterBudgetExceeded
                                    );
                                    assert_eq!(rejected.to_le_bytes(), original);
                                }
                                assert_eq!(source.to_le_bytes(), original);
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(cases, 2450);
}
fn permutations(n: usize) -> Vec<Vec<u64>> {
    fn generate(prefix: &mut Vec<u64>, n: usize, output: &mut Vec<Vec<u64>>) {
        if prefix.len() == n {
            output.push(prefix.clone());
            return;
        }
        for i in 0..n as u64 {
            if !prefix.contains(&i) {
                prefix.push(i);
                generate(prefix, n, output);
                prefix.pop();
            }
        }
    }
    let mut output = Vec::new();
    generate(&mut Vec::new(), n, &mut output);
    output
}
#[test]
fn core_properties_all_small_permutations_inverse_and_composition_are_exact() {
    let limits = KernelLimits {
        max_changed_parameters: 64,
        max_scratch_bytes: 4096,
        max_work_units: 100_000,
    };
    let mut cases = 0;
    for n in 0..=6usize {
        let values: Vec<f64> = (0..3 * n)
            .map(|i| if i == 0 { -0.0 } else { i as f64 })
            .collect();
        let source = CpuTensor::from_f64("w", vec![3, n as u64], values.clone()).unwrap();
        let region = TensorRegion {
            tensor: "w".to_owned(),
            start: vec![0, 0],
            shape: vec![3, n as u64],
        };
        let reverse: Vec<u64> = (0..n as u64).rev().collect();
        for permutation in permutations(n) {
            cases += 1;
            assert!(cases <= 874);
            let mut tensor = source.clone();
            apply_representation(
                &mut tensor,
                &region,
                RepresentationEdit::PermuteAxis {
                    axis: 1,
                    permutation: &permutation,
                },
                &policy(DType::F64),
                limits,
            )
            .unwrap();
            for row in 0..3 {
                for col in 0..n {
                    assert_eq!(
                        tensor.f64_values().unwrap()[row * n + col].to_bits(),
                        values[row * n + permutation[col] as usize].to_bits()
                    );
                }
            }
            let mut inverse = vec![0; n];
            for (i, p) in permutation.iter().enumerate() {
                inverse[*p as usize] = i as u64;
            }
            let mut restored = tensor.clone();
            apply_representation(
                &mut restored,
                &region,
                RepresentationEdit::PermuteAxis {
                    axis: 1,
                    permutation: &inverse,
                },
                &policy(DType::F64),
                limits,
            )
            .unwrap();
            assert_eq!(restored.to_le_bytes(), source.to_le_bytes());
            apply_representation(
                &mut tensor,
                &region,
                RepresentationEdit::PermuteAxis {
                    axis: 1,
                    permutation: &reverse,
                },
                &policy(DType::F64),
                limits,
            )
            .unwrap();
            let composition: Vec<u64> = reverse.iter().map(|q| permutation[*q as usize]).collect();
            let mut equivalent = source.clone();
            apply_representation(
                &mut equivalent,
                &region,
                RepresentationEdit::PermuteAxis {
                    axis: 1,
                    permutation: &composition,
                },
                &policy(DType::F64),
                limits,
            )
            .unwrap();
            assert_eq!(tensor.to_le_bytes(), equivalent.to_le_bytes());
        }
    }
    assert_eq!(cases, 874);
}
