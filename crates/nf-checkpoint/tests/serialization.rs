use nf_checkpoint::serialization::{serialize_tensors, SerializationLimits, TensorPayload};
use nf_checkpoint::tensor_file::{inspect_safetensors, TensorFileLimits};
use nf_core::error::ErrorCode;
use nf_core::schemas::DType;
use std::collections::BTreeMap;

#[test]
fn replay_serialization_uses_standard_format_and_stable_bytes() {
    let data = 3.0f64.to_le_bytes();
    let tensors = BTreeMap::from([(
        "w".to_owned(),
        TensorPayload {
            dtype: DType::F64,
            shape: &[1],
            bytes: &data,
        },
    )]);
    let first = serialize_tensors(&tensors, SerializationLimits::default()).unwrap();
    let again = serialize_tensors(&tensors, SerializationLimits::default()).unwrap();
    assert_eq!(first.bytes, again.bytes);
    assert_eq!(first.digest, again.digest);
    let inventory = inspect_safetensors(&first.bytes, TensorFileLimits::default()).unwrap();
    assert_eq!(
        &first.bytes[inventory.tensors["w"].data_range.clone()],
        &data
    );
    assert!(first.staging_upper_bound <= SerializationLimits::default().max_staging_bytes);
}
#[test]
fn replay_serialization_bounds_and_host_overflow_fail_before_library_calls() {
    let data = 3.0f64.to_le_bytes();
    let tensors = BTreeMap::from([(
        "w".to_owned(),
        TensorPayload {
            dtype: DType::F64,
            shape: &[1],
            bytes: &data,
        },
    )]);
    assert_eq!(
        serialize_tensors(
            &tensors,
            SerializationLimits {
                max_staging_bytes: 1,
                ..SerializationLimits::default()
            }
        )
        .unwrap_err()
        .code,
        ErrorCode::LimitExceeded
    );
    assert!(serialize_tensors(
        &tensors,
        SerializationLimits {
            max_checkpoint_bytes: 8,
            ..SerializationLimits::default()
        }
    )
    .is_err());
    let bad = BTreeMap::from([(
        "w".to_owned(),
        TensorPayload {
            dtype: DType::F64,
            shape: &[u64::MAX, 2],
            bytes: &[],
        },
    )]);
    assert!(serialize_tensors(&bad, SerializationLimits::default()).is_err());
    let bad = BTreeMap::from([(
        "w".to_owned(),
        TensorPayload {
            dtype: DType::F64,
            shape: &[2],
            bytes: &data,
        },
    )]);
    assert_eq!(
        serialize_tensors(&bad, SerializationLimits::default())
            .unwrap_err()
            .code,
        ErrorCode::IntegrityMismatch
    );
}
