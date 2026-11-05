use nf_checkpoint::tensor_file::{inspect_safetensors, TensorFileLimits};
use nf_core::error::ErrorCode;
use nf_core::schemas::DType;

fn file(header: &str, data: &[u8]) -> Vec<u8> {
    let mut metadata = header.as_bytes().to_vec();
    metadata.resize(metadata.len().div_ceil(8) * 8, b' ');
    let mut out = (metadata.len() as u64).to_le_bytes().to_vec();
    out.extend(metadata);
    out.extend_from_slice(data);
    out
}

fn inspect(
    header: &str,
    data: &[u8],
) -> Result<nf_checkpoint::tensor_file::TensorFileInventory, nf_core::error::CoreError> {
    inspect_safetensors(&file(header, data), TensorFileLimits::default())
}

#[test]
fn safetensors_reader_inspects_real_format_without_loading_models() {
    let raw = file(
        r#"{"weight":{"dtype":"F64","shape":[1],"data_offsets":[0,8]},"__metadata__":{"note":"inert model text"}}"#,
        &2.5f64.to_le_bytes(),
    );
    let inventory = inspect_safetensors(&raw, TensorFileLimits::default()).unwrap();
    assert_eq!(inventory.tensors["weight"].dtype, DType::F64);
    assert_eq!(inventory.tensors["weight"].shape, [1]);
    assert_eq!(
        &raw[inventory.tensors["weight"].data_range.clone()],
        &2.5f64.to_le_bytes()
    );
    assert_eq!(inventory.metadata["note"], "inert model text");
    assert_eq!(
        inventory.file_digest,
        nf_core::canonical::ContentDigest::of_bytes(&raw)
    );
}

#[test]
fn safetensors_reader_all_supported_dtypes_and_scalar_shapes() {
    for (name, dtype) in [
        ("F32", DType::F32),
        ("F64", DType::F64),
        ("F16", DType::F16),
        ("BF16", DType::Bf16),
        ("I8", DType::I8),
        ("U8", DType::U8),
        ("I32", DType::I32),
        ("I64", DType::I64),
    ] {
        let header = format!(
            r#"{{"scalar":{{"dtype":"{name}","shape":[],"data_offsets":[0,{}]}}}}"#,
            dtype.byte_width()
        );
        let data = vec![0; dtype.byte_width() as usize];
        assert_eq!(
            inspect(&header, &data).unwrap().tensors["scalar"].dtype,
            dtype
        );
    }
}

#[test]
fn safetensors_reader_rejects_duplicate_and_unknown_fields() {
    for header in [
        r#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
        r#"{"x":{"dtype":"U8","dtype":"F32","shape":[1],"data_offsets":[0,1]}}"#,
    ] {
        assert_eq!(
            inspect(header, &[0]).unwrap_err().code,
            ErrorCode::DuplicateKey
        );
    }
    assert!(inspect(
        r#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1],"execute":"not allowed"}}"#,
        &[0]
    )
    .is_err());
    assert!(inspect(r#"{"__metadata__":{"bad":{"code":"not a string"}}}"#, &[]).is_err());
}

#[test]
fn safetensors_reader_offsets_overlap_holes_and_extra_payload_are_rejected() {
    for header in [
        r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"b":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
        r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}"#,
        r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[1,0]}}"#,
        r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,3]}}"#,
    ] {
        assert!(inspect(header, &[0, 0]).is_err());
    }
    assert!(inspect(
        r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
        &[0, 0]
    )
    .is_err());
}

#[test]
fn safetensors_reader_bounds_before_allocating_or_indexing() {
    let huge = u64::MAX.to_le_bytes();
    assert_eq!(
        inspect_safetensors(&huge, TensorFileLimits::default())
            .unwrap_err()
            .code,
        ErrorCode::LimitExceeded
    );
    assert!(inspect_safetensors(&[0; 7], TensorFileLimits::default()).is_err());
    let mut truncated = 32u64.to_le_bytes().to_vec();
    truncated.extend_from_slice(b"{}");
    assert!(inspect_safetensors(&truncated, TensorFileLimits::default()).is_err());
    assert!(inspect(
        r#"{"x":{"dtype":"F64","shape":[18446744073709551615,2],"data_offsets":[0,8]}}"#,
        &[0; 8]
    )
    .is_err());
    let valid = file(
        r#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
        &[0],
    );
    for limits in [
        TensorFileLimits {
            max_file_bytes: 8,
            ..TensorFileLimits::default()
        },
        TensorFileLimits {
            max_header_bytes: 1,
            ..TensorFileLimits::default()
        },
        TensorFileLimits {
            max_rank: 0,
            ..TensorFileLimits::default()
        },
    ] {
        assert!(inspect_safetensors(&valid, limits).is_err());
    }
}

#[test]
fn safetensors_reader_unsupported_dtype_and_invalid_text_are_explicit() {
    let unsupported = inspect(
        r#"{"x":{"dtype":"BOOL","shape":[1],"data_offsets":[0,1]}}"#,
        &[0],
    )
    .unwrap_err();
    assert_eq!(unsupported.code, ErrorCode::UnsupportedCapability);
    assert_eq!(unsupported.path, "/x/dtype");
    let mut invalid_utf8 = 8u64.to_le_bytes().to_vec();
    invalid_utf8.extend_from_slice(&[0xff; 8]);
    assert!(inspect_safetensors(&invalid_utf8, TensorFileLimits::default()).is_err());
    assert!(inspect_safetensors(b"not a pickle loader", TensorFileLimits::default()).is_err());
}
