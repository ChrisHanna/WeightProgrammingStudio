use nf_core::canonical::ContentDigest;
use nf_core::error::ErrorCode;
use nf_core::schemas::{DType, FactoryRef, ModelManifest, TensorFormat, TensorSpec, Validated};
use nf_graph::inventory::{
    tensor_set_digest, verify_inventory, InventoryLimits, TensorObservation, TensorRole,
};
use std::collections::BTreeMap;

fn spec(bytes: &[u8], group: &str) -> TensorSpec {
    TensorSpec {
        shape: vec![bytes.len() as u64],
        dtype: DType::U8,
        digest: ContentDigest::of_bytes(bytes),
        storage_group: group.to_owned(),
    }
}
fn manifest(
    parameters: BTreeMap<String, TensorSpec>,
    buffers: BTreeMap<String, TensorSpec>,
) -> Validated<ModelManifest> {
    let digest = tensor_set_digest(&parameters).unwrap();
    Validated::new(ModelManifest {
        schema_version: 1,
        architecture: FactoryRef {
            id: "fixture.cpu".to_owned(),
            version: "1".to_owned(),
            config: ContentDigest::of_bytes(b"fixture"),
        },
        parameters,
        buffers,
        parameter_digest: digest,
        format: TensorFormat::DenseSafetensors,
        packing_policy: None,
        assets: BTreeMap::new(),
        usage_rights: "synthetic fixture".to_owned(),
    })
    .unwrap()
}
fn observation<'a>(name: &'a str, bytes: &'a [u8]) -> TensorObservation<'a> {
    TensorObservation {
        name,
        role: TensorRole::Parameter,
        shape: &[4],
        dtype: DType::U8,
        bytes,
    }
}

#[test]
fn tensor_inventory_equal_values_are_not_shared_storage() {
    let a = vec![1u8, 2, 3, 4];
    let b = vec![1u8, 2, 3, 4];
    let declared = manifest(
        BTreeMap::from([
            ("a".to_owned(), spec(&a, "a")),
            ("b".to_owned(), spec(&b, "b")),
        ]),
        BTreeMap::new(),
    );
    let inventory = verify_inventory(
        &declared,
        &[observation("a", &a), observation("b", &b)],
        InventoryLimits::default(),
    )
    .unwrap();
    assert_eq!(inventory.alias_groups().len(), 2);
    assert_eq!(
        inventory.entries()["a"].content_digest,
        inventory.entries()["b"].content_digest
    );
    assert_eq!(inventory.entries()["a"].device, "cpu");
    assert_eq!(inventory.observed_bytes(), 8);
}

#[test]
fn tensor_inventory_declared_aliases_require_real_shared_bytes() {
    let a = vec![1u8, 2, 3, 4];
    let b = vec![1u8, 2, 3, 4];
    let declared = manifest(
        BTreeMap::from([
            ("a".to_owned(), spec(&a, "shared")),
            ("b".to_owned(), spec(&a, "shared")),
        ]),
        BTreeMap::new(),
    );
    assert_eq!(
        verify_inventory(
            &declared,
            &[observation("a", &a), observation("b", &b)],
            InventoryLimits::default()
        )
        .unwrap_err()
        .code,
        ErrorCode::IntegrityMismatch
    );
    let inventory = verify_inventory(
        &declared,
        &[observation("b", &a), observation("a", &a)],
        InventoryLimits::default(),
    )
    .unwrap();
    assert_eq!(inventory.alias_groups()["shared"], ["a", "b"]);
    assert_eq!(inventory.parameter_digest(), &declared.parameter_digest);
}

#[test]
fn tensor_inventory_undeclared_and_partial_aliases_are_rejected() {
    let data = [1u8, 2, 3, 4, 5, 6];
    let declared = manifest(
        BTreeMap::from([
            ("a".to_owned(), spec(&data[..4], "a")),
            ("b".to_owned(), spec(&data[..4], "b")),
        ]),
        BTreeMap::new(),
    );
    assert!(verify_inventory(
        &declared,
        &[observation("a", &data[..4]), observation("b", &data[..4])],
        InventoryLimits::default()
    )
    .is_err());
    let partial = manifest(
        BTreeMap::from([
            ("a".to_owned(), spec(&data[..4], "a")),
            ("b".to_owned(), spec(&data[2..], "b")),
        ]),
        BTreeMap::new(),
    );
    assert_eq!(
        verify_inventory(
            &partial,
            &[observation("a", &data[..4]), observation("b", &data[2..])],
            InventoryLimits::default()
        )
        .unwrap_err()
        .code,
        ErrorCode::UnsupportedCapability
    );
}

#[test]
fn tensor_inventory_parameters_and_buffers_are_both_checked() {
    let parameter = [1u8, 2, 3, 4];
    let buffer = [5u8, 6, 7, 8];
    let declared = manifest(
        BTreeMap::from([("weight".to_owned(), spec(&parameter, "weight"))]),
        BTreeMap::from([("state".to_owned(), spec(&buffer, "state"))]),
    );
    let mut state = observation("state", &buffer);
    state.role = TensorRole::Buffer;
    let inventory = verify_inventory(
        &declared,
        &[observation("weight", &parameter), state],
        InventoryLimits::default(),
    )
    .unwrap();
    assert_eq!(inventory.entries()["state"].role, TensorRole::Buffer);
    assert_eq!(
        inventory.buffer_digest(),
        &tensor_set_digest(&declared.buffers).unwrap()
    );
    assert!(verify_inventory(
        &declared,
        &[
            observation("weight", &parameter),
            observation("state", &buffer)
        ],
        InventoryLimits::default()
    )
    .is_err());
}

#[test]
fn tensor_inventory_missing_duplicate_content_shape_and_dtype_fail() {
    let data = [1u8, 2, 3, 4];
    let declared = manifest(
        BTreeMap::from([("a".to_owned(), spec(&data, "a"))]),
        BTreeMap::new(),
    );
    assert!(verify_inventory(&declared, &[], InventoryLimits::default()).is_err());
    assert!(verify_inventory(
        &declared,
        &[observation("a", &[9, 2, 3, 4])],
        InventoryLimits::default()
    )
    .is_err());
    let mut wrong_shape = observation("a", &data);
    wrong_shape.shape = &[2, 2];
    assert!(verify_inventory(&declared, &[wrong_shape], InventoryLimits::default()).is_err());
    let mut wrong_dtype = observation("a", &data);
    wrong_dtype.dtype = DType::F32;
    assert!(verify_inventory(&declared, &[wrong_dtype], InventoryLimits::default()).is_err());
    let two = manifest(
        BTreeMap::from([
            ("a".to_owned(), spec(&data, "a")),
            ("b".to_owned(), spec(&data, "b")),
        ]),
        BTreeMap::new(),
    );
    assert_eq!(
        verify_inventory(
            &two,
            &[observation("a", &data), observation("a", &data)],
            InventoryLimits::default()
        )
        .unwrap_err()
        .code,
        ErrorCode::DuplicateKey
    );
}

#[test]
fn tensor_inventory_packing_and_read_budgets_fail_closed() {
    let data = [1u8, 2, 3, 4];
    let base = manifest(
        BTreeMap::from([("a".to_owned(), spec(&data, "a"))]),
        BTreeMap::new(),
    );
    assert_eq!(
        verify_inventory(
            &base,
            &[observation("a", &data)],
            InventoryLimits {
                max_observed_bytes: 3,
                ..InventoryLimits::default()
            }
        )
        .unwrap_err()
        .code,
        ErrorCode::LimitExceeded
    );
    let mut packed = (*base).clone();
    packed.format = TensorFormat::QuantizedSafetensors;
    packed.packing_policy = Some(FactoryRef {
        id: "unknown.packing".to_owned(),
        version: "1".to_owned(),
        config: ContentDigest::of_bytes(b"unknown"),
    });
    let packed = Validated::new(packed).unwrap();
    assert_eq!(
        verify_inventory(
            &packed,
            &[observation("a", &data)],
            InventoryLimits::default()
        )
        .unwrap_err()
        .code,
        ErrorCode::UnsupportedCapability
    );
}

#[test]
fn tensor_inventory_parameter_identity_is_order_stable_and_checked() {
    let a = vec![1u8, 2, 3, 4];
    let b = vec![4u8, 3, 2, 1];
    let base = manifest(
        BTreeMap::from([
            ("a".to_owned(), spec(&a, "a")),
            ("b".to_owned(), spec(&b, "b")),
        ]),
        BTreeMap::new(),
    );
    let first = verify_inventory(
        &base,
        &[observation("a", &a), observation("b", &b)],
        InventoryLimits::default(),
    )
    .unwrap();
    let second = verify_inventory(
        &base,
        &[observation("b", &b), observation("a", &a)],
        InventoryLimits::default(),
    )
    .unwrap();
    assert_eq!(first.parameter_digest(), second.parameter_digest());
    assert_eq!(
        first.source_manifest_digest(),
        second.source_manifest_digest()
    );
    let mut corrupt = (*base).clone();
    corrupt.parameter_digest = ContentDigest::of_bytes(b"not the model");
    let corrupt = Validated::new(corrupt).unwrap();
    assert!(verify_inventory(
        &corrupt,
        &[observation("a", &a), observation("b", &b)],
        InventoryLimits::default()
    )
    .is_err());
    let encoded = serde_json::to_string(&first).unwrap();
    assert!(!encoded.contains("empty_tag"));
    assert!(!encoded.contains("start"));
}

#[test]
fn tensor_inventory_empty_pointer_equality_does_not_establish_aliases() {
    let mut a = spec(&[], "shared");
    a.shape = vec![0];
    let base = manifest(
        BTreeMap::from([("a".to_owned(), a.clone()), ("b".to_owned(), a)]),
        BTreeMap::new(),
    );
    let entries = [
        TensorObservation {
            name: "a",
            role: TensorRole::Parameter,
            shape: &[0],
            dtype: DType::U8,
            bytes: &[],
        },
        TensorObservation {
            name: "b",
            role: TensorRole::Parameter,
            shape: &[0],
            dtype: DType::U8,
            bytes: &[],
        },
    ];
    assert!(verify_inventory(&base, &entries, InventoryLimits::default()).is_err());
}
