use nf_core::canonical::{decode_record, CanonicalDocument, ContentDigest, Limits};
use nf_core::error::{CoreError, ErrorCode};
use serde::Deserialize;

fn parse(input: &str) -> CanonicalDocument {
    CanonicalDocument::parse(input.as_bytes(), Limits::default()).unwrap()
}

fn rejected(input: &str, code: ErrorCode) {
    let error = CanonicalDocument::parse(input.as_bytes(), Limits::default()).unwrap_err();
    assert_eq!(error.code, code, "{input}: {error}");
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Vector {
    input: String,
    canonical: String,
    digest: String,
}

#[test]
fn canonical_shared_golden_vectors() {
    let vectors: Vec<Vector> =
        serde_json::from_str(include_str!("../../../protocols/canonical-v1-vectors.json")).unwrap();
    for vector in vectors {
        let document = parse(&vector.input);
        assert_eq!(document.bytes(), vector.canonical.as_bytes());
        assert_eq!(document.digest().as_str(), vector.digest);
        assert_eq!(parse(&vector.canonical).bytes(), document.bytes());
    }
}

#[test]
fn core_conformance_shared_error_vectors() {
    let vectors: Vec<serde_json::Value> =
        serde_json::from_str(include_str!("../../../protocols/error-v1-vectors.json")).unwrap();
    for vector in vectors {
        let error = CanonicalDocument::parse(
            vector["input"].as_str().unwrap().as_bytes(),
            Limits::default(),
        )
        .unwrap_err();
        assert_eq!(serde_json::to_value(error.code).unwrap(), vector["code"]);
        assert_eq!(error.path, vector["path"].as_str().unwrap());
    }
}

#[test]
fn canonical_rejects_duplicate_and_escape_equivalent_keys() {
    let error = CanonicalDocument::parse(br#"{"outer":[{"a/b~":1,"a/b~":2}]}"#, Limits::default())
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::DuplicateKey);
    assert_eq!(error.path, "/outer/0/a~1b~0");
    rejected(r#"{"a":1,"\u0061":2}"#, ErrorCode::DuplicateKey);
}

#[test]
fn canonical_integer_ranges_do_not_fall_back_to_float() {
    assert_eq!(
        parse("18446744073709551615").bytes(),
        b"18446744073709551615"
    );
    assert_eq!(
        parse("-9223372036854775808").bytes(),
        b"-9223372036854775808"
    );
    rejected("18446744073709551616", ErrorCode::IntegerOutOfRange);
    rejected("-9223372036854775809", ErrorCode::IntegerOutOfRange);
    assert_eq!(parse("-0").bytes(), b"0");
}

#[test]
fn canonical_finite_float_rules() {
    rejected("1e400", ErrorCode::NonFinite);
    rejected("-1e400", ErrorCode::NonFinite);
    rejected("NaN", ErrorCode::InvalidJson);
    rejected("Infinity", ErrorCode::InvalidJson);
    assert_eq!(parse("1.0").bytes(), b"1.0");
    assert_eq!(parse("-0.0").bytes(), b"-0.0");
    assert_eq!(parse("1e0").digest(), parse("1.0").digest());
    assert_ne!(parse("1").digest(), parse("1.0").digest());
}

#[test]
fn canonical_strings_do_not_trigger_number_checks() {
    let value = r#"{"text":"18446744073709551616 and \"1e999\"","digit":2}"#;
    assert!(parse(value).value().is_object());
    assert_ne!(parse("\"é\"").digest(), parse("\"e\\u0301\"").digest());
}

#[test]
fn canonical_parsing_is_structurally_bounded() {
    for (input, limits) in [
        (
            "{}",
            Limits {
                max_bytes: 1,
                ..Limits::default()
            },
        ),
        (
            "[[]]",
            Limits {
                max_depth: 0,
                ..Limits::default()
            },
        ),
        (
            "[1,2]",
            Limits {
                max_nodes: 2,
                ..Limits::default()
            },
        ),
    ] {
        assert_eq!(
            CanonicalDocument::parse(input.as_bytes(), limits)
                .unwrap_err()
                .code,
            ErrorCode::LimitExceeded
        );
    }
    let at_limit = format!("{}0{}", "[".repeat(64), "]".repeat(64));
    assert!(CanonicalDocument::parse(at_limit.as_bytes(), Limits::default()).is_ok());
    let over_limit = format!("{}0{}", "[".repeat(65), "]".repeat(65));
    rejected(&over_limit, ErrorCode::LimitExceeded);
}

#[test]
fn canonical_rejects_invalid_utf8_and_trailing_content() {
    assert_eq!(
        CanonicalDocument::parse(&[0xff], Limits::default())
            .unwrap_err()
            .code,
        ErrorCode::InvalidJson
    );
    for input in ["{} {}", "[1,]", "{a:1}", "01", "\u{feff}{}"] {
        rejected(input, ErrorCode::InvalidJson);
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Payload {
    enabled: bool,
}

#[test]
fn canonical_typed_record_rejects_unknown_fields_and_versions() {
    let record = decode_record::<Payload>(
        br#"{"schema_version":1,"kind":"test","payload":{"enabled":true}}"#,
    )
    .unwrap();
    assert!(record.payload.enabled);
    assert_eq!(record.kind, "test");
    for input in [
        br#"{"schema_version":1,"kind":"test","payload":{"enabled":true,"extra":1}}"#.as_slice(),
        br#"{"schema_version":1,"kind":"test","payload":{"enabled":true},"extra":1}"#,
    ] {
        assert_eq!(
            decode_record::<Payload>(input).unwrap_err().code,
            ErrorCode::UnknownField
        );
    }
    let error = decode_record::<Payload>(
        br#"{"schema_version":2,"kind":"test","payload":{"enabled":true}}"#,
    )
    .unwrap_err();
    assert_eq!(error.code, ErrorCode::UnsupportedVersion);
    assert_eq!(error.path, "/schema_version");
}

#[test]
fn canonical_digest_parser_and_error_wire_format() {
    let digest = ContentDigest::of_bytes(b"abc");
    assert_eq!(
        digest.to_string(),
        "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    let roundtrip: ContentDigest =
        serde_json::from_str(&serde_json::to_string(&digest).unwrap()).unwrap();
    assert_eq!(digest, roundtrip);
    for invalid in ["abc", "sha256:abc", &format!("sha256:{}", "A".repeat(64))] {
        assert!(invalid.parse::<ContentDigest>().is_err());
    }
    let error = CoreError::new(ErrorCode::DuplicateKey, "/payload/a", "duplicate");
    let bytes = serde_json::to_string(&error).unwrap();
    assert!(bytes.contains("DUPLICATE_KEY"));
    assert_eq!(serde_json::from_str::<CoreError>(&bytes).unwrap(), error);
    assert!(serde_json::from_str::<CoreError>(
        r#"{"code":"DUPLICATE_KEY","path":"","message":"bad","extra":1}"#
    )
    .is_err());
}
