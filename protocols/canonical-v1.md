# Canonical records v1

Rust is the authoritative encoder. Python uses these same vectors and core APIs, not a second approximate encoding. This protocol is intentionally not an assertion of RFC 8785 compatibility.

## JSON domain

Inputs are UTF-8 JSON with no byte-order mark or trailing content. A generic document can be any JSON value. Records use the strict envelope `schema_version`, `kind`, and `payload`. Version 1 is the only supported envelope version. Domain payloads also reject unknown fields.

Object keys are decoded strings sorted by their UTF-8 byte sequence. Duplicate decoded keys are rejected at every depth, including escape-equivalent keys. String content is not Unicode-normalized: NFC and NFD strings remain distinct. Serialization uses minimal JSON whitespace, literal Unicode, and the escaping behavior of pinned serde_json 1.0.140. Arrays preserve order.

Integer tokens must fit signed i64 when negative or unsigned u64 otherwise. They never silently overflow into floating-point values. Negative integer zero normalizes to zero. Decimal/exponent tokens use finite IEEE-754 binary64, serialized by the pinned serializer's shortest round-tripping JSON representation. Thus `1` and `1.0` remain distinct, float negative zero retains its sign, and decimal precision follows binary64 semantics. NaN, infinity, and overflowing exponents are rejected. A future numeric-format change requires a protocol version and new vectors.

Default input limits are 1,048,576 bytes, depth 64 (root depth zero), and 100,000 values. Callers may impose tighter limits. Depth cannot exceed 64. Parsing validates bounds while consuming input, before constructing an unbounded tree. Unknown schema fields are rejected by typed record decoding, not guessed by generic canonicalization.

## Identity and errors

A content digest is `sha256:` followed by 64 lowercase hexadecimal digits. It hashes canonical UTF-8 bytes for records, and exact artifact bytes for raw blobs. No newline, implicit schema migration, key normalization, or hardware identity is appended to the hashed bytes. Record schema version and artifact kind must be explicit in their manifests.

Errors have stable `code`, JSON-pointer `path`, and human-readable `message` fields. Root errors have the empty pointer. Syntax/lexical errors include byte or line/column information in the message; duplicate and structural errors include their decoded pointer. Message wording is not an identity protocol. Error records themselves reject unknown fields.

Shared vectors freeze exact bytes and SHA-256 values. Changing a dependency or encoder is permitted only if all version-1 vectors and rejection cases still pass, or through an explicitly versioned migration.
