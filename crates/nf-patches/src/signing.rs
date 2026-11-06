//! Ed25519 (RFC 8032) signatures over patch digests.
//!
//! This is the single signing implementation shared by the `nf` CLI and the
//! Python extension. Keys come from the operating system's CSPRNG; there is
//! deliberately no way to derive one from a passphrase or label. Verification
//! always takes the public key from the caller: a key carried inside a signed
//! document can only prove that document is self-consistent, not who signed it.

use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use nf_core::canonical::ContentDigest;
use nf_core::error::{CoreError, ErrorCode, Result};

pub const SIGNATURE_SCHEME: &str = "ed25519_v1";
const PRIVATE_PREFIX: &str = "ed25519_priv:";
const PUBLIC_PREFIX: &str = "ed25519_pub:";
const SIGNATURE_PREFIX: &str = "ed25519_sig:";

fn invalid(message: impl Into<String>) -> CoreError {
    CoreError::new(ErrorCode::InvalidValue, "/patch/signing", message)
}

fn rejected(message: impl Into<String>) -> CoreError {
    CoreError::new(ErrorCode::IntegrityMismatch, "/patch/signing", message)
}

fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn from_hex<const N: usize>(text: &str) -> Option<[u8; N]> {
    if text.len() != N * 2 || !text.is_ascii() {
        return None;
    }
    let mut out = [0u8; N];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&text[i * 2..i * 2 + 2], 16).ok()?;
    }
    Some(out)
}

/// Binds the signer identity and the patch digest with domain separation.
fn signed_message(signer: &str, patch_digest: &str) -> Vec<u8> {
    let mut message = b"nf-patch-sig-v2\0".to_vec();
    message.extend_from_slice(signer.as_bytes());
    message.push(0);
    message.extend_from_slice(patch_digest.as_bytes());
    message
}

fn parse_private(private_key: &str) -> Result<SigningKey> {
    let bytes = private_key
        .strip_prefix(PRIVATE_PREFIX)
        .and_then(from_hex::<32>)
        .ok_or_else(|| invalid("signing key must be ed25519_priv: followed by 64 hex characters"))?;
    Ok(SigningKey::from_bytes(&bytes))
}

pub fn encode_public_key(key: &VerifyingKey) -> String {
    format!("{PUBLIC_PREFIX}{}", to_hex(key.as_bytes()))
}

pub fn key_fingerprint(public_key: &str) -> String {
    ContentDigest::of_bytes(format!("nf-pubkey-fp-v2:{public_key}").as_bytes()).to_string()
}

#[derive(Debug)]
pub struct GeneratedKeypair {
    pub private_key: String,
    pub public_key: String,
    pub key_fingerprint: String,
}

pub fn generate_keypair() -> Result<GeneratedKeypair> {
    let mut seed = [0u8; 32];
    getrandom::getrandom(&mut seed)
        .map_err(|e| CoreError::new(ErrorCode::Io, "/patch/signing", e.to_string()))?;
    let signing = SigningKey::from_bytes(&seed);
    let public_key = encode_public_key(&signing.verifying_key());
    Ok(GeneratedKeypair {
        private_key: format!("{PRIVATE_PREFIX}{}", to_hex(&seed)),
        key_fingerprint: key_fingerprint(&public_key),
        public_key,
    })
}

/// Returns `(signature, public_key)`.
pub fn sign_digest(private_key: &str, signer: &str, patch_digest: &str) -> Result<(String, String)> {
    let signing = parse_private(private_key)?;
    let signature = signing.sign(&signed_message(signer, patch_digest));
    Ok((
        format!("{SIGNATURE_PREFIX}{}", to_hex(&signature.to_bytes())),
        encode_public_key(&signing.verifying_key()),
    ))
}

/// Fails closed unless `signature` is valid for this exact signer, digest and trusted key.
pub fn verify_digest(
    trusted_public_key: &str,
    signer: &str,
    patch_digest: &str,
    signature: &str,
) -> Result<()> {
    let key_bytes = trusted_public_key
        .strip_prefix(PUBLIC_PREFIX)
        .and_then(from_hex::<32>)
        .ok_or_else(|| invalid("public key must be ed25519_pub: followed by 64 hex characters"))?;
    let key = VerifyingKey::from_bytes(&key_bytes)
        .map_err(|_| invalid("public key is not a valid Ed25519 point"))?;
    let signature_bytes = signature
        .strip_prefix(SIGNATURE_PREFIX)
        .and_then(from_hex::<64>)
        .ok_or_else(|| rejected("signature must be ed25519_sig: followed by 128 hex characters"))?;
    key.verify_strict(
        &signed_message(signer, patch_digest),
        &Signature::from_bytes(&signature_bytes),
    )
    .map_err(|_| rejected("Ed25519 signature verification failed for the supplied public key"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIGEST: &str = "sha256:0000";

    #[test]
    fn matches_rfc8032_test_vector_public_key() {
        let private = "ed25519_priv:9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60";
        let (_, public) = sign_digest(private, "s", DIGEST).unwrap();
        assert_eq!(
            public,
            "ed25519_pub:d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
        );
    }

    #[test]
    fn round_trip_and_every_tamper_fails() {
        let kp = generate_keypair().unwrap();
        let other = generate_keypair().unwrap();
        assert_ne!(kp.private_key, other.private_key);
        let (sig, public) = sign_digest(&kp.private_key, "alice", DIGEST).unwrap();
        assert_eq!(public, kp.public_key);
        verify_digest(&kp.public_key, "alice", DIGEST, &sig).unwrap();
        assert!(verify_digest(&other.public_key, "alice", DIGEST, &sig).is_err());
        assert!(verify_digest(&kp.public_key, "mallory", DIGEST, &sig).is_err());
        assert!(verify_digest(&kp.public_key, "alice", "sha256:1111", &sig).is_err());
        assert!(verify_digest(&kp.public_key, "alice", DIGEST, "ed25519_sig:00").is_err());
    }

    #[test]
    fn non_key_strings_are_rejected() {
        assert!(sign_digest("my-passphrase", "s", DIGEST).is_err());
        assert!(verify_digest("ed25519_pub:zz", "s", DIGEST, "ed25519_sig:00").is_err());
    }
}
