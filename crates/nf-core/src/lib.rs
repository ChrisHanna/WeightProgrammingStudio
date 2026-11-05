//! Authoritative, Python-independent Foundry contracts and identities.

pub mod canonical;
pub mod capabilities;
pub mod error;
pub mod protocol;
pub mod schemas;
pub mod study;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub const WORKSPACE_CRATES: [&str; 9] = [
    "nf-core",
    "nf-checkpoint",
    "nf-weight-ir",
    "nf-surgeon",
    "nf-graph",
    "nf-patches",
    "nf-experiments",
    "nf-cli",
    "nf-python",
];

#[cfg(test)]
mod tests {
    #[test]
    fn package_smoke_version_and_inventory() {
        assert_eq!(super::VERSION, "0.1.0");
        let unique: std::collections::BTreeSet<_> = super::WORKSPACE_CRATES.iter().collect();
        assert_eq!(unique.len(), 9);
    }
}
