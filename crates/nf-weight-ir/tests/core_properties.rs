#[path = "common/mod.rs"]
#[allow(dead_code)]
mod common;
#[path = "common/compiler_fixture.rs"]
#[allow(dead_code)]
mod compiler_fixture;
#[path = "support/parser_smoke.rs"]
mod parser_smoke;
#[test]
fn core_properties_bounded_parser_mutations_keep_identities_or_fail_closed() {
    let counts = parser_smoke::run();
    assert_eq!(counts.cases, 4096);
}
