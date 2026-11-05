#[path = "../tests/common/mod.rs"]
#[allow(dead_code)]
mod common;
#[path = "../tests/common/compiler_fixture.rs"]
#[allow(dead_code)]
mod compiler_fixture;
#[path = "../tests/support/parser_smoke.rs"]
mod parser_smoke;
fn main() {
    println!("{:?}", parser_smoke::run());
}
