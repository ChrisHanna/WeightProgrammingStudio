// The fixture is a fixed, statically linked computation image, not a user program.
// A missing static toolchain is a build error: never substitute a dynamic worker.
// This native-only ABI must never embed a build-host image into another target.
fn main() {
    let host = std::env::var("HOST").expect("Cargo HOST");
    let target = std::env::var("TARGET").expect("Cargo TARGET");
    assert!(host == target
        && std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux")
        && matches!(std::env::var("CARGO_CFG_TARGET_ARCH").as_deref(), Ok("x86_64" | "aarch64")),
        "fixed static fixture requires native little-endian Linux x86_64/aarch64; cross compilation is not qualified");
    let source = "src/coordinator/final_fixture/linear_worker.c";
    println!("cargo:rerun-if-changed={source}");
    let output = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap())
        .join("nf-final-linear-worker");
    assert!(
        std::process::Command::new("/usr/bin/cc")
            .args([
                "-std=c11",
                "-O2",
                "-static",
                "-no-pie",
                "-fno-fast-math",
                "-ffp-contract=off",
                "-Wall",
                "-Wextra",
                "-Werror",
                source,
                "-o"
            ])
            .arg(output)
            .status()
            .expect("static fixture compiler")
            .success(),
        "fixed static fixture compilation failed"
    );
}
