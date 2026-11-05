use std::process::Command;

#[test]
fn package_smoke_cli_version() {
    let output = Command::new(env!("CARGO_BIN_EXE_nf"))
        .arg("--version")
        .output()
        .unwrap();
    assert!(output.status.success());
    assert_eq!(String::from_utf8(output.stdout).unwrap(), "nf 0.1.0\n");
}

#[test]
fn package_smoke_unknown_command_is_not_executed() {
    let output = Command::new(env!("CARGO_BIN_EXE_nf"))
        .args(["study", "run"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
}
