"""M1 acceptance uses actual offline binaries and a clean installed wheel."""

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import neural_foundry as nf
import pytest
from neural_foundry.demo import run

ROOT = Path(__file__).resolve().parents[1]


def test_m1_vertical_slice_native_without_python_executes_patch_export_and_restore(
    tmp_path: Path,
) -> None:
    binary = ROOT / "target/debug/examples/offline_direct"
    assert binary.is_file(), "cargo build -p nf-cli --example offline_direct is required"
    completed = subprocess.run(
        [
            str(binary),
            str(ROOT / "python/neural_foundry/data/m1_fixture.json"),
            str(tmp_path / "store"),
        ],
        env={"PATH": ""},
        check=True,
        capture_output=True,
    )
    proof = json.loads(completed.stdout)
    assert proof["values"] == [2, 4, 6, 8]
    assert proof["changed_parameters"] == 4
    assert proof["export"]["candidate"] == proof["replayed_candidate"]
    assert proof["export"]["transaction"]["before_parameters"] == proof["restored_source"]
    assert proof["source_preserved"]
    assert proof["promotion_status"] == "not_evaluated"
    assert not (tmp_path / "active").exists()


def test_m1_vertical_slice_packaged_python_fixture_is_real_native_work() -> None:
    result = run()
    assert result["values"] == [2.0, 4.0, 6.0, 8.0]
    assert result["source_preserved"] and result["snapshot_parity"]
    assert result["candidate"] == result["replayed_candidate"]
    assert result["receipt"]["changed_parameters"] == 4
    assert result["promotion_status"] == "not_evaluated"


def test_m1_vertical_slice_capability_scope_agrees_and_has_no_research_claim() -> None:
    completed = subprocess.run(
        [str(ROOT / "target/debug/nf"), "core", "capabilities"], check=True, capture_output=True
    )
    manifest = nf.capabilities()
    assert manifest == json.loads(completed.stdout)["result"]
    assert manifest["evidence_scope"] == "software_fixture_tests_only"
    assert manifest["qualified_on_this_host"] is True
    assert manifest["backend"]["dtypes"] == ["f32", "f64"]
    assert manifest["model_factories"] == [
        {"id": "fixture.cpu", "version": "1", "smoke_adapter": "nf.linear_fixture.v1"},
        {
            "id": "nf.arithmetic.relu",
            "version": "1",
            "smoke_adapter": "nf.arithmetic_relu.v1",
            "dtypes": ["f64"],
            "scope": "synthetic_integer_pairs_minus8_to8",
        },
    ]
    assert len(manifest["operations"]) == 8
    assert "scientific_acceptance" in manifest["unavailable"]
    assert "study_execution" in manifest["unavailable"]
    assert "generated_gym_execution" in manifest["unavailable"]


def test_m1_vertical_slice_locked_dependency_notices_are_present_and_current() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_dependencies.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    assert b"notices match" in completed.stdout
    inventory = json.loads((ROOT / "licenses/dependencies.json").read_bytes())
    notices = (ROOT / "python/neural_foundry/data/THIRD_PARTY_NOTICES.txt").read_text()
    lock = tomllib.loads((ROOT / "Cargo.lock").read_text())
    resolved = {
        (p["name"], p["version"])
        for p in lock["package"]
        if p.get("source") == "registry+https://github.com/rust-lang/crates.io-index"
    }
    reviewed = {(p["name"], p["version"]) for p in inventory["packages"]}
    assert reviewed == resolved
    assert "manifest_path" not in json.dumps(inventory)
    for package in inventory["packages"]:
        assert package["name"] in notices
        assert package["license_files"]


@pytest.fixture(scope="session")
def m1_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    configured = os.environ.get("NF_TEST_WHEEL")
    if configured:
        wheel = Path(configured).resolve(strict=True)
        assert wheel.suffix == ".whl"
        return wheel
    output = tmp_path_factory.mktemp("m1-built-wheel")
    env = os.environ.copy()
    env.pop("CONDA_PREFIX", None)
    env["CARGO_BUILD_JOBS"] = "2"
    env["PYO3_PYTHON"] = sys.executable
    subprocess.run(
        [
            sys.executable,
            "-m",
            "maturin",
            "build",
            "--locked",
            "--interpreter",
            sys.executable,
            "--out",
            str(output),
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
    )
    wheels = list(output.glob("*.whl"))
    assert len(wheels) == 1
    return wheels[0]


def test_m1_vertical_slice_actual_wheel_in_clean_offline_environment(
    m1_wheel: Path, tmp_path: Path
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/verify_wheel.py"),
            str(m1_wheel),
            str(tmp_path / "clean-wheel"),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    proof = json.loads(completed.stdout)
    assert proof["module_entrypoint_passed"] is True
    assert proof["proof"]["result"]["source_preserved"]
    assert proof["proof"]["result"]["values"] == [2.0, 4.0, 6.0, 8.0]
    assert Path(proof["proof"]["module"]).is_relative_to(Path(proof["isolated_environment"]))
