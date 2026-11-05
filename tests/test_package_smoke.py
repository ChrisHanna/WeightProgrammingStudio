"""Exercise the installed native extension, not a Python replacement."""

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import neural_foundry
from neural_foundry import _native

ROOT = Path(__file__).resolve().parents[1]


def test_package_smoke_native_extension() -> None:
    assert neural_foundry.__version__ == "0.1.0"
    assert _native.version() == neural_foundry.__version__
    assert any(str(_native.__file__).endswith(suffix) for suffix in (".so", ".pyd"))
    assert len(neural_foundry.workspace_crates()) == 9


def test_package_smoke_workspace_dependency_direction() -> None:
    names = set(neural_foundry.workspace_crates())
    metadata = {}
    for name in names:
        manifest = ROOT / "crates" / name / "Cargo.toml"
        metadata[name] = tomllib.loads(manifest.read_text())
    for name in names - {"nf-python"}:
        assert "pyo3" not in metadata[name].get("dependencies", {})
        assert "nf-python" not in metadata[name].get("dependencies", {})
    assert not (set(metadata["nf-core"].get("dependencies", {})) & names)


def test_package_smoke_import_needs_no_research_stack() -> None:
    code = (
        "import json, sys, neural_foundry; "
        "print(json.dumps([name for name in "
        "('torch', 'transformers', 'numpy', 'requests') if name in sys.modules]))"
    )
    result = subprocess.run([sys.executable, "-I", "-c", code], check=True, capture_output=True)
    assert json.loads(result.stdout) == []
