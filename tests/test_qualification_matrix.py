"""Keeps docs/qualification-matrix.md, the public docs and the qualification policy honest."""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "docs" / "qualification-matrix.md"
COLUMNS = ["Operation", "Architecture", "Dtype", "Backend", "Shape policy", "Evidence", "Test"]
LEVELS = {"verified", "experimental", "fixture-only", "unsupported"}
# Rows at this scope make no claim about any model architecture.
TENSOR_SCOPE = "tensor"


def matrix_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    header: list[str] | None = None
    for line in MATRIX.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            header = None
            continue
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        if header is None:
            header = cells
            # The evidence-level legend is the only other table allowed in this file.
            assert header == COLUMNS or header[0] == "Level", f"unexpected matrix header: {header}"
        elif header == COLUMNS and not set(cells[0]) <= {"-", ":", " "}:
            assert len(cells) == len(COLUMNS), f"malformed row: {line}"
            rows.append(dict(zip(COLUMNS, cells, strict=True)))
    return rows


def find_test(reference: str) -> ast.FunctionDef:
    path, _, name = reference.partition("::")
    file = ROOT / path
    assert file.is_file(), f"{reference}: file does not exist"
    for node in ast.parse(file.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{reference}: test does not exist")


def has_qualification_marker(node: ast.FunctionDef) -> bool:
    return any(ast.unparse(d) == "pytest.mark.qualification" for d in node.decorator_list)


def test_matrix_has_rows_at_every_evidence_level() -> None:
    assert {row["Evidence"] for row in matrix_rows()} == LEVELS


@pytest.mark.parametrize("row", matrix_rows(), ids=lambda r: f"{r['Operation']}-{r['Architecture']}-{r['Dtype']}")
def test_matrix_row_is_backed_by_what_it_claims(row: dict[str, str]) -> None:
    level, test = row["Evidence"], row["Test"]
    assert level in LEVELS
    assert all(row[column] for column in COLUMNS if column != "Test")
    if level in ("verified", "fixture-only"):
        assert test not in ("", "none"), f"{level} row needs a test: {row}"
    if test not in ("", "none"):
        node = find_test(test)
        if level == "verified" and row["Architecture"] != TENSOR_SCOPE:
            # A model-level claim needs a test that cannot pass by being skipped.
            assert has_qualification_marker(node), f"{test} lacks @pytest.mark.qualification"
    if level == "verified":
        assert "fixture" not in row["Architecture"].lower()


def test_matrix_parser_rejects_a_missing_test() -> None:
    with pytest.raises(AssertionError):
        find_test("tests/test_qualification_matrix.py::test_that_does_not_exist")
    with pytest.raises(AssertionError):
        find_test("tests/no_such_file.py::test_anything")


@pytest.mark.parametrize(
    "phrase",
    [
        "Enterprise & Open-Source",
        "100% behavioral survival",
        "preserve 100% of its behavioral gain",
        "feedback_tools_proven_necessary is True",
        "are essential compared to",
        "Measured Results on Open-Weight Models",
        "All 20 audit requirements are enforced",
        "sandbox_cleaned_up",
    ],
)
def test_retired_claims_stay_out_of_public_docs(phrase: str) -> None:
    for name in ("README.md", "docs/weight-programming-studio.md"):
        assert phrase not in (ROOT / name).read_text(encoding="utf-8"), f"{phrase!r} in {name}"


def test_public_docs_state_limitations_and_link_the_matrix() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "## Limitations" in readme
    assert "docs/qualification-matrix.md" in readme
    assert "qualification-matrix.md" in (ROOT / "docs/weight-programming-studio.md").read_text(encoding="utf-8")


def run_policy(tmp_path: Path, body: str, *, qualification: bool) -> subprocess.CompletedProcess[str]:
    shutil.copy(ROOT / "tests" / "conftest.py", tmp_path / "conftest.py")
    (tmp_path / "test_policy_sample.py").write_text(body, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "NF_QUALIFICATION"}
    if qualification:
        env["NF_QUALIFICATION"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--strict-markers", "-p", "no:cacheprovider", "--rootdir", str(tmp_path), str(tmp_path)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


MARKED_SKIP = """
import pytest

@pytest.mark.qualification
def test_needs_checkpoint():
    pytest.skip("checkpoint missing")
"""
UNMARKED_SKIP = """
import pytest

def test_platform_specific():
    pytest.skip("not on this platform")
"""
MODULE_SKIP = """
import pytest
pytest.importorskip("a_package_that_is_not_installed")

def test_never_collected():
    pass
"""


def test_skipped_qualification_test_is_reported_as_skipped_in_normal_runs(tmp_path: Path) -> None:
    result = run_policy(tmp_path, MARKED_SKIP, qualification=False)
    assert result.returncode == 0, result.stdout
    assert "1 skipped" in result.stdout


def test_skipped_qualification_test_fails_a_qualification_run(tmp_path: Path) -> None:
    result = run_policy(tmp_path, MARKED_SKIP, qualification=True)
    assert result.returncode == 1, result.stdout
    assert "1 failed" in result.stdout
    assert "checkpoint missing" in result.stdout


def test_unmarked_skip_stays_a_skip_in_a_qualification_run(tmp_path: Path) -> None:
    result = run_policy(tmp_path, UNMARKED_SKIP, qualification=True)
    assert result.returncode == 0, result.stdout


def test_module_skipped_at_collection_fails_a_qualification_run(tmp_path: Path) -> None:
    assert run_policy(tmp_path, MODULE_SKIP, qualification=False).returncode != 1
    assert run_policy(tmp_path, MODULE_SKIP, qualification=True).returncode != 0
