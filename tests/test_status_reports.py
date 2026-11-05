"""T040 positive native/SDK/CLI snapshot integration; local CPU fixtures only."""

import inspect
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
from neural_foundry import _native
from neural_foundry import status_reports as reports


def markdown_data(markdown: str) -> dict[str, Any]:
    return {
        match[1]: json.loads(match[3])
        for match in re.finditer(
            r"^## (\w+)\n\n(`{3,})json\n(.*?)\n\2$", markdown, re.MULTILINE | re.DOTALL
        )
    }


def test_python_has_no_sqlite_reader_or_accounting_fold() -> None:
    source = inspect.getsource(reports)
    assert "import sqlite3" not in source
    assert "def _project_accounting" not in source
    assert "def _capture(" not in source
    assert "_native.capture_status_snapshot" in source


def test_native_handle_cannot_be_constructed_from_authority_dto() -> None:
    with pytest.raises(TypeError):
        _native.CommittedStatusSnapshot()
    with pytest.raises(TypeError):
        _native.CommittedStatusSnapshot({"accepted": True})
    with pytest.raises(TypeError):
        reports.CommittedSnapshot()
    for value in ({"accepted": True}, object.__new__(reports.CommittedSnapshot)):
        with pytest.raises(TypeError):
            reports.render_json(value)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            reports.render_markdown(value)  # type: ignore[arg-type]


def test_missing_capture_never_creates_source(tmp_path: Path) -> None:
    missing = tmp_path / "missing" / "events.sqlite3"
    with pytest.raises(reports.SnapshotError) as caught:
        reports.capture_snapshot(missing, "sha256:" + "0" * 64)
    assert isinstance(caught.value.__cause__, _native.NativeError)
    assert caught.value.__cause__.code == "IO"
    assert not missing.parent.exists()


def test_capture_never_rewrites_existing_foreign_data(tmp_path: Path) -> None:
    path = tmp_path / "events.sqlite3"
    original = b"not SQLite; do not initialize or reinterpret"
    path.write_bytes(original)
    with pytest.raises(reports.SnapshotError) as caught:
        reports.capture_snapshot(path, "sha256:" + "0" * 64)
    assert isinstance(caught.value.__cause__, _native.NativeError)
    assert caught.value.__cause__.code != "UNSUPPORTED_CAPABILITY"
    assert path.read_bytes() == original


def test_native_digest_errors_are_not_replaced_by_python_parsing(tmp_path: Path) -> None:
    with pytest.raises(reports.SnapshotError) as caught:
        reports.capture_snapshot(tmp_path / "missing.db", "latest")
    assert isinstance(caught.value.__cause__, _native.NativeError)
    assert "Ledger::read_committed_status" not in str(caught.value)


def test_facade_delegates_both_formats_to_same_handle(monkeypatch: Any, tmp_path: Path) -> None:
    # Deliberate boundary double, NOT a committed native capture or security test.
    calls: list[tuple[str, str]] = []

    class HandleDouble:
        def render_json(self) -> str:
            return '{"authority":{"accepted":false},"costs":null}\n'

        def render_markdown(self) -> str:
            return "native markdown sentinel\n"

    handle = HandleDouble()

    def capture(path: str, digest: str) -> HandleDouble:
        calls.append((path, digest))
        return handle

    monkeypatch.setattr(_native, "CommittedStatusSnapshot", HandleDouble)
    monkeypatch.setattr(_native, "capture_status_snapshot", capture)
    path = tmp_path / "explicit.db"
    snapshot = reports.capture_snapshot(path, "sha256:" + "a" * 64)
    assert calls == [(str(path), "sha256:" + "a" * 64)]
    copy = snapshot.to_dict()
    copy["authority"]["accepted"] = True
    assert snapshot.to_dict()["authority"]["accepted"] is False
    assert reports.render_json(snapshot) == handle.render_json()
    assert reports.render_markdown(snapshot) == handle.render_markdown()
    assert reports.render_formats(snapshot) == {
        "json": handle.render_json(), "markdown": handle.render_markdown(),
    }
    assert len(calls) == 1
    assert not path.exists()


@pytest.mark.parametrize("actor", ["proposer", "final_evaluator"])
def test_required_real_coordinator_capture_and_cpu_fixture_retention(
    tmp_path: Path, actor: str
) -> None:
    from test_worker_broker import call, server

    with server(tmp_path, actor=actor) as (channel, _, _, welcome):
        path = tmp_path / actor / "ledger/events.sqlite3"
        source_before = path.read_bytes()
        before = reports.capture_snapshot(path, welcome["contract"])
        assert path.read_bytes() == source_before
        before_json = reports.render_json(before)
        reply = call(channel, welcome, 0, {
            "type": "infer", "model": "target", "task": welcome["tasks"][0],
        })
        assert reply["ok"] and reply["result"]["output"] == [5.0, 11.0]
        after = reports.capture_snapshot(path, welcome["contract"])
        data = after.to_dict()
        totals = data["budgets"]["aggregate"]
        assert totals["spent"]["evaluations"] == 1
        assert totals["spent"]["gpu_ms"] == 0
        assert totals["unknown_charged"]["cpu_ms"] > 0
        assert totals["held"]["workers"] == 1  # root pool, not pool plus child
        assert data["split_evidence"]
        assert data["authority"]["accepted"] is False
        formats = reports.render_formats(after)
        assert markdown_data(formats["markdown"]) == json.loads(formats["json"])
        assert data["snapshot"]["consistency"] == "single committed coordinator read transaction"
        assert data["controls"]["status"] == "unknown"
        assert all(gate["status"] == "unknown" for gate in data["gym_gates"])
        assert data["measurement_settings"]["actual_measurements"]["status"] == "unavailable"
        for arm in data["baselines"]:
            if not arm["configuration"]["enabled"]:
                assert arm["status"] == "not_run"
                assert arm["scores"] is None and arm["costs"] is None

        binary = Path(__file__).resolve().parents[1] / "target/debug/nf"
        assert binary.is_file(), "cargo build --offline -p nf-cli is required"
        database_before_cli = path.read_bytes()
        cli = subprocess.run(
            [str(binary), "study", "report", str(path), welcome["contract"],
             "--format", "both"], capture_output=True, text=True, check=True,
        )
        bundle = json.loads(cli.stdout)
        assert cli.stderr == ""
        cli_data = json.loads(bundle["json"])
        assert markdown_data(bundle["markdown"]) == cli_data
        assert cli_data["snapshot"] == data["snapshot"]
        assert cli_data["budgets"]["aggregate"]["spent"] == totals["spent"]
        assert path.read_bytes() == database_before_cli

        status = subprocess.run(
            [str(binary), "study", "status", str(path), welcome["contract"]],
            capture_output=True, text=True, check=True,
        )
        assert json.loads(status.stdout)["snapshot"] == data["snapshot"]
        denied = subprocess.run(
            [str(binary), "study", "report", str(path), welcome["contract"],
             "--format", "public"], capture_output=True, text=True, check=False,
        )
        assert denied.returncode == 2 and denied.stdout == ""
        assert json.loads(denied.stderr)["error"]["code"] == "INVALID_VALUE"
        assert reports.render_json(before) == before_json
        assert before.to_dict()["budgets"]["aggregate"]["spent"]["evaluations"] == 0


def test_real_final_fixture_adapter_across_sdk_and_cli(tmp_path: Path) -> None:
    """Real Rust-created final evidence; no inserted scientific authority DTOs."""
    project = Path(__file__).resolve().parents[1]
    cargo = Path.home() / ".cargo/bin/cargo"
    output = tmp_path / "native"
    env = dict(os.environ, NF_T040_TEST_FIXTURE_OUTPUT=str(output))
    result = subprocess.run(
        [str(cargo), "test", "--offline", "-p", "nf-experiments", "--test",
         "status_reports", "final_evidence_tests::status_reports_real_final_positive_failed_and_incomplete_are_distinct",
         "--", "--exact", "--test-threads=1"],
        cwd=project, env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for name, status, arm_count in (("positive", "passed", 4), ("failed", "failed", 4),
                                    ("incomplete", "incomplete", 0)):
        directory = output / name
        database = directory / "events.sqlite3"
        contract = (directory / "contract.txt").read_text()
        original = database.read_bytes()
        snapshot = reports.capture_snapshot(database, contract)
        data = snapshot.to_dict()
        evidence = data["final_evidence"]["records"][0]
        assert evidence["recorded_status"] == status
        assert len(evidence["receipt_links"]) == arm_count
        assert data["authority"]["accepted"] is False
        assert data["authority"]["promotion_authorized"] is False
        assert "unresolved_hard_wall_and_cpu" in evidence["strict_t038_publication"]
        limitations = {row["id"]: row for row in data["known_limitations"]}
        assert limitations["t038_completion_publication"]["status"] == "unresolved"
        formats = reports.render_formats(snapshot)
        assert markdown_data(formats["markdown"]) == json.loads(formats["json"])
        for forbidden in ('"outputs"', '"samples"', '"observations"'):
            assert forbidden not in formats["json"]
        cli = subprocess.run(
            [str(project / "target/debug/nf"), "study", "report", str(database), contract,
             "--format", "both"], capture_output=True, text=True, check=True,
        )
        bundle = json.loads(cli.stdout)
        cli_data = json.loads(bundle["json"])
        assert cli_data["final_evidence"] == data["final_evidence"]
        assert markdown_data(bundle["markdown"]) == cli_data
        assert database.read_bytes() == original
