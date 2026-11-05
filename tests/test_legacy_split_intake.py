"""Real native exposure journals and legacy two-way intake, no mock migration."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from test_split_broker import request, server

BINARY = Path(__file__).resolve().parents[1] / "target/debug/examples/fixture_exposure"


def run(tmp_path: Path, **options: Any) -> dict[str, Any]:
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    process = subprocess.run(
        [str(BINARY), str(root)],
        input=json.dumps(options),
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    result: dict[str, Any] = json.loads(process.stdout)
    assert "error" not in result, result
    value: dict[str, Any] = result["ok"]
    assert value["idempotent"] is True and value["persisted"] is True
    assert value["assessment"]["independent_three_way"] is False
    assert value["assessment"]["allowed_destination"] == "development_only"
    assert value["admission"]["development"]["allowed"] is True
    assert value["admission"]["selection"]["allowed"] is False
    assert value["admission"]["holdout"]["allowed"] is False
    return value


def test_legacy_split_intake_missing_history_is_unknown_not_unseen(tmp_path: Path) -> None:
    result = run(tmp_path, history="missing", bucket="sealed", destination="development")
    assert len(result["assessment"]["unknown"]) == 1
    assert len(result["assessment"]["known_exposed"]) == 1
    assert "ok" in result["binding"]
    assert len(result["events"]) == 6
    imported = [e["body"] for e in result["events"] if e["body"]["stage"] == "imported"]
    assert any("unknown_legacy_history" in row["availability"] for row in imported)
    assert all(row["cost"]["cpu_ms"] is None for row in imported)
    assert all(row["shown"] == [] for row in imported)


def test_legacy_split_intake_partial_score_and_teacher_feedback_remain_exposure(
    tmp_path: Path,
) -> None:
    result = run(tmp_path, history="partial", destination="development", bucket="edit")
    assert len(result["assessment"]["known_exposed"]) == 2
    assert result["assessment"]["unknown"] == []
    assert len(result["events"]) == 9
    imported = [e for e in result["events"] if e["body"]["stage"] == "imported"]
    partial = next(e for e in imported if e["body"]["recipient"] == "old-score-feedback")
    teacher = next(e for e in imported if e["body"]["recipient"] == "old-teacher")
    assert partial["body"]["shown"] == ["summary"]
    assert partial["body"]["context"]["seed"] == 42
    assert partial["body"]["context"]["attempt"] == 1
    assert partial["body"]["outcome"] == "failed_or_partial"
    assert teacher["body"]["shown"] == ["feedback"]
    assert teacher["body"]["context"]["seed"] == 99
    assert teacher["body"]["context"]["retry_of"] == partial["id"]


@pytest.mark.parametrize("bucket", ["edit", "sealed"])
@pytest.mark.parametrize("history", ["missing", "partial"])
@pytest.mark.parametrize("role", ["selection", "holdout"])
def test_legacy_split_intake_two_way_buckets_cannot_be_renamed_independent(
    tmp_path: Path, bucket: str, history: str, role: str
) -> None:
    result = run(tmp_path, bucket=bucket, history=history, destination=role)
    assert result["binding"]["error"]["code"] == "SCOPE_VIOLATION"
    assert "independent" in result["binding"]["error"]["message"]


@pytest.mark.parametrize("role", ["selection", "holdout"])
def test_legacy_split_intake_rehashing_names_versions_nonce_and_group_ids_cannot_erase_history(
    tmp_path: Path, role: str
) -> None:
    result = run(tmp_path, bucket="edit", history="partial", destination=role, renamed=True)
    assert result["binding"]["error"]["code"] == "SCOPE_VIOLATION"
    assert len(result["events"]) == 9


def test_legacy_split_intake_renamed_exposed_data_may_only_remain_development(
    tmp_path: Path,
) -> None:
    result = run(
        tmp_path, bucket="edit", history="partial", destination="development", renamed=True
    )
    assert "ok" in result["binding"]
    assert result["assessment"]["independent_three_way"] is False


def test_legacy_split_intake_modern_scope_tracks_seed_retry_partial_and_failed_access(
    tmp_path: Path,
) -> None:
    with server(tmp_path) as controller:
        welcome = controller.grant("proposer")
        controller.ok("read", request=request(welcome), context={"seed": 17, "attempt": 0})
        events = controller.ok("events")
        reads = [
            e
            for e in events
            if e["body"]["stage"] == "completed" and e["body"]["shown"] == ["input"]
        ]
        assert len(reads) == 1
        first = reads[0]
        assert first["body"]["context"]["seed"] == 17
        assert first["body"]["context"]["model"] is None
        controller.ok(
            "read",
            request=request(welcome),
            context={"seed": 17, "attempt": 1, "retry_of": first["id"]},
        )
        controller.ok("corrupt_input")
        assert "error" in controller.send(
            "read", request=request(welcome), context={"seed": 17, "attempt": 2}
        )
        events = controller.ok("events")
        input_events = [e for e in events if e["body"]["intended"] == ["input"]]
        assert len(input_events) == 9
        retry = next(
            e
            for e in input_events
            if e["body"]["stage"] == "completed" and e["body"]["context"]["attempt"] == 1
        )
        assert retry["body"]["context"]["retry_of"] == first["id"]
        failed = next(e for e in input_events if e["body"]["stage"] == "failed")
        assert failed["body"]["shown"] == [] and failed["body"]["cost"]["cpu_ms"] is None
        assert (
            failed["body"]["subject"]["package_version"]
            and failed["body"]["subject"]["dataset_version"]
        )
        controller.ok("reopen")
        assert len(controller.ok("events")) == len(events)


def test_legacy_split_intake_summary_feedback_is_exposure_not_a_label_read(tmp_path: Path) -> None:
    with server(tmp_path) as controller:
        event = controller.ok("feedback", split="selection", value=0.25)
        assert event["body"]["shown"] == ["feedback", "summary"]
        assert event["body"]["subject"]["split"] == "selection"
        assert event["body"]["role"] == "proposer"
        assert event["body"]["context"]["seed"] == 17
        assert event["body"]["context"]["artifact"]
        assert "label" not in event["body"]["shown"]
        assert "error" in controller.send("feedback", split="holdout", value=0.25)
        assert "error" in controller.send(
            "feedback", split="selection", value={"secret_label": 1}, operation="bad-label"
        )
        controller.ok("final_phase")
        controller.ok("freeze")
        assert "error" in controller.send(
            "feedback", split="selection", value=0.5, operation="post-final-feedback"
        )
