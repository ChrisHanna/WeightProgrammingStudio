from __future__ import annotations

import json
import os
import signal
import subprocess
from pathlib import Path
from typing import Any

import pytest
from neural_foundry import nemo_gym
from neural_foundry.nemo_gym import NeMoGymError


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_zero_valued_task_id_is_not_treated_as_missing() -> None:
    assert nemo_gym._task_id({"uuid": 0, "task_id": "fallback"}) == 0


def test_duplicate_materialized_inputs_are_rejected_before_dpo_pairing(tmp_path: Path) -> None:
    materialized = tmp_path / "rollouts_materialized_inputs.jsonl"
    rollouts = tmp_path / "rollouts.jsonl"
    row = {
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "task_input": {"uuid": "task-0"},
    }
    _write_jsonl(materialized, [row, row])
    _write_jsonl(rollouts, [])
    _write_jsonl(tmp_path / "rollouts_failures.jsonl", [])

    with pytest.raises(NeMoGymError, match="duplicate materialized-input identity"):
        nemo_gym.import_mcqa_preferences(
            None,
            materialized_inputs_jsonl=materialized,
            rollouts_jsonl=rollouts,
        )


def test_timeout_cleanup_escalates_to_process_group_after_leader_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name == "nt":
        pytest.skip("Process-group escalation is exercised under WSL")
    calls = []

    class Process:
        pid = 123
        returncode = None
        waits = 0

        def poll(self) -> int:
            return 0

        def wait(self, timeout: float | None = None) -> int:
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("gym", timeout)
            self.returncode = -signal.SIGKILL
            return self.returncode

    monkeypatch.setattr(nemo_gym.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    nemo_gym._stop_process_tree(Process())

    assert calls == [(123, signal.SIGTERM), (123, signal.SIGKILL)]