from __future__ import annotations

import os

from neural_foundry import gym_service
from neural_foundry.gym_service import GymStudioService


def test_state_persistence_falls_back_without_fchmod_or_directory_fsync(
    tmp_path, monkeypatch,
) -> None:
    service = GymStudioService(state_dir=tmp_path)
    document = {"kind": "windows-state-test", "value": "durable"}
    identity = "a" * 64
    chmod_calls = []
    fsync_calls = []
    original_chmod = os.chmod
    original_fsync = os.fsync

    def record_chmod(path, mode):
        chmod_calls.append((path, mode))
        return original_chmod(path, mode)

    def record_fsync(descriptor):
        fsync_calls.append(descriptor)
        return original_fsync(descriptor)

    monkeypatch.delattr(gym_service.os, "fchmod")
    monkeypatch.setattr(gym_service, "_WINDOWS", True)
    monkeypatch.setattr(gym_service.os, "chmod", record_chmod)
    monkeypatch.setattr(gym_service.os, "fsync", record_fsync)

    service._persist_state("portability", identity, document)

    state_path = tmp_path / "portability" / f"{identity}.json"
    assert service._load_state("portability", identity) == document
    assert any(path.name.startswith(".nf-gym-state-") and mode == 0o600 for path, mode in chmod_calls)
    assert len(fsync_calls) == 1
    assert state_path.is_file()