from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

from neural_foundry import serving, vllm_qualification


def test_vllm_serving_rejects_insecure_serialization(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    with pytest.raises(serving.ServingRuntimeUnavailableError, match="forbids insecure serialization"):
        with serving._vllm_startup_env():
            pytest.fail("insecure serving started")
    server = vllm_qualification._VllmServer(tmp_path / "python", tmp_path, 0.1, tmp_path / "vllm.log")
    with pytest.raises(vllm_qualification.VllmQualificationFailure) as failure:
        server.__enter__()
    assert failure.value.code == "VLLM_INSECURE_SERIALIZATION_FORBIDDEN"
    assert not (tmp_path / "vllm.log").exists()
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "0")
    with serving._vllm_startup_env():
        assert os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] == "0"
    monkeypatch.setitem(sys.modules, "vllm.envs", SimpleNamespace(VLLM_ALLOW_INSECURE_SERIALIZATION=True))
    with pytest.raises(serving.ServingRuntimeUnavailableError, match="forbids insecure serialization"):
        with serving._vllm_startup_env():
            pytest.fail("cached insecure vLLm settings were accepted")