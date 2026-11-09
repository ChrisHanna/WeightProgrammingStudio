from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from neural_foundry.gym_artifacts import CandidateRef
from neural_foundry.gym_job import GymJobError
from neural_foundry.gym_service import _document_digest
from neural_foundry.provenance import resolve_checkpoint
from neural_foundry.vllm_container_adapter import VllmCandidateServer
from neural_foundry.vllm_candidate_files import verify_vllm_candidate_files


def _reference(tmp_path: Path) -> CandidateRef:
    root = tmp_path / "base"
    root.mkdir()
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        (root / name).write_text(name)
    files = resolve_checkpoint(root).file_digests
    digest = _document_digest({"model_id": "test/model", "revision": "a" * 40, "files": files})
    return CandidateRef("test/model", "full", digest, digest, digest, str(root), "base", "unchanged_base", "qwen3")


def test_container_adapter_is_network_isolated_and_never_claims_loaded_identity(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace
    transformers = pytest.importorskip("transformers")
    tokenizer = SimpleNamespace(chat_template="template", special_tokens_map={}, get_vocab=lambda: {"token": 1},
                                 encode=lambda text, **kwargs: [1, 2, 3], apply_chat_template=lambda messages, **kwargs: "rendered")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    reference = _reference(tmp_path)
    calls = []
    container_id = "a" * 64
    completion = {"finish_reason": "stop", "text": '{"answer":1}'}

    def docker(command, **kwargs):
        calls.append((command, kwargs.get("input")))
        if command[1] == "create":
            output = container_id.encode() + b"\n"
        elif command[1] == "exec" and command[-1] == "/v1/completions":
            payload = json.loads(kwargs["input"])
            assert payload["prompt"] == [1, 2, 3] and payload["add_special_tokens"] is False
            assert "messages" not in payload
            output = json.dumps({"choices": [completion]}).encode()
        else:
            output = b""
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr=b"")

    monkeypatch.setattr(subprocess, "run", docker)
    server = VllmCandidateServer(reference, revision="a" * 40)
    with server as adapter:
        result = adapter.infer({"format": "json_extraction_v1", "messages": [{"role": "user", "content": "extract"}],
                                "output_schema": {"type": "object", "properties": {}, "additionalProperties": False}})
        assert result.response == {"answer": 1}
        assert adapter.loaded_identity()["local_loaded_files_verified"] is False
        assert adapter.loaded_identity()["served_model_identity_verified"] is False
        assert "state_digest" not in adapter.loaded_identity()
        assert result.interaction["prompt_renderer_digest"] == adapter.inference_config["prompt_renderer_digest"]
        completion["finish_reason"] = "length"
        with pytest.raises(GymJobError, match="incomplete"):
            adapter.infer({"format": "chat_messages_v1", "messages": [{"role": "user", "content": "extract"}]})
    create = calls[0][0]
    assert "--init" in create
    assert create[create.index("--network") + 1] == "none"
    assert create[create.index("--log-driver") + 1] == "none"
    assert "VLLM_ALLOW_INSECURE_SERIALIZATION=0" in create
    assert "VLLM_ALLOW_INSECURE_SERIALIZATION=1" not in create
    assert "--disable-uvicorn-access-log" in create and "--no-enable-log-outputs" in create
    assert "--no-enable-log-requests" in create and "VLLM_USE_FLASHINFER_SAMPLER=0" in create
    assert calls[-2][0][1:] == ["stop", "--timeout", "20", container_id]
    assert calls[-1][0][1:] == ["rm", container_id]
    with pytest.raises(GymJobError, match="not running"):
        adapter.infer({"format": "chat_messages_v1", "messages": []})


def test_tampered_checkpoint_never_starts_docker(tmp_path, monkeypatch) -> None:
    reference = _reference(tmp_path)
    (Path(reference.artifact_path) / "model.safetensors").write_text("changed")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("Docker should not start"))
    with pytest.raises(GymJobError, match="differ"):
        with VllmCandidateServer(reference, revision="a" * 40):
            pass


def test_cleanup_accepts_only_confirmed_absence(tmp_path, monkeypatch) -> None:
    server = VllmCandidateServer(_reference(tmp_path), revision="a" * 40)
    server._container = "a" * 64

    def docker(command, **kwargs):
        error = b"No such object: missing" if command[1] == "inspect" else b"container already removed"
        return subprocess.CompletedProcess(command, 1, stdout=b"", stderr=error)

    monkeypatch.setattr(subprocess, "run", docker)
    server._cleanup()
    assert server._container is None


def test_cleanup_failure_does_not_mask_original_probe_failure(tmp_path, monkeypatch) -> None:
    server = VllmCandidateServer(_reference(tmp_path), revision="a" * 40)
    server._container = "a" * 64

    def docker(command, **kwargs):
        status = 0 if command[1] == "inspect" else 1
        return subprocess.CompletedProcess(command, status, stdout=b"[]", stderr=b"removal failed")

    monkeypatch.setattr(subprocess, "run", docker)
    original = ValueError("original identity probe failed")
    server.__exit__(ValueError, original, None)
    assert "original identity probe failed" in str(original)
    assert any("removal failed" in note for note in original.__notes__)
    assert server._container == "a" * 64


def test_cleanup_uses_bounded_force_fallback_after_stop_timeout(tmp_path, monkeypatch) -> None:
    server = VllmCandidateServer(_reference(tmp_path), revision="a" * 40)
    server._container = "a" * 64
    calls = []

    def docker(command, **kwargs):
        calls.append(command)
        if command[1] == "stop":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(command, 0, stdout=b"removed", stderr=b"")

    monkeypatch.setattr(subprocess, "run", docker)
    server._cleanup()
    assert calls[-1][1:] == ["rm", "--force", "a" * 64]
    assert server._container is None


def test_real_vllm_container_serves_verified_candidate() -> None:
    if os.environ.get("NF_VLLM_LIVE") != "1":
        pytest.skip("set NF_VLLM_LIVE=1 for the GPU container smoke test")
    model_id = "Qwen/Qwen3-0.6B"
    revision = "c1899de289a04d12100db370d81485cdf75e47ca"
    checkpoint = resolve_checkpoint(model_id, revision=revision)
    digest = _document_digest({"model_id": model_id, "revision": revision, "files": checkpoint.file_digests})
    reference = CandidateRef(model_id, "full", digest, digest, digest, str(checkpoint.path),
                             "qwen-base", "unchanged_base", "qwen3")
    server = VllmCandidateServer(reference, revision=revision, max_new_tokens=32, timeout_seconds=300)
    with server as adapter:
        response = adapter.infer({"format": "json_extraction_v1",
                                  "messages": [{"role": "user", "content": 'Return the object {"answer":1}.'}],
                                  "output_schema": {"type": "object", "properties": {"answer": {"type": "integer"}},
                                                    "required": ["answer"], "additionalProperties": False}})
        assert isinstance(response.response, dict) and type(response.response.get("answer")) is int
        assert adapter.loaded_identity()["served_model_identity_verified"] is False


def test_real_vllm_container_serves_published_sft_candidate() -> None:
    artifact = os.environ.get("NF_VLLM_SFT_ARTIFACT")
    if not artifact:
        pytest.skip("set NF_VLLM_SFT_ARTIFACT to a published full SFT candidate")
    root = Path(artifact)
    manifest = json.loads((root / "artifact_manifest.json").read_bytes())
    candidate = manifest["metadata"]["candidate"]
    reference = CandidateRef(manifest["model_id"], manifest["kind"], manifest["base_digest"], manifest["model_digest"],
                             candidate["artifact_digest"], str(root), candidate["candidate_id"], candidate["method_id"], "qwen3")
    with VllmCandidateServer(reference, revision="c1899de289a04d12100db370d81485cdf75e47ca",
                             max_new_tokens=32, timeout_seconds=300) as adapter:
        response = adapter.infer({"format": "json_extraction_v1",
                                  "messages": [{"role": "user", "content": 'Return the object {"answer":1}.'}],
                                  "output_schema": {"type": "object", "properties": {"answer": {"type": "integer"}},
                                                    "required": ["answer"], "additionalProperties": False}})
        assert isinstance(response.response, dict) and type(response.response.get("answer")) is int
        assert adapter.reference == reference
        assert adapter.loaded_identity()["served_model_identity_verified"] is False


def test_real_vllm_nonserving_diagnostic() -> None:
    if os.environ.get("NF_VLLM_DIAGNOSTIC") != "1":
        pytest.skip("set NF_VLLM_DIAGNOSTIC=1 for the isolated startup probe")
    model_id = "Qwen/Qwen3-0.6B"
    revision = "c1899de289a04d12100db370d81485cdf75e47ca"
    checkpoint = resolve_checkpoint(model_id, revision=revision)
    digest = _document_digest({"model_id": model_id, "revision": revision, "files": checkpoint.file_digests})
    reference = CandidateRef(model_id, "full", digest, digest, digest, str(checkpoint.path),
                             "qwen-base", "unchanged_base", "qwen3")
    files = verify_vllm_candidate_files(reference, revision=revision)
    name = "nf-vllm-probe-" + uuid4().hex
    with tempfile.TemporaryDirectory(prefix="nf-vllm-probe-") as temporary:
        root = Path(temporary)
        for filename in files:
            target = root / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(reference.artifact_path) / filename, target)
        from neural_foundry.gym_artifacts import artifact_files
        from neural_foundry.vllm_container_adapter import VLLM_IMAGE
        assert artifact_files(root) == files
        command = ["docker", "run", "--rm", "--name", name, "--gpus", "all", "--network", "none", "--read-only",
                   "--shm-size", "1g", "--tmpfs", "/tmp:rw,exec,nosuid,nodev,size=2g", "--mount",
                   f"type=bind,src={root},dst=/model,readonly", "--env", "VLLM_ALLOW_INSECURE_SERIALIZATION=0",
                   "--env", "VLLM_USE_FLASHINFER_SAMPLER=0", "--env", "HOME=/tmp", "--env", "DO_NOT_TRACK=1",
                   "--env", "VLLM_NO_USAGE_STATS=1",
                   "--env", "HF_HOME=/tmp/hf", "--env", "XDG_CACHE_HOME=/tmp/cache", "--env", "VLLM_CACHE_ROOT=/tmp/vllm",
                   "--env", "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor", "--env", "TRITON_CACHE_DIR=/tmp/triton",
                   "--env", "HF_HUB_OFFLINE=1", "--env", "TRANSFORMERS_OFFLINE=1", "--entrypoint", "python3", VLLM_IMAGE,
                   "-c", "from vllm import LLM; LLM(model='/model',dtype='bfloat16',max_model_len=512,"
                         "gpu_memory_utilization=0.25,enforce_eager=True); print('ready')"]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=300)
            assert result.returncode == 0 and "ready" in result.stdout, (result.stdout + result.stderr)[-10000:]
        finally:
            subprocess.run(["docker", "rm", "--force", name], capture_output=True, timeout=30)