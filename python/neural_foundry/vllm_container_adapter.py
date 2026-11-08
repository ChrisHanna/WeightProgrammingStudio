"""Network-isolated vLLM serving for verified full-model candidate files."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from .gym_artifacts import CandidateRef, artifact_files
from .gym_job import GymJobError
from .gym_lifecycle import AdapterIdentity, ModelAdapter
from .gym_service import CandidateIdentity, ExecutionResponse, _document_digest
from .provenance import parse_strict_json, sha256_file
from .vllm_candidate_files import verify_vllm_candidate_files


VLLM_IMAGE = "sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3"
_HTTP_CLIENT = (
    "import sys, urllib.request\n"
    "body = sys.stdin.buffer.read()\n"
    "req = urllib.request.Request('http://127.0.0.1:8000' + sys.argv[1], "
    "data=body if body else None, headers={'Content-Type':'application/json'})\n"
    "with urllib.request.urlopen(req, timeout=60) as response: sys.stdout.buffer.write(response.read())\n"
)


class VllmCandidateServer:
    def __init__(self, reference: CandidateRef, *, revision: str, image: str = VLLM_IMAGE,
                 context_tokens: int = 512, max_new_tokens: int = 64, timeout_seconds: float = 240) -> None:
        if os.name == "nt":
            raise GymJobError("network-isolated vLLM serving requires a Linux Docker runtime")
        if (not isinstance(image, str) or len(image) != 71 or not image.startswith("sha256:")
            or any(char not in "0123456789abcdef" for char in image[7:])):
            raise GymJobError("vLLM image must be pinned by its immutable sha256 ID")
        if (type(context_tokens) is not int or type(max_new_tokens) is not int or context_tokens < 1
            or max_new_tokens < 1 or max_new_tokens >= context_tokens or not 1 <= timeout_seconds <= 600):
            raise GymJobError("invalid vLLM context or timeout budget")
        self.reference, self.revision, self.image = reference, revision, image
        self.context_tokens, self.max_new_tokens, self.timeout_seconds = context_tokens, max_new_tokens, timeout_seconds
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._container: str | None = None
        self._name = "nf-vllm-" + uuid4().hex
        self._owner = uuid4().hex
        self._snapshot: Path | None = None

    def _docker(self, arguments: list[str], *, data: bytes | None = None, timeout: float = 30) -> subprocess.CompletedProcess[bytes]:
        try:
            return subprocess.run(["docker", *arguments], input=data, capture_output=True, timeout=timeout, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            raise GymJobError("vLLM Docker command failed or timed out") from exc

    def _cleanup(self) -> None:
        container = self._container
        if container is None:
            try:
                inspected = self._docker(["inspect", "--format", '{{ index .Config.Labels "nf.owner" }}', self._name], timeout=15)
                if inspected.returncode == 0 and inspected.stdout.decode().strip() == self._owner:
                    container = self._name
            except GymJobError:
                pass
        if container is not None:
            result = self._docker(["rm", "--force", container], timeout=30)
            if result.returncode != 0:
                raise GymJobError("could not remove owned vLLM container")
            self._container = None
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
            self._snapshot = None

    def __enter__(self) -> ModelAdapter:
        if self._temporary is not None or self._container is not None:
            raise GymJobError("vLLM server is already open")
        files = verify_vllm_candidate_files(self.reference, revision=self.revision)
        self._temporary = tempfile.TemporaryDirectory(prefix="nf-vllm-candidate-")
        self._snapshot = Path(self._temporary.name)
        try:
            for name in files:
                source = Path(self.reference.artifact_path) / name
                target = self._snapshot / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            if artifact_files(self._snapshot) != files:
                raise GymJobError("vLLM mounted snapshot differs from verified candidate files")
            if "," in str(self._snapshot):
                raise GymJobError("vLLM snapshot path cannot contain a comma")
            create = ["create", "--name", self._name, "--label", f"nf.owner={self._owner}", "--network", "none",
                      "--read-only", "--gpus", "all", "--log-driver", "none", "--shm-size", "1g",
                      "--tmpfs", "/tmp:rw,exec,nosuid,nodev,size=2g", "--mount",
                      f"type=bind,src={self._snapshot},dst=/model,readonly"]
            for name, value in {"VLLM_ALLOW_INSECURE_SERIALIZATION": "0", "VLLM_NO_USAGE_STATS": "1",
                                "VLLM_USE_FLASHINFER_SAMPLER": "0", "HOME": "/tmp",
                                "DO_NOT_TRACK": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                                "HF_HOME": "/tmp/hf", "XDG_CACHE_HOME": "/tmp/cache", "VLLM_CACHE_ROOT": "/tmp/vllm",
                                "TORCHINDUCTOR_CACHE_DIR": "/tmp/torchinductor", "TRITON_CACHE_DIR": "/tmp/triton",
                                "VLLM_LOGGING_LEVEL": "ERROR"}.items():
                create.extend(["--env", f"{name}={value}"])
            create.extend([self.image, "/model", "--host", "127.0.0.1", "--port", "8000", "--served-model-name",
                           "nf-candidate", "--dtype", "bfloat16", "--max-model-len", str(self.context_tokens),
                           "--gpu-memory-utilization", "0.25", "--enforce-eager", "--disable-log-stats",
                           "--disable-uvicorn-access-log", "--no-enable-log-requests", "--no-enable-log-outputs", "--no-enable-log-deltas"])
            created = self._docker(create, timeout=60)
            container = created.stdout.decode("ascii", errors="replace").strip()
            if created.returncode != 0 or len(container) != 64 or any(char not in "0123456789abcdef" for char in container):
                raise GymJobError("vLLM container creation failed")
            self._container = container
            if self._docker(["start", container], timeout=60).returncode != 0:
                raise GymJobError("vLLM container failed to start")
            deadline = time.monotonic() + self.timeout_seconds
            while time.monotonic() < deadline:
                if self._docker(["exec", container, "python3", "-c", _HTTP_CLIENT, "/health"], timeout=10).returncode == 0:
                    break
                state = self._docker(["inspect", "--format", "{{.State.Running}}", container], timeout=10)
                if state.returncode != 0 or state.stdout.strip() != b"true":
                    raise GymJobError("vLLM container exited before becoming ready")
                time.sleep(1)
            else:
                raise GymJobError("vLLM server readiness timed out")
            if artifact_files(self._snapshot) != files:
                raise GymJobError("vLLM mounted files changed during startup")
            return self._adapter(files)
        except BaseException:
            self._cleanup()
            raise

    def _adapter(self, files: dict[str, str]) -> ModelAdapter:
        reference = self.reference
        identity = AdapterIdentity("vllm_container", "1", _document_digest({
            "adapter": sha256_file(__file__), "candidate_verifier": sha256_file(Path(__file__).with_name("vllm_candidate_files.py")),
            "image": self.image,
        }))
        candidate = CandidateIdentity(reference.candidate_id, reference.method_id, reference.base_digest,
                                      reference.model_digest, reference.artifact_digest)
        settings = {"image": self.image, "device": "cuda:0", "precision": "bfloat16", "context_tokens": self.context_tokens,
                    "max_new_tokens": self.max_new_tokens, "network": "none", "seed": 0}

        def loaded_identity() -> Mapping[str, Any]:
            if self._container is None or self._snapshot is None:
                raise GymJobError("vLLM container is not running")
            return {"model_id": reference.model_id, "revision": self.revision, "checkpoint_digest": reference.base_digest,
                    "model_family": reference.family, "device": "cuda:0", "precision": "bfloat16",
                    "mounted_files_verified": True, "local_loaded_files_verified": False,
                    "served_model_identity_verified": False, "artifact_digest": reference.artifact_digest,
                    "config_digest": files["config.json"], "tokenizer_digest": files["tokenizer.json"]}

        def infer(interaction: Mapping[str, Any]) -> ExecutionResponse:
            if self._container is None or not isinstance(interaction, Mapping):
                raise GymJobError("vLLM container is not running or interaction is malformed")
            kind = interaction.get("format")
            if kind not in {"chat_messages_v1", "json_extraction_v1"} or not isinstance(interaction.get("messages"), list):
                raise GymJobError("vLLM requires chat or JSON extraction messages")
            payload: dict[str, Any] = {"model": "nf-candidate", "messages": interaction["messages"], "temperature": 0,
                                       "max_tokens": self.max_new_tokens, "chat_template_kwargs": {"enable_thinking": False}}
            if kind == "json_extraction_v1":
                payload["response_format"] = {"type": "json_schema", "json_schema": {
                    "name": "nf_output", "strict": True, "schema": interaction.get("output_schema")}}
            data = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
            result = self._docker(["exec", "-i", self._container, "python3", "-c", _HTTP_CLIENT, "/v1/chat/completions"],
                                  data=data, timeout=75)
            if result.returncode != 0:
                raise GymJobError("vLLM inference failed")
            try:
                choices = json.loads(result.stdout)["choices"]
                if not isinstance(choices, list) or len(choices) != 1 or choices[0].get("finish_reason") != "stop":
                    raise ValueError("completion did not finish normally")
                text = choices[0]["message"]["content"]
                if not isinstance(text, str):
                    raise ValueError("completion has no text")
                response = parse_strict_json(text.encode("utf-8")) if kind == "json_extraction_v1" else text
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                raise GymJobError("vLLM returned an incomplete or invalid completion") from exc
            return ExecutionResponse(response, {"raw_text": text, "served_model_identity_verified": False})

        return ModelAdapter(identity, candidate, reference.family, ("chat_messages_v1", "json_extraction_v1"), settings,
                            infer, loaded_identity, artifact_kinds=("full",), artifact_kind="full", reference=reference)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._cleanup()