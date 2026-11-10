"""Generate text through real serving runtimes, with one result shape and checked model identity.

Each adapter refuses to run when its runtime is missing; nothing falls back to another backend.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .gym_job import is_finite_number
from .provenance import sha256_file


class ServingRuntimeUnavailableError(RuntimeError):
    """The requested runtime isn't installed, or failed to run."""


DEFAULT_SAMPLER: dict[str, Any] = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 64, "seed": 17}


def normalize_sampler(sampler: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = {**DEFAULT_SAMPLER, **dict(sampler or {})}
    unknown = set(merged) - set(DEFAULT_SAMPLER)
    if unknown:
        raise ValueError(f"Unsupported sampler settings: {sorted(unknown)}")
    if (not is_finite_number(merged["temperature"]) or merged["temperature"] < 0
        or not is_finite_number(merged["top_p"]) or not 0 < merged["top_p"] <= 1
        or type(merged["max_tokens"]) is not int or merged["max_tokens"] <= 0
        or type(merged["seed"]) is not int or merged["seed"] < 0):
        raise ValueError("Sampler settings are out of range")
    return merged


@dataclass(frozen=True)
class GenerationResult:
    text: str
    runtime: str
    model_digest: str
    sampler: dict[str, Any]
    duration_seconds: float
    token_ids: tuple[int, ...] | None = None


class ServingAdapter(Protocol):
    runtime: str
    model_digest: str

    def generate(self, prompt: str, sampler: Mapping[str, Any] | None = None) -> GenerationResult: ...


def _check_digest(path: Path, expected: str | None) -> str:
    actual = sha256_file(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Model identity mismatch for {path}: expected {expected}, found {actual}")
    return actual


class LlamaCppAdapter:
    """Run a GGUF file through a llama.cpp binary.

    Not yet qualified on a real binary, and `export_gguf_v3` output isn't expected to load.
    """

    runtime = "llama.cpp"

    def __init__(
        self,
        gguf_path: str | Path,
        *,
        binary: str | None = None,
        expected_digest: str | None = None,
        timeout_seconds: float = 300.0,
    ) -> None:
        self.gguf_path = Path(gguf_path)
        self.binary = binary or shutil.which("llama-cli") or shutil.which("llama-completion")
        if not self.binary:
            raise ServingRuntimeUnavailableError("llama-cli is not installed")
        self.model_digest = _check_digest(self.gguf_path, expected_digest)
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt: str, sampler: Mapping[str, Any] | None = None) -> GenerationResult:
        s = normalize_sampler(sampler)
        start = time.monotonic()
        try:
            done = subprocess.run(
                [
                    str(self.binary), "-m", str(self.gguf_path), "--temp", str(s["temperature"]),
                    "--top-p", str(s["top_p"]), "-s", str(s["seed"]), "-n", str(s["max_tokens"]),
                    "--no-display-prompt", "-p", prompt,
                ],
                capture_output=True, text=True, check=False, timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise ServingRuntimeUnavailableError(f"llama.cpp timed out after {self.timeout_seconds}s") from exc
        if done.returncode != 0:
            raise ServingRuntimeUnavailableError(f"llama.cpp exited {done.returncode}: {done.stderr[-300:]}")
        return GenerationResult(done.stdout.strip(), self.runtime, self.model_digest, s, time.monotonic() - start)


def _weights_digest(directory: Path, expected: str | None) -> str:
    shards = sorted(directory.glob("*.safetensors"))
    if not shards:
        raise ServingRuntimeUnavailableError(f"{directory} has no safetensors weights")
    if len(shards) == 1:
        return _check_digest(shards[0], expected)
    actual = "sha256:" + hashlib.sha256("\n".join(f"{p.name}:{sha256_file(p)}" for p in shards).encode()).hexdigest()
    if expected is not None and actual != expected:
        raise ValueError(f"Model identity mismatch for {directory}: expected {expected}, found {actual}")
    return actual


_TRANSFORMERS_GENERATION_LOCK = threading.RLock()


class TransformersAdapter:
    runtime = "transformers"

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        expected_digest: str | None = None,
        dtype: str = "float32",
        device: str = "cpu",
        max_context_tokens: int | None = None,
    ) -> None:
        """Load a checkpoint in an explicit arithmetic dtype.

        The dtype is explicit because transformers otherwise follows `config.json`: Pythia declares
        float16, and running in f16 changed greedy output compared with `PythiaModel`.
        """
        self.directory = Path(checkpoint_dir)
        if max_context_tokens is not None and (type(max_context_tokens) is not int or max_context_tokens < 1):
            raise ValueError("max_context_tokens must be a positive integer")
        self.model_digest = _weights_digest(self.directory, expected_digest)
        try:
            self._transformers = importlib.import_module("transformers")
            self._torch = importlib.import_module("torch")
        except ImportError as exc:
            raise ServingRuntimeUnavailableError("transformers is not installed") from exc
        self.tokenizer = self._transformers.AutoTokenizer.from_pretrained(self.directory, local_files_only=True)
        self.model = self._transformers.AutoModelForCausalLM.from_pretrained(
            self.directory, local_files_only=True, dtype=getattr(self._torch, dtype)
        )
        self.model = self.model.to(device).eval()
        self.device = self._torch.device(device)
        self.context_limit = max_context_tokens or int(getattr(self.model.config, "max_position_embeddings", 0))
        if type(self.context_limit) is not int or self.context_limit < 1:
            raise ValueError("An explicit positive context limit is required")
        self.dtype = dtype

    def generate(self, prompt: str, sampler: Mapping[str, Any] | None = None) -> GenerationResult:
        s = normalize_sampler(sampler)
        start = time.monotonic()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if int(inputs["input_ids"].shape[1]) + int(s["max_tokens"]) > self.context_limit:
            raise ValueError("Tokenized prompt plus output budget exceeds the configured context limit")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        do_sample = float(s["temperature"]) > 0
        options = {"temperature": float(s["temperature"]), "top_p": float(s["top_p"])} if do_sample else {}
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with _TRANSFORMERS_GENERATION_LOCK, self._torch.random.fork_rng(devices=devices), self._torch.inference_mode():
            if do_sample:
                self._torch.random.default_generator.manual_seed(int(s["seed"]))
                if devices:
                    self._torch.cuda.default_generators[devices[0]].manual_seed(int(s["seed"]))
            output = self.model.generate(**inputs, max_new_tokens=int(s["max_tokens"]), do_sample=do_sample, **options)
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return GenerationResult(
            text, self.runtime, self.model_digest, s, time.monotonic() - start,
            token_ids=tuple(int(t) for t in new_tokens),
        )


# vLLM's default TRITON_ATTN missed parity on Pythia-14M by up to 0.41 nats; see vllm_qualification.py.
QUALIFIED_ATTENTION_BACKEND = "FLEX_ATTENTION"

# FlashInfer can JIT-compile a sampler at startup, which fails with FileNotFoundError('ninja') where
# ninja is missing. `vllm_qualification` serves with this off too; temperature 0 never uses that sampler.
_VLLM_STARTUP_ENV = {"VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_ALLOW_INSECURE_SERIALIZATION": "0"}
_vllm_startup_lock = threading.Lock()


def _require_secure_vllm_serialization() -> None:
    """Insecure serialization is restricted to isolated identity probes, never serving."""
    loaded_envs = sys.modules.get("vllm.envs")
    if (os.environ.get("VLLM_ALLOW_INSECURE_SERIALIZATION", "0") != "0"
        or (loaded_envs is not None and bool(getattr(loaded_envs, "VLLM_ALLOW_INSECURE_SERIALIZATION", False)))):
        raise ServingRuntimeUnavailableError("vLLM serving forbids insecure serialization")


@contextmanager
def _vllm_startup_env() -> Iterator[None]:
    """Set vLLM's startup variables only while it starts; the caller's values come back afterwards.

    The environment is process-wide. The lock serializes adapters, but other threads can see the value
    during startup.
    """
    with _vllm_startup_lock:
        _require_secure_vllm_serialization()
        saved = {name: os.environ.get(name) for name in _VLLM_STARTUP_ENV}
        os.environ.update(_VLLM_STARTUP_ENV)
        try:
            yield
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


class VllmAdapter:
    """Run a checkpoint through vLLM in-process, with the settings `vllm_qualification` serves under.

    Not qualified: the check in `vllm_qualification` currently fails its held-out logprob limit, and it
    only ever covers `vllm serve` on Pythia-14M.
    """

    runtime = "vllm"

    def __init__(
        self, checkpoint_dir: str | Path, *, expected_digest: str | None = None, gpu_memory_utilization: float = 0.05,
        dtype: str = "float32", device: str = "cuda:0", max_context_tokens: int | None = None,
    ) -> None:
        self.directory = Path(checkpoint_dir)
        self.model_digest = _weights_digest(self.directory, expected_digest)
        if max_context_tokens is not None and (type(max_context_tokens) is not int or max_context_tokens < 1):
            raise ValueError("max_context_tokens must be a positive integer")
        self.context_limit = max_context_tokens
        # vLLM may read the variable at import, at engine start or in a worker it spawns, so all are covered.
        with _vllm_startup_env():
            try:
                self._vllm = importlib.import_module("vllm")
            except ImportError as exc:
                raise ServingRuntimeUnavailableError("vllm is not installed") from exc
            _require_secure_vllm_serialization()
            try:
                self.engine = self._vllm.LLM(
                    model=str(self.directory),
                    dtype=dtype,
                    device=device,
                    attention_backend=QUALIFIED_ATTENTION_BACKEND,
                    generation_config="vllm",
                    gpu_memory_utilization=gpu_memory_utilization,
                    **({"max_model_len": max_context_tokens} if max_context_tokens is not None else {}),
                )
            except FileNotFoundError as exc:
                if exc.filename != "ninja" and "ninja" not in str(exc):
                    raise
                raise ServingRuntimeUnavailableError(f"vLLM needs ninja to start, even with the FlashInfer sampler off: {exc}") from exc

    def generate(self, prompt: str, sampler: Mapping[str, Any] | None = None) -> GenerationResult:
        s = normalize_sampler(sampler)
        tokenizer = self.engine.get_tokenizer()
        context = self.context_limit or int(self.engine.llm_engine.model_config.max_model_len)
        if len(tokenizer.encode(prompt)) + int(s["max_tokens"]) > context:
            raise ValueError("Tokenized prompt plus output budget exceeds the configured context limit")
        start = time.monotonic()
        params = self._vllm.SamplingParams(
            temperature=float(s["temperature"]), top_p=float(s["top_p"]), max_tokens=int(s["max_tokens"]), seed=int(s["seed"])
        )
        text = self.engine.generate([prompt], params)[0].outputs[0].text.strip()
        return GenerationResult(text, self.runtime, self.model_digest, s, time.monotonic() - start)
