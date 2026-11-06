"""Generate text through real serving runtimes, with one result shape and checked model identity.

Each adapter refuses to run when its runtime is missing; nothing falls back to another backend.
"""

from __future__ import annotations

import hashlib
import importlib
import shutil
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .provenance import sha256_file


class ServingRuntimeUnavailableError(RuntimeError):
    """The requested runtime isn't installed, or failed to run."""


DEFAULT_SAMPLER: dict[str, Any] = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 64, "seed": 17}


def normalize_sampler(sampler: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = {**DEFAULT_SAMPLER, **dict(sampler or {})}
    unknown = set(merged) - set(DEFAULT_SAMPLER)
    if unknown:
        raise ValueError(f"Unsupported sampler settings: {sorted(unknown)}")
    if float(merged["temperature"]) < 0 or not 0 < float(merged["top_p"]) <= 1 or int(merged["max_tokens"]) <= 0:
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


class TransformersAdapter:
    runtime = "transformers"

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        expected_digest: str | None = None,
        dtype: str = "float32",
    ) -> None:
        """Load a checkpoint in an explicit arithmetic dtype.

        The dtype is explicit because transformers otherwise follows `config.json`: Pythia declares
        float16, and running in f16 changed greedy output compared with `PythiaModel`.
        """
        self.directory = Path(checkpoint_dir)
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

    def generate(self, prompt: str, sampler: Mapping[str, Any] | None = None) -> GenerationResult:
        s = normalize_sampler(sampler)
        self._torch.manual_seed(int(s["seed"]))
        start = time.monotonic()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        do_sample = float(s["temperature"]) > 0
        options = {"temperature": float(s["temperature"]), "top_p": float(s["top_p"])} if do_sample else {}
        output = self.model.generate(**inputs, max_new_tokens=int(s["max_tokens"]), do_sample=do_sample, **options)
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return GenerationResult(
            text, self.runtime, self.model_digest, s, time.monotonic() - start,
            token_ids=tuple(int(t) for t in new_tokens),
        )


# vLLM's default TRITON_ATTN missed parity on Pythia-14M by up to 0.41 nats; see vllm_qualification.py.
QUALIFIED_ATTENTION_BACKEND = "FLEX_ATTENTION"


class VllmAdapter:
    """Run a checkpoint through vLLM in-process, with the settings `vllm_qualification` serves under.

    Not qualified: the check in `vllm_qualification` currently fails its held-out logprob limit, and it
    only ever covers `vllm serve` on Pythia-14M.
    """

    runtime = "vllm"

    def __init__(
        self, checkpoint_dir: str | Path, *, expected_digest: str | None = None, gpu_memory_utilization: float = 0.05
    ) -> None:
        self.directory = Path(checkpoint_dir)
        self.model_digest = _weights_digest(self.directory, expected_digest)
        try:
            self._vllm = importlib.import_module("vllm")
        except ImportError as exc:
            raise ServingRuntimeUnavailableError("vllm is not installed") from exc
        self.engine = self._vllm.LLM(
            model=str(self.directory),
            dtype="float32",
            attention_backend=QUALIFIED_ATTENTION_BACKEND,
            generation_config="vllm",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=256,
        )

    def generate(self, prompt: str, sampler: Mapping[str, Any] | None = None) -> GenerationResult:
        s = normalize_sampler(sampler)
        start = time.monotonic()
        params = self._vllm.SamplingParams(
            temperature=float(s["temperature"]), top_p=float(s["top_p"]), max_tokens=int(s["max_tokens"]), seed=int(s["seed"])
        )
        text = self.engine.generate([prompt], params)[0].outputs[0].text.strip()
        return GenerationResult(text, self.runtime, self.model_digest, s, time.monotonic() - start)
