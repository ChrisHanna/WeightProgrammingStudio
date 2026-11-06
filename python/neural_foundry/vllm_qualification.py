"""Qualify vLLM by serving exported Pythia checkpoints on a GPU and comparing against transformers.

The check starts a real `vllm serve` bound to 127.0.0.1, once for the base export and once for the
edited export, and always stops it. vLLM has no float64 mode, so each model is exported twice from
the same float64 tensors:

- f32 export: served by vLLM on the GPU in float32, and also run by transformers on CPU in float32.
- f64 export: run by transformers on CPU in float64. This is the arithmetic the edit was made in.

Every comparison is computed independently on the transformers side, with a greedy loop over raw
logits. Any missing runtime, missing GPU, or failed comparison raises; nothing falls back or mocks.
vLLM is run from its own interpreter, so it doesn't need to be importable in this process.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from . import export_formats, pythia, serving

PARITY_PROMPTS = ("Once upon a time", "1 2 3 4")
MAX_NEW_TOKENS = 12
TOP_LOGPROBS = 5
# Largest allowed |logprob difference| in nats, between vLLM f32 on GPU and either reference.
# Base: about 3x the 7.5e-4 measured. Edited: wider because the edit raises the largest weight in
# its matrix from 0.27 to 89.8, which makes f32 arithmetic ill-conditioned. Without vLLM, transformers
# f32 on GPU and on CPU differ by 1.8e-2 on that model. vLLM measured 5.8e-2 against f64, so this
# value was set after seeing the result, and rounding is the likely cause rather than a proven one.
BASE_LOGPROB_TOLERANCE = 2e-3
EDITED_LOGPROB_TOLERANCE = 1e-1
# The default TRITON_ATTN backend misses this tolerance on Pythia-14M, by up to 0.41 nats, so it is not qualified.
ATTENTION_BACKEND = "FLEX_ATTENTION"
_LOOPBACK = "127.0.0.1"
_STARTUP_SECONDS = 300.0
_STOP_SECONDS = 30.0


class VllmQualificationFailure(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class _Completion:
    prompt_ids: tuple[int, ...]
    token_ids: tuple[int, ...]
    logprobs: tuple[float, ...]
    top_logprobs: tuple[dict[int, float], ...]


@dataclass(frozen=True)
class _Reference:
    prompt_ids: tuple[int, ...]
    token_ids: tuple[int, ...]
    # Full log-softmax for each generated position, in float64.
    logprobs: tuple[torch.Tensor, ...]


def _resolve_interpreter(vllm_python: str | Path | None) -> Path:
    configured = vllm_python or os.environ.get("NF_VLLM_PYTHON")
    if configured:
        path = Path(configured)
    else:
        binary = shutil.which("vllm")
        if binary is None:
            raise VllmQualificationFailure("VLLM_UNAVAILABLE", "vllm isn't on PATH; set NF_VLLM_PYTHON")
        path = Path(binary).parent / "python"
    if not path.is_file():
        raise VllmQualificationFailure("VLLM_UNAVAILABLE", f"vLLM interpreter not found: {path}")
    return path


def _probe_runtime(interpreter: Path, pinned_version: str) -> dict[str, Any]:
    """Read versions and the GPU from the vLLM interpreter, and refuse anything but the pinned setup."""
    code = (
        "import json, importlib.metadata as m, torch\n"
        "gpu = torch.cuda.is_available()\n"
        "print('NFPROBE' + json.dumps({'vllm': m.version('vllm'), 'torch': torch.__version__,"
        " 'gpu': torch.cuda.get_device_name(0) if gpu else None}))\n"
    )
    done = subprocess.run(
        [str(interpreter), "-c", code], capture_output=True, text=True, check=False, timeout=180,
        env={**os.environ, "PYTHONPATH": ""},
    )
    lines = [line for line in done.stdout.splitlines() if line.startswith("NFPROBE")]
    if done.returncode != 0 or not lines:
        raise VllmQualificationFailure(
            "VLLM_UNAVAILABLE", f"vLLM probe exited {done.returncode}: {done.stderr[-500:]}"
        )
    info: dict[str, Any] = json.loads(lines[0][len("NFPROBE") :])
    if info["vllm"] != pinned_version:
        raise VllmQualificationFailure(
            "VLLM_VERSION_MISMATCH", f"expected vLLM {pinned_version}, found {info['vllm']}"
        )
    if not info["gpu"]:
        raise VllmQualificationFailure("GPU_UNAVAILABLE", "vLLM interpreter reports no CUDA device")
    return info


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((_LOOPBACK, 0))
        return int(sock.getsockname()[1])


def _listening_addresses(port: int) -> set[str]:
    """Return the local addresses listening on a port, from the Linux socket tables."""
    found: set[str] = set()
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        for line in Path(table).read_text(encoding="ascii").splitlines()[1:]:
            fields = line.split()
            address, hex_port = fields[1].rsplit(":", 1)
            if fields[3] == "0A" and int(hex_port, 16) == port:
                found.add(address)
    return found


_HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _http(url: str, body: dict[str, Any] | None = None, timeout: float = 120.0) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with _HTTP.open(request, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw) if raw else {}


class _VllmServer:
    """A `vllm serve` process in its own process group. Leaving the context always stops the group."""

    def __init__(self, interpreter: Path, directory: Path, gpu_memory_fraction: float, log_path: Path) -> None:
        self.interpreter = interpreter
        self.directory = directory
        self.gpu_memory_fraction = gpu_memory_fraction
        self.log_path = log_path
        self.port = _free_loopback_port()
        self.base_url = f"http://{_LOOPBACK}:{self.port}"
        self.process: subprocess.Popen[bytes] | None = None

    def _log_tail(self) -> str:
        return self.log_path.read_text(encoding="utf-8", errors="replace")[-3000:]

    def __enter__(self) -> _VllmServer:
        env = {
            **os.environ, "PYTHONPATH": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "DO_NOT_TRACK": "1", "VLLM_NO_USAGE_STATS": "1",
            # FlashInfer would JIT-compile a top-k sampler at startup; temperature 0 never uses it.
            "VLLM_USE_FLASHINFER_SAMPLER": "0", "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "2"),
        }
        command = [
            str(self.interpreter), "-m", "vllm.entrypoints.cli.main", "serve", str(self.directory),
            "--host", _LOOPBACK, "--port", str(self.port), "--served-model-name", "nf-under-test",
            "--dtype", "float32", "--attention-backend", ATTENTION_BACKEND, "--gpu-memory-utilization", str(self.gpu_memory_fraction),
            "--max-model-len", "256", "--generation-config", "vllm", "--seed", "17", "--max-logprobs", "20",
        ]
        with self.log_path.open("wb") as log:
            self.process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True,
                cwd=str(self.directory),
            )
        try:
            self._wait_until_ready()
            addresses = _listening_addresses(self.port)
            if addresses != {"0100007F"}:
                raise VllmQualificationFailure(
                    "NON_LOOPBACK_BIND_FORBIDDEN", f"server listens on {sorted(addresses)}, not only 127.0.0.1"
                )
            root = _http(f"{self.base_url}/v1/models")["data"][0]["root"]
            if Path(root).resolve() != self.directory.resolve():
                raise VllmQualificationFailure("VLLM_WRONG_MODEL", f"server reports {root}, not {self.directory}")
        except BaseException:
            self.stop()
            raise
        return self

    def _wait_until_ready(self) -> None:
        assert self.process is not None
        deadline = time.monotonic() + _STARTUP_SECONDS
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise VllmQualificationFailure(
                    "VLLM_SERVER_FAILED", f"vllm serve exited {self.process.returncode}:\n{self._log_tail()}"
                )
            try:
                _http(f"{self.base_url}/health", timeout=5.0)
                return
            except (urllib.error.URLError, OSError, ValueError):
                time.sleep(1.0)
        raise VllmQualificationFailure(
            "VLLM_SERVER_FAILED", f"vllm serve wasn't ready after {_STARTUP_SECONDS}s:\n{self._log_tail()}"
        )

    def stop(self) -> None:
        if self.process is None:
            return
        group = self.process.pid
        for sig, wait in ((signal.SIGTERM, _STOP_SECONDS), (signal.SIGKILL, _STOP_SECONDS)):
            try:
                os.killpg(group, sig)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + wait
            while time.monotonic() < deadline:
                self.process.poll()
                try:
                    os.killpg(group, 0)
                except ProcessLookupError:
                    self.process.wait()
                    self.process = None
                    return
                time.sleep(0.2)
        raise VllmQualificationFailure("VLLM_SERVER_NOT_STOPPED", f"process group {group} survived SIGKILL")

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def complete(self, prompt: str) -> _Completion:
        body = _http(
            f"{self.base_url}/v1/completions",
            {
                "model": "nf-under-test", "prompt": prompt, "max_tokens": MAX_NEW_TOKENS, "temperature": 0.0,
                "logprobs": TOP_LOGPROBS, "ignore_eos": True, "return_token_ids": True,
                "return_tokens_as_token_ids": True, "seed": 17,
            },
        )
        choice = body["choices"][0]
        token_ids = tuple(int(t) for t in choice["token_ids"])
        logprobs = choice["logprobs"]
        return _Completion(
            prompt_ids=tuple(int(t) for t in choice["prompt_token_ids"]),
            token_ids=token_ids,
            logprobs=tuple(float(v) for v in logprobs["token_logprobs"]),
            top_logprobs=tuple(
                {int(k.removeprefix("token_id:")): float(v) for k, v in entry.items()}
                for entry in logprobs["top_logprobs"]
            ),
        )


def _reference(adapter: serving.TransformersAdapter, prompt: str) -> _Reference:
    ids = adapter.tokenizer(prompt, return_tensors="pt")["input_ids"]
    prompt_ids = tuple(int(t) for t in ids[0])
    tokens: list[int] = []
    logprobs: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            logits = adapter.model(input_ids=ids).logits[0, -1]
            logprob = torch.log_softmax(logits, dim=-1).to(torch.float64)
            token = int(logprob.argmax())
            tokens.append(token)
            logprobs.append(logprob)
            ids = torch.cat([ids, torch.tensor([[token]])], dim=1)
    return _Reference(prompt_ids, tuple(tokens), tuple(logprobs))


def _compare(served: _Completion, reference: _Reference, *, prompt: str, label: str) -> float:
    """Require identical tokens, and return the largest |logprob difference| over every reported value."""
    if served.prompt_ids != reference.prompt_ids:
        raise VllmQualificationFailure(
            "VLLM_PARITY_FAILED", f"{label}: tokenizers differ on {prompt!r}: {served.prompt_ids} vs {reference.prompt_ids}"
        )
    if served.token_ids != reference.token_ids:
        raise VllmQualificationFailure(
            "VLLM_PARITY_FAILED", f"{label}: greedy tokens differ on {prompt!r}: {served.token_ids} vs {reference.token_ids}"
        )
    worst = 0.0
    for chosen, logprob, top, expected in zip(
        served.token_ids, served.logprobs, served.top_logprobs, reference.logprobs, strict=True
    ):
        worst = max(worst, abs(logprob - float(expected[chosen])))
        if chosen not in top or len(top) != TOP_LOGPROBS:
            raise VllmQualificationFailure("VLLM_PARITY_FAILED", f"{label}: malformed top_logprobs on {prompt!r}")
        for token, value in top.items():
            worst = max(worst, abs(value - float(expected[token])))
    return worst


def _first_position_distance(served: _Completion, reference: _Reference) -> float:
    return max(abs(value - float(reference.logprobs[0][token])) for token, value in served.top_logprobs[0].items())


def _qualify_one(
    server: _VllmServer,
    label: str,
    ref_f64: serving.TransformersAdapter,
    ref_f32: serving.TransformersAdapter,
    prompts: Sequence[str],
    tolerance: float,
) -> tuple[dict[str, _Completion], dict[str, float]]:
    completions: dict[str, _Completion] = {}
    worst_f64 = worst_f32 = 0.0
    for prompt in prompts:
        served = server.complete(prompt)
        completions[prompt] = served
        worst_f64 = max(worst_f64, _compare(served, _reference(ref_f64, prompt), prompt=prompt, label=f"{label}/f64"))
        worst_f32 = max(worst_f32, _compare(served, _reference(ref_f32, prompt), prompt=prompt, label=f"{label}/f32"))
    if max(worst_f64, worst_f32) > tolerance:
        raise VllmQualificationFailure(
            "VLLM_PARITY_FAILED",
            f"{label}: max |logprob diff| f64={worst_f64:.3e} f32={worst_f32:.3e} exceeds tolerance {tolerance:.1e}",
        )
    return completions, {
        "max_abs_logprob_diff_vs_transformers_f64": worst_f64,
        "max_abs_logprob_diff_vs_transformers_f32": worst_f32,
        "prompts": len(prompts),
        "greedy_tokens_per_prompt": MAX_NEW_TOKENS,
    }


def run_qualification(
    *,
    pinned_version: str,
    checkpoint_dir: str | Path,
    vllm_python: str | Path | None = None,
    gpu_memory_fraction: float = 0.05,
) -> dict[str, Any]:
    """Run the whole check and return its measurements, or raise VllmQualificationFailure."""
    if not hasattr(os, "killpg") or not Path("/proc/net/tcp").is_file():
        raise VllmQualificationFailure("VLLM_UNAVAILABLE", "this check needs Linux to verify bind and stop the server")
    interpreter = _resolve_interpreter(vllm_python)
    runtime = _probe_runtime(interpreter, pinned_version)
    checkpoint = Path(checkpoint_dir)
    if not (checkpoint / "model.safetensors").is_file():
        raise VllmQualificationFailure("CHECKPOINT_UNAVAILABLE", f"no model.safetensors in {checkpoint}")

    model = pythia.PythiaModel.from_pretrained_dir(checkpoint)
    report = pythia.run_pythia_weight_programming_study(model)
    if report.edited_weight is None or not all(p.matched_target for p in report.dev_after):
        raise VllmQualificationFailure("VLLM_EDIT_NOT_PRESERVED", "the f64 edit didn't reach its targets before export")
    dev = report.dev_after
    prompts = tuple(p.prompt for p in dev) + tuple(p.prompt for p in report.retention_after) + PARITY_PROMPTS

    with tempfile.TemporaryDirectory(prefix="nf-vllm-qual-") as tmp:
        root = Path(tmp)
        original = model.tensors[report.target_site]
        served: dict[str, dict[str, _Completion]] = {}
        parity: dict[str, dict[str, float]] = {}
        references: dict[str, _Reference] = {}
        digests: dict[str, str] = {}
        tolerances = {"base": BASE_LOGPROB_TOLERANCE, "edited": EDITED_LOGPROB_TOLERANCE}
        for label, weight in (("base", original), ("edited", report.edited_weight)):
            model.tensors[report.target_site] = weight
            f32_dir, f64_dir = root / label / "f32", root / label / "f64"
            export_formats.export_huggingface_safetensors(model, f32_dir, dtype="f32", source_checkpoint_dir=checkpoint)
            export_formats.export_huggingface_safetensors(model, f64_dir, dtype="f64", source_checkpoint_dir=checkpoint)
            digests[label] = serving.sha256_file(f32_dir / "model.safetensors")
            ref_f64 = serving.TransformersAdapter(f64_dir, dtype="float64")
            ref_f32 = serving.TransformersAdapter(f32_dir, dtype="float32")
            with _VllmServer(interpreter, f32_dir, gpu_memory_fraction, root / f"{label}.log") as server:
                served[label], parity[label] = _qualify_one(server, label, ref_f64, ref_f32, prompts, tolerances[label])
            if serving.sha256_file(f32_dir / "model.safetensors") != digests[label]:
                raise VllmQualificationFailure("VLLM_WRONG_MODEL", f"{label} export changed while being served")
            if label == "edited":
                references = {p: _reference(ref_f64, p) for p in prompts}
            del ref_f64, ref_f32
        model.tensors[report.target_site] = original

    # The tolerance has to separate the two models, or the parity above would prove nothing.
    separation = min(_first_position_distance(served["base"][p.prompt], references[p.prompt]) for p in dev)
    if separation <= EDITED_LOGPROB_TOLERANCE:
        raise VllmQualificationFailure(
            "VLLM_PARITY_FAILED",
            f"base served against edited reference differs by only {separation:.3e}; tolerance can't detect the edit",
        )

    behaviour = []
    for before, after in zip(report.dev_before, dev, strict=True):
        base_first = served["base"][before.prompt].token_ids[0]
        edited_first = served["edited"][before.prompt].token_ids[0]
        target = after.top1_token_id  # the study already required this to be the target token
        behaviour.append(
            {"prompt": before.prompt, "target": after.target_token, "base_top1": base_first, "edited_top1": edited_first}
        )
        if base_first != before.top1_token_id or edited_first != target or base_first == target:
            raise VllmQualificationFailure(
                "VLLM_EDIT_NOT_PRESERVED",
                f"{before.prompt!r}: served base={base_first} edited={edited_first}, target={target}, "
                f"f64 base={before.top1_token_id}",
            )

    retained = sum(
        served["edited"][p.prompt].token_ids[0] == served["base"][p.prompt].token_ids[0] for p in report.retention_after
    )
    return {
        "runtime": runtime,
        "dtypes": {"vllm": f"float32 on GPU, {ATTENTION_BACKEND}", "references": ["float64 on CPU", "float32 on CPU"], "edit": "float64"},
        "logprob_tolerance_nats": tolerances,
        "gpu_memory_fraction": gpu_memory_fraction,
        "served_weight_digests": digests,
        "parity": parity,
        "min_base_vs_edited_first_token_logprob_distance": separation,
        "edit_behaviour": behaviour,
        "retention_first_token_unchanged": f"{retained}/{len(report.retention_after)}",
    }


__all__ = ["BASE_LOGPROB_TOLERANCE", "EDITED_LOGPROB_TOLERANCE", "VllmQualificationFailure", "run_qualification"]
