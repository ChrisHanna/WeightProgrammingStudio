"""Run serving adapters on real runtimes and real checkpoints.

A missing runtime or checkpoint skips with a visible reason, and fails when NF_QUALIFICATION=1.
Set NF_PYTHIA_DIRS (os.pathsep-separated) to choose checkpoints, and NF_LLAMA_CLI for the binary.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from neural_foundry import export_formats, pythia, serving  # noqa: E402

PROMPTS = ("The capital of France is", "Once upon a time", "1 2 3 4")
MAX_TOKENS = 12


def _absent(reason: str) -> None:
    if os.environ.get("NF_QUALIFICATION") == "1":
        pytest.fail(f"NF_QUALIFICATION=1 requires this runtime: {reason}")
    pytest.skip(reason)


def _checkpoints() -> list[Path]:
    configured = os.environ.get("NF_PYTHIA_DIRS", os.pathsep.join(("/tmp/pythia-14m", "/tmp/pythia-31m")))
    return [Path(p) for p in configured.split(os.pathsep) if (Path(p) / "model.safetensors").is_file()]


@pytest.fixture(params=["pythia-14m", "pythia-31m"])
def checkpoint(request: pytest.FixtureRequest) -> Path:
    if importlib.util.find_spec("transformers") is None:
        _absent("transformers is not installed")
    for path in _checkpoints():
        if path.name == request.param:
            return path
    _absent(f"checkpoint {request.param} not found; set NF_PYTHIA_DIRS")
    raise AssertionError


def _greedy_ids(model: pythia.PythiaModel, prompt: str, count: int) -> tuple[int, ...]:
    ids = list(model.tokenizer.encode(prompt))
    start = len(ids)
    for _ in range(count):
        logits, _ = model.forward_with_mlp_key(ids)
        ids.append(int(logits[-1].argmax()))
    return tuple(ids[start:])


def _assert_agrees(model: pythia.PythiaModel, directory: Path) -> None:
    adapter = serving.TransformersAdapter(directory, dtype="float64")
    for prompt in PROMPTS:
        ours = model.tokenizer.encode(prompt)
        inputs = adapter.tokenizer(prompt, return_tensors="pt")
        assert inputs["input_ids"][0].tolist() == list(ours)
        with torch.no_grad():
            theirs_logits = adapter.model(**inputs).logits[0]
        our_logits, _ = model.forward_with_mlp_key(ours)
        assert torch.max(torch.abs(theirs_logits - our_logits)) < 1e-2
        result = adapter.generate(prompt, {"max_tokens": MAX_TOKENS})
        assert result.token_ids == _greedy_ids(model, prompt, MAX_TOKENS)


def test_transformers_agrees_with_forward_pass_on_real_checkpoint(checkpoint: Path) -> None:
    _assert_agrees(pythia.PythiaModel.from_pretrained_dir(checkpoint), checkpoint)


def test_transformers_runs_edited_export_and_edit_changes_output(checkpoint: Path, tmp_path: Path) -> None:
    model = pythia.PythiaModel.from_pretrained_dir(checkpoint)
    before = {p: _greedy_ids(model, p, MAX_TOKENS) for p in PROMPTS}
    site = model.mlp_down_proj_site(1)
    model.tensors[site] = model.tensors[site] * 1.5
    export_formats.export_huggingface_safetensors(model, tmp_path, dtype="f64", source_checkpoint_dir=checkpoint)
    _assert_agrees(model, tmp_path)
    assert {p: _greedy_ids(model, p, MAX_TOKENS) for p in PROMPTS} != before


def test_export_config_describes_the_written_dtype(checkpoint: Path, tmp_path: Path) -> None:
    model = pythia.PythiaModel.from_pretrained_dir(checkpoint)
    export_formats.export_huggingface_safetensors(model, tmp_path, dtype="f32", source_checkpoint_dir=checkpoint)
    adapter = serving.TransformersAdapter(tmp_path)
    assert next(adapter.model.parameters()).dtype == torch.float32
    assert "float32" in (tmp_path / "config.json").read_text(encoding="utf-8")


def test_llama_cpp_loads_exported_gguf(tmp_path: Path) -> None:
    binary = os.environ.get("NF_LLAMA_CLI") or shutil.which("llama-cli") or shutil.which("llama-completion")
    if not binary:
        _absent("llama.cpp is not installed; set NF_LLAMA_CLI")
    paths = _checkpoints()
    if not paths:
        _absent("no Pythia checkpoint found; set NF_PYTHIA_DIRS")
    model = pythia.PythiaModel.from_pretrained_dir(paths[0])
    gguf = export_formats.export_gguf_v3(model, tmp_path / "model.gguf")
    result = serving.LlamaCppAdapter(gguf, binary=binary).generate(PROMPTS[0], {"max_tokens": MAX_TOKENS})
    assert result.text


def test_vllm_generates_from_real_checkpoint() -> None:
    if importlib.util.find_spec("vllm") is None:
        _absent("vllm is not installed")
    paths = _checkpoints()
    if not paths:
        _absent("no Pythia checkpoint found; set NF_PYTHIA_DIRS")
    result = serving.VllmAdapter(paths[0]).generate(PROMPTS[0], {"max_tokens": MAX_TOKENS})
    assert result.text
