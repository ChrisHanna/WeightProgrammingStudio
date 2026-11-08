"""Local-only loading through the standard Transformers runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ...model_digest import model_state_digest
from ...provenance import resolve_checkpoint


class UnsupportedModel(RuntimeError):
    """The model cannot satisfy this runtime's required contract."""


class WeightEditInputError(ValueError):
    """The caller supplied invalid edit inputs or permissions."""


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    tokenizer: Any
    config: Any
    text_config: Any
    base_digest: str
    hidden_size: int
    checkpoint_path: Path
    file_digests: dict[str, str]
    shared_storage_groups: tuple[tuple[str, ...], ...]
    checkpoint_tensor_names: frozenset[str]


def storage_groups(model: Any) -> tuple[tuple[str, ...], ...]:
    groups: dict[tuple[str, int], list[str]] = {}
    for name, tensor in model.state_dict().items():
        if tensor.numel():
            groups.setdefault((str(tensor.device), tensor.untyped_storage().data_ptr()), []).append(name)
    return tuple(sorted(tuple(sorted(names)) for names in groups.values() if len(names) > 1))


def require_editable(loaded: LoadedModel, tensor_names: list[str], *, allow_shared: bool = False) -> None:
    if type(allow_shared) is not bool or not tensor_names or len(set(tensor_names)) != len(tensor_names):
        raise WeightEditInputError("edit tensor names must be nonempty and unique")
    state = loaded.model.state_dict()
    if any(name not in state for name in tensor_names):
        raise WeightEditInputError("edit names must identify loaded model tensors")
    if not allow_shared and any(set(group) & set(tensor_names) for group in storage_groups(loaded.model)):
        raise WeightEditInputError("editing shared tensor storage requires explicit permission")


def _checkpoint_names(directory: Path, files: dict[str, str]) -> frozenset[str]:
    from safetensors import safe_open

    names = set()
    for filename in files:
        if filename.endswith(".safetensors"):
            with safe_open(directory / filename, framework="pt", device="cpu") as tensors:
                keys = set(tensors.keys())
            if keys & names:
                raise UnsupportedModel("checkpoint repeats tensor names across safetensors files")
            names.update(keys)
    if not names:
        raise UnsupportedModel("checkpoint has no safetensors tensor index")
    return frozenset(names)


def text_configuration(config: Any) -> tuple[Any, int]:
    get_text_config = getattr(config, "get_text_config", None)
    if not callable(get_text_config):
        raise UnsupportedModel("config does not expose get_text_config")
    text_config = get_text_config()
    for name in ("hidden_size", "n_embd", "d_model"):
        value = getattr(text_config, name, None)
        if value is not None:
            if type(value) is not int or value < 1:
                raise UnsupportedModel("text configuration has an invalid hidden size")
            return text_config, value
    raise UnsupportedModel("text configuration has no hidden size")


def load_model(model_ref: str | Path, *, dtype: torch.dtype, device: str | torch.device) -> LoadedModel:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
        raise UnsupportedModel("weight editing requires an explicit floating-point dtype")
    selected_device = torch.device(device)
    if selected_device.type not in {"cpu", "cuda"}:
        raise UnsupportedModel("initial weight editing supports CPU or a single CUDA device")
    checkpoint = resolve_checkpoint(model_ref, allow_download=False)
    try:
        config = AutoConfig.from_pretrained(checkpoint.path, local_files_only=True, trust_remote_code=False)
        text_config, hidden = text_configuration(config)
        if getattr(config, "quantization_config", None) is not None or getattr(text_config, "quantization_config", None) is not None:
            raise UnsupportedModel("packed quantized checkpoints require a separate dequantization path")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint.path, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint.path, config=config, local_files_only=True, trust_remote_code=False,
            torch_dtype=dtype, use_safetensors=True,
        ).to(selected_device).eval()
    except (ValueError, ImportError, OSError) as exc:
        raise UnsupportedModel("checkpoint cannot load under the standard local Transformers contract") from exc
    if resolve_checkpoint(checkpoint.path).file_digests != checkpoint.file_digests:
        raise UnsupportedModel("checkpoint files changed during loading")
    groups = storage_groups(model)
    names = _checkpoint_names(checkpoint.path, checkpoint.file_digests)
    aliases = {name for group in groups if set(group) & names for name in group}
    unexplained = set(model.state_dict()) - names - aliases
    if unexplained or names - set(model.state_dict()):
        raise UnsupportedModel("runtime tensor names do not match the checkpoint tensor index")
    return LoadedModel(model, tokenizer, config, text_config, model_state_digest(model), hidden,
                       checkpoint.path, dict(checkpoint.file_digests), groups, names)