"""Explicit numerical preparation for generic weight editing."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import torch

from .discover import Site
from .loader import LoadedModel, UnsupportedModel, WeightEditInputError


@contextmanager
def deterministic_execution(model: torch.nn.Module) -> Iterator[None]:
    if next(model.parameters()).device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise WeightEditInputError("set CUBLAS_WORKSPACE_CONFIG before CUDA initialization for deterministic weight editing")
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark, deterministic = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    matmul_tf32, cudnn_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = benchmark, deterministic
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = matmul_tf32, cudnn_tf32


def working_weight(loaded: LoadedModel, site: Site, *, compute_dtype: torch.dtype = torch.float64) -> torch.Tensor:
    from safetensors import safe_open
    from ...provenance import sha256_file

    if compute_dtype not in {torch.float32, torch.float64}:
        raise WeightEditInputError("edit computation requires float32 or float64")
    if site.tensor_name not in loaded.checkpoint_tensor_names:
        raise UnsupportedModel("working weight has no verified checkpoint tensor name")
    for filename, digest in loaded.file_digests.items():
        if not filename.endswith(".safetensors"):
            continue
        path = loaded.checkpoint_path / filename
        if sha256_file(path) != digest:
            raise UnsupportedModel("checkpoint weights changed before edit computation")
        with safe_open(path, framework="pt", device="cpu") as tensors:
            if site.tensor_name not in tensors.keys():
                continue
            weight = tensors.get_tensor(site.tensor_name).to(compute_dtype).clone()
        if sha256_file(path) != digest:
            raise UnsupportedModel("checkpoint weights changed while preparing edit computation")
        weight = weight.T if site.transposed else weight
        if tuple(weight.shape) != (site.output_size, site.input_size) or not bool(torch.isfinite(weight).all()):
            raise UnsupportedModel("working weight has invalid shape or nonfinite values")
        return weight.contiguous()
    raise UnsupportedModel("checkpoint working weight is missing")