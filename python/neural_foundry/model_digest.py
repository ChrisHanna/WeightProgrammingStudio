"""Deterministic tensor-state identity shared by isolated model workers."""

from __future__ import annotations

import hashlib
from typing import Any


def model_state_digest(model: Any) -> str:
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}\0".encode())
        digest.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return "sha256:" + digest.hexdigest()