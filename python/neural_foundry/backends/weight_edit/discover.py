"""Discover decoder stacks and linear weight sites without family-specific code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ...gym_service import _document_digest
from .loader import UnsupportedModel, text_configuration


@dataclass(frozen=True)
class BlockList:
    path: str
    count: int


@dataclass(frozen=True)
class Site:
    tensor_name: str
    layer: int
    role: str
    input_size: int
    output_size: int
    transposed: bool
    dtype: str

    def document(self) -> dict[str, Any]:
        return {"tensor_name": self.tensor_name, "layer": self.layer, "role": self.role,
                "in": self.input_size, "out": self.output_size, "transposed": self.transposed, "dtype": self.dtype}


@dataclass(frozen=True)
class SiteMap:
    sites: tuple[Site, ...]

    @property
    def digest(self) -> str:
        return _document_digest({"kind": "nf_weight_edit_sites_v1", "sites": [site.document() for site in self.sites]})


def find_blocks(model: nn.Module) -> BlockList:
    config, _ = text_configuration(model.config)
    layer_count = next((getattr(config, name) for name in ("num_hidden_layers", "n_layer")
                        if getattr(config, name, None) is not None), None)
    if layer_count is not None and (type(layer_count) is not int or layer_count < 1):
        raise UnsupportedModel("invalid configured transformer layer count")
    candidates = []
    for path, module in model.named_modules():
        if not isinstance(module, nn.ModuleList) or not len(module):
            continue
        if len({type(child) for child in module}) != 1 or (layer_count is not None and len(module) != layer_count):
            continue
        size = sum(parameter.numel() for parameter in module.parameters())
        if size:
            candidates.append((size, path, len(module)))
    if not candidates:
        raise UnsupportedModel("no unambiguous transformer blocks")
    largest = max(candidate[0] for candidate in candidates)
    winners = [candidate for candidate in candidates if candidate[0] == largest]
    if len(winners) != 1:
        raise UnsupportedModel("ambiguous blocks")
    _, path, count = winners[0]
    return BlockList(path, count)


def find_sites(model: nn.Module, blocks: BlockList, hidden: int) -> SiteMap:
    from transformers.pytorch_utils import Conv1D

    if type(hidden) is not int or hidden < 1:
        raise UnsupportedModel("invalid hidden size for site discovery")
    stack = model.get_submodule(blocks.path)
    if not isinstance(stack, nn.ModuleList) or len(stack) != blocks.count:
        raise UnsupportedModel("discovered block structure changed")
    sites = []
    for layer, block in enumerate(stack):
        for relative, module in block.named_modules():
            if not isinstance(module, (nn.Linear, Conv1D)):
                continue
            weight = module.weight
            if weight.ndim != 2 or not torch.is_floating_point(weight):
                raise UnsupportedModel("site has packed or non-matrix weights")
            transposed = isinstance(module, Conv1D)
            inputs, outputs = weight.shape if transposed else reversed(weight.shape)
            if outputs == hidden and inputs != hidden:
                role = "residual_write_mlp"
            elif outputs == hidden and inputs == hidden:
                role = "residual_write_unknown"
            elif inputs == hidden and outputs == 3 * hidden:
                role = "residual_read_fused"
            elif inputs == hidden:
                role = "residual_read"
            else:
                role = "unknown"
            path = ".".join(part for part in (blocks.path, str(layer), relative, "weight") if part)
            sites.append(Site(path, layer, role, int(inputs), int(outputs), transposed, str(weight.dtype)))
    if not sites:
        raise UnsupportedModel("transformer blocks have no standard linear weight sites")
    return SiteMap(tuple(sorted(sites, key=lambda site: (site.layer, site.tensor_name))))


def check_profile(sites: SiteMap, profile: Any, blocks: BlockList) -> None:
    discovered = {site.tensor_name: site for site in sites.sites}
    checks = (("mlp_down_proj_site", {"residual_write_mlp"}),
              ("mlp_up_proj_site", {"residual_read", "residual_read_fused"}),
              ("attn_out_site", {"residual_write_attn", "residual_write_unknown"}))
    for layer in range(blocks.count):
        for method, roles in checks:
            tensor_name = getattr(profile, method)(layer)
            site = discovered.get(tensor_name)
            if site is None or site.role not in roles:
                raise UnsupportedModel("discovered sites differ from the optional architecture profile")