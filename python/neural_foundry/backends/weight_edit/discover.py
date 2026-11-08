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
    checkpoint_names_verified: bool = False

    @property
    def digest(self) -> str:
        return _document_digest({"kind": "nf_weight_edit_sites_v1", "sites": [site.document() for site in self.sites],
                                 "checkpoint_names_verified": self.checkpoint_names_verified,
                                 "role_method": "config_structure_branch_ablation_v1"})


def _account_block_parameters(block: nn.Module) -> None:
    from transformers.pytorch_utils import Conv1D

    accounted = set()
    for path, module in block.named_modules():
        parameters = dict(module.named_parameters(recurse=False, remove_duplicate=False))
        if isinstance(module, (nn.Linear, Conv1D)):
            known = {"weight", "bias"}
        elif (isinstance(module, nn.LayerNorm) or isinstance(module, getattr(nn, "RMSNorm", nn.LayerNorm))
              or (hasattr(module, "variance_epsilon") or hasattr(module, "eps"))
              and parameters and all(parameter.ndim <= 1 for parameter in parameters.values())):
            known = {"weight", "bias"}
        else:
            known = {"bias"} if parameters.get("bias") is not None and parameters["bias"].ndim <= 1 else set()
        for name in parameters.keys() & known:
            accounted.add(".".join(part for part in (path, name) if part))
    unexplained = set(dict(block.named_parameters(remove_duplicate=False))) - accounted
    if unexplained:
        raise UnsupportedModel("unaccounted block parameters: " + ", ".join(sorted(unexplained)))


def _projection_shape(module: nn.Module) -> tuple[int, int]:
    from transformers.pytorch_utils import Conv1D

    shape = module.weight.shape
    return (int(shape[0]), int(shape[1])) if isinstance(module, Conv1D) else (int(shape[1]), int(shape[0]))


def _attention_sizes(config: Any, hidden: int) -> tuple[int, int, int] | None:
    heads = getattr(config, "num_attention_heads", getattr(config, "n_head", None))
    kv_heads = getattr(config, "num_key_value_heads", None) or heads
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None and type(heads) is int and heads > 0 and hidden % heads == 0:
        head_dim = hidden // heads
    if any(type(value) is not int or value < 1 for value in (heads, kv_heads, head_dim)):
        return None
    query, kv = heads * head_dim, kv_heads * head_dim
    return query, kv, query + 2 * kv


def _branch_roles(block: nn.Module, hidden: int, config: Any) -> dict[str, tuple[str, str]]:
    from transformers.pytorch_utils import Conv1D

    linears = {name: module for name, module in block.named_modules() if isinstance(module, (nn.Linear, Conv1D))}
    sizes = _attention_sizes(config, hidden)
    assignments: dict[str, tuple[str, str]] = {}
    branches = sorted(block.named_modules(), key=lambda item: item[0].count("."), reverse=True)
    for branch_name, branch in branches:
        if not branch_name or isinstance(branch, (nn.Linear, Conv1D)):
            continue
        members = {name: module for name, module in linears.items() if name.startswith(branch_name + ".")}
        if not members or any(name in assignments for name in members):
            continue
        shapes = {name: _projection_shape(module) for name, module in members.items()}
        if sizes is not None:
            query, kv, fused = sizes
            attention_writers = []
            combined = False
            for writer in members:
                remaining = {name: shape for name, shape in shapes.items() if name != writer}
                separate = len(remaining) == 3 and sorted(remaining.values()) == sorted([(hidden, query), (hidden, kv), (hidden, kv)])
                combined = len(remaining) == 1 and list(remaining.values()) == [(hidden, fused)]
                if shapes[writer] == (query, hidden) and (separate or combined):
                    attention_writers.append(writer)
            if attention_writers:
                for name in members:
                    role = "attention_writer_candidate" if name in attention_writers else ("residual_read_fused" if len(members) == 2 else "residual_read")
                    assignments[name] = (role, branch_name)
                continue
        writers = [name for name, (inputs, outputs) in shapes.items() if outputs == hidden and inputs != hidden]
        for writer in writers:
            intermediate = shapes[writer][0]
            reads = [shape for name, shape in shapes.items() if name != writer]
            if (len(reads) == 1 and reads[0] in {(hidden, intermediate), (hidden, 2 * intermediate)}
                or len(reads) == 2 and all(shape == (hidden, intermediate) for shape in reads)):
                assignments[writer] = ("mlp_writer_candidate", branch_name)
                for name in members:
                    if name != writer:
                        assignments[name] = ("residual_read", branch_name)
                break
    return assignments


def _confirm_writers(model: nn.Module, candidates: dict[str, tuple[str, str]], config: Any) -> set[str]:
    if not candidates:
        return set()
    if model.training:
        raise UnsupportedModel("role ablation requires an evaluation-mode model")
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def observe(name: str):
        def hook(module: Any, args: Any, output: Any) -> None:
            value = output[0] if isinstance(output, tuple) else output
            if not isinstance(value, torch.Tensor) or name in captured:
                raise UnsupportedModel("ambiguous branch output during role ablation")
            captured[name] = value.detach().clone()
        return hook

    try:
        names = set(candidates) | {branch for _, branch in candidates.values()}
        for name in names:
            handles.append(model.get_submodule(name).register_forward_hook(observe(name)))
        token = getattr(config, "bos_token_id", None)
        if type(token) is not int or not 0 <= token < getattr(config, "vocab_size", 0):
            token = 0
        inputs = torch.tensor([[token, token]], device=next(model.parameters()).device)
        with torch.no_grad():
            model(input_ids=inputs)
        baseline = captured.copy()
        confirmed = set()
        for name, (_, branch) in candidates.items():
            if name not in baseline or branch not in baseline or not torch.equal(baseline[name], baseline[branch]):
                continue
            zero_hook = model.get_submodule(name).register_forward_hook(lambda module, args, output: torch.zeros_like(output))
            try:
                captured.clear()
                with torch.no_grad():
                    model(input_ids=inputs)
                if branch in captured and bool(torch.equal(captured[branch], torch.zeros_like(captured[branch]))):
                    confirmed.add(name)
            finally:
                zero_hook.remove()
        return confirmed
    finally:
        for handle in handles:
            handle.remove()


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


def find_sites(model: nn.Module, blocks: BlockList, hidden: int, *, checkpoint_tensor_names: frozenset[str] | None = None) -> SiteMap:
    from transformers.pytorch_utils import Conv1D

    if type(hidden) is not int or hidden < 1:
        raise UnsupportedModel("invalid hidden size for site discovery")
    stack = model.get_submodule(blocks.path)
    if not isinstance(stack, nn.ModuleList) or len(stack) != blocks.count:
        raise UnsupportedModel("discovered block structure changed")
    config, configured_hidden = text_configuration(model.config)
    if hidden != configured_hidden:
        raise UnsupportedModel("discovery hidden size differs from the text configuration")
    roles = {}
    for layer, block in enumerate(stack):
        _account_block_parameters(block)
        for relative, (role, branch) in _branch_roles(block, hidden, config).items():
            roles[f"{blocks.path}.{layer}.{relative}"] = (role, f"{blocks.path}.{layer}.{branch}")
    candidates = {name: value for name, value in roles.items() if value[0].endswith("writer_candidate")}
    confirmed = _confirm_writers(model, candidates, config)
    for branch in {branch for _, branch in candidates.values()}:
        if sum(name in confirmed for name, (_, owner) in candidates.items() if owner == branch) > 1:
            raise UnsupportedModel("ambiguous projection writer after ablation")
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
            path = ".".join(part for part in (blocks.path, str(layer), relative, "weight") if part)
            module_path = path.removesuffix(".weight")
            role = roles.get(module_path, ("unknown", ""))[0]
            experts = next((getattr(config, name) for name in ("num_local_experts", "num_experts", "n_routed_experts")
                            if type(getattr(config, name, None)) is int and getattr(config, name) > 1), None)
            if experts is not None and inputs == hidden and outputs == experts:
                role = "router"
            if role.endswith("writer_candidate"):
                branch = roles[module_path][1]
                if module_path in confirmed:
                    role = "residual_write_attn" if role == "attention_writer_candidate" else "residual_write_mlp"
                elif role == "attention_writer_candidate" and any(owner == branch for name, (_, owner) in candidates.items() if name in confirmed):
                    role = "residual_read"
                else:
                    role = "residual_write_unknown"
            sites.append(Site(path, layer, role, int(inputs), int(outputs), transposed, str(weight.dtype)))
    if not sites:
        raise UnsupportedModel("transformer blocks have no standard linear weight sites")
    if checkpoint_tensor_names is not None and any(site.tensor_name not in checkpoint_tensor_names for site in sites):
        raise UnsupportedModel("discovered site names differ from the checkpoint tensor index")
    return SiteMap(tuple(sorted(sites, key=lambda site: (site.layer, site.tensor_name))), checkpoint_tensor_names is not None)


def check_profile(sites: SiteMap, profile: Any, blocks: BlockList) -> None:
    discovered = {site.tensor_name: site for site in sites.sites}
    checks = (("mlp_down_proj_site", {"residual_write_mlp"}),
              ("mlp_up_proj_site", {"residual_read", "residual_read_fused"}),
              ("attn_out_site", {"residual_write_attn"}))
    for layer in range(blocks.count):
        for method, roles in checks:
            tensor_name = getattr(profile, method)(layer)
            site = discovered.get(tensor_name)
            if site is None or site.role not in roles:
                raise UnsupportedModel("discovered sites differ from the optional architecture profile")