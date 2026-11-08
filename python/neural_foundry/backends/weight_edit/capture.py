"""Capture real-runtime site activations with scoped forward hooks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .discover import SiteMap
from .loader import LoadedModel, UnsupportedModel


@dataclass(frozen=True)
class Activations:
    inputs: torch.Tensor
    outputs: torch.Tensor
    output_gradients: torch.Tensor | None = None


def capture(
    loaded: LoadedModel, sites: SiteMap, prompts: Sequence[str],
    positions: Sequence[Sequence[int]] | None = None, *,
    response_starts: Sequence[int] | None = None,
    loss_labels: Sequence[Sequence[int]] | None = None,
) -> dict[str, Activations]:
    if not prompts or isinstance(prompts, (str, bytes)) or any(not isinstance(prompt, str) for prompt in prompts):
        raise UnsupportedModel("capture requires a nonempty prompt sequence")
    if not sites.sites or len({site.tensor_name for site in sites.sites}) != len(sites.sites):
        raise UnsupportedModel("capture requires unique sites")
    if loaded.model.training:
        raise UnsupportedModel("activation capture requires an evaluation-mode model")
    for values in (positions, response_starts, loss_labels):
        if values is not None and len(values) != len(prompts):
            raise UnsupportedModel("capture positions or labels do not match the prompt count")
    if positions is not None and response_starts is not None:
        raise UnsupportedModel("choose explicit positions or every response token, not both")
    handles = []
    active: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    collected = {site.tensor_name: [[], [], []] for site in sites.sites}
    gradient_mode = loss_labels is not None

    def hook_for(site: Any):
        name = site.tensor_name
        def hook(module: Any, args: Any, output: Any) -> None:
            if name in active or not args or not isinstance(args[0], torch.Tensor) or not isinstance(output, torch.Tensor):
                raise UnsupportedModel("site was called repeatedly or returned unsupported activations")
            if args[0].ndim != 3 or output.ndim != 3 or args[0].shape[:2] != output.shape[:2]:
                raise UnsupportedModel("site activations are not batch-by-token matrices")
            if args[0].shape[-1] != site.input_size or output.shape[-1] != site.output_size:
                raise UnsupportedModel("site activation dimensions differ from the discovered map")
            if gradient_mode:
                output.retain_grad()
            active[name] = (args[0], output)
        return hook

    try:
        for site in sites.sites:
            module = loaded.model.get_submodule(site.tensor_name.removesuffix(".weight"))
            handles.append(module.register_forward_hook(hook_for(site)))
        device = next(loaded.model.parameters()).device
        for index, prompt in enumerate(prompts):
            active.clear()
            encoded = loaded.tokenizer(prompt, return_tensors="pt").to(device)
            length = int(encoded["input_ids"].shape[1])
            if response_starts is not None:
                start = response_starts[index]
                if type(start) is not int or not 0 <= start < length:
                    raise UnsupportedModel("response token start is outside the prompt")
                selected = list(range(start, length))
            else:
                selected = list(positions[index]) if positions is not None else [length - 1]
            if not selected or any(type(position) is not int or not 0 <= position < length for position in selected):
                raise UnsupportedModel("capture token positions are outside the prompt")
            with torch.enable_grad() if gradient_mode else torch.no_grad():
                if gradient_mode:
                    labels = torch.tensor([loss_labels[index]], dtype=torch.long, device=device)
                    if labels.shape != encoded["input_ids"].shape:
                        raise UnsupportedModel("loss labels do not match tokenized input")
                    result = loaded.model(**encoded, labels=labels)
                else:
                    result = loaded.model(**encoded)
                if set(active) != set(collected):
                    raise UnsupportedModel("not every selected site ran during capture")
                if gradient_mode:
                    if result.loss.ndim != 0 or not bool(torch.isfinite(result.loss)):
                        raise UnsupportedModel("capture loss is not finite")
                    gradients = torch.autograd.grad(result.loss, [active[name][1] for name in collected])
                else:
                    gradients = [None] * len(collected)
                for name, gradient in zip(collected, gradients, strict=True):
                    inputs, outputs = active[name]
                    collected[name][0].append(inputs[0, selected].detach().to(device="cpu", dtype=torch.float32))
                    collected[name][1].append(outputs[0, selected].detach().to(device="cpu", dtype=torch.float32))
                    if gradient is not None:
                        collected[name][2].append(gradient[0, selected].detach().to(device="cpu", dtype=torch.float32))
        return {name: Activations(torch.cat(values[0]), torch.cat(values[1]),
                                  torch.cat(values[2]) if values[2] else None) for name, values in collected.items()}
    finally:
        for handle in handles:
            handle.remove()