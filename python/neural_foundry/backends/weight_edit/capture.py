"""Capture real-runtime site activations with scoped forward hooks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from contextlib import ExitStack
from typing import Any
import hashlib

import torch

from .discover import SiteMap
from ...prompt_rendering import PromptInputError, PromptRenderer, RenderedPrompt
from .loader import LoadedModel, UnsupportedModel, WeightEditInputError
from .numerics import deterministic_execution


@dataclass(frozen=True)
class Activations:
    inputs: torch.Tensor
    outputs: torch.Tensor
    output_gradients: torch.Tensor | None = None
    renderer_digest: str = ""
    prompt_digests: tuple[str, ...] = ()


def capture(
    loaded: LoadedModel, sites: SiteMap, prompts: Sequence[str | Mapping[str, Any] | RenderedPrompt],
    positions: Sequence[Sequence[int]] | None = None, *,
    response_starts: Sequence[int] | None = None,
    loss_labels: Sequence[Sequence[int]] | None = None,
    output_dtype: torch.dtype = torch.float32,
    max_capture_bytes: int = 256 * 1024 * 1024,
    sink: Callable[[str, torch.Tensor, torch.Tensor, torch.Tensor | None, str], None] | None = None,
) -> dict[str, Activations]:
    if output_dtype not in {torch.float32, torch.float64}:
        raise WeightEditInputError("capture output dtype must be float32 or float64")
    element_size = torch.empty((), dtype=output_dtype).element_size()
    if type(max_capture_bytes) is not int or max_capture_bytes < 1 or (sink is not None and not callable(sink)):
        raise WeightEditInputError("capture requires a positive byte budget and a callable sink")
    if not prompts or isinstance(prompts, (str, bytes)) or any(not isinstance(prompt, (str, Mapping, RenderedPrompt)) for prompt in prompts):
        raise WeightEditInputError("capture requires a nonempty prompt sequence")
    if not sites.sites or len({site.tensor_name for site in sites.sites}) != len(sites.sites):
        raise WeightEditInputError("capture requires unique sites")
    if loaded.model.training:
        raise UnsupportedModel("activation capture requires an evaluation-mode model")
    for values in (positions, response_starts, loss_labels):
        if values is not None and len(values) != len(prompts):
            raise WeightEditInputError("capture positions or labels do not match the prompt count")
    if positions is not None and response_starts is not None:
        raise WeightEditInputError("choose explicit positions or every response token, not both")
    renderer = PromptRenderer(loaded.tokenizer)
    prompt_digests = []
    captured_bytes = 0
    handles = []
    execution = ExitStack()
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
        execution.enter_context(deterministic_execution(loaded.model))
        for site in sites.sites:
            module = loaded.model.get_submodule(site.tensor_name.removesuffix(".weight"))
            handles.append(module.register_forward_hook(hook_for(site)))
        device = next(loaded.model.parameters()).device
        for index, prompt in enumerate(prompts):
            active.clear()
            try:
                if isinstance(prompt, RenderedPrompt):
                    if (prompt.renderer_digest != renderer.digest or not prompt.token_ids
                        or any(type(token) is not int or token < 0 for token in prompt.token_ids)):
                        raise PromptInputError("pre-rendered demonstration differs from the tokenizer rendering policy")
                    rendered = prompt
                else:
                    rendered = renderer.render(prompt)
            except PromptInputError as exc:
                raise WeightEditInputError(str(exc)) from exc
            if sink is None:
                prompt_digests.append(rendered.prompt_digest)
            tokens = torch.tensor([rendered.token_ids], dtype=torch.long, device=device)
            encoded = {"input_ids": tokens, "attention_mask": torch.ones_like(tokens)}
            length = int(encoded["input_ids"].shape[1])
            if response_starts is not None:
                start = response_starts[index]
                if type(start) is not int or not 0 <= start < length:
                    raise WeightEditInputError("response token start is outside the rendered prompt")
                selected = list(range(start, length))
            else:
                selected = list(positions[index]) if positions is not None else [length - 1]
            if not selected or any(type(position) is not int or not 0 <= position < length for position in selected):
                raise WeightEditInputError("capture token positions are outside the rendered prompt")
            chunk_bytes = sum(len(selected) * (site.input_size + site.output_size * (2 if gradient_mode else 1)) * element_size for site in sites.sites)
            if chunk_bytes + (captured_bytes if sink is None else 0) > max_capture_bytes:
                raise WeightEditInputError("activation capture exceeds its byte budget; use streaming statistics or fewer positions")
            captured_bytes += chunk_bytes
            with torch.enable_grad() if gradient_mode else torch.no_grad():
                if gradient_mode:
                    labels = torch.tensor([loss_labels[index]], dtype=torch.long, device=device)
                    if labels.shape != encoded["input_ids"].shape:
                        raise WeightEditInputError("loss labels do not match rendered tokenized input")
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
                    captured_inputs = inputs[0, selected].detach().to(device="cpu", dtype=output_dtype)
                    captured_outputs = outputs[0, selected].detach().to(device="cpu", dtype=output_dtype)
                    captured_gradient = gradient[0, selected].detach().to(device="cpu", dtype=output_dtype) if gradient is not None else None
                    if sink is not None:
                        sink(name, captured_inputs, captured_outputs, captured_gradient, rendered.prompt_digest)
                    else:
                        collected[name][0].append(captured_inputs)
                        collected[name][1].append(captured_outputs)
                        if captured_gradient is not None:
                            collected[name][2].append(captured_gradient)
        if sink is not None:
            return {}
        return {name: Activations(torch.cat(values[0]), torch.cat(values[1]),
                                  torch.cat(values[2]) if values[2] else None, renderer.digest,
                                  tuple(prompt_digests)) for name, values in collected.items()}
    finally:
        for handle in handles:
            handle.remove()
        execution.close()


@dataclass
class CovarianceStatistics:
    count: int
    input_sum: torch.Tensor
    output_sum: torch.Tensor
    input_gram: torch.Tensor
    input_output_cross: torch.Tensor
    prompt_digest: str = ""
    renderer_digest: str = ""


def capture_statistics(
    loaded: LoadedModel, sites: SiteMap, prompts: Sequence[str | Mapping[str, Any]],
    positions: Sequence[Sequence[int]] | None = None, *, response_starts: Sequence[int] | None = None,
    compute_dtype: torch.dtype = torch.float64, max_statistics_bytes: int = 256 * 1024 * 1024,
) -> dict[str, CovarianceStatistics]:
    if compute_dtype not in {torch.float32, torch.float64}:
        raise WeightEditInputError("statistics must accumulate in float32 or float64")
    element_size = torch.empty((), dtype=compute_dtype).element_size()
    needed = sum((site.input_size ** 2 + site.input_size * site.output_size + site.input_size + site.output_size) * element_size for site in sites.sites)
    if type(max_statistics_bytes) is not int or needed > max_statistics_bytes or max_statistics_bytes < 1:
        raise WeightEditInputError("covariance statistics exceed their byte budget")
    statistics = {site.tensor_name: CovarianceStatistics(0, torch.zeros(site.input_size, dtype=compute_dtype),
                    torch.zeros(site.output_size, dtype=compute_dtype), torch.zeros((site.input_size, site.input_size), dtype=compute_dtype),
                    torch.zeros((site.input_size, site.output_size), dtype=compute_dtype)) for site in sites.sites}
    hashes = {name: hashlib.sha256() for name in statistics}
    renderer_digest = PromptRenderer(loaded.tokenizer).digest

    def consume(name: str, inputs: torch.Tensor, outputs: torch.Tensor, gradient: Any, prompt_digest: str) -> None:
        values = statistics[name]
        inputs, outputs = inputs.to(compute_dtype), outputs.to(compute_dtype)
        if not bool(torch.isfinite(inputs).all()) or not bool(torch.isfinite(outputs).all()):
            raise UnsupportedModel("nonfinite activations cannot produce covariance statistics")
        values.count += inputs.shape[0]
        values.input_sum += inputs.sum(dim=0)
        values.output_sum += outputs.sum(dim=0)
        values.input_gram += inputs.T @ inputs
        values.input_output_cross += inputs.T @ outputs
        hashes[name].update(prompt_digest.encode())

    capture(loaded, sites, prompts, positions, response_starts=response_starts, output_dtype=compute_dtype, sink=consume)
    for name, values in statistics.items():
        values.prompt_digest = "sha256:" + hashes[name].hexdigest()
        values.renderer_digest = renderer_digest
    return statistics