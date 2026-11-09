from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from neural_foundry.backends.weight_edit.capture import capture
from neural_foundry.backends.weight_edit.discover import Site, SiteMap
from neural_foundry.backends.weight_edit.loader import WeightEditInputError


def _loaded():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
            with torch.no_grad():
                self.projection.weight.fill_(1.0)

        def forward(self, input_ids, attention_mask, labels=None):
            inputs = (input_ids.double() + 1e-12).unsqueeze(-1)
            outputs = self.projection(inputs)
            return SimpleNamespace(logits=outputs, loss=outputs.square().mean())

    tokenizer = SimpleNamespace(chat_template=None, special_tokens_map={}, get_vocab=lambda: {"token": 1},
                                 encode=lambda prompt, **kwargs: [1])
    return SimpleNamespace(model=Model().eval(), tokenizer=tokenizer)


def test_capture_preserves_requested_float64_precision_and_gradients() -> None:
    loaded = _loaded()
    sites = SiteMap((Site("projection.weight", 0, "residual_write_mlp", 1, 1, False, "torch.float64"),))
    default = capture(loaded, sites, ["token"])["projection.weight"]
    precise = capture(loaded, sites, ["token"], output_dtype=torch.float64, loss_labels=[[1]])["projection.weight"]
    assert default.inputs.dtype == torch.float32 and default.inputs.item() == 1.0
    assert precise.inputs.dtype == precise.outputs.dtype == precise.output_gradients.dtype == torch.float64
    assert precise.inputs.item() == precise.outputs.item() == 1.0 + 1e-12
    assert precise.output_gradients.item() == 2.0 * (1.0 + 1e-12)
    assert loaded.model.projection.weight.grad is None and not loaded.model.projection._forward_hooks


def test_capture_budget_uses_actual_output_element_size() -> None:
    loaded = _loaded()
    sites = SiteMap((Site("projection.weight", 0, "residual_write_mlp", 1, 1, False, "torch.float64"),))
    capture(loaded, sites, ["token"], max_capture_bytes=8)
    with pytest.raises(WeightEditInputError, match="byte budget"):
        capture(loaded, sites, ["token"], max_capture_bytes=8, output_dtype=torch.float64)
    with pytest.raises(WeightEditInputError, match="float32 or float64"):
        capture(loaded, sites, ["token"], output_dtype=torch.bfloat16)
    assert not loaded.model.projection._forward_hooks