"""T047: bounded weight/activation inspection tools, cap exhaustion, and hook cleanup."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from neural_foundry.inspection import (
    InspectionCap,
    InspectionError,
    SiteDescriptor,
)
from neural_foundry.pretrained import intake_local_qwen2, write_mini_qwen2_checkpoint


def test_bounded_inspection_slices_summaries_and_declared_sites(tmp_path: Path) -> None:
    ckpt_dir = write_mini_qwen2_checkpoint(tmp_path / "mini-qwen2")
    bundle = intake_local_qwen2(ckpt_dir)
    inspector = bundle.inspector(InspectionCap(max_bytes=4096, max_slice_elements=16))

    sites = inspector.declared_sites()
    site_names = {s["name"] for s in sites}
    assert {
        "embed_tokens",
        "layers.0.self_attn",
        "layers.0.mlp.down_proj",
        "layers.0.residual",
        "lm_head",
    }.issubset(site_names)

    sliced = inspector.inspect_slice("layers.0.mlp.down_proj", start=(0, 0), shape=(2, 4))
    assert sliced["shape"] == [2, 4]
    assert len(sliced["values"]) == 8
    assert sliced["bytes_charged"] == 64
    assert sliced["state_preserved"] is True
    assert sliced["evidence_label"]["support"] == "suggestive"

    summary = inspector.summarize_site("layers.0.self_attn")
    assert summary["shape"] == [4, 4]
    assert summary["count"] == 16
    assert summary["l2_norm"] > 0.0
    assert summary["state_preserved"] is True

    with pytest.raises(InspectionError) as unknown_err:
        inspector.inspect_slice("undeclared_site", start=(0, 0), shape=(1, 1))
    assert unknown_err.value.code == "UNKNOWN_SITE"


def test_bounded_inspection_cap_exhaustion_and_hook_exceptions_preserve_state(
    tmp_path: Path,
) -> None:
    ckpt_dir = write_mini_qwen2_checkpoint(tmp_path / "mini-qwen2")
    bundle = intake_local_qwen2(ckpt_dir)
    model = bundle.build_torch_module()

    # 1. Slice element cap exhaustion
    small_slice_inspector = bundle.inspector(
        InspectionCap(max_bytes=65536, max_slice_elements=4)
    )
    with pytest.raises(InspectionError) as elem_cap:
        small_slice_inspector.inspect_slice(
            "layers.0.mlp.down_proj", start=(0, 0), shape=(2, 4)
        )
    assert elem_cap.value.code == "CAP_EXHAUSTED"

    # 2. Positive bounded activation capture removes hooks and preserves parameters/buffers
    inspector = bundle.inspector(
        InspectionCap(max_bytes=65536, max_examples=4, max_sites=4, max_tokens=16)
    )
    capture = inspector.capture_activations(
        model,
        site_names=("layers.0.self_attn", "layers.0.mlp.down_proj", "layers.1.residual"),
        examples=([1, 2, 3], [4, 5]),
        input_provenance="development:mini_qwen2_probe",
    )
    assert capture["hooks_removed"] is True
    assert capture["state_preserved"] is True
    assert capture["before_parameters"] == capture["after_parameters"]
    assert capture["total_tokens"] == 5
    assert len(capture["examples"]) == 2
    assert set(capture["examples"][0]["sites"]) == {
        "layers.0.self_attn",
        "layers.0.mlp.down_proj",
        "layers.1.residual",
    }
    for mod in model.modules():
        assert len(getattr(mod, "_forward_hooks", {})) == 0

    # 3. Token cap exhaustion still removes all hooks and preserves model state
    with pytest.raises(InspectionError) as tok_cap:
        inspector.capture_activations(
            model,
            site_names=("layers.0.self_attn",),
            examples=([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]),
            input_provenance="development:token_overflow",
        )
    assert tok_cap.value.code == "CAP_EXHAUSTED"
    for mod in model.modules():
        assert len(getattr(mod, "_forward_hooks", {})) == 0

    # 4. Hook exception removes hooks and preserves state on exit
    def raising_hook(site_name: str, _output: Any) -> None:
        if site_name == "layers.0.mlp.down_proj":
            raise RuntimeError("simulated hook failure")

    with pytest.raises(RuntimeError, match="simulated hook failure"):
        inspector.capture_activations(
            model,
            site_names=("layers.0.self_attn", "layers.0.mlp.down_proj"),
            examples=([1, 2],),
            input_provenance="development:hook_exception",
            custom_hook=raising_hook,
        )
    for mod in model.modules():
        assert len(getattr(mod, "_forward_hooks", {})) == 0

    # 5. Hook attempting in-place parameter mutation is caught as STATE_DRIFT and restored
    before_down = model.model.layers[0].mlp.down_proj.weight.detach().clone()

    def mutating_hook(_site_name: str, _output: Any) -> None:
        with torch.no_grad():
            model.model.layers[0].mlp.down_proj.weight.add_(99.0)

    with pytest.raises(InspectionError) as drift_err:
        inspector.capture_activations(
            model,
            site_names=("layers.0.mlp.down_proj",),
            examples=([1, 2],),
            input_provenance="development:mutating_hook",
            custom_hook=mutating_hook,
        )
    assert drift_err.value.code == "STATE_DRIFT"
    assert torch.equal(model.model.layers[0].mlp.down_proj.weight, before_down)
    for mod in model.modules():
        assert len(getattr(mod, "_forward_hooks", {})) == 0


def test_bounded_inspection_rejects_invalid_site_descriptors() -> None:
    with pytest.raises(InspectionError) as bad_kind:
        SiteDescriptor(
            name="bad",
            kind="unregistered_kind",
            tensor_name="w",
            module_path="m",
            shape=(2, 2),
            dtype="f64",
        )
    assert bad_kind.value.code == "INVALID_SITE"
