"""The repair must work from behaviour alone and must not invent a defect."""

from __future__ import annotations

import inspect

import torch

from neural_foundry import pythia, repair

GYM = pythia.ALGORITHMIC_THREE_SKILL_GYM
PROMPTS = (*(f"Task: {e}" for e, _ in GYM.base_rules), *GYM.retention_prompts)


def _with_defect(model):  # type: ignore[no-untyped-def]
    from neural_foundry.export_formats import _apply_state_tensors_to_clone, _extract_model_state_tensors

    layer = int(model.config["num_hidden_layers"]) - 1
    site = model.mlp_down_proj_site(layer)
    token = model.tokenizer.vocab["\u0120the"]
    direction = repair.output_direction_for_token(model, token)
    key = torch.stack(
        [model.forward_with_mlp_key(model.tokenizer.encode(p), capture_layer=layer)[1] for p in PROMPTS]
    ).mean(0)
    state = _extract_model_state_tensors(model)
    state[site] = state[site] + 120.0 * torch.outer(direction, key / torch.linalg.norm(key))
    return _apply_state_tensors_to_clone(model, state), token


def test_repair_signature_takes_no_defect_location() -> None:
    assert set(inspect.signature(repair.repair_repetition_attractor).parameters) == {
        "model", "development_prompts", "factors", "min_share",
    }


def test_repair_finds_and_reduces_injected_attractor() -> None:
    defective, token = _with_defect(pythia.build_mini_pythia_model())
    result = repair.repair_repetition_attractor(defective, PROMPTS)
    assert result.diagnosis.token_id == token
    assert result.selected is not None
    assert result.selected.attractor_share < result.diagnosis.prompt_share
    assert result.candidate_digest.startswith("sha256:")


def test_healthy_model_is_left_alone() -> None:
    model = pythia.build_mini_pythia_model()
    diagnosis = repair.diagnose_attractor(model, PROMPTS)
    result = repair.repair_repetition_attractor(model, PROMPTS)
    if not diagnosis.found:
        assert result.selected is None and result.edited_weight is None


def test_sham_edit_does_not_remove_attractor() -> None:
    defective, token = _with_defect(pythia.build_mini_pythia_model())
    layer = int(defective.config["num_hidden_layers"]) - 1
    sham = repair.sham_output_projection(defective, layer, factor=1.0)
    after = repair.diagnose_attractor(defective, PROMPTS, override={layer: sham})
    assert after.found and after.token_id == token
