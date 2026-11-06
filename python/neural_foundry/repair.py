"""Diagnose a repetition attractor from model behaviour, and repair it with a small direct edit.

Nothing here is told where a defect is. The attractor is identified from predictions alone, and
candidates are chosen using only the prompts passed in, so callers must keep sealed data out.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .method_plans import VerifiedMethodSteps, low_rank_step, verify_method_steps
from .pythia import _compile_and_apply_rust_low_rank_update


@dataclass(frozen=True)
class AttractorDiagnosis:
    found: bool
    token_id: int | None
    token: str | None
    prompt_share: float
    prompts_examined: int


@dataclass(frozen=True)
class RepairCandidate:
    layer: int
    site: str
    factor: float
    attractor_share: float
    unaffected_predictions_kept: float


@dataclass(frozen=True)
class AttractorRepair:
    diagnosis: AttractorDiagnosis
    candidates_tried: tuple[RepairCandidate, ...]
    selected: RepairCandidate | None
    edited_weight: torch.Tensor | None
    plan_digest: str
    candidate_digest: str
    edit_steps: VerifiedMethodSteps | tuple[()] = ()


def diagnose_attractor(
    model: Any,
    prompts: Sequence[str],
    *,
    min_share: float = 0.5,
    override: dict[int, torch.Tensor] | None = None,
) -> AttractorDiagnosis:
    """Report the token that dominates predictions across unrelated prompts, if there is one."""
    if not prompts:
        raise ValueError("prompts must be non-empty")
    predictions = [model.evaluate_prompt(p, layer_down_proj_override=override) for p in prompts]
    token_id, count = Counter(p.top1_token_id for p in predictions).most_common(1)[0]
    share = count / len(predictions)
    if share < min_share:
        return AttractorDiagnosis(False, None, None, share, len(predictions))
    token = next(p.top1_token for p in predictions if p.top1_token_id == token_id)
    return AttractorDiagnosis(True, int(token_id), token, share, len(predictions))


def output_direction_for_token(model: Any, token_id: int) -> torch.Tensor:
    vector = model.effective_unembedding()[token_id].to(torch.float64)
    return vector / torch.linalg.norm(vector)


def output_projection_factors(weight: torch.Tensor, direction: torch.Tensor, *, factor: float) -> tuple[torch.Tensor, torch.Tensor]:
    w = weight.to(torch.float64)
    return (-factor * direction).unsqueeze(1).contiguous(), (direction @ w).unsqueeze(0).contiguous()


def project_out_output_direction(
    weight: torch.Tensor,
    direction: torch.Tensor,
    *,
    factor: float,
    smoke_probe: torch.Tensor,
) -> tuple[torch.Tensor, str, str]:
    """Remove `factor` of `direction` from the weight's output, through the Rust executor."""
    w = weight.to(torch.float64)
    left, right = output_projection_factors(w, direction, factor=factor)
    edited, plan, candidate, _, _, reload_ok = _compile_and_apply_rust_low_rank_update(
        w_orig=w,
        left_factors=left,
        right_factors=right,
        smoke_probe_vec=smoke_probe.to(torch.float64),
    )
    if not reload_ok:
        raise ValueError("Rust executor reload parity failed")
    return edited, plan, candidate


def calibrated_factor(model: Any, weight: torch.Tensor, direction: torch.Tensor, *, samples: int = 64) -> float:
    """Factor that shrinks a token's output row to the median size of other tokens' rows."""
    w = weight.to(torch.float64)
    unembedding = model.effective_unembedding().to(torch.float64)
    step = max(1, int(unembedding.shape[0]) // samples)
    others = unembedding[::step]
    others = others / torch.linalg.norm(others, dim=1, keepdim=True).clamp_min(1e-12)
    typical = float(torch.linalg.norm(others @ w, dim=1).median())
    size = float(torch.linalg.norm(direction @ w))
    if size <= typical:
        return 0.0
    return 1.0 - typical / size


def smoke_probe_for(model: Any, layer: int) -> torch.Tensor:
    return model.forward_with_mlp_key(model.tokenizer.encode("Hello world"), capture_layer=layer)[1]


def repair_repetition_attractor(
    model: Any,
    development_prompts: Sequence[str],
    *,
    factors: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
    min_share: float = 0.5,
) -> AttractorRepair:
    """Sweep sites and strengths on development prompts; return the mildest edit with the lowest attractor share.

    A candidate must lower the attractor share, or nothing is selected. Finding nothing is a valid result.
    """
    diagnosis = diagnose_attractor(model, development_prompts, min_share=min_share)
    if not diagnosis.found or diagnosis.token_id is None:
        return AttractorRepair(diagnosis, (), None, None, "", "")

    before = {p: model.evaluate_prompt(p).top1_token_id for p in development_prompts}
    unaffected = [p for p, token in before.items() if token != diagnosis.token_id]
    direction = output_direction_for_token(model, diagnosis.token_id)

    tried: list[RepairCandidate] = []
    best: tuple[tuple[float, float, float], RepairCandidate, torch.Tensor, str, str] | None = None
    for layer in range(int(model.config["num_hidden_layers"])):
        site = model.mlp_down_proj_site(layer)
        probe = smoke_probe_for(model, layer)
        calibrated = calibrated_factor(model, model.tensors[site], direction)
        for factor in sorted({*factors, *([calibrated] if calibrated > 0.0 else [])}):
            edited, plan, candidate = project_out_output_direction(
                model.tensors[site], direction, factor=float(factor), smoke_probe=probe
            )
            override = {layer: edited}
            predictions = {p: model.evaluate_prompt(p, layer_down_proj_override=override).top1_token_id for p in development_prompts}
            share = sum(t == diagnosis.token_id for t in predictions.values()) / len(predictions)
            kept = sum(predictions[p] == before[p] for p in unaffected) / len(unaffected) if unaffected else 1.0
            record = RepairCandidate(layer, site, float(factor), share, kept)
            tried.append(record)
            # Once the token no longer dominates, the mildest edit wins: it may legitimately be predicted sometimes.
            rank = (0.0 if share < min_share else share, -kept, float(factor))
            if share < diagnosis.prompt_share and (best is None or rank < best[0]):
                best = (rank, record, edited, plan, candidate)

    if best is None:
        return AttractorRepair(diagnosis, tuple(tried), None, None, "", "")
    chosen = best[1]
    left, right = output_projection_factors(model.tensors[chosen.site], direction, factor=chosen.factor)
    steps = verify_method_steps(model, (low_rank_step(chosen.site, "attractor_projection", left, right),), {chosen.site: best[2]})
    return AttractorRepair(diagnosis, tuple(tried), chosen, best[2], best[3], best[4], steps)


def sham_output_projection(
    model: Any,
    layer: int,
    *,
    factor: float,
    seed: int = 17,
) -> torch.Tensor:
    """Same-shaped edit along a random direction: if this helps equally, the diagnosis gets no credit."""
    weight = model.tensors[model.mlp_down_proj_site(layer)].to(torch.float64)
    generator = torch.Generator().manual_seed(seed)
    direction = torch.randn(weight.shape[0], dtype=torch.float64, generator=generator)
    direction = direction / torch.linalg.norm(direction)
    edited, _, _ = project_out_output_direction(
        weight, direction, factor=factor, smoke_probe=smoke_probe_for(model, layer)
    )
    return edited
