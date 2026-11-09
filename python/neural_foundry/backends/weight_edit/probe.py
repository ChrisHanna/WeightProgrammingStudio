"""Measured reload and serving qualification for generic weight edits."""

from __future__ import annotations

from pathlib import Path
import hashlib
from typing import Any

import torch

from ...model_digest import model_state_digest
from ...prompt_rendering import PromptRenderer
from ...gym_service import _document_digest, _wire_document
from .apply import apply, save, tensor_digests
from .capture import capture
from .discover import SiteMap, find_blocks, find_sites
from .loader import LoadedModel, UnsupportedModel, load_model
from .methods import Delta
from .numerics import deterministic_execution


IDENTITY_PROMPTS = (
    "Return the number 1.", "Return the number 2.", "Complete: red, green,", "Complete: one, two,",
    "Write one JSON key.", "Name a fruit.", "Name a weekday.", "Name a shape.",
    "Return a lowercase letter.", "Complete: The capital", "Write a boolean.", "Write a short word.",
    "Complete: A small", "Return an integer.", "Complete: Hello", "Write a punctuation mark.",
)


def record_identity_probes(loaded: LoadedModel) -> dict[str, Any]:
    renderer = PromptRenderer(loaded.tokenizer)
    records = []
    device = next(loaded.model.parameters()).device
    with deterministic_execution(loaded.model), torch.no_grad():
        for prompt in IDENTITY_PROMPTS:
            rendered = renderer.render(prompt)
            ids = torch.tensor([rendered.token_ids], dtype=torch.long, device=device)
            logits = loaded.model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0, -1].float()
            probabilities = torch.log_softmax(logits, dim=-1)
            token = int(logits.argmax())
            value = float(probabilities[token])
            if not torch.isfinite(probabilities[token]):
                raise UnsupportedModel("identity probe produced nonfinite log probabilities")
            records.append({"prompt_token_ids": list(rendered.token_ids), "prompt_digest": rendered.prompt_digest,
                            "greedy_token_id": token, "greedy_logprob": value})
    return {"kind": "nf_weight_identity_probes_v1", "model_digest": model_state_digest(loaded.model),
            "renderer_digest": renderer.digest, "records": records, "logprob_atol": 0.05,
            "precision": str(next(loaded.model.parameters()).dtype).removeprefix("torch.")}


def _logits(loaded: LoadedModel, prompts: tuple[str, ...]) -> list[torch.Tensor]:
    renderer = PromptRenderer(loaded.tokenizer)
    device = next(loaded.model.parameters()).device
    values = []
    with deterministic_execution(loaded.model), torch.no_grad():
        for prompt in prompts:
            rendered = renderer.render(prompt)
            ids = torch.tensor([rendered.token_ids], dtype=torch.long, device=device)
            values.append(loaded.model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits.detach().float().cpu())
    return values


def probe_model(model_ref: str | Path, out_dir: str | Path, *, dtype: torch.dtype, device: str,
                model_id: str, serving_factory: Any = None) -> dict[str, Any]:
    loaded = load_model(model_ref, dtype=dtype, device=device)
    blocks = find_blocks(loaded.model)
    sites = find_sites(loaded.model, blocks, loaded.hidden_size, checkpoint_tensor_names=loaded.checkpoint_tensor_names)
    writers = [site for site in sites.sites if site.role == "residual_write_mlp"]
    if not writers:
        raise UnsupportedModel("compatibility requires a verified MLP write site")
    site = writers[len(writers) // 2]
    prompts = IDENTITY_PROMPTS[:8]
    before = _logits(loaded, prompts)
    tensors_before = tensor_digests(loaded.model)
    zero = Delta("dense", D=torch.zeros((site.output_size, site.input_size), dtype=torch.float64))
    apply(loaded, {site.tensor_name: zero}, sites=sites, method="compatibility_zero", settings={})
    if any(not torch.equal(first, second) for first, second in zip(before, _logits(loaded, prompts), strict=True)):
        raise UnsupportedModel("zero edit changed logits")
    key = capture(loaded, SiteMap((site,), checkpoint_names_verified=True), [IDENTITY_PROMPTS[0]])[site.tensor_name].inputs[0].double()
    direction = torch.linspace(-1, 1, site.output_size, dtype=torch.float64)
    if not key.norm() or not direction.norm():
        raise UnsupportedModel("probe has no nonzero activation or target direction")
    weight_norm = float(loaded.model.get_parameter(site.tensor_name).detach().float().norm())
    strength = max(weight_norm, 1.0) * 0.001
    delta = Delta("low_rank", U=(direction / direction.norm() * strength).unsqueeze(1),
                  V=(key / key.norm()).unsqueeze(0))
    edited = apply(loaded, {site.tensor_name: delta}, sites=sites, method="compatibility_rank_one",
                   settings={"strength": strength, "relative_frobenius_strength": 0.001})
    after = _logits(loaded, prompts)
    difference = max(float((first - second).abs().max()) for first, second in zip(before, after, strict=True))
    if not 0 < difference <= 1.0:
        raise UnsupportedModel(f"tiny edit did not produce a measured bounded logit change: {difference}")
    if {name for name in tensors_before if tensors_before[name] != edited.after_tensors[name]} != {site.tensor_name}:
        raise UnsupportedModel("probe changed an unexpected tensor roster")
    identities = record_identity_probes(loaded)
    request_digest = "sha256:" + hashlib.sha256((loaded.base_digest + sites.digest).encode()).hexdigest()
    reference = save(edited, out_dir, model_id=model_id, request_digest=request_digest, identity_probes=identities)
    report = {"kind": "nf_weight_edit_compatibility_v1", "supported": False, "model_id": model_id,
              "precision": str(dtype).removeprefix("torch."),
              "base_digest": loaded.base_digest, "site_map_digest": sites.digest, "candidate": vars(reference),
              "zero_edit_exact": True, "max_logit_change": difference, "unchanged_tensors_verified": True,
              "reload_verified": True, "serving_verified": False, "identity_probe_count": len(identities["records"])}
    if serving_factory is not None:
        if next(loaded.model.parameters()).device.type == "cuda":
            loaded.model.to("cpu")
            torch.cuda.empty_cache()
        with serving_factory(reference) as served:
            identity = served.loaded_identity()
            if identity.get("served_model_identity_verified") is not True:
                raise UnsupportedModel("served identity probes did not pass")
        report.update({"supported": True, "serving_verified": True})
        report["serving_evidence"] = identity["identity_probe_evidence"]
    else:
        report["reasons"] = ["serving qualification not run"]
    report["probe_digest"] = _document_digest(report)
    output = Path(out_dir)
    (output.parent / (output.name + ".probe.json")).write_bytes(_wire_document(report))
    return report