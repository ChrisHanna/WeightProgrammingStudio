"""Transactional tensor application and plain Transformers export."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ... import ir
from ...gym_artifacts import CandidateRef, artifact_files, write_candidate_manifest
from ...gym_service import _document_digest, _wire_document
from ...model_digest import model_state_digest
from ...prompt_rendering import PromptRenderer
from .discover import SiteMap
from .loader import LoadedModel, UnsupportedModel, WeightEditInputError, load_model, require_editable, storage_groups
from .methods.base import Delta


def tensor_digests(model: Any) -> dict[str, str]:
    return {name: "sha256:" + hashlib.sha256(
        f"{tensor.dtype}:{tuple(tensor.shape)}\0".encode() + tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest() for name, tensor in model.state_dict().items()}


@dataclass(frozen=True)
class EditedModel:
    loaded: LoadedModel
    before_tensors: dict[str, str]
    after_tensors: dict[str, str]
    receipt: dict[str, Any]


def apply(loaded: LoadedModel, deltas: Mapping[str, Delta], *, sites: SiteMap,
          method: str, settings: Mapping[str, Any], allow_shared: bool = False) -> EditedModel:
    if not isinstance(deltas, Mapping) or not deltas or not isinstance(method, str) or not method:
        raise WeightEditInputError("apply requires named deltas and a method identity")
    if model_state_digest(loaded.model) != loaded.base_digest:
        raise UnsupportedModel("loaded model changed before edit application")
    require_editable(loaded, list(deltas), allow_shared=allow_shared)
    by_name = {site.tensor_name: site for site in sites.sites}
    if set(deltas) - by_name.keys() or not sites.checkpoint_names_verified:
        raise WeightEditInputError("edited sites require a checkpoint-verified site map")
    groups = storage_groups(loaded.model)
    if any(len(set(group) & deltas.keys()) > 1 for group in groups):
        raise WeightEditInputError("shared storage may be edited only once")
    before = tensor_digests(loaded.model)
    backups, replacements, transactions, delta_records = {}, {}, {}, {}
    for name, delta in deltas.items():
        site = by_name[name]
        if site.role not in {"residual_write_mlp", "residual_write_attn"} or not isinstance(delta, Delta):
            raise WeightEditInputError("delta targets an unsupported edit role")
        parameter = loaded.model.get_parameter(name)
        original = parameter.detach().cpu().to(torch.float64)
        canonical = original.T.contiguous() if site.transposed else original.contiguous()
        change = delta.matrix().to(torch.float64)
        if change.shape != canonical.shape or not bool(torch.isfinite(change).all()):
            raise WeightEditInputError("delta shape differs from its site or contains nonfinite values")
        transaction = ir.WeightIRTransactionBuilder.execute_matrix_edit(canonical, operation_type="add_delta",
                      delta_matrix=change, smoke_probe=torch.ones(site.input_size, dtype=torch.float64),
                      provenance_tag="weight_edit.generic.v1", engine_name=method)
        ir.require_reload_parity(transaction)
        replacement = torch.frombuffer(bytearray(transaction.edited_bytes), dtype=torch.float64).reshape(canonical.shape).clone()
        replacement = replacement.T if site.transposed else replacement
        replacement = replacement.to(device=parameter.device, dtype=parameter.dtype)
        if not bool(torch.isfinite(replacement).all()):
            raise WeightEditInputError("casting the edited tensor produced nonfinite values")
        backups[name], replacements[name] = parameter.detach().clone(), replacement
        transactions[name] = {"plan_digest": transaction.plan_digest, "candidate_digest": transaction.candidate_digest,
                               "reload_parity_verified": True}
        delta_records[name] = {"kind": delta.kind, "shape": list(change.shape), "dtype": str(change.dtype),
                              "digest": "sha256:" + hashlib.sha256(change.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
    try:
        with torch.no_grad():
            for name, replacement in replacements.items():
                loaded.model.get_parameter(name).copy_(replacement)
        after = tensor_digests(loaded.model)
        allowed = set(deltas)
        if allow_shared:
            allowed |= {name for group in groups if set(group) & deltas.keys() for name in group}
        changed = {name for name in before if before[name] != after[name]}
        if changed - allowed:
            raise UnsupportedModel("unlisted tensors changed during edit application")
        norms = {name: float(torch.linalg.vector_norm(replacements[name].detach().cpu().to(torch.float64)
                                 - backups[name].detach().cpu().to(torch.float64)))
                 for name in replacements}
        receipt = {"kind": "nf_weight_edit_receipt_v1", "base_digest": loaded.base_digest,
                   "site_map_digest": sites.digest, "edit_digest": _document_digest({"deltas": delta_records}),
                   "renderer_digest": PromptRenderer(loaded.tokenizer).digest, "method": method,
                   "settings": json.loads(_wire_document(settings)), "allow_shared": allow_shared,
                   "shared_storage_groups": [list(group) for group in groups if set(group) & deltas.keys()],
                   "edited_tensors": sorted(changed), "requested_tensors": sorted(deltas),
                   "change_norms": norms, "transactions": transactions, "model_digest": model_state_digest(loaded.model)}
        return EditedModel(loaded, before, after, receipt)
    except BaseException:
        with torch.no_grad():
            for name, original in backups.items():
                loaded.model.get_parameter(name).copy_(original)
        raise


def save(edited: EditedModel, out_dir: str | Path, *, model_id: str, request_digest: str) -> CandidateRef:
    output = Path(out_dir).absolute()
    if tensor_digests(edited.loaded.model) != edited.after_tensors:
        raise UnsupportedModel("edited model changed before export")
    output.mkdir(parents=True, exist_ok=False)
    try:
        config_before = edited.loaded.config.to_dict()
        edited.loaded.model.save_pretrained(output, safe_serialization=True)
        edited.loaded.tokenizer.save_pretrained(output)
        reloaded = load_model(output, dtype=next(edited.loaded.model.parameters()).dtype, device="cpu")
        config_after = reloaded.config.to_dict()
        for config in (config_before, config_after):
            config.pop("_name_or_path", None)
            config.pop("transformers_version", None)
        if config_before != config_after:
            raise UnsupportedModel("export changed the model configuration")
        if (reloaded.base_digest != edited.receipt["model_digest"] or tensor_digests(reloaded.model) != edited.after_tensors
            or PromptRenderer(reloaded.tokenizer).digest != edited.receipt["renderer_digest"]):
            raise UnsupportedModel("fresh export reload differs in tensor state or prompt rendering")
        receipt = {**edited.receipt, "training_request_digest": request_digest, "fresh_reload_verified": True}
        (output / "training_receipt.json").write_bytes(_wire_document(receipt))
        digest = _document_digest({"training_request": request_digest, "files": artifact_files(output)})
        candidate_id = "weight-edit:" + digest[7:23]
        candidate = {"candidate_id": candidate_id, "method_id": "weight_edit", "base_model_digest": receipt["base_digest"],
                     "candidate_model_digest": receipt["model_digest"], "artifact_digest": digest}
        write_candidate_manifest(output, model_id=model_id, kind="weight_edit", base_digest=receipt["base_digest"],
                                 model_digest=receipt["model_digest"], request_digest=request_digest,
                                 metadata={"candidate": candidate, "weight_edit_receipt": receipt})
        return CandidateRef(model_id, "weight_edit", receipt["base_digest"], receipt["model_digest"], digest,
                            str(output), candidate_id, "weight_edit", edited.loaded.text_config.model_type)
    except BaseException:
        shutil.rmtree(output)
        raise