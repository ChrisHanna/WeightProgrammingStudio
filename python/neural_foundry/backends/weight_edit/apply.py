"""Transactional tensor application and plain Transformers export."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ... import ir
from ...gym_artifacts import CandidateRef, artifact_files, read_candidate_manifest, write_candidate_manifest
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
        chunks = []
        replacement = canonical.clone()
        if bool(torch.count_nonzero(change)):
            for column_start in range(0, site.input_size, 4096):
                column_stop = min(column_start + 4096, site.input_size)
                width = column_stop - column_start
                rows_per_chunk = min(524288 // width, (1_000_000 - 8 * width) // (2 * width + 8))
                for start in range(0, site.output_size, rows_per_chunk):
                    stop = min(start + rows_per_chunk, site.output_size)
                    block = canonical[start:stop, column_start:column_stop].contiguous()
                    transaction = ir.WeightIRTransactionBuilder.execute_matrix_edit(block, operation_type="add_delta",
                                  delta_matrix=change[start:stop, column_start:column_stop].contiguous(),
                                  smoke_probe=torch.ones(width, dtype=torch.float64),
                                  provenance_tag="weight_edit.generic.v1", engine_name=method)
                    ir.require_reload_parity(transaction)
                    replacement[start:stop, column_start:column_stop] = torch.frombuffer(bytearray(transaction.edited_bytes), dtype=torch.float64).reshape(block.shape)
                    chunks.append({"row_start": start, "row_stop": stop, "column_start": column_start,
                                   "column_stop": column_stop, "plan_digest": transaction.plan_digest,
                                   "candidate_digest": transaction.candidate_digest, "reload_parity_verified": True})
        replacement = replacement.T if site.transposed else replacement
        replacement = replacement.to(device=parameter.device, dtype=parameter.dtype)
        if not bool(torch.isfinite(replacement).all()):
            raise WeightEditInputError("casting the edited tensor produced nonfinite values")
        backups[name], replacements[name] = parameter.detach().clone(), replacement
        transactions[name] = {"chunks": chunks, "zero_noop_verified": not chunks, "reload_parity_verified": True}
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


def save(edited: EditedModel, out_dir: str | Path, *, model_id: str, request_digest: str,
         identity_probes: Mapping[str, Any] | None = None) -> CandidateRef:
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
        dtype_metadata = {}
        selected_dtype = str(next(edited.loaded.model.parameters()).dtype).removeprefix("torch.")
        for name in ("dtype", "torch_dtype"):
            if name in config_after:
                if config_after[name] != selected_dtype:
                    raise UnsupportedModel("export dtype metadata differs from the selected tensor precision")
                if config_before.get(name) != config_after[name]:
                    dtype_metadata[name] = {"source": config_before.get(name), "export": config_after[name]}
                    config_before[name] = config_after[name]
        for config in (config_before, config_after):
            config.pop("_name_or_path", None)
            config.pop("transformers_version", None)
        if config_before != config_after:
            raise UnsupportedModel("export changed the model configuration")
        if (reloaded.base_digest != edited.receipt["model_digest"] or tensor_digests(reloaded.model) != edited.after_tensors
            or PromptRenderer(reloaded.tokenizer).digest != edited.receipt["renderer_digest"]):
            raise UnsupportedModel("fresh export reload differs in tensor state or prompt rendering")
        receipt = {**edited.receipt, "training_request_digest": request_digest, "fresh_reload_verified": True,
               "explicit_dtype_metadata": dtype_metadata}
        if identity_probes is not None:
            if (identity_probes.get("model_digest") != receipt["model_digest"]
                or identity_probes.get("renderer_digest") != receipt["renderer_digest"]):
                raise UnsupportedModel("creation-time identity probes differ from the exported model")
            receipt["identity_probes"] = {**dict(identity_probes), "weight_files": {
                name: digest for name, digest in artifact_files(output).items() if name.endswith(".safetensors")}}
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


def seal_serving_probes(reference: CandidateRef, identity: Mapping[str, Any], portability: Mapping[str, Any]) -> CandidateRef:
    from dataclasses import replace
    from .serving_probes import verify_identity

    verify_identity(identity, identity)
    root = Path(reference.artifact_path)
    receipt_path, manifest_path = root / "training_receipt.json", root / "artifact_manifest.json"
    original_receipt, original_manifest = receipt_path.read_bytes(), manifest_path.read_bytes()
    receipt = json.loads(original_receipt)
    request_digest = receipt["training_request_digest"]
    read_candidate_manifest(root, request_digest)
    files = artifact_files(root)
    weights = {name: digest for name, digest in files.items() if name.endswith(".safetensors")}
    if (_document_digest({"training_request": request_digest, "files": files}) != reference.artifact_digest
        or identity["model_digest"] != reference.model_digest or identity["weight_files"] != weights
        or identity["renderer_digest"] != receipt["renderer_digest"] or portability.get("verified") is not True):
        raise UnsupportedModel("creation-time serving evidence differs from the exported candidate")
    updated = {**receipt, "identity_probes": dict(identity), "portability_verification": dict(portability)}
    descriptor, temporary_name = tempfile.mkstemp(prefix=".nf-serving-receipt-", dir=root)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_wire_document(updated))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, receipt_path)
        digest = _document_digest({"training_request": request_digest, "files": artifact_files(root)})
        candidate_id = "weight-edit:" + digest[7:23]
        candidate = {"candidate_id": candidate_id, "method_id": reference.method_id, "base_model_digest": reference.base_digest,
                     "candidate_model_digest": reference.model_digest, "artifact_digest": digest}
        write_candidate_manifest(root, model_id=reference.model_id, kind=reference.kind, base_digest=reference.base_digest,
                                 model_digest=reference.model_digest, request_digest=request_digest,
                                 metadata={"candidate": candidate, "weight_edit_receipt": updated})
        return replace(reference, artifact_digest=digest, candidate_id=candidate_id)
    except BaseException:
        receipt_path.write_bytes(original_receipt)
        manifest_path.write_bytes(original_manifest)
        raise
    finally:
        temporary.unlink(missing_ok=True)