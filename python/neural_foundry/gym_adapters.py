"""Optional Transformers/PEFT adapters for the shared research lifecycle."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from importlib import metadata
from pathlib import Path
from typing import Any

import torch

from .gym_artifacts import CandidateRef, artifact_files
from .gym_job import GymJobError, validate_task_instance
from .gym_lifecycle import AdapterIdentity, ModelAdapter, TrainerAdapter
from .gym_service import CandidateIdentity, ExecutionResponse, _document_digest, _wire_document
from .model_digest import model_state_digest as _state_digest
from .provenance import parse_strict_json, resolve_checkpoint, sha256_file
from .prompt_rendering import PromptRenderer


def _parse_generated_json(text: str) -> Any:
    candidate = text.strip()
    if candidate.startswith("```json") and candidate.endswith("```"):
        candidate = candidate[len("```json"): -len("```")].strip()
    try:
        return parse_strict_json(candidate.encode())
    except ValueError:
        return text


def _resolve_training_checkpoint(request: Mapping[str, Any], base: CandidateRef) -> Any:
    identity = request["loaded_base"]
    if identity.get("model_id") != base.model_id or not isinstance(identity.get("revision"), str):
        raise GymJobError("training base identity differs from its reference")
    checkpoint = resolve_checkpoint(base.model_id, revision=identity["revision"])
    digest = _document_digest({"model_id": base.model_id, "revision": identity["revision"],
                               "files": checkpoint.file_digests})
    if digest != base.base_digest or Path(checkpoint.path).resolve() != Path(base.artifact_path).resolve():
        raise GymJobError("training checkpoint files or path differ from the frozen base reference")
    return checkpoint


def transformers_model(
    model_id: str, revision: str, *, device: str, precision: str, context_tokens: int,
    max_new_tokens: int, seed: int, tool_handlers: Mapping[str, Callable[[Mapping[str, Any]], Any]] | None = None,
) -> ModelAdapter:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint = resolve_checkpoint(model_id, revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint.path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint.path, local_files_only=True, dtype=getattr(torch, precision), attn_implementation="sdpa").to(device).eval()
    if resolve_checkpoint(checkpoint.path).file_digests != checkpoint.file_digests:
        raise GymJobError("model assets changed while loading")
    base = _document_digest({"model_id": model_id, "revision": revision, "files": checkpoint.file_digests})
    settings = {"device": device, "precision": precision, "context_tokens": context_tokens, "max_new_tokens": max_new_tokens, "seed": seed}
    candidate = CandidateIdentity(model_id.rsplit("/", 1)[-1], "unchanged_base", base, base, base)
    adapter = _model_adapter(model, tokenizer, checkpoint, candidate, settings, tool_handlers=tool_handlers)
    return adapter


def _model_adapter(
    model: Any, tokenizer: Any, checkpoint: Any, candidate: CandidateIdentity, settings: dict[str, Any],
    *, tool_handlers: Mapping[str, Callable[[Mapping[str, Any]], Any]] | None = None,
    artifact_path: str | Path | None = None,
    loaded_asset_digests: Mapping[str, str] | None = None,
) -> ModelAdapter:
    renderer = PromptRenderer(tokenizer)
    settings = {**settings, "prompt_renderer_digest": renderer.digest}
    state_digest = _state_digest(model)
    assets = loaded_asset_digests or checkpoint.file_digests
    identity = {
        "model_id": checkpoint.model_id, "revision": checkpoint.revision, "checkpoint_digest": candidate.base_model_digest,
        "config_digest": assets["config.json"], "tokenizer_digest": assets["tokenizer.json"],
        "state_digest": state_digest, "model_family": model.config.model_type, "device": settings["device"],
        "precision": settings["precision"], "local_loaded_files_verified": True, "attention": "sdpa",
        "prompt_renderer_digest": renderer.digest,
    }

    def infer(interaction):
        interaction_format = interaction.get("format")
        if interaction_format not in {"chat_messages_v1", "json_extraction_v1", "tool_messages_v1"}:
            raise GymJobError("Transformers adapter requires a declared chat, JSON, or bounded-tool projection")
        if interaction_format == "tool_messages_v1":
            declarations = interaction.get("tools")
            if not isinstance(declarations, list) or not 1 <= len(declarations) <= 8 or interaction.get("max_tool_calls") != 1:
                raise GymJobError("tool interaction must declare 1-8 tools and exactly one allowed call")
            if tool_handlers is None or any(row.get("name") not in tool_handlers for row in declarations):
                raise GymJobError("model adapter has no trusted implementation for every declared tool")
        messages = interaction.get("messages")
        if not isinstance(messages, list):
            raise GymJobError("projected interaction has no message list")
        messages = json.loads(_wire_document(messages))
        if interaction_format == "tool_messages_v1":
            tool_contract = json.dumps(declarations, ensure_ascii=False, sort_keys=True)
            instruction = (
                "Choose exactly one listed tool. Return only a JSON object with exactly these keys: "
                "tool (the exact listed name) and arguments (an object matching its schema). "
                "Do not include markdown, reasoning, or additional keys. Listed tools: " + tool_contract
            )
            messages = [{"role": "system", "content": instruction}, *messages]

        def generate(selected_messages: list[dict[str, Any]]) -> tuple[str, list[int]]:
            rendered = renderer.render({"format": "chat_messages_v1", "messages": selected_messages})
            ids = torch.tensor([rendered.token_ids], dtype=torch.long, device=settings["device"])
            inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            if inputs["input_ids"].shape[1] + settings["max_new_tokens"] > settings["context_tokens"]:
                raise GymJobError("projected interaction exceeds the configured model context")
            with torch.inference_mode():
                tokens = model.generate(**inputs, max_new_tokens=settings["max_new_tokens"], do_sample=False, pad_token_id=tokenizer.eos_token_id)
            generated = tokens[0, inputs["input_ids"].shape[1]:]
            return tokenizer.decode(generated, skip_special_tokens=True), generated.tolist()

        text, token_ids = generate(messages)
        generated_response = _parse_generated_json(text)
        evidence = {"raw_text": text, "generated_token_ids": token_ids, "loaded_identity": identity,
                    "interaction_format": interaction_format, "prompt_renderer_digest": renderer.digest}
        if interaction_format in {"chat_messages_v1", "json_extraction_v1"}:
            return ExecutionResponse(generated_response, evidence)
        if not isinstance(generated_response, Mapping) or set(generated_response) != {"tool", "arguments"}:
            evidence["tool_call_invalid"] = "expected exactly tool and arguments"
            return ExecutionResponse(generated_response, evidence)
        declaration = next((row for row in declarations if row["name"] == generated_response["tool"]), None)
        arguments = generated_response["arguments"]
        if declaration is None or not isinstance(arguments, Mapping):
            evidence["tool_call_invalid"] = "undeclared tool or malformed arguments"
            return ExecutionResponse(generated_response, evidence)
        argument_schema = declaration.get("arguments", {})
        try:
            validate_task_instance(argument_schema, arguments, field="tool.arguments")
        except GymJobError as exc:
            evidence["tool_call_invalid"] = str(exc)
            return ExecutionResponse(generated_response, evidence)
        result = tool_handlers[declaration["name"]](json.loads(_wire_document(arguments)))
        bounded_result = json.loads(_wire_document(result))
        evidence["tool_trace"] = [{"name": declaration["name"], "arguments": json.loads(_wire_document(arguments)), "result": bounded_result}]
        return ExecutionResponse({"calls": evidence["tool_trace"]}, evidence)

    formats = ("chat_messages_v1", "json_extraction_v1", "tool_messages_v1") if tool_handlers else ("chat_messages_v1", "json_extraction_v1")
    kind = candidate.method_id if candidate.method_id in {"lora", "qlora", "weight_edit"} else "full"
    reference = CandidateRef(checkpoint.model_id, kind, candidate.base_model_digest,
                             candidate.candidate_model_digest, candidate.artifact_digest,
                             str(artifact_path or checkpoint.path), candidate.candidate_id,
                             candidate.method_id, model.config.model_type)

    def load_candidate(ref: CandidateRef) -> ModelAdapter:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if ref.kind not in {"lora", "qlora", "full", "weight_edit"} or ref.model_id != checkpoint.model_id or ref.base_digest != candidate.base_model_digest:
            raise GymJobError("Transformers model cannot load the candidate reference")
        if resolve_checkpoint(checkpoint.path).file_digests != checkpoint.file_digests:
            raise GymJobError("model assets changed before candidate loading")
        artifact_directory = Path(ref.artifact_path)
        candidate_files = artifact_files(artifact_directory)
        try:
            receipt = json.loads((artifact_directory / "training_receipt.json").read_bytes())
            request_digest = receipt["training_request_digest"]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise GymJobError("candidate training receipt is unreadable") from exc
        if _document_digest({"training_request": request_digest, "files": candidate_files}) != ref.artifact_digest:
            raise GymJobError("candidate files differ from their artifact identity")
        candidate_assets = dict(checkpoint.file_digests)
        model_path = checkpoint.path
        if ref.kind in {"full", "weight_edit"}:
            model_path = artifact_directory
            candidate_assets = resolve_checkpoint(model_path).file_digests
        else:
            if "tokenizer.json" not in candidate_files:
                raise GymJobError("LoRA candidate tokenizer is missing")
            candidate_assets["tokenizer.json"] = candidate_files["tokenizer.json"]
        load_options = {"local_files_only": True, "dtype": getattr(torch, settings["precision"]), "attn_implementation": "sdpa"}
        if ref.kind == "qlora":
            from transformers import BitsAndBytesConfig

            if settings["precision"] != "bfloat16" or not settings["device"].startswith("cuda"):
                raise GymJobError("QLoRA loading requires CUDA bfloat16 compute")
            load_options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            )
            load_options["device_map"] = {"": settings["device"]}
        loaded = AutoModelForCausalLM.from_pretrained(model_path, **load_options)
        if ref.kind != "qlora":
            loaded = loaded.to(settings["device"])
        if ref.kind in {"lora", "qlora"}:
            from peft import PeftModel

            loaded = PeftModel.from_pretrained(loaded, ref.artifact_path, local_files_only=True)
        loaded = loaded.eval()
        loaded_tokenizer = AutoTokenizer.from_pretrained(ref.artifact_path, local_files_only=True)
        if resolve_checkpoint(checkpoint.path).file_digests != checkpoint.file_digests:
            raise GymJobError("model assets changed while loading the candidate")
        if artifact_files(artifact_directory) != candidate_files:
            raise GymJobError("candidate assets changed while loading")
        if _state_digest(loaded) != ref.model_digest:
            raise GymJobError("loaded candidate state differs from its reference")
        identity = CandidateIdentity(ref.candidate_id, ref.method_id, ref.base_digest, ref.model_digest, ref.artifact_digest)
        return _model_adapter(loaded, loaded_tokenizer, checkpoint, identity, settings,
                              tool_handlers=tool_handlers, artifact_path=ref.artifact_path,
                              loaded_asset_digests=candidate_assets)

    supported_kinds = ("full", "lora", "weight_edit")
    try:
        if metadata.version("bitsandbytes") == "0.49.1" and settings["device"].startswith("cuda"):
            supported_kinds += ("qlora",)
    except metadata.PackageNotFoundError:
        pass
    adapter = ModelAdapter(AdapterIdentity(f"transformers:{candidate.candidate_id}", "1", sha256_file(__file__)), candidate,
                           model.config.model_type, formats, settings, infer, lambda: identity,
                           supported_kinds, kind, reference, load_candidate)
    object.__setattr__(adapter, "runtime_model", model)
    object.__setattr__(adapter, "runtime_tokenizer", tokenizer)
    object.__setattr__(adapter, "checkpoint", checkpoint)
    return adapter


def peft_sequence_trainer(*, checkpoint_hook: Callable[[int], None] | None = None) -> TrainerAdapter:
    from .backends.peft.adapter import peft_sequence_trainer as create_trainer

    return create_trainer(checkpoint_hook=checkpoint_hook)
