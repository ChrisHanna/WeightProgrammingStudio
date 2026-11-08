"""PEFT training and artifact reload implementation."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import torch

from ... import gym_adapters, model_digest
from ...gym_adapters import _resolve_training_checkpoint, _state_digest
from ...gym_artifacts import CandidateRef
from ...gym_job import GymJobError
from ...gym_lifecycle import AdapterIdentity, TrainerAdapter
from ...gym_service import _document_digest, _wire_document
from ...provenance import sha256_file
from . import config as config_module
from .config import PeftBackendConfig


def peft_sequence_trainer(*, checkpoint_hook: Callable[[int], None] | None = None) -> TrainerAdapter:
    def load(request, base, directory):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        checkpoint = _resolve_training_checkpoint(request, base)
        settings = dict(request["inference_config"])
        model = AutoModelForCausalLM.from_pretrained(checkpoint.path, local_files_only=True, dtype=getattr(torch, settings["precision"]), attn_implementation="sdpa").to(settings["device"])
        model = PeftModel.from_pretrained(model, directory, local_files_only=True).eval()
        state = _state_digest(model)
        receipt_path = directory / "training_receipt.json"
        try:
            receipt = json.loads(receipt_path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GymJobError(f"training artifact receipt is unreadable: {exc}") from exc
        recipe = request["recipe"]
        expected_steps = list(range(1, recipe["steps"] + 1))
        if (
            receipt.get("kind") != "nf_peft_training_receipt_v2"
            or receipt.get("training_request_digest") != request["training_request_digest"]
            or receipt.get("dataset_digest") != request["dataset_digest"]
            or receipt.get("recipe_digest") != _document_digest(recipe)
            or receipt.get("committed_steps") != expected_steps
            or receipt.get("completed_steps") != recipe["steps"]
        ):
            raise GymJobError("training receipt does not match the frozen recipe, dataset, or exact step roster")
        files = {path.name: sha256_file(path) for path in sorted(directory.iterdir()) if path.is_file() and path.name != "artifact_manifest.json"}
        artifact = _document_digest({"training_request": request["training_request_digest"], "files": files})
        return CandidateRef(base.model_id, "lora", base.base_digest, state, artifact, str(directory),
                    f"trained:{request['training_request_digest'][7:23]}", "lora", base.family)

    def train(request, examples, base, directory):
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        recipe = request["recipe"]
        config = PeftBackendConfig.from_recipe(recipe)
        if recipe["max_sequence_tokens"] > recipe["tokens_per_batch"] or recipe["max_sequence_tokens"] > request["inference_config"]["context_tokens"]:
            raise GymJobError("sequence budget exceeds batch or base context")
        checkpoint = _resolve_training_checkpoint(request, base)
        settings = dict(request["inference_config"])
        tokenizer = AutoTokenizer.from_pretrained(checkpoint.path, local_files_only=True)
        encoded = []
        dataset_tokens = 0
        for example in examples:
            messages = example["interaction"]["messages"]
            if len(messages) < 2 or messages[-1]["role"] != "assistant":
                raise GymJobError("sequence trainer requires complete assistant demonstrations")
            prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True, enable_thinking=False)
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, enable_thinking=False)
            if len(ids) > recipe["max_sequence_tokens"] or ids[:len(prefix)] != prefix or len(ids) <= len(prefix):
                raise GymJobError("projection violates sequence budget or assistant boundary")
            dataset_tokens += len(ids)
            if dataset_tokens > recipe["max_dataset_tokens"]:
                raise GymJobError("training dataset exceeds its frozen total-token budget")
            encoded.append((ids, len(prefix)))
        if not encoded:
            raise GymJobError("frozen training dataset is empty")
        if settings["device"].split(":", 1)[0] != "cuda" or not torch.cuda.is_available():
            raise GymJobError("PEFT sequence trainer requires an available CUDA device")
        targets = config.target_modules
        torch.manual_seed(recipe["seed"])
        model = AutoModelForCausalLM.from_pretrained(checkpoint.path, local_files_only=True, dtype=getattr(torch, settings["precision"]), attn_implementation="sdpa").to(settings["device"])
        model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=recipe["rank"], lora_alpha=recipe["rank"] * 2,
                               target_modules=list(targets), lora_dropout=0.0)).train()
        optimizer = torch.optim.AdamW((value for value in model.parameters() if value.requires_grad), lr=recipe["learning_rate"])
        torch.cuda.reset_peak_memory_stats(settings["device"])
        losses = []
        committed_steps = []
        resume_path = directory.parent / f".{directory.name}.training-state.pt"
        resume_binding = _document_digest({
            "kind": "nf_peft_resume_binding_v1", "training_request_digest": request["training_request_digest"],
            "dataset_digest": request["dataset_digest"], "loaded_base": request["loaded_base"],
            "recipe": recipe, "examples_digest": request["projected_examples_digest"],
        })
        start_step = 0
        if resume_path.is_file():
            state = torch.load(resume_path, map_location="cpu", weights_only=True)
            if state.get("kind") != "nf_peft_training_checkpoint_v1" or state.get("resume_binding") != resume_binding:
                raise GymJobError("training resume checkpoint does not match the frozen request")
            from peft import set_peft_model_state_dict

            set_peft_model_state_dict(model, state["adapter_state"])
            optimizer.load_state_dict(state["optimizer_state"])
            for optimizer_state in optimizer.state.values():
                for key, value in optimizer_state.items():
                    if isinstance(value, torch.Tensor):
                        optimizer_state[key] = value.to(settings["device"])
            torch.set_rng_state(state["cpu_rng_state"])
            torch.cuda.set_rng_state(state["cuda_rng_state"], settings["device"])
            start_step = state["completed_steps"]
            committed_steps = list(state["committed_steps"])
            losses = list(state["losses"])
            if (type(start_step) is not int or not 0 <= start_step <= recipe["steps"]
                or state.get("data_position") != start_step
                or committed_steps != list(range(1, start_step + 1))):
                raise GymJobError("training resume checkpoint has a malformed step roster")
        from peft import get_peft_model_state_dict

        for step in range(start_step, recipe["steps"]):
            optimizer.zero_grad(set_to_none=True)
            total_tokens = 0
            selected = []
            for offset in range(len(encoded)):
                item = encoded[(step + offset) % len(encoded)]
                if total_tokens + len(item[0]) > recipe["tokens_per_batch"]:
                    break
                selected.append(item)
                total_tokens += len(item[0])
            for ids, boundary in selected:
                inputs = torch.tensor([ids], device=settings["device"])
                labels = inputs.clone()
                labels[:, :boundary] = -100
                loss = model(input_ids=inputs, labels=labels).loss / len(selected)
                loss.backward()
                losses.append(float(loss.detach()))
            optimizer.step()
            completed = step + 1
            committed_steps.append(completed)
            state = {
                "kind": "nf_peft_training_checkpoint_v1", "resume_binding": resume_binding,
                "completed_steps": completed, "data_position": completed, "committed_steps": committed_steps,
                "adapter_state": {name: tensor.detach().cpu().clone() for name, tensor in get_peft_model_state_dict(model).items()},
                "optimizer_state": optimizer.state_dict(), "cpu_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(settings["device"]), "losses": losses,
            }
            descriptor, temporary_name = tempfile.mkstemp(prefix=".nf-training-checkpoint-", dir=resume_path.parent)
            try:
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "wb") as stream:
                    torch.save(state, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_name, resume_path)
            finally:
                Path(temporary_name).unlink(missing_ok=True)
            if checkpoint_hook is not None:
                checkpoint_hook(completed)
        directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        model.save_pretrained(directory, safe_serialization=True)
        tokenizer.save_pretrained(directory)
        receipt = {"kind": "nf_peft_training_receipt_v2", "training_request_digest": request["training_request_digest"],
                   "dataset_digest": request["dataset_digest"], "recipe": recipe, "recipe_digest": config.digest,
                   "completed_steps": recipe["steps"], "committed_steps": committed_steps, "resumed_from_step": start_step,
                   "base_checkpoint": request["loaded_base"], "attention": "sdpa",
                   "peak_vram_bytes": torch.cuda.max_memory_allocated(settings["device"]), "losses": losses}
        receipt_tmp = directory / ".training_receipt.tmp"
        receipt_tmp.write_bytes(_wire_document(receipt))
        os.replace(receipt_tmp, directory / "training_receipt.json")
        resume_path.unlink(missing_ok=True)
        del model, optimizer
        torch.cuda.empty_cache()
        return load(request, base, directory)

    implementation_digest = _document_digest({"worker": sha256_file(__file__),
                                               "config": sha256_file(config_module.__file__),
                                               "model_helpers": sha256_file(gym_adapters.__file__),
                                               "state_digest": sha256_file(model_digest.__file__)})
    return TrainerAdapter(AdapterIdentity("peft-sequence-lora", "1", implementation_digest), ("chat_messages_v1",), ("qwen3", "llama"), ("lora",), train, load, {"lora": "lora"})