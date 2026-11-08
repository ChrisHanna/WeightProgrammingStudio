"""Local, pinned NeMo AutoModel SFT, LoRA, and QLoRA trainer adapter."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ... import execution_runner, gym_adapters, model_digest
from ...execution_runner import run_local_worker
from ...gym_artifacts import CandidateRef, artifact_files
from ...gym_job import GymJobError, is_finite_number
from ...gym_lifecycle import AdapterIdentity, TrainerAdapter
from ...gym_service import _document_digest, _wire_document
from ...provenance import sha256_file
from . import config as config_module
from .config import (
    AUTOMODEL_PYTHON,
    AUTOMODEL_VERSIONS,
    PEFT_RELOAD_PYTHON,
    PEFT_RELOAD_VERSIONS,
    QLORA_BITSANDBYTES_VERSION,
    AutoModelBackendConfig,
    AutoModelSFTRecipe,
)


def _sft_rows(examples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for example in examples:
        interaction = example.get("interaction")
        if not isinstance(interaction, Mapping) or interaction.get("format") not in {"chat_messages_v1", "json_extraction_v1"}:
            raise GymJobError("AutoModel SFT requires chat or JSON message demonstrations")
        messages = interaction.get("messages")
        if (not isinstance(messages, list) or len(messages) < 2 or not isinstance(messages[-1], Mapping)
            or messages[-1].get("role") != "assistant"):
            raise GymJobError("AutoModel SFT requires completed assistant demonstrations")
        rows.append({"messages": messages})
    if not rows:
        raise GymJobError("AutoModel SFT requires a nonempty training dataset")
    return rows


def automodel_sft_trainer(config: AutoModelBackendConfig) -> TrainerAdapter:
    python = Path(config.python_executable)
    executable = python.parent / "automodel"
    probe = Path(__file__).parent / "worker" / "probe.py"
    if not python.is_file() or not executable.is_file():
        raise GymJobError("AutoModel Python interpreter or CLI is missing")
    if os.name == "nt":
        raise GymJobError("the local AutoModel backend requires a Linux runtime")
    environment = {
        "PATH": str(python.parent) + os.pathsep + os.defpath,
        "HOME": os.environ.get("HOME", str(Path.home())), "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1",
    }
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        environment["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    implementation = _document_digest({
        "adapter": sha256_file(__file__), "config": sha256_file(config_module.__file__),
        "probe": sha256_file(probe), "model_helpers": sha256_file(gym_adapters.__file__),
        "state_digest": sha256_file(model_digest.__file__), "runner": sha256_file(execution_runner.__file__),
        "runtime_config": config.document(),
    })

    def inspect(mode: str, *arguments: str, cwd: Path, interpreter: Path | None = None) -> dict[str, Any]:
        selected = interpreter or python
        worker_environment = {**environment, "PATH": str(selected.parent) + os.pathsep + os.defpath}
        result = run_local_worker([str(selected), str(probe), mode, *arguments], cwd=cwd,
                                  timeout_seconds=min(config.timeout_seconds, 180.0), env=worker_environment,
                                  cleanup_descendants=True)
        if result.returncode != 0 or result.timed_out:
            raise GymJobError(f"AutoModel {mode} probe failed; stderr digest {result.receipt['stderr_digest']}")
        lines = [line for line in result.stdout.splitlines() if line.startswith(b"NF_AUTOMODEL_INSPECTION=")]
        if len(lines) != 1:
            raise GymJobError(f"AutoModel {mode} probe returned no unique inspection record")
        try:
            document = json.loads(lines[0].split(b"=", 1)[1])
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GymJobError(f"AutoModel {mode} probe returned invalid JSON") from exc
        if not isinstance(document, dict):
            raise GymJobError(f"AutoModel {mode} probe returned a non-object")
        return document

    def runtime(cwd: Path) -> dict[str, Any]:
        result = inspect("runtime", cwd=cwd)
        if (result.get("versions") != AUTOMODEL_VERSIONS or result.get("python") != list(AUTOMODEL_PYTHON)
            or not isinstance(result.get("gpu"), str)):
            raise GymJobError("AutoModel runtime version or GPU identity mismatch")
        return result

    def reload_runtime(cwd: Path, *, quantized: bool = False) -> tuple[Path, dict[str, Any]]:
        if config.reload_python_executable is None or not Path(config.reload_python_executable).is_file():
            raise GymJobError("AutoModel LoRA requires a pinned PEFT reload interpreter")
        interpreter = Path(config.reload_python_executable)
        result = inspect("reload-runtime", cwd=cwd, interpreter=interpreter)
        if result.get("versions") != PEFT_RELOAD_VERSIONS or result.get("python") != list(PEFT_RELOAD_PYTHON):
            raise GymJobError("AutoModel PEFT reload runtime version mismatch")
        if quantized:
            expected = {"bitsandbytes": QLORA_BITSANDBYTES_VERSION}
            if (inspect("quant-runtime", cwd=cwd) != expected
                or inspect("quant-runtime", cwd=cwd, interpreter=interpreter) != expected):
                raise GymJobError("AutoModel QLoRA bitsandbytes version mismatch")
            result = {**result, **expected}
        return interpreter, result

    def load(request: Mapping[str, Any], base: CandidateRef, directory: Path) -> CandidateRef:
        runtime(directory.parent)
        checkpoint = gym_adapters._resolve_training_checkpoint(request, base)
        recipe = AutoModelSFTRecipe.from_document(request["recipe"])
        try:
            receipt = json.loads((directory / "training_receipt.json").read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GymJobError("AutoModel training receipt is unreadable") from exc
        if (not isinstance(receipt, dict) or receipt.get("kind") != "nf_automodel_sft_receipt_v1"
            or receipt.get("training_request_digest") != request["training_request_digest"]
            or receipt.get("dataset_digest") != request["dataset_digest"]
            or receipt.get("recipe_digest") != recipe.digest or receipt.get("backend_config_digest") != config.digest
            or receipt.get("committed_steps") != list(range(recipe.steps))
            or receipt.get("implementation_digest") != implementation):
            raise GymJobError("AutoModel receipt differs from the frozen training request")
        try:
            saved_config_digest = sha256_file(directory / "automodel_config.json")
        except OSError as exc:
            raise GymJobError("AutoModel configuration is missing or unreadable") from exc
        if saved_config_digest != receipt.get("worker_config_digest"):
            raise GymJobError("AutoModel configuration differs from its receipt")
        files = artifact_files(directory)
        if recipe.method in {"lora", "qlora"}:
            reload_python, reload_info = reload_runtime(directory.parent, quantized=recipe.method == "qlora")
            if receipt.get("reload_runtime") != reload_info:
                raise GymJobError("AutoModel PEFT reload runtime differs from its receipt")
            checked = inspect(recipe.method, str(checkpoint.path), str(directory), request["inference_config"]["precision"],
                              cwd=directory.parent, interpreter=reload_python)
        else:
            checked = inspect("model", str(directory), request["inference_config"]["precision"], cwd=directory.parent)
        if (checked.get("state_digest") != receipt.get("model_digest") or checked.get("family") != base.family
            or artifact_files(directory) != files):
            raise GymJobError("AutoModel reloaded checkpoint differs from its receipt")
        digest = _document_digest({"training_request": request["training_request_digest"], "files": files})
        return CandidateRef(base.model_id, recipe.method if recipe.method in {"lora", "qlora"} else "full", base.base_digest,
                    checked["state_digest"], digest, str(directory),
                    f"trained:{request['training_request_digest'][7:23]}", recipe.method, base.family)

    def train(request: Mapping[str, Any], examples: Sequence[Mapping[str, Any]], base: CandidateRef, directory: Path) -> CandidateRef:
        recipe = AutoModelSFTRecipe.from_document(request["recipe"])
        settings = request["inference_config"]
        if (settings.get("device") not in {"cuda", "cuda:0"}
            or settings.get("precision") not in {"float32", "float16", "bfloat16"}
            or recipe.max_sequence_tokens > settings.get("context_tokens", 0)):
            raise GymJobError("AutoModel requires a supported local CUDA device, precision, and context budget")
        if recipe.method == "qlora" and settings.get("precision") != "bfloat16":
            raise GymJobError("AutoModel QLoRA requires bfloat16 compute")
        if _document_digest({"examples": list(examples)}) != request["projected_examples_digest"]:
            raise GymJobError("AutoModel projected examples differ from the frozen request")
        checkpoint = gym_adapters._resolve_training_checkpoint(request, base)
        runtime_info = runtime(directory.parent)
        reload_python, reload_info = (reload_runtime(directory.parent, quantized=recipe.method == "qlora")
                          if recipe.method in {"lora", "qlora"} else (None, None))
        rows = _sft_rows(examples)
        with tempfile.TemporaryDirectory(prefix=".nf-automodel-", dir=directory.parent) as temp:
            work = Path(temp)
            dataset = work / "train.jsonl"
            dataset.write_bytes(b"".join(_wire_document(row) + b"\n" for row in rows))
            dataset_digest = sha256_file(dataset)
            token_report = inspect("dataset", str(checkpoint.path), str(dataset), cwd=work)
            lengths = token_report.get("lengths")
            if (token_report.get("rows") != len(rows) or not isinstance(lengths, list) or len(lengths) != len(rows)
                or any(type(length) is not int or length <= 0 or length > recipe.max_sequence_tokens for length in lengths)
                or token_report.get("total_tokens") != sum(lengths) or sum(lengths) > recipe.max_dataset_tokens):
                raise GymJobError("AutoModel dataset exceeds the frozen token budgets")
            checkpoint_dir = work / "checkpoints"
            dtype = {"float32": "float32", "float16": "fp16", "bfloat16": "bf16"}[settings["precision"]]
            train_config = {
                "recipe": "TrainFinetuneRecipeForNextTokenPrediction", "seed": recipe.seed,
                "step_scheduler": {"global_batch_size": 1, "local_batch_size": 1, "max_steps": recipe.steps,
                                   "ckpt_every_steps": recipe.steps, "val_every_steps": recipe.steps + 1},
                "dist_env": {"backend": "nccl", "timeout_minutes": 1},
                "model": {"_target_": "nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained",
                          "pretrained_model_name_or_path": str(checkpoint.path), "torch_dtype": dtype},
                "checkpoint": {"enabled": True, "checkpoint_dir": str(checkpoint_dir),
                               "model_save_format": "safetensors", "save_consolidated": "final"},
                "distributed": {"strategy": "fsdp2", "dp_size": "none", "tp_size": 1, "cp_size": 1,
                                "sequence_parallel": False},
                "loss_fn": {"_target_": "nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy"},
                "dataset": {"_target_": "nemo_automodel.components.datasets.llm.chat_dataset.ChatDataset",
                            "path_or_dataset_id": str(dataset), "seq_length": recipe.max_sequence_tokens,
                            "mask_history": True, "mask_generation_prompt": True},
                "packed_sequence": {"packed_sequence_size": 0},
                "dataloader": {"_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
                               "collate_fn": "nemo_automodel.components.datasets.utils.default_collater", "shuffle": False},
                "optimizer": {"_target_": "torch.optim.AdamW", "lr": recipe.learning_rate, "weight_decay": 0},
            }
            if recipe.method in {"lora", "qlora"}:
                train_config["peft"] = {"_target_": "nemo_automodel.components._peft.lora.PeftConfig",
                                        "target_modules": [pattern for target in recipe.target_modules
                                                           for pattern in (target, f"*.{target}")], "dim": recipe.rank,
                                        "alpha": recipe.rank * 2, "use_triton": False}
            if recipe.method == "qlora":
                train_config["quantization"] = {"load_in_4bit": True, "load_in_8bit": False,
                                                "bnb_4bit_compute_dtype": "bfloat16", "bnb_4bit_use_double_quant": True,
                                                "bnb_4bit_quant_type": "nf4"}
            config_path = work / "recipe.yaml"
            config_path.write_bytes(_wire_document(train_config))
            config_digest = sha256_file(config_path)
            execution = run_local_worker([str(executable), str(config_path), "--nproc-per-node", "1"], cwd=work,
                                         timeout_seconds=config.timeout_seconds, env=environment, cleanup_descendants=True)
            if execution.timed_out or execution.returncode != 0:
                failure_dir = Path(tempfile.mkdtemp(prefix=".nf-automodel-failure-", dir=directory.parent))
                (failure_dir / "stdout.log").write_bytes(execution.stdout)
                (failure_dir / "stderr.log").write_bytes(execution.stderr)
                raise GymJobError(f"AutoModel training failed; private logs: {failure_dir}")
            if sha256_file(dataset) != dataset_digest or sha256_file(config_path) != config_digest:
                raise GymJobError("AutoModel dataset or configuration changed during training")
            gym_adapters._resolve_training_checkpoint(request, base)
            try:
                training_rows = [json.loads(line) for line in (checkpoint_dir / "training.jsonl").read_bytes().splitlines() if line]
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise GymJobError("AutoModel training log is missing or malformed") from exc
            if ([row.get("step") for row in training_rows] != list(range(recipe.steps))
                or any(not is_finite_number(row.get("loss")) for row in training_rows)):
                raise GymJobError("AutoModel training log does not contain the exact finite step roster")
            if recipe.method in {"lora", "qlora"}:
                outputs = [path for path in checkpoint_dir.glob("epoch_*_step_*/model")
                           if (path / "adapter_model.safetensors").is_file()]
            else:
                outputs = list(checkpoint_dir.glob("epoch_*_step_*/model/consolidated"))
            if len(outputs) != 1 or not artifact_files(outputs[0]):
                raise GymJobError("AutoModel did not produce one verified consolidated checkpoint")
            shutil.copytree(outputs[0], directory)
            if recipe.method in {"lora", "qlora"}:
                model_info = inspect(recipe.method, str(checkpoint.path), str(directory), settings["precision"],
                                     cwd=work, interpreter=reload_python)
            else:
                model_info = inspect("model", str(directory), settings["precision"], cwd=work)
            if model_info.get("family") != base.family or not isinstance(model_info.get("state_digest"), str):
                raise GymJobError("AutoModel checkpoint has an unexpected model family or state")
            shutil.copyfile(config_path, directory / "automodel_config.json")
            receipt = {
                "kind": "nf_automodel_sft_receipt_v1", "training_request_digest": request["training_request_digest"],
                "dataset_digest": request["dataset_digest"], "projected_examples_digest": request["projected_examples_digest"],
                "materialized_dataset_digest": dataset_digest, "recipe_digest": recipe.digest,
                "backend_config_digest": config.digest, "implementation_digest": implementation,
                "worker_config_digest": config_digest, "runtime": runtime_info, "execution": execution.receipt,
                "reload_runtime": reload_info,
                "committed_steps": list(range(recipe.steps)), "losses": [row["loss"] for row in training_rows],
                "model_digest": model_info["state_digest"], "token_count": token_report["total_tokens"],
            }
            (directory / "training_receipt.json").write_bytes(_wire_document(receipt))
        return load(request, base, directory)

    supported = {"sft": "full"}
    if config.reload_python_executable is not None:
        try:
            reload_runtime(Path.cwd())
        except GymJobError:
            pass
        else:
            supported["lora"] = "lora"
            try:
                reload_runtime(Path.cwd(), quantized=True)
            except GymJobError:
                pass
            else:
                supported["qlora"] = "qlora"
    return TrainerAdapter(AdapterIdentity("nemo_automodel.sft", "1", implementation),
                          ("chat_messages_v1", "json_extraction_v1"), ("qwen3", "llama"), tuple(supported), train, load, supported)