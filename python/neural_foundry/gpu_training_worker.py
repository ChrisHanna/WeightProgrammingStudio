"""Trusted local worker for LoRA/SFT checkpoint training on a verified GPU runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .baselines import _conversion_drift
from .cumulative_edit import train_restricted
from .edit_plan import tensor_state_digest
from .gpu import select_training_compute
from .provenance import inference_identity_digest, resolve_checkpoint
from .pythia import PythiaBPETokenizer, PythiaModel, Qwen2CausalLM


def _load_model(payload: dict[str, Any]) -> PythiaModel | Qwen2CausalLM:
    model = _load_unverified_model(payload)
    if tensor_state_digest(model.tensors) != payload.get("input_state_digest"):
        raise ValueError("GPU input tensors differ from the caller's exact state")
    if inference_identity_digest(model) != payload.get("inference_identity_digest"):
        raise ValueError("GPU config or tokenizer differs from the caller's inference identity")
    return model


def _load_unverified_model(payload: dict[str, Any]) -> PythiaModel | Qwen2CausalLM:
    checkpoint = payload.get("checkpoint_identity")
    if checkpoint is not None:
        if not isinstance(checkpoint, dict) or set(checkpoint) != {
            "path", "model_id", "revision", "source", "file_digests"
        }:
            raise ValueError("Pinned checkpoint identity is malformed")
        resolved = resolve_checkpoint(
            str(checkpoint["path"]), revision=checkpoint["revision"], allow_download=False
        )
        expected_digests = checkpoint["file_digests"]
        if not isinstance(expected_digests, dict) or resolved.file_digests != expected_digests:
            raise ValueError("Pinned checkpoint files changed before GPU training")
        if resolved.path != Path(checkpoint["path"]).resolve():
            raise ValueError("Pinned checkpoint path resolved to a different directory")
        config = json.loads((resolved.path / "config.json").read_bytes())
        if config.get("model_type") == "gpt_neox" and payload["model_type"] == "PythiaModel":
            model = PythiaModel.from_pretrained_dir(resolved.path, model_name=str(checkpoint["model_id"]))
        elif config.get("model_type") == "qwen2" and payload["model_type"] == "Qwen2CausalLM":
            model = Qwen2CausalLM.from_pretrained_dir(resolved.path, model_name=str(checkpoint["model_id"]))
        else:
            raise ValueError("Pinned checkpoint architecture changed after training qualification")
        confirmed = resolve_checkpoint(str(checkpoint["path"]), revision=checkpoint["revision"], allow_download=False)
        if confirmed.file_digests != expected_digests:
            raise ValueError("Pinned checkpoint files changed while loading the GPU model")
        model.checkpoint_identity = resolved
        return model
    tokenizer_payload = payload["tokenizer"]
    tokenizer = PythiaBPETokenizer(
        vocab=tokenizer_payload["vocab"],
        merges=tokenizer_payload["merges"],
        byte_level=bool(tokenizer_payload["byte_level"]),
        added_tokens=tokenizer_payload["added_tokens"],
    )
    model_type = str(payload["model_type"])
    model_name = str(payload["model_name"])
    config = dict(payload["config"])
    tensors = dict(payload["tensors"])
    if model_type == "PythiaModel":
        return PythiaModel(
            model_name=model_name, config=config, tokenizer=tokenizer, tensors=tensors
        )
    if model_type == "Qwen2CausalLM":
        return Qwen2CausalLM(
            model_name=model_name, config=config, tokenizer=tokenizer, tensors=tensors
        )
    raise ValueError(f"Unsupported model type {model_type!r}")


def _device_name(device: str) -> str:
    return torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda:") else "cpu"


def run_worker(input_path: Path, output_path: Path) -> dict[str, Any]:
    payload = torch.load(input_path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1 or payload.get("kind") != "restricted_training_jobs":
        raise ValueError("Unsupported GPU training worker payload")
    model = _load_model(payload)
    settings = payload["runtime"]
    placement = select_training_compute(settings["device"])
    precision = settings["precision"]
    if precision not in {"float32", "float64", "bfloat16"}:
        raise ValueError("unsupported training precision")
    converted_base = {name: value.detach().to(dtype=getattr(torch, precision)) for name, value in model.tensors.items()}
    conversion_drift = _conversion_drift(model.tensors, converted_base)
    training_base_digest = tensor_state_digest(converted_base)
    checkpoints: list[dict[str, Any]] = []
    for job in payload["jobs"]:
        if job.get("recipe_schema_version") != 1 or not all(
            isinstance(job.get(field), str) and job[field]
            for field in ("gym_id", "gym_digest", "dataset_digest")
        ):
            raise ValueError("Training job is missing its versioned Gym/data identity")
        recipe = dict(job["recipe"])

        def record(
            step: int,
            weights: dict[str, torch.Tensor],
            wall: float,
            *,
            selected_job: dict[str, Any] = job,
        ) -> None:
            checkpoints.append(
                {
                    "job_id": str(selected_job["job_id"]),
                    "gym_id": str(selected_job["gym_id"]),
                    "gym_digest": str(selected_job["gym_digest"]),
                    "dataset_digest": str(selected_job["dataset_digest"]),
                    "kind": str(selected_job["kind"]),
                    "recipe_index": int(selected_job["recipe_index"]),
                    "step": step,
                    "weights": weights,
                    "wall_seconds": wall,
                }
            )

        train_restricted(
            model,
            tuple(job["sites"]),
            recipe,
            tuple((str(prompt), str(target)) for prompt, target in job["dev"]),
            tuple(str(prompt) for prompt in job["guard"]),
            record,
            int(job.get("seed", 0)),
            training_device=placement.device,
            training_dtype=getattr(torch, precision),
        )
    result = {
        "schema_version": 1,
        "kind": "restricted_training_checkpoints",
        "input_state_digest": tensor_state_digest(model.tensors),
        "inference_identity_digest": inference_identity_digest(model),
        "runtime": {
            "torch": str(torch.__version__),
            "device": placement.device,
            "dtype": precision,
            "device_name": _device_name(placement.device),
            "precision": precision,
            "training_base_state_digest": training_base_digest,
            "conversion_drift_l2_norm": conversion_drift,
        },
        "checkpoints": checkpoints,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    result = run_worker(args.input_path, args.output_path)
    print(
        json.dumps(
            {
                "kind": result["kind"],
                "runtime": result["runtime"],
                "checkpoints": len(result["checkpoints"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
