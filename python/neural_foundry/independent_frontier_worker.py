"""One-shot isolated process for evaluating a frozen model frontier."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch

from .edit_plan import tensor_state_digest
from .frontier_evaluation import evaluate_frozen_frontier
from .independent_eval import EvalConfig
from .provenance import inference_identity_digest, resolve_checkpoint
from .pythia import PythiaBPETokenizer, PythiaModel, Qwen2CausalLM


def _model(payload: dict[str, Any]) -> PythiaModel | Qwen2CausalLM:
    checkpoint_identity = payload.get("checkpoint_identity")
    if checkpoint_identity is not None:
        checkpoint = resolve_checkpoint(
            checkpoint_identity["path"], revision=checkpoint_identity["revision"], allow_download=False,
        )
        if checkpoint.file_digests != checkpoint_identity["file_digests"]:
            raise ValueError("Pinned evaluator checkpoint files differ from the frozen identity")
        if payload["model_type"] == "PythiaModel":
            model = PythiaModel.from_pretrained_dir(checkpoint.path, model_name=payload["model_name"])
        elif payload["model_type"] == "Qwen2CausalLM":
            model = Qwen2CausalLM.from_pretrained_dir(checkpoint.path, model_name=payload["model_name"])
        else:
            raise ValueError(f"Unsupported evaluator checkpoint model type {payload['model_type']!r}")
        if (
            tensor_state_digest(model.tensors) != payload["expected_state_digest"]
            or inference_identity_digest(model) != payload["expected_inference_identity_digest"]
        ):
            raise ValueError("Pinned evaluator checkpoint state/config/tokenizer differs from the frozen identity")
        return model
    tokenizer_payload = payload["tokenizer"]
    tokenizer = PythiaBPETokenizer(
        vocab=tokenizer_payload["vocab"],
        merges=tokenizer_payload["merges"],
        byte_level=bool(tokenizer_payload["byte_level"]),
        added_tokens=tokenizer_payload["added_tokens"],
    )
    model_name = str(payload["model_name"])
    config = dict(cast(Mapping[str, Any], payload["config"]))
    tensors = dict(cast(Mapping[str, torch.Tensor], payload["tensors"]))
    if payload["model_type"] == "PythiaModel":
        return PythiaModel(
            model_name=model_name,
            config=config,
            tokenizer=tokenizer,
            tensors=tensors,
        )
    if payload["model_type"] == "Qwen2CausalLM":
        return Qwen2CausalLM(
            model_name=model_name,
            config=config,
            tokenizer=tokenizer,
            tensors=tensors,
        )
    raise ValueError(f"Unsupported independent evaluator model {payload['model_type']!r}")


def run_worker(input_path: Path, output_path: Path) -> dict[str, Any]:
    payload = torch.load(input_path, map_location="cpu", weights_only=True)
    if payload.get("kind") != "nf_independent_frontier_job_v1":
        raise ValueError("Unsupported independent frontier worker payload")
    config_payload = payload.get("eval_config")
    config = EvalConfig(**config_payload) if config_payload else None
    report = evaluate_frozen_frontier(
        _model(payload["model"]),
        payload["snapshot"],
        artifact_root=payload["artifact_root"],
        target_cases=payload["target_cases"],
        eval_config=config,
        execution_mode="isolated_subprocess",
        claim_exposure=False,
    )
    output_path.write_text(json.dumps(report, indent=1, allow_nan=False), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    report = run_worker(args.input_path, args.output_path)
    print(
        json.dumps(
            {
                "snapshot_digest": report["snapshot_digest"],
                "evaluated": len(report["evaluated"]),
                "promoted": len(report["promoted_candidate_ids"]),
                "complete": report["complete"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
