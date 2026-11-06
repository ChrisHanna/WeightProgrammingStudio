"""Evaluators the custodian can run, and the candidate artifact format.

A plan selects an evaluator by name from a fixed list; a plan can't name code to import.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from .generators import Item
from .plan import StudyPlan, digest_bytes

_DOWN_PROJ = re.compile(r"^gpt_neox\.layers\.(\d+)\.mlp\.dense_4h_to_h\.weight$")

ItemScores = dict[str, dict[str, float]]


class Evaluator(Protocol):
    def score(self, items: Sequence[Item], artifact: bytes | None) -> ItemScores:
        """Return {item_id: {"forced_choice_accuracy": 0|1, "top1_accuracy": 0|1}}. None means the base model."""

    def retention(self, items: Sequence[Item], artifact: bytes) -> ItemScores:
        """Return {item_id: {"agreement": 0|1, "kl": nats}} comparing the candidate with base."""


def encode_candidate_artifact(tensors: Mapping[str, Any]) -> bytes:
    """Encode edited float64 tensors using the safetensors layout."""
    header: dict[str, Any] = {}
    chunks: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        tensor = tensors[name].detach().to("cpu").double().contiguous()
        raw = struct.pack(f"<{tensor.numel()}d", *tensor.reshape(-1).tolist())
        header[name] = {"dtype": "F64", "shape": list(tensor.shape), "data_offsets": [offset, offset + len(raw)]}
        chunks.append(raw)
        offset += len(raw)
    encoded = json.dumps(header, sort_keys=True).encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks)


def decode_candidate_artifact(data: bytes) -> dict[int, Any]:
    """Decode an artifact to {layer: down-projection}. Any other tensor is refused."""
    import torch

    size = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8 : 8 + size])
    body = data[8 + size :]
    override: dict[int, Any] = {}
    for name, entry in header.items():
        match = _DOWN_PROJ.match(name)
        if match is None or entry["dtype"] != "F64":
            raise ValueError(f"unsupported candidate tensor {name!r}")
        begin, end = entry["data_offsets"]
        values = torch.frombuffer(bytearray(body[begin:end]), dtype=torch.float64)
        override[int(match.group(1))] = values.reshape(entry["shape"])
    if not override:
        raise ValueError("candidate artifact has no tensors")
    return override


class PythiaEvaluator:
    """Forced choice and top-1 at the last token, using the repo's exact GPT-NeoX forward pass."""

    def __init__(self, plan: StudyPlan) -> None:
        import torch

        from ..pythia import PythiaModel

        torch.set_num_threads(plan.torch_threads)
        root = Path(str(plan.checkpoint_dir))
        actual = digest_bytes((root / "model.safetensors").read_bytes())
        if actual != plan.checkpoint_digest:
            raise ValueError(f"checkpoint digest {actual} doesn't match the plan")
        self._model = PythiaModel.from_pretrained_dir(root)
        self._base_log_probs: dict[str, Any] = {}

    def _last_logits(self, prompt: str, override: dict[int, Any] | None) -> Any:
        ids = self._model.tokenizer.encode(prompt)
        return self._model.forward_with_mlp_key(ids, layer_down_proj_override=override)[0][-1]

    def score(self, items: Sequence[Item], artifact: bytes | None) -> ItemScores:
        override = decode_candidate_artifact(artifact) if artifact is not None else None
        vocab = self._model.tokenizer.vocab
        scores: ItemScores = {}
        for item in items:
            logits = self._last_logits(item.prompt, override)
            target = vocab[item.target]
            scores[item.item_id] = {
                "forced_choice_accuracy": float(logits[target] > logits[vocab[item.alternative]]),
                "top1_accuracy": float(int(logits.argmax()) == target),
            }
        return scores

    def retention(self, items: Sequence[Item], artifact: bytes) -> ItemScores:
        import torch

        override = decode_candidate_artifact(artifact)
        scores: ItemScores = {}
        for item in items:
            base = self._base_log_probs.get(item.item_id)
            if base is None:
                base = torch.log_softmax(self._last_logits(item.prompt, None), dim=-1)
                self._base_log_probs[item.item_id] = base
            edited = torch.log_softmax(self._last_logits(item.prompt, override), dim=-1)
            scores[item.item_id] = {
                "agreement": float(int(base.argmax()) == int(edited.argmax())),
                "kl": float((base.exp() * (base - edited)).sum()),
            }
        return scores


class SyntheticEvaluator:
    """For harness tests only. An artifact is JSON naming per-task success probabilities."""

    BASE_P = 0.5

    @staticmethod
    def _unit(*parts: str) -> float:
        digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") / 2**64

    def score(self, items: Sequence[Item], artifact: bytes | None) -> ItemScores:
        doc: dict[str, Any] = json.loads(artifact) if artifact is not None else {"p": {}, "salt": "base"}
        scores: ItemScores = {}
        for item in items:
            # Correctness is shared within a cluster, as the real correlation would be.
            p = float(doc["p"].get(item.task, self.BASE_P))
            hit = float(self._unit(doc["salt"], item.cluster_id) < p)
            scores[item.item_id] = {"forced_choice_accuracy": hit, "top1_accuracy": hit}
        return scores

    def retention(self, items: Sequence[Item], artifact: bytes) -> ItemScores:
        doc = json.loads(artifact)
        return {
            item.item_id: {
                "agreement": float(self._unit(doc["salt"], "agree", item.item_id) < float(doc["agreement"])),
                "kl": float(doc["kl"]),
            }
            for item in items
        }


def build_evaluator(plan: StudyPlan) -> Evaluator:
    if plan.evaluator == "pythia_mlp_override":
        return PythiaEvaluator(plan)
    if plan.evaluator == "synthetic_bernoulli":
        return SyntheticEvaluator()
    raise ValueError(f"unknown evaluator {plan.evaluator!r}")
