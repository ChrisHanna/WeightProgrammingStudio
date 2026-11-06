"""Check an exported edit without importing neural_foundry.

Recomputes the whole-model digest from the safetensors file, then runs the export and the base through
`transformers` so a caller can compare logits. Prints one JSON document.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def state_digest(path: Path) -> str:
    tensors = load_file(str(path))
    hasher = hashlib.sha256()
    for name in sorted(tensors):
        value = tensors[name]
        hasher.update(name.encode("utf-8") + b"\0" + ",".join(str(d) for d in value.shape).encode("ascii") + b"\0")
        hasher.update(value.astype("<f8").tobytes())
    return f"sha256:{hasher.hexdigest()}"


def last_logits(model_dir: str, prompts: list[str]) -> tuple[list[list[float]], list[list[int]]]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float64).eval()
    logits, token_ids = [], []
    with torch.no_grad():
        for prompt in prompts:
            ids = tokenizer(prompt, return_tensors="pt").input_ids
            logits.append(model(ids).logits[0, -1].tolist())
            token_ids.append(ids[0].tolist())
    return logits, token_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    prompts = json.loads(Path(args.prompts_file).read_text(encoding="utf-8"))
    exported, token_ids = last_logits(args.export_dir, prompts)
    base, _ = last_logits(args.base_dir, prompts)
    np.savez(args.output, exported=np.array(exported), base=np.array(base))
    print(json.dumps({"state_digest": state_digest(Path(args.export_dir) / "model.safetensors"), "token_ids": token_ids}))


if __name__ == "__main__":
    main()
