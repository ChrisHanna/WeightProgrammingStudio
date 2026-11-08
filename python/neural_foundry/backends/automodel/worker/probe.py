"""Inspect an AutoModel runtime, local dataset, or reloaded checkpoint."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path


def main() -> None:
    import torch

    mode = sys.argv[1]
    if mode == "runtime":
        versions = {name: importlib.metadata.version(name) for name in (
            "nemo-automodel", "torch", "triton", "transformers", "torchao", "torchdata", "datasets",
        )}
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        probe = torch.ones((2, 2), device="cuda") @ torch.ones((2, 2), device="cuda")
        torch.cuda.synchronize()
        if not bool(torch.equal(probe.cpu(), torch.full((2, 2), 2.0))):
            raise RuntimeError("CUDA kernel probe failed")
        result = {"versions": versions, "python": list(sys.version_info[:3]), "gpu": torch.cuda.get_device_name(0)}
    elif mode == "reload-runtime":
        versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")}
        result = {"versions": versions, "python": list(sys.version_info[:3])}
    elif mode == "quant-runtime":
        result = {"bitsandbytes": importlib.metadata.version("bitsandbytes")}
    elif mode == "dataset":
        from nemo_automodel.components.datasets.llm.chat_dataset import ChatDataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(sys.argv[2], local_files_only=True, trust_remote_code=False)
        dataset = ChatDataset(sys.argv[3], tokenizer, mask_history=True, mask_generation_prompt=True)
        lengths = []
        for item in dataset:
            if not any(int(label) != -100 for label in item["labels"]):
                raise ValueError("dataset example has no supervised tokens")
            lengths.append(len(item["input_ids"]))
        result = {"rows": len(lengths), "lengths": lengths, "total_tokens": sum(lengths)}
    elif mode in {"model", "lora", "qlora"}:
        from transformers import AutoModelForCausalLM

        helper = Path(__file__).resolve().parents[3] / "model_digest.py"
        spec = importlib.util.spec_from_file_location("nf_model_digest", helper)
        if spec is None or spec.loader is None:
            raise RuntimeError("model digest helper is unavailable")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        precision = sys.argv[4] if mode in {"lora", "qlora"} else sys.argv[3]
        options = {"local_files_only": True, "trust_remote_code": False,
                   "dtype": getattr(torch, precision), "attn_implementation": "sdpa"}
        if mode == "qlora":
            from transformers import BitsAndBytesConfig

            options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            )
            options["device_map"] = {"": 0}
        model = AutoModelForCausalLM.from_pretrained(sys.argv[2], **options)
        if mode in {"lora", "qlora"}:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, sys.argv[3], local_files_only=True)
        model.eval()
        result = {"state_digest": module.model_state_digest(model), "family": model.config.model_type}
    else:
        raise ValueError("unsupported AutoModel inspection mode")
    print("NF_AUTOMODEL_INSPECTION=" + json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()