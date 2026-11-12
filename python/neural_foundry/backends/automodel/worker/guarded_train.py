"""Bound CUDA allocations before invoking the official in-process AutoModel CLI."""

import atexit
import json
import os
from pathlib import Path


def main() -> None:
    import torch

    maximum_cuda_allocated_bytes = int(os.environ["NF_MAX_CUDA_ALLOCATED_BYTES"])
    memory_receipt_path = os.environ["NF_CUDA_MEMORY_RECEIPT"]

    device = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(device).total_memory
    if (type(maximum_cuda_allocated_bytes) is not int
        or not 0 < maximum_cuda_allocated_bytes <= total):
        raise ValueError("training CUDA allocation ceiling exceeds device capacity")
    torch.cuda.set_per_process_memory_fraction(maximum_cuda_allocated_bytes / total, device)
    torch.cuda.reset_peak_memory_stats(device)

    def report() -> None:
        path = Path(memory_receipt_path)
        document = {
            "kind": "nf_training_cuda_allocation_v1",
            "limit_bytes": maximum_cuda_allocated_bytes,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "device_total_bytes": total,
        }
        with path.open("x", encoding="utf-8") as stream:
            json.dump(document, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())

    atexit.register(report)
    if "RANK" not in os.environ:
        raise RuntimeError("guarded AutoModel entrypoint requires a torchrun worker")
    from nemo_automodel.cli.app import main as automodel_main

    automodel_main()


if __name__ == "__main__":
    main()
