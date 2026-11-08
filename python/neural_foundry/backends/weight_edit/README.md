# Generic Weight Editor

## Rules

1. Editor code is not model-family-specific. Family knowledge belongs only in optional override data, which checks discovery rather than replacing it.
2. The editor never implements a forward pass. It uses the selected real model runtime; the standard path uses Transformers.
3. An edit is tensor changes keyed by tensor name, not executable model code.
4. Standard-path edited artifacts must reload in plain Transformers and serve in vLLM without CHAXA code. Other runtime paths must declare and verify their own portability contract rather than imply this one passed.
5. Support is established only by the compatibility probe, including serving checks. Loading successfully or appearing in a matrix is not proof of support.

## Runtime Boundaries

Transformers is a local runtime and checkpoint format here, not a requirement to contact the Hugging Face website. The first implementation accepts local safetensors checkpoints and already-cached snapshots, never downloads, and always uses `trust_remote_code=False`.

Reviewed vendor code is a separate future exception, not a flag on this loader. It requires a recorded review, pinned code digest, qualified network-disabled isolation, and a receipt identifying the code and runtime. No such exception is currently implemented.

NeMo/Megatron and Megatron Bridge are later runtime/conversion paths. GGUF and vendor checkpoint import/export are later format paths. Neither is silently routed through this Transformers loader. Quantized packed weights, mixture-of-experts editing, and hybrid Mamba editing need their own tested handling.

## Coverage Order

1. Standard Transformers decoder checkpoints: Pythia, Qwen, SmolLM2, Phi, Gemma, Llama, OLMo, and Granite.
2. NeMo/Megatron checkpoints and tested Bridge conversions.
3. Reviewed vendor implementations and expert-level edits, starting with feasible small checkpoints. Trillion-parameter Kimi checkpoints are not a single-5090 target.
4. GGUF and vendor checkpoint importers/exporters.

These are test targets, not a supported-model allowlist. Config and checkpoint revision determine runtime requirements; model brand alone does not.

## Implementation Status

The initial loader uses the real Transformers model, extracts its text configuration and hidden size, and records the shared tensor-state digest. Structural discovery finds a uniquely largest homogeneous block stack and records standard Linear and Conv1D sites, including transposed storage. Optional profiles check tensor names and compatible shapes; they do not assign ambiguous roles. Square sites remain `residual_write_unknown` until activation-provenance evidence can distinguish reads and attention writes. Non-square fused GQA and fused gate/up semantics are not yet resolved from structure.

The initial tests use tiny local instances of seven standard architectures, not downloaded production checkpoints. Capture, tensor-only edit methods, application/export, equivalence checks, the complete compatibility probe, lifecycle registration, and the evidence-generated catalog are subsequent work. Distributed loading is not yet implemented.