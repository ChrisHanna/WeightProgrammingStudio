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

The loader records shared-storage groups (including views) and compares runtime state names with actual safetensors keys, allowing only recorded tied aliases. Application rejects shared storage unless explicitly permitted, applies it once, and records every changed alias. Proposed matrices are verified through the native transaction path before copying replacements into the real model; application rolls back on failure.

Discovery uses configured head and key/value dimensions, projection-group structure, and real-model output ablation to distinguish attention and MLP writers. It verifies that a writer produces the enclosing branch's output and that zeroing the writer zeroes that output. This is a branch-output check, not a general residual-dataflow proof. Unresolved writers remain unknown. GQA fused-QKV widths use query plus twice key/value width, not three times hidden size. Profiles check roles and names without assigning them. Every block parameter must be accounted for as a projection, norm, or bias. Unexplained parameters, including packed 3D experts, are refused; recognized routers are explicitly labelled rather than exposed as residual reads.

Capture uses the shared prompt renderer: chat models receive their tokenizer's chat template, without duplicate special tokens. Raw text is an explicit mode. Token positions refer to the rendered sequence. Rendering and prompt digests accompany capture evidence. Invalid caller inputs raise `WeightEditInputError`, separately from unsupported runtime behavior.

Activation capture runs the real model under scoped hooks and deterministic algorithms, with TF32 disabled. CUDA requires `CUBLAS_WORKSPACE_CONFIG=:4096:8` (or `:16:8`) before initialization. Process flags and hooks are restored on failure. Capture assumes exclusive use of an evaluation-mode model. Retained samples have a byte budget; streaming covariance accumulates in float32 or float64 without keeping every sample. Covariance itself has a byte budget because its storage grows with feature dimension.

Working weights are read from verified original checkpoint tensors into owned float32/float64 matrices, rather than taken from a lower-precision inference copy. Conv1D orientation is normalized. The tensor-only method interface returns dense or low-rank deltas, and the legacy Gram solve is shared with the existing engine.

Capture has an explicit `output_dtype` (float32 by default, float64 for strict reference checks), including gradients and byte-budget accounting. Streaming statistics request their actual computation dtype rather than promoting already-rounded float32 samples.

The reference wrappers have an opt-in `match_transformers_internals` comparison mode. The unchanged default preserves historical tensor-precision arithmetic. Strict float64 is required except for the documented Transformers 4.57.3 internals: GPT-NeoX attention softmax and rotary; Qwen2 attention softmax, rotary, and RMSNorm. All other operations remain float64. Matched float64 per-site relative errors measured 2.95e-16 (GPT-NeoX) and 1.06e-16 (Qwen2), below the 1e-12 gate. Normal float32 comparisons use 8 times float32 epsilon; deltas are checked and greedy predictions after edits must match exactly. The comparison exception affects reference tests, not generic edit computation on the real Transformers model.

Export uses ordinary safe-serialized Transformers weights and tokenizer assets, verifies a fresh reload, checks configuration and prompt rendering, and produces a weight-edit CandidateRef with tensor/state/edit/site/renderer receipts. Explicit dtype metadata normalization is verified and recorded. Only listed tensors or explicitly permitted tied aliases may differ. Large projection edits use native row/column tiles within the existing checkpoint, smoke-work, and smoke-vector limits.

The compatibility probe implements zero-edit exactness, a bounded relative-norm edit, unchanged-tensor checks, fresh reload, and 16 fixed one-token serving probes. Identity and portability are separate: vLLM records creation-time identity at serving precision, bound to weight files, renderer, image, and context; a fresh vLLM run must match within 1e-6 and produce the same greedy token IDs. Transformers-generated records cannot qualify as vLLM identity.

Portability compares the same prompt/token sequences across Transformers and vLLM. Its limit is the measured unedited-base maximum log-probability difference plus a fixed 1e-6 numerical margin, frozen before editing and bound by a calibration digest. Candidate observations cannot enlarge this limit. Identity records are sealed into the candidate receipt before its final artifact digest is published.

The initial cross-runtime design measured Qwen3 passing in float32; that result is not a qualification of the redesigned bf16 path. Redesigned bf16 and the complete three-model matrix remain pending. SmolLM2 serving probes passed separately, but its full run was not qualified because cleanup failed. Init-enabled graceful stop/removal and bounded forced-removal fallback are implemented. No generic supported-model catalog is claimed.

The tests use tiny local instances of seven standard architectures, the original mini GPT-NeoX and Qwen2 engines, and the already-cached Qwen3 checkpoint. Complete compatibility qualification, lifecycle registration, the proof-loop weight-edit arm, and the evidence-generated catalog remain subsequent work. Distributed loading is not yet implemented. Stock NeMo Gym rollout rendering is separately confirmed not to use the shared token-ID path; no positive NeMo rendering claim is made.

## Advanced Methods

The [methods contract](methods/README.md) records the next milestone: locate, calculated edits, and measured preservation, followed by sequential updates, merging, portable directions, and restricted training. It also specifies the frozen gym-to-edit request, bounded preservation panel, and receipts. These remain planned; holdout is final one-shot evaluation, never feedback for iterative stopping or target optimization.