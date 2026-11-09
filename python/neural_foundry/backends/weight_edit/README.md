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

Bf16 portability can use a base-only near-tie policy when both runtimes report measured top-two probabilities. The allowed pair must be the same in both base runs, the Transformers reference-base gap must be below the measured same-token discrepancy, and the observed gap difference must fit the sum of the two per-token error bounds. At least 12 of 16 probes must remain decisive. Candidate checks compare the same token's log probabilities and cannot introduce new allowed pairs or enlarge the noise limit. Same-runtime identity remains exact-token. A reference gap of zero can remain a valid tie even if the observed gap exceeds the single-token discrepancy: a gap involves two tokens.

These rules are now frozen as near-tie policy v2 in `protocols/near-tie-policy-v2.json`, with the revision reason and the SmolLM2 evidence that prompted it. Subsequent model results must not adjust v2; a future numerical change requires a new version and approval. Historical v1 evidence remains unchanged. Nemotron's recorded failure is classified as a same-token log-probability gap from its failure path. Its raw base/top-two records were not retained, so offline v2 replay is unavailable, not passed or failed; no new GPU run is used to fill that gap.

SmolLM2 bf16 base measurements confirm one near-tie (probe 6) and 15 decisive probes. The reference gap is zero, the vLLM gap is 0.125, and measured per-token discrepancy is 0.10782361030578613. The same pair is present in both runtimes; the gap difference is within twice that measured discrepancy. Raw measurements and a separately digest-bound analysis are retained under `docs/evidence/weight-edit-smollm2-bf16-near-ties.json` and `docs/evidence/weight-edit-smollm2-bf16-near-tie-calibration.json`. This passes base portability calibration, not full edited-candidate bf16 qualification.

Qwen3-0.6B passes the redesigned bf16 compatibility probe, including separate identity and base-calibrated portability; its generated report is retained under `docs/evidence/weight-edit-qwen3-bf16.json`. SmolLM2-1.7B-Instruct passes separately in float32, recorded under `docs/evidence/weight-edit-smollm2-float32.json`; its bf16 unedited-base portability failed because a fixed prompt selected different greedy tokens. Nemotron bf16 failed the frozen candidate portability limit, recorded under `docs/evidence/weight-edit-nemotron-bf16.json`. These are compatibility perturbations, not task-quality evidence or promotion authority. Candidate paths in these test reports are temporary, not a durable product artifact store. Failed model gates are recorded without invented metrics or candidates. Init-enabled graceful stop/removal and bounded forced-removal fallback are implemented. No generic supported-model catalog is claimed.

The tests use tiny local instances of seven standard architectures, the original mini GPT-NeoX and Qwen2 engines, and the already-cached Qwen3 checkpoint. Complete compatibility qualification, lifecycle registration, the proof-loop weight-edit arm, and the evidence-generated catalog remain subsequent work. Distributed loading is not yet implemented. Stock NeMo Gym rollout rendering is separately confirmed not to use the shared token-ID path; no positive NeMo rendering claim is made.

## Lifecycle Trainer

`weight_edit.generic@1` is registered as a probe-gated TrainerAdapter. It rejects stale base state, file lineage, renderer, family, context, job, or request digests before dispatch. The execution runner invokes a bounded worker, which independently verifies its runtime/helper identity. The initial recipe uses the unchanged legacy key-value solve with target vectors derived from teacher-forced token-loss output gradients on approved demonstrations. This is an explicit gradient-based surrogate, not a ROME/MEMIT implementation or a fully gradient-free method.

Training and preservation token/key budgets are frozen in the recipe. A separately supplied public-train preservation panel is measured before and after editing; candidates exceeding the declared logit-drift bound or producing no effective tensor edit are rejected. Panel provenance is caller-supplied public input, not independent custody or a global safety guarantee. The candidate is exported, checked for portability against the frozen base calibration, sealed with vLLM creation identity, verified on fresh serving, and bound to its execution receipt before lifecycle publication. Tiny-model numerical and mocked-serving orchestration tests pass; the live weight-edit proof-loop arm remains the next validation gate.

## Advanced Methods

The [methods contract](methods/README.md) records the next milestone: locate, calculated edits, and measured preservation, followed by sequential updates, merging, portable directions, and restricted training. It also specifies the frozen gym-to-edit request, bounded preservation panel, and receipts. These remain planned; holdout is final one-shot evaluation, never feedback for iterative stopping or target optimization.