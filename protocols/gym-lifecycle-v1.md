# Gym Studio Lifecycle v1

This protocol connects reusable task, model, trainer, evaluator, artifact, and Scorecard adapters behind one dashboard-neutral in-process entry point. It is local research infrastructure, not an execution sandbox or release authority.

## Adapter contracts

`GymLifecycle` registers four versioned adapter types:

- `TaskAdapter` binds a `GymJobSpec`, interaction type, versioned input/response projection, and a trusted callback that registers its scenario provider, verifier, and scorer. Providers declare stable `scenario_id`, `item_id`, and `family_id` values where available. Structured scenario input is retained as structured data; `prompt` has no special treatment.
- `ModelAdapter` binds one candidate/base identity, family, supported interaction and artifact formats, inference configuration, generation callable, loaded-model identity probe, and candidate reference/loader for training.
- `TrainerAdapter` declares supported interaction formats, model families, methods, and each method's output artifact kind. It consumes frozen projected examples, a frozen training request, and a `CandidateRef`; it returns a reference that it can reload. The model adapter independently loads and checks the resulting candidate state before publication.
- `EvaluatorAdapter` returns typed metrics and a receipt. A local `owns_holdout` declaration is not proof of independent custody.

Compatibility is checked before scenario execution or training dispatch. Unsupported task/model/trainer combinations fail preflight. The task projection defines a versioned trainer format. Structured inputs, exact generated responses, model interaction/request details, generated token IDs, and bounded tool traces are persisted only in private response artifacts; Scorecards contain metric and lineage digests, not raw responses.

Adapter selectors may use `id@version`; an unversioned ID is accepted only when
exactly one version is registered. Optional job backend selectors are checked
against the selected adapters. `neural_foundry.backends.discover_backends()`
lists installed, exact-version-compatible factories; `GymLifecycle.register_backend`
constructs and registers a selected factory. The built-in discovery currently
contains PEFT and AutoModel. The `peft` and `peft-qlora` extras use the CUDA
PyTorch index for `torch 2.9.1+cu128`; `nemo-automodel` and
`nemo-automodel-qlora` use the CUDA index for `torch 2.10.0+cu130`. These are
separate environments, not a single combined install. Dependency presence is
not GPU qualification.

`chat_messages_v1` represents single-response demonstrations. `tool_messages_v1` represents a bounded interaction with one declared tool call. A model adapter may execute only locally registered trusted tool handlers. Its generated arguments are checked against the declared closed schema before dispatch; tool outputs are bounded JSON and remain in private interaction evidence.

`InteractionFormatRegistry` converts bounded chat, single-call tool, and JSON
extraction projections to and from NeMo Responses API rows. The converter is a
data transformation, not a task compiler, scorer bridge, or custody boundary.
Its supported shapes are checked against the NeMo Gym 0.7.0 MCQA request model;
other resource servers must explicitly preserve the format metadata.

## Frozen data and split custody

The Studio dataset authority is pinned in the durable state store. Split assignments are keyed by this authority and by full structured-input, item, document, and family identities. They are not selected by job metadata or job ID. Synthetic variants must explicitly name their source group. A document/group assigned to one split cannot be reassigned to another under a revised job or source digest in the same authority store.

A training dataset is built from selected terminal practice attempts, not from caller metrics. Every example binds the complete input, exact response artifact, interaction type/evidence, scenario and group identities, source split, frozen plan and attempt, scorer/verifier identities, and response-artifact digest. Only valid verifier-passed train-split examples are admitted. Caller-approved responses require explicit reported-data opt-in and remain marked `caller_supplied`; they are not model-run evidence. Dataset manifests contain references/digests, not raw response text, and examples can be reloaded after service restart.

Scenario manifests are validated once at the content-addressed load boundary and indexed by scenario ID. Per-attempt scoring and dataset assembly use the verified index rather than rehashing the full panel for each row.

## Training request and artifact lineage

A training request binds the task/job digest, dataset digest, projected-example digest, projection identity/version, interaction format, exact base candidate and loaded checkpoint/tokenizer/config identity, trainer identity, and complete recipe including seed, method, learning rate, step count, sequence/token budgets, rank, and target modules.

New artifact manifests declare one of `lora`, `qlora`, `full`, `weight_edit`, or `own_model` and bind the frozen request, base and candidate model digests, and artifact file digests. The lifecycle also reads its older request-bound manifest format. Artifact inventories reject symlinks and include nested files; verification does not make an untrusted, concurrently mutable filesystem an immutable snapshot.

The manifest and `CandidateIdentity` can represent a candidate without a base. A base-free worker probe must bind both its artifact digest and candidate state digest; the probe remains trusted in-process code, not independent attestation. Base-dependent methods still require a base. No from-scratch training backend is implemented.

The PEFT sequence trainer supports Qwen3 and Llama-family causal models for `chat_messages_v1` and LoRA. It tokenizes the full projected dataset before dispatch, checks prompt/assistant boundaries and per-sequence, per-batch, and total-dataset token limits, then calls backward one example at a time while accumulating gradients. This bounds activation memory to one sequence, subject to the configured maximum sequence length and the selected base/precision/device. It uses Transformers SDPA; receipt memory measurements are local observations, not a capacity guarantee for other hardware.

The PEFT adapter, worker implementation, and typed recipe config live under `neural_foundry.backends.peft`. The compatibility factory remains available from `gym_adapters`. This worker currently runs in the caller's process, not in an isolated or remote worker. The shared execution runner presently supports local subprocesses only; its receipt records the command, exit status, timeout and output-byte digests. It is not a sandbox and does not provide Docker, Kubernetes, or remote placement.

`nemo_automodel.sft@1` supports local single-GPU full SFT, LoRA, and NF4
QLoRA on the declared Qwen3 and Llama families. It runs AutoModel in a separate
interpreter against only frozen training examples, checks token and step
budgets, stores the worker configuration and receipt, and reloads the candidate
before publication. The tested matrix is Python 3.13.14, AutoModel 0.7.0,
Torch 2.10.0+cu130, Triton 3.6.0, Transformers 5.18.0, torchao 0.18.0,
torchdata 0.11.0, and datasets 5.1.0. LoRA reload additionally requires a
separate Python 3.12.13 interpreter with Torch 2.9.1+cu128, Transformers
4.57.3, and PEFT 0.18.0; QLoRA requires bitsandbytes 0.49.1 in both runtimes.
Set `NF_AUTOMODEL_PYTHON` for discovery and provide the reload interpreter in
`AutoModelBackendConfig`. These pins describe the tested direct dependencies,
not a fully locked or license-reviewed transitive environment. Current training
uses temporary workspaces and does not resume interrupted AutoModel jobs.

The public NeMo dataset compiler writes digest-bound train or validation rows,
but cannot bind a scorer or compile an executable environment yet. The
reverification runner works with NeMo Gym 0.7.0 on declared public splits; its
receipt explicitly denies scorer-binding and custody authority. A caller's split
declaration is not proof of data custody. There is not yet a vLLM lifecycle
adapter, NeMo scorer bridge, NeMo RL trainer, or custody evaluator. A vLLM
model-list response naming a path is not evidence that the server loaded
particular weight bytes. The standalone reward/KL guard is not wired to RL.

After every optimizer step, the trainer atomically commits adapter weights, optimizer state, CPU/CUDA RNG state, exact completed step, data position, request binding, and accumulated loss summary. Resume is accepted only when request, dataset, base, recipe, and projected-example identities match. The final receipt requires the exact contiguous step roster. A killed worker can resume from the last committed step without publishing an incomplete adapter.

Trainer output is written to a request-specific staging directory. The lifecycle reloads and verifies it, writes a digest-bearing file manifest, and atomically renames it to the published artifact path. If the process stops after publication but before indexing the candidate, restart verifies the manifest and reloads the committed artifact rather than retraining. The resulting candidate record binds the dataset, frozen training request, loaded candidate identity, and exact artifact files.

## Evaluation and Scorecard lineage

The evaluator receives the same candidate record produced by training (or the unchanged base record for inference-only runs). The evaluation receipt binds the candidate-record and artifact digests, evaluator identity, split/panel and run digests, metrics, and custody status. The Scorecard references that receipt and candidate record using the same structure for different tasks and model families.

The current public-panel evaluator measures validation only. Holdout access remains unavailable through this path; native independent custody and owner approval are not implemented. Scorecards therefore keep `release_authorized: false`. Missing required controls, reproducibility, safety, or holdout evidence leaves the Scorecard incomplete even when the validation task gate passes.

The lifecycle rejects a holdout result from an evaluator without `owns_holdout`.
That declaration alone is not evidence of custody or a sandbox, and the current
in-process adapter API does not isolate a malicious evaluator.

## Local acceptance evidence

`scripts/run_gym_lifecycle_real.py` exercises support routing with pinned Qwen3-0.6B and SmolLM2-1.7B bases through the same Studio lifecycle and PEFT LoRA trainer, then runs a bounded Qwen tool lookup through that lifecycle. Its real training records use explicitly approved caller-supplied teaching responses, so the dataset/training lineage is valid as reported training input, not a demonstration that the base model solved the task.

The measured CUDA run at `artifacts/gym-lifecycle-training-resume-20261108-final` interrupted Qwen after optimizer step 1 had been committed, resumed from that step, and completed the exact two-step roster. SmolLM2 completed both steps without interruption. Both artifacts reloaded against the pinned base; their receipts record the exact dataset, request, recipe, base identity, step roster, and peak allocated VRAM. A subsequent exact lifecycle retry reused both published artifacts.

Validation accuracy was 0/1 for each trained support candidate. The bounded tool model generated one `lookup_ticket` call, the local handler returned the billing queue, and the task verifier passed; that Scorecard remains incomplete because no holdout, controls, or reproducibility evidence was run. None of these execution receipts qualifies task quality or authorizes release.
