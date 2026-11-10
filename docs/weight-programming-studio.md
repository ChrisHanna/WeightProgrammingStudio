# Weight Programming Studio — Architecture & PRD Traceability Guide

This document maps product requirements to code. A row showing where code lives is **not** evidence that the requirement is met. Evidence levels are in the [qualification matrix](qualification-matrix.md), and limits are in the [README](../README.md#limitations).

---

## 1. Crate & Package Architecture

### Rust Workspace (`crates/`)
| Crate | Responsibility |
| :--- | :--- |
| `nf-core` | Canonical JSON serialization (`CanonicalDocument`), SHA-256 content digests (`ContentDigest`), capability manifests, and schema validation (`MethodRecord`, `EvidenceLabel`, `Permissions`, `StudyContract`). |
| `nf-checkpoint` | Safetensors parser/serializer, content-addressed `ArtifactStore`, and atomic checkpoint export/restore (`ExportLimits`). |
| `nf-graph` | Tensor inventory verification (`inventory`) and symbolic architecture dependency & shape solver (`shape_solver`, `T100`). |
| `nf-weight-ir` | Declarative Weight IR (`program`), search-space resolution (`search_space`), CPU/CUDA compiler lowering (`compilation`), and structural surgery lowering (`structural_lowering`, `T101`). |
| `nf-surgeon` | Copy-on-write transactional weight execution (`transaction`), basic/representation kernels (`basic`, `representation`), history certification (`history`), and native `Qwen2` causal smoke adapter (`transaction::qwen2`). |
| `nf-patches` | Immutable patch DAG replay (`replay`), multi-patch dependency & conflict composition (`composition`, `T069`), and bounded evolutionary mutation/recombination (`evolutionary_ops`, `T070`). |
| `nf-experiments` | Durable SQLite study coordinator, split broker, exposure ledger, OCI/static-ELF sandbox, statistical plans, scientific loop (`scientific_loop`, `T057`), typed scientific graph (`scientific_graph`, `T064`), and versioned recipe library (`library_lifecycle`, `T090`). |
| `nf-cli` | Binaries `nf` and `weightctl` exposing core IR/checkpoint/study commands plus `eval` (`import`, `run`, `compare`, `verify`) and `studio` (`solve-shape`, `lower-structural`, `compose-patches`, `scientific-graph`, `library`). |
| `nf-python` | PyO3 `abi3` native extension (`neural_foundry._native`) bridging Rust handles and studio solvers into Python. |

### Python Studio SDK (`python/neural_foundry/`)
| Module | Responsibility |
| :--- | :--- |
| `studio.py` | Unified `WeightProgrammingStudio` facade: selectable model catalog, reusable task gyms, Rust PyO3 solvers, `EasyEdit`/`mergekit` launchers, frontier agent factory, and `T120` release conformance reporter. |
| `evals.py` | First-class Evaluation Engine (`import_suite`, `import_lm_eval_harness_task`, `import_inspect_ai_eval_spec`, `run_eval`, `compare_runs`, `verify_promotion`, `render_dashboard_markdown`). |
| `pythia.py` | `PythiaModel` (`GPTNeoXForCausalLM`, e.g., `EleutherAI/pythia-14m` and `EleutherAI/pythia-31m`), `Qwen2CausalLM`, `ReusableTaskGym`, and the Contrastive-Prefix-Nullspace Weight Programmer. |
| `prd_adapters.py` | `EasyEdit` adapters (`ROME`, `MEMIT`, `AlphaEdit`), `mergekit` adapters (`task_arithmetic`, `ties`, `dare`), `FrontierAgentProgrammer` (`claude-4-5-sonnet`, `gpt-6-astra`), and Priority Experiments `A`–`D` runner. |
| `gpu.py` | Executable CUDA probing, external GPU runtime discovery, DLPack view contracts, copy-on-write GPU dispatch, training placement, and VRAM leasing (`T108`–`T109`). |
| `gpu_training_worker.py` | Trusted local LoRA/SFT worker. It returns CPU `float64` checkpoints to a CPU parent; direct-edit scoring, replay, and archive decisions never cross that process boundary. |
| `frontier_evaluation.py` / `independent_frontier_worker.py` | Loads digest-bound replacement tensors from a frozen snapshot and evaluates each materialized frontier candidate in a fresh CPU process. No search-loop or proposer object crosses the boundary. |
| `inspection.py` | Bounded weight statistics, activation summaries, attention entropy, and gradient-assisted saliency (`T047`). |
| `diagnostics.py` | Failure clustering, contrastive localization, causal activation patching, and structured hypothesis cards (`T048`–`T051`). |
| `autonomous.py` | Interchangeable agent provider contracts, constrained IR synthesis, multi-branch search, ablation pruning, and Gym Scientist audits (`T052`–`T058`). |
| `flagship.py` | Pretrained repetition-loop repair, matched LoRA baseline comparison, independent replay bundles, and M3 readiness reporting (`T060`–`T063`). |
| `m4_modalities.py` & `m4_ecosystems.py` | Fixture-level rule construction and transfer, small linear baselines, and evaluators that score caller-supplied numbers (`evidence` field). External gym adapters don't run the named packages. Transfer benchmark and proposer training are unsupported. |
| `m5_surgery.py` | Tensor slicing for surgery, grouped quantization, and evaluators over caller-supplied numbers. Serving qualification, systems profile and WeightEfficiencyBench are unsupported. |
| `export_formats.py` | Native HuggingFace Safetensors and **GGUF v3** (`F32`, `F16`, `Q8_0`) export and reload round-trip verification, block-quantized `Q8_0` matrix editing (`execute_quantized_q8_matrix_edit`), and sharded Safetensors `mmap` copy-on-write checkpoint loader (`ShardedMmapCheckpoint`, up to `64 GiB`). |
| `patch_ux.py` | Layer-by-layer weight diff summaries (`diff_model_weights`: Frobenius, spectral $\ell_2$, and $\ell_\infty$ norms), cryptographic patch signing (`sign_patch_envelope`, `verify_patch_signature`), and one-command rollback (`create_rollback_handle`, `rollback_model_patch`). |
### Method Dispatch Contracts

`studio.methods.list_methods()` includes a versioned `contract` for every entry:
declared argument kinds, result types, output scope, owning implementation ID,
supported model types, and evidence level. Local ROME/MEMIT/AlphaEdit and merging
implementations are not labeled as execution of the EasyEdit or mergekit packages.
The existing `lora_baseline` is a matrix-training fixture; surgery returns sliced
tensors, not a standalone checkpoint. Model-edit reports also do not imply a
qualified export or accepted model.

`studio.methods.run()` checks declared arguments before dispatch and result types
after execution. These checks do not certify numerical fidelity, enforce a study
permission, or grant promotion authority. Implementation IDs name local code and
protocol versions, not source-hash attestations. Existing custom registrations
without a contract stay compatible and are listed as `undeclared`.

```python
import torch
from neural_foundry.studio import MethodContract

@studio.methods.register(
   "matrix_identity",
   contract=MethodContract(
      inputs=(("weight", "matrix"),),
      output_scope="matrix",
      result_types=(torch.Tensor,),
   ),
)
def matrix_identity(weight: torch.Tensor) -> torch.Tensor:
   return weight.clone()
```

Native family accounting, sealed-data custody, GPU reservations, independent
statistics, serving qualification, and recipe publication remain separate
integration requirements; a registry contract does not establish any of them.

### CPU/GPU Ownership

Canonical model state, closed-form proposal synthesis, candidate acceptance, edit-graph replay, hashing, and independent CPU reference evaluation remain on CPU `float64`. LoRA/SFT optimization uses a CUDA `float32` training copy when a synchronized kernel probe succeeds. A separate `NF_GPU_PYTHON` interpreter is allowed to perform that gradient work, but every returned checkpoint is copied to CPU `float64` and rescored by the parent before entering a frontier. vLLM qualification independently requires its own executable CUDA probe and never treats driver visibility as GPU readiness.

goal-search --gym KEY` pass a registered `ReusableTaskGym` into the iterative
`WeightProgrammingStudio.run_goal_search(..., gym=...)` and the `nf-studio goal-search --gym KEY` command pass a registered `ReusableTaskGym` into the iterative
controller. Public corpus splits are rebuilt to exclude that Gym's task and
holdout text; the Gym content digest is part of the cycle identity. Search output
directories must be new or empty and are never silently reused. External GPU
training leases are process-local cooperative reservations only.

The Python workflow is a local research preview, not native experiment authority:
it does not issue a native family lease, reserve native budgets, prove sealed
target custody, or mint a promotion grant. Independent deployment remains
ineligible until those native witnesses and a genuinely held-out target panel are
integrated. The runner explicitly records `promotion_authorized: false`.
---

## 2. Product Requirements Traceability (`WP-01`–`WP-24`)

| PRD ID | Requirement | Implementation | Status |
| :--- | :--- | :--- | :--- |
| **WP-01** | Import & inspect open-weight models | `nf model inspect`, `neural_foundry.inspection.BoundedInspector`, `neural_foundry.pretrained.intake_local_qwen2`, `neural_foundry.pythia.PythiaModel.from_pretrained_dir` | experimental |
| **WP-02** | Interchangeable agents | `neural_foundry.prd_adapters.FrontierAgentProgrammer`, `neural_foundry.autonomous.AgentProviderContract` | experimental; no live call audited |
| **WP-03** | Versioned `weightctl` CLI & tool API | `crates/nf-cli/src/weightctl.rs`, `crates/nf-cli/src/evals.rs`, `crates/nf-cli/src/studio.rs` | not verified in this audit |
| **WP-04** | Typed Weight IR & immutable outputs | `crates/nf-weight-ir/src/program.rs`, `crates/nf-weight-ir/src/compilation.rs`, `crates/nf-surgeon/src/transaction.rs` | verified for f32/f64 tensors |
| **WP-05** | Direct tensor ops, projections & deltas | `crates/nf-surgeon/src/basic.rs`, `crates/nf-surgeon/src/representation.rs` | verified for f32/f64 tensors |
| **WP-06** | Import custom gyms & feedback contracts | `crates/nf-experiments/src/intake.rs`, `crates/nf-experiments/src/validation.rs`, `neural_foundry.pythia.ReusableTaskGym` | not individually audited |
| **WP-07** | External eval suites & promotion gates | `neural_foundry.evals` (`import_suite`, `import_lm_eval_harness_task`, `import_inspect_ai_eval_spec`, `verify_promotion`) | not individually audited |
| **WP-08** | Agent loops with resource limits & resume | `crates/nf-experiments/src/coordinator.rs`, `crates/nf-experiments/src/manual_workflow.rs` | not individually audited |
| **WP-09** | Sealed-test isolation & contamination checks | `crates/nf-experiments/src/coordinator/splits.rs`, `crates/nf-experiments/src/coordinator/exposure.rs`, `neural_foundry.evals.run_eval` | not individually audited |
| **WP-10** | Full edit, probe, eval & lineage ledger | `crates/nf-patches/src/replay.rs`, `crates/nf-experiments/src/scientific_graph.rs` | not individually audited |
| **WP-11** | Approval, atomic promotion, rollback & export | `crates/nf-experiments/src/coordinator/promotion.rs`, `crates/nf-checkpoint/src/export.rs` | not individually audited |
| **WP-12** | Tenant isolation, permissions, budgets & audit | `crates/nf-experiments/src/sandbox.rs`; `enforce_private_gym_policy` only checks caller declarations | not verified; no tenant isolation found |
| **WP-13** | `EasyEdit` adapters (`ROME`, `MEMIT`, `AlphaEdit`) | `neural_foundry.prd_adapters.run_easyedit_adapter`, `WeightProgrammingStudio.run_easyedit` | not individually audited |
| **WP-14** | `mergekit` adapters (`task_arithmetic`, `ties`, `dare`) | `neural_foundry.prd_adapters.run_mergekit_adapter`, `WeightProgrammingStudio.run_mergekit` | not individually audited |
| **WP-15** | Causal activation patching & probes | `neural_foundry.diagnostics`; `evaluate_mechanism_confirmation` scores caller-supplied effects | fixture-only |
| **WP-16** | Multi-candidate branch comparison | `neural_foundry.pythia.run_pythia_gym_task_learning_study`, `neural_foundry.autonomous` | not individually audited |
| **WP-17** | LoRA & fine-tuning baselines | `neural_foundry.m4_ecosystems` baselines train one linear matrix | fixture-only |
| **WP-18** | Reusable compatibility-checked recipes | `crates/nf-experiments/src/library_lifecycle.rs`; `evaluate_library_recipe_reuse` takes the outcome as an argument | not verified in this audit |
| **WP-19** | Checkpoint branching, composition & conflicts | `crates/nf-patches/src/composition.rs`, `WeightProgrammingStudio.compose_patches` | not individually audited |
| **WP-20** | Evolutionary search over edit programs | `crates/nf-patches/src/evolutionary_ops.rs`, `neural_foundry.m4_modalities.compute_pareto_population_frontiers` | not individually audited |
| **WP-21** | Program synthesis for small neural networks | `neural_foundry.m4_modalities.construct_algorithmic_relu_rule` (`CG-09`) | not individually audited |
| **WP-22** | Cross-model / cross-version weight transfer | `neural_foundry.m4_modalities` transfer on a two-layer fixture; `run_transfer_benchmark` not implemented | fixture-only; no cross-model transfer |
| **WP-23** | Searchable registry of recipes & gym results | `VerifiedLibraryCatalog`, `ScientificGraphStore`, `ResearchMemoryStore` | not individually audited |
| **WP-24** | Learned weight-space proposers | `build_proposer_dataset` only; `run_proposer_training_study` raises `NotImplementedError` | unsupported |

---

## 3. Running Reusable Gyms & Priority Experiments `A`–`D`

```python
from neural_foundry.studio import WeightProgrammingStudio

studio = WeightProgrammingStudio()

# 1. Run a reusable multi-skill Gym on any selectable model (Pythia-14m, Pythia-31m, Qwen2, or mini fixture)
gym_report = studio.run_reusable_gym_study(
    "nf/mini-pythia-fixture",
    gym="algorithmic_three_skill",
)
print(gym_report.promotion_verification["markdown_dashboard"])

# 2. Run PRD Priority Experiments A, B, C, and D end-to-end
prd_report = studio.run_priority_experiments("nf/mini-pythia-fixture")
print(prd_report.all_experiments_succeeded)  # fixture run; not a qualification
```

---

## 4. Extensible Studio Registries & `WeightIRTransactionBuilder`

`WeightProgrammingStudio` exposes four pluggable registries plus a reusable Rust Weight-IR transaction builder so researchers can extend every layer without modifying core files:

1. **`studio.models` (`ModelRegistry`)**:
   - Pre-populated with `EleutherAI/pythia-14m`, `EleutherAI/pythia-31m`, `Qwen/qwen2-dense-mini`, `nf/mini-pythia-fixture`, and `BUILTIN_ARCHITECTURE_PROFILES` (`gpt_neox`, `qwen2`, `llama`, `mistral`, `gemma`).
   - Register custom architectures with `studio.models.register_architecture(ArchitectureSiteProfile(...))` and custom models with `studio.models.register_model(...)`.
2. **`studio.gyms` (`GymRegistry`)**:
   - Pre-populated with `algorithmic_three_skill` and `factual_knowledge`.
   - Build and register new multi-skill gyms from prompt-target rules via `studio.gyms.from_rules(...)`.
3. **`studio.methods` (`MethodRegistry`)**:
   - Pre-populated with `rome`, `memit`, `alphaedit`, `contrastive_subspace`, `task_arithmetic`, `ties`, `dare`, `mlp_surgery`, `attention_surgery`, and `lora_baseline`.
   - Register custom weight-editing methods via `@studio.methods.register("my_method")` and invoke them via `studio.methods.run("my_method", ...)`.
4. **`studio.agents` (`AgentRegistry`)**:
   - Pre-populated with `claude-4-5-sonnet` and `gpt-6-astra`.
   - Register custom LLM or local agent backends via `studio.agents.register_agent("my-custom-agent", extra_tool_schemas=[...])`.
5. **`studio.ir_builder` (`WeightIRTransactionBuilder` in `neural_foundry.ir`)**:
   - Compiles and executes `low_rank_update`, `add_delta`, `scale`, and `subspace_projection` operations across `f16`, `bf16`, `f32`, and `f64` tensors through the Rust `nf-weight-ir` compiler and `nf-surgeon` transactional engine in a single call, returning a `CompiledTransactionReceipt` with verified reload parity.

---

## 5. 20-Point Scientific & Engineering Assurance Matrix

These gates describe the intended checks in `tests/test_audit_20_points.py`. They were audited at fixture scope only, and rows 15, 17 and 19 weren't exercised against a live provider, a Docker daemon or hosted CI. Two tests in that file failed in this audit against an older `nf` binary; they need a rebuild to be judged.

| # | Assurance Gate | Enforcement & Implementation |
| ---: | :--- | :--- |
| **1** | **Fail-Closed Evals** | `neural_foundry.evals.run_eval` and `crates/nf-cli/src/evals.rs` reject unknown model IDs (`--model nonexistent`), missing checkpoints, or missing task/type scores (`MISSING_MODEL`, `MISSING_SCORE`); no fallback default scores exist. |
| **2** | **Real Score Provenance** | Every evaluation run records `score_provenance` (`producer`, `harness`, `runtime`, `sampler`, `seed`, `measurement_mode`). Precomputed dictionaries are labeled `supplied_fact`; only live model/callable executions are labeled `measured_result`. |
| **3** | **Single Readiness Source of Truth** | `generate_release_conformance_report` in `studio.py` derives `capability_gates` and `milestones` directly from `_native.capabilities_json()` and `_native.m2_capabilities_json()` (`crates/nf-core/src/capabilities.rs`) plus live probes. |
| **4** | **CLI `verify` Parity** | `nf eval verify` / `weightctl eval verify` enforces all `required_eval_types`, `noop_blocked`, `robustness`, `efficiency`, `bounded_parameter_change_pct`, and `out_of_target_side_effect_non_regression` checks identically to Python `verify_promotion`. |
| **5** | **Sampler & Runtime Identity in Hash** | Decoding settings (`temperature`, `top_p`, `presence_penalty`, `frequency_penalty`, `max_tokens`, `seed`) and `runtime` (`pytorch_cpu`, `pytorch_cuda`, `llama.cpp`, `vllm`, `rust_cpu`, `rust_cuda`) are bound into `experiment_identity_digest` and `run_digest`. |
| **6** | **Mandatory No-Op & Random Controls** | Every Gym study (`run_pythia_gym_task_learning_study`) compiles and runs `control_noop_null_edit` and `control_random_matched_budget` in Rust and reports `net_gain_over_noop` and `net_gain_over_random`. |
| **7** | **Established Baselines & External Probing** | `probe_external_baseline_packages()` checks for installed `easyeditor`, `mergekit`, and `peft`, labels adapter receipts with `implementation_source`, and fails closed when `require_external_package=True` and the external library is absent. |
| **8** | **Operand-Disjoint Holdout & Exposure Ledger** | `ReusableTaskGym.holdout_rules` use unseen numbers/operands/expressions (`dev_and_holdout_expression_disjoint=True`), sealed by a hash-chained `ExposureLedgerReceipt` proving `holdout_exposure_count_during_synthesis == 0`. |
| **9** | **Pre-Registered Metrics & Multiplicity** | `pre_register_study_plan` and `evaluate_grouped_bootstrap_and_multiplicity` freeze the hypothesis family before execution and compute cluster-resampled (`group_id`) bootstrap CIs with Holm-Bonferroni multiplicity correction. |
| **10** | **Out-of-Target Regression Suite** | `verify_promotion` enforces `out_of_target_side_effect_non_regression`, blocking promotion if retention or per-capability performance drops below policy floors. |
| **11** | **Native `f16`/`bf16` & Quantized `Q8_0` Support** | `crates/nf-surgeon` and `crates/nf-weight-ir` natively load, validate, and edit IEEE-754 `f16` and `bf16` tensors with `f32`/`f64` accumulation; `export_formats.execute_quantized_q8_matrix_edit` supports 32-element block-quantized `Q8_0` editing. |
| **12** | **Large & Sharded `mmap` Checkpoints** | `StateLoadLimits::large_model()` and `CompilerBounds::large_model()` scale up to `64 GiB`, and `ShardedMmapCheckpoint` (`export_formats.py`) provides zero-copy `mmap` reads and copy-on-write per-shard export for `model.safetensors.index.json`. |
| **13** | **Transactional GPU Execution Path** | `gpu.execute_transactional_gpu_edit_and_eval` unites DLPack GPU/device execution with Rust Weight IR parity verification, smoke probes, and source-immutability rollback guarantees. |
| **14** | **HF Safetensors & GGUF v3 Round-Trip** | `export_huggingface_safetensors`, `reload_and_verify_huggingface_safetensors`, `export_gguf_v3` (`F32`, `F16`, `Q8_0`), and `reload_and_verify_gguf_v3` rebuild checkpoints through this package and compare a few probe prompts. Fixture-only. |
| **15** | **Live LLM-Provider Agent Harness** | `FrontierAgentProgrammer` supports live HTTP API calls (Anthropic/OpenAI) and pluggable `llm_transport`, recording turn-by-turn `AgentTurnTranscript` logs, token counts, and USD costs, and failing closed on budget overrun (`AgentBudgetExceededError`). |
| **16** | **Feedback-Tool vs. Weight-Reading Contrast** | `compare_weight_reading_only_vs_feedback_enabled_agent` compares weight-reading alone with feedback tools. Without holdout leakage the fixture reaches `4/6` with feedback, so necessity is not established. |
| **17** | **Sandboxed Edit Execution in CI** | `.github/workflows/m1.yml` and `.github/workflows/m2.yml` pull the pinned rootless/no-network container image and run `tests/test_sandbox_isolation.py`. |
| **18** | **Flagship End-to-End Loop-Repair Study** | `flagship.run_flagship_loop_repair_study_and_publish_bundle` injects a repetition defect, attempts a repair, and publishes the measured outcome (passes on the mini fixture; on `Pythia-14m` the loop is removed but retention fails) with signed patch envelopes, transcripts, exposure ledgers, and HF + GGUF v3 bundles. |
| **19** | **Hosted CI Runners & Default Scratch** | Both `.github/workflows/m1.yml` and `.github/workflows/m2.yml` run on `ubuntu-latest` with valid `uv` installation, and all Rust tests default `KIROCREW_SCRATCH` to `std::env::temp_dir()` when unset. |
| **20** | **Patch Diff, Signing, Audit & Rollback UX** | `patch_ux.py` and CLI `nf patch diff|sign|verify-signature|audit|rollback` (`weightctl patch ...`) provide layer-wise Frobenius/$\ell_2$/$\ell_\infty$ diffs, tamper-evident signatures, and one-command rollback. |

---

## 6. Real-Model Gap, Adoption & Trust Layer (unaudited design intent)

This table lists what the code is built for. Its levels are set by the [qualification matrix](qualification-matrix.md). The backdoor scan is a sanity check, and only GPT-NeoX and Qwen2 can be edited.

| Pillar | Capability | Implementation |
| :--- | :--- | :--- |
| **Real-Model Gap: Multi-Architecture Registry** | Registry-driven intake for `Qwen2`, `Qwen3`, `Llama`, `Gemma`, `Gemma2`, `Mistral`, and `GPTNeoX`/`Pythia` rather than single-architecture hardcoded loaders. | `pretrained.PRETRAINED_ARCHITECTURE_REGISTRY`, `pretrained.intake_pretrained_checkpoint`, `pythia.BUILTIN_ARCHITECTURE_PROFILES` |
| **Real-Model Gap: End-to-End `mmap` Streaming** | All pretrained loaders (`intake_pretrained_checkpoint`, `intake_local_qwen2`, `PythiaModel.from_pretrained_dir`, `Qwen2CausalLM.from_pretrained_dir`) stream single-file and multi-shard checkpoints via `ShardedMmapCheckpoint` (`mmap` zero-copy reads) instead of `Path.read_bytes()`. | `export_formats.ShardedMmapCheckpoint`, `pretrained.py`, `pythia.py` |
| **Real-Model Gap: `FP8`, package-only `NF_Q4` (not standard `Q4_K`) & `RoundingContract`** | Support `bf16`, `f16`, `fp8` (`fp8_e4m3fn`, `fp8_e5m2`), and GGUF v3 `Q8_0`, `Q4_0`, and package-only `NF_Q4` (not standard `Q4_K`) super-blocks with an explicit `RoundingContract` (`source_dtype`, `accumulation_dtype="f32"`, `output_dtype`, `rounding_mode`, `ulp_tolerance`, `contract_digest`) committed in the patch identity. | `ir.RoundingContract`, `export_formats.execute_fp8_matrix_edit`, `export_formats.quantize_matrix_nf_q4` |
| **Real-Model Gap: Live Runtimes & GPU Agent Tools** | Support `pytorch_cpu`, `pytorch_cuda`, `transformers`, `vllm`, and `llama_cpp` in `evals.py` (`probe_external_serving_runtimes`), and expose GPU-resident forward passes and activation probes (`gpu_forward_and_activation_probe`) inside `FrontierAgentProgrammer.run_autonomous_tool_loop`. | `evals.py`, `gpu.py`, `prd_adapters.py` |
| **Adoption UX: Prebuilt ABI3 Wheels & One-Command CLI** | Prebuilt `abi3-py312` wheel workflow (`.github/workflows/wheels.yml`), `nf-studio` script entrypoint, and top-level `nf repair --model Qwen/... --failure looping` and `nf verify <patch>` in both Rust CLI (`crates/nf-cli/src/studio.rs`) and Python (`studio.repair`, `studio.verify_patch`). | `.github/workflows/wheels.yml`, `pyproject.toml`, `crates/nf-cli/src/studio.rs`, `studio.py` |
| **Adoption UX: Hugging Face Hub, `TransformerLens` & Tool API** | Export HF Hub model-card bundles (`export_huggingface_hub_bundle`: `README.md` + `weight_patch.nf.json` + `model.safetensors`), wrap models in `TransformerLensHookAdapter` (`run_with_cache`, `run_with_hooks`), and expose `WeightEditorToolInterface` (`studio.editor_tool`) for Claude Code, Codex, and custom agent loops. | `export_formats.py`, `pythia.TransformerLensHookAdapter`, `studio.PluginRegistry`, `studio.WeightEditorToolInterface` |
| **Trust Moat: Open Patch Format v1, SPDX SBOM & Backdoor Scanner** | `nf_open_weight_patch_v1` bundles an SPDX Weight-Patch SBOM (`spdx-weight-patch-sbom-v1`), `rounding_contract`, `evaluation_provenance`, and `BackdoorScanReport` (`scan_patch_for_backdoor_patterns` catching undeclared layer mutations, trojan spectral/$\ell_\infty$ spikes, non-finite weights, and neutral-prompt hijacking), plus independent third-party replication (`flagship.replicate_published_flagship_bundle`). | `patch_ux.py`, `flagship.py`, `crates/nf-cli/src/studio.rs` |



---

## 7. Checking Inference Against `transformers`

`tests/test_inference_reference.py` compares tokenization, logits, hidden activations and edit deltas with `transformers` on real Pythia checkpoints. It also compares attention weights and sweeps context length up to 2048 tokens, and it prints the precision, attention implementation, context lengths and tolerance for each level. It skips when `transformers` or a checkpoint is missing, and with `NF_QUALIFICATION=1` those same conditions fail instead. It never downloads anything. Its docstring explains the tolerances.

```bash
NF_PYTHIA_14M_DIR=/path/to/pythia-14m NF_PYTHIA_31M_DIR=/path/to/pythia-31m python -m pytest tests/test_inference_reference.py -v -s
```

This covers Pythia only. `Qwen2CausalLM` has been run only on a synthetic fixture and has not been checked against a real checkpoint.
