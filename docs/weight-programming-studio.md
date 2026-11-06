# Weight Programming Studio — Architecture & PRD Traceability Guide

This document maps the **Weight Programming Studio** product requirements (`WP-01`–`WP-24`, Experimental Search Intelligence, and Priority Experiments `A`–`D`) to their authoritative Rust and Python implementations in this repository.

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
| `gpu.py` | Hardware GPU probing, DLPack view contracts (`export_dlpack_contract`, `import_dlpack_verified`), copy-on-write GPU dispatch (`dispatch_weight_operation`), and VRAM leasing (`GpuResourceScheduler`, `T108`–`T109`). |
| `inspection.py` | Bounded weight statistics, activation summaries, attention entropy, and gradient-assisted saliency (`T047`). |
| `diagnostics.py` | Failure clustering, contrastive localization, causal activation patching, and structured hypothesis cards (`T048`–`T051`). |
| `autonomous.py` | Interchangeable agent provider contracts, constrained IR synthesis, multi-branch search, ablation pruning, and Gym Scientist audits (`T052`–`T058`). |
| `flagship.py` | Pretrained repetition-loop repair, matched LoRA baseline comparison, independent replay bundles, and M3 readiness reporting (`T060`–`T063`). |
| `m4_modalities.py` & `m4_ecosystems.py` | Research memory, circuit tracing, sparse features, Pareto frontiers, cross-gym interference, symbolic rule construction (`CG-09`), weight transfer (`CG-10`), training/RL baselines (`CG-11`), external gym adapters, and library reuse (`CG-13`). |
| `m5_surgery.py` | Coupled MLP width slicing, GQA/RoPE attention head surgery, layer reorganization, structural constructors, group quantization, compression accounting, and serving/efficiency benchmarks (`T102`–`T119`). |
| `export_formats.py` | Native HuggingFace Safetensors and **GGUF v3** (`F32`, `F16`, `Q8_0`) export and reload round-trip verification, block-quantized `Q8_0` matrix editing (`execute_quantized_q8_matrix_edit`), and sharded Safetensors `mmap` copy-on-write checkpoint loader (`ShardedMmapCheckpoint`, up to `64 GiB`). |
| `patch_ux.py` | Layer-by-layer weight diff summaries (`diff_model_weights`: Frobenius, spectral $\ell_2$, and $\ell_\infty$ norms), cryptographic patch signing (`sign_patch_envelope`, `verify_patch_signature`), and one-command rollback (`create_rollback_handle`, `rollback_model_patch`). |

---

## 2. Product Requirements Traceability (`WP-01`–`WP-24`)

| PRD ID | Requirement | Implementation |
| :--- | :--- | :--- |
| **WP-01** | Import & inspect open-weight models | `nf model inspect`, `neural_foundry.inspection.BoundedInspector`, `neural_foundry.pretrained.intake_local_qwen2`, `neural_foundry.pythia.PythiaModel.from_pretrained_dir` |
| **WP-02** | Interchangeable Claude & GPT-6 agents | `neural_foundry.prd_adapters.FrontierAgentProgrammer` (`claude-4-5-sonnet`, `gpt-6-astra`) & `neural_foundry.autonomous.AgentProviderContract` |
| **WP-03** | Versioned `weightctl` CLI & tool API | `crates/nf-cli/src/weightctl.rs`, `crates/nf-cli/src/evals.rs`, `crates/nf-cli/src/studio.rs` |
| **WP-04** | Typed Weight IR & immutable outputs | `crates/nf-weight-ir/src/program.rs`, `crates/nf-weight-ir/src/compilation.rs`, `crates/nf-surgeon/src/transaction.rs` |
| **WP-05** | Direct tensor ops, projections & deltas | `crates/nf-surgeon/src/basic.rs` & `crates/nf-surgeon/src/representation.rs` |
| **WP-06** | Import custom gyms & feedback contracts | `crates/nf-experiments/src/intake.rs`, `crates/nf-experiments/src/validation.rs`, `neural_foundry.pythia.ReusableTaskGym` |
| **WP-07** | External eval suites & promotion gates | `neural_foundry.evals` (`import_suite`, `import_lm_eval_harness_task`, `import_inspect_ai_eval_spec`, `verify_promotion`) |
| **WP-08** | Agent loops with resource limits & resume | `crates/nf-experiments/src/coordinator.rs`, `crates/nf-experiments/src/manual_workflow.rs` |
| **WP-09** | Sealed-test isolation & contamination checks | `crates/nf-experiments/src/coordinator/splits.rs`, `crates/nf-experiments/src/coordinator/exposure.rs`, `neural_foundry.evals.run_eval` |
| **WP-10** | Full edit, probe, eval & lineage ledger | `crates/nf-patches/src/replay.rs`, `crates/nf-experiments/src/scientific_graph.rs` |
| **WP-11** | Approval, atomic promotion, rollback & export | `crates/nf-experiments/src/coordinator/promotion.rs`, `crates/nf-checkpoint/src/export.rs` |
| **WP-12** | Tenant isolation, permissions, budgets & audit | `crates/nf-experiments/src/sandbox.rs`, `neural_foundry.m4_ecosystems.enforce_private_gym_policy` |
| **WP-13** | `EasyEdit` adapters (`ROME`, `MEMIT`, `AlphaEdit`) | `neural_foundry.prd_adapters.run_easyedit_adapter`, `WeightProgrammingStudio.run_easyedit` |
| **WP-14** | `mergekit` adapters (`task_arithmetic`, `ties`, `dare`) | `neural_foundry.prd_adapters.run_mergekit_adapter`, `WeightProgrammingStudio.run_mergekit` |
| **WP-15** | Causal activation patching & probes | `neural_foundry.diagnostics`, `neural_foundry.m4_modalities.evaluate_mechanism_confirmation` |
| **WP-16** | Multi-candidate branch comparison | `neural_foundry.pythia.run_pythia_gym_task_learning_study`, `neural_foundry.autonomous` |
| **WP-17** | Automated LoRA & fine-tuning baselines | `neural_foundry.m4_ecosystems` (`run_lora_baseline`, `run_sft_baseline`, `run_distillation_baseline`, `run_rl_baseline`) |
| **WP-18** | Reusable compatibility-checked recipes | `crates/nf-experiments/src/library_lifecycle.rs`, `neural_foundry.m4_ecosystems.evaluate_library_recipe_reuse` |
| **WP-19** | Checkpoint branching, composition & conflicts | `crates/nf-patches/src/composition.rs`, `WeightProgrammingStudio.compose_patches` |
| **WP-20** | Evolutionary search over edit programs | `crates/nf-patches/src/evolutionary_ops.rs`, `neural_foundry.m4_modalities.compute_pareto_population_frontiers` |
| **WP-21** | Program synthesis for small neural networks | `neural_foundry.m4_modalities.construct_algorithmic_relu_rule` (`CG-09`) |
| **WP-22** | Cross-model / cross-version weight transfer | `neural_foundry.m4_modalities` (`estimate_permutation_alignment`, `solve_fitted_linear_alignment`, `execute_mapped_weight_transfer`) |
| **WP-23** | Searchable registry of recipes & gym results | `VerifiedLibraryCatalog`, `ScientificGraphStore`, `ResearchMemoryStore` |
| **WP-24** | Learned weight-space proposers | `neural_foundry.m4_ecosystems` (`build_proposer_dataset`, `run_proposer_training_study`, `evaluate_learned_proposer_inference`) |

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
assert prd_report.all_experiments_succeeded is True
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

All 20 audit requirements are enforced across the Rust core, Python SDK, CLI, and CI workflows (`tests/test_audit_20_points.py`):

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
| **14** | **HF Safetensors & GGUF v3 Round-Trip** | `export_huggingface_safetensors`, `reload_and_verify_huggingface_safetensors`, `export_gguf_v3` (`F32`, `F16`, `Q8_0`), and `reload_and_verify_gguf_v3` rebuild edited checkpoints from disk and verify 100% behavioral preservation. |
| **15** | **Live LLM-Provider Agent Harness** | `FrontierAgentProgrammer` supports live HTTP API calls (Anthropic/OpenAI) and pluggable `llm_transport`, recording turn-by-turn `AgentTurnTranscript` logs, token counts, and USD costs, and failing closed on budget overrun (`AgentBudgetExceededError`). |
| **16** | **Feedback-Tool vs. Weight-Reading Contrast** | `compare_weight_reading_only_vs_feedback_enabled_agent` empirically demonstrates why weight-reading alone fails (`< 50%` holdout) whereas forward passes, activation probes, and practice sets reach `100%` holdout accuracy. |
| **17** | **Sandboxed Edit Execution in CI** | `.github/workflows/m1.yml` and `.github/workflows/m2.yml` pull the pinned rootless/no-network container image and run `tests/test_sandbox_isolation.py`. |
| **18** | **Flagship End-to-End Loop-Repair Study** | `flagship.run_flagship_loop_repair_study_and_publish_bundle` repairs a documented pathological repetition attractor (`1.0` $\to$ `0.0` loop rate) and exports signed patch envelopes, transcripts, exposure ledgers, and HF + GGUF v3 bundles. |
| **19** | **Hosted CI Runners & Default Scratch** | Both `.github/workflows/m1.yml` and `.github/workflows/m2.yml` run on `ubuntu-latest` with valid `uv` installation, and all Rust tests default `KIROCREW_SCRATCH` to `std::env::temp_dir()` when unset. |
| **20** | **Patch Diff, Signing, Audit & Rollback UX** | `patch_ux.py` and CLI `nf patch diff|sign|verify-signature|audit|rollback` (`weightctl patch ...`) provide layer-wise Frobenius/$\ell_2$/$\ell_\infty$ diffs, tamper-evident signatures, and one-command rollback. |

---

## 6. Closing the Real-Model Gap, Easy Adoption & Supply-Chain Trust Layer

| Pillar | Capability | Implementation |
| :--- | :--- | :--- |
| **Real-Model Gap: Multi-Architecture Registry** | Registry-driven intake for `Qwen2`, `Qwen3`, `Llama`, `Gemma`, `Gemma2`, `Mistral`, and `GPTNeoX`/`Pythia` rather than single-architecture hardcoded loaders. | `pretrained.PRETRAINED_ARCHITECTURE_REGISTRY`, `pretrained.intake_pretrained_checkpoint`, `pythia.BUILTIN_ARCHITECTURE_PROFILES` |
| **Real-Model Gap: End-to-End `mmap` Streaming** | All pretrained loaders (`intake_pretrained_checkpoint`, `intake_local_qwen2`, `PythiaModel.from_pretrained_dir`, `Qwen2CausalLM.from_pretrained_dir`) stream single-file and multi-shard checkpoints via `ShardedMmapCheckpoint` (`mmap` zero-copy reads) instead of `Path.read_bytes()`. | `export_formats.ShardedMmapCheckpoint`, `pretrained.py`, `pythia.py` |
| **Real-Model Gap: `FP8`, `Q4_K_M` & `RoundingContract`** | Support `bf16`, `f16`, `fp8` (`fp8_e4m3fn`, `fp8_e5m2`), and GGUF v3 `Q8_0`, `Q4_0`, and `Q4_K_M` super-blocks with an explicit `RoundingContract` (`source_dtype`, `accumulation_dtype="f32"`, `output_dtype`, `rounding_mode`, `ulp_tolerance`, `contract_digest`) committed in the patch identity. | `ir.RoundingContract`, `export_formats.execute_fp8_matrix_edit`, `export_formats.quantize_matrix_q4_k_m` |
| **Real-Model Gap: Live Runtimes & GPU Agent Tools** | Support `pytorch_cpu`, `pytorch_cuda`, `transformers`, `vllm`, and `llama_cpp` in `evals.py` (`probe_external_serving_runtimes`), and expose GPU-resident forward passes and activation probes (`gpu_forward_and_activation_probe`) inside `FrontierAgentProgrammer.run_autonomous_tool_loop`. | `evals.py`, `gpu.py`, `prd_adapters.py` |
| **Adoption UX: Prebuilt ABI3 Wheels & One-Command CLI** | Prebuilt `abi3-py312` wheel workflow (`.github/workflows/wheels.yml`), `nf-studio` script entrypoint, and top-level `nf repair --model Qwen/... --failure looping` and `nf verify <patch>` in both Rust CLI (`crates/nf-cli/src/studio.rs`) and Python (`studio.repair`, `studio.verify_patch`). | `.github/workflows/wheels.yml`, `pyproject.toml`, `crates/nf-cli/src/studio.rs`, `studio.py` |
| **Adoption UX: Hugging Face Hub, `TransformerLens` & Tool API** | Export HF Hub model-card bundles (`export_huggingface_hub_bundle`: `README.md` + `weight_patch.nf.json` + `model.safetensors`), wrap models in `TransformerLensHookAdapter` (`run_with_cache`, `run_with_hooks`), and expose `WeightEditorToolInterface` (`studio.editor_tool`) for Claude Code, Codex, and custom agent loops. | `export_formats.py`, `pythia.TransformerLensHookAdapter`, `studio.PluginRegistry`, `studio.WeightEditorToolInterface` |
| **Trust Moat: Open Patch Format v1, SPDX SBOM & Backdoor Scanner** | `nf_open_weight_patch_v1` bundles an SPDX Weight-Patch SBOM (`spdx-weight-patch-sbom-v1`), `rounding_contract`, `evaluation_provenance`, and `BackdoorScanReport` (`scan_patch_for_backdoor_patterns` catching undeclared layer mutations, trojan spectral/$\ell_\infty$ spikes, non-finite weights, and neutral-prompt hijacking), plus independent third-party replication (`flagship.replicate_published_flagship_bundle`). | `patch_ux.py`, `flagship.py`, `crates/nf-cli/src/studio.rs` |


