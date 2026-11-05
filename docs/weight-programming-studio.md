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
   - Compiles and executes `low_rank_update`, `add_delta`, `scale`, and `subspace_projection` operations through the Rust `nf-weight-ir` compiler and `nf-surgeon` transactional engine in a single call, returning a `CompiledTransactionReceipt` with verified reload parity.

