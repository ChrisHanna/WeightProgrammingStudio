# Qualification Matrix

Each row is one combination of operation, architecture, dtype, backend and shape policy, qualified separately. `tests/test_qualification_matrix.py` parses this table and fails if a row breaks the rules below.

## Evidence levels

| Level | Meaning | Rule the test enforces |
| --- | --- | --- |
| `verified` | The cited test passes at the stated scope. | Must cite an existing test. Unless the architecture is `tensor`, the test must carry `@pytest.mark.qualification`. The architecture can't be a fixture. |
| `fixture-only` | Works on a small built-in or synthetic model or tensor. This proves the software path only. | Must cite an existing test. |
| `experimental` | Code exists, but no qualification run has passed. | Any cited test must exist. |
| `unsupported` | Refused or not implemented. | Any cited test must exist. |

`tensor` means the row is about a raw tensor and claims nothing about any model architecture. **No model architecture is `verified` for editing yet** (the one model row below is serving parity for one configuration, not editing), because no real-checkpoint test meets the Milestone 1 pass conditions in [milestones.md](milestones.md).

A row records that the cited test passed when this matrix was last audited. The test only checks that the citation exists. It doesn't run the cited test. Run the full suite with `NF_QUALIFICATION=1` to make skipped qualification tests fail.

## Tensor operations

| Operation | Architecture | Dtype | Backend | Shape policy | Evidence | Test |
| --- | --- | --- | --- | --- | --- | --- |
| Native IR families (scale, mask, replace, permute, linear map, projection, low rank) | tensor | f32, f64 | native Rust CPU | same shape | verified | tests/test_cpu_differential.py::test_cpu_differential_all_native_families_against_real_torch |
| Rectangle-scoped edit keeps other coordinates | tensor | f32, f64 | native Rust CPU | same shape | verified | tests/test_cpu_differential.py::test_cpu_differential_every_small_rectangle_preserves_other_coordinates |
| add_delta | tensor | f16, bf16, f32 | native Rust CPU | same shape | fixture-only | tests/test_audit_20_points.py::test_points_11_12_13_14_f16_bf16_quantized_sharded_mmap_gpu_and_gguf_roundtrip |
| Native IR families through trusted Torch worker | tensor | f32, f64 | nf.trusted_torch_cpu | same shape | experimental | tests/test_torch_edit_backend.py::test_torch_edit_backend_all_families_actual_worker_full_bytes |
| scale on packed blocks | tensor | Q8_0 | Python + native Rust CPU | same shape | fixture-only | tests/test_audit_20_points.py::test_points_11_12_13_14_f16_bf16_quantized_sharded_mmap_gpu_and_gguf_roundtrip |
| Standard Q4_K editing or export | any | Q4_K | none | none | unsupported | tests/test_enhancements.py::test_q4_k_is_not_claimed |
| Device dispatch with bf16 | tensor | bf16 | Torch CPU; CUDA not exercised | same shape | fixture-only | tests/test_gpu_support.py::test_gpu_dispatch_dlpack_contract_immutability_and_no_silent_fallback |
| Sharded copy-on-write export | tensor | f32 | Python mmap | same shape | fixture-only | tests/test_audit_20_points.py::test_points_11_12_13_14_f16_bf16_quantized_sharded_mmap_gpu_and_gguf_roundtrip |

## Model edits

| Operation | Architecture | Dtype | Backend | Shape policy | Evidence | Test |
| --- | --- | --- | --- | --- | --- | --- |
| EditPlan apply and replay | GPT-NeoX mini fixture | f64 | native Rust CPU | same shape | fixture-only | tests/test_edit_plan.py::test_third_party_replay_reproduces_exact_weights |
| Gym study | GPT-NeoX Pythia-14m | f64 | native Rust CPU | same shape | experimental | tests/test_pythia_edit.py::test_real_eleutherai_pythia_14m_weight_programming_if_cached |
| Gym study | GPT-NeoX Pythia-31m | f64 | native Rust CPU | same shape | experimental | none |
| Attractor repair | GPT-NeoX mini fixture | f64 | native Rust CPU | same shape | fixture-only | tests/test_attractor_repair.py::test_repair_finds_and_reduces_injected_attractor |
| Attractor repair | GPT-NeoX Pythia-14m, Pythia-31m | f64 | native Rust CPU | same shape | experimental | none |
| Pretrained intake and methods | Qwen2 small local checkpoint fixture | f64 | native Rust CPU | same shape | fixture-only | tests/test_pretrained_adapter.py::test_pretrained_adapter_qwen2_intake_and_two_methods_through_native_pipeline |
| Attractor repair, gym study | Qwen2 real checkpoint | f64 | native Rust CPU | same shape | experimental | none |
| Inference and editing | Qwen3, Llama, Mistral, Gemma, Gemma2 | any | none | none | unsupported | none |
| Export and reload | GPT-NeoX mini fixture | f32, f16, Q8_0 | Python exporter and reader | same shape | fixture-only | tests/test_audit_20_points.py::test_points_11_12_13_14_f16_bf16_quantized_sharded_mmap_gpu_and_gguf_roundtrip |

## Structural surgery and programs

| Operation | Architecture | Dtype | Backend | Shape policy | Evidence | Test |
| --- | --- | --- | --- | --- | --- | --- |
| MLP, attention and layer surgery on supplied tensors | tensor | f64 | Python Torch | shape-changing | fixture-only | tests/test_m5_surgery_and_efficiency.py::test_mlp_surgery_attention_surgery_layer_reorganization_and_structural_constructors |
| Surgery producing a loadable checkpoint | any | any | none | shape-changing | unsupported | none |
| Rule construction (5-op ReLU) | two-layer fixture | f64 | Python Torch | fixed | fixture-only | tests/test_m4_engines.py::test_rule_construction_compiles_exact_relu_weights_with_independent_oracle |
| Mapped transfer with supplied or inferred permutation | two-layer fixture | f64 | Python Torch | same shape | fixture-only | tests/test_m4_engines.py::test_mapped_transfer_permutation_alignment_fitted_alignment_and_transfer_protocols |
| Transfer benchmark, proposer training | any | any | none | none | unsupported | tests/test_m4_engines.py::test_library_reuse_strategy_retrieval_cross_model_protocol_proposer_dataset_proposer_training_learned_proposer_transfer_benchmark_research_source_records_and_m4_feature_acceptance |
| Serving parity only, one configuration: vLLM 0.33.0 `vllm serve` with FLEX_ATTENTION, greedy, against transformers | GPT-NeoX (Pythia-14M only) | f32 served, f32 and f64 references | vLLM on CUDA | same shape | verified | tests/test_real_runtimes.py::test_vllm_serves_exported_base_and_edit_with_transformers_parity |
| TensorRT qualification, systems profile, WeightEfficiencyBench | any | any | none | none | unsupported | tests/test_m5_surgery_and_efficiency.py::test_quantization_gym_compression_gym_generation_gym_serving_protocol_vllm_adapter_tensorrt_adapter_systems_profile_kernel_admission_efficiency_benchmark_and_backend_qualification |
