# Method replay status

Status of the milestone rule "Every method returns an `EditPlan`". Replayed means a test in `tests/test_method_replay.py` starts a fresh process that holds only the base weights, a signed edit graph and a public key, and reproduces the `tensor_state_digest` of the method's own candidate. The expected digest is computed from the method's weights without using the plan.

All replay runs in f64 on the Rust CPU executor, because the graph has no precision or device field. The graph format is unchanged.

Each method carries the operation and operands it actually ran (`method_plans.MethodEditStep`) rather than using `EditPlan.adopt_weight`, because `w + (w' - w)` isn't guaranteed to equal `w'` in f64.

## Enforcement

`edit_plan_from_steps` accepts only `VerifiedMethodSteps`, and only `verify_method_steps` creates those, after a bitwise check against the method's weights. Each method path runs that check before returning, so a mismatch raises, or reaches the agent as a refused proposal. Verified steps are bound to their base tensors and are refused on a different base.

This closes the method path, not the language: Python can't stop deliberate use of private names, and `EditPlan` stays open for direct edits.

## Methods

| Method | Where the steps are | Fixture replay | Real Pythia-14M replay |
|---|---|---|---|
| Repair `attractor_projection` (`looping`) | `AttractorRepair.edit_steps`, `studio.repair()["edit_plan"]` | Yes | Yes, with the injected defect from `test_attractor_repair.py` |
| Repair `association_edit` (`parity`; `capitals` and `factual` share the path) | `studio.repair()["edit_plan"]` | Yes, `parity` only | Yes, `parity` only |
| `run_autonomous_tool_loop`: `dampen_singular_components`, `project_out_activation_direction`, `project_out_output_direction`, `edit_associations` | `FrontierAgentSessionReport.submitted_edit_steps` | Yes, each with a scripted transcript; no live LLM | Yes, each with a scripted transcript; no live LLM |
| `run_reasoning_loop_repair_experiment`, both arms | `submitted_edit_steps` | Yes | Yes |
| `run_pythia_gym_task_learning_study`, selected branch | `PythiaGymTaskReport.edit_steps` | Yes | Yes |
| ROME | `EasyEditAdapterReceipt.edit_steps` | Yes | Yes |
| MEMIT | `EasyEditAdapterReceipt.edit_steps` | Yes, one layer, several, and duplicates | Yes, one layer and two |
| AlphaEdit | `EasyEditAdapterReceipt.edit_steps` | Yes, two layers | Yes, two layers |
| mergekit `task_arithmetic`, `ties`, `dare` | `MergekitAdapterReceipt.edit_steps` | Yes | Yes |
| `compose_overlapping_patches_orthogonal` | `OrthogonalPatchCompositionReceipt.edit_steps` | Yes | Yes |
| Flagship study | `edit_graph_injected_defect.json` and `edit_graph_repair.json` in the bundle | Yes, from the clean model and the bundle, using `flagship.replay_flagship_edit_graphs` | Not run |

With duplicate `target_layers`, each pass restarts from the base weight, so the last edit wins. Steps are keyed by layer to match, and a test covers it.

The flagship transcript leaves out steps and weights, because tensors can't be canonicalized. The bundle publishes the graphs instead. The defect is a graph too, so the bundle doesn't need a copy of the defective checkpoint. The public key in the bundle's receipt only shows the bundle is self-consistent, so a reader should get the key from a trusted source.

## Device and precision

`execute_transactional_gpu_edit_and_eval` states what can replay it, in `replay_precision`, `replay_device` and `replay_refused_reason`. Tested on CPU and on an RTX 5090 (`cuda:0`).

| Edit | Ran on | Replay |
|---|---|---|
| `add_delta`, f64 | CPU, `cuda:0` | Replayed: f64 on the Rust CPU executor, same digest |
| `low_rank_update`, f64 | CPU, `cuda:0` | Refused in the test's run: PyTorch's output isn't bitwise equal to the Rust result on either device, so no plan is issued |
| `subspace_projection`, `scale`, f64 | Not tested | Refused: replay isn't wired for these |
| Any operation, f32, f16 or bf16 | f32 and bf16 tested on CPU, `cuda:0` | Refused: the graph replays only f64 |

The refused GPU edits still have the existing tolerance-based parity check against Rust. That check isn't a bitwise replay.

## Methods a plan can't express

These are confirmed by tests. None of them has a fake plan.

| Method | Why |
|---|---|
| `execute_mlp_width_surgery` | It changes tensor shapes. A graph step keeps its tensor's shape, so it can't reach the narrower model. |
| `execute_attention_head_surgery` | It changes tensor shapes and head counts. A graph has no shape or config operand. |
| `execute_layer_reorganization` | It works on a bare list of tensors and returns names such as `model.layers.0.weight`, which aren't tensors in the model. Removing a layer also changes `num_hidden_layers`. |
| `execute_quantized_q8_matrix_edit`, and the Q4 and FP8 edits built the same way | They accumulate in f32 and requantize to bytes. A graph has no precision or requantize operand, and the Q8 test shows an f64 replay lands on different weights. Storing the result with `replace_slice` would only record it, not replay the method. The Q4 and FP8 edits are excluded by the same design, without their own test. |
| `apply_registered_structural_constructor` | It measures an identity on probes and returns a receipt. It produces no weights, so there is nothing to replay. |
| `quantize_weight_grouped` | It returns a lossy int8 representation with scales, not an edit to a model's f64 weights. |

Expressing these would need a format change, which is out of scope here.

## Memory

On a defective real Pythia-14M, `studio.repair` peaked at 936,136 KB of resident memory, against 929,932 KB before this work (one run each). The plan's candidate replaces the model copy that `repair` and the flagship study already made, and verification doesn't copy a model.
