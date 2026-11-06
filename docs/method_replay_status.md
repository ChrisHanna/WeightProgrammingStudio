# Method replay status

Status of the milestone rule "Every method returns an `EditPlan`". Replayed means a test in `tests/test_method_replay.py` starts a fresh process that holds only the base weights, a signed edit graph and a public key, and reproduces the `tensor_state_digest` of the method's own candidate. The expected digest is computed from the method's weights without using the plan.

Each method carries the operation and operands it actually ran (`method_plans.MethodEditStep`) rather than using `EditPlan.adopt_weight`, because `w + (w' - w)` isn't guaranteed to equal `w'` in f64. The graph format is unchanged.

## Methods

| Method | Where the steps are | Fixture replay | Real Pythia-14M replay |
|---|---|---|---|
| Repair `attractor_projection` (`looping`) | `AttractorRepair.edit_steps`, `studio.repair()["edit_plan"]` | Yes | Yes, with the injected defect from `test_attractor_repair.py` |
| Repair `association_edit` (`parity`; `capitals` and `factual` share the path) | `studio.repair()["edit_plan"]` | Yes, `parity` only | Not run; ROME and MEMIT below cover the adapter |
| `run_autonomous_tool_loop`: `dampen_singular_components`, `project_out_activation_direction`, `project_out_output_direction`, `edit_associations` | `FrontierAgentSessionReport.submitted_edit_steps` | Yes, each with a scripted transcript; no live LLM | Not run |
| ROME | `EasyEditAdapterReceipt.edit_steps` | Yes | Yes |
| MEMIT, one layer and several | `EasyEditAdapterReceipt.edit_steps` | Yes | Yes, one layer |
| AlphaEdit, several layers | `EasyEditAdapterReceipt.edit_steps` | Yes | Not run |
| mergekit `task_arithmetic`, `dare` | `MergekitAdapterReceipt.edit_steps` | Yes | Not run |
| mergekit `ties` | `MergekitAdapterReceipt.edit_steps` | Yes | Yes |

With duplicate `target_layers`, each pass restarts from the base weight, so the last edit wins. Steps are keyed by layer to match, and a test covers it.

`studio.repair` raises if the plan doesn't bitwise reproduce the method's weights. The adapters and the agent loop only carry the steps: call `method_plans.verified_method_plan` to get the same check.

On a defective real Pythia-14M, `studio.repair` peaked at 934,512 KB of resident memory, against 929,932 KB before this change. The plan's candidate replaces the model copy `repair` already made, so it adds no full-model copy.

## Methods a plan can't express

These are confirmed by tests. None of them has a fake plan.

| Method | Why |
|---|---|
| `execute_mlp_width_surgery` | It changes tensor shapes. A graph step keeps its tensor's shape, so it can't reach the narrower model. |
| `execute_attention_head_surgery` | It changes tensor shapes and head counts. A graph has no shape or config operand. |
| `execute_layer_reorganization` | It works on a bare list of tensors and returns names such as `model.layers.0.weight`, which aren't tensors in the model. Removing a layer also changes `num_hidden_layers`. |
| `apply_registered_structural_constructor` | It measures an identity on probes and returns a receipt. It produces no weights, so there is nothing to replay. |
| `quantize_weight_grouped` | It returns a lossy int8 representation with scales, not an edit to a model's f64 weights. |

Expressing the three surgeries would need a format change, which is out of scope here.

## Not converted

These haven't been wired up or tested. Their edits are low-rank or derived from a converted method, but no replay has been shown.

| Call site | Note |
|---|---|
| `pythia.py:1520` Gym study, branch 3 | One `low_rank_update`. The report doesn't expose its factors. |
| `prd_adapters.py:1049` weight-reading-only control arm | One `low_rank_update`. It is a control and exposes no weights. |
| `flagship.py:773` flagship repair | Calls `repair_repetition_attractor`, so `edit_steps` exist, but the flagship artifacts don't include a graph. |
| `run_reasoning_loop_repair_experiment`, feedback arm | Reports the Gym study above, so `submitted_edit_steps` stays empty. `flagship.py:925` serializes this report, and tensors in the field would break that if it were filled. |
| `patch_ux.py` orthogonal patch composition | One `add_delta`. |
| `export_formats.py` quantized block edits, `gpu.py` backend parity | These edit in f32 or on other backends, so they aren't f64 model edits. |
