# Protocol, fixed before any sweep or search ran

This commit precedes all results in this folder. The code carries the same values.

## Multi-site families (`retention_multisite.py`)

Closed-form families: `single`, `parallel`, `joint`, `sequential`, `compensated`, `readout`, `readout_plus`. Grids are
in `grids()`. Selection repeats `retention_research`: require 100% development and extra-prefix accuracy at or above
the reference's floor, then maximise the mean of practice and tuning retention, then prefer a smaller edit. Validation
is reported and not used to select.

## Cumulative search (`cumulative_edit.py`)

- **Objective.** `J = task + retention - 0.1 * KL - c * changed_parameters`.
  `task` is accuracy on development and extra-prefix items (30). `retention` and `KL` use the accept guard.
- **Cost.** Primary `c = 1e-7` per changed parameter, so one 65,536-parameter tensor costs 0.0066. Sensitivity arms,
  declared now and not used to choose: `c = 0` and `c = 1e-6`.
- **Acceptance and stopping.** Take the candidate with the highest `J` in the round. Accept it if `J` rises by at least
  `0.005`. Stop at the first round where no candidate does.
- **Candidates each round.** Every layer's `mlp.dense_4h_to_h`, `attention.dense` and `mlp.dense_h_to_4h`.
  `task` moves at 7 strengths and 2 protection weights. From round 2, `repair` moves cancel drift on the solve guard.
- **Data separation.** The corpus's fit part is split by template group into a solve guard (seen by solves) and an
  accept guard (decides steps). Validation, three report-only prefixes, practice and tuning retention decide nothing
  and are logged every round, so overfitting to the accept data shows.
- **No scope cap.** There is no tensor or parameter limit. The hard safety limit is 40 rounds; the output says if it binds.
- **Known exclusions, stated in advance.** `embed_in` and `embed_out` exceed the executor's 16 MiB source bound at f64,
  so they cannot be `EditPlan` steps. `attention.query_key_value` has no closed-form step here. Norms and biases are not
  matrices, which `EditPlan` requires. So "any tensor" is not fully met. The output lists these exclusions.
- **Replay.** Each accepted step is `EditPlan.low_rank`. The final graph must replay to the same plan and state digests.
- **Comparators at equal task accuracy.** The best single-tensor sweep entry at the search's final development
  accuracy. LoRA and fine-tuning restricted to the tensors the search picked, using the published learning rate and
  length, guard weights 1 and 10, the fit corpus as guard, and the same public selection at that development bar.

## Seal

- Evaluated once per method after the freeze commit, each logged in `ledger.json`: 120 new retention prompts (60 stems,
  60 other genres) and 12 new task items. The old holdout is scored too, never for selection, with its known shortcut
  (AND->False, OR->True) stated next to any number. The original and earlier fresh sealed sets are not evaluated.
- Controls: no edit, and random edits of matching per-site norm and rank.
- Tests: two-sided exact McNemar on the new retention set against the stated comparator, a paired bootstrap interval
  for the difference, and Holm across every listed comparison. The primary multi-site family is named in the freeze.
- The author of the new prompts had read the older sets, so they may share a style. They were written before any
  result from this work and checked for exact and word-trigram overlap with all public and older sets.
