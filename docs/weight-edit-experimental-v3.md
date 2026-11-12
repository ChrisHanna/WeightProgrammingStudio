# Experimental five-fix weight editing

Status: **CPU software prototype only**, implemented under separate user approval.
This does not authorize model experiments, live jobs, checkpoint loading, tuning,
promotion, or holdout access. CPU correctness is not task gain, preservation, or
live qualification. The existing real method remains unsolved: the comparable
legacy profile had zero exact gain and failed preservation; invalid old sweeps
are not acceptance evidence. The paused proposal/history and v2 policy are unchanged.

## Entry points and explicit configuration

`edit_method="value_scaled_retained_edit_v1"` selects a separate recipe through
`validate_recipe` and the single-site `calculate_edit` path. Legacy recipes,
`LegacyKeyValueEdit`, and legacy gradient normalization are unchanged.
`train_payload` refuses this method **before checkpoint loading**.
Multi-site composition is explicitly unsupported.

Required recipe fields:

- `method="weight_edit"`, `edit_method` above, `experimental_opt_in=true`.
- Explicit `max_sequence_tokens`, `max_training_tokens`, `max_keys`.
  Budgets are checked before any forward. Training tokens count every actual
  forward pass, including the protected-prompt recapture after covariance
  accumulation; keys count every captured position, including repeated passes.
  `capture_budget` records stage-level and total forward counts, forward tokens,
  and captured positions. Protected full responses are forwarded only when
  they contribute value-prediction keys not already in their prompt.
- Nonempty `retention_interactions`, `retention_source="public_train"`.
- Nonempty `protected_training_examples`, `protected_training_source="public_train"`.
  Each protected example carries an approved assistant-response `interaction`
  and explicit caller assertion `already_passing=true`. This assertion is not
  independently verified by the tensor prototype.
- `value_fields`: explicit unique flat JSON field names, or `null` when examples
  provide `value_spans=[[start,end,group],...]` in approved response character offsets.
  Task-provided spans are semantic authority, not inferred labels.
- `experimental_settings`: all of `strength`, `ridge_alpha`, `retention_beta`,
  `protection_epsilon`, `protection_zero_absolute`, `rank_rtol`, `rank_atol`,
  `solver_atol`, `solver_max_iterations`. There are **no calibration defaults**.
  These settings are fixture inputs, not approved production parameters.

No v2 thresholds, `target_step`, specialist retention policy, or preservation
policy version is reinterpreted. A future live policy and custody/qualification
integration need separate approval.

## Objective and five fixes

1. The value helper supports exact compact sorted flat JSON scalar values.
   Property names, quotation marks, other syntax, and EOS are unlabelled.
   Nested/escaped selected values are explicitly refused. Task-provided spans
   support other approved response semantics. Offsets must cover every character
   in every declared nonempty value span; mixed-boundary tokens are refused.
   Labels are at token index `t`; writer keys/gradients are captured at `t-1`.
   Capture verifies selected positions, labels, and input IDs. It computes
   next-token cross entropy itself, rather than trusting a model's loss shift.
   Loss is summed across all selected demonstration tokens and divided by their
   global count, including when demonstrations are captured separately.
2. Raw writer-output loss gradients are retained, including zero rows.
   No row/column normalization or active-key filtering is performed.
3. API keys are `[key_count,input_dimension]`; canonical `K=keys.T`.
   `ridge=alpha*mean(diag(K K.T))`, with the edit Gram a **sum**, not a
   normalized covariance or `K.T K`. Nonpositive/nonfinite scales are refused.
4. `CovarianceStatistics` accumulates every rendered nonpadding prompt token
   (capture is unpadded, one prompt at a time) from independent retention prompts
   and protected passing prompts. `C=input_gram/count` is a token-weighted,
   uncentered second moment; no per-prompt averaging and no last-token sampling.
   Each protected prompt contributes once to this covariance. Its prompt keys
   **also** appear in the separately declared protection constraint, together
   with its approved value-prediction keys. Those response keys are not added to
   covariance. Prediction keys already in that example's prompt are not duplicated
   in the constraint. This is intentional covariance-plus-constraint protection, not
   an extra implicit covariance multiplier. Supplied duplicate prompts would
   be repeated token observations; callers must enforce dataset authority.
5. `Y=-strength*h_rms/g_rms*G.T`, where both RMS values average every coordinate
   at the same value-prediction positions. `h_rms` is **writer-output**, not
   residual-stream, RMS. One global factor preserves gradient norm ratios.
   Zero/nonfinite global activation/gradient scales are refused.

The solver minimizes
`||D K-Y||_F² + beta tr(D C D.T) + ridge ||D||_F²`
subject to `||D P||_F <= epsilon ||W P||_F`.
If the reference is exactly zero, the required explicit
`protection_zero_absolute` is the absolute bound; no hidden relative fallback.
`W` is the canonical **stored** base tensor, matching application semantics.

For zero bound, SVD rank uses `max(rank_atol,rank_rtol*max(singular_values))`;
the objective is solved inside the resulting nullspace. The actual residual
against **all original protected keys** is then checked, so discarded directions
cannot silently remove protection. Full-rank exact protection can force a zero
update, explicitly recorded as `zero_effective_bf16_update`, never task success.
For positive bounds, the constrained quadratic is solved through a bounded
Lagrange-multiplier bracket/bisection of `A+lambda P P.T`; it is not a post-hoc
projection of an unrelated solution. Budget exhaustion or numerical infeasibility
is a refusal. `solver_atol` is the explicit absolute residual allowance, including
for rank truncation and casting, not a task preservation threshold.

## Diagnostics, casting, and remaining gaps

Diagnostics report gradient quantiles/zeros, group counts/NLL, activation,
gradient and target RMS, per-key target/write fractions, actual ridge and
conditioning, protected rank/reference/bound, token counts/reductions,
tensor fit/retention/protection residuals, realized activation ratios, and the
effective bf16 delta norm. An all-token arm without the extra constraint is not
exposed by this protected-only prototype; no ablation runs are authorized.

Before returning, the solver checks both its tensor delta and the simulated
effective bf16 delta `(W+D).bfloat16()-W.bfloat16()` against the same constraint.
Failure is explicit; there is no success-shaped fallback. Focused CPU tests
serialize/reload these tensors in memory. This is not checkpoint reload or
fresh-server verification. Existing transactional application/rollback remains
the responsibility of the unchanged generic apply layer. A local linear bound
does not guarantee nonlinear behavior, exact output, format validity, or task
preservation; these remain unmeasured and live execution remains blocked.
