# Weight-Edit V3 Research Proposal

**Status: proposal only; not approved policy, protocol, implementation, or experiment authorization.** This note is isolated on `wip/weight-edit-paused`. Finish the approved robustness-v2 comparisons and access accounting first. This proposal does not change `protocols/specialist-retention-v2.json`, v2 thresholds, the frozen recipe, or any result. No v3 code, CPU experiment, GPU job, training-data change, or holdout access is authorized.

## Motivation And Evidence

The current production worker validates `edit_method="legacy_key_value_v1"`, selects one midpoint MLP writer by default, forms target shifts from teacher-forced token-loss output gradients, and applies the legacy Gram solve. The generic apply layer can accept multiple named deltas, and a multi-site calculator exists, but that route is not the worker's selected recipe path.

The approved specialist-retention v2 policy is frozen at zero retention weight. It selects among 14 middle MLP sites and four strengths, caps screening at 56 candidates and exact candidates at 12, requires at least four net public-validation cases with the paired-bootstrap lower bound above zero, and retains 100% format and task-scoped required-set gates. Preservation is report-only for specialist mode. These are v2 facts, not v3 defaults or approval.

The recent robustness-v2 SFT/LoRA results are not weight-edit evidence. The 192-case public cards count base 52, old SFT 49, new SFT 72, and new LoRA 86. New SFT gains 36 net clean cases; new LoRA gains 50, not 36. Each loses 16 net injection cases on `scan_footer_conflict`. Old SFT already has those 16 regressions. In the inspected failures, the body holds the correct values and SFT copies the untrusted alternate footer record; an "ignore footer" explanation is contradicted by these examples. This may motivate a future train-only intervention, but it does not demonstrate that weight editing solves it. Data augmentation and training duration changed together, so this is not a data-only causal ablation.

The existing [root-cause audit and deferred method decisions](integration-roadmap.md#deferred-method-decisions) identify five foundation problems, not just missing protected keys. On the frozen 155-failure dataset, 1,550 of 2,625 edit positions (59.05%) are JSON syntax; 920 are value contents and 155 are EOS. Retention captures only one final prompt-token key per interaction. These are measured implementation deficiencies, not proof of each one's causal effect on task failures. V3 must address all five explicitly.

## Hypothesis

A value-focused, scale-aware single-site edit with full-token retention will fit transaction values more effectively and break fewer already-passing behaviors than the legacy recipe at a comparable effective edit size. Protected passing keys are part of retention, not a substitute for the other four fixes. The hypothesis fails if train-only target fit does not improve, preservation fails after bf16 casting/reload, or the approved public evaluation does not improve over the same-site legacy baseline.

## Proposed Method

Introduce a separately named method, provisionally `value_scaled_retained_edit_v1`, in a future recipe version. Keep the first experiment to one checkpoint-verified MLP writer. The full method must include the following five changes; do not carry the existing target-gradient recipe forward unchanged.

### 1. Value-token loss only

Use teacher-forced loss only on customer, product, and quantity value contents, excluding JSON syntax, property names, punctuation, and EOS. Build the mask from the frozen tokenizer and rendered answer spans, and verify causal next-token alignment. The frozen audit has no mixed-boundary tokens; a future tokenizer/dataset with such tokens must fail preflight for an explicit boundary policy rather than silently including syntax or dropping value coverage. Reject empty value masks. Record group counts and losses; keep exact JSON/format scoring unchanged. This is a new edit objective, not a change to the scorer or demonstration answers.

### 2. No per-key gradient normalization

Retain the magnitudes of value-loss gradients `g_i`; do not replace each gradient by `g_i / ||g_i||`. This preserves relative signal across keys instead of giving weak syntax/value directions equal unit weight. Keep the loss reduction explicit and frozen, report raw gradient quantiles, and reject nonfinite gradients. Zero gradients remain zero and are counted. The single global target calibration in item 5 must not reintroduce per-key normalization.

### 3. Ridge scaled to the observed Gram diagonal

With column-stacked edit keys `K`, use `ridge = alpha * mean(diag(K K^T))`, where `alpha` is dimensionless and selected using train-only evidence under an approved rule. Do not reuse the absolute `1e-3`. The edit-key Gram, its sample weighting, and reduction are explicit; do not ambiguously substitute `K^T K` or a separately normalized covariance. Reject nonpositive/nonfinite scale rather than silently falling back to the old ridge. Report the diagonal scale, alpha, actual ridge, spectrum/conditioning, and target-fit residual.

### 4. All-token covariance retention, including protected keys

Build retention statistics using the existing `CovarianceStatistics` over **all nonpadding prompt-token activations**, not one final key per prompt, from approved independent public-train retention prompts and already-passing train examples. Use explicit token-weighted second-moment statistics, recording token counts and reduction. Do not mix holdout or public selection cases into retention. Proposed objective:

`min_Delta ||Delta K - Y||_F^2 + beta * tr(Delta C_ret Delta^T) + ridge * ||Delta||_F^2`

`C_ret` is the recorded full-token second moment. Protected passing examples also supply `P`, covering their prompt-token keys and teacher-forced approved value-token prediction positions. Propose a separate constraint `||Delta P||_F <= epsilon * ||W P||_F`, with a separately approved absolute policy if `W P` is zero. This preserves the local response on passing examples while broader covariance discourages drift elsewhere. Rank-revealing factorization may enforce the constraint, but rank cutoffs, feasibility behavior, epsilon, beta, and fallback policy require approval; an infeasible constraint is a reported refusal, not permission to drop protection.

Protected keys fit **inside this retention fix**. Their contribution to covariance and any extra constraint weight must be recorded so it is not accidental double weighting. Report rank, conditioning, target residual, retention penalty, and protected-response residual before and after bf16 conversion/reload. A local linear constraint cannot guarantee unchanged exact outputs or nonlinear behavior: measure broken passing examples and the independent preservation panel as well.

### 5. Activation-scaled target displacement

Construct value targets with one global scale, for example `Y = -s * h_rms / g_rms * G`, where `G` contains the unnormalized value-loss gradients, `h_rms` is the RMS write-output activation at those train value positions, and `g_rms` is the RMS gradient over the same coordinates. Thus the intended aggregate target RMS is `s * h_rms` and relative per-key gradient magnitudes remain intact. Define both RMS reductions precisely in the approved recipe; do not normalize individual columns. Reject nonfinite or zero global scales rather than inventing a target. Select dimensionless `s` on train only. Report target/activation RMS, per-key shift-fraction distribution, realized `Delta K`/activation ratio, and effective bf16 update size. Write-output activations are not mislabeled as full residual-stream norms.

This remains a gradient-surrogate weight edit, not ROME, MEMIT, or a gradient-free method. Acceptance uses the unchanged task scorer and separately approved policy, not successful target fit alone.

Defer coordinated multi-site composition until the single-site experiment is understood. The existing per-site calculator captures multiple sites but does not establish that independently calculated updates compose correctly through a nonlinear model.

## Proposed Research Sequence

1. Finish approved robustness-v2 reporting first: reconcile scenario-level counts, old-model comparisons and capped frontier, preserve the footer diagnosis, and record the overlap-only out-of-custody holdout read honestly. Required-set gains remain provisional until audit thresholds/custody are accepted.
2. Before implementation, approve the complete five-fix specification, boundary policy, gradient/loss reductions, covariance weighting, protected-key constraint, train-only calibration procedure, budgets, and versioned policy. No numerical hyperparameters or pass thresholds are approved in this note.
3. After separate implementation approval, CPU tests should cover value-only causal masking including mixed boundaries; retained relative gradient magnitudes; rescaling behavior of ridge and targets; all-token rather than last-token retention; rank-deficient protected keys, infeasible constraints, deterministic solving, bf16 effective change, and transactional rollback. Compare against `legacy_key_value_v1` on matched captured inputs.
4. After separate run approval, freeze checkpoint/data/split digests, site, train-only settings, candidate roster, budget, receipt schema, and acceptance policy before any model job. Propose a bounded ablation ladder: legacy; value loss; plus raw gradients; plus scaled ridge; plus all-token retention/protection; plus activation-scaled targets. Include an all-token-retention arm without the protected-key constraint to isolate that addition. These dependent ablations are order-sensitive, not independent causal estimates; approve the exact roster and cap before execution.
5. Compare on the same pinned base and approved public evaluation. Report task/required exact match, format validity, paired fixed/broken counts, passing-train retention, per-template results, all five scale/coverage diagnostics, cast/reload residuals, preservation, loaded-model identity, and runtime/cost receipts. No export or promotion based on train fit alone.
6. Keep validation exposure explicit. Any validation-selected search interval is exploratory. Independent confirmation requires a separately approved, newly frozen evaluation design. Prior synthetic holdout had an overlap-only read outside custody; do not call it unaccessed. No holdout prompts or labels may guide tuning or iteration, and future overlap checks belong inside independently controlled custody with aggregate-only receipts.

## Policy And Change Control

- `specialist-retention-v2.json` remains byte-for-byte unchanged and authoritative for v2 only.
- Do not reinterpret a v2 recipe as v3 or add v3 fields to the v2 validator.
- This proposal does not activate a v3 task gate, required-set threshold, preservation gate, site budget, or solver budget. All such numbers and any new policy version require separate user approval before implementation or execution.
- No candidate may be promoted, released, or described as broadly safe from this research. The current specialist task-scoped gate remains distinct from generalist qualification.

## Decision Needed Before Implementation

Approve or reject the complete five-fix foundation, with protected keys under item 4, and the proposed single-site ablation design. Approve overlap thresholds and a custody contract separately from the numerical method. Implementation and execution each require explicit authorization after the v2 closure report. Until then, this remains a proposal; no code or model behavior changes.