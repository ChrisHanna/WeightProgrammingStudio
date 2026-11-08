# Evidence-Guided Edit Methods

This is an implementation contract and roadmap, not a list of available methods. No advanced method is registered yet. Complete generic-editor application/export, old-engine equivalence, and compatibility qualification before migrating existing edit math or advertising support.

## First Milestone

1. `locate.py`: gradient attribution first, then matched-case causal tracing through scoped hooks on the real model. Record the ranked sites, rendered positions, clean/corrupted input identities, objective, intervention, measured recovery, and ranking digest. Unknown roles and routers are not automatically eligible edit sites.
2. `closed_form.py`: covariance-weighted single-site key/value updates, then coordinated multi-layer batch updates. Use verified float32/float64 working weights and streamed covariance. Explicitly record regularization, covariance convention, target optimization, and allocation across layers. ROME-style or MEMIT-style does not imply equivalence to a reference implementation.
3. `protect.py`: constrain the proposed delta against frozen preservation keys, permitted layers/tensors, and change-norm budgets. Check the mathematical residual, the cast-back delta, and actual model behavior on the preservation panel before accepting any candidate.

The first useful gym bridge is a frozen edit request plus a bounded preservation-set builder. Failure-pattern grouping and the iterative goal-search controller follow; do not wait for every later method to test this milestone.

## Method Boundary

Methods calculate named tensor changes. They do not load models, implement forward passes, write checkpoints, or claim release authority. Model/runtime details stay in verified loading, site discovery, capture, and application. Every method declares its supported site roles; portability across models is conditional on those roles, runtime compatibility, and verified tensor identity.

The compute contract is a site, canonical `W[out,in]`, `keys[n,in]`, `targets[n,out]`, frozen method settings, and optional preservation statistics, producing a dense delta or explicit low-rank factors. Tensor orientation is normalized outside method math. Computation is float32 or float64 under the numerical policy; application casts back once and verifies finite values and effective changes. Signed native operations remain the application authority where required by the existing transaction contract.

Locate and target optimization may call the real runtime through the capture layer. Every temporary intervention is removed in `finally`, including failure and cancellation. Causal-tracing corruption and restoration positions are bound to the same rendering policy used in capture and inference, not raw-string offsets.

## Protection Claims

For preservation keys `K`, a null-space projection can bound `delta_W @ K` at one linear site. This is not a guarantee about every output of the nonlinear model, unseen inputs, or global safety. Both the local algebraic residual and end-to-end behavioral drift must be measured with stated tolerances after casting, application, and fresh reload.

Passing cases do not make protection free: capture, rank estimation, numerical conditioning, projection, and behavioral checks have memory and execution costs. Preservation panels have size/token budgets and frozen digests. Adding a prior gym changes the panel version and therefore the edit request and receipt.

If validation prompts influence edit constraints or target selection, that is development exposure. Do not describe the same panel as untouched validation. Prefer a separate public preservation panel, and keep a fresh validation panel for selection evidence.

## Scorer Objective

Bind the scorer's implementation digest, version, configuration, and output schema into each request. Exact JSON match is nondifferentiable; a gradient-based target optimizer needs a declared surrogate, such as token loss, or an explicitly budgeted black-box scorer objective. Record the surrogate and its digest. Acceptance still uses the real frozen scorer, not a substituted training loss.

The existing synthetic scorer reports aggregate exact match only. Per-field diagnostics and failure grouping are not implemented. Do not infer field-level evidence from that aggregate or silently change a frozen scorer identity to add it.

## Gym Bridge

The proposed `edit_request.py` binds the job, base/candidate identity, split manifest, scorer/verifier identities, projection/renderer, scored attempt roster, approved expected outputs, eligible sites, method budget, and preservation-set digest. Reject incomplete attempts, unauthenticated caller-provided evidence, stale artifacts, and holdout materialization.

The preservation builder selects frozen public passing cases and approved prior-gym cases, deduplicates by stable task identity, caps cases and tokens, and records provenance, selection policy, exclusions, and digest. It does not receive sealed holdout cases.

`gym_failures.py` groups only explicit, versioned diagnostics from terminal scored attempts. Missing diagnostics are unavailable, not guessed. Incorrect but complete model outputs are distinguishable from infrastructure failure and incomplete execution.

The controller runs gym, locate, calculate, protect, apply, reload, and public re-evaluation through goal search. Stop on a development target, no improvement, or an edit/exposure budget. Freeze the selected candidate before one-shot independent holdout scoring. Holdout target attainment is not an iterative stopping rule; a holdout result cannot seed another edit round without a newly authorized evaluation/exposure contract.

Different validation templates provide sampled generalization evidence, not proof that memorization or regressions are impossible.

## Receipt

Each accepted or rejected edit records:

- Base, input-candidate, output-candidate, checkpoint, site-map, edit, and renderer digests.
- Frozen job, split, scorer/verifier, attempt roster, expected-output, and edit-request identities.
- Located sites, ranking objective, intervention evidence, and ranked-site digest.
- Method identity/settings, computation dtype, deterministic policy, surrogate identity, and costs.
- Preservation-set identity, size/selection budgets, basis rank, conditioning, and projection tolerances.
- Per-tensor requested and effective change norms, tied-storage permission/aliases, and permitted-layer checks.
- Local preservation residual and measured end-to-end score/logit drift after fresh reload.
- Fixed failures, newly broken public cases, conflicts, edit count, and verification status.

Incomplete verification is never an accepted score. Receipts describe measured scope; they do not promote a candidate or attest served weights on their own.

## Later Methods

1. `sequential.py`: ordered, digest-bound history against the current edited model; earlier edits join the preservation contract. Conflicts combine key-space overlap with incompatible targets, not overlap alone. Rollback replays/recalculates later edits from the correct base instead of merely subtracting one delta.
2. `merge.py`: task vectors, then explicitly tested TIES/DARE-style operations. Rebase requires compatible tensor identity/orientation, tokenizer/runtime checks, and fresh task/preservation/serving tests; matching names alone is insufficient.
3. `steer.py`: portable projection or bias changes derived from bounded behavior directions. Refusal/safety preservation is a required default, never a mechanism for bypassing safeguards. No runtime steering hook is exported.
4. Restricted training: migrate the selected-tensor fallback only after equivalence tests. Apply the same site permissions, preservation constraints, numerical policy, verification, and receipt rules.

Goal search selects methods/settings using common public scorecards. It does not select on sealed holdout feedback. Dashboard presentation must distinguish algebraic bounds, sampled behavioral checks, and unverified guarantees.

## Estimates And Gates

Planning estimates remain roughly six weeks for locate/closed-form/protection, eleven to twelve weeks for the full method sequence, and about four further weeks for the complete gym bridge, after the generic base is robust. These are estimates, not observed completion dates. The architecture sweep, real-checkpoint exact-role tests, old-engine equivalence, plain export/reload, and vLLM qualification are prerequisite evidence gates, not shortcuts replaced by this roadmap.