# Product Integration Roadmap

**Status date: 2026-11-09.** This is the product sequencing plan and the working mathematical contract for GymEdit. A listed capability is planned until its implementation, tests, and measured evidence are linked here. Passing fixtures alone does not qualify a real model or gym. Holdout remains sealed until each candidate is frozen.

## Phase 1: Make Edits Work

**Estimate: about four weeks.** Public validation is the selection split; compare against the unchanged base, never holdout. The final user-facing report has one results table, with task gain and preservation evidence in every candidate row.

1. Implement approved specialist gate v2: paired public-validation task improvement of at least four net cases, paired-bootstrap 95% mean-gain interval lower bound strictly above zero, and 100% format validity plus 100% task-scoped required-set success. Because validation selects candidates, intervals are exploratory and are not independent confirmation. The policy is frozen in `protocols/specialist-retention-v2.json`; the implementation still needs to be wired into the workflow.
2. Specialist edits use retention weight zero and may use stronger strengths and multiple layers. This must change the calculation, not merely suppress the preservation rejection. Generalist retains the frozen preservation-v2 gates.
3. Rebuild public training data with varied, train-only templates at validation-like difficulty. Re-baseline the unchanged base on new train and unchanged validation. Require at least ten base-failing approved training examples before editing; build target keys only from failures, and protect passing examples. Refuse the edit when the minimum is not met.
4. Use the optimized public search path: capture the selected layer activations/gradients once, compute each per-layer delta once, scale it across strengths, apply in memory, score, and restore. Screen by teacher-forced answer likelihood in batches; exact-match generation is limited to finalists. Use successive halving and checkpoint every candidate for resume. Finalists alone receive export, fresh reload, vLLM identity, and existing verification gates.
5. Diagnose strengths `1e-2`, `0.1`, `0.5`, and `1.0` over the middle layers. Report task exact match and expected-answer log-probability before/after on train and public validation. Stop and report if strength 1.0 on the best layer still has zero effective change; investigate site, token position, or application before advancing.
6. Quick improvements: spread edits across the best 3-5 layers and allow 10-50 bounded target-optimization steps.
7. Activation profiling in `python/neural_foundry/backends/weight_edit/profile.py`: per-layer/per-neuron activity on gym traffic versus general text; activation-times-gradient attribution contrasting failures with passes; and zero-site ablation against public gym score. Emit a ranked site list and layers-by-parts heatmap.
8. Implement and measure the GymEdit core solve and synthetic company-facts gym described below.
9. Score both the JSON-extraction and company-facts gyms: base, SFT, current edit, and GymEdit core solve, in specialist and generalist modes.

**Current evidence and implementation status:** the unchanged Qwen3 base scored 62/64 exact match on the old 64-case train split and 45/64 on public validation. The old `1e-3` single-site edit scored 62/64 on those same train examples and reduced mean expected-token log-probability from -1.07954 to -1.08178; it did not fit additional examples. Its public validation score was 44/64. This is measured lack of gain for that legacy recipe, not evidence against GymEdit. The earlier preservation-v2 result and failure remain in `docs/evidence/weight-edit-preservation-v2-qwen3.json`.

Batched inference, a 16-case likelihood-first screen, append-only fsynced candidate checkpoints/resume, a one-candidate profile command, and one-pass all-selected-site capture are implemented and covered by focused tests. The single-candidate profile has not yet been run. The earlier legacy serial per-layer sweep ended without writing a final report or per-candidate records; its completed candidate count and partial scores were lost and are unavailable. Do not treat it as evidence or restart that old code path. No holdout has been accessed.

The approved specialist gate is a selection rule, not broad safety qualification. The required set is task-scoped invoice prompt-injection robustness, and its limits must be described narrowly. The paired-bootstrap interval on validation-selected candidates is exploratory. Thresholds and required-set definitions cannot change without user approval and a new policy version.

## Phase 2: Proof Report

**Estimate: 1-2 weeks.**

1. Add a frontier comparison after the user selects and configures a provider.
2. Add gym-reward-derived target modes (best answer and reward-weighted answer), with the target distribution, quality bar, reward baseline, temperature, and sample budget frozen in the run request.
3. Produce the proof report: one-shot holdout scores, task and format/safety gates, preservation, cost, end-to-end speed profile, activation heatmap, lineage, and evidence classification. Freeze each candidate before its single holdout evaluation. A failed gate does not authorize another edit based on that holdout.

The frontier provider is not selected. No credential or live frontier call is assumed.

## Phase 3: Practical Editing

**Estimate: 2-3 weeks.**

1. Sequential edits with explicit dependencies, conflict detection, and correct-base rollback/replay.
2. Scale to 1,000+ edits with bounded storage, execution, and replay budgets.
3. Qualify more model families and revisions.
4. Qualify vLLM serving of LoRA, QLoRA, and weight-edit artifacts. A registry entry or fixture reload is not serving qualification.

## Phase 4: Quality

**Estimate: 1-2 weeks, in parallel.**

1. Resolve the existing backlog and review findings; obtain green Python and Cargo acceptance runs.
2. Add contract tests and CI GPU smoke tests with explicit skip/failure semantics.
3. Consolidate common validation and evidence checks in a shared module without loosening any gate.

## Phase 5: GRPO

**Estimate: 5-6 weeks.**

1. Integrate NeMo token-ID routing.
2. Integrate the reward guard.
3. Add NeMo reinforcement-learning GRPO and DPO execution.
4. Qualify AutoModel batch settings.

## Phase 6: More Gym Types

**Estimate: 6-8 weeks.**

1. Tool-use and multi-turn gyms.
2. Gymnasium and PettingZoo adapters.
3. Sandboxed environments with explicit authority and resource limits.
4. A game gym demonstration.

## Phase 7: Control-Plane Store

**Estimate: 2-3 weeks.**

1. Add `ControlPlaneStore` with SQLite locally and a Postgres product backend. Store jobs, job versions, runs, events, candidates, receipts, scorecards, decisions, custody claims, and artifact references.
2. Add the run state machine and event stream.
3. Add signing and verification contracts.

## Phase 8: Trust Services

**Estimate: 4-6 weeks.**

1. Custody service.
2. Standard scorecard.
3. Explicit approve/reject decisions and versioned gate catalog.
4. Docker placement, artifact store, and immutable execution images.

## Phase 9: API Endpoints

**Estimate: 2-3 weeks.**

1. HTTP API for jobs, runs, event streams, candidates, scorecards, changes, comparisons, approvals/rejections, backends, models, gym compilation, and audit.
2. Catalog and schemas.
3. Users, organizations, and permissions.
4. Make the CLI an API client.

## Phase 10: Dashboard and Scale

1. Dashboard over durable runs, events, scorecards, evidence, and decisions.
2. Kubernetes or Ray placement and a qualified vLLM pool.
3. Capacity reuse, pruning, and candidate merging.

---

# GymEdit Core Solve

This section defines the intended linear-map solve. It is a design contract; it does not claim the ROME/MEMIT methods, located attribution, optimized targets, or company-facts gym are implemented or qualified. The current legacy editor is a single-site teacher-forced-gradient update and is not this solve.

## Notation

At an eligible linear site, write

$$v = Wk + b, \qquad W \in \mathbb{R}^{d_{out}\times d_{in}}.$$

`k` is the input activation at the chosen decision token, and `v` is the site output. Orient tensors as `W[out, in]`; verify site and tensor identity from the loaded checkpoint. Rendering, token IDs, response positions, scorer, and train split are frozen and digest-bound.

## 1. Failure Selection

Evaluate the unchanged base on the public train split with the frozen scorer. For case $i$ with response $\hat y_i$ and reward $r_i$:

$$F = \{i:r_i<1\}, \qquad B = \{i:r_i=1\}.$$

Only failing cases may contribute target keys. Passing cases join the protected set. Refuse editing if $|F|<10$; do not silently lower the minimum. Report the full base train and validation baselines before editing.

## 2. Locate Sites

Use teacher-forced negative log-likelihood of the approved answer as the differentiable attribution surrogate:

$$\ell(x)=-\sum_t\log p_\theta(y_t\mid x,y_{<t}).$$

For site $s$ and case $x$, score the activation-gradient interaction and contrast failures against passes:

$$a_s(x)=\frac{\|h_s(x)\odot\nabla_{h_s}\ell(x)\|_1}{\dim(h_s)}, \qquad A_s=\operatorname{mean}_{x\in F}a_s(x)-\operatorname{mean}_{x\in B}a_s(x).$$

Rank only discovered, role-verified sites. Confirm leading candidates by zeroing the site under scoped hooks and measuring the public task score. Remove hooks on every exit path. Bind token positions, site map, intervention, scorer, and rankings in the profile artifact.

## 3. Keys

For failing case $i$ and approved wording $j$, capture the site's input at the decision token and its current output:

$$k_{ij}=\operatorname{input}_W(x_{ij}), \qquad v^0_{ij}=Wk_{ij}+b.$$

Stack keys as columns, $K=[k_{ij}]\in\mathbb{R}^{d_{in}\times n}$, and targets/residuals as columns. Average or weight paraphrase keys only under an explicit, frozen policy. Capture one activation/gradient pass for all eligible layers and reuse the records across strengths.

## 4. Optimized Targets

With the site output intervened on and model weights frozen, optimize a target $v_i^*$ over the approved failing-answer wordings:

$$v_i^*=\arg\min_v\left[-\sum_jw_j\log p_v(y_j\mid x_i)+\beta\,\mathrm{KL}(p_0(\cdot\mid c)\|p_v(\cdot\mid c))+\gamma\|v-v_i^0\|_2^2\right].$$

Run 20-50 bounded gradient steps. Enforce a frozen KL cap, keep the best feasible iterate, and restore the unedited model exactly after each target search. The KL scope and contexts are recorded; a local next-token KL bound is not a global behavior guarantee.

For reference-answer targets, $w_j=1$. For gym-reward targets, sample a frozen number of outputs, score them, compute advantages, discard outputs below a frozen quality bar, then normalize:

$$A_j=r(x,y_j)-\frac1G\sum_{k=1}^{G}r(x,y_k), \qquad w_j=\frac{\exp(A_j/\tau)}{\sum_{k\in Q}\exp(A_k/\tau)}.$$

Bind the reward scorer, samples, quality bar, baseline, temperature, and target digest. Holdout is never used for target generation or iterative selection.

Residuals are $R=[v_i^*-v_i^0]\in\mathbb{R}^{d_{out}\times n}$.

## 5. Covariance

Use the existing streaming capture for the uncentered second moment of inputs from a frozen general-text sample:

$$\widehat C=\frac1m\sum_j z_jz_j^\top, \qquad C=\mu\widehat C+\eta I, \quad \mu\ge0,\ \eta>0.$$

Record sample identity, renderer, token/position selection, count, accumulation dtype, regularization, and digest. The sample is separate from holdout and cannot be changed after candidate scoring.

## 6. Core Solve

Solve the stated objective:

$$\min_{\Delta W}\|\Delta W K-R\|_F^2+\operatorname{tr}(\Delta W C\Delta W^\top).$$

The solution is

$$\Delta W=RK^\top(KK^\top+C)^{-1}.$$

Never form an explicit inverse. Use a checked Cholesky factorization/solve, report conditioning and residual, and fail closed on nonfinite or ill-conditioned systems. One key is ROME-like; multiple keys form a batch solve. These names describe the mathematical family, not a claim of paper-level fidelity until reference tests pass.

## 7. Protection

Generalist mode protects public passing cases and eligible earlier-gym keys. Let $P$ contain protected input keys. Use an approximate null space by taking eigenvectors of $PP^\top$ with eigenvalues below a preregistered threshold $\epsilon$, as columns of $N$. Report the threshold, spectrum, retained rank, and residual:

$$\widetilde K=N^\top K, \qquad \widetilde C=N^\top C N,$$

$$\Delta W=R\widetilde K^\top(\widetilde K\widetilde K^\top+\widetilde C)^{-1}N^\top.$$

The intended local bound is $\Delta W P\approx0$, measured by $\|\Delta W P\|_F$ after casting. It is not a guarantee of nonlinear model behavior or unseen safety.

Specialist mode uses retention weight zero and protects only the frozen required set; the approved specialist task gate still requires a public-validation gain and the required set. Generalist mode additionally retains the unchanged preservation-v2 gates: exact-promoted-weight float32 drift, bf16 top-1/top-5, and mean KL. No final gate is changed here.

## 8. Multiple Layers

For $q$ selected layers around the attribution peak, process in forward order. At layer $l$, divide the current remaining residual by the number of unprocessed layers, solve that layer, apply in memory, then recapture the next-layer keys under the updated upstream weights:

$$R_l=\frac{R_{remaining,l}}{q-l+1}, \qquad \Delta W_l=R_lK_l^\top(K_lK_l^\top+C_l)^{-1}.$$

Record each per-layer site, residual, covariance, solve diagnostics, and actual cast change. The same candidate's train/validation task gain and preservation accompany every strength/layer result. Export and fresh serving verification are finalist-only.

## 9. Strength

For fixed optimized residual targets, the linear solve is homogeneous. If $R(\alpha)=\alpha R$:

$$\Delta W(\alpha)=\alpha\Delta W(1).$$

Compute once per layer and scale in memory for strengths `1e-2`, `0.1`, `0.5`, and `1.0`. Restore the exact original tensor between candidates. Strength selection uses public validation only and is exploratory; no holdout feedback enters the search.

## 10. Apply and Verify

Apply once in the declared storage dtype:

$$W_{new}=\operatorname{cast}_{bf16}(\operatorname{float32}(W)+\operatorname{float32}(\Delta W)).$$

Check every value is finite, verify the expected changed-tensor roster, and verify all other tensors are byte-identical. Profile memory/time. Export only finalists; then fresh-load, verify state/config/tokenizer/renderer, run vLLM identity/portability and existing verification gates. Restore the in-memory base after every diagnostic candidate. Do not publish a sweep candidate.

## 11. Specialist Gate V2

On the same public validation cases, use paired exact-match differences $d_i=r_i(edited)-r_i(base)$:

$$\operatorname{net\ gain}=\sum_i d_i\ge4, \qquad \operatorname{CI}=\text{paired-bootstrap 95\% percentile interval of }\operatorname{mean}(d).$$

Accept specialist mode only if net gain is at least four cases, the lower confidence bound is strictly above zero, format validity is 100%, and the frozen task-scoped required set is 100%. The gate is approved as policy v2; a CI after candidate selection on the same validation panel is exploratory, not independent confirmation. Generalist also must pass all unchanged preservation-v2 gates. Holdout is one-shot after candidate freeze.

## 12. Iterate

Re-run the public train gym with edited weights, rebuild failing set $F$ from remaining failures, and stop when there is no public gain, the round budget is exhausted, or $|F|<10$. Freeze the candidate before one holdout evaluation. Never use holdout responses, scores, or failure patterns to seed another edit.

---

## Company-Facts Gym Contract

Build a synthetic product catalogue gym with facts such as product-to-price and SKU-to-category. Generate separate train, public validation, and holdout facts and wordings; holdout facts and wording templates are unseen during training, target optimization, profile/site selection, and strength search. The base must measurably lack at least ten train facts before editing. Use the same frozen scorer, controls, artifact lineage, task/format/safety gates, and final report schema as the JSON gym. Do not claim a ROME/MEMIT win until both gyms have measured base/SFT/current-edit/core-solve results.

## Execution Order and Evidence Rules

Each step is measured, tested, and committed separately. Every candidate row in the final report contains task gain and preservation together, with train fit and expected-answer likelihood where applicable. Keep raw prompts, responses, token IDs, target/covariance inputs, timings, failures, checkpoints, and digests for every run. Failed or interrupted runs remain history; do not overwrite or reclassify them. No holdout access until each finalist is frozen. Release, generalist, broad safety, and independent-custody claims remain separately gated.
