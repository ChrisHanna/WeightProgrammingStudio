# Pre-registered design: where do direct edits beat gradient training?

**Status: design only. This confirmatory study has not been run.** Its sample sizes and thresholds are fixed here. The
data in `RESULTS.md` is exploratory, informed the design, and may not be counted as evidence in the study.

## Question

For a bounded behaviour change to an existing model, on which measures does a closed-form direct edit do better than
LoRA or fine-tuning *when all three reach equal task accuracy*? Retention is not expected to be one of them, and the
study is designed so that a loss there is reported as plainly as a win elsewhere.

## Arms

| Arm | Definition |
|---|---|
| Edit | One closed-form low-rank `EditPlan` on one tensor, from a recipe frozen before the study |
| Cumulative | The `cumulative_edit` search, with its objective, cost and stopping rule unchanged |
| LoRA | Rank-limited adapters, recipe tuned on public splits only |
| Fine-tuning | All parameters, recipe tuned on public splits only |
| Restricted | LoRA and fine-tuning on only the tensors Cumulative picked for that task |
| Controls | No edit; random edit of matching norm and rank per tensor |

All arms get the same task items and the same guard prompts. Tuning budget for each is fixed at **60 configurations**,
and all configurations are logged, so that no arm is favoured by having been searched longer.

## Units and sample sizes

- **Tasks:** 24 tasks, from at least 6 skill kinds. Each has at least 12 development and 12 held-out items with new
  operands and prefixes, and must have **no single-feature shortcut**. A shortcut check script runs before the
  freeze: a task fails if any rule on one token (such as AND->False, OR->True) gets over 75% of its held-out items.
- **Models:** Pythia-14M, Pythia-31M and one model family other than GPT-NeoX. The third is required for H1-H4 to be
  generalised beyond this architecture; without it, the claim is limited to Pythia.
- **Seeds:** 3 per stochastic arm per task.
- **Retention set:** 600 new prompts, drawn by someone other than the corpus author, or sampled from a public text
  corpus by a script with a committed seed, stratified by genre.
- **Power for H5 (retention):** the unit is the task, as prompts inside a task are not independent. With 24 tasks per
  model and a paired-difference standard deviation assumed at 15 points, a one-sided non-inferiority test at alpha
  0.0083 (Holm's strictest step, 0.05/6) has about 80% power to establish non-inferiority when the true difference is
  0, against a margin of 10 points: n = ((2.39 + 0.84) x 15 / 10)^2 = about 23.5, so 24 just suffices. The 15-point
  standard deviation is a guess from a single task, so a blinded check at 8 tasks may raise the task count to no more
  than 40, and the study reports that it did.

## Equal task accuracy

Methods are compared at a **bar of 90%** task accuracy on public validation items, per task. For each arm, the recipe
is tuned to meet the bar on public items and then evaluated once on held-out items, whose accuracy is also reported. Any arm-task pair that cannot meet the
bar counts as a **failure to reach the bar**, is reported as a rate (H6), and is not silently dropped. Retention
comparisons use pairs where both arms met the bar, and a sensitivity analysis imputes 0 retention for failures.

## Hypotheses and thresholds

Each is one-sided and paired by task and model. Holm correction applies across H1-H6 at family alpha 0.05.

| ID | Hypothesis: Edit against each of LoRA and fine-tuning | Metric | Support requires |
|---|---|---|---|
| H1 | Fewer parameters changed | Coordinates that differ bitwise from source | Median ratio at most 1/4 vs LoRA, 1/20 vs fine-tuning, CI excludes 1 |
| H2 | Lower wall time to a candidate | Solve plus executor, against training; no sharing of the CPU, 5 repetitions, fixed thread count | Median ratio at most 1/5, CI excludes 1 |
| H3 | Lower peak memory above the loaded model | VmHWM minus RSS after load, in a fresh process | Median ratio at most 1/2, CI excludes 1 |
| H4 | Smaller reversal artefact | Bytes needed to restore source bits, verified by digest, for merged weights | Median ratio at most 1/10 vs fine-tuning; vs LoRA no claim expected, tested two-sided |
| H5 | Retention non-inferior at equal task accuracy | Top-1 retention, then KL | Lower CI bound of Edit minus comparator above -10 points |
| H6 | Reaches the bar as often | Share of task-model pairs reaching 90% | Lower CI bound above -15 points |

**Pre-declared expected losses.** H5 and H6 may fail. If they do, the conclusion is that direct edits are a cheaper
tool with worse behaviour preservation, not that they win. H1-H4 alone do not establish superiority as a way to change
models; they establish only cost.

**Reversibility definition.** A method "restores exactly" only if the SHA-256 of the restored source-dtype bytes equals
that of the source. Keeping a snapshot makes every method reversible, so reversibility alone is not a differentiator;
the measure is artefact size. An unmerged LoRA adapter restores by removal at no artefact cost, and that is recorded in
its favour.

**Cumulative questions (secondary).** Whether Cumulative reaches the 90% bar, its footprint as a share of fine-tuning's,
and retention against Edit and Restricted at equal accuracy. Overfit is the gap between accept and report splits per round.

## Analysis

- Primary estimate: paired differences or log ratios by task, within model, with a cluster bootstrap by task
  (10,000 draws, committed seed), then the Holm step.
- Prompt-level statistics are not used for primary inference.
- No result is called superior unless its hypothesis clears the threshold after correction.
- Report every task and model, including failures, in the output table.

## Integrity

- Commit the task set, retention set digest, recipes, tuning budgets, analysis script and this document before any
  sealed evaluation. Record the commit in the ledger.
- All sealed sets are new. Existing holdout and sealed sets may not be used as data. Consultations are one per frozen
  method per task, and repeats are visible.
- Tasks and retention prompts are stored encrypted or held by a second party until the freeze, as an author who reads
  sealed text may be biased. A self-administered seal, as in this repository so far, must be disclosed as a limit.
- Timing runs on an otherwise idle machine, or the study reports machine load and treats wall time as rough. H2 is
  not confirmed on a shared machine.
- Deviations go in a dated log and are reported with results.

## Stopping and invalidation

Stop and re-register if: a shortcut is found in a task after the freeze (drop that task, and report it); any sealed
data enters tuning (the affected method is excluded); or the executor refuses an arm for more than 20% of tasks (the
result is reported as a capability finding about the executor, not about the method).
