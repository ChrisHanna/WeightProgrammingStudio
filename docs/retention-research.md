# Retention research: can an association edit keep general behaviour?

**Result: mostly negative.** No method kept full task accuracy and clearly raised sealed retention. The best
fresh-set retention among methods that passed the development bar was 62.5%, up from 45% for the
reference edit. That method lost holdout accuracy (4/6 against 5/6) and scored lower on the original sealed set (50%
against 60%). Nothing approaches the 98% retention that `studio.py` requires for promotion.

Model: Pythia-14M, editing `gpt_neox.layers.5.mlp.dense_4h_to_h.weight` unless stated. Code:
`python/neural_foundry/retention_research.py`. All edits are closed-form and applied by the Rust executor.

## Results

Each method was frozen before sealed evaluation. Retention means the top-1 token is unchanged.

| Method | Dev | Validation | Practice retention | Tuning retention (n=149) | Holdout | Sealed, original (n=20) | Sealed, fresh (n=80) [95% CI] |
|---|---|---|---|---|---|---|---|
| **Reference: strength 0.1** (`strength_only`) | 100% | 83% | 100%* | 40% | 5/6 | **60%** | **45%** [35, 56] |
| Adapter at strength 80 (`baseline_adapter`) | 100% | 83% | 100%* | 0% | 6/6 | 0% | 0% [0, 5] |
| Corpus + damping (`combination`) | 100% | 83% | 55% | 59% | 4/6 | 50% | **62.5%** [52, 72] |
| Damping, ridge 1 (`damping`) | 100% | 83% | 100%* | 42% | 4/6 | **75%** | 51% [40, 62] |
| Covariance, soft (`covariance_soft`) | 100% | 100% | 55% | 42% | 6/6 | 40% | 51% [40, 62] |
| Hard-negative mining | 100% | 83% | 25% | 44% | 4/6 | 40% | 44% [33, 55] |
| Corpus, soft (`corpus_soft`) | 100% | 83% | 25% | 42% | 4/6 | 40% | 42.5% [32, 53] |
| Corpus, low-rank null space (`corpus_lowrank_null`) | 100% | 83% | 55% | 38% | 5/6 | 40% | 42.5% [32, 53] |
| Covariance null space (`covariance_null`) | 100% | 83% | 45% | 43% | 6/6 | 35% | 36% [27, 47] |
| Rank limit 6 (`rank_limit`) | 100% | 83% | 100%* | 24% | 6/6 | 40% | 25% [17, 35] |
| Layer 4 scope (`region_layer`) | 100% | 100% | 80%* | 11% | 6/6 | 15% | 5% [2, 12] |
| Neuron scope, 32 neurons (`region_neurons`) **failed dev bar** | 92% | 50% | 95%* | 5% | 1/6 | 15% | 11% [6, 20] |
| Relaxed task bar (`tradeoff_dev75`) **failed dev bar** | 75% | 67% | 80% | 67% | 4/6 | 70% | **80%** [70, 87] |
| Control: random edit, same norm and rank as strength 80 | 0% | 0% | 0% | 1% | 0/6 | 0% | 4% [1, 10] |
| Control: no edit | 0% | 0% | 100% | 100% | 0/6 | 100% | 100% [95, 100] |

\* Practice prompts were protected by the solve, so this figure is in-sample and says little. Corpus and covariance
methods never saw the practice prompts, so their lower figures are honest held-out measurements.

## What this shows

1. **The 60% figure came from a low-strength edit.** The earlier session's candidate used strength 0.1. Its delta norm
   (4.484) matches `strength_only`. At strength 80, the value in the README examples, sealed retention is 0% and
   the edit's output on unseen prompts is about 424× the weights' own, with predictions collapsing onto task
   tokens such as ` even` and ` no`.
2. **Edit magnitude is the largest lever found.** Going from strength 80 to 0.1 moved fresh retention from 0% to 45%.
   Strength 0.05 failed the dev bar in the reference family, so little room remains.
3. **The 60% overstated general retention.** The reference scored 45% on the fresh set and 40% on templates it had
   never seen. The original sealed set closely mirrors the practice set's sentence shapes, and the protected practice
   prompts probably help it. This is a likely explanation, not a tested one. On the fresh set's shape-matched half the
   reference scored 52.5%, against 37.5% on the diverse half.
4. **A broad protected corpus alone did not beat the short list.** The corpus methods landed at 36–51% on the fresh set
   at their best settings. Only corpus protection combined with damping did better, and it cost holdout accuracy.
5. **Task accuracy and retention trade off.** Allowing dev to fall to 75% reached 80% fresh retention; across the sweep,
   dev at 50% reached 77% tuning retention. Task and generic keys overlap heavily: only 26% of task key energy lies
   outside the top 50 directions of the corpus, and 7% outside the top 300. So a null space large enough to protect
   behaviour removes most of the task signal. A single linear edit to these weights may not separate them well; this
   sweep supports that but does not prove it.

No method dominates. `combination` was the best feasible method on tuning retention, so it is the choice this
protocol's data supports, yet it gives up holdout accuracy. `damping` and `covariance_soft` each win on different
columns, and picking among them from the sealed columns would be selecting on the seal.

## Protocol and seal accounting

- **Selection:** within each family, require 100% dev and 100% on extra development prefixes (`Item:`, `Question:`,
  `Problem:`), then maximise the mean of practice and tuning retention, breaking ties by smaller edit. Validation was
  not used to select.
- **Corpus:** 493 generic prompts in 44 groups, split by group into 344 for fitting and 149 for tuning. Prompts sharing
  exact text or a word trigram with the sealed, practice or task sets were dropped.
- **Sweep:** 482 configurations in three rounds; the later rounds extended the strength grid and added the frontier
  after public results showed winners at the grid edge. 55 were refused by the executor and are recorded as failures in
  `retention-research-tuning.json`.
- **Seal order:** frozen selections and the fresh set were committed (`5484833`) before the seal pass ran.
- **Consultations in this work:** 15 methods, each consulting holdout, original sealed and fresh sealed exactly once:
  45 entries, no repeats (`retention-research-ledger.json`). The earlier session had already consulted the holdout and
  original sealed set for the reference candidate, so that candidate has seen those two sets twice.

## Limitations

- **Original sealed set is contaminated.** It had already reported the 60% figure, and I read its text before writing
  the corpus. Treat the fresh set as the primary number.
- **Fresh set is one author's.** I wrote it and the corpus, so they may share style, and it is only 80 prompts. With 15
  methods compared, its confidence intervals are unadjusted and overlapping ones should not be read as differences.
  `combination` against the reference ([52, 72] and [35, 56]) overlaps.
- **No valid paired test against the reference.** The stored McNemar tests compare with strength 80, because I assumed
  that was the reference until after the seal pass. Those p-values are not meaningful here, and per-prompt results
  weren't saved. Fixing that needs another consultation, which I did not make.
- **Region scoping was only partly tested.** The executor refuses edits needing more than 4096 coordinate spans
  (`actual-coordinate receipt exceeds 4096 spans`), which excluded the 64, 128 and 256 neuron settings and all
  neuron-restricted `combination` settings. Only 16 and 32 neurons ran, so this does not rule out wider scoping.
- **Holdout is six prompts,** so 4/6 against 5/6 is weak evidence by itself. Top-1 identity is also a strict metric:
  an unchanged distribution with a tie flip counts as a failure. Original margins are small (mean 0.59 logits).
- **One model, one task, one seed.** Refits on 80% of the corpus moved `combination` tuning retention between 48% and
  62%, so small differences between corpus methods are within noise.

## Reproduce

```bash
PYTHONPATH=python python -m neural_foundry.retention_research tune --checkpoint /tmp/pythia-14m
```

```bash
PYTHONPATH=python python -m neural_foundry.retention_research seal --checkpoint /tmp/pythia-14m
```

`seal` refuses to consult again unless given `--force-reconsult`, and the ledger then records the repeat.
