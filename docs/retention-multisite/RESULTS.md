# Multi-site and cumulative edits: results

**Short answer.** Scope that grows or spreads did raise retention over the selected single-tensor edit on one new
sealed set, but no closed-form edit that spreads across layers beat gradient training restricted to the same tensors.
The cumulative search stopped after 5 steps at 1.5% of the model. It did not approach fine-tuning's footprint, and its
later steps did not improve retention on data that did not decide them. The highest retention came from editing six
output rows, which had low accuracy on new task items. Task generalisation was poor for every method.

Scope: Pythia-14M, one task family, one author's prompts, one sealed pass. Protocol: `PROTOCOL.md`, committed
(`4d77aad`) before any sweep. Selections frozen in `406491f` before any seal evaluation.

## Sealed results

120 new retention prompts, 12 new task items, each scored once. All intervals are 95%. Methods met the public
development bar of 12/12, but they were not equal on new task items, so read the two columns together.

| Method | Parameters changed | Retention [CI] | Mean KL | New task items [CI] | Old holdout* |
|---|---:|---|---:|---|---|
| Control: no edit | 0 | 100% [97, 100] | 0 | 0/12 [0, 24] | 0/6 |
| **Single tensor, selected** | 65,536 | 48% [40, 57] | 0.63 | 5/12 [19, 68] | 6/6 |
| Control: random, same norm and rank | 65,536 | 10% [6, 17] | 3.10 | 0/12 | 0/6 |
| Reference, strength 0.1 | 65,536 | 46% [37, 55] | 1.20 | 6/12 [25, 75] | 5/6 |
| Prior `combination` | 65,536 | 61% [52, 69] | 0.36 | 3/12 [9, 53] | 4/6 |
| **Sequential, layers 3 and 5 (primary multi-site)** | 131,072 | 68% [59, 75] | 0.22 | 4/12 [14, 61] | 4/6 |
| Control: random for sequential | 131,072 | 60% [51, 68] | 0.76 | 0/12 | 0/6 |
| Compensated, layers 3 and 5 | 131,072 | 31% [23, 40] | 1.64 | 7/12 [32, 81] | 6/6 |
| Joint, layers 3-5 (failed the dev bar: 11/12) | 196,608 | 22% [15, 30] | 1.90 | 7/12 [32, 81] | 6/6 |
| Parallel, layers 3 and 5 | 131,072 | 2% [0, 6] | 9.86 | 7/12 [32, 81] | 5/6 |
| **Cumulative, c=1e-7 (and c=0, identical)** | 212,992 | 65% [56, 73] | 0.33 | 5/12 [19, 68] | 4/6 |
| Cumulative, c=1e-6 | 98,304 | 72% [63, 79] | 0.24 | 2/12 [5, 45] | 4/6 |
| Control: random for cumulative c=1e-7 | 212,992 | 38% [30, 47] | 1.68 | 0/12 | 0/6 |
| Readout, six output rows | 768 | 99% [95, 100] | 0.006 | 4/12 [14, 61] | 5/6 |
| Control: random for readout | 768 | 83% [75, 88] | 0.15 | 0/12 | 1/6 |
| Readout plus layer 5 | 66,304 | 83% [76, 89] | 0.056 | 4/12 [14, 61] | 3/6 |
| LoRA restricted to c=1e-7's tensors | 212,992 | 84% [77, 90] | 0.04 | 7/12 [32, 81] | 4/6 |
| Fine-tuning restricted to c=1e-7's tensors | 212,992 | 85% [78, 90] | 0.03 | 7/12 [32, 81] | 5/6 |
| LoRA restricted to c=1e-6's tensors | 98,304 | 85% [78, 90] | 0.05 | 7/12 [32, 81] | 3/6 |
| Fine-tuning restricted to c=1e-6's tensors | 98,304 | 78% [70, 85] | 0.06 | 7/12 [32, 81] | 3/6 |
| LoRA, all down-projections, public guard | 393,216 | 78% [69, 84] | 0.14 | 7/12 [32, 81] | 3/6 |
| LoRA, all down-projections, fit-corpus guard | 393,216 | 83% [75, 88] | 0.07 | 5/12 [19, 68] | 3/6 |
| Full fine-tuning, public guard | 7,641,205 | 69% [60, 77] | 0.19 | 7/12 [32, 81] | 6/6 |
| Full fine-tuning, fit-corpus guard | 7,686,144 | 73% [64, 80] | 0.23 | 7/12 [32, 81] | 4/6 |

\* **The old holdout has a known shortcut:** AND->False and OR->True solve its Boolean items, so 6/6 does not show the
logic was learned. The new task items remove that shortcut: following it scores exactly 2/4 on their Boolean items.
Most edited methods got exactly that pattern, wrong on True AND True and False OR False, which suggests the shortcut is
what they learned, not the logic. Five gradient arms got 3/4: the four restricted arms and full fine-tuning with the fit guard. None got 4/4.

### Registered comparisons (retention, Holm across 12)

| Against the selected single tensor | Difference [CI, points] | Holm p |
|---|---|---|
| Sequential (primary) | +19 [+10, +29] | 0.0009 |
| Cumulative c=1e-7 | +17 [+8, +26] | 0.0016 |
| Cumulative c=1e-6 | +23 [+14, +33] | <0.001 |
| Readout | +51 [+42, +60] | <0.001 |
| Readout plus layer 5 | +35 [+27, +43] | <0.001 |
| Compensated | -18 [-28, -7] | 0.0031 |
| Joint | -27 [-37, -17] | <0.001 |
| Parallel | -47 [-56, -38] | <0.001 |
| Restricted LoRA, fine-tuning (four arms) | +30 to +37 | <0.001 |

These are significant at the prompt level on one task and one prompt set. They do not generalise beyond that.

### Exploratory comparisons (not registered, computed afterwards from saved results, no new consultation)

| Comparison | Difference [CI, points] | Unadjusted p |
|---|---|---|
| Cumulative c=1e-7 vs restricted LoRA | -19 [-30, -9] | 0.0008 |
| Cumulative c=1e-7 vs restricted fine-tuning | -20 [-30, -10] | 0.0003 |
| Sequential vs its random control | +8 [-4, +19] | 0.25 |
| Cumulative c=1e-7 vs its random control | +27 [+16, +38] | <0.001 |
| Readout vs its random control | +17 [+10, +23] | <0.001 |
| Cumulative c=1e-7 vs sequential | -3 [-13, +8] | 0.77 |

## A. Does multi-site separate task from general behaviour?

- **Not by spreading alone.** Parallel, joint and compensated all scored below the single tensor at their selected
  settings. The extra-prefix floor in the selection rule pushed these to large edits; for example, parallel had a 56% tuning-retention setting at full development accuracy that this floor ruled out.
- **Sequential's gain may be mostly a smaller edit.** Its norm is 2.85 against 6.66 for the single tensor. A random edit
  of the same norm and rank kept 60%, and its 8-point lead over that control has an interval including zero. This
  confound was not designed out, so sequential does not show better separation.
- **The output rows did separate, in a limited sense.** Readout changes 768 coordinates and kept 99%. Random rows also
  kept 83%, as top-1 rarely depends on six tokens. It got 4/12 on new items, and it was not a true `EditPlan` step
  (see Limitations). It is a token-level association rather than new reasoning.

## Cumulative search

The search ended by its stopping rule in all three arms. The 40-round safety limit did not bind. Every graph replayed
to the same plan and state digests. Round 1 tested 252 candidates and later rounds 324.

Primary arm, `c = 1e-7`. "Accept" decided the step. The other columns did not.

| Round | Site | Move | J gain | Accept: task / retention | Val | Report prefixes | Tuning retention | Practice | Cumulative changed |
|---|---|---|---:|---|---|---|---|---|---:|
| 1 | L5 `dense_4h_to_h` | task | 0.472 | 80% / 71% | 83% | 78% | 59.1% | 80% | 65,536 |
| 2 | L5 `dense_h_to_4h` | repair | 0.102 | 83% / 77% | 67% | 72% | 61.7% | 85% | 131,072 |
| 3 | L2 `attention.dense` | task | 0.087 | 97% / 74% | 83% | 72% | 62.4% | 75% | 147,456 |
| 4 | L4 `dense_4h_to_h` | repair | 0.020 | 97% / 76% | 83% | 78% | 59.7% | 60% | 212,992 |
| 5 | L5 `dense_h_to_4h` | task | 0.040 | 100% / 77% | 83% | 83% | 59.1% | 60% | 212,992 |

Round 6: best gain -0.0009, so it stopped.

- **Overfitting shows.** From round 1 to 5, accept retention rose 71% to 77% and accept task 80% to 100%. On the unused
  splits, validation stayed at 83%, tuning retention ended where it began (59.1%), and practice fell from 80% to 60%.
  Report-prefix accuracy did rise, 78% to 83%. Steps after round 1 mostly bought accuracy on the items used to accept.
- **No convergence to fine-tuning's footprint.** It stopped at 212,992 parameters: 1.5% of the model, 2.8% of fine-tuning's
  7.6M changed coordinates. Removing the cost (`c = 0`) gave the identical model, so the price did not cause that stop.
  Without a price the search still ran out of useful closed-form moves. This does not say what a richer move set
  would do. `c = 1e-6` stopped earlier, at 98,304 with 3 steps, and took a different path.
- **At equal development accuracy, the search lost to training on its own tensors.** It beat the single tensor
  (+17), but restricted LoRA and fine-tuning kept about 20 points more with the same changed-parameter count, and got
  7/12 against 5/12 on new items (within noise). The search's tensor choice was worth more than its edits. The
  equality is on the 12 development items only: restricted arms scored lower on extra prefixes (78-100% against 100%).
  Those restricted arms inherit the search's tensor choice, which took 54 minutes.

## B. Where direct edits can win: measured costs

Wall times are rough: the machine was shared, with 2 threads, three repetitions on 14M and one on 31M. Peak is the
process high-water mark, and the load row is the model alone.

| Pythia-14M | Single edit | Sequential | Readout | Cumulative (replay) | LoRA | Fine-tuning |
|---|---:|---:|---:|---:|---:|---:|
| Parameters changed | 65,536 | 131,072 | 768 | 212,992 | 393,216 | 7,641,205 |
| Share of model | 0.47% | 0.93% | 0.005% | 1.5% | 2.8% | 54% |
| Wall time to candidate, one recipe | 2-3 s | 7-8 s | under 1 s | 1.3 s | 96-106 s | 196-223 s |
| Peak memory (load is about 418 MiB) | 475 MiB | 475 MiB | 635 MiB | 651 MiB | 531 MiB | 1,165 MiB |
| Plan payload | 31 KB | 92 KB | 6 KB | 282 KB, JSON | 123 KB | 91.7 MB |
| Bit-exact restore from before-image | yes | yes | yes | yes | yes | yes |
| Before-image size | 524 KB | 1 MB | 6 KB | 1.7 MB | 3.1 MB | 61 MB |
| Bit-exact via delta inverse, rounded to f32 | yes | yes | yes | yes | yes | **no** |
| Sealed retention | 48% | 68% | 99% | 65% | 78% | 69% |
| New task items | 5/12 | 4/12 | 4/12 | 5/12 | 7/12 | 7/12 |

| Pythia-31M, one run; retention not measured | Single edit | LoRA | Fine-tuning |
|---|---:|---:|---:|
| Parameters changed | 262,144 | 1,572,864 | 17,641,708 |
| Wall time | 4.6 s | 173 s | 337 s |
| Peak memory (load is about 595 MiB) | 703 MiB | 768 MiB | 2,102 MiB |
| Plan payload | 61 KB | 246 KB | 212 MB |
| Dev / validation | 12/12, 5/6 | 12/12, 5/6 | 12/12, 6/6 |
| Bit-exact via delta inverse, rounded to f32 | yes | no | no |

**Supported, descriptively:**
- Time for one solve: about 40 times faster than LoRA and 80 times faster than fine-tuning.
- Footprint and artefact size against fine-tuning: orders of magnitude smaller.

**Qualifications:**
- **Time ignores search.** The single-tensor configuration came from a sweep of 432 settings, and the cumulative search
  took about 54 minutes against 1.3 s to replay. LoRA's 100 s is one recipe; its search took 88 minutes. Neither total
  was run on an equal budget.
- **Reversibility is not a real difference in kind.** A before-image restores any method exactly, so what differs is
  artefact size. Delta inversion failed for trained merged weights in some runs, but it passed for edits only as an
  observed result, without a guarantee: inversion can fail for very small weights. An unmerged LoRA adapter is removed
  at no cost (true by construction, not separately run), which edits cannot match.
- **Against LoRA the footprint gap is small,** 6x in parameters, and only 4x in payload. On retention and task, LoRA wins.

**Not supported:** better retention at equal task accuracy from any closed-form edit that spreads across hidden layers.
Readout is the exception on retention and scored worse on new task items than LoRA.

## Limitations

- **"Any tensor" was not met.** `embed_in` and `embed_out` exceed the executor's 16 MiB f64 source bound, so they were
  never search candidates; this limit applied in every round. `query_key_value` has no closed-form move, and norms and
  biases are not matrices. Those 6 + 2 matrices and all vectors are excluded. The excluded output matrix is the one
  where the readout family did best, so this may have hurt the search.
- **Readout is not a full `EditPlan` step.** Its rows ran through the Rust executor as a smaller matrix and were spliced
  back in Python, and the receipt covers those rows only. It has no whole-model replay digest.
- **Top-1 retention has a low floor at small norms,** as the random controls show. Comparisons of methods with very
  different norms should be read with their control, and with KL.
- **The new task set is 12 items.** Differences such as 7/12 and 5/12 are within noise, with wide overlapping intervals.
  So task accuracy on new items is *not* shown to be equal across methods, and the "equal task accuracy" here means
  public development accuracy only. No superiority claim on task items is supported.
- **One author, one model, one task family, one pass.** I wrote the sealed prompts having read the older ones, so they
  may share style. Prompt-level tests treat prompts as independent. Tuning budgets differ across methods. Selection
  reused the prior work's rule; its extra-prefix bar decided a lot, including selecting parallel at 2% retention.
- **Frozen-code change:** after the sweeps and searches, and before the freeze, I added a step that records identical
  arms once (`c = 0`), so it isn't tested twice. It changes no solve or score.
- **Refit stability (public):** tuning retention over 80% refits was 63-71% for sequential, 38-46% for single, 97-98%
  for readout.

## Seal accounting

Ledger: `ledger.json`, 75 entries, 25 methods by 3 splits, no repeats. Splits: new retention, new task, and old holdout.
The old holdout was never used for selection. The original sealed set and the earlier fresh sealed set were not
evaluated in any phase; they are only compared by text against the new set, in a test. I read their text earlier in
this work, as they are in the source files, and the new set was written after that.

## Reproduce

```bash
PYTHONPATH=python python -m neural_foundry.retention_multisite tune --families single,parallel,joint,sequential,compensated,readout,readout_plus,gradient_fit_guard
python -m neural_foundry.cumulative_edit search --cost 1e-07   # and 0, 1e-06
python -m neural_foundry.cumulative_edit restricted --cost 1e-07   # and 1e-06
python -m neural_foundry.retention_multisite freeze
python -m neural_foundry.retention_multisite seal     # refuses a repeat
python -m neural_foundry.retention_multisite cost --method single
```
