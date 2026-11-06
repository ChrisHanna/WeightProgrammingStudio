# Real gradient baselines on Pythia-14M, three-skill gym

Code: `python/neural_foundry/baselines.py`. Runner: `scripts/run_real_baselines.py`. Raw data is in this folder.

| | Original | Direct agent edit* | LoRA (selected) | Full fine-tuning (selected) |
|---|---|---|---|---|
| Development (12) | 0/12 | not re-measured | 12/12 | 12/12 |
| Validation (6) | 0/6 | not re-measured | 6/6 | 6/6 |
| Practice retention (20) | 100% | not re-measured | 85% | 80% |
| **Holdout (6)** | 0/6 | **5/6** | **3/6** | **6/6** |
| **Sealed retention (20)** | 100% | **60%** | **70%** | **80%** |
| Optimizer steps, selected run | | | 300 | 150 |
| Training wall time, selected run | | | 58 s | 136 s |
| Trainable parameters | | | 15,360 | 14,067,712 |
| Changed parameters | | | 393,216 | 7,641,205 |
| Peak process memory, including model | | | 472 MiB | 1,064 MiB |
| Search: configs / steps / training time | | | 60 / 18,000 / 88 min | 22 / 5,400 / 81 min |

\* Figures given in the task, not re-measured here.

Selected configs:
- LoRA: all six MLP down-projections, rank 4, learning rate 3e-4, guard weight 1, 300 steps.
- Fine-tuning: all parameters, learning rate 3e-5, guard weight 1, 150 steps.

## Result

Full fine-tuning scored higher than the direct agent edit on both sealed measures. LoRA scored lower on the
holdout and higher on sealed retention. So the claim that direct edits beat gradient training is not supported
by this run. Neither does this run show that fine-tuning is better:

- **The samples are small.** 6/6 and 5/6 differ by one prompt. 16/20 and 12/20 give a two-sided Fisher exact
  p of about 0.3.
- **The search effort differs.** Each baseline's weights came from a search over many configs, and the direct
  edit came from one agent session. Only one config per method reached the holdout, so the sealed scores are
  not selected on, but the comparison is still between different amounts of tuning.
- **The holdout is a weak test of skill.** Its Boolean items can be solved by mapping AND to False and OR to
  True, which is also true of the development items. It has no `True AND True` case. A perfect score does not
  show that the model learned the logic. This applies to direct edits as well.

## Cost

Fine-tuning is the least targeted option: it changed 54% of the model's parameters, while a direct edit to the
final down-projection can touch at most 65,536. The parameters it left unchanged are nearly all embeddings of
tokens that never appear in the data. Other sessions were using the machine during these runs, so wall times
are rough.

## Method

- Training runs the model's own forward code. Results load as merged weights and are scored by the same
  `evaluate_prompt` used for direct edits.
- Loss: cross-entropy on development items, plus KL to the original model on a retention guard.
- Selection: validation accuracy plus retention on 10 practice prompts left out of the guard during tuning.
  Ties go to lower validation loss, then fewer steps. The selected recipe was then retrained with all public
  retention prompts in the guard, which direct edits may also protect.
- A second tuning round was added because the first round's best scores sat at the lowest learning rates. It
  used public splits only and happened before any sealed scoring.
- Holdout: scored once, for the selected config of each method and the unedited model. LoRA's other two seeds
  were scored on public splits only (practice retention 90% each).
- Merged weights are not committed. The digests in `refit_report.json` identify them.
