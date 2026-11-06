# Promotion gate calibration

This records how reference edits behave under the unchanged promotion gate. **No thresholds were changed because of it, and no real candidate passes.** That is a valid result: it means the gate has not yet been shown to pass any real edit, not that its limits are proven right.

Made by `scripts/calibrate_promotion_gate.py` at commit `db15959` on `EleutherAI/pythia-14m`, one arm at a time on one thread, using public data. Raw results are in [`docs/evidence/promotion_gate_calibration.json`](evidence/promotion_gate_calibration.json).

| Arm | Status | Target gain (at least 0.01) | Task agreement (at least 0.98) | Free-text agreement (at least 0.98) | KL, nats per token (at most 0.0488) | Perplexity ratio (at most 1.05) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| No-op, exact | not evaluated | n/a | n/a | n/a | n/a | n/a |
| No-op, scaled by 1 + 1e-9 | rejected | 0.00 | 1.000 | 1.000 | 1.5e-15 | 1.00 |
| Random, branch 3's norm | rejected | 0.00 | 0.000 | 0.011 | 8.48 | 6,295 |
| Direct low-rank edit | rejected | 1.00 | 0.000 | 0.000 | 10.40 | 47,802 |
| Gym branch 3 | rejected | 1.00 | 0.000 | 0.008 | 12.85 | 696,994 |
| LoRA, the study's 25 steps | rejected | 0.50 | 0.000 | 0.000 | 13.12 | 920,937 |
| Fine-tuning, 25 untuned steps | rejected | 1.00 | 0.385 | 0.311 | 1.25 | 4.01 |

## What this shows

- **The no-op can't be promoted.** The evaluator refuses an exact no-op as identical to the base model. A near no-op passes every retention, KL and perplexity limit, then fails on target gain. It also fails both random-control checks, because noise of that size changes almost nothing either.
- **The retention limits can be met.** The near no-op shows this, but only for an edit that does nothing.
- **Every real edit that gained its target failed all retention limits.** Fine-tuning came closest, and it is still far outside them.

## Limits

- **Fine-tuning and LoRA are not tuned baselines.** Fine-tuning uses `lr=1e-4`, 25 steps and retention weight 1.0, with no search, and LoRA is the study's fixed 25-step run. They don't show what those methods can achieve.
- **The arms have different targets.** The direct edit is scored on its own four associations. The others use the Gym's 12 development prompts, which they were trained on, so target gain isn't held-out evidence for any arm.
- **One model and one corpus.** All results are from Pythia-14M and the public evaluator corpus. A per-task limit of 0.95 on 16 prompts allows no changed answer.
- **No post-selection seal was consulted.** The independent evaluator's corpus and the Gym's `holdout_rules` are the data kept apart from the selection guards, and both are public in this repository. A fresh sealed set still has to come from the controlled study.
