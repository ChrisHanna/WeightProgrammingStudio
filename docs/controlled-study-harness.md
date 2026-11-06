# Controlled-Study Harness

`python/neural_foundry/study/` runs a study whose holdout the proposer can't see or guess. It is the harness for the Controlled study milestone in [milestones.md](milestones.md). No scientific study has been run with it yet.

## Order of work

```text
freeze plan -> synthesize on development items -> freeze manifest
  -> custodian draws holdout -> one query per candidate -> decide
```

| Step | What is enforced |
| --- | --- |
| Freeze plan | The plan's digest is the SHA-256 of its file bytes. The custodian re-checks it on every operation and binds its state directory to it. |
| Freeze manifest | One manifest per study. It lists every candidate, one for each planned synthesis seed, and all exposed operands. |
| Draw holdout | Seeds come from OS entropy after the freeze. The draw excludes declared exposure, and a commitment digest is returned. |
| Query | Each frozen digest gets one query, spent before evaluation starts, so a failure or crash can't be retried. Only aggregates come back. |
| Decide | Every failed gate is recorded as a rejection reason. A failed candidate can't be dropped. |

A single manifest matters as much as the single query. Without it, a proposer could perturb weights slightly, get a new digest and turn queries into a search.

## Statistics

- **Primary metric:** forced-choice accuracy between the target and the alternative token. Labels are exactly balanced, so a model giving one constant answer scores 0.5.
- **Cluster:** one expression rendered under several prefixes. Items become cluster means before resampling, so more prefixes add no false precision.
- **CI:** stratified percentile bootstrap of the paired delta, at level `1 - alpha / tasks` (Bonferroni).
- **Multiplicity:** a one-sided sign-flip test per task, Holm-adjusted across tasks.
- **Seeds:** the CIs cover holdout sampling for these candidates only. Variation across synthesis seeds is gated separately by `min_seed_pass_fraction`, because a bootstrap over a few seeds isn't meaningful.
- **Retention:** the worst candidate's CI bound on top-1 agreement and KL from base, on neutral prompts drawn from a sealed seed.

A plan labelled `harness_demo` or `harness_test` can never promote. A `confirmatory` plan needs at least 200 items per task, 3 synthesis seeds and a real evaluator.

## Control arms

A plan's `control_arms` names the arms. A confirmatory plan must have all four, and other plans have none unless they ask.

| Arm | Artifact | How it's built |
| --- | --- | --- |
| `base` | A marker with no weights, one per study | The unedited model |
| `lora` | One per synthesis seed | Tuned with `baselines.py` on development data, then fit on that seed's development data |
| `fine_tune` | One per synthesis seed | The same, training every parameter |
| `placebo` | One per synthesis seed | A random edit with the candidate's singular values, so its norm and rank match. It sees no data |

- **Same path:** every arm is listed in the one manifest and spends one query per digest. The decision waits for all of them.
- **Missing arm:** freeze is refused with `CONTROL_MISSING`, and no holdout is drawn.
- **Failed arm:** it's recorded as `CONTROL_EVALUATION_FAILED` and rejects the study.
- **Tuning:** `lora` and `fine_tune` must carry a tuning record in the manifest, so the manifest digest binds it. Freeze is refused if the record used a non-development prefix or an undeclared operand. Tuning selects on forced-choice accuracy on validation operands plus retention on a tuning fold.
- **Placebo check:** the custodian compares each placebo's norm and rank with its candidate's at freeze (`PLACEBO_MISMATCH`), because an oversized placebo would flatter any candidate.
- **Comparison:** candidates and controls are paired by cluster and pooled over seeds. Promotion needs each task to beat every comparator on the corrected CI and the Holm-adjusted p-value, and on the seed-consistency gate.
- **Multiplicity:** there's one Holm family of `tasks x (1 + comparators)` tests, which is 12 for the full design. CIs use Bonferroni over the same count.
- **Placebo gate:** if a placebo reaches `min_effect` against base on a task, the study is rejected (`PLACEBO_BEATS_BASE`), since the items or evaluator would be rewarding noise.

## Item generators

A plan selects items by name through `item_generator`, from the fixed list in `generators.py`. The custodian is its own process, so a new generator has to be added to that list; a plan can't name code to import. The custodian checks each draw for exact label balance, cluster ID format and excluded operands (`BAD_GENERATOR`).

`templated_v1`, the current generator, has `confirmatory_ready = False` because its holdout items have a known shortcut. A confirmatory study refuses to freeze on it (`GENERATOR_NOT_QUALIFIED`).

## Launching the confirmatory study

It hasn't been run. It needs:

1. A replacement generator in `_GENERATORS` with `confirmatory_ready = True`.
2. A real candidate method. The demo's scripted AlphaEdit at strength 80 destroys retention.
3. A plan labelled `confirmatory` with at least 200 items per task, 3 synthesis seeds and preregistered thresholds, including `min_control_advantage`.
4. A confirmatory runner. `scripts/study_harness_demo.py` is deliberately fixed to `harness_demo`, with tiny grids.
5. A separate OS account for the custodian, as described under Limits.

Estimate at 2 threads on an idle machine, scaled from the demo's timings: about 100 forward passes per second, 0.4 s per LoRA step and 1.5 s per fine-tuning step, with 12 training prompts.

| Stage | Assumption | Time |
| --- | --- | --- |
| Tune LoRA | 60 configs x 300 steps, the size of the `run_real_baselines.py` grid | About 2 h |
| Tune fine-tuning | 22 configs x about 230 steps | About 2 h |
| Fit controls | 3 seeds x 2 arms, up to 300 steps | Up to 30 min |
| Score 13 arms | 600 items per task (100 clusters x 2 draws x 3 prefixes), 200 retention prompts | About 5 min |

That's about 5 hours, plus candidate synthesis. It's a rough scaling, not a measurement: `all_linear` LoRA steps cost more than the demo's sites, and more training prompts raise step costs in proportion. Disk use is about 700 MB, mostly fine-tuned artifacts and their sealed copies.

## Limits

- **Sealing separates processes, not users.** Anything running as the same OS user can read the state directory or delete the ledger, which would reset the one-query limit. The hash chain detects edits to earlier ledger entries, but not deletion of the file or its tail. A confirmatory run should give the custodian its own account, and the proposer shouldn't have shell access to that state.
- **Exposure is self-declared.** Declared operands are excluded. Undeclared guesses can't be detected from weights; the defence is the sampling space, where a guessed number has about a 4% chance of being in a 400-number draw from 10,000.
- **Boolean disjointness is per expression.** The only operands are True and False, so sharing them is unavoidable. Short expressions are excluded from the draw (3 to 5 terms only), and the space is finite.
- **Prompts share templates.** Development and holdout differ in operands and prefixes, not in sentence frame, so this tests generalization across operands, not across phrasings.
- **Tuning records are self-reported.** The custodian checks a record's prefixes and declared operands, but it can't prove from weights that an arm came from the recipe it names.
- **Default control margin is zero.** `min_control_advantage` defaults to 0, which tests for any advantage. A confirmatory plan should preregister a margin if it wants a larger one.
- **Aggregates still carry a little information.** With one manifest there's no adaptive loop to exploit it, and plans need at least 20 clusters per draw.

## Demo

[study/harness_demo_controls_pythia14m.json](study/harness_demo_controls_pythia14m.json) is a **harness demo, not a scientific result**. It ran every arm from `scripts/study_harness_demo.py` at the commit it records, in 306 s, with 120 items per task, the shortcut items, and tiny tuning grids. It was rejected, which the label guarantees. It shows that all the arms run through the protocol. Its numbers say nothing about the candidate or the controls.

The section below describes the earlier single-arm demo, from an earlier version of that script.

### Earlier demo

[study/harness_demo_pythia14m.json](study/harness_demo_pythia14m.json) is a **harness demo, not a scientific result**. It came from `scripts/study_harness_demo.py`: one untuned scripted AlphaEdit candidate per seed, with 4 rules per task at strength 80.

The record's `commit` field is empty because git can't read this worktree from WSL. The demo ran on uncommitted harness code on top of `0a4fa59`, before typing-only cleanups to the package, so this record doesn't meet the milestone's run-record rules.

It was rejected, which the label guarantees whatever the numbers say. It shows that the protocol runs on a real checkpoint and that repeated queries are refused. It says nothing about AlphaEdit, because the strength and rules were never tuned.

An identity artifact (an unedited weight) scores exactly like base in the test suite, so the damage the demo reports comes from the edit and not from the evaluator.
