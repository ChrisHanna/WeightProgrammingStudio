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

## Limits

- **Sealing separates processes, not users.** Anything running as the same OS user can read the state directory or delete the ledger, which would reset the one-query limit. The hash chain detects edits to earlier ledger entries, but not deletion of the file or its tail. A confirmatory run should give the custodian its own account, and the proposer shouldn't have shell access to that state.
- **Exposure is self-declared.** Declared operands are excluded. Undeclared guesses can't be detected from weights; the defence is the sampling space, where a guessed number has about a 4% chance of being in a 400-number draw from 10,000.
- **Boolean disjointness is per expression.** The only operands are True and False, so sharing them is unavoidable. Short expressions are excluded from the draw (3 to 5 terms only), and the space is finite.
- **Prompts share templates.** Development and holdout differ in operands and prefixes, not in sentence frame, so this tests generalization across operands, not across phrasings.
- **No control arm yet.** Candidates are compared only with the base model. The milestone's tuned LoRA and a placebo edit aren't implemented.
- **Aggregates still carry a little information.** With one manifest there's no adaptive loop to exploit it, and plans need at least 20 clusters per draw.

## Demo

[study/harness_demo_pythia14m.json](study/harness_demo_pythia14m.json) is a **harness demo, not a scientific result**. It came from `scripts/study_harness_demo.py`: one untuned scripted AlphaEdit candidate per seed, with 4 rules per task at strength 80.

The record's `commit` field is empty because git can't read this worktree from WSL. The demo ran on uncommitted harness code on top of `0a4fa59`, before typing-only cleanups to the package, so this record doesn't meet the milestone's run-record rules.

It was rejected, which the label guarantees whatever the numbers say. It shows that the protocol runs on a real checkpoint and that repeated queries are refused. It says nothing about AlphaEdit, because the strength and rules were never tuned.

An identity artifact (an unedited weight) scores exactly like base in the test suite, so the damage the demo reports comes from the edit and not from the evaluator.
