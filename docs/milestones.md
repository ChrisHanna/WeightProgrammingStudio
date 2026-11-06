# Milestones and Definition of Done

A capability is done when its pass conditions hold on a real checkpoint. Passing on a fixture proves the software path only.

## Rules for every milestone

- **Done is binary.** Each condition has a test or a recorded run that shows it.
- **Skipped is not passed.** Normal CI may skip a test that needs a checkpoint, runtime or backend, and must report it as skipped. A qualification run fails if any of them is missing.
- **Summary-only metadata isn't replayable.** Only a document that carries or content-addresses its operands can be called replayable.
- **Evaluation stays separate from synthesis:**

  ```text
  Gym synthesis -> freeze proposal -> independent evaluation
  ```

  The Gym chooses and emits candidates. It never uses post-freeze holdout results to change them.
- **Every method returns an `EditPlan`.** ROME, MEMIT, mergekit, repair, agents, surgery and Gym output can synthesize differently, but must produce a common plan. Existing wrappers stay stable.

## Milestone 1: Generic Gym Proposal / GPT-NeoX Qualification

Status: **pipeline qualified on Pythia-14M; the qualified edit itself is harmful.** Every condition below has a test or recorded run. The conditions cover replay and measurement, not edit quality.

Recorded run: `docs/evidence/milestone1/`, made by `scripts/qualify_milestone1.py` at commit `8cb4070` on `EleutherAI/pythia-14m` revision `cf967c0`, starting from the proposal the Gym emitted.

| # | Status | Evidence |
| --- | --- | --- |
| 1 | Pass | `tests/test_gym_proposal.py`; deep-frozen values and per-operation schemas in `tests/test_milestone1.py` |
| 2 | Pass, `mlp_down` role only | Resolver tests in `tests/test_milestone1.py`, and the recorded run |
| 3 | Pass | Compilation tests, and byte-identical output from two processes with different hash seeds in the recorded run |
| 4 | Pass | `replay.json`: replay and the f64 export both reproduce the signed digest |
| 5 | Pass, see limit | Replay ran in a directory holding only the envelope and key, with the adapter and Gym modules blocked from import |
| 6 | Pass | `report.json`, cross-checked against `transformers` (largest logit difference under 1e-4) |
| 7 | Pass, see limit | `run_record.signed.json`, verified by `test_recorded_qualification_run_verifies` |
| 8 | Pass | Missing checkpoint, native runtime and `transformers` each fail; a wrong checkpoint or changed code also fails |
| 9 | Pass | `test_missing_checkpoint_fails_qualification_and_skips_normal_ci` |

Measured result, which is not a success claim:

- Target accuracy on the Gym's own 12 development prompts went from 0/12 to 12/12. Those prompts were used in synthesis.
- Top-1 agreement with the base model on free text is 0.8%, and 0.0 on all six retention tasks.
- KL divergence is 12.8 nats per token. Perplexity rises from 138 to about 96 million.
- The Gym's own report gives 100% retention on five prompts. The independent evaluator contradicts it, so that figure isn't evidence of retention.

Known limits:

- The Gym's selection is hard-coded to branch 3 and ignores the scores.
- The study still evaluates its holdout rules after freezing, so the qualification run exposes them. They aren't used for any choice, and the same rules are public in the repository.
- Importing the package loads its other modules, so condition 5 shows the named modules weren't used, not that the rest of the package is absent.
- The signing key is ephemeral: it proves integrity since the run, not who ran it.
- The recorded run executes a `/tmp` copy of the code. The script refuses a copy whose Python package differs from the recorded repository, but the prebuilt native extension is identified only by its digest.

### Pass conditions

1. `GymWeightProposal` contains semantic sites, operations, factors or operands, constraints, evidence source and expected digests.
2. A GPT-NeoX adapter resolves semantic sites into concrete tensors and validates their shapes and roles.
3. Compiling a proposal produces the same executable edit graph every time.
4. A fresh process applies the graph to the base checkpoint and reproduces the signed candidate parameter digest.
5. Replay works without the proposer, the Gym report or the original edited model.
6. An independent evaluator measures target accuracy, retention, KL divergence, perplexity, top-1 agreement and controls.
7. A complete signed run record is emitted.
8. Qualification fails if the checkpoint, runtime or required backend is unavailable.
9. Normal CI may skip this qualification test and reports it as skipped, not passed.

### Non-goals

- Cross-architecture factor portability. Raw factors from one architecture need an explicit alignment recipe to move to another.
- Natural-defect claims.
- Statistical superiority claims.
- Qwen2 qualification before parity and a real checkpoint exist.

### Contract

| Form | Scope | Contains |
| --- | --- | --- |
| `GymWeightProposalSpec` | Architecture-neutral and reusable | Semantic sites, operations, constraints, evidence source |
| `ResolvedGymWeightProposal` | Checkpoint-specific and replayable | Tensor names, dimensions, operands, base and candidate digests |

Implemented in `python/neural_foundry/gym_proposal.py`. Unknown schema versions are rejected.

### Run record

Each result records: commit, checkpoint revision and digest, dependency lock, runtime version, device, precision, sampler, random seeds, edit graph digest, command and artifact manifest.

## Later milestones

Each needs its own pass conditions before work starts.

| Milestone | Depends on | Must define |
| --- | --- | --- |
| Qwen2 qualification | Milestone 1, a real checkpoint, parity | Same contract as Milestone 1 |
| Retention evaluation | Milestone 1 evaluator | Task-level matrix, KL divergence, perplexity on a fixed corpus, target-versus-retention tradeoff, safety and refusal checks, confidence intervals |
| Natural-failure repair | Retention evaluation | Source, collection procedure, expected behaviour, severity, checkpoint revision, and whether any defect was injected. One failed prompt isn't a natural defect |
| Qualification matrix | Milestone 1 | Operation, architecture, dtype, backend and shape policy, each qualified independently |
| Resource limits | `EditPlan` touched-tensor records | Peak RAM, temporary bytes, copy count, wall time and shard limits, preserving source precision |
| Controlled study | All above | Fresh sealed holdouts, seeds, tuned LoRA, promotion thresholds |
| Trust and versioning | Formats settled | Public-key distribution, signer identity, key rotation, and schema compatibility and migration |
