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

Status: **schema only.** Compilation starts after the `spectral_clip` vertical reports, so it reuses that path.

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
