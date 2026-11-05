# Requirements: Direct Weight-Space Programming

## Goal

The Neural Programming Foundry will make direct reading and modification of a model's actual parameter tensors its primary research mechanism. An AI Weight-Space Programmer will use explicit knowledge, multiple gyms, and optional teacher models to propose edits. A Direct Weight-Space Editor will apply those edits to an isolated candidate model. Multi-gym verification will determine whether the candidate can be accepted as a new checkpoint.

Direct editing is distinct from gradient-based training. LoRA, supervised fine-tuning (SFT), distillation, and reinforcement learning (RL) remain separately labeled comparison baselines or explicitly enabled fallbacks. They are not substitutes for evidence that direct editing works.

This is a research framework, not a promise that useful circuits can always be found, edits will succeed, or transformations will transfer across models. Circuit explanations are hypotheses until supported by intervention evidence. No claim about a particular MIT experiment is a dependency or an acceptance condition.

## Starting point and scope

The working directory contains no existing implementation or steering documents. This feature therefore defines a new framework rather than an extension of a local training pipeline.

The initial implementation must support the complete propose–edit–verify–accept workflow on at least one local, inspectable model with writable parameters. Its tests must include a small deterministic model with known parameter-level behavior. Reports must distinguish this synthetic evidence from results on independently trained models. Supported model formats, architectures, tensor types, and available evaluation backends must be declared before a run; unsupported cases must fail explicitly.

The feature includes three intervention types: weight repair, weight construction, and weight transfer. Transfer may produce an explicit unsupported or inconclusive result rather than an accepted model. Cross-task, cross-size, and cross-architecture generalization are research questions requiring separate evidence.

Production serving, automatic deployment, unbounded autonomous training, and downloading models or paid data without authorization are outside scope. Approving this specification does not authorize an experiment, training run, cloud job, or external data transfer.

## 1. Research intent and method identity

**User story:** As a researcher, I want direct edits treated as the main research method, so that results answer whether neural weights can be programmed without training.

### Acceptance criteria

1.1 WHEN a user creates a study, THE SYSTEM SHALL identify direct weight-space programming as the primary method and record the requested intervention type: repair, construction, or transfer.

1.2 WHEN a method produces a candidate, THE SYSTEM SHALL record whether it used direct tensor writes, gradient-based training, a hybrid process, or no parameter changes.

1.3 IF any candidate-producing step uses backpropagation, an optimizer, learned adapter fitting, or trained representation alignment, THEN THE SYSTEM SHALL label the resulting method as trained or hybrid, not gradient-free direct editing.

1.4 WHEN a direct-only run executes, THE SYSTEM SHALL reject configured training steps and record the operations used to construct and apply its edits; a name supplied by an agent SHALL NOT establish method identity.

1.5 IF a direct edit fails, THEN THE SYSTEM SHALL preserve that outcome rather than silently switching methods or reporting a fallback as direct-edit success.

## 2. Run contracts and bounded autonomy

**User story:** As a study owner, I want to freeze the run's goals, limits, and pass rules, so that the agent cannot change the test after seeing results.

### Acceptance criteria

2.1 BEFORE editing begins, THE SYSTEM SHALL require a run contract containing the source model identity, intervention goal, permitted tensor scope, permitted tools, knowledge and teacher sources, gym suite, baseline methods, random seeds, and acceptance rules.

2.2 BEFORE execution begins, THE SYSTEM SHALL require finite limits for wall time, candidate attempts, model evaluations, tokens used by the programmer or teachers, memory use, and external spend where applicable.

2.3 WHEN the contract is frozen, THE SYSTEM SHALL assign it an immutable content identity and include that identity in every candidate and report.

2.4 IF a user changes a frozen contract, THEN THE SYSTEM SHALL create a new study revision and SHALL NOT merge its results into the old study as if its rules were unchanged.

2.5 WHEN the user stops a run or a limit is reached, THE SYSTEM SHALL stop scheduling work, cancel controlled active work, preserve completed evidence, and mark unfinished steps as incomplete. Backend cleanup limitations SHALL be reported.

2.6 IF model downloads, paid services, external model calls, or training fallbacks are not authorized by the contract, THEN THE SYSTEM SHALL not invoke them.

## 3. Model inspection and circuit hypotheses

**User story:** As a weight-space programmer, I want to inspect parameters and activations, so that edit proposals refer to real parts of the target model.

### Acceptance criteria

3.1 WHEN a supported model is opened, THE SYSTEM SHALL expose its architecture, parameter names, shapes, dtypes, devices, tied or shared parameters, and checkpoint identity.

3.2 WHEN a permitted diagnostic example is run, THE SYSTEM SHALL allow bounded activation capture at declared sites and associate each observation with its input, model identity, and capture settings.

3.3 WHEN the agent proposes a responsible circuit, THE SYSTEM SHALL record the affected parameters or sites, observed failure, causal hypothesis, supporting evidence, and remaining uncertainty.

3.4 IF the evidence is only activation similarity, parameter proximity, or correlation, THEN THE SYSTEM SHALL label it as suggestive rather than proof of a circuit's function.

3.5 WHEN diagnostic ablations or probes are used, THE SYSTEM SHALL record their outcomes separately from the proposed final edit and SHALL NOT persist diagnostic mutations into the source model.

3.6 IF inspection cannot identify a supported edit target, THEN THE SYSTEM SHALL return an explicit inconclusive result without fabricating a circuit explanation.

## 4. Explicit edit proposals

**User story:** As a reviewer, I want each proposed change to be precise and traceable, so that I can see what was changed and test the stated claim.

### Acceptance criteria

4.1 BEFORE an edit is applied, THE SYSTEM SHALL produce a proposal containing its intervention type, source model identity, hypothesis, knowledge provenance, exact tensor targets, update operation, required preconditions, and expected gym effects.

4.2 WHEN an update is proposed, THE SYSTEM SHALL represent the actual replacement values or a deterministic, bounded procedure that constructs them, including its inputs and seeds.

4.3 WHEN the proposal is shown, THE SYSTEM SHALL summarize affected parameter counts, tensor shapes, edit magnitudes, and the expected costs and risks.

4.4 IF a proposal exceeds the permitted tensor scope, requires unsupported operations, or lacks required preconditions, THEN THE SYSTEM SHALL reject it before applying changes.

4.5 WHEN a proposal cites knowledge or teacher output, THE SYSTEM SHALL distinguish supplied facts, model-generated claims, measured evidence, and untested assumptions.

## 5. Safe direct tensor editing

**User story:** As a model owner, I want edits applied to an isolated copy, so that failed experiments cannot damage my source checkpoint.

### Acceptance criteria

5.1 WHEN applying a direct proposal, THE SYSTEM SHALL modify actual parameter tensors without requiring gradient descent, backward passes, or optimizer steps in the direct candidate-producing pathway.

5.2 WHEN applying an edit, THE SYSTEM SHALL verify the source checkpoint identity and the targeted tensors' names, shapes, dtypes, index ranges, and sharing constraints before mutation.

5.3 IF a model uses quantized, packed, tied, or otherwise constrained parameters, THEN THE SYSTEM SHALL either enforce a declared compatible editing policy or reject the operation as unsupported.

5.4 WHEN an edit completes, THE SYSTEM SHALL record pre-edit and post-edit tensor identities and verify that no parameters outside the declared edit scope changed.

5.5 IF application fails, creates non-finite parameters, or leaves the model unable to run its required smoke test, THEN THE SYSTEM SHALL reject the candidate and discard partial changes without altering the source.

5.6 WHEN editing succeeds, THE SYSTEM SHALL preserve a reproducible edit record and a reversible path back to the exact source checkpoint.

## 6. Weight repair

**User story:** As a researcher, I want to repair a measured model failure through direct edits, so that I can test whether selected weights cause the problem.

### Acceptance criteria

6.1 WHEN a repair study begins, THE SYSTEM SHALL capture the failing behavior and the source model's scores before proposing a repair.

6.2 WHEN proposing a repair, THE SYSTEM SHALL link selected weights to a testable hypothesis and state the desired behavior change without assuming that the hypothesis is correct.

6.3 WHEN a repair candidate is evaluated, THE SYSTEM SHALL test the target failure and the full frozen retention suite, including examples not used to construct the edit.

6.4 IF the target failure improves but required retention or resource checks fail, THEN THE SYSTEM SHALL reject the candidate for acceptance while retaining the measured repair evidence.

## 7. Weight construction from explicit knowledge

**User story:** As a researcher, I want to construct a neural computation from an algorithm or rule, so that the framework can test programming rather than fitting alone.

### Acceptance criteria

7.1 WHEN a construction study begins, THE SYSTEM SHALL record the explicit algorithm, rule, or knowledge source, its input/output contract, intended domain, and target model insertion or replacement site.

7.2 WHEN constructing weights, THE SYSTEM SHALL produce concrete parameter values and record how the stated rule maps to those values.

7.3 WHEN a constructed computation is tested, THE SYSTEM SHALL compare its outputs with an independent reference implementation on declared normal, boundary, and invalid-input cases.

7.4 WHEN the construction is inserted into a target model, THE SYSTEM SHALL verify both its local behavior and the integrated model's full gym behavior; standalone correctness SHALL NOT establish integrated success.

7.5 IF construction requires an architecture change, THEN THE SYSTEM SHALL require that change to be allowed in the frozen contract and report it separately from same-architecture weight edits.

## 8. Weight transfer with explicit alignment

**User story:** As a researcher, I want to use teacher computations as guidance, so that transfer can be tested without assuming that two networks share a weight basis.

### Acceptance criteria

8.1 WHEN a teacher is selected, THE SYSTEM SHALL record its identity, access mode, usage rights as supplied by the user, and the computations or behavior sought for transfer.

8.2 IF the teacher exposes only outputs, THEN THE SYSTEM SHALL label it as behavioral guidance and SHALL NOT claim to have inspected or copied its internal weights.

8.3 BEFORE any internal weight transfer, THE SYSTEM SHALL record the proposed source-to-target mapping, representation alignment or adaptation method, compatibility assumptions, and supporting diagnostics.

8.4 IF the mapping cannot be justified or implemented for the model pair, THEN THE SYSTEM SHALL return unsupported or inconclusive rather than imply successful transfer.

8.5 WHEN a transfer candidate is tested, THE SYSTEM SHALL compare it with the unchanged target and, where permitted by the frozen study, an unaligned-transfer control; unsafe incompatible copying SHALL NOT be required.

8.6 IF alignment or adaptation is trained, THEN THE SYSTEM SHALL report its training data, compute, and method identity and classify the end-to-end candidate as hybrid or trained.

8.7 WHEN transfer is reported as successful, THE SYSTEM SHALL require measured improvement and retention on the target model; teacher quality or copied tensor counts alone SHALL NOT count as success.

## 9. Multi-gym verification and leakage control

**User story:** As a researcher, I want every candidate tested across multiple gyms, so that narrow gains do not hide lost capabilities.

### Acceptance criteria

9.1 WHEN defining a study, THE SYSTEM SHALL support a frozen suite of at least five named gyms, with explicit target and retention roles, dataset versions, task counts, metrics, and pass thresholds.

9.2 WHEN a candidate reaches verification, THE SYSTEM SHALL evaluate the unchanged source and candidate under matching prompts, decoding rules, seeds, scoring rules, and runtime conditions across every required gym.

9.3 WHEN reporting results, THE SYSTEM SHALL show per-gym changes, generalization scores, retention scores, latency, peak memory, sample counts, uncertainty where estimable, and all failed or missing checks.

9.4 BEFORE search begins, THE SYSTEM SHALL separate examples available for inspection and edit development from examples used for candidate selection and a locked final holdout.

9.5 WHEN the programmer requests evidence, THE SYSTEM SHALL withhold locked holdout inputs, labels, and item-level results from the edit search; final holdout results SHALL NOT feed further edits within that study revision.

9.6 IF a final holdout is reused to guide changes, THEN THE SYSTEM SHALL mark it as exposed and require a new unexposed holdout before making a fresh generalization claim.

9.7 WHEN a study claims compositional, out-of-distribution, cross-size, or cross-architecture generalization, THE SYSTEM SHALL identify and evaluate the corresponding held-out split or model pair separately.

## 10. Acceptance, rejection, and checkpoints

**User story:** As a model owner, I want a clear keep-or-reject gate, so that only edits meeting the frozen rules become accepted checkpoints.

### Acceptance criteria

10.1 BEFORE evaluation, THE SYSTEM SHALL require a minimum target improvement, maximum allowed regression for each retention gym, latency and memory ceilings, and a declared rule for statistical uncertainty or exact deterministic checks.

10.2 WHEN a candidate passes all required checks, THE SYSTEM SHALL mark it accepted and export a new checkpoint containing the edited parameter values directly, without depending on an unapplied patch or training adapter.

10.3 IF any required check fails, times out, is missing, or is inconclusive under the frozen rule, THEN THE SYSTEM SHALL not mark the candidate accepted.

10.4 BEFORE acceptance is finalized, THE SYSTEM SHALL reload the exported checkpoint in a fresh model instance and verify its parameter identity and declared inference-parity checks.

10.5 WHEN a candidate is rejected, THE SYSTEM SHALL keep its proposal, scores, failure reasons, and lineage without replacing the accepted checkpoint.

10.6 WHEN the user restores the source or a prior accepted checkpoint, THE SYSTEM SHALL verify its identity and keep that restore action in the run history.

## 11. Comparisons and optional fallbacks

**User story:** As a researcher, I want fair method comparisons, so that direct-edit claims cannot be explained by hidden extra data or compute.

### Acceptance criteria

11.1 WHEN comparing methods, THE SYSTEM SHALL include the unchanged target and a no-op edit control and SHALL support separately configured LoRA, SFT, distillation, and RL baselines.

11.2 WHEN a training baseline or fallback is requested, THE SYSTEM SHALL run it only when its backend, data access, and resource budget are explicitly enabled; otherwise it SHALL report not run or unsupported.

11.3 WHEN reporting comparisons, THE SYSTEM SHALL disclose each method's source checkpoint, examples and teacher access, selection attempts, trainable and edited parameter counts, training steps, wall time, and memory use.

11.4 WHEN counting direct-edit cost, THE SYSTEM SHALL include programmer inference, inspection, diagnostic probes, teacher use, alignment, edit construction, verification, failed attempts, and export rather than reporting only the tensor-write time.

11.5 IF methods differ in data, trainable parameters, compute, or runtime conditions, THEN THE SYSTEM SHALL show those differences and SHALL NOT label the comparison matched on those dimensions.

11.6 WHEN a fallback produces a useful model, THE SYSTEM SHALL preserve the failed direct-edit result and report fallback success under its actual method name.

## 12. Verified transformation library

**User story:** As a researcher, I want to retain verified transformations, so that later studies can test reuse rather than rediscover edits blindly.

### Acceptance criteria

12.1 WHEN a transformation passes a study's acceptance gate, THE SYSTEM SHALL allow it into a versioned verified library with its edit construction rule, prerequisites, source lineage, model constraints, gym evidence, and resource costs.

12.2 WHEN an edit lacks passing evidence, THE SYSTEM SHALL keep it outside the verified library, with a rejected, proposed, or inconclusive status.

12.3 WHEN a library transformation is applied to a new model or task, THE SYSTEM SHALL check its prerequisites and require fresh verification before accepting the resulting model.

12.4 WHEN presenting a transformation, THE SYSTEM SHALL distinguish its proven scope from untested portability claims and list both successful and failed reuse attempts.

12.5 IF later evidence contradicts a verified claim, THEN THE SYSTEM SHALL support marking that version as disputed or withdrawn without deleting its original evidence.

## 13. Reproducibility and user-visible evidence

**User story:** As a reviewer, I want a complete evidence trail, so that I can reproduce an edit and see what the experiment did not prove.

### Acceptance criteria

13.1 WHEN a run executes, THE SYSTEM SHALL retain the frozen contract, proposal history, model and data identities, parameter changes, tool and method records, seeds, runtime versions, hardware, scores, costs, and stop reasons.

13.2 WHEN the user requests run status, THE SYSTEM SHALL show the current stage, consumed limits, candidate counts, and accepted, rejected, unsupported, or incomplete outcomes.

13.3 WHEN exporting a report, THE SYSTEM SHALL distinguish observed results, hypotheses, synthetic demonstrations, independently trained model evidence, and unsupported scientific claims.

13.4 WHEN replaying a supported deterministic edit against its exact source, THE SYSTEM SHALL reproduce the recorded edited parameter identity; numerical inference tolerances and nondeterministic backends SHALL be declared separately.

13.5 IF dependencies, source artifacts, or required data are missing, THEN THE SYSTEM SHALL report the missing items and SHALL NOT claim a successful replay.

## 14. Security and resource requirements

**User story:** As a model owner, I want bounded tools and protected model artifacts, so that research cannot leak weights or execute untrusted content.

### Acceptance criteria

14.1 WHEN loading model or edit artifacts, THE SYSTEM SHALL use a supported safe loading path and SHALL NOT execute arbitrary checkpoint code as a side effect of inspection.

14.2 WHEN knowledge, gym examples, teacher responses, or model metadata contain instructions, THE SYSTEM SHALL treat that content as research data rather than authority to change permissions, budgets, or acceptance rules.

14.3 WHEN agent-generated edit procedures execute, THE SYSTEM SHALL restrict them to declared model operations, artifact locations, resource limits, and allowed tool access; a restricted workspace path alone SHALL NOT count as execution isolation.

14.4 IF a proposal requires unsupported isolation or permissions, THEN THE SYSTEM SHALL reject execution rather than run unrestricted generated code.

14.5 WHEN a run would send weights, activations, examples, or sensitive outputs to an external service, THE SYSTEM SHALL require that data class and destination to be permitted by the run contract and SHALL record the transfer.

14.6 WHEN measuring latency and memory, THE SYSTEM SHALL declare device, precision, workload size, warm-up, repetition count, timing boundaries, and measured memory scope; unavailable device-level measures SHALL be labeled unavailable rather than replaced with misleading values.

14.7 WHEN inspection or evaluation exceeds a resource limit, THE SYSTEM SHALL stop the affected work and return an explicit failure or incomplete result without relaxing acceptance thresholds.

## Feature-level acceptance

The feature is complete only when automated tests demonstrate the following observable behavior:

- A known repair changes actual parameters, improves the declared failure, passes the five-gym gate, exports a standalone checkpoint, and preserves results after reload.
- A known construction produces parameters from an explicit rule, passes independent rule tests, and is verified inside the target model.
- A compatible teacher-to-target example exercises an explicit mapping and records whether the mapping was trained; an incompatible pair produces an explicit unsupported or inconclusive result.
- A target improvement with a retention regression is rejected. Missing gym evidence, invalid tensors, exceeded budgets, source-identity mismatch, and reload-parity failure also prevent acceptance.
- A no-op candidate does not satisfy a positive improvement threshold, and the source checkpoint remains unchanged after both successful and failed edits.
- The direct-only pathway completes without training steps, while trained or hybrid methods remain separately labeled and require explicit authorization.
- A verified library entry can be replayed on its exact source; reuse on a different model requires new evidence and does not inherit acceptance.

These are software acceptance tests. They do not establish that autonomous editing works on large pretrained models or generalizes across architectures; those conclusions require separately authorized, held-out research studies.
