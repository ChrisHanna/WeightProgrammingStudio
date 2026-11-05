# Native gym validation and exact-version approval v1

## Supported scope

`gyms.validate(snapshot, plan, sandbox_root, enabled=False)` defaults off. Explicit controller opt-in uses the actual qualified local CPU OCI profile, not a coordinator import/exec or a mocked scorer. `sandbox_root` is an existing physical owner-held0700 directory. Only the exact immutable selected Snapshot assets, fixed probe shim and bounded JSON input enter the worker. No live model/Torch object, checkpoint store, controller credential or arbitrary host path/callback is supplied. Image/dependency support is currently the pinned T026 stdlib CPU image with an empty additional-package lock; other images/dependencies are typed unsupported rather than installed or silently ignored.

This is real finite software validation of the custom protocol below. It does NOT prove distributional coverage, medical/privacy compliance, scorer truth for all inputs, resistance to every reward-hacking strategy, holdout independence, model quality or scientific acceptance. Explicit approval covers only the exact validated snapshot/runtime/protocol. Generic study dispatch, contract permission, budgets, split access and generated-gym enablement remain separately gated. No checkpoint is changed or promoted.

## Controller fixture plan

The strict version1 plan freezes `snapshot`, a bounded `truth_provenance` reference,2–8 distinct cases,2–4 unique seeds, scalar scoring gates and finite sandbox limits. There are exactly2 replays per case/seed and at most32 samples total. Each case declares an ID, input, independently supplied expected output,1–4 schema-valid wrong outputs, and optional episode actions/termination. At least2 distinct inputs/truths are required so a constant oracle cannot pass by construction. The gym's own oracle never generates these expected labels. The truth reference is an explicit controller-fixture provenance assertion, not an authenticated external oracle certificate.

The native instance checker enforces the frozen bounded non-null schemas before probes. Wrong outputs cannot equal the supplied truth. Episode actions are schema-checked, nonempty, at most32 and within declared maximum steps. Expected termination/truncation is explicit. Validation cannot increase the frozen episode wall allowance. Scoring gates require finite ordered `minimum <= bad_max < good_min <= maximum`; passing good scores cannot compensate for any bad/invalid/other failed cell. All checks must pass.

The validation labels, bad examples and inputs are deliberately exposed to this evaluation worker and recorded in the private plan. They must not be presented as unseen final/holdout evidence. Later exact exposure/split policy still applies. These finite known probes are not hidden challenges: arbitrary Python can inspect its own process and recognize tests, so a sophisticated overfitted/malicious program can satisfy finite observations. Native validation establishes the observed composite snapshot+shim behavior, not an unforgeable proof that a particular Python function body executed or that the scorer cannot later be gamed. Approval requires a separate trusted controller action and must never be inferred from untrusted JSON saying `passed`.

## Versioned custom Python protocol

These are Foundry custom-fixture calls, not invented Gymnasium/NeMo/Verifiers compatibility. Framework-specific adapters require their own later actual-protocol qualifications.

Single-shot entrypoints:

```python
def evaluate(inputs, seed: int): ...  # Returns declared output.
def oracle(inputs): ...              # Compared with independent fixture truth.
def score(inputs, output, reference): ...  # Finite scalar reward; not bool/dict.
```

Episode factory returns an object with:

```python
reset(inputs=..., seed=...) -> observation
step(action) -> {"observation": ..., "reward": finite_number,
                 "terminated": bool, "truncated": bool}
result() -> declared_result
close() -> None
```

The selected module's actual path is checked against its declared snapshot file to reject accidental stdlib/module-cache shadowing. Imports and attribute resolution occur only inside the non-root sandbox child. The fixed shim clones JSON arguments, redirects ordinary user stdout to captured stderr, records raw call returns/errors, repeats seeds and invokes close. Import-time writes, subprocesses, syscalls and cleanup remain subject to the same actual OCI limits. Worker stdout is reserved for a bounded JSON envelope; direct writes that corrupt it fail validation. Untrusted code may return arbitrary/forged probe data; it cannot edit the native plan/checks or construct live validation/approval handles.

For every case/seed/replay, Rust independently checks observed output schema and truth equality, oracle/truth agreement, good/observed/wrong scalar scores and null-invalid behavior. Null is invalid for every supported non-null output schema; it must raise/reject or score at most the bad bound, never receive a positive pass. Repeated normalized observations must match exactly for the same seed. For episodes Rust additionally checks reset observations, bounded ordered step traces, exact step fields, finite rewards, schemas, terminal/truncation flags, and successful close. Missing termination, wrong flag types, premature/ambiguous flags, failed close and missing/partial samples reject. Resource/cleanup/runtime failures remain inconclusive, never accepted as missing cells.

The protocol and native checker/schema code have immutable content identities. Changed code/data/scorer/oracle/schema/seeds/image/lock/probe shim/checker/runtime profile requires matching new validation; merely keeping a human version label or presenting an old report cannot approve changed bytes.

## Durable evidence and approval

Rust writes the canonical plan, full native sandbox receipt, bounded canonical raw observations and native-derived report to CAS. The report binds exact snapshot/plan/protocol/image/profile/environment and evidence identities, per-gate booleans, status and explicit measurement scope. Runtime environment identity binds queried engine/OS/security/cgroup/image facts plus frozen native/guardian policy; runtime capability still requires the actual guest preflight. Unknown measurements stay null, not zero. CPU/peak memory are guest-cgroup values and wall is the trusted guardian interval; daemon/startup/controller overhead is unmeasured and must be conservatively charged by the study dispatcher. `brokered_model_calls=0` names that limited scope, not arbitrary Python-internal operation counts.

The guardian performs final CPU/BOOTTIME samples as well as periodic checks, closing the successful-exit-between-monitor-ticks race. Native validation separately requires complete measured costs within the frozen request before emitting `passed`. Sandbox asset receipts v2 bind the actual selected file-content identities; old v1 no-asset journals remain readable through the defaulted optional field. No arbitrary host-output mount or artifact path is introduced.

`GymValidationHandle` is a private-field frozen native witness minted only by actual execution and native checks. Persisted DTOs, success JSON, hashes or mocked reports cannot restore this authority. Failed/inconclusive reports are retained and cannot approve. After a process restart, re-run the saved exact plan under qualified isolation rather than minting a witness from a deserialized report.

`gyms.approve(snapshot, validation, approval_reference)` is a separate explicit controller-only action. It requires the live passing witness, exact matching snapshot and current native policy/protocol, intact plan/runtime/observation/report evidence, and a bounded nonempty approval reference. It restores/verifies all snapshot assets before publishing an immutable approval record. `GymApprovalHandle` cannot be constructed, edited or forged from DTOs. Its record binds snapshot, validation, plan, protocol, environment and runtime profile. Native `Approval::check_environment(snapshot, qualified_runtime)` rejects a different snapshot/runtime/protocol before downstream use. The approval reference alone is not authority, nor does approval grant study execution, extra tools/network, private data export or weight promotion.

## Actual tests

`python -m pytest -q -k gym_validation` uses the real installed extension and Docker backend. It verifies default-off/no-coordinator-import behavior, genuine passing scalar and terminated/truncated episodes, constant-positive/self-attested/invalid-output scorers, wrong oracles/evaluators, seed state drift, absent termination, failed close, resource failures, schema/frozen-limit weakening, exact-version stale approval rejection, corrupt CAS receipt denial and import side effects confined to the actual sandbox. Missing prerequisites fail rather than skip. These are software fixtures, not a model research success claim.
