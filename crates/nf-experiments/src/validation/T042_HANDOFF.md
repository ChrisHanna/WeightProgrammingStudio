# T042 trusted dispatch and completeness handoff

## Implemented boundary

Native validation and reference audits now share the fixed supervisor in
`/mnt/c/Users/NotAPC/Desktop/weightspaceprogramming/crates/nf-experiments/src/validation/reference_worker.py`.
The supervisor never imports gym/policy assets. Before spawning callbacks it
sets and verifies Linux PR_SET_DUMPABLE=0, preventing same-UID callback access
to its proc descriptors, memory and ptrace surface. Callback stdout goes to a
private reply pipe, not the supervisor's lifecycle output. Each dispatch has a
fresh 256-bit challenge; strict bounded replies and a final challenged shutdown
are required before the supervisor emits any batch. The callback has no future
challenge and cannot fabricate a completed batch by writing stdout and exiting
during import. Callback subprocesses remain inside the qualified no-network
container with its memory, PID, CPU, wall, output and cleanup bounds. No assets
are imported or executed in the host or trusted supervisor interpreter.

The reviewer's exact os.write(1)/os._exit(0) batch-forgery fixture is tested in
both single-shot and episode modes. Import-time forged current-challenge load
acknowledgements followed by exit also fail. Real callbacks cannot access the
supervisor's proc output descriptor/memory. Native validation uses this boundary
too; the old shared-interpreter worker is retired and fails closed if invoked.

This authenticates SUPERVISOR DISPATCH, NOT the internal semantics of arbitrary
Python. Callback replies, entrypoint-resolution claims, controller truths,
roles and coverage designations remain adversarial or controller assertions.
A malicious callback can deliberately implement the request/reply protocol or
recognize exposed fixtures. Finite probes cannot prove that arbitrary malicious
scorers are honest, that requested Python symbols actually performed the work,
or that an environment has broad scientific validity. No audit-to-approval,
holdout, model-editing or scientific-acceptance conversion exists.

## Completeness and accounting

Full native status `passed` now requires all six executed roles: valid, invalid,
constant, metadata_only, weak and strong; correct valid/strong behavior; weak
non-saturation; scorer/oracle checks; shortcut separation; and complete explicit
coverage declarations. Coverage JSON is {required: [group], case_groups:
{case_id: group}}; every plan case must be mapped and every required group
represented. Extra case IDs and malformed/oversized declarations fail preflight.
Missing controls or coverage yield only `probe_passed` on otherwise successful
runs, with audit_complete=false and explicit missing_prerequisites. Failed or
partial execution cannot become a full pass. Supplied-observation analysis also
requires weak/strong saturation controls and checks strong-policy correctness;
it remains explicitly non-native supplied-result analysis, not execution.

The native binding and Python wrapper accept coverage and a sticky
GymAuditCancellationHandle. One atomic flag reaches every replay, decision and
scoring job. Cancellation prevents full success and preserves incomplete
receipts. Explicit recover_reference_audits requires exclusive root ownership;
qualification may already recover pending jobs, so its returned count describes
only additional explicit-sweep recoveries. A real killed-controller fixture
checks the persisted recovered job has confirmed cleanup and ineligible outputs.

Per-audit asset staging amplification is conservatively reserved and capped at
64 MiB before launching audit jobs. Reports expose that reservation, maximum
per-job cgroup memory peak and actual controller wall time. CPU/wall/memory
measurements are explicitly container-only; controller wall excludes sandbox
qualification. Memory is maximum job peak, not simultaneous aggregate memory;
staging is a reservation, not measured root-retained storage. Existing finite
job counts, aggregate CPU/wall reservations and frozen episode wall bounds remain.
Test CPU reservations now include the separate interpreter/supervisor overhead;
no product limit or frozen episode wall bound was relaxed. Callback values are
serialized immediately, so later callback mutation cannot rewrite earlier
observations. Genuine prefix-replay drift is still rejected.

## Verification (offline, actual qualified Docker and installed PyO3 extension)

- Offline cargo check for nf-experiments and nf-python: passed.
- Offline maturin develop into the project CPython 3.12 environment: passed.
- Combined trusted-audit, episode, Gym Scientist and native validation suites:
  97 passed in 539.87 seconds, no skips/mocks/host callback substitutes.
- Additional actual killed-controller recovery test: 1 passed in 5.58 seconds.
- Offline Rust reference_policy_execution integration suite: 6 passed in 38.08s.
- Ruff, including both supervisor scripts and changed Python tests: passed.
- Mypy on gym_scientist, trusted-audit and episode tests: passed.
- Strict Clippy is blocked by pre-existing/cross-owner type_complexity in
  `/mnt/c/Users/NotAPC/Desktop/weightspaceprogramming/crates/nf-experiments/src/coordinator/final_fixture.rs:31`.
  Clippy with only that lint suppressed passed for nf-experiments and nf-python.
- No network, downloads, model execution, weight edits or non-fixture studies.

## Exact ownership/integration handoffs

- Sandbox owner: no sandbox or guardian source was edited here. The supervisor
  requires subprocess execution and PR_SET/GET_DUMPABLE in the qualified image;
  the real positive/descriptor tests exercise both. Preserve those capabilities
  in future seccomp profiles or fail qualification. Root-wide journal quotas,
  workspace reclamation, crash-safe retention and post-teardown accounting stay
  sandbox-owned. Concurrent sandbox changes adding storage/recovery were built
  and exercised, but this report does not independently certify all their bars.
- Intake/store owner: gyms.restore intentionally returns a read-only CAS handle;
  trying to publish an audit through it fails `read-only artifact store cannot
  publish`. Audit execution needs a writable imported snapshot handle today.
  A future explicit writable-reopen API must be owner-approved, not a silent
  permission upgrade. The recovery fixture reimports unchanged fixture bytes.
- Workflow/CLI owner: GymResearchWorkflow.run and gym-research commands continue
  to analyze supplied JSON rows; they never pretend to execute policies. The
  actual native path is run_reference_audit with immutable snapshot handles,
  coverage and cancellation. A CLI native-execution loader must preserve this
  distinction and provide a writable controller CAS explicitly.
- Conductor: this worker is not bound to a work item; work_brief returned
  not_bound. No structured done claim or acceptance verdict could be recorded.
  No sandbox, coordinator, intake or unrelated source was edited. This checkout
  has no Git metadata, so there is no commit or pull request to report.
