# T038 supervised boundary — operational evidence, not release authorization

## Independent audit and native qualification (2026-11-05)

The approved requirements 2.2/2.5 and design sections 5/6 require finite
admission, monotonic deadlines, cancellation, incomplete unfinished work and
explicit cleanup uncertainty. They do not impose an exact SQLite fsync cutoff.
An explicit frozen whole-operation hard-publication requirement still cannot
be satisfied by this backend; no existing contract or budget scope was edited.

`Ledger::qualify_fixture_final(contract, store)` now returns a native
`NativeFixtureQualification` only for a fully verified passing decision under
the exact current frozen profile. It is not deserializable or caller-constructible.
It independently resolves source/no-op/best/candidate artifacts, verifies launch
weights against their tensors, and links the actual supervisor receipt. The
no-op manifest is now separately retained in CAS, as the candidate already was.
Live and historical output validation independently recompute the registered
non-fused 2x2 f64 math; repeat/no-op parity alone cannot certify correct output.

The witness explicitly identifies `operational_cancellation` timing scope and
unsupported physical SQLite/scheduler tail. It is evidence for a promotion
consumer, not promotion authority, a current lease, fresh-runtime qualification,
or an independent-review boolean. Consumers must check their frozen timing
requirements plus current stop/fence/runtime/release policy. Promotion source
and its existing release gates were NOT edited; full T038/T039 release remains
open pending consumer integration and review of that boundary. Strict callers
must still stop on `require_strict_fixture_publication` failure.

Audit execution used offline cached dependencies and
`CARGO_TARGET_DIR=/tmp/nf-t038-audit`, with each test individually capped at 90s.
The preceding verification counts below are historical, not this audit's totals.

### Final audit results

- 68 tests passed: all 38 trusted-fixture integration tests, 10 Rust supervisor
  component tests, 12 Python real-process tests, and 8 focused fixture unit tests.
  Each was run individually with timeout <=90s; no skips count as passes.
- Final-code 15-boundary real process-death/reopen matrix: 70.96s; deterministic
  durability matrix: 50.79s. Four additional supervisor crash boundaries retain
  unknown holds and reject inserted/rehashed fake cleanup witnesses (18.45s).
- Positive qualification/reopen plus CAS tampering: 5.49s; registration before
  authorization plus named scratch removal: 5.19s; realistic retention-failure
  qualification refusal: 5.20s; post-authorization durable stop/cancel: 8.66s.
- Maximum 32-cell panel: 10.89s; 1/64-probe exact accounting: 17.08s. The frozen
  storage reservation remains 4 MiB. Image preparation narrows the SQLite cap
  to <=1.375 MiB while the <=1 MiB image exists; successful preparation seals
  and closes the source, then checked removal precedes cap restoration. Unknown
  crash cleanup retains the reservation. Unit tests cover cap denial before
  scratch creation. A rejected globally smaller cap broke the 32-cell case;
  that approach was replaced by this tested phase-specific bound.
- Consistent wrong math passes the old decoder parity checks but fails the new
  trusted Rust recomputation. Both live and historical paths use this check.
- Real reader-blocked final observation COMMIT: 18.33s. Operational evidence
  qualifies with its explicit timing scope; strict-publication remains refused.
  A reader-blocked provisional COMMIT remains quarantined (18.06s). No fsync
  completion timestamp or exact physical cutoff is claimed.
- Initial audit test failures (missing no-op manifest and read-only CAS tamper
  setup) were fixed and rerun successfully; they were not counted as passes.
- Native Linux x86-64 was exercised. The static C worker and build guard were
  inspected, not edited. No aarch64, full-workspace or power-loss claim is made.

Consumer integration: use the native `qualify_fixture_final(contract, store)`
result and its contract/decision/timing_scope/digest accessors, not a reviewer
boolean. Calling it does not change stage, accepted pointer, costs or release
policy. The promotion owner still needs reviewed integration under its own
reservation, live fencing/stop/runtime checks and explicit frozen timing scope;
this audit did not edit coordinator.rs, budgets, audit, promotion or registration.

## Implemented and exercised

- Native execution requires an irreversible finite Linux hard RLIMIT_AS no
  larger than the contract host-memory allowance. Existing mappings must also
  fit. Unlimited or incompatible callers fail before reservation/disclosure.
  This is enforced address-space admission, not sampled RSS or an asserted
  allocation envelope. Launch the dedicated native caller under prlimit; the
  library never lowers limits on a shared controller. RSS remains a metric.
- The positive native fixture actually executes edit/no-op/export/reload,
  matched arms, scoring and analysis under the kernel limit and is accepted.
- Reservation admission and intent persistence are one SQLite transaction.
- Resource settlement and provisional completion share one IMMEDIATE
  transaction. Final lease/stop/stage/cancellation/resource checks follow
  receipt serialization. Other connections see held admission until that commit.
- Readable fixture evidence now additionally requires an immutable, hash-linked
  commit_observation made AFTER SQLite's FULL synchronous completion commit
  returned. It records same-boot time, CPU, epoch, decision and completion link.
  Late returns and crashes before observation remain incomplete on reopen;
  recovery never manufactures a past observation. The observation write checks
  current lease/stop/stage/cancellation again, under an IMMEDIATE transaction.
  This is useful recovery evidence, NOT hard deadline/CPU enforcement.
- Interrupted native work retains process identity (boot, PID, start ticks).
  reconcile_fixture_final requires a fresh valid lease and proof that the old
  process disappeared or was replaced. A live or merely zombie group leader
  does not prove cleanup. Unknown CPU/storage costs are charged in full.
- Debug-only failpoints interrupt or abruptly exit at 15 admission, execution,
  decision, completion and observation boundaries. Reopen/reconciliation is
  tested twice for idempotence; unobserved completions never grant acceptance.
- Durable reading rejects missing/oversized hard-memory proofs, old completion
  semantics and malformed observation links/time/CPU/epoch. Old records lacking
  observation remain incomplete, not grandfathered into acceptance.
- require_strict_fixture_publication returns typed UnsupportedCapability before
  admission. Native fixture execution remains available for positive workflow
  evidence, explicitly not strict T038 sign-off. Promotion flags remain false.

## Unsupported physical-publication capability — reproduced, NOT waived

The original strict_deadline_includes_authoritative_sqlite_commit negative
regression is no longer ignored and now passes: a late completion commit cannot
return a decision or become readable evidence. A second regression holds an
actual SQLite SHARED reader lock until past the operation/lease deadline;
COMMIT really waits, succeeds, and leaves durable completion without readable
acceptance after reopen. This is stronger than pausing only a precommit hook.

Neither test proves a physical-publication hard cutoff. Persisting the new
observation has its own unchecked commit interval. The executable
observation_persistence_counterexample_keeps_strict_capability_unavailable test
pauses AFTER the observation's last fence, crosses the deadline and confirms
that fixture evidence can still persist. It also confirms strict capability
remains unavailable. This test documents a counterexample, NOT a relaxed
acceptance criterion. The operational qualification keeps this limitation
explicit; it does not claim that the approved specification requires hard fsync.

SQLite FULL synchronous durability provides no bounded fsync completion time.
Busy timeouts bound lock retry, not filesystem I/O. Progress/commit hooks run
before durable commit and cannot revoke a write already made durable. A timer
or SIGKILL cannot guarantee scheduling or interrupt uninterruptible kernel I/O;
process CPU limits do not establish an exact per-operation subsecond ceiling
through commit. This backend has no qualified hard wall/CPU publication
primitive. No watchdog, accounting ceiling or timestamp is labeled as one.

Resource admission remains held through provisional completion and is released
atomically with the final observation. CPU/storage ceiling charges are conservative accounting, not
proof all tail work stays within them. The hard RLIMIT_AS and database growth
cap remain enforced through the observation, but CPU/wall enforcement is still
missing. Cancellation linearizes at the last acquire-load and stop/lease writes
serialize with the observation transaction; neither is claimed to linearize at
fsync return. A crash without an observation can only reconcile costs/cleanup,
never recover acceptance from a precommit stamp.

Strict callers must use require_strict_fixture_publication and stop on its
UnsupportedCapability error, not fall back and claim T038 compliance. Meeting
an explicit strict-publication contract needs a qualified durable-publication/storage
primitive with hard wall/CPU semantics, or an approved contract revision. The
operational witness is not that revision or a strict capability. No release
authorization or claim of hard-fsync enforcement occurred here.

## Historical verification before this independent audit

All commands ran offline from the project root using cached dependencies.

- cargo test --offline --workspace -- --test-threads=1 completed successfully:
  337 passed, zero failed, 14 ignored across top-level test targets. The 14
  ignored entries are seven protected promotion tests collected twice (unit
  and integration); they are NOT passes and were not edited by this worker.
- Native fixture target: 31 passed, zero failed/ignored, including original
  strict-deadline negative regression, real reader-blocked commit, simultaneous
  stop/cancel, stale owner, observation counterexample and 15-boundary crash
  recovery. The positive edit/no-op/export/reload fixture and largest admitted
  panel pass under dedicated hard AS=1,000,000,000 subprocesses.
- Separate bounded integration run: 78 passed, zero failed/ignored across
  fixture, panel, budgets, cancellation, leases, reload, policy and status tests.
- cargo test --offline -p nf-experiments --test acceptance_gate_trusted_fixture
  -- --list confirms strict_deadline_includes_authoritative_sqlite_commit is
  collected among 31 tests. No --ignored option is needed or used.
- Kernel adversaries: oversized Vec reservation and anonymous mmap fail;
  raising the hard limit fails. Unsupported callers create no reservation.
- Earlier wide attempts failed during concurrent protected promotion edits
  (bad module path, private-field accesses, then unit-test crate imports).
  These failures were not counted as passes; the subsequent full retry passed.

Logs retained on this host:

- /tmp/t038-publication-workspace-retry.log (successful wide suite)
- /tmp/t038-publication-integrations.log (78 passing integration tests)
- /tmp/t038-publication-workspace.log (earlier compile failure)
- /tmp/t038-publication-regressions.log (earlier compile failure)

## Scope and availability

No edits to sandbox.rs, validation, promotion.rs or committed_status.rs.
Promotion remains unconditionally disabled in the existing promotion module.
No network/downloads/paid calls or non-fixture studies were performed.
The project directory has no Git repository, so no commit or PR was created.
work_brief returned not_bound; structured worker reporting is unavailable.
Unavailable MCP servers: @auto-improvement:auto-improvement and @mochi:mochi.
