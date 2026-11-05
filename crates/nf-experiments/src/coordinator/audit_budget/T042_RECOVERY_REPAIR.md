# T042 recovery repair — implementation handoff

## Scope and invariants

Read the full independent review at
/home/ot/.kiro/crew/subagents/26f76ad471e5bbf6/result.txt before editing.
No coordinator.rs, final_fixture, manual_workflow, promotion, CLI/shared binding,
or task-checklist source was edited. No nested agents or external network used.
The worker session was unbound; work_brief returned not_bound. Declared servers
@auto-improvement:auto-improvement and @mochi:mochi were unavailable, not retried.

Recovery commits stop BEFORE reading the optional runtime binding. It then
revalidates lease/owner and reads the binding in an IMMEDIATE transaction. A bind
that wins first is visible; a later bind observes stop and cannot stage. Only
the exact attested directory identity can be recovered. An opaque cleanup guard
keeps its OS lock through cleanup-evidence and settlement commits. No binding
after the fence proves no authorized staging; the supplied unrelated root is
not swept. Error/busy/wrong-root paths retain the reservation. Costs stay unknown
at their full reserved ceilings; repeated recovery does not double-account.

The barrier test starts with no binding, allows another connection to bind A
and run a real OCI container, then attempts recovery with B. B cannot release;
even A cannot release while the actual job holds its root lock. After stop
tears down the container, exact-root recovery and idempotent reopen succeed.

## Controller memory and local OCI control

Admission is now one definition: 128 MiB guest cgroup + 256 MiB controller
allowance = 384 MiB. Durable audit execution requires a preinstalled finite
hard RLIMIT_AS no greater than the controller allowance or the reserved host
remainder. Unlimited/oversized controllers are rejected BEFORE claim/qualifier.
Current mappings must fit the soft ceiling. The kernel test proves that raising
the hard limit fails and an over-ceiling anonymous mapping is denied.

This bounds the dedicated controller process, including its baseline and
qualification, NOT the whole process tree/host. The report names Docker control
helpers, daemon, unrelated host processes, and operator recovery as exclusions.
Guest metrics remain cgroup-only; controller hard address space is reported
separately. Standalone Python audits explicitly report no controller guarantee.
No whole-host/whole-process-tree guarantee or physical disk capacity is claimed.

Docker's Go CLI cannot initialize under a 256 MiB address-space cap. The fixed,
embedded stdlib-only Engine adapter uses ONLY /var/run/docker.sock, never pulls,
and supports only the existing finite launcher operations. Image, mounts,
capabilities, cgroups and guest isolation still pass the native configuration
verification and real qualification probe. Adapter source is in profile identity.

Cancellation exposed a deferred Engine-create race. Native intent now records
create_started; the trusted helper durably records the successful terminal
response before returning. Cancellation/revocation drains in-flight create;
cleanup cannot certify absence without terminal evidence. A permanently missing
terminal response retains the hold for operator Engine reconciliation. Finalized
journals also recheck actual absence, allowing cleanup of pre-fix test orphans.
Active caller cancellation returns an inconclusive, cleaned report; persisted
stop/revocation/deadline errors remain authority failures, never positive results.

## Exact legacy blocker — no unsafe reconciliation API

Migrated legacy_audit claims contain jobs/allowance but NO authenticated
reservation-to-root inventory. Sweeping an operator-selected root cannot prove
absence elsewhere. The public recovery error explicitly requires a complete
externally verified historical inventory and a separately reviewed migration.
The migration/reopen regression proves holds remain and no cleanup row appears.
No DTO, expiry, partial sweep, or unsupported root/profile re-attestation can
release these holds. Existing profile-bound identities must not be silently
rewritten on upgrade. Legacy lifecycle recovery is NOT claimed implemented.

## Verification on the final compiled source

Offline Cargo builds and maturin editable rebuild succeeded. Target directory:
/tmp/nf-t042-independent-26f76ad4. Each named test ran with timeout --signal=KILL
90s; dedicated hard-memory child tests additionally have an 85-second timeout.
Progress was emitted per test; all runs below completed within their deadlines.

- 51/51 Rust unit tests: 30 audit-budget/migration/recovery/memory, six sandbox
  authority, 11 sandbox storage/crash, four reference-storage/ENOSPC.
- 2/2 public durable_dispatch integration tests in reference_policy_execution:
  real native worker excludes audit at the owner boundary; durable audit excludes
  native dispatch across connections and still completes real OCI execution.
- 11/11 Python cases against the rebuilt extension: all four immutable-gym
  scorer variants, both direct reviewer forgeries, full six-control single-shot,
  terminated and truncated episode audits, sticky active cancellation, controller
  crash/recovery. Individual per-job CPU/wall hard limits were NOT weakened.
- Both forgery tests prove a callback-local marker immediately before the attack
  write, the supervisor's reply-protocol failure, nonzero exit, no aggregate
  observation, confirmed cleanup, and continued negative audit status.
- Aggregate Python wall reservations now include explicit controller overhead.
- Ruff passes for the two changed Python tests and the Engine adapter.
- No remaining sandbox container belongs to this session. Two pre-existing
  containers from other sessions were inspected only and left untouched.

The maximum-retention test preserves one maximal 64-file/depth case plus all 19
maximum 8 MiB payload / 64 KiB output cycles, using eight files in later cycles
to avoid redundant quadratic tree scans. Final run completed in 48.90 seconds.
The final terminated/truncated full audits completed in about 69/70 seconds.

No immutable git snapshot exists in this project directory, and no new
independent review was launched (explicit no-nested-agents constraint). These
are implementation/test results, not independent review or release acceptance.
