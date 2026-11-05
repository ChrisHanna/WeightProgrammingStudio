# T041 supplied-IR execution — implementation and verification handoff

## Current supported path (not a metadata-fixture substitute)

Operation `nf.manual.supplied_ir.linear_cpu.v2` executes the exact supplied edit
and an explicit caller-supplied no-op through `Ledger::run_fixture_final` in a
dedicated hard-limited process. It performs real edit/smoke/export/core reload,
source/no-op/best-known/candidate controls, paired holdout scoring, statistics,
coordinator reservation/settlement, native process recording and cleanup.
No protected coordinator, final-fixture, promotion, budget, audit, benchmark,
capability, or shared registration file was changed for this implementation.

This supports the coordinator's finite 2x2 f64 linear CPU profile, not arbitrary
architectures or general model search. The supplied search space is resolved and
both exact programs are compiled against it; there is no autonomous proposer.
Software admission stages explicitly delegate scientific work to the real final
dispatcher. They do not manufacture stage receipts. The resulting study remains
at Verify; this is not a claim of whole-study lifecycle completion or promotion.
Strict T038 durable-publication and T039 promotion gates remain closed/unchanged.
`supplied_ir_executed` is true only with a passed coordinator final-execution
check; an incomplete final decision is never relabeled successful execution.

## Explicit no-cycle revision, with no silent contract/program rewrite

- Legacy plan schema v1 and its manifest-in-method_protocol freeze semantics are
  retained. Existing freezes/grants are never reinterpreted as supplied-IR grants.
- Schema v2 requires `control_program`, an explicit no-op for the same source and
  objective. Neither program is synthesized or rewritten by the workflow.
- In v2, contract `intervention.method_protocol` equals the edit's existing
  `objective.experiment_protocol`. That stable protocol ID is NOT a program hash.
- The separate frozen manifest binds exact contract, source export, original
  program, search space, policy, smoke, compiled edit/control and canonical input
  bundle digests. The bundle contains no reference back to the manifest.
- `plan.operator_binding` supplies the exact revision/manifest/source-export/
  source-model identity that the OS operator must explicitly approve. The grant
  also binds frozen contract, permissions, executable bytes, store/ledger/database
  canonical paths and device/inode identities, expiry and operation.
- Freeze retains the unchanged programs, bundle and source manifest in CAS.
  Planning only reads/decodes/resolves/compiles: no inference, probes, downloads,
  runtime qualification, subprocesses or persistent writes.

## Operator deployment and invocation (not performed here)

The fixed authority is `/etc/neural-foundry/manual-execution`, traversed without
symlinks, root-owned and not group/other writable. Canonical single-link records,
content addressing, expiry and permanent revocation markers are verified.
Only real/effective uid 0 can issue/revoke. Execution refuses root. There is no
caller-selected registry, environment override, approval boolean or agent flag.
The administrator must provision the registry and use a reviewed root-owned
operator executable, never run mutable researcher code as root. Issuance only
hashes bounded executable bytes and validates static bindings; it executes none.

1. With the reviewed CLI, explicitly request `nf study runtime-profile INPUT`
   where INPUT is a JSON file containing `{"schema_version":1}`. Freeze that
   observed runtime in the supported contract. This command is separate from plan
   and grants no execution authority. Rebuilding the executable changes identity.
2. Compose schema v2 with the explicit source export, edit, control, search space,
   policy and smoke suite. Keep the approved protocol ID in both contract/program.
   Plan, review its input binding, then freeze with its exact plan digest.
3. An administrator issues `nf study authorize INPUT` containing:
   `{"request":{"schema_version":1,"store":"/absolute/store","ledger":"/absolute/ledger","contract_digest":"sha256:..."},"operation":"nf.manual.supplied_ir.linear_cpu.v2","supplied":PLAN_OPERATOR_BINDING,"worker":"/absolute/reviewed/nf","worker_digest":"sha256:...","expires_unix_seconds":EXPIRY}`.
   Expiry must be future and no more than 24 hours away. Worker must be the exact
   reviewed executable, <=64 MiB (use a debug-disabled/stripped deployment build).
4. Separately, the administrator starts the reviewed root-owned immutable CLI
   with `nf study serve-authority INPUT` (`{"schema_version":1}`). This binds only
   the fixed local Unix socket inside the existing registry; it never creates
   the registry or removes a stale socket. Inspect/remove stale sockets only as
   an explicit operator recovery action. Do not remove grants, locks, consumed
   markers or revocations. Stable lock files must be operator-only mode 0600.
   Pass the authorization ID to ordinary CLI/SDK `run`. The child first obtains
   an authenticated root-service consume-once response, durable outside SQLite.
   Missing service, busy registry, or uncertain response fails closed; an
   uncertain consumption burns the grant and must not trigger automatic retry.
   The parent seals both request and approved executable into memfd and launches without inherited environment,
   with child-only hard AS/CPU/file/core/fd limits and bounded output/waiting.
   The child re-verifies OS authority, observed hard limits and executable digest,
   reloads only the approved frozen bundle, and calls the existing dispatcher.
5. Revocation/expiry are observed before admission, by a live cancellation watcher,
   and before returning. Controller cancellation persists stop; Python now passes
   its real boundary cancellation token. Stop is not a cleanup acknowledgment.
   Resume discovers native intents and invokes final-fixture reconciliation after
   the dedicated child is gone; unknown costs/holds remain conservative. It never
   redispatches interrupted or already-attempted work. Grant IDs are retained in
   admission journal messages; final evidence retains the exact compiled plans.

No production root registry provisioning, operator approval or privilege escalation
was attempted. Tests use an explicitly private, compile-time-only authority seam.

## Verification and remaining review

All commands were offline with CARGO_TARGET_DIR=/tmp/nf-t041-supplied and individual
<=90-second timeouts. Debug-disabled builds keep executable identity tests bounded.
The CLI, native extension and experiments library build successfully.
16 substantive Rust behavior cases and 11 Python/CLI cases passed individually.
The Rust fixture-export helper also ran for Python input generation. The private
child-entry adapter is exercised by positive and revoked subprocess tests, not
counted as a standalone behavior test. Python loaded the freshly built extension
from the target directory, not the source-tree ABI3 binary.

Positive proof runs the real sealed executable launcher and exact verifier, checks
changed parameters=4, explicit no-op changes=0, real export/reload tensor values
[0.5, 1, 1.5, 2], retained source values [1, 2, 3, 4], unchanged supplied program
bytes, an accepted coordinator fixture decision and retained charged evaluations.
Other tests cover forged authority, wrong executable/source/manifest/store/
permissions, expiry, revocation after reservation, pure planning with deliberately
failing smoke expectations, durable stop/cancel/recovery and no uncertain rerun.

No cross-owner dispatcher changes were necessary for this supported profile.
General architectures need a separately qualified coordinator backend, not a
fallback to metadata fixtures. Production deployment still requires genuine OS
operator approval. The initial implementation performed tests and self-review,
not independent review or T038/T039 sign-off. The subsequent independent review
of that implementation is recorded below.

## Independent security review — 2026-11-05

Reviewed the preceding implementation/result, T041's approved task and SDK design,
manual service, authority, supplied dispatcher, CLI and Python/native SDK surfaces.
No nested agents, network requests, production registry provisioning, privilege
escalation, or protected-file edits were used.

**Fixed:** a controller cancellation arriving while a successful child response
was being drained/decoded could miss the final poll and return success without
persisting stop. The parent now rechecks cancellation after decoding, durably
requests stop, and returns TransactionAborted instead. A deterministic test
cancels exactly after decoding the actual live child's response. The child may
already have completed; evidence/costs remain, but the invocation is not called
successful and cannot redispatch. The small fix itself was self-tested by this
reviewer; no second independent review of that delta is claimed.

**Test correction:** the CLI test still expected the old denial-only facade's
UnsupportedCapability/exit 2 for a forged grant. It now requires the implemented
PermissionDenied/exit 3 rejection; production denial was not weakened.

**Boundary review:** production authority remains the fixed root-owned registry,
with per-invocation operation, contract, permissions, store/ledger/database inode,
expiry and revocation checks. Issuance validates static bindings and hashes code;
it never probes or runs researcher code as root. The non-root child verifies its
executable and hard limits; the parent seals bytes before exec. The exact v2
manifest/bundle binds source, unchanged supplied program, explicit control, policy
and compiled plans without a manifest/contract hash cycle. Live withdrawal uses
cancellation; recovery never redispatches uncertain work. No additional confirmed
scope escalation was found in the reviewed finite linear CPU path. This is not a
host-root/same-UID runtime-tampering sandbox or a general-model qualification.

**Fresh positive proof:** supplied scale factor 0.375 (not a default program)
produced reloaded tensor [0.375, 0.75, 1.125, 1.5], changed four parameters, kept
the explicit control at zero changes, retained source [1, 2, 3, 4] and identical
program bytes, and received the real coordinator fixture acceptance decision.

**Added adverse tests:** same-UID byte-identical CAS/registry-record forgery
against the production uid-0 record rule; wrong store; real expiration during
admitted work; late controller cancellation; actual dispatcher exit after durable
admission. Crash recovery waits the real 60-second lease (no DB edits), converts
the held attempt into one spent attempt, charges uncertain CPU/storage at the
reserved ceiling, preserves reservation count, and denies replay with a still-live
grant. Existing forged/revoked/expired/wrong-permissions/source/image/replay and
in-flight revocation tests also pass. All authority seams remain compile-time
private test code, never a production path or selectable authority override.

**Fresh verification:** offline debug-disabled builds of CLI/native extension;
14 authorization test entries in bounded partitions (one is the child IPC adapter),
8 original workflow entries (one fixture-export helper), 2 CLI tests and 11 Python
conformance cases passed. Python explicitly loaded the freshly built native
extension, not the source-tree binary. Individual commands were capped at 90s;
crash recovery passed in about 67s and expiry in about 12s. One combined run hit
its 90s cap; the remaining cases were rerun successfully in separate partitions.
Transient cross-owner compile errors resolved without changes by this reviewer;
a fresh normal lib-test build and the production-boundary unit test also passed.
Use CARGO_TARGET_DIR=/tmp/nf-t041-supplied, CARGO_PROFILE_TEST_DEBUG=0,
CARGO_PROFILE_DEV_DEBUG=0 and cargo --offline. Run --test manual_workflow_authorization
with --skip supplied_ir_crashed_child_retains_charge_and_recovery_never_reexecutes,
then run that crash case alone, each under timeout 90s.

**Absent production provisioning:** no actual /etc authority directory, root-issued
production grant, or production run is claimed. Strict publication, promotion,
whole-study completion and arbitrary architectures remain outside this review.

Unavailable MCP servers: @auto-improvement:auto-improvement and @mochi:mochi.
work_brief returned not_bound; structured work reporting is unavailable here.

## Second independent hardening pass — 2026-11-05

The earlier review's absence-of-additional-gaps conclusion is superseded: inode
binding alone DID permit rollback/replay, and registry writes were not atomic.
This pass independently reviewed that implementation and fixed these issues:

- Atomic synced staging plus rename-NOREPLACE for immutable grants, revocations
  and consumption. Orphan staging cannot grant authority. Idempotent revocation
  re-syncs an already published marker after an unknown result.
- Stable operator-only 0600 locks serialize issuance/revocation/consumption.
  Busy revocation is NOT reported successful. Readable flock files are refused:
  read-only access would otherwise let researchers lock out revocation.
- A real local operator service durably consumes each grant outside the
  researcher-owned ledger BEFORE dispatch. It uses authenticated SO_PEERCRED,
  exact grant/static identity validation, store/ledger owner scoping, bounded
  input and socket I/O, and no shell, probes or user code execution. Production
  pins the root peer and registry; there is no selectable production trust root.
  Root service startup refuses mutable researcher-owned executable paths.
- Consumers check permitted/effective/ambient capabilities and all four UIDs
  from /proc/thread-self/status, not the process leader. Root and saved-root
  consumers fail closed. No privileged capability mutation was performed here.
- Exact manifest schema rejects unknown callback/path metadata and malformed
  digest fields; unprivileged recompilation checks the entire retained manifest.
- Controller return rechecks cancellation and live authority after decoding a
  child response. It persists stop rather than returning authorized success.
- A sealed finite stdin document replaces the pre-deadline blocking pipe write.

Fresh evidence uses CARGO_TARGET_DIR=/tmp/nf-t041-hardening2, offline cargo,
CARGO_BUILD_JOBS=2 and CARGO_PROFILE_{DEV,TEST}_DEBUG=0, each command <=90s.
All 27 authorization entries passed in partitions (two are IPC child adapters),
8 original workflow entries (one export helper), 2 CLI tests and 11 Python cases.
The real crash/recovery case passed after waiting the actual lease (66.6s).
Python loaded /tmp/nf-t041-hardening2/debug/lib_native.so explicitly.
Positive supplied scale 0.375 still produced [0.375,0.75,1.125,1.5], unchanged
source/program, real controls/export/reload and accepted fixture evidence.
The test then restored the PRE-RUN database bytes while preserving its inode,
verified the original live grant and Frozen state, and proved a fresh child
cannot execute it again. Other new cases cover real publisher process exits
mid-write/after-consume/after-revoke, uncertain replies, reconnect/reopen,
lock contention, root-peer impersonation rejection and bounded incomplete IPC.
Withdrawal immediately before the coordinator publication fence also passed:
watcher-driven cancellation rolled back completion/commit-observation records,
kept authority.accepted false, retained costs and denied replay/recovery dispatch.
Process-exit/error injection is NOT a hardware power-loss durability proof.

**Open mandatory verification/deployment gates; NOT whole-task completion:**
The positive tests use an isolated compile-time operator-principal seam owned by
uid 1000, NOT a literal uid-0 registry. The real uid-0 record rule rejects those
same-UID records. An attempted unprivileged user namespace was denied with
Operation not permitted; no escalation or /etc provisioning was attempted.
A preapproved isolated root/unprivileged test environment is needed for literal
root-owned registry and full positive production CLI/SDK/service integration.
The CLI/SDK planning/control/error conformance cases do not substitute for it.
The new privileged service and hardening delta were self-reviewed/tested by this
worker, NOT independently reviewed by another agent; no nested agents were used.
Conductor review and explicit operator approval/install remain required before
production execution. No new production authority or deployment is claimed.

Cancellation/revocation are checked at logical admission/return fences and by a
live watcher. Late cancellation does not erase already committed final evidence
or refund costs. This is not a claim of atomic external revocation against the
protected coordinator's SQLite finalization commit, nor an adversarial same-UID
runtime/SQLite-tampering sandbox, strict publication, or promotion qualification.
Protected files were not edited. A cross-owner review note: promotion's separate
operator code also uses researcher-readable flock files; its owner should assess
revocation-lock denial of service independently rather than treating this manual
registry repair as covering promotion.
