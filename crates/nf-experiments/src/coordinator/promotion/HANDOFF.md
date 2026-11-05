# T039 operational release / safety handoff

## Current release boundary (supersedes historical status below)

The legacy ungranted `Ledger::promote_fixture` still refuses. Production
`Ledger::promote_fixture_with_grant` now ALSO fails closed before reservation,
intent or child work: root issuance alone is insufficient against same-inode DB
rollback, and no trusted privileged monotonic consumption publisher is installed.
There is no agent-writable hash/file fallback. The real positive native 2x2 f64
fixture path remains exercised through the private operator-registry test seam,
which simulates operator-controlled consume-once publication outside SQLite.
No deployed root service, independent engineering review, strict-fsync timing,
scientific generalization, deployment, or general model adapter is claimed.

### Independent-review hardening and current verification

- Immutable grants; retained stable lock sidecars; atomic separately fsynced
  revocation/consumption markers. A torn staging write cannot corrupt authentic
  historical authorization. I/O failure after rename is UNKNOWN outcome, never
  reported success; revocation retry syncs again. Crashes can leave staging files.
- Calling-thread `/proc/thread-self/status`, not `/proc/self/status`; reject
  nonzero inheritable/permitted/effective/ambient capabilities and any root UID
  among real/effective/saved/filesystem IDs. Missing/malformed fields deny.
- Private monotonic consumption precedes admission. Failed DB admission burns
  approval; SQLite rollback cannot remove that external marker. Production has
  no privileged publisher and explicitly refuses instead of claiming protection.
- Historical audit validates retained immutable authorization and consumption
  records without treating revocation/expiration as retrospective invalidation.
  This is not global historical-head rollback detection or whole-registry backup
  protection. Root/kernel/mount/clock/verifier remain trusted.

Verified offline with unique CARGO_TARGET_DIR=/tmp/nf-promotion-hardening2 and
individual named TERM timeouts 85s (KILL grace 3s), no zero-test/ignored passes:

- 11 registry unit tests, including 27 abrupt writer-exit subprocess scenarios,
  injected partial/full writes, file/directory/lock fsyncs and rename boundaries,
  simultaneous issuers/consumers, stable cross-process locks, expiration and
  privilege-field rejection: PASS.
- Nine additional exact promotion unit tests: real fresh-child positive
  publication/reopen/idempotence/tamper plus authentic history after torn and
  published revocation; actual same-inode SQLite restore/replay refusal; live
  registered-child revocation AND expiry; event-failure recovery; exact-input
  tampering; admission/publication failures; concurrent identical promotions;
  byte-copied ledger refusal; strict-scope/ungranted qualification refusal: PASS.
- Two production-library integration tests: arbitrary-digest approval refusal
  and ungranted no-side-effect refusal: PASS. No counterfeit approval from hash.
- Three compile-fail opacity doctests: PASS. Total 25 named tests; subprocess
  scenarios are not inflated into that count. No full repository suite claimed.
- Live capability-bearing thread divergence could not be reproduced: this
  process inherited no capabilities. Native worker TID selection and malformed/
  nonzero-field checks passed; no escalation, setcap or namespace trick attempted.
- No real /etc writes, network, nested agents or edits outside promotion ownership.
  Earlier builds temporarily failed on other owners' missing files/borrow error;
  those owners resolved them and the listed checks subsequently passed.

Logs: /tmp/nf-promotion-hardening2-{registry,positive,rollback,live-revoke-expiry,
event-failure,inputs,admission-failure,concurrent-promotion,copy,strict,
production-digest,production-ungranted,doctests}.log.
The pre-hardening files are retained under /tmp/nf-promotion-hardening2-before.
The project has no Git metadata; no commit/PR exists. work_brief returned not_bound.
The unavailable MCP servers @auto-improvement:auto-improvement and @mochi:mochi
were not retried. The following earlier narrative is subject to this current
production limitation; older verification counts remain historical evidence.

### Native qualification and explicit approval

1. `qualify_fixture_promotion(lease, store, PromotionScope::OperationalFixture)`
   calls T038's actual `qualify_fixture_final` and consumes its opaque
   `NativeFixtureQualification`. It rechecks current frozen profile, final
   decision/chain and actual CAS artifacts. No report boolean, imported JSON,
   environment flag, or test-only witness constructor can produce this handle.
2. The unprivileged host prepares `fixture_promotion_approval_request`. A
   SEPARATE root operator inspects it and issues it into the fixed root-owned
   registry using `operator::issue` or the standalone operator frontend. Only
   then can `approve_fixture_promotion` resolve its identifier. Arbitrary
   digests/agent text are refused. See OPERATOR.md for the runnable frontend,
   OS trust assumptions, revocation protocol and isolated test seam.
3. `promote_fixture_with_grant` takes that opaque non-deserializable grant. It
   checks store/family/contract, owner/epoch/boot/deadline, exact edit/smoke/runtime,
   operation/predecessor, verifier version, native profile/receipt chain and
   scope. Admission, fresh-child launch/registration and final publication
   revalidate live approval. Shared registry locks span actual admission/final
   commits, not the native work: successful revocation during reload denies
   publication; a busy revocation is never reported as successful.
4. The actual reload retains its pre-work pidfd registration, bounded supervisor,
   durable stop/lease monitor and cleanup proof. One commit settles resources and
   publishes acceptance, pointer, frontier and event. Identical committed calls
   are idempotent; interrupted calls cannot rerun from archived witness JSON.
5. Reopen audits the release receipt and native evidence bindings, without
   constructing either a live native witness or an execution grant. Relinked,
   rehashed wrong native scope/profile/receipt/model identities are refused.

StrictFsync is explicitly UnsupportedCapability BEFORE work. Operational means
stop/cancellation/lease checks at the transactional fence, NOT a deadline/CPU
bound through physical SQLite persistence or scheduling. Frozen profiles are not
silently revised. Acceptance scope is
`trusted_native_linear_fixture_operational_only_not_generalization_or_strict_fsync`;
its release origin explicitly says `not_independent_review`.

### Historical pre-registry verification (not authorization-repair evidence)

The counts and positive external approval test below predate OS-authenticated
issuance. That former arbitrary-digest positive test is now a refusal regression;
positive registry issuance is exercised through the private isolated test seam.
Current authorization-repair evidence uses /tmp/nf-t039-auth and /tmp/t039-auth-*.log.

All commands used offline dependencies, CARGO_TARGET_DIR=/tmp/nf-t039-release,
and individual exact-test timeout <=90 seconds. No skips or zero-test passes.

- 35 promotion unit tests, including eight release-boundary cases: PASS.
- 15 actual worker cleanup/cancellation/syscall-failure tests: PASS.
- Two external promotion integration tests (production library, without cfg(test)
  authority): native-qualified positive publication/reopen/idempotence and
  ungranted no-side-effect refusal: PASS.
- Six standalone real Torch reload/inference integration tests: PASS, including
  tied parameters/buffers, alias/dtype/config/packing/artifact/parity corruption,
  substitute identity, cancellation, reaping and wall expiry.
- Three compile-fail doctests establish Acceptance/qualification/grant opacity:
  PASS. Total: 61 target-qualified test cases; no named test timed out.
- Real registered-child stop/takeover now uses the PUBLIC granted route. Actual
  controller-crash tests before/after gate release, crash/reopen atomic rollback,
  unknown-cost recovery, failed registration/event commits and stale/tamper
  refusal all passed. No leftover test/reload child was observed at inspection.
- Evidence: /tmp/nf-t039-release-results.tsv (50 unit cases),
  /tmp/nf-t039-release-integration-results.tsv, /tmp/nf-t039-release-*.log.
  The full repository suite, Python binding suite and independent engineering
  review were NOT run or claimed. Two initial compile errors in new test code
  (helper visibility and a missing clone) were fixed before successful reruns.

### Exact cross-owner integration handoff

Production execution requires a separately reviewed, installed privileged
monotonic consumption authority. Its durable consume-once operation must precede
SQLite admission; its callers must be authenticated and unable to rewrite the
registry. No such service is available here, so production stays fail closed.
The private seam is not a substitute for that installation. T038's native
qualifier is consumed on the operational fixture path tested here. CLI/SDK/workflow owners must expose
explicit operator consent, choose the operational scope, retain the opaque
qualification/grant in the trusted controller, and invoke the three methods
above. Do not deserialize audit receipts, automatically grant from a passing
score, or route a strict-fsync requirement to this operational capability.
No CLI/SDK, shared lib, coordinator/budget/audit, final_fixture or supervisor
implementation was edited by this worker. Model worker production code already
had the required registered-launch callback and needed no changes.

The project has no Git metadata, so no commit/PR was produced. work_brief returned
not_bound; structured work reporting is unavailable. The unavailable MCP servers
were @auto-improvement:auto-improvement and @mochi:mochi; neither was retried.
No nested agents, network, download, paid work or non-fixture study was used.

## Historical private-path record (not current release status)

The sections below retain earlier engineering evidence and then-open handoffs;
claims that public promotion is unconditionally disabled and that launch
registration is absent are superseded by the current release boundary above.

## Implemented within promotion ownership

- Opaque `Acceptance` with immutable accessors and a compile-fail construction test.
- Atomic reservation + durable intent through the existing `reserve_with` helper.
- One admission owner: an idempotent reservation return cannot write another
  in-flight attempt's terminal marker.
- Atomic successful settlement + acceptance/pointer/frontier/journal publication
  through `finish_budget_with`. Failed finalization keeps its reservation.
- Explicit no-rerun recovery: a dead owner or a durable terminal marker is needed;
  uncertain child cleanup stays reserved. Known prelaunch failures and reaped
  attempts can be reconciled without manufacturing reload capabilities.
- Read-only fence monitor bridges durable stop, lease loss, allocation deadline,
  caller cancellation and unreadable fences to the actual worker cancellation
  token. It joins on every exit and cancels on monitor unwind.
- Lease-independent, intent-bound append of reload/supervision audit evidence;
  a lost lease cannot publish or settle authority, but does not discard cleanup.
- Both finite hard address-space ceilings (controller + reload child) must fit
  the allocation. RSS sampling is diagnostic, not the enforcement certificate.
- Archived native replay source, approved smoke suite, inputs and seeds permit
  deterministic reconstruction of the frozen probe digest and child parity /
  inventory audit. Accounting replay validates reservation ownership, completed
  settlement/evidence, costs and resource ceilings. No archived live witness.
- Historical head reader invokes the native T038 resolver and binds its full
  immutable chain around that audit, then rechecks it in the pointer snapshot.
- Family-lineage audit detects a rewound/deleted mutable head, forks and orphan
  predecessors against retained canonical acceptance records.

## Verification (offline, Linux, real qualified local Torch runtime)

- `cargo test --offline -p nf-experiments --lib coordinator::promotion::tests -- --test-threads=1`
  => 11 passed, 0 failed, 7 explicitly ignored former PUBLIC positive cases.
- `cargo test --offline -p nf-experiments --test promotion_atomic -- --test-threads=1`
  => 1 passed (unconditional refusal/no side effects), 7 explicitly ignored.
- `cargo test --offline -p nf-experiments --doc coordinator::promotion`
  => 1 compile-fail opacity test passed.
- Private tests actually publish/reopen/idempotently read, race identical native
  attempts, reject rehashed failed final decisions and relinked wrong reload
  outputs, inject admission/publication failures, and crash after provisional
  settlement + publication but before commit (both roll back; reservation held).
- Actual reload children are pinned/stopped with pidfds before durable stop,
  takeover or deadline. Tests require exit/reaping, incomplete evidence and no
  pointer. Recovery crash tests distinguish prechild, reaped and unknown cleanup.
- Tests needing T038 launch a dedicated finite-hard-RLIMIT_AS subprocess; they
  never lower a shared controller's process limit or add a production bypass.

## Recovery follow-up (unfinished; public promotion still disabled)

- Added crate-private `Ledger::register_fixture_reload_child(&Lease, &OwnedFd)`.
  It derives native identity from the owned pidfd and checks live parent/group,
  native owner, lease/epoch, admission, reservation, deadline and durable stop.
  Immutable `child_registered` evidence is bound to the intent/reservation/epoch.
- Registered fixed NON-FORKING children can now be recovered on the same boot:
  open pidfd BEFORE identity comparison; signal/reap through that exact handle.
  A different start identity is never signalled. Recovery records whether this
  process actually reaped the child; ECHILD leaves reaping to the actual reaper.
- Global `all_cleanup_confirmed` includes promotion admission/registration and
  rejects cleanup reports while the registered exact child remains alive.
  Unregistered same-boot children remain unknown/reserved; reboot proof remains.
- Crash harness waits are bounded. The real reload test observer validates its
  native parent/group/private job argument and live pidfd before SIGSTOP.
- Verified offline: all 15 active promotion tests passed individually (90s bound
  each, slowest 34.3s); 7 intentionally disabled public-positive tests not run.
  All 9 cancellation/recovery integration tests passed individually (45s bound).
  New kernel-child tests exercise real controller SIGKILL, adopted child reaping,
  live inventory, duplicate/stopped/non-pidfd refusal, foreign registration and
  synthetic mismatched-start non-signalling. They do NOT test a reload launch
  handshake; that cross-owner integration is absent.
- Prior workspace log actually finished all promotion tests successfully, then
  stopped in T038 `legacy_or_forged_memory_and_publication_proofs_never_authorize`.
  No T038/sandbox/status/validation/checklist file was edited or rerun here.

## Exact remaining owner / independent-review prerequisites

1. `model_worker.rs` / `model_worker/reload.rs` / bootstrap owner: expose a native
   launch registration callback receiving the still-owned pidfd before allowing
   model work. Connect the PRIVATE promotion call to
   `Ledger::register_fixture_reload_child`; a failed callback must kill/reap the
   child and never release its work gate. Keep the existing parent-death/race
   protection and bounded cancellation/wait4 behavior. A pre-exec blocking gate
   cannot simply use `Command::spawn` (its exec-error pipe would deadlock); the
   trusted bootstrap must acknowledge/wait appropriately or use reviewed native
   spawn plumbing. Add actual reload crash tests before/after durable registration
   and gate release. Promotion's owned files cannot supply this missing hook.
2. Completed locally: global cleanup inventory and registered-child recovery are
   integrated through promotion evidence, without changing ledger schema. This
   does NOT solve same-boot crash recovery for today's UNREGISTERED reload child.
   Never release that reservation from scratch process.json or a missing row.
3. T038 owner and independent reviewer: complete/review T038 on its own evidence.
   These private T039 tests do not satisfy that review. Keep the public refusal
   regardless of any passing report, environment setting or private test.
4. Release semantics: T039's final timing fence is precommit, not an fsync-finish
   or hard CPU-tail guarantee. The final accounting/publication transaction is
   atomic, but its commit duration is not bounded by a precommit observation.
   Do not enable a product promising commit-completion deadlines on this basis.
5. Archival/storage boundary: unsupported verifier-code versions fail closed;
   old-version support needs an explicitly reviewed/versioned audit registry.
   Retained-record lineage detects head-only rollback, not wholesale replacement
   of an entire database by its owner. That stronger property needs a trusted
   external monotonic anchor. No plaintext hash is claimed to authenticate it.

No network was used. No files outside the assigned promotion/test ownership were
edited. The directory has no Git metadata, so no commit or pull request was made.
`work_brief` returned `not_bound`; structured work reporting is unavailable here.
Unavailable MCP servers: `@auto-improvement:auto-improvement`, `@mochi:mochi`.
