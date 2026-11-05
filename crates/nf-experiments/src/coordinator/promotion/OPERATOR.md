# T039 OS-separated operator approval

## Current production limitation — monotonic publisher required

Root issuance authenticates approval but cannot stop replay after an agent restores
SQLite into its original inode. Production execution therefore fails closed with
UnsupportedCapability before reservation/intent/child work: no reviewed privileged
consume-once publisher is installed. There is NO hash-based or agent-writable
fallback. Qualification, request preparation and root issuance remain available,
but issuance alone is NOT sufficient to execute production promotion.

The private unit-test registry simulates an operator-controlled monotonic publisher:
it exclusively locks the retained grant lock, atomically publishes and fsyncs a
one-shot consumption marker BEFORE SQLite admission, then retains a shared lock
through admission. A failed admission or controller crash burns that approval.
Replaying an older SQLite image cannot remove this marker. This is test-seam
protocol evidence, NOT evidence of a deployed root service. Installing one requires
separate review of authenticated requests, privilege separation, durability and
recovery; never run an agent-owned binary as root or grant generic sudo.

This mechanism prevents repeated execution of the SAME approval. It does not
recover a deleted database, detect every missing historical head, protect a
root-compromised registry, or prevent a trusted operator deliberately issuing a
new approval. Whole-registry rollback is outside the trusted-host model.

## Trust boundary

Production uses only `/etc/nf-promotion`: no environment flag, caller path,
manual-workflow grant, imported report, or approval digest can mint permission.
Every registry ancestor and grant is opened descriptor-relative with NOFOLLOW.
Ancestors must be root-owned directories without group/other write permission;
grants must be root-owned regular single-link files, bounded to 512 KiB, without
group/other write permission. Root and capability-bearing promotion consumers
are refused using the CALLING THREAD's `/proc/thread-self/status` inheritable,
permitted, effective and ambient capability sets, never the leader's status.
Real/effective/saved/filesystem UID zero and malformed/missing fields also refuse. Issuance/revocation require root. The registry need not contain secrets.

This assumes a trusted Linux host, mount/user namespace, wall clock, kernel,
root operator, and deployed verifier binary. The agent must have neither root,
sudo/delegated issuance rights, nor a way to replace the host registry/verifier.
A same-UID agent cannot create a genuine root-owned registry entry. This does
not defend a root-compromised host, an agent-patched verifier, or availability
against a reader deliberately holding advisory locks. Install reviewed binaries
from a root-controlled staging area; NEVER run agent-writable source as root.
The local source tree is not claimed to have been installed as a trusted service.

## Working issuance path

1. Unprivileged host calls `qualify_fixture_promotion` with OperationalFixture.
   This consumes the real opaque NativeFixtureQualification, not assertions.
2. Call `fixture_promotion_approval_request` with that qualification and the exact
   lease/edit/smoke/runtime/operation/predecessor. Save its JSON for inspection.
   It binds ledger-store identity AND the canonical ledger path plus directory/
   database device/inode identities, frozen contract/family, chain/decision/verifier,
   native scope/profile/receipts, lease owner/boot/deadline/epoch and all inputs.
   A byte-copied database/lease is NOT the approved physical store. Path/inode
   identity alone does NOT detect same-inode restoration. The separate monotonic
   authority is required for replay protection; see the production limitation above.
3. A separate root operator inspects the COMPLETE request, checks it is intended,
   and explicitly issues it with a Unix-millisecond expiration using the frontend
   below, or the public `operator::issue(request, expires_unix_ms)` API.
4. Return the resulting registry identifier to the unprivileged host. It calls
   `approve_fixture_promotion` with the SAME inputs and identifier, then
   `promote_fixture_with_grant`. Production currently refuses execution because
   the privileged consumption publisher is unavailable. The positive fresh-child
   fixture path is tested only through the private monotonic-registry seam.
   Merely knowing/crafting an identifier never authenticates approval.

The standalone frontend is in `operator_main.rs`. It is not the nf CLI and has
no automatic privilege escalation. Build it WITHOUT privilege, offline, using
this workspace's library artifacts (each command is bounded):

```sh
export PATH=/home/ot/.cargo/bin:$PATH
export CARGO_TARGET_DIR=/tmp/nf-promotion-hardening2
export CARGO_NET_OFFLINE=true
cd /mnt/c/Users/NotAPC/Desktop/weightspaceprogramming
timeout -k 3s 85s cargo build --offline -p nf-experiments --lib -j 4
timeout -k 3s 30s rustc --edition=2021 \
  crates/nf-experiments/src/coordinator/promotion/operator_main.rs \
  -L dependency="$CARGO_TARGET_DIR/debug/deps" \
  --extern nf_core="$(find "$CARGO_TARGET_DIR/debug/deps" -name 'libnf_core-*.rlib' -print -quit)" \
  --extern nf_experiments="$(find "$CARGO_TARGET_DIR/debug/deps" -name 'libnf_experiments-*.rlib' -print -quit)" \
  -o "$CARGO_TARGET_DIR/promotion-operator"
```

After independent review, a root operator provisions `/etc/nf-promotion` as
root:root 0755 and installs the reviewed frontend in a root-controlled location.
No test or implementation step here provisions or writes actual `/etc`.
The operator uses that installed frontend as follows:

```text
promotion-operator inspect REQUEST.json
promotion-operator issue REQUEST.json INSPECTED_REQUEST_DIGEST EXPIRES_UNIX_MS
promotion-operator revoke GRANT_IDENTIFIER
```

`inspect` is unprivileged and prints canonical request JSON plus its digest.
`issue` verifies the digest against the bytes read at issuance, preventing a
request-file inspection race. That digest is a binding, NOT authentication:
only the separate root process can issue. No remote-human authentication is
claimed. A human or trusted root controller must actually choose to issue.
Do not authorize an agent to run the issue/revoke commands through generic sudo.

## Revocation, expiry and crash semantics

Grants are immutable canonical JSON; NEVER replace/unlink grants, stable `.lock`
files, `.revoked` markers or `.consumed` markers after publication. Every issuer,
consumer and revoker locks the SAME retained `.lock` inode, not the grant inode.
Issuance/revocation/consumption write a unique staging file, fsync it, atomically
rename it under that lock, then fsync the registry directory. Torn staging files
have no authority and do not damage grant provenance. Revocation publishes a
separate `.revoked` marker; it never shortens or rewrites grant JSON.

A busy result means NOT revoked. Other I/O errors have an UNKNOWN outcome: a
marker might already be published; retry revocation to sync the directory before
reporting success. No pending/busy/error is reported as successful revocation.
Crashed writers release kernel locks; the stable lock file itself is retained.
The protocol assumes a local filesystem honoring atomic rename and fsync. Tests
inject syscall-boundary failures/process exits, NOT physical power loss.

Admission holds a shared lock through reservation/intent commit. The lock is
released during native work, so root can revoke a running attempt. Launch and
registration recheck approval. The publishing writer reacquires and validates
the live record, retaining its lock in the caller until settlement has really
committed or rolled back. Successful revocation before that fence blocks pointer,
frontier, event and budget settlement publication; revocation after it serializes
after publication. Crashed processes release locks; unknown child cleanup remains
reserved and must use the existing bounded native recovery path, never replay.

Expiration is checked using the trusted host clock at admission, launch and the
last logical publication fence. OperationalFixture does NOT promise wall/CPU
limits through unbounded SQLite fsync or scheduler tail. StrictFsync remains
UnsupportedCapability, with no operational fallback. Revoked/expired approval
cannot authorize even an idempotent execution call. Historical pointer reads
require the retained root-issued provenance (including for revoked/expired
records), never issuance or reconstruction of a live grant. Ungranted legacy
acceptance and invented approval identifiers fail historical audit too.

## Test seam and evidence

Only unit-test builds contain a private thread-local temporary-registry seam.
It substitutes the test UID for the root principal; it is NOT root authentication
and is absent from production/integration library builds. Positive tests exercise
the actual issuance/verification APIs, native qualification, gated fresh child,
transaction, reopen and idempotence through that seam. Production integration
tests reject the old arbitrary-digest approval and ungranted publication.
Registry tests cover modes, owners, links, exact inputs, expiry, revocation,
independent-process flock exclusion, issuer/consumer races, every staged write,
file/directory fsync and rename boundary, and abrupt writer exits. The native
positive test checks real accepted history after torn and published revocation;
a real fresh-child promotion is followed by same-inode SQLite restoration and
must refuse replay before any new reservation or child. Calling-thread tests
verify native TID selection and capability-field refusal; no privilege escalation
is attempted to manufacture capability-bearing credentials. Native tests revoke/expire approval while a
real registered child is paused, then require successful reload with no final
publication. Current hardening uses unique target `/tmp/nf-promotion-hardening2`,
offline dependencies and per-command timeouts <=90 seconds. See HANDOFF.md for
current verified results; older `/tmp/t039-auth-*.log` results predate this repair.
