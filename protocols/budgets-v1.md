# Family reservations and conservative preflight v1

`nf_experiments::coordinator::budgets` implements real transactional accounting on the coordinator's SQLite connection. Ledger methods include preflight, reserve, settle, close_branch, recover_reservation and budget_snapshot. A Reservation is an opaque non-deserializable accounting handle, bound to store/family/request; it is **not a worker execution grant**. No model, provider or generated code runs through this module.

## Durable schema and authority

An explicit atomic schema-1 -> schema-2 migration preserves all prior canonical contracts, workflow events, family-budget origins and final reservations. It checks the exact version-one schema binding before adding immutable budget requests/outcomes/faults, a fixed family clock epoch and migration provenance; then updates the schema binding/version. Unknown/different schemas are still rejected. Reopening never truncates or resets old accounting. The migration test retains the same frozen contract/event identities and original budget origin.

Each mutation rechecks the exact current coordinator lease/fence and uses an immediate SQL transaction. Clock samples are refreshed after SQL lock acquisition; lease checks also resample the kernel clock instead of trusting a time captured before a busy wait. A live lease is required again before commit. Deadlines cannot be stretched by waiting on another writer. Exact request/settlement retries are idempotent; conflicting bodies under the same operation/attempt ID fail. Costs are never owned by Python or reset by a changed contract revision.

## Quantities and hierarchy

Resources explicitly distinguish cumulative CPU/GPU milliseconds, admitted candidate attempts, model evaluations, programmer/teacher tokens, micro-USD spend and retained storage bytes from concurrently held host/device bytes and worker slots. Family spent plus held capacity must fit the current frozen limits, including reduced revision caps. Memory/worker holds are not additive lifetime costs. Snapshot measured peak host/device values are explicitly unknown (None), not zero observations; this module does not claim it measured RSS or GPU allocator usage.

Branch reservations hold their bounded pool once against the family. Children carve capacity from that parent; total child spent/held cost and active footprints cannot exceed its ceiling. Completed child costs accrue once, reduce the parent's remaining hold, and cannot be billed again when the branch closes. Unused capacity is reusable within the still-open branch. Closed branches require every child to have reconciled/released resources. Work leaves cannot own children; missing/cyclic/deep/disconnected hierarchies fail. Requests are bounded to 4096 per family, hierarchy depth 16, settlement history 16 per request and breach history 256 per family; the main SQLite page/value caps also apply.

Candidate admission consumes an attempt even when the outcome is rejected, incomplete or cancelled; that count cannot be refunded as zero. Retries are sequential, bounded by the frozen per-operation retry cap, and keyed by family/root operation/attempt across revisions. New root operations still consume the same finite family pool. Recovery can reacquire an existing accounting handle through a current native lease, but cannot mint a new family budget behind it.

## Known, unknown and breached work

A Usage field is Some(value), including known zero, or None for unknown. Once a job has confirmed resource release, an unknown cost charges its full reserved ceiling until actual evidence reconciles it. In particular unknown billing cannot release currency for another call. Known values cannot be changed or relabeled unknown; only previously unknown fields can become measured. Each immutable settlement retains its operation, outcome and evidence reference. Evidence authentication belongs to the trusted broker/supervisor; an untrusted gym must not call accounting APIs directly.

Unconfirmed cleanup is Incomplete with unknown usage and retains the entire reservation, including memory and worker capacity. A mere timeout/cancel request does not mean a worker stopped. After cleanup, failed/cancelled work still accrues its actual or conservative unknown costs. Retained storage is charged cumulatively; this task exposes no arbitrary deletion/refund claim.

Reported cost above its reservation is not dropped or silently capped: an immutable report records request/contract, ceiling, measured usage and evidence, retains the outstanding reservation, and halts all new admission for that family. The snapshot exposes the breach reports separately rather than pretending its provisional counters are a complete known total. Revisions/retries cannot erase this halt. Further supervision/manual reconciliation policy is deliberately not a mechanism to rewrite known costs.

## Time and external calls

The first reservation fixes a family BOOTTIME/kernel-boot epoch. Elapsed wall time includes idle gaps and coordinator restarts within that boot. Every request/child fits both the remaining family deadline and its parent allocation deadline. A boot change or unusable clock denies new work instead of restarting time; snapshots report unknown elapsed time while preserving costs. A separate budget approval capability is not implemented.

External requests must match an exact permitted destination/source/payload policy, have finite maximum token and price bounds, and reserve those maxima first. Missing price bounds are unsupported; missing token/reservation/permission bounds are denied. Zero spend does not authorize an unknown-price request. All tools/teacher/training/attribution still need their own frozen permissions. These checks do not fabricate provider quotes, consent or billing receipts.

## Preflight is not hard-limit qualification

Preflight evaluates real contract/lease policy, family/branch affordability and finite bounds before admission. It never trusts a caller-supplied boolean asserting resource enforcement. Generic worker dispatch is fail-closed: diagnostic reports state dispatch_allowed=false with the missing qualified wall/CPU/memory/storage supervisor and scoped-worker authority. Branch allocation bookkeeping itself needs no model worker. T024 adds a separate operator-trusted native fixture path that actually qualifies/applies OS limits and checks the same reservation; it does not turn a generic diagnostic DTO into a grant.

T024 qualifies only the fixed trusted native fixture supervisor. T026 sandbox qualification is still mandatory for untrusted Python/gym work. Accounting does not advertise an always-unsupported adapter as a released integration, nor claim a reservation itself enforces OS resource limits. The existing M1 software-capability manifest still marks study/generated-gym execution unavailable. Required unqualified hard limits cannot fall back to in-process execution or monitored-but-unenforced guarantees. No external paid call or model training was authorized or run in these tests.

Tests exercise actual SQLite races, family/branch cap enforcement, unknown-billing reconciliation, cancelled/unclean work, finite external bounds, retry/revision evasion, exact idempotency, immutable known costs, retained overruns, live-clock expiry across SQL waits, and version-one migration without lost identities. They establish accounting/control behavior, not a model-quality or physical-resource measurement result.
