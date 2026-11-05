# Immutable patch ancestry and exact replay v1

Patch records are content-addressed data, not execution certificates. `nf_patches::replay::PatchGraph` imports a bounded DAG; replay rebuilds every reachable plan from actual sources and operands and obtains fresh transaction receipts. An imported hash or claimed success cannot become a verified history witness by deserialization.

## Records and graph checks

Each body records version, immutable original/base/result model bindings, parent patch IDs, program, compiler/policy/backend identities, frozen search space, smoke suite, operand digests, expected plan/candidate/buffer identities, cumulative effects, and resource admission costs. Its ID hashes canonical body bytes. Parents are patch IDs; the program separately identifies their exact candidate IDs.

Import rejects missing parents, cycles, duplicate IDs/parents, changed body hashes, mismatched model/contract/space bindings, and implicit rebases. Graphs contain at most 64 records with bounded parent lists and canonical input limits. Single-parent execution is supported. Multi-parent graph metadata can be inspected but replay rejects it until a qualified reconciliation protocol exists.

## Independent execution and history

`nf_surgeon::history::certify_history` actually executes a compiled plan, checks the expected candidate identity, serializes its real tensors through the official Safetensors implementation, reloads the snapshot, and creates a private `VerifiedHistory`. It retains actual result binding, root scope, cumulative masks, method history, depth, and cost bounds. It has no deserializer or unchecked public constructor.

Child execution requires this witness, the exact resulting source snapshot and parent candidate, the same frozen root/contract/objective/space, and a fresh recheck of cumulative write limits. A `PriorEffects` DTO cannot replace the witness. Inspection, fitting, unknown-origin, hybrid, and adaptation history cannot be silently erased. Different experiment protocols still require their own approved compile policy and later workflow authorization.

Replaying persisted records rebuilds the chain in topological order, recompiles against actual input bytes, compares compiler/backend/plan identity, runs real transactions and smoke, reloads their snapshots, and compares result model, buffers, cumulative effects, and costs. Missing operands, false result claims, stale versions, and incorrect costs fail explicitly. Parent IDs are never stripped to make a child appear to be a root.

## Bounded materialization and costs

The shared checkpoint serializer uses official Safetensors tensor views/serialization, checked host shapes/byte sizes, header/file limits, and a conservative staging bound before copying model data. One canonical tensor per verified storage group is sufficient; reload reconstructs declared ties and verifies their bytes. This is in-memory snapshot materialization for lineage, not file export or promotion.

Replay limits bound node count, operand-byte admission, materialized checkpoint plus numeric payload bytes, kernel work, verification scans, smoke work, and depth. Source and metadata overhead, caller-owned input pools, and process RSS remain separately supervised. Stored `HistoryCosts` are explicit algorithmic admission bounds and actual serialized lengths, not measured GPU/CPU time or total research expenditure. Shared history is executed once per reachable node, not inferred from record claims.

## Restore and evidence

Root restoration verifies the supplied immutable SourceState and exact checkpoint/model binding, then returns that state. It does not subtract rounded deltas or erase graph/failure records. File-based active pointers and accepted checkpoint promotion are separate export/workflow operations.

All replayed candidates retain transaction stage `tensor_scope_and_smoke` and promotion status `not_evaluated`. Actual operand-production history still requires the experiment ledger; local numerical replay cannot independently prove where externally supplied values came from. Multi-gym quality, retention, independent holdouts, and fresh-process export/reload parity are not established by this component.
