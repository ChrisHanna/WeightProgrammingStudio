# Paired original/no-op controls v1

`controls.run(cache, runtime, no_op_plan, smoke_suite, protocol, store_root)` performs real paired controls for the registered public synthetic arithmetic fixture. It needs the live qualified T032 runtime and immutable compiled plan; it does not accept Python editor callbacks, arbitrary model factories, private gym task data or controller leases. Neither a reported hash nor a deserialized success receipt can construct a completed `PairedControlsHandle`.

## Matched execution

The protocol freezes registered task/item/group/family/input/split identities, arithmetic head/oracle, task order, distinct seeds, exact CPU f64/batch-one/concurrency-one/cache-disabled settings, reset policy, arm order, limits and context/protocol IDs. Native inventory resolution rejects renamed tasks/groups/families or input relabeling. Duplicate exact tasks are dropped preserving first order; at most 32 distinct tasks, 4 distinct seeds and 64 task/seed forwards are admitted. Repeated seeds are rejected rather than counted as independent samples. This is a public synthetic fixture audit, not a heldout-data grant or scientific acceptance.

The no-op is actually applied using its compiled backend: native transaction or the T033 verified Torch edit path. A plan with any actual parameter/buffer changes is rejected as a no-op even if its text calls itself one. The candidate then goes through native standalone export, a verified CAS root and fresh native restore. Exact parameter/buffer bytes, storage-group/alias identities, architecture/config/graph and auxiliary assets must match the source. It is not implemented by bypassing apply or simply copying a source score.

The original and restored no-op are independently loaded into fresh trusted Torch child processes. Each input has an explicit Torch seed, set before its forward; construction is also seeded and source parameters overwrite the temporary constructor values. Both arms use identical task/seed sequences and runtime settings. Actual finite outputs must match bit for bit across arms. Independent direct arithmetic truth produces per-item/group/seed observations; aggregate accuracy uses distinct task counts (all declared seed repeats must pass a task). Shared groups remain shared, and repeated tasks/seeds do not inflate independent denominators.

The no-op accuracy gain is exactly zero and `no_op_positive_gain_eligible=false`, including saturated reference fixtures and corrupt-source fixtures. These controls make no latency/memory improvement claim. Repeated/interleaved performance measurements and uncertainty remain separate later tasks.

## Exact source evidence cache

The cache is a private in-memory native witness store, bounded to 16 entries, not a writable receipt dictionary or a persistent result importer. Its native canonical key binds the actual source manifest/checkpoint/assets, live runtime identity, compiled control plan, smoke suite, normalized complete protocol, controller code, task order/content, seed order, head, context and measurement/settings bounds. A changed source, runtime, task sequence, seed, protocol, context, setting, limit or plan cannot reuse unmatched evidence. Logically identical dropped task duplicates do not change the normalized workload key.

Only an actual completed source inference can be cached, and insertion occurs after no-op apply/export/reload and completed paired behavior checks. Failed/cancelled/incomplete workers produce no paired witness or cache insertion. Concurrent use is serialized by a bounded cancellable cache wait; cached receipt DTOs cannot be injected. Loss of cache state requires real reevaluation, not reconstructed success from JSON.

A hit reuses the historical source execution receipt and its exact evidence key, but the no-op still goes through apply/export/reload and a new process. Reports distinguish historical source CPU/RSS evidence from `new_source_forward_calls=0`, record fresh no-op calls, and preserve the original source receipt ID. They do not claim a cache hit makes total CPU/cost zero. Native apply/export/reload, parent hashing, cache lookup and runtime setup costs remain additional unknown costs outside the child's kernel measurements; later coordinator budget admission must conservatively account for them.

## Runtime receipt evolution and authority

T032 inference receipts/output reports now use schema version 2 with explicit per-input seed schedules and a native seed-schedule digest. Existing `models.infer` uses explicit all-zero seeds; control inference supplies its frozen schedule. Historical version-1 records had no qualified seed schedule and are not rewritten or treated as seeded evidence. Probe/edit protocols remain their own existing versions. Runtime code hashing invalidates incompatible live runtime identities.

Receipts retain complete application/export/actual process evidence and source identities. Intent/output eligibility and cleanup rules from the underlying native, CAS and trusted-process layers remain in force. No-op artifacts are not promoted. `outputs_eligible=false`, `promotion_status=not_evaluated` and the public-fixture-only scope are explicit. This standalone diagnostic API is not a study-budget reservation, brokered private split evaluation, permission escalation, or scientific generalization claim.
