# Three-way split broker v1 (coordinator schema4)

## Exact source catalogs, not independent-sample certificates

`coordinator::splits::publish_dataset` stores a bounded version1 dataset manifest with role and explicitly named task/group/family identities plus immutable canonical input/label blob references. Payloads must be bounded valid canonical JSON,16KiB each; datasets have1–256 items. A frozen contract's split `dataset` reference must resolve to these actual bytes, not a DTO claiming a missing hash is available.

`Ledger::bind_split_catalog` checks the current live coordinator lease and exact contract/binding. It revalidates every dataset/blob, canonical manifest identity, role, count and exact task/group/family inventory against the frozen contract. It rejects duplicate tasks and cross-role task/group/input-byte aliases. All previously bound catalogs in the study family participate, not just the current gym: renaming an exact consumed holdout input under a new item/group/revision cannot make it fresh development/selection/final evidence. Same-role shared data remains traceable; it is not automatically an extra independent sample.

The immutable catalog binds exact development, selection and locked holdout manifests plus payload references. It does not certify semantic grouping, ownership, independence, prior unknown exposure or absence of labels embedded in caller-supplied code. Assertions such as KnownUnexposed remain assertions until the exact exposure/legacy intake tasks resolve their histories. Semantic/near-duplicate/legacy lineage checks are not replaced by hashing input bytes.

## Frozen final panel before private metadata

`freeze_final_panel` runs only under a live lease in Verify, before any final-worker metadata is exposed. It freezes the exact contract/revision, declared finalist model identities, original/best-known controls, all final-required bindings, seeds, repeats, comparison family and panel protocol. Empty, duplicate or over-policy finalist sets reject. An identical request is idempotent; a changed panel requires a new approved contract revision, not an update to the old row.

These are frozen model references, explicitly labeled `frozen_identities_not_model_execution_evidence`. They are not verified model-loading/execution witnesses or a scientific result. A later model/evaluation resolver must verify the actual artifacts and execution evidence; freezing arbitrary metadata does not make a model available or accepted.

No holdout grant can open without this actual durable panel. Opening is recorded irreversibly BEFORE exposing holdout worker task IDs/metadata, not only after labels are successfully read. Native Verify/Learn phases cannot reopen candidate/search work; the budget admission policy and every split-broker request recheck that phase. A new approved family revision inherits final reservations and exact source history instead of clearing exposure.

## Least-privilege worker scopes

`grant_split_worker` is a trusted controller constructor requiring the live coordinator lease, registered exact binding, real qualified OCI capability and bounded lifetime. It creates private non-deserializable `Grant` state, a random worker ID and a random scoped lease secret. The secret is NOT the coordinator Lease, store credential, SQLite capability or model handle; only its hash enters durable grant storage. The worker receives a bounded Welcome containing only its own role's task IDs, contract/revision/binding, panel reference if final, and label permission.

| Recipient | Inputs | Labels |
| --- | --- | --- |
| Proposer | Development | Denied |
| DevelopmentEvaluator | Development | Development |
| SelectionEvaluator | Selection | Selection |
| FinalEvaluator | Locked holdout | Locked holdout |

Proposers never receive selection/holdout inventories or label hashes. Changing an envelope's worker secret, contract, revision, binding, split, panel or item does not widen scope. Unknown fields/oversized frames reject. Cross-worker or cross-binding token substitution fails. Worker deadlines are fixed, boot/fence-bound and at most the creating coordinator lease; renewing the coordinator cannot silently extend an old scope. Each worker has at most64 data accesses and a contract has at most1024 worker grants.

`read_split` verifies the live lease/store/epoch/boot, persisted grant digest and secret hash, role/fields/item/panel, current workflow stage and owner stop before data I/O. It durably records an access-start intent before resolving input/label bytes, retaining partial/failed reads rather than erasing them. It rechecks the fence after CAS I/O and before disclosure. No request path or raw artifact key is accepted. Payloads contain only the authorized item/fields, split classification and access reference, with `scoped_data_not_a_dispatch_or_promotion_grant` status.

Exact shown-field/completion/retry/semantic exposure ledgers are extended by the following task. This task's conservative access-start rows are not claims that a failed read displayed its complete payload.

## Actual separate OS access domain

A controller-owned `ScopedPayload::in_domain` requires the same qualified runtime environment plus live lease/phase/deadline checks before and after the calculation. It sends ONLY its scoped canonical bytes through the private read-only stdin file to a fresh qualified OCI process. The private coordinator/store/catalog/other splits are never mounted. Private PID/IPC/network/filesystems, UID65534, no host credentials/socket/GPU and the T026 kernel/guardian quotas apply. A proposer cannot inspect a final worker's input mount or persistent scratch; each container has private bounded scratch destroyed on exit. This is not an in-process Python dictionary filter presented as adversarial isolation.

The method is a low-level controller data-isolation calculation, not a study/model dispatcher or budget reservation. The trusted controller already owns the corpus and must route only the issued scope; the untrusted guest never receives this object, the Ledger, the coordinator lease or filesystem authority. Native study execution remains gated by explicit contract permissions, budget reservations, validated/approved gym identity, model resolution and output policy. Every sandbox output remains ineligible for promotion. Resource/cancel/crash accounting of general study jobs is not replaced by this data-scope API.

Previously authorized bytes cannot be made unknown again: an operator or worker may remember past development data. The implementation denies fresh reads/dispatches and model/candidate admission after final closure; it does not pretend to erase memory. Separate evaluator processes also prevent an old proposer scope from receiving new final labels through a shared mount. Data embedded in previously supplied source or already exposed elsewhere requires explicit provenance/exposure policy, not a new role name.

The older trusted CPU fixture inference broker now also requires an actual frozen panel before creating a final scope, records the opening and rechecks phase/panel on every request (including cached requests). Its original assurance remains explicitly trusted-fixture discipline, not the new OS-isolated data-domain qualification.

## Durable schema migration and verification

Schema3→4 appends immutable catalogs, panels, worker scopes, access starts and holdout openings, with update/delete guards. The unchanged schema1/MIGRATION_V2/MIGRATION_V3 byte hashes remain authoritative; schema4 adds its own binding/provenance in one transaction. Existing contracts, event chains, leases/fences, conservative budgets/faults/stops/worker observations and final reservations are retained. Historical migration tests strip only empty schema4 tables before recreating their real v1/v2 fixtures; no prior migration constant is reinterpreted.

`cargo test -p nf-experiments split_broker` tests actual schema3 migration with live lease, retained cost hold, stop, events and final reservations. `python -m pytest -q -k split_broker` builds/uses the real `fixture_splits` Rust controller, SQLite/CAS and qualifying Docker runtime: exact field roles, forged identities/tokens/panels, actual dataset/group/input aliases, unresolved references, freeze-before-holdout, immutable panel/max finalists, reopen persistence, stop/expiry/quotas, post-final search/prepared-scope denial, inherited holdout renaming and actual cross-container/host-path privacy are checked. Complete metadata printed by the fixture server is visible only to its trusted test controller, never to its sandbox guest. No missing-runtime skip or mock grants isolation evidence.
