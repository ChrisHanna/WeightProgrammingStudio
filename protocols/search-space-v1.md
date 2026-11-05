# Transformation Search Space v1

A search space constrains permitted exploration; Weight IR describes one proposed change; strategy bindings describe how proposals are chosen. None is an execution capability or scientific success verdict. Use `parse_search_space`, `resolve_space`, and `ResolvedSearchSpace::check_program` through the core.

## Frozen record

Required fields are `schema_version` (1), exact root `base_model`, frozen `contract` and `objective` digests, `transformations`, `sites`, `constraints`, and `strategy`. Canonical JSON v1 determines identity. Unknown fields, operations, strategies, and versions fail closed. Changing a bound, strategy, site, or capability yields a different identity and requires a legal study revision before execution.

Operation domains select actual target dtypes, per-operation artifact-byte bounds, finite coefficient intervals for scale/projection/low-rank operations, positive rank intervals for projection/low-rank operations, and map sides where relevant. Artifact byte cost deduplicates repeated logical operand IDs within an operation; it is not the final tensor-write cost. The compiler still verifies real artifacts, numerical values, and dimensions.

Site selectors are exact tensor names or nonempty literal prefixes, never regexes or executable expressions. Each selector has a role, whole-tensor or explicit coordinate boxes, and allowed axes. Resolution uses verified inventory, rejects no-match selectors and out-of-shape roles/axes/boxes, and caps total expanded site regions at 4,096. An individual operation must fit one allowed coordinate box; an agent can express a larger permitted union as separate instructions.

Constraints bound changed coordinates, total operand bytes, temporary memory, and operation count, plus buffer permissions, directness, fully gradient-free inspection where requested, shape policy, and named quality constraints. Quality constraints include exact gym binding, metric, units, baseline, evaluation split, maximum regression, and uncertainty protocol. These are pending measurement requirements, not statically proven scores. A hard quality constraint cannot be traded against a Pareto objective or removed by changing IR verification bindings.

Strategy records choose scripted/causal/learned proposals, sequential/branch/evolutionary search, Pareto selection, explicit engine/version, and seed. Registry capability and actual strategy implementation are separate runtime checks. A valid strategy name does not authorize loading arbitrary plugins or make an unimplemented engine available.

## Scope and cumulative budgets

Operation/axis/coordinate permission applies to every actual storage alias, not merely the target name. Physical shared storage is counted once. Equal-valued independent tensors are separate storage groups. Partial-overlap layouts are unsupported unless the inventory adapter qualifies them.

The physical coordinate bound is the union of potential current writes and coordinator-verified cumulative parent effects. A disjoint-box subtraction algorithm avoids enumerating individual parameters; it caps work at 1,000,000 subtraction steps and fragments globally at 4,096. Scalar tensors count as one coordinate; zero-sized regions count as none. The result is a conservative upper bound on actual changed parameters until execution checks the values. Repeated or overlapping writes do not multiply their coordinate count.

Single-parent cumulative records must identify the exact requested parent, original root manifest, same frozen space, resulting source binding, and cumulative writes. `PriorEffects` is a transport record, not proof of receipt authenticity. The coordinator must supply it from verified committed history, and compilation/execution must match fresh actual source inventory. The domain checker rejects missing/wrong parent records; multiple-parent reconciliation remains unsupported until a registered merge protocol supplies its complete lineage policy.

## Separate guarantees

Domain checking performs no I/O, inference, training, or tensor mutation. It enforces resource declarations, membership, source bindings, directness, and exact required quality bindings. The returned report contains the coordinate upper bound and the count of still-pending quality checks. It is not a promotion decision.

Compilation additionally verifies tensor preconditions, actual artifact hashes/layout/values, source/contract authority, alias effects, backend capabilities, and allocation plans. Execution verifies real effects and produces receipts. Independent evaluation supplies quality/resource evidence. No earlier stage can silently stand in for a later guarantee.
