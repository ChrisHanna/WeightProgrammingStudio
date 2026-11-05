# Native transactional execution v1

`nf_surgeon::transaction` executes only an immutable `CompiledPlan` against a loaded immutable `SourceState`. It returns a `VerifiedCandidate` only after all steps, effects, numeric guards, and required smoke checks pass. It performs no checkpoint file writes, training, model download, or experiment promotion.

## Source and candidate ownership

Loading validates real dense Safetensors bytes, all tensor/storage-group identities, parameter-set digest, inert configuration/assets, and declared sharing. Native parameters are currently finite f32/f64. Other immutable buffer dtypes can be retained as opaque bytes; packed/quantized parameters or unsupported state adapters fail explicitly.

One private Arc-backed group represents each actual declared shared allocation. Logical aliases resolve to that group, not separate equal-valued tensors. Source APIs expose no mutable tensor/manifest/asset access. Fresh state/layout/content checks bind execution to the compiled source rather than trusting a cached parameter name or digest alone.

Candidate maps initially share every group with the source. Only touched groups copy, once per physical group. Logical target names resolve to canonical group tensors before kernel invocation. Copy-byte totals cannot exceed the compiled bound; aliases do not multiply them.

Each step snapshots other group Arc identities without copying numeric data, checks the current group's out-of-region digest before/after, and rejects metadata/layout drift. This prevents changes outside the current instruction even when a broader declared effect would cover them. The final pass verifies every parameter/buffer group, actual bit-level changes, derived effective masks, finite/magnitude invariants, unchanged source, and unchanged assets/config/ties. Buffers cannot change through the initial parameter-only executor.

## Resources and smoke

Execution requires finite verification-byte, smoke-work, and vector-element limits. Verification admission conservatively covers full hashes, final comparisons, and per-step mask scans; the bound is not measured disk traffic or CPU time. Smoke-work reporting is also an admission upper bound. Process RSS, wall time, metadata overhead, and external services remain supervisor/workflow responsibilities.

The initial registered smoke adapter is `nf.linear_fixture.v1` for factory `fixture.cpu`, version 1, empty configuration, and a nonempty floating matrix parameter w. It evaluates real bounded linear forward probes in source precision. Probe inputs, optional expected outputs, tolerances, output magnitude, and suite identity are explicit. Empty/unbound suites, changed suite digests, unknown required adapters, shape failures, non-finite outputs, or failed expected values prevent a candidate result. Unknown models never receive a no-op smoke fallback.

## Receipts and limits

Complete receipts bind plan/source/result identities, full before/after tensor hashes, physical changed-coordinate count, buffer preservation, copy totals, step work, and smoke evidence. Method target-update and final changed-count fields reflect observed execution. Upstream input-history assurance is explicitly declared-by-plan until the full experiment ledger authenticates how operands were produced.

The receipt stage is tensor scope and smoke, with promotion status not evaluated. Independent multi-gym quality, retention, statistical evidence, standalone export, fresh reload parity, and a verified experiment contract remain required for acceptance. Primitive or transaction success cannot substitute for them.

The candidate manifest updates weight hashes and removes the old checkpoint asset reference. It is an in-memory result, not a fictional new checkpoint file. Export later creates the real checkpoint and its final manifest binding. Source checkpoint bytes and source/accepted pointers remain unchanged.

The initial executor rejects nonempty parent-candidate chains until qualified patch-history replay supplies their complete ancestry policy. There are no production fault callbacks. Unit-only fault injection tests aborts after every instruction, out-of-mask/buffer/alias/config/shape/dtype damage, late numeric errors, and smoke failures without changing source state or returning partial candidates.
