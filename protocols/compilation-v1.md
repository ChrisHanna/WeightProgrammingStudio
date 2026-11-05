# Same-shape compilation v1

`nf_weight_ir::compilation::compile` is a pure planning operation. It accepts a validated program, frozen compile policy, resolved transformation space, real Safetensors operand bytes, and a source snapshot consisting of real checkpoint bytes, inert assets, a validated manifest, and fresh borrowed CPU observations. It performs no filesystem/network access, inference, training, or checkpoint writes.

## Inputs and trust boundary

The coordinator or standalone model owner supplies the policy. Policy records are not execution leases. The artifact broker remains responsible for authenticating data origins and resolving only authorized bytes; bytes alone cannot prove that their declared provenance is true. Compile-time access checks cannot replace holdout isolation or registry authorization.

The compiler hashes the actual source checkpoint, verifies its Safetensors layout, re-registers observed parameter/buffer bytes and real aliases, and checks configuration/tokenizer/declared asset identities. Every storage group needs coverage in the checkpoint. Tied weights may have one canonical file entry, while the trusted loader reconstructs the declared runtime sharing. Fresh tensor preconditions and same-shape layout must match; values from a valid committed parent can differ, but names, shapes, dtypes, roles, device, and ties cannot change silently.

Exact contract, source, objective, protocol, search-space, and mandatory gym/scorer/split/validation/smoke/parity identities are frozen by the policy. Extra gyms are allowed only when their complete bindings were preapproved. Gradient-assisted inspection needs its separate grant; fitted lineage needs training permission and cannot bypass a direct-only space.

The registered lowering targets are `nf.native_cpu` and `nf.trusted_torch_cpu`, version `1`, with explicitly supported f32/f64 target and accumulation profiles. The latter requires a separate live T032 runtime and T033 paired-execution witness; a native reference execution alone is not Torch qualification. See `torch-edit-backend-v1.md`. Target f64 cannot silently accumulate in f32. Declared capabilities are limited to the supported CPU vocabulary. This is a compilation profile, not a claim that execution kernels or a real serving backend have already qualified.

## Artifact and operation checks

Operand files are immutable `Arc<[u8]>` objects identified by exact hashes. Safetensors metadata, sizes, dtypes, shapes, and named tensors are checked against the descriptors. Unsupported types, NaN/infinity, implicit casts, invalid binary masks, incompatible matrices/factors, and missing blobs fail before a plan is returned. Mutating a caller's Arc through copy-on-write does not alter the retained plan bytes.

Projection basis dimensions, rank, allowed side, frozen orthonormal tolerance, and actual Gram products are checked with the selected accumulation precision and a finite scalar-work limit. Constructors without registered lowering are rejected. Parsing support is not constructor implementation.

The compiler derives effective reads/writes from the actual instructions and storage aliases. Every effective alias needs explicit valid preconditions and effect coverage. A conservative declared region cannot expand the actual execution mask. Buffer mutation, graph surgery, configuration changes, and tokenizer changes need separately qualified adapters and are unsupported in this initial executor path.

## Allocation and work plan

The report distinguishes unique retained operand bytes, full touched-storage candidate copies, tracking-bit storage, peak kernel scratch, output tensor-data bytes, validated input bytes, scalar validation work, and kernel work units. Bounds use checked arithmetic and are enforced before execution. Matrix kernels have explicit region/result workspace bounds; projection additionally reserves coefficient workspace.

`estimated_working_bytes` covers these retained operand and numeric execution buffers. It is not whole-process RSS, allocator overhead, the caller's existing source buffers, or a full checkpoint/export-file size. Metadata has separate byte/node/cardinality admission bounds. Output tensor-byte limits exclude headers/sidecars; export and worker supervisors enforce their own storage and process-memory limits. Logical input-byte and arithmetic counters are not measured disk traffic, GPU time, or total research cost.

The executor must stay within the plan's declared buffers and recheck source identity and numerical results at application time. A source may change after compilation, so a valid plan is not permission to mutate a stale model.

## Outputs and preservation

`CompiledPlan` has private fields and no deserializer. It binds program, policy, source, compiler/profile versions, search space, effective masks, operand bytes, and allocation/work bounds. Its read-only summary can be serialized for review, but a serialized summary cannot become an executable plan. Recompilation is required to recover a plan from records.

The initial checked exact rule is `identity_noop` v1: no instructions or writes and an explicit condition on the actual source manifest. Unsupported exact-preservation rules fail closed. Bounded-error claims remain pending under the frozen verification protocol; experimental claims assert no equivalence. Quality constraints always remain pending independent measurement. None of these static reports promotes a checkpoint or establishes a scientific gain.
