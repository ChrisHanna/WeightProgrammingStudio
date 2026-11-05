# Tensor-set identity and CPU storage registration v1

A tensor-set digest hashes canonical JSON v1 with `schema_version: 1`, `kind: "tensor_set"`, and a `tensors` array sorted by exact tensor name. Each entry contains `name`, `shape`, `dtype`, and `content_digest`. The latter hashes exact contiguous CPU bytes. An empty buffer set has a well-defined digest. Names, shapes, dtypes, and bytes all contribute; raw process addresses never do.

The full model manifest separately records architecture/configuration, assets, usage provenance, buffers, and declared storage groups. Tensor-set identity alone does not establish architecture compatibility or tied storage.

CPU registration takes real borrowed byte slices from a trusted runtime adapter. It checks every parameter and buffer against a validated manifest: exact names/roles, dimensions, dtype, byte length, digest, and parameter-set identity. It imposes explicit tensor-count and observed-byte limits before hashing. The inventory is metadata only and grants no model mutation capability.

Aliases require identical nonempty live storage spans and compatible declared metadata. Equal digests from independent allocations do not establish sharing. Conversely, actually shared storage under different declared groups is rejected. Partial overlaps require a qualified layout adapter and are currently unsupported. Empty slices cannot prove aliasing from pointer equality and cannot qualify a tied-storage claim.

Returned group members are name-sorted and stable across observation order. CPU device tags describe the supported observed path; arbitrary CUDA strings or serialized inventories cannot qualify GPU storage. Verified inventories have no deserialization constructor. Packed/quantized formats remain unsupported by this registration path until a qualified adapter supplies their full policy.

Scientific evidence for a circuit remains separate from these storage and integrity checks.
