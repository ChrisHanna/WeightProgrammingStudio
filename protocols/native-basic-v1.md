# Native CPU basic edits v1

`nf_surgeon::basic` supplies replace, add-delta, scale, and binary-mask arithmetic on caller-owned contiguous finite f32/f64 tensors. These functions are not checkpoint loaders, experiment gates, or permission brokers. The transaction driver must create isolated candidates, resolve authorized aliases to canonical storage, and verify compiled source/effect facts before calling them.

Tensor names, shape products, byte-compatible dtype, slice starts/widths, and immutable input lengths are checked. Regions use row-major coordinates, including scalars, empty dimensions, and boundary slices. Numeric operands must match target dtype and region shape; there is no broadcasting or implicit cast. Masks are shape-matched u8 zero/one values. Inputs are immutable typed slices or retained `VerifiedOperand` values, never dynamic callbacks.

Numerical policy explicitly chooses f32/f64 accumulation, nearest-ties-even output rounding, non-finite rejection, and finite magnitude bounds. f64 targets cannot silently accumulate in f32. An input may be finite yet overflow in arithmetic or when rounded to the target dtype; that outcome is rejected.

The first pass validates every result and computes actual changed-coordinate count before the first write. The second pass applies the same deterministic scalar computation with stable inputs and unique coordinates. No per-coordinate vectors, full edit previews, or operand copies are allocated. Mask validation is one extra linear pass. Compiler work admission reflects two cell visits for replace/add/scale and three for mask; these are not CPU-time estimates.

Changed-coordinate counts compare target bits, so signed-zero flips count and equal-value writes do not. This is per primitive, not a count of distinct changes across an entire multi-step transaction. The transaction layer measures global scope and final/cumulative effects separately.

On shape, dtype, mask, non-finite, magnitude, or changed-coordinate budget failure, all original tensor bytes remain intact. These guarantees rely on Rust's immutable-slice ownership rules; unsafe external code that violates those rules is outside the API's trust boundary.

Copy-on-write source groups, immutable checkpoint files, whole-candidate smoke tests, alias receipts, and acceptance remain the responsibilities of later transaction/export/verification stages. Primitive success alone does not establish a model improvement.
