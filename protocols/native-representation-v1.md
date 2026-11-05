# Native CPU representation edits v1

`nf_surgeon::representation::apply_representation` applies typed direct edits to one caller-owned contiguous finite f32/f64 candidate region. It does not load checkpoints, run a training optimizer, publish accepted models, or infer a circuit's function.

## Semantics

| Operation | Exact declared computation |
| --- | --- |
| Axis permutation | New coordinate i reads old coordinate permutation[i] on the selected region axis |
| Left linear map | W_new = A W, with square A matching the region's row dimension |
| Right linear map | W_new = W A, with square A matching its column dimension |
| Low-rank update | W_new = W + strength L R, with explicit positive compatible rank |
| Left subspace projection | W_new = W - strength U (U-transpose W) |
| Right subspace projection | W_new = W - strength (W U) U-transpose |

Matrix operations require rank-two regions. Permutations support arbitrary supported rank and explicit bijections. Inputs are immutable typed arrays or compiled verified operand bytes, not callbacks or Python expressions. Shape/dtype mismatches are rejected rather than broadcast or cast implicitly.

Projection checks finite basis values, rank, allowed side/dimension, and U-transpose U against identity under the declared accumulation precision and tolerance. The standalone numeric tolerance is nonnegative and below 0.01; compiled programs are also subject to their tighter frozen tolerance. Strength zero does not bypass invalid-input checks.

Accumulation and output rounding use the same explicit numerical policy as basic kernels. Source regions and operands remain unchanged until every result, magnitude, and bit-based changed-coordinate budget passes. Non-finite intermediates or final values cause rejection. Scratch allocation and finite work bounds are checked before numeric buffers or expensive validation loops are created.

## Shared resource admission

Compiler and kernels call `nf_weight_ir::cost::representation_cost`. Its bounds include input checks, copies, arithmetic, final validation, and commit. Work units are conservative algorithmic bounds, not measured CPU time. Permutation validation uses bounded quadratic comparisons without a separate seen-set allocation.

Permutation scratch is one region-sized target-dtype buffer. Matrix operations use two region-sized accumulation-dtype buffers; projection also uses rank times the opposite-axis length in coefficients. No Vec-of-enum scalar buffers inflate these bounds. Receipts report requested numeric scratch bytes, not allocator overhead, metadata, source buffers, or process RSS. Worker-level memory limits remain a separate guarantee.

Empty regions validate their input contracts and then return without potentially enormous empty outer loops. Checked products guard shape, workspace, and work overflow.

## Coupling and evidence

A permutation can be paired with the inverse downstream axis permutation to preserve a known computation. Tests verify this using distinct upstream/downstream tensors and restore their bytes with inverse permutations. Individual kernels do not establish arbitrary model equivalence or atomically commit multiple model tensors. The transactional driver applies the coupled instruction sequence to isolated candidate groups and publishes only a fully verified result.

Atomic single-kernel behavior, toy projection correctness, and known coupled examples are software evidence. Generalization, retention, circuit explanations, transfer, and real inference improvements still require independent evaluation and checkpoint verification.
