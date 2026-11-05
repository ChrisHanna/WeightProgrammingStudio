# Bounded core properties and CPU differential checks v1

T019 adds genuine finite-domain numerical and parser tests. These establish software behavior; they do not prove model quality, research gains or universal mathematical correctness.

## Finite native properties

`cargo test --workspace core_properties` executes three tests:

- 2450 f32/f64 shape/rectangle cases: every two-dimensional shape from 0..4 and every contained rectangle, including empty boundaries. Real scale writes match independent coordinates; outside bytes remain identical; a smaller actual-change budget rejects atomically.
- 874 permutations: every bijection for axis lengths 0..6 on three-row tensors. Native permutation agrees with direct indexing, inversion restores exact bytes including signed zero, and sequential permutation equals independently composed indexing.
- 4096 deterministic mutations of JSON, Weight IR, frozen search spaces and actual standard Safetensors. Successful cases retain round-trip canonical/program/space identities and actual checkpoint hashes. Errors are explicit. Every parser family must accept at least one valid seed so an all-error implementation cannot pass.

Numerical tensor/property sizes and case counts are fixed. Parser input is capped at 8192 bytes, generic JSON at 4096 nodes/depth 32, tensor files at 16 tensors/rank 8/header 2048 bytes. The smoke loop checks a ten-second wall deadline. Its standalone process can additionally run under 256 MiB address-space and ten-second CPU limits:

```sh
CARGO_BUILD_JOBS=2 cargo build -p nf-weight-ir --example parser_fuzz_smoke
(ulimit -v 262144; ulimit -t 10; ./target/debug/examples/parser_fuzz_smoke)
```

This is reproducible mutation smoke, not a coverage-guided fuzz campaign or a substitute for longer sanitizer/fuzzer runs. Deadline checks occur between bounded parser calls. Process address-space/CPU limits apply to the standalone native fuzz target, not the compiler or Python/Torch test process.

The initial bounded process run completed all 4096 cases: canonical JSON 322 successes, IR 4, search spaces 6 and checkpoints 11. No panic or identity mismatch occurred; success counts are not code-coverage measurements.

## Real PyTorch reference

The optional `differential` extra pins `torch==2.9.1+cpu`. Install through the explicit CPU index (not the default CUDA-capable distribution) with the project interpreter. uv setup used here:

```sh
UV_CACHE_DIR="$KIROCREW_SCRATCH/uv-cache" UV_LINK_MODE=copy \
  uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cpu 'torch==2.9.1+cpu'
.venv/bin/python -m pytest -q -k cpu_differential \
  --basetemp="$KIROCREW_SCRATCH/pytest-differential"
```

No model package/checkpoint is downloaded. Tests construct tiny independent CPU tensors, run under no_grad, and assert no gradient-dependent state. The extension sees only bytes/records, never a Torch object or callback. A missing Torch installation fails collection rather than skipping qualification. This framework is optional runtime tooling, never imported by the SDK itself; installed-package smoke tests verify no research-stack import.

Twenty-two cases compare native f32/f64 replace, add, scale, U8 mask, both permutation axes, left/right linear maps, left/right projection subtraction and rank-one updates against actual PyTorch operations. Two more tests enumerate every nonempty rectangle of a 2x2 matrix in each dtype, compare sliced Torch operations and untouched coordinates, and preserve the original source. All edits pass the real compiler, transaction, alias, numeric and smoke gates. Tolerances are fixed at absolute 1e-6 for f32/1e-12 for f64 with zero relative tolerance; rectangle checks are exact. No-op metadata or mocks cannot satisfy these cases. NumPy conversion is not used; the reference works without NumPy despite Torch's import-time optional-NumPy warning.

The compiler requires explicit matching write effects for each target, including ties. Rectangle fixtures narrow their declared effective alias regions before compilation; the domain remains a fixed whole-tensor allowed site. Failures are not worked around by weakening source, mask, dtype, or permission checks.
