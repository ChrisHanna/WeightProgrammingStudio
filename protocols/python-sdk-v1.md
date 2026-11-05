# Python core SDK and cross-language conformance v1

`neural_foundry` exposes lightweight record helpers plus `ir` and `surgeon` modules. Import starts no models, studies, network clients or research stack. Existing version/workspace APIs remain available. NativeError, CancelledError, NativePanicError and CancellationHandle are aliases of the actual extension types, not Python replacements.

## Record and version control

`validate(kind, value)` accepts JSON bytes, an ordinary caller-owned mapping, or an opaque RecordHandle. Mappings are encoded to JSON; native Rust then parses, validates and canonicalizes the typed record. Handles are immutable and carry canonical JSON/digest, not authority to execute an edit. Using a different record kind reruns that kind's Rust checks. The SDK has no SHA or JSON canonicalization implementation; `canonicalize` and `content_digest` delegate to native bounded Rust helpers. Raw byte digests and canonical typed record identities remain distinct.

`negotiate(required=None)` sends an exact, bounded version request to `nf_core::protocol`. Default requirements are core_api, canonical_json, weight_ir, opset and search_space version 1. Response includes actual core version, explicitly agreed versions, and the supported protocol table. Unknown protocols/versions fail; there is no nearest-version downgrade. An empty requirement map fails instead of falsely confirming compatibility. This handshake is protocol compatibility, not a model/backend capability or scientific acceptance claim. Every facade operation negotiates the current SDK's requirements before submission.

Python mapping conversion is convenience for a caller already running Python. It is not a sandbox and its Python allocation/conversion cost is outside native byte admission. Arbitrary objects unsupported by the standard JSON encoder raise ordinary TypeError; raw bytes preserve exact wire failures through Rust. Non-finite JSON literals reach the Rust parser and cannot become valid weights. All native limits, type checks, owned-data/GIL restrictions and cancellation behavior from python-boundary-v1 still apply. Untrusted gym code must not receive SDK/model handles.

## IR and surgery

```
source = neural_foundry.surgeon.load(manifest, checkpoint_bytes, assets)
program = neural_foundry.ir.program(program_record)
space = neural_foundry.ir.search_space(space_record)
plan = neural_foundry.ir.compile(source, program, space, compile_policy, operands)
candidate = neural_foundry.surgeon.apply(plan, smoke_suite)
```

Inputs may be mappings/record handles/JSON bytes; output source/plan/candidate handles are genuine opaque native objects. Compile is a pure checked plan, not an edit or model forward. Apply uses the same isolated native transaction/smoke gate as CLI. `candidate.snapshot()` materializes and reloads real checkpoint bytes. No source mutation, mutable tensor view, callback, Python numerical fallback, implicit permission grant, or promotion API is added. Exported CLI and SDK results retain unaccepted core statuses.

## Shared conformance

The CLI also exposes `nf json canonicalize JSON`, `nf blob digest FILE`, and `nf protocol negotiate REQUEST.json`. These call the same Rust implementations as the extension. Tests reuse canonical-v1-vectors.json and error-v1-vectors.json in native Rust and Python/CLI; expected identities, error codes, JSON pointer paths and messages are shared evidence, not recomputed Python reference logic.

Conformance tests compare actual CLI/extension/SDK program, search-space, source and compiled-plan identities, and complete candidate receipts after a real parameter edit. They reopen genuine on-disk standalone source fixtures with the native store, apply/export through CLI, and compare the SDK's numerical results/source preservation. Missing or stale native binaries fail rather than skip; build with `cargo build -p nf-cli` before standalone Python conformance tests. Pytest temporary roots must be under the session scratch directory. These tests establish software agreement only, not target-quality or retention gains.
