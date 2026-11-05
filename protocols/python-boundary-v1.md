# PyO3 control boundary v1

`neural_foundry._native` is the installed abi3 extension. Python supplies bounded owned data; Rust validates, compiles and executes. No Python implementation substitutes for model integrity, parameter identities, search-space admission, numerical kernels or scope checks.

## Handles and submission

SourceHandle, PlanHandle, CandidateHandle and RecordHandle are frozen opaque native classes with no Python constructors, dictionaries, mutable views or deserializers. A plan captures its exact immutable SourceState; apply cannot substitute another source. Plans can be shared across threads; each execution uses the core's isolated copy-on-write transaction. Returned tensor data and JSON are immutable copied Python bytes, never storage views. Candidate snapshots serialize and reload genuine standard checkpoint bytes and retain unaccepted status.

The native API accepts real Python bytes and native dicts containing string names and bytes values, not arbitrary conversion objects. It never calls `__bytes__`, a Python numeric callback, pickle, eval, import hooks, or user model methods. Asset/operand maps are scanned and bounded before copying. Metadata is at most 1 MiB, checkpoint and operand pools at most 64 MiB, assets at most 4 MiB, entries at most 1024, and names at most 512 UTF-8 bytes. Core compiler/load/kernel budgets still apply independently. Malformed JSON, duplicate/unknown fields, unknown versions and inconsistent records fail through the same Rust parsers as CLI submission.

`validate_record` supports program, search_space, manifest, compile_policy and smoke_suite. Its canonical JSON and identity come from the validated typed Rust record, not a second Python hashing algorithm. Validation is not authorization or execution: a validated record cannot be passed as a plan or candidate.

## GIL, errors and cancellation

Heavy parsing, hashing, source decoding, compilation, transaction work and snapshot serialization run with the GIL released. Workers capture only owned Rust data and Arc handles. There are no Python callbacks in those closures. A thread-local guard denies nested worker reentry and resets on success, error or unwind. NativeError carries stable `code`, `path` and `message`; CancelledError and NativePanicError are typed subclasses. Wrong Python argument types are ordinary TypeError before core submission.

A catch_unwind boundary converts recoverable Rust panics into INTERNAL_PANIC without returning a partial candidate. It cannot catch process aborts, allocator termination or external kill signals; finite process-level supervision remains necessary. The error deliberately does not expose a panic payload as user data. Python signal checks run after reacquiring the GIL.

CancellationHandle contains a sticky shared AtomicBool. It cannot reset a stopped attempt. Requests are checked before copying submission data, before/after worker operations, before/after each bounded instruction, every 4096 final-verifier cells, around smoke and before returning a candidate. Cancelled work drops its isolated state and returns TRANSACTION_ABORTED, never a partial success. An individual kernel or opaque library parse is not forcibly interrupted; its admitted work/byte cap bounds that interval. Cancellation racing after a completed return does not revoke an already returned immutable result. This boundary provides no checkpoint publication or active-reference mutation API.

## Tests and build

Rust tests verify real GIL release by another Python-attached thread, typed Python exceptions, panic containment/reentry cleanup, sticky cancellation and cancellation after actual instructions without source changes. Python tests use the installed extension, real Safetensors bytes, true reconstructed ties, native compile/scale/smoke/snapshot, concurrent shared-plan isolation, denied metadata/permissions and rejection of forged handles/conversion callbacks. Fixtures establish software behavior, not model research gains.

The project Cargo config defaults PYO3_PYTHON to the project's .venv interpreter so embedded tests do not bind an unrelated inherited Conda ABI. An explicit caller/Maturin PYO3_PYTHON overrides that default. Native CLI/core builds remain independent of Python; embedded boundary tests require the configured development interpreter/library. Unset the inherited CONDA_PREFIX only for the existing Maturin develop invocation if it conflicts with VIRTUAL_ENV.
