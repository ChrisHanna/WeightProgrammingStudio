# WeightRepairBench v1 — manual component profile (T044)

## Authority and compatibility

Protocol `weight-repair-benchmark-v1`, schema version 1. This implementation is
an arithmetic development-fixture profile, not the general scientific benchmark
or a released manual study. Its scope is
`synthetic_component_fixture_not_scientific_acceptance`. Unknown fields,
versions, duplicate JSON keys, malformed SHA-256 identities, nonfinite numbers,
incomplete panels, missing evidence references, and accounting inconsistencies
are rejected by Rust. Rust parses bounded canonical JSON (1 MiB, depth 64).
Record validation establishes structural consistency, NOT receipt authenticity,
execution custody, sandbox qualification, independent final evidence, or acceptance.

Task bindings include source parameter and model manifest digests, five exact gym
identities, all three distinct split identities, runtime/platform identity, seed,
permissions, finite bounds, frozen compiled-plan identities, and comparison panel.
A changed task requires a new canonical task digest on every result/attempt.
Programs are native compiled-plan digests, not editable labels. v1 supports only
inspect/direct_edit permission; it never grants network, fitting, or training.

Failure strata are `injected_known`, `natural`, and `adversarial`. Injected-known
requires a ground-truth identity and exposure disclosure. Natural records cannot
claim planted ground truth. The runnable arithmetic fixture is explicitly
injected-known; changing its result to a natural/adversarial claim is rejected.
Generic task-record support for natural failures is NOT a natural-failure runner.

## Required comparisons and evidence

All ten method slots are mandatory: original, no_op, best_known, direct, decoding,
activation, lora, sft, distillation, rl. The fixture additionally preserves the
`direct_failed` attempt. No fallback may replace or relabel that failed method.
Executed comparisons use the same source, development cases, five arithmetic
oracles and tolerance. No-op actually compiles/applies factor 1 and must preserve
exact parameter identity and original scores. Best-known executes the independently
constructed reference fixture. Direct factor 2 and a genuinely changing failed
factor 0.5 are supplied known-target edits, not discovered repairs. The native IR
compiler and surgeon enforce parameter scope, effects and tensor/smoke checks.
Original/no-op correctly report failed behavior on the corrupted sum, not a
failure to execute. The direct behavioral result is derived from all five heads.

Each executed arm requires observations/source-integrity digests and complete
scores; edited arms require the frozen plan and apply-receipt digest. The Python
bundle retains digest-addressed JSON evidence documents, including real per-case
inputs/outputs, aggregates, plan summaries, native apply receipts and source
identity checks. These are replay/debug documents, never authenticated coordinator
receipts. Retained inputs are public development data, not protected holdout.
Standalone export, fresh-process reload and coordinator controls are NOT executed
by this component. Its no-op is therefore not a substitute for the full matched
apply/export/reload control protocol.

Decoding/activation are unsupported because this scalar model has neither a token
decoder nor a qualified activation intervention backend. They are separate
non-weight/runtime methods, never encoded as direct edits. LoRA/SFT/distillation/RL
are explicitly not_run because no training is authorized. Unrun slots have null
candidate/program, empty evidence/scores/costs and a reason; no invented zero costs
or fabricated scores. Localization and mechanism remain independently
not_evaluated even when behavior passes. No localization accuracy is inferred from
the supplied known repair coordinate.

## Cost and resource accounting

Every executed operation and shared preparation record has a unique operation ID,
scope (attempt ID or shared), phase list, evidence identity and complete quantity
map. Units: wall/CPU/GPU in seconds; host/device peaks and I/O/storage in bytes;
local/remote inference in tokens; quoted/reserved/actual spend in USD; forward,
backward, optimizer/fitting, inspected/changed/trainable-parameter and proposal
counts in count. Known values are finite and nonnegative; integral quantities
must be integers. Unknown/unavailable values are null with a reason, never zero.
No unknown-to-zero conversion or shared operation charged twice is allowed.
Peak memory is not additive; quoted/reserved spend is not actual spend. This
profile intentionally implements NO total-efficiency ranking or amortization.

The fixture measures controller wall/process CPU intervals and dispatched forward
calls including apply smoke probes. It explicitly counts no backward/training,
agent inference, network spend or GPU work in its evaluation arms. Custom trusted
preparation is not instrumented for those operations: its counters remain unknown.
Shared setup/preparation is allocated once, not repeated in each arm. Parameter
inspection and I/O/peak counters not fully instrumented remain unknown, including
for failed edits. No tensor-write-only efficiency claim is permitted. Phase
coverage must agree with cost records for setup/diagnosis/search/controls/failed
edits/verification; export/reload/recovery are not_run in the provided runner.

Total budget verdict is always incomplete: preparation counters and some resources
are unknown. Known arm+shared wall/forward subtotals must still fit bounds; this
partial check is not a full budget proof. Maximum 289 cases, 16 attempt slots,
10,000 forwards and 60 cooperative wall seconds are permitted. Native IR has
additional finite tensor/work bounds. Wall checks occur after operations: there
is NO hard process/CPU sandbox deadline claim. Bound the containing process;
trusted prepare callbacks are executable controller code, never untrusted plugins.
The 17 seed is a fixed protocol identity; no stochastic search/training is run.

## Runnable fixture and individually bounded checks

From the project root, with the rebuilt native extension including the registered
benchmark validator (the runner refuses an older unregistered extension):

```python
from neural_foundry.repair_benchmark import run_fixture
bundle = run_fixture()
# bundle has task, result, evidence_documents; no implicit filesystem export.
```

Run individual Python tests with a containing-process timeout of 150 seconds:
`PYTHONPATH=python:tests .venv/bin/python -m pytest -q tests/test_repair_benchmark.py::test_repair_benchmark_actual_records_checked_by_rust_library`
The test runs the real fixture, then passes its records to the authoritative Rust
checker and checks rejection of a missing observation reference. It runs Cargo
offline with one job and subprocess deadlines (120 seconds first, 30 second retry).
Other Python cases cover actual controls/repair/retention/failure, cost unknowns,
missing/false plans, native canonical identity (including integer budgets), native
error metadata, duplicate keys, oversized input, stale task binding, method labels,
original/direct parameter identities, missing required evidence, fabricated unrun
costs, holdout relabeling, and the absent-registration refusal.

Dedicated Rust tests import the public `nf_experiments::repair_benchmark` module,
the same authority used by the registered PyO3 endpoint. Execute each named case separately with a
120-second containing-process deadline, for example:
`CARGO_TARGET_DIR=/tmp/nf-t044-complete cargo test --offline -j 1 -p nf-experiments --test repair_benchmark repair_benchmark_units_unknowns_shared_double_count_and_budget -- --exact`
The environment-driven `repair_benchmark_actual_python_record_when_supplied` case
only validates actual Python execution when `NF_REPAIR_BENCH_RECORD` is supplied;
an ordinary Cargo invocation without it does NOT establish cross-language execution.

## Registered Rust authority and SDK boundary

`nf_experiments::repair_benchmark` is a public module. The PyO3 function
`validate_repair_benchmark(task_json: bytes, result_json: bytes | None = None) -> bytes`
copies bounded caller bytes before releasing the GIL and uses the standard
worker/panic/error translation boundary. It calls `Task::parse` or
`ResultRecord::parse` and returns the checked original document canonicalized by
nf-core. It does NOT reserialize typed f64 fields: doing that would change integer
budget tokens and break content identities. Invalid records raise NativeError,
not an accepted boolean. The SDK and native stub expose the registered function;
`repair_benchmark.validate(task, result=None)` never downgrades to Python-only
acceptance. `run_fixture` validates its task before comparisons and its complete
result before returning. Original/no-op must preserve the exact source identity;
direct and failed-direct must have distinct changed candidates and frozen plans.
Decoding/activation are unsupported runtime slots; training slots must be not_run.

Record validation checks required evidence references, not the authenticity or
truth of arbitrary caller-supplied documents. The runnable fixture retains real
native receipts and observations and tests their digest identities. No serialized
benchmark record confers coordinator capability or scientific acceptance.

No CLI registration is required for this component runner. Full workflow linkage remains
`unavailable_pending_authorized_workflow`: consume released T038/T039/T041
contract, controls, split custody, final evaluator, cost accounting and promotion
interfaces before creating a workflow-backed profile. Never convert these
component documents into final study receipts or weaken publication/reload gates.
