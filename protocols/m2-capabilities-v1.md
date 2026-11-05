# M2 capability and gate protocol v1 (T045)

## Scope and compatibility

`nf_core::capabilities::m1_manifest()` is unchanged. Its M1 milestone,
backend metadata, unavailable operations, evidence scope, and promotion status
remain M1 claims. `m2_manifest()` is a separate **source inventory**, not a host
probe, execution permission, scientific result, or replacement for M1.
The Python registration surface is deliberately unchanged by this task.
The standalone checker is the machine-readable M2 execution surface.

M2 is a bounded manual multi-gym research product, not full platform completion.
NeMo Gym, Prime Intellect Verifiers, Gymnasium, autonomous research agents, rich
evolution, general transfer, structural/quantized edits, confidential deployment,
GPU qualification, and serving qualification remain explicitly incomplete.
A supported smaller path does not waive its applicable integrity gates.

## Two versioned records

Both records have integer `schema_version: 1` and `milestone: "M2"`.
The Rust inventory has `kind: "m2_capability_inventory"`; the checker emits
`kind: "m2_gate_report"`. Neither is a promotion credential.

The five distinct `states` fields are:

| Field | Current meaning |
| --- | --- |
| `implemented` | `component_subset`: source components exist; no end-to-end assertion |
| `host_qualified` | Inventory: `not_evaluated`; report: `not_qualified`, or `observed_component_tests_only` after sandbox, trusted-worker, and split suites actually pass on unchanged source |
| `workflow_integrated` | `incomplete`: passing individual suites cannot establish authorized five-gym dispatch |
| `release_enabled` | `false`: mandatory owner-controlled authority prerequisites remain unresolved |
| `scientific_evidence` | `not_established`: software or injected-known-target success is not discovery, mechanism verification, pretrained repair, transfer, or generalization |

A report also contains source digests before/after execution, source stability,
host system/machine/Python identity, checks, explicit release blockers, required
compliance IDs, and missing advanced capabilities. It makes no unmeasured GPU,
OCI image identity, binary freshness, or installed-extension equivalence claim.
`selected_checks_passed` concerns only requested component checks and requires at
least one requested check plus unchanged source. It is not M2 qualification.

## Check execution and evidence

Run with the already prepared local Python environment, for example:

```sh
CARGO_TARGET_DIR=/tmp/nf-m2-check .venv/bin/python scripts/check_m2_release.py --gate capabilities --timeout 90
CARGO_TARGET_DIR=/tmp/nf-m2-check .venv/bin/python scripts/check_m2_release.py --gate runtime_identity --gate imports --gate reports
CARGO_TARGET_DIR=/tmp/nf-m2-check .venv/bin/python scripts/check_m2_release.py --gate isolated_wheel --wheel /absolute/path/to/fresh.whl
.venv/bin/python scripts/check_m2_release.py --require-release
```

Each `--gate` selects a fixed suite, not a user command or historical JSON receipt.
`--gate all` selects every required check. Default/no selection emits missing
execution evidence and exits 1. Selected-check mode exits 0 only for successful
selected checks; `--require-release` independently requires release readiness and
currently always exits 1. Invalid arguments exit 2. JSON is printed to stdout.
There is no flag/environment variable for granting release or scientific success.

Each executed check records the exact command, return code, timeout flag, elapsed
seconds, complete-log hash/size, and a bounded log tail. Python checks count actual
JUnit cases and reject zero cases, collection errors, failures, skips, and xfails;
unexpected passes are strict failures. Native checks select entire integration
test binaries, check discovery totals, and run every exact test separately; zero
or ignored tests fail. Each required Python file must contribute cases, not hide
behind another nonempty file. Each command has a 1..90 second cancellation
deadline; timeout sends SIGKILL to its process group. This trusted test-runner
cleanup is not hostile descendant containment or a physical fsync cutoff.
Cargo is offline. No runtime image,
model, tool, or Python dependency is downloaded by the checker.

The source identity covers relevant source/config/spec/protocol files, ignoring
build/cache products. Source changes during checking invalidate selected success.
CI freezes that identity before building and passes `--expected-source` to every
checker invocation: intervening drift refuses before test execution. Without an
expected identity, only within-invocation source stability is observed.

`runtime_identity` actually imports the installed native extension and measures
its M1 manifest, package/distribution/native/core versions, native binary hash,
installed Python source hashes, and native-backed M1 edit/replay/export fixture.
It compares Cargo/Python source versions and installed Python bytes; any mismatch
fails. Same-version native binaries are NOT proven fresh by this probe. The M1
manifest's legacy architecture-derived `qualified_on_this_host` is preserved,
not promoted to M2 sandbox, runtime, integration or release qualification.
`component_observations` records actual selected M1/M2 suite outcomes and counts,
including intake, portfolio, broker, contracts, independent verification, reports,
supervision and promotion. Source inventory and historical worker receipts never
substitute for these measurements. Unrun capabilities stay `not_run`.

This is local diagnostic evidence, not cryptographic attestation against a hostile
host. Build outputs are not assumed to match source bytes. Freshness stays a
blocking prerequisite even when runtime versions or isolated wheel tests pass.
The isolated-wheel gate requires the existing offline verifier's actual JSON
proof and matching source/runtime versions, not just exit zero. Its scope is
minimum imports and the M1 demo, not M2 integrated execution.

## Mandatory gate mapping

| Gate/check | Required concern, without implying existing integration |
| --- | --- |
| runtime_identity + imports + m1_executor + isolated_wheel | Actual M1 manifest/fixture, version and Python-source agreement, core operations, minimum-product imports without research-stack side effects |
| sandbox + trusted_worker + splits | CG-01 actual hostile-code isolation, trusted-model worker, separate evaluator access domains |
| native_contracts + contracts | CG-03/04 contract authority, hard budgets, stop/recovery and promotion refusals |
| splits | CG-05 three roles, exposure, frozen holdout custody |
| controls + measurements + vertical_slice | CG-06 matched controls, required five-gym metrics and resource evidence |
| tensor_scope + trusted_worker | CG-07/08 undeclared edits, standalone fresh-process reload and parity |
| methods_costs | CG-02/11/12 attribution, method/cost completeness and honest evidence origin |
| reports | CG-14 committed JSON/Markdown report equivalence |
| gym_audit | Standalone deterministic audit component, not model improvement |
| capabilities | Schema, negative checks, M1 preservation and non-release semantics |

All CG-01 through CG-08, CG-11, CG-12, and CG-14 remain required. This mapping is
an obligation list, not proof that each existing test suite fully establishes it.
T043 integrated evidence and owner review must establish the complete chain.

## Release remains fail-closed

The source-controlled prerequisite list retains evidence of T038 bounded work,
cancellation, incomplete outcomes, cleanup disclosure and independent review;
T039 authoritative atomic promotion and reviewed release; T041 integrated
authorized custom-gym dispatch; T043 integrated mixed five-gym evidence; and
fresh wheel/native-build equivalence. `release_prerequisites` lists every required
check plus unresolved owner evidence. These entries mean evidence is not yet
established by this checker, NOT that all corresponding components are absent.
The legacy promotion entry point still denies release without authority. The newer native live-grant operational fixture
promotion and operator-scoped manual dispatch are present components, not absent
implementations. Their fixture-only operational scope does not establish
independent engineering review or mixed five-gym dispatch;
`scope_limits` preserves those distinctions in inventory and reports. A positive
refusal test proves denial, not publication; an actual operational grant test
proves only that explicitly limited path. SDK/CLI planning/status APIs alone are
not execution.

### Timing scope is not a hidden milestone requirement

Approved requirements 2.2/2.5/14.7 and design sections 5/6 require finite budgets,
work admission/cancellation at limits, incomplete unfinished work, and visible
cleanup limitations. Atomic pointer publication remains mandatory. They do NOT
universally require physical SQLite fsync or all cleanup to finish by an exact
wall/CPU cutoff. The report's default `operational_cancellation` scope preserves
that distinction; lack of strict physical publication alone is not an M2 blocker.

A frozen contract explicitly requesting the stronger physical whole-operation
cutoff must still fail preflight on an unqualified backend. The diagnostic option
`--timing-scope strict_physical_publication` refuses before executing any selected
checks and reports the exact unsupported scope. It cannot grant an operational
fallback or revise an existing frozen contract. Do not equate accounted ceilings,
precommit timestamps, watchdog signals, or successful operational fixtures with
strict physical publication. Unknown cleanup/costs and known overruns remain
visible and cannot become passing mandatory evidence.

Removing a prerequisite requires reviewed implementation plus actual integration
and qualification evidence from its owner. No collection count, architecture,
runner label, fixture receipt, copied JSON, benchmark score, environment flag,
or green component suite removes it. No report aggregator can authorize an
accepted pointer. Scientific claims require a separately approved study and
independent positive evidence; this gate never invents them.

## CI and local verification

The M2 workflow uses an explicitly prepared Linux CPU runner. Preparation must
provide pinned project development tools, offline Cargo caches, the exact approved
OCI image/profile, and the trusted CPU reference runtime. The workflow never pulls
an image or downloads dependencies. Missing preparation fails rather than skips.
It builds/installs a fresh wheel in a clean environment and runs the component
checks with the prebuild source identity before the release-readiness step.
The release step is intentionally red while mandatory prerequisites are unresolved; do not make it optional or mark it
successful to obtain a green milestone. Existing M1 CI remains independent.

The Rust inventory tests and `tests/test_m2_capabilities.py` validate the checker.
The positive schema test runs actual installed-extension import tests; synthetic
JUnit and forged-green inputs are labeled adversarial negatives, not acceptance
receipts. Real sandbox/wheel/full workflow qualification must be run separately;
a passing checker unit suite does not establish those capabilities.
