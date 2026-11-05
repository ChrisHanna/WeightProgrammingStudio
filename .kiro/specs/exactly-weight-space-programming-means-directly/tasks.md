# Tasks: Neural Programming Foundry

## Scope and execution rules

This is the ordered implementation plan for the approved design at `/mnt/c/Users/NotAPC/Desktop/weightspaceprogramming/.kiro/specs/exactly-weight-space-programming-means-directly/design.md` and original requirements at `/mnt/c/Users/NotAPC/Desktop/weightspaceprogramming/.kiro/specs/exactly-weight-space-programming-means-directly/requirements.md`.

Work in `/mnt/c/Users/NotAPC/Desktop/weightspaceprogramming`. The repository currently contains specifications, not an implementation. Implementation follows the absolute monorepo locations in design Section 3: Rust crates under `/mnt/c/Users/NotAPC/Desktop/weightspaceprogramming/crates`, the SDK under `/mnt/c/Users/NotAPC/Desktop/weightspaceprogramming/python/neural_foundry`, and public protocol assets under `/mnt/c/Users/NotAPC/Desktop/weightspaceprogramming/protocols`.

- Execute T001–T120 in order. Each task depends on prior tasks and includes its own tests. Check a task only after its implementation and applicable verification pass.
- Numeric references identify acceptance criteria in the requirements document. `F2-*`, `OSS-*`, `GYM-*`, and `MRP-*` identify approved additions in design Section 22. `CG-*` identifies the compliance gates in design Section 27.
- Commands below are planned test targets, not claims that tests already exist. Implement the corresponding tests with the task. Selectors must collect tests; zero collected tests, mocks of an advertised real integration, and skipped mandatory qualification do not establish completion.
- M1–M2 are the minimum research product. M3 adds autonomous research. M4 and M5 do not become hidden dependencies of earlier milestone examples.
- Rust owns authoritative contracts, identity, Weight IR, effects, execution plans, persistence, and scheduling. Python owns research/model/gym adapters and analyses. Python cannot bypass core checks.
- Generated gyms default off. No in-process execution of untrusted Python, live torch/model handles in untrusted gyms, or unrestricted subprocess fallback is permitted.
- Non-writing backward attribution is logged inspection. Direct parameter edits using it must say gradient-assisted inspection; strict fully gradient-free profiles forbid it. Gradient-derived update values or fitted adaptations are hybrid/trained.
- Fixture tests are local, finite, and synthetic. No task authorizes a model download, paid call, cloud job, full-size study, or non-fixture training. Build approval is distinct from approval of a frozen research study.
- Check resource headroom before large builds/tests. Use targeted tests first. Optional dependencies and real hardware qualification have explicit capability states; missing prerequisites cannot silently qualify a feature.
- Code readiness, milestone capability qualification, and scientific success are separate statuses. In particular, flagship discovery, cross-model predictive value, and measured backend improvement cannot be marked successful merely because a runner exists.

## Milestone map

| Milestone | Tasks | Usable result |
| --- | --- | --- |
| M1 | T001–T020 | Trusted native edits and replay |
| M2 | T021–T045 | Manual user-gym experiments and independent verification |
| M3 | T046–T063 | Bounded autonomous weight programmer |
| M4 | T064–T099 | Advanced engines and cross-model research |
| M5 | T100–T120 | Architecture surgery and real-backend optimization |

## M1 — Trusted Weight IR and Rust execution

- [x] **T001 — Scaffold the Rust workspace and Python build.** Create the nine designed crates, dependency direction, Maturin/PyO3 package, test configuration, and offline CLI/package smoke entry points. Pin compatible toolchains/dependencies; keep Go and GPU dependencies optional.
  - Verify: `cargo check --workspace`; `python -m pytest -q -k package_smoke` after building the extension.
  - _Requirements: 13.1, 13.5; OSS-01, OSS-06, OSS-09; MRP-01._

- [x] **T002 — Implement canonical records, digests, and typed errors.** Add strict Rust identity encoding with finite numeric rules, duplicate/unknown-field rejection, schema versions, and stable error locations/codes. Add shared golden vectors for later Python conformance.
  - Verify: `cargo test -p nf-core canonical`; include overflow, duplicate-key, Unicode, and non-finite cases.
  - _Requirements: 2.3, 13.1, 13.4; OSS-02._

- [x] **T003 — Implement core model, permission, and provenance schemas.** Define manifests, access classes, method axes, operation authorization, evidence origin/support fields, and bounded capability descriptors. Reject unknown or contradictory permissions.
  - Verify: `cargo test -p nf-core schemas`; round-trip all enums and invalid combinations.
  - _Requirements: 1.2–1.4, 4.5, 14.2, 14.5; F2-01; CG-02, CG-12._

- [x] **T004 — Implement the content-addressed artifact store.** Add staged writes, verified publication, explicit storage roots/quotas, read-only source snapshots, and authorized reference resolution. Reject symlink/path escapes and mutable source aliases.
  - Verify: `cargo test -p nf-checkpoint artifact_store`; inject corrupt blobs, traversal, quota exhaustion, and interrupted writes.
  - _Requirements: 5.2, 5.6, 13.5, 14.1; OSS-04._

- [x] **T005 — Implement safe single-file Safetensors inspection.** Validate headers, offsets, checked shape arithmetic, lengths, supported dtypes, names, and inert architecture assets without pickle or remote code.
  - Verify: `cargo test -p nf-checkpoint safetensors_reader`; include hostile metadata and truncated payloads.
  - _Requirements: 3.1, 5.2–5.3, 14.1; OSS-04; CG-07._

- [x] **T006 — Implement sharded checkpoint processing.** Validate indexes and shard completeness, stream/hash bounded chunks, and stage only affected shards. Record I/O and allocations; detect source changes before publication.
  - Verify: `cargo test -p nf-checkpoint sharded_io`; test missing/extra shards, bounded buffers, and changing sources.
  - _Requirements: 5.2, 5.6, 11.4, 13.4; OSS-04._

- [x] **T007 — Implement tensor inventory and alias contracts.** Describe parameters, buffers, canonical names, shapes, dtypes, storage groups, and declared graph signatures. Verify real sharing rather than inferring it from equal values.
  - Verify: `cargo test -p nf-graph tensor_inventory`; cover tied versus equal-but-independent tensors and unknown packing.
  - _Requirements: 3.1, 5.2–5.4; OSS-03; CG-07._

- [x] **T008 — Implement Weight IR parsing and operation schemas.** Add versioned typed instructions, immutable operands, effects, preservation classes, verification references, and strict extension rules. Unsupported operations must fail closed.
  - Verify: `cargo test -p nf-weight-ir parsing`; round-trip valid programs and reject semantic/version mismatches.
  - _Requirements: 4.1–4.4, 5.1; OSS-02._

- [x] **T009 — Implement Transformation Search Space validation.** Add allowed operations/sites, operand domains, directness and shape policies, cumulative effect limits, and strategy identities. Resolve selectors against the source inventory.
  - Verify: `cargo test -p nf-weight-ir search_space`; reject valid IR outside its permitted exploration domain.
  - _Requirements: 2.1, 4.4; MRP-03; CG-03._

- [x] **T010 — Implement same-shape IR compilation.** Check axes, dtypes, aliases, effect masks, bounded allocations, preservation preconditions, and search-space membership. Emit immutable dry-run plans with no inference/network/write effects.
  - Verify: `cargo test -p nf-weight-ir compilation`; include unresolved bounds, budget overrun, and stale source signatures.
  - _Requirements: 4.2–4.4, 5.2–5.4; OSS-02; MRP-03._

- [x] **T011 — Implement basic native CPU tensor operations.** Execute replace/add/scale/mask on declared f32/f64 regions with explicit rounding and finite-value checks. Count actual changed coordinates rather than patch operand size.
  - Verify: `cargo test -p nf-surgeon basic_ops`; include boundary slices, no-ops, invalid casts, and NaN/infinity.
  - _Requirements: 5.1–5.5; OSS-02; CG-07._

- [x] **T012 — Implement native representation operations.** Add bounded projection, low-rank update, linear map/rotation, and coupled permutation operations with checked dimensions and declared numerical policies.
  - Verify: `cargo test -p nf-surgeon representation_ops`; test projection conditions and downstream inverse permutations.
  - _Requirements: 4.2, 5.1–5.4; OSS-02; CG-07._

- [x] **T013 — Implement transactional surgeon execution.** Apply only compiled plans to isolated state; verify full parameters/buffers and actual write masks; publish complete receipts only. Reject alias escapes, undeclared drift, instability, and stale inputs.
  - Verify: `cargo test -p nf-surgeon transaction`; inject faults after each operation and preserve source identity.
  - _Requirements: 5.2–5.6, 10.5; CG-07._

- [x] **T014 — Implement immutable patch ancestry and exact replay.** Record base, parents, program, compiler/backend identity, effects, and costs. Reject cycles/missing parents; restore by immutable checkpoint identity, not delta subtraction.
  - Verify: `cargo test -p nf-patches replay`; replay deterministic edits and reject wrong-base application.
  - _Requirements: 5.6, 10.6, 13.4–13.5; F2-05._

- [x] **T015 — Implement standalone checkpoint export and core restore.** Materialize Safetensors plus validated metadata with atomic publication, digest checks, and alias reconstruction contracts. Keep source and active-reference changes separate.
  - Verify: `cargo test -p nf-checkpoint export_restore`; reopen exported bytes and reject corrupt/incomplete manifests.
  - _Requirements: 10.2, 10.4–10.6, 13.4; CG-08._

- [x] **T016 — Implement native core CLI commands.** Expose model inspection, IR/search-space validation, compile/explain, apply, replay, and restore through the same Rust APIs. Read commands must not execute studies.
  - Verify: `cargo test -p nf-cli core_commands`; include typed exit codes and a no-Python execution environment.
  - _Requirements: 13.2, 13.4–13.5; OSS-01, OSS-02; F2-12._

- [x] **T017 — Implement the PyO3 control boundary.** Expose immutable handles, strict record submission, typed errors, cancellation handles, and GIL-safe core operations. Contain panics; prohibit mutable source views and unsafe callback reentry.
  - Verify: `cargo test -p nf-python ffi`; `python -m pytest -q -k native_boundary`.
  - _Requirements: 5.4, 13.4, 14.3; OSS-01._

- [x] **T018 — Implement Python core facades and cross-language conformance.** Add SDK record ergonomics, surgeon/IR APIs, version negotiation, and Rust-authoritative validation. Reuse shared golden identities/errors instead of independent Python hashing.
  - Verify: `python -m pytest -q -k core_conformance`; compare CLI, extension, and native digests.
  - _Requirements: 4.1–4.4, 13.4; OSS-01, OSS-02; F2-12._

- [x] **T019 — Add bounded property and parser-fuzz tests.** Exercise shapes, effects, equivalent permutations, checkpoint metadata, and malformed IR with allocation/time caps. Add differential native operation tests against PyTorch.
  - Verify: `cargo test --workspace core_properties`; `python -m pytest -q -k cpu_differential`; run bounded fuzz smoke targets.
  - _Requirements: 5.2–5.5, 13.4, 14.1; OSS-02, OSS-04._

- [x] **T020 — Package and test the M1 offline vertical slice.** Add Apache-2.0 metadata/notices, dependency/license policy, wheel/native smoke CI, and executable direct-edit/replay examples. Emit capability metadata without claiming gym or scientific readiness.
  - Verify: native example without Python; clean Maturin wheel import; `python -m pytest -q -k m1_vertical_slice`.
  - _Requirements: 5.1, 13.1, 13.4–13.5; OSS-01, OSS-06; MRP-01._

## M2 — Multi-gym experimentation and the minimum research product

- [x] **T021 — Implement study, gym, and portfolio contracts.** Add exact bindings, role/metric schemas, thresholds, no-op/control requirements, three split roles, runtime profile, and finite execution limits. Support the standard five-gym and labeled exploratory profiles.
  - Verify: `cargo test -p nf-core study_contract`; reject missing thresholds, unsupported roles, and unlimited budgets.
  - _Requirements: 2.1–2.2, 9.1, 10.1; GYM-16, GYM-17; CG-03, CG-06._

- [x] **T022 — Implement durable freeze, revisions, and coordinator leases.** Persist canonical contracts, events, legal stages, fencing tokens, and immutable references through Rust-owned SQLite transactions. Changed conditions create linked revisions.
  - Verify: `cargo test -p nf-experiments contracts_and_leases`; test stale writers, duplicate commits, and edited frozen fields.
  - _Requirements: 2.3–2.4, 13.1; F2-02; CG-03._

- [x] **T023 — Implement budget reservations and preflight enforcement.** Account for wall/CPU time, attempts, evaluations, tokens, spend, memory, and storage across branches/retries. Deny unsupported hard limits and unapproved external work.
  - Verify: `cargo test -p nf-experiments budgets`; test unknown billing, reservation races, and revision-based budget evasion.
  - _Requirements: 2.2, 2.6, 11.4, 14.7; CG-03, CG-11._

- [x] **T024 — Implement supervised cancellation and recovery.** Add worker process groups, persisted stop requests, bounded cleanup, crash reconciliation, and incomplete outcomes. Resume cannot reset costs/exposure or promote partial results.
  - Verify: `cargo test -p nf-experiments cancellation_recovery`; interrupt every manual-workflow stage.
  - _Requirements: 2.5, 10.3, 13.2; CG-04._

- [x] **T025 — Implement scoped worker IPC and inference brokerage.** Use bounded versioned envelopes, lease/contract IDs, split roles, model/tool quotas, and staged output handles. Never pass live model/torch objects or coordinator credentials to a gym.
  - Verify: `python -m pytest -q -k worker_broker`; deny forged leases, hidden artifact requests, and oversized outputs.
  - _Requirements: 2.6, 9.5, 14.3–14.5; GYM-19; CG-01._

- [x] **T026 — Implement the qualified sandbox launcher.** Run untrusted Python only in a resource-limited isolated subprocess with explicit mounts, no credentials/network by default, and enforced process/time/memory/output limits. No sandbox means rejection, not in-process exec.
  - Verify: `python -m pytest -q -k sandbox_isolation`; real configured sandbox escape/cleanup tests and default-off denial tests.
  - _Requirements: 14.3–14.4, 14.7; GYM-19; CG-01._

- [x] **T027 — Implement custom gym package/file intake.** Snapshot explicit local assets, dependencies, licenses, and manifests without coordinator-side import execution. Track code/data changes as new identities; support single-shot and episode schemas.
  - Verify: `python -m pytest -q -k custom_gym_import`; include side-effectful imports, path escapes, and changed sources.
  - _Requirements: 14.1–14.4; GYM-14, GYM-15, GYM-22._

- [x] **T028 — Implement gym validation receipts.** Validate schemas and resources, then sandbox known-good/wrong/invalid outputs, scorer/oracle agreement, reset/seed/termination, costs, and cleanup. Bind approval to the exact validated version.
  - Verify: `python -m pytest -q -k gym_validation`; reject constant-positive scorers, invalid oracles, and stale receipts.
  - _Requirements: 9.1, 14.3–14.4; GYM-18; CG-01._

- [x] **T029 — Implement the three-way split broker.** Store development/selection/locked-holdout manifests and group-disjointness checks. Freeze the final panel before opening holdout in a separate access domain.
  - Verify: `python -m pytest -q -k split_broker`; deny hidden inputs/labels and post-holdout search.
  - _Requirements: 9.4–9.7; CG-05._

- [x] **T030 — Implement exact exposure and legacy split intake.** Record item/group/version, recipient, shown fields, seeds, started/completed/failed access, and retries. Reject renaming exposed data or treating two-way edit/sealed history as independent three-way evidence.
  - Verify: `cargo test -p nf-experiments exposure`; `python -m pytest -q -k legacy_split_intake`.
  - _Requirements: 9.5–9.6, 13.1; GYM-22; CG-05._

- [x] **T031 — Implement the arithmetic model and five built-in gyms.** Build the designed ReLU heads, corrupt-sum source, independent arithmetic scorers, frozen splits, and a user-packaged sum gym that can mix with built-in retention heads.
  - Verify: `python -m pytest -q -k arithmetic_gyms`; check exact values, domains, and synthetic provenance.
  - _Requirements: 6.1, 9.1–9.2, 13.3; GYM-16; CG-06._

- [x] **T032 — Implement safe PyTorch model workers.** Load registered factories from verified Safetensors/config, reconstruct aliases, run bounded inference/smoke tests, and reject remote custom code or unsupported types/layouts.
  - Verify: `python -m pytest -q -k model_worker`; exercise buffers, tied weights, invalid formats, and cleanup.
  - _Requirements: 3.1, 5.3, 5.5, 14.1; CG-07._

- [x] **T033 — Implement the trusted PyTorch edit backend.** Dispatch compiled same-shape plans under inference mode; return digests/effects and allocation/copy receipts. Compare results with native references and disallow training or silent casts.
  - Verify: `python -m pytest -q -k torch_edit_backend`; inject backward/update requests and outside-scope mutations.
  - _Requirements: 1.3–1.4, 5.1–5.5; OSS-02; CG-02, CG-07._

- [x] **T034 — Implement GymPortfolio binding and coverage.** Add user/Foundry composition, objectives, weights, hard constraints, validation state, and fixed schedules. Weights cannot offset a failed required gym; report missing risk coverage.
  - Verify: `python -m pytest -q -k portfolio_coverage`; distinguish valid exploratory subsets from five-gym assurance and missing surgery checks.
  - _Requirements: 9.1, 10.1; GYM-16–GYM-18; MRP-01._

- [x] **T035 — Implement paired original and no-op controls.** Run matched tasks/settings and route the no-op through apply/export checks. Cache source results only under identical evidence keys and prevent duplicate tasks inflating sample counts.
  - Verify: `python -m pytest -q -k paired_controls`; assert no-op identity and no positive accuracy gain.
  - _Requirements: 9.2, 11.1; CG-06._

- [x] **T036 — Implement latency and memory protocols.** Measure declared CPU runtime statistics and RSS with warm-up/repeats/order controls; record unavailable device metrics separately. Bind thresholds to exact units, scope, and workload.
  - Verify: `python -m pytest -q -k measurement_protocol`; use controlled clocks for gate logic and real CPU measurement smoke tests.
  - _Requirements: 9.3, 10.1, 14.6; CG-06._

- [x] **T037 — Implement deterministic and paired statistical analyses.** Add exact fixture checks and seeded group-aware paired bootstrap with minimum support, uncertainty, and multiplicity across final metrics/candidates/contrasts.
  - Verify: `python -m pytest -q -k paired_statistics`; include insufficient groups, metric directions, noise, and non-finite outcomes.
  - _Requirements: 9.3, 10.1–10.3; F2-06; CG-06._

- [x] **T038 — Implement the immutable evidence acceptance gate.** Require every frozen target/retention/resource check and full evidence panel; separate failed, inconclusive, missing, and incomplete results. Freeze final candidate IDs before holdout evaluation.
  - Verify: `cargo test -p nf-experiments acceptance_gate`; inject missing gyms, retention damage, and threshold changes.
  - _Requirements: 6.3–6.4, 9.5, 10.1–10.3; CG-05, CG-06._

- [x] **T039 — Implement fresh-process reload and atomic promotion.** Reload staged standalone weights with no live candidate reference; verify tensor/buffer/alias/config identity and probe parity before advancing an accepted pointer.
  - Verify: `python -m pytest -q -k reload_promotion`; corrupt one coordinate, alias, artifact, dtype, and output.
  - _Requirements: 5.4–5.5, 10.2–10.6; CG-07, CG-08._

- [x] **T040 — Implement committed status and JSON/Markdown reports.** Render budgets, stage, attempts, controls, gym gates, measurement settings, provenance/evidence labels, stop reasons, and baseline statuses from one event snapshot. Disabled training arms are not run, not fabricated results.
  - Verify: `python -m pytest -q -k status_reports`; compare formats and detect hidden failures or synthetic relabeling.
  - _Requirements: 4.5, 11.2–11.5, 13.1–13.5; CG-11, CG-12, CG-14._

- [x] **T041 — Implement the manual ResearchWorkflow SDK and native study commands.** Wire plan/validate/freeze/run/status/stop/resume/report around supplied IR and the Rust coordinator. Require an explicit store, frozen digest, and execution capability.
  - Verify: `python -m pytest -q -k manual_workflow`; assert planning performs no inference/download/provider call.
  - _Requirements: 2.1–2.6, 13.2; F2-02, F2-12; MRP-01._

- [x] **T042 — Implement standalone deterministic Gym Scientist audits.** Add GymResearchWorkflow and gym research commands to inspect task validity, obvious shortcuts, scorer agreement, saturation, and coverage using supplied reference policies without editing weights.
  - Verify: `python -m pytest -q -k gym_scientist_audit`; detect trivial scorers and separate valid/invalid policies.
  - _Requirements: 9.1, 13.3, 14.2–14.4; MRP-02; GYM-18._

- [x] **T043 — Implement the M2 end-to-end negative-path suite.** Exercise a mixed five-gym manual repair, harmful edit, source/no-op controls, held-out verification, reload, cancellation, and recovery. Assert partial evidence never promotes.
  - Verify: `python -m pytest -q -k m2_vertical_slice`; real sandbox qualification is mandatory for its custom-code path.
  - _Requirements: 5.4–5.6, 6.3–6.4, 9.1–9.6, 10.1–10.6; CG-01–CG-08._

- [x] **T044 — Implement the manual WeightRepairBench harness.** Add versioned tasks/results and original/no-op/decoding/activation comparison slots, fixture runs, complete cost fields, and explicit not-run training arms. Keep injected and natural failures separate.
  - Verify: `python -m pytest -q -k repair_benchmark`; reject missing required comparison/evidence records.
  - _Requirements: 6.1–6.4, 11.1–11.6, 13.3; OSS-07._

- [x] **T045 — Publish machine-readable M2 capability and gate checks.** Add CI/release checks for minimum-product imports, sandbox capability, contracts, independent verification, reports, and explicitly incomplete advanced adapters/engines. Do not equate M2 with full feature completion.
  - Verify: `python -m pytest -q -k m2_capabilities`; run targeted Cargo and Python milestone suites.
  - _Requirements: 13.2–13.5, 14.4; MRP-01; CG-01–CG-08, CG-11, CG-12, CG-14._

## M3 — Autonomous weight programmer

- [x] **T046 — Implement versioned research-engine registration.** Register trusted engines with schemas, compatible models, permissions, determinism, training behavior, estimates, and operation collection. Reject runtime self-registration from agent text.
  - Verify: `cargo test -p nf-experiments engine_registry`; test version and capability mismatches.
  - _Requirements: 1.2–1.4, 2.1, 14.2; F2-01._

- [x] **T047 — Implement bounded weight/activation inspection tools.** Expose slices, summaries, declared sites, input provenance, and temporary hooks with byte/time/example caps. Remove hooks and preserve parameter/buffer state on every exit.
  - Verify: `python -m pytest -q -k bounded_inspection`; include cap exhaustion and hook exceptions.
  - _Requirements: 3.1–3.2, 3.5, 14.7; DBG-01._

- [x] **T048 — Implement matched execution diagnosis on a small transformer.** Add a seeded causal transformer with residual/attention/MLP sites and known diagnostic ground truth. Compare successful/failed tasks with explicit matching variables and limitations.
  - Verify: `python -m pytest -q -k matched_transformer_diagnosis`; confirm site and task identities.
  - _Requirements: 3.1–3.4; DBG-01, DBG-02; F2-03._

- [x] **T049 — Implement gradient attribution receipts and classification.** Run permitted saliency on isolated snapshots, record backward/memory costs and unchanged state, and distinguish inspection-only localization from gradient-derived update values or fitting.
  - Verify: `python -m pytest -q -k gradient_attribution_policy`; strict profiles deny backward and state drift invalidates inspection.
  - _Requirements: 1.2–1.4, 3.2–3.5, 11.4; CG-02._

- [x] **T050 — Implement diagnostic patching, ablations, and counterfactuals.** Use temporary state, declared donor/recipient sites, matched controls, intervention reversal, and safe reset. Keep diagnostic effects out of final checkpoint mutations.
  - Verify: `python -m pytest -q -k causal_probes`; test buffer drift, failed cleanup, and unrelated-input controls.
  - _Requirements: 3.3–3.6, 5.4; DBG-03; F2-03._

- [x] **T051 — Implement structured hypotheses and predicted outcomes.** Require failure evidence, sites, alternatives, falsifiers, controls, uncertainty, origin/support labels, and intended gym effects. Suggestive evidence cannot auto-promote a causal claim.
  - Verify: `python -m pytest -q -k hypothesis_records`; preserve contradictions and reject missing evidence links.
  - _Requirements: 3.3–3.6, 4.1, 4.5; CG-12._

- [x] **T052 — Implement restricted scientist/provider tools.** Support configurable local and remote reasoning clients plus fake protocol tests; route every action through allowed tools, budgets, payload classes, and capability checks. No shell or hidden-data reader.
  - Verify: `python -m pytest -q -k scientist_tools`; reject prompt injection and forged tool/permission requests.
  - _Requirements: 2.6, 9.5, 14.2–14.5; F2-04, F2-12._

- [x] **T053 — Implement the authorized teacher gateway.** Separate behavioral and internal access, identity/rights, payload permissions, request reservations, token/cost receipts, and uncertain completion handling. No default external contact.
  - Verify: `python -m pytest -q -k teacher_gateway`; deny weight access from output-only teachers and unauthorized destinations.
  - _Requirements: 8.1–8.2, 11.4, 14.5; CG-11._

- [x] **T054 — Implement hypothesis-to-Weight-IR proposal submission.** Validate targets, exact value artifacts/constructors, effects, risks, source/search-space bindings, and attribution lineage. Count malformed/denied proposals in attempts and cost history.
  - Verify: `python -m pytest -q -k proposal_submission`; reject valid IR outside allowed sites and hidden gradient-derived deltas.
  - _Requirements: 1.3–1.5, 4.1–4.5; MRP-03._

- [x] **T055 — Implement bounded sequential and branch search.** Evaluate distinct candidate branches against frozen original/best-known references, reuse only valid cached evidence, and stop at limits. Require no evolutionary or merge engine for this scheduler.
  - Verify: `python -m pytest -q -k bounded_branch_search`; test ancestor effects, duplicate candidates, and budget cancellation.
  - _Requirements: 2.2–2.5, 6.1–6.4; F2-05; MRP-01, MRP-03._

- [x] **T056 — Implement agent-directed Gym Scientist proposals.** Investigate measurement weaknesses, draft new/corrected gyms, validate through an independent reference path, and require exact-version approval before binding. Task count alone cannot pass quality targets.
  - Verify: `python -m pytest -q -k agent_gym_research`; reject self-validating or unapproved scorer changes.
  - _Requirements: 9.4–9.6, 14.2–14.4; GYM-20; MRP-02._

- [x] **T057 — Implement scientific-loop revision handling.** Permit return to profile/baseline/diagnosis; invalidate affected comparisons after gym/objective/space changes and require legal new revisions/approvals. Final feedback cannot restart search.
  - Verify: `cargo test -p nf-experiments scientific_loop`; include threshold/gym changes and cumulative exposure/budget inheritance.
  - _Requirements: 2.4, 9.5–9.6; F2-02; CG-03, CG-05._

- [x] **T058 — Implement complete research-cost and experience manifests.** Join proposals, inspection/backward calls, failures, teachers, alignment status, branch costs, source references, and model geometry/failure features. Distinguish zero, estimated, reserved, and unknown costs.
  - Verify: `python -m pytest -q -k research_ledger`; shared costs count once and failed attempts remain visible.
  - _Requirements: 1.2–1.5, 11.3–11.6, 13.1; MRP-04; CG-11._

- [x] **T059 — Implement one declared local pretrained-model adapter.** Add safe pinned HF intake for a supported dense Qwen-style architecture, recognized site/config mapping, tokenizer provenance, and default-denied downloads/custom code. Qualify mini-config tests separately from a real user-supplied checkpoint.
  - Verify: `python -m pytest -q -k pretrained_adapter`; test offline tiny configurations and unsupported variants without downloading weights.
  - _Requirements: 3.1–3.2, 5.3, 14.1; OSS-08._

- [x] **T060 — Implement opaque-solution autonomy tests.** Build brokered task/target packets with solution artifacts withheld; exercise tool-based diagnosis, agent proposals, and at least two distinct non-no-op branches. Separate fake/scripted protocol tests from genuine agent qualification.
  - Verify: `python -m pytest -q -k opaque_target_protocol`; known recipe/coordinate exposure must invalidate the unseen-autonomous label.
  - _Requirements: 4.1, 6.1–6.4, 9.5, 13.3; MRP-05._

- [x] **T061 — Implement the frozen unseen-model flagship runner.** Validate custodian-selected target ancestry/exposure, baseline competence, five-gym controls, independent holdout, allowed search space, branch counts, and approved resources. Do not execute a real study by default.
  - Verify: `python -m pytest -q -k flagship_protocol`; reject published-solution packets, human-supplied repairs, and altered thresholds.
  - _Requirements: 2.1–2.6, 9.1–9.7, 10.1–10.4; MRP-05._

- [x] **T062 — Implement independent reproduction bundles.** Export program/artifact/code/dependency/runtime identities and controlled inputs for fresh-environment replay. Report parameter replay, inference repeatability, and scientific replication as separate verdicts.
  - Verify: `python -m pytest -q -k independent_replay`; missing artifacts or mismatched conditions cannot claim reproduction.
  - _Requirements: 13.1, 13.3–13.5; MRP-05; CG-14._

- [x] **T063 — Integrate the M3 autonomous workflow and readiness report.** Connect scientist, inspection, proposals, branches, independent gate, export/reload, and Gym Scientist. Emit separate software-ready, provider-qualified, and flagship-success states.
  - Verify: `python -m pytest -q -k m3_vertical_slice`; full negative-path and no-solution-exposure assertions; no mocked scientific-success badge.
  - _Requirements: 1.1–1.5, 6.1–6.4, 10.1–10.5, 13.3; MRP-01, MRP-05._

## M4 — Advanced research engines and cross-model learning

- [x] **T064 — Implement the typed scientific graph.** Persist model/circuit/behavior/hypothesis/patch/gym/experiment/evidence nodes and constrained relations with version events. Keep it separate from the architecture graph and deny direct worker database writes.
  - Verify: `cargo test -p nf-experiments scientific_graph`; reject incompatible edges and unconfirmed causal promotion.
  - _Requirements: 3.3–3.4, 12.4, 13.1; F2-09._

- [x] **T065 — Implement evidence-aware research-memory queries.** Retrieve related successes, failures, contradictions, costs, and unresolved alternatives while preserving access classes and support grades. A repeated model claim cannot become a measured fact.
  - Verify: `python -m pytest -q -k research_memory`; exclude holdout/private records from proposer contexts.
  - _Requirements: 4.5, 9.5, 12.4; F2-09; CG-12._

- [x] **T066 — Implement a pinned circuit-tracing adapter.** Map a selected trusted tracing/TransformerLens-compatible implementation into versioned sites/features/attribution artifacts. Validate against its actual supported protocol and a real local fixture.
  - Verify: `python -m pytest -q -k circuit_tracing_adapter`; mocked traces cannot qualify the adapter.
  - _Requirements: 3.2–3.6; DBG-04; F2-03._

- [x] **T067 — Implement sparse-feature representation support.** Load authorized feature bases/checkpoints, record training provenance and reconstruction error, and map feature interventions to model sites with compatibility checks.
  - Verify: `python -m pytest -q -k sparse_features`; reject incompatible bases and unsupported precision/layouts.
  - _Requirements: 3.1–3.4, 4.5; DBG-04; F2-03._

- [x] **T068 — Implement causal ranking and independent confirmation.** Rank localization using declared intervention/control evidence, then challenge explanations with reserved inputs and alternative interventions. Expose benchmark protocol adapters without claiming unrun MIB compliance.
  - Verify: `python -m pytest -q -k mechanism_confirmation`; a repaired behavior with a false explanation must keep split verdicts.
  - _Requirements: 3.3–3.6, 9.7; DBG-05, DBG-06; CG-12._

- [x] **T069 — Implement patch dependency and conflict composition.** Check bases, read/write/alias/packing groups, ordering, graph/runtime effects, and noncommutation. Explicit reconciliation creates a new program; disjoint writes still require behavioral evaluation.
  - Verify: `cargo test -p nf-patches composition`; test cycles, implicit rebases, and disjoint-but-interfering effects.
  - _Requirements: 4.4, 5.4, 6.4; F2-05; MRP-03._

- [x] **T070 — Implement bounded mutation and recombination.** Generate seeded allowed-space changes with prerequisite/scope/cumulative-budget checks. Every composed or mutated proposal must recompile against its frozen space.
  - Verify: `cargo test -p nf-patches evolutionary_ops`; reject out-of-space offspring and write-budget evasion.
  - _Requirements: 2.2, 4.4; F2-05; MRP-03._

- [x] **T071 — Implement population diversity and Pareto selection.** Maintain feasible provisional versus independently verified frontiers across quality, retention, latency, memory, and complexity. Preserve uncertain nondominance instead of inventing a scalar winner.
  - Verify: `python -m pytest -q -k pareto_population`; include true tradeoffs, noisy ties, infeasible elites, and final-panel multiplicity.
  - _Requirements: 9.3, 10.1–10.3; F2-06._

- [x] **T072 — Implement cross-gym and patch-interference analysis.** Join exact A/B/A+B experiments with paired groups/seeds and uncertainty. Report positive/negative transfer without imputing missing cells or claiming causal circuit sharing from correlation.
  - Verify: `python -m pytest -q -k gym_interference`; exercise shared groups, missing cells, and multiplicity.
  - _Requirements: 6.4, 9.3, 9.7; GYM-23._

- [x] **T073 — Implement the concrete rule-to-weights construction pathway.** Compile arithmetic from an explicit rule, preserve tensor derivations, and use an independent oracle for normal/boundary/invalid inputs before integrated gyms and reload verification.
  - Verify: `python -m pytest -q -k rule_construction`; reject shared oracle/constructor logic and invalid-input false positives.
  - _Requirements: 7.1–7.5; F2-13; CG-09._

- [x] **T074 — Implement recorded deterministic mapping transfer.** Translate a compatible permuted teacher computation into the target; include unchanged/no-op and safe unaligned controls, mapping receipts, target retention, and unsupported incompatible pairs.
  - Verify: `python -m pytest -q -k mapped_transfer`; copied counts or low residual alone cannot pass.
  - _Requirements: 8.1–8.5, 8.7; CG-10._

- [x] **T075 — Implement inferred permutation correspondence.** Estimate compatible mappings using declared permitted evidence, diagnostics, thresholds, and algorithm provenance. Separate supplied permutation, inferred correspondence, and target-fitted adaptation labels.
  - Verify: `python -m pytest -q -k permutation_alignment`; include wrong correspondences and exposed calibration data.
  - _Requirements: 8.3–8.7, 11.3–11.4; F2-07; CG-10._

- [x] **T076 — Implement fitted mapping/model-stitching research support.** Add a bounded registered linear/adaptation solver with frozen calibration inputs and trained/hybrid lineage, even when fitting is closed-form. Reject unsupported insertion/dimension contracts.
  - Verify: `python -m pytest -q -k fitted_alignment`; direct-only contracts must deny this fitted path.
  - _Requirements: 1.3, 8.3–8.6; F2-07._

- [x] **T077 — Implement three-level transfer study protocols.** Separate independently initialized same-architecture, cross-size, and cross-family target groups with safe compatibility checks, held-out verification, and explicit inconclusive outcomes.
  - Verify: `python -m pytest -q -k transfer_protocols`; a permuted fixture cannot satisfy independent-model evidence.
  - _Requirements: 8.4–8.7, 9.7, 13.3; F2-07; DBG-07._

- [x] **T078 — Implement the authorized LoRA fixture baseline.** Fit a bounded low-rank head update, record data/steps/parameters/cost, merge before standalone evaluation, and preserve failed direct results separately.
  - Verify: `python -m pytest -q -k lora_baseline`; default not-run behavior and merged reload parity.
  - _Requirements: 11.1–11.6; CG-11._

- [x] **T079 — Implement the authorized SFT fixture baseline.** Train only declared parameters with the frozen supervised objective and schedule; require explicit data/resource permission and complete optimizer lineage.
  - Verify: `python -m pytest -q -k sft_baseline`; reject disabled training and unauthorized parameter updates.
  - _Requirements: 1.3, 11.1–11.6; CG-11._

- [x] **T080 — Implement the authorized distillation fixture baseline.** Train against permitted teacher outputs with matched evaluation and teacher-generation costs. Output-only guidance must not be reported as internal circuit transfer.
  - Verify: `python -m pytest -q -k distillation_baseline`; deny unauthorized teacher payloads and hidden costs.
  - _Requirements: 8.2, 11.1–11.6, 14.5; CG-11._

- [x] **T081 — Implement the authorized RL fixture baseline.** Use the designed fixed-variance Gaussian action REINFORCE fixture with frozen reward, seeds, clipping, and step limits; distinguish training policy behavior from mean-output evaluation.
  - Verify: `python -m pytest -q -k rl_baseline`; bound all episodes and record failures without requiring optimizer success.
  - _Requirements: 1.3, 11.1–11.6; CG-11._

- [x] **T082 — Implement external and adaptive memory method adapters.** Record memory artifact, reset/update/persistence rules, capacity, tool permissions, and any trained/test-time updates. Static weight exports must not depend on hidden runtime state.
  - Verify: `python -m pytest -q -k memory_methods`; reject hidden-label writes and direct-only relabeling.
  - _Requirements: 1.2–1.4, 10.2, 11.3–11.5; F2-10._

- [x] **T083 — Implement continual-learning comparison protocols.** Freeze episode order, source state, adaptation access, and reset controls; measure retention, forward/backward transfer, recovery, state growth, and cumulative cost.
  - Verify: `python -m pytest -q -k continual_protocols`; detect cross-episode and split contamination.
  - _Requirements: 6.4, 9.2–9.7, 11.5; F2-10._

- [x] **T084 — Implement and qualify the NeMo Gym adapter.** Pin the actual framework release, normalize task/tool/scoring lifecycle, declare service/endpoint needs, and run in the established sandbox/broker contract.
  - Verify: `python -m pytest -q -k nemo_adapter`; real local valid/invalid/replay/cleanup fixture, not only mocks.
  - _Requirements: 14.3–14.5; GYM-14, GYM-15, GYM-18._

- [x] **T085 — Implement and qualify the Prime Intellect Verifiers adapter.** Pin the actual verifier protocol, preserve reward components/rollout errors, and avoid treating arbitrary reward as accuracy.
  - Verify: `python -m pytest -q -k verifiers_adapter`; real local rollout/scoring and denial tests.
  - _Requirements: 9.1, 14.3–14.4; GYM-15, GYM-18._

- [x] **T086 — Implement and qualify the Gymnasium adapter.** Preserve spaces, explicit model/action bridges, seeded reset, step, termination/truncation, and episode return with bounded tools/resources.
  - Verify: `python -m pytest -q -k gymnasium_adapter`; real local multi-step replay and cleanup.
  - _Requirements: 9.1, 14.3–14.4; GYM-15, GYM-18._

- [x] **T087 — Implement qualified private-data storage and export controls.** Enforce namespace/access classes, configured encryption/key requirements, keyed private exposure IDs, restricted logs, destination policies, and separately approved publication/training exports.
  - Verify: `python -m pytest -q -k private_gym_policy`; deny payload, hash, path, log, and small-group disclosure routes.
  - _Requirements: 9.5, 14.5; GYM-21, GYM-22._

- [x] **T088 — Implement richer standalone Gym Scientist investigations.** Add redundancy/shortcut/difficulty experiments, adaptive challenge proposals, independent quality validation, and approval/rebaseline workflows. Permit fewer/better gyms instead of rewarding task count.
  - Verify: `python -m pytest -q -k measurement_research`; test duplicated distributions, confounding, and self-validating proposals.
  - _Requirements: 9.3–9.7; GYM-20, GYM-23; MRP-02._

- [x] **T089 — Implement community gym package intake.** Validate immutable digests, publisher/signature origin, compatibility, license/rights, and sandbox requests without trusting score claims or running install scripts in the core.
  - Verify: `python -m pytest -q -k community_gym`; offline import, signature mismatch, malicious metadata, and explicit download denial.
  - _Requirements: 14.1–14.5; GYM-24; OSS-06._

- [x] **T090 — Implement versioned verified-library lifecycle.** Require complete accepted evidence for verified entries; append disputes/withdrawals and retain contradictions/failures. Default reuse excludes disputed or withdrawn versions.
  - Verify: `cargo test -p nf-experiments library_lifecycle`; invalid or missing acceptance/reload receipts cannot promote entries.
  - _Requirements: 12.1–12.2, 12.4–12.5; CG-13._

- [x] **T091 — Implement library replay and fresh reuse verification.** Check recipe prerequisites and exact-source digest replay; create new target/task studies and keep every failed/unsupported reuse attempt. No inherited acceptance on matching shapes.
  - Verify: `python -m pytest -q -k library_reuse`; exercise new models, changed tasks, missing artifacts, and disputed versions.
  - _Requirements: 12.3–12.5, 13.4–13.5; CG-13._

- [x] **T092 — Implement history-informed intervention strategy retrieval.** Query architecture/depth/geometry/failure features and both successful/failed interventions; emit bounded recommendations with evidence grades and costs, not accepted edits.
  - Verify: `python -m pytest -q -k strategy_retrieval`; preserve contradictory history and block target holdout leakage.
  - _Requirements: 4.5, 9.5, 12.4; MRP-04._

- [x] **T093 — Implement the cross-model study runner.** Freeze discovery/validation/final model groups, strategy cut-off, target budgets, paired protocols, and no-history/random/history-informed controls. Include source-plus-target cost and uncertainty.
  - Verify: `python -m pytest -q -k cross_model_protocol`; prevent ancestry overlap and target-outcome strategy tuning.
  - _Requirements: 9.7, 11.3–11.5, 13.3; MRP-04._

- [x] **T094 — Implement proposer-training dataset export.** Build approved experience datasets with failure records, model/task-group splits, deduplication, and private/holdout exclusions. Preserve extraction version and license/provenance.
  - Verify: `python -m pytest -q -k proposer_dataset`; deny restricted records and near-identical cross-split transformations.
  - _Requirements: 9.4–9.7, 14.5; F2-04; MRP-04._

- [x] **T095 — Implement a bounded learned-proposer training backend.** Train a small structured-program proposer in a separate authorized fixture study; record objective, examples, gradients, steps, compute, and frozen output artifact identity.
  - Verify: `python -m pytest -q -k proposer_training`; direct studies cannot silently trigger training.
  - _Requirements: 1.2–1.4, 11.3–11.5; F2-04; MRP-04._

- [x] **T096 — Implement learned-proposer inference and comparison.** Produce constrained IR rather than full checkpoints; compare validity, evaluation-to-success, target quality, and amortized/unamortized total costs against no-history proposals on held-out groups.
  - Verify: `python -m pytest -q -k learned_proposer`; learned origin cannot erase target fitting or imply measured advantage.
  - _Requirements: 1.2–1.4, 4.4, 11.3–11.5; F2-04; MRP-04._

- [x] **T097 — Implement WeightTransferBench comparison execution.** Wire safe unaligned/aligned controls and enabled distillation/matched training with independent target evaluation, scope labels, complete costs, and explicit missing-comparison status.
  - Verify: `python -m pytest -q -k transfer_benchmark`; incompatible copies are never forced and fixture permutations never count as cross-family results.
  - _Requirements: 8.4–8.7, 9.7, 11.1–11.6; OSS-07._

- [x] **T098 — Implement research-source and data-attribution provenance.** Ingest authorized source/version/license/scope and available training-data-to-circuit evidence through typed adapters. Missing training data or unverified paper summaries remain unavailable/suggestive, not fabricated causal attribution.
  - Verify: `python -m pytest -q -k research_source_records`; reject unsupported reproduction and data-attribution claims.
  - _Requirements: 3.4, 4.5, 13.3; F2-14._

- [x] **T099 — Integrate full construction, transfer, and library acceptance.** Run concrete repair/construction/mapped-transfer fixtures through independent gyms, method accounting, export/reload, library version/reuse, and status reports. Emit per-engine qualification rather than an unconditional platform-success flag.
  - Verify: `python -m pytest -q -k m4_feature_acceptance`; all original feature-level software cases and CG-09/10/13 negatives.
  - _Requirements: 7.1–7.5, 8.1–8.7, 12.1–12.5; CG-09, CG-10, CG-13._

## M5 — Architecture and inference optimization

- [x] **T100 — Implement the architecture dependency graph and solver.** Model modules, ports, symbolic dimensions, residual constraints, aliases, attention groups, configuration, and cache/runtime layouts. Reject unsupported shape expressions and inconsistent dependencies.
  - Verify: `cargo test -p nf-graph shape_solver`; include residual, tied-head, overflow, and contradictory-shape cases.
  - _Requirements: 4.4, 5.2–5.4, 7.5; OSS-03._

- [x] **T101 — Implement structural IR lowering and preservation rules.** Expand surgery into tensor/config/metadata rewrites, classify exact-under-conditions versus bounded/experimental effects, and require rebuilt runtime/parity checks. Bind the frozen transformation space.
  - Verify: `cargo test -p nf-weight-ir structural_lowering`; invalid exactness claims or forbidden shape changes must fail.
  - _Requirements: 4.2–4.4, 5.3–5.4, 7.5; OSS-03; MRP-03._

- [x] **T102 — Implement MLP channel and intermediate-width surgery.** Slice matching up/gate rows and down columns, preserve branch alignment, and rewrite supported intermediate-size configuration. Reject incomplete whole-model hidden-width edits.
  - Verify: `python -m pytest -q -k mlp_surgery`; check graph validity, effect masks, smoke, and export/reload.
  - _Requirements: 5.2–5.5, 7.5, 10.4; OSS-03._

- [x] **T103 — Implement attention-head surgery.** Rewrite Q/K/V/output dimensions and supported head metadata with explicit grouped/multi-query/rotary constraints. Unsupported layouts must reject rather than infer an unsafe mapping.
  - Verify: `python -m pytest -q -k attention_surgery`; valid and invalid grouping/cache cases.
  - _Requirements: 5.2–5.5, 7.5; OSS-03._

- [x] **T104 — Implement layer removal and reorganization.** Update indexes, references, configuration, and cache/runtime dependencies atomically. Treat learned-layer reordering as experimental and require full behavioral evaluation.
  - Verify: `python -m pytest -q -k layer_reorganization`; reject stale references and false function-preservation claims.
  - _Requirements: 5.4–5.5, 7.5, 10.3–10.4; OSS-03._

- [x] **T105 — Implement registered structural constructors.** Add at least one valid identity-preserving insertion/split/merge rule on a supported fixture with checked algebraic conditions. Unknown transformer neutrality or vocabulary/width changes remain unsupported.
  - Verify: `python -m pytest -q -k structural_constructors`; prove fixture parity and reject violated preconditions.
  - _Requirements: 4.2, 7.1–7.5; OSS-03; F2-13._

- [x] **T106 — Implement quantized representation and edit policy.** Version scales/zero points/group sizes/rounding/packing metadata and calibration data; validate finite bounded conversion. Reject unsupported packed edits or implicit dequantization.
  - Verify: `python -m pytest -q -k quantized_policy`; test group coupling, dtype constraints, and round-trip error.
  - _Requirements: 5.3–5.5, 14.6; OSS-03; F2-08._

- [x] **T107 — Implement graph-aware compression and sparsity effects.** Distinguish masking from removed computation, count inserted/removed parameters, report declared FLOPs estimates, and export executable supported changed graphs.
  - Verify: `python -m pytest -q -k compression_graph`; zeros alone cannot earn latency or FLOP-removal claims.
  - _Requirements: 5.4, 9.3, 10.4; F2-08._

- [x] **T108 — Implement qualified GPU dispatch and tensor exchange.** Use existing PyTorch/CUDA kernels first; negotiate DLPack/device/stride/stream/lifetime contracts and bounded transfer receipts. Prohibit mutable source sharing and silent backend/precision fallback.
  - Verify: `python -m pytest -q -k gpu_dispatch`; CPU ownership tests plus separately qualified real-GPU differential tests.
  - _Requirements: 5.2–5.4, 11.4, 14.6; OSS-05._

- [x] **T109 — Implement GPU leasing and measurement enforcement.** Reserve device time/memory, synchronize timings, record allocator versus device-wide scopes, and deny unsupported strict device limits. Account for compilation and transfers separately.
  - Verify: `python -m pytest -q -k gpu_resources`; actual device qualification and unsupported-enforcement denial paths.
  - _Requirements: 2.2, 14.6–14.7; OSS-05; CG-06._

- [x] **T110 — Implement the Quantization Gym.** Compare retained quality and real format/runtime memory/throughput against frozen calibration/reference conditions. Reference dequantization correctness cannot substitute for low-bit speed evidence.
  - Verify: `python -m pytest -q -k quantization_gym`; missing native-runtime measures block performance claims.
  - _Requirements: 9.2–9.3, 10.1, 14.6; F2-08._

- [x] **T111 — Implement the Compression Gym.** Evaluate executable graph parameter/FLOP counts and actual latency/memory with retention gates, source/no-op controls, and repeated matched measurements.
  - Verify: `python -m pytest -q -k compression_gym`; reject quality regressions hidden by smaller parameter counts.
  - _Requirements: 6.4, 9.3, 10.1–10.3, 14.6; F2-08._

- [x] **T112 — Implement the Generation Gym.** Measure task success, output tokens, repetition, and latency under matched prompts/decoding/stops/limits. Shorter failed answers cannot qualify as reasoning efficiency.
  - Verify: `python -m pytest -q -k generation_gym`; test truncation, changed stopping rules, and success non-inferiority.
  - _Requirements: 9.2–9.3, 10.1, 14.6; F2-08._

- [x] **T113 — Implement Serving Gym protocols and the real PyTorch backend.** Freeze TTFT/TPOT/throughput/tail definitions, concurrency, cache policy, workload, hardware, and utilization sampling. Separate cold/warm measurements and actual traces from mocks.
  - Verify: `python -m pytest -q -k serving_protocol`; run a bounded real local backend measurement and missing-metric denials.
  - _Requirements: 9.3, 10.1, 14.6; F2-08._

- [x] **T114 — Implement and qualify the vLLM serving adapter.** Pin supported versions/models, validate endpoints, launch only with authorization and explicit loopback binding, and collect real request/runtime identities and metrics.
  - Verify: `python -m pytest -q -k vllm_adapter`; actual backend qualification is distinct from mocked transport tests.
  - _Requirements: 2.6, 14.5–14.6; F2-08._

- [x] **T115 — Implement and qualify the TensorRT-LLM adapter.** Record build/engine identity, precision, supported architecture, compilation cost, and fresh runtime parity. An exported float checkpoint alone cannot verify the compiled engine.
  - Verify: `python -m pytest -q -k tensorrt_adapter`; real compiled-backend qualification plus invalid-engine denial tests.
  - _Requirements: 10.4, 11.4, 14.6; F2-08._

- [x] **T116 — Implement end-to-end systems profiling.** Separate compile/validation, checkpoint I/O/hashing, CPU allocations, transfers, kernels, synchronization, export, and whole-study cost. Compare Rust-assisted and Python/PyTorch paths with equal integrity work.
  - Verify: `python -m pytest -q -k systems_profile`; detect omitted hashing/copies and mismatched workloads.
  - _Requirements: 11.3–11.5, 14.6; OSS-04, OSS-05._

- [x] **T117 — Implement profiled custom-kernel admission.** Add registration/conformance and performance-gate support for optional Triton/CUDA implementations. Require profiler evidence, bounded workload, reference correctness, compile cost, and measured end-to-end gain before activation; do not write speculative kernels without a bottleneck.
  - Verify: `python -m pytest -q -k kernel_admission`; reject faster isolated kernels that lose end to end and missing qualification evidence.
  - _Requirements: 5.4, 11.4–11.5, 14.6; OSS-05._

- [x] **T118 — Implement WeightEfficiencyBench execution.** Compare supported pruning/quantization/compression references on the same backend/hardware/workload and retain quality/cost Pareto results. Missing reference arms prevent a complete-comparison claim.
  - Verify: `python -m pytest -q -k efficiency_benchmark`; include fake file-size speedups and incomplete costs.
  - _Requirements: 9.2–9.3, 10.1–10.3, 11.1–11.5; OSS-07._

- [x] **T119 — Implement real-backend qualification artifacts.** Run an explicitly authorized bounded qualification protocol and emit actual hardware/configuration, repeated measures, quality retention, reload parity, and gain/no-gain verdicts. Code readiness or a skipped run must not set M5 measured-improvement status.
  - Verify: `python -m pytest -q -k backend_qualification`; bounded local measurement plus tests for no gain, unavailable hardware, and incomparable conditions.
  - _Requirements: 10.1–10.4, 13.3, 14.6; F2-08; MRP-01._

- [x] **T120 — Integrate full release conformance and capability reporting.** Wire CPU/native/Python CI, schema/IR compatibility, package/license/security checks, optional hardware matrices, and full feature regression gates. Generate truthful milestone, integration, and scientific-result status from evidence rather than completed task count.
  - Verify: `cargo fmt --check`; `cargo clippy --workspace --all-targets -- -D warnings`; `cargo test --workspace`; `python -m pytest -q` after resource checks; separately run configured integration qualifications.
  - _Requirements: 13.1–13.5, 14.1–14.7; OSS-01–OSS-09; CG-01–CG-14; MRP-01._

## Completion semantics

M1 and M2 are independently reviewable product exits. Their completion does not assert an autonomous discovery, all-framework support, confidential-data qualification, or general transfer. M3 adds the autonomous path, but the unseen-pretrained-model flagship requires its own approved study and positive independent evidence. M4 cross-model strategy learning must beat its frozen controls on held-out model groups before any predictive-generalization claim. M5 must show real retained-quality backend improvement before that scientific/performance success is reported.

All original feature-level software criteria remain required for full feature acceptance. Unsupported or incomplete mandatory qualifications stay open even when their harness code is finished. No task permits an agent to weaken a contract, change the target after final feedback, run generated code in-process, claim a mocked integration as real, or replace a missing experiment with a success narrative.
