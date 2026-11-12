# Control-plane review and proposed delivery order

Reviewed 2026-11-11. **Planning only; no implementation, model run, download,
provider call, training, test-split access or release is authorized by this plan.**

The product is the control plane. CORD and the synthetic/company-facts gyms
are integration and regression workloads, not products whose accuracy must
be optimized before the platform can progress.

## 1. Assessment of the supplied review

The overall diagnosis is right: we have substantial execution/evidence
building blocks, not yet a cohesive operator-facing control plane. Several
claims need updating or narrowing.

| Area | Current evidence | Remaining gap |
| --- | --- | --- |
| Plugin registry | Exact kind/name/version/implementation/dependency resolution exists in [proof_registry.py](../python/neural_foundry/proof_registry.py). | No unified discoverable descriptors/config schemas/capability catalog for planning or a UI. |
| Genericity | Candidate methods, execution and evidence are data; base selection and the real CORD workload have run. | Recorded replay still constructs an invoice task; serving evidence imports a JSON-loop helper; shared serving probes remain under weight editing. |
| Policies | Versioned provider/training/scoring policies and real resource enforcement exist. | The generic validator still fixes probe count16, tolerance1e-6 and public-report purpose. Deployment invariants and configurable policy values need explicit separation. |
| New base models | Qwen3-4B was selected from a run spec using the generic cached-checkpoint exporter. | Approved acquisition is separate/manual. A product needs an explicit acquisition/preflight stage, not unrestricted downloads inside execution. |
| Orchestration | [GymLifecycle.run](../python/neural_foundry/gym_lifecycle.py) trains, reloads, publishes and evaluates; the frontier controller handles one bounded provider workflow. | No generic durable coordinator for the complete multi-candidate run graph. Public proof steps still require external sequencing. |
| State and artifacts | SQLite attempt/reservation records, digest-bound lifecycle JSON, chunk storage, response journals and candidate manifests are real. | No canonical run/stage/attempt index, cross-stage event stream or general artifact-location interface. Filenames and manually assembled roots still coordinate workflows. |
| Approvals | Native exact gym-snapshot approval, in-memory exact-version approval and promotion-eligibility decisions exist. | No unified durable, authenticated operator approval and release/promotion workflow. Eligibility is not an owner ship decision. |
| API / dashboard | Studio is a Python SDK/CLI. No control-plane HTTP service or frontend application was found in the inspected application paths. | Add an API over the shared application service, then a thin operator UI. |
| Real second gym | CORD now has an actual trained local candidate, serving export, identity checks and a matched frontier comparison. | Keep company-facts as a software composition test; do not require another quality campaign to prove the control plane. |
| Quality blocker | The approved real-data LoRA experiment passed its predeclared F1/link/resource gates. | This is useful integration evidence, not production quality or independent final-test qualification. More receipt tuning is not a platform prerequisite. |
| Custody | Stage1 claims/audits/account separation exist; evaluator implementations include legacy Pythia and synthetic study paths. | Registered Gym/vLLM/provider final evaluation is not qualified; native bridge, signing and deployment work remain. |
| CI / portability | M1 CI runs on native Ubuntu and uses the documented capability runner. | It is not WSL-only. Contributor setup, fast CPU contracts and supported-host/backend boundaries need a clearer single entry point. |

This was a targeted source/backlog review, not a line-by-line audit or a fresh
full-suite certification.

## 2. Important architecture correction: one run view, separate authorities

Do **not** solve fragmented visibility by merging every database/ledger into
one mutable store.

- The run store owns orchestration state, stage attempts, approvals, events,
  outcomes and artifact references.
- The budget ledger remains the authoritative source for reservations and
  spending authorization.
- The custodian retains private holdout data and its one-shot claims outside
  the proposer/operator service's data boundary.
- The Rust exposure ledger retains its own enforcement semantics.

Use stable run/candidate/stage/attempt IDs and digest-bound references to
correlate those authorities. Publish safe metadata/events into a single
operator view; do not give that view direct access to sealed prompts or the
ability to reset spending/exposure history.

Start with SQLite and a filesystem-backed artifact adapter locally. Avoid a
Postgres migration, object-store dependency, Kubernetes or Ray requirement
before the first usable local vertical slice. Keep interfaces open to them.

## 3. What the completed experiments teach the control plane

| Finding | Required platform behavior |
| --- | --- |
| A plain-chat projection bypassed JSON guidance and parsing despite an output schema being present. | Validate capabilities across the actual task -> rendered request -> transport -> parsed response -> scorer path before expensive execution. A passing scorer-only fake is insufficient. |
| A training-only candidate list omitted the base teacher method. | Compile the full dependency graph, including internal/base roles, before dispatch. Scored candidates and internal execution roles must be distinct. |
| A panel abort initially lost104 returned responses. | Journal responses/results incrementally and attach every completed unit to durable attempts; never depend on an in-memory final aggregate. |
| A generation cap was treated as a fatal transport error. | Separate valid completion, invalid/model-quality output, infrastructure failure, cancellation and uncertain dispatch outcomes. Preserve raw evidence. |
| Scorer v1/v2 measured different things using identical answers. | Scorers and metric policies are immutable versions. Rescoring is a first-class offline operation linked to original outputs, not a new inference run or a silent score replacement. |
| Model, prompt and decoder changes materially affected results. | Candidate identity binds weights, prompt/projection, schema, decoding and policy, not just a model name. |
| The 512-row and128/512 provider limits needed explicit extensions. | Validate policy/capability compatibility up front; avoid benchmark-specific ceilings hidden inside otherwise generic code. |
| GPU staging affected whether the memory plan was credible. | Placement describes coordinator, training worker, reload and serving separately. Enforce limits in the process actually allocating resources. |
| A direct dataset audit differed from the worker bytes after persistence. | Verify the materialized dataset, masks and budgets that the worker actually consumes; planning proxies must be labelled. |
| Credentials existed locally but not in the worker environment. | Use runtime credential references and preflight availability checks. Never put secrets in specs, receipts, event logs or process arguments. |
| Startup recovery required explicit new approval. | Persist approval scope and recovery eligibility. An interrupted or uncertain started operation must not become an automatic retry. |

Run completion, gate outcome and release authorization are separate concepts.
A completed experiment can fail its quality gates. Passing those gates must
not automatically authorize shipment or a final-test query.

## 4. Proposed work orders

### CP-01 — Run/plan contracts, plugin descriptors and preflight

**First contract decision, before writing the scheduler.**

- Define immutable run definition/compiled-plan records and stable IDs for
  candidates, stages, attempts, approvals, outcomes and artifacts.
- Extend the existing registry with descriptors: config schema, supported
  interaction/artifact/model families, operation/cost capabilities, required
  runtime and resource/placement requirements. Do not invent a second registry.
- Compile a stage dependency graph from a run definition. Include acquisition,
  audits, teacher/base roles, training, export, identity and evaluation where
  required; allow provider-only and offline-rescore plans.
- Resolve an immutable execution bundle before approval. Replace manual
  "core-stable" flags with verified plugin/config/runtime identities; source
  drift invalidates an unstarted plan rather than silently changing the run.
- Extract genuinely shared serving identity/evidence from invoice/weight-edit
  ownership. Recorded-provider replay receives the selected Gym projection.
- Move variable probe/tolerance/coverage values into bound policy documents.
  Keep non-negotiable deployment/trust safeguards explicit and fail closed;
  do not turn custody into a freely switchable `report_only=false` flag.

**Acceptance:** invoice, company-facts and CORD configurations compile through
the same planner; incompatible formats, missing internal roles, unavailable
runtimes/assets and unsupported policies fail before dispatch or spend.
Catalog inspection neither loads models nor executes plugin factories.

### CP-02 — Durable run store, event log and artifact index

Depends on CP-01's record/identity contract.

- Add run, stage, attempt, approval, ordered event and artifact-reference
  records with schema versions/migrations. Store immutable specs/plans by digest.
- Record transitions and their events transactionally. Stage commands use
  durable operation IDs/outbox-style intents rather than untracked shell loops.
- Index existing immutable artifacts and receipts instead of copying/replacing
  weights or flattening authoritative ledgers into the run database.
- Make current status a projection of durable records. `STATUS.json` can remain
  a compatibility view, not a second authority.
- Persist adapter-reported progress with typed units (steps, cases, bytes),
  current operation and heartbeat. "What is running?" must be a direct query,
  not a scan of artifact filenames, Docker processes or shell sessions.
- Initial state vocabulary: draft/validated/awaiting-approval/queued/running/
  verifying/completed; explicit failed/cancelled/needs-attention outcomes.
  Finalize exact transition rules in CP-01.

**Acceptance:** process restart preserves history and reconstructs the same
state/timeline; no filename guessing; corrupted/missing artifacts surface
explicitly; duplicate requests cannot create duplicate attempts.

### CP-03 — Generic orchestration, bounded approvals and recovery

Depends on CP-01 and CP-02. Deliver with CP-02 as the **first local vertical
slice**, not as a separate months-long platform.

- Adapt the existing lifecycle/stage functions into one coordinator; do not
  rewrite trainers, servers, providers or score logic in the orchestrator.
- A single CLI request carries an approved plan to a result, including waiting
  states and dependencies. Eliminate new task-specific sequencing controllers.
- Bind approvals to spec/plan, datasets, model/plugin/policy digests, operations,
  external-provider destination, money/attempt/time/GPU limits and recovery scope.
- Record operator identity, decision, time, reason and superseded approvals.
  Start with an authenticated local approval record; do not label it signed
  shipment authorization before the later signing work is qualified.
- Use stage claims/leases/heartbeats; cancel only owned processes/containers.
  Preserve artifacts and reservations on failure.
- Distinguish safe pre-dispatch work, recoverable committed results and
  ambiguous started operations. The latter require attention/new approval,
  never blind resubmission. Custody's no-retry-after-start rule stays stricter.

**Acceptance:** one documented CLI path executes a mixed local/provider
software-fixture plan and produces indexed comparisons without manual step
ordering. Unapproved work cannot dispatch. Kill/restart tests cover:

1. before dispatch;
2. after dispatch but before receipt;
3. receipt persisted but state not finalized;
4. mid-response stream;
5. budget exhaustion, timeout, allocation failure and user cancellation.

There must be no duplicate paid call or training round during recovery.
No new live model/API run is required for these software acceptance tests.

### CP-04 — API and catalog access

Depends on CP-03; catalog descriptors originate in CP-01.

- A thin HTTP application over the same application service/store as the CLI.
- Versioned endpoints for catalog/schemas, validation/plan creation, approvals,
  start/cancel, run/stage/attempt state, ordered events and artifact metadata.
- Idempotency keys, validation/error contracts and operator authorization.
- Stream events or expose an ordered cursor; do not scrape worker log files
  into a separate UI state machine.
- Keep secrets and sealed content out of responses. Artifact downloads use
  authorized references, not arbitrary filesystem paths.

**Acceptance:** API and CLI show the same state and enforce the same approvals;
restart preserves reads; repeated start requests do not duplicate work.

### CP-05 — Thin operator dashboard

Depends on CP-04. Build this **before waiting for all Stage2 custody work**.

- Pick configured Gym/model/provider/recipe from the catalog.
- Show compiled plan, preflight findings, cost estimate versus reservation/
  usage, resource limits and explicit approval.
- Show live stage timeline, failures/recovery eligibility, artifact lineage
  and metric-policy-labelled comparisons.
- Display operational status, gate outcome and release eligibility separately.
  Unqualified release/final-test actions remain disabled.

**Acceptance:** demonstrate the entire operator flow through the API using
fixtures and imported CORD receipts, without new model runs. UI rendering or
refresh must never start work. No dashboard-specific execution path.
Investor polish, billing and multi-tenant administration are not this milestone.

### CP-06 — Portable placement and artifact backing

Design alongside CP-01; implement after CP-02 and in parallel with the API/UI
where feasible.

- Runtime profiles replace hardcoded WSL home/virtualenv paths in logical specs.
  Record actual resolved executable/image identity in execution receipts.
- Define coordinator/worker/reload/serving placement and resource ownership.
- Pin backend images/runtime manifests; vLLM is not the only backend.
- Use stable native-Linux workspaces for qualified Linux execution instead of
  depending on Windows-mount working-directory behavior.
- Generalize existing filesystem artifact primitives into a permissioned,
  digest-verified interface. Add a remote backing only when needed.
- Model acquisition is an explicitly approved operation with pinned identity,
  size/license checks and asset receipts. Execution remains offline by default.

**Acceptance:** execute a software plan in a second clean Linux workspace
without source changes or a developer-home path in the logical run definition;
retrieve and verify its artifacts through the same interface.

### CP-07 — Real custody integration and release decisions

Depends on CP-03 and the required CP-06 placement/artifact boundaries; not on
dashboard completion.

- Bridge the registered Gym evaluator, local serving and provider contracts to
  the isolated custodian. Reuse existing evaluator/claim primitives.
- Qualify Stage2 native exposure-ledger bridge, signing and container deployment.
- Bind aggregate receipts, policy/gate results and owner promotion/rejection
  decisions to exact candidate/evidence versions.
- Keep private material and one-shot authority outside the operator service.
  Define how interrupted final starts remain permanently consumed.
- Any external-provider final evaluation needs explicit dataset-owner transfer
  consent for that evaluation, not a blanket provider toggle.

**Acceptance:** software-fixture qualification proves account/data separation,
aggregate-only release, strict crash behavior, signature/receipt tamper
rejection and unauthorized-evaluation denial. A fresh real final draw or
CORD test use remains a separate future user authorization.

### CP-08 — Contributor and contract acceptance, parallel throughout

- One documented fast CPU/software contract entry point without model weights,
  paid APIs or a GPU. Separate live/native capability qualifications clearly.
- Keep the documented two-thread capability-runner Cargo rule in native Linux CI.
  Do not mislabel Linux-specific isolation as portable Windows execution.
- Cover all plugin kinds and actual task/request/response boundaries; keep the
  regression cases discovered by CORD.
- Add migrations, replay/reconciliation, concurrent claims and process-crash
  tests as part of the corresponding work orders, not after the dashboard.
- Refresh the full Python/native acceptance baseline before claiming a
  control-plane milestone. The latest105 targeted passes are not a fresh full
  Cargo/full-repository certificate.

## 5. Delivery checkpoints and explicit deferrals

1. **Local control-plane MVP:** CP-01/02/03; one approved CLI flow, durable
   timeline, indexed evidence and restart-safe bounded execution.
2. **Operator product slice:** CP-04/05; API/catalog and a usable thin UI.
3. **Portable worker slice:** CP-06, with the placement contract designed early.
4. **Qualified release slice:** CP-07, without blocking ordinary public
   development on the entire custody build.

CP-08 runs throughout. Start implementation with the first checkpoint.
Do not add another receipt-training round to any acceptance criterion.

Keep parked: synthetic quality runs, the two unscored factorial candidates,
weight-edit research, full SFT, new datasets/model downloads, fresh holdouts and
final-test scoring. Defer distributed scheduling, Kubernetes/Ray, Postgres,
cloud object storage and broad UI polish until a concrete product need.

## 6. Source evidence and review boundary

Representative inspected source locations:

- Registry resolution: [proof_registry.py](../python/neural_foundry/proof_registry.py), lines15–55.
- Remaining task coupling: [proof_plugins.py](../python/neural_foundry/proof_plugins.py), lines184 and246.
- Hardcoded public validation: [proof_pipeline.py](../python/neural_foundry/proof_pipeline.py), lines36–68.
- Existing lifecycle and persisted records: [gym_lifecycle.py](../python/neural_foundry/gym_lifecycle.py), lines90–175 and375–435.
- Attempt database/state chunks: [gym_service.py](../python/neural_foundry/gym_service.py), lines395–505.
- Independent budget authority: [public_budget.py](../python/neural_foundry/public_budget.py), lines49–151.
- File status projection: [execution_runner.py](../python/neural_foundry/execution_runner.py), lines30–57.
- One-stage public CLI: [run_public_proof.py](../scripts/run_public_proof.py), line15.
- Specialized frontier controller: [cord_frontier.py](../python/neural_foundry/cord_frontier.py), lines154–235.
- Exact snapshot approval: [gyms.py](../python/neural_foundry/gyms.py), line59.
- Existing approval/eligibility primitives: [autonomous.py](../python/neural_foundry/autonomous.py), line600; [promotion_gate.py](../python/neural_foundry/promotion_gate.py), line152.
- SDK model inventory: [studio.py](../python/neural_foundry/studio.py), lines124–220.
- Legacy custodian evaluator dispatch: [evaluators.py](../python/neural_foundry/study/evaluators.py), line167.
- Native Linux CI: [m1.yml](../.github/workflows/m1.yml), lines7 and27–54.

The backlog review also covered the existing integration roadmap and Phase A/B/
custody documentation. Older synthetic quality blockers and company-facts-only
claims are historical; they are not current permission or a reason to block
control-plane development.

No full audit, new model experiment or test-suite run was performed for this
planning document. Existing signed-patch/approval primitives must not be
mistaken for a qualified signed control-plane release workflow.
