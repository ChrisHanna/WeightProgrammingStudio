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

## 3a. Proposed MVP data and execution contract

These are design proposals to finalize in CP-01, not already implemented APIs.

### Supported run shapes from the first delivery

| Shape | Required behavior |
| --- | --- |
| Local evaluation | Evaluate one or several cached candidates against frozen panels. |
| Provider evaluation | Evaluate bounded API candidates with explicit transfer/cost approval. |
| Training | Materialize -> train -> export -> verify -> evaluate -> compare. |
| Weight editing | Qualify base/sites -> capture approved edit/retention inputs -> compute/apply patch -> verify/reload -> evaluate task and preservation. |
| Mixed comparison | Local and provider candidates share a frozen evaluation definition and metric policy. |
| Offline analysis | Import receipts, rescore with an explicit scorer version, compare without inference. |

Candidate model references must be candidate-specific: a comparison can contain
different base models, not only different methods sharing one global base.
Legacy run-level `base_model` becomes a compatibility default, not a limitation
of the new logical model.

Plans use registered operation types and explicit dependencies, not arbitrary
untrusted shell commands or one script per task. Optional approved asset
acquisition is a predecessor; cached/offline execution remains the default.

### Minimum records

| Record | Purpose |
| --- | --- |
| RunDefinition | Immutable requested candidates, assets, panels, policies and purpose. |
| CompiledPlan | Resolved plugin/runtime identities, operation graph, limits and approval requirements. |
| Candidate | Its own base/model/artifact references, method, prompt/projection, schema and decoding identity. |
| Stage | One operation with immutable inputs, dependencies and expected result type. |
| Attempt | One claimed dispatch with operation ID, worker/lease, timestamps and receipt/error references. |
| Event | Append-only ordered run transition/progress/approval/evidence record. |
| Approval | Actor, exact plan digest, permitted operations/destinations, limits, expiry and recovery scope. |
| ArtifactReference | Content digest, type, producer attempt, byte size, permitted locations/access and availability. |
| Outcome | Execution completion, versioned measurements, gate result and separate release eligibility. |
| AuthorityReference | Bound budget/custody/exposure-ledger identity and safe verified state reference. |

Store large payloads in the artifact backing, not database rows. Keep original
specs/receipts immutable when importing historical runs. Migrations version the
metadata schema without rewriting evidence or resetting authoritative ledgers.

### State and recovery rules

Proposed run progression:

`draft -> planned -> awaiting_approval -> queued -> running -> verifying -> completed`

Failures, cancellation, blocked dependencies and uncertain started operations
have explicit states/reasons. A quality rejection is an outcome of a completed
measurement, not an infrastructure failure.

The coordinator writes dispatch intent and claims ownership before executing.
Results are persisted and verified before the completion event. On restart:

- verified committed receipts may be adopted, not recomputed;
- definitely unstarted work may become ready again within its approval;
- ambiguous training/API dispatch stops in `needs_attention`;
- a custodian's started final claim remains consumed under its stricter rules.

Do not promise exactly-once execution at an external provider after a network
timeout. Prevent blind duplicate dispatch; preserve uncertainty, spending
reservations and operator decisions. Checkpoint resume is not implicit MVP
retry support: it requires an explicitly qualified backend continuation contract.

### Local runtime and artifact approach

Use the existing Python application/stage adapters and Rust enforcement
primitives. Begin with one local coordinator service, SQLite and filesystem
artifacts, not a new microservice topology.

Logical specs reference runtime-profile IDs and content-addressed assets.
Worker receipts record resolved executable/image, device, versions and actual
placement. Secrets are runtime references resolved immediately before launch;
they never enter run definitions or evidence payloads.

Coordinator, trainer, reload and serving processes have separate resource
requirements. Resource admission prevents incompatible co-residency and
limits are enforced in the allocating process. The system reports estimates,
reservations, measured usage and unknown values distinctly.

## 3b. Gym families and scale: design now, qualify adapters separately

The target includes **Gymnasium, 3D/simulation and workflow gyms**, not only
one-shot document parsing. Core contracts must accommodate those families
from the beginning. This does not authorize installing engines, starting
simulations, training agents or claiming an unqualified adapter works.

Use explicit capabilities rather than assuming every Gym is a collection of
prompt/reference JSON rows:

| Contract concern | Required extension point |
| --- | --- |
| Interaction | One-shot requests, stateful episodes and multi-step/tool workflows. |
| Environment lifecycle | Session creation, reset/seed, step/action and close; termination and truncation remain distinct. |
| Actor/policy | LLM client, local learned policy or scripted actor, with typed observation/action compatibility. |
| Observations/actions | Text, discrete/continuous values, structured data and references to images, frames, tensors or other binary assets. |
| Evidence | Responses, trajectories, state snapshots, tool traces and replay metadata, linked to the exact candidate/policy version. |
| Metrics | Versioned success/reward/return/length/collision/field metrics; reward is not silently relabelled accuracy. |
| Budgets | Episodes, environment steps, simulated/wall time, tool/API calls, GPU compute/render memory and recorded-data volume. |
| Placement | CPU/GPU compute, graphics/headless rendering, engine/runtime dependencies, approved network/tool access and concurrency limits. |
| Reproducibility | Environment/world/asset/engine versions, seeds, initial state and explicit replay qualification; determinism is not assumed. |

Represent large trajectories/videos/snapshots through chunked artifact
references, not giant event payloads or database rows. Keep detailed simulation
state in the environment/worker boundary; the coordinator tracks lifecycle,
ownership, budgets, progress and evidence references.

The run plan is an operation graph. A workflow Gym's internal task/tool graph is
separate: changing a business workflow must not change the control-plane
scheduler. External tool side effects need explicit permissions, receipts and
idempotency/recovery rules; replaying a trace must not repeat live side effects.

Add stateful-environment and multi-step-workflow **software fixtures** to the
first contract/coordinator acceptance matrix, alongside existing one-shot gyms.
Real Gymnasium/3D adapters are subsequent explicitly qualified integrations.
The existing `run_gymnasium_adapter` is a toy dynamics fixture whose receipt
states `external_package_executed=false`; it is not live Gymnasium integration.

## 3c. Weight editing: first-class product capability, unresolved method research

Weight editing is a major project capability, not a feature to omit because
the first real workload used LoRA.

The current backend has an edit-method interface, capture/calculation/apply
machinery, patch lineage, compatibility probes, reload/serving identity and
preservation checks. Its worker recipe still accepts
`edit_method="legacy_key_value_v1"`. This is implemented machinery, not a
demonstrated generally successful editing method.

The recorded comparable expanded profile had zero net exact-match gain on
train and validation and failed its bf16 preservation gate. Other historical
reports contain invalid comparison configurations and must not be used as
successful acceptance evidence. The v3 five-fix proposal remains proposal-only
on `wip/weight-edit-paused`; it was not implemented or run.

Separate two tracks:

1. **Control-plane integration now:** represent candidate transformation as a
   method capability, not always an optimizer epoch. Accommodate fine-tuning,
   adapter learning and weight editing using existing adapters where possible.
   Edits need target/site/input/cache/retention references, compute budgets,
   patch/full-model output types, transactional apply/rollback, effective
   precision-change diagnostics and task/preservation evidence. Do not force
   edit operations to invent training steps or require `steps == row_count`.
   Qualify orchestration with tensor/software fixtures and existing receipts;
   the legacy method can remain explicitly labelled research/limited.
2. **Method research, separate approval:** the v3 foundation proposes
   value-token-only loss, unnormalized key gradients, Gram-diagonal-scaled
   ridge, all-token covariance/protected-key retention and activation-scaled
   targets. Reconcile this old JSON-specific proposal with the chosen future
   workload before approving implementation or bounded experiments.

Do not treat successful SFT/LoRA as evidence that editing is solved, or make
the entire control-plane build wait for an editing breakthrough. Platform
support belongs in the initial architecture; algorithm effectiveness has its
own evidence and approval gates. No weight-edit research run is authorized.

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

**Acceptance:** one documented CLI entry point uses the same coordinator,
store, approval model and stage contracts to execute all of:

- local evaluation-only;
- bounded provider-only evaluation;
- training -> export -> identity verification -> evaluation;
- offline rescoring/comparison of saved outputs, with zero inference calls.
- candidate-edit/verify/preservation workflows using software/tensor fixtures;
- stateful episode and tool/workflow fixtures, without installing live engines.

Exercise multiple gyms and candidate counts, including mixed local/provider
comparisons, through configuration rather than task-specific controllers.
Use software fixtures and existing receipts; this does not authorize new
model runs. A CORD-specific runner with a database is **not** an acceptable
control-plane MVP.

"Vertical slice" means delivery across planning, approval, execution, evidence
and querying layers. It does not mean a single hardcoded run shape. The
execution primitives already support several operations; the coordinator must
unify them rather than reduce their capabilities.

Indexed comparisons require no manual step ordering. Unapproved work cannot
dispatch. Kill/restart tests cover:

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

Proposed initial API surface:

| Endpoint | Behavior |
| --- | --- |
| `GET /v1/catalog` | Describe plugin/config/capability metadata without executing factories. |
| `POST /v1/runs` | Create/validate a run definition; does not start work. |
| `POST /v1/runs/{id}/plan` | Compile and return an immutable plan/preflight result. |
| `POST /v1/approvals` | Record a bounded authenticated decision for an exact plan. |
| `POST /v1/runs/{id}/start` | Claim approved work idempotently. |
| `POST /v1/runs/{id}/cancel` | Request owned-work cancellation and retain evidence. |
| `GET /v1/runs` / `GET /v1/runs/{id}` | Query current authoritative state and outcomes. |
| `GET /v1/runs/{id}/events?after=sequence` | Read/stream ordered events with a reconnect cursor. |
| `GET /v1/artifacts/{id}` | Read authorized artifact metadata/content by reference. |
| `POST /v1/rescores` | Create an explicitly offline derived run from existing outputs. |

Local binding and operator authentication precede any remotely reachable service.
Error/status contracts distinguish validation, approval, capacity, budget,
execution, evidence and quality outcomes.

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

### Reviewable implementation batches for checkpoint 1

1. **Contracts and capability metadata:** schemas, IDs, operation/plugin
   descriptors, planner and incompatibility checks. Review the public contracts
   before persistence code depends on them.
2. **Persistence and queries:** SQLite migrations, transactional transitions,
   event ordering, artifact indexing and historical-receipt import.
3. **Coordinator and approvals:** dependency scheduling over existing stage
   functions, admission, claims/leases, bounded authorization, progress and
   cancellation. CLI calls this same application service.
4. **Recovery and workflow matrix:** failure injection, reconciliation and
   all supported run shapes, including edit and stateful/workflow fixtures,
   across multiple gyms/candidate counts.

Completion of these batches must yield a useful local control plane, not only
schemas and a database. API/dashboard follow without introducing another
execution implementation.

### Explicit acceptance matrix

- Every supported run shape uses the same coordinator/store/approval paths.
- At least two distinct Gym plugins execute fixture plans; CORD's existing
  real receipts can be imported and rescored without new inference.
- Candidate counts vary and mixed-provider comparisons are configuration-driven.
- A chat/JSON-schema capability mismatch and a missing internal teacher role
  fail before dispatch.
- Materialized data/labels/limits are validated against the actual worker input.
- A new scorer produces a new outcome linked to old responses; originals remain.
- Changed plans/plugins/policies invalidate prior approval.
- Concurrent/repeated start requests do not duplicate attempts.
- Kill/restart boundaries preserve progress and do not repeat uncertain side effects.
- API budget reservations survive restart and cannot be reset by a new UI request.
- Resource/time limits and cancellation apply only to owned processes.
- Completed runs, passed gates and release authorization remain separate.
- All mandatory software acceptance is network/GPU/model-download free.
- No HTTP read, UI refresh or catalog request initiates paid/training work.

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
