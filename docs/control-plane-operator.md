# Private control-plane operator guide

This is the PostgreSQL-backed, single-host Neural Foundry control plane, not the
separate `apps/chaxa` SQLite application. It neither replaces nor migrates Chaxa.
The API, CLI and browser are thin adapters over the same durable service; only
workers execute stages.

## Start the private stack

From the repository root in WSL/Linux:

```sh
python3 deploy/control-plane/scripts/bootstrap.py
sh deploy/control-plane/scripts/cp.sh up --build -d
sh deploy/control-plane/scripts/cp.sh ps
```

Docker must be available to WSL. Bootstrap generates private secret files under
`artifacts/control-plane-local/secrets` without printing values or overwriting
existing secrets. Keep that directory out of version control and backups intended
for public sharing. WSL filesystem permissions must support private secret files.
See [deployment operations](control-plane-deployment.md) for deployment-specific
requirements, backup and recovery.

Open **http://127.0.0.1:8787**. The API serves the browser assets directly; there is
no Node build, external CDN, separate frontend server or browser-side workflow engine.
The default published port is loopback-only. Do not expose it on a public interface.
The stack's migration step runs before API/worker startup. PostgreSQL and artifacts
use persistent Docker volumes; deleting volumes destroys their contents.

The login accepts the operator token from the private generated `operator_token`
file. Obtain it through your local secret-management workflow; do not paste it into
chat, URLs, screenshots or shell arguments. The password input is cleared when
submitted. Browser credentials remain in JavaScript memory only, not cookies,
local/session storage or the URL. Reloading locks the console and never creates,
approves or starts a run. Locking does not cancel server-side work.

For an already provisioned database, install the declared `control-plane` optional
dependency group and configure:

| Variable | Purpose |
|---|---|
| `CP_DATABASE_URL` | Private PostgreSQL DSN; required at ASGI startup |
| `CP_ARTIFACT_ROOT` | Required artifact directory; share durably with workers |
| `CP_OPERATOR_TOKEN_FILE` | Preferred token file; takes precedence over the environment token |
| `CP_OPERATOR_TOKEN` | Alternative runtime secret; never put it in tracked configuration |
| `CP_ALLOWED_HOSTS` | Optional comma-separated exact hostnames, without ports; default `localhost,127.0.0.1,::1` |

Start the API with the project environment:

```sh
python -m uvicorn neural_foundry.control_plane.api:app \
  --host 127.0.0.1 --port 8787 --no-access-log
```

Importing the ASGI module does not connect to a database. Lifespan startup fails
explicitly when required configuration is absent. `/health` confirms process
responsiveness; `/ready` checks database/schema readiness without returning a DSN,
schema document or private details. Readiness is not a worker-liveness guarantee.
Use deployment health checks and inspect worker/stage events as well.

## Product workspace

The root page is the project-centered model workspace. It organizes capabilities
into customer workstreams rather than requiring users to assemble scheduler
stages. The advanced run console remains at `/operations`, in a separately
unlocked browser session. Both use the same PostgreSQL service, artifact backing,
authentication boundary, and worker.

1. **Create a project.** Projects persist models, datasets, gyms, recipes,
   recorded-answer evidence, and linked experiments. Counts come from stored
   resources, not example dashboards.
2. **Register a model reference.** Record the model ID, revision, source and
   optional declared weights digest. Registration does not download weights or
   independently verify the checkpoint.
3. **Import examples.** JSON/JSONL imports map input/expected-answer columns,
   retain explicit training/validation splits, and show defaults for missing IDs
   or splits. Expected answers are JSON objects. The server validates bounds,
   identities and cross-split overlap before storing an immutable asset version.
   These are development datasets, not sealed-test custody.
4. **Build a gym.** Connect a JSON-extraction dataset, or configure a supported
   stateful-counter or no-side-effect tool-workflow template. **Try this gym**
   uses real bounded task logic. Manual stateful actions are replayed from the
   declared seed within a trusted software fixture and closed after each request.
   This is not a real browser, 3D renderer, or policy-training session. Publishing
   a gym stores its definition without starting any model.
5. **Evaluate recorded answers.** Import candidate responses linked to validation
   example IDs, including malformed JSON responses as strings. Select a frozen
   gym and evidence assets from the same dataset. Review the resulting durable
   evaluation plan, authorize its exact limits, and explicitly start it. The
   worker computes actual metrics from saved outputs with pinned source
   identities; this is not new inference or independent final evaluation.
6. **Prepare an improvement.** Fine-tuning, weight-edit and composition screens
   save typed, versioned recipes linked to project assets. A saved recipe is not
   a trained model. Launch requires a registered qualified runtime, compatible
   actual inputs and an exact approved plan; entering a profile name cannot
   enable execution. Weight-edit scientific effectiveness remains experimental.
7. **Inspect and export.** Compare measurements only within matching frozen
   gym/dataset/metric-policy contexts. The evidence manifest export preserves
   project provenance, asset definitions (including saved examples/responses)
   and artifact references. Protect it as private project data. The platform does
   not add runtime credentials, model weight files, a signed deployment, or a
   release authorization.

Asset versions are immutable. Reopening a published gym creates a new version
instead of changing old evaluation inputs. BrowserGym, FinRL, real 3D engines,
new provider/model calls, and final-test custody are not enabled by these forms.
The deployment remains private and single-operator, not a public multi-tenant SaaS.

Model, dataset and recorded-answer inspectors expose **Create next version** for
the current version. Old gyms and experiments keep their original references.
Composition plans must name at least two distinct project-owned source models,
including the selected base. Saved-output evaluation is bounded to 256 validation
examples per candidate in this deployment; larger datasets are not silently
truncated.

An operator-installed fine-tuning runtime must declare a trusted
`QualifiedGymBinding`: exact project/gym identity, version and content digest,
resolved proof-registry gym reference, gym configuration digest, and supported
method. These records are installed in trusted deployment code, not editable
qualification assertions submitted by a browser. Planning and dispatch recheck
the binding, selected model identity, saved recipe parameters, and exact
train/validation data. A compatible runtime label alone does not grant execution.

## Advanced operations console

1. Unlock the console and open **Create experiment**. In **Guided setup**, name
   the experiment, select one or more available catalog workloads, and label your
   candidates. Each candidate receives its own dependency graph for every selected
   workload. Current candidates use the built-in CPU fixture policy; labels do not
   load models or change its behavior. Settings are generated from plugin schemas;
   blank values keep template/server defaults. Optional limits apply per stage.
   The generated specification can be previewed without executing anything.
   **Advanced JSON** retains arbitrary supported run specifications; switching back
   to guided setup asks before abandoning advanced changes. The service remains
   authoritative for validation. BrowserGym, FinRL and live 3D engines are explicitly
   unavailable until their integrations are separately qualified.
2. **Create draft experiment** persists the specification with an idempotency key. It
   does not execute work.
3. **Generate plan**, expand **Plan & preflight**, and review stages, dependencies,
   runtime profile, model-call flags and aggregate resource limits. A rejected
   preflight is an error, not permission to bypass the gate.
4. Review the editable approved-limits JSON, then **Approve this plan**. The
   approval binds the exact plan digest and limits. It still does not start work.
5. **Start approved run** is a separate, confirmed action. The service validates
   authority; the worker claims durable stages. Start retries reuse a key within
   the browser session. For retries across sessions use the CLI and the original
   key; never blindly invent a new authorization after an uncertain response.
6. Watch stages, progress, attempt counts and the ordered **Live timeline**.
   Polling only performs GETs. The display retains the latest 500 events; the API
   supports durable history retrieval using its sequence cursor.
7. Inspect each stage's artifact, the full run document, and **Outcomes & metric
   policy**. **Cancel run** records cancellation; inspect subsequent durable
   state/events rather than assuming already-issued external work was undone.

Filter experiments by name/ID and status. **Refresh** reads both the selected
experiment and its events; it never creates, approves, starts or retries work.
The lifecycle and next-action guidance explain draft, review, approved, running,
failed and attention-required states. Plan summaries show aggregate resource
ceilings, not cost or completion-time predictions. Results separate execution,
measured quality and release permission; the metrics table preserves candidate,
stage, metric version and unit without averaging unlike gyms.

**Copy setup** copies an existing specification into the advanced editor without
changing the original or creating a run. Unsaved edits are session-only and are
lost on reload; explicitly created drafts remain in PostgreSQL. Controls are
disabled while a command is pending or database readiness is unavailable.
Network timeouts preserve create/start idempotency keys; inspect durable state
before retrying after an uncertain response.

Dependency-free browser-helper regression tests run with
`node --test tests/control_plane_ui.test.mjs tests/control_plane_workspace_ui.test.mjs`
(Node 18+). HTTP/static-boundary tests
run with `python -m pytest -q tests/test_control_plane_api.py`.

Plans also bind the execution implementation and dependency fingerprint. If a
deployment changes that fingerprint, a queued run stops in `needs_attention`
with an audit event instead of executing changed code under an old approval.
Review the change and create a fresh run and approval; do not blindly retry it.

Current executable catalog templates are **bounded CPU software fixtures**.
The installed-wheel deployment reports repository-backed proof-registry
references as unavailable in `proof_registry_status`; catalog inspection does
not attempt to register missing repository files. Those backends still require
an explicitly qualified runtime and deployment permission.
Their outputs verify software contracts, not pretrained-model performance,
GPU/provider quality, production readiness or release authority. No preset
performance results are shown. Existing real receipts or legacy artifacts retain
their own provenance, evidence class and versioned metric policy; importing a
document is not a fresh independent evaluation or authority upgrade.

Live provider/training/proof operations are disabled without a qualified,
explicitly trusted runtime profile. This operator workflow must not download
models, initiate GPU/live-provider runs, or access holdouts. Offline rescoring is
represented by catalog stages/new runs and the same approval gates, not a bypass
endpoint. A completed execution is distinct from quality success; release
authorization remains false unless a separately governed system establishes it.

## HTTP CLI

The CLI uses HTTP only, never a direct database path. Set the token-file path
and origin at runtime; do not pass a token argument:

```sh
export CP_OPERATOR_TOKEN_FILE="$PWD/artifacts/control-plane-local/secrets/operator_token"
export CP_API_URL=http://127.0.0.1:8787
nf-control-plane health
nf-control-plane catalog
nf-control-plane create spec.json --idempotency-key operator-draft-001
nf-control-plane list
nf-control-plane plan RUN_ID
nf-control-plane approve RUN_ID --plan-digest PLAN_DIGEST --limits limits.json
nf-control-plane start RUN_ID --approval-id APPROVAL_ID --idempotency-key operator-start-001
nf-control-plane events RUN_ID --after 0
nf-control-plane cancel RUN_ID
```

Save a catalog template's `spec` as `spec.json`; copy the reviewed plan's `limits`
object into `limits.json`. These are input documents, not secret-token files.
Module invocation, `python -m neural_foundry.control_plane.cli`, is equivalent.
Use stable, distinct idempotency keys per intended creation/start; reuse the same
key for retries of that action. CLI failures intentionally avoid printing raw
proxy responses, credentials or tracebacks. Remote CLI origins require HTTPS;
redirects are not followed with authorization.

On PowerShell, environment assignment uses `$env:CP_OPERATOR_TOKEN_FILE = '...'`
and `$env:CP_API_URL = 'http://127.0.0.1:8787'`.

## HTTP v1 boundary

Every `/v1` route requires `Authorization: Bearer …`, including catalog, runs,
events and artifacts. No URL-token authentication is accepted.

| Method | Route | Body / parameters |
|---|---|---|
| GET | `/health` | Public, minimal process status |
| GET | `/ready` | Public, minimal database readiness; 503 when unavailable |
| GET | `/v1/catalog` | Plugin descriptors and versioned template specs |
| POST | `/v1/runs` | `{ "spec": {...}, "idempotency_key": "..." }` |
| GET | `/v1/runs` | `limit=1..100`, default 100 |
| GET | `/v1/runs/{id}` | Run, stages, attempts, plan, approval, outcome |
| POST | `/v1/runs/{id}/plan` | No body required |
| POST | `/v1/runs/{id}/approve` | `{ "plan_digest": "...", "limits": {...} }` |
| POST | `/v1/runs/{id}/start` | `{ "approval_id": "...", "idempotency_key": "..." }` |
| POST | `/v1/runs/{id}/cancel` | No body required |
| GET | `/v1/runs/{id}/events` | `after=0` or last received event `sequence` |
| POST | `/v1/artifacts` | Existing artifact JSON document |
| GET | `/v1/artifacts/{id}` | Stored artifact |

Events are ordered by durable sequence and returned in bounded pages. Advance
`after` to the last received `sequence` and repeat until no further events arrive.
JSON errors have `{"error":{"code":"...","message":"..."}}`; malformed bodies
are rejected without echoing their contents. Bodies are bounded to 1 MiB.
Unknown/unexpected backend failures return generic errors, not SQL or tracebacks.

The public index/static assets contain no credentials or private run data.
Same-origin Host/Origin checks, constant-time bearer comparison, restrictive
self-only CSP, no-store responses and text-only DOM rendering defend this thin
private boundary. No permissive CORS is enabled. Keep access logging disabled
and do not install middleware that logs authorization, request bodies or DSNs.

There is **one operator role**, recorded as `operator`; no per-user RBAC,
tenant isolation, public signup or production identity provider is implemented.
This is not a production multi-tenant control plane. Before any remote exposure,
add a reviewed TLS/private-network gateway and operational identity/audit policy;
never weaken host/origin checks to `*` or place bearer secrets in URLs.
