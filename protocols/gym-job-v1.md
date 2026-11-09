# Gym Job and Studio API v1

This contract binds a business job to typed inputs/outputs, versioned scenario
sources, scorer/verifier implementations, candidate methods, and required gates.
It is a data contract for SDK/service clients. Loading a job does not import code,
execute scenarios, trust caller metrics, approve a Gym, or authorize release.

## Job document

The strict JSON root has `schema_version: 1`, `kind: "nf_gym_job"`, a stable
`job_id`, display `title` and `description`, `task`, non-empty
`scenario_sources`, `scorer`, `verifier`, `candidate_methods`, `gates`, and
`metadata`. Unknown fields and duplicate JSON keys are rejected. The complete
document is bounded to 1 MiB and receives a canonical content digest.

An optional `backends` object pins any of the `task`, `model`, `trainer`, and
`evaluator` roles to an exact `adapter_id@version`. The lifecycle rejects a
selected adapter that differs from a pinned role before execution. Older jobs
without `backends` keep their original document and digest.

`task.input_schema` and `task.output_schema` use a bounded, closed JSON Schema
subset: object, array, string, number, integer, and boolean types; properties,
required fields, enums, item/string bounds, and numeric bounds. Objects reject
undeclared fields. `$ref`, executable callbacks, and unbounded schemas are not
accepted.

Every scenario source declares a unique `source_id`, `provider_id`, provider
version and implementation digest, a content digest, `origin` (`real` or
`simulated`), split roles, and provider parameters. The union of sources must
cover train, validation, and holdout. Combining multiple sources is declarative;
the provider/runtime must still materialize the declared source and prove that
the resulting split identities match the contract. The in-process service invokes
only explicitly registered provider callables for public train/validation splits;
holdout materialization is refused and must move behind a native custodian before
it can support release gates. Public manifests contain scenario inputs and
reference digests; bounded reference payloads are passed only to verifier/scorer
adapters, never to candidate executors.

Materialization stores a canonical byte snapshot internally and returns a detached
document. Provider-owned inputs and caller mutations cannot change that snapshot.
Every lookup, including plan freezing, verifies the manifest digest. Executors
receive detached scenario inputs rather than the internal manifest.

Scorer and verifier references each bind component ID, version,
implementation digest, and parameters. Candidate methods have the same ID/version/
implementation-digest/parameters shape. A digest is an identity, not proof that
the named code is correct. Implementations are resolved only from the explicit
in-process `GymComponentRegistry`; a job document cannot install or import them.
The public attempt path invokes registered candidate executors in batches, checks
responses against `task.output_schema`, then invokes the registered verifier and
scorer with the private reference. These callables are trusted in-process code,
not an untrusted-code sandbox; only source IDs/versions/digests are validated,
not code correctness. Candidate executors must be registered against exact
candidate model/artifact digests. The built-in service does not establish that an
endpoint is serving those bytes, provide a CUDA inference runtime/KV cache, or
authenticate any measurement. Caller-supplied metric observations remain
`reported_*`, not measured evidence.

Each gate names a purpose (`task`, `behavior`, `controls`, `reproducibility`, or
`safety`), metric, split, comparator, finite threshold, and required flag. Every
job must include required task, behavior, controls, and reproducibility gates.
Safety gates are available for jobs that require them. Gate IDs are unique and
the purpose/split combinations are constrained, for example controls use the
`controls` split and reproducibility uses `reproducibility`.

## Dashboard-neutral API

`neural_foundry.gym_service.GymStudioService` exposes the versioned
`nf_gym_studio_api_v1` operations `capabilities`, `validate_job`,
`materialize_public_scenarios`, `freeze_practice_plan`, `run_practice_batch`,
`record_practice_run`, `compare_unchanged_bases`, and `score_candidate`.
`WeightProgrammingStudio` exposes these through `describe_gym_api`,
`register_gym_component`, `register_gym_candidate_executor`,
`validate_gym_job`, `materialize_gym_scenarios`, `freeze_gym_practice_plan`,
`run_gym_practice_batch`, `record_gym_practice_run`,
`compare_unchanged_gym_bases`, and `score_gym_candidate`.
These are in-process service methods; no web server or UI is included. Their
JSON-compatible documents are the boundary a future dashboard can transport.

Caller attempt submissions contain only `attempt_id` and the raw JSON `response`,
inside `{practice_plan, attempts}`. The service validates, verifies, and scores
the response and forces `execution_kind: caller_supplied` with no executor digest.
Server-generated attempt documents include execution fields; submitting these is
readback validation only and requires an identical record already saved by the
service. A caller cannot insert a claimed `registered_candidate` result. Run
documents separately count caller submissions and registered executions;
`complete` means roster completion, not proof that a model ran or passed quality.
Contradictory status/validity combinations are rejected. Invalid outputs are
recorded explicitly without passing malformed data to scorer/verifier adapters.
Scalars, arrays, objects, and bounded private JSON references use typed digest
records; legacy object-reference digests remain readable. Unsigned legacy attempt
files are imported as reported caller data, never upgraded to model-run evidence.

Practice plans freeze expected attempt IDs before dispatch. Runs verify scenario
membership and scorer/verifier identities and become complete only when every
planned attempt has a terminal record. When `gym_state_dir` is configured, plans,
public scenario bundles, private verifier references and terminal attempts are
persisted with owner-only directory/file modes. Terminal attempts use SQLite
transactions keyed by `(plan_digest, attempt_id)`; a conflicting replacement is
rejected, and disjoint worker results do not overwrite each other. Legacy attempt
files are imported transactionally. Each result writes one row, not the complete
history. Reads refresh the saved roster across service instances. After restart,
missing attempts remain incomplete and the same frozen manifest/plan must be
recovered. Comparison retries dispatch only missing attempts, skip completed
bases, and rebuild results from saved records. Derived run JSON is a snapshot;
the attempt database is authoritative for recovery. The local store is not
encrypted, tamper-proof native custody, or a distributed worker coordinator.

Each dispatch claims at most 128 attempt IDs transactionally before invoking an
executor. Process-owned POSIX locks prevent a live second worker from executing
the same IDs, and process death permits recovery of stale database claims. Other
executor subprocesses must retain the owning lease for their full lifetime; this
is not an independently qualified distributed lease protocol. Terminal lookups
use indexed keys rather than reloading the full history per attempt. Executor
exceptions are recorded separately as retryable infrastructure errors; they do
not permanently fail the requested roster. Returned model outputs are terminal
scored or invalid outcomes. Comparison retries chunk only missing IDs into bounded
batches. Claims on native Windows require WSL/POSIX.

The count ceiling is 100,000 attempts, subject to a 256 MiB aggregate state bound.
Native canonical JSON remains capped at 1 MiB per node. Large documents are stored
as content-addressed bounded chunks with a compact `nf_gym_chunk_store_v1` root.
Nodes distinguish values, object-field references, array chunks, and single large
structured array items; every node is native-canonicalized and digest-verified.
Large document identities use the domain-separated `nf_gym_chunk_root_v1` record.
Expanded API documents are detached views, not a request to raise native limits.
Restore limits include referenced chunk bytes and depth. Large responses use the
same chunked identity scheme; raw response text is not stored in attempt records.
Practice attempt records contain IDs, round/status, response/executor digests,
and finite metrics, not raw prompt or response text. Candidate scorecards evaluate
every declared gate using the same profile for every declared method and retain
`reported_pass`, `reported_fail`, `not_run`, or `blocked` status for each gate.

Metric observations accepted by this Python API are caller-provided reports, not
authenticated measurements. The service labels their outcome `reported_*`,
includes `metric_inputs_authenticated: false`, and always returns
`release_authorized: false`. A complete reported pass means only
`reported_pass_pending_verification`. Candidate artifact digests do not prove
which checkpoint is running at an endpoint; scorecards mark served-model identity
unverified. Native independent custody, serving-identity verification, and
explicit owner approval remain separate prerequisites.

Independent frontier evaluation holds an exclusive OS-backed lease for the whole
evaluation, including report persistence. A live evaluator blocks a second owner;
process exit releases the lease without a stale-timeout guess. On POSIX the
isolated child inherits the lease descriptor, so killing its parent does not
permit another evaluator while that child survives. Isolated evaluation on
Windows requires WSL; native Windows lease inheritance is not qualified. Only
the identical frozen evaluation request can resume after the prior owner exits.
These local leases do not establish independent holdout custody.

Evaluator cache requests bind tensor values, actual tensor dtype/device, Torch
math mode, model configuration, tokenizer vocabulary/merges/added tokens, target
panel, and evaluator configuration. Only reports with `complete: true` finalize
claims. Artifact/load failures stay retryable and cannot become cached success.
Search limits reject nonfinite and fractional budgets, share evaluation caps across
direct/control/training checkpoints, and gate new dispatch between work units by
monotonic time. Active in-process native operations are not an independently
qualified hard-kill sandbox; external workers have bounded subprocess supervision.

## Local CUDA retry demonstration

`scripts/run_gym_cuda_comparison.py` runs a four-sentence public synthetic contract
classification panel through two distinct pinned Pythia checkpoints. It uses
the existing GPT-NeoX forward implementation on CUDA float32 and constrained
next-token A/B/C choices, not canned outputs or free-form clause extraction.
Receipts record the immutable revisions, checkpoint/config/tokenizer file hashes,
loaded device and precision, actual choice logits, and completed attempt IDs.
This is the named `pythia_constrained_contract_smoke_v1` fixture. The driver kills
a paused worker after the second checkpoint commits one result, then starts a
fresh process and checks that only missing attempts run. It does not claim an
in-flight kernel failure test. Checks survive `python -O`; Linux preflight and
bounded progress supervision precede the kill. A completed retry must dispatch
nothing. The CPU reference environment is unchanged.

Run with the separate CUDA interpreter and the project Python source on
`PYTHONPATH`, passing `--output-dir` for a fresh evidence directory. Downloads
require explicit `--allow-download`; only pinned inference files and the model
card are fetched. The optional GPU download client is `huggingface-hub==0.36.0`,
not a mandatory wheel or CPU reference dependency. This is measured local-model
execution evidence, not serving-endpoint authentication, task qualification,
independent holdout evidence, or release approval.

## Generated responses and training evidence

`scripts/run_gym_cuda_generated.py` takes a required run configuration, not fixed
model or label choices. `protocols/contract-generated-cuda-v1.json` is an explicit
example using immutable Qwen3 and Llama-family SmolLM2 revisions, CUDA bfloat16,
a 4096-token context, and source-grounded JSON clause/span extraction, including
both clause types in one contract. It records actual decoded text and token IDs;
it never replaces invalid generation with a Python-built answer. Task failures
remain invalid or failed-verification observations, not invented successful data.
The inference fault emits a marker after the first real forward and continues
generation until killed. The other fault pauses after generation but before
persistence. Both runs verify missing-only recovery and zero completed redispatch.

Strict real-base comparisons require distinct base digests, registered execution
completion, and trusted worker probes of immutable checkpoint revision, loaded
tensor state, config/tokenizer hashes, device, and precision. Distinct-family
requirements use the worker's loaded architecture, not caller display names.
These are trusted local worker identities, not remote serving attestations.

`--training-config protocols/gpu-training-smoke-v1.json --training-only` runs
explicit per-method LoRA/SFT recipes, seeds, checkpoint steps, assigned device,
and precision. It is a two-step execution smoke, not a training-quality benchmark.
Worker dispatch/receipts bind the exact input tensor and config/tokenizer states;
edited models cannot silently reuse stale original-checkpoint references. Learned
deltas are measured against the converted training base and lifted onto the
canonical source; precision conversion drift is reported separately. Reload uses
fresh exported config/tokenizer assets, checks stored dtype, and refuses missing
or mismatched assets and empty probes. Native release qualification is separate.

The optional CUDA environment uses `torch==2.9.1+cu128`,
`transformers==4.57.3`, and `huggingface-hub==0.36.0`; these are not mandatory wheel
or locked CPU reference dependencies. `.github/workflows/gpu-evidence.yml` is a
manual self-hosted Linux/GPU recipe requiring prequalified interpreters and the
explicit local training checkpoint. It uploads JSON/log evidence, not weights.
Adding the workflow is not evidence that a remote runner executed it.

## Example

```json
{
  "schema_version": 1,
  "kind": "nf_gym_job",
  "job_id": "vendor-contract-review",
  "title": "Find renewal and termination clauses",
  "description": "Return source-grounded clauses; abstain when absent.",
  "task": {
    "input_schema": {
      "type": "object",
      "properties": {"contract_text": {"type": "string", "maxLength": 16384}},
      "required": ["contract_text"],
      "additionalProperties": false
    },
    "output_schema": {
      "type": "object",
      "properties": {"clauses": {"type": "array", "items": {"type": "object"}, "maxItems": 32}},

The Python NeMo MCQA bridge, `register_nemo_mcqa_scenario_sources`, registers
train, validation, and holdout panel identities as versioned scenario sources.
It maps each prompt to the public `input` and keeps the expected answer token in
the private verifier reference. The adapter requires an explicit `real` or
`simulated` origin. Registering a holdout source does not make it materializable
through the public service; holdout evaluation remains outside this Python path.
      "required": ["clauses"],
      "additionalProperties": false
    }
  },
  "scenario_sources": [],
  "scorer": {"component_id": "contract_clause_evidence", "version": "1.0.0", "implementation_digest": "sha256:<64 lowercase hex characters>", "parameters": {}},
  "verifier": {"component_id": "citation_grounding", "version": "1.0.0", "implementation_digest": "sha256:<64 lowercase hex characters>", "parameters": {}},
  "candidate_methods": [],
  "gates": [],
  "metadata": {}
}
```

The example's empty source/method/gate arrays are placeholders, not a valid job;
the validator requires real/simulated split coverage, at least one candidate
method, and all four required gate purposes.