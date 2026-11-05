# Scoped worker IPC and fixture inference broker v1

`nf_experiments::coordinator::broker::Session` owns the authoritative ledger/lease, exact source snapshot, artifact store and runtime scope. Workers receive only bounded Welcome/Envelope/Reply data over a private Unix channel. The worker-specific lease_id is a random scope credential, **not** the coordinator Lease/fence/store nonce, DB writer, filesystem root, Python model or Torch handle.

## Authority and access roles

The host/controller-only constructor resolves the frozen gym binding, exact contract/revision, current live coordinator lease and parent budget allocation. It verifies source checkpoint/manifest/parameters/buffers/config/tokenizer identities, actual CPU fixture model/runtime compatibility, and exact item/group inventory for the selected split. Source data remains an immutable Rust SourceState. Model slot `target` permits only registered bounded linear-fixture inference; no weight read/write, custom factory/callback, hidden file, network endpoint, or arbitrary shell API exists.

Actors map exactly to development/proposer, selection/evaluator and holdout/final evaluator. Final scopes require the current Verify stage and the frozen required panel; proposal/selection scopes cannot be created after final verification begins. The current supported assurance is explicitly trusted_fixture_broker + fixture_only. A DTO asking for a stronger separate-process/account generalization claim is unsupported here, not silently mapped to in-process secrecy. T026 must qualify hostile-code isolation before untrusted Python is launched. The actual separate Python client test establishes wire/interface behavior only, not a sandbox escape qualification.

Welcome contains scoped worker/lease IDs, exact contract/revision/binding/split, permitted model/task/artifact/tool names and a byte bound. It contains no target parameters, checkpoint path, DB path, coordinator credentials or live objects. Proposers get task inputs but no label handle. Evaluators may read labels only through their own exact scoped artifact. Unknown/guessed paths or handles are dictionary denials, never filesystem reads. Labels and task IDs retain split/private classification; an agent cannot choose a privileged role by changing its envelope.

## Wire and quotas

Frames use a four-byte big-endian byte length followed by UTF-8 JSON. Frames are capped at 16384 bytes before allocation. Rust canonical parsing limits depth/nodes, rejects duplicate/unknown fields and unsupported variants/versions. Every envelope binds exact scope credentials, contract/revision/split and a sequence. Same sequence/body replays return the original result without another forward; changed input at an existing sequence is an idempotency conflict. Stop/lease/deadline validity is checked before cached results, so replay is not a revocation bypass.

Messages, forwards, input units, tool calls, response bytes, staged payload bytes and transport frames have finite independent limits. Total wire response accounting includes errors and replays. Both channel directions recompute the remaining fixed BOOTTIME scope deadline before each I/O; sending a partial header slowly cannot refresh the lease or block forever. Large scopes that cannot fit a Welcome/frame fail rather than pushing a whole dataset/model through IPC. Larger payload workflows must use separately scoped batches/handles.

Actions are task_input, infer(target/task), bounded exact fixture.sum, artifact(handle), and stage_output(result/document). Infer inputs come from the already assigned task snapshot, not an arbitrary worker-supplied replacement. Every attempted native forward is counted before execution and failures still count; denied slots/tasks do not become model calls. All actual numerical work uses the same Rust f32/f64 registered forward as the transaction smoke path and checks source/numeric/work bounds without mutation.

## Durable costs and exposure receipts

A parent branch allocation holds the worker's finite budget. Setup metadata and each admitted request reserve a child before work; per-request inference evaluations are reconciled exactly, while unmeasured CPU/storage costs remain conservatively charged at their reserved ceilings. These are not invented wall/RSS or total efficiency observations. Successful response replay does not bill the shared forward again. Quotas can be reissued only by a trusted controller; restarting a worker does not reset family costs or revive a closed parent allocation.

Private content-addressed receipts retain scope, exact task/payload identities, artifact-class mappings, requested operations, budget reservation, completion/failure and result identity. Requested access is recorded before model/tool execution; failure/crash cannot be silently relabeled success. Worker scope credentials are stripped from those receipts. The full persistent task exposure/independence and final-feedback policy remains a later ledger gate; receipt identity does not prove semantic task independence.

Staged worker outputs are immutable payload+manifest artifacts with exact contract/worker/binding/split, unreviewed_worker_output classification and staged_not_promoted status. The output slot cannot select a destination path or checkpoint. Raw result JSON, private labels, guessed accepted flags or a successful score cannot advance an accepted model pointer. Scoring validation, export review and private/public report policy remain mandatory later gates.

## Actual conformance tests

`cargo build -p nf-experiments --example fixture_broker` builds a test-support Rust server, not a released gym adapter. `python -m pytest -q -k worker_broker` runs actual private socket-pair IPC, including a separate Python process with only the assigned FD/scope. It executes genuine Rust inference, checks known outputs/source preservation, tools and staged receipts, denies forged leases/splits/contracts/models/paths/labels, checks exact replay/model billing and quotas, exercises partial/oversized frames and role-separated evaluator labels. Missing native binaries fail rather than skip.

The fixture server uses controller-owned source/test records; it does not load arbitrary gym code or claim OS-private holdout isolation under a shared UID. Base SDK import remains free of Torch/network clients. Untrusted gym execution stays off until a real qualified sandbox exists; no in-process exec fallback is introduced.
