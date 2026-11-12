# Gym runtime contracts v1 — software fixtures only

`python/neural_foundry/gym_runtime.py` supplies dependency-free contracts and a
bounded **in-process conformance runner**. `gym_runtime_fixtures.py` supplies
scripted counter/trajectory and two-step workflow fixtures. These are control-plane
integration building blocks, not engine qualification, agent training, model
inference or a second orchestration service.

## Contract and compatibility

- `Capabilities(protocol_version=1)` explicitly selects `one_shot`,
  `stateful_episode` or `workflow`. Descriptors can name future runtime-profile
  IDs through `placement`, with an explicit `runtime_profile_version`.
  `RuntimeRequirements(schema_version=1)` describes CPU/GPU compute,
  none/software/GPU rendering, headless mode, compute/render memory in bytes,
  concurrency and versioned dependencies. These are planning requirements,
  **not** execution qualification or enforced resource reservations.
  Actor and environment descriptors must agree before session creation.
  `run_episodes` separately admits only `cpu_local@1` with default requirements:
  CPU compute, no renderer, headless, one session, no extra dependencies or
  memory reservations. Other profiles or nondefault resource requirements are
  rejected **before factory creation**, even if labelled `cpu_local`.
  Existing CPU defaults and the one-shot bridge remain unchanged.
- A `ValueDescriptor` combines the existing GymJob closed JSON-schema subset
  with text/structured/discrete/continuous/binary semantics. Discrete choices
  are explicit enums; continuous scalars/vectors use numeric constraints and
  array length bounds. Actors are arbitrary policies implementing
  `reset(seed)` / `act(Payload)`, not necessarily LLMs.
- `Payload` freezes JSON metadata, limits it to 64 KiB, and carries at most 64
  binary `ArtifactReference`s. References contain digest, media type, byte size,
  opaque backing-location ID and producer ID. Images, video frames, tensors,
  snapshots and trajectory chunks belong in artifact backing, not inline JSON.
  References do **not** fetch, authenticate, verify or retain backing bytes.
- A factory declares `EnvironmentSpec` and creates a session for an owner ID.
  The runner calls `create -> reset(seed) -> step(action, allowance) -> close`;
  reset repeats between episodes on the same owned session. Implementations
  must reject stepping before reset, after episode completion, or after close.
  `terminated` and `truncated` are independent: **both may be true**.
- `Transition.reward` and the computed episode return are not accuracy.
  The environment's immutable version/digest binds reward semantics.
  Separate `Measurement(name, version, value, unit)` records carry success,
  position or task-specific measurements; no implicit gate or release decision
  is made. Environment truncation can finish execution normally without success.
- `task_adapter_capabilities`, `TaskAdapterActor` and `OneShotTaskFactory` bridge
  existing `single_response` `TaskAdapter`s: frozen GymJob schemas -> existing
  projection/format check -> callable policy -> output validation -> supplied
  measurement callback. Plain responses and existing `ExecutionResponse`
  wrappers are accepted; wrapper interaction metadata remains the legacy
  service's evidence, not automatically copied into a trajectory. This bridge
  neither binds a service nor reads a dataset or holdout. Other legacy
  interaction types are deliberately not reclassified as one-shot.

**No existing TaskAdapter signature, GymJob schema, lifecycle behavior, registry
or serialized job format changes.** Legacy callers need not import this module.
The new runtime does not register parallel plugins, schedule stages, manage
approval or custody, retry operations, or replace the budget ledger. A future
approved worker can invoke this helper and attach its evidence to its attempt.

## Limits, ownership and errors

`Budgets` caps requested episodes, total started environment step calls,
elapsed wall seconds, environment-reported simulated seconds, recorded tool
calls, and recorded bytes. Episode overflow and incompatible descriptors fail
before session creation. Step/wall/recording/simulation admission is checked
before the next actor action/environment step; wall admission is checked again
after actor reset/action and before environment reset/step. Each step receives
remaining simulated seconds, tool calls, recorded bytes and exact tool
permissions in `StepAllowance`. Environments must admit tool calls **before**
dispatch; the workflow fixture does so, before changing its state. Observed
tool/simulation overruns or undeclared permissions are runtime contract errors.

Recorded volume units are **bytes**, conservatively counting encoded JSON
payload/transition/metric/tool metadata plus every referenced artifact's declared
byte size at every appearance (no deduplication). This is not actual disk
allocation, compressed size, or authoritative spending. The fixed run header,
environment asset inventory and Python object overhead are not included.
Initial/returned evidence can itself exceed remaining volume; it is retained,
the overshoot is reported, and no next action is dispatched. Real workers need
bounded artifact writers to enforce hard pre-write storage ceilings.

These are **cooperative** limits for trusted, returning CPU adapters, not
process timeouts or a security sandbox. A blocking create/reset/actor/step/close
cannot be interrupted here. Reported simulated time and tool traces are not
independently verified. There is no actor-side network/tool dispatch facility.
Checking a returned trace is after-the-fact conformance checking: it cannot
prevent a malicious adapter from dispatching an unapproved tool, hiding calls,
or performing side effects. Passing permissions to a trusted environment is
not OS/network isolation or an authenticated external authorization boundary.
Likewise, byte accounting is cooperative observation, not a hard memory or
storage quota. Only offline trace playback lacks a dispatch capability by design.
Future API/model actors require separate authoritative call/compute admission,
and live engines require process/device/render isolation and timeout cleanup.

The runner owns the returned environment session and attempts close in `finally`,
including policy/environment failures and budget stops. The factory owns cleanup
of allocations made before a failed `create` returns. Actor lifetime and artifact
backing remain caller-owned; close must not delete shared assets. Records expose
whether a session was created, whether close succeeded, and a separate cleanup
error. Primary errors retain class names, not potentially sensitive exception
messages. Completed transitions remain available on failure; a failed step
dispatch consumes a step but has no invented transition. There is no automatic
retry or claim of exactly-once execution.

## Evidence and replay

`RunRecord` binds owner/attempt ID, actual seeds, initial observations, environment,
engine and asset identities, policy identity and candidate ID, transitions,
metrics, tool receipts, actual usage and termination/truncation flags.
`ReplayQualification` defaults to `unqualified`; a seed alone proves nothing.
`fixture_verified` requires an evidence artifact and applies only to the fixture
scope named there, not a real engine. Identity digests and evidence backing must
be verified by a future worker/artifact adapter; declaring them is not proof.
The helper returns records; durable/chunked trajectory writing and event indexing
belong to the existing/future worker and control-plane artifact boundary.

Workflow internal task/tool sequencing is confined to the environment, separate
from the outer stage graph. Tool traces bind permissions, request/response,
receipt IDs and (for side effects) idempotency keys. A declared idempotency key
does not establish an external provider's recovery guarantees.
`replay_tool_trace` only returns stored responses and accepts **no executor**:
even historical side-effect receipts cannot trigger dispatch during playback.
Real side effects, permission enforcement outside trusted adapters, uncertain
dispatch recovery and authenticated approvals are not qualified here.

The mock 3D fixture observation holds a caller-owned RGB8 frame reference and a
structured position. Tests construct tiny frame bytes in memory, not a scene,
renderer, physics engine, tensor/model run, network request or download.
The existing `m4_ecosystems.run_gymnasium_adapter` remains unchanged: its toy
dynamics receipt says `external_package_executed=false`; neither module imports
or qualifies live Gymnasium.

## Executable CPU example and checks

```python
from neural_foundry.gym_runtime import Budgets, Payload, run_episodes
from neural_foundry.gym_runtime_fixtures import CounterFactory, ScriptedActor

factory = CounterFactory(target=2)
actor = ScriptedActor(factory.spec.capabilities, (Payload(1), Payload(1)))
record = run_episodes(
    factory, actor, owner_id="fixture-attempt", seeds=(0,), budgets=Budgets(steps=2)
)
assert record.closed and record.episodes[0].terminated
assert record.episodes[0].episode_return == 3.0
assert record.environment.replay.status == "unqualified"
```

In the existing cached CPU environment:

```text
python -m pytest tests/test_gym_runtime.py tests/test_gym_job.py::test_lifecycle_selects_exact_adapter_version tests/test_gym_job.py::test_one_studio_lifecycle_reuses_tasks_bases_and_restart_lineage --basetemp=.gym-runtime-pytest -q
python -m ruff check python/neural_foundry/gym_runtime.py python/neural_foundry/gym_runtime_fixtures.py tests/test_gym_runtime.py
```

Remove the project-local `.gym-runtime-pytest` scratch directory after the run.

Coverage executes stateful and workflow sessions, real legacy projection/schema
bridging, seeded resets, artifact references, termination plus truncation,
pre-dispatch limits, permission denial, allowance violation, playback without
side effects, actor/step/cleanup failures and retained partial evidence. These
are software acceptance fixtures, not a live runtime acceptance certificate.

## Conditional live-integration effort (not authorization)

Rough engineering estimates for one experienced implementer, **after** explicit
approval, with approved assets/dependencies already available:

| Target | Indicative effort and conditions |
| --- | --- |
| Basic Gymnasium | About 2–5 engineering days for one CPU environment with simple spaces, scripted policy, pinned version and reset/step/close smoke/replay evidence. Complex spaces, vectorization, render artifacts and learned policies add scope; package installation and actual execution have not been tested. |
| Selected headless 3D engine | About 1–3 weeks for one simple world **if** the chosen engine already has a stable Python/headless API and approved redistributable assets. Graphics/driver/device isolation, packaging, timing, frame capture and cleanup qualification can extend this to 4–8+ weeks. No engine has been selected or exercised, so this estimate has high uncertainty. |
| Workflow with real side effects | A read-only local adapter may take 3–7 days. One real side-effect integration is more plausibly 2–6+ weeks, conditional on tool APIs, credentials, approvals, sandboxing, receipts, idempotency and recovery semantics. Irreversible actions or uncertain external dispatch may require operator reconciliation rather than replay/retry. |

These estimates exclude quality campaigns, production deployment, distributed
execution, GPU/model inference and independent evaluation/custody qualification.
