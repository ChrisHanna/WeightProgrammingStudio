"""CPU-local, cooperative gym contracts; no engine, inference or tool dispatch."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

from .gym_job import (
    GymJobError,
    _freeze_json,
    _is_digest,
    _nonempty,
    _thaw,
    _validate_schema,
    is_finite_number,
    validate_task_instance,
)

if TYPE_CHECKING:
    from .gym_lifecycle import TaskAdapter


@dataclass(frozen=True)
class RuntimeIdentity:
    name: str
    version: str
    digest: str

    def __post_init__(self) -> None:
        _nonempty(self.name, "identity.name")
        _nonempty(self.version, "identity.version")
        if not _is_digest(self.digest):
            raise GymJobError("runtime identity requires a sha256 digest")


@dataclass(frozen=True)
class ArtifactReference:
    """Opaque backing-store key, never an instruction to fetch a URI."""

    digest: str
    media_type: str
    byte_size: int
    location_id: str
    producer_id: str

    def __post_init__(self) -> None:
        if not _is_digest(self.digest):
            raise GymJobError("artifact requires a sha256 digest")
        if type(self.byte_size) is not int or self.byte_size < 0:
            raise GymJobError("artifact byte size must be a nonnegative integer")
        for name in ("media_type", "location_id", "producer_id"):
            _nonempty(getattr(self, name), name)


@dataclass(frozen=True)
class Payload:
    data: Any
    artifacts: tuple[ArtifactReference, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", _freeze_json(self.data, "payload"))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        if any(not isinstance(ref, ArtifactReference) for ref in self.artifacts):
            raise GymJobError("binary payloads require artifact references")
        if len(self.artifacts) > 64 or len(self.encode()) > 65_536:
            raise GymJobError("payload metadata exceeds inline bound; use artifact references")

    def document(self) -> dict[str, Any]:
        return {"data": _thaw(self.data), "artifacts": [asdict(ref) for ref in self.artifacts]}

    def encode(self) -> bytes:
        return json.dumps(self.document(), sort_keys=True, allow_nan=False).encode("utf-8")

    @property
    def recorded_bytes(self) -> int:
        # Conservative: repeated references are charged again, not deduplicated.
        return len(self.encode()) + sum(ref.byte_size for ref in self.artifacts)


@dataclass(frozen=True)
class ValueDescriptor:
    kind: Literal["text", "structured", "discrete", "continuous", "binary"]
    schema: Mapping[str, Any]
    artifact_media_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"text", "structured", "discrete", "continuous", "binary"}:
            raise GymJobError("unknown value descriptor kind")
        _validate_schema(self.schema, "descriptor")
        object.__setattr__(self, "schema", _freeze_json(self.schema, "descriptor"))
        object.__setattr__(self, "artifact_media_types", tuple(self.artifact_media_types))
        if self.kind == "discrete" and "enum" not in self.schema:
            raise GymJobError("discrete descriptor requires explicit choices")
        if self.kind == "text" and self.schema["type"] != "string":
            raise GymJobError("text descriptor requires string schema")
        if self.kind == "continuous" and self.schema["type"] not in {"number", "array"}:
            raise GymJobError("continuous descriptor requires numeric scalar/vector schema")
        if (self.kind == "continuous" and self.schema["type"] == "array"
                and self.schema["items"]["type"] != "number"):
            raise GymJobError("continuous vectors require numeric items")
        if self.kind == "binary" and not self.artifact_media_types:
            raise GymJobError("binary descriptor requires artifact media types")

    def validate(self, value: Payload) -> None:
        validate_task_instance(self.schema, value.data, field=self.kind)
        if self.kind == "binary" and not value.artifacts:
            raise GymJobError("binary observation requires an artifact reference")
        if any(ref.media_type not in self.artifact_media_types for ref in value.artifacts):
            raise GymJobError("undeclared artifact media type")


@dataclass(frozen=True)
class RuntimeRequirements:
    """Planning requirements, not a claim that a runner enforces these resources."""

    schema_version: int = 1
    compute_device: Literal["cpu", "gpu"] = "cpu"
    render_device: Literal["none", "software", "gpu"] = "none"
    headless: bool = True
    compute_memory_bytes: int = 0
    render_memory_bytes: int = 0
    concurrency: int = 1
    dependencies: tuple[RuntimeIdentity, ...] = ()

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise GymJobError("unsupported runtime requirements schema version")
        if self.compute_device not in {"cpu", "gpu"}:
            raise GymJobError("unknown compute device requirement")
        if self.render_device not in {"none", "software", "gpu"}:
            raise GymJobError("unknown render device requirement")
        if type(self.headless) is not bool:
            raise GymJobError("headless requirement must be boolean")
        for name in ("compute_memory_bytes", "render_memory_bytes", "concurrency"):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name == "concurrency" else 0):
                raise GymJobError(f"invalid runtime requirement {name}")
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        if any(not isinstance(item, RuntimeIdentity) for item in self.dependencies):
            raise GymJobError("runtime dependencies require versioned identities")


@dataclass(frozen=True)
class Capabilities:
    interaction: Literal["one_shot", "stateful_episode", "workflow"]
    observation: ValueDescriptor
    action: ValueDescriptor
    protocol_version: int = 1
    placement: str = "cpu_local"
    runtime_profile_version: str = "1"
    resources: RuntimeRequirements = field(default_factory=RuntimeRequirements)

    def __post_init__(self) -> None:
        if self.protocol_version != 1:
            raise GymJobError("only runtime protocol v1 descriptors are supported")
        _nonempty(self.placement, "runtime profile ID")
        _nonempty(self.runtime_profile_version, "runtime profile version")
        if not isinstance(self.resources, RuntimeRequirements):
            raise GymJobError("resources require typed runtime requirements")
        if self.interaction not in {"one_shot", "stateful_episode", "workflow"}:
            raise GymJobError("unknown interaction capability")


@dataclass(frozen=True)
class ReplayQualification:
    status: Literal["unqualified", "fixture_verified", "not_replayable"] = "unqualified"
    evidence: ArtifactReference | None = None

    def __post_init__(self) -> None:
        if self.status not in {"unqualified", "fixture_verified", "not_replayable"}:
            raise GymJobError("unknown replay qualification")
        if self.status == "fixture_verified" and self.evidence is None:
            raise GymJobError("verified replay requires evidence, not just a seed")


@dataclass(frozen=True)
class EnvironmentSpec:
    identity: RuntimeIdentity
    engine: RuntimeIdentity
    assets: tuple[ArtifactReference, ...]
    capabilities: Capabilities
    replay: ReplayQualification = field(default_factory=ReplayQualification)


@dataclass(frozen=True)
class ToolPermission:
    name: str
    side_effects: bool = False

    def __post_init__(self) -> None:
        _nonempty(self.name, "tool permission")
        if type(self.side_effects) is not bool:
            raise GymJobError("side_effects must be boolean")


@dataclass(frozen=True)
class ToolTrace:
    permission: ToolPermission
    request: Payload
    response: Payload
    receipt_id: str
    idempotency_key: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.receipt_id, "tool receipt")
        if self.permission.side_effects:
            _nonempty(self.idempotency_key, "side-effect idempotency key")


def replay_tool_trace(trace: tuple[ToolTrace, ...]) -> tuple[Payload, ...]:
    """Offline evidence playback only: deliberately accepts no executor."""
    return tuple(entry.response for entry in trace)


@dataclass(frozen=True)
class Measurement:
    name: str
    version: str
    value: float
    unit: str

    def __post_init__(self) -> None:
        for name in ("name", "version", "unit"):
            _nonempty(getattr(self, name), name)
        if not is_finite_number(self.value):
            raise GymJobError("measurement must be finite")


@dataclass(frozen=True)
class Transition:
    observation: Payload
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    simulated_seconds: float = 0.0
    metrics: tuple[Measurement, ...] = ()
    tools: tuple[ToolTrace, ...] = ()

    def __post_init__(self) -> None:
        if not is_finite_number(self.reward):
            raise GymJobError("reward must be finite")
        if not is_finite_number(self.simulated_seconds) or self.simulated_seconds < 0:
            raise GymJobError("simulated time must be finite and nonnegative")
        if type(self.terminated) is not bool or type(self.truncated) is not bool:
            raise GymJobError("termination/truncation flags must be boolean")


@dataclass(frozen=True)
class Budgets:
    episodes: int = 1
    steps: int = 100
    wall_seconds: float = 10.0
    simulated_seconds: float = 100.0
    tool_calls: int = 0
    recorded_bytes: int = 1_048_576

    def __post_init__(self) -> None:
        for name in ("episodes", "steps", "tool_calls", "recorded_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise GymJobError(f"{name} must be a nonnegative integer")
        for name in ("wall_seconds", "simulated_seconds"):
            if not is_finite_number(getattr(self, name)) or getattr(self, name) < 0:
                raise GymJobError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class StepAllowance:
    simulated_seconds: float
    tool_calls: int
    recorded_bytes: int
    permissions: tuple[ToolPermission, ...]


class Environment(Protocol):
    def reset(self, *, seed: int) -> Payload: ...
    def step(self, action: Payload, *, allowance: StepAllowance) -> Transition: ...
    def close(self) -> None: ...


class EnvironmentFactory(Protocol):
    spec: EnvironmentSpec

    def create(self, *, owner_id: str) -> Environment: ...


class Actor(Protocol):
    identity: RuntimeIdentity
    candidate_id: str
    capabilities: Capabilities

    def reset(self, *, seed: int) -> None: ...
    def act(self, observation: Payload) -> Payload: ...


@dataclass(frozen=True)
class StepRecord:
    action: Payload
    transition: Transition


@dataclass(frozen=True)
class EpisodeRecord:
    seed: int
    initial: Payload | None
    steps: tuple[StepRecord, ...]
    terminated: bool
    truncated: bool
    reason: str | None

    @property
    def episode_return(self) -> float:
        return sum(step.transition.reward for step in self.steps)


@dataclass(frozen=True)
class RunRecord:
    protocol_version: int
    owner_id: str
    environment: EnvironmentSpec
    policy: RuntimeIdentity
    candidate_id: str
    episodes: tuple[EpisodeRecord, ...]
    status: Literal["completed", "budget_exhausted", "error"]
    error: str | None
    cleanup_error: str | None
    session_created: bool
    closed: bool
    steps: int
    tool_calls: int
    simulated_seconds: float
    recorded_bytes: int
    wall_seconds: float


def _step_bytes(action: Payload, result: Transition) -> int:
    metadata = json.dumps({
        "reward": result.reward, "terminated": result.terminated,
        "truncated": result.truncated, "simulated_seconds": result.simulated_seconds,
        "metrics": [asdict(metric) for metric in result.metrics],
        "tools": [{"permission": asdict(tool.permission), "receipt_id": tool.receipt_id,
                   "idempotency_key": tool.idempotency_key} for tool in result.tools],
    }, allow_nan=False).encode("utf-8")
    return (len(metadata) + action.recorded_bytes + result.observation.recorded_bytes
            + sum(tool.request.recorded_bytes + tool.response.recorded_bytes
                  for tool in result.tools))


def run_episodes(
    factory: EnvironmentFactory,
    actor: Actor,
    *,
    owner_id: str,
    seeds: tuple[int, ...],
    budgets: Budgets,
    permissions: tuple[ToolPermission, ...] = (),
    clock: Callable[[], float] = time.monotonic,
) -> RunRecord:
    """Own and close one session. Cooperative limits, not process isolation."""
    _nonempty(owner_id, "owner")
    _nonempty(actor.candidate_id, "candidate")
    if actor.capabilities != factory.spec.capabilities:
        raise GymJobError("actor/environment capabilities differ before creation")
    capabilities = factory.spec.capabilities
    if (capabilities.placement != "cpu_local"
            or capabilities.runtime_profile_version != "1"
            or capabilities.resources != RuntimeRequirements()):
        raise GymJobError("CPU fixture runner does not support requested runtime profile/resources")
    if not seeds or any(type(seed) is not int for seed in seeds):
        raise GymJobError("explicit integer episode seeds required")
    if len(seeds) > budgets.episodes:
        raise GymJobError("episode budget exceeded before creation")
    if len({permission.name for permission in permissions}) != len(permissions):
        raise GymJobError("tool permissions must have unique names")
    started = clock()
    session = None
    closed = False
    records: list[EpisodeRecord] = []
    steps = calls = recorded = 0
    simulated = 0.0
    status: Literal["completed", "budget_exhausted", "error"] = "completed"
    error = None
    cleanup_error = None
    initial = None
    current: list[StepRecord] = []
    active_seed = seeds[0]

    def exhausted() -> str | None:
        if clock() - started >= budgets.wall_seconds:
            return "wall_seconds"
        if steps >= budgets.steps:
            return "steps"
        if simulated >= budgets.simulated_seconds:
            return "simulated_seconds"
        if recorded >= budgets.recorded_bytes:
            return "recorded_bytes"
        return None

    try:
        if clock() - started >= budgets.wall_seconds:
            status = "budget_exhausted"
            records.append(EpisodeRecord(seeds[0], None, (), False, True, "wall_seconds"))
        else:
            session = factory.create(owner_id=owner_id)
            for seed in seeds:
                active_seed, initial, current = seed, None, []
                reason = exhausted()
                if reason:
                    records.append(EpisodeRecord(seed, None, (), False, True, reason))
                    status = "budget_exhausted"
                    break
                actor.reset(seed=seed)
                reason = exhausted()
                if reason:
                    records.append(EpisodeRecord(seed, None, (), False, True, reason))
                    status = "budget_exhausted"
                    break
                initial = session.reset(seed=seed)
                factory.spec.capabilities.observation.validate(initial)
                recorded += initial.recorded_bytes
                observation = initial
                terminated = truncated = False
                while True:
                    reason = exhausted()
                    if reason:
                        truncated, status = True, "budget_exhausted"
                        break
                    action = actor.act(observation)
                    factory.spec.capabilities.action.validate(action)
                    reason = exhausted()
                    if reason or action.recorded_bytes > budgets.recorded_bytes - recorded:
                        reason = reason or "recorded_bytes"
                        truncated, status = True, "budget_exhausted"
                        break
                    allowance = StepAllowance(
                        budgets.simulated_seconds - simulated, budgets.tool_calls - calls,
                        budgets.recorded_bytes - recorded, permissions,
                    )
                    steps += 1
                    result = session.step(action, allowance=allowance)
                    calls += len(result.tools)
                    simulated += result.simulated_seconds
                    recorded += _step_bytes(action, result)
                    current.append(StepRecord(action, result))
                    factory.spec.capabilities.observation.validate(result.observation)
                    if any(tool.permission not in permissions for tool in result.tools):
                        raise GymJobError("environment reported an unapproved tool call")
                    if calls > budgets.tool_calls or simulated > budgets.simulated_seconds:
                        raise GymJobError("environment exceeded its cooperative step allowance")
                    terminated, truncated = result.terminated, result.truncated
                    reason = None
                    if recorded > budgets.recorded_bytes or clock() - started >= budgets.wall_seconds:
                        reason = ("recorded_bytes" if recorded > budgets.recorded_bytes
                                  else "wall_seconds")
                        truncated, status = True, "budget_exhausted"
                    if terminated or truncated:
                        break
                    if factory.spec.capabilities.interaction == "one_shot":
                        raise GymJobError("one-shot environment did not finish in one step")
                    observation = result.observation
                records.append(EpisodeRecord(
                    seed, initial, tuple(current), terminated, truncated, reason,
                ))
                current, initial = [], None
                if status != "completed":
                    break
    except Exception as exc:  # noqa: BLE001 -- preserve partial evidence at the plugin boundary
        status, error = "error", type(exc).__name__
        records.append(EpisodeRecord(
            active_seed, initial, tuple(current),
            current[-1].transition.terminated if current else False,
            current[-1].transition.truncated if current else False, error,
        ))
    finally:
        if session is not None:
            try:
                session.close()
                closed = True
            except Exception as exc:  # noqa: BLE001 -- cleanup failure is a distinct run error
                cleanup_error = type(exc).__name__
                status, error = "error", error or f"close:{cleanup_error}"
    return RunRecord(
        1, owner_id, factory.spec, actor.identity, actor.candidate_id, tuple(records),
        status, error, cleanup_error, session is not None, closed,
        steps, calls, simulated, recorded, clock() - started,
    )


def task_adapter_capabilities(task: TaskAdapter) -> Capabilities:
    """Reuse the actual frozen GymJob schemas without changing legacy jobs."""
    task.identity.document()
    task.projection.document()
    if task.interaction_type != "single_response":
        raise GymJobError("only single_response TaskAdapters have a one-shot bridge")
    return Capabilities(
        "one_shot", ValueDescriptor("structured", task.job.task_input_schema),
        ValueDescriptor("structured", task.job.task_output_schema),
    )


@dataclass
class TaskAdapterActor:
    """Thin project -> callable policy bridge; no model assumptions or inference."""

    task: TaskAdapter
    identity: RuntimeIdentity
    candidate_id: str
    infer: Callable[[Mapping[str, Any]], Any]

    @property
    def capabilities(self) -> Capabilities:
        return task_adapter_capabilities(self.task)

    def reset(self, *, seed: int) -> None:
        pass

    def act(self, observation: Payload) -> Payload:
        from .gym_service import ExecutionResponse

        self.capabilities.observation.validate(observation)
        request = self.task.project(_thaw(observation.data), None)
        if request.get("format") != self.task.interaction_format:
            raise GymJobError("task projection produced an undeclared interaction format")
        inferred = self.infer(request)
        response = Payload(inferred.response if isinstance(inferred, ExecutionResponse) else inferred)
        self.capabilities.action.validate(response)
        return response


class OneShotTaskSession:
    def __init__(
        self, capabilities: Capabilities, task_input: Payload,
        measure: Callable[[Any], tuple[Measurement, ...]],
    ) -> None:
        self.capabilities, self.task_input, self.measure = capabilities, task_input, measure
        self.closed, self.finished = False, True

    def reset(self, *, seed: int) -> Payload:
        if self.closed:
            raise GymJobError("session is closed")
        self.finished = False
        return self.task_input

    def step(self, action: Payload, *, allowance: StepAllowance) -> Transition:
        if self.closed or self.finished:
            raise GymJobError("open reset session required")
        self.capabilities.action.validate(action)
        self.finished = True
        return Transition(
            self.task_input, terminated=True, metrics=self.measure(_thaw(action.data)),
        )

    def close(self) -> None:
        self.closed = True


class OneShotTaskFactory:
    """A single supplied input, not a dataset/holdout reader or scorer registry."""

    def __init__(
        self, task: TaskAdapter, task_input: Any, *,
        measure: Callable[[Any], tuple[Measurement, ...]],
    ) -> None:
        capabilities = task_adapter_capabilities(task)
        self.task_input, self.measure = Payload(task_input), measure
        capabilities.observation.validate(self.task_input)
        self.spec = EnvironmentSpec(
            RuntimeIdentity(task.identity.adapter_id, task.identity.version, task.job.digest),
            RuntimeIdentity(task.projection.adapter_id, task.projection.version,
                            task.projection.implementation_digest),
            (), capabilities,
        )

    def create(self, *, owner_id: str) -> OneShotTaskSession:
        return OneShotTaskSession(self.spec.capabilities, self.task_input, self.measure)
