"""Software-only protocol fixtures: no external packages, renderer or real tools."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from .gym_job import GymJobError
from .gym_runtime import (
    ArtifactReference,
    Capabilities,
    EnvironmentSpec,
    Measurement,
    Payload,
    RuntimeIdentity,
    StepAllowance,
    ToolPermission,
    ToolTrace,
    Transition,
    ValueDescriptor,
)


def fixture_identity(name: str) -> RuntimeIdentity:
    return RuntimeIdentity(name, "1", "sha256:" + hashlib.sha256(name.encode()).hexdigest())


@dataclass
class ScriptedActor:
    capabilities: Capabilities
    actions: tuple[Payload, ...]
    candidate_id: str = "scripted-fixture-candidate-v1"
    identity: RuntimeIdentity = field(default_factory=lambda: fixture_identity("scripted-actor"))
    index: int = 0

    def reset(self, *, seed: int) -> None:
        self.index = 0

    def act(self, observation: Payload) -> Payload:
        self.capabilities.observation.validate(observation)
        if self.index >= len(self.actions):
            raise GymJobError("scripted actor exhausted")
        action = self.actions[self.index]
        self.index += 1
        return action


class FixtureSession:
    def __init__(self) -> None:
        self.closed = False
        self.finished = True
        self.position = 0

    def _open(self) -> None:
        if self.closed:
            raise GymJobError("session is closed")

    def close(self) -> None:
        self.closed = True


class CounterSession(FixtureSession):
    def __init__(self, *, target: int, horizon: int, frame: ArtifactReference | None) -> None:
        super().__init__()
        self.target, self.horizon, self.frame = target, horizon, frame
        self.steps = 0

    def _observation(self) -> Payload:
        return Payload({"position": self.position}, (self.frame,) if self.frame else ())

    def reset(self, *, seed: int) -> Payload:
        self._open()
        self.position, self.steps, self.finished = seed % 2, 0, False
        return self._observation()

    def step(self, action: Payload, *, allowance: StepAllowance) -> Transition:
        self._open()
        if self.finished:
            raise GymJobError("reset required before stepping")
        if allowance.simulated_seconds < 0.25:
            raise GymJobError("insufficient simulated-time allowance")
        validate = ValueDescriptor("discrete", {"type": "integer", "enum": [-1, 1]})
        validate.validate(action)
        self.position += action.data
        self.steps += 1
        terminated = self.position >= self.target
        truncated = self.steps >= self.horizon and not terminated
        self.finished = terminated or truncated
        return Transition(
            self._observation(), float(self.position), terminated, truncated, 0.25,
            (Measurement("position", "1", self.position, "cells"),),
        )


class CounterFactory:
    """Optional frame is a caller-owned mock artifact, not rendered simulation."""

    def __init__(
        self, *, target: int = 2, horizon: int = 10, frame: ArtifactReference | None = None,
    ) -> None:
        if target < 1 or horizon < 1:
            raise GymJobError("counter target and horizon must be positive")
        self.target, self.horizon, self.frame = target, horizon, frame
        capabilities = Capabilities(
            "stateful_episode",
            ValueDescriptor("structured", {
                "type": "object", "properties": {"position": {"type": "integer"}},
                "required": ["position"], "additionalProperties": False,
            }, (frame.media_type,) if frame else ()),
            ValueDescriptor("discrete", {"type": "integer", "enum": [-1, 1]}),
        )
        self.spec = EnvironmentSpec(
            fixture_identity("counter-with-mock-frame" if frame else "counter"),
            fixture_identity("python-software-fixture"), (frame,) if frame else (),
            capabilities,
        )
        self.sessions: list[CounterSession] = []
        self.owners: list[str] = []

    def create(self, *, owner_id: str) -> CounterSession:
        session = CounterSession(target=self.target, horizon=self.horizon, frame=self.frame)
        self.sessions.append(session)
        self.owners.append(owner_id)
        return session


LOOKUP_PERMISSION = ToolPermission("fixture.lookup")


class WorkflowSession(FixtureSession):
    def reset(self, *, seed: int) -> Payload:
        self._open()
        self.position, self.finished = 0, False
        return Payload("lookup")

    def step(self, action: Payload, *, allowance: StepAllowance) -> Transition:
        self._open()
        if self.finished:
            raise GymJobError("reset required before stepping")
        expected = "lookup" if self.position == 0 else "finish"
        if action.data != expected:
            raise GymJobError("workflow action out of order")
        if self.position == 0:
            if LOOKUP_PERMISSION not in allowance.permissions or allowance.tool_calls < 1:
                raise GymJobError("lookup requires permission and a tool-call allowance")
            trace = ToolTrace(
                LOOKUP_PERMISSION, Payload({"key": "fixture"}), Payload({"value": 7}),
                "fixture-lookup-receipt-v1",
            )
            self.position = 1
            return Transition(Payload("finish"), tools=(trace,))
        self.finished = True
        return Transition(
            Payload("done"), reward=2.5, terminated=True,
            metrics=(Measurement("success", "1", 1.0, "boolean"),),
        )


class WorkflowFactory:
    spec = EnvironmentSpec(
        fixture_identity("workflow"), fixture_identity("python-no-side-effects"), (),
        Capabilities(
            "workflow", ValueDescriptor("text", {"type": "string"}),
            ValueDescriptor("discrete", {"type": "string", "enum": ["lookup", "finish"]}),
        ),
    )

    def __init__(self) -> None:
        self.sessions: list[WorkflowSession] = []

    def create(self, *, owner_id: str) -> WorkflowSession:
        session = WorkflowSession()
        self.sessions.append(session)
        return session
