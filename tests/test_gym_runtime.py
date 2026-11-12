from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
from neural_foundry.gym_job import (
    AcceptanceGate,
    CandidateMethod,
    ComponentReference,
    GymJobError,
    GymJobSpec,
    ScenarioSource,
)
from neural_foundry.gym_lifecycle import AdapterIdentity, TaskAdapter
from neural_foundry.gym_runtime import (
    ArtifactReference,
    Budgets,
    Measurement,
    OneShotTaskFactory,
    Payload,
    ReplayQualification,
    RuntimeRequirements,
    StepAllowance,
    TaskAdapterActor,
    ToolPermission,
    ToolTrace,
    Transition,
    ValueDescriptor,
    replay_tool_trace,
    run_episodes,
    task_adapter_capabilities,
)
from neural_foundry.gym_runtime_fixtures import (
    LOOKUP_PERMISSION,
    CounterFactory,
    ScriptedActor,
    WorkflowFactory,
    fixture_identity,
)


def run_counter(*, factory=None, budgets=None, actions=(1, 1), seeds=(0,)):
    factory = factory or CounterFactory()
    actor = ScriptedActor(factory.spec.capabilities, tuple(Payload(value) for value in actions))
    result = run_episodes(
        factory, actor, owner_id="attempt-1", seeds=seeds, budgets=budgets or Budgets(),
    )
    return factory, actor, result


def test_stateful_actor_execution_seed_reset_ownership_and_return():
    factory, actor, result = run_counter(seeds=(0, 1), budgets=Budgets(episodes=2))
    assert result.status == "completed"
    assert result.closed and factory.sessions[0].closed
    assert factory.owners == ["attempt-1"]
    assert result.steps == 3
    assert [episode.seed for episode in result.episodes] == [0, 1]
    assert [episode.initial.data["position"] for episode in result.episodes] == [0, 1]
    assert [episode.episode_return for episode in result.episodes] == [3.0, 2.0]
    assert all(episode.terminated and not episode.truncated for episode in result.episodes)
    assert result.episodes[0].steps[-1].transition.metrics == (
        Measurement("position", "1", 2, "cells"),
    )
    assert result.policy == actor.identity and result.candidate_id == actor.candidate_id
    assert result.environment.engine.name == "python-software-fixture"
    assert result.environment.replay.status == "unqualified"
    with pytest.raises(GymJobError, match="closed"):
        factory.sessions[0].reset(seed=0)


def test_environment_horizon_and_runner_step_truncation_are_not_termination():
    _, _, horizon = run_counter(factory=CounterFactory(target=8, horizon=1))
    _, _, bounded = run_counter(budgets=Budgets(steps=1))
    for result in (horizon, bounded):
        assert result.episodes[0].truncated
        assert not result.episodes[0].terminated
        assert result.steps == 1 and result.closed
    assert horizon.status == "completed"
    assert bounded.status == "budget_exhausted"
    assert bounded.episodes[0].reason == "steps"


def test_simulated_time_and_recorded_data_limits_stop_further_steps():
    _, _, simulated = run_counter(budgets=Budgets(simulated_seconds=0.25))
    assert simulated.steps == 1
    assert simulated.episodes[0].reason == "simulated_seconds"
    _, _, data = run_counter(budgets=Budgets(recorded_bytes=1))
    assert data.steps == 0
    assert data.episodes[0].reason == "recorded_bytes"
    assert simulated.closed and data.closed


def test_wall_limit_is_checked_after_actor_before_environment_step():
    factory = CounterFactory()
    now = [0.0]

    class SlowActor(ScriptedActor):
        def act(self, observation):
            now[0] = 2.0
            return super().act(observation)

    actor = SlowActor(factory.spec.capabilities, (Payload(1),))
    result = run_episodes(
        factory, actor, owner_id="wall", seeds=(0,), budgets=Budgets(wall_seconds=1),
        clock=lambda: now[0],
    )
    assert result.status == "budget_exhausted" and result.steps == 0
    assert result.episodes[0].reason == "wall_seconds"
    assert result.closed


def test_wall_limit_after_actor_reset_prevents_environment_reset():
    factory = CounterFactory()
    now = [0.0]

    class SlowResetActor(ScriptedActor):
        def reset(self, *, seed):
            now[0] = 2.0
            super().reset(seed=seed)

    actor = SlowResetActor(factory.spec.capabilities, (Payload(1),))
    result = run_episodes(
        factory, actor, owner_id="reset-wall", seeds=(0,), budgets=Budgets(wall_seconds=1),
        clock=lambda: now[0],
    )
    assert result.status == "budget_exhausted" and result.steps == 0
    assert result.episodes[0].initial is None
    assert factory.sessions[0].finished  # Environment reset was not dispatched.
    assert result.closed


def test_preflight_rejects_episode_limit_and_incompatible_actor_before_create():
    factory = CounterFactory()
    actor = ScriptedActor(factory.spec.capabilities, (Payload(1),))
    with pytest.raises(GymJobError, match="episode budget"):
        run_episodes(factory, actor, owner_id="preflight", seeds=(0, 1), budgets=Budgets())
    actor.capabilities = WorkflowFactory.spec.capabilities
    with pytest.raises(GymJobError, match="capabilities"):
        run_episodes(factory, actor, owner_id="preflight", seeds=(0,), budgets=Budgets())
    assert factory.sessions == []


@pytest.mark.parametrize(("placement", "resources"), [
    ("gpu_headless", RuntimeRequirements(
        compute_device="gpu", render_device="gpu", headless=True,
        compute_memory_bytes=1024, render_memory_bytes=512,
        dependencies=(fixture_identity("future-engine"),),
    )),
    ("cpu_local", RuntimeRequirements(compute_device="gpu")),
    ("cpu_local", RuntimeRequirements(render_device="gpu")),
    ("cpu_local", RuntimeRequirements(render_device="software", headless=False)),
])
def test_future_runtime_requirements_are_describable_but_refused_before_creation(
    placement, resources,
):
    factory = CounterFactory()
    capabilities = replace(factory.spec.capabilities, placement=placement, resources=resources)
    factory.spec = replace(factory.spec, capabilities=capabilities)
    actor = ScriptedActor(capabilities, (Payload(1),))
    with pytest.raises(GymJobError, match="runtime profile/resources"):
        run_episodes(factory, actor, owner_id="unsupported", seeds=(0,), budgets=Budgets())
    assert factory.sessions == [] and factory.owners == []
    assert actor.index == 0


@pytest.mark.parametrize("actions", [(7,), (1,)])
def test_invalid_action_and_actor_exhaustion_preserve_partial_evidence_and_close(actions):
    _, _, result = run_counter(actions=actions)
    assert result.status == "error" and result.error == "GymJobError"
    assert result.steps == (0 if actions == (7,) else 1)
    assert len(result.episodes[0].steps) == result.steps
    assert result.closed
    assert not result.episodes[0].terminated
    assert not result.episodes[0].truncated


def test_environment_error_and_close_error_are_not_quality_failures(monkeypatch):
    factory = CounterFactory()
    create = factory.create

    def create_broken(*, owner_id):
        session = create(owner_id=owner_id)

        def broken_step(action, *, allowance):
            raise RuntimeError("internal detail not copied to the receipt")

        session.step = broken_step
        return session

    monkeypatch.setattr(factory, "create", create_broken)
    _, _, result = run_counter(factory=factory)
    assert result.error == "RuntimeError" and result.closed

    def close_broken(*, owner_id):
        session = create(owner_id=owner_id)

        def broken_close():
            raise RuntimeError("cleanup failed")

        session.close = broken_close
        return session

    monkeypatch.setattr(factory, "create", close_broken)
    _, _, result = run_counter(factory=factory)
    assert result.error == "close:RuntimeError" and not result.closed


def test_mock_3d_frame_is_reference_only_and_charged_to_data_budget():
    frame_bytes = bytes(range(12))  # 2x2 RGB8 software fixture, no renderer.
    ref = ArtifactReference(
        "sha256:" + hashlib.sha256(frame_bytes).hexdigest(),
        "application/x-rgb8", len(frame_bytes), "fixture-frame-1", "fixture-producer",
    )
    factory, _, result = run_counter(factory=CounterFactory(frame=ref))
    initial = result.episodes[0].initial
    assert initial.artifacts == (ref,)
    assert initial.data == {"position": 0}
    assert frame_bytes not in initial.encode()
    assert result.environment.assets == (ref,)
    assert initial.recorded_bytes == len(initial.encode()) + len(frame_bytes)
    replay = run_counter(factory=factory)[2]
    assert replay.episodes == result.episodes
    # A test rerun is not a blanket qualification for an engine or external assets.
    assert replay.environment.replay.status == "unqualified"
    with pytest.raises(GymJobError, match="evidence"):
        ReplayQualification("fixture_verified")


def test_workflow_permission_tool_budget_and_offline_no_side_effect_replay():
    factory = WorkflowFactory()
    actor = ScriptedActor(factory.spec.capabilities, (Payload("lookup"), Payload("finish")))
    result = run_episodes(
        factory, actor, owner_id="workflow", seeds=(42,), budgets=Budgets(tool_calls=1),
        permissions=(LOOKUP_PERMISSION,),
    )
    assert result.status == "completed" and result.closed
    assert result.steps == 2 and result.tool_calls == 1
    assert result.episodes[0].episode_return == 2.5
    trace = result.episodes[0].steps[0].transition.tools
    assert replay_tool_trace(trace) == (Payload({"value": 7}),)
    # Even historical side-effect receipts are playback data, not dispatch instructions.
    historical = replace(
        trace[0], permission=ToolPermission("send", True), idempotency_key="send-once",
    )
    assert replay_tool_trace((historical,)) == (historical.response,)
    assert result.episodes[0].steps[-1].transition.metrics[0].name == "success"
    for budget, permissions in ((Budgets(), (LOOKUP_PERMISSION,)), (Budgets(tool_calls=1), ())):
        denied = run_episodes(
            factory, actor, owner_id="denied", seeds=(0,), budgets=budget,
            permissions=permissions,
        )
        assert denied.status == "error" and denied.tool_calls == 0 and denied.closed
        assert factory.sessions[-1].position == 0


@pytest.mark.parametrize("violation", ["tool_count", "simulated", "permission"])
def test_runner_detects_misbehaving_environment_allowance_reports(monkeypatch, violation):
    factory = CounterFactory()
    create = factory.create

    def create_bad(*, owner_id):
        session = create(owner_id=owner_id)

        def bad_step(action, *, allowance):
            tool = ToolTrace(LOOKUP_PERMISSION, Payload(None), Payload(None), "receipt")
            return Transition(
                Payload({"position": 0}), simulated_seconds=2 if violation == "simulated" else 0,
                tools=(tool,) if violation != "simulated" else (),
            )

        session.step = bad_step
        return session

    monkeypatch.setattr(factory, "create", create_bad)
    actor = ScriptedActor(factory.spec.capabilities, (Payload(1),))
    result = run_episodes(
        factory, actor, owner_id="bad", seeds=(0,),
        budgets=Budgets(simulated_seconds=1),
        permissions=() if violation == "permission" else (LOOKUP_PERMISSION,),
    )
    assert result.status == "error" and result.closed
    assert result.steps == 1 and len(result.episodes[0].steps) == 1


def test_terminal_transition_and_time_limit_can_coincide(monkeypatch):
    factory = CounterFactory(target=1)
    create = factory.create

    def create_simultaneous(*, owner_id):
        session = create(owner_id=owner_id)
        step = session.step

        def simultaneous(action, *, allowance):
            return replace(step(action, allowance=allowance), truncated=True)

        session.step = simultaneous
        return session

    monkeypatch.setattr(factory, "create", create_simultaneous)
    _, _, result = run_counter(factory=factory)
    assert result.status == "completed" and result.steps == 1
    assert result.episodes[0].terminated and result.episodes[0].truncated


def test_recorded_volume_exhaustion_preserves_final_transition_and_stops_next_dispatch():
    _, _, baseline = run_counter()
    # Permit the initial observation but less than one transition's full recording volume.
    limit = baseline.episodes[0].initial.recorded_bytes + 1
    _, _, result = run_counter(budgets=Budgets(recorded_bytes=limit))
    assert result.status == "budget_exhausted" and result.steps == 0
    # If action admission fits but the returned evidence is larger, retain that evidence
    # and report the overshoot rather than silently dropping the transition.
    _, _, result = run_counter(budgets=Budgets(recorded_bytes=limit + Payload(1).recorded_bytes))
    assert result.steps == 1 and result.recorded_bytes > limit
    assert result.episodes[0].truncated and result.episodes[0].reason == "recorded_bytes"
    assert result.closed


def test_payload_descriptors_validate_real_discrete_continuous_and_binary_values():
    vector = ValueDescriptor("continuous", {
        "type": "array", "minItems": 2, "maxItems": 2,
        "items": {"type": "number", "minimum": -1, "maximum": 1},
    })
    vector.validate(Payload([0.25, -0.5]))
    for value in ([2, 0], ["x", 0], [1], [float("nan"), 0]):
        with pytest.raises(GymJobError):
            vector.validate(Payload(value))
    with pytest.raises(GymJobError, match="inline bound"):
        Payload("x" * 70_000)
    binary = ValueDescriptor("binary", {"type": "null"}, ("image/png",))
    with pytest.raises(GymJobError, match="artifact reference"):
        binary.validate(Payload(None))


def _one_shot_task():
    digest = fixture_identity("legacy-task").digest
    component = ComponentReference("fixture", "1", digest, {})
    job = GymJobSpec(
        "fixture-task", "Fixture", "Software-only one-shot task",
        {"type": "string"}, {"type": "string", "enum": ["answer"]},
        (ScenarioSource(
            "fixture", "fixture", "1", digest, digest, "simulated",
            ("train", "validation", "holdout"), {},
        ),),
        component, component, (CandidateMethod("fixture", "1", digest, {}),),
        tuple(AcceptanceGate(name, name, "fixture", split, "ge", 0) for name, split in (
            ("task", "validation"), ("behavior", "validation"),
            ("controls", "controls"), ("reproducibility", "reproducibility"),
        )),
        {},
    )
    identity = AdapterIdentity("fixture", "1", digest)
    return TaskAdapter(
        identity, job, "single_response", identity, "fixture_request_v1",
        lambda value, response: {"format": "fixture_request_v1", "input": value},
        lambda service: None,
    )


def test_existing_task_adapter_one_shot_bridge_executes_projection_and_keeps_job_unchanged():
    task = _one_shot_task()
    original = task.job.to_document()
    requests = []

    def infer(request):
        requests.append(request)
        return "answer"

    actor = TaskAdapterActor(task, fixture_identity("callable-policy"), "candidate-1", infer)
    factory = OneShotTaskFactory(
        task, "question", measure=lambda value: (Measurement("exact_match", "1", 1, "ratio"),),
    )
    result = run_episodes(factory, actor, owner_id="legacy", seeds=(3,), budgets=Budgets())
    assert result.status == "completed" and result.closed
    assert requests == [{"format": "fixture_request_v1", "input": "question"}]
    assert result.episodes[0].steps[0].action.data == "answer"
    assert result.episodes[0].terminated and result.steps == 1
    assert result.episodes[0].episode_return == 0  # No invented accuracy-to-reward mapping.
    assert task.job.to_document() == original
    with pytest.raises(GymJobError, match="single_response"):
        task_adapter_capabilities(replace(task, interaction_type="tool_episode"))
    actor.task = replace(task, project=lambda value, response: {"format": "wrong"})
    rejected = run_episodes(factory, actor, owner_id="bad-projection", seeds=(3,), budgets=Budgets())
    assert rejected.status == "error" and rejected.steps == 0 and rejected.closed


def test_one_shot_bridge_accepts_existing_execution_response_wrapper():
    from neural_foundry.gym_service import ExecutionResponse

    task = _one_shot_task()
    actor = TaskAdapterActor(
        task, fixture_identity("wrapped-policy"), "wrapped-candidate",
        lambda request: ExecutionResponse("answer", {"fixture": True}),
    )
    factory = OneShotTaskFactory(task, "question", measure=lambda value: ())
    result = run_episodes(factory, actor, owner_id="wrapped", seeds=(0,), budgets=Budgets())
    assert result.status == "completed" and result.episodes[0].steps[0].action.data == "answer"


def test_fixture_reset_step_close_state_machine_is_enforced():
    factory = CounterFactory(target=1)
    session = factory.create(owner_id="direct")
    allowance = StepAllowance(1, 0, 1024, ())
    with pytest.raises(GymJobError, match="reset"):
        session.step(Payload(1), allowance=allowance)
    session.reset(seed=0)
    assert session.step(Payload(1), allowance=allowance).terminated
    with pytest.raises(GymJobError, match="reset"):
        session.step(Payload(1), allowance=allowance)
    session.close()
    session.close()
    with pytest.raises(GymJobError, match="closed"):
        session.step(Payload(1), allowance=allowance)
