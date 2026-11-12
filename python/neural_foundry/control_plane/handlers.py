"""Bounded real CPU fixture operations. No checkpoint, network or live engine loading."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from .models import digest, fail

if TYPE_CHECKING:
    from ..gym_lifecycle import TaskAdapter
    from ..gym_runtime import Budgets, Payload, TaskAdapterActor


class _TaskActor:
    """Expose mutable Actor protocol fields while delegating the existing bridge."""

    def __init__(self, bridge: TaskAdapterActor) -> None:
        self.bridge = bridge
        self.capabilities = bridge.capabilities
        self.identity = bridge.identity
        self.candidate_id = bridge.candidate_id

    def reset(self, *, seed: int) -> None:
        self.bridge.reset(seed=seed)

    def act(self, observation: Payload) -> Payload:
        return self.bridge.act(observation)


def _budgets(bound: dict[str, int], *, tools: bool = False) -> Budgets:
    from ..gym_runtime import Budgets
    return Budgets(episodes=bound["episodes"], steps=bound["steps"],
                   wall_seconds=bound["wall_seconds"], simulated_seconds=256,
                   tool_calls=bound["steps"] if tools else 0,
                   recorded_bytes=bound["recorded_bytes"])


def _score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = []
    for row in rows:
        try:
            expected, actual = json.loads(row["input"]), json.loads(row["output"])
            valid, correct = True, expected == actual
        except (ValueError, TypeError):
            valid, correct = False, False
        scored.append({**row, "valid_json": valid, "exact_match": correct})
    return {"rows": scored, "metrics": [
        {"name": "json_exact_match", "version": "1", "unit": "ratio",
         "value": sum(r["exact_match"] for r in scored) / len(scored)},
        {"name": "json_validity", "version": "1", "unit": "ratio",
         "value": sum(r["valid_json"] for r in scored) / len(scored)}],
            "quality": "passed" if all(r["exact_match"] for r in scored) else "failed"}


def _task() -> TaskAdapter:
    from ..gym_job import (
        AcceptanceGate,
        CandidateMethod,
        ComponentReference,
        GymJobSpec,
        ScenarioSource,
    )
    from ..gym_lifecycle import AdapterIdentity, TaskAdapter
    implementation = digest({"fixture": "control-plane-json-task-v1"})
    component = ComponentReference("cp-fixture", "1", implementation, {})
    job = GymJobSpec(
        "cp-fixture", "JSON software fixture", "Inline development samples only",
        {"type": "string"}, {"type": "string"},
        (ScenarioSource("cp-fixture", "cp-fixture", "1", implementation, implementation,
                        "simulated", ("train", "validation", "holdout"), {}),),
        component, component, (CandidateMethod("cp-fixture", "1", implementation, {}),),
        tuple(AcceptanceGate(name, name, "cp-fixture", split, "ge", 0)
              for name, split in (("task", "validation"), ("behavior", "validation"),
                                  ("controls", "controls"), ("reproducibility", "reproducibility"))), {})
    identity = AdapterIdentity("cp-fixture", "1", implementation)
    return TaskAdapter(identity, job, "single_response", identity, "cp_fixture_v1",
                       lambda value, response: {"format": "cp_fixture_v1", "input": value},
                       lambda service: None)


def local_eval(config: dict[str, Any], bound: dict[str, int],
               dependencies: list[dict[str, Any]], candidate: str) -> dict[str, Any]:
    from ..gym_runtime import Measurement, OneShotTaskFactory, TaskAdapterActor, run_episodes
    from ..gym_runtime_fixtures import fixture_identity
    inputs = config.get("inputs", ['{"value":1}'])
    outputs = config.get("outputs", list(inputs))
    if len(inputs) > bound["steps"]:
        fail("budget_exceeded", "Fixture sample count exceeds approved step limits.")
    task, rows = _task(), []
    for text, output in zip(inputs, outputs):
        def infer(request: Mapping[str, Any], answer: str = output) -> str:
            return answer

        actor = _TaskActor(TaskAdapterActor(task, fixture_identity("cp-json-software-policy"), candidate, infer))
        factory = OneShotTaskFactory(task, text, measure=lambda value: (
            Measurement("response_bytes", "1", len(value.encode()), "bytes"),))
        record = run_episodes(factory, actor, owner_id=candidate, seeds=(0,), budgets=_budgets(bound))
        if record.status != "completed" or not record.closed:
            fail("fixture_execution_error", "TaskAdapter fixture did not complete cleanly.", 409)
        rows.append({"input": text, "output": record.episodes[0].steps[0].action.data})
    return {**_score(rows), "steps": len(rows), "bridge": "gym_runtime.TaskAdapterActor"}


def provider(config: dict[str, Any], bound: dict[str, int],
             dependencies: list[dict[str, Any]], candidate: str) -> dict[str, Any]:
    result = local_eval(config, bound, dependencies, candidate)
    # Exercise a wire-format roundtrip only, never dispatch an HTTP request.
    envelopes = [json.loads(json.dumps({"choices": [{"message": {"content": r["output"]}}]}))
                 for r in result["rows"]]
    rows = [{"input": r["input"], "output": e["choices"][0]["message"]["content"]}
            for r, e in zip(result["rows"], envelopes)]
    return {**_score(rows), "steps": len(rows), "transport": "in_memory_json_roundtrip",
            "transport_success": True, "provider_calls": 0}


def rescore(config: dict[str, Any], bound: dict[str, int],
            dependencies: list[dict[str, Any]], candidate: str) -> dict[str, Any]:
    rows = next((d["rows"] for d in reversed(dependencies) if "rows" in d), None)
    if rows is None:
        return local_eval(config, bound, dependencies, candidate)
    if len(rows) > bound["steps"]:
        fail("budget_exceeded", "Offline rows exceed approved step limits.")
    return {**_score([{"input": r["input"], "output": r["output"]} for r in rows]),
            "steps": len(rows), "offline": True}


def training(config: dict[str, Any], bound: dict[str, int],
             dependencies: list[dict[str, Any]], candidate: str) -> dict[str, Any]:
    import torch
    steps = config.get("steps", 8)
    if steps > bound["steps"]:
        fail("budget_exceeded", "Training transform exceeds approved step limits.")
    x = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64, device="cpu")
    target = 2 * x
    weights = torch.zeros((2, 2), dtype=torch.float64, requires_grad=True)
    before = float((x @ weights - target).square().mean().detach())
    for _ in range(steps):
        loss = (x @ weights - target).square().mean()
        (gradient,) = torch.autograd.grad(loss, (weights,))
        weights = (weights - config.get("learning_rate", .25) * gradient).detach().requires_grad_()
    after = float((x @ weights - target).square().mean().detach())
    return {"tensor": weights.detach().tolist(), "steps": steps, "quality": "not_evaluated",
            "training_loss_before": before, "training_loss_after": after,
            "method": "cpu_linear_gradient_transform", "model_training_claim": False}


def export(config: dict[str, Any], bound: dict[str, int],
           dependencies: list[dict[str, Any]], candidate: str) -> dict[str, Any]:
    tensor = next((d["tensor"] for d in reversed(dependencies) if "tensor" in d), None)
    if tensor is None:
        fail("missing_input", "Export requires a tensor-producing dependency.", 409)
    serialized = json.dumps(tensor, allow_nan=False)
    restored = json.loads(serialized)
    return {"tensor": restored, "tensor_digest": digest(restored), "format": "json-tensor-v1",
            "roundtrip_verified": tensor == restored, "steps": 1, "quality": "not_evaluated"}


def identity(config: dict[str, Any], bound: dict[str, int],
             dependencies: list[dict[str, Any]], candidate: str) -> dict[str, Any]:
    if not dependencies:
        fail("missing_input", "Identity inspection requires a dependency.", 409)
    source = dependencies[-1]
    return {"source_digest": digest(source), **({"tensor": source["tensor"]} if "tensor" in source else {}),
            "steps": 1, "quality": "not_evaluated"}


def evaluate(config: dict[str, Any], bound: dict[str, int],
             dependencies: list[dict[str, Any]], candidate: str) -> dict[str, Any]:
    tensor = next((d["tensor"] for d in reversed(dependencies) if "tensor" in d), None)
    if tensor is None:
        return local_eval(config, bound, dependencies, candidate)
    import torch
    value = torch.tensor(tensor, dtype=torch.float64)
    mse = float((value - 2 * torch.eye(2, dtype=torch.float64)).square().mean())
    return {"metrics": [{"name": "software_tensor_mse", "version": "1", "unit": "squared_error", "value": mse}],
            "quality": "passed" if mse < 1 else "failed", "steps": 1}


def interaction(config: dict[str, Any], bound: dict[str, int], dependencies: list[dict[str, Any]],
                candidate: str, *, workflow: bool) -> dict[str, Any]:
    from ..gym_runtime import Payload, run_episodes
    from ..gym_runtime_fixtures import (
        LOOKUP_PERMISSION,
        CounterFactory,
        ScriptedActor,
        WorkflowFactory,
    )
    factory = WorkflowFactory() if workflow else CounterFactory(target=config.get("target", 2))
    actions = (Payload("lookup"), Payload("finish")) if workflow else (Payload(1),) * bound["steps"]
    actor = ScriptedActor(factory.spec.capabilities, actions, candidate_id=candidate)
    record = run_episodes(factory, actor, owner_id=candidate, seeds=tuple(config.get("seeds", [0])),
                          budgets=_budgets(bound, tools=workflow),
                          permissions=(LOOKUP_PERMISSION,) if workflow else ())
    if record.status == "error" or not record.closed:
        fail("fixture_execution_error", "Runtime fixture failed or could not close.", 409)
    return {"steps": record.steps, "episodes": len(record.episodes), "tool_calls": record.tool_calls,
            "session_closed": record.closed, "runtime_status": record.status,
            "returns": [e.episode_return for e in record.episodes],
            "metrics": [{"name": "episode_completion", "version": "1", "unit": "ratio",
                         "value": sum(e.terminated for e in record.episodes) / len(record.episodes)}],
            "quality": "passed" if all(e.terminated for e in record.episodes) else "failed"}


def weight_edit(config: dict[str, Any], bound: dict[str, int],
                dependencies: list[dict[str, Any]], candidate: str) -> dict[str, Any]:
    import torch

    from ..backends.weight_edit.capture import CovarianceStatistics
    from ..backends.weight_edit.discover import Site
    from ..backends.weight_edit.methods.experimental import ValueScaledRetainedEdit
    eye = torch.eye(2, dtype=torch.float64)
    stat = CovarianceStatistics(2, eye.sum(0), torch.zeros(2, dtype=torch.float64),
                                eye, torch.zeros((2, 2), dtype=torch.float64))
    solver = ValueScaledRetainedEdit(stat, eye, eye)
    settings = {"strength": config.get("strength", .1), "ridge_alpha": .2,
                "retention_beta": .4, "protection_epsilon": 1., "protection_zero_absolute": 10.,
                "rank_rtol": 1e-12, "rank_atol": 0., "solver_atol": 1e-10,
                "solver_max_iterations": 32}
    delta = solver.compute(Site("fixture.weight", 0, "residual_write_mlp", 2, 2, False,
                                "torch.float64"), eye, eye, eye, settings).matrix()
    return {"tensor": (eye + delta).tolist(), "delta": delta.tolist(),
            "diagnostics": solver.diagnostics, "steps": 1, "quality": "not_evaluated",
            "no_task_gain_or_preservation_claim": True}


def compare(config: dict[str, Any], bound: dict[str, int],
            dependencies: list[dict[str, Any]], candidate: str) -> dict[str, Any]:
    if len(dependencies) < 2:
        fail("missing_input", "Metric comparison requires at least two dependencies.", 409)
    baseline = dependencies[0]
    reference = {(m["name"], m["version"], m["unit"]): m["value"] for m in baseline.get("metrics", [])}
    if not reference:
        fail("missing_metrics", "Comparison dependencies require versioned measurements.", 409)
    comparisons = []
    for other in dependencies[1:]:
        values = {(m["name"], m["version"], m["unit"]): m["value"] for m in other.get("metrics", [])}
        if set(reference) != set(values):
            fail("incompatible_metrics", "Metric names, versions and units must match exactly.", 409)
        comparisons.append({"baseline_model": baseline.get("candidate_model"),
                            "candidate_model": other.get("candidate_model"),
                            "deltas": [{"name": key[0], "version": key[1], "unit": key[2],
                                        "candidate_minus_baseline": values[key] - reference[key]}
                                       for key in sorted(reference)]})
    return {"comparisons": comparisons, "steps": len(comparisons), "quality": "not_evaluated"}


HANDLERS: dict[str, Callable[[dict[str, Any], dict[str, int], list[dict[str, Any]], str], dict[str, Any]]] = {
    "software.local_eval": local_eval, "software.provider": provider,
    "software.training": training, "software.export": export, "software.identity": identity,
    "software.eval": evaluate, "software.offline_rescore": rescore,
    "software.episode": lambda c, b, d, i: interaction(c, b, d, i, workflow=False),
    "software.workflow": lambda c, b, d, i: interaction(c, b, d, i, workflow=True),
    "software.weight_edit": weight_edit,
    "software.compare": compare,
}


def execute(stage: dict[str, Any], dependencies: list[dict[str, Any]]) -> dict[str, Any]:
    started = time.monotonic()
    result = HANDLERS[stage["operation"]](stage["config"], stage["limits"], dependencies,
                                        stage.get("candidate_id") or "unassigned-fixture")
    if time.monotonic() - started > stage["limits"]["wall_seconds"]:
        fail("budget_exceeded", "Cooperative fixture wall-time bound was exceeded.", 409)
    return {**result, "execution": "completed", "model_calls": False,
            "release_authorized": False, "runtime_profile": "cpu-software-fixture-v1"}
