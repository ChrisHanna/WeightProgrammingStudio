"""Every edit method yields one plan, and a fresh process replays it to the method's own candidate."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch

from neural_foundry import export_formats, flagship, gpu, m5_surgery, method_plans, prd_adapters, pythia, repair
from neural_foundry.edit_plan import EditGraphError, EditPlan, sign_edit_graph, tensor_state_digest
from neural_foundry.gpt_neox_adapter import compile_to_edit_graph
from neural_foundry.export_formats import _apply_state_tensors_to_clone, _extract_model_state_tensors, export_huggingface_safetensors
from neural_foundry.patch_ux import compose_overlapping_patches_orthogonal, generate_ed25519_keypair
from neural_foundry.studio import WeightProgrammingStudio

GYM = pythia.ALGORITHMIC_THREE_SKILL_GYM
PROMPTS = (*(f"Task: {e}" for e, _ in GYM.base_rules), *GYM.retention_prompts)
REAL_DIR = Path(os.environ.get("NF_PYTHIA_14M_DIR", "/tmp/pythia-14m"))
ATTRACTOR = "Ġthe"


def _real_model() -> Any:
    if not (REAL_DIR / "model.safetensors").exists():
        pytest.skip(f"real Pythia-14M checkpoint not at {REAL_DIR}")
    return pythia.PythiaModel.from_pretrained_dir(REAL_DIR, model_name="EleutherAI/pythia-14m")


def _with_defect(model: Any) -> Any:
    """The same injected attractor as tests/test_attractor_repair.py."""
    layer = int(model.config["num_hidden_layers"]) - 1
    site = model.mlp_down_proj_site(layer)
    direction = repair.output_direction_for_token(model, model.tokenizer.vocab[ATTRACTOR])
    key = torch.stack([model.forward_with_mlp_key(model.tokenizer.encode(p), capture_layer=layer)[1] for p in PROMPTS]).mean(0)
    state = _extract_model_state_tensors(model)
    state[site] = state[site] + 120.0 * torch.outer(direction, key / torch.linalg.norm(key))
    return _apply_state_tensors_to_clone(model, state)


def _method_digest(base: Any, weights: Mapping[str, torch.Tensor]) -> str:
    """The candidate as the method itself produced it, computed without any plan."""
    return tensor_state_digest({**_extract_model_state_tensors(base), **weights})


_FIXTURE_REPLAY = """
import json, sys, torch
from neural_foundry import pythia
from neural_foundry.edit_plan import apply_signed_edit_graph
model = pythia.build_mini_pythia_model()
model.tensors = torch.load(sys.argv[1], weights_only=True)
envelope = json.load(open(sys.argv[2], encoding="utf-8"))
print(json.dumps({"candidate_state_digest": apply_signed_edit_graph(model, envelope, public_key=open(sys.argv[3], encoding="utf-8").read().strip()).state_digest}))
"""


def _replay_in_fresh_process(tmp_path: Path, base: Any, graph: Mapping[str, Any], *, source_dir: Path | None = None, base_on_disk: bool = False) -> dict[str, Any]:
    """Replay from base weights, the signed graph and a public key: nothing else crosses over.

    Real checkpoints go through the `python -m neural_foundry.edit_plan` command. The fixture has no
    tokenizer file, so a small script loads its weights into a newly built fixture model.
    """
    keys = generate_ed25519_keypair()
    envelope, key = tmp_path / "graph.json", tmp_path / "key.pub"
    envelope.write_text(json.dumps(sign_edit_graph(graph, signing_key=keys["private_key"])), encoding="utf-8")
    key.write_text(keys["public_key"], encoding="utf-8")
    if base_on_disk or source_dir is not None:
        checkpoint = REAL_DIR
        if not base_on_disk:
            checkpoint = tmp_path / "base"
            export_huggingface_safetensors(base, checkpoint, dtype="f64", source_checkpoint_dir=source_dir)
        command = ["-m", "neural_foundry.edit_plan", "--checkpoint", str(checkpoint), "--envelope", str(envelope), "--public-key-file", str(key)]
    else:
        weights = tmp_path / "base.pt"
        torch.save(_extract_model_state_tensors(base), weights)
        command = ["-c", _FIXTURE_REPLAY, str(weights), str(envelope), str(key)]
    result = subprocess.run([sys.executable, *command], capture_output=True, text=True, cwd=tmp_path, check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _assert_replays(tmp_path: Path, base: Any, steps: Any, weights: Mapping[str, torch.Tensor], **options: Any) -> None:
    plan = method_plans.verified_method_plan(base, steps, weights)
    expected = _method_digest(base, weights)
    assert plan.execute().state_digest == expected
    report = _replay_in_fresh_process(tmp_path, base, plan.to_graph(), **options)
    assert report["candidate_state_digest"] == expected


# ---- The bitwise check itself.


def test_verified_plan_refuses_weights_it_does_not_reproduce() -> None:
    model = pythia.build_mini_pythia_model()
    site = model.mlp_down_proj_site(0)
    weight = model.tensors[site].to(torch.float64)
    step = method_plans.add_delta_step(site, "t", torch.ones_like(weight))
    good = weight + 1.0
    method_plans.verified_method_plan(model, (step,), {site: good})
    off_by_one = good.clone()
    off_by_one[0, 0] = torch.nextafter(off_by_one[0, 0], torch.tensor(float("inf"), dtype=torch.float64))
    with pytest.raises(EditGraphError):
        method_plans.verified_method_plan(model, (step,), {site: off_by_one})
    with pytest.raises(EditGraphError):
        method_plans.verified_method_plan(model, (step,), {model.mlp_down_proj_site(1): good})


def test_a_plan_cannot_be_built_from_unverified_steps() -> None:
    model = pythia.build_mini_pythia_model()
    site = model.mlp_down_proj_site(0)
    step = method_plans.add_delta_step(site, "t", torch.ones_like(model.tensors[site]))
    with pytest.raises(TypeError):
        method_plans.edit_plan_from_steps(model, (step,))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        method_plans.VerifiedMethodSteps((step,), {}, {})


def test_verified_steps_are_refused_on_a_different_base() -> None:
    model = pythia.build_mini_pythia_model()
    site = model.mlp_down_proj_site(0)
    weight = model.tensors[site].to(torch.float64)
    verified = method_plans.verify_method_steps(model, (method_plans.add_delta_step(site, "t", torch.ones_like(weight)),), {site: weight + 1.0})
    other = _apply_state_tensors_to_clone(model, {**_extract_model_state_tensors(model), site: weight * 2.0})
    with pytest.raises(EditGraphError):
        method_plans.edit_plan_from_steps(other, verified)


def _corrupting(step_builder):  # type: ignore[no-untyped-def]
    """A method whose recorded operands differ from what it actually ran."""

    def corrupted(*args, **kwargs):  # type: ignore[no-untyped-def]
        step = step_builder(*args, **kwargs)
        return method_plans.MethodEditStep(step.tensor, step.method, step.operation_type, {k: v * 1.5 for k, v in step.operands.items()}, step.factor)

    return corrupted


def test_mismatched_steps_are_refused_by_every_method_path(monkeypatch: pytest.MonkeyPatch) -> None:
    model = pythia.build_mini_pythia_model()
    defective = _with_defect(model)
    site = model.mlp_down_proj_site(1)
    base = model.tensors[site].to(torch.float64)
    monkeypatch.setattr(repair, "low_rank_step", _corrupting(method_plans.low_rank_step))
    monkeypatch.setattr(prd_adapters, "low_rank_step", _corrupting(method_plans.low_rank_step))
    monkeypatch.setattr(prd_adapters, "add_delta_step", _corrupting(method_plans.add_delta_step))
    monkeypatch.setattr(pythia, "low_rank_step", _corrupting(method_plans.low_rank_step))
    monkeypatch.setattr(pythia, "select_gym_branch", lambda trials: (tuple(trials), trials[0]))
    with pytest.raises(EditGraphError):
        repair.repair_repetition_attractor(defective, PROMPTS)
    with pytest.raises(EditGraphError):
        prd_adapters.run_easyedit_adapter(model, algorithm="ROME", associations=[(f"Task: {GYM.base_rules[0][0]}", GYM.base_rules[0][1])])
    with pytest.raises(EditGraphError):
        prd_adapters.run_mergekit_adapter(model, algorithm="task_arithmetic", checkpoint_weights=[base + 0.01, base - 0.02])
    with pytest.raises(EditGraphError):
        pythia.run_pythia_gym_task_learning_study(model, gym=GYM, target_strength=80.0)
    # The agent sees a refused proposal as an error, and nothing can be submitted.
    report = _scripted_session(model, AGENT_PROPOSALS["dampen"], expect_candidate=False)
    assert report.submitted_edit_steps == ()
    assert "EditGraphError" in report.transcript[0].tool_observation["error"]
    with pytest.raises(EditGraphError):
        prd_adapters.FrontierAgentProgrammer("claude-4-5-sonnet").run_reasoning_loop_repair_experiment(model, enable_feedback_tools=False)


def test_two_steps_keep_their_own_operands() -> None:
    model = pythia.build_mini_pythia_model()
    a, b = model.mlp_down_proj_site(0), model.mlp_down_proj_site(1)
    wa, wb = model.tensors[a].to(torch.float64), model.tensors[b].to(torch.float64)
    steps = (
        method_plans.add_delta_step(a, "t", torch.ones_like(wa)),
        method_plans.add_delta_step(b, "t", 2.0 * torch.ones_like(wb)),
    )
    edited = method_plans.verified_method_plan(model, steps, {a: wa + 1.0, b: wb + 2.0}).execute().edited_tensors
    assert torch.equal(edited[a], wa + 1.0) and torch.equal(edited[b], wb + 2.0)


# ---- Repair.


def test_attractor_repair_replays_on_fixture(tmp_path: Path) -> None:
    defective = _with_defect(pythia.build_mini_pythia_model())
    found = repair.repair_repetition_attractor(defective, PROMPTS)
    assert found.selected is not None and found.edited_weight is not None
    _assert_replays(tmp_path, defective, found.edit_steps, {found.selected.site: found.edited_weight})


def test_healthy_repair_has_no_steps() -> None:
    model = pythia.build_mini_pythia_model()
    found = repair.repair_repetition_attractor(model, PROMPTS)
    assert (found.selected is None) == (found.edit_steps == ())


def test_studio_looping_repair_returns_its_plan(tmp_path: Path) -> None:
    defective = _with_defect(pythia.build_mini_pythia_model())
    result = WeightProgrammingStudio().repair(defective, failure="looping")
    plan = result["edit_plan"]
    assert plan.execute().model is result["edited_model"]
    expected = tensor_state_digest(result["edited_model"].tensors)
    assert _replay_in_fresh_process(tmp_path, defective, plan.to_graph())["candidate_state_digest"] == expected


def test_studio_repair_with_no_attractor_has_no_plan() -> None:
    result = WeightProgrammingStudio().repair(pythia.build_mini_pythia_model(), failure="looping")
    assert (result["open_patch"] is None) == (result["edit_plan"] is None)


def test_studio_association_repair_returns_its_plan(tmp_path: Path) -> None:
    model = pythia.build_mini_pythia_model()
    result = WeightProgrammingStudio().repair(model, failure="parity")
    expected = tensor_state_digest(result["edited_model"].tensors)
    assert _replay_in_fresh_process(tmp_path, model, result["edit_plan"].to_graph())["candidate_state_digest"] == expected


def test_real_pythia_studio_repair_replays(tmp_path: Path) -> None:
    defective = _with_defect(_real_model())
    result = WeightProgrammingStudio().repair(defective, failure="looping")
    assert result["edit_plan"] is not None, result["verification"]
    expected = tensor_state_digest(result["edited_model"].tensors)
    report = _replay_in_fresh_process(tmp_path, defective, result["edit_plan"].to_graph(), source_dir=REAL_DIR)
    assert report["candidate_state_digest"] == expected


# ---- Association edits.


def _weights_by_site(model: Any, receipt: Any) -> dict[str, torch.Tensor]:
    return {model.mlp_down_proj_site(layer): w for layer, w in receipt.edited_weights_by_layer.items()}


@pytest.mark.parametrize(
    ("algorithm", "count", "layers"),
    [("ROME", 1, None), ("MEMIT", 2, None), ("MEMIT", 2, (0, 1)), ("AlphaEdit", 2, (0, 1))],
)
def test_association_edit_replays_on_fixture(tmp_path: Path, algorithm: Any, count: int, layers: Any) -> None:
    model = pythia.build_mini_pythia_model()
    pairs = [(f"Task: {e}", t) for e, t in GYM.base_rules[:count]]
    receipt = prd_adapters.run_easyedit_adapter(model, algorithm=algorithm, associations=pairs, target_layers=layers)
    _assert_replays(tmp_path, model, receipt.edit_steps, _weights_by_site(model, receipt))


def test_duplicate_target_layers_replay_last_edit_wins(tmp_path: Path) -> None:
    model = pythia.build_mini_pythia_model()
    pairs = [(f"Task: {e}", t) for e, t in GYM.base_rules[:2]]
    receipt = prd_adapters.run_easyedit_adapter(model, algorithm="MEMIT", associations=pairs, target_layers=(1, 1))
    assert len(receipt.edit_steps) == 1
    assert len(receipt.layer_plan_digests) == 2
    _assert_replays(tmp_path, model, receipt.edit_steps, _weights_by_site(model, receipt))


@pytest.mark.parametrize(("algorithm", "count"), [("ROME", 1), ("MEMIT", 2)])
def test_real_pythia_association_edit_replays(tmp_path: Path, algorithm: Any, count: int) -> None:
    model = _real_model()
    pairs = [(f"Task: {e}", t) for e, t in GYM.base_rules[:count]]
    receipt = prd_adapters.run_easyedit_adapter(model, algorithm=algorithm, associations=pairs)
    _assert_replays(tmp_path, model, receipt.edit_steps, _weights_by_site(model, receipt), base_on_disk=True)


@pytest.mark.parametrize(("algorithm", "count"), [("MEMIT", 2), ("AlphaEdit", 2)])
def test_real_pythia_multi_layer_association_edit_replays(tmp_path: Path, algorithm: Any, count: int) -> None:
    model = _real_model()
    pairs = [(f"Task: {e}", t) for e, t in GYM.base_rules[:count]]
    receipt = prd_adapters.run_easyedit_adapter(model, algorithm=algorithm, associations=pairs, target_layers=(4, 5))
    assert len(receipt.edit_steps) == 2
    _assert_replays(tmp_path, model, receipt.edit_steps, _weights_by_site(model, receipt), base_on_disk=True)


def test_real_pythia_association_repair_replays(tmp_path: Path) -> None:
    model = _real_model()
    result = WeightProgrammingStudio().repair(model, failure="parity")
    expected = tensor_state_digest(result["edited_model"].tensors)
    report = _replay_in_fresh_process(tmp_path, model, result["edit_plan"].to_graph(), base_on_disk=True)
    assert report["candidate_state_digest"] == expected


# ---- Agent loop, driven by a scripted transcript. No live LLM is called.


AGENT_PROPOSALS: dict[str, dict[str, Any]] = {
    "dampen": {"operation": "dampen_singular_components", "components": [0, 1], "factor": 0.5},
    "activation": {"operation": "project_out_activation_direction", "positive_prompts": ["Rule: hello"], "negative_prompts": ["Rule: the"], "factor": 0.5},
    "output": {"operation": "project_out_output_direction", "token": ATTRACTOR, "factor": 0.5},
    "associations": {"operation": "edit_associations", "associations": [{"prompt": f"Task: {GYM.base_rules[0][0]}", "target": GYM.base_rules[0][1]}]},
}


def _scripted_session(model: Any, proposal: Mapping[str, Any], *, expect_candidate: bool = True) -> Any:
    agent: Any = None

    def transport(messages: list[dict[str, Any]], tools: object) -> dict[str, Any]:
        turn = sum(1 for m in messages if m["role"] == "assistant")
        if turn == 0:
            return {"tool_call": {"name": "propose_weight_program", "arguments": dict(proposal)}}
        observation = agent._transcript[-1].tool_observation
        if not expect_candidate:
            return {}
        assert "candidate_id" in observation, observation
        return {"tool_call": {"name": "submit_candidate", "arguments": {"candidate_id": observation["candidate_id"]}}}

    agent = prd_adapters.FrontierAgentProgrammer("claude-4-5-sonnet", llm_transport=transport)
    return agent.run_autonomous_tool_loop(model, max_turns=4)


@pytest.mark.parametrize("label", sorted(AGENT_PROPOSALS))
def test_scripted_agent_submission_replays(tmp_path: Path, label: str) -> None:
    model = pythia.build_mini_pythia_model()
    report = _scripted_session(model, AGENT_PROPOSALS[label])
    assert report.session_kind == "agent_driven"
    weights = {model.mlp_down_proj_site(layer): w for layer, w in report.submitted_weights_by_layer.items()}
    _assert_replays(tmp_path, model, report.submitted_edit_steps, weights)


@pytest.mark.parametrize("label", sorted(AGENT_PROPOSALS))
def test_real_pythia_scripted_agent_submission_replays(tmp_path: Path, label: str) -> None:
    model = _real_model()
    report = _scripted_session(model, AGENT_PROPOSALS[label])
    weights = {model.mlp_down_proj_site(layer): w for layer, w in report.submitted_weights_by_layer.items()}
    _assert_replays(tmp_path, model, report.submitted_edit_steps, weights, base_on_disk=True)


def test_scripted_agent_without_submission_has_no_steps() -> None:
    report = prd_adapters.FrontierAgentProgrammer("claude-4-5-sonnet", llm_transport=lambda m, t: {}).run_autonomous_tool_loop(
        pythia.build_mini_pythia_model(), max_turns=2
    )
    assert report.submitted_edit_steps == ()


# ---- mergekit.


@pytest.mark.parametrize("algorithm", ["task_arithmetic", "ties", "dare"])
def test_mergekit_replays_on_fixture(tmp_path: Path, algorithm: Any) -> None:
    model = pythia.build_mini_pythia_model()
    site = model.mlp_down_proj_site(int(model.config["num_hidden_layers"]) - 1)
    base = model.tensors[site].to(torch.float64)
    generator = torch.Generator().manual_seed(3)
    checkpoints = [base + 0.01 * torch.randn(base.shape, dtype=torch.float64, generator=generator) for _ in range(2)]
    receipt = prd_adapters.run_mergekit_adapter(model, algorithm=algorithm, checkpoint_weights=checkpoints)
    _assert_replays(tmp_path, model, receipt.edit_steps, {receipt.target_site: receipt.merged_weight})


@pytest.mark.parametrize("algorithm", ["task_arithmetic", "ties", "dare"])
def test_real_pythia_mergekit_replays(tmp_path: Path, algorithm: Any) -> None:
    model = _real_model()
    site = model.mlp_down_proj_site(int(model.config["num_hidden_layers"]) - 1)
    base = model.tensors[site].to(torch.float64)
    generator = torch.Generator().manual_seed(3)
    checkpoints = [base + 0.01 * torch.randn(base.shape, dtype=torch.float64, generator=generator) for _ in range(2)]
    receipt = prd_adapters.run_mergekit_adapter(model, algorithm=algorithm, checkpoint_weights=checkpoints)
    _assert_replays(tmp_path, model, receipt.edit_steps, {receipt.target_site: receipt.merged_weight}, base_on_disk=True)


# ---- Gym study, the two reasoning-loop arms, patch composition.


def _replay_emitted(tmp_path: Path, model: Any, study: Any, kind: str, *, prefix: str = "branch") -> int:
    """Replay frozen candidates from their operands in a fresh process, selected or not. Returns how many."""
    replayed = 0
    for candidate in study.gym_proposals:
        if candidate.branch_id.startswith(prefix):
            room = tmp_path / candidate.branch_id
            room.mkdir()
            report = _replay_in_fresh_process(room, model, compile_to_edit_graph(model, candidate.proposal), **_on_disk(kind))
            assert report["candidate_state_digest"] == candidate.proposal.expected_candidate_digest
            replayed += 1
    return replayed


def _load(kind: str) -> Any:
    return _real_model() if kind == "real" else pythia.build_mini_pythia_model()


def _on_disk(kind: str) -> dict[str, bool]:
    return {"base_on_disk": kind == "real"}


@pytest.mark.parametrize("kind", ["fixture", "real"])
def test_gym_study_replays(tmp_path: Path, kind: str) -> None:
    model = _load(kind)
    study = pythia.run_pythia_gym_task_learning_study(model, gym=GYM, target_strength=80.0)
    # Selection doesn't decide what can replay: every emitted candidate must, from its operands.
    assert _replay_emitted(tmp_path, model, study, kind) == 3
    if kind == "real":
        assert study.selected_branch_id is None and study.edit_steps == () and study.edited_weight is None
        assert [c.selected for c in study.gym_proposals] == [False, False, False]
        return
    if study.selected_branch_id is None:
        assert study.edit_steps == () and study.edited_weight is None
        assert study.holdout_after == ()
        return
    site = model.mlp_down_proj_site(int(model.config["num_hidden_layers"]) - 1)
    _assert_replays(tmp_path, model, study.edit_steps, {site: study.edited_weight}, **_on_disk(kind))


@pytest.mark.parametrize("kind", ["fixture", "real"])
@pytest.mark.parametrize("feedback", [False, True])
def test_reasoning_loop_arm_replays(tmp_path: Path, kind: str, feedback: bool) -> None:
    model = _load(kind)
    report = prd_adapters.FrontierAgentProgrammer("claude-4-5-sonnet").run_reasoning_loop_repair_experiment(model, enable_feedback_tools=feedback)
    if kind == "real" and feedback:
        # This arm submits the Gym's selection, and no branch qualifies on the real checkpoint.
        assert report.submitted_edit_steps == () and report.submitted_weights_by_layer is None and report.held_out_accuracy_after is None
        # The arm's rejected candidate must still replay.
        study = pythia.run_pythia_gym_task_learning_study(model, gym=GYM, target_strength=80.0)
        assert _replay_emitted(tmp_path, model, study, kind, prefix="branch_3") == 1
        return
    if report.submitted_weights_by_layer is None:
        assert report.submitted_edit_steps == ()
        study = pythia.run_pythia_gym_task_learning_study(model, gym=GYM, target_strength=80.0)
        assert _replay_emitted(tmp_path, model, study, kind, prefix="branch_3") == 1
        return
    weights = {model.mlp_down_proj_site(layer): w for layer, w in report.submitted_weights_by_layer.items()}
    _assert_replays(tmp_path, model, report.submitted_edit_steps, weights, **_on_disk(kind))


@pytest.mark.parametrize("kind", ["fixture", "real"])
def test_patch_composition_replays(tmp_path: Path, kind: str) -> None:
    model = _load(kind)
    site = model.mlp_down_proj_site(1)
    base = model.tensors[site].to(torch.float64)
    generator = torch.Generator().manual_seed(5)
    edited = [base + 0.01 * torch.randn(base.shape, dtype=torch.float64, generator=generator) for _ in range(2)]
    receipt = compose_overlapping_patches_orthogonal(base, edited, target_site=site)
    _assert_replays(tmp_path, model, receipt.edit_steps, {site: receipt.composed_weight}, **_on_disk(kind))


# ---- Device and precision.


def _devices() -> list[str]:
    return ["cpu", *(["cuda:0"] if gpu.probe_gpu_hardware().cuda_executable else [])]


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_f64_device_add_delta_replays_on_rust_cpu_and_says_so(tmp_path: Path, device: str) -> None:
    if device not in _devices():
        pytest.skip("CUDA is not available in this environment")
    model = pythia.build_mini_pythia_model()
    site = model.mlp_down_proj_site(1)
    base = model.tensors[site].to(torch.float64)
    delta = 0.01 * torch.randn(base.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
    receipt = gpu.execute_transactional_gpu_edit_and_eval(
        base.to(device), tensor_name=site, operation="add_delta", requested_device=device, requested_dtype="f64", accumulation_dtype="f64", delta=delta.to(device)
    )
    assert receipt.execution_device == device
    assert (receipt.replay_precision, receipt.replay_device, receipt.replay_refused_reason) == ("f64", "cpu_rust", "")
    _assert_replays(tmp_path, model, receipt.edit_steps, {site: receipt.output_tensor})


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_f64_device_low_rank_replays_only_if_bitwise_equal(tmp_path: Path, device: str) -> None:
    if device not in _devices():
        pytest.skip("CUDA is not available in this environment")
    model = pythia.build_mini_pythia_model()
    site = model.mlp_down_proj_site(1)
    base = model.tensors[site].to(torch.float64)
    generator = torch.Generator().manual_seed(5)
    left = torch.randn((base.shape[0], 2), dtype=torch.float64, generator=generator)
    right = torch.randn((2, base.shape[1]), dtype=torch.float64, generator=generator)
    receipt = gpu.execute_transactional_gpu_edit_and_eval(
        base.to(device), tensor_name=site, operation="low_rank_update", requested_device=device, requested_dtype="f64", accumulation_dtype="f64", left_basis=left.to(device), right_basis=right.to(device)
    )
    print(f"{device}: steps={len(receipt.edit_steps)} reason={receipt.replay_refused_reason!r}")
    if receipt.edit_steps:
        _assert_replays(tmp_path, model, receipt.edit_steps, {site: receipt.output_tensor})
    else:
        assert "bitwise" in receipt.replay_refused_reason


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("dtype", ["f32", "bf16"])
def test_narrow_device_edit_gets_no_plan_and_says_why(device: str, dtype: Any) -> None:
    if device not in _devices():
        pytest.skip("CUDA is not available in this environment")
    base = torch.eye(4, dtype={"f32": torch.float32, "bf16": torch.bfloat16}[dtype], device=device)
    receipt = gpu.execute_transactional_gpu_edit_and_eval(
        base, operation="add_delta", requested_device=device, requested_dtype=dtype, accumulation_dtype="f32", delta=0.25 * torch.ones(4, 4, dtype=torch.float32, device=device)
    )
    assert receipt.edit_steps == () and "only f64" in receipt.replay_refused_reason


def test_quantized_edit_cannot_be_a_plan() -> None:
    from neural_foundry.edit_plan import _SCALAR_ARGUMENTS

    weight = (torch.arange(64, dtype=torch.float64).reshape(8, 8) + 0.123) / 37.0
    q8, shape = export_formats.quantize_matrix_q8_0(weight.tolist())
    delta = torch.full((8, 8), 0.0173, dtype=torch.float64)
    receipt = export_formats.execute_quantized_q8_matrix_edit(q8, shape, operation_type="add_delta", delta_matrix=delta)
    # The graph has no precision or requantize operand, and an f64 replay lands on different weights.
    assert "dtype" not in _SCALAR_ARGUMENTS and "accumulation_dtype" not in _SCALAR_ARGUMENTS
    f64_result = torch.tensor(export_formats.dequantize_matrix_q8_0(q8, shape), dtype=torch.float64) + delta
    assert not torch.equal(torch.tensor(receipt.dequantized_matrix, dtype=torch.float64), f64_result)


# ---- Flagship bundle.


_FLAGSHIP_REPLAY = """
import json, sys
from neural_foundry import pythia
from neural_foundry.edit_plan import tensor_state_digest
from neural_foundry.flagship import replay_flagship_edit_graphs
candidate = replay_flagship_edit_graphs(pythia.build_mini_pythia_model(), sys.argv[1], public_key=sys.argv[2])
print(candidate.state_digest)
"""


def test_flagship_bundle_replays_from_its_artifacts(tmp_path: Path) -> None:
    model = pythia.build_mini_pythia_model()
    bundle = tmp_path / "bundle"
    report = flagship.run_flagship_loop_repair_study_and_publish_bundle(model, output_dir=bundle)
    public_key = json.loads((bundle / "verification_receipt.json").read_bytes())["open_patch_public_key"]
    published = {Path(f).name for f in report.published_files}
    assert flagship.FLAGSHIP_DEFECT_GRAPH in published and flagship.FLAGSHIP_REPAIR_GRAPH in published
    # The transcript still serializes, and carries no tensors.
    assert json.loads((bundle / "agent_transcript.json").read_bytes())["submitted_edit_steps"] == []
    repaired = (bundle / flagship.FLAGSHIP_REPAIR_GRAPH).read_bytes()
    expected = json.loads(repaired)["patch"]["candidate_state_digest"]
    result = subprocess.run([sys.executable, "-c", _FLAGSHIP_REPLAY, str(bundle), public_key], capture_output=True, text=True, cwd=tmp_path, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


# ---- Methods a plan can't honestly express.


def test_mlp_width_surgery_cannot_be_a_plan() -> None:
    model = pythia.build_mini_pythia_model()
    site = model.mlp_down_proj_site(0)
    down = model.tensors[site].to(torch.float64)
    inner = down.T
    result = m5_surgery.execute_mlp_width_surgery(gate_proj=inner, up_proj=inner, down_proj=down, keep_indices=[0, 1])
    assert result.down_proj.shape[1] != down.shape[1]
    with pytest.raises(Exception, match=r"(?i)shape|dimension|mismatch"):
        EditPlan(model).replace_slice(site, result.down_proj).preview()
    # A region write keeps the tensor's shape, so it can never reach the surgery's narrower model.
    written = EditPlan(model).replace_slice(site, result.down_proj, region=_region_of(result.down_proj)).preview()
    assert written.edited_tensors[site].shape == down.shape


def _region_of(matrix: torch.Tensor) -> Any:
    from neural_foundry.edit_plan import Region

    return Region(rows=(0, int(matrix.shape[0])), cols=(0, int(matrix.shape[1])))


def test_attention_head_surgery_cannot_be_a_plan() -> None:
    model = pythia.build_mini_pythia_model()
    site = model.mlp_down_proj_site(0)
    head_dim = 2
    matrix = torch.ones((8, 8), dtype=torch.float64)
    # Four Q heads share two KV heads, so Q heads 0 and 1 are the group that reads KV head 0.
    kv_matrix = torch.ones((4, 8), dtype=torch.float64)
    result = m5_surgery.execute_attention_head_surgery(
        q_proj=matrix, k_proj=kv_matrix, v_proj=kv_matrix, o_proj=matrix, keep_q_heads=[0, 1], keep_kv_heads=[0], head_dim=head_dim
    )
    # Pruning changes tensor shapes and the head-count config. A graph has neither a shape nor a config operand.
    assert result.q_proj.shape != matrix.shape and result.k_proj.shape != matrix.shape
    assert result.num_attention_heads == 2
    with pytest.raises(Exception, match=r"(?i)shape|dimension|mismatch"):
        EditPlan(model).replace_slice(site, result.q_proj).preview()


def test_layer_reorganization_cannot_be_a_plan() -> None:
    model = pythia.build_mini_pythia_model()
    layers = [model.tensors[model.mlp_down_proj_site(i)].to(torch.float64) for i in range(int(model.config["num_hidden_layers"]))]
    result = m5_surgery.execute_layer_reorganization(layers, new_layer_order=[0])
    assert result.num_hidden_layers == 1 < len(layers)
    name = next(iter(result.layer_weights))
    with pytest.raises(ValueError, match="Unknown tensor"):
        EditPlan(model).replace_slice(name, result.layer_weights[name])


def test_structural_constructor_edits_no_weights() -> None:
    weight = torch.eye(3, dtype=torch.float64)
    receipt = m5_surgery.apply_registered_structural_constructor(weight, weight, constructor_id="split_linear_channel_exact.v1")
    assert not any(isinstance(v, torch.Tensor) for v in vars(receipt).values())


def test_grouped_quantization_is_a_representation_not_a_model_edit() -> None:
    weight = torch.tensor([[0.123, 0.456, 0.789, 0.321]], dtype=torch.float64)
    artifact = m5_surgery.quantize_weight_grouped(weight, group_size=4)
    assert artifact.qweight.dtype == torch.int8
    assert artifact.max_roundtrip_abs_error > 0.0


def test_status_table_lists_every_method() -> None:
    table = (Path(__file__).parent.parent / "docs" / "method_replay_status.md").read_text(encoding="utf-8")
    for name in (
        "attractor_projection", "association_edit", "ROME", "MEMIT", "AlphaEdit", "task_arithmetic", "ties", "dare",
        "run_autonomous_tool_loop", "execute_mlp_width_surgery", "execute_attention_head_surgery",
        "execute_layer_reorganization", "apply_registered_structural_constructor", "quantize_weight_grouped",
        "execute_quantized_q8_matrix_edit", "execute_transactional_gpu_edit_and_eval", "compose_overlapping_patches_orthogonal",
    ):
        assert name in table, name


