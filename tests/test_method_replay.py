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

from neural_foundry import m5_surgery, method_plans, prd_adapters, pythia, repair
from neural_foundry.edit_plan import EditGraphError, EditPlan, sign_edit_graph, tensor_state_digest
from neural_foundry.export_formats import _apply_state_tensors_to_clone, _extract_model_state_tensors, export_huggingface_safetensors
from neural_foundry.patch_ux import generate_ed25519_keypair
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
    good = method_plans.edit_plan_from_steps(model, (step,)).preview().edited_tensors[site]
    method_plans.verified_method_plan(model, (step,), {site: good})
    off_by_one = good.clone()
    off_by_one[0, 0] = torch.nextafter(off_by_one[0, 0], torch.tensor(float("inf"), dtype=torch.float64))
    with pytest.raises(EditGraphError):
        method_plans.verified_method_plan(model, (step,), {site: off_by_one})
    with pytest.raises(EditGraphError):
        method_plans.verified_method_plan(model, (step,), {model.mlp_down_proj_site(1): good})


def test_two_steps_keep_their_own_operands() -> None:
    model = pythia.build_mini_pythia_model()
    a, b = model.mlp_down_proj_site(0), model.mlp_down_proj_site(1)
    steps = (
        method_plans.add_delta_step(a, "t", torch.ones_like(model.tensors[a])),
        method_plans.add_delta_step(b, "t", 2.0 * torch.ones_like(model.tensors[b])),
    )
    edited = method_plans.edit_plan_from_steps(model, steps).preview().edited_tensors
    assert float((edited[a] - model.tensors[a]).max()) == pytest.approx(1.0)
    assert float((edited[b] - model.tensors[b]).max()) == pytest.approx(2.0)


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


# ---- Agent loop, driven by a scripted transcript. No live LLM is called.


AGENT_PROPOSALS: dict[str, dict[str, Any]] = {
    "dampen": {"operation": "dampen_singular_components", "components": [0, 1], "factor": 0.5},
    "activation": {"operation": "project_out_activation_direction", "positive_prompts": ["Rule: hello"], "negative_prompts": ["Rule: the"], "factor": 0.5},
    "output": {"operation": "project_out_output_direction", "token": ATTRACTOR, "factor": 0.5},
    "associations": {"operation": "edit_associations", "associations": [{"prompt": f"Task: {GYM.base_rules[0][0]}", "target": GYM.base_rules[0][1]}]},
}


def _scripted_session(model: Any, proposal: Mapping[str, Any]) -> Any:
    agent: Any = None

    def transport(messages: list[dict[str, Any]], tools: object) -> dict[str, Any]:
        turn = sum(1 for m in messages if m["role"] == "assistant")
        if turn == 0:
            return {"tool_call": {"name": "propose_weight_program", "arguments": dict(proposal)}}
        observation = agent._transcript[-1].tool_observation
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


def test_scripted_agent_without_submission_has_no_steps() -> None:
    report = prd_adapters.FrontierAgentProgrammer("claude-4-5-sonnet", llm_transport=lambda m, t: {}).run_autonomous_tool_loop(
        pythia.build_mini_pythia_model(), max_turns=2
    )
    assert report.submitted_edit_steps == ()
