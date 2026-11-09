from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch

from neural_foundry import gym_adapters
from neural_foundry.backends.automodel.adapter import _sft_rows
from neural_foundry.execution_runner import run_local_worker
from neural_foundry.gym_job import GymJobError
from neural_foundry.gym_service import CandidateIdentity
from neural_foundry.reference_json_extraction import generate_scenarios, reference_task


def test_automodel_accepts_completed_json_demonstrations() -> None:
    task = reference_task(rows_per_template=1)
    row = generate_scenarios(split="train", seed=17, parameters={"rows_per_template": 1})[0]
    interaction = task.project(row["input"], row["reference"])
    assert _sft_rows([{"interaction": interaction}]) == [{"messages": interaction["messages"]}]
    with pytest.raises(GymJobError, match="completed"):
        _sft_rows([{"interaction": task.project(row["input"], None)}])
    with pytest.raises(GymJobError, match="chat or JSON"):
        _sft_rows([{"interaction": {"format": "tool_messages_v1", "messages": interaction["messages"]}}])


def test_transformers_adapter_parses_json_projection(tmp_path, monkeypatch) -> None:
    digest = "sha256:" + "a" * 64
    monkeypatch.setattr(gym_adapters, "_state_digest", lambda model: digest)

    class Inputs(dict):
        def to(self, device):
            return self

    class Tokenizer:
        eos_token_id = 0
        chat_template = "template"
        special_tokens_map = {}

        def get_vocab(self):
            return {"token": 1}

        def encode(self, text, **kwargs):
            return [1]

        def apply_chat_template(self, messages, **kwargs):
            return "prompt"

        def __call__(self, prompt, **kwargs):
            return Inputs(input_ids=torch.tensor([[1]]))

        def decode(self, tokens, **kwargs):
            return '{"customer":"Aster","product":"cables","quantity":2}'

    class Model:
        config = SimpleNamespace(model_type="qwen3")

        def generate(self, **kwargs):
            return torch.tensor([[1, 2]])

    checkpoint = SimpleNamespace(model_id="test/model", revision="a" * 40, path=tmp_path,
                                 file_digests={"config.json": digest, "tokenizer.json": digest})
    candidate = CandidateIdentity("base", "unchanged_base", digest, digest, digest)
    adapter = gym_adapters._model_adapter(Model(), Tokenizer(), checkpoint, candidate,
                                          {"device": "cpu", "precision": "float32", "context_tokens": 32,
                                           "max_new_tokens": 8, "seed": 1})
    row = generate_scenarios(split="train", seed=1, parameters={"rows_per_template": 1})[0]
    result = adapter.infer(reference_task().project(row["input"], None))
    assert "json_extraction_v1" in adapter.interaction_formats
    assert result.response == {"customer": "Aster", "product": "cables", "quantity": 2}
    assert result.interaction["interaction_format"] == "json_extraction_v1"


@pytest.mark.parametrize("families,accepted", [(('*',), True), (('other_family',), False)])
def test_lifecycle_handles_generic_family_preflight_without_serving(tmp_path, families, accepted) -> None:
    from dataclasses import replace
    from neural_foundry.gym_artifacts import CandidateRef
    from neural_foundry.gym_lifecycle import AdapterIdentity, EvaluatorAdapter, GymLifecycle, ModelAdapter, TrainerAdapter
    from neural_foundry.gym_service import GymStudioService

    task = reference_task(rows_per_template=1)
    digest = task.identity.implementation_digest

    def reached_binding(service):
        raise RuntimeError("preflight passed before any model execution")

    task = replace(task, bind=reached_binding)
    candidate = CandidateIdentity("base", "unchanged_base", digest, digest, digest)
    reference = CandidateRef("test/model", "full", digest, digest, digest, str(tmp_path), "base", "unchanged_base", "new_family")
    model = ModelAdapter(AdapterIdentity("test_model", "1", digest), candidate, "new_family",
                         ("json_extraction_v1",), {}, lambda value: None, lambda: {},
                         artifact_kinds=("full", "weight_edit"), reference=reference, load_candidate=lambda value: None)
    trainer = TrainerAdapter(AdapterIdentity("test_trainer", "1", digest), ("json_extraction_v1",), families,
                             ("weight_edit",), lambda *args: None, artifact_kinds={"weight_edit": "weight_edit"})
    evaluator = EvaluatorAdapter(AdapterIdentity("test_evaluator", "1", digest), lambda *args: {})
    lifecycle = GymLifecycle(GymStudioService(state_dir=tmp_path / "state"))
    for adapter in (task, model, trainer, evaluator):
        lifecycle.register(adapter)
    request = {"run_id": "preflight", "task": task.identity.adapter_id, "model": "test_model",
               "trainer": "test_trainer", "evaluator": "test_evaluator", "recipe": {"method": "weight_edit", "seed": 17}}
    if accepted:
        with pytest.raises(RuntimeError, match="preflight passed"):
            lifecycle.run(request)
    else:
        with pytest.raises(GymJobError, match="unsupported task/model/trainer"):
            lifecycle.run(request)


def test_real_automodel_json_fixture_lifecycle(tmp_path) -> None:
    gpu_python = os.environ.get("NF_GPU_PYTHON")
    automodel_python = os.environ.get("NF_AUTOMODEL_PYTHON")
    if not gpu_python or not automodel_python or os.environ.get("NF_AUTOMODEL_LIVE") != "1":
        pytest.skip("set NF_GPU_PYTHON, NF_AUTOMODEL_PYTHON and NF_AUTOMODEL_LIVE=1")
    code = '''
import json, sys
from pathlib import Path
from neural_foundry.backends.automodel.config import AutoModelBackendConfig
from neural_foundry.gym_adapters import transformers_model
from neural_foundry.gym_lifecycle import AdapterIdentity, EvaluatorAdapter, GymLifecycle, public_panel_evaluator
from neural_foundry.gym_service import GymStudioService
from neural_foundry.reference_json_extraction import generate_scenarios, reference_task
task = reference_task(rows_per_template=1)
service = GymStudioService(state_dir=Path(sys.argv[2]) / "state")
base = transformers_model("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca",
                          device="cuda:0", precision="bfloat16", context_tokens=512, max_new_tokens=80, seed=17)
evaluator = EvaluatorAdapter(AdapterIdentity("public", "1", task.identity.implementation_digest), public_panel_evaluator)
lifecycle = GymLifecycle(service)
for adapter in (task, base, evaluator): lifecycle.register(adapter)
lifecycle.register_backend("nemo_automodel.sft@1", config=AutoModelBackendConfig(sys.argv[1], 300))
rows = generate_scenarios(split="train", seed=17, parameters={"rows_per_template":1})
request = {"run_id":"json-fixture", "task":"reference_invoice_extraction@1", "model":base.identity.adapter_id+"@1",
           "trainer":"nemo_automodel.sft@1", "evaluator":"public@1",
           "recipe":{"method":"sft", "seed":17, "steps":1, "max_sequence_tokens":256,
                     "tokens_per_batch":256, "max_dataset_tokens":512, "learning_rate":0.00001},
           "approved_responses":{row["scenario_id"]:row["reference"] for row in rows}}
result = lifecycle.run(request)
assert result["training_request"]["interaction_format"] == "json_extraction_v1"
assert result["candidate_record"]["candidate"]["method_id"] == "sft"
assert result["approval"]["release_authorized"] is False
print(json.dumps({"candidate":result["candidate_record"]["candidate"], "scorecard":result["scorecard"]}))
'''
    result = run_local_worker([gpu_python, "-c", code, automodel_python, str(tmp_path)], cwd=tmp_path, timeout_seconds=480)
    assert result.returncode == 0 and not result.timed_out, (result.stdout + result.stderr).decode("utf-8", "replace")[-8000:]
    assert json.loads(result.stdout.decode().splitlines()[-1])["candidate"]["method_id"] == "sft"