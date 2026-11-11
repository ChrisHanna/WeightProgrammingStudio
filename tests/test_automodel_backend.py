"""AutoModel configuration and opt-in real lifecycle contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from neural_foundry.backends.automodel.config import AutoModelBackendConfig, AutoModelSFTRecipe
from neural_foundry.execution_runner import run_local_worker
from neural_foundry.gym_job import GymJobError


def test_automodel_recipe_and_runtime_config_are_frozen(tmp_path: Path) -> None:
    config = AutoModelBackendConfig(str(tmp_path / "python"))
    recipe = {"method": "sft", "seed": 17, "steps": 1, "max_sequence_tokens": 32,
              "tokens_per_batch": 64, "max_dataset_tokens": 128, "learning_rate": 0.00001}
    assert config.document()["versions"]["nemo-automodel"] == "0.7.0"
    assert AutoModelSFTRecipe.from_document(recipe).document() == recipe
    assert AutoModelSFTRecipe.from_document(recipe).digest != AutoModelSFTRecipe.from_document({**recipe, "seed": 18}).digest
    for method in ("lora", "qlora"):
        candidate = {**recipe, "method": method, "rank": 2, "target_modules": ["q_proj", "v_proj"]}
        assert AutoModelSFTRecipe.from_document(candidate).document() == candidate
    for changed in ({"method": "qlora"}, {"steps": True}, {"learning_rate": float("nan")},
                    {"max_sequence_tokens": 65}, {"extra": 1}):
        with pytest.raises(GymJobError):
            AutoModelSFTRecipe.from_document({**recipe, **changed})


def test_automodel_checkpoint_selection_uses_final_committed_step(tmp_path: Path) -> None:
    from neural_foundry.backends.automodel.adapter import _is_final_checkpoint_output

    intermediate = tmp_path / "epoch_0_step_319" / "model"
    final = tmp_path / "epoch_1_step_383" / "model"
    consolidated_final = final / "consolidated"

    assert not _is_final_checkpoint_output(intermediate, method="lora", final_step=383)
    assert _is_final_checkpoint_output(final, method="lora", final_step=383)
    assert _is_final_checkpoint_output(consolidated_final, method="sft", final_step=383)


@pytest.mark.parametrize("method", ["sft", "lora", "qlora"])
def test_automodel_real_lifecycle_publication_and_retry(tmp_path: Path, method: str) -> None:
    gpu_python = os.environ.get("NF_GPU_PYTHON")
    automodel_python = os.environ.get("NF_AUTOMODEL_PYTHON")
    if not gpu_python or not automodel_python or os.environ.get("NF_AUTOMODEL_LIVE") != "1":
        pytest.skip("set NF_GPU_PYTHON, NF_AUTOMODEL_PYTHON and NF_AUTOMODEL_LIVE=1")
    code = """
import hashlib, json, sys
from pathlib import Path
from neural_foundry.backends.automodel.config import AutoModelBackendConfig
from neural_foundry.gym_adapters import transformers_model
from neural_foundry.gym_job import AcceptanceGate, CandidateMethod, ComponentReference, GymJobSpec, ScenarioSource
from neural_foundry.gym_lifecycle import AdapterIdentity, EvaluatorAdapter, GymLifecycle, TaskAdapter, public_panel_evaluator
from neural_foundry.gym_service import GymStudioService
def digest(value):
    return 'sha256:' + hashlib.sha256(value.encode()).hexdigest()
identity = digest('synthetic-automodel-contract')
source = ScenarioSource('source', 'provider', '1', identity, digest('source'), 'simulated', ('train','validation','holdout'), {})
job = GymJobSpec(job_id='automodel-synthetic', title='Synthetic routing', description='Return the label.',
    task_input_schema={'type':'object','properties':{'text':{'type':'string'}},'required':['text'],'additionalProperties':False},
    task_output_schema={'type':'string'}, scenario_sources=(source,),
    scorer=ComponentReference('score','1',identity,{}), verifier=ComponentReference('verify','1',identity,{}),
    candidate_methods=tuple(CandidateMethod(name,'1',identity,{}) for name in ('unchanged_base','sft','lora','qlora')),
    gates=(AcceptanceGate('task','task','accuracy','validation','ge',1.0),
           AcceptanceGate('behavior','behavior','accuracy','holdout','ge',1.0),
           AcceptanceGate('controls','controls','beats_control','controls','eq',1.0),
           AcceptanceGate('repeat','reproducibility','repeat','reproducibility','eq',1.0)),
    metadata={}, backends={'trainer':'nemo_automodel.sft@1'})
service = GymStudioService(state_dir=Path(sys.argv[2]) / 'state')
def bind(service):
    service.components.register('scenario_provider','provider','1',identity,
        lambda *,split,**kwargs:[{'scenario_id':split,'item_id':split,'family_id':split,
                                 'input':{'text':'route '+split},'reference':'billing'}])
    service.components.register('verifier','verify','1',identity,
        lambda *,reference,response,**kwargs:{'passed':response==reference})
    service.components.register('scorer','score','1',identity,
        lambda *,verification,**kwargs:{'accuracy':float(verification['passed'])})
    for method in job.candidate_methods:
        service.components.register('candidate_method',method.method_id,method.version,identity,lambda **kwargs:None)
def project(value, response):
    messages = [{'role':'user','content':value['text']}]
    if response is not None:
        messages.append({'role':'assistant','content':response})
    return {'format':'chat_messages_v1','messages':messages}
task = TaskAdapter(AdapterIdentity('synthetic-task','1',identity),job,'single_response',
                   AdapterIdentity('projection','1',identity),'chat_messages_v1',project,bind)
base = transformers_model('Qwen/Qwen3-0.6B','c1899de289a04d12100db370d81485cdf75e47ca',
    device='cuda:0',precision='bfloat16',context_tokens=512,max_new_tokens=16,seed=17)
evaluator = EvaluatorAdapter(AdapterIdentity('public','1',identity),public_panel_evaluator)
lifecycle = GymLifecycle(service)
for adapter in (task,base,evaluator): lifecycle.register(adapter)
lifecycle.register_backend('nemo_automodel.sft@1',config=AutoModelBackendConfig(sys.argv[1],240,sys.argv[4]))
recipe = {'method':sys.argv[3],'seed':17,'steps':1,'max_sequence_tokens':256,
          'tokens_per_batch':256,'max_dataset_tokens':256,'learning_rate':0.00001}
if recipe['method'] in {'lora','qlora'}: recipe.update({'rank':2,'target_modules':['q_proj','v_proj']})
request = {'run_id':'synthetic','task':'synthetic-task@1','model':base.identity.adapter_id+'@1',
           'trainer':'nemo_automodel.sft@1','evaluator':'public@1','recipe':recipe,
           'approved_responses':{'train':'billing'}}
first = lifecycle.run(request)
second = lifecycle.run(request)
assert first['artifact_reused'] is False and second['artifact_reused'] is True
assert first['candidate_record']['candidate'] == second['candidate_record']['candidate']
assert second['approval']['release_authorized'] is False
print(json.dumps({'artifact_reused':second['artifact_reused'],'decision':second['scorecard']['decision']}))
"""
    result = run_local_worker([gpu_python, "-c", code, automodel_python, str(tmp_path), method, gpu_python],
                              cwd=tmp_path, timeout_seconds=420)
    assert result.returncode == 0 and not result.timed_out, (result.stdout + result.stderr).decode("utf-8", "replace")[-6000:]
    assert json.loads(result.stdout.decode().splitlines()[-1])["artifact_reused"] is True