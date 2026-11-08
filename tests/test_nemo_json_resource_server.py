from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from neural_foundry.gym_service import GymStudioService
from neural_foundry.nemo_json_compiler import compile_json_gym
from neural_foundry.reference_json_extraction import reference_task


def test_real_nemo_verifier_scoring_and_audit(tmp_path) -> None:
    python = os.environ.get("NF_NEMO_GYM_PYTHON")
    if not python:
        pytest.skip("set NF_NEMO_GYM_PYTHON to the pinned NeMo Gym interpreter")
    task = reference_task(rows_per_template=1)
    service = GymStudioService(state_dir=tmp_path / "state")
    task.bind(service)
    package = tmp_path / "package"
    compile_json_gym(task, service, split="validation", seed=7, output_dir=package)
    code = '''
import copy, importlib.util, json, os, pathlib, sys
from fastapi.testclient import TestClient
root = pathlib.Path(sys.argv[1])
server = root / "resources_servers" / "nf_json_extraction"
os.environ["NEMO_GYM_EXTRA_ROOTS"] = str(root)
from nemo_gym.resources_server_registry import discover_resources_servers
assert discover_resources_servers()["nf_json_extraction"].path == server
spec = importlib.util.spec_from_file_location("nf_json_app", server / "app.py")
app = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = app
spec.loader.exec_module(app)
config = app.JsonResourcesConfig(host="127.0.0.1", port=1, entrypoint="app.py", name="test")
instance = app.JsonResourcesServer.model_construct(config=config, server_client=None)
client = TestClient(instance.setup_webserver())
row = json.loads((server / "data" / "validation.jsonl").read_text().splitlines()[0])
def request(text, status="completed"):
    return {**row, "response": {"id":"fixture", "created_at":0, "model":"fixture", "object":"response",
        "status":status, "output":[{"id":"message", "type":"message", "role":"assistant", "status":"completed",
        "content":[{"type":"output_text", "text":text, "annotations":[]}]}],
        "parallel_tool_calls":False, "tool_choice":"none", "tools":[]}}
correct = request(json.dumps(row["nf_reference"]))
result = client.post("/verify", json=correct)
assert result.status_code == 200, result.text
assert result.json()["reward"] == 1.0 and result.json()["mask_sample"] is False, result.text
incorrect = request(json.dumps({**row["nf_reference"], "customer":"wrong"}))
result = client.post("/verify", json=incorrect)
assert result.status_code == 200 and result.json()["reward"] == 0.0 and result.json()["mask_sample"] is False, result.text
malformed = client.post("/verify", json=request("not json"))
assert malformed.status_code == 200 and malformed.json()["reward"] == 0.0 and malformed.json()["mask_sample"] is False, malformed.text
incomplete = client.post("/verify", json=request(json.dumps(row["nf_reference"]), "incomplete"))
assert incomplete.status_code == 200 and incomplete.json()["mask_sample"] is True, incomplete.text
tampered = copy.deepcopy(correct)
tampered["nf_reference"]["customer"] = "forged"
result = client.post("/verify", json=tampered)
assert result.status_code == 200 and result.json()["mask_sample"] is True, result.text
tampered = copy.deepcopy(correct)
tampered["nf_prompt_projection_digest"] = "sha256:" + "0" * 64
result = client.post("/verify", json=tampered)
assert result.status_code == 200 and result.json()["mask_sample"] is True, result.text
'''
    env = {**os.environ, "NF_NEMO_SCORER_PYTHON": sys.executable,
           "NF_NEMO_SCORER_PYTHONPATH": str(Path(__file__).resolve().parents[1] / "python")}
    result = subprocess.run([python, "-c", code, str(package)], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr