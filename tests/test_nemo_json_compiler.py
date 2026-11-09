from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace

import pytest

from neural_foundry.gym_job import GymJobError
from neural_foundry.gym_service import GymStudioService
from neural_foundry.nemo_json_compiler import compile_json_gym
from neural_foundry.reference_json_extraction import generate_scenarios, reference_task


@pytest.mark.parametrize("split", ["train", "validation"])
def test_compile_public_json_package(tmp_path, split: str) -> None:
    task = reference_task(rows_per_template=2)
    service = GymStudioService(state_dir=tmp_path / "state")
    task.bind(service)
    destination = tmp_path / "package"
    receipt = compile_json_gym(task, service, split=split, seed=17, output_dir=destination)
    server = destination / "resources_servers" / "nf_json_extraction"
    rows = [json.loads(line) for line in (server / "data" / f"{split}.jsonl").read_text().splitlines()]
    config = json.loads((server / "configs" / "nf_json_extraction.yaml").read_text())
    expected_rows = len(generate_scenarios(
        split=split, seed=17, parameters={"rows_per_template": 2},
    ))
    assert len(rows) == receipt["rows"] == expected_rows
    assert config["nf_json_resources_server"]["resources_servers"]["nf_json_extraction"]["verified"] is False
    assert receipt["execution_ready"] is False and receipt["verifier_bound"] is True
    assert (server / "app.py").is_file()
    for row in rows:
        assert row["nf_scorer_digest"] == task.job.scorer.implementation_digest
        assert row["nf_scenario"]["split"] == split
        assert "nf_reference" not in row["responses_create_params"]
        assert row["responses_create_params"]["text"]["format"]["schema"] == task.job.to_document()["task"]["output_schema"]
    with pytest.raises(FileExistsError):
        compile_json_gym(task, service, split=split, seed=17, output_dir=destination)


def test_holdout_and_non_json_are_rejected_before_output(tmp_path) -> None:
    task = reference_task()
    service = GymStudioService(state_dir=tmp_path / "state")
    task.bind(service)
    output = tmp_path / "package"
    with pytest.raises(GymJobError, match="never holdout"):
        compile_json_gym(task, service, split="holdout", seed=1, output_dir=output)
    assert not output.exists()
    with pytest.raises(GymJobError, match="JSON extraction"):
        compile_json_gym(replace(task, interaction_format="tool_messages_v1"), service,
                         split="train", seed=1, output_dir=output)
    assert not output.exists()


def test_generated_rows_match_installed_nemo_request_model(tmp_path) -> None:
    python = os.environ.get("NF_NEMO_GYM_PYTHON")
    if not python:
        pytest.skip("set NF_NEMO_GYM_PYTHON to the pinned NeMo Gym interpreter")
    task = reference_task(rows_per_template=1)
    service = GymStudioService(state_dir=tmp_path / "state")
    task.bind(service)
    destination = tmp_path / "package"
    compile_json_gym(task, service, split="train", seed=17, output_dir=destination)
    server = destination / "resources_servers" / "nf_json_extraction"
    code = (
        "import json, sys\n"
        "from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming\n"
        "from omegaconf import OmegaConf\n"
        "rows = [json.loads(line) for line in open(sys.argv[1])]\n"
        "for row in rows: NeMoGymResponseCreateParamsNonStreaming.model_validate(row['responses_create_params'])\n"
        "assert OmegaConf.load(sys.argv[2])['nf_json_resources_server']\n"
    )
    result = subprocess.run([python, "-c", code, str(server / "data" / "train.jsonl"),
                             str(server / "configs" / "nf_json_extraction.yaml")], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr