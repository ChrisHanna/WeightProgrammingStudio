"""NeMo resource server for public digest-bound JSON scoring."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, ClassVar

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig, BaseVerifyRequest, BaseVerifyResponse, ReverifyMode, SimpleResourcesServer,
)
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from pydantic import ConfigDict


class JsonResourcesConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS


class JsonVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    nf_reference: Any
    nf_scenario: dict[str, Any]
    nf_job_digest: str
    nf_scorer_digest: str


class JsonVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    nf_reference: Any
    nf_scenario: dict[str, Any]
    nf_job_digest: str
    nf_scorer_digest: str


def _score(body: JsonVerifyRequest, response_text: str) -> float:
    server = Path(__file__).parent
    root = server.parent.parent
    receipt = json.loads((root / "receipt.json").read_bytes())
    job = json.loads((root / "job.json").read_bytes())
    if receipt.get("split") not in {"train", "validation"}:
        raise ValueError("only public splits are supported")
    data = (server / "data" / f"{receipt['split']}.jsonl").read_bytes()
    if "sha256:" + hashlib.sha256(data).hexdigest() != receipt.get("dataset_digest"):
        raise ValueError("dataset digest mismatch")
    if body.nf_job_digest != receipt.get("job_digest") or body.nf_scorer_digest != receipt.get("scorer", {}).get("implementation_digest"):
        raise ValueError("job or scorer identity mismatch")
    rows = [json.loads(line) for line in data.splitlines() if line]
    matches = [row for row in rows if row.get("nf_scenario", {}).get("scenario_id") == body.nf_scenario.get("scenario_id")]
    if len(matches) != 1:
        raise ValueError("scenario is not in the frozen public roster")
    row = matches[0]
    original = NeMoGymResponseCreateParamsNonStreaming.model_validate(row["responses_create_params"])
    if (row["nf_scenario"] != body.nf_scenario or row["nf_reference"] != body.nf_reference
        or row["nf_job_digest"] != body.nf_job_digest or row["nf_scorer_digest"] != body.nf_scorer_digest
        or original.input != body.responses_create_params.input or original.text != body.responses_create_params.text):
        raise ValueError("request differs from the frozen public scenario")
    python = os.environ.get("NF_NEMO_SCORER_PYTHON", "")
    pythonpath = os.environ.get("NF_NEMO_SCORER_PYTHONPATH", "")
    if not Path(python).is_absolute() or not Path(pythonpath).is_absolute():
        raise ValueError("absolute scorer interpreter and package root are required")
    worker = Path(pythonpath) / "neural_foundry" / "nemo_json_score_worker.py"
    if "sha256:" + hashlib.sha256(worker.read_bytes()).hexdigest() != receipt.get("scorer_worker_digest"):
        raise ValueError("scorer worker digest mismatch")
    payload = json.dumps({"job": job, "scenario": body.nf_scenario, "reference": body.nf_reference,
                          "response_text": response_text}, allow_nan=False, separators=(",", ":"))
    env = {"PATH": os.defpath, "PYTHONPATH": pythonpath, "HOME": os.environ.get("HOME", str(root))}
    result = subprocess.run([python, "-m", "neural_foundry.nemo_json_score_worker"], input=payload,
                            capture_output=True, text=True, timeout=45, env=env, cwd=root)
    if result.returncode != 0:
        raise ValueError("scorer worker failed")
    scored = json.loads(result.stdout)
    if scored.get("scorer_digest") != body.nf_scorer_digest or type(scored.get("score")) is not float:
        raise ValueError("scorer worker returned an invalid identity or score")
    if not 0 <= scored["score"] <= 1:
        raise ValueError("scorer worker returned an out-of-range score")
    return scored["score"]


class JsonVerifier:
    async def verify(self, body: JsonVerifyRequest) -> JsonVerifyResponse:
        response = body.response
        items = response.output
        if (response.status != "completed" or len(items) != 1 or items[0].type != "message"
            or items[0].status != "completed" or len(items[0].content) != 1
            or items[0].content[0].type != "output_text"):
            return JsonVerifyResponse(**body.model_dump(), reward=0.0, mask_sample=True,
                                      failure_reason="incomplete or unsupported response")
        try:
            score = await asyncio.to_thread(_score, body, items[0].content[0].text)
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
            return JsonVerifyResponse(**body.model_dump(), reward=0.0, mask_sample=True,
                                      failure_reason="scorer or public package verification failed")
        return JsonVerifyResponse(**body.model_dump(), reward=score)


class JsonResourcesServer(JsonVerifier, SimpleResourcesServer):
    ray_enabled = False
    config: JsonResourcesConfig


if __name__ == "__main__":
    JsonResourcesServer.run_webserver()