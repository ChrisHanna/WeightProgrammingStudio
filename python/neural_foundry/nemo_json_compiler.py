"""Compile public JSON-extraction Gym Job data for NeMo Gym."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .gym_job import GymJobError, _validate_schema
from .gym_lifecycle import TaskAdapter
from .gym_service import GymStudioService, _document_digest, _wire_document
from .records import canonicalize, content_digest


def compile_json_gym(
    task: TaskAdapter, service: GymStudioService, *, split: str, seed: int, output_dir: str | Path,
) -> dict[str, Any]:
    if split not in {"train", "validation"}:
        raise GymJobError("NeMo JSON compilation accepts public train or validation only, never holdout")
    if task.interaction_format != "json_extraction_v1" or task.interaction_type != "single_response":
        raise GymJobError("NeMo JSON compilation requires single-response JSON extraction")
    output = Path(output_dir).absolute()
    if output.exists():
        raise FileExistsError(f"NeMo package already exists: {output}")
    batch = service.materialize_public_scenarios(task.job, split=split, seed=seed)
    output_schema = task.job.to_document()["task"]["output_schema"]
    rows = []
    for scenario in batch["scenarios"]:
        interaction = task.project(scenario["input"], None)
        if (not isinstance(interaction, dict) or set(interaction) != {"format", "messages", "output_schema"}
            or interaction["format"] != "json_extraction_v1" or interaction["output_schema"] != output_schema):
            raise GymJobError("JSON projection must match the frozen job output schema")
        _validate_schema(interaction["output_schema"], "output_schema")
        messages = interaction["messages"]
        if (not isinstance(messages, list) or not messages
            or any(not isinstance(message, dict) or set(message) != {"role", "content"}
                   or message["role"] not in {"system", "developer", "user"}
                   or not isinstance(message["content"], str) for message in messages)):
            raise GymJobError("JSON projection must contain public input messages")
        reference = service._scenario_references.get((batch["manifest_digest"], scenario["scenario_id"]))
        if str(content_digest(canonicalize({"kind": "nf_gym_reference_v1", "reference": reference}))) != scenario["reference_digest"]:
            raise GymJobError("public scenario reference is missing or changed")
        rows.append({
            "responses_create_params": {"input": messages, "text": {"format": {
                "type": "json_schema", "name": "nf_output", "strict": True, "schema": output_schema,
            }}},
            "nf_reference": reference,
            "nf_scenario": scenario,
            "nf_job_digest": task.job.digest,
            "nf_scorer_digest": task.job.scorer.implementation_digest,
        })
    dataset = b"".join(_wire_document(row) + b"\n" for row in rows)
    config = {
        "nf_json_resources_server": {"resources_servers": {"nf_json_extraction": {
            "entrypoint": "app.py", "domain": "other", "verified": False,
            "datasets": [{"name": split, "type": split,
                          "jsonl_fpath": f"resources_servers/nf_json_extraction/data/{split}.jsonl",
                          "num_repeats": 1}],
        }}},
        "nf_json_simple_agent": {"responses_api_agents": {"simple_agent": {
            "entrypoint": "app.py", "resources_server": {"type": "resources_servers", "name": "nf_json_resources_server"},
            "model_server": {"type": "responses_api_models", "name": "policy_model"},
        }}},
    }
    receipt = {
        "kind": "nf_nemo_json_package_v1", "job_digest": task.job.digest, "split": split,
        "scenario_manifest_digest": batch["manifest_digest"], "scorer": task.job.scorer.to_document(),
        "verifier": task.job.verifier.to_document(), "rows": len(rows),
        "dataset_digest": "sha256:" + hashlib.sha256(dataset).hexdigest(),
        "scorer_worker_digest": "sha256:" + hashlib.sha256(Path(__file__).with_name("nemo_json_score_worker.py").read_bytes()).hexdigest(),
        "execution_ready": False, "verifier_bound": True,
    }
    receipt["receipt_digest"] = _document_digest(receipt)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".nf-nemo-json-", dir=output.parent))
    try:
        server = staging / "resources_servers" / "nf_json_extraction"
        (server / "data").mkdir(parents=True)
        (server / "configs").mkdir()
        (server / "data" / f"{split}.jsonl").write_bytes(dataset)
        (server / "configs" / "nf_json_extraction.yaml").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        (server / "app.py").write_bytes(Path(__file__).with_name("nemo_json_resource_server.py").read_bytes())
        (staging / "job.json").write_text(json.dumps(task.job.to_document(), indent=2) + "\n", encoding="utf-8")
        (staging / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"NeMo package already exists: {output}")
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return receipt