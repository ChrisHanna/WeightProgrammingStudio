from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from neural_foundry import json_proof_workflow
from neural_foundry.gym_artifacts import CandidateRef, artifact_files, write_candidate_manifest
from neural_foundry.gym_job import GymJobError
from neural_foundry.gym_service import _document_digest
from neural_foundry.provenance import resolve_checkpoint
from neural_foundry.reference_json_extraction import reference_task


def _references(tmp_path: Path):
    base_dir, sft_dir = tmp_path / "base", tmp_path / "sft"
    for root in (base_dir, sft_dir):
        root.mkdir()
        for name in ("config.json", "tokenizer.json", "model.safetensors"):
            (root / name).write_text(name)
    revision = "a" * 40
    base_digest = _document_digest({"model_id": "test/model", "revision": revision,
                                    "files": resolve_checkpoint(base_dir).file_digests})
    base = CandidateRef("test/model", "full", base_digest, base_digest, base_digest, str(base_dir),
                        "base", "unchanged_base", "qwen3")
    state, request = "sha256:" + "b" * 64, "sha256:" + "c" * 64
    (sft_dir / "training_receipt.json").write_text(json.dumps({"training_request_digest": request, "model_digest": state}))
    digest = _document_digest({"training_request": request, "files": artifact_files(sft_dir)})
    sft = CandidateRef("test/model", "full", base_digest, state, digest, str(sft_dir), "sft", "sft", "qwen3")
    candidate = {"candidate_id": "sft", "method_id": "sft", "base_model_digest": base_digest,
                 "candidate_model_digest": state, "artifact_digest": digest}
    write_candidate_manifest(sft_dir, model_id="test/model", kind="full", base_digest=base_digest,
                             model_digest=state, request_digest=request, metadata={"candidate": candidate})
    return base, sft, revision


class _FakeModel:
    def __init__(self, *, fail=False):
        self.fail = fail

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def loaded_identity(self):
        return {"fixture": True, "served_model_identity_verified": False}

    def infer(self, interaction):
        if self.fail:
            raise RuntimeError("frontier unavailable")
        return {}


def test_workflow_compiles_trains_and_compares_without_exposing_credentials(tmp_path, monkeypatch) -> None:
    base, sft, revision = _references(tmp_path)
    task = reference_task(rows_per_template=1)
    result = {"job_digest": task.job.digest, "revision": revision, "base": vars(base), "sft": vars(sft),
              "training_request_digest": "sha256:" + "c" * 64, "candidate_record_digest": "sha256:" + "d" * 64}
    monkeypatch.setenv("NF_FRONTIER_API_KEY", "private-token")

    def train(command, *, env, **kwargs):
        assert "NF_FRONTIER_API_KEY" not in env
        return SimpleNamespace(returncode=0, timed_out=False, receipt={},
                               stdout=b"NF_JSON_PROOF_TRAINING=" + json.dumps(result).encode() + b"\n")

    monkeypatch.setattr(json_proof_workflow, "run_local_worker", train)
    monkeypatch.setattr(json_proof_workflow, "VllmCandidateServer", lambda *args, **kwargs: _FakeModel())
    output = tmp_path / "proof"
    report = json_proof_workflow.run_json_proof(output, gpu_python=sys.executable, automodel_python=sys.executable,
                                                 frontier_factory=lambda: _FakeModel())
    assert report["complete"] is True and report["comparison"]["scenario_count"] == 2
    assert report["scorer_audit"]["passed"] is True and report["scorer_audit"]["cases"] == 12
    assert all(card["value"] == 0 for card in report["comparison"]["scorecards"].values())
    assert not list(output.glob("nemo_*/resources_servers/nf_json_extraction/data/holdout.jsonl"))
    assert json.loads((output / "report.json").read_text()) == report
    assert "private-token" not in json.dumps(report)
    assert report["release_authorized"] is False


def test_missing_frontier_fails_before_creating_output(tmp_path, monkeypatch) -> None:
    for name in ("NF_FRONTIER_URL", "NF_FRONTIER_MODEL", "NF_FRONTIER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    output = tmp_path / "proof"
    with pytest.raises(GymJobError, match="endpoint"):
        json_proof_workflow.run_json_proof(output, gpu_python=sys.executable, automodel_python=sys.executable)
    assert not output.exists()


def test_real_gpu_workflow_with_explicit_frontier_stub(tmp_path) -> None:
    gpu_python = os.environ.get("NF_GPU_PYTHON")
    automodel_python = os.environ.get("NF_AUTOMODEL_PYTHON")
    if not gpu_python or not automodel_python or os.environ.get("NF_JSON_PROOF_LIVE") != "1":
        pytest.skip("set NF_GPU_PYTHON, NF_AUTOMODEL_PYTHON and NF_JSON_PROOF_LIVE=1")
    report = json_proof_workflow.run_json_proof(tmp_path / "proof", gpu_python=gpu_python,
                                               automodel_python=automodel_python, frontier_factory=lambda: _FakeModel())
    assert report["complete"] is True, report["comparison"].get("failure")
    assert report["comparison"]["scorecards"]["frontier"]["model_identity"]["fixture"] is True
    assert report["comparison"]["served_model_identity_verified"] is False