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
from neural_foundry.gym_service import _document_digest, _wire_document
from neural_foundry.provenance import resolve_checkpoint
from neural_foundry.reference_json_extraction import TEMPLATES, generate_scenarios, reference_task


def _references(tmp_path: Path):
    base_dir, sft_dir, edited_dir = tmp_path / "base", tmp_path / "sft", tmp_path / "edited"
    for root in (base_dir, sft_dir, edited_dir):
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
    edited_request = "sha256:" + "e" * 64
    (edited_dir / "training_receipt.json").write_text(json.dumps({"training_request_digest": edited_request, "model_digest": state}))
    edited_digest = _document_digest({"training_request": edited_request, "files": artifact_files(edited_dir)})
    edited = CandidateRef("test/model", "weight_edit", base_digest, state, edited_digest, str(edited_dir), "edited", "weight_edit", "qwen3")
    write_candidate_manifest(edited_dir, model_id="test/model", kind="weight_edit", base_digest=base_digest,
        model_digest=state, request_digest=edited_request, metadata={"candidate": {
        "candidate_id": "edited", "method_id": "weight_edit", "base_model_digest": base_digest,
        "candidate_model_digest": state, "artifact_digest": edited_digest}})
    return base, sft, edited, revision


def _failure_report(tmp_path: Path, *, rows_per_template: int = 1, seed: int = 17) -> Path:
    task = reference_task(rows_per_template=rows_per_template)
    rows = generate_scenarios(split="train", seed=seed, parameters={"rows_per_template": rows_per_template})
    report = {"kind": "nf_json_public_split_baseline_v1", "scenario_origin": "synthetic",
              "current_job_digest": task.job.digest, "seed": seed, "rows_per_template": rows_per_template,
              "base_state_digest": "sha256:" + "a" * 64, "base_candidate_digest": "sha256:" + "b" * 64,
              "holdout_accessed": False,
              "train": {"score": {"responses": [{"scenario_id": row["scenario_id"], "exact_match": 0.0} for row in rows]},
                        "failures": [row["scenario_id"] for row in rows], "failure_count": len(rows)}}
    report["evidence_digest"] = _document_digest(report)
    path = tmp_path / "base-failures.json"
    path.write_bytes(_wire_document(report))
    return path


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
    base, sft, edited, revision = _references(tmp_path)
    task = reference_task(rows_per_template=1)
    failure_report = _failure_report(tmp_path, rows_per_template=1)
    result = {"job_digest": task.job.digest, "revision": revision, "base": vars(base), "sft": vars(sft),
              "training_request_digest": "sha256:" + "c" * 64, "candidate_record_digest": "sha256:" + "d" * 64,
              "weight_edit": vars(edited), "weight_edit_training_request_digest": "sha256:" + "e" * 64,
              "weight_edit_candidate_record_digest": "sha256:" + "f" * 64, "dataset_digest": "sha256:" + "a" * 64}
    monkeypatch.setenv("NF_FRONTIER_API_KEY", "private-token")

    def train(command, *, env, **kwargs):
        assert "NF_FRONTIER_API_KEY" not in env
        assert command[-4:-1] == ["0.01", "2", "generalist"]
        assert command[-1] == str(failure_report)
        return SimpleNamespace(returncode=0, timed_out=False, receipt={},
                               stdout=b"NF_JSON_PROOF_TRAINING=" + json.dumps(result).encode() + b"\n")

    monkeypatch.setattr(json_proof_workflow, "run_local_worker", train)
    monkeypatch.setattr(json_proof_workflow, "VllmCandidateServer", lambda *args, **kwargs: _FakeModel())
    output = tmp_path / "proof"
    report = json_proof_workflow.run_json_proof(output, gpu_python=sys.executable, automodel_python=sys.executable,
                                                 rows_per_template=1, frontier_factory=lambda: _FakeModel(),
                                                 base_failure_report_path=failure_report)
    assert report["complete"] is True and report["comparison"]["scenario_count"] == 2
    assert report["scorer_audit"]["passed"] is True
    assert report["scorer_audit"]["cases"] == 3 * (len(TEMPLATES["train"]) + len(TEMPLATES["validation"]))
    assert all(card["value"] == 0 for card in report["comparison"]["scorecards"].values())
    assert not list(output.glob("nemo_*/resources_servers/nf_json_extraction/data/holdout.jsonl"))
    assert json.loads((output / "report.json").read_text()) == report
    assert "private-token" not in json.dumps(report)
    assert report["release_authorized"] is False
    assert set(report["comparison"]["scorecards"]) == {"base", "sft", "weight_edit", "frontier"}
    assert all(card["newly_broken_cases"] == 0 for card in report["comparison"]["scorecards"].values())
    second = json_proof_workflow.run_json_proof(tmp_path / "local-proof", gpu_python=sys.executable,
                                                automodel_python=sys.executable, rows_per_template=1,
                                                base_failure_report_path=failure_report)
    assert second["frontier_configured"] is False
    assert set(second["comparison"]["scorecards"]) == {"base", "sft", "weight_edit"}


def test_missing_frontier_fails_before_creating_output(tmp_path, monkeypatch) -> None:
    from neural_foundry.json_frontier_model import JsonFrontierModel
    for name in ("NF_FRONTIER_URL", "NF_FRONTIER_MODEL", "NF_FRONTIER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    output = tmp_path / "proof"
    with pytest.raises(GymJobError, match="endpoint"):
        json_proof_workflow.run_json_proof(output, gpu_python=sys.executable, automodel_python=sys.executable,
                          frontier_factory=JsonFrontierModel.from_env)
    assert not output.exists()


def test_specialist_reports_three_arms_but_blocks_holdout_on_required_set_failure(tmp_path, monkeypatch):
    base, sft, edited, revision = _references(tmp_path)
    task = reference_task(rows_per_template=1)
    failure_report = _failure_report(tmp_path, rows_per_template=1)
    result = {"job_digest": task.job.digest, "revision": revision, "base": vars(base), "sft": vars(sft),
              "training_request_digest": "sha256:" + "c" * 64, "candidate_record_digest": "sha256:" + "d" * 64,
              "weight_edit": vars(edited), "weight_edit_training_request_digest": "sha256:" + "e" * 64,
              "weight_edit_candidate_record_digest": "sha256:" + "f" * 64, "dataset_digest": "sha256:" + "a" * 64,
              "retention_mode": "specialist"}
    receipt_path = Path(edited.artifact_path) / "training_receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    receipt.update({"retention_mode": "specialist", "preservation_verification": {"passed": False}})
    receipt_path.write_text(json.dumps(receipt))
    monkeypatch.setattr(json_proof_workflow, "verify_vllm_candidate_files", lambda *args, **kwargs: None)
    def train(command, **kwargs):
        assert command[-2] == "specialist" and command[-1].endswith("base-failures.json")
        return SimpleNamespace(returncode=0, timed_out=False, receipt={},
                               stdout=b"NF_JSON_PROOF_TRAINING=" + json.dumps(result).encode() + b"\n")
    monkeypatch.setattr(json_proof_workflow, "run_local_worker", train)
    monkeypatch.setattr(json_proof_workflow, "VllmCandidateServer", lambda *args, **kwargs: _FakeModel())
    def forbidden(*args, **kwargs):
        raise AssertionError("holdout must remain untouched")
    monkeypatch.setattr(json_proof_workflow, "compare_synthetic_holdout", forbidden)
    report = json_proof_workflow.run_json_proof(tmp_path / "specialist", gpu_python=sys.executable,
                    automodel_python=sys.executable, rows_per_template=1, retention_mode="specialist",
                    base_failure_report_path=failure_report)
    assert report["complete"] and report["public_validation_comparison"]["complete"]
    assert set(report["public_validation_comparison"]["scorecards"]) == {"base", "sft", "weight_edit"}
    assert report["comparison"] is None and not report["holdout_accessed"] and not report["specialist_gate_passed"]
    assert report["preservation"] == {"passed": False} and not report["preservation_gating"]
    acceptance = report["public_validation_comparison"]["scorecards"]["weight_edit"]["acceptance"]
    assert acceptance["policy_version"] == 2 and acceptance["policy_status"] == "approved_frozen"
    assert "confidence_interval_lower_bound" in acceptance["checks"]


def test_retained_failed_public_recipes_do_not_claim_holdout_scores():
    path = Path(__file__).resolve().parents[1] / "docs" / "evidence" / "weight-edit-three-way-preservation-failures.json"
    report = json.loads(path.read_bytes())
    assert report["evidence_digest"] == _document_digest({key: value for key, value in report.items() if key != "evidence_digest"})
    assert report["three_way_complete"] is False and report["holdout_evaluated"] is False
    assert report["scores"] is None and report["release_authorized"] is False
    assert [trial["target_step"] for trial in report["trials"]] == [0.01, 0.001, 0.0001]
    assert all(trial["limit"] == 0.25 and trial["complete"] is False for trial in report["trials"])
    assert report["trials"][0]["measured_drift"] is None
    assert report["trials"][1]["measured_drift"] == 0.5625 and report["trials"][2]["measured_drift"] == 0.4375


def test_public_scorer_uses_batched_model_interface_when_available():
    from neural_foundry.json_proof_loop import score_public_rows
    from neural_foundry.reference_json_extraction import generate_scenarios

    task = reference_task(rows_per_template=1)

    class Batched:
        called = False

        def infer_many(self, interactions):
            self.called = True
            assert len(interactions) == 4
            return [{"customer": "wrong", "product": "wrong", "quantity": 1} for _ in interactions]

        def infer(self, interaction):
            raise AssertionError("serial inference should not run when infer_many is provided")

        def loaded_identity(self):
            return {"fixture": True}

    model = Batched()
    rows = generate_scenarios(split="validation", seed=17, parameters={"rows_per_template": 2})
    score = score_public_rows(task, model, rows)
    assert model.called and score["cases"] == 4 and score["format_valid"] == 1.0
    assert score["exact_match"] == 0.0

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


def test_real_gpu_local_three_way_weight_edit_proof(tmp_path) -> None:
    gpu_python = os.environ.get("NF_GPU_PYTHON")
    automodel_python = os.environ.get("NF_AUTOMODEL_PYTHON")
    if not gpu_python or not automodel_python or os.environ.get("NF_JSON_PROOF_LOCAL_LIVE") != "1":
        pytest.skip("set NF_GPU_PYTHON, NF_AUTOMODEL_PYTHON and NF_JSON_PROOF_LOCAL_LIVE=1")
    report = json_proof_workflow.run_json_proof(tmp_path / "proof", gpu_python=gpu_python,
                                               automodel_python=automodel_python, rows_per_template=32, sft_steps=32)
    assert report["complete"] is True, report["comparison"].get("failure")
    assert report["frontier_configured"] is False
    assert set(report["comparison"]["scorecards"]) == {"base", "sft", "weight_edit"}
    assert report["comparison"]["scenario_count"] == 64
    assert report["comparison"]["preservation_count"] > 0