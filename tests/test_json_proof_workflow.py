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
from neural_foundry.json_injection_training_data import (
    preview_train_injection_rows,
    generate_train_injection_rows,
)


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


def test_frontier_env_file_reads_allowlisted_names_without_exposing_key(tmp_path: Path, monkeypatch) -> None:
    from neural_foundry.json_frontier_model import (
        OPENAI_CHAT_COMPLETIONS_URL,
        OPENAI_DEFAULT_MODEL,
        JsonFrontierModel,
    )

    secret = "test-only-key-value"
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"OPENAI_API_KEY='{secret}'\nNF_FRONTIER_MODEL=gpt-4.1-mini\nUNRELATED=value\n",
        encoding="utf-8",
    )
    for name in ("OPENAI_API_KEY", "NF_FRONTIER_API_KEY", "NF_FRONTIER_URL", "NF_FRONTIER_MODEL"):
        monkeypatch.delenv(name, raising=False)

    frontier = JsonFrontierModel.from_env(env_file=env_file)

    assert frontier._endpoint == OPENAI_CHAT_COMPLETIONS_URL
    assert frontier.loaded_identity()["provider_reported_model"] == "gpt-4.1-mini"
    assert secret not in json.dumps(frontier.loaded_identity())
    assert OPENAI_DEFAULT_MODEL == "gpt-6-luna"
    monkeypatch.setenv("NF_FRONTIER_MODEL", "gpt-4.1-nano")
    overridden = JsonFrontierModel.from_env(env_file=env_file)
    assert overridden.loaded_identity()["provider_reported_model"] == "gpt-4.1-nano"


def test_weight_edit_preservation_panel_respects_backend_limit() -> None:
    from neural_foundry.json_proof_train_worker import bound_preservation_panel

    interactions = [{"scenario": index} for index in range(165)]
    panel = bound_preservation_panel(interactions)
    assert len(panel) == 128
    assert panel == interactions[:128]


def test_training_execution_rejects_tampered_preparation_before_loading(tmp_path) -> None:
    import runpy
    module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "run_gym_lifecycle_real.py"))
    document = module["prepare_json_training_request"](rows_per_template=1)
    document["training_rows"][0]["input"]["text"] = "tampered"
    with pytest.raises(GymJobError, match="digest differs"):
        module["execute_json_training_request"](document, output=tmp_path / "must-not-exist", method="sft",
            automodel_python="/missing/python", reload_python="/missing/reload")
    assert not (tmp_path / "must-not-exist").exists()


def test_training_preparation_freezes_only_train_without_loading_or_dispatch(tmp_path, monkeypatch) -> None:
    import runpy
    from neural_foundry import reference_json_extraction

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_gym_lifecycle_real.py"
    module = runpy.run_path(str(script))
    original = reference_json_extraction.generate_scenarios
    splits = []

    def train_only(**kwargs):
        splits.append(kwargs["split"])
        assert kwargs["split"] == "train"
        return original(**kwargs)

    monkeypatch.setattr(reference_json_extraction, "generate_scenarios", train_only)
    namespace = module["main"].__globals__
    def forbidden(*args, **kwargs):
        raise AssertionError("preparation must not load a model, resolve a checkpoint, or start a trainer")
    for name in ("resolve_checkpoint", "transformers_model", "peft_sequence_trainer"):
        monkeypatch.setitem(namespace, name, forbidden)
    output = tmp_path / "prepared"
    monkeypatch.setattr(sys, "argv", [str(script), "--prepare-json-training", "--output-dir", str(output),
                                    "--rows-per-template", "1"])
    module["main"]()
    document = json.loads((output / "training-preparation.json").read_bytes())
    assert splits == ["train"]
    assert document["training_cases"] == len(TEMPLATES["train"])
    assert set(document["intents"]) == {"sft", "lora"}
    assert not document["ready_to_train"] and not document["holdout_accessed"] and not document["provider_called"]
    assert not document["download_allowed"] and not document["weight_edit_dispatched"]
    assert document["preparation_digest"] == _document_digest({
        key: value for key, value in document.items() if key != "preparation_digest"})
    for intent in document["intents"].values():
        request = intent["lifecycle_request"]
        assert intent["request_digest"] == _document_digest(request)
        assert all(name.startswith("train:") for name in request["approved_responses"])


def test_robustness_v2_preparation_freezes_and_consumes_all_injection_rows(tmp_path) -> None:
    import runpy
    module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "run_gym_lifecycle_real.py"))
    document = module["prepare_json_robustness_v2_training_request"]()
    contract = document["data_contract"]
    rows = document["training_rows"]
    injected = [row for row in rows if row["scenario_id"].startswith("train:injection-v1:")]

    assert document["kind"] == "nf_json_training_preparation_v2"
    assert contract["base_rows"] == 320 and contract["injection_rows"] == 64
    assert document["training_cases"] == 384 and contract["ordered_steps"] == 384
    assert contract["injection_rows_consumed_by_ordered_steps"] == len(injected) == 64
    assert all(row["scenario_id"].startswith("train:")
               and not row["scenario_id"].startswith("train:injection-v1:") for row in rows[:128:2])
    assert all(rows[index + 1]["scenario_id"].startswith("train:injection-v1:") for index in range(0, 128, 2))
    assert all(intent["lifecycle_request"]["recipe"]["steps"] == 384
               for intent in document["intents"].values())
    assert document["preparation_digest"] == _document_digest({
        key: value for key, value in document.items() if key != "preparation_digest"})
    assert document["holdout_accessed"] is False and document["ready_to_train"] is False

    document["training_rows"][0]["input"]["text"] = "tampered"
    with pytest.raises(GymJobError, match="digest differs"):
        module["execute_json_training_request"](document, output=tmp_path / "must-not-exist", method="sft",
            automodel_python="/missing/python", reload_python="/missing/reload")
    assert not (tmp_path / "must-not-exist").exists()


def test_public_validation_v2_panel_is_frozen_and_disjoint(tmp_path) -> None:
    import runpy
    lifecycle = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "run_gym_lifecycle_real.py"))
    validation = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "run_json_validation.py"))
    preparation = lifecycle["prepare_json_robustness_v2_training_request"]()
    prep_path = tmp_path / "preparation" / "training-preparation.json"
    prep_path.parent.mkdir()
    prep_path.write_text(json.dumps(preparation), encoding="utf-8")

    panel = validation["freeze_public_validation_v2"](tmp_path)

    assert panel["kind"] == "nf_json_public_validation_panel_v2"
    assert len(panel["rows"]) == panel["disjointness_audit"]["panel_cases"] == 192
    assert panel["disjointness_audit"]["by_category"] == {"clean_compositional": 96, "injection": 96}
    assert panel["disjointness_audit"]["by_template"] == {
        "ap_posted_table": 32, "forwarded_mail_conflict": 32, "goods_receipt_crosscheck": 32,
        "portal_annotation_conflict": 32, "remittance_reconciliation": 32, "scan_footer_conflict": 32,
    }
    assert panel["disjointness_audit"]["passed"] is True
    assert panel["holdout_accessed"] is False
    assert panel["rows_digest"] == _document_digest({"rows": panel["rows"]})
    assert panel["document_digest"] == _document_digest({
        key: value for key, value in panel.items() if key != "document_digest"})


def test_frontier_panel_claim_prevents_budget_reset(tmp_path, monkeypatch) -> None:
    import runpy
    from neural_foundry.json_frontier_model import JsonFrontierModel

    lifecycle = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "run_gym_lifecycle_real.py"))
    validation = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "run_json_validation.py"))
    preparation = lifecycle["prepare_json_robustness_v2_training_request"]()
    prep_path = tmp_path / "preparation" / "training-preparation.json"
    prep_path.parent.mkdir()
    prep_path.write_text(json.dumps(preparation), encoding="utf-8")
    validation["freeze_public_validation_v2"](tmp_path)
    monkeypatch.setattr(JsonFrontierModel, "from_env", lambda **kwargs: JsonFrontierModel(
        endpoint="https://api.openai.com/v1/chat/completions", model="gpt-6-luna",
        api_key="test-only", **kwargs))

    def fail_before_network(self):
        raise GymJobError("test interruption before network")

    monkeypatch.setattr(JsonFrontierModel, "__enter__", fail_before_network)
    with pytest.raises(GymJobError, match="test interruption"):
        validation["score_frontier_panel"](tmp_path)
    with pytest.raises(FileExistsError):
        validation["score_frontier_panel"](tmp_path)


@pytest.mark.parametrize("tampered", [False, True])
def test_prior_overlap_audit_annotation_never_reads_holdout(tmp_path, monkeypatch, tampered) -> None:
    import runpy
    import sys

    audit = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "audit_json_contamination.py"))
    prior = {"kind": "nf_json_contamination_audit_v1", "audit_passed": False,
             "synthetic_holdout": {"independent_custody_verified": False, "content_emitted": False}}
    prior["report_digest"] = _document_digest(prior)
    if tampered:
        prior["audit_passed"] = True
    source, output = tmp_path / "prior.json", tmp_path / "correction.json"
    source.write_text(json.dumps(prior), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["audit", "--annotate-prior-audit", str(source), "--output", str(output)])
    monkeypatch.setitem(audit["main"].__globals__, "generate_scenarios",
                        lambda **kwargs: pytest.fail("annotation must not read holdout"))
    if tampered:
        with pytest.raises(ValueError, match="digest or kind"):
            audit["main"]()
        assert not output.exists()
    else:
        audit["main"]()
        corrected = json.loads(output.read_text(encoding="utf-8"))
        assert corrected["holdout_accessed"] is True
        assert corrected["holdout_access_purpose"] == "overlap_only"
        assert corrected["synthetic_holdout"]["access_boundary"] == "outside_independent_custody"
        assert corrected["annotation_reads_holdout"] is False
        assert corrected["audit_passed"] is False
        assert corrected["supersedes_report_digest"] == prior["report_digest"]
        assert corrected["report_digest"] == _document_digest({
            key: value for key, value in corrected.items() if key != "report_digest"})
        assert json.loads(source.read_text(encoding="utf-8")) == prior


@pytest.mark.parametrize("wrong_label", [False, True])
def test_public_label_audit_preserves_panel_or_refuses_bad_labels(wrong_label) -> None:
    import runpy
    from neural_foundry.json_injection_training_data import generate_public_validation_v2_rows

    audit = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "audit_json_public_panel.py"))
    rows = generate_public_validation_v2_rows(seed=20261112)
    if wrong_label:
        rows[0]["reference"] = {"customer": "Wrong", "product": "Wrong", "quantity": 1}
    panel = {"kind": "nf_json_public_validation_panel_v2", "rows": rows,
             "rows_digest": _document_digest({"rows": rows})}
    panel["document_digest"] = _document_digest(panel)
    card = {"complete": True, "panel_digest": panel["document_digest"],
            "panel_rows_digest": panel["rows_digest"],
            "task": {"responses": [{"scenario_id": row["scenario_id"], "response": row["reference"]}
                                  for row in rows]}}
    card["document_digest"] = _document_digest(card)
    result = audit["audit_panel"](panel, card)
    assert result["clean_cases"] == 96
    assert result["label_mismatches"] == int(wrong_label)
    if wrong_label:
        with pytest.raises(GymJobError, match="unresolved public labels"):
            audit["freeze_v3"](panel, result)
    else:
        frozen = audit["freeze_v3"](panel, result)
        assert frozen["rows"] == rows
        assert frozen["rows_digest"] == panel["rows_digest"]
        assert frozen["independent_confirmation_panel"] is False
        assert frozen["prior_public_exposure_retained"] is True
        assert frozen["document_digest"] == _document_digest({
            key: value for key, value in frozen.items() if key != "document_digest"})


def test_public_contamination_audit_cannot_materialize_holdout(monkeypatch) -> None:
    import runpy

    root = Path(__file__).resolve().parents[1]
    lifecycle = runpy.run_path(str(root / "scripts" / "run_gym_lifecycle_real.py"))
    audit = runpy.run_path(str(root / "scripts" / "audit_json_contamination.py"))
    preparation = lifecycle["prepare_json_robustness_v2_training_request"]()
    monkeypatch.setitem(audit["build_audit"].__globals__, "generate_scenarios",
                        lambda **kwargs: pytest.fail("public audit must not generate holdout"))
    report = audit["build_audit"](preparation)
    assert report["kind"] == "nf_json_contamination_audit_v2"
    assert report["injection_vs_required"]["exact_input_matches"] == 0
    assert report["injection_vs_synthetic_holdout"] is None
    assert report["synthetic_holdout"]["holdout_accessed"] is False
    assert report["synthetic_holdout"]["status"] == "not_checked_pending_approved_custodian_interface"
    assert report["audit_passed"] is False


def test_injection_training_generator_is_fresh_train_only_and_balanced() -> None:
    from neural_foundry.json_proof_loop import public_required_rows

    rows = generate_train_injection_rows(seed=20261111, examples_per_template=16)
    preview = preview_train_injection_rows(seed=20261111, examples_per_template=16)
    repeated = generate_train_injection_rows(seed=20261111, examples_per_template=16)
    alternate_seed = preview_train_injection_rows(seed=20261112, examples_per_template=16)
    required_texts = {row["input"]["text"] for row in public_required_rows(seed=17)}

    assert len(rows) == 64
    assert len({row["scenario_id"] for row in rows}) == 64
    assert len({row["input"]["text"] for row in rows}) == 64
    assert rows == repeated
    assert alternate_seed["rows_digest"] != preview["rows_digest"]
    assert all(row["scenario_id"].startswith("train:") for row in rows)
    assert not {row["input"]["text"] for row in rows} & required_texts
    assert preview["total_examples"] == 64
    assert preview["by_template"] == {name: 16 for name in (
        "packing_slip", "receiving_log", "vendor_bill", "warehouse_ticket")}
    assert preview["by_attack"] == {name: 16 for name in (
        "alternate_record", "extra_property", "quantity_override", "quoted_vendor")}
    assert preview["unique_inputs"] == 64
    assert preview["holdout_accessed"] is False
    assert preview["required_set_used_for_generation"] is False


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
    def forbidden_holdout(*args, **kwargs):
        raise AssertionError("holdout must wait for a complete public validation comparison")
    monkeypatch.setattr(json_proof_workflow, "compare_synthetic_holdout", forbidden_holdout)
    output = tmp_path / "proof"
    report = json_proof_workflow.run_json_proof(output, gpu_python=sys.executable, automodel_python=sys.executable,
                                                 rows_per_template=1, frontier_factory=lambda: _FakeModel(),
                                                 base_failure_report_path=failure_report)
    assert report["complete"] is False
    assert report["public_validation_comparison"]["failure"] == {
        "stage": "base", "reason": "stub model identity is not eligible"
    }
    assert report["comparison"] is None and report["holdout_accessed"] is False
    assert report["scorer_audit"]["passed"] is True
    assert report["scorer_audit"]["cases"] == 3 * (len(TEMPLATES["train"]) + len(TEMPLATES["validation"]))
    assert not list(output.glob("nemo_*/resources_servers/nf_json_extraction/data/holdout.jsonl"))
    assert json.loads((output / "report.json").read_text()) == report
    assert "private-token" not in json.dumps(report)
    assert report["release_authorized"] is False
    second = json_proof_workflow.run_json_proof(tmp_path / "local-proof", gpu_python=sys.executable,
                                                automodel_python=sys.executable, rows_per_template=1,
                                                base_failure_report_path=failure_report)
    assert second["frontier_configured"] is False
    assert second["public_validation_comparison"]["failure"]["stage"] == "base"
    assert second["holdout_accessed"] is False


def test_missing_frontier_fails_before_creating_output(tmp_path, monkeypatch) -> None:
    from neural_foundry.json_frontier_model import JsonFrontierModel
    for name in ("NF_FRONTIER_URL", "NF_FRONTIER_MODEL", "NF_FRONTIER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    output = tmp_path / "proof"
    with pytest.raises(GymJobError, match="credential"):
        json_proof_workflow.run_json_proof(output, gpu_python=sys.executable, automodel_python=sys.executable,
                          frontier_factory=lambda: JsonFrontierModel.from_env(env_file=tmp_path / "missing.env"))
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
    assert report["complete"] is False
    assert report["public_validation_comparison"]["complete"] is False
    assert report["public_validation_comparison"]["failure"]["stage"] == "base"
    assert report["comparison"] is None and not report["holdout_accessed"] and not report["specialist_gate_passed"]
    assert report["preservation"] == {"passed": False} and not report["preservation_gating"]


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
    assert report["complete"] is False
    assert report["public_validation_comparison"]["failure"] == {
        "stage": "frontier", "reason": "stub model identity is not eligible"
    }
    assert report["comparison"] is None and report["holdout_accessed"] is False


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