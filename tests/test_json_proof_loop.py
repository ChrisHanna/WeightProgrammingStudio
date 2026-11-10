from __future__ import annotations

from contextlib import contextmanager

import pytest

from neural_foundry import json_proof_loop
from neural_foundry.gym_job import GymJobError
from neural_foundry.gym_service import ExecutionResponse
from neural_foundry.reference_json_extraction import generate_scenarios, reference_task


def _factory(responses, *, fail=False, identity=None):
    @contextmanager
    def factory():
        class Model:
            def loaded_identity(self):
                return identity if identity is not None else {
                    "test_stub": True, "served_model_identity_verified": False
                }

            def infer(self, interaction):
                if fail:
                    raise RuntimeError("private details must not leave the evaluator")
                return ExecutionResponse(responses.pop(0), {})

        yield Model()
    return factory


def _reported_identity(model_id: str) -> dict[str, object]:
    return {"provider": "openai", "provider_reported_model": model_id,
            "provider_host": "api.openai.com", "served_model_identity_verified": False}


@pytest.fixture
def scoring_only(monkeypatch):
    monkeypatch.setattr(json_proof_loop, "_require_model_evidence", lambda *args: None)
    monkeypatch.setattr(json_proof_loop, "_require_response_evidence", lambda *args: None)


def test_generation_that_hits_token_cap_is_a_failure_even_when_answer_matches() -> None:
    task = reference_task(rows_per_template=1)
    row = generate_scenarios(split="validation", seed=17, parameters={"rows_per_template": 1})[0]

    class Model:
        def loaded_identity(self):
            return {"served_model_identity_verified": False}

        def infer(self, interaction):
            return ExecutionResponse(row["reference"], {"hit_generation_cap": True})

    score = json_proof_loop.score_public_rows(task, Model(), [row])
    assert score["exact_match"] == 0.0
    assert score["format_valid"] == 0.0
    assert score["responses"][0]["hit_generation_cap"] is True


def test_synthetic_holdout_compares_three_models_with_common_scorer(scoring_only) -> None:
    task = reference_task(rows_per_template=1)
    rows = generate_scenarios(split="holdout", seed=31, parameters={"rows_per_template": 1})
    correct = [row["reference"] for row in rows]
    models = {
        "base": _factory([None, None], identity=_reported_identity("base")),
        "sft": _factory(list(correct), identity=_reported_identity("sft")),
        "frontier": _factory([correct[0], None], identity=_reported_identity("frontier")),
    }
    report = json_proof_loop.compare_synthetic_holdout(task, seed=31, models=models)
    assert report["complete"] is True
    assert [report["scorecards"][role]["value"] for role in ("base", "sft", "frontier")] == [0.0, 1.0, 0.5]
    assert report["evidence_class"] == "synthetic" and report["release_authorized"] is False
    assert report["independent_custody_verified"] is False and report["served_model_identity_verified"] is False
    assert "reference" not in str(report) and "responses" not in report


def test_public_validation_allows_base_sft_frontier_without_weight_edit(scoring_only) -> None:
    task = reference_task(rows_per_template=1)
    rows = generate_scenarios(split="validation", seed=31, parameters={"rows_per_template": 1})
    required = json_proof_loop.public_required_rows(seed=31)
    correct = [row["reference"] for row in [*rows, *required]]
    models = {role: _factory(list(correct), identity=_reported_identity(role))
              for role in ("base", "sft", "frontier")}

    report = json_proof_loop.compare_public_validation(task, seed=31, models=models)

    assert report["complete"] is True
    assert set(report["scorecards"]) == {"base", "sft", "frontier"}
    assert report["holdout_accessed"] is False
    assert report["release_authorized"] is False


def test_validation_and_holdout_refuse_labeled_stub_before_its_inference() -> None:
    task = reference_task(rows_per_template=1)
    validation = generate_scenarios(split="validation", seed=31, parameters={"rows_per_template": 1})
    required = json_proof_loop.public_required_rows(seed=31)
    holdout = generate_scenarios(split="holdout", seed=31, parameters={"rows_per_template": 1})
    calls = []

    @contextmanager
    def stub_frontier():
        class Model:
            def loaded_identity(self):
                return {"provider": "test", "model_id": "labelled-test-stub"}

            def infer(self, interaction):
                calls.append(interaction)
                return None

        yield Model()

    public_models = {
        "base": stub_frontier,
        "sft": _factory([row["reference"] for row in [*validation, *required]], identity=_reported_identity("sft")),
        "frontier": stub_frontier,
    }
    public_report = json_proof_loop.compare_public_validation(task, seed=31, models=public_models)
    assert public_report["complete"] is False
    assert public_report["failure"] == {"stage": "base", "reason": "stub model identity is not eligible"}

    holdout_models = {
        "base": stub_frontier,
        "sft": _factory([row["reference"] for row in holdout], identity=_reported_identity("sft")),
        "frontier": stub_frontier,
    }
    holdout_report = json_proof_loop.compare_synthetic_holdout(task, seed=31, models=holdout_models)
    assert holdout_report["complete"] is False
    assert holdout_report["failure"] == {"stage": "base", "reason": "GymJobError"}
    assert calls == []


def test_incomplete_comparison_never_scores_partial_results(monkeypatch, scoring_only) -> None:
    task = reference_task(rows_per_template=1)
    models = {
        "base": _factory([None, None], identity=_reported_identity("base")),
        "sft": _factory([None, None], identity=_reported_identity("sft")),
        "frontier": _factory([], fail=True, identity=_reported_identity("frontier")),
    }
    monkeypatch.setattr(json_proof_loop, "score_response", lambda **kwargs: (_ for _ in ()).throw(AssertionError("scored")))
    report = json_proof_loop.compare_synthetic_holdout(task, seed=31, models=models)
    assert report["complete"] is False and report["scorecards"] is None
    assert report["failure"] == {"stage": "frontier", "reason": "RuntimeError"}
    missing = json_proof_loop.compare_synthetic_holdout(task, seed=31, models={"base": models["base"]})
    assert missing["complete"] is False and missing["scorecards"] is None


def test_weight_edit_arm_uses_common_holdout_and_reports_preservation_regressions(scoring_only) -> None:
    task = reference_task(rows_per_template=1)
    rows = generate_scenarios(split="holdout", seed=31, parameters={"rows_per_template": 1})
    preservation = generate_scenarios(split="train", seed=32, parameters={"rows_per_template": 1})[:1]
    correct = [row["reference"] for row in rows]
    preserved = preservation[0]["reference"]
    models = {
        "base": _factory([None, None, preserved], identity=_reported_identity("base")),
        "sft": _factory([*correct, preserved], identity=_reported_identity("sft")),
        "weight_edit": _factory([*correct, None], identity=_reported_identity("weight-edit")),
    }
    report = json_proof_loop.compare_synthetic_holdout(task, seed=31, models=models, preservation_rows=preservation)
    assert report["complete"] and report["preservation_count"] == 1
    assert report["scorecards"]["weight_edit"]["value"] == 1.0
    assert report["scorecards"]["weight_edit"]["newly_broken_cases"] == 1
    assert report["scorecards"]["sft"]["newly_broken_cases"] == 0


@pytest.mark.parametrize("identity", [{}, _reported_identity("gpt-4o-mini-2024-07-18"),
    {"served_model_identity_verified": True, "model_digest": "sha256:" + "a" * 64}])
def test_unlabelled_hardcoded_responder_is_refused_without_inference(identity) -> None:
    task = reference_task(rows_per_template=1)
    calls = []

    @contextmanager
    def factory():
        class Hardcoded:
            def loaded_identity(self):
                return identity

            def infer(self, interaction):
                calls.append(interaction)
                return {}

        for role in ("base", "frontier"):
            with pytest.raises(GymJobError, match="requires"):
                json_proof_loop._require_model_evidence(Hardcoded(), role)
        yield Hardcoded()

    models = {role: factory for role in ("base", "sft", "frontier")}
    for compare in (json_proof_loop.compare_public_validation, json_proof_loop.compare_synthetic_holdout):
        report = compare(task, seed=31, models=models)
        assert report["complete"] is False and report["scorecards"] is None
        assert report["failure"]["stage"] == "base"
    assert calls == []


def test_local_evidence_binds_verified_probe_to_candidate_digest() -> None:
    from dataclasses import replace
    from neural_foundry.gym_artifacts import CandidateRef
    from neural_foundry.gym_lifecycle import AdapterIdentity, ModelAdapter
    from neural_foundry.gym_service import CandidateIdentity, _document_digest
    from neural_foundry.backends.weight_edit.serving_probes import verify_identity

    digest = "sha256:" + "a" * 64
    record = {"kind": "nf_vllm_weight_identity_v2", "model_digest": digest,
              "renderer_digest": digest, "image": digest, "context_tokens": 512, "precision": "float32",
              "weight_files": {"model.safetensors": digest}, "records": [
                  {"prompt_token_ids": [index], "greedy_token_id": 1, "greedy_logprob": -0.5}
                  for index in range(16)]}
    evidence = {**verify_identity(record, record), "model_digest": digest,
                "expected_record": record, "observed_record": record}
    evidence["evidence_digest"] = _document_digest(evidence)
    identity = {"served_model_identity_verified": True, "model_digest": digest, "identity_probe_evidence": evidence}
    reference = CandidateRef("test/model", "full", digest, digest, digest, "/test-only", "candidate", "sft", "qwen3")
    model = ModelAdapter(AdapterIdentity("vllm_container", "1", digest),
        CandidateIdentity("candidate", "sft", digest, digest, digest), "qwen3", ("json_extraction_v1",), {},
        lambda interaction: {}, lambda: identity, reference=reference)
    json_proof_loop._require_model_evidence(model, "sft")
    foreign = replace(model, reference=replace(reference, model_digest="sha256:" + "b" * 64))
    with pytest.raises(GymJobError, match="weights evidence"):
        json_proof_loop._require_model_evidence(foreign, "sft")
    evidence["observed_record"] = {**record, "model_digest": "sha256:" + "b" * 64}
    evidence["evidence_digest"] = _document_digest({key: value for key, value in evidence.items() if key != "evidence_digest"})
    with pytest.raises(GymJobError, match="foreign"):
        json_proof_loop._require_model_evidence(model, "sft")