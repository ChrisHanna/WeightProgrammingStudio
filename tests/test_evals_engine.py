"""First-class Evaluation Engine, sealed-answer isolation, and CLI eval commands."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
from neural_foundry.evals import (
    EVAL_TYPES,
    EVALUATOR_KINDS,
    EvaluationError,
    compare_runs,
    import_suite,
    run_eval,
    verify_promotion,
)
from neural_foundry.records import canonicalize

CLI = Path(os.environ.get("NF_CLI", "/tmp/nf-t043-joined/debug/nf"))


def _enterprise_suite_spec() -> dict[str, Any]:
    return {
        "suite_id": "enterprise-reasoning",
        "version": "1.2.0",
        "evaluator_kind": "enterprise_dataset",
        "scorer_id": "enterprise.reasoning.scorer",
        "scorer_version": "2.0.0",
        "eval_types": sorted(EVAL_TYPES),
        "tasks": [
            {
                "task_id": "base-1",
                "group_id": "g-base",
                "eval_type": "baseline",
                "capability": "Enterprise SQL",
                "input": "SELECT count(*) FROM orders",
                "expected": 42,
                "sealed": False,
            },
            {
                "task_id": "dev-1",
                "group_id": "g-dev",
                "eval_type": "development",
                "capability": "Enterprise Reasoning",
                "input": "Q1",
                "expected": "A1",
                "sealed": False,
            },
            {
                "task_id": "gen-1",
                "group_id": "g-gen",
                "eval_type": "generalization",
                "capability": "Unseen Financial Reasoning",
                "input": "H1",
                "expected": "S1",
                "sealed": True,
            },
            {
                "task_id": "ret-1",
                "group_id": "g-ret",
                "eval_type": "retention",
                "capability": "General Coding & Math",
                "input": "R1",
                "expected": "R_OK",
                "sealed": True,
            },
            {
                "task_id": "saf-1",
                "group_id": "g-saf",
                "eval_type": "safety",
                "capability": "Safety Guardrails",
                "input": "S_IN",
                "expected": "REFUSE",
                "sealed": True,
            },
            {
                "task_id": "rob-1",
                "group_id": "g-rob",
                "eval_type": "robustness",
                "capability": "Prompt Perturbations",
                "input": "P_IN",
                "expected": "P_OK",
                "sealed": True,
            },
            {
                "task_id": "eff-1",
                "group_id": "g-eff",
                "eval_type": "efficiency",
                "capability": "Cost vs LoRA",
                "input": "E_IN",
                "expected": "E_OK",
                "sealed": True,
            },
            {
                "task_id": "prod-1",
                "group_id": "g-prod",
                "eval_type": "production",
                "capability": "Production Canary",
                "input": "C_IN",
                "expected": "C_OK",
                "sealed": True,
            },
        ],
    }


def test_evals_engine_supports_all_eight_types_five_evaluators_and_agent_isolation(
    tmp_path: Path,
) -> None:
    assert len(EVAL_TYPES) == 8
    assert len(EVALUATOR_KINDS) == 5

    suite_dir = tmp_path / "enterprise-reasoning"
    suite_dir.mkdir()
    (suite_dir / "suite.json").write_bytes(canonicalize(_enterprise_suite_spec()))
    suite = import_suite(suite_dir)
    assert suite.suite_id == "enterprise-reasoning"
    assert suite.evaluator_kind == "enterprise_dataset"

    baseline_model = {
        "model_id": "qwen2-1.5b-base",
        "eval_type_scores": {
            "baseline": 0.72,
            "development": 0.72,
            "generalization": 0.70,
            "retention": 0.95,
            "safety": 0.98,
            "robustness": 0.76,
            "efficiency": 0.60,
            "production": 0.88,
        },
        "edit_lineage": {
            "program_digest": None,
            "plan_digest": None,
            "changed_parameters": 0,
            "total_parameters": 1_000_000,
            "method": "baseline_unchanged",
        },
    }
    candidate_model = {
        "model_id": "candidate-042",
        "eval_type_scores": {
            "baseline": 0.72,
            "development": 0.94,
            "generalization": 0.89,
            "retention": 0.9424,  # 99.2% relative retention vs 0.95
            "safety": 0.98,
            "robustness": 0.85,
            "efficiency": 0.91,
            "production": 0.92,
        },
        "edit_lineage": {
            "program_digest": "sha256:" + "1" * 64,
            "plan_digest": "sha256:" + "2" * 64,
            "changed_parameters": 800,
            "total_parameters": 1_000_000,
            "method": "direct_tensor_edit",
        },
    }

    # 1. Agent role can inspect development results to improve edits, but NEVER sees sealed evals
    agent_view = run_eval(model=candidate_model, suite=suite, caller_role="agent")
    assert set(agent_view["scores_by_type"]) == {"baseline", "development"}
    assert "generalization" not in agent_view["scores_by_type"]
    assert "retention" not in agent_view["scores_by_type"]
    assert agent_view["sealed_answers_exposed"] is False
    with pytest.raises(EvaluationError) as agent_verify_err:
        verify_promotion(candidate_run=agent_view, policy="production", caller_role="agent")
    assert agent_verify_err.value.code == "AGENT_CONTROL_FORBIDDEN"

    # 2. Independent verifier runs full comparison and promotion verification
    base_run = run_eval(model=baseline_model, suite=suite, caller_role="verifier")
    cand_run = run_eval(
        model=candidate_model,
        suite=suite,
        compare_to=base_run,
        caller_role="verifier",
    )
    receipt = verify_promotion(
        candidate_run=cand_run,
        baseline_run=base_run,
        policy="production",
        caller_role="verifier",
    )
    assert receipt["status"] == "passed"
    assert receipt["promote_eligible"] is True
    assert receipt["independent_verifier"] is True
    assert receipt["sealed_answers_exposed"] is False
    dash = receipt["dashboard"]
    assert dash["target_success_pct"] == 94.0
    assert dash["baseline_success_pct"] == 72.0
    assert dash["skill_retention_pct"] == 99.2
    assert dash["unseen_task_success_pct"] == 89.0
    assert dash["changed_parameters_pct"] == 0.08
    assert "Model Improvement Report: candidate-042 vs qwen2-1.5b-base" in receipt["markdown_dashboard"]

    # 3. Mismatched inference settings or no-op edits are rejected
    bad_settings_model = dict(candidate_model, inference_settings={"temperature": 0.7})
    with pytest.raises(EvaluationError) as settings_err:
        run_eval(model=bad_settings_model, suite=suite, caller_role="verifier")
    assert settings_err.value.code == "INFERENCE_SETTINGS_MISMATCH"

    noop_receipt = verify_promotion(
        candidate_run=compare_runs(base_run, base_run) and dict(base_run, comparison=compare_runs(base_run, base_run)),
        baseline_run=base_run,
        policy="production",
    )
    assert noop_receipt["status"] == "rejected"
    assert noop_receipt["promote_eligible"] is False


def test_evals_cli_import_run_compare_and_verify(tmp_path: Path) -> None:
    assert CLI.is_file(), f"Expected built nf CLI at {CLI}"
    suite_dir = tmp_path / "enterprise-reasoning"
    suite_dir.mkdir()
    (suite_dir / "suite.json").write_bytes(canonicalize(_enterprise_suite_spec()))

    # 1. nf eval import ./evals/enterprise-reasoning
    imp = subprocess.run(
        [str(CLI), "eval", "import", str(suite_dir)],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    imp_res = json.loads(imp.stdout)["result"]
    assert imp_res["suite_id"] == "enterprise-reasoning"

    # 2. nf eval run --model candidate-042 --suite <suite> --compare-to qwen2-1.5b
    run_out = subprocess.run(
        [
            str(CLI),
            "eval",
            "run",
            "--model",
            "candidate-042",
            "--suite",
            str(suite_dir),
            "--compare-to",
            "qwen2-1.5b",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    run_res = json.loads(run_out.stdout)["result"]
    assert run_res["model_id"] == "candidate-042"
    assert run_res["comparison"]["dashboard"]["target_success_pct"] == 94.0

    # 3. nf eval compare qwen2-1.5b candidate-042
    cmp_out = subprocess.run(
        [str(CLI), "eval", "compare", "qwen2-1.5b", "candidate-042"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    cmp_res = json.loads(cmp_out.stdout)["result"]
    assert cmp_res["matched_inference_settings"] is True
    assert cmp_res["dashboard"]["changed_parameters_pct"] == 0.08

    # 4. nf eval verify --model candidate-042 --policy production
    ver_out = subprocess.run(
        [
            str(CLI),
            "eval",
            "verify",
            "--model",
            "candidate-042",
            "--policy",
            "production",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    ver_res = json.loads(ver_out.stdout)["result"]
    assert ver_res["status"] == "passed"
    assert ver_res["promote_eligible"] is True
    assert ver_res["sealed_answers_exposed"] is False
