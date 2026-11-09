"""Compile, train, and compare the synthetic JSON reference task."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

from .execution_runner import run_local_worker
from .gym_artifacts import CandidateRef
from .gym_job import GymJobError
from .gym_job import is_finite_number
from .gym_service import GymStudioService, _document_digest
from .json_frontier_model import JsonFrontierModel
from .json_proof_loop import compare_synthetic_holdout, compare_public_validation
from .json_scorer_audit import audit_reference_scorer
from .nemo_json_compiler import compile_json_gym
from .reference_json_extraction import generate_scenarios, reference_task
from .vllm_candidate_files import verify_vllm_candidate_files
from .vllm_container_adapter import VLLM_IMAGE, VllmCandidateServer


def run_json_proof(
    output_dir: str | Path, *, gpu_python: str, automodel_python: str, seed: int = 17,
    rows_per_template: int = 32, frontier_factory: Callable[[], Any] | None = None,
    compatibility_probe_path: str | Path | None = None, sft_steps: int = 32,
    weight_edit_target_step: float = 0.01,
    preservation_policy_version: int = 2,
    retention_mode: str = "generalist", training_state_dir: str | Path | None = None,
) -> dict[str, Any]:
    if (type(seed) is not int or seed < 0 or type(rows_per_template) is not int
        or not 1 <= rows_per_template <= 64 or type(sft_steps) is not int or not 1 <= sft_steps <= 256):
        raise GymJobError("synthetic proof seed or row count is invalid")
    if not is_finite_number(weight_edit_target_step) or not 0 < weight_edit_target_step <= 1:
        raise GymJobError("weight-edit target step is invalid")
    if type(preservation_policy_version) is not int or preservation_policy_version not in {1, 2}:
        raise GymJobError("preservation policy version must be 1 or 2")
    if retention_mode not in {"generalist", "specialist"} or (retention_mode == "specialist" and preservation_policy_version != 2):
        raise GymJobError("specialist mode requires explicit preservation v2; unknown retention modes are forbidden")
    if training_state_dir is not None and not Path(training_state_dir).is_dir():
        raise GymJobError("reused training state must be an existing directory")
    for interpreter in (gpu_python, automodel_python):
        if not isinstance(interpreter, str) or not Path(interpreter).is_absolute() or not Path(interpreter).is_file():
            raise GymJobError("synthetic proof requires absolute existing GPU and AutoModel interpreters")
    frontier = frontier_factory() if frontier_factory is not None else None
    probe_path = Path(compatibility_probe_path) if compatibility_probe_path is not None else Path(__file__).resolve().parents[2] / "docs" / "evidence" / "weight-edit-qwen3-bf16.json"
    if not probe_path.is_file():
        raise GymJobError("weight-edit compatibility probe is missing")
    task = reference_task(rows_per_template=rows_per_template)
    audit = audit_reference_scorer(task)
    output = Path(output_dir).absolute()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    service = GymStudioService(state_dir=output / "compiler_state")
    task.bind(service)
    compiled = {split: compile_json_gym(task, service, split=split, seed=seed, output_dir=output / f"nemo_{split}")
                for split in ("train", "validation")}
    allowed_env = ("HOME", "PATH", "HF_HOME", "HF_HUB_CACHE", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")
    environment = {name: os.environ[name] for name in allowed_env if name in os.environ}
    environment.update({"PYTHONPATH": str(Path(__file__).parent.parent), "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                        "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    state = Path(training_state_dir).absolute() if training_state_dir is not None else output / "training_state"
    trained = run_local_worker([gpu_python, "-m", "neural_foundry.json_proof_train_worker", str(state),
                                automodel_python, str(seed), str(rows_per_template), str(probe_path), str(sft_steps), str(weight_edit_target_step), str(preservation_policy_version), retention_mode], cwd=output, timeout_seconds=2400,
                               cleanup_descendants=True, env=environment)
    if trained.returncode != 0 or trained.timed_out:
        raise GymJobError(f"synthetic proof training failed; stderr digest {trained.receipt['stderr_digest']}")
    lines = [line for line in trained.stdout.splitlines() if line.startswith(b"NF_JSON_PROOF_TRAINING=")]
    if len(lines) != 1:
        raise GymJobError("synthetic proof training returned no unique receipt")
    try:
        result = json.loads(lines[0].split(b"=", 1)[1])
        base, sft, edited = CandidateRef(**result["base"]), CandidateRef(**result["sft"]), CandidateRef(**result["weight_edit"])
        revision = result["revision"]
    except (ValueError, KeyError, TypeError) as exc:
        raise GymJobError("synthetic proof training receipt is malformed") from exc
    if (result.get("job_digest") != task.job.digest or base.method_id != "unchanged_base" or sft.method_id != "sft"
        or base.model_id != sft.model_id or base.base_digest != sft.base_digest or base.family != sft.family
        or edited.kind != "weight_edit" or edited.method_id != "weight_edit" or edited.model_id != base.model_id
        or edited.base_digest != base.base_digest or edited.family != base.family):
        raise GymJobError("synthetic proof candidate lineage differs from its frozen job")
    verify_vllm_candidate_files(base, revision=revision)
    verify_vllm_candidate_files(sft, revision=revision)
    verify_vllm_candidate_files(edited, revision=revision)
    models = {
        "base": lambda: VllmCandidateServer(base, revision=revision),
        "sft": lambda: VllmCandidateServer(sft, revision=revision),
        "weight_edit": lambda: VllmCandidateServer(edited, revision=revision),
    }
    if frontier is not None:
        models["frontier"] = lambda: frontier
    training_inputs = {_document_digest(row["input"]) for row in generate_scenarios(split="train", seed=seed, parameters={"rows_per_template": rows_per_template})}
    preservation_rows = [row for row in generate_scenarios(split="train", seed=seed + 1, parameters={"rows_per_template": rows_per_template})
                         if _document_digest(row["input"]) not in training_inputs]
    public_comparison = None
    preservation = None
    if retention_mode == "specialist":
        receipt = json.loads((Path(edited.artifact_path) / "training_receipt.json").read_bytes())
        if receipt.get("retention_mode") != "specialist" or result.get("retention_mode") != "specialist":
            raise GymJobError("specialist artifact differs from the frozen retention mode")
        preservation = receipt["preservation_verification"]
        public_comparison = compare_public_validation(task, seed=seed, models=models)
    holdout_allowed = retention_mode == "generalist" or (public_comparison["complete"]
                      and public_comparison["scorecards"]["weight_edit"]["acceptance"]["passed"])
    comparison = compare_synthetic_holdout(task, seed=seed, models=models, preservation_rows=preservation_rows) if holdout_allowed else None
    report = {"kind": "nf_json_proof_workflow_v1", "job_digest": task.job.digest,
              "scorer_audit": audit,
              "public_packages": {split: receipt["receipt_digest"] for split, receipt in compiled.items()},
              "training_request_digest": result["training_request_digest"],
              "candidate_record_digest": result["candidate_record_digest"], "vllm_image": VLLM_IMAGE,
              "weight_edit_training_request_digest": result["weight_edit_training_request_digest"],
              "weight_edit_candidate_record_digest": result["weight_edit_candidate_record_digest"],
              "shared_dataset_digest": result["dataset_digest"], "frontier_configured": frontier is not None,
              "retention_mode": retention_mode, "public_validation_comparison": public_comparison,
              "preservation": preservation, "preservation_gating": retention_mode == "generalist",
              "holdout_accessed": holdout_allowed, "holdout_complete": bool(comparison and comparison["complete"]),
              "generalist_authorized": False,
              "comparison": comparison, "complete": comparison["complete"] if comparison else public_comparison["complete"],
              "specialist_gate_passed": bool(public_comparison and public_comparison["complete"]
                                            and public_comparison["scorecards"]["weight_edit"]["acceptance"]["passed"]),
              "release_authorized": False}
    report["report_digest"] = _document_digest(report)
    descriptor = os.open(output / "report.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the synthetic JSON-extraction proof workflow")
    parser.add_argument("output_dir")
    parser.add_argument("--gpu-python", required=True)
    parser.add_argument("--automodel-python", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--rows-per-template", type=int, default=32)
    parser.add_argument("--sft-steps", type=int, default=32)
    parser.add_argument("--weight-edit-target-step", type=float, default=0.01)
    parser.add_argument("--preservation-policy", type=int, choices=(1, 2), default=2)
    parser.add_argument("--retention-mode", choices=("generalist", "specialist"), default="generalist")
    parser.add_argument("--training-state")
    parser.add_argument("--compatibility-probe")
    parser.add_argument("--frontier", action="store_true")
    args = parser.parse_args()
    report = run_json_proof(args.output_dir, gpu_python=args.gpu_python, automodel_python=args.automodel_python,
                            seed=args.seed, rows_per_template=args.rows_per_template, sft_steps=args.sft_steps,
                            compatibility_probe_path=args.compatibility_probe,
                            weight_edit_target_step=args.weight_edit_target_step,
                            preservation_policy_version=args.preservation_policy,
                            retention_mode=args.retention_mode, training_state_dir=args.training_state,
                            frontier_factory=JsonFrontierModel.from_env if args.frontier else None)
    print(json.dumps({"complete": report["complete"], "report_digest": report["report_digest"]}))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())