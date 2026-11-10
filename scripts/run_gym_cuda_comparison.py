"""Pinned local CUDA inference and killed-worker retry evidence, not release qualification."""

from __future__ import annotations

import argparse
import json
import os
import selectors
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from neural_foundry.gpu import select_training_compute
from neural_foundry.gym_job import (
    AcceptanceGate,
    CandidateMethod,
    ComponentReference,
    GymComponentRegistry,
    GymJobSpec,
    ScenarioSource,
)
from neural_foundry.gym_service import CandidateIdentity, GymStudioService
from neural_foundry.provenance import CheckpointUnavailableError, resolve_checkpoint, sha256_file
from neural_foundry.pythia import PythiaModel
from neural_foundry.records import canonicalize, content_digest

CHECKPOINTS = (
    ("EleutherAI/pythia-14m", "cf967c0a9a04383db6f7b1108d86b2962634b4ac"),
    ("EleutherAI/pythia-31m", "e556ace21b489575e94e9d50b6dad2fcc7419679"),
)
PANELS = {
    "train": (("The license renews each year.", "renewal"),),
    "validation": (
        ("This contract renews automatically every year.", "renewal"),
        ("Either party may terminate this contract with thirty days notice.", "termination"),
        ("Invoices are payable within sixty days.", "none"),
        ("The agreement renews for an additional twelve months.", "renewal"),
    ),
    "holdout": (("The supplier may end the agreement with written notice.", "termination"),),
}
LABELS = {"A": "renewal", "B": "termination", "C": "none"}


def _digest(document: Any) -> str:
    if not isinstance(document, dict):
        document = {"value": document}
    return str(content_digest(canonicalize(document)))


def _checkpoint_path(model_id: str, revision: str) -> Path:
    return Path.home() / ".cache" / "neural-foundry" / "checkpoints" / f"{model_id.replace('/', '--')}--{revision}"


def _write(path: Path, document: Any) -> None:
    with path.open("wb") as stream:
        stream.write(canonicalize(document))
        stream.flush()
        os.fsync(stream.fileno())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _supervise_partial(command: list[str], output: Path, timeout: float, *, marker: bytes = b"NF_CUDA_PARTIAL_COMMITTED") -> int:
    error_path = output / "partial.stderr.log"
    with error_path.open("wb") as errors:
        worker = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors)
        selector = selectors.DefaultSelector()
        try:
            if worker.stdout is None:
                raise RuntimeError("worker stdout is unavailable")
            selector.register(worker.stdout, selectors.EVENT_READ)
            deadline = time.monotonic() + timeout
            pending = bytearray()
            while b"\n" not in pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise RuntimeError("CUDA worker startup/progress deadline expired")
                data = os.read(worker.stdout.fileno(), 4096)
                if not data or len(pending) + len(data) > 8192:
                    raise RuntimeError(f"CUDA worker failed before its progress marker; see {error_path}")
                pending.extend(data)
            _require(pending.split(b"\n", 1)[0] == marker, "unexpected worker progress marker")
        finally:
            selector.close()
            if worker.poll() is None:
                worker.kill()
            worker.communicate(timeout=30)
    return worker.returncode


def _job() -> tuple[GymJobSpec, GymComponentRegistry]:
    implementation = sha256_file(__file__)
    registry = GymComponentRegistry()

    def provide(*, split: str, **kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "scenario_id": f"contract-{split}-{index}",
                "input": {"contract_text": text},
                "reference": {"clause_type": expected},
            }
            for index, (text, expected) in enumerate(PANELS[split])
        ]

    def verify(*, reference: dict[str, str], response: Any, output_valid: bool, **kwargs: Any) -> dict[str, bool]:
        return {"passed": output_valid and response["clause_type"] == reference["clause_type"]}

    def score(*, verification: dict[str, bool], **kwargs: Any) -> dict[str, float]:
        return {"accuracy": float(verification["passed"])}

    for kind, name, function in (
        ("scenario_provider", "public-contract-panel", provide),
        ("verifier", "exact-clause-class", verify),
        ("scorer", "classification-accuracy", score),
        ("candidate_method", "unchanged_base", lambda **kwargs: None),
    ):
        registry.register(kind, name, "1.0.0", implementation, function)
    spec = GymJobSpec(
        job_id="public-contract-cuda-comparison",
        title="Classify contract clauses",
        description="Classify a public synthetic contract sentence using constrained next-token choices.",
        task_input_schema={
            "type": "object", "properties": {"contract_text": {"type": "string", "maxLength": 1024}},
            "required": ["contract_text"], "additionalProperties": False,
        },
        task_output_schema={
            "type": "object", "properties": {"clause_type": {"type": "string", "enum": list(LABELS.values())}},
            "required": ["clause_type"], "additionalProperties": False,
        },
        scenario_sources=tuple(
            ScenarioSource(
                f"contracts-{split}", "public-contract-panel", "1.0.0", implementation,
                _digest(rows), "simulated", (split,), {"split": split},
            )
            for split, rows in PANELS.items()
        ),
        scorer=ComponentReference("classification-accuracy", "1.0.0", implementation, {}),
        verifier=ComponentReference("exact-clause-class", "1.0.0", implementation, {}),
        candidate_methods=(CandidateMethod("unchanged_base", "1.0.0", implementation, {}),),
        gates=(
            AcceptanceGate("task", "task", "accuracy", "validation", "ge", 0.9),
            AcceptanceGate("behavior", "behavior", "accuracy", "holdout", "ge", 0.9),
            AcceptanceGate("controls", "controls", "beats_control", "controls", "eq", 1.0),
            AcceptanceGate("reproducibility", "reproducibility", "repeat", "reproducibility", "eq", 1.0),
        ),
        metadata={"evidence_scope": "local_inference_only", "release_authorized": False},
    )
    return spec, registry


def _worker(output: Path, phase: str) -> None:
    placement = select_training_compute("cuda:0")
    job, registry = _job()
    service = GymStudioService(registry, state_dir=output / "state")
    panel = service.materialize_public_scenarios(job, split="validation", seed=42)
    candidates = []
    loads: list[dict[str, Any]] = []
    dispatches: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    def bind(model_id: str, revision: str) -> None:
        checkpoint = resolve_checkpoint(_checkpoint_path(model_id, revision), revision=revision)
        checkpoint_digest = _digest({"model_id": model_id, "revision": revision, "files": checkpoint.file_digests})
        candidate = CandidateIdentity(
            model_id.rsplit("/", 1)[-1], "unchanged_base", checkpoint_digest,
            checkpoint_digest, checkpoint_digest,
        )
        candidates.append(candidate)
        model: PythiaModel | None = None

        def execute(*, candidate: dict[str, Any], scenarios: tuple[dict[str, Any], ...]) -> dict[str, Any]:
            nonlocal model
            if model is None:
                if resolve_checkpoint(checkpoint.path, revision=revision).file_digests != checkpoint.file_digests:
                    raise RuntimeError("Checkpoint changed before loading")
                model = PythiaModel.from_pretrained_dir(checkpoint.path, model_name=model_id)
                if resolve_checkpoint(checkpoint.path, revision=revision).file_digests != checkpoint.file_digests:
                    raise RuntimeError("Checkpoint changed while loading")
                model.tensors = {
                    name: tensor.to(device=placement.device, dtype=torch.float32)
                    for name, tensor in model.tensors.items()
                }
                if not all(tensor.device.type == "cuda" and tensor.dtype == torch.float32 for tensor in model.tensors.values()):
                    raise RuntimeError("Loaded tensors are not CUDA float32")
                loads.append({
                    "candidate_id": candidate["candidate_id"], "model_id": model_id,
                    "revision": revision, "checkpoint_path": str(checkpoint.path),
                    "checkpoint_digest": checkpoint_digest, "file_digests": checkpoint.file_digests,
                    "tokenizer_digest": checkpoint.file_digests["tokenizer.json"],
                    "device": placement.device, "device_name": torch.cuda.get_device_name(0),
                    "precision": "float32", "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda, "parameter_count": model.total_parameters,
                    "local_loaded_files_verified": True, "decoding": "constrained_next_token_A_B_C",
                })
            requested = [row["attempt_id"] for row in scenarios]
            selected = scenarios[:1] if phase == "partial" and model_id == CHECKPOINTS[1][0] else scenarios
            responses = {}
            inference = []
            choice_ids = {label: model.tokenizer.encode(" " + label) for label in LABELS}
            if any(len(ids) != 1 for ids in choice_ids.values()):
                raise RuntimeError("A choice label is not a single tokenizer token")
            with torch.inference_mode():
                for row in selected:
                    prompt = (
                        "Classify the contract clause.\nA: renewal\nB: termination\nC: neither\n"
                        f"Contract: {row['input']['contract_text']}\nAnswer:"
                    )
                    ids = model.tokenizer.encode(prompt)
                    logits, _ = model.forward_with_mlp_key(ids)
                    torch.cuda.synchronize(0)
                    if logits.device.type != "cuda" or not bool(torch.isfinite(logits).all()):
                        raise RuntimeError("CUDA inference failed")
                    choice_logits = {label: float(logits[-1, tokens[0]].item()) for label, tokens in choice_ids.items()}
                    label = max(choice_logits, key=lambda value: choice_logits[value])
                    responses[row["attempt_id"]] = {"clause_type": LABELS[label]}
                    inference.append({
                        "attempt_id": row["attempt_id"], "prompt_digest": _digest(prompt),
                        "input_token_ids_digest": _digest(ids), "choice_token_ids": choice_ids,
                        "choice_logits": choice_logits, "selected_label": label,
                    })
            dispatches.append({
                "candidate_id": candidate["candidate_id"], "requested_attempt_ids": requested,
                "completed_attempt_ids": list(responses), "inference": inference,
            })
            return responses

        def measured_execute(**kwargs: Any) -> dict[str, Any]:
            try:
                return execute(**kwargs)
            except Exception as exc:
                errors.append({"model_id": model_id, "error_type": type(exc).__name__, "message": str(exc)})
                raise

        service.register_candidate_executor(candidate, executor_digest=sha256_file(__file__), execute=measured_execute)

    for model_id, revision in CHECKPOINTS:
        bind(model_id, revision)
    comparison = service.compare_unchanged_bases(job, candidates, scenario_manifest_digest=panel["manifest_digest"])
    receipt = {
        "kind": "nf_gym_cuda_comparison_evidence_v1", "phase": phase,
        "evidence_grade": "measured_local_model", "scenario_origin": "simulated",
        "loaded_models": loads, "dispatches": dispatches, "errors": errors, "comparison": comparison,
        "authority": {"release_authorized": False, "independent_holdout_custody": False},
    }
    receipt["receipt_digest"] = _digest(receipt)
    _write(output / f"{phase}.json", receipt)
    if errors or comparison["status"] == "failed":
        raise RuntimeError(f"Measured inference failed: {errors}")
    if phase == "partial":
        print("NF_CUDA_PARTIAL_COMMITTED", flush=True)
        sys.stdin.read(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--worker", choices=("partial", "resume", "cached"))
    parser.add_argument("--startup-seconds", type=float, default=300.0)
    args = parser.parse_args()
    if not sys.platform.startswith("linux") or os.name != "posix":
        parser.error("This named CUDA smoke fixture requires Linux/WSL")
    if not 0 < args.startup_seconds < float("inf"):
        parser.error("startup-seconds must be finite and positive")
    output = args.output_dir.resolve()
    if args.worker:
        _worker(output, args.worker)
        return
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Use an empty output directory for a fresh demonstration")
    output.mkdir(parents=True, mode=0o700, exist_ok=True)
    for model_id, revision in CHECKPOINTS:
        try:
            resolve_checkpoint(model_id, revision=revision)
        except CheckpointUnavailableError:
            if not args.allow_download:
                raise
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model_id, revision=revision, token=False,
                allow_patterns=["README.md", "config.json", "model.safetensors", "tokenizer*.json", "special_tokens_map.json"],
            )
        cached = resolve_checkpoint(model_id, revision=revision)
        local = _checkpoint_path(model_id, revision)
        if not local.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(cached.path, local, symlinks=False)
        if resolve_checkpoint(local, revision=revision).file_digests != cached.file_digests:
            raise RuntimeError("Materialized checkpoint files differ from the pinned snapshot")
    command = [sys.executable, str(Path(__file__).resolve()), "--output-dir", str(output), "--worker"]
    killed_returncode = _supervise_partial([*command, "partial"], output, args.startup_seconds)
    _require(killed_returncode == -signal.SIGKILL, f"Expected a killed worker, got {killed_returncode}")
    for phase in ("resume", "cached"):
        subprocess.run([*command, phase], check=True, timeout=180)
    partial, resumed, cached = [json.loads((output / f"{phase}.json").read_bytes()) for phase in ("partial", "resume", "cached")]
    before = partial["comparison"]
    after = resumed["comparison"]
    _require([run["status"] for run in before["runs"]] == ["complete", "incomplete"], "unexpected partial terminal states")
    _require(after["status"] == "complete" and cached["comparison"] == after, "completed retry differs from saved evidence")
    _require(len(resumed["dispatches"]) == 1 and not cached["dispatches"] and not cached["loaded_models"], "completed work was dispatched again")
    missing = set(before["runs"][1]["expected_attempt_ids"]) - {row["attempt_id"] for row in before["runs"][1]["attempts"]}
    _require(set(resumed["dispatches"][0]["requested_attempt_ids"]) == missing, "retry did not dispatch exactly missing attempts")
    _require(resumed["dispatches"][0]["candidate_id"] == before["runs"][1]["candidate"]["candidate_id"], "wrong checkpoint dispatched on retry")
    report = {
        "kind": "nf_gym_cuda_retry_demonstration_v1", "evidence_grade": "measured_local_model",
        "fixture_name": "pythia_constrained_contract_smoke_v1",
        "scenario_manifest_digest": after["scenario_manifest_digest"],
        "initial_terminal_counts": [len(run["attempts"]) for run in before["runs"]],
        "final_terminal_counts": [len(run["attempts"]) for run in after["runs"]],
        "killed_worker_returncode": killed_returncode, "retried_attempt_ids": sorted(missing),
        "completed_retry_dispatched": False, "loaded_models": partial["loaded_models"],
        "phase_receipt_digests": [phase["receipt_digest"] for phase in (partial, resumed, cached)],
        "authority": {"release_authorized": False, "independent_holdout_custody": False, "remote_serving_identity_verified": False},
    }
    report["report_digest"] = _digest(report)
    _write(output / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()