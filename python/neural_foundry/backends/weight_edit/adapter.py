"""Probe-gated generic weight-edit lifecycle trainer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...execution_runner import run_local_worker
from ...gym_artifacts import CandidateRef, artifact_files, read_candidate_manifest
from ...gym_job import GymJobError, GymJobSpec, is_finite_number
from ...gym_lifecycle import AdapterIdentity, TrainerAdapter
from ...gym_service import _document_digest, _wire_document
from ...provenance import parse_strict_json, sha256_file


@dataclass(frozen=True)
class WeightEditBackendConfig:
    python_executable: str
    compatibility_probe_path: str
    timeout_seconds: float = 900.0
    gpu_memory_utilization: float = 0.25

    def __post_init__(self) -> None:
        for value in (self.python_executable, self.compatibility_probe_path):
            if not isinstance(value, str) or not Path(value).is_absolute() or not Path(value).is_file():
                raise GymJobError("weight edit requires existing absolute runtime and compatibility-probe paths")
        if not is_finite_number(self.timeout_seconds) or not 0 < self.timeout_seconds <= 3600:
            raise GymJobError("weight-edit timeout must be finite and bounded")
        if not is_finite_number(self.gpu_memory_utilization) or not 0 < self.gpu_memory_utilization <= 0.9:
            raise GymJobError("weight-edit serving GPU budget is invalid")

    def document(self) -> dict[str, Any]:
        return vars(self).copy()


def validate_recipe(recipe: Mapping[str, Any]) -> dict[str, Any]:
    fields = {"method", "edit_method", "seed", "target_step", "retention_nullspace_weight", "max_sequence_tokens",
              "max_training_tokens", "max_keys", "preservation_interactions", "preservation_source", "max_preservation_logit_drift"}
    if not isinstance(recipe, Mapping) or set(recipe) != fields or recipe.get("method") != "weight_edit" or recipe.get("edit_method") != "legacy_key_value_v1":
        raise GymJobError("weight-edit recipe has unsupported methods or fields")
    if type(recipe["seed"]) is not int or recipe["seed"] < 0:
        raise GymJobError("weight-edit seed must be a nonnegative integer")
    for name, maximum in (("max_sequence_tokens", 4096), ("max_training_tokens", 1_000_000), ("max_keys", 8192)):
        if type(recipe[name]) is not int or not 1 <= recipe[name] <= maximum:
            raise GymJobError(f"weight-edit {name} is outside its budget")
    if not is_finite_number(recipe["target_step"]) or not 0 < recipe["target_step"] <= 1:
        raise GymJobError("weight-edit target step is invalid")
    if not is_finite_number(recipe["retention_nullspace_weight"]) or recipe["retention_nullspace_weight"] < 0:
        raise GymJobError("weight-edit retention weight is invalid")
    if not is_finite_number(recipe["max_preservation_logit_drift"]) or recipe["max_preservation_logit_drift"] < 0:
        raise GymJobError("weight-edit preservation drift bound is invalid")
    if recipe["preservation_source"] != "public_train":
        raise GymJobError("weight-edit preservation cannot use holdout or selection validation")
    panel = recipe["preservation_interactions"]
    if not isinstance(panel, (list, tuple)) or not 1 <= len(panel) <= 128:
        raise GymJobError("weight-edit preservation panel must be bounded and nonempty")
    return parse_strict_json(_wire_document(recipe))


def weight_edit_trainer(config: WeightEditBackendConfig, job: GymJobSpec) -> TrainerAdapter:
    probe = parse_strict_json(Path(config.compatibility_probe_path).read_bytes())
    if (not isinstance(probe, dict) or probe.get("probe_digest") != _document_digest({key: value for key, value in probe.items() if key != "probe_digest"})
        or probe.get("supported") is not True or probe.get("serving_verified") is not True
        or probe.get("portability_verification", {}).get("verified") is not True):
        raise GymJobError("weight-edit backend requires a passed compatibility probe")
    if "weight_edit" not in {method.method_id for method in job.candidate_methods}:
        raise GymJobError("frozen job has not authorized weight editing")
    worker = Path(__file__).with_name("worker.py")
    identity = AdapterIdentity("weight_edit.generic", "1", _document_digest({
        "adapter": sha256_file(__file__), "probe": probe["probe_digest"], "config": config.document(), "job": job.digest,
        "worker": sha256_file(worker) if worker.is_file() else None,
        "helpers": {name: sha256_file(Path(__file__).parent / name) for name in
                ("apply.py", "capture.py", "demonstrations.py", "loader.py", "numerics.py", "methods/base.py", "serving_probes.py", "near_ties.py")},
    }))

    def preflight(request: Mapping[str, Any], base: CandidateRef) -> dict[str, Any]:
        if request.get("training_request_digest") != _document_digest({key: value for key, value in request.items() if key != "training_request_digest"}):
            raise GymJobError("weight-edit training request digest is invalid")
        if (request.get("job_digest") != job.digest or probe.get("model_id") != base.model_id
            or probe.get("base_digest") != request.get("loaded_base", {}).get("state_digest")
            or probe.get("precision") != request.get("inference_config", {}).get("precision")
            or probe.get("candidate", {}).get("family") != base.family
            or probe.get("portability_calibration", {}).get("renderer_digest") != request.get("inference_config", {}).get("prompt_renderer_digest")
            or probe.get("portability_calibration", {}).get("context_tokens") != request.get("inference_config", {}).get("context_tokens")
            or request.get("base_reference") != vars(base)):
            raise GymJobError("weight-edit request base differs from the qualified frozen model")
        return validate_recipe(request["recipe"])

    def load(request: Mapping[str, Any], base: CandidateRef, directory: Path, *, require_execution: bool = True) -> CandidateRef:
        preflight(request, base)
        receipt = parse_strict_json((directory / "training_receipt.json").read_bytes())
        if (receipt.get("training_request_digest") != request["training_request_digest"]
            or receipt.get("trainer_implementation_digest") != identity.implementation_digest
            or receipt.get("compatibility_probe_digest") != probe["probe_digest"]
            or receipt.get("identity_probes", {}).get("kind") != "nf_vllm_weight_identity_v2"
            or receipt.get("portability_verification", {}).get("verified") is not True):
            raise GymJobError("weight-edit receipt differs from its frozen training request")
            if require_execution and (receipt.get("worker_execution", {}).get("returncode") != 0
                          or receipt.get("worker_execution", {}).get("timed_out") is not False):
                raise GymJobError("weight-edit worker execution receipt is missing or incomplete")
        manifest = read_candidate_manifest(directory, request["training_request_digest"])
        if manifest["kind"] != "weight_edit" or manifest["base_digest"] != base.base_digest:
            raise GymJobError("weight-edit artifact lineage differs from the base")
        candidate = manifest["metadata"]["candidate"]
        digest = _document_digest({"training_request": request["training_request_digest"], "files": artifact_files(directory)})
        if digest != candidate["artifact_digest"]:
            raise GymJobError("weight-edit files differ from the candidate reference")
        return CandidateRef(base.model_id, "weight_edit", base.base_digest, manifest["model_digest"], digest,
                            str(directory), candidate["candidate_id"], "weight_edit", base.family)

    def train(request: Mapping[str, Any], examples: Sequence[Mapping[str, Any]], base: CandidateRef, directory: Path) -> CandidateRef:
        preflight(request, base)
        if _document_digest({"examples": list(examples)}) != request["projected_examples_digest"]:
            raise GymJobError("weight-edit demonstrations differ from the frozen request")
        if not worker.is_file():
            raise GymJobError("weight-edit calculation worker is unavailable")
        payload = {"request": dict(request), "examples": list(examples), "base": vars(base), "job": job.to_document(),
                   "probe": probe, "config": config.document(), "implementation_digest": identity.implementation_digest,
                   "output_dir": str(directory)}
        import tempfile
        import os
        with tempfile.TemporaryDirectory(prefix=".nf-weight-edit-", dir=directory.parent) as temporary:
            path = Path(temporary) / "request.json"
            path.write_bytes(_wire_document(payload))
            environment = {"PATH": os.defpath, "HOME": os.environ.get("HOME", str(Path.home())),
                           "PYTHONPATH": str(Path(__file__).resolve().parents[3]), "HF_HUB_OFFLINE": "1",
                           "TRANSFORMERS_OFFLINE": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "2"}
            result = run_local_worker([config.python_executable, "-m", "neural_foundry.backends.weight_edit.worker", str(path)],
                                      cwd=directory.parent, timeout_seconds=config.timeout_seconds, env=environment,
                                      cleanup_descendants=True)
        if result.returncode != 0 or result.timed_out:
            raise GymJobError(f"weight-edit worker failed; stderr digest {result.receipt['stderr_digest']}")
        output = parse_strict_json(result.stdout)
        previous = load(request, base, directory, require_execution=False)
        if (output.get("kind") != "nf_weight_edit_worker_result_v1"
            or output.get("candidate") != vars(previous)):
            raise GymJobError("worker result differs from the verified staged candidate")
        from .apply import seal_serving_probes
        receipt = parse_strict_json((directory / "training_receipt.json").read_bytes())
        seal_serving_probes(previous, receipt["identity_probes"], receipt["portability_verification"], worker_execution=result.receipt)
        return load(request, base, directory)

    return TrainerAdapter(identity, ("chat_messages_v1", "json_extraction_v1"), ("*",), ("weight_edit",), train, load,
                          {"weight_edit": "weight_edit"})


from .. import BackendFactory, registry

registry.register(BackendFactory("weight_edit.generic", "1", {}, weight_edit_trainer))