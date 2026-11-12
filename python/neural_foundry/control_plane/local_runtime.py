"""Operator-only local AutoModel registration, qualification and host worker.

No uploaded import names, shell commands, model paths or self-asserted readiness
are accepted. The only installed adapter is owned flat JSON + cached Qwen3-0.6B LoRA.
The coordinator uses ``python``; every proof stage, including post-training
materialization, uses ``reload_python``. Only the trainer launches AutoModel.
Version-2 descriptors bind both interpreters' complete environments and source.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from ..provenance import resolve_checkpoint, sha256_file
from .models import ControlPlaneError, digest, document, encode, fail, identifier, keys

if TYPE_CHECKING:
    from ..proof_registry import ProofRegistry
    from .bridges import BridgeContext, TrustedRuntimeProfile
    from .service import ControlPlaneService

MODEL_ID = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MAX_CUDA = 8 * 1024**3
PACKAGES = ("torch", "transformers", "nemo-automodel", "triton", "torchao", "torchdata",
            "datasets", "peft", "psycopg", "safetensors")
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MODULE = "neural_foundry.control_plane.local_runtime"
CAPABILITIES = frozenset({"model_training", "model_export", "model_identity", "model_scoring"})
CONFIG_KEYS = {"version", "id", "name", "python", "reload_python", "root", "model", "limits"}


def source_identity() -> str:
    # Relative shipped Python identities are stable in a wheel and source checkout;
    # native bytes and installed distributions belong to the host descriptor only.
    return digest({path.relative_to(PACKAGE_ROOT).as_posix(): sha256_file(path)
                   for path in sorted(PACKAGE_ROOT.rglob("*.py"))})


def environment_identity() -> dict[str, Any]:
    from importlib.metadata import distributions
    return {"python": platform.python_version(), "executable": os.path.abspath(sys.executable),
            "executable_digest": sha256_file(sys.executable),
            "platform": sys.platform, "machine": platform.machine(),
            "distributions": sorted({(d.metadata["Name"], d.version) for d in distributions()
                                     if d.metadata.get("Name")}),
            "native": {p.name: sha256_file(p) for p in sorted([
                *PACKAGE_ROOT.glob("_native*.so"), *PACKAGE_ROOT.glob("_native*.pyd")])}}


def runtime_metadata() -> dict[str, Any]:
    """CPU-only identity of one interpreter and its complete installed environment."""
    return {"python": list(sys.version_info[:3]),
            "versions": {name: version(name) for name in ("torch", "transformers", "peft")},
            "executable_digest": sha256_file(sys.executable),
            "environment": environment_identity(), "source": source_identity()}


def read_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError):
        fail("runtime_config_error", "Cannot read the operator runtime configuration.", 503)
    if isinstance(value, dict):
        value.setdefault("reload_python", "/home/ot/.venvs/nf-peft-qlora/bin/python")
    keys(value, CONFIG_KEYS | {"descriptor", "qualification"}, CONFIG_KEYS)
    identifier(value["id"], "runtime profile")
    if (value["version"] != 1 or not isinstance(value["name"], str)
            or not 1 <= len(value["name"]) <= 128):
        fail("runtime_config_error", "Unsupported runtime configuration.")
    for field in ("root", "python", "reload_python"):
        if not isinstance(value[field], str) or not value[field].startswith("/"):
            fail("runtime_config_error", "Runtime paths must be operator-owned absolute Linux paths.")
    if value["model"] != {"model_id": MODEL_ID, "revision": REVISION}:
        fail("runtime_config_error", "Only the approved cached Qwen3-0.6B revision is supported.")
    caps = value["limits"]
    maxima = {"steps": 2, "wall_seconds": 600, "cuda_allocated_bytes": MAX_CUDA,
              "validation_examples": 8, "context_tokens": 512, "max_new_tokens": 64}
    keys(caps, set(maxima), set(maxima))
    if any(type(caps[k]) is not int or not 1 <= caps[k] <= high for k, high in maxima.items()):
        fail("runtime_config_error", "Runtime limits exceed the explicitly approved qualification scope.")
    if caps["context_tokens"] <= caps["max_new_tokens"]:
        fail("runtime_config_error", "Generation must fit inside the bounded model context.")
    return cast(dict[str, Any], value)


def config_body(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: config[key] for key in CONFIG_KEYS}


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encode(value))
        stream.flush()
        os.fsync(stream.fileno())


def preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    """Inspect installed metadata and cached bytes, without importing Torch/models."""
    reasons: list[str] = []
    if sys.platform != "linux" or os.path.abspath(sys.executable) != config["python"]:
        reasons.append("Run this command inside WSL using the exact configured interpreter.")
    packages: dict[str, str] = {}
    for name in PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            reasons.append(f"Required installed package missing: {name}.")
    from ..backends.automodel.config import AUTOMODEL_PYTHON, AUTOMODEL_VERSIONS
    for name, expected in {**AUTOMODEL_VERSIONS, "peft": "0.18.0", "psycopg": "3.3.8"}.items():
        if name in packages and packages[name] != expected:
            reasons.append(f"Installed {name} differs from the selected runtime ({expected}).")
    if tuple(sys.version_info[:3]) != AUTOMODEL_PYTHON:
        reasons.append("Python differs from the selected AutoModel runtime (3.13.14).")
    reload_metadata: dict[str, Any] = {}
    try:
        reload_metadata = inspect_reload(config["reload_python"])
        from ..backends.automodel.config import PEFT_RELOAD_PYTHON, PEFT_RELOAD_VERSIONS
        if (reload_metadata["python"] != list(PEFT_RELOAD_PYTHON)
                or reload_metadata["versions"] != PEFT_RELOAD_VERSIONS):
            reasons.append("PEFT reload interpreter does not match the backend's pinned runtime.")
        if reload_metadata["source"] != source_identity():
            reasons.append("PEFT orchestration interpreter resolves different project source code.")
        if reload_metadata["environment"]["executable"] != config["reload_python"]:
            reasons.append("PEFT orchestration metadata resolves a different interpreter executable.")
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        reasons.append("Cannot inspect the existing pinned PEFT reload interpreter.")
    model: dict[str, Any] = dict(config["model"])
    try:
        checkpoint = resolve_checkpoint(MODEL_ID, revision=REVISION, allow_download=False)
        config_json = json.loads((checkpoint.path / "config.json").read_bytes())
        if config_json.get("model_type") != "qwen3" or config_json.get("auto_map"):
            reasons.append("Cached checkpoint is not the supported non-remote-code Qwen3 architecture.")
        if not any(name.endswith(".safetensors") for name in checkpoint.file_digests):
            reasons.append("Cached safetensors weights are missing.")
        model.update({"weights_digest": digest(checkpoint.file_digests),
                      "files": checkpoint.file_digests})
    except (OSError, ValueError, RuntimeError) as exc:
        reasons.append("Approved cached checkpoint is unavailable: " + type(exc).__name__ + ". No downloads allowed.")
    gpu: dict[str, Any] = {}
    try:
        gpu = inspect_gpu()
        if "5090" not in gpu["name"] or gpu["free_mib"] * 1024**2 < config["limits"]["cuda_allocated_bytes"]:
            reasons.append("The approved RTX 5090 or required free CUDA budget is unavailable.")
    except (OSError, ValueError, subprocess.SubprocessError):
        reasons.append("Cannot inspect the local NVIDIA GPU with nvidia-smi.")
    descriptor = {"version": 2, "adapter": "owned-json-automodel-lora-v2",
                  "execution_interpreters": {"coordinator": config["python"],
                                             "trainer": config["python"],
                                             "orchestrator": config["reload_python"]},
                  "config_digest": digest(config_body(config)), "source": source_identity(),
                  "environment": digest(environment_identity()), "model": model,
                  "reload_environment": digest(reload_metadata),
                  "limits": config["limits"], "methods": ["lora"], "gpu": gpu}
    return {"ready": not reasons, "reason": "; ".join(reasons) if reasons else "Preflight passed; not yet qualified.",
            "descriptor": descriptor, "packages": packages, "qualified": False}


def inspect_reload(interpreter: str) -> dict[str, Any]:
    result = subprocess.run([interpreter, "-c",
        ("import json;"
         "from neural_foundry.control_plane.local_runtime import runtime_metadata;"
         "print(json.dumps(runtime_metadata(),sort_keys=True))")],
        capture_output=True, text=True, timeout=30, check=True)
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise TypeError("reload interpreter metadata is not an object")
    return value


def inspect_gpu() -> dict[str, Any]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,uuid",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15, check=True)
    rows = [r.split(", ") for r in result.stdout.strip().splitlines()]
    if len(rows) != 1 or len(rows[0]) != 4:
        raise ValueError("exactly one NVIDIA GPU is required")
    name, total, free, uuid = rows[0]
    return {"name": name, "total_mib": int(total), "free_mib": int(free), "uuid": uuid}


def registry() -> tuple[ProofRegistry, dict[str, dict[str, Any]]]:
    from .. import proof_plugins
    from ..gym_lifecycle import AdapterIdentity
    from ..proof_registry import ProofRegistry
    from . import owned_json
    result = ProofRegistry()
    references = {}
    factories: dict[str, Callable[..., Any]] = {
        "gym": owned_json.owned_task, "model": proof_plugins.training_model,
        "trainer": proof_plugins.automodel_trainer, "training_audit": proof_plugins.automodel_coverage,
        "exporter": owned_json.local_export, "augmenter": owned_json.no_augmentation,
        "public_audit": owned_json.owned_audit, "server": owned_json.LocalExecution,
        "evidence": owned_json.local_evidence,
    }
    source = source_identity()
    for kind, factory in factories.items():
        references[kind] = result.register(kind, AdapterIdentity("owned-json-" + kind, "1",
            digest({"kind": kind, "source": source})), factory)
    return result, references


def compile_proof(config: Mapping[str, Any], descriptor: Mapping[str, Any], *,
                  model: Mapping[str, Any], dataset: dict[str, Any],
                  recipe: Mapping[str, Any], binding: Mapping[str, Any]) -> dict[str, Any]:
    from ..gym_service import _document_digest
    from .owned_json import output_schema
    from .product import validate_dataset
    validate_dataset(dataset)
    declared = {k: model[k] for k in ("model_id", "revision")}
    if declared != config["model"] or (
            "weights_digest" in model and model["weights_digest"] != descriptor["model"]["weights_digest"]):
        fail("runtime_model_mismatch", "The model is not the exact cached qualified checkpoint.", 409)
    if recipe["method"] != "lora":
        fail("runtime_method_mismatch", "Only LoRA is qualified for this runtime.", 409)
    if recipe.get("runtime_profile", config["id"]) != config["id"]:
        fail("runtime_binding_changed", "Saved recipe selected a different runtime profile.", 409)
    caps = config["limits"]
    train = [r for r in dataset["examples"] if r["split"] == "train"]
    validation = [r for r in dataset["examples"] if r["split"] == "validation"]
    if not 1 <= len(train) <= caps["steps"] or not 1 <= len(validation) <= caps["validation_examples"]:
        fail("runtime_budget_exceeded", "This profile requires one complete pass over at most two train rows and bounded validation.", 409)
    parameters = {"learning_rate": 0.0001, "epochs": 1, "steps": len(train), "rank": 4,
                  "alpha": 8, "batch_size": 1, "seed": 17}
    requested = recipe["parameters"]
    if set(requested) - set(parameters):
        fail("runtime_recipe_mismatch", "Unsupported LoRA parameter.")
    parameters.update(requested)
    if (parameters["steps"] != len(train) or parameters["epochs"] != 1
            or parameters["batch_size"] != 1 or parameters["rank"] != 4
            or parameters["alpha"] != 8 or parameters["learning_rate"] != 0.0001
            or parameters["seed"] != 17):
        fail("runtime_recipe_mismatch", "Only the exact qualified bounded LoRA recipe is installed.", 409)
    _, refs = registry()

    def rows(source: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
        return [{"scenario_id": split + ":" + r["id"], "input": r["input"],
                 "reference": r["expected"], "item_id": r["id"], "family_id": "owned-json"}
                for r in source]

    training, public = rows(train, "train"), rows(validation, "validation")
    role = recipe.get("role", "candidate")
    identifier(role, "candidate role")
    policy = {"version": 1, "context_tokens": caps["context_tokens"],
              "max_new_tokens": caps["max_new_tokens"], "report_only": True, "adaptive_retraining": False,
              "identity": {"version": 1, "probe_count": 16, "logprob_atol": 1e-6},
              "api": {"version": 1, "maximum_api_attempts": 1, "maximum_cost_microusd": 1,
                      "calls_authorized": False}}
    backend_recipe = {"method": "lora", "seed": parameters["seed"], "steps": len(train),
                      "learning_rate": parameters["learning_rate"], "rank": 4,
                      "target_modules": ["q_proj", "v_proj"], "max_sequence_tokens": caps["context_tokens"],
                      "tokens_per_batch": caps["context_tokens"],
                      "max_dataset_tokens": len(train) * caps["context_tokens"]}
    compiled = cast(dict[str, Any], document({
        "kind": "nf_public_proof_run", "version": 1, "run_id": "owned-" + digest(binding)[7:31],
        "base_model": declared, "plugins": {k: v for k, v in refs.items() if k not in {"server", "evidence"}},
        "download_allowed": False, "holdout_accessed": False, "coverage_policy": "complete_ordered_pass",
        "policy": policy, "gym": {"output_schema": output_schema(dataset["examples"])},
        "training_rows": training, "training_rows_digest": _document_digest({"rows": training}),
        "public_sets": [{"name": "owned-validation", "rows": public,
                         "rows_digest": _document_digest({"rows": public})}],
        "candidates": [{"role": role, "method": "lora", "recipe": backend_recipe,
                        "execution": {"kind": "server", "plugin": refs["server"], "configuration": {}},
                        "evidence": refs["evidence"]}],
        "gates": [{"name": "owned-exact", "set": "owned-validation", "metric": "exact_match",
                   "minimum": len(public)}],
        "runtime_binding": {"descriptor_digest": digest(descriptor), "inputs": {
            "model": dict(model), "dataset": dataset, "recipe": dict(recipe), "binding": dict(binding)}},
        "product_binding": dict(binding),
    }))
    from ..proof_pipeline import validate_spec
    validate_spec(compiled, registry()[0])
    return compiled


def stage_limits(config: Mapping[str, Any], proof: Mapping[str, Any], operation: str) -> dict[str, int]:
    count = len(proof["public_sets"][0]["rows"]) if operation == "proof.score" else (
        len(proof["training_rows"]) if operation == "proof.train" else 1)
    # Each operation is separately approved; actual elapsed time is enforced by subprocess.
    return {"steps": count, "episodes": 1, "wall_seconds": config["limits"]["wall_seconds"],
            "recorded_bytes": 262144}


def validate_compiled_proof(profile: TrustedRuntimeProfile, proof: dict[str, Any],
                            limits: dict[str, int]) -> None:
    config = read_config(Path(profile.operator_config))
    inputs = proof.get("runtime_binding", {}).get("inputs", {})
    if (not profile.descriptor or digest(config.get("descriptor")) != digest(profile.descriptor)
            or proof != compile_proof(config, profile.descriptor, **inputs)):
        fail("runtime_binding_changed", "Compiled proof differs from its immutable assets or installed runtime.", 409)
    if not any(limits == stage_limits(config, proof, op) for op in
               ("proof.train", "proof.export", "proof.identity", "proof.score")):
        fail("runtime_budget_exceeded", "Dispatch limits differ from the exact scoped runtime limits.", 409)


def validate_owned_assets(service: ControlPlaneService, proof: dict[str, Any]) -> None:
    """Re-resolve pins at dispatch; an uploaded coherent JSON document is not custody."""
    from .product import ProductWorkspace
    workspace = ProductWorkspace(service)
    binding = proof.get("product_binding", {})
    inputs = proof.get("runtime_binding", {}).get("inputs", {})
    pins = binding.get("assets", [])
    if (binding.get("qualification_only") or not isinstance(pins, list)
            or len(pins) < 4 or len({p["id"] for p in pins}) != len(pins)):
        fail("runtime_asset_pins_changed", "Live dispatch requires exact saved project asset pins.", 409)
    assets: dict[str, list[dict[str, Any]]] = {}
    with service.store.transaction() as conn:
        for pin in pins:
            asset = workspace._asset(conn, binding["project_id"], pin["id"], pin["kind"])
            if {k: asset[k] for k in ("id", "kind", "version", "artifact_id")} != pin:
                fail("runtime_asset_pins_changed", "A saved project asset pin differs.", 409)
            assets.setdefault(asset["kind"], []).append(asset)
    if (set(assets) != {"recipe", "model", "dataset", "gym"}
            or any(len(assets[k]) != 1 for k in ("recipe", "model", "dataset"))):
        fail("runtime_asset_pins_changed", "Compiled proof asset roster differs.", 409)
    for kind in ("recipe", "model", "dataset"):
        if assets[kind][0]["document"] != inputs.get(kind):
            fail("runtime_asset_pins_changed", "Compiled proof input differs from its stored asset.", 409)
    recipe = assets["recipe"][0]["document"]
    dataset_id = recipe.get("dataset_asset_id") or assets["gym"][0]["document"].get("dataset_asset_id")
    if (recipe["model_asset_id"] != assets["model"][0]["id"]
            or dataset_id != assets["dataset"][0]["id"]
            or set(recipe["gym_asset_ids"]) != {g["id"] for g in assets["gym"]}
            or any(g["document"] != {"template": "json_exact", "config": {},
                   "dataset_asset_id": dataset_id} for g in assets["gym"])):
        fail("runtime_asset_pins_changed", "Recipe model, dataset or gym identities differ.", 409)


def worker_eligible(profile: TrustedRuntimeProfile) -> bool:
    return not profile.descriptor or (
        sys.platform == "linux" and os.path.abspath(sys.executable) == profile.automodel_python)


def worker_can_claim(profiles: Mapping[str, TrustedRuntimeProfile], plan: Mapping[str, Any]) -> bool:
    scoped = plan.get("runtime_implementations", {})
    if scoped:
        return (all(name in profiles and worker_eligible(profiles[name]) for name in scoped)
                and all(s["operation"].startswith("proof.")
                        and s["config"]["runtime_profile"] in scoped for s in plan["stages"]))
    return not any(p.descriptor and worker_eligible(p) for p in profiles.values())


def verify_host(config: Mapping[str, Any], *,
                role: Literal["coordinator", "orchestrator"] = "coordinator") -> None:
    descriptor = config["descriptor"]
    expected_python = config["python"] if role == "coordinator" else config["reload_python"]
    if os.path.abspath(sys.executable) != expected_python:
        fail("runtime_environment_changed", "Operation is running in the wrong pinned interpreter.", 409)
    training = runtime_metadata() if role == "coordinator" else inspect_reload(config["python"])
    orchestration = inspect_reload(config["reload_python"]) if role == "coordinator" else runtime_metadata()
    if (descriptor.get("version") != 2
            or descriptor.get("execution_interpreters") != {
                "coordinator": config["python"], "trainer": config["python"],
                "orchestrator": config["reload_python"]}
            or training["source"] != descriptor["source"]
            or orchestration["source"] != descriptor["source"]
            or digest(training["environment"]) != descriptor["environment"]
            or digest(orchestration) != descriptor["reload_environment"]
            or digest(config_body(config)) != descriptor["config_digest"]):
        fail("runtime_environment_changed", "Host source, interpreter, packages or configuration changed.", 409)
    gpu = inspect_gpu()
    if (any(gpu[k] != descriptor["gpu"][k] for k in ("name", "total_mib", "uuid"))
            or gpu["free_mib"] * 1024**2 < config["limits"]["cuda_allocated_bytes"]):
        fail("runtime_gpu_changed", "Qualified GPU identity or available CUDA budget differs.", 409)
    checkpoint = resolve_checkpoint(MODEL_ID, revision=REVISION, allow_download=False)
    if (checkpoint.file_digests != descriptor["model"]["files"]
            or digest(checkpoint.file_digests) != descriptor["model"]["weights_digest"]):
        fail("runtime_model_changed", "Cached model bytes changed after qualification.", 409)


def _profile(config: dict[str, Any], path: Path) -> TrustedRuntimeProfile:
    from .bridges import TrustedRuntimeProfile
    return TrustedRuntimeProfile(config["id"], Path(config["root"]), qualified=True,
        deployment_capabilities=CAPABILITIES, automodel_python=config["python"],
        reload_python=config["reload_python"], registry=registry()[0], descriptor=config["descriptor"],
        operator_config=str(path.resolve()),
        callbacks={op: _dispatch_local for op in ("proof.train", "proof.export", "proof.identity", "proof.score")})


def install_configured_runtime(service: ControlPlaneService) -> None:
    path = os.environ.get("CP_RUNTIME_CONFIG_FILE")
    if not path:
        return
    config = read_config(Path(path))
    descriptor, qualification = config.get("descriptor"), config.get("qualification")
    if (not isinstance(descriptor, dict) or not isinstance(qualification, dict)
            or qualification.get("complete") is not True
            or qualification.get("descriptor_digest") != digest(descriptor)
            or qualification.get("config_digest") != digest(config_body(config))
            or qualification.get("receipt_digest") != digest(
                {k: v for k, v in qualification.items() if k != "receipt_digest"})
            or descriptor.get("source") != source_identity()
            or descriptor.get("version") != 2
            or descriptor.get("config_digest") != digest(config_body(config))):
        fail("runtime_not_qualified", "Operator manifest lacks a completed, current qualification receipt.", 503)
    service.install_runtime_profile(_profile(config, Path(path)))


def runtime_descriptor(profile: TrustedRuntimeProfile) -> dict[str, Any]:
    if not profile.descriptor:
        return {"id": profile.id, "name": profile.id, "methods": [], "models": [],
                "ready": False, "reason": "Legacy profile requires explicit saved proof/gym bindings.",
                "limits": {}, "qualification": None}
    config = read_config(Path(profile.operator_config))
    heartbeat = Path(profile.operator_config).with_name("worker-heartbeat.json")
    ready = False
    try:
        status = json.loads(heartbeat.read_bytes())
        ready = (status["descriptor_digest"] == digest(profile.descriptor)
                 and status["runtime_profile"] == profile.id and status["state"] == "ready"
                 and 0 <= time.time() - status["updated_at"] <= 30)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return {"id": profile.id, "name": config["name"], "methods": ["lora"],
            "models": [{k: profile.descriptor["model"][k] for k in ("model_id", "revision", "weights_digest")}],
            "ready": ready, "reason": ("Scoped local runtime qualified; model quality is not certified." if ready
                                      else "Qualified profile installed; waiting for a current host-worker heartbeat."),
            "limits": dict(config["limits"]), "qualification": config["qualification"],
            "recipe_defaults": {"learning_rate": 0.0001, "epochs": 1, "rank": 4,
                                "alpha": 8, "batch_size": 1, "seed": 17}}


def write_worker_heartbeat(path: Path, config: Mapping[str, Any], worker_id: str, state: str) -> None:
    value = {"runtime_profile": config["id"], "descriptor_digest": digest(config["descriptor"]),
             "worker_id": worker_id, "pid": os.getpid(), "state": state, "updated_at": time.time()}
    scratch = path.with_name(".worker-heartbeat-" + str(os.getpid()))
    try:
        _write_once(scratch, value)
        # Non-secret status is read by the unprivileged coordinator through a read-only mount.
        os.chmod(scratch, 0o644)
        os.replace(scratch, path)
    finally:
        scratch.unlink(missing_ok=True)


def _dispatch_local(context: BridgeContext) -> dict[str, Any]:
    profile, proof, root = context["profile"], context["document"], context["root"]
    config = read_config(Path(profile.operator_config))
    verify_host(config)
    validate_compiled_proof(profile, proof, context["limits"])
    # The operation is set by dispatch, not read from user-controlled proof contents.
    operation = context["operation"]
    if context["limits"] != stage_limits(config, proof, operation):
        fail("runtime_budget_exceeded", "Operation limits differ from exact approved limits.", 409)
    root.mkdir(parents=True, exist_ok=True)
    return _run_child(config, Path(profile.operator_config), root, proof, operation,
                      context["limits"]["wall_seconds"])


def _run_child(config: Mapping[str, Any], config_path: Path, root: Path,
               proof: Mapping[str, Any], operation: str, wall: int) -> dict[str, Any]:
    from ..execution_runner import run_local_worker
    stage = operation.removeprefix("proof.")
    job = root / (stage + "-dispatch")
    job.mkdir(exist_ok=False)
    _write_once(job / "input.json", proof)
    inherited = {"HOME", "PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "HF_HOME",
                 "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "PYTHONPATH", "OMP_NUM_THREADS",
                 "MKL_NUM_THREADS", "LANG", "LC_ALL"}
    env = {k: v for k, v in os.environ.items() if k in inherited}
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false"})
    result = run_local_worker(
        [config["reload_python"], "-m", MODULE, "_execute", "--config", str(config_path),
         "--root", str(root), "--operation", operation],
        cwd=Path.cwd(), timeout_seconds=wall, env=env, cleanup_descendants=True)
    _write_once(job / "process-receipt.json", result.receipt)
    for name, content in (("stdout.log", result.stdout), ("stderr.log", result.stderr)):
        fd = os.open(job / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
    if result.timed_out or result.returncode:
        fail("local_runtime_failed", "Bounded local operation failed; preserved attempt requires operator attention.", 409)
    try:
        value = json.loads((job / "output.json").read_bytes())
    except (OSError, ValueError):
        fail("local_runtime_failed", "Local operation completed without a verified output receipt.", 409)
    return dict(value)


def _execute(config: dict[str, Any], root: Path, operation: str) -> None:
    """Internal subprocess entrypoint. No import/string execution from documents."""
    verify_host(config, role="orchestrator")
    import torch

    from .. import proof_pipeline
    from .owned_json import LocalExecution, local_evidence, local_export
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(config["limits"]["cuda_allocated_bytes"] / total, 0)
    torch.cuda.reset_peak_memory_stats(0)
    signal.signal(signal.SIGTERM, lambda *_args: (_ for _ in ()).throw(KeyboardInterrupt()))
    stage = operation.removeprefix("proof.")
    job = root / (stage + "-dispatch")
    proof = json.loads((job / "input.json").read_bytes())
    inputs = proof["runtime_binding"]["inputs"]
    if proof != compile_proof(config, config["descriptor"], **inputs):
        fail("runtime_binding_changed", "Internal proof specification changed.", 409)
    selected = proof["candidates"][0]
    role = selected["role"]
    plugins, _ = registry()
    proof_pipeline.validate_spec(proof, plugins)
    if operation == "proof.train":
        result = proof_pipeline.train(proof, role=role, root=root, automodel_python=config["python"],
            reload_python=config["reload_python"], registry=plugins, trusted_training_limits={
                "base_staging_device": "cpu", "training_device": "cuda:0",
                "worker_timeout_seconds": max(1, config["limits"]["wall_seconds"] - 30),
                "maximum_cuda_allocated_bytes": config["limits"]["cuda_allocated_bytes"]})
        if not result["coverage"]["complete"]:
            fail("runtime_coverage_failed", "Training did not consume every frozen row.", 409)
    elif operation == "proof.export":
        result = local_export(spec=proof, role=role, root=root)
    elif operation == "proof.identity":
        with LocalExecution(spec=proof, candidate=selected, root=root) as model:
            local_evidence(model=model, policy=proof["policy"])
            result = {"loaded_identity": model.loaded_identity(), "verified": True}
    elif operation == "proof.score":
        result = proof_pipeline.score(proof, role=role, root=root, registry=plugins)
        if not result["complete"] or not result["gates_complete"]:
            fail("runtime_score_incomplete", "Original owned validation scoring did not complete.", 409)
    else:
        fail("runtime_operation_refused", "Operation is not part of this installed adapter.")
    peak = torch.cuda.max_memory_allocated(0)
    if peak > config["limits"]["cuda_allocated_bytes"]:
        fail("runtime_budget_exceeded", "CUDA allocation ceiling exceeded.", 409)
    _write_once(job / "output.json", {**result, "steps": stage_limits(config, proof, operation)["steps"],
        "cuda_peak_allocated_bytes": peak, "model_calls": True,
        "quality": "not_evaluated", "release_authorized": False})


def qualify(config: dict[str, Any], path: Path, *, steps: int, cuda: int, wall: int) -> dict[str, Any]:
    if steps != 2 or cuda != MAX_CUDA or wall != 600 or (
            config["limits"]["steps"] != steps or config["limits"]["cuda_allocated_bytes"] != cuda
            or config["limits"]["wall_seconds"] != wall):
        fail("qualification_budget_refused", "Explicit qualification flags must match the approved 2-step/8-GiB/600-second budget.")
    checked = preflight(config)
    if not checked["ready"]:
        fail("runtime_preflight_failed", checked["reason"], 409)
    root = Path(config["root"])
    root.mkdir(parents=True, exist_ok=True)
    # A preserved exclusive claim prevents any automatic or accidental second GPU attempt.
    _write_once(root / "qualification.started.json", {
        "config_digest": digest(config_body(config)), "descriptor": checked["descriptor"],
        "started_at": time.time(), "automatic_retry_authorized": False})
    prepared = {**config_body(config), "descriptor": checked["descriptor"]}
    internal = root / "qualification-config.json"
    _write_once(internal, prepared)
    dataset = {"format": "json_examples_v1", "examples": [
        {"id": "owned-train-a", "split": "train", "input": "Classify the sky as blue.", "expected": {"label": "blue"}},
        {"id": "owned-train-b", "split": "train", "input": "Classify fresh grass as green.", "expected": {"label": "green"}},
        {"id": "owned-validation", "split": "validation", "input": "Classify a ripe lemon as yellow.", "expected": {"label": "yellow"}},
    ]}
    recipe = {"method": "lora", "parameters": {}}
    binding = {"project_id": "operator-qualification", "method": "lora", "role": "candidate",
               "assets": [], "qualification_only": True}
    proof = compile_proof(prepared, checked["descriptor"], model=config["model"],
                          dataset=dataset, recipe=recipe, binding=binding)
    _write_once(root / "qualification-proof.json", proof)
    started = time.monotonic()
    results: dict[str, Any] = {}
    try:
        work = root / "qualification"
        work.mkdir(exist_ok=False)
        for stage in ("train", "export", "identity", "score"):
            remaining = int(wall - (time.monotonic() - started))
            if remaining <= 0:
                fail("qualification_timeout", "Qualification wall budget exhausted.", 409)
            result = _run_child(prepared, internal, work, proof, "proof." + stage, remaining)
            results[stage] = {"digest": digest(result), "steps": result["steps"]}
        # Completion is a runtime conformance claim, not a validation accuracy claim.
        receipt = {"version": 1, "complete": True, "descriptor_digest": digest(checked["descriptor"]),
                   "config_digest": digest(config_body(config)), "proof_digest": digest(proof),
                   "results": results, "duration_seconds": time.monotonic() - started,
                   "model_quality_claim": False, "holdout_accessed": False, "release_authorized": False}
        receipt["receipt_digest"] = digest(receipt)
        if digest(config_body(read_config(path))) != digest(config_body(config)):
            fail("runtime_config_changed", "Operator configuration changed during qualification.", 409)
        _write_once(root / "qualification.receipt.json", receipt)
        manifest = {**prepared, "qualification": receipt}
        _write_once(root / "runtime.json", manifest)
        promoted = path.with_name(path.name + ".qualified")
        _write_once(promoted, manifest)
        os.replace(promoted, path)
        return receipt
    except BaseException as exc:
        _write_once(root / "qualification.failed.json", {
            "complete": False, "failure_type": type(exc).__name__, "completed_stages": list(results),
            "duration_seconds": time.monotonic() - started, "automatic_retry_authorized": False})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "qualify", "worker", "_execute"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--max-cuda-bytes", type=int)
    parser.add_argument("--wall-seconds", type=int)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--operation", choices=("proof.train", "proof.export", "proof.identity", "proof.score"))
    args = parser.parse_args()
    try:
        config = read_config(args.config)
        if args.mode == "preflight":
            result = preflight(config)
            print(json.dumps(result, sort_keys=True))
            return 0 if result["ready"] else 1
        if args.mode == "qualify":
            if args.steps is None or args.max_cuda_bytes is None or args.wall_seconds is None:
                parser.error("qualify requires --steps, --max-cuda-bytes and --wall-seconds")
            print(json.dumps(qualify(config, args.config, steps=args.steps,
                                     cuda=args.max_cuda_bytes, wall=args.wall_seconds)))
        elif args.mode == "_execute":
            if args.root is None or args.operation is None:
                parser.error("internal execution requires --root and --operation")
            _execute(config, args.root, args.operation)
        else:
            import fcntl

            from .artifacts import HTTPArtifactStore
            from .service import ControlPlaneService
            os.environ["CP_RUNTIME_CONFIG_FILE"] = str(args.config.resolve())
            verify_host(config)
            token_file = os.environ.get("CP_OPERATOR_TOKEN_FILE")
            token = Path(token_file).read_text().strip() if token_file else os.environ.get("CP_OPERATOR_TOKEN", "")
            artifacts = HTTPArtifactStore(os.environ.get("CP_ARTIFACT_API_URL", "http://127.0.0.1:8787"), token)
            service = ControlPlaneService(os.environ.get("CP_DATABASE_URL", ""), Path(config["root"]),
                                          artifact_store=artifacts)
            service.health()
            stopping = threading.Event()
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, lambda *_args: stopping.set())
            worker_id = os.environ.get("CP_WORKER_ID", "wsl-automodel-worker")
            lock_fd = os.open(Path(config["root"]) / "worker.lock", os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                os.close(lock_fd)
                raise
            heartbeat_path = args.config.resolve().with_name("worker-heartbeat.json")
            heartbeat_stopped = threading.Event()

            def heartbeat() -> None:
                while not heartbeat_stopped.wait(5):
                    try:
                        write_worker_heartbeat(heartbeat_path, config, worker_id, "ready")
                    except OSError:
                        stopping.set()
                        return

            write_worker_heartbeat(heartbeat_path, config, worker_id, "ready")
            keeper = threading.Thread(target=heartbeat, name="local-runtime-heartbeat", daemon=True)
            keeper.start()
            print(json.dumps({"worker": "ready", "runtime_profile": config["id"],
                              "worker_id": worker_id}), flush=True)
            try:
                while not stopping.is_set():
                    if service.worker_tick(worker_id) is None:
                        stopping.wait(1)
            finally:
                heartbeat_stopped.set()
                keeper.join(timeout=6)
                try:
                    write_worker_heartbeat(heartbeat_path, config, worker_id, "stopped")
                finally:
                    try:
                        service.abandon_owned(worker_id)
                    finally:
                        os.close(lock_fd)
        return 0
    except (ControlPlaneError, OSError, ValueError) as exc:
        print(json.dumps({"error": exc.code if isinstance(exc, ControlPlaneError) else type(exc).__name__,
                          "message": exc.message if isinstance(exc, ControlPlaneError) else
                          "Operator runtime operation failed; inspect preserved local receipts."}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
