"""CPU-only conformance for operator-installed local training runtimes."""

from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.parse import quote

import pytest
from neural_foundry.control_plane import local_runtime as runtime
from neural_foundry.control_plane.artifacts import HTTPArtifactStore
from neural_foundry.control_plane.bridges import dispatch
from neural_foundry.control_plane.models import (
    ControlPlaneError,
    compile_plan,
    digest,
    validate_spec,
)
from neural_foundry.control_plane.owned_json import owned_audit, owned_task
from neural_foundry.control_plane.service import ControlPlaneService


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "source_identity", lambda: digest("shipped-source"))
    monkeypatch.setattr(runtime, "environment_identity", lambda: {"installed": "host"})
    monkeypatch.setattr(runtime, "runtime_metadata", lambda: {
        "source": runtime.source_identity(), "environment": runtime.environment_identity()})
    reload_metadata = {"source": runtime.source_identity(), "environment": {"installed": "reload"}}
    monkeypatch.setattr(runtime, "inspect_reload", lambda _python: reload_metadata)
    gpu = {"name": "RTX 5090", "uuid": "qualified-gpu", "total_mib": 32607, "free_mib": 29631}
    monkeypatch.setattr(runtime, "inspect_gpu", lambda: gpu)
    files = {"config.json": digest("config"), "model.safetensors": digest("weights")}
    monkeypatch.setattr(runtime, "resolve_checkpoint",
                        lambda *_args, **_kwargs: SimpleNamespace(file_digests=files))
    config = {
        "version": 1, "id": "wsl-automodel", "name": "Local AutoModel",
        "python": str(Path(sys.executable).absolute()),
        "reload_python": "/home/ot/.venvs/nf-peft-qlora/bin/python",
        "root": str(tmp_path),
        "model": {"model_id": runtime.MODEL_ID, "revision": runtime.REVISION},
        "limits": {"steps": 2, "wall_seconds": 600, "cuda_allocated_bytes": runtime.MAX_CUDA,
                   "validation_examples": 8, "context_tokens": 512, "max_new_tokens": 64},
    }
    descriptor = {"version": 2, "adapter": "owned-json-automodel-lora-v2",
                  "execution_interpreters": {"coordinator": config["python"],
                                             "trainer": config["python"],
                                             "orchestrator": config["reload_python"]},
                  "config_digest": digest(config), "source": runtime.source_identity(),
                  "environment": digest(runtime.environment_identity()),
                  "reload_environment": digest(runtime.inspect_reload("ignored")),
                  "model": {**config["model"], "files": files, "weights_digest": digest(files)},
                  "gpu": gpu,
                  "limits": config["limits"], "methods": ["lora"]}
    qualification = {"complete": True, "descriptor_digest": digest(descriptor),
                     "config_digest": digest(config)}
    qualification["receipt_digest"] = digest(qualification)
    config.update(descriptor=descriptor, qualification=qualification)
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(config))
    return config, path, runtime._profile(config, path)


def dataset():
    return {"format": "json_examples_v1", "examples": [
        {"id": "a", "split": "train", "input": "Owned first train question.", "expected": {"label": "a"}},
        {"id": "b", "split": "train", "input": "Owned second train question.", "expected": {"label": "b"}},
        {"id": "c", "split": "validation", "input": "Owned unseen validation question.", "expected": {"label": "c"}},
    ]}


def proof(config):
    return runtime.compile_proof(config, config["descriptor"], model=config["model"],
        dataset=dataset(), recipe={"method": "lora", "parameters": {}},
        binding={"project_id": "project", "method": "lora", "role": "candidate",
                 "assets": [{"kind": "gym", "id": "gym", "version": 1, "artifact_id": digest("gym")}]})


def test_compiler_freezes_original_rows_and_real_backend_recipe(configured):
    config, _, _ = configured
    compiled = proof(config)
    from neural_foundry.backends.automodel.config import AutoModelSFTRecipe
    recipe = AutoModelSFTRecipe.from_document(compiled["candidates"][0]["recipe"])
    assert recipe.steps == 2 and recipe.target_modules == ("q_proj", "v_proj")
    assert [r["input"] for r in compiled["training_rows"]] == [
        r["input"] for r in dataset()["examples"] if r["split"] == "train"]
    assert compiled["public_sets"][0]["rows"][0]["input"] == dataset()["examples"][-1]["input"]
    assert owned_audit(spec=compiled)["public_passed"]
    task = owned_task(compiled["gym"])
    assert task.job.task_input_schema["type"] == "string"
    assert task.project("Owned question", {"label": "answer"})["messages"][-1]["role"] == "assistant"
    assert not compiled["policy"]["api"]["calls_authorized"]
    assert compiled["download_allowed"] is False


@pytest.mark.parametrize("change", [
    lambda c, d, r: c["model"].update(revision="changed"),
    lambda c, d, r: r.update(method="sft"),
    lambda c, d, r: r["parameters"].update(steps=1),
    lambda c, d, r: r["parameters"].update(rank=8),
    lambda c, d, r: r["parameters"].update(batch_size=2),
    lambda c, d, r: d["examples"].append(
        {"id": "extra", "split": "train", "input": "Third train row", "expected": {"label": "d"}}),
    lambda c, d, r: d["examples"][0].update(expected={"label": {"nested": True}}),
    lambda c, d, r: d["examples"][-1].update(input="  Owned first train question. "),
])
def test_unsupported_models_methods_parameters_shapes_and_caps_refused(configured, change):
    config, _, _ = configured
    model, data, recipe = deepcopy(config["model"]), dataset(), {"method": "lora", "parameters": {}}
    copied = deepcopy(config)
    copied["model"] = model
    change(copied, data, recipe)
    with pytest.raises((ControlPlaneError, ValueError)):
        runtime.compile_proof(config, config["descriptor"], model=model, dataset=data,
                              recipe=recipe, binding={})


@pytest.mark.parametrize("operation", ["proof.train", "proof.export", "proof.identity", "proof.score"])
def test_actual_authorized_dispatch_reaches_only_fixed_child_runner_sentinel(configured, monkeypatch, operation):
    config, path, profile = configured
    compiled = proof(config)
    calls = []

    def sentinel(command, **kwargs):
        assert command[:4] == [config["reload_python"], "-m", runtime.MODULE, "_execute"]
        assert command[0] != config["python"]
        assert "--config" in command and str(path) in command
        assert kwargs["timeout_seconds"] == 600
        assert "CP_OPERATOR_TOKEN" not in kwargs["env"]
        root = Path(command[command.index("--root") + 1])
        job = root / (operation.removeprefix("proof.") + "-dispatch")
        frozen = json.loads((job / "input.json").read_bytes())
        assert frozen == compiled
        calls.append(command)
        (job / "output.json").write_text(json.dumps({"sentinel": True}))
        return SimpleNamespace(receipt={"returncode": 0}, returncode=0, timed_out=False,
                               stdout=b"", stderr=b"")

    monkeypatch.setenv("CP_OPERATOR_TOKEN", "never-export-this")
    monkeypatch.setattr("neural_foundry.execution_runner.run_local_worker", sentinel)
    result = dispatch(operation, profile, document=compiled, role="candidate", run_id="run",
        plan_digest="approved", approval={"id": "approval", "plan_digest": "approved"},
        limits=runtime.stage_limits(config, compiled, operation))
    assert result["sentinel"] and len(calls) == 1
    # Started operations are never automatically repeated.
    with pytest.raises(FileExistsError):
        dispatch(operation, profile, document=compiled, role="candidate", run_id="run",
            plan_digest="approved", approval={"id": "approval", "plan_digest": "approved"},
            limits=runtime.stage_limits(config, compiled, operation))
    assert len(calls) == 1


def test_generated_proof_reaches_training_after_real_train_only_materialization(configured, monkeypatch, tmp_path):
    from neural_foundry import proof_pipeline
    from neural_foundry.gym_artifacts import CandidateRef
    from neural_foundry.gym_lifecycle import AdapterIdentity, ModelAdapter, TrainerAdapter
    from neural_foundry.gym_service import CandidateIdentity

    config, _, _ = configured
    compiled = proof(config)
    plugins, _ = runtime.registry()
    original = plugins.resolve
    identity = digest("cpu-sentinel")
    candidate = CandidateIdentity("cpu-base", "unchanged_base", identity, identity, identity)
    reference = CandidateRef(runtime.MODEL_ID, "full", identity, identity, identity,
                             str(tmp_path), "cpu-base", "unchanged_base", "qwen3")
    model = ModelAdapter(AdapterIdentity("cpu-base", "1", identity), candidate, "qwen3",
        ("json_extraction_v1",), {"seed": 17, "device": "cpu"}, lambda _: None, dict,
        artifact_kinds=("full", "lora"), reference=reference, load_candidate=lambda _: None)

    class SentinelReached(Exception):
        pass

    def refuse_training(request, examples, base, directory):
        assert base == reference
        assert request["recipe"]["steps"] == 2
        assert len(examples) == 2
        messages = [row["interaction"]["messages"] for row in examples]
        assert [row[-2]["content"] for row in messages] == [r["input"] for r in compiled["training_rows"]]
        assert all("unseen validation" not in str(row) for row in examples)
        raise SentinelReached

    trainer = TrainerAdapter(AdapterIdentity("cpu-sentinel", "1", identity),
        ("json_extraction_v1",), ("qwen3",), ("lora",), refuse_training,
        load=lambda _request, base, _directory: base,
        artifact_kinds={"lora": "lora"})

    def resolve(kind, ref):
        if kind == "model":
            return lambda **_kwargs: model
        if kind == "trainer":
            return lambda **_kwargs: trainer
        return original(kind, ref)

    monkeypatch.setattr(plugins, "resolve", resolve)
    with pytest.raises(SentinelReached):
        proof_pipeline.train(compiled, role="candidate", root=tmp_path / "proof",
                             automodel_python="not-executed", reload_python="not-executed", registry=plugins)


def test_execute_materializes_after_trainer_only_in_verified_orchestration_context(configured, monkeypatch, tmp_path):
    from neural_foundry.gym_artifacts import CandidateRef
    from neural_foundry.gym_lifecycle import AdapterIdentity, ModelAdapter, TrainerAdapter
    from neural_foundry.gym_service import CandidateIdentity

    config, _, _ = configured
    compiled = proof(config)
    plugins, refs = runtime.registry()
    original = plugins.resolve
    identity = digest("cpu-materialize-sentinel")
    candidate = CandidateIdentity("cpu-base", "unchanged_base", identity, identity, identity)
    reference = CandidateRef(runtime.MODEL_ID, "full", identity, identity, identity,
                             str(tmp_path), "cpu-base", "unchanged_base", "qwen3")
    events = []

    class MaterializationReached(Exception):
        pass

    def materialize(_reference):
        assert events == ["orchestrator-verified", "trainer-returned"]
        events.append("materialize")
        raise MaterializationReached

    model = ModelAdapter(AdapterIdentity("cpu-base", "1", identity), candidate, "qwen3",
        ("json_extraction_v1",), {"seed": 17, "device": "cpu"}, lambda _: None, dict,
        artifact_kinds=("full", "lora"), reference=reference, load_candidate=materialize)

    def train(request, examples, base, directory):
        assert request["recipe"]["steps"] == len(examples) == 2
        assert base == reference and events == ["orchestrator-verified"]
        events.append("trainer-returned")
        return CandidateRef(base.model_id, "lora", base.base_digest, identity, identity,
                            str(directory), "cpu-candidate", "lora", "qwen3")

    trainer = TrainerAdapter(AdapterIdentity("cpu-sentinel", "1", identity),
        ("json_extraction_v1",), ("qwen3",), ("lora",), train,
        load=lambda _request, base, _directory: base, artifact_kinds={"lora": "lora"})

    def resolve(kind, ref):
        if kind == "model":
            return lambda **_kwargs: model
        if kind == "trainer":
            def trainer_factory(**kwargs):
                assert kwargs["automodel_python"] == config["python"]
                assert kwargs["reload_python"] == config["reload_python"]
                return trainer
            return trainer_factory
        return original(kind, ref)

    def verified(_config, *, role):
        assert role == "orchestrator"
        events.append("orchestrator-verified")

    monkeypatch.setattr(plugins, "resolve", resolve)
    monkeypatch.setattr(runtime, "registry", lambda: (plugins, refs))
    monkeypatch.setattr(runtime, "verify_host", verified)
    # Guard calls stay on the production _execute path, with no CUDA initialization.
    import torch
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: SimpleNamespace(total_memory=32 * 1024**3))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda *_args: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *_args: None)
    monkeypatch.setattr(runtime.signal, "signal", lambda *_args: None)
    root = tmp_path / "orchestrator"
    job = root / "train-dispatch"
    job.mkdir(parents=True)
    (job / "input.json").write_text(json.dumps(compiled))
    with pytest.raises(MaterializationReached):
        runtime._execute(config, root, "proof.train")
    assert events == ["orchestrator-verified", "trainer-returned", "materialize"]


@pytest.mark.parametrize("mutation", ["code", "environment", "reload_packages", "reload_code",
                                    "model", "proof", "limits", "approval"])
def test_dispatch_refuses_changed_bound_inputs_before_child(configured, monkeypatch, mutation):
    config, _, profile = configured
    compiled = proof(config)
    bound = runtime.stage_limits(config, compiled, "proof.train")
    approval = {"id": "approval", "plan_digest": "approved"}
    runner = Mock(side_effect=AssertionError("GPU job must never execute"))
    monkeypatch.setattr("neural_foundry.execution_runner.run_local_worker", runner)
    if mutation == "code":
        monkeypatch.setattr(runtime, "source_identity", lambda: digest("changed-code"))
    elif mutation == "environment":
        monkeypatch.setattr(runtime, "environment_identity", lambda: {"changed": True})
    elif mutation.startswith("reload_"):
        changed = deepcopy(runtime.inspect_reload(config["reload_python"]))
        changed["environment" if mutation == "reload_packages" else "source"] = {"changed": True}
        monkeypatch.setattr(runtime, "inspect_reload", lambda _python: changed)
    elif mutation == "model":
        monkeypatch.setattr(runtime, "resolve_checkpoint",
                            lambda *_a, **_k: SimpleNamespace(file_digests={"changed": digest("x")}))
    elif mutation == "proof":
        compiled["training_rows"][0]["input"] = "substituted"
    elif mutation == "limits":
        bound["wall_seconds"] = 599
    else:
        approval["plan_digest"] = "different"
    with pytest.raises(ControlPlaneError):
        dispatch("proof.train", profile, document=compiled, role="candidate", run_id="run",
                 plan_digest="approved", approval=approval, limits=bound)
    runner.assert_not_called()


def test_orchestration_checks_own_full_environment_and_training_environment(configured, monkeypatch):
    config, _, _ = configured
    training = runtime.runtime_metadata()
    orchestration = runtime.inspect_reload(config["reload_python"])
    monkeypatch.setattr(runtime.sys, "executable", config["reload_python"])
    monkeypatch.setattr(runtime, "runtime_metadata", lambda: orchestration)
    monkeypatch.setattr(runtime, "inspect_reload", lambda python: training if python == config["python"] else None)
    runtime.verify_host(config, role="orchestrator")
    with pytest.raises(ControlPlaneError, match="wrong pinned interpreter"):
        runtime.verify_host(config)
    changed = deepcopy(training)
    changed["environment"] = {"modified_training_dependency": True}
    monkeypatch.setattr(runtime, "inspect_reload", lambda _python: changed)
    with pytest.raises(ControlPlaneError, match="configuration changed"):
        runtime.verify_host(config, role="orchestrator")


def test_runtime_limits_are_scoped_and_plan_pins_trusted_descriptor(configured):
    config, _, profile = configured
    service = object.__new__(ControlPlaneService)
    service.runtime_profiles = {profile.id: profile}
    compiled = proof(config)
    spec = {"version": 1, "name": "local", "candidates": [{
        "id": "candidate", "model": {"id": runtime.MODEL_ID, "revision": runtime.REVISION}, "method": "lora"}],
        "stages": [{"id": "train", "operation": "proof.train", "depends_on": [],
                    "candidate_id": "candidate", "config": {"runtime_profile": profile.id,
                        "artifact_id": digest(compiled), "role": "candidate"},
                    "limits": runtime.stage_limits(config, compiled, "proof.train")}]}
    validated = validate_spec(spec, service._operations())
    plan = compile_plan(validated, runtime_descriptors={profile.id: dict(profile.descriptor)})
    assert plan["runtime_implementations"] == {profile.id: digest(profile.descriptor)}
    mixed = deepcopy(validated)
    mixed["stages"].append({"id": "cpu", "operation": "software.identity", "depends_on": [],
                            "config": {}, "limits": {}})
    with pytest.raises(ControlPlaneError, match="cannot mix"):
        compile_plan(mixed, runtime_descriptors={profile.id: dict(profile.descriptor)})
    spec["stages"][0]["operation"] = "software.training"
    spec["stages"][0]["config"] = {}
    with pytest.raises(ControlPlaneError):
        validate_spec(spec, service._operations())


def test_worker_placement_skips_other_environment_before_global_guard(configured, monkeypatch):
    _, _, profile = configured
    plan = {"runtime_implementations": {profile.id: digest(profile.descriptor)}, "stages": [
        {"operation": "proof.train", "config": {"runtime_profile": profile.id}}]}
    assert runtime.worker_can_claim({profile.id: profile}, plan)
    assert not runtime.worker_can_claim({}, plan)
    monkeypatch.setattr(runtime.sys, "platform", "docker-cpu")
    assert not runtime.worker_can_claim({profile.id: profile}, plan)
    monkeypatch.setattr(runtime.sys, "platform", "linux")
    assert not runtime.worker_can_claim({profile.id: profile}, {"stages": [{"operation": "software.training"}]})
    assert runtime.worker_can_claim({}, {"stages": [{"operation": "software.training"}]})


def test_ready_requires_current_matching_host_heartbeat(configured):
    config, path, profile = configured
    assert runtime.runtime_descriptor(profile)["ready"] is False
    heartbeat = path.with_name("worker-heartbeat.json")
    runtime.write_worker_heartbeat(heartbeat, config, "test-worker", "ready")
    assert runtime.runtime_descriptor(profile)["ready"] is True
    value = json.loads(heartbeat.read_bytes())
    value["updated_at"] = 0
    heartbeat.write_text(json.dumps(value))
    assert runtime.runtime_descriptor(profile)["ready"] is False


def test_installed_config_only_and_manifest_tamper_refusal(configured, monkeypatch):
    config, path, profile = configured
    monkeypatch.setenv("CP_RUNTIME_CONFIG_FILE", str(path))
    service = SimpleNamespace(install_runtime_profile=Mock())
    runtime.install_configured_runtime(service)
    assert service.install_runtime_profile.call_args.args[0].id == profile.id
    config["qualification"]["complete"] = False
    path.write_text(json.dumps(config))
    with pytest.raises(ControlPlaneError, match="qualification"):
        runtime.install_configured_runtime(service)


@pytest.mark.parametrize("explicit_dataset", [True, False])
def test_dispatch_rechecks_actual_immutable_asset_records(configured, monkeypatch, explicit_dataset):
    from neural_foundry.control_plane.product import ProductWorkspace
    config, _, _ = configured
    documents = {
        "model": config["model"], "dataset": dataset(),
        "gym": {"template": "json_exact", "config": {}, "dataset_asset_id": "dataset"},
        "recipe": {"method": "lora", "parameters": {}, "model_asset_id": "model",
                   "dataset_asset_id": "dataset", "gym_asset_ids": ["gym"], "runtime_profile": config["id"]},
    }
    if not explicit_dataset:
        documents["recipe"].pop("dataset_asset_id")
    assets = {kind: {"id": kind, "kind": kind, "version": 1,
                    "artifact_id": digest(value), "document": value} for kind, value in documents.items()}
    binding = {"project_id": "project", "role": "candidate", "method": "lora",
               "assets": [{k: asset[k] for k in ("id", "kind", "version", "artifact_id")}
                          for asset in assets.values()]}
    compiled = runtime.compile_proof(config, config["descriptor"], model=documents["model"],
        dataset=documents["dataset"], recipe=documents["recipe"], binding=binding)
    service = SimpleNamespace(store=SimpleNamespace(transaction=lambda: nullcontext(None)))
    monkeypatch.setattr(ProductWorkspace, "_asset",
                        lambda _self, _conn, _project, asset_id, _kind: assets[asset_id])
    runtime.validate_owned_assets(service, compiled)
    assets["gym"]["version"] = 2
    with pytest.raises(ControlPlaneError, match="asset pin differs"):
        runtime.validate_owned_assets(service, compiled)


def test_preflight_is_metadata_only_and_missing_cache_is_precise(configured, monkeypatch, tmp_path):
    from neural_foundry.backends.automodel.config import (
        AUTOMODEL_VERSIONS,
        PEFT_RELOAD_PYTHON,
        PEFT_RELOAD_VERSIONS,
    )
    config, _, _ = configured
    installed = {**AUTOMODEL_VERSIONS, "peft": "0.18.0", "psycopg": "3.3.8", "safetensors": "0.6.2"}
    monkeypatch.setattr(runtime, "version", installed.__getitem__)
    monkeypatch.setattr(runtime.sys, "version_info", (3, 13, 14))
    monkeypatch.setattr(runtime, "inspect_reload", lambda _python: {
        "python": list(PEFT_RELOAD_PYTHON), "versions": PEFT_RELOAD_VERSIONS,
        "source": runtime.source_identity(), "environment": {"executable": config["reload_python"]}})
    checkpoint_dir = tmp_path / "cache"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_text('{"model_type":"qwen3"}')
    monkeypatch.setattr(runtime, "resolve_checkpoint", lambda *_a, **_k: SimpleNamespace(
        path=checkpoint_dir, file_digests=config["descriptor"]["model"]["files"]))
    assert runtime.preflight(config)["ready"]

    def missing(*_args, **_kwargs):
        raise RuntimeError("no cache")

    monkeypatch.setattr(runtime, "resolve_checkpoint", missing)
    result = runtime.preflight(config)
    assert not result["ready"] and "No downloads allowed" in result["reason"]


def test_qualification_requires_explicit_budget_without_starting_any_process(configured, monkeypatch):
    config, _, _ = configured
    runner = Mock(side_effect=AssertionError("No GPU qualification in CPU tests"))
    monkeypatch.setattr(runtime, "preflight", runner)
    with pytest.raises(ControlPlaneError):
        runtime.qualify(config, Path("ignored"), steps=3, cuda=runtime.MAX_CUDA, wall=600)
    runner.assert_not_called()


def test_remote_artifact_integrity_and_publication_readback(monkeypatch):
    value = {"owned": "data"}
    expected = digest(value)
    store = HTTPArtifactStore("http://127.0.0.1:8787", "runtime-only-secret")
    calls = []

    def request(path, payload=None):
        calls.append((path, payload))
        return {"document": value} if path else {"id": expected, "digest": expected}

    monkeypatch.setattr(store, "_request", request)
    assert store.put(value)["id"] == expected
    assert calls == [("", value), ("/" + quote(expected, safe=""), None)]
    monkeypatch.setattr(store, "_request", lambda *_args: {"document": {"substituted": True}})
    with pytest.raises(ControlPlaneError, match="digest differs"):
        store.read(expected)


def test_remote_artifact_auth_errors_and_redirects_do_not_leak_token(monkeypatch):
    from urllib.error import HTTPError

    from neural_foundry.control_plane import artifacts
    store = HTTPArtifactStore("http://127.0.0.1:8787", "runtime-only-secret")

    def open_request(request, **_kwargs):
        assert request.get_header("Authorization") == "Bearer runtime-only-secret"
        raise HTTPError(request.full_url, 401, "secret-server-message", {}, None)

    monkeypatch.setattr(artifacts, "build_opener", lambda *_args: SimpleNamespace(open=open_request))
    with pytest.raises(ControlPlaneError) as error:
        store.read(digest({}))
    assert "secret" not in str(error.value)
    assert artifacts._NoRedirect().redirect_request(None) is None
    with pytest.raises(ControlPlaneError):
        HTTPArtifactStore("http://untrusted.example", "secret")
    with pytest.raises(ControlPlaneError):
        store.read("../credential-file")
