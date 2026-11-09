from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
import torch

from neural_foundry.backends.weight_edit.adapter import WeightEditBackendConfig, validate_recipe, weight_edit_trainer
from neural_foundry.gym_job import GymJobError
from neural_foundry.gym_service import _document_digest
from neural_foundry.reference_json_extraction import reference_task
from neural_foundry.backends.weight_edit.demonstrations import prepare_demonstration
from neural_foundry.prompt_rendering import PromptRenderer


def test_backend_refuses_unqualified_models_before_dispatch(tmp_path):
    probe = {"kind": "nf_weight_edit_compatibility_v1", "supported": False}
    probe["probe_digest"] = _document_digest(probe)
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(probe))
    config = WeightEditBackendConfig(sys.executable, str(path))
    with pytest.raises(GymJobError, match="passed compatibility probe"):
        weight_edit_trainer(config, reference_task().job)


def test_generic_weight_edit_selector_is_registered():
    from neural_foundry.backends import registry
    factory = registry.require("weight_edit.generic@1")
    assert factory.create is weight_edit_trainer


def test_recipe_requires_public_preservation_and_explicit_budgets():
    recipe = {"method": "weight_edit", "edit_method": "legacy_key_value_v1", "seed": 17, "target_step": 0.01,
              "retention_nullspace_weight": 20.0, "max_sequence_tokens": 512, "max_training_tokens": 16384,
              "max_keys": 4096, "preservation_source": "public_train", "max_preservation_logit_drift": 0.25,
              "preservation_interactions": [{"format": "chat_messages_v1", "messages": [{"role": "user", "content": "retain"}]}]}
    assert validate_recipe(recipe) == recipe
    for change in ({"preservation_source": "holdout"}, {"target_step": float("nan")}, {"max_keys": True},
                   {"preservation_interactions": []}, {"edit_method": "unknown"}):
        with pytest.raises(GymJobError):
            validate_recipe({**recipe, **change})
    assert validate_recipe({**recipe, "preservation_policy_version": 2})["preservation_policy_version"] == 2
    with pytest.raises(GymJobError, match="frozen"):
        validate_recipe({**recipe, "preservation_policy_version": 2, "max_preservation_logit_drift": 1.0})
    with pytest.raises(GymJobError, match="version"):
        validate_recipe({**recipe, "preservation_policy_version": 3})
    specialist = {**recipe, "preservation_policy_version": 2, "retention_mode": "specialist"}
    assert validate_recipe(specialist) == specialist
    with pytest.raises(GymJobError, match="explicit"):
        validate_recipe({**recipe, "retention_mode": "specialist"})
    with pytest.raises(GymJobError, match="retention_mode"):
        validate_recipe({**recipe, "retention_mode": "skip_all_checks"})


def test_teacher_forcing_preserves_served_prompt_tokens_and_shifts_targets():
    tokenizer = SimpleNamespace(chat_template="template", special_tokens_map={}, get_vocab=lambda: {"token": 1},
                                 eos_token_id=9, encode=lambda text, **kwargs: [1, 2] if text == "prompt" else [3, 4],
                                 apply_chat_template=lambda messages, **kwargs: "prompt")
    renderer = PromptRenderer(tokenizer)
    interaction = {"format": "chat_messages_v1", "messages": [{"role": "user", "content": "question"},
                                                                  {"role": "assistant", "content": "answer"}]}
    demonstration = prepare_demonstration(renderer, interaction)
    assert demonstration.rendered.token_ids == (1, 2, 3, 4, 9)
    assert demonstration.labels == (-100, -100, 3, 4, 9)
    assert demonstration.positions == (1, 2, 3)
    assert demonstration.rendered.renderer_digest == renderer.digest


@pytest.fixture
def numerical_fixture(tmp_path):
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from neural_foundry.backends.weight_edit.loader import load_model
    from neural_foundry.backends.weight_edit.worker import calculate_edit
    from neural_foundry.reference_json_extraction import generate_scenarios

    config = transformers.Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                     num_attention_heads=2, num_key_value_heads=1, head_dim=12, max_position_embeddings=128)
    model = transformers.AutoModelForCausalLM.from_config(config).eval()
    model.save_pretrained(tmp_path / "base", safe_serialization=True)
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel({"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]")
    tokenizer.chat_template = "{% for message in messages %}{{message['content']}} {% endfor %}{% if add_generation_prompt %}[BOS]{% endif %}"
    tokenizer.save_pretrained(tmp_path / "base")
    loaded = load_model(tmp_path / "base", dtype=torch.float32, device="cpu")
    task = reference_task(rows_per_template=1)
    row = generate_scenarios(split="train", seed=17, parameters={"rows_per_template": 1})[0]
    examples = [{"interaction": task.project(row["input"], row["reference"])}]
    recipe = {"method": "weight_edit", "edit_method": "legacy_key_value_v1", "seed": 17, "target_step": 0.01,
              "retention_nullspace_weight": 20.0, "max_sequence_tokens": 128, "max_training_tokens": 512,
              "max_keys": 256, "preservation_source": "public_train", "max_preservation_logit_drift": 0.25,
              "preservation_interactions": [task.project(row["input"], None)]}
    return loaded, task, examples, recipe


def test_worker_calculates_real_model_gradient_targets_with_bounded_keys(numerical_fixture):
    from neural_foundry.backends.weight_edit.worker import calculate_edit
    loaded, task, examples, recipe = numerical_fixture
    sites, site, delta, evidence = calculate_edit(loaded, task.job, examples, recipe)
    assert site.role == "residual_write_mlp" and sites.checkpoint_names_verified
    assert delta.matrix().dtype == torch.float64 and bool(torch.isfinite(delta.matrix()).all())
    assert bool(torch.count_nonzero(delta.matrix())) and evidence["gradient_inspection_used"] is True
    assert evidence["key_count"] > 0 and evidence["training_tokens"] <= 512
    assert all(parameter.grad is None for parameter in loaded.model.parameters())
    with pytest.raises(GymJobError, match="budgets"):
        calculate_edit(loaded, task.job, examples, {**recipe, "max_keys": 1})


def test_trainer_calculates_exports_and_seals_execution_receipt(tmp_path, numerical_fixture, monkeypatch):
    from pathlib import Path
    from neural_foundry.backends.weight_edit import adapter, worker
    from neural_foundry.backends.weight_edit.probe import probe_model, record_portability_probes
    from neural_foundry.backends.weight_edit.serving_probes import verify_identity
    from neural_foundry.gym_artifacts import CandidateRef, artifact_files
    from neural_foundry.backends.weight_edit.loader import load_model
    from neural_foundry.execution_runner import WorkerResult

    loaded, task, examples, recipe = numerical_fixture

    class FakeServing:
        revision = "a" * 40

        def __init__(self, reference):
            self.reference = reference

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def record_identity_probes(self):
            local = load_model(self.reference.artifact_path, dtype=torch.float32, device="cpu")
            record = record_portability_probes(local)
            return {**record, "kind": "nf_vllm_weight_identity_v2", "model_digest": self.reference.model_digest,
                    "image": "sha256:" + "3" * 64, "context_tokens": 128,
                    "weight_files": {name: digest for name, digest in artifact_files(Path(self.reference.artifact_path)).items() if name.endswith(".safetensors")}}

        def loaded_identity(self):
            receipt = json.loads((Path(self.reference.artifact_path) / "training_receipt.json").read_bytes())
            evidence = verify_identity(receipt["identity_probes"], self.record_identity_probes())
            return {"served_model_identity_verified": True, "identity_probe_evidence": evidence}

    report = probe_model(loaded.checkpoint_path, tmp_path / "compatibility", dtype=torch.float32, device="cpu",
                         model_id="test/model", serving_factory=FakeServing)
    assert report["supported"] is True
    probe_path = tmp_path / "compatibility.probe.json"
    trainer = weight_edit_trainer(WeightEditBackendConfig(sys.executable, str(probe_path)), task.job)
    base_digest = _document_digest({"model_id": "test/model", "revision": "a" * 40, "files": loaded.file_digests})
    base = CandidateRef("test/model", "full", base_digest, base_digest, base_digest, str(loaded.checkpoint_path),
                        "base", "unchanged_base", "qwen3")
    request = {"job_digest": task.job.digest, "loaded_base": {"model_id": "test/model", "revision": "a" * 40,
                "state_digest": loaded.base_digest}, "base_reference": vars(base), "recipe": recipe,
               "inference_config": {"device": "cpu", "precision": "float32", "context_tokens": 128,
                                    "prompt_renderer_digest": PromptRenderer(loaded.tokenizer).digest},
               "projected_examples_digest": _document_digest({"examples": examples}),
               "dataset_digest": "sha256:" + "b" * 64}
    request["training_request_digest"] = _document_digest(request)
    monkeypatch.setattr(worker, "_resolve_training_checkpoint", lambda *args: SimpleNamespace(path=loaded.checkpoint_path))

    def run(command, **kwargs):
        payload = json.loads(Path(command[-1]).read_bytes())
        reference = worker.train_payload(payload, serving_factory=FakeServing)
        execution = {"kind": "nf_local_worker_receipt_v1", "returncode": 0, "timed_out": False,
                     "stderr_digest": "sha256:" + "0" * 64, "command": command}
        stdout = json.dumps({"kind": "nf_weight_edit_worker_result_v1", "candidate": vars(reference)}).encode()
        return WorkerResult(0, False, execution, stdout, b"")

    monkeypatch.setattr(adapter, "run_local_worker", run)
    reference = trainer.train(request, examples, base, tmp_path / "trained")
    assert reference.kind == "weight_edit" and reference.base_digest == base_digest
    assert trainer.load(request, base, tmp_path / "trained") == reference
    receipt = json.loads((tmp_path / "trained" / "training_receipt.json").read_bytes())
    assert receipt["worker_execution"]["returncode"] == 0
    assert receipt["preservation_max_logit_drift"] <= recipe["max_preservation_logit_drift"]
    receipt_path = tmp_path / "trained" / "training_receipt.json"
    original_receipt = receipt_path.read_bytes()
    altered = {key: value for key, value in receipt.items() if key != "worker_execution"}
    receipt_path.write_text(json.dumps(altered))
    try:
        with pytest.raises(GymJobError, match="execution receipt"):
            trainer.load(request, base, tmp_path / "trained")
    finally:
        receipt_path.write_bytes(original_receipt)
    def failed_run(*args, **kwargs):
        return WorkerResult(1, False, {"kind": "nf_local_worker_receipt_v1", "returncode": 1, "timed_out": False,
                                      "stderr_digest": "sha256:" + "c" * 64}, b"", b"worker preflight failed")
    monkeypatch.setattr(adapter, "run_local_worker", failed_run)
    with pytest.raises(GymJobError, match="local diagnostics"):
        trainer.train(request, examples, base, tmp_path / "failed-training")
    assert (tmp_path / "weight-edit-worker-failure" / "stderr.txt").read_bytes() == b"worker preflight failed"