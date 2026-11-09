from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from neural_foundry import nemo_gym
from neural_foundry.baselines import build_splits
from neural_foundry.gym_data import (
    GymDataError,
    gym_digest,
    load_gym_spec,
    materialize_public_task_data,
)
from neural_foundry.nemo_gym import (
    NeMoGymError,
    NeMoGymRunner,
    NeMoGymTaskError,
    NeMoTaskAdapter,
    NeMoTaskAdapterRegistry,
    load_mcqa_panels,
    simulate_arithmetic_mcqa,
)
from neural_foundry.pythia import ReusableTaskGym, build_mini_pythia_model


def _mcqa_row(task_id: str, question: str, answer: str) -> dict[str, Any]:
    return {
        "uuid": task_id,
        "responses_create_params": {
            "input": [{"role": "user", "content": f"{question}\nA: choice A\nB: choice B\nAnswer:"}]
        },
        "options": [{"A": "choice A"}, {"B": "choice B"}],
        "expected_answer": answer,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _write_test_runner_receipt(output_dir: Path) -> dict[str, Any]:
    materialized = [json.loads(line) for line in (output_dir / "rollouts_materialized_inputs.jsonl").read_text().splitlines() if line]
    (output_dir / "requested_roster.json").write_text(json.dumps({
        "kind": "nf_nemo_requested_roster_v1",
        "expected_keys": [[row["_ng_task_index"], row["_ng_rollout_index"]] for row in materialized],
    }))
    report = nemo_gym.inspect_run_artifacts(output_dir)
    receipt = {
        **report,
        "agent": "test-agent",
        "model_name": "test-model",
        "model_endpoint": "http://127.0.0.1:8000/v1",
        "config_digests": {"test.yaml": "sha256:" + hashlib.sha256(b"config").hexdigest()},
        "input_digest": "sha256:" + hashlib.sha256(b"input").hexdigest(),
        "command": ["gym", "eval", "run"],
    }
    (output_dir / nemo_gym.RUN_RECEIPT_NAME).write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def test_nemo_adapter_registry_declares_capabilities_and_allows_eval_only_tasks() -> None:
    registry = NeMoTaskAdapterRegistry()
    assert registry.list_adapters()["mcqa_single_token"] == {
        "weight_space": True,
        "lora": True,
        "sft": True,
        "dpo": True,
        "evaluation": True,
    }
    registry.register(NeMoTaskAdapter(adapter_id="interactive_tools", supports_evaluation=True))
    assert registry.list_adapters()["interactive_tools"]["evaluation"] is True
    assert registry.list_adapters()["interactive_tools"]["sft"] is False
    with pytest.raises(NeMoGymTaskError, match="evaluation-only"):
        registry.load_panels("interactive_tools")


def test_nemo_mcqa_panels_feed_identical_wsp_and_lora_data(tmp_path: Path) -> None:
    model = build_mini_pythia_model()
    train = _write_jsonl(
        tmp_path / "train.jsonl",
        [_mcqa_row("tr-1", "Which answer is first?", "A"), _mcqa_row("tr-2", "Which answer is second?", "B")],
    )
    validation = _write_jsonl(
        tmp_path / "validation.jsonl",
        [_mcqa_row("va-1", "Which answer is third?", "A"), _mcqa_row("va-2", "Which answer is fourth?", "B")],
    )
    holdout = _write_jsonl(
        tmp_path / "holdout.jsonl",
        [_mcqa_row("ho-1", "Which answer is fifth?", "A"), _mcqa_row("ho-2", "Which answer is sixth?", "B")],
    )

    panels = load_mcqa_panels(train, validation, holdout, tokenizer=model.tokenizer)
    assert panels.source_manifest()["gym_digest"] == gym_digest(panels.gym)
    splits = build_splits(panels.gym, (), model=model)
    shared_data = materialize_public_task_data(panels.gym, tokenizer=model.tokenizer)

    assert splits.dev == shared_data.train_items
    assert splits.val == shared_data.validation_items
    assert splits.dataset_digest == shared_data.dataset_digest
    assert len(panels.gym.holdout_rules) == 2
    assert panels.source_manifest()["source_panels"]["train"]["sha256"].startswith("sha256:")
    changed_train = _write_jsonl(
        tmp_path / "changed-train.jsonl",
        [_mcqa_row("tr-1", "Which answer is first changed?", "A"), _mcqa_row("tr-2", "Which answer is second?", "B")],
    )
    changed_panels = load_mcqa_panels(changed_train, validation, holdout, tokenizer=model.tokenizer)
    assert changed_panels.gym.gym_id != panels.gym.gym_id

    spec_path = panels.write_spec(tmp_path / "nemo-mcqa-gym.json")
    assert spec_path.is_file()
    reloaded = load_gym_spec(spec_path)
    reloaded_data = materialize_public_task_data(reloaded, tokenizer=model.tokenizer)
    assert reloaded_data.train_items == splits.dev
    assert reloaded_data.validation_items == splits.val
    with pytest.raises(FileExistsError):
        panels.write_spec(spec_path)


def test_shared_materializer_rejects_prompt_overlap_after_prefix_expansion() -> None:
    gym = ReusableTaskGym(
        gym_id="gym.prefix-overlap.train-validation.v1",
        skills_taught=("multiple_choice_qa",),
        base_rules=(("x", "A"),),
        validation_rules=(("Task: x", "B"),),
        gym_dev_prefixes=("Task:",),
        gym_val_prefixes=(),
    )
    with pytest.raises(GymDataError, match="materialized prompt.*both train and validation"):
        materialize_public_task_data(gym)

    holdout_gym = ReusableTaskGym(
        gym_id="gym.prefix-overlap.train-holdout.v1",
        skills_taught=("multiple_choice_qa",),
        base_rules=(("Task: x", "A"),),
        validation_rules=(("validation", "B"),),
        gym_dev_prefixes=(),
        gym_val_prefixes=(),
        holdout_rules=(("x", "A"),),
        holdout_prefixes=("Task:",),
    )
    with pytest.raises(GymDataError, match="materialized prompt.*both train and holdout"):
        materialize_public_task_data(holdout_gym)


def test_shared_materializer_enforces_text_bound_without_tokenizer() -> None:
    gym = ReusableTaskGym(
        gym_id="gym.oversized-prompt.v1",
        skills_taught=("multiple_choice_qa",),
        base_rules=(("x" * 16_385, "A"),),
        validation_rules=(("validation", "B"),),
        gym_dev_prefixes=(),
        gym_val_prefixes=(),
    )
    with pytest.raises(GymDataError, match="text bound"):
        materialize_public_task_data(gym)


def test_nemo_string_prompt_obeys_same_text_bound_as_message_prompt() -> None:
    oversized = "x" * 16_385
    from neural_foundry.nemo_gym import _single_user_prompt

    for task_input in (
        {"input": oversized},
        {"input": [{"role": "user", "content": oversized}]},
    ):
        with pytest.raises(NeMoGymTaskError, match="prompt exceeds the byte limit"):
            _single_user_prompt(
                {"responses_create_params": task_input},
                label="oversized prompt",
            )


def test_seeded_arithmetic_panels_are_reproducible_disjoint_and_shared(tmp_path: Path) -> None:
    model = build_mini_pythia_model()
    panels = simulate_arithmetic_mcqa(
        tmp_path / "first",
        tokenizer=model.tokenizer,
        seed=1729,
        train_size=6,
        validation_size=4,
        holdout_size=4,
    )
    repeated = simulate_arithmetic_mcqa(
        tmp_path / "repeated",
        tokenizer=model.tokenizer,
        seed=1729,
        train_size=6,
        validation_size=4,
        holdout_size=4,
    )
    assert panels.source_digests == repeated.source_digests
    assert panels.source_metadata == repeated.source_metadata

    rows = {
        split: [json.loads(line) for line in (tmp_path / "first" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        for split in ("train", "validation", "holdout")
    }
    for left, right in (("train", "validation"), ("train", "holdout"), ("validation", "holdout")):
        assert {row["task_id"] for row in rows[left]}.isdisjoint(row["task_id"] for row in rows[right])
        assert {row["responses_create_params"]["input"][0]["content"] for row in rows[left]}.isdisjoint(
            row["responses_create_params"]["input"][0]["content"] for row in rows[right]
        )

    splits = build_splits(panels.gym, (), model=model)
    shared_data = materialize_public_task_data(panels.gym, tokenizer=model.tokenizer)
    assert splits.dev == shared_data.train_items
    assert splits.val == shared_data.validation_items
    assert splits.dataset_digest == shared_data.dataset_digest


def test_nemo_panels_bind_into_job_spec_with_private_answer_references(tmp_path: Path) -> None:
    from neural_foundry.gym_job import (
        AcceptanceGate,
        CandidateMethod,
        ComponentReference,
        GymComponentRegistry,
        GymJobSpec,
    )
    from neural_foundry.gym_service import GymStudioService
    from neural_foundry.nemo_gym import register_nemo_mcqa_scenario_sources

    model = build_mini_pythia_model()
    panels = load_mcqa_panels(
        _write_jsonl(tmp_path / "train.jsonl", [_mcqa_row("tr", "Which answer is first?", "A")]),
        _write_jsonl(tmp_path / "validation.jsonl", [_mcqa_row("va", "Which answer is second?", "B")]),
        _write_jsonl(tmp_path / "holdout.jsonl", [_mcqa_row("ho", "Which answer is third?", "A")]),
        tokenizer=model.tokenizer,
    )
    components = GymComponentRegistry()
    sources = register_nemo_mcqa_scenario_sources(components, panels, origin="real")
    scorer_digest = "sha256:" + hashlib.sha256(b"test-scorer-v1").hexdigest()
    verifier_digest = "sha256:" + hashlib.sha256(b"test-verifier-v1").hexdigest()
    method_digest = "sha256:" + hashlib.sha256(b"test-method-v1").hexdigest()
    components.register("scorer", "exact-answer", "1.0.0", scorer_digest, lambda **_kwargs: {})
    components.register("verifier", "exact-answer", "1.0.0", verifier_digest, lambda **_kwargs: {"passed": True})
    components.register("candidate_method", "unchanged_base", "1.0.0", method_digest, lambda **_kwargs: None)
    job = GymJobSpec(
        job_id="nemo-contract-review",
        title="Find clauses",
        description="Return one answer token for each prompt.",
        task_input_schema={
            "type": "object", "properties": {"prompt": {"type": "string"}},
            "required": ["prompt"], "additionalProperties": False,
        },
        task_output_schema={
            "type": "object", "properties": {"answer_token": {"type": "string"}},
            "required": ["answer_token"], "additionalProperties": False,
        },
        scenario_sources=sources,
        scorer=ComponentReference("exact-answer", "1.0.0", scorer_digest, {}),
        verifier=ComponentReference("exact-answer", "1.0.0", verifier_digest, {}),
        candidate_methods=(CandidateMethod("unchanged_base", "1.0.0", method_digest, {}),),
        gates=(
            AcceptanceGate("task", "task", "accuracy", "validation", "ge", 0.9),
            AcceptanceGate("behavior", "behavior", "unsupported", "holdout", "le", 0.0),
            AcceptanceGate("controls", "controls", "beats_control", "controls", "eq", 1.0),
            AcceptanceGate("reproducibility", "reproducibility", "repeat", "reproducibility", "eq", 1.0),
        ),
        metadata={"origin": "nemo-gym"},
    )
    service = GymStudioService(components)
    batch = service.materialize_public_scenarios(job, split="train", seed=5)
    assert batch["scenarios"][0]["input"]["prompt"] == panels.gym.base_rules[0][0]
    assert "reference" not in batch["scenarios"][0]
    assert batch["scenarios"][0]["reference_digest"].startswith("sha256:")
    from neural_foundry.gym_lifecycle import AdapterIdentity, TaskAdapter

    task = TaskAdapter(
        AdapterIdentity("nemo-test", "1", scorer_digest), job, "single_response",
        AdapterIdentity("nemo-projection", "1", scorer_digest), "chat_messages_v1",
        lambda value, _response: {"format": "chat_messages_v1", "messages": [{"role": "user", "content": value["prompt"]}]},
        lambda _service: None,
    )
    output = tmp_path / "compiled.jsonl"
    receipt = nemo_gym.compile_public_nemo_dataset(task, service, split="train", seed=5, output_path=output)
    row = json.loads(output.read_text())
    assert receipt["dataset_digest"] == "sha256:" + hashlib.sha256(output.read_bytes()).hexdigest()
    assert receipt["execution_ready"] is False and receipt["verifier_bound"] is False
    assert row["metadata"]["nf_scorer_digest"] == scorer_digest
    assert "reference" not in row and "expected_answer" not in row
    with pytest.raises(NeMoGymError, match="public train or validation"):
        nemo_gym.compile_public_nemo_dataset(task, service, split="holdout", seed=5, output_path=tmp_path / "holdout-output.jsonl")
    import os
    import subprocess

    if python := os.environ.get("NF_NEMO_GYM_PYTHON"):
        code = ("import json,sys; from resources_servers.mcqa.app import MCQARunRequest; "
                "row=MCQARunRequest.model_validate(json.load(sys.stdin)); "
                "print(row.metadata['nf_scorer_digest'])")
        result = subprocess.run([python, "-c", code], input=json.dumps(row), capture_output=True,
                                text=True, timeout=60, check=False)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == scorer_digest


@pytest.mark.parametrize(
    "options",
    [
        {"seed": -1},
        {"seed": True},
        {"train_size": 0},
        {"validation_size": 1.5},
        {"train_size": 90, "validation_size": 10, "holdout_size": 1},
    ],
)
def test_seeded_arithmetic_generator_rejects_invalid_configuration(
    tmp_path: Path, options: dict[str, Any]
) -> None:
    model = build_mini_pythia_model()
    with pytest.raises(NeMoGymTaskError):
        simulate_arithmetic_mcqa(tmp_path / "invalid", tokenizer=model.tokenizer, **options)


def test_studio_cli_generates_simulated_gym_spec_and_provenance(tmp_path: Path) -> None:
    from neural_foundry.gym_data import load_gym_spec
    from neural_foundry.studio import main

    output = tmp_path / "simulated-gym"
    assert main([
        "simulate-nemo-gym",
        "--adapter", "simulated_arithmetic_mcqa",
        "--model", "nf/mini-pythia-fixture",
        "--output-dir", str(output),
        "--seed", "7",
        "--train-size", "4",
        "--validation-size", "2",
        "--holdout-size", "2",
    ]) == 0
    gym = load_gym_spec(output / "gym.json")
    manifest = json.loads((output / "gym.sources.json").read_text(encoding="utf-8"))
    assert len(gym.base_rules) == 4
    assert len(gym.validation_rules) == 2
    assert len(gym.holdout_rules) == 2
    assert manifest["source_metadata"]["kind"] == "deterministic_simulator"
    assert manifest["source_metadata"]["seed"] == 7


def test_mcqa_import_refuses_non_single_turn_or_multitoken_targets(tmp_path: Path) -> None:
    model = build_mini_pythia_model()
    train = _write_jsonl(
        tmp_path / "train.jsonl",
        [{
            **_mcqa_row("tr-1", "Question?", "A"),
            "responses_create_params": {"input": [
                {"role": "system", "content": "Rules"},
                {"role": "user", "content": "Question? Answer:"},
            ]},
        }],
    )
    validation = _write_jsonl(tmp_path / "validation.jsonl", [_mcqa_row("va", "Question 2?", "B")])
    holdout = _write_jsonl(tmp_path / "holdout.jsonl", [_mcqa_row("ho", "Question 3?", "A")])
    with pytest.raises(NeMoGymTaskError, match="one text user message"):
        load_mcqa_panels(train, validation, holdout, tokenizer=model.tokenizer)

    train = _write_jsonl(tmp_path / "train.jsonl", [_mcqa_row("tr-1", "Question?", "A")])
    with pytest.raises(NeMoGymTaskError, match="one ASCII answer letter"):
        bad = _write_jsonl(tmp_path / "validation.jsonl", [_mcqa_row("va", "Question 2?", "AA")])
        load_mcqa_panels(train, bad, holdout, tokenizer=model.tokenizer)


def test_sequence_sft_refuses_multiturn_or_tool_trajectories() -> None:
    from neural_foundry.nemo_gym import _assistant_text

    with pytest.raises(NeMoGymTaskError, match="single-turn"):
        _assistant_text(
            {"output": [
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "A"}]},
                {"type": "function_call", "name": "lookup", "arguments": "{}"},
            ]},
            label="tool trajectory",
        )
    with pytest.raises(NeMoGymTaskError, match="tool, multi-part"):
        _assistant_text(
            {"output": [{
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "A"},
                    {"type": "refusal", "refusal": "refused"},
                ],
            }]},
            label="multipart response",
        )


def test_successful_nemo_rollout_trains_sequence_lora_and_scores_validation(tmp_path: Path) -> None:
    import torch
    from neural_foundry.baselines import BaselineConfig
    from neural_foundry.nemo_gym import train_mcqa_rollouts

    model = build_mini_pythia_model()
    train_row = _mcqa_row("tr-1", "The capital of France is", "A")
    validation_row = _mcqa_row("va-1", "The capital of Japan is", "B")
    holdout_row = _mcqa_row("ho-1", "The capital of Germany is", "A")
    panels = load_mcqa_panels(
        _write_jsonl(tmp_path / "train.jsonl", [train_row]),
        _write_jsonl(tmp_path / "validation.jsonl", [validation_row]),
        _write_jsonl(tmp_path / "holdout.jsonl", [holdout_row]),
        tokenizer=model.tokenizer,
    )
    run_dir = tmp_path / "successful-run"
    run_dir.mkdir()
    materialized_path = _write_jsonl(
        run_dir / "rollouts_materialized_inputs.jsonl",
        [{"_ng_task_index": 0, "_ng_rollout_index": 0, "task_input": train_row}],
    )
    rollout_path = _write_jsonl(
        run_dir / "rollouts.jsonl",
        [{
            "_ng_task_index": 0,
            "_ng_rollout_index": 0,
            "reward": 1.0,
            "mask_sample": False,
            "evaluation_completed": True,
            "response": {"output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": " A"}],
            }]},
        }],
    )
    _write_jsonl(run_dir / "rollouts_failures.jsonl", [])
    _write_test_runner_receipt(run_dir)
    config = BaselineConfig(
        kind="lora", lr=0.1, steps=4, targets="final_down_proj", rank=2, seed=3
    )

    training = train_mcqa_rollouts(
        model=model,
        panels=panels,
        materialized_inputs_jsonl=materialized_path,
        rollouts_jsonl=rollout_path,
        config=config,
        retention_guard=(),
    )

    assert training.dataset.accepted_rollouts == 1
    assert training.dataset.source_runner_receipt_digest.startswith("sha256:")
    assert training.dataset.examples == ((panels.gym.base_rules[0][0], " A"),)
    assert training.fit.merged_weights
    assert training.fit.cost.optimizer_steps == 4
    assert training.validation_examples == 1
    assert 0.0 <= training.validation_accuracy <= 1.0
    assert torch.isfinite(torch.tensor(training.fit.final_train_loss))


def test_reward_margin_builds_same_task_dpo_pair_and_trains_lora(tmp_path: Path) -> None:
    import torch
    from neural_foundry.baselines import DPOConfig
    from neural_foundry.nemo_gym import import_mcqa_preferences, train_mcqa_preferences

    model = build_mini_pythia_model()
    train_row = _mcqa_row("tr-1", "The capital of France is", "A")
    panels = load_mcqa_panels(
        _write_jsonl(tmp_path / "train.jsonl", [train_row]),
        _write_jsonl(tmp_path / "validation.jsonl", [_mcqa_row("va-1", "The capital of Japan is", "B")]),
        _write_jsonl(tmp_path / "holdout.jsonl", [_mcqa_row("ho-1", "The capital of Germany is", "A")]),
        tokenizer=model.tokenizer,
    )
    run_dir = tmp_path / "dpo-run"
    run_dir.mkdir()
    _write_jsonl(
        run_dir / "rollouts_materialized_inputs.jsonl",
        [
            {"_ng_task_index": 0, "_ng_rollout_index": index, "task_input": train_row}
            for index in range(2)
        ],
    )
    _write_jsonl(
        run_dir / "rollouts.jsonl",
        [
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
                "reward": 1.0,
                "mask_sample": False,
                "evaluation_completed": True,
                "response": {"output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": " A"}]}]},
            },
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 1,
                "reward": 0.0,
                "mask_sample": False,
                "evaluation_completed": True,
                "response": {"output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": " B"}]}]},
            },
        ],
    )
    _write_jsonl(run_dir / "rollouts_failures.jsonl", [])
    _write_test_runner_receipt(run_dir)
    input_path, rollout_path = run_dir / "rollouts_materialized_inputs.jsonl", run_dir / "rollouts.jsonl"
    dataset = import_mcqa_preferences(
        panels,
        materialized_inputs_jsonl=input_path,
        rollouts_jsonl=rollout_path,
        minimum_reward_margin=0.5,
    )
    assert dataset.pairs == ((panels.gym.base_rules[0][0], " A", " B"),)

    training = train_mcqa_preferences(
        model=model,
        panels=panels,
        materialized_inputs_jsonl=input_path,
        rollouts_jsonl=rollout_path,
        config=DPOConfig(learning_rate=0.05, steps=3, rank=2, seed=2, retention_weight=0.0),
        minimum_reward_margin=0.5,
        retention_guard=(),
    )
    assert training.dataset.dataset_digest == dataset.dataset_digest
    assert training.dataset.source_runner_receipt_digest.startswith("sha256:")
    assert training.fit.merged_weights
    assert torch.isfinite(torch.tensor(training.fit.final_train_loss))
    assert training.validation_examples == 1


def test_sft_import_requires_runner_receipt_and_rejects_changed_rollouts(tmp_path: Path) -> None:
    from neural_foundry.nemo_gym import import_successful_mcqa_rollouts

    model = build_mini_pythia_model()
    train_row = _mcqa_row("tr-1", "The capital of France is", "A")
    panels = load_mcqa_panels(
        _write_jsonl(tmp_path / "train.jsonl", [train_row]),
        _write_jsonl(tmp_path / "validation.jsonl", [_mcqa_row("va-1", "The capital of Japan is", "B")]),
        _write_jsonl(tmp_path / "holdout.jsonl", [_mcqa_row("ho-1", "The capital of Germany is", "A")]),
        tokenizer=model.tokenizer,
    )
    run_dir = tmp_path / "receipt-run"
    run_dir.mkdir()
    materialized_path = _write_jsonl(
        run_dir / "rollouts_materialized_inputs.jsonl",
        [{"_ng_task_index": 0, "_ng_rollout_index": 0, "task_input": train_row}],
    )
    rollout = {
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "reward": 1.0,
        "mask_sample": False,
        "evaluation_completed": True,
        "response": {"output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": " A"}],
        }]},
    }
    rollout_path = _write_jsonl(run_dir / "rollouts.jsonl", [rollout])
    _write_jsonl(run_dir / "rollouts_failures.jsonl", [])
    kwargs = {
        "panels": panels,
        "materialized_inputs_jsonl": materialized_path,
        "rollouts_jsonl": rollout_path,
    }
    with pytest.raises(NeMoGymError, match="runner receipt"):
        import_successful_mcqa_rollouts(**kwargs)

    _write_test_runner_receipt(run_dir)
    _write_jsonl(rollout_path, [{**rollout, "reward": 0.5}])
    with pytest.raises(NeMoGymError, match="receipt does not match"):
        import_successful_mcqa_rollouts(**kwargs)


def test_studio_nemo_sft_cli_emits_reproducible_training_artifact(tmp_path: Path) -> None:
    from neural_foundry.studio import main

    train_row = _mcqa_row("tr-1", "The capital of France is", "A")
    validation_row = _mcqa_row("va-1", "The capital of Japan is", "B")
    holdout_row = _mcqa_row("ho-1", "The capital of Germany is", "A")
    train = _write_jsonl(tmp_path / "train.jsonl", [train_row])
    validation = _write_jsonl(tmp_path / "validation.jsonl", [validation_row])
    holdout = _write_jsonl(tmp_path / "holdout.jsonl", [holdout_row])
    run_dir = tmp_path / "nemo-run"
    run_dir.mkdir()
    _write_jsonl(
        run_dir / "rollouts_materialized_inputs.jsonl",
        [{"_ng_task_index": 0, "_ng_rollout_index": 0, "task_input": train_row}],
    )
    _write_jsonl(
        run_dir / "rollouts.jsonl",
        [{
            "_ng_task_index": 0,
            "_ng_rollout_index": 0,
            "reward": 1.0,
            "mask_sample": False,
            "evaluation_completed": True,
            "response": {"output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": " A"}],
            }]},
        }],
    )
    _write_jsonl(run_dir / "rollouts_failures.jsonl", [])
    _write_test_runner_receipt(run_dir)
    output = tmp_path / "sft-output"

    assert main([
        "nemo-sft",
        "--model", "nf/mini-pythia-fixture",
        "--train-jsonl", str(train),
        "--validation-jsonl", str(validation),
        "--holdout-jsonl", str(holdout),
        "--materialized-inputs", str(run_dir / "rollouts_materialized_inputs.jsonl"),
        "--rollouts", str(run_dir / "rollouts.jsonl"),
        "--output-dir", str(output),
        "--steps", "2",
        "--retention-weight", "0",
    ]) == 0
    receipt = json.loads((output / "training_receipt.json").read_text(encoding="utf-8"))
    assert receipt["kind"] == "nf_nemo_mcqa_sequence_sft_v1"
    assert receipt["accepted_rollouts"] == 1
    assert receipt["holdout_used"] is False
    assert receipt["source_rollout_digest"].startswith("sha256:")
    assert receipt["source_runner_receipt_digest"].startswith("sha256:")
    assert receipt["base_model_digest"].startswith("sha256:")
    assert receipt["candidate_model_digest"].startswith("sha256:")
    assert (output / "merged_weights.pt").is_file()
    from neural_foundry.edit_plan import tensor_state_digest
    from neural_foundry.studio import WeightProgrammingStudio

    candidate, loaded_receipt = WeightProgrammingStudio().load_nemo_sft_candidate(
        build_mini_pythia_model(), output
    )
    assert loaded_receipt == receipt
    assert tensor_state_digest(candidate.tensors) == receipt["candidate_model_digest"]


def test_studio_nemo_dpo_cli_emits_reloadable_preference_candidate(tmp_path: Path) -> None:
    from neural_foundry.edit_plan import tensor_state_digest
    from neural_foundry.studio import WeightProgrammingStudio, main

    train_row = _mcqa_row("tr-1", "The capital of France is", "A")
    train = _write_jsonl(tmp_path / "train.jsonl", [train_row])
    validation = _write_jsonl(
        tmp_path / "validation.jsonl", [_mcqa_row("va-1", "The capital of Japan is", "B")]
    )
    holdout = _write_jsonl(
        tmp_path / "holdout.jsonl", [_mcqa_row("ho-1", "The capital of Germany is", "A")]
    )
    run_dir = tmp_path / "dpo-run"
    run_dir.mkdir()
    _write_jsonl(
        run_dir / "rollouts_materialized_inputs.jsonl",
        [{"_ng_task_index": 0, "_ng_rollout_index": index, "task_input": train_row} for index in range(2)],
    )
    _write_jsonl(
        run_dir / "rollouts.jsonl",
        [
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": index,
                "reward": reward,
                "mask_sample": False,
                "evaluation_completed": True,
                "response": {"output": [{
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": response}],
                }]},
            }
            for index, reward, response in ((0, 1.0, " A"), (1, 0.0, " B"))
        ],
    )
    _write_jsonl(run_dir / "rollouts_failures.jsonl", [])
    _write_test_runner_receipt(run_dir)
    output = tmp_path / "dpo-output"

    assert main([
        "nemo-dpo",
        "--model", "nf/mini-pythia-fixture",
        "--train-jsonl", str(train),
        "--validation-jsonl", str(validation),
        "--holdout-jsonl", str(holdout),
        "--materialized-inputs", str(run_dir / "rollouts_materialized_inputs.jsonl"),
        "--rollouts", str(run_dir / "rollouts.jsonl"),
        "--output-dir", str(output),
        "--steps", "2",
        "--rank", "2",
        "--minimum-reward-margin", "0.5",
        "--retention-weight", "0",
    ]) == 0
    receipt = json.loads((output / "training_receipt.json").read_text(encoding="utf-8"))
    assert receipt["kind"] == "nf_nemo_mcqa_dpo_v1"
    assert receipt["preference_pairs"] == 1
    assert receipt["source_runner_receipt_digest"].startswith("sha256:")
    candidate, loaded_receipt = WeightProgrammingStudio().load_nemo_sft_candidate(
        build_mini_pythia_model(), output
    )
    assert loaded_receipt == receipt
    assert tensor_state_digest(candidate.tensors) == receipt["candidate_model_digest"]


def test_studio_cli_imports_nemo_mcqa_into_a_portable_gym_spec(tmp_path: Path) -> None:
    from neural_foundry.gym_data import load_gym_spec
    from neural_foundry.studio import main

    train = _write_jsonl(tmp_path / "train.jsonl", [_mcqa_row("tr", "The capital of France is", "A")])
    validation = _write_jsonl(tmp_path / "validation.jsonl", [_mcqa_row("va", "The capital of Japan is", "B")])
    holdout = _write_jsonl(tmp_path / "holdout.jsonl", [_mcqa_row("ho", "The capital of Germany is", "A")])
    spec_path = tmp_path / "nemo-gym.json"

    assert main([
        "import-nemo-panels",
        "--adapter", "mcqa_single_token",
        "--model", "nf/mini-pythia-fixture",
        "--train-jsonl", str(train),
        "--validation-jsonl", str(validation),
        "--holdout-jsonl", str(holdout),
        "--output-spec", str(spec_path),
    ]) == 0
    assert load_gym_spec(spec_path).base_rules[0][0].endswith("Answer:")
    assert Path(str(spec_path) + ".sources.json").is_file()


def _fake_artifacts(output_path: Path, *, masked: bool = False, missing: bool = False) -> None:
    root = output_path.parent
    materialized = [{
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "task_input": {"responses_create_params": {"input": "Question?"}},
    }]
    result = [] if missing else [{
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "reward": 0.0 if masked else 1.0,
        "mask_sample": masked,
        "evaluation_completed": True,
        "response": {"output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "A"}],
        }]},
    }]
    _write_jsonl(root / "rollouts_materialized_inputs.jsonl", materialized)
    _write_jsonl(output_path, result)
    _write_jsonl(root / "rollouts_failures.jsonl", [])


@pytest.mark.parametrize("change", [{"mask_sample": "false"}, {"evaluation_completed": 1}, {"reward": 1e308}, {"reward": True}])
def test_nemo_completion_rejects_malformed_status_and_reward(tmp_path: Path, change: dict[str, Any]) -> None:
    _fake_artifacts(tmp_path / "rollouts.jsonl")
    (tmp_path / "requested_roster.json").write_text(json.dumps({"kind": "nf_nemo_requested_roster_v1", "expected_keys": [[0, 0]]}))
    row = json.loads((tmp_path / "rollouts.jsonl").read_text())
    _write_jsonl(tmp_path / "rollouts.jsonl", [{**row, **change}])
    assert nemo_gym.inspect_run_artifacts(tmp_path)["complete"] is False


def test_nemo_empty_outputs_are_not_complete(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "rollouts.jsonl", [])
    _write_jsonl(tmp_path / "rollouts_materialized_inputs.jsonl", [])
    assert nemo_gym.inspect_run_artifacts(tmp_path)["complete"] is False


def test_zero_valued_mcqa_task_identity_is_preserved(tmp_path: Path) -> None:
    model = build_mini_pythia_model()
    panels = load_mcqa_panels(
        _write_jsonl(tmp_path / "train.jsonl", [_mcqa_row(0, "The capital of France is", "A")]),
        _write_jsonl(tmp_path / "validation.jsonl", [_mcqa_row(1, "The capital of Japan is", "B")]),
        _write_jsonl(tmp_path / "holdout.jsonl", [_mcqa_row(2, "The capital of Germany is", "A")]),
        tokenizer=model.tokenizer,
    )
    assert panels.train_task_ids == ("0",)
    assert nemo_gym._task_id({"uuid": 0, "task_id": "fallback"}) == 0


def test_mcqa_panel_loader_accepts_zero_task_id(tmp_path: Path) -> None:
    model = build_mini_pythia_model()
    panels = load_mcqa_panels(
        _write_jsonl(tmp_path / "train.jsonl", [_mcqa_row(0, "The capital of France is", "A")]),
        _write_jsonl(tmp_path / "validation.jsonl", [_mcqa_row(1, "The capital of Japan is", "B")]),
        _write_jsonl(tmp_path / "holdout.jsonl", [_mcqa_row(2, "The capital of Germany is", "A")]),
        tokenizer=model.tokenizer,
    )
    assert panels.train_task_ids == ("0",)


def test_dpo_import_rejects_duplicate_materialized_input_identity(tmp_path: Path) -> None:
    model = build_mini_pythia_model()
    panels = load_mcqa_panels(
        _write_jsonl(tmp_path / "train.jsonl", [_mcqa_row("tr", "The capital of France is", "A")]),
        _write_jsonl(tmp_path / "validation.jsonl", [_mcqa_row("va", "The capital of Japan is", "B")]),
        _write_jsonl(tmp_path / "holdout.jsonl", [_mcqa_row("ho", "The capital of Germany is", "A")]),
        tokenizer=model.tokenizer,
    )
    run_dir = tmp_path / "duplicate-dpo"
    run_dir.mkdir()
    row = {"_ng_task_index": 0, "_ng_rollout_index": 0,
           "task_input": _mcqa_row("tr", "The capital of France is", "A")}
    _write_jsonl(run_dir / "rollouts_materialized_inputs.jsonl", [row, row])
    _write_jsonl(run_dir / "rollouts.jsonl", [])
    _write_jsonl(run_dir / "rollouts_failures.jsonl", [])
    with pytest.raises(NeMoGymError, match="duplicate materialized-input identity"):
        nemo_gym.import_mcqa_preferences(
            panels, materialized_inputs_jsonl=run_dir / "rollouts_materialized_inputs.jsonl",
            rollouts_jsonl=run_dir / "rollouts.jsonl",
        )


def test_timeout_cleanup_signals_group_after_cli_leader_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    import os
    import signal
    import subprocess

    if os.name == "nt":
        pytest.skip("Process-group escalation is exercised under WSL")
    calls = []

    class Process:
        pid = 123
        returncode = None
        waits = 0

        def poll(self):
            return 0

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("gym", timeout)
            self.returncode = -signal.SIGKILL
            return self.returncode

    monkeypatch.setattr(nemo_gym.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    nemo_gym._stop_process_tree(Process())
    assert calls == [(123, signal.SIGTERM), (123, signal.SIGKILL)]


def test_timeout_stops_the_entire_posix_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    import os
    import signal
    import subprocess

    if os.name == "nt":
        pytest.skip("POSIX process-group escalation is checked under WSL")
    signals: list[int] = []

    class Process:
        pid = 123
        returncode = None
        waits = 0

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("fake-cli", timeout)
            self.returncode = -signal.SIGKILL
            return self.returncode

    process = Process()
    monkeypatch.setattr(nemo_gym.os, "killpg", lambda _pid, sig: signals.append(sig))
    nemo_gym._stop_process_tree(process)
    assert signals == [signal.SIGTERM, signal.SIGKILL]


def test_local_worker_timeout_closes_child_inherited_pipes(tmp_path: Path) -> None:
    import os
    import sys

    from neural_foundry.execution_runner import run_local_worker

    if os.name == "nt":
        pytest.skip("POSIX process-group behavior is exercised under WSL")
    child = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    parent = "import subprocess,sys,time; subprocess.Popen([sys.executable, '-c', sys.argv[1]]); time.sleep(60)"
    result = run_local_worker([sys.executable, "-c", parent, child], cwd=tmp_path, timeout_seconds=2)
    assert result.timed_out is True
    assert result.receipt["timed_out"] is True
    assert result.receipt["duration_seconds"] < 15


def test_local_worker_receipt_hashes_raw_output_bytes(tmp_path: Path) -> None:
    import hashlib
    import sys

    from neural_foundry.execution_runner import run_local_worker

    result = run_local_worker(
        [sys.executable, "-c", "import sys; sys.stdout.buffer.write(bytes([255]))"],
        cwd=tmp_path, timeout_seconds=30,
    )
    assert result.returncode == 0
    assert result.receipt["stdout_digest"] == "sha256:" + hashlib.sha256(b"\xff").hexdigest()


def test_local_worker_cleans_process_tree_on_cancellation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from neural_foundry import execution_runner

    class Process:
        calls = 0

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt()
            return b"", b""

    process = Process()
    stopped = []
    monkeypatch.setattr(execution_runner.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(execution_runner, "stop_process_tree", stopped.append)
    with pytest.raises(KeyboardInterrupt):
        execution_runner.run_local_worker(["worker"], cwd=tmp_path, timeout_seconds=1)
    assert stopped == [process] and process.calls == 2


def test_local_worker_can_clean_descendants_after_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from neural_foundry import execution_runner

    class Process:
        returncode = 0

        def communicate(self, timeout=None):
            return b"", b""

    process = Process()
    stopped = []
    monkeypatch.setattr(execution_runner.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(execution_runner, "stop_process_tree", stopped.append)
    result = execution_runner.run_local_worker(["worker"], cwd=tmp_path, timeout_seconds=1, cleanup_descendants=True)
    assert result.returncode == 0 and stopped == [process]


@pytest.mark.parametrize("interaction", [
    {"format": "chat_messages_v1", "messages": [{"role": "user", "content": "classify this"}]},
    {"format": "tool_messages_v1", "messages": [{"role": "user", "content": "lookup"}],
     "max_tool_calls": 1, "tools": [{"name": "lookup", "description": "Find an item",
                                     "arguments": {"type": "object", "properties": {}, "additionalProperties": False}}]},
    {"format": "json_extraction_v1", "messages": [{"role": "user", "content": "extract"}],
     "output_schema": {"type": "object", "properties": {"name": {"type": "string"}},
                       "required": ["name"], "additionalProperties": False}},
])
def test_interaction_formats_roundtrip_nemo_rows(interaction: dict[str, Any]) -> None:
    from neural_foundry.gym_formats import InteractionFormatRegistry
    from neural_foundry.gym_job import GymJobError

    adapter = InteractionFormatRegistry().get(interaction["format"])
    row = adapter.to_nemo(interaction)
    assert adapter.from_nemo(row) == interaction
    row["responses_create_params"]["extra"] = True
    with pytest.raises(GymJobError, match="unsupported parameters"):
        adapter.from_nemo(row)


def test_interaction_converter_rejects_mismatched_format() -> None:
    from neural_foundry.gym_formats import InteractionFormatRegistry
    from neural_foundry.gym_job import GymJobError

    adapter = InteractionFormatRegistry().get("chat_messages_v1")
    with pytest.raises(GymJobError, match="selected format"):
        adapter.to_nemo({"format": "tool_messages_v1", "messages": []})
    with pytest.raises(GymJobError, match="unsupported message"):
        adapter.to_nemo({"format": "chat_messages_v1", "messages": [{"role": [], "content": "x"}]})


@pytest.mark.parametrize("interaction", [
    {"format": "chat_messages_v1", "messages": [{"role": "user", "content": "classify this"}]},
    {"format": "tool_messages_v1", "messages": [{"role": "user", "content": "lookup"}],
     "max_tool_calls": 1, "tools": [{"name": "lookup", "description": "Find an item",
                                     "arguments": {"type": "object", "properties": {}, "additionalProperties": False}}]},
    {"format": "json_extraction_v1", "messages": [{"role": "user", "content": "extract"}],
     "output_schema": {"type": "object", "properties": {"name": {"type": "string"}},
                       "required": ["name"], "additionalProperties": False}},
])
def test_interaction_formats_validate_against_installed_nemo(interaction: dict[str, Any]) -> None:
    import os
    import subprocess

    from neural_foundry.gym_formats import InteractionFormatRegistry

    python = os.environ.get("NF_NEMO_GYM_PYTHON")
    if not python:
        pytest.skip("set NF_NEMO_GYM_PYTHON to a NeMo Gym 0.7.0 interpreter")
    adapter = InteractionFormatRegistry().get(interaction["format"])
    code = (
        "import importlib.metadata as m,json,sys; "
        "from resources_servers.mcqa.app import MCQARunRequest; "
        "assert m.version('nemo-gym') == '0.7.0'; "
        "obj=MCQARunRequest.model_validate(json.load(sys.stdin)); "
        "print(obj.model_dump_json(exclude_unset=True,exclude_none=True))"
    )
    result = subprocess.run([python, "-c", code], input=json.dumps(adapter.to_nemo(interaction)),
                            capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    assert adapter.from_nemo(json.loads(result.stdout)) == interaction


def test_high_reward_wrong_mcqa_answer_is_not_training_evidence(tmp_path: Path) -> None:
    model = build_mini_pythia_model()
    train_row = _mcqa_row("train", "The capital of France is", "A")
    panels = load_mcqa_panels(
        _write_jsonl(tmp_path / "train.jsonl", [train_row]),
        _write_jsonl(tmp_path / "validation.jsonl", [_mcqa_row("validation", "The capital of Japan is", "B")]),
        _write_jsonl(tmp_path / "holdout.jsonl", [_mcqa_row("holdout", "The capital of Germany is", "A")]),
        tokenizer=model.tokenizer,
    )
    output = tmp_path / "run"
    output.mkdir()
    _fake_artifacts(output / "rollouts.jsonl")
    _write_jsonl(output / "rollouts_materialized_inputs.jsonl", [{"_ng_task_index": 0, "_ng_rollout_index": 0, "task_input": train_row}])
    rollout = json.loads((output / "rollouts.jsonl").read_text())
    rollout["response"]["output"][0]["content"][0]["text"] = " B"
    _write_jsonl(output / "rollouts.jsonl", [rollout])
    _write_test_runner_receipt(output)
    with pytest.raises(NeMoGymTaskError, match="known MCQA target"):
        nemo_gym.import_successful_mcqa_rollouts(
            panels, materialized_inputs_jsonl=output / "rollouts_materialized_inputs.jsonl", rollouts_jsonl=output / "rollouts.jsonl",
        )


def _runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, masked: bool = False, missing: bool = False) -> tuple[NeMoGymRunner, list[list[str]]]:
    from neural_foundry import execution_runner

    tmp_path.mkdir(parents=True, exist_ok=True)
    python = tmp_path / "python"
    gym = tmp_path / "gym"
    python.touch()
    gym.touch()
    config = tmp_path / "mcqa.yaml"
    config.write_text("{}", encoding="utf-8")
    _write_jsonl(tmp_path / "inputs.jsonl", [_mcqa_row("task", "Question?", "A")])
    root = tmp_path / "gym-root"
    root.mkdir()
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        calls.append(command)
        if command[1:2] == ["-c"]:
            return SimpleNamespace(returncode=0, stdout=json.dumps({"nemo_gym": "0.7.0", "python": [3, 13, 14]}), stderr="")
        raise AssertionError("the NeMo evaluation CLI must use supervised Popen")

    class FakeProcess:
        pid = 12345
        returncode = 0

        def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]:
            return b"", b""

        def poll(self) -> int:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            return self.returncode

    def fake_popen(command: list[str], **kwargs: Any) -> FakeProcess:
        calls.append(command)
        output = Path(command[command.index("--output") + 1])
        _fake_artifacts(output, masked=masked, missing=missing)
        return FakeProcess()

    monkeypatch.setattr(nemo_gym.subprocess, "run", fake_run)
    monkeypatch.setattr(nemo_gym.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(execution_runner, "stop_process_tree", lambda _process: None)
    return NeMoGymRunner(python, gym_executable=gym), calls


def test_nemo_gym_runner_uses_pinned_cli_and_writes_complete_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, calls = _runner(tmp_path, monkeypatch)
    output = tmp_path / "run"
    receipt = runner.run_evaluation(
        gym_root=tmp_path / "gym-root",
        config_paths=(tmp_path / "mcqa.yaml",),
        agent="mcqa_simple_agent",
        input_jsonl=tmp_path / "inputs.jsonl",
        output_dir=output,
        model_url="http://127.0.0.1:8000/v1",
        model_name="candidate",
    )

    assert receipt["complete"] is True
    assert receipt["mean_reward"] == 1.0
    assert receipt["execution"]["placement"] == "local"
    assert receipt["execution"]["command"] == receipt["command"]
    assert receipt["execution"]["returncode"] == 0
    assert calls[1][1:5] == ["eval", "run", "--no-serve", "--agent"]
    assert "--config" in calls[1]
    assert (output / "neural_foundry_nemo_gym_receipt.json").is_file()


def test_nemo_gym_runner_preserves_virtualenv_python_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_python = tmp_path / "real-python"
    real_python.touch()
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    try:
        python.symlink_to(real_python)
    except OSError:
        pytest.skip("symlink creation unavailable")
    (bin_dir / "gym").touch()
    calls = []

    def fake_run(command: list[str], **_kwargs: Any) -> Any:
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=json.dumps({"nemo_gym": "0.7.0", "python": [3, 13, 14]}))

    monkeypatch.setattr(nemo_gym.subprocess, "run", fake_run)
    runner = NeMoGymRunner(python)
    assert runner.python == python.absolute()
    assert calls[0][0] == str(python.absolute())
    assert runner.gym == (bin_dir / "gym").resolve()


def test_nemo_reverification_binds_roster_and_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner, calls = _runner(tmp_path, monkeypatch)
    original = tmp_path / "original"
    original.mkdir()
    _fake_artifacts(original / "rollouts.jsonl")
    receipt = runner.run_reverification(
        gym_root=tmp_path / "gym-root", config_paths=(tmp_path / "mcqa.yaml",),
        materialized_inputs=original / "rollouts_materialized_inputs.jsonl",
        rollouts=original / "rollouts.jsonl", output_dir=tmp_path / "reverified", split="validation",
    )
    assert calls[1][2:4] == ["eval", "reverify"]
    assert receipt["complete"] is True
    assert receipt["authority"]["scorer_binding_verified"] is False
    assert receipt["python_version"] == [3, 13, 14]
    assert receipt["result_digest"] == "sha256:" + hashlib.sha256((tmp_path / "reverified" / "reverified.jsonl").read_bytes()).hexdigest()
    original_response = json.loads((original / "rollouts.jsonl").read_text())
    original_response["response"]["output"][0]["content"][0]["text"] = "different"
    _write_jsonl(original / "rollouts.jsonl", [original_response])
    with pytest.raises(NeMoGymError, match="changed a stored response"):
        runner.run_reverification(
            gym_root=tmp_path / "gym-root", config_paths=(tmp_path / "mcqa.yaml",),
            materialized_inputs=original / "rollouts_materialized_inputs.jsonl",
            rollouts=original / "rollouts.jsonl", output_dir=tmp_path / "changed-response", split="validation",
        )
    _write_jsonl(original / "rollouts.jsonl", [])
    with pytest.raises(NeMoGymError, match="complete unique"):
        runner.run_reverification(
            gym_root=tmp_path / "gym-root", config_paths=(tmp_path / "mcqa.yaml",),
            materialized_inputs=original / "rollouts_materialized_inputs.jsonl",
            rollouts=original / "rollouts.jsonl", output_dir=tmp_path / "invalid", split="validation",
        )
    with pytest.raises(NeMoGymError, match="outside custody"):
        runner.run_reverification(
            gym_root=tmp_path / "gym-root", config_paths=(tmp_path / "mcqa.yaml",),
            materialized_inputs=original / "rollouts_materialized_inputs.jsonl",
            rollouts=original / "rollouts.jsonl", output_dir=tmp_path / "holdout", split="holdout",
        )


def test_real_nemo_reverification_of_synthetic_rollout(tmp_path: Path) -> None:
    import os

    python = os.environ.get("NF_NEMO_GYM_PYTHON")
    if not python or os.environ.get("NF_NEMO_GYM_LIVE") != "1":
        pytest.skip("set NF_NEMO_GYM_PYTHON and NF_NEMO_GYM_LIVE=1 for the real server test")
    config = tmp_path / "mcqa.json"
    config.write_text(json.dumps({"mcqa": {"resources_servers": {"mcqa": {"entrypoint": "app.py", "domain": "knowledge"}}}}))
    inputs = _write_jsonl(tmp_path / "inputs.jsonl", [{
        "_ng_task_index": 0, "_ng_rollout_index": 0, "task_input": _mcqa_row("synthetic", "Choose A", "A"),
    }])
    response = {
        "id": "resp-synthetic", "created_at": 0, "model": "synthetic", "object": "response",
        "parallel_tool_calls": False, "tool_choice": "auto", "tools": [],
        "output": [{"id": "msg-synthetic", "type": "message", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": "\\boxed{A}", "annotations": []}]}],
    }
    rollouts = _write_jsonl(tmp_path / "rollouts.jsonl", [{
        "_ng_task_index": 0, "_ng_rollout_index": 0, "response": response,
        "reward": 0.0, "mask_sample": False, "evaluation_completed": True,
    }])
    receipt = NeMoGymRunner(python).run_reverification(
        gym_root=tmp_path, config_paths=(config,), materialized_inputs=inputs, rollouts=rollouts,
        output_dir=tmp_path / "reverified", split="validation", timeout_seconds=180,
    )
    assert receipt["rows"] == 1 and receipt["complete"] is True
    assert receipt["authority"]["scorer_binding_verified"] is False
    output = json.loads((tmp_path / "reverified" / "reverified.jsonl").read_text())
    assert output["reward"] == 1.0


@pytest.mark.parametrize("changed_file", ["inputs.jsonl", "mcqa.yaml"])
def test_nemo_gym_runner_rejects_changed_run_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed_file: str
) -> None:
    runner, _ = _runner(tmp_path, monkeypatch)
    original_popen = nemo_gym.subprocess.Popen

    def mutating_popen(command: list[str], **kwargs: Any) -> Any:
        process = original_popen(command, **kwargs)
        (tmp_path / changed_file).write_text("changed", encoding="utf-8")
        return process

    monkeypatch.setattr(nemo_gym.subprocess, "Popen", mutating_popen)
    output = tmp_path / "run"
    with pytest.raises(NeMoGymError, match="changed during evaluation"):
        runner.run_evaluation(
            gym_root=tmp_path / "gym-root", config_paths=(tmp_path / "mcqa.yaml",),
            agent="mcqa_simple_agent", input_jsonl=tmp_path / "inputs.jsonl",
            output_dir=output, model_url="http://127.0.0.1:8000/v1", model_name="candidate",
        )
    assert not (output / "neural_foundry_nemo_gym_receipt.json").exists()


def test_nemo_gym_runner_does_not_mark_masked_or_missing_rollouts_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, _ = _runner(tmp_path, monkeypatch, masked=True)
    receipt = runner.run_evaluation(
        gym_root=tmp_path / "gym-root", config_paths=(tmp_path / "mcqa.yaml",),
        agent="mcqa_simple_agent", input_jsonl=tmp_path / "inputs.jsonl",
        output_dir=tmp_path / "masked", model_url="http://127.0.0.1:8000/v1", model_name="candidate",
    )
    assert receipt["complete"] is False
    assert receipt["failed_or_masked"] == 1

    runner, _ = _runner(tmp_path / "missing", monkeypatch, missing=True)
    receipt = runner.run_evaluation(
        gym_root=tmp_path / "missing" / "gym-root", config_paths=(tmp_path / "missing" / "mcqa.yaml",),
        agent="mcqa_simple_agent", input_jsonl=tmp_path / "missing" / "inputs.jsonl",
        output_dir=tmp_path / "missing" / "run", model_url="http://127.0.0.1:8000/v1", model_name="candidate",
    )
    assert receipt["complete"] is False
    assert receipt["missing_rollout_keys"] == [[0, 0]]


def test_nemo_gym_runner_pins_version_and_refuses_credentials_in_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, _ = _runner(tmp_path, monkeypatch)
    with pytest.raises(NeMoGymError, match="supports only version 0.7.0"):
        NeMoGymRunner(tmp_path / "python", gym_executable=tmp_path / "gym", expected_version="0.7.0rc0")
    with pytest.raises(NeMoGymError, match="credentials"):
        runner.run_evaluation(
            gym_root=tmp_path / "gym-root", config_paths=(tmp_path / "mcqa.yaml",),
            agent="mcqa_simple_agent", input_jsonl=tmp_path / "inputs.jsonl",
            output_dir=tmp_path / "run", model_url="http://user:secret@127.0.0.1:8000/v1", model_name="candidate",
        )

    python = tmp_path / "old-python"
    gym = tmp_path / "old-gym"
    python.touch()
    gym.touch()
    monkeypatch.setattr(
        nemo_gym.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"nemo_gym": "0.7.0", "python": [3, 12, 0]}),
            stderr="",
        ),
    )
    with pytest.raises(NeMoGymError, match="3.13.14"):
        NeMoGymRunner(python, gym_executable=gym)
