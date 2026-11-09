"""Pinned NeMo Gym evaluation and a narrow MCQA-to-model-training adapter.

NeMo Gym runs in its own Python environment. The Studio process only validates
its version and invokes the public CLI; it never imports NeMo Gym modules.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from .execution_runner import run_local_worker, stop_process_tree
from .gym_data import gym_digest as shared_gym_digest
from .gym_data import gym_spec_document, materialize_public_task_data
from .gym_formats import InteractionFormatRegistry
from .gym_job import GymComponentRegistry, ScenarioSource, is_finite_number
from .gym_service import _document_digest
from .pythia import ReusableTaskGym

NEMO_GYM_VERSION = "0.7.0"
NEMO_GYM_PYTHON_MINIMUM = (3, 13, 14)
MAX_PANEL_BYTES = 256 * 1024 * 1024
MAX_PANEL_ROWS = 100_000
MAX_ROLLOUT_BYTES = 1024 * 1024 * 1024
RUN_RECEIPT_NAME = "neural_foundry_nemo_gym_receipt.json"
_stop_process_tree = stop_process_tree


class NeMoGymError(RuntimeError):
    """NeMo Gym is unavailable, incompatible, or returned incomplete evidence."""


class NeMoGymTaskError(ValueError):
    """A NeMo Gym row is outside the supported one-turn MCQA training profile."""


def compile_public_nemo_dataset(
    task: Any, service: Any, *, split: str, seed: int, output_path: str | Path,
) -> dict[str, Any]:
    if split not in {"train", "validation"}:
        raise NeMoGymError("NeMo dataset compilation only supports public train or validation splits")
    batch = service.materialize_public_scenarios(task.job, split=split, seed=seed)
    converter = InteractionFormatRegistry().get(task.interaction_format)
    rows = []
    for scenario in batch["scenarios"]:
        row = converter.to_nemo(task.project(scenario["input"], None))
        row["metadata"].update({
            "nf_job_digest": task.job.digest, "nf_scenario_manifest_digest": batch["manifest_digest"],
            "nf_scenario_id": scenario["scenario_id"], "nf_reference_digest": scenario["reference_digest"],
            "nf_scorer_digest": task.job.scorer.implementation_digest,
            "nf_verifier_digest": task.job.verifier.implementation_digest,
        })
        rows.append(row)
    payload = b"".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                                  allow_nan=False).encode("utf-8") + b"\n" for row in rows)
    if len(payload) > MAX_PANEL_BYTES:
        raise NeMoGymError("compiled NeMo dataset exceeds its byte limit")
    output = Path(output_path).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    receipt = {
        "kind": "nf_nemo_public_dataset_v1", "job_digest": task.job.digest, "split": split,
        "scenario_manifest_digest": batch["manifest_digest"], "interaction_format": task.interaction_format,
        "projection": task.projection.document(), "scorer": task.job.scorer.to_document(),
        "verifier": task.job.verifier.to_document(), "dataset_digest": _digest_bytes(payload),
        "rows": len(rows), "execution_ready": False, "verifier_bound": False,
    }
    receipt["receipt_digest"] = _document_digest(receipt)
    return receipt


@dataclass(frozen=True)
class NeMoMCQAPanels:
    gym: ReusableTaskGym
    source_digests: Mapping[str, str]
    source_row_counts: Mapping[str, int]
    task_id_digests: Mapping[str, str]
    train_task_ids: tuple[str, ...]
    source_metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_spec_document(self) -> dict[str, Any]:
        return gym_spec_document(self.gym)

    def write_spec(self, path: str | Path) -> Path:
        output = Path(path)
        if output.exists():
            raise FileExistsError(f"Gym spec already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_spec_document(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return output

    def source_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "nf_nemo_mcqa_source_v1",
            "gym_id": self.gym.gym_id,
            "gym_digest": shared_gym_digest(self.gym),
            "adapter_target": f"nemo-gym-mcqa-v{NEMO_GYM_VERSION}",
            "source_panels": {
                split: {
                    "sha256": digest,
                    "rows": self.source_row_counts[split],
                    "task_ids_sha256": self.task_id_digests[split],
                }
                for split, digest in self.source_digests.items()
            },
            "source_metadata": dict(self.source_metadata),
            "training_scope": "single_turn_text_mcqa_one_token_target",
        }


def _digest_bytes(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_jsonl(path: Path, *, max_bytes: int, max_rows: int = MAX_PANEL_ROWS) -> tuple[list[dict[str, Any]], str]:
    size = path.stat().st_size
    if size > max_bytes:
        raise NeMoGymTaskError(f"JSONL file exceeds the byte limit: {path}")
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            digest.update(line)
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise NeMoGymTaskError(f"Invalid JSONL row {line_number} in {path}: {exc}") from exc
            if not isinstance(value, dict):
                raise NeMoGymTaskError(f"JSONL row {line_number} in {path} is not an object")
            rows.append(value)
            if len(rows) > max_rows:
                raise NeMoGymTaskError(f"JSONL file exceeds the row limit: {path}")
    return rows, "sha256:" + digest.hexdigest()


def _single_user_prompt(row: Mapping[str, Any], *, label: str) -> str:
    params = row.get("responses_create_params")
    if not isinstance(params, Mapping):
        raise NeMoGymTaskError(f"{label} is missing responses_create_params")
    inputs = params.get("input")
    if isinstance(inputs, str):
        content = inputs
    elif isinstance(inputs, list) and len(inputs) == 1 and isinstance(inputs[0], Mapping):
        message = inputs[0]
        content = message.get("content")
        if message.get("role") != "user":
            raise NeMoGymTaskError(f"{label} must contain exactly one text user message")
    else:
        raise NeMoGymTaskError(f"{label} must contain exactly one text user message")
    if not isinstance(content, str) or not content.strip():
        raise NeMoGymTaskError(f"{label} must contain exactly one text user message")
    if len(content.encode("utf-8")) > 16_384:
        raise NeMoGymTaskError(f"{label} prompt exceeds the byte limit")
    return content


def _mcqa_rule(row: Mapping[str, Any], tokenizer: Any, *, split: str, index: int) -> tuple[str, str]:
    label = f"{split} row {index}"
    prompt = _single_user_prompt(row, label=label)
    answer = row.get("expected_answer")
    if not isinstance(answer, str) or len(answer) != 1 or not answer.isascii() or not answer.isalpha():
        raise NeMoGymTaskError(f"{label} expected_answer must be one ASCII answer letter")
    options = row.get("options")
    if not isinstance(options, list) or not options or not all(isinstance(item, Mapping) for item in options):
        raise NeMoGymTaskError(f"{label} must include the NeMo MCQA options list")
    option_keys = [key for item in options for key in item]
    if answer not in option_keys:
        raise NeMoGymTaskError(f"{label} expected_answer is absent from options")
    target_ids = tokenizer.encode(" " + answer)
    if len(target_ids) != 1:
        raise NeMoGymTaskError(
            f"{label} answer {answer!r} is not one token for this model; this adapter only supports one-token WSP targets"
        )
    target = tokenizer.id_to_token.get(target_ids[0])
    if target is None or tokenizer.encode(tokenizer.decode_token(target_ids[0])) != target_ids:
        raise NeMoGymTaskError(f"{label} answer token does not round-trip")
    return prompt, target


def _load_mcqa_panels(
    train_jsonl: str | Path,
    validation_jsonl: str | Path,
    holdout_jsonl: str | Path,
    *,
    tokenizer: Any,
    gym_id: str = "gym.nemo.mcqa.v1",
    skills_taught: Sequence[str] = ("multiple_choice_qa",),
) -> NeMoMCQAPanels:
    """Convert three NeMo MCQA JSONL panels into shared WSP and LoRA/SFT data.

    This is data preparation, not Gym execution. Use `NeMoGymRunner` to run a
    model endpoint against NeMo Gym's actual resources server and verifier.
    """
    sources = {
        "train": Path(train_jsonl).resolve(),
        "validation": Path(validation_jsonl).resolve(),
        "holdout": Path(holdout_jsonl).resolve(),
    }
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    digests: dict[str, str] = {}
    counts: dict[str, int] = {}
    task_id_digests: dict[str, str] = {}
    task_ids_by_split: dict[str, tuple[str, ...]] = {}
    rules: dict[str, tuple[tuple[str, str], ...]] = {}
    all_task_ids: dict[str, str] = {}
    for split, path in sources.items():
        rows, digest = _read_jsonl(path, max_bytes=MAX_PANEL_BYTES)
        if not rows:
            raise NeMoGymTaskError(f"{split} MCQA panel is empty")
        rows_by_split[split] = rows
        digests[split] = digest
        counts[split] = len(rows)
        converted: list[tuple[str, str]] = []
        task_ids: set[str] = set()
        ordered_task_ids: list[str] = []
        for index, row in enumerate(rows):
            task_id = _task_id(row)
            if task_id is None or task_id == "":
                raise NeMoGymTaskError(f"{split} row {index} is missing a stable task identity")
            identity = json.dumps(task_id, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if identity in task_ids:
                raise NeMoGymTaskError(f"{split} contains duplicate task identity {task_id!r}")
            if identity in all_task_ids:
                raise NeMoGymTaskError(
                    f"task identity {task_id!r} occurs in both {all_task_ids[identity]} and {split}"
                )
            task_ids.add(identity)
            all_task_ids[identity] = split
            ordered_task_ids.append(identity)
            converted.append(_mcqa_rule(row, tokenizer, split=split, index=index))
        rules[split] = tuple(converted)
        task_ids_by_split[split] = tuple(ordered_task_ids)
        task_id_digests[split] = _digest_bytes(
            json.dumps(sorted(task_ids), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )

    source_identity_digest = _digest_bytes(
        json.dumps(digests, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    source_bound_gym_id = f"{gym_id}:{source_identity_digest.removeprefix('sha256:')[:16]}"
    gym = ReusableTaskGym(
        gym_id=source_bound_gym_id,
        skills_taught=tuple(skills_taught),
        base_rules=rules["train"],
        validation_rules=rules["validation"],
        holdout_rules=rules["holdout"],
        gym_dev_prefixes=(),
        gym_val_prefixes=(),
        holdout_prefixes=(),
    )
    materialize_public_task_data(gym, tokenizer=tokenizer)
    return NeMoMCQAPanels(
        gym=gym,
        source_digests=digests,
        source_row_counts=counts,
        task_id_digests=task_id_digests,
        train_task_ids=task_ids_by_split["train"],
    )


@dataclass(frozen=True)
class NeMoTaskAdapter:
    """Explicit projection from a NeMo task dataset into reusable model-training panels."""

    adapter_id: str
    supports_evaluation: bool = True
    supports_weight_space: bool = False
    supports_lora: bool = False
    supports_sft: bool = False
    supports_dpo: bool = False
    load_panels: Callable[..., NeMoMCQAPanels] | None = None
    generate_panels: Callable[..., NeMoMCQAPanels] | None = None
    train_rollouts: Callable[..., Any] | None = None
    train_preferences: Callable[..., Any] | None = None


class NeMoTaskAdapterRegistry:
    """Extensible task-to-training adapters; evaluation itself remains task-agnostic."""

    def __init__(self) -> None:
        self._adapters: dict[str, NeMoTaskAdapter] = {}
        self.register(
            NeMoTaskAdapter(
                adapter_id="mcqa_single_token",
                supports_evaluation=True,
                supports_weight_space=True,
                supports_lora=True,
                supports_sft=True,
                supports_dpo=True,
                load_panels=_load_mcqa_panels,
                train_rollouts=train_mcqa_rollouts,
                train_preferences=train_mcqa_preferences,
            )
        )
        self.register(
            NeMoTaskAdapter(
                adapter_id="simulated_arithmetic_mcqa",
                supports_evaluation=True,
                supports_weight_space=True,
                supports_lora=True,
                supports_sft=True,
                supports_dpo=True,
                generate_panels=simulate_arithmetic_mcqa,
                train_rollouts=train_mcqa_rollouts,
                train_preferences=train_mcqa_preferences,
            )
        )

    def register(self, adapter: NeMoTaskAdapter) -> None:
        if not adapter.adapter_id.strip():
            raise NeMoGymTaskError("NeMo task adapter ID must be non-empty")
        if adapter.adapter_id in self._adapters:
            raise NeMoGymTaskError(f"NeMo task adapter {adapter.adapter_id!r} is already registered")
        if not adapter.supports_evaluation and not any(
            (adapter.supports_weight_space, adapter.supports_lora, adapter.supports_sft, adapter.supports_dpo)
        ):
            raise NeMoGymTaskError("NeMo task adapter must declare an evaluation or training capability")
        has_training_capability = any(
            (adapter.supports_weight_space, adapter.supports_lora, adapter.supports_sft, adapter.supports_dpo)
        )
        if has_training_capability and not (
            callable(adapter.load_panels) or callable(adapter.generate_panels)
        ):
            raise NeMoGymTaskError("Training-capable NeMo adapters must load or generate task panels")
        if (adapter.supports_lora or adapter.supports_sft) and not callable(adapter.train_rollouts):
            raise NeMoGymTaskError("LoRA/SFT-capable NeMo adapters must provide a rollout trainer")
        if adapter.supports_dpo and not callable(adapter.train_preferences):
            raise NeMoGymTaskError("DPO-capable NeMo adapters must provide a preference trainer")
        self._adapters[adapter.adapter_id] = adapter

    def get(self, adapter_id: str) -> NeMoTaskAdapter:
        try:
            return self._adapters[adapter_id]
        except KeyError as exc:
            raise NeMoGymTaskError(
                f"Unknown NeMo task adapter {adapter_id!r}; available: {sorted(self._adapters)}"
            ) from exc

    def list_adapters(self) -> dict[str, dict[str, bool]]:
        return {
            key: {
                "weight_space": adapter.supports_weight_space,
                "lora": adapter.supports_lora,
                "sft": adapter.supports_sft,
                "dpo": adapter.supports_dpo,
                "evaluation": adapter.supports_evaluation,
            }
            for key, adapter in self._adapters.items()
        }

    def load_panels(self, adapter_id: str, **kwargs: Any) -> NeMoMCQAPanels:
        adapter = self.get(adapter_id)
        if not any((adapter.supports_weight_space, adapter.supports_lora, adapter.supports_sft, adapter.supports_dpo)):
            raise NeMoGymTaskError(f"NeMo adapter {adapter_id!r} is evaluation-only")
        assert adapter.load_panels is not None
        return adapter.load_panels(**kwargs)

    def generate_panels(self, adapter_id: str, **kwargs: Any) -> NeMoMCQAPanels:
        adapter = self.get(adapter_id)
        if adapter.generate_panels is None:
            raise NeMoGymTaskError(f"NeMo adapter {adapter_id!r} does not provide deterministic task generation")
        return adapter.generate_panels(**kwargs)

    def train_from_rollouts(self, adapter_id: str, **kwargs: Any) -> Any:
        adapter = self.get(adapter_id)
        if not (adapter.supports_lora or adapter.supports_sft) or adapter.train_rollouts is None:
            raise NeMoGymTaskError(f"NeMo adapter {adapter_id!r} has no rollout-training implementation")
        return adapter.train_rollouts(**kwargs)

    def train_from_preferences(self, adapter_id: str, **kwargs: Any) -> Any:
        adapter = self.get(adapter_id)
        if not adapter.supports_dpo or adapter.train_preferences is None:
            raise NeMoGymTaskError(f"NeMo adapter {adapter_id!r} does not support DPO")
        return adapter.train_preferences(**kwargs)


def load_mcqa_panels(
    train_jsonl: str | Path,
    validation_jsonl: str | Path,
    holdout_jsonl: str | Path,
    *,
    tokenizer: Any,
    gym_id: str = "gym.nemo.mcqa.v1",
    skills_taught: Sequence[str] = ("multiple_choice_qa",),
) -> NeMoMCQAPanels:
    """Convenience entry point for the registered one-token MCQA adapter."""
    return NeMoTaskAdapterRegistry().load_panels(
        "mcqa_single_token",
        train_jsonl=train_jsonl,
        validation_jsonl=validation_jsonl,
        holdout_jsonl=holdout_jsonl,
        tokenizer=tokenizer,
        gym_id=gym_id,
        skills_taught=skills_taught,
    )


def simulate_arithmetic_mcqa(
    output_dir: str | Path,
    *,
    tokenizer: Any,
    seed: int = 42,
    train_size: int = 24,
    validation_size: int = 12,
    holdout_size: int = 12,
    gym_id: str = "gym.simulated.single_digit_arithmetic.v1",
) -> NeMoMCQAPanels:
    """Generate a reproducible no-source-data MCQA Gym with disjoint operand panels."""
    sizes = {"train": train_size, "validation": validation_size, "holdout": holdout_size}
    if type(seed) is not int or seed < 0:
        raise NeMoGymTaskError("simulation seed must be a non-negative integer")
    if any(type(size) is not int or size < 1 for size in sizes.values()):
        raise NeMoGymTaskError("simulation panel sizes must be positive integers")
    operand_pairs = [(left, right) for left in range(10) for right in range(10) if left + right < 10]
    requested = sum(sizes.values())
    if requested > len(operand_pairs):
        raise NeMoGymTaskError(f"requested {requested} scenarios, but this generator has {len(operand_pairs)} unique sums")
    random.Random(seed).shuffle(operand_pairs)
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Simulation output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    output.chmod(0o700)
    offset = 0
    panel_paths: dict[str, Path] = {}
    generator_digest = _digest_bytes(
        b"neural_foundry.simulated_single_digit_arithmetic.v1|operands=0..9|sum<10"
    )
    for split, size in sizes.items():
        split_rows: list[dict[str, Any]] = []
        for left, right in operand_pairs[offset : offset + size]:
            task_seed = f"{seed}:{split}:{left}:{right}"
            chooser = random.Random(task_seed)
            answer = left + right
            distractors = chooser.sample([value for value in range(10) if value != answer], 3)
            choices = [(letter, value) for letter, value in zip("ABCD", [answer, *distractors], strict=True)]
            chooser.shuffle(choices)
            label = next(letter for letter, value in choices if value == answer)
            options = [{letter: str(value)} for letter, value in choices]
            option_text = "\n".join(f"{letter}: {value}" for letter, value in choices)
            row_id = _digest_bytes(f"{generator_digest}:{task_seed}".encode("ascii"))
            prompt = f"Task: Number {left} plus {right} is\n{option_text}\nAnswer with one letter."
            split_rows.append({
                "uuid": row_id,
                "task_id": row_id,
                "metadata": {"generator": "single_digit_arithmetic.v1", "seed": seed, "split": split},
                "responses_create_params": {"input": [{"role": "user", "content": prompt}]},
                "options": options,
                "expected_answer": label,
            })
        panel_path = output / f"{split}.jsonl"
        with panel_path.open("x", encoding="utf-8") as stream:
            for row in split_rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        panel_paths[split] = panel_path
        offset += size
    panels = load_mcqa_panels(
        panel_paths["train"],
        panel_paths["validation"],
        panel_paths["holdout"],
        tokenizer=tokenizer,
        gym_id=gym_id,
        skills_taught=("single_digit_addition", "multiple_choice_reasoning"),
    )
    return replace(
        panels,
        source_metadata={
            "kind": "deterministic_simulator",
            "generator_id": "neural_foundry.single_digit_arithmetic.v1",
            "generator_digest": generator_digest,
            "seed": seed,
            "split_sizes": sizes,
            "unique_scenario_count": requested,
        },
    )


def register_nemo_mcqa_scenario_sources(
    components: GymComponentRegistry,
    panels: NeMoMCQAPanels,
    *,
    origin: Literal["real", "simulated"],
) -> tuple[ScenarioSource, ...]:
    """Bind NeMo train/validation/holdout panels to JobSpec scenario providers."""
    if origin not in {"real", "simulated"}:
        raise NeMoGymTaskError("NeMo scenario origin must be explicitly real or simulated")
    provider_id = "neural_foundry.nemo_mcqa_panels"
    provider_version = "1.0.0"
    provider_digest = _sha256_file(Path(__file__))
    rules_by_split = {
        "train": panels.gym.base_rules,
        "validation": panels.gym.validation_rules,
        "holdout": panels.gym.holdout_rules,
    }
    sources = tuple(
        ScenarioSource(
            source_id=f"nemo-mcqa-{split}:{panels.source_digests[split][7:23]}",
            provider_id=provider_id,
            provider_version=provider_version,
            provider_digest=provider_digest,
            source_digest=panels.source_digests[split],
            origin=origin,
            split_roles=(split,),
            parameters={"split": split},
        )
        for split in ("train", "validation", "holdout")
    )
    source_by_id = {source.source_id: source for source in sources}

    def provide(
        *,
        parameters: Mapping[str, Any],
        source_id: str,
        source_digest: str,
        origin: str,
        split: str,
        seed: int,
    ) -> list[dict[str, Any]]:
        source = source_by_id.get(source_id)
        if (
            source is None
            or parameters != {"split": split}
            or source.source_digest != source_digest
            or source.origin != origin
            or split not in rules_by_split
            or type(seed) is not int
            or seed < 0
        ):
            raise NeMoGymTaskError("NeMo MCQA scenario request differs from the frozen source declaration")
        return [
            {
                "scenario_id": f"{split}-{_digest_bytes(f'{source_digest}:{index}'.encode('ascii'))[7:31]}",
                "input": {"prompt": prompt},
                "reference": {"expected_token": target, "task_index": index},
            }
            for index, (prompt, target) in enumerate(rules_by_split[split])
        ]

    components.register("scenario_provider", provider_id, provider_version, provider_digest, provide)
    return sources


@dataclass(frozen=True)
class NeMoSFTDataset:
    examples: tuple[tuple[str, str], ...]
    dataset_digest: str
    source_rollout_digest: str
    source_input_digest: str
    source_runner_receipt_digest: str
    accepted_rollouts: int
    excluded_low_reward_rollouts: int


@dataclass(frozen=True)
class NeMoPreferenceDataset:
    pairs: tuple[tuple[str, str, str], ...]
    dataset_digest: str
    source_rollout_digest: str
    source_input_digest: str
    source_runner_receipt_digest: str
    task_pairs: int


@dataclass(frozen=True)
class NeMoRolloutTrainingRun:
    dataset: NeMoSFTDataset
    fit: Any
    base_model_digest: str
    candidate_model_digest: str
    validation_accuracy: float
    validation_examples: int
    validation_digest: str


@dataclass(frozen=True)
class NeMoPreferenceTrainingRun:
    dataset: NeMoPreferenceDataset
    fit: Any
    base_model_digest: str
    candidate_model_digest: str
    validation_accuracy: float
    validation_examples: int
    validation_digest: str


def _materialized_prompt(row: Mapping[str, Any], *, label: str) -> str:
    task_input = row.get("task_input", row)
    if not isinstance(task_input, Mapping):
        raise NeMoGymError(f"{label} task_input is malformed")
    return _single_user_prompt(task_input, label=label)


def import_successful_mcqa_rollouts(
    panels: NeMoMCQAPanels,
    *,
    materialized_inputs_jsonl: str | Path,
    rollouts_jsonl: str | Path,
    minimum_reward: float = 1.0,
) -> NeMoSFTDataset:
    """Convert complete, unmasked, high-reward train-panel MCQA rollouts to sequence-SFT pairs.

    Validation and holdout rows are rejected by identity/index before training data
    is produced. Failed, masked, ungraded, and partial runs are never used.
    """
    if not is_finite_number(minimum_reward) or not 0.0 <= minimum_reward <= 1.0:
        raise NeMoGymTaskError("minimum_reward must be finite and within [0, 1]")
    input_path = Path(materialized_inputs_jsonl).resolve()
    rollout_path = Path(rollouts_jsonl).resolve()
    expected_input_path = rollout_path.parent / "rollouts_materialized_inputs.jsonl"
    expected_rollout_path = rollout_path.parent / "rollouts.jsonl"
    if input_path != expected_input_path or rollout_path != expected_rollout_path:
        raise NeMoGymError("Training must consume the exact standard artifacts health-checked in one NeMo run directory")
    run_report, runner_receipt_digest = _verify_runner_receipt(rollout_path.parent)
    materialized, input_digest = _read_jsonl(input_path, max_bytes=MAX_ROLLOUT_BYTES)
    rollouts, rollout_digest = _read_jsonl(rollout_path, max_bytes=MAX_ROLLOUT_BYTES)
    _require_imported_artifact_digests(
        rollout_path.parent, run_report, input_digest, rollout_digest
    )
    inputs_by_key = {_rollout_key(row, label="materialized input"): row for row in materialized}
    rollouts_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rollouts:
        key = _rollout_key(row, label="rollout")
        if key in rollouts_by_key:
            raise NeMoGymError(f"duplicate NeMo rollout identity {key!r}")
        rollouts_by_key[key] = row
    if len(inputs_by_key) != len(materialized) or set(inputs_by_key) != set(rollouts_by_key):
        raise NeMoGymError("materialized inputs and rollouts do not form a complete one-to-one panel")

    examples: list[tuple[str, str]] = []
    excluded = 0
    for key in sorted(rollouts_by_key):
        task_index, _rollout_index = key
        if task_index >= len(panels.gym.base_rules):
            raise NeMoGymTaskError(
                f"rollout task index {task_index} is outside the declared MCQA training panel"
            )
        expected_prompt, expected_target = panels.gym.base_rules[task_index]
        materialized_row = inputs_by_key[key]
        task_input = materialized_row.get("task_input", materialized_row)
        if not isinstance(task_input, Mapping):
            raise NeMoGymError(f"materialized input {key} has malformed task data")
        task_id = _task_id(task_input)
        task_identity = json.dumps(task_id, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if task_id is None or task_identity != panels.train_task_ids[task_index]:
            raise NeMoGymTaskError(f"materialized input {key} does not match the frozen NeMo train task ID")
        actual_prompt = _single_user_prompt(task_input, label=f"materialized input {key}")
        if actual_prompt != expected_prompt:
            raise NeMoGymTaskError(
                f"rollout {key} prompt does not match the frozen training row {panels.train_task_ids[task_index]!r}"
            )
        row = rollouts_by_key[key]
        reward = row.get("reward")
        if (
            row.get("mask_sample") is True
            or row.get("evaluation_completed") is False
            or row.get("failure_kind") is not None
            or isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not math.isfinite(float(reward))
        ):
            raise NeMoGymError(f"rollout {key} is masked, failed, or ungraded")
        if float(reward) < minimum_reward:
            excluded += 1
            continue
        response = _assistant_text(row.get("response"), label=f"rollout {key}")
        if response.strip() != expected_target.removeprefix("Ġ").strip():
            raise NeMoGymTaskError(f"high-reward rollout {key} does not answer its known MCQA target")
        if not response.strip():
            raise NeMoGymError(f"successful rollout {key} contains no assistant text")
        if len(response.encode("utf-8")) > 65_536:
            raise NeMoGymTaskError(f"successful rollout {key} response exceeds the training text bound")
        examples.append((expected_prompt, response))
    if not examples:
        raise NeMoGymTaskError("no training examples meet the reward threshold")
    digest_payload = {
        "schema_version": 1,
        "kind": "nf_nemo_mcqa_sft_dataset_v1",
        "source_gym_digest": panels.source_manifest()["gym_digest"],
        "source_runner_receipt_digest": runner_receipt_digest,
        "source_input_digest": input_digest,
        "source_rollout_digest": rollout_digest,
        "minimum_reward": minimum_reward,
        "examples": examples,
    }
    dataset_digest = _digest_bytes(
        json.dumps(digest_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return NeMoSFTDataset(
        examples=tuple(examples),
        dataset_digest=dataset_digest,
        source_rollout_digest=rollout_digest,
        source_input_digest=input_digest,
        source_runner_receipt_digest=runner_receipt_digest,
        accepted_rollouts=len(examples),
        excluded_low_reward_rollouts=excluded,
    )


def import_mcqa_preferences(
    panels: NeMoMCQAPanels,
    *,
    materialized_inputs_jsonl: str | Path,
    rollouts_jsonl: str | Path,
    minimum_reward_margin: float = 0.1,
) -> NeMoPreferenceDataset:
    """Build chosen/rejected DPO pairs from repeated, complete rollouts on training tasks only."""
    if not is_finite_number(minimum_reward_margin) or not 0.0 < minimum_reward_margin <= 1.0:
        raise NeMoGymTaskError("minimum_reward_margin must be finite and within (0, 1]")
    input_path = Path(materialized_inputs_jsonl).resolve()
    rollout_path = Path(rollouts_jsonl).resolve()
    if (
        input_path != rollout_path.parent / "rollouts_materialized_inputs.jsonl"
        or rollout_path != rollout_path.parent / "rollouts.jsonl"
    ):
        raise NeMoGymError("DPO must use the exact standard artifacts health-checked in one NeMo run directory")
    run_report, runner_receipt_digest = _verify_runner_receipt(rollout_path.parent)
    materialized, input_digest = _read_jsonl(input_path, max_bytes=MAX_ROLLOUT_BYTES)
    rollouts, rollout_digest = _read_jsonl(rollout_path, max_bytes=MAX_ROLLOUT_BYTES)
    _require_imported_artifact_digests(
        rollout_path.parent, run_report, input_digest, rollout_digest
    )
    inputs_by_key = {_rollout_key(row, label="materialized input"): row for row in materialized}
    if len(inputs_by_key) != len(materialized):
        raise NeMoGymError("DPO materialized inputs contain duplicate rollout identity")
    rows_by_task: dict[int, list[tuple[float, str]]] = {}
    seen_rollouts: set[tuple[int, int]] = set()
    for row in rollouts:
        key = _rollout_key(row, label="rollout")
        if key in seen_rollouts or key not in inputs_by_key:
            raise NeMoGymError(f"duplicate or unpaired NeMo rollout identity {key!r}")
        seen_rollouts.add(key)
        task_index, _rollout_index = key
        if task_index >= len(panels.gym.base_rules):
            raise NeMoGymTaskError(f"DPO rollout task index {task_index} is outside the training panel")
        source_row = inputs_by_key[key]
        task_input = source_row.get("task_input", source_row)
        if not isinstance(task_input, Mapping):
            raise NeMoGymError(f"materialized input {key} has malformed task data")
        task_id = _task_id(task_input)
        task_identity = json.dumps(task_id, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if task_id is None or task_identity != panels.train_task_ids[task_index]:
            raise NeMoGymTaskError(f"DPO rollout {key} is not bound to a frozen training task ID")
        prompt = _single_user_prompt(task_input, label=f"materialized input {key}")
        if prompt != panels.gym.base_rules[task_index][0]:
            raise NeMoGymTaskError(f"DPO rollout {key} prompt differs from its frozen training task")
        reward = row.get("reward")
        if (
            row.get("mask_sample") is True
            or row.get("evaluation_completed") is False
            or row.get("failure_kind") is not None
            or isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not math.isfinite(float(reward))
        ):
            raise NeMoGymError(f"DPO rollout {key} is masked, failed, or ungraded")
        response = _assistant_text(row.get("response"), label=f"DPO rollout {key}")
        if not response.strip():
            raise NeMoGymTaskError(f"DPO rollout {key} has an empty response")
        rows_by_task.setdefault(task_index, []).append((float(reward), response))
    if seen_rollouts != set(inputs_by_key):
        raise NeMoGymError("DPO materialized inputs and rollouts do not form a complete panel")

    pairs: list[tuple[str, str, str]] = []
    for task_index, task_responses in sorted(rows_by_task.items()):
        chosen = max(task_responses, key=lambda item: item[0])
        rejected = min(task_responses, key=lambda item: item[0])
        if chosen[0] - rejected[0] < minimum_reward_margin or chosen[1] == rejected[1]:
            continue
        if chosen[1].strip() != panels.gym.base_rules[task_index][1].removeprefix("Ġ").strip():
            raise NeMoGymTaskError("chosen MCQA preference does not match the known target")
        pairs.append((panels.gym.base_rules[task_index][0], chosen[1], rejected[1]))
    if not pairs:
        raise NeMoGymTaskError("no within-task rollout pairs meet the DPO reward margin")
    dataset_digest = _digest_bytes(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "nf_nemo_mcqa_dpo_dataset_v1",
                "gym_digest": panels.source_manifest()["gym_digest"],
                "source_runner_receipt_digest": runner_receipt_digest,
                "source_input_digest": input_digest,
                "source_rollout_digest": rollout_digest,
                "minimum_reward_margin": minimum_reward_margin,
                "pairs": pairs,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return NeMoPreferenceDataset(
        pairs=tuple(pairs),
        dataset_digest=dataset_digest,
        source_rollout_digest=rollout_digest,
        source_input_digest=input_digest,
        source_runner_receipt_digest=runner_receipt_digest,
        task_pairs=len(pairs),
    )


def train_mcqa_rollouts(
    *,
    model: Any,
    panels: NeMoMCQAPanels,
    materialized_inputs_jsonl: str | Path,
    rollouts_jsonl: str | Path,
    config: Any,
    minimum_reward: float = 1.0,
    retention_guard: Sequence[str] | None = None,
    training_device: str = "cpu",
) -> NeMoRolloutTrainingRun:
    """Train LoRA/SFT on successful full responses and score the distinct validation panel."""
    from .baselines import accuracy, fit_sequence_baseline, merged_model
    from .edit_plan import tensor_state_digest
    from .gym_data import materialize_public_task_data

    if config.kind not in {"lora", "sft"}:
        raise NeMoGymTaskError("NeMo trajectory training supports only LoRA or full SFT recipes")
    if not panels.gym.holdout_rules:
        raise NeMoGymTaskError("NeMo training requires a separately declared holdout panel")
    dataset = import_successful_mcqa_rollouts(
        panels,
        materialized_inputs_jsonl=materialized_inputs_jsonl,
        rollouts_jsonl=rollouts_jsonl,
        minimum_reward=minimum_reward,
    )
    fit = fit_sequence_baseline(
        model,
        config,
        examples=dataset.examples,
        retention_guard=tuple(retention_guard if retention_guard is not None else panels.gym.retention_prompts),
        training_device=training_device,
    )
    candidate = merged_model(model, fit.merged_weights)
    task_data = materialize_public_task_data(panels.gym, tokenizer=model.tokenizer)
    validation_accuracy = accuracy(candidate, task_data.validation_items)
    return NeMoRolloutTrainingRun(
        dataset=dataset,
        fit=fit,
        base_model_digest=tensor_state_digest(model.tensors),
        candidate_model_digest=tensor_state_digest(candidate.tensors),
        validation_accuracy=validation_accuracy,
        validation_examples=len(task_data.validation_items),
        validation_digest=task_data.validation_digest,
    )


def train_mcqa_preferences(
    *,
    model: Any,
    panels: NeMoMCQAPanels,
    materialized_inputs_jsonl: str | Path,
    rollouts_jsonl: str | Path,
    config: Any,
    minimum_reward_margin: float = 0.1,
    retention_guard: Sequence[str] | None = None,
    training_device: str = "cpu",
) -> NeMoPreferenceTrainingRun:
    """Train DPO LoRA on within-task reward preferences and score held-out task rules."""
    from .baselines import accuracy, fit_preference_dpo, merged_model
    from .edit_plan import tensor_state_digest
    from .gym_data import materialize_public_task_data

    dataset = import_mcqa_preferences(
        panels,
        materialized_inputs_jsonl=materialized_inputs_jsonl,
        rollouts_jsonl=rollouts_jsonl,
        minimum_reward_margin=minimum_reward_margin,
    )
    fit = fit_preference_dpo(
        model,
        config,
        pairs=dataset.pairs,
        retention_guard=(
            tuple(retention_guard) if retention_guard is not None else panels.gym.retention_prompts
        ),
        training_device=training_device,
    )
    candidate = merged_model(model, fit.merged_weights)
    task_data = materialize_public_task_data(panels.gym, tokenizer=model.tokenizer)
    return NeMoPreferenceTrainingRun(
        dataset=dataset,
        fit=fit,
        base_model_digest=tensor_state_digest(model.tensors),
        candidate_model_digest=tensor_state_digest(candidate.tensors),
        validation_accuracy=accuracy(candidate, task_data.validation_items),
        validation_examples=len(task_data.validation_items),
        validation_digest=task_data.validation_digest,
    )


def load_nemo_sft_candidate(model: Any, run_directory: str | Path) -> tuple[Any, dict[str, Any]]:
    """Reload a NeMo-trained candidate only when its weights and base/candidate digests agree."""
    import torch

    from .baselines import merged_model, weights_digest
    from .edit_plan import tensor_state_digest

    root = Path(run_directory).resolve()
    receipt_path = root / "training_receipt.json"
    if not receipt_path.is_file():
        raise NeMoGymError("NeMo training receipt is missing")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NeMoGymError(f"NeMo training receipt is unreadable: {exc}") from exc
    if receipt.get("schema_version") != 1 or receipt.get("kind") not in {
        "nf_nemo_mcqa_sequence_sft_v1",
        "nf_nemo_mcqa_dpo_v1",
    }:
        raise NeMoGymError("Unsupported NeMo training receipt schema or kind")
    if tensor_state_digest(model.tensors) != receipt.get("base_model_digest"):
        raise NeMoGymError("NeMo SFT artifact was trained from a different base model")
    weights_name = receipt.get("weights_path")
    if not isinstance(weights_name, str) or Path(weights_name).name != weights_name:
        raise NeMoGymError("NeMo SFT receipt has an unsafe weights path")
    weights_path = (root / weights_name).resolve()
    if not weights_path.is_relative_to(root) or not weights_path.is_file():
        raise NeMoGymError("NeMo SFT weights are missing or escape the run directory")
    try:
        weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise NeMoGymError(f"NeMo SFT weights could not be loaded safely: {exc}") from exc
    if not isinstance(weights, dict) or not weights:
        raise NeMoGymError("NeMo SFT weights must be a non-empty tensor map")
    for name, value in weights.items():
        if (
            name not in model.tensors
            or not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.dtype != torch.float64
            or tuple(value.shape) != tuple(model.tensors[name].shape)
            or not bool(torch.isfinite(value).all().item())
        ):
            raise NeMoGymError(f"NeMo SFT weight {name!r} violates the candidate tensor contract")
    if weights_digest(weights) != receipt.get("weights_digest"):
        raise NeMoGymError("NeMo SFT weight-map digest does not match the receipt")
    candidate = merged_model(model, weights)
    if tensor_state_digest(candidate.tensors) != receipt.get("candidate_model_digest"):
        raise NeMoGymError("Reloaded NeMo SFT candidate digest does not match the receipt")
    return candidate, receipt


def _rollout_key(row: Mapping[str, Any], *, label: str) -> tuple[int, int]:
    task_index, rollout_index = row.get("_ng_task_index"), row.get("_ng_rollout_index")
    if type(task_index) is not int or task_index < 0 or type(rollout_index) is not int or rollout_index < 0:
        raise NeMoGymError(f"{label} is missing valid NeMo task/rollout indices")
    return task_index, rollout_index


def _task_id(row: Mapping[str, Any]) -> Any:
    for field_name in ("uuid", "task_id", "id"):
        value = row.get(field_name)
        if value is not None:
            return value
    return None


def _same_reverified_response(original: Any, result: Any) -> bool:
    if isinstance(original, dict):
        return (isinstance(result, dict) and set(original) <= set(result)
                and all(result[key] is None for key in set(result) - set(original))
                and all(_same_reverified_response(value, result[key]) for key, value in original.items()))
    if isinstance(original, list):
        return (isinstance(result, list) and len(original) == len(result)
                and all(_same_reverified_response(left, right) for left, right in zip(original, result, strict=True)))
    if isinstance(original, bool) or isinstance(result, bool):
        return type(original) is type(result) and original == result
    return original == result


def _assistant_text(response: Any, *, label: str) -> str:
    if not isinstance(response, Mapping) or not isinstance(response.get("output"), list):
        raise NeMoGymError(f"{label} has no Responses API output")
    output = response["output"]
    if len(output) != 1 or not isinstance(output[0], Mapping):
        raise NeMoGymTaskError(f"{label} is not a single-turn text-only response")
    message = output[0]
    content = message.get("content")
    if (
        message.get("type") != "message"
        or message.get("role") != "assistant"
        or not isinstance(content, list)
        or len(content) != 1
        or not isinstance(content[0], Mapping)
        or content[0].get("type") != "output_text"
        or not isinstance(content[0].get("text"), str)
    ):
        raise NeMoGymTaskError(f"{label} contains tool, multi-part, or non-text trajectory data")
    return content[0]["text"]


def inspect_run_artifacts(output_dir: str | Path) -> dict[str, Any]:
    """Join NeMo outputs by stable rollout indices and report whether evidence is complete."""
    root = Path(output_dir).resolve()
    rollout_path = root / "rollouts.jsonl"
    input_path = root / "rollouts_materialized_inputs.jsonl"
    failure_path = root / "rollouts_failures.jsonl"
    for path in (rollout_path, input_path):
        if not path.is_file():
            raise NeMoGymError(f"NeMo Gym omitted required artifact {path.name}")
    materialized, inputs_digest = _read_jsonl(input_path, max_bytes=MAX_ROLLOUT_BYTES)
    rollouts, rollouts_digest = _read_jsonl(rollout_path, max_bytes=MAX_ROLLOUT_BYTES)
    input_by_key = {_rollout_key(row, label="materialized input"): row for row in materialized}
    roster_path = root / "requested_roster.json"
    roster_verified = False
    expected_keys: set[tuple[int, int]] = set()
    if roster_path.is_file():
        if roster_path.stat().st_size > 8 * 1024 * 1024:
            raise NeMoGymError("requested rollout roster exceeds its byte limit")
        roster = json.loads(roster_path.read_bytes())
        rows = roster.get("expected_keys") if isinstance(roster, dict) else None
        if not isinstance(roster, dict) or roster.get("kind") != "nf_nemo_requested_roster_v1" or not isinstance(rows, list):
            raise NeMoGymError("requested rollout roster is malformed")
        for key in rows:
            if not isinstance(key, list) or len(key) != 2 or any(type(index) is not int or index < 0 for index in key):
                raise NeMoGymError("requested rollout key is malformed")
            expected_keys.add(tuple(key))
        if len(expected_keys) != len(rows):
            raise NeMoGymError("requested rollout keys are duplicated")
        roster_verified = bool(expected_keys) and expected_keys == set(input_by_key)
    rollout_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rollouts:
        key = _rollout_key(row, label="rollout")
        if key in rollout_by_key:
            raise NeMoGymError(f"NeMo Gym emitted duplicate rollout identity {key!r}")
        rollout_by_key[key] = row
    if len(input_by_key) != len(materialized):
        raise NeMoGymError("NeMo Gym emitted duplicate materialized-input identity")
    unknown = sorted(set(rollout_by_key) - set(input_by_key))
    if unknown:
        raise NeMoGymError(f"NeMo Gym emitted rollouts with no materialized input: {unknown[:8]}")

    failed = 0
    rewards: list[float] = []
    for key, row in rollout_by_key.items():
        reward = row.get("reward")
        is_failed = (
            row.get("mask_sample") is not False
            or row.get("evaluation_completed") is not True
            or row.get("failure_kind") is not None
            or not is_finite_number(reward)
            or not 0 <= reward <= 1
        )
        if is_failed:
            failed += 1
        else:
            rewards.append(float(reward))
    failure_rows: list[dict[str, Any]] = []
    failure_digest = None
    if failure_path.is_file():
        failure_rows, failure_digest = _read_jsonl(failure_path, max_bytes=MAX_ROLLOUT_BYTES)
    missing = sorted(set(input_by_key) - set(rollout_by_key))
    complete = roster_verified and bool(materialized) and not missing and failed == 0 and not failure_rows
    return {
        "schema_version": 1,
        "kind": "nf_nemo_gym_evaluation_v1",
        "nemo_gym_version": NEMO_GYM_VERSION,
        "execution_scope": "external_ne_mo_gym_cli_model_endpoint",
        "complete": complete,
        "requested_roster_verified": roster_verified,
        "materialized_inputs": len(materialized),
        "rollouts": len(rollouts),
        "failed_or_masked": failed,
        "failure_sidecar_rows": len(failure_rows),
        "missing_rollout_keys": [list(key) for key in missing[:100]],
        "mean_reward": math.fsum(value / len(rewards) for value in rewards) if rewards else None,
        "scored_rewards": len(rewards),
        "artifact_digests": {
            "materialized_inputs": inputs_digest,
            "rollouts": rollouts_digest,
            "failures": failure_digest,
            "requested_roster": _sha256_file(roster_path) if roster_path.is_file() else None,
        },
        "authority": {
            "native_family_ledger": False,
            "sealed_target_custody": False,
            "promotion_authorized": False,
        },
    }


def _is_sha256_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _verify_runner_receipt(output_dir: str | Path) -> tuple[dict[str, Any], str]:
    root = Path(output_dir).resolve()
    report = inspect_run_artifacts(root)
    if not report["complete"]:
        raise NeMoGymError("NeMo runner receipt cannot certify an incomplete or masked rollout run")
    receipt_path = root / RUN_RECEIPT_NAME
    try:
        raw_receipt = receipt_path.read_bytes()
        receipt = json.loads(raw_receipt)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NeMoGymError(f"NeMo Gym runner receipt is missing or unreadable: {exc}") from exc
    if not isinstance(receipt, dict):
        raise NeMoGymError("NeMo Gym runner receipt must be a JSON object")
    report_fields = (
        "schema_version",
        "kind",
        "nemo_gym_version",
        "execution_scope",
        "complete",
        "requested_roster_verified",
        "materialized_inputs",
        "rollouts",
        "failed_or_masked",
        "failure_sidecar_rows",
        "missing_rollout_keys",
        "mean_reward",
        "scored_rewards",
        "artifact_digests",
        "authority",
    )
    if any(receipt.get(field) != report[field] for field in report_fields):
        raise NeMoGymError("NeMo Gym runner receipt does not match the current rollout artifacts")
    if (
        not isinstance(receipt.get("agent"), str)
        or not receipt["agent"].strip()
        or not isinstance(receipt.get("model_name"), str)
        or not receipt["model_name"].strip()
        or not isinstance(receipt.get("model_endpoint"), str)
        or not receipt["model_endpoint"].strip()
        or not _is_sha256_digest(receipt.get("input_digest"))
        or not isinstance(receipt.get("config_digests"), dict)
        or not receipt["config_digests"]
        or any(not _is_sha256_digest(value) for value in receipt["config_digests"].values())
        or not isinstance(receipt.get("command"), list)
        or not receipt["command"]
        or any(not isinstance(value, str) for value in receipt["command"])
    ):
        raise NeMoGymError("NeMo Gym runner receipt is missing its execution identity")
    return report, _digest_bytes(raw_receipt)


def _require_imported_artifact_digests(
    output_dir: str | Path,
    report: Mapping[str, Any],
    input_digest: str,
    rollout_digest: str,
) -> None:
    expected = report["artifact_digests"]
    failure_path = Path(output_dir) / "rollouts_failures.jsonl"
    failure_digest = _sha256_file(failure_path) if failure_path.is_file() else None
    if (
        input_digest != expected["materialized_inputs"]
        or rollout_digest != expected["rollouts"]
        or failure_digest != expected["failures"]
    ):
        raise NeMoGymError("NeMo rollout artifacts changed while being imported for training")


class NeMoGymRunner:
    """Run pinned NeMo Gym CLI evaluations against an already-running model endpoint."""

    def __init__(
        self,
        python_executable: str | Path,
        *,
        gym_executable: str | Path | None = None,
        expected_version: str = NEMO_GYM_VERSION,
    ) -> None:
        if expected_version != NEMO_GYM_VERSION:
            raise NeMoGymError(f"NeMo Gym adapter supports only version {NEMO_GYM_VERSION}")
        self.python = Path(python_executable).expanduser().absolute()
        if not self.python.is_file():
            raise NeMoGymError(f"NeMo Gym Python interpreter not found: {self.python}")
        if gym_executable is None:
            name = "gym.exe" if os.name == "nt" else "gym"
            candidate = self.python.parent / name
            if not candidate.is_file():
                candidate = self.python.parent.parent / "Scripts" / "gym.exe"
            if not candidate.is_file():
                raise NeMoGymError("NeMo Gym CLI not found beside its pinned Python interpreter")
            self.gym = candidate.resolve()
        else:
            self.gym = Path(gym_executable).resolve()
        if not self.gym.is_file():
            raise NeMoGymError(f"NeMo Gym executable not found: {self.gym}")
        self.version = self._probe_version()
        if self.version["nemo_gym"] != expected_version:
            raise NeMoGymError(
                f"NeMo Gym version mismatch: expected {expected_version}, found {self.version['nemo_gym']}"
            )
        if tuple(self.version["python"]) < NEMO_GYM_PYTHON_MINIMUM:
            raise NeMoGymError(
                "NeMo Gym v0.7.0 requires Python >= 3.13.14; found "
                + ".".join(str(part) for part in self.version["python"])
            )

    def _probe_version(self) -> dict[str, Any]:
        code = (
            "import importlib.metadata as m, json, sys; "
            "print(json.dumps({'nemo_gym': m.version('nemo-gym'), 'python': list(sys.version_info[:3])}))"
        )
        try:
            result = subprocess.run(
                [str(self.python), "-c", code], capture_output=True, text=True, check=False, timeout=30
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise NeMoGymError(f"Could not probe NeMo Gym installation: {exc}") from exc
        if result.returncode != 0:
            raise NeMoGymError("The selected interpreter does not have the nemo-gym distribution installed")
        try:
            metadata = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError) as exc:
            raise NeMoGymError("Could not parse NeMo Gym runtime identity") from exc
        if (not isinstance(metadata, dict) or not isinstance(metadata.get("nemo_gym"), str)
            or not isinstance(metadata.get("python"), list) or len(metadata["python"]) != 3
            or any(type(part) is not int or part < 0 for part in metadata["python"])):
            raise NeMoGymError("NeMo Gym runtime identity is incomplete")
        return metadata

    def run_reverification(
        self, *, gym_root: str | Path, config_paths: Sequence[str | Path],
        materialized_inputs: str | Path, rollouts: str | Path, output_dir: str | Path,
        split: str,
        timeout_seconds: float = 3600.0,
    ) -> dict[str, Any]:
        root = Path(gym_root).resolve()
        configs = tuple(Path(path).resolve() for path in config_paths)
        inputs_path = Path(materialized_inputs).resolve()
        rollouts_path = Path(rollouts).resolve()
        output = Path(output_dir).resolve()
        if split not in {"train", "validation"}:
            raise NeMoGymError("reverification outside custody only supports public train or validation splits")
        if not root.is_dir() or not configs or any(not path.is_file() for path in configs):
            raise NeMoGymError("reverification requires a Gym root and existing configuration files")
        if not inputs_path.is_file() or not rollouts_path.is_file():
            raise NeMoGymError("reverification requires existing inputs and rollouts")
        if os.name == "nt":
            raise NeMoGymError("NeMo reverification process-group cleanup requires a POSIX host")
        if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
            raise NeMoGymError("timeout_seconds must be finite and positive")
        inputs, input_digest = _read_jsonl(inputs_path, max_bytes=MAX_ROLLOUT_BYTES)
        original, rollout_digest = _read_jsonl(rollouts_path, max_bytes=MAX_ROLLOUT_BYTES)
        input_keys = [_rollout_key(row, label="materialized input") for row in inputs]
        original_keys = [_rollout_key(row, label="rollout") for row in original]
        if (not input_keys or len(set(input_keys)) != len(input_keys)
            or len(set(original_keys)) != len(original_keys) or set(input_keys) != set(original_keys)):
            raise NeMoGymError("reverification requires the complete unique input and rollout roster")
        config_digests = {str(path): _sha256_file(path) for path in configs}
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"NeMo Gym output directory is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        output.chmod(0o700)
        result_path = output / "reverified.jsonl"
        worker = Path(__file__).parent / "backends" / "nemo_gym" / "worker" / "reverify.py"
        worker_digest = _sha256_file(worker)
        command = [str(self.python), str(worker), "eval", "reverify", "--inputs", str(inputs_path),
                   "--rollouts", str(rollouts_path), "--output", str(result_path), "--disable-aggregation"]
        for config in configs:
            command.extend(("--config", str(config)))
        try:
            execution = run_local_worker(command, cwd=root, timeout_seconds=timeout_seconds, cleanup_descendants=True)
        except OSError as exc:
            raise NeMoGymError(f"Could not start NeMo Gym reverification: {exc}") from exc
        if execution.timed_out or execution.returncode != 0:
            (output / "stdout.log").write_bytes(execution.stdout)
            (output / "stderr.log").write_bytes(execution.stderr)
            raise NeMoGymError("NeMo Gym reverification did not complete successfully; inspect private output logs")
        if (_sha256_file(inputs_path) != input_digest or _sha256_file(rollouts_path) != rollout_digest
            or _sha256_file(worker) != worker_digest
            or any(_sha256_file(path) != config_digests[str(path)] for path in configs)):
            raise NeMoGymError("NeMo Gym reverification inputs or configuration changed during execution")
        if not result_path.is_file():
            raise NeMoGymError("NeMo Gym omitted the reverification output")
        results, result_digest = _read_jsonl(result_path, max_bytes=MAX_ROLLOUT_BYTES)
        result_keys = [_rollout_key(row, label="reverified rollout") for row in results]
        if len(set(result_keys)) != len(result_keys) or set(result_keys) != set(input_keys):
            raise NeMoGymError("NeMo Gym reverification changed the rollout roster")
        original_by_key = dict(zip(original_keys, original, strict=True))
        responses_match = all(
            "response" in row and "response" in original_by_key[key]
            and _same_reverified_response(original_by_key[key]["response"], row["response"])
            for key, row in zip(result_keys, results, strict=True)
        )
        if not responses_match:
            raise NeMoGymError("NeMo Gym reverification changed a stored response")
        complete = all(
            row.get("mask_sample") is False and row.get("evaluation_completed", True) is True
            and row.get("failure_kind") is None and is_finite_number(row.get("reward"))
            and 0 <= row["reward"] <= 1 for row in results
        )
        receipt = {
            "kind": "nf_nemo_reverification_v1", "nemo_gym_version": self.version["nemo_gym"],
            "python_version": self.version["python"],
            "worker_digest": worker_digest, "declared_split": split,
            "input_digest": input_digest, "rollout_digest": rollout_digest, "result_digest": result_digest,
            "config_digests": config_digests, "command": command, "execution": execution.receipt,
            "rows": len(results), "complete": complete,
            "authority": {"scorer_binding_verified": False, "sealed_target_custody": False,
                          "promotion_authorized": False},
        }
        receipt["receipt_digest"] = _document_digest(receipt)
        (output / "neural_foundry_nemo_reverify_receipt.json").write_text(
            json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return receipt

    def run_evaluation(
        self,
        *,
        gym_root: str | Path,
        config_paths: Sequence[str | Path],
        agent: str,
        input_jsonl: str | Path,
        output_dir: str | Path,
        model_url: str,
        model_name: str,
        timeout_seconds: float = 3600.0,
        expected_rollouts_per_task: int = 1,
    ) -> dict[str, Any]:
        root = Path(gym_root).resolve()
        configs = tuple(Path(path).resolve() for path in config_paths)
        input_path = Path(input_jsonl).resolve()
        output = Path(output_dir).resolve()
        if not root.is_dir() or not agent.strip() or not model_name.strip():
            raise NeMoGymError("Gym root, agent, and served model name are required")
        if not configs or any(not path.is_file() for path in configs):
            raise NeMoGymError("At least one existing NeMo Gym config path is required")
        if not input_path.is_file():
            raise NeMoGymError(f"NeMo Gym input JSONL not found: {input_path}")
        requested_rows, requested_digest = _read_jsonl(input_path, max_bytes=MAX_PANEL_BYTES)
        if not requested_rows or type(expected_rollouts_per_task) is not int or expected_rollouts_per_task < 1:
            raise NeMoGymError("NeMo run requires a nonempty requested roster and positive rollout count")
        config_digests = {str(path): _sha256_file(path) for path in configs}
        parsed_url = urlsplit(model_url)
        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.hostname
        ):
            raise NeMoGymError("model_url must be an HTTP(S) model-server URL")
        if (
            parsed_url.username is not None
            or parsed_url.password is not None
            or parsed_url.query
            or parsed_url.fragment
        ):
            raise NeMoGymError("model_url must not contain credentials, query parameters, or fragments")
        if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
            raise NeMoGymError("timeout_seconds must be finite and positive")
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"NeMo Gym output directory is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        output.chmod(0o700)
        (output / "requested_roster.json").write_text(json.dumps({
            "kind": "nf_nemo_requested_roster_v1", "input_digest": requested_digest,
            "expected_keys": [[index, rollout] for index in range(len(requested_rows)) for rollout in range(expected_rollouts_per_task)],
        }), encoding="utf-8")
        rollouts_path = output / "rollouts.jsonl"
        command = [
            str(self.gym), "eval", "run", "--no-serve",
            "--agent", agent,
            "--input", str(input_path),
            "--output", str(rollouts_path),
            "--model-type", "openai_model",
            "--model-url", model_url,
            "--model", model_name,
        ]
        for config in configs:
            command.extend(("--config", str(config)))
        safe_host = parsed_url.hostname
        if parsed_url.port is not None:
            safe_host = f"{safe_host}:{parsed_url.port}"
        safe_url = urlunsplit((parsed_url.scheme, safe_host, parsed_url.path, "", ""))
        safe_command = [safe_url if item == model_url else item for item in command]
        try:
            execution = run_local_worker(
                command, cwd=root, timeout_seconds=timeout_seconds, receipt_command=safe_command,
            )
        except OSError as exc:
            raise NeMoGymError(f"Could not start NeMo Gym: {exc}") from exc
        if execution.timed_out:
            raise NeMoGymError(f"NeMo Gym evaluation exceeded {timeout_seconds}s; process tree stopped")
        if execution.returncode != 0:
            raise NeMoGymError(f"NeMo Gym exited {execution.returncode}; inspect the isolated run configuration and logs")
        if _sha256_file(input_path) != requested_digest or any(
            _sha256_file(path) != config_digests[str(path)] for path in configs
        ):
            raise NeMoGymError("NeMo Gym input or configuration changed during evaluation")
        report = inspect_run_artifacts(output)
        receipt = {
            **report,
            "agent": agent,
            "model_name": model_name,
            "model_endpoint": safe_url,
            "config_digests": config_digests,
            "input_digest": requested_digest,
            "command": safe_command,
            "execution": execution.receipt,
        }
        receipt_path = output / RUN_RECEIPT_NAME
        if receipt_path.exists():
            raise FileExistsError(f"NeMo Gym receipt already exists: {receipt_path}")
        receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return receipt


__all__ = [
    "NEMO_GYM_VERSION",
    "NeMoGymError",
    "NeMoGymRunner",
    "NeMoGymTaskError",
    "NeMoMCQAPanels",
    "NeMoPreferenceDataset",
    "NeMoPreferenceTrainingRun",
    "NeMoRolloutTrainingRun",
    "NeMoSFTDataset",
    "NeMoTaskAdapter",
    "NeMoTaskAdapterRegistry",
    "compile_public_nemo_dataset",
    "import_mcqa_preferences",
    "import_successful_mcqa_rollouts",
    "inspect_run_artifacts",
    "load_mcqa_panels",
    "load_nemo_sft_candidate",
    "register_nemo_mcqa_scenario_sources",
    "train_mcqa_preferences",
    "train_mcqa_rollouts",
]
