"""Versioned, shared materialization for prompt-target Gym datasets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pythia import ReusableTaskGym

GYM_SPEC_SCHEMA_VERSION = 1
GYM_SPEC_KIND = "nf_text_task_gym"
MAX_GYM_SPEC_BYTES = 1_048_576
MAX_RULES_PER_SPLIT = 10_000
MAX_TEXT_BYTES = 16_384


class GymDataError(ValueError):
    """A Gym spec or its materialized data violates the shared task contract."""


@dataclass(frozen=True)
class PublicTaskData:
    schema_version: int
    gym_id: str
    gym_digest: str
    dataset_digest: str
    train_digest: str
    validation_digest: str
    train_items: tuple[tuple[str, str], ...]
    validation_items: tuple[tuple[str, str], ...]


def _canonical_digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _gym_identity_document(gym: ReusableTaskGym) -> dict[str, Any]:
    prefixes = {
        "train": list(gym.gym_dev_prefixes),
        "validation": list(gym.gym_val_prefixes),
        "holdout": list(gym.holdout_prefixes),
    }
    return {
        "schema_version": GYM_SPEC_SCHEMA_VERSION,
        "kind": GYM_SPEC_KIND,
        "gym_id": gym.gym_id,
        "skills_taught": list(gym.skills_taught),
        "train_rules": [{"prompt": prompt, "target": target} for prompt, target in gym.base_rules],
        "validation_rules": [
            {"prompt": prompt, "target": target} for prompt, target in gym.validation_rules
        ],
        "holdout_rules": [{"prompt": prompt, "target": target} for prompt, target in gym.holdout_rules],
        "prefixes": prefixes,
        "retention_prompts": list(gym.retention_prompts),
        "selection_retention_guard_prompts": list(gym.selection_retention_guard_prompts),
    }


def gym_spec_document(gym: ReusableTaskGym) -> dict[str, Any]:
    document = _gym_identity_document(gym)
    document.pop("holdout_rules")
    document["prefixes"].pop("holdout")
    document.pop("selection_retention_guard_prompts")
    return document


def gym_digest(gym: ReusableTaskGym) -> str:
    return _canonical_digest(_gym_identity_document(gym))


def _normalize_rules(value: Any, name: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list) or not value or len(value) > MAX_RULES_PER_SPLIT:
        raise GymDataError(f"{name} must be a non-empty bounded list of prompt/target rules")
    rules: list[tuple[str, str]] = []
    seen: set[str] = set()
    for row in value:
        if not isinstance(row, Mapping) or set(row) != {"prompt", "target"}:
            raise GymDataError(f"{name} entries must contain exactly prompt and target")
        prompt, target = row["prompt"], row["target"]
        if not isinstance(prompt, str) or not isinstance(target, str):
            raise GymDataError(f"{name} prompt and target values must be strings")
        if not prompt.strip() or not target.strip():
            raise GymDataError(f"{name} prompt and target values must be non-empty")
        if len(prompt.encode("utf-8")) > MAX_TEXT_BYTES or len(target.encode("utf-8")) > MAX_TEXT_BYTES:
            raise GymDataError(f"{name} prompt or target exceeds the text bound")
        if prompt in seen:
            raise GymDataError(f"{name} contains duplicate prompt {prompt!r}")
        seen.add(prompt)
        rules.append((prompt, target))
    return tuple(rules)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise GymDataError(f"duplicate JSON field {key!r}")
        output[key] = value
    return output


def load_gym_spec(path: str | Path) -> ReusableTaskGym:
    """Load a data-only v1 Gym spec; this never imports or executes Gym code."""
    spec_path = Path(path)
    if spec_path.stat().st_size > MAX_GYM_SPEC_BYTES:
        raise GymDataError("Gym spec exceeds the byte limit")
    with spec_path.open("rb") as stream:
        raw = stream.read(MAX_GYM_SPEC_BYTES + 1)
    if len(raw) > MAX_GYM_SPEC_BYTES:
        raise GymDataError("Gym spec exceeds the byte limit")
    try:
        document = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GymDataError(f"Gym spec is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(document, Mapping):
        raise GymDataError("Gym spec root must be an object")
    allowed = {
        "schema_version", "kind", "gym_id", "skills_taught", "train_rules", "validation_rules",
        "holdout_rules", "prefixes", "retention_prompts", "selection_retention_guard_prompts",
    }
    if set(document) - allowed:
        raise GymDataError(f"unknown Gym spec fields: {sorted(set(document) - allowed)}")
    if document.get("schema_version") != GYM_SPEC_SCHEMA_VERSION or document.get("kind") != GYM_SPEC_KIND:
        raise GymDataError("unsupported Gym spec kind or schema_version")
    gym_id = document.get("gym_id")
    skills = document.get("skills_taught")
    prefixes = document.get("prefixes")
    if not isinstance(gym_id, str) or not gym_id.strip() or len(gym_id.encode("utf-8")) > 512:
        raise GymDataError("gym_id must be a non-empty string")
    if (
        not isinstance(skills, list)
        or not skills
        or len(skills) > 256
        or any(not isinstance(item, str) or not item.strip() for item in skills)
        or len({item.strip() for item in skills}) != len(skills)
    ):
        raise GymDataError("skills_taught must be a non-empty list of strings")
    if not isinstance(prefixes, Mapping) or set(prefixes) not in (
        {"train", "validation"}, {"train", "validation", "holdout"}
    ):
        raise GymDataError("prefixes must define public train and validation lists, with optional private holdout")

    def strings(
        value: Any, name: str, *, required: bool = True, allow_duplicates: bool = False
    ) -> tuple[str, ...]:
        if not isinstance(value, list) or (required and not value) or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            raise GymDataError(f"{name} must be a {'non-empty ' if required else ''}list of strings")
        if (
            len(value) > 256
            or any(len(item.encode("utf-8")) > MAX_TEXT_BYTES for item in value)
            or (not allow_duplicates and len({item.strip() for item in value}) != len(value))
        ):
            raise GymDataError(f"{name} contains duplicates or exceeds its bound")
        return tuple(item.strip() for item in value)

    retention = strings(document.get("retention_prompts", []), "retention_prompts", required=False)
    selection = strings(
        document.get("selection_retention_guard_prompts", []),
        "selection_retention_guard_prompts",
        required=False,
    )
    gym = ReusableTaskGym(
        gym_id=gym_id.strip(),
        skills_taught=tuple(item.strip() for item in skills),
        base_rules=_normalize_rules(document.get("train_rules"), "train_rules"),
        holdout_rules=_normalize_rules(document["holdout_rules"], "holdout_rules") if document.get("holdout_rules") else (),
        gym_dev_prefixes=strings(prefixes.get("train"), "prefixes.train", required=False),
        gym_val_prefixes=strings(
            prefixes.get("validation"), "prefixes.validation", required=False
        ),
        holdout_prefixes=strings(
            prefixes.get("holdout", []), "prefixes.holdout", required=False, allow_duplicates=True
        ),
        retention_prompts=retention or ReusableTaskGym.__dataclass_fields__["retention_prompts"].default,
        selection_retention_guard_prompts=(
            selection
            or ReusableTaskGym.__dataclass_fields__["selection_retention_guard_prompts"].default
        ),
        validation_rules=_normalize_rules(document.get("validation_rules"), "validation_rules"),
    )
    materialize_public_task_data(gym)
    return gym


def resolve_gym(value: str | ReusableTaskGym) -> ReusableTaskGym:
    """Resolve built-ins, a JSON spec path, or an already validated Gym value."""
    if isinstance(value, ReusableTaskGym):
        return value
    path = Path(value)
    if path.is_file():
        return load_gym_spec(path)
    from .pythia import ALGORITHMIC_THREE_SKILL_GYM, FACTUAL_KNOWLEDGE_GYM

    builtins = {
        "algorithmic_three_skill": ALGORITHMIC_THREE_SKILL_GYM,
        ALGORITHMIC_THREE_SKILL_GYM.gym_id: ALGORITHMIC_THREE_SKILL_GYM,
        "factual_knowledge": FACTUAL_KNOWLEDGE_GYM,
        FACTUAL_KNOWLEDGE_GYM.gym_id: FACTUAL_KNOWLEDGE_GYM,
    }
    try:
        gym = builtins[value]
    except KeyError as exc:
        raise GymDataError(f"unknown Gym key or spec path: {value!r}") from exc
    materialize_public_task_data(gym)
    return gym


def _expand(
    rules: Sequence[tuple[str, str]], prefixes: Sequence[str], split: str
) -> tuple[tuple[str, str], ...]:
    if not rules:
        raise GymDataError(f"{split} rules must be non-empty")
    if not prefixes:
        return tuple(rules)
    if len(set(prefixes)) != len(prefixes) or any(not prefix.strip() for prefix in prefixes):
        raise GymDataError(f"{split} prefixes must be non-empty and unique")
    return tuple((f"{prefix} {prompt}", target) for prefix in prefixes for prompt, target in rules)


def _validate_model_items(
    items: Sequence[tuple[str, str]], tokenizer: Any | None, split: str
) -> tuple[tuple[str, str], ...]:
    for prompt, target in items:
        if (
            not isinstance(prompt, str)
            or not prompt.strip()
            or not isinstance(target, str)
            or not target.strip()
        ):
            raise GymDataError(f"{split} contains an invalid prompt/target item")
        if len(prompt.encode("utf-8")) > MAX_TEXT_BYTES or len(target.encode("utf-8")) > MAX_TEXT_BYTES:
            raise GymDataError(f"{split} prompt or target exceeds the text bound")
        if tokenizer is None:
            continue
        if not tokenizer.encode(prompt):
            raise GymDataError(f"{split} prompt is not encodable by the selected model: {prompt!r}")
        target_id = tokenizer.vocab.get(target)
        if target_id is None:
            raise GymDataError(f"{split} target is not a single vocabulary token: {target!r}")
        decoded = tokenizer.decode_token(target_id)
        if tokenizer.encode(decoded) != [target_id]:
            raise GymDataError(f"{split} target does not round-trip as one token: {target!r}")
    return tuple(items)


def materialize_public_task_data(
    gym: ReusableTaskGym, *, tokenizer: Any | None = None
) -> PublicTaskData:
    """Materialize shared train/validation items and reject rule-level leakage."""
    if not gym.base_rules or not gym.validation_rules:
        raise GymDataError("versioned Gym training requires explicit train_rules and validation_rules")
    split_rules = {
        "train": tuple(gym.base_rules),
        "validation": tuple(gym.validation_rules),
        "holdout": tuple(gym.holdout_rules),
    }
    seen: dict[str, str] = {}
    for split, rules in split_rules.items():
        if len(rules) > MAX_RULES_PER_SPLIT:
            raise GymDataError(f"{split} exceeds the rule count bound")
        for prompt, target in rules:
            if not isinstance(prompt, str) or not isinstance(target, str) or not prompt.strip() or not target.strip():
                raise GymDataError(f"{split} contains an invalid prompt/target rule")
            previous = seen.get(prompt)
            if previous is not None:
                raise GymDataError(f"prompt rule {prompt!r} occurs in multiple splits")
            seen[prompt] = split
    train = _validate_model_items(
        _expand(gym.base_rules, gym.gym_dev_prefixes, "train"), tokenizer, "train"
    )
    validation = _validate_model_items(
        _expand(gym.validation_rules, gym.gym_val_prefixes, "validation"), tokenizer, "validation"
    )
    holdout = materialize_holdout_items(gym, tokenizer=tokenizer)
    seen_materialized: dict[str, str] = {}
    for split, items in (("train", train), ("validation", validation), ("holdout", holdout)):
        for prompt, _ in items:
            previous = seen_materialized.get(prompt)
            if previous is not None:
                raise GymDataError(
                    f"materialized prompt {prompt!r} occurs in both {previous} and {split} splits"
                )
            seen_materialized[prompt] = split
    train_digest = _canonical_digest({"split": "train", "items": train})
    validation_digest = _canonical_digest({"split": "validation", "items": validation})
    dataset_digest = _canonical_digest(
        {
            "schema_version": GYM_SPEC_SCHEMA_VERSION,
            "gym_digest": gym_digest(gym),
            "train_digest": train_digest,
            "validation_digest": validation_digest,
            "holdout_digest": _canonical_digest({"split": "holdout", "items": holdout}),
        }
    )
    return PublicTaskData(
        schema_version=GYM_SPEC_SCHEMA_VERSION,
        gym_id=gym.gym_id,
        gym_digest=gym_digest(gym),
        dataset_digest=dataset_digest,
        train_digest=train_digest,
        validation_digest=validation_digest,
        train_items=train,
        validation_items=validation,
    )


def materialize_holdout_items(
    gym: ReusableTaskGym, *, tokenizer: Any | None = None
) -> tuple[tuple[str, str], ...]:
    """Materialize the separately requested held-out rules; training paths never call this."""
    if not gym.holdout_rules:
        return ()
    if not gym.holdout_prefixes:
        return _validate_model_items(gym.holdout_rules, tokenizer, "holdout")
    return _validate_model_items(
        tuple(
            (f"{gym.holdout_prefixes[index % len(gym.holdout_prefixes)]} {prompt}", target)
            for index, (prompt, target) in enumerate(gym.holdout_rules)
        ),
        tokenizer,
        "holdout",
    )