"""Digest-bound limits for explicitly approved external trainer runs."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .gym_job import GymJobError
from .provenance import parse_strict_json, sha256_file


def bound_training_policy(spec: Mapping[str, Any]) -> dict[str, Any] | None:
    reference = spec.get("training_policy_reference")
    if reference is None:
        return None
    path = Path(reference["path"])
    if not path.is_absolute() or sha256_file(path) != reference["file_digest"]:
        raise GymJobError("training policy file digest differs")
    policy = parse_strict_json(path.read_bytes())
    if (policy.get("kind") != "nf_public_training_policy" or policy.get("version") != 1
        or any(type(policy.get(key)) is not int or policy[key] <= 0 for key in
               ("maximum_rows", "maximum_steps", "worker_timeout_seconds",
                "maximum_cuda_allocated_bytes"))
        or policy["base_staging_device"] != "cpu" or policy["training_device"] != "cuda:0"
        or policy.get("automatic_retries_authorized") is not False
        or policy.get("adaptive_recipe_changes_authorized") is not False):
        raise GymJobError("unsupported bounded training policy")
    return policy
