"""M4 Training/Memory Baselines, External Gym Adapters, and Cross-Model Learning (T078–T089, T091–T099).

Implements:
- T078 (`lora_baseline`), T079 (`sft_baseline`), T080 (`distillation_baseline`),
  T081 (`rl_baseline`, CG-11), T082 (`memory_methods`), T083 (`continual_protocols`).
- T084 (`nemo_adapter`), T085 (`verifiers_adapter`), T086 (`gymnasium_adapter`),
  T087 (`private_gym_policy`), T088 (`measurement_research`), T089 (`community_gym`).
- T091 (`library_reuse`, CG-13), T092 (`strategy_retrieval`), T093 (`cross_model_protocol`),
  T094 (`proposer_dataset`), T095 (`proposer_training`), T096 (`learned_proposer`),
  T097 (`transfer_benchmark`), T098 (`research_source_records`), T099 (`m4_feature_acceptance`).

Evidence: the baselines train single linear matrices on supplied tensors. The adapters do not
import NeMo-Gym, Verifiers or Gymnasium. Evaluators score numbers the caller supplies. Results carry
an ``evidence`` field. run_transfer_benchmark and run_proposer_training_study are not implemented
and raise NotImplementedError once their contract checks pass.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import torch

from .export_formats import _map_file, _parse_safetensors_header, export_huggingface_safetensors
from .m4_modalities import (
    EvidenceBasis,
    EvidenceOrigin,
    M4ContractError,
    SupportGrade,
    construct_algorithmic_relu_rule,
    estimate_permutation_alignment,
    execute_mapped_weight_transfer,
)
from .records import canonicalize, content_digest


def _digest(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    try:
        return str(content_digest(canonicalize(raw)))
    except (ValueError, TypeError, RuntimeError):
        return f"sha256:{hashlib.sha256(raw).hexdigest()}"


# ============================================================================
# T078–T081: Authorized Training & RL Baselines (CG-11)
# ============================================================================


@dataclass(frozen=True)
class BaselineExecutionReceipt:
    baseline_kind: Literal["lora", "sft", "distillation", "reinforce_rl"]
    authorized_in_contract: bool
    dataset_digest: str
    optimizer_steps: int
    trainable_parameters: int
    total_compute_flops: int
    merged_into_base_weights: bool
    # None: these baselines never write the result to disk and reload it.
    reload_parity_verified: None
    failed_direct_attempts_retained: int
    production_history: Literal["trained", "hybrid"]
    eval_loss_before: float
    eval_loss_after: float
    extra_metadata: dict[str, Any]
    # Losses are measured on a linear fixture; total_compute_flops is a formula, not a count.
    evidence: EvidenceBasis = "measured_on_fixture"


def run_lora_baseline(
    base_weight: torch.Tensor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    authorized_in_contract: bool,
    rank: int = 2,
    steps: int = 12,
    lr: float = 0.05,
    failed_direct_attempts_retained: int = 1,
    merge_before_eval: bool = True,
) -> tuple[torch.Tensor, BaselineExecutionReceipt]:
    """Fits a bounded low-rank adapter, merges into base weights, and preserves failed direct attempts (T078)."""
    if not authorized_in_contract:
        raise M4ContractError(
            "BASELINE_NOT_AUTHORIZED",
            "LoRA baseline is disabled by default and requires explicit StudyContract permission (CG-11)",
        )
    if not merge_before_eval:
        raise M4ContractError("UNMERGED_LORA_FORBIDDEN", "LoRA adapter must be merged before matched evaluation")
    if failed_direct_attempts_retained < 0:
        raise M4ContractError("INVALID_HISTORY", "failed direct attempts cannot be erased")

    w0 = base_weight.detach().clone().to(dtype=torch.float64)
    x = inputs.detach().to(dtype=torch.float64)
    y = targets.detach().to(dtype=torch.float64)
    out_dim, in_dim = int(w0.shape[0]), int(w0.shape[1])

    torch.manual_seed(7)
    lora_a = (torch.randn(rank, in_dim, dtype=torch.float64) * 0.05).requires_grad_(True)
    lora_b = torch.zeros(out_dim, rank, dtype=torch.float64, requires_grad=True)

    with torch.no_grad():
        loss_before = float(torch.mean(((x @ w0.T) - y) ** 2).item())

    for _ in range(steps):
        pred = x @ (w0 + lora_b @ lora_a).T
        loss = torch.mean((pred - y) ** 2)
        grad_a, grad_b = torch.autograd.grad(loss, (lora_a, lora_b))
        with torch.no_grad():
            lora_a -= lr * grad_a
            lora_b -= lr * grad_b

    with torch.no_grad():
        merged = (w0 + lora_b @ lora_a).contiguous()
        loss_after = float(torch.mean(((x @ merged.T) - y) ** 2).item())

    trainable = rank * (out_dim + in_dim)
    flops = steps * int(x.shape[0]) * out_dim * in_dim * 2
    return merged, BaselineExecutionReceipt(
        baseline_kind="lora",
        authorized_in_contract=True,
        dataset_digest=_digest({"n": int(x.shape[0]), "in": in_dim, "out": out_dim}),
        optimizer_steps=steps,
        trainable_parameters=trainable,
        total_compute_flops=flops,
        merged_into_base_weights=True,
        reload_parity_verified=None,
        failed_direct_attempts_retained=failed_direct_attempts_retained,
        production_history="trained",
        eval_loss_before=loss_before,
        eval_loss_after=loss_after,
        extra_metadata={"rank": rank, "lr": lr},
    )


def run_sft_baseline(
    base_weight: torch.Tensor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    authorized_in_contract: bool,
    allowed_site: str = "head.weight",
    target_site: str = "head.weight",
    steps: int = 10,
    lr: float = 0.05,
) -> tuple[torch.Tensor, BaselineExecutionReceipt]:
    """Trains declared parameters under a frozen objective and schedule with full optimizer lineage (T079)."""
    if not authorized_in_contract:
        raise M4ContractError("SFT_NOT_AUTHORIZED", "SFT baseline is disabled unless explicitly authorized (CG-11)")
    if target_site != allowed_site:
        raise M4ContractError(
            "UNAUTHORIZED_PARAMETER_SITE",
            f"SFT attempted to update '{target_site}' outside authorized '{allowed_site}'",
        )
    w = base_weight.detach().clone().to(dtype=torch.float64).requires_grad_(True)
    x = inputs.detach().to(dtype=torch.float64)
    y = targets.detach().to(dtype=torch.float64)

    with torch.no_grad():
        loss_before = float(torch.mean(((x @ w.T) - y) ** 2).item())

    for _ in range(steps):
        loss = torch.mean(((x @ w.T) - y) ** 2)
        (grad_w,) = torch.autograd.grad(loss, (w,))
        with torch.no_grad():
            w -= lr * grad_w

    with torch.no_grad():
        final_w = w.detach().clone()
        loss_after = float(torch.mean(((x @ final_w.T) - y) ** 2).item())

    return final_w, BaselineExecutionReceipt(
        baseline_kind="sft",
        authorized_in_contract=True,
        dataset_digest=_digest({"site": target_site, "n": int(x.shape[0])}),
        optimizer_steps=steps,
        trainable_parameters=int(final_w.numel()),
        total_compute_flops=steps * int(x.shape[0]) * int(final_w.numel()) * 2,
        merged_into_base_weights=True,
        reload_parity_verified=None,
        failed_direct_attempts_retained=0,
        production_history="trained",
        eval_loss_before=loss_before,
        eval_loss_after=loss_after,
        extra_metadata={"schedule": "constant_lr", "lr": lr, "site": target_site},
    )


def run_distillation_baseline(
    student_weight: torch.Tensor,
    teacher_weight: torch.Tensor,
    inputs: torch.Tensor,
    *,
    authorized_in_contract: bool,
    teacher_generation_flops: int,
    claim_internal_circuit_transfer: bool = False,
    steps: int = 10,
    lr: float = 0.05,
) -> tuple[torch.Tensor, BaselineExecutionReceipt]:
    """Trains student against permitted teacher outputs and charges teacher-generation cost (T080)."""
    if not authorized_in_contract:
        raise M4ContractError("DISTILLATION_NOT_AUTHORIZED", "distillation baseline requires contract permission")
    if teacher_generation_flops <= 0:
        raise M4ContractError(
            "HIDDEN_TEACHER_COST_FORBIDDEN",
            "distillation must charge positive teacher_generation_flops",
        )
    if claim_internal_circuit_transfer:
        raise M4ContractError(
            "OUTPUT_ONLY_NOT_INTERNAL_TRANSFER",
            "output-only distillation cannot be claimed as internal weight/circuit transfer (T080)",
        )
    x = inputs.detach().to(dtype=torch.float64)
    tw = teacher_weight.detach().to(dtype=torch.float64)
    with torch.no_grad():
        teacher_outputs = x @ tw.T

    sw, sft_rec = run_sft_baseline(
        student_weight,
        x,
        teacher_outputs,
        authorized_in_contract=True,
        steps=steps,
        lr=lr,
    )
    return sw, BaselineExecutionReceipt(
        baseline_kind="distillation",
        authorized_in_contract=True,
        dataset_digest=sft_rec.dataset_digest,
        optimizer_steps=steps,
        trainable_parameters=sft_rec.trainable_parameters,
        total_compute_flops=sft_rec.total_compute_flops + teacher_generation_flops,
        merged_into_base_weights=True,
        reload_parity_verified=None,
        failed_direct_attempts_retained=0,
        production_history="trained",
        eval_loss_before=sft_rec.eval_loss_before,
        eval_loss_after=sft_rec.eval_loss_after,
        extra_metadata={
            "transfer_modality": "output_only_distillation",
            "teacher_generation_flops": teacher_generation_flops,
        },
    )


def run_rl_baseline(
    mean_weight: torch.Tensor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    authorized_in_contract: bool,
    episodes: int = 16,
    max_episodes_limit: int = 64,
    sigma: float = 0.1,
    lr: float = 0.05,
    grad_clip: float = 1.0,
    seed: int = 42,
) -> tuple[torch.Tensor, BaselineExecutionReceipt]:
    """Fixed-variance Gaussian REINFORCE baseline with bounded rollouts and mean-policy evaluation (T081)."""
    if not authorized_in_contract:
        raise M4ContractError("RL_NOT_AUTHORIZED", "RL baseline requires explicit authorization")
    if episodes <= 0 or episodes > max_episodes_limit:
        raise M4ContractError(
            "UNBOUNDED_RL_EPISODES",
            f"episodes {episodes} exceeds bounded rollout limit {max_episodes_limit}",
        )
    mu = mean_weight.detach().clone().to(dtype=torch.float64)
    x = inputs.detach().to(dtype=torch.float64)
    y = targets.detach().to(dtype=torch.float64)

    loss_before = float(torch.mean(((x @ mu.T) - y) ** 2).item())
    gen = torch.Generator().manual_seed(seed)
    failed_episodes = 0

    for _ in range(episodes):
        eps = torch.randn(mu.shape, generator=gen, dtype=torch.float64)
        sampled_w = mu + sigma * eps
        pred = x @ sampled_w.T
        reward = -float(torch.mean((pred - y) ** 2).item())
        if not torch.isfinite(torch.tensor(reward)):
            failed_episodes += 1
            continue
        baseline_reward = -float(torch.mean(((x @ mu.T) - y) ** 2).item())
        advantage = reward - baseline_reward
        grad = (advantage / (sigma**2)) * eps
        grad_norm = float(grad.norm().item())
        if grad_norm > grad_clip:
            grad = grad * (grad_clip / grad_norm)
        mu = mu + lr * grad

    # Evaluate deterministic mean policy separately from stochastic training rollout policy
    loss_after = float(torch.mean(((x @ mu.T) - y) ** 2).item())
    return mu, BaselineExecutionReceipt(
        baseline_kind="reinforce_rl",
        authorized_in_contract=True,
        dataset_digest=_digest({"seed": seed, "episodes": episodes}),
        optimizer_steps=episodes,
        trainable_parameters=int(mu.numel()),
        total_compute_flops=episodes * int(x.shape[0]) * int(mu.numel()) * 2,
        merged_into_base_weights=True,
        reload_parity_verified=None,
        failed_direct_attempts_retained=failed_episodes,
        production_history="trained",
        eval_loss_before=loss_before,
        eval_loss_after=loss_after,
        extra_metadata={
            "policy_family": "fixed_variance_gaussian_reinforce",
            "sigma": sigma,
            "grad_clip": grad_clip,
            "stochastic_training_separated_from_mean_eval": True,
        },
    )


# ============================================================================
# T082 & T083: Memory Methods and Continual-Learning Protocols
# ============================================================================


@dataclass(frozen=True)
class MemoryMethodReceipt:
    memory_id: str
    capacity_slots: int
    used_slots: int
    # None: no memory is exercised, so resetting is not observed.
    reset_between_eval_items: None
    static_export_self_contained: bool
    production_history: Literal["hybrid"]
    evidence: EvidenceBasis = "caller_supplied"


def verify_external_memory_method(
    *,
    memory_id: str,
    capacity_slots: int,
    entries: list[tuple[str, float]],
    contains_hidden_holdout_labels: bool = False,
    claim_direct_only_weight_edit: bool = False,
    static_export_requires_hidden_daemon: bool = False,
) -> MemoryMethodReceipt:
    """Checks caller declarations about a memory method; does not run or inspect one (T082)."""
    if contains_hidden_holdout_labels:
        raise M4ContractError(
            "MEMORY_LABEL_LEAKAGE",
            "external memory write attempted to store holdout labels",
        )
    if claim_direct_only_weight_edit:
        raise M4ContractError(
            "MEMORY_RELABEL_FORBIDDEN",
            "runtime external memory method cannot be labeled as direct_only weight edit",
        )
    if static_export_requires_hidden_daemon:
        raise M4ContractError(
            "HIDDEN_RUNTIME_STATE_FORBIDDEN",
            "static checkpoint exports must not depend on undeclared runtime memory state",
        )
    if len(entries) > capacity_slots:
        raise M4ContractError("MEMORY_CAPACITY_EXCEEDED", "entries exceed declared memory capacity")
    return MemoryMethodReceipt(
        memory_id=memory_id,
        capacity_slots=capacity_slots,
        used_slots=len(entries),
        reset_between_eval_items=None,
        static_export_self_contained=True,
        production_history="hybrid",
    )


@dataclass(frozen=True)
class ContinualEpisodeMetrics:
    episode_Order: tuple[str, ...]
    retention_score: float
    forward_transfer: float
    backward_transfer: float
    # None: recovery is not observed.
    recovery_steps: None
    state_growth_bytes: int
    isolated_episodes: bool
    evidence: EvidenceBasis = "caller_supplied"


def evaluate_continual_protocol(
    *,
    episode_order: tuple[str, ...],
    pre_matrix: list[list[float]],
    post_matrix: list[list[float]],
    state_growth_bytes: int,
    cross_episode_holdout_contaminated: bool = False,
) -> ContinualEpisodeMetrics:
    """Summarises caller-supplied score matrices; runs no episodes (T083)."""
    if cross_episode_holdout_contaminated:
        raise M4ContractError(
            "CONTINUAL_EPISODE_CONTAMINATION",
            "cross-episode or split contamination detected in continual protocol",
        )
    n = len(episode_order)
    if n < 2 or len(pre_matrix) != n or len(post_matrix) != n:
        raise M4ContractError("INVALID_CONTINUAL_MATRIX", "continual protocol requires >=2 square episode matrices")
    bwt = sum(post_matrix[n - 1][j] - post_matrix[j][j] for j in range(n - 1)) / (n - 1)
    fwt = sum(pre_matrix[j][j] for j in range(1, n)) / (n - 1)
    retention = sum(post_matrix[n - 1][j] for j in range(n - 1)) / (n - 1)
    return ContinualEpisodeMetrics(
        episode_Order=episode_order,
        retention_score=retention,
        forward_transfer=fwt,
        backward_transfer=bwt,
        recovery_steps=None,
        state_growth_bytes=state_growth_bytes,
        isolated_episodes=True,
    )


# ============================================================================
# T084–T089: External Gym Adapters, Private Policy, Measurement Research, Community Gym
# ============================================================================


@dataclass(frozen=True)
class ExternalGymRunReceipt:
    adapter_family: Literal["nemo_gym", "verifiers", "gymnasium"]
    pinned_version: str
    episode_count: int
    mean_reward: float
    task_accuracy: float | None
    reward_components: dict[str, float]
    # These adapters run in-process and do not import the named package; pinned_version is only a
    # label the caller must match.
    sandbox_used: Literal[False]
    external_package_executed: Literal[False]
    # None: no replay was performed.
    replay_deterministic: bool | None
    evidence: EvidenceBasis = "builtin_fixture_stub"


def run_nemo_adapter(
    *,
    pinned_version: str = "0.4.0",
    prompts: list[str],
    model_fn: Any,
    is_mock_only: bool = False,
) -> ExternalGymRunReceipt:
    """Calls model_fn twice per prompt in-process and averages its scores; does not run NeMo-Gym (T084)."""
    if is_mock_only:
        raise M4ContractError("MOCK_QUALIFICATION_DENIED", "mock-only NeMo-Gym runs cannot qualify adapter")
    if pinned_version != "0.4.0":
        raise M4ContractError("UNPINNED_NEMO_VERSION", f"unsupported NeMo-Gym version '{pinned_version}'")
    if not prompts:
        raise M4ContractError("EMPTY_NEMO_PROMPTS", "non-empty prompt batch required")
    r1 = [float(model_fn(p)) for p in prompts]
    r2 = [float(model_fn(p)) for p in prompts]
    if r1 != r2:
        raise M4ContractError("NEMO_REPLAY_NONDETERMINISTIC", "local NeMo-Gym replay drifted")
    mean_r = sum(r1) / len(r1)
    return ExternalGymRunReceipt(
        adapter_family="nemo_gym",
        pinned_version=pinned_version,
        episode_count=len(prompts),
        mean_reward=mean_r,
        task_accuracy=None,
        reward_components={"model_fn_score": mean_r},
        sandbox_used=False,
        external_package_executed=False,
        replay_deterministic=True,
    )


def run_verifiers_adapter(
    *,
    pinned_version: str = "0.2.1",
    rollouts: list[dict[str, float]],
    conflate_reward_as_accuracy: bool = False,
) -> ExternalGymRunReceipt:
    """Averages caller-supplied rollout scores, keeping reward apart from accuracy; does not run Verifiers (T085)."""
    if conflate_reward_as_accuracy:
        raise M4ContractError(
            "REWARD_AS_ACCURACY_FORBIDDEN",
            "shaped verifier reward components must not be conflated with binary task accuracy (T085)",
        )
    if pinned_version != "0.2.1" or not rollouts:
        raise M4ContractError("INVALID_VERIFIERS_RUN", "pinned version 0.2.1 and non-empty rollouts required")
    fmt_scores = [float(r["format_reward"]) for r in rollouts]
    correctness = [float(r["correctness_binary"]) for r in rollouts]
    total_r = [0.2 * f + 0.8 * c for f, c in zip(fmt_scores, correctness, strict=True)]
    return ExternalGymRunReceipt(
        adapter_family="verifiers",
        pinned_version=pinned_version,
        episode_count=len(rollouts),
        mean_reward=sum(total_r) / len(total_r),
        task_accuracy=sum(correctness) / len(correctness),
        reward_components={
            "format_reward": sum(fmt_scores) / len(fmt_scores),
            "correctness_binary": sum(correctness) / len(correctness),
        },
        sandbox_used=False,
        external_package_executed=False,
        replay_deterministic=None,
        evidence="caller_supplied",
    )


def run_gymnasium_adapter(
    *,
    pinned_version: str = "1.0.0",
    seed: int = 11,
    max_steps: int = 5,
    action_bridge: Any | None = None,
) -> ExternalGymRunReceipt:
    """Runs a built-in toy dynamics loop through action_bridge, twice; does not run Gymnasium (T086)."""
    if action_bridge is None:
        raise M4ContractError(
            "MISSING_MODEL_ACTION_BRIDGE",
            "Gymnasium adapter requires an explicit model-to-action bridge (T086)",
        )

    def episode() -> float:
        state = float(seed % 5)
        episode_return = 0.0
        for step in range(max_steps):
            action = float(action_bridge(state, step))
            state = state * 0.5 + action
            episode_return += 1.0 / (1.0 + abs(state))
        return episode_return

    total_reward = episode()
    replayed = episode() == total_reward
    return ExternalGymRunReceipt(
        adapter_family="gymnasium",
        pinned_version=pinned_version,
        episode_count=1,
        mean_reward=total_reward,
        task_accuracy=None,
        reward_components={"episode_return": total_reward, "steps": float(max_steps)},
        sandbox_used=False,
        external_package_executed=False,
        replay_deterministic=replayed,
    )


@dataclass(frozen=True)
class PrivateGymAuditReceipt:
    namespace: str
    encrypted_at_rest: bool
    key_id: str
    redacted_log_summary: str
    publication_approved: bool
    training_approved: bool
    # Nothing is encrypted or inspected here; encrypted_at_rest repeats the caller's declaration.
    evidence: EvidenceBasis = "caller_supplied"


def enforce_private_gym_policy(
    *,
    namespace: str,
    encrypted_at_rest: bool,
    key_id: str,
    export_destination: Literal["internal_encrypted_cas", "public_unencrypted_bucket"],
    log_contains_raw_payload_or_path: bool = False,
    publication_approved: bool = False,
    training_approved: bool = False,
) -> PrivateGymAuditReceipt:
    """Enforces private gym encryption, log redaction, and separate publication/training approvals (T087)."""
    if not encrypted_at_rest or not key_id:
        raise M4ContractError("UNENCRYPTED_PRIVATE_GYM", "private gym requires encryption and key_id")
    if export_destination == "public_unencrypted_bucket":
        raise M4ContractError("UNENCRYPTED_EXPORT_DENIED", "private gym cannot export to unencrypted destination")
    if log_contains_raw_payload_or_path:
        raise M4ContractError(
            "PRIVATE_GYM_LOG_DISCLOSURE",
            "private gym logs must not disclose raw item payloads, hashes, or filesystem paths",
        )
    return PrivateGymAuditReceipt(
        namespace=namespace,
        encrypted_at_rest=True,
        key_id=key_id,
        redacted_log_summary=f"namespace={namespace} status=evaluated items=redacted",
        publication_approved=publication_approved,
        training_approved=training_approved,
    )


@dataclass(frozen=True)
class MeasurementResearchReport:
    initial_gym_count: int
    pruned_non_redundant_gym_count: int
    redundant_pairs: tuple[tuple[str, str], ...]
    shortcut_gyms_flagged: tuple[str, ...]
    human_approval_recorded: bool
    evidence: EvidenceBasis = "caller_supplied"


def evaluate_measurement_research(
    gym_score_vectors: dict[str, list[float]],
    *,
    shortcut_suspected_gyms: tuple[str, ...] = (),
    human_approval_recorded: bool = False,
    claim_progress_from_raw_task_count: bool = False,
) -> MeasurementResearchReport:
    """Detects redundant/duplicated gym distributions and shortcut vulnerabilities (T088)."""
    if claim_progress_from_raw_task_count:
        raise M4ContractError(
            "TASK_COUNT_INFLATION_REJECTED",
            "increasing raw gym/task count without reducing redundancy or shortcuts is not measurement progress",
        )
    if not human_approval_recorded:
        raise M4ContractError("MISSING_MEASUREMENT_APPROVAL", "gym portfolio changes require approval")

    names = sorted(gym_score_vectors.keys())
    redundant: list[tuple[str, str]] = []
    dropped: set[str] = set(shortcut_suspected_gyms)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = torch.tensor(gym_score_vectors[names[i]], dtype=torch.float64)
            b = torch.tensor(gym_score_vectors[names[j]], dtype=torch.float64)
            if a.numel() == b.numel() and float(torch.max(torch.abs(a - b)).item()) < 1e-6:
                redundant.append((names[i], names[j]))
                dropped.add(names[j])

    return MeasurementResearchReport(
        initial_gym_count=len(names),
        pruned_non_redundant_gym_count=len(names) - len(dropped),
        redundant_pairs=tuple(redundant),
        shortcut_gyms_flagged=shortcut_suspected_gyms,
        human_approval_recorded=True,
    )


@dataclass(frozen=True)
class CommunityGymPackage:
    package_id: str
    payload_bytes: bytes
    expected_digest: str
    publisher_id: str
    signature_hex: str
    license_spdx: str
    contains_install_script: bool = False
    claimed_historical_score: float | None = None


@dataclass(frozen=True)
class CommunityGymAdmissionReceipt:
    package_id: str
    verified_digest: str
    publisher_verified: bool
    sandbox_isolation_required: bool
    historical_scores_trusted: bool


def verify_community_gym_package(
    pkg: CommunityGymPackage,
    *,
    trusted_publisher_key: bytes,
    execute_on_coordinator: bool = False,
) -> CommunityGymAdmissionReceipt:
    """Validates community gym digests and HMAC signatures offline without trusting scores or install scripts (T089)."""
    if pkg.contains_install_script:
        raise M4ContractError(
            "INSTALL_SCRIPT_FORBIDDEN",
            "community gym packages with install-time scripts are rejected",
        )
    if execute_on_coordinator:
        raise M4ContractError(
            "COORDINATOR_EXECUTION_FORBIDDEN",
            "community gym code must run inside the isolated sandbox, never on the coordinator",
        )
    actual_digest = f"sha256:{hashlib.sha256(pkg.payload_bytes).hexdigest()}"
    if actual_digest != pkg.expected_digest:
        raise M4ContractError("DIGEST_MISMATCH", "community gym payload digest mismatch")
    expected_sig = hmac.new(trusted_publisher_key, pkg.payload_bytes, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected_sig, pkg.signature_hex):
        raise M4ContractError("SIGNATURE_MISMATCH", "community gym publisher signature verification failed")
    if pkg.license_spdx not in ("Apache-2.0", "MIT", "CC-BY-4.0"):
        raise M4ContractError("UNSUPPORTED_LICENSE", f"unsupported community gym license '{pkg.license_spdx}'")

    return CommunityGymAdmissionReceipt(
        package_id=pkg.package_id,
        verified_digest=actual_digest,
        publisher_verified=True,
        sandbox_isolation_required=True,
        historical_scores_trusted=False,  # Supplied scores are never trusted without local re-evaluation
    )


# ============================================================================
# T091–T099: Cross-Model Learning, Library Reuse, Proposers, and M4 Acceptance
# ============================================================================


@dataclass(frozen=True)
class LibraryReuseAttempt:
    recipe_id: str
    version: int
    version_status: Literal["verified", "disputed", "withdrawn", "proposed"]
    target_model_id: str
    target_task_id: str
    prerequisites_satisfied: bool
    source_digest_verified: bool
    fresh_target_evaluation_passed: bool
    accepted_on_target: bool
    failure_recorded: bool
    # Digest, prerequisite and evaluation outcomes are the caller's booleans; none is checked here.
    evidence: EvidenceBasis = "caller_supplied"


def evaluate_library_recipe_reuse(
    *,
    recipe_id: str,
    version: int,
    version_status: Literal["verified", "disputed", "withdrawn", "proposed"],
    target_model_id: str,
    target_task_id: str,
    prerequisites_satisfied: bool,
    source_digest_verified: bool,
    fresh_target_evaluation_passed: bool | None,
    accept_by_shape_compatibility_alone: bool = False,
) -> LibraryReuseAttempt:
    """Enforces fresh target verification for library recipe reuse and retains failed attempts (T091, CG-13)."""
    if accept_by_shape_compatibility_alone or fresh_target_evaluation_passed is None:
        raise M4ContractError(
            "INHERITED_ACCEPTANCE_FORBIDDEN",
            "library recipes never inherit acceptance on a new model or task by shape compatibility alone (CG-13)",
        )
    if version_status in ("disputed", "withdrawn", "proposed"):
        raise M4ContractError(
            "NON_VERIFIED_RECIPE_REUSE_DENIED",
            f"cannot reuse recipe {recipe_id}@v{version} with status '{version_status}'",
        )
    if not prerequisites_satisfied or not source_digest_verified:
        return LibraryReuseAttempt(
            recipe_id=recipe_id,
            version=version,
            version_status=version_status,
            target_model_id=target_model_id,
            target_task_id=target_task_id,
            prerequisites_satisfied=prerequisites_satisfied,
            source_digest_verified=source_digest_verified,
            fresh_target_evaluation_passed=False,
            accepted_on_target=False,
            failure_recorded=True,
        )
    accepted = bool(fresh_target_evaluation_passed)
    return LibraryReuseAttempt(
        recipe_id=recipe_id,
        version=version,
        version_status=version_status,
        target_model_id=target_model_id,
        target_task_id=target_task_id,
        prerequisites_satisfied=True,
        source_digest_verified=True,
        fresh_target_evaluation_passed=accepted,
        accepted_on_target=accepted,
        failure_recorded=not accepted,
    )


@dataclass(frozen=True)
class StrategyRecommendation:
    strategy_id: str
    architecture_family: str
    failure_feature: str
    supporting_studies: int
    contradicting_studies: int
    estimated_cost_cpu_ms: int
    evidence_grade: SupportGrade
    is_accepted_edit: bool = False
    evidence: EvidenceBasis = "caller_supplied"


def retrieve_repair_strategies(
    history: list[dict[str, Any]],
    *,
    architecture_family: str,
    failure_feature: str,
    include_target_holdout: bool = False,
) -> list[StrategyRecommendation]:
    """Emits ranked strategy recommendations with contradiction counts and never emits accepted edits (T092)."""
    if include_target_holdout:
        raise M4ContractError("STRATEGY_HOLDOUT_LEAKAGE", "strategy retrieval cannot read target holdout outcomes")
    grouped: dict[str, dict[str, int]] = {}
    for row in history:
        if row.get("architecture_family") != architecture_family or row.get("failure_feature") != failure_feature:
            continue
        sid = str(row["strategy_id"])
        entry = grouped.setdefault(sid, {"sup": 0, "con": 0, "cost": 0})
        if row.get("outcome") == "success":
            entry["sup"] += 1
        else:
            entry["con"] += 1
        if "cost_cpu_ms" not in row:
            raise M4ContractError("MISSING_STRATEGY_COST", f"history row for '{sid}' has no cost_cpu_ms")
        entry["cost"] += int(row["cost_cpu_ms"])

    recs: list[StrategyRecommendation] = []
    for sid, stats in grouped.items():
        grade: SupportGrade = (
            "contradicted"
            if stats["con"] > stats["sup"]
            else ("suggestive" if stats["sup"] > 0 and stats["con"] == 0 else "inconclusive")
        )
        recs.append(
            StrategyRecommendation(
                strategy_id=sid,
                architecture_family=architecture_family,
                failure_feature=failure_feature,
                supporting_studies=stats["sup"],
                contradicting_studies=stats["con"],
                estimated_cost_cpu_ms=stats["cost"],
                evidence_grade=grade,
                is_accepted_edit=False,
            )
        )
    return sorted(recs, key=lambda r: (r.supporting_studies - r.contradicting_studies), reverse=True)


@dataclass(frozen=True)
class CrossModelTransferStudyResult:
    source_model_group: str
    target_model_group: str
    no_history_evals_to_success: int
    random_proposer_evals_to_success: int
    history_informed_evals_to_success: int
    source_discovery_cost_ms: int
    target_adaptation_cost_ms: int
    sample_efficiency_improved: bool
    evidence: EvidenceBasis = "caller_supplied"


def evaluate_cross_model_protocol(
    *,
    source_model_group: str,
    target_model_group: str,
    shared_checkpoint_ancestry: bool,
    tuned_strategy_on_target_holdout: bool,
    no_history_evals: int,
    random_evals: int,
    history_informed_evals: int,
    source_discovery_cost_ms: int,
    target_adaptation_cost_ms: int,
) -> CrossModelTransferStudyResult:
    """Compares caller-supplied evaluation counts and checks study design; runs no study (T093)."""
    if source_model_group == target_model_group or shared_checkpoint_ancestry:
        raise M4ContractError(
            "CROSS_MODEL_ANCESTRY_OVERLAP",
            "source and target model groups must not share checkpoint ancestry in cross-model transfer",
        )
    if tuned_strategy_on_target_holdout:
        raise M4ContractError(
            "TARGET_OUTCOME_STRATEGY_TUNING",
            "strategy selection cannot be tuned on target holdout outcomes",
        )
    if source_discovery_cost_ms <= 0 or target_adaptation_cost_ms <= 0:
        raise M4ContractError("MISSING_TRANSFER_COST", "both source discovery and target adaptation costs are required")

    return CrossModelTransferStudyResult(
        source_model_group=source_model_group,
        target_model_group=target_model_group,
        no_history_evals_to_success=no_history_evals,
        random_proposer_evals_to_success=random_evals,
        history_informed_evals_to_success=history_informed_evals,
        source_discovery_cost_ms=source_discovery_cost_ms,
        target_adaptation_cost_ms=target_adaptation_cost_ms,
        sample_efficiency_improved=(
            history_informed_evals < no_history_evals and history_informed_evals < random_evals
        ),
    )


@dataclass(frozen=True)
class ProposerDatasetArtifact:
    dataset_digest: str
    train_count: int
    val_count: int
    deduplicated_dropped: int
    license_spdx: str


def build_proposer_dataset(
    records: list[dict[str, Any]],
    *,
    license_spdx: str = "Apache-2.0",
) -> ProposerDatasetArtifact:
    """Builds a deduplicated proposer training dataset while rejecting holdout/private or near-duplicate cross-split leakage (T094)."""
    seen_keys: dict[str, str] = {}
    train_count = 0
    val_count = 0
    dedup_dropped = 0
    for rec in records:
        access = str(rec.get("access", "development"))
        if access in ("holdout", "private"):
            raise M4ContractError(
                "RESTRICTED_PROPOSER_RECORD",
                f"proposer dataset cannot ingest '{access}' record",
            )
        norm_key = str(rec.get("failure_signature", "")).strip().lower()
        split = str(rec.get("split", "train"))
        if norm_key in seen_keys:
            if seen_keys[norm_key] != split:
                raise M4ContractError(
                    "CROSS_SPLIT_NEAR_DUPLICATE",
                    f"near-identical failure_signature '{norm_key}' appears across train and val splits",
                )
            dedup_dropped += 1
            continue
        seen_keys[norm_key] = split
        if split == "train":
            train_count += 1
        else:
            val_count += 1

    return ProposerDatasetArtifact(
        dataset_digest=_digest(
            {"train": train_count, "val": val_count, "license": license_spdx, "signatures": seen_keys}
        ),
        train_count=train_count,
        val_count=val_count,
        deduplicated_dropped=dedup_dropped,
        license_spdx=license_spdx,
    )


@dataclass(frozen=True)
class TrainedProposerArtifact:
    proposer_artifact_digest: str
    training_study_id: str
    dataset_digest: str
    steps: int
    compute_flops: int


def run_proposer_training_study(
    dataset: ProposerDatasetArtifact,
    *,
    training_study_id: str,
    triggered_implicitly_inside_direct_study: bool = False,
    authorized_proposer_training: bool = True,
    steps: int = 20,
) -> TrainedProposerArtifact:
    """Not implemented: no proposer training backend exists. Enforces the contract, then raises (T095)."""
    if triggered_implicitly_inside_direct_study or not authorized_proposer_training:
        raise M4ContractError(
            "IMPLICIT_PROPOSER_TRAINING_FORBIDDEN",
            "direct target-model studies cannot silently trigger proposer training (T095)",
        )
    raise NotImplementedError(
        f"proposer training is not implemented (study '{training_study_id}', dataset "
        f"{dataset.dataset_digest}, {steps} steps); no model would be trained"
    )


@dataclass(frozen=True)
class LearnedProposerEvaluation:
    proposer_artifact_digest: str
    ir_valid_rate: float
    learned_evals_to_success: int
    no_history_evals_to_success: int
    random_evals_to_success: int
    target_update_classification: Literal["direct_tensor_edit"]
    proposal_origin: Literal["learned_proposer"]
    # Rates and counts are copied from the arguments; no proposer is run.
    evidence: EvidenceBasis = "caller_supplied"


def evaluate_learned_proposer_inference(
    proposer: TrainedProposerArtifact,
    *,
    ir_valid_rate: float,
    learned_evals_to_success: int,
    no_history_evals_to_success: int,
    random_evals_to_success: int,
    target_used_gradient_updates: bool = False,
) -> LearnedProposerEvaluation:
    """Records caller-supplied proposer metrics and checks lineage; runs no proposer (T096)."""
    if target_used_gradient_updates:
        raise M4ContractError(
            "LEARNED_ORIGIN_LINEAGE_ERASURE",
            "learned proposer origin cannot erase target-model fitting/gradient lineage (T096)",
        )
    if ir_valid_rate <= 0.0 or learned_evals_to_success <= 0:
        raise M4ContractError("INVALID_PROPOSER_EVAL", "positive validity rate and evals-to-success required")
    return LearnedProposerEvaluation(
        proposer_artifact_digest=proposer.proposer_artifact_digest,
        ir_valid_rate=ir_valid_rate,
        learned_evals_to_success=learned_evals_to_success,
        no_history_evals_to_success=no_history_evals_to_success,
        random_evals_to_success=random_evals_to_success,
        target_update_classification="direct_tensor_edit",
        proposal_origin="learned_proposer",
    )


@dataclass(frozen=True)
class TransferBenchmarkReport:
    scope_label: Literal["same_architecture", "cross_size", "cross_family"]
    unchanged_mse: float
    safe_unaligned_mse: float
    aligned_direct_mse: float
    matched_lora_mse: float | None
    missing_comparisons: tuple[str, ...]
    total_cost_flops: int


def run_transfer_benchmark(
    *,
    scope_label: Literal["same_architecture", "cross_size", "cross_family"],
    is_synthetic_permutation: bool,
    force_incompatible_raw_copy: bool = False,
    include_lora_comparison: bool = True,
) -> TransferBenchmarkReport:
    """Not implemented: it used to return fixed MSE values. Enforces the contract, then raises (T097).

    execute_mapped_weight_transfer measures one transfer on supplied tensors.
    """
    if force_incompatible_raw_copy:
        raise M4ContractError(
            "INCOMPATIBLE_RAW_COPY_FORBIDDEN",
            "incompatible cross-size/cross-family raw tensor copies must never be forced (T097)",
        )
    if is_synthetic_permutation and scope_label != "same_architecture":
        raise M4ContractError(
            "SYNTHETIC_PERMUTATION_MISLABELED",
            "permuted synthetic fixtures cannot be labeled as cross_size or cross_family",
        )
    raise NotImplementedError(
        f"transfer benchmark is not implemented for scope '{scope_label}' "
        f"(include_lora_comparison={include_lora_comparison}); no measurement exists"
    )


@dataclass(frozen=True)
class ResearchSourceRecord:
    source_id: str
    version: str
    license_spdx: str
    claim_summary: str
    origin: EvidenceOrigin
    support: SupportGrade
    training_corpus_provenance: str | None
    # A local reproduction is only ever declared here. This function cannot observe a study.
    local_reproduction_declared: bool = False
    evidence: EvidenceBasis = "caller_supplied"


def ingest_research_source_record(
    *,
    source_id: str,
    version: str,
    license_spdx: str,
    claim_summary: str,
    locally_reproduced_under_protocol: bool = False,
    training_corpus_provenance: str | None = None,
    fabricate_missing_corpus_attribution: bool = False,
    claim_reproduced_without_local_study: bool = False,
) -> ResearchSourceRecord:
    """Records a source as a supplied fact; a declared local reproduction never raises its grade (T098)."""
    if fabricate_missing_corpus_attribution:
        raise M4ContractError(
            "FABRICATED_DATA_ATTRIBUTION",
            "cannot fabricate training-data-to-circuit attribution when pretraining corpus is unavailable (T098)",
        )
    if claim_reproduced_without_local_study and not locally_reproduced_under_protocol:
        raise M4ContractError(
            "UNVERIFIED_REPRODUCTION_CLAIM",
            "external paper claims cannot be marked independently_confirmed_under_protocol without a local study",
        )
    return ResearchSourceRecord(
        source_id=source_id,
        version=version,
        license_spdx=license_spdx,
        claim_summary=claim_summary,
        origin="supplied_fact",
        support="suggestive",
        training_corpus_provenance=training_corpus_provenance,
        local_reproduction_declared=locally_reproduced_under_protocol,
    )


@dataclass(frozen=True)
class M4FeatureAcceptanceReport:
    cg09_construction_passed: bool
    cg10_transfer_passed: bool
    cg11_methods_and_costs_passed: bool
    cg12_evidence_quality_passed: bool
    cg13_library_lifecycle_passed: bool
    # The merged LoRA fixture weight was written to safetensors and read back identical; part of CG-11.
    cg11_reload_parity_verified: bool
    # Empty: these gates exercise fixtures and contract refusals, so they qualify no engine.
    qualified_engines: tuple[str, ...]
    all_gates_passed: bool
    real_model_evaluated: Literal[False] = False
    evidence: EvidenceBasis = "measured_on_fixture"


_BASELINE_WEIGHT_NAME = "merged_weight"


def reload_exported_baseline_weight(checkpoint_dir: Path) -> torch.Tensor:
    """Read a weight back from the exported safetensors bytes, sharing nothing with the tensor that was written."""
    blob = _map_file(checkpoint_dir / "model.safetensors")
    header, data_start = _parse_safetensors_header(blob)
    meta = header[_BASELINE_WEIGHT_NAME]
    if meta["dtype"] != "F64":
        raise M4ContractError("RELOAD_PARITY_FAILED", f"expected an F64 weight, found {meta['dtype']}")
    start, end = meta["data_offsets"]
    shape = [int(d) for d in meta["shape"]]
    values = struct.unpack(f"<{math.prod(shape)}d", blob[data_start + start : data_start + end])
    return torch.tensor(values, dtype=torch.float64).reshape(shape)


def verify_baseline_reload_parity(merged: torch.Tensor, probe_inputs: torch.Tensor, checkpoint_dir: Path) -> bool:
    """True only if the exported weight reloads with identical bits and identical outputs on the probes.

    This is a same-process file round trip of one fixture matrix, not a fresh-process model reload.
    """
    reloaded = reload_exported_baseline_weight(checkpoint_dir)
    if reloaded.shape != merged.shape or not torch.equal(reloaded, merged):
        return False
    return torch.equal(probe_inputs @ reloaded.T, probe_inputs @ merged.T)


def _lora_reload_parity(merged: torch.Tensor, probe_inputs: torch.Tensor) -> bool:
    with tempfile.TemporaryDirectory() as scratch:
        fixture = SimpleNamespace(tensors={_BASELINE_WEIGHT_NAME: merged}, config={})
        export_huggingface_safetensors(fixture, scratch, dtype="f64")
        return verify_baseline_reload_parity(merged, probe_inputs, Path(scratch))


def run_m4_feature_acceptance_suite(
    *,
    allow_partial_qualification: bool = False,
) -> M4FeatureAcceptanceReport:
    """Software-path checks on small fixtures for CG-09 to CG-13 (T099). A pass is not a qualification.

    CG-09 to CG-11 compute on tiny tensors; CG-11 also exports the merged LoRA weight and checks its
    reload. CG-12 and CG-13 check only that the labelling and reuse
    contracts refuse bad inputs.
    """
    if allow_partial_qualification:
        raise M4ContractError("PARTIAL_QUALIFICATION_DENIED", "M4 acceptance requires all capability gates to pass")

    # 1. CG-09: Rule construction with independent oracle
    def oracle_five(x: float, y: float) -> tuple[float, float, float, float, float]:
        return (x + y, x - y, max(x, y), min(x, y), abs(x - y))

    constructed = construct_algorithmic_relu_rule(independent_oracle=oracle_five)
    cg09_ok = (
        constructed.normal_pass_rate == 1.0
        and constructed.boundary_pass_rate == 1.0
        and constructed.invalid_rejected_properly
    )

    # 2. CG-10: Mapped weight transfer with permutation alignment and controls
    t_w1 = torch.tensor([[1.0, 2.0], [-1.0, 1.5], [0.5, -0.5]], dtype=torch.float64)
    t_w2 = torch.tensor([[1.0, -0.5, 2.0]], dtype=torch.float64)
    perm_gt = (2, 0, 1)
    probe_x = torch.tensor(
        [[1.0, 0.5], [-0.5, 2.0], [1.5, -1.0], [0.2, 0.8], [-1.2, -0.4]],
        dtype=torch.float64,
    )
    t_act = torch.relu(t_w1 @ probe_x.T)
    s_act = t_act[list(perm_gt), :]
    align = estimate_permutation_alignment(t_act, s_act, supplied_permutation=perm_gt, mode="supplied_map")
    s_w1_init = torch.zeros_like(t_w1)
    s_w2_init = torch.zeros_like(t_w2)
    target_y = (torch.relu(probe_x @ t_w1.T)) @ t_w2.T
    transfer_rep = execute_mapped_weight_transfer(
        teacher_w1=t_w1,
        teacher_w2=t_w2,
        student_w1=s_w1_init,
        student_w2=s_w2_init,
        alignment=align,
        eval_inputs=probe_x,
        eval_targets=target_y,
    )
    cg10_ok = transfer_rep.functional_transfer_verified and transfer_rep.aligned_transfer_mse < 1e-10

    # 3. CG-11: Authorized baseline cost accounting & reload parity
    lora_merged, lora_rec = run_lora_baseline(
        torch.zeros(1, 2, dtype=torch.float64),
        probe_x,
        target_y,
        authorized_in_contract=True,
        rank=1,
        steps=8,
    )
    cg11_reload_ok = _lora_reload_parity(lora_merged, probe_x)
    cg11_ok = lora_rec.eval_loss_after < lora_rec.eval_loss_before and cg11_reload_ok

    # 4. CG-12: a declared reproduction must not raise a source's evidence grade
    src_rec = ingest_research_source_record(
        source_id="fixture.source",
        version="1.0",
        license_spdx="Apache-2.0",
        claim_summary="fixture claim",
        locally_reproduced_under_protocol=True,
    )
    cg12_ok = src_rec.origin == "supplied_fact" and src_rec.support == "suggestive"
    try:
        ingest_research_source_record(
            source_id="fixture.source",
            version="1.0",
            license_spdx="Apache-2.0",
            claim_summary="fixture claim",
            claim_reproduced_without_local_study=True,
        )
        cg12_ok = False
    except M4ContractError:
        pass

    # 5. CG-13: a failed fresh evaluation is not accepted, and shape alone is refused
    reuse_rec = evaluate_library_recipe_reuse(
        recipe_id="fixture.recipe",
        version=1,
        version_status="verified",
        target_model_id="fixture.target",
        target_task_id="fixture.task",
        prerequisites_satisfied=True,
        source_digest_verified=True,
        fresh_target_evaluation_passed=False,
    )
    cg13_ok = not reuse_rec.accepted_on_target and reuse_rec.failure_recorded
    try:
        evaluate_library_recipe_reuse(
            recipe_id="fixture.recipe",
            version=1,
            version_status="verified",
            target_model_id="fixture.target",
            target_task_id="fixture.task",
            prerequisites_satisfied=True,
            source_digest_verified=True,
            fresh_target_evaluation_passed=None,
        )
        cg13_ok = False
    except M4ContractError:
        pass

    all_ok = cg09_ok and cg10_ok and cg11_ok and cg12_ok and cg13_ok
    return M4FeatureAcceptanceReport(
        cg09_construction_passed=cg09_ok,
        cg10_transfer_passed=cg10_ok,
        cg11_methods_and_costs_passed=cg11_ok,
        cg11_reload_parity_verified=cg11_reload_ok,
        cg12_evidence_quality_passed=cg12_ok,
        cg13_library_lifecycle_passed=cg13_ok,
        qualified_engines=(),
        all_gates_passed=all_ok,
    )
