"""Cumulative edit search: scope is not fixed in advance; it grows only when a step pays for itself.

Declared before any run (see docs/retention-multisite/PROTOCOL.md):

    J = task + retention - 0.1 * KL - cost_per_parameter * changed_parameters

`task` is accuracy on the development and extra-prefix items. `retention` and `KL` are measured on the accept guard,
a group-wise part of the fit corpus that solves never see. A candidate is accepted when it has the highest J of its
round and raises J by at least MIN_GAIN. The search stops at the first round with no such candidate.

Reporting splits (validation, report prefixes, practice, tuning) are measured each round and never used to accept.
Every accepted step is an `EditPlan.low_rank` step, so the sequence replays to the same digest.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
import random
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from . import baselines as bl
from . import retention_multisite as ms
from . import retention_research as rr
from .edit_plan import EditConstraints, EditPlan, apply_edit_graph, tensor_state_digest
from .goal_search import (
    STRICT_GENERAL_PROFILE,
    AcceptanceProfile,
    CandidateArchive,
    CandidateArtifact,
    CandidateRecord,
    FrontierSnapshot,
    NextMoveDecision,
    SearchBudgets,
    SearchObservation,
    diagnose_next_move,
)
from .gpu import (
    ExternalGpuRuntime,
    GpuDispatchError,
    GpuResourceScheduler,
    TrainingComputePlacement,
    discover_external_gpu_runtime,
    select_training_compute,
)
from .pythia import ALGORITHMIC_THREE_SKILL_GYM as GYM
from .pythia import PythiaModel, ReusableTaskGym
from .records import canonicalize, content_digest
from .study.arms import placebo_weights

KL_WEIGHT = 0.1
MIN_GAIN = 0.005
PRIMARY_COST = 1e-7  # One 65,536-parameter tensor costs 0.0066, about one accept-guard prompt.
SENSITIVITY_COSTS = (0.0, 1e-6)
SAFETY_ROUNDS = 40  # A hard safety limit, not a design choice. The output says if it binds.
STRENGTHS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
TASK_ALPHAS = (1e3, 1e5)
REPAIR = ((1e3, 1.0), (1e5, 1.0), (1e3, 100.0), (1e5, 100.0))
RANK_LIMIT = 12
REPORT_PREFIXES = ("Example:", "Case:", "Sample:")  # Base rules only, used only for reporting.
EXECUTOR_SOURCE_LIMIT_BYTES = 16_777_216


class TrainingBudgetExpired(RuntimeError):
    """The frozen search budget expired during a local training recipe."""


def candidate_sites(model: Any) -> list[str]:
    """Matrices with an implemented first-order closed-form step for the model architecture."""
    layers = int(model.config["num_hidden_layers"])
    model_type = str(model.config.get("model_type", ""))
    if model_type == "gpt_neox":
        names = ("mlp.dense_4h_to_h", "attention.dense", "mlp.dense_h_to_4h")
        prefix = "gpt_neox"
    elif model_type == "qwen2":
        names = ("mlp.down_proj", "self_attn.o_proj")
        prefix = "model"
    else:
        raise ValueError(f"Cumulative search has no candidate-site adapter for {model_type!r}")
    return [f"{prefix}.layers.{layer}.{name}.weight" for layer in range(layers) for name in names]


def excluded_sites(model: Any) -> dict[str, list[str]]:
    open_sites = set(candidate_sites(model))
    out: dict[str, list[str]] = {"executor_source_bound": [], "no_closed_form_implemented": [], "not_a_matrix": []}
    for name, tensor in model.tensors.items():
        if name in open_sites:
            continue
        if tensor.ndim != 2:
            out["not_a_matrix"].append(name)
        elif tensor.numel() * 8 > EXECUTOR_SOURCE_LIMIT_BYTES:
            out["executor_source_bound"].append(name)
        else:
            out["no_closed_form_implemented"].append(name)
    return out


def accept_split(
    seed: int = 41,
    accept_share: float = 0.4,
    *,
    model: Any | None = None,
    gym: Any = GYM,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Group-wise split of encodable fit prompts into (solve, accept)."""
    fit = set(rr.split_corpus(gym=gym)[0])
    groups = [
        (g, [p for p in prompts if p in fit and (model is None or model.tokenizer.encode(p))])
        for g, prompts in sorted(rr.build_corpus(gym).items())
    ]
    groups = [g for g in groups if g[1]]
    random.Random(seed).shuffle(groups)
    target = accept_share * sum(len(p) for _, p in groups)
    solve: list[str] = []
    accept: list[str] = []
    for _, prompts in groups:
        (accept if len(accept) < target else solve).extend(prompts)
    return tuple(solve), tuple(accept)


@dataclass(frozen=True)
class SearchSplits:
    solve_guard: tuple[str, ...]
    accept_guard: tuple[str, ...]
    monitor_guard: tuple[str, ...]


def goal_search_splits(
    seed: int = 41,
    solve_share: float = 0.4,
    accept_share: float = 0.4,
    *,
    model: Any | None = None,
    gym: Any = GYM,
) -> SearchSplits:
    """Partition fit prompts by group into solve, accept, and an untouched monitor split."""
    monitor_share = 1.0 - solve_share - accept_share
    if solve_share <= 0 or accept_share <= 0 or monitor_share <= 0:
        raise ValueError("solve, accept and monitor shares must all be positive and sum to 1")
    fit = set(rr.split_corpus(gym=gym)[0])
    groups = [
        (g, [p for p in prompts if p in fit and (model is None or model.tokenizer.encode(p))])
        for g, prompts in sorted(rr.build_corpus(gym).items())
    ]
    groups = [g for g in groups if g[1]]
    random.Random(seed).shuffle(groups)
    total = sum(len(prompts) for _, prompts in groups)
    targets = (solve_share * total, accept_share * total)
    solve: list[str] = []
    accept: list[str] = []
    monitor: list[str] = []
    for _, prompts in groups:
        if len(solve) < targets[0]:
            solve.extend(prompts)
        elif len(accept) < targets[1]:
            accept.extend(prompts)
        else:
            monitor.extend(prompts)
    if not solve or not accept or not monitor:
        raise ValueError("The corpus has too few prompt groups for a three-way goal-search split")
    return SearchSplits(tuple(solve), tuple(accept), tuple(monitor))


@dataclass(frozen=True)
class Move:
    site: str
    mode: str  # "task" moves the task target and protects; "repair" cancels drift and holds the task effect.
    strength: float
    alpha: float
    ridge: float


@dataclass(frozen=True)
class EvaluatedMove:
    candidate_id: str
    proposal_digest: str
    move: Move
    left: torch.Tensor
    right: torch.Tensor
    tensors: Mapping[str, torch.Tensor]
    score: Mapping[str, float]
    rejection_reason: str | None = None


def objective(task: float, retention: float, kl: float, changed: int, cost: float) -> float:
    return task + retention - KL_WEIGHT * kl - cost * changed


def round_moves(model: Any, first_round: bool) -> list[Move]:
    moves: list[Move] = []
    for site in candidate_sites(model):
        moves += [Move(site, "task", s, a, 1.0) for s in STRENGTHS for a in TASK_ALPHAS]
        if not first_round:  # With no edit yet there is no drift to repair.
            moves += [Move(site, "repair", 0.0, a, r) for a, r in REPAIR]
    return moves


def moves_for_policy(moves: Sequence[Move], decision: NextMoveDecision) -> list[Move]:
    if decision.move_policy == "task":
        return [move for move in moves if move.mode == "task"]
    if decision.move_policy == "retention_repair":
        repair = [move for move in moves if move.mode == "repair"]
        return repair + [move for move in moves if move.mode == "task" and move.strength <= 0.2]
    return list(moves) if decision.move_policy != "none" else []


class Capture:
    """Inputs at every site and the final residual at the last token, for one model state."""

    def __init__(self, model: Any, prompts: Sequence[str]) -> None:
        inputs: list[dict[str, torch.Tensor]] = []
        residuals: list[torch.Tensor] = []
        for p in prompts:
            sink: list[torch.Tensor] = []
            site: dict[str, torch.Tensor] = {}
            model.forward_with_mlp_key(model.tokenizer.encode(p), residual_sink=sink, site_inputs=site)
            inputs.append(site)
            residuals.append(sink[0][-1])
        self.residual = torch.stack(residuals, dim=1)
        self.inputs = {name: torch.stack([row[name] for row in inputs], dim=1) for name in inputs[0]}


class CumulativeSearch:
    def __init__(
        self,
        model: Any,
        cost: float,
        *,
        acceptance_profile: AcceptanceProfile | None = None,
        budgets: SearchBudgets | None = None,
        cycle_id: str | None = None,
        gym: Any = GYM,
        include_training_baselines: bool | None = None,
        training_device: str = "auto",
        gpu_python: str | Path | None = None,
        gpu_scheduler: GpuResourceScheduler | None = None,
    ) -> None:
        self.model = model
        self.cost = cost
        self.gym = gym
        gym_digest = str(content_digest(canonicalize(dataclasses.asdict(gym))))
        self.gym_digest = gym_digest
        self.cycle_id = cycle_id or (
            f"cumulative:{acceptance_profile.profile_id if acceptance_profile else 'legacy'}:"
            f"{gym.gym_id}:{gym_digest}:{cost:g}"
        )
        self.mlab = ms.MultiLab(model, gym=gym)
        self.lab = self.mlab.lab
        if acceptance_profile is None:
            self.solve_guard, self.accept_guard = accept_split(model=model, gym=gym)
            self.monitor_guard: tuple[str, ...] = ()
        else:
            splits = goal_search_splits(model=model, gym=gym)
            self.solve_guard = splits.solve_guard
            self.accept_guard = splits.accept_guard
            self.monitor_guard = splits.monitor_guard
        self.acceptance_profile = acceptance_profile
        self.budgets = budgets or SearchBudgets(min_gain=MIN_GAIN, max_rounds=SAFETY_ROUNDS)
        self.archive = CandidateArchive(self.cycle_id, acceptance_profile) if acceptance_profile else None
        self.include_training_baselines = (
            acceptance_profile is not None if include_training_baselines is None else include_training_baselines
        )
        self.artifact_parent: Path | None = None
        self.artifact_directory: Path | None = None
        self.frontier_snapshot: FrontierSnapshot | None = None
        self.external_gpu_runtime = None
        if self.include_training_baselines and gpu_python is not None and training_device != "cpu":
            self.external_gpu_runtime = discover_external_gpu_runtime(
                gpu_python, device=training_device if training_device.startswith("cuda:") else None,
            )
        if self.external_gpu_runtime is not None:
            self.training_compute = TrainingComputePlacement(self.external_gpu_runtime.device, "f32", "verified_external_cuda_kernel")
        else:
            try:
                self.training_compute = select_training_compute(training_device)
            except GpuDispatchError as exc:
                if not self.include_training_baselines or not training_device.startswith("cuda:") or exc.code not in {"GPU_KERNEL_UNAVAILABLE", "GPU_DEVICE_UNAVAILABLE"}:
                    raise
                self.external_gpu_runtime = discover_external_gpu_runtime(device=training_device)
                if self.external_gpu_runtime is None:
                    raise
                self.training_compute = TrainingComputePlacement(self.external_gpu_runtime.device, "f32", "verified_external_cuda_kernel")
        self.gpu_scheduler = gpu_scheduler or GpuResourceScheduler()
        self.gpu_training_lease: dict[str, Any] | None = None
        if self.external_gpu_runtime is None and self.include_training_baselines and training_device == "auto" and self.training_compute.device == "cpu":
            self.external_gpu_runtime = discover_external_gpu_runtime(gpu_python)
        if self.external_gpu_runtime is not None and self.external_gpu_runtime.free_memory_bytes > 0:
            self.gpu_scheduler.vram_capacity_bytes_per_device = min(
                self.gpu_scheduler.vram_capacity_bytes_per_device,
                self.external_gpu_runtime.free_memory_bytes,
            )
        self.search_sites: set[str] = set()
        self.report_items = [
            (prompt, target)
            for prefix in REPORT_PREFIXES
            for expression, target in gym.base_rules
            for prompt in (f"{prefix} {expression}",)
            if model.tokenizer.encode(prompt)
        ]
        self.practice_prompts = tuple(
            prompt for prompt in ms.PRACTICE_RETENTION_PROMPTS if model.tokenizer.encode(prompt)
        )
        if not self.report_items or not self.practice_prompts:
            raise ValueError("The selected tokenizer cannot encode report-only Gym prompts")
        self.plan = EditPlan(model, EditConstraints())
        self.tensors: dict[str, torch.Tensor] = {}
        self.task_target = ms.contrast(self.lab.unit_targets)
        dev = [p for p, _ in self.lab.dev_items]
        self.dev_prompts = dev
        self.base_dev = Capture(model, dev)
        self.base_solve = Capture(model, self.solve_guard)

    def current(self) -> Any:
        return bl.merged_model(self.model, self.tensors)

    def changed(self, tensors: Mapping[str, torch.Tensor]) -> int:
        return ms.changed_parameters(self.model, tensors)

    def training_runtime_document(self) -> dict[str, Any]:
        if self.external_gpu_runtime is not None:
            return {
                "mode": "external_gpu_worker",
                "device": self.external_gpu_runtime.device,
                "dtype": "f32",
                "reason": "verified_external_cuda_kernel",
                "python_executable": self.external_gpu_runtime.python_executable,
                "torch_version": self.external_gpu_runtime.torch_version,
                "device_name": self.external_gpu_runtime.device_name,
                "torch_arch_list": list(self.external_gpu_runtime.torch_arch_list),
                "total_memory_bytes": self.external_gpu_runtime.total_memory_bytes,
                "free_memory_at_preflight_bytes": self.external_gpu_runtime.free_memory_bytes,
                "reservation_scope": "process_local_cooperative_not_native_authority",
                "reservation": getattr(self, "gpu_training_lease", None),
            }
        return {
            "mode": "in_process",
            "device": self.training_compute.device,
            "dtype": self.training_compute.dtype,
            "reason": self.training_compute.reason,
        }

    def _structural_integrity_rate(self, tensors: Mapping[str, torch.Tensor]) -> float:
        checks = (
            all(name in self.model.tensors for name in tensors),
            all(
                name in self.model.tensors and tuple(value.shape) == tuple(self.model.tensors[name].shape)
                for name, value in tensors.items()
            ),
            all(bool(torch.isfinite(value).all().item()) for value in tensors.values()),
        )
        return sum(checks) / len(checks)

    def _archive_scores(
        self,
        score: Mapping[str, float],
        base_task: float,
        elapsed: float,
        tensors: Mapping[str, torch.Tensor],
        artifact: CandidateArtifact | None,
    ) -> dict[str, float | int]:
        return {
            "target_accuracy": float(score["task"]),
            "target_gain": float(score["task"] - base_task),
            "retention_accuracy": float(score["retention"]),
            "kl_nats": float(score["kl"]),
            "structural_integrity_rate": self._structural_integrity_rate(tensors),
            "changed_parameters": int(score["changed"]),
            "wall_seconds": elapsed,
            "artifact_bytes": artifact.file_bytes if artifact is not None else 0,
        }

    @staticmethod
    def _file_digest(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"

    def _materialize_candidate(
        self, candidate_id: str, replacements: Mapping[str, torch.Tensor]
    ) -> tuple[str, CandidateArtifact | None]:
        tensors = {
            name: value.detach().to(device="cpu", dtype=torch.float64).clone()
            for name, value in sorted(replacements.items())
        }
        state_digest = tensor_state_digest({**self.model.tensors, **tensors})
        artifact_directory = getattr(self, "artifact_directory", None)
        artifact_parent = getattr(self, "artifact_parent", None)
        if artifact_directory is None or artifact_parent is None:
            return state_digest, None
        unknown = sorted(set(tensors) - set(self.model.tensors))
        if unknown:
            raise ValueError(f"Cannot materialize unknown tensors: {unknown}")
        if not tensors:
            return state_digest, None
        artifact_directory.mkdir(parents=True, exist_ok=True)
        filename = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:32] + ".pt"
        path = artifact_directory / filename
        temporary = path.with_suffix(".pt.tmp")
        torch.save(tensors, temporary)
        reloaded = torch.load(temporary, map_location="cpu", weights_only=True)
        if set(reloaded) != set(tensors):
            raise RuntimeError("Candidate artifact reload changed tensor names")
        if any(not torch.equal(reloaded[name], tensors[name]) for name in tensors):
            raise RuntimeError("Candidate artifact reload changed tensor values")
        if tensor_state_digest({**self.model.tensors, **reloaded}) != state_digest:
            raise RuntimeError("Candidate artifact reload changed the candidate state digest")
        os.replace(temporary, path)
        relative_path = path.relative_to(artifact_parent).as_posix()
        return state_digest, CandidateArtifact(
            relative_path=relative_path,
            file_digest=self._file_digest(path),
            state_digest=state_digest,
            tensor_names=tuple(tensors),
            file_bytes=path.stat().st_size,
        )

    def _archive_id(
        self,
        number: int,
        parent_id: str,
        move: Move,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> tuple[str, str]:
        proposal = {"round": number, "parent": parent_id, **dataclasses.asdict(move)}
        operands = tensor_state_digest({"left_factor": left, "right_factor": right})
        digest = str(content_digest(canonicalize({**proposal, "operand_digest": operands})))
        return f"round-{number}:{digest.removeprefix('sha256:')[:16]}", digest

    def _archive_round(
        self,
        number: int,
        parent_id: str,
        base_target: float,
        trials: Sequence[EvaluatedMove],
        accepted: EvaluatedMove | None,
        accepted_state_digest: str | None,
        decision: NextMoveDecision,
        elapsed: float,
    ) -> str | None:
        if self.archive is None:
            return None
        for trial in trials:
            chosen = accepted is not None and trial.candidate_id == accepted.candidate_id
            state_digest, artifact = self._materialize_candidate(trial.candidate_id, trial.tensors)
            if chosen and accepted_state_digest != state_digest:
                raise RuntimeError("The Rust plan state digest differs from the archived candidate tensors")
            self.archive.add(
                CandidateRecord(
                    candidate_id=trial.candidate_id,
                    parent_candidate_id=parent_id,
                    round_number=number,
                    arm="direct",
                    method=f"low_rank_{trial.move.mode}",
                    proposal_digest=trial.proposal_digest,
                    state_digest=state_digest,
                    scores=self._archive_scores(trial.score, base_target, elapsed, trial.tensors, artifact),
                    proposal={"move": dataclasses.asdict(trial.move)},
                    search_state="accepted" if chosen else "rejected",
                    replay_verified=chosen,
                    reload_parity_verified=chosen,
                    rejection_reasons=() if chosen else (
                        trial.rejection_reason
                        or ("not_selected" if accepted else "below_min_gain"),
                    ),
                    next_move_reason=decision.rule_id,
                    artifact=artifact,
                )
            )
        if accepted is None:
            return None
        controls = (
            ("random", self._random_weights(accepted.tensors, 0xA11CE + number)),
            ("placebo", placebo_weights(self.model.tensors, accepted.tensors, 0xB0CAE + number)),
        )
        for method, weights in controls:
            if not self._reserve_candidate_evaluation():
                break
            control_score = self.accept_score(weights)
            control_id = f"{accepted.candidate_id}:{method}"
            control_digest, artifact = self._materialize_candidate(control_id, weights)
            self.archive.add(
                CandidateRecord(
                    candidate_id=control_id,
                    parent_candidate_id=accepted.candidate_id,
                    round_number=number,
                    arm="control",
                    method=method,
                    proposal_digest=control_digest,
                    state_digest=control_digest,
                    scores=self._archive_scores(control_score, base_target, elapsed, weights, artifact),
                    proposal={"matched_candidate_id": accepted.candidate_id, "seed": (0xA11CE if method == "random" else 0xB0CAE) + number},
                    search_state="rejected",
                    replay_verified=False,
                    reload_parity_verified=False,
                    rejection_reasons=("control_arm",),
                    next_move_reason=decision.rule_id,
                    artifact=artifact,
                )
            )
        return accepted.candidate_id

    def _random_weights(self, candidate: Mapping[str, torch.Tensor], seed: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(seed)
        random_weights: dict[str, torch.Tensor] = {}
        for name in sorted(candidate):
            delta = candidate[name] - self.model.tensors[name]
            direction = torch.randn(delta.shape, dtype=delta.dtype, device=delta.device, generator=generator)
            norm = torch.linalg.vector_norm(direction)
            if float(norm) == 0:
                raise ValueError(f"Cannot construct random control for unchanged tensor {name!r}")
            random_weights[name] = self.model.tensors[name] + direction * (torch.linalg.vector_norm(delta) / norm)
        return random_weights

    def _archive_training_checkpoint(
        self,
        *,
        base_id: str,
        base_target: float,
        sites: tuple[str, ...],
        kind: str,
        recipe_index: int,
        recipe: Mapping[str, Any],
        step: int,
        weights: dict[str, torch.Tensor],
        wall: float,
    ) -> None:
        assert self.archive is not None
        if not self._reserve_candidate_evaluation():
            return
        score = self.accept_score(weights)
        proposal = {
            "kind": kind,
            "recipe": dict(recipe),
            "recipe_schema_version": 1,
            "dataset": {
                "schema_version": self.lab.task_data.schema_version,
                "gym_id": self.lab.task_data.gym_id,
                "gym_digest": self.lab.task_data.gym_digest,
                "dataset_digest": self.lab.task_data.dataset_digest,
                "train_digest": self.lab.task_data.train_digest,
                "validation_digest": self.lab.task_data.validation_digest,
            },
            "sites": sites,
            "step": step,
            "training_runtime": self.training_runtime_document(),
        }
        proposal_digest = str(content_digest(canonicalize(proposal)))
        candidate_id = (
            f"{kind}-{recipe_index}:{step}:{proposal_digest.removeprefix('sha256:')[:16]}"
        )
        state_digest, artifact = self._materialize_candidate(candidate_id, weights)
        self.archive.add(
            CandidateRecord(
                candidate_id=candidate_id,
                parent_candidate_id=base_id,
                round_number=0,
                arm="baseline",
                method=kind,
                proposal_digest=proposal_digest,
                state_digest=state_digest,
                scores=self._archive_scores(score, base_target, wall, weights, artifact),
                proposal=proposal,
                search_state="accepted",
                replay_verified=False,
                reload_parity_verified=False,
                next_move_reason="training_competitor",
                artifact=artifact,
            )
        )

    def _archive_training_competitors(self, base_id: str, base_target: float) -> None:
        if self.archive is None or not self.include_training_baselines:
            return
        sites = tuple(sorted(self.search_sites)) or bl.lora_sites(self.model, "all_down_proj")
        if not sites:
            raise RuntimeError("No supported training sites are available for baseline candidates")
        jobs: list[dict[str, Any]] = []
        for kind in ("lora", "sft"):
            for recipe_index, recipe in enumerate(restricted_grid(kind)):
                jobs.append(
                    {
                        "job_id": f"{kind}-{recipe_index}",
                        "recipe_schema_version": 1,
                        "kind": kind,
                        "recipe_index": recipe_index,
                        "sites": sites,
                        "recipe": dict(recipe),
                        "dev": tuple(self.lab.dev_items),
                        "guard": tuple(self.solve_guard),
                        "gym_id": self.lab.task_data.gym_id,
                        "gym_digest": self.lab.task_data.gym_digest,
                        "dataset_digest": self.lab.task_data.dataset_digest,
                        "seed": 0,
                    }
                )
        remaining_evaluations = max(0, self.budgets.max_candidate_evaluations - getattr(self, "_evaluations_used", 0))
        bounded_jobs = []
        for job in jobs:
            if remaining_evaluations <= 0 or self._training_seconds_remaining() <= 0:
                reason = "search_wall_budget_exhausted_before_dispatch" if self._training_seconds_remaining() <= 0 else "evaluation_budget_exhausted_before_dispatch"
                self._archive_training_dispositions(base_id, (job,), sites, reason, "skipped")
                continue
            recipe = dict(job["recipe"])
            recipe["checkpoints"] = tuple(recipe["checkpoints"])[:remaining_evaluations]
            recipe["steps"] = max(recipe["checkpoints"])
            bounded_jobs.append({**job, "recipe": recipe})
            remaining_evaluations -= len(recipe["checkpoints"])
        jobs = bounded_jobs
        if self.external_gpu_runtime is not None:
            parameter_count = sum(int(tensor.numel()) for tensor in self.model.tensors.values())
            scheduler = getattr(self, "gpu_scheduler", None)
            if scheduler is None:
                scheduler = GpuResourceScheduler()
                self.gpu_scheduler = scheduler
            if self.external_gpu_runtime.free_memory_bytes > 0:
                scheduler.vram_capacity_bytes_per_device = min(
                    scheduler.vram_capacity_bytes_per_device,
                    self.external_gpu_runtime.free_memory_bytes,
                )
            cycle_id = getattr(self, "cycle_id", self.archive.cycle_id)
            for kind in ("lora", "sft"):
                kind_jobs = [job for job in jobs if job["kind"] == kind]
                remaining = self._training_seconds_remaining()
                if remaining <= 0:
                    self._archive_training_dispositions(
                        base_id, kind_jobs, sites, "search_wall_budget_exhausted_before_training", "skipped"
                    )
                    continue
                multiplier = 6 if kind == "lora" else 20
                requested_bytes = parameter_count * multiplier + 512 * 1024 * 1024
                if requested_bytes > scheduler.vram_capacity_bytes_per_device:
                    self._archive_training_dispositions(
                        base_id, kind_jobs, sites, "estimated_training_memory_exceeds_available_vram", "skipped"
                    )
                    continue
                timeout_seconds = min(3600.0, remaining)
                try:
                    lease = scheduler.acquire_lease(
                        holder=f"goal-search:{cycle_id}:{kind}",
                        device_index=int(self.external_gpu_runtime.device.split(":")[1]),
                        reserved_vram_bytes=requested_bytes,
                        ttl_ms=max(1_000, int(timeout_seconds * 1000) + 5_000),
                    )
                except GpuDispatchError as exc:
                    self._archive_training_dispositions(
                        base_id, kind_jobs, sites, f"gpu_lease_unavailable:{type(exc).__name__}", "skipped"
                    )
                    continue
                self.gpu_training_lease = {
                    "lease_id": lease.lease_id,
                    "device_index": lease.device_index,
                    "reserved_vram_bytes": lease.reserved_vram_bytes,
                    "expires_ns": lease.expires_ns,
                    "scope": "process_local_cooperative_not_native_authority",
                }
                try:
                    checkpoints = train_restricted_external(
                        self.model,
                        kind_jobs,
                        self.external_gpu_runtime,
                        timeout_seconds=timeout_seconds,
                    )
                except (RuntimeError, subprocess.TimeoutExpired) as exc:
                    self._archive_training_dispositions(
                        base_id, kind_jobs, sites, f"gpu_training_failed:{type(exc).__name__}", "failed"
                    )
                    continue
                finally:
                    scheduler.release_lease(lease.lease_id)
                expected = sum(len(tuple(job["recipe"]["checkpoints"])) for job in kind_jobs)
                if len(checkpoints) != expected:
                    self._archive_training_dispositions(
                        base_id, kind_jobs, sites, "gpu_worker_returned_incomplete_checkpoint_set", "failed"
                    )
                    continue
                jobs_by_id = {str(job["job_id"]): job for job in kind_jobs}
                for checkpoint in checkpoints:
                    job = jobs_by_id.get(str(checkpoint["job_id"]))
                    if job is None:
                        raise RuntimeError(f"GPU worker returned unknown job {checkpoint['job_id']!r}")
                    if checkpoint["kind"] != job["kind"] or checkpoint["recipe_index"] != job["recipe_index"]:
                        raise RuntimeError("GPU worker checkpoint metadata disagrees with its job")
                    for field in ("gym_id", "gym_digest", "dataset_digest"):
                        if checkpoint.get(field) != job[field]:
                            raise RuntimeError(f"GPU worker checkpoint {field} disagrees with its job")
                    self._archive_training_checkpoint(
                        base_id=base_id,
                        base_target=base_target,
                        sites=sites,
                        kind=str(job["kind"]),
                        recipe_index=int(job["recipe_index"]),
                        recipe=job["recipe"],
                        step=int(checkpoint["step"]),
                        weights=dict(checkpoint["weights"]),
                        wall=float(checkpoint["wall_seconds"]),
                    )
            return
        for index, local_job in enumerate(jobs):
            if self._training_seconds_remaining() <= 0:
                self._archive_training_dispositions(
                    base_id, jobs[index:], sites, "search_wall_budget_exhausted_before_training", "skipped"
                )
                break

            def record_checkpoint(
                step: int,
                weights: dict[str, torch.Tensor],
                wall: float,
                *,
                selected_job: Mapping[str, Any] = local_job,
            ) -> None:
                self._archive_training_checkpoint(
                    base_id=base_id,
                    base_target=base_target,
                    sites=sites,
                    kind=str(selected_job["kind"]),
                    recipe_index=int(selected_job["recipe_index"]),
                    recipe=selected_job["recipe"],
                    step=step,
                    weights=weights,
                    wall=wall,
                )

            try:
                train_restricted(
                    self.model,
                    sites,
                    local_job["recipe"],
                    local_job["dev"],
                    local_job["guard"],
                    record_checkpoint,
                    local_job["seed"],
                    training_device=self.training_compute.device,
                    deadline_monotonic=self._training_deadline_monotonic,
                )
            except TrainingBudgetExpired:
                self._archive_training_dispositions(
                    base_id, jobs[index:index + 1], sites, "search_wall_budget_expired_during_training", "failed"
                )
                self._archive_training_dispositions(
                    base_id, jobs[index + 1:], sites, "search_wall_budget_exhausted_before_training", "skipped"
                )
                break

    def _training_seconds_remaining(self) -> float:
        deadline = getattr(self, "_training_deadline_monotonic", None)
        return math.inf if deadline is None else max(0.0, deadline - time.perf_counter())

    def _reserve_candidate_evaluation(self) -> bool:
        budgets = getattr(self, "budgets", None)
        if self._training_seconds_remaining() <= 0:
            return False
        count = getattr(self, "_evaluations_used", 0)
        if budgets is not None and count >= budgets.max_candidate_evaluations:
            return False
        self._evaluations_used = count + 1
        return True

    def _archive_training_dispositions(
        self,
        base_id: str,
        jobs: Sequence[Mapping[str, Any]],
        sites: Sequence[str],
        reason: str,
        state: str,
    ) -> None:
        assert self.archive is not None
        for job in jobs:
            proposal = {
                "kind": job["kind"],
                "recipe": dict(job["recipe"]),
                "recipe_schema_version": 1,
                "dataset": {
                    "schema_version": self.lab.task_data.schema_version,
                    "gym_id": self.lab.task_data.gym_id,
                    "gym_digest": self.lab.task_data.gym_digest,
                    "dataset_digest": self.lab.task_data.dataset_digest,
                    "train_digest": self.lab.task_data.train_digest,
                    "validation_digest": self.lab.task_data.validation_digest,
                },
                "sites": tuple(sites),
                "status": state,
                "reason": reason,
            }
            proposal_digest = str(content_digest(canonicalize(proposal)))
            self.archive.add(CandidateRecord(
                candidate_id=f"{job['kind']}-{job['recipe_index']}:disposition:{proposal_digest.removeprefix('sha256:')[:16]}",
                parent_candidate_id=base_id,
                round_number=0,
                arm="baseline",
                method=str(job["kind"]),
                proposal_digest=proposal_digest,
                state_digest=None,
                scores={},
                proposal=proposal,
                search_state=state,  # type: ignore[arg-type]
                replay_verified=False,
                reload_parity_verified=False,
                rejection_reasons=(reason,),
                next_move_reason="training_disposition",
                artifact=None,
            ))

    def _solve(self, move: Move, dev: Capture, guard: Capture, drift_task: torch.Tensor, drift_guard: torch.Tensor, current: Any) -> tuple[torch.Tensor, torch.Tensor]:
        k = ms.contrast(dev.inputs[move.site])
        kp = guard.inputs[move.site]
        if move.mode == "task":
            y = self.task_target * move.strength - drift_task
            yp = None
        else:
            y = torch.zeros_like(self.task_target)
            yp = -drift_guard
        if "h_to_4h" in move.site:
            # The step reaches the residual through GELU and the down-projection. The mean GELU slope is taken as 1/2.
            down = current.tensors[move.site.replace("dense_h_to_4h", "dense_4h_to_h")]
            to_hidden = 2.0 * torch.linalg.pinv(down)
            y = to_hidden @ y
            yp = to_hidden @ yp if yp is not None else None
        delta = ms.ridge_map(k, y, kp, yp, move.alpha, move.ridge)
        return ms.low_rank(delta, RANK_LIMIT)

    def measure(self, tensors: Mapping[str, torch.Tensor], items: Sequence[tuple[str, str]]) -> float:
        return sum(self.mlab.task_hits(items, tensors)) / len(items)

    def accept_score(self, tensors: Mapping[str, torch.Tensor]) -> dict[str, float]:
        task = self.measure(tensors, [*self.lab.dev_items, *self.lab.dev_aug_items])
        detail = self.mlab.retention_detail(self.accept_guard, tensors)
        return {
            "task": task,
            "retention": detail["retention"],
            "kl": detail["mean_kl"],
            "changed": self.changed(tensors),
            "J": objective(task, detail["retention"], detail["mean_kl"], self.changed(tensors), self.cost),
        }

    def monitor_score(self, tensors: Mapping[str, torch.Tensor]) -> dict[str, float] | None:
        if not self.monitor_guard:
            return None
        detail = self.mlab.retention_detail(self.monitor_guard, tensors)
        return {"retention": detail["retention"], "kl": detail["mean_kl"]}

    def report(self, tensors: Mapping[str, torch.Tensor]) -> dict[str, float]:
        """Splits that never decide a step."""
        tune = self.mlab.retention_detail(self.lab.tune, tensors)
        practice = self.mlab.retention_detail(self.practice_prompts, tensors)
        return {
            "dev": self.measure(tensors, self.lab.dev_items),
            "dev_aug": self.measure(tensors, self.lab.dev_aug_items),
            "val": self.measure(tensors, self.lab.val_items),
            "report_prefixes": self.measure(tensors, self.report_items),
            "tune": tune["retention"],
            "tune_kl": tune["mean_kl"],
            "practice": practice["retention"],
        }

    def run(self, log_path: Path) -> dict[str, Any]:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_parent = log_path.parent.resolve()
        artifact_suffix = hashlib.sha256(self.cycle_id.encode("utf-8")).hexdigest()[:12]
        self.artifact_directory = self.artifact_parent / f"{log_path.stem}_{artifact_suffix}.artifacts"
        start = time.time()
        start_monotonic = time.perf_counter()
        self._evaluations_used = 0
        self._training_deadline_monotonic = (
            start_monotonic + self.budgets.max_wall_seconds
            if self.budgets.max_wall_seconds is not None
            else None
        )
        state = self.accept_score({})
        base_target = float(state["task"])
        reported = self.report({})
        initial_monitor = self.monitor_score({})
        monitored = initial_monitor
        parent_id = "base"
        if self.archive is not None:
            state_digest = tensor_state_digest(self.model.tensors)
            parent_id = f"base:{state_digest.removeprefix('sha256:')[:16]}"
            self.archive.add(
                CandidateRecord(
                    candidate_id=parent_id,
                    parent_candidate_id=None,
                    round_number=0,
                    arm="baseline",
                    method="no_op",
                    proposal_digest=state_digest,
                    state_digest=state_digest,
                    scores=self._archive_scores(state, base_target, 0.0, {}, None),
                    proposal={"kind": "no_op"},
                    search_state="accepted",
                    replay_verified=True,
                    reload_parity_verified=True,
                    next_move_reason=None,
                )
            )
        rounds: list[dict[str, Any]] = []
        refused = 0
        evaluations = 0
        no_gain_rounds = 0
        last_best_gain: float | None = None
        stop = "safety_rounds_bound"
        max_rounds = min(SAFETY_ROUNDS, self.budgets.max_rounds)
        for number in range(1, max_rounds + 1):
            if self._training_seconds_remaining() <= 0 or self._evaluations_used >= self.budgets.max_candidate_evaluations:
                stop = "budget_exhausted"
                break
            if self.acceptance_profile is None:
                decision = NextMoveDecision("legacy_search", "Legacy cumulative search", "balanced")
            else:
                if monitored is None or initial_monitor is None:
                    raise ValueError("Goal-driven search requires a monitor split")
                decision = diagnose_next_move(
                    self.acceptance_profile,
                    self.budgets,
                    SearchObservation(
                        round_number=number - 1,
                        candidate_evaluations=evaluations,
                        changed_parameters=int(state["changed"]),
                        elapsed_seconds=time.time() - start,
                        target_accuracy=state["task"],
                        retention_accuracy=state["retention"],
                        kl_nats=state["kl"],
                        monitor_retention_accuracy=monitored["retention"],
                        monitor_kl_nats=monitored["kl"],
                        initial_monitor_retention_accuracy=initial_monitor["retention"],
                        initial_monitor_kl_nats=initial_monitor["kl"],
                        best_gain=last_best_gain,
                        no_gain_rounds=no_gain_rounds,
                    ),
                )
                if decision.terminal_state is not None:
                    stop = decision.terminal_state
                    rounds.append(
                        {
                            "round": number,
                            "stopped": True,
                            "best_gain": last_best_gain,
                            "candidates": 0,
                            "decision": decision.to_document(),
                        }
                    )
                    break

            current = self.current()
            dev = Capture(current, self.dev_prompts)
            guard = Capture(current, self.solve_guard)
            drift_task = ms.contrast(dev.residual - self.base_dev.residual)
            drift_guard = guard.residual - self.base_solve.residual
            best: tuple[float, EvaluatedMove] | None = None
            trials: list[EvaluatedMove] = []
            moves = moves_for_policy(round_moves(self.model, number == 1), decision)
            remaining = self.budgets.max_candidate_evaluations - self._evaluations_used
            if self.acceptance_profile is not None and remaining <= 0:
                stop = "budget_exhausted"
                break
            if self.acceptance_profile is not None:
                moves = moves[:remaining]
            for move in moves:
                if self._training_seconds_remaining() <= 0 or self._evaluations_used >= self.budgets.max_candidate_evaluations:
                    stop = "budget_exhausted"
                    break
                try:
                    left, right = self._solve(move, dev, guard, drift_task, drift_guard, current)
                except ValueError:  # The solve had nothing to do, which is an empty candidate and not an error to report.
                    refused += 1
                    continue
                if self._training_seconds_remaining() <= 0:
                    stop = "budget_exhausted"
                    break
                trial_plan = self.plan.branch()
                trial_plan.low_rank(move.site, left.contiguous(), right.contiguous())
                trial_candidate = trial_plan.preview()
                trial = {
                    name: tensor
                    for name, tensor in trial_candidate.edited_tensors.items()
                    if not torch.equal(tensor, self.model.tensors[name])
                }
                if not self._reserve_candidate_evaluation():
                    stop = "budget_exhausted"
                    break
                score = self.accept_score(trial)
                candidate_id, proposal_digest = self._archive_id(number, parent_id, move, left, right)
                rejection_reason = None
                changed_limit = self.budgets.max_changed_parameters
                if changed_limit is not None and int(score["changed"]) > changed_limit:
                    rejection_reason = "changed_parameter_budget_exceeded"
                evaluated = EvaluatedMove(
                    candidate_id,
                    proposal_digest,
                    move,
                    left,
                    right,
                    trial,
                    score,
                    rejection_reason,
                )
                trials.append(evaluated)
                self.search_sites.add(move.site)
                evaluations = self._evaluations_used
                if rejection_reason is None and (best is None or score["J"] > best[0]):
                    best = (score["J"], evaluated)
            gain = None if best is None else best[0] - state["J"]
            last_best_gain = gain
            if best is None or gain is None or gain < self.budgets.min_gain:
                self._archive_round(
                    number,
                    parent_id,
                    base_target,
                    trials,
                    None,
                    None,
                    decision,
                    time.time() - start,
                )
                no_gain_rounds += 1
                row = {
                    "round": number,
                    "stopped": self.acceptance_profile is None,
                    "best_gain": gain,
                    "candidates": len(moves),
                    "decision": decision.to_document(),
                }
                rounds.append(row)
                if self.acceptance_profile is None:
                    stop = "no_candidate_cleared_min_gain"
                    break
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                continue

            no_gain_rounds = 0
            _, evaluated = best
            move, left, right, score = evaluated.move, evaluated.left, evaluated.right, evaluated.score
            self.plan.low_rank(move.site, left.contiguous(), right.contiguous())
            candidate = self.plan.preview()  # Rust executes the whole sequence from the base model.
            self.tensors = dict(candidate.tensors if hasattr(candidate, "tensors") else candidate.model.tensors)
            self.tensors = {k: v for k, v in self.tensors.items() if not torch.equal(v, self.model.tensors[k])}
            if self.tensors.keys() != evaluated.tensors.keys() or any(
                not torch.equal(self.tensors[name], evaluated.tensors[name]) for name in evaluated.tensors
            ):
                raise RuntimeError("The replayed edit graph differs from the scored candidate tensors")
            accepted_id = self._archive_round(
                number,
                parent_id,
                base_target,
                trials,
                evaluated,
                candidate.state_digest,
                decision,
                time.time() - start,
            )
            if accepted_id is not None:
                parent_id = accepted_id
            if self._training_seconds_remaining() <= 0:
                state, reported = score, None
                stop = "budget_exhausted"
                break
            after = self.report(self.tensors)
            monitor_after = self.monitor_score(self.tensors)
            row = {
                "round": number,
                "site": move.site,
                "mode": move.mode,
                "strength": move.strength,
                "alpha": move.alpha,
                "ridge": move.ridge,
                "candidates": len(moves),
                "J_gain_accept": score["J"] - state["J"],
                "accept": score,
                "report": after,
                "monitor": monitor_after,
                "marginal_report_task_val": after["val"] - reported["val"],
                "marginal_report_task_prefixes": after["report_prefixes"] - reported["report_prefixes"],
                "marginal_report_retention_tune": after["tune"] - reported["tune"],
                "marginal_report_kl_tune": after["tune_kl"] - reported["tune_kl"],
                "decision": decision.to_document(),
                "cumulative_changed": score["changed"],
                "tensors_touched": sorted(self.tensors),
                "plan_digest": candidate.plan_digest,
                "state_digest": candidate.state_digest,
                "elapsed_seconds_rough": time.time() - start,
            }
            rounds.append(row)
            state, reported = score, after
            if monitor_after is not None:
                monitored = monitor_after
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({k: row[k] for k in ("round", "site", "mode", "strength", "J_gain_accept", "cumulative_changed", "report")}), flush=True)
        if self.archive is not None and self.include_training_baselines:
            base_id = next(record.candidate_id for record in self.archive.records if record.method == "no_op")
            self._archive_training_competitors(base_id, base_target)
        frontier = self.archive.freeze() if self.archive is not None else None
        self.frontier_snapshot = frontier
        terminal_state = self.archive.terminal_state() if self.archive is not None else None
        return {
            "cost_per_parameter": self.cost,
            "gym_id": self.gym.gym_id,
            "gym_digest": self.gym_digest,
            "task_data": {
                "schema_version": self.lab.task_data.schema_version,
                "dataset_digest": self.lab.task_data.dataset_digest,
                "train_digest": self.lab.task_data.train_digest,
                "validation_digest": self.lab.task_data.validation_digest,
            },
            "kl_weight": KL_WEIGHT,
            "min_gain": self.budgets.min_gain,
            "stop_reason": stop,
            "safety_rounds": max_rounds,
            "safety_limit_bound": stop == "safety_rounds_bound",
            "excluded_sites": excluded_sites(self.model),
            "empty_candidates": refused,
            "candidate_evaluations": self._evaluations_used,
            "acceptance_profile": self.acceptance_profile.to_document() if self.acceptance_profile else None,
            "acceptance_profile_digest": self.acceptance_profile.digest if self.acceptance_profile else None,
            "training_runtime": self.training_runtime_document(),
            "archive_terminal_state": terminal_state,
            "frontier_snapshot": frontier.to_document() if frontier is not None else None,
            "frontier_artifact_directory": (
                self.artifact_directory.relative_to(self.artifact_parent).as_posix()
                if self.artifact_directory is not None and self.artifact_parent is not None
                else None
            ),
            "monitor_initial": initial_monitor,
            "monitor_final": monitored,
            "search_splits": {
                "solve": len(self.solve_guard),
                "accept": len(self.accept_guard),
                "monitor": len(self.monitor_guard),
            },
            "rounds": rounds,
            "final_report": reported,
            "final_accept": state,
            "total_changed": state["changed"],
            "total_parameters": self.model.total_parameters,
            "wall_seconds_rough": time.time() - start,
        }


def graph_path(root: Path, cost: float) -> Path:
    return root / f"cumulative_graph_c{cost:g}.json"


def run_search(model: Any, cost: float, root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    log = root / f"cumulative_rounds_c{cost:g}.jsonl"
    log.unlink(missing_ok=True)
    search = CumulativeSearch(model, cost)
    result = search.run(log)
    if search.plan._steps:
        graph = search.plan.to_graph()
        graph_path(root, cost).write_text(json.dumps(graph), encoding="utf-8")
        replayed = apply_edit_graph(model, graph)
        result["replay"] = {
            "plan_digest": replayed.plan_digest,
            "state_digest": replayed.state_digest,
            "matches": replayed.plan_digest == graph["plan_digest"] and replayed.state_digest == graph["candidate_state_digest"],
            "steps": len(graph["steps"]),
        }
    else:
        result["replay"] = None
    (root / f"cumulative_result_c{cost:g}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def independent_target_cases(
    model: Any, items: Sequence[tuple[str, str]]
) -> tuple[tuple[str, str], ...]:
    """Convert internal vocabulary labels to decoded text with an exact one-token round trip."""
    cases: list[tuple[str, str]] = []
    for prompt, target in items:
        try:
            token_id = int(model.tokenizer.vocab[target])
        except KeyError as exc:
            raise ValueError(f"Target token {target!r} is absent from the model vocabulary") from exc
        decoded = model.tokenizer.decode_token(token_id)
        if model.tokenizer.encode(decoded) != [token_id]:
            raise ValueError(
                f"Target token {target!r} does not decode to exactly one round-trippable token"
            )
        cases.append((prompt, decoded))
    return tuple(cases)


def run_goal_search(
    model: Any,
    root: str | Path,
    *,
    gym: Any = GYM,
    cost: float = PRIMARY_COST,
    acceptance_profile: AcceptanceProfile = STRICT_GENERAL_PROFILE,
    budgets: SearchBudgets | None = None,
    cycle_id: str | None = None,
    include_training_baselines: bool = True,
    training_device: str = "auto",
    gpu_python: str | Path | None = None,
    gpu_scheduler: GpuResourceScheduler | None = None,
    evaluate_independently: bool = True,
    isolate_independent_evaluation: bool = True,
    evaluator: Any = None,
    eval_config: Any = None,
    evaluation_timeout_seconds: float = 3600.0,
) -> dict[str, Any]:
    """Run, freeze, replay, and independently evaluate one goal-driven search cycle."""
    if isinstance(gym, str):
        from .gym_data import resolve_gym

        gym = resolve_gym(gym)
    if not isinstance(gym, ReusableTaskGym):
        raise TypeError("goal-search gym must be a ReusableTaskGym or its registered key")
    output = Path(root)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(
            f"Goal-search output directory is not empty: {output}; start a new run directory"
        )
    log = output / f"goal_search_rounds_c{cost:g}.jsonl"
    search = CumulativeSearch(
        model,
        cost,
        acceptance_profile=acceptance_profile,
        budgets=budgets,
        cycle_id=cycle_id,
        gym=gym,
        include_training_baselines=include_training_baselines,
        training_device=training_device,
        gpu_python=gpu_python,
        gpu_scheduler=gpu_scheduler,
    )
    result = search.run(log)
    if search.plan._steps:
        graph = search.plan.to_graph()
        graph_file = output / f"goal_search_graph_c{cost:g}.json"
        graph_file.write_text(json.dumps(graph), encoding="utf-8")
        replayed = apply_edit_graph(model, graph)
        result["replay"] = {
            "plan_digest": replayed.plan_digest,
            "state_digest": replayed.state_digest,
            "matches": (
                replayed.plan_digest == graph["plan_digest"]
                and replayed.state_digest == graph["candidate_state_digest"]
            ),
            "steps": len(graph["steps"]),
            "graph_path": graph_file.relative_to(output).as_posix(),
        }
    else:
        result["replay"] = None
    snapshot = search.frontier_snapshot
    if snapshot is None:
        raise RuntimeError("Goal search completed without freezing a frontier")
    if evaluate_independently:
        from .frontier_evaluation import (
            evaluate_frozen_frontier,
            evaluate_frozen_frontier_isolated,
        )

        try:
            from .gym_data import materialize_holdout_items

            holdout_items = materialize_holdout_items(gym, tokenizer=model.tokenizer)
            if not holdout_items:
                raise ValueError("Gym has no separate holdout rules")
            target_cases = independent_target_cases(model, holdout_items)
        except ValueError as exc:
            result["independent_evaluation"] = {
                "schema_version": 1,
                "kind": "nf_frontier_evaluation_v1",
                "execution_mode": "not_started",
                "snapshot_digest": snapshot.snapshot_digest,
                "complete": False,
                "independent_terminal_state": "qualification_incomplete",
                "not_evaluated": [{"candidate_id": "*", "reason": f"target_panel_invalid: {exc}"}],
                "promoted_candidate_ids": [],
                "rejected_candidate_ids": [],
            }
        else:
            evaluation_options = {
                "artifact_root": output,
                "target_cases": target_cases,
                "eval_config": eval_config,
            }
            if isolate_independent_evaluation and evaluator is None:
                result["independent_evaluation"] = evaluate_frozen_frontier_isolated(
                    model,
                    snapshot,
                    timeout_seconds=evaluation_timeout_seconds,
                    **evaluation_options,
                )
            else:
                result["independent_evaluation"] = evaluate_frozen_frontier(
                    model,
                    snapshot,
                    evaluator=evaluator,
                    **evaluation_options,
                )
    else:
        result["independent_evaluation"] = None
    result["authority"] = {
        "scope": "local_python_search_preview",
        "native_family_ledger_integrated": False,
        "gpu_reservation_scope": (
            "process_local_cooperative_not_native_authority"
            if search.gpu_training_lease is not None
            else "not_reserved"
        ),
        "sealed_target_custody_verified": False,
        "promotion_authorized": False,
    }
    result_path = output / f"goal_search_result_c{cost:g}.json"
    result["result_path"] = str(result_path)
    result_path.write_text(json.dumps(result, indent=1, allow_nan=False), encoding="utf-8")
    return result


def replay_graph(model: Any, path: Path) -> dict[str, torch.Tensor]:
    candidate = apply_edit_graph(model, json.loads(path.read_text(encoding="utf-8")))
    return {k: v for k, v in candidate.model.tensors.items() if not torch.equal(v, model.tensors[k])}


def restricted_grid(kind: str) -> list[dict[str, Any]]:
    """The published learning rate and length for each kind, with both guard weights tried on the fit corpus."""
    if kind == "lora":
        return [{"recipe_schema_version": 1, "kind": kind, "lr": 3e-4, "steps": 300, "weight": w, "rank": 4, "checkpoints": (100, 200, 300)} for w in (1.0, 10.0)]
    return [{"recipe_schema_version": 1, "kind": kind, "lr": 3e-5, "steps": 150, "weight": w, "rank": 0, "checkpoints": (50, 100, 150)} for w in (1.0, 10.0)]


def _validate_training_recipe(recipe: Mapping[str, Any], seed: Any) -> None:
    from .gym_job import is_finite_number

    if recipe.get("kind") not in {"lora", "sft"}:
        raise ValueError("unsupported training method")
    for name in ("steps", "rank"):
        if type(recipe.get(name)) is not int or recipe[name] < (1 if name == "steps" else 0):
            raise ValueError(f"invalid integral training {name}")
    if recipe["kind"] == "lora" and recipe["rank"] < 1:
        raise ValueError("LoRA rank must be positive")
    if not is_finite_number(recipe.get("lr")) or recipe["lr"] <= 0 or not is_finite_number(recipe.get("weight")) or recipe["weight"] < 0:
        raise ValueError("training rates and weights must be finite and in range")
    if type(seed) is not int or seed < 0:
        raise ValueError("training seed must be a nonnegative integer")
    checkpoints = recipe.get("checkpoints")
    if not isinstance(checkpoints, (list, tuple)) or not checkpoints or any(type(step) is not int or not 0 < step <= recipe["steps"] for step in checkpoints):
        raise ValueError("training checkpoints must be positive integral steps within the recipe")
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("training checkpoint steps must be unique")


def train_restricted_external(
    model: Any,
    jobs: Sequence[Mapping[str, Any]],
    runtime: ExternalGpuRuntime,
    *,
    timeout_seconds: float = 3600.0,
    training_precision: str = "float32",
) -> tuple[dict[str, Any], ...]:
    """Run one trusted batch of gradient jobs off-process on a preflighted GPU runtime."""
    from .gym_job import is_finite_number

    if not is_finite_number(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("GPU worker timeout must be finite and positive")
    if not isinstance(training_precision, str) or training_precision not in {"float32", "float64", "bfloat16"}:
        raise ValueError("unsupported training precision")
    for job in jobs:
        _validate_training_recipe(job["recipe"], job["seed"])
    tokenizer = model.tokenizer
    merges = [
        " ".join(pair)
        for pair, _ in sorted(tokenizer.bpe_ranks.items(), key=lambda entry: entry[1])
    ]
    checkpoint_identity = getattr(model, "checkpoint_identity", None)
    payload = {
        "schema_version": 1,
        "kind": "restricted_training_jobs",
        "model_type": type(model).__name__,
        "model_name": model.model_name,
        "config": model.config,
        "tokenizer": {
            "vocab": tokenizer.vocab,
            "merges": merges,
            "byte_level": tokenizer.byte_level,
            "added_tokens": tokenizer.added_tokens,
        },
        "jobs": [dict(job) for job in jobs],
        "runtime": {"device": runtime.device, "precision": training_precision},
    }
    input_state_digest = tensor_state_digest(model.tensors)
    from .provenance import inference_identity_digest

    payload["input_state_digest"] = input_state_digest
    payload["inference_identity_digest"] = inference_identity_digest(model)
    if checkpoint_identity is not None and getattr(model, "_checkpoint_state_digest", None) == input_state_digest:
        checkpoint_path = Path(checkpoint_identity.path).resolve()
        if not checkpoint_path.is_dir():
            raise RuntimeError("Pinned GPU training checkpoint directory is unavailable")
        payload["checkpoint_identity"] = {
            "path": str(checkpoint_path),
            "model_id": str(checkpoint_identity.model_id),
            "revision": checkpoint_identity.revision,
            "source": str(checkpoint_identity.source),
            "file_digests": dict(checkpoint_identity.file_digests),
        }
    else:
        payload["tensors"] = {
            name: tensor.detach().to(device="cpu", dtype=torch.float64).clone()
            for name, tensor in model.tensors.items()
        }
    source_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="nf-gpu-training-") as directory:
        root = Path(directory)
        input_path = root / "input.pt"
        output_path = root / "output.pt"
        torch.save(payload, input_path)
        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        environment = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                item for item in (str(source_root), existing_pythonpath) if item
            ),
            "PYTHONNOUSERSITE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
        }
        completed = subprocess.run(
            [
                runtime.python_executable,
                "-m",
                "neural_foundry.gpu_training_worker",
                str(input_path),
                str(output_path),
            ],
            cwd=source_root.parent,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
        if completed.returncode != 0 or not output_path.is_file():
            detail = (completed.stderr or completed.stdout).strip()[-2000:]
            raise RuntimeError(
                f"GPU training worker exited {completed.returncode} without a result: {detail}"
            )
        result = torch.load(output_path, map_location="cpu", weights_only=True)
    if result.get("schema_version") != 1 or result.get("kind") != "restricted_training_checkpoints":
        raise RuntimeError("GPU training worker returned an invalid checkpoint envelope")
    if result.get("input_state_digest") != input_state_digest:
        raise RuntimeError("GPU training worker used a different input tensor state")
    if result.get("inference_identity_digest") != payload["inference_identity_digest"]:
        raise RuntimeError("GPU training worker used a different config or tokenizer")
    reported_runtime = result.get("runtime", {})
    if reported_runtime.get("torch") != runtime.torch_version or reported_runtime.get("device") != runtime.device or reported_runtime.get("precision") != training_precision:
        raise RuntimeError("GPU training worker runtime changed after preflight")
    checkpoints = tuple(dict(checkpoint) for checkpoint in result["checkpoints"])
    for checkpoint in checkpoints:
        checkpoint["worker_runtime"] = dict(reported_runtime)
        checkpoint["input_state_digest"] = input_state_digest
    if any(
        tensor.device.type != "cpu" or tensor.dtype != torch.float64
        for checkpoint in checkpoints
        for tensor in dict(checkpoint["weights"]).values()
    ):
        raise RuntimeError("GPU training worker did not return canonical CPU float64 weights")
    if any(name not in model.tensors or tuple(tensor.shape) != tuple(model.tensors[name].shape) or not bool(torch.isfinite(tensor).all())
           for checkpoint in checkpoints for name, tensor in dict(checkpoint["weights"]).items()):
        raise RuntimeError("GPU worker returned nonfinite or mismatched tensor state")
    return checkpoints


def train_restricted(
    model: Any,
    sites: Sequence[str],
    recipe: Mapping[str, Any],
    dev: Sequence[tuple[str, str]],
    guard: Sequence[str],
    on_checkpoint: Any = None,
    seed: int = 0,
    training_device: str | torch.device = "cpu",
    training_dtype: torch.dtype | None = None,
    deadline_monotonic: float | None = None,
) -> tuple[dict[str, torch.Tensor], float, int]:
    """Train on a declared device and return CPU float64 tensors for canonical scoring."""
    _validate_training_recipe(recipe, seed)
    device = torch.device(training_device)
    if device.type == "cuda":
        from .gpu import probe_gpu_hardware

        inventory = probe_gpu_hardware()
        if not inventory.cuda_executable:
            raise RuntimeError(f"CUDA training requested but unavailable: {inventory.kernel_probe_error}")
    dtype = training_dtype or (torch.float32 if device.type == "cuda" else torch.float64)
    training_model = copy.copy(model)
    training_model.tensors = {
        name: tensor.detach().to(device=device, dtype=dtype)
        for name, tensor in model.tensors.items()
    }
    generator = torch.Generator(device=device).manual_seed(seed)
    base = training_model.tensors
    guard = tuple(guard)
    with torch.no_grad():
        reference = {
            prompt: F.log_softmax(bl._last_logits(training_model, prompt), dim=-1)
            for prompt in guard
        }
    if recipe["kind"] == "sft":
        live = {site: base[site].clone().requires_grad_(True) for site in sites}
        params = list(live.values())

        def state() -> dict[str, torch.Tensor]:
            return live
    else:
        factors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for site in sites:
            out_dim, in_dim = base[site].shape
            rank = min(int(recipe["rank"]), int(out_dim), int(in_dim))
            left = torch.randn(
                rank, int(in_dim), dtype=dtype, device=device, generator=generator
            ) / int(in_dim) ** 0.5
            right = torch.zeros(int(out_dim), rank, dtype=dtype, device=device)
            factors[site] = (left.requires_grad_(True), right.requires_grad_(True))
        params = [tensor for pair in factors.values() for tensor in pair]

        def state() -> dict[str, torch.Tensor]:
            return {
                site: base[site] + right @ left
                for site, (left, right) in factors.items()
            }

    optimizer = torch.optim.Adam(params, lr=recipe["lr"])
    trained = bl.merged_model(training_model, {})
    started = time.perf_counter()
    excluded = 0.0

    def canonical_snapshot() -> dict[str, torch.Tensor]:
        with torch.no_grad():
            return bl._training_update_snapshot(model.tensors, base, state())

    for step in range(1, int(recipe["steps"]) + 1):
        if deadline_monotonic is not None and time.perf_counter() >= deadline_monotonic:
            raise TrainingBudgetExpired("The frozen training budget expired before the next optimizer step")
        trained.tensors = {**base, **state()}
        loss = bl._mean_ce(trained, dev)
        picks = torch.randperm(len(guard), generator=generator, device=device)[:8]
        kl = torch.stack(
            [
                F.kl_div(
                    F.log_softmax(bl._last_logits(trained, guard[int(index)]), dim=-1),
                    reference[guard[int(index)]],
                    log_target=True,
                    reduction="sum",
                )
                for index in picks
            ]
        ).mean()
        total = loss + recipe["weight"] * kl
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        if on_checkpoint is not None and step in recipe["checkpoints"]:
            paused = time.perf_counter()
            on_checkpoint(step, canonical_snapshot(), paused - started - excluded)
            excluded += time.perf_counter() - paused
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (
        canonical_snapshot(),
        time.perf_counter() - started - excluded,
        sum(int(parameter.numel()) for parameter in params),
    )


def run_restricted(
    model: PythiaModel,
    cost: float,
    root: Path,
    weights_dir: Path,
    *,
    training_device: str = "auto",
    gpu_python: str | Path | None = None,
) -> None:
    """Select public LoRA/SFT recipes, using a verified GPU worker when available."""
    result = json.loads((root / f"cumulative_result_c{cost:g}.json").read_text(encoding="utf-8"))
    if not result["replay"]:
        print("The search accepted nothing, so there are no tensors to restrict to", flush=True)
        (root / f"restricted_c{cost:g}.json").write_text(
            json.dumps({"skipped": "search accepted no step"}), encoding="utf-8"
        )
        return
    sites = sorted({r["site"] for r in result["rounds"] if "site" in r})
    bar = result["final_report"]["dev"]
    mlab = ms.MultiLab(model)
    placement = select_training_compute(training_device)
    external_runtime = (
        discover_external_gpu_runtime(gpu_python)
        if training_device == "auto" and placement.device == "cpu"
        else None
    )
    if external_runtime is not None:
        runtime_document = {
            "mode": "external_gpu_worker",
            "device": "cuda:0",
            "dtype": "f32",
            "python_executable": external_runtime.python_executable,
            "torch_version": external_runtime.torch_version,
            "device_name": external_runtime.device_name,
        }
    else:
        runtime_document = {
            "mode": "in_process",
            "device": placement.device,
            "dtype": placement.dtype,
            "reason": placement.reason,
        }
    jobs: list[dict[str, Any]] = []
    for kind in ("lora", "sft"):
        for recipe_index, recipe in enumerate(restricted_grid(kind)):
            jobs.append(
                {
                    "job_id": f"{kind}-{recipe_index}",
                    "kind": kind,
                    "recipe_index": recipe_index,
                    "sites": tuple(sites),
                    "recipe": dict(recipe),
                    "dev": tuple(mlab.lab.dev_items),
                    "guard": tuple(mlab.lab.fit),
                    "seed": 0,
                }
            )
    weights_dir.mkdir(parents=True, exist_ok=True)
    doc: dict[str, Any] = {
        "sites": sites,
        "development_bar": bar,
        "training_runtime": runtime_document,
        "trials": [],
        "selected": {},
    }
    best: dict[str, tuple[tuple[float, ...], dict[str, Any]]] = {}
    gpu_capacity = (
        external_runtime.free_memory_bytes
        if external_runtime is not None and external_runtime.free_memory_bytes > 0
        else 32 * 1024 * 1024 * 1024
    )
    gpu_scheduler = GpuResourceScheduler(
        vram_capacity_bytes_per_device=gpu_capacity
    )

    def record(job: Mapping[str, Any], step: int, weights: dict[str, torch.Tensor], wall: float) -> None:
        kind = str(job["kind"])
        recipe = job["recipe"]
        scores = mlab.score_public(weights)
        row = {
            "kind": kind,
            "steps": step,
            "weight": recipe["weight"],
            "lr": recipe["lr"],
            "rank": recipe["rank"],
            "scores": scores,
            "wall_seconds_rough": wall,
        }
        doc["trials"].append(row)
        print(row, flush=True)
        if scores["dev"] < bar - 1e-9 or "tune" not in scores:
            return
        key = ((scores["practice"] + scores["tune"]) / 2, -scores["delta_norm"])
        if kind not in best or key > best[kind][0]:
            best[kind] = (key, row)
            torch.save(weights, weights_dir / f"restricted_{kind}_c{cost:g}.pt")

    if external_runtime is not None:
        timeout_seconds = 3600.0
        reserved_bytes = sum(int(tensor.numel()) for tensor in model.tensors.values()) * 16 + 256 * 1024 * 1024
        lease = gpu_scheduler.acquire_lease(
            holder=f"restricted:{model.model_name}:{cost:g}",
            reserved_vram_bytes=reserved_bytes,
            ttl_ms=int(timeout_seconds * 1000) + 60_000,
        )
        doc["training_runtime"]["reservation_scope"] = "process_local_cooperative_not_native_authority"
        doc["training_runtime"]["reservation"] = {
            "lease_id": lease.lease_id,
            "reserved_vram_bytes": lease.reserved_vram_bytes,
            "expires_ns": lease.expires_ns,
        }
        try:
            checkpoints = train_restricted_external(
                model, jobs, external_runtime, timeout_seconds=timeout_seconds
            )
        finally:
            gpu_scheduler.release_lease(lease.lease_id)
        jobs_by_id = {str(job["job_id"]): job for job in jobs}
        for checkpoint in checkpoints:
            job = jobs_by_id[str(checkpoint["job_id"])]
            record(job, int(checkpoint["step"]), dict(checkpoint["weights"]), float(checkpoint["wall_seconds"]))
    else:
        for job in jobs:
            def record_local(
                step: int,
                weights: dict[str, torch.Tensor],
                wall: float,
                *,
                selected_job: Mapping[str, Any] = job,
            ) -> None:
                record(selected_job, step, weights, wall)

            lease = None
            if placement.device.startswith("cuda:"):
                reserved_bytes = sum(int(tensor.numel()) for tensor in model.tensors.values()) * 16 + 256 * 1024 * 1024
                lease = gpu_scheduler.acquire_lease(
                    holder=f"restricted:{model.model_name}:{cost:g}:{job['job_id']}",
                    device_index=int(placement.device.partition(":")[2]),
                    reserved_vram_bytes=reserved_bytes,
                    ttl_ms=3_600_000,
                )
            try:
                train_restricted(
                    model,
                    job["sites"],
                    job["recipe"],
                    job["dev"],
                    job["guard"],
                    record_local,
                    training_device=placement.device,
                )
            finally:
                if lease is not None:
                    gpu_scheduler.release_lease(lease.lease_id)
    for kind in ("lora", "sft"):
        selected = best.get(kind)
        if selected is None:
            doc["selected"][kind] = None
            continue
        weights = torch.load(
            weights_dir / f"restricted_{kind}_c{cost:g}.pt",
            map_location="cpu",
            weights_only=True,
        )
        doc["selected"][kind] = {
            **selected[1],
            "weights_digest": bl.weights_digest(weights),
            "changed_parameters": ms.changed_parameters(model, weights),
        }
    (root / f"restricted_c{cost:g}.json").write_text(
        json.dumps(doc, indent=1), encoding="utf-8"
    )


def freeze_cumulative(doc: dict[str, Any], rows: Sequence[Mapping[str, Any]], root: Path, floor: float) -> None:
    """Adds the search results to the freeze. The comparator is the best single tensor at the same development bar."""
    doc["cumulative"] = {}
    doc["restricted"] = {}
    comparator_of: dict[str, str] = {}
    single_rows = [r for r in rows if r["family"] == "single" and "tune" in r.get("scores", {})]
    equal: dict[str, Any] = {}
    # The primary first, so a sensitivity arm that reached the identical model is recorded once and not tested twice.
    paths = sorted(root.glob("cumulative_result_c*.json"), key=lambda p: p.stem != f"cumulative_result_c{PRIMARY_COST:g}")
    seen: dict[str, str] = {}
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        tag = path.stem.removeprefix("cumulative_result_c")
        name = f"cumulative_c{tag}"
        if not result["replay"]:
            doc["cumulative_empty"] = {**doc.get("cumulative_empty", {}), name: result["stop_reason"]}
            continue
        if result["replay"]["state_digest"] in seen:
            doc["cumulative_identical"] = {**doc.get("cumulative_identical", {}), name: seen[result["replay"]["state_digest"]]}
            continue
        seen[result["replay"]["state_digest"]] = name
        doc["cumulative"][name] = {
            "graph": str(root / f"cumulative_graph_c{tag}.json"),
            "state_digest": result["replay"]["state_digest"],
            "steps": result["replay"]["steps"],
            "public": result["final_report"],
        }
        bar = result["final_report"]["dev"]
        eligible = [r for r in single_rows if r["scores"]["dev"] >= bar - 1e-9]
        top = max(eligible, key=lambda r: ((r["scores"]["practice"] + r["scores"]["tune"]) / 2, -r["scores"]["delta_norm"]))
        single_name = f"single_at_dev_{bar:.3f}"
        equal[single_name] = {"config": top["config"], "public": top["scores"]}
        comparator_of[name] = single_name
        restricted_path = root / f"restricted_c{tag}.json"
        if restricted_path.exists():
            restricted = json.loads(restricted_path.read_text(encoding="utf-8"))
            for kind, entry in restricted.get("selected", {}).items():
                if entry:
                    label = f"restricted_{kind}_c{tag}"
                    doc["restricted"][label] = {**entry, "sites": restricted["sites"]}
                    comparator_of[label] = single_name
    doc["single_at_equal_task"] = equal
    primary = f"cumulative_c{PRIMARY_COST:g}"
    doc["comparisons"]["cumulative_primary"] = primary if primary in doc["cumulative"] else None
    doc["comparisons"]["comparator_of"] = comparator_of
    doc["comparisons"]["secondary"] = [*doc["comparisons"]["secondary"], *doc["cumulative"], *doc["restricted"]]


def load_extra_methods(lab: ms.MultiLab, frozen: Mapping[str, Any], weights_dir: Path) -> dict[str, dict[str, torch.Tensor]]:
    out: dict[str, dict[str, torch.Tensor]] = {}
    model = lab.model
    for name, entry in frozen.get("single_at_equal_task", {}).items():
        out[name] = lab.execute(lab.solve(ms.config_from(entry["config"]))).merged
    for name, entry in frozen.get("cumulative", {}).items():
        merged = replay_graph(model, Path(entry["graph"]))
        out[name] = merged
    primary = frozen["comparisons"].get("cumulative_primary")
    if primary:
        factors: dict[str, Any] = {}
        for site, weight in out[primary].items():
            left, right = ms.low_rank(weight - model.tensors[site], 128)
            factors[site] = (left, right, None)
        out["control_random_cumulative"] = lab.random_like(ms.MultiCandidate(dict(out[primary]), factors)).merged
    for name, entry in frozen.get("restricted", {}).items():
        weights = torch.load(weights_dir / f"{name}.pt")
        if bl.weights_digest(weights) != entry["weights_digest"]:
            raise RuntimeError(f"{name} weights do not match the freeze")
        out[name] = weights
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=["search", "restricted"])
    parser.add_argument("--cost", type=float, default=PRIMARY_COST)
    parser.add_argument("--checkpoint", default="/tmp/pythia-14m")
    parser.add_argument("--dir", default="docs/retention-multisite")
    parser.add_argument("--weights-dir", default="/tmp/nf-ms-weights")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--training-device", default="auto")
    parser.add_argument("--gpu-python")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model = PythiaModel.from_pretrained_dir(args.checkpoint, model_name="EleutherAI/pythia-14m")
    if args.phase == "search":
        run_search(model, args.cost, Path(args.dir))
    else:
        run_restricted(
            model,
            args.cost,
            Path(args.dir),
            Path(args.weights_dir),
            training_device=args.training_device,
            gpu_python=args.gpu_python,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
