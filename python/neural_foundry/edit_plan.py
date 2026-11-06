"""Compose direct weight edits as one previewable, constrained, reversible plan.

Every step compiles through the Rust executor, including derived edits such as `spectral_clip`,
which become a verified `add_delta`. A plan touching several tensors runs as a chain of single-tensor
Rust transactions: all-or-nothing here, because nothing reaches a model until every step and
constraint passes, but not yet one Rust plan.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch

from .export_formats import (
    _apply_state_tensors_to_clone,
    _extract_model_state_tensors,
    export_huggingface_safetensors,
)
from .gym_proposal import GymProposalError, ProposedOperation, ResolvedGymWeightProposal, ResolvedOperation
from .ir import WeightIRTransactionBuilder, unpack_matrix_bytes
from .records import NativeError, canonicalize, content_digest


class EditConstraintViolation(ValueError):
    """A plan broke a declared limit. Nothing was applied."""


@dataclass(frozen=True)
class EditConstraints:
    max_changed_parameters: int | None = None
    max_parameter_magnitude: float | None = None
    max_frobenius_delta: float | None = None
    max_spectral_delta: float | None = None
    allowed_tensor_sites: tuple[str, ...] | None = None
    forbidden_tensors: tuple[str, ...] = ()
    preserve_prompts: tuple[str, ...] = ()
    min_retention: float = 1.0


@dataclass(frozen=True)
class Region:
    """Half-open row and column ranges. `None` means the whole axis."""

    rows: tuple[int, int] | None = None
    cols: tuple[int, int] | None = None

    def resolve(self, shape: Sequence[int]) -> tuple[tuple[int, int], tuple[int, int]]:
        rows = self.rows or (0, int(shape[0]))
        cols = self.cols or (0, int(shape[1]))
        for (lo, hi), width in ((rows, shape[0]), (cols, shape[1])):
            if not 0 <= lo < hi <= width:
                raise ValueError(f"Region {rows}, {cols} is outside tensor shape {tuple(shape)}")
        return (rows[0], cols[0]), (rows[1] - rows[0], cols[1] - cols[0])


@dataclass(frozen=True)
class StepReceipt:
    tensor: str
    operation: str
    plan_digest: str
    candidate_digest: str
    before_parameter_digest: str
    after_parameter_digest: str
    changed_parameters: int
    spectrum_before: tuple[float, ...] | None = None
    spectrum_after: tuple[float, ...] | None = None
    resolved: Mapping[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class PlanCandidate:
    model: Any
    edited_tensors: dict[str, torch.Tensor]
    steps: tuple[StepReceipt, ...]
    plan_digest: str
    changed_parameters: int
    frobenius_delta: float
    spectral_delta: float
    retention: float | None
    state_digest: str = ""


EDIT_GRAPH_MAX_OPERAND_BYTES = 256 * 1024 * 1024
_MATRIX_ARGUMENTS = ("delta_matrix", "left_factor", "right_factor", "orthonormal_basis", "value_matrix", "mask_matrix", "map_matrix")
_SCALAR_ARGUMENTS = ("operation_type", "factor", "projection_side", "axis", "permutation", "region_start", "region_shape")


class EditGraphError(ValueError):
    """An edit graph couldn't be replayed, or replay didn't reproduce the signed result."""


def tensor_state_digest(tensors: Mapping[str, torch.Tensor]) -> str:
    """Digest of a whole model's weights, simple enough to recompute without this package.

    SHA-256 over each tensor in sorted name order: UTF-8 name, NUL, shape as comma-joined decimals,
    NUL, then the values as little-endian f64.
    """
    hasher = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().to(torch.float64)
        hasher.update(name.encode("utf-8") + b"\0" + ",".join(str(d) for d in tensor.shape).encode("ascii") + b"\0")
        hasher.update(tensor.numpy().astype("<f8").tobytes())
    return f"sha256:{hasher.hexdigest()}"


def _encode_matrix(value: Any) -> dict[str, Any]:
    matrix = torch.as_tensor(value, dtype=torch.float64)
    raw = struct.pack(f"<{matrix.numel()}d", *matrix.reshape(-1).tolist())
    return {"shape": list(matrix.shape), "dtype": "f64", "data_b64": base64.b64encode(raw).decode("ascii")}


def _decode_matrix(doc: Mapping[str, Any]) -> torch.Tensor:
    shape = [int(d) for d in doc["shape"]]
    if len(shape) != 2 or doc.get("dtype") != "f64":
        raise EditGraphError("Graph operands must be 2D f64 matrices")
    raw = base64.b64decode(doc["data_b64"], validate=True)
    if len(raw) != shape[0] * shape[1] * 8:
        raise EditGraphError("Graph operand size doesn't match its shape")
    return torch.tensor(struct.unpack(f"<{shape[0] * shape[1]}d", raw), dtype=torch.float64).reshape(shape)


def _spectrum(matrix: torch.Tensor, limit: int = 16) -> tuple[float, ...]:
    return tuple(round(float(v), 6) for v in torch.linalg.svdvals(matrix)[:limit])


@dataclass
class EditPlan:
    base_model: Any
    constraints: EditConstraints = field(default_factory=EditConstraints)
    _steps: list[tuple[str, str, Callable[[torch.Tensor], dict[str, Any]], bool]] = field(default_factory=list)
    _last: PlanCandidate | None = None
    _proposal: Mapping[str, Any] | None = None

    @classmethod
    def from_resolved_proposal(cls, base_model: Any, proposal: ResolvedGymWeightProposal) -> EditPlan:
        """Compile a frozen Gym proposal for exactly this base model, or raise.

        `base_checkpoint_digest` and `expected_candidate_digest` are `tensor_state_digest` values: they
        cover the weights as f64, so they don't depend on the checkpoint's storage dtype or file layout.
        The returned plan has been previewed and reproduced the expected candidate.
        """
        model_type = str(base_model.config.get("model_type", ""))
        if proposal.architecture != model_type:
            raise GymProposalError(f"Proposal is for {proposal.architecture!r}, but the base model is {model_type!r}")
        if tensor_state_digest(_extract_model_state_tensors(base_model)) != proposal.base_checkpoint_digest:
            raise GymProposalError("This isn't the base checkpoint the proposal was resolved for")
        try:
            constraints = EditConstraints(**{k: tuple(v) if isinstance(v, list) else v for k, v in proposal.spec.constraints.items()})
        except TypeError as exc:
            raise GymProposalError(f"Unknown proposal constraint: {exc}") from exc
        plan = cls(base_model, constraints)
        for planned, resolved in zip(proposal.spec.operations, proposal.operations, strict=True):
            plan._add_resolved(planned, resolved)
        candidate = plan.preview()
        if candidate.state_digest != proposal.expected_candidate_digest:
            raise GymProposalError(f"Proposal produced {candidate.state_digest}, not the expected {proposal.expected_candidate_digest}")
        plan._proposal = {
            "proposal_id": proposal.spec.proposal_id,
            "evidence_source": proposal.spec.evidence_source,
            "synthesis_split_digest": proposal.spec.synthesis_split_digest,
            "schema_version": proposal.schema_version,
        }
        return plan

    def _add_resolved(self, planned: ProposedOperation, resolved: ResolvedOperation) -> None:
        tensors = self.base_model.tensors
        if planned.site.role != "mlp_down" or planned.site.layer >= int(self.base_model.config["num_hidden_layers"]):
            raise GymProposalError(f"The base model has no site {planned.site!r}")
        if resolved.tensor != self.base_model.mlp_down_proj_site(planned.site.layer):
            raise GymProposalError(f"{resolved.tensor!r} isn't the tensor for {planned.site!r}")
        if tuple(tensors[resolved.tensor].shape) != tuple(resolved.shape):
            raise GymProposalError(f"{resolved.tensor!r} has shape {tuple(tensors[resolved.tensor].shape)}, not {tuple(resolved.shape)}")
        unknown = (set(resolved.arguments) - (set(_SCALAR_ARGUMENTS) - {"operation_type"})) | (set(resolved.operands) - set(_MATRIX_ARGUMENTS))
        if unknown:
            raise GymProposalError(f"Resolved operation has unknown arguments: {sorted(unknown)}")
        if sum(len(str(doc.get("data_b64", ""))) for doc in resolved.operands.values()) > EDIT_GRAPH_MAX_OPERAND_BYTES:
            raise GymProposalError("Resolved operands exceed the size limit")
        try:
            operands = {k: _decode_matrix(v) for k, v in resolved.operands.items()}
        except (EditGraphError, KeyError, TypeError, ValueError) as exc:
            raise GymProposalError(f"Invalid operand: {exc}") from exc
        arguments = dict(resolved.arguments)

        def build(w: torch.Tensor) -> dict[str, Any]:
            return {**self._target(None, w.shape), **arguments, **operands, "operation_type": resolved.operation}

        self._add(resolved.tensor, resolved.operation, build)

    def add_constraint(self, **limits: Any) -> EditPlan:
        self.constraints = replace(self.constraints, **limits)
        self._last = None
        return self

    def _add(self, tensor: str, op: str, build: Callable[[torch.Tensor], dict[str, Any]], spectral: bool = False) -> EditPlan:
        tensors = self.base_model.tensors
        if tensor not in tensors:
            raise ValueError(f"Unknown tensor {tensor!r}")
        if tensors[tensor].ndim != 2:
            raise ValueError(f"Direct edits need a matrix; {tensor!r} has {tensors[tensor].ndim} dimensions")
        c = self.constraints
        if tensor in c.forbidden_tensors or (c.allowed_tensor_sites is not None and tensor not in c.allowed_tensor_sites):
            raise EditConstraintViolation(f"Tensor {tensor!r} is not allowed by this plan's constraints")
        self._steps.append((tensor, op, build, spectral))
        self._last = None
        self._proposal = None
        return self

    @staticmethod
    def _target(region: Region | None, shape: Sequence[int]) -> dict[str, Any]:
        start, size = (region or Region()).resolve(shape)
        return {"region_start": start, "region_shape": size}

    # ---- Operations the Rust executor implements directly.
    def scale(self, tensor: str, factor: float, *, region: Region | None = None) -> EditPlan:
        return self._add(tensor, "scale", lambda w: {"operation_type": "scale", "factor": factor, **self._target(region, w.shape)})

    def add(self, tensor: str, delta: Any, *, region: Region | None = None) -> EditPlan:
        return self._add(tensor, "add_delta", lambda w: {"operation_type": "add_delta", "delta_matrix": delta, **self._target(region, w.shape)})

    def low_rank(self, tensor: str, left: Any, right: Any, *, strength: float = 1.0, region: Region | None = None) -> EditPlan:
        return self._add(tensor, "low_rank_update", lambda w: {"operation_type": "low_rank_update", "left_factor": left, "right_factor": right, "factor": strength, **self._target(region, w.shape)})

    def project(self, tensor: str, basis: Any, *, side: str = "left", strength: float = 1.0, region: Region | None = None) -> EditPlan:
        return self._add(tensor, "subspace_projection", lambda w: {"operation_type": "subspace_projection", "orthonormal_basis": basis, "projection_side": side, "factor": strength, **self._target(region, w.shape)})

    def replace_slice(self, tensor: str, value: Any, *, region: Region | None = None) -> EditPlan:
        return self._add(tensor, "replace_slice", lambda w: {"operation_type": "replace_slice", "value_matrix": value, **self._target(region, w.shape)})

    def mask(self, tensor: str, mask: Any, *, region: Region | None = None) -> EditPlan:
        return self._add(tensor, "mask", lambda w: {"operation_type": "mask", "mask_matrix": mask, **self._target(region, w.shape)})

    def permute(self, tensor: str, axis: int, permutation: Sequence[int], *, region: Region | None = None) -> EditPlan:
        return self._add(tensor, "permute_axis", lambda w: {"operation_type": "permute_axis", "axis": axis, "permutation": list(permutation), **self._target(region, w.shape)})

    def linear_map(self, tensor: str, matrix: Any, *, side: str = "left", region: Region | None = None) -> EditPlan:
        return self._add(tensor, "linear_map", lambda w: {"operation_type": "linear_map", "map_matrix": matrix, "projection_side": side, **self._target(region, w.shape)})

    def select_channels(self, tensor: str, channels: Sequence[int], *, axis: int = 0) -> EditPlan:
        """Zero every channel on `axis` except those listed."""
        keep = set(int(c) for c in channels)

        def build(w: torch.Tensor) -> dict[str, Any]:
            mask = torch.zeros_like(w)
            index = sorted(keep)
            if axis == 0:
                mask[index, :] = 1.0
            else:
                mask[:, index] = 1.0
            return {"operation_type": "mask", "mask_matrix": mask, **self._target(None, w.shape)}

        return self._add(tensor, "select_channels", build)

    # ---- Derived edits: computed here, then executed as a verified add_delta.
    def _derived(self, tensor: str, op: str, compute: Callable[[torch.Tensor], torch.Tensor], spectral: bool = False) -> EditPlan:
        return self._add(tensor, op, lambda w: {"operation_type": "add_delta", "delta_matrix": compute(w) - w, **self._target(None, w.shape)}, spectral)

    def soft_threshold(self, tensor: str, threshold: float) -> EditPlan:
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        return self._derived(tensor, "soft_threshold", lambda w: torch.sign(w) * torch.clamp(w.abs() - threshold, min=0.0))

    def clamp(self, tensor: str, limit: float) -> EditPlan:
        if limit <= 0:
            raise ValueError("limit must be positive")
        return self._derived(tensor, "clamp", lambda w: torch.clamp(w, -limit, limit))

    def _spectral(self, tensor: str, op: str, edit: Callable[[torch.Tensor], torch.Tensor]) -> EditPlan:
        """Build the delta from only the singular values that move, so an edit that moves none is exactly zero."""

        def build(w: torch.Tensor) -> dict[str, Any]:
            u, s, vh = torch.linalg.svd(w, full_matrices=False)
            change = edit(s) - s
            moved = change != 0
            return {"operation_type": "add_delta", "delta_matrix": (u[:, moved] * change[moved]) @ vh[moved], **self._target(None, w.shape)}

        return self._add(tensor, op, build, True)

    def spectral_clip(self, tensor: str, max_singular_value: float) -> EditPlan:
        if max_singular_value <= 0:
            raise ValueError("max_singular_value must be positive")
        return self._spectral(tensor, "spectral_clip", lambda s: torch.clamp(s, max=max_singular_value))

    def rank_truncate(self, tensor: str, rank: int) -> EditPlan:
        if rank < 1:
            raise ValueError("rank must be at least 1")

        def keep(s: torch.Tensor) -> torch.Tensor:
            out = s.clone()
            out[rank:] = 0.0
            return out

        return self._spectral(tensor, "rank_truncate", keep)

    def remove_noise_floor(self, tensor: str, relative_threshold: float) -> EditPlan:
        return self._spectral(tensor, "remove_noise_floor", lambda s: torch.where(s >= relative_threshold * s[0], s, torch.zeros_like(s)))

    def spectral_shift(self, tensor: str, shift: float) -> EditPlan:
        return self._spectral(tensor, "spectral_shift", lambda s: torch.clamp(s + shift, min=0.0))

    def interpolate(self, tensor: str, other: torch.Tensor, weight: float) -> EditPlan:
        if not 0.0 <= weight <= 1.0:
            raise ValueError("weight must be in [0, 1]")
        return self._derived(tensor, "interpolate", lambda w: (1.0 - weight) * w + weight * other.to(torch.float64))

    # ---- Activation-guided edits.
    def _activation(self, prompts: Sequence[str], layer: int) -> torch.Tensor:
        model = self.base_model
        return torch.stack(
            [model.forward_with_mlp_key(model.tokenizer.encode(p), capture_layer=layer)[1].to(torch.float64) for p in prompts]
        ).mean(0)

    def suppress_activation_direction(
        self, layer: int, positive_prompts: Sequence[str], negative_prompts: Sequence[str], *, strength: float = 1.0
    ) -> EditPlan:
        """Remove the input direction separating two prompt groups from one layer's weights."""
        tensor = self.base_model.mlp_down_proj_site(layer)
        direction = self._activation(positive_prompts, layer) - self._activation(negative_prompts, layer)
        norm = torch.linalg.norm(direction)
        if float(norm) <= 1e-9:
            raise ValueError("The prompt groups have no measurable activation difference")
        unit = direction / norm
        return self._add(
            tensor,
            "suppress_activation_direction",
            lambda w: {
                "operation_type": "low_rank_update",
                "left_factor": (-strength * (w @ unit)).unsqueeze(1),
                "right_factor": unit.unsqueeze(0),
                "factor": 1.0,
                **self._target(None, w.shape),
            },
        )

    def ablate_channel(self, tensor: str, channel: int, *, axis: int = 0) -> EditPlan:
        def build(w: torch.Tensor) -> dict[str, Any]:
            mask = torch.ones_like(w)
            if axis == 0:
                mask[channel, :] = 0.0
            else:
                mask[:, channel] = 0.0
            return {"operation_type": "mask", "mask_matrix": mask, **self._target(None, w.shape)}

        return self._add(tensor, "ablate_channel", build)

    # ---- Existing methods join the same graph.
    def adopt_weight(self, tensor: str, edited_weight: torch.Tensor, *, method: str = "external_method") -> EditPlan:
        """Bring in a weight from ROME, MEMIT, AlphaEdit, a merge or anything else, as a verified step."""
        target = edited_weight.detach().to(torch.float64)
        return self._derived(tensor, method, lambda w: target)

    def branch(self) -> EditPlan:
        """An independent copy of this plan. Steps added to either don't affect the other."""
        return EditPlan(self.base_model, self.constraints, list(self._steps))

    # ---- Lifecycle.
    def preview(self) -> PlanCandidate:
        """Run every step in Rust and check every limit, without changing the base model."""
        if not self._steps:
            raise ValueError("This plan has no steps")
        base = _extract_model_state_tensors(self.base_model)
        working: dict[str, torch.Tensor] = {}
        receipts: list[StepReceipt] = []
        for tensor, op, build, spectral in self._steps:
            current = working.get(tensor, base[tensor]).to(torch.float64)
            arguments = build(current)
            tx = WeightIRTransactionBuilder.execute_matrix_edit(current, provenance_tag=f"edit_plan.{op}.v1", engine_name="studio.edit_plan", **arguments)
            edited = torch.tensor(unpack_matrix_bytes(tx.edited_bytes, tuple(current.shape), "f64"), dtype=torch.float64)
            working[tensor] = edited
            receipts.append(
                StepReceipt(
                    tensor, op, tx.plan_digest, tx.candidate_digest, tx.before_parameter_digest,
                    tx.after_parameter_digest, tx.changed_parameters,
                    _spectrum(current) if spectral else None, _spectrum(edited) if spectral else None,
                    arguments,
                )
            )
        changed = 0
        frobenius = 0.0
        spectral_delta = 0.0
        magnitude = 0.0
        for name, edited in working.items():
            delta = edited - base[name].to(torch.float64)
            changed += int((delta != 0).sum())
            frobenius += float(torch.linalg.norm(delta)) ** 2
            spectral_delta = max(spectral_delta, float(torch.linalg.matrix_norm(delta, ord=2)))
            magnitude = max(magnitude, float(edited.abs().max()))
        frobenius **= 0.5
        c = self.constraints
        for label, value, limit in (
            ("changed parameters", changed, c.max_changed_parameters),
            ("parameter magnitude", magnitude, c.max_parameter_magnitude),
            ("Frobenius change", frobenius, c.max_frobenius_delta),
            ("spectral change", spectral_delta, c.max_spectral_delta),
        ):
            if limit is not None and value > limit:
                raise EditConstraintViolation(f"{label} {value:.6g} exceeds the limit of {limit:.6g}")
        candidate_model = _apply_state_tensors_to_clone(self.base_model, {**base, **working})
        retention = None
        if c.preserve_prompts:
            kept = sum(
                candidate_model.evaluate_prompt(p).top1_token_id == self.base_model.evaluate_prompt(p).top1_token_id
                for p in c.preserve_prompts
            )
            retention = kept / len(c.preserve_prompts)
            if retention < c.min_retention:
                raise EditConstraintViolation(f"retention {retention:.3f} is below the required {c.min_retention:.3f}")
        digest = str(content_digest(canonicalize({"schema_version": 1, "kind": "nf_edit_plan", "steps": [r.plan_digest for r in receipts]})))
        self._last = PlanCandidate(
            candidate_model, working, tuple(receipts), digest, changed, frobenius, spectral_delta, retention,
            tensor_state_digest(candidate_model.tensors),
        )
        return self._last

    def execute(self) -> PlanCandidate:
        """Return the checked candidate. The base model is never modified; the candidate is a new model."""
        return self._last or self.preview()

    def to_graph(self) -> dict[str, Any]:
        """An executable record of this plan: every operand needed to reproduce the candidate."""
        candidate = self.execute()
        steps = []
        for index, step in enumerate(candidate.steps):
            arguments = dict(step.resolved)
            steps.append(
                {
                    "id": f"step_{index}",
                    "depends_on": [f"step_{index - 1}"] if index else [],
                    "tensor": step.tensor,
                    "operation": step.operation,
                    "arguments": {k: arguments[k] for k in _SCALAR_ARGUMENTS if arguments.get(k) is not None},
                    "operands": {k: _encode_matrix(arguments[k]) for k in _MATRIX_ARGUMENTS if arguments.get(k) is not None},
                    "before_parameter_digest": step.before_parameter_digest,
                    "after_parameter_digest": step.after_parameter_digest,
                }
            )
        return {
            "schema_version": 1,
            "kind": "nf_edit_graph_v1",
            "base_model_id": str(getattr(self.base_model, "model_name", "")),
            "base_state_digest": tensor_state_digest(_extract_model_state_tensors(self.base_model)),
            "candidate_state_digest": candidate.state_digest,
            "steps": steps,
            "plan_digest": candidate.plan_digest,
            **({"proposal": dict(self._proposal)} if self._proposal else {}),
        }

    def rollback(self) -> Any:
        """Discard the candidate and steps, and return the untouched base model."""
        self._last = None
        self._steps.clear()
        return self.base_model

    def export(self, output_dir: str | Path, *, dtype: str = "f64", source_checkpoint_dir: str | Path | None = None) -> Path:
        return export_huggingface_safetensors(self.execute().model, output_dir, dtype=dtype, source_checkpoint_dir=source_checkpoint_dir)


@dataclass(frozen=True)
class SearchResult:
    label: str
    candidate: PlanCandidate | None
    development_accuracy: float
    retention: float
    frobenius_delta: float
    is_control: bool
    on_pareto_frontier: bool
    rejection_reason: str | None = None


def search_edits(
    model: Any,
    *,
    development: Sequence[tuple[str, str]],
    retention_prompts: Sequence[str],
    proposals: Mapping[str, Callable[[EditPlan], Any]],
    constraints: EditConstraints | None = None,
    seed: int = 17,
) -> tuple[SearchResult, ...]:
    """Score proposals beside no-op and random controls, keeping rejected ones, on development and retention data."""

    def scores(candidate: Any) -> tuple[float, float]:
        accuracy = sum(candidate.evaluate_prompt(p, target_token=t).matched_target for p, t in development) / max(len(development), 1)
        kept = sum(candidate.evaluate_prompt(p).top1_token_id == model.evaluate_prompt(p).top1_token_id for p in retention_prompts)
        return accuracy, kept / max(len(retention_prompts), 1)

    results: list[SearchResult] = []
    accuracy, kept = scores(model)
    results.append(SearchResult("control_noop", None, accuracy, kept, 0.0, True, False))
    site = model.mlp_down_proj_site(int(model.config["num_hidden_layers"]) - 1)
    weight = model.tensors[site].to(torch.float64)
    generator = torch.Generator().manual_seed(seed)
    left = torch.randn(weight.shape[0], 1, dtype=torch.float64, generator=generator)
    right = torch.randn(1, weight.shape[1], dtype=torch.float64, generator=generator)
    scale = 0.1 * float(torch.linalg.norm(weight)) / float(torch.linalg.norm(left) * torch.linalg.norm(right))
    plans: dict[str, tuple[Callable[[EditPlan], Any], bool]] = {
        "control_random": (lambda p: p.low_rank(site, left, right, strength=scale), True),
        **{label: (build, False) for label, build in proposals.items()},
    }
    for label, (build, is_control) in plans.items():
        plan = EditPlan(model, constraints or EditConstraints())
        try:
            build(plan)
            candidate = plan.preview()
        except (EditConstraintViolation, ValueError, RuntimeError, NativeError) as exc:
            results.append(SearchResult(label, None, 0.0, 0.0, 0.0, is_control, False, str(exc)))
            continue
        accuracy, kept = scores(candidate.model)
        results.append(SearchResult(label, candidate, accuracy, kept, candidate.frobenius_delta, is_control, False))

    eligible = [r for r in results if r.rejection_reason is None and not r.is_control]

    def dominated(r: SearchResult) -> bool:
        return any(
            o.development_accuracy >= r.development_accuracy
            and o.retention >= r.retention
            and o.frobenius_delta <= r.frobenius_delta
            and (o.development_accuracy, o.retention, -o.frobenius_delta) != (r.development_accuracy, r.retention, -r.frobenius_delta)
            for o in eligible
        )

    frontier = {r.label for r in eligible if not dominated(r)}
    ranked = [replace(r, on_pareto_frontier=r.label in frontier) for r in results]
    return tuple(sorted(ranked, key=lambda r: (not r.on_pareto_frontier, -r.development_accuracy, -r.retention)))


def apply_edit_graph(base_model: Any, graph: Mapping[str, Any]) -> PlanCandidate:
    """Replay a graph through Rust on `base_model`. Raises unless every digest is reproduced."""
    if graph.get("kind") != "nf_edit_graph_v1" or graph.get("schema_version") != 1:
        raise EditGraphError("Not an nf_edit_graph_v1 document")
    steps = graph.get("steps")
    if not isinstance(steps, list) or not steps:
        raise EditGraphError("Graph has no steps")
    total = 0
    for step in steps:
        for operand in step.get("operands", {}).values():
            total += len(operand["data_b64"])
    if total > EDIT_GRAPH_MAX_OPERAND_BYTES:
        raise EditGraphError("Graph operands exceed the size limit")
    base = _extract_model_state_tensors(base_model)
    # The per-step digests only cover edited tensors, so a base differing elsewhere needs this check.
    if tensor_state_digest(base) != graph.get("base_state_digest"):
        raise EditGraphError("This isn't the base model the graph was made for")
    working: dict[str, torch.Tensor] = {}
    receipts: list[StepReceipt] = []
    for step in steps:
        tensor, op = str(step["tensor"]), str(step["operation"])
        if tensor not in base:
            raise EditGraphError(f"Base model has no tensor {tensor!r}")
        unknown = (set(step["arguments"]) - set(_SCALAR_ARGUMENTS)) | (set(step["operands"]) - set(_MATRIX_ARGUMENTS))
        if unknown:
            raise EditGraphError(f"Graph step has unknown arguments: {sorted(unknown)}")
        current = working.get(tensor, base[tensor]).to(torch.float64)
        arguments = {**step["arguments"], **{k: _decode_matrix(v) for k, v in step["operands"].items()}}
        tx = WeightIRTransactionBuilder.execute_matrix_edit(
            current, provenance_tag=f"edit_plan.{op}.v1", engine_name="studio.edit_plan", **arguments
        )
        if tx.before_parameter_digest != step["before_parameter_digest"]:
            raise EditGraphError(f"{step['id']}: this isn't the base model the graph was made for")
        if tx.after_parameter_digest != step["after_parameter_digest"]:
            raise EditGraphError(f"{step['id']}: replay produced different weights")
        working[tensor] = torch.tensor(unpack_matrix_bytes(tx.edited_bytes, tuple(current.shape), "f64"), dtype=torch.float64)
        receipts.append(
            StepReceipt(tensor, op, tx.plan_digest, tx.candidate_digest, tx.before_parameter_digest, tx.after_parameter_digest, tx.changed_parameters)
        )
    digest = str(content_digest(canonicalize({"schema_version": 1, "kind": "nf_edit_plan", "steps": [r.plan_digest for r in receipts]})))
    if digest != graph.get("plan_digest"):
        raise EditGraphError("Replay produced a different plan digest")
    changed = frobenius = spectral = 0.0
    for name, edited in working.items():
        delta = edited - base[name].to(torch.float64)
        changed += int((delta != 0).sum())
        frobenius += float(torch.linalg.norm(delta)) ** 2
        spectral = max(spectral, float(torch.linalg.matrix_norm(delta, ord=2)))
    state = tensor_state_digest({**base, **working})
    if state != graph.get("candidate_state_digest"):
        raise EditGraphError("Replay produced a different whole-model digest")
    return PlanCandidate(
        _apply_state_tensors_to_clone(base_model, {**base, **working}), working, tuple(receipts), digest, int(changed), frobenius**0.5, spectral, None, state
    )


def sign_edit_graph(graph: Mapping[str, Any], *, signing_key: str, signer_id: str = "nf.edit_graph.v1") -> dict[str, Any]:
    from .patch_ux import sign_patch_envelope

    return sign_patch_envelope(graph, signing_key=signing_key, signer_id=signer_id)


def apply_signed_edit_graph(base_model: Any, envelope: Mapping[str, Any], *, public_key: str) -> PlanCandidate:
    """Check the signature against a trusted key first, then replay and check every digest."""
    from .patch_ux import verify_patch_signature

    verify_patch_signature(envelope, public_key=public_key)
    return apply_edit_graph(base_model, envelope["patch"])


def verify_exported_candidate(checkpoint_dir: str | Path, expected_state_digest: str) -> str:
    """Read an exported checkpoint from disk and raise unless its weights match the signed digest.

    Only an f64 export can pass: narrower dtypes round the edit, so they aren't the signed weights.
    """
    blob = (Path(checkpoint_dir) / "model.safetensors").read_bytes()
    (header_size,) = struct.unpack_from("<Q", blob, 0)
    header = json.loads(blob[8 : 8 + header_size].decode("utf-8"))
    tensors: dict[str, torch.Tensor] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if meta["dtype"] != "F64":
            raise EditGraphError(f"Export stores {name!r} as {meta['dtype']}, which can't hold the signed f64 weights exactly")
        start, end = meta["data_offsets"]
        tensors[name] = torch.frombuffer(bytearray(blob[8 + header_size + start : 8 + header_size + end]), dtype=torch.float64).reshape(meta["shape"])
    actual = tensor_state_digest(tensors)
    if actual != expected_state_digest:
        raise EditGraphError(f"Exported weights {actual} don't match the signed candidate {expected_state_digest}")
    return actual


def _replay_main(argv: Sequence[str] | None = None) -> int:
    """Replay a signed graph from only a base checkpoint, the envelope and a public key."""
    parser = argparse.ArgumentParser(prog="python -m neural_foundry.edit_plan", description=_replay_main.__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--envelope", required=True)
    parser.add_argument("--public-key-file", required=True)
    parser.add_argument("--export-dir", help="Also export the replayed candidate as f64 safetensors and verify it")
    args = parser.parse_args(argv)

    from .pythia import PythiaModel

    model = PythiaModel.from_pretrained_dir(args.checkpoint)
    envelope = json.loads(Path(args.envelope).read_text(encoding="utf-8"))
    candidate = apply_signed_edit_graph(model, envelope, public_key=Path(args.public_key_file).read_text(encoding="utf-8").strip())
    report = {
        "plan_digest": candidate.plan_digest,
        "candidate_state_digest": candidate.state_digest,
        "steps": [{"tensor": s.tensor, "operation": s.operation, "after_parameter_digest": s.after_parameter_digest} for s in candidate.steps],
    }
    if args.export_dir:
        export_huggingface_safetensors(candidate.model, args.export_dir, dtype="f64", source_checkpoint_dir=args.checkpoint)
        report["export_state_digest"] = verify_exported_candidate(args.export_dir, candidate.state_digest)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_replay_main())
