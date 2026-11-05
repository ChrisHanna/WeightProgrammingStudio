"""Weight IR validation, pure native compilation, and reusable transaction builder."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from . import _native, surgeon
from .records import (
    CancellationHandle,
    RecordHandle,
    RecordInput,
    _wire,
    canonicalize,
    content_digest,
    negotiate,
    validate,
)

PlanHandle = _native.PlanHandle


def program(record: RecordInput) -> RecordHandle:
    return validate("program", record)


def search_space(record: RecordInput) -> RecordHandle:
    return validate("search_space", record)


def compile(
    source: _native.SourceHandle,
    program: RecordInput,
    search_space: RecordInput,
    policy: RecordInput,
    operands: Mapping[str, bytes] | None = None,
    cancellation: CancellationHandle | None = None,
) -> PlanHandle:
    """Use actual source/operand bytes and the explicit frozen Rust policy."""
    negotiate()
    return _native.compile_plan(
        source,
        _wire(program),
        _wire(search_space),
        _wire(policy),
        dict(operands or {}),
        cancellation,
    )


def _sha256(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def pack_f64_matrix(matrix: Any) -> tuple[bytes, tuple[int, int]]:
    """Packs a 2D float matrix or torch.Tensor into little-endian f64 bytes without importing torch at top level."""
    if hasattr(matrix, "detach") and hasattr(matrix, "reshape"):
        shape_tuple = (int(matrix.shape[0]), int(matrix.shape[1]))
        flat = [float(x) for x in matrix.detach().contiguous().reshape(-1).tolist()]
    else:
        rows = [list(r) for r in matrix]
        shape_tuple = (len(rows), len(rows[0]) if rows else 0)
        flat = [float(x) for r in rows for x in r]
    raw = struct.pack(f"<{len(flat)}d", *flat)
    return raw, shape_tuple


def build_named_safetensors_f64(name: str, matrix: Any) -> tuple[bytes, str, int, tuple[int, int]]:
    """Builds a canonical single-tensor F64 Safetensors blob and returns `(safetensors_bytes, raw_digest, raw_len, shape)`."""
    raw_bytes, shape = pack_f64_matrix(matrix)
    header = {
        name: {
            "dtype": "F64",
            "shape": [shape[0], shape[1]],
            "data_offsets": [0, len(raw_bytes)],
        }
    }
    raw_hdr = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    pad = (8 - (len(raw_hdr) % 8)) % 8
    raw_hdr += b" " * pad
    st = struct.pack("<Q", len(raw_hdr)) + raw_hdr + raw_bytes
    return st, _sha256(raw_bytes), len(raw_bytes), shape


@dataclass(frozen=True)
class CompiledTransactionReceipt:
    """Verified result of compiling and executing a Weight IR operation in Rust."""

    operation_type: str
    shape: tuple[int, int]
    edited_bytes: bytes
    plan_digest: str
    candidate_digest: str
    before_parameter_digest: str
    after_parameter_digest: str
    changed_parameters: int
    reload_parity_verified: bool
    receipt: dict[str, Any]


class WeightIRTransactionBuilder:
    """Reusable builder that compiles and executes Weight IR operations through Rust (`nf-weight-ir` + `nf-surgeon`).

    Eliminates manual JSON boilerplate when authoring custom weight-editing methods,
    model-merging adapters, or autonomous agent tools.
    """

    @staticmethod
    def execute_matrix_edit(
        base_matrix: Any,
        *,
        operation_type: str,
        smoke_probe: Sequence[float] | Any | None = None,
        provenance_tag: str = "studio.custom_edit.v1",
        engine_name: str = "studio.weight_ir",
        factor: float = 1.0,
        delta_matrix: Any | None = None,
        left_factor: Any | None = None,
        right_factor: Any | None = None,
        orthonormal_basis: Any | None = None,
        projection_side: str = "left",
    ) -> CompiledTransactionReceipt:
        ckpt_bytes, w_before_digest, _, (out_dim, in_dim) = build_named_safetensors_f64("w", base_matrix)
        cfg_bytes = b"{}"
        cfg_digest = str(content_digest(cfg_bytes))
        ckpt_digest = str(content_digest(ckpt_bytes))
        param_set_digest = str(
            content_digest(
                canonicalize(
                    {
                        "schema_version": 1,
                        "kind": "tensor_set",
                        "tensors": [
                            {
                                "name": "w",
                                "shape": [out_dim, in_dim],
                                "dtype": "f64",
                                "content_digest": w_before_digest,
                            }
                        ],
                    }
                )
            )
        )
        manifest = {
            "schema_version": 1,
            "architecture": {"id": "fixture.cpu", "version": "1", "config": cfg_digest},
            "parameter_digest": param_set_digest,
            "parameters": {
                "w": {
                    "shape": [out_dim, in_dim],
                    "dtype": "f64",
                    "digest": w_before_digest,
                    "storage_group": "w",
                }
            },
            "buffers": {},
            "format": "dense_safetensors",
            "packing_policy": None,
            "assets": {"config": cfg_digest, "checkpoint": ckpt_digest},
            "usage_rights": "Apache-2.0",
        }
        source_handle = surgeon.load(manifest, ckpt_bytes, {"config": cfg_bytes})
        graph_digest = str(content_digest(f"{provenance_tag}.graph".encode()))
        base_model_ref = {
            "checkpoint": ckpt_digest,
            "manifest": source_handle.manifest_digest,
            "parameter_digest": source_handle.parameter_digest,
            "architecture_graph": graph_digest,
            "config": cfg_digest,
            "tokenizer": None,
        }
        contract_d = str(content_digest(f"{provenance_tag}.contract".encode()))
        objective_d = str(content_digest(f"{provenance_tag}.objective".encode()))
        exp_d = str(content_digest(f"{provenance_tag}.experiment".encode()))
        portfolio_d = str(content_digest(f"{provenance_tag}.portfolio".encode()))
        vproto_d = str(content_digest(f"{provenance_tag}.vproto".encode()))
        parity_d = str(content_digest(f"{provenance_tag}.parity".encode()))
        binding_d = str(content_digest(f"{provenance_tag}.binding".encode()))
        gym_d = str(content_digest(f"{provenance_tag}.gym".encode()))
        val_d = str(content_digest(f"{provenance_tag}.val".encode()))
        scorer_d = str(content_digest(f"{provenance_tag}.scorer".encode()))
        split_d = str(content_digest(f"{provenance_tag}.split".encode()))
        uncert_d = str(content_digest(f"{provenance_tag}.uncertainty".encode()))
        hyp_d = str(content_digest(f"{provenance_tag}.hypothesis".encode()))

        if smoke_probe is None:
            probe_list = [1.0 / max(in_dim, 1)] * in_dim
        elif hasattr(smoke_probe, "tolist"):
            probe_list = [float(v) for v in smoke_probe.tolist()]
        else:
            probe_list = [float(v) for v in smoke_probe]

        smoke_suite = {
            "schema_version": 1,
            "adapter": "nf.linear_fixture.v1",
            "probes": [
                {
                    "input": probe_list,
                    "expected": None,
                    "absolute_tolerance": 0.0,
                    "max_abs_output": 100_000.0,
                }
            ],
        }
        smoke_digest = str(content_digest(canonicalize(smoke_suite)))
        region = {"tensor": "w", "start": [0, 0], "shape": [out_dim, in_dim]}

        operands: dict[str, bytes] = {}
        artifacts: dict[str, Any] = {}
        transformations: dict[str, Any] = {}
        op_spec: dict[str, Any]

        if operation_type == "scale":
            transformations["scale"] = {
                "dtypes": ["f64"],
                "coefficient": {"min": -100.0, "max": 100.0},
                "rank": None,
                "sides": [],
                "max_operand_bytes": 0,
            }
            op_spec = {"type": "scale", "target": region, "factor": float(factor)}
        elif operation_type == "add_delta":
            if delta_matrix is None:
                raise ValueError("add_delta requires delta_matrix")
            d_bytes, _, d_len, d_shape = build_named_safetensors_f64("d", delta_matrix)
            d_digest = str(content_digest(d_bytes))
            operands["delta_op"] = d_bytes
            artifacts["delta_op"] = {
                "digest": d_digest,
                "tensor": "d",
                "dtype": "f64",
                "shape": [d_shape[0], d_shape[1]],
                "data_bytes": d_len,
                "artifact_bytes": len(d_bytes),
                "access": "development",
                "provenance": {
                    "schema_version": 1,
                    "origin": "supplied_fact",
                    "support": "untested",
                    "provenance": provenance_tag,
                    "evidence": [],
                    "access": "development",
                },
            }
            transformations["add_delta"] = {
                "dtypes": ["f64"],
                "coefficient": None,
                "rank": None,
                "sides": [],
                "max_operand_bytes": max(4_194_304, len(d_bytes) * 2),
            }
            op_spec = {"type": "add_delta", "target": region, "delta": "delta_op"}
        elif operation_type == "low_rank_update":
            if left_factor is None or right_factor is None:
                raise ValueError("low_rank_update requires left_factor and right_factor")
            u_bytes, _, u_len, u_shape = build_named_safetensors_f64("u", left_factor)
            v_bytes, _, v_len, v_shape = build_named_safetensors_f64("v", right_factor)
            u_digest = str(content_digest(u_bytes))
            v_digest = str(content_digest(v_bytes))
            rank = u_shape[1]
            operands["left_u"] = u_bytes
            operands["right_v"] = v_bytes
            artifacts["left_u"] = {
                "digest": u_digest,
                "tensor": "u",
                "dtype": "f64",
                "shape": [u_shape[0], u_shape[1]],
                "data_bytes": u_len,
                "artifact_bytes": len(u_bytes),
                "access": "development",
                "provenance": {
                    "schema_version": 1,
                    "origin": "supplied_fact",
                    "support": "untested",
                    "provenance": f"{provenance_tag}.left_u",
                    "evidence": [],
                    "access": "development",
                },
            }
            artifacts["right_v"] = {
                "digest": v_digest,
                "tensor": "v",
                "dtype": "f64",
                "shape": [v_shape[0], v_shape[1]],
                "data_bytes": v_len,
                "artifact_bytes": len(v_bytes),
                "access": "development",
                "provenance": {
                    "schema_version": 1,
                    "origin": "supplied_fact",
                    "support": "untested",
                    "provenance": f"{provenance_tag}.right_v",
                    "evidence": [],
                    "access": "development",
                },
            }
            transformations["low_rank_update"] = {
                "dtypes": ["f64"],
                "coefficient": {"min": -100.0, "max": 100.0},
                "rank": {"min": 1, "max": max(64, rank)},
                "sides": [],
                "max_operand_bytes": max(4_194_304, (len(u_bytes) + len(v_bytes)) * 2),
            }
            op_spec = {
                "type": "low_rank_update",
                "target": region,
                "left": "left_u",
                "right": "right_v",
                "strength": float(factor),
            }
        elif operation_type == "subspace_projection":
            if orthonormal_basis is None:
                raise ValueError("subspace_projection requires orthonormal_basis")
            q_bytes, _, q_len, q_shape = build_named_safetensors_f64("q", orthonormal_basis)
            q_digest = str(content_digest(q_bytes))
            operands["basis_q"] = q_bytes
            artifacts["basis_q"] = {
                "digest": q_digest,
                "tensor": "q",
                "dtype": "f64",
                "shape": [q_shape[0], q_shape[1]],
                "data_bytes": q_len,
                "artifact_bytes": len(q_bytes),
                "access": "development",
                "provenance": {
                    "schema_version": 1,
                    "origin": "supplied_fact",
                    "support": "untested",
                    "provenance": f"{provenance_tag}.basis_q",
                    "evidence": [],
                    "access": "development",
                },
            }
            transformations["subspace_projection"] = {
                "dtypes": ["f64"],
                "coefficient": {"min": -2.0, "max": 2.0},
                "rank": {"min": 1, "max": max(64, q_shape[1])},
                "sides": [projection_side],
                "max_operand_bytes": max(4_194_304, len(q_bytes) * 2),
            }
            op_spec = {
                "type": "subspace_projection",
                "target": region,
                "basis": "basis_q",
                "side": projection_side,
                "strength": float(factor),
                "orthonormal_tolerance": 0.000001,
            }
        else:
            raise ValueError(f"Unsupported matrix operation_type: {operation_type!r}")

        max_art_bytes = max(4_194_304, sum(len(b) for b in operands.values()) * 2)
        search_space_doc = {
            "schema_version": 1,
            "base_model": base_model_ref,
            "contract": contract_d,
            "objective": objective_d,
            "transformations": transformations,
            "sites": [
                {
                    "selector": {"type": "tensor", "name": "w"},
                    "role": "parameter",
                    "regions": {"type": "whole_tensor"},
                    "allowed_axes": [0, 1],
                }
            ],
            "constraints": {
                "max_changed_parameters": out_dim * in_dim,
                "max_artifact_bytes": max_art_bytes,
                "max_temporary_bytes": 16_777_216,
                "max_operations": 4,
                "allow_buffer_writes": False,
                "require_direct_weight_edits": True,
                "require_fully_gradient_free": True,
                "require_preserve_shapes": True,
                "quality_limits": [
                    {
                        "binding": binding_d,
                        "metric": "accuracy",
                        "units": "fraction",
                        "baseline": ckpt_digest,
                        "max_regression": 0.0,
                        "split": "holdout",
                        "uncertainty_protocol": uncert_d,
                    }
                ],
            },
            "strategy": {
                "proposal": "scripted",
                "search": "sequential",
                "selection": "pareto",
                "engine": engine_name,
                "engine_version": "1",
                "seed": 17,
            },
        }
        space_digest = str(content_digest(canonicalize(search_space_doc)))
        verification_block = {
            "portfolio": portfolio_d,
            "protocol": vproto_d,
            "gyms": [
                {
                    "binding": binding_d,
                    "gym": gym_d,
                    "validation": val_d,
                    "scorer": scorer_d,
                    "split": split_d,
                    "role": "target",
                }
            ],
            "smoke_suite": smoke_digest,
            "parity_suite": parity_d,
        }
        program_doc = {
            "ir_version": 1,
            "opset_version": 1,
            "base_model": base_model_ref,
            "objective": {
                "contract": contract_d,
                "objective": objective_d,
                "experiment_protocol": exp_d,
                "search_space": space_digest,
            },
            "operations": [
                {
                    "id": "studio_matrix_op",
                    "depends_on": [],
                    "operation": op_spec,
                }
            ],
            "artifacts": artifacts,
            "preconditions": [
                {
                    "tensor": "w",
                    "role": "parameter",
                    "shape": [out_dim, in_dim],
                    "dtype": "f64",
                    "before_digest": w_before_digest,
                    "storage_group": "w",
                }
            ],
            "required_capabilities": ["dense.cpu"],
            "constraints": {
                "gradient_free": True,
                "preserve_tensor_shapes": True,
                "max_changed_parameters": out_dim * in_dim,
                "max_artifact_bytes": max_art_bytes,
                "max_temporary_bytes": 16_777_216,
                "numerical": {
                    "accumulation_dtype": "f64",
                    "output_rounding": "nearest_ties_even",
                    "non_finite": "reject",
                    "max_abs_parameter": 10_000.0,
                    "backend_abs_tolerance": 0.0,
                },
            },
            "effects": {
                "reads": [region],
                "writes": [region],
                "buffer_writes": [],
                "graph_changes": False,
                "config_changes": [],
                "tokenizer_changes": False,
            },
            "preservation": {"type": "experimental"},
            "verification": verification_block,
            "lineage": {
                "parent_candidates": [],
                "hypothesis": hyp_d,
                "method": {
                    "schema_version": 1,
                    "target_update": "direct_tensor_edit",
                    "proposal_origin": "scripted",
                    "adaptation": "none",
                    "production_history": "direct_only",
                    "inspection_uses_backprop": False,
                    "parameter_update_uses_backprop": False,
                    "gradient_derived_update_values": False,
                    "alignment_fitted": False,
                    "optimizer_steps": 0,
                    "changed_parameters": out_dim * in_dim,
                },
            },
            "extensions": {},
        }
        policy_doc = {
            "schema_version": 1,
            "source": base_model_ref,
            "contract": contract_d,
            "objective": objective_d,
            "experiment_protocol": exp_d,
            "search_space": space_digest,
            "verification": verification_block,
            "allowed_extra_gyms": [],
            "permissions": {
                "schema_version": 1,
                "tools": ["inspect", "direct_edit"],
                "external_destinations": {},
                "allow_gradient_inspection": False,
                "allow_parameter_training": False,
                "require_fully_gradient_free": True,
                "allow_generated_gym": False,
                "allow_downloads": False,
            },
            "operand_access": ["development"],
            "backend": {
                "id": "nf.native_cpu",
                "version": "1",
                "operations": [operation_type],
                "capabilities": ["dense.cpu"],
                "dtypes": ["f64"],
                "accumulation_dtypes": ["f64"],
                "fixed_scratch_bytes": 4096,
            },
            "bounds": {
                "max_metadata_bytes": 1_048_576,
                "max_source_checkpoint_bytes": 16_777_216,
                "max_source_asset_bytes": 65_536,
                "max_source_tensors": 16,
                "max_observed_source_bytes": 16_777_216,
                "max_temporary_bytes": 16_777_216,
                "max_output_tensor_bytes": 16_777_216,
                "max_validation_work": 10_000_000,
                "max_kernel_work": 10_000_000,
                "max_parameter_magnitude": 10_000.0,
                "max_backend_abs_tolerance": 0.0,
                "max_orthonormal_tolerance": 0.000001,
            },
        }

        plan_handle = compile(
            source_handle,
            program_doc,
            search_space_doc,
            policy_doc,
            operands,
        )
        candidate_handle = surgeon.apply(plan_handle, smoke_suite)
        receipt_dict: dict[str, Any] = json.loads(candidate_handle.receipt_json().decode("utf-8"))
        reloaded = candidate_handle.snapshot()
        reload_ok = reloaded.parameter_digest == candidate_handle.parameter_digest
        edited_bytes = candidate_handle.tensor_bytes("w")

        return CompiledTransactionReceipt(
            operation_type=operation_type,
            shape=(out_dim, in_dim),
            edited_bytes=edited_bytes,
            plan_digest=plan_handle.digest,
            candidate_digest=candidate_handle.digest,
            before_parameter_digest=w_before_digest,
            after_parameter_digest=candidate_handle.parameter_digest,
            changed_parameters=int(receipt_dict.get("changed_parameters", out_dim * in_dim)),
            reload_parity_verified=reload_ok,
            receipt=receipt_dict,
        )

