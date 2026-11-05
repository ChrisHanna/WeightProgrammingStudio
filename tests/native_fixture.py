"""Small real checkpoint fixture; hashes here are test data, not SDK logic."""

import hashlib
import json
import struct
from dataclasses import dataclass
from typing import Any

from neural_foundry import _native


def wire(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


@dataclass
class NativeFixture:
    manifest: dict[str, Any]
    checkpoint: bytes
    assets: dict[str, bytes]
    program: dict[str, Any]
    space: dict[str, Any]
    policy: dict[str, Any]
    smoke: dict[str, Any]
    source: _native.SourceHandle


def make_fixture(dtype: str = "f64") -> NativeFixture:
    assert dtype in ("f32", "f64")
    data = struct.pack("<4d" if dtype == "f64" else "<4f", 1, 2, 3, 4)
    header = wire({"w": {"dtype": dtype.upper(), "shape": [2, 2], "data_offsets": [0, len(data)]}})
    header += b" " * (-len(header) % 8)
    checkpoint = struct.pack("<Q", len(header)) + header + data
    config = digest(b"{}")
    specs = {
        name: {"dtype": dtype, "shape": [2, 2], "digest": digest(data), "storage_group": "weights"}
        for name in ("tied", "w")
    }
    parameter_identity = {
        "schema_version": 1,
        "kind": "tensor_set",
        "tensors": [
            {"name": name, "shape": [2, 2], "dtype": dtype, "content_digest": digest(data)}
            for name in specs
        ],
    }
    manifest = {
        "schema_version": 1,
        "architecture": {"id": "fixture.cpu", "version": "1", "config": config},
        "parameters": specs,
        "buffers": {},
        "parameter_digest": digest(wire(parameter_identity)),
        "format": "dense_safetensors",
        "packing_policy": None,
        "assets": {"config": config, "checkpoint": digest(checkpoint)},
        "usage_rights": "synthetic test fixture",
    }
    assets = {"config": b"{}"}
    source = _native.load_source(wire(manifest), checkpoint, assets)
    base = {
        "checkpoint": digest(checkpoint),
        "manifest": source.manifest_digest,
        "parameter_digest": source.parameter_digest,
        "architecture_graph": digest(b"graph"),
        "config": config,
        "tokenizer": None,
    }
    smoke = {
        "schema_version": 1,
        "adapter": "nf.linear_fixture.v1",
        "probes": [
            {
                "input": [1.0, 2.0],
                "expected": None,
                "absolute_tolerance": 0.0,
                "max_abs_output": 1000.0,
            }
        ],
    }
    smoke_id = _native.validate_record("smoke_suite", wire(smoke)).digest
    verification = {
        "portfolio": digest(b"portfolio"),
        "protocol": digest(b"verification"),
        "gyms": [
            {
                "binding": digest(b"binding"),
                "gym": digest(b"gym"),
                "validation": digest(b"validation"),
                "scorer": digest(b"scorer"),
                "split": digest(b"selection"),
                "role": "target",
            }
        ],
        "smoke_suite": smoke_id,
        "parity_suite": digest(b"parity"),
    }
    space = {
        "schema_version": 1,
        "base_model": base,
        "contract": digest(b"contract"),
        "objective": digest(b"objective"),
        "transformations": {
            "scale": {
                "dtypes": [dtype],
                "coefficient": {"min": -4.0, "max": 4.0},
                "rank": None,
                "sides": [],
                "max_operand_bytes": 0,
            }
        },
        "sites": [
            {
                "selector": {"type": "tensor", "name": name},
                "role": "parameter",
                "regions": {"type": "whole_tensor"},
                "allowed_axes": [0, 1],
            }
            for name in specs
        ],
        "constraints": {
            "max_changed_parameters": 4,
            "max_artifact_bytes": 0,
            "max_temporary_bytes": 4096,
            "max_operations": 8,
            "allow_buffer_writes": False,
            "require_direct_weight_edits": True,
            "require_fully_gradient_free": True,
            "require_preserve_shapes": True,
            "quality_limits": [
                {
                    "binding": digest(b"binding"),
                    "metric": "accuracy",
                    "units": "fraction",
                    "baseline": digest(checkpoint),
                    "max_regression": 0.01,
                    "split": "holdout",
                    "uncertainty_protocol": digest(b"uncertainty"),
                }
            ],
        },
        "strategy": {
            "proposal": "causal_hypothesis",
            "search": "branch",
            "selection": "pareto",
            "engine": "bounded.branch",
            "engine_version": "1",
            "seed": 17,
        },
    }
    space_id = _native.validate_record("search_space", wire(space)).digest
    regions = [{"tensor": name, "start": [0, 0], "shape": [2, 2]} for name in specs]
    numerical = {
        "accumulation_dtype": dtype,
        "output_rounding": "nearest_ties_even",
        "non_finite": "reject",
        "max_abs_parameter": 100.0,
        "backend_abs_tolerance": 0.0,
    }
    program = {
        "ir_version": 1,
        "opset_version": 1,
        "base_model": base,
        "objective": {
            "contract": digest(b"contract"),
            "objective": digest(b"objective"),
            "experiment_protocol": digest(b"experiment"),
            "search_space": space_id,
        },
        "operations": [
            {
                "id": "op0",
                "depends_on": [],
                "operation": {
                    "type": "scale",
                    "target": {"tensor": "w", "start": [0, 0], "shape": [2, 2]},
                    "factor": 2.0,
                },
            }
        ],
        "artifacts": {},
        "preconditions": [
            {
                "tensor": name,
                "role": "parameter",
                "shape": [2, 2],
                "dtype": dtype,
                "before_digest": spec["digest"],
                "storage_group": "weights",
            }
            for name, spec in specs.items()
        ],
        "required_capabilities": ["dense.cpu"],
        "constraints": {
            "gradient_free": True,
            "preserve_tensor_shapes": True,
            "max_changed_parameters": 4,
            "max_artifact_bytes": 0,
            "max_temporary_bytes": 4096,
            "numerical": numerical,
        },
        "effects": {
            "reads": regions,
            "writes": regions,
            "buffer_writes": [],
            "graph_changes": False,
            "config_changes": [],
            "tokenizer_changes": False,
        },
        "preservation": {"type": "experimental"},
        "verification": verification,
        "lineage": {
            "parent_candidates": [],
            "hypothesis": digest(b"hypothesis"),
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
                "changed_parameters": 4,
            },
        },
        "extensions": {},
    }
    permissions = {
        "schema_version": 1,
        "tools": ["inspect", "direct_edit"],
        "external_destinations": {},
        "allow_gradient_inspection": False,
        "allow_parameter_training": False,
        "require_fully_gradient_free": True,
        "allow_generated_gym": False,
        "allow_downloads": False,
    }
    bounds = {
        "max_metadata_bytes": 1048576,
        "max_source_checkpoint_bytes": 1048576,
        "max_source_asset_bytes": 4096,
        "max_source_tensors": 16,
        "max_observed_source_bytes": 4096,
        "max_temporary_bytes": 4096,
        "max_output_tensor_bytes": 4096,
        "max_validation_work": 100000,
        "max_kernel_work": 100000,
        "max_parameter_magnitude": 100.0,
        "max_backend_abs_tolerance": 0.0,
        "max_orthonormal_tolerance": 1e-6,
    }
    policy = {
        "schema_version": 1,
        "source": base,
        "contract": digest(b"contract"),
        "objective": digest(b"objective"),
        "experiment_protocol": digest(b"experiment"),
        "search_space": space_id,
        "verification": verification,
        "allowed_extra_gyms": [],
        "permissions": permissions,
        "operand_access": ["development"],
        "backend": {
            "id": "nf.native_cpu",
            "version": "1",
            "operations": ["scale"],
            "capabilities": ["dense.cpu"],
            "dtypes": [dtype],
            "accumulation_dtypes": [dtype],
            "fixed_scratch_bytes": 64,
        },
        "bounds": bounds,
    }
    return NativeFixture(manifest, checkpoint, assets, program, space, policy, smoke, source)
