"""T059: declared local pretrained Qwen2 adapter, offline mini-configs, and multi-method pipeline."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import neural_foundry as nf
import pytest
from neural_foundry import ir, surgeon
from neural_foundry.pretrained import (
    QWEN2_FACTORY_ID,
    PretrainedAdapterError,
    _build_safetensors,
    intake_local_qwen2,
    write_mini_qwen2_checkpoint,
)
from neural_foundry.records import canonicalize, content_digest


def _id(label: str) -> str:
    return content_digest(label.encode("utf-8"))


def _qwen2_compile_inputs(
    bundle: Any,
    *,
    operation: dict[str, Any],
    read_regions: list[dict[str, Any]],
    write_regions: list[dict[str, Any]],
    operands_meta: dict[str, Any],
    changed_params_bound: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    smoke = bundle.smoke_suite()
    smoke_digest = content_digest(canonicalize(smoke))
    base_model = {
        "checkpoint": bundle.checkpoint_digest,
        "manifest": bundle.manifest_digest,
        "parameter_digest": bundle.manifest["parameter_digest"],
        "architecture_graph": _id("qwen2.graph.v1"),
        "config": bundle.manifest["architecture"]["config"],
        "tokenizer": bundle.manifest["assets"]["tokenizer"],
    }
    preconditions = [
        {
            "tensor": name,
            "role": "parameter",
            "shape": spec["shape"],
            "dtype": spec["dtype"],
            "before_digest": spec["digest"],
            "storage_group": spec["storage_group"],
        }
        for name, spec in sorted(bundle.manifest["parameters"].items())
    ]
    space = {
        "schema_version": 1,
        "base_model": base_model,
        "contract": _id("contract.qwen2"),
        "objective": _id("objective.qwen2"),
        "transformations": {
            "scale": {
                "dtypes": ["f64"],
                "coefficient": {"min": 0.0, "max": 4.0},
                "rank": None,
                "sides": [],
                "max_operand_bytes": 0,
            },
            "low_rank_update": {
                "dtypes": ["f64"],
                "coefficient": {"min": -2.0, "max": 2.0},
                "rank": {"min": 1, "max": 4},
                "sides": [],
                "max_operand_bytes": 4096,
            },
            "subspace_projection": {
                "dtypes": ["f64"],
                "coefficient": {"min": -2.0, "max": 2.0},
                "rank": {"min": 1, "max": 4},
                "sides": ["left", "right"],
                "max_operand_bytes": 4096,
            },
        },
        "sites": [
            {
                "selector": {"type": "tensor", "name": name},
                "role": "parameter",
                "regions": {"type": "whole_tensor"},
                "allowed_axes": list(range(len(spec["shape"]))),
            }
            for name, spec in sorted(bundle.manifest["parameters"].items())
        ],
        "constraints": {
            "max_changed_parameters": changed_params_bound,
            "max_artifact_bytes": 4096,
            "max_temporary_bytes": 65536,
            "max_operations": 8,
            "allow_buffer_writes": False,
            "require_direct_weight_edits": True,
            "require_fully_gradient_free": True,
            "require_preserve_shapes": True,
            "quality_limits": [
                {
                    "binding": _id("binding.qwen2"),
                    "metric": "accuracy",
                    "units": "fraction",
                    "baseline": bundle.checkpoint_digest,
                    "max_regression": 0.0,
                    "split": "holdout",
                    "uncertainty_protocol": _id("uncertainty.qwen2"),
                }
            ],
        },
        "strategy": {
            "proposal": "scripted",
            "search": "sequential",
            "selection": "pareto",
            "engine": "nf.engine.direct_subspace_repair",
            "engine_version": "1.0.0",
            "seed": 17,
        },
    }
    space_digest = content_digest(canonicalize(space))
    verification = {
        "portfolio": _id("portfolio.qwen2"),
        "protocol": _id("protocol.qwen2"),
        "gyms": [
            {
                "binding": _id("binding.qwen2"),
                "gym": _id("gym.qwen2"),
                "validation": _id("validation.qwen2"),
                "scorer": _id("scorer.qwen2"),
                "split": _id("selection.qwen2"),
                "role": "target",
            }
        ],
        "smoke_suite": smoke_digest,
        "parity_suite": _id("parity.qwen2"),
    }
    program = {
        "ir_version": 1,
        "opset_version": 1,
        "base_model": base_model,
        "objective": {
            "contract": _id("contract.qwen2"),
            "objective": _id("objective.qwen2"),
            "experiment_protocol": _id("experiment.qwen2"),
            "search_space": space_digest,
        },
        "operations": [{"id": "op0", "depends_on": [], "operation": operation}],
        "artifacts": operands_meta,
        "preconditions": preconditions,
        "required_capabilities": ["dense.cpu"],
        "constraints": {
            "gradient_free": True,
            "preserve_tensor_shapes": True,
            "max_changed_parameters": changed_params_bound,
            "max_artifact_bytes": 4096,
            "max_temporary_bytes": 65536,
            "numerical": {
                "accumulation_dtype": "f64",
                "output_rounding": "nearest_ties_even",
                "non_finite": "reject",
                "max_abs_parameter": 100.0,
                "backend_abs_tolerance": 0.0,
            },
        },
        "effects": {
            "reads": read_regions,
            "writes": write_regions,
            "buffer_writes": [],
            "graph_changes": False,
            "config_changes": [],
            "tokenizer_changes": False,
        },
        "preservation": {"type": "experimental"},
        "verification": verification,
        "lineage": {
            "parent_candidates": [],
            "hypothesis": _id("hypothesis.qwen2"),
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
                "changed_parameters": changed_params_bound,
            },
        },
        "extensions": {},
    }
    policy = {
        "schema_version": 1,
        "source": base_model,
        "contract": _id("contract.qwen2"),
        "objective": _id("objective.qwen2"),
        "experiment_protocol": _id("experiment.qwen2"),
        "search_space": space_digest,
        "verification": verification,
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
            "operations": ["scale", "low_rank_update", "subspace_projection"],
            "capabilities": ["dense.cpu"],
            "dtypes": ["f32", "f64"],
            "accumulation_dtypes": ["f64"],
            "fixed_scratch_bytes": 64,
        },
        "bounds": {
            "max_metadata_bytes": 1_048_576,
            "max_source_checkpoint_bytes": 1_048_576,
            "max_source_asset_bytes": 65536,
            "max_source_tensors": 64,
            "max_observed_source_bytes": 65536,
            "max_temporary_bytes": 65536,
            "max_output_tensor_bytes": 65536,
            "max_validation_work": 500_000,
            "max_kernel_work": 500_000,
            "max_parameter_magnitude": 100.0,
            "max_backend_abs_tolerance": 0.0,
            "max_orthonormal_tolerance": 1e-6,
        },
    }
    return program, space, policy, smoke


def test_pretrained_adapter_qwen2_intake_and_two_methods_through_native_pipeline(
    tmp_path: Path,
) -> None:
    ckpt_dir = write_mini_qwen2_checkpoint(tmp_path / "qwen2-mini")
    bundle = intake_local_qwen2(
        ckpt_dir, expected_revision="sha256:" + "a" * 64
    )
    assert bundle.manifest["architecture"]["id"] == QWEN2_FACTORY_ID
    assert bundle.qualification_stratum == "mini_config_offline_fixture"
    assert bundle.tokenizer_provenance["vocab_entries"] == 16
    assert (
        bundle.manifest["parameters"]["model.embed_tokens.weight"]["storage_group"]
        == bundle.manifest["parameters"]["lm_head.weight"]["storage_group"]
        == "shared.embed_lm_head"
    )

    source = bundle.load_native_source()
    target_tensor = "model.layers.0.mlp.down_proj.weight"
    region = {"tensor": target_tensor, "start": [0, 0], "shape": [4, 8]}

    # Method 1: Subspace projection on Qwen2 MLP down_proj
    basis_bytes = _build_safetensors(
        {"basis": ("f64", (4, 1), struct.pack("<4d", 1.0, 0.0, 0.0, 0.0))}
    )
    proj_op = {
        "type": "subspace_projection",
        "target": region,
        "basis": "u_basis",
        "side": "left",
        "strength": 0.5,
        "orthonormal_tolerance": 1e-6,
    }
    proj_operands_meta = {
        "u_basis": {
            "digest": content_digest(basis_bytes),
            "tensor": "basis",
            "dtype": "f64",
            "shape": [4, 1],
            "data_bytes": 32,
            "artifact_bytes": len(basis_bytes),
            "access": "development",
            "provenance": {
                "schema_version": 1,
                "origin": "supplied_fact",
                "support": "untested",
                "provenance": "qwen2_subspace_basis",
                "evidence": [],
                "access": "development",
            },
        }
    }
    prog1, space1, policy1, smoke1 = _qwen2_compile_inputs(
        bundle,
        operation=proj_op,
        read_regions=[region],
        write_regions=[region],
        operands_meta=proj_operands_meta,
        changed_params_bound=32,
    )
    plan1 = ir.compile(source, prog1, space1, policy1, {"u_basis": basis_bytes})
    cand1 = surgeon.apply(plan1, smoke1)
    receipt1 = json.loads(cand1.receipt_json())
    assert receipt1["stage"] == "tensor_scope_and_smoke"
    assert receipt1["changed_parameters"] == 8
    assert receipt1["promotion_status"] == "not_evaluated"
    reloaded1 = cand1.snapshot()
    assert reloaded1.parameter_digest != source.parameter_digest

    # Method 2: Low-rank update on the same Qwen2 checkpoint through the shared native executor
    left_bytes = _build_safetensors(
        {"left": ("f64", (4, 1), struct.pack("<4d", 0.5, -0.5, 0.25, -0.25))}
    )
    right_bytes = _build_safetensors(
        {
            "right": (
                "f64",
                (1, 8),
                struct.pack("<8d", 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2),
            )
        }
    )
    lora_op = {
        "type": "low_rank_update",
        "target": region,
        "left": "l_factor",
        "right": "r_factor",
        "strength": 1.0,
    }
    lora_operands_meta = {
        "l_factor": {
            "digest": content_digest(left_bytes),
            "tensor": "left",
            "dtype": "f64",
            "shape": [4, 1],
            "data_bytes": 32,
            "artifact_bytes": len(left_bytes),
            "access": "development",
            "provenance": {
                "schema_version": 1,
                "origin": "supplied_fact",
                "support": "untested",
                "provenance": "qwen2_low_rank_left",
                "evidence": [],
                "access": "development",
            },
        },
        "r_factor": {
            "digest": content_digest(right_bytes),
            "tensor": "right",
            "dtype": "f64",
            "shape": [1, 8],
            "data_bytes": 64,
            "artifact_bytes": len(right_bytes),
            "access": "development",
            "provenance": {
                "schema_version": 1,
                "origin": "supplied_fact",
                "support": "untested",
                "provenance": "qwen2_low_rank_right",
                "evidence": [],
                "access": "development",
            },
        },
    }
    prog2, space2, policy2, smoke2 = _qwen2_compile_inputs(
        bundle,
        operation=lora_op,
        read_regions=[region],
        write_regions=[region],
        operands_meta=lora_operands_meta,
        changed_params_bound=32,
    )
    plan2 = ir.compile(
        source,
        prog2,
        space2,
        policy2,
        {"l_factor": left_bytes, "r_factor": right_bytes},
    )
    cand2 = surgeon.apply(plan2, smoke2)
    receipt2 = json.loads(cand2.receipt_json())
    assert receipt2["changed_parameters"] == 32
    assert cand2.digest != cand1.digest
    # Replay produces identical candidate digest
    assert surgeon.apply(plan2, smoke2).digest == cand2.digest


def test_pretrained_adapter_rejects_downloads_remote_code_pickle_and_unsupported_variants(
    tmp_path: Path,
) -> None:
    ckpt_dir = write_mini_qwen2_checkpoint(tmp_path / "qwen2-valid")

    # 1. Default-denied downloads and remote hub strings
    with pytest.raises(PretrainedAdapterError) as dl_err:
        intake_local_qwen2(ckpt_dir, allow_download=True)
    assert dl_err.value.code == "DOWNLOAD_DISABLED"

    with pytest.raises(PretrainedAdapterError) as hub_err:
        intake_local_qwen2("Qwen/Qwen2.5-0.5B")
    assert hub_err.value.code == "DOWNLOAD_DISABLED"

    # 2. Default-denied trust_remote_code parameter and config flags
    with pytest.raises(PretrainedAdapterError) as rc_err:
        intake_local_qwen2(ckpt_dir, trust_remote_code=True)
    assert rc_err.value.code == "REMOTE_CODE_FORBIDDEN"

    remote_cfg_dir = write_mini_qwen2_checkpoint(tmp_path / "qwen2-remote-cfg")
    cfg = json.loads((remote_cfg_dir / "config.json").read_bytes())
    cfg["trust_remote_code"] = True
    (remote_cfg_dir / "config.json").write_bytes(canonicalize(cfg))
    with pytest.raises(PretrainedAdapterError) as cfg_rc_err:
        intake_local_qwen2(remote_cfg_dir)
    assert cfg_rc_err.value.code == "REMOTE_CODE_FORBIDDEN"

    # 3. Unsupported MoE / architecture variants
    moe_dir = write_mini_qwen2_checkpoint(tmp_path / "qwen2-moe")
    moe_cfg = json.loads((moe_dir / "config.json").read_bytes())
    moe_cfg["architectures"] = ["Qwen2MoeForCausalLM"]
    (moe_dir / "config.json").write_bytes(canonicalize(moe_cfg))
    with pytest.raises(PretrainedAdapterError) as moe_err:
        intake_local_qwen2(moe_dir)
    assert moe_err.value.code == "UNSUPPORTED_ARCHITECTURE"

    # 4. Unsafe pickle / .bin artifacts rejected
    pickle_dir = write_mini_qwen2_checkpoint(tmp_path / "qwen2-pickle")
    (pickle_dir / "pytorch_model.bin").write_bytes(b"pickle-bytes")
    with pytest.raises(PretrainedAdapterError) as pkl_err:
        intake_local_qwen2(pickle_dir)
    assert pkl_err.value.code == "UNSAFE_ARTIFACT"

    # 5. Revision mismatch rejected
    with pytest.raises(PretrainedAdapterError) as rev_err:
        intake_local_qwen2(ckpt_dir, expected_revision="sha256:" + "f" * 64)
    assert rev_err.value.code == "REVISION_MISMATCH"

    # 6. Tied-embedding alias scope enforcement in native compiler
    bundle = intake_local_qwen2(ckpt_dir)
    source = bundle.load_native_source()
    only_embed_region = {
        "tensor": "model.embed_tokens.weight",
        "start": [0, 0],
        "shape": [1, 4],
    }
    prog, space, policy, _ = _qwen2_compile_inputs(
        bundle,
        operation={"type": "scale", "target": only_embed_region, "factor": 0.5},
        read_regions=[only_embed_region],
        write_regions=[only_embed_region],
        operands_meta={},
        changed_params_bound=8,
    )
    with pytest.raises(nf.NativeError) as alias_err:
        ir.compile(source, prog, space, policy)
    assert alias_err.value.code == "ALIAS_SCOPE_ESCAPE"
