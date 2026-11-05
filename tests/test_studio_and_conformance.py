"""Tests for WeightProgrammingStudio PyO3/CLI integration and T120 Full Release Conformance."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from neural_foundry.records import content_digest
from neural_foundry.studio import (
    WeightProgrammingStudio,
    generate_release_conformance_report,
)


def test_studio_rust_native_shape_solver_and_structural_lowering_and_cli(tmp_path: Path) -> None:
    """Verifies PyO3 `_native` and `weightctl studio` CLI bindings for M4/M5 Rust engines."""
    studio = WeightProgrammingStudio()

    # 1. Solve architecture shape via Rust PyO3
    graph = {
        "family_id": "hf.qwen2.dense",
        "symbols": {
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_q_heads": 4,
            "num_kv_heads": 2,
            "head_dim": 4,
        },
        "ports": [
            {
                "tensor_name": "model.embed_tokens.weight",
                "dims": [
                    {"type": "symbol", "name": "vocab_size"},
                    {"type": "symbol", "name": "hidden_size"},
                ],
                "alias_group": "tied_embed",
            },
            {
                "tensor_name": "lm_head.weight",
                "dims": [
                    {"type": "symbol", "name": "vocab_size"},
                    {"type": "symbol", "name": "hidden_size"},
                ],
                "alias_group": "tied_embed",
            },
        ],
        "constraints": [
            {
                "kind": "tied_weight_alias",
                "source_tensor": "model.embed_tokens.weight",
                "target_tensor": "lm_head.weight",
            },
            {
                "kind": "attention_group_divisibility",
                "num_q_heads_symbol": "num_q_heads",
                "num_kv_heads_symbol": "num_kv_heads",
                "head_dim_symbol": "head_dim",
                "hidden_size_symbol": "hidden_size",
            },
        ],
    }
    solved = studio.solve_architecture_shape(graph)
    assert solved["kv_group_ratio"] == 2
    assert solved["total_parameter_elements"] == 64 * 16  # Tied alias counted once!

    # 2. Lower structural IR via Rust PyO3
    lower_req = {
        "operation": {
            "op": "slice_mlp_intermediate",
            "layer_index": 0,
            "keep_channels": [0, 2, 4, 6],
            "original_intermediate_size": 8,
        },
        "claimed_preservation": {"type": "experimental"},
        "search_space_digest": str(content_digest(b"space-structural")),
        "space_allows_shape_changes": True,
        "hidden_size": 16,
    }
    lowered = studio.lower_structural_ir(lower_req)
    assert lowered["changes_tensor_shapes"] is True
    assert lowered["config_updates"]["intermediate_size"] == 4
    assert len(lowered["rewrites"]) == 3

    # 3. Verify `weightctl studio solve-shape` CLI command
    graph_file = tmp_path / "graph.json"
    graph_file.write_text(json.dumps(graph), encoding="utf-8")
    weightctl_bin = Path("/tmp/nf-t043-joined/debug/weightctl")
    if weightctl_bin.exists():
        proc = subprocess.run(
            [str(weightctl_bin), "studio", "solve-shape", str(graph_file)],
            capture_output=True,
            text=True,
            check=True,
        )
        cli_out = json.loads(proc.stdout)
        assert cli_out["ok"] is True
        assert cli_out["result"]["kv_group_ratio"] == 2


def test_release_conformance_and_capability_reporting_t120() -> None:
    """T120 (`release_conformance`): generates truthful milestone, capability-gate, and hardware status from evidence."""
    report = generate_release_conformance_report()
    assert report.schema_version == 1
    assert len(report.workspace_crates) == 9
    assert report.derived_from_task_checkbox_count is False
    assert all(report.capability_gates.values())
    assert report.milestones["M1"].live_scientific_or_hardware_claim_status == "verified_on_fixture_scope"
    assert report.milestones["M2"].live_scientific_or_hardware_claim_status == "verified_on_fixture_scope"
    assert (
        report.milestones["M3"].live_scientific_or_hardware_claim_status
        == "requires_live_provider_or_unseen_model_study"
    )
    assert (
        report.milestones["M4"].live_scientific_or_hardware_claim_status
        == "requires_held_out_cross_model_study"
    )

    with pytest.raises(ValueError, match="T120 violation"):
        generate_release_conformance_report(derive_scientific_claims_from_checkbox_count=True)


def test_studio_reusable_gym_across_user_selected_models() -> None:
    """Verifies that the same reusable Gym & solver components work across user-selected Pythia and Qwen2 models."""
    studio = WeightProgrammingStudio()
    catalog = studio.list_selectable_models()
    assert len(catalog) >= 4
    reusable_gyms = studio.list_reusable_gyms()
    assert "algorithmic_three_skill" in reusable_gyms
    assert "factual_knowledge" in reusable_gyms

    for entry in catalog:
        if not entry["available"]:
            continue
        model_id = str(entry["model_id"])
        rep = studio.run_reusable_gym_study(
            model_id,
            gym="algorithmic_three_skill",
            target_strength=80.0,
        )
        assert rep.model_name == model_id
        assert rep.reload_parity_verified is True
        assert rep.branches[2].gym_dev_accuracy == 1.0
        assert rep.branches[2].gym_prefix_val_accuracy == 1.0
        assert rep.branches[2].retention_accuracy == 1.0
        assert rep.promotion_verification["promote_eligible"] is True
        assert rep.promotion_verification["status"] == "passed"

