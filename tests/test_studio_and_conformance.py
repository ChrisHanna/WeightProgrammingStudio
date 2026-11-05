"""Tests for WeightProgrammingStudio PyO3/CLI integration and T120 Full Release Conformance."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import torch
from neural_foundry.pythia import ArchitectureSiteProfile, build_mini_pythia_model
from neural_foundry.records import content_digest
from neural_foundry.studio import (
    WeightProgrammingStudio,
    generate_release_conformance_report,
)
from neural_foundry.studio import (
    main as studio_cli_main,
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


def test_studio_extensible_registries_ir_builder_and_cli() -> None:
    """Verifies that users can register custom models, architectures, gyms, methods, and agents into WeightProgrammingStudio."""
    studio = WeightProgrammingStudio()

    # 1. Register a custom architecture profile and custom model loader
    studio.models.register_architecture(
        ArchitectureSiteProfile(
            family_id="custom_decoder",
            mlp_down_proj_template="decoder.blocks.{layer}.ffn.w2.weight",
            mlp_up_proj_template="decoder.blocks.{layer}.ffn.w1.weight",
            attn_out_template="decoder.blocks.{layer}.attn.wo.weight",
            embed_in_tensor="decoder.tok_embeddings.weight",
            unembed_out_tensor="decoder.output.weight",
            final_norm_weight_tensor="decoder.norm.weight",
        )
    )
    assert "custom_decoder" in studio.models.list_architectures()

    studio.models.register_model(
        "org/custom-pythia-model",
        loader=build_mini_pythia_model,
        architecture="GPTNeoXForCausalLM",
    )
    loaded = studio.select_model("org/custom-pythia-model")
    assert loaded.total_parameters > 0

    # 2. Register a custom Task Gym from prompt-target rules
    custom_gym = studio.gyms.from_rules(
        "gym.custom.boolean_logic.v1",
        skills_taught=("boolean_logic",),
        base_rules=[
            ("Boolean True AND False gives", "ĠFalse"),
            ("Boolean True OR False gives", "ĠTrue"),
            ("Boolean False AND True gives", "ĠFalse"),
            ("Boolean False OR True gives", "ĠTrue"),
        ],
        register_as="custom_boolean",
    )
    assert custom_gym.gym_id == "gym.custom.boolean_logic.v1"
    custom_rep = studio.methods.run(
        "contrastive_subspace",
        "org/custom-pythia-model",
        gym="custom_boolean",
        target_strength=80.0,
    )
    assert custom_rep.reload_parity_verified is True
    assert custom_rep.branches[2].gym_dev_accuracy == 1.0

    # 3. Register a custom weight-editing method using studio.ir_builder (WeightIRTransactionBuilder)
    @studio.methods.register("spectral_dampener", category="custom_repair", description="Rank-1 singular mode projection")
    def _spectral_dampener(weight_matrix: torch.Tensor, *, damping: float = 0.5) -> Any:
        u, _, _ = torch.linalg.svd(weight_matrix, full_matrices=False)
        q_top1 = u[:, :1].contiguous()
        return studio.ir_builder.execute_matrix_edit(
            weight_matrix,
            operation_type="subspace_projection",
            orthonormal_basis=q_top1,
            projection_side="left",
            factor=damping,
            provenance_tag="custom.spectral_dampener.v1",
        )

    tx_receipt = studio.methods.run("spectral_dampener", torch.eye(4, dtype=torch.float64), damping=0.5)
    assert tx_receipt.reload_parity_verified is True
    assert tx_receipt.before_parameter_digest != tx_receipt.after_parameter_digest

    # 4. Register a custom Frontier Agent
    studio.agents.register_agent(
        "local-llama-scientist",
        extra_tool_schemas=[{"name": "custom_circuit_probe", "version": "1.0.0", "input_schema": {"type": "object"}}],
    )
    custom_agent = studio.create_frontier_agent("local-llama-scientist")
    assert custom_agent.agent_id == "local-llama-scientist"
    assert len(custom_agent.tool_schemas) == 6

    # 5. Verify CLI `catalog` entrypoint
    assert studio_cli_main(["catalog"]) == 0


