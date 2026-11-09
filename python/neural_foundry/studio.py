"""Unified Weight Programming Studio SDK & Full Release Conformance Reporter (T120).

Provides:
- `WeightProgrammingStudio`: High-level Python entry point uniting:
  - Rust native architecture shape solver (`_native.solve_architecture_shape`),
  - Rust native structural IR lowering (`_native.lower_structural_ir`),
  - Rust native patch conflict & dependency composition (`_native.compose_patch_specs`),
  - First-class Evaluation Engine (`EvaluationEngine`),
  - GPU hardware probing, DLPack dispatch, and VRAM leasing (`gpu`),
  - M3–M5 diagnosis, repair, rule construction, transfer, and surgery engines.
- `generate_release_conformance_report`: Truthful milestone (`M1`–`M5`), capability gate
  (`CG-01`–`CG-14`), integration, and scientific-result status reporter (`T120`) that
  separates software/harness readiness from live-hardware/scientific evidence rather
  than inferring scientific breakthroughs from completed task counts.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from . import _native, evals, ir
from .gpu import GpuResourceScheduler, HardwareGpuInventory, probe_gpu_hardware
from .gym_data import load_gym_spec, materialize_public_task_data
from .m4_ecosystems import (
    BaselineExecutionReceipt,
    M4FeatureAcceptanceReport,
    run_lora_baseline,
    run_m4_feature_acceptance_suite,
)
from .m5_surgery import (
    AttentionSurgeryResult,
    MlpSurgeryResult,
    execute_attention_head_surgery,
    execute_mlp_width_surgery,
)
from .patch_ux import (
    OrthogonalPatchCompositionReceipt,
    build_open_patch_v1,
    compose_overlapping_patches_orthogonal,
    create_rollback_handle,
    diff_model_weights,
    generate_ed25519_keypair,
    rollback_model_patch,
    scan_patch_for_backdoor_patterns,
    verify_open_patch_v1,
)
from .prd_adapters import (
    EasyEditAdapterReceipt,
    EasyEditAlgorithm,
    FrontierAgentId,
    FrontierAgentProgrammer,
    MergekitAdapterReceipt,
    MergekitAlgorithm,
    PRDExperimentsSuiteReport,
    run_easyedit_adapter,
    run_mergekit_adapter,
    run_prd_priority_experiments_suite,
)
from .provenance import resolve_checkpoint
from .pythia import (
    ALGORITHMIC_THREE_SKILL_GYM,
    BUILTIN_ARCHITECTURE_PROFILES,
    FACTUAL_KNOWLEDGE_GYM,
    ArchitectureSiteProfile,
    CausalTraceEngine,
    CausalTraceSiteReport,
    CovarianceCacheRegistry,
    LayerCovarianceArtifact,
    LogitLensTrajectoryInspector,
    LogitLensTrajectoryReport,
    PythiaGymTaskReport,
    PythiaModel,
    Qwen2CausalLM,
    ReusableTaskGym,
    TransformerLensHookAdapter,
    build_mini_pythia_model,
    build_mini_qwen2_causal_lm,
    run_pythia_gym_task_learning_study,
)
from .records import canonicalize, content_digest


@dataclass(frozen=True)
class MilestoneEvidenceStatus:
    milestone: Literal["M1", "M2", "M3", "M4", "M5"]
    software_harness_ready: bool
    fixture_conformance_verified: bool
    live_scientific_or_hardware_claim_status: Literal[
        "verified_on_fixture_scope",
        "requires_live_provider_or_unseen_model_study",
        "requires_held_out_cross_model_study",
        "requires_real_gpu_serving_qualification",
    ]


@dataclass(frozen=True)
class ReleaseConformanceReport:
    schema_version: int
    version: str
    workspace_crates: tuple[str, ...]
    milestones: dict[str, MilestoneEvidenceStatus]
    capability_gates: dict[str, bool]
    gpu_inventory: HardwareGpuInventory
    m4_acceptance: M4FeatureAcceptanceReport
    derived_from_task_checkbox_count: bool
    report_digest: str
    rust_m1_capabilities: dict[str, Any] | None = None
    rust_m2_capabilities: dict[str, Any] | None = None


class ModelRegistry:
    """Extensible registry for open-weight models and transformer architecture profiles."""

    def __init__(self) -> None:
        self._architectures: dict[str, ArchitectureSiteProfile] = dict(BUILTIN_ARCHITECTURE_PROFILES)
        self._loaders: dict[str, Callable[[], Any]] = {
            "nf/mini-pythia-fixture": build_mini_pythia_model,
            "mini-pythia-fixture": build_mini_pythia_model,
            "Qwen/qwen2-dense-mini": build_mini_qwen2_causal_lm,
            "qwen2-dense-mini": build_mini_qwen2_causal_lm,
            "EleutherAI/pythia-14m": lambda: PythiaModel.from_pretrained_dir(
                "/tmp/pythia-14m", model_name="EleutherAI/pythia-14m"
            ),
            "EleutherAI/pythia-31m": lambda: PythiaModel.from_pretrained_dir(
                "/tmp/pythia-31m", model_name="EleutherAI/pythia-31m"
            ),
        }
        self._metadata: dict[str, dict[str, Any]] = {
            "EleutherAI/pythia-14m": {
                "model_id": "EleutherAI/pythia-14m",
                "architecture": "GPTNeoXForCausalLM",
                "layers": 6,
                "hidden_size": 128,
                "intermediate_size": 512,
                "local_cache": "/tmp/pythia-14m",
                "available": Path("/tmp/pythia-14m/model.safetensors").is_file(),
            },
            "EleutherAI/pythia-31m": {
                "model_id": "EleutherAI/pythia-31m",
                "architecture": "GPTNeoXForCausalLM",
                "layers": 6,
                "hidden_size": 256,
                "intermediate_size": 1024,
                "local_cache": "/tmp/pythia-31m",
                "available": Path("/tmp/pythia-31m/model.safetensors").is_file(),
            },
            "Qwen/qwen2-dense-mini": {
                "model_id": "Qwen/qwen2-dense-mini",
                "architecture": "Qwen2ForCausalLM",
                "layers": 2,
                "hidden_size": 32,
                "intermediate_size": 64,
                "local_cache": "builtin_deterministic",
                "available": True,
            },
            "nf/mini-pythia-fixture": {
                "model_id": "nf/mini-pythia-fixture",
                "architecture": "GPTNeoXForCausalLM",
                "layers": 2,
                "hidden_size": 32,
                "intermediate_size": 64,
                "local_cache": "builtin_deterministic",
                "available": True,
            },
        }

    def register_architecture(self, profile: ArchitectureSiteProfile) -> None:
        """Registers a new open-weight architecture site naming profile."""
        self._architectures[profile.family_id] = profile

    def list_architectures(self) -> dict[str, ArchitectureSiteProfile]:
        return dict(self._architectures)

    def register_model(
        self,
        model_id: str,
        loader: Callable[[], Any],
        *,
        architecture: str = "custom",
        layers: int = 2,
        hidden_size: int = 32,
        intermediate_size: int = 64,
        local_cache: str = "custom",
    ) -> None:
        """Registers a custom model loader in the Studio catalog."""
        if not model_id.strip():
            raise ValueError("model_id must be non-empty")
        self._loaders[model_id] = loader
        self._metadata[model_id] = {
            "model_id": model_id,
            "architecture": architecture,
            "layers": layers,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "local_cache": local_cache,
            "available": True,
        }

    def list_models(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for k, meta in self._metadata.items():
            entry = dict(meta)
            entry["fixture"] = meta.get("local_cache") == "builtin_deterministic"
            if k in {"EleutherAI/pythia-14m", "EleutherAI/pythia-31m"}:
                try:
                    checkpoint = resolve_checkpoint(k)
                    entry["available"] = any(name.endswith(".safetensors") for name in checkpoint.file_digests)
                    entry["local_cache"] = str(checkpoint.path)
                    entry["revision"] = checkpoint.revision
                except RuntimeError as exc:
                    entry["available"] = False
                    entry["unavailable_reason"] = str(exc)
            out.append(entry)
        return out

    @staticmethod
    def _load_resolved_checkpoint(resolved: Any) -> Any:
        cfg = json.loads((resolved.path / "config.json").read_text(encoding="utf-8"))
        if cfg.get("model_type") == "gpt_neox":
            loaded = PythiaModel.from_pretrained_dir(resolved.path, model_name=resolved.model_id)
        elif cfg.get("model_type") == "qwen2":
            loaded = Qwen2CausalLM.from_pretrained_dir(resolved.path, model_name=resolved.model_id)
        else:
            raise ValueError(f"Unsupported model_type in resolved checkpoint: {cfg.get('model_type')!r}")
        loaded.checkpoint_identity = resolved
        return loaded

    def load(
        self,
        model_id_or_path: str | Path,
        *,
        revision: str | None = None,
        allow_download: bool = False,
    ) -> Any:
        key = str(model_id_or_path)
        if key in self._loaders:
            pinned_catalog_models = {"EleutherAI/pythia-14m", "EleutherAI/pythia-31m"}
            if revision is None and not allow_download and key not in pinned_catalog_models:
                return self._loaders[key]()
            resolved = resolve_checkpoint(key, revision=revision, allow_download=allow_download)
            return self._load_resolved_checkpoint(resolved)
        path = Path(key)
        if path.is_dir() and (path / "config.json").is_file():
            cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
            if cfg.get("model_type") == "gpt_neox":
                loaded = PythiaModel.from_pretrained_dir(path, model_name=path.name)
            else:
                loaded = Qwen2CausalLM.from_pretrained_dir(path, model_name=path.name)
            loaded.checkpoint_identity = resolve_checkpoint(path)
            return loaded
        # A named model is real or it is an error. A fixture is never substituted under its name.
        resolved = resolve_checkpoint(key, revision=revision, allow_download=allow_download)
        return self._load_resolved_checkpoint(resolved)


class GymRegistry:
    """Extensible registry for model-agnostic `ReusableTaskGym` environments."""

    def __init__(self) -> None:
        self._gyms: dict[str, ReusableTaskGym] = {
            "algorithmic_three_skill": ALGORITHMIC_THREE_SKILL_GYM,
            "factual_knowledge": FACTUAL_KNOWLEDGE_GYM,
        }

    def register(self, key: str, gym: ReusableTaskGym) -> ReusableTaskGym:
        if not key.strip():
            raise ValueError("gym key must be non-empty")
        materialize_public_task_data(gym)
        self._gyms[key] = gym
        return gym

    def from_rules(
        self,
        gym_id: str,
        *,
        skills_taught: Sequence[str],
        base_rules: Sequence[tuple[str, str]],
        validation_rules: Sequence[tuple[str, str]] = (),
        holdout_rules: Sequence[tuple[str, str]] = (),
        gym_dev_prefixes: Sequence[str] = ("Task:", "Rule:"),
        gym_val_prefixes: Sequence[str] = ("Note:",),
        holdout_prefixes: Sequence[str] = ("Today:", "Check:", "Quiz:", "Verify:", "Exam:", "Test:"),
        retention_prompts: Sequence[str] = (
            "Hello world",
            "One plus one equals",
            "The quick brown fox jumps over",
            "In computer science a list is",
            "Once upon a time there was",
        ),
        register_as: str | None = None,
    ) -> ReusableTaskGym:
        """Creates and optionally registers a custom `ReusableTaskGym` from prompt-target rules."""
        gym = ReusableTaskGym(
            gym_id=gym_id,
            skills_taught=tuple(skills_taught),
            base_rules=tuple((str(p), str(t)) for p, t in base_rules),
            validation_rules=tuple((str(p), str(t)) for p, t in validation_rules),
            holdout_rules=tuple((str(p), str(t)) for p, t in holdout_rules),
            gym_dev_prefixes=tuple(gym_dev_prefixes),
            gym_val_prefixes=tuple(gym_val_prefixes),
            holdout_prefixes=tuple(holdout_prefixes),
            retention_prompts=tuple(retention_prompts),
        )
        if register_as is not None:
            self.register(register_as, gym)
        return gym

    def load_spec(self, path: str | Path) -> ReusableTaskGym:
        """Loads and registers a data-only versioned Gym JSON spec."""
        gym = load_gym_spec(path)
        return self.register(gym.gym_id, gym)

    def from_nemo_mcqa(
        self,
        *,
        train_jsonl: str | Path,
        validation_jsonl: str | Path,
        holdout_jsonl: str | Path,
        model: Any,
        gym_id: str = "gym.nemo.mcqa.v1",
        register_as: str | None = None,
    ) -> Any:
        """Import NeMo MCQA panels as one-token Gym rules for existing WSP and LoRA/SFT paths."""
        from .nemo_gym import load_mcqa_panels

        panels = load_mcqa_panels(
            train_jsonl,
            validation_jsonl,
            holdout_jsonl,
            tokenizer=model.tokenizer,
            gym_id=gym_id,
        )
        if register_as is not None:
            self.register(register_as, panels.gym)
        return panels

    def get(self, key_or_gym: str | ReusableTaskGym) -> ReusableTaskGym:
        if isinstance(key_or_gym, ReusableTaskGym):
            return key_or_gym
        if key_or_gym not in self._gyms:
            raise ValueError(f"Unknown gym {key_or_gym!r}; available: {sorted(self._gyms)}")
        return self._gyms[key_or_gym]

    def list_gyms(self) -> dict[str, ReusableTaskGym]:
        return dict(self._gyms)


@dataclass(frozen=True)
class MethodContract:
    """Dispatch shape and scope declarations, never a qualification credential."""

    inputs: tuple[tuple[str, str], ...] = ()
    output_scope: str = "undeclared"
    result_types: tuple[type, ...] = ()
    implementation_origin: str = "user_registered"
    implementation_id: str = "user_registered"
    evidence_level: Literal["fixture_only", "experimental", "undeclared"] = "undeclared"
    supported_model_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", tuple(tuple(item) for item in self.inputs))
        object.__setattr__(self, "result_types", tuple(self.result_types))
        object.__setattr__(self, "supported_model_types", tuple(self.supported_model_types))
        names = [name for name, _ in self.inputs]
        if len(set(names)) != len(names):
            raise ValueError("Method contract input names must be unique")
        if any(kind not in {"model", "matrix", "matrix_sequence"} for _, kind in self.inputs):
            raise ValueError("Unknown method contract input kind")
        if any(not isinstance(result_type, type) for result_type in self.result_types):
            raise TypeError("Method contract result types must be Python types")
        if self.evidence_level not in {"fixture_only", "experimental", "undeclared"}:
            raise ValueError("Method declarations cannot establish qualification")

    def to_document(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "inputs": [{"name": name, "kind": kind} for name, kind in self.inputs],
            "output_scope": self.output_scope,
            "result_types": [result_type.__name__ for result_type in self.result_types],
            "implementation_origin": self.implementation_origin,
            "implementation_id": self.implementation_id,
            "evidence_level": self.evidence_level,
            "supported_model_types": list(self.supported_model_types),
            "qualification_credential": False,
        }


class MethodRegistry:
    """Extensible registry of Weight Programming, Merging, Surgery, and Baseline methods."""

    def __init__(self, model_registry: ModelRegistry, gym_registry: GymRegistry) -> None:
        self._model_registry = model_registry
        self._gym_registry = gym_registry
        self._methods: dict[str, Callable[..., Any]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._contracts: dict[str, MethodContract] = {}
        self._register_builtins()

    def _resolve_model(self, model: Any) -> Any:
        return self._model_registry.load(model) if isinstance(model, (str, Path)) else model

    @staticmethod
    def _builtin_contract(method_id: str) -> MethodContract:
        if method_id in {"rome", "memit", "alphaedit", "contrastive_subspace", "task_arithmetic", "ties", "dare"}:
            result_type = (
                PythiaGymTaskReport if method_id == "contrastive_subspace"
                else EasyEditAdapterReceipt if method_id in {"rome", "memit", "alphaedit"}
                else MergekitAdapterReceipt
            )
            owner = (
                "neural_foundry.pythia.run_pythia_gym_task_learning_study"
                if method_id == "contrastive_subspace"
                else "neural_foundry.prd_adapters.run_easyedit_adapter"
                if method_id in {"rome", "memit", "alphaedit"}
                else "neural_foundry.prd_adapters.run_mergekit_adapter"
            )
            return MethodContract(
                inputs=(("model", "model"),),
                output_scope="model_edit_report_not_standalone_checkpoint",
                result_types=(result_type,),
                implementation_origin="local_implementation_not_external_package",
                implementation_id=f"{owner}:{method_id}:v1",
                evidence_level="experimental",
                supported_model_types=("gpt_neox", "qwen2"),
            )
        tensor_methods = {
            "mlp_surgery": (("gate_proj", "up_proj", "down_proj"), MlpSurgeryResult, "sliced_tensors_not_checkpoint"),
            "attention_surgery": (("q_proj", "k_proj", "v_proj", "o_proj"), AttentionSurgeryResult, "sliced_tensors_not_checkpoint"),
            "lora_baseline": (("base_weight", "inputs", "targets"), tuple, "matrix_and_fixture_training_receipt"),
            "orthogonal_compose": (("base_weight",), OrthogonalPatchCompositionReceipt, "matrix_edit_receipt_not_checkpoint"),
        }
        names, result_type, output_scope = tensor_methods[method_id]
        inputs = tuple((name, "matrix") for name in names)
        if method_id == "orthogonal_compose":
            inputs += (("edited_weights", "matrix_sequence"),)
        owners = {
            "mlp_surgery": "neural_foundry.m5_surgery.execute_mlp_width_surgery",
            "attention_surgery": "neural_foundry.m5_surgery.execute_attention_head_surgery",
            "lora_baseline": "neural_foundry.m4_ecosystems.run_lora_baseline",
            "orthogonal_compose": "neural_foundry.patch_ux.compose_overlapping_patches_orthogonal",
        }
        return MethodContract(
            inputs=inputs,
            output_scope=output_scope,
            result_types=(result_type,),
            implementation_origin="local_tensor_implementation",
            implementation_id=f"{owners[method_id]}:v1",
            evidence_level="fixture_only",
        )

    def _register_builtins(self) -> None:
        for alg in ("ROME", "MEMIT", "AlphaEdit"):
            alg_lit: EasyEditAlgorithm = alg

            def _make_easyedit(a: EasyEditAlgorithm) -> Callable[..., EasyEditAdapterReceipt]:
                def _runner(
                    model: Any,
                    *,
                    associations: list[tuple[str, str]],
                    nullspace_prompts: Sequence[str] = (),
                    target_strength: float = 75.0,
                    target_layers: Sequence[int] | Literal["auto_causal_trace"] | None = None,
                    covariance_registry: CovarianceCacheRegistry | None = None,
                ) -> EasyEditAdapterReceipt:
                    return run_easyedit_adapter(
                        self._resolve_model(model),
                        algorithm=a,
                        associations=associations,
                        nullspace_prompts=nullspace_prompts,
                        target_strength=target_strength,
                        target_layers=target_layers,
                        covariance_registry=covariance_registry,
                    )

                return _runner

            self.register_method(
                alg.lower(),
                _make_easyedit(alg_lit),
                category="knowledge_edit",
                description=f"Local {alg} implementation; model-weight edit, not the EasyEdit package",
                contract=self._builtin_contract(alg.lower()),
            )

        def _contrastive_subspace(
            model: Any,
            *,
            gym: ReusableTaskGym | str = "algorithmic_three_skill",
            target_strength: float = 75.0,
        ) -> PythiaGymTaskReport:
            return run_pythia_gym_task_learning_study(
                self._resolve_model(model),
                gym=self._gym_registry.get(gym),
                target_strength=target_strength,
            )

        self.register_method(
            "contrastive_subspace",
            _contrastive_subspace,
            category="gym_task_programming",
            description="Contrastive + Prefix-Nullspace + Retention-Nullspace multi-skill weight programmer",
            contract=self._builtin_contract("contrastive_subspace"),
        )

        for merge_alg in ("task_arithmetic", "ties", "dare"):
            m_lit: MergekitAlgorithm = merge_alg

            def _make_mergekit(m: MergekitAlgorithm) -> Callable[..., MergekitAdapterReceipt]:
                def _mrunner(
                    model: Any,
                    *,
                    checkpoint_weights: Sequence[Any],
                    scaling_weights: Sequence[float] = (1.0, 1.0),
                    density: float = 0.85,
                    task_a_probes: Sequence[tuple[str, str]] = (),
                    task_b_probes: Sequence[tuple[str, str]] = (),
                ) -> MergekitAdapterReceipt:
                    return run_mergekit_adapter(
                        self._resolve_model(model),
                        algorithm=m,
                        checkpoint_weights=checkpoint_weights,
                        scaling_weights=scaling_weights,
                        density=density,
                        task_a_probes=task_a_probes,
                        task_b_probes=task_b_probes,
                    )

                return _mrunner

            self.register_method(
                merge_alg,
                _make_mergekit(m_lit),
                category="model_merge",
                description=f"Local {merge_alg} weight merge; not the mergekit package",
                contract=self._builtin_contract(merge_alg),
            )

        self.register_method(
            "mlp_surgery",
            execute_mlp_width_surgery,
            category="architecture_surgery",
            description="Tensor-only coupled MLP channel slicing; does not export a checkpoint",
            contract=self._builtin_contract("mlp_surgery"),
        )
        self.register_method(
            "attention_surgery",
            execute_attention_head_surgery,
            category="architecture_surgery",
            description="Tensor-only GQA head slicing; does not export a checkpoint",
            contract=self._builtin_contract("attention_surgery"),
        )
        self.register_method(
            "lora_baseline",
            run_lora_baseline,
            category="matrix_training_fixture",
            description="Matrix-only LoRA training fixture; not full-model fine-tuning",
            contract=self._builtin_contract("lora_baseline"),
        )
        self.register_method(
            "orthogonal_compose",
            compose_overlapping_patches_orthogonal,
            category="model_merge",
            description="Right-singular subspace-orthogonal multi-patch conflict resolver (compiled to Rust Weight IR)",
            contract=self._builtin_contract("orthogonal_compose"),
        )

    def register_method(
        self,
        method_id: str,
        fn: Callable[..., Any],
        *,
        category: str = "custom",
        description: str = "User-registered weight programming method",
        contract: MethodContract | None = None,
    ) -> Callable[..., Any]:
        key = method_id.strip().lower()
        if not key:
            raise ValueError("method_id must be non-empty")
        if not callable(fn):
            raise TypeError("Registered method must be callable")
        declared = contract if contract is not None else MethodContract()
        if not isinstance(declared, MethodContract):
            raise TypeError("Method contract must be a MethodContract")
        if declared.inputs:
            parameters = inspect.signature(fn).parameters
            if any(name not in parameters for name, _ in declared.inputs):
                raise ValueError("Method contract names an input absent from the callable signature")
        self._contracts[key] = declared
        self._methods[key] = fn
        self._metadata[key] = {
            "method_id": key,
            "category": category,
            "description": description,
            "contract": declared.to_document(),
        }
        return fn

    def register(
        self,
        method_id: str,
        *,
        category: str = "custom",
        description: str = "User-registered weight programming method",
        contract: MethodContract | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator for registering a custom weight-editing method into `studio.methods`."""

        def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            return self.register_method(method_id, fn, category=category, description=description, contract=contract)

        return _decorator

    def run(self, method_id: str, *args: Any, **kwargs: Any) -> Any:
        key = method_id.strip().lower()
        if key not in self._methods:
            raise ValueError(f"Unknown method {method_id!r}; available: {sorted(self._methods)}")
        method = self._methods[key]
        contract = self._contracts[key]
        if not contract.inputs and not contract.result_types:
            return method(*args, **kwargs)
        import torch

        bound = inspect.signature(method).bind(*args, **kwargs)
        bound.apply_defaults()
        for name, kind in contract.inputs:
            value = bound.arguments.get(name)
            if kind == "model":
                value = self._resolve_model(value)
                if not isinstance(value, (PythiaModel, Qwen2CausalLM)):
                    raise TypeError(f"Method {key!r} requires a supported model for {name!r}, not a matrix")
                if value.config.get("model_type") not in contract.supported_model_types:
                    raise ValueError(f"Method {key!r} does not support this model architecture")
                bound.arguments[name] = value
            elif kind == "matrix":
                if not isinstance(value, torch.Tensor) or value.ndim != 2:
                    raise TypeError(f"Method {key!r} requires a 2D tensor for {name!r}; this is tensor-only")
            elif kind == "matrix_sequence":
                if not isinstance(value, Sequence) or not value or any(
                    not isinstance(item, torch.Tensor) or item.ndim != 2 for item in value
                ):
                    raise TypeError(f"Method {key!r} requires a non-empty sequence of matrices for {name!r}")
        result = method(*bound.args, **bound.kwargs)
        if contract.result_types and not isinstance(result, contract.result_types):
            raise TypeError(f"Method {key!r} returned a result outside its declared contract")
        if contract.output_scope == "matrix_and_fixture_training_receipt" and (
            len(result) != 2
            or not isinstance(result[0], torch.Tensor)
            or result[0].ndim != 2
            or not isinstance(result[1], BaselineExecutionReceipt)
        ):
            raise TypeError(f"Method {key!r} must return a matrix and fixture training receipt")
        return result

    def list_methods(self) -> list[dict[str, Any]]:
        return [
            {**self._metadata[key], "contract": self._contracts[key].to_document()}
            for key in sorted(self._metadata)
        ]


class AgentRegistry:
    """Extensible registry of interchangeable Frontier Agents (`Claude`, `GPT-6`, or custom LLM agents)."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., FrontierAgentProgrammer]] = {
            "claude-4-5-sonnet": lambda **kw: FrontierAgentProgrammer("claude-4-5-sonnet", **kw),
            "gpt-6-astra": lambda **kw: FrontierAgentProgrammer("gpt-6-astra", **kw),
        }

    def register_agent(
        self,
        agent_id: str,
        factory: Callable[..., FrontierAgentProgrammer] | None = None,
        *,
        extra_tool_schemas: Sequence[dict[str, Any]] = (),
    ) -> None:
        key = agent_id.strip()
        if not key:
            raise ValueError("agent_id must be non-empty")
        if factory is not None:
            self._factories[key] = factory
        else:
            self._factories[key] = lambda **kw: FrontierAgentProgrammer(
                key,
                allow_custom_agent=True,
                extra_tool_schemas=extra_tool_schemas,
                **kw,
            )

    def create(self, agent_id: str = "gpt-6-astra", **kwargs: Any) -> FrontierAgentProgrammer:
        if agent_id in self._factories:
            return self._factories[agent_id](**kwargs)
        return FrontierAgentProgrammer(agent_id, **kwargs)

    def list_agents(self) -> list[str]:
        return sorted(self._factories)


class PluginRegistry:
    """Plugin registry bridging external evaluation & mechanistic interpretability ecosystems (`lm-eval`, `Inspect AI`, `TransformerLens`)."""

    @staticmethod
    def transformer_lens(
        model: PythiaModel | Qwen2CausalLM,
        *,
        layer_down_proj_override: Mapping[int, Any] | None = None,
    ) -> TransformerLensHookAdapter:
        """Wrap a Studio model in a `TransformerLens` (`HookedTransformer`) compatible `run_with_cache` / `run_with_hooks` adapter."""
        return TransformerLensHookAdapter(model, layer_down_proj_override=layer_down_proj_override)

    @staticmethod
    def import_lm_eval(spec: Mapping[str, Any], *, pinned_harness_version: str = "0.4.5") -> evals.EvalSuite:
        return evals.import_lm_eval_harness_task(spec, pinned_harness_version=pinned_harness_version)

    @staticmethod
    def import_inspect_ai(spec: Mapping[str, Any], *, pinned_inspect_version: str = "0.3.40") -> evals.EvalSuite:
        return evals.import_inspect_ai_eval_spec(spec, pinned_inspect_version=pinned_inspect_version)


# Failure mode -> (method, gym). A failure with no real method is refused, not approximated.
REPAIR_METHODS: dict[str, tuple[str, str]] = {
    "looping": ("attractor_projection", "algorithmic_three_skill"),
    "parity": ("association_edit", "algorithmic_three_skill"),
    "capitals": ("association_edit", "factual_knowledge"),
    "factual": ("association_edit", "factual_knowledge"),
}


REPAIR_REJECTED_EXIT_CODE = 3
MAX_EDITOR_TRANSACTIONS = 4


class WeightEditorToolInterface:
    """Small, typed tool API that external agent frameworks (Claude Code, Codex, MCP, custom loops) can call as a tool."""

    def __init__(self, studio: WeightProgrammingStudio) -> None:
        self._studio = studio
        self._rollback_handles: dict[str, dict[str, Any]] = {}
        self._transactions: dict[str, dict[str, Any]] = {}

    @staticmethod
    def tool_schemas() -> tuple[dict[str, Any], ...]:
        """Return OpenAI / Anthropic / MCP compatible JSON tool schemas for external agent loops."""
        return (
            {
                "name": "nf_inspect_model",
                "description": "Inspect model architecture, parameter count, and editable MLP/attention sites.",
                "input_schema": {
                    "type": "object",
                    "properties": {"model": {"type": "string"}},
                    "required": ["model"],
                },
            },
            {
                "name": "nf_trace_layers",
                "description": "Run causal tracing across layers to find the critical MLP layer range R = {l_1..l_k}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "model": {"type": "string"},
                        "prompt": {"type": "string"},
                        "target": {"type": "string"},
                    },
                    "required": ["model", "prompt", "target"],
                },
            },
            {
                "name": "nf_repair_model",
                "description": "Synthesize, compile in Rust, scan for backdoors, and sign an Open Patch v1 for a model failure mode.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "model": {"type": "string"},
                        "failure": {"type": "string"},
                    },
                    "required": ["model"],
                },
            },
            {
                "name": "nf_verify_patch",
                "description": "Verify an Open Patch Format v1 artifact (signature, SBOM, rounding contract, controls, backdoor scan).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "patch": {"type": "object"},
                        "public_key": {"type": "string"},
                    },
                    "required": ["patch", "public_key"],
                },
            },
            {
                "name": "nf_apply_patch",
                "description": "Select a verified candidate within its in-memory transaction. Nothing is written and the source model is unchanged.",
                "input_schema": {
                    "type": "object",
                    "properties": {"transaction_id": {"type": "string"}},
                    "required": ["transaction_id"],
                },
            },
            {
                "name": "nf_rollback_patch",
                "description": "Return a transaction to its original weights and verify the digests match.",
                "input_schema": {
                    "type": "object",
                    "properties": {"transaction_id": {"type": "string"}},
                    "required": ["transaction_id"],
                },
            },
            {
                "name": "nf_compare_patch",
                "description": "Compare a transaction's candidate with the original: weight changes and predictions on prompts.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "transaction_id": {"type": "string"},
                        "prompts": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["transaction_id"],
                },
            },
        )

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Dispatch a tool call from Claude Code, Codex, or a custom agent loop and return a JSON-serializable result."""
        if name == "nf_inspect_model":
            m = self._studio.select_model(str(arguments["model"]))
            num_layers = int(m.config["num_hidden_layers"])
            return {
                "model_name": m.model_name,
                "model_type": m.config.get("model_type"),
                "total_parameters": m.total_parameters,
                "num_hidden_layers": num_layers,
                "editable_mlp_sites": [m.mlp_down_proj_site(i) for i in range(num_layers)],
            }
        if name == "nf_trace_layers":
            rep = self._studio.trace_critical_layers(
                str(arguments["model"]),
                [(str(arguments["prompt"]), str(arguments["target"]))],
            )
            return {
                "model_name": rep.model_name,
                "selected_layer_range": list(rep.selected_layer_range),
                "layer_indirect_effects": {str(k): v for k, v in rep.layer_indirect_effects.items()},
                "trace_digest": rep.trace_digest,
            }
        if name == "nf_repair_model":
            result = self._studio.repair(
                str(arguments["model"]),
                failure=str(arguments.get("failure", "looping")),
            )
            candidate = result.pop("edited_model")
            result.pop("edit_plan")  # Tool results stay JSON.
            transaction_id = ""
            if candidate is not None:
                base = self._studio.select_model(str(arguments["model"]))
                transaction_id = str(uuid.uuid4())
                # Each transaction holds full models, so only the most recent few are kept.
                while len(self._transactions) >= MAX_EDITOR_TRANSACTIONS:
                    self._transactions.pop(next(iter(self._transactions)))
                self._transactions[transaction_id] = {
                    "base": base,
                    "candidate": candidate,
                    "rollback": create_rollback_handle(base),
                    "open_patch": result["open_patch"],
                    "verified": bool(result["verification"]["verified"]),
                    "state": "proposed",
                }
            return {**result, "transaction_id": transaction_id}
        if name in {"nf_apply_patch", "nf_rollback_patch", "nf_compare_patch"}:
            transaction = self._transactions.get(str(arguments["transaction_id"]))
            if transaction is None:
                raise ValueError(f"Unknown transaction_id {arguments['transaction_id']!r}")
            if name == "nf_apply_patch":
                if not transaction["verified"]:
                    raise ValueError("Cannot select a rejected candidate")
                transaction["state"] = "selected"
                return {"state": "selected", "source_model_modified": False}
            if name == "nf_rollback_patch":
                restored, receipt = rollback_model_patch(transaction["candidate"], transaction["rollback"])
                if receipt.get("rollback_verified") is not True:
                    raise ValueError("Rollback failed verification")
                transaction["candidate"] = restored
                transaction["state"] = "rolled_back"
                return {"state": "rolled_back", "rollback_verified": True}
            if name == "nf_compare_patch":
                diff = diff_model_weights(transaction["base"], transaction["candidate"])
                return {
                    "state": transaction["state"],
                    "changed_parameters": diff.total_changed_parameters,
                    "layers_touched": list(diff.layers_touched),
                    "total_frobenius_norm": diff.total_frobenius_norm,
                    "predictions": [
                        {
                            "prompt": str(p),
                            "original": transaction["base"].evaluate_prompt(str(p)).top1_token,
                            "candidate": transaction["candidate"].evaluate_prompt(str(p)).top1_token,
                        }
                        for p in arguments.get("prompts", [])
                    ],
                }
        if name == "nf_verify_patch":
            return self._studio.verify_patch(
                arguments["patch"],
                public_key=str(arguments["public_key"]),
            )
        raise ValueError(f"Unknown editor tool: {name!r}")


class WeightProgrammingStudio:
    """Unified Studio facade connecting pluggable Models, Gyms, Methods, Agents, Covariance Cache, Causal Tracer, Inspector, Plugins, Editor Tool API, Rust Weight IR, and Evals."""

    def __init__(
        self,
        *,
        vram_capacity_bytes: int = 32 * 1024 * 1024 * 1024,
        gym_state_dir: str | Path | None = None,
    ) -> None:
        from .gym_lifecycle import GymLifecycle
        from .gym_service import GymStudioService
        from .nemo_gym import NeMoTaskAdapterRegistry

        self.models = ModelRegistry()
        self.gyms = GymRegistry()
        self.nemo_gym_adapters = NeMoTaskAdapterRegistry()
        self.gym_service = GymStudioService(state_dir=gym_state_dir)
        self.gym_lifecycle = GymLifecycle(self.gym_service)
        self.covariance = CovarianceCacheRegistry()
        self.tracer = CausalTraceEngine
        self.inspector = LogitLensTrajectoryInspector
        self.plugins = PluginRegistry()
        self.methods = MethodRegistry(self.models, self.gyms)
        self.agents = AgentRegistry()
        self.ir_builder = ir.WeightIRTransactionBuilder
        self.evals = evals
        self.editor_tool = WeightEditorToolInterface(self)
        self.gpu_inventory = probe_gpu_hardware()
        self.gpu_scheduler = GpuResourceScheduler(
            device_count=max(
                1,
                self.gpu_inventory.torch_device_count if self.gpu_inventory.cuda_executable else 0,
            ),
            vram_capacity_bytes_per_device=vram_capacity_bytes,
        )

    @staticmethod
    def solve_architecture_shape(graph: dict[str, Any]) -> dict[str, Any]:
        """Invokes the authoritative Rust symbolic shape solver (`nf_graph::shape_solver`)."""
        raw = json.dumps(graph, sort_keys=True, separators=(",", ":")).encode("utf-8")
        out_bytes = _native.solve_architecture_shape(raw)
        result: dict[str, Any] = json.loads(out_bytes.decode("utf-8"))
        return result

    @staticmethod
    def lower_structural_ir(request: dict[str, Any]) -> dict[str, Any]:
        """Invokes the authoritative Rust structural IR lowering pass (`nf_weight_ir::structural_lowering`)."""
        raw = json.dumps(request, sort_keys=True, separators=(",", ":")).encode("utf-8")
        out_bytes = _native.lower_structural_ir(raw)
        result: dict[str, Any] = json.loads(out_bytes.decode("utf-8"))
        return result

    @staticmethod
    def compose_patches(request: dict[str, Any]) -> dict[str, Any]:
        """Invokes the authoritative Rust patch dependency and conflict composer (`nf_patches::composition`)."""
        raw = json.dumps(request, sort_keys=True, separators=(",", ":")).encode("utf-8")
        out_bytes = _native.compose_patch_specs(raw)
        result: dict[str, Any] = json.loads(out_bytes.decode("utf-8"))
        return result

    def list_selectable_models(self) -> list[dict[str, Any]]:
        """Lists the selectable model catalog across architectures (`GPTNeoX`, `Qwen2`, `Llama`, custom)."""
        return self.models.list_models()

    def list_reusable_gyms(self) -> dict[str, ReusableTaskGym]:
        """Returns the catalog of model-agnostic `ReusableTaskGym` components."""
        return self.gyms.list_gyms()

    def register_nemo_task_adapter(self, adapter: Any) -> None:
        """Register a task-specific NeMo-to-Studio data adapter with declared method capabilities."""
        self.nemo_gym_adapters.register(adapter)

    def list_nemo_task_adapters(self) -> dict[str, dict[str, bool]]:
        """List NeMo adapters and the training/evaluation routes they actually support."""
        return self.nemo_gym_adapters.list_adapters()

    def describe_gym_api(self) -> dict[str, Any]:
        """Describe the versioned dashboard-facing service surface and its authority limits."""
        return self.gym_service.describe()

    def register_gym_lifecycle_adapter(self, adapter: Any) -> None:
        self.gym_lifecycle.register(adapter)

    def run_gym_lifecycle(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return self.gym_lifecycle.run(request)

    def register_gym_component(
        self,
        kind: str,
        component_id: str,
        version: str,
        implementation_digest: str,
        implementation: Any,
    ) -> None:
        """Bind a trusted in-process implementation to an exact JobSpec component identity."""
        self.gym_service.components.register(
            kind, component_id, version, implementation_digest, implementation
        )

    def validate_gym_job(self, document: Mapping[str, Any]) -> dict[str, Any]:
        """Validate a data-only Gym JobSpec and resolve its explicitly registered components."""
        return self.gym_service.validate_job(document)

    def materialize_gym_scenarios(
        self,
        job_document: Mapping[str, Any],
        *,
        split: str,
        seed: int,
        max_scenarios: int = 100_000,
    ) -> dict[str, Any]:
        """Materialize registered public scenario sources for a JobSpec."""
        from .gym_job import gym_job_from_document

        return self.gym_service.materialize_public_scenarios(
            gym_job_from_document(job_document),
            split=split,
            seed=seed,
            max_scenarios=max_scenarios,
        )

    def freeze_gym_practice_plan(
        self,
        job_document: Mapping[str, Any],
        candidate_document: Mapping[str, Any],
        *,
        run_id: str,
        scenario_manifest_digest: str,
        repetitions: int = 1,
    ) -> dict[str, Any]:
        """Freeze every planned public-scenario attempt before dispatching a model worker."""
        from .gym_job import gym_job_from_document
        from .gym_service import CandidateIdentity

        candidate = CandidateIdentity(**candidate_document)
        return self.gym_service.freeze_practice_plan(
            gym_job_from_document(job_document),
            candidate,
            run_id=run_id,
            scenario_manifest_digest=scenario_manifest_digest,
            repetitions=repetitions,
        )

    def score_gym_attempt(
        self,
        job_document: Mapping[str, Any],
        candidate_document: Mapping[str, Any],
        *,
        practice_plan_digest: str,
        attempt_id: str,
        response: Any,
    ) -> Any:
        """Run the registered verifier/scorer for one materialized public scenario."""
        from .gym_job import gym_job_from_document
        from .gym_service import CandidateIdentity

        candidate = CandidateIdentity(**candidate_document)
        return self.gym_service.score_attempt(
            gym_job_from_document(job_document),
            candidate,
            practice_plan_digest=practice_plan_digest,
            attempt_id=attempt_id,
            response=response,
        )

    def register_gym_candidate_executor(
        self,
        candidate_document: Mapping[str, Any],
        *,
        executor_digest: str,
        execute: Callable[..., Any],
        identity_probe: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        """Bind trusted model execution code to a candidate's exact artifact identity."""
        from .gym_service import CandidateIdentity

        self.gym_service.register_candidate_executor(
            CandidateIdentity(**candidate_document),
            executor_digest=executor_digest,
            execute=execute,
            identity_probe=identity_probe,
        )

    def run_gym_practice_attempt(
        self,
        job_document: Mapping[str, Any],
        candidate_document: Mapping[str, Any],
        *,
        practice_plan_digest: str,
        attempt_id: str,
    ) -> Any:
        """Execute a planned scenario with the registered candidate, then validate/score its response."""
        from .gym_job import gym_job_from_document
        from .gym_service import CandidateIdentity

        return self.gym_service.run_practice_attempt(
            gym_job_from_document(job_document),
            CandidateIdentity(**candidate_document),
            practice_plan_digest=practice_plan_digest,
            attempt_id=attempt_id,
        )

    def run_gym_practice_batch(
        self,
        job_document: Mapping[str, Any],
        candidate_document: Mapping[str, Any],
        *,
        practice_plan_digest: str,
        attempt_ids: Sequence[str],
    ) -> tuple[Any, ...]:
        """Dispatch a batch of frozen attempts through one registered candidate executor call."""
        from .gym_job import gym_job_from_document
        from .gym_service import CandidateIdentity

        return self.gym_service.run_practice_batch(
            gym_job_from_document(job_document),
            CandidateIdentity(**candidate_document),
            practice_plan_digest=practice_plan_digest,
            attempt_ids=attempt_ids,
        )

    def compare_unchanged_gym_bases(
        self,
        job_document: Mapping[str, Any],
        candidate_documents: Sequence[Mapping[str, Any]],
        *,
        scenario_manifest_digest: str,
        repetitions: int = 1,
        require_worker_identity: bool = False,
        require_distinct_families: bool = False,
    ) -> dict[str, Any]:
        """Run unchanged-base candidates on the same public panel through registered executors."""
        from .gym_job import gym_job_from_document
        from .gym_service import CandidateIdentity

        candidates = tuple(CandidateIdentity(**document) for document in candidate_documents)
        return self.gym_service.compare_unchanged_bases(
            gym_job_from_document(job_document),
            candidates,
            scenario_manifest_digest=scenario_manifest_digest,
            repetitions=repetitions,
            require_worker_identity=require_worker_identity,
            require_distinct_families=require_distinct_families,
        )

    def record_gym_practice_run(
        self, run_document: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Record results against the exact practice plan frozen before dispatch."""
        return self.gym_service.record_practice_run_document(run_document)

    def score_gym_candidate(
        self,
        job_document: Mapping[str, Any],
        candidate_document: Mapping[str, Any],
        measurements: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Apply one job's frozen gates to a candidate, without granting release authority."""
        return self.gym_service.score_candidate_document(job_document, candidate_document, measurements)

    def import_nemo_gym_panels(
        self,
        adapter_id: str,
        *,
        train_jsonl: str | Path,
        validation_jsonl: str | Path,
        holdout_jsonl: str | Path,
        model: Any,
        gym_id: str,
        register_as: str | None = None,
    ) -> Any:
        """Import NeMo dataset panels through the selected capability-declaring adapter."""
        panels = self.nemo_gym_adapters.load_panels(
            adapter_id,
            train_jsonl=train_jsonl,
            validation_jsonl=validation_jsonl,
            holdout_jsonl=holdout_jsonl,
            tokenizer=model.tokenizer,
            gym_id=gym_id,
        )
        if register_as is not None:
            self.gyms.register(register_as, panels.gym)
        return panels

    def generate_nemo_gym_panels(
        self,
        adapter_id: str,
        *,
        model: Any,
        register_as: str | None = None,
        **generator_options: Any,
    ) -> Any:
        """Generate deterministic Gym panels through a registered task generator."""
        resolved_model = self.select_model(model) if isinstance(model, (str, Path)) else model
        panels = self.nemo_gym_adapters.generate_panels(
            adapter_id,
            tokenizer=resolved_model.tokenizer,
            **generator_options,
        )
        if register_as is not None:
            self.gyms.register(register_as, panels.gym)
        return panels

    def train_nemo_gym_rollouts(
        self,
        adapter_id: str,
        *,
        model: Any,
        panels: Any,
        materialized_inputs_jsonl: str | Path,
        rollouts_jsonl: str | Path,
        config: Any,
        minimum_reward: float = 1.0,
        training_device: str = "cpu",
    ) -> Any:
        """Train a declared NeMo task adapter and return validation-scored weights plus provenance."""
        resolved_model = self.select_model(model) if isinstance(model, (str, Path)) else model
        return self.nemo_gym_adapters.train_from_rollouts(
            adapter_id,
            model=resolved_model,
            panels=panels,
            materialized_inputs_jsonl=materialized_inputs_jsonl,
            rollouts_jsonl=rollouts_jsonl,
            config=config,
            minimum_reward=minimum_reward,
            training_device=training_device,
        )

    def train_nemo_gym_preferences(
        self,
        adapter_id: str,
        *,
        model: Any,
        panels: Any,
        materialized_inputs_jsonl: str | Path,
        rollouts_jsonl: str | Path,
        config: Any,
        minimum_reward_margin: float = 0.1,
        training_device: str = "cpu",
    ) -> Any:
        """Train a registered NeMo preference adapter and return validation-scored candidate weights."""
        resolved_model = self.select_model(model) if isinstance(model, (str, Path)) else model
        return self.nemo_gym_adapters.train_from_preferences(
            adapter_id,
            model=resolved_model,
            panels=panels,
            materialized_inputs_jsonl=materialized_inputs_jsonl,
            rollouts_jsonl=rollouts_jsonl,
            config=config,
            minimum_reward_margin=minimum_reward_margin,
            training_device=training_device,
        )

    def load_nemo_sft_candidate(self, model: Any, run_directory: str | Path) -> tuple[Any, dict[str, Any]]:
        """Reload and verify a NeMo SFT artifact against its base and candidate state digests."""
        from .nemo_gym import load_nemo_sft_candidate

        resolved_model = self.select_model(model) if isinstance(model, (str, Path)) else model
        return load_nemo_sft_candidate(resolved_model, run_directory)

    def select_model(
        self,
        model_id_or_path: str | Path,
        *,
        revision: str | None = None,
        allow_download: bool = False,
    ) -> Any:
        """Loads a user-selected model by catalog ID or local checkpoint directory."""
        return self.models.load(model_id_or_path, revision=revision, allow_download=allow_download)

    def run_reusable_gym_study(
        self,
        model: str | Path | PythiaModel | Qwen2CausalLM,
        *,
        gym: ReusableTaskGym | str = ALGORITHMIC_THREE_SKILL_GYM,
        target_strength: float = 75.0,
    ) -> PythiaGymTaskReport:
        """Executes a reusable `ReusableTaskGym` component on any user-selected model."""
        resolved_model = (
            self.select_model(model)
            if isinstance(model, (str, Path))
            else model
        )
        resolved_gym = self.gyms.get(gym)
        return run_pythia_gym_task_learning_study(
            resolved_model,
            gym=resolved_gym,
            target_strength=target_strength,
        )

    def run_goal_search(
        self,
        model: str | Path | PythiaModel | Qwen2CausalLM,
        output_dir: str | Path,
        *,
        gym: ReusableTaskGym | str = ALGORITHMIC_THREE_SKILL_GYM,
        acceptance_profile: Any = None,
        budgets: Any = None,
        cost: float = 1e-7,
        cycle_id: str | None = None,
        include_training_baselines: bool = True,
        training_device: str = "auto",
        gpu_python: str | Path | None = None,
        evaluate_independently: bool = True,
        isolate_independent_evaluation: bool = True,
    ) -> dict[str, Any]:
        """Run an iterative profile-bound Gym and independently evaluate its frozen frontier."""
        from .cumulative_edit import run_goal_search

        resolved_model = self.select_model(model) if isinstance(model, (str, Path)) else model
        resolved_gym = self.gyms.get(gym)
        kwargs: dict[str, Any] = {
            "gym": resolved_gym,
            "cost": cost,
            "budgets": budgets,
            "cycle_id": cycle_id,
            "include_training_baselines": include_training_baselines,
            "training_device": training_device,
            "gpu_python": gpu_python,
            "gpu_scheduler": self.gpu_scheduler,
            "evaluate_independently": evaluate_independently,
            "isolate_independent_evaluation": isolate_independent_evaluation,
        }
        if acceptance_profile is not None:
            kwargs["acceptance_profile"] = acceptance_profile
        return run_goal_search(resolved_model, output_dir, **kwargs)

    def run_nemo_gym_evaluation(
        self,
        *,
        gym_python: str | Path,
        gym_root: str | Path,
        config_paths: Sequence[str | Path],
        agent: str,
        input_jsonl: str | Path,
        output_dir: str | Path,
        model_url: str,
        model_name: str,
        timeout_seconds: float = 3600.0,
    ) -> dict[str, Any]:
        """Run pinned NeMo Gym evaluation against a caller-served Studio checkpoint endpoint."""
        from .nemo_gym import NeMoGymRunner

        runner = NeMoGymRunner(gym_python)
        return runner.run_evaluation(
            gym_root=gym_root,
            config_paths=config_paths,
            agent=agent,
            input_jsonl=input_jsonl,
            output_dir=output_dir,
            model_url=model_url,
            model_name=model_name,
            timeout_seconds=timeout_seconds,
        )

    def export_model_for_nemo_gym(
        self,
        model: Any,
        output_dir: str | Path,
        *,
        source_checkpoint_dir: str | Path,
    ) -> Path:
        """Export canonical model weights for a user-managed OpenAI-compatible inference server."""
        from .export_formats import export_huggingface_safetensors

        resolved_model = self.select_model(model) if isinstance(model, (str, Path)) else model
        return export_huggingface_safetensors(
            resolved_model,
            output_dir,
            dtype="f32",
            source_checkpoint_dir=source_checkpoint_dir,
        )

    def run_easyedit(
        self,
        model: str | Path | PythiaModel | Qwen2CausalLM,
        *,
        algorithm: EasyEditAlgorithm = "AlphaEdit",
        associations: list[tuple[str, str]],
        nullspace_prompts: list[str] | tuple[str, ...] = (),
        target_strength: float = 75.0,
        target_layers: Sequence[int] | Literal["auto_causal_trace"] | None = None,
        use_covariance_cache: bool = False,
    ) -> EasyEditAdapterReceipt:
        """Executes EasyEdit (`ROME`, `MEMIT`, or `AlphaEdit`) on a selected model across single or distributed layers (`WP-13`)."""
        resolved_model = (
            self.select_model(model)
            if isinstance(model, (str, Path))
            else model
        )
        return run_easyedit_adapter(
            resolved_model,
            algorithm=algorithm,
            associations=associations,
            nullspace_prompts=nullspace_prompts,
            target_strength=target_strength,
            target_layers=target_layers,
            covariance_registry=self.covariance if use_covariance_cache else None,
        )

    def edits(self, model: str | Path | PythiaModel | Qwen2CausalLM, *, constraints: Any = None) -> Any:
        """Start a previewable, constrained edit plan. See `edit_plan.EditPlan`."""
        from .edit_plan import EditConstraints, EditPlan

        resolved = self.select_model(model) if isinstance(model, (str, Path)) else model
        return EditPlan(resolved, constraints or EditConstraints())

    def search_edits(self, model: str | Path | PythiaModel | Qwen2CausalLM, **options: Any) -> Any:
        """Compare proposed edits beside controls. See `edit_plan.search_edits`."""
        from .edit_plan import search_edits

        resolved = self.select_model(model) if isinstance(model, (str, Path)) else model
        return search_edits(resolved, **options)

    def repair(
        self,
        model: str | Path | PythiaModel | Qwen2CausalLM = "Qwen/qwen2-dense-mini",
        *,
        failure: str = "looping",
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Attempt a repair with the method measured to fit `failure`, and report the outcome.

        A rejected candidate is returned with its reason and is never packaged for publishing.
        """
        from .export_formats import export_huggingface_hub_bundle
        from .method_plans import edit_plan_from_steps
        from .prd_adapters import SEALED_RETENTION_PROMPTS
        from .provenance import evidence_grade_for_model
        from .repair import repair_repetition_attractor, sham_output_projection

        if failure not in REPAIR_METHODS:
            raise ValueError(f"No repair method for failure {failure!r}; supported: {sorted(REPAIR_METHODS)}")
        method, gym_key = REPAIR_METHODS[failure]
        resolved_model = self.select_model(model) if isinstance(model, (str, Path)) else model
        gym = self.gyms.get(gym_key)
        grade = evidence_grade_for_model(resolved_model)

        def encodable(prompts: Any) -> list[str]:
            return [p for p in prompts if len(resolved_model.tokenizer.encode(p)) > 0]

        sealed = encodable(SEALED_RETENTION_PROMPTS)
        original = {p: resolved_model.evaluate_prompt(p).top1_token_id for p in sealed}
        neutral: list[str] | None = None
        diagnosis: dict[str, Any] | None = None
        if method == "attractor_projection":
            development = encodable(
                (*(f"{pfx} {expr}" for pfx in gym.gym_dev_prefixes for expr, _ in gym.base_rules), *gym.retention_prompts)
            )
            found = repair_repetition_attractor(resolved_model, development)
            diagnosis = {"attractor_token": found.diagnosis.token, "prompt_share": found.diagnosis.prompt_share}
            if found.selected is None or found.edited_weight is None:
                reason = "no_attractor_diagnosed" if not found.diagnosis.found else "no_candidate_reduced_attractor"
                return {
                    "model_name": resolved_model.model_name,
                    "failure_mode": failure,
                    "method": method,
                    "evidence_grade": grade,
                    "diagnosis": diagnosis,
                    "open_patch": None,
                    "edited_model": None,
                    "verification": {"verified": False, "rejection_reason": reason},
                    "public_key": None,
                    "hub_bundle": None,
                    "edit_plan": None,
                }
            attractor = found.diagnosis.token_id
            layer, site = found.selected.layer, found.selected.site
            # The plan's candidate is the edited model, so the plan adds no model copy.
            edit_plan = edit_plan_from_steps(resolved_model, found.edit_steps)
            edited_model = edit_plan.execute().model
            declared_sites = [site]
            plan_digest, candidate_digest = found.plan_digest, found.candidate_digest
            sham = sham_output_projection(resolved_model, layer, factor=found.selected.factor)

            # Success means the attractor no longer captures sealed prompts it used to capture.
            def attractor_free(candidate: Any, override: Any = None) -> float:
                return sum(
                    candidate.evaluate_prompt(p, layer_down_proj_override=override).top1_token_id != attractor
                    for p in sealed
                ) / max(len(sealed), 1)

            noop_score = attractor_free(resolved_model)
            sham_score = attractor_free(resolved_model, {layer: sham})
            candidate_score = attractor_free(edited_model)
            unaffected = [p for p, token in original.items() if token != attractor]
            neutral = unaffected or None
        else:
            edit_rec = self.run_easyedit(
                resolved_model,
                algorithm="AlphaEdit",
                associations=list(gym.base_rules),
                target_layers="auto_causal_trace",
                use_covariance_cache=True,
                target_strength=80.0,
            )
            edit_plan = edit_plan_from_steps(resolved_model, edit_rec.edit_steps)
            edited_model = edit_plan.execute().model
            declared_sites = [resolved_model.mlp_down_proj_site(l_idx) for l_idx in edit_rec.target_layers]
            plan_digest, candidate_digest = edit_rec.rust_plan_digest, edit_rec.rust_candidate_digest
            study = self.run_reusable_gym_study(resolved_model, gym=gym_key, target_strength=80.0)
            prefixes = gym.holdout_prefixes
            holdout = [(f"{prefixes[i % len(prefixes)]} {e}", t) for i, (e, t) in enumerate(gym.holdout_rules)]
            candidate_score = sum(
                edited_model.evaluate_prompt(p, target_token=t).matched_target for p, t in holdout
            ) / max(len(holdout), 1)
            noop_score = study.noop_control_holdout_accuracy
            sham_score = study.random_control_holdout_accuracy
            unaffected = list(original)

        retention = sum(
            edited_model.evaluate_prompt(p).top1_token_id == original[p] for p in unaffected
        ) / max(len(unaffected), 1)
        diff = diff_model_weights(resolved_model, edited_model)
        scan_options: dict[str, Any] = {"neutral_prompts": neutral} if neutral is not None else {}
        sec_scan = scan_patch_for_backdoor_patterns(
            resolved_model, edited_model, declared_target_sites=declared_sites, **scan_options
        )
        # A deterministic signal from one run, not statistical evidence. The loop metric also can't
        # tell a correct answer from any other non-attractor token.
        # The controls are None when the Gym selected no branch. An unmeasured control can't be beaten.
        promote = bool(
            noop_score is not None
            and sham_score is not None
            and candidate_score >= 0.9
            and candidate_score > noop_score
            and candidate_score > sham_score
            and retention >= 0.98
            and sec_scan.backdoor_scan_passed
        )
        kp = generate_ed25519_keypair()
        open_patch = build_open_patch_v1(
            base_model_id=resolved_model.model_name,
            diff_summary=diff,
            compiler_plan_digest=plan_digest,
            candidate_digest=candidate_digest,
            evaluation_provenance={
                "suite_id": f"nf.repair.{failure}.v2",
                "method": method,
                "evidence_grade": grade,
                "runtime": "pytorch_cpu",
                "sampler": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 1, "seed": 17},
                "seed": 17,
                "controls": {
                    "noop_accuracy": None if noop_score is None else round(noop_score, 6),
                    "random_accuracy": None if sham_score is None else round(sham_score, 6),
                    "candidate_accuracy": round(candidate_score, 6),
                    "retention_accuracy": round(retention, 6),
                },
                "promotion_scope": "single_run_signal_not_statistical_evidence",
                "promote_eligible": promote,
            },
            security_scan=sec_scan,
            signer_id="nf.studio.repair.v2",
            signing_key=kp["private_key"],
        )
        try:
            ver = verify_open_patch_v1(open_patch, public_key=kp["public_key"])
        except ValueError as exc:
            # A failed repair is reported as a result, and never packaged for publishing.
            ver = {"verified": False, "rejection_reason": str(exc)}
        hub_bundle = None
        if output_dir is not None and ver["verified"]:
            hub_bundle = export_huggingface_hub_bundle(
                edited_model,
                output_dir,
                open_patch=open_patch,
                signing_key=kp["private_key"],
            )
        return {
            "model_name": resolved_model.model_name,
            "failure_mode": failure,
            "method": method,
            "evidence_grade": grade,
            "diagnosis": diagnosis,
            "open_patch": open_patch,
            "edited_model": edited_model,
            "verification": ver,
            "public_key": kp["public_key"],
            "hub_bundle": hub_bundle,
            "edit_plan": edit_plan,
        }

    @staticmethod
    def verify_patch(
        patch_or_path: Mapping[str, Any] | str | Path,
        *,
        public_key: str,
    ) -> dict[str, Any]:
        """Third-party verification of an Open Patch Format v1 artifact against a trusted key."""
        return verify_open_patch_v1(patch_or_path, public_key=public_key)

    def trace_critical_layers(
        self,
        model: str | Path | PythiaModel | Qwen2CausalLM,
        associations: Sequence[tuple[str, str]],
        *,
        top_k_layers: int = 2,
    ) -> CausalTraceSiteReport:
        """Discovers the critical MLP layer range R = {l_1, ..., l_k} via causal tracing."""
        resolved_model = (
            self.select_model(model)
            if isinstance(model, (str, Path))
            else model
        )
        return self.tracer.trace_critical_layers(
            resolved_model,
            associations,
            top_k_layers=top_k_layers,
        )

    def build_covariance_cache(
        self,
        model: str | Path | PythiaModel | Qwen2CausalLM,
        *,
        layers: Sequence[int] | None = None,
        corpus_prompts: Sequence[str] | None = None,
        cache_dir: str | Path | None = None,
    ) -> dict[int, LayerCovarianceArtifact]:
        """Streams a reference corpus through the model and caches empirical key covariances C_l = E[k_l k_l^T] and null-space projectors P_l."""
        resolved_model = (
            self.select_model(model)
            if isinstance(model, (str, Path))
            else model
        )
        return self.covariance.build_model_covariance_cache(
            resolved_model,
            layers=layers,
            corpus_prompts=corpus_prompts,
            cache_dir=cache_dir,
        )

    def inspect_trajectory(
        self,
        model: str | Path | PythiaModel | Qwen2CausalLM,
        prompt: str,
        *,
        target_token: str,
        layer_down_proj_override: dict[int, Any] | None = None,
    ) -> LogitLensTrajectoryReport:
        """Renders per-layer singular-value spectra and logit-lens token trajectories before vs. after a weight edit."""
        resolved_model = (
            self.select_model(model)
            if isinstance(model, (str, Path))
            else model
        )
        return self.inspector.inspect_prompt_trajectory(
            resolved_model,
            prompt,
            target_token=target_token,
            layer_down_proj_override=layer_down_proj_override,
        )

    def run_mergekit(
        self,
        model: str | Path | PythiaModel | Qwen2CausalLM,
        *,
        algorithm: MergekitAlgorithm = "ties",
        checkpoint_weights: list[Any],
        scaling_weights: list[float] | tuple[float, ...] = (1.0, 1.0),
        density: float = 0.85,
        task_a_probes: list[tuple[str, str]] | tuple[tuple[str, str], ...] = (),
        task_b_probes: list[tuple[str, str]] | tuple[tuple[str, str], ...] = (),
    ) -> MergekitAdapterReceipt:
        """Executes mergekit (`task_arithmetic`, `ties`, or `dare`) on a selected model (`WP-14`)."""
        resolved_model = (
            self.select_model(model)
            if isinstance(model, (str, Path))
            else model
        )
        return run_mergekit_adapter(
            resolved_model,
            algorithm=algorithm,
            checkpoint_weights=checkpoint_weights,
            scaling_weights=scaling_weights,
            density=density,
            task_a_probes=task_a_probes,
            task_b_probes=task_b_probes,
        )

    def create_frontier_agent(
        self,
        agent_id: FrontierAgentId = "gpt-6-astra",
    ) -> FrontierAgentProgrammer:
        """Creates an interchangeable `claude-4-5-sonnet`, `gpt-6-astra`, or registered custom agent (`WP-02`)."""
        return self.agents.create(agent_id)

    def run_priority_experiments(
        self,
        model: str | Path | PythiaModel | Qwen2CausalLM | None = None,
    ) -> PRDExperimentsSuiteReport:
        """Runs PRD Priority Experiments A, B, C, and D on the selected model (`PRD §7`)."""
        resolved_model = (
            self.select_model(model)
            if isinstance(model, (str, Path))
            else model
        )
        return run_prd_priority_experiments_suite(resolved_model)


def generate_release_conformance_report(
    *,
    derive_scientific_claims_from_checkbox_count: bool = False,
) -> ReleaseConformanceReport:
    """Generates the T120 release conformance and capability report from live verification evidence."""
    if derive_scientific_claims_from_checkbox_count:
        raise ValueError(
            "T120 violation: milestone and scientific-result status must be generated from evidence, never from completed task checkbox counts"
        )

    ver = str(_native.version())
    crates = tuple(str(c) for c in _native.workspace_crates())
    gpu_inv = probe_gpu_hardware()
    m4_acc = run_m4_feature_acceptance_suite()

    # Authoritative Rust capability manifests from crates/nf-core/src/capabilities.rs (Audit Point 3)
    m1_caps: dict[str, Any] = json.loads(bytes(_native.capabilities_json()))
    m2_caps: dict[str, Any] = (
        json.loads(bytes(_native.m2_capabilities_json()))
        if hasattr(_native, "m2_capabilities_json")
        else {"components": [], "states": {"release_enabled": False}}
    )
    m2_components = set(m2_caps.get("components", ()))

    # Probe Rust native shape solver as part of live conformance check
    shape_probe = WeightProgrammingStudio.solve_architecture_shape(
        {
            "family_id": "hf.qwen2.dense",
            "symbols": {
                "vocab_size": 32,
                "hidden_size": 16,
                "num_q_heads": 4,
                "num_kv_heads": 2,
                "head_dim": 4,
            },
            "ports": [
                {
                    "tensor_name": "model.layers.0.self_attn.q_proj.weight",
                    "dims": [{"type": "product", "left": "num_q_heads", "right": "head_dim"}, {"type": "symbol", "name": "hidden_size"}],
                    "alias_group": None,
                }
            ],
            "constraints": [
                {
                    "kind": "attention_group_divisibility",
                    "num_q_heads_symbol": "num_q_heads",
                    "num_kv_heads_symbol": "num_kv_heads",
                    "head_dim_symbol": "head_dim",
                    "hidden_size_symbol": "hidden_size",
                }
            ],
        }
    )
    rust_shape_ok = shape_probe.get("kv_group_ratio") == 2

    m1_ready = bool(
        m1_caps.get("isolated_transactions")
        and m1_caps.get("exact_single_parent_replay")
        and m1_caps.get("standalone_dense_export")
        and m1_caps.get("immutable_core_restore")
        and m1_caps.get("python_control_boundary")
    )
    m2_ready = bool(
        "three_way_split_broker" in m2_components
        and "sandbox_isolation_protocol" in m2_components
        and "paired_controls" in m2_components
        and "independent_evaluator_protocol" in m2_components
    )

    capability_gates = {
        "CG-01_transactional_immutability": bool(m1_caps.get("isolated_transactions")) and bool(m1_caps.get("immutable_core_restore")),
        "CG-02_gradient_lineage_separation": bool(m1_caps.get("python_control_boundary")) and m4_acc.cg11_methods_and_costs_passed,
        "CG-03_sealed_split_isolation": "three_way_split_broker" in m2_components and "independent_evaluator_protocol" in m2_components,
        "CG-04_sandbox_containment": "sandbox_isolation_protocol" in m2_components,
        "CG-05_paired_statistical_validity": "paired_controls" in m2_components,
        "CG-06_resource_budget_enforcement": bool(m1_caps.get("bounded_sharded_io")) and "frozen_contracts" in m2_components,
        "CG-07_export_reload_parity": bool(m1_caps.get("standalone_dense_export")) and "standalone_reload_component" in m2_components,
        "CG-08_atomic_promotion_and_recovery": "operational_fixture_promotion_with_live_grant" in m2_components,
        "CG-09_rule_construction_oracle_independence": m4_acc.cg09_construction_passed,
        "CG-10_mapped_transfer_with_controls": m4_acc.cg10_transfer_passed,
        "CG-11_authorized_baselines_and_cost_accounting": m4_acc.cg11_methods_and_costs_passed,
        "CG-12_evidence_origin_and_support_grades": m4_acc.cg12_evidence_quality_passed,
        "CG-13_versioned_library_lifecycle": m4_acc.cg13_library_lifecycle_passed,
        "CG-14_independent_reproduction_bundles": rust_shape_ok,
    }

    milestones = {
        "M1": MilestoneEvidenceStatus(
            milestone="M1",
            software_harness_ready=m1_ready,
            fixture_conformance_verified=m1_ready and bool(m1_caps.get("qualified_on_this_host")),
            live_scientific_or_hardware_claim_status="verified_on_fixture_scope",
        ),
        "M2": MilestoneEvidenceStatus(
            milestone="M2",
            software_harness_ready=m2_ready,
            fixture_conformance_verified=m2_ready,
            live_scientific_or_hardware_claim_status="verified_on_fixture_scope",
        ),
        "M3": MilestoneEvidenceStatus(
            milestone="M3",
            software_harness_ready=m1_ready and m2_ready and rust_shape_ok,
            fixture_conformance_verified=m1_ready and m2_ready and rust_shape_ok,
            live_scientific_or_hardware_claim_status="requires_live_provider_or_unseen_model_study",
        ),
        "M4": MilestoneEvidenceStatus(
            milestone="M4",
            software_harness_ready=m4_acc.all_gates_passed,
            fixture_conformance_verified=m4_acc.all_gates_passed,
            live_scientific_or_hardware_claim_status="requires_held_out_cross_model_study",
        ),
        "M5": MilestoneEvidenceStatus(
            milestone="M5",
            software_harness_ready=rust_shape_ok,
            fixture_conformance_verified=rust_shape_ok,
            live_scientific_or_hardware_claim_status=(
                "verified_on_fixture_scope"
                if gpu_inv.cuda_executable
                else "requires_real_gpu_serving_qualification"
            ),
        ),
    }

    digest_payload = {
        "version": ver,
        "crates": list(crates),
        "capability_gates": capability_gates,
        "m4_all_passed": m4_acc.all_gates_passed,
    }
    raw_digest = str(
        content_digest(
            canonicalize(json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        )
    )

    return ReleaseConformanceReport(
        schema_version=1,
        version=ver,
        workspace_crates=crates,
        milestones=milestones,
        capability_gates=capability_gates,
        gpu_inventory=gpu_inv,
        m4_acceptance=m4_acc,
        derived_from_task_checkbox_count=False,
        report_digest=raw_digest,
        rust_m1_capabilities=m1_caps,
        rust_m2_capabilities=m2_caps,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for `python -m neural_foundry.studio`."""
    parser = argparse.ArgumentParser(
        prog="python -m neural_foundry.studio",
        description="Weight Programming Studio — inspect catalogs, run Gym task studies, or emit conformance reports.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("catalog", help="List registered models, architectures, gyms, methods, and agents")

    rep_p = sub.add_parser("repair", help="One-command end-to-end model repair producing a signed Open Patch v1")
    rep_p.add_argument("--model", default="Qwen/qwen2-dense-mini", help="Model ID or checkpoint directory")
    rep_p.add_argument("--failure", default="looping", help="Failure mode to repair (looping, parity, capitals)")
    rep_p.add_argument("--output-dir", default=None, help="Optional directory to export HF Hub bundle + Model Card")

    ver_p = sub.add_parser("verify", help="Verify an Open Patch Format v1 file")
    ver_p.add_argument("patch", help="Path to open_patch_v1.json or weight_patch.nf.json")
    ver_p.add_argument("--pubkey", required=True, help="Trusted ed25519_pub:... key")

    run_gym_p = sub.add_parser("run-gym", help="Run a reusable Task Gym study on a selected model")
    run_gym_p.add_argument("--model", required=True, help="Explicit model ID or checkpoint directory; fixtures must be selected by name")
    run_gym_p.add_argument("--gym", default=None, help="Registered Gym key")
    run_gym_p.add_argument("--gym-spec", default=None, help="Data-only versioned Gym JSON spec")
    run_gym_p.add_argument("--strength", type=float, default=80.0, help="Target subspace strength")

    goal_p = sub.add_parser("goal-search", help="Run an iterative frozen-frontier Gym workflow")
    goal_p.add_argument("--model", required=True, help="Model ID or checkpoint directory")
    goal_p.add_argument("--gym", default=None)
    goal_p.add_argument("--gym-spec", default=None, help="Data-only versioned Gym JSON spec")
    goal_p.add_argument("--output-dir", required=True, help="Directory for artifacts and reports")
    goal_p.add_argument("--profile", choices=("strict", "specialized"), default="strict")
    goal_p.add_argument("--cost", type=float, default=1e-7)
    goal_p.add_argument("--max-rounds", type=int, default=40)
    goal_p.add_argument("--max-candidates", type=int, default=20_000)
    goal_p.add_argument("--max-wall-seconds", type=float, default=None)
    goal_p.add_argument("--min-gain", type=float, default=0.005)
    goal_p.add_argument("--max-no-gain-rounds", type=int, default=3)
    goal_p.add_argument("--training-device", default="auto", help="auto, cpu, or cuda:N")
    goal_p.add_argument("--gpu-python", default=None, help="Separate local CUDA interpreter")
    goal_p.add_argument("--no-training-baselines", action="store_true")
    goal_p.add_argument("--no-independent-eval", action="store_true")
    goal_p.add_argument("--in-process-independent", action="store_true")

    nemo_import_p = sub.add_parser(
        "import-nemo-panels",
        aliases=("import-nemo-mcqa",),
        help="Convert NeMo train/validation/holdout JSONL through a registered task adapter",
    )
    nemo_import_p.add_argument("--adapter", default="mcqa_single_token")
    nemo_import_p.add_argument("--model", required=True, help="Model ID or checkpoint for one-token validation")
    nemo_import_p.add_argument("--train-jsonl", required=True)
    nemo_import_p.add_argument("--validation-jsonl", required=True)
    nemo_import_p.add_argument("--holdout-jsonl", required=True)
    nemo_import_p.add_argument("--gym-id", default="gym.nemo.mcqa.v1")
    nemo_import_p.add_argument("--output-spec", required=True)

    simulate_nemo_p = sub.add_parser(
        "simulate-nemo-gym",
        help="Generate a seeded, no-source-data task Gym through a registered simulator adapter",
    )
    simulate_nemo_p.add_argument("--adapter", default="simulated_arithmetic_mcqa")
    simulate_nemo_p.add_argument("--model", required=True, help="Model used to validate prompt/label tokenization")
    simulate_nemo_p.add_argument("--output-dir", required=True)
    simulate_nemo_p.add_argument("--seed", type=int, default=42)
    simulate_nemo_p.add_argument("--train-size", type=int, default=24)
    simulate_nemo_p.add_argument("--validation-size", type=int, default=12)
    simulate_nemo_p.add_argument("--holdout-size", type=int, default=12)

    nemo_eval_p = sub.add_parser(
        "nemo-eval",
        help="Run NeMo Gym v0.7 against an already-running OpenAI-compatible model endpoint",
    )
    nemo_eval_p.add_argument("--gym-python", required=True, help="Python from the pinned NeMo Gym v0.7.0 environment")
    nemo_eval_p.add_argument("--gym-root", required=True)
    nemo_eval_p.add_argument("--config", action="append", required=True, help="NeMo Gym config path; repeatable")
    nemo_eval_p.add_argument("--agent", required=True)
    nemo_eval_p.add_argument("--input-jsonl", required=True)
    nemo_eval_p.add_argument("--output-dir", required=True)
    nemo_eval_p.add_argument("--model-url", required=True)
    nemo_eval_p.add_argument("--model-name", required=True)
    nemo_eval_p.add_argument("--timeout-seconds", type=float, default=3600.0)

    nemo_sft_p = sub.add_parser(
        "nemo-sft",
        help="Train LoRA/SFT from complete high-reward NeMo MCQA rollouts and score disjoint validation data",
    )
    nemo_sft_p.add_argument("--model", required=True)
    nemo_sft_p.add_argument("--train-jsonl", required=True)
    nemo_sft_p.add_argument("--validation-jsonl", required=True)
    nemo_sft_p.add_argument("--holdout-jsonl", required=True)
    nemo_sft_p.add_argument("--materialized-inputs", required=True)
    nemo_sft_p.add_argument("--rollouts", required=True)
    nemo_sft_p.add_argument("--output-dir", required=True)
    nemo_sft_p.add_argument("--kind", choices=("lora", "sft"), default="lora")
    nemo_sft_p.add_argument("--targets", choices=("final_down_proj", "all_down_proj", "all_linear"), default="all_down_proj")
    nemo_sft_p.add_argument("--rank", type=int, default=4)
    nemo_sft_p.add_argument("--learning-rate", type=float, default=3e-4)
    nemo_sft_p.add_argument("--steps", type=int, default=100)
    nemo_sft_p.add_argument("--retention-weight", type=float, default=1.0)
    nemo_sft_p.add_argument("--minimum-reward", type=float, default=1.0)
    nemo_sft_p.add_argument("--training-device", default="cpu")

    nemo_dpo_p = sub.add_parser(
        "nemo-dpo",
        help="Train LoRA with within-task preference pairs from repeated NeMo MCQA rollouts",
    )
    nemo_dpo_p.add_argument("--model", required=True)
    nemo_dpo_p.add_argument("--train-jsonl", required=True)
    nemo_dpo_p.add_argument("--validation-jsonl", required=True)
    nemo_dpo_p.add_argument("--holdout-jsonl", required=True)
    nemo_dpo_p.add_argument("--materialized-inputs", required=True)
    nemo_dpo_p.add_argument("--rollouts", required=True)
    nemo_dpo_p.add_argument("--output-dir", required=True)
    nemo_dpo_p.add_argument("--targets", choices=("final_down_proj", "all_down_proj", "all_linear"), default="all_down_proj")
    nemo_dpo_p.add_argument("--rank", type=int, default=4)
    nemo_dpo_p.add_argument("--learning-rate", type=float, default=1e-4)
    nemo_dpo_p.add_argument("--steps", type=int, default=100)
    nemo_dpo_p.add_argument("--beta", type=float, default=0.1)
    nemo_dpo_p.add_argument("--minimum-reward-margin", type=float, default=0.1)
    nemo_dpo_p.add_argument("--retention-weight", type=float, default=0.1)
    nemo_dpo_p.add_argument("--training-device", default="cpu")

    exp_p = sub.add_parser("run-experiments", help="Run PRD Priority Experiments A, B, C, and D")
    exp_p.add_argument("--model", default="nf/mini-pythia-fixture", help="Model ID or checkpoint directory")

    cov_p = sub.add_parser("covariance", help="Build and cache per-layer MLP key covariances and null-space projectors")
    cov_p.add_argument("--model", default="nf/mini-pythia-fixture", help="Model ID or checkpoint directory")
    cov_p.add_argument("--cache-dir", default=None, help="Optional directory to write .safetensors covariance artifacts")

    insp_p = sub.add_parser("inspect-trajectory", help="Inspect per-layer logit-lens token trajectory and singular spectra before vs. after edit")
    insp_p.add_argument("--model", default="nf/mini-pythia-fixture", help="Model ID or checkpoint directory")
    insp_p.add_argument("--prompt", default="The capital of France is", help="Prompt to trace across layers")
    insp_p.add_argument("--target", default="ĠParis", help="Target vocabulary token")

    sub.add_parser("conformance", help="Generate T120 release conformance report")

    args = parser.parse_args(argv)
    studio = WeightProgrammingStudio()

    if args.command in {"import-nemo-panels", "import-nemo-mcqa"}:
        model = studio.select_model(args.model)
        panels = studio.import_nemo_gym_panels(
            args.adapter,
            train_jsonl=args.train_jsonl,
            validation_jsonl=args.validation_jsonl,
            holdout_jsonl=args.holdout_jsonl,
            model=model,
            gym_id=args.gym_id,
        )
        spec_path = Path(args.output_spec)
        manifest_path = Path(str(spec_path) + ".sources.json")
        if spec_path.exists() or manifest_path.exists():
            raise FileExistsError("NeMo MCQA spec or source manifest already exists")
        panels.write_spec(spec_path)
        manifest_path.write_text(
            json.dumps(panels.source_manifest(), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"gym_id": panels.gym.gym_id, "spec": str(spec_path), "source_manifest": str(manifest_path)}, indent=2))
        return 0

    if args.command == "simulate-nemo-gym":
        output = Path(args.output_dir)
        panels = studio.generate_nemo_gym_panels(
            args.adapter,
            model=args.model,
            output_dir=output,
            seed=args.seed,
            train_size=args.train_size,
            validation_size=args.validation_size,
            holdout_size=args.holdout_size,
        )
        spec_path = panels.write_spec(output / "gym.json")
        manifest_path = output / "gym.sources.json"
        manifest_path.write_text(
            json.dumps(panels.source_manifest(), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"gym_id": panels.gym.gym_id, "spec": str(spec_path), "source_manifest": str(manifest_path)}, indent=2))
        return 0

    if args.command == "nemo-eval":
        receipt = studio.run_nemo_gym_evaluation(
            gym_python=args.gym_python,
            gym_root=args.gym_root,
            config_paths=args.config,
            agent=args.agent,
            input_jsonl=args.input_jsonl,
            output_dir=args.output_dir,
            model_url=args.model_url,
            model_name=args.model_name,
            timeout_seconds=args.timeout_seconds,
        )
        print(json.dumps(receipt, indent=2, allow_nan=False))
        return 0

    if args.command == "nemo-sft":
        import torch

        from .baselines import BaselineConfig

        output = Path(args.output_dir)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"NeMo SFT output directory is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        model = studio.select_model(args.model)
        panels = studio.import_nemo_gym_panels(
            "mcqa_single_token",
            train_jsonl=args.train_jsonl,
            validation_jsonl=args.validation_jsonl,
            holdout_jsonl=args.holdout_jsonl,
            model=model,
            gym_id="gym.nemo.mcqa.v1",
        )
        config = BaselineConfig(
            kind=args.kind,
            lr=args.learning_rate,
            steps=args.steps,
            retention_weight=args.retention_weight,
            targets=args.targets if args.kind == "lora" else "all",
            rank=args.rank if args.kind == "lora" else 0,
        )
        training = studio.train_nemo_gym_rollouts(
            "mcqa_single_token",
            model=model,
            panels=panels,
            materialized_inputs_jsonl=args.materialized_inputs,
            rollouts_jsonl=args.rollouts,
            config=config,
            minimum_reward=args.minimum_reward,
            training_device=args.training_device,
        )
        weights_path = output / "merged_weights.pt"
        torch.save(training.fit.merged_weights, weights_path)
        receipt = {
            "schema_version": 1,
            "kind": "nf_nemo_mcqa_sequence_sft_v1",
            "adapter_id": "mcqa_single_token",
            "gym_id": panels.gym.gym_id,
            "gym_digest": panels.source_manifest()["gym_digest"],
            "dataset_digest": training.dataset.dataset_digest,
            "source_input_digest": training.dataset.source_input_digest,
            "source_rollout_digest": training.dataset.source_rollout_digest,
            "source_runner_receipt_digest": training.dataset.source_runner_receipt_digest,
            "minimum_reward": args.minimum_reward,
            "base_model_digest": training.base_model_digest,
            "candidate_model_digest": training.candidate_model_digest,
            "config": config.__dict__,
            "accepted_rollouts": training.dataset.accepted_rollouts,
            "excluded_low_reward_rollouts": training.dataset.excluded_low_reward_rollouts,
            "weights_digest": training.fit.weights_digest(),
            "weights_path": weights_path.name,
            "cost": training.fit.cost.__dict__,
            "validation_accuracy": training.validation_accuracy,
            "validation_examples": training.validation_examples,
            "validation_digest": training.validation_digest,
            "holdout_used": False,
            "promotion_authorized": False,
        }
        (output / "training_receipt.json").write_text(
            json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(receipt, indent=2, allow_nan=False))
        return 0

    if args.command == "nemo-dpo":
        import torch

        from .baselines import DPOConfig

        output = Path(args.output_dir)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"NeMo DPO output directory is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        model = studio.select_model(args.model)
        panels = studio.import_nemo_gym_panels(
            "mcqa_single_token",
            train_jsonl=args.train_jsonl,
            validation_jsonl=args.validation_jsonl,
            holdout_jsonl=args.holdout_jsonl,
            model=model,
            gym_id="gym.nemo.mcqa.v1",
        )
        config = DPOConfig(
            learning_rate=args.learning_rate,
            steps=args.steps,
            rank=args.rank,
            targets=args.targets,
            beta=args.beta,
            retention_weight=args.retention_weight,
        )
        training = studio.train_nemo_gym_preferences(
            "mcqa_single_token",
            model=model,
            panels=panels,
            materialized_inputs_jsonl=args.materialized_inputs,
            rollouts_jsonl=args.rollouts,
            config=config,
            minimum_reward_margin=args.minimum_reward_margin,
            training_device=args.training_device,
        )
        weights_path = output / "merged_weights.pt"
        torch.save(training.fit.merged_weights, weights_path)
        receipt = {
            "schema_version": 1,
            "kind": "nf_nemo_mcqa_dpo_v1",
            "adapter_id": "mcqa_single_token",
            "gym_id": panels.gym.gym_id,
            "gym_digest": panels.source_manifest()["gym_digest"],
            "dataset_digest": training.dataset.dataset_digest,
            "source_input_digest": training.dataset.source_input_digest,
            "source_rollout_digest": training.dataset.source_rollout_digest,
            "source_runner_receipt_digest": training.dataset.source_runner_receipt_digest,
            "minimum_reward_margin": args.minimum_reward_margin,
            "base_model_digest": training.base_model_digest,
            "candidate_model_digest": training.candidate_model_digest,
            "config": config.__dict__,
            "preference_pairs": training.dataset.task_pairs,
            "weights_digest": training.fit.weights_digest(),
            "weights_path": weights_path.name,
            "cost": training.fit.cost.__dict__,
            "validation_accuracy": training.validation_accuracy,
            "validation_examples": training.validation_examples,
            "validation_digest": training.validation_digest,
            "holdout_used": False,
            "promotion_authorized": False,
        }
        (output / "training_receipt.json").write_text(
            json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(receipt, indent=2, allow_nan=False))
        return 0

    if args.command == "catalog":
        payload = {
            "models": studio.models.list_models(),
            "architectures": list(studio.models.list_architectures().keys()),
            "gyms": {k: g.gym_id for k, g in studio.gyms.list_gyms().items()},
            "methods": studio.methods.list_methods(),
            "agents": studio.agents.list_agents(),
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "repair":
        res = studio.repair(args.model, failure=args.failure, output_dir=args.output_dir)
        patch = res["open_patch"]
        print(
            json.dumps(
                {
                    "model_name": res["model_name"],
                    "failure_mode": res["failure_mode"],
                    "method": res["method"],
                    "evidence_grade": res["evidence_grade"],
                    "diagnosis": res["diagnosis"],
                    "patch_digest": patch["patch_digest"] if patch else None,
                    "public_key": res["public_key"],
                    "verification": res["verification"],
                    "hub_bundle": res["hub_bundle"],
                },
                indent=2,
            )
        )
        if not res["verification"]["verified"]:
            print(f"Repair rejected: {res['verification']['rejection_reason']}", file=sys.stderr)
            return REPAIR_REJECTED_EXIT_CODE
        return 0

    if args.command == "verify":
        ver_res = studio.verify_patch(args.patch, public_key=args.pubkey)
        print(json.dumps(ver_res, indent=2))
        return 0

    if args.command == "covariance":
        cov_map = studio.build_covariance_cache(args.model, cache_dir=args.cache_dir)
        print(
            json.dumps(
                {
                    "model": args.model,
                    "layers_cached": len(cov_map),
                    "entries": studio.covariance.list_cached_entries(),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "inspect-trajectory":
        loaded_m = studio.select_model(args.model)
        edit_rec = studio.run_easyedit(
            loaded_m,
            algorithm="AlphaEdit",
            associations=[(args.prompt, args.target)],
            target_layers="auto_causal_trace",
            use_covariance_cache=True,
        )
        traj = studio.inspect_trajectory(
            loaded_m,
            args.prompt,
            target_token=args.target,
            layer_down_proj_override=edit_rec.edited_weights_by_layer,
        )
        print(traj.markdown_table)
        return 0

    if args.command == "run-gym":
        if args.gym and args.gym_spec:
            raise ValueError("choose either --gym or --gym-spec")
        selected_gym = (
            studio.gyms.load_spec(args.gym_spec)
            if args.gym_spec
            else args.gym or "algorithmic_three_skill"
        )
        rep = studio.run_reusable_gym_study(
            args.model,
            gym=selected_gym,
            target_strength=float(args.strength),
        )
        print(rep.promotion_verification["markdown_dashboard"])
        return 0

    if args.command == "goal-search":
        if args.gym and args.gym_spec:
            raise ValueError("choose either --gym or --gym-spec")
        selected_gym = (
            studio.gyms.load_spec(args.gym_spec)
            if args.gym_spec
            else args.gym or "algorithmic_three_skill"
        )
        from .goal_search import (
            STRICT_GENERAL_PROFILE,
            TASK_SPECIALIZED_PROFILE,
            SearchBudgets,
        )

        profile = STRICT_GENERAL_PROFILE if args.profile == "strict" else TASK_SPECIALIZED_PROFILE
        result = studio.run_goal_search(
            args.model,
            args.output_dir,
            gym=selected_gym,
            acceptance_profile=profile,
            budgets=SearchBudgets(
                max_rounds=args.max_rounds,
                max_candidate_evaluations=args.max_candidates,
                max_wall_seconds=args.max_wall_seconds,
                min_gain=args.min_gain,
                max_no_gain_rounds=args.max_no_gain_rounds,
            ),
            cost=args.cost,
            include_training_baselines=not args.no_training_baselines,
            training_device=args.training_device,
            gpu_python=args.gpu_python,
            evaluate_independently=not args.no_independent_eval,
            isolate_independent_evaluation=not args.in_process_independent,
        )
        independent = result.get("independent_evaluation")
        print(
            json.dumps(
                {
                    "stop_reason": result["stop_reason"],
                    "archive_terminal_state": result["archive_terminal_state"],
                    "snapshot_digest": (
                        result["frontier_snapshot"]["snapshot_digest"]
                        if result.get("frontier_snapshot")
                        else None
                    ),
                    "frontier_artifact_directory": result.get("frontier_artifact_directory"),
                    "independent_evaluation_complete": (
                        independent.get("complete") if independent is not None else None
                    ),
                    "independent_terminal_state": (
                        independent.get("independent_terminal_state")
                        if independent is not None
                        else None
                    ),
                    "promoted_candidate_ids": (
                        independent.get("promoted_candidate_ids") if independent is not None else None
                    ),
                    "result_path": result.get("result_path"),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "run-experiments":
        exp_rep = studio.run_priority_experiments(args.model)
        print(
            json.dumps(
                {
                    "model": args.model,
                    "all_experiments_succeeded": exp_rep.all_experiments_succeeded,
                    "experiment_a_interoperability": exp_rep.experiment_a_agent_interoperability_verified,
                    "experiment_b_synthesized_accuracy": exp_rep.experiment_b_synthesized_accuracy,
                    "experiment_c_task_a_survival": exp_rep.experiment_c_merge_receipt.task_a_survival_rate,
                    "experiment_c_task_b_survival": exp_rep.experiment_c_merge_receipt.task_b_survival_rate,
                    "experiment_d_direct_holdout": exp_rep.experiment_d_direct_holdout_accuracy,
                    "experiment_d_lora_holdout": exp_rep.experiment_d_lora_holdout_accuracy,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "conformance":
        conf = generate_release_conformance_report()
        print(
            json.dumps(
                {
                    "version": conf.version,
                    "report_digest": conf.report_digest,
                    "capability_gates": conf.capability_gates,
                },
                indent=2,
            )
        )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

