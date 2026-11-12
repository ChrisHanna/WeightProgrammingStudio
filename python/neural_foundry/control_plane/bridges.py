"""Fixed existing-authority adapters. Only trusted deployment code can install a profile.

These are not enabled by environment strings, catalog inspection, or uploaded specs.
The deployment owner must qualify the runtime and cap each capability separately.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

from .models import digest, fail, identifier

if TYPE_CHECKING:
    from ..proof_registry import ProofRegistry

CAPABILITIES = {
    "proof.train": "model_training", "proof.export": "model_export",
    "proof.identity": "model_identity", "proof.score": "model_scoring",
    "proof.offline_cordial_rescore": "offline_receipt_scoring",
}


@dataclass(frozen=True)
class QualifiedGymBinding:
    project_id: str
    gym_asset_id: str
    gym_version: int
    gym_artifact_id: str
    proof_gym_reference_digest: str
    proof_gym_configuration_digest: str
    methods: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        identifier(self.project_id, "project ID")
        identifier(self.gym_asset_id, "gym asset ID")
        if type(self.gym_version) is not int or self.gym_version < 1:
            fail("invalid_runtime_profile", "Qualified gym version must be positive.")
        for value in (self.gym_artifact_id, self.proof_gym_reference_digest, self.proof_gym_configuration_digest):
            if not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
                fail("invalid_runtime_profile", "Qualified gym binding requires content digests.")
        if not isinstance(self.methods, frozenset) or not self.methods or self.methods - {"lora", "sft"}:
            fail("invalid_runtime_profile", "Qualified gym methods must explicitly select lora and/or sft.")


@dataclass(frozen=True)
class TrustedRuntimeProfile:
    id: str
    root: Path
    qualified: bool = False
    deployment_capabilities: frozenset[str] = frozenset()
    automodel_python: str = ""
    reload_python: str = ""
    registry: ProofRegistry | None = None
    callbacks: Mapping[str, Callable[[BridgeContext], dict[str, Any]]] = field(default_factory=dict)
    product_gym_bindings: tuple[QualifiedGymBinding, ...] = ()
    descriptor: Mapping[str, Any] = field(default_factory=dict)
    operator_config: str = ""

    def __post_init__(self) -> None:
        identifier(self.id, "runtime profile")
        if set(self.callbacks) - set(CAPABILITIES):
            fail("invalid_runtime_profile", "Runtime callback is not a registered bridge.")
        if not isinstance(self.product_gym_bindings, tuple) or any(
                not isinstance(binding, QualifiedGymBinding) for binding in self.product_gym_bindings):
            fail("invalid_runtime_profile", "Product gym bindings must be trusted immutable qualification records.")


def require_product_gym_binding(profile: TrustedRuntimeProfile, *, project_id: str,
                                gym_asset_id: str, gym_version: int, gym_artifact_id: str,
                                proof_gym_reference: Any, proof_gym_configuration: Any, method: str) -> None:
    if not profile.qualified or profile.registry is None:
        fail("runtime_not_qualified", "Operator must install a qualified proof registry and explicit product gym binding.", 409)
    from ..gym_job import GymJobError
    try:
        profile.registry.resolve("gym", proof_gym_reference)
    except (GymJobError, TypeError, KeyError):
        fail("runtime_not_qualified", "The qualified proof gym reference must resolve in the installed registry.", 409)
    if not any(
        binding.project_id == project_id and binding.gym_asset_id == gym_asset_id
        and binding.gym_version == gym_version and binding.gym_artifact_id == gym_artifact_id
        and binding.proof_gym_reference_digest == digest(proof_gym_reference)
        and binding.proof_gym_configuration_digest == digest(proof_gym_configuration)
        and method in binding.methods for binding in profile.product_gym_bindings
    ):
        fail("runtime_not_qualified", "Operator must qualify this exact saved gym identity, proof registry reference, configuration and method.", 409)


class BridgeContext(TypedDict):
    profile: TrustedRuntimeProfile
    document: dict[str, Any]
    role: str
    root: Path
    limits: dict[str, int]
    operation: str


def _train(context: BridgeContext) -> dict[str, Any]:
    from ..proof_pipeline import train
    profile = context["profile"]
    return train(context["document"], role=context["role"], root=context["root"],
                 automodel_python=profile.automodel_python, reload_python=profile.reload_python,
                 registry=profile.registry)


def _export(context: BridgeContext) -> dict[str, Any]:
    from ..proof_plugins import serving_export
    return serving_export(spec=context["document"], role=context["role"], root=context["root"])


def _identity(context: BridgeContext) -> dict[str, Any]:
    from ..proof_plugins import candidate_evidence
    from ..proof_registry import builtin_registry
    spec, role, profile = context["document"], context["role"], context["profile"]
    candidate = next(c for c in spec["candidates"] if c["role"] == role)
    registry = profile.registry or builtin_registry()[0]
    execution = candidate["execution"]
    factory = registry.resolve(execution["kind"], execution["plugin"])
    with factory(spec=spec, candidate=candidate, root=context["root"]) as model:
        candidate_evidence(model=model, policy=spec["policy"])
        return {"loaded_identity": model.loaded_identity(), "release_authorized": False}


def _score(context: BridgeContext) -> dict[str, Any]:
    from ..proof_pipeline import score
    return score(context["document"], role=context["role"], root=context["root"],
                 registry=context["profile"].registry)


def _rescore(context: BridgeContext) -> dict[str, Any]:
    from ..cord_receipt_plugin import receipt_metrics_v2
    rows = context["document"].get("rows")
    if not isinstance(rows, list) or not rows or len(rows) > context["limits"]["steps"]:
        fail("invalid_input", "Receipt rescore requires a bounded inline row artifact.")
    measured = [receipt_metrics_v2(reference=row["reference"], response=row["response"],
                                   output_valid=row.get("output_valid", True)) for row in rows]
    return {"rows": measured, "metric_version": "cord-receipt-v2", "model_calls": False,
            "steps": len(rows), "release_authorized": False}


BRIDGES: dict[str, Callable[[BridgeContext], dict[str, Any]]] = {
           "proof.train": _train, "proof.export": _export, "proof.identity": _identity,
           "proof.score": _score, "proof.offline_cordial_rescore": _rescore}


def dispatch(operation: str, profile: TrustedRuntimeProfile, *, document: dict[str, Any], role: str,
             run_id: str, plan_digest: str, approval: dict[str, Any],
             limits: dict[str, int]) -> dict[str, Any]:
    capability = CAPABILITIES.get(operation)
    if (not profile.qualified or capability not in profile.deployment_capabilities
            or approval.get("plan_digest") != plan_digest or not approval.get("id")):
        fail("runtime_not_authorized", "Bridge requires qualification, deployment capability and exact approval.", 403)
    identifier(run_id, "run ID")
    identifier(role, "candidate role")
    if "product_binding" in document:
        binding = document["product_binding"]
        gyms = [asset for asset in binding["assets"] if asset["kind"] == "gym"]
        if not gyms or role != binding["role"]:
            fail("runtime_not_qualified", "Product proof binding requires its exact qualified role and gyms.", 409)
        if profile.descriptor:
            from .local_runtime import validate_compiled_proof
            validate_compiled_proof(profile, document, limits)
        for gym in ([] if profile.descriptor else gyms):
            require_product_gym_binding(
                profile, project_id=binding["project_id"], gym_asset_id=gym["id"], gym_version=gym["version"],
                gym_artifact_id=gym["artifact_id"], proof_gym_reference=document["plugins"]["gym"],
                proof_gym_configuration=document["gym"], method=binding["method"])
    context: BridgeContext = {"profile": profile, "document": document, "role": role,
               "root": profile.root.resolve() / run_id, "limits": limits, "operation": operation}
    callback = profile.callbacks.get(operation) or BRIDGES[operation]
    # Callbacks are Python callables installed by trusted deployment code, never import strings.
    result = callback(context)
    if not isinstance(result, dict):
        fail("bridge_result_error", "Existing authority bridge did not return a JSON document.", 409)
    return {"execution": "completed", "quality": "not_evaluated", **result,
            "authority": "existing_proof_backend", "runtime_profile": profile.id,
            "release_authorized": False}
