"""Versioned proof plugins, deliberately independent of study/custody."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .gym_job import GymJobError
from .gym_lifecycle import AdapterIdentity
from .gym_service import _document_digest
from .provenance import sha256_file


class ProofRegistry:
    """Resolve exact implementation identities; never silently select a latest plugin."""

    kinds = frozenset({"gym", "model", "provider", "trainer", "exporter", "augmenter",
                       "public_audit", "server", "evidence", "training_audit"})

    def __init__(self) -> None:
        self._plugins: dict[tuple[str, str, str], tuple[str, Callable[..., Any], dict[str, str]]] = {}

    def register(self, kind: str, identity: AdapterIdentity, factory: Callable[..., Any], *,
                 dependencies: Mapping[str, str] | None = None) -> dict[str, Any]:
        identity.document()
        key = (kind, identity.adapter_id, identity.version)
        if kind not in self.kinds or key in self._plugins or not callable(factory):
            raise GymJobError("invalid or duplicate proof plugin")
        manifest = dict(dependencies or {})
        self._plugins[key] = (identity.implementation_digest, factory, dict(manifest))
        reference = identity.document()
        return {**reference, **({"dependencies": dict(manifest)} if dependencies is not None else {})}

    def resolve(self, kind: str, reference: Mapping[str, Any]) -> Callable[..., Any]:
        if kind not in self.kinds or not isinstance(reference, Mapping):
            raise GymJobError("unknown versioned proof plugin")
        try:
            digest, factory, dependencies = self._plugins[(kind, reference["adapter_id"], reference["version"])]
        except KeyError as error:
            raise GymJobError("unknown versioned proof plugin") from error
        if digest != reference.get("implementation_digest"):
            raise GymJobError("proof plugin implementation changed")
        if reference.get("dependencies", {}) != dependencies:
            raise GymJobError("proof plugin declared dependencies changed")
        root = Path(__file__).resolve().parents[2]
        for name, expected in dependencies.items():
            relative = PurePosixPath(name)
            path = root.joinpath(*relative.parts)
            if (relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name
                or not path.resolve().is_relative_to(root.resolve())):
                raise GymJobError("invalid proof plugin dependency path")
            try:
                if sha256_file(path) != expected:
                    raise GymJobError("proof plugin dependency changed")
            except OSError as error:
                raise GymJobError("proof plugin dependency missing") from error
        return factory


def builtin_registry() -> tuple[ProofRegistry, dict[str, dict[str, Any]]]:
    from . import proof_plugins
    from .proof_dependencies import builtin_dependencies

    registry = ProofRegistry()
    references = {}
    def register(kind: str, name: str, factory: Callable[..., Any], dependency_kind: str) -> dict[str, Any]:
        dependencies = builtin_dependencies(dependency_kind)
        digest = _document_digest({"factory_module_digest": sha256_file(proof_plugins.__file__),
                                   "dependencies": dependencies})
        return registry.register(kind, AdapterIdentity(name, "2", digest), factory, dependencies=dependencies)
    for kind, name, factory in (
        ("gym", "reference-invoice", proof_plugins.invoice_task),
        ("model", "transformers", proof_plugins.training_model),
        ("trainer", "automodel", proof_plugins.automodel_trainer),
        ("training_audit", "automodel-ordered-coverage", proof_plugins.automodel_coverage),
        ("provider", "openai", proof_plugins.openai_provider),
        ("exporter", "serving-full", proof_plugins.serving_export),
        ("server", "vllm", proof_plugins.vllm_server),
        ("augmenter", "balanced-authority", proof_plugins.balanced_augmentation),
        ("public_audit", "json-overlap", proof_plugins.public_overlap_audit),
        ("evidence", "candidate-weights", proof_plugins.candidate_evidence),
    ):
        references[kind] = register(kind, name, factory, kind)
    references["api_evidence"] = register("evidence", "openai-receipts", proof_plugins.api_evidence, "provider")
    references["recorded_provider"] = register(
        "provider", "recorded-public-responses", proof_plugins.recorded_provider, "recorded")
    references["recorded_evidence"] = register(
        "evidence", "recorded-api-receipts", proof_plugins.recorded_evidence, "recorded")
    return registry, references
