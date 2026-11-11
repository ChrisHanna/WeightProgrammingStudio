"""Versioned proof plugins, deliberately independent of study/custody."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .gym_job import GymJobError
from .gym_lifecycle import AdapterIdentity
from .provenance import sha256_file


class ProofRegistry:
    """Resolve exact implementation identities; never silently select a latest plugin."""

    kinds = frozenset({"gym", "model", "provider", "trainer", "exporter", "augmenter",
                       "public_audit", "server", "evidence", "training_audit"})

    def __init__(self) -> None:
        self._plugins: dict[tuple[str, str, str], tuple[str, Callable[..., Any]]] = {}

    def register(self, kind: str, identity: AdapterIdentity, factory: Callable[..., Any]) -> dict[str, str]:
        identity.document()
        key = (kind, identity.adapter_id, identity.version)
        if kind not in self.kinds or key in self._plugins or not callable(factory):
            raise GymJobError("invalid or duplicate proof plugin")
        self._plugins[key] = (identity.implementation_digest, factory)
        return identity.document()

    def resolve(self, kind: str, reference: Mapping[str, str]) -> Callable[..., Any]:
        if kind not in self.kinds or not isinstance(reference, Mapping):
            raise GymJobError("unknown versioned proof plugin")
        try:
            digest, factory = self._plugins[(kind, reference["adapter_id"], reference["version"])]
        except KeyError as error:
            raise GymJobError("unknown versioned proof plugin") from error
        if digest != reference.get("implementation_digest"):
            raise GymJobError("proof plugin implementation changed")
        return factory


def builtin_registry() -> tuple[ProofRegistry, dict[str, dict[str, str]]]:
    from . import proof_plugins

    registry = ProofRegistry()
    references = {}
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
        references[kind] = registry.register(kind, AdapterIdentity(name, "1", sha256_file(proof_plugins.__file__)), factory)
    references["api_evidence"] = registry.register("evidence",
        AdapterIdentity("openai-receipts", "1", sha256_file(proof_plugins.__file__)),
        proof_plugins.api_evidence)
    references["recorded_provider"] = registry.register("provider",
        AdapterIdentity("recorded-public-responses", "1", sha256_file(proof_plugins.__file__)),
        proof_plugins.recorded_provider)
    references["recorded_evidence"] = registry.register("evidence",
        AdapterIdentity("recorded-api-receipts", "1", sha256_file(proof_plugins.__file__)),
        proof_plugins.recorded_evidence)
    return registry, references
