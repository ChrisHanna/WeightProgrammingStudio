"""Optional lifecycle backend packages and their versioned factories."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from importlib import import_module, metadata
from typing import Any

from ..gym_job import GymJobError


@dataclass(frozen=True)
class BackendFactory:
	backend_id: str
	version: str
	dependencies: Mapping[str, str]
	create: Callable[..., Any]
	availability: Callable[..., bool] | None = None


class BackendRegistry:
	def __init__(self) -> None:
		self._factories: dict[str, BackendFactory] = {}

	def register(self, factory: BackendFactory) -> None:
		if (not isinstance(factory.backend_id, str) or not factory.backend_id or "@" in factory.backend_id
			or not isinstance(factory.version, str) or not factory.version or "@" in factory.version
			or not isinstance(factory.dependencies, Mapping) or not callable(factory.create)):
			raise GymJobError("backend factory requires an ID, version, and callable")
		if any(not isinstance(name, str) or not name or not isinstance(version, str) or not version
			   for name, version in factory.dependencies.items()):
			raise GymJobError("backend dependencies require exact version pins")
		selector = f"{factory.backend_id}@{factory.version}"
		if selector in self._factories:
			raise GymJobError("backend factory is already registered")
		self._factories[selector] = factory

	@staticmethod
	def _missing(factory: BackendFactory) -> dict[str, str | None]:
		mismatches = {}
		for name, expected in factory.dependencies.items():
			try:
				actual = metadata.version(name)
			except metadata.PackageNotFoundError:
				actual = None
			if actual != expected:
				mismatches[name] = actual
		return mismatches

	def available(self) -> list[str]:
		available = []
		for selector, factory in self._factories.items():
			with suppress(Exception):
				if not self._missing(factory) and (factory.availability is None or factory.availability()):
					available.append(selector)
		return sorted(available)

	def require(self, selector: str) -> BackendFactory:
		factory = self._factories.get(selector)
		if factory is None:
			raise GymJobError("unknown backend or missing explicit backend version")
		mismatches = self._missing(factory)
		if mismatches:
			raise GymJobError(f"backend dependency version mismatch: {mismatches}")
		return factory

	def create(self, selector: str, **configuration: Any) -> Any:
		factory = self.require(selector)
		if factory.availability is not None and not factory.availability(**configuration):
			raise GymJobError("backend runtime is missing or has incompatible versions")
		adapter = factory.create(**configuration)
		if (getattr(getattr(adapter, "identity", None), "adapter_id", None) != factory.backend_id
			or getattr(adapter.identity, "version", None) != factory.version):
			raise GymJobError("backend factory returned an unexpected adapter identity")
		return adapter


registry = BackendRegistry()


def discover_backends() -> list[str]:
	import_module(f"{__name__}.peft")
	import_module(f"{__name__}.automodel")
	return registry.available()