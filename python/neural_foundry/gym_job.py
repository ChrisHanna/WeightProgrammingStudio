"""Versioned, data-only job contracts for dashboard-independent Gym workflows."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Literal

from .records import canonicalize, content_digest

GYM_JOB_SCHEMA_VERSION = 1
GYM_JOB_KIND = "nf_gym_job"
MAX_GYM_JOB_BYTES = 1_048_576
MAX_JOB_COMPONENTS = 256
MAX_JOB_PARAMETERS_BYTES = 262_144
MAX_SCHEMA_DEPTH = 12
MAX_SCHEMA_NODES = 256
_JSON_TYPES = (str, int, float, bool, type(None))
_SCHEMA_TYPES = {"array", "boolean", "integer", "number", "object", "string"}
_SCHEMA_FIELDS = {
    "type", "enum", "properties", "required", "additionalProperties", "items",
    "minItems", "maxItems", "minLength", "maxLength", "minimum", "maximum",
}
_SPLITS = {"train", "validation", "holdout"}
_GATE_SPLITS = {"validation", "holdout", "safety", "controls", "reproducibility"}
_COMPARATORS = {"eq", "ge", "gt", "le", "lt"}
_GATE_PURPOSES = {"task", "behavior", "controls", "reproducibility", "safety"}


class GymJobError(ValueError):
    """A Gym job contract is malformed, unbounded, or references unknown components."""


def is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _freeze_json(value: Any, where: str, *, depth: int = 0) -> Any:
    if depth > MAX_SCHEMA_DEPTH:
        raise GymJobError(f"{where} exceeds the JSON depth bound")
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise GymJobError(f"{where} has a non-string key")
        return MappingProxyType(
            {key: _freeze_json(item, f"{where}.{key}", depth=depth + 1) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, where, depth=depth + 1) for item in value)
    if type(value) in (int, float) and not is_finite_number(value):
        raise GymJobError(f"{where} must not contain non-finite numbers")
    if value is None or type(value) in _JSON_TYPES:
        return value
    raise GymJobError(f"{where} contains unsupported value type {type(value).__name__}")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _nonempty(value: Any, field: str, *, limit: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.encode("utf-8")) > limit:
        raise GymJobError(f"{field} must be a non-empty bounded string")
    return value.strip()


def _parameters(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GymJobError(f"{field} must be an object")
    frozen = _freeze_json(value, field)
    try:
        encoded = json.dumps(_thaw(frozen), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError, RecursionError) as exc:
        raise GymJobError(f"{field} is not bounded JSON data") from exc
    if len(encoded.encode("utf-8")) > MAX_JOB_PARAMETERS_BYTES:
        raise GymJobError(f"{field} exceeds the parameter byte limit")
    return frozen


def _validate_schema(schema: Any, field: str) -> None:
    nodes = 0

    def visit(value: Any, where: str, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > MAX_SCHEMA_NODES or depth > MAX_SCHEMA_DEPTH:
            raise GymJobError(f"{field} exceeds its schema depth or node bound")
        if not isinstance(value, Mapping) or set(value) - _SCHEMA_FIELDS:
            raise GymJobError(f"{where} contains unsupported schema fields")
        kind = value.get("type")
        if not isinstance(kind, str) or kind not in _SCHEMA_TYPES:
            raise GymJobError(f"{where}.type must be one of {sorted(_SCHEMA_TYPES)}")
        count_bounds = {"minItems", "maxItems", "minLength", "maxLength"}
        numeric_bounds = {"minimum", "maximum"}
        applicable_bounds = {
            "array": {"minItems", "maxItems"},
            "string": {"minLength", "maxLength"},
            "integer": numeric_bounds,
            "number": numeric_bounds,
            "object": set(),
            "boolean": set(),
        }[kind]
        for bound in (count_bounds | numeric_bounds) & set(value):
            bound_value = value[bound]
            if bound not in applicable_bounds:
                raise GymJobError(f"{where}.{bound} is not valid for {kind} schemas")
            if bound in count_bounds:
                if type(bound_value) is not int or bound_value < 0:
                    raise GymJobError(f"{where}.{bound} must be a non-negative integer")
            elif not is_finite_number(bound_value):
                raise GymJobError(f"{where}.{bound} must be a finite number")
        if "enum" in value:
            enum = value["enum"]
            if not isinstance(enum, (list, tuple)) or not enum:
                raise GymJobError(f"{where}.enum must be a non-empty list")
            _freeze_json(enum, f"{where}.enum", depth=depth)
        if kind == "object":
            properties, required = value.get("properties"), value.get("required", [])
            if not isinstance(properties, Mapping) or not isinstance(required, (list, tuple)):
                raise GymJobError(f"{where} object schemas require properties and required lists")
            if value.get("additionalProperties", False) is not False:
                raise GymJobError(f"{where} must reject undeclared object properties")
            if not all(isinstance(key, str) and key for key in properties):
                raise GymJobError(f"{where}.properties keys must be non-empty strings")
            if any(not isinstance(key, str) or key not in properties for key in required):
                raise GymJobError(f"{where}.required must name declared properties")
            for name, child in properties.items():
                visit(child, f"{where}.properties.{name}", depth + 1)
        elif kind == "array":
            if "items" not in value:
                raise GymJobError(f"{where} array schemas require an items schema")
            visit(value["items"], f"{where}.items", depth + 1)
        for low, high in (("minItems", "maxItems"), ("minLength", "maxLength"), ("minimum", "maximum")):
            if low in value and high in value and value[low] > value[high]:
                raise GymJobError(f"{where}.{low} must not exceed {high}")

    visit(schema, field, 0)


def validate_task_instance(schema: Mapping[str, Any], value: Any, *, field: str = "task") -> None:
    """Validate one JSON task value against the supported, closed schema subset."""
    _validate_schema(schema, f"{field}.schema")

    def visit(spec: Mapping[str, Any], item: Any, where: str) -> None:
        kind = spec["type"]
        if kind == "object":
            if not isinstance(item, Mapping):
                raise GymJobError(f"{where} must be an object")
            properties = spec["properties"]
            required = spec.get("required", [])
            if set(item) - set(properties):
                raise GymJobError(f"{where} contains undeclared properties")
            missing = set(required) - set(item)
            if missing:
                raise GymJobError(f"{where} is missing required properties: {sorted(missing)}")
            for name, child in properties.items():
                if name in item:
                    visit(child, item[name], f"{where}.{name}")
        elif kind == "array":
            if not isinstance(item, (list, tuple)):
                raise GymJobError(f"{where} must be an array")
            if len(item) < spec.get("minItems", 0) or len(item) > spec.get("maxItems", 100_000):
                raise GymJobError(f"{where} item count is outside its schema bounds")
            for index, child in enumerate(item):
                visit(spec["items"], child, f"{where}[{index}]")
        elif kind == "string":
            if not isinstance(item, str):
                raise GymJobError(f"{where} must be a string")
            if len(item) < spec.get("minLength", 0) or len(item) > spec.get("maxLength", 16_384):
                raise GymJobError(f"{where} length is outside its schema bounds")
        elif kind == "integer":
            if type(item) is not int or not is_finite_number(item):
                raise GymJobError(f"{where} must be an integer")
            if item < spec.get("minimum", -math.inf) or item > spec.get("maximum", math.inf):
                raise GymJobError(f"{where} is outside its numeric bounds")
        elif kind == "number":
            if not is_finite_number(item):
                raise GymJobError(f"{where} must be a finite number")
            if item < spec.get("minimum", -math.inf) or item > spec.get("maximum", math.inf):
                raise GymJobError(f"{where} is outside its numeric bounds")
        elif kind == "boolean" and type(item) is not bool:
            raise GymJobError(f"{where} must be a boolean")
        if "enum" in spec:
            encoded = canonicalize({"value": _thaw(item)})
            if all(encoded != canonicalize({"value": _thaw(option)}) for option in spec["enum"]):
                raise GymJobError(f"{where} is outside its declared enum")

    visit(schema, value, field)


@dataclass(frozen=True)
class ScenarioSource:
    source_id: str
    provider_id: str
    provider_version: str
    provider_digest: str
    source_digest: str
    origin: Literal["real", "simulated"]
    split_roles: tuple[str, ...]
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        for field in ("source_id", "provider_id", "provider_version"):
            _nonempty(getattr(self, field), f"scenario.{field}")
        if not _is_digest(self.provider_digest) or not _is_digest(self.source_digest):
            raise GymJobError("scenario provider and source identities must be sha256 digests")
        if not isinstance(self.origin, str) or self.origin not in {"real", "simulated"}:
            raise GymJobError("scenario.origin must be 'real' or 'simulated'")
        if not isinstance(self.split_roles, (list, tuple)) or not self.split_roles or any(not isinstance(role, str) for role in self.split_roles) or len(set(self.split_roles)) != len(self.split_roles):
            raise GymJobError("scenario.split_roles must be non-empty and unique")
        if not set(self.split_roles) <= _SPLITS:
            raise GymJobError(f"scenario.split_roles must be within {sorted(_SPLITS)}")
        object.__setattr__(self, "parameters", _parameters(self.parameters, "scenario.parameters"))

    def to_document(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "provider_digest": self.provider_digest,
            "source_digest": self.source_digest,
            "origin": self.origin,
            "split_roles": list(self.split_roles),
            "parameters": _thaw(self.parameters),
        }


@dataclass(frozen=True)
class ComponentReference:
    component_id: str
    version: str
    implementation_digest: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        _nonempty(self.component_id, "component.component_id")
        _nonempty(self.version, "component.version")
        if not _is_digest(self.implementation_digest):
            raise GymJobError("component implementation_digest must be a sha256 digest")
        object.__setattr__(self, "parameters", _parameters(self.parameters, "component.parameters"))

    def to_document(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "version": self.version,
            "implementation_digest": self.implementation_digest,
            "parameters": _thaw(self.parameters),
        }


@dataclass(frozen=True)
class CandidateMethod:
    method_id: str
    version: str
    implementation_digest: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        _nonempty(self.method_id, "candidate_method.method_id")
        _nonempty(self.version, "candidate_method.version")
        if not _is_digest(self.implementation_digest):
            raise GymJobError("candidate method implementation_digest must be a sha256 digest")
        object.__setattr__(self, "parameters", _parameters(self.parameters, "candidate_method.parameters"))

    def to_document(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "version": self.version,
            "implementation_digest": self.implementation_digest,
            "parameters": _thaw(self.parameters),
        }


@dataclass(frozen=True)
class AcceptanceGate:
    gate_id: str
    purpose: str
    metric_id: str
    split: str
    comparator: str
    threshold: float
    required: bool = True

    def __post_init__(self) -> None:
        _nonempty(self.gate_id, "gate.gate_id")
        if not isinstance(self.purpose, str) or self.purpose not in _GATE_PURPOSES:
            raise GymJobError(f"gate.purpose must be one of {sorted(_GATE_PURPOSES)}")
        _nonempty(self.metric_id, "gate.metric_id")
        if not isinstance(self.split, str) or self.split not in _GATE_SPLITS:
            raise GymJobError(f"gate.split must be one of {sorted(_GATE_SPLITS)}")
        if not isinstance(self.comparator, str) or self.comparator not in _COMPARATORS:
            raise GymJobError(f"gate.comparator must be one of {sorted(_COMPARATORS)}")
        if not is_finite_number(self.threshold):
            raise GymJobError("gate.threshold must be finite numeric data")
        if type(self.required) is not bool:
            raise GymJobError("gate.required must be a boolean")
        expected_splits = {
            "task": {"validation", "holdout"},
            "behavior": {"validation", "holdout"},
            "controls": {"controls"},
            "reproducibility": {"reproducibility"},
            "safety": {"safety", "holdout"},
        }
        if self.split not in expected_splits[self.purpose]:
            raise GymJobError(f"gate.split is incompatible with purpose {self.purpose!r}")

    def to_document(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "purpose": self.purpose,
            "metric_id": self.metric_id,
            "split": self.split,
            "comparator": self.comparator,
            "threshold": self.threshold,
            "required": self.required,
        }


@dataclass(frozen=True)
class GymJobSpec:
    job_id: str
    title: str
    description: str
    task_input_schema: Mapping[str, Any]
    task_output_schema: Mapping[str, Any]
    scenario_sources: tuple[ScenarioSource, ...]
    scorer: ComponentReference
    verifier: ComponentReference
    candidate_methods: tuple[CandidateMethod, ...]
    gates: tuple[AcceptanceGate, ...]
    metadata: Mapping[str, Any]
    schema_version: int = GYM_JOB_SCHEMA_VERSION
    backends: Mapping[str, str] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != GYM_JOB_SCHEMA_VERSION:
            raise GymJobError(f"unsupported Gym job schema_version {self.schema_version!r}")
        _nonempty(self.job_id, "job_id")
        _nonempty(self.title, "title")
        _nonempty(self.description, "description", limit=4096)
        if not self.scenario_sources or not self.candidate_methods or not self.gates:
            raise GymJobError("a Gym job requires scenario sources, candidate methods, and acceptance gates")
        for label, values, names in (
            ("scenario_sources", self.scenario_sources, [row.source_id for row in self.scenario_sources]),
            ("candidate_methods", self.candidate_methods, [row.method_id for row in self.candidate_methods]),
            ("gates", self.gates, [row.gate_id for row in self.gates]),
        ):
            if len(values) > MAX_JOB_COMPONENTS or len(names) != len(set(names)):
                raise GymJobError(f"{label} must be bounded and have unique IDs")
        scenario_splits = {split for source in self.scenario_sources for split in source.split_roles}
        if scenario_splits != _SPLITS:
            raise GymJobError(f"scenario sources must cover exactly {sorted(_SPLITS)}")
        required_purposes = {"task", "behavior", "controls", "reproducibility"}
        missing_purposes = required_purposes - {
            gate.purpose for gate in self.gates if gate.required
        }
        if missing_purposes:
            raise GymJobError(f"required gates are missing purposes: {sorted(missing_purposes)}")
        _validate_schema(self.task_input_schema, "task_input_schema")
        _validate_schema(self.task_output_schema, "task_output_schema")
        if not isinstance(self.backends, Mapping) or set(self.backends) - {"task", "model", "trainer", "evaluator"}:
            raise GymJobError("job backends must name supported adapter roles")
        for selector in self.backends.values():
            if (not isinstance(selector, str) or selector.count("@") != 1
                or any(not part or part != part.strip() for part in selector.split("@"))):
                raise GymJobError("job backends require exact adapter ID and version selectors")
        object.__setattr__(self, "task_input_schema", _parameters(self.task_input_schema, "task_input_schema"))
        object.__setattr__(self, "task_output_schema", _parameters(self.task_output_schema, "task_output_schema"))
        object.__setattr__(self, "metadata", _parameters(self.metadata, "metadata"))
        object.__setattr__(self, "backends", _parameters(self.backends, "backends"))
        encoded = json.dumps(self.to_document(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > MAX_GYM_JOB_BYTES:
            raise GymJobError("Gym job spec exceeds the byte limit")

    def to_document(self) -> dict[str, Any]:
        document = {
            "schema_version": self.schema_version,
            "kind": GYM_JOB_KIND,
            "job_id": self.job_id,
            "title": self.title,
            "description": self.description,
            "task": {
                "input_schema": _thaw(self.task_input_schema),
                "output_schema": _thaw(self.task_output_schema),
            },
            "scenario_sources": [row.to_document() for row in self.scenario_sources],
            "scorer": self.scorer.to_document(),
            "verifier": self.verifier.to_document(),
            "candidate_methods": [row.to_document() for row in self.candidate_methods],
            "gates": [row.to_document() for row in self.gates],
            "metadata": _thaw(self.metadata),
        }
        if self.backends:
            document["backends"] = _thaw(self.backends)
        return document

    @property
    def digest(self) -> str:
        return str(content_digest(canonicalize(self.to_document())))


class GymComponentRegistry:
    """Explicit in-process bindings; loading a job never imports or executes its references."""

    _KINDS: ClassVar[frozenset[str]] = frozenset(
        {"scenario_provider", "scorer", "verifier", "candidate_method"}
    )

    def __init__(self) -> None:
        self._components: dict[tuple[str, str, str, str], Any] = {}

    def register(
        self,
        kind: str,
        component_id: str,
        version: str,
        implementation_digest: str,
        implementation: Any,
    ) -> None:
        if kind not in self._KINDS:
            raise GymJobError(f"unsupported Gym component kind {kind!r}")
        _nonempty(component_id, "component_id")
        _nonempty(version, "version")
        if not _is_digest(implementation_digest):
            raise GymJobError("component implementation_digest must be a sha256 digest")
        key = (kind, component_id, version, implementation_digest)
        if key in self._components:
            raise GymJobError(f"Gym component {key!r} is already registered")
        if not callable(implementation):
            raise GymJobError("Gym component implementation must be callable")
        self._components[key] = implementation

    def resolve(self, kind: str, component_id: str, version: str, implementation_digest: str) -> Any:
        try:
            return self._components[(kind, component_id, version, implementation_digest)]
        except KeyError as exc:
            raise GymJobError(
                f"unregistered {kind} {component_id!r} version {version!r} implementation {implementation_digest!r}"
            ) from exc

    def validate_job(self, spec: GymJobSpec) -> None:
        for source in spec.scenario_sources:
            self.resolve(
                "scenario_provider", source.provider_id, source.provider_version, source.provider_digest
            )
        self.resolve("scorer", spec.scorer.component_id, spec.scorer.version, spec.scorer.implementation_digest)
        self.resolve("verifier", spec.verifier.component_id, spec.verifier.version, spec.verifier.implementation_digest)
        for method in spec.candidate_methods:
            self.resolve(
                "candidate_method", method.method_id, method.version, method.implementation_digest
            )

    def list_components(self) -> list[dict[str, str]]:
        return [
            {
                "kind": kind,
                "component_id": component_id,
                "version": version,
                "implementation_digest": implementation_digest,
            }
            for kind, component_id, version, implementation_digest in sorted(self._components)
        ]


def _object(
    value: Any,
    field: str,
    required: set[str],
    optional: set[str] | frozenset[str] = frozenset(),
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or required - set(value) or set(value) - required - optional:
        raise GymJobError(f"{field} has missing or unknown fields")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GymJobError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def gym_job_from_document(document: Mapping[str, Any]) -> GymJobSpec:
    root = _object(
        document,
        "job",
        {
            "schema_version", "kind", "job_id", "title", "description", "task",
            "scenario_sources", "scorer", "verifier", "candidate_methods", "gates", "metadata",
        },
        {"backends"},
    )
    if root["schema_version"] != GYM_JOB_SCHEMA_VERSION or root["kind"] != GYM_JOB_KIND:
        raise GymJobError("unsupported Gym job kind or schema_version")
    task = _object(root["task"], "task", {"input_schema", "output_schema"})

    def component(value: Any, field: str) -> ComponentReference:
        row = _object(value, field, {"component_id", "version", "implementation_digest", "parameters"})
        return ComponentReference(
            row["component_id"], row["version"], row["implementation_digest"], row["parameters"]
        )

    scenarios, methods, gates = root["scenario_sources"], root["candidate_methods"], root["gates"]
    if not isinstance(scenarios, list) or not scenarios:
        raise GymJobError("scenario_sources must be a non-empty list")
    if not isinstance(methods, list) or not methods:
        raise GymJobError("candidate_methods must be a non-empty list")
    if not isinstance(gates, list) or not gates:
        raise GymJobError("gates must be a non-empty list")
    scenario_rows = []
    for index, value in enumerate(scenarios):
        row = _object(
            value,
            f"scenario_sources[{index}]",
            {
                "source_id", "provider_id", "provider_version", "provider_digest", "source_digest",
                "origin", "split_roles", "parameters",
            },
        )
        if not isinstance(row["split_roles"], list) or not all(isinstance(item, str) for item in row["split_roles"]):
            raise GymJobError(f"scenario_sources[{index}].split_roles must be a list of strings")
        scenario_rows.append(ScenarioSource(
            row["source_id"], row["provider_id"], row["provider_version"], row["provider_digest"],
            row["source_digest"], row["origin"], tuple(row["split_roles"]), row["parameters"],
        ))
    method_rows = []
    for index, value in enumerate(methods):
        row = _object(
            value,
            f"candidate_methods[{index}]",
            {"method_id", "version", "implementation_digest", "parameters"},
        )
        method_rows.append(CandidateMethod(
            row["method_id"], row["version"], row["implementation_digest"], row["parameters"]
        ))
    gate_rows = []
    for index, value in enumerate(gates):
        row = _object(
            value,
            f"gates[{index}]",
            {"gate_id", "purpose", "metric_id", "split", "comparator", "threshold"},
            {"required"},
        )
        gate_rows.append(AcceptanceGate(
            row["gate_id"], row["purpose"], row["metric_id"], row["split"], row["comparator"],
            row["threshold"], row.get("required", True),
        ))
    if not isinstance(root["metadata"], Mapping):
        raise GymJobError("metadata must be an object")
    return GymJobSpec(
        job_id=root["job_id"],
        title=root["title"],
        description=root["description"],
        task_input_schema=task["input_schema"],
        task_output_schema=task["output_schema"],
        scenario_sources=tuple(scenario_rows),
        scorer=component(root["scorer"], "scorer"),
        verifier=component(root["verifier"], "verifier"),
        candidate_methods=tuple(method_rows),
        gates=tuple(gate_rows),
        metadata=root["metadata"],
        schema_version=root["schema_version"],
        backends=root.get("backends", {}),
    )


def load_gym_job_spec(path: str | Path) -> GymJobSpec:
    """Load a bounded, strict JSON job spec; component references are never imported."""
    source = Path(path)
    if source.stat().st_size > MAX_GYM_JOB_BYTES:
        raise GymJobError("Gym job spec exceeds the byte limit")
    try:
        document = json.loads(source.read_bytes(), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GymJobError(f"Gym job spec is unreadable: {exc}") from exc
    if not isinstance(document, Mapping):
        raise GymJobError("Gym job spec root must be an object")
    return gym_job_from_document(document)