"""Strict, JSON-only public contracts and immutable document fingerprints."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, NoReturn


class ControlPlaneError(Exception):
    def __init__(self, code: str, message: str, status: int = 400) -> None:
        self.code, self.message, self.status = code, message, status
        super().__init__(message)


def fail(code: str, message: str, status: int = 400) -> NoReturn:
    raise ControlPlaneError(code, message, status) from None


def encode(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError, RecursionError):
        fail("invalid_json", "A finite JSON document is required.")


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(encode(value)).hexdigest()


def document(value: Any) -> Any:
    raw = encode(value)
    if len(raw) > 262_144:
        fail("document_too_large", "JSON document exceeds the inline size limit.")
    return json.loads(raw)


def identifier(value: Any, field: str = "identifier") -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value):
        fail("invalid_identifier", f"Invalid {field}.")
    return value


def actor_name(value: Any) -> str:
    # Authentication/authorization belong to the transport. This is audit data.
    return identifier(value, "actor")


def keys(value: Any, allowed: set[str], required: set[str]) -> None:
    if not isinstance(value, dict) or set(value) - allowed or required - set(value):
        fail("invalid_spec", "Document has missing or unsupported fields.")


DEFAULT_LIMITS = {"steps": 64, "episodes": 4, "wall_seconds": 20, "recorded_bytes": 131072}
MAX_LIMITS = {"steps": 256, "episodes": 16, "wall_seconds": 20, "recorded_bytes": 262144}


def limits(value: Any, *, defaults: bool = True,
           ceilings: dict[str, int] | None = None) -> dict[str, int]:
    maximum = ceilings if ceilings is not None else MAX_LIMITS
    keys(value, set(maximum), set())
    result = dict(ceilings if ceilings is not None else DEFAULT_LIMITS) if defaults else {}
    for key, number in value.items():
        if type(number) is not int or not 1 <= number <= maximum[key]:
            fail("invalid_limits", "Limits must be positive bounded integers.")
        result[key] = number
    return result


def validate_spec(value: Any, registry: dict[str, dict[str, Any]]) -> dict[str, Any]:
    spec = document(value)
    if not isinstance(spec, dict):
        fail("invalid_spec", "Run specification must be a JSON object.")
    keys(spec, {"version", "name", "purpose", "candidates", "stages"}, {"version", "name", "candidates", "stages"})
    if type(spec["version"]) is not int or spec["version"] != 1 or spec.get("purpose", "development") != "development":
        fail("invalid_spec", "Only version 1 development specifications are supported.")
    if not isinstance(spec["name"], str) or not 1 <= len(spec["name"]) <= 128:
        fail("invalid_spec", "A bounded run name is required.")
    candidates = spec["candidates"]
    stages = spec["stages"]
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 16:
        fail("invalid_spec", "Specify between one and sixteen candidates.")
    if not isinstance(stages, list) or not 1 <= len(stages) <= 64:
        fail("invalid_spec", "Specify between one and sixty-four stages.")
    candidate_ids = set()
    for candidate in candidates:
        keys(candidate, {"id", "model", "method"}, {"id", "model", "method"})
        cid = identifier(candidate["id"], "candidate ID")
        if cid in candidate_ids:
            fail("invalid_spec", "Candidate IDs must be unique.")
        candidate_ids.add(cid)
        keys(candidate["model"], {"id", "revision"}, {"id", "revision"})
        for val in (*candidate["model"].values(), candidate["method"]):
            if not isinstance(val, str) or not 1 <= len(val) <= 256:
                fail("invalid_spec", "Versioned model references and methods must be bounded strings.")
    ids = set()
    for stage in stages:
        keys(stage, {"id", "operation", "depends_on", "candidate_id", "config", "limits"},
             {"id", "operation", "depends_on", "config", "limits"})
        sid = identifier(stage["id"], "stage ID")
        if sid in ids:
            fail("invalid_spec", "Stage IDs must be unique.")
        ids.add(sid)
        if not isinstance(stage["operation"], str) or stage["operation"] not in registry:
            fail("unknown_operation", "Operation is not registered.")
        if not registry[stage["operation"]]["enabled"]:
            fail("capability_disabled", "Operation requires a separately trusted runtime profile.")
        if stage.get("candidate_id") is not None and (
                not isinstance(stage["candidate_id"], str) or stage["candidate_id"] not in candidate_ids):
            fail("invalid_spec", "Stage references an unknown candidate.")
        deps = stage["depends_on"]
        if not isinstance(deps, list) or any(not isinstance(d, str) for d in deps) or len(set(deps)) != len(deps):
            fail("invalid_spec", "Dependencies must be distinct stage IDs.")
        validate_config(stage["operation"], stage["config"])
        profile = stage["config"].get("runtime_profile")
        scoped = registry[stage["operation"]].get("runtime_limits", {}).get(profile)
        limits(stage["limits"], ceilings=scoped)
    done: set[str] = set()
    while len(done) < len(ids):
        ready = {s["id"] for s in stages if s["id"] not in done and set(s["depends_on"]) <= done}
        if not ready:
            fail("invalid_dag", "Stage dependencies contain a cycle or unknown stage.")
        done |= ready
    spec.setdefault("purpose", "development")
    return spec


def validate_config(operation: str, config: Any) -> None:
    if operation == "analysis.saved_json":
        from .product import POLICY
        fields = {"metric_policy", "example_count"}
        fields |= {prefix + suffix for prefix in ("gym", "dataset", "evidence", "model")
                   for suffix in ("_artifact_id", "_asset_id", "_version")}
        keys(config, fields, fields)
        if config["metric_policy"] != POLICY or type(config["example_count"]) is not int or not 1 <= config["example_count"] <= 256:
            fail("invalid_config", "Recorded analysis requires the supported metric policy and bounded row count.")
        for prefix in ("gym", "dataset", "evidence", "model"):
            identifier(config[prefix + "_asset_id"], "asset ID")
            if not isinstance(config[prefix + "_artifact_id"], str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", config[prefix + "_artifact_id"]):
                fail("invalid_config", "Recorded analysis inputs require pinned artifact digests.")
            if type(config[prefix + "_version"]) is not int or config[prefix + "_version"] < 1:
                fail("invalid_config", "Asset versions must be positive integers.")
        return
    if operation.startswith("proof."):
        keys(config, {"runtime_profile", "artifact_id", "role"}, {"runtime_profile", "artifact_id", "role"})
        for field in ("runtime_profile", "role"):
            identifier(config[field], field)
        if not isinstance(config["artifact_id"], str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", config["artifact_id"]):
            fail("invalid_config", "Live bridges require a content-addressed input artifact.")
        return
    allowed = {
        "software.local_eval": {"inputs", "outputs"},
        "software.provider": {"inputs", "outputs"},
        "software.eval": {"inputs", "outputs"},
        "software.offline_rescore": {"inputs", "outputs"},
        "software.training": {"steps", "learning_rate"},
        "software.export": set(), "software.identity": set(),
        "software.compare": set(),
        "software.episode": {"target", "seeds"},
        "software.workflow": {"seeds"},
        "software.weight_edit": {"strength"},
    }
    keys(config, allowed[operation], set())
    for name in ("inputs", "outputs"):
        if name in config and (not isinstance(config[name], list) or not 1 <= len(config[name]) <= 64
                               or any(not isinstance(x, str) or len(x) > 1024 for x in config[name])):
            fail("invalid_config", "Fixture inputs and outputs must be bounded text arrays.")
    if "outputs" in config and len(config["outputs"]) != len(config.get("inputs", ["{\"value\":1}"])):
        fail("invalid_config", "Fixture input and output counts must match.")
    for name, high in (("steps", 64), ("target", 32)):
        if name in config and (type(config[name]) is not int or not 1 <= config[name] <= high):
            fail("invalid_config", "Fixture integer configuration exceeds its bound.")
    for name in ("learning_rate", "strength"):
        if name in config and (type(config[name]) not in (int, float) or not 0 < config[name] <= 1):
            fail("invalid_config", "Fixture numeric configuration exceeds its bound.")
    if "seeds" in config and (not isinstance(config["seeds"], list) or not 1 <= len(config["seeds"]) <= 16
                              or any(type(x) is not int or not 0 <= x < 2**31 for x in config["seeds"])):
        fail("invalid_config", "Fixture seeds must be bounded nonnegative integers.")


def compile_plan(spec: dict[str, Any], *,
                 runtime_descriptors: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    from .implementation import execution_implementation

    descriptors = runtime_descriptors or {}
    stages = [{**s, "limits": limits(s["limits"], ceilings=(
        {"steps": 8, "episodes": 1, "wall_seconds": descriptors[s["config"]["runtime_profile"]]["limits"]["wall_seconds"],
         "recorded_bytes": 262144} if s["operation"].startswith("proof.")
        and s["config"]["runtime_profile"] in descriptors else None))} for s in spec["stages"]]
    software_only = all(s["operation"].startswith(("software.", "analysis.")) for s in stages)
    runtime_profile = ("cpu-recorded-analysis-v1" if all(s["operation"].startswith("analysis.") for s in stages)
                       else "cpu-software-fixture-v1" if software_only else "per-stage-qualified-profile")
    plan = {"version": 1, "spec_digest": digest(spec), "stages": stages,
            "runtime_profile": runtime_profile,
            "model_calls": False if software_only else "possible",
            "limits": {key: sum(s["limits"][key] for s in stages) for key in DEFAULT_LIMITS},
            "release_authorized": False, "execution_implementation": execution_implementation()}
    if descriptors:
        selected = {s["config"]["runtime_profile"] for s in stages if s["operation"].startswith("proof.")}
        bound_profiles = selected & descriptors.keys()
        if bound_profiles:
            if any(not s["operation"].startswith("proof.")
                   or s["config"]["runtime_profile"] not in bound_profiles for s in stages):
                fail("mixed_runtime_plan", "Scoped host proof plans cannot mix software or legacy runtimes.")
            plan["runtime_implementations"] = {name: digest(descriptors[name]) for name in sorted(bound_profiles)}
    for stage in stages:
        config, bound = stage["config"], stage["limits"]
        required_steps = config.get("steps", 8 if stage["operation"] == "software.training" else 1)
        if stage["operation"] == "analysis.saved_json":
            required_steps = config["example_count"]
        if stage["operation"] in {"software.local_eval", "software.provider", "software.eval", "software.offline_rescore"}:
            required_steps = len(config.get("inputs", ['{"value":1}']))
        if stage["operation"] == "software.workflow":
            required_steps = 2 * len(config.get("seeds", [0]))
        required_episodes = len(config.get("seeds", [0]))
        if required_steps > bound["steps"] or required_episodes > bound["episodes"]:
            fail("budget_exceeded", "Requested fixture work exceeds compiled stage limits.")
    return {**plan, "digest": digest(plan)}
