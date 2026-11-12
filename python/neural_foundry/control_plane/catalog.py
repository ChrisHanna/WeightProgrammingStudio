"""Static inspection descriptors: importing this module loads no execution libraries."""

from copy import deepcopy
from typing import Any

OPERATIONS: dict[str, dict[str, Any]] = {
    name: {"id": name, "version": "1", "enabled": True,
           "capabilities": [capability], "side_effect_class": "local_artifact_write",
           "runtime_profile": "cpu-software-fixture-v1", "model_calls": False,
           "budgets": {"steps": 256, "episodes": 16, "wall_seconds": 20, "recorded_bytes": 262144}}
    for name, capability in (
        ("software.local_eval", "task_adapter_one_shot"),
        ("software.provider", "software_transport_fixture"),
        ("software.training", "cpu_tensor_transform"),
        ("software.export", "json_tensor_export"),
        ("software.identity", "content_identity"),
        ("software.eval", "versioned_metrics"),
        ("software.compare", "versioned_metric_comparison"),
        ("software.offline_rescore", "offline_scoring"),
        ("software.episode", "stateful_episode"),
        ("software.workflow", "tool_workflow"),
        ("software.weight_edit", "experimental_tensor_correctness"),
    )
}
for _id in ("live.proof", "live.provider", "live.training", "proof.train", "proof.export",
            "proof.identity", "proof.score", "proof.offline_cordial_rescore"):
    OPERATIONS[_id] = {"id": _id, "version": "1", "enabled": False,
                       "capabilities": ["external_authority_reference"],
                       "side_effect_class": "external_compute",
                       "runtime_profile": "explicit-trusted-profile-required",
                       "config_schema": {
                           "type": "object", "additionalProperties": False,
                           "required": ["runtime_profile", "artifact_id", "role"],
                           "properties": {"runtime_profile": {"type": "string"},
                                          "artifact_id": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
                                          "role": {"type": "string"}}}}

_CONFIG = {
    "inputs": {"type": "array", "minItems": 1, "maxItems": 64,
               "items": {"type": "string", "maxLength": 1024}},
    "outputs": {"type": "array", "minItems": 1, "maxItems": 64,
                "items": {"type": "string", "maxLength": 1024}},
    "steps": {"type": "integer", "minimum": 1, "maximum": 64},
    "learning_rate": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
    "strength": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
    "target": {"type": "integer", "minimum": 1, "maximum": 32},
    "seeds": {"type": "array", "minItems": 1, "maxItems": 16,
              "items": {"type": "integer", "minimum": 0, "maximum": 2147483647}},
}
for _name, _fields in (
    ("local_eval", ("inputs", "outputs")), ("provider", ("inputs", "outputs")),
    ("eval", ("inputs", "outputs")), ("offline_rescore", ("inputs", "outputs")),
    ("training", ("steps", "learning_rate")), ("export", ()), ("identity", ()), ("compare", ()),
    ("episode", ("target", "seeds")), ("workflow", ("seeds",)),
    ("weight_edit", ("strength",)),
):
    OPERATIONS["software." + _name]["config_schema"] = {
        "type": "object", "properties": {k: _CONFIG[k] for k in _fields},
        "additionalProperties": False,
    }


def catalog_document() -> dict[str, Any]:
    def template(name: str, operations: list[str]) -> dict[str, Any]:
        return {"id": name, "name": name.replace("-", " ").title(),
                "description": "Bounded CPU software fixture; no model inference or release claim.",
                "spec": {"version": 1, "name": name, "purpose": "development",
                         "candidates": [{"id": "candidate-a", "model": {
                             "id": "software-fixture", "revision": "1"}, "method": "fixture"}],
                         "stages": [{"id": f"stage-{i}", "operation": "software." + op,
                                     "depends_on": [f"stage-{i-1}"] if i else [],
                                     "candidate_id": "candidate-a", "config": {}, "limits": {}}
                                    for i, op in enumerate(operations)]}}
    from ..proof_dependencies import ROOT
    references: dict[str, dict[str, Any]] = {}
    registry_status = {
        "available": False,
        "reason": "Repository-backed proof dependencies require a qualified runtime.",
    }
    if (ROOT / "python" / "neural_foundry").is_dir():
        from ..proof_registry import builtin_registry
        _, references = builtin_registry()
        registry_status = {"available": True, "reason": "Repository dependency bundle present."}
    return deepcopy({"version": 1, "plugins": list(OPERATIONS.values()),
                     "proof_registry_references": references,
                     "proof_registry_status": registry_status, "templates": [
        template("local-evaluation", ["local_eval"]),
        template("provider-software-transport", ["provider", "offline_rescore"]),
        template("software-training-export", ["training", "export", "identity", "eval"]),
        template("stateful-counter", ["episode"]),
        template("software-workflow", ["workflow"]),
        template("experimental-tensor-workflow", ["weight_edit", "export", "identity"]),
    ]})
