"""WeightRepairBench v1 trusted, development-only component runner.

Supply compiled plans through ``prepare(model)``; this is trusted controller code,
not a gym plugin or authorization boundary. No training, network, final scoring,
export/reload, or promotion is dispatched. Task and result records are checked
by the registered Rust validator; this is structural, not scientific acceptance.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from . import _native, arithmetic, ir, surgeon

PROTOCOL = "weight-repair-benchmark-v1"
SCOPE = "synthetic_component_fixture_not_scientific_acceptance"
METHODS = (
    "original",
    "no_op",
    "best_known",
    "direct",
    "decoding",
    "activation",
    "lora",
    "sft",
    "distillation",
    "rl",
)
COST_UNITS = {
    "wall": "seconds",
    "cpu": "seconds",
    "gpu": "seconds",
    "host_peak": "bytes",
    "device_peak": "bytes",
    "bytes_read": "bytes",
    "bytes_written": "bytes",
    "bytes_stored": "bytes",
    "local_tokens": "tokens",
    "remote_tokens": "tokens",
    "quoted_spend": "USD",
    "reserved_spend": "USD",
    "actual_spend": "USD",
    "forwards": "count",
    "backwards": "count",
    "optimizer_steps": "count",
    "fitting_steps": "count",
    "inspected_parameters": "count",
    "changed_parameters": "count",
    "trainable_parameters": "count",
    "proposal_attempts": "count",
}
PHASES = (
    "setup",
    "diagnosis",
    "failed_edits",
    "search",
    "controls",
    "verification",
    "export",
    "reload",
    "recovery",
)
# Each value is (native compiled plan, native smoke-suite record input).
Prepare = Callable[[Any], Mapping[str, tuple[Any, dict[str, Any]]]]


def _wire(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, separators=(",", ":")).encode()


def _digest(value: Any) -> str:
    return str(_native.content_digest(_native.canonical_json(_wire(value))))


def _file_digest(path: Path) -> str:
    # Debug native extensions can exceed the native bytes-argument limit. Hash
    # actual bytes incrementally, with a separate bounded trusted-file reader.
    size = path.stat().st_size
    if size > 512 * 1024 * 1024:
        raise ValueError("fixture runtime file exceeds 512 MiB")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        remaining = size
        while remaining:
            block = stream.read(min(remaining, 1024 * 1024))
            if not block:
                raise ValueError("runtime file truncated during hashing")
            digest.update(block)
            remaining -= len(block)
        if stream.read(1):
            raise ValueError("runtime file grew during hashing")
    return "sha256:" + digest.hexdigest()


def validate(task: Mapping[str, Any], result: Mapping[str, Any] | None = None) -> bytes:
    """Authoritative native structural check, never a receipt authentication.

    Never silently downgrade to a Python validator when registration is absent.
    Raw-byte callers should use the native endpoint to test duplicate JSON keys.
    """
    checker = getattr(_native, "validate_repair_benchmark", None)
    if checker is None:
        raise NotImplementedError("Rust WeightRepairBench registration unavailable; fixture only")
    checked = checker(_wire(task), None if result is None else _wire(result))
    if not isinstance(checked, bytes):
        raise TypeError("native benchmark validator must return canonical bytes")
    return checked


def _cost(
    operation: str,
    scope: str,
    phases: list[str],
    evidence: str,
    wall: float,
    cpu: float,
    forwards: int,
    proposals: int,
    changed: int | None = None,
) -> dict[str, Any]:
    quantities: dict[str, Any] = {
        name: {
            "status": "unknown",
            "value": None,
            "unit": unit,
            "reason": "not instrumented across complete operation; never treated as zero",
        }
        for name, unit in COST_UNITS.items()
    }
    # These zeros apply only to this runner's operations. Supplied preparation
    # remains unknown for every counter except its measured controller time.
    known = {"wall": wall, "cpu": cpu, "forwards": forwards, "proposal_attempts": proposals}
    if scope != "shared":
        known.update(
            backwards=0,
            optimizer_steps=0,
            fitting_steps=0,
            trainable_parameters=0,
            local_tokens=0,
            remote_tokens=0,
            quoted_spend=0,
            reserved_spend=0,
            actual_spend=0,
            gpu=0,
        )
        if changed is not None:
            known["changed_parameters"] = changed
    for name, value in known.items():
        quantities[name] = {
            "status": "known",
            "value": value,
            "unit": COST_UNITS[name],
            "reason": "controller clock or explicit native CPU fixture operation count",
        }
    return {
        "operation_id": operation,
        "scope": scope,
        "phases": phases,
        "evidence": evidence,
        "quantities": quantities,
    }


def _prepare(model: Any) -> Mapping[str, tuple[Any, dict[str, Any]]]:
    """Compile three supplied, known-target scale programs; no fitting/search."""
    meta = arithmetic.describe(model)
    manifest = json.loads(model.source.manifest_json())
    base = {
        "checkpoint": meta["checkpoint"],
        "manifest": model.source.manifest_digest,
        "parameter_digest": model.source.parameter_digest,
        "architecture_graph": meta["graph"],
        "config": meta["config"],
        "tokenizer": None,
    }
    # Synthetic component identities do not assert a frozen/authorized contract.
    d = _digest({"fixture": PROTOCOL, "authority": "none"})
    smoke = {
        "schema_version": 1,
        "adapter": "nf.arithmetic_relu.v1",
        "probes": [
            {"input": pair, "expected": None, "absolute_tolerance": 0.0, "max_abs_output": 1000.0}
            for pair in [[0.0, 0.0], [-8.0, 8.0], [8.0, -8.0]]
        ],
    }
    verification: dict[str, Any] = {
        "portfolio": d,
        "protocol": d,
        "gyms": [
            {
                "binding": _digest({"fixture_binding": g["id"]}),
                "gym": g["id"],
                "validation": d,
                "scorer": d,
                "split": d,
                "role": "target" if g["head"] == "sum" else "retention",
            }
            for g in meta["gyms"]
        ],
        "smoke_suite": _native.validate_record("smoke_suite", _wire(smoke)).digest,
        "parity_suite": d,
    }
    space = {
        "schema_version": 1,
        "base_model": base,
        "contract": d,
        "objective": d,
        "transformations": {
            "scale": {
                "dtypes": ["f64"],
                "coefficient": {"min": -4.0, "max": 4.0},
                "rank": None,
                "sides": [],
                "max_operand_bytes": 0,
            }
        },
        "sites": [
            {
                "selector": {"type": "tensor", "name": name},
                "role": "parameter",
                "regions": {"type": "whole_tensor"},
                "allowed_axes": [0, 1],
            }
            for name in manifest["parameters"]
        ],
        "constraints": {
            "max_changed_parameters": 1,
            "max_artifact_bytes": 0,
            "max_temporary_bytes": 4096,
            "max_operations": 1,
            "allow_buffer_writes": False,
            "require_direct_weight_edits": True,
            "require_fully_gradient_free": True,
            "require_preserve_shapes": True,
            "quality_limits": [
                {
                    "binding": verification["gyms"][0]["binding"],
                    "metric": "accuracy",
                    "units": "fraction",
                    "baseline": meta["checkpoint"],
                    "max_regression": 0.0,
                    "split": "selection",
                    "uncertainty_protocol": d,
                }
            ],
        },
        "strategy": {
            "proposal": "causal_hypothesis",
            "search": "branch",
            "selection": "pareto",
            "engine": "bounded.branch",
            "engine_version": "1",
            "seed": 17,
        },
    }
    space_id = _native.validate_record("search_space", _wire(space)).digest
    region = {"tensor": "heads.weight", "start": [0, 0], "shape": [1, 1]}
    permissions = {
        "schema_version": 1,
        "tools": ["inspect", "direct_edit"],
        "external_destinations": {},
        "allow_gradient_inspection": False,
        "allow_parameter_training": False,
        "require_fully_gradient_free": True,
        "allow_generated_gym": False,
        "allow_downloads": False,
    }
    policy = {
        "schema_version": 1,
        "source": base,
        "contract": d,
        "objective": d,
        "experiment_protocol": d,
        "search_space": space_id,
        "verification": verification,
        "allowed_extra_gyms": [],
        "permissions": permissions,
        "operand_access": ["development"],
        "backend": {
            "id": "nf.native_cpu",
            "version": "1",
            "operations": ["scale"],
            "capabilities": ["dense.cpu"],
            "dtypes": ["f64"],
            "accumulation_dtypes": ["f64"],
            "fixed_scratch_bytes": 64,
        },
        "bounds": {
            "max_metadata_bytes": 1048576,
            "max_source_checkpoint_bytes": 1048576,
            "max_source_asset_bytes": 4096,
            "max_source_tensors": 16,
            "max_observed_source_bytes": 4096,
            "max_temporary_bytes": 4096,
            "max_output_tensor_bytes": 4096,
            "max_validation_work": 100000,
            "max_kernel_work": 100000,
            "max_parameter_magnitude": 100.0,
            "max_backend_abs_tolerance": 0.0,
            "max_orthonormal_tolerance": 1e-6,
        },
    }
    plans = {}
    for name, factor in [("no_op", 1.0), ("direct_failed", 0.5), ("direct", 2.0)]:
        program = {
            "ir_version": 1,
            "opset_version": 1,
            "base_model": base,
            "objective": {
                "contract": d,
                "objective": d,
                "experiment_protocol": d,
                "search_space": space_id,
            },
            "operations": [
                {
                    "id": "op0",
                    "depends_on": [],
                    "operation": {"type": "scale", "target": region, "factor": factor},
                }
            ],
            "artifacts": {},
            "preconditions": [
                {
                    "tensor": n,
                    "role": "parameter",
                    "shape": s["shape"],
                    "dtype": "f64",
                    "before_digest": s["digest"],
                    "storage_group": s["storage_group"],
                }
                for n, s in manifest["parameters"].items()
            ],
            "required_capabilities": ["dense.cpu"],
            "constraints": {
                "gradient_free": True,
                "preserve_tensor_shapes": True,
                "max_changed_parameters": 1,
                "max_artifact_bytes": 0,
                "max_temporary_bytes": 4096,
                "numerical": {
                    "accumulation_dtype": "f64",
                    "output_rounding": "nearest_ties_even",
                    "non_finite": "reject",
                    "max_abs_parameter": 100.0,
                    "backend_abs_tolerance": 0.0,
                },
            },
            "effects": {
                "reads": [region],
                "writes": [region],
                "buffer_writes": [],
                "graph_changes": False,
                "config_changes": [],
                "tokenizer_changes": False,
            },
            "preservation": {"type": "experimental"},
            "verification": verification,
            "lineage": {
                "parent_candidates": [],
                "hypothesis": _digest({"supplied_factor": factor}),
                "method": {
                    "schema_version": 1,
                    "target_update": "direct_tensor_edit",
                    "proposal_origin": "scripted",
                    "adaptation": "none",
                    "production_history": "direct_only",
                    "inspection_uses_backprop": False,
                    "parameter_update_uses_backprop": False,
                    "gradient_derived_update_values": False,
                    "alignment_fitted": False,
                    "optimizer_steps": 0,
                    "changed_parameters": 1,
                },
            },
            "extensions": {},
        }
        plans[name] = (ir.compile(model.source, program, space, policy, {}), smoke)
    return plans


def run_fixture(prepare: Prepare | None = None) -> dict[str, Any]:
    """Run five real matched native arithmetic arms, retaining a failed edit.

    ``prepare`` returns exactly no_op/direct/direct_failed plans for the supplied
    corrupt fixture, with at most three bounded smoke probes per plan. It must
    perform only authorized direct compilation, without external work/training.
    Preparation counters not instrumented by this runner remain unknown.
    Limits are finite input/operation limits and post-operation wall checks, NOT
    hard process/CPU sandbox enforcement. Callers must bound the containing test
    process. Exceptions propagate; no fallback success replaces a failed run.
    """
    start, cpu_start = time.monotonic(), time.process_time()
    model = arithmetic.fixture("corrupt_sum")
    meta = arithmetic.describe(model)
    source = model.source.parameter_digest
    checkpoint = model.checkpoint_bytes()
    plans = dict((prepare or _prepare)(model))
    if set(plans) != {"no_op", "direct", "direct_failed"}:
        raise ValueError("exact three supplied compiled comparison plans required")
    for plan, smoke in plans.values():
        if not 1 <= len(smoke.get("probes", [])) <= 3:
            raise ValueError("one to three finite smoke probes required")
        if len(_wire(json.loads(plan.summary_json()))) > 65536:
            raise ValueError("bounded plan summary required")
    programs = {name: plan.digest for name, (plan, _) in plans.items()}
    evidence_documents: dict[str, Any] = {}

    def retain(value: Any) -> str:
        identity = _digest(value)
        evidence_documents[identity] = value
        return identity

    for plan, _ in plans.values():
        retain(json.loads(plan.summary_json()))
    cases = [c for c in meta["inventory"]["cases"] if c["split"] == "development"]
    if not 1 <= len(cases) <= 289:
        raise ValueError("bounded development inventory required")
    runtime = retain(
        {
            "backend": "nf.native_cpu",
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "native_extension": _file_digest(Path(_native.__file__)),
            "runner_code": _file_digest(Path(__file__)),
            "capabilities": json.loads(_native.capabilities_json()),
        }
    )
    task = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "task_version": "arithmetic-corrupt-sum-v1",
        "source": source,
        "model": model.source.manifest_digest,
        "gyms": {g["head"]: g["id"] for g in meta["gyms"]},
        "splits": {
            role: _digest(
                {
                    "inventory": meta["inventory_id"],
                    "role": role,
                    "cases": [c for c in meta["inventory"]["cases"] if c["split"] == role],
                }
            )
            for role in ("development", "selection", "holdout")
        },
        "failure": {
            "kind": "injected_known",
            "provenance": _digest(json.loads(model.asset_bytes("provenance"))),
            "ground_truth": _digest({"tensor": "heads.weight", "start": [0, 0], "shape": [1, 1]}),
            "solution_exposure": "hand-constructed known target and supplied repair; not discovery",
        },
        "programs": programs,
        "permissions": ["inspect", "direct_edit"],
        "budget": {
            "max_cases": 289,
            "max_attempts": 16,
            "max_forwards": 10000,
            "max_wall_seconds": 60,
        },
        "runtime": runtime,
        "seed": 17,
        "comparisons": list(METHODS),
    }
    validate(task)
    task_id = _digest(task)
    prep_evidence = retain({"programs": programs, "runtime": runtime})
    shared = _cost(
        "fixture-preparation",
        "shared",
        ["setup", "diagnosis", "search"],
        prep_evidence,
        time.monotonic() - start,
        time.process_time() - cpu_start,
        0,
        0,
    )
    # Preparation counters cannot be inferred from compiled plans. This is an
    # incomplete total budget, not zero preparation inference/proposal cost.
    for key in ("forwards", "proposal_attempts"):
        shared["quantities"][key].update(
            status="unknown", value=None, reason="supplied preparation not instrumented"
        )
    attempts = []
    for name in ("original", "no_op", "best_known", "direct_failed", "direct"):
        wall, cpu = time.monotonic(), time.process_time()
        candidate = None
        changed = 0 if name == "original" else None
        smoke_forwards = 0
        method = "direct" if name == "direct_failed" else name
        program = None
        evidence: dict[str, str] = {}
        if name in plans:
            plan, smoke = plans[name]
            candidate = surgeon.apply(plan, smoke)
            runtime_source = candidate.snapshot()
            receipt = json.loads(candidate.receipt_json())
            evidence["apply_receipt"] = retain(receipt)
            changed = receipt["changed_parameters"]
            program = programs[name]
            smoke_forwards = len(smoke["probes"])
        elif name == "best_known":
            runtime_source = arithmetic.fixture("reference").source
        else:
            runtime_source = model.source
        rows = [
            {
                "item": c["item"],
                "group": c["group"],
                "input": [c["x1"], c["x2"]],
                "output": runtime_source.forward([float(c["x1"]), float(c["x2"])]),
            }
            for c in cases
        ]
        scores = {
            head: {
                "count": len(rows),
                "correct": sum(
                    abs(row["output"][index] - arithmetic.expected(head, *row["input"]))
                    <= arithmetic.TOLERANCE
                    for row in rows
                ),
            }
            for index, head in enumerate(arithmetic.HEADS)
        }
        if model.source.parameter_digest != source or model.checkpoint_bytes() != checkpoint:
            raise RuntimeError("source changed during fixture comparison")
        passed = all(row["correct"] == row["count"] for row in scores.values())
        evidence["observations"] = retain(
            {"scores": scores, "rows": rows, "tolerance": arithmetic.TOLERANCE}
        )
        evidence["source_integrity"] = retain(
            {"before": source, "after": model.source.parameter_digest, "checkpoint_unchanged": True}
        )
        if name == "no_op" and runtime_source.parameter_digest != source:
            raise ValueError("supplied no-op must preserve exact parameter identity")
        if name == "direct_failed" and passed:
            raise ValueError(
                "supplied failed direct must actually fail the complete behavior panel"
            )
        cost = _cost(
            f"fixture-{name}",
            name,
            ["failed_edits"]
            if name == "direct_failed"
            else ["controls" if method != "direct" else "verification"],
            evidence["observations"],
            time.monotonic() - wall,
            time.process_time() - cpu,
            len(cases) + smoke_forwards,
            int(name in plans),
            changed,
        )
        attempts.append(
            {
                "id": name,
                "method": method,
                "status": "completed" if passed else "failed",
                "reason": "actual native development fixture; no final evaluator or fresh-process reload",
                "task": task_id,
                "source": source,
                "candidate": runtime_source.parameter_digest,
                "program": program,
                "scores": {
                    head: {"correct": row["correct"], "count": row["count"]}
                    for head, row in scores.items()
                },
                "behavioral": "passed" if passed else "failed",
                "localization": "not_evaluated",
                "mechanism": "not_evaluated",
                "gradient_inspection": False,
                "gradient_update": False,
                "evidence": evidence,
                "costs": [cost],
            }
        )
        if time.monotonic() - start > 60:
            raise TimeoutError("fixture cooperative wall budget exhausted; no acceptance result")
    for method in METHODS[4:]:
        attempts.append(
            {
                "id": method,
                "method": method,
                "status": "unsupported" if method in {"decoding", "activation"} else "not_run",
                "reason": "no scalar decoding/activation backend"
                if method in {"decoding", "activation"}
                else "training not authorized or dispatched",
                "task": task_id,
                "source": source,
                "candidate": None,
                "program": None,
                "scores": {},
                "behavioral": "not_evaluated",
                "localization": "not_evaluated",
                "mechanism": "not_evaluated",
                "gradient_inspection": False,
                "gradient_update": False,
                "evidence": {},
                "costs": [],
            }
        )
    result = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "task": task_id,
        "scope": SCOPE,
        "evaluation_split": "development",
        "attempts": attempts,
        "shared_costs": [shared],
        "failed_direct_attempts": sum(
            a["method"] == "direct" and a["status"] == "failed" for a in attempts
        ),
        "phase_coverage": {
            p: "not_run" if p in {"export", "reload", "recovery"} else "measured_or_unknown"
            for p in PHASES
        },
        "efficiency_claim": "not_ranked",
        "budget_verdict": "incomplete",
        "workflow_linkage": "unavailable_pending_authorized_workflow",
    }
    validate(task, result)
    return {"task": task, "result": result, "evidence_documents": evidence_documents}
