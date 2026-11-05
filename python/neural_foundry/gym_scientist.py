"""Finite, non-authoritative audits of supplied development/selection observations.

No policy, scorer, oracle, snapshot asset or model is loaded or executed. Truth,
policy roles and scores are controller assertions, not authenticated evidence.
Use gyms.validate/approve separately for native exact-version authority. The
native reference audit is exposed separately by run_reference_audit when its
binding is installed; this module never emulates execution with Python callbacks.

The v1 JSON request contains objective, snapshot, evidence_class,
truth_provenance, seeds, cases [{id, group, input, expected}], policies
[{id, role, provenance?}], required_coverage, scoring
{minimum, bad_max, good_min, maximum}, and results
[{case, policy, seed, replay, output, oracle_output, score, invalid_score}].
Optional provenance binds supplied snapshot/validation/observations SHA256 references,
not native authority. invalid_score=null asserts that the null-output probe raised
rather than returning a score; it is not an invented zero-cost/zero-score result.
Roles are valid, invalid, constant, metadata_only, weak, strong.
There must be two replays of each case/policy/seed. Null is an invalid output,
as in gym-validation-v1. No final/holdout evidence is accepted. CLI commands
plan, validate, run and report read this bounded JSON from stdin and emit JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from typing import Any, cast

from . import _native

MAX_BYTES = 262_144
ROLES = {"valid", "invalid", "constant", "metadata_only", "weak", "strong"}
SCOPE = "supplied_exposed_observations_not_native_validation_or_scientific_acceptance"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _fields(value: Any, names: str) -> None:
    if type(value) is not dict or set(value) != set(names.split()):
        raise ValueError(f"Expected exactly these fields: {names}")


def _label(value: Any) -> None:
    if type(value) is not str or not value.strip() or len(value) > 512:
        raise ValueError("Expected a nonempty bounded label")


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _unique(rows: Any, maximum: int) -> None:
    if type(rows) is not list or not 1 <= len(rows) <= maximum:
        raise ValueError("List outside frozen audit bounds")
    if len({_canonical(row) for row in rows}) != len(rows):
        raise ValueError("Duplicate declarations")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _load(payload: str | bytes) -> dict[str, Any]:
    # Exact built-in wire types exclude callable/custom mapping execution hooks.
    if type(payload) not in (str, bytes) or len(payload) > MAX_BYTES:
        raise ValueError("Supply bounded JSON text/bytes, never code or callbacks")
    if type(payload) is str and len(payload.encode("utf-8")) > MAX_BYTES:
        raise ValueError("Audit exceeds byte bound")
    try:
        value = json.loads(payload, object_pairs_hook=_pairs)
        pending = [(value, 0)]
        nodes = 0
        while pending:
            item, depth = pending.pop()
            nodes += 1
            if nodes > 32_768 or depth > 16:
                raise ValueError("Audit exceeds structural bounds")
            if isinstance(item, dict):
                pending.extend((v, depth + 1) for v in item.values())
                pending.extend((k, depth + 1) for k in item)
            elif isinstance(item, list):
                pending.extend((v, depth + 1) for v in item)
            elif isinstance(item, str) and len(item) > 4096:
                raise ValueError("Audit string exceeds bound")
            elif type(item) in (int, float) and not _number(item):
                raise ValueError("Non-finite audit number")
    except (RecursionError, OverflowError, UnicodeError) as exc:
        raise ValueError("Invalid bounded JSON") from exc
    _fields(value, "schema_version objective snapshot evidence_class truth_provenance seeds "
            "cases policies required_coverage scoring results")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("Unsupported audit schema")
    for key in ("objective", "snapshot", "truth_provenance"):
        _label(value[key])
    if value["evidence_class"] not in ("development", "selection"):
        raise ValueError("Only authorized exposed development/selection evidence is supported")
    for key, maximum in (("cases", 32), ("policies", 16), ("seeds", 4),
                         ("required_coverage", 32)):
        _unique(value[key], maximum)
    if not 2 <= len(value["cases"]) or not 2 <= len(value["seeds"]):
        raise ValueError("At least two cases and seeds are required")
    for seed in value["seeds"]:
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("Seeds must be bounded integers")
    for group in value["required_coverage"]:
        _label(group)
    for case in value["cases"]:
        _fields(case, "id group input expected")
        _label(case["id"])
        _label(case["group"])
        if case["expected"] is None:
            raise ValueError("Null cannot be fixture truth")
    for policy in value["policies"]:
        if type(policy) is not dict:
            raise ValueError("Expected a policy declaration")
        _fields(policy, "id role provenance" if "provenance" in policy else "id role")
        if "provenance" in policy:
            _fields(policy["provenance"], "snapshot validation observations")
            for identity in policy["provenance"].values():
                if (type(identity) is not str or len(identity) != 71
                        or not identity.startswith("sha256:")
                        or any(c not in "0123456789abcdef" for c in identity[7:])):
                    raise ValueError("Provenance requires exact SHA256 references")
        _label(policy["id"])
        if type(policy["role"]) is not str or policy["role"] not in ROLES:
            raise ValueError("Unknown reference-policy role")
    for key in ("cases", "policies"):
        if len({r["id"] for r in value[key]}) != len(value[key]):
            raise ValueError("Duplicate identity")
    gates = value["scoring"]
    _fields(gates, "minimum bad_max good_min maximum")
    if not all(_number(v) for v in gates.values()) or not (
        gates["minimum"] <= gates["bad_max"] < gates["good_min"] <= gates["maximum"]
    ):
        raise ValueError("Scoring requires finite ordered bounds")
    results = value["results"]
    if type(results) is not list or len(results) > 4096:
        raise ValueError("Results outside audit bounds")
    identities: set[tuple[str, str, int, int]] = set()
    for row in results:
        _fields(row, "case policy seed replay output oracle_output score invalid_score")
        if (type(row["case"]) is not str or type(row["policy"]) is not str
                or row["case"] not in {c["id"] for c in value["cases"]}
                or row["policy"] not in {p["id"] for p in value["policies"]}
                or type(row["seed"]) is not int or row["seed"] not in value["seeds"]
                or type(row["replay"]) is not int or row["replay"] not in (0, 1)):
            raise ValueError("Unknown result cell")
        identity = (row["case"], row["policy"], row["seed"], row["replay"])
        if identity in identities:
            raise ValueError("Duplicate result cell")
        identities.add(identity)
        if not _number(row["score"]) or (
            row["invalid_score"] is not None and not _number(row["invalid_score"])
        ):
            raise ValueError("Scores must be finite scalar numbers, not booleans")
    return cast(dict[str, Any], value)


class GymResearchWorkflow:
    """Immutable wire snapshot; deterministic analysis only, without execution authority."""

    def __init__(self, request: str | bytes) -> None:
        self._request = _canonical(_load(request))

    def plan(self) -> dict[str, Any]:
        request = json.loads(self._request)
        return {
            "schema_version": 1,
            "request_digest": "sha256:" + hashlib.sha256(self._request.encode()).hexdigest(),
            "snapshot": request["snapshot"],
            "scope": SCOPE,
            "expected_cells": len(request["cases"]) * len(request["policies"])
            * len(request["seeds"]) * 2,
            "execution": "none_supplied_results_only",
            "native_approval": False,
            "weights_modified": False,
            "reference_provenance": {p["id"]: p.get("provenance") for p in request["policies"]},
            "limitations": [
                "Supplied truth, roles, observations and snapshot reference are unauthenticated.",
                "No schema, sandbox, permission, budget or exact-version approval is established.",
                "This supplied-result API does not execute policies; use run_reference_audit.",
                "Finite exposed fixtures cannot establish holdout independence or broad validity.",
            ],
        }

    def validate(self) -> dict[str, Any]:
        """Validate wire structure/completeness only; this is NOT gyms.validate."""
        plan = self.plan()
        rows = json.loads(self._request)["results"]
        return {**plan, "input_valid": True, "observed_cells": len(rows),
                "complete": len(rows) == plan["expected_cells"]}

    def run(self) -> dict[str, Any]:
        request = json.loads(self._request)
        report = self.validate()
        findings: set[str] = set()
        cases = {c["id"]: c for c in request["cases"]}
        gates = request["scoring"]
        rows = request["results"]
        if len({_canonical(c["input"]) for c in cases.values()}) < 2 or len(
            {_canonical(c["expected"]) for c in cases.values()}
        ) < 2:
            findings.add("insufficient_distinct_inputs_or_truths")
        missing = sorted(set(request["required_coverage"]) - {c["group"] for c in cases.values()})
        if missing:
            findings.add("missing_required_coverage")
        signatures = [(_canonical(c["input"]), _canonical(c["expected"])) for c in cases.values()]
        if len(set(signatures)) < len(signatures):
            findings.add("duplicate_task_content")
        truths_by_input: dict[str, set[str]] = {}
        for inputs, truth in signatures:
            truths_by_input.setdefault(inputs, set()).add(truth)
        if any(len(truths) > 1 for truths in truths_by_input.values()):
            findings.add("conflicting_fixture_truth")
        roles = {p["role"] for p in request["policies"]}
        if not ROLES <= roles:
            findings.add("missing_reference_controls")
        by_cell: dict[tuple[str, str, int], list[str]] = {}
        counterexamples: list[dict[str, Any]] = []
        counterexample_count = 0
        for row in rows:
            defects = []
            truth = cases[row["case"]]["expected"]
            correct = _canonical(row["output"]) == _canonical(truth)
            if _canonical(row["oracle_output"]) != _canonical(truth):
                defects.append("oracle_disagreement")
            if not all(gates["minimum"] <= row[k] <= gates["maximum"]
                       for k in ("score", "invalid_score") if row[k] is not None):
                defects.append("score_out_of_bounds")
            if row["invalid_score"] is not None and row["invalid_score"] > gates["bad_max"]:
                defects.append("invalid_output_rewarded")
            if (correct and row["score"] < gates["good_min"]) or (
                not correct and row["score"] > gates["bad_max"]
            ):
                defects.append("scorer_disagreement")
            findings.update(defects)
            if defects:
                counterexample_count += 1
                if len(counterexamples) < 16:
                    counterexamples.append({**{k: row[k] for k in ("case", "policy", "seed", "replay")},
                                            "defects": defects})
            key = (row["case"], row["policy"], row["seed"])
            by_cell.setdefault(key, []).append(_canonical({
                k: row[k] for k in ("output", "oracle_output", "score", "invalid_score")
            }))
        if any(len(set(replays)) > 1 for replays in by_cell.values()):
            findings.add("replay_drift")
        summaries: dict[str, Any] = {}
        for policy in request["policies"]:
            samples = [r for r in rows if r["policy"] == policy["id"]]
            correct_count = sum(_canonical(r["output"]) == _canonical(cases[r["case"]]["expected"])
                                for r in samples)
            passing = sum(r["score"] >= gates["good_min"] for r in samples)
            summaries[policy["id"]] = {"role": policy["role"], "samples": len(samples),
                                       "correct": correct_count, "passing": passing}
            if policy["role"] in ("valid", "strong") and correct_count != len(samples):
                findings.add(policy["role"] + "_policy_failed")
            if policy["role"] == "invalid" and (correct_count or passing):
                findings.add("invalid_policy_not_separated")
            if passing and policy["role"] in ("constant", "metadata_only"):
                findings.add(policy["role"] + "_shortcut_rewarded")
            if samples and correct_count == len(samples) and policy["role"] == "weak":
                findings.add("weak_policy_saturation")
            if policy["role"] == "constant" and len(
                {_canonical(r["output"]) for r in samples}
            ) > 1:
                findings.add("constant_policy_role_mismatch")
        return {**report, "status": "inconclusive" if not report["complete"] else (
            "issues_found" if findings else "no_issues_observed"),
            "findings": sorted(findings), "missing_coverage": missing,
            "policies": summaries,
            "counterexamples": counterexamples,
            "counterexample_count": counterexample_count,
            "counterexamples_truncated": counterexample_count > len(counterexamples),
            "recommendations": [("Review findings against independently supplied reserved cases; "
                                 "changed gyms require new native validation and explicit approval.")]
            if findings else []}

    def report(self) -> dict[str, Any]:
        """Recompute fresh data; a result is never a native validation/approval handle."""
        return self.run()


def run_reference_audit(
    snapshot: _native.GymSnapshotHandle,
    plan: str | bytes,
    policy_snapshots: list[_native.GymSnapshotHandle],
    roles: list[str],
    exposure: str,
    budget: str | bytes,
    sandbox_root: str,
    *,
    enabled: bool = False,
    coverage: str | bytes | None = None,
    cancellation: _native.GymAuditCancellationHandle | None = None,
) -> _native.GymReferenceAuditHandle:
    """Execute real native probes; NEVER fall back to supplied-result analysis.

    Handles must name trusted controller-selected snapshots in the same CAS.
    Plans contain exposed independent truths, never observations. The native
    binding must qualify the pinned sandbox and enforce all bounds itself.
    For episode gyms, each single-shot policy evaluates the current observation
    with seed and returns an action. Policies are stateless across decisions;
    native execution replays each action prefix in an isolated unchanged gym.
    Case actions declare the finite decision horizon (not policy actions).
    Budget reservations include every reset/replay, decision and scoring job;
    their per-policy wall reservation cannot exceed the frozen episode limit.
    Full status 'passed' requires executed valid/invalid/constant/metadata_only/
    weak/strong controls and coverage JSON {required: [group], case_groups:
    {case_id: group}} covering every case and required group. Otherwise successful
    execution is only 'probe_passed'. Roles/groups/truths are controller claims.
    A cancellation handle interrupts active jobs and suppresses full success.
    Recovery is explicit via recover_reference_audits with exclusive root access.
    Container CPU/wall/memory measurements exclude controller/qualification cost;
    the report separately measures audit controller wall time. Budget JSON requires
    explicit storage_bytes: a controller-allocated invocation allowance, not a new
    study grant. Each max_executions slot charges the full 16 MiB sandbox ceiling
    (including source/input/guardian/journals), plus bounded CAS publications.
    No cleanup/dedup/early-exit refunds are inferred. Insufficient allowance fails
    before qualification/staging. Physical study storage remains unknown and
    capacity_guaranteed is false. Durable family settlement across retries, prior
    CAS/SQLite data and qualification/recovery costs require a separate controller
    reservation; this standalone exposed audit does not authorize study execution.
    Dispatch evidence and finite probes cannot prove malicious scorer semantics.
    The returned audit is not a GymValidationHandle or approval/holdout grant.
    """
    if type(enabled) is not bool or not enabled:
        raise ValueError("UNTRUSTED_GYM_EXECUTION_DISABLED")
    if (type(plan) not in (str, bytes) or type(budget) not in (str, bytes)
            or type(sandbox_root) is not str or type(exposure) is not str
            or exposure not in ("development", "selection")
            or type(policy_snapshots) is not list or not 1 <= len(policy_snapshots) <= 8
            or type(roles) is not list or len(roles) != len(policy_snapshots)
            or any(type(role) is not str or role not in ROLES for role in roles)):
        raise ValueError("Native audit requires bounded snapshots, roles and exposed JSON plans")
    if coverage is not None and (type(coverage) not in (str, bytes) or len(
        coverage.encode() if isinstance(coverage, str) else coverage
    ) > 4096):
        raise ValueError("Coverage requires bounded JSON text/bytes")
    if cancellation is not None and not isinstance(cancellation, _native.GymAuditCancellationHandle):
        raise TypeError("Native audit cancellation handle required")
    if len(plan.encode() if isinstance(plan, str) else plan) > 65536 or len(
        budget.encode() if isinstance(budget, str) else budget
    ) > 4096:
        raise ValueError("Native audit JSON exceeds bounds")

    if not hasattr(_native, "run_gym_reference_audit"):
        raise NotImplementedError(
            "Native run_gym_reference_audit binding is unavailable; "
            "supplied observations cannot substitute for execution"
        )
    return _native.run_gym_reference_audit(
        snapshot, plan, policy_snapshots, roles, exposure, budget, sandbox_root, enabled,
        coverage, cancellation,
    )


def recover_reference_audits(sandbox_root: str, *, enabled: bool = False) -> int:
    """Reconcile crashed jobs; exclusively own the root (never recover active peers)."""
    if type(enabled) is not bool or not enabled:
        raise ValueError("UNTRUSTED_GYM_EXECUTION_DISABLED")
    if type(sandbox_root) is not str or not sandbox_root or len(sandbox_root) > 4096:
        raise ValueError("Bounded controller sandbox root required")
    return _native.recover_gym_reference_audits(sandbox_root, enabled)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "validate", "run", "report"))
    args = parser.parse_args(argv)
    try:
        workflow = GymResearchWorkflow(sys.stdin.buffer.read(MAX_BYTES + 1))
        result = getattr(workflow, args.command)()
    except ValueError as exc:
        print(json.dumps({"status": "invalid_input", "error": str(exc)}))
        return 2
    print(_canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
