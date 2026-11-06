"""Holdout custodian: a separate OS process speaking JSON lines over stdin and stdout.

Protocol, in order: `freeze` one manifest naming every candidate and every exposed
operand; `query` each candidate digest once; `decide`.

- Seeds come from OS entropy after the freeze, and the draw excludes declared exposure.
- One manifest per study, so a proposer can't turn queries into a search by resubmitting
  slightly perturbed weights under new digests.
- A query is consumed before evaluation starts, so a crash can't buy a second look.
- Only aggregates leave the process. Errors cross the wire as codes, and tracebacks go to
  a sealed log because they can contain prompts.

This separates processes, not users: anything running as the same OS user can read the
state directory or delete the ledger. A confirmatory run should give the custodian its
own account. See docs/controlled-study-harness.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, BinaryIO

from .analysis import Interval, cluster_bootstrap_ci, holm_adjust, sign_flip_p_value
from .decision import PromotionDecision, RetentionGate, TaskEffect, decide
from .evaluators import Evaluator, build_evaluator
from .generators import RETENTION_TASK, Item, draw_clusters, draw_retention, render
from .plan import PlanError, PlanTamperedError, StudyPlan, digest_bytes, load_frozen_plan

GENESIS = "sha256:" + "0" * 64
NUMBER_TASKS = ("parity", "comparison")


class Refused(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def canonical(doc: Any) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_private(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def draw_sealed_holdout(
    plan: StudyPlan, seeds: list[int], retention_seed: int, exposed: dict[str, list[str]]
) -> list[Item]:
    """Regenerate the holdout from sealed seeds. Draws are disjoint from exposure and each other."""
    excluded_numbers = {operand for task in NUMBER_TASKS for operand in exposed.get(task, [])}
    excluded_boolean = set(exposed.get("boolean", []))
    items: list[Item] = []
    for draw_index, seed in enumerate(seeds):
        for task in plan.tasks:
            exclude = excluded_boolean if task == "boolean" else excluded_numbers
            clusters = draw_clusters(
                task,
                seed,
                plan.clusters_per_task_per_draw,
                domain=plan.number_domain,
                exclude=exclude,
                tag=f"draw{draw_index}",
            )
            for cluster in clusters:
                exclude.update(cluster.operands)
            items += render(clusters, plan.holdout_prefixes)
    items += draw_retention(
        retention_seed, plan.retention_items, exclude=frozenset(exposed.get(RETENTION_TASK, [])), tag="sealed"
    )
    return items


def holdout_commitment(seeds: list[int], retention_seed: int, items: list[Item]) -> str:
    doc = {"seeds": seeds, "retention_seed": retention_seed, "items": [asdict(item) for item in items]}
    return digest_bytes(canonical(doc))


def audit_sealed_state(state_dir: str | Path) -> dict[str, Any]:
    """For auditors after the study ends: reproduce the holdout and check it against the commitment.

    The proposer must not call this before the decision; the wire protocol has no equivalent.
    """
    root = Path(state_dir)
    sealed = json.loads((root / "sealed" / "holdout.json").read_bytes())
    plan = StudyPlan.from_dict(json.loads((root / "plan.json").read_bytes()))
    items = draw_sealed_holdout(plan, sealed["seeds"], sealed["retention_seed"], sealed["exposed"])
    commitment = holdout_commitment(sealed["seeds"], sealed["retention_seed"], items)
    if commitment != sealed["commitment"]:
        raise ValueError("sealed holdout doesn't reproduce its commitment")
    return {"commitment": commitment, "items": items, "exposed": sealed["exposed"]}


class Custodian:
    def __init__(self, plan_path: Path, plan_digest: str, state_dir: Path) -> None:
        self.plan_path = plan_path
        self.plan_digest = plan_digest
        self.state = state_dir
        self.sealed = state_dir / "sealed"
        self.ledger_path = state_dir / "ledger.jsonl"
        self.plan = load_frozen_plan(plan_path, plan_digest)
        self.events = self._read_ledger()
        self.evaluator: Evaluator | None = None
        self.items: list[Item] | None = None
        self.prompts: list[str] = []
        if self.events:
            if self.events[0]["plan_digest"] != plan_digest:
                raise Refused("PLAN_CHANGED", "this state directory is bound to a different plan digest")
        else:
            self.sealed.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(self.sealed, 0o700)
            (self.sealed / "candidates").mkdir(mode=0o700, exist_ok=True)
            _write_private(state_dir / "plan.json", plan_path.read_bytes())
            self._append("registered", plan_digest=plan_digest)
        if self._event("freeze"):
            self._load_items()

    # Ledger ---------------------------------------------------------------------------------

    def _read_ledger(self) -> list[dict[str, Any]]:
        if not self.ledger_path.exists():
            return []
        events: list[dict[str, Any]] = []
        previous = GENESIS
        for line in self.ledger_path.read_bytes().splitlines():
            event = json.loads(line)
            if event["previous"] != previous:
                raise Refused("LEDGER_CORRUPT", "the ledger hash chain is broken")
            previous = digest_bytes(line)
            events.append(event)
        return events

    def _append(self, kind: str, **fields: Any) -> None:
        previous = digest_bytes(canonical(self.events[-1])) if self.events else GENESIS
        event = {"kind": kind, "previous": previous, **fields}
        with open(self.ledger_path, "ab") as handle:
            handle.write(canonical(event) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.events.append(event)

    def _event(self, kind: str) -> dict[str, Any] | None:
        return next((event for event in self.events if event["kind"] == kind), None)

    def _check_plan(self) -> None:
        try:
            load_frozen_plan(self.plan_path, self.plan_digest)
        except (OSError, PlanTamperedError) as exc:
            raise Refused("PLAN_TAMPERED", "the plan file no longer matches its frozen digest") from exc

    # Holdout --------------------------------------------------------------------------------

    def _load_items(self) -> None:
        sealed = json.loads((self.sealed / "holdout.json").read_bytes())
        self.items = draw_sealed_holdout(self.plan, sealed["seeds"], sealed["retention_seed"], sealed["exposed"])
        if holdout_commitment(sealed["seeds"], sealed["retention_seed"], self.items) != sealed["commitment"]:
            raise Refused("HOLDOUT_CORRUPT", "the sealed holdout doesn't match its commitment")
        self.prompts = [item.prompt for item in self.items]

    def _validate_manifest(self, manifest: dict[str, Any]) -> list[dict[str, Any]]:
        if manifest.get("plan_digest") != self.plan_digest:
            raise Refused("WRONG_PLAN", "the manifest names a different plan digest")
        candidates = manifest.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise Refused("BAD_MANIFEST", "the manifest must list candidates")
        seeds = [candidate.get("synthesis_seed") for candidate in candidates]
        if sorted(seeds) != sorted(self.plan.synthesis_seeds):
            raise Refused("SEED_MISMATCH", "candidates must match the planned synthesis seeds exactly")
        digests = [candidate.get("candidate_digest") for candidate in candidates]
        if len(set(digests)) != len(digests):
            raise Refused("BAD_MANIFEST", "candidate digests must be distinct")
        exposed = manifest.get("exposed_operands")
        allowed = {*self.plan.tasks, RETENTION_TASK}
        if not isinstance(exposed, dict) or not set(exposed) <= allowed or not set(self.plan.tasks) <= set(exposed):
            raise Refused("BAD_MANIFEST", "exposed_operands must declare every task, even if empty")
        return candidates

    def freeze(self, manifest: dict[str, Any]) -> dict[str, Any]:
        self._check_plan()
        if self._event("freeze"):
            raise Refused("ALREADY_FROZEN", "this study has already frozen its only manifest")
        candidates = self._validate_manifest(manifest)
        for candidate in candidates:
            data = Path(str(candidate.get("artifact_path"))).read_bytes()
            if digest_bytes(data) != candidate["candidate_digest"]:
                raise Refused("DIGEST_MISMATCH", f"artifact for seed {candidate['synthesis_seed']} doesn't match its digest")
            # Keep a private copy: later changes to the proposer's file can't reach evaluation.
            name = candidate["candidate_digest"].removeprefix("sha256:")
            _write_private(self.sealed / "candidates" / name, data)
        seeds = [secrets.randbits(128) for _ in range(self.plan.holdout_draws)]
        retention_seed = secrets.randbits(128)
        exposed = manifest["exposed_operands"]
        try:
            items = draw_sealed_holdout(self.plan, seeds, retention_seed, exposed)
        except ValueError as exc:
            raise Refused("DOMAIN_EXHAUSTED", "the holdout domain is too small after exclusions") from exc
        commitment = holdout_commitment(seeds, retention_seed, items)
        doc = {"seeds": seeds, "retention_seed": retention_seed, "exposed": exposed, "commitment": commitment}
        _write_private(self.sealed / "holdout.json", canonical(doc))
        manifest_digest = digest_bytes(canonical(manifest))
        self._append(
            "freeze",
            manifest_digest=manifest_digest,
            holdout_commitment=commitment,
            candidates=[
                {"candidate_digest": c["candidate_digest"], "synthesis_seed": c["synthesis_seed"]} for c in candidates
            ],
        )
        self.items = items
        self.prompts = [item.prompt for item in items]
        return {
            "manifest_digest": manifest_digest,
            "holdout_commitment": commitment,
            "task_items": sum(item.task != RETENTION_TASK for item in items),
            "retention_items": sum(item.task == RETENTION_TASK for item in items),
        }

    # Scoring --------------------------------------------------------------------------------

    def _tasks_items(self) -> list[Item]:
        assert self.items is not None
        return [item for item in self.items if item.task != RETENTION_TASK]

    def _retention_items(self) -> list[Item]:
        assert self.items is not None
        return [item for item in self.items if item.task == RETENTION_TASK]

    def _get_evaluator(self) -> Evaluator:
        if self.evaluator is None:
            self.evaluator = build_evaluator(self.plan)
        return self.evaluator

    def _base_scores(self) -> dict[str, dict[str, float]]:
        path = self.sealed / "base_scores.json"
        if path.exists():
            scores: dict[str, dict[str, float]] = json.loads(path.read_bytes())
            return scores
        scores = self._get_evaluator().score(self._tasks_items(), None)
        _write_private(path, canonical(scores))
        return scores

    def _cluster_values(
        self, task: str, metric: str, scores: dict[str, dict[str, float]]
    ) -> tuple[list[str], list[float]]:
        grouped: dict[str, list[float]] = {}
        for item in self._tasks_items():
            if item.task == task:
                grouped.setdefault(item.cluster_id, []).append(scores[item.item_id][metric])
        return list(grouped), [sum(values) / len(values) for values in grouped.values()]

    @staticmethod
    def _stratum(cluster_id: str) -> int:
        return int(cluster_id.split(":")[1].removeprefix("draw"))

    def _candidate_aggregate(
        self, base: dict[str, dict[str, float]], scores: dict[str, dict[str, float]], retention: dict[str, dict[str, float]]
    ) -> dict[str, Any]:
        plan = self.plan
        tasks: dict[str, Any] = {}
        for task_index, task in enumerate(plan.tasks):
            metrics: dict[str, Any] = {}
            for metric in (plan.primary_metric, *plan.secondary_metrics):
                ids, base_values = self._cluster_values(task, metric, base)
                _, values = self._cluster_values(task, metric, scores)
                deltas = [v - b for v, b in zip(values, base_values, strict=True)]
                interval = cluster_bootstrap_ci(
                    deltas,
                    [self._stratum(cluster) for cluster in ids],
                    level=plan.corrected_ci_level,
                    resamples=plan.bootstrap_resamples,
                    seed=plan.analysis_seed + task_index,
                )
                metrics[metric] = {
                    "base": sum(base_values) / len(base_values),
                    "candidate": sum(values) / len(values),
                    **{key: getattr(interval, key) for key in ("mean", "low", "high", "level", "clusters")},
                }
            tasks[task] = metrics
        agreement, kl = self._retention_intervals(retention)
        return {"tasks": tasks, "retention": {"agreement": asdict(agreement), "kl": asdict(kl)}}

    def _retention_intervals(self, retention: dict[str, dict[str, float]]) -> tuple[Interval, Interval]:
        items = self._retention_items()
        strata = [0] * len(items)

        def interval(key: str) -> Interval:
            return cluster_bootstrap_ci(
                [retention[item.item_id][key] for item in items],
                strata,
                level=self.plan.corrected_ci_level,
                resamples=self.plan.bootstrap_resamples,
                seed=self.plan.analysis_seed + 1000,
            )

        return interval("agreement"), interval("kl")

    def _score_path(self, digest: str) -> Path:
        return self.sealed / f"scores_{digest.removeprefix('sha256:')}.json"

    def query(self, digest: Any) -> dict[str, Any]:
        self._check_plan()
        freeze = self._event("freeze")
        if freeze is None:
            raise Refused("NOT_FROZEN", "no manifest has been frozen")
        if digest not in {candidate["candidate_digest"] for candidate in freeze["candidates"]}:
            raise Refused("UNKNOWN_CANDIDATE", "this digest is not in the frozen manifest")
        if any(event["kind"] == "query_started" and event["candidate_digest"] == digest for event in self.events):
            raise Refused("ALREADY_QUERIED", "this candidate has already used its only query")
        # Consumed before any evaluation: failure or a crash past this line can't be retried.
        self._append("query_started", candidate_digest=digest)
        try:
            data = (self.sealed / "candidates" / digest.removeprefix("sha256:")).read_bytes()
            if digest_bytes(data) != digest:
                raise ValueError("sealed candidate copy doesn't match its digest")
            evaluator = self._get_evaluator()
            base = self._base_scores()
            scores = evaluator.score(self._tasks_items(), data)
            retention = evaluator.retention(self._retention_items(), data)
            aggregate = self._candidate_aggregate(base, scores, retention)
        except Exception:
            traceback.print_exc()
            self._append("query_failed", candidate_digest=digest)
            raise Refused("EVALUATION_FAILED", "evaluation failed; this candidate's query is spent") from None
        _write_private(self._score_path(digest), canonical({"scores": scores, "retention": retention}))
        self._append("query_finished", candidate_digest=digest, aggregate_digest=digest_bytes(canonical(aggregate)))
        return {"candidate_digest": digest, "aggregate": aggregate}

    # Decision -------------------------------------------------------------------------------

    def _study_decision(self, freeze: dict[str, Any]) -> PromotionDecision:
        plan = self.plan
        # A query started but never finished (a crash, say) counts as failed and can't be dropped.
        finished = {e["candidate_digest"] for e in self.events if e["kind"] == "query_finished"}
        failed = tuple(c["candidate_digest"] for c in freeze["candidates"] if c["candidate_digest"] not in finished)
        if failed:
            return decide(
                plan,
                plan_digest=self.plan_digest,
                manifest_digest=freeze["manifest_digest"],
                effects=(),
                retention=None,
                failed_candidates=failed,
            )
        results = {
            c["synthesis_seed"]: json.loads(self._score_path(c["candidate_digest"]).read_bytes())
            for c in freeze["candidates"]
        }
        base = self._base_scores()
        metric = plan.primary_metric
        rows = []
        for task_index, task in enumerate(plan.tasks):
            ids, base_values = self._cluster_values(task, metric, base)
            per_seed = {seed: self._cluster_values(task, metric, r["scores"])[1] for seed, r in results.items()}
            pooled = [sum(column) / len(column) for column in zip(*per_seed.values(), strict=True)]
            deltas = [p - b for p, b in zip(pooled, base_values, strict=True)]
            interval = cluster_bootstrap_ci(
                deltas,
                [self._stratum(cluster) for cluster in ids],
                level=plan.corrected_ci_level,
                resamples=plan.bootstrap_resamples,
                seed=plan.analysis_seed + task_index,
            )
            p_value = sign_flip_p_value(deltas, resamples=plan.permutation_resamples, seed=plan.analysis_seed + task_index)
            seed_deltas = {
                str(seed): sum(values) / len(values) - sum(base_values) / len(base_values)
                for seed, values in per_seed.items()
            }
            rows.append((task, base_values, pooled, interval, p_value, seed_deltas))
        adjusted = holm_adjust([row[4] for row in rows])
        effects = []
        for (task, base_values, pooled, interval, p_value, seed_deltas), p_adjusted in zip(rows, adjusted, strict=True):
            effects.append(
                TaskEffect(
                    task=task,
                    base_accuracy=sum(base_values) / len(base_values),
                    candidate_accuracy=sum(pooled) / len(pooled),
                    mean_delta=interval.mean,
                    ci_low=interval.low,
                    ci_high=interval.high,
                    ci_level=interval.level,
                    p_value=p_value,
                    p_adjusted=p_adjusted,
                    clusters=interval.clusters,
                    items=interval.clusters * len(plan.holdout_prefixes),
                    seed_deltas=seed_deltas,
                )
            )
        intervals = [self._retention_intervals(r["retention"]) for r in results.values()]
        retention = RetentionGate(
            worst_agreement_ci_low=min(pair[0].low for pair in intervals),
            worst_kl_ci_high=max(pair[1].high for pair in intervals),
            items=len(self._retention_items()),
        )
        return decide(
            plan,
            plan_digest=self.plan_digest,
            manifest_digest=freeze["manifest_digest"],
            effects=tuple(effects),
            retention=retention,
        )

    def decide(self) -> dict[str, Any]:
        self._check_plan()
        freeze = self._event("freeze")
        if freeze is None:
            raise Refused("NOT_FROZEN", "no manifest has been frozen")
        started = {e["candidate_digest"] for e in self.events if e["kind"] == "query_started"}
        if started != {c["candidate_digest"] for c in freeze["candidates"]}:
            raise Refused("INCOMPLETE", "every frozen candidate must be queried before the decision")
        path = self.state / "decision.json"
        if path.exists():
            stored: dict[str, Any] = json.loads(path.read_bytes())
            return stored
        doc = self._study_decision(freeze).to_dict()
        doc["ledger_head"] = digest_bytes(canonical(self.events[-1]))
        doc["holdout_commitment"] = freeze["holdout_commitment"]
        _write_private(path, canonical(doc))
        self._append("decision", promoted=doc["promoted"], decision_digest=digest_bytes(canonical(doc)))
        return doc

    # Wire -----------------------------------------------------------------------------------

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        operation = request.get("op")
        if operation == "freeze":
            return self.freeze(request.get("manifest") or {})
        if operation == "query":
            return self.query(request.get("candidate_digest"))
        if operation == "decide":
            return self.decide()
        raise Refused("UNKNOWN_OPERATION", "the custodian answers only freeze, query and decide")

    def egress(self, payload: dict[str, Any]) -> bytes:
        """Final check on everything leaving the process: no holdout prompt may appear."""
        data = json.dumps(payload).encode("utf-8")
        text = data.decode("utf-8")
        if any(prompt in text for prompt in self.prompts):
            traceback.print_stack()
            return json.dumps({"ok": False, "code": "EGRESS_BLOCKED", "message": "response withheld"}).encode("utf-8")
        return data


def _send(stream: BinaryIO, data: bytes) -> None:
    stream.write(data + b"\n")
    stream.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-digest", required=True)
    parser.add_argument("--state-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    # Take over stdout so nothing but protocol frames can reach the proposer.
    wire = os.fdopen(os.dup(1), "wb")
    requests = os.fdopen(os.dup(0), "rb")
    args.state_dir.mkdir(parents=True, exist_ok=True)
    log = os.open(args.state_dir / "custodian.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.dup2(log, 1)
    os.dup2(log, 2)

    try:
        custodian = Custodian(args.plan, args.plan_digest, args.state_dir)
    except PlanTamperedError:
        _send(wire, json.dumps({"ok": False, "code": "PLAN_TAMPERED", "message": "plan digest mismatch"}).encode())
        return 2
    except Refused as refused:
        _send(wire, json.dumps({"ok": False, "code": refused.code, "message": refused.message}).encode())
        return 2
    except (PlanError, OSError, ValueError):
        traceback.print_exc()
        _send(wire, json.dumps({"ok": False, "code": "STARTUP_FAILED", "message": "see the custodian log"}).encode())
        return 2
    _send(wire, json.dumps({"ok": True, "ready": True, "plan_digest": args.plan_digest}).encode())

    for line in requests:
        request_id = hashlib.sha256(line).hexdigest()[:12]
        try:
            response = {"ok": True, "result": custodian.handle(json.loads(line))}
        except Refused as refused:
            response = {"ok": False, "code": refused.code, "message": refused.message}
        except Exception:
            traceback.print_exc()
            response = {"ok": False, "code": "INTERNAL_ERROR", "message": f"see the custodian log, request {request_id}"}
        _send(wire, custodian.egress(response))
    return 0


if __name__ == "__main__":
    sys.exit(main())
