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
import stat
import sys
import traceback
from collections.abc import Callable
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from typing import Any, BinaryIO

from .analysis import Interval, cluster_bootstrap_ci, holm_adjust, sign_flip_p_value
from .audits import get_audit_plugin, validate_aggregate
from .claims import (
    ClaimSpent,
    StateBusy,
    StrictClaims,
    _sync_directory,
    exclusive_state,
    write_private,
)
from .decision import ControlEffect, PromotionDecision, RetentionGate, TaskEffect, decide
from .evaluators import Evaluator, build_evaluator
from .generators import (
    RETENTION_TASK,
    GeneratorContractError,
    Item,
    check_clusters,
    get_item_generator,
    render,
)
from .plan import (
    BASE_ARM,
    PLACEBO_ARM,
    TUNED_ARMS,
    PlanError,
    PlanTamperedError,
    StudyPlan,
    base_arm_artifact,
    digest_bytes,
    load_frozen_plan,
)

GENESIS = "sha256:" + "0" * 64
PLACEBO_NORM_TOLERANCE = 1e-6


class Refused(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def canonical(doc: Any) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_private(path: Path, data: bytes) -> None:
    write_private(path, data)


def _serialized(method: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    @wraps(method)
    def locked(self: Custodian, *args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            with exclusive_state(self.state):
                self.events = self._read_ledger()
                if self._event("freeze") and self.items is None:
                    self._load_items()
                return method(self, *args, **kwargs)
        except StateBusy as error:
            raise Refused("CUSTODIAN_BUSY", "another custodian operation owns this state") from error
    return locked


def draw_sealed_holdout(
    plan: StudyPlan, seeds: list[int], retention_seed: int, exposed: dict[str, list[str]]
) -> list[Item]:
    """Regenerate the holdout from sealed seeds. Draws are disjoint from exposure and each other."""
    generator = get_item_generator(plan.item_generator)
    shared = {operand for task in generator.shared_operand_tasks for operand in exposed.get(task, [])}
    excluded = {
        task: shared if task in generator.shared_operand_tasks else set(exposed.get(task, [])) for task in plan.tasks
    }
    items: list[Item] = []
    for draw_index, seed in enumerate(seeds):
        for task in plan.tasks:
            exclude = excluded[task]
            tag = f"draw{draw_index}"
            clusters = generator.draw_clusters(
                task, seed, plan.clusters_per_task_per_draw, domain=plan.number_domain, exclude=exclude, tag=tag
            )
            check_clusters(clusters, task, plan.clusters_per_task_per_draw, exclude, tag)
            for cluster in clusters:
                exclude.update(cluster.operands)
            items += render(clusters, plan.holdout_prefixes)
    items += generator.draw_retention(
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
    manifest = json.loads((root / "manifest.json").read_bytes())
    return {"commitment": commitment, "items": items, "exposed": sealed["exposed"], "manifest": manifest}


class Custodian:
    def __init__(self, plan_path: Path, plan_digest: str, state_dir: Path) -> None:
        self.plan_path = plan_path
        self.plan_digest = plan_digest
        self.state = state_dir
        self.sealed = state_dir / "sealed"
        self.ledger_path = state_dir / "ledger.jsonl"
        self.plan = load_frozen_plan(plan_path, plan_digest)
        self.evaluator: Evaluator | None = None
        self.items: list[Item] | None = None
        self.prompts: list[str] = []
        self.account_boundary = "same_user_process_only"
        with exclusive_state(self.state):
            self.claims = StrictClaims(self.state)
            self.events = self._read_ledger()
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
        _sync_directory(self.state)
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
        controls = manifest.get("controls", [])
        if not isinstance(controls, list):
            raise Refused("BAD_MANIFEST", "controls must be a list")
        digests = [entry.get("candidate_digest") for entry in (*candidates, *controls)]
        if len(set(digests)) != len(digests):
            raise Refused("BAD_MANIFEST", "candidate and control digests must all be distinct")
        exposed = manifest.get("exposed_operands")
        allowed = {*self.plan.tasks, RETENTION_TASK}
        if not isinstance(exposed, dict) or not set(exposed) <= allowed or not set(self.plan.tasks) <= set(exposed):
            raise Refused("BAD_MANIFEST", "exposed_operands must declare every task, even if empty")
        self._validate_controls(controls)
        self._validate_tuning(manifest.get("tuning", {}), exposed)
        if "audit" in manifest:
            self._validate_audit_spec(manifest["audit"])
        return candidates

    @staticmethod
    def _validate_audit_spec(spec: Any) -> None:
        if (not isinstance(spec, dict)
            or set(spec) != {"plugin", "config", "training_rows_digest", "artifact_path"}
            or not isinstance(spec["plugin"], str)
            or not isinstance(spec["config"], dict)
            or not isinstance(spec["artifact_path"], str)
            or not 1 <= len(spec["artifact_path"]) <= 4096
            or "\0" in spec["artifact_path"]
            or not Path(spec["artifact_path"]).is_absolute()
            or not isinstance(spec["training_rows_digest"], str)):
            raise Refused("BAD_AUDIT", "audit requires a frozen bounded artifact and registry configuration")
        try:
            get_audit_plugin(spec["plugin"]).validate(spec["config"])
        except ValueError as error:
            raise Refused("BAD_AUDIT", "unknown or invalid audit plugin configuration") from error

    def _validate_controls(self, controls: list[dict[str, Any]]) -> None:
        """Every planned arm must be present, for every seed: a control can't be left out of a study."""
        wanted = {
            (arm, None if arm == BASE_ARM else seed)
            for arm in self.plan.arms
            for seed in self.plan.synthesis_seeds
        }
        found = [(entry.get("arm"), entry.get("synthesis_seed")) for entry in controls]
        if len(set(found)) != len(found) or set(found) - wanted:
            raise Refused("UNPLANNED_CONTROL", "the manifest lists a control the plan doesn't name, or lists one twice")
        missing = wanted - set(found)
        if missing:
            arms = sorted({arm for arm, _ in missing})
            raise Refused("CONTROL_MISSING", f"the manifest is missing planned control arms: {arms}")
        for entry in controls:
            if entry["arm"] == BASE_ARM:
                data = Path(str(entry.get("artifact_path"))).read_bytes()
                if data != base_arm_artifact(self.plan_digest):
                    raise Refused("BAD_MANIFEST", "the base arm must submit the base marker for this plan")

    def _validate_tuning(self, tuning: Any, exposed: dict[str, list[str]]) -> None:
        """Tuning is recorded in the frozen manifest, and all it saw must be excluded from the holdout."""
        for arm in set(self.plan.arms) & set(TUNED_ARMS):
            record = tuning.get(arm) if isinstance(tuning, dict) else None
            if not isinstance(record, dict) or not record.get("trials") or not record.get("selected"):
                raise Refused("TUNING_NOT_RECORDED", f"the manifest has no tuning record for the {arm} arm")
            data = record.get("data") or {}
            prefixes = data.get("prefixes")
            if not prefixes or not set(prefixes) <= set(self.plan.development_prefixes):
                raise Refused("TUNING_NOT_DEVELOPMENT", f"the {arm} arm wasn't tuned on development prefixes only")
            operands = data.get("operands") or {}
            retention = data.get("retention_prompts") or []
            undeclared = [
                task for task, values in operands.items() if not set(values) <= set(exposed.get(task, []))
            ]
            if undeclared or not set(retention) <= set(exposed.get(RETENTION_TASK, [])):
                raise Refused("TUNING_NOT_DECLARED", f"the {arm} arm's tuning data isn't all declared as exposed")

    def _validate_placebos(self, candidates: list[dict[str, Any]], controls: list[dict[str, Any]]) -> None:
        """A placebo must match its candidate's norm and rank, or the comparison could be rigged with a big one."""
        if PLACEBO_ARM not in self.plan.arms:
            return
        by_seed = {candidate["synthesis_seed"]: candidate for candidate in candidates}
        evaluator = self._get_evaluator()

        def profile_of(entry: dict[str, Any]) -> dict[str, dict[str, float]]:
            name = entry["candidate_digest"].removeprefix("sha256:")  # The sealed copy, already digest-checked.
            return evaluator.edit_profile((self.sealed / "candidates" / name).read_bytes())

        for placebo in (entry for entry in controls if entry["arm"] == PLACEBO_ARM):
            candidate = profile_of(by_seed[placebo["synthesis_seed"]])
            profile = profile_of(placebo)
            matches = set(candidate) == set(profile) and all(
                candidate[site]["rank"] == profile[site]["rank"]
                and abs(candidate[site]["norm"] - profile[site]["norm"])
                <= PLACEBO_NORM_TOLERANCE * max(candidate[site]["norm"], 1e-12)
                for site in candidate
            )
            if not matches:
                raise Refused("PLACEBO_MISMATCH", f"the placebo for seed {placebo['synthesis_seed']} doesn't match")

    @_serialized
    def freeze(self, manifest: dict[str, Any]) -> dict[str, Any]:
        self._check_plan()
        if self._event("freeze"):
            raise Refused("ALREADY_FROZEN", "this study has already frozen its only manifest")
        if self.plan.label == "confirmatory" and not get_item_generator(self.plan.item_generator).confirmatory_ready:
            raise Refused("GENERATOR_NOT_QUALIFIED", f"{self.plan.item_generator} isn't fit for a confirmatory study")
        candidates = self._validate_manifest(manifest)
        controls = manifest.get("controls", [])
        for candidate in (*candidates, *controls):
            data = Path(str(candidate.get("artifact_path"))).read_bytes()
            if digest_bytes(data) != candidate["candidate_digest"]:
                raise Refused("DIGEST_MISMATCH", f"artifact for seed {candidate.get('synthesis_seed')} doesn't match its digest")
            # Keep a private copy: later changes to the proposer's file can't reach evaluation.
            name = candidate["candidate_digest"].removeprefix("sha256:")
            _write_private(self.sealed / "candidates" / name, data)
        self._validate_placebos(candidates, controls)
        seeds = [secrets.randbits(128) for _ in range(self.plan.holdout_draws)]
        retention_seed = secrets.randbits(128)
        exposed = manifest["exposed_operands"]
        try:
            items = draw_sealed_holdout(self.plan, seeds, retention_seed, exposed)
        except GeneratorContractError as exc:
            raise Refused("BAD_GENERATOR", str(exc)) from exc
        except ValueError as exc:
            raise Refused("DOMAIN_EXHAUSTED", "the holdout domain is too small after exclusions") from exc
        commitment = holdout_commitment(seeds, retention_seed, items)
        doc = {"seeds": seeds, "retention_seed": retention_seed, "exposed": exposed, "commitment": commitment}
        _write_private(self.sealed / "holdout.json", canonical(doc))
        manifest_digest = digest_bytes(canonical(manifest))
        _write_private(self.state / "manifest.json", canonical(manifest))  # Holds no holdout data, and binds tuning.
        self._append(
            "freeze",
            manifest_digest=manifest_digest,
            holdout_commitment=commitment,
            candidates=[
                {"candidate_digest": c["candidate_digest"], "synthesis_seed": c["synthesis_seed"]} for c in candidates
            ],
            controls=[
                {"arm": c["arm"], "candidate_digest": c["candidate_digest"], "synthesis_seed": c["synthesis_seed"]}
                for c in controls
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
        self,
        base: dict[str, dict[str, float]],
        scores: dict[str, dict[str, float]],
        retention: dict[str, dict[str, float]] | None,
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
        if retention is None:
            return {"tasks": tasks, "retention": None}
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

    @_serialized
    def query(self, digest: Any) -> dict[str, Any]:
        self._check_plan()
        freeze = self._event("freeze")
        if freeze is None:
            raise Refused("NOT_FROZEN", "no manifest has been frozen")
        arms = {c["candidate_digest"]: "candidate" for c in freeze["candidates"]}
        arms.update({c["candidate_digest"]: c["arm"] for c in freeze["controls"]})
        if digest not in arms:
            raise Refused("UNKNOWN_CANDIDATE", "this digest is not in the frozen manifest")
        if any(event["kind"] == "query_started" and event["candidate_digest"] == digest for event in self.events):
            raise Refused("ALREADY_QUERIED", "this candidate has already used its only query")
        try:
            claim = self.claims.begin(f"query:{digest}", {"plan_digest": self.plan_digest, "candidate_digest": digest})
        except ClaimSpent as error:
            raise Refused("ALREADY_QUERIED", "this candidate has already consumed its only start") from error
        # Consumed before any evaluation: failure or a crash past this line can't be retried.
        self._append("query_started", candidate_digest=digest)
        try:
            data = (self.sealed / "candidates" / digest.removeprefix("sha256:")).read_bytes()
            if digest_bytes(data) != digest:
                raise ValueError("sealed candidate copy doesn't match its digest")
            evaluator = self._get_evaluator()
            base = self._base_scores()
            if arms[digest] == BASE_ARM:
                scores, retention = base, None
            else:
                scores = evaluator.score(self._tasks_items(), data)
                retention = evaluator.retention(self._retention_items(), data)
            aggregate = self._candidate_aggregate(base, scores, retention)
        except Exception:
            traceback.print_exc()
            self._append("query_failed", candidate_digest=digest)
            raise Refused("EVALUATION_FAILED", "evaluation failed; this candidate's query is spent") from None
        _write_private(self._score_path(digest), canonical({"scores": scores, "retention": retention}))
        self._append("query_finished", candidate_digest=digest, aggregate_digest=digest_bytes(canonical(aggregate)))
        receipt = {"candidate_digest": digest, "arm": arms[digest], "aggregate": aggregate}
        self.claims.finish(claim, receipt)
        return receipt

    @_serialized
    def audit(self, spec: dict[str, Any]) -> dict[str, Any]:
        self._check_plan()
        frozen = self._event("freeze")
        if frozen is None:
            raise Refused("NOT_FROZEN", "no audit manifest has been frozen")
        manifest = json.loads((self.state / "manifest.json").read_bytes())
        if digest_bytes(canonical(manifest)) != frozen["manifest_digest"]:
            raise Refused("MANIFEST_CHANGED", "the audit manifest differs from its frozen digest")
        self._validate_audit_spec(spec)
        if spec != manifest.get("audit"):
            raise Refused("AUDIT_CHANGED", "the audit request differs from the frozen manifest")
        try:
            claim = self.claims.begin("audit", {"plan_digest": self.plan_digest, **spec})
        except ClaimSpent as error:
            raise Refused("ALREADY_AUDITED", "this manifest has consumed its only audit start") from error
        self._append("audit_started", request_digest=digest_bytes(canonical(spec)))
        try:
            descriptor = os.open(spec["artifact_path"], os.O_RDONLY
                                 | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
            with os.fdopen(descriptor, "rb") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > 8 * 1024 * 1024:
                    raise ValueError("audit requires a bounded regular training artifact")
                raw = stream.read(8 * 1024 * 1024 + 1)
            if len(raw) > 8 * 1024 * 1024:
                raise ValueError("training artifact exceeds the audit bound")
            from ..provenance import parse_strict_json

            document = parse_strict_json(raw)
            if (set(document) != {"rows"} or not isinstance(document["rows"], list)
                or not all(isinstance(row, dict) for row in document["rows"])
                or digest_bytes(canonical(document)) != spec["training_rows_digest"]):
                raise ValueError("training artifact differs from its frozen rows digest")
            _write_private(self.sealed / "audit-training.json", raw)
            plugin = get_audit_plugin(spec["plugin"])
            if self.items is None:
                raise ValueError("sealed audit items are unavailable")
            result = plugin.audit(document["rows"], self.items, spec["config"])
            validate_aggregate(result)
            if result["thresholds"] != spec["config"]["thresholds"]:
                raise ValueError("audit plugin changed the frozen thresholds")
            module_file = getattr(sys.modules.get(plugin.__class__.__module__), "__file__", None)
            if not isinstance(module_file, str):
                raise TypeError("audit plugin has no verifiable implementation file")
            receipt = {
                "kind": "nf_custodian_audit_receipt_v1",
                "plan_digest": self.plan_digest,
                "manifest_digest": frozen["manifest_digest"],
                "holdout_commitment": frozen["holdout_commitment"],
                "training_rows_digest": spec["training_rows_digest"],
                "training_artifact_digest": digest_bytes(raw),
                "plugin": spec["plugin"],
                "config_digest": digest_bytes(canonical(spec["config"])),
                "plugin_implementation_digest": digest_bytes(Path(module_file).read_bytes()),
                "aggregate": result,
                "exposure_journal": "custodian_local_only",
                "native_ledger_bridge": False,
                "signed": False,
                "account_boundary": self.account_boundary,
            }
            receipt["receipt_digest"] = digest_bytes(canonical(receipt))
            self.claims.finish(claim, receipt)
            self._append("audit_finished", receipt_digest=receipt["receipt_digest"])
            return receipt
        except (OSError, ValueError, TypeError):
            traceback.print_exc()
            self._append("audit_failed")
            raise Refused("AUDIT_FAILED", "audit failed; this manifest's audit start is spent") from None

    # Decision -------------------------------------------------------------------------------

    def _study_decision(self, freeze: dict[str, Any]) -> PromotionDecision:
        plan = self.plan
        # A query started but never finished (a crash, say) counts as failed and can't be dropped.
        finished = {e["candidate_digest"] for e in self.events if e["kind"] == "query_finished"}
        failed = tuple(c["candidate_digest"] for c in freeze["candidates"] if c["candidate_digest"] not in finished)
        failed_controls = tuple(
            f"{c['arm']}:{c['candidate_digest']}" for c in freeze["controls"] if c["candidate_digest"] not in finished
        )
        if failed or failed_controls:
            return decide(
                plan,
                plan_digest=self.plan_digest,
                manifest_digest=freeze["manifest_digest"],
                effects=(),
                retention=None,
                failed_candidates=failed,
                failed_controls=failed_controls,
            )
        results = {
            c["synthesis_seed"]: json.loads(self._score_path(c["candidate_digest"]).read_bytes())
            for c in freeze["candidates"]
        }
        base = self._base_scores()
        metric = plan.primary_metric
        control_results: dict[str, dict[int, dict[str, Any]]] = {}
        for entry in freeze["controls"]:
            if entry["arm"] != BASE_ARM:
                scores = json.loads(self._score_path(entry["candidate_digest"]).read_bytes())
                control_results.setdefault(entry["arm"], {})[entry["synthesis_seed"]] = scores
        rows = []
        control_rows: list[tuple[str, str, list[float], list[float], list[float], Interval, float, dict[str, float]]] = []
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
            for arm_index, arm in enumerate(plan.comparator_arms, start=1):
                per_control = {
                    seed: self._cluster_values(task, metric, r["scores"])[1] for seed, r in control_results[arm].items()
                }
                control = [sum(column) / len(column) for column in zip(*per_control.values(), strict=True)]
                paired = [p - c for p, c in zip(pooled, control, strict=True)]
                analysis_seed = plan.analysis_seed + 100 * arm_index + task_index
                arm_interval = cluster_bootstrap_ci(
                    paired,
                    [self._stratum(cluster) for cluster in ids],
                    level=plan.corrected_ci_level,
                    resamples=plan.bootstrap_resamples,
                    seed=analysis_seed,
                )
                arm_p = sign_flip_p_value(paired, resamples=plan.permutation_resamples, seed=analysis_seed)
                arm_seed_deltas = {
                    str(seed): (sum(per_seed[seed]) - sum(per_control[seed])) / len(per_seed[seed]) for seed in per_seed
                }
                control_rows.append((arm, task, base_values, pooled, control, arm_interval, arm_p, arm_seed_deltas))
        # One Holm family: every task against the base model and against every comparator.
        adjusted = holm_adjust([row[4] for row in rows] + [row[6] for row in control_rows])
        controls = tuple(
            ControlEffect(
                arm=arm,
                task=task,
                control_accuracy=sum(control) / len(control),
                candidate_accuracy=sum(pooled) / len(pooled),
                mean_delta=interval.mean,
                ci_low=interval.low,
                ci_high=interval.high,
                ci_level=interval.level,
                p_value=p_value,
                p_adjusted=p_adjusted,
                clusters=interval.clusters,
                seed_deltas=seed_deltas,
                control_minus_base=(sum(control) - sum(base_values)) / len(control),
            )
            for (arm, task, base_values, pooled, control, interval, p_value, seed_deltas), p_adjusted in zip(
                control_rows, adjusted[len(rows) :], strict=True
            )
        )
        effects = []
        for (task, base_values, pooled, interval, p_value, seed_deltas), p_adjusted in zip(
            rows, adjusted[: len(rows)], strict=True
        ):
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
            controls=controls,
        )

    @_serialized
    def decide(self) -> dict[str, Any]:
        self._check_plan()
        freeze = self._event("freeze")
        if freeze is None:
            raise Refused("NOT_FROZEN", "no manifest has been frozen")
        started = {e["candidate_digest"] for e in self.events if e["kind"] == "query_started"}
        if started != {c["candidate_digest"] for c in (*freeze["candidates"], *freeze["controls"])}:
            raise Refused("INCOMPLETE", "every frozen candidate and control must be queried before the decision")
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
        if operation == "audit":
            return self.audit(request.get("audit") or {})
        if operation == "decide":
            return self.decide()
        raise Refused("UNKNOWN_OPERATION", "the custodian answers only freeze, audit, query and decide")

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
    parser.add_argument("--required-uid", type=int,
                        help="Require a dedicated non-root deployment account and admin-owned code")
    parser.add_argument("--software-fixtures-only", action="store_true",
                        help="Refuse real-model/confirmatory plans in Stage 1 deployment")
    args = parser.parse_args(argv)

    # Take over stdout so nothing but protocol frames can reach the proposer.
    wire = os.fdopen(os.dup(1), "wb")
    requests = os.fdopen(os.dup(0), "rb")
    args.state_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    if args.required_uid is not None:
        if os.name == "nt" or args.required_uid == 0 or os.getuid() != args.required_uid:
            _send(wire, json.dumps({"ok": False, "code": "ACCOUNT_BOUNDARY_FAILED",
                                   "message": "dedicated account required"}).encode())
            return 2
        state_info = args.state_dir.stat()
        path = Path(__file__).resolve()
        while True:
            info = path.stat()
            if info.st_uid != 0 or info.st_mode & 0o022:
                _send(wire, json.dumps({"ok": False, "code": "DEPLOYMENT_NOT_PROTECTED",
                                       "message": "admin-owned immutable deployment required"}).encode())
                return 2
            if path.parent == path:
                break
            path = path.parent
        if state_info.st_uid != args.required_uid or state_info.st_mode & 0o077:
            _send(wire, json.dumps({"ok": False, "code": "STATE_NOT_PRIVATE",
                                   "message": "private account-owned state required"}).encode())
            return 2
    log = os.open(args.state_dir / "custodian.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.dup2(log, 1)
    os.dup2(log, 2)

    try:
        if args.software_fixtures_only:
            fixture_plan = load_frozen_plan(args.plan, args.plan_digest)
            if fixture_plan.label != "harness_test" or fixture_plan.evaluator != "synthetic_bernoulli":
                raise Refused("FIXTURE_ONLY", "this Stage 1 deployment accepts software fixtures only")
        custodian = Custodian(args.plan, args.plan_digest, args.state_dir)
        if args.required_uid is not None:
            custodian.account_boundary = "separate_os_account"
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
    _send(wire, json.dumps({"ok": True, "ready": True, "plan_digest": args.plan_digest,
                           "account_boundary": custodian.account_boundary}).encode())

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
