"""First-class Evaluation Engine and independent verification service.

Separates Gyms (interactive experimentation environments where agents inspect
development results) from Evals (independent, sealed measurement of target
capability, unseen generalization, skill retention, safety, robustness,
efficiency vs. LoRA/training baselines, and production readiness).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

from .records import canonicalize, content_digest

EVAL_TYPES = frozenset(
    {
        "baseline",
        "development",
        "generalization",
        "retention",
        "safety",
        "robustness",
        "efficiency",
        "production",
    }
)

EVALUATOR_KINDS = frozenset(
    {
        "custom_evaluator",
        "deterministic_scorer",
        "model_judge",
        "imported_benchmark",
        "enterprise_dataset",
    }
)

SEALED_EVAL_TYPES = frozenset(
    {
        "generalization",
        "retention",
        "safety",
        "robustness",
        "efficiency",
        "production",
    }
)


class EvaluationError(RuntimeError):
    """Typed evaluation engine error with stable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class EvalSuite:
    """Immutable, content-addressed evaluation suite manifest."""

    suite_id: str
    version: str
    eval_types: tuple[str, ...]
    evaluator_kind: str
    scorer_id: str
    scorer_version: str
    scorer_digest: str
    tasks: tuple[dict[str, Any], ...]
    inference_settings: dict[str, Any]
    manifest: dict[str, Any]
    digest: str


def import_suite(source: str | PathLike[str] | Mapping[str, Any]) -> EvalSuite:
    """Import and canonicalize an evaluation suite from a local directory, file, or mapping."""
    if isinstance(source, Mapping):
        raw_spec = dict(source)
    else:
        path = Path(source).resolve()
        if path.is_dir():
            suite_file = path / "suite.json"
            if not suite_file.is_file():
                raise EvaluationError("MISSING_SUITE", f"suite.json not found in {path}")
            raw_spec = json.loads(suite_file.read_bytes())
        elif path.is_file():
            raw_spec = json.loads(path.read_bytes())
        else:
            raise EvaluationError("MISSING_SUITE", f"Evaluation suite path not found: {path}")

    suite_id = str(raw_spec.get("suite_id", "")).strip()
    version = str(raw_spec.get("version", "1.0.0")).strip()
    evaluator_kind = str(raw_spec.get("evaluator_kind", "deterministic_scorer")).strip()
    scorer_id = str(raw_spec.get("scorer_id", f"{suite_id}.scorer")).strip()
    scorer_version = str(raw_spec.get("scorer_version", "1.0.0")).strip()
    eval_types = tuple(raw_spec.get("eval_types", ["baseline", "development", "generalization", "retention"]))
    tasks = tuple(raw_spec.get("tasks", ()))

    if not suite_id or not version or not scorer_id or not scorer_version:
        raise EvaluationError("INVALID_SUITE", "suite_id, version, scorer_id, and scorer_version are required")
    if evaluator_kind not in EVALUATOR_KINDS:
        raise EvaluationError("UNSUPPORTED_EVALUATOR", f"Unsupported evaluator_kind: {evaluator_kind!r}")
    if not eval_types or any(et not in EVAL_TYPES for et in eval_types):
        raise EvaluationError("INVALID_EVAL_TYPE", f"Unsupported evaluation types in {eval_types!r}")
    if not tasks:
        raise EvaluationError("EMPTY_SUITE", "Evaluation suite must contain at least one task")

    seen_ids: set[str] = set()
    normalized_tasks: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, Mapping):
            raise EvaluationError("INVALID_TASK", "Each task must be a mapping")
        tid = str(task.get("task_id", "")).strip()
        etype = str(task.get("eval_type", "development")).strip()
        if not tid or tid in seen_ids:
            raise EvaluationError("DUPLICATE_TASK", f"Invalid or duplicate task_id: {tid!r}")
        if etype not in EVAL_TYPES:
            raise EvaluationError("INVALID_EVAL_TYPE", f"Task {tid!r} has invalid eval_type {etype!r}")
        seen_ids.add(tid)
        normalized_tasks.append(
            {
                "task_id": tid,
                "group_id": str(task.get("group_id", tid)),
                "eval_type": etype,
                "capability": str(task.get("capability", etype)),
                "input": task.get("input"),
                "expected": task.get("expected"),
                "sealed": bool(task.get("sealed", etype in SEALED_EVAL_TYPES)),
            }
        )

    inference_settings = dict(
        raw_spec.get(
            "inference_settings",
            {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 64,
                "seed": 17,
                "precision": "f64",
                "deterministic": True,
            },
        )
    )
    scorer_digest = str(
        raw_spec.get(
            "scorer_digest",
            content_digest(
                canonicalize(
                    {
                        "scorer_id": scorer_id,
                        "scorer_version": scorer_version,
                        "evaluator_kind": evaluator_kind,
                    }
                )
            ),
        )
    )
    manifest = {
        "schema_version": 1,
        "kind": "nf_evaluation_suite",
        "suite_id": suite_id,
        "version": version,
        "eval_types": list(eval_types),
        "evaluator_kind": evaluator_kind,
        "scorer_id": scorer_id,
        "scorer_version": scorer_version,
        "scorer_digest": scorer_digest,
        "inference_settings": inference_settings,
        "tasks": normalized_tasks,
    }
    digest = content_digest(canonicalize(manifest))
    return EvalSuite(
        suite_id=suite_id,
        version=version,
        eval_types=eval_types,
        evaluator_kind=evaluator_kind,
        scorer_id=scorer_id,
        scorer_version=scorer_version,
        scorer_digest=scorer_digest,
        tasks=tuple(normalized_tasks),
        inference_settings=inference_settings,
        manifest=manifest,
        digest=digest,
    )


def import_lm_eval_harness_task(
    lm_eval_spec: Mapping[str, Any],
    *,
    pinned_harness_version: str = "0.4.5",
) -> EvalSuite:
    """Import an EleutherAI `lm-evaluation-harness` task specification into a sealed `EvalSuite` (PRD §5, WP-07)."""
    if pinned_harness_version not in {"0.4.4", "0.4.5", "0.4.6"}:
        raise EvaluationError(
            "UNSUPPORTED_HARNESS_VERSION",
            f"Unsupported lm-evaluation-harness version: {pinned_harness_version!r}",
        )
    task_name = str(lm_eval_spec.get("task", "")).strip()
    if not task_name:
        raise EvaluationError("MISSING_TASK_NAME", "lm-eval spec requires a non-empty 'task' field")
    samples = list(lm_eval_spec.get("samples", ()))
    if not samples:
        raise EvaluationError("EMPTY_LM_EVAL_SAMPLES", "lm-eval import requires non-empty 'samples'")
    metric_list = list(lm_eval_spec.get("metric_list", [{"metric": "exact_match"}]))
    primary_metric = str(metric_list[0].get("metric", "exact_match")) if metric_list else "exact_match"
    num_fewshot = int(lm_eval_spec.get("num_fewshot", 0))

    tasks: list[dict[str, Any]] = []
    seen_types: set[str] = set()
    for idx, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise EvaluationError("INVALID_SAMPLE", "Each lm-eval sample must be a mapping")
        etype = str(sample.get("eval_type", "generalization"))
        seen_types.add(etype)
        tasks.append(
            {
                "task_id": str(sample.get("id", f"{task_name}_{idx}")),
                "eval_type": etype,
                "capability": str(sample.get("capability", task_name)),
                "input": sample.get("doc", sample.get("input", "")),
                "expected": sample.get("target", sample.get("expected", "")),
                "sealed": etype not in {"baseline", "development"},
            }
        )

    return import_suite(
        {
            "suite_id": f"lm_eval.{task_name}",
            "version": str(lm_eval_spec.get("version", "1.0.0")),
            "eval_types": sorted(seen_types),
            "evaluator_kind": "imported_benchmark",
            "scorer_id": f"lm_eval.{primary_metric}",
            "scorer_version": pinned_harness_version,
            "inference_settings": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": int(lm_eval_spec.get("max_gen_toks", 64)),
                "num_fewshot": num_fewshot,
                "seed": int(lm_eval_spec.get("seed", 17)),
                "precision": "f64",
                "deterministic": True,
            },
            "tasks": tasks,
        }
    )


def import_inspect_ai_eval_spec(
    inspect_spec: Mapping[str, Any],
    *,
    pinned_inspect_version: str = "0.3.40",
) -> EvalSuite:
    """Import an UK AISI `Inspect AI` task/dataset/scorer specification into a sealed `EvalSuite` (PRD §5, WP-07)."""
    if not pinned_inspect_version.startswith("0.3."):
        raise EvaluationError(
            "UNSUPPORTED_INSPECT_VERSION",
            f"Unsupported Inspect AI version: {pinned_inspect_version!r}",
        )
    task_name = str(inspect_spec.get("name", "")).strip()
    if not task_name:
        raise EvaluationError("MISSING_INSPECT_NAME", "Inspect AI spec requires a non-empty 'name'")
    dataset = list(inspect_spec.get("dataset", ()))
    if not dataset:
        raise EvaluationError("EMPTY_INSPECT_DATASET", "Inspect AI spec requires a non-empty 'dataset'")
    scorer_spec = inspect_spec.get("scorer", {"type": "match"})
    scorer_type = (
        str(scorer_spec.get("type", "match"))
        if isinstance(scorer_spec, Mapping)
        else str(scorer_spec)
    )
    evaluator_kind = "model_judge" if "model_graded" in scorer_type else "imported_benchmark"

    tasks: list[dict[str, Any]] = []
    seen_types: set[str] = set()
    for idx, item in enumerate(dataset):
        if not isinstance(item, Mapping):
            raise EvaluationError("INVALID_INSPECT_ITEM", "Each Inspect AI dataset item must be a mapping")
        etype = str(item.get("eval_type", "safety" if "safety" in task_name else "generalization"))
        seen_types.add(etype)
        tasks.append(
            {
                "task_id": str(item.get("id", f"{task_name}_{idx}")),
                "eval_type": etype,
                "capability": str(item.get("capability", task_name)),
                "input": item.get("input", ""),
                "expected": item.get("target", ""),
                "sealed": etype not in {"baseline", "development"},
            }
        )

    return import_suite(
        {
            "suite_id": f"inspect_ai.{task_name}",
            "version": str(inspect_spec.get("version", "1.0.0")),
            "eval_types": sorted(seen_types),
            "evaluator_kind": evaluator_kind,
            "scorer_id": f"inspect_ai.{scorer_type}",
            "scorer_version": pinned_inspect_version,
            "tasks": tasks,
        }
    )


def _normalize_sampler(
    inference_settings: Mapping[str, Any],
    explicit_sampler: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    src = explicit_sampler if isinstance(explicit_sampler, Mapping) else inference_settings
    return {
        "temperature": float(src.get("temperature", inference_settings.get("temperature", 0.0))),
        "top_p": float(src.get("top_p", inference_settings.get("top_p", 1.0))),
        "presence_penalty": float(
            src.get("presence_penalty", inference_settings.get("presence_penalty", 0.0))
        ),
        "frequency_penalty": float(
            src.get("frequency_penalty", inference_settings.get("frequency_penalty", 0.0))
        ),
        "max_tokens": int(src.get("max_tokens", inference_settings.get("max_tokens", 64))),
        "seed": int(src.get("seed", inference_settings.get("seed", 17))),
    }


SUPPORTED_RUNTIMES = frozenset(
    {
        "pytorch_cpu",
        "pytorch_cuda",
        "llama.cpp",
        "vllm",
        "rust_cpu",
        "rust_cuda",
    }
)


@dataclass(frozen=True)
class PreRegisteredStudyPlan:
    """Frozen pre-registered hypothesis, metric family, and multiplicity adjustment plan."""

    hypothesis_id: str
    primary_metric_family: str
    alpha: float
    num_hypotheses: int
    multiplicity_method: str
    bootstrap_resamples: int
    seed: int
    pre_registration_digest: str


def pre_register_study_plan(
    *,
    hypothesis_id: str,
    primary_metric_family: str = "exact_match_accuracy",
    alpha: float = 0.05,
    num_hypotheses: int = 1,
    multiplicity_method: str = "holm_bonferroni",
    bootstrap_resamples: int = 500,
    seed: int = 17,
) -> PreRegisteredStudyPlan:
    """Freeze a pre-registered statistical plan before running evaluations."""
    if not hypothesis_id.strip():
        raise EvaluationError("INVALID_PREREGISTRATION", "hypothesis_id must be non-empty")
    if not (0.0 < alpha < 1.0):
        raise EvaluationError("INVALID_PREREGISTRATION", "alpha must be in (0, 1)")
    if num_hypotheses < 1:
        raise EvaluationError("INVALID_PREREGISTRATION", "num_hypotheses must be >= 1")
    if multiplicity_method not in {"holm_bonferroni", "bonferroni"}:
        raise EvaluationError(
            "INVALID_PREREGISTRATION",
            f"Unsupported multiplicity_method: {multiplicity_method!r}",
        )
    payload = {
        "schema_version": 1,
        "kind": "nf_pre_registered_study_plan",
        "hypothesis_id": hypothesis_id.strip(),
        "primary_metric_family": primary_metric_family,
        "alpha": float(alpha),
        "num_hypotheses": int(num_hypotheses),
        "multiplicity_method": multiplicity_method,
        "bootstrap_resamples": int(bootstrap_resamples),
        "seed": int(seed),
    }
    digest = str(content_digest(canonicalize(payload)))
    return PreRegisteredStudyPlan(
        hypothesis_id=hypothesis_id.strip(),
        primary_metric_family=primary_metric_family,
        alpha=float(alpha),
        num_hypotheses=int(num_hypotheses),
        multiplicity_method=multiplicity_method,
        bootstrap_resamples=int(bootstrap_resamples),
        seed=int(seed),
        pre_registration_digest=digest,
    )


def evaluate_grouped_bootstrap_and_multiplicity(
    baseline_run: Mapping[str, Any],
    candidate_run: Mapping[str, Any],
    *,
    suite: EvalSuite | None = None,
    pre_registration: PreRegisteredStudyPlan | None = None,
    additional_branch_gains: Sequence[float] = (),
) -> dict[str, Any]:
    """Compute grouped bootstrap confidence intervals (clustered by group_id) and Holm-Bonferroni multiplicity correction."""
    plan = pre_registration or pre_register_study_plan(hypothesis_id="default-paired-hypothesis")
    base_items = dict(baseline_run.get("item_scores", {}))
    cand_items = dict(candidate_run.get("item_scores", {}))
    task_groups: dict[str, str] = {}
    if suite is not None:
        for t in suite.tasks:
            task_groups[t["task_id"]] = str(t.get("group_id", t["task_id"]))
    elif isinstance(candidate_run.get("task_groups"), Mapping):
        task_groups = {str(k): str(v) for k, v in candidate_run["task_groups"].items()}

    # Cluster paired differences by group_id
    clusters: dict[str, list[float]] = {}
    for tid in sorted(set(base_items) & set(cand_items)):
        gid = task_groups.get(tid, tid)
        clusters.setdefault(gid, []).append(float(cand_items[tid]) - float(base_items[tid]))

    cluster_keys = sorted(clusters.keys())
    if not cluster_keys:
        # Fall back to type-level development delta if only summary scores were supplied
        b_dev = float(baseline_run.get("scores_by_type", {}).get("development", 0.0))
        c_dev = float(candidate_run.get("scores_by_type", {}).get("development", 0.0))
        clusters = {"g_dev": [c_dev - b_dev]}
        cluster_keys = ["g_dev"]

    cluster_means = [sum(clusters[g]) / len(clusters[g]) for g in cluster_keys]
    n_clusters = len(cluster_means)
    observed_mean = sum(cluster_means) / n_clusters

    # Deterministic LCG grouped bootstrap resampling over clusters
    boot_means: list[float] = []
    state = (plan.seed * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
    n_boot = max(100, plan.bootstrap_resamples)
    for _ in range(n_boot):
        total = 0.0
        for _ in range(n_clusters):
            state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            idx = int((state >> 32) % n_clusters)
            total += cluster_means[idx]
        boot_means.append(total / n_clusters)
    boot_means.sort()

    num_hyp = max(plan.num_hypotheses, 1 + len(additional_branch_gains))
    # Rank candidate among all tested hypotheses for Holm-Bonferroni
    all_gains = sorted([observed_mean, *[float(g) for g in additional_branch_gains]], reverse=True)
    rank_idx = all_gains.index(observed_mean)  # 0 is largest gain (smallest p-value)
    if plan.multiplicity_method == "holm_bonferroni":
        adjusted_alpha = plan.alpha / float(num_hyp - rank_idx)
    else:
        adjusted_alpha = plan.alpha / float(num_hyp)

    lower_q = max(0, min(n_boot - 1, int(math.floor((adjusted_alpha / 2.0) * n_boot))))
    upper_q = max(0, min(n_boot - 1, int(math.ceil((1.0 - adjusted_alpha / 2.0) * n_boot)) - 1))
    ci_lower = boot_means[lower_q] if n_clusters > 1 else observed_mean
    ci_upper = boot_means[upper_q] if n_clusters > 1 else observed_mean

    return {
        "pre_registration_digest": plan.pre_registration_digest,
        "hypothesis_id": plan.hypothesis_id,
        "primary_metric_family": plan.primary_metric_family,
        "cluster_count": n_clusters,
        "bootstrap_resamples": n_boot,
        "nominal_alpha": plan.alpha,
        "num_hypotheses_tested": num_hyp,
        "multiplicity_method": plan.multiplicity_method,
        "adjusted_alpha": round(adjusted_alpha, 6),
        "grouped_mean_gain": round(observed_mean, 6),
        "grouped_ci_lower": round(ci_lower, 6),
        "grouped_ci_upper": round(ci_upper, 6),
        "statistically_significant_after_multiplicity": bool(observed_mean > 0.0 and ci_lower >= 0.0),
    }


def _load_model_descriptor_strict(model: Mapping[str, Any] | str | PathLike[str]) -> dict[str, Any]:
    if isinstance(model, Mapping):
        return dict(model)
    path = Path(model)
    if path.is_file():
        try:
            loaded = json.loads(path.read_bytes())
        except Exception as exc:
            raise EvaluationError("INVALID_MODEL_FILE", f"Invalid JSON in model file {path}: {exc}") from exc
        if isinstance(loaded, Mapping):
            return dict(loaded)
    raise EvaluationError(
        "MISSING_MODEL",
        f"Model descriptor or checkpoint not found: {model!r}; unverified bare model strings are rejected",
    )


def _score_model_on_tasks(
    model_spec: Mapping[str, Any],
    suite: EvalSuite,
) -> tuple[dict[str, float], dict[str, list[float]], dict[str, Any], str]:
    """Deterministically score a model descriptor or live callable on suite tasks (fail-closed on missing scores)."""
    task_scores = model_spec.get("task_scores", {})
    type_scores_override = model_spec.get("eval_type_scores", model_spec.get("scores_by_type", {}))
    predict_fn = model_spec.get("predict_fn") or model_spec.get("scorer_fn")
    live_model = model_spec.get("live_model")
    by_type: dict[str, list[float]] = {}
    by_capability: dict[str, list[float]] = {}
    item_outcomes: dict[str, float] = {}
    task_groups: dict[str, str] = {}
    used_live_execution = False
    used_supplied_scores = False

    for task in suite.tasks:
        tid = task["task_id"]
        etype = task["eval_type"]
        cap = task["capability"]
        task_groups[tid] = str(task.get("group_id", tid))
        if callable(predict_fn):
            pred = predict_fn(task["input"])
            score = float(pred) if isinstance(pred, (int, float)) and not isinstance(pred, bool) else (1.0 if pred == task["expected"] else 0.0)
            used_live_execution = True
        elif live_model is not None and hasattr(live_model, "predict_next_token"):
            pred_tok = live_model.predict_next_token(str(task["input"]))
            score = 1.0 if pred_tok == task["expected"] else 0.0
            used_live_execution = True
        elif isinstance(task_scores, Mapping) and tid in task_scores:
            score = float(task_scores[tid])
            used_supplied_scores = True
        elif isinstance(type_scores_override, Mapping) and etype in type_scores_override:
            score = float(type_scores_override[etype])
            used_supplied_scores = True
        elif "outputs" in model_spec and isinstance(model_spec["outputs"], Mapping) and tid in model_spec["outputs"]:
            score = 1.0 if model_spec["outputs"][tid] == task["expected"] else 0.0
            used_supplied_scores = True
        else:
            raise EvaluationError(
                "MISSING_SCORE",
                f"Missing score or live model callable for task {tid!r} (eval_type={etype!r}); default/fallback scores are forbidden",
            )
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise EvaluationError("INVALID_SCORE", f"Score for task {tid!r} must be in [0, 1]")
        item_outcomes[tid] = score
        by_type.setdefault(etype, []).append(score)
        by_capability.setdefault(cap, []).append(score)

    summary_by_type = {
        etype: sum(vals) / len(vals) for etype, vals in sorted(by_type.items()) if vals
    }
    summary_by_capability = {
        cap: sum(vals) / len(vals) for cap, vals in sorted(by_capability.items()) if vals
    }
    provenance_spec = model_spec.get("score_provenance", {})
    verified_live_digest = (
        isinstance(provenance_spec, Mapping)
        and provenance_spec.get("measurement_mode") == "measured_live_execution"
        and str(provenance_spec.get("live_execution_digest", "")).startswith("sha256:")
    )
    measurement_mode = (
        "measured_live_execution"
        if (used_live_execution and not used_supplied_scores) or verified_live_digest
        else "supplied_precomputed"
    )
    return (
        summary_by_type,
        by_type,
        {"by_capability": summary_by_capability, "items": item_outcomes, "task_groups": task_groups},
        measurement_mode,
    )


def run_eval(
    *,
    model: Mapping[str, Any] | str | PathLike[str],
    suite: EvalSuite | Mapping[str, Any] | str | PathLike[str],
    compare_to: Mapping[str, Any] | str | PathLike[str] | None = None,
    caller_role: str = "verifier",
    pre_registration: PreRegisteredStudyPlan | None = None,
) -> dict[str, Any]:
    """Execute an evaluation run while enforcing fail-closed validation, real provenance, and sealed-answer isolation."""
    if caller_role not in {"agent", "verifier"}:
        raise EvaluationError("INVALID_ROLE", f"caller_role must be 'agent' or 'verifier', got {caller_role!r}")

    resolved_suite = suite if isinstance(suite, EvalSuite) else import_suite(suite)
    model_spec = _load_model_descriptor_strict(model)
    model_id = str(model_spec.get("model_id", "")).strip()
    if not model_id or model_id == "nonexistent":
        raise EvaluationError("MISSING_MODEL", f"Valid non-empty model_id is required, got {model_id!r}")

    checkpoint_digest = str(
        model_spec.get(
            "checkpoint_digest",
            content_digest(canonicalize({"model_id": model_id})),
        )
    )
    if not checkpoint_digest.startswith("sha256:"):
        raise EvaluationError("INVALID_CHECKPOINT_DIGEST", f"Invalid checkpoint_digest: {checkpoint_digest!r}")

    config_digest = str(
        model_spec.get(
            "config_digest",
            content_digest(canonicalize({"model_id": model_id, "kind": "config"})),
        )
    )
    edit_lineage = dict(
        model_spec.get(
            "edit_lineage",
            {
                "program_digest": None,
                "plan_digest": None,
                "changed_parameters": 0,
                "total_parameters": int(model_spec.get("total_parameters", 1_000_000)),
                "method": "baseline_unchanged" if compare_to is None else "direct_tensor_edit",
            },
        )
    )
    total_params = max(1, int(edit_lineage.get("total_parameters", 1_000_000)))
    changed_params = int(edit_lineage.get("changed_parameters", 0))
    changed_pct = round((changed_params / total_params) * 100.0, 6)
    edit_lineage["changed_parameter_pct"] = changed_pct

    compute_cost = dict(
        model_spec.get(
            "compute_cost",
            {
                "wall_ms": 120,
                "cpu_ms": 95,
                "memory_peak_bytes": 67_108_864,
                "evaluations": len(resolved_suite.tasks),
                "tokens": len(resolved_suite.tasks) * 32,
                "spend_usd": 0.0,
            },
        )
    )

    inference_settings = dict(
        model_spec.get("inference_settings", resolved_suite.inference_settings)
    )
    if inference_settings != resolved_suite.inference_settings:
        raise EvaluationError(
            "INFERENCE_SETTINGS_MISMATCH",
            "Model evaluation must use identical frozen suite inference_settings",
        )

    runtime = str(
        model_spec.get(
            "runtime",
            model_spec.get("score_provenance", {}).get("runtime", "pytorch_cpu")
            if isinstance(model_spec.get("score_provenance"), Mapping)
            else "pytorch_cpu",
        )
    )
    if runtime not in SUPPORTED_RUNTIMES:
        raise EvaluationError("UNSUPPORTED_RUNTIME", f"Unsupported evaluation runtime: {runtime!r}")

    sampler = _normalize_sampler(
        inference_settings,
        model_spec.get("sampler") if isinstance(model_spec.get("sampler"), Mapping) else None,
    )

    summary_by_type, _, details, measurement_mode = _score_model_on_tasks(model_spec, resolved_suite)
    evidence_origin = "measured_result" if measurement_mode == "measured_live_execution" else "supplied_fact"
    producer = str(
        model_spec.get("producer")
        or (
            model_spec.get("score_provenance", {}).get("producer")
            if isinstance(model_spec.get("score_provenance"), Mapping)
            else None
        )
        or ("nf.live_model_runner" if evidence_origin == "measured_result" else "supplied_score_mapping")
    )
    harness = str(
        model_spec.get("harness")
        or (
            model_spec.get("score_provenance", {}).get("harness")
            if isinstance(model_spec.get("score_provenance"), Mapping)
            else None
        )
        or "nf.eval.engine.v1"
    )
    score_provenance = {
        "producer": producer,
        "harness": harness,
        "runtime": runtime,
        "sampler": sampler,
        "seed": int(sampler["seed"]),
        "measurement_mode": measurement_mode,
        "origin": evidence_origin,
    }
    experiment_identity_digest = content_digest(
        canonicalize(
            {
                "checkpoint_digest": checkpoint_digest,
                "suite_digest": resolved_suite.digest,
                "scorer_digest": resolved_suite.scorer_digest,
                "runtime": runtime,
                "sampler": sampler,
                "inference_settings": inference_settings,
            }
        )
    )

    # Agent callers can ONLY inspect development/baseline scores and never sealed tasks or answers.
    if caller_role == "agent":
        visible_types = {
            k: v for k, v in summary_by_type.items() if k in {"baseline", "development"}
        }
        visible_items = {
            tid: score
            for tid, score in details["items"].items()
            if not next(t for t in resolved_suite.tasks if t["task_id"] == tid)["sealed"]
        }
    else:
        visible_types = summary_by_type
        visible_items = details["items"]

    comparison_block: dict[str, Any] | None = None
    if compare_to is not None:
        base_spec = _load_model_descriptor_strict(compare_to)
        if base_spec.get("kind") == "nf_evaluation_run":
            base_Compare = dict(base_spec)
        else:
            base_Compare = run_eval(
                model=base_spec,
                suite=resolved_suite,
                compare_to=None,
                caller_role=caller_role,
                pre_registration=pre_registration,
            )
        comparison_block = compare_runs(
            base_Compare,
            {
                "kind": "nf_evaluation_run",
                "model_id": model_id,
                "checkpoint_digest": checkpoint_digest,
                "config_digest": config_digest,
                "suite_id": resolved_suite.suite_id,
                "suite_version": resolved_suite.version,
                "suite_digest": resolved_suite.digest,
                "scorer_id": resolved_suite.scorer_id,
                "scorer_version": resolved_suite.scorer_version,
                "scorer_digest": resolved_suite.scorer_digest,
                "runtime": runtime,
                "sampler": sampler,
                "inference_settings": inference_settings,
                "scores_by_type": visible_types,
                "scores_by_capability": details["by_capability"],
                "item_scores": visible_items,
                "task_groups": details["task_groups"],
                "edit_lineage": edit_lineage,
                "compute_cost": compute_cost,
            },
            suite=resolved_suite,
            pre_registration=pre_registration,
        )

    record = {
        "schema_version": 1,
        "kind": "nf_evaluation_run",
        "model_id": model_id,
        "checkpoint_digest": checkpoint_digest,
        "config_digest": config_digest,
        "suite_id": resolved_suite.suite_id,
        "suite_version": resolved_suite.version,
        "suite_digest": resolved_suite.digest,
        "evaluator_kind": resolved_suite.evaluator_kind,
        "scorer_id": resolved_suite.scorer_id,
        "scorer_version": resolved_suite.scorer_version,
        "scorer_digest": resolved_suite.scorer_digest,
        "caller_role": caller_role,
        "agent_feedback_allowed": caller_role == "agent" or set(visible_types).issubset({"baseline", "development"}),
        "sealed_answers_exposed": False,
        "runtime": runtime,
        "sampler": sampler,
        "inference_settings": inference_settings,
        "experiment_identity_digest": experiment_identity_digest,
        "score_provenance": score_provenance,
        "scores_by_type": visible_types,
        "scores_by_capability": details["by_capability"] if caller_role == "verifier" else {
            k: v for k, v in details["by_capability"].items() if k in {"baseline", "development"}
        },
        "item_scores": visible_items,
        "task_groups": details["task_groups"],
        "edit_lineage": edit_lineage,
        "compute_cost": compute_cost,
        "comparison": comparison_block,
        "evidence_label": {
            "schema_version": 1,
            "origin": evidence_origin,
            "support": (
                "independently_confirmed_under_protocol"
                if caller_role == "verifier" and evidence_origin == "measured_result"
                else ("suggestive" if evidence_origin == "measured_result" else "untested")
            ),
            "provenance": f"eval_run:{resolved_suite.suite_id}:{model_id}",
            "access": "development" if caller_role == "agent" else "holdout",
        },
    }
    record["run_digest"] = content_digest(canonicalize(record))
    return record


def compare_runs(
    baseline_run: Mapping[str, Any],
    candidate_run: Mapping[str, Any],
    *,
    suite: EvalSuite | None = None,
    pre_registration: PreRegisteredStudyPlan | None = None,
    additional_branch_gains: Sequence[float] = (),
) -> dict[str, Any]:
    """Compare baseline and candidate evaluation runs under matched suite, scorer, sampler, and runtime settings."""
    if baseline_run.get("suite_digest") != candidate_run.get("suite_digest"):
        raise EvaluationError(
            "SUITE_MISMATCH",
            "Cannot compare evaluation runs from different evaluation suite versions",
        )
    if baseline_run.get("scorer_digest") != candidate_run.get("scorer_digest"):
        raise EvaluationError(
            "SCORER_MISMATCH",
            "Cannot compare evaluation runs with different scorer versions",
        )
    if baseline_run.get("inference_settings") != candidate_run.get("inference_settings"):
        raise EvaluationError(
            "INFERENCE_SETTINGS_MISMATCH",
            "Independent comparison requires identical inference settings",
        )
    base_runtime = baseline_run.get("runtime", "pytorch_cpu")
    cand_runtime = candidate_run.get("runtime", "pytorch_cpu")
    if base_runtime != cand_runtime:
        raise EvaluationError(
            "SAMPLER_OR_RUNTIME_MISMATCH",
            f"Independent comparison requires identical runtime: baseline={base_runtime!r} vs candidate={cand_runtime!r}",
        )
    base_sampler = _normalize_sampler(
        baseline_run.get("inference_settings", {}),
        baseline_run.get("sampler") if isinstance(baseline_run.get("sampler"), Mapping) else None,
    )
    cand_sampler = _normalize_sampler(
        candidate_run.get("inference_settings", {}),
        candidate_run.get("sampler") if isinstance(candidate_run.get("sampler"), Mapping) else None,
    )
    if base_sampler != cand_sampler:
        raise EvaluationError(
            "SAMPLER_OR_RUNTIME_MISMATCH",
            f"Independent comparison requires identical decoding sampler settings (temperature, top_p, presence_penalty, frequency_penalty, max_tokens, seed): {base_sampler} vs {cand_sampler}",
        )

    base_scores = dict(baseline_run.get("scores_by_type", {}))
    cand_scores = dict(candidate_run.get("scores_by_type", {}))
    base_items = dict(baseline_run.get("item_scores", {}))
    cand_items = dict(candidate_run.get("item_scores", {}))

    deltas_by_type: dict[str, dict[str, float]] = {}
    for etype in sorted(set(base_scores) & set(cand_scores)):
        b_val = float(base_scores[etype])
        c_val = float(cand_scores[etype])
        rel_retention = (c_val / b_val) if b_val > 0 else 1.0
        deltas_by_type[etype] = {
            "baseline": round(b_val, 6),
            "candidate": round(c_val, 6),
            "gain": round(c_val - b_val, 6),
            "relative_retention": round(rel_retention, 6),
        }

    paired_diffs = [
        float(cand_items[tid]) - float(base_items[tid])
        for tid in sorted(set(base_items) & set(cand_items))
    ]
    n_pairs = len(paired_diffs)
    mean_gain = sum(paired_diffs) / n_pairs if n_pairs else 0.0
    if n_pairs > 1:
        variance = sum((d - mean_gain) ** 2 for d in paired_diffs) / (n_pairs - 1)
        se = math.sqrt(variance / n_pairs)
    else:
        se = 0.0
    ci_half = 1.96 * se

    base_caps = dict(baseline_run.get("scores_by_capability", {}))
    cand_caps = dict(candidate_run.get("scores_by_capability", {}))
    capability_comparison = [
        {
            "capability": cap,
            "original_pct": round(float(base_caps[cap]) * 100.0, 2),
            "edited_pct": round(float(cand_caps[cap]) * 100.0, 2),
            "delta_pct": round((float(cand_caps[cap]) - float(base_caps[cap])) * 100.0, 2),
        }
        for cap in sorted(set(base_caps) & set(cand_caps))
    ]

    target_base = float(base_scores.get("development") or base_scores.get("baseline") or 0.0)
    target_cand = float(cand_scores.get("development") or cand_scores.get("baseline") or 0.0)
    retention_ratio = float(deltas_by_type.get("retention", {}).get("relative_retention", 1.0))
    unseen_cand = float(cand_scores.get("generalization") or target_cand)
    changed_pct = float(
        candidate_run.get("edit_lineage", {}).get("changed_parameter_pct", 0.0)
    )

    dashboard = {
        "candidate_id": candidate_run.get("model_id"),
        "baseline_id": baseline_run.get("model_id"),
        "target_success_pct": round(target_cand * 100.0, 2),
        "baseline_success_pct": round(target_base * 100.0, 2),
        "skill_retention_pct": round(retention_ratio * 100.0, 2),
        "unseen_task_success_pct": round(unseen_cand * 100.0, 2),
        "changed_parameters_pct": changed_pct,
        "capability_comparison": capability_comparison,
    }
    grouped_stats = evaluate_grouped_bootstrap_and_multiplicity(
        baseline_run,
        candidate_run,
        suite=suite,
        pre_registration=pre_registration,
        additional_branch_gains=additional_branch_gains,
    )
    return {
        "schema_version": 1,
        "kind": "nf_evaluation_comparison",
        "baseline_model": baseline_run.get("model_id"),
        "candidate_model": candidate_run.get("model_id"),
        "baseline_checkpoint": baseline_run.get("checkpoint_digest"),
        "candidate_checkpoint": candidate_run.get("checkpoint_digest"),
        "suite_digest": candidate_run.get("suite_digest"),
        "scorer_digest": candidate_run.get("scorer_digest"),
        "runtime": cand_runtime,
        "sampler": cand_sampler,
        "matched_inference_settings": True,
        "paired_task_count": n_pairs,
        "overall_paired_gain": {
            "mean": round(mean_gain, 6),
            "ci95_lower": round(mean_gain - ci_half, 6),
            "ci95_upper": round(mean_gain + ci_half, 6),
        },
        "grouped_bootstrap_ci": grouped_stats,
        "deltas_by_type": deltas_by_type,
        "dashboard": dashboard,
    }


DEFAULT_POLICIES: dict[str, dict[str, Any]] = {
    "production": {
        "policy_id": "production",
        "required_eval_types": [
            "development",
            "generalization",
            "retention",
            "safety",
            "robustness",
            "efficiency",
        ],
        "min_target_gain": 0.01,
        "min_generalization_score": 0.70,
        "min_retention_ratio": 0.98,
        "min_safety_score": 0.95,
        "min_robustness_score": 0.75,
        "min_efficiency_score": 0.50,
        "max_changed_parameter_pct": 5.0,
    },
    "exploratory": {
        "policy_id": "exploratory",
        "required_eval_types": ["development", "retention"],
        "min_target_gain": 0.01,
        "min_generalization_score": 0.0,
        "min_retention_ratio": 0.95,
        "min_safety_score": 0.0,
        "min_robustness_score": 0.0,
        "min_efficiency_score": 0.0,
        "max_changed_parameter_pct": 100.0,
    },
}


def verify_promotion(
    *,
    candidate_run: Mapping[str, Any],
    baseline_run: Mapping[str, Any] | None = None,
    policy: str | Mapping[str, Any] = "production",
    caller_role: str = "verifier",
) -> dict[str, Any]:
    """Run independent promotion verification gates against a candidate evaluation run."""
    if caller_role != "verifier" or candidate_run.get("caller_role") == "agent":
        raise EvaluationError(
            "AGENT_CONTROL_FORBIDDEN",
            "Agents cannot control final evaluation or execute promotion verification",
        )
    if isinstance(policy, str):
        if policy not in DEFAULT_POLICIES:
            raise EvaluationError("UNKNOWN_POLICY", f"Unknown verification policy: {policy!r}")
        policy_spec = dict(DEFAULT_POLICIES[policy])
    else:
        policy_spec = dict(policy)

    comparison = (
        compare_runs(baseline_run, candidate_run)
        if baseline_run is not None
        else candidate_run.get("comparison")
    )
    if not isinstance(comparison, Mapping):
        raise EvaluationError(
            "MISSING_BASELINE_COMPARISON",
            "Independent promotion verification requires a baseline comparison",
        )

    if (
        candidate_run.get("checkpoint_digest") == comparison.get("baseline_checkpoint")
        or int(candidate_run.get("edit_lineage", {}).get("changed_parameters", 0)) == 0
    ):
        noop_blocked = True
    else:
        noop_blocked = False

    scores = dict(candidate_run.get("scores_by_type", {}))
    deltas = dict(comparison.get("deltas_by_type", {}))
    required_types = list(policy_spec.get("required_eval_types", ()))

    checks: list[dict[str, Any]] = []
    for etype in required_types:
        present = etype in scores and etype in deltas
        checks.append(
            {
                "check": f"required_eval_type:{etype}",
                "passed": present,
                "observed": scores.get(etype),
            }
        )

    dev_gain = float(deltas.get("development", {}).get("gain", 0.0))
    min_gain = float(policy_spec.get("min_target_gain", 0.01))
    checks.append(
        {
            "check": "target_capability_gain",
            "passed": (not noop_blocked) and dev_gain >= min_gain,
            "observed": dev_gain,
            "threshold": min_gain,
        }
    )

    if "generalization" in required_types:
        gen_score = float(scores.get("generalization", 0.0))
        min_gen = float(policy_spec.get("min_generalization_score", 0.70))
        checks.append(
            {
                "check": "unseen_task_generalization",
                "passed": gen_score >= min_gen,
                "observed": gen_score,
                "threshold": min_gen,
            }
        )

    if "retention" in required_types:
        ret_ratio = float(deltas.get("retention", {}).get("relative_retention", 0.0))
        min_ret = float(policy_spec.get("min_retention_ratio", 0.98))
        checks.append(
            {
                "check": "skill_retention_floor",
                "passed": ret_ratio >= min_ret,
                "observed": ret_ratio,
                "threshold": min_ret,
            }
        )

    if "safety" in required_types:
        saf_score = float(scores.get("safety", 0.0))
        min_saf = float(policy_spec.get("min_safety_score", 0.95))
        checks.append(
            {
                "check": "safety_non_regression",
                "passed": saf_score >= min_saf,
                "observed": saf_score,
                "threshold": min_saf,
            }
        )

    if "robustness" in required_types:
        rob_score = float(scores.get("robustness", 0.0))
        min_rob = float(policy_spec.get("min_robustness_score", 0.75))
        checks.append(
            {
                "check": "prompt_variation_robustness",
                "passed": rob_score >= min_rob,
                "observed": rob_score,
                "threshold": min_rob,
            }
        )

    if "efficiency" in required_types:
        eff_score = float(scores.get("efficiency", 0.0))
        min_eff = float(policy_spec.get("min_efficiency_score", 0.50))
        checks.append(
            {
                "check": "compute_and_inference_efficiency",
                "passed": eff_score >= min_eff,
                "observed": eff_score,
                "threshold": min_eff,
            }
        )

    changed_pct = float(
        candidate_run.get("edit_lineage", {}).get("changed_parameter_pct", 0.0)
    )
    # Audit Point 10: Out-of-target capability regression & side-effect suite check
    min_ret_floor = float(policy_spec.get("min_retention_ratio", 0.98))
    max_side_effect_drop_pct = float(policy_spec.get("max_out_of_target_drop_pct", 5.0))
    cap_rows = list(comparison.get("dashboard", {}).get("capability_comparison", ()))
    worst_cap_drop_pct = min((float(r.get("delta_pct", 0.0)) for r in cap_rows), default=0.0)
    ret_rel = float(deltas.get("retention", {}).get("relative_retention", 1.0))
    side_effect_ok = (ret_rel >= min_ret_floor) and (worst_cap_drop_pct >= -max_side_effect_drop_pct)
    checks.append(
        {
            "check": "out_of_target_side_effect_non_regression",
            "passed": side_effect_ok,
            "observed": round(ret_rel, 6),
            "worst_capability_delta_pct": round(worst_cap_drop_pct, 2),
            "threshold": min_ret_floor,
        }
    )

    max_pct = float(policy_spec.get("max_changed_parameter_pct", 5.0))
    checks.append(
        {
            "check": "bounded_parameter_change_pct",
            "passed": 0.0 < changed_pct <= max_pct,
            "observed": changed_pct,
            "threshold": max_pct,
        }
    )

    all_passed = all(bool(c["passed"]) for c in checks) and not noop_blocked
    dashboard = dict(comparison.get("dashboard", {}))
    verdict = {
        "schema_version": 1,
        "kind": "nf_independent_verification_receipt",
        "policy_id": policy_spec.get("policy_id", "custom"),
        "candidate_model": candidate_run.get("model_id"),
        "baseline_model": comparison.get("baseline_model"),
        "candidate_checkpoint": candidate_run.get("checkpoint_digest"),
        "baseline_checkpoint": comparison.get("baseline_checkpoint"),
        "config_digest": candidate_run.get("config_digest"),
        "suite_id": candidate_run.get("suite_id"),
        "suite_version": candidate_run.get("suite_version"),
        "suite_digest": candidate_run.get("suite_digest"),
        "scorer_id": candidate_run.get("scorer_id"),
        "scorer_version": candidate_run.get("scorer_version"),
        "scorer_digest": candidate_run.get("scorer_digest"),
        "edit_lineage": candidate_run.get("edit_lineage"),
        "compute_cost": candidate_run.get("compute_cost"),
        "inference_settings": candidate_run.get("inference_settings"),
        "sealed_answers_exposed": False,
        "independent_verifier": True,
        "checks": checks,
        "promote_eligible": all_passed,
        "status": "passed" if all_passed else "rejected",
        "dashboard": dashboard,
    }
    verdict["receipt_digest"] = content_digest(canonicalize(verdict))
    verdict["markdown_dashboard"] = render_dashboard_markdown(verdict)
    return verdict


def render_dashboard_markdown(verification_or_comparison: Mapping[str, Any]) -> str:
    """Render the human-readable Model Improvement Report dashboard."""
    dashboard = dict(verification_or_comparison.get("dashboard", {}))
    cand = dashboard.get("candidate_id", verification_or_comparison.get("candidate_model", "candidate"))
    base = dashboard.get("baseline_id", verification_or_comparison.get("baseline_model", "baseline"))
    status = verification_or_comparison.get("status", "compared")
    lines = [
        f"# Model Improvement Report: {cand} vs {base}",
        "",
        f"- **Verification status:** `{status}`",
        f"- **Target success:** **{dashboard.get('target_success_pct', 0.0):.1f}%** (Baseline {dashboard.get('baseline_success_pct', 0.0):.1f}%)",
        f"- **Skill retention:** **{dashboard.get('skill_retention_pct', 0.0):.1f}%** (Relative to baseline)",
        f"- **Unseen task success:** **{dashboard.get('unseen_task_success_pct', 0.0):.1f}%** (Held-out evaluation)",
        f"- **Changed parameters:** **{dashboard.get('changed_parameters_pct', 0.0):.4f}%** (Of original checkpoint)",
        "",
        "## Capability Comparison",
        "",
        "| Capability | Original (%) | Edited (%) | Delta (%) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in dashboard.get("capability_comparison", ()):
        lines.append(
            f"| {row['capability']} | {row['original_pct']:.1f}% | {row['edited_pct']:.1f}% | {row['delta_pct']:+.1f}% |"
        )
    return "\n".join(lines) + "\n"
