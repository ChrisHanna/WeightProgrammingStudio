"""Load frozen search artifacts and evaluate them without reading synthesis state."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch

from . import independent_eval, promotion_gate
from .edit_plan import tensor_state_digest
from .goal_search import CandidateArtifact, FrontierSnapshot, validate_frontier_snapshot
from .provenance import inference_identity_digest, resolve_checkpoint
from .records import canonicalize, content_digest


class FrontierEvaluationError(ValueError):
    """A frozen frontier cannot be materialized or evaluated safely."""


MAX_EXPOSURE_RETRIES = 3


MAX_CANDIDATE_ARTIFACT_BYTES = 512 * 1024 * 1024


_ACTIVE_EVALUATION_LEASE: ContextVar[Path | None] = ContextVar("evaluation_lease", default=None)
_ACTIVE_EVALUATION_DESCRIPTOR: ContextVar[int | None] = ContextVar("evaluation_descriptor", default=None)


@contextmanager
def _evaluation_lease(
    snapshot: FrontierSnapshot | Mapping[str, Any], artifact_root: str | Path
) -> Iterator[None]:
    document = snapshot.to_document() if isinstance(snapshot, FrontierSnapshot) else snapshot
    root = Path(artifact_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    stem = ".nf-independent-frontier-" + str(document["snapshot_digest"]).removeprefix("sha256:")[:24]
    path = root / f"{stem}.lease"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            if os.name == "nt":
                import msvcrt

                if os.fstat(descriptor).st_size == 0:
                    os.write(descriptor, b"\0")
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise FrontierEvaluationError("Independent evaluator lease is already active") from exc
        token = _ACTIVE_EVALUATION_LEASE.set(path)
        descriptor_token = _ACTIVE_EVALUATION_DESCRIPTOR.set(descriptor)
        try:
            yield
        finally:
            _ACTIVE_EVALUATION_LEASE.reset(token)
            _ACTIVE_EVALUATION_DESCRIPTOR.reset(descriptor_token)
    finally:
        os.close(descriptor)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _qualification_roster(records: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(
        candidate_id
        for candidate_id, record in records.items()
        if record.get("arm") in {"direct", "baseline"}
        and not (record.get("arm") == "baseline" and record.get("method") == "no_op")
        and record.get("search_state") not in {"failed", "skipped"}
    )


def _isolated_model_payload(model: Any) -> dict[str, Any]:
    current_state = tensor_state_digest(model.tensors)
    checkpoint = getattr(model, "checkpoint_identity", None)
    if checkpoint is not None and getattr(model, "_checkpoint_state_digest", None) == current_state:
        path = Path(checkpoint.path).resolve()
        files = dict(checkpoint.file_digests)
        if all(not (path / name).is_symlink() for name in files):
            resolved = resolve_checkpoint(path, revision=checkpoint.revision)
            if resolved.file_digests == files:
                return {
                    "model_type": type(model).__name__, "model_name": str(model.model_name),
                    "checkpoint_identity": {
                        "path": str(path), "model_id": str(checkpoint.model_id),
                        "revision": checkpoint.revision, "file_digests": files,
                    },
                    "expected_state_digest": current_state,
                    "expected_inference_identity_digest": inference_identity_digest(model),
                }
    tokenizer = model.tokenizer
    merges = [" ".join(pair) for pair, _ in sorted(tokenizer.bpe_ranks.items(), key=lambda entry: entry[1])]
    return {
        "model_type": type(model).__name__, "model_name": str(model.model_name),
        "config": dict(model.config),
        "tokenizer": {
            "vocab": dict(tokenizer.vocab), "merges": merges,
            "byte_level": bool(tokenizer.byte_level), "added_tokens": dict(tokenizer.added_tokens),
        },
        "tensors": {name: value.detach().to(device="cpu").clone() for name, value in model.tensors.items()},
    }


def load_candidate_artifact(
    record: Mapping[str, Any],
    base_tensors: Mapping[str, torch.Tensor],
    artifact_root: str | Path,
) -> dict[str, torch.Tensor]:
    """Load one frozen replacement map after verifying its complete artifact contract."""
    raw = record.get("artifact")
    if not isinstance(raw, Mapping):
        raise FrontierEvaluationError(f"Candidate {record.get('candidate_id')!r} has no artifact")
    try:
        artifact = CandidateArtifact(
            relative_path=str(raw["relative_path"]),
            file_digest=str(raw["file_digest"]),
            state_digest=str(raw["state_digest"]),
            tensor_names=tuple(str(name) for name in raw["tensor_names"]),
            file_bytes=int(raw["file_bytes"]),
            format=str(raw.get("format", "nf_torch_tensor_map_v1")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FrontierEvaluationError(f"Candidate artifact is malformed: {exc}") from exc
    root = Path(artifact_root).resolve()
    path = (root / artifact.relative_path).resolve()
    if not path.is_relative_to(root):
        raise FrontierEvaluationError("Candidate artifact escapes the frozen artifact root")
    if not path.is_file():
        raise FrontierEvaluationError(f"Candidate artifact is missing: {path}")
    if not 0 < artifact.file_bytes <= MAX_CANDIDATE_ARTIFACT_BYTES:
        raise FrontierEvaluationError("Candidate artifact exceeds the verified decode byte limit")
    with path.open("rb") as stream:
        if os.fstat(stream.fileno()).st_size != artifact.file_bytes:
            raise FrontierEvaluationError("Candidate artifact byte count changed after the frontier freeze")
        verified_bytes = stream.read(artifact.file_bytes + 1)
    if len(verified_bytes) != artifact.file_bytes or "sha256:" + hashlib.sha256(verified_bytes).hexdigest() != artifact.file_digest:
        raise FrontierEvaluationError("Candidate artifact file digest changed after the frontier freeze")
    try:
        loaded = torch.load(io.BytesIO(verified_bytes), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise FrontierEvaluationError(f"Candidate artifact cannot be loaded safely: {exc}") from exc
    if not isinstance(loaded, dict) or tuple(loaded) != artifact.tensor_names:
        raise FrontierEvaluationError("Candidate artifact tensor names changed after the frontier freeze")
    unknown = sorted(set(loaded) - set(base_tensors))
    if unknown:
        raise FrontierEvaluationError(f"Candidate artifact contains unknown tensors: {unknown}")
    tensors: dict[str, torch.Tensor] = {}
    for name, value in loaded.items():
        if not isinstance(value, torch.Tensor):
            raise FrontierEvaluationError(f"Candidate artifact value {name!r} is not a tensor")
        if value.device.type != "cpu" or value.dtype != torch.float64:
            raise FrontierEvaluationError(f"Candidate artifact tensor {name!r} is not CPU float64")
        if tuple(value.shape) != tuple(base_tensors[name].shape):
            raise FrontierEvaluationError(f"Candidate artifact tensor {name!r} changed shape")
        if not bool(torch.isfinite(value).all().item()):
            raise FrontierEvaluationError(f"Candidate artifact tensor {name!r} is non-finite")
        tensors[name] = value
    state_digest = tensor_state_digest({**base_tensors, **tensors})
    if state_digest != artifact.state_digest or state_digest != record.get("state_digest"):
        raise FrontierEvaluationError("Candidate artifact state digest changed after the frontier freeze")
    return tensors


def evaluate_frozen_frontier(
    model: Any,
    snapshot: FrontierSnapshot | Mapping[str, Any],
    *,
    artifact_root: str | Path,
    target_cases: Sequence[independent_eval.TargetCase | tuple[str, str]],
    evaluator: Callable[..., dict[str, Any]] | None = None,
    eval_config: independent_eval.EvalConfig | None = None,
    execution_mode: str = "in_process",
    claim_exposure: bool = True,
) -> dict[str, Any]:
    """Evaluate under a process-owned exclusive lease, released on error or process exit."""
    lease = _evaluation_lease(snapshot, artifact_root) if claim_exposure else nullcontext()
    with lease:
        return _evaluate_frozen_frontier(
            model, snapshot, artifact_root=artifact_root, target_cases=target_cases,
            evaluator=evaluator, eval_config=eval_config, execution_mode=execution_mode,
            claim_exposure=claim_exposure,
        )


def _evaluate_frozen_frontier(
    model: Any,
    snapshot: FrontierSnapshot | Mapping[str, Any],
    *,
    artifact_root: str | Path,
    target_cases: Sequence[independent_eval.TargetCase | tuple[str, str]],
    evaluator: Callable[..., dict[str, Any]] | None = None,
    eval_config: independent_eval.EvalConfig | None = None,
    execution_mode: str = "in_process",
    claim_exposure: bool = True,
) -> dict[str, Any]:
    """Evaluate each materialized frontier candidate exactly from the frozen snapshot."""
    document = snapshot.to_document() if isinstance(snapshot, FrontierSnapshot) else dict(snapshot)
    try:
        validate_frontier_snapshot(document)
    except ValueError as exc:
        raise FrontierEvaluationError(f"Frozen frontier failed integrity validation: {exc}") from exc
    evaluator = evaluator or independent_eval.evaluate_candidate
    evaluator_id = (
        "builtin:nf_independent_eval.evaluate_candidate:v1"
        if evaluator is independent_eval.evaluate_candidate
        else f"{getattr(evaluator, '__module__', type(evaluator).__module__)}:{getattr(evaluator, '__qualname__', type(evaluator).__qualname__)}"
    )
    claim_paths: tuple[Path, Path] | None = None
    if claim_exposure:
        request_digest = _evaluation_request_digest(
            document,
            model,
            target_cases,
            eval_config,
            execution_mode=execution_mode,
            evaluator_id=evaluator_id,
        )
        claim_paths, cached = _begin_evaluation_claim(document, artifact_root, request_digest)
        if cached is not None:
            return cached
    records = {str(record["candidate_id"]): record for record in document["records"]}
    evaluated: list[dict[str, Any]] = []
    not_evaluated: list[dict[str, str]] = []
    eligible = {str(value) for value in document.get("eligible_ids", ())}
    qualification_roster = _qualification_roster(records)
    roster = set(qualification_roster)
    for candidate_id, record in records.items():
        if candidate_id in roster:
            continue
        if record.get("arm") == "control":
            reason = "control_generated_by_independent_evaluator_or_paired_search"
        elif record.get("method") == "no_op":
            reason = "unchanged_base_reference"
        elif record.get("search_state") in {"failed", "skipped"}:
            reason = f"search_state_{record['search_state']}"
        else:
            reason = "not_a_candidate_method"
        not_evaluated.append({"candidate_id": candidate_id, "reason": reason})
    for candidate_id in qualification_roster:
        record = records.get(candidate_id)
        if record is None:
            raise FrontierEvaluationError(
                f"Frozen frontier candidate {candidate_id!r} is absent from snapshot records"
            )
        try:
            tensors = load_candidate_artifact(record, model.tensors, artifact_root)
            frozen = independent_eval.freeze_candidate(
                model.tensors,
                tensors,
                candidate_id=candidate_id,
                target_cases=target_cases,
            )
            report = evaluator(model, frozen, config=eval_config)
            decision = promotion_gate.decide_promotion(report)
        except (FrontierEvaluationError, independent_eval.IndependentEvalError) as exc:
            not_evaluated.append(
                {"candidate_id": candidate_id, "reason": f"evaluation_refused: {exc}"}
            )
            continue
        evaluated.append(
            {
                "candidate_id": candidate_id,
                "arm": record.get("arm"),
                "method": record.get("method"),
                "profile_eligible_before_independent_eval": (
                    candidate_id in eligible if record.get("arm") == "direct" else None
                ),
                "evaluator_report": report,
                "promotion_decision": decision,
            }
        )
    for entry in evaluated:
        record = records[entry["candidate_id"]]
        artifact_reloaded = True  # load_candidate_artifact above verified file and state digests.
        if record.get("arm") == "direct":
            checks = {
                "profile_eligible": entry["profile_eligible_before_independent_eval"] is True,
                "replay_verified": record.get("replay_verified") is True,
                "reload_parity_verified": record.get("reload_parity_verified") is True,
                "independent_target_custody": False,
                "independent_promotion_passed": entry["promotion_decision"].get("promote_eligible") is True,
            }
            method_checks = {"kind": "direct_replay", "artifact_reload_verified": artifact_reloaded}
        else:
            proposal = record.get("proposal")
            dataset = proposal.get("dataset") if isinstance(proposal, Mapping) else None
            recipe = proposal.get("recipe") if isinstance(proposal, Mapping) else None
            recipe_bound = bool(
                isinstance(proposal, Mapping)
                and proposal.get("recipe_schema_version") == 1
                and isinstance(proposal.get("kind"), str)
                and isinstance(recipe, Mapping)
                and type(recipe.get("seed")) is int
                and isinstance(dataset, Mapping)
                and all(
                    isinstance(dataset.get(field), str) and dataset[field].startswith("sha256:")
                    for field in ("gym_digest", "dataset_digest", "train_digest", "validation_digest")
                )
            )
            method_checks = {
                "kind": "trained_candidate",
                "artifact_reload_verified": artifact_reloaded,
                "recipe_and_dataset_bound": recipe_bound,
                "retraining_reproducibility_verified": False,
            }
            checks = {
                "artifact_reload_verified": artifact_reloaded,
                "recipe_and_dataset_bound": recipe_bound,
                "retraining_reproducibility_verified": False,
                "independent_target_custody": False,
                "independent_promotion_passed": entry["promotion_decision"].get("promote_eligible") is True,
            }
        passed = all(checks.values())
        entry["method_checks"] = method_checks
        entry["deployment_decision"] = {
            "status": "passed" if passed else "rejected",
            "promote_eligible": passed,
            "checks": checks,
            "failed_checks": [name for name, value in checks.items() if not value],
        }
    promoted = tuple(
        entry["candidate_id"] for entry in evaluated
        if entry["deployment_decision"]["promote_eligible"] is True
    )
    rejected = tuple(
        entry["candidate_id"] for entry in evaluated
        if entry["deployment_decision"]["promote_eligible"] is False
    )
    refused = any(
        item["reason"].startswith("evaluation_refused") for item in not_evaluated
    )
    incomplete_roster = any(item["candidate_id"] in roster for item in not_evaluated)
    if refused or incomplete_roster:
        independent_terminal_state = "qualification_incomplete"
    elif promoted:
        independent_terminal_state = "independent_promotion_feasible"
    elif rejected:
        independent_terminal_state = "independent_rejected"
    else:
        independent_terminal_state = "frontier_empty"
    payload = {
        "schema_version": 1,
        "kind": "nf_frontier_evaluation_v1",
        "execution_mode": execution_mode,
        "snapshot_digest": document["snapshot_digest"],
        "profile_id": document["profile_id"],
        "profile_digest": document["profile_digest"],
        "promotion_policy": {
            "id": "strict-general-promotion-v1",
            "scope": "general behavior preservation; separate from specialized profile eligibility",
            "target_custody": "not_proven_by_python_frontier_path",
        },
        "target_case_count": len(target_cases),
        "search_frontier_ids": list(document["search_frontier_ids"]),
        "qualification_roster_ids": list(qualification_roster),
        "evaluated": evaluated,
        "not_evaluated": not_evaluated,
        "promoted_candidate_ids": list(promoted),
        "rejected_candidate_ids": list(rejected),
        "complete": not refused and not incomplete_roster,
        "independent_terminal_state": independent_terminal_state,
    }
    payload["report_digest"] = str(content_digest(canonicalize(payload)))
    return _finish_evaluation_claim(claim_paths, payload) if claim_paths is not None else payload


def evaluate_frozen_frontier_isolated(
    model: Any,
    snapshot: FrontierSnapshot | Mapping[str, Any],
    *,
    artifact_root: str | Path,
    target_cases: Sequence[independent_eval.TargetCase | tuple[str, str]],
    eval_config: independent_eval.EvalConfig | None = None,
    evaluator_python: str | Path | None = None,
    timeout_seconds: float = 3600.0,
) -> dict[str, Any]:
    """Hold the exclusive lease throughout the isolated worker's entire lifetime."""
    if os.name == "nt":
        raise FrontierEvaluationError("Isolated evaluator lease inheritance requires a POSIX runtime; use WSL")
    with _evaluation_lease(snapshot, artifact_root):
        return _evaluate_frozen_frontier_isolated(
            model, snapshot, artifact_root=artifact_root, target_cases=target_cases,
            eval_config=eval_config, evaluator_python=evaluator_python,
            timeout_seconds=timeout_seconds,
        )


def _evaluate_frozen_frontier_isolated(
    model: Any,
    snapshot: FrontierSnapshot | Mapping[str, Any],
    *,
    artifact_root: str | Path,
    target_cases: Sequence[independent_eval.TargetCase | tuple[str, str]],
    eval_config: independent_eval.EvalConfig | None = None,
    evaluator_python: str | Path | None = None,
    timeout_seconds: float = 3600.0,
) -> dict[str, Any]:
    """Evaluate in a fresh CPU process that receives no proposer or search-loop object."""
    model_type = type(model).__name__
    if model_type not in {"PythiaModel", "Qwen2CausalLM"}:
        raise FrontierEvaluationError(
            f"Isolated frontier evaluation does not support model type {model_type!r}"
        )
    document = snapshot.to_document() if isinstance(snapshot, FrontierSnapshot) else dict(snapshot)
    try:
        validate_frontier_snapshot(document)
    except ValueError as exc:
        raise FrontierEvaluationError(f"Frozen frontier failed integrity validation: {exc}") from exc
    target_case_rows = [
        [case.prompt, case.target] if isinstance(case, independent_eval.TargetCase) else list(case)
        for case in target_cases
    ]
    request_digest = _evaluation_request_digest(
        document,
        model,
        target_cases,
        eval_config,
        execution_mode="isolated_subprocess",
        evaluator_id="builtin:nf_independent_eval.evaluate_candidate:v1",
        execution_options={
            "evaluator_python": str(evaluator_python) if evaluator_python is not None else None,
            "timeout_seconds": timeout_seconds,
        },
    )
    claim_paths, cached = _begin_evaluation_claim(document, artifact_root, request_digest)
    if cached is not None:
        return cached
    payload = {
        "schema_version": 1,
        "kind": "nf_independent_frontier_job_v1",
        "snapshot": document,
        "artifact_root": str(Path(artifact_root).resolve()),
        "target_cases": target_case_rows,
        "eval_config": asdict(eval_config) if eval_config is not None else None,
        "model": _isolated_model_payload(model),
    }
    source_root = Path(__file__).resolve().parents[1]
    python = str(evaluator_python or sys.executable)
    with tempfile.TemporaryDirectory(prefix="nf-independent-frontier-") as directory:
        root = Path(directory)
        input_path = root / "input.pt"
        output_path = root / "report.json"
        torch.save(payload, input_path)
        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        environment = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                item for item in (str(source_root), existing_pythonpath) if item
            ),
            "PYTHONNOUSERSITE": "1",
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "2"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "2"),
        }
        lease_descriptor = _ACTIVE_EVALUATION_DESCRIPTOR.get()
        if lease_descriptor is None:
            raise FrontierEvaluationError("Isolated evaluation requires an active evaluator lease")
        completed = subprocess.run(
            [
                python,
                "-m",
                "neural_foundry.independent_frontier_worker",
                str(input_path),
                str(output_path),
            ],
            cwd=source_root.parent,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
            pass_fds=(lease_descriptor,),
        )
        if completed.returncode != 0 or not output_path.is_file():
            detail = (completed.stderr or completed.stdout).strip()[-2000:]
            raise FrontierEvaluationError(
                f"Independent evaluator exited {completed.returncode} without a report: {detail}"
            )
        report = cast(dict[str, Any], json.loads(output_path.read_text(encoding="utf-8")))
    if report.get("kind") != "nf_frontier_evaluation_v1":
        raise FrontierEvaluationError("Independent worker returned the wrong report kind")
    if report.get("execution_mode") != "isolated_subprocess":
        raise FrontierEvaluationError("Independent worker did not report isolated execution")
    if report.get("snapshot_digest") != document["snapshot_digest"]:
        raise FrontierEvaluationError("Independent worker evaluated a different frontier snapshot")
    return _finish_evaluation_claim(claim_paths, report)


def _evaluation_request_digest(
    document: Mapping[str, Any],
    model: Any,
    target_cases: Sequence[independent_eval.TargetCase | tuple[str, str]],
    eval_config: independent_eval.EvalConfig | None,
    *,
    execution_mode: str,
    evaluator_id: str,
    execution_options: Mapping[str, Any] | None = None,
) -> str:
    target_rows = [
        [case.prompt, case.target] if isinstance(case, independent_eval.TargetCase) else list(case)
        for case in target_cases
    ]
    tokenizer = getattr(model, "tokenizer", None)
    tokenizer_state = None if tokenizer is None else {
        "type": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "vocab": getattr(tokenizer, "vocab", None),
        "merges": [list(pair) for pair, _ in sorted(getattr(tokenizer, "bpe_ranks", {}).items(), key=lambda row: row[1])],
        "added_tokens": getattr(tokenizer, "added_tokens", None),
        "byte_level": getattr(tokenizer, "byte_level", None),
    }
    inference_identity = hashlib.sha256(json.dumps({
        "config": getattr(model, "config", None), "tokenizer": tokenizer_state,
    }, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()).hexdigest()
    return str(content_digest(canonicalize({
        "kind": "nf_independent_frontier_request_v1",
        "snapshot_digest": document["snapshot_digest"],
        "base_model_digest": independent_eval.parameter_digest(model.tensors),
        "tensor_runtime": {name: {"dtype": str(tensor.dtype), "device": str(tensor.device)} for name, tensor in model.tensors.items()},
        "torch_math_mode": {
            "version": str(torch.__version__), "float32_precision": torch.get_float32_matmul_precision(),
            "cuda_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        },
        "config_tokenizer_digest": f"sha256:{inference_identity}",
        "target_cases": target_rows,
        "eval_config": asdict(eval_config) if eval_config is not None else None,
        "execution_mode": execution_mode,
        "evaluator_id": evaluator_id,
        "execution_options": dict(execution_options or {}),
    })))


def _persist_json(path: Path, document: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=".nf-frontier-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _begin_evaluation_claim(
    document: Mapping[str, Any], artifact_root: str | Path, request_digest: str
) -> tuple[tuple[Path, Path], dict[str, Any] | None]:
    """Bind exposure to one frozen request; identical retries resume/cache, changed requests refuse."""
    root = Path(artifact_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    stem = ".nf-independent-frontier-" + str(document["snapshot_digest"]).removeprefix("sha256:")[:24]
    claim_path = root / f"{stem}.claim.json"
    report_path = root / f"{stem}.report.json"
    if _ACTIVE_EVALUATION_LEASE.get() != root / f"{stem}.lease":
        raise FrontierEvaluationError("Evaluation claim requires an exclusive evaluator lease")
    legacy_marker = root / f"{stem}.claimed"
    if legacy_marker.exists() and not claim_path.exists():
        raise FrontierEvaluationError("Legacy exposure claim cannot be safely resumed")
    claim = {
        "kind": "nf_independent_frontier_claim_v2",
        "snapshot_digest": document["snapshot_digest"],
        "request_digest": request_digest,
        "state": "in_progress",
    }
    if claim_path.exists():
        try:
            previous = json.loads(claim_path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FrontierEvaluationError(f"Exposure claim record is unreadable: {exc}") from exc
        if (
            previous.get("snapshot_digest") != document["snapshot_digest"]
            or previous.get("request_digest") != request_digest
        ):
            raise FrontierEvaluationError("This snapshot is already bound to a different evaluation request")
        if previous.get("state") == "complete":
            try:
                cached = json.loads(report_path.read_bytes())
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise FrontierEvaluationError(f"Completed evaluation report is missing or unreadable: {exc}") from exc
            report_digest = cached.pop("report_digest", None)
            if (
                report_digest is None
                or str(content_digest(canonicalize(cached))) != report_digest
                or report_digest != previous.get("report_digest")
            ):
                raise FrontierEvaluationError("Cached evaluation report failed digest verification")
            cached["report_digest"] = report_digest
            return (claim_path, report_path), cached
        if previous.get("state") not in {"in_progress", "retryable"}:
            raise FrontierEvaluationError("Exposure claim has an unsupported state")
        previous["state"] = "in_progress"
        retry_count = previous.get("retry_count", 0)
        if type(retry_count) is not int or retry_count < 0:
            raise FrontierEvaluationError("Exposure claim retry count is malformed")
        if retry_count >= MAX_EXPOSURE_RETRIES:
            raise FrontierEvaluationError("Independent evaluation retry limit is exhausted")
        previous["retry_count"] = retry_count + 1
        previous["last_retry_ns"] = __import__("time").time_ns()
        previous["owner_pid"] = os.getpid()
        _persist_json(claim_path, previous)
    else:
        claim["retry_count"] = 0
        claim["created_ns"] = __import__("time").time_ns()
        claim["owner_pid"] = os.getpid()
        _persist_json(claim_path, claim)
    return (claim_path, report_path), None


def _finish_evaluation_claim(
    claim_paths: tuple[Path, Path] | None, report: dict[str, Any]
) -> dict[str, Any]:
    if claim_paths is None:
        return report
    claim_path, report_path = claim_paths
    if _ACTIVE_EVALUATION_LEASE.get() != claim_path.with_suffix("").with_suffix(".lease"):
        raise FrontierEvaluationError("Finishing an evaluation requires its exclusive lease")
    report_digest = report.get("report_digest")
    if not isinstance(report_digest, str) or str(
        content_digest(canonicalize({key: value for key, value in report.items() if key != "report_digest"}))
    ) != report_digest:
        raise FrontierEvaluationError("Independent evaluation report has an invalid digest")
    _persist_json(report_path, report)
    claim = json.loads(claim_path.read_bytes())
    claim["state"] = "complete" if report.get("complete") is True else "retryable"
    claim["report_digest"] = report_digest
    if claim["state"] == "complete":
        claim["completed_ns"] = __import__("time").time_ns()
    _persist_json(claim_path, claim)
    return report
