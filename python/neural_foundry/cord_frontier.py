"""One-pass, explicitly released matched CORD provider comparison controller.

``prepare`` writes only a ready configuration. After the parent's core-stable
release, run ``freeze --core-stable`` then ``run-logged --core-stable`` exactly
once. Execution uses the single approved artifact root and durable shared ledger;
failed claims/reservations are never removed and no retry path is provided.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from .cord_receipt_plugin import (
    EVALUATION_POLICY,
    cord_task,
    evaluation_policy,
    registry,
    summarize,
)
from .gym_job import GymJobError
from .gym_service import _document_digest
from .openai_transport import OPENAI_CHAT_COMPLETIONS_URL
from .proof_pipeline import (
    audit,
    candidate_task,
    read_document,
    score,
    validate_spec,
    write_document,
)
from .proof_policy import load_policy
from .provenance import sha256_file
from .public_budget import PublicBudget, initialize_public_budget

PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT / "artifacts" / "cord-bipia-first-20261111" / "guided-one" / "spec-base4b.json"
ESTIMATES = PROJECT / "artifacts" / "cord-scorer-v2-20261111" / "resource-estimates.json"
POLICY = PROJECT / "protocols" / "cord-frontier-policy-v1.json"
DEFAULT_ROOT = PROJECT / "artifacts" / "cord-frontier-approved-20261111"
MODEL = "gpt-6-luna"


def _reference() -> dict[str, Any]:
    return {"path": str(POLICY), "file_digest": sha256_file(POLICY), "calls_authorized": True}


def _approved_policy() -> dict[str, Any]:
    policy = load_policy(_reference())
    api = policy["api"]
    if (policy["version"] != 2 or policy["max_new_tokens"] != 2560
        or policy["context_tokens"] != 5120 or api["maximum_api_attempts"] != 200
        or api["maximum_cost_microusd"] != 500000 or api["input_overhead_token_bound"] != 1024
        or api["model_rates"] != {MODEL: {"input_microusd_per_million_tokens": 100000,
                                        "output_microusd_per_million_tokens": 500000}}):
        raise GymJobError("frontier policy differs from exact explicit user approval")
    return policy


def _source() -> dict[str, Any]:
    estimates = json.loads(ESTIMATES.read_bytes())
    approved_digest = "sha256:41c25b60866f2d8da08a94e04d55510c5b7bd29f3830a7c5ddaa1d63234310fa"
    if (sha256_file(SOURCE) != approved_digest or estimates["sources"][
            "artifacts/cord-bipia-first-20261111/guided-one/spec-base4b.json"] != approved_digest):
        raise GymJobError("approved original CORD source file changed")
    source = read_document(SOURCE)
    if (len(source["candidates"]) != 1
        or [len(panel["rows"]) for panel in source["public_sets"]] != [100, 100]
        or source["training_rows"]):
        raise GymJobError("approved exact ordered 200 validation cases differ")
    return source


def _requests(source: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    task = candidate_task(cord_task({**source["gym"], "scorer_version": "2"}),
                          source["candidates"][0])
    requests = []
    wire_bytes = 0
    for panel in source["public_sets"]:
        for row in panel["rows"]:
            interaction = task.project(row["input"], None)
            payload = {"model": MODEL, "messages": interaction["messages"],
                       "temperature": policy["temperature"], "reasoning_effort": policy["reasoning_effort"],
                       "max_completion_tokens": policy["max_new_tokens"],
                       "response_format": {"type": "json_schema", "json_schema": {
                           "name": "nf_output", "strict": True,
                           "schema": task.job.to_document()["task"]["output_schema"]}}}
            size = len(json.dumps(payload, allow_nan=False).encode())
            wire_bytes += size
            api = policy["api"]
            rates = api["model_rates"][MODEL]
            input_bound = size + api["input_overhead_token_bound"]
            reservation = ((input_bound * rates["input_microusd_per_million_tokens"] + 999999) // 1000000
                           + (policy["max_new_tokens"] * rates["output_microusd_per_million_tokens"]
                              + 999999) // 1000000)
            requests.append({"scenario_id": row["scenario_id"], "panel": panel["name"],
                             "request_digest": _document_digest(payload),
                             "reservation_microusd": reservation})
    total = sum(row["reservation_microusd"] for row in requests)
    if total > policy["api"]["maximum_cost_microusd"]:
        raise GymJobError("complete roster exceeds approved reservation ceiling")
    return {"requests": requests, "wire_bytes": wire_bytes,
            "total_reservation_microusd": total}


def prepare(root: Path) -> dict[str, Any]:
    """Offline approval/configuration only; do not bind mutable registry sources yet."""
    source = _source()
    policy_reference = _reference()
    policy = _approved_policy()
    request_plan = _requests(source, policy)
    estimates = json.loads(ESTIMATES.read_bytes())["frontier"]
    if request_plan["wire_bytes"] != estimates["input_wire_bytes"]["sum"]:
        raise GymJobError("serialized provider inputs differ from reviewed matched request estimate")
    return write_document(root / "ready-config.json", {
        "kind": "nf_cord_frontier_ready_v1", "run_id": "cord-frontier-approved-20261111",
        "root": str(root), "source_path": str(SOURCE), "source_file_digest": sha256_file(SOURCE),
        "resource_estimates_file_digest": sha256_file(ESTIMATES),
        "policy_reference": policy_reference, "request_plan": request_plan,
        "approval": {"model": MODEL, "maximum_api_attempts": 200,
                     "maximum_cost_microusd": 500000, "automatic_retries": 0,
                     "provider_transfer_user_approved": True,
                     "provider_transfer_scope": "exact frozen 100 clean and 100 internal BIPIA validation inputs",
                     "internal_bipia_evaluation_approved": True,
                     "credential_source": "runtime OPENAI_API_KEY only; no .env reads",
                     "native_test_and_bipia_test_attacks_sealed": True},
        "cost_uncertainty": {
            "reservation_is_not_bill": True, "provider_encoding_verified": False,
            "input_bound": "serialized UTF-8 bytes + 1024 per request; not provider-guaranteed",
            "reviewed_input_proxy": "Qwen tokenizer, not GPT-6 Luna encoding",
            "reviewed_input_proxy_tokens": 113923, "reviewed_output_proxy_tokens": 80647,
            "reviewed_expected_usd_proxy": 0.0517158,
            "reviewed_unrounded_conservative_usd": 0.316818,
            "official_input_usd_per_million": 0.10, "official_output_usd_per_million": 0.50,
            "cache_discount_assumed": False},
        "ready_for_final_freeze": True, "core_stable_released": False,
        "provider_requests": 0, "holdout_accessed": False, "release_authorized": False})


def _ready(root: Path) -> dict[str, Any]:
    config = read_document(root / "ready-config.json")
    if (config["root"] != str(root) or config["source_file_digest"] != sha256_file(SOURCE)
        or config["resource_estimates_file_digest"] != sha256_file(ESTIMATES)
        or config["policy_reference"] != _reference()
        or config["request_plan"] != _requests(_source(), _approved_policy())):
        raise GymJobError("ready configuration or exact matched requests changed")
    return config


def freeze(root: Path, *, core_stable: bool = False) -> dict[str, Any]:
    if not core_stable:
        raise GymJobError("final freeze requires parent's explicit core-stable release")
    config = _ready(root)
    if root.resolve() != DEFAULT_ROOT.resolve():
        raise GymJobError("this approval has exactly one shared run root and budget ledger")
    source = _source()
    custom, refs = registry()
    spec = {key: deepcopy(value) for key, value in source.items() if key != "document_digest"}
    spec.update({
        "run_id": config["run_id"], "gym": {**source["gym"], "scorer_version": "2",
                                           "methods": ["provider"]},
        "plugins": {key: refs[key] for key in source["plugins"]},
        "candidates": [{"role": "frontier", "method": "provider",
                        "system_prompt": source["candidates"][0]["system_prompt"],
                        "evidence": refs["api_evidence"], "execution": {
                            "kind": "provider", "plugin": refs["provider"],
                            "configuration": {"endpoint": OPENAI_CHAT_COMPLETIONS_URL, "model": MODEL}}}],
        "policy_reference": config["policy_reference"],
        "policy": _approved_policy(),
        "public_budget": {"version": 1, "ledger_id": config["run_id"],
                          "path": str(root / "budget.sqlite3")},
        "frontier_approval": config["approval"], "cost_uncertainty": config["cost_uncertainty"],
        "configuration_digest": config["document_digest"],
        "matched_source_file_digest": config["source_file_digest"],
        "headline_metric": deepcopy(evaluation_policy()["headline"]),
        "scoring_policy_reference": {"path": str(EVALUATION_POLICY),
                                     "file_digest": sha256_file(EVALUATION_POLICY)},
        "controller_sources": {str(path): sha256_file(path) for path in
                               (Path(__file__), PROJECT / "scripts" / "run_cord_frontier.py")},
        "core_stable_released": True,
    })
    validate_spec(spec, custom)
    task = custom.resolve("gym", spec["plugins"]["gym"])(spec["gym"])
    if "provider" not in {method.method_id for method in task.job.candidate_methods}:
        raise GymJobError("CORD task must explicitly register the provider candidate method before freeze")
    audit(spec, custom)
    return write_document(root / "frozen-spec.json", spec)


def execute(root: Path, *, core_stable: bool = False) -> dict[str, Any]:
    if not core_stable:
        raise GymJobError("execution requires parent's explicit core-stable release")
    config = _ready(root)
    if root.resolve() != DEFAULT_ROOT.resolve():
        raise GymJobError("this approval has exactly one shared run root and budget ledger")
    spec = read_document(root / "frozen-spec.json")
    spec.pop("document_digest")
    custom, _ = registry()
    validate_spec(spec, custom)
    if (spec.get("core_stable_released") is not True
        or spec.get("configuration_digest") != config["document_digest"]
        or spec.get("controller_sources") != {str(path): sha256_file(path) for path in
                                             (Path(__file__), PROJECT / "scripts" / "run_cord_frontier.py")}):
        raise GymJobError("frozen run lacks exact released ready-configuration binding")
    # This exclusive claim is never removed, even on a pre-dispatch failure.
    write_document(root / "execution.started.json", {
        "run_spec_digest": _document_digest(spec), "automatic_retries": 0,
        "maximum_api_attempts": 200, "core_stable_released": True})
    try:
        initialize_public_budget(spec)
        result = score(spec, role="frontier", root=root / "results", registry=custom)
        budget = PublicBudget(spec).snapshot()
        if (budget["attempts"] != 200 or sum(panel["cases"] for panel in result["scores"].values()) != 200):
            raise GymJobError("frontier did not complete the exact 200-request roster")
        return write_document(root / "frontier-card.json", {
            "kind": "nf_cord_frontier_card_v1", "complete": True,
            "run_spec_digest": _document_digest(spec), "summary": summarize(spec, result),
            "score_path": str(root / "results" / "public-score-frontier.json"),
            "score_document_digest": result["document_digest"], "budget": budget,
            "budget_is_reservations_not_invoice": True, "approval": config["approval"],
            "cost_uncertainty": config["cost_uncertainty"],
            "holdout_accessed": False, "release_authorized": False})
    except Exception as error:
        write_document(root / "execution-failure.json", {
            "complete": False, "failure_type": type(error).__name__,
            "run_spec_digest": _document_digest(spec), "automatic_retries": 0,
            "reservations_retained": True, "holdout_accessed": False})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "freeze", "run-logged", "execute"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--core-stable", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage == "prepare":
        result = prepare(root)
    elif args.stage == "freeze":
        result = freeze(root, core_stable=args.core_stable)
    elif args.stage == "execute":
        result = execute(root, core_stable=args.core_stable)
    else:
        from .execution_runner import run_logged_worker

        if not args.core_stable:
            parser.error("logged execution requires parent's explicit core-stable release")
        _ready(root)
        read_document(root / "frozen-spec.json")
        write_document(root / "launch.started.json", {"automatic_retries": 0, "root": str(root)})
        command = [sys.executable, "-m", "neural_foundry.cord_frontier", "execute",
                   "--root", str(root), "--core-stable"]
        environment = {**os.environ, "PYTHONPATH": str(PROJECT / "python"),
                       "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
                       "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        worker = run_logged_worker(command, cwd=PROJECT, log_dir=root / "logs",
                                   timeout_seconds=14400, env=environment)
        print(json.dumps(worker.receipt, sort_keys=True))
        raise SystemExit(worker.returncode if not worker.timed_out else 124)
    print(json.dumps({key: value for key, value in result.items()
                      if key in {"kind", "document_digest", "complete", "core_stable_released"}}))


if __name__ == "__main__":
    main()
