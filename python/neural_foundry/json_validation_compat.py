"""Approved train-artifact export and validation-only comparison; never holdout."""

from __future__ import annotations

import argparse
import gc
import json
import runpy
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from neural_foundry.gym_artifacts import (
    CandidateRef,
    artifact_files,
    read_candidate_manifest,
    write_candidate_manifest,
)
from neural_foundry.gym_job import GymJobError
from neural_foundry.gym_service import ExecutionResponse, _document_digest
from neural_foundry.json_injection_training_data import generate_public_validation_v2_rows
from neural_foundry.json_proof_loop import (
    _require_model_evidence,
    _require_response_evidence,
    public_required_rows,
    score_public_rows,
)
from neural_foundry.provenance import sha256_file
from neural_foundry.reference_json_extraction import generate_scenarios, reference_task
from neural_foundry.vllm_candidate_files import verify_vllm_candidate_files
from neural_foundry.vllm_container_adapter import VllmCandidateServer


def read_document(path: Path, digest_field: str) -> dict[str, Any]:
    document = json.loads(path.read_bytes())
    if document.get(digest_field) != _document_digest({key: value for key, value in document.items() if key != digest_field}):
        raise GymJobError(f"document digest differs: {path.name}")
    return document


def write_document(path: Path, document: dict[str, Any]) -> None:
    document["document_digest"] = _document_digest(document)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)


def trained_reference(root: Path, method: str) -> tuple[CandidateRef, dict[str, Any]]:
    result_path = root / method / "training-result.json"
    if method == "lora" and not result_path.is_file():
        result_path = root / "lora-final" / "training-result.json"
    result = read_document(result_path, "result_digest")
    record = result["candidate_record"]
    for relative, expected in record["files"].items():
        if sha256_file(Path(record["artifact_root"]) / relative) != expected:
            raise GymJobError("trained candidate file differs")
    directory = Path(record["artifact_root"]) / "artifact"
    manifest = read_candidate_manifest(directory, record["training_request_digest"])
    candidate = record["candidate"]
    if manifest["model_digest"] != candidate["candidate_model_digest"]:
        raise GymJobError("trained candidate state differs")
    reference = CandidateRef(result["base"]["model_id"], result["training_request"]["artifact_kind"],
        candidate["base_model_digest"], candidate["candidate_model_digest"], candidate["artifact_digest"],
        str(directory), candidate["candidate_id"], method, result["base"]["family"])
    return reference, result


def merge_lora(root: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from neural_foundry.model_digest import model_state_digest

    source, result = trained_reference(root, "lora")
    base = CandidateRef(**result["base"])
    verify_vllm_candidate_files(base, revision=result["revision"])
    output = root / "lora-merged"
    output.mkdir(exist_ok=False)
    original = AutoModelForCausalLM.from_pretrained(base.artifact_path, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0").eval()
    adapter = PeftModel.from_pretrained(original, source.artifact_path, local_files_only=True).eval()
    if model_state_digest(adapter) != source.model_digest:
        raise GymJobError("LoRA reload differs from frozen training result")
    merged = adapter.merge_and_unload(safe_merge=True).eval()
    merged_digest = model_state_digest(merged)
    merged.save_pretrained(output, safe_serialization=True)
    AutoTokenizer.from_pretrained(source.artifact_path, local_files_only=True).save_pretrained(output)
    del merged, adapter, original
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = AutoModelForCausalLM.from_pretrained(output, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0").eval()
    if model_state_digest(reloaded) != merged_digest:
        raise GymJobError("merged full-model reload changed its weights digest")
    del reloaded
    gc.collect()
    torch.cuda.empty_cache()
    request = result["training_request"]["training_request_digest"]
    receipt = {"export_kind": "peft_merged_full_v1", "training_request_digest": request,
        "model_digest": merged_digest, "reload_state_digest": merged_digest,
        "source_adapter_digest": source.artifact_digest, "source_candidate_model_digest": source.model_digest,
        "source_result_digest": result["result_digest"], "holdout_accessed": False, "release_authorized": False}
    (output / "training_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    artifact_digest = _document_digest({"training_request": request, "files": artifact_files(output)})
    reference = CandidateRef(base.model_id, "full", base.base_digest, merged_digest, artifact_digest,
        str(output), source.candidate_id + ":merged", "lora", base.family)
    candidate = {"candidate_id": reference.candidate_id, "method_id": "lora", "base_model_digest": base.base_digest,
        "candidate_model_digest": merged_digest, "artifact_digest": artifact_digest}
    write_candidate_manifest(output, model_id=base.model_id, kind="full", base_digest=base.base_digest,
        model_digest=merged_digest, request_digest=request, metadata={"candidate": candidate})
    verify_vllm_candidate_files(reference, revision=result["revision"])
    write_document(root / "lora-merged-reference.json", {"reference": asdict(reference), "revision": result["revision"],
        "source_result_digest": result["result_digest"], "reloaded": True, "holdout_accessed": False})
    print("LoRA merged export and bf16 GPU reload verified", flush=True)


def reference_for(root: Path, role: str, source_root: Path | None = None) -> tuple[CandidateRef, str]:
    if role in {"old-sft", "old-lora"}:
        if source_root is None:
            raise GymJobError("historical candidates require --source-root")
        if role == "old-sft":
            reference, result = trained_reference(source_root, "sft")
            return reference, result["revision"]
        document = read_document(source_root / "lora-merged-reference.json", "document_digest")
        reference = CandidateRef(**document["reference"])
        verify_vllm_candidate_files(reference, revision=document["revision"])
        return reference, document["revision"]
    if role == "lora":
        document = read_document(root / "lora-merged-reference.json", "document_digest")
        return CandidateRef(**document["reference"]), document["revision"]
    reference, result = trained_reference(root, "sft")
    return (CandidateRef(**result["base"]) if role == "base" else reference), result["revision"]


def record_identity(root: Path, role: str, source_root: Path | None = None) -> None:
    reference, revision = reference_for(root, role, source_root)
    server = VllmCandidateServer(reference, revision=revision, max_new_tokens=128, timeout_seconds=480)
    with server:
        record = server.record_identity_probes()
    write_document(root / f"identity-{role}.json", {"reference": asdict(reference), "revision": revision,
        "probe_record": record, "holdout_accessed": False})
    print(f"{role}: fixed vLLM weight identity recorded", flush=True)


def freeze_public_validation_v2(root: Path) -> dict[str, Any]:
    preparation = read_document(root / "preparation" / "training-preparation.json", "preparation_digest")
    if preparation.get("kind") != "nf_json_training_preparation_v2":
        raise GymJobError("the hard public panel requires the frozen robustness-v2 training preparation")
    rows = generate_public_validation_v2_rows(seed=20261112, examples_per_template=32)
    old_train = generate_scenarios(split="train", seed=17, parameters={"rows_per_template": 32})
    old_validation = generate_scenarios(split="validation", seed=17, parameters={"rows_per_template": 32})
    required = public_required_rows(seed=17)
    lifecycle_script = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts" / "run_gym_lifecycle_real.py"))
    expected_preparation = lifecycle_script["prepare_json_robustness_v2_training_request"]()
    if preparation["preparation_digest"] != expected_preparation["preparation_digest"]:
        raise GymJobError("training preparation differs from the frozen robustness-v2 data contract")
    comparison_sets = {
        "augmented_train": preparation["training_rows"],
        "legacy_train": old_train,
        "legacy_validation": old_validation,
        "required_set": required,
    }
    panel_inputs = {row["input"]["text"] for row in rows}
    panel_ids = {row["scenario_id"] for row in rows}
    if len(panel_inputs) != len(rows) or len(panel_ids) != len(rows):
        raise GymJobError("hard public panel contains duplicate inputs or scenario IDs")
    for name, reference_rows in comparison_sets.items():
        reference_inputs = {row["input"]["text"] for row in reference_rows}
        reference_ids = {row["scenario_id"] for row in reference_rows}
        if panel_inputs & reference_inputs or panel_ids & reference_ids:
            raise GymJobError(f"hard public panel overlaps {name}")
    document = {
        "kind": "nf_json_public_validation_panel_v2",
        "seed": 20261112,
        "examples_per_template": 32,
        "rows": rows,
        "rows_digest": _document_digest({"rows": rows}),
        "training_preparation_digest": preparation["preparation_digest"],
        "disjointness_audit": {
            "compared_sets": list(comparison_sets),
            "panel_cases": len(rows),
            "unique_inputs": len(panel_inputs),
            "unique_scenario_ids": len(panel_ids),
            "by_category": {category: sum(row["panel_category"] == category for row in rows)
                             for category in ("clean_compositional", "injection")},
            "by_template": {template: sum(row["family_id"] == template for row in rows)
                            for template in sorted({row["family_id"] for row in rows})},
            "passed": True,
        },
        "holdout_accessed": False,
        "release_authorized": False,
    }
    path = root / "public-validation-v2" / "panel.json"
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    write_document(path, document)
    return read_document(path, "document_digest")


def score(root: Path, role: str, source_root: Path | None = None) -> None:
    from neural_foundry.json_frontier_model import JsonFrontierModel, frontier_proof_configuration

    preparation = read_document(root / "preparation" / "training-preparation.json", "preparation_digest")
    if preparation["frontier_configuration"] != frontier_proof_configuration():
        raise GymJobError("frontier configuration differs from frozen preparation")
    task = reference_task(rows_per_template=32)
    if preparation.get("kind") == "nf_json_training_preparation_v2":
        panel = read_document(root / "public-validation-v2" / "panel.json", "document_digest")
        if (panel.get("kind") != "nf_json_public_validation_panel_v2"
            or panel.get("training_preparation_digest") != preparation["preparation_digest"]
            or panel.get("disjointness_audit", {}).get("passed") is not True
            or panel.get("rows_digest") != _document_digest({"rows": panel.get("rows")})):
            raise GymJobError("frozen hard public panel differs from the approved train-only preparation")
        rows = panel["rows"]
    else:
        panel = None
        rows = generate_scenarios(split="validation", seed=17, parameters={"rows_per_template": 32})
    required = public_required_rows(seed=17)
    if role == "frontier":
        for local in ("base", "sft", "lora"):
            card = read_document(root / f"validation-{local}.json", "document_digest")
            if card.get("complete") is not True or card.get("preparation_digest") != preparation["preparation_digest"]:
                raise GymJobError("frontier must wait for verified local validation")
        claim = root / "frontier-one-shot.claim"
        with claim.open("x", encoding="utf-8") as stream:
            stream.write(preparation["preparation_digest"])
        model_context = JsonFrontierModel.from_env()
    else:
        frozen = read_document(root / f"identity-{role}.json", "document_digest")
        reference, revision = reference_for(root, role, source_root)
        if frozen["reference"] != asdict(reference):
            raise GymJobError("frozen serving identity changed")
        model_context = VllmCandidateServer(reference, revision=revision, max_new_tokens=128,
            timeout_seconds=480, identity_record=frozen["probe_record"])
    try:
        with model_context as model:
            _require_model_evidence(model, role)
            task_score = score_public_rows(task, model, rows)
            required_score = None if role in {"old-sft", "old-lora"} else score_public_rows(task, model, required)
            scored_rows = rows if required_score is None else [*rows, *required]
            scored_responses = task_score["responses"] if required_score is None else [*task_score["responses"], *required_score["responses"]]
            for row, response in zip(scored_rows, scored_responses, strict=True):
                _require_response_evidence(model, task.project(row["input"], None),
                    ExecutionResponse(response["response"], response["execution_evidence"] or {}), role)
            if isinstance(model_context, VllmCandidateServer):
                model_context.verify_end_identity()
            identity = dict(model.loaded_identity())
    except Exception as error:
        if isinstance(model_context, VllmCandidateServer):
            write_document(root / f"validation-{role}-rejected-{uuid4().hex}.json", {
                "role": role, "complete": False, "failure": str(error),
                "identity_probe_evidence": model_context._identity_evidence,
                "holdout_accessed": False, "release_authorized": False})
        raise
    write_document(root / f"validation-{role}.json", {"role": role, "split": "validation", "complete": True,
        "preparation_digest": preparation["preparation_digest"], "task": task_score, "required_set": required_score,
        "public_panel": (None if panel is None else {
            "kind": panel["kind"], "document_digest": panel["document_digest"], "rows_digest": panel["rows_digest"],
            "disjointness_audit": panel["disjointness_audit"],
        }),
        "model_identity": identity, "holdout_accessed": False, "release_authorized": False})
    required_summary = "not_scored" if required_score is None else str(required_score["exact_match"])
    print(f"{role}: validation exact={task_score['exact_match']} format={task_score['format_valid']} required={required_summary}", flush=True)


def score_frontier_panel(root: Path) -> None:
    from neural_foundry.json_frontier_model import JsonFrontierModel, frontier_proof_configuration

    preparation = read_document(root / "preparation" / "training-preparation.json", "preparation_digest")
    panel = read_document(root / "public-validation-v2" / "panel.json", "document_digest")
    if (preparation.get("kind") != "nf_json_training_preparation_v2"
        or panel.get("kind") != "nf_json_public_validation_panel_v2"
        or len(panel.get("rows", [])) != 192
        or panel.get("training_preparation_digest") != preparation["preparation_digest"]
        or panel.get("disjointness_audit", {}).get("passed") is not True
        or panel.get("rows_digest") != _document_digest({"rows": panel.get("rows")})):
        raise GymJobError("frontier requires the frozen, digest-verified 192-case public panel")
    limits = {"maximum_api_attempts": len(panel["rows"]), "maximum_cost_microusd": 100000}
    model_context = JsonFrontierModel.from_env(**limits)
    model_context.require_proof_configuration()
    claim = root / "frontier-192-one-shot.claim"
    with claim.open("x", encoding="utf-8") as stream:
        json.dump({"panel_digest": panel["document_digest"], **limits}, stream, sort_keys=True)
    task = reference_task(rows_per_template=32)
    with model_context as model:
        _require_model_evidence(model, "frontier")
        task_score = score_public_rows(task, model, panel["rows"])
        for row, response in zip(panel["rows"], task_score["responses"], strict=True):
            _require_response_evidence(model, task.project(row["input"], None),
                ExecutionResponse(response["response"], response["execution_evidence"] or {}), "frontier")
        identity = dict(model.loaded_identity())
    configuration = frontier_proof_configuration(**limits)
    document = {
        "role": "gpt-6-luna",
        "split": "public_validation_v2",
        "complete": True,
        "panel_digest": panel["document_digest"],
        "panel_rows_digest": panel["rows_digest"],
        "preparation_digest": preparation["preparation_digest"],
        "task": task_score,
        "decoding_mode": {
            "temperature": configuration["temperature"],
            "reasoning_effort": configuration["reasoning_effort"],
            "max_completion_tokens": configuration["max_new_tokens"],
            "response_format": "strict_json_schema",
            "top_p": "provider_default_unset",
            "seed": "not_set",
        },
        "frontier_configuration": configuration,
        "model_identity": identity,
        "reserved_cost_microusd": max(
            (receipt["reserved_cost_microusd"] for receipt in identity["api_response_receipts"]), default=0),
        "api_attempts": len(identity["api_response_receipts"]),
        "holdout_accessed": False,
        "required_set_scored": False,
        "release_authorized": False,
    }
    write_document(root / "validation-frontier-192.json", document)
    print(f"gpt-6-luna: validation exact={task_score['exact_match']} format={task_score['format_valid']} "
          f"attempts={document['api_attempts']} reserved_cost_microusd={document['reserved_cost_microusd']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("merge-lora", "identity", "freeze-public-v2", "score", "score-frontier-panel"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--role", choices=("base", "sft", "lora", "frontier", "old-sft", "old-lora"))
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    if args.stage == "merge-lora":
        merge_lora(args.root.resolve())
    elif args.stage == "identity" and args.role in {"base", "sft", "lora", "old-sft", "old-lora"}:
        record_identity(args.root.resolve(), args.role, args.source_root.resolve() if args.source_root else None)
    elif args.stage == "freeze-public-v2" and args.role is None:
        panel = freeze_public_validation_v2(args.root.resolve())
        print(json.dumps({"document_digest": panel["document_digest"], "rows_digest": panel["rows_digest"],
                          "audit": panel["disjointness_audit"], "holdout_accessed": False}, sort_keys=True))
    elif args.stage == "score" and args.role:
        score(args.root.resolve(), args.role, args.source_root.resolve() if args.source_root else None)
    elif args.stage == "score-frontier-panel" and args.role is None:
        score_frontier_panel(args.root.resolve())
    else:
        parser.error("a supported role is required")


if __name__ == "__main__":
    main()