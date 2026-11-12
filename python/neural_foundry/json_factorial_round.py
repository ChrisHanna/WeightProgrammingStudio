"""Freeze the approved 0.6B factorial round and a separate UNSCORED public panel."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .gym_job import GymJobError
from .gym_service import _document_digest
from .json_confirmation_panel import (
    CLEAN_TEMPLATES,
    INJECTION_LAYOUTS,
    confirmation_rows,
    rendered_reference,
)
from .json_factorial_training import FACTORS, PHRASING_FAMILY
from .proof_pipeline import audit, read_document, validate_spec, write_document
from .proof_policy import load_policy, policy_reference
from .proof_registry import builtin_registry
from .provenance import sha256_file
from .public_data_audit import input_text, ngrams, overlap_counts

TRAIN_SEED = 20261118
CONFIRMATION_SEED = 20261129


def build_spec(source: Path, *, policy_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    historical = read_document(source)
    if (historical["run_id"] != "json-balanced-authority-contract-v2-20261111"
        or historical["base_model"]["model_id"] != "Qwen/Qwen3-0.6B"):
        raise GymJobError("factorial round requires the historical clean 0.6B contract")
    registry, plugins = builtin_registry()
    factory = registry.resolve("augmenter", plugins["factorial_augmenter"])
    compared = [panel for panel in historical["public_sets"] if panel["name"] in historical["audit"]["public_sets"]]
    forbidden_exact = {input_text(row) for panel in compared for row in panel["rows"]}
    forbidden_grams = set().union(*(ngrams(input_text(row), 5) for panel in compared for row in panel["rows"]))
    excluded = []
    for attempt in range(128):
        configuration = {"version": 1, "seed": TRAIN_SEED + attempt}
        new = factory(configuration)
        if all(input_text(row) not in forbidden_exact and not ngrams(input_text(row), 5) & forbidden_grams
               for row in new):
            break
        excluded.append({"configuration": configuration, "rows_digest": _document_digest({"rows": new})})
    else:
        raise GymJobError("bounded old-public-only factorial value rejection exhausted")
    base = [row for row in historical["training_rows"]
            if not row["scenario_id"].startswith("train:authority-v1:")]
    removed = [row for row in historical["training_rows"] if row not in base]
    if len(base) != 320 or len(removed) != 64 or len(new) != 64:
        raise GymJobError("factorial round must replace exactly 64 benign authority rows")
    rows = []
    for index, row in enumerate(base):
        rows.append(deepcopy(row))
        if index % 5 == 4:
            rows.append(new[index // 5])
    spec = {key: deepcopy(value) for key, value in historical.items() if key != "document_digest"}
    spec.update({
        "run_id": "json-factorial-authority-0p6b-20261111",
        "plugins": {kind: plugins[kind] for kind in historical["plugins"]},
        "training_rows": rows, "training_rows_digest": _document_digest({"rows": rows}),
        "augmentation": {"plugin": plugins["factorial_augmenter"], "configuration": configuration,
                         "rows_digest": _document_digest({"rows": new}), "new_rows": 64},
        "policy_reference": policy_reference(), "policy": load_policy(policy_reference()),
        "training_contract": {
            "version": 3, "base_rows": 320, "factorial_authority_rows": 64,
            "retained_clean_base_rows_digest": _document_digest({"rows": base}),
            "removed_benign_authority_rows_digest": _document_digest({"rows": removed}),
            "source_spec_digest": historical["document_digest"],
            "source_spec_file_digest": sha256_file(source),
            "factors": FACTORS, "cells": 8, "rows_per_cell": 8,
            "within_every_cell": {"trusted_format": ["plain", "json"], "value_pairs": 2, "swaps": 2},
            "matched_conditions": "same reference, decoy, imperative and all three alternate fields",
            "phrasing_family": PHRASING_FAMILY,
            "ordered_steps": 384, "ordering": "five_clean_base_then_one_factorial",
            "fresh_pinned_base_per_method": True, "continue_old_sft": False,
            "optimizer_and_recipe_unchanged": True, "model_outcomes_used": False,
            "no_quality_adaptive_retries": True,
            "prior_clean_rejection_receipt": historical["training_contract"]["rejection_sampling"],
            "factorial_value_rejection": {
                "initial_seed": TRAIN_SEED, "selected_configuration": configuration,
                "excluded_candidates": excluded, "comparisons": {
                    panel["name"]: panel["rows_digest"] for panel in compared},
                "unit": "complete_matched64_roster", "wording_and_labels_unchanged": True,
                "fresh_confirmation_used": False, "model_outcomes_used": False,
            },
        },
    })
    spec["plugins"]["augmenter"] = plugins["factorial_augmenter"]
    spec["audit"] = {**historical["audit"], "policy_path": str(policy_path.resolve()),
                     "policy_file_digest": sha256_file(policy_path)}
    for candidate in spec["candidates"]:
        if candidate["method"] not in {"sft", "lora"}:
            raise GymJobError("factorial round authorizes only one SFT and one LoRA")
        candidate["evidence"] = plugins["evidence"]
        candidate["execution"] = {"kind": "server", "plugin": plugins["server"]}
    if [panel["name"] for panel in spec["public_sets"]] != ["required16", "public64", "panel192"]:
        raise GymJobError("only the original public score sets are authorized")
    if spec["policy"]["api"]["calls_authorized"] is not False:
        raise GymJobError("factorial round forbids new API calls")
    validate_spec(spec, registry)
    return spec, audit(spec, registry)


def existing_public_data(project: Path) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Only explicit public/train artifact kinds; never enumerate holdout data files."""
    groups: dict[str, list[dict[str, Any]]] = {}
    sources = []
    artifacts = project / "artifacts"
    for directory in sorted(artifacts.iterdir()):
        if not directory.is_dir() or directory.name.startswith("phase-b-factorial-"):
            continue
        if not directory.name.startswith(("json-", "phase-b-balanced-authority-", "phase-b-clean-contract-")):
            continue
        paths = set(directory.rglob("run-spec.json")) | set(directory.rglob("training-preparation.json"))
        paths.update(path for path in directory.rglob("panel.json")
                     if any(part.startswith("public-") for part in path.relative_to(directory).parts[:-1]))
        for path in sorted(paths):
            field = "preparation_digest" if path.name == "training-preparation.json" else "document_digest"
            document = read_document(path, field)
            kind = document.get("kind", "")
            collected = []
            if kind in {"nf_public_proof_run", "nf_json_training_preparation_v2",
                        "nf_json_training_preparation_v1"}:
                collected.extend(document.get("training_rows", []))
                for panel in document.get("public_sets", []):
                    collected.extend(panel["rows"])
            elif kind in {"nf_json_public_validation_panel_v2", "nf_json_public_validation_panel_v3"}:
                collected.extend(document["rows"])
            if collected:
                name = str(path.relative_to(project))
                groups[name] = collected
                sources.append({"path": name, "file_digest": sha256_file(path),
                                "document_digest": document[field], "rows": len(collected),
                                "rows_digest": _document_digest({"rows": collected})})
    if not groups:
        raise GymJobError("no prior public/train comparisons found")
    return groups, sources


def audit_tokens(*, source: Path, output: Path, policy_path: Path) -> dict[str, Any]:
    from nemo_automodel.components.datasets.llm.chat_dataset import ChatDataset
    from transformers import AutoTokenizer

    from .prompt_rendering import PromptRenderer
    from .reference_json_extraction import project

    spec, _ = build_spec(source, policy_path=policy_path)
    confirmation = confirmation_rows(seed=CONFIRMATION_SEED)
    tokenizer = AutoTokenizer.from_pretrained(
        spec["base_model"]["model_id"], revision=spec["base_model"]["revision"],
        local_files_only=True, trust_remote_code=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset_path = output.with_name("token-audit-training.jsonl")
    with dataset_path.open("x", encoding="utf-8") as stream:
        for row in spec["training_rows"]:
            stream.write(json.dumps({"messages": project(row["input"], row["reference"])["messages"]}) + "\n")
    dataset = ChatDataset(str(dataset_path), tokenizer, mask_history=True, mask_generation_prompt=True)
    lengths = [len(item["input_ids"]) for item in dataset]
    renderer = PromptRenderer(tokenizer)
    prompt_sets = {"training": spec["training_rows"], "fresh_confirmation_UNSCORED": confirmation}
    prompt_sets.update({panel["name"]: panel["rows"] for panel in spec["public_sets"]})
    prompt_maxima = {name: max(len(renderer.render(project(row["input"], None)).token_ids)
                              for row in rows) for name, rows in prompt_sets.items()}
    passed = (len(lengths) == 384 and max(lengths) <= 256 and sum(lengths) <= 196608
              and all(value + spec["policy"]["max_new_tokens"] <= 512 for value in prompt_maxima.values()))
    receipt = write_document(output, {
        "kind": "nf_factorial_no_forward_token_audit_v1", "passed": passed,
        "training_rows_digest": spec["training_rows_digest"],
        "confirmation_rows_digest": _document_digest({"rows": confirmation}),
        "tokenizer_base": spec["base_model"], "renderer_digest": renderer.digest,
        "training_token_lengths": lengths, "maximum_training_tokens": max(lengths),
        "total_training_tokens": sum(lengths), "maximum_prompt_tokens": prompt_maxima,
        "maximum_input_characters": max(len(row["input"]["text"]) for rows in prompt_sets.values() for row in rows),
        "training_sequence_cap": 256, "context_cap": 512, "output_cap": 128,
        "dataset_file_digest": sha256_file(dataset_path), "model_loaded": False,
        "inference_calls": 0, "fresh_confirmation_scoring_authorized": False, "holdout_accessed": False,
    })
    if not passed:
        raise GymJobError("frozen sequence/context/token budgets exceeded")
    return receipt


def freeze(*, source: Path, output: Path, project: Path, policy_path: Path,
           token_audit: Path) -> dict[str, Any]:
    if output.exists():
        raise GymJobError("freeze destination must be fresh and exclusive")
    spec, training_audit = build_spec(source, policy_path=policy_path)
    confirmation = confirmation_rows(seed=CONFIRMATION_SEED)
    tokens = read_document(token_audit)
    if (tokens.get("passed") is not True or tokens["training_rows_digest"] != spec["training_rows_digest"]
        or tokens["confirmation_rows_digest"] != _document_digest({"rows": confirmation})):
        raise GymJobError("no-forward token audit is missing or differs")
    if any(rendered_reference(row) != row["reference"] for row in confirmation):
        raise GymJobError("independent rendered-source label check failed")
    comparisons, sources = existing_public_data(project)
    comparisons["new_factorial_train384"] = spec["training_rows"]
    # This is a one-shot reporting audit, never rejection feedback for either dataset.
    overlap = {name: overlap_counts(confirmation, rows) for name, rows in comparisons.items()}
    exact_disjoint = all(value["exact_input_matches"] == 0 for value in overlap.values())
    fivegram_disjoint = all(value["ngram_overlap"]["5"]["distinct_shared_ngrams"] == 0
                           for value in overlap.values())
    old_families = {row["family_id"] for rows in comparisons.values() for row in rows
                    if "family_id" in row}
    template_disjoint = not (set(CLEAN_TEMPLATES) | set(INJECTION_LAYOUTS)) & old_families
    timestamp = datetime.now(UTC).isoformat()
    code = {str(path.relative_to(project)): sha256_file(path)
            for path in (Path(__file__), Path(__file__).with_name("json_factorial_training.py"),
                         Path(__file__).with_name("json_confirmation_panel.py"),
                         Path(__file__).with_name("proof_registry.py"),
                         Path(__file__).with_name("proof_plugins.py"),
                         Path(__file__).with_name("proof_dependencies.py"),
                         project / "scripts" / "prepare_json_factorial_round.py",
                         project / "tests" / "test_json_factorial_round.py")}
    panel = {
        "kind": "nf_json_public_confirmation_unscored_v1", "frozen_at": timestamp,
        "seed": CONFIRMATION_SEED, "seed_independent_of_training": True,
        "rows": confirmation, "rows_digest": _document_digest({"rows": confirmation}),
        "templates": {**CLEAN_TEMPLATES, **{name: list(value) for name, value in INJECTION_LAYOUTS.items()}},
        "examples_per_template": 32, "factors": FACTORS, "injection_rows_per_cell": 12,
        "scoring_authorized": False, "scored": False, "public_only": True,
        "inference_calls": 0, "label_validation": "independent_rendered_genuine_source_parser_all192",
        "code_digests": code, "holdout_accessed": False, "release_authorized": False,
    }
    spec["frozen_at"] = timestamp
    spec["code_digests"] = code
    spec["no_forward_token_audit_digest"] = tokens["document_digest"]
    spec["confirmation_excluded_from_all_scoring_sets"] = True
    report = {
        "kind": "nf_public_confirmation_disjointness_once_v1", "frozen_at": timestamp,
        "comparisons": sources, "overlap": overlap,
        "input_disjoint": exact_disjoint, "template_families_disjoint": template_disjoint,
        "zero_shared_5grams": fivegram_disjoint,
        "training_required16_old192_zero_5gram_gate_unchanged": True,
        "panel_or_training_tuned_from_this_audit": False, "audit_runs": 1,
        "label_checks_passed": len(confirmation), "scoring_authorized": False,
        "blockers": ([] if exact_disjoint and template_disjoint and fivegram_disjoint
                     else ["fresh_confirmation_overlap"]),
        "holdout_accessed": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    frozen_spec = write_document(output / "run-spec.json", spec)
    frozen_panel = write_document(output / "fresh-confirmation-UNSCORED.json", panel)
    frozen_audit = write_document(output / "public-audit.json", training_audit)
    frozen_overlap = write_document(output / "confirmation-disjointness-once.json", report)
    receipt = write_document(output / "freeze-receipt.json", {
        "kind": "nf_factorial_round_pre_results_freeze_v1", "frozen_at": timestamp,
        "run_spec_digest": frozen_spec["document_digest"],
        "training_rows_digest": spec["training_rows_digest"],
        "augmentation_rows_digest": spec["augmentation"]["rows_digest"],
        "confirmation_digest": frozen_panel["document_digest"],
        "confirmation_rows_digest": panel["rows_digest"],
        "public_audit_digest": frozen_audit["document_digest"],
        "confirmation_audit_digest": frozen_overlap["document_digest"],
        "no_forward_token_audit_digest": tokens["document_digest"],
        "no_forward_token_audit_path": str(token_audit),
        "code_digests": code, "before_any_new_evaluation": True,
        "training_cases": 384, "steps_per_method": 384, "confirmation_cases": 192,
        "confirmation_by_template": dict(Counter(row["family_id"] for row in confirmation)),
        "fresh_confirmation_scoring_authorized": False, "inference_calls": 0,
        "status": "READY_FOR_RESULTS" if not report["blockers"] else "BLOCKED",
        "blockers": report["blockers"], "gpu_training_started": False,
        "holdout_accessed": False, "release_authorized": False,
    })
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-audit", type=Path, required=True)
    parser.add_argument("--audit-tokens-only", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    policy_path = project / "protocols" / "json-contamination-v1.json"
    if args.audit_tokens_only:
        print(audit_tokens(source=args.source.resolve(), output=args.token_audit.resolve(), policy_path=policy_path))
        return
    receipt = freeze(source=args.source.resolve(), output=args.output.resolve(), project=project,
                     policy_path=policy_path, token_audit=args.token_audit.resolve())
    print(receipt)
    if receipt["status"] != "READY_FOR_RESULTS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
