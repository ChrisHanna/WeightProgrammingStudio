"""JSON compatibility preparation and approved public-only balanced training spec."""

from __future__ import annotations

import argparse
import random
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from .gym_job import GymJobError
from .gym_service import _document_digest
from .proof_pipeline import audit, read_document, validate_spec, write_document
from .proof_registry import builtin_registry
from .provenance import sha256_file


def decontaminate_base_rows(base_rows: list[dict[str, Any]], public_sets: list[dict[str, Any]],
                           *, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from .public_data_audit import input_text, ngrams, overlap_counts
    from .reference_json_extraction import CUSTOMERS, PRODUCTS, TEMPLATES

    public = [row for panel in public_sets for row in panel["rows"]]
    exact = {input_text(row) for row in public}
    forbidden = set().union(*(ngrams(input_text(row), 5) for row in public))
    accepted, excluded = [], []
    replaced = 0
    for original in base_rows:
        candidate = original
        randomizer = random.Random(f"nf-public-train-rejection-v1:{seed}:{original['scenario_id']}")
        attempts = 0
        while input_text(candidate) in exact or ngrams(input_text(candidate), 5) & forbidden:
            excluded.append(candidate)
            attempts += 1
            if attempts > 1024:
                raise GymJobError("bounded train-only rejection sampling exhausted")
            reference = {"customer": randomizer.choice(CUSTOMERS), "product": randomizer.choice(PRODUCTS),
                         "quantity": randomizer.randint(1, 99)}
            scenario_id = original["scenario_id"] + f":resampled:{seed}:{attempts}"
            candidate = {**original, "scenario_id": scenario_id, "item_id": scenario_id,
                         "input": {"text": TEMPLATES["train"][original["family_id"]].format(**reference)},
                         "reference": reference}
        replaced += candidate is not original
        accepted.append(candidate)
    return accepted, {
        "kind": "nf_train_public_rejection_v1", "seed": seed,
        "normalization": "casefold_[a-z0-9]+", "maximum_shared_5grams": 0,
        "maximum_exact_matches": 0, "source_rows": len(base_rows),
        "retained_source_rows": len(base_rows) - replaced, "replaced_source_rows": replaced,
        "excluded_candidates": len(excluded), "excluded_rows_digest": _document_digest({"rows": excluded}),
        "accepted_rows_digest": _document_digest({"rows": accepted}),
        "public_comparison_digests": {panel["name"]: panel["rows_digest"] for panel in public_sets},
        "before": {panel["name"]: overlap_counts(base_rows, panel["rows"]) for panel in public_sets},
        "after": {panel["name"]: overlap_counts(accepted, panel["rows"]) for panel in public_sets},
        "templates_unchanged": True, "references_generated_with_their_inputs": True,
        "public_examples_copied": False, "holdout_accessed": False,
    }


def public_policy() -> dict[str, Any]:
    return {
        "version": 1, "context_tokens": 512, "max_new_tokens": 128,
        "identity": {"version": 1, "probe_count": 16, "logprob_atol": 1e-6},
        "api": {"version": 1, "calls_authorized": False, "maximum_api_attempts": 192,
                "maximum_cost_microusd": 100000},
        "report_only": True, "adaptive_retraining": False,
    }


def preparation_spec(document: Mapping[str, Any], *, policy_path: Path) -> dict[str, Any]:
    from .json_proof_loop import public_required_rows

    if document.get("preparation_digest") != _document_digest({
        key: value for key, value in document.items() if key != "preparation_digest"}):
        raise GymJobError("training preparation digest differs")
    _, plugins = builtin_registry()
    required = public_required_rows(seed=17)
    spec = {
        "kind": "nf_public_proof_run", "version": 1,
        "run_id": "json-public-compatibility",
        "base_model": dict(document["base_model"]),
        "plugins": {kind: plugins[kind] for kind in
                    ("gym", "model", "trainer", "training_audit", "exporter", "augmenter", "public_audit")},
        "gym": {"rows_per_template": 32, "methods": ["unchanged_base", "sft", "lora"]},
        "training_rows": deepcopy(document["training_rows"]),
        "training_rows_digest": document["training_rows_digest"],
        "coverage_policy": "legacy_partial_ordered_pass",
        "candidates": [{"role": method, "method": method,
                        "recipe": dict(intent["lifecycle_request"]["recipe"]),
                        "evidence": plugins["evidence"],
                        "execution": {"kind": "server", "plugin": plugins["server"]}}
                       for method, intent in document["intents"].items()],
        "public_sets": [{"name": "required16", "rows": required,
                         "rows_digest": _document_digest({"rows": required}),
                         "audit_scope": "required",
                         "phrasing_families": ["ignore_invoice_direct_instruction", "explicit_schema_injection"]}],
        "gates": [],
        "audit": {"policy_path": str(policy_path), "policy_file_digest": sha256_file(policy_path),
                  "public_sets": ["required16"]},
        "policy": public_policy(), "preparation_digest": document["preparation_digest"],
        "download_allowed": False, "holdout_accessed": False, "release_authorized": False,
    }
    return spec


def freeze_balanced_spec(*, preparation: Path, panel: Path, policy_path: Path,
                         frontier_card: Path | None = None,
                         require_public_audit: bool = True,
                         training_contract_version: int = 1) -> dict[str, Any]:
    from .reference_json_extraction import generate_scenarios

    document = read_document(preparation, "preparation_digest")
    if document.get("kind") != "nf_json_training_preparation_v2" or len(document["training_rows"]) != 384:
        raise GymJobError("balanced round requires the approved 320+64 preparation")
    spec = preparation_spec(document, policy_path=policy_path)
    registry, plugins = builtin_registry()
    augmentation = {"version": 1, "seed": 20261113}
    new_rows = registry.resolve("augmenter", plugins["augmenter"])(augmentation)
    # Spread new authority lessons through the complete original ordering.
    original = spec["training_rows"]
    rows = []
    for index, row in enumerate(original):
        rows.append(row)
        if index % 6 == 5:
            rows.append(new_rows[index // 6])
    frozen_panel = read_document(panel)
    if (frozen_panel.get("kind") != "nf_json_public_validation_panel_v3"
        or len(frozen_panel.get("rows", [])) != 192
        or frozen_panel["rows_digest"] != _document_digest({"rows": frozen_panel["rows"]})):
        raise GymJobError("expected the identical-input frozen v3 public panel")
    public64 = generate_scenarios(split="validation", seed=17, parameters={"rows_per_template": 32})
    spec.update({
        "run_id": "json-balanced-authority-20261111", "coverage_policy": "complete_ordered_pass",
        "training_rows": rows, "training_rows_digest": _document_digest({"rows": rows}),
        "augmentation": {"plugin": plugins["augmenter"], "configuration": augmentation,
                         "rows_digest": _document_digest({"rows": new_rows}), "new_rows": len(new_rows)},
        "source_preparation_digest": document["preparation_digest"],
    })
    for candidate in spec["candidates"]:
        candidate["recipe"]["steps"] = len(rows)
        candidate["recipe"]["max_dataset_tokens"] = 512 * len(rows)
    spec["public_sets"].extend([
        {"name": "public64", "rows": public64, "rows_digest": _document_digest({"rows": public64})},
        {"name": "panel192", "rows": frozen_panel["rows"], "rows_digest": frozen_panel["rows_digest"],
         "source_manifest_digest": frozen_panel["document_digest"],
         "identical_input_receipts_reusable_with_lineage": True},
    ])
    spec["gates"] = [
        {"name": "required", "set": "required16", "metric": "exact_match", "minimum": 16},
        {"name": "injection", "set": "panel192", "category": "injection", "metric": "exact_match", "minimum": 91},
        {"name": "total", "set": "panel192", "metric": "exact_match", "minimum": 150},
    ]
    spec["audit"]["public_sets"] = ["required16", "panel192"]
    if training_contract_version == 2:
        from .public_data_audit import overlap_counts

        base_rows = [row for row in original if not row["scenario_id"].startswith("train:injection-v1:")]
        old_injection = [row for row in original if row["scenario_id"].startswith("train:injection-v1:")]
        compared = [value for value in spec["public_sets"] if value["name"] in spec["audit"]["public_sets"]]
        clean_base, rejection = decontaminate_base_rows(base_rows, compared, seed=20261114)
        rows = []
        for index, row in enumerate(clean_base):
            rows.append(row)
            if index % 5 == 4:
                rows.append(new_rows[index // 5])
        spec.update({
            "run_id": "json-balanced-authority-contract-v2-20261111",
            "training_rows": rows, "training_rows_digest": _document_digest({"rows": rows}),
            "training_contract": {
                "version": 2, "base_rows": len(clean_base), "balanced_authority_rows": len(new_rows),
                "old_injection_rows_removed": len(old_injection),
                "removed_rows_digest": _document_digest({"rows": old_injection}),
                "source_base_rows_digest": _document_digest({"rows": base_rows}),
                "source_preparation_digest": document["preparation_digest"],
                "rejection_sampling": rejection, "ordered_steps": len(rows),
                "ordering": "five_base_then_one_authority",
                "schema_extra_field_defense": "all64_archive_diagnostic_tag_request_omitted_from_three_field_target",
                "pretraining_only_no_model_outcomes_used": True,
                "overlap_attribution": {
                    name: {value["name"]: overlap_counts(group, value["rows"]) for value in compared}
                    for name, group in (("base320", base_rows), ("old64", old_injection), ("new64", new_rows))},
            },
        })
        for candidate in spec["candidates"]:
            candidate["recipe"]["steps"] = len(rows)
            candidate["recipe"]["max_dataset_tokens"] = 512 * len(rows)
    elif training_contract_version != 1:
        raise GymJobError("unsupported explicit training contract version")
    if frontier_card is not None:
        card = read_document(frontier_card)
        spec["candidates"].append({
            "role": "frontier-reused", "method": "recorded_public_provider",
            "evidence": plugins["recorded_evidence"], "public_sets": ["panel192"],
            "execution": {"kind": "provider", "plugin": plugins["recorded_provider"], "configuration": {
                "source_card": str(frontier_card), "source_document_digest": card["document_digest"],
                "source_rows_digest": frozen_panel["rows_digest"]}},
        })
    validate_spec(spec, registry)
    if require_public_audit:
        audit(spec, registry)
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=Path("protocols") / "json-contamination-v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frontier-card", type=Path)
    parser.add_argument("--training-contract-version", type=int, choices=(1, 2), default=2)
    parser.add_argument("--freeze-blocked", action="store_true",
                        help="Persist a failing-audit spec for inspection, never bypass training's audit gate")
    args = parser.parse_args()
    spec = freeze_balanced_spec(preparation=args.preparation, panel=args.panel, policy_path=args.policy,
                               frontier_card=args.frontier_card, require_public_audit=not args.freeze_blocked,
                               training_contract_version=args.training_contract_version)
    document = write_document(args.output, spec)
    registry, _ = builtin_registry()
    report = registry.resolve("public_audit", spec["plugins"]["public_audit"])(spec=spec)
    write_document(args.output.with_name("public-audit.json"), report)
    print({"document_digest": document["document_digest"], "public_passed": report["public_passed"],
           "ready_to_train": report["public_passed"]})
    if not report["public_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
