"""Reproduce v1 and apply v2 to saved validation answers; never execute a provider."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_foundry.cord_receipt_plugin import (
    EVALUATION_POLICY,
    evaluation_policy,
    receipt_metrics,
    receipt_metrics_v2,
    summarize,
)
from neural_foundry.gym_job import GymJobError
from neural_foundry.gym_service import _document_digest
from neural_foundry.proof_pipeline import read_document, write_document
from neural_foundry.provenance import parse_strict_json, sha256_file


def rescore(source_root: Path) -> dict:
    models = {}
    sources = {}
    for role in ("base06b", "base4b"):
        spec_path = source_root / f"spec-{role}.json"
        card_path = source_root / role / f"public-score-{role}.json"
        spec, card = read_document(spec_path), read_document(card_path)
        if (card.get("complete") is not True or card["run_spec_digest"] != spec["document_digest"]
            or card["role"] != role):
            raise GymJobError("offline CORD rescore requires a matching completed frozen scorecard")
        if [panel["name"] for panel in spec["public_sets"]] != ["cord-clean", "cord-bipia"]:
            raise GymJobError("offline CORD rescore is restricted to the two frozen validation panels")
        records = []
        index = 0
        for panel in spec["public_sets"]:
            rows = panel["rows"]
            saved = card["scores"][panel["name"]]
            if (len(rows) != 100 or saved["cases"] != 100 or len(saved["responses"]) != 100
                or saved["rows_digest"] != panel["rows_digest"]
                or panel["rows_digest"] != _document_digest({"rows": rows})):
                raise GymJobError("offline CORD rescore requires every original validation case")
            for row, response in zip(rows, saved["responses"], strict=True):
                if row["native_split"] != "validation" or row["scenario_id"] != response["scenario_id"]:
                    raise GymJobError("offline CORD rescore refuses non-validation or reordered rows")
                path = source_root / role / f"public-score-{role}-responses" / f"{index:06d}.json"
                journal = read_document(path)
                if (journal["run_spec_digest"] != spec["document_digest"]
                    or journal["row_digest"] != _document_digest(row)
                    or journal["public_set"] != panel["name"]
                    or journal["response_record"] != response):
                    raise GymJobError("offline CORD rescore journal differs from its saved answer")
                evidence = response["execution_evidence"]
                if evidence.get("response_format") != "json_schema":
                    raise GymJobError("offline CORD rescore excludes earlier misconfigured attempts")
                if (evidence["completion_finish_reason"] == "stop"
                    and not evidence.get("invalid_completion_json")
                    and parse_strict_json(evidence["raw_text"].encode()) != response["response"]):
                    raise GymJobError("saved parsed answer differs from raw completion")
                kwargs = {"reference": row["reference"], "response": response["response"],
                          "output_valid": response["format_valid"]}
                legacy, current = receipt_metrics(**kwargs), receipt_metrics_v2(**kwargs)
                original = legacy if response["format_valid"] else {key: 0.0 for key in legacy}
                if original != response["metrics"]:
                    raise GymJobError("v1 no longer reproduces the original frozen per-case metrics")
                records.append({"scenario_id": row["scenario_id"], "public_set": panel["name"],
                                "source_response_digest": _document_digest(response),
                                "v1": legacy, "v2": current})
                index += 1
        sources[str(spec_path)] = sha256_file(spec_path)
        sources[str(card_path)] = sha256_file(card_path)
        models[role] = {"base_model": spec["base_model"], "source_score_digest": card["document_digest"],
                        "source_run_spec_digest": spec["document_digest"],
                        "v1": summarize(spec, card, scorer_version="1"),
                        "v2": summarize(spec, card, scorer_version="2"),
                        "responses": records}
    for path, digest in sources.items():
        if sha256_file(path) != digest:
            raise GymJobError("offline source files changed during rescoring")
    return {
        "kind": "nf_cord_offline_scorer_comparison_v2", "models": models,
        "saved_responses_rescored": 400, "v1_per_case_metrics_reproduced": True,
        "evaluation_policy": evaluation_policy(),
        "policy_reference": {"path": str(EVALUATION_POLICY), "file_digest": sha256_file(EVALUATION_POLICY)},
        "scorer_source_digest": sha256_file(receipt_metrics.__code__.co_filename),
        "source_files": sources, "new_inference_calls": 0, "training_performed": False,
        "test_accessed": False, "internal_evaluation_only": True,
        "public_bipia_performance_claims_authorized": False,
        "comparison_caveat": "v2 excludes parent indices from extraction metrics; this is a scoring change, not a model gain",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = write_document(args.output.resolve(), rescore(args.source_root.resolve()))
    print(json.dumps({key: result[key] for key in (
        "kind", "saved_responses_rescored", "v1_per_case_metrics_reproduced",
        "new_inference_calls", "document_digest")}, sort_keys=True))


if __name__ == "__main__":
    main()
