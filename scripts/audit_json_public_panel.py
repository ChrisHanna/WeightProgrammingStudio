"""Audit public clean-panel labels and freeze an audit-only v3 panel; never read holdout."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from neural_foundry.gym_job import GymJobError, validate_task_instance
from neural_foundry.gym_service import _document_digest
from neural_foundry.provenance import parse_strict_json
from neural_foundry.reference_json_extraction import OUTPUT_SCHEMA

_CURRENT_RECORD = {
    "ap_posted_table": r"POSTED line: (?P<quantity>\d+) x (?P<product>[^\n]+?) for (?P<customer>[^\n]+)\nUse the posted line\.",
    "goods_receipt_crosscheck": r"Accepted: (?P<quantity>\d+) (?P<product>[^\n]+)\nReceiving account: (?P<customer>[^\n]+)\.",
    "remittance_reconciliation": r"settled invoice=(?P<quantity>\d+) (?P<product>[^;]+); payee=(?P<customer>[^;]+)\.",
}
_DISTRACTOR_RECORD = {
    "ap_posted_table": r"VOID line: (?P<quantity>\d+) x (?P<product>[^\n]+?) for (?P<customer>[^\n]+)\n",
    "goods_receipt_crosscheck": r"Expected \(not received\): (?P<quantity>\d+) (?P<product>[^\n]+?) for (?P<customer>[^\n]+)\n",
    "remittance_reconciliation": r"previous transfer=(?P<quantity>\d+) to (?P<customer>[^;]+?) for (?P<product>[^;]+);",
}


def _record(pattern: str, text: str) -> dict[str, Any]:
    matches = list(re.finditer(pattern, text))
    if len(matches) != 1:
        raise GymJobError("public label audit requires exactly one identifiable record")
    values = matches[0].groupdict()
    result = {"customer": values["customer"], "product": values["product"],
              "quantity": int(values["quantity"])}
    validate_task_instance(OUTPUT_SCHEMA, result, field="audited_record")
    return result


def _verify(document: Mapping[str, Any], digest_field: str) -> None:
    if document.get(digest_field) != _document_digest({
        key: value for key, value in document.items() if key != digest_field
    }):
        raise GymJobError("public audit source digest differs")


def audit_panel(panel: Mapping[str, Any], frontier: Mapping[str, Any]) -> dict[str, Any]:
    _verify(panel, "document_digest")
    _verify(frontier, "document_digest")
    rows = panel["rows"]
    if (panel.get("kind") != "nf_json_public_validation_panel_v2"
        or panel.get("rows_digest") != _document_digest({"rows": rows})
        or len(rows) != 192
        or frontier.get("complete") is not True
        or frontier.get("panel_digest") != panel["document_digest"]
        or frontier.get("panel_rows_digest") != panel["rows_digest"]):
        raise GymJobError("public audit requires the completed frontier on the frozen v2 panel")
    responses = frontier["task"]["responses"]
    if (len({row["scenario_id"] for row in rows}) != len(rows)
        or [row["scenario_id"] for row in rows] != [row["scenario_id"] for row in responses]):
        raise GymJobError("public audit response roster differs")
    templates: dict[str, dict[str, int]] = {}
    failures = []
    for row, response in zip(rows, responses, strict=True):
        if row["panel_category"] != "clean_compositional":
            continue
        family = row["family_id"]
        if family not in _CURRENT_RECORD:
            raise GymJobError("unknown clean-template selection contract")
        text = row["input"]["text"]
        expected = _record(_CURRENT_RECORD[family], text)
        distractor = _record(_DISTRACTOR_RECORD[family], text)
        counts = templates.setdefault(family, {
            "cases": 0, "label_mismatches": 0, "frontier_correct": 0,
            "frontier_full_distractor": 0, "frontier_other_error": 0,
        })
        counts["cases"] += 1
        counts["label_mismatches"] += expected != row["reference"]
        correct = response["response"] == expected
        counts["frontier_correct"] += correct
        if not correct:
            full_distractor = response["response"] == distractor
            counts["frontier_full_distractor"] += full_distractor
            counts["frontier_other_error"] += not full_distractor
            failures.append({
                "scenario_id": row["scenario_id"],
                "expected": expected,
                "response": response["response"],
                "classification": "full_distractor" if full_distractor else "other_or_mixed_record",
            })
    if {family: counts["cases"] for family, counts in templates.items()} != {
        family: 32 for family in _CURRENT_RECORD
    }:
        raise GymJobError("public audit requires all 96 clean cases")
    report: dict[str, Any] = {
        "kind": "nf_json_public_label_audit_v1",
        "source_panel_digest": panel["document_digest"],
        "source_rows_digest": panel["rows_digest"],
        "frontier_card_digest": frontier["document_digest"],
        "clean_cases": 96,
        "label_mismatches": sum(counts["label_mismatches"] for counts in templates.values()),
        "by_template": templates,
        "public_frontier_errors": failures,
        "selection_contract": {
            "ap_posted_table": "POSTED line, not VOID line",
            "goods_receipt_crosscheck": "Accepted goods and Receiving account, not Expected (not received)",
            "remittance_reconciliation": "settled invoice and payee, not previous transfer",
        },
        "holdout_accessed": False,
        "new_api_calls": 0,
    }
    report["report_digest"] = _document_digest(report)
    return report


def freeze_v3(panel: Mapping[str, Any], audit: Mapping[str, Any]) -> dict[str, Any]:
    _verify(panel, "document_digest")
    _verify(audit, "report_digest")
    if (audit.get("source_panel_digest") != panel["document_digest"]
        or audit.get("source_rows_digest") != panel["rows_digest"]
        or audit.get("kind") != "nf_json_public_label_audit_v1"
        or audit.get("clean_cases") != 96
        or audit.get("label_mismatches") != 0):
        raise GymJobError("cannot freeze a v3 panel with unresolved public labels")
    document = {key: value for key, value in panel.items() if key != "document_digest"}
    document.update({
        "kind": "nf_json_public_validation_panel_v3",
        "source_panel_digest": panel["document_digest"],
        "label_audit_digest": audit["report_digest"],
        "selection_contract": audit["selection_contract"],
        "version_change": "label-audit metadata only; identical prompts, references, IDs and order",
        "independent_confirmation_panel": False,
        "prior_public_exposure_retained": True,
    })
    document["document_digest"] = _document_digest(document)
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--frontier-card", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panel = parse_strict_json(args.panel.read_bytes())
    frontier = parse_strict_json(args.frontier_card.read_bytes())
    audit = audit_panel(panel, frontier)
    frozen = freeze_v3(panel, audit)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, document in (("label-audit.json", audit), ("panel.json", frozen)):
        with (args.output / name).open("x", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"audit_digest": audit["report_digest"], "label_mismatches": audit["label_mismatches"],
                      "by_template": audit["by_template"], "panel_digest": frozen["document_digest"],
                      "holdout_accessed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
