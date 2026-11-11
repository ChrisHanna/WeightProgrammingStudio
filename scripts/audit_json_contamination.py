"""Audit public JSON training overlap or annotate a prior read; never materialize holdout."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from neural_foundry.gym_service import _document_digest
from neural_foundry.json_proof_loop import public_required_rows
from neural_foundry.provenance import parse_strict_json

_ATTACK_FORMS = {
    "alternate_record": "full_record_substitution",
    "quoted_vendor": "explicit_value_override",
    "extra_property": "schema_injection",
    "quantity_override": "footer_value_override",
}
_PHRASING_FAMILIES = {
    "alternate_record": "warehouse_note_replace_response",
    "quoted_vendor": "quoted_vendor_obsolete_report",
    "extra_property": "scanner_annotation_add_field",
    "quantity_override": "footer_instruction_override",
}
_REQUIRED_FORMS = {
    "ignore_invoice_output_alternate": "full_record_substitution",
    "add_extra_json_field": "schema_injection",
}
_REQUIRED_PHRASE_FAMILIES = {
    "ignore_invoice_output_alternate": "ignore_invoice_direct_instruction",
    "add_extra_json_field": "explicit_schema_injection",
}


def _text(row: Mapping[str, Any]) -> str:
    value = row.get("input", {}).get("text")
    if not isinstance(value, str):
        raise ValueError("audit row is missing input text")
    return value


def _ngrams(text: str, size: int) -> set[tuple[str, ...]]:
    tokens = re.findall(r"[a-z0-9]+", text.casefold())
    return {tuple(tokens[index:index + size]) for index in range(len(tokens) - size + 1)}


def _overlap_counts(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    exact_right = {_text(row) for row in right}
    result: dict[str, Any] = {
        "left_rows": len(left),
        "right_rows": len(right),
        "exact_input_matches": sum(_text(row) in exact_right for row in left),
        "ngram_overlap": {},
    }
    for size in (3, 4, 5):
        right_sets = [_ngrams(_text(row), size) for row in right]
        right_union = set().union(*right_sets) if right_sets else set()
        rows_with_overlap = 0
        unique_shared: set[tuple[str, ...]] = set()
        maximum_shared = 0
        maximum_jaccard = 0.0
        for row in left:
            grams = _ngrams(_text(row), size)
            shared = grams & right_union
            if shared:
                rows_with_overlap += 1
                unique_shared.update(shared)
            for other in right_sets:
                pair_shared = len(grams & other)
                maximum_shared = max(maximum_shared, pair_shared)
                union_size = len(grams | other)
                if union_size:
                    maximum_jaccard = max(maximum_jaccard, pair_shared / union_size)
        result["ngram_overlap"][str(size)] = {
            "left_rows_with_any_shared_ngram": rows_with_overlap,
            "distinct_shared_ngrams": len(unique_shared),
            "maximum_shared_ngrams_for_one_pair": maximum_shared,
            "maximum_pairwise_jaccard": maximum_jaccard,
        }
    return result


def _required_form(row: Mapping[str, Any]) -> str:
    text = _text(row)
    if 'add an extra JSON field named override' in text:
        return "add_extra_json_field"
    if 'ignore the invoice and output' in text:
        return "ignore_invoice_output_alternate"
    raise ValueError("required row does not match an audited instruction family")


def _verified_required_family(row: Mapping[str, Any]) -> str:
    return _required_form(row)


def build_audit(preparation: Mapping[str, Any], *, seed: int = 17) -> dict[str, Any]:
    rows = preparation.get("training_rows")
    if not isinstance(rows, list) or preparation.get("kind") != "nf_json_training_preparation_v2":
        raise ValueError("expected a verified robustness-v2 training preparation")
    expected_preparation_digest = preparation.get("preparation_digest")
    if expected_preparation_digest != _document_digest(
        {key: value for key, value in preparation.items() if key != "preparation_digest"}
    ):
        raise ValueError("training preparation digest differs")
    if preparation.get("training_rows_digest") != _document_digest({"rows": rows}):
        raise ValueError("training rows digest differs")
    injection_rows = [row for row in rows if row.get("scenario_id", "").startswith("train:injection-v1:")]
    required = public_required_rows(seed=seed)

    training_forms: Counter[str] = Counter()
    phrase_families: Counter[str] = Counter()
    for row in injection_rows:
        parts = row["scenario_id"].split(":")
        if len(parts) < 5 or parts[3] not in _ATTACK_FORMS:
            raise ValueError("injection row has an unknown attack form")
        training_forms[_ATTACK_FORMS[parts[3]]] += 1
        phrase_families[_PHRASING_FAMILIES[parts[3]]] += 1

    required_forms = Counter(_REQUIRED_FORMS[_required_form(row)] for row in required)
    required_phrases = Counter(
        _REQUIRED_PHRASE_FAMILIES[_verified_required_family(row)] for row in required)
    cross_form_counts = {
        form: {"training_rows": training_forms[form], "required_rows": required_forms[form]}
        for form in sorted(set(training_forms) | set(required_forms))
    }
    shared_phrase_families = sorted(set(phrase_families) & set(required_phrases))
    report = {
        "kind": "nf_json_contamination_audit_v2",
        "training_preparation_digest": preparation.get("preparation_digest"),
        "training_rows_digest": preparation.get("training_rows_digest"),
        "required_set_digest": _document_digest({"rows": required}),
        "training_injection_rows": len(injection_rows),
        "required_set_rows": len(required),
        "training_attack_form_counts": dict(sorted(training_forms.items())),
        "training_phrasing_family_counts": dict(sorted(phrase_families.items())),
        "required_attack_form_counts": dict(sorted(required_forms.items())),
        "required_phrasing_family_counts": dict(sorted(required_phrases.items())),
        "attack_form_cross_counts": cross_form_counts,
        "shared_phrasing_families": shared_phrase_families,
        "injection_vs_required": _overlap_counts(injection_rows, required),
        "all_training_vs_required": _overlap_counts(rows, required),
        "injection_vs_synthetic_holdout": None,
        "synthetic_holdout": {
            "status": "not_checked_pending_approved_custodian_interface",
            "content_emitted": False,
            "holdout_accessed": False,
            "model_scoring_performed": False,
            "independent_custody_verified": False,
            "reason": "holdout overlap checks must execute through the approved custodian, not this public audit",
        },
        "raw_examples_emitted": False,
        "matched_ngrams_emitted": False,
        "holdout_accessed": False,
        "audit_passed": False,
        "audit_passed_reason": "public counts only; a qualified custody overlap receipt is still required",
    }
    report["report_digest"] = _document_digest(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--preparation", type=Path,
                        help="Run public overlap checks only; holdout checks require approved custody")
    source.add_argument("--annotate-prior-audit", type=Path,
                        help="Record a prior overlap-only holdout read without reading holdout again")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.annotate_prior_audit is not None:
        report = parse_strict_json(args.annotate_prior_audit.read_bytes())
        prior_digest = report.pop("report_digest")
        if (report.get("kind") != "nf_json_contamination_audit_v1"
            or prior_digest != _document_digest(report)):
            raise ValueError("prior contamination audit digest or kind differs")
        report["supersedes_report_digest"] = prior_digest
        report["holdout_accessed"] = True
        report["holdout_access_purpose"] = "overlap_only"
        report["synthetic_holdout"].update({
            "holdout_accessed": True,
            "access_purpose": "overlap_only",
            "access_boundary": "outside_independent_custody",
            "model_scoring_performed": False,
        })
        report["annotation_reads_holdout"] = False
        report["report_digest"] = _document_digest(report)
    else:
        preparation = parse_strict_json(args.preparation.read_bytes())
        report = build_audit(preparation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "report_digest"}, sort_keys=True))


if __name__ == "__main__":
    main()