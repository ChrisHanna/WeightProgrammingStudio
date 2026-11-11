"""Public-only overlap statistics; no private dataset or custody dependencies."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


def input_text(row: Mapping[str, Any]) -> str:
    value = row.get("input", {}).get("text")
    if not isinstance(value, str):
        raise TypeError("audit row is missing input text")
    return value


def ngrams(text: str, size: int) -> set[tuple[str, ...]]:
    tokens = re.findall(r"[a-z0-9]+", text.casefold())
    return {tuple(tokens[index:index + size]) for index in range(len(tokens) - size + 1)}


def overlap_counts(left: Sequence[Mapping[str, Any]],
                   right: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    exact_right = {input_text(row) for row in right}
    result: dict[str, Any] = {
        "left_rows": len(left), "right_rows": len(right),
        "exact_input_matches": sum(input_text(row) in exact_right for row in left),
        "ngram_overlap": {},
    }
    for size in (3, 4, 5):
        right_sets = [ngrams(input_text(row), size) for row in right]
        right_union = set().union(*right_sets) if right_sets else set()
        shared_union: set[tuple[str, ...]] = set()
        overlapping = maximum_shared = 0
        maximum_jaccard = 0.0
        for row in left:
            grams = ngrams(input_text(row), size)
            shared = grams & right_union
            overlapping += bool(shared)
            shared_union.update(shared)
            for other in right_sets:
                maximum_shared = max(maximum_shared, len(grams & other))
                if grams | other:
                    maximum_jaccard = max(maximum_jaccard, len(grams & other) / len(grams | other))
        result["ngram_overlap"][str(size)] = {
            "left_rows_with_any_shared_ngram": overlapping,
            "distinct_shared_ngrams": len(shared_union),
            "maximum_shared_ngrams_for_one_pair": maximum_shared,
            "maximum_pairwise_jaccard": maximum_jaccard,
        }
    return result
