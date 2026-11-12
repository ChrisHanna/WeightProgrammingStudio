"""Strict semantic-span masks with explicit causal next-token alignment."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from ...provenance import parse_strict_json
from .demonstrations import Demonstration, prepare_demonstration
from .loader import WeightEditInputError


def json_value_spans(response: str, fields: Sequence[str]) -> list[tuple[int, int, str]]:
    values = parse_strict_json(response.encode())
    if (not isinstance(values, dict) or not fields or len(set(fields)) != len(fields)
        or any(not isinstance(field, str) or field not in values for field in fields)):
        raise WeightEditInputError("JSON value helper requires explicit unique existing fields")
    parts, spans = ["{"], []
    for index, field in enumerate(sorted(values)):
        parts.append(("," if index else "") + json.dumps(field) + ":")
        value = values[field]
        literal = json.dumps(value, allow_nan=False)
        start = sum(map(len, parts))
        if field in fields:
            if isinstance(value, (dict, list)) or (isinstance(value, str) and "\\" in literal):
                raise WeightEditInputError("nested or escaped JSON semantic boundaries are unsupported")
            quoted = isinstance(value, str)
            spans.append((start + int(quoted), start + len(literal) - int(quoted), field))
        parts.append(literal)
    parts.append("}")
    if "".join(parts) != response:
        raise WeightEditInputError("JSON helper requires exact compact sorted JSON")
    return spans


def prepare_value_demonstration(renderer, interaction: Mapping, *, spans=None, value_fields=None):
    base = prepare_demonstration(renderer, interaction)
    response = interaction["messages"][-1]["content"]
    if spans is not None and value_fields is not None:
        raise WeightEditInputError("choose task-provided spans or JSON helper fields, not both")
    if spans is None:
        if interaction.get("format") != "json_extraction_v1" or value_fields is None:
            raise WeightEditInputError("value objective requires task semantic spans or JSON helper configuration")
        spans = json_value_spans(response, value_fields)
    if not isinstance(spans, (list, tuple)) or not spans:
        raise WeightEditInputError("semantic value spans must be nonempty")
    previous = 0
    for item in spans:
        if (not isinstance(item, (list, tuple)) or len(item) != 3
            or type(item[0]) is not int or type(item[1]) is not int
            or not previous <= item[0] < item[1] <= len(response)
            or not isinstance(item[2], str) or not item[2]):
            raise WeightEditInputError("semantic spans must be ordered, disjoint, nonempty character ranges with groups")
        previous = item[1]
    encoded = renderer.tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
    response_ids = renderer.tokenizer.encode(response, add_special_tokens=False)
    offsets = encoded["offset_mapping"]
    if encoded["input_ids"] != response_ids or len(offsets) != len(response_ids):
        raise WeightEditInputError("value offsets differ from frozen response token IDs")
    prompt_length = len(base.rendered.token_ids) - len(response_ids) - 1
    labels = [-100] * len(base.labels)
    positions, groups, covered = [], [], set()
    for index, (left, right) in enumerate(offsets):
        if not 0 <= left < right <= len(response):
            raise WeightEditInputError("unsupported token offsets")
        overlap = [(start, stop, group) for start, stop, group in spans if left < stop and right > start]
        if not overlap:
            continue
        if len(overlap) != 1 or not overlap[0][0] <= left < right <= overlap[0][1]:
            raise WeightEditInputError("mixed semantic value/syntax boundary token")
        target = prompt_length + index
        position = target - 1
        if position < 0 or base.rendered.token_ids[position + 1] != response_ids[index]:
            raise WeightEditInputError("value mask causal next-token alignment failed")
        labels[target] = response_ids[index]
        positions.append(position)
        groups.append(overlap[0][2])
        covered.update(range(left, right))
    expected = {index for start, stop, _ in spans for index in range(start, stop)}
    if not positions or covered != expected:
        raise WeightEditInputError("empty value mask or incomplete semantic value token coverage")
    return Demonstration(base.rendered, tuple(labels), tuple(positions)), groups
