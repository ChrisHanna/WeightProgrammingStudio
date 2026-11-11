"""Allowlisted aggregate-only audit plugins; no gym-specific custodian branches."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from .generators import Item


class AuditPlugin(Protocol):
    def validate(self, config: Mapping[str, Any]) -> None: ...

    def audit(self, training: Sequence[Mapping[str, Any]], sealed: Sequence[Item],
              config: Mapping[str, Any]) -> dict[str, Any]: ...


def _text(row: Mapping[str, Any], path: Sequence[str]) -> str:
    value: Any = row
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError("training text path differs from audit configuration")
        value = value[key]
    if not isinstance(value, str) or not value or len(value) > 8192:
        raise ValueError("audit requires bounded nonempty training text")
    return value


def _grams(text: str, size: int) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.casefold())
    return {tuple(words[index:index + size]) for index in range(len(words) - size + 1)}


class TextOverlapAudit:
    def validate(self, config: Mapping[str, Any]) -> None:
        if set(config) != {"text_path", "thresholds"}:
            raise ValueError("unknown audit configuration fields")
        path, thresholds = config["text_path"], config["thresholds"]
        if (not isinstance(path, list) or not 1 <= len(path) <= 8
            or any(not isinstance(key, str) or not key or len(key) > 64 for key in path)):
            raise ValueError("invalid training text path")
        if (not isinstance(thresholds, Mapping)
            or set(thresholds) != {"exact_input_matches", "distinct_shared_5grams"}
            or any(type(value) is not int or not 0 <= value <= 4096 for value in thresholds.values())):
            raise ValueError("explicit bounded overlap thresholds required")

    def audit(self, training: Sequence[Mapping[str, Any]], sealed: Sequence[Item],
              config: Mapping[str, Any]) -> dict[str, Any]:
        self.validate(config)
        if not training or len(training) > 4096 or not sealed:
            raise ValueError("audit row count outside supported bounds")
        texts = [_text(row, config["text_path"]) for row in training]
        held_texts = {item.prompt for item in sealed}
        counts = {"exact_input_matches": sum(text in held_texts for text in texts)}
        for size in (3, 4, 5):
            held_grams = set().union(*(_grams(text, size) for text in held_texts))
            shared = set().union(*(_grams(text, size) & held_grams for text in texts))
            counts[f"distinct_shared_{size}grams"] = len(shared)
        thresholds = dict(config["thresholds"])
        checks = {name: counts[name] <= value for name, value in thresholds.items()}
        return {"training_rows": len(training), "sealed_rows": len(sealed),
                "counts": counts, "thresholds": thresholds, "checks": checks,
                "passed": all(checks.values())}


AUDIT_PLUGINS: Mapping[str, AuditPlugin] = {"text_overlap_v1": TextOverlapAudit()}


def get_audit_plugin(name: str) -> AuditPlugin:
    if name not in AUDIT_PLUGINS:
        raise ValueError("unknown audit registry name")
    return AUDIT_PLUGINS[name]


def validate_aggregate(result: Mapping[str, Any]) -> None:
    if set(result) != {"training_rows", "sealed_rows", "counts", "thresholds", "checks", "passed"}:
        raise ValueError("audit plugin returned an unsupported release schema")
    for name in ("training_rows", "sealed_rows"):
        if type(result[name]) is not int or not 1 <= result[name] <= 1_000_000:
            raise ValueError("invalid audit aggregate size")
    if (set(result["counts"]) != {"exact_input_matches", "distinct_shared_3grams",
                                  "distinct_shared_4grams", "distinct_shared_5grams"}
        or any(type(value) is not int or not 0 <= value <= 1_000_000
               for value in result["counts"].values())
        or set(result["thresholds"]) != {"exact_input_matches", "distinct_shared_5grams"}
        or any(type(value) is not int or not 0 <= value <= 4096
               for value in result["thresholds"].values())
        or set(result["checks"]) != set(result["thresholds"])
        or any(type(value) is not bool for value in result["checks"].values())
        or type(result["passed"]) is not bool):
        raise ValueError("invalid audit aggregate fields")
    expected = {name: result["counts"][name] <= value for name, value in result["thresholds"].items()}
    if result["checks"] != expected or result["passed"] != all(expected.values()):
        raise ValueError("audit result differs from the frozen threshold checks")
