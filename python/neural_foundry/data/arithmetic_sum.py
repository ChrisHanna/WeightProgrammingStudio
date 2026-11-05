"""Apache-2.0 synthetic user-supplied sum gym; no model objects or callbacks."""

import math


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("a finite numeric value is required")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite value")
    return number


def _pair(inputs: object) -> tuple[float, float]:
    if not isinstance(inputs, dict) or set(inputs) != {"x1", "x2"}:
        raise ValueError("exact x1/x2 input fields required")
    x1, x2 = _number(inputs["x1"]), _number(inputs["x2"])
    if any(v != int(v) or not -8 <= v <= 8 for v in (x1, x2)):
        raise ValueError("integer arithmetic fixture domain is [-8,8]")
    return x1, x2


def evaluate(inputs: object, seed: int) -> float:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("integer seed required")
    x1, x2 = _pair(inputs)
    return x1 + x2


def oracle(inputs: object) -> float:
    x1, x2 = _pair(inputs)
    return x1 + x2


def score(inputs: object, output: object, reference: object) -> float:
    _pair(inputs)
    if isinstance(output, bool) or not isinstance(output, (int, float)):
        return 0.0
    return float(abs(_number(output) - _number(reference)) <= 1e-10)
