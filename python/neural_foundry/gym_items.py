"""Versioned item sets for the three-skill gym, audited against shortcut policies.

`algorithmic_three_skill.v1` is a read-only view of the original gym, for measuring it.
`algorithmic_three_skill.v2` is built so that the enumerated shortcuts stay near chance.
See docs/gym_item_sets.md for the measured scores and the limits of this audit.

Only shortcut policies may touch a v2 holdout. Nothing here loads or scores a model, and
v2 is deliberately not in `studio.GymRegistry`, whose `run-gym` would score its holdout.

This module uses only the standard library. `Cluster` and `Item` use the field names of
the study harness's generators, plus `tier`, so that harness can draw from these sets.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import product

SPLITS = ("development", "validation", "holdout")
TASKS = ("parity", "boolean", "comparison")
TOKENS = {
    "parity": ("Ġodd", "Ġeven"),
    "boolean": ("ĠTrue", "ĠFalse"),
    "comparison": ("Ġyes", "Ġno"),
}
V1_KEY = "algorithmic_three_skill.v1"
V2_KEY = "algorithmic_three_skill.v2"
V2_PREFIXES = {"development": ("Task:", "Rule:"), "validation": ("Note:", "Given:"), "holdout": ("Today:", "Check:")}
V2_PUBLIC_SEED = 20261106
# Committed, so this holdout is a public candidate. A sealed study passes a private seed.
V2_CANDIDATE_HOLDOUT_SEED = 7406112
PARITY_CLUSTERS = 40
COMPARISON_CLUSTERS = 48
BOOLEAN_COMPOUND_CLUSTERS = 40
MAX_ATTEMPTS = 20_000

# Bounds on the score of a shortcut on all of a split, and on the compound Boolean tier.
SIMPLE_BOUND = 0.625
# Policies fitted on development and scored there are in-sample, so they get more room.
FITTED_IN_SAMPLE_BOUND = 0.70
# Policies that run part of the real algorithm cannot reach chance. Each bound is explicit.
PARTIAL_BOUNDS = {
    "boolean": {
        "ignore_not": 0.75,
        "left_to_right": 0.80,
        "first_pair_only": 0.75,
        "last_pair_only": 0.75,
        "majority_literals_true": 0.70,
    },
    "comparison": {
        "lexicographic": 0.75,
        "digit_sum_greater": 0.75,
        "fit:digit_counts": 0.75,
        "longer_is_greater_tie_no": 0.70,
        "longer_is_greater_tie_yes": 0.70,
        "operands_differ": 0.625,
    },
    "parity": {},
}
# A two-operand truth table puts these at 6/8 whatever the item set: see docs/gym_item_sets.md.
BOOLEAN_CORE_BOUND = 0.75


@dataclass(frozen=True)
class Cluster:
    cluster_id: str
    task: str
    operands: tuple[str, ...]
    body: str
    target: str
    alternative: str
    tier: str = "main"


@dataclass(frozen=True)
class Item:
    item_id: str
    cluster_id: str
    task: str
    prompt: str
    target: str
    alternative: str
    tier: str = "main"
    body: str = ""
    prefix: str = ""


@dataclass(frozen=True)
class ItemSet:
    key: str
    public_seed: int | None
    holdout_seed: int | None
    clusters: tuple[tuple[str, tuple[Cluster, ...]], ...]
    prefixes: tuple[tuple[str, tuple[str, ...]], ...]

    def split_clusters(self, split: str, task: str | None = None) -> tuple[Cluster, ...]:
        chosen = dict(self.clusters)[split]
        return tuple(c for c in chosen if task is None or c.task == task)

    def items(self, split: str, task: str | None = None) -> tuple[Item, ...]:
        return render(self.split_clusters(split, task), dict(self.prefixes)[split])

    def digest(self) -> str:
        payload = {"key": self.key, "clusters": self.clusters, "prefixes": self.prefixes}
        payload["clusters"] = [[s, [asdict(c) for c in cs]] for s, cs in self.clusters]
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def render(clusters: Iterable[Cluster], prefixes: Sequence[str]) -> tuple[Item, ...]:
    """Render every cluster under every prefix, so that no prefix predicts a label."""
    return tuple(
        Item(
            f"{c.cluster_id}:{position}", c.cluster_id, c.task, f"{prefix} {c.body}", c.target, c.alternative,
            c.tier, c.body, prefix,
        )
        for c in clusters
        for position, prefix in enumerate(prefixes)
    )


def is_positive(item: Item | Cluster) -> bool:
    return item.target == TOKENS[item.task][0]


def evaluate_boolean(terms: Sequence[tuple[bool, bool]], operators: Sequence[str]) -> bool:
    """Evaluate (negated, value) terms with NOT above AND above OR."""
    if len(operators) != len(terms) - 1:
        raise ValueError("a Boolean expression needs one operator between each pair of terms")
    groups = [[terms[0][1] != terms[0][0]]]
    for operator, (negated, value) in zip(operators, terms[1:], strict=True):
        literal = value != negated
        if operator == "AND":
            groups[-1].append(literal)
        elif operator == "OR":
            groups.append([literal])
        else:
            raise ValueError(f"unknown operator {operator!r}")
    return any(all(group) for group in groups)


@lru_cache(maxsize=None)
def _parse_boolean(body: str) -> tuple[tuple[tuple[bool, bool], ...], tuple[str, ...]]:
    match = re.fullmatch(r"Boolean (.+) gives", body)
    if not match:
        raise ValueError(f"not a Boolean body: {body!r}")
    terms: list[tuple[bool, bool]] = []
    operators: list[str] = []
    negated = False
    for word in match[1].split():
        if word == "NOT":
            negated = not negated
        elif word in ("AND", "OR"):
            operators.append(word)
        else:
            terms.append((negated, word == "True"))
            negated = False
    return tuple(terms), tuple(operators)


def _number(item: Item) -> str:
    match = re.fullmatch(r"Number (\d+) has parity", item.body)
    if not match:
        raise ValueError(f"not a parity body: {item.body!r}")
    return match[1]


def _pair(item: Item) -> tuple[str, str]:
    match = re.fullmatch(r"Relation (\d+) > (\d+) is", item.body)
    if not match:
        raise ValueError(f"not a comparison body: {item.body!r}")
    return match[1], match[2]


def _terms(item: Item) -> tuple[tuple[bool, bool], ...]:
    return _parse_boolean(item.body)[0]


def _operators(item: Item) -> tuple[str, ...]:
    return _parse_boolean(item.body)[1]


def _literals(item: Item) -> list[bool]:
    return [value != negated for negated, value in _terms(item)]


def _digit_sum(text: str) -> int:
    return sum(int(c) for c in text)


def _longer(item: Item, tie: bool) -> bool:
    a, b = _pair(item)
    return tie if len(a) == len(b) else len(a) > len(b)


def _left_to_right(item: Item) -> bool:
    literals = _literals(item)
    value = literals[0]
    for operator, literal in zip(_operators(item), literals[1:], strict=True):
        value = (value and literal) if operator == "AND" else (value or literal)
    return value


def _pair_only(item: Item, start: int) -> bool:
    terms = _terms(item)
    operators = _operators(item)
    index = 0 if start == 0 else len(operators) - 1
    return evaluate_boolean(terms[index : index + 2], operators[index : index + 1])


def _true_rule(item: Item) -> bool:
    if item.task == "parity":
        return int(_number(item)) % 2 == 1
    if item.task == "boolean":
        terms, operators = _parse_boolean(item.body)
        return evaluate_boolean(terms, operators)
    a, b = _pair(item)
    return int(a) > int(b)


Rule = Callable[[Item], bool]

_COMMON_RULES: dict[str, Rule] = {
    "constant_positive": lambda _: True,
    "constant_negative": lambda _: False,
    "body_chars_odd": lambda i: len(i.body) % 2 == 1,
    "body_words_odd": lambda i: len(i.body.split()) % 2 == 1,
}
_RULES: dict[str, dict[str, Rule]] = {
    "parity": {
        "digit_count_odd": lambda i: len(_number(i)) % 2 == 1,
        "first_digit_odd": lambda i: int(_number(i)[0]) % 2 == 1,
        "tens_digit_odd": lambda i: len(_number(i)) > 1 and int(_number(i)[-2]) % 2 == 1,
        "digit_sum_odd_excluding_last": lambda i: _digit_sum(_number(i)[:-1]) % 2 == 1,
    },
    "boolean": {
        "first_operator_map": lambda i: _operators(i)[0] == "OR",
        "last_operator_map": lambda i: _operators(i)[-1] == "OR",
        "any_or": lambda i: "OR" in _operators(i),
        "first_operand_word": lambda i: _terms(i)[0][1],
        "last_operand_word": lambda i: _terms(i)[-1][1],
        "first_literal": lambda i: _literals(i)[0],
        "last_literal": lambda i: _literals(i)[-1],
        "any_true_word": lambda i: any(v for _, v in _terms(i)),
        "all_true_word": lambda i: all(v for _, v in _terms(i)),
        "majority_literals_true": lambda i: 2 * sum(_literals(i)) > len(_literals(i)),
        "xor_of_literals": lambda i: sum(_literals(i)) % 2 == 1,
        "has_not": lambda i: any(n for n, _ in _terms(i)),
        "ignore_not": lambda i: evaluate_boolean([(False, v) for _, v in _terms(i)], _operators(i)),
        "left_to_right": _left_to_right,
        "first_pair_only": lambda i: _pair_only(i, 0),
        "last_pair_only": lambda i: _pair_only(i, -1),
    },
    "comparison": {
        "first_operand_odd": lambda i: int(_pair(i)[0]) % 2 == 1,
        "second_operand_even": lambda i: int(_pair(i)[1]) % 2 == 0,
        "operand_sum_odd": lambda i: (int(_pair(i)[0]) + int(_pair(i)[1])) % 2 == 1,
        "first_digit_greater": lambda i: _pair(i)[0][0] > _pair(i)[1][0],
        "last_digit_greater": lambda i: _pair(i)[0][-1] > _pair(i)[1][-1],
        "digit_sum_greater": lambda i: _digit_sum(_pair(i)[0]) > _digit_sum(_pair(i)[1]),
        "lexicographic": lambda i: _pair(i)[0] > _pair(i)[1],
        "longer_is_greater_tie_no": lambda i: _longer(i, False),
        "longer_is_greater_tie_yes": lambda i: _longer(i, True),
        "operands_differ": lambda i: _pair(i)[0] != _pair(i)[1],
    },
}
_FEATURES: dict[str, dict[str, Callable[[Item], object]]] = {
    "parity": {
        "digit_count": lambda i: len(_number(i)),
        "first_digit": lambda i: _number(i)[0],
        "tens_digit": lambda i: _number(i)[-2:-1],
    },
    "boolean": {
        "operator_signature": _operators,
        "first_operator": lambda i: _operators(i)[0],
        "last_operator": lambda i: _operators(i)[-1],
        "arity": lambda i: len(_terms(i)),
        "has_not": lambda i: any(n for n, _ in _terms(i)),
        "first_word": lambda i: _terms(i)[0][1],
        "last_word": lambda i: _terms(i)[-1][1],
        "true_word_count": lambda i: sum(v for _, v in _terms(i)),
    },
    "comparison": {
        "digit_counts": lambda i: (len(_pair(i)[0]), len(_pair(i)[1])),
        "first_digit_of_first": lambda i: _pair(i)[0][0],
        "first_digit_of_second": lambda i: _pair(i)[1][0],
        "last_digit_of_first": lambda i: _pair(i)[0][-1],
    },
}
_NUMERIC: dict[str, dict[str, Callable[[Item], float]]] = {
    "parity": {"value": lambda i: float(_number(i))},
    "boolean": {},
    "comparison": {
        "first_operand": lambda i: float(_pair(i)[0]),
        "second_operand": lambda i: float(_pair(i)[1]),
    },
}


def _fit_feature(feature: Callable[[Item], object], development: Sequence[Item]) -> Rule:
    votes: dict[object, int] = Counter()
    for item in development:
        votes[feature(item)] += 1 if is_positive(item) else -1
    return lambda item: votes.get(feature(item), 0) > 0


def _fit_threshold(value: Callable[[Item], float], development: Sequence[Item]) -> Rule:
    best = (-1, 0.0, True)
    values = sorted({value(i) for i in development})
    for threshold in [(low + high) / 2 for low, high in zip(values, values[1:], strict=False)] or values:
        for above in (True, False):
            hits = sum(((value(i) >= threshold) == above) == is_positive(i) for i in development)
            if hits > best[0]:
                best = (hits, threshold, above)
    _, chosen, direction = best
    return lambda item: (value(item) >= chosen) == direction


def shortcut_policies(task: str, development: Sequence[Item]) -> dict[str, Rule]:
    """Fixed rules, plus rules fitted on the development items, which never see other splits."""
    policies = {**_COMMON_RULES, **_RULES[task]}
    features = {**_FEATURES[task], "prefix": lambda i: i.prefix}
    for name, feature in features.items():
        policies[f"fit:{name}"] = _fit_feature(feature, development)
    for name, value in _NUMERIC[task].items():
        policies[f"fit_threshold:{name}"] = _fit_threshold(value, development)
    answers = {i.body: is_positive(i) for i in development}
    policies["memorize_development"] = lambda i: answers.get(i.body, False)
    return policies


def accuracy(rule: Rule, items: Sequence[Item]) -> float:
    if not items:
        raise ValueError("cannot score an empty split")
    return sum(rule(i) == is_positive(i) for i in items) / len(items)


def policy_bound(task: str, name: str, split: str) -> float | None:
    """The stated bound for a policy on a whole split, or None where a bound would be meaningless."""
    if name == "memorize_development" and split == "development":
        return None  # A lookup table is perfect on its own table by definition.
    if name in PARTIAL_BOUNDS[task]:
        return PARTIAL_BOUNDS[task][name]
    if name.startswith("fit") and split == "development":
        return FITTED_IN_SAMPLE_BOUND
    return SIMPLE_BOUND


def policy_floor(task: str, name: str) -> float | None:
    """A fixed simple rule far below chance is a shortcut too, because its inverse scores high."""
    if name in PARTIAL_BOUNDS[task] or name.startswith("fit") or name == "memorize_development":
        return None
    return 1.0 - SIMPLE_BOUND


def shortcut_scores(item_set: ItemSet) -> dict[str, dict[str, dict[str, float]]]:
    """Return scores as {task: {policy: {split: accuracy}}}. Boolean adds core and compound views."""
    report: dict[str, dict[str, dict[str, float]]] = {}
    for task in TASKS:
        development = item_set.items("development", task)
        table: dict[str, dict[str, float]] = {}
        for name, rule in shortcut_policies(task, development).items():
            row: dict[str, float] = {}
            for split in SPLITS:
                items = item_set.items(split, task)
                row[split] = accuracy(rule, items)
                if task == "boolean":
                    for tier in ("core", "compound"):
                        subset = [i for i in items if i.tier == tier]
                        if subset:
                            row[f"{split}/{tier}"] = accuracy(rule, subset)
            table[name] = row
        report[task] = table
    return report


def _within_bounds(task: str, split: str, clusters: Sequence[Cluster], prefixes: Sequence[str],
                   development: Sequence[Item]) -> bool:
    items = render(clusters, prefixes)
    for name, rule in shortcut_policies(task, development).items():
        bound = policy_bound(task, name, split)
        if bound is None:
            continue
        floor = policy_floor(task, name)
        compound = [i for i in items if i.tier == "compound"]
        for subset in (items, compound):
            if not subset or (subset is compound and name == "memorize_development"):
                continue
            score = accuracy(rule, subset)
            if score > bound or (floor is not None and score < floor):
                return False
    return True


def _cluster(task: str, tag: str, index: int, operands: tuple[str, ...], body: str, positive: bool,
             tier: str = "main") -> Cluster:
    yes, no = TOKENS[task]
    target, alternative = (yes, no) if positive else (no, yes)
    return Cluster(f"{task}:{tag}:{index}", task, operands, body, target, alternative, tier)


def _draw_number(rng: random.Random, digits: int, used: set[str], *, last: int | None = None,
                 lead: int | None = None) -> str:
    for _ in range(100_000):
        first = lead if lead is not None else rng.randint(1, 9)
        middle = "".join(str(rng.randint(0, 9)) for _ in range(digits - 2))
        final = last if last is not None else rng.randint(0, 9)
        text = f"{first}{middle}{final}"
        if text not in used:
            used.add(text)
            return text
    raise ValueError("number space is exhausted for this draw")


def _draw_parity(rng: random.Random, tag: str, exclude: Collection[str]) -> tuple[list[Cluster], set[str]]:
    used = set(exclude)
    per_digit = PARITY_CLUSTERS // 10
    plans = [(last, 2 + (repeat + last // 2) % 3) for last in range(10) for repeat in range(per_digit)]
    rng.shuffle(plans)
    clusters = []
    for index, (last, digits) in enumerate(plans):
        number = _draw_number(rng, digits, used, last=last)
        clusters.append(_cluster("parity", tag, index, (number,), f"Number {number} has parity", last % 2 == 1))
    return clusters, used - set(exclude)


def _comparison_pair(rng: random.Random, kind: str, used: set[str]) -> tuple[str, str]:
    """Return (smaller, larger). Kinds defeat different digit-level shortcuts."""
    for _ in range(100_000):
        if kind == "same_lead":
            digits = rng.randint(3, 5)
            lead = rng.randint(1, 9)
            a = _draw_number(rng, digits, used, lead=lead)
            b = _draw_number(rng, digits, used, lead=lead)
        elif kind == "lead_contradicts_length":
            short, long = sorted(rng.sample((3, 4, 5), 2))
            small_lead = rng.randint(2, 9)
            a = _draw_number(rng, short, used, lead=small_lead)
            b = _draw_number(rng, long, used, lead=rng.randint(1, small_lead - 1))
        else:
            digits = rng.randint(3, 5)
            low, high = sorted(rng.sample(range(1, 10), 2))
            a = _draw_number(rng, digits, used, lead=low)
            b = _draw_number(rng, digits, used, lead=high)
        if int(a) != int(b):
            return (a, b) if int(a) < int(b) else (b, a)
    raise ValueError("number space is exhausted for this draw")


def _draw_comparison(rng: random.Random, tag: str, exclude: Collection[str]) -> tuple[list[Cluster], set[str]]:
    used = set(exclude)
    kinds = ("same_lead", "lead_contradicts_length", "same_length_lead_differs")
    half = COMPARISON_CLUSTERS // 2
    equal = half // 4
    plans = [("gt", kinds[n % 3]) for n in range(half)]
    plans += [("lt", kinds[n % 3]) for n in range(half - equal)]
    plans += [("eq", "")] * equal
    rng.shuffle(plans)
    clusters = []
    for index, (relation, kind) in enumerate(plans):
        if relation == "eq":
            left = right = _draw_number(rng, rng.randint(3, 5), used)
        else:
            small, large = _comparison_pair(rng, kind, used)
            left, right = (large, small) if relation == "gt" else (small, large)
        body = f"Relation {left} > {right} is"
        clusters.append(_cluster("comparison", tag, index, (left, right), body, relation == "gt"))
    return clusters, used - set(exclude)


def _boolean_words(terms: Sequence[tuple[bool, bool]], operators: Sequence[str]) -> str:
    words = ["NOT " * negated + str(value) for negated, value in terms[:1]]
    for operator, (negated, value) in zip(operators, terms[1:], strict=True):
        words += [operator, "NOT " * negated + str(value)]
    return " ".join(words)


def _mixed_signatures(arity: int) -> list[tuple[str, ...]]:
    return [s for s in product(("AND", "OR"), repeat=arity - 1) if len(set(s)) == 2]


def _forms_by_label(signature: tuple[str, ...], split: str | None = None) -> dict[bool, list[tuple[bool, ...]]]:
    """List the forms of a signature, or the third of them reserved for one split.

    The fixed partition keeps the splits disjoint and stops the earlier splits starving the holdout.
    """
    forms: dict[bool, list[tuple[bool, ...]]] = {True: [], False: []}
    for literals in product((False, True), repeat=len(signature) + 1):
        forms[evaluate_boolean([(False, v) for v in literals], signature)].append(literals)
    if split is not None:
        for label, chosen in forms.items():
            random.Random(f"{V2_KEY}|partition|{signature}|{label}").shuffle(chosen)
            forms[label] = chosen[SPLITS.index(split) :: len(SPLITS)]
    return forms


def _canonical(literals: Sequence[bool], signature: Sequence[str]) -> str:
    """The expression with NOTs resolved, so that NOT False and True count as the same exposure."""
    return _boolean_words([(False, v) for v in literals], signature)


def _pick_form(rng: random.Random, forms: Sequence[tuple[bool, ...]], signature: tuple[str, ...], label: bool,
               hits: Counter, picked: int) -> tuple[bool, ...]:
    """Pick the form that keeps the literal-level shortcuts closest to chance so far.

    AND and OR are monotone, so true literals predict a true result in a random draw.
    """

    def guesses(literals: tuple[bool, ...]) -> dict[str, bool]:
        item = Item("", "", "boolean", "", "", "", body=f"Boolean {_canonical(literals, signature)} gives")
        return {
            "first": literals[0],
            "last": literals[-1],
            "majority": 2 * sum(literals) > len(literals),
            "xor": sum(literals) % 2 == 1,
            "left_to_right": _left_to_right(item),
            "first_pair": _pair_only(item, 0),
            "last_pair": _pair_only(item, -1),
        }

    def imbalance(literals: tuple[bool, ...]) -> tuple[float, float]:
        total = sum(
            (hits[name] + (guess == label) - (picked + 1) / 2) ** 2 for name, guess in guesses(literals).items()
        )
        return total, rng.random()

    chosen = min(forms, key=imbalance)
    hits.update(name for name, guess in guesses(chosen).items() if guess == label)
    return chosen


def _boolean_core(tag: str) -> list[Cluster]:
    clusters = []
    rows = product(("AND", "OR"), (True, False), (True, False))
    for index, (operator, a, b) in enumerate(rows):
        expression = f"{a} {operator} {b}"
        value = evaluate_boolean([(False, a), (False, b)], (operator,))
        clusters.append(
            _cluster("boolean", f"{tag}:core", index, (expression,), f"Boolean {expression} gives", value, "core")
        )
    return clusters


def _draw_boolean(rng: random.Random, tag: str, exclude: Collection[str]) -> tuple[list[Cluster], set[str]]:
    """Draw a label-balanced pair of expressions per mixed-operator signature.

    Every signature gives True and False equally often here, so operators predict nothing.
    Same-operator chains are left out because their rare label cannot appear in all splits.
    """
    pairs_wanted = BOOLEAN_COMPOUND_CLUSTERS // 2
    plan: list[tuple[str, ...]] = []
    for arity in (3, 4):
        plan += _mixed_signatures(arity)
    arity_five = _mixed_signatures(5)
    rng.shuffle(arity_five)
    plan += arity_five[: pairs_wanted - len(plan)]
    rng.shuffle(plan)
    clusters = []
    used: set[str] = set()
    hits: Counter = Counter()
    for signature in plan:
        forms = _forms_by_label(signature, tag)
        for label in (True, False):
            literals = _pick_form(rng, forms[label], signature, label, hits, len(clusters))
            used.add(_canonical(literals, signature))
            terms = []
            for literal in literals:
                negated = rng.random() < 0.4
                terms.append((negated, literal != negated))
            expression = _boolean_words(terms, signature)
            index = len(clusters)
            clusters.append(
                _cluster("boolean", f"{tag}:compound", index, (expression,), f"Boolean {expression} gives", label,
                         "compound")
            )
    rng.shuffle(clusters)
    return _boolean_core(tag) + clusters, used


_DRAW = {"parity": _draw_parity, "boolean": _draw_boolean, "comparison": _draw_comparison}


def _draw_split(task: str, seed: int, split: str, exclude: Collection[str],
                development: Sequence[Item]) -> tuple[list[Cluster], set[str], list[Item]]:
    """Draw candidates from the seed until one passes. Only shortcut policies see the candidates."""
    rng = random.Random(f"{V2_KEY}|{task}|{split}|{seed}")
    prefixes = V2_PREFIXES[split]
    for _ in range(MAX_ATTEMPTS):
        clusters, used = _DRAW[task](rng, split, exclude)
        items = list(render(clusters, prefixes))
        reference = development or items
        if _within_bounds(task, split, clusters, prefixes, reference):
            return clusters, used, items
    raise ValueError(f"no {task} {split} draw met the shortcut bounds for seed {seed}")


def build_v2(public_seed: int = V2_PUBLIC_SEED, holdout_seed: int = V2_CANDIDATE_HOLDOUT_SEED) -> ItemSet:
    """Build v2. Development and validation depend only on the public seed."""
    per_split: dict[str, list[Cluster]] = {split: [] for split in SPLITS}
    development: dict[str, list[Item]] = {}
    exposed: set[str] = set()  # Numbers are not reused across skills, so a later split excludes them all.
    for split in SPLITS:
        seed = holdout_seed if split == "holdout" else public_seed
        for task in TASKS:
            clusters, used, items = _draw_split(task, seed, split, exposed, development.get(task, []))
            per_split[split] += clusters
            exposed |= used
            if split == "development":
                development[task] = items
    return ItemSet(
        key=V2_KEY,
        public_seed=public_seed,
        holdout_seed=holdout_seed,
        clusters=tuple((split, tuple(per_split[split])) for split in SPLITS),
        prefixes=tuple((split, V2_PREFIXES[split]) for split in SPLITS),
    )


# A copy of ALGORITHMIC_THREE_SKILL_GYM, kept here to avoid importing torch. A test compares them.
V1_BASE_RULES = (
    ("Number 9 has parity", "Ġodd"),
    ("Number 4 has parity", "Ġeven"),
    ("Boolean False AND True gives", "ĠFalse"),
    ("Boolean False OR True gives", "ĠTrue"),
    ("Relation 9 > 2 is", "Ġyes"),
    ("Relation 2 > 6 is", "Ġno"),
)
V1_HOLDOUT_RULES = (
    ("Number 7 has parity", "Ġodd"),
    ("Number 2 has parity", "Ġeven"),
    ("Boolean True AND False gives", "ĠFalse"),
    ("Boolean True OR False gives", "ĠTrue"),
    ("Relation 7 > 3 is", "Ġyes"),
    ("Relation 1 > 8 is", "Ġno"),
)
_V1_TASK = {"Number": "parity", "Boolean": "boolean", "Relation": "comparison"}


def _v1_clusters(rules: Sequence[tuple[str, str]], tag: str) -> tuple[Cluster, ...]:
    clusters = []
    for index, (body, target) in enumerate(rules):
        task = _V1_TASK[body.split()[0]]
        yes, no = TOKENS[task]
        alternative = no if target == yes else yes
        clusters.append(Cluster(f"{task}:v1:{tag}:{index}", task, (body,), body, target, alternative, "core"))
    return tuple(clusters)


def build_v1() -> ItemSet:
    """A view of the original gym. Validation repeats the development expressions under a new prefix."""
    return ItemSet(
        key=V1_KEY,
        public_seed=None,
        holdout_seed=None,
        clusters=(
            ("development", _v1_clusters(V1_BASE_RULES, "development")),
            ("validation", _v1_clusters(V1_BASE_RULES, "validation")),
            ("holdout", _v1_clusters(V1_HOLDOUT_RULES, "holdout")),
        ),
        # The gym gives each holdout rule one prefix in turn; policies here ignore that pairing
        # apart from fit:prefix, which cannot transfer because the holdout prefixes are unseen.
        prefixes=(
            ("development", ("Task:", "Rule:")),
            ("validation", ("Note:",)),
            ("holdout", ("Holdout:",)),
        ),
    )


@lru_cache(maxsize=None)
def get_item_set(key: str) -> ItemSet:
    if key == V1_KEY:
        return build_v1()
    if key == V2_KEY:
        return build_v2()
    raise ValueError(f"unknown item set {key!r}; available: {[V1_KEY, V2_KEY]}")


def reference_rule() -> Rule:
    """The real rule for each skill. It must score 1.0, which shows that the set is solvable."""
    return _true_rule


if __name__ == "__main__":
    print(json.dumps({key: shortcut_scores(get_item_set(key)) for key in (V1_KEY, V2_KEY)}, indent=1, default=str))
