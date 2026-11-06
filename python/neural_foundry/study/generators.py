"""Seeded item generators for parity, Boolean logic, comparison and retention.

This code is public: the secrecy of a holdout rests on its seed, and on the sampling
space being too large for a guessed rule to matter. A cluster is one underlying
expression; each is rendered once per prefix, so items in a cluster are correlated.

Parity and comparison have real operands (numbers), and no number is used twice in a
draw. Boolean operands are only True and False, so their disjointness is necessarily
at the level of whole expressions.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

TOKENS = {
    "parity": ("Ġodd", "Ġeven"),
    "boolean": ("ĠTrue", "ĠFalse"),
    "comparison": ("Ġyes", "Ġno"),
}
RETENTION_TASK = "retention"
_BOOLEAN_ARITIES = (3, 4, 5)
_RETENTION_SUBJECTS = (
    "old sailor", "small village", "young doctor", "tall building", "new teacher", "quiet river", "local baker",
    "red bicycle", "busy market", "famous painter", "wooden bridge", "curious child", "distant planet", "stone wall",
    "green garden", "retired engineer", "morning train", "city library", "winter storm", "gentle horse",
)
_RETENTION_PREDICATES = (
    "was known for", "stood near the", "had always been", "looked across the", "seemed to be", "was found in the",
    "could be seen from", "belonged to the", "moved toward the", "appeared in the", "became part of", "waited beside the",
    "was built by", "turned away from", "started with the", "ended up in", "was named after", "came back to",
    "remained close to", "depended on the",
)


@dataclass(frozen=True)
class Cluster:
    cluster_id: str
    task: str
    operands: tuple[str, ...]  # the strings that count as exposure of this cluster
    body: str
    target: str
    alternative: str


@dataclass(frozen=True)
class Item:
    item_id: str
    cluster_id: str
    task: str
    prompt: str
    target: str
    alternative: str


def evaluate_boolean(terms: Sequence[tuple[bool, bool]], operators: Sequence[str]) -> bool:
    """Evaluate (negated, value) terms with NOT above AND above OR. The tests cover precedence."""
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


def _labels(n: int, rng: random.Random) -> list[bool]:
    if n % 2:
        raise ValueError("cluster count must be even so that labels balance exactly")
    labels = [True, False] * (n // 2)
    rng.shuffle(labels)
    return labels


def _parity(n: int, rng: random.Random, domain: tuple[int, int], exclude: Collection[str], tag: str) -> list[Cluster]:
    odd, even = TOKENS["parity"]
    pool = [v for v in range(domain[0], domain[1] + 1) if str(v) not in exclude]
    rng.shuffle(pool)
    odds = [v for v in pool if v % 2]
    evens = [v for v in pool if not v % 2]
    if len(odds) < n // 2 or len(evens) < n // 2:
        raise ValueError("number domain is too small for this draw after exclusions")
    clusters = []
    for index, is_odd in enumerate(_labels(n, rng)):
        value = odds.pop() if is_odd else evens.pop()
        target, alternative = (odd, even) if is_odd else (even, odd)
        clusters.append(
            Cluster(f"parity:{tag}:{index}", "parity", (str(value),), f"Number {value} has parity", target, alternative)
        )
    return clusters


def _comparison(
    n: int, rng: random.Random, domain: tuple[int, int], exclude: Collection[str], tag: str
) -> list[Cluster]:
    yes, no = TOKENS["comparison"]
    pool = [v for v in range(domain[0], domain[1] + 1) if str(v) not in exclude]
    if len(pool) < 2 * n:
        raise ValueError("number domain is too small for this draw after exclusions")
    rng.shuffle(pool)
    clusters = []
    for index, is_greater in enumerate(_labels(n, rng)):
        low, high = sorted((pool.pop(), pool.pop()))
        left, right = (high, low) if is_greater else (low, high)
        target, alternative = (yes, no) if is_greater else (no, yes)
        clusters.append(
            Cluster(
                f"comparison:{tag}:{index}",
                "comparison",
                (str(left), str(right)),
                f"Relation {left} > {right} is",
                target,
                alternative,
            )
        )
    return clusters


def _boolean_expression(rng: random.Random) -> tuple[str, bool]:
    arity = rng.choice(_BOOLEAN_ARITIES)
    terms = [(rng.random() < 0.25, rng.random() < 0.5) for _ in range(arity)]
    operators = [rng.choice(("AND", "OR")) for _ in range(arity - 1)]
    words = ["NOT " * terms[0][0] + str(terms[0][1])]
    for operator, (negated, value) in zip(operators, terms[1:], strict=True):
        words += [operator, "NOT " * negated + str(value)]
    return " ".join(words), evaluate_boolean(terms, operators)


def _boolean(n: int, rng: random.Random, exclude: Collection[str], tag: str) -> list[Cluster]:
    true_token, false_token = TOKENS["boolean"]
    clusters: list[Cluster] = []
    seen: set[str] = set()
    for index, wanted in enumerate(_labels(n, rng)):
        for _ in range(200_000):
            expression, value = _boolean_expression(rng)
            if value == wanted and expression not in seen and expression not in exclude:
                break
        else:
            raise ValueError("Boolean expression space is exhausted for this draw")
        seen.add(expression)
        target, alternative = (true_token, false_token) if value else (false_token, true_token)
        clusters.append(
            Cluster(f"boolean:{tag}:{index}", "boolean", (expression,), f"Boolean {expression} gives", target, alternative)
        )
    return clusters


def draw_clusters(
    task: str,
    seed: int,
    n: int,
    *,
    domain: tuple[int, int] = (0, 9999),
    exclude: Collection[str] = frozenset(),
    tag: str = "dev",
) -> list[Cluster]:
    """Draw n label-balanced clusters. The same arguments always give the same clusters."""
    rng = random.Random(f"{task}|{seed}")
    if task == "parity":
        return _parity(n, rng, domain, exclude, tag)
    if task == "boolean":
        return _boolean(n, rng, exclude, tag)
    if task == "comparison":
        return _comparison(n, rng, domain, exclude, tag)
    raise ValueError(f"unknown task {task!r}")


def render(clusters: Iterable[Cluster], prefixes: Sequence[str]) -> list[Item]:
    return [
        Item(f"{c.cluster_id}:{position}", c.cluster_id, c.task, f"{prefix} {c.body}", c.target, c.alternative)
        for c in clusters
        for position, prefix in enumerate(prefixes)
    ]


def draw_retention(seed: int, n: int, *, exclude: Collection[str] = frozenset(), tag: str = "dev") -> list[Item]:
    """Draw neutral prompts for retention. Each prompt is its own cluster, with no target."""
    rng = random.Random(f"{RETENTION_TASK}|{seed}")
    pool = [f"The {s} {p}" for s in _RETENTION_SUBJECTS for p in _RETENTION_PREDICATES]
    pool = [prompt for prompt in pool if prompt not in exclude]
    if len(pool) < n:
        raise ValueError("retention corpus is too small for this draw after exclusions")
    chosen = rng.sample(pool, n)
    return [
        Item(f"{RETENTION_TASK}:{tag}:{i}", f"{RETENTION_TASK}:{tag}:{i}", RETENTION_TASK, prompt, "", "")
        for i, prompt in enumerate(chosen)
    ]


class GeneratorContractError(ValueError):
    """An item generator returned something the statistics or the sealing can't rely on."""


class ItemGenerator(Protocol):
    """What the study needs from a source of items. A plan selects one by name.

    `confirmatory_ready` is False until a generator's items are fit to support a scientific claim;
    the custodian refuses to freeze a confirmatory study on a generator that says False.
    `shared_operand_tasks` are tasks whose operands share one pool, so a number exposed for one
    is excluded from all of them.
    """

    name: str
    confirmatory_ready: bool
    shared_operand_tasks: tuple[str, ...]

    def draw_clusters(
        self, task: str, seed: int, n: int, *, domain: tuple[int, int], exclude: Collection[str], tag: str
    ) -> list[Cluster]: ...

    def draw_retention(self, seed: int, n: int, *, exclude: Collection[str], tag: str) -> list[Item]: ...


class TemplatedGeneratorV1:
    """The original one-sentence templates. Their holdout items have a known shortcut: harness runs only."""

    name = "templated_v1"
    confirmatory_ready = False
    shared_operand_tasks = ("parity", "comparison")

    def draw_clusters(
        self, task: str, seed: int, n: int, *, domain: tuple[int, int], exclude: Collection[str], tag: str
    ) -> list[Cluster]:
        return draw_clusters(task, seed, n, domain=domain, exclude=exclude, tag=tag)

    def draw_retention(self, seed: int, n: int, *, exclude: Collection[str], tag: str) -> list[Item]:
        return draw_retention(seed, n, exclude=exclude, tag=tag)


# A fixed list, like the evaluators: a plan can't name code to import. The custodian is its own process,
# so a replacement generator must be added to this module, not registered at runtime by a proposer.
_GENERATORS: dict[str, Callable[[], ItemGenerator]] = {"templated_v1": TemplatedGeneratorV1}
DEFAULT_GENERATOR = "templated_v1"


def get_item_generator(name: str) -> ItemGenerator:
    try:
        return _GENERATORS[name]()
    except KeyError:
        raise ValueError(f"unknown item generator {name!r}") from None


def register_item_generator(name: str, factory: Callable[[], ItemGenerator]) -> None:
    """Add a generator to this process only. For tests; see the note on `_GENERATORS`."""
    if name in _GENERATORS:
        raise ValueError(f"item generator {name!r} is already registered")
    _GENERATORS[name] = factory


def check_clusters(clusters: Sequence[Cluster], task: str, n: int, exclude: Collection[str], tag: str) -> None:
    """Hold any generator to what the statistics assume: balanced labels, correct IDs, no exposed operand."""
    if len(clusters) != n or len({c.cluster_id for c in clusters}) != n:
        raise GeneratorContractError(f"{task}: expected {n} distinct clusters")
    for cluster in clusters:
        if cluster.task != task or not cluster.cluster_id.startswith(f"{task}:{tag}:"):
            raise GeneratorContractError(f"{task}: cluster IDs must look like {task}:{tag}:<index>")
        if not cluster.operands or cluster.target == cluster.alternative:
            raise GeneratorContractError(f"{task}: clusters need operands and two different answers")
        if set(cluster.operands) & set(exclude):
            raise GeneratorContractError(f"{task}: a generator returned an excluded operand")
    if set(Counter(c.target for c in clusters).values()) != {n // 2}:
        raise GeneratorContractError(f"{task}: labels must balance exactly, so a constant answer scores 0.5")
