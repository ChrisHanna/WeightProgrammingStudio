"""Shortcut audit of the versioned three-skill item sets. No model is loaded or scored here."""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "python" / "neural_foundry"
# Loaded by path: the set must not need the native extension, torch or the study harness.
_spec = importlib.util.spec_from_file_location("nf_gym_items_under_test", PACKAGE / "gym_items.py")
assert _spec is not None and _spec.loader is not None
g = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = g
_spec.loader.exec_module(g)

V2 = g.get_item_set(g.V2_KEY)
V1 = g.get_item_set(g.V1_KEY)
ALTERNATE_HOLDOUTS = [g.build_v2(holdout_seed=seed) for seed in (1, 2, 3)]
V2_DIGEST = "44bdf00ce0a877b919ea6d0d4e6362a7fc7329dd6ae36e2aaac786271bb867d0"


def _gym_literal(name: str) -> ast.Call:
    tree = ast.parse((PACKAGE / "pythia.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == name:
            assert isinstance(node.value, ast.Call)
            return node.value
    raise AssertionError(f"{name} not found")


def test_v1_gym_is_unchanged_and_still_registered() -> None:
    call = _gym_literal("ALGORITHMIC_THREE_SKILL_GYM")
    fields = {k.arg: k.value for k in call.keywords}
    assert ast.literal_eval(fields["gym_id"]) == "gym.reusable.algorithmic_parity_boolean_inequality.v1"
    assert ast.literal_eval(fields["base_rules"]) == g.V1_BASE_RULES
    assert ast.literal_eval(fields["holdout_rules"]) == g.V1_HOLDOUT_RULES
    assert '"algorithmic_three_skill": ALGORITHMIC_THREE_SKILL_GYM' in (PACKAGE / "studio.py").read_text(
        encoding="utf-8"
    )


def test_v1_fails_the_audit_in_the_way_reported() -> None:
    scores = g.shortcut_scores(V1)
    assert scores["boolean"]["first_operator_map"]["holdout"] == 1.0
    assert scores["boolean"]["first_operator_map"]["development"] == 1.0
    assert scores["parity"]["fit_threshold:value"]["holdout"] == 1.0
    assert scores["comparison"]["fit_threshold:second_operand"]["holdout"] == 1.0
    bodies = {c.body for c in V1.split_clusters("holdout")}
    assert "Boolean True AND True gives" not in bodies


def test_v2_is_reproducible_and_pinned() -> None:
    assert g.build_v2().digest() == V2.digest() == V2_DIGEST


def test_public_splits_do_not_depend_on_the_holdout_seed() -> None:
    other = ALTERNATE_HOLDOUTS[0]
    for split in ("development", "validation"):
        assert other.split_clusters(split) == V2.split_clusters(split)
    assert other.split_clusters("holdout") != V2.split_clusters("holdout")


@pytest.mark.parametrize("item_set", [V2, *ALTERNATE_HOLDOUTS], ids=["candidate", "seed1", "seed2", "seed3"])
def test_every_shortcut_is_within_its_stated_bound(item_set: g.ItemSet) -> None:
    scores = g.shortcut_scores(item_set)
    for task, table in scores.items():
        assert len(table) >= 10
        for name, row in table.items():
            for split in g.SPLITS:
                bound = g.policy_bound(task, name, split)
                if bound is None:
                    continue
                floor = g.policy_floor(task, name)
                views = [split] + ([f"{split}/compound"] if task == "boolean" else [])
                for view in views:
                    if view.endswith("compound") and name == "memorize_development":
                        continue
                    label = f"{task}/{name}/{view}={row[view]}"
                    assert row[view] <= bound, label
                    if floor is not None:
                        assert row[view] >= floor, label


@pytest.mark.parametrize("split", g.SPLITS)
def test_constant_answers_score_exactly_chance(split: str) -> None:
    scores = g.shortcut_scores(V2)
    for task in g.TASKS:
        assert scores[task]["constant_positive"][split] == 0.5
        assert scores[task]["constant_negative"][split] == 0.5
        assert scores[task]["fit:prefix"][split] == 0.5


def test_boolean_core_shortcuts_stop_at_the_truth_table_limit() -> None:
    """Operator maps and operand copies reach exactly 6/8 on any full two-operand table."""
    scores = g.shortcut_scores(V2)["boolean"]
    for split in g.SPLITS:
        assert scores["first_operator_map"][f"{split}/core"] == g.BOOLEAN_CORE_BOUND
        assert scores["first_operand_word"][f"{split}/core"] == g.BOOLEAN_CORE_BOUND
        assert scores["last_operand_word"][f"{split}/core"] == g.BOOLEAN_CORE_BOUND
        assert scores["first_operator_map"][f"{split}/compound"] == 0.5


def test_no_shortcut_is_perfect_outside_its_own_training_split() -> None:
    for task, table in g.shortcut_scores(V2).items():
        for name, row in table.items():
            if name == "memorize_development":
                continue
            for split in g.SPLITS:
                assert row[split] < 1.0, f"{task}/{name}/{split}"


@pytest.mark.parametrize("split", g.SPLITS)
def test_real_rule_scores_perfectly(split: str) -> None:
    assert g.accuracy(g.reference_rule(), V2.items(split)) == 1.0


@pytest.mark.parametrize("split", g.SPLITS)
def test_labels_are_exactly_balanced(split: str) -> None:
    for task in g.TASKS:
        labels = Counter(g.is_positive(c) for c in V2.split_clusters(split, task))
        assert labels[True] == labels[False] > 0
    clusters = V2.split_clusters(split, "boolean")
    for tier in ("core", "compound"):
        labels = Counter(g.is_positive(c) for c in clusters if c.tier == tier)
        assert labels[True] == labels[False] > 0
    per_signature: dict[tuple[str, ...], Counter] = {}
    for item in V2.items(split, "boolean"):
        if item.tier == "compound":
            signature = g._operators(item)
            per_signature.setdefault(signature, Counter())[g.is_positive(item)] += 1
    assert all(c[True] == c[False] for c in per_signature.values())


@pytest.mark.parametrize("split", g.SPLITS)
def test_every_split_covers_the_full_truth_tables(split: str) -> None:
    bodies = {c.body for c in V2.split_clusters(split, "boolean") if c.tier == "core"}
    assert bodies == {
        f"Boolean {a} {op} {b} gives" for op in ("AND", "OR") for a in (True, False) for b in (True, False)
    }
    assert {c.operands[0][-1] for c in V2.split_clusters(split, "parity")} == set("0123456789")
    relations = Counter()
    for cluster in V2.split_clusters(split, "comparison"):
        left, right = map(int, cluster.operands)
        relations["gt" if left > right else "lt" if left < right else "eq"] += 1
        assert g.is_positive(cluster) == (left > right)
    assert relations["gt"] and relations["lt"] and relations["eq"]


def test_splits_are_disjoint_except_the_stated_boolean_core() -> None:
    numbers = {
        split: {o for task in ("parity", "comparison") for c in V2.split_clusters(split, task) for o in c.operands}
        for split in g.SPLITS
    }
    canonical = {
        split: {
            re.sub(r"\s+", " ", " ".join(_canonical_words(c.body)))
            for c in V2.split_clusters(split, "boolean")
            if c.tier == "compound"
        }
        for split in g.SPLITS
    }
    prompts = {split: {i.prompt for i in V2.items(split)} for split in g.SPLITS}
    core = {
        split: {c.body for c in V2.split_clusters(split, "boolean") if c.tier == "core"} for split in g.SPLITS
    }
    for a, b in (("development", "validation"), ("development", "holdout"), ("validation", "holdout")):
        assert numbers[a].isdisjoint(numbers[b])
        assert canonical[a].isdisjoint(canonical[b])
        assert prompts[a].isdisjoint(prompts[b])
        assert set(g.V2_PREFIXES[a]).isdisjoint(g.V2_PREFIXES[b])
        assert core[a] == core[b]  # Eight expressions exist, so this tier cannot be disjoint.
    v1_numbers = set("0123456789")
    assert all(n not in v1_numbers for split in g.SPLITS for n in numbers[split])


def _canonical_words(body: str) -> list[str]:
    terms, operators = g._parse_boolean(body)
    words = [str(terms[0][1] != terms[0][0])]
    for operator, (negated, value) in zip(operators, terms[1:], strict=True):
        words += [operator, str(value != negated)]
    return words


def test_no_model_code_reaches_the_new_holdout() -> None:
    tree = ast.parse((PACKAGE / "gym_items.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "the module must not import from the package"
            imported.add((node.module or "").split(".")[0])
    assert imported <= set(sys.stdlib_module_names) | {"__future__"}
    for path in PACKAGE.rglob("*.py"):
        if path.name != "gym_items.py":
            assert "gym_items" not in path.read_text(encoding="utf-8"), path


def test_unknown_item_set_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown item set"):
        g.get_item_set("algorithmic_three_skill.v3")
