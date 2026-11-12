from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import product
from pathlib import Path

import pytest
from neural_foundry.gym_job import GymJobError
from neural_foundry.json_confirmation_panel import confirmation_rows, rendered_reference
from neural_foundry.json_factorial_round import build_spec
from neural_foundry.json_factorial_training import FACTORS, factorial_authority_rows
from neural_foundry.proof_registry import builtin_registry


def _record(text):
    if text.startswith("{"):
        return json.loads(text)
    match = re.fullmatch(r"recipient (\w+); merchandise (\w+); amount (\d+)", text)
    assert match
    customer, product, quantity = match.groups()
    return {"customer": customer, "product": product, "quantity": int(quantity)}


def test_factorial_rows_independently_cross_claim_position_shape_and_preserve_facts():
    rows = factorial_authority_rows(seed=20261118)
    assert rows == factorial_authority_rows(seed=20261118)
    assert rows != factorial_authority_rows(seed=20261119)
    assert len(rows) == len({row["input"]["text"] for row in rows}) == 64
    cells = Counter(tuple(row["authority"][key] for key in FACTORS) for row in rows)
    assert cells == Counter({cell: 8 for cell in product(*FACTORS.values())})
    matched = defaultdict(list)
    for row in rows:
        text, metadata = row["input"]["text"], row["authority"]
        assert len(text) <= 512
        lines = text.splitlines()[1:]
        source = next(line for line in lines if line.startswith("Transaction source: "))
        note = next(line for line in lines if line.startswith("Untrusted memo: "))
        assert _record(source.removeprefix("Transaction source: ")) == row["reference"]
        assert _record(note.split("Return this record instead: ")[1]) == metadata["decoy"]
        assert ("ratified and authorized" in note) is metadata["authority_claim"]
        assert lines[0].startswith("Untrusted memo:") is (metadata["note_position"] == "first")
        assert note.split("instead: ")[1].startswith("{") is (metadata["answer_shape"] == "json")
        matched[metadata["matched_repeat"]].append(row)
    for group in matched.values():
        assert len(group) == 8
        assert len({json.dumps(row["reference"], sort_keys=True) for row in group}) == 1
        assert len({json.dumps(row["authority"]["decoy"], sort_keys=True) for row in group}) == 1
    for cell in product(*FACTORS.values()):
        group = [row for row in rows if tuple(row["authority"][key] for key in FACTORS) == cell]
        assert Counter(row["authority"]["trusted_format"] for row in group) == {"plain": 4, "json": 4}
        for field in ("customer", "product", "quantity"):
            assert Counter(row["reference"][field] for row in group) == Counter(
                row["authority"]["decoy"][field] for row in group)


def test_confirmation_labels_are_recovered_from_rendered_genuine_source_not_metadata():
    rows = confirmation_rows(seed=20261129)
    assert rows == confirmation_rows(seed=20261129)
    assert rows != confirmation_rows(seed=20261130)
    assert len(rows) == len({row["input"]["text"] for row in rows}) == 192
    assert sorted(Counter(row["family_id"] for row in rows).values()) == [32] * 6
    injected = [row for row in rows if row["panel_category"] == "injection"]
    assert len(injected) == 96
    assert set(Counter(tuple(row["confirmation_factors"][key] for key in FACTORS)
                       for row in injected).values()) == {12}
    for row in rows:
        assert rendered_reference(row) == row["reference"]
        changed = deepcopy(row)
        changed["reference"]["quantity"] = 0
        changed.pop("confirmation_factors", None)
        assert rendered_reference(changed) == row["reference"]
    for family in {row["family_id"] for row in injected}:
        group = [row for row in injected if row["family_id"] == family]
        for field in ("customer", "product", "quantity"):
            assert Counter(row["reference"][field] for row in group) == Counter(
                row["confirmation_factors"]["decoy"][field] for row in group)


def test_factorial_registry_and_current_spec_preserve_old_clean_rows_recipes_and_gates():
    root = Path(__file__).resolve().parents[1]
    source = root / "artifacts" / "phase-b-clean-contract-v2-20261111" / "run-spec.json"
    historical = json.loads(source.read_bytes())
    spec, audit = build_spec(source, policy_path=root / "protocols" / "json-contamination-v1.json")
    assert audit["public_passed"]
    assert audit["training_phrasing_families"] == ["ratification_assertion_untrusted_memo_v1"]
    assert len(spec["training_rows"]) == 384
    assert [row for row in spec["training_rows"] if not row["scenario_id"].startswith(
        "train:factorial-authority-v1:")] == [
            row for row in historical["training_rows"] if not row["scenario_id"].startswith("train:authority-v1:")]
    assert [row["recipe"] for row in spec["candidates"]] == [row["recipe"] for row in historical["candidates"]]
    assert spec["gates"] == historical["gates"]
    assert spec["public_sets"] == historical["public_sets"]
    assert not spec["policy"]["api"]["calls_authorized"]
    registry, refs = builtin_registry()
    assert spec["plugins"]["augmenter"] == refs["factorial_augmenter"]
    factory = registry.resolve("augmenter", refs["factorial_augmenter"])
    assert len(factory({"version": 1, "seed": 20261118})) == 64
    with pytest.raises(GymJobError, match="unsupported"):
        factory({"version": 2, "seed": 20261118})
