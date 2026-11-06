"""The research protocol's guards: corpus disjointness, group-wise splits, and the seal ledger."""

from __future__ import annotations

from pathlib import Path

import pytest

from neural_foundry import retention_research as rr


def test_corpus_is_disjoint_from_every_evaluation_set() -> None:
    fit, tune = rr.split_corpus()
    guarded = (*rr.SEALED_RETENTION_PROMPTS, *rr.FRESH_SEALED_PROMPTS, *rr.PRACTICE_RETENTION_PROMPTS, *rr.gym_task_texts())
    guarded_trigrams = set().union(*(rr._trigrams(p) for p in guarded))
    for prompt in (*fit, *tune):
        assert prompt not in guarded
        assert not rr._trigrams(prompt) & guarded_trigrams


def test_fresh_sealed_set_is_new() -> None:
    assert not set(rr.FRESH_SEALED_PROMPTS) & {*rr.SEALED_RETENTION_PROMPTS, *rr.PRACTICE_RETENTION_PROMPTS}


def test_split_never_shares_a_group() -> None:
    fit, tune = set(rr.split_corpus()[0]), set(rr.split_corpus()[1])
    assert not fit & tune
    for prompts in rr.build_corpus().values():
        assert set(prompts) <= fit or set(prompts) <= tune


def test_ledger_refuses_a_repeat_and_records_a_forced_one(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    rr.SealLedger(path).consult("m", "sealed_fresh", "final")
    ledger = rr.SealLedger(path)
    with pytest.raises(PermissionError):
        ledger.consult("m", "sealed_fresh", "again")
    ledger.consult("m", "sealed_fresh", "again", force=True)
    assert rr.SealLedger(path).count("m", "sealed_fresh") == 2
    assert rr.SealLedger(path).entries[-1]["repeat"] is True
