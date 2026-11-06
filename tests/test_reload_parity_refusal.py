"""A false reload-parity receipt must stop preview, execute and replay."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from neural_foundry import edit_plan as ep
from neural_foundry import pythia


def _plan() -> tuple[Any, ep.EditPlan]:
    model = pythia.build_mini_pythia_model()
    plan = ep.EditPlan(model)
    plan.scale(model.mlp_down_proj_site(0), 1.001)
    return model, plan


@pytest.fixture
def false_receipt(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Forces every receipt false, and records that each refused call really reached Rust."""
    calls: list[bool] = []
    real = ep.WeightIRTransactionBuilder.execute_matrix_edit

    def forced(*args: Any, **kwargs: Any) -> Any:
        tx = real(*args, **kwargs)
        calls.append(tx.reload_parity_verified)
        return replace(tx, reload_parity_verified=False)

    monkeypatch.setattr(ep.WeightIRTransactionBuilder, "execute_matrix_edit", staticmethod(forced))
    return calls


def test_a_true_receipt_still_goes_through() -> None:
    model, plan = _plan()
    assert plan.preview().state_digest
    assert ep.apply_edit_graph(model, json.loads(json.dumps(plan.to_graph()))).state_digest == plan.execute().state_digest


def test_preview_raises_on_a_false_receipt(false_receipt: list[bool]) -> None:
    _, plan = _plan()
    with pytest.raises(ep.ReloadParityError, match="reload parity"):
        plan.preview()
    assert false_receipt == [True]


def test_execute_raises_on_a_false_receipt(false_receipt: list[bool]) -> None:
    _, plan = _plan()
    with pytest.raises(ep.ReloadParityError):
        plan.execute()
    assert plan._last is None  # noqa: SLF001 - nothing refused may be kept for a later execute().


def test_replay_raises_on_a_false_receipt(monkeypatch: pytest.MonkeyPatch) -> None:
    model, plan = _plan()
    graph = json.loads(json.dumps(plan.to_graph()))
    real = ep.WeightIRTransactionBuilder.execute_matrix_edit
    monkeypatch.setattr(
        ep.WeightIRTransactionBuilder,
        "execute_matrix_edit",
        staticmethod(lambda *a, **k: replace(real(*a, **k), reload_parity_verified=False)),
    )
    with pytest.raises(ep.ReloadParityError):
        ep.apply_edit_graph(model, graph)


def test_a_missing_or_non_boolean_receipt_is_refused() -> None:
    for value in (None, 1, "true"):
        with pytest.raises(ep.ReloadParityError):
            ep._require_reload_parity(type("Tx", (), {"reload_parity_verified": value})(), tensor="t", operation="op")  # noqa: SLF001
