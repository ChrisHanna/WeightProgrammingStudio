"""A false reload-parity receipt must stop preview, execute and replay."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from neural_foundry import edit_plan as ep
from neural_foundry import ir, prd_adapters, pythia


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


def test_only_exactly_true_passes_the_central_check() -> None:
    for value in (None, 1, "true", False):
        with pytest.raises(ir.ReloadParityError):
            ir.require_reload_parity(type("Tx", (), {"reload_parity_verified": value})())
    with pytest.raises(ir.ReloadParityError):
        ir.require_reload_parity(object())
    ir.require_reload_parity(type("Tx", (), {"reload_parity_verified": True})())


@pytest.mark.parametrize("value", [False, None, 1])
def test_studies_and_adapters_stop_on_a_bad_receipt(monkeypatch: pytest.MonkeyPatch, value: Any) -> None:
    real = ir.WeightIRTransactionBuilder.execute_matrix_edit
    monkeypatch.setattr(ir.WeightIRTransactionBuilder, "execute_matrix_edit", staticmethod(lambda *a, **k: replace(real(*a, **k), reload_parity_verified=value)))
    model = pythia.build_mini_pythia_model()
    with pytest.raises(ir.ReloadParityError):
        pythia.run_pythia_weight_programming_study(model, target_strength=80.0)
    with pytest.raises(ir.ReloadParityError):
        pythia.run_pythia_gym_task_learning_study(model, target_strength=85.0)
    with pytest.raises(ir.ReloadParityError):
        prd_adapters.run_easyedit_adapter(model, algorithm="ROME", associations=[("Task: Number 9 has parity", "\u0120odd")])
    base = model.tensors[model.mlp_down_proj_site(0)].to(torch.float64)
    with pytest.raises(ir.ReloadParityError):
        prd_adapters.run_mergekit_adapter(model, algorithm="task_arithmetic", checkpoint_weights=[base + 0.01, base - 0.02])


def test_every_transaction_call_site_goes_through_the_check() -> None:
    package = Path(pythia.__file__).parent
    for source in package.glob("*.py"):
        text = source.read_text(encoding="utf-8")
        calls = len(re.findall(r"= (?:ir\.)?WeightIRTransactionBuilder\.execute_matrix_edit\(", text))
        checks = len(re.findall(r"require_reload_parity\(\w+\)", text))
        assert checks >= calls, source.name
