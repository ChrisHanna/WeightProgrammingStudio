"""PEFT trainer adapter factory."""

from __future__ import annotations

from collections.abc import Callable

from ...gym_lifecycle import TrainerAdapter


def peft_sequence_trainer(*, checkpoint_hook: Callable[[int], None] | None = None) -> TrainerAdapter:
    from .. import registry

    registry.require("peft-sequence-lora@1")
    from .worker import peft_sequence_trainer as create_trainer

    return create_trainer(checkpoint_hook=checkpoint_hook)