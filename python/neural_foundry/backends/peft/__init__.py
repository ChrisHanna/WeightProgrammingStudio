"""PEFT lifecycle backend."""

from .. import BackendFactory, registry
from .adapter import peft_sequence_trainer

registry.register(BackendFactory(
	"peft-sequence-lora", "1",
	{"torch": "2.9.1+cu128", "transformers": "4.57.3", "peft": "0.18.0"},
	peft_sequence_trainer,
))

__all__ = ["peft_sequence_trainer"]