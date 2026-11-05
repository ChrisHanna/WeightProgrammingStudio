"""Fixed trusted model processes. Importing this module never imports Torch.

This is not a way to run untrusted gyms or Python architecture factories.
"""

from __future__ import annotations

import sys
import sysconfig
from os import PathLike
from pathlib import Path

from . import _native
from .records import CancellationHandle, RecordInput, _wire, negotiate


def qualify(
    scratch_root: str | PathLike[str],
    *,
    enabled: bool = False,
    python: str | PathLike[str] | None = None,
    site_packages: str | PathLike[str] | None = None,
) -> _native.TorchRuntimeHandle:
    """Explicitly approve local pinned trusted dependencies and actual probes."""
    negotiate()
    if not hasattr(_native, "qualify_torch_worker"):
        raise NotImplementedError("Native Linux trusted model worker is unavailable")
    return _native.qualify_torch_worker(
        str(Path(sys.executable if python is None else python).absolute()),
        str(
            Path(
                sysconfig.get_path("purelib") if site_packages is None else site_packages
            ).absolute()
        ),
        _native.__file__,
        str(Path(scratch_root).absolute()),
        enabled,
    )


def infer(
    runtime: _native.TorchRuntimeHandle,
    source: _native.SourceHandle,
    inputs: list[list[float]],
    *,
    cpu_seconds: int = 10,
    wall_ms: int = 15000,
    address_space_bytes: int = 2147483648,
    output_bytes: int = 262144,
    cancellation: CancellationHandle | None = None,
) -> _native.ModelInferenceHandle:
    """Return copied outputs and a native receipt, never a live model object."""
    negotiate()
    return _native.infer_model_worker(
        runtime,
        source,
        inputs,
        _wire(
            {
                "cpu_seconds": cpu_seconds,
                "wall_ms": wall_ms,
                "address_space_bytes": address_space_bytes,
                "output_bytes": output_bytes,
            }
        ),
        cancellation,
    )


def edit(
    runtime: _native.TorchRuntimeHandle,
    plan: _native.PlanHandle,
    smoke_suite: RecordInput,
    *,
    cpu_seconds: int = 10,
    wall_ms: int = 15000,
    address_space_bytes: int = 2147483648,
    output_bytes: int = 262144,
    cancellation: CancellationHandle | None = None,
) -> _native.TorchEditHandle:
    """Execute a compiled trusted-Torch policy, admit only full native-byte parity.

    No runtime operation dictionaries, training, or Python editor callbacks.
    The candidate retains the core transaction identities, effects and ancestry.
    """
    negotiate()
    return _native.edit_model_worker(
        runtime,
        plan,
        _wire(smoke_suite),
        _wire(
            {
                "cpu_seconds": cpu_seconds,
                "wall_ms": wall_ms,
                "address_space_bytes": address_space_bytes,
                "output_bytes": output_bytes,
            }
        ),
        cancellation,
    )
