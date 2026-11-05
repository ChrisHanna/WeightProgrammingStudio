"""Local-first neural programming. Importing the package performs no research work."""

import importlib
from typing import TYPE_CHECKING, Any

from . import (
    _native,
    autonomous,
    diagnostics,
    evals,
    flagship,
    inspection,
    ir,
    pretrained,
    repair_benchmark,
    surgeon,
)
from .records import (
    CancellationHandle,
    CancelledError,
    NativeError,
    NativePanicError,
    canonicalize,
    capabilities,
    content_digest,
    negotiate,
    validate,
)
from .research_workflow import ResearchWorkflow

if TYPE_CHECKING:
    from . import (
        gpu,
        m4_ecosystems,
        m4_modalities,
        m5_surgery,
        prd_adapters,
        pythia,
        studio,
    )

__version__: str = _native.version()

_LAZY_SUBMODULES = frozenset(
    {
        "gpu",
        "m4_ecosystems",
        "m4_modalities",
        "m5_surgery",
        "prd_adapters",
        "pythia",
        "studio",
    }
)


def __getattr__(name: str) -> Any:
    if name in _LAZY_SUBMODULES:
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def workspace_crates() -> tuple[str, ...]:
    """Return the workspace inventory supplied by the Rust core."""
    return tuple(_native.workspace_crates())


__all__ = [
    "CancellationHandle",
    "CancelledError",
    "NativeError",
    "NativePanicError",
    "ResearchWorkflow",
    "__version__",
    "autonomous",
    "canonicalize",
    "capabilities",
    "content_digest",
    "diagnostics",
    "evals",
    "flagship",
    "gpu",
    "inspection",
    "ir",
    "m4_ecosystems",
    "m4_modalities",
    "m5_surgery",
    "negotiate",
    "prd_adapters",
    "pretrained",
    "pythia",
    "repair_benchmark",
    "studio",
    "surgeon",
    "validate",
    "workspace_crates",
]
