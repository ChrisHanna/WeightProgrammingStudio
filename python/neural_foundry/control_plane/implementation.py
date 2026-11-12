"""Fingerprint installed execution sources without importing execution dependencies."""

from __future__ import annotations

import ast
import platform
import sys
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

from ..provenance import sha256_file
from .models import digest, fail

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOTS = (
    "control_plane.implementation", "control_plane.models", "control_plane.service",
    "control_plane.store", "control_plane.artifacts", "control_plane.worker",
    "control_plane.handlers", "control_plane.bridges",
    # Subprocess entrypoints are not discoverable from ordinary Python import statements.
    "backends.automodel.worker.guarded_train", "backends.automodel.worker.probe",
    "backends.weight_edit.experimental_worker", "backends.weight_edit.worker",
)


def _module_files(name: str) -> list[Path]:
    if name != "neural_foundry" and not name.startswith("neural_foundry."):
        return []
    path = PACKAGE_ROOT.joinpath(*name.split(".")[1:])
    for candidate in (path.with_suffix(".py"), path / "__init__.py"):
        if candidate.is_file():
            return [candidate]
    # The native extension is an execution dependency, not an import-time probe.
    if name == "neural_foundry._native":
        return sorted([*PACKAGE_ROOT.glob("_native*.so"), *PACKAGE_ROOT.glob("_native*.pyd")])
    return []


def _source_files() -> set[Path]:
    pending = ["neural_foundry", *("neural_foundry." + name for name in ROOTS)]
    visited = set()
    while pending:
        name = pending.pop()
        for path in _module_files(name):
            if path in visited:
                continue
            visited.add(path)
            parts = name.split(".")
            pending.extend(".".join(parts[:index]) for index in range(1, len(parts)))
            if len(visited) > 512:
                fail("implementation_unavailable", "Execution dependency closure exceeds its bound.", 503)
            if path.suffix != ".py":
                continue
            package = name.split(".") if path.name == "__init__.py" else name.split(".")[:-1]
            for node in ast.walk(ast.parse(path.read_bytes())):
                if isinstance(node, ast.Import):
                    pending.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    base = package[:len(package) - node.level + 1] if node.level else []
                    target = ".".join([*base, *(node.module.split(".") if node.module else [])])
                    pending.append(target)
                    pending.extend(target + "." + alias.name for alias in node.names)
    return visited


def execution_implementation() -> dict[str, Any]:
    """Content binding works in both source checkout and wheel-only installations.

    Distribution versions conservatively bind the deployed dependency environment;
    no library/model is loaded to inspect it. Deployment images must be immutable.
    """
    try:
        files = {path.relative_to(PACKAGE_ROOT).as_posix(): sha256_file(path)
                 for path in sorted(_source_files())}
        environment = sorted({(dist.metadata["Name"], dist.version) for dist in distributions()
                              if dist.metadata.get("Name")})
    except (OSError, SyntaxError, ValueError):
        fail("implementation_unavailable", "Execution implementation could not be fingerprinted.", 503)
    manifest = {"version": 1, "files": files, "distributions": environment,
                "python": platform.python_implementation() + "-" + platform.python_version(),
                "platform": sys.platform, "machine": platform.machine()}
    return {"version": 1, "digest": digest(manifest), "file_count": len(files)}
