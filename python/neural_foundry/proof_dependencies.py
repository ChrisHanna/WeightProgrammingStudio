"""Declared built-in roots plus static repository-local Python import closure."""

from __future__ import annotations

import ast
from pathlib import Path

from .gym_job import GymJobError
from .provenance import sha256_file

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "python"

# Dynamic worker entrypoints/configs are declared explicitly rather than inferred from sys.modules.
ROOTS = {
    "gym": ("reference_json_extraction",),
    "model": ("gym_adapters", "model_digest", "prompt_rendering"),
    "trainer": ("backends.automodel.adapter", "backends.automodel.config", "backends.automodel.worker.probe"),
    "training_audit": ("backends.automodel.adapter", "proof_pipeline"),
    "provider": ("openai_provider", "openai_transport", "public_budget", "proof_policy"),
    "exporter": ("gym_artifacts", "model_digest", "vllm_candidate_files"),
    "server": ("vllm_container_adapter", "vllm_candidate_files", "backends.weight_edit.serving_probes"),
    "augmenter": ("json_authority_training",),
    "factorial_augmenter": ("json_factorial_training",),
    "public_audit": ("public_data_audit",),
    "evidence": ("json_proof_loop", "backends.weight_edit.serving_probes", "openai_provider"),
    "recorded": ("reference_json_extraction", "json_frontier_model"),
}


def _module_path(name: str) -> Path | None:
    if not name.startswith("neural_foundry"):
        return None
    path = PACKAGE.joinpath(*name.split("."))
    for candidate in (path.with_suffix(".py"), path / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def source_closure(modules: tuple[str, ...]) -> set[Path]:
    pending = ["neural_foundry." + name for name in modules]
    visited: set[Path] = set()
    while pending:
        name = pending.pop()
        path = _module_path(name)
        if path is None or path in visited:
            continue
        visited.add(path)
        if len(visited) > 256:
            raise GymJobError("declared public plugin source closure exceeded its bound")
        package = name.split(".") if path.name == "__init__.py" else name.split(".")[:-1]
        for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
            if isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names if alias.name.startswith("neural_foundry"))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package[:len(package) - node.level + 1]
                    target = ".".join([*base, *(node.module.split(".") if node.module else [])])
                else:
                    target = node.module or ""
                if target.startswith("neural_foundry"):
                    pending.append(target)
                    pending.extend(target + "." + alias.name for alias in node.names)
    return visited


def builtin_dependencies(kind: str) -> dict[str, str]:
    names = ROOTS[kind]
    paths = source_closure(names)
    paths.update(ROOT / "python" / "neural_foundry" / name
                 for name in ("proof_plugins.py", "proof_registry.py", "proof_dependencies.py",
                              "proof_policy.py", "proof_pipeline.py"))
    paths.add(ROOT / "protocols" / "public-proof-policy-v1.json")
    if kind == "public_audit":
        paths.add(ROOT / "protocols" / "json-contamination-v1.json")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in sorted(paths)}
