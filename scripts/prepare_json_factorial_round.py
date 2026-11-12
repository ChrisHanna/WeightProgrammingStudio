"""Bounded public-only tests, no-forward budgets, then exclusive pre-results freeze."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from neural_foundry.execution_runner import run_logged_worker, write_status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tokenizer-python", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    cpu = project / ".venv" / "bin" / "python"
    source = project / "artifacts" / "phase-b-clean-contract-v2-20261111" / "run-spec.json"
    frozen = root / "frozen"
    token_receipt = root / "no-forward-token-audit.json"
    tests = ["tests/test_json_factorial_round.py", "tests/test_public_proof_pipeline.py",
             "tests/test_public_base_checkpoint.py", "tests/test_public_provider_hardening.py"]
    lint = ["python/neural_foundry/" + name + ".py" for name in (
        "json_factorial_training", "json_confirmation_panel", "json_factorial_round",
        "proof_registry", "proof_dependencies", "proof_plugins")]
    common = ["-m", "neural_foundry.json_factorial_round", "--source", str(source),
              "--output", str(frozen), "--token-audit", str(token_receipt)]
    commands = [
        ("public-tests", [str(cpu), "-m", "pytest", "-q", "--basetemp", str(root / "pytest-data"), *tests]),
        ("lint", [str(cpu), "-m", "ruff", "check", *lint, *tests,
                  "scripts/prepare_json_factorial_round.py"]),
        ("token-audit", [str(args.tokenizer_python), *common, "--audit-tokens-only"]),
        ("freeze", [str(cpu), *common]),
    ]
    env = {**os.environ, "PYTHONPATH": str(project / "python"), "PYTHONDONTWRITEBYTECODE": "1",
           "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    for stage, command in commands:
        logs = root / (stage + "-command")
        status = root / (stage + "-command-status.json")
        write_status(status, state="waiting_on_job", step=stage, job_receipt=logs / "receipt.json")
        result = run_logged_worker(command, cwd=project, log_dir=logs, timeout_seconds=600, env=env)
        write_status(status, state="done" if result.returncode == 0 else "blocked",
                     step=stage, job_receipt=logs / "receipt.json")
        print(stage, result.returncode, result.stdout.decode(errors="replace")[-2000:],
              result.stderr.decode(errors="replace")[-2000:], flush=True)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
