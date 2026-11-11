"""Compatibility entrypoint; new public runs use --spec and the generic runner."""

from __future__ import annotations

import sys

from neural_foundry.json_validation_compat import (  # noqa: F401
    freeze_public_validation_v2,
    merge_lora,
    read_document,
    record_identity,
    reference_for,
    score,
    score_frontier_panel,
    trained_reference,
    write_document,
)


def main() -> None:
    if "--spec" in sys.argv:
        from neural_foundry.proof_pipeline import main as run
    else:
        from neural_foundry.json_validation_compat import main as run
    run()


if __name__ == "__main__":
    main()
