"""Thin custom-registry entrypoint; inference remains in the generic proof pipeline."""

import argparse
import json
from pathlib import Path

from neural_foundry.cord_receipt_plugin import compose, registry, summarize
from neural_foundry.proof_pipeline import audit, read_document, score, validate_spec, write_document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "audit", "export", "identity", "score"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--core-stable", action="store_true")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--role")
    args = parser.parse_args()
    if args.stage == "freeze":
        if not args.core_stable or not args.config:
            parser.error("freeze requires coordinator core-stable signal and --config")
        config = read_document(args.config)
        config.pop("document_digest")
        prepared = read_document(Path(config.pop("prepared_path")))
        if config.pop("prepared_digest") != prepared["document_digest"]:
            parser.error("prepared source/row digest differs from approved configuration")
        if not prepared["ready_for_inference"]:
            parser.error("prepared cases exceed frozen budgets; no truncation/drop permitted")
        config["prepared"] = prepared
        spec, _ = compose(config)
        result = write_document(args.spec, spec)
    else:
        custom, _ = registry()
        spec = read_document(args.spec)
        spec.pop("document_digest")
        validate_spec(spec, custom)
        if args.stage == "audit":
            result = audit(spec, custom)
            write_document(args.root / "public-audit.json", result)
        else:
            if not args.role:
                parser.error("--role required")
            candidate = next(c for c in spec["candidates"] if c["role"] == args.role)
            if args.stage == "export":
                result = custom.resolve("exporter", spec["plugins"]["exporter"])(
                    spec=spec, role=args.role, root=args.root)
            elif args.stage == "identity":
                result = custom.resolve("server", candidate["execution"]["plugin"])(
                    spec=spec, candidate=candidate, root=args.root, freeze_identity=True)
            else:
                result = score(spec, role=args.role, root=args.root, registry=custom)
                write_document(args.root / f"cord-summary-{args.role}.json", summarize(spec, result))
    print(json.dumps({k: v for k, v in result.items()
                      if k in {"kind", "role", "document_digest", "public_passed", "complete"}}))


if __name__ == "__main__":
    main()
