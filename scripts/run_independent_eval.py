"""Scale one Pythia tensor through the Rust executor, freeze it, evaluate it and sign the run record.

The edit is made first and handed over as a frozen candidate, so the evaluator sees only the result.
Add transformers to PYTHONPATH to cross-check the evaluator's logits against an independent forward pass.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

import torch

from neural_foundry import _native, independent_eval as ie
from neural_foundry import run_record as rr
from neural_foundry.edit_plan import EditPlan
from neural_foundry.patch_ux import generate_ed25519_keypair
from neural_foundry.pythia import PythiaModel
from neural_foundry.records import canonicalize, content_digest


def transformers_parity(checkpoint: Path, model: PythiaModel, edited: dict[str, torch.Tensor], sentences: list[str]) -> dict[str, object]:
    from transformers import GPTNeoXForCausalLM

    hf = GPTNeoXForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float64).eval()
    pythia_candidate = ie._with_tensors(model, edited)
    worst = {"base": 0.0, "candidate": 0.0}
    for label, ours in (("base", model), ("candidate", pythia_candidate)):
        if label == "candidate":
            with torch.no_grad():
                state = hf.state_dict()
                for name, value in edited.items():
                    state[name].copy_(value)
        for sentence in sentences:
            ids = model.tokenizer.encode(sentence)
            with torch.inference_mode():
                theirs = hf(torch.tensor([ids])).logits[0]
            worst[label] = max(worst[label], float((ie._logits(ours, ids) - theirs).abs().max()))
    return {"sentences": len(sentences), "max_abs_logit_difference": worst}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint-revision", required=True, help="Full upstream commit the files were checked against")
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--factor", type=float, required=True)
    parser.add_argument("--sweep", type=float, nargs="*", default=[])
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--no-parity", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    commit = rr.capture_commit(args.repo)
    if commit["dirty"] and not args.allow_dirty:
        raise rr.RunRecordError("The work tree has uncommitted changes")

    model = PythiaModel.from_pretrained_dir(args.checkpoint, model_name=args.model_id)
    corpus = ie.load_corpus()
    targets = [tuple(c) for c in corpus["tasks"]["capitals"]]
    plan = EditPlan(model).scale(args.tensor, args.factor)
    result = plan.execute()
    graph = plan.to_graph()
    candidate = ie.freeze_candidate(model.tensors, result.edited_tensors, candidate_id=f"scale_{args.factor}", target_cases=targets)
    config = ie.EvalConfig()
    report = ie.evaluate_candidate(model, candidate, config=config)
    report["edit"] = {"tensor": args.tensor, "factor": args.factor, "changed_parameters": result.changed_parameters, "frobenius_delta": result.frobenius_delta}

    sweep = {}
    for factor in args.sweep:
        edited = {args.tensor: EditPlan(model).scale(args.tensor, factor).execute().edited_tensors[args.tensor]}
        frozen = ie.freeze_candidate(model.tensors, edited, candidate_id=f"scale_{factor}", target_cases=targets)
        sweep[str(factor)] = ie.evaluate_candidate(model, frozen, config=ie.EvalConfig(random_control_seeds=(101, 102, 103)))
    report["sweep"] = sweep
    report["transformers_parity"] = (
        "not run" if args.no_parity else transformers_parity(args.checkpoint, model, dict(candidate.tensors), corpus["text"][:8])
    )

    args.out.mkdir(parents=True, exist_ok=True)
    paths = {"report.json": report, "edit_graph.json": graph}
    for name, document in paths.items():
        (args.out / name).write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    version = tomllib.loads((args.repo / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    record = rr.build_run_record(
        commit=commit,
        checkpoint=rr.capture_checkpoint(args.checkpoint, model_id=args.model_id, revision=args.checkpoint_revision),
        dependency_lock=rr.capture_dependencies(args.repo),
        runtime=rr.capture_runtime(_native.__file__, package_version=version),
        device="cpu",
        precision="f64 compute from the checkpoint's stored precision",
        sampler="deterministic_full_distribution_no_sampling",
        seeds={"bootstrap": config.bootstrap_seed, "random_controls": list(config.random_control_seeds)},
        edit_graph_digest=str(content_digest(canonicalize(graph))),
        command=[Path(sys.executable).name, *sys.argv],
        artifact_manifest=[rr.file_entry(args.out / name, relative_to=args.out) for name in paths],
    )
    # An ephemeral key proves integrity since the run, not who ran it. Key distribution is a later milestone.
    keys = generate_ed25519_keypair()
    envelope = rr.sign_run_record(record, signing_key=keys["private_key"])
    rr.verify_run_record(envelope, public_key=keys["public_key"], artifact_root=args.out)
    (args.out / "run_record.signed.json").write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out / "run_record.pub").write_text(keys["public_key"] + "\n", encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
