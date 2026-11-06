"""Milestone 1 qualification: proposal, compile, signed graph, replay, evaluation, signed run record.

Each stage runs in its own process and sees only the files it is given. Replay and evaluation run with
the proposer's modules blocked from import. A missing checkpoint or runtime raises, so this script has
no way to skip. Effect size isn't a pass condition here: the report records whatever was measured.

Qualification and promotion are separate. Qualifying a candidate proves it compiles, replays and can be
measured, and says nothing about whether the edit is any good. `--candidate` names any emitted branch,
selected or not, and promotion.json records whether that branch may be promoted, and why not.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import torch

from neural_foundry import _native, gym_proposal as gp
from neural_foundry import independent_eval as ie
from neural_foundry import run_record as rr
from neural_foundry.edit_plan import sign_edit_graph, tensor_state_digest
from neural_foundry.patch_ux import generate_ed25519_keypair, verify_patch_signature
from neural_foundry.promotion_gate import decide_promotion
from neural_foundry.pythia import ALGORITHMIC_THREE_SKILL_GYM, PythiaModel, run_pythia_gym_task_learning_study
from neural_foundry.records import canonicalize, content_digest

MODEL_ID = "EleutherAI/pythia-14m"
# Checked against huggingface.co/api/models/EleutherAI/pythia-14m: the LFS sha256 at this revision.
PYTHIA_14M_REVISION = "cf967c0a9a04383db6f7b1108d86b2962634b4ac"
PYTHIA_14M_SAFETENSORS_SHA256 = "sha256:116a02532db461f91386a5b20f942ff2c8d4de7341e21b55caafc3d7b25f49a1"
GYM = ALGORITHMIC_THREE_SKILL_GYM
QUALIFICATION_SCOPE = "Proves compilation, replay and measurement of this candidate. It is not evidence of edit quality."
BLOCKED_DURING_VERIFICATION = ("neural_foundry.gpt_neox_adapter", "neural_foundry.gym_scientist", "neural_foundry.gyms")
_BLOCKING_PRELUDE = f"""
import runpy, sys
class _Block:
    def find_spec(self, name, path=None, target=None):
        if name in {BLOCKED_DURING_VERIFICATION!r}:
            raise ImportError("blocked during verification: " + name)
        return None
sys.meta_path.insert(0, _Block())
"""


class QualificationError(RuntimeError):
    """A requirement is missing or a stage disagreed. The qualification fails."""


def require_runtime(checkpoint: Path, repo: Path) -> None:
    if not (checkpoint / "model.safetensors").is_file():
        raise QualificationError(f"Checkpoint is missing: {checkpoint / 'model.safetensors'}")
    actual = rr.sha256_file(checkpoint / "model.safetensors")
    if actual != PYTHIA_14M_SAFETENSORS_SHA256:
        raise QualificationError(f"Checkpoint is {actual}, not the pinned {PYTHIA_14M_SAFETENSORS_SHA256}")
    if importlib.util.find_spec("transformers") is None:
        raise QualificationError("The transformers cross-check backend is missing")
    package = Path(gp.__file__).parent
    for source in sorted(package.glob("*.py")):
        tracked = repo / "python" / "neural_foundry" / source.name
        if not tracked.is_file() or tracked.read_bytes() != source.read_bytes():
            raise QualificationError(f"Running code differs from the recorded repository: {source.name}")


def _write(path: Path, document: Any) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stage_synthesize(args: argparse.Namespace) -> None:
    """The Gym proposes and selects. Its report is discarded here: only the frozen proposals go on."""
    model = PythiaModel.from_pretrained_dir(args.checkpoint, model_name=MODEL_ID)
    report = run_pythia_gym_task_learning_study(model, gym=GYM)
    _write(
        args.out / "gym_proposals.json",
        [{"branch_id": c.branch_id, "selected": c.selected, "proposal": gp.proposal_to_document(c.proposal)} for c in report.gym_proposals],
    )
    _write(
        args.out / "selection.json",
        {
            "selected_branch_id": report.selected_branch_id,
            "retention_guard_digest": report.selection_retention_guard_digest,
            "branches": [
                {
                    "branch_id": b.branch_id,
                    "development_accuracy": b.gym_dev_accuracy,
                    "validation_accuracy": b.gym_prefix_val_accuracy,
                    "synthesis_retention_accuracy": b.retention_accuracy,
                    "retention_guard_accuracy": b.retention_guard_accuracy,
                    "retention_guard_source": b.retention_guard_source,
                    "reload_parity_verified": b.reload_parity_verified,
                    "disqualified_because": list(b.disqualified_because),
                }
                for b in report.branches
            ],
        },
    )
    name = args.candidate or report.selected_branch_id
    if name is None:
        raise QualificationError(f"The Gym selected no branch; name one with --candidate to qualify it. {report.promotion_verification['reason']}")
    named = next((c for c in report.gym_proposals if c.branch_id == name), None)
    if named is None:
        raise QualificationError(f"The Gym emitted no candidate {name!r}: {[c.branch_id for c in report.gym_proposals]}")
    _write(args.out / "proposal.json", gp.proposal_to_document(named.proposal))
    # The Gym's development cases, in the evaluator's text form. These aren't the sealed holdout rules.
    cases = [[f"{prefix} {expression}", target.replace("\u0120", " ")] for prefix in GYM.gym_dev_prefixes for expression, target in GYM.base_rules]
    _write(args.out / "target_cases.json", cases)


def stage_compile(args: argparse.Namespace) -> None:
    """Read only the frozen proposal and the base checkpoint."""
    from neural_foundry.gpt_neox_adapter import compile_to_edit_graph

    proposal = gp.proposal_from_document(_read(args.proposal))
    graph = compile_to_edit_graph(PythiaModel.from_pretrained_dir(args.checkpoint, model_name=MODEL_ID), proposal)
    _write(args.out, {"compiled": gp.compile_proposal(proposal), "edit_graph": graph})


def stage_evaluate(args: argparse.Namespace) -> None:
    """Measure the replayed export, checked against the signed digest, without the proposer."""
    envelope = _read(args.envelope)
    verify_patch_signature(envelope, public_key=args.public_key_file.read_text(encoding="utf-8").strip())
    signed = envelope["patch"]["candidate_state_digest"]
    base = PythiaModel.from_pretrained_dir(args.checkpoint, model_name=MODEL_ID)
    replayed = PythiaModel.from_pretrained_dir(args.export, model_name="replayed")
    if tensor_state_digest(replayed.tensors) != signed:
        raise QualificationError("The replayed export doesn't match the signed candidate digest")
    edited = {n: t for n, t in replayed.tensors.items() if not torch.equal(t, base.tensors[n])}
    config = ie.EvalConfig(bootstrap_samples=args.bootstrap_samples, random_control_seeds=tuple(args.control_seeds))
    candidate = ie.freeze_candidate(base.tensors, edited, candidate_id="milestone1", target_cases=[tuple(c) for c in _read(args.target_cases)])
    report = ie.evaluate_candidate(base, candidate, config=config)

    spec = importlib.util.spec_from_file_location("run_independent_eval", Path(__file__).with_name("run_independent_eval.py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parity = module.transformers_parity(args.checkpoint, base, dict(candidate.tensors), ie.load_corpus()["text"][:8])
    if max(parity["max_abs_logit_difference"].values()) >= 1e-4:
        raise QualificationError(f"The evaluator's logits disagree with transformers: {parity}")
    report["transformers_parity"] = parity
    report["signed_candidate_state_digest"] = signed
    report["replayed_tensors"] = sorted(edited)
    _write(args.out, report)


def promotion_record(report: dict[str, Any], selection: dict[str, Any], candidate: str) -> dict[str, Any]:
    """The gate's decision, plus the Gym's. A candidate needs both, so an unselected one is rejected even if it measures well."""
    gate = decide_promotion(report)
    branch = next(b for b in selection["branches"] if b["branch_id"] == candidate)
    selected = selection["selected_branch_id"] == candidate
    reasons = [f"gate: {check}" for check in gate["failed_checks"]]
    if gate["status"] == "not_evaluated":
        reasons.append(f"gate: {gate['reason']}")
    if not selected:
        reasons.append(f"selection: not selected, selected branch is {selection['selected_branch_id']}")
    reasons += [f"selection: {reason}" for reason in branch["disqualified_because"]]
    return {
        "candidate_branch_id": candidate,
        "qualification_scope": QUALIFICATION_SCOPE,
        "promotion_status": "passed" if gate["promote_eligible"] and selected else "rejected",
        "rejection_reasons": reasons,
        "gym_selected_branch_id": selection["selected_branch_id"],
        "gate": gate,
    }


def _stage(args: argparse.Namespace, name: str, extra: list[str], *, seed: int) -> None:
    subprocess.run([sys.executable, __file__, "--stage", name, "--checkpoint", str(args.checkpoint), *extra], check=True, env={**os.environ, "PYTHONHASHSEED": str(seed), "OMP_NUM_THREADS": "2"})


def _blocked_run(command: list[str], *, cwd: Path, runner: str) -> subprocess.CompletedProcess[str]:
    code = _BLOCKING_PRELUDE + f"sys.argv = {command!r}\n" + runner
    done = subprocess.run([sys.executable, "-c", code], cwd=cwd, check=False, capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "2"})
    if done.returncode != 0:
        raise QualificationError(f"Verification stage failed:\n{done.stderr}")
    return done


def orchestrate(args: argparse.Namespace) -> Path:
    args.checkpoint, args.repo, args.out = args.checkpoint.resolve(), args.repo.resolve(), args.out.resolve()
    require_runtime(args.checkpoint, args.repo)
    commit = rr.capture_commit(args.repo)
    if commit["dirty"] and not args.allow_dirty:
        raise QualificationError("The work tree has uncommitted changes")
    out: Path = args.out
    if out.exists():
        raise QualificationError(f"Refusing to reuse {out}")
    synth, scratch, evidence = out / "synthesis", out / "scratch", out / "evidence"
    for folder in (synth, scratch, evidence):
        folder.mkdir(parents=True)

    _stage(args, "synthesize", ["--out", str(synth), *(["--candidate", args.candidate] if args.candidate else [])], seed=1)
    proposal, cases = evidence / "proposal.json", evidence / "target_cases.json"
    shutil.copy(synth / "proposal.json", proposal)
    shutil.copy(synth / "target_cases.json", cases)
    shutil.copy(synth / "gym_proposals.json", evidence / "gym_proposals.json")
    shutil.copy(synth / "selection.json", evidence / "selection.json")

    # Condition 3: two fresh processes with different hash seeds must write identical bytes.
    first, second = scratch / "compiled_a.json", scratch / "compiled_b.json"
    _stage(args, "compile", ["--proposal", str(proposal), "--out", str(first)], seed=2)
    _stage(args, "compile", ["--proposal", str(proposal), "--out", str(second)], seed=3)
    if first.read_bytes() != second.read_bytes():
        raise QualificationError("Two compilations of one proposal differ")
    compiled = _read(first)
    graph = compiled["edit_graph"]
    if graph["candidate_state_digest"] != compiled["compiled"]["expected_candidate_digest"]:
        raise QualificationError("The edit graph doesn't reproduce the proposal's digest")
    _write(evidence / "compiled_proposal.json", compiled["compiled"])
    _write(evidence / "edit_graph.json", graph)

    # An ephemeral key proves integrity since the run, not who ran it. Key distribution is a later milestone.
    keys = generate_ed25519_keypair()
    envelope = evidence / "edit_graph.signed.json"
    _write(envelope, sign_edit_graph(graph, signing_key=keys["private_key"]))
    public = evidence / "signing.pub"
    public.write_text(keys["public_key"] + "\n", encoding="utf-8", newline="\n")

    # Conditions 4 and 5: an empty directory holding only the envelope and key, and the proposer blocked.
    room = scratch / "replay"
    room.mkdir()
    shutil.copy(envelope, room / "envelope.json")
    shutil.copy(public, room / "public.txt")
    export = scratch / "export"
    done = _blocked_run(
        ["edit_plan", "--checkpoint", str(args.checkpoint), "--envelope", "envelope.json", "--public-key-file", "public.txt", "--export-dir", str(export)],
        cwd=room,
        runner="runpy.run_module('neural_foundry.edit_plan', run_name='__main__', alter_sys=True)\n",
    )
    replay = json.loads(done.stdout.strip().splitlines()[-1])
    if replay["candidate_state_digest"] != graph["candidate_state_digest"] or replay["export_state_digest"] != graph["candidate_state_digest"]:
        raise QualificationError("Replay didn't reproduce the signed digest")
    if sorted(p.name for p in room.iterdir()) != ["envelope.json", "public.txt"]:
        raise QualificationError("Replay read or wrote something unexpected in its directory")
    _write(evidence / "replay.json", {**replay, "blocked_modules": list(BLOCKED_DURING_VERIFICATION), "directory_contents": ["envelope.json", "public.txt"]})

    # Condition 6: this process gets the replay output and the targets, never the proposal.
    report = evidence / "report.json"
    _blocked_run(
        [
            __file__, "--stage", "evaluate", "--checkpoint", str(args.checkpoint), "--envelope", str(room / "envelope.json"),
            "--public-key-file", str(room / "public.txt"), "--export", str(export), "--target-cases", str(cases), "--out", str(report),
            "--bootstrap-samples", str(args.bootstrap_samples), "--control-seeds", *[str(s) for s in args.control_seeds],
        ],
        cwd=room,
        runner=f"runpy.run_path({__file__!r}, run_name='__main__')\n",
    )

    # A rejection is a result, not a failed run.
    selection = _read(evidence / "selection.json")
    named = args.candidate or selection["selected_branch_id"]
    _write(evidence / "promotion.json", promotion_record(_read(report), selection, named))

    # Condition 7.
    version = tomllib.loads((args.repo / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    names = ("proposal.json", "gym_proposals.json", "target_cases.json", "compiled_proposal.json", "edit_graph.json", "edit_graph.signed.json", "signing.pub", "replay.json", "report.json", "selection.json", "promotion.json")
    record = rr.build_run_record(
        commit=commit,
        checkpoint=rr.capture_checkpoint(args.checkpoint, model_id=MODEL_ID, revision=PYTHIA_14M_REVISION),
        dependency_lock=rr.capture_dependencies(args.repo),
        runtime=rr.capture_runtime(_native.__file__, package_version=version),
        device="cpu",
        precision="f64 compute from the checkpoint's stored precision",
        sampler="deterministic_full_distribution_no_sampling",
        seeds={"bootstrap": ie.EvalConfig().bootstrap_seed, "random_controls": list(args.control_seeds)},
        edit_graph_digest=str(content_digest(canonicalize(graph))),
        command=[Path(sys.executable).name, *sys.argv],
        artifact_manifest=[rr.file_entry(evidence / n, relative_to=evidence) for n in names],
    )
    signed = rr.sign_run_record(record, signing_key=keys["private_key"])
    rr.verify_run_record(signed, public_key=keys["public_key"], artifact_root=evidence)
    _write(evidence / "run_record.signed.json", signed)
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("synthesize", "compile", "evaluate"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--proposal", type=Path)
    parser.add_argument("--candidate", help="The emitted branch to qualify. Defaults to the Gym's selection, and is required if there is none")
    parser.add_argument("--envelope", type=Path)
    parser.add_argument("--public-key-file", type=Path)
    parser.add_argument("--export", type=Path)
    parser.add_argument("--target-cases", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=ie.EvalConfig().bootstrap_samples)
    parser.add_argument("--control-seeds", type=int, nargs="+", default=list(ie.EvalConfig().random_control_seeds))
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage:
        {"synthesize": stage_synthesize, "compile": stage_compile, "evaluate": stage_evaluate}[args.stage](args)
        return 0
    if args.repo is None:
        raise QualificationError("--repo is required")
    print(orchestrate(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
