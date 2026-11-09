"""Milestone 1 conditions 6 and 7. Most tests use a stub model, so only the last needs a checkpoint."""

from __future__ import annotations

import ast
import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from neural_foundry import independent_eval as ie
from neural_foundry import run_record as rr
from neural_foundry.patch_ux import generate_ed25519_keypair

PACKAGE = Path(ie.__file__).parent
PYTHIA_14M = Path("/tmp/pythia-14m")
VOCAB = 211
TARGETS = [("The capital of France is", " Paris"), ("1 + 1 =", " 2")]


class _WordTokenizer:
    def encode(self, text: str) -> list[int]:
        return [sum(ord(c) * (i + 1) for i, c in enumerate(w)) % VOCAB for w in re.findall(r"\S+", text)]


class _StubModel:
    """A tiny bag-of-context model with the interface the evaluator reads."""

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(7)
        self.tokenizer = _WordTokenizer()
        self.tensors = {
            "embed": torch.randn(VOCAB, 16, dtype=torch.float64, generator=generator),
            "out": torch.randn(16, VOCAB, dtype=torch.float64, generator=generator),
        }

    def forward_with_mlp_key(self, ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tensors["embed"][ids].cumsum(dim=0)
        return x @ self.tensors["out"], x[-1]


@pytest.fixture(scope="module")
def stub() -> _StubModel:
    return _StubModel()


@pytest.fixture(scope="module")
def config() -> ie.EvalConfig:
    return ie.EvalConfig(bootstrap_samples=200, random_control_seeds=(1, 2))


def _freeze(model: _StubModel, factor: float = 0.5) -> ie.FrozenCandidate:
    return ie.freeze_candidate(model.tensors, {"out": model.tensors["out"] * factor}, candidate_id="scale", target_cases=TARGETS)


def _record_fields(tmp_path: Path) -> dict:
    artifact = tmp_path / "report.json"
    artifact.write_text("{}", encoding="utf-8")
    entry = rr.file_entry(artifact, relative_to=tmp_path)
    digest = "sha256:" + "a" * 64
    return {
        "commit": {"sha": "b" * 40, "dirty": False},
        "checkpoint": {"model_id": "org/model", "revision": "c" * 40, "digest": digest, "files": [entry]},
        "dependency_lock": {"lock_files": [entry], "environment_digest": digest, "installed_packages": ["torch==2.9.1"]},
        "runtime": {"package_version": "0.1.0", "python": "3.12.3", "torch": "2.9.1", "native_extension_digest": digest, "platform": "linux"},
        "device": "cpu",
        "precision": "f64",
        "sampler": "deterministic_full_distribution_no_sampling",
        "seeds": {"bootstrap": 1, "random_controls": [2, 3]},
        "edit_graph_digest": digest,
        "command": ["python", "run.py"],
        "artifact_manifest": [entry],
    }


# ---- Independence from proposers.
def test_evaluator_imports_only_stdlib_and_torch() -> None:
    tree = ast.parse((PACKAGE / "independent_eval.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "the evaluator must not import from its own package"
            imported.add((node.module or "").split(".")[0])
    assert imported - set(sys.stdlib_module_names) == {"torch"}


def test_evaluator_loads_without_the_package(tmp_path: Path) -> None:
    script = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('isolated_eval', {str(PACKAGE / 'independent_eval.py')!r})\n"
        "m = importlib.util.module_from_spec(spec); sys.modules['isolated_eval'] = m; spec.loader.exec_module(m)\n"
        "assert m.load_corpus()['corpus_id'] == 'independent_eval_corpus_v1'\n"
        "assert not [n for n in sys.modules if n.startswith('neural_foundry')]\n"
    )
    done = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr


# ---- The corpus.
def test_corpus_is_pinned_and_a_changed_corpus_is_refused(tmp_path: Path) -> None:
    corpus = ie.load_corpus()
    assert len(corpus["tasks"]) == 6
    assert all(len(cases) == 16 for cases in corpus["tasks"].values())
    corpus["text"][0] += " tampered"
    changed = tmp_path / "corpus.json"
    changed.write_text(json.dumps(corpus), encoding="utf-8")
    with pytest.raises(ie.IndependentEvalError, match="does not match"):
        ie.load_corpus(changed)


# ---- Measurements.
def test_report_has_every_required_measurement(stub: _StubModel, config: ie.EvalConfig) -> None:
    report = ie.evaluate_candidate(stub, _freeze(stub), config=config)
    json.dumps(report, allow_nan=False)
    assert report["target"]["metrics"]["n"] == len(TARGETS)
    assert set(report["retention_matrix"]) == set(ie.load_corpus()["tasks"])
    for row in report["retention_matrix"].values():
        assert set(row) == {"candidate", "noop", "random_seed_1", "random_seed_2"}
        assert row["noop"]["top1_agreement"]["value"] == 1.0
    text = report["candidate"]["text"]
    assert text["kl_nats_per_token"]["value"] > 0
    low, high = text["candidate_perplexity"]["ci95"]
    assert low <= text["candidate_perplexity"]["value"] <= high
    assert report["controls"]["noop"]["passed"] is True
    assert report["target"]["case_count"] == len(TARGETS)
    assert "cases" not in report["target"]


def test_kl_matches_a_hand_computed_example() -> None:
    base = torch.log(torch.tensor([0.5, 0.5], dtype=torch.float64))
    other = torch.log(torch.tensor([0.25, 0.75], dtype=torch.float64))
    kl, tv = ie._kl_tv(base, other)
    assert float(kl) == pytest.approx(0.5 * torch.log(torch.tensor(2.0)).item() + 0.5 * torch.log(torch.tensor(2 / 3)).item())
    assert float(tv) == pytest.approx(0.25)


def test_perplexity_of_a_uniform_model_is_the_vocabulary_size(stub: _StubModel, config: ie.EvalConfig) -> None:
    flat = copy.copy(stub)
    flat.tensors = {**stub.tensors, "out": torch.zeros_like(stub.tensors["out"])}
    edited = {"out": torch.full_like(stub.tensors["out"], 1e-3) * flat.tensors["embed"][0, 0].sign()}
    report = ie.evaluate_candidate(flat, ie.freeze_candidate(flat.tensors, edited, candidate_id="c", target_cases=TARGETS), config=config)
    assert report["candidate"]["text"]["base_perplexity"]["value"] == pytest.approx(VOCAB)


def test_wilson_interval_is_correct_and_keeps_width_at_the_extremes() -> None:
    assert ie.wilson_interval(0, 0) is None
    low, high = ie.wilson_interval(10, 10)
    assert high == 1.0 and low == pytest.approx(0.7225, abs=1e-3)
    low, high = ie.wilson_interval(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-3) and high == pytest.approx(0.7634, abs=1e-3)


def test_random_control_matches_support_and_norm(stub: _StubModel) -> None:
    mask = torch.zeros_like(stub.tensors["out"])
    mask[:3] = 1
    edited = stub.tensors["out"] * (1 - 0.5 * mask)
    candidate = ie.freeze_candidate(stub.tensors, {"out": edited}, candidate_id="c", target_cases=TARGETS)
    control = ie._random_control(stub.tensors, candidate, 5)["out"]
    delta = control - stub.tensors["out"]
    assert torch.equal(delta != 0, edited != stub.tensors["out"])
    assert float(torch.linalg.norm(delta)) == pytest.approx(float(torch.linalg.norm(edited - stub.tensors["out"])))


def test_evaluation_is_deterministic_and_leaves_the_model_unchanged(stub: _StubModel, config: ie.EvalConfig) -> None:
    before = ie.parameter_digest(stub.tensors)
    first = ie.evaluate_candidate(stub, _freeze(stub), config=config)
    second = ie.evaluate_candidate(stub, _freeze(stub), config=config)
    assert first == second
    assert ie.parameter_digest(stub.tensors) == before


# ---- Refusals.
def test_noop_control_failure_raises(stub: _StubModel, config: ie.EvalConfig) -> None:
    drift = {"calls": 0}

    class Nondeterministic(_StubModel):
        def forward_with_mlp_key(self, ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
            drift["calls"] += 1
            logits, key = super().forward_with_mlp_key(ids)
            return logits + 1e-3 * (drift["calls"] % 3) * torch.arange(VOCAB, dtype=torch.float64), key

    model = Nondeterministic()
    with pytest.raises(ie.ControlFailure):
        ie.evaluate_candidate(model, _freeze(model), config=config)


def test_wrong_base_and_tampered_candidate_are_refused(stub: _StubModel, config: ie.EvalConfig) -> None:
    other = _StubModel()
    other.tensors = {**stub.tensors, "out": stub.tensors["out"] + 1}
    with pytest.raises(ie.IndependentEvalError, match="frozen against"):
        ie.evaluate_candidate(other, _freeze(stub), config=config)
    candidate = _freeze(stub)
    candidate.tensors["out"][0, 0] += 1.0  # type: ignore[index]
    with pytest.raises(ie.IndependentEvalError, match="hash to"):
        ie.evaluate_candidate(stub, candidate, config=config)


def test_identical_candidate_missing_targets_and_multitoken_targets_are_refused(stub: _StubModel, config: ie.EvalConfig) -> None:
    same = ie.freeze_candidate(stub.tensors, {"out": stub.tensors["out"]}, candidate_id="c", target_cases=TARGETS)
    with pytest.raises(ie.IndependentEvalError, match="identical"):
        ie.evaluate_candidate(stub, same, config=config)
    with pytest.raises(ie.IndependentEvalError, match="no target cases"):
        ie.evaluate_candidate(stub, ie.freeze_candidate(stub.tensors, {"out": stub.tensors["out"] * 2}, candidate_id="c", target_cases=[]), config=config)
    two_words = [("a b", " two words")]
    with pytest.raises(ie.IndependentEvalError, match="exactly 1"):
        ie.evaluate_candidate(stub, ie.freeze_candidate(stub.tensors, {"out": stub.tensors["out"] * 2}, candidate_id="c", target_cases=two_words), config=config)


# ---- The run record.
def test_run_record_signs_verifies_and_checks_artifacts(tmp_path: Path) -> None:
    keys = generate_ed25519_keypair()
    envelope = rr.sign_run_record(rr.build_run_record(**_record_fields(tmp_path)), signing_key=keys["private_key"])
    assert rr.verify_run_record(envelope, public_key=keys["public_key"], artifact_root=tmp_path)["device"] == "cpu"
    (tmp_path / "report.json").write_text("{ }", encoding="utf-8")
    with pytest.raises(rr.RunRecordError, match="does not match"):
        rr.verify_run_record(envelope, public_key=keys["public_key"], artifact_root=tmp_path)
    (tmp_path / "report.json").unlink()
    with pytest.raises(rr.RunRecordError, match="missing"):
        rr.verify_run_record(envelope, public_key=keys["public_key"], artifact_root=tmp_path)


def test_run_record_rejects_tampering_and_the_wrong_key(tmp_path: Path) -> None:
    keys = generate_ed25519_keypair()
    envelope = rr.sign_run_record(rr.build_run_record(**_record_fields(tmp_path)), signing_key=keys["private_key"])
    with pytest.raises(rr.RunRecordError, match="signature"):
        rr.verify_run_record(envelope, public_key=generate_ed25519_keypair()["public_key"])
    changed = copy.deepcopy(envelope)
    changed["patch"]["device"] = "cuda"
    with pytest.raises(rr.RunRecordError, match="signature"):
        rr.verify_run_record(changed, public_key=keys["public_key"])


@pytest.mark.parametrize("field", ["commit", "checkpoint", "dependency_lock", "runtime", "device", "precision", "sampler", "seeds", "edit_graph_digest", "command", "artifact_manifest"])
def test_each_run_record_field_is_required(tmp_path: Path, field: str) -> None:
    record = rr.build_run_record(**_record_fields(tmp_path))
    del record[field]
    with pytest.raises(rr.RunRecordError, match=field):
        rr.validate_run_record(record)
    keys = generate_ed25519_keypair()
    with pytest.raises(rr.RunRecordError):
        rr.sign_run_record(record, signing_key=keys["private_key"])


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("device",), "unknown"),
        (("device",), ""),
        (("sampler",), "None"),
        (("precision",), None),
        (("command",), []),
        (("seeds",), {}),
        (("seeds", "bootstrap"), "seven"),
        (("seeds", "bootstrap"), 1.5),
        (("commit", "sha"), "abc123"),
        (("commit", "sha"), "unknown"),
        (("commit", "dirty"), "no"),
        (("checkpoint", "revision"), "main"),
        (("checkpoint", "digest"), "deadbeef"),
        (("runtime", "torch"), "unknown"),
        (("edit_graph_digest",), "sha256:short"),
        (("artifact_manifest",), []),
        (("dependency_lock", "installed_packages"), []),
    ],
)
def test_run_record_refuses_unknown_values(tmp_path: Path, path: tuple[str, ...], value: object) -> None:
    fields = _record_fields(tmp_path)
    holder: dict = fields
    for key in path[:-1]:
        holder = holder[key]
    holder[path[-1]] = value
    with pytest.raises(rr.RunRecordError):
        rr.build_run_record(**fields)


def test_run_record_refuses_extra_fields_and_missing_files(tmp_path: Path) -> None:
    record = rr.build_run_record(**_record_fields(tmp_path))
    record["note"] = "extra"
    with pytest.raises(rr.RunRecordError, match="unexpected"):
        rr.validate_run_record(record)
    with pytest.raises(rr.RunRecordError, match="does not exist"):
        rr.file_entry(tmp_path / "absent.json")
    with pytest.raises(rr.RunRecordError, match="does not exist"):
        rr.capture_checkpoint(tmp_path / "absent", model_id="m", revision="c" * 40)


def test_capture_commit_fails_outside_a_repository(tmp_path: Path) -> None:
    with pytest.raises(rr.RunRecordError, match="git"):
        rr.capture_commit(tmp_path)


def test_frontier_roster_excludes_failed_and_skipped_searches() -> None:
    from neural_foundry.frontier_evaluation import _qualification_roster

    records = {
        "passing": {"arm": "direct", "method": "wsp", "search_state": "accepted"},
        "failed": {"arm": "direct", "method": "wsp", "search_state": "failed"},
        "skipped": {"arm": "baseline", "method": "lora", "search_state": "skipped"},
        "base": {"arm": "baseline", "method": "no_op", "search_state": "accepted"},
    }
    assert _qualification_roster(records) == ("passing",)


def test_frontier_sealed_exposure_retry_limit_is_enforced(tmp_path: Path) -> None:
    from neural_foundry import frontier_evaluation as evaluation
    from neural_foundry.records import canonicalize, content_digest

    document = {"snapshot_digest": "sha256:" + "d" * 64}
    request = str(content_digest(canonicalize({"request": "same"})))
    with evaluation._evaluation_lease(document, tmp_path):
        paths, cached = evaluation._begin_evaluation_claim(document, tmp_path, request)
    assert cached is None
    for _ in range(evaluation.MAX_EXPOSURE_RETRIES):
        with evaluation._evaluation_lease(document, tmp_path):
            same_paths, cached = evaluation._begin_evaluation_claim(document, tmp_path, request)
        assert same_paths == paths and cached is None
    with (
        pytest.raises(evaluation.FrontierEvaluationError, match="retry limit is exhausted"),
        evaluation._evaluation_lease(document, tmp_path),
    ):
        evaluation._begin_evaluation_claim(document, tmp_path, request)


def test_frontier_isolated_payload_prefers_verified_checkpoint_identity(tmp_path: Path) -> None:
    from types import SimpleNamespace

    import torch

    from neural_foundry.edit_plan import tensor_state_digest
    from neural_foundry.frontier_evaluation import _isolated_model_payload
    from neural_foundry.provenance import resolve_checkpoint

    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        (tmp_path / name).write_bytes(b"{}")
    checkpoint = resolve_checkpoint(tmp_path, revision="a" * 40)
    tokenizer = SimpleNamespace(vocab={"x": 1}, bpe_ranks={}, added_tokens={}, byte_level=False)
    model = SimpleNamespace(
        model_name="pinned-mini",
        config={"model_type": "gpt_neox"},
        tensors={"weight": torch.ones((2, 2), dtype=torch.float64)},
        tokenizer=tokenizer,
        checkpoint_identity=checkpoint,
    )
    model._checkpoint_state_digest = tensor_state_digest(model.tensors)

    payload = _isolated_model_payload(model)

    assert "checkpoint_identity" in payload and "tensors" not in payload
    assert payload["checkpoint_identity"]["file_digests"] == checkpoint.file_digests


# ---- Real checkpoint. Skipped in normal CI, and a failure during a qualification run.
def test_pythia_14m_scale_edit_is_measured() -> None:
    if not (PYTHIA_14M / "model.safetensors").is_file():
        if os.environ.get("NF_QUALIFICATION"):
            pytest.fail("Qualification requires the Pythia-14M checkpoint")
        pytest.skip("Pythia-14M checkpoint is not available")
    from neural_foundry.edit_plan import EditPlan
    from neural_foundry.pythia import PythiaModel

    model = PythiaModel.from_pretrained_dir(PYTHIA_14M)
    site = model.mlp_down_proj_site(5)
    plan = EditPlan(model).scale(site, 0.9)
    candidate = ie.freeze_candidate(
        model.tensors, plan.execute().edited_tensors, candidate_id="scale", target_cases=[tuple(c) for c in ie.load_corpus()["tasks"]["capitals"]]
    )
    report = ie.evaluate_candidate(model, candidate, config=ie.EvalConfig(bootstrap_samples=200, random_control_seeds=(1,)))
    assert report["controls"]["noop"]["passed"] is True
    assert report["candidate"]["text"]["kl_nats_per_token"]["value"] > 0
    assert 1.0 < report["candidate"]["text"]["base_perplexity"]["value"] < 50304
