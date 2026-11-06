"""Checkpoint resolution, repair dispatch, evidence grades, provenance, schema, scan, serving and editor tools."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest
import torch

from neural_foundry import evals, export_formats, flagship, patch_ux, provenance, pythia, serving, studio

ST = studio.WeightProgrammingStudio()


def test_named_model_is_never_replaced_by_a_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    with pytest.raises(provenance.CheckpointUnavailableError):
        ST.select_model("Qwen/Qwen2.5-0.5B")


def test_cache_resolution_records_revision_and_file_digests(tmp_path: Path) -> None:
    repo = tmp_path / "models--org--name"
    snapshot = repo / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("abc123")
    (snapshot / "config.json").write_text("{}")
    found = provenance.resolve_checkpoint("org/name", cache_dir=tmp_path)
    assert found.revision == "abc123" and found.source == "huggingface_cache"
    assert found.file_digests["config.json"].startswith("sha256:")
    with pytest.raises(provenance.CheckpointUnavailableError):
        provenance.resolve_checkpoint("org/name", revision="other", cache_dir=tmp_path)


def test_unknown_failure_is_refused() -> None:
    with pytest.raises(ValueError, match="No repair method"):
        ST.repair("nf/mini-pythia-fixture", failure="hallucination")


def test_looping_uses_the_attractor_method_and_reports_nothing_found() -> None:
    result = ST.repair("nf/mini-pythia-fixture", failure="looping")
    assert result["method"] == "attractor_projection"
    assert result["evidence_grade"] == "fixture_synthetic"
    if result["open_patch"] is None:
        assert result["verification"]["rejection_reason"].startswith("no_")


def test_cli_exit_code_reflects_rejection(tmp_path: Path) -> None:
    code = studio.main(["repair", "--model", "Qwen/qwen2-dense-mini", "--failure", "parity"])
    assert code == studio.REPAIR_REJECTED_EXIT_CODE


def test_evidence_grades() -> None:
    suite = evals.import_suite(
        {
            "suite_id": "s", "version": "1", "evaluator_kind": "deterministic_scorer", "scorer_id": "x",
            "scorer_version": "1", "eval_types": ["baseline"],
            "tasks": [{"task_id": "t", "eval_type": "baseline", "capability": "c", "input": "i", "expected": "o"}],
        }
    )
    grade = lambda m: evals.run_eval(model=m, suite=suite)["score_provenance"]["evidence_grade"]  # noqa: E731
    assert grade({"model_id": "m", "task_scores": {"t": 1.0}}) == "supplied_precomputed"
    assert grade({"model_id": "m", "predict_fn": lambda p: "o"}) == "measured_caller_callable"
    assert provenance.evidence_grade_for_model(pythia.build_mini_pythia_model()) in provenance.EVIDENCE_GRADES


def test_artifact_manifest_detects_swapped_files(tmp_path: Path) -> None:
    key = patch_ux.generate_ed25519_keypair()
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "config.json").write_text("{}")
    provenance.write_artifact_manifest(tmp_path, signing_key=key["private_key"], signer_id="s", open_patch_digest="sha256:" + "a" * 64)
    assert provenance.verify_artifact_manifest(tmp_path, public_key=key["public_key"])["files_verified"] == 2
    (tmp_path / "model.safetensors").write_bytes(b"evil")
    with pytest.raises(ValueError, match="mismatch"):
        provenance.verify_artifact_manifest(tmp_path, public_key=key["public_key"])
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    with pytest.raises(ValueError):
        provenance.verify_artifact_manifest(tmp_path, public_key=patch_ux.generate_ed25519_keypair()["public_key"])


def test_flagship_replication_rejects_a_swapped_weight_file(tmp_path: Path) -> None:
    model = pythia.build_mini_pythia_model()
    report = flagship.run_flagship_loop_repair_study_and_publish_bundle(model, output_dir=tmp_path)
    with open(tmp_path / "huggingface_export" / "model.safetensors", "r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"\x7f")
    with pytest.raises(ValueError, match="mismatch"):
        flagship.replicate_published_flagship_bundle(tmp_path, public_key=report.public_key, reference_model=model)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b.update(extra=1),
        lambda b: b["sbom"].update(base_checkpoint_digest="sha256:" + "9" * 64),
        lambda b: b["sbom"].update(changed_parameters=10_000),
        lambda b: b.update(candidate_checkpoint_digest="not-a-digest"),
        lambda b: b.update(operations=[]),
        lambda b: b["evaluation_provenance"]["controls"].update(noop_accuracy=2.0),
    ],
)
def test_schema_rejects_inconsistent_patches_in_python_and_rust(tmp_path: Path, mutate: object) -> None:
    import os
    import subprocess

    from neural_foundry.records import canonicalize
    from test_evidence_integrity import _signed_open_patch, _valid_body

    key = patch_ux.generate_ed25519_keypair()
    body = _valid_body()
    mutate(body)  # type: ignore[operator]
    patch = _signed_open_patch(body, key)
    with pytest.raises(ValueError):
        patch_ux.verify_open_patch_v1(patch, public_key=key["public_key"])
    path = tmp_path / "p.json"
    path.write_bytes(canonicalize(patch))
    cli = os.environ.get("NF_CLI", "target/debug/nf")
    assert subprocess.run([cli, "verify", str(path), "--pubkey", key["public_key"]], capture_output=True, check=False).returncode != 0


def test_scan_limits_scale_and_report_their_limits() -> None:
    model = pythia.build_mini_pythia_model()
    report = patch_ux.scan_patch_for_weight_anomalies(model, model, declared_target_sites=[]).to_dict()
    assert report["scope"] == "anomaly_sanity_check_not_backdoor_proof"
    assert "unlisted_triggers" in report["coverage"]["not_covered"]
    assert report["policy_version"] == patch_ux.SCAN_POLICY_VERSION

    from neural_foundry.export_formats import _apply_state_tensors_to_clone, _extract_model_state_tensors

    site = model.mlp_down_proj_site(1)
    state = _extract_model_state_tensors(model)
    state[site] = state[site] * 1000.0
    spiked = _apply_state_tensors_to_clone(model, state)
    flagged = patch_ux.scan_patch_for_weight_anomalies(model, spiked, declared_target_sites=[site])
    assert any("spike" in reason for reason in flagged.flagged_reasons)


def test_reload_uses_memory_mapping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = pythia.build_mini_pythia_model()
    export_formats.export_huggingface_safetensors(model, tmp_path, dtype="f32")
    original = Path.read_bytes

    def refuse(self: Path) -> bytes:
        assert self.suffix != ".safetensors", "checkpoint was read whole"
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", refuse)
    _, receipt = export_formats.reload_and_verify_huggingface_safetensors(tmp_path, model, ["Hello world"])
    assert receipt.file_digest.startswith("sha256:")


def test_llama_adapter_runs_a_real_subprocess_and_checks_identity(tmp_path: Path) -> None:
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF")
    binary = tmp_path / "llama-cli"
    binary.write_text("#!/bin/sh\necho ' four '\n")
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    adapter = serving.LlamaCppAdapter(gguf, binary=str(binary))
    result = adapter.generate("2+2=", {"max_tokens": 4})
    assert result.text == "four" and result.runtime == "llama.cpp" and result.sampler["max_tokens"] == 4
    with pytest.raises(ValueError, match="identity"):
        serving.LlamaCppAdapter(gguf, binary=str(binary), expected_digest="sha256:" + "0" * 64)
    with pytest.raises(ValueError, match="Unsupported"):
        adapter.generate("x", {"bogus": 1})
    suite = evals.import_suite(
        {
            "suite_id": "s", "version": "1", "evaluator_kind": "deterministic_scorer", "scorer_id": "x",
            "scorer_version": "1", "eval_types": ["baseline"],
            "tasks": [{"task_id": "t", "eval_type": "baseline", "capability": "c", "input": "2+2=", "expected": "four"}],
        }
    )
    run = evals.run_eval(model={"model_id": "m", "serving_adapter": adapter}, suite=suite)
    assert run["score_provenance"]["evidence_grade"] == "measured_live_runtime"


def test_missing_runtime_refuses_instead_of_falling_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"x")
    real = serving.importlib.import_module

    def missing(name: str) -> object:
        if name in {"vllm", "transformers"}:
            raise ImportError(name)
        return real(name)

    monkeypatch.setattr(serving.importlib, "import_module", missing)
    for adapter in (serving.VllmAdapter, serving.TransformersAdapter):
        with pytest.raises(serving.ServingRuntimeUnavailableError):
            adapter(tmp_path)


class _FakeVllm:
    """Stands in for the vllm module: records the variable at import and at engine start. No GPU, no server."""

    def __init__(self, error: Exception | None = None) -> None:
        self.seen: list[str | None] = []
        self.error = error

    def LLM(self, **_: object) -> object:  # noqa: N802
        self.seen.append(os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"))
        if self.error is not None:
            raise self.error
        return object()


def _fake_vllm_import(monkeypatch: pytest.MonkeyPatch, fake: _FakeVllm) -> None:
    def load(name: str) -> object:
        assert name == "vllm"
        fake.seen.append(os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"))
        return fake

    monkeypatch.setattr(serving.importlib, "import_module", load)


@pytest.mark.parametrize("before", [None, "1"], ids=["unset", "caller_set"])
def test_vllm_adapter_turns_off_flashinfer_only_during_startup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, before: str | None) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    fake = _FakeVllm()
    _fake_vllm_import(monkeypatch, fake)
    if before is None:
        monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)
    else:
        monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", before)
    serving.VllmAdapter(tmp_path)
    assert fake.seen == ["0", "0"]
    assert os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") == before


def test_vllm_adapter_reports_missing_ninja_and_still_restores_the_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    _fake_vllm_import(monkeypatch, _FakeVllm(FileNotFoundError(2, "No such file or directory", "ninja")))
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "1")
    with pytest.raises(serving.ServingRuntimeUnavailableError, match="needs ninja"):
        serving.VllmAdapter(tmp_path)
    assert os.environ["VLLM_USE_FLASHINFER_SAMPLER"] == "1"


def test_vllm_adapter_does_not_relabel_other_missing_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    _fake_vllm_import(monkeypatch, _FakeVllm(FileNotFoundError(2, "No such file or directory", "config.json")))
    with pytest.raises(FileNotFoundError, match="config.json"):
        serving.VllmAdapter(tmp_path)


def test_editor_transaction_lifecycle() -> None:
    tool = ST.editor_tool
    repaired = tool.call_tool("nf_repair_model", {"model": "Qwen/qwen2-dense-mini", "failure": "parity"})
    json.dumps({k: v for k, v in repaired.items() if k != "open_patch"})
    transaction = repaired["transaction_id"]
    assert transaction
    with pytest.raises(ValueError, match="rejected"):
        tool.call_tool("nf_apply_patch", {"transaction_id": transaction})
    compared = tool.call_tool("nf_compare_patch", {"transaction_id": transaction, "prompts": ["Hello world"]})
    assert compared["changed_parameters"] > 0
    assert tool.call_tool("nf_rollback_patch", {"transaction_id": transaction})["rollback_verified"] is True
    assert tool.call_tool("nf_compare_patch", {"transaction_id": transaction})["changed_parameters"] == 0
    assert "nf_export_bundle" not in {t["name"] for t in tool.tool_schemas()}
    with pytest.raises(ValueError, match="Unknown"):
        tool.call_tool("nf_apply_patch", {"transaction_id": "nope"})
    assert torch.is_tensor(ST.select_model("Qwen/qwen2-dense-mini").tensors[ST.select_model("Qwen/qwen2-dense-mini").mlp_down_proj_site(0)])


def _clone_with(model, **changes):  # type: ignore[no-untyped-def]
    from neural_foundry.export_formats import _apply_state_tensors_to_clone, _extract_model_state_tensors

    state = _extract_model_state_tensors(model)
    state.update(changes)
    return _apply_state_tensors_to_clone(model, state), state


def test_diff_rejects_shape_changes_and_missing_tensors_and_reports_added() -> None:
    model = pythia.build_mini_pythia_model()
    site = model.mlp_down_proj_site(1)
    edited = pythia.build_mini_pythia_model()
    edited.tensors = {**model.tensors, site: torch.zeros(3, 3, dtype=torch.float64)}
    with pytest.raises(ValueError, match="changed shape"):
        patch_ux.diff_model_weights(model, edited)
    edited.tensors = {k: v for k, v in model.tensors.items() if k != site}
    with pytest.raises(ValueError, match="missing"):
        patch_ux.diff_model_weights(model, edited)
    edited.tensors = {**model.tensors, "extra.weight": torch.ones(2, 2, dtype=torch.float64)}
    diff = patch_ux.diff_model_weights(model, edited)
    assert diff.added_tensors == ("extra.weight",) and "extra.weight" in diff.layers_touched


def test_q4_k_is_not_claimed(tmp_path: Path) -> None:
    model = pythia.build_mini_pythia_model()
    with pytest.raises(ValueError, match="not supported"):
        export_formats.export_gguf_v3(model, tmp_path / "x.gguf", quantization="q4_k_m")
    assert export_formats.NF_TYPE_Q4_BLOCK > 1000


def test_tokenizer_assets_copied_and_publishing_needs_them(tmp_path: Path) -> None:
    model = pythia.build_mini_pythia_model()
    source = tmp_path / "source"
    source.mkdir()
    (source / "tokenizer.json").write_text("{}")
    out = tmp_path / "out"
    export_formats.export_huggingface_safetensors(model, out, source_checkpoint_dir=source)
    assert (out / "tokenizer.json").is_file()
    key = patch_ux.generate_ed25519_keypair()
    bundle = export_formats.export_huggingface_hub_bundle(
        model, tmp_path / "bare", open_patch={"patch_digest": "sha256:" + "a" * 64}, signing_key=key["private_key"]
    )
    assert bundle["tokenizer_included"] is False
    with pytest.raises(ValueError, match="no tokenizer"):
        export_formats.export_huggingface_hub_bundle(
            model, tmp_path / "bare2", open_patch={"patch_digest": "sha256:" + "a" * 64},
            signing_key=key["private_key"], repo_id="o/r", push_to_hub=True,
        )


def test_answer_match_needs_a_whole_leading_word() -> None:
    assert evals._answer_matches(" 4.", "4")
    assert not evals._answer_matches("40", "4")
    assert not evals._answer_matches("not 4", "4")


def test_unnamed_model_gets_no_real_model_grade() -> None:
    model = pythia.build_mini_pythia_model()
    model.model_name = "acme/huge-model"
    assert provenance.evidence_grade_for_model(model) == "unverified_model"


def test_manifest_rejects_added_files_and_a_different_patch(tmp_path: Path) -> None:
    key = patch_ux.generate_ed25519_keypair()
    (tmp_path / "weight_patch.nf.json").write_text(json.dumps({"patch_digest": "sha256:" + "a" * 64}))
    provenance.write_artifact_manifest(tmp_path, signing_key=key["private_key"], signer_id="s", open_patch_digest="sha256:" + "b" * 64)
    with pytest.raises(ValueError, match="not the patch"):
        provenance.verify_artifact_manifest(tmp_path, public_key=key["public_key"])
    (tmp_path / "planted.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="unlisted"):
        provenance.verify_artifact_manifest(tmp_path, public_key=key["public_key"])


def test_receipt_names_what_the_adapter_actually_ran(tmp_path: Path) -> None:
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF")
    binary = tmp_path / "llama-cli"
    binary.write_text("#!/bin/sh\necho four\n")
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    adapter = serving.LlamaCppAdapter(gguf, binary=str(binary))
    suite = evals.import_suite(
        {
            "suite_id": "s", "version": "1", "evaluator_kind": "deterministic_scorer", "scorer_id": "x",
            "scorer_version": "1", "eval_types": ["baseline"],
            "tasks": [{"task_id": "t", "eval_type": "baseline", "capability": "c", "input": "q", "expected": "four"}],
        }
    )
    run = evals.run_eval(model={"model_id": "m", "runtime": "vllm", "checkpoint_digest": "sha256:" + "0" * 64, "serving_adapter": adapter}, suite=suite)
    assert run["score_provenance"]["runtime"] == "llama.cpp"
    assert run["score_provenance"]["checkpoint_digest_source"] == "serving_adapter_verified_file"
