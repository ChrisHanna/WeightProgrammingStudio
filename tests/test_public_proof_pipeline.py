from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
from neural_foundry.gym_job import ComponentReference, GymJobError
from neural_foundry.gym_lifecycle import AdapterIdentity
from neural_foundry.gym_service import ExecutionResponse, _document_digest
from neural_foundry.json_authority_training import balanced_authority_rows
from neural_foundry.json_public_run import public_policy
from neural_foundry.proof_pipeline import (
    read_document,
    score,
    training_coverage,
    training_task,
    validate_spec,
    write_document,
)
from neural_foundry.proof_registry import ProofRegistry, builtin_registry
from neural_foundry.reference_json_extraction import reference_task


def test_balanced_rows_have_shared_role_vocabularies_and_full_factorial_balance() -> None:
    rows = balanced_authority_rows(seed=20261113)
    assert rows == balanced_authority_rows(seed=20261113)
    assert rows != balanced_authority_rows(seed=20261114)
    assert len(rows) == len({row["input"]["text"] for row in rows}) == 64
    assert all(row["scenario_id"].startswith("train:") and len(row["input"]["text"]) <= 512 for row in rows)
    for field in ("customer", "product", "quantity"):
        assert Counter(row["reference"][field] for row in rows) == Counter(
            row["authority"]["decoy"][field] for row in rows)
    for field in ("location", "trusted_format", "decoy_format"):
        assert sorted(Counter(row["authority"][field] for row in rows).values()) == [32, 32]
    assert set(Counter(row["authority"]["disagreement"] for row in rows).values()) == {16}
    for row in rows:
        assert row["reference"] != row["authority"]["decoy"]


def test_registry_rejects_duplicate_unknown_and_changed_implementation() -> None:
    registry = ProofRegistry()
    identity = AdapterIdentity("custom", "1", _document_digest({"plugin": 1}))
    reference = registry.register("provider", identity, lambda: "ok")
    assert registry.resolve("provider", reference)() == "ok"
    with pytest.raises(GymJobError, match="duplicate"):
        registry.register("provider", identity, lambda: None)
    with pytest.raises(GymJobError, match="unknown"):
        registry.resolve("trainer", reference)
    with pytest.raises(GymJobError, match="changed"):
        registry.resolve("provider", {**reference, "implementation_digest": _document_digest({})})


def _custom_spec():
    registry = ProofRegistry()
    references = {}
    task = reference_task(rows_per_template=1)
    scorer = ComponentReference("custom_score", "1", task.identity.implementation_digest, {})
    original_bind = task.bind

    def bind(service):
        original_bind(service)
        service.components.register("scorer", scorer.component_id, scorer.version,
                                    scorer.implementation_digest,
                                    lambda *, reference, response, output_valid: {
                                        "semantic_success": float(output_valid and reference == response)})

    task = replace(task, job=replace(task.job, scorer=scorer), bind=bind)
    reference = {"customer": "Birch", "product": "lamps", "quantity": 3}
    public = [{"scenario_id": "validation:arbitrary:1", "input": {"text": "unused"},
               "reference": reference, "panel_category": "custom"}]

    class Server:
        def __enter__(self):
            return SimpleNamespace(
                infer=lambda _: ExecutionResponse(reference, {"receipt": "custom"}),
                loaded_identity=lambda: {"provider": "custom_contract_test"})

        def __exit__(self, *_args):
            pass

    def evidence(**kwargs):
        if kwargs.get("result") is not None:
            assert kwargs["result"].interaction["receipt"] == "custom"

    for kind, factory in (
        ("gym", lambda _: task), ("model", lambda **_: None),
        ("trainer", lambda **_: None), ("exporter", lambda **_: None),
        ("training_audit", lambda **_: {}),
        ("augmenter", lambda _: []),
        ("public_audit", lambda **_: {"public_passed": True}),
        ("provider", lambda **_: Server()), ("evidence", evidence),
    ):
        references[kind] = registry.register(kind, AdapterIdentity(
            f"custom-{kind}", "1", _document_digest({"kind": kind})), factory)
    spec = {
        "kind": "nf_public_proof_run", "version": 1, "run_id": "custom-run",
        "download_allowed": False, "holdout_accessed": False,
        "base_model": {"model_id": "explicit/custom", "revision": "pinned"},
        "policy": public_policy(), "gym": {},
        "plugins": {key: references[key] for key in
                    ("gym", "model", "trainer", "training_audit", "exporter", "augmenter", "public_audit")},
        "training_rows": [], "training_rows_digest": _document_digest({"rows": []}),
        "candidates": [{"role": "arbitrary-candidate", "method": "arbitrary-method",
                        "execution": {"kind": "provider", "plugin": references["provider"]},
                        "evidence": references["evidence"]}],
        "public_sets": [{"name": "arbitrary-set", "rows": public,
                         "rows_digest": _document_digest({"rows": public})}],
        "gates": [{"name": "arbitrary-gate", "set": "arbitrary-set",
                   "metric": "semantic_success", "minimum": 1}],
    }
    return spec, registry


def test_generic_runner_dispatches_custom_roles_provider_evidence_and_gym_scorer(tmp_path) -> None:
    spec, registry = _custom_spec()
    result = score(spec, role="arbitrary-candidate", root=tmp_path, registry=registry)
    assert result["gate_passed"]
    assert result["gates"]["arbitrary-gate"]["observed"] == 1
    assert result["scores"]["arbitrary-set"]["metrics"] == {"semantic_success": 1.0}
    assert result["report_only"] and not result["release_authorized"]
    assert read_document(tmp_path / "public-score-arbitrary-candidate.json") == result


def test_partial_panel_evidence_cannot_pass_full_frozen_gate(tmp_path) -> None:
    spec, registry = _custom_spec()
    second = deepcopy(spec["public_sets"][0])
    second["name"] = "missing"
    spec["public_sets"].append(second)
    spec["gates"].append({**spec["gates"][0], "name": "missing", "set": "missing"})
    spec["candidates"][0]["public_sets"] = ["arbitrary-set"]
    result = score(spec, role="arbitrary-candidate", root=tmp_path, registry=registry)
    assert not result["gates_complete"] and not result["gate_passed"]


@pytest.mark.parametrize("change", ["holdout", "role-path", "row-digest", "identity", "api-budget"])
def test_spec_refuses_unsafe_or_unfrozen_contracts(change) -> None:
    spec, registry = _custom_spec()
    if change == "holdout":
        spec["holdout_accessed"] = True
    elif change == "role-path":
        spec["candidates"][0]["role"] = "../escape"
    elif change == "row-digest":
        spec["public_sets"][0]["rows"][0]["input"]["text"] = "changed"
    elif change == "identity":
        spec["policy"]["identity"]["logprob_atol"] = 0.1
    else:
        spec["policy"]["api"]["maximum_api_attempts"] = 0
    with pytest.raises(GymJobError):
        validate_spec(spec, registry)


def test_scoring_refusal_writes_bounded_failure_receipt(tmp_path) -> None:
    spec, registry = _custom_spec()
    identity = AdapterIdentity("refusing", "1", _document_digest({"refusal": 1}))

    def refuse(**_kwargs):
        raise GymJobError("private diagnostic must not leak")

    spec["candidates"][0]["execution"]["plugin"] = registry.register("provider", identity, refuse)
    with pytest.raises(GymJobError, match="private diagnostic"):
        score(spec, role="arbitrary-candidate", root=tmp_path, registry=registry)
    failures = list(tmp_path.glob("*failure*.json"))
    assert len(failures) == 1
    result = read_document(failures[0])
    assert result["failure_type"] == "GymJobError" and not result["complete"]
    assert "private diagnostic" not in json.dumps(result)


def test_schedule_coverage_rejects_short_or_shuffled_training() -> None:
    roster = [f"train:row:{index}" for index in range(448)]
    config = {"dataloader": {"shuffle": False},
              "step_scheduler": {"global_batch_size": 1, "local_batch_size": 1, "max_steps": 448},
              "packed_sequence": {"packed_sequence_size": 0}}
    receipt = {"committed_steps": list(range(448))}
    result = training_coverage(roster, receipt, config, 448)
    assert result["complete"] and result["unique_rows_consumed"] == 448
    assert result["per_step_scenario_ids"] == roster
    assert not result["direct_per_batch_id_telemetry"]
    with pytest.raises(GymJobError, match="commit"):
        training_coverage(roster, {"committed_steps": list(range(447))}, config, 448)
    config["dataloader"]["shuffle"] = True
    with pytest.raises(GymJobError, match="schedule"):
        training_coverage(roster, receipt, config, 448)


def test_frozen_training_source_refuses_nontraining_materialization() -> None:
    from neural_foundry.gym_service import GymStudioService

    rows = balanced_authority_rows(seed=1)
    task = training_task(reference_task(rows_per_template=1), rows)
    service = GymStudioService()
    task.bind(service)
    source = task.job.scenario_sources[0]
    provider = service.components.resolve("scenario_provider", source.provider_id,
                                          source.provider_version, source.provider_digest)
    materialized = provider(split="train")
    assert [row["scenario_id"] for row in materialized] == [row["scenario_id"] for row in rows]
    assert all("authority" not in row for row in materialized)
    with pytest.raises(GymJobError, match="train-only"):
        provider(split="holdout")


def test_openai_plugin_refuses_unapproved_calls_without_reading_credentials(monkeypatch, tmp_path) -> None:
    from neural_foundry.proof_plugins import openai_provider

    monkeypatch.setattr("os.environ", {})
    with pytest.raises(GymJobError, match="not authorized"):
        openai_provider(spec={"policy": public_policy()}, candidate={}, root=tmp_path)


def test_document_tampering_is_rejected(tmp_path) -> None:
    path = tmp_path / "spec.json"
    write_document(path, {"kind": "test", "value": 1})
    document = json.loads(path.read_bytes())
    document["value"] = 2
    path.write_text(json.dumps(document))
    with pytest.raises(GymJobError, match="digest differs"):
        read_document(path)


def test_builtin_registry_has_no_study_custody_dependency() -> None:
    registry, references = builtin_registry()
    assert callable(registry.resolve("trainer", references["trainer"]))
    assert callable(registry.resolve("exporter", references["exporter"]))
    assert callable(registry.resolve("provider", references["provider"]))


def test_new_authority_examples_pass_required_and_panel_overlap_rules() -> None:
    from neural_foundry.json_injection_training_data import generate_public_validation_v2_rows
    from neural_foundry.json_proof_loop import public_required_rows
    from neural_foundry.public_data_audit import overlap_counts

    rows = balanced_authority_rows(seed=20261113)
    for public in (public_required_rows(seed=17),
                   generate_public_validation_v2_rows(seed=20261112, examples_per_template=32)):
        counts = overlap_counts(rows, public)
        assert counts["exact_input_matches"] == 0
        assert counts["ngram_overlap"]["5"]["distinct_shared_ngrams"] == 0


def test_full_serving_export_preserves_lineage_and_requires_source_spec(monkeypatch, tmp_path) -> None:
    from neural_foundry import proof_plugins, vllm_candidate_files
    from neural_foundry.gym_artifacts import CandidateRef

    spec, _ = _custom_spec()
    digest = _document_digest({"weights": 1})
    reference = CandidateRef("test/model", "full", digest, digest, digest,
                             str(tmp_path / "weights"), "candidate", "custom", "qwen3")
    result = {"run_spec_digest": _document_digest(spec), "revision": "pinned",
              "result_digest": _document_digest({"source": 1})}
    monkeypatch.setattr(proof_plugins, "trained_reference", lambda *_: (reference, result))
    verified = []
    monkeypatch.setattr(vllm_candidate_files, "verify_vllm_candidate_files",
                        lambda value, **_: verified.append(value))
    output = proof_plugins.serving_export(spec=spec, role="arbitrary-candidate", root=tmp_path)
    assert verified == [reference] and output["reference"]["method_id"] == "custom"
    assert output["source_result_digest"] == result["result_digest"] and output["reloaded"]
    result["run_spec_digest"] = _document_digest({"different": 1})
    with pytest.raises(GymJobError, match="different run spec"):
        proof_plugins.serving_export(spec=spec, role="arbitrary-candidate", root=tmp_path)


def test_server_freezes_and_reuses_identity_bound_to_export_and_policy(monkeypatch, tmp_path) -> None:
    from dataclasses import asdict

    from neural_foundry import proof_plugins, vllm_container_adapter
    from neural_foundry.gym_artifacts import CandidateRef

    spec, _ = _custom_spec()
    digest = _document_digest({"weights": 1})
    reference = CandidateRef("test/model", "full", digest, digest, digest,
                             str(tmp_path / "weights"), "candidate", "custom", "qwen3")
    write_document(tmp_path / "export-arbitrary-candidate.json", {
        "reference": asdict(reference), "revision": "pinned", "run_spec_digest": _document_digest(spec)})
    calls = []

    class Server:
        def __init__(self, value, **kwargs):
            assert value == reference
            calls.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def record_identity_probes(self):
            return {"fixed": "probes"}

    monkeypatch.setattr(vllm_container_adapter, "VllmCandidateServer", Server)
    candidate = spec["candidates"][0]
    proof_plugins.vllm_server(spec=spec, candidate=candidate, root=tmp_path, freeze_identity=True)
    proof_plugins.vllm_server(spec=spec, candidate=candidate, root=tmp_path)
    assert calls[1]["identity_record"] == {"fixed": "probes"}
    assert calls[1]["max_new_tokens"] == spec["policy"]["max_new_tokens"]
    assert calls[1]["snapshot_root"].is_relative_to(tmp_path)
    spec["policy"]["max_new_tokens"] = 64
    with pytest.raises(GymJobError, match="another run spec"):
        proof_plugins.vllm_server(spec=spec, candidate=candidate, root=tmp_path)


def test_capped_output_cannot_pass_even_when_response_matches_reference() -> None:
    from neural_foundry.proof_pipeline import score_public_rows

    task = reference_task(rows_per_template=1)
    rows = [{"scenario_id": "validation:cap:1", "input": {"text": "unused"},
             "reference": {"customer": "Birch", "product": "lamps", "quantity": 3}}]
    model = SimpleNamespace(infer=lambda _: ExecutionResponse(
        rows[0]["reference"], {"hit_generation_cap": True}))
    result = score_public_rows(task, model, rows)
    assert result["metrics"]["exact_match"] == 0.0
    assert not result["responses"][0]["format_valid"]


def test_public_command_wrapper_uses_per_job_status_not_global(tmp_path, monkeypatch) -> None:
    import runpy
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_public_proof.py"
    module = runpy.run_path(str(script))
    runtime = tmp_path / "python"
    runtime.write_text("test runtime")
    monkeypatch.chdir(tmp_path)
    log_dir = Path("one-job")
    real_status = module["main"].__globals__["write_status"]

    def absolute_status(path, **kwargs):
        assert path.is_absolute()
        return real_status(path, **kwargs)

    monkeypatch.setitem(module["main"].__globals__, "write_status", absolute_status)
    monkeypatch.setattr(sys, "argv", [str(script), "audit", "--spec", "spec.json",
                                    "--root", str(tmp_path / "run"), "--python", str(runtime),
                                    "--log-dir", str(log_dir)])
    monkeypatch.setitem(module["main"].__globals__, "run_logged_worker",
                        lambda *_, **__: SimpleNamespace(
                            receipt={"status": "completed"}, returncode=0, timed_out=False))
    with pytest.raises(SystemExit) as stopped:
        module["main"]()
    assert stopped.value.code == 0
    assert json.loads((tmp_path / "one-job-status.json").read_bytes())["state"] == "done"
    assert not (tmp_path / "STATUS.json").exists()
    with pytest.raises(SystemExit) as repeated:
        module["main"]()
    assert repeated.value.code == 2
    assert json.loads((tmp_path / "one-job-status.json").read_bytes())["state"] == "done"


def test_training_refuses_failed_public_audit_before_creating_output(tmp_path) -> None:
    from neural_foundry.proof_pipeline import train

    spec, registry = _custom_spec()
    spec["plugins"]["public_audit"] = registry.register("public_audit", AdapterIdentity(
        "blocked-public", "1", _document_digest({"blocked": True})),
        lambda **_: {"public_passed": False})
    with pytest.raises(GymJobError, match="public contamination gate"):
        train(spec, role="arbitrary-candidate", root=tmp_path / "uncreated",
              automodel_python="not-loaded", reload_python="not-loaded", registry=registry)
    assert not (tmp_path / "uncreated").exists()


def test_frozen_source_materializes_exact_448_row_order_in_gym(tmp_path) -> None:
    from neural_foundry.gym_service import GymStudioService
    from neural_foundry.json_injection_training_data import (
        generate_train_injection_rows,
        interleave_train_injection_rows,
    )
    from neural_foundry.reference_json_extraction import generate_scenarios

    original = interleave_train_injection_rows(
        generate_scenarios(split="train", seed=17, parameters={"rows_per_template": 32}),
        generate_train_injection_rows(seed=20261111))
    new = balanced_authority_rows(seed=20261113)
    rows = []
    for index, row in enumerate(original):
        rows.append(row)
        if index % 6 == 5:
            rows.append(new[index // 6])
    task = training_task(reference_task(rows_per_template=32), rows)
    service = GymStudioService(state_dir=tmp_path)
    task.bind(service)
    manifest = service.materialize_public_scenarios(task.job, split="train", seed=17)
    assert len(manifest["scenarios"]) == 448
    assert [row["scenario_id"] for row in manifest["scenarios"]] == [row["scenario_id"] for row in rows]


def test_local_response_evidence_does_not_rehash_all_model_weights_per_response(monkeypatch) -> None:
    from neural_foundry import json_proof_loop
    from neural_foundry.proof_plugins import candidate_evidence

    def forbidden(*_args):
        raise AssertionError("candidate weight identity is checked before and after the public session")

    monkeypatch.setattr(json_proof_loop, "_require_model_evidence", forbidden)
    candidate_evidence(model=None, policy=public_policy(), interaction={"request": "public"},
                       result=ExecutionResponse({}, {}))


def test_v2_training_contract_removes_old_injection_and_rejects_base_collisions(tmp_path) -> None:
    import runpy
    from pathlib import Path

    from neural_foundry.json_injection_training_data import generate_public_validation_v2_rows
    from neural_foundry.json_public_run import freeze_balanced_spec
    from neural_foundry.proof_pipeline import audit
    from neural_foundry.reference_json_extraction import TEMPLATES

    project = Path(__file__).resolve().parents[1]
    legacy = runpy.run_path(str(project / "scripts" / "run_gym_lifecycle_real.py"))
    preparation = legacy["prepare_json_robustness_v2_training_request"]()
    preparation_path = tmp_path / "preparation.json"
    preparation_path.write_text(json.dumps(preparation))
    panel_rows = generate_public_validation_v2_rows(seed=20261112)
    panel_path = tmp_path / "panel.json"
    write_document(panel_path, {"kind": "nf_json_public_validation_panel_v3", "rows": panel_rows,
                               "rows_digest": _document_digest({"rows": panel_rows})})
    kwargs = {"preparation": preparation_path, "panel": panel_path,
              "policy_path": project / "protocols" / "json-contamination-v1.json",
              "training_contract_version": 2}
    spec = freeze_balanced_spec(**kwargs)
    assert spec == freeze_balanced_spec(**kwargs)
    contract = spec["training_contract"]
    assert contract["version"] == 2 and contract["old_injection_rows_removed"] == 64
    assert contract["rejection_sampling"]["excluded_candidates"] >= 12
    assert contract["rejection_sampling"]["replaced_source_rows"] == 12
    assert contract["rejection_sampling"]["retained_source_rows"] == 308
    assert len(spec["training_rows"]) == 384
    assert all(row["recipe"]["steps"] == 384 for row in spec["candidates"])
    assert not any(row["scenario_id"].startswith("train:injection-v1:") for row in spec["training_rows"])
    for row in spec["training_rows"]:
        if not row["scenario_id"].startswith("train:authority-v1:"):
            assert row["input"]["text"] == TEMPLATES["train"][row["family_id"]].format(**row["reference"])
    assert audit(spec, builtin_registry()[0])["public_passed"]
    assert [gate["minimum"] for gate in spec["gates"]] == [16, 91, 150]


def test_local_infrastructure_retry_preserves_failed_claim_and_cannot_replace_score(tmp_path) -> None:
    spec, registry = _custom_spec()
    candidate = spec["candidates"][0]
    original = registry.resolve("provider", candidate["execution"]["plugin"])
    broken = [True]

    def factory(**kwargs):
        if broken[0]:
            raise FileNotFoundError("startup-only fixture")
        return original(**kwargs)

    candidate["execution"] = {"kind": "server", "plugin": registry.register(
        "server", AdapterIdentity("local-startup", "1", _document_digest({"startup": 1})), factory)}
    with pytest.raises(FileNotFoundError):
        score(spec, role=candidate["role"], root=tmp_path, registry=registry)
    failure = next(tmp_path.glob("*failure*.json"))
    assert read_document(failure)["failed_stage"] == "server_startup"
    broken[0] = False
    result = score(spec, role=candidate["role"], root=tmp_path, registry=registry,
                   retry_of_failure=failure)
    assert result["gate_passed"] and result["retry_of_failure_digest"] == read_document(failure)["document_digest"]
    assert len(list(tmp_path.glob("*.started.json"))) == 2
    with pytest.raises(GymJobError, match="cannot be replaced"):
        score(spec, role=candidate["role"], root=tmp_path, registry=registry,
              retry_of_failure=failure)
