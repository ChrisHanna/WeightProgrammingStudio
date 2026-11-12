"""First public-proof plugins. Task-specific behavior stays outside the runner."""

from __future__ import annotations

import gc
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Self

from .gym_job import CandidateMethod, GymJobError
from .gym_service import _document_digest
from .provenance import sha256_file


def invoice_task(configuration: Mapping[str, Any]) -> Any:
    from .reference_json_extraction import reference_task

    task = reference_task(rows_per_template=configuration["rows_per_template"])
    methods = tuple(CandidateMethod(name, "1", task.identity.implementation_digest, {})
                    for name in configuration["methods"])
    original_bind = task.bind

    def bind(service: Any) -> None:
        original_bind(service)
        existing = {method.method_id for method in task.job.candidate_methods}
        for method in methods:
            if method.method_id not in existing:
                service.components.register("candidate_method", method.method_id, method.version,
                                            method.implementation_digest, lambda **_kwargs: None)

    return replace(task, job=replace(task.job, candidate_methods=methods), bind=bind)


def training_model(*, base_model: Mapping[str, Any], policy: Mapping[str, Any], seed: int) -> Any:
    from .gym_adapters import transformers_model

    return transformers_model(**base_model, device="cuda:0", precision="bfloat16",
                              context_tokens=policy["context_tokens"],
                              max_new_tokens=policy["max_new_tokens"], seed=seed)


def automodel_trainer(*, automodel_python: str, reload_python: str) -> Any:
    from .backends.automodel.adapter import automodel_sft_trainer
    from .backends.automodel.config import AutoModelBackendConfig

    for interpreter in (automodel_python, reload_python):
        if not interpreter or not Path(interpreter).is_absolute() or not Path(interpreter).is_file():
            raise GymJobError("approved runtime interpreter is missing")
    return automodel_sft_trainer(AutoModelBackendConfig(
        automodel_python, timeout_seconds=1800, reload_python_executable=reload_python))


def automodel_coverage(*, artifact: Path, roster: list[str], steps: int) -> dict[str, Any]:
    from .proof_pipeline import training_coverage

    receipt = json.loads((artifact / "training_receipt.json").read_bytes())
    config = json.loads((artifact / "automodel_config.json").read_bytes())
    if sha256_file(artifact / "automodel_config.json") != receipt.get("worker_config_digest"):
        raise GymJobError("training coverage configuration differs from the committed receipt")
    return training_coverage(roster, receipt, config, steps)


def balanced_augmentation(configuration: Mapping[str, Any]) -> list[dict[str, Any]]:
    from .json_authority_training import balanced_authority_rows

    if configuration.get("version") != 1:
        raise GymJobError("unsupported balanced-authority configuration")
    return balanced_authority_rows(seed=configuration["seed"])


def public_overlap_audit(*, spec: Mapping[str, Any]) -> dict[str, Any]:
    from .public_data_audit import overlap_counts

    config = spec["audit"]
    policy_path = Path(config["policy_path"])
    if sha256_file(policy_path) != config["policy_file_digest"]:
        raise GymJobError("public contamination policy changed")
    policy = json.loads(policy_path.read_bytes())
    if policy.get("kind") != "nf_contamination_policy" or policy.get("version") != 1:
        raise GymJobError("unsupported public contamination policy")
    train = spec["training_rows"]
    injected = [row for row in train if row["scenario_id"].startswith(
        ("train:injection-v1:", "train:authority-v1:", "train:factorial-authority-v1:"))]
    attack_phrases = {
        "alternate_record": "warehouse_note_replace_response",
        "quoted_vendor": "quoted_vendor_obsolete_report",
        "extra_property": "scanner_annotation_add_field",
        "quantity_override": "footer_instruction_override",
    }
    training_families = set()
    for row in injected:
        if row["scenario_id"].startswith(("train:authority-v1:", "train:factorial-authority-v1:")):
            training_families.add(row["authority"]["phrasing_family"])
        else:
            training_families.add(attack_phrases[row["scenario_id"].split(":")[3]])
    counts, checks = {}, {}
    for name in config["public_sets"]:
        panel = next(panel for panel in spec["public_sets"] if panel["name"] == name)
        required_scope = panel.get("audit_scope") == "required"
        # The approved same zero-exact/zero-fivegram rule also applies to the public panel.
        counts[name] = {"all_training": overlap_counts(train, panel["rows"]),
                        "injection": overlap_counts(injected, panel["rows"])}
        for scope, overlap in counts[name].items():
            rule = policy["all_training_vs_required" if scope == "all_training" else "injection_vs_required"]
            checks[f"{name}:{scope}"] = (
                overlap["exact_input_matches"] <= rule["maximum_exact_input_matches"]
                and overlap["ngram_overlap"]["5"]["distinct_shared_ngrams"]
                <= rule["maximum_distinct_shared_5grams"])
        if required_scope:
            required_families = set(panel["phrasing_families"])
            checks[f"{name}:phrasing_families"] = not (training_families & required_families)
    if not config["public_sets"] and train:
        # Compatibility preparations never become proof just because they lack a panel.
        checks["no_public_sets"] = False
    return {
        "kind": "nf_public_overlap_audit_v1", "public_passed": bool(checks) and all(checks.values()),
        "checks": checks, "overlap": counts, "policy_file_digest": config["policy_file_digest"],
        "training_rows_digest": spec["training_rows_digest"],
        "training_phrasing_families": sorted(training_families),
        "public_set_digests": {panel["name"]: panel["rows_digest"] for panel in spec["public_sets"]
                               if panel["name"] in config["public_sets"]},
        "audit_passed": False, "independent_custody_verified": False,
        "holdout_status": "not_checked_public_only_retired_synthetic_is_not_final_proof",
        "holdout_accessed": False, "release_authorized": False,
    }


def openai_provider(*, spec: Mapping[str, Any], candidate: Mapping[str, Any], root: Path,
                    task: Any = None, api_key: str | None = None) -> Any:
    from .openai_provider import OpenAIProvider
    from .proof_policy import require_bound_policy
    from .public_budget import PublicBudget

    if spec["policy"]["api"].get("calls_authorized") is not True:
        raise GymJobError("provider calls are not authorized by the frozen run policy")
    require_bound_policy(spec)
    PublicBudget(spec)
    if task is None:
        raise GymJobError("provider requires the selected Gym context")
    # Never read .env from this generic adapter; credentials are runtime-only.
    return OpenAIProvider(spec=spec, candidate=candidate, task=task,
                          api_key=api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", ""))


def api_evidence(*, model: Any, policy: Mapping[str, Any],
                 interaction: Mapping[str, Any] | None = None, result: Any = None) -> None:
    from .openai_provider import OpenAIProvider

    if type(model) is not OpenAIProvider:
        raise GymJobError("provider evidence requires the registered Gym-neutral client")
    model.require_configuration()
    if model.policy != policy:
        raise GymJobError("provider budgets or decoding differ from the run policy")
    if interaction is not None:
        model.verify_response_receipt(interaction, result)


class RecordedPublicProvider:
    """Replay exact, independently digest-checked public interactions, never new calls."""

    def __init__(self, *, spec: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
        from .proof_pipeline import read_document

        config = candidate["execution"]["configuration"]
        source = read_document(Path(config["source_card"]))
        if source["document_digest"] != config["source_document_digest"] or source.get("complete") is not True:
            raise GymJobError("recorded public response card changed or is incomplete")
        rows_digest = config["source_rows_digest"]
        panels = [panel for panel in spec["public_sets"] if panel["name"] in candidate["public_sets"]]
        if (len(panels) != 1 or panels[0]["rows_digest"] != rows_digest
            or source["task"]["panel_digest"] != rows_digest):
            raise GymJobError("recorded responses require identical public inputs and order")
        self.source = source
        self._responses: dict[str, Any] = {}
        self._used: set[str] = set()
        task = invoice_task(spec["gym"])
        if (source["frontier_configuration"]["max_new_tokens"] != spec["policy"]["max_new_tokens"]
            or source["frontier_configuration"]["scorer"]["implementation_digest"]
            != task.job.scorer.implementation_digest):
            raise GymJobError("recorded provider decoding or scorer identity differs")
        for row, record in zip(panels[0]["rows"], source["task"]["responses"], strict=True):
            if row["scenario_id"] != record["scenario_id"]:
                raise GymJobError("recorded public response order changed")
            receipt = record.get("execution_evidence", {}).get("api_response_receipt", {})
            interaction_digest = _document_digest(task.project(row["input"], None))
            if (receipt.get("receipt_digest") != _document_digest(
                    {key: value for key, value in receipt.items() if key != "receipt_digest"})
                or receipt.get("interaction_digest") != interaction_digest
                or receipt.get("response_digest") != _document_digest(record["response"])
                or receipt.get("configuration_digest") != source["frontier_configuration"]["configuration_digest"]
                or receipt not in source["model_identity"]["api_response_receipts"]):
                raise GymJobError("recorded public response lacks matching original receipt")
            self._responses[interaction_digest] = record

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        pass

    def infer(self, interaction: Mapping[str, Any]) -> Any:
        from .gym_service import ExecutionResponse

        key = _document_digest(interaction)
        if key not in self._responses or key in self._used:
            raise GymJobError("no unused exact public response exists")
        self._used.add(key)
        record = self._responses[key]
        return ExecutionResponse(record["response"], record["execution_evidence"])

    def loaded_identity(self) -> dict[str, Any]:
        return {"provider": "recorded_public_responses", "new_api_calls": 0,
                "source_document_digest": self.source["document_digest"],
                "source_panel_digest": self.source["panel_digest"],
                "source_rows_digest": self.source["panel_rows_digest"],
                "source_model_identity": self.source["model_identity"]}


def recorded_provider(*, spec: Mapping[str, Any], candidate: Mapping[str, Any], root: Path,
                      task: Any = None) -> Any:
    return RecordedPublicProvider(spec=spec, candidate=candidate)


def recorded_evidence(*, model: Any, policy: Mapping[str, Any],
                      interaction: Mapping[str, Any] | None = None, result: Any = None) -> None:
    if type(model) is not RecordedPublicProvider:
        raise GymJobError("recorded provider evidence requires a checked public source card")
    if interaction is not None:
        key = _document_digest(interaction)
        record = model._responses.get(key)
        if (record is None or result.response != record["response"]
            or dict(result.interaction) != record["execution_evidence"]):
            raise GymJobError("recorded response evidence changed")


def candidate_evidence(*, model: Any, policy: Mapping[str, Any],
                       interaction: Mapping[str, Any] | None = None, result: Any = None) -> None:
    from .json_proof_loop import _require_model_evidence

    if interaction is not None:
        return
    _require_model_evidence(model, "candidate")
    evidence = model.loaded_identity()["identity_probe_evidence"]
    if (evidence["fixed_prompts"] != policy["identity"]["probe_count"]
        or evidence["logprob_atol"] != policy["identity"]["logprob_atol"]):
        raise GymJobError("serving identity differs from the frozen probe policy")


def trained_reference(root: Path, role: str) -> tuple[Any, dict[str, Any]]:
    from .gym_artifacts import CandidateRef, read_candidate_manifest
    from .proof_pipeline import read_document

    result = read_document(root / role / "training-result.json", "result_digest")
    record = result["candidate_record"]
    artifact_root = Path(record["artifact_root"])
    for relative, expected in record["files"].items():
        path = (artifact_root / relative).resolve()
        if not path.is_relative_to(artifact_root.resolve()) or sha256_file(path) != expected:
            raise GymJobError("trained candidate file differs")
    directory = artifact_root / "artifact"
    manifest = read_candidate_manifest(directory, record["training_request_digest"])
    candidate = record["candidate"]
    if manifest["model_digest"] != candidate["candidate_model_digest"]:
        raise GymJobError("trained candidate state differs")
    reference = CandidateRef(result["base"]["model_id"], result["training_request"]["artifact_kind"],
        candidate["base_model_digest"], candidate["candidate_model_digest"], candidate["artifact_digest"],
        str(directory), candidate["candidate_id"], candidate["method_id"], result["base"]["family"])
    return reference, result


def serving_export(*, spec: Mapping[str, Any], role: str, root: Path) -> dict[str, Any]:
    from .gym_artifacts import CandidateRef, artifact_files, write_candidate_manifest
    from .proof_pipeline import write_document
    from .vllm_candidate_files import verify_vllm_candidate_files

    candidate = next(row for row in spec["candidates"] if row["role"] == role)
    if candidate["method"] == "unchanged_base":
        from .provenance import resolve_checkpoint

        model = spec["base_model"]
        checkpoint = resolve_checkpoint(model["model_id"], revision=model["revision"], allow_download=False)
        config = json.loads((checkpoint.path / "config.json").read_bytes())
        family = config.get("model_type")
        if not isinstance(family, str) or not family:
            raise GymJobError("base checkpoint has no declared model family")
        digest = _document_digest({"model_id": model["model_id"], "revision": model["revision"],
                                   "files": checkpoint.file_digests})
        reference = CandidateRef(model["model_id"], "full", digest, digest, digest,
                                 str(checkpoint.path), "base-" + digest.removeprefix("sha256:")[:16],
                                 "unchanged_base", family)
        verify_vllm_candidate_files(reference, revision=model["revision"])
        return write_document(root / f"export-{role}.json", {
            "kind": "nf_serving_export_v1", "role": role, "reference": asdict(reference),
            "revision": model["revision"], "run_spec_digest": _document_digest(spec),
            "source_kind": "unchanged_checkpoint", "source_result_digest": None,
            "source_reference": asdict(reference), "source_files": checkpoint.file_digests,
            "files_verified": True, "reloaded": False, "model_loading_performed": False,
            "holdout_accessed": False, "release_authorized": False})
    source, result = trained_reference(root, role)
    if result.get("run_spec_digest") != _document_digest(spec):
        raise GymJobError("training result belongs to a different run spec")
    reference = source
    if source.kind == "lora":
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .model_digest import model_state_digest

        base = CandidateRef(**result["base"])
        verify_vllm_candidate_files(base, revision=result["revision"])
        output = root / f"serving-{role}"
        output.mkdir(exist_ok=False)
        original = AutoModelForCausalLM.from_pretrained(base.artifact_path, local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0").eval()
        adapter = PeftModel.from_pretrained(original, source.artifact_path, local_files_only=True).eval()
        if model_state_digest(adapter) != source.model_digest:
            raise GymJobError("LoRA reload differs from frozen training result")
        merged = adapter.merge_and_unload(safe_merge=True).eval()
        merged_digest = model_state_digest(merged)
        merged.save_pretrained(output, safe_serialization=True)
        AutoTokenizer.from_pretrained(source.artifact_path, local_files_only=True).save_pretrained(output)
        del merged, adapter, original
        gc.collect()
        torch.cuda.empty_cache()
        reloaded = AutoModelForCausalLM.from_pretrained(output, local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0").eval()
        if model_state_digest(reloaded) != merged_digest:
            raise GymJobError("merged serving export reload changed its weights digest")
        del reloaded
        gc.collect()
        torch.cuda.empty_cache()
        request = result["training_request"]["training_request_digest"]
        receipt = {"export_kind": "peft_merged_full_v1", "training_request_digest": request,
            "model_digest": merged_digest, "reload_state_digest": merged_digest,
            "source_adapter_digest": source.artifact_digest, "source_candidate_model_digest": source.model_digest,
            "source_result_digest": result["result_digest"], "holdout_accessed": False, "release_authorized": False}
        (output / "training_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        artifact_digest = _document_digest({"training_request": request, "files": artifact_files(output)})
        reference = CandidateRef(base.model_id, "full", base.base_digest, merged_digest, artifact_digest,
            str(output), source.candidate_id + ":merged", source.method_id, base.family)
        write_candidate_manifest(output, model_id=base.model_id, kind="full", base_digest=base.base_digest,
            model_digest=merged_digest, request_digest=request, metadata={"candidate": {
                "candidate_id": reference.candidate_id, "method_id": reference.method_id,
                "base_model_digest": base.base_digest, "candidate_model_digest": merged_digest,
                "artifact_digest": artifact_digest}})
    elif source.kind != "full":
        raise GymJobError("serving exporter does not support this artifact kind")
    verify_vllm_candidate_files(reference, revision=result["revision"])
    return write_document(root / f"export-{role}.json", {
        "kind": "nf_serving_export_v1", "role": role, "reference": asdict(reference),
        "revision": result["revision"], "run_spec_digest": _document_digest(spec),
        "source_result_digest": result["result_digest"], "source_reference": asdict(source),
        "reloaded": True, "holdout_accessed": False, "release_authorized": False})


def vllm_server(*, spec: Mapping[str, Any], candidate: Mapping[str, Any],
                root: Path, freeze_identity: bool = False) -> Any:
    from .gym_artifacts import CandidateRef
    from .proof_pipeline import read_document, write_document
    from .vllm_container_adapter import VllmCandidateServer

    role, policy = candidate["role"], spec["policy"]
    exported = read_document(root / f"export-{role}.json")
    if exported["run_spec_digest"] != _document_digest(spec):
        raise GymJobError("serving export belongs to another run spec")
    reference = CandidateRef(**exported["reference"])
    configuration = candidate["execution"].get("configuration", {})
    if (not isinstance(configuration, Mapping)
        or set(configuration) - {"gpu_memory_utilization"}):
        raise GymJobError("unsupported local serving configuration")
    snapshot_root = root / "serving-snapshots"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    arguments = {"revision": exported["revision"], "context_tokens": policy["context_tokens"],
                 "max_new_tokens": policy["max_new_tokens"], "timeout_seconds": 480,
                 "snapshot_root": snapshot_root,
                 "gpu_memory_utilization": configuration.get("gpu_memory_utilization", 0.25)}
    identity_path = root / f"identity-{role}.json"
    if freeze_identity:
        server = VllmCandidateServer(reference, **arguments)
        with server:
            probes = server.record_identity_probes()
        return write_document(identity_path, {
            "reference": asdict(reference), "probe_record": probes,
            "run_spec_digest": _document_digest(spec), "holdout_accessed": False})
    frozen = read_document(identity_path)
    if frozen["reference"] != asdict(reference) or frozen["run_spec_digest"] != _document_digest(spec):
        raise GymJobError("frozen serving identity changed")
    return VllmCandidateServer(reference, identity_record=frozen["probe_record"], **arguments)
