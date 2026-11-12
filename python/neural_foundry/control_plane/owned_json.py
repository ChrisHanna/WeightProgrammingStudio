"""Owned flat JSON tasks and a local, candidate-bound Transformers proof adapter."""

from __future__ import annotations

import gc
from collections.abc import Callable, Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from ..gym_job import (
    AcceptanceGate,
    CandidateMethod,
    ComponentReference,
    GymJobError,
    GymJobSpec,
    ScenarioSource,
    validate_task_instance,
)
from ..gym_lifecycle import AdapterIdentity, TaskAdapter
from ..gym_service import GymStudioService, _document_digest
from ..provenance import sha256_file
from .models import encode, fail


def output_schema(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Deliberately narrow qualification: identical keys and scalar types only."""
    properties: dict[str, Any] = {}
    for row in examples:
        expected = row["expected"]
        if not expected or len(expected) > 16:
            fail("unsupported_task", "Owned JSON requires one to sixteen flat scalar fields.")
        current: dict[str, Any] = {}
        for key, value in expected.items():
            if not isinstance(key, str) or not key or len(key) > 64:
                fail("unsupported_task", "JSON field names must be bounded nonempty strings.")
            if type(value) is str and len(value) <= 256:
                current[key] = {"type": "string", "maxLength": 256}
            elif type(value) is bool:
                current[key] = {"type": "boolean"}
            elif type(value) is int and abs(value) <= 1_000_000:
                current[key] = {"type": "integer", "minimum": -1_000_000, "maximum": 1_000_000}
            else:
                fail("unsupported_task", "Only bounded string, boolean and integer JSON fields are qualified.")
        if properties and current != properties:
            fail("unsupported_task", "All examples must use identical keys and scalar types.")
        properties = current
    return {"type": "object", "properties": properties, "required": sorted(properties),
            "additionalProperties": False}


def owned_task(configuration: Mapping[str, Any]) -> TaskAdapter:
    schema = configuration["output_schema"]
    implementation = sha256_file(__file__)
    identity = _document_digest({"implementation": implementation, "configuration": configuration})
    input_schema = {"type": "string", "maxLength": 8192}
    source = ScenarioSource("owned-project-train", "owned-json-train", "1", identity,
                            identity, "real", ("train", "validation", "holdout"), {})
    scorer = ComponentReference("owned-json-exact", "1", identity, {})
    verifier = ComponentReference("owned-json-valid-exact", "1", identity, {})
    methods = tuple(CandidateMethod(name, "1", identity, {}) for name in ("unchanged_base", "lora"))
    job = GymJobSpec(
        job_id="owned-json-v1", title="Owned flat JSON task",
        description="Exact JSON scoring on immutable project-owned validation examples.",
        task_input_schema=input_schema, task_output_schema=schema, scenario_sources=(source,),
        scorer=scorer, verifier=verifier, candidate_methods=methods,
        # GymJobSpec includes the full research gate vocabulary. This adapter only
        # scores owned validation and never asserts holdout/control/repeat passage.
        gates=(AcceptanceGate("task", "task", "exact_match", "validation", "ge", 1.0),
               AcceptanceGate("behavior", "behavior", "exact_match", "holdout", "ge", 1.0),
               AcceptanceGate("controls", "controls", "beats_control", "controls", "eq", 1.0),
               AcceptanceGate("repeat", "reproducibility", "repeat", "reproducibility", "eq", 1.0)),
        metadata={"evidence_class": "owned", "holdout_accessed": False,
                  "complete_gym_gates_evaluated": False})

    def project(input_value: Any, response: Any) -> Mapping[str, Any]:
        validate_task_instance(input_schema, input_value, field="input")
        messages = [{"role": "system", "content": "Return only a JSON object matching this schema: "
                     + encode(schema).decode()}, {"role": "user", "content": input_value}]
        if response is not None:
            validate_task_instance(schema, response, field="response")
            messages.append({"role": "assistant", "content": encode(response).decode()})
        return {"format": "json_extraction_v1", "messages": messages, "output_schema": schema}

    def verify(*, reference: Any, response: Any, output_valid: bool = True,
               **_kwargs: Any) -> dict[str, Any]:
        try:
            validate_task_instance(schema, response, field="response")
            valid = output_valid and encode(reference) == encode(response)
        except GymJobError:
            valid = False
        return {"passed": valid}

    def score(**kwargs: Any) -> dict[str, float]:
        return {"exact_match": float(verify(**kwargs)["passed"])}

    def provider(**_kwargs: Any) -> Any:
        raise GymJobError("owned task scenarios must come from the frozen train-only roster")

    def bind(service: GymStudioService) -> None:
        service.components.register("scenario_provider", source.provider_id, "1", identity, provider)
        service.components.register("scorer", scorer.component_id, "1", identity, score)
        service.components.register("verifier", verifier.component_id, "1", identity, verify)
        for method in methods:
            service.components.register("candidate_method", method.method_id, "1", identity, lambda **_kwargs: None)

    return TaskAdapter(AdapterIdentity("owned-json-task", "1", identity), job, "single_response",
                       AdapterIdentity("owned-json-projection", "1", identity),
                       "json_extraction_v1", project, bind)


def owned_audit(*, spec: Mapping[str, Any]) -> dict[str, Any]:
    import unicodedata
    def normalize(value: str) -> str:
        return " ".join(unicodedata.normalize("NFC", value).split())
    train = {normalize(row["input"]) for row in spec["training_rows"]}
    panels = spec["public_sets"]
    clean = bool(train) and len(panels) == 1 and bool(panels[0]["rows"])
    clean = clean and not train.intersection(normalize(row["input"]) for row in panels[0]["rows"])
    return {"public_passed": bool(clean), "policy": "owned-normalized-input-disjoint-v1",
            "train_rows_digest": spec["training_rows_digest"],
            "validation_rows_digest": panels[0]["rows_digest"], "holdout_accessed": False}


def no_augmentation(_configuration: Mapping[str, Any]) -> list[dict[str, Any]]:
    raise GymJobError("owned tasks do not authorize augmentation")


def _probes(model: Any, tokenizer: Any, policy: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    from ..model_digest import model_state_digest
    from ..prompt_rendering import PromptRenderer
    renderer = PromptRenderer(tokenizer)
    records = []
    with torch.inference_mode():
        for index in range(16):
            rendered = renderer.render({"format": "chat_messages_v1", "messages": [
                {"role": "user", "content": f"Identity check {index}: return a JSON object."}]})
            ids = torch.tensor([rendered.token_ids], device="cuda:0", dtype=torch.long)
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0, -1].float()
            probabilities = torch.log_softmax(logits, dim=-1)
            token = int(torch.argmax(probabilities).item())
            records.append({"prompt_token_ids": list(rendered.token_ids), "greedy_token_id": token,
                            "greedy_logprob": float(probabilities[token].item())})
    return {"kind": "nf_local_transformers_identity_v1", "model_digest": model_state_digest(model),
            "precision": "bfloat16", "renderer_digest": renderer.digest,
            "context_tokens": policy["context_tokens"], "records": records}


class LocalExecution:
    """Verify exported bytes, reloaded weights and same-runtime fixed prompt probes."""

    def __init__(self, *, spec: Mapping[str, Any], candidate: Mapping[str, Any], root: Path,
                 record_probes: bool = False) -> None:
        self.spec, self.candidate, self.root = spec, candidate, root
        self.record_probes = record_probes
        self.adapter: Any = None
        self.model: Any = None
        self.tokenizer: Any = None

    def __enter__(self) -> Any:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from ..gym_adapters import _model_adapter
        from ..gym_artifacts import CandidateRef
        from ..gym_service import CandidateIdentity
        from ..proof_pipeline import read_document, write_document
        from ..proof_plugins import trained_reference
        from ..provenance import resolve_checkpoint
        role = self.candidate["role"]
        export = read_document(self.root / f"export-{role}.json")
        source, result = trained_reference(self.root, role)
        if (export["run_spec_digest"] != _document_digest(self.spec)
                or export["source_result_digest"] != result["result_digest"]
                or export["source_reference"] != asdict(source)):
            raise GymJobError("export provenance differs from the exact trained candidate")
        reference = CandidateRef(**export["reference"])
        checkpoint = resolve_checkpoint(reference.artifact_path, allow_download=False)
        checkpoint = replace(checkpoint, model_id=self.spec["base_model"]["model_id"],
                             revision=self.spec["base_model"]["revision"])
        self.files = dict(checkpoint.file_digests)
        tokenizer_loader: Callable[..., Any] = AutoTokenizer.from_pretrained
        model_loader: Callable[..., Any] = AutoModelForCausalLM.from_pretrained
        self.tokenizer = tokenizer_loader(checkpoint.path, local_files_only=True,
                                         trust_remote_code=False)
        self.model = model_loader(
            checkpoint.path, local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0").eval()
        observed = _probes(self.model, self.tokenizer, self.spec["policy"])
        if observed["model_digest"] != reference.model_digest:
            raise GymJobError("exported model weights differ from their candidate identity")
        probe_path = self.root / f"export-probes-{role}.json"
        if self.record_probes:
            write_document(probe_path, {"probe": observed, "files": self.files,
                                       "run_spec_digest": _document_digest(self.spec)})
        expected = read_document(probe_path)
        from ..backends.weight_edit.serving_probes import compare_records
        if (expected["files"] != self.files
                or expected["run_spec_digest"] != _document_digest(self.spec)
                or expected["probe"]["model_digest"] != observed["model_digest"]
                or compare_records(expected["probe"], observed) > 1e-6):
            raise GymJobError("local export reload identity probes differ")
        identity = CandidateIdentity(reference.candidate_id, "lora", reference.base_digest,
                                     reference.model_digest, reference.artifact_digest)
        self.adapter = _model_adapter(self.model, self.tokenizer, checkpoint, identity, {
            "device": "cuda:0", "precision": "bfloat16", "seed": self.candidate["recipe"]["seed"],
            "context_tokens": self.spec["policy"]["context_tokens"],
            "max_new_tokens": self.spec["policy"]["max_new_tokens"]},
            artifact_path=reference.artifact_path)
        self.evidence = {"verified": True, "fixed_prompts": 16, "logprob_atol": 1e-6,
                         "expected_digest": _document_digest(expected["probe"]),
                         "observed_digest": _document_digest(observed),
                         "model_digest": reference.model_digest, "files": self.files,
                         "run_spec_digest": _document_digest(self.spec)}
        return self

    def loaded_identity(self) -> dict[str, Any]:
        return {**self.adapter.loaded_identity(), "local_transformers_identity": self.evidence}

    def infer(self, interaction: Mapping[str, Any]) -> Any:
        return self.adapter.infer(interaction)

    def verify_end_identity(self) -> None:
        from ..model_digest import model_state_digest
        from ..provenance import resolve_checkpoint
        if (resolve_checkpoint(self.adapter.reference.artifact_path).file_digests != self.files
                or model_state_digest(self.model) != self.evidence["model_digest"]):
            raise GymJobError("candidate changed during validation")

    def __exit__(self, *_args: object) -> None:
        import torch
        self.adapter, self.model, self.tokenizer = None, None, None
        gc.collect()
        torch.cuda.empty_cache()


def local_evidence(*, model: Any, policy: Mapping[str, Any],
                   interaction: Mapping[str, Any] | None = None, result: Any = None) -> None:
    if type(model) is not LocalExecution:
        raise GymJobError("local evidence requires the installed candidate-bound adapter")
    evidence = model.loaded_identity()["local_transformers_identity"]
    if (evidence["verified"] is not True or evidence["fixed_prompts"] != policy["identity"]["probe_count"]
            or evidence["logprob_atol"] != policy["identity"]["logprob_atol"]):
        raise GymJobError("local identity policy differs")
    if interaction is not None:
        identity = result.interaction.get("loaded_identity", {})
        if identity.get("state_digest") != evidence["model_digest"]:
            raise GymJobError("validation response came from a different candidate")


def local_export(*, spec: Mapping[str, Any], role: str, root: Path) -> dict[str, Any]:
    from ..proof_plugins import serving_export
    result = serving_export(spec=spec, role=role, root=root)
    candidate = next(c for c in spec["candidates"] if c["role"] == role)
    with LocalExecution(spec=spec, candidate=candidate, root=root, record_probes=True) as model:
        local_evidence(model=model, policy=spec["policy"])
    return result
