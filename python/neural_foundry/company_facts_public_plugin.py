"""Second-Gym software fixture: plugins/config composed over the unchanged public runner."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

from .gym_job import (
    AcceptanceGate,
    CandidateMethod,
    ComponentReference,
    GymJobError,
    GymJobSpec,
    ScenarioSource,
    validate_task_instance,
)
from .gym_lifecycle import AdapterIdentity, TaskAdapter
from .gym_service import ExecutionResponse, _document_digest
from .proof_pipeline import audit, read_document, score, validate_spec, write_document
from .proof_policy import load_policy
from .proof_registry import ProofRegistry, builtin_registry
from .provenance import sha256_file

INPUT = {"type": "object", "properties": {
    "company": {"type": "string", "maxLength": 80},
    "attribute": {"type": "string", "maxLength": 40},
    "note": {"type": "string", "maxLength": 160}},
    "required": ["company", "attribute", "note"], "additionalProperties": False}
OUTPUT = {"type": "object", "properties": {
    "value": {"type": "string", "maxLength": 100},
    "source": {"type": "string", "maxLength": 100}},
    "required": ["value", "source"], "additionalProperties": False}


def project(input_value: Any, response: Any) -> dict[str, Any]:
    validate_task_instance(INPUT, input_value, field="company-facts input")
    messages = [{"role": "user", "content": json.dumps(input_value, sort_keys=True)}]
    if response is not None:
        validate_task_instance(OUTPUT, response, field="company-facts output")
        messages.append({"role": "assistant", "content": json.dumps(response, sort_keys=True)})
    return {"format": "chat_messages_v1", "messages": messages, "output_schema": OUTPUT}


def supported_fact(*, reference: Any, response: Any, output_valid: bool = True, **_: Any) -> dict[str, float]:
    try:
        validate_task_instance(OUTPUT, response, field="company-facts response")
    except GymJobError:
        return {"supported_fact": 0.0}
    return {"supported_fact": float(output_valid and reference == response)}


def company_task(configuration: Mapping[str, Any]) -> TaskAdapter:
    digest = sha256_file(__file__)
    source = ScenarioSource("public-company-questions", "company-public-fixtures", "1", digest,
                            _document_digest(configuration), "simulated",
                            ("train", "validation", "holdout"), {})
    component = ComponentReference("company-supported-fact", "1", digest, {})
    method = CandidateMethod(configuration["method"], "1", digest, {})
    job = GymJobSpec(
        job_id="company-facts-public-fixture-v1", title="Company facts software fixture",
        description="Synthetic public catalog lookup; not a model-quality or holdout claim.",
        task_input_schema=INPUT, task_output_schema=OUTPUT, scenario_sources=(source,),
        scorer=component, verifier=component, candidate_methods=(method,),
        gates=(AcceptanceGate("public-fixture", "task", "supported_fact", "validation", "ge", 1.0),
               AcceptanceGate("private-behavior", "behavior", "supported_fact", "holdout", "ge", 1.0),
               AcceptanceGate("control", "controls", "beats_control", "controls", "eq", 1.0),
               AcceptanceGate("repeat", "reproducibility", "repeat", "reproducibility", "eq", 1.0)),
        metadata={"evidence_class": "software_fixture", "non_fixture_gates": "not_executed_not_passed"})

    def bind(service: Any) -> None:
        def scenarios(*, split: str, **_kwargs: Any) -> list[dict[str, Any]]:
            if split != "validation":
                raise GymJobError("company fixture provider is public-validation-only")
            return list(configuration["rows"])

        service.components.register("scenario_provider", source.provider_id, "1", digest, scenarios)
        service.components.register("scorer", component.component_id, "1", digest, supported_fact)
        service.components.register("verifier", component.component_id, "1", digest,
                                    lambda **kwargs: {"passed": supported_fact(**kwargs)["supported_fact"] == 1.0})
        service.components.register("candidate_method", method.method_id, "1", digest, lambda **_kwargs: None)

    return TaskAdapter(AdapterIdentity("company-facts", "1", digest), job, "single_response",
                       AdapterIdentity("company-facts-chat", "1", digest), "chat_messages_v1", project, bind)


class CatalogFixture:
    def __init__(self, catalog: Mapping[str, Any]) -> None:
        self.catalog = catalog
        self.receipts: list[dict[str, Any]] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        pass

    def infer(self, interaction: Mapping[str, Any]) -> ExecutionResponse:
        query = json.loads(interaction["messages"][0]["content"])
        validate_task_instance(INPUT, query, field="company-facts query")
        record = self.catalog.get(query["company"], {})
        response = ({"value": record[query["attribute"]], "source": record["source"]}
                    if query["attribute"] in record else {"value": "not listed", "source": "none"})
        receipt = {"interaction_digest": _document_digest(interaction),
                   "response_digest": _document_digest(response),
                   "catalog_digest": _document_digest(self.catalog)}
        receipt["receipt_digest"] = _document_digest(receipt)
        self.receipts.append(receipt)
        return ExecutionResponse(response, {"company_fact_fixture_receipt": receipt})

    def loaded_identity(self) -> dict[str, Any]:
        return {"provider": "company-catalog-fixture", "fixture": True,
                "evidence_class": "software_fixture", "catalog_digest": _document_digest(self.catalog),
                "model_quality_accepted": False, "served_model_identity_verified": False,
                "new_api_calls": 0}


def provider(*, spec: Mapping[str, Any], **_: Any) -> CatalogFixture:
    return CatalogFixture(spec["gym"]["catalog"])


def evidence(*, model: Any, interaction: Mapping[str, Any] | None = None,
             result: Any = None, **_: Any) -> None:
    if type(model) is not CatalogFixture:
        raise GymJobError("company fixture evidence requires its explicit software provider")
    if interaction is not None:
        receipt = result.interaction.get("company_fact_fixture_receipt")
        if (receipt not in model.receipts
            or receipt["interaction_digest"] != _document_digest(interaction)
            or receipt["response_digest"] != _document_digest(result.response)):
            raise GymJobError("company fixture response receipt differs")


def public_audit(*, spec: Mapping[str, Any]) -> dict[str, Any]:
    return {"kind": "nf_company_facts_fixture_public_audit_v1",
            "public_passed": not spec["training_rows"], "audit_passed": False,
            "evidence_class": "software_fixture", "holdout_accessed": False,
            "independent_custody_verified": False, "release_authorized": False}


def compose(configuration: Mapping[str, Any]) -> tuple[dict[str, Any], ProofRegistry]:
    if (configuration.get("kind") != "nf_company_facts_public_fixture_config"
        or configuration.get("version") not in {1, 2}
        or configuration.get("evidence_class") != "software_fixture"):
        raise GymJobError("unsupported company fixture configuration")
    registry, references = builtin_registry()
    bound_reference = None
    if configuration["version"] == 2:
        bound_reference = dict(configuration["policy_reference"])
        policy_path = Path(bound_reference["path"])
        if not policy_path.is_absolute():
            policy_path = Path(__file__).resolve().parents[2] / policy_path
        bound_reference["path"] = str(policy_path)
        policy = load_policy(bound_reference)
    else:
        policy = configuration["policy"]
    custom = {}
    for kind, factory in (("gym", company_task), ("provider", provider),
                          ("evidence", evidence), ("public_audit", public_audit)):
        custom[kind] = registry.register(kind, AdapterIdentity(
            "company-facts-" + kind, "1", sha256_file(__file__)), factory)
    rows = []
    for index, question in enumerate(configuration["questions"]):
        record = configuration["catalog"].get(question["company"], {})
        reference = ({"value": record[question["attribute"]], "source": record["source"]}
                     if question["attribute"] in record else {"value": "not listed", "source": "none"})
        rows.append({"scenario_id": f"validation:company-facts:{index}", "input": question,
                     "reference": reference, "family_id": "public-catalog"})
    spec = {
        "kind": "nf_public_proof_run", "version": 1, "run_id": "company-facts-fixture-v1",
        "base_model": {"model_id": "software-fixture/company-catalog", "revision": "catalog-v1"},
        "plugins": {key: custom.get(key, references[key]) for key in
                    ("gym", "model", "trainer", "training_audit", "exporter", "augmenter", "public_audit")},
        "gym": {"catalog": configuration["catalog"], "rows": rows,
                "method": configuration["candidate"]["method"]},
        "training_rows": [], "training_rows_digest": _document_digest({"rows": []}),
        "candidates": [{**configuration["candidate"], "evidence": custom["evidence"],
                        "execution": {"kind": "provider", "plugin": custom["provider"]}}],
        "public_sets": [{"name": "company-public", "rows": rows,
                         "rows_digest": _document_digest({"rows": rows})}],
        "gates": [{"name": "software-contract", "set": "company-public",
                   "metric": "supported_fact", "minimum": len(rows)}],
        "policy": policy, "download_allowed": False,
        "holdout_accessed": False, "release_authorized": False,
        "configuration_digest": _document_digest(configuration),
    }
    if bound_reference is not None:
        spec["policy_reference"] = bound_reference
    validate_spec(spec, registry)
    return spec, registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--generic-contract", type=Path, required=True)
    args = parser.parse_args()
    before = read_document(args.generic_contract)
    if any(sha256_file(Path(path)) != digest for path, digest in before["files"].items()):
        raise GymJobError("generic core changed after its pre-plugin contract freeze")
    configuration = json.loads(args.config.read_bytes())
    spec, registry = compose(configuration)
    frozen = write_document(args.root / "run-spec.json", spec)
    public = audit(spec, registry)
    write_document(args.root / "public-audit.json", public)
    result = score(spec, role=configuration["candidate"]["role"], root=args.root, registry=registry)
    after = {path: sha256_file(Path(path)) for path in before["files"]}
    unchanged = after == before["files"]
    receipt = write_document(args.root / "modularity-acceptance.json", {
        "kind": "nf_second_gym_modularity_acceptance_v1",
        "generic_contract_digest": before["document_digest"], "generic_core_unchanged": unchanged,
        "generic_files_after": after, "run_spec_digest": frozen["document_digest"],
        "score_digest": result["document_digest"],
        "modularity_accepted": unchanged and result["gate_passed"], "scope": "software_fixture_only",
        "model_quality_accepted": False, "final_proof_eligible": False,
        "training_dispatched": False, "new_api_calls": 0, "downloads": False,
        "holdout_accessed": False, "release_authorized": False})
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
