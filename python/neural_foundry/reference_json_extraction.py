"""Deterministic synthetic invoice extraction reference task."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping

from .gym_job import AcceptanceGate, CandidateMethod, ComponentReference, GymJobError, GymJobSpec, ScenarioSource, validate_task_instance
from .gym_lifecycle import AdapterIdentity, TaskAdapter
from .gym_service import GymStudioService, _document_digest


INPUT_SCHEMA = {
    "type": "object", "properties": {"text": {"type": "string", "maxLength": 512}},
    "required": ["text"], "additionalProperties": False,
}
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "customer": {"type": "string", "maxLength": 64},
        "product": {"type": "string", "maxLength": 64},
        "quantity": {"type": "integer", "minimum": 1, "maximum": 99},
    },
    "required": ["customer", "product", "quantity"], "additionalProperties": False,
}
TEMPLATES = {
    "train": {
        "order": "Order for {customer}: {quantity} units of {product}.",
        "fields": "Customer: {customer}; product: {product}; quantity: {quantity}.",
    },
    "validation": {
        "shipment": "Ship {quantity} {product} to customer {customer}.",
        "purchase": "{customer} purchased {quantity} units of {product}.",
    },
    "holdout": {
        "ledger": "Invoice line | buyer={customer} | item={product} | units={quantity}",
        "request": "For {customer}, fulfill the request for {product} ({quantity} units).",
    },
}
CUSTOMERS = ("Aster", "Birch", "Cedar", "Dahlia", "Elm", "Fennel", "Grove", "Hazel")
PRODUCTS = ("cables", "folders", "keyboards", "lamps", "markers", "notebooks", "panels", "sensors")


def _implementation_digest() -> str:
    return "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def generate_scenarios(*, split: str, seed: int, parameters: Mapping[str, Any], **_: Any) -> list[dict[str, Any]]:
    if split not in TEMPLATES or type(seed) is not int or seed < 0:
        raise GymJobError("invalid synthetic extraction split or seed")
    count = parameters.get("rows_per_template")
    if set(parameters) != {"rows_per_template"} or type(count) is not int or not 1 <= count <= 64:
        raise GymJobError("rows_per_template must be an integer from 1 to 64")
    randomizer = random.Random(f"invoice-extraction-v1:{split}:{seed}")
    rows = []
    for template_id, template in TEMPLATES[split].items():
        for index in range(count):
            reference = {"customer": randomizer.choice(CUSTOMERS), "product": randomizer.choice(PRODUCTS),
                         "quantity": randomizer.randint(1, 99)}
            text = template.format(**reference)
            rows.append({"scenario_id": f"{split}:{template_id}:{seed}:{index}", "family_id": template_id,
                         "item_id": f"{split}:{template_id}:{seed}:{index}", "input": {"text": text},
                         "reference": reference})
    return rows


def verify_response(*, reference: Any, response: Any, output_valid: bool = True, **_: Any) -> dict[str, Any]:
    try:
        validate_task_instance(OUTPUT_SCHEMA, response, field="response")
        validate_task_instance(OUTPUT_SCHEMA, reference, field="reference")
    except GymJobError:
        return {"passed": False}
    return {"passed": bool(output_valid and response == reference)}


def score_response(*, reference: Any, response: Any, output_valid: bool = True, **_: Any) -> dict[str, float]:
    return {"exact_match": float(verify_response(reference=reference, response=response, output_valid=output_valid)["passed"])}


def project(input_value: Any, response: Any) -> dict[str, Any]:
    validate_task_instance(INPUT_SCHEMA, input_value, field="input")
    messages = [{"role": "user", "content": "Extract customer, product, and quantity as JSON. " + input_value["text"]}]
    if response is not None:
        validate_task_instance(OUTPUT_SCHEMA, response, field="response")
        messages.append({"role": "assistant", "content": json.dumps(response, sort_keys=True, separators=(",", ":"))})
    return {"format": "json_extraction_v1", "messages": messages, "output_schema": OUTPUT_SCHEMA}


def reference_task(*, rows_per_template: int = 8) -> TaskAdapter:
    identity = _implementation_digest()
    source_digest = _document_digest({"templates": TEMPLATES, "customers": CUSTOMERS, "products": PRODUCTS})
    source = ScenarioSource("synthetic-invoices", "reference_invoice_generator", "1", identity, source_digest,
                            "simulated", ("train", "validation", "holdout"), {"rows_per_template": rows_per_template})
    scorer = ComponentReference("reference_invoice_exact_match", "1", identity, {})
    verifier = ComponentReference("reference_invoice_verifier", "1", identity, {})
    methods = tuple(CandidateMethod(name, "1", identity, {}) for name in ("unchanged_base", "sft"))
    methods += (CandidateMethod("weight_edit", "1", identity, {"target_surrogate": "teacher_forced_token_loss",
                                                               "gradient_inspection_used": True}),)
    job = GymJobSpec(
        job_id="synthetic-invoice-extraction-v1", title="Synthetic invoice extraction",
        description="Extract customer, product, and quantity from synthetic invoice text.",
        task_input_schema=INPUT_SCHEMA, task_output_schema=OUTPUT_SCHEMA, scenario_sources=(source,),
        scorer=scorer, verifier=verifier, candidate_methods=methods,
        gates=(AcceptanceGate("task", "task", "exact_match", "validation", "ge", 1.0),
               AcceptanceGate("behavior", "behavior", "exact_match", "holdout", "ge", 1.0),
               AcceptanceGate("controls", "controls", "beats_control", "controls", "eq", 1.0),
               AcceptanceGate("repeat", "reproducibility", "repeat", "reproducibility", "eq", 1.0)),
        metadata={"evidence_class": "synthetic"},
    )

    def bind(service: GymStudioService) -> None:
        current = _implementation_digest()
        if current != identity:
            raise GymJobError("reference task implementation changed after job creation")
        service.components.register("scenario_provider", source.provider_id, source.provider_version, identity, generate_scenarios)
        service.components.register("scorer", scorer.component_id, scorer.version, identity, score_response)
        service.components.register("verifier", verifier.component_id, verifier.version, identity, verify_response)
        for method in methods:
            service.components.register("candidate_method", method.method_id, method.version, identity, lambda **kwargs: None)

    return TaskAdapter(AdapterIdentity("reference_invoice_extraction", "1", identity), job, "single_response",
                       AdapterIdentity("reference_invoice_projection", "1", identity), "json_extraction_v1", project, bind)