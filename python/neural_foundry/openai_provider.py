"""Gym-neutral structured chat provider with run-wide durable authorization."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Protocol, Self

from .gym_job import GymJobError
from .gym_lifecycle import TaskAdapter
from .gym_service import ExecutionResponse, _document_digest
from .openai_transport import completion, validate_endpoint
from .proof_policy import require_bound_policy
from .public_budget import PublicBudget, unsigned_spec


class PublicProvider(Protocol):
    def __enter__(self) -> Self: ...
    def __exit__(self, *_args: object) -> None: ...
    def infer(self, interaction: Mapping[str, Any]) -> ExecutionResponse: ...
    def loaded_identity(self) -> Mapping[str, Any]: ...
    def verify_response_receipt(self, interaction: Mapping[str, Any], result: Any) -> None: ...


class OpenAIProvider:
    def __init__(self, *, spec: Mapping[str, Any], candidate: Mapping[str, Any],
                 task: TaskAdapter, api_key: str) -> None:
        self._spec = json.loads(json.dumps(unsigned_spec(spec), allow_nan=False))
        self.policy = require_bound_policy(self._spec)
        if self.policy["api"]["calls_authorized"] is not True:
            raise GymJobError("provider calls are not authorized by the frozen run policy")
        declared = next((row for row in spec["candidates"] if row["role"] == candidate["role"]), None)
        if declared != candidate:
            raise GymJobError("provider candidate differs from the frozen run")
        configuration = candidate["execution"]["configuration"]
        if set(configuration) != {"endpoint", "model"}:
            raise GymJobError("provider requires explicit endpoint/model data, not embedded credentials")
        self._endpoint, self._model = configuration["endpoint"], configuration["model"]
        self._host = validate_endpoint(self._endpoint)
        if self._model not in self.policy["api"]["model_rates"]:
            raise GymJobError("provider model has no frozen price authorization")
        self.budget = PublicBudget(self._spec)
        if not isinstance(api_key, str) or not api_key:
            raise GymJobError("provider requires a runtime API credential")
        self._api_key = api_key
        self.role = candidate["role"]
        self.context = {
            "gym": task.identity.document(), "job_digest": task.job.digest,
            "projection": task.projection.document(), "scorer": task.job.scorer.to_document(),
            "interaction_format": task.interaction_format,
            "output_schema": task.job.to_document()["task"]["output_schema"],
        }
        self.configuration = {
            "kind": "nf_gym_openai_configuration_v2", "model": self._model,
            "endpoint": self._endpoint, "policy_digest": _document_digest(self.policy),
            "run_spec_digest": _document_digest(self._spec), "gym_context": self.context,
        }
        self.configuration["configuration_digest"] = _document_digest(self.configuration)
        self._response_receipts: list[dict[str, Any]] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        pass

    def require_configuration(self) -> None:
        if require_bound_policy(self._spec) != self.policy:
            raise GymJobError("provider policy changed")
        if self.configuration["configuration_digest"] != _document_digest(
                {key: value for key, value in self.configuration.items() if key != "configuration_digest"}):
            raise GymJobError("provider Gym/scorer configuration changed")
        self.budget.snapshot()

    def infer(self, interaction: Mapping[str, Any]) -> ExecutionResponse:
        self.require_configuration()
        interaction = json.loads(json.dumps(interaction, allow_nan=False))
        if (interaction.get("format") != self.context["interaction_format"]
            or interaction.get("output_schema") != self.context["output_schema"]
            or not isinstance(interaction.get("messages"), list) or not interaction["messages"]
            or any(not isinstance(message, Mapping) or set(message) != {"role", "content"}
                   or message.get("role") not in {"system", "developer", "user", "assistant"}
                   or not isinstance(message.get("content"), str)
                   for message in interaction["messages"])):
            raise GymJobError("provider interaction differs from selected Gym format/schema")
        payload = {"model": self._model, "messages": interaction["messages"],
                   "temperature": self.policy["temperature"],
                   "reasoning_effort": self.policy["reasoning_effort"],
                   "max_completion_tokens": self.policy["max_new_tokens"],
                   "response_format": {"type": "json_schema", "json_schema": {
                       "name": "nf_output", "strict": True, "schema": self.context["output_schema"]}}}
        reservation = self.budget.reserve(role=self.role, payload=payload)
        result = completion(endpoint=self._endpoint, api_key=self._api_key, payload=payload,
                            interaction=interaction, receipt_fields={
                                "configuration_digest": self.configuration["configuration_digest"],
                                "scorer_digest": self.context["scorer"]["implementation_digest"],
                                "gym_context_digest": _document_digest(self.context),
                                "budget_binding_digest": reservation["binding_digest"],
                                "reservation_digest": reservation["reservation_digest"],
                                "api_attempts": reservation["attempt"],
                                "reserved_cost_microusd": reservation["run_reserved_cost_microusd"]})
        self._response_receipts.append(dict(result.interaction["api_response_receipt"]))
        return result

    def verify_response_receipt(self, interaction: Mapping[str, Any], result: Any) -> None:
        self.require_configuration()
        receipt = result.interaction.get("api_response_receipt") if isinstance(result, ExecutionResponse) else None
        if (not isinstance(receipt, Mapping) or receipt not in self._response_receipts
            or receipt.get("receipt_digest") != _document_digest(
                {key: value for key, value in receipt.items() if key != "receipt_digest"})
            or receipt.get("configuration_digest") != self.configuration["configuration_digest"]
            or receipt.get("scorer_digest") != self.context["scorer"]["implementation_digest"]
            or receipt.get("interaction_digest") != _document_digest(interaction)
            or receipt.get("response_digest") != _document_digest(result.response)):
            raise GymJobError("provider response lacks exact Gym-bound receipt evidence")

    def loaded_identity(self) -> Mapping[str, Any]:
        self.require_configuration()
        return {"provider": "openai", "pinned_model": self._model, "provider_host": self._host,
                "evidence_class": self._spec.get("evidence_class", "provider_response"),
                "configuration": self.configuration, "budget": self.budget.snapshot(),
                "api_response_receipts": list(self._response_receipts),
                "served_model_identity_verified": False}
