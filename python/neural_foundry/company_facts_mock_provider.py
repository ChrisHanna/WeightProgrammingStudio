"""Offline second-Gym conformance composition; HTTP and credentials are fixture-only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Self
from unittest.mock import patch

from .company_facts_public_plugin import compose
from .gym_job import GymJobError
from .gym_lifecycle import AdapterIdentity
from .gym_service import _document_digest
from .proof_pipeline import read_document, score, write_document
from .proof_plugins import openai_provider
from .proof_policy import load_policy
from .proof_registry import builtin_registry
from .provenance import sha256_file
from .public_budget import PublicBudget, initialize_public_budget


class FixtureResponse:
    def __init__(self, document: dict[str, Any]) -> None:
        self.document = document

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        pass

    def read(self, _limit: int) -> bytes:
        return json.dumps(self.document).encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--generic-contract", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    before = read_document(args.generic_contract)
    if any(sha256_file(Path(path)) != digest for path, digest in before["files"].items()):
        raise GymJobError("generic core changed after revised contract freeze")
    configuration = json.loads(args.config.read_bytes())
    spec, registry = compose(configuration)
    _, plugins = builtin_registry()
    spec["run_id"] = "company-facts-mock-provider-conformance-v2"
    spec["policy_reference"]["calls_authorized"] = True
    spec["policy"] = load_policy(spec["policy_reference"])
    spec["evidence_class"] = "software_fixture"
    spec["public_budget"] = {"version": 1, "ledger_id": "company-facts-mocked-transport-v2",
                             "path": str(root / "mock-provider-budget.sqlite3")}
    dependencies = {**plugins["provider"]["dependencies"],
                    "python/neural_foundry/company_facts_mock_provider.py": sha256_file(__file__)}

    def fixture_provider(**kwargs: Any) -> Any:
        return openai_provider(**kwargs, api_key="test-only-fixture-key")

    provider_reference = registry.register("provider", AdapterIdentity(
        "openai-software-conformance", "2", _document_digest(dependencies)),
        fixture_provider, dependencies=dependencies)
    candidate = spec["candidates"][0]
    candidate["execution"] = {"kind": "provider", "plugin": provider_reference,
                              "configuration": {"model": "gpt-6-luna",
                                                "endpoint": "https://provider.invalid/v1/chat/completions"}}
    candidate["evidence"] = plugins["api_evidence"]
    frozen = write_document(root / "run-spec.json", spec)
    initialize_public_budget(spec)
    calls = []

    class FixtureOpener:
        def open(self, request: Any, timeout: float) -> FixtureResponse:
            if request.full_url != candidate["execution"]["configuration"]["endpoint"] or timeout != 60:
                raise GymJobError("unexpected fixture dispatch")
            if request.get_header("Authorization") != "Bearer test-only-fixture-key":
                raise GymJobError("fixture must not use a real runtime credential")
            payload = json.loads(request.data)
            query = json.loads(payload["messages"][0]["content"])
            record = configuration["catalog"].get(query["company"], {})
            response = ({"value": record[query["attribute"]], "source": record["source"]}
                        if query["attribute"] in record else {"value": "not listed", "source": "none"})
            calls.append(payload)
            return FixtureResponse({
                "id": f"mock-completion-{len(calls)}", "created": 1700000000,
                "model": payload["model"], "system_fingerprint": "software-fixture-only",
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(response)}}]})

    with patch("urllib.request.build_opener", return_value=FixtureOpener()):
        result = score(spec, role=candidate["role"], root=root, registry=registry)
    after = {path: sha256_file(Path(path)) for path in before["files"]}
    budget = PublicBudget(spec).snapshot()
    receipt = write_document(root / "mock-provider-conformance.json", {
        "kind": "nf_gym_provider_mock_conformance_v2", "scope": "software_fixture_only",
        "generic_contract_digest": before["document_digest"], "generic_core_unchanged": after == before["files"],
        "run_spec_digest": frozen["document_digest"], "score_digest": result["document_digest"],
        "conformance_passed": result["gate_passed"] and after == before["files"]
                              and len(calls) == budget["attempts"],
        "simulated_dispatches": len(calls), "durable_budget": budget,
        "network_mocked": True, "credentials_are_test_only": True, "real_network_calls": 0,
        "model_forward_performed": False, "training_dispatched": False, "downloads": False,
        "holdout_accessed": False, "model_quality_accepted": False, "release_authorized": False})
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
